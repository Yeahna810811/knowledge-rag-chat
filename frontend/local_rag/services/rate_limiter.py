"""基于 Redis 的固定窗口限流。

为什么是「INCR + EXPIRE」而不是「GET 然后 SET」：
后者在并发下是经典竞态——两个请求同时 GET 到 59，都判为未超限，
都 SET 60，窗口内实际放行了几百个请求。限流形同虚设。
所以计数与过期设置必须是一次原子操作。

两条实现路径：
1. Lua 脚本（首选）：一次 EVAL 完成 INCR / EXPIRE / TTL，天然原子。
2. 事务管道（回落）：MULTI/EXEC 包住 INCR + TTL，事后再补 EXPIRE。
   部分托管 Redis 与 fakeredis 不支持 EVAL，这时走这条路。

身份标识为什么用 session_id：
项目当前没有 User / JWT 体系，session_id 是现成的、前端已经在传的标识，
足够把不同使用者分开。等接入认证后，把 `_key()` 里的标识换成 user_id 即可。

故障策略：
**fail-open**。Redis 挂掉时限流直接放行并记录 warning。
理由是限流的目的是保护系统不被打爆，而不是拒绝服务——
Redis 已经挂了，再让所有请求 429，等于把 Redis 故障放大成全站不可用。
（要改成 fail-closed 的话，改 `check()` 里降级分支的返回值即可。）
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from frontend.local_rag.cache.redis_client import RedisClient

logger = logging.getLogger(__name__)

RATE_LIMIT_PREFIX = "rag:rate_limit"

# KEYS[1] = 窗口计数 key，ARGV[1] = 窗口秒数
# 返回 {当前计数, 剩余 TTL}
_FIXED_WINDOW_LUA = """
local count = redis.call("INCR", KEYS[1])
if count == 1 then
  redis.call("EXPIRE", KEYS[1], ARGV[1])
end
local ttl = redis.call("TTL", KEYS[1])
if ttl < 0 then
  redis.call("EXPIRE", KEYS[1], ARGV[1])
  ttl = tonumber(ARGV[1])
end
return {count, ttl}
"""


@dataclass
class RateLimitResult:
    """限流判定结果。

    degraded=True 表示「这次判定没有真正生效」（Redis 不可用或限流关闭），
    调用方据此放行。不要只依赖 allowed 判断——放行有两种原因：
    额度还没用完，或者压根没限成。
    """

    allowed: bool
    limit: int
    remaining: int
    retry_after: int
    degraded: bool = False

    def as_detail(self) -> str:
        return (
            f"请求过于频繁：每 {self.retry_after}s 内最多 {self.limit} 次，"
            f"请 {self.retry_after}s 后重试"
        )


class RateLimiter:
    """固定窗口限流：同一标识在 window_seconds 内最多 limit_requests 次请求。"""

    def __init__(
        self,
        redis_client: RedisClient,
        *,
        enabled: bool = True,
        limit_requests: int = 60,
        window_seconds: int = 60,
    ) -> None:
        self._redis = redis_client
        self.enabled = enabled
        self.limit = max(0, int(limit_requests))
        self.window_seconds = max(1, int(window_seconds))
        # Lua 不可用时不再重复尝试（否则每个请求都会打一条 warning）
        self._lua_disabled = False
        self._metrics = {
            "allowed": 0,
            "rate_limited": 0,
            "error": 0,
        }

    # ------------------------------------------------------------------ 键设计
    def build_key(self, identity: str) -> str:
        """rag:rate_limit:{identity}:{window_index}

        窗口序号按 wall clock 整除得到，窗口自然滚动时 key 自动切换，
        不需要任何定时清理任务。
        """
        window_index = int(time.time()) // self.window_seconds
        return f"{RATE_LIMIT_PREFIX}:{identity}:{window_index}"

    # -------------------------------------------------------------------- 判定
    def check(self, identity: str) -> RateLimitResult:
        if not self.enabled:
            return RateLimitResult(
                allowed=True,
                limit=self.limit,
                remaining=self.limit,
                retry_after=0,
                degraded=True,
            )

        outcome = self._incr_window(self.build_key(identity))
        if outcome is None:
            # Redis 不可用：fail-open（理由见文件头注释）
            self._metrics["error"] += 1
            logger.warning("限流计数失败（Redis 不可用），本次请求放行")
            return RateLimitResult(
                allowed=True,
                limit=self.limit,
                remaining=self.limit,
                retry_after=0,
                degraded=True,
            )

        count, ttl = outcome
        allowed = count <= self.limit
        if allowed:
            self._metrics["allowed"] += 1
        else:
            self._metrics["rate_limited"] += 1
            logger.warning(
                "触发限流: identity=%s count=%s limit=%s window=%ss",
                identity,
                count,
                self.limit,
                self.window_seconds,
            )

        remaining = max(0, self.limit - count)
        retry_after = ttl if (not allowed and ttl > 0) else 0
        return RateLimitResult(
            allowed=allowed,
            limit=self.limit,
            remaining=remaining,
            retry_after=retry_after,
        )

    # -------------------------------------------------------------- 原子计数
    def _incr_window(self, key: str) -> tuple[int, int] | None:
        """返回 (计数, 剩余 TTL)；None 表示 Redis 不可用。"""
        if not self._lua_disabled:
            raw = self._redis.eval_script(_FIXED_WINDOW_LUA, [key], [self.window_seconds])
            if raw is not None:
                parsed = self._parse_window(raw)
                if parsed is not None:
                    return parsed

            # 走到这里说明这次没拿到结果。关键是要分清两种原因：
            # - Redis 压根不支持 Lua   → 永久关闭，别再浪费一次网络往返
            # - 网络抖动 / 连接失败     → 只是一次，下次还要继续用 Lua
            # 混为一谈的话，一次几秒的抖动就会让限流永久退化到管道实现。
            if self._redis.scripting_supported is False:
                self._lua_disabled = True
                logger.warning("Redis 不支持 Lua，限流改用事务管道实现（原子性等价）")
            else:
                logger.warning("Lua 限流脚本本次不可用，本次改用事务管道（下次仍会重试 Lua）")

        return self._incr_window_pipeline(key)

    def _incr_window_pipeline(self, key: str) -> tuple[int, int] | None:
        try:
            pipe = self._redis.pipeline(transaction=True)
            pipe.incr(key)
            pipe.ttl(key)
            count_raw, ttl_raw = pipe.execute()
        except Exception as exc:  # noqa: BLE001
            logger.warning("限流事务管道执行失败: %s", exc)
            return None

        try:
            count = int(count_raw)
            ttl = int(ttl_raw)
        except (TypeError, ValueError):
            return None

        # count == 1：刚建的 key，必须补 TTL，否则这个窗口永不结束。
        # ttl < 0：key 存在但没有 TTL（EXPIRE 丢过一次），一并修复。
        if count == 1 or ttl < 0:
            self._redis.expire(key, self.window_seconds)
            ttl = self.window_seconds
        return count, ttl

    @staticmethod
    def _parse_window(raw: Any) -> tuple[int, int] | None:
        try:
            return int(raw[0]), int(raw[1])
        except (TypeError, ValueError, IndexError, KeyError):
            return None

    # -------------------------------------------------------------------- 观测
    def metrics(self) -> dict[str, int]:
        return dict(self._metrics)

"""统一 Redis 连接管理。

为什么要有这一层：
Redis 在本项目里是**加速器**，不是数据源——会话历史在 MySQL，知识库在
本地索引文件。所以缓存挂掉的代价是「慢」，不是「错」。这决定了这里的
容错基调：任何操作失败都返回降级值 + 一条 warning，绝不把异常抛给调用方。

这一层同时是唯一的连接出口。业务代码不允许出现 `redis.Redis(...)`：
超时、连接池、health check 这些参数散落在各处的话，线上排障时根本
说不清一个慢请求到底卡在哪个客户端上。

降级契约（调用方依赖这些返回值判断，不要改动语义）：
- ping()        -> True / False
- get()         -> str | None（失败与未命中都返回 None）
- set()         -> True / False
- delete()      -> True / False
- incr()        -> int | None
- expire()      -> True / False
- ttl()         -> int | None
- eval_script() -> 任意结果 | None（Redis 不支持 Lua 时返回 None）
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Sequence

import redis

logger = logging.getLogger(__name__)


class RedisClient:
    """Redis 连接门面：统一超时、连接池与异常降级。

    `client` 参数是为测试留的注入口——传进来的实例（比如 fakeredis）由
    调用方管理生命周期，close() 不会去关它。
    """

    def __init__(
        self,
        url: str = "redis://127.0.0.1:6379/0",
        *,
        client: Any | None = None,
        socket_connect_timeout: float = 1.0,
        socket_timeout: float = 1.0,
        max_connections: int = 20,
        health_check_interval: int = 30,
    ) -> None:
        self.url = url
        self._injected = client is not None
        # 三态：None=还不知道，True=支持，False=确认不支持。
        # 区分「不支持」和「这次连不上」很关键——后者只是临时抖动，
        # 不该让调用方永久放弃 Lua 路径。
        self._scripting_supported: bool | None = None
        if client is not None:
            self._client = client
        else:
            self._client = redis.Redis.from_url(
                url,
                decode_responses=True,
                socket_connect_timeout=socket_connect_timeout,
                socket_timeout=socket_timeout,
                health_check_interval=health_check_interval,
                max_connections=max_connections,
                # 连接断开后自动重连一次，避免单条命令失败就把整个请求打回降级路径
                retry_on_timeout=True,
            )

    # -------------------------------------------------------------- 底层实例
    @property
    def raw(self) -> Any:
        """暴露底层客户端，供需要 Lua / pipeline 的调用方使用。

        只有 RateLimiter 这类确实需要原子能力的组件才该碰它；
        普通读写一律走下面的安全方法。
        """
        return self._client

    def pipeline(self, transaction: bool = True) -> Any:
        """返回一个事务管道。调用方负责 execute() 时的异常捕获。"""
        return self._client.pipeline(transaction=transaction)

    # ------------------------------------------------------------------ 状态
    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis ping 失败: %s", exc)
            return False

    @property
    def healthy(self) -> bool:
        return self.ping()

    # ------------------------------------------------------------------ 读写
    def get(self, key: str) -> str | None:
        try:
            return self._client.get(key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis GET 失败，按未命中处理 (key=%s): %s", key, exc)
            return None

    def set(self, key: str, value: str, ex: int | None = None, nx: bool = False) -> bool:
        try:
            result = self._client.set(key, value, ex=ex, nx=nx)
            return bool(result)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis SET 失败 (key=%s): %s", key, exc)
            return False

    def delete(self, key: str) -> bool:
        try:
            return bool(self._client.delete(key))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis DELETE 失败 (key=%s): %s", key, exc)
            return False

    def incr(self, key: str) -> int | None:
        try:
            return int(self._client.incr(key))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis INCR 失败 (key=%s): %s", key, exc)
            return None

    def expire(self, key: str, seconds: int) -> bool:
        try:
            return bool(self._client.expire(key, seconds))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis EXPIRE 失败 (key=%s): %s", key, exc)
            return False

    def ttl(self, key: str) -> int | None:
        try:
            return int(self._client.ttl(key))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis TTL 失败 (key=%s): %s", key, exc)
            return None

    # -------------------------------------------------------------------- Lua
    @property
    def scripting_supported(self) -> bool | None:
        """Redis 是否支持 Lua 脚本。None = 还没试过。"""
        return self._scripting_supported

    def eval_script(
        self,
        script: str,
        keys: Sequence[str],
        args: Sequence[Any] = (),
    ) -> Any | None:
        """执行 Lua 脚本，返回 None 表示「这次没拿到结果」。

        返回 None 有两种截然不同的原因，用 `scripting_supported` 区分：
        - Redis 不支持脚本（部分托管 Redis、fakeredis）→ 永久，别再试了
        - 网络抖动 / 连接失败                → 临时，下次还能用 Lua

        为什么不抛异常：调用方需要的是「降级继续跑」，不是处理异常。
        """
        try:
            result = self._client.eval(script, len(keys), *keys, *args)
            self._scripting_supported = True
            return result
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            if "unknown command" in message or "noscript" in message:
                self._scripting_supported = False
                logger.warning("Redis 不支持 Lua 脚本，后续改用普通命令实现: %s", exc)
            else:
                # 临时失败：不改 _scripting_supported，下次仍然可以走 Lua
                logger.warning("Redis EVAL 本次失败（临时），回落到普通命令实现: %s", exc)
            return None

    # ------------------------------------------------------------------ 生命周期
    def close(self) -> None:
        """释放连接池。注入的客户端不由这里关闭。"""
        if self._injected:
            return
        try:
            self._client.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis 关闭连接失败: %s", exc)

    def __enter__(self) -> "RedisClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

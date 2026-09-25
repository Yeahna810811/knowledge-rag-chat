"""Retrieval 结果缓存（Redis）。

缓存什么 / 不缓存什么：
本项目支持多会话上下文，同一个问题在不同 session 里含义可能完全不同
（"它为什么这样设计？" 里的"它"指代什么，取决于前文）。所以**不能**用
`query -> final answer` 做全局缓存——那样会把 A 会话的答案错给 B 会话。

这一层只缓存「检索结果」，即 query -> documents/chunks。LLM 生成照常执行，
既省掉了重复的 BM25 / 向量检索开销，又不会污染带上下文的最终答案。

失效策略（knowledge-base version）：
不用 `KEYS rag:*` 扫全库，也不做 FLUSHDB——那在生产上是自杀式操作。
改为把知识库版本号拼进 cache key：上传 / 清空知识库后把版本号 +1，
新的检索自然落到新版本的 key 上，旧 key 不再命中，等 TTL 自己过期。

Redis 挂掉会怎样：
缓存 miss 路径照常走原检索器，只是不写 Redis，并记一条 warning。
也就是说 Redis 挂掉只会让系统变慢，不会让系统变错。
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Callable, Sequence

from frontend.local_rag.cache.redis_client import RedisClient

logger = logging.getLogger(__name__)

CACHE_PREFIX = "rag:retrieval"
VERSION_KEY = "rag:kb_version"


def normalize_query(query: str) -> str:
    """查询归一化：只做 strip + 合并空白。

    刻意不做小写化 / 去标点 / 繁简转换——那些会改变语义，
    让「RAG 是什么」和「rag 是什么」撞缓存还无所谓，
    但把专业术语改坏导致串数据是缓存里最难排查的一类 bug。
    归一化只负责消除「用户多打了个空格」这种无意义差异。
    """
    return " ".join(str(query or "").split())


def query_hash(query: str) -> str:
    """SHA-256 摘要。完整问题不进 Redis key：过长、含任意字符、还可能带隐私信息。"""
    return hashlib.sha256(normalize_query(query).encode("utf-8")).hexdigest()


class RetrievalCache:
    """检索结果缓存：命中就返回，未命中就跑一次真实检索再写回。"""

    def __init__(
        self,
        redis_client: RedisClient,
        *,
        enabled: bool = True,
        ttl_seconds: int = 600,
        retrieval_mode: str = "bm25",
        rewrite_mode: str = "off",
        top_k: int = 4,
    ) -> None:
        self._redis = redis_client
        self.enabled = enabled
        self.ttl_seconds = ttl_seconds
        self.retrieval_mode = str(retrieval_mode).lower()
        self.rewrite_mode = str(rewrite_mode).lower()
        self.top_k = int(top_k)
        # 轻量计数器：本阶段不引入 Prometheus，先攒在这里，
        # 下一阶段做 Logging/Metrics 时直接读这些数即可。
        self._metrics = {
            "cache_hit": 0,
            "cache_miss": 0,
            "cache_error": 0,
            "cache_write": 0,
            "cache_bypass": 0,
        }

    # ------------------------------------------------------------------ 键设计
    def build_key(self, query: str, top_k: int | None = None) -> str:
        """rag:retrieval:{kb_version}:{mode}:{rewrite}:{top_k}:{query_hash}

        五个维度缺一不可：少任何一个都会串数据——
        - kb_version   ：知识库更新后旧缓存必须失效
        - retrieval_mode：bm25 / dense / hybrid 召回的文档不同
        - rewrite_mode ：查询改写开与不开，召回结果不同
        - top_k        ：k=4 的 4 条结果不能给 k=8 的请求用
        - query_hash   ：查询本身
        """
        k = self.top_k if top_k is None else int(top_k)
        version = self.current_version()
        return (
            f"{CACHE_PREFIX}:v{version}:{self.retrieval_mode}:"
            f"{self.rewrite_mode}:{k}:{query_hash(query)}"
        )

    # ------------------------------------------------------- knowledge version
    def current_version(self) -> int:
        """当前知识库版本号。Redis 不可用时返回 0（此时缓存读写都会失败，无所谓版本号）。"""
        raw = self._redis.get(VERSION_KEY)
        if raw is None:
            # Redis 全新 / 数据丢失：从 1 重新起。旧 key 本就随 Redis 一起没了，
            # 版本号从头开始不会造成串数据。用 NX 避免并发下互相覆盖。
            self._redis.set(VERSION_KEY, "1", nx=True)
            raw = self._redis.get(VERSION_KEY)
        try:
            return int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0

    def bump_knowledge_version(self) -> int | None:
        """知识库内容变更后调用：版本号 +1，旧缓存自然失效。

        只在「知识库真的变了」之后调用（upload / reset 成功之后）。
        失败的操作不能 bump，否则白白丢弃一整批缓存。
        Redis 不可用返回 None——缓存本来就写不进去，不 bump 也没损失。
        """
        version = self._redis.incr(VERSION_KEY)
        if version is None:
            logger.warning("Redis 不可用，知识库版本号未递增（缓存写入本就失败，不影响正确性）")
            return None
        logger.info("知识库版本更新到 v%s，旧检索缓存已失效", version)
        return int(version)

    # -------------------------------------------------------------------- 读写
    def get(self, key: str) -> list[dict] | None:
        """读缓存。未命中 / 数据损坏 / Redis 不可用都返回 None。"""
        raw = self._redis.get(key)
        if raw is None:
            return None
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError) as exc:
            # 缓存里躺着一条反序列化不了的数据，说明序列化格式改过或被人手改过。
            # 删掉它，避免这条脏数据永远拦在前面。
            self._metrics["cache_error"] += 1
            logger.warning("检索缓存反序列化失败，删除该 key: %s", exc)
            self._redis.delete(key)
            return None
        if not isinstance(payload, list) or not all(isinstance(i, dict) for i in payload):
            self._metrics["cache_error"] += 1
            logger.warning("检索缓存结构异常（不是 list[dict]），删除该 key")
            self._redis.delete(key)
            return None
        return payload

    def put(self, key: str, results: Sequence[dict]) -> None:
        """写缓存。序列化失败或 Redis 不可用都静默跳过。"""
        try:
            payload = json.dumps(list(results), ensure_ascii=False, default=str)
        except (TypeError, ValueError) as exc:
            self._metrics["cache_error"] += 1
            logger.warning("检索结果无法序列化，跳过缓存: %s", exc)
            return
        if self._redis.set(key, payload, ex=self.ttl_seconds):
            self._metrics["cache_write"] += 1

    def get_or_search(
        self,
        query: str,
        top_k: int,
        do_search: Callable[[], list[dict]],
    ) -> list[dict]:
        """命中就返回缓存，否则跑真实检索并写回。

        Redis 不可用时的行为：get 返回 None（当 miss），检索照常执行，
        put 写不进去也无所谓——整条链路只是没有加速，结果完全正确。
        """
        if not self.enabled:
            self._metrics["cache_bypass"] += 1
            return do_search()

        key = self.build_key(query, top_k)
        cached = self.get(key)
        if cached is not None:
            self._metrics["cache_hit"] += 1
            logger.debug("检索缓存命中 key=%s", key)
            return cached

        self._metrics["cache_miss"] += 1
        logger.debug("检索缓存未命中 key=%s", key)
        results = do_search()
        if results:
            self.put(key, results)
        return results

    # -------------------------------------------------------------------- 观测
    def metrics(self) -> dict[str, int]:
        return dict(self._metrics)


class CachedRetrievalStore:
    """给检索入口套一层缓存的代理，对上层 Agent 完全透明。

    为什么用代理而不是改 KnowledgeRetriever：
    缓存是「横切关注点」，和 BM25 / RRF / MMR 这些召回算法没有任何关系。
    塞进检索器内部会让它同时承担「怎么召回」和「要不要缓存」两件事，
    以后动任何一条召回路径都要小心别碰坏缓存。套一层代理则两边都不用改：
    - RetrievalAgent 拿到的还是满足 RetrievalStore 协议的对象
    - KnowledgeRetriever 一行代码都不用动

    只对 search() 做缓存。add_documents / clear 等写操作直接透传——
    顺带在代理里 bump 版本是错误的：代理不知道这次写入到底成功了没有，
    版本失效必须发生在「确认写成功之后」，那是 service 层的职责。
    """

    def __init__(self, store: Any, cache: RetrievalCache) -> None:
        self._store = store
        self._cache = cache

    @property
    def is_ready(self) -> bool:
        return bool(self._store.is_ready)

    def search(self, query: str, k: int = 4) -> list[dict]:
        return self._cache.get_or_search(query, k, lambda: self._store.search(query, k=k))

    def add_documents(self, documents: Sequence[Any]) -> int:
        return self._store.add_documents(documents)

    def clear(self) -> None:
        self._store.clear()

    def save(self) -> None:
        self._store.save()

    def load(self) -> bool:
        return self._store.load()

    def all_chunks(self) -> list[dict]:
        return self._store.all_chunks()

    @property
    def stats(self) -> dict:
        return self._store.stats

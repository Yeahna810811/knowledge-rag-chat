from __future__ import annotations

import logging

from frontend.local_rag.config.settings import Settings, get_settings
from frontend.local_rag.core.agents import AgentOrchestrator
from frontend.local_rag.core.document_processor import DocumentProcessor
from frontend.local_rag.core.embedding_service import EmbeddingService
from frontend.local_rag.core.retrieval import (
    KnowledgeRetriever,
    LexicalStore,
    RetrievalMode,
)
from frontend.local_rag.core.retrieval.hybrid_retriever import HybridConfig
from frontend.local_rag.core.retrieval.query_rewrite import (
    PRFConfig,
    build_rewriter,
)
from frontend.local_rag.cache.redis_client import RedisClient
from frontend.local_rag.core.vector_store import VectorStoreManager
from frontend.local_rag.db.database import Database
from frontend.local_rag.services.conversation_store import ConversationStore
from frontend.local_rag.services.rate_limiter import RateLimiter
from frontend.local_rag.services.retrieval_cache import (
    CachedRetrievalStore,
    RetrievalCache,
)
from frontend.local_rag.utils.file_utils import ensure_dir

logger = logging.getLogger(__name__)

# 喂给 LLM 的上下文轮数上限。
# 注意：这是「读取侧」的窗口，数据库里保存的是完整历史，不在此处裁剪。
MAX_HISTORY_TURNS = 10


class KnowledgeService:
    """知识库服务：多 Agent 协同完成入库、双模式问答、状态与会话管理。"""

    def __init__(
        self,
        settings: Settings,
        database: Database | None = None,
        redis_client: RedisClient | None = None,
    ) -> None:
        self.settings = settings
        self.upload_dir = ensure_dir(settings.upload_dir)

        # 会话历史从内存 dict 换成数据库持久化层。
        # database 可注入，方便测试用临时 SQLite / 复用同一个 engine。
        self._database = database or build_database(settings)
        self._conversation_store = ConversationStore(self._database)

        # Redis 可注入（测试用 fakeredis / 故障桩），不传则按 settings 懒加载。
        # 懒加载的意义：没配 Redis 时，导入、建表这些路径不该去连它。
        self._redis_client = redis_client
        self._retrieval_cache: RetrievalCache | None = None
        self._rate_limiter: RateLimiter | None = None

        # Lazy-init heavy components (embedding model / LLM) on first use.
        self._embedding_service: EmbeddingService | None = None
        self._vector_store_manager: VectorStoreManager | None = None
        self._document_processor: DocumentProcessor | None = None
        self._orchestrator: AgentOrchestrator | None = None
        self._retriever: KnowledgeRetriever | None = None
        self._index_loaded = False

    def _build_retriever(self) -> KnowledgeRetriever:
        """构造统一检索入口，并决定是否需要加载 embedding 模型。

        这里藏着一个实打实的启动优化：
        纯 BM25 模式下，如果稀疏索引已经落盘，就完全不加载 embedding 模型——
        bge-small-zh 权重 + 分词器要好几百 MB，冷启动往往卡在这一步。
        只有当稀疏索引不存在（需要迁移）或模式不是纯 bm25 时才建稠密路。
        """
        mode = RetrievalMode.parse(self.settings.retrieval_mode)
        lexical = LexicalStore(self.settings.bm25_index_dir)

        need_dense = (
            mode is not RetrievalMode.BM25
            or not self.settings.retrieval_lazy_dense
            or not lexical.index_path.exists()
        )

        dense: VectorStoreManager | None = None
        if need_dense:
            self._embedding_service = EmbeddingService(self.settings)
            self._vector_store_manager = VectorStoreManager(
                self.settings, self._embedding_service.get_embeddings()
            )
            dense = self._vector_store_manager

        return KnowledgeRetriever(
            lexical_store=lexical,
            dense_store=dense,
            mode=mode,
            hybrid_config=HybridConfig(
                bm25_weight=self.settings.hybrid_bm25_weight,
                dense_weight=self.settings.hybrid_dense_weight,
            ),
            dense_fallback=self.settings.retrieval_dense_fallback,
            query_rewriter=self._build_rewriter(),
            rewrite_weight=self.settings.query_rewrite_weight,
        )

    def _build_rewriter(self):
        """按配置构造查询改写器。

        llm 模式这里不注入客户端：LLM 客户端由 generation 层持有，
        检索层不该反向依赖生成层。需要 llm 改写时，在 app 启动处
        用 `build_rewriter("llm", complete=...)` 显式装配，
        避免「检索层偷偷多调一次大模型」这种看不见的成本。
        """
        return build_rewriter(
            self.settings.retrieval_query_rewrite,
            config=PRFConfig(
                feedback_docs=self.settings.query_rewrite_feedback_docs,
                expansion_terms=self.settings.query_rewrite_terms,
                expansion_weight=self.settings.query_rewrite_weight,
            ),
        )

    def _ensure_runtime(self) -> AgentOrchestrator:
        if self._orchestrator is not None:
            return self._orchestrator

        self._retriever = self._build_retriever()
        self._document_processor = DocumentProcessor(self.settings)
        self._orchestrator = AgentOrchestrator(
            settings=self.settings,
            document_processor=self._document_processor,
            # 交给 Agent 的是套了缓存的代理，而不是裸检索器。
            # 检索算法本身一行没动，缓存只是外面的一层壳。
            vector_store_manager=self._cached_store(),
            upload_dir=self.upload_dir,
        )
        if not self._index_loaded:
            self._retriever.load()
            self._index_loaded = True
        return self._orchestrator

    @property
    def retriever(self) -> KnowledgeRetriever:
        self._ensure_runtime()
        assert self._retriever is not None
        return self._retriever

    @property
    def vector_store_manager(self) -> KnowledgeRetriever:
        """向后兼容旧字段名，实际返回统一检索入口。"""
        return self.retriever

    @property
    def orchestrator(self) -> AgentOrchestrator:
        return self._ensure_runtime()

    @property
    def database(self) -> Database:
        """全局唯一的 Database 实例：engine 在这里，不在每个请求里重建。"""
        return self._database

    @property
    def conversation_store(self) -> ConversationStore:
        return self._conversation_store

    # ------------------------------------------------------------------ Redis
    @property
    def redis_client(self) -> RedisClient:
        """全局唯一的 RedisClient，连接与连接池都归它管。"""
        if self._redis_client is None:
            self._redis_client = build_redis_client(self.settings)
        return self._redis_client

    @property
    def retrieval_cache(self) -> RetrievalCache:
        if self._retrieval_cache is None:
            self._retrieval_cache = RetrievalCache(
                self.redis_client,
                enabled=self.settings.redis_cache_enabled,
                ttl_seconds=self.settings.redis_cache_ttl_seconds,
                retrieval_mode=self.settings.retrieval_mode,
                rewrite_mode=self.settings.retrieval_query_rewrite,
                top_k=self.settings.retrieval_top_k,
            )
        return self._retrieval_cache

    @property
    def rate_limiter(self) -> RateLimiter:
        if self._rate_limiter is None:
            self._rate_limiter = RateLimiter(
                self.redis_client,
                enabled=self.settings.rate_limit_enabled,
                limit_requests=self.settings.rate_limit_requests,
                window_seconds=self.settings.rate_limit_window_seconds,
            )
        return self._rate_limiter

    def _cached_store(self) -> Any:
        """构造带缓存的检索代理。缓存关闭时直接返回原检索器，不多套一层。"""
        cache = self.retrieval_cache
        if not cache.enabled:
            return self._retriever
        return CachedRetrievalStore(self._retriever, cache)

    def _bump_knowledge_version(self, reason: str) -> None:
        """知识库内容确认变更后让旧检索缓存失效。

        只有 upload / reset 真正成功之后才调用；失败路径不会走到这里。
        Redis 挂掉时 bump 不成功也无所谓：缓存本来就写不进去，
        不存在「旧缓存还能命中」的问题。
        """
        try:
            self.retrieval_cache.bump_knowledge_version()
        except Exception as exc:  # noqa: BLE001
            logger.warning("知识库版本号递增失败（%s），不影响知识库内容: %s", reason, exc)

    def ingest_upload(self, filename: str, content: bytes) -> dict:
        result = self.orchestrator.ingest(filename, content)
        # 先成功再 bump：ingest 抛异常时不会执行到这里，缓存也就不会白白失效。
        self._bump_knowledge_version("upload")
        return result

    def ask(
        self,
        question: str,
        session_id: str = "default",
        mode: str = "rag",
    ) -> dict:
        question = question.strip()
        if not question:
            raise ValueError("问题内容不能为空")

        # 只把最近 MAX_HISTORY_TURNS 轮喂给 LLM，避免 context 无限增长；
        # 完整历史仍然留在数据库里。
        history = self._recent_history(session_id, limit=MAX_HISTORY_TURNS)
        result = self.orchestrator.ask(question, history=history, mode=mode)

        result["session_id"] = session_id
        result["history_turns"] = self._append_turn(
            session_id, question, result["answer"], mode, fallback_turns=len(history)
        )
        return result

    # ------------------------------------------------------------ 会话持久化
    def _recent_history(self, session_id: str, limit: int) -> list[dict]:
        """读历史给 LLM。数据库短暂不可用时降级为空历史，而不是让 /api/ask 500。

        检索 + 生成是主链路，会话持久化是副链路——副链路挂掉不该拖死主链路。
        """
        try:
            return self.conversation_store.get_recent_history(session_id, limit=limit)
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取会话历史失败，降级为空历史: %s", exc)
            return []

    def _append_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        mode: str,
        fallback_turns: int,
    ) -> int:
        """写一轮问答，返回写入后的总轮数（失败时返回降级值）。"""
        try:
            self.conversation_store.append_turn(session_id, question, answer, mode)
            return self.conversation_store.count_turns(session_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("写入会话历史失败，本轮未持久化: %s", exc)
            return fallback_turns + 1

    def get_history(self, session_id: str = "default") -> dict:
        history = self.conversation_store.get_history(session_id)
        return {"session_id": session_id, "history": history, "turns": len(history)}

    def status(self) -> dict:
        """注意：这个方法刻意不调用 _ensure_runtime()。

        /status 是健康检查入口，被 CI 和探针频繁调用。如果它触发
        embedding 模型加载，一次冷启动就会把健康检查拖到几十秒，
        在容器里会直接被 liveness probe 判死。所以这里只看文件是否存在。
        """
        if self._retriever is not None:
            ready = self._retriever.is_ready
            retrieval = self._retriever.stats
        else:
            lexical_file = self.settings.bm25_index_dir / "bm25_index.json"
            faiss_file = self.settings.faiss_index_dir / "index.faiss"
            ready = lexical_file.exists() or faiss_file.exists()
            retrieval = {
                "mode": str(self.settings.retrieval_mode).lower(),
                "note": "runtime 未初始化，仅按索引文件判断",
            }

        return {
            "vector_store_ready": ready,
            "database": self.database_status(),
            "redis": self.redis_status(),
            "upload_dir": str(self.upload_dir),
            "faiss_index_dir": str(self.settings.faiss_index_dir),
            "bm25_index_dir": str(self.settings.bm25_index_dir),
            "retrieval": retrieval,
            "embedding_model": self.settings.embedding_model,
            "chat_model": self.settings.chat_model,
            "dashscope_configured": bool(self.settings.dashscope_api_key),
            "langsmith_enabled": bool(
                self.settings.langchain_tracing_v2 and self.settings.langchain_api_key
            ),
            "active_sessions": self.active_sessions(),
            "agents": [
                "document_parse_agent",
                "retrieval_agent",
                "generation_agent",
            ],
            "modes": ["rag", "chat"],
            "runtime_initialized": self._orchestrator is not None,
        }

    def database_status(self) -> dict:
        """数据库健康状态。

        /api/status 是健康检查入口，数据库临时抖动不能让它 500——
        任何异常都降级成 healthy=False，绝不向上抛。
        """
        try:
            healthy = self.conversation_store.ping()
        except Exception as exc:  # noqa: BLE001
            logger.warning("数据库健康检查异常: %s", exc)
            healthy = False
        return {
            "backend": self._database.backend,
            "healthy": healthy,
        }

    def redis_status(self) -> dict:
        """Redis 健康状态。

        和 database_status() 一样的道理：/api/status 不能因为 Redis 抖动就 500。
        而且这里 ping 一次就够了，绝不为了 status 去初始化 embedding 模型。
        """
        try:
            healthy = self.redis_client.ping()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis 健康检查异常: %s", exc)
            healthy = False
        return {
            "backend": "redis",
            "healthy": healthy,
            "cache_enabled": bool(self.settings.redis_cache_enabled),
        }

    def active_sessions(self) -> int:
        """会话数。数据库不可用时返回 0，不影响 /api/status。"""
        try:
            return self.conversation_store.count_sessions()
        except Exception as exc:  # noqa: BLE001
            logger.warning("统计会话数失败: %s", exc)
            return 0

    def reset(self) -> dict:
        self.retriever.clear()
        self._bump_knowledge_version("reset")
        return {"message": "知识库已全部清空"}

    def close(self) -> None:
        """释放数据库与 Redis 连接。由 app 的 lifespan 在关闭时调用。"""
        self._database.dispose()
        if self._redis_client is not None:
            self._redis_client.close()

    def cache_metrics(self) -> dict:
        """缓存 / 限流的轻量计数，供后续 Observability 阶段接入。"""
        return {
            "cache": self.retrieval_cache.metrics(),
            "rate_limit": self.rate_limiter.metrics(),
        }

    def clear_history(self, session_id: str = "default") -> dict:
        self.conversation_store.clear_history(session_id)
        return {"message": "对话记忆已清空"}


def build_database(settings: Settings) -> Database:
    """按配置构造唯一的 Database 实例（engine 全局复用）。"""
    return Database(
        url=settings.database_url,
        echo=settings.database_echo,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
    )


def build_redis_client(settings: Settings) -> RedisClient:
    """按配置构造唯一的 RedisClient（连接池全局复用）。"""
    return RedisClient(
        url=settings.redis_url,
        socket_connect_timeout=settings.redis_socket_connect_timeout,
        socket_timeout=settings.redis_socket_timeout,
    )


def build_service(settings: Settings | None = None) -> KnowledgeService:
    return KnowledgeService(settings or get_settings())

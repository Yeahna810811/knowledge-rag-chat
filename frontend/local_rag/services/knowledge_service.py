from __future__ import annotations

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
from frontend.local_rag.core.vector_store import VectorStoreManager
from frontend.local_rag.utils.file_utils import ensure_dir

MAX_HISTORY_TURNS = 10


class KnowledgeService:
    """知识库服务：多 Agent 协同完成入库、双模式问答、状态与会话管理。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.upload_dir = ensure_dir(settings.upload_dir)
        self._chat_histories: dict[str, list[dict]] = {}

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
            vector_store_manager=self._retriever,
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

    def ingest_upload(self, filename: str, content: bytes) -> dict:
        return self.orchestrator.ingest(filename, content)

    def ask(
        self,
        question: str,
        session_id: str = "default",
        mode: str = "rag",
    ) -> dict:
        question = question.strip()
        if not question:
            raise ValueError("问题内容不能为空")

        history = self._chat_histories.get(session_id, [])
        result = self.orchestrator.ask(question, history=history, mode=mode)

        history.append({"question": question, "answer": result["answer"]})
        self._chat_histories[session_id] = history[-MAX_HISTORY_TURNS:]
        result["session_id"] = session_id
        result["history_turns"] = len(self._chat_histories[session_id])
        return result

    def get_history(self, session_id: str = "default") -> dict:
        history = self._chat_histories.get(session_id, [])
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
            "active_sessions": len(self._chat_histories),
            "agents": [
                "document_parse_agent",
                "retrieval_agent",
                "generation_agent",
            ],
            "modes": ["rag", "chat"],
            "runtime_initialized": self._orchestrator is not None,
        }

    def reset(self) -> dict:
        self.retriever.clear()
        return {"message": "知识库已全部清空"}

    def clear_history(self, session_id: str = "default") -> dict:
        self._chat_histories.pop(session_id, None)
        return {"message": "对话记忆已清空"}


def build_service(settings: Settings | None = None) -> KnowledgeService:
    return KnowledgeService(settings or get_settings())

from __future__ import annotations

from frontend.local_rag.config.settings import Settings, get_settings
from frontend.local_rag.core.agents import AgentOrchestrator
from frontend.local_rag.core.document_processor import DocumentProcessor
from frontend.local_rag.core.embedding_service import EmbeddingService
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
        self._index_loaded = False

    def _ensure_runtime(self) -> AgentOrchestrator:
        if self._orchestrator is not None:
            return self._orchestrator

        self._embedding_service = EmbeddingService(self.settings)
        self._vector_store_manager = VectorStoreManager(
            self.settings, self._embedding_service.get_embeddings()
        )
        self._document_processor = DocumentProcessor(self.settings)
        self._orchestrator = AgentOrchestrator(
            settings=self.settings,
            document_processor=self._document_processor,
            vector_store_manager=self._vector_store_manager,
            upload_dir=self.upload_dir,
        )
        if not self._index_loaded:
            self._vector_store_manager.load()
            self._index_loaded = True
        return self._orchestrator

    @property
    def vector_store_manager(self) -> VectorStoreManager:
        self._ensure_runtime()
        assert self._vector_store_manager is not None
        return self._vector_store_manager

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
        ready = False
        if self._vector_store_manager is not None:
            ready = self._vector_store_manager.is_ready
        else:
            index_file = self.settings.faiss_index_dir / "index.faiss"
            ready = index_file.exists()

        return {
            "vector_store_ready": ready,
            "upload_dir": str(self.upload_dir),
            "faiss_index_dir": str(self.settings.faiss_index_dir),
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
        self.vector_store_manager.clear()
        return {"message": "知识库已全部清空"}

    def clear_history(self, session_id: str = "default") -> dict:
        self._chat_histories.pop(session_id, None)
        return {"message": "对话记忆已清空"}


def build_service(settings: Settings | None = None) -> KnowledgeService:
    return KnowledgeService(settings or get_settings())

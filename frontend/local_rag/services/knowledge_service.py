from pathlib import Path
from uuid import uuid4

from frontend.local_rag.config.settings import Settings, get_settings
from frontend.local_rag.core.document_processor import DocumentProcessor
from frontend.local_rag.core.embedding_service import EmbeddingService
from frontend.local_rag.core.rag_chain import RAGChain
from frontend.local_rag.core.vector_store import VectorStoreManager
from frontend.local_rag.utils.file_utils import ensure_dir, is_supported_file

MAX_HISTORY_TURNS = 10  # 每个会话最多保留的历史轮数，避免prompt无限增长


class KnowledgeService:
    """知识库服务：实现文档导入、知识库问答（带多轮对话记忆）、状态查看与重置全流程业务逻辑"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.upload_dir = ensure_dir(settings.upload_dir)

        self.embedding_service = EmbeddingService(settings)
        self.vector_store_manager = VectorStoreManager(
            settings, self.embedding_service.get_embeddings()
        )
        self.document_processor = DocumentProcessor(settings)
        self.rag_chain = RAGChain(settings, self.vector_store_manager)

        self.vector_store_manager.load()

        # 按 session_id 维护多轮对话历史: {session_id: [{"question":..., "answer":...}, ...]}
        self._chat_histories: dict[str, list[dict]] = {}

    def ingest_upload(self, filename: str, content: bytes) -> dict:
        if not is_supported_file(filename):
            raise ValueError(
                "不支持该文件格式，仅支持：.txt、.md、.pdf、.docx、.csv"
            )

        safe_name = f"{uuid4().hex}_{Path(filename).name}"
        saved_path = self.upload_dir / safe_name
        saved_path.write_bytes(content)

        chunks = self.document_processor.process_file(saved_path)
        chunk_count = self.vector_store_manager.add_documents(chunks)

        return {
            "filename": filename,
            "saved_as": safe_name,
            "chunk_count": chunk_count,
            "message": "文档入库成功",
        }

    def ask(self, question: str, session_id: str = "default") -> dict:
        question = question.strip()
        if not question:
            raise ValueError("问题内容不能为空")

        history = self._chat_histories.get(session_id, [])

        # 知识库为空时依然可以正常聊天（退化为带记忆的普通AI助手，不强制要求先上传文档）
        result = self.rag_chain.run(question, history=history)

        # 更新该会话的历史记录
        history.append({"question": question, "answer": result["answer"]})
        self._chat_histories[session_id] = history[-MAX_HISTORY_TURNS:]

        return result

    def status(self) -> dict:
        return {
            "vector_store_ready": self.vector_store_manager.is_ready,
            "upload_dir": str(self.upload_dir),
            "faiss_index_dir": str(self.settings.faiss_index_dir),
            "embedding_model": self.settings.embedding_model,
            "chat_model": self.settings.chat_model,
            "dashscope_configured": bool(self.settings.dashscope_api_key),
            "active_sessions": len(self._chat_histories),
        }

    def reset(self) -> dict:
        self.vector_store_manager.clear()
        return {"message": "知识库已全部清空"}

    def clear_history(self, session_id: str = "default") -> dict:
        self._chat_histories.pop(session_id, None)
        return {"message": "对话记忆已清空"}

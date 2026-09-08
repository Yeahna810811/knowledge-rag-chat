from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from langsmith import traceable

from frontend.local_rag.core.agents.base import AgentResult, BaseAgent
from frontend.local_rag.core.document_processor import DocumentProcessor
from frontend.local_rag.core.vector_store import VectorStoreManager
from frontend.local_rag.utils.file_utils import ensure_dir, is_supported_file


class DocumentParseAgent(BaseAgent):
    """Parse uploaded documents, chunk text, and write embeddings into FAISS."""

    name = "document_parse_agent"

    def __init__(
        self,
        document_processor: DocumentProcessor,
        vector_store_manager: VectorStoreManager,
        upload_dir: Path,
    ) -> None:
        self.document_processor = document_processor
        self.vector_store_manager = vector_store_manager
        self.upload_dir = ensure_dir(upload_dir)

    @traceable(name="document_parse_agent", run_type="chain")
    def run(self, filename: str = "", content: bytes = b"", **_: Any) -> AgentResult:
        if not filename or not content:
            return AgentResult(
                agent=self.name,
                success=False,
                message="缺少文件名或文件内容",
            )
        if not is_supported_file(filename):
            return AgentResult(
                agent=self.name,
                success=False,
                message="不支持该文件格式，仅支持：.txt、.md、.pdf、.docx、.csv",
            )

        safe_name = f"{uuid4().hex}_{Path(filename).name}"
        saved_path = self.upload_dir / safe_name
        saved_path.write_bytes(content)

        chunks = self.document_processor.process_file(saved_path)
        chunk_count = self.vector_store_manager.add_documents(chunks)

        return AgentResult(
            agent=self.name,
            success=True,
            message="文档解析入库成功",
            data={
                "filename": filename,
                "saved_as": safe_name,
                "chunk_count": chunk_count,
            },
        )

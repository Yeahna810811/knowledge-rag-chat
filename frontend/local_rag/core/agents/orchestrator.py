from __future__ import annotations

from pathlib import Path
from typing import Any

from langsmith import traceable

from frontend.local_rag.config.settings import Settings
from frontend.local_rag.core.agents.document_agent import DocumentParseAgent
from frontend.local_rag.core.agents.generation_agent import GenerationAgent
from frontend.local_rag.core.agents.retrieval_agent import RetrievalAgent
from frontend.local_rag.core.document_processor import DocumentProcessor
from frontend.local_rag.core.vector_store import VectorStoreManager


class AgentOrchestrator:
    """Coordinate document / retrieval / generation agents for dual-mode QA."""

    def __init__(
        self,
        settings: Settings,
        document_processor: DocumentProcessor,
        vector_store_manager: VectorStoreManager,
        upload_dir: Path,
    ) -> None:
        self.settings = settings
        self.document_agent = DocumentParseAgent(
            document_processor=document_processor,
            vector_store_manager=vector_store_manager,
            upload_dir=upload_dir,
        )
        self.retrieval_agent = RetrievalAgent(
            vector_store_manager=vector_store_manager,
            top_k=settings.retrieval_top_k,
        )
        self.generation_agent = GenerationAgent(settings)

    @traceable(name="ingest_pipeline", run_type="chain")
    def ingest(self, filename: str, content: bytes) -> dict[str, Any]:
        result = self.document_agent.run(filename=filename, content=content)
        if not result.success:
            raise ValueError(result.message)
        return {
            **result.data,
            "message": result.message,
            "agent_trace": [result.agent],
        }

    @traceable(name="ask_pipeline", run_type="chain")
    def ask(
        self,
        question: str,
        history: list[dict] | None = None,
        mode: str = "rag",
    ) -> dict[str, Any]:
        history = history or []
        mode = (mode or "rag").lower()
        if mode not in {"rag", "chat"}:
            raise ValueError("mode 仅支持 rag 或 chat")

        agent_trace: list[str] = []
        sources: list[dict] = []

        if mode == "rag":
            retrieval = self.retrieval_agent.run(question=question)
            agent_trace.append(retrieval.agent)
            sources = retrieval.data.get("sources", [])

        generation = self.generation_agent.run(
            question=question,
            history=history,
            sources=sources,
            mode=mode,
        )
        agent_trace.append(generation.agent)
        if not generation.success:
            raise ValueError(generation.message or "生成失败")

        return {
            "question": question,
            "answer": generation.data["answer"],
            "sources": generation.data.get("sources", []),
            "mode": mode,
            "agent_trace": agent_trace,
        }

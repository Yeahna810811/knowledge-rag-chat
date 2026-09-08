"""Backward-compatible RAGChain wrapper around GenerationAgent + RetrievalAgent."""

from frontend.local_rag.config.settings import Settings
from frontend.local_rag.core.agents.generation_agent import (
    CHAT_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
    GenerationAgent,
)
from frontend.local_rag.core.agents.retrieval_agent import RetrievalAgent
from frontend.local_rag.core.vector_store import VectorStoreManager

SYSTEM_PROMPT = RAG_SYSTEM_PROMPT


class RAGChain:
    """检索增强问答：兼容旧调用方，内部委托给多 Agent。"""

    def __init__(self, settings: Settings, vector_store_manager: VectorStoreManager):
        self.settings = settings
        self.vector_store_manager = vector_store_manager
        self.retrieval_agent = RetrievalAgent(
            vector_store_manager, top_k=settings.retrieval_top_k
        )
        self.generation_agent = GenerationAgent(settings)
        self.system_prompt = SYSTEM_PROMPT
        self.chat_system_prompt = CHAT_SYSTEM_PROMPT

    def run(self, question, history=None, mode: str = "rag"):
        history = history or []
        sources = []
        if mode == "rag":
            retrieval = self.retrieval_agent.run(question=question)
            sources = retrieval.data.get("sources", [])
        generation = self.generation_agent.run(
            question=question,
            history=history,
            sources=sources,
            mode=mode,
        )
        return {
            "question": question,
            "answer": generation.data["answer"],
            "sources": generation.data.get("sources", []),
            "mode": mode,
        }

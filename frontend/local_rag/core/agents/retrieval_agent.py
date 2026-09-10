from __future__ import annotations

from typing import Any, Optional

from langsmith import traceable

from frontend.local_rag.core.agents.base import AgentResult, BaseAgent
from frontend.local_rag.core.retrieval.protocol import RetrievalStore


class RetrievalAgent(BaseAgent):
    """Top-K retrieval over the knowledge base (dense / sparse / hybrid).

    依赖的是 RetrievalStore 协议而非具体实现，所以传入
    VectorStoreManager（纯向量）或 KnowledgeRetriever（按模式路由）都可以，
    检索策略的变化不会波及这一层。
    """

    name = "retrieval_agent"

    def __init__(self, vector_store_manager: RetrievalStore, top_k: int = 4) -> None:
        self.vector_store_manager = vector_store_manager
        self.top_k = top_k

    @traceable(name="retrieval_agent", run_type="retriever")
    def run(self, question: str = "", top_k: Optional[int] = None, **_: Any) -> AgentResult:
        k = top_k or self.top_k
        if not question.strip():
            return AgentResult(agent=self.name, success=False, message="问题为空")

        if not self.vector_store_manager.is_ready:
            return AgentResult(
                agent=self.name,
                success=True,
                message="知识库尚未就绪",
                data={"sources": [], "ready": False},
            )

        sources = self.vector_store_manager.search(question, k=k)
        return AgentResult(
            agent=self.name,
            success=True,
            message=f"检索到 {len(sources)} 条相关片段",
            data={"sources": sources, "ready": True, "top_k": k},
        )

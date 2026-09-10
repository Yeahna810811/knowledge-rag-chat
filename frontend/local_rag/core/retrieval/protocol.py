"""检索存储协议：让 Agent 不再依赖具体的 VectorStoreManager 实现。

为什么需要它：
原来 retrieval_agent / document_agent 都直接标注 VectorStoreManager 类型。
加了 BM25 之后，传给它们的可能是 KnowledgeRetriever（内部又可能是纯稀疏、
纯稠密或融合）。如果不抽象出协议，就只能写 `Any` —— 那等于放弃类型检查。

用 Protocol（结构化子类型）而不是抽象基类，好处是：
VectorStoreManager 和 KnowledgeRetriever 不需要继承任何东西，
只要方法签名对得上就自动满足协议，老代码零改动。
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable


@runtime_checkable
class RetrievalStore(Protocol):
    """知识库检索与写入的最小接口。"""

    @property
    def is_ready(self) -> bool:
        """知识库是否已有内容可供检索。"""
        ...

    def search(self, query: str, k: int = 4) -> list[dict]:
        """返回 top-k 片段，形状为 [{"content": str, "metadata": dict}]。"""
        ...

    def add_documents(self, documents: Sequence[Any]) -> int:
        """写入一批文档，返回实际入库条数。"""
        ...

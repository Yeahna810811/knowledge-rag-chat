"""检索增强层：稀疏索引、融合、重排、混合检索、统一路由。

生产入口是 KnowledgeRetriever —— 它按 settings.retrieval_mode 在
稠密 / 稀疏 / 混合之间路由，对外接口与 VectorStoreManager 一致，
所以 retrieval_agent 无需改动：

    from frontend.local_rag.core.retrieval import (
        KnowledgeRetriever, RetrievalMode, LexicalStore,
    )

    retriever = KnowledgeRetriever(
        lexical_store=LexicalStore(settings.bm25_index_dir),
        dense_store=vector_store_manager,     # 纯 BM25 模式可传 None
        mode=RetrievalMode.parse(settings.retrieval_mode),
    )
    retriever.load()
    docs = retriever.search(question, k=4)

实验接口是 HybridRetriever —— 直接对两路做 RRF 融合，用于 A/B 评测：

    from frontend.local_rag.core.retrieval import HybridRetriever, HybridConfig

    retriever = HybridRetriever(dense_search=vector_store.search)
    retriever.index(all_chunks)
    docs = retriever.search(question, k=4)
"""

from frontend.local_rag.core.retrieval.bm25 import BM25Config, BM25Index, Hit
from frontend.local_rag.core.retrieval.fusion import interleave_dedup, reciprocal_rank_fusion
from frontend.local_rag.core.retrieval.hybrid_retriever import (
    DenseSearchFn,
    HybridConfig,
    HybridRetriever,
)
from frontend.local_rag.core.retrieval.knowledge_retriever import (
    KnowledgeRetriever,
    RetrievalMode,
)
from frontend.local_rag.core.retrieval.lexical_store import LexicalStore
from frontend.local_rag.core.retrieval.protocol import RetrievalStore
from frontend.local_rag.core.retrieval.rerank import (
    CrossEncoderReranker,
    MMRReranker,
    NoOpReranker,
    Reranker,
)

__all__ = [
    "BM25Config",
    "BM25Index",
    "Hit",
    "reciprocal_rank_fusion",
    "interleave_dedup",
    "HybridConfig",
    "HybridRetriever",
    "DenseSearchFn",
    "Reranker",
    "NoOpReranker",
    "MMRReranker",
    "CrossEncoderReranker",
    "KnowledgeRetriever",
    "RetrievalMode",
    "LexicalStore",
    "RetrievalStore",
]

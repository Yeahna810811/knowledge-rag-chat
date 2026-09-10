"""检索增强层：稀疏索引、融合、重排、混合检索。

对外主要入口是 HybridRetriever，它的 search() 签名与
VectorStoreManager.search() 完全一致，可直接替换。

    from frontend.local_rag.core.retrieval import HybridRetriever, HybridConfig

    retriever = HybridRetriever(dense_search=vector_store.search)
    retriever.index(all_chunks)          # 入库后调用
    docs = retriever.search(question, k=4)
"""

from frontend.local_rag.core.retrieval.bm25 import BM25Config, BM25Index, Hit
from frontend.local_rag.core.retrieval.fusion import interleave_dedup, reciprocal_rank_fusion
from frontend.local_rag.core.retrieval.hybrid_retriever import (
    DenseSearchFn,
    HybridConfig,
    HybridRetriever,
)
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
]

"""稠密 + 稀疏混合检索器（Hybrid Retrieval）。

这一层是项目 v1 → v2 的核心改造。动机来自评测数据：
    Hit@3 = 97.5%，但 Hit@1 只有 85%，MRR = 0.9146
说明「正确片段基本被召回了，但经常排在第 2~3 位，被干扰片段压着」。
生成阶段读的是 Top-4，排序靠后被稀释；如果 Top-1 就是噪声，模型更容易跑偏。

混合检索为什么能改善排序：
- 稠密检索（bge-small-zh）擅长语义泛化：「退款怎么办」能召回「退货流程」。
  但对型号、编号、API 名、错误码这类必须精确匹配的字符串不敏感——
  向量空间里 "qwen-plus" 和 "qwen-max" 距离极近，业务上却完全是两回事。
- BM25 恰好相反：字面精确匹配强，语义泛化为零。
- 两路的错误模式不相关，融合后能互相补位，这就是混合检索有效的根本原因。

对外接口与 VectorStoreManager.search() 完全一致（返回 list[dict]），
所以可以零改动替换进 retrieval_agent。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Sequence

from frontend.local_rag.core.retrieval.bm25 import BM25Config, BM25Index
from frontend.local_rag.core.retrieval.fusion import reciprocal_rank_fusion
from frontend.local_rag.core.retrieval.rerank import NoOpReranker, Reranker

logger = logging.getLogger(__name__)

# 稠密检索的签名：(query, k) -> list[{"content": str, "metadata": dict}]
DenseSearchFn = Callable[[str, int], list[dict]]


@dataclass
class HybridConfig:
    """混合检索的可调参数。默认值来自常见经验值，改动前先跑 A/B。"""

    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    rrf_k: int = 60
    candidate_pool: int = 20      # 每路先召回多少候选送进融合
    enable_bm25: bool = True
    enable_dense: bool = True


@dataclass
class HybridRetriever:
    """把「稠密检索 + BM25」用 RRF 融合，再可选重排。

    容错设计（这是能上生产的关键）：
    - 稠密检索不可用 / 抛异常 → 自动降级为纯 BM25，不抛给上层；
    - BM25 索引未构建 → 降级为纯稠密；
    - 两路都不可用 → 返回空列表（与原来「知识库为空」的行为一致）。
    检索是整条链路的第一步，它挂了整条链路就挂了，所以它必须比其它环节更耐操。
    """

    dense_search: DenseSearchFn | None = None
    config: HybridConfig = field(default_factory=HybridConfig)
    bm25_config: BM25Config = field(default_factory=BM25Config)
    reranker: Reranker | None = field(default_factory=NoOpReranker)

    _chunks: list[dict] = field(default_factory=list, init=False)
    _bm25: BM25Index = field(default_factory=BM25Index, init=False)
    _content_to_indices: dict[str, list[int]] = field(default_factory=dict, init=False)
    _degraded: bool = field(default=False, init=False)

    # ---------------- 索引 ----------------
    @property
    def is_ready(self) -> bool:
        return bool(self._chunks)

    @property
    def degraded(self) -> bool:
        """上一次 search 是否发生了降级（用于监控告警）。"""
        return self._degraded

    def index(self, chunks: Sequence[dict]) -> "HybridRetriever":
        """用当前知识库的全部 chunk 建稀疏索引。

        chunks 的形状与 VectorStoreManager.search() 的返回值一致：
        [{"content": str, "metadata": dict}, ...]

        调用时机：每次 add_documents 之后。全量重建，N 在万级以下开销可忽略
        （本项目语料 4 篇文档，实测重建 < 5ms）。
        """
        self._chunks = [dict(c) for c in chunks]
        self._bm25 = BM25Index(self.bm25_config).build([c.get("content", "") for c in self._chunks])

        self._content_to_indices = {}
        for i, chunk in enumerate(self._chunks):
            key = str(chunk.get("content", ""))
            self._content_to_indices.setdefault(key, []).append(i)
        return self

    # ---------------- 检索 ----------------
    def search(self, query: str, k: int = 4) -> list[dict]:
        """返回 top-k，形状与 VectorStoreManager.search() 完全一致。"""
        if not self._chunks:
            return []

        self._degraded = False
        pool = max(self.config.candidate_pool, k)
        ranked_lists: list[list[int]] = []
        weights: list[float] = []
        provenance: dict[int, list[str]] = {}

        # ---- 路 1：稀疏 BM25 ----
        if self.config.enable_bm25 and self._bm25.is_ready:
            hits = self._bm25.search(query, k=pool)
            if hits:
                ranked_lists.append([h.index for h in hits])
                weights.append(self.config.bm25_weight)
                for h in hits:
                    provenance.setdefault(h.index, []).append("bm25")

        # ---- 路 2：稠密向量 ----
        if self.config.enable_dense and self.dense_search is not None:
            try:
                dense_docs = self.dense_search(query, pool)
            except Exception as exc:  # 向量库挂了不能连累整条链路
                self._degraded = True
                logger.warning("dense retrieval failed, degrade to sparse only: %s", exc)
                dense_docs = []

            dense_indices = self._resolve_indices(dense_docs)
            if dense_indices:
                ranked_lists.append(dense_indices)
                weights.append(self.config.dense_weight)
                for i in dense_indices:
                    provenance.setdefault(i, [])
                    if "dense" not in provenance[i]:
                        provenance[i].append("dense")

        if not ranked_lists:
            return []

        fused = reciprocal_rank_fusion(ranked_lists, k=self.config.rrf_k, weights=weights)

        # ---- 重排 ----
        if self.reranker is not None and len(fused) > 1:
            candidates = [self._chunks[i].get("content", "") for i, _ in fused]
            fused_scores = [score for _, score in fused]
            try:
                new_scores = self.reranker.rerank(query, candidates, fused_scores)
            except Exception as exc:
                self._degraded = True
                logger.warning("reranker failed, keep fused order: %s", exc)
                new_scores = None

            if new_scores is not None and len(new_scores) == len(fused):
                fused = sorted(
                    ((idx, float(score)) for (idx, _), score in zip(fused, new_scores)),
                    key=lambda item: (-item[1], item[0]),
                )

        results: list[dict] = []
        for idx, score in fused[:k]:
            chunk = dict(self._chunks[idx])
            meta = dict(chunk.get("metadata", {}) or {})
            meta["retrieval_score"] = round(float(score), 6)
            meta["retrieved_by"] = ",".join(provenance.get(idx, []))
            results.append({"content": chunk.get("content", ""), "metadata": meta})
        return results

    # ---------------- 内部 ----------------
    def _resolve_indices(self, dense_docs: Sequence[dict]) -> list[int]:
        """把稠密检索返回的 chunk 映射回本地下标。

        为什么需要映射：FAISS 只返回 page_content + metadata，不带全局下标，
        而 RRF 必须在「同一个下标空间」里做融合。

        已知局限：如果知识库里存在两段完全相同的文本（content 一模一样），
        映射会取其中最先出现的那个。对融合排序无影响（内容相同 = 信息相同），
        但 metadata.source 可能指向另一处。生产环境可在入库时给每个 chunk
        写入 chunk_id 元数据，用 chunk_id 精确映射——这是 v2.1 的 TODO。
        """
        indices: list[int] = []
        used: set[int] = set()
        for doc in dense_docs:
            key = str(doc.get("content", ""))
            pool = self._content_to_indices.get(key)
            if not pool:
                continue
            for candidate in pool:
                if candidate not in used:
                    used.add(candidate)
                    indices.append(candidate)
                    break
        return indices

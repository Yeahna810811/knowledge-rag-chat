"""检索结果重排模块。

召回阶段使用 Dense 或 BM25 快速获取候选文档，但召回分数并不一定能够
准确反映 Query 与候选文档之间的最终相关性。

对于 Dense Retrieval，Query 和 Document 通常分别编码为向量后计算相似度；
BM25 则基于词频、IDF 和文档长度等统计信息进行匹配。
重排阶段只处理召回得到的少量候选文档，因此可以使用计算成本更高的方法
进一步优化候选结果的顺序。

本模块提供三种重排策略：

- NoOpReranker：
  不执行重排，直接保留原始召回顺序，用于默认或无额外模型的场景。

- MMRReranker：
  基于候选文档之间的相似度进行去冗余，降低高度重复 chunk 同时进入
  Context 的概率。对于存在 chunk_overlap 的知识库，可以提高上下文多样性。

- CrossEncoderReranker：
  使用 Cross-Encoder 对 Query-Document 对进行联合打分，并根据相关性重新排序。
  计算成本高于召回阶段，因此通常只应用于较小的候选集合。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from frontend.local_rag.core.retrieval.tokenizer import tokenize


@runtime_checkable
class Reranker(Protocol):
    """重排器协议：给定 query 和候选文本，返回每个候选的新得分（越大越靠前）。

    scores 是上游（融合层）给出的相关性分，可选。需要它的重排器（如 MMR）
    应当优先使用它；不需要的（如交叉编码器）可忽略。
    """

    @property
    def name(self) -> str: ...

    def rerank(
        self,
        query: str,
        candidates: Sequence[str],
        scores: Sequence[float] | None = None,
    ) -> Sequence[float]: ...


@dataclass
class NoOpReranker:
    """不重排。保持融合层给出的顺序（返回递减的伪得分）。"""

    @property
    def name(self) -> str:
        return "noop"

    def rerank(
        self,
        query: str,
        candidates: Sequence[str],
        scores: Sequence[float] | None = None,
    ) -> Sequence[float]:
        n = len(candidates)
        return [float(n - i) for i in range(n)]


@dataclass
class MMRReranker:
    """MMR（Maximal Marginal Relevance）去冗余重排，零依赖。

    迭代式贪心：每次选一个候选，使得
        score = λ * relevance - (1-λ) * max_similarity_to_already_selected

    relevance 用融合层给的原分（已归一化），similarity 用 token Jaccard。
    λ=1 等价于不重排；λ 越小越强调多样性。

    注意：MMR 是「提升多样性」而不是「提升相关性」。它能把重复片段挤掉，
    让 Top-K 覆盖更多不同信息，但不会让原本排在 5 名的正确片段跳到第 1。
    因此它主要用于降低候选结果冗余、提高上下文多样性，并不以提升 Hit@1 为主要目标。
    """

    lambda_param: float = 0.7

    @property
    def name(self) -> str:
        return f"mmr(lambda={self.lambda_param})"

    def rerank(
        self,
        query: str,
        candidates: Sequence[str],
        scores: Sequence[float] | None = None,
    ) -> Sequence[float]:
        n = len(candidates)
        if n <= 1:
            return [float(max(n, 1))]

        token_sets = [set(tokenize(c)) for c in candidates]

        # 相关性分优先用上游给的真实分数（RRF 融合分），
        # 拿不到时才退化成「按位置给分」。用位置当相关性是偷懒：
        # 它丢弃了「第 1 名和第 2 名差距有多大」这个信息，
        # 导致明明第 2 名分数只低一点点、却和第 5 名一视同仁。
        if scores is not None and len(scores) == n:
            peak = max(scores)
            base = [(s / peak if peak > 0 else (n - i) / n) for i, s in enumerate(scores)]
        else:
            base = [(n - i) / n for i in range(n)]

        def jaccard(i: int, j: int) -> float:
            a, b = token_sets[i], token_sets[j]
            if not a or not b:
                return 0.0
            union = a | b
            if not union:
                return 0.0
            return len(a & b) / len(union)

        selected: list[int] = []
        scores = [0.0] * n
        remaining = list(range(n))

        while remaining:
            best_idx = None
            best_value = -math.inf
            for i in remaining:
                redundancy = max((jaccard(i, j) for j in selected), default=0.0)
                value = self.lambda_param * base[i] - (1.0 - self.lambda_param) * redundancy
                if value > best_value or (value == best_value and best_idx is not None and i < best_idx):
                    best_value = value
                    best_idx = i
            # best_idx 不可能为 None（remaining 非空），仅为类型收敛
            chosen = int(best_idx) if best_idx is not None else remaining[0]
            scores[chosen] = best_value if best_value != -math.inf else 0.0
            selected.append(chosen)
            remaining.remove(chosen)

        # MMR 的 value 可能为负，整体平移保证递减可读（顺序不变）
        floor = min(scores)
        return [s - floor + 1e-9 for s in scores]


@dataclass
class CrossEncoderReranker:
    """交叉编码器重排（生产级，需要模型权重）。

    用法：
        CrossEncoderReranker(model_name="BAAI/bge-reranker-base")
    首次使用会下载约 1.1GB 模型。延迟量级：top-20 候选约 100~300ms（CPU）。

    失败策略交给上层：HybridRetriever 捕获 LoadError 后降级为 NoOpReranker，
    并记录一次 warning——检索链路不能因为重排模型缺失而整体不可用。
    """

    model_name: str = "BAAI/bge-reranker-base"
    max_length: int = 512
    batch_size: int = 16

    _model: object | None = None

    @property
    def name(self) -> str:
        return f"cross-encoder({self.model_name})"

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - 环境相关
                raise RuntimeError(
                    "CrossEncoderReranker needs sentence-transformers. "
                    "Install with: pip install sentence-transformers"
                ) from exc
            self._model = CrossEncoder(
                self.model_name, max_length=self.max_length
            )
        return self._model

    def rerank(
        self,
        query: str,
        candidates: Sequence[str],
        scores: Sequence[float] | None = None,
    ) -> Sequence[float]:
        # 交叉编码器自己算 query-doc 相关性，不需要上游分数
        if not candidates:
            return []
        model = self._ensure_model()
        pairs = [[query, c] for c in candidates]
        raw = model.predict(pairs, batch_size=self.batch_size)  # type: ignore[attr-defined]
        return [float(v) for v in raw]

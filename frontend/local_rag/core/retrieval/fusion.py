"""多路灯召结果的融合层。

核心是 RRF（Reciprocal Rank Fusion）。

为什么是 RRF 而不是「把两路分数归一化后加权相加」：
- 稠密检索（cosine，约 -1~1）和 BM25（无上界，跟 query 长度强相关）的分数
  尺度完全不可比。要加权就得先归一化，而归一化本身会引入新的超参和偏差。
- RRF 只用「排名」不用「分数」，天然跨尺度可比，且对某一路分数整体偏高/偏低
  完全免疫。这也是 Elasticsearch、Vespa、Pinecone 默认都提供 RRF 的原因。

公式：score(d) = Σ_i w_i / (k + rank_i(d))
- k 是阻尼常数（默认 60，来自原始论文）。k 越小，头部排名权重越大；
  k→∞ 时各文档得分趋同，融合退化成「被几路召回」的计数。
- 未出现在某一路召回列表里的文档，该路得分为 0，不会因为有排名而被惩罚。
"""

from __future__ import annotations

from typing import Sequence


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[int]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[int, float]]:
    """把多路「按排名排序的文档下标列表」融合成一个排序。

    Args:
        ranked_lists: 每路一个 list，元素为文档下标，已按该路得分降序。
        k: 阻尼常数。
        weights: 每路的权重，默认等权。

    Returns:
        [(doc_index, fused_score), ...]，按 fused_score 降序。
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(
            f"weights length ({len(weights)}) must match ranked_lists ({len(ranked_lists)})"
        )

    fused: dict[int, float] = {}
    for weight, ranked in zip(weights, ranked_lists):
        if weight <= 0:
            continue
        for rank, doc_index in enumerate(ranked, start=1):
            fused[doc_index] = fused.get(doc_index, 0.0) + weight / (k + rank)

    # 得分相同时按文档下标升序，保证结果可复现（评测必须确定性）
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


def interleave_dedup(ranked_lists: Sequence[Sequence[int]]) -> list[int]:
    """简单的轮询去重融合，仅用于对照实验 / 单测。

    相比 RRF 的缺点：完全丢弃了每路内部的置信度信息，且对列表长度敏感。
    """
    seen: set[int] = []
    out: list[int] = []
    max_len = max((len(r) for r in ranked_lists), default=0)
    for position in range(max_len):
        for ranked in ranked_lists:
            if position < len(ranked):
                doc_index = ranked[position]
                if doc_index not in seen:
                    seen.add(doc_index)
                    out.append(doc_index)
    return out

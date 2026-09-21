"""纯 Python BM25 稀疏检索器。

设计目标：
- 为项目提供稀疏召回及融合排序所需的 BM25 得分；
- 使用项目自定义 tokenizer，支持中文 n-gram 分词；
- 避免额外引入 BM25 第三方依赖，并保持检索逻辑可控。

BM25 相对基础 TF-IDF 的两个重要改进：
1. 词频饱和：词频增加时，相关性得分不会线性增长；
   k1 用于控制词频饱和速度。
2. 长度归一：较长文档天然包含更多词；
   b 用于控制文档长度对相关性得分的影响。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple, Sequence

from frontend.local_rag.core.retrieval.tokenizer import tokenize


class Hit(NamedTuple):
    """一次检索命中：文档下标 + 得分。"""

    index: int
    score: float


@dataclass
class BM25Config:
    k1: float = 1.5        # 词频饱和系数，常用 1.2~2.0
    b: float = 0.75        # 长度归一系数，0~1
    ngram: int = 2         # 中文 n-gram 上限
    delta: float = 0.5     # IDF 平滑项，避免出现负 IDF


@dataclass
class BM25Index:
    """可增量构建的 BM25 倒排索引。

    构建后不可变语义：调用 build() 之前 search() 一律返回空列表，
    这样上层（HybridRetriever）不需要额外判断「索引建没建」。
    """

    config: BM25Config = field(default_factory=BM25Config)

    _doc_tokens: list[list[str]] = field(default_factory=list, init=False)
    _doc_len: list[int] = field(default_factory=list, init=False)
    _tf: list[dict[str, int]] = field(default_factory=list, init=False)
    _df: dict[str, int] = field(default_factory=dict, init=False)
    _avgdl: float = field(default=0.0, init=False)
    _idf: dict[str, float] = field(default_factory=dict, init=False)

    @property
    def n_docs(self) -> int:
        return len(self._doc_tokens)

    @property
    def is_ready(self) -> bool:
        return self.n_docs > 0

    def build(self, documents: Sequence[str]) -> "BM25Index":
        """用一批文档（chunk 文本）构建索引。重复调用=重建，不是追加。"""
        self._doc_tokens = []
        self._doc_len = []
        self._tf = []
        self._df = {}

        for text in documents:
            tokens = tokenize(text, ngram=self.config.ngram)
            self._doc_tokens.append(tokens)
            self._doc_len.append(len(tokens))

            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            self._tf.append(tf)

            for tok in tf:
                self._df[tok] = self._df.get(tok, 0) + 1

        total_len = sum(self._doc_len)
        self._avgdl = (total_len / len(self._doc_len)) if self._doc_len else 0.0
        self._rebuild_idf()
        return self

    def _rebuild_idf(self) -> None:
        n = self.n_docs
        delta = self.config.delta
        self._idf = {
            term: math.log(1.0 + (n - df + delta) / (df + delta))
            for term, df in self._df.items()
        }

    # ---------------- 只读访问器（供查询改写等上层复用） ----------------
    def idf(self, term: str) -> float:
        """单个词的 IDF。没在语料里出现过的词返回 0，表示「无区分度」。"""
        return self._idf.get(term, 0.0)

    def term_frequencies(self, doc_index: int) -> dict[str, int]:
        """某个文档的词频表。

        暴露这个是为了让查询改写能复用索引里已有的分词结果，
        而不是把原文再分一遍（分词是构建期的成本，检索期不该重复付）。
        """
        if 0 <= doc_index < len(self._tf):
            return self._tf[doc_index]
        return {}

    def scores(self, query: str) -> list[float]:
        """返回 query 对全部文档的 BM25 得分（未排序，按下标对齐）。"""
        if not self.is_ready:
            return []

        query_tokens = tokenize(query, ngram=self.config.ngram)
        k1, b = self.config.k1, self.config.b
        avgdl = self._avgdl or 1.0

        out: list[float] = []
        for i, tf_map in enumerate(self._tf):
            doc_len = self._doc_len[i]
            score = 0.0
            for tok in query_tokens:
                idf = self._idf.get(tok)
                if idf is None:  # query 词在语料里没出现过，无区分度
                    continue
                tf = tf_map.get(tok, 0)
                if tf == 0:
                    continue
                norm = tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * doc_len / avgdl))
                score += idf * norm
            out.append(score)
        return out

    def search(self, query: str, k: int = 10) -> list[Hit]:
        """返回 top-k 命中，按得分降序。得分 <= 0 的文档不返回。

        为什么过滤 0 分：一个词都没命中的文档进融合层只会稀释 RRF 的排名，
        属于纯噪声。
        """
        scores = self.scores(query)
        hits = [Hit(i, s) for i, s in enumerate(scores) if s > 0.0]
        hits.sort(key=lambda h: (-h.score, h.index))
        return hits[:k]

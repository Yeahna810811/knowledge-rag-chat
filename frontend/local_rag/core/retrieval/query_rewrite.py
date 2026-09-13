"""查询改写：把「用户会怎么问」翻译成「文档会怎么写」。

背景（数据见 `evaluation/EVALUATION_AUDIT.md` 第四节）：
同一份语料、同一个 BM25，原始问法 MRR 0.893，换成口语化问法只剩 0.510。
掉下去的不是检索算法，是**词汇不匹配**——用户问「块之间留多少重叠」，
文档写的是「CHUNK_OVERLAP 相邻块重叠长度」，两边几乎没有共享词，
BM25 再准也无从下手。

修法有两条路：
1. 继续调检索器（调权重、换更大的 embedding 模型）—— 治标，本质没变；
2. 在检索**之前**改写查询 —— 治本。

本模块实现第 2 条，提供三种改写器：

`PseudoRelevanceFeedbackRewriter`（默认，prf）
    经典 RM3 伪相关反馈：先用原查询取 top-k 命中，把这些命中块里
    「权重高、且原查询没有」的词提出来当扩展词，再带上它们重新打分。
    无监督、零依赖、不需要 LLM。最关键的一点：**扩展词来自语料自身的用词**，
    等于让文档教查询怎么说话，而不是人工去猜同义词。

`LLMQueryRewriter`（llm）
    让模型把口语化问题改写成关键词式查询。生产中通常更准，
    代价是多一次 LLM 调用（延迟 + 成本 + 外部依赖）。

`NoopRewriter`（off）
    不改写，用于对照实验和排障。

三条设计约束（都是踩过坑之后定的）：
1. **原查询必须保持主导**：扩展部分权重默认 0.3。等权 RRF 融合翻车
   已经证明——低质量信号拿到过高权重，会把原本正确的结果挤下去。
2. **不引入人工同义词表**：同义词表看着最省事，但极容易演变成
   「照着评测集写答案」，属于变相作弊。扩展词只能来自语料。
3. **改写必须可观测**：`RewriteResult` 记录原始查询与扩展词，
   出问题时能一眼分清是改写引入的，还是检索本身的问题。
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from frontend.local_rag.core.retrieval.bm25 import BM25Index
from frontend.local_rag.core.retrieval.tokenizer import tokenize

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RewriteResult:
    """一次改写的结果。始终保留原始查询，便于观测与回退。"""

    original: str
    expanded_terms: tuple[str, ...] = ()
    strategy: str = "none"

    @property
    def is_effective(self) -> bool:
        """是否真的产生了扩展词（没扩展就等于没改写，不必多走一次检索）。"""
        return bool(self.expanded_terms)

    @property
    def summary(self) -> str:
        if not self.is_effective:
            return f"{self.strategy}: 未扩展"
        return f"{self.strategy}: +{'/'.join(self.expanded_terms)}"


class BaseQueryRewriter(ABC):
    """查询改写器接口。

    只依赖 BM25Index 的只读接口，所以能被单测轻松替换，
    也不需要真的起一个检索服务。
    """

    name: str = "base"

    @abstractmethod
    def rewrite(
        self,
        query: str,
        *,
        index: BM25Index | None,
        metadata_of: "Callable[[int], dict] | None" = None,
    ) -> RewriteResult:
        """返回改写结果。

        metadata_of：按下标取文档 metadata 的回调。只在需要「按来源限流」
        这类需要知道文档出处的策略里用到；没有也不影响基本改写。
        实现必须保证：任何异常都不应让检索失败。
        """


@dataclass
class PRFConfig:
    """RM3 伪相关反馈的参数。

    参数不是拍脑袋定的，两个「反漂移」开关来自实测失败案例：
    口语化问句「把文字变成一串数字这个活儿，是哪部分代码负责的？」
    第一轮召回了 4 篇同一份无关文档（评测说明），PRF 把它们当相关样本，
    扩展出 reference_answer / evidence / answerable 这类词，
    查询被带得更偏，召回反而更差——这就是经典的查询漂移。

    对策一（跨文档一致性）：扩展词至少要出现在 min_doc_support 篇反馈文档里。
    只在一篇里出现的词，更可能是那份文档的口癖，而不是语料的共性用词。
    对策二（单一来源限流）：同一个来源最多贡献 max_per_source 篇，
    避免整份反馈集被一份文档占满。
    """

    feedback_docs: int = 5      # 反馈集大小
    expansion_terms: int = 6    # 扩展词数量。过多会稀释原查询
    expansion_weight: float = 0.3  # 扩展部分的相对权重（原查询恒为 1.0）
    min_term_len: int = 2       # 过滤单字：中文 unigram 噪声大、区分度低
    min_doc_support: int = 2    # 扩展词至少出现在几篇反馈文档里
    max_per_source: int = 2     # 同一来源最多贡献几篇反馈文档
    min_idf: float = 0.6        # 过滤过于通用的词（idf 低于此值直接丢）


class PseudoRelevanceFeedbackRewriter(BaseQueryRewriter):
    """RM3 伪相关反馈（无监督、离线、零依赖）。

    扩展词打分沿用 RM3 的思路：候选词的「在反馈集里的词频 × 全局 IDF」。
    这样既偏向反馈文档里反复出现的词，又压掉「的」「是」这类到处都有的词。
    """

    name = "prf"

    def __init__(self, config: PRFConfig | None = None) -> None:
        self.config = config or PRFConfig()

    def rewrite(
        self,
        query: str,
        *,
        index: BM25Index | None,
        metadata_of: "Callable[[int], dict] | None" = None,
    ) -> RewriteResult:
        if index is None or not index.is_ready or not query.strip():
            return RewriteResult(query, strategy=self.name)

        # 多取一些候选，再按来源限流，保证反馈集不会被单一文档占满
        raw_hits = index.search(query, k=self.config.feedback_docs * 3)
        if not raw_hits:
            # 一个块都没召回，说明连「相关文档长什么样」都不知道，
            # 这时做伪相关反馈只会越改越偏。
            return RewriteResult(query, strategy=self.name)

        hits = self._limit_by_source(raw_hits, metadata_of)
        if not hits:
            return RewriteResult(query, strategy=self.name)

        query_terms = set(tokenize(query, ngram=index.config.ngram))
        support: dict[str, int] = {}          # 出现在几篇反馈文档里
        weights: dict[str, float] = {}        # 词频 × IDF 累加
        for hit in hits:
            for term, tf in index.term_frequencies(hit.index).items():
                if term in query_terms or len(term) < self.config.min_term_len:
                    continue
                if term.isdigit():
                    continue
                idf = index.idf(term)
                if idf < self.config.min_idf:   # 太通用的词没有区分度
                    continue
                support[term] = support.get(term, 0) + 1
                weights[term] = weights.get(term, 0.0) + tf * idf

        candidates = {
            term: value
            for term, value in weights.items()
            if support[term] >= self.config.min_doc_support
        }
        if not candidates:
            # 没有词能跨文档复现，说明反馈集本身不可靠 —— 宁可返回不改写。
            # 「没把握就不动」比「硬改一通」安全得多。
            return RewriteResult(query, strategy=self.name)

        # 按权重降序；同分时按词本身排序，保证结果可复现（评测要可复现）
        ranked = sorted(candidates.items(), key=lambda kv: (-kv[1], kv[0]))
        terms = tuple(term for term, _ in ranked[: self.config.expansion_terms])
        return RewriteResult(query, expanded_terms=terms, strategy=self.name)

    def _limit_by_source(
        self, hits: list, metadata_of: "Callable[[int], dict] | None"
    ) -> list:
        """同一来源（`source` 里的目录段）最多留 max_per_source 篇。

        没有 metadata 回调时退化成「不去重」——此时仍可用，
        只是少了这道防漂移保险。
        """
        counts: dict[str, int] = {}
        kept = []
        for hit in hits:
            meta = metadata_of(hit.index) if metadata_of is not None else {}
            group = self._source_group(str(meta.get("source", "")))
            if counts.get(group, 0) >= self.config.max_per_source:
                continue
            counts[group] = counts.get(group, 0) + 1
            kept.append(hit)
            if len(kept) >= self.config.feedback_docs:
                break
        return kept

    @staticmethod
    def _source_group(source: str) -> str:
        """把 `evaluation__README__003.md` 归到 `evaluation` 这一组。

        噪声语料是按源文档切块命名的，同一份文档会产出多个 chunk，
        它们在语义上是「同一个来源」，必须一起限流。
        """
        if not source:
            return "?"
        head = source.split("__", 1)[0]
        return head or source


LLM_REWRITE_PROMPT = """你是检索系统的查询改写器。把用户问题改写成更适合关键词检索的形式。

要求：
1. 只输出关键词，用空格分隔，不要输出解释、编号或标点；
2. 关键词要尽量使用技术文档里的正式用词（例如「切开」→ 切分、分块；「留多少重叠」→ 重叠长度）；
3. 最多 {max_terms} 个关键词，保留原问题里的实体与专有名词（模型名、文件名、参数名）。

用户问题：{query}
关键词："""


class LLMQueryRewriter(BaseQueryRewriter):
    """用 LLM 改写查询。

    刻意不直接依赖任何 SDK：只接收一个 `complete(prompt) -> str` 回调，
    由上层注入真实客户端。这样单测可以注入假实现，评测也不受外部服务可用性影响。
    """

    name = "llm"

    _STOPWORDS = frozenset(
        {"关键词", "查询", "用户", "问题", "以下", "一个", "什么", "怎么", "如何", "的", "了"}
    )

    def __init__(
        self,
        complete: Callable[[str], str],
        *,
        max_terms: int = 8,
        expansion_weight: float = 0.3,
    ) -> None:
        self._complete = complete
        self.max_terms = max_terms
        self.expansion_weight = expansion_weight

    def rewrite(
        self,
        query: str,
        *,
        index: BM25Index | None = None,
        metadata_of: "Callable[[int], dict] | None" = None,
    ) -> RewriteResult:
        if not query.strip():
            return RewriteResult(query, strategy=self.name)
        try:
            raw = self._complete(
                LLM_REWRITE_PROMPT.format(query=query, max_terms=self.max_terms)
            )
        except Exception as exc:  # 改写失败不能影响检索，退化成不改写
            logger.warning("llm query rewrite failed, using original query: %s", exc)
            return RewriteResult(query, strategy=self.name)

        terms = self._parse_terms(raw)
        return RewriteResult(query, expanded_terms=terms, strategy=self.name)

    def _parse_terms(self, raw: str) -> tuple[str, ...]:
        pieces = re.split(r"[\s,，、;；/|]+", str(raw))
        out: list[str] = []
        for piece in pieces:
            term = piece.strip().strip("：:。.「」\"'*-#")
            if len(term) < 2 or term in self._STOPWORDS:
                continue
            if term not in out:
                out.append(term)
        return tuple(out[: self.max_terms])


class NoopRewriter(BaseQueryRewriter):
    """不改写。用于对照实验、以及「改写反而更差」时的快速回退。"""

    name = "off"

    def rewrite(
        self,
        query: str,
        *,
        index: BM25Index | None = None,
        metadata_of: "Callable[[int], dict] | None" = None,
    ) -> RewriteResult:
        return RewriteResult(query, strategy=self.name)


def build_rewriter(
    mode: str | None,
    *,
    complete: Callable[[str], str] | None = None,
    config: PRFConfig | None = None,
    llm_max_terms: int = 8,
) -> BaseQueryRewriter:
    """按配置构造改写器。

    未知模式一律退化成 off 而不是抛异常：检索模式是运行时配置，
    配错一个字符串就让服务起不来是不划算的。
    """
    key = str(mode or "off").strip().lower()
    if key in {"prf", "rm3"}:
        return PseudoRelevanceFeedbackRewriter(config)
    if key == "llm" and complete is not None:
        return LLMQueryRewriter(complete, max_terms=llm_max_terms)
    if key == "llm":
        logger.warning("retrieval_query_rewrite=llm 但未提供 LLM 客户端，退化为 off")
    return NoopRewriter()


def score_with_expansion(
    base_scores: Sequence[float],
    expansion_scores: Iterable[Sequence[float]],
    weight: float,
) -> list[float]:
    """原查询得分 + weight × 各扩展词得分。

    之所以把扩展词单独打分再相加，而不是拼成一个长查询串：
    拼接会被 BM25 的词频饱和稀释，而且无法显式控制扩展部分的权重。
    这里权重是配置项，出问题可以调，也可以一键关掉。
    """
    out = [float(s) for s in base_scores]
    if weight <= 0.0:
        return out
    for extra in expansion_scores:
        if len(extra) != len(out):
            continue
        for i, value in enumerate(extra):
            out[i] += weight * float(value)
    return out

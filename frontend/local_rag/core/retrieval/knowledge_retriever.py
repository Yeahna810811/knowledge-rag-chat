"""统一检索入口：按配置在「稠密 / 稀疏 / 混合」之间路由。

为什么要有这一层：
项目原来只有 FAISS 稠密检索一条路，retrieval_agent 直接依赖 VectorStoreManager。
加了 BM25 之后如果让上层自己判断该问谁，编排逻辑会散落到各个 Agent 里。
这一层把「用哪种策略 + 怎么降级」收敛到一个可配置的对象，
对外暴露与 VectorStoreManager 完全相同的接口（is_ready / search /
add_documents / clear / save / load），所以 retrieval_agent 一行都不用改。

三种模式（由 settings.retrieval_mode 控制）：
- dense  ：纯 FAISS，v1 的原始行为，保留用于 A/B 与回滚；
- bm25   ：BM25 主导 + 稠密兜底。**生产默认**，依据见 EVALUATION_AUDIT.md；
- hybrid ：RRF 融合两路，实验用；实测在本项目语料上不优于 BM25 单路。

为什么默认 bm25 而不是 hybrid：
45 chunk 语料、真实 bge-small-zh 向量上的配对 bootstrap 检验显示，
混合检索相对 BM25 单路**没有任何权重的显著收益**：原始问法下显著更差
（ΔMRR -0.069 ~ -0.097，p ≤ 0.031），口语化问法下也只是打平（p 0.30~0.97）。
所以默认走 BM25，融合层保留为可选项，等换了更强的 embedding 模型再重新评估。

查询改写（settings.retrieval_query_rewrite）：
BM25 的已知短板是「词汇不匹配」——口语化提问下 MRR 从 0.893 掉到 0.510。
在检索前先把查询改写成更贴近文档用词的形式（见 query_rewrite.py），
是这个问题在检索层的正解；调检索器权重治不了。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from frontend.local_rag.core.retrieval.bm25 import BM25Config
from frontend.local_rag.core.retrieval.hybrid_retriever import (
    HybridConfig,
    HybridRetriever,
)
from frontend.local_rag.core.retrieval.lexical_store import LexicalStore
from frontend.local_rag.core.retrieval.query_rewrite import (
    BaseQueryRewriter,
    NoopRewriter,
    RewriteResult,
)

logger = logging.getLogger(__name__)


class RetrievalMode(str, Enum):
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"

    @classmethod
    def parse(cls, value: str | None) -> "RetrievalMode":
        raw = str(value or "").strip().lower()
        for member in cls:
            if member.value == raw:
                return member
        logger.warning("unknown retrieval_mode %r, fallback to bm25", value)
        return cls.BM25


@dataclass
class KnowledgeRetriever:
    """稠密 + 稀疏的统一检索门面。

    容错原则：检索挂了不能让整条问答链路报错。
    任何一路抛异常都记 warning 后走另一路，两路都不可用才返回空列表
    （与「知识库为空」的原始行为一致，generation_agent 已有处理）。
    """

    lexical_store: LexicalStore
    dense_store: Any | None = None          # VectorStoreManager，可能为 None（纯 BM25 模式）
    mode: RetrievalMode = RetrievalMode.BM25
    hybrid_config: HybridConfig = field(default_factory=HybridConfig)
    bm25_config: BM25Config = field(default_factory=BM25Config)
    # BM25 零命中时是否回落到稠密路。关掉就是严格纯稀疏，
    # 代价是遇到词汇完全不重叠的提问会直接返回空。
    dense_fallback: bool = True
    # 查询改写器。默认 NoopRewriter（不改写），由 service 按配置注入。
    # 只在 BM25 / 级联路径生效：hybrid 模式内部自建索引与两路融合，
    # 改写在那一层的收益需要单独评估，不在这里默默生效。
    query_rewriter: BaseQueryRewriter = field(default_factory=NoopRewriter)
    rewrite_weight: float = 0.3

    _hybrid: HybridRetriever | None = field(default=None, init=False)
    _hybrid_dirty: bool = field(default=True, init=False)
    _degraded: bool = field(default=False, init=False)
    _last_rewrite: RewriteResult | None = field(default=None, init=False)

    # ---------------- 状态 ----------------
    @property
    def last_rewrite(self) -> RewriteResult | None:
        """最近一次查询改写的详情，供日志 / 排查使用。"""
        return self._last_rewrite

    @property
    def is_ready(self) -> bool:
        """按当前模式判断知识库是否可用。"""
        if self.mode is RetrievalMode.DENSE:
            return bool(self.dense_store is not None and self.dense_store.is_ready)
        if self.lexical_store.is_ready:
            return True
        # 稀疏索引还没建但稠密库有货（老版本部署），仍视为可用
        return bool(self.dense_store is not None and self.dense_store.is_ready)

    @property
    def degraded(self) -> bool:
        """上一次检索是否发生过降级。给监控 / /status 用。"""
        return self._degraded

    @property
    def stats(self) -> dict:
        out = {
            "mode": self.mode.value,
            "degraded": self._degraded,
            "lexical": self.lexical_store.stats(),
        }
        if self.dense_store is not None:
            out["dense"] = {
                "backend": "faiss",
                "ready": bool(getattr(self.dense_store, "is_ready", False)),
            }
        else:
            out["dense"] = {"backend": "faiss", "ready": False, "note": "未加载（纯 BM25 模式）"}
        return out

    # ---------------- 写入 ----------------
    def add_documents(self, documents: Sequence[Any]) -> int:
        """同时喂给稀疏路和稠密路。返回稀疏路实际入库条数。

        两路都必须成功吗？不是。稀疏路是主路，它失败才算真失败；
        稠密路失败只记 warning——它只是兜底和实验用的旁路。
        """
        count = self.lexical_store.add_documents(documents)
        self._hybrid_dirty = True

        if self.dense_store is not None:
            try:
                self.dense_store.add_documents(list(documents))
            except Exception as exc:
                self._degraded = True
                logger.warning("dense store add failed, lexical still available: %s", exc)
        return count

    def clear(self) -> None:
        self.lexical_store.clear()
        self._hybrid = None
        self._hybrid_dirty = True
        if self.dense_store is not None:
            self.dense_store.clear()

    def save(self) -> None:
        self.lexical_store.save()
        if self.dense_store is not None:
            try:
                self.dense_store.save()
            except Exception as exc:
                logger.warning("dense store save failed: %s", exc)

    def load(self) -> bool:
        """恢复索引。返回稀疏路是否恢复成功。"""
        lexical_ok = self.lexical_store.load()
        dense_ok = False
        if self.dense_store is not None:
            try:
                dense_ok = bool(self.dense_store.load())
            except Exception as exc:
                logger.warning("dense store load failed: %s", exc)

        # 迁移：老版本部署只有 FAISS，没有 BM25 索引文件。
        # 这里把 FAISS 里已有的 chunk 导出来补建稀疏索引，一次性完成。
        if not lexical_ok and dense_ok:
            self._migrate_from_dense()
        self._hybrid_dirty = True
        return self.lexical_store.is_ready

    def _migrate_from_dense(self) -> None:
        """从 FAISS 的 docstore 中导出原文，补建 BM25 索引。

        为什么能这么做：LangChain 的 FAISS 除了向量，还把原文存在 docstore 里。
        读它是私有 API（docstore._dict），所以整段包在 try/except 里——
        读不到最多是不迁移，不能让服务起不来。
        """
        try:
            store = getattr(self.dense_store, "_vector_store", None)
            if store is None:
                return
            docstore = getattr(store, "docstore", None)
            docs = list(getattr(docstore, "_dict", {}).values())
            if not docs:
                return
            added = self.lexical_store.add_documents(docs)
            self.lexical_store.save()
            logger.info("migrated %d chunks from FAISS docstore to BM25 index", added)
        except Exception as exc:
            logger.warning("bm25 migration from dense skipped: %s", exc)

    # ---------------- 检索 ----------------
    def search(self, query: str, k: int = 4) -> list[dict]:
        self._degraded = False

        if self.mode is RetrievalMode.DENSE:
            return self._search_dense(query, k)

        if self.mode is RetrievalMode.HYBRID:
            return self._search_hybrid(query, k)

        # ---- 默认：BM25 主导 + 稠密兜底（级联，不是融合）----
        lexical_hits = self._safe_lexical(query, k)
        if lexical_hits:
            return lexical_hits
        if not self.dense_fallback:
            return []
        if self.dense_store is not None and self.dense_store.is_ready:
            fallback = self._search_dense(query, k)
            if fallback:
                self._degraded = True
                logger.info("bm25 miss, fell back to dense for query: %.40s", query)
                return fallback
        return []

    def _safe_lexical(self, query: str, k: int) -> list[dict]:
        try:
            result = self._rewrite(query)
            self._last_rewrite = result
            if result.is_effective:
                # 改写只用于第一轮召回；改写失败/无扩展词时退回原始查询，
                # 保证「改写层挂了，检索仍然按原样工作」。
                return self.lexical_store.search_weighted(
                    query,
                    extra_terms=result.expanded_terms,
                    extra_weight=self.rewrite_weight,
                    k=k,
                )
            return self.lexical_store.search(query, k=k)
        except Exception as exc:
            self._degraded = True
            logger.warning("lexical search failed: %s", exc)
            return []

    def _rewrite(self, query: str) -> RewriteResult:
        """跑一次查询改写。任何异常都退化为「不改写」。"""
        try:
            return self.query_rewriter.rewrite(
                query,
                index=self.lexical_store.index,
                metadata_of=self.lexical_store.metadata_of,
            )
        except Exception as exc:
            logger.warning("query rewrite failed, using original query: %s", exc)
            return RewriteResult(query, strategy="error")

    def _search_dense(self, query: str, k: int) -> list[dict]:
        if self.dense_store is None or not self.dense_store.is_ready:
            return []
        try:
            return self.dense_store.search(query, k=k)
        except Exception as exc:
            self._degraded = True
            logger.warning("dense search failed: %s", exc)
            return []

    def _search_hybrid(self, query: str, k: int) -> list[dict]:
        if self._hybrid is None or self._hybrid_dirty:
            chunks = self.lexical_store.all_chunks()
            if not chunks:
                return self._search_dense(query, k)
            self._hybrid = HybridRetriever(
                dense_search=(
                    self.dense_store.search if self.dense_store is not None else None
                ),
                config=self.hybrid_config,
                bm25_config=self.bm25_config,
            ).index(chunks)
            self._hybrid_dirty = False
        try:
            return self._hybrid.search(query, k=k)
        except Exception as exc:
            self._degraded = True
            logger.warning("hybrid search failed, degrade to bm25: %s", exc)
            return self._safe_lexical(query, k)

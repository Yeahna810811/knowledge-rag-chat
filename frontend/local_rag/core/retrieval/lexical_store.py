"""持久化 BM25 稀疏索引存储。

为什么需要它，而不是每次内存里现场建：
FAISS 索引能落盘（index.faiss + index.pkl），重启后 `VectorStoreManager.load()`
就能恢复。如果稀疏路只在内存里活着，服务一重启就退化成纯稠密检索——
那 BM25 带来的提升等于只在「刚入库完到重启前」这段窗口内有效，生产上没意义。

设计取舍：
1. **落 JSON 不落 pickle**。pickle 反序列化 = 任意代码执行，FAISS 那边的
   `allow_dangerous_deserialization=True` 已经是妥协，这里没必要再开一个口子。
   BM25 的倒排结构可以从原文无损重建，存「原文 + metadata」就够了。
2. **加载时重建索引而不是存倒排表**。重建 = 对每篇文档重新分词，
   代价 O(总 token 数)。实测 54 chunk 重建 < 5ms，万级 chunk 也在百毫秒量级，
   换来的是「存储格式与算法实现解耦」——改 k1/b/分词策略不用做数据迁移。
3. **不依赖 langchain**。只要求文档对象有 `page_content`/`metadata`，
   或字典有 `content`/`metadata`。检索层保持零第三方依赖，单测才好跑。
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from frontend.local_rag.core.retrieval.bm25 import BM25Config, BM25Index

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
INDEX_FILENAME = "bm25_index.json"


def _extract(doc: Any) -> tuple[str, dict]:
    """从 Document 对象或 dict 中取出 (content, metadata)。

    兼容两种输入是为了让这一层不 import langchain，
    同时又能直接吃 DocumentProcessor 产出的 Document。
    """
    if isinstance(doc, dict):
        return str(doc.get("content", "")), dict(doc.get("metadata") or {})
    content = getattr(doc, "page_content", None)
    if content is None and isinstance(doc, str):
        content = doc
    metadata = getattr(doc, "metadata", None)
    return str(content or ""), dict(metadata or {})


@dataclass
class LexicalStore:
    """可持久化的 BM25 稀疏检索存储。

    对外接口刻意与 VectorStoreManager 对齐：
        add_documents / search / save / load / clear / is_ready
    这样上层（KnowledgeRetriever）可以对两路做统一编排。
    """

    index_dir: Path
    config: BM25Config = field(default_factory=BM25Config)

    _texts: list[str] = field(default_factory=list, init=False)
    _metadatas: list[dict] = field(default_factory=list, init=False)
    _index: BM25Index = field(default_factory=BM25Index, init=False)

    # ---------------- 属性 ----------------
    @property
    def index_path(self) -> Path:
        return Path(self.index_dir) / INDEX_FILENAME

    @property
    def is_ready(self) -> bool:
        return bool(self._texts)

    @property
    def n_docs(self) -> int:
        return len(self._texts)

    def metadata_of(self, doc_index: int) -> dict:
        """按下标取 metadata。供查询改写按来源限流使用。"""
        if 0 <= doc_index < len(self._metadatas):
            return dict(self._metadatas[doc_index])
        return {}

    @property
    def index(self) -> BM25Index:
        """底层 BM25 索引的只读引用。

        查询改写需要读 IDF 与词频，但不应自己再分一遍词——
        分词是构建期已经付过的成本。
        """
        return self._index

    # ---------------- 写入 ----------------
    def add_documents(self, documents: Sequence[Any]) -> int:
        """追加文档并重建索引。返回新增条数。

        为什么全量重建而不是增量：BM25 的 IDF 依赖全局 df，
        插入新文档会改变所有词的 IDF，增量更新等价于重算一遍。
        N 在万级以下时重建比维护增量结构更简单也更不容易出错。
        """
        added = 0
        for doc in documents:
            content, metadata = _extract(doc)
            if not content.strip():
                continue
            self._texts.append(content)
            self._metadatas.append(metadata)
            added += 1

        if added:
            self._rebuild()
        return added

    def _rebuild(self) -> None:
        self._index = BM25Index(self.config).build(self._texts)

    # ---------------- 检索 ----------------
    def search(self, query: str, k: int = 4) -> list[dict]:
        """返回 top-k，形状与 VectorStoreManager.search() 一致。"""
        if not self._index.is_ready or not query.strip():
            return []

        hits = self._index.search(query, k=k)
        out: list[dict] = []
        for hit in hits:
            meta = dict(self._metadatas[hit.index])
            meta["retrieval_score"] = round(float(hit.score), 6)
            meta["retrieved_by"] = "bm25"
            out.append({"content": self._texts[hit.index], "metadata": meta})
        return out

    def all_chunks(self) -> list[dict]:
        """导出全部 chunk，供混合检索器建索引 / 重建稠密路。"""
        return [
            {"content": text, "metadata": dict(meta)}
            for text, meta in zip(self._texts, self._metadatas)
        ]

    def search_weighted(
        self,
        query: str,
        extra_terms: Sequence[str] = (),
        extra_weight: float = 0.3,
        k: int = 4,
    ) -> list[dict]:
        """带扩展词的检索：原查询得分 + extra_weight × 各扩展词得分。

        为什么不把扩展词直接拼进查询串：
        拼接后会被 BM25 的词频饱和稀释（5 个扩展词各出现 1 次，
        和原查询的一个核心词权重相当），而且无法控制扩展部分的占比。
        分开打分再加权，权重就是配置项——能调，也能一键关掉。
        """
        if not self._index.is_ready or not query.strip():
            return []

        base = self._index.scores(query)
        if not base:
            return []

        combined = list(base)
        for term in extra_terms:
            if not term:
                continue
            term_scores = self._index.scores(term)
            if len(term_scores) != len(combined):
                continue
            for i, value in enumerate(term_scores):
                combined[i] += extra_weight * value

        hits = [(i, s) for i, s in enumerate(combined) if s > 0.0]
        hits.sort(key=lambda item: (-item[1], item[0]))

        out: list[dict] = []
        for index, score in hits[:k]:
            meta = dict(self._metadatas[index])
            meta["retrieval_score"] = round(float(score), 6)
            meta["retrieved_by"] = "bm25" if not extra_terms else "bm25+rewrite"
            out.append({"content": self._texts[index], "metadata": meta})
        return out

    # ---------------- 持久化 ----------------
    def save(self) -> None:
        """原子写入，避免写一半崩了留下半个文件。"""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": FORMAT_VERSION,
            "config": {
                "k1": self.config.k1,
                "b": self.config.b,
                "ngram": self.config.ngram,
                "delta": self.config.delta,
            },
            "docs": [
                {"content": text, "metadata": meta}
                for text, meta in zip(self._texts, self._metadatas)
            ],
        }

        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=str(self.index_dir), prefix=".bm25_", suffix=".tmp"
        )
        try:
            with open(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            Path(tmp_name).replace(self.index_path)
        except Exception:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def load(self) -> bool:
        """从磁盘恢复。文件不存在返回 False。"""
        if not self.index_path.exists():
            return False
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("bm25 index unreadable, start empty: %s", exc)
            return False

        version = payload.get("version")
        if version != FORMAT_VERSION:
            logger.warning(
                "bm25 index version mismatch (%s != %s), start empty",
                version,
                FORMAT_VERSION,
            )
            return False

        docs = payload.get("docs") or []
        self._texts = [str(d.get("content", "")) for d in docs]
        self._metadatas = [dict(d.get("metadata") or {}) for d in docs]
        self._rebuild()
        return bool(self._texts)

    def clear(self) -> None:
        self._texts = []
        self._metadatas = []
        self._rebuild()
        self.index_path.unlink(missing_ok=True)

    # ---------------- 可观测 ----------------
    def stats(self) -> dict:
        """给 /status 接口用的统计信息。"""
        return {
            "backend": "bm25",
            "docs": self.n_docs,
            "avg_chars": (
                round(sum(len(t) for t in self._texts) / self.n_docs, 1)
                if self._texts
                else 0
            ),
            "persisted": self.index_path.exists(),
            "index_path": str(self.index_path),
        }

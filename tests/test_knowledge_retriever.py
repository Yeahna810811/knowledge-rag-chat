"""KnowledgeRetriever / LexicalStore 的单元测试。

这一层决定生产环境到底走哪条检索路，所以它必须能被无依赖地测：
用假的稠密存储（FakeDense）替代 FAISS，不开模型、不联网、秒级跑完。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frontend.local_rag.core.retrieval import (  # noqa: E402
    KnowledgeRetriever,
    LexicalStore,
    RetrievalMode,
)

PASS = 0
FAIL = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [ok]   {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}  {detail}")


class FakeDense:
    """假的稠密存储：用字符包含度打分的内存实现。"""

    def __init__(self, docs: Sequence[Any] | None = None, fail_on_search: bool = False):
        self.docs = list(docs or [])
        self.fail_on_search = fail_on_search
        self.fail_on_add = False
        self.loaded = False

    @property
    def is_ready(self) -> bool:
        return bool(self.docs)

    def add_documents(self, documents: Sequence[Any]) -> int:
        if self.fail_on_add:
            raise RuntimeError("dense write boom")
        self.docs.extend(documents)
        return len(documents)

    def search(self, query: str, k: int = 4) -> list[dict]:
        if self.fail_on_search:
            raise RuntimeError("dense boom")

        def score(doc: Any) -> float:
            content = doc["content"] if isinstance(doc, dict) else doc.page_content
            overlap = sum(1 for ch in set(query) if ch in content)
            return overlap / max(len(set(query)), 1)

        ranked = sorted(self.docs, key=score, reverse=True)
        out = []
        for doc in ranked[:k]:
            if isinstance(doc, dict):
                out.append({"content": doc["content"], "metadata": dict(doc["metadata"])})
            else:
                out.append({"content": doc.page_content, "metadata": dict(doc.metadata)})
        return out

    def save(self) -> None:
        pass

    def clear(self) -> None:
        self.docs = []


def as_dicts(pairs):
    return [{"content": c, "metadata": m} for c, m in pairs]


# ---------------------------------------------------------------- 持久化
def test_lexical_persistence() -> None:
    print("\n[LexicalStore 持久化]")
    with tempfile.TemporaryDirectory() as tmp:
        store = LexicalStore(Path(tmp))
        check("空库 is_ready=False", store.is_ready is False)
        check("空库 search 返回空", store.search("端口") == [])

        store.add_documents(
            as_dicts(
                [
                    ("服务默认监听端口为 8000。", {"source": "a.md"}),
                    ("向量索引使用 FAISS 并持久化到本地磁盘。", {"source": "b.md"}),
                ]
            )
        )
        check("入库 2 条", store.n_docs == 2)
        store.save()
        check("索引文件已落盘", store.index_path.exists())

        hits = store.search("默认端口", k=2)
        check("能检索到", len(hits) > 0)
        check("首条命中 a.md", hits and hits[0]["metadata"]["source"] == "a.md", str(hits[:1]))
        check("带 retrieval_score", "retrieval_score" in (hits[0]["metadata"] if hits else {}))
        check("标记来源 bm25", hits and hits[0]["metadata"]["retrieved_by"] == "bm25")

        # 重新加载
        fresh = LexicalStore(Path(tmp))
        check("load 成功", fresh.load() is True)
        check("恢复 2 条", fresh.n_docs == 2)
        check("恢复后检索一致", fresh.search("默认端口", k=1)[0]["metadata"]["source"] == "a.md")

        fresh.clear()
        check("clear 后为空", fresh.n_docs == 0)
        check("clear 后文件删除", not fresh.index_path.exists())


def test_lexical_edge_cases() -> None:
    print("\n[LexicalStore 边界]")
    with tempfile.TemporaryDirectory() as tmp:
        store = LexicalStore(Path(tmp))
        # 空白内容必须被跳过，否则会污染 IDF
        added = store.add_documents(as_dicts([("", {"source": "x"}), ("   ", {}), ("有效内容", {})]))
        check("空白文档被跳过", added == 1, f"added={added}")

    with tempfile.TemporaryDirectory() as tmp:
        store = LexicalStore(Path(tmp))
        check("load 不存在的文件返回 False", store.load() is False)

        # 损坏的 JSON 不能让服务起不来
        store.index_path.parent.mkdir(parents=True, exist_ok=True)
        store.index_path.write_text("{ broken json", encoding="utf-8")
        check("损坏文件被容错", store.load() is False)


# ---------------------------------------------------------------- 路由
def test_mode_routing() -> None:
    print("\n[模式路由]")
    docs = as_dicts(
        [
            ("服务默认监听端口为 8000。", {"source": "a.md"}),
            ("向量索引使用 FAISS 并持久化到本地磁盘。", {"source": "b.md"}),
        ]
    )
    with tempfile.TemporaryDirectory() as tmp:
        # bm25 模式：走稀疏，稠密不该被问到
        lex = LexicalStore(Path(tmp))
        lex.add_documents(docs)
        dense = FakeDense(docs, fail_on_search=True)  # 一旦被问就抛异常
        r = KnowledgeRetriever(lexical_store=lex, dense_store=dense, mode=RetrievalMode.BM25)
        hits = r.search("默认端口是多少", k=2)
        check("bm25 模式不触达稠密路", len(hits) > 0 and hits[0]["metadata"]["source"] == "a.md")
        check("bm25 命中不算降级", r.degraded is False)

        # bm25 零命中 → 回落稠密。
        # 注意查询必须用语料里确定不存在的字符：中文 unigram 会让
        # 「量子纠缠」的「量」命中「向量」，那样根本走不到回落分支。
        NO_HIT_QUERY = "zzqxvw jjkk"
        check("前置条件：稀疏路确实零命中", lex.search(NO_HIT_QUERY, k=2) == [])

        lex2 = LexicalStore(Path(tmp) / "2")
        lex2.add_documents(docs)
        dense2 = FakeDense(docs)
        r2 = KnowledgeRetriever(lexical_store=lex2, dense_store=dense2, mode=RetrievalMode.BM25)
        hits2 = r2.search(NO_HIT_QUERY, k=2)
        check("零命中回落稠密", len(hits2) > 0, str(hits2))
        check("回落触发降级标记", r2.degraded is True)

        # 稠密也不可用 → 空列表
        r3 = KnowledgeRetriever(
            lexical_store=lex2, dense_store=FakeDense(docs, fail_on_search=True),
            mode=RetrievalMode.BM25,
        )
        check("两路都不可用返回空", r3.search(NO_HIT_QUERY, k=2) == [])

        # dense 模式：只走稠密
        lex3 = LexicalStore(Path(tmp) / "3")
        dense4 = FakeDense(docs)
        r4 = KnowledgeRetriever(lexical_store=lex3, dense_store=dense4, mode=RetrievalMode.DENSE)
        check("dense 模式 is_ready 看稠密", r4.is_ready is True)
        check("dense 模式有结果", len(r4.search("端口", k=2)) > 0)

        # hybrid 模式
        lex5 = LexicalStore(Path(tmp) / "5")
        lex5.add_documents(docs)
        r5 = KnowledgeRetriever(
            lexical_store=lex5, dense_store=FakeDense(docs), mode=RetrievalMode.HYBRID
        )
        out = r5.search("向量索引 FAISS", k=2)
        check("hybrid 有结果", len(out) > 0)
        check("hybrid 标记 retrieved_by", "retrieved_by" in out[0]["metadata"])


def test_write_and_fault_tolerance() -> None:
    print("\n[写入与容错]")
    docs = as_dicts([("服务默认监听端口为 8000。", {"source": "a.md"})])
    with tempfile.TemporaryDirectory() as tmp:
        lex = LexicalStore(Path(tmp))
        dense = FakeDense()
        dense.fail_on_add = True  # 稠密路写失败
        r = KnowledgeRetriever(lexical_store=lex, dense_store=dense, mode=RetrievalMode.BM25)
        n = r.add_documents(docs)
        check("稠密失败不影响稀疏入库", n == 1, f"n={n}")
        # degraded 是「上一次操作」的状态，必须在下次 search 把标志重置之前断言
        check("稠密失败标记降级", r.degraded is True)
        check("稀疏仍可检索", len(r.search("端口")) > 0)

        # 稀疏路自身异常也不能冒泡
        r2 = KnowledgeRetriever(lexical_store=lex, dense_store=FakeDense(docs),
                                mode=RetrievalMode.BM25)
        original = r2.lexical_store.search
        r2.lexical_store.search = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        out = r2.search("端口", k=2)
        check("稀疏异常被吞掉并回落", isinstance(out, list))
        r2.lexical_store.search = original


def test_migration_from_dense() -> None:
    print("\n[从 FAISS 迁移]")
    docs = as_dicts([("服务默认监听端口为 8000。", {"source": "a.md"})])
    with tempfile.TemporaryDirectory() as tmp:
        lex = LexicalStore(Path(tmp))

        class HasDocstore(FakeDense):
            def __init__(self, docs):
                super().__init__(docs)
                self._vector_store = type(
                    "S", (), {"docstore": type("D", (), {"_dict": {i: d for i, d in enumerate(docs)}})}
                )()

        dense = HasDocstore(docs)
        r = KnowledgeRetriever(lexical_store=lex, dense_store=dense, mode=RetrievalMode.BM25)
        # 手动触发迁移路径
        r._migrate_from_dense()
        check("迁移后稀疏库有内容", lex.n_docs == 1, f"n={lex.n_docs}")
        check("迁移后已落盘", lex.index_path.exists())
        check("迁移后可检索", len(lex.search("端口", k=1)) == 1)


def test_mode_parse() -> None:
    print("\n[模式解析]")
    check("dense", RetrievalMode.parse("dense") is RetrievalMode.DENSE)
    check("BM25 大小写不敏感", RetrievalMode.parse("BM25") is RetrievalMode.BM25)
    check("hybrid", RetrievalMode.parse("hybrid") is RetrievalMode.HYBRID)
    check("非法值回落 bm25", RetrievalMode.parse("nonsense") is RetrievalMode.BM25)
    check("空值回落 bm25", RetrievalMode.parse(None) is RetrievalMode.BM25)


if __name__ == "__main__":
    test_lexical_persistence()
    test_lexical_edge_cases()
    test_mode_routing()
    test_write_and_fault_tolerance()
    test_migration_from_dense()
    test_mode_parse()
    print(f"\n通过 {PASS} 项，失败 {FAIL} 项")
    sys.exit(1 if FAIL else 0)

"""检索增强层的单元测试（零第三方依赖，可直接运行）。

    python tests/test_retrieval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.local_rag.core.retrieval import (
    BM25Config,
    BM25Index,
    HybridConfig,
    HybridRetriever,
    MMRReranker,
    NoOpReranker,
    reciprocal_rank_fusion,
)
from frontend.local_rag.core.retrieval.tokenizer import tokenize

PASSED = 0
FAILED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [ok]   {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name} {detail}")


# ---------------------------------------------------------------- tokenizer
def test_tokenizer() -> None:
    print("\n[tokenizer]")
    # 全角 FAISS 与半角 FAISS 必须归一化成同一个 token
    check("NFKC 归一：全角=半角", tokenize("ＦＡＩＳＳ") == tokenize("faiss"))
    # 中文同时产出 unigram 和 bigram
    tokens = tokenize("向量索引")
    check("中文产 unigram", "向" in tokens and "量" in tokens)
    check("中文产 bigram", "向量" in tokens and "量索" in tokens and "索引" in tokens)
    # 倒序词的 bigram 不同——这是 bigram 优于字符集合的地方
    check("词序敏感", "上海" in tokenize("上海") and "海上" in tokenize("海上"))


# --------------------------------------------------------------------- bm25
def test_bm25() -> None:
    print("\n[bm25]")
    docs = [
        "后端框架使用 FastAPI，默认监听端口为 8000。",
        "向量索引使用 FAISS，并持久化到本地磁盘。",
        "今天天气不错，适合出去散步。",
    ]
    index = BM25Index().build(docs)
    check("索引已构建", index.is_ready and index.n_docs == 3)

    hits = index.search("默认监听端口", k=3)
    check("端口问题命中第 0 篇", hits[0].index == 0, f"got {hits[0].index}")

    hits = index.search("向量索引 FAISS", k=3)
    check("FAISS 问题命中第 1 篇", hits[0].index == 1, f"got {hits[0].index}")

    # 完全不相关的 query：语料里没出现过的词，得分全为 0，不应返回
    # （必须用纯拉丁词——中文单字在 500 字语料里几乎必然撞上，这是 unigram 的固有特性）
    check("无命中返回空列表", index.search("kubernetes prometheus grafana", k=3) == [])

    # 长度归一：同一份索引里，命中同样的词，短文档得分更高。
    # 注意必须放在多文档索引里测——单文档索引的 avgdl 恒等于自身长度，看不出差异。
    mixed = BM25Index().build(["端口 8000", "端口 8000" + "无关内容" * 200])
    scores = mixed.scores("端口")
    check("长度归一生效：短文档得分更高", scores[0] > scores[1], f"{scores[0]:.4f} vs {scores[1]:.4f}")

    # k1=0 时退化为布尔匹配：同一文档内重复出现不再加分
    bool_like = BM25Index(BM25Config(k1=0.0)).build(["端口 端口 端口"]).search("端口")
    tf_sat = BM25Index(BM25Config(k1=1.5)).build(["端口 端口 端口"]).search("端口")
    check("k1 控制词频饱和", bool_like[0].score < tf_sat[0].score)


# ---------------------------------------------------------------------- rrf
def test_rrf() -> None:
    print("\n[rrf]")
    # 经典场景：文档 A 两路都靠前，B 只在一路第一，另一路没召回
    fused = reciprocal_rank_fusion([[0, 1, 2], [1, 0, 3]])
    top = fused[0][0]
    check("两路都靠前者胜出", top == 0, f"got {top}")

    # 只被一路召回的文档也要出现在结果里（不是取交集）
    indices = {i for i, _ in fused}
    check("融合是并集不是交集", indices == {0, 1, 2, 3}, f"got {indices}")

    # 权重：把第一路权重调大，它的第一名应该反超
    weighted = reciprocal_rank_fusion([[1, 0], [0, 1]], weights=[5.0, 1.0])
    check("权重可改变排序", weighted[0][0] == 1, f"got {weighted[0][0]}")

    # 确定性：相同输入必须得到相同输出（评测可复现的前提）
    check(
        "结果确定可复现",
        reciprocal_rank_fusion([[2, 1], [1, 2]]) == reciprocal_rank_fusion([[2, 1], [1, 2]]),
    )
    check("空输入返回空", reciprocal_rank_fusion([]) == [])


# ------------------------------------------------------------------ hybrid
def test_hybrid() -> None:
    print("\n[hybrid]")
    chunks = [
        {"content": "后端框架使用 FastAPI，默认监听端口为 8000。", "metadata": {"source": "a.md"}},
        {"content": "向量索引使用 FAISS，并持久化到本地磁盘。", "metadata": {"source": "b.md"}},
        {"content": "系统使用 Sentence-Transformers 生成文本向量。", "metadata": {"source": "c.md"}},
    ]

    # 稠密路故意只认「端口」这一个语义，模拟与 BM25 不同的召回偏好
    def dense_search(query: str, k: int):
        if "端口" in query:
            return [chunks[1], chunks[0]]  # 稠密路排错：把 b 排在第一
        return []

    retriever = HybridRetriever(dense_search=dense_search).index(chunks)
    results = retriever.search("默认监听端口", k=2)
    check("混合检索有结果", len(results) == 2)
    check(
        "BM25 纠错：端口文档回到第一",
        "端口" in results[0]["content"],
        f"got {results[0]['content'][:20]}",
    )
    check("元数据带召回来源", "retrieved_by" in results[0]["metadata"])

    # 降级：稠密路抛异常，必须仍然返回结果而不是崩掉
    def broken_dense(query: str, k: int):
        raise RuntimeError("faiss down")

    degraded = HybridRetriever(dense_search=broken_dense).index(chunks)
    out = degraded.search("向量索引 FAISS", k=2)
    check("稠密路故障时不崩", len(out) > 0)
    check("降级被标记", degraded.degraded is True)

    # 两路都关掉 -> 空列表，与「知识库为空」行为一致
    empty = HybridRetriever(
        dense_search=dense_search, config=HybridConfig(enable_bm25=False, enable_dense=False)
    ).index(chunks)
    check("两路都禁用返回空", empty.search("端口", k=2) == [])

    # 未建索引时不应抛异常
    check("未建索引返回空", HybridRetriever(dense_search=dense_search).search("x", k=2) == [])


# -------------------------------------------------------------------- rerank
def test_rerank() -> None:
    print("\n[rerank]")
    # 第 2 条是第 1 条的近似重复，第 3 条内容不同但同样相关
    candidates = [
        "端口配置为 8000",
        "端口配置为 8000（补充）",
        "端口配置为 8080 的前端代理",
    ]

    mmr = MMRReranker(lambda_param=0.3)
    scores = mmr.rerank("端口配置", candidates)
    order = sorted(range(len(candidates)), key=lambda i: -scores[i])
    check(
        "MMR 提升多样性：非重复片段前移",
        order.index(2) < order.index(1),
        f"order={order}",
    )

    # 上游给了真实相关性分时，MMR 必须优先用它而不是位置
    scored = mmr.rerank("端口配置", candidates, [1.0, 0.95, 2.0])
    scored_order = sorted(range(3), key=lambda i: -scored[i])
    check("MMR 采用上游相关性分", scored_order[0] == 2, f"order={scored_order}")

    noop = NoOpReranker().rerank("端口配置", candidates)
    check("NoOp 严格保持传入顺序", list(noop) == sorted(noop, reverse=True))

    # 重排不能改变候选数量
    check("重排不改变候选数量", len(scores) == len(candidates) == 3)


def main() -> None:
    test_tokenizer()
    test_bm25()
    test_rrf()
    test_hybrid()
    test_rerank()
    print("\n" + "=" * 46)
    print(f"通过 {PASSED} 项，失败 {FAILED} 项")
    print("=" * 46)
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()

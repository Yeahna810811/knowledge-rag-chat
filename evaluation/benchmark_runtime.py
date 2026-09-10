"""检索落地的运行时对比：入库耗时、查询延迟、冷启动成本。

评测只回答"哪个更准"，这个脚本回答"哪个更能上生产"。
两者结论可能不同——比如混合检索准确率没优势，但如果它不增加延迟，
那作为可选项就有意义；反过来 BM25 更准且更快，那就是 doubly worth it。

对比项：
1. 冷启动：稠密路要加载 bge-small-zh 权重 + 分词器，稀疏路零模型；
2. 入库：稠密要跑 BERT 前向（每 chunk 一次），稀疏只做分词；
3. 查询：稠密一次前向 + 向量点积，稀疏走倒排表；
4. 内存：稠密要常驻模型 + 全部向量，稀疏只要倒排表。

稠密路用 evaluation/local_bge_embedder.py（numpy 手写 BERT，加载真实权重），
它与生产用的 bge-small-zh 是同一份权重、同一套池化方式，只是推理实现不同，
所以**入库/查询的相对量级可比**，绝对值会偏慢（生产有 torch 加速）。

用法：
    python evaluation/benchmark_runtime.py
    python evaluation/benchmark_runtime.py --queries 200
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate_retrieval_ab import build_chunks  # noqa: E402
from frontend.local_rag.core.retrieval.bm25 import BM25Index  # noqa: E402
from frontend.local_rag.core.retrieval.lexical_store import LexicalStore  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.2f} ms"


def benchmark(chunks: list[dict], n_queries: int) -> dict:
    texts = [c["content"] for c in chunks]
    queries = [f"{t[:18]}" for t in texts[:n_queries]]

    print(f"\n语料: {len(chunks)} chunk / {sum(len(t) for t in texts)} 字符")
    print(f"查询: {len(queries)} 条\n")

    result: dict = {"chunks": len(chunks), "queries": len(queries)}

    # ---------------- 稀疏路 ----------------
    t0 = time.perf_counter()
    store = LexicalStore(Path(PROJECT_ROOT) / "evaluation" / ".bench_bm25")
    store.add_documents(chunks)
    sparse_ingest = time.perf_counter() - t0
    result["sparse_ingest_s"] = sparse_ingest

    t0 = time.perf_counter()
    for q in queries:
        store.search(q, k=4)
    sparse_query_total = time.perf_counter() - t0
    result["sparse_query_ms"] = sparse_query_total / len(queries) * 1000

    # 持久化 + 重载（模拟服务重启）
    t0 = time.perf_counter()
    store.save()
    fresh = LexicalStore(Path(PROJECT_ROOT) / "evaluation" / ".bench_bm25")
    fresh.load()
    sparse_reload = time.perf_counter() - t0
    result["sparse_reload_s"] = sparse_reload

    # ---------------- 稠密路 ----------------
    dense: dict = {}
    try:
        from evaluation.local_bge_embedder import LocalBgeEmbedder

        t0 = time.perf_counter()
        embedder = LocalBgeEmbedder()
        dense["model_load_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        matrix = embedder.embed_documents(texts)
        dense["ingest_s"] = time.perf_counter() - t0

        import numpy as np

        mat = __import__("numpy").array(matrix, dtype="float32")

        t0 = time.perf_counter()
        for q in queries:
            v = __import__("numpy").array(embedder.embed_query(q), dtype="float32")
            mat @ v
        dense["query_ms"] = (time.perf_counter() - t0) / len(queries) * 1000

        result["dense"] = dense
    except Exception as exc:  # 稠密路跑不了也要给出稀疏路的数字
        print(f"  [warn] 稠密路不可用，跳过对比: {type(exc).__name__}: {exc}")
        result["dense"] = None

    return result


def report(r: dict) -> None:
    print("=" * 62)
    print(f"{'指标':<26}{'BM25 稀疏路':>18}{'稠密向量路':>18}")
    print("-" * 62)

    d = r.get("dense") or {}
    rows = [
        ("冷启动（加载模型）", "0 ms（无模型）", fmt_ms(d["model_load_s"]) if d.get("model_load_s") else "n/a"),
        (f"入库 {r['chunks']} 个 chunk", fmt_ms(r["sparse_ingest_s"]), fmt_ms(d["ingest_s"]) if d.get("ingest_s") else "n/a"),
        ("单条查询延迟", f"{r['sparse_query_ms']:.2f} ms", f"{d['query_ms']:.2f} ms" if d.get("query_ms") else "n/a"),
        ("重启后重建索引", fmt_ms(r["sparse_reload_s"]), "n/a（读文件+反序列化）"),
    ]
    for name, a, b in rows:
        print(f"{name:<26}{a:>18}{b:>18}")

    print("-" * 62)
    if d.get("ingest_s"):
        print(f"入库速度倍数：稠密 / 稀疏 = {d['ingest_s'] / max(r['sparse_ingest_s'], 1e-9):.1f}x")
    if d.get("query_ms"):
        print(f"查询速度倍数：稠密 / 稀疏 = {d['query_ms'] / max(r['sparse_query_ms'], 1e-9):.1f}x")
    if d.get("model_load_s"):
        print(f"冷启动节省：{d['model_load_s']:.2f} s（稀疏路不需要加载 embedding 模型）")
    print("=" * 62)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument(
        "--corpus",
        nargs="+",
        default=None,
        help="默认使用 corpus + extended_noise 全量语料",
    )
    args = ap.parse_args()

    if args.corpus:
        paths = [Path(p) for p in args.corpus]
    else:
        paths = sorted((PROJECT_ROOT / "evaluation" / "corpus").glob("*.md")) + sorted(
            (PROJECT_ROOT / "evaluation" / "extended_noise").glob("*.md")
        )
    chunks = build_chunks(paths, 500, 50)

    r = benchmark(chunks, args.queries)
    report(r)


if __name__ == "__main__":
    main()

"""查询改写 A/B：验证「在检索前改写查询」能否补上词汇不匹配的短板。

要回答的问题（承接 EVALUATION_AUDIT.md 第四节）：
口语化问法下 BM25 的 MRR 只有 0.510，稠密路 0.313——两路都塌了。
既然问题出在「用户用词 ≠ 文档用词」，那么在检索前改写查询应该有效。
本脚本用同一份语料、同一个 BM25 索引，只切换改写策略，量化收益。

对照策略：
    bm25                不改写（基线）
    bm25+prf(w=0.3)     伪相关反馈，扩展词权重 0.3
    bm25+prf(w=0.5)     权重 0.5，用来验证「权重越高越好吗」
    bm25+prf(w=0.8)     权重 0.8，逼近「扩展词与原查询平权」
    bm25+dense          原查询走 BM25，零命中时回落稠密（现有生产兜底路径）

判读口径与主评测一致：bigram 包含度判定相关性 + 配对 bootstrap 显著性检验。
只看绝对值的涨跌没有意义——1 道题 = 1.25 个百分点。

用法：
    python evaluation/evaluate_query_rewrite.py
    python evaluation/evaluate_query_rewrite.py --dataset evaluation/eval_dataset_oral_v3.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_retrieval_ab import (  # noqa: E402
    build_chunks,
    first_relevant_rank,
    token_containment,
)
from evaluation.significance_test import (  # noqa: E402
    paired_bootstrap,
    sign_test,
)
from frontend.local_rag.core.retrieval.lexical_store import LexicalStore  # noqa: E402
from frontend.local_rag.core.retrieval.query_rewrite import (  # noqa: E402
    PRFConfig,
    PseudoRelevanceFeedbackRewriter,
)

POOL = 20          # 召回池：先取 20 个候选，再按 top_k 判定命中
TOP_K = 4          # 生产配置 retrieval_top_k
BOOTSTRAP_ROUNDS = 10000
SEED = 20260910

CORPUS = sorted((PROJECT_ROOT / "evaluation" / "corpus").glob("*.md")) + sorted(
    (PROJECT_ROOT / "evaluation" / "extended_noise").glob("*.md")
)
PRF_WEIGHTS = (0.3, 0.5, 0.8)


def load_answerable(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [r for r in rows if r["answerable"] and str(r.get("evidence", "")).strip()]


def build_store() -> tuple[LexicalStore, list[dict]]:
    chunks = build_chunks(CORPUS, 500, 50)
    store = LexicalStore(index_dir=Path("/tmp/rag_qr_index"))
    store.add_documents(
        [{"content": c["content"], "metadata": c["metadata"]} for c in chunks]
    )
    return store, chunks


def make_rank_fn(store: LexicalStore, chunks: list[dict], weight: float | None):
    """weight=None 表示不改写（基线）。"""
    content_to_index = {c["content"]: i for i, c in enumerate(chunks)}
    rewriter = None if weight is None else PseudoRelevanceFeedbackRewriter(
        PRFConfig(expansion_weight=weight)
    )

    def rank(query: str) -> list[int]:
        if rewriter is None:
            hits = store.search(query, k=POOL)
        else:
            result = rewriter.rewrite(
                query, index=store.index, metadata_of=store.metadata_of
            )
            if result.is_effective:
                hits = store.search_weighted(
                    query, result.expanded_terms, weight, k=POOL
                )
            else:
                hits = store.search(query, k=POOL)
        return [content_to_index[h["content"]] for h in hits if h["content"] in content_to_index]

    return rank


def evaluate(rank_fn, chunks: list[dict], items: list[dict]) -> tuple[dict, list[float]]:
    hits_at = {1: 0, 3: 0, 5: 0}
    rr: list[float] = []
    for item in items:
        ids = rank_fn(item["question"])[:POOL]
        rank = first_relevant_rank(
            [chunks[i] for i in ids], item, token_containment
        )
        for k in hits_at:
            if rank and rank <= k:
                hits_at[k] += 1
        rr.append(1.0 / rank if rank else 0.0)
    n = len(items)
    return (
        {
            "Hit@1": hits_at[1] / n,
            "Hit@3": hits_at[3] / n,
            "Hit@5": hits_at[5] / n,
            "MRR": sum(rr) / n,
        },
        rr,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "evaluation" / "eval_dataset_oral_v3.jsonl")
    ap.add_argument("--rounds", type=int, default=BOOTSTRAP_ROUNDS)
    args = ap.parse_args()

    items = load_answerable(args.dataset)
    store, chunks = build_store()

    print("=" * 78)
    print("查询改写 A/B  |  数据集:", args.dataset.name)
    print(f"可答题 {len(items)} 题  |  语料 {len(chunks)} chunk  |  top_k={TOP_K}")
    print("=" * 78)

    results: dict[str, list[float]] = {}
    print(f"\n{'策略':<20}{'Hit@1':>8}{'Hit@3':>8}{'Hit@5':>8}{'MRR':>9}")
    print("-" * 78)

    for label, weight in [("bm25", None)] + [(f"bm25+prf(w={w})", w) for w in PRF_WEIGHTS]:
        metrics, rr = evaluate(make_rank_fn(store, chunks, weight), chunks, items)
        results[label] = rr
        print(
            f"{label:<20}{metrics['Hit@1']:>8.3f}{metrics['Hit@3']:>8.3f}"
            f"{metrics['Hit@5']:>8.3f}{metrics['MRR']:>9.4f}"
        )

    print("\n" + "=" * 78)
    print("配对显著性检验（对照：bm25 不改写）")
    print("-" * 78)
    print(f"{'对比策略':<20}{'ΔMRR':>10}{'95% CI':>24}{'p(boot)':>10}{'提升/下降':>12}{'p(sign)':>9}")
    print("-" * 78)
    base = results["bm25"]
    for label in results:
        if label == "bm25":
            continue
        point, lo, hi, p = paired_bootstrap(results[label], base, args.rounds, SEED)
        up, dn, ps = sign_test(results[label], base)
        verdict = "显著" if (lo > 0 or hi < 0) else "不显著"
        print(
            f"{label:<20}{point:>+10.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>24}"
            f"{p:>10.3f}{f'{up}/{dn}':>12}{ps:>9.3f}  {verdict}"
        )

    print("\n判读：CI 不跨 0 才叫显著；扩展词权重越高≠越好，看 CI 位置。")


if __name__ == "__main__":
    main()

"""对检索策略差异做配对显著性检验（bootstrap + 符号检验）。

为什么需要这个：
评测集只有 40 道可答题，1 道题 = 2.5 个百分点。策略之间几个百分点的差距
极可能只是噪声。不检验就下结论，等于拿随机波动当成果——这恰恰是
上一轮「TF-IDF 代理得出混合检索有效」翻车的根因。

方法：
1. 配对 bootstrap：对「题目」这一层做有放回重采样（不是对文档重采样），
   重算两策略的 MRR 差值，得到差值的置信区间；
2. 符号检验：统计「提升题数 / 下降题数」在二项分布下的 p 值。

判读标准：
- 差值 95% CI 完全落在 0 的同一侧 → 差异可信；
- 否则 → 差异不显著，不应作为项目结论。

用法：
    python evaluation/significance_test.py
    python evaluation/significance_test.py --dataset evaluation/eval_dataset_oral.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate_retrieval_ab import (  # noqa: E402
    build_chunks,
    first_relevant_rank,
    token_containment,
)
from evaluation.local_bge_embedder import LocalBgeEmbedder  # noqa: E402
from frontend.local_rag.core.retrieval.bm25 import BM25Index  # noqa: E402
from frontend.local_rag.core.retrieval.fusion import (  # noqa: E402
    reciprocal_rank_fusion,
)

import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_CORPUS = sorted((PROJECT_ROOT / "evaluation" / "corpus").glob("*.md")) + sorted(
    (PROJECT_ROOT / "evaluation" / "extended_noise").glob("*.md")
)
BOOTSTRAP_ROUNDS = 10000
SEED = 20260910


def load_answerable(dataset_path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [r for r in rows if r["answerable"] and str(r.get("evidence", "")).strip()]


def build_rank_fns(chunks: list[dict], top_pool: int = 20):
    """返回 {策略名: query -> doc_id 列表}。"""
    texts = [c["content"] for c in chunks]
    emb = LocalBgeEmbedder()
    matrix = np.array(emb.embed_documents(texts), dtype=np.float32)
    bm25 = BM25Index().build(texts)

    def dense(q: str):
        v = np.array(emb.embed_query(q), dtype=np.float32)
        return [int(i) for i in np.argsort(-(matrix @ v))[:top_pool]]

    def lexical(q: str):
        return [int(h.index) for h in bm25.search(q, k=top_pool)]

    def hybrid_factory(bm25_weight: float):
        def _fn(q: str):
            fused = reciprocal_rank_fusion(
                [lexical(q), dense(q)], weights=[bm25_weight, 1.0]
            )
            return [int(i) for i, _ in fused]

        return _fn

    return {"dense": dense, "bm25": lexical, "_hybrid": hybrid_factory}


def per_question_rr(rank_fn, chunks: list[dict], items: list[dict], k: int = 5) -> list[float]:
    """每题的 reciprocal rank，未命中记 0。"""
    out = []
    for item in items:
        ids = rank_fn(item["question"])[:k]
        rank = first_relevant_rank([chunks[i] for i in ids], item, token_containment)
        out.append(1.0 / rank if rank else 0.0)
    return out


def paired_bootstrap(a: list[float], b: list[float], rounds: int, seed: int):
    """a - b 的 bootstrap 分布。返回 (点估计, CI下界, CI上界, p_two_sided)。"""
    rng = random.Random(seed)
    n = len(a)
    diffs = sorted(a[i] - b[i] for i in range(n))
    point = sum(diffs) / n

    boot = []
    for _ in range(rounds):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boot.append(s / n)
    boot.sort()
    lo = boot[int(0.025 * rounds)]
    hi = boot[int(0.975 * rounds)]

    # 双尾 p：bootstrap 分布跨过 0 的比例
    frac_pos = sum(1 for x in boot if x > 0) / rounds
    frac_neg = sum(1 for x in boot if x < 0) / rounds
    p = 2.0 * min(frac_pos, frac_neg)
    return point, lo, hi, min(p, 1.0)


def sign_test(a: list[float], b: list[float]) -> tuple[int, int, float]:
    """符号检验：只看方向不看幅度，对小样本更稳健。"""
    up = sum(1 for x, y in zip(a, b) if x > y)
    dn = sum(1 for x, y in zip(a, b) if x < y)
    n = up + dn
    if n == 0:
        return up, dn, 1.0
    # 二项分布双尾精确 p（n 小，直接算）
    k = max(up, dn)
    tail = sum(_binom(n, i) for i in range(k, n + 1)) / (2.0**n)
    return up, dn, min(2.0 * tail, 1.0)


def _binom(n: int, k: int) -> int:
    from math import comb

    return comb(n, k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="evaluation/eval_dataset.jsonl")
    ap.add_argument("--rounds", type=int, default=BOOTSTRAP_ROUNDS)
    args = ap.parse_args()

    dataset_path = PROJECT_ROOT / args.dataset
    items = load_answerable(dataset_path)
    chunks = build_chunks(FULL_CORPUS, 500, 50)
    fns = build_rank_fns(chunks)

    print("=" * 74)
    print(f"配对显著性检验  |  数据集: {args.dataset}")
    print(f"可答题 {len(items)} 题  |  语料 {len(chunks)} chunks  |  bootstrap {args.rounds} 轮")
    print("=" * 74)

    strategies = {
        "dense": fns["dense"],
        "bm25": fns["bm25"],
        "hybrid(1.0:1)": fns["_hybrid"](1.0),
        "hybrid(1.5:1)": fns["_hybrid"](1.5),
        "hybrid(2.0:1)": fns["_hybrid"](2.0),
    }

    rrs = {name: per_question_rr(fn, chunks, items) for name, fn in strategies.items()}

    print(f"\n{'策略':<16}{'MRR':>8}{'Hit@1':>8}   (Hit@1 = 排第1的题数 / 总题数)")
    print("-" * 52)
    for name, rr in rrs.items():
        mrr = sum(rr) / len(rr)
        hit1 = sum(1 for x in rr if x == 1.0) / len(rr)
        print(f"{name:<16}{mrr:>8.4f}{hit1:>8.3f}")

    print(f"\n对照基准: bm25 单路")
    print("-" * 74)
    print(f"{'对比策略':<16}{'ΔMRR':>9}{'95% CI':>22}{'p(boot)':>10}{'提升/下降':>11}{'p(sign)':>9}")
    print("-" * 74)
    for name, rr in rrs.items():
        if name == "bm25":
            continue
        point, lo, hi, p_boot = paired_bootstrap(rr, rrs["bm25"], args.rounds, SEED)
        up, dn, p_sign = sign_test(rr, rrs["bm25"])
        ci = f"[{lo:+.4f}, {hi:+.4f}]"
        verdict = "显著" if hi < 0 or lo > 0 else "不显著"
        print(
            f"{name:<16}{point:>+9.4f}{ci:>22}{p_boot:>10.3f}"
            f"{f'{up}/{dn}':>11}{p_sign:>9.3f}  {verdict}"
        )

    print("\n判读：ΔMRR 的 95% 置信区间若跨越 0，说明差距可能是抽样噪声，不应作为项目结论。")


if __name__ == "__main__":
    main()

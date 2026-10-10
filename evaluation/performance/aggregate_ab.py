"""把配对 A/B 的 30 个原始 CSV 汇总成一张中位数表。

为什么需要这一步：
单轮压测在本机噪声太大——n=20 时 ask 模式（一个根本没被改动的代码路径）
5 轮跑出来的 P50 峰谷差就有 19~35%；n=100 能压到 1~15%，但个别格子仍会
出现"命中缓存比未命中还慢"这种明显违背因果的结果。

所以任何**单轮**数字都不能直接写进 README。这里对 5 轮交错测量取中位数，
输出 `ab_summary.csv`，README 的性能表格一律从这张表来，
原始 30 个文件保留在 `ab_fix_vs_baseline/` 供复核。

用法：
    python evaluation/performance/aggregate_ab.py
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
SRC = RESULTS / "ab_fix_vs_baseline"
OUT = RESULTS / "ab_summary.csv"

# 场景 -> (文件名模板, 该场景要统计的指标)
SCENARIOS = {
    "main": ("{side}_rep{i}.csv", False),
    "cache_miss": ("{side}_cache_miss_rep{i}.csv", True),
    "cache_hit": ("{side}_cache_hit_rep{i}.csv", True),
}

METRICS = [
    "throughput_rps",
    "mean_ms",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "ttft_mean_ms",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "ttft_p99_ms",
    "retrieval_mean_ms",
    "retrieval_p50_ms",
    "error_rate",
]

FIELDS = ["scenario", "side", "mode", "concurrency", "n_rounds", "requests_per_round"] + METRICS


def median_of_rounds(rows: list[dict], col: str) -> str:
    values = []
    for r in rows:
        v = r.get(col, "")
        if v not in ("", None):
            try:
                values.append(float(v))
            except ValueError:
                pass
    if not values:
        return ""
    return f"{statistics.median(values):.2f}"


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"缺少原始数据目录：{SRC}\n先跑交错 A/B 并把 CSV 放进去。")

    out_rows: list[dict] = []

    for scenario, (template, _) in SCENARIOS.items():
        # 每个 (side, mode, concurrency) 收集 5 轮
        buckets: dict[tuple[str, str, int], list[dict]] = {}
        for side in ("fixed", "baseline"):
            i = 1
            while True:
                path = SRC / template.format(side=side, i=i)
                if not path.exists():
                    break
                for r in csv.DictReader(open(path, encoding="utf-8")):
                    key = (side, r["mode"], int(r["concurrency"]))
                    buckets.setdefault(key, []).append(r)
                i += 1

        for (side, mode, c), rows in sorted(buckets.items()):
            rec = {
                "scenario": scenario,
                "side": side,
                "mode": mode,
                "concurrency": c,
                "n_rounds": len(rows),
                "requests_per_round": rows[0].get("requests", ""),
            }
            for m in METRICS:
                rec[m] = median_of_rounds(rows, m)
            out_rows.append(rec)

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out_rows)

    print(f"[ok] {OUT}  ({len(out_rows)} 行)")

    # 顺手打出修复 vs 基线的 delta，方便直接抄进 README
    print("\n=== P50 delta（修复 vs 基线，取中位数，负=更快）===")
    idx = {(r["scenario"], r["side"], r["mode"], int(r["concurrency"])): r for r in out_rows}
    for scenario in SCENARIOS:
        for (s, side, mode, c), r in idx.items():
            if s != scenario or side != "fixed":
                continue
            b = idx.get((scenario, "baseline", mode, c))
            if not b or not r["p50_ms"] or not b["p50_ms"]:
                continue
            f, v = float(r["p50_ms"]), float(b["p50_ms"])
            print(f"  {scenario:<10} {mode:<7} c={c:<3} {f:8.2f} vs {v:8.2f}  {(f-v)/v*100:+6.1f}%")


if __name__ == "__main__":
    main()

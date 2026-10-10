"""从 performance CSV 生成图表。

图表只能由真实 CSV 数据驱动：这里不做任何平滑、不外推、不补点。
某个分位数在 CSV 里是空值，图上就留空，不允许拿均值顶替。

用法：
    python evaluation/performance/make_charts.py
    python evaluation/performance/make_charts.py \
        --results-dir evaluation/performance/results
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # 无 GUI 环境必须切 Agg，否则 import pyplot 会去找 display
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS", "PingFang SC", "Heiti SC", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str | None) -> float | None:
    if value in (None, "", "-"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def pick(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    out = [r for r in rows if r["mode"] == mode]
    out.sort(key=lambda r: int(r["concurrency"]))
    return out


def line_chart(
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    series: list[tuple[str, list[float], list[float | None], str]],
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, xs, ys, marker in series:
        ax.plot(xs, ys, marker=marker, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def bar_chart(
    path: Path,
    title: str,
    ylabel: str,
    labels: list[str],
    groups: list[tuple[str, list[float | None]]],
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / max(1, len(groups))
    positions = list(range(len(labels)))
    for index, (name, values) in enumerate(groups):
        offset = [p + index * width for p in positions]
        ax.bar(offset, values, width=width, label=name)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks([p + 0.4 - width / 2 for p in positions])
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "performance" / "results",
    )
    args = parser.parse_args()
    results = args.results_dir
    results.mkdir(parents=True, exist_ok=True)

    summary = read_csv(results / "performance_summary.csv")
    if not summary:
        print(f"[warn] 没有找到 {results / 'performance_summary.csv'}，先跑 run_load_test.py")
        return 1

    made: list[Path] = []

    # ---- 1/2. 延迟分位数 vs 并发
    for mode in ("ask", "stream"):
        rows = pick(summary, mode)
        if not rows:
            continue
        xs = [int(r["concurrency"]) for r in rows]
        series = [
            ("P50", xs, [to_float(r["p50_ms"]) for r in rows], "o"),
            ("P95", xs, [to_float(r["p95_ms"]) for r in rows], "s"),
            ("P99", xs, [to_float(r["p99_ms"]) for r in rows], "^"),
        ]
        target = results / f"latency_percentiles_{mode}.png"
        line_chart(
            target,
            f"{mode.upper()} latency vs concurrency (FakeLLM)",
            "concurrency",
            "latency (ms)",
            series,
        )
        made.append(target)

    # ---- 3. 吞吐 vs 并发
    rows_all = sorted(summary, key=lambda r: (r["mode"], int(r["concurrency"])))
    xs_ask = [int(r["concurrency"]) for r in pick(summary, "ask")]
    xs_stream = [int(r["concurrency"]) for r in pick(summary, "stream")]
    series = []
    if xs_ask:
        series.append(
            ("ask", xs_ask, [to_float(r["throughput_rps"]) for r in pick(summary, "ask")], "o")
        )
    if xs_stream:
        series.append(
            (
                "stream",
                xs_stream,
                [to_float(r["throughput_rps"]) for r in pick(summary, "stream")],
                "s",
            )
        )
    target = results / "throughput_vs_concurrency.png"
    line_chart(
        target,
        "Throughput vs concurrency (FakeLLM)",
        "concurrency",
        "requests / second",
        series,
    )
    made.append(target)

    # ---- 4. 错误率 vs 并发
    target = results / "error_rate_vs_concurrency.png"
    line_chart(
        target,
        "Error rate vs concurrency (FakeLLM)",
        "concurrency",
        "error rate",
        [
            ("ask", xs_ask, [to_float(r["error_rate"]) for r in pick(summary, "ask")], "o"),
            (
                "stream",
                xs_stream,
                [to_float(r["error_rate"]) for r in pick(summary, "stream")],
                "s",
            ),
        ],
    )
    made.append(target)

    # ---- 5. 流式 TTFT 分位数 vs 并发
    stream_rows = pick(summary, "stream")
    if stream_rows:
        xs = [int(r["concurrency"]) for r in stream_rows]
        target = results / "ttft_percentiles_stream.png"
        line_chart(
            target,
            "Streaming TTFT vs concurrency (FakeLLM)",
            "concurrency",
            "TTFT (ms)",
            [
                ("TTFT P50", xs, [to_float(r["ttft_p50_ms"]) for r in stream_rows], "o"),
                ("TTFT P95", xs, [to_float(r["ttft_p95_ms"]) for r in stream_rows], "s"),
                ("TTFT P99", xs, [to_float(r["ttft_p99_ms"]) for r in stream_rows], "^"),
            ],
        )
        made.append(target)

    # ---- 6. cache hit vs miss
    miss = read_csv(results / "cache_miss_summary.csv")
    hit = read_csv(results / "cache_hit_summary.csv")
    if miss and hit:
        miss_rows = sorted(miss, key=lambda r: int(r["concurrency"]))
        hit_rows = sorted(hit, key=lambda r: int(r["concurrency"]))
        labels = [f"c={r['concurrency']}" for r in miss_rows]
        target = results / "cache_hit_vs_miss_retrieval.png"
        bar_chart(
            target,
            "Retrieval latency: cache miss vs hit (FakeLLM, stream)",
            "retrieval mean (ms)",
            labels,
            [
                ("cache miss", [to_float(r["retrieval_mean_ms"]) for r in miss_rows]),
                ("cache hit", [to_float(r["retrieval_mean_ms"]) for r in hit_rows]),
            ],
        )
        made.append(target)

        target = results / "cache_hit_vs_miss_e2e.png"
        bar_chart(
            target,
            "End-to-end latency: cache miss vs hit (FakeLLM, stream)",
            "E2E mean (ms)",
            labels,
            [
                ("cache miss", [to_float(r["mean_ms"]) for r in miss_rows]),
                ("cache hit", [to_float(r["mean_ms"]) for r in hit_rows]),
            ],
        )
        made.append(target)

    for path in made:
        print(f"[ok] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

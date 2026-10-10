"""性能压测脚本的自检（Stage 5）。

为什么要有这个测试：
压测脚本本身是"测量工具"，如果它算错分位数，后面所有结论都是错的，
而且这种错误非常隐蔽——P99 算成均值，出来的数字看起来也"很合理"。
所以这里用**已知答案**把统计函数钉死。

刻意不做什么：
- 不起服务、不发网络请求、不调真实 LLM。CI 里跑这些既慢又烧额度，
  真实 benchmark 按 README 手动执行。
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.performance.run_load_test import (  # noqa: E402
    REQUEST_FIELDS,
    SUMMARY_FIELDS,
    RequestRecord,
    percentile,
    request_row,
    summarize,
    write_csv,
)

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


# ============================================ 分位数：必须用已知答案校验
def test_percentile_known_values() -> None:
    # 1..100 的均匀分布，线性插值法下各分位数有解析解
    samples = [float(i) for i in range(1, 101)]

    # pos = (100-1) * p
    # P50 -> pos = 49.5 -> (50 + 51) / 2 = 50.5
    check("P50 of 1..100 == 50.5", percentile(samples, 50) == 50.5,
          str(percentile(samples, 50)))
    # P95 -> pos = 94.05 -> 95 + 0.05 * (96 - 95) = 95.05
    check("P95 of 1..100 == 95.05", abs(percentile(samples, 95) - 95.05) < 1e-9,
          str(percentile(samples, 95)))
    # P99 -> pos = 98.01 -> 99 + 0.01 * 1 = 99.01
    check("P99 of 1..100 == 99.01", abs(percentile(samples, 99) - 99.01) < 1e-9,
          str(percentile(samples, 99)))
    check("P0 == min", percentile(samples, 0) == 1.0, str(percentile(samples, 0)))
    check("P100 == max", percentile(samples, 100) == 100.0,
          str(percentile(samples, 100)))


def test_percentile_not_mean() -> None:
    """P99 绝不能退化成均值或最大值。"""
    # 90 个 10ms + 10 个 1000ms：均值 109ms，P99 应落在 1000ms 那一端。
    # （不能用 99+1 的极端比例：那时 P99 恰好落在分界点上，
    #   数值会和均值撞在一起，这个断言就失去了区分能力。）
    samples = [10.0] * 90 + [1000.0] * 10
    mean = sum(samples) / len(samples)
    p99 = percentile(samples, 99)
    check("P99 不等于均值", abs(p99 - mean) > 100.0, f"p99={p99} mean={mean}")
    check("P99 接近尾端", abs(p99 - 1000.0) < 1e-6, str(p99))
    check("P99 不等于 P50", abs(p99 - percentile(samples, 50)) > 100.0, str(p99))
    check("P50 落在主体段", abs(percentile(samples, 50) - 10.0) < 1e-6,
          str(percentile(samples, 50)))


def test_percentile_edge_cases() -> None:
    check("空样本返回 None（不能用 0 冒充）", percentile([], 95) is None)
    check("单样本返回该样本", percentile([42.0], 99) == 42.0)
    check("两样本 P50 取插值", abs(percentile([0.0, 10.0], 50) - 5.0) < 1e-9,
          str(percentile([0.0, 10.0], 50)))
    # 顺序无关
    a = percentile([5.0, 1.0, 3.0], 50)
    b = percentile([3.0, 5.0, 1.0], 50)
    check("分位数与输入顺序无关", a == b, f"{a} vs {b}")


def test_summarize_empty_is_none() -> None:
    s = summarize([])
    check("空样本 count == 0", s["count"] == 0)
    check("空样本 mean 为 None", s["mean"] is None)
    check("空样本 p50 为 None", s["p50"] is None)
    check("空样本 p95 为 None", s["p95"] is None)
    check("空样本 p99 为 None", s["p99"] is None)


def test_summarize_reports_sample_size() -> None:
    s = summarize([1.0, 2.0, 3.0])
    check("summarize 带样本量", s["count"] == 3, str(s["count"]))
    check("min/max 正确", s["min"] == 1.0 and s["max"] == 3.0)


# ============================================ CSV 落盘
def test_write_csv_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "summary.csv"
        record = RequestRecord(
            request_index=1,
            request_id="req-1",
            mode="stream",
            llm_type="fake",
            cache_state="mixed",
            concurrency=5,
            success=True,
            status=200,
            cache_hit="true",
            ttft_ms=12.345,
            total_ms=67.891,
            retrieval_ms=3.0,
            delta_count=6,
            done_received=True,
            duplicate_delta=False,
            error_type="",
        )
        write_csv(path, REQUEST_FIELDS, [request_row(record)])
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        check("CSV 写入 1 行", len(rows) == 1, str(len(rows)))
        check("request_id 保留", rows[0]["request_id"] == "req-1")
        check("ttft 保留两位小数", rows[0]["ttft_ms"] == "12.35", rows[0]["ttft_ms"])
        check("total 保留两位小数", rows[0]["total_ms"] == "67.89", rows[0]["total_ms"])
        check("字段集合一致", list(rows[0].keys()) == list(REQUEST_FIELDS))


def test_summary_fields_are_stable() -> None:
    """字段名是报告与图表的契约，改动必须是有意识的。"""
    for field in (
        "mode", "llm_type", "cache_state", "concurrency", "requests",
        "success", "errors", "error_rate", "throughput_rps",
        "mean_ms", "p50_ms", "p95_ms", "p99_ms",
        "ttft_mean_ms", "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms",
        "sample_size_latency", "sample_size_ttft",
    ):
        check(f"summary 含字段 {field}", field in SUMMARY_FIELDS)


# ============================================ runner
def main() -> int:
    print("=" * 60)
    print("Performance stats self-check")
    print("=" * 60)
    test_percentile_known_values()
    test_percentile_not_mean()
    test_percentile_edge_cases()
    test_summarize_empty_is_none()
    test_summarize_reports_sample_size()
    test_write_csv_roundtrip()
    test_summary_fields_are_stable()
    print("=" * 60)
    print(f"通过 {PASSED} 项，失败 {FAILED} 项")
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())

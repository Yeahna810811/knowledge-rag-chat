"""HTTP load test for /api/ask and /api/ask/stream.

Design notes
------------
* Measures from the **client side** over real HTTP, so the numbers include
  FastAPI routing, rate limiting, Redis, retrieval, persistence and SSE
  framing — not just in-process function calls.
* Percentiles are computed from the actual latency samples (linear
  interpolation between closest ranks, the same convention as
  ``numpy.percentile``). Sample size is always written next to them.
* TTFT for streaming = client-side wall clock from request start to the first
  ``delta`` event. Heartbeat/comment frames and non-delta events do not count.
* Nothing is hardcoded: base URL, concurrency levels, request count, warm-up,
  output dir and question pool are all CLI flags.

Usage
-----
    # application-level (FakeLLM harness, no API quota)
    python -m uvicorn evaluation.performance.fake_app:app --port 8010
    python evaluation/performance/run_load_test.py \
        --base-url http://127.0.0.1:8010 \
        --llm-type fake \
        --concurrency 1 5 10 20 \
        --requests-per-level 20

    # real LLM (requires a valid DASHSCOPE_API_KEY in the server process)
    python run.py
    python evaluation/performance/run_load_test.py \
        --base-url http://127.0.0.1:8000 --llm-type real
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 默认问题池：取自 evaluation/corpus 覆盖的主题，避免压测脚本去猜业务问题。
DEFAULT_QUESTIONS: tuple[str, ...] = (
    "RAG 系统的整体架构是怎样的？",
    "BM25 和 Dense 检索有什么区别？",
    "会话历史是怎么持久化的？",
    "Redis 在这个项目里承担什么职责？",
    "SSE 流式输出的心跳帧有什么作用？",
    "知识库文档上传后要经过哪些处理步骤？",
    "检索结果缓存的失效策略是什么？",
    "限流是怎么实现的，Redis 挂掉会怎样？",
    "Docker Compose 里有哪些服务？",
    "前端是如何消费 SSE 事件的？",
)


# ------------------------------------------------------------------ 统计


def percentile(samples: Sequence[float], pct: float) -> float | None:
    """线性插值百分位（与 numpy.percentile 默认方法一致）。

    空样本返回 None——调用方必须处理，不允许用 0 冒充。
    """
    if not samples:
        return None
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    # 位置 = (n-1) * p，落在两个样本之间时线性插值
    pos = (len(ordered) - 1) * (pct / 100.0)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)


def _round(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(value, digits)


def summarize(samples: Sequence[float]) -> dict[str, float | None]:
    if not samples:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(samples),
        "mean": _round(statistics.fmean(samples)),
        "p50": _round(percentile(samples, 50)),
        "p95": _round(percentile(samples, 95)),
        "p99": _round(percentile(samples, 99)),
        "min": _round(min(samples)),
        "max": _round(max(samples)),
    }


# ------------------------------------------------------------------ 结果记录


@dataclass
class RequestRecord:
    request_index: int
    request_id: str
    mode: str
    llm_type: str
    cache_state: str
    concurrency: int
    success: bool
    status: int | None
    cache_hit: str  # "true" / "false" / "unknown"
    ttft_ms: float | None
    total_ms: float
    retrieval_ms: float | None
    delta_count: int
    done_received: bool
    duplicate_delta: bool
    error_type: str
    error_detail: str = ""


@dataclass
class LevelResult:
    mode: str
    llm_type: str
    cache_state: str
    concurrency: int
    requests: int
    success: int
    errors: int
    error_rate: float
    total_duration_s: float
    throughput_rps: float
    latency: dict[str, float | None]
    ttft: dict[str, float | None]
    retrieval: dict[str, float | None]
    done_rate: float
    duplicate_delta_count: int
    cache_hit_count: int
    cache_miss_count: int
    cache_unknown_count: int
    error_types: dict[str, int] = field(default_factory=dict)


# ------------------------------------------------------------------ 单次请求


async def _one_ask(
    client: httpx.AsyncClient,
    question: str,
    session_id: str,
    index: int,
    mode: str,
    llm_type: str,
    cache_state: str,
    concurrency: int,
) -> RequestRecord:
    started = time.perf_counter()
    try:
        response = await client.post(
            "/api/ask",
            json={"question": question, "session_id": session_id, "mode": "rag"},
            timeout=120.0,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        request_id = response.headers.get("X-Request-ID", "")
        success = 200 <= response.status_code < 300
        retrieval_ms: float | None = None
        error_type = ""
        if success:
            try:
                payload = response.json()
                timing = payload.get("timing") or {}
                retrieval_ms = timing.get("retrieval_ms")
            except Exception:  # noqa: BLE001
                retrieval_ms = None
        else:
            error_type = _error_type_from_response(response)
        return RequestRecord(
            request_index=index,
            request_id=request_id,
            mode=mode,
            llm_type=llm_type,
            cache_state=cache_state,
            concurrency=concurrency,
            success=success,
            status=response.status_code,
            cache_hit="unknown",  # /api/ask 不回传 cache_hit（见 README）
            ttft_ms=None,
            total_ms=elapsed_ms,
            retrieval_ms=retrieval_ms,
            delta_count=0,
            done_received=success,
            duplicate_delta=False,
            error_type=error_type,
        )
    except Exception as exc:  # noqa: BLE001 - 网络层异常也要计进 error rate
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return RequestRecord(
            request_index=index,
            request_id="",
            mode=mode,
            llm_type=llm_type,
            cache_state=cache_state,
            concurrency=concurrency,
            success=False,
            status=None,
            cache_hit="unknown",
            ttft_ms=None,
            total_ms=elapsed_ms,
            retrieval_ms=None,
            delta_count=0,
            done_received=False,
            duplicate_delta=False,
            error_type=type(exc).__name__,
            error_detail=str(exc)[:200],
        )


def _error_type_from_response(response: httpx.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            detail = payload.get("detail")
            if isinstance(detail, dict):
                return str(detail.get("error_type") or f"http_{response.status_code}")
    except Exception:  # noqa: BLE001
        pass
    return f"http_{response.status_code}"


async def _one_stream(
    client: httpx.AsyncClient,
    question: str,
    session_id: str,
    index: int,
    mode: str,
    llm_type: str,
    cache_state: str,
    concurrency: int,
) -> RequestRecord:
    started = time.perf_counter()
    ttft_ms: float | None = None
    delta_count = 0
    done_received = False
    cache_hit = "unknown"
    retrieval_ms: float | None = None
    status: int | None = None
    request_id = ""
    error_type = ""
    last_delta: str | None = None
    duplicate_delta = False
    event_name = ""

    try:
        async with client.stream(
            "POST",
            "/api/ask/stream",
            json={"question": question, "session_id": session_id, "mode": "rag"},
            timeout=120.0,
        ) as response:
            status = response.status_code
            request_id = response.headers.get("X-Request-ID", "")
            if status != 200:
                body = await response.aread()
                error_type = f"http_{status}"
                try:
                    payload = json.loads(body.decode("utf-8"))
                    if isinstance(payload, dict):
                        detail = payload.get("detail")
                        if isinstance(detail, dict):
                            error_type = str(detail.get("error_type") or error_type)
                except Exception:  # noqa: BLE001
                    pass
                total_ms = (time.perf_counter() - started) * 1000.0
                return RequestRecord(
                    index, request_id, mode, llm_type, cache_state, concurrency,
                    False, status, cache_hit, ttft_ms, total_ms, retrieval_ms,
                    delta_count, done_received, duplicate_delta, error_type,
                )

            async for raw_line in response.aiter_lines():
                if raw_line.startswith(":"):
                    continue  # 心跳注释帧，不算业务事件
                if raw_line.startswith("event:"):
                    event_name = raw_line[len("event:"):].strip()
                    continue
                if not raw_line.startswith("data:"):
                    continue
                data_text = raw_line[len("data:"):].strip()
                # SSE 的 data 帧本身就是事件负载（没有再嵌套一层 "data"）：
                #   sources -> {"sources": [...], "retrieval_ms": x}
                #   done    -> {..., "timing": {"total_ms","ttft_ms","retrieval_ms","cache_hit"}}
                #   delta   -> {"text": "..."}
                try:
                    payload = json.loads(data_text)
                except Exception:  # noqa: BLE001
                    payload = {}
                if not isinstance(payload, dict):
                    payload = {}

                if event_name != "delta":
                    if event_name == "sources":
                        retrieval_ms = payload.get("retrieval_ms")
                    elif event_name == "done":
                        done_received = True
                        timing = payload.get("timing") or {}
                        hit = timing.get("cache_hit")
                        if hit is True:
                            cache_hit = "true"
                        elif hit is False:
                            cache_hit = "false"
                        if retrieval_ms is None:
                            retrieval_ms = timing.get("retrieval_ms")
                    elif event_name == "error":
                        error_type = str(payload.get("error_type") or "stream_error")
                    event_name = ""
                    continue

                # delta
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - started) * 1000.0
                text = payload.get("text", "") if payload else data_text
                # 只判"相邻两个 delta 完全相同"。
                # 不能用"整条流里出现过相同文本"：中文按字切分时
                # "的""是"这类字本来就会重复出现，那样会把正常输出误报成重复。
                if text == last_delta:
                    duplicate_delta = True
                last_delta = text
                delta_count += 1
                event_name = ""

        total_ms = (time.perf_counter() - started) * 1000.0
        success = done_received and not error_type and delta_count > 0
        if not done_received and not error_type:
            error_type = "stream_incomplete"
        return RequestRecord(
            index, request_id, mode, llm_type, cache_state, concurrency,
            success, status, cache_hit, ttft_ms, total_ms, retrieval_ms,
            delta_count, done_received, duplicate_delta, error_type,
        )
    except Exception as exc:  # noqa: BLE001
        total_ms = (time.perf_counter() - started) * 1000.0
        return RequestRecord(
            index, request_id, mode, llm_type, cache_state, concurrency,
            False, status, cache_hit, ttft_ms, total_ms, retrieval_ms,
            delta_count, done_received, duplicate_delta,
            error_type or type(exc).__name__, str(exc)[:200],
        )


# ------------------------------------------------------------------ 并发档位


async def run_level(
    base_url: str,
    mode: str,
    llm_type: str,
    cache_state: str,
    concurrency: int,
    requests: int,
    warmup: int,
    questions: Sequence[str],
    session_pool_size: int,
    request_offset: int,
) -> tuple[LevelResult, list[RequestRecord]]:
    """跑一个并发档位。

    用固定大小的 worker 池 + 队列分发，而不是"每个请求一个 task"一次性全放
    出去——后者在 20 并发 × 20 请求下会瞬时创建 400 个连接，测的是客户端
    而不是服务端。
    """
    ask = mode == "ask"

    def question_for(i: int) -> str:
        if cache_state == "miss":
            # 每个请求一个唯一问题：缓存必然未命中。
            # 用 uuid 而不是序号，避免 warm-up 与正式样本撞到同一个问题。
            return f"{questions[i % len(questions)]}（样本 {uuid.uuid4().hex[:8]}）"
        # hit / mixed：固定池，warm-up 之后稳定命中
        return questions[i % len(questions)]

    # 队列只装正式样本；warm-up 单独跑，绝不进统计。
    queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(requests):
        queue.put_nowait(i)

    records: list[RequestRecord] = []
    lock = asyncio.Lock()

    async with httpx.AsyncClient(
        base_url=base_url,
        limits=httpx.Limits(max_connections=max(concurrency * 2, 10),
                            max_keepalive_connections=max(concurrency * 2, 10)),
    ) as client:
        # ---- 缓存预热：把整个问题池都跑一遍，保证 hit 状态是确定性的
        # （否则 warm-up 只覆盖池的前几个问题，后面的仍会 miss，
        #   测出来的就是"混合态"而不是"命中态"）
        if cache_state in ("hit", "mixed"):
            for i, q in enumerate(questions):
                sid = f"perf-prime-{i % session_pool_size}"
                if ask:
                    await _one_ask(client, q, sid, -1, mode, llm_type, cache_state, concurrency)
                else:
                    await _one_stream(client, q, sid, -1, mode, llm_type, cache_state, concurrency)

        # ---- warm-up：不计入统计，用于 JIT / 连接池预热
        for i in range(warmup):
            q = question_for(i)
            sid = f"perf-warm-{i % session_pool_size}"
            if ask:
                await _one_ask(client, q, sid, -1, mode, llm_type, cache_state, concurrency)
            else:
                await _one_stream(client, q, sid, -1, mode, llm_type, cache_state, concurrency)

        started_all = time.perf_counter()

        async def worker() -> None:
            while True:
                try:
                    i = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                q = question_for(i)
                sid = f"perf-{i % session_pool_size}"
                if ask:
                    record = await _one_ask(
                        client, q, sid, request_offset + i, mode, llm_type,
                        cache_state, concurrency,
                    )
                else:
                    record = await _one_stream(
                        client, q, sid, request_offset + i, mode, llm_type,
                        cache_state, concurrency,
                    )
                async with lock:
                    records.append(record)

        await asyncio.gather(*[worker() for _ in range(concurrency)])
        total_duration_s = time.perf_counter() - started_all

    success_records = [r for r in records if r.success]
    total_samples = [r.total_ms for r in records]
    ttft_samples = [r.ttft_ms for r in records if r.ttft_ms is not None]
    retrieval_samples = [r.retrieval_ms for r in records if r.retrieval_ms is not None]

    error_types: dict[str, int] = {}
    for r in records:
        if not r.success and r.error_type:
            error_types[r.error_type] = error_types.get(r.error_type, 0) + 1

    result = LevelResult(
        mode=mode,
        llm_type=llm_type,
        cache_state=cache_state,
        concurrency=concurrency,
        requests=len(records),
        success=len(success_records),
        errors=len(records) - len(success_records),
        error_rate=round((len(records) - len(success_records)) / len(records), 4)
        if records else 0.0,
        total_duration_s=round(total_duration_s, 3),
        throughput_rps=round(len(records) / total_duration_s, 2) if total_duration_s else 0.0,
        latency=summarize(total_samples),
        ttft=summarize(ttft_samples),
        retrieval=summarize(retrieval_samples),
        done_rate=round(
            sum(1 for r in records if r.done_received) / len(records), 4
        ) if records else 0.0,
        duplicate_delta_count=sum(1 for r in records if r.duplicate_delta),
        cache_hit_count=sum(1 for r in records if r.cache_hit == "true"),
        cache_miss_count=sum(1 for r in records if r.cache_hit == "false"),
        cache_unknown_count=sum(1 for r in records if r.cache_hit == "unknown"),
        error_types=error_types,
    )
    return result, records


# ------------------------------------------------------------------ 输出


SUMMARY_FIELDS = [
    "mode", "llm_type", "cache_state", "concurrency", "requests", "success",
    "errors", "error_rate", "total_duration_s", "throughput_rps",
    "mean_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms",
    "ttft_mean_ms", "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms",
    "retrieval_mean_ms", "retrieval_p50_ms",
    "done_rate", "duplicate_delta_count",
    "cache_hit_count", "cache_miss_count", "cache_unknown_count",
    "error_types", "sample_size_latency", "sample_size_ttft",
]

REQUEST_FIELDS = [
    "request_index", "request_id", "mode", "llm_type", "cache_state",
    "concurrency", "success", "status", "cache_hit", "ttft_ms", "total_ms",
    "retrieval_ms", "delta_count", "done_received", "duplicate_delta",
    "error_type",
]


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summary_row(result: LevelResult) -> dict[str, Any]:
    return {
        "mode": result.mode,
        "llm_type": result.llm_type,
        "cache_state": result.cache_state,
        "concurrency": result.concurrency,
        "requests": result.requests,
        "success": result.success,
        "errors": result.errors,
        "error_rate": result.error_rate,
        "total_duration_s": result.total_duration_s,
        "throughput_rps": result.throughput_rps,
        "mean_ms": result.latency["mean"],
        "p50_ms": result.latency["p50"],
        "p95_ms": result.latency["p95"],
        "p99_ms": result.latency["p99"],
        "min_ms": result.latency["min"],
        "max_ms": result.latency["max"],
        "ttft_mean_ms": result.ttft["mean"],
        "ttft_p50_ms": result.ttft["p50"],
        "ttft_p95_ms": result.ttft["p95"],
        "ttft_p99_ms": result.ttft["p99"],
        "retrieval_mean_ms": result.retrieval["mean"],
        "retrieval_p50_ms": result.retrieval["p50"],
        "done_rate": result.done_rate,
        "duplicate_delta_count": result.duplicate_delta_count,
        "cache_hit_count": result.cache_hit_count,
        "cache_miss_count": result.cache_miss_count,
        "cache_unknown_count": result.cache_unknown_count,
        "error_types": json.dumps(result.error_types, ensure_ascii=False) if result.error_types else "",
        "sample_size_latency": result.latency["count"],
        "sample_size_ttft": result.ttft["count"],
    }


def request_row(record: RequestRecord) -> dict[str, Any]:
    return {
        "request_index": record.request_index,
        "request_id": record.request_id,
        "mode": record.mode,
        "llm_type": record.llm_type,
        "cache_state": record.cache_state,
        "concurrency": record.concurrency,
        "success": record.success,
        "status": record.status if record.status is not None else "",
        "cache_hit": record.cache_hit,
        "ttft_ms": _round(record.ttft_ms),
        "total_ms": _round(record.total_ms),
        "retrieval_ms": _round(record.retrieval_ms),
        "delta_count": record.delta_count,
        "done_received": record.done_received,
        "duplicate_delta": record.duplicate_delta,
        "error_type": record.error_type,
    }


# ------------------------------------------------------------------ CLI


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HTTP load test for /api/ask and /api/ask/stream.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--mode", nargs="+", choices=["ask", "stream"], default=["ask", "stream"],
        help="要测的接口，默认两个都测。",
    )
    parser.add_argument(
        "--llm-type", choices=["fake", "real"], default="fake",
        help="服务端 LLM 类型。只作为结果标签，脚本本身不会去伪造它。",
    )
    parser.add_argument(
        "--concurrency", nargs="+", type=int, default=[1, 5, 10, 20],
    )
    parser.add_argument("--requests-per-level", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--question-pool-size", type=int, default=10)
    parser.add_argument("--session-pool-size", type=int, default=50)
    parser.add_argument(
        "--cache-state", choices=["mixed", "hit", "miss"], default="mixed",
        help=(
            "mixed=固定问题池（预热后多为命中）；"
            "hit=固定池（语义同 mixed，显式标注）；"
            "miss=每请求唯一问题，必然未命中。"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "evaluation" / "performance" / "results",
    )
    parser.add_argument(
        "--summary-filename", default="performance_summary.csv",
        help="换名可以避免覆盖上一次结果（例如 cache_ab_summary.csv）。",
    )
    parser.add_argument(
        "--requests-filename", default="performance_requests.csv",
    )
    parser.add_argument(
        "--report-filename", default="performance_report.md",
    )
    parser.add_argument(
        "--no-report", action="store_true", help="只写 CSV，不生成 markdown 报告。",
    )
    return parser.parse_args(argv)


def build_report(args: argparse.Namespace, results: list[LevelResult]) -> str:
    lines: list[str] = []
    lines.append("# Performance Benchmark")
    lines.append("")
    lines.append(f"- Base URL: `{args.base_url}`")
    lines.append(f"- LLM type: **{args.llm_type}**")
    lines.append(f"- Cache state: `{args.cache_state}`")
    lines.append(f"- Concurrency levels: {args.concurrency}")
    lines.append(f"- Requests per level: {args.requests_per_level}（warm-up {args.warmup} 不计入）")
    lines.append(f"- Question pool size: {args.question_pool_size}")
    lines.append(f"- Session pool size: {args.session_pool_size}（限流按 session 计）")
    lines.append("")
    lines.append(
        "> 百分位为线性插值法（同 numpy.percentile 默认），样本量见表内 "
        "`sample_size`。TTFT 为客户端视角：发起请求到收到第一个 delta 事件。"
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| mode | concurrency | requests | success | error_rate | throughput_rps "
        "| mean_ms | p50_ms | p95_ms | p99_ms | ttft_p50 | ttft_p95 | ttft_p99 |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in results:
        lines.append(
            f"| {r.mode} | {r.concurrency} | {r.requests} | {r.success} "
            f"| {r.error_rate} | {r.throughput_rps} | {r.latency['mean']} "
            f"| {r.latency['p50']} | {r.latency['p95']} | {r.latency['p99']} "
            f"| {_fmt(r.ttft['p50'])} | {_fmt(r.ttft['p95'])} | {_fmt(r.ttft['p99'])} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for r in results:
        bits = [
            f"- `{r.mode}` @ c={r.concurrency}:",
            f"done_rate={r.done_rate}",
            f"duplicate_delta={r.duplicate_delta_count}",
            f"cache hit={r.cache_hit_count} / miss={r.cache_miss_count} / unknown={r.cache_unknown_count}",
        ]
        if r.retrieval["mean"] is not None:
            bits.append(
                f"retrieval mean={r.retrieval['mean']}ms p50={r.retrieval['p50']}ms "
                f"(n={r.retrieval['count']})"
            )
        if r.error_types:
            bits.append(f"errors={json.dumps(r.error_types, ensure_ascii=False)}")
        lines.append(" ".join(bits))
    lines.append("")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "-" if value is None else str(value)


async def main_async(args: argparse.Namespace) -> list[LevelResult]:
    questions = list(DEFAULT_QUESTIONS)[: max(1, args.question_pool_size)]
    if len(questions) < args.question_pool_size:
        questions = list(DEFAULT_QUESTIONS)
    results: list[LevelResult] = []
    all_records: list[RequestRecord] = []
    offset = 0

    for mode in args.mode:
        for concurrency in args.concurrency:
            print(
                f"[run] mode={mode} llm={args.llm_type} cache={args.cache_state} "
                f"c={concurrency} n={args.requests_per_level}",
                flush=True,
            )
            result, records = await run_level(
                base_url=args.base_url,
                mode=mode,
                llm_type=args.llm_type,
                cache_state=args.cache_state,
                concurrency=concurrency,
                requests=args.requests_per_level,
                warmup=args.warmup,
                questions=questions,
                session_pool_size=args.session_pool_size,
                request_offset=offset,
            )
            offset += args.requests_per_level
            results.append(result)
            all_records.extend(records)
            print(
                f"      -> success={result.success}/{result.requests} "
                f"p50={result.latency['p50']}ms p95={result.latency['p95']}ms "
                f"rps={result.throughput_rps}",
                flush=True,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / args.summary_filename,
        SUMMARY_FIELDS,
        [summary_row(r) for r in results],
    )
    write_csv(
        args.output_dir / args.requests_filename,
        REQUEST_FIELDS,
        [request_row(r) for r in all_records],
    )
    if not args.no_report:
        (args.output_dir / args.report_filename).write_text(
            build_report(args, results), encoding="utf-8"
        )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    asyncio.run(main_async(args))
    print(f"[done] results -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

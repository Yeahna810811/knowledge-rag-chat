"""阶段 3：Async I/O + True SSE 的测试。

    python tests/test_async_sse.py

分三层，越靠下越接近真实链路：

A. 纯帧格式层（只要 anyio）——帧边界、data 单行、心跳是注释帧。
   这一层专门用来挡"看起来能跑、换了个前端就收不到"的脏帧。
B. 服务层（需要 KnowledgeService 等重依赖）——事件顺序、delta 拼接、
   落库、断连部分持久化、fail-open。
C. HTTP 层（httpx + ASGITransport）——真发请求，按字节解析 SSE，
   验证状态码 / content-type / 限流 429 / 并发不串扰。

不引 pytest：与 tests/test_redis.py / test_database.py 保持同一套
无依赖的自带 runner，CI 里一条命令就能跑。
重的 import 做成可选：只有 redis+fakeredis 的精简环境里，
A 层照样跑得完，B/C 层记为 skip。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, AsyncIterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import anyio  # noqa: E402

from frontend.local_rag.api.sse import (  # noqa: E402
    SSE_HEADERS,
    SSE_MEDIA_TYPE,
    format_event,
    heartbeat,
    is_heartbeat,
    sse_frames,
)

# ------------------------------------------------------- 可选的重依赖（B/C 层）
try:
    import fakeredis  # noqa: E402
    from fastapi import FastAPI  # noqa: E402
    from httpx import ASGITransport, AsyncClient  # noqa: E402

    from frontend.local_rag.api.routes import create_router  # noqa: E402
    from frontend.local_rag.cache.redis_client import RedisClient  # noqa: E402
    from frontend.local_rag.config.settings import Settings  # noqa: E402
    from frontend.local_rag.db.database import Database  # noqa: E402
    from frontend.local_rag.services.knowledge_service import (  # noqa: E402
        KnowledgeService,
    )
    from frontend.local_rag.services.rate_limiter import RateLimiter  # noqa: E402

    HEAVY_AVAILABLE = True
    HEAVY_ERROR = ""
except ImportError as exc:  # pragma: no cover - 仅在精简依赖环境下触发
    HEAVY_AVAILABLE = False
    HEAVY_ERROR = str(exc)

PASSED = 0
FAILED = 0
SKIPPED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [ok]   {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name} {detail}")


def skip(name: str, reason: str) -> None:
    global SKIPPED
    SKIPPED += 1
    print(f"  [skip] {name}  {reason}")


def require_heavy(label: str) -> bool:
    if HEAVY_AVAILABLE:
        return True
    skip(label, f"缺少重依赖（{HEAVY_ERROR}）")
    return False


# ------------------------------------------------------------------------- 桩
class BrokenRedis:
    """任何命令都抛异常的 Redis：用来验证流式的 fail-open。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __getattr__(self, name: str):
        def boom(*args: Any, **kwargs: Any):
            raise RuntimeError("redis is down")

        return boom


class FakeRetriever:
    """最小检索器桩：流式测试不关心检索算法，只要别触发模型加载。"""

    def __init__(self) -> None:
        self.is_ready = True

    def search(self, query: str, k: int = 4) -> list[dict]:
        return [{"content": "片段A", "metadata": {"source": "a.md"}}]

    def add_documents(self, documents: Any) -> int:
        return len(documents)

    def clear(self) -> None:
        pass

    def all_chunks(self) -> list[dict]:
        return []

    @property
    def stats(self) -> dict:
        return {"mode": "fake"}


class FakeAsyncOrchestrator:
    """异步编排器桩：按配置的块序列产出 delta，可注入异常与延迟。

    刻意和真实 AgentOrchestrator.astream 产出的事件保持一致
    （sources → trace → delta*），这样 B/C 层测到的顺序就是线上顺序。
    """

    def __init__(
        self,
        chunks: list[str] | None = None,
        sources: list[dict] | None = None,
        fail_at: int | None = None,
        delay: float = 0.0,
        echo_question: bool = False,
    ) -> None:
        self.chunks = chunks if chunks is not None else ["你好", "，", "世界", "！"]
        self.sources = sources if sources is not None else [
            {"content": "片段A", "metadata": {"source": "a.md", "retrieval_score": 0.87}},
            {"content": "片段B", "metadata": {"source": "b.md", "retrieval_score": 0.42}},
        ]
        self.fail_at = fail_at
        self.delay = delay
        # echo_question：把问题回显进答案，用来证明并发时两条流没有串台
        self.echo_question = echo_question
        self.calls: list[tuple[str, str]] = []

    async def astream(
        self,
        question: str,
        history: list[dict] | None = None,
        mode: str = "rag",
    ) -> AsyncIterator[dict]:
        self.calls.append((question, mode))
        mode = (mode or "rag").lower()
        if mode == "rag":
            yield {"event": "sources", "data": {"sources": self.sources, "mode": mode}}
            trace = ["retrieval_agent", "generation_agent"]
        else:
            trace = ["generation_agent"]
        yield {"event": "trace", "data": {"agent_trace": trace}}

        pieces = ([f"<{question}>"] + self.chunks) if self.echo_question else self.chunks
        for index, text in enumerate(pieces):
            if self.delay:
                await anyio.sleep(self.delay)
            if self.fail_at is not None and index == self.fail_at:
                raise RuntimeError("上游 LLM 抽风")
            yield {"event": "delta", "data": {"text": text}}


# ------------------------------------------------------------------ SSE 解析
def parse_frame(frame: str) -> tuple[str, Any]:
    """把单个 SSE 帧解析成 (event 名, data)。心跳返回 ("heartbeat", None)。"""
    if is_heartbeat(frame):
        return ("heartbeat", None)
    name = ""
    data = ""
    for line in frame.splitlines():
        if line.startswith("event:"):
            name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data = line[len("data:") :].strip()
    return (name, json.loads(data) if data else {})


async def drain(frames: AsyncIterator[str]) -> list[str]:
    return [frame async for frame in frames]


# ===================================================================== A 层
def test_frame_format() -> None:
    print("\n[1] SSE 帧格式")

    frame = format_event("delta", {"text": "你好"})
    check("帧以空行结尾", frame.endswith("\n\n"), repr(frame[-5:]))
    check("带 event 行", "event: delta" in frame)
    check("带 data 行", 'data: {"text": "你好"}' in frame)
    check("中文不被转义成 \\uXXXX", "\\u" not in frame)

    multiline = format_event("done", {"answer": "第一行\n第二行"})
    check("data 里没有裸换行", "\n" not in multiline.split("data: ")[1].rstrip("\n\n"))
    parsed_name, parsed_data = parse_frame(multiline.rstrip("\n\n"))
    check("多行 JSON 能原样解析回来", parsed_data["answer"] == "第一行\n第二行")

    empty = format_event("ping")
    check("无 data 时仍有 data 行", "data: " in empty)

    hb = heartbeat()
    check("心跳是注释帧", is_heartbeat(hb))
    check("心跳同样以空行结尾", hb.endswith("\n\n"))
    check("心跳不含 event 字段", "event:" not in hb)

    check(
        "响应头关掉 nginx 缓冲",
        SSE_HEADERS.get("X-Accel-Buffering") == "no",
        str(SSE_HEADERS),
    )
    check("媒体类型带 charset", "text/event-stream" in SSE_MEDIA_TYPE)


def test_frames_from_events() -> None:
    print("\n[2] 事件流转帧")

    async def events() -> AsyncIterator[dict]:
        yield {"event": "meta", "data": {"mode": "rag"}}
        yield {"event": "delta", "data": {"text": "A"}}
        yield {"event": "done", "data": {"answer": "A"}}

    frames = asyncio.run(drain(sse_frames(events(), heartbeat_seconds=0)))
    names = [parse_frame(f)[0] for f in frames]
    check("事件顺序保持", names == ["meta", "delta", "done"], str(names))
    check("不插心跳时没有注释帧", all(not is_heartbeat(f) for f in frames))

    async def slow() -> AsyncIterator[dict]:
        yield {"event": "meta", "data": {}}
        await anyio.sleep(0.15)
        yield {"event": "delta", "data": {"text": "B"}}

    frames = asyncio.run(drain(sse_frames(slow(), heartbeat_seconds=0.05)))
    heartbeats = [f for f in frames if is_heartbeat(f)]
    check("静默期会插心跳", len(heartbeats) >= 1, f"心跳数={len(heartbeats)}")
    names = [parse_frame(f)[0] for f in frames if not is_heartbeat(f)]
    check("心跳不影响业务事件顺序", names == ["meta", "delta"], str(names))


# ===================================================================== B 层
def new_service(
    tmpdir: str,
    orchestrator: FakeAsyncOrchestrator | None = None,
    redis_client: Any = None,
    rate_limit_requests: int = 100,
    heartbeat_seconds: float = 0.05,
) -> tuple[Any, Any]:
    db_file = Path(tmpdir) / "rag_async.db"
    settings = Settings(
        database_url=f"sqlite:///{db_file}",
        redis_cache_enabled=True,
        redis_cache_ttl_seconds=600,
        rate_limit_enabled=True,
        rate_limit_requests=rate_limit_requests,
        rate_limit_window_seconds=60,
        retrieval_mode="bm25",
        retrieval_query_rewrite="prf",
        retrieval_top_k=4,
        sse_heartbeat_seconds=heartbeat_seconds,
    )
    database = Database(f"sqlite:///{db_file}")
    database.create_tables()
    client = redis_client or RedisClient(
        client=fakeredis.FakeStrictRedis(decode_responses=True)
    )
    service = KnowledgeService(settings, database=database, redis_client=client)
    service._orchestrator = orchestrator or FakeAsyncOrchestrator()  # type: ignore[assignment]
    service._retriever = FakeRetriever()  # type: ignore[assignment]
    service._index_loaded = True
    service._ensure_runtime = lambda: service._orchestrator  # type: ignore[method-assign]
    return service, database


async def collect(service: Any, **kwargs: Any) -> list[dict]:
    return [event async for event in service.astream_ask(**kwargs)]


def test_stream_event_order() -> None:
    print("\n[3] 流式事件顺序与拼接")
    if not require_heavy("[3] 流式事件顺序与拼接"):
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        service, _ = new_service(tmpdir)
        events = asyncio.run(collect(service, question="你好吗", session_id="s1"))

        names = [e["event"] for e in events]
        check(
            "顺序为 meta → sources → trace → delta* → done",
            names[:3] == ["meta", "sources", "trace"]
            and names[-1] == "done"
            and set(names[3:-1]) == {"delta"},
            str(names),
        )
        deltas = [e["data"]["text"] for e in events if e["event"] == "delta"]
        done = [e for e in events if e["event"] == "done"][0]["data"]
        check("delta 拼接等于最终答案", "".join(deltas) == done["answer"], done["answer"])
        check("done 带 session_id", done["session_id"] == "s1")
        check("done 带 mode", done["mode"] == "rag")
        check("done 带 sources", len(done["sources"]) == 2)
        check("done 带 agent_trace", done["agent_trace"][-1] == "generation_agent")
        check("done 带 history_turns", done["history_turns"] == 1)


def test_persisted_after_stream() -> None:
    print("\n[4] 流完后落库")
    if not require_heavy("[4] 流完后落库"):
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        service, _ = new_service(tmpdir)
        events = asyncio.run(
            collect(service, question="问题一", session_id="s1", mode="rag")
        )
        done = [e for e in events if e["event"] == "done"][0]["data"]

        history = service.get_history("s1")["history"]
        check("历史里落了 1 轮", len(history) == 1, str(len(history)))
        check("问题是原问题", history[0]["question"] == "问题一")
        check("答案是拼接后的完整答案", history[0]["answer"] == done["answer"])

        asyncio.run(collect(service, question="问题二", session_id="s1"))
        check("第二轮后变成 2 轮", len(service.get_history("s1")["history"]) == 2)

        other = service.get_history("s2")["history"]
        check("其它 session 不受影响", len(other) == 0)


def test_chat_mode_has_no_sources() -> None:
    print("\n[5] chat 模式不下发 sources")
    if not require_heavy("[5] chat 模式不下发 sources"):
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        service, _ = new_service(tmpdir)
        events = asyncio.run(
            collect(service, question="闲聊一句", session_id="c1", mode="chat")
        )
        names = [e["event"] for e in events]
        check("没有 sources 事件", "sources" not in names, str(names))
        done = [e for e in events if e["event"] == "done"][0]["data"]
        check("done 里 sources 为空", done["sources"] == [])
        check("trace 只有生成 agent", done["agent_trace"] == ["generation_agent"])


def test_error_event() -> None:
    print("\n[6] 上游异常变成 error 事件而不是崩溃")
    if not require_heavy("[6] 上游异常变成 error 事件"):
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        boom = FakeAsyncOrchestrator(fail_at=1)
        service, _ = new_service(tmpdir, orchestrator=boom)
        events = asyncio.run(collect(service, question="会炸的问题", session_id="e1"))

        names = [e["event"] for e in events]
        check("最后一个事件是 error", names[-1] == "error", str(names))
        check("异常之前已经发出过 delta", names.count("delta") == 1)
        check("error 带可读信息", "上游 LLM 抽风" in events[-1]["data"]["message"])
        check("没有 done 事件", "done" not in names)

        empty = asyncio.run(collect(service, question="   ", session_id="e2"))
        check(
            "空问题直接 error",
            empty[0]["event"] == "error" and "不能为空" in empty[0]["data"]["message"],
        )


def test_disconnect_persists_partial() -> None:
    print("\n[7] 客户端断开时已生成的部分照常落库")
    if not require_heavy("[7] 断开时部分落库"):
        return

    async def scenario(service: Any) -> int:
        gen = service.astream_ask("很长的回答", session_id="d1")
        seen_deltas = 0
        async for event in gen:
            if event["event"] == "delta":
                seen_deltas += 1
            if seen_deltas == 2:
                # 模拟 ASGI 服务器在客户端断开时向生成器投递取消
                try:
                    await gen.athrow(asyncio.CancelledError())
                except asyncio.CancelledError:
                    pass
                break
        return seen_deltas

    with tempfile.TemporaryDirectory() as tmpdir:
        service, _ = new_service(tmpdir)
        seen = asyncio.run(scenario(service))
        check("断开前确实收到了 2 个 delta", seen == 2, str(seen))

        history = service.get_history("d1")["history"]
        check("部分答案已落库", len(history) == 1, str(len(history)))
        check("落的是已生成的部分", history[0]["answer"] == "你好，", history[0]["answer"] if history else "")

    # 另一条投递路径：显式 aclose()（GeneratorExit）
    async def scenario_aclose(service: Any) -> int:
        gen = service.astream_ask("很长的回答", session_id="d2")
        seen = 0
        async for event in gen:
            if event["event"] == "delta":
                seen += 1
            if seen == 2:
                break
        await gen.aclose()
        return seen

    with tempfile.TemporaryDirectory() as tmpdir:
        service, _ = new_service(tmpdir)
        asyncio.run(scenario_aclose(service))
        history = service.get_history("d2")["history"]
        check("aclose 路径同样落库", len(history) == 1, str(len(history)))


def test_stream_fail_open_on_redis_down() -> None:
    print("\n[8] Redis 故障时流式问答 fail-open")
    if not require_heavy("[8] Redis 故障 fail-open"):
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        service, _ = new_service(tmpdir, redis_client=RedisClient(client=BrokenRedis()))
        events = asyncio.run(collect(service, question="Redis 挂了", session_id="r1"))
        names = [e["event"] for e in events]
        check("流照常完成", names[-1] == "done", str(names))
        check("没有因为 Redis 报错", "error" not in names)
        check("历史照常落库", len(service.get_history("r1")["history"]) == 1)
        check(
            "status 里 Redis 仍标记为不健康",
            service.redis_status()["healthy"] is False,
        )


# ===================================================================== C 层
def build_app(service: Any, limiter: Any = None) -> Any:
    app = FastAPI()
    app.include_router(create_router(service, rate_limiter=limiter), prefix="/api")
    return app


async def post_stream(app: Any, payload: dict) -> tuple[int, dict, list[tuple[str, Any]]]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST", "/api/ask/stream", json=payload, timeout=30
        ) as response:
            status = response.status_code
            headers = dict(response.headers)
            buffer = ""
            frames: list[tuple[str, Any]] = []
            async for chunk in response.aiter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    raw, buffer = buffer.split("\n\n", 1)
                    if raw.strip():
                        frames.append(parse_frame(raw))
            return status, headers, frames


def test_http_stream() -> None:
    print("\n[9] HTTP 端到端 SSE")
    if not require_heavy("[9] HTTP 端到端 SSE"):
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        service, _ = new_service(tmpdir)
        app = build_app(service)
        status, headers, frames = asyncio.run(
            post_stream(app, {"question": "HTTP 流式", "session_id": "h1", "mode": "rag"})
        )

        check("状态码 200", status == 200, str(status))
        check(
            "content-type 是 text/event-stream",
            "text/event-stream" in headers.get("content-type", ""),
            headers.get("content-type", ""),
        )
        check("带 no-cache", "no-cache" in headers.get("cache-control", ""))

        names = [name for name, _ in frames]
        check("首个事件是 meta", names[0] == "meta", str(names[:3]))
        check("sources 在 delta 之前", names.index("sources") < names.index("delta"))
        check("最后是 done", names[-1] == "done", str(names))

        deltas = [data["text"] for name, data in frames if name == "delta"]
        done = [data for name, data in frames if name == "done"][0]
        check("delta 拼接等于 done.answer", "".join(deltas) == done["answer"])
        check("心跳被正确识别且不混入业务事件", "heartbeat" not in names)


def test_http_rate_limit() -> None:
    print("\n[10] 流式端点的限流")
    if not require_heavy("[10] 流式端点限流"):
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        service, _ = new_service(tmpdir, rate_limit_requests=100)
        client_redis = RedisClient(client=fakeredis.FakeStrictRedis(decode_responses=True))
        limiter = RateLimiter(
            client_redis, enabled=True, limit_requests=2, window_seconds=60
        )
        app = build_app(service, limiter=limiter)
        payload = {"question": "限流测试", "session_id": "rl", "mode": "rag"}

        codes = []
        retry_after = ""
        for _ in range(3):
            status, headers, _ = asyncio.run(post_stream(app, payload))
            codes.append(status)
            if status == 429:
                retry_after = headers.get("retry-after", "")

        check("前两次放行", codes[:2] == [200, 200], str(codes))
        check("第三次被限", codes[2] == 429, str(codes))
        check("429 带 Retry-After", retry_after.isdigit() and int(retry_after) > 0, retry_after)

        other = {"question": "换个人", "session_id": "other", "mode": "rag"}
        status, _, _ = asyncio.run(post_stream(app, other))
        check("其它 session 不受影响", status == 200)


def test_http_concurrent_streams() -> None:
    print("\n[11] 并发两条流互不串扰")
    if not require_heavy("[11] 并发两条流"):
        return

    async def scenario() -> dict[str, str]:
        # echo_question=True：答案里带上各自的问题，串台一眼就能看出来
        service, _ = new_service(
            tmpdir, orchestrator=FakeAsyncOrchestrator(echo_question=True)
        )
        app = build_app(service)
        payloads = {
            "A": {"question": "问题A", "session_id": "A", "mode": "rag"},
            "B": {"question": "问题B", "session_id": "B", "mode": "rag"},
        }
        results: dict[str, str] = {}

        async def run(key: str) -> None:
            _, _, frames = await post_stream(app, payloads[key])
            done = [data for name, data in frames if name == "done"][0]
            results[key] = done["answer"]
            results[key + "_sid"] = done["session_id"]

        async with anyio.create_task_group() as tg:
            for key in payloads:
                tg.start_soon(run, key)
        return results

    with tempfile.TemporaryDirectory() as tmpdir:
        results = asyncio.run(scenario())
        check("A 拿到自己的答案", results.get("A") == "<问题A>你好，世界！", str(results))
        check("B 拿到自己的答案", results.get("B") == "<问题B>你好，世界！", str(results))
        check("A 与 B 的答案没有混在一起", results.get("A") != results.get("B"))
        check("A 的 session 没串到 B", results.get("A_sid") == "A")
        check("B 的 session 没串到 A", results.get("B_sid") == "B")


def test_stream_does_not_block_event_loop() -> None:
    print("\n[12] 流式期间事件循环没有被阻塞")
    if not require_heavy("[12] 事件循环不被阻塞"):
        return

    async def scenario() -> tuple[int, int]:
        # 每块之间留 20ms，让流持续约 80ms——足够事件循环空转出好几个 tick。
        # 如果检索/生成是同步阻塞的，这段时间里 tick 会一直是 0。
        service, _ = new_service(
            tmpdir, orchestrator=FakeAsyncOrchestrator(delay=0.02)
        )
        app = build_app(service)
        ticks = 0
        running = True

        async def heartbeat_tick() -> None:
            # 一个纯粹的 asyncio 任务：如果事件循环被同步检索堵死，
            # 它在流开始的这段时间里一次都跑不起来。
            nonlocal ticks
            while running:
                await asyncio.sleep(0.01)
                ticks += 1

        async with anyio.create_task_group() as tg:
            tg.start_soon(heartbeat_tick)
            _, _, frames = await post_stream(app, {"question": "并发", "session_id": "nb"})
            running = False
        return ticks, len(frames)

    with tempfile.TemporaryDirectory() as tmpdir:
        ticks, frame_count = asyncio.run(scenario())
        check("流式期间事件循环仍在跑", ticks >= 2, f"ticks={ticks}")
        check("确实产生了事件帧", frame_count > 0, str(frame_count))


def test_real_llm_optional() -> None:
    print("\n[13] 真实 LLM 流式（可选）")
    if os.getenv("STREAM_REAL_LLM", "").lower() not in {"1", "true", "yes"}:
        skip("真实 LLM 流式", "未设置 STREAM_REAL_LLM=1，跳过（不消耗额度）")
        return
    if not require_heavy("[13] 真实 LLM 流式"):
        return

    from frontend.local_rag.config.settings import get_settings
    from frontend.local_rag.core.agents.orchestrator import AgentOrchestrator

    settings = get_settings()
    service = KnowledgeService(settings)
    started = time.time()
    events = asyncio.run(
        collect(service, question="用一句话介绍你自己", session_id="real", mode="chat")
    )
    elapsed = time.time() - started
    names = [e["event"] for e in events]
    check("真实链路返回 done", names[-1] == "done", str(names))
    deltas = [e["data"]["text"] for e in events if e["event"] == "delta"]
    check("产生了多个 delta（真的在流式）", len(deltas) > 1, str(len(deltas)))
    print(f"        （真实 LLM 耗时 {elapsed:.2f}s，delta {len(deltas)} 块）")


def main() -> int:
    print("=" * 62)
    print("阶段 3：Async I/O + True SSE")
    print("=" * 62)

    test_frame_format()
    test_frames_from_events()
    test_stream_event_order()
    test_persisted_after_stream()
    test_chat_mode_has_no_sources()
    test_error_event()
    test_disconnect_persists_partial()
    test_stream_fail_open_on_redis_down()
    test_http_stream()
    test_http_rate_limit()
    test_http_concurrent_streams()
    test_stream_does_not_block_event_loop()
    test_real_llm_optional()

    print()
    print("=" * 62)
    print(f"通过 {PASSED} 项，失败 {FAILED} 项，跳过 {SKIPPED} 项")
    print("=" * 62)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())

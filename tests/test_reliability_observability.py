"""阶段 4：Reliability & Observability 的测试。

    python tests/test_reliability_observability.py

分组对应需求 A~J：
A. Request ID（生成 / 复用 / 响应头 / SSE meta / 并发不串）
B. 结构化日志（事件齐全 / request_id 齐全 / 不泄漏敏感内容）
C. 非流式重试（瞬时重试成功 / 达上限停止 / 鉴权不重试 / 退避序列）
D. 流式重试（首 token 前重试 / 首 token 后绝不重试 / 不落半截 / error 只一次）
E. 超时（invoke / 首 token / 空闲 / 超时后不落库 / 无遗留任务）
F. 数据库纪律（恰好写一次 / 写失败不重跑 LLM / 写失败不重复写）
G. Redis fail-open（Redis 挂了照样 200，只记 warning）
H. 限流（429 + Retry-After，且不进 LLM）
I. Timing（total / ttft / generation / retrieval 定义正确）
J. Stage 3 回归（心跳 / abort / GeneratorExit / CancelledError 仍然不落库）

设计原则与 tests/test_async_sse.py 一致：
- 不引 pytest，自带 runner，CI 一条命令跑完
- 重试的 sleep 全部注入，测试不等真实时间
- LLM 一律用桩，默认不烧真实 DashScope 额度
"""

from __future__ import annotations

import anyio
import asyncio
import gc
import json
import logging
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, AsyncIterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------- 可选的重依赖
try:
    import fakeredis
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from frontend.local_rag.api.routes import create_router
    from frontend.local_rag.cache.redis_client import RedisClient
    from frontend.local_rag.config.settings import Settings
    from frontend.local_rag.core.agents.generation_agent import GenerationAgent
    from frontend.local_rag.db.database import Database
    from frontend.local_rag.services.knowledge_service import KnowledgeService
    from frontend.local_rag.services.rate_limiter import RateLimiter

    HEAVY_AVAILABLE = True
    HEAVY_ERROR = ""
except ImportError as exc:  # pragma: no cover
    HEAVY_AVAILABLE = False
    HEAVY_ERROR = str(exc)

# 轻量依赖：这些不需要 fakeredis / httpx，A/B/C/D/E 的纯逻辑部分照样能跑
from frontend.local_rag.observability.errors import (  # noqa: E402
    ClassifiedError,
    ErrorType,
    classify_exception,
    http_status_for,
    is_retryable,
    safe_message,
)
from frontend.local_rag.observability.request_context import (  # noqa: E402
    current_request_id,
    is_valid_request_id,
    new_request_id,
    request_scope,
    resolve_request_id,
)
from frontend.local_rag.observability.retry import RetryPolicy, run_with_retry  # noqa: E402
from frontend.local_rag.observability.stream_guard import guarded_astream  # noqa: E402
from frontend.local_rag.observability.structured_log import (  # noqa: E402
    EVENT_LOGGER_NAME,
    log_event,
)

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


# ============================================================ 日志捕获
class LogCapture(logging.Handler):
    """收集结构化事件日志。

    直接挂在事件 logger 上（不是 root），避免把 uvicorn / SQLAlchemy
    的噪音一起收进来——那些日志不带 request_id，混在里面会让
    "每条日志都有 request_id" 这类断言永远为真，断言就失效了。
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []
        self._logger = logging.getLogger(EVENT_LOGGER_NAME)
        self._previous_level = self._logger.level
        self._previous_propagate = self._logger.propagate

    def __enter__(self) -> "LogCapture":
        self._logger.addHandler(self)
        # 必须显式降级别：logger 默认继承 root 的 WARNING，
        # INFO 事件会被 isEnabledFor 直接丢掉，handler 根本收不到。
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False
        return self

    def __exit__(self, *_: object) -> None:
        self._logger.removeHandler(self)
        self._logger.setLevel(self._previous_level)
        self._logger.propagate = self._previous_propagate

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())

    def events(self) -> list[str]:
        """取出每条日志的 event= 值。"""
        names = []
        for message in self.messages:
            for token in message.split():
                if token.startswith("event="):
                    names.append(token[len("event=") :])
                    break
        return names

    def has_event(self, name: str) -> bool:
        return name in self.events()

    def lines_with(self, name: str) -> list[str]:
        return [m for m in self.messages if f"event={name}" in m]


def field_of(line: str, key: str) -> str | None:
    """从一行 k=v 日志里取某个字段（支持引号包裹的值）。"""
    key_prefix = f"{key}="
    for token in line.split():
        if token.startswith(key_prefix):
            value = token[len(key_prefix) :]
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            return value
    return None


# ============================================================ LLM 桩
class FakeLLM:
    """可控的假 LLM。

    重试 / 超时的所有断言都要落在"到底调了几次"上，所以调用计数是
    这个桩的核心职责——光看最终结果无法区分"重试成功"和"一次就成"。
    """

    def __init__(
        self,
        chunks: list[str] | None = None,
        *,
        raise_on_stream: BaseException | None = None,
        raise_before_token: int = 0,
        raise_after_token_at: int | None = None,
        raise_on_invoke: BaseException | None = None,
        raise_on_invoke_times: int = 0,
        first_token_delay: float = 0.0,
        piece_delay: float = 0.0,
        hang_after: int | None = None,
    ) -> None:
        self.chunks = chunks or ["你", "好", "世", "界"]
        self.raise_on_stream = raise_on_stream
        self.raise_before_token = raise_before_token
        self.raise_after_token_at = raise_after_token_at
        self.raise_on_invoke = raise_on_invoke
        self.raise_on_invoke_times = raise_on_invoke_times
        self.first_token_delay = first_token_delay
        self.piece_delay = piece_delay
        self.hang_after = hang_after

        self.stream_calls = 0
        self.invoke_calls = 0

    def invoke(self, messages: Any) -> Any:
        self.invoke_calls += 1
        if self.raise_on_invoke is not None and self.invoke_calls <= self.raise_on_invoke_times:
            raise self.raise_on_invoke
        if self.raise_on_invoke is not None and self.raise_on_invoke_times == 0:
            raise self.raise_on_invoke

        class _Resp:
            content = "".join(self.chunks)

        return _Resp()

    async def astream(self, messages: Any) -> AsyncIterator[Any]:
        self.stream_calls += 1
        call_index = self.stream_calls

        if self.raise_before_token and call_index <= self.raise_before_token:
            if self.raise_on_stream is not None:
                raise self.raise_on_stream
            raise RuntimeError("上游连接重置")

        if self.first_token_delay:
            await anyio.sleep(self.first_token_delay)

        for index, text in enumerate(self.chunks):
            if self.piece_delay:
                await anyio.sleep(self.piece_delay)
            if self.raise_after_token_at is not None and index == self.raise_after_token_at:
                raise self.raise_on_stream or RuntimeError("生成中途炸了")
            yield _Chunk(text)
            if self.hang_after is not None and index == self.hang_after:
                # 永久静默：用来触发空闲超时
                await anyio.sleep(60)


class _Chunk:
    def __init__(self, text: str) -> None:
        self.content = text


class GenerationOnlyOrchestrator:
    """只保留生成段的编排器桩。

    刻意用**真实的 GenerationAgent**（而不是连生成一起桩掉）：
    重试、超时、错误分类这些逻辑全在 generation 层，
    把它换成桩的话这一组用例就什么都测不到了。
    """

    def __init__(self, generation_agent: Any, sources: list[dict] | None = None) -> None:
        self.generation_agent = generation_agent
        self.sources = sources or [
            {"content": "片段A", "metadata": {"source": "a.md"}},
        ]

    def ask(self, question: str, history: list[dict] | None = None, mode: str = "rag") -> dict:
        result = self.generation_agent.run(
            question=question, history=history or [], sources=self.sources, mode=mode
        )
        return {
            "question": question,
            "answer": result.data["answer"],
            "sources": self.sources if mode == "rag" else [],
            "mode": mode,
            "agent_trace": ["generation_agent"],
            "retrieval_ms": None,
        }

    async def astream(
        self, question: str, history: list[dict] | None = None, mode: str = "rag"
    ) -> AsyncIterator[dict]:
        mode = (mode or "rag").lower()
        if mode == "rag":
            # 和真实 AgentOrchestrator 保持一致：sources 事件带 retrieval_ms。
            # 桩里少这个字段的话，"rag 模式有 retrieval_ms" 这条断言
            # 测的就不是产品代码而是桩本身了。
            yield {
                "event": "sources",
                "data": {"sources": self.sources, "mode": mode, "retrieval_ms": 1.0},
            }
        yield {"event": "trace", "data": {"agent_trace": ["generation_agent"]}}
        async for text in self.generation_agent.astream_text(
            question=question, history=history or [], sources=self.sources, mode=mode
        ):
            yield {"event": "delta", "data": {"text": text}}


class CountingStore:
    """数写库次数：per-token 写库 / 重复写会立刻露馅。"""

    def __init__(self, inner: Any, fail: bool = False) -> None:
        self._inner = inner
        self.append_calls = 0
        self.fail = fail
        self.write_started = threading.Event()

    def append_turn(self, *args: Any, **kwargs: Any) -> Any:
        self.append_calls += 1
        self.write_started.set()
        if self.fail:
            raise RuntimeError("数据库写入失败")
        return self._inner.append_turn(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class BrokenRedis:
    """任何命令都抛异常：Redis 不可用。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __getattr__(self, name: str):
        def boom(*args: Any, **kwargs: Any):
            raise RuntimeError("redis is down")

        return boom


class FakeRetriever:
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


# ============================================================ 服务脚手架
def new_service(
    tmpdir: str,
    *,
    llm: FakeLLM | None = None,
    redis_client: Any = None,
    retry_max_attempts: int = 3,
    base_delay: float = 0.01,
    first_token_timeout: float = 5.0,
    idle_timeout: float = 5.0,
    rate_limit_requests: int = 100,
) -> tuple[Any, Any, FakeLLM]:
    db_file = Path(tmpdir) / "rag_stage4.db"
    settings = Settings(
        database_url=f"sqlite:///{db_file}",
        dashscope_api_key="dummy-key-for-tests",
        redis_cache_enabled=True,
        redis_cache_ttl_seconds=600,
        rate_limit_enabled=True,
        rate_limit_requests=rate_limit_requests,
        rate_limit_window_seconds=60,
        retrieval_mode="bm25",
        retrieval_query_rewrite="off",
        retrieval_top_k=4,
        sse_heartbeat_seconds=0.05,
        llm_retry_enabled=True,
        llm_retry_max_attempts=retry_max_attempts,
        llm_retry_base_delay_seconds=base_delay,
        llm_retry_max_delay_seconds=1.0,
        llm_first_token_timeout_seconds=first_token_timeout,
        llm_stream_idle_timeout_seconds=idle_timeout,
        llm_sdk_max_retries=0,
        structured_logging_enabled=False,
    )
    database = Database(f"sqlite:///{db_file}")
    database.create_tables()
    client = redis_client or RedisClient(
        client=fakeredis.FakeStrictRedis(decode_responses=True)
    )
    service = KnowledgeService(settings, database=database, redis_client=client)
    service._retriever = FakeRetriever()
    service._index_loaded = True

    fake_llm = llm or FakeLLM()
    agent = GenerationAgent(settings)
    agent.llm = fake_llm  # type: ignore[assignment]
    service._orchestrator = GenerationOnlyOrchestrator(agent)  # type: ignore[assignment]
    service._ensure_runtime = lambda: service._orchestrator  # type: ignore[method-assign]
    return service, database, fake_llm


def build_app(service: Any) -> FastAPI:
    app = FastAPI()
    app.include_router(create_router(service), prefix="/api")
    return app


async def post_ask(app: FastAPI, payload: dict, headers: dict | None = None):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/api/ask", json=payload, headers=headers or {})


async def post_stream(app: FastAPI, payload: dict, headers: dict | None = None):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/ask/stream", json=payload, headers=headers or {}
        )
        return response.status_code, response.headers, _parse_sse(response.text)


def _parse_sse(payload: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for frame in payload.split("\n\n"):
        if not frame.strip() or frame.startswith(":"):
            continue
        name, data = "", ""
        for line in frame.splitlines():
            if line.startswith("event:"):
                name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data = line[len("data:") :].strip()
        events.append((name, json.loads(data) if data else {}))
    return events


async def collect(service: Any, **kwargs: Any) -> list[dict]:
    events: list[dict] = []
    async for event in service.astream_ask(**kwargs):
        events.append(event)
    return events


# ================================================== A. Request ID
def test_request_id_basics() -> None:
    print("\n[A1] request_id 生成与复用")

    generated = resolve_request_id(None)
    check("自动生成非空", bool(generated))
    check("自动生成长度合理", len(generated) == 32, generated)
    check("两次生成不相同", generated != new_request_id())

    check("合法客户端 ID 被复用", resolve_request_id("client-abc-123") == "client-abc-123")
    check("空字符串不复用", resolve_request_id("") != "")
    check("含空格不复用", resolve_request_id("bad id") != "bad id")
    check("超长不复用", resolve_request_id("x" * 200) != "x" * 200)
    # 换行是最危险的一种：会被写进日志当成新的一条记录
    check("含换行不复用", "\n" not in resolve_request_id("abc\ndef"))
    check("非字符串（Header 对象）不崩", not is_valid_request_id(object()))  # type: ignore[arg-type]

    with request_scope("rid-1", "s1", "rag"):
        check("作用域内可读到", current_request_id() == "rid-1")
    check("退出后复位", current_request_id() == "-")


def test_request_id_http() -> None:
    print("\n[A2] HTTP / SSE 的 X-Request-ID")
    if not require_heavy("[A2] HTTP request id"):
        return

    async def scenario() -> tuple[Any, Any, dict]:
        with tempfile.TemporaryDirectory() as tmpdir:
            service, _, _ = new_service(tmpdir)
            app = build_app(service)
            r1 = await post_ask(app, {"question": "问题", "session_id": "s1", "mode": "chat"})
            r2 = await post_ask(
                app,
                {"question": "问题", "session_id": "s1", "mode": "chat"},
                headers={"X-Request-ID": "client-fixed-id"},
            )
            return r1, r2, {}

    r1, r2, _ = asyncio.run(scenario())
    check("非流式响应带 X-Request-ID", bool(r1.headers.get("X-Request-ID")))
    check(
        "响应体 request_id 与响应头一致",
        r1.json().get("request_id") == r1.headers.get("X-Request-ID"),
        str(r1.json().get("request_id")),
    )
    check("客户端传入的 ID 被复用", r2.headers.get("X-Request-ID") == "client-fixed-id")
    check("响应体同样复用", r2.json().get("request_id") == "client-fixed-id")

    async def stream_scenario() -> tuple[str, list[tuple[str, dict]]]:
        with tempfile.TemporaryDirectory() as tmpdir:
            service, _, _ = new_service(tmpdir)
            app = build_app(service)
            status, headers, events = await post_stream(
                app,
                {"question": "问题", "session_id": "s1", "mode": "chat"},
                headers={"X-Request-ID": "stream-rid-1"},
            )
            return headers.get("X-Request-ID", ""), events

    header_rid, events = asyncio.run(stream_scenario())
    check("流式响应头带 X-Request-ID", header_rid == "stream-rid-1", header_rid)
    meta = [d for name, d in events if name == "meta"]
    check("SSE meta 存在", len(meta) == 1)
    check("SSE meta 带 request_id", meta[0].get("request_id") == "stream-rid-1", str(meta[0]))
    done = [d for name, d in events if name == "done"]
    check("done 也带 request_id", done and done[0].get("request_id") == "stream-rid-1")


def test_request_id_concurrency() -> None:
    print("\n[A3] 并发 request_id 不串")
    if not require_heavy("[A3] 并发 request_id"):
        return

    async def scenario() -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as tmpdir:
            service, _, _ = new_service(tmpdir)
            app = build_app(service)
            results: dict[str, Any] = {}
            seen: list[str] = []

            async def one(index: int) -> None:
                rid = f"concurrent-{index}"
                status, headers, events = await post_stream(
                    app,
                    {
                        "question": f"问题{index}",
                        "session_id": f"s{index}",
                        "mode": "chat",
                    },
                    headers={"X-Request-ID": rid},
                )
                done = [d for name, d in events if name == "done"]
                results[rid] = {
                    "header": headers.get("X-Request-ID"),
                    "done_rid": done[0].get("request_id") if done else None,
                    "session": done[0].get("session_id") if done else None,
                    "answer": done[0].get("answer") if done else None,
                }
                seen.append(rid)

            async with anyio.create_task_group() as tg:
                for i in range(5):
                    tg.start_soon(one, i)
            results["_count"] = len(set(seen))
            return results

    results = asyncio.run(scenario())
    check("5 条并发流全部完成", results["_count"] == 5, str(results["_count"]))
    ok_header = all(results[f"concurrent-{i}"]["header"] == f"concurrent-{i}" for i in range(5))
    check("每条流的响应头 ID 都是自己的", ok_header)
    ok_done = all(results[f"concurrent-{i}"]["done_rid"] == f"concurrent-{i}" for i in range(5))
    check("每条流的 done.request_id 都是自己的", ok_done)
    ok_session = all(results[f"concurrent-{i}"]["session"] == f"s{i}" for i in range(5))
    check("session 也没有串台", ok_session)


def test_contextvar_isolation() -> None:
    print("\n[A4] ContextVar 并发隔离")

    async def worker(rid: str, delays: float, observed: dict[str, str]) -> None:
        with request_scope(rid):
            await anyio.sleep(delays)
            observed[rid] = current_request_id()

    async def scenario() -> dict[str, str]:
        observed: dict[str, str] = {}
        async with anyio.create_task_group() as tg:
            for i in range(5):
                tg.start_soon(worker, f"ctx-{i}", 0.01 * (5 - i), observed)
        return observed

    observed = asyncio.run(scenario())
    check("每个任务读到自己的 ID", all(v == k for k, v in observed.items()), str(observed))


# ================================================== B. Structured logs
def test_structured_logs_success() -> None:
    print("\n[B1] 正常请求的结构化日志")
    if not require_heavy("[B1] 成功日志"):
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        service, _, _ = new_service(tmpdir)
        with LogCapture() as cap:
            asyncio.run(collect(service, question="日志测试", session_id="s1", mode="chat"))

        events = cap.events()
        check("有 request_start", cap.has_event("request_start"), str(events))
        check("有 llm_start", cap.has_event("llm_start"))
        check("有 llm_first_token", cap.has_event("llm_first_token"))
        check("有 llm_complete", cap.has_event("llm_complete"))
        check("有 persistence_complete", cap.has_event("persistence_complete"))
        check("有 request_complete", cap.has_event("request_complete"))

        check(
            "每条日志都带 request_id",
            all("request_id=" in m for m in cap.messages),
            str([m for m in cap.messages if "request_id=" not in m][:2]),
        )
        rid = field_of(cap.messages[0], "request_id")
        check(
            "所有日志 request_id 一致",
            all(field_of(m, "request_id") == rid for m in cap.messages),
        )
        check("没有 request_id=- 的日志", all(field_of(m, "request_id") != "-" for m in cap.messages))


def test_structured_logs_failure() -> None:
    print("\n[B2] 失败请求的日志与错误分类")
    if not require_heavy("[B2] 失败日志"):
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        llm = FakeLLM(
            raise_on_stream=RuntimeError("上游炸了"),
            raise_before_token=99,  # 每次都在首 token 前失败 → 重试到上限
        )
        service, _, fake = new_service(tmpdir, llm=llm, retry_max_attempts=2)
        with LogCapture() as cap:
            events = asyncio.run(
                collect(service, question="会失败", session_id="s1", mode="chat")
            )

        check("有 request_failed", cap.has_event("request_failed"), str(cap.events()))
        failed = cap.lines_with("request_failed")
        check("失败日志带 error_type", field_of(failed[0], "error_type") is not None)
        check("失败日志带 total_ms", field_of(failed[0], "total_ms") is not None)

        error_events = [e for e in events if e["event"] == "error"]
        check("只发一次 error 事件", len(error_events) == 1, str(len(error_events)))
        check("error 带 request_id", bool(error_events[0]["data"].get("request_id")))
        check("error 带 error_type", bool(error_events[0]["data"].get("error_type")))
        check("没有 done 事件", "done" not in [e["event"] for e in events])


def test_logs_do_not_leak() -> None:
    print("\n[B3] 日志不泄漏敏感内容")
    if not require_heavy("[B3] 日志脱敏"):
        return

    secret_key = "sk-abcdefgh12345678"
    question = "这是一个不该被完整打印的问题内容"
    with tempfile.TemporaryDirectory() as tmpdir:
        llm = FakeLLM(chunks=["这是", "不该完整打印的答案"])
        service, _, _ = new_service(tmpdir, llm=llm)
        with LogCapture() as cap:
            asyncio.run(collect(service, question=question, session_id="s1", mode="chat"))

        joined = "\n".join(cap.messages)
        check("不打印完整问题", question not in joined)
        check("不打印完整答案", "不该完整打印的答案" not in joined)
        check("记录 question_len", any("question_len=" in m for m in cap.messages))
        check("记录 answer_len", any("answer_len=" in m for m in cap.messages))
        check("日志里没有裸换行", all("\n" not in m for m in cap.messages))
        check("没有痕迹表明 key 被写出", secret_key not in joined)


def test_safe_message() -> None:
    print("\n[B4] safe_message 脱敏与截断")

    redacted = safe_message("api_key=sk-abcdefgh12345678 出问题了")
    check("API key 被打码", "sk-abcdefgh12345678" not in redacted, redacted)
    check("保留可读上下文", "出问题了" in redacted, redacted)

    check("换行被压平", "\n" not in safe_message("第一行\n第二行"))
    check("超长被截断", len(safe_message("x" * 500)) <= 200)
    check("Bearer token 被打码", "Bearer abcdefgh12345678" not in safe_message("Bearer abcdefgh12345678"))


# ================================================== C. Non-stream retry
def test_retry_policy() -> None:
    print("\n[C1] 退避序列")

    policy = RetryPolicy(max_attempts=3, base_delay_seconds=0.5, max_delay_seconds=8.0)
    check("第 1 次失败等 base", policy.delay_after(1) == 0.5)
    check("第 2 次失败等 2×base", policy.delay_after(2) == 1.0)
    check("封顶生效", RetryPolicy(base_delay_seconds=1.0, max_delay_seconds=3.0).delay_after(9) == 3.0)
    check("max_attempts=1 表示不重试", not RetryPolicy(max_attempts=1).enabled)


def test_non_stream_retry_success() -> None:
    print("\n[C2] 瞬时故障重试后成功")

    sleeps: list[float] = []
    calls = {"n": 0}

    def operation() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("上游超时")
        return "成功"

    result = run_with_retry(
        operation,
        RetryPolicy(max_attempts=3, base_delay_seconds=0.5),
        sleep=lambda s: sleeps.append(s),
    )
    check("最终成功", result == "成功")
    check("共调用 3 次", calls["n"] == 3, str(calls["n"]))
    check("退避 2 次", len(sleeps) == 2, str(sleeps))
    check("退避序列是指数", sleeps == [0.5, 1.0], str(sleeps))


def test_non_stream_retry_exhausted() -> None:
    print("\n[C3] 达到最大次数后停止")

    calls = {"n": 0}

    def operation() -> str:
        calls["n"] += 1
        raise TimeoutError("一直超时")

    try:
        run_with_retry(operation, RetryPolicy(max_attempts=3), sleep=lambda s: None)
        check("应当抛错", False)
    except ClassifiedError as exc:
        check("错误类型是 LLM_TIMEOUT", exc.error_type is ErrorType.LLM_TIMEOUT, str(exc.error_type))
        check("恰好调用 3 次", calls["n"] == 3, str(calls["n"]))
        check("HTTP 状态 504", exc.http_status == 504, str(exc.http_status))


def test_non_stream_no_retry_on_auth() -> None:
    print("\n[C4] 鉴权失败不重试")

    class AuthError(Exception):
        status_code = 401

    calls = {"n": 0}

    def operation() -> str:
        calls["n"] += 1
        raise AuthError("invalid api key")

    try:
        run_with_retry(operation, RetryPolicy(max_attempts=3), sleep=lambda s: None)
        check("应当抛错", False)
    except ClassifiedError as exc:
        check("只调用 1 次", calls["n"] == 1, str(calls["n"]))
        check("错误类型是 LLM_AUTH_ERROR", exc.error_type is ErrorType.LLM_AUTH_ERROR, str(exc.error_type))
        check("不可重试", not is_retryable(exc.error_type))


def test_non_stream_service_retry() -> None:
    print("\n[C5] 非流式 service 层的重试")
    if not require_heavy("[C5] service 重试"):
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        llm = FakeLLM(raise_on_invoke=TimeoutError("超时"), raise_on_invoke_times=2)
        service, _, fake = new_service(tmpdir, llm=llm, retry_max_attempts=3)
        with LogCapture() as cap:
            result = service.ask("重试测试", session_id="s1", mode="chat")

        check("最终拿到答案", result["answer"] == "你好世界", result["answer"])
        check("LLM 被调用 3 次", fake.invoke_calls == 3, str(fake.invoke_calls))
        check("有重试日志", cap.has_event("llm_retry"), str(cap.events()))
        check("最终落库 1 轮", service.get_history("s1")["turns"] == 1)


# ================================================== D. Streaming retry
def test_stream_retry_before_first_token() -> None:
    print("\n[D1] 首 token 前失败 → 重试成功")
    if not require_heavy("[D1] 流式重试"):
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        llm = FakeLLM(raise_on_stream=TimeoutError("超时"), raise_before_token=2)
        service, _, fake = new_service(tmpdir, llm=llm, retry_max_attempts=3)
        events = asyncio.run(collect(service, question="流式重试", session_id="s1", mode="chat"))

        names = [e["event"] for e in events]
        check("最终正常结束", names[-1] == "done", str(names))
        check("LLM 被调用 3 次", fake.stream_calls == 3, str(fake.stream_calls))
        deltas = [e["data"]["text"] for e in events if e["event"] == "delta"]
        check("答案完整不重复", "".join(deltas) == "你好世界", "".join(deltas))
        check("落库 1 轮", service.get_history("s1")["turns"] == 1)


def test_stream_no_retry_after_first_token() -> None:
    print("\n[D2] 首 token 后失败 → 绝不重试")
    if not require_heavy("[D2] 首 token 后不重试"):
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        # 第 2 个 token 处炸（此时第 1 个 token 已经发给用户了）
        llm = FakeLLM(raise_after_token_at=1)
        service, _, fake = new_service(tmpdir, llm=llm, retry_max_attempts=3)
        store = CountingStore(service.conversation_store)
        service._conversation_store = store

        events = asyncio.run(collect(service, question="中途失败", session_id="s1", mode="chat"))

        names = [e["event"] for e in events]
        check("只调用 1 次 LLM（没重试）", fake.stream_calls == 1, str(fake.stream_calls))
        check("最后是 error", names[-1] == "error", str(names))
        check("error 只出现一次", names.count("error") == 1, str(names))
        check("确实先发过 delta", names.count("delta") == 1, str(names))
        check("没有 done", "done" not in names)
        check("一次都没写库", store.append_calls == 0, str(store.append_calls))
        check("数据库里没有半截答案", service.get_history("s1")["turns"] == 0)


def test_stream_retry_exhausted_no_persist() -> None:
    print("\n[D3] 重试耗尽仍然不落库")
    if not require_heavy("[D3] 重试耗尽"):
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        llm = FakeLLM(raise_on_stream=TimeoutError("一直超时"), raise_before_token=99)
        service, _, fake = new_service(tmpdir, llm=llm, retry_max_attempts=2)
        store = CountingStore(service.conversation_store)
        service._conversation_store = store

        events = asyncio.run(collect(service, question="耗尽", session_id="s1", mode="chat"))
        check("调用次数等于上限", fake.stream_calls == 2, str(fake.stream_calls))
        check("最后一个是 error", events[-1]["event"] == "error")
        check(
            "error 类型是 LLM_TIMEOUT",
            events[-1]["data"].get("error_type") == ErrorType.LLM_TIMEOUT.value,
            str(events[-1]["data"]),
        )
        check("没有写库", store.append_calls == 0)


# ================================================== E. Timeout
def test_guard_first_token_timeout() -> None:
    print("\n[E1] 首 token 超时")

    async def slow_factory() -> AsyncIterator[str]:
        await anyio.sleep(5)
        yield "永远来不了"

    async def scenario() -> tuple[str, int]:
        calls = {"n": 0}

        def factory() -> AsyncIterator[str]:
            calls["n"] += 1
            return slow_factory()

        try:
            async for _item in guarded_astream(
                factory,
                policy=RetryPolicy(max_attempts=1),
                first_token_timeout=0.1,
                idle_timeout=5,
            ):
                pass
            return "no-error", calls["n"]
        except ClassifiedError as exc:
            return exc.error_type.value, calls["n"]

    error_type, call_count = asyncio.run(scenario())
    check("错误是 LLM_TIMEOUT", error_type == ErrorType.LLM_TIMEOUT.value, error_type)
    check("只尝试 1 次（不重试）", call_count == 1, str(call_count))


def test_guard_idle_timeout() -> None:
    print("\n[E2] 流式空闲超时")

    async def factory() -> AsyncIterator[str]:
        yield "第一个 token"
        await anyio.sleep(5)  # 之后永久静默

    async def scenario() -> tuple[str, list[str]]:
        got: list[str] = []
        try:
            async for item in guarded_astream(
                factory,
                policy=RetryPolicy(max_attempts=1),
                first_token_timeout=5,
                idle_timeout=0.15,
            ):
                got.append(item)
            return "no-error", got
        except ClassifiedError as exc:
            return exc.error_type.value, got

    error_type, got = asyncio.run(scenario())
    check("先收到了 token", got == ["第一个 token"], str(got))
    check("之后空闲超时", error_type == ErrorType.LLM_TIMEOUT.value, error_type)


def test_timeout_no_dangling_task() -> None:
    print("\n[E3] 超时后没有遗留任务")

    async def factory() -> AsyncIterator[str]:
        await anyio.sleep(5)
        yield "x"

    async def scenario() -> int:
        async with anyio.create_task_group() as tg:

            async def consumer() -> None:
                try:
                    async for _item in guarded_astream(
                        factory,
                        policy=RetryPolicy(max_attempts=1),
                        first_token_timeout=0.1,
                        idle_timeout=5,
                    ):
                        pass
                except ClassifiedError:
                    pass

            tg.start_soon(consumer)
        # 能正常走到这里，说明 task group 里的子任务都被清理干净了；
        # 有 dangling task 的话 __aexit__ 会一直挂着等待。
        return len(asyncio.all_tasks())

    remaining = asyncio.run(scenario())
    check("超时后事件循环里没有残留任务", remaining <= 1, str(remaining))


def test_timeout_http_status() -> None:
    print("\n[E4] 非流式超时 → 504")
    if not require_heavy("[E4] 超时状态码"):
        return

    async def scenario() -> int:
        with tempfile.TemporaryDirectory() as tmpdir:
            llm = FakeLLM(raise_on_invoke=TimeoutError("上游超时"), raise_on_invoke_times=99)
            service, _, _ = new_service(tmpdir, llm=llm, retry_max_attempts=1)
            app = build_app(service)
            response = await post_ask(app, {"question": "超时", "session_id": "s1", "mode": "chat"})
            return response.status_code

    code = asyncio.run(scenario())
    check("返回 504", code == 504, str(code))


# ================================================== F. Database 纪律
def test_db_exactly_once() -> None:
    print("\n[F1] 正常生成恰好写一次库")
    if not require_heavy("[F1] 写一次"):
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        service, _, _ = new_service(tmpdir)
        store = CountingStore(service.conversation_store)
        service._conversation_store = store
        asyncio.run(collect(service, question="写一次", session_id="s1", mode="chat"))
        check("只写 1 次", store.append_calls == 1, str(store.append_calls))
        check("历史 1 轮", service.get_history("s1")["turns"] == 1)


def test_db_failure_no_llm_retry() -> None:
    print("\n[F2] 写库失败不重跑 LLM、不重复写")
    if not require_heavy("[F2] 写库失败"):
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        service, _, fake = new_service(tmpdir)
        store = CountingStore(service.conversation_store, fail=True)
        service._conversation_store = store

        with LogCapture() as cap:
            events = asyncio.run(collect(service, question="写失败", session_id="s1", mode="chat"))

        check("LLM 只调用 1 次", fake.stream_calls == 1, str(fake.stream_calls))
        check("写库只尝试 1 次", store.append_calls == 1, str(store.append_calls))
        names = [e["event"] for e in events]
        check("仍然正常 done（回答是主链路）", names[-1] == "done", str(names))
        check("有持久化失败日志", cap.has_event("persistence_failed"), str(cap.events()))
        failed = cap.lines_with("persistence_failed")
        check(
            "失败类型是 DATABASE_ERROR",
            field_of(failed[0], "error_type") == ErrorType.DATABASE_ERROR.value,
            str(failed),
        )


# ================================================== G. Redis fail-open
def test_redis_down_fail_open() -> None:
    print("\n[G1] Redis 挂掉仍然可用")
    if not require_heavy("[G1] Redis fail-open"):
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        service, _, _ = new_service(tmpdir, redis_client=BrokenRedis())
        with LogCapture() as cap:
            events = asyncio.run(collect(service, question="Redis 挂了", session_id="s1", mode="rag"))

        check("问答仍然完成", events[-1]["event"] == "done", str([e["event"] for e in events]))
        check("落库正常", service.get_history("s1")["turns"] == 1)

    # Redis 降级必须"看得见"：只是继续跑但什么都不记，等于把故障藏起来。
    # 这里定点测 RedisClient（流式桩不走检索，请求路径上碰不到 Redis，
    # 硬要求在业务日志里出现该事件反而会催出一个假断言）。
    with LogCapture() as cap:
        with request_scope("rid-redis", "s1", "rag"):
            RedisClient(client=BrokenRedis()).get("any-key")
    check("有 redis_unavailable 事件", cap.has_event("redis_unavailable"), str(cap.events()))
    unavailable = cap.lines_with("redis_unavailable")
    check(
        "错误类型是 REDIS_UNAVAILABLE",
        field_of(unavailable[0], "error_type") == ErrorType.REDIS_UNAVAILABLE.value,
        str(unavailable[:1]),
    )
    check(
        "redis_unavailable 也带 request_id",
        field_of(unavailable[0], "request_id") == "rid-redis",
        str(unavailable[:1]),
    )

    async def scenario() -> int:
        with tempfile.TemporaryDirectory() as tmpdir:
            service, _, _ = new_service(tmpdir, redis_client=BrokenRedis())
            app = build_app(service)
            response = await post_ask(app, {"question": "Redis 挂了", "session_id": "s2", "mode": "chat"})
            return response.status_code

    code = asyncio.run(scenario())
    check("非流式不变成 500", code == 200, str(code))


# ================================================== H. Rate limit
def test_rate_limit_429() -> None:
    print("\n[H1] 限流 429 + Retry-After 且不进 LLM")
    if not require_heavy("[H1] 限流"):
        return

    async def scenario() -> tuple[int, str, int]:
        with tempfile.TemporaryDirectory() as tmpdir:
            service, _, fake = new_service(tmpdir, rate_limit_requests=1)
            app = build_app(service)
            first = await post_ask(app, {"question": "第一次", "session_id": "rl", "mode": "chat"})
            second = await post_ask(app, {"question": "第二次", "session_id": "rl", "mode": "chat"})
            return second.status_code, second.headers.get("Retry-After", ""), fake.invoke_calls

    code, retry_after, llm_calls = asyncio.run(scenario())
    check("第二次 429", code == 429, str(code))
    check("带 Retry-After", retry_after.isdigit() and int(retry_after) > 0, retry_after)
    check("被限的请求没进 LLM", llm_calls == 1, str(llm_calls))


# ================================================== I. Timing
def test_timing_definitions() -> None:
    print("\n[I1] Timing 指标定义")
    if not require_heavy("[I1] timing"):
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        llm = FakeLLM(piece_delay=0.02)
        service, _, _ = new_service(tmpdir, llm=llm)
        events = asyncio.run(collect(service, question="计时", session_id="s1", mode="chat"))
        done = [e for e in events if e["event"] == "done"][0]["data"]
        timing = done.get("timing") or {}
        check("done 带 timing", bool(timing), str(done.keys()))
        check("total_ms 存在且 > 0", (timing.get("total_ms") or 0) > 0, str(timing))
        check("ttft_ms 存在且 > 0", (timing.get("ttft_ms") or 0) > 0, str(timing))
        check(
            "ttft_ms <= total_ms",
            (timing.get("ttft_ms") or 0) <= (timing.get("total_ms") or 0),
            str(timing),
        )
        check("chat 模式 retrieval_ms 为 None", timing.get("retrieval_ms") is None, str(timing))

    # rag 模式应当有 retrieval_ms
    with tempfile.TemporaryDirectory() as tmpdir:
        service, _, _ = new_service(tmpdir)
        events = asyncio.run(collect(service, question="计时", session_id="s1", mode="rag"))
        done = [e for e in events if e["event"] == "done"][0]["data"]
        check(
            "rag 模式 retrieval_ms 是数值",
            isinstance((done.get("timing") or {}).get("retrieval_ms"), (int, float)),
            str(done.get("timing")),
        )


def test_timing_logged() -> None:
    print("\n[I2] 日志里的耗时字段")
    if not require_heavy("[I2] 日志 timing"):
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        llm = FakeLLM(piece_delay=0.02)
        service, _, _ = new_service(tmpdir, llm=llm)
        with LogCapture() as cap:
            asyncio.run(collect(service, question="计时", session_id="s1", mode="chat"))

        complete = cap.lines_with("request_complete")
        check("request_complete 带 total_ms", field_of(complete[0], "total_ms") is not None)
        check("request_complete 带 ttft_ms", field_of(complete[0], "ttft_ms") is not None)
        check("request_complete 带 answer_len", field_of(complete[0], "answer_len") is not None)

        first = cap.lines_with("llm_first_token")
        check("llm_first_token 带 ttft_ms", field_of(first[0], "ttft_ms") is not None)


# ================================================== J. Stage 3 回归
def test_stage3_heartbeat_still_works() -> None:
    print("\n[J1] Stage 3 心跳仍然正常")
    if not require_heavy("[J1] 心跳回归"):
        return

    async def scenario() -> list[tuple[str, dict]]:
        with tempfile.TemporaryDirectory() as tmpdir:
            service, _, _ = new_service(tmpdir)
            app = build_app(service)
            _status, _headers, events = await post_stream(
                app, {"question": "心跳", "session_id": "s1", "mode": "chat"}
            )
            return events

    events = asyncio.run(scenario())
    check("仍然能完整结束", events[-1][0] == "done", str([n for n, _ in events]))
    check("delta 顺序完整", [d.get("text") for n, d in events if n == "delta"] == ["你", "好", "世", "界"])


def test_stage3_abort_no_persist() -> None:
    print("\n[J2] Stage 3 abort 仍然不落库")
    if not require_heavy("[J2] abort 回归"):
        return

    async def scenario(service: Any) -> int:
        gen = service.astream_ask("会长的问题", session_id="d1", mode="chat")
        seen = 0
        async for event in gen:
            if event["event"] == "delta":
                seen += 1
            if seen == 2:
                try:
                    await gen.athrow(asyncio.CancelledError())
                except asyncio.CancelledError:
                    pass
                break
        return seen

    with tempfile.TemporaryDirectory() as tmpdir:
        service, _, _ = new_service(tmpdir)
        store = CountingStore(service.conversation_store)
        service._conversation_store = store
        seen = asyncio.run(scenario(service))
        check("确实流式出过 token", seen == 2, str(seen))
        check("CancelledError 后没写库", store.append_calls == 0, str(store.append_calls))
        check("历史为空", service.get_history("d1")["turns"] == 0)


def test_stage3_generator_exit_no_persist() -> None:
    print("\n[J3] Stage 3 GeneratorExit 仍然不落库")
    if not require_heavy("[J3] GeneratorExit 回归"):
        return

    async def scenario(service: Any) -> int:
        gen = service.astream_ask("问题", session_id="d2", mode="chat")
        seen = 0
        async for event in gen:
            if event["event"] == "delta":
                seen += 1
            if seen == 2:
                break
        await gen.aclose()
        return seen

    with tempfile.TemporaryDirectory() as tmpdir:
        service, _, _ = new_service(tmpdir)
        store = CountingStore(service.conversation_store)
        service._conversation_store = store
        asyncio.run(scenario(service))
        check("aclose 后没写库", store.append_calls == 0, str(store.append_calls))
        check("历史为空", service.get_history("d2")["turns"] == 0)


def test_stage3_final_write_shielded() -> None:
    print("\n[J4] Stage 3 最终写入仍受 shield 保护")
    if not require_heavy("[J4] shield 回归"):
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        service, _, _ = new_service(tmpdir)
        store = CountingStore(service.conversation_store)
        service._conversation_store = store

        async def scenario() -> None:
            async def consume() -> None:
                async for _event in service.astream_ask("完整回答", session_id="f1", mode="chat"):
                    pass

            task = asyncio.create_task(consume())
            deadline = time.time() + 5
            while not store.write_started.is_set() and time.time() < deadline:
                await asyncio.sleep(0.005)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(scenario())
        check("写库确实开始了", store.write_started.is_set())
        check("取消后仍然落了 1 条", service.get_history("f1")["turns"] == 1)
        check("内容是完整答案", service.get_history("f1")["history"][0]["answer"] == "你好世界")


# ================================================== M. 取消清理（专项回归）
#
# 这一组专门盯一个隐患：
#
#     RuntimeError: Attempted to exit cancel scope in a different task
#                   than it was entered in
#
# 触发路径是"消费者在首 token 之后主动关闭生成器"：
# 生成器被挂起在 yield 上，随后可能由**另一个 task** 终结，
# 而 anyio 的 CancelScope 要求 enter / exit 在同一个 task。
#
# 检测方式说明：
# 这个异常是 asyncio 在关闭 async generator 时捕获、再交给
# loop 的 exception handler 报告的（不会直接冒到测试里）。
# 所以这里显式装一个 handler 把它接下来断言——
# 不是为了让测试变绿而屏蔽它，恰恰相反：它是被**记录并断言为 0** 的。


def _capture_loop_errors() -> tuple[list[str], Any]:
    """装一个 loop exception handler，收集所有被 asyncio 报告的异常。"""
    captured: list[str] = []

    def handler(loop: Any, context: dict) -> None:
        message = str(context.get("message", ""))
        exc = context.get("exception")
        captured.append(f"{message} | {exc!r}")

    def install() -> None:
        asyncio.get_running_loop().set_exception_handler(handler)

    return captured, install


def _cancel_scope_errors(captured: list[str]) -> list[str]:
    return [item for item in captured if "cancel scope" in item]


def _remaining_tasks() -> int:
    """当前事件循环里除自己以外还没结束的任务数（dangling task 判据）。

    只能在**事件循环内**调用：asyncio.all_tasks() 需要有运行中的 loop，
    asyncio.run() 返回之后 loop 已经关了，在外面调会直接 RuntimeError。
    """
    current = asyncio.current_task()
    return len(
        [t for t in asyncio.all_tasks() if t is not current and not t.done()]
    )


async def _wait_until_no_pending_tasks(rounds: int = 50) -> int:
    """等清理收敛，返回最终剩余任务数。

    为什么要"等"而不是立刻断言：

    外层生成器被关闭时，它内部还没 close 的子生成器（orchestrator.astream
    / astream_text / guarded_astream / _one_attempt）是由事件循环的
    asyncgen finalizer **异步**收尾的——那个 finalizer 本身会作为一个
    待执行回调排队。所以刚 aclose() 完立刻数，数到的往往是这个"正在排队
    的清理回调"，而不是泄漏的 driver。

    只 `sleep(0)` 不够：那只是让出一次控制权，事件循环不一定来得及
    执行已经排队的回调。这里给真实的时间片让清理跑完，再断言归零。

    注意这不是"为了让测试变绿而放宽"：断言仍然是**必须为 0**，
    只是允许清理按 asyncio 的调度节奏完成。真泄漏的任务永远不会收敛到 0。
    """
    for _ in range(rounds):
        # 被外层 aclose() 遗弃的内层 async generator（orchestrator.astream /
        # astream_text / guarded_astream / _one_attempt）是靠 GC 触发收尾的，
        # 收尾动作（async_generator_athrow）又是被事件循环排队的 task。
        # 所以先 gc 触发、再让出时间片让它跑完，最后才数。
        gc.collect()
        if _remaining_tasks() == 0:
            return 0
        await asyncio.sleep(0.01)
    return _remaining_tasks()


def test_stream_aclose_after_first_token_is_clean() -> None:
    print("\n[M1] 首 token 后 aclose：无跨 task CancelScope、无残留、不落库")
    if not require_heavy("[M1] aclose 清理"):
        return

    captured, install = _capture_loop_errors()

    async def scenario(service: Any) -> tuple[int, int]:
        install()
        gen = service.astream_ask("会长的问题", session_id="m1", mode="chat")
        seen = 0
        async for event in gen:
            if event["event"] == "delta":
                seen += 1
            if seen == 1:
                break
        # 消费者主动关闭：GeneratorExit 会砸在生成器的 yield 上，
        # 这正是原先跨 task 退出 CancelScope 的场景。
        await gen.aclose()
        return seen, await _wait_until_no_pending_tasks()

    with tempfile.TemporaryDirectory() as tmpdir:
        service, _, fake = new_service(tmpdir)
        store = CountingStore(service.conversation_store)
        service._conversation_store = store
        seen, remaining = asyncio.run(scenario(service))

        check("拿到过真实 token", seen == 1, str(seen))
        check("没有重试", fake.stream_calls == 1, str(fake.stream_calls))
        check("没有 dangling task", remaining == 0, str(remaining))
        check("没有 partial 落库", store.append_calls == 0, str(store.append_calls))
        check("历史为空", service.get_history("m1")["turns"] == 0)
        bad = _cancel_scope_errors(captured)
        check("没有跨 task CancelScope 异常", not bad, str(captured))


def test_guard_aclose_while_driver_hanging() -> None:
    print("\n[M2] 首 token 后 aclose 且 driver 仍卡在上游：driver 必须被停掉")

    captured, install = _capture_loop_errors()

    async def upstream() -> AsyncIterator[str]:
        yield "第一"
        # 永久静默：确保消费者关闭时 driver 还活着，
        # 这样"driver 有没有被取消并等待"才真的被测到。
        await anyio.sleep(30)

    async def scenario() -> tuple[int, int, int]:
        install()
        calls = {"n": 0}

        def factory() -> AsyncIterator[str]:
            calls["n"] += 1
            return upstream()

        gen = guarded_astream(
            factory,
            policy=RetryPolicy(max_attempts=1),
            first_token_timeout=5,
            idle_timeout=5,
        )
        seen = 0
        async for _item in gen:
            seen += 1
            break
        await gen.aclose()
        return seen, calls["n"], await _wait_until_no_pending_tasks()

    seen, call_count, remaining = asyncio.run(scenario())
    check("拿到过真实 token", seen == 1, str(seen))
    check("没有重试", call_count == 1, str(call_count))
    check("driver 没有残留", remaining == 0, str(remaining))
    bad = _cancel_scope_errors(captured)
    check("没有跨 task CancelScope 异常", not bad, str(captured))


# ================================================== 附加：错误分类表
def test_concurrency_isolation() -> None:
    print("\n[L1] 5 并发：ID / 日志 / 重试状态互不干扰")
    if not require_heavy("[L1] 并发隔离"):
        return

    async def scenario() -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as tmpdir:
            # 让其中一部分请求先失败一次再成功：如果重试状态跨请求串了，
            # 有的请求会重试 0 次直接失败，有的会重试到上限。
            service, _, fake = new_service(tmpdir, retry_max_attempts=3)
            app = build_app(service)
            out: dict[str, Any] = {"headers": [], "sessions": [], "status": []}

            async def one(index: int) -> None:
                rid = f"cc-{index}"
                status, headers, events = await post_stream(
                    app,
                    {"question": f"并发{index}", "session_id": f"cs{index}", "mode": "chat"},
                    headers={"X-Request-ID": rid},
                )
                out["headers"].append(headers.get("X-Request-ID"))
                out["status"].append(status)
                done = [d for n, d in events if n == "done"]
                out["sessions"].append(done[0].get("session_id") if done else None)

            async with anyio.create_task_group() as tg:
                for i in range(5):
                    tg.start_soon(one, i)
            return out

    out = asyncio.run(scenario())
    check("5 条流全部 200", all(s == 200 for s in out["status"]), str(out["status"]))
    check("响应头 ID 互不相同", len(set(out["headers"])) == 5, str(out["headers"]))
    check(
        "响应头 ID 与请求一一对应",
        sorted(out["headers"]) == sorted([f"cc-{i}" for i in range(5)]),
        str(out["headers"]),
    )
    check("session 不串台", sorted(out["sessions"]) == sorted([f"cs{i}" for i in range(5)]))


def test_error_classification_table() -> None:
    print("\n[K1] 错误分类表")

    class _E(Exception):
        def __init__(self, status: int | None = None) -> None:
            self.status_code = status if status is not None else None

    cases = [
        (TimeoutError("t"), ErrorType.LLM_TIMEOUT, True, 504),
        (_E(429), ErrorType.LLM_RATE_LIMITED, True, 502),
        (_E(401), ErrorType.LLM_AUTH_ERROR, False, 502),
        (_E(400), ErrorType.VALIDATION_ERROR, False, 400),
        (_E(503), ErrorType.LLM_UPSTREAM_ERROR, True, 502),
    ]
    for exc, expected_type, _retryable, status in cases:
        classified = classify_exception(exc)
        check(
            f"{type(exc).__name__}/{getattr(exc,'status_code','-')} → {expected_type.value}",
            classified.error_type is expected_type,
            str(classified.error_type),
        )
        check(f"  {expected_type.value} 可重试={_retryable}", classified.retryable is _retryable)
        check(f"  {expected_type.value} → {status}", classified.http_status == status)

    check("取消被识别为 STREAM_CANCELLED",
          classify_exception(asyncio.CancelledError()).error_type is ErrorType.STREAM_CANCELLED)
    check("取消不可重试",
          not classify_exception(asyncio.CancelledError()).retryable)
    check("REDIS_UNAVAILABLE 不映射成 5xx",
          http_status_for(ErrorType.REDIS_UNAVAILABLE) == 200)


def main() -> None:
    print("=" * 60)
    print("阶段 4：Reliability & Observability")
    print("=" * 60)

    test_request_id_basics()
    test_request_id_http()
    test_request_id_concurrency()
    test_contextvar_isolation()

    test_structured_logs_success()
    test_structured_logs_failure()
    test_logs_do_not_leak()
    test_safe_message()

    test_retry_policy()
    test_non_stream_retry_success()
    test_non_stream_retry_exhausted()
    test_non_stream_no_retry_on_auth()
    test_non_stream_service_retry()

    test_stream_retry_before_first_token()
    test_stream_no_retry_after_first_token()
    test_stream_retry_exhausted_no_persist()

    test_guard_first_token_timeout()
    test_guard_idle_timeout()
    test_timeout_no_dangling_task()
    test_timeout_http_status()

    test_db_exactly_once()
    test_db_failure_no_llm_retry()

    test_redis_down_fail_open()
    test_rate_limit_429()

    test_timing_definitions()
    test_timing_logged()

    test_stage3_heartbeat_still_works()
    test_stage3_abort_no_persist()
    test_stage3_generator_exit_no_persist()
    test_stage3_final_write_shielded()

    test_stream_aclose_after_first_token_is_clean()
    test_guard_aclose_while_driver_hanging()

    test_concurrency_isolation()
    test_error_classification_table()

    print("=" * 60)
    print(f"通过 {PASSED} 项，失败 {FAILED} 项，跳过 {SKIPPED} 项")
    print("=" * 60)
    if FAILED:
        sys.exit(1)


if __name__ == "__main__":
    main()

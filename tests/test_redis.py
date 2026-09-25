"""Redis 检索缓存 + 限流的测试。

    python tests/test_redis.py

两层测试：
A. 默认用 fakeredis 跑全部单元用例——不需要任何 Redis 服务，CI 可直接跑。
   注意 fakeredis 不支持 Lua（连 EVAL 命令都没有），所以这组用例实际
   覆盖了 RateLimiter 的「事务管道回落」分支。
B. 设了 TEST_REDIS_URL（或 REDIS_URL）且能连通时，额外跑一遍真实 Redis
   集成用例——这组才会走到 Lua 分支。连不上时记为跳过，不算失败。

真实 Redis 用例刻意覆盖「A 层覆盖不到」的东西：Lua 脚本路径、
真实 TTL 过期、以及跨两个独立客户端的持久化。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fakeredis  # noqa: E402
import redis  # noqa: E402

from frontend.local_rag.cache.redis_client import RedisClient  # noqa: E402
from frontend.local_rag.config.settings import Settings  # noqa: E402
from frontend.local_rag.db.database import Database  # noqa: E402
from frontend.local_rag.services.rate_limiter import RateLimiter  # noqa: E402
from frontend.local_rag.services.retrieval_cache import (  # noqa: E402
    VERSION_KEY,
    CachedRetrievalStore,
    RetrievalCache,
    query_hash,
)

# 下面这几个 import 会连带拉起 langchain / sentence-transformers 等重型依赖。
# 单独跑 Redis 的 CI job 只装 redis + fakeredis，装不动这些——所以做成可选：
# 缺依赖时相关的服务级用例记为 skip，Redis 本身的用例照常跑。
try:
    from fastapi import HTTPException
    from frontend.local_rag.api.routes import AskRequest, create_router
    from frontend.local_rag.services.knowledge_service import KnowledgeService

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


# ------------------------------------------------------------------------- 桩
class FakeRetriever:
    """记录调用次数的假检索器：用来证明「第二次真的没再检索」。"""

    def __init__(self, results: list[dict] | None = None) -> None:
        self.is_ready = True
        self.calls: list[tuple[str, int]] = []
        self._results = results or [
            {"content": "片段A", "metadata": {"source": "a.md", "retrieval_score": 0.87}},
            {"content": "片段B", "metadata": {"source": "b.md", "retrieval_score": 0.42}},
        ]

    def search(self, query: str, k: int = 4) -> list[dict]:
        self.calls.append((query, k))
        return [dict(item) for item in self._results]

    def add_documents(self, documents: Any) -> int:
        return len(documents)

    def clear(self) -> None:
        self._results = []

    def all_chunks(self) -> list[dict]:
        return [dict(item) for item in self._results]

    @property
    def stats(self) -> dict:
        return {"mode": "fake"}


class FakeOrchestrator:
    """替掉真实编排器，避免单测去加载 embedding 模型 / 调 LLM。"""

    def __init__(self) -> None:
        self.ingested: list[str] = []

    def ingest(self, filename: str, content: bytes) -> dict:
        self.ingested.append(filename)
        return {"filename": filename, "chunk_count": 3}

    def ask(
        self,
        question: str,
        history: list[dict] | None = None,
        mode: str = "rag",
    ) -> dict:
        return {
            "question": question,
            "answer": f"答案-{question}",
            "sources": [],
            "mode": mode,
            "agent_trace": ["retrieval_agent", "generation_agent"],
        }


class FlakyLuaRedis:
    """eval 第一次抛 ConnectionError（模拟网络抖动），之后自己模拟 Lua 语义。

    用来验证一个容易写错的降级逻辑：**临时失败不能永久关掉 Lua**。
    fakeredis 本身不支持 eval，所以这个桩必须自己把脚本语义实现出来，
    才能区分「抖动」和「不支持」这两种同样返回 None 的情况。
    """

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        self.fail_times = 1
        self.eval_calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def eval(self, script: str, numkeys: int, *args: Any) -> list[int]:
        self.eval_calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise redis.exceptions.ConnectionError("transient blip")
        key = args[0]
        window = int(args[numkeys])
        count = int(self._backend.incr(key))
        if count == 1:
            self._backend.expire(key, window)
        return [count, int(self._backend.ttl(key))]


class BrokenRedis:
    """任何操作都抛 ConnectionError 的桩，模拟 Redis 完全不可用。"""

    def __getattr__(self, name: str) -> Any:
        def _raise(*_: Any, **__: Any) -> Any:
            raise redis.exceptions.ConnectionError("redis is down")

        return _raise


def require_heavy(name: str) -> bool:
    """服务级用例需要 langchain 等重型依赖；缺依赖时跳过而不是失败。"""
    if HEAVY_AVAILABLE:
        return True
    skip(name, f"未安装重型依赖（{HEAVY_ERROR}），仅在完整依赖环境下执行")
    return False


def new_redis() -> tuple[RedisClient, Any]:
    """返回一个 fakeredis 支撑的 RedisClient，以及底层实例（便于断言）。"""
    fake = fakeredis.FakeStrictRedis(decode_responses=True)
    return RedisClient(client=fake), fake


def new_cache(**kwargs: Any) -> tuple[RetrievalCache, RedisClient]:
    client, _ = new_redis()
    params: dict[str, Any] = {
        "ttl_seconds": 600,
        "retrieval_mode": "bm25",
        "rewrite_mode": "prf",
        "top_k": 4,
    }
    params.update(kwargs)
    return RetrievalCache(client, **params), client


def make_service(redis_client: RedisClient, db_file: Path) -> KnowledgeService:
    """构造一个不加载重型依赖的 KnowledgeService（编排器与检索器用桩替换）。

    这里测的是「Redis 故障时的降级与 MySQL 持久化」，不是检索算法本身——
    检索算法由 tests/test_retrieval.py 与 test_knowledge_retriever.py 覆盖。
    """
    settings = Settings(
        database_url=f"sqlite:///{db_file}",
        redis_cache_enabled=True,
        redis_cache_ttl_seconds=600,
        rate_limit_enabled=True,
        rate_limit_requests=5,
        rate_limit_window_seconds=60,
        retrieval_mode="bm25",
        retrieval_query_rewrite="prf",
        retrieval_top_k=4,
    )
    database = Database(f"sqlite:///{db_file}")
    database.create_tables()
    service = KnowledgeService(settings, database=database, redis_client=redis_client)
    service._orchestrator = FakeOrchestrator()
    service._retriever = FakeRetriever()
    service._index_loaded = True
    service._ensure_runtime = lambda: service._orchestrator  # type: ignore[method-assign]
    return service


# --------------------------------------------------------------------------- 1
def test_ping() -> None:
    print("\n[1] Redis ping")
    client, _ = new_redis()
    check("正常 Redis ping 返回 True", client.ping() is True)
    check("healthy 属性与 ping 一致", client.healthy is True)
    check("读写往返正常", client.set("k", "v") and client.get("k") == "v")

    broken = RedisClient(client=BrokenRedis())
    check("Redis 不可用 ping 返回 False 而不是抛异常", broken.ping() is False)
    check("Redis 不可用 get 返回 None", broken.get("k") is None)
    check("Redis 不可用 incr 返回 None", broken.incr("c") is None)


# ------------------------------------------------------------------------- 2-4
def test_cache_miss_hit() -> None:
    print("\n[2-4] Cache Miss / Hit / 第二次不再检索")
    cache, _ = new_cache()
    retriever = FakeRetriever()
    store = CachedRetrievalStore(retriever, cache)

    first = store.search("什么是 RAG", k=4)
    check("首次为 miss，结果来自真实检索", len(retriever.calls) == 1)
    check("miss 后结果非空", len(first) == 2)

    second = store.search("什么是 RAG", k=4)
    check("第二次命中缓存，底层检索器没有被再次调用", len(retriever.calls) == 1)
    check("命中结果与原结果一致", second == first)
    check(
        "metrics 记录 1 miss + 1 hit",
        cache.metrics()["cache_miss"] == 1 and cache.metrics()["cache_hit"] == 1,
        str(cache.metrics()),
    )


# --------------------------------------------------------------------------- 5
def test_different_query_no_collision() -> None:
    print("\n[5] 不同 query 不串缓存")
    cache, _ = new_cache()
    retriever = FakeRetriever()
    store = CachedRetrievalStore(retriever, cache)

    store.search("问题甲", k=4)
    store.search("问题乙", k=4)
    check("两个不同 query 各检索一次", len(retriever.calls) == 2)
    check("query hash 不同", query_hash("问题甲") != query_hash("问题乙"))

    # 归一化：多余空白不产生不同的 key
    store.search("  什么是  RAG  ", k=4)
    check(
        "多余空白归一化后命中同一缓存",
        len(retriever.calls) == 3 and query_hash("什么是 RAG") == query_hash("  什么是  RAG  "),
    )


# ------------------------------------------------------------------------ 6-8
def test_key_dimensions() -> None:
    print("\n[6-8] 不同 mode / top_k / rewrite 不串缓存")
    base, _ = new_cache()

    modes = [
        new_cache(retrieval_mode="bm25")[0],
        new_cache(retrieval_mode="dense")[0],
        new_cache(retrieval_mode="hybrid")[0],
    ]
    check(
        "三种 retrieval mode 的 key 互不相同",
        len({m.build_key("q", 4) for m in modes}) == 3,
    )
    check(
        "不同 top_k 的 key 不同",
        base.build_key("q", 4) != base.build_key("q", 8),
    )
    check(
        "不同 rewrite mode 的 key 不同",
        new_cache(rewrite_mode="off")[0].build_key("q", 4)
        != new_cache(rewrite_mode="prf")[0].build_key("q", 4),
    )

    # 功能层面：换个 mode 必须真的 miss
    retriever = FakeRetriever()
    CachedRetrievalStore(retriever, new_cache(retrieval_mode="bm25")[0]).search("q", 4)
    store_hybrid = CachedRetrievalStore(retriever, new_cache(retrieval_mode="hybrid")[0])
    store_hybrid.search("q", 4)
    check("换 retrieval mode 后重新走真实检索", len(retriever.calls) == 2)

    # key 里不出现原始问题原文（避免长文本 / 隐私信息进 key）
    key = base.build_key("这是一个很长很长的问题" * 20, 4)
    check("key 中不含原始问题文本", "这是一个很长" not in key)
    check("key 前缀符合约定", key.startswith("rag:retrieval:v"))


# --------------------------------------------------------------------------- 9
def test_ttl_expiry() -> None:
    print("\n[9] TTL 到期后重新 Miss")
    cache, _ = new_cache(ttl_seconds=1)
    retriever = FakeRetriever()
    store = CachedRetrievalStore(retriever, cache)

    store.search("会过期的问题", k=4)
    store.search("会过期的问题", k=4)
    check("TTL 内命中缓存", len(retriever.calls) == 1)

    time.sleep(1.2)
    store.search("会过期的问题", k=4)
    check("TTL 过期后重新检索", len(retriever.calls) == 2, str(retriever.calls))


# -------------------------------------------------------------------------- 10
def test_version_bump_invalidates() -> None:
    print("\n[10] KB version bump 后旧缓存不再命中")
    cache, client = new_cache()
    retriever = FakeRetriever()
    store = CachedRetrievalStore(retriever, cache)

    before = store.search("上传前的问题", k=4)
    old_key = cache.build_key("上传前的问题", 4)
    check("首次检索并写入缓存", len(retriever.calls) == 1)
    check("缓存已写入", client.get(old_key) is not None)

    cache.bump_knowledge_version()
    check("版本号已递增", int(client.get(VERSION_KEY)) == 2)

    after = store.search("上传前的问题", k=4)
    check("bump 后旧 key 不再命中，重新检索", len(retriever.calls) == 2)
    check("结果仍然正确", after == before)
    check("新 key 与旧 key 不同", cache.build_key("上传前的问题", 4) != old_key)


# ---------------------------------------------------------------------- 11-12
def test_upload_reset_bump() -> None:
    print("\n[11-12] upload / reset 后版本号递增")
    if not require_heavy("[11-12] upload / reset 后版本号递增"):
        return
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        client, fake = new_redis()
        service = make_service(client, Path(tmp) / "svc.db")

        v0 = service.retrieval_cache.current_version()
        service.ingest_upload("doc.md", "# 内容".encode("utf-8"))
        v1 = service.retrieval_cache.current_version()
        check("upload 成功后版本号 +1", v1 == v0 + 1, f"{v0} -> {v1}")

        service.reset()
        v2 = service.retrieval_cache.current_version()
        check("reset 成功后版本号 +1", v2 == v1 + 1, f"{v1} -> {v2}")

        # 失败路径不能 bump：ingest 抛异常时版本号必须保持不变
        def boom(filename: str, content: bytes) -> dict:
            raise ValueError("解析失败")

        service._orchestrator.ingest = boom  # type: ignore[method-assign]
        try:
            service.ingest_upload("bad.md", b"x")
        except ValueError:
            pass
        check(
            "upload 失败时版本号不变",
            service.retrieval_cache.current_version() == v2,
            str(service.retrieval_cache.current_version()),
        )


# -------------------------------------------------------------------------- 13
def test_redis_down_rag_fallback() -> None:
    print("\n[13] Redis 关闭后 RAG 仍然可用")
    broken = RedisClient(client=BrokenRedis())

    retriever = FakeRetriever()
    cache = RetrievalCache(broken, enabled=True, ttl_seconds=60)
    store = CachedRetrievalStore(retriever, cache)
    results = store.search("Redis 挂了还能检索吗", k=4)
    check("Redis 不可用时检索照常返回结果", len(results) == 2)
    check("底层检索器被调用", len(retriever.calls) == 1)
    check(
        "未命中被记为 miss，且没有写进 Redis",
        cache.metrics()["cache_miss"] == 1 and cache.metrics()["cache_write"] == 0,
        str(cache.metrics()),
    )

    # 以下服务级用例需要重型依赖
    if not require_heavy("[13] Redis 关闭后 RAG 问答仍成功（服务级）"):
        return
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        service = make_service(broken, Path(tmp) / "down.db")
        result = service.ask("Redis 挂了还能问吗", session_id="s1", mode="rag")
        check("RAG 问答仍成功", result["answer"].startswith("答案-"))
        check("返回结构不变", {"question", "answer", "sources", "mode"} <= set(result))
        check("mode 仍为 rag", result["mode"] == "rag")

        status = service.redis_status()
        check("status 返回 redis healthy=False", status["healthy"] is False)
        check("status 不抛异常且带 backend 字段", status["backend"] == "redis")


# -------------------------------------------------------------------------- 14
def test_redis_down_chat() -> None:
    print("\n[14] Redis 关闭后 Chat 仍然可用")
    if not require_heavy("[14] Redis 关闭后 Chat 仍然可用"):
        return
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        service = make_service(RedisClient(client=BrokenRedis()), Path(tmp) / "chat.db")
        result = service.ask("你好", session_id="s1", mode="chat")
        check("chat 问答仍成功", result["answer"] == "答案-你好")
        check("mode 为 chat", result["mode"] == "chat")


# -------------------------------------------------------------------------- 15
def test_redis_down_mysql_persistence() -> None:
    print("\n[15] Redis 关闭后会话持久化仍正常")
    if not require_heavy("[15] Redis 关闭后会话持久化仍正常"):
        return
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        db_file = Path(tmp) / "persist.db"
        service = make_service(RedisClient(client=BrokenRedis()), db_file)
        service.ask("第一条", session_id="s1", mode="rag")
        service.ask("第二条", session_id="s1", mode="chat")

        history = service.get_history("s1")
        check("两轮问答都落库", history["turns"] == 2, str(history["turns"]))

        # 换个实例（等价于服务重启）再读，历史必须还在
        from frontend.local_rag.db.models import Message
        from frontend.local_rag.services.conversation_store import ConversationStore

        reopened_db = Database(f"sqlite:///{db_file}")
        reopened = ConversationStore(reopened_db)
        check("重启后历史仍存在", len(reopened.get_history("s1")) == 2)

        # get_history() 只返回 {question, answer}（给 LLM 用的格式），
        # mode 要直接查 Message 表才能验到。
        with reopened_db.session() as session:
            modes = [r.mode for r in session.query(Message).order_by(Message.id).all()]
        check("mode 分别记录为 rag / chat", modes == ["rag", "chat"], str(modes))
        reopened_db.dispose()
        # Redis 挂掉时 status 里的 database 段不受影响
        check("数据库仍健康", service.database_status()["healthy"] is True)


# ---------------------------------------------------------------------- 16-17
def test_rate_limit_basic() -> None:
    print("\n[16-17] 限流：前 N 次通过，第 N+1 次拒绝")
    client, _ = new_redis()
    limiter = RateLimiter(client, enabled=True, limit_requests=3, window_seconds=60)

    results = [limiter.check("session-a") for _ in range(3)]
    check("前 3 次全部放行", all(r.allowed for r in results))
    check("remaining 递减", [r.remaining for r in results] == [2, 1, 0])

    blocked = limiter.check("session-a")
    check("第 4 次被拒绝", blocked.allowed is False)
    check("retry_after > 0", blocked.retry_after > 0, str(blocked.retry_after))
    check("被拒绝时不算 degraded", blocked.degraded is False)
    check("metrics 记录 1 次限流", limiter.metrics()["rate_limited"] == 1)

    if not require_heavy("[17] HTTP 层返回 429"):
        return

    # HTTP 层：路由必须真的返回 429 + Retry-After
    router = create_router(service=object(), rate_limiter=limiter)  # type: ignore[arg-type]
    endpoint = next(r.endpoint for r in router.routes if getattr(r, "path", "") == "/ask")
    try:
        endpoint(AskRequest(question="超限请求", session_id="session-a"))
        http_ok = False
        detail = "没有抛出 HTTPException"
    except HTTPException as exc:
        http_ok = exc.status_code == 429
        detail = f"status={exc.status_code}"
        if http_ok:
            http_ok = str(exc.headers.get("Retry-After", "")) == str(blocked.retry_after)
            detail += f" retry-after={exc.headers.get('Retry-After')}"
    check("HTTP 层返回 429 且带 Retry-After", http_ok, detail)


# -------------------------------------------------------------------------- 18
def test_rate_limit_window_rollover() -> None:
    print("\n[18] 限流窗口到期后恢复")
    client, _ = new_redis()
    limiter = RateLimiter(client, enabled=True, limit_requests=2, window_seconds=1)

    check("第 1 次放行", limiter.check("s").allowed is True)
    check("第 2 次放行", limiter.check("s").allowed is True)
    check("第 3 次被拒", limiter.check("s").allowed is False)

    time.sleep(1.2)
    check("窗口滚动后重新放行", limiter.check("s").allowed is True)


# -------------------------------------------------------------------------- 19
def test_rate_limit_session_isolation() -> None:
    print("\n[19] 不同 session_id 限流互不影响")
    client, _ = new_redis()
    limiter = RateLimiter(client, enabled=True, limit_requests=2, window_seconds=60)

    limiter.check("alice")
    limiter.check("alice")
    check("alice 达到上限后被拒", limiter.check("alice").allowed is False)
    check("bob 不受影响，仍然放行", limiter.check("bob").allowed is True)
    check("bob 第二次仍放行", limiter.check("bob").allowed is True)
    check("bob 第三次才被拒", limiter.check("bob").allowed is False)


# -------------------------------------------------------------------------- 20
def test_rate_limit_fail_open() -> None:
    print("\n[20] Redis 故障时 Rate Limit fail-open")
    limiter = RateLimiter(RedisClient(client=BrokenRedis()), enabled=True,
                          limit_requests=1, window_seconds=60)

    results = [limiter.check("s") for _ in range(5)]
    check("Redis 不可用时全部放行（fail-open）", all(r.allowed for r in results))
    check("放行时标记 degraded", all(r.degraded for r in results))
    check("错误被计数", limiter.metrics()["error"] == 5, str(limiter.metrics()))

    # 限流关闭时同样放行且标记 degraded
    client, _ = new_redis()
    off = RateLimiter(client, enabled=False, limit_requests=1, window_seconds=60)
    r = off.check("s")
    check("限流关闭时放行", r.allowed is True and r.degraded is True)


# -------------------------------------------------------------------------- 21
def test_lua_recovers_after_transient_error() -> None:
    print("\n[21] Lua 瞬时故障后能自动恢复（不被永久禁用）")
    fake = FlakyLuaRedis(fakeredis.FakeStrictRedis(decode_responses=True))
    client = RedisClient(client=fake)
    limiter = RateLimiter(client, enabled=True, limit_requests=5, window_seconds=60)

    first = limiter.check("s")
    check("抖动时仍能完成计数（回落管道）", first.allowed and first.remaining == 4)
    check("抖动不判定为不支持脚本", client.scripting_supported is not False)
    check("Lua 未被永久禁用", limiter._lua_disabled is False)

    second = limiter.check("s")
    check("下一次仍走 Lua", fake.eval_calls == 2, f"eval_calls={fake.eval_calls}")
    check("计数继续递增", second.remaining == 3, str(second.remaining))

    # 对照：真的不支持脚本时（fakeredis 原生 eval 会报 unknown command）
    # 必须永久关闭，避免每个请求都白跑一次网络往返
    plain = RedisClient(client=fakeredis.FakeStrictRedis(decode_responses=True))
    plain_limiter = RateLimiter(plain, enabled=True, limit_requests=5, window_seconds=60)
    plain_limiter.check("s")
    check("确认不支持时脚本标记置 False", plain.scripting_supported is False)
    check("确认不支持后永久走管道", plain_limiter._lua_disabled is True)


# ------------------------------------------------------------ 真实 Redis 集成
def _redis_url() -> str:
    return os.environ.get("TEST_REDIS_URL") or os.environ.get("REDIS_URL") or ""


def test_redis_integration() -> None:
    print("\n[真实 Redis] integration test")
    url = _redis_url()
    if not url:
        skip("真实 Redis 集成用例", "未设置 TEST_REDIS_URL / REDIS_URL")
        return

    client = RedisClient(url)
    try:
        if not client.ping():
            skip("真实 Redis 集成用例", f"无法连接 {url}")
            return
    except Exception as exc:  # noqa: BLE001
        skip("真实 Redis 集成用例", f"连接异常 {exc}")
        return

    marker = f"rag:test:{int(time.time() * 1000)}"
    other: RedisClient | None = None
    try:
        # Lua 分支：fakeredis 覆盖不到，只有真实 Redis 会走到
        limiter = RateLimiter(client, enabled=True, limit_requests=3, window_seconds=30)
        allowed = [limiter.check(marker) for _ in range(3)]
        check("真实 Redis：Lua 路径前 3 次放行", all(r.allowed for r in allowed))
        check("真实 Redis：第 4 次被拒", limiter.check(marker).allowed is False)
        check(
            "真实 Redis：确实用了 Lua 而不是回落",
            limiter._lua_disabled is False,
            "Lua 被禁用说明 EVAL 不可用",
        )

        # 真实 TTL
        short = RateLimiter(client, enabled=True, limit_requests=1, window_seconds=1)
        check("真实 Redis：窗口内第 1 次放行", short.check(marker).allowed is True)
        check("真实 Redis：第 2 次被拒", short.check(marker).allowed is False)
        time.sleep(1.2)
        check("真实 Redis：真实 TTL 过期后恢复", short.check(marker).allowed is True)

        # 缓存：跨两个独立客户端持久化
        cache = RetrievalCache(
            client, enabled=True, ttl_seconds=30,
            retrieval_mode="bm25", rewrite_mode="prf", top_k=4,
        )
        retriever = FakeRetriever()
        store = CachedRetrievalStore(retriever, cache)
        first = store.search(f"{marker}-query", k=4)

        other = RedisClient(url)
        other_cache = RetrievalCache(
            other, enabled=True, ttl_seconds=30,
            retrieval_mode="bm25", rewrite_mode="prf", top_k=4,
        )
        check("真实 Redis：另一个客户端能读到缓存", other_cache.get(cache.build_key(f"{marker}-query", 4)) == first)

        v0 = cache.current_version()
        cache.bump_knowledge_version()
        check("真实 Redis：版本号递增", cache.current_version() == v0 + 1)
        store2 = CachedRetrievalStore(retriever, cache)
        store2.search(f"{marker}-query", k=4)
        check("真实 Redis：bump 后重新检索", len(retriever.calls) == 2)

        # 清理：只删自己写的前缀，不做 FLUSHDB
        client.delete(marker)
    finally:
        client.close()
        if other is not None:
            other.close()


def main() -> None:
    test_ping()
    test_cache_miss_hit()
    test_different_query_no_collision()
    test_key_dimensions()
    test_ttl_expiry()
    test_version_bump_invalidates()
    test_upload_reset_bump()
    test_redis_down_rag_fallback()
    test_redis_down_chat()
    test_redis_down_mysql_persistence()
    test_rate_limit_basic()
    test_rate_limit_window_rollover()
    test_rate_limit_session_isolation()
    test_rate_limit_fail_open()
    test_lua_recovers_after_transient_error()
    test_redis_integration()

    print("\n" + "=" * 46)
    print(f"通过 {PASSED} 项，失败 {FAILED} 项，跳过 {SKIPPED} 项")
    print("=" * 46)
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()

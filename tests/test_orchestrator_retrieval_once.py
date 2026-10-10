"""流式路径检索次数回归测试（Stage 5 缺陷修复）。

为什么要有这个测试：
压测时发现 `AgentOrchestrator.astream()` 里检索被执行了**两次**——
第一次的返回值被第二次覆盖，纯属浪费；更糟的是它污染了缓存归因：
首次查询走 miss（cache_miss +1），第二次命中刚写入的缓存（cache_hit +1），
于是 `timing.cache_hit` 对**首次查询**也报 True。

这类 bug 的隐蔽之处在于：功能完全正确，只是慢一倍 + 指标错。
单看输出永远发现不了，所以必须用"计数"把它钉死。

刻意不做什么：
- 不起服务、不连 Redis、不调真实 LLM。用 `object.__new__` 绕开
  AgentOrchestrator.__init__（它会去构造 Settings / DocumentProcessor，
  需要 .env 和磁盘），只替换本用例关心的两个 agent。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import anyio  # noqa: E402

from frontend.local_rag.core.agents.base import AgentResult  # noqa: E402
from frontend.local_rag.core.agents.orchestrator import (  # noqa: E402
    AgentOrchestrator,
)
from frontend.local_rag.observability.errors import (  # noqa: E402
    ClassifiedError,
    ErrorType,
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


# ---------------------------------------------------------------- 测试替身


class CountingRetrievalAgent:
    """记录 run() 被调用了几次的检索替身。"""

    name = "retrieval_agent"

    def __init__(self, sources=None, raises: BaseException | None = None) -> None:
        self.calls = 0
        self.questions: list[str] = []
        self._sources = sources if sources is not None else [{"id": "c1", "text": "x"}]
        self._raises = raises

    def run(self, question: str = "", top_k: int | None = None, **_: object) -> AgentResult:
        self.calls += 1
        self.questions.append(question)
        if self._raises is not None:
            raise self._raises
        return AgentResult(
            agent=self.name,
            success=True,
            message="ok",
            data={"sources": self._sources, "ready": True},
        )


class StubGenerationAgent:
    """逐块吐文本的生成替身，不碰网络。"""

    name = "generation_agent"

    def __init__(self, pieces=None) -> None:
        self._pieces = pieces if pieces is not None else ["你", "好"]

    async def astream_text(self, **_: object):
        for piece in self._pieces:
            yield piece


def make_orchestrator(retrieval: CountingRetrievalAgent) -> AgentOrchestrator:
    """只装配本用例需要的两个 agent，绕开 __init__ 的重依赖。"""
    orch = object.__new__(AgentOrchestrator)
    orch.retrieval_agent = retrieval
    orch.generation_agent = StubGenerationAgent()
    return orch


async def collect(orch: AgentOrchestrator, **kwargs) -> list[dict]:
    events = []
    async for event in orch.astream(**kwargs):
        events.append(event)
    return events


# ---------------------------------------------------------------- 用例


async def main() -> None:
    print("\n[A] rag 模式：检索必须恰好执行一次")

    r = CountingRetrievalAgent()
    events = await collect(make_orchestrator(r), question="什么是 RAG？", mode="rag")
    check("rag 模式检索调用次数 == 1", r.calls == 1, f"实际 {r.calls}")

    src_events = [e for e in events if e.get("event") == "sources"]
    check("恰好产出 1 个 sources 事件", len(src_events) == 1, f"实际 {len(src_events)}")
    if src_events:
        got = src_events[0]["data"]["sources"]
        check("sources 用的是检索结果（未被丢弃）", got == r._sources, str(got))
        check("sources 带 retrieval_ms", "retrieval_ms" in src_events[0]["data"])

    deltas = [e for e in events if e.get("event") == "delta"]
    check("delta 事件数量 == 生成块数（2）", len(deltas) == 2, f"实际 {len(deltas)}")

    print("\n[B] 事件顺序：sources 必须早于 delta")

    names = [e.get("event") for e in events]
    check("事件顺序 == sources, trace, delta, delta",
          names == ["sources", "trace", "delta", "delta"], str(names))

    print("\n[C] chat 模式：完全不检索")

    r2 = CountingRetrievalAgent()
    events2 = await collect(make_orchestrator(r2), question="你好", mode="chat")
    check("chat 模式检索调用次数 == 0", r2.calls == 0, f"实际 {r2.calls}")
    check("chat 模式无 sources 事件",
          not [e for e in events2 if e.get("event") == "sources"])
    check("chat 模式仍有 delta", len([e for e in events2 if e.get("event") == "delta"]) == 2)

    print("\n[D] 多次调用互不串味")

    r3 = CountingRetrievalAgent()
    orch3 = make_orchestrator(r3)
    await collect(orch3, question="q1", mode="rag")
    await collect(orch3, question="q2", mode="rag")
    check("两次流式请求共检索 2 次（不是 4 次）", r3.calls == 2, f"实际 {r3.calls}")
    check("每次都传了各自的问题", r3.questions == ["q1", "q2"], str(r3.questions))

    print("\n[E] 检索异常仍要正确归类（不能被吞）")

    r4 = CountingRetrievalAgent(raises=RuntimeError("索引损坏"))
    err = None
    try:
        await collect(make_orchestrator(r4), question="q", mode="rag")
    except ClassifiedError as exc:
        err = exc
    check("RuntimeError 被包装成 ClassifiedError", err is not None, "未抛出")
    if err is not None:
        check("错误类型 == RETRIEVAL_ERROR",
              err.error_type == ErrorType.RETRIEVAL_ERROR, str(err.error_type))
    check("失败路径也只检索 1 次", r4.calls == 1, f"实际 {r4.calls}")

    r5 = CountingRetrievalAgent(
        raises=ClassifiedError(ErrorType.RETRIEVAL_ERROR, "已经是归类过的错误")
    )
    err5 = None
    try:
        await collect(make_orchestrator(r5), question="q", mode="rag")
    except ClassifiedError as exc:
        err5 = exc
    check("已归类的 ClassifiedError 原样抛出（不二次包装）",
          err5 is not None and err5.message == "已经是归类过的错误",
          str(err5))

    print("\n[F] 非法 mode 早失败")

    r6 = CountingRetrievalAgent()
    rejected = False
    try:
        await collect(make_orchestrator(r6), question="q", mode="nope")
    except ValueError:
        rejected = True
    check("非法 mode 抛 ValueError", rejected)
    check("非法 mode 不触发检索", r6.calls == 0, f"实际 {r6.calls}")

    print("\n[G] 缓存归因：一次请求只能 +1 一个计数器")

    # 修复前：astream 检索两次 -> miss+1 且 hit+1 -> cache_hit 恒为 True。
    # 修复后：一次请求只可能 hit+1 或 miss+1，below 模拟这两种可达状态。
    from frontend.local_rag.services.knowledge_service import KnowledgeService

    svc = object.__new__(KnowledgeService)

    class StubCache:
        def __init__(self, metrics):
            self._m = metrics

        def metrics(self):
            return dict(self._m)

    class BrokenCache:
        def metrics(self):
            raise RuntimeError("redis down")

    def with_cache(cache) -> None:
        # retrieval_cache 是懒加载 property，直接赋值会被拒，
        # 所以写它背后的私有字段。
        svc._retrieval_cache = cache

    with_cache(StubCache({"cache_hit": 1, "cache_miss": 0}))
    check("命中时 cache_hit == True",
          svc._cache_hit_since({"cache_hit": 0, "cache_miss": 0}) is True)

    with_cache(StubCache({"cache_hit": 0, "cache_miss": 1}))
    check("未命中时 cache_hit == False",
          svc._cache_hit_since({"cache_hit": 0, "cache_miss": 0}) is False)

    with_cache(StubCache({"cache_hit": 0, "cache_miss": 0}))
    check("无检索（chat）时 cache_hit == None",
          svc._cache_hit_since({"cache_hit": 0, "cache_miss": 0}) is None)

    check("拿不到快照时 cache_hit == None", svc._cache_hit_since(None) is None)

    with_cache(BrokenCache())
    check("缓存指标取不到时返回 None（不炸主链路）",
          svc._cache_hit_since({"cache_hit": 0, "cache_miss": 0}) is None)


if __name__ == "__main__":
    anyio.run(main)
    print(f"\n{'=' * 56}")
    print(f"通过 {PASSED} 项，失败 {FAILED} 项")
    print("=" * 56)
    sys.exit(1 if FAILED else 0)

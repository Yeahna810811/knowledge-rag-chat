"""进程内探针：数一次流式请求到底触发了几次检索（Stage 5 缺陷修复验证）。

为什么不用延迟来衡量：
压测的端到端延迟噪声太大（同一份代码 ask 模式 c=20 的 P50 能在
77ms 和 98ms 之间跳，±20%），单次跑出来的"修复收益"根本不可信。

但**检索次数**是整数计数器，没有噪声：
- 修复前：一次流式请求 = cache_miss +1 且 cache_hit +1（检索两次，
  第一次 miss 写缓存，第二次立刻命中刚写进去的那条）
- 修复后：一次全新问题 = cache_miss +1、cache_hit +0

计数不会撒谎，所以用它当确定性证据。

用法（进程内起服务，约需几分钟 import）：
    PERF_WORK_DIR=/tmp/rag-perf-work python evaluation/performance/probe_retrieval_count.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import anyio  # noqa: E402

from evaluation.performance.fake_app import build_service  # noqa: E402


def snapshot(service) -> dict:
    m = service.cache_metrics()["cache"]
    return {k: int(m.get(k, 0)) for k in ("cache_hit", "cache_miss", "cache_write", "cache_error")}


async def drain(service, question: str, session: str) -> None:
    async for _event in service.astream_ask(question, session_id=session, mode="rag"):
        pass


async def main() -> None:
    print("[probe] 构建服务（import 重，请耐心）...", flush=True)
    service = build_service()
    print("[probe] 服务就绪\n", flush=True)

    print("=" * 64)
    print("场景 1：3 个全新问题，各自流式问一次（预期 miss+3 / hit+0）")
    print("=" * 64)
    before = snapshot(service)
    for i in range(3):
        await drain(service, f"探针全新问题 {i} 关于检索缓存的计数行为", session=f"probe-{i}")
    after = snapshot(service)
    d = {k: after[k] - before[k] for k in before}
    print(f"  delta = {d}")
    ok1 = d["cache_miss"] == 3 and d["cache_hit"] == 0
    print(f"  -> {'✅ 每个新问题只检索一次' if ok1 else '❌ 仍然重复检索（hit 应为 0）'}\n")

    print("=" * 64)
    print("场景 2：重复问同一个问题 2 次（预期 hit+2 / miss+0）")
    print("=" * 64)
    q = "重复提问用来验证缓存命中的问题"
    await drain(service, q, session="probe-repeat")  # 第一次：写入
    before = snapshot(service)
    for _ in range(2):
        await drain(service, q, session="probe-repeat")
    after = snapshot(service)
    d2 = {k: after[k] - before[k] for k in before}
    print(f"  delta = {d2}")
    ok2 = d2["cache_hit"] == 2 and d2["cache_miss"] == 0
    print(f"  -> {'✅ 重复提问全部命中缓存' if ok2 else '❌ 缓存未命中'}\n")

    print("=" * 64)
    print("场景 3：非流式 ask() 对照（本来就只检索一次，应 miss+1 / hit+0）")
    print("=" * 64)
    before = snapshot(service)
    service.ask("非流式路径的对照问题", session_id="probe-sync", mode="rag")
    after = snapshot(service)
    d3 = {k: after[k] - before[k] for k in before}
    print(f"  delta = {d3}")
    ok3 = d3["cache_miss"] == 1 and d3["cache_hit"] == 0
    print(f"  -> {'✅ 同步路径只检索一次' if ok3 else '❌ 同步路径异常'}\n")

    print("=" * 64)
    print(f"结论：{'全部通过' if (ok1 and ok2 and ok3) else '存在失败项'}")
    print("=" * 64)
    sys.exit(0 if (ok1 and ok2 and ok3) else 1)


if __name__ == "__main__":
    anyio.run(main)

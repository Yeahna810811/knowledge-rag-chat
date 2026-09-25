"""Server-Sent Events 帧构造与流包装。

为什么手写而不引 sse-starlette：
    这个模块要做的事只有三件——把事件格式化成 `event:` / `data:` 帧、
    在链路静默时插心跳、以及在客户端断开时让生成器收到取消。
    加起来不到 120 行，语义完全可控（尤其是「心跳帧必须是注释帧，
    不能被前端当成业务事件」这种细节），为一个已经验收过的工程引入
    新依赖不划算。

帧格式的三个硬约束（违反任何一条前端都会收不到或收到脏数据）：
1. 每个帧以空行结尾，即 `\\n\\n` —— 少一个换行浏览器就认为帧没收完。
2. data 里不能出现裸换行 —— 所以统一走 json.dumps，中文用 ensure_ascii=False
   保持可读，字符串里的换行会被 json 转义成 \\n，不会破坏帧边界。
3. 心跳用注释帧 `: ping\\n\\n`（冒号开头），前端的 event 解析器必须忽略它，
   否则会被当成 event 名空缺的脏帧。
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Mapping

import anyio

# charset 必须显式带上：否则中文 delta 在部分浏览器里会按 latin-1 解出乱码。
SSE_MEDIA_TYPE = "text/event-stream; charset=utf-8"

# no-cache      : 禁止中间层缓存这条流
# no-transform  : 禁止压缩中间层改写（gzip 会把流式响应压成一次性响应）
# X-Accel-Buffering: nginx 专用开关，不关掉的话 nginx 会攒够一个 buffer 才下发，
#                    流式效果直接退化成"等半天一次性出来"
SSE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def encode_data(data: Any) -> str:
    """把任意可序列化对象编成单行 JSON。

    json.dumps 不会产生裸换行，天然满足 SSE 的帧边界要求。
    """
    return json.dumps(data, ensure_ascii=False)


def format_event(event: str, data: Any = None, event_id: str | None = None) -> str:
    """构造一个完整的 SSE 帧（已含结尾空行）。"""
    lines: list[str] = []
    if event_id:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    # data 为空时也要发一个空 data 行：只有 event 没有 data 的帧在部分
    # 客户端实现里会被直接丢弃。
    lines.append(f"data: {encode_data(data) if data is not None else ''}")
    return "\n".join(lines) + "\n\n"


def heartbeat(comment: str = "ping") -> str:
    """心跳帧。

    用注释帧而不是 event 帧：注释帧对前端是"透明"的，不需要前端写
    任何忽略逻辑，也不会污染事件流。
    """
    return f": {comment}\n\n"


def is_heartbeat(frame: str) -> bool:
    """判断一个帧是不是心跳注释帧（测试与前端解析共用同一套判定）。"""
    return frame.startswith(":")


async def sse_frames(
    events: AsyncIterator[Mapping[str, Any]],
    heartbeat_seconds: float = 15.0,
) -> AsyncIterator[str]:
    """把事件字典流转成 SSE 文本帧流，并在静默期插心跳。

    heartbeat_seconds 是"等待下一个事件的最长时间"：上游卡住（检索慢、
    LLM 首 token 慢）超过这个时间就先发一个心跳，用来穿透代理的空闲
    超时，同时让前端能区分"服务端还活着但还没出字"和"连接真断了"。
    传 0 或负数表示不插心跳。

    这里必须做生产者/消费者分离，不能简单地写成
        `with anyio.fail_after(t): event = await gen.__anext__()`
    因为超时取消会直接砸进上游生成器内部的 await（比如 LLM 那次还没
    返回的 sleep / 网络读），把**生成器本身**取消掉：流从此断掉，后面
    的 delta 全部丢失——心跳反而成了杀死流的东西。
    （实测朴素实现：3 个事件只收到 1 个，第二个被心跳超时吃掉。）

    正确做法：上游跑在独立的生产者任务里往内存流送，本协程只消费内存流。
    超时取消落在"等内存流"这一步，碰不到上游生成器。
    """
    if heartbeat_seconds <= 0:
        async for event in events:
            yield _frame_of(event)
        return

    # 缓冲区设为 1：存得下生产者已取到、消费者还没发走的那个事件，
    # 既不会因为背压卡住上游，也不会攒一批事件延迟下发。
    send_stream, receive_stream = anyio.create_memory_object_stream(max_buffer_size=1)

    async def producer() -> None:
        async with send_stream:
            async for event in events:
                await send_stream.send(event)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(producer)
        while True:
            try:
                with anyio.fail_after(heartbeat_seconds):
                    event = await receive_stream.receive()
            except TimeoutError:
                yield heartbeat()
                continue
            except anyio.EndOfStream:
                return
            yield _frame_of(event)


def _frame_of(event: Mapping[str, Any]) -> str:
    """事件字典 → 单个 SSE 帧。"""
    name = str(event.get("event", "message"))
    return format_event(name, event.get("data"), event.get("id"))

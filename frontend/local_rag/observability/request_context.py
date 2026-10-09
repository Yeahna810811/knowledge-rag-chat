"""Request ID 与请求上下文传播。

为什么用 ContextVar 而不是全局变量或显式层层传参：
- 全局变量：并发请求之间必然串号，这是最糟的一类 bug——日志看着正常，
  但 A 请求的耗时被记到 B 头上。
- 显式传参：要靠"每一个函数都记得往下传"，漏一个就断链；而日志点散落在
  retrieval / generation / persistence 各处，全都改签名代价太大。
- ContextVar：一次绑定，当前任务（含其 await 链）内所有代码都能读到，
  并发任务天然隔离。

跨线程的那一跳已经实测过：`anyio.to_thread.run_sync` 会复制当前 context，
所以卸载到线程池的检索 / 读写库同样拿得到 request_id。

异步生成器的边界要特别小心：
`astream_ask` 这类 async generator 是在路由返回之后才被迭代的，路由里的
`bind` 已经出栈了。所以 service 层必须**在生成器内部**再绑一次，
并在结束时用 `finally` 复位——否则 ContextVar 会泄漏到下一个请求。
"""

from __future__ import annotations

import contextvars
import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

# 允许复用客户端传入的 request_id，但必须是"安全字符集"。
# 理由：这个值会进日志、进响应头、进 SSE 帧；放行任意字符串等于
# 给别人一个往日志里塞换行 / 控制字符的入口（日志注入）。
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")

_REQUEST_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "rag_request_id", default=""
)
_SESSION_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "rag_session_id", default=""
)
_MODE: contextvars.ContextVar[str] = contextvars.ContextVar("rag_mode", default="")

UNKNOWN = "-"


def new_request_id() -> str:
    """服务端生成 request_id。用 hex（无连字符）方便双击选中和 grep。"""
    return uuid.uuid4().hex


def is_valid_request_id(value: Optional[str]) -> bool:
    """客户端传来的 request_id 是否可直接复用。

    isinstance 检查不是多余防御：FastAPI 的 `Header(default=None)` 参数
    在被**绕过 ASGI 直接调用**时，默认值就是这个 Header 对象本身，
    不判类型会让 `.strip()` 直接炸在业务代码里。
    """
    if not value or not isinstance(value, str):
        return False
    candidate = value.strip()
    return bool(_REQUEST_ID_PATTERN.match(candidate))


def resolve_request_id(incoming: Optional[str]) -> str:
    """有合法的客户端 ID 就复用（便于跨服务串联），否则服务端生成一个。

    不合法时**静默替换**而不是报错：请求头是客户端可控输入，
    为它返回 400 属于拿别人的格式问题惩罚业务请求。
    """
    if is_valid_request_id(incoming):
        return str(incoming).strip()
    return new_request_id()


@dataclass(frozen=True)
class RequestContext:
    """一次请求的不可变上下文快照。

    frozen：上下文不该被下游改着改着就变了，那样日志里的 session 会和
    实际落库的 session 对不上。
    """

    request_id: str = ""
    session_id: str = ""
    mode: str = ""

    def fields(self) -> dict[str, str]:
        return {
            "request_id": self.request_id or UNKNOWN,
            "session_id": self.session_id or UNKNOWN,
            "mode": self.mode or UNKNOWN,
        }

    def bind(self) -> "ContextTokens":
        return bind_context(self.request_id, self.session_id, self.mode)


@dataclass(frozen=True)
class ContextTokens:
    """三个 ContextVar 的 set token，用于 finally 里精确复位。"""

    request_id: contextvars.Token
    session_id: contextvars.Token
    mode: contextvars.Token


def bind_context(
    request_id: str = "", session_id: str = "", mode: str = ""
) -> ContextTokens:
    return ContextTokens(
        request_id=_REQUEST_ID.set(request_id or ""),
        session_id=_SESSION_ID.set(session_id or ""),
        mode=_MODE.set(mode or ""),
    )


def reset_context(tokens: ContextTokens) -> None:
    """复位到 bind 之前的值。

    ContextVar 的 reset 只认它自己发的 token，顺序无所谓，
    但三个都得复位——漏一个就会污染同一任务里的下一次请求。
    """
    _REQUEST_ID.reset(tokens.request_id)
    _SESSION_ID.reset(tokens.session_id)
    _MODE.reset(tokens.mode)


@contextmanager
def request_scope(
    request_id: str = "", session_id: str = "", mode: str = ""
) -> Iterator[str]:
    """绑定上下文并 yield request_id，退出时自动复位。"""
    tokens = bind_context(request_id, session_id, mode)
    try:
        yield request_id
    finally:
        reset_context(tokens)


def current_request_id() -> str:
    return _REQUEST_ID.get() or UNKNOWN


def current_session_id() -> str:
    return _SESSION_ID.get() or UNKNOWN


def current_mode() -> str:
    return _MODE.get() or UNKNOWN


def context_fields() -> dict[str, str]:
    """给结构化日志用的上下文字段（每次调用都带上，避免漏打）。"""
    return {
        "request_id": current_request_id(),
        "session_id": current_session_id(),
        "mode": current_mode(),
    }


def current_context() -> RequestContext:
    return RequestContext(
        request_id=_REQUEST_ID.get(),
        session_id=_SESSION_ID.get(),
        mode=_MODE.get(),
    )

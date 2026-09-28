"""统一错误分类。

为什么要有这一层：
上游（DashScope / OpenAI SDK）抛出来的异常是一个大杂烩——APITimeoutError、
APIConnectionError、RateLimitError、BadRequestError…… 直接把它们往上抛，
调用方只能 `except Exception`，然后要么一股脑重试（把 401 也重试了），
要么一股脑不重试（把网络抖动也放过了）。两种都是错的。

分类层只回答三个问题：
1. 这是什么类型的错误？（日志 / 指标用）
2. 该不该重试？（重试策略用）
3. 对客户端该返回什么状态码？（响应已成帧时这条用不上，见下）

关于 SSE 的硬约束：
流式响应一旦开始，HTTP 状态码已经发出去了，改不了。所以 error_type 在
流式链路里的作用是"发给前端一个可机读的分类"，不是改状态码。
"""

from __future__ import annotations

import asyncio
import re
from enum import Enum
from typing import Any, Optional

# ---------------------------------------------------------------- 敏感信息脱敏
# 顺序有意义：先匹配最具体的（sk- 开头），再匹配泛化的。
_REDACTIONS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"sk-[A-Za-z0-9_-]{8,}"), "sk-***"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{8,}"), "Bearer ***"),
    (re.compile(r"(?i)\b(api[_-]?key|token|secret|password|passwd)\b\s*[:=]\s*\S+"),
     r"\1=***"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{16,}"), "gh* ***"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AKIA***"),
)

# 日志注入防护：换行会被日志系统当成新的一条记录，
# 控制字符会破坏终端 / 采集器。统一压成空格。
_CONTROL_CHARS = re.compile(r"[\r\n\t\x00-\x1f\x7f]+")

MAX_MESSAGE_LENGTH = 200


def safe_message(text: Any, limit: int = MAX_MESSAGE_LENGTH) -> str:
    """把任意异常/文本压成"可以安全写进日志和发给前端"的一行。

    三件事：
    1. 脱敏——key / token / 密码相关的片段整段替换
    2. 去控制字符——防止日志注入
    3. 截断——上游异常可能带几 KB 的 HTML 错误页，全量塞进日志没意义
    """
    raw = str(text)
    collapsed = _CONTROL_CHARS.sub(" ", raw).strip()
    for pattern, replacement in _REDACTIONS:
        collapsed = pattern.sub(replacement, collapsed)
    collapsed = " ".join(collapsed.split())
    if len(collapsed) > limit:
        collapsed = collapsed[: limit - 3] + "..."
    return collapsed


class ErrorType(str, Enum):
    """错误类型。值是给日志和前端看的稳定字符串，不要随便改。"""

    # ---- LLM 上游 ----
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_RATE_LIMITED = "LLM_RATE_LIMITED"
    LLM_UPSTREAM_ERROR = "LLM_UPSTREAM_ERROR"
    LLM_AUTH_ERROR = "LLM_AUTH_ERROR"
    # ---- 本服务各环节 ----
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    REDIS_UNAVAILABLE = "REDIS_UNAVAILABLE"
    # ---- 请求级 ----
    RATE_LIMITED = "RATE_LIMITED"
    STREAM_CANCELLED = "STREAM_CANCELLED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"

    def __str__(self) -> str:  # 让 f-string 直接输出值而不是 "ErrorType.X"
        return self.value


# 只有「瞬时故障」才在集合里。判据是"再试一次有可能成功"：
# 超时、连接重置、上游限流、上游 5xx 都符合；鉴权失败、参数错误不符合
# ——重试一百次也是一样的错，只会白白拖长用户等待。
RETRYABLE_ERROR_TYPES = frozenset(
    {
        ErrorType.LLM_TIMEOUT,
        ErrorType.LLM_RATE_LIMITED,
        ErrorType.LLM_UPSTREAM_ERROR,
    }
)

# 响应头尚未发出时用的状态码。
# 504 = 我们等上游等超时了；502 = 上游自己说它错了（含重试后仍限流）；
# 429 = 我们自己的限流，和上游无关。
HTTP_STATUS_BY_ERROR_TYPE = {
    ErrorType.LLM_TIMEOUT: 504,
    ErrorType.LLM_UPSTREAM_ERROR: 502,
    ErrorType.LLM_RATE_LIMITED: 502,
    ErrorType.LLM_AUTH_ERROR: 502,
    ErrorType.RATE_LIMITED: 429,
    ErrorType.RETRIEVAL_ERROR: 500,
    ErrorType.DATABASE_ERROR: 500,
    ErrorType.VALIDATION_ERROR: 400,
    ErrorType.STREAM_CANCELLED: 499,
    ErrorType.REDIS_UNAVAILABLE: 200,  # fail-open：不该因为缓存挂了而失败
    ErrorType.INTERNAL_ERROR: 500,
}


def is_retryable(error_type: ErrorType) -> bool:
    return error_type in RETRYABLE_ERROR_TYPES


def http_status_for(error_type: ErrorType) -> int:
    return HTTP_STATUS_BY_ERROR_TYPE.get(error_type, 500)


class ClassifiedError(Exception):
    """带分类的异常：重试策略、HTTP 状态、日志都读它。

    `cause` 保留原始异常供服务端日志深挖，**不要**把它塞进响应体——
    原始异常里可能有 URL、header、provider 的内部报错正文。
    """

    def __init__(
        self,
        error_type: ErrorType,
        message: str,
        *,
        cause: BaseException | None = None,
        retryable: Optional[bool] = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = safe_message(message)
        self.cause = cause
        self.retryable = is_retryable(error_type) if retryable is None else retryable

    @property
    def http_status(self) -> int:
        return http_status_for(self.error_type)

    def as_payload(self) -> dict[str, str]:
        """SSE / HTTP 里给前端的最小错误载荷：只有类型和人话，没有细节。"""
        return {"error_type": self.error_type.value, "message": self.message}

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"ClassifiedError({self.error_type.value}, {self.message!r})"


# ------------------------------------------------------------------ 分类规则
_TIMEOUT_NAMES = {
    "TimeOut", "TimeoutError", "APITimeoutError", "ReadTimeout",
    "ConnectTimeout", "WriteTimeout", "PoolTimeout",
}
_RATE_LIMIT_NAMES = {"RateLimitError", "TooManyRequestsError"}
_AUTH_NAMES = {"AuthenticationError", "PermissionDeniedError", "ForbiddenError"}
_INVALID_REQUEST_NAMES = {
    "BadRequestError", "NotFoundError", "UnprocessableEntityError",
    "InvalidRequestError", "ConflictError",
}
_CONNECTION_NAMES = {
    "APIConnectionError", "ConnectionError", "ConnectionResetError",
    "BrokenPipeError", "ChunkedEncodingError", "RemoteProtocolError",
    "ReadError", "ProtocolError",
}

_TIMEOUT_HINTS = ("timeout", "timed out", "deadline exceeded")
_RATE_LIMIT_HINTS = ("rate limit", "rate_limit", "too many requests", "429")
_AUTH_HINTS = ("api key", "apikey", "unauthorized", "invalid key", "401", "403", "forbidden")
_INVALID_HINTS = ("bad request", "invalid request", "malformed", "400", "422")
_CONNECTION_HINTS = (
    "connection reset", "connection refused", "connection aborted",
    "temporarily unavailable", "network", "dns", "name resolution",
)


def _status_code_of(exc: BaseException) -> Optional[int]:
    """从异常上抠 HTTP 状态码。

    不同 SDK 放的位置不一样：openai 用 `status_code`，有的用 `http_status`，
    还有的只在 `response.status_code` 里。全都试一遍，拿不到就返回 None
    交给名字 / 文本启发式。
    """
    for attr in ("status_code", "http_status", "status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and 100 <= value <= 599:
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int) and 100 <= value <= 599:
            return value
    return None


def _is_cancellation(exc: BaseException) -> bool:
    if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
        return True
    # anyio 的取消异常类随后端不同而不同（asyncio / trio），动态取
    try:  # pragma: no cover - anyio 一定在
        import anyio

        if isinstance(exc, anyio.get_cancelled_exc_class()):
            return True
    except Exception:
        pass
    return False


def classify_exception(exc: BaseException) -> ClassifiedError:
    """把任意异常归类成 ClassifiedError。

    判定顺序刻意从"最确定"到"最模糊"：
    已分类 → 取消 → 状态码 → 异常类名 → 异常文本 → 兜底。
    文本匹配放最后是因为它最不可靠，但也是唯一能兜住
    "第三方库抛了个裸 RuntimeError('connection reset')" 的手段。
    """
    if isinstance(exc, ClassifiedError):
        return exc

    if _is_cancellation(exc):
        return ClassifiedError(
            ErrorType.STREAM_CANCELLED, "客户端已断开或请求被取消", cause=exc
        )

    name = type(exc).__name__
    text = safe_message(exc, limit=400).lower()
    status = _status_code_of(exc)

    # Redis / 数据库异常优先判：这两个是本项目自己的依赖，
    # 名字很特征，且处理方式和 LLM 完全不同（fail-open / 不重试）。
    if "redis" in name.lower() or "redis" in text:
        return ClassifiedError(ErrorType.REDIS_UNAVAILABLE, "Redis 不可用", cause=exc)
    if "sqlalchemy" in name.lower() or "operationalerror" in name.lower() \
            or "dbapi" in name.lower() or "pymysql" in text:
        return ClassifiedError(ErrorType.DATABASE_ERROR, "数据库不可用", cause=exc)

    if status is not None:
        if status == 429:
            return ClassifiedError(ErrorType.LLM_RATE_LIMITED, "上游限流", cause=exc)
        if status in (401, 403):
            return ClassifiedError(ErrorType.LLM_AUTH_ERROR, "上游鉴权失败", cause=exc)
        if 400 <= status < 500:
            return ClassifiedError(
                ErrorType.VALIDATION_ERROR, "上游拒绝了本次请求", cause=exc
            )
        if status >= 500:
            return ClassifiedError(
                ErrorType.LLM_UPSTREAM_ERROR, "上游服务异常", cause=exc
            )

    if name in _TIMEOUT_NAMES or isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return ClassifiedError(ErrorType.LLM_TIMEOUT, "上游调用超时", cause=exc)
    if name in _RATE_LIMIT_NAMES:
        return ClassifiedError(ErrorType.LLM_RATE_LIMITED, "上游限流", cause=exc)
    if name in _AUTH_NAMES:
        return ClassifiedError(ErrorType.LLM_AUTH_ERROR, "上游鉴权失败", cause=exc)
    if name in _INVALID_REQUEST_NAMES:
        return ClassifiedError(ErrorType.VALIDATION_ERROR, "上游拒绝了本次请求", cause=exc)
    if name in _CONNECTION_NAMES:
        return ClassifiedError(ErrorType.LLM_UPSTREAM_ERROR, "上游连接异常", cause=exc)

    for hints, error_type, message in (
        (_TIMEOUT_HINTS, ErrorType.LLM_TIMEOUT, "上游调用超时"),
        (_RATE_LIMIT_HINTS, ErrorType.LLM_RATE_LIMITED, "上游限流"),
        (_AUTH_HINTS, ErrorType.LLM_AUTH_ERROR, "上游鉴权失败"),
        (_INVALID_HINTS, ErrorType.VALIDATION_ERROR, "上游拒绝了本次请求"),
        (_CONNECTION_HINTS, ErrorType.LLM_UPSTREAM_ERROR, "上游连接异常"),
    ):
        if any(hint in text for hint in hints):
            return ClassifiedError(error_type, message, cause=exc)

    # 兜底：认不出来就当内部错误，但把**脱敏截断后**的原始信息带出来。
    # 完全吞掉原始信息的话，排障时只剩一句"服务内部错误"，等于没有线索；
    # 安全意识体现在 safe_message 上（去控制字符、脱敏 key、截断到 120 字），
    # 而不是把消息清空——那是把排障成本和安全性一起丢掉了。
    return ClassifiedError(
        ErrorType.INTERNAL_ERROR,
        f"服务内部错误：{safe_message(exc, limit=120)}",
        cause=exc,
    )

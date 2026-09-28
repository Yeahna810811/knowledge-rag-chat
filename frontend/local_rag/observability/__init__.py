"""Stage 4：可靠性与可观测性。

这一层解决的是"出故障时说不清"的问题，不是新增业务能力。拆成五个模块，
每个只管一件事：

- request_context：request_id 及其传播（ContextVar）
- errors：统一错误分类 + 该不该重试 + HTTP 状态码 + 安全消息
- structured_log：结构化事件日志（k=v 单行，grep 友好）
- retry：指数退避重试（非流式）
- stream_guard：流式链路的超时 + 首 token 前重试 + 任务清理

刻意不引入 OpenTelemetry / Prometheus / structlog：本阶段的诉求是
"能从日志定位一次请求"，一套格式统一的 logging 就够了，引入 collector
反而把排障路径变长。
"""

from frontend.local_rag.observability.errors import (
    ClassifiedError,
    ErrorType,
    classify_exception,
    http_status_for,
    is_retryable,
    safe_message,
)
from frontend.local_rag.observability.request_context import (
    RequestContext,
    bind_context,
    context_fields,
    current_request_id,
    is_valid_request_id,
    new_request_id,
    request_scope,
    resolve_request_id,
    reset_context,
)
from frontend.local_rag.observability.retry import (
    RetryPolicy,
    arun_with_retry,
    run_with_retry,
)
from frontend.local_rag.observability.structured_log import (
    EVENT_LOGGER_NAME,
    Timer,
    configure_logging,
    log_event,
)
from frontend.local_rag.observability.stream_guard import guarded_astream

__all__ = [
    "ClassifiedError",
    "ErrorType",
    "classify_exception",
    "http_status_for",
    "is_retryable",
    "safe_message",
    "RequestContext",
    "bind_context",
    "context_fields",
    "current_request_id",
    "is_valid_request_id",
    "new_request_id",
    "request_scope",
    "resolve_request_id",
    "reset_context",
    "RetryPolicy",
    "arun_with_retry",
    "run_with_retry",
    "EVENT_LOGGER_NAME",
    "Timer",
    "configure_logging",
    "log_event",
    "guarded_astream",
]

"""结构化事件日志。

格式选择：单行 `k=v`，而不是 JSON。
理由很实际——出故障时人是拿着终端在 grep：`grep request_id=abc | grep error_type=`
比 `jq` 快得多，而且 JSON 里的引号转义会让人眼瞎。字段顺序固定，
前五个永远是一样的，扫日志时可以靠肌肉记忆。

字段纪律（这是本模块存在的主要理由）：
**不写完整 question / answer / 文档内容**，只写长度。
原因不是洁癖：
1. 日志会被收集、会被留存、会被第三个人看到；
2. 完整内容对排障几乎没有帮助——"这次请求 5123ms"有用，
   "这次请求的内容是……"没用，真要看内容去看数据库或 trace。
需要片段时只允许截断后的 preview，且默认不输出。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping

EVENT_LOGGER_NAME = "local_rag.events"

logger = logging.getLogger(EVENT_LOGGER_NAME)

# 事件名字典。写出来是为了让"哪些事件存在"变成可枚举的事实，
# 而不是散落在各处的字符串字面量——后者改一个名字谁都不知道。
EV_REQUEST_START = "request_start"
EV_REQUEST_COMPLETE = "request_complete"
EV_REQUEST_FAILED = "request_failed"
EV_RETRIEVAL_COMPLETE = "retrieval_complete"
EV_LLM_START = "llm_start"
EV_LLM_FIRST_TOKEN = "llm_first_token"
EV_LLM_COMPLETE = "llm_complete"
EV_LLM_RETRY = "llm_retry"
EV_PERSISTENCE_COMPLETE = "persistence_complete"
EV_PERSISTENCE_FAILED = "persistence_failed"
EV_STREAM_CANCELLED = "stream_cancelled"
EV_REDIS_UNAVAILABLE = "redis_unavailable"
EV_RATE_LIMITED = "rate_limited"

# 事件名 → 默认级别。失败/降级类默认 WARNING，
# 正常生命周期默认 INFO，方便直接按级别过滤噪音。
_LEVEL_BY_EVENT = {
    EV_REQUEST_FAILED: logging.WARNING,
    EV_PERSISTENCE_FAILED: logging.WARNING,
    EV_STREAM_CANCELLED: logging.INFO,  # 用户主动停止不算异常
    EV_REDIS_UNAVAILABLE: logging.WARNING,
    EV_RATE_LIMITED: logging.WARNING,
    EV_LLM_RETRY: logging.WARNING,
}

_MAX_FIELD_LENGTH = 120


class Timer:
    """perf_counter 计时器。

    刻意不用 wall clock（time.time）：系统时间会被 NTP 调整、
    会被人为改，用它算差值可能得到负数，也能得到跳变几十秒的结果。
    perf_counter 单调，只用来算间隔。
    """

    __slots__ = ("_started",)

    def __init__(self) -> None:
        self._started = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._started) * 1000.0

    def elapsed_ms_rounded(self, digits: int = 1) -> float:
        return round(self.elapsed_ms(), digits)


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.1f}"
    text = str(value)
    if any(char in text for char in (" ", "\t", "\r", "\n", '"')):
        text = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{text}"'
    if len(text) > _MAX_FIELD_LENGTH:
        return text[: _MAX_FIELD_LENGTH - 3] + "..."
    return text


def log_event(
    event: str,
    *,
    level: int | None = None,
    fields: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    """打一条结构化事件日志。

    上下文字段（request_id / session_id / mode）自动补齐：
    调用方不用、也不该手抄——手抄就容易抄错成上一个请求的。
    """
    from frontend.local_rag.observability.request_context import context_fields

    merged: dict[str, Any] = {"event": event}
    merged.update(context_fields())
    if fields:
        merged.update(fields)
    merged.update(kwargs)

    # 固定字段顺序：event / request_id / session_id / mode 永远在最前，
    # 其余按插入顺序，扫日志时位置稳定。
    ordered = ["event", "request_id", "session_id", "mode"]
    parts = [f"{key}={_format_value(merged.pop(key))}" for key in ordered if key in merged]
    parts += [f"{key}={_format_value(value)}" for key, value in merged.items()]

    resolved_level = level if level is not None else _LEVEL_BY_EVENT.get(event, logging.INFO)
    logger.log(resolved_level, " ".join(parts))


def configure_logging(level: int = logging.INFO, *, force: bool = False) -> None:
    """给事件 logger 装一个默认 handler（幂等）。

    幂等很重要：uvicorn 会自己配 logging，测试里也会反复 import，
    每次都 addHandler 会让同一条日志打 N 遍——那会让人误以为重试了 N 次。
    """
    if logger.handlers and not force:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    if force:
        for existing in list(logger.handlers):
            logger.removeHandler(existing)
    logger.addHandler(handler)
    logger.setLevel(level)
    # 不设 propagate=False：留给上层（uvicorn / pytest）决定要不要再收一份

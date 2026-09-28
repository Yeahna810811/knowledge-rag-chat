"""指数退避重试（非流式路径）。

三条纪律：

1. **只重试瞬时故障**，判据来自 errors.is_retryable（超时 / 上游限流 / 上游 5xx
   和连接类）。鉴权失败、参数错误重试一百次也一样错，只会让用户多等一遍。

2. **重试次数有上限**，且 `max_attempts` 是"总尝试次数"而不是"重试次数"。
   写成"重试 3 次"很容易被理解成总共 4 次调用，这里用总次数避免歧义。

3. **取消必须原样向上抛**。这是最容易写错的一条：如果为了重试而
   `except Exception` 一把抓，用户按了停止也会被当成"失败"去重试——
   不仅浪费额度，还会让 abort 语义整个失效。

sleep 可注入是为了让测试不真睡：不注入的话"退避 3 次"的用例要跑
好几秒，而它真正要验证的是"退避次数和间隔序列对不对"，跟真实时间无关。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from frontend.local_rag.observability.errors import (
    ClassifiedError,
    classify_exception,
    is_retryable,
)


def _is_cancellation(exc: BaseException) -> bool:
    """取消类异常：绝对不能吞掉去重试。"""
    if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
        return True
    try:  # pragma: no cover - anyio 一定在
        import anyio

        return isinstance(exc, anyio.get_cancelled_exc_class())
    except Exception:
        return False


@dataclass(frozen=True)
class RetryPolicy:
    """退避参数。

    max_attempts=3 的含义：最多调用 3 次（首次 + 2 次重试）。
    """

    max_attempts: int = 3
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 8.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts 至少为 1")
        if self.base_delay_seconds < 0:
            raise ValueError("base_delay_seconds 不能为负")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds 不能小于 base_delay_seconds")

    @property
    def enabled(self) -> bool:
        return self.max_attempts > 1

    def delay_after(self, attempt: int) -> float:
        """第 attempt 次调用失败后该等多久（attempt 从 1 开始）。

        指数增长但封顶：不封顶的话第 10 次重试会等到几分钟，
        用户早就把页面关了，这个等待毫无意义。
        """
        raw = self.base_delay_seconds * (2 ** max(0, attempt - 1))
        return min(raw, self.max_delay_seconds)


def run_with_retry(
    operation: Callable[[], Any],
    policy: RetryPolicy,
    *,
    sleep: Callable[[float], Any] | None = None,
    on_retry: Callable[[int, ClassifiedError, float], None] | None = None,
    classify: Callable[[BaseException], ClassifiedError] = classify_exception,
) -> Any:
    """同步重试。返回 operation() 的结果，最终失败抛 ClassifiedError。"""
    do_sleep = sleep if sleep is not None else time.sleep

    attempt = 1
    while True:
        try:
            return operation()
        except BaseException as exc:  # noqa: BLE001 - 需要统一分类
            if _is_cancellation(exc):
                raise
            classified = classify(exc)
            if not classified.retryable or not is_retryable(classified.error_type):
                raise classified from exc
            if attempt >= policy.max_attempts:
                raise classified from exc

            delay = policy.delay_after(attempt)
            if on_retry is not None:
                on_retry(attempt, classified, delay)
            do_sleep(delay)
            attempt += 1


async def arun_with_retry(
    operation: Callable[[], Any],
    policy: RetryPolicy,
    *,
    sleep: Callable[[float], Any] | None = None,
    on_retry: Callable[[int, ClassifiedError, float], None] | None = None,
    classify: Callable[[BaseException], ClassifiedError] = classify_exception,
) -> Any:
    """异步版：operation 可以是协程函数，sleep 也是 await 的。"""
    if sleep is None:  # pragma: no cover - 默认路径
        import anyio

        async def do_sleep(seconds: float) -> None:
            await anyio.sleep(seconds)
    else:
        do_sleep = sleep  # type: ignore[assignment]

    attempt = 1
    while True:
        try:
            result = operation()
            if hasattr(result, "__await__"):
                result = await result
            return result
        except BaseException as exc:  # noqa: BLE001
            if _is_cancellation(exc):
                raise
            classified = classify(exc)
            if not classified.retryable or not is_retryable(classified.error_type):
                raise classified from exc
            if attempt >= policy.max_attempts:
                raise classified from exc

            delay = policy.delay_after(attempt)
            if on_retry is not None:
                on_retry(attempt, classified, delay)
            await do_sleep(delay)  # type: ignore[misc]
            attempt += 1


def policy_from_settings(settings: Any) -> RetryPolicy:
    """从 Settings 构造策略；重试关掉时返回 max_attempts=1 的"不重试"策略。"""
    enabled = bool(getattr(settings, "llm_retry_enabled", True))
    max_attempts = int(getattr(settings, "llm_retry_max_attempts", 3) or 1)
    if not enabled or max_attempts <= 1:
        max_attempts = 1
    return RetryPolicy(
        max_attempts=max_attempts,
        base_delay_seconds=float(getattr(settings, "llm_retry_base_delay_seconds", 0.5)),
        max_delay_seconds=float(getattr(settings, "llm_retry_max_delay_seconds", 8.0)),
    )

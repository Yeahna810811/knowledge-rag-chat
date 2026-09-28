"""流式链路的守卫：首 token 超时、空闲超时、首 token 前重试。

这是 Stage 4 最需要小心的一块，有三个坑：

**坑一：不能用 fail_after 直接包上游生成器的 __anext__。**
Stage 3 做心跳的时候已经踩过一次：`with fail_after(t): await gen.__anext__()`
超时后取消会砸进生成器内部的 await（那次还没返回的 LLM 网络读），
把生成器**本身**取消掉。做心跳时表现为"流被心跳杀死"，做超时时表现会更糟——
看起来是超时，实际是把一次可能还能救回来的生成给掐了。
所以这里沿用 Stage 3 的生产者/消费者分离：上游跑在独立任务里往内存流送，
超时只作用在"等内存流"这一步，取消落在消费者侧，碰不到上游。

**坑二：超时后不能留 dangling task。**
生产者如果卡在一次 LLM 网络读上，消费者这边超时返回了，那个任务还活着，
继续烧着连接和额度。所以每次尝试都开一个 task group。
检测到超时或上游错误后，先取消该 task group，让 producer 和生成器完成清理，
等 task group 完整退出之后，再向外抛业务异常。

这样既可以保证后台任务被清掉，也避免在 task group 内直接抛 ClassifiedError
时被 AnyIO 包装成 ExceptionGroup。

顺带一提，`gen.aclose()` 要放 shield 里——取消已经生效时不屏蔽的话，
清理动作自己就会被取消掉。

**坑三：首 token 之后绝不能透明重试。**
这是本模块存在的核心理由。第一次已经吐给用户"你好，我是…"，
第二次重试又从头生成一遍，用户看到的就是两遍开头。
所以 `sent_any` 一旦为真，任何错误都直接向上抛，一次都不重试。
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator, Callable

import anyio

from frontend.local_rag.observability.errors import (
    ClassifiedError,
    ErrorType,
    classify_exception,
    is_retryable,
)
from frontend.local_rag.observability.retry import RetryPolicy, _is_cancellation


_ITEM = "item"
_END = "end"
_ERROR = "error"


async def _one_attempt(
    factory: Callable[[], AsyncIterator[Any]],
    *,
    first_token_timeout: float | None,
    idle_timeout: float | None,
    on_first_token: Callable[[float], None] | None,
    started_at: float,
) -> AsyncIterator[Any]:
    """跑一次上游生成，套上首 token / 空闲超时，并保证任务清理干净。"""

    send_stream, receive_stream = anyio.create_memory_object_stream(
        max_buffer_size=1
    )

    async def producer() -> None:
        """独立运行上游 async generator，并把结果转发到内存流。"""
        gen: AsyncIterator[Any] | None = None

        try:
            gen = factory()

            async for item in gen:
                await send_stream.send((_ITEM, item))

            await send_stream.send((_END, None))

        except BaseException as exc:  # noqa: BLE001 - 需要转成内部事件
            # CancelledError / GeneratorExit 等取消语义必须原样向上抛。
            # 不能把用户主动停止包装成普通 LLM error，更不能参与 retry。
            if _is_cancellation(exc):
                raise

            try:
                await send_stream.send((_ERROR, exc))
            except Exception:
                # 消费者可能已经因为 timeout / disconnect 离开。
                pass

        finally:
            if gen is not None:
                try:
                    # 如果 task group 已经进入取消状态，
                    # 普通 await 也可能马上再次被取消。
                    # shield 保证真正执行 generator 清理。
                    with anyio.CancelScope(shield=True):
                        await gen.aclose()
                except Exception:
                    # 清理异常不能覆盖真正的上游异常或取消语义。
                    pass

            try:
                await send_stream.aclose()
            except Exception:
                pass

    async def receive(timeout: float | None) -> tuple[str, Any]:
        """等待 producer 发来的下一个内部事件。

        timeout 为 None 或 <= 0 时表示不限制等待时间。

        注意：
        timeout 只包 receive_stream.receive()，
        不直接包上游 generator.__anext__()。
        """
        if timeout is None or timeout <= 0:
            return await receive_stream.receive()

        with anyio.fail_after(timeout):
            return await receive_stream.receive()

    # 关键修复：
    #
    # 不在 anyio TaskGroup 内直接 raise ClassifiedError。
    # 否则在 Python 3.9 + exceptiongroup / AnyIO 环境中，
    # TaskGroup 退出时可能把业务异常包装成 ExceptionGroup，
    # 导致 guarded_astream 外层的：
    #
    #     except ClassifiedError
    #
    # 无法直接捕获。
    #
    # 因此这里先记住错误，取消并等待 producer 清理，
    # 等 TaskGroup 完整退出以后再抛出。
    pending_error: ClassifiedError | None = None

    try:
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(producer)

            try:
                kind, payload = await receive(first_token_timeout)

            except TimeoutError:
                pending_error = ClassifiedError(
                    ErrorType.LLM_TIMEOUT,
                    f"LLM 首个 token 超时（>{first_token_timeout}s）",
                )

                # producer 可能还卡在 LLM 网络读取中。
                # 必须主动取消，不能让它变成 dangling task。
                task_group.cancel_scope.cancel()

            else:
                while True:
                    if kind == _ERROR:
                        # 上游生成器主动失败。
                        # 先统一分类，再让 task group 清理 producer。
                        pending_error = classify_exception(payload)
                        task_group.cancel_scope.cancel()
                        break

                    if kind == _END:
                        # 正常生成结束。
                        break

                    # 当前 kind == _ITEM。
                    #
                    # TTFT 必须以第一个真实 LLM chunk 为准，
                    # 不是 SSE meta / heartbeat / trace。
                    if on_first_token is not None:
                        on_first_token(
                            (time.perf_counter() - started_at) * 1000.0
                        )
                        on_first_token = None

                    yield payload

                    try:
                        kind, payload = await receive(idle_timeout)

                    except TimeoutError:
                        pending_error = ClassifiedError(
                            ErrorType.LLM_TIMEOUT,
                            (
                                "LLM 流式输出空闲超时"
                                f"（>{idle_timeout}s 没有新内容）"
                            ),
                        )

                        # 已经等待不到新的 chunk。
                        # 取消仍在后台等待网络数据的 producer。
                        task_group.cancel_scope.cancel()
                        break

                    except anyio.EndOfStream:
                        # producer 已关闭发送端。
                        # 没有额外错误时按正常结束处理。
                        break

        # 只有 task group 已经退出到这里之后才抛业务异常。
        #
        # 此时：
        # - producer 已经取消 / 结束
        # - gen.aclose() 已经执行
        # - 不应该存在 dangling task
        # - ClassifiedError 不再被 TaskGroup 包成 ExceptionGroup
        if pending_error is not None:
            raise pending_error

    finally:
        try:
            await receive_stream.aclose()
        except Exception:
            pass


async def guarded_astream(
    factory: Callable[[], AsyncIterator[Any]],
    *,
    policy: RetryPolicy,
    first_token_timeout: float | None = None,
    idle_timeout: float | None = None,
    sleep: Callable[[float], Any] | None = None,
    on_retry: Callable[[int, ClassifiedError, float], None] | None = None,
    on_first_token: Callable[[float], Any] | None = None,
) -> AsyncIterator[Any]:
    """带超时与"首 token 前才重试"语义的安全流式生成。

    factory 每次调用都必须返回一个**全新的**异步生成器。

    原因：

    首 token 前如果发生 retryable transient failure，
    guarded_astream 会重新调用 factory 发起新一次 LLM 请求。

    如果 factory 每次返回的是同一个已经开始消费甚至已经耗尽的生成器，
    retry 可能静默得到空结果。

    重试纪律：

    1. 首 token 前：
       retryable error 可以有限重试。

    2. 首 token 后：
       任何错误都不透明重试。

       因为用户已经看到了第一轮生成内容，
       如果重新从头请求 LLM，会出现重复开头。

    3. CancelledError / GeneratorExit：
       原样向上抛，不参与 retry。
    """

    if sleep is None:
        # 正式运行路径。
        do_sleep = anyio.sleep
    else:
        # 测试可以注入一个不会真实等待的 sleep，
        # 用来验证 exponential backoff。
        do_sleep = sleep  # type: ignore[assignment]

    started_at = time.perf_counter()

    # 一旦为 True，就说明至少有一个真实 chunk 已经交给下游。
    sent_any = False

    # max_attempts 表示总尝试次数：
    # attempt=1 是首次调用，不是第一次 retry。
    attempt = 1

    while True:
        try:
            async for item in _one_attempt(
                factory,
                first_token_timeout=first_token_timeout,
                idle_timeout=idle_timeout,
                on_first_token=on_first_token,
                started_at=started_at,
            ):
                # 必须先设置 sent_any，再 yield。
                #
                # yield 后控制权就交给下游。
                # 如果下游在收到 chunk 后发生 disconnect / cancellation，
                # 我们回来时必须已经知道"用户已经看到过 token"。
                sent_any = True

                yield item

            # 正常完整结束。
            return

        except ClassifiedError as exc:
            # ---------------------------------------------------------
            # 最重要的 streaming retry 边界：
            # ---------------------------------------------------------
            #
            # 已经向用户发送过真实 token：
            #
            #     第一次：
            #     "RAG 是一种..."
            #
            # 如果这时候重新请求 LLM：
            #
            #     第二次：
            #     "RAG 是一种..."
            #
            # 用户就会看到重复内容。
            #
            # 所以首 token 之后绝对不透明 retry。
            if sent_any:
                raise

            # 首 token 前虽然可以 retry，
            # 但必须同时满足：
            #
            # 1. 当前 error type 属于 transient / retryable
            # 2. 还没达到最大尝试次数
            if (
                not is_retryable(exc.error_type)
                or attempt >= policy.max_attempts
            ):
                raise

            delay = policy.delay_after(attempt)

            if on_retry is not None:
                on_retry(attempt, exc, delay)

            await do_sleep(delay)  # type: ignore[misc]

            attempt += 1

        except BaseException:
            # CancelledError / GeneratorExit 等取消语义，
            # 以及其它所有非 ClassifiedError：
            #
            # 一律原样向上抛。
            #
            # 特别不能把用户点击"停止"理解成 transient failure
            # 然后重新调用 LLM。
            raise
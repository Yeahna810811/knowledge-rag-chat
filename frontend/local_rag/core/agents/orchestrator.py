from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator

import anyio
from langsmith import traceable

from frontend.local_rag.config.settings import Settings
from frontend.local_rag.core.agents.document_agent import DocumentParseAgent
from frontend.local_rag.core.agents.generation_agent import GenerationAgent
from frontend.local_rag.core.agents.retrieval_agent import RetrievalAgent
from frontend.local_rag.core.document_processor import DocumentProcessor
from frontend.local_rag.core.retrieval.protocol import RetrievalStore
from frontend.local_rag.observability.errors import (
    ClassifiedError,
    ErrorType,
    safe_message,
)
from frontend.local_rag.observability.structured_log import (
    EV_RETRIEVAL_COMPLETE,
    Timer,
    log_event,
)

# 检索阶段的错误单独归类：它和 LLM 错误的处理方式不同
# （LLM 错误可重试，检索错误不重试——检索是本地计算）。
RETRIEVAL_ERROR_TYPE = ErrorType.RETRIEVAL_ERROR


class AgentOrchestrator:
    """Coordinate document / retrieval / generation agents for dual-mode QA."""

    def __init__(
        self,
        settings: Settings,
        document_processor: DocumentProcessor,
        vector_store_manager: RetrievalStore,
        upload_dir: Path,
    ) -> None:
        self.settings = settings
        self.document_agent = DocumentParseAgent(
            document_processor=document_processor,
            vector_store_manager=vector_store_manager,
            upload_dir=upload_dir,
        )
        self.retrieval_agent = RetrievalAgent(
            vector_store_manager=vector_store_manager,
            top_k=settings.retrieval_top_k,
        )
        self.generation_agent = GenerationAgent(settings)

    @traceable(name="ingest_pipeline", run_type="chain")
    def ingest(self, filename: str, content: bytes) -> dict[str, Any]:
        result = self.document_agent.run(filename=filename, content=content)
        if not result.success:
            raise ValueError(result.message)
        return {
            **result.data,
            "message": result.message,
            "agent_trace": [result.agent],
        }

    @traceable(name="ask_pipeline", run_type="chain")
    def ask(
        self,
        question: str,
        history: list[dict] | None = None,
        mode: str = "rag",
    ) -> dict[str, Any]:
        history = history or []
        mode = (mode or "rag").lower()
        if mode not in {"rag", "chat"}:
            raise ValueError("mode 仅支持 rag 或 chat")

        agent_trace: list[str] = []
        sources: list[dict] = []
        retrieval_ms: float | None = None

        if mode == "rag":
            retrieval_timer = Timer()
            sources = []
            try:
                retrieval = self.retrieval_agent.run(question=question)
            except Exception as exc:  # noqa: BLE001
                # 检索是本地计算（BM25 / FAISS），失败通常是索引损坏 / 目录不可读
                # 这类确定性故障——重试只会再算一遍同样的错误，所以不重试。
                log_event(
                    EV_RETRIEVAL_COMPLETE,
                    status="failed",
                    error_type=RETRIEVAL_ERROR_TYPE.value,
                    retrieval_ms=retrieval_timer.elapsed_ms_rounded(),
                )
                raise ClassifiedError(
                    ErrorType.RETRIEVAL_ERROR, f"检索失败：{safe_message(exc)}", cause=exc
                ) from exc
            retrieval_ms = retrieval_timer.elapsed_ms_rounded()
            agent_trace.append(retrieval.agent)
            sources = retrieval.data.get("sources", [])
            log_event(
                EV_RETRIEVAL_COMPLETE,
                status="ok",
                retrieval_ms=retrieval_ms,
                sources=len(sources),
            )

        generation = self.generation_agent.run(
            question=question,
            history=history,
            sources=sources,
            mode=mode,
        )
        agent_trace.append(generation.agent)
        if not generation.success:
            raise ValueError(generation.message or "生成失败")

        return {
            "question": question,
            "answer": generation.data["answer"],
            "sources": generation.data.get("sources", []),
            "mode": mode,
            "agent_trace": agent_trace,
            # 追加字段，不改已有语义：chat 模式下没有检索，给 None 而不是 0，
            # 0 会被误读成"检索耗时 0 毫秒"。
            "retrieval_ms": retrieval_ms,
        }

    async def astream(
        self,
        question: str,
        history: list[dict] | None = None,
        mode: str = "rag",
    ) -> AsyncIterator[dict[str, Any]]:
        """流式问答编排：产出事件字典，由上层翻译成 SSE 帧。

        和同步版 ask() 的关键差别只有一个：检索这段是阻塞的（BM25 分词、
        FAISS 相似度、Redis 缓存读写都是同步代码），如果直接 await 它，
        **整个事件循环会被堵住**，同一进程里其它并发的 SSE 连接全部跟着卡死。
        所以检索显式丢到线程池（anyio.to_thread.run_sync），
        而生成段是真正的异步 I/O，直接 astream 逐块转发。

        产出的事件顺序：sources（仅 rag）→ trace → delta* 。
        sources 必须排在 delta 之前：前端要先把"参考资料"渲染出来，
        用户才能边看答案边对照出处。
        """
        history = history or []
        mode = (mode or "rag").lower()
        if mode not in {"rag", "chat"}:
            raise ValueError("mode 仅支持 rag 或 chat")

        sources: list[dict] = []
        agent_trace: list[str] = []

        if mode == "rag":
            # 检索卸载到线程池（BM25 / FAISS / Redis 全是同步代码），
            # 计时只包这一段：它是"检索"而不是"整个请求"。
            retrieval_timer = Timer()
            try:
                retrieval = await anyio.to_thread.run_sync(
                    lambda: self.retrieval_agent.run(question=question)
                )
            except ClassifiedError:
                raise
            except Exception as exc:  # noqa: BLE001
                log_event(
                    EV_RETRIEVAL_COMPLETE,
                    status="failed",
                    error_type=RETRIEVAL_ERROR_TYPE.value,
                    retrieval_ms=retrieval_timer.elapsed_ms_rounded(),
                )
                raise ClassifiedError(
                    ErrorType.RETRIEVAL_ERROR, f"检索失败：{safe_message(exc)}", cause=exc
                ) from exc
            retrieval_ms = retrieval_timer.elapsed_ms_rounded()
            agent_trace.append(retrieval.agent)
            sources = retrieval.data.get("sources", [])
            log_event(
                EV_RETRIEVAL_COMPLETE,
                status="ok",
                retrieval_ms=retrieval_ms,
                sources=len(sources),
            )
            yield {
                "event": "sources",
                "data": {
                    "sources": sources,
                    "mode": mode,
                    "retrieval_ms": retrieval_ms,
                },
            }

        agent_trace.append(self.generation_agent.name)
        yield {"event": "trace", "data": {"agent_trace": agent_trace}}

        async for text in self.generation_agent.astream_text(
            question=question,
            history=history,
            sources=sources,
            mode=mode,
        ):
            yield {"event": "delta", "data": {"text": text}}

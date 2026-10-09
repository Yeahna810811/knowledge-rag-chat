from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from langchain_openai import ChatOpenAI
from langsmith import traceable

from frontend.local_rag.config.settings import Settings
from frontend.local_rag.core.agents.base import AgentResult, BaseAgent
from frontend.local_rag.observability.errors import ClassifiedError
from frontend.local_rag.observability.retry import (
    RetryPolicy,
    policy_from_settings,
    run_with_retry,
)
from frontend.local_rag.observability.stream_guard import guarded_astream
from frontend.local_rag.observability.structured_log import (
    EV_LLM_COMPLETE,
    EV_LLM_RETRY,
    EV_LLM_START,
    Timer,
    log_event,
)

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """你是企业知识库客服助手，优先依据【参考资料】回答用户问题。

回答规则：
1. 知识库事实性问题只能依据【参考资料】回答，禁止用模型自身知识猜测或补充未提供的事实。
2. 如果【参考资料】没有直接提供问题所需信息，请明确告知：当前知识库资料未提供该信息，建议联系人工客服进一步确认。
3. 资料不足后不要继续推测并发量、成本、GPU、SLA、云平台、文件限制等未写明的内容。
4. 有明确答案时简洁直接回答，可结合对话历史理解追问，但历史不能当作知识库证据。
5. 普通闲聊可以自然、礼貌地回应。
"""

CHAT_SYSTEM_PROMPT = """你是一个友好、专业的 AI 对话助手。
请自然地回答用户问题，保持简洁清晰。这是普通 AI 对话模式，不依赖企业知识库检索结果。
"""


def _chunk_text(content: Any) -> str:
    """把 LLM chunk 的 content 规整成纯文本增量。

    不同 provider 的 content 形状不一样：多数是 str，多模态模型会给
    list[dict]（形如 [{"type": "text", "text": "..."}]）。这里统一收敛成
    字符串，取不到文本时返回空串——宁可少发一个空 delta，也不要把
    "{'type': 'text', ...}" 这种字典 repr 打到用户屏幕上。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "".join(parts)
    if isinstance(content, dict) and "text" in content:
        return str(content["text"])
    return ""


class GenerationAgent(BaseAgent):
    """Generate answers in RAG (grounded customer-service) or plain chat mode."""

    name = "generation_agent"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if not settings.dashscope_api_key:
            raise ValueError(
                "未检测到 DASHSCOPE_API_KEY，请在 .env 文件中配置阿里云百炼的 API Key"
            )
        # 两个刻意的非默认配置：
        #
        # max_retries=0 —— 关掉 openai SDK 的内建重试。
        #   默认它是 2，会在 transport 层静默重发：不打日志、不带 request_id，
        #   流式场景还会把整条流重跑一遍。留着它就是
        #   "SDK retry × 本项目 retry"，失败时根本数不清到底发了几遍。
        #   关掉之后重试逻辑全部收敛到 run_with_retry / guarded_astream。
        #
        # timeout —— 给非流式调用一个硬上界。流式另有首 token / 空闲超时
        #   （在 guarded_astream 里），因为流式总时长天然不可预测，
        #   用总时长卡会把长回答误杀。
        self.llm = ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            temperature=0.5,
            max_retries=settings.llm_sdk_max_retries,
            timeout=settings.llm_request_timeout_seconds,
        )
        self._retry_policy: RetryPolicy = policy_from_settings(settings)

    # ------------------------------------------------------------- 观测辅助
    def _log_retry(self, attempt: int, error: ClassifiedError, delay: float) -> None:
        log_event(
            EV_LLM_RETRY,
            status="retrying",
            error_type=error.error_type.value,
            attempt=attempt,
            next_attempt=attempt + 1,
            max_attempts=self._retry_policy.max_attempts,
            delay_s=delay,
            message=error.message,
        )

    def build_messages(
        self,
        question: str = "",
        history: list[dict] | None = None,
        sources: list[dict] | None = None,
        mode: str = "rag",
    ) -> list[dict]:
        """把「问题 + 历史 + 检索到的资料」编成 ChatOpenAI 的 messages。

        抽出来的原因：一次性生成（invoke）和流式生成（astream）必须走
        **完全相同**的拼装逻辑。否则流式和非流式会得到两套不同的 prompt，
        出现"流式回答质量变差"这种最难排查的问题——大家只会怀疑流式的锅，
        实际是 prompt 悄悄分叉了。
        """
        history = history or []
        sources = sources or []
        mode = (mode or "rag").lower()

        if mode == "chat":
            system_prompt = CHAT_SYSTEM_PROMPT
            user_content = question
        else:
            system_prompt = RAG_SYSTEM_PROMPT
            if sources:
                context = "\n\n".join(item["content"] for item in sources)
            else:
                context = "（未检索到知识库相关内容）"
            user_content = f"【参考资料】\n{context}\n\n【用户问题】\n{question}"

        messages = [{"role": "system", "content": system_prompt}]
        for turn in history:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})
        messages.append({"role": "user", "content": user_content})
        return messages

    @traceable(name="generation_agent", run_type="llm")
    def run(
        self,
        question: str = "",
        history: list[dict] | None = None,
        sources: list[dict] | None = None,
        mode: str = "rag",
        **_: Any,
    ) -> AgentResult:
        mode = (mode or "rag").lower()
        sources = sources or []
        messages = self.build_messages(
            question=question, history=history, sources=sources, mode=mode
        )

        # 只写长度不写内容：完整 question / answer 进日志既没必要也有风险，
        # 排障真正需要的是"这次花了多久、重试了几次"。
        gen_timer = Timer()
        log_event(
            EV_LLM_START,
            mode=mode,
            question_len=len(question),
            sources=len(sources),
            streaming=False,
        )

        try:
            response = run_with_retry(
                lambda: self.llm.invoke(messages),
                self._retry_policy,
                on_retry=self._log_retry,
            )
        except ClassifiedError as exc:
            log_event(
                EV_LLM_COMPLETE,
                status="failed",
                error_type=exc.error_type.value,
                generation_ms=gen_timer.elapsed_ms_rounded(),
                message=exc.message,
            )
            raise

        answer = response.content if isinstance(response.content, str) else str(response.content)
        log_event(
            EV_LLM_COMPLETE,
            status="ok",
            generation_ms=gen_timer.elapsed_ms_rounded(),
            answer_len=len(answer),
        )

        return AgentResult(
            agent=self.name,
            success=True,
            message="生成完成",
            data={
                "answer": answer,
                "mode": mode,
                "sources": sources if mode == "rag" else [],
            },
        )

    async def astream_text(
        self,
        question: str = "",
        history: list[dict] | None = None,
        sources: list[dict] | None = None,
        mode: str = "rag",
    ) -> AsyncIterator[str]:
        """流式生成：逐个 chunk 产出纯文本增量。

        这里刻意 **不** 套 @traceable：LangSmith 的 traceable 面向
        "一次调用一次返回"，套在异步生成器上会把整条流的生命周期搅乱，
        而且会让流式链路多一层不可控的包装。流式链路的可观测性
        留给后续 Observability 阶段统一处理。

        另一个细节：只 yield 非空文本。上游 chunk 里有相当比例是空串
        （尤其是首 chunk 带 role 信息时），全量转发会让前端多渲染几百次
        空字符串。

        可靠性（Stage 4）由 guarded_astream 提供，两条规则：
        - 首 token / 空闲超时
        - **只有在还没吐出任何 token 之前**失败才重试
          （吐了之后再重试，用户会看到两遍开头；详见 stream_guard 注释）
        """
        mode = (mode or "rag").lower()
        sources = sources or []
        messages = self.build_messages(
            question=question, history=history, sources=sources, mode=mode
        )

        gen_timer = Timer()
        started_at = gen_timer

        log_event(
            EV_LLM_START,
            mode=mode,
            question_len=len(question),
            sources=len(sources),
            streaming=True,
        )

        # factory 必须是"每次调用返回全新生成器"：重试就是靠重新调用它
        # 来重发一次 LLM 请求。
        def factory() -> AsyncIterator[str]:
            return self._raw_astream(messages)

        pieces: list[str] = []
        try:
            # TTFT 不在这里记：它由 service 层统一定义并上报
            # （request start → 第一个真实 token 发给用户），含检索耗时。
            # 两层各记一次的话，同一条流会出现两个 ttft_ms 且数值不同，
            # 排障时反而多一个"到底以哪个为准"的疑问。
            async for text in guarded_astream(
                factory,
                policy=self._retry_policy,
                first_token_timeout=self.settings.llm_first_token_timeout_seconds,
                idle_timeout=self.settings.llm_stream_idle_timeout_seconds,
                on_retry=self._log_retry,
                on_first_token=None,
            ):
                pieces.append(text)
                yield text
        except ClassifiedError as exc:
            log_event(
                EV_LLM_COMPLETE,
                status="failed",
                error_type=exc.error_type.value,
                streaming=True,
                generation_ms=gen_timer.elapsed_ms_rounded(),
                partial_len=sum(len(p) for p in pieces),
                message=exc.message,
            )
            raise

        log_event(
            EV_LLM_COMPLETE,
            status="ok",
            streaming=True,
            generation_ms=gen_timer.elapsed_ms_rounded(),
            answer_len=sum(len(p) for p in pieces),
        )

    async def _raw_astream(self, messages: list[dict]) -> AsyncIterator[str]:
        """最原始的一层：把 ChatOpenAI 的 chunk 流转成纯文本增量。"""
        async for chunk in self.llm.astream(messages):
            text = _chunk_text(chunk.content)
            if text:
                yield text

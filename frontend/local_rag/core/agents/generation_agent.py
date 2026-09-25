from __future__ import annotations

from typing import Any, AsyncIterator

from langchain_openai import ChatOpenAI
from langsmith import traceable

from frontend.local_rag.config.settings import Settings
from frontend.local_rag.core.agents.base import AgentResult, BaseAgent

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
        self.llm = ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            temperature=0.5,
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

        response = self.llm.invoke(messages)
        answer = response.content if isinstance(response.content, str) else str(response.content)

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
        """
        mode = (mode or "rag").lower()
        sources = sources or []
        messages = self.build_messages(
            question=question, history=history, sources=sources, mode=mode
        )

        async for chunk in self.llm.astream(messages):
            text = _chunk_text(chunk.content)
            if text:
                yield text

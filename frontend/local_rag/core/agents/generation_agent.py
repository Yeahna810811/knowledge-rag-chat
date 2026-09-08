from __future__ import annotations

from typing import Any

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

    @traceable(name="generation_agent", run_type="llm")
    def run(
        self,
        question: str = "",
        history: list[dict] | None = None,
        sources: list[dict] | None = None,
        mode: str = "rag",
        **_: Any,
    ) -> AgentResult:
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

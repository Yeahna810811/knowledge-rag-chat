from langchain_openai import ChatOpenAI
from frontend.local_rag.config.settings import Settings
from frontend.local_rag.core.vector_store import VectorStoreManager

SYSTEM_PROMPT = """你是一个严格基于知识库资料回答问题的 RAG 助手。

回答规则：
1. 对于知识库事实性问题，只能依据【参考资料】回答。
2. 禁止使用模型自身知识补充、猜测或推断参考资料中没有明确提供的信息。
3. 如果【参考资料】没有直接提供问题所需的信息，只回答：当前知识库资料未提供该信息。
4. 回答资料不足后立即停止，不要继续解释、推测、举例或提供可能性。
5. 资料没有提到某功能，不等于系统不支持该功能。
6. 禁止推测并发用户数、服务器成本、GPU 型号、SLA、云平台、文件大小限制等未明确提供的信息。
7. 有明确答案时简洁直接回答，不加入参考资料无法支持的额外细节。
8. 可以结合对话历史理解追问，但历史内容不能作为知识库中不存在事实的依据。
9. 普通闲聊可以自然回答，不需要强行引用知识库。
"""


class RAGChain:
    """检索增强问答：结合对话历史 + 知识库检索内容，调用大模型生成回答"""

    def __init__(self, settings: Settings, vector_store_manager: VectorStoreManager):
        self.settings = settings
        self.vector_store_manager = vector_store_manager
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

    def run(self, question, history=None):
        if history is None:
            history = []
        # 1. 检索知识库相关片段
        docs = []
        if self.vector_store_manager and self.vector_store_manager.is_ready:
            docs = self.vector_store_manager.search(
                question, k=self.settings.retrieval_top_k
            )
        if docs:
            context = "\n\n".join(d["content"] for d in docs)
        else:
            context = "（未检索到知识库相关内容）"

        # 2. 组装消息：系统提示 + 历史对话 + 当前问题（带检索上下文）
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in history:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})
        user_content = f"【参考资料】\n{context}\n\n【用户问题】\n{question}"
        messages.append({"role": "user", "content": user_content})

        # 3. 调用阿里云百炼大模型生成回答
        response = self.llm.invoke(messages)
        answer = response.content

        return {
            "question": question,
            "answer": answer,
            "sources": docs,
        }
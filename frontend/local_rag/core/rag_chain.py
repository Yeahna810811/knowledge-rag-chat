from langchain_openai import ChatOpenAI
from frontend.local_rag.config.settings import Settings
from frontend.local_rag.core.vector_store import VectorStoreManager

SYSTEM_PROMPT = """你是一个友好、自然的AI聊天助手，同时具备知识库检索增强能力。
规则：
1. 你能记住本次对话中之前说过的内容（见下方"对话历史"），回答时要结合上下文，像正常聊天一样自然。
2. 如果【参考资料】里有跟用户问题相关的内容，优先依据参考资料回答，并可以指出信息来自知识库。
3. 如果【参考资料】跟用户问题无关（比如用户在闲聊、问你是谁、问之前说过的话等），完全不要提"知识库中没有相关信息"这种话，直接当作普通聊天正常回答即可，参考资料这时可以忽略。
4. 不要逐字复述参考资料，要用自己的话组织回答。
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
            context = "（本次提问未检索到知识库相关内容，可忽略此项，按普通聊天回答）"

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
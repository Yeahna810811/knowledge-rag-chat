from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingService:
    """真正的本地语义向量服务（基于 sentence-transformers，免费、无需API Key）"""

    def __init__(self, settings):
        self.embedding = HuggingFaceEmbeddings(
            model_name=settings.embedding_model
        )

    def get_embedding(self, text):
        return self.embedding.embed_query(text)

    def embed_documents(self, texts):
        return self.embedding.embed_documents(texts)

    def get_embeddings(self):
        """返回 Embeddings 对象，给 FAISS 使用"""
        return self.embedding

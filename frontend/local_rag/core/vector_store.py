from typing import Optional
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from frontend.local_rag.config.settings import Settings
from frontend.local_rag.utils.file_utils import ensure_dir


class VectorStoreManager:
    """FAISS 向量库管理"""

    def __init__(self, settings: Settings, embeddings) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.index_dir = ensure_dir(settings.faiss_index_dir)
        self._vector_store: Optional[FAISS] = None

    @property
    def is_ready(self) -> bool:
        return self._vector_store is not None

    # -------------------------
    # 加载
    # -------------------------
    def load(self) -> bool:
        index_path = self.index_dir / "index.faiss"

        if not index_path.exists():
            return False

        self._vector_store = FAISS.load_local(
            folder_path=str(self.index_dir),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )
        return True

    # -------------------------
    # 添加文档
    # -------------------------
    def add_documents(self, documents: list[Document]) -> int:
        if not documents:
            return 0

        if self._vector_store is None:
            self._vector_store = FAISS.from_documents(
                documents,
                self.embeddings
            )
        else:
            self._vector_store.add_documents(documents)

        self.save()
        return len(documents)

    # -------------------------
    # ⭐ 核心：搜索（你缺的就是这个）
    # -------------------------
    def search(self, query: str, k: int = 4):
        if self._vector_store is None:
            return []

        docs = self._vector_store.similarity_search(query, k=k)

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in docs
        ]

    # -------------------------
    # 保存
    # -------------------------
    def save(self) -> None:
        if self._vector_store is None:
            return

        self._vector_store.save_local(str(self.index_dir))

    # -------------------------
    # 清空
    # -------------------------
    def clear(self) -> None:
        self._vector_store = None

        if self.index_dir.exists():
            for file in self.index_dir.iterdir():
                if file.is_file():
                    file.unlink()
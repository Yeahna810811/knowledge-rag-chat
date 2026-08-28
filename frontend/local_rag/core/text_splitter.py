from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from frontend.local_rag.config.settings import Settings


def split_documents(documents: list[Document], settings: Settings) -> list[Document]:
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],
    )
    return splitter.split_documents(documents)

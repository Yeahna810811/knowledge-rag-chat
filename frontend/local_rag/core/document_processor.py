from pathlib import Path

from langchain_core.documents import Document

from frontend.local_rag.config.settings import Settings
from frontend.local_rag.core.document_loader import load_document
from frontend.local_rag.core.text_splitter import split_documents


class DocumentProcessor:
    """Load uploaded files and convert them into text chunks."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def process_file(self, file_path: Path) -> list[Document]:
        documents = load_document(file_path)
        return split_documents(documents, self.settings)

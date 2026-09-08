from pathlib import Path

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

from frontend.local_rag.utils.file_utils import get_file_extension


def load_document(file_path: Path) -> list[Document]:
    """Load a local file into LangChain Document objects."""
    suffix = get_file_extension(file_path.name)

    if suffix in {".txt", ".md"}:
        # Markdown uses TextLoader for fewer optional deps; structure is preserved for chunking.
        loader = TextLoader(str(file_path), encoding="utf-8")
    elif suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(file_path))
    elif suffix == ".csv":
        loader = CSVLoader(str(file_path))
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    documents = loader.load()
    for doc in documents:
        doc.metadata.setdefault("source", file_path.name)
    return documents

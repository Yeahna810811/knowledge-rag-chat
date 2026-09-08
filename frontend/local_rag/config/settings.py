from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# local_rag 包所在目录（.../frontend/local_rag）
PACKAGE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """应用配置，从环境变量 / .env 文件加载"""

    model_config = SettingsConfigDict(
        env_file=str(PACKAGE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- 向量检索（本地 embedding） ----
    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    # ---- 阿里云百炼 DashScope（AI聊天大模型） ----
    dashscope_api_key: str = ""
    chat_model: str = "qwen-plus"
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # ---- 文本切分 ----
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_top_k: int = 4

    # ---- LangSmith 可观测性（可选） ----
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "knowledge-rag-chat"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    # ---- CI / Webhook ----
    webhook_secret: str = ""

    # ---- 存储路径（绝对路径，不受启动时所在目录影响） ----
    upload_dir: Path = PACKAGE_DIR / "data" / "uploads"
    faiss_index_dir: Path = PACKAGE_DIR / "data" / "faiss_index"

    # ---- 服务 ----
    host: str = "0.0.0.0"
    port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()


def configure_observability(settings: Optional[Settings] = None) -> None:
    """Enable LangSmith tracing from settings / environment when configured."""
    import os

    settings = settings or get_settings()
    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
    elif os.environ.get("LANGCHAIN_TRACING_V2", "").lower() in {"1", "true", "yes"}:
        # Honor pre-set env vars even if .env bool is false
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.langchain_project)

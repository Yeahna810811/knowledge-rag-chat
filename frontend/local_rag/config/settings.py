from functools import lru_cache
from pathlib import Path

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

    # ---- 向量检索（本地免费 embedding，无需API Key） ----
    embedding_model: str = "all-MiniLM-L6-v2"

    # ---- 阿里云百炼 DashScope（AI聊天大模型） ----
    dashscope_api_key: str = ""
    chat_model: str = "qwen-plus"
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # ---- 文本切分 ----
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_top_k: int = 4

    # ---- 存储路径（绝对路径，不受启动时所在目录影响） ----
    upload_dir: Path = PACKAGE_DIR / "data" / "uploads"
    faiss_index_dir: Path = PACKAGE_DIR / "data" / "faiss_index"

    # ---- 服务 ----
    host: str = "0.0.0.0"
    port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()

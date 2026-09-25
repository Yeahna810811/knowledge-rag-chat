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

    # ---- 检索策略 ----
    # dense  = 纯 FAISS 向量（v1 原始行为）
    # bm25   = BM25 主导 + 稠密兜底（默认，依据见 evaluation/EVALUATION_AUDIT.md）
    # hybrid = RRF 融合两路（实验用，实测不优于 bm25）
    retrieval_mode: str = "bm25"
    # BM25 主导模式下，是否允许在稀疏路零命中时回落到稠密路
    retrieval_dense_fallback: bool = True
    # 纯 BM25 模式下跳过 embedding 模型加载（冷启动不再依赖模型文件）
    retrieval_lazy_dense: bool = True
    # hybrid 模式下 BM25 相对稠密路的 RRF 权重
    hybrid_bm25_weight: float = 1.0
    hybrid_dense_weight: float = 1.0

    # ---- 查询改写（解决口语化提问与文档用词之间的「词汇不匹配」） ----
    # off = 不改写（对照 / 排障）
    # prf = 伪相关反馈（默认，离线无依赖，扩展词取自语料自身用词）
    # llm = 用大模型改写口语化问题（更准，但多一次 LLM 调用）
    retrieval_query_rewrite: str = "prf"
    # 扩展词相对原查询的权重。原查询恒为 1.0，扩展部分默认只占 0.3——
    # 等权 RRF 的教训：低质量信号拿到过高权重会把正确结果挤下去。
    query_rewrite_weight: float = 0.3
    # 伪相关反馈的扩展词数量与反馈文档数
    query_rewrite_terms: int = 6
    query_rewrite_feedback_docs: int = 3

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
    bm25_index_dir: Path = PACKAGE_DIR / "data" / "bm25_index"

    # ---- 数据库（会话持久化）----
    # 生产 / Docker 用 MySQL：
    #   mysql+pymysql://rag_user:rag_password@mysql:3306/knowledge_rag?charset=utf8mb4
    # 缺省回落到 SQLite 文件，保证本地与 CI 不强制依赖 MySQL 服务。
    # 密码一律走环境变量，不写死在代码里。
    database_url: str = f"sqlite:///{PACKAGE_DIR / 'data' / 'rag_app.db'}"
    database_echo: bool = False
    # 仅对 MySQL 等非 SQLite 后端生效的连接池参数
    database_pool_size: int = 5
    database_max_overflow: int = 10

    # ---- Redis（检索缓存 + 限流；不是数据源，挂掉只降级不中断）----
    # Docker Compose 内部网络写服务名：redis://redis:6379/0
    # 需要密码时直接带在 URL 里：redis://:password@host:6379/0
    redis_url: str = "redis://127.0.0.1:6379/0"
    # Redis 超时刻意设短：缓存查不到大不了多检索一次，
    # 不能让一次缓存查询把整个 /api/ask 拖住。
    redis_socket_connect_timeout: float = 1.0
    redis_socket_timeout: float = 1.0

    # ---- Retrieval Cache ----
    redis_cache_enabled: bool = True
    redis_cache_ttl_seconds: int = 600

    # ---- Rate Limit（/api/ask，按 session_id 计）----
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60

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

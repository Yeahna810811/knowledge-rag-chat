"""数据库持久化层。

对外只暴露三个入口：Database（连接）、Base（ORM 基类）、Conversation / Message（模型）。
"""

from frontend.local_rag.db.database import Base, Database, utc_now
from frontend.local_rag.db.models import (
    MODE_CHAT,
    MODE_RAG,
    VALID_MODES,
    Conversation,
    Message,
)

__all__ = [
    "Base",
    "Database",
    "Conversation",
    "Message",
    "MODE_RAG",
    "MODE_CHAT",
    "VALID_MODES",
    "utc_now",
]

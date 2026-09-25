"""会话持久化 ORM 模型。

两张表：

    conversations  一个 session_id 一行
    messages       一轮问答一行（question / answer / mode）

关系：Conversation 1 -> N Message，删除 Conversation 时级联删除其 Message。

兼容性：模型必须同时能在 MySQL 8 与 SQLite 上建表——主键类型与长文本类型
都做了 dialect variant，单元测试因此可以用临时 SQLite 跑，不依赖 MySQL 服务。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import Mapped, mapped_column, relationship

from frontend.local_rag.db.database import Base, utc_now

# MySQL 用 BIGINT；SQLite 的自增主键必须是 INTEGER（否则 AUTOINCREMENT 不生效）
PK = BigInteger().with_variant(Integer, "sqlite")

# 长答案可能远超 MySQL TEXT 的 64KB，MySQL 侧用 LONGTEXT，其它后端回落无长度 TEXT
LONG_TEXT = Text().with_variant(LONGTEXT, "mysql")

# MySQL 侧统一 utf8mb4 + InnoDB，保证 emoji / 生僻字不被截断，且外键真正生效
MYSQL_TABLE_ARGS = {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"}

# 与 GenerationAgent / routes 的 Literal["rag", "chat"] 保持一致
MODE_RAG = "rag"
MODE_CHAT = "chat"
VALID_MODES = (MODE_RAG, MODE_CHAT)


class Conversation(Base):
    """一次会话（对应一个 session_id）。"""

    __tablename__ = "conversations"
    __table_args__ = (
        # session_id 唯一 + 建索引（唯一索引同时满足「唯一」和「可快速按 id 查」）
        Index("ix_conversations_session_id", "session_id", unique=True),
        MYSQL_TABLE_ARGS,
    )

    id: Mapped[int] = mapped_column(PK, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=utc_now, onupdate=utc_now
    )

    messages: Mapped[list["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        # ORM 级级联：不依赖数据库外键是否开启（SQLite 默认关闭外键）
        passive_deletes=False,
        order_by="Message.id",
    )

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"Conversation(id={self.id}, session_id={self.session_id!r})"


class Message(Base):
    """一轮问答。

    注意：这里存的是「全部历史」，不限 10 条。
    只取最近 10 轮是读取侧（ConversationStore.get_recent_history）的责任，
    这样历史可以无限增长，而喂给 LLM 的 context 恒定为最近 N 轮。
    """

    __tablename__ = "messages"
    __table_args__ = (
        # 查历史的主查询是 where conversation_id = ? order by created_at desc limit N
        Index("ix_messages_conversation_created", "conversation_id", "created_at"),
        MYSQL_TABLE_ARGS,
    )

    id: Mapped[int] = mapped_column(PK, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        PK,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    question: Mapped[str] = mapped_column(LONG_TEXT, nullable=False)
    answer: Mapped[str] = mapped_column(LONG_TEXT, nullable=False)
    mode: Mapped[str] = mapped_column(String(16), nullable=False, default=MODE_RAG)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=utc_now
    )

    conversation: Mapped["Conversation"] = relationship(
        "Conversation", back_populates="messages"
    )

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"Message(id={self.id}, conversation_id={self.conversation_id}, mode={self.mode!r})"

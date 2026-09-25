"""会话历史的数据库读写封装。

KnowledgeService 只跟这一层打交道，不直接碰 Session / ORM 模型。

两条最重要的约束：

1. 返回给 GenerationAgent 的 history 格式恒为
   `[{"question": ..., "answer": ...}, ...]`，与改造前的内存 dict 完全一致，
   所以 GenerationAgent / RetrievalAgent 一行都不用改。
2. 数据库保存全部历史，但 `get_recent_history()` 只取最近 N 轮，
   且返回顺序必须是「旧 -> 新」，这样 LLM 拿到的多轮对话顺序才正确。
"""

from __future__ import annotations

import logging

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from frontend.local_rag.db.database import Database, utc_now
from frontend.local_rag.db.models import MODE_RAG, VALID_MODES, Conversation, Message

logger = logging.getLogger(__name__)

MAX_SESSION_ID_LENGTH = 128


def normalize_session_id(session_id: str) -> str:
    """校验并规范化 session_id（1~128 字符）。"""
    sid = (session_id or "").strip()
    if not sid:
        raise ValueError("session_id 不能为空")
    if len(sid) > MAX_SESSION_ID_LENGTH:
        raise ValueError(f"session_id 长度不能超过 {MAX_SESSION_ID_LENGTH} 个字符")
    return sid


def normalize_mode(mode: str | None) -> str:
    """mode 只落库 'rag' / 'chat'，其它值回落 'rag'。"""
    normalized = (mode or MODE_RAG).strip().lower()
    return normalized if normalized in VALID_MODES else MODE_RAG


class ConversationStore:
    """会话持久化仓储。所有方法都自己开/关 Session，不跨调用持有连接。"""

    def __init__(self, database: Database) -> None:
        self._database = database

    @property
    def database(self) -> Database:
        return self._database

    # ---------------------------------------------------------------- 内部
    @staticmethod
    def _find_conversation(session: Session, session_id: str) -> Conversation | None:
        return session.execute(
            select(Conversation).where(Conversation.session_id == session_id)
        ).scalar_one_or_none()

    @staticmethod
    def _get_or_create(session: Session, session_id: str) -> Conversation:
        """同一 session_id 只创建一次。

        并发下两个请求可能同时走到 insert，靠唯一索引兜底：
        捕获 IntegrityError 后回滚并重新查询，而不是让请求 500。
        """
        existing = ConversationStore._find_conversation(session, session_id)
        if existing is not None:
            return existing

        try:
            conversation = Conversation(session_id=session_id)
            session.add(conversation)
            session.flush()  # 立刻拿到自增 id
            return conversation
        except IntegrityError:
            session.rollback()
            again = ConversationStore._find_conversation(session, session_id)
            if again is None:
                raise
            return again

    @staticmethod
    def _to_turns(rows) -> list[dict]:
        return [{"question": question, "answer": answer} for question, answer in rows]

    # ---------------------------------------------------------------- 写入
    def get_or_create_conversation(self, session_id: str) -> Conversation:
        sid = normalize_session_id(session_id)
        with self._database.session() as session:
            return self._get_or_create(session, sid)

    def append_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        mode: str = MODE_RAG,
    ) -> Message:
        """写入一轮问答。数据库保存全部历史，不在此处裁剪。"""
        sid = normalize_session_id(session_id)
        normalized_mode = normalize_mode(mode)

        with self._database.session() as session:
            conversation = self._get_or_create(session, sid)
            message = Message(
                conversation_id=conversation.id,
                question=question,
                answer=answer,
                mode=normalized_mode,
            )
            session.add(message)
            conversation.updated_at = utc_now()
            session.flush()
            return message

    # ---------------------------------------------------------------- 读取
    def get_recent_history(self, session_id: str, limit: int = 10) -> list[dict]:
        """取最近 N 轮，返回顺序为旧 -> 新（LLM 对话顺序）。

        先按 created_at/id 倒序取 N 条，再整体反转——比「取全部再切片」省内存，
        也比「正序 limit」语义正确（正序 limit 拿到的是最早的 N 条）。
        """
        if limit <= 0:
            return []

        sid = normalize_session_id(session_id)
        with self._database.session() as session:
            conversation = self._find_conversation(session, sid)
            if conversation is None:
                return []
            rows = session.execute(
                select(Message.question, Message.answer)
                .where(Message.conversation_id == conversation.id)
                .order_by(Message.created_at.desc(), Message.id.desc())
                .limit(limit)
            ).all()

        return self._to_turns(reversed(rows))

    def get_history(self, session_id: str) -> list[dict]:
        """取该会话的全部历史，顺序旧 -> 新。"""
        sid = normalize_session_id(session_id)
        with self._database.session() as session:
            conversation = self._find_conversation(session, sid)
            if conversation is None:
                return []
            rows = session.execute(
                select(Message.question, Message.answer)
                .where(Message.conversation_id == conversation.id)
                .order_by(Message.created_at.asc(), Message.id.asc())
            ).all()

        return self._to_turns(rows)

    # ---------------------------------------------------------------- 统计 / 清理
    def count_turns(self, session_id: str) -> int:
        sid = normalize_session_id(session_id)
        with self._database.session() as session:
            conversation = self._find_conversation(session, sid)
            if conversation is None:
                return 0
            return int(
                session.execute(
                    select(func.count(Message.id)).where(
                        Message.conversation_id == conversation.id
                    )
                ).scalar_one()
            )

    def count_sessions(self) -> int:
        with self._database.session() as session:
            return int(session.execute(select(func.count(Conversation.id))).scalar_one())

    def clear_history(self, session_id: str) -> bool:
        """删除该会话及其全部消息，返回是否真的删掉了东西。"""
        sid = normalize_session_id(session_id)
        with self._database.session() as session:
            conversation = self._find_conversation(session, sid)
            if conversation is None:
                return False
            # 先批量删消息（不依赖数据库外键是否开启），再删会话
            session.execute(
                delete(Message).where(Message.conversation_id == conversation.id)
            )
            session.delete(conversation)
        return True

    def ping(self) -> bool:
        return self._database.ping()

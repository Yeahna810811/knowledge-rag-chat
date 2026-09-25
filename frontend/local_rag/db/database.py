"""数据库连接统一管理入口。

设计原则：全项目只有这里能调用 `create_engine()`。业务代码（KnowledgeService、
ConversationStore）只拿 `Database` 实例，不自己拼 URL、不自己建 engine。

后端兼容：
- 生产 / Docker：MySQL 8（`mysql+pymysql://...`，带连接池与 pool_pre_ping）
- 本地 / CI / 单元测试：SQLite 文件（`sqlite:///...`，零外部依赖）

URL 统一由 `DATABASE_URL` 环境变量提供，缺省时回落到 SQLite 文件，
保证 GitHub Actions 在没有 MySQL 服务的情况下不会直接失败。
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """全项目唯一的 ORM Declarative Base。"""


def utc_now() -> datetime:
    """返回 naive UTC 时间。

    不用 `datetime.utcnow()`（已弃用），也不用 tz-aware：MySQL 的 DATETIME
    不存时区，SQLite 更没有时区类型，带 tzinfo 反而会在落库时被静默丢掉，
    读回来再和 naive 值比较就会炸。统一存 naive UTC 是两边都能自洽的选择。
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def backend_name(url: str) -> str:
    """从 URL 里取后端名：mysql / sqlite / postgresql ..."""
    try:
        return make_url(url).get_backend_name()
    except Exception:  # noqa: BLE001 - 畸形 URL 时至少不要在这里炸
        return "unknown"


def _sqlite_path(url: str) -> Path | None:
    """从 sqlite URL 中取出数据库文件路径（非 sqlite 返回 None）。

    用 SQLAlchemy 自己的 make_url 解析，而不是手写字符串切分——
    三斜杠 / 四斜杠 / Windows 盘符这些写法自己切很容易切错。
    """
    if backend_name(url) != "sqlite":
        return None
    try:
        database = make_url(url).database
    except Exception:  # noqa: BLE001
        return None
    return Path(database) if database else None


def _build_engine(
    url: str,
    echo: bool = False,
    pool_size: int = 5,
    max_overflow: int = 10,
) -> Engine:
    kwargs: dict[str, Any] = {"echo": echo, "future": True}

    if backend_name(url) == "sqlite":
        db_path = _sqlite_path(url)
        if db_path is not None:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        # FastAPI 会在线程池里跑同步接口，默认只允许创建连接的那个线程用
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        # MySQL / 其它服务端数据库：启用生产级连接池配置
        kwargs.update(
            {
                "pool_pre_ping": True,  # 每次取连接先探活，避免 MySQL wait_timeout 断连
                "pool_size": pool_size,
                "max_overflow": max_overflow,
                "pool_recycle": 2800,  # 小于 MySQL 默认 wait_timeout(28800)
                "pool_timeout": 30,
            }
        )

    return create_engine(url, **kwargs)


class Database:
    """持有 engine 与 sessionmaker，是唯一的数据库连接出口。"""

    def __init__(
        self,
        url: str,
        echo: bool = False,
        pool_size: int = 5,
        max_overflow: int = 10,
    ) -> None:
        if not url or not url.strip():
            raise ValueError("database_url 不能为空")

        self.url = url.strip()
        self.echo = echo
        self._engine: Engine = _build_engine(self.url, echo, pool_size, max_overflow)
        self._session_factory = sessionmaker(
            bind=self._engine,
            autoflush=False,
            # 关闭过期：commit 后仍能读取对象属性，省一次刷新查询，
            # 也避免 ORM 对象出了 session 就被 detach 得干干净净。
            expire_on_commit=False,
            future=True,
        )

    # ---------------------------------------------------------------- 属性
    @property
    def engine(self) -> Engine:
        return self._engine

    @property
    def session_factory(self) -> sessionmaker[Session]:
        return self._session_factory

    @property
    def backend(self) -> str:
        """返回 'mysql' / 'sqlite' 等，供 /api/status 展示与测试断言使用。"""
        return backend_name(self.url)

    @property
    def is_sqlite(self) -> bool:
        return self.backend == "sqlite"

    # ---------------------------------------------------------------- 生命周期
    @contextmanager
    def session(self) -> Iterator[Session]:
        """一次数据库操作的 session 作用域：正常 commit，异常 rollback，必定 close。

        粒度是「一次业务操作」，不是「一次 HTTP 请求」——engine 全局复用，
        session 即用即关，避免连接泄漏。
        """
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_tables(self) -> None:
        """建表（幂等）。导入放在函数内，避免 database <-> models 循环导入。"""
        from frontend.local_rag.db.models import Conversation, Message  # noqa: F401

        Base.metadata.create_all(self._engine)

    def drop_tables(self) -> None:
        """仅测试使用：删表。"""
        from frontend.local_rag.db.models import Conversation, Message  # noqa: F401

        Base.metadata.drop_all(self._engine)

    def ping(self) -> bool:
        """连通性探测。注意：不抛异常，失败返回 False——/api/status 依赖这一点。"""
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:  # noqa: BLE001 - 健康检查必须吞掉一切
            logger.warning("数据库 ping 失败: %s", exc)
            return False

    def dispose(self) -> None:
        """释放连接池，应用关闭时调用。"""
        self._engine.dispose()

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"Database(backend={self.backend!r}, echo={self.echo})"

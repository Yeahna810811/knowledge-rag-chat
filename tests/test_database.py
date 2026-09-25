"""会话持久化层（SQLAlchemy + ConversationStore）的单元测试。

    python tests/test_database.py

默认使用临时 SQLite 文件，不需要 MySQL 服务——CI 可以直接跑。
如果 TEST_DATABASE_URL 或 DATABASE_URL 指向 MySQL，会额外执行一轮真实
MySQL integration test（连不上时记为跳过，不算失败）。
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import func, select  # noqa: E402

from frontend.local_rag.db.database import Database  # noqa: E402
from frontend.local_rag.db.models import Conversation, Message  # noqa: E402
from frontend.local_rag.services.conversation_store import (  # noqa: E402
    ConversationStore,
)

PASSED = 0
FAILED = 0
SKIPPED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [ok]   {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name} {detail}")


def skip(name: str, reason: str) -> None:
    global SKIPPED
    SKIPPED += 1
    print(f"  [skip] {name}  {reason}")


def new_store(db_file: Path) -> tuple[Database, ConversationStore]:
    """在一个临时 SQLite 文件上新建一个完整的 Database + Store。"""
    database = Database(f"sqlite:///{db_file}")
    database.create_tables()
    return database, ConversationStore(database)


# --------------------------------------------------------------------------- 1
def test_create_conversation() -> None:
    print("\n[1] 创建 conversation")
    with tempfile.TemporaryDirectory() as tmp:
        db, store = new_store(Path(tmp) / "t1.db")
        try:
            conv = store.get_or_create_conversation("session-create")
            check("conversation 已创建", conv is not None)
            check("session_id 正确", conv.session_id == "session-create")
            check("自增 id 已生成", conv.id is not None)

            again = store.get_or_create_conversation("session-create")
            check("重复调用复用同一行", again.id == conv.id)
            check("会话数仍为 1", store.count_sessions() == 1)

            other = store.get_or_create_conversation("session-create-2")
            check("不同 session_id 建不同行", other.id != conv.id)
            check("会话数变为 2", store.count_sessions() == 2)
        finally:
            db.dispose()


# --------------------------------------------------------------------------- 2
def test_append_one_turn() -> None:
    print("\n[2] 写入一轮 question/answer")
    with tempfile.TemporaryDirectory() as tmp:
        db, store = new_store(Path(tmp) / "t2.db")
        try:
            store.append_turn("s2", "知识库默认端口是多少？", "默认端口是 8000。", "rag")
            history = store.get_history("s2")

            check("历史有 1 轮", len(history) == 1, f"got {len(history)}")
            check(
                "question 正确",
                history[0]["question"] == "知识库默认端口是多少？",
                str(history),
            )
            check("answer 正确", history[0]["answer"] == "默认端口是 8000。", str(history))
            check("格式与 GenerationAgent 兼容", set(history[0]) == {"question", "answer"})
            check("count_turns 为 1", store.count_turns("s2") == 1)
        finally:
            db.dispose()


# --------------------------------------------------------------------------- 3
def test_persistence_across_restart() -> None:
    print("\n[3] 重建 Database/ConversationStore 后历史仍在（模拟服务重启）")
    with tempfile.TemporaryDirectory() as tmp:
        db_file = Path(tmp) / "t3.db"

        db1, store1 = new_store(db_file)
        try:
            store1.append_turn("s3", "第一轮问题", "第一轮答案", "rag")
            store1.append_turn("s3", "第二轮问题", "第二轮答案", "chat")
        finally:
            db1.dispose()  # 模拟进程退出，连接池全部释放

        db2, store2 = new_store(db_file)  # 模拟服务重启，全新 Database 对象
        try:
            history = store2.get_history("s3")
            check("重启后仍有 2 轮", len(history) == 2, f"got {len(history)}")
            check("重启后内容一致", history[0]["question"] == "第一轮问题")
            check("重启后第二轮一致", history[1]["answer"] == "第二轮答案")
            check("重启后会话数不重复计", store2.count_sessions() == 1)
        finally:
            db2.dispose()


# --------------------------------------------------------------------------- 4
def test_sessions_are_isolated() -> None:
    print("\n[4] 两个 session_id 数据互不影响")
    with tempfile.TemporaryDirectory() as tmp:
        db, store = new_store(Path(tmp) / "t4.db")
        try:
            store.append_turn("alice", "A 的问题 1", "A 的答案 1", "rag")
            store.append_turn("alice", "A 的问题 2", "A 的答案 2", "rag")
            store.append_turn("bob", "B 的问题 1", "B 的答案 1", "chat")

            alice = store.get_history("alice")
            bob = store.get_history("bob")

            check("alice 有 2 轮", len(alice) == 2, f"got {len(alice)}")
            check("bob 有 1 轮", len(bob) == 1, f"got {len(bob)}")
            check("bob 看不到 alice 的历史", bob[0]["question"] == "B 的问题 1")
            check(
                "alice 看不到 bob 的历史",
                all(turn["question"].startswith("A 的问题") for turn in alice),
            )
            check("两个会话分别计数", store.count_sessions() == 2)
            check("不存在的 session 返回空", store.get_history("nobody") == [])
        finally:
            db.dispose()


# --------------------------------------------------------------------------- 5
def test_full_history_vs_recent_window() -> None:
    print("\n[5] 超过 10 轮：数据库保存全部，get_recent_history 只返回最近 10 轮")
    with tempfile.TemporaryDirectory() as tmp:
        db, store = new_store(Path(tmp) / "t5.db")
        try:
            total = 15
            for i in range(1, total + 1):
                store.append_turn("s5", f"问题{i}", f"答案{i}", "rag")

            full = store.get_history("s5")
            recent = store.get_recent_history("s5", limit=10)

            check(f"数据库保存全部 {total} 轮", len(full) == total, f"got {len(full)}")
            check("最近窗口只返回 10 轮", len(recent) == 10, f"got {len(recent)}")
            check(
                "窗口内是最新的 10 轮（问题6~问题15）",
                recent[0]["question"] == "问题6" and recent[-1]["question"] == "问题15",
                f"{recent[0]['question']} .. {recent[-1]['question']}",
            )
            check(
                "窗口是全文历史的尾部切片",
                [t["question"] for t in recent] == [t["question"] for t in full[-10:]],
            )
            check("count_turns 统计全量", store.count_turns("s5") == total)
        finally:
            db.dispose()


# --------------------------------------------------------------------------- 6
def test_recent_history_order() -> None:
    print("\n[6] 最近 10 轮返回顺序必须是旧 -> 新")
    with tempfile.TemporaryDirectory() as tmp:
        db, store = new_store(Path(tmp) / "t6.db")
        try:
            for i in range(1, 13):
                store.append_turn("s6", f"Q{i}", f"A{i}", "rag")

            recent = store.get_recent_history("s6", limit=10)
            questions = [turn["question"] for turn in recent]
            answers = [turn["answer"] for turn in recent]

            check("问题顺序 Q3 -> Q12", questions == [f"Q{i}" for i in range(3, 13)], str(questions))
            check("答案顺序 A3 -> A12", answers == [f"A{i}" for i in range(3, 13)], str(answers))
            check(
                "与数据库全量顺序一致（旧 -> 新）",
                questions == [t["question"] for t in store.get_history("s6")][-10:],
            )
        finally:
            db.dispose()


# --------------------------------------------------------------------------- 7
def test_clear_history() -> None:
    print("\n[7] clear_history 后对应会话不再有历史")
    with tempfile.TemporaryDirectory() as tmp:
        db, store = new_store(Path(tmp) / "t7.db")
        try:
            store.append_turn("s7-a", "问题", "答案", "rag")
            store.append_turn("s7-a", "问题2", "答案2", "rag")
            store.append_turn("s7-b", "问题", "答案", "rag")

            deleted = store.clear_history("s7-a")
            check("返回已删除", deleted is True)
            check("被清空的会话历史为空", store.get_history("s7-a") == [])
            check("被清空会话 turns 为 0", store.count_turns("s7-a") == 0)
            check("其它会话不受影响", len(store.get_history("s7-b")) == 1)
            check("会话数减 1", store.count_sessions() == 1)
            check("重复清空幂等", store.clear_history("s7-a") is False)

            # 级联：会话删掉后，Message 不能留下孤儿行
            with db.session() as session:
                orphans = session.query(Message).count()
            check("消息已级联删除（无孤儿行）", orphans == 1, f"orphans={orphans}")
        finally:
            db.dispose()


# --------------------------------------------------------------------------- 8
def test_modes() -> None:
    print("\n[8] rag / chat 两种 mode 都能保存")
    with tempfile.TemporaryDirectory() as tmp:
        db, store = new_store(Path(tmp) / "t8.db")
        try:
            store.append_turn("s8", "知识库问题", "知识库答案", "rag")
            store.append_turn("s8", "闲聊问题", "闲聊答案", "chat")

            with db.session() as session:
                rows = (
                    session.query(Message)
                    .join(Conversation, Message.conversation_id == Conversation.id)
                    .filter(Conversation.session_id == "s8")
                    .order_by(Message.id)
                    .all()
                )
            check("两条消息都落库", len(rows) == 2, f"got {len(rows)}")
            check("mode 分别为 rag / chat", [r.mode for r in rows] == ["rag", "chat"], str([r.mode for r in rows]))
            check("历史读取不受 mode 影响", len(store.get_history("s8")) == 2)

            # 非法 mode 必须被规范化，不能污染数据
            store.append_turn("s8", "未知模式", "答案", "weird")
            with db.session() as session:
                modes = [r.mode for r in session.query(Message).order_by(Message.id).all()]
            check("非法 mode 回落 rag", modes[-1] == "rag", str(modes))
        finally:
            db.dispose()


# --------------------------------------------------------------------------- 9
def test_ping() -> None:
    print("\n[9] 数据库 ping")
    with tempfile.TemporaryDirectory() as tmp:
        db, store = new_store(Path(tmp) / "t9.db")
        try:
            check("store.ping() 正常", store.ping() is True)
            check("database.ping() 正常", db.ping() is True)
            check("backend 识别为 sqlite", db.backend == "sqlite")

            # 指向一个目录而不是文件：SQLite 必然打开失败。
            # 这里验证的是「ping 不抛异常」——/api/status 依赖这个行为。
            dead = Database(f"sqlite:///{tmp}")
            check("不可用时 ping 返回 False 而不是抛异常", dead.ping() is False)
            dead.dispose()
        finally:
            db.dispose()


# -------------------------------------------------------------------------- 10
def test_concurrent_create() -> None:
    print("\n[10] 并发创建同一 session_id（覆盖唯一键冲突的重试分支）")
    with tempfile.TemporaryDirectory() as tmp:
        db, store = new_store(Path(tmp) / "t10.db")
        try:
            sid = "race-session"
            errors: list[str] = []
            ids: list[int] = []
            barrier = threading.Barrier(6)

            def worker() -> None:
                try:
                    barrier.wait()  # 尽量让所有线程同时撞唯一键
                    ids.append(store.get_or_create_conversation(sid).id)
                except Exception as exc:  # noqa: BLE001
                    errors.append(repr(exc))

            threads = [threading.Thread(target=worker) for _ in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            check("6 线程并发创建不抛异常", not errors, str(errors[:2]))
            check("只产生一个 conversation id", len(set(ids)) == 1, f"ids={set(ids)}")
            with db.session() as session:
                rows = session.execute(
                    select(func.count(Conversation.id)).where(Conversation.session_id == sid)
                ).scalar_one()
            check("数据库里只落 1 行", rows == 1, f"rows={rows}")
        finally:
            db.dispose()


# -------------------------------------------------------------------------- 11
def test_utf8mb4_roundtrip() -> None:
    print("\n[11] 四字节字符（emoji）往返")
    with tempfile.TemporaryDirectory() as tmp:
        db, store = new_store(Path(tmp) / "t11.db")
        try:
            # MySQL 侧只有 utf8mb4 存得下；utf8(utf8mb3) 会直接报错或截断
            text = "你好 👋🚀😀 中文与 emoji 混排 —— ünïcode ✅"
            store.append_turn("utf8", text, text, "rag")
            turn = store.get_history("utf8")[0]
            check("question 无损", turn["question"] == text, repr(turn["question"]))
            check("answer 无损", turn["answer"] == text, repr(turn["answer"]))
        finally:
            db.dispose()


# -------------------------------------------------------------------------- 12
def test_long_text() -> None:
    print("\n[12] 超长 answer（覆盖 MySQL LONGTEXT）")
    with tempfile.TemporaryDirectory() as tmp:
        db, store = new_store(Path(tmp) / "t12.db")
        try:
            # 40000 个汉字 ≈ 120KB，远超 MySQL TEXT 的 65535 字节上限
            long_answer = "长" * 40000
            store.append_turn("longtext", "长文本问题", long_answer, "rag")
            turn = store.get_history("longtext")[0]
            check(
                "超长 answer 完整存回",
                turn["answer"] == long_answer,
                f"got {len(turn['answer'])} chars, want {len(long_answer)}",
            )
        finally:
            db.dispose()


# ------------------------------------------------------------------ MySQL 集成
def _mysql_url() -> str:
    for var in ("TEST_DATABASE_URL", "DATABASE_URL"):
        url = os.environ.get(var, "").strip()
        if url.startswith("mysql"):
            return url
    return ""


def test_mysql_integration() -> None:
    print("\n[MySQL integration]")
    url = _mysql_url()
    if not url:
        skip("真实 MySQL 集成测试", "未设置 TEST_DATABASE_URL / DATABASE_URL，改用 SQLite")
        return

    db = Database(url)
    try:
        if not db.ping():
            skip("真实 MySQL 集成测试", "MySQL 不可达")
            return
        db.create_tables()
        store = ConversationStore(db)
        sid = "mysql-integration-test"

        store.clear_history(sid)
        store.append_turn(sid, "MySQL 问题 1", "MySQL 答案 1", "rag")
        for i in range(2, 13):
            store.append_turn(sid, f"MySQL 问题 {i}", f"MySQL 答案 {i}", "chat")

        full = store.get_history(sid)
        recent = store.get_recent_history(sid, limit=10)
        check("MySQL 保存全部 12 轮", len(full) == 12, f"got {len(full)}")
        check("MySQL 最近窗口 10 轮", len(recent) == 10, f"got {len(recent)}")
        check(
            "MySQL 窗口顺序旧 -> 新",
            recent[0]["question"] == "MySQL 问题 3" and recent[-1]["question"] == "MySQL 问题 12",
            str([t["question"] for t in recent]),
        )
        check("MySQL backend 识别正确", db.backend == "mysql")

        store.clear_history(sid)
        check("MySQL clear_history 生效", store.get_history(sid) == [])
    finally:
        db.dispose()


def main() -> None:
    test_create_conversation()
    test_append_one_turn()
    test_persistence_across_restart()
    test_sessions_are_isolated()
    test_full_history_vs_recent_window()
    test_recent_history_order()
    test_clear_history()
    test_modes()
    test_ping()
    test_concurrent_create()
    test_utf8mb4_roundtrip()
    test_long_text()
    test_mysql_integration()

    print("\n" + "=" * 46)
    print(f"通过 {PASSED} 项，失败 {FAILED} 项，跳过 {SKIPPED} 项")
    print("=" * 46)
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()

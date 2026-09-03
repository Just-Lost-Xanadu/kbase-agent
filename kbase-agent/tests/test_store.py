"""会话记录 store 单测：不依赖外部模型/网络，用临时 sqlite 文件。"""

import asyncio

from app import store


def _run(coro):
    return asyncio.run(coro)


def test_save_and_read_roundtrip(tmp_path, monkeypatch):
    db = tmp_path / "checkpoints.sqlite"
    monkeypatch.setattr(store, "DB_PATH", str(db))
    _run(store.init_db())
    assert store.has_schema(str(db))

    _run(
        store.save_turn(
            "s1",
            "张三还剩几天年假？",
            "张三还剩几天年假？",
            "张三剩 6 天，最多结转 3 天。",
            ["员工手册_示例.md"],
        )
    )
    sessions = _run(store.list_sessions())
    assert len(sessions) == 1
    assert sessions[0]["id"] == "s1"
    assert sessions[0]["message_count"] == 2
    assert "张三" in sessions[0]["last_preview"]

    # 第二轮走同一会话：消息累计、标题不重复覆盖
    _run(
        store.save_turn(
            "s1",
            "张三还剩几天年假？",
            "那离职能折算吗？",
            "离职未休年假按日工资折算。",
            ["员工手册_示例.md"],
        )
    )
    messages = _run(store.get_messages("s1"))
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[-1]["sources"] == ["员工手册_示例.md"]
    assert len(_run(store.list_sessions())) == 1


def test_sessions_ordered_by_activity(tmp_path, monkeypatch):
    db = tmp_path / "checkpoints.sqlite"
    monkeypatch.setattr(store, "DB_PATH", str(db))
    _run(store.init_db())
    _run(store.save_turn("a", "旧会话", "旧会话", "答", []))
    _run(store.save_turn("b", "新会话", "新会话", "答", []))
    sessions = _run(store.list_sessions())
    assert sessions[0]["id"] == "b"
    assert sessions[1]["id"] == "a"

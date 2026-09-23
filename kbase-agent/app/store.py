"""会话记录存取（conversations / messages 两张表，供前端读历史）。

与 LangGraph checkpoint 的关系：
- Agent 运行状态（消息、工具调用、可恢复续聊）由 checkpoint 管，本模块不重复存；
- 这里只存"会话列表 + 可展示消息"给 GET /api/sessions 等只读接口用，
  表结构与 LangGraph 内部表（checkpoints 等）完全分开。
共用同一个 SQLite 文件（见 settings.checkpoint_db），WAL 模式文件级生效。
"""

import json
import sqlite3
from pathlib import Path

import aiosqlite

from app.config import settings

DB_PATH = str(Path(settings.checkpoint_db).resolve())

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL DEFAULT '',
    user_id    TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    sources         TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at);
CREATE TABLE IF NOT EXISTS run_traces (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT,
    question        TEXT,
    answer          TEXT,
    sources         TEXT,
    steps           TEXT,
    prompt_tokens   INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    total_tokens    INTEGER DEFAULT 0,
    cost_cny        REAL DEFAULT 0,
    duration_ms     INTEGER DEFAULT 0,
    error           TEXT,
    started_at      TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE INDEX IF NOT EXISTS idx_run_traces_started ON run_traces(started_at);
"""

# 增量迁移：已有的 checkpoints.sqlite 是 CREATE TABLE IF NOT EXISTS 建出来的，
# 加字段不会自动生效（IF NOT EXISTS 遇到已存在的表直接跳过）。所以新增列必须显式 ALTER。
# 口径：run_traces.sources 存"答案真实引用"，retrieved_sources 存"本轮检索命中"——
# 两者是不同口径（见 app/agent/graph.py 的 _parse_citations 说明），不能共用一列。
MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    ("run_traces", "retrieved_sources", "ALTER TABLE run_traces ADD COLUMN retrieved_sources TEXT"),
)

# get_trace 的显式列清单：原先用 `SELECT *` + 位置 zip，任何一次加列（就像上面这条迁移）
# 都会让取值整体错位或直接 KeyError。这里改成显式列出，加列时必须同步改这里。
TRACE_COLUMNS = (
    "id", "conversation_id", "question", "answer", "sources", "retrieved_sources",
    "steps", "prompt_tokens", "completion_tokens", "total_tokens",
    "cost_cny", "duration_ms", "error", "started_at",
)


async def _conn() -> aiosqlite.Connection:
    db = await aiosqlite.connect(DB_PATH)
    await db.execute("PRAGMA busy_timeout=5000")
    return db


async def init_db() -> None:
    """建表、跑增量迁移并确保 WAL（幂等，服务启动时调用一次）。"""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    db = await _conn()
    try:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.executescript(SCHEMA)
        for table, column, ddl in MIGRATIONS:
            cur = await db.execute(f"PRAGMA table_info({table})")
            existing = {row[1] for row in await cur.fetchall()}
            await cur.close()
            if column not in existing:
                await db.execute(ddl)
        await db.commit()
    finally:
        await db.close()


async def save_turn(
    conversation_id: str,
    title: str,
    user_content: str,
    answer: str,
    sources: list[str],
) -> None:
    """记录一轮对话：新建/更新会话 + 追加 user/assistant 两条消息。"""
    now = "datetime('now', 'localtime')"
    db = await _conn()
    try:
        await db.execute(
            f"INSERT INTO conversations (id, title, user_id, created_at, updated_at) "
            f"VALUES (?, ?, NULL, {now}, {now}) "
            f"ON CONFLICT(id) DO NOTHING",
            (conversation_id, title),
        )
        await db.execute(
            f"UPDATE conversations SET updated_at = {now} WHERE id = ?",
            (conversation_id,),
        )
        await db.execute(
            "INSERT INTO messages (conversation_id, role, content, sources) VALUES (?, 'user', ?, NULL)",
            (conversation_id, user_content),
        )
        await db.execute(
            "INSERT INTO messages (conversation_id, role, content, sources) VALUES (?, 'assistant', ?, ?)",
            (
                conversation_id,
                answer,
                json.dumps(sources, ensure_ascii=False) if sources else None,
            ),
        )
        await db.commit()
    finally:
        await db.close()


async def list_sessions(limit: int = 50) -> list[dict]:
    """按最近活跃倒序返回会话列表（含消息数/最后一条预览）。"""
    db = await _conn()
    try:
        cur = await db.execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   (SELECT COUNT(*) FROM messages m
                     WHERE m.conversation_id = c.id) AS msg_count,
                   (SELECT content FROM messages m
                     WHERE m.conversation_id = c.id
                     ORDER BY m.id DESC LIMIT 1) AS last_content
            FROM conversations c
            -- 二级键 id：updated_at 由 datetime('now','localtime') 生成、精度只到秒，
            -- 演示里"连续两次新建会话"落在同一秒很常见，只按 updated_at 排会得到不确定顺序
            ORDER BY c.updated_at DESC, c.id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cur.fetchall()
        await cur.close()
    finally:
        await db.close()
    return [
        {
            "id": r[0],
            "title": r[1],
            "created_at": r[2],
            "updated_at": r[3],
            "message_count": r[4] or 0,
            "last_preview": (r[5] or "")[:80],
        }
        for r in rows
    ]


async def get_messages(conversation_id: str) -> list[dict]:
    """返回某会话全部消息（role/content/sources/时间），用于前端渲染历史。"""
    db = await _conn()
    try:
        cur = await db.execute(
            "SELECT role, content, sources, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        )
        rows = await cur.fetchall()
        await cur.close()
    finally:
        await db.close()
    out = []
    for role, content, sources, created_at in rows:
        parsed = []
        if sources:
            try:
                parsed = json.loads(sources)
            except (json.JSONDecodeError, TypeError):
                parsed = []
        out.append({"role": role, "content": content, "sources": parsed, "created_at": created_at})
    return out


def has_schema(db_path: str) -> bool:
    """轻量同步探测（测试用）：库里是否已建 conversations 表。"""
    try:
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='conversations'"
            ).fetchone()
            return row is not None
        finally:
            conn.close()
    except sqlite3.Error:
        return False


async def save_trace(
    conversation_id: str | None,
    recorder_summary: dict,
    answer: str,
    sources: list[str],
    retrieved_sources: list[str] | None = None,
) -> None:
    """持久化一次 Agent 运行的 trace（观测用，与消息记录互不影响）。

    sources / retrieved_sources 是两个口径（答案引用 vs 检索命中），分别入列，不合并。
    """
    db = await _conn()
    try:
        await db.execute(
            """
            INSERT INTO run_traces
                (conversation_id, question, answer, sources, retrieved_sources, steps,
                 prompt_tokens, completion_tokens, total_tokens,
                 cost_cny, duration_ms, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                recorder_summary.get("question", ""),
                answer,
                json.dumps(sources, ensure_ascii=False) if sources else None,
                json.dumps(retrieved_sources, ensure_ascii=False) if retrieved_sources else None,
                json.dumps(recorder_summary.get("steps", []), ensure_ascii=False),
                recorder_summary.get("prompt_tokens", 0),
                recorder_summary.get("completion_tokens", 0),
                recorder_summary.get("total_tokens", 0),
                recorder_summary.get("cost_cny", 0.0),
                recorder_summary.get("duration_ms", 0),
                (recorder_summary.get("error") or "")[:500] or None,
            ),
        )
        await db.commit()
    finally:
        await db.close()


async def list_traces(limit: int = 50) -> list[dict]:
    """按时间倒序列出 trace（不含 steps 明细，列表页用）。"""
    db = await _conn()
    try:
        cur = await db.execute(
            """
            SELECT id, conversation_id, question, prompt_tokens, completion_tokens,
                   total_tokens, cost_cny, duration_ms, error, started_at
            FROM run_traces ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        )
        rows = await cur.fetchall()
        await cur.close()
    finally:
        await db.close()
    return [
        {
            "id": r[0],
            "conversation_id": r[1],
            "question": r[2],
            "prompt_tokens": r[3] or 0,
            "completion_tokens": r[4] or 0,
            "total_tokens": r[5] or 0,
            "cost_cny": round(r[6] or 0.0, 4),
            "duration_ms": r[7] or 0,
            "error": r[8],
            "started_at": r[9],
        }
        for r in rows
    ]


async def get_trace(trace_id: int) -> dict | None:
    """取单条 trace 完整信息（含 steps 明细，detail 页用）。"""
    db = await _conn()
    try:
        cur = await db.execute(
            f"SELECT {', '.join(TRACE_COLUMNS)} FROM run_traces WHERE id = ?", (trace_id,)
        )
        row = await cur.fetchone()
        await cur.close()
    finally:
        await db.close()
    if row is None:
        return None
    data = dict(zip(TRACE_COLUMNS, row))
    data["cost_cny"] = round(data["cost_cny"] or 0.0, 4)
    for key in ("sources", "retrieved_sources", "steps"):
        try:
            data[key] = json.loads(data[key]) if data[key] else []
        except (json.JSONDecodeError, TypeError):
            data[key] = []
    return data

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
    retrieved_sources TEXT,
    tool_calls      INTEGER,
    unverified_sources TEXT,
    -- 同一轮的两个显式排序/幂等键（见下方 INDEXES 与 save_turn 的说明）：
    --   turn_seq   本轮在请求到达时分配的全局递增序号，用来给历史消息定序
    --              （并发时"写入数据库的先后"与"用户发起的先后"可能相反）；
    --   request_id 客户端提供的幂等键，重试/重发时靠它去重。
    turn_seq        INTEGER,
    request_id      TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at);
CREATE TABLE IF NOT EXISTS run_traces (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT,
    question        TEXT,
    answer        TEXT,
    sources       TEXT,
    unverified_sources TEXT,
    tool_calls    INTEGER,
    steps         TEXT,
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
    # trace 也要能审计「查无此据」与本轮实际执行了几次工具：否则同一次运行，
    # /api/chat 与 messages 有这两个字段，/api/runs 却查不到，观测口径不一致。
    ("run_traces", "unverified_sources", "ALTER TABLE run_traces ADD COLUMN unverified_sources TEXT"),
    ("run_traces", "tool_calls", "ALTER TABLE run_traces ADD COLUMN tool_calls INTEGER"),
    # 历史消息也要带上引用可验证性字段：否则刷新页面后，"查无此据"的红色提示
    # 会消失，同一条编造引用在实时响应与历史里显示成两个样子。
    ("messages", "retrieved_sources", "ALTER TABLE messages ADD COLUMN retrieved_sources TEXT"),
    ("messages", "tool_calls", "ALTER TABLE messages ADD COLUMN tool_calls INTEGER"),
    ("messages", "unverified_sources", "ALTER TABLE messages ADD COLUMN unverified_sources TEXT"),
    ("messages", "turn_seq", "ALTER TABLE messages ADD COLUMN turn_seq INTEGER"),
    ("messages", "request_id", "ALTER TABLE messages ADD COLUMN request_id TEXT"),
)

# 依赖**新增列**的索引必须放在迁移之后建：SCHEMA 里的 CREATE INDEX 在旧库上会
# 先于 ALTER 执行，提示 "no such column: turn_seq"，服务直接起不来。
INDEXES = """
-- 同一轮的两条消息（user/assistant）共用一个 turn_seq，所以唯一键要带 role。
CREATE UNIQUE INDEX IF NOT EXISTS uq_messages_turn
    ON messages(conversation_id, turn_seq, role) WHERE turn_seq IS NOT NULL;
-- 幂等键：同一会话里同一个 request_id 只允许落一轮，重试直接冲突被忽略。
CREATE UNIQUE INDEX IF NOT EXISTS uq_messages_request
    ON messages(conversation_id, request_id) WHERE request_id IS NOT NULL;
"""

# get_trace 的显式列清单：原先用 `SELECT *` + 位置 zip，任何一次加列（就像上面这条迁移）
# 都会让取值整体错位或直接 KeyError。这里改成显式列出，加列时必须同步改这里。
TRACE_COLUMNS = (
    "id", "conversation_id", "question", "answer", "sources", "retrieved_sources",
    "unverified_sources", "tool_calls", "steps", "prompt_tokens",
    "completion_tokens", "total_tokens", "cost_cny", "duration_ms",
    "error", "started_at",
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
                try:
                    await db.execute(ddl)
                except sqlite3.OperationalError as exc:
                    # 多 worker / 多容器同时启动并命中共用旧库时，双方都可能先看到
                    # "列缺失"，后执行 ALTER 的一方会收到 duplicate column。
                    # 这说明迁移已经由另一个进程完成，视为成功而不是启动失败。
                    if "duplicate column" not in str(exc).lower():
                        raise
        await db.executescript(INDEXES)
        await db.commit()
    finally:
        await db.close()


async def save_turn(
    conversation_id: str,
    title: str,
    user_content: str,
    answer: str,
    sources: list[str],
    retrieved_sources: list[str] | None = None,
    tool_calls: int | None = None,
    unverified_sources: list[str] | None = None,
    turn_seq: int | None = None,
    request_id: str | None = None,
) -> None:
    """记录一轮对话：新建/更新会话 + 追加 user/assistant 两条消息。

    两条 INSERT 都带 `ON CONFLICT DO NOTHING`，靠 `uq_messages_turn` /
    `uq_messages_request` 两个唯一索引去重：客户端超时重试、前端重发、取消后再点一次，
    都不会在历史里堆出两轮一模一样的问答（此前 messages 表没有任何唯一约束，
    同一次提问刷新页面看到两遍，是这套系统里最容易被截图的一类 bug）。
    """
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
            "INSERT INTO messages "
            "(conversation_id, role, content, sources, turn_seq, request_id) "
            "VALUES (?, 'user', ?, NULL, ?, ?) "
            "ON CONFLICT DO NOTHING",
            (conversation_id, user_content, turn_seq, request_id),
        )
        await db.execute(
            "INSERT INTO messages "
            "(conversation_id, role, content, sources, retrieved_sources, tool_calls, "
            " unverified_sources, turn_seq, request_id) "
            "VALUES (?, 'assistant', ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT DO NOTHING",
            (
                conversation_id,
                answer,
                json.dumps(sources, ensure_ascii=False) if sources else None,
                json.dumps(retrieved_sources, ensure_ascii=False)
                if retrieved_sources is not None else None,
                tool_calls,
                json.dumps(unverified_sources, ensure_ascii=False)
                if unverified_sources is not None else None,
                turn_seq,
                request_id,
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
    """返回某会话全部消息（role/content/sources/时间），用于前端渲染历史。

    排序先看 `turn_seq`（请求到达时分配的序号），再看 `id`：
    同一会话的两个请求是串行跑图、**并发写库**的，后发起的那轮完全可能先 INSERT，
    那时只按自增 id 排，历史里第 2 问会显示在第 1 问上面。
    旧库没有 turn_seq 的消息值为 NULL，`ORDER BY turn_seq IS NULL, turn_seq, id`
    把 NULL 排在最后、组内退回按 id，行为与改造前一致。
    """
    db = await _conn()
    try:
        cur = await db.execute(
            "SELECT role, content, sources, retrieved_sources, tool_calls, "
            "unverified_sources, created_at FROM messages "
            "WHERE conversation_id = ? "
            "ORDER BY turn_seq IS NULL, turn_seq ASC, id ASC",
            (conversation_id,),
        )
        rows = await cur.fetchall()
        await cur.close()
    finally:
        await db.close()

    def _list(raw) -> list[str]:
        if not raw:
            return []
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
        return value if isinstance(value, list) else []

    out = []
    for role, content, sources, retrieved, tool_calls, unverified, created_at in rows:
        out.append(
            {
                "role": role,
                "content": content,
                "sources": _list(sources),
                "retrieved_sources": _list(retrieved),
                # 旧消息没有该列（NULL）→ 前端按 undefined 处理，不做"本轮未检索"判断
                "tool_calls": tool_calls,
                "unverified_sources": _list(unverified),
                "created_at": created_at,
            }
        )
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
    unverified_sources: list[str] | None = None,
    tool_calls: int | None = None,
) -> None:
    """持久化一次 Agent 运行的 trace（观测用，与消息记录互不影响）。

    sources / retrieved_sources / unverified_sources 是三个口径
    （答案引用 / 检索命中 / 查无此据），分别入列，不合并。
    """
    db = await _conn()
    try:
        await db.execute(
            """
            INSERT INTO run_traces
                (conversation_id, question, answer, sources, retrieved_sources,
                 unverified_sources, tool_calls, steps,
                 prompt_tokens, completion_tokens, total_tokens,
                 cost_cny, duration_ms, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                recorder_summary.get("question", ""),
                answer,
                json.dumps(sources, ensure_ascii=False) if sources else None,
                # 新数据即使为空也写 "[]"：SQL NULL 专门留给"字段尚未采集"的旧数据，
                # 这样 API 消费方可以按 null / [] 区分两代口径（README 的判据）。
                json.dumps(retrieved_sources, ensure_ascii=False)
                if retrieved_sources is not None else None,
                json.dumps(unverified_sources, ensure_ascii=False)
                if unverified_sources is not None else None,
                tool_calls,
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
    # strict=True：列清单与 SELECT 结果必须等长。静默截断/错位会让 trace 字段
    # 整体偏移，而且查出来"有值"，比直接报错更难发现。
    data = dict(zip(TRACE_COLUMNS, row, strict=True))
    data["cost_cny"] = round(data["cost_cny"] or 0.0, 4)
    for key in ("sources", "steps"):
        try:
            data[key] = json.loads(data[key]) if data[key] else []
        except (json.JSONDecodeError, TypeError):
            data[key] = []
    # retrieved_sources / unverified_sources 的 SQL NULL 有专门语义：旧数据没有采集
    # 这个字段。全部抹成 [] 会让 README 承诺的"按 null 区分旧数据"失效。
    for key in ("retrieved_sources", "unverified_sources"):
        if data[key] is None:
            continue
        try:
            data[key] = json.loads(data[key])
        except (json.JSONDecodeError, TypeError):
            data[key] = []
    return data

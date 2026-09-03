"""LangGraph Agent：把 app/mcp/servers.py 的 FastMCP 工具经 langchain-mcp-adapters 真接入。

图结构：agent(调模型决策) -> tools(执行工具/护栏) -> agent ... -> 直接作答。
护栏：recursion_limit + 单步超时 + 工具输出截断 + 重复调用检测（均在 tools 节点）。
状态持久化：AsyncSqliteSaver（SQLite WAL，checkpoint 落盘，服务重启可续聊）；
生产并发更高时可换 PostgresSaver，表结构同 LangGraph checkpoint 约定。
"""

import asyncio
import json
import sys
import uuid
from pathlib import Path

import aiosqlite
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from app.agent.state import AgentState
from app.config import settings
from app.guardrails import (
    AgentLimits,
    default_recursion_limit,
    is_duplicate_call,
    truncate_tool_output,
)

ROOT = Path(__file__).resolve().parents[2]

SYSTEM_PROMPT = (
    "你是企业内部知识助手 Agent。回答用户问题时遵守：\n"
    "1. 涉及制度/规则（员工手册、产品 FAQ）先用 retrieve_knowledge；\n"
    "2. 问个人状态（年假剩余、报销进度、加班调休等）先 retrieve_knowledge 拿规则，"
    "再 query_business_db 拿个人数据，两者结合再作答；\n"
    "3. 知识库没有的内容如实说『资料中没有』，绝不编造；回答末尾用【来源：文件名】列出引用；\n"
    "4. 同一工具、同一参数不要重复调用；若重复说明你在原地打转，请基于已有信息作答；\n"
    f"5. 每个问题最多执行 {AgentLimits.max_steps} 步工具调用，超限立即用当前信息收尾。"
)


def _route(state: AgentState) -> str:
    last = state["messages"][-1]
    has_tool_calls = bool(getattr(last, "tool_calls", None))
    return "tools" if has_tool_calls else END


def _text_of(content) -> str:
    """把 LangChain 消息内容规整成纯文本。

    MCP 工具经 langchain-mcp-adapters 返回的是 content block 列表
    （[{'type': 'text', 'text': ...}]），直接 str() 会得到 Python repr，
    导致【来源：】标记匹配不到、模型上下文也被 repr 污染。
    """
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "\n".join(part for part in parts if part)
    return "" if content is None else str(content)


def _parse_sources(messages: list) -> list[str]:
    sources: list[str] = []
    for message in messages:
        if getattr(message, "type", "") != "tool":
            continue
        for line in _text_of(message.content).splitlines():
            line = line.strip()
            if line.startswith("【来源：") and "】" in line:
                source = line.split("】", 1)[0][len("【来源：") :]
                if source and source not in sources:
                    sources.append(source)
    return sources


def _final_answer(messages: list) -> str:
    for message in reversed(messages):
        if (
            getattr(message, "type", "") == "ai"
            and not getattr(message, "tool_calls", None)
            and getattr(message, "content", "")
        ):
            return str(message.content)
    return ""


def _build_graph(llm, tools: list, checkpointer):
    tools_by_name = {tool.name: tool for tool in tools}

    async def agent_node(state: AgentState) -> dict:
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response]}

    async def tools_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None) or []
        new_messages: list = []
        history = list(state.get("tool_call_history", []))

        for call in calls:
            name = call.get("name", "")
            args = call.get("args") or {}
            key = (name, json.dumps(args, ensure_ascii=False, sort_keys=True))
            tool = tools_by_name.get(name)

            if tool is None:
                content = f"未找到工具：{name}"
            elif is_duplicate_call(history, key):
                content = (
                    f"检测到重复工具调用（{name} {args}，历史中已执行过），"
                    "为避免死循环本次不再执行。请基于已有信息作答，或换个问法。"
                )
            else:
                try:
                    raw = await asyncio.wait_for(
                        tool.ainvoke(args), timeout=AgentLimits.step_timeout_seconds
                    )
                    content = truncate_tool_output(
                        _text_of(raw), AgentLimits.max_tool_output_chars
                    )
                except asyncio.TimeoutError:
                    content = f"工具 {name} 调用超时（>{AgentLimits.step_timeout_seconds}s）"
                except Exception as exc:  # noqa: BLE001
                    content = f"工具 {name} 调用失败：{type(exc).__name__}: {exc}"

            new_messages.append(
                ToolMessage(content=content, tool_call_id=call.get("id", ""), name=name)
            )
            history.append(key)

        return {"messages": new_messages, "tool_call_history": history}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        _route,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)


class AgentRuntime:
    """持有编译好的图与 MCP 工具。会话标识 thread_id，崩溃后同 session 可续聊。"""

    def __init__(self, graph, tools: list, saver: AsyncSqliteSaver | None = None, conn=None):
        self.graph = graph
        self.tools = tools
        self._saver = saver
        self._conn = conn

    def _config(self, session_id: str | None) -> dict:
        return {
            "configurable": {"thread_id": session_id or uuid.uuid4().hex},
            # 默认按 AgentLimits.max_steps 推导，避免与 prompt 承诺的步数不一致
            "recursion_limit": settings.max_recursion
            or default_recursion_limit(),
        }

    @staticmethod
    def _to_lc_messages(messages: list[dict]) -> list:
        converted: list = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role in {"user", "human"}:
                converted.append(HumanMessage(content=content))
            elif role in {"assistant", "ai"}:
                converted.append(AIMessage(content=content))
        return converted

    async def _build_input(
        self, messages: list[dict], session_id: str | None
    ) -> tuple[list, dict]:
        """构造本轮图输入：SystemMessage 只在 thread 为空时注入一次。

        带 checkpointer 时同一 thread 的历史消息由 LangGraph 自动拼接，
        若每轮都重新注入 SystemMessage 会导致系统提示词重复且顺序错乱。
        注意 API 契约：同一 session 每次只追加最新一条用户消息。
        """
        config = self._config(session_id)
        snapshot = await self.graph.aget_state(config)
        existing = bool((snapshot.values or {}).get("messages"))
        lc_messages = self._to_lc_messages(messages)
        if not existing:
            lc_messages = [SystemMessage(content=SYSTEM_PROMPT)] + lc_messages
        return lc_messages, config

    async def ainvoke(self, messages: list[dict], session_id: str | None = None) -> dict:
        lc_messages, config = await self._build_input(messages, session_id)
        state = await self.graph.ainvoke({"messages": lc_messages}, config=config)
        final_messages = state["messages"]
        return {
            "answer": _final_answer(final_messages),
            "sources": _parse_sources(final_messages),
        }

    async def astream(self, messages: list[dict], session_id: str | None = None):
        """按 super-step 产出 (node, state_update) 增量，供 SSE 逐步下发。"""
        lc_messages, config = await self._build_input(messages, session_id)
        async for update in self.graph.astream(
            {"messages": lc_messages}, config=config, stream_mode="updates"
        ):
            yield update
        # 图跑完后从 checkpoint 取最终状态，避免再跑一次
        snapshot = await self.graph.aget_state(config)
        values = snapshot.values or {}
        final_messages = values.get("messages", [])
        yield {
            "answer": _final_answer(final_messages),
            "sources": _parse_sources(final_messages),
        }

    async def aclose(self) -> None:
        # adapters 每个工具调用自己开/关 stdio 会话，无需常驻清理；
        # 这里关掉 SQLite checkpoint 连接
        if self._conn is not None:
            try:
                await self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None


async def _open_checkpointer() -> tuple[AsyncSqliteSaver, aiosqlite.Connection]:
    """打开（必要时创建）SQLite checkpoint 库并启用 WAL。

    WAL：读写不互锁，配合 uvicorn 单写者足够；路径见 CHECKPOINT_DB（默认 ./data/checkpoints.sqlite）。
    """
    db_path = Path(settings.checkpoint_db).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(db_path))
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA busy_timeout=5000")
    saver = AsyncSqliteSaver(conn)
    await saver.setup()
    return saver, conn


async def create_runtime() -> AgentRuntime:
    """声明 MCP 服务器并枚举工具（adapters 会为每个工具调用自开 stdio 会话），再编译图。"""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    from app.llm import make_llm

    saver, conn = await _open_checkpointer()
    client = MultiServerMCPClient(
        {
            "kbase": {
                "command": sys.executable,
                "args": ["-m", "app.mcp.servers"],
                "cwd": str(ROOT),
                "transport": "stdio",
            }
        }
    )
    try:
        tools = await client.get_tools()
    except Exception:
        await conn.close()
        raise
    llm = make_llm(temperature=settings.temperature).bind_tools(tools)
    graph = _build_graph(llm, tools, checkpointer=saver)
    return AgentRuntime(graph=graph, tools=tools, saver=saver, conn=conn)


async def run_single_question(question: str, session_id: str | None = None) -> dict:
    """CLI/脚本用：一次 asyncio.run 内建运行时并答一个问题（演示/自测）。"""
    runtime = await create_runtime()
    try:
        return await runtime.ainvoke([{"role": "user", "content": question}], session_id)
    finally:
        await runtime.aclose()

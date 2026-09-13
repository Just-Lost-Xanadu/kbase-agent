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
from typing import Annotated, TypedDict

import aiosqlite
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.config import settings
from app.guardrails import (
    AgentLimits,
    default_recursion_limit,
    is_duplicate_call,
    truncate_tool_output,
)

ROOT = Path(__file__).resolve().parents[2]

# ---- 图状态 AgentState（原 app/agent/state.py，并入本文件：状态定义就近其消费方）----


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    # 引用来源不放进图状态：最终答案的 sources 由 ainvoke/astream 收尾时
    # 从消息流解析（见 _parse_sources），避免图里维护冗余字段。
    # 工具调用去重历史：只保留最近 max_steps 次（窗口在 tools_node 维护）。
    # 实际生效范围是"单次运行内"：LangGraph 序列化会把 tuple 还原成 list，
    # 跨轮次读回的条目与本次的 tuple key 永不相等 —— 这正是期望行为：
    # 同会话重复提问应当重新检索拿新数据，而不是复用上一轮的旧上下文。
    tool_call_history: list[tuple[str, str]]


# ---- LLM 工厂（原 app/llm.py，并入本文件：唯一消费者就是本模块的 create_runtime）----


def make_llm(temperature: float = 0.0) -> ChatOpenAI:
    """构造 OpenAI 兼容的 LLM 客户端（默认指向 DeepSeek）。

    未配置 key（缺失或仍是占位符 sk-your-key）时抛明确中文报错，避免带占位 key 悄悄调失败。
    .env 由 app.config 的 load_dotenv() 定位（从 app/config.py 所在目录向上查找，与 CWD 无关）；
    但 CHROMA_PATH / CHECKPOINT_DB 是相对路径，因此仍建议从项目根（kbase-agent/）启动。
    """
    if not settings.api_key or settings.api_key == "sk-your-key":
        raise ValueError(
            "未配置有效的 DEEPSEEK_API_KEY。请在项目根目录创建 .env"
            "（复制 .env.example 并填入真实 key，占位符 sk-your-key 不会被接受），"
            "或先执行：$env:DEEPSEEK_API_KEY='sk-真实key'。"
            "注意 CHROMA_PATH / CHECKPOINT_DB 是相对路径，请从 kbase-agent 目录启动。"
        )
    return ChatOpenAI(
        model=settings.model_name,
        base_url=settings.base_url,
        api_key=settings.api_key,
        temperature=temperature,
    )

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
            # 用 _text_of 收口：content 若是 content block 列表，str() 会得到 Python repr
            return _text_of(message.content)
    return ""


def _usage_tokens(message) -> tuple[int, int]:
    """从 LLM 返回消息里取 token 用量（兼容 usage_metadata / token_usage）。"""
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict):
        return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
    meta = getattr(message, "response_metadata", None) or {}
    token_usage = (meta.get("token_usage") or {}) if isinstance(meta, dict) else {}
    return int(token_usage.get("prompt_tokens") or 0), int(token_usage.get("completion_tokens") or 0)


def _build_graph(llm, tools: list, checkpointer):
    tools_by_name = {tool.name: tool for tool in tools}

    async def agent_node(state: AgentState) -> dict:
        from app.observability import Recorder, Step, get_recorder, now_ms, estimate_cost

        rec: Recorder | None = get_recorder()
        t0 = now_ms()
        try:
            response = await llm.ainvoke(state["messages"])
            ok, note = True, ""
        except Exception as exc:  # noqa: BLE001
            response = None
            ok, note = False, f"{type(exc).__name__}: {exc}"
        if rec is not None:
            if response is not None:
                p, c = _usage_tokens(response)
            else:
                p = c = 0
            rec.add(
                Step(
                    node="agent", name="llm", duration_ms=now_ms() - t0,
                    prompt_tokens=p, completion_tokens=c,
                    cost_cny=estimate_cost(p, c), ok=ok, note=note[:200],
                )
            )
            if not ok:
                rec.error = note
        if response is None:
            raise RuntimeError(f"LLM 调用失败: {note}")
        return {"messages": [response]}

    async def tools_node(state: AgentState) -> dict:
        from app.observability import Recorder, Step, get_recorder, now_ms

        rec: Recorder | None = get_recorder()
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None) or []
        new_messages: list = []
        # 判重窗口 = 最近 max_steps 次工具调用，实际只在"本次运行内"能命中：
        # history 存进 checkpoint 后 tuple 会被序列化成 list，跨轮次读回时
        # tuple key 与 list 条目不相等，因此不会误拦跨轮次的合法重复查询
        # （跨轮次重复提问本就应当重新检索，见 guardrails.is_duplicate_call）。
        history = list(state.get("tool_call_history", []))[-AgentLimits.max_steps :]

        for call in calls:
            name = call.get("name", "")
            args = call.get("args") or {}
            key = (name, json.dumps(args, ensure_ascii=False, sort_keys=True))
            tool = tools_by_name.get(name)
            t0 = now_ms()

            if tool is None:
                content = f"未找到工具：{name}"
            elif is_duplicate_call(history, key):
                content = (
                    f"检测到重复工具调用（{name} {args}，本轮已执行过），"
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

            if rec is not None:
                note = ""
                ok = True
                if tool is None:
                    ok = False
                    note = "工具不存在"
                elif content.startswith("检测到重复工具调用"):
                    note = "重复调用，已跳过"
                elif "超时" in content or "调用失败" in content:
                    ok = False
                    note = content[:160]
                rec.add(
                    Step(node="tools", name=name, duration_ms=now_ms() - t0,
                         ok=ok, note=note)
                )

        return {
            "messages": new_messages,
            "tool_call_history": history[-AgentLimits.max_steps :],
        }

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
    ) -> tuple[list, dict, int]:
        """构造本轮图输入：SystemMessage 只在 thread 为空时注入一次。

        带 checkpointer 时同一 thread 的历史消息由 LangGraph 自动拼接，
        若每轮都重新注入 SystemMessage 会导致系统提示词重复且顺序错乱。
        注意 API 契约：同一 session 每次只追加最新一条用户消息。

        返回 (lc_messages, config, prior_count)：
        prior_count = 本轮运行前 checkpoint 里已有的消息条数 —— 收尾时用它做偏移，
        使 sources 只统计"本轮新增消息"里的工具来源，避免把历史轮次的来源累计进来
        （多轮后 state["messages"] 是完整历史，若不偏移，每轮答案都会挂上前几轮检索过的所有来源）。
        """
        config = self._config(session_id)
        snapshot = await self.graph.aget_state(config)
        prior_messages = (snapshot.values or {}).get("messages") or []
        existing = bool(prior_messages)
        lc_messages = self._to_lc_messages(messages)
        if not existing:
            lc_messages = [SystemMessage(content=SYSTEM_PROMPT)] + lc_messages
        return lc_messages, config, len(prior_messages)

    async def ainvoke(self, messages: list[dict], session_id: str | None = None) -> dict:
        lc_messages, config, prior = await self._build_input(messages, session_id)
        state = await self.graph.ainvoke({"messages": lc_messages}, config=config)
        final_messages = state["messages"]
        # 只从本轮新增的消息里解析来源（历史轮次来源不累计）
        fresh = final_messages[prior:]
        return {
            "answer": _final_answer(final_messages),
            "sources": _parse_sources(fresh),
        }

    async def astream(self, messages: list[dict], session_id: str | None = None):
        """按 super-step 产出 (node, state_update) 增量，供 SSE 逐步下发。"""
        lc_messages, config, prior = await self._build_input(messages, session_id)
        async for update in self.graph.astream(
            {"messages": lc_messages}, config=config, stream_mode="updates"
        ):
            yield update
        # 图跑完后从 checkpoint 取最终状态，避免再跑一次
        snapshot = await self.graph.aget_state(config)
        values = snapshot.values or {}
        final_messages = values.get("messages", [])
        fresh = final_messages[prior:]
        yield {
            "answer": _final_answer(final_messages),
            "sources": _parse_sources(fresh),
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
    try:
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA busy_timeout=5000")
        saver = AsyncSqliteSaver(conn)
        await saver.setup()
    except BaseException:
        # 建表/PRAGMA 失败、或握手期间被取消（CancelledError 属 BaseException）：
        # 都要先关掉刚开的连接再抛出，否则这个连接会一直挂着
        await conn.close()
        raise
    return saver, conn


async def create_runtime() -> AgentRuntime:
    """工厂：声明 MCP 服务器与工具、建图，返回一个封装好生命周期(resource)的 AgentRuntime。

    OOP/设计要点：真正"需要对象封装"的是这里的**运行时生命周期**——checkpointer 的
    AsyncSqliteSaver、sqlite 连接 conn、graph、tools 都需要在请求结束后正确关闭
    (aclose/close)，不能靠模块级散落。所以返回 AgentRuntime(封装 graph+tools+saver+conn)
    并约定由调用方负责用毕 aclose()：这是"用对象管理成对分配/释放资源(open/close)"的典型场景，
    普通函数无法靠返回值表达"记得关连接"的约束。
    （内部 get_tools / make_llm / 建图任一步抛错，都会先关掉已开 sqlite 连接再 raise，
    避免泄漏 —— 见下方 try/except。）
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    saver, conn = await _open_checkpointer()
    client = MultiServerMCPClient(
        {
            # 一个 MCP server("kbase"，由 app.mcp.servers 承载)里放两把工具：
            # retrieve_knowledge(查知识库) 与 query_business_db(查业务数据)。
            # 都是真 MCP stdio 子进程接入，LangGraph bind_tools 后由模型按需调用。
            "kbase": {
                "command": sys.executable,
                "args": ["-m", "app.mcp.servers"],
                "cwd": str(ROOT),
                "transport": "stdio",
            }
        }
    )
    try:
        # 先校验 key（纯本地、零成本）再拉起 MCP 子进程：否则缺 key 时每次重试
        # 都要白付一次子进程启动 + 索引加载，才在 make_llm 处报错。
        llm = make_llm(temperature=settings.temperature)
        tools = await client.get_tools()
        llm = llm.bind_tools(tools)
        graph = _build_graph(llm, tools, checkpointer=saver)
    except BaseException:
        # 任一步失败（含 CancelledError 这类 BaseException）都要先关掉已开的 sqlite
        # 连接：services.ensure_services 允许每次请求重试，不关就每请求泄漏一个连接+线程。
        await conn.close()
        raise
    return AgentRuntime(graph=graph, tools=tools, saver=saver, conn=conn)


async def run_single_question(question: str, session_id: str | None = None) -> dict:
    """CLI/脚本用：一次 asyncio.run 内建运行时并答一个问题（演示/自测）。"""
    runtime = await create_runtime()
    try:
        return await runtime.ainvoke([{"role": "user", "content": question}], session_id)
    finally:
        await runtime.aclose()

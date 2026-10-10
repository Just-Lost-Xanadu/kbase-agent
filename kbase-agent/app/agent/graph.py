"""LangGraph Agent：把 app/mcp/servers.py 的 FastMCP 工具经 langchain-mcp-adapters 真接入。

图结构：agent(调模型决策) -> tools(执行工具/护栏) -> agent ... -> 直接作答。
护栏：recursion_limit + 单步超时 + 工具输出截断 + 重复调用检测（均在 tools 节点）。
状态持久化：AsyncSqliteSaver（SQLite WAL，checkpoint 落盘，服务重启可续聊）；
生产并发更高时可换 PostgresSaver，表结构同 LangGraph checkpoint 约定。
"""

import asyncio
import contextlib
import json
import re
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

# ---- MCP 工具会话：一个常驻 stdio 子进程，而不是"每次调用新起一个" ----
#
# 背景（本模块最重要的一次性能改动，实测数字见 README）：
# `MultiServerMCPClient.get_tools()` 返回的 LangChain 工具是**无会话**的——它的
# `ainvoke` 内部会 `create_session(...)` 新起一个 stdio 子进程、握手、调用、再关掉。
# 于是每次工具调用都要重新付一遍：解释器启动 + `import chromadb/onnxruntime` +
# 子进程内 BM25/Chroma 冷加载。实测 `retrieve_knowledge` 单次 3.6~3.8s（三次 3782/3643/3676ms），
# 其中子进程启动约 3.0s、子进程内冷加载约 0.68s。一次回答调 1~2 把工具，延迟大头就在这里。
#
# `client.session(server)` 提供的是**常驻会话**，用 `load_mcp_tools(session)` 绑定上去之后，
# 同一进程内后续每次调用只需一次 stdio 往返。实测同一台机器：首次 683ms（子进程内冷加载），
# 之后 **11~12ms**，两个工具并发 **17ms**。
#
# 这里额外做两件事，都不是"新功能"，而是常驻会话自带的失败面必须补上：
#   1) 失败自愈：子进程被 OOM/手工杀掉后，会话即失效。若只把死会话一直握着，
#      服务会**永久**对每个请求回"工具调用失败"，只能重启进程——这与项目
#      "索引坏了自动重建"的既有口径不一致。所以这里在传输层失败时把会话标记为失效，
#      下一次派单前重建（重建时整个工具表一起换掉）。
#   2) 只对**传输层**失败重建：工具业务错误（检索抛异常、业务库没这个人）绝不能
#      触发重建——那会把一次正常报错放大成一次子进程重启。

# 传输层失败：子进程死掉/管道关闭时 MCP SDK 抛的那几类。故意不含 asyncio.TimeoutError——
# 工具超时只说明这一次调用慢，会话本身通常还活着（`retrieve_knowledge` 冷加载实测 683ms，
# 而单步超时是 120s），因超时重建会把"慢"升级成"重启"。
try:  # pragma: no cover - anyio 是 mcp 的传递依赖，正常一定在
    import anyio

    _TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
        anyio.ClosedResourceError,
        anyio.BrokenResourceError,
        anyio.EndOfStream,
    )
except ImportError:  # pragma: no cover
    _TRANSPORT_ERRORS = ()


class ToolSessionStale(RuntimeError):
    """本次工具调用命中的是**已失效**会话：会话已被别的调用判死，需要重新派单。

    调用方（tools_node）据此重取一次工具表再重试，而不是把这次失败算进 trace。
    """


class ToolSessionUnavailable(RuntimeError):
    """会话重建也失败了（子进程起不来、索引/依赖坏了）：如实报错，不抛给上层 502。"""


def _is_transport_failure(exc: BaseException) -> bool:
    """判断异常是否属于"会话/子进程死了"，而不是"工具业务逻辑报错"。"""
    if _TRANSPORT_ERRORS and isinstance(exc, _TRANSPORT_ERRORS):
        return True
    # McpError 只在请求已发出但连接断了时出现（SDK 在 send_request 里对已关闭的
    # 传输抛 ClosedResourceError，这里再兜一层字符串判断，避免因 SDK 版本换异常类型而漏判）
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in ("closedresource", "brokenresource", "endofstream", "connection closed", "session is closed")
    )


class _Lease:
    """一次会话占用的凭据。`generation` 用来判断"我拿到的会话是否已经被判死"。"""

    __slots__ = ("by_name", "generation", "session", "tools")

    def __init__(self, tools: list, generation: int, session):
        self.tools = tools
        self.by_name = {tool.name: tool for tool in tools}
        self.generation = generation
        self.session = session


class McpToolSession:
    """常驻 MCP stdio 会话 + 工具表，带失效自愈。

    生命周期由 AgentRuntime 持有（`aclose()` 时释放）；`tools` 属性是当前有效工具表的快照，
    调用方每次派单前用 `lease()` 现取一份——这样"工具表被整体换掉"不会让图里留着旧对象。
    """

    def __init__(self, connections: dict, session_factory=None):
        self._connections = connections
        # 会话工厂：默认走真 MCP stdio（`_spawn`）。抽成可注入的工厂是为了让单测能在
        # 不拉起子进程的前提下验证"派单/并发/判重/失效重建"这套逻辑——测的是本类的状态机，
        # 不是 MCP 传输本身（传输由集成冒烟覆盖）。
        self._session_factory = session_factory
        self._client = None
        self._stack: contextlib.AsyncExitStack | None = None
        self._session = None
        self._tools: list = []
        self._generation = 0
        self._lock = asyncio.Lock()

    @property
    def tools(self) -> list:
        return self._tools

    def lease(self, generation: int | None) -> "_Lease | None":
        """按调用方持有的 generation 取一份工具表。

        - `generation is None`：调用方首次派单，拿当前的（没有就返回 None，由 ensure 去建）。
        - generation 与当前一致：正常复用。
        - generation 落后：说明会话在本次派单期间被判死并重建过（或正在重建）——
          返回 None，让调用方 `ensure()` 之后重新取，避免继续用已被关闭的会话。
        """
        if not self._tools:
            return None
        if generation is None or generation == self._generation:
            return _Lease(self._tools, self._generation, self._session)
        return None

    async def ensure(self) -> "_Lease":
        """确保有一个可用会话，返回它的租约。已有效时是零开销（不加锁、不往返）。"""
        if self._tools:
            return _Lease(self._tools, self._generation, self._session)
        async with self._lock:
            if self._tools:  # 等锁期间别人已经建好了
                return _Lease(self._tools, self._generation, self._session)
            return await self._spawn()

    async def invalidate(self, generation: int) -> None:
        """把 generation 号会话判死并释放；下一代按需重建。

        只处理"还活着的那一代"：并发的多个失败调用会一起走到这里，重复调用必须是幂等的，
        否则第二个调用会把别人刚建好的新会话关掉。
        """
        async with self._lock:
            if generation != self._generation:
                return
            self._generation += 1   # 让所有还持有旧租约的调用方在下次 lease() 时拿到 None
            self._tools = []
            self._session = None
            stack, self._stack = self._stack, None
        if stack is not None:
            with contextlib.suppress(Exception):
                await stack.aclose()

    async def _spawn(self) -> "_Lease":
        if self._session_factory is not None:
            # 测试路径：工厂返回 (tools, session)，不涉及子进程。
            # 这里同样要把失败归一成 ToolSessionUnavailable：否则 tools_node 只认这个异常，
            # 别处抛出来的原始异常会漏到 HTTP 层变成 502（而期望行为是"降级作答"）。
            try:
                tools, session = await self._session_factory()
            except Exception as exc:
                raise ToolSessionUnavailable(
                    f"MCP 工具会话建立失败：{type(exc).__name__}: {exc}"
                ) from exc
            self._session = session
            self._tools = tools
            return _Lease(tools, self._generation, session)

        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools

        try:
            client = self._client or MultiServerMCPClient(self._connections)
            self._client = client
            stack = contextlib.AsyncExitStack()
            await stack.__aenter__()
            try:
                # 只连第一个（也是唯一一个）server：本项目的连接表刻意只有一个 "kbase"
                server_name = next(iter(self._connections))
                session = await stack.enter_async_context(client.session(server_name))
                tools = await load_mcp_tools(session, server_name=server_name)
            except BaseException:
                # 建立失败要把已经开的栈收干净（子进程、stdio 管道），否则每失败一次泄漏一个进程
                with contextlib.suppress(Exception):
                    await stack.aclose()
                raise
        except Exception as exc:
            raise ToolSessionUnavailable(
                f"MCP 工具会话建立失败：{type(exc).__name__}: {exc}"
            ) from exc

        self._stack = stack
        self._session = session
        self._tools = tools
        return _Lease(tools, self._generation, session)

    async def aclose(self) -> None:
        """释放常驻会话（服务关停时调用）。"""
        async with self._lock:
            self._generation += 1
            self._tools = []
            self._session = None
            stack, self._stack = self._stack, None
        if stack is not None:
            with contextlib.suppress(Exception):
                await stack.aclose()


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
    "5. 问个人状态但用户没说是哪位员工时：仍要先 retrieve_knowledge 取出相关制度规则并据此说明规则，"
    "再请用户补充姓名以查询个人数据；不要猜测或沿用他人数据；\n"
    f"6. 每个问题最多执行 {AgentLimits.max_steps} 步工具调用，超限立即用当前信息收尾；\n"
    "7. 回答使用纯文本排版：可用短横线分条，但不要输出 Markdown 标记（如 # 标题、** 加粗）。"
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
    """**检索口径**：本轮工具返回过的来源（top_k 命中里带【来源：文件名】标记的那些）。

    注意这不是"答案引用了什么"——不要拿它当引用准确性用（见 _parse_citations 的说明）。
    """
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


_CITATION_RE = re.compile(r"【来源：([^】]+)】")
# 同一个【来源：…】括号内的多个文件名分隔符（模型会写"A、B、C"，也可能用逗号/分号/斜杠）
# 与 app/retrieval/loader.SUPPORTED_EXTS 一致（有单测断言两者相等，防漂移）
_SOURCE_EXTS: tuple[str, ...] = (".md", ".txt", ".docx", ".xlsx", ".pdf")
_CITATION_SPLIT = re.compile(r"[、,，;；/]+")


def _looks_like_source_name(text: str) -> bool:
    """这段文字看起来像"语料里的文件名"吗（以已知扩展名结尾）。

    判据只用扩展名，不必读语料目录：本项目的 source 一律是 `data/docs` 下的真实文件名
    （见 app/retrieval/loader.py 的 SUPPORTED_EXTS），都以这五种扩展名结尾。
    """
    return text.strip().lower().endswith(_SOURCE_EXTS)


def _split_citation_names(raw: str) -> list[str]:
    """把【来源：…】括号里的内容拆成若干个来源名（**保真优先**）。

    两种真实写法都要支持：
      - 模型把多篇写在一个括号里：`【来源：A.md、B.md、C.md】` → 拆成三条；
      - 模型在括号里写一句**描述**：实测那条假来源
        `【来源：业务系统个人数据查询结果（知识库检索失败,无制度文件可引用）】`
        —— 按逗号硬拆会把一句人话切成语义不完整的两段（旧实现就把这条存成了
        `["业务系统个人数据查询结果（知识库检索失败", "无制度文件可引用）"]`）。

    规则：**只在"相邻两段里至少有一段像文件名"处分组**。于是连续的非文件名片段
    （也就是那句描述）会自然粘回一起，而 `A.md、B.md` 照旧拆开；合法文件名与描述混在
    一个括号里时也不会被粘住。README 承诺的"答案写了什么要如实保留"在两种写法下都成立。
    """
    separators = list(_CITATION_SPLIT.finditer(raw))
    segments: list[str] = []
    pos = 0
    for match in separators:
        segments.append(raw[pos:match.start()])
        pos = match.end()
    segments.append(raw[pos:])
    if len(segments) == 1:
        return [segments[0].strip()] if segments[0].strip() else []

    looks_like_file = [_looks_like_source_name(segment) for segment in segments]
    groups: list[str] = []
    start = 0
    for index in range(1, len(segments)):
        if looks_like_file[index - 1] or looks_like_file[index]:
            groups.append(raw[start:separators[index - 1].start()])
            start = separators[index - 1].end()
    groups.append(raw[start:])
    return [group.strip() for group in groups if group.strip()]


def _parse_citations(answer: str) -> list[str]:
    """**引用口径**：答案正文里真实出现的【来源：文件名】，按出现顺序去重。

    与 _parse_sources 的区别（两者口径不同，不要混）：
      - `_parse_sources` 看的是**工具返回过什么**（top_k 命中，每篇都被标注出来）；
      - `_parse_citations` 看的是**答案正文里到底引用了什么**。
    实测差值不是边角情况而是常态：复核 /api/runs 里 9 条带 sources 的 trace，
    **9/9 都比答案正文引用多 1 条**（例：sources=['员工手册_示例.md','入职转正与离职制度_示例.md']，
    而正文只标了【来源：员工手册_示例.md】）。前端把工具口径直接渲染成"来源："标签，
    等于在 100% 的轮次里替答案多声明了一篇引用——对一个主打"引用溯源"的项目，
    这是会被一眼看穿的过度声明。

    因此 API 的 `sources` 字段改为本函数的结果（与用户直觉一致：答案引用了哪些来源），
    工具口径改名 `retrieved_sources` 一并返回，信息不丢、语义不再混淆。

    匹配用整段扫描而不是"只看行首"：模型既可能把引用单独列在末尾（prompt 的推荐做法），
    也可能写在句子中间（如"按【来源：员工手册_示例.md】的规定"），两种都是真实引用。
    """
    citations: list[str] = []
    for match in _CITATION_RE.finditer(answer or ""):
        # 一个括号里塞多篇（`【来源：A、B、C】`）要拆开：prompt 只要求"用【来源：文件名】列出引用"，
        # 没规定一篇一个括号，模型两种写法都会出现。不拆则会返回单元素复合串
        # （实测库里就有 `'薪酬与绩效制度_示例.md、绩效系数对照表.xlsx、员工手册_示例.md'`），
        # 前端渲染成一个标签、按 len(sources) 统计的消费方也会数错。
        for source in _split_citation_names(match.group(1)):
            source = source.strip()
            if source and source not in citations:
                citations.append(source)
    return citations


def _split_verified_citations(
    citations: list[str], retrieved: list[str]
) -> tuple[list[str], list[str]]:
    """把答案里标出来的引用拆成"本轮检索真的返回过"与"查无此据"两组。

    为什么必须有这一步（实测到的真实缺陷，不是假想）：
    `_parse_citations` 只是**照抄模型写的字**。模型完全可以在正文里写出一个知识库里
    根本不存在的来源。实测一次线上对话（问"张三还剩几天年假？"）：

      - 两次工具调用**都成功**（`retrieve_knowledge` 23ms、`query_business_db` 24ms，
        trace 里 `ok=True`、无 error），`retrieved_sources` 里也确实有《员工手册》；
      - 但模型在答案里写的却是 `【来源：业务系统个人数据查询结果（知识库检索失败,无制度文件可引用）】`
        —— 括号里那句话描述的是**一件没发生过的事**（知识库检索没有失败）。

    于是 `sources` 里出现了一个"看起来像引用、其实查无此据"的条目。对一个把"引用溯源"
    当核心卖点的项目，这比"引用多了/少了"严重得多：它把一次正常的检索说成了失败，
    还把这句话冒充成了来源名。所以这里做一次**可验证性**判定：
    引用必须出现在本轮工具真正返回过的来源里，否则归入 `unverified_sources`。

    为什么用"本轮检索到的文件集合"当基准，而不是"语料里的文件全集"：
      `data/docs/` 里的文件名不在 API 层可见（检索跑在 MCP 子进程里），要拿全集就得再开一次
      IO/依赖。而"本轮有没有检索到它"本来就是这两个字段要回答的问题，用它当基准不需要新依赖，
      且判据更严——即便语料里真有《员工手册》，若本轮没检索到，那条引用同样**没有本轮证据**。

    保留在 `sources` 里而不是直接删掉：删掉等于替模型粉饰，"答案写了什么"这个事实要留住；
    消费方按 `unverified_sources` 就能一眼看出哪些引用站不住。
    """
    retrieved_set = set(retrieved)
    # 工具因超时/异常没返回任何来源时，retrieved_set 为空，此时**不做判定**：
    # 那种情况下所有引用都"无法验证"，但原因是工具挂了而不是模型编造，混为一谈会让
    # 这个字段在故障时刷屏、反而失去信号意义。故障本身由 trace 的 ok=False 表达。
    if not retrieved_set:
        return list(citations), []
    verified = [c for c in citations if c in retrieved_set]
    unverified = [c for c in citations if c not in retrieved_set]
    return verified, unverified


def _final_answer(messages: list) -> str:
    """取"最后一条非工具调用的 AI 消息"作为答案。

    调用方必须传**本轮新增的消息切片**（见 AgentRuntime.ainvoke / astream 的 fresh）：
    若把完整历史传进来，本轮最终 AI 消息 content 为空串/None 时（模型偶发空回复），
    这里会一路往前找到**上一轮的答案**并当成本轮 answer 返回——既写进会话记录，
    又会让端到端评测的回答率虚高。限定在本轮消息里取，取不到就如实返回空串。
    """
    for message in reversed(messages):
        if (
            getattr(message, "type", "") == "ai"
            and not getattr(message, "tool_calls", None)
        ):
            # 用 _text_of 收口：content 若是 content block 列表，str() 会得到 Python repr。
            # 判空也必须在**规整之后**做：内容是 `[]` 或 `[{"type":"text","text":""}]` 时
            # `getattr(..., "content", "")` 是个 truthy 的容器，旧写法会误以为"这条有内容"，
            # 于是取到一个空串答案并就此返回（既不回退到上一轮，也不继续往前找）。
            text = _text_of(getattr(message, "content", ""))
            if text:
                return text
    return ""


def _usage_tokens(message) -> tuple[int, int]:
    """从 LLM 返回消息里取 token 用量（兼容 usage_metadata / token_usage）。"""
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict):
        return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
    meta = getattr(message, "response_metadata", None) or {}
    token_usage = (meta.get("token_usage") or {}) if isinstance(meta, dict) else {}
    return int(token_usage.get("prompt_tokens") or 0), int(token_usage.get("completion_tokens") or 0)


def _build_graph(llm, tool_session: "McpToolSession", checkpointer):
    async def agent_node(state: AgentState) -> dict:
        from app.observability import (
            Recorder,
            Step,
            estimate_cost,
            get_recorder,
            now_ms,
        )

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

        # 拿一份"当前有效的工具表 + 会话代次"。工具表会随会话重建整体换掉，
        # 所以每次派单都现取，不能在编译期把工具对象存进闭包（那样会话一旦重建，
        # 图里握着的就是已被关闭的旧会话，之后每次调用都失败）。
        #
        # `ensure()` 里含"拉起子进程 + 握手 + 子进程内冷加载索引"，实测约 0.7~3.0s。
        # 第一次派单本来就要付这笔钱；但**会话重建**（子进程被 OOM/手工杀掉）也会走这里，
        # 而 tools_node 是同步节点、卡住就等于卡住整个请求。所以同样套一层 step_timeout，
        # 把"子进程起不来时整个请求挂死"变成"这一步明确报超时、模型据此收尾"。
        try:
            lease = await asyncio.wait_for(
                tool_session.ensure(), timeout=AgentLimits.step_timeout_seconds
            )
        except TimeoutError:
            note = f"工具会话建立超时（>{AgentLimits.step_timeout_seconds}s）"
            if rec is not None:
                rec.add(Step(node="tools", name="*", duration_ms=0, ok=False, note=note))
                rec.error = note
            return {
                "messages": [
                    ToolMessage(
                        content=f"{note}。请基于已有信息作答，或请用户稍后重试。",
                        tool_call_id=call.get("id", ""),
                        name=call.get("name", ""),
                    )
                    for call in calls
                ]
            }
        except ToolSessionUnavailable as exc:
            # 子进程起不来（依赖缺失、索引损坏等）：如实告诉模型，让本轮以"工具不可用"收尾。
            # 不抛异常——抛出去会让 HTTP 层回 502，而模型其实还能基于历史信息作答。
            note = f"工具会话不可用：{exc}"[:300]
            if rec is not None:
                rec.add(Step(node="tools", name="*", duration_ms=0, ok=False, note=note))
                rec.error = note
            return {
                "messages": [
                    ToolMessage(
                        content=f"工具暂时不可用（{exc}）。请基于已有信息作答，或请用户稍后重试。",
                        tool_call_id=call.get("id", ""),
                        name=call.get("name", ""),
                    )
                    for call in calls
                ]
            }
        tools_by_name = lease.by_name

        # 判重窗口 = 最近 max_steps 次工具调用，实际只在"本次运行内"能命中：
        # history 存进 checkpoint 后 tuple 会被序列化成 list，跨轮次读回时
        # tuple key 与 list 条目不相等，因此不会误拦跨轮次的合法重复查询
        # （跨轮次重复提问本就应当重新检索，见 guardrails.is_duplicate_call）。
        history = list(state.get("tool_call_history", []))[-AgentLimits.max_steps :]

        # ---- 第一步：先"派单"（纯同步判断，必须在任何执行之前做完）----
        # 为什么不能边执行边判重：同一批里模型可能吐出两个完全相同的调用。串行实现里
        # 第二个会被第一个刚追加进 history 的 key 拦下；如果改成"先并发跑、事后再判重"，
        # 两个都会被执行——判重这道护栏就被这个改动悄悄拆掉了。所以先把整批的
        # 执行/跳过/报错决定一次性算出来，执行阶段只负责跑，不再改判。
        def plan() -> list[dict]:
            planned: list[dict] = []
            seen = list(history)
            for call in calls:
                name = call.get("name", "")
                args = call.get("args") or {}
                key = (name, json.dumps(args, ensure_ascii=False, sort_keys=True))
                tool = tools_by_name.get(name)
                if tool is None:
                    planned.append({"call": call, "name": name, "key": key, "tool": None,
                                    "content": f"未找到工具：{name}", "ok": False,
                                    "note": "工具不存在"})
                elif is_duplicate_call(seen, key):
                    planned.append({
                        "call": call, "name": name, "key": key, "tool": None,
                        "content": (
                            f"检测到重复工具调用（{name} {args}，本轮已执行过），"
                            "为避免死循环本次不再执行。请基于已有信息作答，或换个问法。"
                        ),
                        "ok": True, "note": "重复调用，已跳过",
                    })
                else:
                    seen.append(key)
                    planned.append({"call": call, "name": name, "key": key,
                                    "tool": tool, "content": None, "ok": True, "note": ""})
            return planned

        # ---- 第二步：并发执行本批里"要跑"的调用 ----
        # 模型在一次 assistant 消息里给出多个 tool_call，本身就表示它认为这些调用互不依赖
        # （例如"先查制度规则 + 再查个人数据"）。串行 for-await 会让每个调用各付一次往返：
        # 改成 gather 后两把工具并发（实测 8127ms → 4663ms；换成常驻会话后是 17ms 量级）。
        async def run_tool(item: dict) -> None:
            """执行单个工具调用，把结果写回 item。异常一律收敛成文本，不外抛。"""
            name = item["name"]
            args = item["call"].get("args") or {}
            t0 = now_ms()
            try:
                raw = await asyncio.wait_for(
                    item["tool"].ainvoke(args), timeout=AgentLimits.step_timeout_seconds
                )
                item["content"] = truncate_tool_output(
                    _text_of(raw), AgentLimits.max_tool_output_chars
                )
            except TimeoutError:
                item["content"] = (
                    f"工具 {name} 调用超时（>{AgentLimits.step_timeout_seconds}s）"
                )
                item["ok"] = False
                item["note"] = item["content"][:160]
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                # 会话/子进程死了：把这一代会话判死并让本调用**重试一次**。
                # 只重试一次是刻意的——若重建后仍然失败，说明不是"会话过期"而是真故障，
                # 再重试只会把延迟翻倍却不改变结果。
                if _is_transport_failure(exc):
                    await tool_session.invalidate(lease.generation)
                    try:
                        fresh = await tool_session.ensure()
                    except ToolSessionUnavailable as rebuild_exc:
                        item["content"] = (
                            f"工具 {name} 调用失败：MCP 会话已断开且重建失败"
                            f"（{rebuild_exc}）"
                        )
                        item["ok"] = False
                        item["note"] = item["content"][:160]
                        return
                    retry_tool = fresh.by_name.get(name)
                    if retry_tool is None:
                        item["content"] = f"工具 {name} 调用失败：会话重建后该工具已不存在"
                        item["ok"] = False
                        item["note"] = item["content"][:160]
                        return
                    try:
                        raw = await asyncio.wait_for(
                            retry_tool.ainvoke(args),
                            timeout=AgentLimits.step_timeout_seconds,
                        )
                        item["content"] = truncate_tool_output(
                            _text_of(raw), AgentLimits.max_tool_output_chars
                        )
                        item["retried"] = "会话失效后重建，重试成功"
                        item["duration_ms"] = now_ms() - t0
                        return
                    except Exception as retry_exc:  # noqa: BLE001
                        exc = retry_exc
                item["content"] = f"工具 {name} 调用失败：{type(exc).__name__}: {exc}"
                item["ok"] = False
                item["note"] = item["content"][:160]
            item["duration_ms"] = now_ms() - t0

        planned = plan()
        to_run = [item for item in planned if item["tool"] is not None]
        if to_run:
            await asyncio.gather(*(run_tool(item) for item in to_run))

        # ---- 第三步：按模型给出的顺序组装消息与 trace ----
        # 顺序必须稳定：ToolMessage 要严格对应各自的 tool_call_id，且 trace 的 steps
        # 不能因为"谁先跑完"而抖动（否则同一份报告两次跑出来的 steps 顺序都不一样）。
        new_messages: list = []
        for item in planned:
            new_messages.append(
                ToolMessage(
                    content=item["content"],
                    tool_call_id=item["call"].get("id", ""),
                    name=item["name"],
                )
            )
            if rec is not None:
                step = Step(node="tools", name=item["name"],
                            duration_ms=item.get("duration_ms", 0),
                            ok=item["ok"], note=item["note"])
                # trace 里要能看出"这一步到底调了哪把工具、什么参数"：原先 steps 只记 node/name，
                # 排查时只能靠顺序猜（name 与 node 在旧数据里都是 "tools"）。
                step.tool_calls = [item["name"]]
                step.tool_args = [item["call"].get("args") or {}]
                if item.get("retried"):
                    step.note = (step.note + "；" + item["retried"]).strip("；")
                rec.add(step)

        return {
            "messages": new_messages,
            # 与串行实现口径一致：整批调用的 key 都进历史（含被跳过/未找到的），
            # 只保留最近 max_steps 条；用 seen 而不是重算，避免与判重时的顺序漂移。
            "tool_call_history": (history + [i["key"] for i in planned])[-AgentLimits.max_steps :],
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
    """持有编译好的图与常驻 MCP 工具会话。会话标识 thread_id，崩溃后同 session 可续聊。"""

    def __init__(
        self,
        graph,
        tools: list,
        saver: AsyncSqliteSaver | None = None,
        conn=None,
        tool_session: McpToolSession | None = None,
    ):
        self.graph = graph
        # tools 只是"建图时的快照"，供 CLI/脚本/测试读工具名；真正派单走 tool_session，
        # 这样会话重建换掉工具表时不用重新编译图。
        self.tools = tools
        self._saver = saver
        self._conn = conn
        self._tool_session = tool_session

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
        # 答案与来源都只从本轮新增的消息里取：
        # sources 不累计历史轮次；answer 也不回退到上一轮（见 _final_answer 说明）
        fresh = final_messages[prior:]
        return self._result(fresh)

    @staticmethod
    def _result(fresh: list) -> dict:
        """统一收尾口径：answer / sources / retrieved_sources / tool_calls。

        两个来源字段是两个不同的问题，必须分开返回（见 _parse_citations 的实测说明）：
          - sources            = 答案正文里真实标注的【来源：X】——前端"引用来源"与用户直觉一致；
          - retrieved_sources  = 本轮工具返回过的来源（top_k 命中的文件）——观测/评测的检索口径。
        历史字段 `sources` 原本是后者，实测 9/9 条 trace 都比答案实际引用多 1 条，
        前端把它渲染成"来源："属于替答案多声明引用，因此本版本把语义改成前者。

        `tool_calls`（本轮实际执行的工具调用次数）是给上面两个字段**消歧**用的：
        多轮续聊时模型可能直接凭上下文作答、一次工具都不调（实测存在）。那时
        `retrieved_sources` 为空，可答案正文里仍写着【来源：员工手册_示例.md】——那是历史轮次
        留下的引用，本轮并没有检索。只有两个列表时，消费方无法区分
        "检索了但没命中"（召回问题）和"根本没检索"（这条引用没有本轮证据）。

        `unverified_sources`（查无此据的引用）与 `tool_calls` 是同一类"让引用可被质疑"的字段：
        实测模型会写出知识库里不存在的来源名、甚至写出"（知识库检索失败…）"这种**描述了一件
        没发生过的事**的假来源。见 `_split_verified_citations`。它只报事实、不改写答案。
        """
        answer = _final_answer(fresh)
        citations = _parse_citations(answer)
        retrieved = _parse_sources(fresh)
        _, unverified = _split_verified_citations(citations, retrieved)
        return {
            "answer": answer,
            "sources": citations,
            "retrieved_sources": retrieved,
            "unverified_sources": unverified,
            "tool_calls": sum(1 for m in fresh if getattr(m, "type", "") == "tool"),
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
        # 与 ainvoke 同口径：答案与来源都只看本轮新增消息
        yield self._result(final_messages[prior:])

    async def aclose_thread(self, session_id: str) -> None:
        """删掉某个 session 的 checkpoint 线程（评测/回归专用）。

        为什么需要：`ainvoke(session_id=None)` 会现造一个 uuid thread，而 checkpoint 表是
        服务真正在用的那个 `data/checkpoints.sqlite`。评测跑 40 条就是 40 个永不回收的线程
        （实测：只跑过几轮评测的库里有 257 个 thread_id、13.4 MB，而真实会话只有 9 个）。
        评测用例本就是一次性的，跑完即删，别让它污染服务侧的会话库。
        """
        if self._saver is None:
            return
        try:
            await self._saver.adelete_thread(session_id)
        except Exception:  # noqa: BLE001, S110
            # 清理失败不该让评测本身失败（指标已经拿到了）
            pass

    async def aclose(self) -> None:
        # 先关常驻 MCP 会话（含 stdio 子进程），再关 SQLite checkpoint 连接
        if self._tool_session is not None:
            try:
                await self._tool_session.aclose()
            except Exception:  # noqa: BLE001, S110
                pass
            self._tool_session = None
        if self._conn is not None:
            try:
                await self._conn.close()
            except Exception:  # noqa: BLE001, S110
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
    """工厂：建常驻 MCP 工具会话、建图，返回一个封装好生命周期(resource)的 AgentRuntime。

    OOP/设计要点：真正"需要对象封装"的是这里的**运行时生命周期**——checkpointer 的
    AsyncSqliteSaver、sqlite 连接 conn、常驻 MCP 会话(McpToolSession)、graph 都需要在请求
    结束后正确关闭 (aclose)，不能靠模块级散落。所以返回 AgentRuntime(封装 graph+tools+saver+
    conn+tool_session) 并约定由调用方负责用毕 aclose()：这是"用对象管理成对分配/释放资源
    (open/close)"的典型场景，普通函数无法靠返回值表达"记得关连接"的约束。
    （内部 make_llm / 建会话 / 建图任一步抛错，都会先关掉已开资源再 raise，避免泄漏
    —— 见下方 try/except。）
    """
    saver, conn = await _open_checkpointer()
    tool_session = McpToolSession(
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
        # 建会话并取工具表：启动即暴露"子进程起不来/索引坏了"，而不是等到第一次提问。
        # 这一版工具表在服务生命周期内复用（不再每次调用新起子进程），
        # 因此这次启动开销从"每次都付"变成"只付一次"。
        lease = await tool_session.ensure()
        tools = lease.tools
        llm = llm.bind_tools(tools)
        graph = _build_graph(llm, tool_session, checkpointer=saver)
    except BaseException:
        # 任一步失败（含 CancelledError 这类 BaseException）都要先关掉已开资源：
        # services.ensure_services 允许每次请求重试，不关就每请求泄漏一个连接+子进程。
        await tool_session.aclose()
        await conn.close()
        raise
    return AgentRuntime(
        graph=graph, tools=tools, saver=saver, conn=conn, tool_session=tool_session
    )


async def run_single_question(question: str, session_id: str | None = None) -> dict:
    """CLI/脚本用：一次 asyncio.run 内建运行时并答一个问题（演示/自测）。"""
    runtime = await create_runtime()
    try:
        return await runtime.ainvoke([{"role": "user", "content": question}], session_id)
    finally:
        await runtime.aclose()

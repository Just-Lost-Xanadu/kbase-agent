from dataclasses import dataclass


@dataclass
class AgentLimits:
    max_steps: int = 25
    step_timeout_seconds: int = 120
    max_tool_output_chars: int = 4000


def default_recursion_limit(max_steps: int = AgentLimits.max_steps) -> int:
    """LangGraph recursion_limit 的默认值，与 prompt 里的 max_steps 同源。

    LangGraph 的 recursion_limit 计"节点执行步数"：每轮工具调用要经过
    agent + tools 两个节点（约 2 步），再加首尾各 1 步，因此取 2×max_steps+5，
    保证模型按 prompt 承诺最多执行 max_steps 次工具调用时，不会被框架先掐断。
    需要更早兜底死循环时可设小 MAX_RECURSION 覆盖。
    """
    return max_steps * 2 + 5


def truncate_tool_output(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    head_end = int(limit * 0.6)
    tail_start = max(head_end, len(text) - int(limit * 0.2))
    head = text[:head_end]
    tail = text[tail_start:]
    return f"{head}\n...[输出过长已截断，共 {len(text)} 字符]...\n{tail}"


def is_duplicate_call(
    call_history: list[tuple[str, str]], key: tuple[str, str]
) -> bool:
    """对给定调用历史判重：同工具、同参数在历史中出现过即视为重复。

    调用方（graph.tools_node）传入的是"最近 max_steps 次"的滑动窗口：
    窗口内能兜住 A→B→A 式隔步打转，窗口外（更早轮次的相同提问）允许重跑，
    避免跨轮次的合法重复查询被误拦。工具调用都是幂等查询，跳过不丢信息。
    """
    return key in call_history

"""Agent 护栏：把"防死循环/防上下文爆炸/防资源拉爆"的三件事集中成纯函数 + 一个默认配置 dataclass。

设计说明：这些护栏刻意写成**纯函数 / 无副作用的 dataclass 默认配置**，而不是散落在 graph 里——
graph 里只调用它们并把结果写回状态，方便单独用 pytest 测每个护栏（tests/test_units.py 就是这么测的）；
也因为纯函数可组合、无隐藏状态，面试讲"护栏"时能一条条讲清对应哪个函数、入参出参。

三座防线（面试必讲三层）：
  1) recursion_limit（框架层）   —— LangGraph 节点步数上限，掐停无限循环；见 default_recursion_limit。
  2) 工具输出截断（上下文层）    —— 超长工具结果只保留头尾+中间提示，防把上下文塞爆；见 truncate_tool_output。
  3) 重复工具调用判重（逻辑层，窗口 = 最近 max_steps 次调用）—— A→B→A 原地打转时拦截并让模型收尾；见 is_duplicate_call。
每层答"哪一层、在哪拦"是面试高频点。
"""
from dataclasses import dataclass


@dataclass
class AgentLimits:
    """护栏的默认档位（单一事实来源）。

    说明：max_steps / step_timeout_seconds / max_tool_output_chars 是写死的默认值（暂无 .env 项）；
    仅 recursion_limit 可由 .env 的 MAX_RECURSION 覆盖（见 config.py）。
    max_steps 同时被 prompt 第 5 条（写明"最多执行 N 步工具调用"）与 default_recursion_limit 引用——
    改这里要同步确认两处口径，不要只动一处造成"prompt 承诺 25 步，框架却 20 步就掐"这类不一致。
    （另注意：.env 里的 MAX_RECURSION 会直接覆盖推导值，设小了同样会破坏这个一致性。）
    """
    max_steps: int = 25                    # 提示词/框架的"一次回答最多工具调用次数"
    step_timeout_seconds: int = 120        # 单次工具调用的顶层兜底超时(包住整个 stdio 往返)
    max_tool_output_chars: int = 4000      # 工具结果回给模型前的最大字符数


def default_recursion_limit(max_steps: int = AgentLimits.max_steps) -> int:
    """LangGraph recursion_limit 的默认值，与 prompt 里的 max_steps 同源。

    LangGraph 的 recursion_limit 计"节点执行步数"：每轮工具调用要经过
    agent + tools 两个节点（约 2 步），再加首尾各 1 步，因此取 2×max_steps+5，
    保证模型按 prompt 承诺最多执行 max_steps 次工具调用时，不会被框架先掐断。
    需要更早兜底死循环时可设小 MAX_RECURSION 覆盖。
    """
    return max_steps * 2 + 5


def truncate_tool_output(text: str, limit: int = 4000) -> str:
    """工具结果过长时做"保头保尾中间截断"，既省 token 又不丢关键首尾。

    策略：保留长度 limit 的 60% 作开头 + 最后 20% 作为尾巴（中间用[输出过长已截断]标记）。
    为什么保头保尾：工具返回常是"头部结论/列头 + 尾部明细/统计"，纯从中间切会对模型更有用；
    留中间提示避免模型误以为这就是完整输出。（若要更省可改成摘要，README 已列方向。）
    """
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

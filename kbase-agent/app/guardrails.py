from dataclasses import dataclass


@dataclass
class AgentLimits:
    max_steps: int = 25
    step_timeout_seconds: int = 120
    max_tool_output_chars: int = 4000


def truncate_tool_output(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    head_end = int(limit * 0.6)
    tail_start = max(head_end, len(text) - int(limit * 0.2))
    head = text[:head_end]
    tail = text[tail_start:]
    return f"{head}\n...[输出过长已截断，共 {len(text)} 字符]...\n{tail}"


def detect_duplicate_tool_call(call_history: list[tuple[str, str]]) -> bool:
    if len(call_history) < 2:
        return False
    return call_history[-1] == call_history[-2]

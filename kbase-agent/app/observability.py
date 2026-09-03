"""轻量可观测：记录每次 Agent 运行的节点级 trace 与 token/成本估算。

原理：不侵入 LangGraph state，用 contextvars 在当前请求任务内挂一个 Recorder，
graph 的 agent/tools 节点只做只读采集（get_recorder()），调用方（API / 脚本）
在结束后读取并持久化。并发请求各自任务上下文隔离，互不串扰。
"""

import contextvars
import time
from dataclasses import dataclass, field, asdict

from app.config import settings

_current: contextvars.ContextVar["Recorder | None"] = contextvars.ContextVar(
    "kb_recorder", default=None
)


@dataclass
class Step:
    node: str            # agent / tools
    name: str = ""       # llm 或工具名
    duration_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_cny: float = 0.0
    ok: bool = True
    note: str = ""


class Recorder:
    def __init__(self, question: str = ""):
        self.question = question
        self.steps: list[Step] = []
        self.started_at = time.time()
        self.error: str = ""
        self.answer: str = ""

    # ---- 采集（graph 节点调用）----
    def add(self, step: Step) -> None:
        self.steps.append(step)

    def summarize(self) -> dict:
        total_in = sum(s.prompt_tokens for s in self.steps)
        total_out = sum(s.completion_tokens for s in self.steps)
        cost = round(sum(s.cost_cny for s in self.steps), 4)
        return {
            "question": self.question,
            "duration_ms": int((time.time() - self.started_at) * 1000),
            "llm_calls": sum(1 for s in self.steps if s.name == "llm"),
            "tool_calls": sum(1 for s in self.steps if s.name != "llm"),
            "prompt_tokens": total_in,
            "completion_tokens": total_out,
            "total_tokens": total_in + total_out,
            "cost_cny": cost,
            "error": self.error,
            "steps": [asdict(s) for s in self.steps],
        }


def set_recorder(rec: "Recorder | None") -> contextvars.Token:
    return _current.set(rec)


def reset_recorder(token: contextvars.Token) -> None:
    _current.reset(token)


def get_recorder() -> "Recorder | None":
    return _current.get()


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens / 1_000_000 * settings.llm_input_price
        + completion_tokens / 1_000_000 * settings.llm_output_price
    )


def now_ms() -> int:
    return int(time.time() * 1000)

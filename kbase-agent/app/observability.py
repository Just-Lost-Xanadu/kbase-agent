"""轻量可观测：记录每次 Agent 运行的节点级 trace 与 token/成本估算。

原理：不侵入 LangGraph state，用 contextvars 在当前请求任务内挂一个 Recorder（隐式上下文，
无需把 recorder 作为参数一路透传到每个 graph 节点），graph 的 agent/tools 节点只做只读采集
（get_recorder()），调用方（API / 脚本）在结束后读取并持久化。并发请求各自任务上下文隔离，
互不串扰（这正是 contextvars 相对"模块级全局变量"的核心价值）。

数据落点：Recorder.summarize() 产出 dict，由 API 层存到 store 的 run_traces 表，并暴露
GET /api/runs 查询——回答"调试 Agent 时怎么观察某次跑了哪些节点/花了多少 token/成本"。

成本口径：cost 是"估算值"——由 estimate_cost() 按 settings 里每百万 token 单价（env:
LLM_PRICE_INPUT/LLM_PRICE_OUTPUT）乘 token 数算得，非真实账单。面试时口径说"单价来自配置、
估算而非计费回执"更稳。
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
    """一次节点/工具级的观测单元。

    node: "agent"(一次 LLM 调用) 或 "tools"(一次工具调用)。
    该 dataclass 仅是"记录单元"：无自身行为逻辑、只作为数据携带，故用 @dataclass 而非完整类
    最合适（Python 里"纯数据传输"用 dataclass，有方法逻辑的才是需要设计对象的类）。
    """
    node: str            # agent(LLM) / tools(工具)
    name: str = ""       # llm 或具体工具名
    duration_ms: int = 0
    prompt_tokens: int = 0          # 该次输入 token
    completion_tokens: int = 0      # 该次输出 token
    cost_cny: float = 0.0           # 该次估算成本（由 estimate_cost 折算）
    ok: bool = True                 # 是否成功（异常节点置 False）
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

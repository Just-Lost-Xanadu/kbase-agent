from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    # 引用来源不放进图状态：最终答案的 sources 由 ainvoke/astream 收尾时
    # 从消息流解析（见 graph._parse_sources），避免图里维护冗余字段。
    tool_call_history: list[tuple[str, str]]

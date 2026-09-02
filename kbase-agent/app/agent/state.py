from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    sources: list[str]
    tool_call_history: list[tuple[str, str]]

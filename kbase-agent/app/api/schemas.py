from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(description="历史消息 + 最新用户输入")
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list, description="引用来源")

from langchain_openai import ChatOpenAI

from app.config import settings


def make_llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.model_name,
        base_url=settings.base_url,
        api_key=settings.api_key,
        temperature=temperature,
    )

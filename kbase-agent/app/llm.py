from langchain_openai import ChatOpenAI

from app.config import settings


def make_llm(temperature: float = 0.0) -> ChatOpenAI:
    if not settings.api_key or settings.api_key == "sk-your-key":
        raise ValueError(
            "未配置有效的 DEEPSEEK_API_KEY。请在项目根目录创建 .env"
            "（复制 .env.example 并填入真实 key，占位符 sk-your-key 不会被接受），"
            "或先执行：$env:DEEPSEEK_API_KEY='sk-真实key'。"
            "注意 .env 需位于当前工作目录（从项目根启动 uvicorn/脚本）。"
        )
    return ChatOpenAI(
        model=settings.model_name,
        base_url=settings.base_url,
        api_key=settings.api_key,
        temperature=temperature,
    )

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # 模型（默认 DeepSeek，OpenAI 兼容层模型无关）
    model_name: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    # embedding：默认 fastembed（ONNX，免 torch）；flagembedding 才用 bge-m3（需 .[embed]）
    embed_backend: str = os.getenv("EMBED_BACKEND", "fastembed")
    fastembed_model: str = os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-zh-v1.5")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "bge-m3")

    # rerank：opt-in（FlagReranker 依赖 torch，需 pip install -e ".[embed]"）
    rerank_enabled: bool = _flag("RERANK_ENABLED", False)
    rerank_model: str = os.getenv("RERANK_MODEL", "bge-reranker-v2-m3")

    # 向量库（开发用 Chroma）
    chroma_path: str = os.getenv("CHROMA_PATH", "./data/chroma")
    collection_name: str = os.getenv("COLLECTION_NAME", "kbase")

    # Agent 护栏：recursion_limit 可选覆盖；不设则用默认值
    # （app.guardrails.default_recursion_limit()，由 max_steps 推导 = 2×max_steps+5）。
    # 这样 prompt 承诺的 max_steps 步工具调用不会被框架提前掐断。
    max_recursion: int | None = (
        int(os.getenv("MAX_RECURSION")) if os.getenv("MAX_RECURSION") else None
    )


settings = Settings()

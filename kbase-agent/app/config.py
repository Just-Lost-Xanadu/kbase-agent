import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _num(name: str, default: float, cast=float):
    """读数值型环境变量，出错时报出变量名与原始值。

    直接用 float(os.getenv(...)) 的话，`.env` 里写成 `LLM_TEMPERATURE=0.7 # 注释`
    或漏了数字，会在 **import 期**抛一个裸 ValueError（"could not convert string to float"），
    既不说是哪个变量、也没有上下文——排查成本远高于这一层包装。
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return cast(default)
    try:
        return cast(raw.strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"环境变量 {name} 的值无法解析为数字：{raw!r}（检查 .env，注意不要带行内注释）"
        ) from exc


@dataclass
class Settings:
    # 模型（默认 DeepSeek，OpenAI 兼容层模型无关）
    model_name: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    temperature: float = _num("LLM_TEMPERATURE", 0.0)

    # 成本核算（元/百万 token，仅用于 trace 估算，可按实际套餐改）
    llm_input_price: float = _num("LLM_PRICE_INPUT", 2.0)
    llm_output_price: float = _num("LLM_PRICE_OUTPUT", 8.0)

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

    # Agent checkpoint 持久化（SQLite + WAL，重启不丢会话）
    checkpoint_db: str = os.getenv("CHECKPOINT_DB", "./data/checkpoints.sqlite")

    # Agent 护栏：recursion_limit 可选覆盖；不设则用默认值
    # （app.guardrails.default_recursion_limit()，由 max_steps 推导 = 2×max_steps+5）。
    # 这样 prompt 承诺的 max_steps 步工具调用不会被框架提前掐断。
    max_recursion: int | None = (
        _num("MAX_RECURSION", 0, int) if os.getenv("MAX_RECURSION", "").strip() else None
    )


settings = Settings()

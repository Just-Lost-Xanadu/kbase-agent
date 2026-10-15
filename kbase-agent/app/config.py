import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# 项目根目录（app/ 的上一级）。用于把下面几个**数据路径**统一解析成绝对路径。
ROOT = Path(__file__).resolve().parents[1]


def _path(name: str, default: str) -> str:
    """读路径型环境变量，并把相对路径**按项目根**（不是进程 cwd）解析成绝对路径。

    为什么必须这么做：`FASTEMBED_CACHE_DIR` / `CHROMA_PATH` / `CHECKPOINT_DB` 的默认值
    原先写的是 `./data/...`，按**当前工作目录**解析。而 API 进程与它拉起的 MCP stdio 子进程
    的 cwd 并不保证一致——`graph.py` 拉起子进程时显式传了 `cwd=ROOT`，主进程却可能在别处
    （IDE、调度器、或任何 `cd` 到别的目录再运行 `uvicorn` 的场景）。
    于是父进程写 `<cwd>/data/chroma`、子进程读 `<ROOT>/data/chroma`，**指向两个不同的索引目录，
    而且不报任何错**——表现为"索引明明建好了，检索却一直说没有索引"，
    或者更糟：两边各自建一份索引，数字对不上。
    归一成绝对路径之后，父子进程无论如何都指向同一处。
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return str((ROOT / default).resolve())
    p = Path(raw.strip()).expanduser()
    return str(p if p.is_absolute() else (ROOT / p).resolve())


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


def _recursion() -> int | None:
    """读 MAX_RECURSION 并校验下限；不合法就回退默认值（None），不让一个环境变量打死整个服务。

    为什么需要这一层：LangGraph 对 recursion_limit 的要求是 **>= 1**，而它按"节点执行步数"计
    （一轮工具调用要经 agent + tools 两个节点）。原先的用法是
    `settings.max_recursion or 默认值`，而 `or` 只兜住了 **0**（falsy）——
    `MAX_RECURSION=-1` 会被 LangGraph 直接拒绝、`MAX_RECURSION=1` 会让第一个节点就超限，
    两者都表现为"每个 /chat 请求 502"，而报错信息完全指不到这个环境变量。
    非数字仍然显式报错（`_num` 会带出变量名与原始值）——那是配置写错了，应该立刻可见。
    """
    raw = os.getenv("MAX_RECURSION", "")
    if not raw.strip():
        return None
    value = _num("MAX_RECURSION", 0, int)
    if value < 2:
        print(
            f"[config] MAX_RECURSION={value} 不合法（至少要 2：一轮工具调用要经 agent+tools "
            "两个节点），已回退为默认值（2×max_steps+5）。",
            file=sys.stderr,
        )
        return None
    return value


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
    # 模型权重缓存目录。必须显式指定：fastembed 不指定时的默认值是**系统临时目录**
    # （%TEMP%\fastembed_cache），而临时目录会被系统/清理工具删掉——实测遇到过一次，
    # 后果是"索引还在、模型没了"，index_docs.py 必须重新联网下载，离线环境直接跑不起来。
    # 放到项目 data/ 下（已 gitignore）之后缓存跟着仓库走，不会平白消失。
    # 注意 _path 会把相对路径按**项目根**解析成绝对路径（见其 docstring：
    # 主进程与 MCP 子进程的 cwd 不一致时，相对路径会让两边指向不同的目录）。
    fastembed_cache_dir: str = _path("FASTEMBED_CACHE_DIR", "data/fastembed_cache")
    # 注意：本字段**只在 embed_backend="flagembedding" 时被读取**（app/retrieval/embedder.py）。
    # 默认的 fastembed 后端走的是上面的 fastembed_model，所以这个默认值在默认路径上不生效，
    # 改它是没有用的——要换默认路径的模型请改 FASTEMBED_MODEL。
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "bge-m3")

    # rerank：opt-in（FlagReranker 依赖 torch，需 pip install -e ".[embed]"）
    rerank_enabled: bool = _flag("RERANK_ENABLED", False)
    rerank_model: str = os.getenv("RERANK_MODEL", "bge-reranker-v2-m3")

    # 向量库（开发用 Chroma）。同 _path：相对值按项目根解析。
    chroma_path: str = _path("CHROMA_PATH", "data/chroma")
    collection_name: str = os.getenv("COLLECTION_NAME", "kbase")

    # Agent checkpoint 持久化（SQLite + WAL，重启不丢会话）。同 _path。
    checkpoint_db: str = _path("CHECKPOINT_DB", "data/checkpoints.sqlite")

    # Agent 护栏：recursion_limit 可选覆盖；不设则用默认值
    # （app.guardrails.default_recursion_limit()，由 max_steps 推导 = 2×max_steps+5）。
    # 这样 prompt 承诺的 max_steps 步工具调用不会被框架提前掐断。
    #
    # ⚠️ 必须校验下限：LangGraph 对 recursion_limit 的要求是 **>= 1**，而它实际按
    # "节点执行步数"计——一轮工具调用就要经 agent + tools 两个节点。所以
    #   · `MAX_RECURSION=1`  → 第一个节点就超限，每个请求立刻 GraphRecursionError；
    #   · `MAX_RECURSION=-1` → LangGraph 直接抛 `ValueError: recursion_limit must be at least 1`；
    # 而原先的写法是 `settings.max_recursion or 默认值`，`or` 只兜住了 **0**（falsy），
    # 负数与 1 会原样传下去——一个环境变量就能让整个服务对所有请求回 502。
    # 这里把"不合法"和"没配"一视同仁地回退到默认值，并在 stderr 说明原因。
    max_recursion: int | None = _recursion()


settings = Settings()

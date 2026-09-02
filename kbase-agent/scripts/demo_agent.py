"""命令行演示：python scripts/demo_agent.py "张三还剩几天年假？"（默认示例问题）"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval.pipeline import RetrievalPipeline  # noqa: E402


def _ensure_index() -> None:
    pipeline = RetrievalPipeline()
    if not pipeline.is_indexed():
        pipeline.index()


async def _ask(question: str) -> None:
    from app.agent.graph import run_single_question

    result = await run_single_question(question)
    print("\n==== 答案 ====")
    print(result["answer"])
    if result["sources"]:
        print("\n==== 引用来源 ====")
        for source in result["sources"]:
            print(f"- {source}")


def main() -> None:
    question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "我今年还剩几天年假？按手册规则能结转吗？"
    )
    _ensure_index()
    asyncio.run(_ask(question))


if __name__ == "__main__":
    main()

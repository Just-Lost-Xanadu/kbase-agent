"""端到端回归评测：真实跑 Agent，验证回答引用是否覆盖真值来源。

与 scripts/eval.py 的区别：
- eval.py 只测"检索层 top-k 是否命中"（离线、免费、快）
- 本脚本让 Agent 对每条金标问题真实调用 LLM/工具，判最终回答或 sources 是否覆盖 expected_source，
  并顺带统计耗时/token/成本——是"prompt/模型/工具改动后不劣化"的回归手段。

注意：会真实调用 DeepSeek API（40 条约 1~3 元量级），不像 eval.py 那样零成本。

用法：
    python scripts/eval_e2e.py              # 全量 40 条
    python scripts/eval_e2e.py --limit 5    # 只跑前 5 条（冒烟）
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import hit_rate  # noqa: E402

EVAL_FILE = Path(__file__).resolve().parents[1] / "eval" / "questions.jsonl"


async def _run_case(runtime, question: str, expected_source: str) -> dict:
    from app.observability import Recorder, reset_recorder, set_recorder

    rec = Recorder(question=question)
    token = set_recorder(rec)
    try:
        result = await runtime.ainvoke([{"role": "user", "content": question}])
        error = ""
    except Exception as exc:  # noqa: BLE001
        result = {"answer": "", "sources": []}
        error = str(exc)[:200]
        rec.error = error
    finally:
        reset_recorder(token)
    summary = rec.summarize()

    answer = result["answer"] or ""
    sources = result["sources"] or []
    # 引用覆盖：真值来源出现在"结构化 sources"或"回答正文【来源】标记"里都算
    cited = any(expected_source in s for s in sources) or (expected_source in answer)
    return {
        "id": 0,  # 由调用方填充
        "answer_nonempty": bool(answer.strip()),
        "citation_covered": cited,
        "tool_calls": summary["tool_calls"],
        "llm_calls": summary["llm_calls"],
        "duration_ms": summary["duration_ms"],
        "cost_cny": summary["cost_cny"],
        "total_tokens": summary["total_tokens"],
        "error": error or None,
    }


async def main(limit: int) -> None:
    from app.agent.graph import create_runtime

    cases = [json.loads(line) for line in EVAL_FILE.read_text(encoding="utf-8").splitlines()]
    if limit and limit > 0:
        cases = cases[:limit]

    runtime = await create_runtime()
    results = []
    try:
        for idx, case in enumerate(cases, start=1):
            r = await _run_case(runtime, case["question"], case["expected_source"])
            r["id"] = case["id"]
            results.append(r)
            mark = "OK" if r["citation_covered"] and r["answer_nonempty"] else "FAIL"
            print(
                f"[{idx}/{len(cases)}] #{case['id']} {mark} "
                f"tools={r['tool_calls']} cost=¥{r['cost_cny']:.4f} "
                f"tok={r['total_tokens']} {r['error'] or ''}"
            )
    finally:
        await runtime.aclose()

    total_cost = round(sum(r["cost_cny"] for r in results), 4)
    print("\n==== 端到端评测汇总 ====")
    print(json.dumps({
        "cases": len(results),
        "answer_rate": round(hit_rate([{"hit": r["answer_nonempty"]} for r in results]), 4),
        "citation_accuracy": round(hit_rate([{"hit": r["citation_covered"]} for r in results]), 4),
        "total_cost_cny": total_cost,
        "avg_tokens": round(sum(r["total_tokens"] for r in results) / max(len(results), 1)),
    }, ensure_ascii=False, indent=2))

    failures = [r for r in results if not (r["citation_covered"] and r["answer_nonempty"])]
    if failures:
        print("\n失败用例（需坏例调参的入手点）：")
        for r in failures:
            print(f"  #{r['id']}  citation={r['citation_covered']} "
                  f"answer={r['answer_nonempty']} err={r['error']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全量）")
    args = parser.parse_args()
    asyncio.run(main(limit=args.limit))

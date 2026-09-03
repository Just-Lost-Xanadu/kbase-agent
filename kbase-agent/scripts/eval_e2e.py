"""端到端回归评测（Harness）：真实跑 Agent，产出可复现的评测报告与基线 diff。

分层说明（与 scripts/eval.py 的区别）：
- eval.py        ：检索层 Harness（离线、零成本、快），测 top-k 是否命中真值来源。
- eval_e2e.py    ：Agent 层 Harness（真实调 DeepSeek），测"回答可用 + 引用覆盖真值来源"，
                   并统计耗时/token/成本。是 prompt/模型/工具改动后防劣化的回归手段。

用法：
    python scripts/eval_e2e.py --tag baseline              # 全量 40 条，记为 baseline
    python scripts/eval_e2e.py --tag v2-prompt             # 改动后再跑一次
    python scripts/eval_e2e.py --tag v2-prompt --compare baseline   # 打印与基线的回归 diff
    python scripts/eval_e2e.py --limit 5                   # 冒烟（不落报告）
报告产物：docs/eval-reports/{tag}.json（含 per-case，可被 --compare 引用）
"""

import argparse
import asyncio
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:  # Python < 3.7
    pass

from eval.metrics import hit_rate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVAL_FILE = ROOT / "eval" / "questions.jsonl"
REPORT_DIR = ROOT / "docs" / "eval-reports"


def _load_cases() -> list[dict]:
    return [json.loads(line) for line in EVAL_FILE.read_text(encoding="utf-8").splitlines()]


def _metrics(results: list[dict]) -> dict:
    return {
        "cases": len(results),
        "answer_rate": round(hit_rate([{"hit": r["answer_nonempty"]} for r in results]), 4),
        "citation_accuracy": round(hit_rate([{"hit": r["citation_covered"]} for r in results]), 4),
        "total_cost_cny": round(sum(r["cost_cny"] for r in results), 4),
        "avg_tokens": round(sum(r["total_tokens"] for r in results) / max(len(results), 1)),
        "passed": sum(1 for r in results if r["answer_nonempty"] and r["citation_covered"]),
    }


def _compare(tag_a: str, tag_b: str, a: dict, b: dict) -> None:
    """打印 A(tag_b) vs B(tag_a) 的回归 diff。"""
    ma, mb = a["metrics"], b["metrics"]
    print(f"\n==== 回归 diff：{tag_b} vs {tag_a} ====")
    rows = [
        ("cases", "{}", "{}", ""),
        ("answer_rate", "{:.4f}", "{:.4f}", "{:+.4f}"),
        ("citation_accuracy", "{:.4f}", "{:.4f}", "{:+.4f}"),
        ("avg_tokens", "{:.0f}", "{:.0f}", "{:+.0f}"),
        ("total_cost_cny", "{:.4f}", "{:.4f}", "{:+.4f}"),
    ]
    for key, fa, fb, fd in rows:
        base_v = ma[key]
        new_v = mb[key]
        diff = (new_v - base_v) if key != "cases" else 0
        print(f"  {key:<18} {fa.format(base_v)} -> {fb.format(new_v)}  {fd.format(diff)}")

    base_by_id = {r["id"]: r for r in a["per_case"]}
    new_by_id = {r["id"]: r for r in b["per_case"]}
    regressed = [
        r for r in b["per_case"]
        if r["id"] in base_by_id
        and base_by_id[r["id"]]["citation_covered"] and not r["citation_covered"]
    ]
    recovered = [
        r for r in b["per_case"]
        if r["id"] in base_by_id
        and not base_by_id[r["id"]]["citation_covered"] and r["citation_covered"]
    ]
    if regressed:
        print("\n  [!] 本次劣化（需坏例调参）：")
        for r in regressed:
            print(f"    #{r['id']}  base OK -> now FAIL  err={r['error']}")
    if recovered:
        print("\n  [+] 本次修复：")
        for r in recovered:
            print(f"    #{r['id']}  base FAIL -> now OK")
    if not regressed and not recovered:
        print("  （无逐条变化）")


async def main(limit: int, tag: str | None, compare: str | None) -> None:
    from app.agent.graph import create_runtime

    cases = _load_cases()
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

    metrics = _metrics(results)
    print("\n==== 端到端评测汇总 ====")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    report = {
        "tag": tag or f"run-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "per_case": results,
    }
    if tag and not limit:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORT_DIR / f"{tag}.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已写入: {out}")

    if compare and tag:
        base_file = REPORT_DIR / f"{compare}.json"
        if not base_file.exists():
            print(f"\n[x] 找不到基线报告 {base_file}，请先 --tag {compare} 跑一次。")
        else:
            base = json.loads(base_file.read_text(encoding="utf-8"))
            _compare(compare, tag, base, report)

    failures = [r for r in results if not (r["citation_covered"] and r["answer_nonempty"])]
    if failures:
        print("\n失败用例（坏例调参入手点）：")
        for r in failures:
            print(f"  #{r['id']}  citation={r['citation_covered']} "
                  f"answer={r['answer_nonempty']} err={r['error']}")


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
    cited = any(expected_source in s for s in sources) or (expected_source in answer)
    return {
        "id": 0,
        "answer_nonempty": bool(answer.strip()),
        "citation_covered": cited,
        "tool_calls": summary["tool_calls"],
        "llm_calls": summary["llm_calls"],
        "duration_ms": summary["duration_ms"],
        "cost_cny": summary["cost_cny"],
        "total_tokens": summary["total_tokens"],
        "error": error or None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全量）")
    parser.add_argument("--tag", type=str, default=None, help="本次运行命名（写入 docs/eval-reports/{tag}.json）")
    parser.add_argument("--compare", type=str, default=None, help="与 docs/eval-reports/{tag}.json 的基线做回归 diff")
    args = parser.parse_args()
    asyncio.run(main(limit=args.limit, tag=args.tag, compare=args.compare))

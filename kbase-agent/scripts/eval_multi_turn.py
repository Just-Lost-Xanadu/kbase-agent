"""多轮 / 工具边界回归：同一 session 连问多轮，检查不串人、不编引用、不默认员工。

与 `eval_e2e.py` 的分工：
- `eval_e2e.py` 测的是 **60 条单轮金标**，分母固定、与已入库基线可比；
- 本脚本测的是 **多轮与工具边界**，用例少、只做冒烟，**不并入 60 条分母**，
  避免改动已有评测口径。

覆盖四类真实难例：
1. 同一会话追问（第二轮可能不再调工具，直接基于上下文作答）；
2. 个人数据没给姓名（工具会拒绝猜默认员工，答案必须反问姓名）；
3. 同一会话切换员工（上一轮张三、下一轮李四，不能把上一个人的数据带过来）；
4. 查询不存在员工（工具无结果，答案必须如实说明而不是编一个数）。

用法：
    python scripts/eval_multi_turn.py                 # 全量 4 条
    python scripts/eval_multi_turn.py --limit 2       # 冒烟
    python scripts/eval_multi_turn.py --verbose       # 打印每轮答案

退出码：0 全通过；1 有失败。需要 DEEPSEEK_API_KEY（真实调 Agent）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASES_FILE = ROOT / "eval" / "multi_turn.jsonl"


def load_cases(path: Path = CASES_FILE) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def _check_turn(turn: dict, result: dict) -> list[str]:
    failures: list[str] = []
    answer = result.get("answer", "") or ""
    if not answer.strip():
        failures.append("答案为空")
    for bad in turn.get("forbid", []):
        if bad in answer:
            failures.append(f"出现禁止内容：{bad}")
    expect = turn.get("expect_any") or []
    if expect and not any(word in answer for word in expect):
        failures.append(f"未命中任一关键词：{expect}")
    unverified = result.get("unverified_sources") or []
    if unverified:
        failures.append(f"出现查无此据的引用：{unverified}")
    return failures


async def run_case(runtime, case: dict) -> dict:
    session_id = f"mt-{case['id']}-{uuid.uuid4().hex[:8]}"
    results: list[dict] = []
    try:
        for turn in case["turns"]:
            result = await runtime.ainvoke(
                [{"role": "user", "content": turn["question"]}],
                session_id=session_id,
            )
            results.append(result)
    finally:
        await runtime.aclose_thread(session_id)

    failures: list[str] = []
    for idx, (turn, result) in enumerate(zip(case["turns"], results, strict=True), start=1):
        for message in _check_turn(turn, result):
            failures.append(f"第 {idx} 轮：{message}")
    return {"id": case["id"], "name": case.get("name", ""), "results": results, "failures": failures}


async def main_async(args: argparse.Namespace) -> int:
    from app.agent.graph import create_runtime

    cases = load_cases()
    if args.limit:
        cases = cases[: args.limit]

    runtime = await create_runtime()
    rows: list[dict] = []
    try:
        for idx, case in enumerate(cases, start=1):
            row = await run_case(runtime, case)
            rows.append(row)
            status = "OK" if not row["failures"] else "FAIL"
            print(f"[{idx}/{len(cases)}] {row['id']} {status} [{row['name']}]")
            if args.verbose or row["failures"]:
                for turn, result in zip(case["turns"], row["results"], strict=True):
                    print(f"  Q: {turn['question']}")
                    print(f"  A: {result.get('answer', '')[:300]}")
                for failure in row["failures"]:
                    print(f"  ! {failure}")
    finally:
        await runtime.aclose()

    failed = [row for row in rows if row["failures"]]
    print("\n==== 多轮 / 工具边界回归 ====")
    print(json.dumps(
        {
            "cases": len(rows),
            "passed": len(rows) - len(failed),
            "failed": len(failed),
            "failures": {row["id"]: row["failures"] for row in failed},
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 1 if failed else 0


def main() -> int:
    # 与 eval_e2e.py 同口径：Windows 终端默认编码未必是 UTF-8，中文答案/汇总
    # 直接 print 可能乱码或 UnicodeEncodeError。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全量）")
    parser.add_argument("--verbose", action="store_true", help="打印每轮问答")
    args = parser.parse_args()
    if args.limit < 0:
        # 负数不是"全量"：cases[:-1] 会静默少跑最后一条，和 eval_e2e 里修过的
        # "--limit -3 被当成全量" 是同一类问题，入口直接拒绝。
        raise SystemExit(f"--limit 必须 >= 0（0 表示全量），收到 {args.limit}")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

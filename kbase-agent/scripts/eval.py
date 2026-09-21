"""检索层评测（离线、零成本）：40+ 条金标里，top-k 是否命中真值来源。

金标集有三类用例（写法见 eval/questions.jsonl）：
  1. 单源 `expected_source`      —— 答案在一篇文档里；
  2. 多跳 `expected_sources`     —— 需要多篇文档共同回答，判定要求**全部命中**；
  3. 应拒答 `should_refuse: true` —— 语料里根本没有答案。
     检索层**不判命中**（没有真值来源），只统计条数；"到底有没有编造"由端到端层判
     （见 scripts/eval_e2e.py 的 refusal_ok）。把这类放进检索层只会污染分母。

用法：python scripts/eval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval.pipeline import RetrievalPipeline  # noqa: E402
from eval.metrics import gold_sources, is_refusal_case  # noqa: E402

EVAL_FILE = Path(__file__).resolve().parents[1] / "eval" / "questions.jsonl"


def run() -> None:
    pipeline = RetrievalPipeline()
    cases = [json.loads(line) for line in EVAL_FILE.read_text(encoding="utf-8").splitlines()]
    results = []
    for case in cases:
        hits = pipeline.retrieve(case["question"])
        predicted = [h["source"] for h in hits]
        gold = gold_sources(case)
        retrieval_hit = bool(gold) and all(any(g in s for s in predicted) for g in gold)
        results.append(
            {
                "id": case["id"],
                "retrieval_hit": retrieval_hit,
                "sources": predicted,
                "expected_source": case.get("expected_source"),
                "expected_sources": case.get("expected_sources"),
                "should_refuse": is_refusal_case(case),
                "difficulty": case.get("difficulty"),
                "note": case.get("note"),
            }
        )
    from eval.metrics import summarize

    summary = summarize(results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # 按难度分档补一层视图：一个总命中率会掩盖"难例全军覆没"。
    by_diff: dict[str, list[dict]] = {}
    for r in results:
        if r["should_refuse"]:
            continue
        by_diff.setdefault(r.get("difficulty") or "未标注", []).append(r)
    if by_diff:
        print("\n== 按难度分档（检索层，k=3）==")
        for diff, rows in sorted(by_diff.items()):
            ok = sum(1 for r in rows if r["retrieval_hit"])
            print(f"  {diff:<10} {ok}/{len(rows)} = {ok / len(rows):.4f}")


if __name__ == "__main__":
    run()

"""检索层评测（离线、零成本）：60 条金标里，top-k 是否命中真值来源。

金标集有三类用例（写法见 eval/questions.jsonl）：
  1. 单源 `expected_source`      —— 答案在一篇文档里；
  2. 多跳 `expected_sources`     —— 需要多篇文档共同回答，判定要求**全部命中**；
  3. 应拒答 `should_refuse: true` —— 语料里根本没有答案。
     检索层**不判命中**（没有真值来源），只统计条数；"到底有没有编造"由端到端层判
     （见 scripts/eval_e2e.py 的 refusal_ok）。把这类放进检索层只会污染分母。

用法：
    python scripts/eval.py                        # 默认：向量 + BM25 走 RRF，top_k=3
    python scripts/eval.py --route vector         # 只跑向量路（消融）
    python scripts/eval.py --route bm25           # 只跑 BM25 路（消融）
    python scripts/eval.py --top-k 5              # 换 top_k
    python scripts/eval.py --sweep-k              # 扫 top_k=1,2,3,5 × 三条路（复现 README 两张表）
    python scripts/eval.py --sweep-k --ks 1,3,5   # 自定义扫描点
    python scripts/eval.py --json out.json        # 额外把 summary 落盘（可选）

为什么要有 --route / --sweep-k：README「检索路消融」与「top-k 扫描」两张表是
"混合检索到底带来了什么"的唯一证据（也是简历上最有价值的数字），此前只能用**改代码**的方式
复现（`retrieve()` 写死走融合，脚本没有任何命令行开关）。现在它们是一条命令，可复核、可进 CI。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval.pipeline import RetrievalPipeline
from eval.metrics import gold_sources, is_refusal_case, summarize

EVAL_FILE = Path(__file__).resolve().parents[1] / "eval" / "questions.jsonl"

ROUTES = ("vector", "bm25", "hybrid")
ROUTE_LABELS = {"vector": "仅向量", "bm25": "仅 BM25", "hybrid": "RRF"}


def load_cases() -> list[dict]:
    return [
        json.loads(line)
        for line in EVAL_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _hit(gold: list[str], predicted: list[str]) -> bool:
    """真值是否全部命中：多跳要求每一篇都在 top-k 里（单源退化为"那一篇命中"）。"""
    return all(any(g in s for s in predicted) for g in gold)


def evaluate(pipeline, cases: list[dict], route: str = "hybrid", top_k: int = 3) -> list[dict]:
    """逐条评测，返回 per-case 结果（含 predicted sources，便于坏例定位）。"""
    results = []
    for case in cases:
        hits = pipeline.retrieve(case["question"], top_k=top_k, route=route)
        predicted = [h["source"] for h in hits]
        gold = gold_sources(case)
        results.append(
            {
                "id": case["id"],
                "route": route,
                "top_k": top_k,
                "retrieval_hit": bool(gold) and _hit(gold, predicted),
                "sources": predicted,
                "expected_source": case.get("expected_source"),
                "expected_sources": case.get("expected_sources"),
                "should_refuse": is_refusal_case(case),
                "difficulty": case.get("difficulty"),
                "note": case.get("note"),
            }
        )
    return results


def _print_by_difficulty(results: list[dict]) -> None:
    """按难度分档补一层视图：一个总命中率会掩盖"难例全军覆没"。"""
    by_diff: dict[str, list[dict]] = {}
    for r in results:
        if r["should_refuse"]:
            continue
        by_diff.setdefault(r.get("difficulty") or "未标注", []).append(r)
    if not by_diff:
        return
    print("\n== 按难度分档（检索层）==")
    for diff, rows in sorted(by_diff.items()):
        ok = sum(1 for r in rows if r["retrieval_hit"])
        print(f"  {diff:<10} {ok}/{len(rows)} = {ok / len(rows):.4f}")


def run(
    route: str = "hybrid",
    top_k: int = 3,
    pipeline=None,
    cases: list[dict] | None = None,
    verbose: bool = True,
    json_path: str | None = None,
) -> dict:
    """跑一遍检索层评测并返回 summary（默认 hybrid + top_k=3，与旧行为完全一致）。"""
    pipeline = pipeline if pipeline is not None else RetrievalPipeline()
    cases = cases if cases is not None else load_cases()
    results = evaluate(pipeline, cases, route=route, top_k=top_k)
    summary = summarize(results)
    summary["route"] = route
    summary["top_k"] = top_k
    if json_path:
        Path(json_path).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if not verbose:
        return summary
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    _print_by_difficulty(results)
    return summary


def sweep(
    pipeline=None,
    cases: list[dict] | None = None,
    ks=(1, 2, 3, 5),
    verbose: bool = True,
) -> list[dict]:
    """三条检索路 × 多个 top_k 的扫描表（README「检索路消融 / top-k 扫描」的来源）。

    `rrf_minus_best_single` 为**负数**时说明该 top_k 下融合反而不如最好的单路——k=1 就是
    这种情况（两路的第一名打平时按插入顺序取胜）。所以本项目的用法是"RRF 做粗召回、
    精度交给可选 rerank"，而不是指望它在 top-1 上更强。
    """
    pipeline = pipeline if pipeline is not None else RetrievalPipeline()
    cases = cases if cases is not None else load_cases()
    scored = [c for c in cases if gold_sources(c)]
    rows: list[dict] = []
    for top_k in ks:
        row: dict = {"top_k": top_k}
        for route in ROUTES:
            results = evaluate(pipeline, cases, route=route, top_k=top_k)
            hits = sum(1 for r in results if r["retrieval_hit"])
            row[route] = {
                "hits": hits,
                "scored": len(scored),
                "hit_rate": round(hits / len(scored), 4) if scored else 0.0,
            }
        best_single = max(row["vector"]["hit_rate"], row["bm25"]["hit_rate"])
        row["rrf_minus_best_single"] = round(row["hybrid"]["hit_rate"] - best_single, 4)
        rows.append(row)
    if verbose:
        print(f"\n== top-k 扫描（scored {len(scored)} 条有真值用例；差值为负表示融合不如单路）==")
        print(f"  {'top_k':<7}{'仅向量':<11}{'仅 BM25':<11}{'RRF':<11}RRF-单路最好")
        for row in rows:
            print(
                f"  {row['top_k']:<7}{row['vector']['hit_rate']:<11.4f}"
                f"{row['bm25']['hit_rate']:<11.4f}{row['hybrid']['hit_rate']:<11.4f}"
                f"{row['rrf_minus_best_single']:+.4f}"
            )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--route",
        choices=ROUTES,
        default="hybrid",
        help="只跑某一路（默认 hybrid = 向量 + BM25 走 RRF 融合）",
    )
    parser.add_argument("--top-k", type=int, default=3, help="取前 k 条来源（默认 3）")
    parser.add_argument(
        "--sweep-k",
        action="store_true",
        help="打印三路 × 多个 top_k 的扫描表（README 消融表的复现入口）",
    )
    parser.add_argument(
        "--ks", type=str, default="1,2,3,5", help="--sweep-k 的扫描点，逗号分隔（默认 1,2,3,5）"
    )
    parser.add_argument(
        "--json", dest="json_path", type=str, default=None, help="把 summary 额外写入该 JSON 文件"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.sweep_k:
        ks = tuple(int(x) for x in args.ks.split(",") if x.strip())
        sweep(ks=ks)
        return 0
    run(route=args.route, top_k=args.top_k, json_path=args.json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""scripts/eval.py 与 scripts/eval_e2e.py 的 CLI 回归。

为什么要有这个文件：README「检索路消融」「top-k 扫描」两张表（简历上最有价值的数字）
此前**在仓库里没有任何实现**——`scripts/eval.py` 连命令行开关都没有，只能改代码；
`--reanalyze` 又会无差别重写全部历史报告。这里把"可复现"与"不误改历史基线"两件事锁住。
"""

import json

import pytest

import scripts.eval as eval_cli
import scripts.eval_e2e as e2e_cli

CASES = [
    {"id": 1, "question": "q1", "expected_source": "a.md"},
    {"id": 2, "question": "q2", "expected_source": "b.md"},
    {"id": 3, "question": "q3", "should_refuse": True, "expected_keywords": []},
]


class _StubStore:
    def __init__(self, hits, log, name):
        self.hits, self.log, self.name = list(hits), log, name

    def search(self, query, top_k=5):
        self.log.append((self.name, query, top_k))
        return [{"source": s, "content": "", "chunk_id": s} for s in self.hits[:top_k]]


class _StubPipeline:
    """模拟三路检索：按 top_k 切片，用来验证"扫描的确实是 k，而不是常量"。

    路由语义与 app/retrieval/pipeline.RetrievalPipeline.retrieve(route=...) 一致：
    vector/bm25 走单路，hybrid 走"两路并集"（真实现是 RRF 融合，这里只关心命中集合）。
    """

    def __init__(self, vector=(), bm25=()):
        self.log = []
        self.vector_store = _StubStore(vector, self.log, "vector")
        self._bm25 = _StubStore(bm25, self.log, "bm25")

    def ensure_ready(self):
        self.log.append("ensure_ready")

    def retrieve(self, query, top_k=3, route="hybrid"):
        self.log.append(("retrieve", route, top_k))
        if route == "vector":
            return self.vector_store.search(query, top_k=top_k)
        if route == "bm25":
            return self._bm25.search(query, top_k=top_k)
        merged = list(dict.fromkeys(self.vector_store.hits + self._bm25.hits))
        return [{"source": s, "content": "", "chunk_id": s} for s in merged[:top_k]]


def test_run_defaults_to_hybrid_and_top_k_3():
    """默认行为必须与旧版一致：融合路 + top_k=3（否则历史数字不可比）。"""
    pipe = _StubPipeline(vector=["b.md", "a.md"], bm25=["a.md", "b.md"])
    summary = eval_cli.run(pipeline=pipe, cases=CASES, verbose=False)

    assert ("retrieve", "hybrid", 3) in pipe.log
    assert summary["cases"] == 3 and summary["scored_cases"] == 2
    assert summary["refusal_cases"] == 1, "应拒答用例不参与命中率，只计数"
    assert summary["topk_hit_rate"] == 1.0
    assert summary["route"] == "hybrid" and summary["top_k"] == 3


def test_run_route_selects_single_route():
    """--route bm25 只能碰 BM25 那一路（消融要真的只跑一路）。"""
    pipe = _StubPipeline(vector=["a.md"], bm25=["b.md"])
    summary = eval_cli.run(pipeline=pipe, cases=CASES, route="bm25", verbose=False)

    assert ("retrieve", "bm25", 3) in pipe.log
    assert summary["topk_hit_rate"] == 0.5, "只命中 #2（b.md），#1 的 a.md 不在 BM25 结果里"


def test_sweep_varies_top_k_and_reports_all_routes():
    """--sweep-k 必须真的换 k：k=1 时 a.md 排第 2 位会漏，k=2 就命中。"""
    pipe = _StubPipeline(vector=["b.md", "a.md"], bm25=["b.md", "a.md"])
    rows = eval_cli.sweep(pipeline=pipe, cases=CASES, ks=(1, 2), verbose=False)

    by_k = {row["top_k"]: row for row in rows}
    assert set(by_k) == {1, 2}
    assert set(by_k[1]) >= {"vector", "bm25", "hybrid", "rrf_minus_best_single"}
    # 每路都按 top_k 切片：k=1 只看到 b.md（命中 #2），k=2 才拿到 a.md（命中 #1）
    assert by_k[1]["vector"]["hit_rate"] == 0.5
    assert by_k[2]["vector"]["hit_rate"] == 1.0
    assert by_k[2]["vector"]["scored"] == 2, "分母只数有真值的用例（应拒答不参与）"


def test_sweep_reports_negative_gap_when_fusion_loses_at_small_k():
    """k=1 时融合可能不如单路——README 里 0.5536(RRF) vs 0.6964(BM25) 就是这个形状。"""
    pipe = _StubPipeline(vector=["x.md"], bm25=["a.md"])
    row = eval_cli.sweep(pipeline=pipe, cases=CASES, ks=(1,), verbose=False)[0]

    assert row["bm25"]["hit_rate"] == 0.5, "BM25 的第一名 a.md 命中 #1"
    assert row["hybrid"]["hit_rate"] == 0.0, "融合的第一名是 x.md，两条都没命中"
    assert row["rrf_minus_best_single"] == -0.5, "差值必须能是负数（如实反映融合不占优）"


def test_eval_cli_rejects_unknown_route():
    with pytest.raises(SystemExit):
        eval_cli.main(["--route", "bogus"])


# —— eval_e2e 的 CLI 校验：三种"静默什么都不做"的情形都改成显式报错 ——


def test_reanalyze_requires_tag_or_all():
    with pytest.raises(SystemExit) as excinfo:
        e2e_cli.main(["--reanalyze"])
    assert excinfo.value.code == 2


def test_compare_requires_tag():
    with pytest.raises(SystemExit) as excinfo:
        e2e_cli.main(["--compare", "corpus-v2"])
    assert excinfo.value.code == 2


def test_tag_with_limit_is_rejected():
    with pytest.raises(SystemExit) as excinfo:
        e2e_cli.main(["--tag", "smoke", "--limit", "5"])
    assert excinfo.value.code == 2


def test_all_requires_reanalyze():
    with pytest.raises(SystemExit) as excinfo:
        e2e_cli.main(["--all"])
    assert excinfo.value.code == 2


def _fake_report(case_id=1, tag="unit-x"):
    return {
        "tag": tag,
        "created_at": "2026-01-01 00:00:00",
        "metrics": {},
        "per_case": [
            {
                "id": case_id,
                "refusal_case": False,
                "answer_nonempty": True,
                "citation_covered": True,
                "answer_cited": True,
                "answer": "最多结转 3 天至次年一季度末",
                "expected_keywords": [],
                "duration_ms": 10,
                "cost_cny": 0.001,
                "total_tokens": 100,
                "tool_calls": 1,
                "llm_calls": 2,
                "refusal_ok": None,
                "must_not_contain": [],
                "keyword_coverage": None,
                "keyword_missed": [],
            }
        ],
    }


def test_reanalyze_tag_only_touches_that_report(tmp_path, monkeypatch):
    """--reanalyze --tag 只能改那一份：别的报告必须逐字节不变（历史基线不可被顺手改掉）。"""
    reports = tmp_path / "eval-reports"
    reports.mkdir()
    target = reports / "unit-x.json"
    other = reports / "keep-me.json"
    target.write_text(json.dumps(_fake_report(tag="unit-x"), ensure_ascii=False), encoding="utf-8")
    other_text = json.dumps(_fake_report(case_id=2, tag="keep-me"), ensure_ascii=False)
    other.write_text(other_text, encoding="utf-8")

    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps({"id": 1, "question": "q", "expected_keywords": ["结转", "3天"]}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(e2e_cli, "REPORT_DIR", reports)
    monkeypatch.setattr(e2e_cli, "EVAL_FILE", questions)

    e2e_cli.reanalyze(tag="unit-x")

    updated = json.loads(target.read_text(encoding="utf-8"))
    assert updated["metrics"]["answer_keyword_coverage"] == 1.0, "关键词覆盖率没有按当前金标重算"
    assert other.read_text(encoding="utf-8") == other_text, "不该动的报告被改写了"


def test_reanalyze_unknown_tag_exits_2(tmp_path, monkeypatch):
    reports = tmp_path / "eval-reports"
    reports.mkdir()
    (reports / "exists.json").write_text(json.dumps(_fake_report()), encoding="utf-8")
    monkeypatch.setattr(e2e_cli, "REPORT_DIR", reports)
    with pytest.raises(SystemExit) as excinfo:
        e2e_cli.reanalyze(tag="nope")
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# 回归：CLI 入口与异步入口**不能同名**
# ---------------------------------------------------------------------------
# 真实缺陷（本轮 ruff F811 暴露出来的）：文件里曾有 `async def main(limit, tag, compare)`
# 与 `def main(argv=None) -> int` 两个同名函数，后者覆盖前者；而 `main()` 内部又写
# `asyncio.run(main(limit=..., tag=..., compare=...))`，于是调到了自己 →
# `TypeError: main() got an unexpected keyword argument 'limit'`。
# 影响面：README 里写的 `python scripts/eval_e2e.py --limit 5` 这类命令**整条是坏的**。
# 之所以一直没被发现：既有测试只覆盖了参数校验的报错分支（那些分支在 asyncio.run 之前
# 就 return 了），正常路径一条都没走。


def test_e2e_cli_limit_path_reaches_async_entry(monkeypatch):
    """--limit 路径必须真的走到异步入口，而不是调回 CLI 自己。"""
    seen = {}

    async def fake_run_eval(limit, tag, compare):
        seen["args"] = (limit, tag, compare)

    monkeypatch.setattr(e2e_cli, "run_eval", fake_run_eval)
    assert e2e_cli.main(["--limit", "3"]) == 0
    assert seen.get("args") == (3, None, None), "CLI 没有把参数传给异步入口"


def test_e2e_cli_full_path_passes_tag_and_compare(monkeypatch):
    """--tag + --compare 这条落在报告路径也要走通。"""
    seen = {}

    async def fake_run_eval(limit, tag, compare):
        seen["args"] = (limit, tag, compare)

    monkeypatch.setattr(e2e_cli, "run_eval", fake_run_eval)
    assert e2e_cli.main(["--tag", "unit-run", "--compare", "baseline"]) == 0
    assert seen.get("args") == (0, "unit-run", "baseline")


def test_e2e_cli_main_and_async_entry_are_distinct():
    """结构断言：main 是同步 CLI，run_eval 是协程——禁止再同名。"""
    import inspect

    assert not inspect.iscoroutinefunction(e2e_cli.main), "main 必须是同步 CLI 入口"
    assert inspect.iscoroutinefunction(e2e_cli.run_eval), "run_eval 必须是协程入口"

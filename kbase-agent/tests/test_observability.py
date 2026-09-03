"""observability / run_traces 存取：纯逻辑，不依赖模型与网络。"""

import asyncio

from app import store
from app.observability import Recorder, Step, estimate_cost


def test_estimate_cost_uses_config():
    # settings 默认输入 2 元 / 输出 8 元（每百万 token）
    cost = estimate_cost(500_000, 250_000)
    assert round(cost, 4) == round(0.5 * 2.0 + 0.25 * 8.0, 4)


def test_recorder_summarize():
    rec = Recorder(question="测试")
    rec.add(Step(node="agent", name="llm", duration_ms=100, prompt_tokens=10, completion_tokens=5, ok=True))
    rec.add(Step(node="tools", name="retrieve_knowledge", duration_ms=50, ok=False, note="超时"))
    s = rec.summarize()
    assert s["llm_calls"] == 1
    assert s["tool_calls"] == 1
    assert s["total_tokens"] == 15
    assert len(s["steps"]) == 2
    assert s["steps"][1]["ok"] is False


def _run(coro):
    return asyncio.run(coro)


def test_trace_roundtrip(tmp_path, monkeypatch):
    db = tmp_path / "checkpoints.sqlite"
    monkeypatch.setattr(store, "DB_PATH", str(db))
    _run(store.init_db())

    rec = Recorder(question="张三还剩几天年假？")
    rec.add(Step(node="agent", name="llm", duration_ms=120, prompt_tokens=300, completion_tokens=150))
    rec.add(Step(node="tools", name="retrieve_knowledge", duration_ms=40))
    _run(store.save_trace("s1", rec.summarize(), "答案正文", ["员工手册_示例.md"]))

    traces = _run(store.list_traces())
    assert len(traces) == 1
    assert traces[0]["cost_cny"] == round(rec.summarize()["cost_cny"], 4)
    assert traces[0]["total_tokens"] == 450

    detail = _run(store.get_trace(traces[0]["id"]))
    assert detail is not None
    assert detail["answer"] == "答案正文"
    assert detail["sources"] == ["员工手册_示例.md"]
    assert detail["steps"][0]["name"] == "llm"

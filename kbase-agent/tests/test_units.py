"""纯逻辑单元测试：不依赖外部模型/网络，装好 base 依赖即可跑。"""

import json

import pytest

from app.guardrails import (
    AgentLimits,
    default_recursion_limit,
    is_duplicate_call,
    truncate_tool_output,
)
from app.retrieval.chunker import fixed_size_chunk, split_documents
from app.retrieval.hybrid import hybrid_search
from app.retrieval.keyword import tokenize
from eval.metrics import summarize


def test_tokenize_keeps_digits_and_cjk_bigrams():
    tokens = tokenize("API 并发超限报 429 怎么处理")
    assert "429" in tokens
    assert "api" in tokens
    assert "并发" in tokens


def test_truncate_keeps_head_tail_and_marker():
    text = "字" * 5000
    out = truncate_tool_output(text, limit=1000)
    assert len(out) < len(text)
    assert "截断" in out


def test_fixed_size_chunk_windows():
    # 文本必须"逐位置可区分"：若用 "一"*1000 或周期为 10 的数字串，偏移量又都是周期整数倍，
    # 任何切法切出的片段都长得一样，断言会恒真、根本测不出 overlap。
    # 这里每 4 个字符一个唯一编号（0000/0001/...），错一格就必然失败。
    text = "".join(f"{i:04d}" for i in range(250))
    assert len(text) == 1000
    chunks = fixed_size_chunk(text, chunk_size=400, overlap=100)
    # step = chunk_size - overlap = 300 → 起点 0/300/600/900
    assert len(chunks) == 4
    assert all(len(c) <= 400 for c in chunks)
    # 相邻块重叠 100 字符，且必须是同一段原文（错位/无重叠都会失败）
    assert chunks[1][:100] == chunks[0][-100:] == text[300:400]
    # 非重叠部分按 step 前进
    assert chunks[1][100:] == text[400:700]
    assert chunks[2][100:] == text[700:1000]


def test_split_documents_ids():
    docs = [{"content": "一二三四五六七八九十", "source": "a.md"}]
    chunks = split_documents(docs, methods=("fixed",))
    assert chunks
    assert chunks[0]["chunk_id"] == "a.md#fixed#0"
    assert chunks[0]["method"] == "fixed"


def test_hybrid_rrf_merges_by_chunk_id():
    vector_hits = [
        {"chunk_id": "a#1", "content": "x", "source": "s"},
        {"chunk_id": "b#1", "content": "y", "source": "s"},
    ]
    keyword_hits = [
        {"chunk_id": "b#1", "content": "y", "source": "s"},
        {"chunk_id": "c#1", "content": "z", "source": "s"},
    ]
    merged = hybrid_search(vector_hits, keyword_hits)
    assert [h["chunk_id"] for h in merged] == ["b#1", "a#1", "c#1"]  # 双路命中排最前


def test_metrics_summarize_keys():
    results = [
        {"retrieval_hit": True, "sources": ["员工手册_示例.md"], "expected_source": "员工手册_示例.md"},
        {"retrieval_hit": False, "sources": ["产品FAQ_示例.md"], "expected_source": "员工手册_示例.md"},
    ]
    summary = summarize(results)
    assert summary["cases"] == 2
    assert summary["topk_hit_rate"] == 0.5
    assert summary["citation_accuracy"] == 0.5


def test_parse_sources_from_content_blocks():
    # MCP adapters 把工具输出包成 [{'type':'text','text':...}]，此前 sources 恒为空
    from langchain_core.messages import AIMessage, ToolMessage

    from app.agent.graph import _parse_sources

    messages = [
        ToolMessage(
            content=[{"type": "text", "text": "【来源：员工手册_示例.md】\n规则正文"}],
            tool_call_id="t1",
            name="retrieve_knowledge",
        ),
        AIMessage(content="回答正文"),
    ]
    assert _parse_sources(messages) == ["员工手册_示例.md"]


def test_final_answer_does_not_fall_back_to_previous_turn():
    """本轮最终 AI 消息为空时，答案不能回退到上一轮（否则等于把旧答案当新答案）。"""
    from langchain_core.messages import AIMessage, HumanMessage

    from app.agent.graph import _final_answer

    previous_turn = [AIMessage(content="上一轮的答案")]
    current_turn = [HumanMessage(content="新问题"), AIMessage(content="")]
    # 本轮切片里没有可用答案 → 如实返回空，而不是上一轮的"上一轮的答案"
    assert _final_answer(current_turn) == ""
    # 对照：若误传完整历史，就会拿到上一轮的答案（这就是修掉的那个回退）
    assert _final_answer(previous_turn + current_turn) == "上一轮的答案"


def test_is_duplicate_call_full_history():
    # A→B→A 式的隔步重复也要判出（全历史判重，而非仅相邻）
    history = [
        ("retrieve_knowledge", '{"question": "a"}'),
        ("query_business_db", '{"question": "b"}'),
    ]
    assert is_duplicate_call(history, ("retrieve_knowledge", '{"question": "a"}')) is True
    assert is_duplicate_call(history, ("query_business_db", '{"question": "c"}')) is False
    assert is_duplicate_call([], ("retrieve_knowledge", "{}")) is False


def test_recursion_limit_derived_from_max_steps():
    # recursion_limit 与 prompt 承诺的 max_steps 同源，避免框架提前掐断
    assert default_recursion_limit() == AgentLimits.max_steps * 2 + 5
    assert default_recursion_limit(25) == 55


# —— 身份解析（安全回归）——
# 背景：早期实现是 `next((r for r in records if r["name"] in question), records[0])`，
# 两个后果都不是"功能缺失"而是"静默给出错误数据"：
#   1) 问题里没写姓名时默认返回 records[0]（张三）——把他人年假/报销金额当成提问人的数据；
#   2) 问题里出现多个姓名时只取列表里第一个命中——同样静默错人。
# 下面三条把"拒绝而不是猜"的行为锁死，避免以后重构时退回去。


def test_query_business_db_refuses_without_employee_name(tmp_path, monkeypatch):
    from app.mcp import servers

    monkeypatch.setattr(servers, "RECORDS_FILE", tmp_path / "records.jsonl")
    out = servers.query_business_db(question="我今年还剩几天年假？")
    assert "无法确定员工身份" in out
    # 关键断言：只给出一句拒绝说明，绝不夹带任何员工的个人数据
    # （提示语里以"张三"举例是允许的，所以这里判"没返回数据体"，而不是判"不含张三"）
    assert "annual_leave_remaining" not in out
    assert "latest_expense" not in out
    assert not out.lstrip().startswith("{")


def test_query_business_db_refuses_on_ambiguous_names(tmp_path, monkeypatch):
    from app.mcp import servers

    monkeypatch.setattr(servers, "RECORDS_FILE", tmp_path / "records.jsonl")
    out = servers.query_business_db(question="张三和李四谁年假多？")
    assert "多个员工姓名" in out
    assert "annual_leave_remaining" not in out


def test_query_business_db_returns_the_named_employee(tmp_path, monkeypatch):
    from app.mcp import servers

    monkeypatch.setattr(servers, "RECORDS_FILE", tmp_path / "records.jsonl")
    out = servers.query_business_db(question="李四还剩几天年假？")
    assert '"employee": "李四"' in out
    # 指名李四就绝不能返回张三的数据
    assert '"employee": "张三"' not in out


def test_business_records_cache_invalidates_on_edit(tmp_path, monkeypatch):
    """缓存必须"改了数据就生效"：这是缓存最容易引入的静默错误（改了却读到旧值）。"""
    from app.mcp import servers

    f = tmp_path / "records.jsonl"
    monkeypatch.setattr(servers, "RECORDS_FILE", f)
    monkeypatch.setattr(servers, "_records_cache", None)
    f.write_text(
        json.dumps({"name": "赵六", "annual_leave_total": 10, "annual_leave_remaining": 2,
                    "overtime_balance_hours": 0, "latest_expense": None}, ensure_ascii=False),
        encoding="utf-8",
    )
    assert '"annual_leave_remaining": 2' in servers.query_business_db(question="赵六年假？")

    # 直接改文件（mtime/size 会变），必须立刻读到新值
    f.write_text(
        json.dumps({"name": "赵六", "annual_leave_total": 10, "annual_leave_remaining": 7,
                    "overtime_balance_hours": 0, "latest_expense": None}, ensure_ascii=False),
        encoding="utf-8",
    )
    out = servers.query_business_db(question="赵六年假？")
    assert '"annual_leave_remaining": 7' in out, f"缓存没有失效，读到旧值：{out}"


def test_business_records_bad_line_is_skipped(tmp_path, monkeypatch, capsys):
    """坏行不能让整把工具失败：应跳过坏行并在 stderr 记明行号，其余记录照常可用。"""
    from app.mcp import servers

    f = tmp_path / "records.jsonl"
    monkeypatch.setattr(servers, "RECORDS_FILE", f)
    monkeypatch.setattr(servers, "_records_cache", None)
    good = json.dumps({"name": "钱七", "annual_leave_total": 10,
                       "annual_leave_remaining": 5, "overtime_balance_hours": 0,
                       "latest_expense": None}, ensure_ascii=False)
    f.write_text(good + "\n{这行不是JSON\n", encoding="utf-8")

    out = servers.query_business_db(question="钱七年假？")
    assert '"employee": "钱七"' in out, f"坏行把好记录一起弄丢了：{out}"
    assert "第 2 行" in capsys.readouterr().err, "坏行必须留下可定位的日志"


# —— 答案内容质量指标 ——


def test_keyword_coverage_ignores_whitespace_and_misses():
    from eval.metrics import keyword_coverage

    # 模型写「3 天」、金标写「3天」，去空白归一化后必须算命中
    cov, missed = keyword_coverage("最多结转 3 天至次年一季度末", ["结转", "3天", "一季度"])
    assert cov == 1.0
    assert missed == []

    # 只泛泛说"可以结转"、没给出关键事实 → 覆盖率下降且能报出缺了哪些
    cov2, missed2 = keyword_coverage("可以结转到明年，具体请咨询 HR", ["结转", "3天", "一季度"])
    assert cov2 == round(1 / 3, 4) or abs(cov2 - 1 / 3) < 1e-9
    assert missed2 == ["3天", "一季度"]

    # 空答案 / 无关键词：都不得抛异常
    assert keyword_coverage("", ["结转"])[0] == 0.0
    assert keyword_coverage("任意答案", [])[0] == 0.0


# —— 金标用例的三类写法与指标口径 ——
# 背景：金标集从 40 条单源扩到含"多跳/冲突/应拒答"三类难例。三类用例的判定口径不同，
# 混在一起算会让指标失真（最典型：把没有真值的应拒答用例算进引用覆盖率的分母，
# 指标凭空下降——那是口径 bug，不是效果变化）。


def test_gold_sources_supports_single_multi_and_refusal():
    from eval.metrics import gold_sources, is_refusal_case

    single = {"id": 1, "expected_source": "a.md"}
    multi = {"id": 2, "expected_sources": ["a.md", "b.md"]}
    refuse = {"id": 3, "should_refuse": True, "expected_keywords": []}
    bare = {"id": 4}   # 没写真值来源，等同于应拒答

    assert gold_sources(single) == ["a.md"]
    assert gold_sources(multi) == ["a.md", "b.md"]   # 多跳优先于单源
    assert gold_sources(refuse) == []
    assert gold_sources(bare) == []

    assert is_refusal_case(refuse) is True
    assert is_refusal_case(bare) is True
    assert is_refusal_case(single) is False
    assert is_refusal_case(multi) is False


def test_multi_hop_requires_every_source():
    """多跳用例只召回一半不算命中——否则"两篇才答得全"的难度就被抹掉了。"""
    from eval.metrics import citation_accuracy

    results = [
        {"expected_sources": ["a.md", "b.md"], "sources": ["a.md", "b.md", "c.md"]},  # 全中
        {"expected_sources": ["a.md", "b.md"], "sources": ["a.md", "c.md"]},          # 缺一篇 → 不算
        {"expected_sources": ["a.md", "b.md"], "sources": ["b.md"]},                  # 只有一篇 → 不算
    ]
    # citation_accuracy 返回原始比值（取整发生在 summarize 里）
    assert abs(citation_accuracy(results) - 1 / 3) < 1e-9


def test_refusal_cases_are_excluded_from_citation_denominator():
    """应拒答用例没有真值来源，不能算进引用覆盖率的分母（否则指标凭空下降）。"""
    from eval.metrics import citation_accuracy, summarize

    results = [
        {"expected_source": "a.md", "sources": ["a.md"], "retrieval_hit": True},
        {"expected_source": "b.md", "sources": ["a.md"], "retrieval_hit": False},
        {"should_refuse": True, "sources": [], "retrieval_hit": False},   # 不参与
    ]
    # 分母是 2（真有真值的用例），不是 3
    assert citation_accuracy(results) == 0.5

    summary = summarize(results)
    assert summary["cases"] == 3
    assert summary["scored_cases"] == 2
    assert summary["refusal_cases"] == 1
    assert summary["topk_hit_rate"] == 0.5      # 同样只在 2 条上算
    assert summary["citation_accuracy"] == 0.5


def test_e2e_metrics_exclude_refusal_cases_from_all_denominators():
    """应拒答用例必须从**每一个**以真值为分母的指标里排除掉，不能只排除一部分。

    首版只把 citation_accuracy 排除了，关键词覆盖率与答案引用率没排 —— 应拒答用例的
    关键词列表是空的（keyword_coverage=0.0）、answer_cited 恒为 False，于是它们被当成
    "失败"算进分母，两个指标凭空掉了约 4 个点。这类口径 bug 很难从数字本身看出来，
    所以用一条测试把"四个分母口径一致"钉死。
    """
    from scripts.eval_e2e import _metrics

    results = [
        {"refusal_case": False, "answer_nonempty": True, "citation_covered": True,
         "answer_cited": True, "keyword_coverage": 1.0, "cost_cny": 0.01,
         "total_tokens": 100},
        {"refusal_case": False, "answer_nonempty": True, "citation_covered": False,
         "answer_cited": False, "keyword_coverage": 0.5, "cost_cny": 0.01,
         "total_tokens": 100},
        # 应拒答：没有真值来源，keyword_coverage 恒为 0、answer_cited 恒为 False
        {"refusal_case": True, "answer_nonempty": True, "citation_covered": False,
         "answer_cited": False, "keyword_coverage": 0.0, "refusal_ok": True,
         "cost_cny": 0.01, "total_tokens": 100},
    ]
    m = _metrics(results)
    assert m["cases"] == 3 and m["scored_cases"] == 2 and m["refusal_cases"] == 1
    # 四个以真值为分母的指标都只看那 2 条
    assert m["answer_rate"] == 1.0
    assert m["citation_accuracy"] == 0.5
    assert m["answer_citation_rate"] == 0.5
    assert m["answer_keyword_coverage"] == 0.75     # (1.0 + 0.5) / 2，不是 /3
    assert m["keyword_full_hit_rate"] == 0.5
    # 应拒答单独算，且全过
    assert m["refusal_accuracy"] == 1.0
    # 成本与 token 是全量口径（它们与真值无关）
    assert m["total_cost_cny"] == 0.03


def test_refusal_ok_flags_fabrication_and_missing_disclaimer():
    """应拒答判定：既要如实说"资料里没有"，又不能把没有的事说成真的。"""
    from scripts.eval_e2e import _refusal_ok

    case = {"must_not_contain": ["行权价", "期权池"]}

    ok, fab = _refusal_ok("资料中没有关于股权激励的内容，建议咨询人力资源部。", case)
    assert ok is True and fab == []

    # 哑火 / 答非所问：没有如实说明 → 不通过
    ok, _ = _refusal_ok("公司提供有竞争力的薪酬体系。", case)
    assert ok is False

    # 说了"资料中没有"却又把不存在的事实说得像真的 → 不通过，并报出命中的特征串
    ok, fab = _refusal_ok("资料中没有明确说明，但行权价一般为每股 10 元。", case)
    assert ok is False and fab == ["行权价"]


def test_refusal_ok_without_must_not_contain_still_checks_disclaimer():
    from scripts.eval_e2e import _refusal_ok

    ok, fab = _refusal_ok("这个资料里没有收录，我无法确认。", {})
    assert ok is True and fab == []


# —— 索引重建：不能残留旧 chunk ——
# 背景（已实测复现）：chunk_id 是 `source#method#idx`，带 method 但不带 chunk_size/overlap。
# 于是换切分法时旧 id 覆盖不到 → 向量库残留旧切片，而 BM25 的 sidecar 是整份覆盖写的、
# 只有新切片，两路口径不一致（实测 recursive→fixed 后库内 23 条 vs sidecar 13 条）。
# 这条测试锁住"index() 先 reset 再 add"的调用顺序。


def test_index_resets_vector_store_before_adding(tmp_path, monkeypatch):
    from app.config import settings
    from app.retrieval.pipeline import RetrievalPipeline

    # sidecar 会写到 settings.chroma_path 下，必须指到临时目录，别污染真实索引
    monkeypatch.setattr(settings, "chroma_path", str(tmp_path / "chroma"), raising=False)

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "d.md").write_text("制度正文。" * 60, encoding="utf-8")

    calls: list[str] = []

    class FakeStore:
        def reset(self) -> None:
            calls.append("reset")

        def add(self, chunks) -> None:
            calls.append(f"add:{len(chunks)}")

        def count(self) -> int:
            return 0

    pipeline = RetrievalPipeline(embedder=object(), vector_store=FakeStore())
    pipeline.index(str(docs), method="fixed")

    assert calls, "index() 应当调用 vector_store"
    assert calls[0] == "reset", f"reset 必须先于 add 执行，实际顺序={calls}"
    assert calls[1].startswith("add:"), f"add 应紧随 reset，实际顺序={calls}"
    assert int(calls[1].split(":")[1]) > 0


# —— sources 语义：答案引用 ≠ 检索命中 ——
# 背景（已实测）：字段 `sources` 原本是"本轮工具返回过的来源"（top_k 命中里的文件名），
# 而前端把它渲染成"来源："标签。复核带来源的 trace，检索口径**从不少于**答案正文实际引用
# （改口径那次 9 条 trace 里 9 条都多 1 条）——等于每轮都替答案多声明一篇引用。
# 现在拆成两个字段：sources=答案引用、retrieved_sources=检索命中。


def test_parse_citations_only_reads_the_answer_text():
    from app.agent.graph import _parse_citations

    answer = "结论如下。\n\n【来源：员工手册_示例.md】\n另见【来源：产品FAQ_示例.md】"
    assert _parse_citations(answer) == ["员工手册_示例.md", "产品FAQ_示例.md"]
    # 去重、保序；没标注就是空
    assert _parse_citations("【来源：a.md】\n【来源：a.md】") == ["a.md"]
    assert _parse_citations("没有标注来源的答案") == []
    assert _parse_citations("") == []


def test_parse_citations_splits_multiple_sources_written_in_one_bracket():
    """`【来源：A、B、C】` 要拆成三项，不能当成一个复合串。

    实测库里就有这种行（`'薪酬与绩效制度_示例.md、绩效系数对照表.xlsx、员工手册_示例.md'`）：
    prompt 只要求"用【来源：文件名】列出引用"，没规定一篇一个括号，模型两种写法都会出现。
    不拆的话 sources 变成单元素复合串，前端渲染成一个标签、按 len(sources) 统计的消费方也数错。
    """
    from app.agent.graph import _parse_citations

    assert _parse_citations("结论。\n\n【来源：a.md、b.md、c.md】") == ["a.md", "b.md", "c.md"]
    assert _parse_citations("【来源：a.md, b.md;c.md/d.md】") == ["a.md", "b.md", "c.md", "d.md"]
    assert _parse_citations("【来源：a.md，b.md】") == ["a.md", "b.md"]
    # 拆分后仍要去重、保序
    assert _parse_citations("【来源：a.md、b.md】\n【来源：a.md】") == ["a.md", "b.md"]


def test_services_init_is_single_flight_even_if_first_caller_is_cancelled(monkeypatch):
    """首个请求在初始化途中被取消时，第二个请求不能把全量建索引再跑一遍。

    背景：`asyncio.to_thread` 的取消只是"放弃 await"，线程会继续跑到底；老实现里
    `async with lock` 会随取消一起退出（services_ok 还没置位），下一个请求重新进临界区、
    is_indexed() 仍为 False → 第二次 index()：两个线程同时 reset/add 同一个 collection、
    同时写同一个 chunks.jsonl.tmp，留下半截 sidecar 与对不上的分块/向量数。
    """
    import asyncio
    import threading
    import time

    from fastapi import FastAPI

    from app import services

    calls = {"index": 0, "ready": 0}
    started = threading.Event()

    class _StubPipeline:
        def is_indexed(self) -> bool:
            return False

        def index(self) -> None:
            calls["index"] += 1
            started.set()
            time.sleep(0.2)

        def ensure_ready(self) -> None:
            calls["ready"] += 1

    monkeypatch.setattr(
        "app.retrieval.pipeline.RetrievalPipeline", lambda *a, **k: _StubPipeline()
    )
    monkeypatch.setattr("app.mcp.servers.set_pipeline", lambda pipeline: None)

    async def _fake_create_runtime():
        return object()

    monkeypatch.setattr("app.agent.graph.create_runtime", _fake_create_runtime)

    app = FastAPI()

    async def scenario():
        first = asyncio.create_task(services.ensure_services(app))
        # 等初始化真正进到 to_thread 里的 index()，再取消第一个调用者
        for _ in range(200):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set(), "初始化没能开始"
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        # 第二个请求：应当等同一个初始化 task，而不是自己重跑一遍
        return await services.ensure_services(app)

    pipeline, runtime = asyncio.run(scenario())
    assert calls["index"] == 1, f"初始化被并发重跑了 {calls['index']} 次"
    assert calls["ready"] == 1
    assert pipeline is not None and runtime is not None


def test_result_separates_answer_citations_from_retrieved_sources():
    """核心回归：正文只引用 1 篇、但工具命中了 2 篇时，sources 必须是 1 篇。"""
    from langchain_core.messages import AIMessage, ToolMessage

    from app.agent.graph import AgentRuntime

    fresh = [
        ToolMessage(
            content=(
                "【来源：员工手册_示例.md】\n结转规则…\n\n---\n\n"
                "【来源：入职转正与离职制度_示例.md】\n离职规则…"
            ),
            tool_call_id="t1",
            name="retrieve_knowledge",
        ),
        AIMessage(content="最多结转 3 天。\n\n【来源：员工手册_示例.md】"),
    ]
    result = AgentRuntime._result(fresh)
    assert result["sources"] == ["员工手册_示例.md"]                      # 答案真的引用了什么
    assert result["retrieved_sources"] == [                              # 检索到底命中了什么
        "员工手册_示例.md",
        "入职转正与离职制度_示例.md",
    ]
    # 旧口径（把命中当引用）会让 sources 变成 2 条——这条断言就是防止回退
    assert len(result["sources"]) < len(result["retrieved_sources"])
    # 本轮确实跑过工具：1 次
    assert result["tool_calls"] == 1


def test_result_reports_zero_tool_calls_when_model_answers_from_context():
    """多轮续聊里模型不调工具直接作答时：tool_calls=0，且引用仍会被解析出来。

    实测存在这种轮次（第一轮查过年假，第二轮问"再重复一遍"就直接答了）。
    消费方必须能区分"本轮没检索"与"检索了没命中"——否则那条【来源：】看起来
    和真的查过库一模一样，而它其实来自历史轮次。
    """
    from langchain_core.messages import AIMessage

    from app.agent.graph import AgentRuntime

    fresh = [AIMessage(content="按《员工手册》最多结转 3 天。\n\n【来源：员工手册_示例.md】")]
    result = AgentRuntime._result(fresh)
    assert result["tool_calls"] == 0
    assert result["retrieved_sources"] == []
    assert result["sources"] == ["员工手册_示例.md"]


# —— 引用可验证性：答案写了【来源：X】，但本轮根本没检索到 X ——
# 背景（真实实测，不是假想）：一次线上对话问"张三还剩几天年假？"，
# retrieve_knowledge(23ms) 与 query_business_db(24ms) **两次调用都成功**、trace 里 ok=True、
# retrieved_sources 里也确实有《员工手册》，但模型在答案里写的引用是
# 【来源：业务系统个人数据查询结果（知识库检索失败,无制度文件可引用）】——括号里那句话说的是
# 一件**没有发生过**的事。修复前 _parse_citations 只是照抄模型写的字，于是这个"假来源"
# 被当成正常引用返回给前端。对一个主打"引用溯源"的项目，这比"引用多了/少了"严重得多。


def test_unverified_citation_from_prompt_injection_style_fabrication_is_flagged():
    """照抄实测到的那条假引用：必须被标成 unverified，而不是当成正常来源。"""
    from langchain_core.messages import AIMessage, ToolMessage

    from app.agent.graph import AgentRuntime

    fresh = [
        ToolMessage(
            content="【来源：员工手册_示例.md】\n年假 10 天…",
            tool_call_id="t1",
            name="retrieve_knowledge",
        ),
        AIMessage(
            content=(
                "张三的年假情况如下：\n- 年假总额：10 天\n- 剩余年假：6 天\n\n"
                "【来源：业务系统个人数据查询结果（知识库检索失败,无制度文件可引用）】"
            )
        ),
    ]
    result = AgentRuntime._result(fresh)
    assert result["sources"] == [
        "业务系统个人数据查询结果（知识库检索失败",
        "无制度文件可引用）",
    ], "答案写了什么要如实保留"
    assert result["retrieved_sources"] == ["员工手册_示例.md"]
    # 关键：这两条都查无此据（本轮检索只返回了《员工手册》）
    assert result["unverified_sources"] == result["sources"]


def test_verified_citations_are_not_flagged():
    """真实引用不能被误判成查无此据——否则这个字段会在正常轮次里刷屏。"""
    from langchain_core.messages import AIMessage, ToolMessage

    from app.agent.graph import AgentRuntime

    fresh = [
        ToolMessage(
            content=(
                "【来源：员工手册_示例.md】\n年假…\n\n---\n\n"
                "【来源：入职转正与离职制度_示例.md】\n离职…"
            ),
            tool_call_id="t1",
            name="retrieve_knowledge",
        ),
        AIMessage(content="最多结转 3 天。\n\n【来源：员工手册_示例.md】"),
    ]
    result = AgentRuntime._result(fresh)
    assert result["sources"] == ["员工手册_示例.md"]
    assert result["unverified_sources"] == []


def test_partially_unverified_citations_split_correctly():
    """一条真、一条假：只有假的那条进 unverified_sources。"""
    from langchain_core.messages import AIMessage, ToolMessage

    from app.agent.graph import AgentRuntime

    fresh = [
        ToolMessage(
            content="【来源：员工手册_示例.md】\n年假…",
            tool_call_id="t1",
            name="retrieve_knowledge",
        ),
        AIMessage(
            content="按制度…\n\n【来源：员工手册_示例.md】【来源：并不存在的制度.md】"
        ),
    ]
    result = AgentRuntime._result(fresh)
    assert result["sources"] == ["员工手册_示例.md", "并不存在的制度.md"]
    assert result["unverified_sources"] == ["并不存在的制度.md"]


def test_no_verdict_when_tools_returned_nothing():
    """工具本轮什么都没返回时不做判定。

    那种情况下所有引用都"无法验证"，但原因是工具挂了（或模型没调工具），不是模型编造。
    混为一谈会让这个字段在故障时刷屏、反而失去信号意义——故障本身由 trace 的 ok=False /
    tool_calls=0 表达。
    """
    from langchain_core.messages import AIMessage

    from app.agent.graph import AgentRuntime

    result = AgentRuntime._result(
        [AIMessage(content="按《员工手册》…\n\n【来源：员工手册_示例.md】")]
    )
    assert result["retrieved_sources"] == []
    assert result["tool_calls"] == 0
    assert result["unverified_sources"] == [], "无工具返回时不该下'查无此据'的结论"
    assert result["sources"] == ["员工手册_示例.md"]


def test_final_answer_skips_ai_message_with_empty_content_blocks():
    """content 是空 block 列表时不能被当成"这条有答案"，否则会返回空串给用户。"""
    from langchain_core.messages import AIMessage

    from app.agent.graph import _final_answer

    assert _final_answer([AIMessage(content=[])]) == ""
    assert _final_answer([AIMessage(content=[{"type": "text", "text": ""}])]) == ""
    # 有真实 block 内容时必须取到
    assert (
        _final_answer([AIMessage(content=[{"type": "text", "text": "真实答案"}])])
        == "真实答案"
    )


# —— 工具调用：并发执行 + 批内判重 ——
# 背景：tools_node 原来是 for-await 串行执行一批 tool_calls，而每个调用各自新起一个
# MCP stdio 子进程，耗时直接相加（实测两个工具 8127ms → 并发 4663ms，省 43%）。
# 改并发时最容易顺手拆掉的护栏就是"批内重复调用判重"（串行实现里第二个相同调用
# 会被第一个刚写进 history 的 key 拦下），所以这条测试同时锁住"并发"与"仍判重"。


def test_tools_node_runs_batch_concurrently_and_still_dedupes():
    import asyncio
    import time

    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langgraph.checkpoint.memory import InMemorySaver

    from app.agent.graph import McpToolSession, _build_graph

    events: list[str] = []

    class SlowTool:
        def __init__(self, name: str):
            self.name = name

        async def ainvoke(self, args):
            events.append(f"start:{self.name}")
            await asyncio.sleep(0.05)
            events.append(f"end:{self.name}")
            return [{"type": "text", "text": f"{self.name} 的结果"}]

    class StubLLM:
        """第一轮吐 3 个 tool_calls（其中第 3 个与第 1 个完全相同），之后给最终答案。"""

        def __init__(self):
            self.calls = 0

        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "t_a", "args": {"q": "1"}, "id": "c1"},
                        {"name": "t_b", "args": {"q": "2"}, "id": "c2"},
                        {"name": "t_a", "args": {"q": "1"}, "id": "c3"},  # 与 c1 完全相同
                    ],
                )
            return AIMessage(content="最终答案")

    graph = _build_graph(
        StubLLM(),
        McpToolSession({}, session_factory=_stub_session_factory([[SlowTool("t_a"), SlowTool("t_b")]])),
        checkpointer=InMemorySaver(),
    )
    start = time.perf_counter()
    out = asyncio.run(
        graph.ainvoke(
            {"messages": [HumanMessage(content="hi")]},
            config={"configurable": {"thread_id": "t"}},
        )
    )
    elapsed = time.perf_counter() - start

    # 1) 并发：两个工具都"开始"了才出现第一个"结束"（串行时事件必然是 start/end 交替）
    assert events[:2] == ["start:t_a", "start:t_b"], f"工具调用没有并发执行：{events}"
    # 2) 串行两次 0.05s×2=0.1s；并发约 0.05s。给足余量，只断言"明显快于串行"
    assert elapsed < 0.09, f"耗时 {elapsed:.3f}s 接近串行(0.1s)，并发可能没生效"

    tool_messages = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 3, "每个 tool_call 都应有一条 ToolMessage（顺序与 id 对应）"
    assert [m.tool_call_id for m in tool_messages] == ["c1", "c2", "c3"]
    # 3) 判重仍然生效：第 3 个（与第 1 个同工具同参数）必须被跳过，且没有真的再跑一遍
    assert "检测到重复工具调用" in tool_messages[2].content
    assert "检测到重复工具调用" not in tool_messages[0].content
    assert events.count("start:t_a") == 1, f"重复调用被真的执行了：{events}"

    # 4) 最终答案仍取到（图能正常收敛）
    from app.agent.graph import _final_answer

    assert _final_answer(out["messages"]) == "最终答案"


# —— 常驻 MCP 工具会话：一次建会话、多次复用 + 会话死了能自愈 ——
# 背景：`MultiServerMCPClient.get_tools()` 返回的工具是**无会话**的，每次 ainvoke 都在内部
# 新起一个 stdio 子进程（实测 retrieve_knowledge 单次 3.6~3.8s，其中子进程启动约 3.0s、
# 子进程内冷加载 BM25/Chroma 约 0.68s）；改成常驻会话后首次 683ms、之后 11~12ms。
# 常驻会话带来的新失败面是"子进程死了会话就失效"，所以下面同时锁住自愈路径。


class _StubTool:
    """最小工具替身：记录调用次数，可选择在第 N 次抛指定异常。"""

    def __init__(self, name: str, fail_with: BaseException | None = None, fail_first: int = 0):
        self.name = name
        self.calls = 0
        self._fail_with = fail_with
        self._fail_first = fail_first

    async def ainvoke(self, args):
        self.calls += 1
        if self._fail_with is not None and self.calls <= self._fail_first:
            raise self._fail_with
        return [{"type": "text", "text": f"{self.name}:{args.get('q', '')}"}]


def _stub_session_factory(tool_batches: list[list]):
    """按调用次序逐批返回工具表的会话工厂（第 N 次建会话用第 N 批，用完后重复最后一批）。"""
    state = {"n": 0}

    async def factory():
        idx = min(state["n"], len(tool_batches) - 1)
        state["n"] += 1
        return tool_batches[idx], object()

    return factory


def test_tool_session_reuses_one_session_for_many_calls():
    """核心回归：N 次取租约只能建 1 次会话（这正是"省掉每次新起子进程"的机制本身）。"""
    import asyncio

    from app.agent.graph import McpToolSession

    built: list[int] = []

    async def factory():
        built.append(1)
        return [_StubTool("t_a")], object()

    async def main():
        session = McpToolSession({}, session_factory=factory)
        leases = [await session.ensure() for _ in range(5)]
        return leases

    leases = asyncio.run(main())
    assert len(built) == 1, f"会话被重建了 {len(built)} 次，常驻复用没有生效"
    assert len({id(lease.session) for lease in leases}) == 1, "多次取租约拿到了不同会话"


def test_tool_session_rebuilds_only_on_transport_failure():
    """会话被判定失效后才重建；且新建的会话必须真的换掉工具表。"""
    import asyncio

    from app.agent.graph import McpToolSession

    built: list[int] = []

    async def factory():
        built.append(1)
        return [_StubTool("t_a")], object()

    async def main():
        session = McpToolSession({}, session_factory=factory)
        first = await session.ensure()
        # 判死 -> 下一代重建
        await session.invalidate(first.generation)
        second = await session.ensure()
        # 重复判死必须是幂等的：第二个并发的失败调用不该把新会话关掉
        await session.invalidate(first.generation)
        third = await session.ensure()
        return first, second, third

    first, second, third = asyncio.run(main())
    assert len(built) == 2, f"会话应恰好重建 1 次，实际建了 {len(built)} 次"
    assert second.generation == third.generation, "并发的重复判死把新会话也判死了"
    assert third.session is second.session, "重复判死后不应再新建会话"


def test_tools_node_returns_tool_call_identity_in_trace():
    """trace 的 tools 步必须能看出"调了哪把工具、什么参数"。

    修复前：所有 tools 步的 node 都是 "tools"，Step 里没有存工具名/参数，
    /api/runs 的 steps 只能靠顺序猜这一步查的是知识库还是业务库。
    """
    import asyncio

    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver

    from app.agent.graph import McpToolSession, _build_graph
    from app.observability import Recorder, reset_recorder, set_recorder

    class OnceLLM:
        def __init__(self):
            self.calls = 0

        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "t_a", "args": {"q": "年假"}, "id": "c1"}],
                )
            return AIMessage(content="答案")

    graph = _build_graph(
        OnceLLM(),
        McpToolSession({}, session_factory=_stub_session_factory([[_StubTool("t_a")]])),
        checkpointer=InMemorySaver(),
    )

    async def main():
        rec = Recorder(question="q")
        token = set_recorder(rec)
        try:
            await graph.ainvoke(
                {"messages": [HumanMessage(content="hi")]},
                config={"configurable": {"thread_id": "t"}},
            )
        finally:
            reset_recorder(token)
        return rec.summarize()

    summary = asyncio.run(main())
    tool_steps = [s for s in summary["steps"] if s["node"] == "tools"]
    assert len(tool_steps) == 1
    assert tool_steps[0]["tool_calls"] == ["t_a"]
    assert tool_steps[0]["tool_args"] == [{"q": "年假"}]


def test_tools_node_self_heals_when_session_dies():
    """子进程死掉后：本次调用重建会话并重试成功，答案里不该出现"调用失败"。"""
    import asyncio

    import anyio
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langgraph.checkpoint.memory import InMemorySaver

    from app.agent.graph import McpToolSession, _build_graph

    class OnceLLM:
        def __init__(self):
            self.calls = 0

        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "t_a", "args": {"q": "x"}, "id": "c1"}],
                )
            return AIMessage(content="最终答案")

    # 第一代会话：调用即抛"管道关闭"（模拟子进程被杀）
    dead = _StubTool("t_a", fail_with=anyio.ClosedResourceError(), fail_first=1)
    # 第二代会话：正常
    healthy = _StubTool("t_a")
    session = McpToolSession(
        {}, session_factory=_stub_session_factory([[dead], [healthy]])
    )
    graph = _build_graph(OnceLLM(), session, checkpointer=InMemorySaver())

    out = asyncio.run(
        graph.ainvoke(
            {"messages": [HumanMessage(content="hi")]},
            config={"configurable": {"thread_id": "t"}},
        )
    )
    tool_messages = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "调用失败" not in tool_messages[0].content, tool_messages[0].content
    assert tool_messages[0].content == "t_a:x", "重试没有拿到新会话的结果"
    assert healthy.calls == 1, "重建后的会话没有被使用"


def test_tools_node_does_not_rebuild_session_on_business_error():
    """工具业务报错（非传输层）不得触发会话重建——否则一次正常报错会被放大成子进程重启。"""
    import asyncio

    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langgraph.checkpoint.memory import InMemorySaver

    from app.agent.graph import McpToolSession, _build_graph

    class OnceLLM:
        def __init__(self):
            self.calls = 0

        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "t_a", "args": {"q": "x"}, "id": "c1"}],
                )
            return AIMessage(content="最终答案")

    builds: list[int] = []
    bad = _StubTool("t_a", fail_with=ValueError("数据库里没有这个人"), fail_first=99)

    async def factory():
        builds.append(1)
        return [bad], object()

    session = McpToolSession({}, session_factory=factory)
    graph = _build_graph(OnceLLM(), session, checkpointer=InMemorySaver())
    out = asyncio.run(
        graph.ainvoke(
            {"messages": [HumanMessage(content="hi")]},
            config={"configurable": {"thread_id": "t"}},
        )
    )
    tool_messages = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert "数据库里没有这个人" in tool_messages[0].content
    assert len(builds) == 1, f"业务报错错误地触发了 {len(builds)} 次会话重建"


def test_tools_node_reports_unavailable_instead_of_raising():
    """会话建不起来时：本轮必须降级成"工具不可用"的 ToolMessage，不能把异常抛成 502。

    场景：依赖缺失 / 索引损坏 / 子进程起不来。此时模型仍能基于历史信息作答，
    所以"如实说明工具不可用"比"整个请求 502"更有价值。
    """
    import asyncio

    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langgraph.checkpoint.memory import InMemorySaver

    from app.agent.graph import McpToolSession, _build_graph

    class OnceLLM:
        def __init__(self):
            self.calls = 0

        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "t_a", "args": {"q": "x"}, "id": "c1"}],
                )
            return AIMessage(content="最终答案")

    async def broken_factory():
        raise RuntimeError("子进程起不来：No module named 'app.mcp.servers'")

    graph = _build_graph(
        OnceLLM(), McpToolSession({}, session_factory=broken_factory), checkpointer=InMemorySaver()
    )
    out = asyncio.run(
        graph.ainvoke(
            {"messages": [HumanMessage(content="hi")]},
            config={"configurable": {"thread_id": "t"}},
        )
    )
    tool_messages = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "工具暂时不可用" in tool_messages[0].content
    assert "No module named" in tool_messages[0].content, "原始原因要带出来，否则没法定位"
    # 图仍然收敛到最终答案（没有被异常打断）
    assert out["messages"][-1].content == "最终答案"


# —— 索引健壮性：损坏的 sidecar 不能让服务永久 503 ——
# 背景（已实测复现）：index_docs.py 被中途 Ctrl-C 会留下半截 chunks.jsonl。
# 原实现 is_indexed() 只看"文件在不在 + 向量数>0"，于是 is_indexed() 恒 True、
# ensure_ready() 恒抛 JSONDecodeError → 每个请求都 503，且**永远不会触发自动重建**。


def _pipeline_with_sidecar(tmp_path, monkeypatch, sidecar_text: str, count: int = 3):
    from app.config import settings
    from app.retrieval.pipeline import RetrievalPipeline

    chroma = tmp_path / "chroma"
    chroma.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "chroma_path", str(chroma), raising=False)
    (chroma / "chunks.jsonl").write_text(sidecar_text, encoding="utf-8")

    class FakeStore:
        def reset(self) -> None:
            pass

        def add(self, chunks) -> None:
            pass

        def count(self) -> int:
            return count

    return RetrievalPipeline(embedder=object(), vector_store=FakeStore())


def test_index_not_considered_ready_when_sidecar_is_truncated(tmp_path, monkeypatch):
    import json

    good = json.dumps({"content": "规则", "source": "a.md", "chunk_id": "a.md#recursive#0"}, ensure_ascii=False)
    torn = good[:-5]  # 模拟被 Ctrl-C 截断的最后一行
    pipeline = _pipeline_with_sidecar(tmp_path, monkeypatch, good + "\n" + torn)

    # 关键：文件存在、向量数也 >0，但内容读不出来 → 必须判为"没有可用索引"
    assert pipeline.is_indexed() is False
    # 于是会走到"索引不存在"这条**可恢复**的分支，而不是把 JSONDecodeError 抛给每个请求
    with pytest.raises(RuntimeError, match="索引不存在"):
        pipeline.ensure_ready(auto_index=False)


def test_index_not_considered_ready_when_sidecar_is_empty(tmp_path, monkeypatch):
    pipeline = _pipeline_with_sidecar(tmp_path, monkeypatch, "")
    assert pipeline.is_indexed() is False
    with pytest.raises(RuntimeError, match="索引不存在"):
        pipeline.ensure_ready(auto_index=False)


def test_ensure_ready_reports_mismatch_instead_of_dividing_by_zero(tmp_path, monkeypatch):
    """sidecar 有分块、向量库却是空的：给出可读报错，而不是 rank_bm25 的 ZeroDivisionError。"""
    import json

    sidecar = json.dumps({"content": "规则", "source": "a.md", "chunk_id": "a.md#recursive#0"}, ensure_ascii=False)
    pipeline = _pipeline_with_sidecar(tmp_path, monkeypatch, sidecar, count=0)
    assert pipeline.is_indexed() is False
    with pytest.raises(RuntimeError, match="向量库为空"):
        pipeline.ensure_ready(auto_index=False)


def test_index_publishes_sidecar_atomically(tmp_path, monkeypatch):
    """sidecar 必须原子发布：不留下 .tmp，且内容可被 JSON 逐行解析。"""
    import json

    from app.config import settings
    from app.retrieval.pipeline import RetrievalPipeline

    chroma = tmp_path / "chroma"
    monkeypatch.setattr(settings, "chroma_path", str(chroma), raising=False)

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "d.md").write_text("制度正文。" * 60, encoding="utf-8")

    class FakeStore:
        def reset(self) -> None:
            pass

        def add(self, chunks) -> None:
            pass

        def count(self) -> int:
            return 0

    pipeline = RetrievalPipeline(embedder=object(), vector_store=FakeStore())
    pipeline.index(str(docs), method="fixed")

    assert (chroma / "chunks.jsonl").is_file()
    assert not list(chroma.glob("*.tmp")), "原子发布不应留下临时文件"
    for line in (chroma / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
        json.loads(line)

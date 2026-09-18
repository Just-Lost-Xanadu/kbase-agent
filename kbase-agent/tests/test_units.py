"""纯逻辑单元测试：不依赖外部模型/网络，装好 base 依赖即可跑。"""

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

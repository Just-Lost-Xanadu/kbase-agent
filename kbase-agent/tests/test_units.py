"""纯逻辑单元测试：不依赖外部模型/网络，装好 base 依赖即可跑。"""

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
# 而前端把它渲染成"来源："标签。复核 9 条 trace，**9/9 都比答案正文实际引用的多 1 条**
# （例：命中 ['员工手册','入职转正与离职制度']，正文只标了【来源：员工手册】）——
# 等于每轮都替答案多声明一篇引用。现在拆成两个字段：sources=答案引用、retrieved_sources=检索命中。


def test_parse_citations_only_reads_the_answer_text():
    from app.agent.graph import _parse_citations

    answer = "结论如下。\n\n【来源：员工手册_示例.md】\n另见【来源：产品FAQ_示例.md】"
    assert _parse_citations(answer) == ["员工手册_示例.md", "产品FAQ_示例.md"]
    # 去重、保序；没标注就是空
    assert _parse_citations("【来源：a.md】\n【来源：a.md】") == ["a.md"]
    assert _parse_citations("没有标注来源的答案") == []
    assert _parse_citations("") == []


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

    from app.agent.graph import _build_graph

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
        StubLLM(), [SlowTool("t_a"), SlowTool("t_b")], checkpointer=InMemorySaver()
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

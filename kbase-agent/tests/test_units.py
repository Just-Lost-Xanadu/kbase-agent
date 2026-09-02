"""纯逻辑单元测试：不依赖外部模型/网络，装好 base 依赖即可跑。"""

from app.guardrails import truncate_tool_output
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
    text = "一" * 1000
    chunks = fixed_size_chunk(text, chunk_size=400, overlap=100)
    assert len(chunks) >= 2
    assert all(len(c) <= 400 for c in chunks)
    # 相邻块有 overlap 交集，避免断点丢信息
    assert chunks[0][-100:] == chunks[1][:100]


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

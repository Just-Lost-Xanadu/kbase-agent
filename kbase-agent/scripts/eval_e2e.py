"""端到端回归评测（Harness）：真实跑 Agent，产出可复现的评测报告与基线 diff。

分层说明（与 scripts/eval.py 的区别）：
- eval.py        ：检索层 Harness（离线、零成本、快），测 top-k 是否命中真值来源。
- eval_e2e.py    ：Agent 层 Harness（真实调 DeepSeek），六个维度：
                   ① answer_rate      —— 回答非空（最弱口径，只保证"没哑火"）
                   ② citation_accuracy—— **检索口径**：真值来源出现在本轮工具返回的来源里
                   ③ answer_citation_rate
                                      —— **答案引用口径**：真值来源出现在答案正文的【来源：X】里
                                         比 ② 更严：检索到了不等于答案标了。两个都留，因为
                                         ②与已入库的基线可比，③才是"引用溯源"的真问题。
                   ④ answer_keyword_coverage / keyword_full_hit_rate
                                      —— **答案内容质量**：金标关键词覆盖率（纯字符串匹配、零 LLM 成本）
                   ⑤ refusal_accuracy —— **应拒答通过率**：语料里没有答案时，有没有如实说不知道、
                                         而不是编出一个具体承诺（弱代理指标，见 _refusal_ok）
                   ⑥ p50/p95 延迟、token、成本
                   是 prompt/模型/工具/语料改动后防劣化的回归手段。

金标集里的三类用例（写法见 eval/questions.jsonl，metrics 侧统一由 eval.metrics.gold_sources 解析）：
   - 单源 `expected_source` ：答案在一篇文档里；
   - 多跳 `expected_sources`：需要多篇共同回答，②③ 都要求**每一篇都命中**；
   - 应拒答 `should_refuse` ：没有真值来源，不进 ②③ 的分母，只走 ⑤。
报告末尾按 difficulty 分档汇总——一个总命中率会掩盖"难例全军覆没"。

用法：
    python scripts/eval_e2e.py --tag baseline              # 全量 40 条，记为 baseline
    python scripts/eval_e2e.py --tag v2-prompt             # 改动后再跑一次
    python scripts/eval_e2e.py --tag v2-prompt --compare baseline   # 打印与基线的回归 diff
    python scripts/eval_e2e.py --limit 5                   # 冒烟（不落报告）
    python scripts/eval_e2e.py --reanalyze                 # 只重算已有报告的指标（不调 API、不要 key）
                                                           # 改了 questions.jsonl 的金标关键词后跑它即可刷新覆盖率
报告产物：docs/eval-reports/{tag}.json（含 per-case 与 answer 原文，可被 --compare 引用）
"""

import argparse
import asyncio
import datetime
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:  # Python < 3.7
    pass

from eval.metrics import gold_sources, hit_rate, is_refusal_case, keyword_coverage  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVAL_FILE = ROOT / "eval" / "questions.jsonl"
REPORT_DIR = ROOT / "docs" / "eval-reports"

# 应拒答判定用的"如实说不知道"标记。系统提示里明确要求"知识库没有的内容如实说
# 『资料中没有』"，所以模型按契约作答时这些词大概率出现。
REFUSAL_MARKERS = (
    "资料中没有", "资料中未", "资料里没有", "资料里未",
    "未收录", "没有收录", "知识库中没有", "暂无相关", "没有找到相关", "没有相关规定",
)


def _refusal_ok(answer: str, case: dict) -> tuple[bool, list[str]]:
    """应拒答用例的判定：既要说清"资料里没有"，又不能把没有的事说成真的。

    返回 (是否通过, 命中的"编造特征"列表)。

    这是一个**弱代理指标**，与 keyword_coverage 同一性质：字符串匹配判不出
    "编得很像但确实是编的"。它真正能抓住的是两类失败：
      ① 哑火 / 答非所问 —— 没有如实说明资料里没有；
      ② 把语料里根本不存在的具体承诺（行权价、贷款额度…）说得像真的 ——
         这类特征串由用例自带 `must_not_contain` 给出。
    因此对外只用"应拒答通过率"这个说法，不宣称它是"幻觉率"。
    """
    text = answer or ""
    said_unknown = any(m in text for m in REFUSAL_MARKERS)
    fabricated = [m for m in (case.get("must_not_contain") or []) if m in text]
    return (said_unknown and not fabricated), fabricated


def _load_cases() -> list[dict]:
    return [json.loads(line) for line in EVAL_FILE.read_text(encoding="utf-8").splitlines()]


def _percentile(values: list[int | float], q: float) -> float:
    """线性插值分位数（与 numpy.percentile 默认口径一致，避免依赖 numpy）。

    n=40、q=0.95 时落在这两份报告的实测区间内；样本极少（n<3）时退化为最大/最小值，
    该口径已写进 README，便于复核分位数是怎么算出来的。
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = q * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * frac)


def _latency(results: list[dict]) -> dict:
    """从已有 per-case 里算延迟分位（纯离线，数据来自 recorder 的 duration_ms）。

    口径说明：p50/p95 是"单次问答端到端耗时"，含 MCP 子进程启动 + 检索 + LLM 往返；
    样本是 40 条单轮金标集，不是线上流量，只用于改动前后的相对比较。
    """
    durations = [r["duration_ms"] for r in results if r.get("duration_ms") is not None]
    return {
        "p50_duration_ms": round(_percentile(durations, 0.5)),
        "p95_duration_ms": round(_percentile(durations, 0.95)),
        "max_duration_ms": round(max(durations)) if durations else 0,
    }


def _metrics(results: list[dict]) -> dict:
    # 应拒答用例没有真值来源，不能算进"回答率/引用覆盖"的分母——它们各有自己的判定
    # （refusal_accuracy）。混在一起会让引用类指标凭空下降，那是口径 bug 不是效果变化。
    scored = [r for r in results if not r.get("refusal_case")]
    refusal = [r for r in results if r.get("refusal_case")]
    metrics = {
        "cases": len(results),
        "scored_cases": len(scored),
        "refusal_cases": len(refusal),
        "answer_rate": round(hit_rate([{"hit": r["answer_nonempty"]} for r in scored]), 4),
        "citation_accuracy": round(hit_rate([{"hit": r["citation_covered"]} for r in scored]), 4),
        "total_cost_cny": round(sum(r["cost_cny"] for r in results), 4),
        "avg_tokens": round(sum(r["total_tokens"] for r in results) / max(len(results), 1)),
        "passed": sum(1 for r in scored if r["answer_nonempty"] and r["citation_covered"]),
    }
    if refusal:
        metrics["refusal_accuracy"] = round(
            hit_rate([{"hit": bool(r.get("refusal_ok"))} for r in refusal]), 4
        )
    # 答案内容质量维度（2026-09 新增）：金标关键词覆盖率。
    # 只在 per_case 里真的有该字段时输出——旧报告没有存 answer，--reanalyze 会自然跳过，
    # 于是 --compare 也不会拿"有"和"没有"去比，避免误报。
    # 必须遍历 `scored` 而不是 `results`：应拒答用例的 expected_keywords 是空的，
    # keyword_coverage 会返回 0.0，把它们算进分母会让覆盖率凭空掉 4 个点——
    # 这是口径 bug，不是质量下降（首版就踩了：0.8917 其实是 4 个 0 拖出来的）。
    coverages = [r["keyword_coverage"] for r in scored if r.get("keyword_coverage") is not None]
    if coverages:
        metrics["answer_keyword_coverage"] = round(sum(coverages) / len(coverages), 4)
        metrics["keyword_full_hit_rate"] = round(
            sum(1 for c in coverages if c >= 1.0) / len(coverages), 4
        )
    # 答案引用口径（比 citation_accuracy 更严）：真值来源要出现在**答案正文**的【来源：X】里。
    # 同样只看 scored：应拒答用例没有真值来源，answer_cited 恒为 False，不能进分母。
    cited_flags = [r["answer_cited"] for r in scored if r.get("answer_cited") is not None]
    if cited_flags:
        metrics["answer_citation_rate"] = round(
            sum(1 for c in cited_flags if c) / len(cited_flags), 4
        )
    metrics.update(_latency(results))
    return metrics


def _compare(tag_a: str, tag_b: str, a: dict, b: dict) -> None:
    """打印 A(tag_b) vs B(tag_a) 的回归 diff。"""
    ma, mb = a["metrics"], b["metrics"]
    print(f"\n==== 回归 diff：{tag_b} vs {tag_a} ====")
    rows = [
        ("cases", "{}", "{}", ""),
        ("answer_rate", "{:.4f}", "{:.4f}", "{:+.4f}"),
        ("citation_accuracy", "{:.4f}", "{:.4f}", "{:+.4f}"),
        ("answer_citation_rate", "{:.4f}", "{:.4f}", "{:+.4f}"),
        ("refusal_accuracy", "{:.4f}", "{:.4f}", "{:+.4f}"),
        ("answer_keyword_coverage", "{:.4f}", "{:.4f}", "{:+.4f}"),
        ("keyword_full_hit_rate", "{:.4f}", "{:.4f}", "{:+.4f}"),
        ("p50_duration_ms", "{:.0f}", "{:.0f}", "{:+.0f}"),
        ("p95_duration_ms", "{:.0f}", "{:.0f}", "{:+.0f}"),
        ("avg_tokens", "{:.0f}", "{:.0f}", "{:+.0f}"),
        ("total_cost_cny", "{:.4f}", "{:.4f}", "{:+.4f}"),
    ]
    for key, fa, fb, fd in rows:
        base_v = ma.get(key)
        new_v = mb.get(key)
        if base_v is None or new_v is None:  # 旧报告可能没有该指标（如 p95）
            continue
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

    # 关键词覆盖率劣化：能抓到 citation 抓不到的那类劣化
    # （来源引用对了，但答案内容缺了关键事实——例如模型变啰嗦却没说出"3 个月"）。
    kw_regressed = []
    for r in b["per_case"]:
        base_r = base_by_id.get(r["id"])
        if not base_r:
            continue
        b_cov = base_r.get("keyword_coverage")
        n_cov = r.get("keyword_coverage")
        if b_cov is None or n_cov is None:
            continue
        if n_cov < b_cov:
            kw_regressed.append((r, b_cov, n_cov))
    if kw_regressed:
        print("\n  [!] 答案关键词覆盖率下降（citation 抓不到的劣化）：")
        for r, b_cov, n_cov in kw_regressed:
            print(f"    #{r['id']}  {b_cov:.2f} -> {n_cov:.2f}  未覆盖={r.get('keyword_missed')}")
    elif any(r.get("keyword_coverage") is not None for r in a["per_case"]):
        # 只有基线本身也有关键词数据时，"无下降"才是有意义的结论；
        # 否则基线是旧口径报告，这里应当保持沉默而不是给出虚假的安心结论。
        print("  （关键词覆盖率无下降）")

    # 答案引用口径劣化：检索命中了、但答案不再标注【来源：X】。
    # 这是 citation_accuracy 抓不到的另一类劣化（它只看检索侧）。
    cite_regressed = []
    for r in b["per_case"]:
        base_r = base_by_id.get(r["id"])
        if not base_r:
            continue
        b_cited = base_r.get("answer_cited")
        n_cited = r.get("answer_cited")
        if b_cited is None or n_cited is None:
            continue
        if b_cited and not n_cited:
            cite_regressed.append(r["id"])
    if cite_regressed:
        print("\n  [!] 答案引用覆盖下降（检索到了但答案没标注来源）：")
        print(f"    用例 {cite_regressed}")


async def main(limit: int, tag: str | None, compare: str | None) -> None:
    from app.agent.graph import create_runtime

    cases = _load_cases()
    if limit and limit > 0:
        cases = cases[:limit]

    runtime = await create_runtime()
    results = []
    try:
        for idx, case in enumerate(cases, start=1):
            # 每条用例一个独立 thread，跑完立刻删（见 AgentRuntime.aclose_thread）：
            # 不显式给 session_id 的话，LangGraph 每轮现造 uuid 线程且没人回收，
            # 40 条就是 40 个永久残留——而这张表正是服务在用的 data/checkpoints.sqlite。
            thread_id = f"eval-{case['id']}-{uuid.uuid4().hex[:8]}"
            r = await _run_case(runtime, case, thread_id)
            await runtime.aclose_thread(thread_id)
            r["id"] = case["id"]
            results.append(r)
            if r["refusal_case"]:
                mark = "OK" if r.get("refusal_ok") else "FAIL"
                extra = f"refused={r.get('refusal_ok')}"
                if r.get("refusal_fabricated"):
                    extra += f" 编造特征={r['refusal_fabricated']}"
            else:
                mark = "OK" if r["citation_covered"] and r["answer_nonempty"] else "FAIL"
                extra = f"tools={r['tool_calls']}"
            print(
                f"[{idx}/{len(cases)}] #{case['id']} {mark} "
                f"[{r.get('difficulty') or '单源'}] {extra} "
                f"cost=¥{r['cost_cny']:.4f} tok={r['total_tokens']} "
                f"kw={r['keyword_coverage']:.2f} {r['error'] or ''}"
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
        # 结尾补一个换行，与 reanalyze() 的写法保持一致：否则跑完评测再跑一次
        # `--reanalyze`（它会重写同一份报告）就会在 git 里留下"只差一个换行"的脏文件，
        # 每次都这样很烦人，也让人分不清"指标真的变了"还是"只是换行"。
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n报告已写入: {out}")

    if compare and tag:
        base_file = REPORT_DIR / f"{compare}.json"
        if not base_file.exists():
            print(f"\n[x] 找不到基线报告 {base_file}，请先 --tag {compare} 跑一次。")
        else:
            base = json.loads(base_file.read_text(encoding="utf-8"))
            _compare(compare, tag, base, report)

    failures = [
        r for r in results
        if (r.get("refusal_ok") is False)
        or (not r.get("refusal_case") and not (r["citation_covered"] and r["answer_nonempty"]))
    ]
    if failures:
        print("\n失败用例（坏例调参入手点）：")
        for r in failures:
            print(f"  #{r['id']}  [{r.get('difficulty') or '单源'}] "
                  f"refusal_ok={r.get('refusal_ok')} citation={r['citation_covered']} "
                  f"answer={r['answer_nonempty']} err={r['error']}")

    # 按难度分档：一个总命中率会掩盖"难例全军覆没"，这一层视图才是加难金标的意义所在。
    by_diff: dict[str, list[dict]] = {}
    for r in results:
        by_diff.setdefault(r.get("difficulty") or "单源(未标注)", []).append(r)
    print("\n== 按难度分档（端到端）==")
    for diff, rows in sorted(by_diff.items()):
        non_ref = [r for r in rows if not r.get("refusal_case")]
        ref = [r for r in rows if r.get("refusal_case")]
        parts = []
        if non_ref:
            ok = sum(1 for r in non_ref if r["citation_covered"] and r["answer_nonempty"])
            parts.append(f"检索引用 {ok}/{len(non_ref)}")
        if ref:
            ok = sum(1 for r in ref if r.get("refusal_ok"))
            parts.append(f"应拒答 {ok}/{len(ref)}")
        print(f"  {diff:<12} {' | '.join(parts)}")


async def _run_case(runtime, case: dict, thread_id: str | None = None) -> dict:
    from app.observability import Recorder, reset_recorder, set_recorder

    question = case["question"]
    expected_source = case.get("expected_source")
    expected_keywords = case.get("expected_keywords") or []
    gold = gold_sources(case)                 # 单源或多跳（多跳要求全部命中）
    refusal_case = is_refusal_case(case)      # 语料里没有答案，判"有没有编造"

    rec = Recorder(question=question)
    token = set_recorder(rec)
    try:
        result = await runtime.ainvoke(
            [{"role": "user", "content": question}], session_id=thread_id
        )
        error = ""
    except Exception as exc:  # noqa: BLE001
        result = {"answer": "", "sources": [], "retrieved_sources": []}
        error = str(exc)[:200]
        rec.error = error
    finally:
        reset_recorder(token)
    summary = rec.summarize()

    answer = result["answer"] or ""
    sources = result["sources"] or []
    retrieved = result.get("retrieved_sources") or []
    # 口径说明：`citation_covered` 仍然是**检索口径**——真值来源文件名必须出现在
    # "本轮工具返回的来源"（`retrieved_sources`）里。保持它不变是为了让已入库的
    # 四份基线报告（baseline / fullrun-verify / baseline-v2 / post-parallel）继续可比：
    # 它们的 per_case 里只有这个字段，改了含义会让历史基线作废、--compare 失去意义。
    # 多跳用例要求**每一篇真值都出现**——只召回一半等于答不全。
    # 模型在回答正文里复述文件名不算有效引用——避免"看过就复述"被误判为覆盖。
    cited = bool(gold) and all(any(g in s for s in retrieved) for g in gold)
    # 答案引用口径：真值来源必须出现在答案正文的【来源：X】里（比上面那条更严）。
    answer_cited = bool(gold) and all(any(g in s for s in sources) for g in gold)
    # 答案内容质量：金标关键词覆盖率（详见 eval/metrics.keyword_coverage）。
    # 同时把 answer 原文与 expected_keywords 一起存进报告，这样改了关键词只需
    # --reanalyze 离线重算（纯字符串匹配），不必重新花钱调 API。
    cov, missed = keyword_coverage(answer, expected_keywords)
    refusal_ok, fabricated = (None, [])
    if refusal_case:
        refusal_ok, fabricated = _refusal_ok(answer, case)
    return {
        "id": 0,
        "difficulty": case.get("difficulty"),
        "refusal_case": refusal_case,
        "answer_nonempty": bool(answer.strip()),
        "citation_covered": cited,
        "answer_cited": answer_cited,
        "refusal_ok": refusal_ok,
        "refusal_fabricated": fabricated,
        "must_not_contain": case.get("must_not_contain") or [],
        "expected_source": expected_source,
        "expected_sources": case.get("expected_sources"),
        "retrieved_sources": retrieved,
        "expected_keywords": expected_keywords,
        "keyword_coverage": round(cov, 4),
        "keyword_missed": missed,
        "answer": answer,
        "tool_calls": summary["tool_calls"],
        "llm_calls": summary["llm_calls"],
        "duration_ms": summary["duration_ms"],
        "cost_cny": summary["cost_cny"],
        "total_tokens": summary["total_tokens"],
        "error": error or None,
    }


def reanalyze() -> None:
    """离线重算已有报告里的指标（不调 API、不需要 key）。

    用途一：给历史报告补齐"当初跑的时候还没有的指标"（例如 p50/p95 延迟）——
    per-case 里已经存着 duration_ms/token/cost 明细，指标口径变了不必重花钱重跑。

    用途二（2026-09 新增）：**重算关键词覆盖率**。per-case 里存了 answer 原文与
    expected_keywords，所以你可以只改 eval/questions.jsonl 里的金标关键词、再跑一次
    --reanalyze，就拿到新的覆盖率——改评测口径的成本是零，不用重新调模型。
    这也是加这一维时坚持"把 answer 存进报告"的原因。

    会把补齐后的 metrics 写回同一份报告（per-case 原样保留，除了重算出的关键词字段）。
    """
    reports = sorted(REPORT_DIR.glob("*.json"))
    if not reports:
        print(f"[x] {REPORT_DIR} 下没有报告可分析。")
        return
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        results = report.get("per_case") or []
        if not results:
            print(f"[skip] {path.name}：无 per_case")
            continue
        # 先按"已存的 answer + 当前金标关键词"重算关键词覆盖率。
        # 旧报告没存 answer，这一步会整段跳过（于是那些报告不会凭空多出覆盖率指标）。
        # 金标关键词以 eval/questions.jsonl 为唯一事实来源：per_case 里存的那份是
        # "当时跑用的口径"，改了口径必须能被覆盖——否则"改关键词后跑 --reanalyze 即可刷新"
        # 这句话就是假的（第一版实现就踩了这个坑：只读报告里的旧关键词，重算结果纹丝不动）。
        gold = {c["id"]: (c.get("expected_keywords") or []) for c in _load_cases()}
        recomputed = 0
        for r in results:
            answer = r.get("answer")
            if answer is None:
                continue   # 旧报告没存答案原文，无法离线重算
            kws = gold.get(r["id"]) or r.get("expected_keywords") or []
            if not kws:
                continue
            cov, missed = keyword_coverage(answer, kws)
            r["expected_keywords"] = kws
            r["keyword_coverage"] = round(cov, 4)
            r["keyword_missed"] = missed
            recomputed += 1
        # 应拒答用例同样支持离线重算：per_case 里存了 answer 原文与 must_not_contain，
        # 所以调整"如实说不知道"的标记或补充"编造特征"后，跑一次 --reanalyze 即可，
        # 不必重新花钱调模型（与关键词覆盖率同一个设计意图）。
        refusal_recomputed = 0
        for r in results:
            if not r.get("refusal_case"):
                continue
            answer = r.get("answer")
            if answer is None:
                continue
            ok, fab = _refusal_ok(answer, {"must_not_contain": r.get("must_not_contain") or []})
            r["refusal_ok"], r["refusal_fabricated"] = ok, fab
            refusal_recomputed += 1
        before = report.get("metrics") or {}
        after = _metrics(results)
        report["metrics"] = after
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        changed = {k: v for k, v in after.items() if before.get(k) != v}
        note = f"，关键词覆盖率重算 {recomputed} 条" if recomputed else "（无 answer 字段，跳过关键词重算）"
        if refusal_recomputed:
            note += f"，应拒答重算 {refusal_recomputed} 条"
        print(f"[ok] {path.name}  metrics 已重算{note}，变化字段：{changed or '（无）'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全量）")
    parser.add_argument("--tag", type=str, default=None, help="本次运行命名（写入 docs/eval-reports/{tag}.json）")
    parser.add_argument("--compare", type=str, default=None, help="与 docs/eval-reports/{tag}.json 的基线做回归 diff")
    parser.add_argument(
        "--reanalyze",
        action="store_true",
        help="只离线重算已有报告的指标（不调 API、不需要 key）",
    )
    args = parser.parse_args()
    if args.reanalyze:
        reanalyze()
    else:
        asyncio.run(main(limit=args.limit, tag=args.tag, compare=args.compare))

import re


def gold_sources(case: dict) -> list[str]:
    """取一条用例的真值来源列表（单一事实来源，别在别处重写这段逻辑）。

    两种写法都支持：
      - `expected_source`  ：单源，与最初 40 条一致的写法；
      - `expected_sources` ：多跳，需要**多篇文档共同**才能回答，判定要求全部命中。
    两者都空表示"应拒答"用例（语料里本来就没有答案）——检索层对这类不判命中，
    由端到端层判"有没有编造"（见 scripts/eval_e2e.py）。
    """
    many = case.get("expected_sources")
    if many:
        return list(many)
    one = case.get("expected_source")
    return [one] if one else []


def is_refusal_case(case: dict) -> bool:
    """应拒答用例：真值来源为空（或显式标了 should_refuse）。"""
    return bool(case.get("should_refuse")) or not gold_sources(case)


def hit_rate(predictions: list[dict], key: str = "hit") -> float:
    total = len(predictions)
    if total == 0:
        return 0.0
    return sum(1 for p in predictions if p.get(key)) / total


def _normalize(text: str) -> str:
    """归一化：去掉全部空白 + 转小写，只比"字符序列是否出现"。

    为什么要去空白：模型常写「3 天」、语料写「3 天」，而金标关键词可能写成「3天」，
    空白/全角半角差异不该算不命中；大小写同理（PIP / pip）。
    """
    return re.sub(r"\s+", "", text or "").lower()


def keyword_coverage(answer: str, keywords: list[str]) -> tuple[float, list[str]]:
    """答案对"金标关键词"的覆盖率（纯字符串匹配，零 LLM 成本、不额外调 API）。

    这是端到端评测里唯一的**答案内容质量维度**：
      - answer_rate 只判"回答非空"；
      - citation_accuracy 只判"真值来源出现在本轮工具 sources 里"（本质是检索侧判定）；
      - 两者都不看答案内容对不对，本函数补上这一维。
    关键词全部取自 data/docs/ 语料原文的关键事实（数字、专有名词、量词），
    命中即说明该条问题的核心事实被答案覆盖到。

    口径边界（本指标不构成"正确率"）：
      - 只判"有没有提到"，**不判表述是否正确**（不识别否定、条件、张冠李戴）；
      - 关键词由人工从语料中挑选、每条 1~3 个，粒度粗；
      - 因此它是 **coverage（覆盖率）而不是 accuracy（正确率）**，用于回归比较与坏例定位，
        不能当作"答案正确率"使用。
    """
    if not keywords:
        return 0.0, []
    norm = _normalize(answer)
    missed = [k for k in keywords if _normalize(k) not in norm]
    return (len(keywords) - len(missed)) / len(keywords), missed


def _all_present(gold: list[str], predicted: list[str]) -> bool:
    """gold 里的每一篇都要能在 predicted 里找到（子串匹配，容忍路径/前后缀差异）。"""
    return all(any(g in p for p in predicted) for g in gold)


def citation_accuracy(results: list[dict]) -> float:
    """真值来源是否**全部**出现在 sources 里；多跳用例缺一篇即不算命中。

    分母只数"真有真值"的用例（`scored_cases`）。原先分母是 `len(results)`，
    在只有单源用例时两者相同，但一旦加入"应拒答"用例（没有真值）就会把分母算大、
    让这个指标凭空下降——那种下降是口径 bug，不是效果变化。
    """
    scored = 0
    correct = 0
    for r in results:
        gold = gold_sources(r)
        if not gold:
            continue
        scored += 1
        if _all_present(gold, r.get("sources", [])):
            correct += 1
    return correct / scored if scored else 0.0


def summarize(results: list[dict]) -> dict:
    """检索层汇总。应拒答用例不参与命中率（它们没有真值来源），单独计数。"""
    scored = [r for r in results if gold_sources(r)]
    refusal = [r for r in results if not gold_sources(r)]
    retrieval_hit = [{"hit": r.get("retrieval_hit", False)} for r in scored]
    summary = {
        "cases": len(results),
        "scored_cases": len(scored),
        "refusal_cases": len(refusal),
        "topk_hit_rate": round(hit_rate(retrieval_hit), 4),
        "citation_accuracy": round(citation_accuracy(results), 4),
        "per_case": results,
    }
    return summary

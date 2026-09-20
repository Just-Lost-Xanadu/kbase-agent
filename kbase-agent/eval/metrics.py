import re


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


def citation_accuracy(results: list[dict]) -> float:
    correct = 0
    for r in results:
        gold_source = r.get("expected_source")
        predicted_sources = r.get("sources", [])
        if gold_source and any(gold_source in s for s in predicted_sources):
            correct += 1
    return correct / len(results) if results else 0.0


def summarize(results: list[dict]) -> dict:
    retrieval_hit = [
        {"hit": r.get("retrieval_hit", False)} for r in results
    ]
    return {
        "cases": len(results),
        "topk_hit_rate": round(hit_rate(retrieval_hit), 4),
        "citation_accuracy": round(citation_accuracy(results), 4),
        "per_case": results,
    }

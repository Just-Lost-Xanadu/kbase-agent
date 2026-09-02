def hit_rate(predictions: list[dict], key: str = "hit") -> float:
    total = len(predictions)
    if total == 0:
        return 0.0
    return sum(1 for p in predictions if p.get(key)) / total


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

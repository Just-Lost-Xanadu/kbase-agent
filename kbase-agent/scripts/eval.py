import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval.pipeline import RetrievalPipeline  # noqa: E402

EVAL_FILE = Path(__file__).resolve().parents[1] / "eval" / "questions.jsonl"


def run() -> None:
    pipeline = RetrievalPipeline()
    cases = [json.loads(line) for line in EVAL_FILE.read_text(encoding="utf-8").splitlines()]
    results = []
    for case in cases:
        hits = pipeline.retrieve(case["question"])
        retrieval_hit = any(case["expected_source"] in h["source"] for h in hits)
        results.append(
            {
                "id": case["id"],
                "retrieval_hit": retrieval_hit,
                "sources": [h["source"] for h in hits],
                "expected_source": case["expected_source"],
            }
        )
    from eval.metrics import summarize

    print(json.dumps(summarize(results), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()

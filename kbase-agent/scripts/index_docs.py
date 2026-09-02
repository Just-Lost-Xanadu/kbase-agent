import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval.pipeline import RetrievalPipeline  # noqa: E402


def main() -> None:
    pipeline = RetrievalPipeline()
    source_dir = Path(__file__).resolve().parents[1] / "data" / "docs"
    pipeline.index(str(source_dir))


if __name__ == "__main__":
    main()

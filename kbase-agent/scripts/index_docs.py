"""全量重建检索索引（解析 -> 切分 -> 向量 + BM25 sidecar）。

用法：
    python scripts/index_docs.py                          # 默认 recursive / chunk_size=800 / overlap=150
    python scripts/index_docs.py --chunk-size 400         # 换切分粒度（建索引参数，必须整库重建）
    python scripts/index_docs.py --method fixed           # 换切分法（recursive <-> fixed）
    python scripts/index_docs.py --overlap 100            # 换重叠窗口

为什么要有这些开关：切分参数（method / chunk_size / overlap）此前写死在函数默认值上，
想换只能改代码。而"切分粒度到底影响多大"是 RAG 最常被追问的问题之一，答案只能来自实测——
现在它是一条命令。想看三条检索路在多个 chunk_size 下的完整对照，用
`python scripts/eval.py --sweep-chunk-size`（它会自己重建索引，跑完自动恢复默认值）。

注意：本脚本是**全量重建**，每次都先清空 collection 再写入，所以换参数直接重跑即可，
不需要手工删 data/chroma/（原因见 app/retrieval/pipeline.py 的 index()）。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval.chunker import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from app.retrieval.pipeline import RetrievalPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--method",
        choices=("recursive", "fixed"),
        default="recursive",
        help="切分法（默认 recursive：按段落/句号等语义边界优先切）",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"单块最大字符数（默认 {DEFAULT_CHUNK_SIZE}）",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help=f"相邻块重叠字符数（默认 {DEFAULT_OVERLAP}）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pipeline = RetrievalPipeline()
    source_dir = Path(__file__).resolve().parents[1] / "data" / "docs"
    pipeline.index(
        str(source_dir),
        method=args.method,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""解析演示：把 data/docs/ 里的 docx/xlsx/pdf 用 loader 抽成文本并打印预览。

docx/xlsx/pdf 与 md 在同一个 data/docs 目录，建索引时一起分块入库；
本脚本只展示 loader 对 office 格式的解析结果，不写索引。

用法：python scripts/demo_office_parse.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval.loader import load_documents  # noqa: E402

DOCS_DIR = Path(__file__).resolve().parents[1] / "data" / "docs"
OFFICE_EXTS = {".docx", ".xlsx", ".pdf"}


def main() -> None:
    if not DOCS_DIR.is_dir():
        raise SystemExit(f"目录不存在: {DOCS_DIR}")
    office_docs = [
        d for d in load_documents(DOCS_DIR) if Path(d["source"]).suffix.lower() in OFFICE_EXTS
    ]
    if not office_docs:
        raise SystemExit("data/docs 下没有 .docx/.xlsx/.pdf 文件。")
    for doc in office_docs:
        text = doc["content"]
        print("=" * 60)
        print(f"来源: {doc['source']}  （解析后 {len(text)} 字符）")
        print("-" * 60)
        print(text[:600])
    print("=" * 60)
    print("以上文本已随 data/docs 一起进入 chunker → embedding → chroma 检索链路。")


if __name__ == "__main__":
    main()

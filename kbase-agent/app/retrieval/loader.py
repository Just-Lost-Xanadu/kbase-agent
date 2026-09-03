from pathlib import Path

SUPPORTED_EXTS = {".md", ".txt"}


def load_documents(source_dir: str | Path) -> list[dict]:
    """遍历 data/docs 下支持的文档，解析为 [{'content': str, 'source': str}]。

    source 用文件名（含扩展名），与 eval/questions.jsonl 的 expected_source 对应。
    """
    root = Path(source_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"文档目录不存在: {root}")
    documents: list[dict] = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        if path.name.startswith("_"):
            continue
        content = path.read_text(encoding="utf-8").strip()
        if content:
            documents.append({"content": content, "source": path.name})
    if not documents:
        raise ValueError(f"目录下没有可解析的文档: {root}")
    return documents

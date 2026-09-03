"""文档解析：把 data/docs 下支持的格式统一抽取为纯文本/Markdown。

支持：.md / .txt（UTF-8 直读）、.docx / .xlsx / .pdf（文本层抽取）。
文件名以 `_` 开头视为说明文件，不参与建索引。
source 用文件名（含扩展名），与 eval/questions.jsonl 的 expected_source 对应。
扫描件/图片类 PDF 没有文本层，需 OCR，列为扩展。
"""

from pathlib import Path

SUPPORTED_EXTS = {".md", ".txt", ".docx", ".xlsx", ".pdf"}


def _extract_plain_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (UnicodeDecodeError, OSError):
        return None


def _extract_docx(path: Path) -> str | None:
    """Word：段落 + 表格（表格转 markdown 竖线行）。"""
    try:
        from docx import Document

        doc = Document(str(path))
    except Exception:
        return None
    parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for idx, table in enumerate(doc.tables, start=1):
        rows = []
        for row in table.rows:
            cells = [c.text.strip().replace("\n", " ") for c in row.cells]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            parts.append(f"[表格 {idx}]\n" + "\n".join(rows))
    return "\n\n".join(parts) or None


def _extract_xlsx(path: Path) -> str | None:
    """Excel：每个工作表转成文本行（| 分隔），丢失样式只留内容。"""
    try:
        import openpyxl

        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception:
        return None
    try:
        parts = []
        for ws in wb.worksheets:
            lines = []
            for row in ws.iter_rows(values_only=True):
                vals = [str(v).strip() for v in row if v is not None]
                if vals:
                    lines.append(" | ".join(vals))
            if lines:
                parts.append(f"[工作表：{ws.title}]\n" + "\n".join(lines))
    finally:
        wb.close()
    return "\n\n".join(parts) or None


def _extract_pdf(path: Path) -> str | None:
    """PDF：逐页抽取文本层，空页跳过；无文本层的扫描件返回内容为空。"""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
    except Exception:
        return None
    pages = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""
        if text:
            pages.append(f"[第 {idx} 页]\n" + text)
    return "\n\n".join(pages) or None


_BINARY_EXTRACTORS = {
    ".docx": _extract_docx,
    ".xlsx": _extract_xlsx,
    ".pdf": _extract_pdf,
}


def _extract(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return _extract_plain_text(path)
    extractor = _BINARY_EXTRACTORS.get(suffix)
    return extractor(path) if extractor else None


def load_documents(source_dir: str | Path) -> list[dict]:
    """遍历 data/docs 下支持的文档，解析为 [{'content': str, 'source': str}]。

    单个文件解析失败会被跳过（不中断整库建索引）。
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
        content = _extract(path)
        if content:
            documents.append({"content": content, "source": path.name})
    if not documents:
        raise ValueError(f"目录下没有可解析的文档: {root}")
    return documents

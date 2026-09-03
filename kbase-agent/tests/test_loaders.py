"""loader 多格式解析测试：docx/xlsx 运行时生成，pdf 用仓库自带 fixture。"""

from pathlib import Path

from app.retrieval.loader import SUPPORTED_EXTS, load_documents

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _build_docx(path: Path) -> None:
    from docx import Document

    doc = Document()
    doc.add_heading("补卡制度", level=1)
    doc.add_paragraph("忘记打卡须在 3 个工作日内申请补卡。")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "城市"
    table.cell(0, 1).text = "限额"
    table.cell(1, 0).text = "上海"
    table.cell(1, 1).text = "500"
    doc.save(str(path))


def _build_xlsx(path: Path) -> None:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "发薪"
    ws.append(["项目", "说明"])
    ws.append(["发薪日", "每月 10 日"])
    wb.save(str(path))


def test_supported_exts():
    assert {".md", ".txt", ".docx", ".xlsx", ".pdf"} <= SUPPORTED_EXTS


def test_load_docx_xlsx_pdf(tmp_path):
    import shutil

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "_说明.md").write_text("不应被索引", encoding="utf-8")
    (docs_dir / "note.md").write_text("纯文本说明", encoding="utf-8")
    _build_docx(docs_dir / "考勤制度.docx")
    _build_xlsx(docs_dir / "薪酬表.xlsx")
    shutil.copy(FIXTURES / "sample.pdf", docs_dir / "sample.pdf")

    docs = load_documents(docs_dir)
    by_source = {d["source"]: d["content"] for d in docs}

    # _ 前缀文件不参与
    assert "_说明.md" not in by_source

    # docx：段落 + 表格都抽取到
    docx_text = by_source["考勤制度.docx"]
    assert "补卡" in docx_text and "500" in docx_text

    # xlsx：sheet 名 + 单元格内容
    xlsx_text = by_source["薪酬表.xlsx"]
    assert "发薪" in xlsx_text and "每月 10 日" in xlsx_text

    # pdf：文本层逐页抽取
    pdf_text = by_source["sample.pdf"]
    assert "salary" in pdf_text


def test_load_pdf_fixture(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    # 用仓库 fixture 文件拷贝到临时语料目录
    import shutil

    shutil.copy(FIXTURES / "sample.pdf", docs_dir / "sample.pdf")
    docs = load_documents(docs_dir)
    assert docs and docs[0]["source"] == "sample.pdf"
    assert "10th" in docs[0]["content"]

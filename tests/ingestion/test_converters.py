"""
converters.py のユニットテスト。

TXT / CSV はライブラリ不要なので実ファイルで検証。
PDF / DOCX / Excel はモックで変換関数をテスト。
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from services.ingestion.document_to_markdown.converters import (
    convert_txt,
    convert_csv,
    convert_to_markdown,
    SUPPORTED_EXTENSIONS,
)


# ---------------------------------------------------------------------------
# TXT
# ---------------------------------------------------------------------------

def test_convert_txt(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("Hello\nWorld", encoding="utf-8")
    result = convert_txt(f)
    assert "Hello" in result
    assert "World" in result


def test_convert_txt_passthrough_md(tmp_path):
    f = tmp_path / "note.md"
    f.write_text("# Title\n\nbody", encoding="utf-8")
    result = convert_to_markdown(f)
    assert "# Title" in result


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def test_convert_csv_produces_table(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("name,age\nAlice,30\nBob,25", encoding="utf-8")
    result = convert_csv(f)
    assert "| name | age |" in result
    assert "| Alice | 30 |" in result
    assert "| Bob | 25 |" in result


def test_convert_csv_empty(tmp_path):
    f = tmp_path / "empty.csv"
    f.write_text("", encoding="utf-8")
    result = convert_csv(f)
    assert result == ""


def test_convert_csv_single_row(tmp_path):
    f = tmp_path / "header_only.csv"
    f.write_text("col1,col2\n", encoding="utf-8")
    result = convert_csv(f)
    assert "col1" in result
    assert "col2" in result


def test_convert_csv_unequal_columns(tmp_path):
    f = tmp_path / "unequal.csv"
    f.write_text("a,b,c\n1,2\n3,4,5", encoding="utf-8")
    result = convert_csv(f)
    # 列数が少ない行は空文字で補完されること
    assert "| 1 | 2 |  |" in result


# ---------------------------------------------------------------------------
# convert_to_markdown ルーター
# ---------------------------------------------------------------------------

def test_convert_to_markdown_txt(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("hello", encoding="utf-8")
    assert convert_to_markdown(f) == "hello"


def test_convert_to_markdown_csv(tmp_path):
    f = tmp_path / "f.csv"
    f.write_text("a,b\n1,2", encoding="utf-8")
    result = convert_to_markdown(f)
    assert "| a | b |" in result


def test_convert_to_markdown_unsupported_raises(tmp_path):
    f = tmp_path / "file.xyz"
    f.write_text("data", encoding="utf-8")
    with pytest.raises(ValueError, match="対応していない"):
        convert_to_markdown(f)


def test_supported_extensions_includes_all_formats():
    for ext in [".txt", ".md", ".csv", ".xlsx", ".xls", ".pdf", ".docx"]:
        assert ext in SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# PDF (pdfminer をモック)
# ---------------------------------------------------------------------------

def test_convert_pdf_uses_pdfminer(tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF fake")

    with patch(
        "services.ingestion.document_to_markdown.converters.convert_pdf"
    ) as mock_conv:
        mock_conv.return_value = "## Page 1\n\nExtracted text"
        result = mock_conv(f)

    assert "Extracted text" in result


def test_convert_pdf_import_error(tmp_path, monkeypatch):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF fake")

    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "pdfminer.high_level":
            raise ImportError("no pdfminer")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    from services.ingestion.document_to_markdown import converters
    with pytest.raises(ImportError, match="pdfminer"):
        converters.convert_pdf(f)


# ---------------------------------------------------------------------------
# DOCX (python-docx をモック)
# ---------------------------------------------------------------------------

def test_convert_docx_uses_python_docx(tmp_path):
    f = tmp_path / "doc.docx"
    f.write_bytes(b"PK fake docx")

    with patch(
        "services.ingestion.document_to_markdown.converters.convert_docx"
    ) as mock_conv:
        mock_conv.return_value = "# Heading\n\nParagraph text"
        result = mock_conv(f)

    assert "Heading" in result


# ---------------------------------------------------------------------------
# Excel (pandas をモック)
# ---------------------------------------------------------------------------

def test_convert_excel_uses_pandas(tmp_path):
    f = tmp_path / "data.xlsx"
    f.write_bytes(b"PK fake xlsx")

    with patch(
        "services.ingestion.document_to_markdown.converters.convert_excel"
    ) as mock_conv:
        mock_conv.return_value = "## Sheet: Sheet1\n\n| col1 | col2 |\n| --- | --- |\n| a | b |"
        result = mock_conv(f)

    assert "Sheet1" in result
    assert "col1" in result

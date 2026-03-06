"""
各フォーマットから Markdown テキストへの変換関数。

すべての変換関数は (file_path: Path) -> str のシグネチャを持つ。

重いライブラリ（pdfminer, python-docx, pandas）はレイジーインポートし、
未インストール時に明確な ImportError を返す。
"""

from __future__ import annotations

import io
import csv as csv_module
from pathlib import Path


# ---------------------------------------------------------------------------
# TXT / Markdown (passthrough)
# ---------------------------------------------------------------------------

def convert_txt(file_path: Path) -> str:
    """テキストファイルをそのまま返す。"""
    return file_path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def convert_csv(file_path: Path) -> str:
    """CSV ファイルを Markdown テーブルに変換する。"""
    text = file_path.read_text(encoding="utf-8", errors="replace")
    reader = csv_module.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return ""
    return _rows_to_markdown_table(rows)


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------

def convert_excel(file_path: Path) -> str:
    """Excel ファイルの各シートを Markdown テーブルに変換する。"""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas と openpyxl が必要です: pip install pandas openpyxl"
        )

    xl = pd.ExcelFile(file_path)
    sections: list[str] = []
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        sections.append(f"## Sheet: {sheet_name}\n")
        sections.append(_dataframe_to_markdown(df))
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def convert_pdf(file_path: Path) -> str:
    """PDF ファイルからテキストを抽出して Markdown に変換する。"""
    try:
        from pdfminer.high_level import extract_text
    except ImportError:
        raise ImportError(
            "pdfminer.six が必要です: pip install pdfminer.six"
        )

    text = extract_text(str(file_path))
    if not text:
        return ""
    # ページ区切りをセクション区切りに変換
    pages = text.split("\x0c")
    sections = []
    for i, page in enumerate(pages, start=1):
        page = page.strip()
        if page:
            sections.append(f"## Page {i}\n\n{page}")
    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------

_HEADING_MAP = {
    "Heading 1": "#",
    "Heading 2": "##",
    "Heading 3": "###",
    "Heading 4": "####",
}


def convert_docx(file_path: Path) -> str:
    """DOCX ファイルを Markdown に変換する（見出し・段落・テーブル対応）。"""
    try:
        from docx import Document
        from docx.oxml.ns import qn
    except ImportError:
        raise ImportError(
            "python-docx が必要です: pip install python-docx"
        )

    doc = Document(str(file_path))
    lines: list[str] = []

    for block in _iter_block_items(doc):
        block_type = type(block).__name__

        if block_type == "Paragraph":
            style = block.style.name if block.style else "Normal"
            text = block.text.strip()
            if not text:
                lines.append("")
                continue
            prefix = _HEADING_MAP.get(style, "")
            lines.append(f"{prefix} {text}" if prefix else text)

        elif block_type == "Table":
            lines.append(_docx_table_to_markdown(block))

    return "\n\n".join(line for line in lines if line is not None)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_CONVERTERS: dict[str, object] = {
    ".txt":  convert_txt,
    ".md":   convert_txt,
    ".csv":  convert_csv,
    ".xlsx": convert_excel,
    ".xls":  convert_excel,
    ".pdf":  convert_pdf,
    ".docx": convert_docx,
}

SUPPORTED_EXTENSIONS = list(_CONVERTERS.keys())


def convert_to_markdown(file_path: Path) -> str:
    """
    ファイル拡張子を判別して対応するコンバーターを呼び出す。

    Parameters
    ----------
    file_path : Path
        変換対象ファイル。

    Returns
    -------
    str
        Markdown テキスト。

    Raises
    ------
    ValueError
        対応していない拡張子の場合。
    """
    suffix = file_path.suffix.lower()
    converter = _CONVERTERS.get(suffix)
    if converter is None:
        raise ValueError(
            f"対応していないフォーマットです: '{suffix}'\n"
            f"対応フォーマット: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return converter(file_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rows_to_markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:]
    sep = ["---"] * len(header)
    lines = [
        "| " + " | ".join(str(c) for c in header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in body:
        # 列数が header と異なる場合は空文字で埋める
        padded = list(row) + [""] * max(0, len(header) - len(row))
        lines.append("| " + " | ".join(str(c) for c in padded[:len(header)]) + " |")
    return "\n".join(lines)


def _dataframe_to_markdown(df) -> str:
    rows = [list(df.columns)] + df.fillna("").astype(str).values.tolist()
    return _rows_to_markdown_table(rows)


def _iter_block_items(doc):
    """Document の body から Paragraph と Table を順番に yield する。"""
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    parent = doc.element.body
    for child in parent.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            yield Table(child, parent)


def _docx_table_to_markdown(table) -> str:
    rows = []
    for row in table.rows:
        rows.append([cell.text.strip() for cell in row.cells])
    return _rows_to_markdown_table(rows)

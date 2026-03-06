"""
IngestionService のユニットテスト。
KnowledgeManager は tmp_path で初期化した実インスタンスを使用。
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from services.ingestion.document_to_markdown.ingestion_service import IngestionService
from services.knowledge.knowledge_manager.knowledge_manager import KnowledgeManager


@pytest.fixture
def km(tmp_path):
    return KnowledgeManager(knowledge_dir=tmp_path / "knowledge")


@pytest.fixture
def service(km):
    return IngestionService(knowledge_manager=km)


# ---------------------------------------------------------------------------
# ingest (単一ファイル)
# ---------------------------------------------------------------------------

def test_ingest_txt_creates_knowledge_unit(service, km, tmp_path):
    f = tmp_path / "rule.txt"
    f.write_text("This is a rule.", encoding="utf-8")

    unit = service.ingest(f)

    assert unit.knowledge_id == "rule"
    assert "This is a rule." in unit.text
    assert "rule" in km.list_ids()


def test_ingest_csv_produces_markdown_table(service, km, tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("col1,col2\nval1,val2", encoding="utf-8")

    unit = service.ingest(f)

    assert "| col1 | col2 |" in unit.text
    assert "| val1 | val2 |" in unit.text


def test_ingest_uses_custom_knowledge_id(service, km, tmp_path):
    f = tmp_path / "original_name.txt"
    f.write_text("content", encoding="utf-8")

    unit = service.ingest(f, knowledge_id="custom_id")

    assert unit.knowledge_id == "custom_id"
    assert "custom_id" in km.list_ids()


def test_ingest_sets_title_and_source(service, km, tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("content", encoding="utf-8")

    unit = service.ingest(f, title="My Doc", source="manual upload")

    assert unit.title == "My Doc"
    assert unit.source == "manual upload"


def test_ingest_uses_filename_as_default_title(service, km, tmp_path):
    f = tmp_path / "report.txt"
    f.write_text("content", encoding="utf-8")

    unit = service.ingest(f)

    assert unit.title == "report.txt"


def test_ingest_file_not_found_raises(service):
    with pytest.raises(FileNotFoundError):
        service.ingest(Path("/nonexistent/file.txt"))


def test_ingest_unsupported_format_raises(service, tmp_path):
    f = tmp_path / "file.xyz"
    f.write_text("data", encoding="utf-8")

    with pytest.raises(ValueError, match="対応していない"):
        service.ingest(f)


def test_ingest_duplicate_raises_by_default(service, tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("first", encoding="utf-8")
    service.ingest(f)

    with pytest.raises(FileExistsError):
        service.ingest(f)


def test_ingest_overwrite(service, km, tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("first version", encoding="utf-8")
    service.ingest(f)

    f.write_text("updated version", encoding="utf-8")
    unit = service.ingest(f, overwrite=True)

    assert "updated version" in unit.text


# ---------------------------------------------------------------------------
# ingest_directory
# ---------------------------------------------------------------------------

def test_ingest_directory_converts_all_supported(service, km, tmp_path):
    src = tmp_path / "docs"
    src.mkdir()
    (src / "a.txt").write_text("text A", encoding="utf-8")
    (src / "b.csv").write_text("x,y\n1,2", encoding="utf-8")
    (src / "ignored.xyz").write_text("skip this", encoding="utf-8")

    units = service.ingest_directory(src)

    assert len(units) == 2
    ids = km.list_ids()
    assert "a" in ids
    assert "b" in ids
    assert "ignored" not in ids


def test_ingest_directory_not_exist_raises(service):
    with pytest.raises(NotADirectoryError):
        service.ingest_directory(Path("/nonexistent/dir"))


def test_ingest_directory_skips_errors(service, km, tmp_path):
    src = tmp_path / "docs"
    src.mkdir()
    (src / "good.txt").write_text("ok", encoding="utf-8")
    bad = src / "bad.pdf"
    bad.write_bytes(b"not a real pdf")

    # convert_pdf が ImportError を返すようモック
    with patch(
        "services.ingestion.document_to_markdown.converters.convert_pdf",
        side_effect=ImportError("pdfminer not installed"),
    ):
        units = service.ingest_directory(src)

    # good.txt だけ成功
    assert len(units) == 1
    assert "good" in km.list_ids()


# ---------------------------------------------------------------------------
# supported_extensions プロパティ
# ---------------------------------------------------------------------------

def test_supported_extensions_not_empty(service):
    assert len(service.supported_extensions) > 0
    assert ".txt" in service.supported_extensions
    assert ".pdf" in service.supported_extensions

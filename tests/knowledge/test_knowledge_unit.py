import json
import pytest
from services.knowledge.knowledge_manager.knowledge_unit import KnowledgeUnit, load_unit


def test_effective_text_returns_summary_when_present():
    unit = KnowledgeUnit(
        knowledge_id="test",
        text="full text",
        summary="short summary",
    )
    assert unit.effective_text == "short summary"


def test_effective_text_falls_back_to_text():
    unit = KnowledgeUnit(knowledge_id="test", text="full text")
    assert unit.effective_text == "full text"


def test_str_includes_title_and_text():
    unit = KnowledgeUnit(knowledge_id="test", text="body", title="My Title")
    result = str(unit)
    assert "My Title" in result
    assert "body" in result


def test_str_no_title_uses_knowledge_id():
    unit = KnowledgeUnit(knowledge_id="rule_01", text="body")
    result = str(unit)
    assert "rule_01" in result


def test_load_unit_reads_markdown(tmp_path):
    md = tmp_path / "rule_01.md"
    md.write_text("This is the rule.", encoding="utf-8")

    unit = load_unit(md)

    assert unit.knowledge_id == "rule_01"
    assert unit.text == "This is the rule."
    assert unit.title == ""
    assert unit.summary == ""


def test_load_unit_reads_metadata(tmp_path):
    md = tmp_path / "rule_01.md"
    md.write_text("content", encoding="utf-8")
    meta = {"knowledge_id": "rule_01", "title": "Rule One", "source": "manual"}
    (tmp_path / "rule_01.json").write_text(json.dumps(meta), encoding="utf-8")

    unit = load_unit(md)

    assert unit.title == "Rule One"
    assert unit.source == "manual"


def test_load_unit_reads_summary_cache(tmp_path):
    md = tmp_path / "rule_01.md"
    md.write_text("full text", encoding="utf-8")
    (tmp_path / "rule_01.summary.txt").write_text("cached summary", encoding="utf-8")

    unit = load_unit(md)

    assert unit.summary == "cached summary"
    assert unit.effective_text == "cached summary"


def test_load_unit_no_optional_files(tmp_path):
    md = tmp_path / "rule_01.md"
    md.write_text("only markdown", encoding="utf-8")

    unit = load_unit(md)

    assert unit.title == ""
    assert unit.summary == ""
    assert unit.effective_text == "only markdown"

import pytest
from services.knowledge.knowledge_manager.knowledge_manager import KnowledgeManager


@pytest.fixture
def km(tmp_path):
    return KnowledgeManager(knowledge_dir=tmp_path)


def test_empty_manager_returns_no_ids(km):
    assert km.list_ids() == []
    assert len(km) == 0


def test_add_creates_md_and_json(km, tmp_path):
    km.add("rule_01", text="Rule content", title="Rule 1", source="manual")

    assert (tmp_path / "rule_01.md").exists()
    assert (tmp_path / "rule_01.json").exists()
    assert len(km) == 1


def test_add_duplicate_raises(km):
    km.add("rule_01", text="first")
    with pytest.raises(FileExistsError):
        km.add("rule_01", text="second")


def test_add_overwrite(km):
    km.add("rule_01", text="first")
    km.add("rule_01", text="updated", overwrite=True)

    unit = km.load("rule_01")
    assert unit.text == "updated"


def test_load_returns_correct_unit(km):
    km.add("rule_01", text="Rule content", title="Rule 1")

    unit = km.load("rule_01")

    assert unit.knowledge_id == "rule_01"
    assert unit.text == "Rule content"
    assert unit.title == "Rule 1"


def test_load_nonexistent_raises(km):
    with pytest.raises(FileNotFoundError):
        km.load("nonexistent")


def test_delete_removes_all_files(km, tmp_path):
    km.add("rule_01", text="content", title="Rule 1")
    km.save_summary("rule_01", "summary text")

    km.delete("rule_01")

    assert not (tmp_path / "rule_01.md").exists()
    assert not (tmp_path / "rule_01.json").exists()
    assert not (tmp_path / "rule_01.summary.txt").exists()
    assert len(km) == 0


def test_list_ids_sorted(km):
    km.add("rule_03", text="c")
    km.add("rule_01", text="a")
    km.add("rule_02", text="b")

    assert km.list_ids() == ["rule_01", "rule_02", "rule_03"]


def test_load_all(km):
    km.add("rule_01", text="a")
    km.add("rule_02", text="b")

    units = km.load_all()

    assert len(units) == 2
    assert {u.knowledge_id for u in units} == {"rule_01", "rule_02"}


def test_sample_returns_all_when_n_exceeds_count(km):
    km.add("rule_01", text="a")
    km.add("rule_02", text="b")

    sampled = km.sample(n=10)

    assert len(sampled) == 2


def test_sample_returns_n_units(km):
    for i in range(5):
        km.add(f"rule_{i:02d}", text=f"content {i}")

    sampled = km.sample(n=3)

    assert len(sampled) == 3


def test_sample_none_returns_all_shuffled(km):
    for i in range(4):
        km.add(f"rule_{i:02d}", text=f"content {i}")

    sampled = km.sample()

    assert len(sampled) == 4


def test_summary_cache_write_and_read(km):
    km.add("rule_01", text="full text")
    km.save_summary("rule_01", "cached summary")

    unit = km.load("rule_01")

    assert unit.summary == "cached summary"
    assert unit.effective_text == "cached summary"


def test_delete_summary(km):
    km.add("rule_01", text="full text")
    km.save_summary("rule_01", "cached summary")
    km.delete_summary("rule_01")

    unit = km.load("rule_01")

    assert unit.summary == ""
    assert unit.effective_text == "full text"


def test_texts_returns_effective_text(km):
    km.add("rule_01", text="full text")
    km.save_summary("rule_01", "short summary")

    texts = km.texts()

    assert len(texts) == 1
    assert "short summary" in texts[0]

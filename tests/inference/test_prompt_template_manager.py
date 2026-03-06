import pytest
from services.inference.llm_inference_service.prompt_template_manager import (
    PromptTemplateManager,
    DEFAULT_TEMPLATE_NAME,
)


@pytest.fixture
def tm(tmp_path):
    return PromptTemplateManager(prompts_dir=tmp_path)


def test_empty_manager_has_no_templates(tm):
    assert tm.list_names() == []
    assert len(tm) == 0


def test_save_and_load(tm):
    tm.save("my_template", "Hello {knowledge} {log} {question}")
    content = tm.load("my_template")
    assert content == "Hello {knowledge} {log} {question}"


def test_list_names_sorted(tm):
    tm.save("zzz", "c")
    tm.save("aaa", "a")
    tm.save("mmm", "b")
    assert tm.list_names() == ["aaa", "mmm", "zzz"]


def test_load_nonexistent_raises(tm):
    with pytest.raises(FileNotFoundError):
        tm.load("nonexistent")


def test_save_overwrites_by_default(tm):
    tm.save("t", "first")
    tm.save("t", "second")
    assert tm.load("t") == "second"


def test_save_no_overwrite_raises(tm):
    tm.save("t", "first")
    with pytest.raises(FileExistsError):
        tm.save("t", "second", overwrite=False)


def test_delete_removes_file(tm, tmp_path):
    tm.save("t", "content")
    tm.delete("t")
    assert "t" not in tm.list_names()
    assert not (tmp_path / "t.txt").exists()


def test_delete_nonexistent_raises(tm):
    with pytest.raises(FileNotFoundError):
        tm.delete("nonexistent")


def test_load_default_uses_default_name(tmp_path):
    tm = PromptTemplateManager(prompts_dir=tmp_path)
    tm.save(DEFAULT_TEMPLATE_NAME, "default content")
    assert tm.load_default() == "default content"


def test_len(tm):
    tm.save("a", "x")
    tm.save("b", "y")
    assert len(tm) == 2


def test_prompts_dir_created_if_not_exists(tmp_path):
    new_dir = tmp_path / "nested" / "prompts"
    tm = PromptTemplateManager(prompts_dir=new_dir)
    assert new_dir.exists()


def test_template_variables_are_placeholders(tm):
    tm.save("t", "K={knowledge} L={log} Q={question}")
    raw = tm.load("t")
    filled = raw.format(knowledge="K", log="L", question="Q")
    assert filled == "K=K L=L Q=Q"

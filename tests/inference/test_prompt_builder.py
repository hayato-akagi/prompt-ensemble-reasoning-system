from services.inference.llm_inference_service.prompt_builder import build_prompt


def test_build_prompt_injects_all_variables():
    prompt = build_prompt(
        knowledge="Rule A: do not share data.",
        log="User shared customer data.",
        question="Is this a violation?",
    )
    assert "Rule A: do not share data." in prompt
    assert "User shared customer data." in prompt
    assert "Is this a violation?" in prompt


def test_build_prompt_uses_custom_template():
    template = "K={knowledge} L={log} Q={question}"
    prompt = build_prompt("knowledge", "log", "question", template=template)
    assert prompt == "K=knowledge L=log Q=question"


def test_build_prompt_default_template_contains_json_format():
    prompt = build_prompt("k", "l", "q")
    assert "answer" in prompt
    assert "confidence" in prompt
    assert "reason" in prompt


def test_build_prompt_empty_knowledge():
    prompt = build_prompt(knowledge="", log="some log", question="q?")
    assert "some log" in prompt
    assert "q?" in prompt

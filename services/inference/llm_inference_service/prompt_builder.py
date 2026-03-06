"""
Builds a prompt string from a template, knowledge text, log, and question.
"""

_DEFAULT_TEMPLATE = """\
You are a reasoning system. Answer only in JSON format.

Use the following knowledge to reason about the log.

[Knowledge]
{knowledge}

[Log]
{log}

[Question]
{question}

Answer strictly in the following JSON format (no extra text):

{{
  "answer": "yes or no",
  "confidence": <float 0.0-1.0>,
  "reason": "<explanation in one sentence>"
}}
"""


def build_prompt(
    knowledge: str,
    log: str,
    question: str,
    template: str | None = None,
) -> str:
    tmpl = template if template is not None else _DEFAULT_TEMPLATE
    return tmpl.format(knowledge=knowledge, log=log, question=question)

"""
Data model for a single knowledge unit.

A knowledge unit consists of:
  - a markdown text file:  <knowledge_id>.md
  - an optional JSON metadata file: <knowledge_id>.json
  - an optional summary cache file: <knowledge_id>.summary.txt
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class KnowledgeUnit:
    knowledge_id: str
    text: str                          # content of the .md file
    title: str = ""
    source: str = ""
    summary: str = ""                  # loaded from summary cache if present
    metadata: dict = field(default_factory=dict)

    @property
    def effective_text(self) -> str:
        """Return summary if available, otherwise full text."""
        return self.summary if self.summary else self.text

    def __str__(self) -> str:
        header = f"[{self.title or self.knowledge_id}]" if (self.title or self.knowledge_id) else ""
        return f"{header}\n{self.effective_text}".strip()


def load_unit(md_path: Path) -> KnowledgeUnit:
    """
    Load a KnowledgeUnit from a .md file.
    Automatically loads .json metadata and .summary.txt if they exist
    alongside the markdown file.
    """
    knowledge_id = md_path.stem
    text = md_path.read_text(encoding="utf-8")

    # Metadata
    meta: dict = {}
    title = ""
    source = ""
    json_path = md_path.with_suffix(".json")
    if json_path.exists():
        import json
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        title = meta.get("title", "")
        source = meta.get("source", "")

    # Summary cache
    summary = ""
    summary_path = md_path.with_name(f"{knowledge_id}.summary.txt")
    if summary_path.exists():
        summary = summary_path.read_text(encoding="utf-8").strip()

    return KnowledgeUnit(
        knowledge_id=knowledge_id,
        text=text,
        title=title,
        source=source,
        summary=summary,
        metadata=meta,
    )

"""
KnowledgeManager: manages knowledge units stored in data/knowledge/.

Responsibilities:
  - List / load / add / delete knowledge units
  - Save and load summary cache
  - Provide random selection of units for ensemble inference
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .knowledge_unit import KnowledgeUnit, load_unit


class KnowledgeManager:
    """
    Manages knowledge units under a given directory.

    Parameters
    ----------
    knowledge_dir : Path | str | None
        Directory containing .md knowledge files.
        Defaults to <project_root>/data/knowledge/.
    """

    def __init__(self, knowledge_dir: Path | str | None = None) -> None:
        if knowledge_dir is None:
            knowledge_dir = Path(__file__).parents[3] / "data" / "knowledge"
        self._dir = Path(knowledge_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def list_ids(self) -> list[str]:
        """Return sorted list of all knowledge_ids (stem of .md files)."""
        return sorted(p.stem for p in self._dir.glob("*.md"))

    def load(self, knowledge_id: str) -> KnowledgeUnit:
        """Load a single knowledge unit by id."""
        md_path = self._dir / f"{knowledge_id}.md"
        if not md_path.exists():
            raise FileNotFoundError(f"Knowledge not found: {knowledge_id}")
        return load_unit(md_path)

    def load_all(self) -> list[KnowledgeUnit]:
        """Load all knowledge units from the directory."""
        units = []
        for md_path in sorted(self._dir.glob("*.md")):
            try:
                units.append(load_unit(md_path))
            except Exception as e:
                print(f"Warning: failed to load {md_path.name}: {e}")
        return units

    def sample(self, n: int | None = None) -> list[KnowledgeUnit]:
        """
        Return a random sample of knowledge units.

        Parameters
        ----------
        n : int | None
            Number of units to sample. If None or >= total, returns all (shuffled).
        """
        all_units = self.load_all()
        if not all_units:
            return []
        if n is None or n >= len(all_units):
            result = all_units.copy()
            random.shuffle(result)
            return result
        return random.sample(all_units, n)

    def texts(self, units: list[KnowledgeUnit] | None = None) -> list[str]:
        """
        Return effective text (summary if cached, otherwise full text) for each unit.
        If units is None, loads all.
        """
        if units is None:
            units = self.load_all()
        return [str(u) for u in units]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(
        self,
        knowledge_id: str,
        text: str,
        title: str = "",
        source: str = "",
        overwrite: bool = False,
    ) -> KnowledgeUnit:
        """
        Create a new knowledge unit.

        Parameters
        ----------
        knowledge_id : str
            Unique identifier (used as filename stem).
        text : str
            Markdown content.
        title : str
            Human-readable title (stored in metadata JSON).
        source : str
            Origin of the knowledge (stored in metadata JSON).
        overwrite : bool
            If False (default), raises if the id already exists.
        """
        md_path = self._dir / f"{knowledge_id}.md"
        if md_path.exists() and not overwrite:
            raise FileExistsError(
                f"Knowledge '{knowledge_id}' already exists. Use overwrite=True to replace."
            )

        md_path.write_text(text, encoding="utf-8")

        meta = {
            "knowledge_id": knowledge_id,
            "title": title,
            "source": source,
            "summary": None,
            "embedding": None,
        }
        json_path = self._dir / f"{knowledge_id}.json"
        json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        return load_unit(md_path)

    def delete(self, knowledge_id: str) -> None:
        """Delete a knowledge unit and its associated metadata/summary files."""
        for suffix in (".md", ".json", ".summary.txt"):
            p = self._dir / f"{knowledge_id}{suffix}"
            if p.exists():
                p.unlink()

    # ------------------------------------------------------------------
    # Summary cache
    # ------------------------------------------------------------------

    def save_summary(self, knowledge_id: str, summary: str) -> None:
        """Write summary cache for the given knowledge unit."""
        self._validate_exists(knowledge_id)
        summary_path = self._dir / f"{knowledge_id}.summary.txt"
        summary_path.write_text(summary.strip(), encoding="utf-8")

    def delete_summary(self, knowledge_id: str) -> None:
        """Remove summary cache."""
        summary_path = self._dir / f"{knowledge_id}.summary.txt"
        if summary_path.exists():
            summary_path.unlink()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_exists(self, knowledge_id: str) -> None:
        if not (self._dir / f"{knowledge_id}.md").exists():
            raise FileNotFoundError(f"Knowledge not found: {knowledge_id}")

    def __len__(self) -> int:
        return len(self.list_ids())

    def __repr__(self) -> str:
        return f"KnowledgeManager(dir={self._dir}, count={len(self)})"

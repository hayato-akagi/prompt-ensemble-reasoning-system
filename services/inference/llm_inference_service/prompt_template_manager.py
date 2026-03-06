"""
PromptTemplateManager: manages prompt templates stored in data/prompts/.

Template files are plain text with {knowledge}, {log}, {question} placeholders.
Each file stem becomes the template name (e.g. default.txt → "default").
"""

from __future__ import annotations

from pathlib import Path


_DEFAULT_PROMPTS_DIR = Path(__file__).parents[3] / "data" / "prompts"
DEFAULT_TEMPLATE_NAME = "default"


class PromptTemplateManager:
    """
    Manages prompt templates under a given directory.

    Parameters
    ----------
    prompts_dir : Path | str | None
        Directory containing .txt template files.
        Defaults to <project_root>/data/prompts/.
    """

    def __init__(self, prompts_dir: Path | str | None = None) -> None:
        self._dir = Path(prompts_dir) if prompts_dir else _DEFAULT_PROMPTS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def list_names(self) -> list[str]:
        """Return sorted list of available template names (without .txt)."""
        return sorted(p.stem for p in self._dir.glob("*.txt"))

    def load(self, name: str) -> str:
        """Load and return the raw template string for the given name."""
        path = self._dir / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"Template '{name}' not found in {self._dir}.\n"
                f"Available: {self.list_names()}"
            )
        return path.read_text(encoding="utf-8")

    def load_default(self) -> str:
        """Load the default template."""
        return self.load(DEFAULT_TEMPLATE_NAME)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, name: str, content: str, overwrite: bool = True) -> None:
        """Save a template. Overwrites by default."""
        path = self._dir / f"{name}.txt"
        if path.exists() and not overwrite:
            raise FileExistsError(f"Template '{name}' already exists.")
        path.write_text(content, encoding="utf-8")

    def delete(self, name: str) -> None:
        """Delete a template file."""
        path = self._dir / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Template '{name}' not found.")
        path.unlink()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.list_names())

    def __repr__(self) -> str:
        return f"PromptTemplateManager(dir={self._dir}, count={len(self)})"

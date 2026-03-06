"""
LLM client wrapping llama-cpp-python.

Model resolution order:
  1. config/inference.json  →  active_model (model ID)
  2. config/models.json     →  filename for that ID
  3. data/models/<filename> →  local GGUF file

To download a model, run:
    python scripts/download_model.py --model-id <id> --set-active
    # or in Docker:
    docker compose run --rm downloader --model-id <id> --set-active
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from llama_cpp import Llama


_ROOT = Path(__file__).parents[3]
_MODELS_DIR = _ROOT / "data" / "models"
_INFERENCE_JSON = _ROOT / "config" / "inference.json"
_MODELS_JSON = _ROOT / "config" / "models.json"


def _load_inference_config() -> dict[str, Any]:
    with open(_INFERENCE_JSON, encoding="utf-8") as f:
        return json.load(f)


def _load_model_registry() -> list[dict[str, Any]]:
    if not _MODELS_JSON.exists():
        return []
    with open(_MODELS_JSON, encoding="utf-8") as f:
        return json.load(f).get("models", [])


def _resolve_model_path(active_model: str) -> Path:
    """
    active_model ID から data/models/ 内の GGUF ファイルパスを解決する。
    モデルが見つからない場合は FileNotFoundError を raise する。
    """
    registry = _load_model_registry()
    meta = next((m for m in registry if m["id"] == active_model), None)

    if meta is None:
        raise FileNotFoundError(
            f"Model '{active_model}' is not registered in config/models.json.\n"
            "Run: python scripts/download_model.py --list"
        )

    local_path = _MODELS_DIR / meta["filename"]
    if not local_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {local_path}\n"
            f"Download it with:\n"
            f"  python scripts/download_model.py --model-id {active_model} --set-active\n"
            f"  # or in Docker:\n"
            f"  docker compose run --rm downloader --model-id {active_model} --set-active"
        )

    return local_path


class LLMClient:
    """Thin wrapper around Llama for single-shot text generation."""

    def __init__(self, model_id: str | None = None) -> None:
        """
        Parameters
        ----------
        model_id : str | None
            使用するモデル ID。None の場合は config/inference.json の active_model を使用。
        """
        cfg = _load_inference_config()
        model_cfg = cfg["model"]
        gen_cfg = cfg["generation"]

        active_model = model_id or cfg["active_model"]
        model_path = _resolve_model_path(active_model)

        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=model_cfg.get("n_ctx", 2048),
            n_gpu_layers=model_cfg.get("n_gpu_layers", 0),
            verbose=model_cfg.get("verbose", False),
        )
        self._temperature = gen_cfg.get("temperature", 0.7)
        self._max_tokens = gen_cfg.get("max_tokens", 512)
        self._top_p = gen_cfg.get("top_p", 0.95)
        self.model_id = active_model

    def generate(self, prompt: str) -> str:
        """Return raw text output from the model."""
        result = self._llm(
            prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            top_p=self._top_p,
        )
        return result["choices"][0]["text"]

    def generate_json(self, prompt: str) -> dict[str, Any]:
        """
        Call generate() and extract a JSON object from the output.
        Raises ValueError if no valid JSON is found.
        """
        raw = self.generate(prompt)
        return _extract_json(raw)


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object found in text."""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in LLM output:\n{text}")

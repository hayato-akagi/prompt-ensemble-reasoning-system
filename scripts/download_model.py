#!/usr/bin/env python3
"""
GGUF モデルダウンローダー

使い方:
    # 登録済みモデルの一覧を表示
    python scripts/download_model.py --list

    # モデル ID を指定してダウンロード（config/models.json から解決）
    python scripts/download_model.py --model-id qwen2.5-7b-instruct-q4_k_m

    # repo_id と filename を直接指定してダウンロード
    python scripts/download_model.py --repo-id Qwen/Qwen2.5-7B-Instruct-GGUF \\
                                     --filename qwen2.5-7b-instruct-q4_k_m.gguf

    # ダウンロード後にアクティブモデルとして設定する
    python scripts/download_model.py --model-id qwen2.5-7b-instruct-q4_k_m --set-active

Docker コンテナから実行する場合:
    docker compose run --rm downloader --list
    docker compose run --rm downloader --model-id qwen2.5-7b-instruct-q4_k_m --set-active
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_ROOT = Path(__file__).parents[1]
_MODELS_DIR = _ROOT / "data" / "models"
_MODELS_JSON = _ROOT / "config" / "models.json"
_INFERENCE_JSON = _ROOT / "config" / "inference.json"


def load_registry() -> list[dict]:
    if not _MODELS_JSON.exists():
        return []
    with open(_MODELS_JSON, encoding="utf-8") as f:
        return json.load(f).get("models", [])


def find_by_id(model_id: str) -> dict | None:
    return next((m for m in load_registry() if m["id"] == model_id), None)


def _filenames_for_model(meta: dict) -> list[str]:
    filenames = meta.get("filenames")
    if isinstance(filenames, list) and filenames:
        return [str(name) for name in filenames]
    return [str(meta["filename"])]


def list_models() -> None:
    registry = load_registry()
    if not registry:
        print("登録されているモデルがありません (config/models.json)")
        return

    downloaded = {p.name for p in _MODELS_DIR.glob("*.gguf")} if _MODELS_DIR.exists() else set()

    print(f"\n{'ID':<40} {'STATUS':<12} DESCRIPTION")
    print("-" * 80)
    for m in registry:
        required_files = _filenames_for_model(m)
        is_downloaded = all(name in downloaded for name in required_files)
        status = "[downloaded]" if is_downloaded else "[not yet]  "
        print(f"{m['id']:<40} {status} {m['description']}")
    print()


def download_by_id(model_id: str) -> Path:
    meta = find_by_id(model_id)
    if meta is None:
        print(f"Error: model '{model_id}' not found in config/models.json", file=sys.stderr)
        print("Use --list to show available models.", file=sys.stderr)
        sys.exit(1)
    return download_many(meta["repo_id"], _filenames_for_model(meta))


def download_many(repo_id: str, filenames: list[str]) -> Path:
    downloaded_paths: list[Path] = []
    for filename in filenames:
        downloaded_paths.append(download(repo_id, filename))
    return downloaded_paths[0]


def download(repo_id: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    local_path = _MODELS_DIR / filename

    if local_path.exists():
        print(f"Already downloaded: {local_path}")
        return local_path

    print(f"Downloading {filename} from {repo_id} ...")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(_MODELS_DIR),
    )
    print(f"Saved to: {downloaded}")
    return Path(downloaded)


def set_active(model_id: str) -> None:
    if not _INFERENCE_JSON.exists():
        print(f"Error: {_INFERENCE_JSON} not found", file=sys.stderr)
        sys.exit(1)

    with open(_INFERENCE_JSON, encoding="utf-8") as f:
        cfg = json.load(f)

    cfg["active_model"] = model_id

    with open(_INFERENCE_JSON, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Active model set to: {model_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GGUF モデルダウンローダー",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list", action="store_true", help="登録済みモデルの一覧を表示")
    parser.add_argument("--model-id", help="モデル ID（config/models.json に登録済みのもの）")
    parser.add_argument("--repo-id", help="HuggingFace の repo_id（直接指定）")
    parser.add_argument("--filename", help="GGUF ファイル名（直接指定）")
    parser.add_argument(
        "--set-active",
        action="store_true",
        help="ダウンロード後に config/inference.json の active_model を更新する",
    )

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if args.model_id:
        download_by_id(args.model_id)
        if args.set_active:
            set_active(args.model_id)
        return

    if args.repo_id and args.filename:
        download(args.repo_id, args.filename)
        return

    parser.print_help()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CLI: Run evaluation baselines (B1–B4) against data/knowledge/eval_labels.json.

Usage examples
--------------
# すべてのベースラインを実行
docker compose run --rm inference python scripts/run_evaluation.py

# B3 と B4 のみ比較
docker compose run --rm inference python scripts/run_evaluation.py --baselines B3 B4

# N-accuracy curve も実行（N=1,3,5,10,20 で B3 スタイルを計測）
docker compose run --rm inference python scripts/run_evaluation.py --n-curve

# N の値を指定
docker compose run --rm inference python scripts/run_evaluation.py --n-curve --n-values 1 3 5 10

# カスタムラベルファイル
docker compose run --rm inference python scripts/run_evaluation.py --labels data/knowledge/eval_labels.json

Results
-------
data/eval_results/<run_id>/
    summary.json          -- ベースライン比較サマリー
    B1_predictions.json   -- B1 の個別予測
    ...
    n_curve.json          -- N-accuracy curve（--n-curve 時のみ）
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from services.evaluation.evaluator import Evaluator
from services.knowledge.knowledge_manager.knowledge_manager import KnowledgeManager


def _print_metrics(label: str, m) -> None:
    print(f"  Accuracy : {m.accuracy:.3f}")
    print(f"  F1 (yes) : {m.f1:.3f}  (P={m.precision:.3f}  R={m.recall:.3f})")
    print(f"  ECE      : {m.ece:.3f}")
    print(f"  Samples  : {m.n}  (pos={m.support_pos}, neg={m.support_neg})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluation baseline runner for RKSSE"
    )
    parser.add_argument(
        "--baselines", nargs="+", default=["B1", "B2", "B3", "B4"],
        choices=["B1", "B2", "B3", "B4"],
        help="Baselines to run (default: all)",
    )
    parser.add_argument(
        "--n-curve", action="store_true",
        help="Also run N-accuracy curve for B3-style inference",
    )
    parser.add_argument(
        "--n-values", nargs="+", type=int, default=[1, 3, 5, 10, 20],
        help="N values for --n-curve (default: 1 3 5 10 20)",
    )
    parser.add_argument(
        "--labels", default="data/knowledge/eval_labels.json",
        help="Path to eval_labels.json (default: data/knowledge/eval_labels.json)",
    )
    args = parser.parse_args()

    labels_path = _ROOT / args.labels
    if not labels_path.exists():
        print(f"[ERROR] {labels_path} が見つかりません。")
        print("  data/knowledge/eval_labels.json に正解ラベルを配置してください。")
        print("  形式: README.md の「正解ラベルの形式」セクションを参照。")
        sys.exit(1)

    km = KnowledgeManager()
    evaluator = Evaluator(labels_path, km)
    items = evaluator.load_items()

    print(f"評価データ: {labels_path}")
    print(f"サンプル数: {len(items)} 件")
    diff_counts = {}
    for item in items:
        diff_counts[item.difficulty] = diff_counts.get(item.difficulty, 0) + 1
    for d, c in sorted(diff_counts.items()):
        print(f"  {d}: {c} 件")
    print()

    # Prepare output directory
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = _ROOT / "data" / "eval_results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run baselines
    all_results = []
    for bkey in args.baselines:
        cfg = Evaluator.BASELINES[bkey]
        print(f"[{bkey}] {cfg['label']} を実行中...")

        def _progress(done: int, total: int, log_id: str, question: str) -> None:
            q_short = question[:35] + "..." if len(question) > 35 else question
            print(f"  [{done + 1:3d}/{total}] {log_id}: {q_short}")

        result = evaluator.run_baseline(bkey, items, callback=_progress)
        all_results.append(result)

        print(f"\n  --- {cfg['label']} ---")
        _print_metrics(cfg["label"], result.metrics)
        print()

        # Save per-baseline predictions
        pred_path = out_dir / f"{bkey}_predictions.json"
        pred_path.write_text(
            json.dumps([r.to_dict() for r in result.records], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Save summary
    summary = {
        "run_id": run_id,
        "labels_path": str(labels_path),
        "n_items": len(items),
        "baselines": [r.to_summary_dict() for r in all_results],
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Print comparison table
    print("=" * 60)
    print(f"{'ベースライン':<30} {'Acc':>6} {'F1':>6} {'ECE':>6}")
    print("-" * 60)
    for r in all_results:
        m = r.metrics
        print(f"{r.label:<30} {m.accuracy:>6.3f} {m.f1:>6.3f} {m.ece:>6.3f}")
    print("=" * 60)

    # N-accuracy curve
    if args.n_curve:
        print(f"\nN-accuracy curve を実行中 (N={args.n_values}) ...")
        n_results = evaluator.run_n_curve(items, args.n_values)
        n_path = out_dir / "n_curve.json"
        n_path.write_text(
            json.dumps(n_results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n{'N':>4} {'Accuracy':>10} {'F1':>8} {'ECE':>8}")
        print("-" * 34)
        for r in n_results:
            print(f"{r['n']:>4} {r['accuracy']:>10.3f} {r['f1']:>8.3f} {r['ece']:>8.3f}")

    print(f"\n結果を保存しました: {out_dir}")


if __name__ == "__main__":
    main()

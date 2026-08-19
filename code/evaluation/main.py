"""
Evaluation pipeline — runs the router on dataset/sample_messages.csv (labeled dev set)
and compares predictions against the expected action/message_type/reason-signal/evidence.

Usage:
    python code/evaluation/main.py [--dataset-dir dataset]
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from dotenv import load_dotenv
from main import REPO_ROOT, Dataset, KeyPool, build_context, analyze_message, format_row

load_dotenv()

EXACT_COLUMNS = ["action", "message_type"]


def load_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evidence_overlap(pred: str, gold: str) -> bool:
    """Loose match: at least one cited ID overlaps, or both say none."""
    pred_set = {p.strip() for p in (pred or "").split(";") if p.strip() and p.strip() != "none"}
    gold_set = {g.strip() for g in (gold or "").split(";") if g.strip() and g.strip() != "none"}
    if not pred_set and not gold_set:
        return True
    return bool(pred_set & gold_set)


def run_predictions(sample_path: Path, dataset_dir: Path) -> dict:
    pool = KeyPool()
    ds = Dataset(dataset_dir)
    rows = load_csv(sample_path)

    predictions = {}
    for i, row in enumerate(rows, 1):
        ctx = build_context(ds, row, pool)
        result = analyze_message(pool, row, ctx)
        formatted = format_row(row["message_id"], result)
        predictions[row["message_id"]] = formatted
        print(f"  [{i}/{len(rows)}] {row['message_id']} -> pred={result['action']}/{result['message_type']} "
              f"gold={row['action']}/{row['message_type']}", flush=True)
    return predictions


def evaluate(predictions: dict, gold_rows: list) -> dict:
    n = len(gold_rows)
    metrics = {col: {"correct": 0, "total": n} for col in EXACT_COLUMNS + ["evidence_overlap"]}
    mismatches = []

    for gold in gold_rows:
        pred = predictions.get(gold["message_id"], {})
        row_mismatches = []
        for col in EXACT_COLUMNS:
            p = str(pred.get(col, "")).strip().lower()
            g = str(gold.get(col, "")).strip().lower()
            match = p == g
            metrics[col]["correct"] += int(match)
            if not match:
                row_mismatches.append(f"{col}: pred={p!r} gold={g!r}")

        ev_ok = evidence_overlap(pred.get("evidence_message_ids", ""), gold.get("evidence_message_ids", ""))
        metrics["evidence_overlap"]["correct"] += int(ev_ok)
        if not ev_ok:
            row_mismatches.append(f"evidence: pred={pred.get('evidence_message_ids')!r} gold={gold.get('evidence_message_ids')!r}")

        if row_mismatches:
            mismatches.append((gold["message_id"], row_mismatches, pred.get("reason", ""), gold.get("reason", "")))

    return {"metrics": metrics, "mismatches": mismatches, "n": n}


def print_report(result: dict):
    metrics = result["metrics"]
    n = result["n"]
    print(f"\n=== Evaluation Report ({n} rows) ===")
    total_correct = total_items = 0
    for col, m in metrics.items():
        acc = m["correct"] / m["total"] if m["total"] else 0
        print(f"{col:<20} {m['correct']:>3}/{m['total']:<3}  {acc:.0%}")
        total_correct += m["correct"]
        total_items += m["total"]
    print(f"{'Overall':<20} {total_correct:>3}/{total_items:<3}  {total_correct/total_items:.0%}")

    print("\n=== Mismatches ===")
    for msg_id, issues, pred_reason, gold_reason in result["mismatches"]:
        print(f"  {msg_id}:")
        for issue in issues:
            print(f"    {issue}")
        print(f"    pred reason: {pred_reason}")
        print(f"    gold reason: {gold_reason}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset")
    args = parser.parse_args()

    dataset_dir = REPO_ROOT / args.dataset_dir
    sample_path = dataset_dir / "sample_messages.csv"

    print(f"Running evaluation on {sample_path}...")
    predictions = run_predictions(sample_path, dataset_dir)
    gold_rows = load_csv(sample_path)
    result = evaluate(predictions, gold_rows)
    print_report(result)


if __name__ == "__main__":
    main()

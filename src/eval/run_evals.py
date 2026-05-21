"""Run the classifier against the golden set and report metrics.

Outputs:
- Overall accuracy
- Per-class precision, recall, F1
- Confusion matrix (printed to terminal)
- Per-example errors (for inspection)
- JSON results file with run metadata (model, prompt hash, timestamp)

Run from project root:
    python -m src.eval.run_evals
    python -m src.eval.run_evals --model claude-sonnet-4-6  # compare models
    python -m src.eval.run_evals --show-errors  # print misclassifications
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.classifier import CLASSIFIER_MODEL, PROMPT_PATH, classify_email
from src.models import Category, Email

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"


def _load_golden_set(path: Path) -> list[tuple[Email, str]]:
    """Load labeled emails. Returns (Email, true_label) pairs."""
    if not path.exists():
        raise FileNotFoundError(
            f"No golden set at {path}. Run `python -m src.eval.label_helper` first."
        )

    pairs: list[tuple[Email, str]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            email = Email(
                id=rec["email_id"],
                thread_id=rec.get("thread_id", rec["email_id"]),
                sender=rec["sender"],
                sender_email=rec["sender_email"],
                sender_name=None,
                recipient="",  # not stored in golden set; classifier doesn't use it
                subject=rec["subject"],
                body=rec["body"],
                received_at=datetime.fromisoformat(rec["received_at"]),
                labels=rec.get("labels", []),
            )
            pairs.append((email, rec["label"]))
    return pairs


def _compute_metrics(
    predictions: list[tuple[str, str]],
) -> dict[str, Any]:
    """Compute accuracy + per-class precision/recall/F1.

    predictions: list of (true_label, predicted_label)
    """
    n = len(predictions)
    if n == 0:
        return {"accuracy": 0.0, "per_class": {}, "total": 0}

    correct = sum(1 for t, p in predictions if t == p)
    accuracy = correct / n

    # Per-class counts: TP, FP, FN per category
    classes = sorted({c.value for c in Category})
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    support: dict[str, int] = defaultdict(int)

    for true_label, pred_label in predictions:
        support[true_label] += 1
        if true_label == pred_label:
            tp[true_label] += 1
        else:
            fp[pred_label] += 1
            fn[true_label] += 1

    per_class: dict[str, dict[str, float]] = {}
    for c in classes:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[c] = {
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
            "support": support[c],
        }

    return {
        "accuracy": round(accuracy, 3),
        "correct": correct,
        "total": n,
        "per_class": per_class,
    }


def _print_confusion_matrix(
    predictions: list[tuple[str, str]],
) -> None:
    """ASCII confusion matrix. Rows = true, columns = predicted."""
    classes = sorted({c.value for c in Category})
    # Short labels for column headers — full names are too wide
    short = {c: c[:6] for c in classes}

    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for true_label, pred_label in predictions:
        matrix[true_label][pred_label] += 1

    print("\nConfusion matrix (rows=true, cols=predicted):")
    header = "  " + " ".join(f"{short[c]:>7}" for c in classes)
    print(header)
    for true_class in classes:
        row = " ".join(f"{matrix[true_class][c]:>7d}" for c in classes)
        # Mark the diagonal
        print(f"  {row}   ← {true_class}")


def _print_per_class(metrics: dict[str, Any]) -> None:
    print("\nPer-class metrics:")
    print(f"  {'category':<25} {'P':>6} {'R':>6} {'F1':>6} {'n':>5}")
    print(f"  {'-' * 25} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 5}")
    for c, m in metrics["per_class"].items():
        print(
            f"  {c:<25} {m['precision']:>6.3f} {m['recall']:>6.3f} "
            f"{m['f1']:>6.3f} {m['support']:>5d}"
        )


def _print_errors(
    predictions: list[tuple[str, str, Email, str]],
    limit: int = 10,
) -> None:
    """Print misclassifications for inspection."""
    errors = [p for p in predictions if p[0] != p[1]]
    if not errors:
        print("\nNo errors. (Either you're done or your eval set is too easy.)")
        return

    print(f"\nMisclassifications ({len(errors)} total, showing up to {limit}):")
    for true_label, pred_label, email, reasoning in errors[:limit]:
        print(f"\n  TRUE: {true_label}  PRED: {pred_label}")
        print(f"  Subject: {email.subject[:70]}")
        print(f"  From:    {email.sender_email}")
        print(f"  Why model chose {pred_label}: {reasoning}")


def _prompt_hash() -> str:
    """Short hash of the current prompt — pin results to a prompt version."""
    return hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest()[:8]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run classifier evals.")
    parser.add_argument("--model", default=CLASSIFIER_MODEL)
    parser.add_argument("--show-errors", action="store_true")
    parser.add_argument(
        "--golden-set", type=Path, default=GOLDEN_SET_PATH
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    pairs = _load_golden_set(args.golden_set)
    print(f"\nLoaded {len(pairs)} labeled emails from {args.golden_set.name}")
    print(f"Model: {args.model}")
    print(f"Prompt hash: {_prompt_hash()}\n")

    predictions: list[tuple[str, str]] = []
    error_records: list[tuple[str, str, Email, str]] = []

    start = time.time()
    for i, (email, true_label) in enumerate(pairs, 1):
        try:
            result = classify_email(email, model=args.model)
            pred = result.category.value
            reasoning = result.reasoning
        except Exception as e:
            print(f"  [{i}/{len(pairs)}] ERROR on {email.id}: {e}")
            continue

        predictions.append((true_label, pred))
        if true_label != pred:
            error_records.append((true_label, pred, email, reasoning))

        # Live progress
        marker = "✓" if true_label == pred else "✗"
        print(
            f"  [{i:>3}/{len(pairs)}] {marker} "
            f"true={true_label:<22} pred={pred:<22} "
            f"({result.confidence})"
        )

    elapsed = time.time() - start
    metrics = _compute_metrics(predictions)

    print("\n" + "=" * 60)
    print(f"Accuracy: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})")
    print(f"Elapsed:  {elapsed:.1f}s  ({elapsed / max(len(predictions), 1):.2f}s/email)")
    print("=" * 60)

    _print_per_class(metrics)
    _print_confusion_matrix(predictions)

    if args.show_errors:
        _print_errors(error_records)

    # Persist results — tracking over time is what evals are FOR
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{timestamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "model": args.model,
                "prompt_hash": _prompt_hash(),
                "metrics": metrics,
                "elapsed_seconds": round(elapsed, 2),
            },
            indent=2,
        )
    )
    print(f"\nResults saved to {out_path.relative_to(Path.cwd()) if out_path.is_relative_to(Path.cwd()) else out_path}")


if __name__ == "__main__":
    main()

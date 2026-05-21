"""Interactive CLI to build the labeled eval set quickly.

Pulls emails from your inbox, displays each one, prompts for a label
with a single keystroke. Resumable: re-running picks up where you left off
by skipping any email IDs already in golden_set.jsonl.

Run from project root:
    python -m src.eval.label_helper --limit 30
    python -m src.eval.label_helper --query "in:inbox newer_than:7d" --limit 50
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.gmail_auth import build_gmail_service
from src.gmail_client import fetch_emails
from src.models import CATEGORY_DESCRIPTIONS, Category, Email

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.jsonl"

# Single-key shortcuts → category
KEY_MAP: dict[str, Category] = {
    "1": Category.SPAM_OR_PROMO,
    "2": Category.FYI,
    "3": Category.NEEDS_REPLY_LOW,
    "4": Category.NEEDS_REPLY_HIGH,
    "5": Category.URGENT,
}


def _load_existing_ids(path: Path) -> set[str]:
    """Return IDs already labeled, so we can resume."""
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["email_id"])
    return ids


def _print_email(email: Email, idx: int, total: int) -> None:
    """Render one email for labeling."""
    print("\n" + "=" * 80)
    print(f"  Email {idx}/{total}")
    print("=" * 80)
    print(f"From:    {email.sender}")
    print(f"Subject: {email.subject}")
    print(f"Date:    {email.received_at:%Y-%m-%d %H:%M}")
    print(f"Labels:  {', '.join(email.labels) or '(none)'}")
    print("-" * 80)
    # Show snippet + first chunk of body, trimmed to keep terminal manageable
    body_preview = email.body[:1200]
    print(body_preview)
    if len(email.body) > 1200:
        print(f"\n[... {len(email.body) - 1200} more chars truncated ...]")
    print("=" * 80)


def _print_menu() -> None:
    print("\nLabel this email:")
    for key, cat in KEY_MAP.items():
        desc = CATEGORY_DESCRIPTIONS[cat].split(".")[0]  # first sentence only
        print(f"  [{key}] {cat.value:25s} — {desc}")
    print("  [s] skip (don't add to golden set)")
    print("  [q] quit (save progress)")


def _prompt_label() -> str | None:
    """Return Category value, or 'skip', or None for quit."""
    while True:
        choice = input("\n> ").strip().lower()
        if choice in KEY_MAP:
            return KEY_MAP[choice].value
        if choice == "s":
            return "skip"
        if choice == "q":
            return None
        print(f"Invalid choice {choice!r}. Use 1-5, s, or q.")


def _append_label(path: Path, email: Email, label: str) -> None:
    """Append one labeled example. Stores email fields too so eval is
    reproducible even if the email later gets deleted from Gmail."""
    record = {
        "email_id": email.id,
        "label": label,
        "sender": email.sender,
        "sender_email": email.sender_email,
        "subject": email.subject,
        "body": email.body,
        "received_at": email.received_at.isoformat(),
        "labels": email.labels,
    }
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Label emails for the eval set.")
    parser.add_argument(
        "--query",
        default="in:inbox newer_than:14d",
        help="Gmail search query (default: recent inbox).",
    )
    parser.add_argument(
        "--limit", type=int, default=30, help="Max emails to fetch."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=GOLDEN_SET_PATH,
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)  # quieter UI

    existing_ids = _load_existing_ids(args.output)
    print(f"\n{len(existing_ids)} emails already labeled in {args.output.name}")

    service = build_gmail_service()
    print(f"Fetching up to {args.limit} emails matching: {args.query!r}\n")
    emails = fetch_emails(service, query=args.query, limit=args.limit)

    # Filter to unlabeled
    new_emails = [e for e in emails if e.id not in existing_ids]
    print(f"{len(new_emails)} new (filtered out {len(emails) - len(new_emails)} already labeled)")

    if not new_emails:
        print("Nothing to label. Try a broader query.")
        return

    labeled_count = 0
    for i, email in enumerate(new_emails, 1):
        _print_email(email, i, len(new_emails))
        _print_menu()
        label = _prompt_label()

        if label is None:
            print("\nQuitting. Progress saved.")
            break
        if label == "skip":
            print("Skipped.")
            continue

        _append_label(args.output, email, label)
        labeled_count += 1
        print(f"✓ Saved as {label}")

    total = len(_load_existing_ids(args.output))
    print(f"\nDone. Added {labeled_count} new labels. Total in golden set: {total}")

    # Per-class distribution check — balanced sets eval better
    if total > 0:
        counts: dict[str, int] = {}
        with args.output.open() as f:
            for line in f:
                if line.strip():
                    label = json.loads(line)["label"]
                    counts[label] = counts.get(label, 0) + 1
        print("\nClass distribution:")
        for label, count in sorted(counts.items()):
            bar = "█" * count
            print(f"  {label:25s} {count:3d}  {bar}")


if __name__ == "__main__":
    main()

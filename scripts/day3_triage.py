"""Day 3 runner: fetch emails, run the triage graph, write audit log.

Run from project root:
    python -m scripts.day3_triage --limit 10
    python -m scripts.day3_triage --query "is:unread" --limit 20
    python -m scripts.day3_triage --reprocess  # ignore already_processed check
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone

from src.classifier import CLASSIFIER_MODEL
from src.config import DEFAULT_FETCH_LIMIT, DEFAULT_QUERY
from src.db import already_processed, insert_record
from src.drafter import DRAFTER_MODEL
from src.gmail_auth import build_gmail_service
from src.gmail_client import fetch_emails
from src.graph.builder import build_graph
from src.models import ActionTaken, AuditRecord, Email


def _get_user_email(service) -> str:
    """Fetch the authenticated user's email — used for risk domain check."""
    profile = service.users().getProfile(userId="me").execute()
    return profile.get("emailAddress", "")


def _run_one(graph, email: Email, user_email: str) -> AuditRecord:
    """Run the graph on one email, return an AuditRecord."""
    start = time.time()
    error: str | None = None
    final_state: dict = {}

    try:
        final_state = graph.invoke(
            {"email": email, "user_email": user_email, "user_context": ""}
        )
    except Exception as e:
        logging.exception("Graph failed on %s", email.id)
        error = f"{type(e).__name__}: {e}"

    elapsed = time.time() - start

    classification = final_state.get("classification")
    draft = final_state.get("draft")
    risk = final_state.get("risk")
    action = final_state.get("action", ActionTaken.SKIPPED)

    return AuditRecord(
        email_id=email.id,
        thread_id=email.thread_id,
        sender_email=email.sender_email,
        subject=email.subject,
        classification=classification.category if classification else "fyi",
        confidence=classification.confidence if classification else "low",
        classification_reasoning=(
            classification.reasoning if classification else "(failed)"
        ),
        draft_body=draft.body if draft else None,
        risk_level=risk.level if risk else None,
        risk_signals=risk.signals if risk else [],
        action=action,
        processed_at=datetime.now(timezone.utc),
        model_classifier=CLASSIFIER_MODEL,
        model_drafter=DRAFTER_MODEL if draft else None,
        latency_seconds=round(elapsed, 2),
        error=error,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--limit", type=int, default=DEFAULT_FETCH_LIMIT)
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Reprocess emails already in audit log.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("\n=== Day 3: Triage Agent ===\n")

    service = build_gmail_service()
    user_email = _get_user_email(service)
    print(f"User: {user_email}")

    print(f"Fetching: query={args.query!r} limit={args.limit}")
    emails = fetch_emails(service, query=args.query, limit=args.limit)
    print(f"Fetched {len(emails)} emails\n")

    if not args.reprocess:
        before = len(emails)
        emails = [e for e in emails if not already_processed(e.id)]
        print(f"After filtering already-processed: {len(emails)} (skipped {before - len(emails)})\n")

    if not emails:
        print("Nothing to process. Try --reprocess or a broader --query.")
        return

    print("Building graph...")
    graph = build_graph()
    print("Graph compiled. Processing emails:\n")

    counts: dict[str, int] = {}
    total_start = time.time()

    for i, email in enumerate(emails, 1):
        print(f"[{i}/{len(emails)}] {email.short_repr()}")
        record = _run_one(graph, email, user_email)
        insert_record(record)

        action = record.action.value
        counts[action] = counts.get(action, 0) + 1

        # Compact per-email line
        bits = [
            f"  → {record.classification.value} ({record.confidence})",
            f"action: {action}",
        ]
        if record.risk_level:
            risk_str = f"risk: {record.risk_level.value}"
            if record.risk_signals:
                risk_str += f" [{','.join(record.risk_signals[:2])}]"
            bits.append(risk_str)
        bits.append(f"{record.latency_seconds}s")
        if record.error:
            bits.append(f"ERROR: {record.error}")
        print("  " + " | ".join(bits))
        print()

    total_elapsed = time.time() - total_start
    print("=" * 60)
    print(f"Processed {len(emails)} emails in {total_elapsed:.1f}s "
          f"({total_elapsed / len(emails):.1f}s/email avg)")
    print("\nAction breakdown:")
    for action, count in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"  {action:30s} {count:3d}  {bar}")
    print("\nFull audit log in triage.db. Day 4 UI will surface this.")


if __name__ == "__main__":
    main()

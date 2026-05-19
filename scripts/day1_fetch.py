"""Day 1 smoke test: authenticate and fetch a few emails.

Run from project root:
    python -m scripts.day1_fetch

First run: opens browser for OAuth consent.
Subsequent runs: uses cached token.json.
"""
import logging

from src.gmail_auth import build_gmail_service
from src.gmail_client import fetch_emails


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("\n=== Day 1: Gmail OAuth + Fetch ===\n")

    service = build_gmail_service()
    print("✓ Authenticated\n")

    emails = fetch_emails(service, query="is:unread in:inbox", limit=5)
    print(f"✓ Fetched {len(emails)} unread emails\n")

    if not emails:
        print("No unread emails. Try query='in:inbox' to fetch recent read ones.")
        return

    print("--- Inbox preview ---")
    for i, email in enumerate(emails, 1):
        print(f"\n[{i}] {email.short_repr()}")
        print(f"    From: {email.sender_email}")
        print(f"    Labels: {', '.join(email.labels) or '(none)'}")
        print(f"    Snippet: {email.snippet[:120]}")

    print(f"\n--- Full body of email 1 ---")
    print(emails[0].body[:500])
    if len(emails[0].body) > 500:
        print(f"\n[...{len(emails[0].body) - 500} more chars]")


if __name__ == "__main__":
    main()

"""Gmail message fetching and parsing.

Gmail's API returns deeply nested messages with base64url-encoded parts
and a multipart MIME tree. This module flattens that into our Email model.
"""
from __future__ import annotations

import base64
import logging
import re
from datetime import datetime, timezone
from email.utils import parseaddr, parsedate_to_datetime
from typing import Any

from googleapiclient.discovery import Resource

from src.config import DEFAULT_FETCH_LIMIT, DEFAULT_QUERY
from src.models import Email

logger = logging.getLogger(__name__)


def _decode_body(data: str) -> str:
    """Gmail uses URL-safe base64 with no padding."""
    # urlsafe_b64decode requires correct padding — pad to multiple of 4
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")


def _extract_plaintext(payload: dict[str, Any]) -> str:
    """Walk MIME tree, prefer text/plain, fall back to stripped text/html.

    Multipart messages have nested `parts`; single-part messages have body
    directly on the payload. We DFS until we find usable text.
    """
    mime_type = payload.get("mimeType", "")
    body = payload.get("body", {})

    # Leaf node with body data
    if "data" in body:
        decoded = _decode_body(body["data"])
        if mime_type == "text/plain":
            return decoded
        if mime_type == "text/html":
            # Cheap HTML strip — fine for triage, real prod would use bleach
            return re.sub(r"<[^>]+>", " ", decoded)

    # Multipart — recurse, prefer plain over html
    parts = payload.get("parts", [])
    plain_text = ""
    html_fallback = ""
    for part in parts:
        text = _extract_plaintext(part)
        if part.get("mimeType") == "text/plain" and text:
            plain_text = text
            break
        if part.get("mimeType", "").startswith("text/html") and text:
            html_fallback = text

    return plain_text or html_fallback


def _get_header(headers: list[dict[str, str]], name: str) -> str:
    """Case-insensitive header lookup. Gmail returns headers as [{name, value}]."""
    name_lower = name.lower()
    for h in headers:
        if h.get("name", "").lower() == name_lower:
            return h.get("value", "")
    return ""


def _parse_date(raw: str) -> datetime:
    """RFC 2822 date → aware datetime. Fall back to now() on garbage."""
    try:
        dt = parsedate_to_datetime(raw)
        # Some senders send naive dates; assume UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (TypeError, ValueError):
        logger.warning("Unparseable date %r, using now()", raw)
        return datetime.now(timezone.utc)


def _message_to_email(msg: dict[str, Any]) -> Email:
    """Convert a Gmail API message resource to our Email model."""
    payload = msg.get("payload", {})
    headers = payload.get("headers", [])

    from_raw = _get_header(headers, "From")
    sender_name, sender_email = parseaddr(from_raw)

    body = _extract_plaintext(payload).strip()
    # Truncate absurdly long bodies — keeps prompt costs bounded.
    # 8k chars is ~2k tokens; full thread context comes later if needed.
    if len(body) > 8000:
        body = body[:8000] + "\n\n[...truncated...]"

    return Email(
        id=msg["id"],
        thread_id=msg["threadId"],
        sender=from_raw,
        sender_email=sender_email,
        sender_name=sender_name or None,
        recipient=_get_header(headers, "To"),
        subject=_get_header(headers, "Subject") or "(no subject)",
        body=body,
        snippet=msg.get("snippet", ""),
        received_at=_parse_date(_get_header(headers, "Date")),
        labels=msg.get("labelIds", []),
        is_unread="UNREAD" in msg.get("labelIds", []),
    )


def fetch_emails(
    service: Resource,
    query: str = DEFAULT_QUERY,
    limit: int = DEFAULT_FETCH_LIMIT,
) -> list[Email]:
    """Fetch and parse emails matching a Gmail search query.

    Two-step process required by Gmail API:
    1. list() returns message IDs only (cheap, paginated)
    2. get() per message fetches full payload (we batch via loop, not
       async — at limit=20 the overhead is negligible)

    Args:
        service: Authenticated Gmail API client from build_gmail_service().
        query: Gmail search syntax. Examples:
            - "is:unread in:inbox" (default)
            - "from:boss@company.com newer_than:7d"
            - "is:unread -category:promotions"
        limit: Max messages to fetch.
    """
    logger.info("Fetching emails: query=%r limit=%d", query, limit)

    list_resp = (
        service.users()
        .messages()
        .list(userId="me", q=query, maxResults=limit)
        .execute()
    )
    message_refs = list_resp.get("messages", [])

    if not message_refs:
        logger.info("No messages matched query")
        return []

    emails: list[Email] = []
    for ref in message_refs:
        # format=full returns headers + body. metadata-only would save tokens
        # but we need the body for classification.
        msg = (
            service.users()
            .messages()
            .get(userId="me", id=ref["id"], format="full")
            .execute()
        )
        try:
            emails.append(_message_to_email(msg))
        except Exception as e:
            # Don't let one malformed message kill the whole batch
            logger.exception("Failed to parse message %s: %s", ref["id"], e)

    logger.info("Fetched %d emails", len(emails))
    return emails

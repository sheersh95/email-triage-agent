"""Gmail action layer — send, archive, label.

Critical detail: replies need In-Reply-To and References headers plus
the threadId on the API call, otherwise Gmail starts a new thread
instead of replying in place. Most tutorial code skips this and the
result looks broken in the recipient's inbox.

All functions take an authenticated service from build_gmail_service()
and return None on success, raise on failure. Callers wrap with try/except.
"""
from __future__ import annotations

import base64
import logging
from email.message import EmailMessage
from email.utils import make_msgid
from typing import Any

from googleapiclient.discovery import Resource
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

# Custom label name used by the agent for FYI emails
TRIAGE_LABEL_FYI = "Triage/FYI"


def _get_message_headers(service: Resource, message_id: str) -> dict[str, str]:
    """Fetch just the headers we need for threading."""
    msg = (
        service.users()
        .messages()
        .get(userId="me", id=message_id, format="metadata",
             metadataHeaders=["Message-ID", "References", "Subject", "From"])
        .execute()
    )
    headers = msg.get("payload", {}).get("headers", [])
    return {h["name"]: h["value"] for h in headers}


def send_reply(
    service: Resource,
    *,
    thread_id: str,
    in_reply_to_message_id: str,
    to_address: str,
    subject: str,
    body: str,
) -> str:
    """Send a threaded reply. Returns the new message ID.

    Threading requirements:
    - threadId on the API call (groups in Gmail UI)
    - In-Reply-To header (the original message's Message-ID, with angle brackets)
    - References header (chain of Message-IDs, we append to existing)

    Args:
        service: authenticated Gmail API client
        thread_id: Gmail thread ID of the original
        in_reply_to_message_id: Gmail message ID of the original
        to_address: recipient (usually the original sender)
        subject: pre-prefixed with "Re:" by the caller
        body: plaintext reply body
    """
    # Fetch original headers for Message-ID + References chain
    orig_headers = _get_message_headers(service, in_reply_to_message_id)
    orig_message_id = orig_headers.get("Message-ID", "")
    orig_references = orig_headers.get("References", "")

    # Build References chain. Per RFC 5322: append the parent Message-ID.
    if orig_references and orig_message_id:
        references = f"{orig_references} {orig_message_id}"
    elif orig_message_id:
        references = orig_message_id
    else:
        # No Message-ID — rare but possible. Threading by threadId still works
        # in Gmail; the recipient's client may not group it correctly.
        references = ""

    msg = EmailMessage()
    msg["To"] = to_address
    msg["Subject"] = subject
    msg["Message-ID"] = make_msgid()
    if orig_message_id:
        msg["In-Reply-To"] = orig_message_id
    if references:
        msg["References"] = references
    msg.set_content(body)

    # Gmail wants base64url-encoded RFC 822 message
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()

    try:
        sent = (
            service.users()
            .messages()
            .send(
                userId="me",
                body={"raw": raw, "threadId": thread_id},
            )
            .execute()
        )
    except HttpError as e:
        logger.exception("Gmail send failed for thread %s", thread_id)
        raise RuntimeError(f"Gmail API error: {e}") from e

    new_id = sent.get("id", "")
    logger.info("Sent reply in thread %s, new message id=%s", thread_id, new_id)
    return new_id


def archive_message(service: Resource, message_id: str) -> None:
    """Archive by removing the INBOX label (Gmail's definition of archive)."""
    try:
        service.users().messages().modify(
            userId="me",
            id=message_id,
            body={"removeLabelIds": ["INBOX"]},
        ).execute()
        logger.info("Archived %s", message_id)
    except HttpError as e:
        raise RuntimeError(f"Archive failed: {e}") from e


def _get_or_create_label(service: Resource, name: str) -> str:
    """Look up label by name; create if missing. Returns label ID."""
    existing = service.users().labels().list(userId="me").execute().get("labels", [])
    for lbl in existing:
        if lbl["name"] == name:
            return lbl["id"]

    # Create it. Default visibility settings are fine.
    created = (
        service.users()
        .labels()
        .create(
            userId="me",
            body={
                "name": name,
                "labelListVisibility": "labelShow",
                "messageListVisibility": "show",
            },
        )
        .execute()
    )
    logger.info("Created label %r (id=%s)", name, created["id"])
    return created["id"]


def apply_fyi_label(service: Resource, message_id: str) -> None:
    """Apply our 'Triage/FYI' label. Creates the label on first use.

    We don't archive FYI emails — user might still want them in inbox.
    Just label them for easy filtering. Day 5 could add a setting for
    archive-on-label behavior.
    """
    label_id = _get_or_create_label(service, TRIAGE_LABEL_FYI)
    try:
        service.users().messages().modify(
            userId="me",
            id=message_id,
            body={"addLabelIds": [label_id]},
        ).execute()
        logger.info("Labeled %s as %s", message_id, TRIAGE_LABEL_FYI)
    except HttpError as e:
        raise RuntimeError(f"Label failed: {e}") from e

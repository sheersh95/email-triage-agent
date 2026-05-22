"""LangGraph nodes — one function per node.

Each node takes the full state, returns a partial state (delta).
LangGraph merges deltas into the running state automatically.

Pure functions in spirit — side effects (LLM calls) are unavoidable,
but no node mutates its input.
"""
from __future__ import annotations

import logging

from src.classifier import CLASSIFIER_MODEL, classify_email
from src.drafter import DRAFTER_MODEL, draft_reply
from src.graph.state import TriageState
from src.models import ActionTaken, Category
from src.tools.risk import assess_risk

logger = logging.getLogger(__name__)


def classify_node(state: TriageState) -> dict:
    """Classify the email. Returns {classification: ...}."""
    email = state["email"]
    logger.info("classify_node: %s", email.short_repr())
    classification = classify_email(email)
    return {"classification": classification}


def draft_node(state: TriageState) -> dict:
    """Generate a reply draft. Only called for reply-eligible categories."""
    email = state["email"]
    user_context = state.get("user_context", "")
    logger.info("draft_node: drafting reply for %s", email.id)
    draft = draft_reply(email, user_context=user_context)
    return {"draft": draft}


def risk_node(state: TriageState) -> dict:
    """Assess risk of the draft. Determines auto-send vs approval queue."""
    email = state["email"]
    draft = state["draft"]
    classification = state["classification"]
    user_email = state.get("user_email")

    risk = assess_risk(email, draft, classification, user_email=user_email)
    logger.info(
        "risk_node: %s -> %s (signals=%s)",
        email.id, risk.level.value, risk.signals,
    )
    return {"risk": risk}


# ─── Terminal action nodes ──────────────────────────────────────────────
# Day 3: these only RECORD the intended action. Day 4 will execute side
# effects (archive in Gmail, send draft, etc.) once we've reviewed behavior.

def archive_action(state: TriageState) -> dict:
    """Spam/promo → mark for archiving (no actual archive yet)."""
    logger.info("archive_action: would archive %s", state["email"].id)
    return {"action": ActionTaken.ARCHIVED}


def label_fyi_action(state: TriageState) -> dict:
    """FYI → mark for labeling (no actual label applied yet)."""
    logger.info("label_fyi_action: would label %s", state["email"].id)
    return {"action": ActionTaken.LABELED_FYI}


def auto_send_action(state: TriageState) -> dict:
    """Low-risk reply-eligible. Behavior depends on auto-send mode:

    - Auto-send ON: send via Gmail immediately, record as DRAFTED_AUTO_SEND
      with sent timestamp. Failures fall back to needs-approval.
    - Auto-send OFF (default, safer): record as DRAFTED_NEEDS_APPROVAL so
      it shows up in the UI queue.

    Side effects (Gmail send) live here, not in the orchestrator, because
    LangGraph already gives us per-email isolation and retry semantics.
    """
    from src.config import is_auto_send_enabled
    from src.gmail_auth import build_gmail_service
    from src.tools.gmail_actions import send_reply

    email = state["email"]
    draft = state.get("draft")

    if not is_auto_send_enabled() or draft is None:
        logger.info(
            "auto_send_action: queuing for approval (auto_send=%s, has_draft=%s)",
            is_auto_send_enabled(), draft is not None,
        )
        return {"action": ActionTaken.DRAFTED_NEEDS_APPROVAL}

    try:
        service = build_gmail_service()
        send_reply(
            service,
            thread_id=email.thread_id,
            in_reply_to_message_id=email.id,
            to_address=email.sender_email,
            subject=draft.subject,
            body=draft.body,
        )
        logger.info("auto_send_action: SENT reply for %s", email.id)
        return {"action": ActionTaken.DRAFTED_AUTO_SEND}
    except Exception as e:
        # Don't lose the draft — queue for human review instead
        logger.exception("auto-send failed, queuing for approval: %s", e)
        return {
            "action": ActionTaken.DRAFTED_NEEDS_APPROVAL,
            "error": f"auto_send_failed: {e}",
        }


def approval_action(state: TriageState) -> dict:
    """High-risk → queue for human approval."""
    logger.info("approval_action: needs review for %s", state["email"].id)
    return {"action": ActionTaken.DRAFTED_NEEDS_APPROVAL}


def urgent_action(state: TriageState) -> dict:
    """Urgent → notify; still drafts a reply for review."""
    logger.info("urgent_action: urgent flag for %s", state["email"].id)
    return {"action": ActionTaken.NOTIFIED_URGENT}

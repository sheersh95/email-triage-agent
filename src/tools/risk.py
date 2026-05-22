"""Risk assessment for drafted replies.

Hybrid design:
1. Deterministic signals (regex, sender heuristics, length) — fast and explainable.
2. If ANY high-risk signal fires → high risk. Stop.
3. If no signals fire but classification confidence is medium → LLM tiebreaker.
4. Otherwise → low risk.

Why this design:
- Pure LLM risk assessment is unreliable: models are inconsistent on
  "is this risky?" judgments across runs.
- Pure rules miss novel cases (social engineering, subtle commitments).
- Rules-first is auditable: when a draft gets queued, we can show
  exactly which signal fired.

The signal set is intentionally conservative — false positives (extra
human review) are far cheaper than false negatives (auto-sent money
commitments).
"""
from __future__ import annotations

import logging
import os
import re
from functools import lru_cache

from anthropic import Anthropic

from src.models import (
    Classification,
    Draft,
    Email,
    RiskAssessment,
    RiskLevel,
)

logger = logging.getLogger(__name__)

# ─── Deterministic signals ──────────────────────────────────────────────

MONEY_PATTERN = re.compile(
    # Currency symbols don't need \b (they're not word chars). Word-chars
    # (usd/eur/etc) DO need it. Two alternatives:
    r"(?:[$€£₹]\s*\d|\b(?:usd|eur|gbp|inr|rs\.?)\s*\d"
    r"|\b\d+\s*(?:dollars|euros|pounds|rupees)\b"
    r"|\b(?:payment|invoice|refund|wire transfer|payable|"
    r"reimburse|amount owed|outstanding balance)\b)",
    re.IGNORECASE,
)

COMMITMENT_PATTERN = re.compile(
    r"(?i)\b(?:i (?:will|'ll|can|agree|confirm|commit|promise)|"
    r"we (?:will|'ll|can|agree|confirm|commit)|"
    r"yes,? (?:i|we)|absolutely|definitely|consider it done|"
    r"sounds good|will do)\b"
)

LEGAL_HR_PATTERN = re.compile(
    r"(?i)\b(?:lawsuit|attorney|legal action|cease and desist|"
    r"termination|resignation|fired|laid off|harassment|"
    r"discrimination|nda|non-disclosure|severance|compliance|"
    r"hr (?:complaint|investigation)|grievance)\b"
)

CREDENTIAL_PATTERN = re.compile(
    r"(?i)\b(?:password|api[ _]?key|token|credential|ssn|"
    r"social security|credit card|cvv|bank account|routing number)\b"
)

# Length signal — long drafts deserve review even if no other flags
DRAFT_LENGTH_LIMIT_WORDS = 150


def _count_words(text: str) -> int:
    return len(text.split())


def _is_internal_sender(sender_email: str, user_email: str | None) -> bool:
    """Same domain as user → internal. Less risky than external."""
    if not user_email or "@" not in sender_email or "@" not in user_email:
        return False
    return sender_email.split("@")[-1].lower() == user_email.split("@")[-1].lower()


def _check_rules(
    email: Email, draft: Draft, user_email: str | None
) -> list[str]:
    """Return list of high-risk signal names that fired."""
    signals: list[str] = []
    combined_text = f"{email.body}\n{draft.body}"

    if MONEY_PATTERN.search(combined_text):
        signals.append("money_terms")
    if COMMITMENT_PATTERN.search(draft.body):
        signals.append("commitment_language_in_draft")
    if LEGAL_HR_PATTERN.search(combined_text):
        signals.append("legal_or_hr_topic")
    if CREDENTIAL_PATTERN.search(combined_text):
        signals.append("credentials_or_pii")
    if _count_words(draft.body) > DRAFT_LENGTH_LIMIT_WORDS:
        signals.append(f"long_draft_over_{DRAFT_LENGTH_LIMIT_WORDS}_words")
    if not _is_internal_sender(email.sender_email, user_email):
        # External sender alone isn't enough — but combined with anything
        # else it's higher signal. We log it but don't gate on it alone.
        # (If you want stricter behavior, append "external_sender" to signals.)
        pass

    return signals


# ─── LLM tiebreaker ─────────────────────────────────────────────────────

TIEBREAKER_TOOL = {
    "name": "submit_risk",
    "description": "Submit risk judgment for an ambiguous draft.",
    "input_schema": {
        "type": "object",
        "properties": {
            "level": {"type": "string", "enum": ["low", "high"]},
            "reasoning": {"type": "string"},
        },
        "required": ["level", "reasoning"],
    },
}

TIEBREAKER_PROMPT = """You are reviewing a drafted email reply for risk. \
The draft will be auto-sent if you say "low" — so err on the side of "high" \
when uncertain.

Mark as HIGH risk if the draft:
- Makes any commitment, agreement, or promise
- Could damage a professional relationship if wrong
- Touches sensitive personal or business matters
- Sounds like it might be responding to a request without enough context

Mark as LOW risk only if the draft is clearly safe to send unsupervised \
(simple acknowledgments, scheduling confirmations with known contacts, \
informational replies with no commitments).

Original email:
From: {sender}
Subject: {subject}
{body}

Drafted reply:
{draft_body}
"""


@lru_cache(maxsize=1)
def _get_client() -> Anthropic:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY env var not set")
    return Anthropic()


def _llm_tiebreaker(email: Email, draft: Draft) -> tuple[RiskLevel, str]:
    """Ask Haiku for a judgment when rules are silent but confidence is low."""
    client = _get_client()
    prompt = TIEBREAKER_PROMPT.format(
        sender=email.sender,
        subject=email.subject,
        body=email.body[:2000],
        draft_body=draft.body,
    )
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        tools=[TIEBREAKER_TOOL],
        tool_choice={"type": "tool", "name": "submit_risk"},
        messages=[{"role": "user", "content": prompt}],
    )
    from src.usage_meter import extract_usage, record
    record(extract_usage(response, "claude-haiku-4-5-20251001"))
    for block in response.content:
        if block.type == "tool_use":
            level = RiskLevel(block.input["level"])
            return level, block.input["reasoning"]
    # Fail safe — if tool call missing, default to HIGH
    return RiskLevel.HIGH, "tiebreaker_parse_failure_defaulting_high"


# ─── Public API ─────────────────────────────────────────────────────────


def assess_risk(
    email: Email,
    draft: Draft,
    classification: Classification,
    user_email: str | None = None,
) -> RiskAssessment:
    """Combined rule + LLM risk assessment.

    Decision tree:
    1. If any rule fires → HIGH (no LLM call needed)
    2. If classification confidence is "low" → HIGH (we don't trust the classifier)
    3. If classification confidence is "medium" → LLM tiebreaker
    4. Otherwise (confidence high, no rules fired) → LOW
    """
    signals = _check_rules(email, draft, user_email)

    if signals:
        logger.info("Rules fired for %s: %s", email.id, signals)
        return RiskAssessment(level=RiskLevel.HIGH, signals=signals)

    if classification.confidence == "low":
        return RiskAssessment(
            level=RiskLevel.HIGH,
            signals=["classifier_low_confidence"],
        )

    if classification.confidence == "medium":
        level, reasoning = _llm_tiebreaker(email, draft)
        return RiskAssessment(
            level=level, signals=[], llm_reasoning=reasoning
        )

    return RiskAssessment(level=RiskLevel.LOW, signals=[])

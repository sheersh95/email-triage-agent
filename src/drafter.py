"""Draft reply generation using Claude Sonnet 4.6.

Why Sonnet here and Haiku for classification?
- Drafting is open-ended generation where quality matters. Bad drafts
  waste the user's review time even if they catch the issues.
- Classification is bounded multiple choice where Haiku is fine.
- Cost difference per email is ~$0.005 vs ~$0.001. Worth it for drafts.

User context placeholder: in Day 5 we'll pull 20 sent emails and extract
tone/signature/typical length to make drafts sound like the user.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from src.models import Draft, Email

logger = logging.getLogger(__name__)

DRAFTER_MODEL = "claude-sonnet-4-6"
PROMPT_PATH = Path(__file__).parent / "prompts" / "draft.txt"

# Default user context — overridden when we add style extraction
DEFAULT_USER_CONTEXT = (
    "Name: (unknown, write without signing off with a name)\n"
    "Role: professional\n"
    "Style: friendly, concise, direct"
)

DRAFT_TOOL = {
    "name": "submit_draft",
    "description": "Submit the drafted reply.",
    "input_schema": {
        "type": "object",
        "properties": {
            "body": {
                "type": "string",
                "description": "The reply body, plaintext, no salutation placeholders.",
            }
        },
        "required": ["body"],
    },
}


@lru_cache(maxsize=1)
def _load_prompt() -> str:
    return PROMPT_PATH.read_text()


@lru_cache(maxsize=1)
def _get_client() -> Anthropic:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY env var not set")
    return Anthropic()


def draft_reply(
    email: Email,
    user_context: str | None = None,
    model: str = DRAFTER_MODEL,
) -> Draft:
    """Generate a reply draft for the given email.

    user_context behavior:
    - If explicitly provided (non-empty), use it verbatim
    - If None or "", auto-load the cached style profile (from style extraction)
    - If no cached profile exists, fall back to DEFAULT_USER_CONTEXT

    This lets the orchestrator stay agnostic — it can pass user_context=""
    and the drafter handles style lookup transparently.
    """
    client = _get_client()

    # Resolve user_context lazily so we always pick up the most recent profile
    if not user_context:
        # Lazy import to avoid a circular import with src.tools.style
        from src.tools.style import load_profile

        profile = load_profile()
        if profile is not None:
            user_context = profile.to_prompt_block()
        else:
            user_context = DEFAULT_USER_CONTEXT

    prompt = _load_prompt().format(
        user_context=user_context,
        sender=email.sender,
        subject=email.subject,
        body=email.body[:4000],
    )

    response = client.messages.create(
        model=model,
        max_tokens=600,
        tools=[DRAFT_TOOL],
        tool_choice={"type": "tool", "name": "submit_draft"},
        messages=[{"role": "user", "content": prompt}],
    )

    from src.usage_meter import extract_usage, record
    record(extract_usage(response, model))

    body: str | None = None
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_draft":
            body = block.input.get("body")
            break

    if not body:
        raise RuntimeError(f"No draft in response: {response.content!r}")

    # Subject: prefix with "Re:" if not already
    subject = email.subject
    if not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"

    return Draft(subject=subject, body=body.strip(), in_reply_to=email.id)

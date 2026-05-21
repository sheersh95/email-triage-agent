"""Email classifier using Claude Haiku 4.5 via Anthropic SDK.

Design notes:
- Uses `tool_use` to force structured JSON output. Far more reliable than
  "respond in JSON format" prompting, which fails ~5% of the time and
  needs janky cleanup of markdown fences.
- Haiku 4.5 picked over Sonnet because classification is bounded and
  Haiku is ~5x cheaper. We measure to confirm — if Haiku underperforms
  on a class, we promote just that class to Sonnet.
- Retry on JSON parse failure (rare with tool_use, but defensive).
"""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from pydantic import ValidationError

from src.models import CATEGORY_DESCRIPTIONS, Category, Classification, Email

logger = logging.getLogger(__name__)

CLASSIFIER_MODEL = "claude-haiku-4-5-20251001"
PROMPT_PATH = Path(__file__).parent / "prompts" / "classify.txt"

# Tool schema — Anthropic SDK enforces this on the model side, so we
# get guaranteed-valid JSON matching the schema (no markdown, no preamble).
CLASSIFY_TOOL = {
    "name": "submit_classification",
    "description": "Submit the email's category, confidence, and reasoning.",
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": [c.value for c in Category],
                "description": "Single best-fitting category.",
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "How confident in this classification.",
            },
            "reasoning": {
                "type": "string",
                "description": "1-2 sentence justification for the choice.",
            },
        },
        "required": ["category", "confidence", "reasoning"],
    },
}


@lru_cache(maxsize=1)
def _load_prompt_template() -> str:
    return PROMPT_PATH.read_text()


def _build_prompt(email: Email) -> str:
    """Fill the prompt template with email fields and category descriptions."""
    cat_desc = "\n".join(
        f"- **{c.value}**: {desc}" for c, desc in CATEGORY_DESCRIPTIONS.items()
    )
    return _load_prompt_template().format(
        category_descriptions=cat_desc,
        sender=email.sender,
        subject=email.subject,
        received_at=email.received_at.isoformat(),
        body=email.body[:4000],  # extra safety; gmail_client truncates at 8k
    )


@lru_cache(maxsize=1)
def _get_client() -> Anthropic:
    """Lazy-init the client. lru_cache makes it a singleton."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY env var not set. "
            "Get one at https://console.anthropic.com/"
        )
    return Anthropic()


def classify_email(email: Email, model: str = CLASSIFIER_MODEL) -> Classification:
    """Classify an email into one of the Category values.

    Returns Classification with category, confidence, reasoning.
    Raises on repeated parse failures (should be vanishingly rare with tool_use).
    """
    client = _get_client()
    prompt = _build_prompt(email)

    logger.debug("Classifying %s with %s", email.short_repr(), model)

    response = client.messages.create(
        model=model,
        max_tokens=300,  # reasoning is short; cap protects against runaways
        tools=[CLASSIFY_TOOL],
        tool_choice={"type": "tool", "name": "submit_classification"},
        messages=[{"role": "user", "content": prompt}],
    )

    # Record token usage against any active accumulator
    from src.usage_meter import extract_usage, record
    record(extract_usage(response, model))

    # Find the tool_use block — when tool_choice is forced, there should
    # be exactly one, but iterate to be defensive against future API changes.
    tool_input: dict[str, Any] | None = None
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_classification":
            tool_input = block.input
            break

    if tool_input is None:
        # Should never happen with forced tool_choice, but fail loud if it does
        raise RuntimeError(
            f"No tool_use block in response. Got: {response.content!r}"
        )

    try:
        return Classification(**tool_input)
    except ValidationError as e:
        # Tool schema constrains shape; this would mean the model returned
        # an invalid enum value despite the schema. Worth logging loudly.
        logger.error("Pydantic validation failed: %s. Raw: %s", e, tool_input)
        raise

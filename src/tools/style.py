"""Writing style extraction from user's sent emails.

Why this matters:
- Generic LLM drafts sound like ChatGPT: over-formal, "I hope this finds you well",
  long-winded sign-offs. People can tell within 2 seconds and won't trust auto-send.
- A style profile injected into the drafter prompt makes drafts sound like the user.

Approach:
1. Fetch 30 recent sent emails (skipping auto-replies, forwards, threads)
2. Send a sample to Sonnet, ask for a structured style profile
3. Cache the profile to disk; reload until user explicitly refreshes

Cached profile lives at <project_root>/.style_profile.json — gitignored
because it contains personal voice samples.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from anthropic import Anthropic
from googleapiclient.discovery import Resource
from pydantic import BaseModel, Field

from src.config import PROJECT_ROOT
from src.gmail_client import fetch_emails

logger = logging.getLogger(__name__)

STYLE_PROFILE_PATH = PROJECT_ROOT / ".style_profile.json"
STYLE_MODEL = "claude-sonnet-4-6"

# Heuristics to filter junk before sending to the LLM
MIN_BODY_CHARS = 30      # too short = not a real reply
MAX_BODY_CHARS = 2000    # too long = probably a forwarded thread
SAMPLE_SIZE = 30


class StyleProfile(BaseModel):
    """Structured style profile, persisted as JSON."""

    name: str | None = Field(None, description="User's name if inferable from sign-offs")
    role_hint: str | None = Field(None, description="Inferred role/industry, if any")
    tone: str = Field(..., description="e.g. 'warm and concise', 'formal'")
    typical_length: str = Field(..., description="e.g. '2-4 sentences', 'one-liner'")
    common_signoff: str | None = Field(None, description="e.g. 'Thanks,\\nSheersh'")
    style_notes: list[str] = Field(
        default_factory=list,
        description="Quirks like 'uses sentence-case subject lines', 'no exclamation marks'",
    )
    example_phrases: list[str] = Field(
        default_factory=list,
        description="Short snippets the user actually uses",
    )
    extracted_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sample_count: int = 0

    def to_prompt_block(self) -> str:
        """Render as a string suitable for drafter prompt injection."""
        bits: list[str] = []
        if self.name:
            bits.append(f"Name: {self.name}")
        if self.role_hint:
            bits.append(f"Role: {self.role_hint}")
        bits.append(f"Tone: {self.tone}")
        bits.append(f"Typical reply length: {self.typical_length}")
        if self.common_signoff:
            bits.append(f"Sign-off pattern: {self.common_signoff!r}")
        if self.style_notes:
            bits.append("Style notes:")
            bits.extend(f"  - {note}" for note in self.style_notes)
        if self.example_phrases:
            bits.append("Phrases the user actually uses:")
            bits.extend(f"  - {p!r}" for p in self.example_phrases)
        return "\n".join(bits)


# ─── Sent-email collection & cleaning ────────────────────────────────────


def _looks_like_quoted_reply(body: str) -> str:
    """Strip the quoted portion of a reply — '> ...' lines and 'On ... wrote:'."""
    # Truncate at the common reply boundary markers
    boundaries = [
        r"\nOn .+ wrote:",
        r"\n-----Original Message-----",
        r"\nFrom: .+\nSent: ",
        r"\n_+\nFrom:",
    ]
    for pattern in boundaries:
        m = re.search(pattern, body)
        if m:
            body = body[: m.start()]
    # Strip leftover quoted lines
    body = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith(">"))
    return body.strip()


def _is_usable_sample(body: str) -> bool:
    """Filter out auto-replies, out-of-office, one-word OKs."""
    if not body:
        return False
    if len(body) < MIN_BODY_CHARS:
        return False
    if len(body) > MAX_BODY_CHARS:
        return False
    lowered = body.lower()
    junk_markers = [
        "out of office",
        "auto-reply",
        "automatic reply",
        "unsubscribe",
        "this is an automated",
    ]
    return not any(m in lowered for m in junk_markers)


def collect_sent_samples(service: Resource, limit: int = SAMPLE_SIZE) -> list[str]:
    """Fetch recent sent emails, clean them, return body samples."""
    # `in:sent -in:chats` excludes Hangouts/Chat threads which Gmail
    # mixes into the sent label
    emails = fetch_emails(service, query="in:sent -in:chats", limit=limit * 2)

    samples: list[str] = []
    for email in emails:
        cleaned = _looks_like_quoted_reply(email.body)
        if _is_usable_sample(cleaned):
            samples.append(cleaned)
        if len(samples) >= limit:
            break

    logger.info("Collected %d usable sent samples", len(samples))
    return samples


# ─── LLM-based profile extraction ────────────────────────────────────────


STYLE_EXTRACTION_TOOL = {
    "name": "submit_style_profile",
    "description": "Submit the extracted style profile.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": ["string", "null"]},
            "role_hint": {"type": ["string", "null"]},
            "tone": {"type": "string"},
            "typical_length": {"type": "string"},
            "common_signoff": {"type": ["string", "null"]},
            "style_notes": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 8,
            },
            "example_phrases": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 6,
            },
        },
        "required": ["tone", "typical_length"],
    },
}

EXTRACTION_PROMPT = """You are analyzing a user's sent emails to extract their writing style.

Below are {n} of their recent sent emails. Read them and produce a structured \
style profile that another LLM can use to draft new replies in their voice.

Be specific. "Casual and friendly" is useless; "uses lowercase first words, \
no greeting, signs off with just first name" is useful. Pull 3-6 short phrases \
the user ACTUALLY uses (not paraphrases) — these become anchor patterns for \
future drafts.

If the emails have varied recipients/tones, prefer the most common pattern but \
note the variance in style_notes.

Sent emails:
---
{samples}
---
"""


def _get_client() -> Anthropic:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY env var not set")
    return Anthropic()


def extract_profile(samples: list[str], model: str = STYLE_MODEL) -> StyleProfile:
    """Send samples to Sonnet, get back a structured StyleProfile."""
    if not samples:
        raise ValueError("Need at least one sent email sample.")

    # Concatenate samples with separators. Truncate the whole blob to keep
    # cost bounded — 30 samples * 2000 chars = 60k chars worst case = ~15k tokens.
    joined = "\n\n--- next email ---\n\n".join(samples)
    if len(joined) > 40000:
        joined = joined[:40000] + "\n\n[...truncated...]"

    prompt = EXTRACTION_PROMPT.format(n=len(samples), samples=joined)
    client = _get_client()

    response = client.messages.create(
        model=model,
        max_tokens=800,
        tools=[STYLE_EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "submit_style_profile"},
        messages=[{"role": "user", "content": prompt}],
    )

    for block in response.content:
        if block.type == "tool_use":
            data = dict(block.input)
            data["sample_count"] = len(samples)
            return StyleProfile(**data)

    raise RuntimeError(f"No tool_use in response: {response.content!r}")


# ─── Persistence + public API ────────────────────────────────────────────


def save_profile(profile: StyleProfile, path: Path = STYLE_PROFILE_PATH) -> None:
    path.write_text(profile.model_dump_json(indent=2))
    logger.info("Saved style profile to %s", path)


def load_profile(path: Path = STYLE_PROFILE_PATH) -> StyleProfile | None:
    """Return cached profile, or None if not yet extracted."""
    if not path.exists():
        return None
    try:
        return StyleProfile.model_validate_json(path.read_text())
    except Exception as e:
        logger.warning("Cached style profile unparseable (%s); ignoring", e)
        return None


def refresh_profile(service: Resource) -> StyleProfile:
    """Re-extract and save. Called from UI button."""
    samples = collect_sent_samples(service)
    if not samples:
        raise RuntimeError(
            "No usable sent emails found. Try sending a few replies first."
        )
    profile = extract_profile(samples)
    save_profile(profile)
    return profile

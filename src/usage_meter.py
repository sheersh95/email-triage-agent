"""Token usage and cost tracking for Anthropic API calls.

Two layers:
1. `extract_usage(response)` — pull input/output/cache tokens from an
   Anthropic Messages response.
2. `UsageAccumulator` — context-managed accumulator. Wraps a logical
   unit of work (one email processed through the graph) and rolls up
   total tokens + USD cost across all nested calls.

Prices below are list prices as of model release. Update when pricing
changes — these are easy to forget.
"""
from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# USD per 1M tokens (Anthropic list prices, model release)
# Cache reads are 90% cheaper than fresh input; cache writes 25% more expensive.
PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {
        "input": 1.00,
        "output": 5.00,
        "cache_read": 0.10,
        "cache_write": 1.25,
    },
    "claude-sonnet-4-6": {
        "input": 3.00,
        "output": 15.00,
        "cache_read": 0.30,
        "cache_write": 3.75,
    },
    # Fallback for any model not listed — Sonnet pricing, conservative
    "_default": {
        "input": 3.00,
        "output": 15.00,
        "cache_read": 0.30,
        "cache_write": 3.75,
    },
}


@dataclass
class CallUsage:
    """Token + cost record for a single API call."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    @property
    def usd_cost(self) -> float:
        prices = PRICING.get(self.model, PRICING["_default"])
        return (
            self.input_tokens * prices["input"]
            + self.output_tokens * prices["output"]
            + self.cache_read_tokens * prices["cache_read"]
            + self.cache_write_tokens * prices["cache_write"]
        ) / 1_000_000


def extract_usage(response: Any, model: str) -> CallUsage:
    """Pull token counts from an Anthropic Messages response.

    The SDK exposes usage on response.usage with:
      input_tokens, output_tokens,
      cache_creation_input_tokens, cache_read_input_tokens
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        logger.warning("No usage info on response for %s", model)
        return CallUsage(model=model)

    return CallUsage(
        model=model,
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )


# ─── Per-email accumulator ──────────────────────────────────────────────


@dataclass
class UsageAccumulator:
    """Rolling total of CallUsage records within a logical unit of work."""

    calls: list[CallUsage] = field(default_factory=list)

    def add(self, usage: CallUsage) -> None:
        self.calls.append(usage)

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls)

    @property
    def total_usd(self) -> float:
        return sum(c.usd_cost for c in self.calls)

    def models_used(self) -> list[str]:
        seen: dict[str, int] = {}
        for c in self.calls:
            seen[c.model] = seen.get(c.model, 0) + 1
        return [f"{m} x{n}" for m, n in seen.items()]


# ─── Thread-local active accumulator ────────────────────────────────────
#
# Production option: explicitly thread an accumulator through every call.
# Pragmatic option for this project: thread-local active accumulator so
# classifier/drafter/risk modules don't need their signatures changed.
# Per-email graph invocations are single-threaded in our setup, so this
# is safe.

_thread_local = threading.local()


def _active() -> UsageAccumulator | None:
    return getattr(_thread_local, "accumulator", None)


def record(usage: CallUsage) -> None:
    """Record a call against the active accumulator, if any."""
    acc = _active()
    if acc is not None:
        acc.add(usage)


@contextmanager
def measure() -> Iterator[UsageAccumulator]:
    """Context manager: collect all CallUsage records within the block."""
    acc = UsageAccumulator()
    prev = _active()
    _thread_local.accumulator = acc
    try:
        yield acc
    finally:
        _thread_local.accumulator = prev

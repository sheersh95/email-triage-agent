"""LangGraph state definition.

We use a per-email graph (one invocation per email) rather than a batch
graph. Simpler reasoning, easier to debug, trivially parallelizable
later via asyncio.gather or LangGraph's Send API.

The orchestration script loops over emails and invokes the graph for each.
"""
from __future__ import annotations

from typing import TypedDict

from src.models import (
    ActionTaken,
    Classification,
    Draft,
    Email,
    RiskAssessment,
)


class TriageState(TypedDict, total=False):
    """State passed between nodes for a single email.

    `total=False` makes all fields optional — nodes populate them as the
    flow progresses. The classifier node sets `classification`; the drafter
    sets `draft`; the risk node sets `risk`; the action nodes set `action`.
    """

    # Input
    email: Email
    user_email: str  # for risk assessment domain check
    user_context: str  # for drafter style

    # Populated by classify node
    classification: Classification

    # Populated by draft node (only for needs_reply_* and urgent)
    draft: Draft

    # Populated by risk_assess node (only when there's a draft to assess)
    risk: RiskAssessment

    # Populated by action node (terminal)
    action: ActionTaken

    # Telemetry
    latency_seconds: float
    error: str

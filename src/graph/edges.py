"""Conditional edge functions for LangGraph routing.

Each returns the name of the next node based on state.
Keeping routing logic out of the nodes themselves makes the graph
shape readable as a single picture.
"""
from __future__ import annotations

from src.graph.state import TriageState
from src.models import Category, RiskLevel


def route_after_classify(state: TriageState) -> str:
    """Branch on category. Returns next node name."""
    category = state["classification"].category

    if category == Category.SPAM_OR_PROMO:
        return "archive"
    if category == Category.FYI:
        return "label_fyi"
    # URGENT, NEEDS_REPLY_LOW, NEEDS_REPLY_HIGH all need a draft
    return "draft"


def route_after_risk(state: TriageState) -> str:
    """After risk assessment, decide auto_send vs approval.

    Special case: urgent emails always go through approval, even if
    risk is low, because urgency demands human awareness.
    """
    category = state["classification"].category
    risk_level = state["risk"].level

    if category == Category.URGENT:
        return "urgent"

    if category == Category.NEEDS_REPLY_HIGH:
        # High-stakes always queued regardless of risk signals
        return "approval"

    # NEEDS_REPLY_LOW: risk gates it
    return "auto_send" if risk_level == RiskLevel.LOW else "approval"

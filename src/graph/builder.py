"""Build the LangGraph state machine.

Graph shape:

    START
      │
      ▼
    classify ───────────┐
      │                 │
      │ (category)      │
      ▼                 │
    ┌─────────┬─────────┤
    │         │         │
    │         │         ▼
    │         │       draft
    │         │         │
    │         │         ▼
    │         │       risk_assess
    │         │         │
    │         │   ┌─────┼──────┬─────┐
    │         │   │     │      │     │
    ▼         ▼   ▼     ▼      ▼     ▼
  archive  label_fyi  urgent  approval auto_send
    │         │         │       │       │
    └─────────┴─────────┴───────┴───────┘
                       │
                       ▼
                      END
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.graph.edges import route_after_classify, route_after_risk
from src.graph.nodes import (
    approval_action,
    archive_action,
    auto_send_action,
    classify_node,
    draft_node,
    label_fyi_action,
    risk_node,
    urgent_action,
)
from src.graph.state import TriageState


def build_graph():
    """Construct and compile the triage graph."""
    g = StateGraph(TriageState)

    # Add nodes
    g.add_node("classify", classify_node)
    g.add_node("draft", draft_node)
    g.add_node("risk_assess", risk_node)
    g.add_node("archive", archive_action)
    g.add_node("label_fyi", label_fyi_action)
    g.add_node("urgent", urgent_action)
    g.add_node("approval", approval_action)
    g.add_node("auto_send", auto_send_action)

    # Entry
    g.add_edge(START, "classify")

    # Branch after classify: archive/label_fyi terminate; others go to draft
    g.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "archive": "archive",
            "label_fyi": "label_fyi",
            "draft": "draft",
        },
    )

    # Linear: draft -> risk_assess
    g.add_edge("draft", "risk_assess")

    # Branch after risk: urgent, approval, or auto_send
    g.add_conditional_edges(
        "risk_assess",
        route_after_risk,
        {
            "urgent": "urgent",
            "approval": "approval",
            "auto_send": "auto_send",
        },
    )

    # All action nodes terminate
    for terminal in ("archive", "label_fyi", "urgent", "approval", "auto_send"):
        g.add_edge(terminal, END)

    return g.compile()

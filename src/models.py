"""Pydantic models for type-safe email handling."""
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class Category(str, Enum):
    """Email triage categories. Order matters for some UI display logic."""

    SPAM_OR_PROMO = "spam_or_promo"
    FYI = "fyi"
    NEEDS_REPLY_LOW = "needs_reply_low"
    NEEDS_REPLY_HIGH = "needs_reply_high"
    URGENT = "urgent"


# Human-readable descriptions — single source of truth used in both
# the classifier prompt AND the labeling CLI. Changing the taxonomy
# = changing this dict only.
CATEGORY_DESCRIPTIONS: dict[Category, str] = {
    Category.SPAM_OR_PROMO: (
        "Promotional emails, newsletters, cold sales pitches, marketing. "
        "No reply needed; safe to archive."
    ),
    Category.FYI: (
        "Informational: receipts, shipping updates, calendar invites, "
        "automated notifications, system alerts. No reply needed."
    ),
    Category.NEEDS_REPLY_LOW: (
        "Casual messages needing a reply but with low stakes: scheduling "
        "confirmations, simple questions from known contacts, FYI-style "
        "replies. Safe to auto-send a draft."
    ),
    Category.NEEDS_REPLY_HIGH: (
        "Important messages requiring careful reply: anything involving "
        "money/payments, legal/HR, unfamiliar senders making requests, "
        "commitments, sensitive relationships. Always needs human review."
    ),
    Category.URGENT: (
        "Time-sensitive and requires immediate attention. Often from a "
        "boss, customer, or about an active incident. Notify the user "
        "immediately regardless of reply complexity."
    ),
}


class Classification(BaseModel):
    """Output of the classifier node.

    `confidence` is a categorical bucket because LLMs can't produce
    well-calibrated numerical confidence. Low-confidence outputs get
    routed to human review even within auto-send-eligible categories.
    """

    category: Category
    confidence: str = Field(..., pattern="^(high|medium|low)$")
    reasoning: str = Field(..., description="1-2 sentence justification")


class RiskLevel(str, Enum):
    """Risk level for a drafted reply. High → human approval required."""

    LOW = "low"
    HIGH = "high"


class RiskAssessment(BaseModel):
    """Output of risk_assess node. Both the level and which signals fired."""

    level: RiskLevel
    signals: list[str] = Field(
        default_factory=list,
        description="Names of hard rules that fired, e.g. 'money_terms'.",
    )
    llm_reasoning: str | None = Field(
        None, description="LLM tiebreaker reasoning if rules were ambiguous."
    )


class Draft(BaseModel):
    """A drafted reply. body is plaintext; subject auto-prefixed with 'Re:'."""

    subject: str
    body: str
    in_reply_to: str = Field(..., description="Original email ID being replied to")


class ActionTaken(str, Enum):
    """What the agent ultimately did with the email. One per email."""

    ARCHIVED = "archived"
    LABELED_FYI = "labeled_fyi"
    DRAFTED_AUTO_SEND = "drafted_auto_send"
    DRAFTED_NEEDS_APPROVAL = "drafted_needs_approval"
    NOTIFIED_URGENT = "notified_urgent"
    SKIPPED = "skipped"


class AuditRecord(BaseModel):
    """One row per email processed. Becomes the audit log + UI data source."""

    email_id: str
    thread_id: str
    sender_email: str
    subject: str
    classification: Category
    confidence: str
    classification_reasoning: str
    draft_body: str | None = None
    risk_level: RiskLevel | None = None
    risk_signals: list[str] = Field(default_factory=list)
    action: ActionTaken
    processed_at: datetime
    model_classifier: str
    model_drafter: str | None = None
    latency_seconds: float
    error: str | None = None
    # Day 5: cost tracking
    input_tokens: int = 0
    output_tokens: int = 0
    usd_cost: float = 0.0


class Email(BaseModel):
    """Normalized email representation.

    We extract a flat structure from Gmail's nested payload because every
    downstream node (classify, draft, risk) only needs these fields. Keeps
    prompts cheap and tests simple.
    """

    id: str = Field(..., description="Gmail message ID")
    thread_id: str = Field(..., description="Gmail thread ID for context")
    sender: str = Field(..., description="From header, raw")
    sender_email: str = Field(..., description="Just the email address")
    sender_name: str | None = Field(None, description="Display name if present")
    recipient: str = Field(..., description="To header")
    subject: str = Field(default="(no subject)")
    body: str = Field(..., description="Plaintext body, cleaned")
    snippet: str = Field(default="", description="Gmail's own short preview")
    received_at: datetime
    labels: list[str] = Field(default_factory=list, description="Gmail labels")
    is_unread: bool = True

    def short_repr(self) -> str:
        """One-line summary for logs."""
        who = self.sender_name or self.sender_email
        return f"[{self.received_at:%Y-%m-%d %H:%M}] {who}: {self.subject[:60]}"

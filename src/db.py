"""SQLite audit log for processed emails.

One row per email. Stores everything needed for:
- Day 4 UI (approval queue, processed inbox, audit trail)
- Eval comparisons (did model X make different decisions than model Y?)
- Cost/latency tracking over time

Schema is denormalized on purpose — single-table reads are simpler and
the row count will never be large (thousands, not millions).
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from src.config import DB_PATH
from src.models import ActionTaken, AuditRecord, Category, RiskLevel

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    email_id              TEXT PRIMARY KEY,
    thread_id             TEXT NOT NULL,
    sender_email          TEXT NOT NULL,
    subject               TEXT NOT NULL,
    classification        TEXT NOT NULL,
    confidence            TEXT NOT NULL,
    classification_reasoning TEXT NOT NULL,
    draft_body            TEXT,
    risk_level            TEXT,
    risk_signals_json     TEXT NOT NULL DEFAULT '[]',
    action                TEXT NOT NULL,
    processed_at          TEXT NOT NULL,
    model_classifier      TEXT NOT NULL,
    model_drafter         TEXT,
    latency_seconds       REAL NOT NULL,
    error                 TEXT,
    -- Approval workflow (Day 4)
    approval_status       TEXT DEFAULT NULL,
    approved_at           TEXT,
    sent_at               TEXT,
    final_body            TEXT,
    -- Cost tracking (Day 5)
    input_tokens          INTEGER NOT NULL DEFAULT 0,
    output_tokens         INTEGER NOT NULL DEFAULT 0,
    usd_cost              REAL NOT NULL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_action ON audit_log(action);
CREATE INDEX IF NOT EXISTS idx_approval ON audit_log(approval_status);
CREATE INDEX IF NOT EXISTS idx_processed ON audit_log(processed_at DESC);
"""

# Safe migrations — add new columns to old DBs without losing data.
# Run in order; each is a no-op if the column already exists.
_MIGRATIONS = [
    "ALTER TABLE audit_log ADD COLUMN input_tokens INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE audit_log ADD COLUMN output_tokens INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE audit_log ADD COLUMN usd_cost REAL NOT NULL DEFAULT 0.0",
]


def _run_migrations(conn: sqlite3.Connection) -> None:
    """Apply ALTER TABLE migrations idempotently."""
    for stmt in _MIGRATIONS:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as e:
            # 'duplicate column' is expected on already-migrated DBs
            if "duplicate column" not in str(e).lower():
                raise


@contextmanager
def _conn(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    """Context manager: opens connection, ensures schema, commits/closes."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        _run_migrations(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def insert_record(record: AuditRecord, db_path: Path = DB_PATH) -> None:
    """Insert one audit record. Updates if email_id already exists.

    Setting approval_status to 'pending' on insert for drafts that need
    review — drives the Day 4 approval queue.
    """
    approval_status: str | None = None
    sent_at: str | None = None
    if record.action == ActionTaken.DRAFTED_NEEDS_APPROVAL:
        approval_status = "pending"
    elif record.action == ActionTaken.DRAFTED_AUTO_SEND:
        # If we got here it means the graph's auto_send node actually sent.
        # Mark as sent immediately so UI doesn't queue it for re-approval.
        approval_status = "sent"
        sent_at = record.processed_at.isoformat()
    elif record.action == ActionTaken.NOTIFIED_URGENT:
        # Urgent gets a draft but always needs human eyes
        approval_status = "pending"
    elif record.action in (ActionTaken.ARCHIVED, ActionTaken.LABELED_FYI):
        # Spam/FYI: draft-only mode means user confirms in UI;
        # they show as 'pending' so they appear in approval queue with
        # a 'confirm archive/label' button.
        approval_status = "pending"

    with _conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO audit_log (
                email_id, thread_id, sender_email, subject,
                classification, confidence, classification_reasoning,
                draft_body, risk_level, risk_signals_json,
                action, processed_at, model_classifier, model_drafter,
                latency_seconds, error, approval_status, sent_at,
                input_tokens, output_tokens, usd_cost
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(email_id) DO UPDATE SET
                classification = excluded.classification,
                confidence = excluded.confidence,
                classification_reasoning = excluded.classification_reasoning,
                draft_body = excluded.draft_body,
                risk_level = excluded.risk_level,
                risk_signals_json = excluded.risk_signals_json,
                action = excluded.action,
                processed_at = excluded.processed_at,
                model_classifier = excluded.model_classifier,
                model_drafter = excluded.model_drafter,
                latency_seconds = excluded.latency_seconds,
                error = excluded.error,
                input_tokens = excluded.input_tokens,
                output_tokens = excluded.output_tokens,
                usd_cost = excluded.usd_cost
            """,
            (
                record.email_id,
                record.thread_id,
                record.sender_email,
                record.subject,
                record.classification.value,
                record.confidence,
                record.classification_reasoning,
                record.draft_body,
                record.risk_level.value if record.risk_level else None,
                json.dumps(record.risk_signals),
                record.action.value,
                record.processed_at.isoformat(),
                record.model_classifier,
                record.model_drafter,
                record.latency_seconds,
                record.error,
                approval_status,
                sent_at,
                record.input_tokens,
                record.output_tokens,
                record.usd_cost,
            ),
        )


def list_recent(limit: int = 50, db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    """Return recent audit rows for UI display."""
    with _conn(db_path) as conn:
        cursor = conn.execute(
            "SELECT * FROM audit_log ORDER BY processed_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]


def list_pending_approval(db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    """Return drafts awaiting human approval."""
    with _conn(db_path) as conn:
        cursor = conn.execute(
            "SELECT * FROM audit_log WHERE approval_status = 'pending' "
            "ORDER BY processed_at DESC"
        )
        return [dict(row) for row in cursor.fetchall()]


def already_processed(email_id: str, db_path: Path = DB_PATH) -> bool:
    """Check if we've seen this email before. Used to skip in fetch loop."""
    with _conn(db_path) as conn:
        cursor = conn.execute(
            "SELECT 1 FROM audit_log WHERE email_id = ? LIMIT 1", (email_id,)
        )
        return cursor.fetchone() is not None


def get_record(email_id: str, db_path: Path = DB_PATH) -> dict[str, Any] | None:
    """Fetch one row by email_id."""
    with _conn(db_path) as conn:
        cursor = conn.execute(
            "SELECT * FROM audit_log WHERE email_id = ?", (email_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def mark_sent(
    email_id: str, final_body: str, db_path: Path = DB_PATH
) -> None:
    """Record that the draft was sent (after Gmail API call succeeded).

    `final_body` is the post-edit body — user may have modified the draft
    before approving. We store both the original draft and the final.
    """
    sent_at = datetime.now(tz=__import__("datetime").timezone.utc).isoformat()
    with _conn(db_path) as conn:
        conn.execute(
            """
            UPDATE audit_log
               SET approval_status = 'sent',
                   sent_at = ?,
                   final_body = ?
             WHERE email_id = ?
            """,
            (sent_at, final_body, email_id),
        )


def mark_rejected(email_id: str, db_path: Path = DB_PATH) -> None:
    """Mark a draft as rejected — user decided not to send it."""
    rejected_at = datetime.now(tz=__import__("datetime").timezone.utc).isoformat()
    with _conn(db_path) as conn:
        conn.execute(
            """
            UPDATE audit_log
               SET approval_status = 'rejected',
                   approved_at = ?
             WHERE email_id = ?
            """,
            (rejected_at, email_id),
        )


def mark_archived(email_id: str, db_path: Path = DB_PATH) -> None:
    """Mark a spam/promo as archived in Gmail."""
    ts = datetime.now(tz=__import__("datetime").timezone.utc).isoformat()
    with _conn(db_path) as conn:
        conn.execute(
            "UPDATE audit_log SET approval_status='completed', sent_at=? "
            "WHERE email_id = ?",
            (ts, email_id),
        )


def mark_labeled(email_id: str, db_path: Path = DB_PATH) -> None:
    """Mark an FYI as labeled in Gmail."""
    mark_archived(email_id, db_path)  # same status update


def get_stats(db_path: Path = DB_PATH) -> dict[str, Any]:
    """Aggregate stats for the dashboard tab."""
    with _conn(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]

        by_action = dict(
            conn.execute(
                "SELECT action, COUNT(*) FROM audit_log GROUP BY action"
            ).fetchall()
        )

        by_status = dict(
            conn.execute(
                "SELECT COALESCE(approval_status, 'n/a'), COUNT(*) "
                "FROM audit_log GROUP BY approval_status"
            ).fetchall()
        )

        # Cost-relevant: avg latency, total emails per model
        avg_latency = (
            conn.execute(
                "SELECT AVG(latency_seconds) FROM audit_log"
            ).fetchone()[0]
            or 0.0
        )

        sent_count = conn.execute(
            "SELECT COUNT(*) FROM audit_log WHERE approval_status = 'sent'"
        ).fetchone()[0]

        # Cost aggregates
        cost_row = conn.execute(
            """
            SELECT
                COALESCE(SUM(usd_cost), 0)            AS total_usd,
                COALESCE(SUM(input_tokens), 0)         AS total_input,
                COALESCE(SUM(output_tokens), 0)        AS total_output,
                COALESCE(AVG(usd_cost), 0)             AS avg_usd
            FROM audit_log
            """
        ).fetchone()

    return {
        "total": total,
        "by_action": by_action,
        "by_status": by_status,
        "avg_latency_seconds": round(avg_latency, 2),
        "sent_count": sent_count,
        "total_usd": round(cost_row["total_usd"], 4),
        "avg_usd_per_email": round(cost_row["avg_usd"], 5),
        "total_input_tokens": cost_row["total_input"],
        "total_output_tokens": cost_row["total_output"],
    }

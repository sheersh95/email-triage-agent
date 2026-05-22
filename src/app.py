"""Streamlit UI for the email triage agent.

Tabs:
1. Approval Queue — pending drafts, edit + send/reject
2. Inbox — all processed emails (read-only view)
3. Audit Log — every decision with full context
4. Stats — counts, latency, action breakdown

Top bar:
- Auto-send toggle (off by default — safest)
- "Process N new emails" button to trigger a fresh fetch+graph run

Run from project root:
    streamlit run src/app.py
"""
from __future__ import annotations

# Bootstrap: ensure project root is on sys.path so `from src.X import Y`
# works regardless of where Streamlit was launched from. This file lives
# at <project_root>/src/app.py, so project_root = parent of parent.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
import logging
import time
from datetime import datetime, timezone

import streamlit as st

from src.classifier import CLASSIFIER_MODEL
from src.config import (
    DEFAULT_FETCH_LIMIT,
    DEFAULT_QUERY,
    is_auto_send_enabled,
    set_auto_send,
)
from src.db import (
    already_processed,
    get_record,
    get_stats,
    insert_record,
    list_pending_approval,
    list_recent,
    mark_archived,
    mark_labeled,
    mark_rejected,
    mark_sent,
)
from src.drafter import DRAFTER_MODEL
from src.gmail_auth import build_gmail_service
from src.gmail_client import fetch_emails
from src.graph.builder import build_graph
from src.models import ActionTaken, AuditRecord, Email
from src.tools.gmail_actions import (
    apply_fyi_label,
    archive_message,
    send_reply,
)

logging.basicConfig(level=logging.INFO)

# ─── Page config ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Email Triage Agent",
    page_icon="✉️",
    layout="wide",
)

# ─── Cached resources ───────────────────────────────────────────────────


@st.cache_resource
def get_service():
    """Gmail service cached for the session. Re-auth requires app restart."""
    return build_gmail_service()


@st.cache_resource
def get_user_email() -> str:
    profile = get_service().users().getProfile(userId="me").execute()
    return profile.get("emailAddress", "")


@st.cache_resource
def get_graph():
    return build_graph()


# ─── Color/icon helpers ─────────────────────────────────────────────────

CATEGORY_COLORS = {
    "spam_or_promo": "🗑️",
    "fyi": "📋",
    "needs_reply_low": "💬",
    "needs_reply_high": "⚠️",
    "urgent": "🚨",
}

RISK_BADGE = {"low": "🟢 low", "high": "🔴 high"}
STATUS_BADGE = {
    "pending": "⏳ pending",
    "sent": "✅ sent",
    "rejected": "❌ rejected",
    "completed": "✓ completed",
}


def fmt_dt(iso_str: str | None) -> str:
    if not iso_str:
        return "—"
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return iso_str


# ─── Action handlers (called by buttons) ────────────────────────────────


def handle_send(record: dict, edited_body: str) -> None:
    """Send a reply via Gmail, update DB. Optimistic UI: update DB first?
    No — Gmail send is the source of truth. If Gmail fails, DB stays pending.
    """
    try:
        send_reply(
            get_service(),
            thread_id=record["thread_id"],
            in_reply_to_message_id=record["email_id"],
            to_address=record["sender_email"],
            subject=f"Re: {record['subject']}" if not record["subject"].lower().startswith("re:") else record["subject"],
            body=edited_body,
        )
        mark_sent(record["email_id"], edited_body)
        st.toast(f"✅ Sent to {record['sender_email']}", icon="✅")
    except Exception as e:
        st.error(f"Send failed: {e}")


def handle_reject(email_id: str) -> None:
    mark_rejected(email_id)
    st.toast("Draft rejected", icon="❌")


def handle_archive(email_id: str) -> None:
    try:
        archive_message(get_service(), email_id)
        mark_archived(email_id)
        st.toast("Archived in Gmail", icon="🗑️")
    except Exception as e:
        st.error(f"Archive failed: {e}")


def handle_label_fyi(email_id: str) -> None:
    try:
        apply_fyi_label(get_service(), email_id)
        mark_labeled(email_id)
        st.toast("Labeled Triage/FYI in Gmail", icon="📋")
    except Exception as e:
        st.error(f"Label failed: {e}")


def process_new_emails(query: str, limit: int) -> tuple[int, int]:
    """Fetch and run graph on emails not yet in DB. Returns (processed, skipped)."""
    from src.usage_meter import measure

    service = get_service()
    user_email = get_user_email()
    graph = get_graph()

    emails = fetch_emails(service, query=query, limit=limit)
    new_emails = [e for e in emails if not already_processed(e.id)]
    skipped = len(emails) - len(new_emails)

    progress = st.progress(0.0, text="Processing...")
    for i, email in enumerate(new_emails, 1):
        progress.progress(
            i / max(len(new_emails), 1),
            text=f"[{i}/{len(new_emails)}] {email.subject[:60]}",
        )
        start = time.time()
        error: str | None = None
        final_state: dict = {}

        # Per-email cost measurement — all LLM calls inside the graph
        # auto-register against this accumulator.
        with measure() as usage:
            try:
                final_state = graph.invoke(
                    {"email": email, "user_email": user_email, "user_context": ""}
                )
            except Exception as e:
                error = f"{type(e).__name__}: {e}"

        elapsed = time.time() - start
        classification = final_state.get("classification")
        draft = final_state.get("draft")
        risk = final_state.get("risk")
        action = final_state.get("action", ActionTaken.SKIPPED)

        record = AuditRecord(
            email_id=email.id,
            thread_id=email.thread_id,
            sender_email=email.sender_email,
            subject=email.subject,
            classification=classification.category if classification else "fyi",
            confidence=classification.confidence if classification else "low",
            classification_reasoning=(
                classification.reasoning if classification else "(failed)"
            ),
            draft_body=draft.body if draft else None,
            risk_level=risk.level if risk else None,
            risk_signals=risk.signals if risk else [],
            action=action,
            processed_at=datetime.now(timezone.utc),
            model_classifier=CLASSIFIER_MODEL,
            model_drafter=DRAFTER_MODEL if draft else None,
            latency_seconds=round(elapsed, 2),
            error=error,
            input_tokens=usage.total_input_tokens,
            output_tokens=usage.total_output_tokens,
            usd_cost=usage.total_usd,
        )
        insert_record(record)

    progress.empty()
    return len(new_emails), skipped


# ─── Top bar ────────────────────────────────────────────────────────────

st.title("✉️ Email Triage Agent")

with st.container():
    col1, col2, col3, col4 = st.columns([2, 2, 1.5, 1.5])

    with col1:
        try:
            user_email = get_user_email()
            st.caption(f"📧 {user_email}")
        except Exception as e:
            st.error(f"Auth error: {e}")
            st.stop()

    with col2:
        query = st.text_input(
            "Gmail query",
            value=DEFAULT_QUERY,
            label_visibility="collapsed",
            placeholder=DEFAULT_QUERY,
        )

    with col3:
        limit = st.number_input(
            "Limit", min_value=1, max_value=100, value=DEFAULT_FETCH_LIMIT,
            label_visibility="collapsed",
        )

    with col4:
        if st.button("🔄 Process new", use_container_width=True, type="primary"):
            with st.spinner("Running triage graph..."):
                processed, skipped = process_new_emails(query, int(limit))
            st.success(
                f"Processed {processed} new emails (skipped {skipped} already-seen)"
            )
            st.rerun()

# Auto-send toggle — visually distinct, defaults off
auto_send = st.toggle(
    "⚡ Auto-send enabled (low-risk drafts send without approval)",
    value=is_auto_send_enabled(),
    help=(
        "When OFF: every draft queues for your review. "
        "When ON: drafts marked low-risk send automatically; "
        "high-risk still requires approval."
    ),
)
if auto_send != is_auto_send_enabled():
    set_auto_send(auto_send)
    if auto_send:
        st.warning(
            "⚠️ Auto-send is ON. Future low-risk drafts will send automatically."
        )

st.divider()

# ─── Tabs ───────────────────────────────────────────────────────────────

tab_approval, tab_inbox, tab_audit, tab_stats, tab_settings = st.tabs(
    ["📥 Approval Queue", "📬 Inbox", "📜 Audit Log", "📊 Stats", "⚙️ Settings"]
)

# ─── Approval Queue tab ─────────────────────────────────────────────────

with tab_approval:
    pending = list_pending_approval()
    st.subheader(f"Pending review: {len(pending)}")

    if not pending:
        st.info("No drafts pending review. Hit 'Process new' to triage your inbox.")
    else:
        for record in pending:
            cat_icon = CATEGORY_COLORS.get(record["classification"], "")
            risk_str = RISK_BADGE.get(record["risk_level"] or "", "—")

            with st.expander(
                f"{cat_icon}  **{record['subject'][:80]}**  "
                f"— from {record['sender_email']}  ·  {risk_str}",
                expanded=False,
            ):
                meta_col, _ = st.columns([3, 1])
                with meta_col:
                    st.caption(
                        f"**{record['classification']}** "
                        f"({record['confidence']} confidence) · "
                        f"processed {fmt_dt(record['processed_at'])}"
                    )
                    st.caption(
                        f"_{record['classification_reasoning']}_"
                    )
                    signals = json.loads(record.get("risk_signals_json", "[]"))
                    if signals:
                        st.caption(
                            f"🚩 Risk signals: {', '.join(signals)}"
                        )

                if record["draft_body"]:
                    # Editable text area — user can refine before sending
                    edited = st.text_area(
                        "Draft (editable):",
                        value=record["draft_body"],
                        height=200,
                        key=f"draft_{record['email_id']}",
                    )

                    btn_col1, btn_col2, btn_col3, _ = st.columns([1, 1, 1, 3])
                    with btn_col1:
                        if st.button(
                            "✅ Send",
                            key=f"send_{record['email_id']}",
                            type="primary",
                        ):
                            handle_send(record, edited)
                            st.rerun()
                    with btn_col2:
                        if st.button(
                            "❌ Reject", key=f"rej_{record['email_id']}"
                        ):
                            handle_reject(record["email_id"])
                            st.rerun()
                    with btn_col3:
                        if st.button(
                            "💾 Save draft", key=f"save_{record['email_id']}",
                            help="Persist edits without sending"
                        ):
                            # Save edited draft body back to draft_body
                            from src.db import _conn  # internal but fine for now
                            with _conn() as conn:
                                conn.execute(
                                    "UPDATE audit_log SET draft_body = ? "
                                    "WHERE email_id = ?",
                                    (edited, record["email_id"]),
                                )
                            st.toast("Draft saved", icon="💾")
                else:
                    # Non-draft pending: archive (spam) or label (FYI)
                    action = record["action"]
                    if action == "archived":
                        if st.button(
                            "🗑️ Confirm archive in Gmail",
                            key=f"arch_{record['email_id']}",
                        ):
                            handle_archive(record["email_id"])
                            st.rerun()
                    elif action == "labeled_fyi":
                        if st.button(
                            "📋 Apply Triage/FYI label",
                            key=f"lbl_{record['email_id']}",
                        ):
                            handle_label_fyi(record["email_id"])
                            st.rerun()

# ─── Inbox tab ──────────────────────────────────────────────────────────

with tab_inbox:
    st.subheader("All processed emails")
    rows = list_recent(limit=200)

    if not rows:
        st.info("No emails processed yet.")
    else:
        # Render as a compact table
        for row in rows:
            cat_icon = CATEGORY_COLORS.get(row["classification"], "")
            status = STATUS_BADGE.get(
                row["approval_status"] or "n/a", row["approval_status"] or "—"
            )
            with st.container():
                c1, c2, c3, c4, c5 = st.columns([0.4, 3, 2, 1.5, 1.5])
                c1.markdown(f"### {cat_icon}")
                c2.markdown(f"**{row['subject'][:70]}**")
                c2.caption(row["sender_email"])
                c3.caption(f"{row['classification']} ({row['confidence']})")
                c4.caption(status)
                c5.caption(fmt_dt(row["processed_at"]))
                st.divider()

# ─── Audit Log tab ──────────────────────────────────────────────────────

with tab_audit:
    st.subheader("Audit trail")
    st.caption(
        "Every decision the agent made: classification reasoning, "
        "risk signals, model used, latency. The receipts."
    )

    rows = list_recent(limit=100)
    if not rows:
        st.info("No records yet.")
    else:
        for row in rows:
            signals = json.loads(row.get("risk_signals_json", "[]"))
            cat_icon = CATEGORY_COLORS.get(row["classification"], "")
            header = (
                f"{cat_icon} {row['subject'][:60]} — {row['action']}"
            )
            with st.expander(header):
                st.json(
                    {
                        "email_id": row["email_id"],
                        "from": row["sender_email"],
                        "classification": row["classification"],
                        "confidence": row["confidence"],
                        "classification_reasoning": row["classification_reasoning"],
                        "risk_level": row["risk_level"],
                        "risk_signals": signals,
                        "action": row["action"],
                        "approval_status": row["approval_status"],
                        "model_classifier": row["model_classifier"],
                        "model_drafter": row["model_drafter"],
                        "latency_seconds": row["latency_seconds"],
                        "processed_at": fmt_dt(row["processed_at"]),
                        "sent_at": fmt_dt(row["sent_at"]),
                        "error": row["error"],
                    }
                )

# ─── Stats tab ──────────────────────────────────────────────────────────

with tab_stats:
    st.subheader("Agent stats")
    stats = get_stats()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total processed", stats["total"])
    c2.metric("Replies sent", stats["sent_count"])
    c3.metric("Avg latency", f"{stats['avg_latency_seconds']}s")
    pending_count = stats["by_status"].get("pending", 0)
    c4.metric("Pending review", pending_count)

    # Cost row
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total spend", f"${stats['total_usd']:.4f}")
    c6.metric("Avg / email", f"${stats['avg_usd_per_email']:.5f}")
    c7.metric("Input tokens", f"{stats['total_input_tokens']:,}")
    c8.metric("Output tokens", f"{stats['total_output_tokens']:,}")

    if stats["total"] > 0:
        # Extrapolate to a year of email volume — interview talking point
        annual_at_50_per_day = stats["avg_usd_per_email"] * 50 * 365
        st.caption(
            f"💡 At your current avg cost/email, processing 50 emails/day "
            f"would cost ~**${annual_at_50_per_day:.2f}/year**"
        )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**By action**")
        if stats["by_action"]:
            for action, count in sorted(
                stats["by_action"].items(), key=lambda x: -x[1]
            ):
                st.caption(f"{action}: {count}")
        else:
            st.caption("—")

    with col2:
        st.markdown("**By approval status**")
        if stats["by_status"]:
            for status, count in sorted(
                stats["by_status"].items(), key=lambda x: -x[1]
            ):
                st.caption(f"{status}: {count}")
        else:
            st.caption("—")

# ─── Settings tab ──────────────────────────────────────────────────────

with tab_settings:
    st.subheader("Writing style profile")
    st.caption(
        "Extracted from your sent emails. Injected into the drafter so "
        "replies sound like you, not like a generic LLM."
    )

    from src.tools.style import load_profile, refresh_profile

    profile = load_profile()
    if profile:
        st.success(
            f"✓ Profile loaded — extracted from {profile.sample_count} sent emails "
            f"on {profile.extracted_at[:10]}"
        )
        with st.expander("View profile", expanded=False):
            st.code(profile.to_prompt_block(), language="text")
    else:
        st.info("No style profile yet. Extract one to make drafts sound like you.")

    if st.button("🔄 Re-extract style profile", type="primary"):
        with st.spinner("Reading your sent emails and analyzing style..."):
            try:
                new_profile = refresh_profile(get_service())
                st.success(
                    f"Extracted profile from {new_profile.sample_count} sent emails."
                )
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")

    st.divider()
    st.subheader("Database")
    st.caption(f"Audit log: `{stats['total']}` rows  ·  total spend: `${stats['total_usd']:.4f}`")
    if st.button("⚠️ Clear all processed records", type="secondary"):
        from src.db import _conn
        with _conn() as conn:
            conn.execute("DELETE FROM audit_log")
        st.success("Audit log cleared.")
        st.rerun()

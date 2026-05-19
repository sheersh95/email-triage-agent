"""Configuration constants for the email triage agent."""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
CREDENTIALS_PATH = PROJECT_ROOT / "credentials.json"
TOKEN_PATH = PROJECT_ROOT / "token.json"
DB_PATH = PROJECT_ROOT / "triage.db"

# Gmail API scopes — incrementally expanded as features need them.
# Day 1: readonly. Day 4: modify (archive/label) + send.
# Changing this list invalidates token.json — auth code auto-detects this.
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",  # archive + label
    "https://www.googleapis.com/auth/gmail.send",    # send drafts
]

# Fetch settings
DEFAULT_FETCH_LIMIT = 20
DEFAULT_QUERY = "is:unread in:inbox"  # Gmail search syntax

# Runtime mode — controlled by UI toggle in Streamlit.
# When AUTO_SEND_ENABLED, low-risk drafts go straight to Gmail send.
# When disabled (default), every draft is queued for human approval.
# Stored as a file flag so it persists across UI reruns; UI toggle updates it.
AUTO_SEND_FLAG_PATH = PROJECT_ROOT / ".auto_send_enabled"


def is_auto_send_enabled() -> bool:
    return AUTO_SEND_FLAG_PATH.exists()


def set_auto_send(enabled: bool) -> None:
    if enabled:
        AUTO_SEND_FLAG_PATH.touch()
    elif AUTO_SEND_FLAG_PATH.exists():
        AUTO_SEND_FLAG_PATH.unlink()

"""Gmail OAuth handling.

First run: opens browser → user consents → token.json written to disk.
Subsequent runs: loads token.json, auto-refreshes if expired.

Token contains a refresh_token so we don't re-prompt unless the user
revokes access or scopes change.
"""
from __future__ import annotations

import logging
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build

from src.config import CREDENTIALS_PATH, GMAIL_SCOPES, TOKEN_PATH

logger = logging.getLogger(__name__)


def get_credentials(
    credentials_path: Path = CREDENTIALS_PATH,
    token_path: Path = TOKEN_PATH,
    scopes: list[str] = GMAIL_SCOPES,
) -> Credentials:
    """Load cached creds, refresh if needed, or run full OAuth flow.

    Raises:
        FileNotFoundError: if credentials.json missing on first run.
    """
    creds: Credentials | None = None

    # Try to load cached token
    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), scopes)
            logger.info("Loaded cached credentials from %s", token_path)
        except Exception as e:
            # Most commonly: scopes have changed since token was issued.
            # google-auth raises here. We delete the token and re-auth.
            logger.warning(
                "Cached token unusable (%s). Deleting and re-authenticating.", e
            )
            token_path.unlink()
            creds = None

    # Detect scope expansion vs cached token. The library's
    # Credentials.has_scopes() compares; if any new scope missing, re-auth.
    if creds is not None and not creds.has_scopes(scopes):
        logger.warning("Cached token missing requested scopes. Re-auth required.")
        token_path.unlink()
        creds = None

    # If no creds or expired/invalid → refresh or full flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing expired token")
            creds.refresh(Request())
        else:
            if not credentials_path.exists():
                raise FileNotFoundError(
                    f"Missing {credentials_path}. Download OAuth client JSON "
                    "from Google Cloud Console and save it there."
                )
            logger.info("Running OAuth flow — browser will open")
            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_path), scopes
            )
            # port=0 → pick any free port; opens local redirect server
            creds = flow.run_local_server(port=0)

        # Persist for next run
        token_path.write_text(creds.to_json())
        logger.info("Saved credentials to %s", token_path)

    return creds


def build_gmail_service(creds: Credentials | None = None) -> Resource:
    """Return an authenticated Gmail API client."""
    if creds is None:
        creds = get_credentials()
    # cache_discovery=False suppresses a noisy warning in newer environments
    return build("gmail", "v1", credentials=creds, cache_discovery=False)

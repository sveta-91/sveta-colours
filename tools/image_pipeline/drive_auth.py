from __future__ import annotations

import json
from pathlib import Path

from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/drive"]

TOKEN_PATH = Path(__file__).resolve().parent / "oauth-token.json"


def get_drive_service(key_path: Path):
    """Return an authenticated Google Drive API v3 service (service account — read only)."""
    creds = service_account.Credentials.from_service_account_file(
        str(key_path), scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)


def get_user_drive_service(oauth_client_path: Path):
    """Return a Drive service authenticated as the user (for uploads).

    On first run, opens a browser for Google login. Saves a refresh token
    to oauth-token.json so subsequent runs don't need the browser.
    """
    creds = None

    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(oauth_client_path), SCOPES)
            creds = flow.run_local_server(port=0)

        TOKEN_PATH.write_text(json.dumps({
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": creds.scopes and list(creds.scopes),
        }))

    return build("drive", "v3", credentials=creds)

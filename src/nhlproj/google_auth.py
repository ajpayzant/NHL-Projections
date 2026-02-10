from __future__ import annotations
import os
from google.oauth2.service_account import Credentials
import gspread
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

def get_service_account_credentials():
    path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not path:
        raise RuntimeError(
            "Missing GOOGLE_APPLICATION_CREDENTIALS env var. "
            "Point it to your service account JSON."
        )
    return Credentials.from_service_account_file(path, scopes=SCOPES)

def get_gspread_client():
    creds = get_service_account_credentials()
    return gspread.authorize(creds)

def get_drive_service():
    creds = get_service_account_credentials()
    return build("drive", "v3", credentials=creds)

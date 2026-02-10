from __future__ import annotations
import os
import re
import time
from datetime import datetime, timezone
from typing import Optional, List
import numpy as np
import pandas as pd
import requests

DEFAULT_HEADERS_MP = {
    "User-Agent": "Mozilla/5.0 (compatible; NHLDataBot/1.0)",
    "Accept": "text/csv,application/json,text/html;q=0.9,*/*;q=0.8",
    "Referer": "https://moneypuck.com/data.htm",
}

DEFAULT_HEADERS_NHL = {
    "User-Agent": "Mozilla/5.0 (compatible; NHLLineupBot/1.0)",
    "Accept": "application/json,text/plain,*/*",
}

HREF_RE = re.compile(r'href="([^"]+\.csv)"', re.IGNORECASE)

TEAM_MAP = {"LA": "LAK", "NJ": "NJD", "SJ": "SJS", "TB": "TBL", "WAS": "WSH"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def current_nhl_season_start(today: Optional[datetime] = None) -> int:
    today = today or datetime.now()
    return today.year if today.month >= 7 else today.year - 1


def http_get(url: str, timeout: int = 60, headers: Optional[dict] = None) -> requests.Response:
    h = headers or DEFAULT_HEADERS_MP
    r = requests.get(url, headers=h, timeout=timeout)
    r.raise_for_status()
    return r


def download_with_retries(
    url: str,
    out_path: str,
    max_tries: int = 6,
    force: bool = False,
    backoff: float = 2.0,
    sleep_s: float = 0.15,
    headers: Optional[dict] = None,
) -> bool:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if (not force) and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return False

    last_err = None
    for i in range(max_tries):
        try:
            r = http_get(url, headers=headers)
            with open(out_path, "wb") as f:
                f.write(r.content)
            time.sleep(sleep_s)
            return True
        except Exception as e:
            last_err = e
            time.sleep(backoff ** i)

    raise last_err


def list_csvs_from_index(index_url: str) -> List[str]:
    html = http_get(index_url).text
    files = HREF_RE.findall(html)
    return sorted(set([f for f in files if f.lower().endswith(".csv")]))


def parse_game_date(s):
    if pd.isna(s):
        return pd.NaT
    s = str(s).strip()
    if s.isdigit() and len(s) == 8:
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def ensure_minutes(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    med = np.nanmedian(x.values) if len(x) else np.nan
    if np.isfinite(med) and med > 200:
        return x / 60.0
    return x


def first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def standardize_team_game_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "team" not in df.columns:
        c = first_existing_col(df, ["playerTeam", "teamAbbrev", "teamCode"])
        if c:
            df["team"] = df[c]

    if "team_opp" not in df.columns:
        c = first_existing_col(df, ["opposingTeam", "opponentTeam", "oppTeam", "opponent"])
        if c:
            df["team_opp"] = df[c]

    if "gameId" not in df.columns:
        c = first_existing_col(df, ["game_id", "gamePk", "id"])
        if c:
            df["gameId"] = df[c]

    if "gameDate" in df.columns:
        df["gameDate"] = df["gameDate"].map(parse_game_date)
    else:
        c = first_existing_col(df, ["date"])
        if c:
            df["gameDate"] = df[c].map(parse_game_date)

    if "home_or_away" not in df.columns:
        c = first_existing_col(df, ["homeOrAway", "homeAway"])
        if c:
            df["home_or_away"] = df[c]

    if "situation" not in df.columns:
        df["situation"] = "all"

    return df


def strip_and_dedup_columns(df: pd.DataFrame, label="df") -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if not df.columns.is_unique:
        dupes = df.columns[df.columns.duplicated()].tolist()
        print(f"⚠️ {label}: duplicate columns -> dropping (keep first): {dupes[:20]}")
        df = df.loc[:, ~df.columns.duplicated()].copy()
    if not df.columns.is_unique:
        raise RuntimeError(f"{label}: still has duplicate columns after dedupe.")
    return df

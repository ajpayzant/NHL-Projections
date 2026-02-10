from __future__ import annotations
import os
import re
import time
import argparse
import pandas as pd
from datetime import datetime
from gspread.exceptions import APIError

from nhlproj.google_auth import get_gspread_client

def safe_call(fn, *args, **kwargs):
    for attempt in range(10):
        try:
            return fn(*args, **kwargs)
        except APIError as e:
            msg = str(e)
            is_429 = ("429" in msg) or ("Quota exceeded" in msg) or ("Rate Limit" in msg)
            is_5xx = any(code in msg for code in ["500", "502", "503", "504"])
            if is_429 or is_5xx:
                time.sleep(min(60, 2 ** attempt))
                continue
            raise
    raise RuntimeError("Too many retries (Google Sheets API).")

def to_int(x):
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        return int(float(s))
    except:
        return None

def classify_unit(unit):
    if unit in ("F1","F2","F3","F4"):
        return "EV_F"
    if unit in ("D1","D2","D3"):
        return "EV_D"
    if unit == "G":
        return "G"
    if unit.startswith("PP"):
        return "PP"
    if unit.startswith("PK"):
        return "PK"
    return "OTHER"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nhl-root", default=os.getenv("NHL_ROOT", "./data/NHL"))
    ap.add_argument("--sheets-subdir", default="NHL_Manual_Lineups_20252026")
    ap.add_argument("--sheet-url", default="")
    args = ap.parse_args()

    nhl_root = args.nhl_root
    sheets_dir = os.path.join(nhl_root, args.sheets_subdir)
    input_dir = os.path.join(nhl_root, "inputs")
    os.makedirs(input_dir, exist_ok=True)

    link_file = os.path.join(sheets_dir, "NHL_Roster_and_Expected_Lineups_LINK.txt")

    sheet_url = args.sheet_url.strip()
    if not sheet_url:
        if os.path.exists(link_file):
            sheet_url = open(link_file, "r").read().strip()
            print("✅ Loaded SHEET_URL from link file:", sheet_url)
        else:
            raise RuntimeError("No SHEET_URL provided and link file not found.")

    sheet_id = sheet_url.split("/d/")[1].split("/")[0]

    TEAM_ROWS = 320
    ROSTER_START_ROW = 6
    ROSTER_END_ROW   = 55
    LINEUP_START_ROW = 60
    LINEUP_ROWS = [
        ("F1","LW"),("F1","C"),("F1","RW"),
        ("F2","LW"),("F2","C"),("F2","RW"),
        ("F3","LW"),("F3","C"),("F3","RW"),
        ("F4","LW"),("F4","C"),("F4","RW"),
        ("D1","LD"),("D1","RD"),
        ("D2","LD"),("D2","RD"),
        ("D3","LD"),("D3","RD"),
        ("G","Starter"),("G","Backup"),
        ("PP1","P1"),("PP1","P2"),("PP1","P3"),("PP1","P4"),("PP1","P5"),
        ("PP2","P1"),("PP2","P2"),("PP2","P3"),("PP2","P4"),("PP2","P5"),
        ("PK1","F1"),("PK1","F2"),("PK1","D1"),("PK1","D2"),
        ("PK2","F1"),("PK2","F2"),("PK2","D1"),("PK2","D2"),
    ]
    LINEUP_END_ROW = LINEUP_START_ROW + len(LINEUP_ROWS) - 1

    LINEUP_RANGE_A1 = f"A{LINEUP_START_ROW}:D{LINEUP_END_ROW}"
    ROSTER_RANGE_A1 = f"A{ROSTER_START_ROW}:F{ROSTER_END_ROW}"

    gc = get_gspread_client()
    sh = safe_call(gc.open_by_key, sheet_id)

    titles = [ws.title for ws in sh.worksheets()]
    teams = sorted([t for t in titles if re.fullmatch(r"[A-Z]{2,3}", t)])
    print("✅ Found team tabs:", len(teams))
    print("Teams:", teams)

    ranges = []
    for t in teams:
        ranges.append(f"'{t}'!{ROSTER_RANGE_A1}")
        ranges.append(f"'{t}'!{LINEUP_RANGE_A1}")

    resp = safe_call(sh.values_batch_get, ranges)
    value_ranges = resp.get("valueRanges", [])

    pulled_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows_long = []
    roster_index = {}

    for i in range(0, len(value_ranges), 2):
        roster_vr = value_ranges[i]
        lineup_vr = value_ranges[i+1]
        team = roster_vr["range"].split("!")[0].replace("'", "")

        roster_vals = roster_vr.get("values", [])
        lineup_vals = lineup_vr.get("values", [])

        ids = set()
        for r in roster_vals:
            pid = to_int(r[0]) if len(r) > 0 else None
            if pid is not None:
                ids.add(pid)
        roster_index[team] = ids

        for j in range(len(LINEUP_ROWS)):
            u, s = LINEUP_ROWS[j]
            row = lineup_vals[j] if j < len(lineup_vals) else []
            unit = row[0].strip() if len(row) > 0 and str(row[0]).strip() else u
            slot = row[1].strip() if len(row) > 1 and str(row[1]).strip() else s
            player_name = row[2].strip() if len(row) > 2 and str(row[2]).strip() else ""
            player_id = to_int(row[3]) if len(row) > 3 else None

            rows_long.append({
                "pulled_at": pulled_at,
                "team": team,
                "unit": unit,
                "slot": slot,
                "unit_group": classify_unit(unit),
                "playerName_sheet": player_name,
                "playerId": player_id,
                "in_roster_block": int(player_id in ids) if player_id is not None else 0
            })

    lineups_long = pd.DataFrame(rows_long)
    print("✅ lineups_long:", lineups_long.shape)

    rep = []
    for t in teams:
        df = lineups_long[lineups_long["team"] == t].copy()
        filled_name = df["playerName_sheet"].astype(str).str.strip().ne("")
        filled_rows = int(filled_name.sum())
        missing_when_filled = int(df.loc[filled_name, "playerId"].isna().sum())
        not_in_roster = int(((df["playerId"].notna()) & (df["in_roster_block"] == 0)).sum())
        dup_ev = (
            df[df["unit_group"].isin(["EV_F","EV_D","G"]) & df["playerId"].notna()]
            ["playerId"].duplicated()
            .sum()
        )
        rep.append({
            "team": t,
            "filled_lineup_rows": filled_rows,
            "missing_playerId_when_name_filled": missing_when_filled,
            "playerId_not_in_roster_block_rows": not_in_roster,
            "duplicate_playerId_in_EV_or_goalie_rows": int(dup_ev),
        })

    report = pd.DataFrame(rep).sort_values(
        ["missing_playerId_when_name_filled","playerId_not_in_roster_block_rows","duplicate_playerId_in_EV_or_goalie_rows"],
        ascending=False
    )
    print("✅ Quality report top 10:\n", report.head(10).to_string(index=False))

    ev = lineups_long[lineups_long["unit_group"].isin(["EV_F","EV_D"])].copy()
    goalies = lineups_long[lineups_long["unit_group"] == "G"].copy()

    ev_lines = ev.pivot_table(
        index=["pulled_at","team","unit"],
        columns="slot",
        values="playerId",
        aggfunc="first"
    ).reset_index()

    cols_order = ["pulled_at","team","unit","LW","C","RW","LD","RD"]
    for c in cols_order:
        if c not in ev_lines.columns:
            ev_lines[c] = pd.NA
    ev_lines = ev_lines[cols_order].sort_values(["team","unit"])

    goalies_wide = goalies.pivot_table(
        index=["pulled_at","team","unit"],
        columns="slot",
        values="playerId",
        aggfunc="first"
    ).reset_index()

    for c in ["Starter","Backup"]:
        if c not in goalies_wide.columns:
            goalies_wide[c] = pd.NA
    goalies_wide = goalies_wide[["pulled_at","team","Starter","Backup"]].sort_values("team")

    # Save (same filenames)
    lineups_long.to_parquet(os.path.join(input_dir, "manual_lineups_long.parquet"), index=False)
    lineups_long.to_csv(os.path.join(input_dir, "manual_lineups_long.csv"), index=False)

    ev_lines.to_parquet(os.path.join(input_dir, "expected_lines_forwards_defense.parquet"), index=False)
    ev_lines.to_csv(os.path.join(input_dir, "expected_lines_forwards_defense.csv"), index=False)

    goalies_wide.to_parquet(os.path.join(input_dir, "expected_goalies.parquet"), index=False)
    goalies_wide.to_csv(os.path.join(input_dir, "expected_goalies.csv"), index=False)

    report.to_csv(os.path.join(input_dir, "lineup_quality_report.csv"), index=False)

    print("✅ Saved inputs to:", input_dir)

if __name__ == "__main__":
    main()

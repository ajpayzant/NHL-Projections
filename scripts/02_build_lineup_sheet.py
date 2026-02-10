from __future__ import annotations
import os
import time
import argparse
import pandas as pd
import requests
from datetime import datetime
from gspread.exceptions import APIError
from googleapiclient.discovery import build

from nhlproj.common import DEFAULT_HEADERS_NHL
from nhlproj.google_auth import get_gspread_client, get_drive_service, get_service_account_credentials

def fetch_json(url, timeout=30, max_retries=6, sleep_base=1.0):
    for attempt in range(max_retries):
        r = requests.get(url, headers=DEFAULT_HEADERS_NHL, timeout=timeout)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(30, sleep_base * (2 ** attempt)))
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()

def nhl_season_id_for_date(dt=None):
    dt = dt or datetime.now()
    start_year = dt.year if dt.month >= 7 else dt.year - 1
    return f"{start_year}{start_year+1}"

def get_nhl_team_abbrevs_from_standings(date_str):
    url = f"https://api-web.nhle.com/v1/standings/{date_str}"
    data = fetch_json(url)
    teams = []
    for rec in data.get("standings", []):
        ab = rec.get("teamAbbrev", {}).get("default")
        if ab:
            teams.append(ab)
    return sorted(set(teams))

def fetch_team_roster(team_abbrev, season_id):
    url = f"https://api-web.nhle.com/v1/roster/{team_abbrev}/{season_id}"
    data = fetch_json(url)
    rows = []
    for group_key in ["forwards", "defensemen", "goalies"]:
        for p in data.get(group_key, []) or []:
            pid = p.get("id")
            first = (p.get("firstName") or {}).get("default", "")
            last  = (p.get("lastName") or {}).get("default", "")
            name = (first + " " + last).strip()
            rows.append({
                "team": team_abbrev,
                "seasonId": season_id,
                "playerId": pid,
                "name": name,
                "pos": p.get("positionCode",""),
                "shootsCatches": p.get("shootsCatches",""),
                "sweaterNumber": p.get("sweaterNumber", None),
                "rosterGroup": group_key
            })
    return rows

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

def get_or_create_drive_folder(drive_svc, parent_id, folder_name):
    q = (
        f"mimeType='application/vnd.google-apps.folder' and "
        f"name='{folder_name}' and '{parent_id}' in parents and trashed=false"
    )
    res = drive_svc.files().list(q=q, fields="files(id,name)", pageSize=10).execute()
    files = res.get("files", [])
    if files:
        return files[0]["id"]
    meta = {"name": folder_name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]}
    folder = drive_svc.files().create(body=meta, fields="id").execute()
    return folder["id"]

def ensure_drive_path(drive_svc, path_parts):
    parent = "root"
    for part in path_parts:
        parent = get_or_create_drive_folder(drive_svc, parent, part)
    return parent

def move_file_to_folder(drive_svc, file_id, folder_id):
    meta = drive_svc.files().get(fileId=file_id, fields="parents").execute()
    prev_parents = ",".join(meta.get("parents", []))
    drive_svc.files().update(
        fileId=file_id,
        addParents=folder_id,
        removeParents=prev_parents,
        fields="id,parents"
    ).execute()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nhl-root", default=os.getenv("NHL_ROOT", "./data/NHL"))
    ap.add_argument("--sheet-name", default="NHL Roster and Expected Lineups")
    ap.add_argument("--sheets-subdir", default="NHL_Manual_Lineups_20252026")
    args = ap.parse_args()

    project_root = args.nhl_root
    sheets_dir = os.path.join(project_root, args.sheets_subdir)
    os.makedirs(sheets_dir, exist_ok=True)

    today_str = datetime.now().strftime("%Y-%m-%d")
    season_id = nhl_season_id_for_date()

    teams = get_nhl_team_abbrevs_from_standings(today_str)
    print("✅ Found teams:", len(teams))

    all_rows = []
    for ab in teams:
        all_rows.extend(fetch_team_roster(ab, season_id))
        time.sleep(0.15)

    roster_df = pd.DataFrame(all_rows)
    roster_df["playerId"] = pd.to_numeric(roster_df["playerId"], errors="coerce").astype("Int64")
    roster_df["sweaterNumber"] = pd.to_numeric(roster_df["sweaterNumber"], errors="coerce").astype("Int64")
    roster_df = roster_df.sort_values(["team","rosterGroup","pos","name"], na_position="last").reset_index(drop=True)

    roster_csv = os.path.join(sheets_dir, f"rosters_{season_id}_{today_str}.csv")
    roster_df.to_csv(roster_csv, index=False)
    print("✅ saved roster snapshot:", roster_csv)

    # Google clients
    gc = get_gspread_client()
    sh = safe_call(gc.create, args.sheet_name)
    print("✅ Created sheet:", sh.url)

    link_path = os.path.join(sheets_dir, "NHL_Roster_and_Expected_Lineups_LINK.txt")
    with open(link_path, "w") as f:
        f.write(sh.url.strip() + "\n")
    print("✅ saved sheet link:", link_path)

    drive_svc = get_drive_service()
    folder_id = ensure_drive_path(drive_svc, ["NHL", args.sheets_subdir])
    move_file_to_folder(drive_svc, sh.id, folder_id)
    print("✅ moved Google Sheet into Drive folder: NHL/" + args.sheets_subdir)

    # Tabs
    TEAM_ROWS, TEAM_COLS = 320, 8
    MASTER_ROWS, MASTER_COLS = 1200, 8

    existing_titles = {ws.title for ws in sh.worksheets()}
    needed_titles = ["Roster_Master"] + teams

    add_reqs = []
    for title in needed_titles:
        if title not in existing_titles:
            row_count = MASTER_ROWS if title == "Roster_Master" else TEAM_ROWS
            add_reqs.append({
                "addSheet": {
                    "properties": {
                        "title": title,
                        "gridProperties": {"rowCount": row_count, "columnCount": TEAM_COLS}
                    }
                }
            })
    if add_reqs:
        safe_call(sh.batch_update, {"requests": add_reqs})

    try:
        ws1 = sh.worksheet("Sheet1")
        safe_call(sh.del_worksheet, ws1)
    except Exception:
        pass

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

    ROSTER_LABEL_ROW = 4
    ROSTER_HDR_ROW   = 5
    ROSTER_START_ROW = 6
    ROSTER_END_ROW   = 55

    LINEUP_LABEL_ROW = 58
    LINEUP_HDR_ROW   = 59
    LINEUP_START_ROW = 60
    LINEUP_END_ROW   = LINEUP_START_ROW + len(LINEUP_ROWS) - 1

    def blank_grid(nrows, ncols):
        return [[""] * ncols for _ in range(nrows)]

    def make_roster_master_grid(df: pd.DataFrame):
        cols = ["team","seasonId","playerId","name","pos","shootsCatches","sweaterNumber","rosterGroup"]
        sub = df.copy()
        for c in cols:
            if c not in sub.columns:
                sub[c] = ""
        sub = sub[cols].astype(object).where(pd.notnull(sub), "")
        values = [cols] + sub.values.tolist()
        grid = blank_grid(MASTER_ROWS, MASTER_COLS)
        for r in range(min(len(values), MASTER_ROWS)):
            grid[r][:MASTER_COLS] = values[r][:MASTER_COLS]
        return grid

    def make_team_grid(team_abbrev: str, df: pd.DataFrame):
        g = blank_grid(TEAM_ROWS, TEAM_COLS)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        g[0][0] = f"{team_abbrev} — Manual Lines / Goalies / Special Teams"
        g[1][0] = f"Season: {season_id} | Generated: {now_str}"

        g[ROSTER_LABEL_ROW-1][0] = "Roster (from NHL API)"
        roster_cols = ["playerId","name","pos","shootsCatches","sweaterNumber","rosterGroup"]
        g[ROSTER_HDR_ROW-1][0:len(roster_cols)] = roster_cols

        team_roster = df[df["team"] == team_abbrev].copy()
        if not team_roster.empty:
            team_roster = team_roster.sort_values(["rosterGroup","pos","name"], na_position="last")
            roster_vals = team_roster[roster_cols].astype(object).where(pd.notnull(team_roster[roster_cols]), "").values.tolist()
        else:
            roster_vals = []

        max_rows = ROSTER_END_ROW - ROSTER_START_ROW + 1
        for i in range(max_rows):
            rr = ROSTER_START_ROW + i
            if i < len(roster_vals):
                g[rr-1][0:len(roster_cols)] = roster_vals[i][:len(roster_cols)]

        g[LINEUP_LABEL_ROW-1][0] = "Lineup Inputs (edit PlayerName dropdown; playerId auto-fills)"
        g[LINEUP_HDR_ROW-1][0:4] = ["unit","slot","playerName (dropdown)","playerId (auto)"]

        for i, (u, s) in enumerate(LINEUP_ROWS):
            row = LINEUP_START_ROW + i
            g[row-1][0] = u
            g[row-1][1] = s
            g[row-1][2] = ""
            g[row-1][3] = (
                f'=IFERROR(INDEX($A${ROSTER_START_ROW}:$A${ROSTER_END_ROW}, '
                f'MATCH(C{row}, $B${ROSTER_START_ROW}:$B${ROSTER_END_ROW}, 0)),"")'
            )
        return g

    batch_data = [{
        "range": f"'Roster_Master'!A1:H{MASTER_ROWS}",
        "values": make_roster_master_grid(roster_df)
    }]

    for t in teams:
        batch_data.append({
            "range": f"'{t}'!A1:H{TEAM_ROWS}",
            "values": make_team_grid(t, roster_df)
        })

    safe_call(sh.values_batch_update, {"valueInputOption": "USER_ENTERED", "data": batch_data})
    print("✅ Wrote Roster_Master + all team tabs in ONE batch update")
    print("✅ Sheet URL:", sh.url)

    # Formatting + named ranges + dropdown validation
    def to_gridrange(sheet_id, start_row_1, start_col_1, end_row_1, end_col_1):
        return {
            "sheetId": sheet_id,
            "startRowIndex": start_row_1 - 1,
            "endRowIndex": end_row_1,
            "startColumnIndex": start_col_1 - 1,
            "endColumnIndex": end_col_1
        }

    ws_map = {ws.title: ws for ws in sh.worksheets()}
    team_sheet_ids = {t: ws_map[t]._properties["sheetId"] for t in teams}

    HEADER_BG = {"red": 0.90, "green": 0.94, "blue": 1.0}
    TITLE_BG  = {"red": 0.12, "green": 0.12, "blue": 0.12}
    WHITE     = {"red": 1.0, "green": 1.0, "blue": 1.0}

    fmt_requests = []
    for t in teams:
        sid = team_sheet_ids[t]

        fmt_requests.append({
            "updateSheetProperties": {
                "properties": {"sheetId": sid, "gridProperties": {"frozenRowCount": 5}},
                "fields": "gridProperties.frozenRowCount"
            }
        })

        fmt_requests.append({
            "updateDimensionProperties": {
                "range": {"sheetId": sid, "dimension": "COLUMNS", "startIndex": 0, "endIndex": 1},
                "properties": {"pixelSize": 120},
                "fields": "pixelSize"
            }
        })
        fmt_requests.append({
            "updateDimensionProperties": {
                "range": {"sheetId": sid, "dimension": "COLUMNS", "startIndex": 1, "endIndex": 2},
                "properties": {"pixelSize": 240},
                "fields": "pixelSize"
            }
        })
        fmt_requests.append({
            "updateDimensionProperties": {
                "range": {"sheetId": sid, "dimension": "COLUMNS", "startIndex": 2, "endIndex": 8},
                "properties": {"pixelSize": 110},
                "fields": "pixelSize"
            }
        })

        fmt_requests.append({
            "repeatCell": {
                "range": to_gridrange(sid, 1, 1, 2, 8),
                "cell": {"userEnteredFormat": {
                    "backgroundColor": TITLE_BG,
                    "textFormat": {"foregroundColor": WHITE, "bold": True, "fontSize": 12}
                }},
                "fields": "userEnteredFormat(backgroundColor,textFormat)"
            }
        })

        for hdr in [ROSTER_HDR_ROW, LINEUP_HDR_ROW]:
            fmt_requests.append({
                "repeatCell": {
                    "range": to_gridrange(sid, hdr, 1, hdr, 8),
                    "cell": {"userEnteredFormat": {
                        "backgroundColor": HEADER_BG,
                        "textFormat": {"bold": True}
                    }},
                    "fields": "userEnteredFormat(backgroundColor,textFormat.bold)"
                }
            })

        fmt_requests.append({
            "addBanding": {
                "bandedRange": {
                    "range": to_gridrange(sid, ROSTER_HDR_ROW, 1, ROSTER_END_ROW, 6),
                    "rowProperties": {
                        "headerColor": HEADER_BG,
                        "firstBandColor": {"red": 1.0, "green": 1.0, "blue": 1.0},
                        "secondBandColor": {"red": 0.97, "green": 0.98, "blue": 1.0}
                    }
                }
            }
        })

        fmt_requests.append({
            "addBanding": {
                "bandedRange": {
                    "range": to_gridrange(sid, LINEUP_HDR_ROW, 1, LINEUP_END_ROW, 4),
                    "rowProperties": {
                        "headerColor": HEADER_BG,
                        "firstBandColor": {"red": 1.0, "green": 1.0, "blue": 1.0},
                        "secondBandColor": {"red": 0.98, "green": 0.99, "blue": 0.98}
                    }
                }
            }
        })

        fmt_requests.append({
            "setBasicFilter": {
                "filter": {"range": to_gridrange(sid, ROSTER_HDR_ROW, 1, ROSTER_END_ROW, 6)}
            }
        })

        named_range = f"ROSTER_NAMES_{t}"
        fmt_requests.append({
            "addNamedRange": {
                "namedRange": {
                    "name": named_range,
                    "range": to_gridrange(sid, ROSTER_START_ROW, 2, ROSTER_END_ROW, 2)
                }
            }
        })

        fmt_requests.append({
            "repeatCell": {
                "range": to_gridrange(sid, LINEUP_START_ROW, 3, LINEUP_END_ROW, 3),
                "cell": {
                    "dataValidation": {
                        "condition": {"type": "ONE_OF_RANGE", "values": [{"userEnteredValue": f"={named_range}"}]},
                        "strict": True,
                        "showCustomUi": True
                    }
                },
                "fields": "dataValidation"
            }
        })

    safe_call(sh.batch_update, {"requests": fmt_requests})
    print("✅ Applied formatting + dropdowns to all team tabs")
    print("✅ Sheet URL:", sh.url)

if __name__ == "__main__":
    main()

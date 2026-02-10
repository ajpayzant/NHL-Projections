from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from joblib import load

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NHLProjBot/1.0)",
    "Accept": "application/json,text/plain,*/*",
}

TEAM_MAP = {"LA": "LAK", "NJ": "NJD", "SJ": "SJS", "TB": "TBL", "WAS": "WSH"}

def fetch_json(url, timeout=30):
    r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _is_regular_season_game(game):
    game_type = game.get("gameType")
    # NHL API uses integer 2 for regular season in schedule/score payloads.
    return game_type in (2, "2", "R")

def _pull_games_from_schedule_payload(payload, date_from, date_to):
    rows = []
    for day in payload.get("gameWeek", []):
        day_date = pd.to_datetime(day.get("date"), errors="coerce")
        if pd.isna(day_date):
            continue
        day_date = day_date.date()
        if day_date < date_from or day_date > date_to:
            continue
        for g in day.get("games", []):
            if not _is_regular_season_game(g):
                continue
            rows.append({
                "gameId": g.get("id"),
                "gameDateUTC": g.get("startTimeUTC"),
                "homeTeam": g.get("homeTeam", {}).get("abbrev"),
                "awayTeam": g.get("awayTeam", {}).get("abbrev"),
                "scheduleDate": str(day_date),
            })
    return rows

def _pull_games_from_score_payload(payload, date_from, date_to):
    rows = []
    game_date = pd.to_datetime(payload.get("currentDate"), errors="coerce")
    if pd.isna(game_date):
        return rows
    game_date = game_date.date()
    if not (date_from <= game_date <= date_to):
        return rows
    for g in payload.get("games", []):
        if not _is_regular_season_game(g):
            continue
        rows.append({
            "gameId": g.get("id"),
            "gameDateUTC": g.get("startTimeUTC"),
            "homeTeam": g.get("homeTeam", {}).get("abbrev"),
            "awayTeam": g.get("awayTeam", {}).get("abbrev"),
            "scheduleDate": str(game_date),
        })
    return rows

def get_schedule(date_from, date_to):
    games = []
    d = date_from
    while d <= date_to:
        ds = d.strftime("%Y-%m-%d")
        schedule_url = f"https://api-web.nhle.com/v1/schedule/{ds}"
        pulled_for_day = 0
        try:
            schedule_payload = fetch_json(schedule_url)
            day_rows = _pull_games_from_schedule_payload(schedule_payload, date_from, date_to)
            games.extend(day_rows)
            pulled_for_day = len(day_rows)
        except Exception as e:
            print(f"⚠️ schedule endpoint failed for {ds}: {e}")

        # Fallback: score endpoint can still return a day's slate if schedule is unavailable.
        if pulled_for_day == 0:
            score_url = f"https://api-web.nhle.com/v1/score/{ds}"
            try:
                score_payload = fetch_json(score_url)
                games.extend(_pull_games_from_score_payload(score_payload, date_from, date_to))
            except Exception as e:
                print(f"⚠️ score endpoint failed for {ds}: {e}")

        d += timedelta(days=1)

    out = pd.DataFrame(games)
    if out.empty:
        return out
    out = out.dropna(subset=["gameId", "homeTeam", "awayTeam"]).copy()
    out["gameId"] = pd.to_numeric(out["gameId"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["gameId"]).copy()
    out["gameId"] = out["gameId"].astype("int64")
    out = out.drop_duplicates(subset=["gameId"], keep="first").reset_index(drop=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nhl-root", default=os.getenv("NHL_ROOT", "./data/NHL"))
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()

    nhl_root  = args.nhl_root
    feat_dir  = os.path.join(nhl_root, "features")
    model_dir = os.path.join(nhl_root, "models")
    out_dir   = os.path.join(nhl_root, "preds")
    os.makedirs(out_dir, exist_ok=True)

    team_feat_path = os.path.join(feat_dir, "team_features.parquet")
    team_feat = pd.read_parquet(team_feat_path)
    team_feat["gameDate"] = pd.to_datetime(team_feat["gameDate"], errors="coerce")
    team_feat = team_feat.dropna(subset=["team","gameDate"]).copy()

    feat_cols = [c for c in team_feat.columns if c.startswith("team_")]
    team_latest = (
        team_feat.sort_values(["team","gameDate"])
                 .groupby("team", as_index=False)
                 .tail(1)
                 .reset_index(drop=True)
    )
    team_latest = team_latest[["team","gameDate"] + feat_cols].rename(columns={"gameDate":"last_gameDate"})
    print("✅ Latest team snapshots:", team_latest.shape)

    targets = [
        "y_goalsFor","y_goalsAgainst",
        "y_shotsOnGoalFor","y_shotsOnGoalAgainst",
        "y_xGoalsFor","y_xGoalsAgainst",
    ]
    models = {}
    for y in targets:
        p = os.path.join(model_dir, f"team_{y}.joblib")
        models[y] = load(p)
    print("✅ Loaded models:", list(models.keys()))

    today = datetime.now(timezone.utc).date()
    df_sched = get_schedule(today, today + timedelta(days=args.days))
    print("✅ Upcoming games:", df_sched.shape)

    if df_sched.empty:
        print("⚠️ No games found in next window.")
        return

    df_sched["homeTeam"] = df_sched["homeTeam"].astype(str).str.strip().replace(TEAM_MAP)
    df_sched["awayTeam"] = df_sched["awayTeam"].astype(str).str.strip().replace(TEAM_MAP)
    df_sched["scheduleDate_dt"] = pd.to_datetime(df_sched["scheduleDate"], errors="coerce")

    home = df_sched.merge(team_latest, left_on="homeTeam", right_on="team", how="left").drop(columns=["team"])
    home = home.rename(columns={c: f"home_{c}" for c in feat_cols})
    home = home.rename(columns={"last_gameDate":"home_last_gameDate"})

    full = home.merge(team_latest, left_on="awayTeam", right_on="team", how="left").drop(columns=["team"])
    full = full.rename(columns={c: f"away_{c}" for c in feat_cols})
    full = full.rename(columns={"last_gameDate":"away_last_gameDate"})

    full["home_rest_days"] = (full["scheduleDate_dt"] - pd.to_datetime(full["home_last_gameDate"], errors="coerce")).dt.days
    full["away_rest_days"] = (full["scheduleDate_dt"] - pd.to_datetime(full["away_last_gameDate"], errors="coerce")).dt.days

    def predict_team(prefix):
        X = full[[f"{prefix}_{c}" for c in feat_cols]].copy()
        X.columns = feat_cols
        return pd.DataFrame({y: m.predict(X) for y, m in models.items()})

    home_pred = predict_team("home")
    away_pred = predict_team("away")

    out = full[["gameId","scheduleDate","homeTeam","awayTeam","home_rest_days","away_rest_days"]].copy()
    out["home_goals"] = 0.5 * home_pred["y_goalsFor"] + 0.5 * away_pred["y_goalsAgainst"]
    out["away_goals"] = 0.5 * away_pred["y_goalsFor"] + 0.5 * home_pred["y_goalsAgainst"]

    out["home_sog"] = 0.5 * home_pred["y_shotsOnGoalFor"] + 0.5 * away_pred["y_shotsOnGoalAgainst"]
    out["away_sog"] = 0.5 * away_pred["y_shotsOnGoalFor"] + 0.5 * home_pred["y_shotsOnGoalAgainst"]

    out["home_xg"] = 0.5 * home_pred["y_xGoalsFor"] + 0.5 * away_pred["y_xGoalsAgainst"]
    out["away_xg"] = 0.5 * away_pred["y_xGoalsFor"] + 0.5 * home_pred["y_xGoalsAgainst"]

    out["proj_total_goals"] = out["home_goals"] + out["away_goals"]
    out["proj_total_sog"]   = out["home_sog"] + out["away_sog"]

    out = out.sort_values(["scheduleDate","homeTeam","awayTeam"]).reset_index(drop=True)
    out_path = os.path.join(out_dir, "team_game_projections_next7.csv")
    out.to_csv(out_path, index=False)
    print("✅ Saved:", out_path)
    print(out.head(20).to_string(index=False))

if __name__ == "__main__":
    main()

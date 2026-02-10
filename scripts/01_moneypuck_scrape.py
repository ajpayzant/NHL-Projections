from __future__ import annotations
import os
import argparse
import pandas as pd
from nhlproj.common import (
    current_nhl_season_start,
    download_with_retries,
    list_csvs_from_index,
    standardize_team_game_cols,
    parse_game_date,
    ensure_minutes,
)

MP_BASE = "https://moneypuck.com/moneypuck/playerData"

def season_summary_url(season, season_type, entity):
    return f"{MP_BASE}/seasonSummary/{season}/{season_type}/{entity}.csv"

def team_player_gbg_index(season, season_type, kind):
    return f"{MP_BASE}/teamPlayerGameByGame/{season}/{season_type}/{kind}/"

def team_player_gbg_file_url(season, season_type, kind, team_code):
    return f"{team_player_gbg_index(season, season_type, kind)}{team_code}.csv"

def scrape_season_summaries(season, season_type, raw_dir, end_season):
    for entity in ["skaters","goalies","lines","teams"]:
        url = season_summary_url(season, season_type, entity)
        path = os.path.join(raw_dir, "seasonSummary", str(season), season_type, f"{entity}.csv")
        force = (season == end_season)
        download_with_retries(url, path, force=force)

def scrape_team_player_gbg(season, season_type, raw_dir, end_season):
    for kind in ["skaters","goalies"]:
        idx = team_player_gbg_index(season, season_type, kind)
        team_files = list_csvs_from_index(idx)
        team_codes = [f.replace(".csv","") for f in team_files]
        print(f"  {season} {kind}: {len(team_codes)} teams")
        for team in team_codes:
            url = team_player_gbg_file_url(season, season_type, kind, team)
            out_path = os.path.join(raw_dir, "teamPlayerGameByGame", str(season), season_type, kind, f"{team}.csv")
            force = (season == end_season)
            download_with_retries(url, out_path, force=force)

def scrape_team_gbg_all(raw_dir, season_type):
    TEAM_GBG_INDEX = f"{MP_BASE}/careers/gameByGame/{season_type}/teams/"
    team_files = list_csvs_from_index(TEAM_GBG_INDEX)
    print("Teams in careers gameByGame:", len(team_files))
    for f in team_files:
        url = TEAM_GBG_INDEX + f
        out_path = os.path.join(raw_dir, "careers", "gameByGame", season_type, "teams", f)
        download_with_retries(url, out_path, force=True)

def load_all_team_gbg_raw(raw_dir, season_type, start_season, end_season):
    folder = os.path.join(raw_dir, "careers", "gameByGame", season_type, "teams")
    dfs = []
    for fn in os.listdir(folder):
        if not fn.endswith(".csv"):
            continue
        path = os.path.join(folder, fn)
        df = pd.read_csv(path)
        df = standardize_team_game_cols(df)
        if "team" not in df.columns or df["team"].isna().all():
            df["team"] = fn.replace(".csv","")
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    if "season" in all_df.columns:
        all_df = all_df[(all_df["season"] >= start_season) & (all_df["season"] <= end_season)].copy()

    all_df = all_df[all_df["situation"].astype(str).str.lower().eq("all")].copy()
    all_df["gameDate"] = all_df["gameDate"].map(parse_game_date)
    return all_df

def load_team_player_gbg(raw_dir, season_type, start_season, end_season, kind):
    base = os.path.join(raw_dir, "teamPlayerGameByGame")
    dfs = []
    for season in range(start_season, end_season+1):
        folder = os.path.join(base, str(season), season_type, kind)
        if not os.path.exists(folder):
            continue
        for fn in os.listdir(folder):
            if not fn.endswith(".csv"):
                continue
            path = os.path.join(folder, fn)
            df = pd.read_csv(path)
            df = standardize_team_game_cols(df)
            if "season" not in df.columns:
                df["season"] = season
            df = df[df["season"].between(start_season, end_season)].copy()
            df["gameDate"] = df["gameDate"].map(parse_game_date)
            dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out = out[out["situation"].astype(str).str.lower().eq("all")].copy()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nhl-root", default=os.getenv("NHL_ROOT", "./data/NHL"))
    ap.add_argument("--season-type", default="regular")
    ap.add_argument("--start-season", type=int, default=2019)
    ap.add_argument("--end-season", type=int, default=None)
    args = ap.parse_args()

    nhl_root = args.nhl_root
    season_type = args.season_type
    start_season = args.start_season
    end_season = args.end_season or current_nhl_season_start()

    mp_root = os.path.join(nhl_root, "MoneyPuck_Data")
    raw_dir = os.path.join(mp_root, "raw")
    proc_dir = os.path.join(mp_root, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    print("Season range:", start_season, "to", end_season, f"({season_type})")
    for season in range(start_season, end_season+1):
        print("SeasonSummary:", season)
        scrape_season_summaries(season, season_type, raw_dir, end_season)

    for season in range(start_season, end_season+1):
        print("TeamPlayerGBG:", season)
        scrape_team_player_gbg(season, season_type, raw_dir, end_season)

    scrape_team_gbg_all(raw_dir, season_type)
    print("✅ scraping complete")

    team_gbg = load_all_team_gbg_raw(raw_dir, season_type, start_season, end_season)
    skater_gbg = load_team_player_gbg(raw_dir, season_type, start_season, end_season, "skaters")
    goalie_gbg = load_team_player_gbg(raw_dir, season_type, start_season, end_season, "goalies")

    if "icetime" in skater_gbg.columns:
        skater_gbg["icetime_min"] = ensure_minutes(skater_gbg["icetime"])
    if "icetime" in goalie_gbg.columns:
        goalie_gbg["icetime_min"] = ensure_minutes(goalie_gbg["icetime"])

    team_path = os.path.join(proc_dir, "team_game.parquet")
    skater_path = os.path.join(proc_dir, "skater_game.parquet")
    goalie_path = os.path.join(proc_dir, "goalie_game.parquet")

    team_gbg.to_parquet(team_path, index=False)
    skater_gbg.to_parquet(skater_path, index=False)
    goalie_gbg.to_parquet(goalie_path, index=False)

    print("✅ Saved team:", team_path, team_gbg.shape, "| games:", team_gbg["gameId"].nunique())
    print("✅ Saved skater:", skater_path, skater_gbg.shape)
    print("✅ Saved goalie:", goalie_path, goalie_gbg.shape)
    print("date range:", team_gbg["gameDate"].min(), "to", team_gbg["gameDate"].max())

if __name__ == "__main__":
    main()

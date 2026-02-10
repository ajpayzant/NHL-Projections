from __future__ import annotations
import os
import argparse
import pandas as pd
from nhlproj.common import strip_and_dedup_columns

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nhl-root", default=os.getenv("NHL_ROOT", "./data/NHL"))
    args = ap.parse_args()

    nhl_root = args.nhl_root
    mp_proc = os.path.join(nhl_root, "MoneyPuck_Data", "processed")
    feat_dir = os.path.join(nhl_root, "features")
    out_dir = os.path.join(nhl_root, "model_data")
    os.makedirs(out_dir, exist_ok=True)

    team_game = pd.read_parquet(os.path.join(mp_proc, "team_game.parquet"))
    sk_game   = pd.read_parquet(os.path.join(mp_proc, "skater_game.parquet"))
    go_game   = pd.read_parquet(os.path.join(mp_proc, "goalie_game.parquet"))

    team_feat = pd.read_parquet(os.path.join(feat_dir, "team_features.parquet"))
    sk_feat   = pd.read_parquet(os.path.join(feat_dir, "skater_features.parquet"))
    go_feat   = pd.read_parquet(os.path.join(feat_dir, "goalie_features.parquet"))

    for name, df in [("team_game", team_game), ("team_feat", team_feat), ("sk_game", sk_game), ("sk_feat", sk_feat), ("go_game", go_game), ("go_feat", go_feat)]:
        df = strip_and_dedup_columns(df, name)

    print(f"✅ Loaded: team_game {team_game.shape} | team_feat {team_feat.shape} | sk_game {sk_game.shape} | sk_feat {sk_feat.shape} | go_game {go_game.shape} | go_feat {go_feat.shape}")

    # ---------- TEAM MODEL TABLE ----------
    # Identify likely target columns in team_game (same logic as your feature builder)
    def pick_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    gf = pick_col(team_game, ["goalsFor","gf","teamGoalsFor","goals_for"])
    ga = pick_col(team_game, ["goalsAgainst","ga","teamGoalsAgainst","goals_against"])
    sf = pick_col(team_game, ["shotsOnGoalFor","shotsFor","sf","sogFor"])
    sa = pick_col(team_game, ["shotsOnGoalAgainst","shotsAgainst","sa","sogAgainst"])
    xgf = pick_col(team_game, ["xGoalsFor","xgf","expectedGoalsFor","xgoalsFor"])
    xga = pick_col(team_game, ["xGoalsAgainst","xga","expectedGoalsAgainst","xgoalsAgainst"])

    team_targets = {
        "y_goalsFor": gf,
        "y_goalsAgainst": ga,
        "y_shotsOnGoalFor": sf,
        "y_shotsOnGoalAgainst": sa,
        "y_xGoalsFor": xgf,
        "y_xGoalsAgainst": xga,
    }

    team_keep = ["gameId","gameDate","season","team","team_opp","home_or_away"]
    team_keep = [c for c in team_keep if c in team_feat.columns]

    team_model = team_feat.copy()
    for yname, col in team_targets.items():
        if col and col in team_game.columns:
            # join targets from game table by (gameId, team)
            tmp = team_game[["gameId","team",col]].rename(columns={col:yname})
            team_model = team_model.merge(tmp, on=["gameId","team"], how="left")

    # keep base + features + targets
    team_cols = team_keep + [c for c in team_model.columns if c.startswith("team_")] + [c for c in team_model.columns if c.startswith("y_")]
    team_model_out = strip_and_dedup_columns(team_model[team_cols].copy(), "team_model_out")

    team_out_path = os.path.join(out_dir, "team_model_table.parquet")
    team_model_out.to_parquet(team_out_path, index=False)
    print("✅ Saved:", team_out_path, team_model_out.shape)

    # ---------- SKATER MODEL TABLE ----------
    # Targets: goals/assists/points/TOI if available
    pid = pick_col(sk_game, ["playerId","player_id","id","nhlPlayerId"])
    toi = pick_col(sk_game, ["icetime_min","icetime","timeOnIce","toi"])
    goals = pick_col(sk_game, ["goals"])
    assists = pick_col(sk_game, ["assists"])
    points = pick_col(sk_game, ["points"])

    sk_model = sk_feat.merge(
        sk_game[["gameId","team",pid] + [c for c in [goals,assists,points,toi] if c]].copy(),
        left_on=["gameId","team","playerId"],
        right_on=["gameId","team",pid],
        how="left"
    )

    if pid in sk_model.columns:
        sk_model = sk_model.drop(columns=[pid])

    rename_map = {}
    if goals: rename_map[goals] = "y_goals"
    if assists: rename_map[assists] = "y_assists"
    if points: rename_map[points] = "y_points"
    if toi: rename_map[toi] = "y_toi"
    sk_model = sk_model.rename(columns=rename_map)

    sk_cols = [c for c in ["gameId","gameDate","season","team","team_opp","home_or_away","playerId"] if c in sk_model.columns]
    sk_cols += [c for c in sk_model.columns if c.startswith("sk_")]
    sk_cols += [c for c in sk_model.columns if c.startswith("y_")]

    sk_out = strip_and_dedup_columns(sk_model[sk_cols].copy(), "skater_model_out")
    sk_out_path = os.path.join(out_dir, "skater_model_table.parquet")
    sk_out.to_parquet(sk_out_path, index=False)
    print("✅ Saved:", sk_out_path, sk_out.shape)

    # ---------- GOALIE MODEL TABLE ----------
    gpid = pick_col(go_game, ["playerId","player_id","id","nhlPlayerId"])
    gtoi = pick_col(go_game, ["icetime_min","icetime","timeOnIce","toi"])
    ga2 = pick_col(go_game, ["goalsAgainst","ga","goals_against"])
    sa2 = pick_col(go_game, ["shotsAgainst","sa","shots_against"])
    sv2 = pick_col(go_game, ["saves","sv"])

    go_model = go_feat.merge(
        go_game[["gameId","team",gpid] + [c for c in [ga2,sa2,sv2,gtoi] if c]].copy(),
        left_on=["gameId","team","playerId"],
        right_on=["gameId","team",gpid],
        how="left"
    )
    if gpid in go_model.columns:
        go_model = go_model.drop(columns=[gpid])

    rename_map = {}
    if ga2: rename_map[ga2] = "y_goalsAgainst"
    if sa2: rename_map[sa2] = "y_shotsAgainst"
    if sv2: rename_map[sv2] = "y_saves"
    if gtoi: rename_map[gtoi] = "y_toi"
    go_model = go_model.rename(columns=rename_map)

    go_cols = [c for c in ["gameId","gameDate","season","team","team_opp","home_or_away","playerId"] if c in go_model.columns]
    go_cols += [c for c in go_model.columns if c.startswith("go_")]
    go_cols += [c for c in go_model.columns if c.startswith("y_")]

    go_out = strip_and_dedup_columns(go_model[go_cols].copy(), "goalie_model_out")
    go_out_path = os.path.join(out_dir, "goalie_model_table.parquet")
    go_out.to_parquet(go_out_path, index=False)
    print("✅ Saved:", go_out_path, go_out.shape)

    print("✅ 05 complete. Output dir:", out_dir)

if __name__ == "__main__":
    main()

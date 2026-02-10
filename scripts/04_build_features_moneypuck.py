from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
from nhlproj.common import strip_and_dedup_columns, TEAM_MAP

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def ensure_dt(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def ensure_int(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df

def rolling_features(df, group_cols, sort_col, value_cols, windows=(5,10,20), prefix="r"):
    df = df.sort_values(group_cols + [sort_col]).copy()
    for c in value_cols:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        g = df.groupby(group_cols, sort=False)[c]
        for w in windows:
            out = g.apply(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            df[f"{prefix}{w}_{c}"] = out.reset_index(level=group_cols, drop=True)
    return df

def per60(df, num_col, toi_col):
    x = pd.to_numeric(df[num_col], errors="coerce")
    toi = pd.to_numeric(df[toi_col], errors="coerce")
    return (60.0 * x / toi.replace(0, np.nan))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nhl-root", default=os.getenv("NHL_ROOT", "./data/NHL"))
    args = ap.parse_args()

    nhl_root = args.nhl_root
    mp_proc = os.path.join(nhl_root, "MoneyPuck_Data", "processed")
    feat_dir = os.path.join(nhl_root, "features")
    os.makedirs(feat_dir, exist_ok=True)

    team_path = os.path.join(mp_proc, "team_game.parquet")
    skater_path = os.path.join(mp_proc, "skater_game.parquet")
    goalie_path = os.path.join(mp_proc, "goalie_game.parquet")

    team = pd.read_parquet(team_path)
    sk   = pd.read_parquet(skater_path)
    go   = pd.read_parquet(goalie_path)

    team = strip_and_dedup_columns(team, "team")
    sk   = strip_and_dedup_columns(sk, "skater")
    go   = strip_and_dedup_columns(go, "goalie")

    for d in (team, sk, go):
        if "team" in d.columns:
            d["team"] = d["team"].astype(str).str.strip().replace(TEAM_MAP)
        if "team_opp" in d.columns:
            d["team_opp"] = d["team_opp"].astype(str).str.strip().replace(TEAM_MAP)

    for d in (team, sk, go):
        ensure_int(d, "gameId")
        ensure_dt(d, "gameDate")

    # TEAM FEATURES
    team_cols_candidates = {
        "goalsFor": ["goalsFor","gf","teamGoalsFor","goals_for"],
        "goalsAgainst": ["goalsAgainst","ga","teamGoalsAgainst","goals_against"],
        "shotsFor": ["shotsFor","sf","shots_for","teamShotsFor","shotsOnGoalFor","sogFor"],
        "shotsAgainst": ["shotsAgainst","sa","shots_against","teamShotsAgainst","shotsOnGoalAgainst","sogAgainst"],
        "xGoalsFor": ["xGoalsFor","xgf","expectedGoalsFor","xgoalsFor"],
        "xGoalsAgainst": ["xGoalsAgainst","xga","expectedGoalsAgainst","xgoalsAgainst"],
    }
    canon = {}
    for k, cands in team_cols_candidates.items():
        c = pick_col(team, cands)
        if c:
            canon[k] = c
    print("TEAM detected canon map:", canon)

    team_feat = team.copy()
    if "team" in team_feat.columns and "gameDate" in team_feat.columns:
        team_feat = team_feat.sort_values(["team","gameDate"])
        team_feat["days_since_prev_game"] = team_feat.groupby("team")["gameDate"].diff().dt.days

    roll_cols = [canon[k] for k in canon.keys() if canon.get(k) in team_feat.columns]
    if "days_since_prev_game" in team_feat.columns:
        roll_cols = roll_cols + ["days_since_prev_game"]

    team_feat = rolling_features(team_feat, ["team"], "gameDate", roll_cols, (5,10,20), prefix="team_")

    keep_base = [c for c in ["gameId","gameDate","season","team","team_opp","home_or_away"] if c in team_feat.columns]
    keep_out = keep_base + [c for c in team_feat.columns if c.startswith("team_")]
    team_out = strip_and_dedup_columns(team_feat[keep_out].copy(), "team_out")

    team_out_path = os.path.join(feat_dir, "team_features.parquet")
    team_out.to_parquet(team_out_path, index=False)
    print("✅ Saved team features:", team_out.shape, "->", team_out_path)

    # SKATER FEATURES
    pid_col = pick_col(sk, ["playerId","player_id","id","nhlPlayerId"])
    toi_col = pick_col(sk, ["icetime_min","icetime","timeOnIce","toi"])
    if pid_col is None or toi_col is None:
        raise RuntimeError("Skater playerId or TOI column not found.")

    sk_feat = sk.copy()
    ensure_int(sk_feat, pid_col)

    if toi_col != "icetime_min" and "icetime_min" in sk_feat.columns:
        toi_col = "icetime_min"

    count_candidates = [
        "goals","assists","points",
        "shotsOnGoal","shots","shots_on_goal",
        "ixG","iXG","xGoals","xGoalsFor","xG",
        "iFenwick","iCorsi","shotAttempts"
    ]
    count_cols = [c for c in count_candidates if c in sk_feat.columns]

    for c in count_cols:
        sk_feat[f"{c}_per60"] = per60(sk_feat, c, toi_col)

    roll_cols_sk = [toi_col] + [f"{c}_per60" for c in count_cols]
    sk_feat = rolling_features(sk_feat, ["team", pid_col], "gameDate", roll_cols_sk, (5,10,20), prefix="sk_")

    keep_base = [c for c in ["gameId","gameDate","season","team","team_opp","home_or_away", pid_col] if c in sk_feat.columns]
    keep_out = keep_base + [c for c in sk_feat.columns if c.startswith("sk_")]
    sk_out = strip_and_dedup_columns(sk_feat[keep_out].rename(columns={pid_col:"playerId"}).copy(), "sk_out")

    sk_out_path = os.path.join(feat_dir, "skater_features.parquet")
    sk_out.to_parquet(sk_out_path, index=False)
    print("✅ Saved skater features:", sk_out.shape, "->", sk_out_path)

    # GOALIE FEATURES
    g_pid_col = pick_col(go, ["playerId","player_id","id","nhlPlayerId"])
    g_toi_col = pick_col(go, ["icetime_min","icetime","timeOnIce","toi"])
    if g_pid_col is None:
        raise RuntimeError("Goalie playerId column not found.")

    go_feat = go.copy()
    ensure_int(go_feat, g_pid_col)
    if g_toi_col and g_toi_col != "icetime_min" and "icetime_min" in go_feat.columns:
        g_toi_col = "icetime_min"

    ga_col = pick_col(go_feat, ["goalsAgainst","ga","goals_against"])
    sa_col = pick_col(go_feat, ["shotsAgainst","sa","shots_against"])
    sv_col = pick_col(go_feat, ["saves","sv"])

    if sa_col and ga_col:
        go_feat["save_pct"] = 1.0 - (
            pd.to_numeric(go_feat[ga_col], errors="coerce") /
            pd.to_numeric(go_feat[sa_col], errors="coerce").replace(0, np.nan)
        )
    else:
        go_feat["save_pct"] = np.nan

    roll_cols_go = [c for c in [ga_col, sa_col, sv_col, "save_pct", g_toi_col] if c and c in go_feat.columns]
    go_feat = rolling_features(go_feat, ["team", g_pid_col], "gameDate", roll_cols_go, (5,10,20), prefix="go_")

    keep_base = [c for c in ["gameId","gameDate","season","team","team_opp","home_or_away", g_pid_col] if c in go_feat.columns]
    keep_out = keep_base + [c for c in go_feat.columns if c.startswith("go_")]
    go_out = strip_and_dedup_columns(go_feat[keep_out].rename(columns={g_pid_col:"playerId"}).copy(), "go_out")

    go_out_path = os.path.join(feat_dir, "goalie_features.parquet")
    go_out.to_parquet(go_out_path, index=False)
    print("✅ Saved goalie features:", go_out.shape, "->", go_out_path)

    print("✅ Feature build complete:", feat_dir)

if __name__ == "__main__":
    main()

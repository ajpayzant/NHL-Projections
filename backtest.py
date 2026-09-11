"""Backtest the skater points projection against naive baselines.

For a held-out target season, we project it using ONLY data available before it,
then score MAE / RMSE against actual results for players who actually played.
Baselines:
  - naive1  : last season's totals = this season's projection
  - naive3  : simple 3-year average of totals
  - rate1   : last season's per-60 rate * actual TOI (isolates rate skill from volume)
  - model   : our Marcel+age projection

The model should beat naive1/naive3 on totals. We report per-60-rate accuracy too,
since usage/injury noise in totals can mask real rate-projection skill.

Run with `--lines` to score the line-chemistry adjustment as well (see `lines.py`): the
same projection, plus the change in each player's linemates, against the same actuals. That
is the standing check that the coefficient in config still earns its place.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import config as C
import data_layer as dl
import age_curves as ac


def _seasons_frame() -> pd.DataFrame:
    sk = dl.load_moneypuck_skaters()
    bios = dl.load_nhl_skater_bios()
    births = dl.player_birthdates(bios)[["playerId", "birthDate", "positionCode"]]
    df = sk.merge(births, on="playerId", how="left")
    df = df[df["icetime"] >= C.MIN_ICETIME_SEC].copy()
    df["toi_min"] = df["icetime"] / 60.0
    df["points"] = df["I_F_points"]
    df["pts_per60"] = df["points"] * 60.0 / df["toi_min"]
    pos = df["positionCode"].fillna(df["position"])
    df["pos_group"] = np.where(pos.isin(["D"]), "D", "F")
    bd = pd.to_datetime(df["birthDate"], errors="coerce")
    ref = pd.to_datetime((df["mp_season_year"] + 1).astype(str) + "-02-01", errors="coerce")
    df["age"] = (ref - bd).dt.days / 365.25
    return df


def backtest_season(df: pd.DataFrame, curves: dict, target: int,
                    toi_curve: dict | None = None) -> pd.DataFrame:
    if toi_curve is None:
        toi_curve = ac.build_toi_age_curve()
    # Match production: N-season history window (N = len(RECENCY_WEIGHTS)).
    recent_years = [target - i for i in range(1, C.SKATER_HISTORY_SEASONS + 1)]
    hist = df[df["mp_season_year"].isin(recent_years)]
    actual = df[df["mp_season_year"] == target][
        ["playerId", "points", "pts_per60", "toi_min", "games_played"]
    ].rename(columns={"points": "act_points", "pts_per60": "act_rate",
                      "toi_min": "act_toi", "games_played": "act_gp"})

    # Regression targets by position group AND usage tier (TOI/game quartile) — mirrors
    # production project_skaters._positional_means/_tier_target so the backtest reflects
    # the real model: a top-line skater regresses toward top-line rates, not the global
    # positional mean (which includes 4th-liners and under-projects stars).
    h = hist.copy()
    h["toipg"] = h["toi_min"] / h["games_played"]
    pm = {}          # global positional mean rate (fallback)
    tier_q = {}      # TOI/game quartile breakpoints per position
    tier_mean = {}   # {pos_group: {tier: mean_rate}}
    for grp, gg in h.groupby("pos_group"):
        pm[grp] = np.average(gg["pts_per60"], weights=gg["toi_min"])
        qs = gg["toipg"].quantile([0.25, 0.50, 0.75]).to_numpy()
        tier_q[grp] = qs
        tier_mean[grp] = {}
        tier = np.searchsorted(qs, gg["toipg"].to_numpy())
        for t in range(4):
            sub = gg[tier == t]
            if len(sub) >= 20:
                tier_mean[grp][t] = np.average(sub["pts_per60"], weights=sub["toi_min"])

    def _target(grp: float, toipg: float) -> float:
        t = int(np.searchsorted(tier_q[grp], toipg))
        return tier_mean[grp].get(t, pm[grp])

    wmap = dict(zip(recent_years, C.RECENCY_WEIGHTS))
    rows = []
    for pid, g in hist.groupby("playerId"):
        g = g.sort_values("mp_season_year")
        latest = g.iloc[-1]
        pos_group = latest["pos_group"]
        g = g.assign(rec_w=g["mp_season_year"].map(wmap).fillna(0.0))
        g = g.assign(blend_w=g["rec_w"] * g["toi_min"])
        if g["blend_w"].sum() <= 0:
            continue
        total_toi = g["toi_min"].sum()

        sample_toipg = np.average(g["toi_min"] / g["games_played"], weights=g["blend_w"])
        blended_rate = np.average(g["pts_per60"], weights=g["blend_w"])
        k = C.SKATER_REGRESS_TOI_MIN
        target_rate = _target(pos_group, sample_toipg)
        regressed = (blended_rate * total_toi + target_rate * k) / (total_toi + k)

        mean_age = np.average(g["age"], weights=g["blend_w"])
        target_age = mean_age + (target - np.average(g["mp_season_year"], weights=g["blend_w"]))
        mult = ac.age_multiplier(curves, "points", mean_age, target_age)
        model_rate = regressed * mult

        # Volume projection (same as production): recency-blended TOI/game, age-trended.
        # Production uses SKATER_TOI_PRIOR_MIN=0 (no tier prior on TOI), i.e. the raw
        # recency-blended own TOI — which is exactly this.
        toi_per_gp = np.average(g["toi_min"] / g["games_played"], weights=g["blend_w"])
        toi_per_gp *= ac.age_multiplier({"toi": toi_curve}, "toi", mean_age, target_age)
        gp = dict(zip(g["mp_season_year"], g["games_played"]))
        num = den = 0.0
        for i in range(len(C.GP_RECENCY_WEIGHTS)):
            yr = target - 1 - i
            if yr in gp:
                num += C.GP_RECENCY_WEIGHTS[i] * gp[yr]; den += C.GP_RECENCY_WEIGHTS[i]
        proj_gp = float(np.clip(0.80 * (num / den) + 0.20 * 70.0, 1, C.MAX_GP)) if den else 60.0
        model_pts = model_rate * (toi_per_gp * proj_gp) / 60.0

        # Baselines.
        last = g[g["mp_season_year"] == target - 1]
        naive1 = float(last["points"].iloc[0]) if len(last) else np.nan
        naive3 = float(g["points"].mean())
        last_rate = float(last["pts_per60"].iloc[0]) if len(last) else blended_rate

        rows.append({"playerId": pid, "model_pts": model_pts, "naive1": naive1,
                     "naive3": naive3, "model_rate": model_rate, "last_rate": last_rate,
                     "proj_toi": toi_per_gp * proj_gp})

    pred = pd.DataFrame(rows).merge(actual, on="playerId", how="inner")
    return pred


# --------------------------------------------------------------------------- #
# line chemistry                                                              #
# --------------------------------------------------------------------------- #
def line_deltas(target: int) -> pd.Series:
    """playerId -> change in linemate quality between last season's lines and `target`'s.

    This is the out-of-sample version of what the app's Lines page supplies by hand. The
    "new" lineup is the target season's own most-used units, which is the honest stand-in
    for a lineup card: the card is a guess at that deployment, and no part of the target
    season's SCORING enters it -- linemate quality is built from earlier seasons only.

    A player traded mid-season appears on two teams' cards, so the deltas are averaged.
    """
    import lines as ln
    cards = {}
    for team, card in ln.observed_lineups(target, "F").items():
        cards[team] = {ln.FORWARD_LABELS[i]: ids for i, ids in enumerate(card)
                       if i < len(ln.FORWARD_LABELS)}
    d = ln.delta_lq(cards, target)
    if d.empty:
        return pd.Series(dtype=float)
    return d.groupby("playerId")["lines_dlq"].mean()


def apply_lines(pred: pd.DataFrame, dlq: pd.Series, beta: float | None = None
                ) -> pd.DataFrame:
    """Add `model_rate_lines` / `model_pts_lines` to a backtest frame.

    Same arithmetic as `project_skaters._apply_line_chemistry`, with one simplification: the
    cap is taken against the player's whole points-per-60 rather than against his
    even-strength part, because the backtest never splits power play out. It binds on about
    1% of players either way.
    """
    beta = C.LINES_BETA if beta is None else beta
    out = pred.copy()
    d = out["playerId"].map(dlq).astype(float).fillna(0.0)
    d = d.clip(-C.LINES_MAX_DLQ, C.LINES_MAX_DLQ)
    adj = (beta * d).clip(-C.LINES_MAX_RATE_CHANGE * out["model_rate"],
                          C.LINES_MAX_RATE_CHANGE * out["model_rate"])
    out["lines_dlq"] = d
    out["model_rate_lines"] = out["model_rate"] + adj
    out["model_pts_lines"] = out["model_rate_lines"] * out["proj_toi"] / 60.0
    return out


def _score(pred: pd.DataFrame, col: str, actual_col: str) -> tuple[float, float]:
    d = pred.dropna(subset=[col, actual_col])
    err = d[col] - d[actual_col]
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))


def _lines_report(preds: dict[int, pd.DataFrame]) -> None:
    """Score the line adjustment on the seasons already projected above."""
    print("\n\nLine chemistry (--lines): same projection, plus the change in linemates\n")
    print(f"{'season':>7} {'n':>5} {'moved':>6} | {'rate MAE':>9} {'+lines':>8} {'gain':>7} "
          f"| {'pts MAE':>8} {'+lines':>8} {'gain':>7}")
    agg = {"rate": [], "rate_l": [], "pts": [], "pts_l": []}
    for target, pred in preds.items():
        dlq = line_deltas(target)
        p = apply_lines(pred, dlq)
        rp = p[p["act_toi"] >= 300]
        r0, _ = _score(rp, "model_rate", "act_rate")
        r1, _ = _score(rp, "model_rate_lines", "act_rate")
        p0, _ = _score(p, "model_pts", "act_points")
        p1, _ = _score(p, "model_pts_lines", "act_points")
        moved = int((p["lines_dlq"].abs() > 0.01).sum())
        agg["rate"].append(r0); agg["rate_l"].append(r1)
        agg["pts"].append(p0); agg["pts_l"].append(p1)
        print(f"{target:>7} {len(rp):>5} {moved:>6} | {r0:>9.3f} {r1:>8.3f} "
              f"{(r0 - r1) / r0:>6.1%} | {p0:>8.2f} {p1:>8.2f} {(p0 - p1) / p0:>6.1%}")
    r0, r1 = np.mean(agg["rate"]), np.mean(agg["rate_l"])
    p0, p1 = np.mean(agg["pts"]), np.mean(agg["pts_l"])
    print(f"\n{'MEAN':>7} {'':>5} {'':>6} | {r0:>9.3f} {r1:>8.3f} {(r0 - r1) / r0:>6.1%} "
          f"| {p0:>8.2f} {p1:>8.2f} {(p0 - p1) / p0:>6.1%}")
    print(f"\nbeta={C.LINES_BETA}, unit share={C.LINES_UNIT_SHARE}, "
          f"dLQ clamp=+/-{C.LINES_MAX_DLQ}, rate cap={C.LINES_MAX_RATE_CHANGE:.0%}.")
    print("'moved' counts players whose linemates changed enough to be adjusted at all; the "
          "gain is diluted across everyone else, who are untouched by construction.")
    # A negative gain here means the coefficient no longer earns its place -- say so rather
    # than leaving a reader to compare two numbers in a table.
    if p1 > p0 or r1 > r0:
        print("WARNING: the adjustment did not help on this sample. Re-fit before shipping.")


def main(with_lines: bool = False) -> None:
    """Print the whole report. A function, not module-level code, so `run.py --backtest`
    can call it instead of re-executing this file with somebody else's command line."""
    df = _seasons_frame()
    curves = ac.build_skater_age_curves()
    toi_curve = ac.build_toi_age_curve()
    test_seasons = [2022, 2023, 2024, 2025]
    preds: dict[int, pd.DataFrame] = {}

    print("Backtest: season-long POINTS (players with a real target-season sample)\n")
    print(f"{'season':>7} {'n':>5} | {'model MAE':>10} {'naive1 MAE':>11} {'naive3 MAE':>11} "
          f"| {'model RMSE':>11} {'naive1 RMSE':>12}")
    agg = {"model": [], "naive1": [], "naive3": []}
    rate_agg = {"model": [], "last": []}
    for target in test_seasons:
        pred = backtest_season(df, curves, target, toi_curve)
        preds[target] = pred
        m_mae, m_rmse = _score(pred, "model_pts", "act_points")
        n1_mae, n1_rmse = _score(pred, "naive1", "act_points")
        n3_mae, n3_rmse = _score(pred, "naive3", "act_points")
        # Rate skill: predict per-60, scored on players with >=300 target TOI.
        rp = pred[pred["act_toi"] >= 300]
        rm_mae, _ = _score(rp, "model_rate", "act_rate")
        rl_mae, _ = _score(rp, "last_rate", "act_rate")
        agg["model"].append(m_mae); agg["naive1"].append(n1_mae); agg["naive3"].append(n3_mae)
        rate_agg["model"].append(rm_mae); rate_agg["last"].append(rl_mae)
        print(f"{target:>7} {len(pred):>5} | {m_mae:>10.2f} {n1_mae:>11.2f} {n3_mae:>11.2f} "
              f"| {m_rmse:>11.2f} {n1_rmse:>12.2f}")

    print(f"\n{'MEAN':>7}       | {np.mean(agg['model']):>10.2f} {np.mean(agg['naive1']):>11.2f} "
          f"{np.mean(agg['naive3']):>11.2f}")
    print(f"\nPer-60 RATE MAE (>=300 TOI):  model {np.mean(rate_agg['model']):.3f}  "
          f"vs last-season {np.mean(rate_agg['last']):.3f}")

    if with_lines:
        _lines_report(preds)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lines", action="store_true",
                    help="also score the line-chemistry adjustment (lines.py)")
    main(with_lines=ap.parse_args().lines)

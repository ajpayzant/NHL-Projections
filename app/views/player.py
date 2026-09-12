"""One player at a time: what he has done, the ratings behind his projection, and a
way to argue with them.

The board pages answer "who" -- who scores most, who is oversubscribed, who is unsigned.
This page answers "why him", which needs a different shape: his own seasons laid out both
ways, because a total and a rate say different things. Sixty points in 82 games and sixty
points in 62 are the same total and a very different player, and only the per-60 view
separates them.

The ratings ARE the model's opinion of a player -- everything else on him is those rates
multiplied by ice time and then settled against his team. So editing a rate here is the
deepest edit in the app: it changes his claim rather than overriding its result, his
teammates resettle around him, and his floor and ceiling move with it. A season-total lock
(on the Skaters page) says "he finishes with 40 goals"; a rate edit says "he is a
1.4-goals-per-60 player now", which is the honest way to state a leap in ability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import core

# Editable skater ratings: field on the projection, label, ceiling, step. The ceilings are
# loose sanity rails, not opinions -- the point is to stop a typo, not to police a view.
SKATER_RATES = [
    ("goals", "G/60", 4.0, 0.05),
    ("primaryAssists", "A1/60", 4.0, 0.05),
    ("secondaryAssists", "A2/60", 4.0, 0.05),
    ("shots", "SOG/60", 25.0, 0.25),
    ("ixg", "ixG/60", 4.0, 0.05),
    ("pp_points", "PPP/60 of PP", 20.0, 0.25),
    ("sh_points", "SHP/60 of SH", 12.0, 0.10),
    ("blocks", "BLK/60", 12.0, 0.10),
    ("hits", "HIT/60", 20.0, 0.10),
    ("pim", "PIM/60", 20.0, 0.10),
    ("faceoffs_won", "FOW/60", 50.0, 0.50),
]

# Goalie ratings. `save_pct` is stored as a save percentage because that is how the number
# is quoted; the model keeps its complement, goals per shot.
GOALIE_RATES = [
    ("rate_sa_per_60", "Shots against /60", 60.0, 0.25,
     "how busy his net is -- mostly his team, and settled against the team's budget"),
    ("rate_so_per_start", "Shutouts per start", 0.5, 0.005,
     "a rate, not a total: .07 per start is about six in a full season"),
    ("rate_relief_per_start", "Relief appearances per start", 5.0, 0.02,
     "appearances that were not starts, per start -- how often he comes in cold"),
]

HIST_TOTALS = [
    ("season", "Season", None), ("team", "Team", None), ("gp", "GP", "%.0f"),
    ("toi_min", "TOI", "%.0f"), ("toi_per_gp", "TOI/GP", "%.1f"),
    ("goals", "G", "%.0f"), ("assists", "A", "%.0f"), ("points", "PTS", "%.0f"),
    ("shots", "SOG", "%.0f"), ("ixg", "ixG", "%.1f"),
    ("pp_points", "PPP", "%.0f"), ("sh_points", "SHP", "%.0f"),
    ("blocks", "BLK", "%.0f"), ("hits", "HIT", "%.0f"), ("pim", "PIM", "%.0f"),
    ("faceoffs_won", "FOW", "%.0f"),
]
HIST_RATES = [
    ("season", "Season", None), ("team", "Team", None), ("gp", "GP", "%.0f"),
    ("toi_min", "TOI", "%.0f"), ("toi_per_gp", "TOI/GP", "%.1f"),
    ("rate_goals", "G/60", "%.2f"), ("rate_assists", "A/60", "%.2f"),
    ("rate_points", "PTS/60", "%.2f"), ("rate_shots", "SOG/60", "%.2f"),
    ("rate_ixg", "ixG/60", "%.2f"),
    ("pp_toi_per_gp", "PP/GP", "%.2f"), ("rate_pp_points", "PPP/60", "%.2f"),
    ("rate_sh_points", "SHP/60", "%.2f"),
    ("rate_blocks", "BLK/60", "%.2f"), ("rate_hits", "HIT/60", "%.2f"),
    ("rate_pim", "PIM/60", "%.2f"), ("rate_faceoffs_won", "FOW/60", "%.2f"),
]

G_HIST_TOTALS = [
    ("season", "Season", None), ("team", "Team", None), ("gp", "GP", "%.0f"),
    ("starts", "GS", "%.0f"), ("wins", "W", "%.0f"), ("losses", "L", "%.0f"),
    ("otl", "OTL", "%.0f"), ("minutes", "MIN", "%.0f"),
    ("shots_against", "SA", "%.0f"), ("saves", "SV", "%.0f"),
    ("goals_against", "GA", "%.0f"), ("shutouts", "SO", "%.0f"),
    ("save_pct", "SV%", "%.4f"), ("gaa", "GAA", "%.2f"),
]
G_HIST_RATES = [
    ("season", "Season", None), ("team", "Team", None), ("starts", "GS", "%.0f"),
    ("minutes", "MIN", "%.0f"), ("save_pct", "SV%", "%.4f"),
    ("rate_sa_per_60", "SA/60", "%.2f"), ("gaa", "GAA", "%.2f"),
    ("rate_so_per_start", "SO/start", "%.3f"),
    ("rate_relief_per_start", "Relief/start", "%.2f"),
]


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _season_label(years: pd.Series) -> pd.Series:
    y = years.astype("Int64")
    return y.astype(str) + "-" + (y + 1).astype(str).str[-2:]


def _table(df: pd.DataFrame, spec: list[tuple[str, str, str | None]]) -> None:
    cols = [(f, label, fmt) for f, label, fmt in spec if f in df.columns]
    show = df[[f for f, _l, _fm in cols]].copy()
    cfg = {}
    for f, label, fmt in cols:
        cfg[f] = (st.column_config.TextColumn(label) if fmt is None
                  else st.column_config.NumberColumn(label, format=fmt))
    st.dataframe(show, hide_index=True, width="stretch", column_config=cfg)


def _pick(df: pd.DataFrame, kind: str) -> pd.Series | None:
    """Search-and-select over one pool. Returns the chosen row."""
    d = df.copy()
    # The dropdown is where a promotion starts, so a camp player is listed under the camp
    # he is in rather than as a free agent -- searching "EDM" has to find him.
    home = d["team"].astype(str)
    if "camp" in d and "camp_team" in d:
        camp = d["camp"].fillna(False) & (d["camp_team"].fillna("") != "")
        home = home.mask(camp, d["camp_team"].fillna("") + " camp")
    d["_label"] = (d["name"].astype(str) + "  ·  " + home
                   + (("  " + d["position"].astype(str)) if "position" in d else ""))
    d = d.sort_values(["on_roster", "name"], ascending=[False, True])
    labels = d["_label"].tolist()
    key = f"pick_{kind}"
    prior = st.session_state.get(key)
    idx = labels.index(prior) if prior in labels else 0
    choice = st.selectbox(f"{kind}", labels, index=idx, key=key,
                          label_visibility="collapsed",
                          placeholder=f"Search for a {kind.lower()}")
    if choice is None:
        return None
    return d[d["_label"] == choice].iloc[0]


def _delta(row: pd.Series, base: pd.Series | None, col: str, nd: int = 1) -> str | None:
    """What an edit did to this number, or nothing at all if it did nothing."""
    if base is None or col not in base or pd.isna(base[col]):
        return None
    gap = float(row[col]) - float(base[col])
    tol = 0.5 * 10.0 ** -nd
    return None if abs(gap) < tol else f"{gap:+.{nd}f} vs model"


def _baseline_row(base: pd.DataFrame, pid: int) -> pd.Series | None:
    hit = base[base["playerId"] == pid]
    return hit.iloc[0] if len(hit) else None


# --------------------------------------------------------------------------- #
# skater                                                                      #
# --------------------------------------------------------------------------- #
def _skater(row: pd.Series, base: pd.DataFrame) -> None:
    pid = int(row["playerId"])
    b = _baseline_row(base, pid)
    edits = core.scenario().player(pid)

    st.markdown(f"#### {row['name']} · {row['team']} {row['position']}")
    bits = [core.roster_label(row),
            f"{int(row['seasons'])} seasons of NHL history",
            f"sample {row['sample_toi_min']:,.0f} minutes"]
    if row.get("rookie"):
        bits.append("rookie prior")
    if edits:
        bits.append(core.edit_badge(len(edits)))
    st.caption(" · ".join(bits))

    metrics = [("Games", "proj_gp", 0), ("TOI/GP", "proj_toi_per_gp", 1),
               ("Goals", "proj_goals", 1), ("Assists", "proj_assists", 1),
               ("Points", "proj_points", 1), ("Shots", "proj_shots", 0),
               # The projection's OWN points per 60, not the model's `rate_points` input.
               # They are different numbers and only this one describes what is on screen
               # beside it: it is the projected points divided by the projected ice time,
               # so it moves with every rate edit, every ice-time edit and every
               # reallocation, and it equals G/60 plus A/60 by construction.
               ("PTS/60", "per60_points", 2)]
    for col, (label, key, nd) in zip(st.columns(len(metrics)), metrics):
        col.metric(label, core.num(row[key], nd),
                   delta=_delta(row, b, key, max(nd, 1)))

    tab_range, tab_hist, tab_rate, tab_how = st.tabs(
        ["Floor and ceiling", "Prior seasons", "Ratings",
         "How the projection is built"])
    with tab_range:
        _skater_range(row, b)
    with tab_hist:
        _skater_history(pid, row)
    with tab_rate:
        _skater_rate_form(row, b, edits)
    with tab_how:
        _skater_provenance(row)


# The stats worth a range, and how many decimals the range is worth quoting to. Games
# played leads, because it is the largest single source of season-total error and every
# other band on the list is partly a restatement of it.
RANGE_ROWS = [("Games played", "gp", 0), ("Points", "points", 0), ("Goals", "goals", 0),
              ("Assists", "assists", 0), ("Shots", "shots", 0),
              ("PP points", "pp_points", 0), ("Blocks", "blocks", 0),
              ("Hits", "hits", 0), ("PIM", "pim", 0)]


def _skater_range(row: pd.Series, b: pd.Series | None) -> None:
    """The 80% band around every headline total, laid out rather than mentioned.

    A projection is a mean, and a mean on its own invites being read as a prediction. The
    range is the honest object: an 80% central interval, so one season in ten finishes
    below the floor and one in ten above the ceiling. Both tails matter and are shown --
    quoting only the upside is how a projection becomes a sales pitch.
    """
    st.markdown("**An 80% range.** One season in ten finishes below the floor, one in ten "
                "above the ceiling. So the floor is not a worst case and the ceiling is "
                "not a best case: they are the edges of what would be unremarkable.")
    rows = []
    for label, stat, nd in RANGE_ROWS:
        proj = row.get("proj_gp" if stat == "gp" else f"proj_{stat}")
        lo, hi = row.get(f"{stat}_p10"), row.get(f"{stat}_p90")
        if proj is None or lo is None or pd.isna(proj) or pd.isna(lo):
            continue
        per_gp = float(proj) / float(row["proj_gp"]) if float(row["proj_gp"]) > 0 else None
        rows.append({"Stat": label, "Floor": float(lo), "Projection": float(proj),
                     "Ceiling": float(hi), "Spread": float(hi) - float(lo),
                     "Per game": None if stat == "gp" else per_gp})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Floor": st.column_config.NumberColumn("Floor (p10)", format="%.0f"),
        "Projection": st.column_config.NumberColumn(format="%.1f"),
        "Ceiling": st.column_config.NumberColumn("Ceiling (p90)", format="%.0f"),
        "Spread": st.column_config.NumberColumn("Ceiling − floor", format="%.0f"),
        "Per game": st.column_config.NumberColumn(format="%.2f")})

    gp_lo, gp_hi = float(row["gp_p10"]), float(row["gp_p90"])
    said = row.get("stated_gp_sigma")
    stated = said is not None and pd.notna(said)
    bits = [f"Games {core.band(gp_lo, gp_hi)}"]
    if b is not None and "points_p90" in b and pd.notna(b["points_p90"]):
        bits.append(f"the model's own points range was "
                    f"{core.band(b['points_p10'], b['points_p90'])}")
    st.caption(" · ".join(bits))
    if stated:
        st.caption("This games range was stated on the Ratings tab, so it is the reader's "
                   "range and not the model's — and the width of it is carried through to "
                   "every other band above, because a season whose length is uncertain "
                   "has uncertain totals.")
    else:
        st.caption(
            f"The width comes from how predictable his availability is (reliability "
            f"{row['gp_reliability']:.2f} of 1) and from the size of the total itself: "
            f"a 30-goal projection has a wider band than a 10-goal one, and further to "
            f"climb than to fall, which is why the ceiling sits further from the "
            f"projection than the floor does. State a range yourself on the Ratings tab "
            f"if you disagree about how sure this is.")


def _skater_history(pid: int, row: pd.Series) -> None:
    h = core.skater_history()
    h = h[h["playerId"] == pid].sort_values("season", ascending=False).copy()
    if h.empty:
        st.info("No NHL history — this projection comes from the rookie prior for his "
                "position and age, so the ratings tab is the only way to state a view "
                "of him.")
        return
    h["season"] = _season_label(h["season"])

    how = st.radio("How", ["Totals", "Per 60"], horizontal=True,
                   label_visibility="collapsed", key=f"hist_how_{pid}")
    if how == "Totals":
        # The projection leads the column here for the same reason it does in the per-60
        # view: a season total means something only against the seasons a reader remembers.
        totals = {"season": f"{core.SEASON_LABEL} model", "team": row["team"],
                  "gp": float(row["proj_gp"]), "toi_min": float(row["proj_toi"]),
                  "toi_per_gp": float(row["proj_toi_per_gp"]),
                  "pp_toi_per_gp": float(row["proj_pp_toi_per_gp"]),
                  **{stat: row.get(f"proj_{stat}")
                     for stat in ("goals", "assists", "points", "shots", "ixg",
                                  "pp_points", "sh_points", "blocks", "hits", "pim",
                                  "faceoffs_won")}}
        _table(pd.concat([pd.DataFrame([totals]), h], ignore_index=True), HIST_TOTALS)
        st.caption(f"What he actually did, under the {core.SEASON_LABEL} projection. Ice "
                   "time is total minutes; PPP and SHP are power-play and short-handed "
                   "points. Seasons of different lengths are on the Per 60 view.")
    else:
        # The model's own view is appended as a row rather than described in prose: the
        # only useful question about a rate is what it is next to the others.
        #
        # Every rate in it is the PROJECTION divided by the projected ice time, not the
        # `rate_*` input it was built from. That is what makes the row comparable with the
        # seasons under it, which are also totals divided by minutes played -- and it is
        # what keeps the row internally consistent: `rate_points` is blended separately
        # from goals and assists, so a row mixing the two showed a PTS/60 that was not
        # G/60 plus A/60.
        model = {"season": f"{core.SEASON_LABEL} model", "team": row["team"],
                 "gp": float(row["proj_gp"]), "toi_min": float(row["proj_toi"]),
                 "toi_per_gp": float(row["proj_toi_per_gp"]),
                 "pp_toi_per_gp": float(row["proj_pp_toi_per_gp"]),
                 **{f"rate_{stat}": row.get(f"per60_{stat}")
                    for stat in ("goals", "assists", "points", "shots", "ixg",
                                 "pp_points", "sh_points", "blocks", "hits", "pim",
                                 "faceoffs_won")}}
        _table(pd.concat([pd.DataFrame([model]), h], ignore_index=True), HIST_RATES)
        st.caption(
            "Per 60 minutes of ice time, except PPP and SHP which are per 60 minutes of "
            "power play and short handed — a player's points per 60 OF POWER PLAY is a "
            "skill that travels with him, where his PP points per 60 of total ice time "
            "is mostly a statement about how much power-play time his last coach gave "
            f"him. The top row is the {core.SEASON_LABEL} projection put back into the "
            "same units — his projected totals divided by his projected minutes, after "
            "his team's budget has settled — so it can be read straight down the column "
            "against seasons he actually played. Underneath it is a recency-weighted, "
            "age-adjusted blend of those seasons, regressed toward players of the same "
            "position and usage by an amount that depends on how many minutes there are "
            "to learn from; the Ratings tab shows that blend before settlement, which is "
            "the number to edit.")

    st.divider()
    _spark(h, pid)


def _spark(h: pd.DataFrame, pid: int) -> None:
    """One rate over time. A trend is the thing a table of five seasons hides worst."""
    options = {"Points per 60": "rate_points", "Goals per 60": "rate_goals",
               "Shots per 60": "rate_shots", "Expected goals per 60": "rate_ixg",
               "Ice time per game": "toi_per_gp", "PP time per game": "pp_toi_per_gp"}
    pick = st.selectbox("Trend", list(options), key=f"spark_{pid}",
                        label_visibility="collapsed")
    col = options[pick]
    if col not in h.columns:
        return
    d = h[["season", col]].dropna().sort_values("season")
    if len(d) < 2:
        st.caption("Not enough seasons to draw a trend.")
        return
    st.line_chart(d.set_index("season")[col], height=200)


def _skater_rate_form(row: pd.Series, b: pd.Series | None, edits: dict) -> None:
    pid = int(row["playerId"])
    st.markdown("**Ratings.** These are the per-60 rates the whole projection is built "
                "from. Change one and his claim changes at the root: totals, floor, "
                "ceiling and his teammates' share of the team budget all move with it.")

    values: dict[str, float] = {}
    cols = st.columns(4)
    for i, (stat, label, hi, step) in enumerate(SKATER_RATES):
        field = f"rate_{stat}"
        cur = float(row[field])
        model = float(b[field]) if b is not None and field in b else cur
        values[stat] = cols[i % 4].number_input(
            label, 0.0, hi, cur, step, format="%.3f", key=f"rt_{stat}_{pid}",
            help=f"model says {model:.3f}")

    st.markdown("**Volume.** The rates say how good he is; these say how much of the "
                "season he plays. Both are needed for a total.")
    c1, c2, c3 = st.columns(3)
    gp = c1.number_input("Games played", 0.0, float(core.C.MAX_GP),
                         float(row["proj_gp"]), 1.0, key=f"rgp_{pid}")
    toi = c2.number_input("TOI per game", 0.0, 30.0, float(row["proj_toi_per_gp"]),
                          0.25, key=f"rtoi_{pid}")
    pp = c3.number_input("PP TOI per game", 0.0, 8.0,
                         float(row["proj_pp_toi_per_gp"]), 0.25, key=f"rpp_{pid}")

    # A range instead of a number, for the one input where a reader almost never believes
    # a point estimate. Stating 82 games says he plays all of them and nothing can go
    # wrong; stating 70 to 82 says the same expectation with the doubt left in, and the
    # doubt is carried into the points and goals bands as well as the games one.
    use_range = st.checkbox(
        "State a games-played range instead", value="gp_low" in edits,
        key=f"rgpr_{pid}",
        help="Games played is the largest single source of season-total error, so it is "
             "the input where saying how sure you are matters most.")
    lo = hi = None
    if use_range:
        d1, d2, d3 = st.columns([1, 1, 2])
        cur_lo = float(edits.get("gp_low", max(row["gp_p10"], 0.0)))
        cur_hi = float(edits.get("gp_high", row["gp_p90"]))
        lo = d1.number_input("Games, floor", 0.0, float(core.C.MAX_GP), cur_lo, 1.0,
                             key=f"rgplo_{pid}")
        hi = d2.number_input("Games, ceiling", 0.0, float(core.C.MAX_GP), cur_hi, 1.0,
                             key=f"rgphi_{pid}")
        d3.caption(f"An 80% range: the projection becomes the midpoint "
                   f"({(min(lo, hi) + max(lo, hi)) / 2:.0f} games) and the width sets how "
                   f"wide every other band on the Floor and ceiling tab is. The games box "
                   f"above is ignored while this is ticked.")

    st.markdown("**Team.** Where he plays, which is a structural edit rather than a "
                "rating: it moves him into the destination team's lineup and its budget, "
                "and off the one he was on.")
    t1, t2 = st.columns([1, 2])
    teams = core.team_options(core.skaters()[0])
    cur_team = str(row["team"])
    idx = teams.index(cur_team) + 1 if cur_team in teams else 0
    team = t1.selectbox("Team", ["(not on a roster)"] + teams, index=idx, key=f"rtm_{pid}")
    t2.caption("Signing a free agent or moving a player takes effect everywhere at once: "
               "he becomes selectable in the destination team's lines, his production "
               "counts against that team's budget, and the team he left has his minutes "
               "back to give to somebody else. Putting him back to *(not on a roster)* "
               "takes him off every team budget without deleting anything else you have "
               "said about him.")

    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("Save ratings", type="primary", key=f"rsave_{pid}"):
        patch: dict = {}
        for stat, v in values.items():
            field = f"rate_{stat}"
            if abs(float(v) - float(row[field])) >= 5e-4:
                patch[field] = float(v)
        for value, field, now in [(toi, "toi_per_gp", row["proj_toi_per_gp"]),
                                  (pp, "pp_toi_per_gp", row["proj_pp_toi_per_gp"])]:
            if abs(float(value) - float(now)) >= 0.01:
                patch[field] = float(value)
        # The two ways of stating availability are exclusive, and switching between them
        # clears the other, or a stale `gp` lock would silently outrank the new range.
        if use_range:
            patch.update(gp=None, gp_low=float(min(lo, hi)), gp_high=float(max(lo, hi)))
        else:
            patch.update(gp_low=None, gp_high=None)
            if abs(float(gp) - float(row["proj_gp"])) >= 0.01:
                patch["gp"] = float(gp)
        want_team = None if team.startswith("(") else team
        if want_team != (cur_team if cur_team in teams else None):
            patch["team"] = want_team or cur_team
            patch["on_roster"] = want_team is not None
        # `None` means "forget this edit", so a patch of nothing but Nones over fields
        # that were never set is not a change and must not read as one.
        if any(v is not None or k in edits for k, v in patch.items()):
            core.edit_player(pid, **patch)
        else:
            st.info("Nothing changed.")
    rate_edits = {k: v for k, v in edits.items() if k.startswith("rate_")}
    if rate_edits and b2.button("Ratings back to the model", key=f"rclr_{pid}"):
        core.edit_player(pid, **{k: None for k in rate_edits})
    if edits:
        st.caption("Currently overriding: "
                   + ", ".join(f"{k} = {v}" for k, v in sorted(edits.items())))
    st.caption("A rate edit is a statement about ability and still settles against the "
               "team budget, so the total it produces can come in under what the rate "
               "alone implies. To state a total outright instead, lock it on the "
               "Skaters page — a lock is taken out of the budget first, at face value.")


def _skater_provenance(row: pd.Series) -> None:
    st.caption("Rate times ice time is a CLAIM; the projection is that claim settled "
               "against what his team has to give. The gap is the budget at work.")
    rows = []
    for stat, label in [("gp", "Games"), ("toi", "Ice time (min)"), ("goals", "Goals"),
                        ("primaryAssists", "Primary assists"),
                        ("secondaryAssists", "Secondary assists"), ("points", "Points"),
                        ("shots", "Shots"), ("ixg", "Expected goals"),
                        ("pp_points", "PP points"), ("sh_points", "SH points"),
                        ("blocks", "Blocks"), ("hits", "Hits"), ("pim", "PIM"),
                        ("faceoffs_won", "Faceoffs won")]:
        unc, proj = row.get(f"unc_{stat}"), row.get(f"proj_{stat}")
        if unc is None or proj is None or pd.isna(unc) or pd.isna(proj):
            continue
        unc, proj = float(unc), float(proj)
        rows.append({"Stat": label + (" (locked)" if row.get(f"lock_{stat}") else ""),
                     "Rate": float(row.get(f"rate_{stat}", np.nan)),
                     "Claim": unc, "Projection": proj, "Change": proj - unc})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Rate": st.column_config.NumberColumn("Per 60", format="%.2f"),
        "Claim": st.column_config.NumberColumn(format="%.1f"),
        "Projection": st.column_config.NumberColumn(format="%.1f"),
        "Change": st.column_config.NumberColumn(format="%+.1f")})
    st.caption(
        f"Usage tier {int(row['usage_tier'])} of 4 · games-played reliability "
        f"{row['gp_reliability']:.2f} · sample {row['sample_toi_min']:,.0f} minutes at "
        f"{row['sample_toi_per_gp']:.1f} a game. The goal rate is "
        f"{core.C.GOALS_XG_WEIGHT:.0%} expected goals and "
        f"{1 - core.C.GOALS_XG_WEIGHT:.0%} his own finishing, because shot quality "
        f"repeats better than shooting percentage does — his own goals-per-60 before "
        f"that blend was {row['rate_goals_own']:.2f}.")


# --------------------------------------------------------------------------- #
# goalie                                                                      #
# --------------------------------------------------------------------------- #
def _goalie(row: pd.Series, base: pd.DataFrame) -> None:
    pid = int(row["playerId"])
    b = _baseline_row(base, pid)
    edits = core.scenario().goalie(pid)

    st.markdown(f"#### {row['name']} · {row['team']}")
    bits = [core.roster_label(row),
            f"{int(row['seasons'])} seasons of history",
            f"sample {row['sample_shots']:,.0f} shots faced"]
    if row.get("rookie"):
        bits.append("rookie prior")
    if edits:
        bits.append(core.edit_badge(len(edits)))
    st.caption(" · ".join(bits))

    metrics = [("Starts", "proj_starts", 1), ("Games", "proj_gp", 1),
               ("Wins", "proj_wins", 1), ("Shutouts", "proj_shutouts", 1),
               ("Saves", "proj_saves", 0)]
    cols = st.columns(len(metrics) + 2)
    for col, (label, key, nd) in zip(cols, metrics):
        col.metric(label, core.num(row[key], nd), delta=_delta(row, b, key))
    cols[-2].metric("SV%", core.sv(row["proj_save_pct"]))
    cols[-1].metric("GAA", f"{float(row['proj_gaa']):.2f}")
    st.caption(f"Starts {core.band(row['starts_p10'], row['starts_p90'])} floor to "
               f"ceiling · wins {core.band(row['wins_p10'], row['wins_p90'])} · SV% "
               f"{core.sv(row['save_pct_p10'])} to {core.sv(row['save_pct_p90'])}")

    tab_band, tab_hist, tab_rate, tab_how = st.tabs(
        ["Floor and ceiling", "Prior seasons", "Ratings",
         "How the projection is built"])
    with tab_band:
        _goalie_range(row, b)
    with tab_hist:
        _goalie_history(pid, row)
    with tab_rate:
        _goalie_rate_form(row, b, edits)
    with tab_how:
        _goalie_provenance(row)


def _goalie_range(row: pd.Series, b: pd.Series | None) -> None:
    """The same 80% band for a goalie, where the workload question is the whole question.

    Starts are the one band here that is NOT parametric: a backup's season is bimodal --
    either the starter stays healthy or it isn't a backup season at all -- and the band comes
    from a measured quantile table instead of a shape. Save percentage is the opposite case:
    a bounded rate with a symmetric band, and the rate a reader should argue with rather than
    the totals it produces.
    """
    st.markdown("**An 80% range.** One season in ten finishes below the floor, one in ten "
                "above the ceiling. For a goalie the width is mostly a workload question: "
                "how many starts he actually gets is a bigger unknown than how well he "
                "stops the puck.")
    rows = []
    for label, stat, nd in [("Starts", "starts", 1), ("Wins", "wins", 1),
                            ("Saves", "saves", 0), ("Shutouts", "shutouts", 1)]:
        proj, lo, hi = (row.get(f"proj_{stat}"), row.get(f"{stat}_p10"),
                        row.get(f"{stat}_p90"))
        if proj is None or lo is None or pd.isna(proj) or pd.isna(lo):
            continue
        per = float(proj) / float(row["proj_starts"]) if float(row["proj_starts"]) > 0 \
            else None
        rows.append({"Stat": label, "Floor": float(lo), "Projection": float(proj),
                     "Ceiling": float(hi), "Spread": float(hi) - float(lo),
                     "Per start": None if stat == "starts" else per})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Floor": st.column_config.NumberColumn("Floor (p10)", format="%.0f"),
        "Projection": st.column_config.NumberColumn(format="%.1f"),
        "Ceiling": st.column_config.NumberColumn("Ceiling (p90)", format="%.0f"),
        "Spread": st.column_config.NumberColumn("Ceiling − floor", format="%.0f"),
        "Per start": st.column_config.NumberColumn(format="%.2f")})

    # The two rates, in their own units: a save percentage read to four places and a GAA to
    # two do not belong in a column of counting totals rounded to whole numbers.
    st.dataframe(pd.DataFrame([
        {"Rate": "Save percentage", "Floor": core.sv(row["save_pct_p10"]),
         "Projection": core.sv(row["proj_save_pct"]),
         "Ceiling": core.sv(row["save_pct_p90"])},
        {"Rate": "GAA", "Floor": f"{float(row['gaa_p10']):.2f}",
         "Projection": f"{float(row['proj_gaa']):.2f}",
         "Ceiling": f"{float(row['gaa_p90']):.2f}"}]),
        hide_index=True, width="stretch")
    st.caption("The GAA floor is the GOOD end: it is drawn from the save-percentage ceiling "
               "at the shots he is projected to face, so the two rows can never tell "
               "different stories. State his starts or his save percentage on the Ratings "
               "tab and the band around the stated number closes to nothing — an outright "
               "statement is not uncertain.")
    if b is not None and "wins_p90" in b and pd.notna(b["wins_p90"]):
        st.caption(f"The model's own wins range was "
                   f"{core.band(b['wins_p10'], b['wins_p90'], 1)}.")


def _goalie_history(pid: int, row: pd.Series) -> None:
    h = core.goalie_history()
    h = h[h["playerId"] == pid].sort_values("season", ascending=False).copy()
    if h.empty:
        st.info("No NHL history — this projection comes from the rookie goalie prior.")
        return
    h["season"] = _season_label(h["season"])
    how = st.radio("How", ["Totals", "Per 60 and per start"], horizontal=True,
                   label_visibility="collapsed", key=f"ghist_how_{pid}")
    if how == "Totals":
        proj = {"season": f"{core.SEASON_LABEL} model", "team": row["team"],
                "save_pct": float(row["proj_save_pct"]), "gaa": float(row["proj_gaa"]),
                **{stat: row.get(f"proj_{stat}")
                   for stat in ("gp", "starts", "wins", "losses", "otl", "minutes",
                                "shots_against", "saves", "goals_against", "shutouts")}}
        _table(pd.concat([pd.DataFrame([proj]), h], ignore_index=True), G_HIST_TOTALS)
        st.caption(f"The {core.SEASON_LABEL} projection on top, then what he actually did. "
                   "A traded goalie's row lists every team he played for that season; the "
                   "source does not split the totals.")
    else:
        model = {"season": f"{core.SEASON_LABEL} model", "team": row["team"],
                 "starts": float(row["proj_starts"]),
                 "minutes": float(row["proj_minutes"]),
                 "save_pct": float(1.0 - row["rate_ga_per_shot"]),
                 "rate_sa_per_60": float(row["rate_sa_per_60"]),
                 "gaa": float(row["proj_gaa"]),
                 "rate_so_per_start": float(row["rate_so_per_start"]),
                 "rate_relief_per_start": float(row["rate_relief_per_start"])}
        _table(pd.concat([pd.DataFrame([model]), h], ignore_index=True), G_HIST_RATES)
        st.caption(
            "Save percentage and shots against per 60 are the two ratings that carry a "
            "goalie: one is him, the other is mostly his team. Shutouts and relief "
            "appearances are per start, so they survive a change of workload. The top "
            f"row is what the model carries into {core.SEASON_LABEL} — a shot-weighted, "
            "age-adjusted blend regressed on roughly "
            f"{core.C.GOALIE_REGRESS_SHOTS:,.0f} shots, which is about a season and a "
            "half of starting.")
    st.divider()
    opts = {"Save percentage": "save_pct", "Shots against per 60": "rate_sa_per_60",
            "Starts": "starts", "GAA": "gaa"}
    pick = st.selectbox("Trend", list(opts), key=f"gspark_{pid}",
                        label_visibility="collapsed")
    d = h[["season", opts[pick]]].dropna().sort_values("season")
    if len(d) >= 2:
        st.line_chart(d.set_index("season")[opts[pick]], height=200)


def _goalie_rate_form(row: pd.Series, b: pd.Series | None, edits: dict) -> None:
    pid = int(row["playerId"])
    st.markdown("**Ratings.** Save percentage is the goalie; the rest is his workload. "
                "Every one of these is a rate, so stating it moves goals against, saves, "
                "GAA and his partner's share of the crease together.")
    c1, c2, c3, c4 = st.columns(4)
    sv_cur = float(1.0 - row["rate_ga_per_shot"])
    sv_model = float(1.0 - b["rate_ga_per_shot"]) if b is not None else sv_cur
    sv_new = c1.number_input("Save percentage", 0.800, 0.960, sv_cur, 0.001,
                             format="%.4f", key=f"gsv_{pid}",
                             help=f"model says {sv_model:.4f}")
    boxes = [c2, c3, c4]
    values: dict[str, float] = {}
    for box, (field, label, hi, step, tip) in zip(boxes, GOALIE_RATES):
        cur = float(row[field])
        model = float(b[field]) if b is not None and field in b else cur
        values[field] = box.number_input(
            label, 0.0, hi, cur, step, format="%.3f", key=f"g{field}_{pid}",
            help=f"{tip} · model says {model:.3f}")

    st.markdown("**Workload.** Starts is the one number a reader usually knows better "
                "than the model, because it is a depth-chart decision rather than a "
                "measurement.")
    d1, d2 = st.columns([1, 1.4])
    starts = d1.number_input("Starts", 0.0, float(core.C.SEASON_GAMES),
                             float(row["proj_starts"]), 1.0, key=f"gst_{pid}")
    teams = core.team_options(core.goalies()[0])
    cur_team = str(row["team"])
    idx = teams.index(cur_team) + 1 if cur_team in teams else 0
    team = d2.selectbox("Team", ["(not on a roster)"] + teams, index=idx,
                        key=f"gtm_{pid}")

    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("Save ratings", type="primary", key=f"gsave_{pid}"):
        patch: dict = {}
        if abs(sv_new - sv_cur) >= 5e-5:
            patch["save_pct"] = float(sv_new)
        for field, v in values.items():
            if abs(float(v) - float(row[field])) >= 5e-4:
                patch[field] = float(v)
        if abs(float(starts) - float(row["proj_starts"])) >= 0.5:
            patch["starts"] = float(starts)
        want_team = None if team.startswith("(") else team
        if want_team != (cur_team if cur_team in teams else None):
            patch["team"] = want_team or cur_team
            patch["on_roster"] = want_team is not None
        if patch:
            core.edit_goalie(pid, **patch)
        else:
            st.info("Nothing changed.")
    if edits and b2.button("Back to the model", key=f"gclr_{pid}"):
        core.commit(core.scenario().clear_goalie(pid),
                    f"{row['name']} back to the model")
    if edits:
        st.caption("Currently overriding: "
                   + ", ".join(f"{k} = {v}" for k, v in sorted(edits.items())))


def _goalie_provenance(row: pd.Series) -> None:
    st.caption("A goalie's claim is a share of his team's starts, then everything else "
               "follows from the rates: minutes from appearances, shots from minutes, "
               "goals from shots.")
    rows = []
    for stat, label in [("starts", "Starts"), ("gp", "Appearances"),
                        ("minutes", "Minutes"), ("shots_against", "Shots against"),
                        ("goals_against", "Goals against")]:
        unc, proj = row.get(f"unc_{stat}"), row.get(f"proj_{stat}")
        if unc is None or proj is None or pd.isna(unc) or pd.isna(proj):
            continue
        rows.append({"Stat": label + (" (locked)" if row.get(f"lock_{stat}") else ""),
                     "Claim": float(unc), "Projection": float(proj),
                     "Change": float(proj) - float(unc)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Claim": st.column_config.NumberColumn(format="%.1f"),
        "Projection": st.column_config.NumberColumn(format="%.1f"),
        "Change": st.column_config.NumberColumn(format="%+.1f")})
    st.caption(
        f"His claim on the crease is {float(row['claim_start_share']):.1%} of his team's "
        f"starts. A projected number-one takes well under 82 of them because the number "
        f"is an expectation and prices in injury and losing the job — state a specific "
        f"depth-chart plan with the starts box if that is not what you mean. His "
        f"projected save percentage sits a point or two off his own rate because it is "
        f"settled against what his team is expected to concede: that gap is the shot "
        f"quality he actually faces.")


# --------------------------------------------------------------------------- #
def page() -> None:
    core.scenario_bar()
    core.header(f"Player dashboard · {core.SEASON_LABEL}",
                "One player at a time: his own seasons as totals and as rates, and the "
                "ratings the projection is built from.")

    c1, c2 = st.columns([1, 4])
    kind = c1.radio("Kind", ["Skater", "Goalie"], label_visibility="collapsed")
    with c2:
        if kind == "Skater":
            sk, _ = core.skaters()
            row = _pick(sk, "Skater")
        else:
            g, _ = core.goalies()
            row = _pick(g, "Goalie")
    if row is None:
        st.info("Pick a player.")
        return

    st.divider()
    if kind == "Skater":
        _skater(row, core.baseline_skaters()[0])
    else:
        _goalie(row, core.baseline_goalies()[0])

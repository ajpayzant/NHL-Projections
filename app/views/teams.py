"""Teams: the constraint made visible.

Every skater projection on this page was settled against these numbers, so this is where
a reader finds out WHY a player came in under his own rates -- his team is oversubscribed,
or his team's roster is short and the model is holding minutes back for the call-ups who
will actually play them. It is also the right place to edit, because a stated team total
redistributes the whole roster instead of nudging one name.

Four questions a reader brings to a team, and a tab each. What does the model expect of
it (Budget). Who gets that expectation (Roster). Is the expectation plausible given what
this team has actually done (History) -- a budget of 250 goals means one thing for a team
that scored 260, 255 and 248, and something else entirely for one that has not passed 230
since 2019. And how is the roster shaped underneath the totals (Review): a team can hit
its budget with a balanced four lines or with two players and a hole, and only the
distribution of ice time says which.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import core

# (budget column, label, decimals). Ice time first: it is the currency everything else
# is bought with.
SKATER_ROWS = [
    ("toi_min", "Ice time (min)", 0), ("goals", "Goals", 0), ("assists", "Assists", 0),
    ("points", "Points", 0), ("shots", "Shots", 0), ("ixg", "Expected goals", 0),
    ("pp_points", "PP points", 0), ("sh_points", "SH points", 0),
    ("blocks", "Blocks", 0), ("hits", "Hits", 0), ("pim", "PIM", 0),
    ("faceoffs_won", "Faceoffs won", 0),
]
GOALIE_ROWS = [
    ("starts", "Starts", 0), ("appearances", "Appearances", 0), ("minutes", "Minutes", 0),
    ("wins", "Wins", 1), ("shots_against", "Shots against", 0),
    ("goals_against", "Goals against", 0), ("shutouts", "Shutouts", 1),
]
# Budgets worth arguing with. Goals cascades: assists and points follow it.
EDITABLE = [("goals", "Team goals", 400.0), ("shots", "Team shots", 4000.0),
            ("blocks", "Team blocks", 2500.0), ("hits", "Team hits", 3500.0),
            ("pim", "Team PIM", 1800.0)]
EDITABLE_G = [("goalie_wins", "Team wins", 84.0),
              ("goalie_shots_against", "Shots against", 4000.0),
              ("goalie_goals_against", "Goals against", 400.0)]


def _overview(sk: pd.DataFrame, tb: pd.DataFrame, g: pd.DataFrame,
              gb: pd.DataFrame) -> None:
    on = sk[sk["on_roster"]]
    th = core.team_history()
    recent = th[th["season"] >= th["season"].max() - 2]
    hist3 = recent.groupby("team")[["gf_pg", "ga_pg", "xg_share"]].mean()
    sched = core.schedule().groupby("team")["sos_factor"].first()

    rows = []
    for team in tb.index:
        s = on[on["team"] == team]
        gs = g[(g["team"] == team) & g["on_roster"]]
        games = float(tb.at[team, "games"])
        rows.append({
            "Team": team, "Skaters": len(s), "Goalies": len(gs),
            "Roster covers": float(tb.at[team, "toi_coverage"]),
            "Crease covers": float(gb.at[team, "coverage"]) if team in gb.index else float("nan"),
            "Goals": float(s["proj_goals"].sum()),
            "Points": float(s["proj_points"].sum()),
            "Wins": float(gs["proj_wins"].sum()),
            "SV%": (1.0 - gs["proj_goals_against"].sum() / gs["proj_shots_against"].sum())
                   if float(gs["proj_shots_against"].sum()) > 0 else float("nan"),
            # The budget, and the three seasons it is meant to be a forecast of.
            "GF/GP budget": float(tb.at[team, "goals"]) / games if games else float("nan"),
            "GF/GP 3yr": float(hist3["gf_pg"].get(team, float("nan"))),
            "GA/GP budget": (float(gb.at[team, "goals_against"]) / games
                             if team in gb.index and games else float("nan")),
            "GA/GP 3yr": float(hist3["ga_pg"].get(team, float("nan"))),
            "xG% 3yr": float(hist3["xg_share"].get(team, float("nan"))),
            "Schedule": float(sched.get(team, float("nan"))),
            "Held back (min)": float(tb.at[team, "depth_toi"]),
        })
    df = pd.DataFrame(rows).sort_values("Points", ascending=False)
    st.dataframe(df, hide_index=True, width="stretch", height=430, column_config={
        "Roster covers": st.column_config.ProgressColumn(
            format="%.1f%%", min_value=0.0, max_value=1.0,
            help="share of the team's ice time the listed roster accounts for"),
        "Crease covers": st.column_config.ProgressColumn(
            format="%.1f%%", min_value=0.0, max_value=1.0),
        "Goals": st.column_config.NumberColumn(format="%.0f"),
        "Points": st.column_config.NumberColumn(format="%.0f"),
        "Wins": st.column_config.NumberColumn(format="%.1f"),
        "SV%": st.column_config.NumberColumn(format="%.4f"),
        "GF/GP budget": st.column_config.NumberColumn(format="%.2f"),
        "GF/GP 3yr": st.column_config.NumberColumn(
            format="%.2f", help="its own average over the last three seasons"),
        "GA/GP budget": st.column_config.NumberColumn(format="%.2f"),
        "GA/GP 3yr": st.column_config.NumberColumn(format="%.2f"),
        "xG% 3yr": st.column_config.NumberColumn(
            format="%.3f", help="share of the expected goals in its games, three-year "
                                "average — the most persistent thing a team does"),
        "Schedule": st.column_config.NumberColumn(
            format="%.3f", help="above 1.000 is an easier schedule to score on"),
        "Held back (min)": st.column_config.NumberColumn(
            format="%.0f", help="ice time reserved for players not on the listed roster")})
    st.caption(
        f"A real team-season uses about 28 skaters and 3 goalies; these rosters list "
        f"{len(on) / len(tb):.1f} and {len(g[g['on_roster']]) / len(tb):.1f}. The shortfall "
        "is held back rather than handed to the listed players, which is why nobody here is "
        "projected as if he plays every shift of a 28-man workload. The budget columns sit "
        "next to the team's own last three seasons so a forecast that has drifted a long "
        "way from the team's history is visible without opening the team.")

    st.divider()
    st.markdown("**The league, by what persists**")
    st.caption("Share of expected goals against goal differential, three-season averages. "
               "xG share is roughly twice as repeatable as goal differential, so a team "
               "well right of the diagonal has been outscoring its chances and is the one "
               "whose repeat to doubt.")
    scat = recent.groupby("team").agg(xg_share=("xg_share", "mean"),
                                      goal_diff=("gdiff_pg", "mean")).reset_index()
    st.scatter_chart(scat, x="xg_share", y="goal_diff", height=320)


def _budget_table(rows, budgets: pd.Series, allocated: dict[str, float],
                  cov_col) -> pd.DataFrame:
    out = []
    for col, label, nd in rows:
        if col not in budgets.index:
            continue
        full = float(budgets[col])
        eff = float(budgets.get(f"eff_{col}", full * cov_col))
        got = allocated.get(col, float("nan"))
        out.append({"": label, "Team budget": full, "After coverage": eff,
                    "Given to listed players": got, "Difference": got - eff})
    return pd.DataFrame(out)


def _team_page(team: str, sk: pd.DataFrame, tb: pd.DataFrame, g: pd.DataFrame,
               gb: pd.DataFrame) -> None:
    on = sk[(sk["team"] == team) & sk["on_roster"]]
    gs = g[(g["team"] == team) & g["on_roster"]]
    b, gbb = tb.loc[team], (gb.loc[team] if team in gb.index else None)

    m = st.columns(7)
    m[0].metric("Skaters listed", len(on))
    m[1].metric("Goalies listed", len(gs))
    m[2].metric("Roster covers", f"{float(b['toi_coverage']):.0%}",
                help="share of the team's ice time the listed roster accounts for")
    m[3].metric("Goals", core.num(on["proj_goals"].sum(), 0))
    m[4].metric("Points", core.num(on["proj_points"].sum(), 0))
    m[5].metric("Wins", core.num(gs["proj_wins"].sum(), 1))
    # Last season's record, up front. The projected wins beside it are the comparison a
    # reader makes first, and making them look it up on another tab is how a projection
    # gets read without the one number that anchors it.
    last = core.team_history()
    last = last[(last["team"] == team) & last["record"].notna()]
    if not last.empty:
        r = last.iloc[0]
        m[6].metric(f"{int(r['season'])}-{str(int(r['season']) + 1)[-2:]} record",
                    str(r["record"]), delta=f"{int(r['pts'])} pts", delta_color="off")

    tab_b, tab_r, tab_h, tab_v, tab_s, tab_e = st.tabs(
        ["Budget", "Roster", "Team history", "Roster review", "Schedule",
         "Edit the budget"])

    with tab_b:
        st.caption("The budget is what the team is expected to produce. Coverage is how "
                   "much of it the listed roster can account for; the rest is held back "
                   "for players not yet named. 'Given to listed players' should match "
                   "'after coverage' — when it does, the accounting closed.")
        alloc = {col: float(on[f"proj_{col}"].sum()) for col, _, _ in SKATER_ROWS
                 if f"proj_{col}" in on.columns}
        alloc["toi_min"] = float(on["proj_toi"].sum())
        st.dataframe(_budget_table(SKATER_ROWS, b, alloc, float(b["toi_coverage"])),
                     hide_index=True, width="stretch", column_config={
                         "Team budget": st.column_config.NumberColumn(format="%.0f"),
                         "After coverage": st.column_config.NumberColumn(format="%.0f"),
                         "Given to listed players": st.column_config.NumberColumn(format="%.0f"),
                         "Difference": st.column_config.NumberColumn(format="%+.1f")})
        st.caption(f"Games: {float(b['games']):.0f} · dressed skater-games "
                   f"{float(b['skater_games']):,.0f} · power play "
                   f"{float(b['pp_toi_min']):,.0f} min "
                   f"({float(b['pp_coverage']):.0%} covered) · penalty kill "
                   f"{float(b['sh_toi_min']):,.0f} min "
                   f"({float(b['sh_coverage']):.0%} covered) · "
                   f"{float(b['depth_toi']):,.0f} minutes and "
                   f"{float(b['depth_gp']):,.0f} skater-games held back.")

        if gbb is not None:
            st.markdown("**Crease**")
            galloc = {"starts": float(gs["proj_starts"].sum()),
                      "appearances": float(gs["proj_gp"].sum()),
                      "minutes": float(gs["proj_minutes"].sum()),
                      "wins": float(gs["proj_wins"].sum()),
                      "shots_against": float(gs["proj_shots_against"].sum()),
                      "goals_against": float(gs["proj_goals_against"].sum()),
                      "shutouts": float(gs["proj_shutouts"].sum())}
            gt = _budget_table(GOALIE_ROWS, gbb, galloc, float(gbb["coverage"]))
            st.dataframe(gt, hide_index=True, width="stretch", column_config={
                "Team budget": st.column_config.NumberColumn(format="%.0f"),
                "After coverage": st.column_config.NumberColumn(format="%.0f"),
                "Given to listed players": st.column_config.NumberColumn(format="%.0f"),
                "Difference": st.column_config.NumberColumn(format="%+.1f")})
            st.caption(f"Team save percentage {core.sv(gbb['sv_pct'])}, GAA "
                       f"{float(gbb['gaa']):.2f}, crease coverage "
                       f"{float(gbb['coverage']):.0%} — "
                       f"{float(gbb['depth_starts']):.0f} starts held back.")

    with tab_r:
        show = pd.DataFrame({
            "Player": on["name"], "Pos": on["position"], "GP": on["proj_gp"],
            "TOI/GP": on["proj_toi_per_gp"], "PP/GP": on["proj_pp_toi_per_gp"],
            "G": on["proj_goals"], "A": on["proj_assists"], "PTS": on["proj_points"],
            "SOG": on["proj_shots"], "BLK": on["proj_blocks"], "HIT": on["proj_hits"],
            # The projection's own per-60, so it describes the PTS beside it and moves
            # when an edit moves them. `rate_points` is the input rate and does not.
            "PTS/60": on["per60_points"], "Edited": on["edited"],
        }).sort_values("PTS", ascending=False)
        st.dataframe(show, hide_index=True, width="stretch", height=380, column_config={
            "GP": st.column_config.NumberColumn(format="%.0f"),
            "TOI/GP": st.column_config.NumberColumn(format="%.1f"),
            "PP/GP": st.column_config.NumberColumn(format="%.2f"),
            "G": st.column_config.NumberColumn(format="%.1f"),
            "A": st.column_config.NumberColumn(format="%.1f"),
            "PTS": st.column_config.NumberColumn(format="%.1f"),
            "SOG": st.column_config.NumberColumn(format="%.0f"),
            "BLK": st.column_config.NumberColumn(format="%.0f"),
            "HIT": st.column_config.NumberColumn(format="%.0f"),
            "PTS/60": st.column_config.NumberColumn(format="%.2f")})
        gshow = pd.DataFrame({
            "Goalie": gs["name"], "Share": gs["claim_start_share"],
            "GS": gs["proj_starts"], "W": gs["proj_wins"], "SV%": gs["proj_save_pct"],
            "GAA": gs["proj_gaa"],
        }).sort_values("GS", ascending=False)
        st.dataframe(gshow, hide_index=True, width="stretch", column_config={
            "Share": st.column_config.ProgressColumn(format="%.3f", min_value=0.0,
                                                     max_value=1.0),
            "GS": st.column_config.NumberColumn(format="%.1f"),
            "W": st.column_config.NumberColumn(format="%.1f"),
            "SV%": st.column_config.NumberColumn(format="%.4f"),
            "GAA": st.column_config.NumberColumn(format="%.2f")})
        st.caption("Open a single player on the Player dashboard to edit his ratings; "
                   "edit the whole team's totals in the last tab.")
        _roster_moves(team, sk, g)

    with tab_h:
        _history_tab(team, b, gbb)

    with tab_v:
        _review_tab(team, on, gs, b)

    with tab_s:
        _schedule_tab(team)

    with tab_e:
        _edit_form(team, b, gbb)


# --------------------------------------------------------------------------- #
# roster moves                                                                #
# --------------------------------------------------------------------------- #
def _move_label(row, team: str) -> str:
    """One line for the picker: who he is, where he is now, and what he is worth.

    The current situation is the whole point of the label. "Add Player X" is a decision a
    reader cannot make without knowing whether X is a free agent, somebody's camp body or
    another team's second-line centre -- the last of those takes a player OFF a rival, and
    that has to be visible before the click rather than after it.
    """
    where = row["team"] if row["on_roster"] else core.roster_label(row)
    got = row.get("proj_points")
    if got is None or pd.isna(got):
        got = row.get("proj_starts", np.nan)
        worth = "-" if pd.isna(got) else f"{got:.0f} GS"
    else:
        worth = f"{got:.0f} PTS"
    return f"{row['name']} · {row.get('position', 'G')} · {where} · {worth}"


def _move_pool(df: pd.DataFrame, team: str, kind: str) -> dict:
    """Label -> (kind, playerId) for everyone this team could add, likeliest first.

    Three tiers, because a list of 1,800 names sorted only by projected points opens on
    players who retired two years ago. What a reader is nearly always doing here is putting
    back a body the camp cut dropped from THIS team, so those come first; then everyone else
    without a team; then other teams' players, who are a deliberate raid and can be searched
    for by name. Within each tier, best first.
    """
    sort = "proj_points" if "proj_points" in df.columns else "proj_starts"
    cols = [c for c in ("playerId", "name", "position", "team", "on_roster", "camp",
                        "camp_team", sort) if c in df.columns]
    # Narrowed before the row walk: these frames carry several hundred columns and the
    # picker needs eight of them, and this runs on every redraw of the page.
    pool = df.loc[df["team"] != team, cols].copy()
    if pool.empty:
        return {}
    ours = pool.get("camp_team", pd.Series("", index=pool.index)).fillna("") == team
    pool["tier"] = np.where(ours, 0, np.where(pool["on_roster"], 2, 1))
    pool = pool.sort_values(["tier", sort], ascending=[True, False])
    return {_move_label(r, team): (kind, int(r["playerId"])) for _, r in pool.iterrows()}


def _roster_moves(team: str, sk: pd.DataFrame, g: pd.DataFrame) -> None:
    """Assign players to this team, or release them, without leaving the team page.

    Both directions are one scenario edit per player, committed together, because signing
    three forwards is one decision and redrawing the page between them would make a reader
    watch the budget re-settle twice for no reason. A team edit is all it takes to make the
    player selectable in this team's lines and counted in this team's budget -- the lineup
    page pools on team, and settlement pools on `on_roster & team`.
    """
    with st.expander("Move players onto or off this roster"):
        st.caption("Adding a player puts him on this team's budget and in its lineup pool. "
                   "Releasing him makes him a free agent — his projection survives, it is "
                   "just no longer part of any team's totals.")
        add_pool = {**_move_pool(sk, team, "s"), **_move_pool(g, team, "g")}
        # Skaters and goalies are listed together but edited through different buckets, so
        # the kind travels with the pick rather than being guessed back out of the row.
        drop_pool = {}
        for df, kind in ((sk, "s"), (g, "g")):
            here = df[(df["team"] == team) & df["on_roster"]]
            if here.empty:
                continue
            sort = "proj_points" if "proj_points" in here.columns else "proj_starts"
            for _, r in here.sort_values(sort, ascending=False).iterrows():
                drop_pool[_move_label(r, team)] = (kind, int(r["playerId"]))

        left, right = st.columns(2)
        with left:
            add = st.multiselect(f"Assign to {team}", list(add_pool),
                                 key=f"add_{team}",
                                 help="free agents, camp bodies and players currently on "
                                      "another team")
            if st.button("Assign", key=f"do_add_{team}", disabled=not add):
                sc = core.scenario()
                for label in add:
                    kind, pid = add_pool[label]
                    sc = (sc.set_player(pid, team=team, on_roster=True) if kind == "s"
                          else sc.set_goalie(pid, team=team, on_roster=True))
                core.commit(sc, f"{len(add)} added to {team}")
        with right:
            drop = st.multiselect(f"Release from {team}", list(drop_pool),
                                  key=f"drop_{team}",
                                  help="the projection is kept; the player just comes off "
                                       "this team's budget")
            if st.button("Release", key=f"do_drop_{team}", disabled=not drop):
                sc = core.scenario()
                for label in drop:
                    kind, pid = drop_pool[label]
                    sc = (sc.set_player(pid, on_roster=False, team=None) if kind == "s"
                          else sc.set_goalie(pid, on_roster=False, team=None))
                core.commit(sc, f"{len(drop)} released from {team}")

        moved = _moved_here(team, sk, g)
        if not moved.empty:
            st.markdown("**Moved by hand**")
            st.dataframe(moved, hide_index=True, width="stretch")


def _moved_here(team: str, sk: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """The roster moves in force, so a reader can see what he changed and undo it.

    Read from the scenario rather than from the frames: a player assigned to the team he was
    already on leaves no trace in the projection, and one released leaves the team entirely
    and would not be found by looking at this roster.
    """
    sc = core.scenario()
    rows = []
    for df, bucket in ((sk, sc.players), (g, sc.goalies)):
        if df.empty:
            continue
        cols = [c for c in ("name", "team", "on_roster", "camp", "camp_team", "target_age")
                if c in df.columns]
        by_id = df.set_index("playerId")[cols]
        for pid_s, edits in bucket.items():
            if "team" not in edits and "on_roster" not in edits:
                continue
            try:
                row = by_id.loc[int(pid_s)]
            except (KeyError, ValueError):
                continue
            was, now = edits.get("team"), row["team"]
            if now != team and was != team:
                continue
            rows.append({"Player": row["name"],
                         "Now": now if row["on_roster"] else core.roster_label(row),
                         "Listed": bool(row["on_roster"])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# team history                                                                #
# --------------------------------------------------------------------------- #
# (column, label, format). Two views of the same seasons, because the two answer different
# questions and neither is a substitute for the other. Per game is what can be compared
# down a column -- this history spans a 48-game lockout year, two shortened Covid seasons
# and an 84-game season ahead. Totals are what a reader actually remembers a team by, and
# what a season-total budget on the other tabs is denominated in.
#
# The record leads both. It is the sentence a team gets stated in, and a goal budget argued
# without one is being argued in a vacuum.
HIST_HEAD = [("season_label", "Season", None), ("as_named", "As", None),
             ("gp", "GP", "%.0f"), ("record", "Record", None), ("pts", "PTS", "%.0f"),
             ("pts_pct", "P%", "%.3f")]
HIST_COLS = HIST_HEAD + [
    ("gf_pg", "GF/GP", "%.2f"), ("ga_pg", "GA/GP", "%.2f"), ("gdiff_pg", "Diff", "%+.2f"),
    ("xgf_pg", "xGF/GP", "%.2f"), ("xga_pg", "xGA/GP", "%.2f"),
    ("xgdiff_pg", "xDiff", "%+.2f"), ("xg_share", "xG%", "%.3f"),
    ("corsi", "Corsi%", "%.3f"), ("sf_pg", "SF/GP", "%.1f"),
    ("sa_pg", "SA/GP", "%.1f"), ("shoot_pct", "Shoot%", "%.3f"),
    ("save_pct", "Save%", "%.3f"), ("pdo", "PDO", "%.3f"),
    ("hdf_pg", "HDC for", "%.1f"), ("hda_pg", "HDC vs", "%.1f"),
    ("hits_pg", "Hits/GP", "%.1f"), ("pim_pg", "PIM/GP", "%.1f"),
    ("fow_pct", "FO%", "%.3f"),
]
HIST_TOTAL_COLS = HIST_HEAD + [
    ("gf", "GF", "%.0f"), ("ga", "GA", "%.0f"), ("gdiff", "Diff", "%+.0f"),
    ("xgf", "xGF", "%.1f"), ("xga", "xGA", "%.1f"), ("xgdiff", "xDiff", "%+.1f"),
    ("xg_share", "xG%", "%.3f"), ("corsi", "Corsi%", "%.3f"),
    ("sf", "SF", "%.0f"), ("sa", "SA", "%.0f"), ("shoot_pct", "Shoot%", "%.3f"),
    ("save_pct", "Save%", "%.3f"), ("pdo", "PDO", "%.3f"),
    ("hdf", "HDC for", "%.0f"), ("hda", "HDC vs", "%.0f"),
    ("hits", "Hits", "%.0f"), ("pim", "PIM", "%.0f"),
    ("pp_pct", "PP%", "%.3f"), ("pk_pct", "PK%", "%.3f"), ("fow_pct", "FO%", "%.3f"),
]


def _history_tab(team: str, b: pd.Series, gbb: pd.Series | None) -> None:
    th = core.team_history()
    h = th[th["team"] == team].copy()
    if h.empty:
        st.info("No history for this franchise in the source file.")
        return
    h["season_label"] = (h["season"].astype(int).astype(str) + "-"
                         + (h["season"].astype(int) + 1).astype(str).str[-2:])

    # The last five seasons in one line, which is how a team gets described out loud.
    rec = h.dropna(subset=["record"]).head(5)
    if not rec.empty:
        m = st.columns(len(rec))
        for col, (_i, r) in zip(m, rec.iterrows()):
            col.metric(str(r["season_label"]), str(r["record"]),
                       delta=f"{int(r['pts'])} pts", delta_color="off")
        five = rec.head(5)
        st.caption(f"Last {len(five)} seasons: {int(five['w'].sum())}-"
                   f"{int(five['l'].sum())}-{int(five['otl'].sum())}, "
                   f"{five['pts_pct'].mean():.3f} points percentage. Records are wins-"
                   "losses-overtime losses; an overtime or shootout loss still banks a "
                   "point, which is why points and wins do not line up.")

    c1, c2 = st.columns([1, 2])
    n = c1.slider("Seasons", 3, int(len(h)), min(10, len(h)), 1, key=f"hn_{team}")
    how = c2.radio("How", ["Per game", "Season totals"], horizontal=True,
                   key=f"hhow_{team}")
    show = h.head(n)

    spec = HIST_COLS if how == "Per game" else HIST_TOTAL_COLS
    cols = [(c, lab, fmt) for c, lab, fmt in spec if c in show.columns]
    cfg = {c: (st.column_config.TextColumn(lab) if fmt is None
               else st.column_config.NumberColumn(lab, format=fmt))
           for c, lab, fmt in cols}
    st.dataframe(show[[c for c, _l, _f in cols]], hide_index=True, width="stretch",
                 column_config=cfg)
    renamed = sorted(set(show["as_named"]) - {team})
    st.caption(
        ("Season totals. Read down a column with the schedule length in mind: 2012-13 was "
         "48 games, 2019-20 and 2020-21 were shortened, and the season being projected is "
         "84. Switch to per game to compare them directly."
         if how == "Season totals" else
         "Per game, so seasons of different lengths can be compared directly — this "
         "history spans a 48-game lockout season, two shortened Covid seasons and an "
         "84-game season ahead. Switch to season totals for the numbers the budget tabs "
         "are denominated in.")
        + " HDC is high-danger chances. PDO is shooting plus save percentage: it barely "
        "persists year to year, so a team well above 1.000 outscored its own chances and "
        "is the one a projection should doubt most."
        + (f" Seasons played under {', '.join(renamed)} are this franchise's own."
           if renamed else ""))

    st.divider()
    st.markdown("**Is the budget plausible?**")
    st.caption(f"The {core.SEASON_LABEL} budget per game, next to what this team has "
               "actually done. The model builds a budget from a shrunk multi-season "
               "rating, so it should land inside this range and near the recent end of "
               "it — a long way outside is worth a look, and worth an edit if you "
               "disagree.")
    games = float(b["games"])
    # (label, per-game budget, history column, is it a per-game quantity at all)
    rows = [("Goals for", float(b["goals"]) / games, "gf_pg", True),
            ("Shots for", float(b["shots"]) / games, "sf_pg", True),
            ("Hits", float(b["hits"]) / games, "hits_pg", True),
            ("PIM", float(b["pim"]) / games, "pim_pg", True)]
    if gbb is not None:
        rows.insert(1, ("Goals against",
                        float(gbb["goals_against"]) / float(gbb["games"]), "ga_pg", True))
        rows.insert(2, ("Shots against",
                        float(gbb["shots_against"]) / float(gbb["games"]), "sa_pg", True))
        rows.append(("Save percentage", float(gbb["sv_pct"]), "save_pct", False))
    out = []
    for label, budgeted, col, scales in rows:
        if col not in h.columns:
            continue
        s = h[col].dropna()
        last3, last5 = s.head(3), s.head(5)
        out.append({"": label, "Budget": budgeted,
                    # The same budget as a season total, because that is the unit it is
                    # stated and edited in on the other tabs -- 3.21 goals a game is the
                    # readable number and 270 goals is the one being spent.
                    "Budget, season": budgeted * games if scales else np.nan,
                    "Last season": float(s.iloc[0]) if len(s) else np.nan,
                    "3-year": float(last3.mean()) if len(last3) else np.nan,
                    "5-year": float(last5.mean()) if len(last5) else np.nan,
                    "5-year low": float(last5.min()) if len(last5) else np.nan,
                    "5-year high": float(last5.max()) if len(last5) else np.nan})
    cmp_df = pd.DataFrame(out)
    st.dataframe(cmp_df, hide_index=True, width="stretch", column_config={
        **{c: st.column_config.NumberColumn(format="%.2f")
           for c in ("Budget", "Last season", "3-year", "5-year", "5-year low",
                     "5-year high")},
        "Budget, season": st.column_config.NumberColumn(
            f"Budget × {games:.0f} games", format="%.0f")})

    st.divider()
    opts = {"Goals for and against per game": ["gf_pg", "ga_pg"],
            "Standings points": ["pts"],
            "Wins, losses and overtime losses": ["w", "l", "otl"],
            "Expected goals for and against per game": ["xgf_pg", "xga_pg"],
            "Share of expected goals": ["xg_share"],
            "Shooting and save percentage": ["shoot_pct", "save_pct"],
            "PDO": ["pdo"],
            "Power play and penalty kill": ["pp_pct", "pk_pct"],
            "Shots for and against per game": ["sf_pg", "sa_pg"]}
    pick = st.selectbox("Trend", list(opts), key=f"htrend_{team}",
                        label_visibility="collapsed")
    series = [c for c in opts[pick] if c in show.columns]
    if series:
        st.line_chart(show.set_index("season_label")[series].iloc[::-1], height=260)


# --------------------------------------------------------------------------- #
# roster review                                                               #
# --------------------------------------------------------------------------- #
def _review_tab(team: str, on: pd.DataFrame, gs: pd.DataFrame, b: pd.Series) -> None:
    if on.empty:
        st.info("No listed skaters for this team.")
        return
    fwd = on[on["position"] != "D"]
    dmen = on[on["position"] == "D"]
    toi = on["proj_toi"].sum()

    m = st.columns(5)
    m[0].metric("Forwards / D", f"{len(fwd)} / {len(dmen)}")
    m[1].metric("Minutes to forwards",
                core.pct(fwd["proj_toi"].sum() / toi if toi else float("nan")),
                help="a balanced team gives its forwards about 61% of the ice time")
    wa = ((on["target_age"] * on["proj_toi"]).sum() / toi) if toi else float("nan")
    m[2].metric("Age, by ice time", core.num(wa, 1),
                help="average age weighted by projected minutes, which is the age that "
                     "actually plays")
    top6 = fwd.nlargest(6, "proj_toi_per_gp")["proj_toi"].sum()
    m[3].metric("Top 6 F share", core.pct(top6 / toi if toi else float("nan")))
    top4 = dmen.nlargest(4, "proj_toi_per_gp")["proj_toi"].sum()
    m[4].metric("Top 4 D share", core.pct(top4 / toi if toi else float("nan")))

    st.markdown("**Where the ice time goes**")
    dist = on.sort_values("proj_toi_per_gp", ascending=False)
    st.dataframe(pd.DataFrame({
        "Player": dist["name"], "Pos": dist["position"],
        "Age": dist["target_age"], "GP": dist["proj_gp"],
        "TOI/GP": dist["proj_toi_per_gp"], "PP/GP": dist["proj_pp_toi_per_gp"],
        "SH/GP": dist["proj_sh_toi_per_gp"],
        "Share of team minutes": dist["proj_toi"] / toi if toi else np.nan,
        "PTS/60": dist["per60_points"], "PTS": dist["proj_points"],
    }), hide_index=True, width="stretch", height=360, column_config={
        "Age": st.column_config.NumberColumn(format="%.0f"),
        "GP": st.column_config.NumberColumn(format="%.0f"),
        "TOI/GP": st.column_config.NumberColumn(format="%.1f"),
        "PP/GP": st.column_config.NumberColumn(format="%.2f"),
        "SH/GP": st.column_config.NumberColumn(format="%.2f"),
        "Share of team minutes": st.column_config.ProgressColumn(
            format="%.1f%%", min_value=0.0,
            max_value=float((dist["proj_toi"] / toi).max()) if toi else 1.0),
        "PTS/60": st.column_config.NumberColumn(format="%.2f"),
        "PTS": st.column_config.NumberColumn(format="%.1f")})

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**The power play**")
        pp = on.nlargest(8, "proj_pp_toi_per_gp")
        pp_total = on["proj_pp_toi"].sum()
        st.dataframe(pd.DataFrame({
            "Player": pp["name"], "Pos": pp["position"],
            "PP/GP": pp["proj_pp_toi_per_gp"],
            "Share": pp["proj_pp_toi"] / pp_total if pp_total else np.nan,
            "PPP": pp["proj_pp_points"], "PPP/60": pp["per60_pp_points"],
        }), hide_index=True, width="stretch", column_config={
            "PP/GP": st.column_config.NumberColumn(format="%.2f"),
            "Share": st.column_config.ProgressColumn(format="%.1f%%", min_value=0.0,
                                                     max_value=0.25),
            "PPP": st.column_config.NumberColumn(format="%.1f"),
            "PPP/60": st.column_config.NumberColumn(format="%.2f")})
        st.caption(f"{float(b['pp_toi_min']):,.0f} power-play minutes to give, "
                   f"{float(b['pp_coverage']):.0%} of them claimed by these players. A "
                   "first unit is about 3.5 to 4.5 minutes a night.")
    with c2:
        st.markdown("**Age profile**")
        buckets = pd.cut(on["target_age"],
                         [0, 22, 25, 28, 31, 34, 99],
                         labels=["21 and under", "22-24", "25-27", "28-30", "31-33",
                                 "34 and up"])
        prof = on.groupby(buckets, observed=False).agg(
            players=("name", "size"), minutes=("proj_toi", "sum"),
            points=("proj_points", "sum")).reset_index()
        prof["share"] = prof["minutes"] / toi if toi else np.nan
        st.dataframe(prof.rename(columns={"target_age": "Age"}), hide_index=True,
                     width="stretch", column_config={
                         "players": st.column_config.NumberColumn("Players", format="%.0f"),
                         "minutes": st.column_config.NumberColumn("Minutes", format="%.0f"),
                         "points": st.column_config.NumberColumn("Points", format="%.0f"),
                         "share": st.column_config.ProgressColumn(
                             "Share of minutes", format="%.1f%%", min_value=0.0,
                             max_value=1.0)})
        st.caption("Ages are as of February of the projected season, which is how the age "
                   "curves are indexed.")

    st.markdown("**Who the budget takes from**")
    st.caption("Claim minus projection, biggest first. A large cut means this player's own "
               "rates ask for more than his team has left to give — he is the one to check "
               "if you think the team is stronger than the budget says.")
    gap = on.assign(cut=on["unc_points"] - on["proj_points"]).nlargest(8, "cut")
    st.dataframe(pd.DataFrame({
        "Player": gap["name"], "Claimed PTS": gap["unc_points"],
        "Projected PTS": gap["proj_points"], "Cut": -gap["cut"],
        "Claimed TOI/GP": gap["claim_toi_per_gp"],
        "Projected TOI/GP": gap["proj_toi_per_gp"],
    }), hide_index=True, width="stretch", column_config={
        "Claimed PTS": st.column_config.NumberColumn(format="%.1f"),
        "Projected PTS": st.column_config.NumberColumn(format="%.1f"),
        "Cut": st.column_config.NumberColumn(format="%+.1f"),
        "Claimed TOI/GP": st.column_config.NumberColumn(format="%.1f"),
        "Projected TOI/GP": st.column_config.NumberColumn(format="%.1f")})

    if len(gs) < 2:
        st.warning(f"Only {len(gs)} goalie listed. A real team-season uses about three, "
                   "so most of this crease is being held back rather than projected — "
                   "sign a goalie on the Player dashboard if you know who it is.")


# --------------------------------------------------------------------------- #
# schedule                                                                    #
# --------------------------------------------------------------------------- #
def _schedule_tab(team: str) -> None:
    sched = core.schedule()
    s = sched[sched["team"] == team].copy()
    if s.empty:
        st.info("No schedule rows for this team.")
        return
    ratings = core.team_ratings()
    s["gdate"] = pd.to_datetime(s["gameDate"])
    s = s.sort_values("gdate")
    b2b = int((s["b2b_mult"] < 1.0).sum())
    sos = float(s["sos_factor"].iloc[0])
    rank = int((sched.groupby("team")["sos_factor"].first() > sos).sum()) + 1

    m = st.columns(5)
    m[0].metric("Games", len(s))
    m[1].metric("Home", int(s["is_home"].sum()))
    m[2].metric("Back to backs", b2b,
                help="second halves of back-to-backs, where a starter is least likely "
                     "to play")
    m[3].metric("Schedule factor", f"{sos:.3f}",
                help="above 1.000 means an easier schedule for scoring; it is applied to "
                     "the team's budget and re-centred so the league total is unchanged")
    m[4].metric("Rank", f"{rank} of 32", help="1 is the easiest schedule to score on")

    opp = s.groupby("opponent").size().rename("Games").reset_index()
    opp["Their defence"] = opp["opponent"].map(
        ratings["def_rating"] if "def_rating" in ratings else {})
    opp["Their offence"] = opp["opponent"].map(
        ratings["off_rating"] if "off_rating" in ratings else {})
    opp = opp.sort_values("Their defence", ascending=False)
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown("**Who they play**")
        st.dataframe(opp, hide_index=True, width="stretch", height=320, column_config={
            "opponent": st.column_config.TextColumn("Opponent"),
            "Games": st.column_config.NumberColumn(format="%.0f"),
            "Their defence": st.column_config.NumberColumn(
                format="%.3f", help="above 1.000 allows fewer goals than average"),
            "Their offence": st.column_config.NumberColumn(format="%.3f")})
    with c2:
        st.markdown("**By month**")
        by_month = (s.assign(month=s["gdate"].dt.strftime("%b %Y"))
                     .groupby("month", sort=False)
                     .agg(Games=("gameId", "size"),
                          Home=("is_home", "sum"),
                          Back_to_backs=("b2b_mult", lambda v: int((v < 1.0).sum())))
                     .reset_index())
        st.dataframe(by_month, hide_index=True, width="stretch", column_config={
            "month": st.column_config.TextColumn("Month"),
            "Games": st.column_config.NumberColumn(format="%.0f"),
            "Home": st.column_config.NumberColumn(format="%.0f"),
            "Back_to_backs": st.column_config.NumberColumn("B2B", format="%.0f")})
    st.caption("Opponent strength moves a season total only gently — it is one game in 84 "
               "against any one team, and the effect is zero-sum across the league, which "
               "is why it sits on the team budget rather than on each player.")


def _edit_form(team: str, b: pd.Series, gbb: pd.Series | None) -> None:
    edits = core.scenario().team(team)
    st.caption("A stated team total replaces the budget and every player on the roster "
               "resettles around it. Stating goals also moves assists and points, because "
               "the league's assists-per-goal ratio is not negotiable. This does not take "
               "the difference back off the other 31 teams — if you say a team scores 300, "
               "that is what you said.")

    # Either unit, because a reader holds these two beliefs in different units. Goals are
    # thought about per game -- "this is a 3.4-goals-a-game team" -- and hits and PIM are
    # not thought about per game by anybody. What is STORED is always the season total,
    # since that is what the budget is and what settlement spends; per game is a way of
    # typing it, converted at this team's own game count so the two are exactly equivalent.
    games = float(b["games"])
    unit = st.radio("Enter as", ["Season totals", "Per game"], horizontal=True,
                    key=f"tunit{team}",
                    help=f"{team} plays {games:.0f} games, which is the number the two "
                         f"units convert through.")
    per_game = unit == "Per game"

    def box(container, col: str, label: str, cap: float, now: float):
        """One budget input, in whichever unit is selected."""
        cur = edits.get(col)
        div = games if per_game else 1.0
        step = 0.05 if per_game else 5.0
        fmt = "%.2f" if per_game else "%.0f"
        return container.number_input(
            f"{label} / game" if per_game else label, min_value=0.0, max_value=cap / div,
            value=float(cur) / div if cur is not None else None, step=step, format=fmt,
            placeholder=f"model says {now / div:,.2f}" if per_game
            else f"model says {now:,.0f}", key=f"tb{team}{col}{int(per_game)}")

    values: dict[str, float | None] = {}
    cols = st.columns(3)
    for i, (col, label, cap) in enumerate(EDITABLE):
        now = float(b[col]) if col in b.index else 0.0
        values[col] = box(cols[i % 3], col, label, cap, now)
    if gbb is not None:
        cols = st.columns(3)
        for i, (col, label, cap) in enumerate(EDITABLE_G):
            values[col] = box(cols[i % 3], col, label, cap, float(gbb[col[7:]]))

    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("Save budget", type="primary", key=f"tsave{team}"):
        patch: dict = {}
        for col, v in values.items():
            old = edits.get(col)
            # Back to a season total before anything is compared or stored, so switching
            # units cannot register as an edit and a per-game entry means the same thing a
            # total does.
            want = None if v is None else float(v) * (games if per_game else 1.0)
            if want is None and old is not None:
                patch[col] = None
            elif want is not None and (old is None or abs(float(old) - want) >= 0.01):
                patch[col] = want
        if patch:
            core.edit_team(team, **patch)
        else:
            st.info("Nothing to save.")
    if edits and b2.button("Back to the model", key=f"tclr{team}"):
        core.commit(core.scenario().clear_team(team), f"{team} budget back to the model")
    if edits:
        st.caption("Currently overriding: "
                   + ", ".join(f"{k} = {v:g} ({float(v) / games:.2f} a game)"
                               for k, v in sorted(edits.items())))


# --------------------------------------------------------------------------- #
def page() -> None:
    core.scenario_bar()
    sk, tb = core.skaters()
    g, gb = core.goalies()
    core.header(f"Teams · {core.SEASON_LABEL}",
                "Every player projection was settled against these budgets. This is where "
                "a number that looks low usually explains itself.")

    teams = ["League overview"] + list(tb.index)
    edited = set(core.scenario().teams)
    pick = st.selectbox("Team", teams, key="team_pick",
                        format_func=lambda t: t + (" ✎" if t in edited else ""))
    if pick == "League overview":
        _overview(sk, tb, g, gb)
    else:
        _team_page(pick, sk, tb, g, gb)

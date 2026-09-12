"""The scenario: every disagreement in one list, each one removable.

The workbook's overrides lived in a CSV of games played and nowhere else, so there was no
way to answer "what have I changed?" This page is that answer. Everything here is
reversible, because the baseline was never overwritten -- clearing an edit restores the
model's own number exactly.
"""
from __future__ import annotations

import json

import pandas as pd
import requests
import streamlit as st

import core
import library

KNOBS = [
    ("enforce_budgets", "Enforce team budgets", "bool",
     "Off means every player gets his unconstrained claim and team totals stop adding up. "
     "Useful for seeing the raw rates; not a projection."),
    ("budget_tilt", "Budget tilt", (0.0, 1.0, 0.05),
     "How overshoot is taken back. 0 takes it evenly, 1 takes it in proportion to the "
     "claim; below 1 protects the stars, who are the least likely to be wrong."),
    ("goals_xg_weight", "Expected-goals weight in goals", (0.0, 1.0, 0.05),
     "How much a goal projection leans on shot quality rather than the player's own "
     "finishing. Shooting percentage regresses much harder than shot quality does."),
    ("team_rating_shrink", "Team rating shrink", (0.0, 1.0, 0.05),
     "How far a team's own history is pulled toward the league before it becomes a budget."),
    ("skater_regress_toi_min", "Skater regression (minutes)", (0.0, 2000.0, 50.0),
     "Minutes of ice time a skater needs before his own rates outweigh the position prior."),
    ("goalie_regress_shots", "Goalie regression (shots)", (0.0, 3000.0, 50.0),
     "Shots a goalie needs before his own save percentage outweighs the league's."),
]


def _edit_list() -> None:
    sc = core.scenario()
    sk, _ = core.skaters()
    g, _ = core.goalies()
    names = dict(zip(sk["playerId"].astype(str), sk["name"]))
    names.update(dict(zip(g["playerId"].astype(str), g["name"])))
    kind = dict.fromkeys(sk["playerId"].astype(str), "Skater")
    kind.update(dict.fromkeys(g["playerId"].astype(str), "Goalie"))

    rows = []
    for bucket, label in [("players", "Skater"), ("goalies", "Goalie"),
                          ("teams", "Team")]:
        for key, fields in getattr(sc, bucket).items():
            for field, value in sorted(fields.items()):
                rows.append({"What": kind.get(key, label) if bucket != "teams" else "Team",
                             "Who": names.get(key, key), "Field": field,
                             "Value": value, "_bucket": bucket, "_key": key})
    for key, value in sorted(sc.league.items()):
        rows.append({"What": "League", "Who": "all teams", "Field": key, "Value": value,
                     "_bucket": "league", "_key": key})
    # A lineup is one edit, not twelve: undoing it means dropping the whole card, because
    # half a lineup is not a statement about anybody's linemates.
    for team, units in sorted(sc.lines.items()):
        named = sum(len(v) for v in units.values())
        rows.append({"What": "Lines", "Who": team, "Field": "lineup",
                     "Value": f"{len(units)} units, {named} players",
                     "_bucket": "lines", "_key": team})

    if not rows:
        st.success("No edits. Every number in the app is the model's own opinion.")
        return
    df = pd.DataFrame(rows)
    # One column holding 80.0, True and "1 units, 3 players" has no Arrow type, so Streamlit
    # was serialising it, failing, logging a traceback and silently retrying as text. Make it
    # text here instead, and trim the float noise on the way.
    df["Value"] = [f"{v:g}" if isinstance(v, float) else str(v) for v in df["Value"]]
    st.dataframe(df.drop(columns=["_bucket", "_key"]), hide_index=True, width="stretch",
                 height=min(420, 60 + 35 * len(df)))

    st.markdown("**Remove edits**")
    c1, c2 = st.columns([3, 1])
    labels = [f"{r['Who']} · {r['Field']} = {r['Value']}" for r in rows]
    picked = c1.multiselect("Pick the edits to undo", range(len(rows)),
                            format_func=lambda i: labels[i], label_visibility="collapsed")
    if c2.button("Undo selected", disabled=not picked, width="stretch"):
        sc2 = sc
        for i in picked:
            r = rows[i]
            if r["_bucket"] == "players":
                sc2 = sc2.set_player(r["_key"], **{r["Field"]: None})
            elif r["_bucket"] == "goalies":
                sc2 = sc2.set_goalie(r["_key"], **{r["Field"]: None})
            elif r["_bucket"] == "teams":
                sc2 = sc2.set_team(r["_key"], **{r["Field"]: None})
            elif r["_bucket"] == "lines":
                sc2 = sc2.clear_lines(r["_key"])
            else:
                sc2 = sc2.patch_league(**{r["Field"]: None})
        core.commit(sc2, f"Undid {len(picked)} edit{'s' if len(picked) != 1 else ''}")

    st.caption("Whole players can also be cleared from their own page, and 'Clear all "
               "edits' in the sidebar puts everything back to the model.")


def _knobs() -> None:
    sc = core.scenario()
    st.caption("League-wide settings. These change the shape of every projection at once, "
               "so a change here is worth more thought than a change to one player. Each "
               "one is stored only if it differs from the default, so putting it back "
               "leaves no trace.")
    values = {}
    for key, label, spec, help_text in KNOBS:
        cur = sc.knob(key)
        if spec == "bool":
            values[key] = st.toggle(label, value=bool(cur), help=help_text, key=f"kn{key}")
        else:
            lo, hi, step = spec
            values[key] = st.slider(label, lo, hi, float(cur), step, help=help_text,
                                    key=f"kn{key}")
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Apply settings", type="primary"):
        try:
            sc2 = sc.patch_league(**values)
        except ValueError as exc:
            st.error(str(exc))
            return
        if sc2.league == sc.league:
            st.info("Nothing changed.")
        else:
            core.commit(sc2, "League settings applied")
    if sc.league and c2.button("Defaults"):
        core.commit(sc.patch_league(**{k: None for k in sc.league}),
                    "League settings back to default")


def _files() -> None:
    sc = core.scenario()
    if core.MULTIUSER:
        st.caption("Your edits live in this browser session only, so two people can argue "
                   "with the model at the same time without overwriting each other. "
                   "Publishing one puts it in the shared library below, where anyone can "
                   "open it and see exactly what you changed.")
        if not library.durable():
            # An error, not a caption. Publishing into a disk that is about to be wiped
            # looks identical to publishing properly until the day you come back for it,
            # so the page has to be unpleasant about it while it is still fixable.
            st.error(
                "**Nothing published here will survive a restart.** No durable scenario "
                "library is configured, so a published scenario goes to "
                f"{library.where()}. Download the JSON below to keep this one, and set the "
                "library up so it stops happening.", icon="🚨")
            with st.expander("How to make saved scenarios permanent (one-time setup)"):
                st.markdown(
                    "1. On GitHub: **+ → New gist**, one file named `readme.txt` with any "
                    "text, then **Create secret gist**. The id is the last part of its "
                    "URL.\n"
                    "2. **Settings → Developer settings → Personal access tokens → "
                    "Fine-grained tokens → Generate new token**, and under *Account "
                    "permissions* give it **Gists: read and write**. Nothing else.\n"
                    "3. In Streamlit Cloud: your app → **⋮ → Settings → Secrets**, paste "
                    "the two lines below with your own values, and save. The app reboots "
                    "and the library is durable from then on — through restarts, "
                    "redeploys and every weekly data refresh.")
                st.code('gist_id = "…"\ngithub_token = "…"', language="toml")
                st.caption("The gist is private, the token stays in Streamlit's secret "
                           "store, and neither ever enters this public repository.")
    else:
        st.caption("A scenario is a small JSON file of only the disagreements, so it can "
                   "be read, diffed and mailed to someone. The working scenario is saved "
                   "after every edit — nothing here is needed to keep your work.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Publish this scenario**" if core.MULTIUSER else "**Save a copy**")
        name = st.text_input("Name", placeholder="opening-night", key="save_name")
        who = st.text_input("Your name", placeholder="who to credit it to", key="save_who")
        freeze = st.checkbox(
            "Keep today's numbers with it", value=True, key="save_freeze",
            help="Stores the projection exactly as it stands now, player by player. Without "
                 "this the save is only the list of edits, and opening it in February "
                 "reprojects them on February's data — so a preseason number could not be "
                 "read back as it was.")
        if st.button("Publish" if core.MULTIUSER else "Save as",
                     disabled=not name.strip() or sc.is_baseline, width="stretch"):
            try:
                snap = core.frozen_snapshot(sc) if freeze else None
                res = library.save(sc, name, who, frozen_numbers=snap)
            except (ValueError, requests.RequestException) as exc:
                st.error(f"Could not save it: {exc}")
            else:
                # Three outcomes, and they are told apart on purpose. The library is read
                # back after every write, so "saved" here means the scenario was found
                # again and still matched, not merely that nothing raised.
                if not res["verified"]:
                    st.error(f"**{res['name']} did not read back from the library.** "
                             "Treat it as unsaved, download the JSON, and try again.",
                             icon="🚨")
                elif res["durable"]:
                    st.success(f"Saved as **{res['name']}** in {res['where']}.")
                else:
                    st.warning(f"Saved as **{res['name']}**, but only in {res['where']}. "
                               "Download the JSON if you need it tomorrow.", icon="⚠️")
                if res["verified"] and freeze:
                    if res["frozen"]:
                        st.caption(f"Numbers frozen too — {res['frozen_bytes'] / 1024:.0f} "
                                   "KB of projections that will not move again. Open them "
                                   "under *Review one exactly as it was saved*.")
                    else:
                        st.error("The scenario saved but its numbers did not. Reading it "
                                 "back later will reproject the edits on that day's data.",
                                 icon="🚨")
        if sc.is_baseline:
            st.caption("Nothing to save yet — this is the model's own projection.")
        st.download_button("Download this scenario (JSON)", sc.to_json().encode("utf-8"),
                           file_name="scenario.json", mime="application/json",
                           width="stretch")
    with c2:
        st.markdown("**Open a saved one**")
        st.caption(f"Stored in {library.where()}. Two things live under each name: the EDITS, "
                   "which reproject on today's data when you open them, and — if it was "
                   "saved with its numbers — a frozen copy of the projection as it stood "
                   "that day, which never moves again.")
        rows = library.entries()
        if rows:
            st.dataframe(pd.DataFrame(rows)[["name", "author", "saved", "edits", "frozen"]],
                         hide_index=True, width="stretch",
                         height=min(220, 60 + 35 * len(rows)),
                         column_config={
                             "name": "Scenario", "author": "By", "saved": "Saved (UTC)",
                             "edits": "Edits",
                             "frozen": st.column_config.CheckboxColumn(
                                 "Numbers kept",
                                 help="the projection was frozen as it stood when saved")})
        else:
            st.caption("The library is empty. Publish one and it shows up here for "
                       "everyone.")
        pick = st.selectbox("Scenario", ["(none)"] + [r["name"] for r in rows],
                            key="load_pick", label_visibility="collapsed")
        b1, b2 = st.columns([2, 1])
        if b1.button("Open it", disabled=pick == "(none)", width="stretch"):
            try:
                loaded = library.load(pick)
            except (FileNotFoundError, requests.RequestException) as exc:
                st.error(f"Could not open it: {exc}")
            else:
                core.commit(loaded, f"Opened {pick}")
        if b2.button("Delete", disabled=pick == "(none)", width="stretch",
                     help="Removes it from the shared library for everyone."):
            library.delete(pick)
            st.rerun()

        up = st.file_uploader("Or upload a scenario file", type="json")
        if up is not None and st.button("Load the uploaded file", width="stretch"):
            import overrides as ov
            try:
                loaded = ov.Scenario.from_json(up.getvalue().decode("utf-8"))
            except (ValueError, json.JSONDecodeError) as exc:
                st.error(f"That is not a scenario file: {exc}")
                return
            loaded.name = "working"
            core.commit(loaded, "Loaded the uploaded scenario")

    st.divider()
    _frozen_review(pick)


# --------------------------------------------------------------------------- #
# reading a saved projection back exactly as it was                           #
# --------------------------------------------------------------------------- #
FROZEN_LABELS = {
    "name": "Player", "team": "Team", "position": "Pos", "proj_gp": "GP",
    "proj_toi_per_gp": "TOI/GP", "proj_pp_toi_per_gp": "PP/GP", "proj_goals": "G",
    "proj_assists": "A", "proj_points": "PTS", "points_p10": "PTS floor",
    "points_p90": "PTS ceiling", "gp_p10": "GP floor", "gp_p90": "GP ceiling",
    "proj_shots": "SOG", "proj_ixg": "ixG", "proj_pp_points": "PPP",
    "proj_sh_points": "SHP", "proj_blocks": "BLK", "proj_hits": "HIT", "proj_pim": "PIM",
    "proj_faceoffs_won": "FOW", "per60_points": "PTS/60", "edited": "Edited",
    "proj_starts": "GS", "starts_p10": "GS floor", "starts_p90": "GS ceiling",
    "proj_wins": "W", "wins_p10": "W floor", "wins_p90": "W ceiling", "proj_losses": "L",
    "proj_otl": "OTL", "proj_save_pct": "SV%", "proj_gaa": "GAA", "proj_shutouts": "SO",
    "shutouts_p10": "SO floor", "shutouts_p90": "SO ceiling", "proj_saves": "SV",
    "proj_shots_against": "SA", "games": "Games", "goals_budget": "Goals budget",
    "points_budget": "Points budget", "toi_coverage": "Roster covers",
    "goals_projected": "Goals projected", "points_projected": "Points projected",
    "wins_projected": "Wins projected", "team_sv_pct": "Team SV%",
}
# What "then versus now" is worth comparing on. Everything else is in the table anyway.
COMPARE = {"skaters": ("proj_points", "PTS"), "goalies": ("proj_wins", "W"),
           "teams": ("points_projected", "Points projected")}


def _frozen_review(pick: str) -> None:
    st.markdown("**Review one exactly as it was saved**")
    if pick in (None, "(none)"):
        st.caption("Pick a scenario above. If it was saved with its numbers, the projection "
                   "it produced that day can be read back here — unchanged, however much "
                   "data has arrived since.")
        return

    snap = library.frozen(pick)
    if not snap:
        st.info(f"**{pick}** was saved without its numbers, so there is nothing frozen to "
                "read. Opening it reprojects its edits on today's data.")
        return

    head = [f"frozen {(snap.get('frozen_at') or snap.get('saved_at') or '')[:16].replace('T', ' ')} UTC",
            f"by {snap.get('author') or '-'}", snap.get("season", ""),
            snap.get("window", "")]
    st.caption(" · ".join(b for b in head if b))

    kind = st.radio("What", ["skaters", "goalies", "teams"], horizontal=True,
                    format_func=str.title, key="frz_kind", label_visibility="collapsed")
    frz = library.frozen_table(snap, kind)
    if frz.empty:
        st.info("Nothing of that kind was frozen.")
        return

    key_col = "playerId" if kind != "teams" else "team"
    now_col, now_label = COMPARE[kind]
    compare = st.checkbox(
        "Compare with what this scenario says today", key="frz_cmp",
        help="Reprojects the same edits on today's data and shows the difference. The "
             "frozen column never changes; only the comparison does.")

    show = frz.drop(columns=["playerId"], errors="ignore")
    if compare:
        try:
            sc_then = library.load(pick)
        except (FileNotFoundError, requests.RequestException) as exc:
            st.error(f"Could not reload the scenario to compare: {exc}")
            sc_then = None
        if sc_then is not None:
            now = _now_frame(sc_then, kind, now_col)
            merged = frz.merge(now, on=key_col, how="left")
            merged["Δ"] = merged["_now"] - merged[now_col]
            show = merged.drop(columns=["playerId"], errors="ignore").rename(
                columns={"_now": f"{now_label} now"})

    cfg = {}
    for c in show.columns:
        label = FROZEN_LABELS.get(c, c)
        # Bools first: pandas calls a bool column numeric, and "1.0" is not what "edited"
        # means to a reader.
        if pd.api.types.is_bool_dtype(show[c]):
            cfg[c] = st.column_config.CheckboxColumn(label)
        elif pd.api.types.is_numeric_dtype(show[c]):
            fmt = ("%.3f" if c in ("proj_save_pct", "toi_coverage", "team_sv_pct")
                   else "%.2f" if c in ("proj_gaa", "per60_points") or c.endswith("_per_gp")
                   else "%.1f")
            cfg[c] = st.column_config.NumberColumn(label, format=fmt)
        else:
            cfg[c] = st.column_config.TextColumn(label)
    if "Δ" in show.columns:
        cfg["Δ"] = st.column_config.NumberColumn(f"{now_label} change", format="%+.1f")
    st.dataframe(show, hide_index=True, width="stretch", height=460, column_config=cfg)

    c1, c2 = st.columns([1.4, 3])
    c1.download_button(f"Download the frozen {kind} (CSV)",
                       show.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{pick}_{kind}_as_saved.csv", mime="text/csv",
                       width="stretch")
    c2.caption(f"{len(frz)} rows, exactly as projected on "
               f"{(snap.get('frozen_at') or '')[:10]}. Deleting the scenario is the only "
               "thing that removes them.")


def _now_frame(sc_then, kind: str, col: str) -> pd.DataFrame:
    """One column of today's projection of the same edits, keyed for the merge."""
    if kind == "skaters":
        df, _ = core.skaters(sc_then)
        return df[["playerId", col]].rename(columns={col: "_now"})
    if kind == "goalies":
        df, _ = core.goalies(sc_then)
        return df[["playerId", col]].rename(columns={col: "_now"})
    sk, _ = core.skaters(sc_then)
    on = sk[sk["on_roster"]]
    tot = on.groupby("team")["proj_points"].sum().rename("_now").reset_index()
    return tot


def page() -> None:
    core.scenario_bar()
    sc = core.scenario()
    n = sc.count()
    core.header("Scenario",
                "Baseline model — no edits yet." if sc.is_baseline else
                f"{core.edit_badge(n['edits'])} across {n['players']} skaters, "
                f"{n['goalies']} goalies, {n['teams']} teams"
                + (f", {n['lines']} lineups" if n["lines"] else "")
                + (f" and {n['league']} league settings" if n["league"] else "") + ".")

    tab_list, tab_knobs, tab_files = st.tabs(["Edits", "League settings", "Save & share"])
    with tab_list:
        _edit_list()
    with tab_knobs:
        _knobs()
    with tab_files:
        _files()

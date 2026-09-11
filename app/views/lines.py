"""Lines: who plays with whom, and what changing it is worth.

The model reads five years of a player's own history, and that history already contains his
old linemates -- so the model cannot tell that a winger is about to start the season beside
a different centre. A person watching training camp can. This page is where that gets said.

What it does NOT do is guess. A team with no saved lineup gets no line adjustment at all,
and the effect is driven by WHO a player skates with, not by the number on the line: moving
a trio from the third line to the first, with the same three players, changes nothing here
because it is a change in ice time, which is edited on the player's own page.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import config as C
import core
import lines as ln

BLANK = "—"


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _labels(pool: pd.DataFrame) -> tuple[dict[int, str], dict[str, int]]:
    """playerId -> display label, and back. Deepest ice time first.

    The position is part of the label because these labels are what a dropdown of the whole
    roster shows, and "which of these is a centre?" is the first question anyone picking a
    line has. Two players on one roster can share a name, so a duplicate also gets his id.

    A label has to be STABLE, because the label *is* the value held in the slot widget: if it
    changed under a widget, Streamlit would find the stored value missing from the options
    and quietly empty the slot. So the "camp" marker comes from the `camp` column, which the
    model decides before any edit is applied and which therefore does not move when saving a
    lineup promotes a camp player onto the roster.
    """
    r = pool.sort_values("proj_toi_per_gp", ascending=False)
    counts = r["name"].value_counts()
    fwd, back = {}, {}
    for pid, name, pos, camp in zip(r["playerId"], r["name"], r["position"], r["in_camp"]):
        label = (f"{name} · {pos}"
                 + (f" · {int(pid)}" if counts.get(name, 0) > 1 else "")
                 + (" · camp" if camp else ""))
        fwd[int(pid)] = label
        back[label] = int(pid)
    return fwd, back


def _options(pool: pd.DataFrame, to_label: dict[int, str], group: str) -> list[str]:
    """The dropdown list for one kind of slot: roster first, then camp, each by ice time.

    Everyone stays selectable. Position in this data is a listed position, not a promise --
    a defenceman who spends the season on a wing exists, and a dropdown that hides him
    cannot describe the lineup the user is actually looking at. The team's camp players are
    in the list for the same reason: a first-overall pick who is going to play all 82 games
    is not on the model's projected roster, and a dropdown that leaves him out cannot
    describe next season's top line.
    """
    r = pool.copy()
    r["_off"] = (r["pos_group"] != group).astype(int)
    r = r.sort_values(["_off", "in_camp", "proj_toi_per_gp"],
                      ascending=[True, True, False])
    return [BLANK] + [to_label[int(p)] for p in r["playerId"]]


def _stat_maps(pool: pd.DataFrame) -> tuple[dict[int, str], dict[int, float]]:
    """playerId -> the line of numbers printed under his slot, and his points on their own.

    What a card cannot show without this is whether it makes any sense: a fourth-liner on
    the top line at seven minutes a game is a mistake you want to see while you are making
    it, not after re-projecting.
    """
    text, pts = {}, {}
    for pid, toi, p, g, a in zip(pool["playerId"], pool["proj_toi_per_gp"],
                                 pool["proj_points"], pool["proj_goals"],
                                 pool["proj_assists"]):
        text[int(pid)] = f"{toi:.1f} min · {p:.1f} P · {g:.1f} G · {a:.1f} A"
        pts[int(pid)] = float(p)
    return text, pts


# --------------------------------------------------------------------------- #
# the lineup card: one dropdown per slot, kept consistent by swapping          #
# --------------------------------------------------------------------------- #
# Every slot on the card is its own selectbox, and the dropdown is the whole roster. The
# alternative -- a spreadsheet grid -- makes you click a cell before it will even show you
# the choices, which is a lot of clicks to move one winger.
#
# The rule that makes it feel like a lineup card rather than a form: putting a player in a
# slot he is not already in TRADES him with whoever was there. He cannot end up on two
# lines, nothing has to be cleared first, and there is no error to read -- a player is
# always in exactly one place, which is true of real lineups too.
def _slot_keys(team: str) -> list[tuple[str, int, str]]:
    """(unit label, index within the unit, session_state key) for every slot on the card."""
    return [(label, i, f"lnw_{team}_{label}_{i}")
            for label in (*ln.FORWARD_LABELS, *ln.PAIR_LABELS)
            for i in range(ln.UNIT_SIZE[label])]


def _fingerprint(card: dict[str, list[int]]) -> str:
    return repr(sorted((k, tuple(int(p) for p in v)) for k, v in card.items()))


def _seed_card(team: str, card: dict[str, list[int]], to_label: dict[int, str],
               saved: dict, force: bool = False) -> None:
    """Fill the slot widgets from `card`, but only when the SAVED lineup has moved.

    The widgets are the working copy, so re-seeding on every rerun would undo the edit the
    user just made. Re-seeding never is worse: opening a scenario from the library, or
    undoing a lineup on the Scenario page, would leave this page showing a card that is no
    longer in the projection. Keying on a fingerprint of the saved lineup covers both.

    The fingerprint is not enough on its own: Streamlit throws away the state of any widget
    it did not draw on the last run, so looking at Boston, switching to Toronto and switching
    back leaves Boston's eighteen slots with no values at all. If any of them has gone
    missing, the card is re-seeded regardless of the fingerprint.

    This has to run BEFORE the slot widgets are drawn -- Streamlit refuses to let a widget's
    state be set once the widget exists, which is why "Reset the card" flags and reruns
    rather than writing the values itself.
    """
    fp = _fingerprint(saved or {})
    lost = any(k not in st.session_state for _, _, k in _slot_keys(team))
    if st.session_state.get(f"lnfp_{team}") == fp and not force and not lost:
        return
    for label, i, key in _slot_keys(team):
        ids = list(card.get(label, []))
        pid = int(ids[i]) if i < len(ids) and ids[i] is not None else None
        st.session_state[key] = to_label.get(pid, BLANK) if pid is not None else BLANK
    st.session_state[f"lnfp_{team}"] = fp
    st.session_state[f"lnprev_{team}"] = {k: st.session_state[k]
                                          for _, _, k in _slot_keys(team)}


def _on_pick(team: str, key: str) -> None:
    """A slot changed: if that player was somewhere else on the card, trade places."""
    prev = st.session_state.setdefault(f"lnprev_{team}", {})
    new = st.session_state.get(key, BLANK)
    old = prev.get(key, BLANK)
    if new != BLANK:
        for _, _, other in _slot_keys(team):
            if other != key and st.session_state.get(other) == new:
                st.session_state[other] = old
                prev[other] = old
                break
    prev[key] = new


def _card_from_state(team: str, to_id: dict[str, int]) -> dict[str, list[int]]:
    """Read the card back off the widgets. The swap rule keeps it duplicate-free."""
    card: dict[str, list[int]] = {}
    seen: set[int] = set()
    for label, _, key in _slot_keys(team):
        pid = to_id.get(str(st.session_state.get(key, BLANK)))
        if pid is not None and pid not in seen:
            card.setdefault(label, []).append(pid)
            seen.add(pid)
    return card


def _unit_row(team: str, label: str, options: list[str], q: pd.Series,
              to_id: dict[str, int], headings: list[str],
              stats: dict[int, str], pts: dict[int, float]) -> None:
    """One line or pairing: its name, a dropdown per slot with that player's projection
    under it, and what the unit is worth."""
    size = ln.UNIT_SIZE[label]
    cols = st.columns([0.5] + [2] * size + [1.15], vertical_alignment="top")
    cols[0].markdown(f"<div style='padding-top:.45rem;font-weight:600'>{label}</div>",
                     unsafe_allow_html=True)
    ids = []
    for i in range(size):
        key = f"lnw_{team}_{label}_{i}"
        cols[i + 1].selectbox(
            headings[i], options, key=key, label_visibility="collapsed",
            on_change=_on_pick, args=(team, key))
        pid = to_id.get(str(st.session_state.get(key, BLANK)))
        cols[i + 1].markdown(
            "<div style='opacity:.62;font-size:.74rem;margin:-.55rem 0 .3rem .12rem'>"
            f"{stats.get(pid) if pid is not None else '&nbsp;'}</div>",
            unsafe_allow_html=True)
        if pid is not None:
            ids.append(pid)
    known = [float(q.loc[p]) for p in ids if p in q.index]
    cols[-1].markdown(
        "<div style='padding-top:.35rem'>"
        f"<div style='font-size:.95rem;opacity:{'.8' if known else '.3'}'>"
        f"{f'{np.mean(known):.2f}' if known else '—'}</div>"
        f"<div style='opacity:.55;font-size:.74rem'>"
        f"{sum(pts.get(p, 0.0) for p in ids):.0f} pts on this unit</div></div>",
        unsafe_allow_html=True)


def _norm(card: dict[str, list[int]]) -> dict[str, list[int]]:
    """The card as `Scenario.set_lines` would store it, so the two can be compared."""
    out = {str(k): [int(p) for p in v if p not in (None, "")] for k, v in card.items()}
    return {k: v for k, v in out.items() if len(v) > 1}


def _preview(card: dict[str, list[int]], team: str, pool: pd.DataFrame,
             incremental: bool = True) -> pd.DataFrame:
    """What this lineup is worth, without re-running the projection.

    Mirrors the model's own arithmetic (see `project_skaters._apply_line_chemistry`): the
    fitted per-60 effect, then the same cap on how far one player's scoring rate may move.
    Forwards only -- pairings carry no measured signal and are not scored.

    Two modes, and the difference matters. `incremental` is for a card being edited: the
    numbers on screen already contain whatever lineup is saved, so the honest answer to
    "what would saving this do" is the difference between the two cards -- otherwise editing
    a saved lineup would appear to apply the whole adjustment a second time. With
    `incremental=False` the answer is the whole effect of this card against no lineup at
    all, which is what to show for a lineup that is already saved and applied.
    """
    dlq = ln.delta_lq({team: card})
    if dlq.empty:
        return pd.DataFrame()
    unit_of = {int(p): label for label, ids in card.items() for p in ids}
    # The projection frame carries its own `lines_dlq` (the saved lineup's). Keep it as
    # `applied` and drop the name, or pandas suffixes both and neither is `lines_dlq`.
    base = pool.rename(columns={"lines_dlq": "applied"}).drop(
        columns=["lines_rate_mult"], errors="ignore")
    if "applied" not in base.columns:
        base["applied"] = 0.0
    d = base.merge(dlq[["playerId", "lines_dlq"]], on="playerId", how="inner").copy()
    lim = C.LINES_MAX_DLQ
    d["dlq"] = d["lines_dlq"].clip(-lim, lim)
    applied = d["applied"].astype(float).fillna(0.0).clip(-lim, lim)
    even = d[[f"proj_{s}" for s in C.LINES_STATS]].sum(axis=1)
    cap = C.LINES_MAX_RATE_CHANGE * even
    per_dlq = C.LINES_BETA * d["proj_toi"] / 60.0
    want = np.clip(d["dlq"] * per_dlq, -cap, cap)      # where this lineup would put him
    have = np.clip(applied * per_dlq, -cap, cap)       # where the saved one already has him
    d["effect"] = want - (have if incremental else 0.0)
    d["capped"] = np.abs(d["dlq"] * per_dlq) > cap + 1e-9
    d["Unit"] = d["playerId"].map(unit_of)
    return d.sort_values("effect", ascending=False)


# --------------------------------------------------------------------------- #
# sections                                                                    #
# --------------------------------------------------------------------------- #
def _effect_panel(pending: dict[str, list[int]], team: str, pool: pd.DataFrame,
                  budget: float, saved: dict) -> None:
    """What the card on screen is worth, updated on every dropdown change.

    Once the card matches what is saved, this switches from "what saving would do" to what
    the saved lineup IS doing. Showing nothing at that point was the wrong answer to a fair
    question: having saved a lineup, the thing you want on screen is its effect, not an empty
    panel that reads as though the edit evaporated.
    """
    live = bool(saved) and _norm(pending) == _norm(saved)
    prev = _preview(pending, team, pool, incremental=not live)
    moved = prev[prev["effect"].abs() > 0.05] if not prev.empty else pd.DataFrame()
    if moved.empty:
        st.markdown("##### Effect on the projection")
        if live:
            st.info("This lineup is saved, and it matches last season's real deployment — so "
                    "there is nothing for it to change. The adjustment is driven by the "
                    "**change** in who a player skates with; move somebody and this fills in.")
        else:
            st.info("This card would change nothing — it matches either last season's real "
                    "deployment or the lineup already saved. The adjustment is driven by the "
                    "**change** in who a player skates with, so move somebody and this "
                    "fills in.")
        return

    container = st.container(border=True)
    container.markdown("##### Effect on the projection — "
                       + ("saved and applied" if live else "if you save this card"))
    up, down = moved.iloc[0], moved.iloc[-1]
    m = container.columns(4)
    m[0].metric("Players affected", len(moved))
    # Deliberately NOT shown as a delta on the team total: it is not one. The gains and
    # losses inside a team nearly cancel by construction, and the team budget absorbs most
    # of what is left over, so the team's projected points barely move at all.
    m[1].metric("Net across the team", f"{moved['effect'].sum():+.1f} pts",
                help="What is left after the gains and losses inside the team cancel — "
                     "somebody gets better linemates because somebody else got worse ones. "
                     f"The team budget ({budget:,.0f} points) then absorbs most of even "
                     "this, so the team total moves very little. The point of a lineup edit "
                     "is WHO gets the points, not how many.")
    # Labelled by what the numbers actually are. A card can move a whole forward group the
    # same way -- dropping a rookie into the top six lowers everyone's pool quality -- and
    # calling the least-bad row a "gain" would be a small lie on the front of the panel.
    m[2].metric("Biggest gain" if up["effect"] > 0 else "Smallest loss",
                up["name"], delta=f"{up['effect']:+.1f} pts")
    m[3].metric("Biggest drop" if down["effect"] < 0 else "Smallest gain",
                down["name"], delta=f"{down['effect']:+.1f} pts")

    show = moved[["name", "Unit", "position", "proj_toi_per_gp", "proj_points", "dlq",
                  "effect", "capped"]].copy()
    if live:
        # `proj_points` already contains the saved lineup, so the pair to show is what he
        # would be without it and what he is with it.
        show["a"], show["b"] = show["proj_points"] - show["effect"], show["proj_points"]
        head = ("Without the lineup", "In the projection")
    else:
        show["a"], show["b"] = show["proj_points"], show["proj_points"] + show["effect"]
        head = ("Points now", "Points if saved")
    container.dataframe(
        show[["name", "Unit", "position", "proj_toi_per_gp", "a", "b", "effect", "dlq",
              "capped"]],
        hide_index=True, width="stretch", height=min(420, 60 + 35 * len(show)),
        column_config={
            "name": "Player", "Unit": "Unit", "position": "Pos",
            "proj_toi_per_gp": st.column_config.NumberColumn("TOI/gm", format="%.1f"),
            "a": st.column_config.NumberColumn(head[0], format="%.1f"),
            "b": st.column_config.NumberColumn(head[1], format="%.1f"),
            "effect": st.column_config.NumberColumn(
                "Change", format="%+.1f",
                help="Before team budgets settle, so the final number moves a little less: "
                     "a player's gain is partly paid for by his teammates."),
            "dlq": st.column_config.NumberColumn(
                "Linemate change", format="%+.2f",
                help="Change in the points-per-60 quality of the players around him, "
                     "against last season's real lines."),
            "capped": st.column_config.CheckboxColumn(
                "Capped", help=f"His scoring rate hit the "
                               f"{C.LINES_MAX_RATE_CHANGE:.0%} limit."),
        })
    container.caption(
        ("These are estimates, accurate to about a tenth of a point, taken before team "
         "budgets settle. The **Effect of saved lineups** tab re-projects instead of "
         "estimating and is the exact number."))


def _editor(team: str, sk: pd.DataFrame) -> None:
    sc = core.scenario()
    # Two frames, deliberately. `on_team` is who the model has on the roster, and it is what
    # the suggested card is built from. `pool` adds the team's camp players -- the rookies,
    # prospects and depth bodies the September roster cut dropped -- because they are exactly
    # who a person knows something about that the model does not, and a dropdown that leaves
    # a first-overall pick out cannot describe next season's top line.
    camp = sk["camp"].fillna(False) if "camp" in sk else pd.Series(False, index=sk.index)
    ct = sk["camp_team"].fillna("") if "camp_team" in sk else pd.Series("", index=sk.index)
    on_team = sk[sk["team"] == team].copy()
    pool = sk[(sk["team"] == team) | (camp & (ct == team))].copy()
    if pool.empty:
        st.info("Nobody is projected on this team yet.")
        return
    pool["in_camp"] = camp.reindex(pool.index).fillna(False).to_numpy()
    # Named on a card but not on the roster: saving is what puts him there.
    needs_roster = set(int(p) for p in pool.loc[pool["in_camp"] & (pool["team"] != team),
                                                "playerId"])
    to_label, to_id = _labels(pool)
    saved = sc.team_lines(team)
    suggestion = ln.suggested_lineup(team, on_team if not on_team.empty else pool)
    reset = bool(st.session_state.pop(f"lnreset_{team}", False))
    card = suggestion if reset else (saved or suggestion)

    if saved:
        st.success("This team's lineup is saved, so it is affecting the projections below.")
        # A saved lineup can outlive a roster: a trade, or a scenario opened from the
        # library after the data was refreshed. Those names cannot be shown in a dropdown
        # that only lists this roster, so say so rather than showing a blank slot.
        gone = [p for ids in saved.values() for p in ids if p not in to_label]
        if gone:
            st.warning(f"{len(gone)} saved player{'s' if len(gone) != 1 else ''} "
                       f"{'are' if len(gone) != 1 else 'is'} no longer projected on this "
                       "team, so those slots show as blank. Saving again drops them; the "
                       "adjustment already ignores them.")
    else:
        st.info("Nothing is saved for this team, so the model's own projection is untouched. "
                "The lineup below is a starting point — last season's units carried onto "
                "this roster — not an edit. Change what is wrong and save it.")

    _seed_card(team, card, to_label, saved, force=reset)
    q = ln.player_quality()
    stat_text, stat_pts = _stat_maps(pool)

    # Positions on the card are the ones a hockey person expects to see. They are labels
    # only: the arithmetic reads a line as a set, so moving a winger from LW to RW inside
    # the same trio changes nothing. Said out loud under the card rather than left to be
    # discovered.
    for heading, labels, headings, group in (
            ("Forward lines", ln.FORWARD_LABELS, ["LW", "C", "RW"], "F"),
            ("Defence pairings", ln.PAIR_LABELS, ["LD", "RD"], "D")):
        options = _options(pool, to_label, group)
        size = ln.UNIT_SIZE[labels[0]]
        with st.container(border=True):
            st.markdown(f"**{heading}**"
                        + ("" if heading == "Forward lines" else
                           " &nbsp;<span style='opacity:.6;font-size:.85rem'>"
                           "— editable for realism, deliberately not scored</span>"),
                        unsafe_allow_html=True)
            head = st.columns([0.5] + [2] * size + [1.15])
            for col, h in zip(head[1:1 + size], headings[:size]):
                col.markdown(f"<div style='opacity:.6;font-size:.8rem'>{h}</div>",
                             unsafe_allow_html=True)
            head[-1].markdown(
                "<div style='opacity:.6;font-size:.8rem' title='Mean linemate value of the "
                "players on this unit: their regressed 5-on-5 points per 60.'>Unit value"
                "</div>", unsafe_allow_html=True)
            for label in labels:
                _unit_row(team, label, options, q, to_id, headings, stat_text, stat_pts)

    st.caption("Under each name is what he is projected for right now — ice time a game, then "
               "season points, goals and assists. Pick a player who is already on another "
               "unit and the two trade places, so nobody is ever in two spots; pick somebody "
               "who is not on the card at all and he takes the spot, leaving the man he "
               "replaced off it. LW / C / RW are labels for reading the card: the model "
               "scores a line as a set, so the order inside a line does not matter.")

    pending = _card_from_state(team, to_id)
    thin = [l for l, ids in pending.items() if 0 < len(ids) < ln.UNIT_SIZE[l]]
    if thin:
        st.warning(f"Short of players: {', '.join(thin)}. A unit with one name on it is "
                   "ignored, and a short line is scored on the players it does have.")

    named = [int(p) for ids in pending.values() for p in ids]
    promote = [p for p in named if p in needs_roster]
    if promote:
        who = ", ".join(f"**{to_label[p].split(' · ')[0]}**" for p in promote)
        s = len(promote) > 1
        st.warning(
            f"On this card but not on {team}'s projected roster: {who}. Saving the lineup "
            f"puts {'them' if s else 'him'} on it, which is the same edit as naming a team on "
            f"{'their' if s else 'his'} own page — every teammate then resettles around "
            f"{'them' if s else 'him'}. Ice time is still the model's guess at a spare "
            f"forward, so set **TOI/gm** and games on the player page if "
            f"{'they are' if s else 'he is'} going to play a real role.")
    # A player with no NHL 5-on-5 history has no linemate value, so he is not merely
    # unadjusted -- he is invisible to the arithmetic in both directions, including for the
    # stars he is put beside. Anyone naming a rookie to the top line deserves to be told.
    unseen = [p for label, ids in pending.items() if label in ln.FORWARD_LABELS
              for p in ids if p not in q.index]
    if unseen:
        who = ", ".join(to_label[p].split(" · ")[0] for p in unseen)
        st.caption(f"No NHL history, so the line adjustment cannot see {'them' if len(unseen) > 1 else 'him'}"
                   f" — it measures the *change* in linemate quality and an unplayed player "
                   f"has none, in either direction: {who}. Naming "
                   f"{'them' if len(unseen) > 1 else 'him'} here is recorded and puts "
                   f"{'them' if len(unseen) > 1 else 'him'} on the roster; what "
                   f"{'they are' if len(unseen) > 1 else 'he is'} worth has to be stated as "
                   f"ice time and rates on the player page.")

    c1, c2, c3, _ = st.columns([1.2, 1, 1, 1.4])
    if c1.button("Save this lineup", type="primary", width="stretch"):
        # The lineup and the promotions are one commit: a card that names somebody is a
        # statement that he is on the team, and applying half of that would leave the page
        # showing a lineup the projection does not have.
        out = sc.set_lines(team, pending)
        for pid in promote:
            out = out.set_player(pid, team=team)
        core.commit(out, f"{team} lineup saved"
                         + (f" · {len(promote)} added to the roster" if promote else ""))
    if c2.button("Clear this team", width="stretch", disabled=not saved,
                 help="Removes the saved lineup: this team goes back to no line adjustment "
                      "at all, and the card returns to the suggestion. Anyone you added to "
                      "the roster stays on it — that is undone on his own page."):
        core.edit_lines(team, None, f"{team} lineup cleared")
    if c3.button("Reset the card", width="stretch",
                 help="Puts the dropdowns back to last season's real deployment. Changes "
                      "nothing in the projection until you save."):
        st.session_state[f"lnreset_{team}"] = True
        st.rerun()

    _effect_panel(pending, team, pool, float(on_team["proj_points"].sum()), saved)

    with st.expander(f"Everyone {team} can ice — {len(on_team)} on the roster, "
                     f"{len(pool) - len(on_team)} more in camp"):
        avail = pool.copy()
        avail["quality"] = avail["playerId"].map(q)
        avail["Unit"] = avail["playerId"].map(
            {int(p): l for l, ids in pending.items() for p in ids}).fillna("—")
        avail["Status"] = np.where(
            avail["team"] != team, "In camp",
            np.where(avail["in_camp"], "Camp → roster", "Roster"))
        st.dataframe(
            avail.sort_values(["in_camp", "proj_toi_per_gp"], ascending=[True, False])[
                ["name", "position", "Status", "Unit", "proj_toi_per_gp", "proj_points",
                 "quality"]],
            hide_index=True, width="stretch",
            column_config={
                "name": "Player", "position": "Pos",
                "Status": st.column_config.TextColumn(
                    "Status", help="'In camp' is a player the September roster cut left off "
                                   "the projected roster — he is off every team budget until "
                                   "you name him on a line or give him a team on his own "
                                   "page. 'Camp → roster' is one you have already added."),
                "proj_toi_per_gp": st.column_config.NumberColumn("TOI/gm", format="%.1f"),
                "proj_points": st.column_config.NumberColumn("Points", format="%.1f"),
                "quality": st.column_config.NumberColumn(
                    "Linemate value", format="%.2f",
                    help="His 5-on-5 points per 60 over the last three seasons, regressed. "
                         "This is what he is worth TO A LINEMATE, which is not the same as "
                         "his own projection. Blank means no NHL history, and he is left "
                         "out of the arithmetic rather than counted as replacement level."),
            })


def _saved_effect() -> None:
    """The exact effect of every saved lineup, by re-projecting without them."""
    sc = core.scenario()
    if not sc.lines:
        st.caption("No lineups are saved yet.")
        return
    with_lines, _ = core.skaters()
    without, _ = core.skaters(sc.clear_lines())
    m = without[["playerId", "name", "team", "proj_points", "proj_goals",
                 "proj_assists"]].merge(
        with_lines[["playerId", "proj_points", "proj_goals", "proj_assists", "lines_dlq"]],
        on="playerId", suffixes=("_off", "_on"))
    m["Points"] = m["proj_points_on"] - m["proj_points_off"]
    m["Goals"] = m["proj_goals_on"] - m["proj_goals_off"]
    m["Assists"] = m["proj_assists_on"] - m["proj_assists_off"]
    m = m[m["Points"].abs() > 0.05].sort_values("Points", ascending=False)
    st.caption(f"Saved lineups: {', '.join(sorted(sc.lines))}. This compares the projection "
               "you are looking at against the same scenario with the lineups removed, so "
               "it is the real effect after team budgets have settled — not an estimate.")
    if m.empty:
        st.info("The saved lineups match last season's deployment, so nothing moved.")
        return
    st.dataframe(
        m[["name", "team", "proj_points_off", "proj_points_on", "Points", "Goals",
           "Assists", "lines_dlq"]],
        hide_index=True, width="stretch", height=min(600, 60 + 35 * len(m)),
        column_config={
            "name": "Player", "team": "Team",
            "proj_points_off": st.column_config.NumberColumn("Without", format="%.1f"),
            "proj_points_on": st.column_config.NumberColumn("With", format="%.1f"),
            "Points": st.column_config.NumberColumn("Points", format="%+.1f"),
            "Goals": st.column_config.NumberColumn("Goals", format="%+.1f"),
            "Assists": st.column_config.NumberColumn("Assists", format="%+.1f"),
            "lines_dlq": st.column_config.NumberColumn("Linemate change", format="%+.2f"),
        })
    if len(sc.lines) > 1 and st.button("Clear every saved lineup"):
        core.commit(sc.clear_lines(), "All lineups cleared")


def _notes() -> None:
    st.markdown(f"""
**What is actually being measured.** Every skater gets a *linemate value*: his 5-on-5
points per 60 over the last three seasons, shrunk toward his position's average so a short
sample cannot masquerade as a star. A player's *linemate quality* is then how good the
people around him are. The projection is nudged by
`{C.LINES_BETA} x (new linemate quality - last season's)`.

**Why only the change.** The model is built on five seasons of each player's own numbers,
and those numbers already include whatever his old linemates did for him. Telling it "he
plays with a good centre" would double-count. Telling it "he plays with a *different*
centre than the one in his history" is new information, and that is what this page supplies.

**A line is not a player's whole deployment.** The median NHL skater spends only about 36%
of his 5-on-5 minutes on his most-used unit — lines churn constantly through injuries,
matchups and coaches changing their minds. So a saved lineup is read as
{C.LINES_UNIT_SHARE:.0%} his named linemates and {1 - C.LINES_UNIT_SHARE:.0%} the rest of
the team's group, rather than being taken as gospel.

**Line numbers do not matter; linemates do.** Swapping the L1 and L4 labels with the same
players in each changes nothing, because nobody's linemates changed. What a promotion is
really worth in ice time is a separate, larger effect, and it is edited directly as
`TOI/gm` on the player's own page. The same goes for the LW / C / RW and LD / RD headings:
they are there so the card reads like a lineup sheet, but the arithmetic treats a line as a
set of three players, so moving a winger from one side to the other is not an edit.

**Rookies and camp players.** The roster feed lists close to 40 players a team in September,
so before anything is projected the model trims each team to a plausible active roster —
otherwise 40 players claim from a 23-player pool of minutes and everybody, McDavid included,
gets about a fifth shaved off. Whoever is trimmed keeps his projection, is marked as being in
that team's camp, and sits off every budget. They are all in the dropdowns here, because "he
is going to play all 82 games this year" is exactly the kind of thing a person knows in
September and a five-year history file cannot: name one on a line and saving puts him on the
roster. Two things to know about doing it. His ice time is still the model's guess at a spare
forward, which is set on his own page. And a player with no NHL games at all has no linemate
value, so the line arithmetic cannot see him in either direction — not for himself, and not
for the star he is put beside. What a rookie is worth has to be stated, not inferred.

**Defence pairings can be edited but are not scored.** The same test run on pairings found
a correlation of +0.05 with model error and no accuracy gain at all, so pairing edits are
recorded for realism and deliberately left out of the arithmetic. Anyone claiming to
project a defenceman off his partner is guessing.

**How much this is worth.** Tested by holding out one season at a time and re-projecting it
with nothing but prior data (2022-2025, skaters with 300+ minutes): per-60 error fell about
0.9% and season point error about 0.4%, and it was an improvement in three of those four
seasons. Those averages are diluted on purpose — only about 4 forwards in 10 have a linemate
change big enough to be touched at all, so for the players this page is actually about the
effect is roughly two and a half times those numbers. It is a small, real, honestly-measured
edge, not a rewrite of the projection. Scoring the same idea on each player's *realized*
season-long deployment rather than a lineup card is worth about three times as much, which
is why this matters more in season than in September.
""")


def page() -> None:
    core.scenario_bar()
    sk, _ = core.skaters()
    sc = core.scenario()
    n = len(sc.lines)
    core.header("Lines",
                f"{n} team{'s' if n != 1 else ''} with a saved lineup." if n else
                "No lineups saved — every team is on the model's own read.")

    teams = core.team_options(sk)
    saved = set(sc.lines)
    # The picker holds its own state and is never given an `index` after the first draw. It
    # used to take `index=<first saved team>`, and that quietly broke the moment a lineup was
    # saved: changing `index` changes the widget's identity, Streamlit treats it as a new
    # widget, and the box snaps to the new default. Saving Chicago's lineup therefore jumped
    # the page to Boston, and the card you had just been editing looked as though it had been
    # wiped. Landing on a saved team still happens -- but only when there is no choice yet.
    if st.session_state.get("lnteam") not in teams:
        st.session_state["lnteam"] = teams[
            next((i for i, t in enumerate(teams) if t in saved), 0)]
    c1, c2 = st.columns([1, 4], vertical_alignment="bottom")
    pick = c1.selectbox("Team", teams, key="lnteam",
                        format_func=lambda t: f"{t} ✓" if t in saved else t)
    n_on = int((sk["team"] == pick).sum())
    n_camp = int((sk.get("camp", pd.Series(False, index=sk.index)).fillna(False)
                  & (sk.get("camp_team", pd.Series("", index=sk.index)).fillna("") == pick)
                  ).sum())
    c2.caption(f"A tick marks a team whose lineup is saved. {n_on} players are projected on "
               f"{pick} and {n_camp} more are in its camp; all of them are in every dropdown.")

    tab_edit, tab_effect, tab_notes = st.tabs(
        ["Lineup", "Effect of saved lineups", "How this works"])
    with tab_edit:
        _editor(pick, sk)
    with tab_effect:
        _saved_effect()
    with tab_notes:
        _notes()

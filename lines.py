"""Line chemistry: what changing a player's linemates is worth, and what it is not.

The idea in one paragraph
------------------------
A skater's projection is built from his own history, and that history already contains
whatever his old linemates did for him -- a winger who spent three years beside a star
centre has the star baked into his own points per 60. So knowing WHO he plays with adds
nothing on its own. The only thing line data can add is the CHANGE: he is moving up to
the first line, or he was traded and his new team's top centre is worse than his old one.
That change is what this module measures, and it is measured against last season's lines
rather than against nothing.

Linemate quality (LQ)
---------------------
Every player gets a quality number `q`: his 5on5 points per 60, averaged over the last
few completed seasons, weighted by ice time and regressed toward his position's mean so a
90-minute sample does not read as elite. `q` is computed from seasons strictly BEFORE the
season being projected, so nothing about the outcome can leak into the prediction.

A player's LQ is then how good the players around him are. Crucially, a nominal line is
not a player's deployment: the median skater spends only ~36% of his 5on5 minutes on his
single most-used unit. So LQ from a lineup card blends the two things a card knows:

    LQ = share * (mean q of his linemates) + (1 - share) * (mean q of the team's other forwards)

The adjustment is `beta * (LQ_new - LQ_old)` added to his points per 60, then converted
to a multiplier on his goal and assist rates so the scoring mix survives. See config for
the fitted coefficient, the clamps, and what the measured accuracy gain actually is.

Two deliberate limits
---------------------
1. Defense pairings are stored and displayed but do NOT score. The same test on 14-digit
   (pairing) lineIds found a correlation of +0.054 with model error and a leave-one-season-
   out change of -0.1% -- nothing. Editing them is for realism, not for accuracy.
2. A team with no saved lineup gets no adjustment at all, and a player not named in a
   saved lineup gets no adjustment. Silence means "the model's own opinion", exactly as it
   does everywhere else in the scenario system.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

import config as C
import data_layer as dl

FORWARD_LABELS = tuple(f"L{i}" for i in range(1, C.LINES_FORWARD_UNITS + 1))
PAIR_LABELS = tuple(f"D{i}" for i in range(1, C.LINES_PAIR_UNITS + 1))
UNIT_SIZE = {**{l: C.LINES_FORWARD_SIZE for l in FORWARD_LABELS},
             **{l: C.LINES_PAIR_SIZE for l in PAIR_LABELS}}


# --------------------------------------------------------------- observed units
@lru_cache(maxsize=1)
def unit_memberships() -> pd.DataFrame:
    """Explode observed 5on5 units into one row per (season, team, unit, player).

    Units below `LINES_MIN_UNIT_TOI` minutes are dropped: a combination that happened for
    six minutes because of a double shift is not a line, and letting it through would let
    noise decide a team's depth chart.
    """
    lines = dl.load_moneypuck_lines()
    rows = []
    for lid, season, team, ice in zip(lines["lineId"], lines["mp_season_year"],
                                      lines["team"], lines["icetime"]):
        toi = ice / 60.0
        if toi < C.LINES_MIN_UNIT_TOI:
            continue
        ids = dl.split_line_ids(lid)
        for pid in ids:
            rows.append((int(season), str(lid), team, pid, toi, len(ids)))
    out = pd.DataFrame(rows, columns=["season", "lineId", "team", "playerId",
                                      "unit_toi", "unit_size"])
    out["pos_group"] = np.where(out["unit_size"] == C.LINES_PAIR_SIZE, "D", "F")
    return out


@lru_cache(maxsize=4)
def player_quality(target: int = C.TARGET_SEASON) -> pd.Series:
    """playerId -> regressed 5on5 points/60 from the seasons before `target`.

    Regressed toward the position-group mean at `LINES_QUALITY_REGRESS_TOI`, the same
    shrink constant the skater rates themselves use, so an unproven linemate is treated as
    an average one rather than as whatever his small sample happened to say.
    """
    yrs = [target - i for i in range(1, C.LINES_QUALITY_SEASONS + 1)]
    sk = dl.load_moneypuck_situation("5on5")
    h = sk[sk["mp_season_year"].isin(yrs) & (sk["icetime"] > 0)].copy()
    h["toi_min"] = h["icetime"] / 60.0
    h["p60"] = h["I_F_points"] * 60.0 / h["toi_min"]
    h["pos_group"] = np.where(h["position"] == "D", "D", "F")
    pos_mean = {g: float(np.average(gg["p60"], weights=gg["toi_min"]))
                for g, gg in h.groupby("pos_group")}
    agg = h.groupby("playerId").apply(
        lambda g: pd.Series({
            "toi": g["toi_min"].sum(),
            "raw": float(np.average(g["p60"], weights=g["toi_min"])),
            "pos_group": g["pos_group"].iloc[-1],
        }), include_groups=False)
    tgt = agg["pos_group"].map(pos_mean).astype(float)
    k = C.LINES_QUALITY_REGRESS_TOI
    return (agg["raw"] * agg["toi"] + tgt * k) / (agg["toi"] + k)


def observed_lineups(season: int, pos_group: str = "F") -> dict[str, list[list[int]]]:
    """Each team's depth chart as actually deployed in `season`, most-used units first.

    Greedy and disjoint: take the team's highest-TOI unit, then the highest-TOI unit that
    shares nobody with it, and so on. A player appears at most once, which is what makes
    the result a lineup card rather than a list of combinations.
    """
    m = unit_memberships()
    m = m[(m["season"] == season) & (m["pos_group"] == pos_group)]
    if m.empty:
        return {}
    n_units = C.LINES_FORWARD_UNITS if pos_group == "F" else C.LINES_PAIR_UNITS
    size = C.LINES_FORWARD_SIZE if pos_group == "F" else C.LINES_PAIR_SIZE
    out: dict[str, list[list[int]]] = {}
    units = (m.groupby(["team", "lineId"])["unit_toi"].first()
              .reset_index().sort_values("unit_toi", ascending=False))
    for team, g in units.groupby("team"):
        used: set[int] = set()
        card: list[list[int]] = []
        for lid in g["lineId"]:
            ids = dl.split_line_ids(lid)
            if len(ids) != size or used & set(ids):
                continue
            card.append(ids)
            used |= set(ids)
            if len(card) == n_units:
                break
        out[team] = card
    return out


# --------------------------------------------------------------- LQ from a card
def _card_lq(card: dict[str, list[int]], q: pd.Series) -> dict[int, float]:
    """Linemate quality for every player named on ONE team's card.

    The pool term is the mean quality of the team's other players in the same position
    group who are NOT on the player's own unit -- "whoever else you might get put with".
    Unknown players (no NHL history) are dropped from both terms rather than counted as
    zero, which would read as "you now play with a replacement-level ghost".
    """
    qs: dict[int, float] = {}
    unit_of: dict[int, str] = {}
    for label, ids in card.items():
        for pid in ids:
            pid = int(pid)
            if pid in q.index and pid not in qs:
                qs[pid] = float(q.loc[pid])
                unit_of[pid] = label
    share = C.LINES_UNIT_SHARE
    out: dict[int, float] = {}
    for pid, label in unit_of.items():
        mates = [int(p) for p in card[label] if int(p) != pid and int(p) in qs]
        pool = [v for p, v in qs.items()
                if p != pid and int(p) not in {int(x) for x in card[label]}]
        if not mates:
            continue
        mate_q = float(np.mean([qs[p] for p in mates]))
        if pool:
            out[pid] = share * mate_q + (1.0 - share) * float(np.mean(pool))
        else:
            out[pid] = mate_q
    return out


def _as_card(units, labels) -> dict[str, list[int]]:
    """Normalise a stored lineup (label -> ids) to ints, keeping only known labels."""
    card: dict[str, list[int]] = {}
    for label in labels:
        ids = [int(p) for p in (units.get(label) or []) if p not in (None, "")]
        if len(ids) > 1:
            card[label] = ids
    return card


@lru_cache(maxsize=4)
def baseline_lq(target: int = C.TARGET_SEASON) -> pd.Series:
    """Every forward's LQ under last season's REAL lines -- the "before" of the delta.

    Computed on the team he actually played for, which is the point: a traded player's
    baseline is his old deployment, because that is what his own history reflects.
    """
    q = player_quality(target)
    out: dict[int, float] = {}
    for team, card in observed_lineups(target - 1, "F").items():
        named = {FORWARD_LABELS[i]: ids for i, ids in enumerate(card)
                 if i < len(FORWARD_LABELS)}
        out.update(_card_lq(named, q))
    return pd.Series(out, dtype=float)


def delta_lq(lines: dict, target: int = C.TARGET_SEASON) -> pd.DataFrame:
    """Change in linemate quality implied by the saved lineups, per (team, player).

    Keyed by TEAM as well as player on purpose. A saved lineup can name somebody who is no
    longer on that team -- a card carried over from last season, or a trade the roster has
    since caught up with -- and applying an Edmonton line bump to a player now projected in
    Anaheim would be a straightforwardly wrong number. The caller matches on both, so a
    stale name is simply ignored.

    Only forwards score (see the module docstring on pairings). A player is left out --
    meaning no adjustment -- when his team has no saved lineup, when he is not named on it,
    or when he has no prior-season line to be compared against.
    """
    cols = ["team", "playerId", "lines_dlq"]
    if not lines:
        return pd.DataFrame(columns=cols)
    q = player_quality(target)
    base = baseline_lq(target)
    rows = []
    for team, units in lines.items():
        card = _as_card(units or {}, FORWARD_LABELS)
        if not card:
            continue
        for pid, lq in _card_lq(card, q).items():
            if pid in base.index:
                rows.append((str(team), int(pid), lq - float(base.loc[pid])))
    return pd.DataFrame(rows, columns=cols)


def suggested_lineup(team: str, roster: pd.DataFrame,
                     target: int = C.TARGET_SEASON) -> dict[str, list[int]]:
    """A starting point for the editor: last season's units, carried onto this roster.

    Prior units are kept where they survived, holes are filled by projected ice time, and
    anyone left over sits out. Carrying the prior units forward rather than simply ranking
    by ice time matters: it means saving an untouched lineup for a team that did not change
    is close to a no-op, so the adjustment fires where deployment actually moved.

    `roster` needs `playerId`, `pos_group` and `proj_toi_per_gp` for one team.
    """
    r = roster.sort_values("proj_toi_per_gp", ascending=False)
    card: dict[str, list[int]] = {}
    for pos, labels in (("F", FORWARD_LABELS), ("D", PAIR_LABELS)):
        avail = [int(p) for p in r.loc[r["pos_group"] == pos, "playerId"]]
        avail_set = set(avail)
        size = UNIT_SIZE[labels[0]]
        prior = observed_lineups(target - 1, pos).get(team, [])
        kept = [[p for p in unit if p in avail_set] for unit in prior]
        kept = [u for u in kept if u]
        placed: set[int] = set()
        units: list[list[int]] = []
        for unit in kept:
            unit = [p for p in unit if p not in placed][:size]
            if unit:
                units.append(unit)
                placed |= set(unit)
            if len(units) == len(labels):
                break
        spare = [p for p in avail if p not in placed]
        while len(units) < len(labels):
            units.append([])
        for unit in units:                       # fill holes in depth order
            while len(unit) < size and spare:
                unit.append(spare.pop(0))
        for label, unit in zip(labels, units):
            card[label] = unit
    return card

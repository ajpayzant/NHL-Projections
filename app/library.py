"""The shared scenario library: saved projections other people can open.

Locally there is nothing to solve -- a scenario is a small JSON file in `scenarios/` and
the filesystem is yours. Deployed, two things change and both matter.

First, every visitor shares one container, so a single `working.json` on disk would mean
one person's edits silently rewriting everybody else's view. The working scenario is
therefore held in session memory when the app is shared (see `core.MULTIUSER`), and each
visitor starts from the baseline model.

Second, a deployed container's disk is temporary. It is wiped on every redeploy and after
the app sleeps, so "Save as" writing to `scenarios/` would look like it worked and then
lose the file a day later. That is worse than not offering it. So the library has two
backends and says which one it is using:

  gist  -- a GitHub gist, addressed by `gist_id` + `github_token` in Streamlit secrets.
           Durable, versioned by GitHub, shared by every visitor, and the token never
           enters the repository. This is the one that makes "review someone else's
           projections" true.
  disk  -- `scenarios/*.json`. The local default, and the fallback when no gist is
           configured, in which case the app says out loud that saves are temporary.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import requests
import streamlit as st

import config as C
import overrides as ov

GIST_API = "https://api.github.com/gists"
PREFIX = "scenario__"          # so the gist can hold other files without confusing us
TIMEOUT = 20


# --------------------------------------------------------------------------- #
# which backend                                                               #
# --------------------------------------------------------------------------- #
def secret(key: str, default=None):
    """A secret, or the default. Absent secrets are normal, not an error."""
    try:
        return st.secrets.get(key, default)
    except Exception:                                          # noqa: BLE001
        return default


def _gist() -> tuple[str, str] | None:
    gid, tok = secret("gist_id"), secret("github_token")
    return (str(gid), str(tok)) if gid and tok else None


def backend() -> str:
    return "gist" if _gist() else "disk"


# Streamlit Community Cloud checks the repo out under /mount/src and rebuilds that checkout
# on every redeploy, and the container is torn down when the app sleeps. So "disk" there
# means "until the next restart", while on your own machine it means your own filesystem.
# The difference decides whether the app is allowed to call a save permanent, which is the
# one thing it must not get wrong: a save reported as successful and then lost is worse
# than a save that was refused.
EPHEMERAL_DISK = "/mount/src" in C.ROOT.as_posix()


def durable() -> bool:
    """Do saves survive a restart? False means the app has to say so, loudly."""
    return backend() == "gist" or not EPHEMERAL_DISK


def where() -> str:
    """One sentence naming the store, for the page to print next to the save button."""
    if backend() == "gist":
        return ("a GitHub gist, kept until somebody deletes it — it survives restarts, "
                "redeploys and every data refresh")
    if EPHEMERAL_DISK:
        return ("this container's temporary disk, which is wiped when the app restarts, "
                "redeploys or goes to sleep")
    return f"`{C.SCENARIOS}` on this machine, kept until you delete the file"


# --------------------------------------------------------------------------- #
# gist backend                                                                #
# --------------------------------------------------------------------------- #
def _headers(tok: str) -> dict:
    return {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"}


@st.cache_data(ttl=20, show_spinner=False)
def _gist_files(gid: str, tok: str, bust: int = 0) -> dict[str, str]:
    """{name: json text} for every scenario in the gist. Cached briefly.

    `bust` is bumped after a write so the next read cannot serve a stale list -- the
    alternative, a 20-second window in which your own save is invisible, reads as a bug.
    """
    r = requests.get(f"{GIST_API}/{gid}", headers=_headers(tok), timeout=TIMEOUT)
    r.raise_for_status()
    out = {}
    for fname, meta in (r.json().get("files") or {}).items():
        if not fname.startswith(PREFIX) or not fname.endswith(".json"):
            continue
        text = meta.get("content")
        if meta.get("truncated") and meta.get("raw_url"):
            text = requests.get(meta["raw_url"], timeout=TIMEOUT).text
        out[fname[len(PREFIX):-5]] = text or ""
    return out


def _gist_write(gid: str, tok: str, fname: str, content: str | None) -> None:
    body = {"files": {fname: (None if content is None else {"content": content})}}
    r = requests.patch(f"{GIST_API}/{gid}", headers=_headers(tok), json=body,
                       timeout=TIMEOUT)
    r.raise_for_status()
    st.session_state["lib_bust"] = st.session_state.get("lib_bust", 0) + 1


# --------------------------------------------------------------------------- #
# the library                                                                 #
# --------------------------------------------------------------------------- #
def _meta(name: str, text: str) -> dict:
    """Name, author and size of one saved scenario, for the list a reader picks from.

    The size is counted by `Scenario.count()` rather than by walking the JSON here, so
    the number in the library matches the number in the sidebar. Counting it twice is
    how a lineup-only scenario came to be listed as "0 edits" -- which reads as an empty
    save and is the one thing that would stop somebody opening it.
    """
    try:
        sc = ov.Scenario.from_json(text)
    except (ValueError, TypeError):
        return {"name": name, "author": "?", "saved": "?", "edits": 0, "broken": True}
    notes = sc.notes or {}
    return {"name": name, "author": notes.get("author") or "-",
            "saved": (notes.get("saved_at") or "-")[:16].replace("T", " "),
            "edits": sc.count()["edits"], "broken": False}


def entries() -> list[dict]:
    """Every saved scenario, newest first."""
    g = _gist()
    if g:
        try:
            files = _gist_files(*g, bust=st.session_state.get("lib_bust", 0))
        except requests.RequestException as exc:
            st.warning(f"The scenario library is unreachable ({exc}). "
                       "Your own edits are unaffected.")
            return []
        rows = [_meta(n, t) for n, t in files.items()]
    else:
        rows = []
        for p in sorted(C.SCENARIOS.glob("*.json")):
            if p.stem == "working":
                continue
            rows.append(_meta(p.stem, p.read_text(encoding="utf-8")))
    return sorted(rows, key=lambda r: r["saved"], reverse=True)


def names() -> list[str]:
    return [r["name"] for r in entries()]


def save(sc: ov.Scenario, name: str, author: str = "") -> dict:
    """Publish a copy of `sc` under `name`. Overwrites a scenario of the same name.

    Returns what actually happened -- the stored name (which is sanitised, so it is not
    always the name that was typed), the store it went to, whether that store survives a
    restart, and whether reading the library back found it. The caller reports those rather
    than assuming success, because "Saved as opening-night" over a scenario that is already
    gone is the failure that costs somebody an afternoon of edits.
    """
    name = "".join(ch for ch in name.strip() if ch.isalnum() or ch in " -_").strip()
    if not name:
        raise ValueError("A scenario needs a name.")
    notes = dict(sc.notes)
    notes["author"] = (author or "anonymous").strip()[:60]
    notes["saved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # Every bucket, `lines` included. Naming them one by one is what dropped the lineups
    # on publish while the download button kept them, so the copy is asserted to be a copy
    # rather than trusted: if a new bucket is added to Scenario, this fails loudly here
    # instead of silently saving without it.
    copy = ov.Scenario(name=name, players=sc.players, goalies=sc.goalies, teams=sc.teams,
                       league=sc.league, notes=notes, lines=sc.lines)
    assert copy.digest == sc.digest, "the published copy is not the scenario"
    g = _gist()
    if g:
        _gist_write(*g, f"{PREFIX}{name}.json", copy.to_json())
    else:
        copy.save(C.SCENARIOS / f"{name}.json")
    # Read it back. A write that raised nothing is not the same as a scenario that is in
    # the library, and the round trip also proves the stored JSON still parses.
    verified = False
    try:
        verified = load(name).digest == sc.digest
    except (FileNotFoundError, ValueError, requests.RequestException):
        verified = False
    return {"name": name, "backend": backend(), "durable": durable(),
            "verified": verified, "where": where()}


def load(name: str) -> ov.Scenario:
    g = _gist()
    if g:
        files = _gist_files(*g, bust=st.session_state.get("lib_bust", 0))
        if name not in files:
            raise FileNotFoundError(name)
        sc = ov.Scenario.from_json(files[name])
    else:
        path = C.SCENARIOS / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(name)
        sc = ov.Scenario.load(path)
    sc.name = "working"
    return sc


def delete(name: str) -> None:
    g = _gist()
    if g:
        _gist_write(*g, f"{PREFIX}{name}.json", None)
    else:
        (C.SCENARIOS / f"{name}.json").unlink(missing_ok=True)

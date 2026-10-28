"""Field notes (`backend/field_notes.py`): the names and notes given to neurons and attention heads, kept by the server.

The atlas (`#/atlas`, backend/atlas.py) and its scientist chat name what they study, as the neuron explorer this page
follows keeps its names (lyramakesmusic/neuron-explorer, `explorer/names.json`). A neuron's folio reads and writes its
name here too, and its field book beside them (below), whenever this server answers it; with no server it keeps them
in the browser (frontend/src/screens/NeuronFolio/saved.ts). These are shared by every page and every chat this server
answers. One JSON file holds them, outside every checkout so each one the server is started from shares it:
`$THEODOLITE_FIELD_NOTES`, else `${XDG_DATA_HOME:-~/.local/share}/theodolite/field_notes.json`. Its shape:

    {"version": 1,
     "neurons": {"<layer>:<neuron>": {"name", "note", "by", "name_by", "note_by", "at"}},
     "heads":   {"<layer>:<head>":   {"name", "note", "by", "name_by", "note_by", "at"}}}

Layers count from 0 in the keys and in every field, as every field the server sends does (page layer = layer + 1).
`by` is who wrote the entry last ("reader" | "scientist"), `name_by` and `note_by` who wrote its name and its note (a
note added to a scientist's name leaves the name the scientist's), `at` when, ISO-8601 UTC. The routes are the atlas's
(`GET` and `POST /api/atlas/notes`, backend/atlas.py, which answer only pages and scripts on this machine).

The folio's field book (frontend/src/screens/NeuronFolio.tsx, section vii) is kept per face in a file of its own,
`$THEODOLITE_FIELD_BOOK`, else `field_book.json` beside the notes. A file of its own because the notes' file is shared
by every checkout's server, and a server from before reads only version 1 of it (it moves aside any other) and keeps
only the fields it knows on its next write: it never reads or writes this one. Its shape:

    {"version": 1,
     "neurons": {"<layer>:<neuron>": {"+": {"jobs", "pattern", "checked", "by", "at"}, "-": {...}}}}

`jobs` is "one" | "two" | "unknown" (not yet understood) | null (not yet said), `pattern` the face's pattern in a
sentence or two (at most 4,000 characters), `checked` its four ticks {"other_pole", "long_contexts", "steered",
"readers_traced"}, each true or false, `by` who wrote the face last and `at` when. A face with nothing said is
forgotten. The folio's name for a neuron is the notes' `name` above, by "reader" (`GET` and `POST /api/atlas/fieldbook`).

The names and notes the scientist writes (backend/scientist_tools.py `set_name`, `write_note`) are proposals until the
person keeps them, in a third file, `$THEODOLITE_FIELD_PROPOSALS`, else `field_proposals.json` beside the notes:

    {"version": 1,
     "neurons": {"<layer>:<neuron>": {"name"?, "note"?, "chat", "at"}},
     "heads":   {"<layer>:<head>":   {"name"?, "note"?, "chat", "at"}}}

(an empty name or note proposes clearing it; `chat` is the chat that proposed it). Nothing reads a proposal as a name:
the atlas's lists, search, tooltips and map, and later chats, read only the notes above. Keeping one writes it to the
notes as the scientist's (`name_by`, `note_by` "scientist") and removes it; discarding removes it (`GET` and `POST
/api/atlas/proposals`, `POST /api/atlas/proposals/keep` and `/discard`).

Writes take this module's own lock (never the GPU's: notes are written while the model works) and replace the file
atomically (`saved_prompts._write_atomic`). A file that cannot be read is moved aside to `<file>.corrupt-<UTC stamp>`,
never written over, and the next reply says so once (`warning`), as the saved prompts do.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Collection, Dict, Optional, Tuple

from backend.saved_prompts import _has_broken, _has_control, _write_atomic

logger = logging.getLogger("theodolite.field_notes")

FIELD_NOTES_ENV = "THEODOLITE_FIELD_NOTES"
FIELD_BOOK_ENV = "THEODOLITE_FIELD_BOOK"
FIELD_PROPOSALS_ENV = "THEODOLITE_FIELD_PROPOSALS"
FILE_VERSION = 1
MAX_NAME = 120
MAX_NOTE = 4000
MAX_PATTERN = 4000
MAX_ENTRIES = 20000
MAX_PROPOSALS = 5000
MAX_CHAT = 200
KINDS = {"neuron": "neurons", "head": "heads"}
WRITERS = ("reader", "scientist")
FACES = ("+", "-")
JOBS = ("one", "two", "unknown")
TICKS = ("other_pole", "long_contexts", "steered", "readers_traced")

_LOCK = asyncio.Lock()
# each file's warning, kept for the next reply: "notes", "book", "proposals"
_pending: Dict[str, str] = {}
# the notes as last read, per file, with the file's (mtime, size) then: every hover of the atlas reads them
_CACHE: Dict[str, Tuple[Tuple[int, int], Dict[str, Any]]] = {}


def field_notes_path() -> Path:
    """Where the notes live, read on every request (so the environment decides, also in the checks)."""
    env = os.environ.get(FIELD_NOTES_ENV)
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base).expanduser() / "theodolite" / "field_notes.json"


def field_book_path() -> Path:
    """Where the folio's field book lives: `$THEODOLITE_FIELD_BOOK`, else beside the notes."""
    env = os.environ.get(FIELD_BOOK_ENV)
    if env:
        return Path(env).expanduser()
    return field_notes_path().with_name("field_book.json")


def field_proposals_path() -> Path:
    """Where the scientist's proposals wait: `$THEODOLITE_FIELD_PROPOSALS`, else beside the notes."""
    env = os.environ.get(FIELD_PROPOSALS_ENV)
    if env:
        return Path(env).expanduser()
    return field_notes_path().with_name("field_proposals.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _empty() -> Dict[str, Any]:
    return {"version": FILE_VERSION, "neurons": {}, "heads": {}}


def note_key(layer: int, index: int) -> str:
    return f"{int(layer)}:{int(index)}"


def _valid_key(k: Any) -> bool:
    if not isinstance(k, str):
        return False
    a, sep, b = k.partition(":")
    return bool(sep) and a.isdigit() and b.isdigit()


def _valid_entry(e: Any) -> bool:
    if not isinstance(e, dict):
        return False
    fields = (e.get("name"), e.get("note"), e.get("by"), e.get("at"))
    # who wrote the name and who wrote the note, kept apart (an entry from before has only `by`, its last writer)
    authors = [e.get(f, e.get("by")) for f in ("name_by", "note_by")]
    return all(isinstance(t, str) and not _has_broken(t) for t in fields) and e.get("by") in WRITERS and all(a in WRITERS for a in authors)


# what each file is called in its warning, and what starts empty when it is moved aside
_WHAT = {"notes": ("field notes", "the notes start"), "book": ("field book", "the field book starts"),
         "proposals": ("scientist's proposals", "the proposals start")}


def _move_aside(path: Path, why: str, which: str = "notes") -> None:
    """Move an unreadable file out of the way (never written over) and leave a warning for the next reply."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    aside = path.with_name(f"{path.name}.corrupt-{stamp}")
    n = 1
    while aside.exists():
        aside = path.with_name(f"{path.name}.corrupt-{stamp}-{n}")
        n += 1
    os.replace(path, aside)
    what, starts = _WHAT[which]
    logger.warning("The %s file %s could not be read (%s); moved aside to %s", what, path, why, aside)
    _pending[which] = (
        f"The {what} file could not be read ({why}), so it was moved aside to {aside} and {starts} empty. "
        "Nothing in it was lost: the old file is still there."
    )


def _read_json(path: Path, which: str, of_what: str) -> Optional[Dict[str, Any]]:
    """A file's JSON object, of version 1; None when there is no file yet, or when it cannot be read as one (it is then
    moved aside, with its warning)."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    try:
        data = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        _move_aside(path, f"not UTF-8 text: {exc.reason} at byte {exc.start}", which)
        return None
    except ValueError as exc:
        _move_aside(path, f"not JSON: {exc}", which)
        return None
    except RecursionError:
        _move_aside(path, "nested too deeply to read", which)
        return None
    if not isinstance(data, dict) or data.get("version") != FILE_VERSION:
        _move_aside(path, f"not a version {FILE_VERSION} file of {of_what}", which)
        return None
    return data


def read_notes(path: Optional[Path] = None) -> Dict[str, Any]:
    """The notes as stored; empty when there is no file yet. An unreadable file is moved aside."""
    path = path or field_notes_path()
    data = _read_json(path, "notes", "notes")
    if data is None:
        return _empty()
    out = _empty()
    for key in KINDS.values():
        entries = data.get(key, {})
        if not isinstance(entries, dict):
            _move_aside(path, f"its {key} are not a table")
            return _empty()
        if not all(_valid_key(k) and _valid_entry(e) for k, e in entries.items()):
            _move_aside(path, f"a malformed entry in its {key}")
            return _empty()
        out[key] = {k: {**{f: e[f] for f in ("name", "note", "by", "at")}, "name_by": e.get("name_by", e["by"]),
                        "note_by": e.get("note_by", e["by"])} for k, e in entries.items()}
    return out


def take_warning(which: str = "notes") -> Optional[str]:
    """A moved-aside file's warning, once: the notes' by default, or the field book's ("book")."""
    return _pending.pop(which, None)


BROKEN = "holds a broken character (half of an emoji, say), which cannot be kept"


def check_note(kind: Any, name: Any, note: Any, by: Any) -> Tuple[str, Optional[str], Optional[str], str]:
    """(kind, name, note, by) cleaned, or a ValueError saying in plain words what is wrong. The name is trimmed and
    must be one line; the note is kept as written. A name or note left out (None) is kept as it is (`write_note`)."""
    if kind not in KINDS:
        raise ValueError('kind must be "neuron" or "head".')
    name = None if name is None else str(name).strip()
    note = None if note is None else str(note)
    by = "reader" if by is None else str(by)
    if name is None and note is None:
        raise ValueError("Give a name, a note, or both (empty strings clear them).")
    if name is not None:
        if len(name) > MAX_NAME:
            raise ValueError(f"A name can hold at most {MAX_NAME} characters.")
        if _has_control(name):
            raise ValueError("A name cannot hold control characters (a new line or tab, say).")
        if _has_broken(name):
            raise ValueError(f"The name {BROKEN}.")
    if note is not None:
        if len(note) > MAX_NOTE:
            raise ValueError(f"A note can hold at most {MAX_NOTE} characters.")
        if _has_broken(note):
            raise ValueError(f"The note {BROKEN}.")
    if by not in WRITERS:
        raise ValueError(f'by must be one of {", ".join(WRITERS)}.')
    return kind, name, note, by


def _put_note(data: Dict[str, Any], kind: str, key: str, name: Optional[str], note: Optional[str], by: str) -> Optional[Dict[str, Any]]:
    """One entry changed in `data` (the notes as read): its name or note left out (None) kept as it was; an entry whose
    name and note are both empty is forgotten. Returns the entry, or None."""
    table = KINDS[kind]
    entries: Dict[str, Any] = data[table]
    old = entries.get(key) or {}
    # each field keeps its own author: a note added to a name keeps the name's (a scientist's guess stays its own)
    name_by = by if name is not None else old.get("name_by", old.get("by", by))
    note_by = by if note is not None else old.get("note_by", old.get("by", by))
    name = old.get("name", "") if name is None else name
    note = old.get("note", "") if note is None else note
    if not name and not note:
        entries.pop(key, None)
        return None
    if key not in entries and len(entries) >= MAX_ENTRIES:
        raise ValueError(f"The notes on {table} are full ({MAX_ENTRIES}); forget one to keep another.")
    entry = {"name": name, "note": note, "by": by, "name_by": name_by, "note_by": note_by, "at": _utc_now()}
    entries[key] = entry
    return entry


async def write_note(
    kind: str, layer: int, index: int, name: Optional[str], note: Optional[str], by: str, path: Optional[Path] = None
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Keep one entry, its name or note left out (None) kept as it was; an entry whose name and note are both empty is
    forgotten. Returns (the entry or None, all the notes). The arguments are those `check_note` returns; the layer and
    index are checked by the caller against the model."""
    path = path or field_notes_path()
    async with _LOCK:
        data = read_notes(path)
        entry = _put_note(data, kind, note_key(layer, index), name, note, by)
        _write_atomic(path, data)
        _CACHE.pop(str(path), None)
        return entry, data


async def all_notes(path: Optional[Path] = None) -> Dict[str, Any]:
    """Every note, read again only when the file has changed (by its mtime and size). The reply is shared between
    callers: read it, never change it."""
    path = path or field_notes_path()
    async with _LOCK:
        try:
            st = path.stat()
        except FileNotFoundError:
            _CACHE.pop(str(path), None)
            return _empty()
        stamp = (st.st_mtime_ns, st.st_size)
        hit = _CACHE.get(str(path))
        if hit is not None and hit[0] == stamp:
            return hit[1]
        data = read_notes(path)
        if path.exists():  # not moved aside
            _CACHE[str(path)] = (stamp, data)
        return data


# ------------------------------------------------------------------------------------------- the folio's field book


def blank_face() -> Dict[str, Any]:
    return {"jobs": None, "pattern": "", "checked": {t: False for t in TICKS}}


def _valid_face(e: Any) -> bool:
    if not isinstance(e, dict):
        return False
    ck = e.get("checked")
    return (e.get("jobs") is None or e.get("jobs") in JOBS) and isinstance(e.get("pattern"), str) and not _has_broken(e["pattern"]) \
        and isinstance(ck, dict) and all(isinstance(ck.get(t), bool) for t in TICKS) and e.get("by") in WRITERS \
        and isinstance(e.get("at"), str)


def read_book(path: Optional[Path] = None) -> Dict[str, Any]:
    """The field book as stored ({"version", "neurons"}); empty when there is no file yet. An unreadable file is moved
    aside. Each face is rebuilt from the fields this version knows."""
    path = path or field_book_path()
    empty = {"version": FILE_VERSION, "neurons": {}}
    data = _read_json(path, "book", "the field book")
    if data is None:
        return empty
    entries = data.get("neurons", {})
    if not isinstance(entries, dict):
        _move_aside(path, "its neurons are not a table", "book")
        return empty
    ok = all(_valid_key(k) and isinstance(e, dict) and set(e) <= set(FACES) and all(_valid_face(f) for f in e.values())
             for k, e in entries.items())
    if not ok:
        _move_aside(path, "a malformed entry in its neurons", "book")
        return empty
    return {"version": FILE_VERSION, "neurons": {
        k: {s: {"jobs": f["jobs"], "pattern": f["pattern"], "checked": {t: f["checked"][t] for t in TICKS}, "by": f["by"], "at": f["at"]}
            for s, f in e.items()} for k, e in entries.items()}}


def check_face(face: Any, jobs: Any, pattern: Any, checked: Any, by: Any, given: Collection[str]) -> Dict[str, Any]:
    """What to change in one face of the field book ({"face", "by", and of jobs, pattern, checked those `given`}), or a
    ValueError saying in plain words what is wrong. A field not given is kept as it is; `checked` may give some ticks
    only, the others kept."""
    if face not in FACES:
        raise ValueError('A face is "+" or "-".')
    by = "reader" if by is None else str(by)
    if by not in WRITERS:
        raise ValueError(f'by must be one of {", ".join(WRITERS)}.')
    if not set(given) & {"jobs", "pattern", "checked"}:
        raise ValueError("Give the face's jobs, its pattern or its ticks (checked), or any of them.")
    out: Dict[str, Any] = {"face": face, "by": by}
    if "jobs" in given:
        if jobs is not None and jobs not in JOBS:
            raise ValueError('jobs is "one", "two", "unknown" (not yet understood) or null (not yet said), not '
                             f"{json.dumps(jobs, ensure_ascii=False)[:40]}.")
        out["jobs"] = jobs
    if "pattern" in given:
        if not isinstance(pattern, str):
            raise ValueError("A pattern is text: the face's pattern in a sentence or two.")
        if len(pattern) > MAX_PATTERN:
            raise ValueError(f"A pattern can hold at most {MAX_PATTERN} characters.")
        if _has_broken(pattern):
            raise ValueError(f"The pattern {BROKEN}.")
        out["pattern"] = pattern
    if "checked" in given:
        if not isinstance(checked, dict):
            raise ValueError(f"checked is a table of ticks: {', '.join(TICKS)}, each true or false.")
        unknown = [k for k in checked if k not in TICKS]
        if unknown:
            raise ValueError(f"checked takes {', '.join(TICKS)}; there is no tick {unknown[0]!r}.")
        bad = [k for k, v in checked.items() if not isinstance(v, bool)]
        if bad:
            raise ValueError(f"A tick is true or false: {bad[0]} is {json.dumps(checked[bad[0]], ensure_ascii=False)[:40]}.")
        out["checked"] = dict(checked)
    return out


async def write_face(layer: int, n: int, change: Dict[str, Any], path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Keep one face of a neuron's field book (`change` as `check_face` returns it); a face left with nothing said (no
    jobs, no pattern, no tick) is forgotten. Returns the face as kept, or None. The layer and neuron are checked by the
    caller against the model."""
    path = path or field_book_path()
    key = note_key(layer, n)
    face = change["face"]
    async with _LOCK:
        data = read_book(path)
        entries: Dict[str, Any] = data["neurons"]
        old = (entries.get(key) or {}).get(face) or blank_face()
        new = {"jobs": change.get("jobs", old["jobs"]), "pattern": change.get("pattern", old["pattern"]),
               "checked": {t: bool(change.get("checked", {}).get(t, old["checked"][t])) for t in TICKS}}
        faces = dict(entries.get(key) or {})
        if new["jobs"] is None and not new["pattern"] and not any(new["checked"].values()):
            faces.pop(face, None)
            entry = None
        else:
            if key not in entries and len(entries) >= MAX_ENTRIES:
                raise ValueError(f"The field book is full ({MAX_ENTRIES} neurons); forget one to keep another.")
            entry = {**new, "by": change["by"], "at": _utc_now()}
            faces[face] = entry
        if faces:
            entries[key] = {s: faces[s] for s in FACES if s in faces}
        else:
            entries.pop(key, None)
        _write_atomic(path, data)
        return entry


async def all_book(path: Optional[Path] = None) -> Dict[str, Any]:
    """The whole field book, read under the module's lock."""
    async with _LOCK:
        return read_book(path or field_book_path())


# ---------------------------------------------------------------------------------------- the scientist's proposals


def _valid_proposal(e: Any) -> bool:
    if not isinstance(e, dict) or not set(e) <= {"name", "note", "chat", "at"} or not ("name" in e or "note" in e):
        return False
    texts = [e[f] for f in ("name", "note", "chat", "at") if f in e]
    return all(isinstance(t, str) and not _has_broken(t) for t in texts) and isinstance(e.get("chat"), str) and isinstance(e.get("at"), str)


def read_proposals(path: Optional[Path] = None) -> Dict[str, Any]:
    """The proposals waiting for the person; empty when there is no file yet. An unreadable file is moved aside."""
    path = path or field_proposals_path()
    data = _read_json(path, "proposals", "proposals")
    if data is None:
        return _empty()
    out = _empty()
    for key in KINDS.values():
        entries = data.get(key, {})
        if not isinstance(entries, dict) or not all(_valid_key(k) and _valid_proposal(e) for k, e in entries.items()):
            _move_aside(path, f"a malformed entry in its {key}", "proposals")
            return _empty()
        out[key] = {k: {f: e[f] for f in ("name", "note", "chat", "at") if f in e} for k, e in entries.items()}
    return out


async def propose(kind: str, layer: int, index: int, name: Optional[str], note: Optional[str], chat: str,
                  path: Optional[Path] = None) -> Dict[str, Any]:
    """Keep a proposal of the scientist's (its name or note as `check_note` cleaned them, None where it proposes
    neither): a later one replaces the earlier one's name or note, and leaves the other. Nothing in the notes changes.
    Returns the proposal as it waits."""
    path = path or field_proposals_path()
    chat = str(chat)
    if _has_broken(chat):
        raise ValueError(f"The chat's id {BROKEN}.")
    if len(chat) > MAX_CHAT:
        raise ValueError(f"The chat's id is at most {MAX_CHAT} characters.")
    key = note_key(layer, index)
    table = KINDS[kind]
    async with _LOCK:
        data = read_proposals(path)
        entries: Dict[str, Any] = data[table]
        if key not in entries and sum(len(v) for v in (data["neurons"], data["heads"])) >= MAX_PROPOSALS:
            raise ValueError(f"{MAX_PROPOSALS} proposals already wait for the person; ask them to keep or discard some first.")
        e = dict(entries.get(key) or {})
        if name is not None:
            e["name"] = name
        if note is not None:
            e["note"] = note
        e.update({"chat": chat, "at": _utc_now()})
        entries[key] = e
        _write_atomic(path, data)
        return e


async def decide(kind: str, layer: int, index: int, keep: bool, part: Optional[str] = None,
                 path: Optional[Path] = None, notes_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """The person's answer to a proposal: kept, its name or note (`part`; both when None) is written to the notes as the
    scientist's (`name_by`, `note_by` "scientist"), as POST /notes would write it; kept or discarded, that part leaves
    the proposals. None when no such proposal waits. Returns {"parts", "entry" (the notes' entry, when kept),
    "proposal" (what still waits, or None)}."""
    path = path or field_proposals_path()
    notes_path = notes_path or field_notes_path()
    key = note_key(layer, index)
    table = KINDS[kind]
    async with _LOCK:
        data = read_proposals(path)
        e = data[table].get(key)
        if e is None:
            return None
        parts = [p for p in ("name", "note") if p in e and (part is None or part == p)]
        if not parts:
            return None
        entry = None
        if keep:
            notes = read_notes(notes_path)
            entry = _put_note(notes, kind, key, e["name"] if "name" in parts else None, e["note"] if "note" in parts else None, "scientist")
            _write_atomic(notes_path, notes)
            _CACHE.pop(str(notes_path), None)
        rest = {f: v for f, v in e.items() if f not in parts}
        if "name" in rest or "note" in rest:
            data[table][key] = rest
        else:
            data[table].pop(key, None)
            rest = None
        _write_atomic(path, data)
        return {"parts": parts, "entry": entry, "proposal": rest}


async def all_proposals(path: Optional[Path] = None) -> Dict[str, Any]:
    """Every proposal waiting for the person, read under the module's lock."""
    async with _LOCK:
        return read_proposals(path or field_proposals_path())

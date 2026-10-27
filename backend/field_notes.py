"""Field notes (`backend/field_notes.py`): the names and notes given to neurons and attention heads, kept by the server.

The atlas (`#/atlas`, backend/atlas.py) and its scientist chat name what they study, as the neuron explorer this page
follows keeps its names (lyramakesmusic/neuron-explorer, `explorer/names.json`). A neuron's folio keeps its own notes
per viewer in the browser (frontend/src/screens/NeuronFolio/saved.ts); these are shared by every page and every chat
this server answers. One JSON file holds them, outside every checkout so each one the server is started from shares it:
`$THEODOLITE_FIELD_NOTES`, else `${XDG_DATA_HOME:-~/.local/share}/theodolite/field_notes.json`. Its shape:

    {"version": 1,
     "neurons": {"<layer>:<neuron>": {"name", "note", "by", "at"}},
     "heads":   {"<layer>:<head>":   {"name", "note", "by", "at"}}}

Layers count from 0 in the keys and in every field, as every field the server sends does (page layer = layer + 1).
`by` is who wrote the entry last ("reader" | "scientist"), `at` when, ISO-8601 UTC. The routes are the atlas's
(`GET` and `POST /api/atlas/notes`, backend/atlas.py, which answer only pages and scripts on this machine).

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
from typing import Any, Dict, Optional, Tuple

from backend.saved_prompts import _has_broken, _has_control, _write_atomic

logger = logging.getLogger("theodolite.field_notes")

FIELD_NOTES_ENV = "THEODOLITE_FIELD_NOTES"
FILE_VERSION = 1
MAX_NAME = 120
MAX_NOTE = 4000
MAX_ENTRIES = 20000
KINDS = {"neuron": "neurons", "head": "heads"}
WRITERS = ("reader", "scientist")

_LOCK = asyncio.Lock()
_pending_warning: Optional[str] = None
# the notes as last read, per file, with the file's (mtime, size) then: every hover of the atlas reads them
_CACHE: Dict[str, Tuple[Tuple[int, int], Dict[str, Any]]] = {}


def field_notes_path() -> Path:
    """Where the notes live, read on every request (so the environment decides, also in the checks)."""
    env = os.environ.get(FIELD_NOTES_ENV)
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base).expanduser() / "theodolite" / "field_notes.json"


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
    return all(isinstance(t, str) and not _has_broken(t) for t in fields) and e.get("by") in WRITERS


def _move_aside(path: Path, why: str) -> None:
    """Move an unreadable file out of the way (never written over) and leave a warning for the next reply."""
    global _pending_warning
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    aside = path.with_name(f"{path.name}.corrupt-{stamp}")
    n = 1
    while aside.exists():
        aside = path.with_name(f"{path.name}.corrupt-{stamp}-{n}")
        n += 1
    os.replace(path, aside)
    logger.warning("Field notes file %s could not be read (%s); moved aside to %s", path, why, aside)
    _pending_warning = (
        f"The field notes file could not be read ({why}), so it was moved aside to {aside} and the notes start empty. "
        "Nothing in it was lost: the old file is still there."
    )


def read_notes(path: Optional[Path] = None) -> Dict[str, Any]:
    """The notes as stored; empty when there is no file yet. An unreadable file is moved aside."""
    path = path or field_notes_path()
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return _empty()
    try:
        data = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        _move_aside(path, f"not UTF-8 text: {exc.reason} at byte {exc.start}")
        return _empty()
    except ValueError as exc:
        _move_aside(path, f"not JSON: {exc}")
        return _empty()
    except RecursionError:
        _move_aside(path, "nested too deeply to read")
        return _empty()
    if not isinstance(data, dict) or data.get("version") != FILE_VERSION:
        _move_aside(path, f"not a version {FILE_VERSION} file of notes")
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
        out[key] = {k: {f: e[f] for f in ("name", "note", "by", "at")} for k, e in entries.items()}
    return out


def take_warning() -> Optional[str]:
    """The moved-aside file's warning, once."""
    global _pending_warning
    warning, _pending_warning = _pending_warning, None
    return warning


BROKEN = "holds a broken character (half of an emoji, say), which cannot be kept"


def check_note(kind: Any, name: Any, note: Any, by: Any) -> Tuple[str, str, str, str]:
    """(kind, name, note, by) cleaned, or a ValueError saying in plain words what is wrong. The name is trimmed and
    must be one line; the note is kept as written."""
    if kind not in KINDS:
        raise ValueError('kind must be "neuron" or "head".')
    name = "" if name is None else str(name).strip()
    note = "" if note is None else str(note)
    by = "reader" if by is None else str(by)
    if len(name) > MAX_NAME:
        raise ValueError(f"A name can hold at most {MAX_NAME} characters.")
    if _has_control(name):
        raise ValueError("A name cannot hold control characters (a new line or tab, say).")
    if len(note) > MAX_NOTE:
        raise ValueError(f"A note can hold at most {MAX_NOTE} characters.")
    if _has_broken(name):
        raise ValueError(f"The name {BROKEN}.")
    if _has_broken(note):
        raise ValueError(f"The note {BROKEN}.")
    if by not in WRITERS:
        raise ValueError(f'by must be one of {", ".join(WRITERS)}.')
    return kind, name, note, by


async def write_note(
    kind: str, layer: int, index: int, name: str, note: str, by: str, path: Optional[Path] = None
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Keep (or, with an empty name and note, forget) one entry; returns (the entry or None, all the notes). The
    arguments are those `check_note` returns; the layer and index are checked by the caller against the model."""
    path = path or field_notes_path()
    key = note_key(layer, index)
    table = KINDS[kind]
    async with _LOCK:
        data = read_notes(path)
        entries: Dict[str, Any] = data[table]
        if not name and not note:
            entries.pop(key, None)
            entry = None
        else:
            if key not in entries and len(entries) >= MAX_ENTRIES:
                raise ValueError(f"The notes on {table} are full ({MAX_ENTRIES}); forget one to keep another.")
            entry = {"name": name, "note": note, "by": by, "at": _utc_now()}
            entries[key] = entry
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

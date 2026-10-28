"""Saved questions and system prompts (`backend/saved_prompts.py`): the instrument's shelf, kept by its server.

One JSON file holds both lists, outside every checkout so each one the server is started from shares it:
`$THEODOLITE_SAVED_PROMPTS`, else `${XDG_DATA_HOME:-~/.local/share}/theodolite/saved_prompts.json`. Its shape:

    {"version": 1,
     "questions": [{"id", "name", "question", "answer_begins", "chat", "saved_at"}],
     "systems":   [{"id", "name", "text", "saved_at"}]}

Routes (an `APIRouter`, included by backend/server.py above its static mount):
- `GET /api/saved_prompts` -> {questions, systems, file, warning}
- `POST /api/saved_prompts` {kind: "question" | "system", name, question?, answer_begins?, chat?, text?, replace_id?}
  -> the whole list (as GET). A name already used in that list (trimmed, any case) by another entry than
  `replace_id` -> 409.
- `DELETE /api/saved_prompts/{kind}/{id}` -> the whole list; 404 when there is no such entry.

The text is kept exactly as given (only the name is trimmed); ids are made here; `saved_at` is ISO-8601 UTC. Writes
take this module's own lock (never the GPU's: these routes answer while a sweep runs) and replace the file atomically.
A file that cannot be read is moved aside to `<file>.corrupt-<UTC stamp>`, never written over, and the next reply says
so once (`warning`); so is a file that is not UTF-8 text or is nested too deeply to read. The server refuses a page
served anywhere else before any route (backend/local_only.py), and these routes keep a rule of their own as well, so
they refuse a request from a web page served anywhere but this machine (403) in any app they are part of: its Host must
name this machine (a page at another name that resolves here, DNS rebinding, sends its own name as the Host even on a
same-origin GET, which carries no Origin), and so must its Origin when it sends one. A script sent to this machine
(no Origin; curl, the checks) is allowed. A text holding half of a character (a lone surrogate) is refused with 422:
it cannot be written as UTF-8.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger("theodolite.saved_prompts")

SAVED_PROMPTS_ENV = "THEODOLITE_SAVED_PROMPTS"
FILE_VERSION = 1
MAX_NAME = 80
MAX_TEXT = 32000
MAX_ANSWER_BEGINS = 4000
MAX_ENTRIES = 500
KINDS = {"question": "questions", "system": "systems"}
# the hosts a page of the instrument itself is served from (any port: the server, vite's dev and preview servers)
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}

_LOCK = asyncio.Lock()
# a moved-aside file, reported once in the next reply
_pending_warning: Optional[str] = None


def saved_prompts_path() -> Path:
    """Where the lists live, read on every request (so the environment decides, also in tests)."""
    env = os.environ.get(SAVED_PROMPTS_ENV)
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base).expanduser() / "theodolite" / "saved_prompts.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _hostname(url: str) -> Optional[str]:
    """The host a URL (or a bare `host[:port]`, given as `//host:port`) names; None when it names none."""
    try:
        return urlsplit(url).hostname
    except ValueError:
        return None


def same_machine(request: Request) -> None:
    """Refuse a request sent by a page served from anywhere but this machine (403): CORS alone does not stop a request
    being sent, so without this (or the server's own guard, backend/local_only.py) a site the reader visits could change
    the lists. The Host is checked first: a same-origin GET sends no Origin, so a page at another name that resolves to
    this machine (DNS rebinding) is known only by the Host its browser sends. Vite's proxy sends the server's own Host
    (changeOrigin). A script sent to this machine (no Origin, a Host of 127.0.0.1, localhost or [::1], or none at all)
    is allowed."""
    host_header = request.headers.get("host")
    origin = request.headers.get("origin")
    refused = (host_header is not None and _hostname("//" + host_header) not in LOCAL_HOSTS) or (
        origin is not None and _hostname(origin) not in LOCAL_HOSTS
    )
    if refused:
        raise HTTPException(
            status_code=403,
            detail="Saved prompts answer only the instrument's own pages, served from this machine.",
        )


router = APIRouter(prefix="/api/saved_prompts", dependencies=[Depends(same_machine)])


class SavePromptRequest(BaseModel):
    # types only: every limit is checked by `_check_entry`, so a refusal says what was wrong in plain words
    kind: Optional[str] = None
    name: Optional[str] = None
    question: Optional[str] = None
    answer_begins: Optional[str] = None
    chat: Optional[bool] = None
    text: Optional[str] = None
    replace_id: Optional[str] = None


def _empty() -> Dict[str, Any]:
    return {"version": FILE_VERSION, "questions": [], "systems": []}


def _valid_entry(kind: str, e: Any) -> bool:
    if not isinstance(e, dict) or not isinstance(e.get("id"), str) or not isinstance(e.get("name"), str):
        return False
    if not isinstance(e.get("saved_at"), str):
        return False
    if kind == "question":
        texts = (e.get("question"), e.get("answer_begins"))
        if not isinstance(e.get("chat"), bool):
            return False
    else:
        texts = (e.get("text"),)
    # a lone surrogate (a hand-written \ud83d escape) could be neither sent nor written back as UTF-8, so every
    # reply would fail: such a file is moved aside like any other malformed entry
    return all(isinstance(t, str) and not _has_broken(t) for t in (e["id"], e["name"], e["saved_at"], *texts))


def _clean_entry(kind: str, e: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("id", "name", "question", "answer_begins", "chat", "saved_at") if kind == "question" else (
        "id", "name", "text", "saved_at"
    )
    return {k: e[k] for k in keys}


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
    logger.warning("Saved prompts file %s could not be read (%s); moved aside to %s", path, why, aside)
    _pending_warning = (
        f"The saved prompts file could not be read ({why}), so it was moved aside to {aside} and the lists start "
        "empty. Nothing in it was lost: the old file is still there."
    )


def _read(path: Path) -> Dict[str, Any]:
    """The lists as stored; empty when there is no file yet. An unreadable file is moved aside."""
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
        _move_aside(path, f"not a version {FILE_VERSION} list")
        return _empty()
    out = _empty()
    for kind, key in KINDS.items():
        entries = data.get(key, [])
        if not isinstance(entries, list):
            _move_aside(path, f"its {key} are not a list")
            return _empty()
        if not all(_valid_entry(kind, e) for e in entries):
            _move_aside(path, f"a malformed entry in its {key}")
            return _empty()
        out[key] = [_clean_entry(kind, e) for e in entries]
    return out


def _write_atomic(path: Path, data: Dict[str, Any]) -> None:
    """Replace the file in one step: a temporary file in the same directory, flushed and synced, renamed over it,
    then the directory synced, so a crash leaves the old list or the new one, never half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    dir_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _reply(path: Path, data: Dict[str, Any]) -> Dict[str, Any]:
    global _pending_warning
    warning, _pending_warning = _pending_warning, None
    return {"questions": data["questions"], "systems": data["systems"], "file": str(path), "warning": warning}


def _has_control(s: str) -> bool:
    return any(unicodedata.category(ch) == "Cc" for ch in s)


def _has_broken(s: str) -> bool:
    """Half of a character: a lone surrogate (category Cs), which JSON can carry but UTF-8 cannot."""
    return any(unicodedata.category(ch) == "Cs" for ch in s)


BROKEN = "holds a broken character (half of an emoji, say), which cannot be kept"


def _check_entry(body: SavePromptRequest) -> Tuple[str, Dict[str, Any]]:
    """The entry to keep (without id and date), or a 422 saying what is wrong."""
    if body.kind not in KINDS:
        raise HTTPException(status_code=422, detail='kind must be "question" or "system".')
    name = (body.name or "").strip()
    if not 1 <= len(name) <= MAX_NAME:
        raise HTTPException(status_code=422, detail=f"A name needs 1 to {MAX_NAME} characters.")
    if _has_control(name):
        raise HTTPException(status_code=422, detail="A name cannot hold control characters (a new line or tab, say).")
    if _has_broken(name):
        raise HTTPException(status_code=422, detail=f"The name {BROKEN}.")
    if body.kind == "question":
        question = body.question
        if question is None or not 1 <= len(question) <= MAX_TEXT:
            raise HTTPException(status_code=422, detail=f"A saved question needs 1 to {MAX_TEXT} characters of text.")
        answer_begins = body.answer_begins or ""
        if len(answer_begins) > MAX_ANSWER_BEGINS:
            raise HTTPException(
                status_code=422, detail=f"A begun answer can hold at most {MAX_ANSWER_BEGINS} characters."
            )
        if _has_broken(question):
            raise HTTPException(status_code=422, detail=f"The question {BROKEN}.")
        if _has_broken(answer_begins):
            raise HTTPException(status_code=422, detail=f"The begun answer {BROKEN}.")
        return "question", {"name": name, "question": question, "answer_begins": answer_begins, "chat": bool(body.chat)}
    text = body.text
    if text is None or not 1 <= len(text) <= MAX_TEXT:
        raise HTTPException(status_code=422, detail=f"A saved system prompt needs 1 to {MAX_TEXT} characters of text.")
    if _has_broken(text):
        raise HTTPException(status_code=422, detail=f"The system prompt {BROKEN}.")
    return "system", {"name": name, "text": text}


@router.get("")
async def get_saved_prompts() -> Dict[str, Any]:
    path = saved_prompts_path()
    async with _LOCK:
        return _reply(path, _read(path))


@router.post("")
async def save_prompt(body: SavePromptRequest) -> Dict[str, Any]:
    kind, fields = _check_entry(body)
    key = KINDS[kind]
    label = "question" if kind == "question" else "system prompt"
    path = saved_prompts_path()
    async with _LOCK:
        data = _read(path)
        entries: List[Dict[str, Any]] = data[key]
        folded = fields["name"].casefold()
        for e in entries:
            if e["name"].strip().casefold() == folded and e["id"] != body.replace_id:
                raise HTTPException(
                    status_code=409, detail=f"There is already a saved {label} named “{e['name']}”."
                )
        if body.replace_id is not None:
            at = next((i for i, e in enumerate(entries) if e["id"] == body.replace_id), None)
            if at is None:
                raise HTTPException(status_code=404, detail=f"There is no saved {label} to replace with that id.")
            entries[at] = {"id": entries[at]["id"], **fields, "saved_at": _utc_now()}
        else:
            if len(entries) >= MAX_ENTRIES:
                raise HTTPException(
                    status_code=400,
                    detail=f"The list of saved {label}s is full ({MAX_ENTRIES}); forget one to keep another.",
                )
            entries.append({"id": secrets.token_hex(8), **fields, "saved_at": _utc_now()})
        _write_atomic(path, data)
        return _reply(path, data)


@router.delete("/{kind}/{entry_id}")
async def forget_prompt(kind: str, entry_id: str) -> Dict[str, Any]:
    if kind not in KINDS:
        raise HTTPException(status_code=404, detail=f'There is no list called "{kind}": "question" or "system".')
    key = KINDS[kind]
    path = saved_prompts_path()
    async with _LOCK:
        data = _read(path)
        entries = data[key]
        kept = [e for e in entries if e["id"] != entry_id]
        if len(kept) == len(entries):
            raise HTTPException(status_code=404, detail="There is no saved entry with that id.")
        data[key] = kept
        _write_atomic(path, data)
        return _reply(path, data)

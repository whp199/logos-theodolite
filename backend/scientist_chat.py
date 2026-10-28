"""The atlas's scientist chat (`backend/scientist_chat.py`): a scientist beside the map that reads the atlas, runs the
model and points at things on the person's screen.

After the explorer's chat (lyramakesmusic/neuron-explorer, explorer/server.py, "Claude: autointerp and chat"), with
three backends; a chat keeps the one it starts with:
- "claude": Claude Code (`claude -p`, stream-json in and out), one process per chat kept open, so a message sent while
  it works reaches it between tool calls and an interrupt stops the reply without ending the process. Its tools are
  the atlas's (backend/scientist_mcp.py, over MCP) and its own file tools, within its reach (`reach`). The conversation
  is Claude Code's own session (`~/.claude/projects/.../<chat>.jsonl`), resumed turn by turn.
- "codex": Codex (`codex app-server`, JSON-RPC), the same MCP tools and Codex's own shell, within its reach, in a
  Codex home of its own (the person's login, model and effort, nothing else).

Reach (`$THEODOLITE_SCIENTIST_REACH`): "folder", the default, keeps Claude Code to reading, writing and editing files in
its work folder, with no shell, and runs Codex's shell in its workspace-write sandbox (writes only in the work folder,
no network). "home" is the explorer's own setting, taken only when the person sets it: Claude Code's Read, Glob, Grep,
Write, Edit and Bash pre-approved over the home folder, Codex with full access and never asking. Either way the
person's own Claude Code settings are not loaded, neither process is given the endpoints' keys nor the Hugging Face and
cloud credentials in the server's environment (`child_env`; each keeps its own login), and the page
says what the chosen backend can do (`REACH_WORDS`), asking once before the first chat when it is "home". What a
scientist reads (model output, census answers, names and notes, attachments) can carry instructions; with a shell
over the home folder nothing stands between them and it.
- "openai": any OpenAI-compatible Chat Completions endpoint (LM Studio, Ollama, OpenRouter, vLLM, llama.cpp's server)
  that supports tool calling. The endpoint runs no tools, so the server runs the loop: it streams the model's reply,
  runs each tool call through the atlas's own routes in this process, sends the results back, and goes on until the
  model answers without a call (or `MAX_ROUNDS`). Such a model has the atlas's tools only: no shell and no files.

Endpoints ("profiles", `profiles()`): lmstudio (http://127.0.0.1:1234/v1), ollama (http://127.0.0.1:11434/v1),
openrouter (https://openrouter.ai/api/v1, its key in $OPENROUTER_API_KEY), custom ($THEODOLITE_OPENAI_BASE_URL with
$THEODOLITE_OPENAI_API_KEY and $THEODOLITE_OPENAI_MODEL), and any named in `<scientist folder>/endpoints.json`. A
profile names the environment variable its key is in, never the key: the key is read when a request is made and is
never sent to a page. A page picks a profile by name and a model; it cannot point a key at another address.

Every route answers only the instrument's own pages (`atlas.local_pages_only`); a chat's process works in the
scientist folder (`$THEODOLITE_SCIENTIST_DIR`, else `<atlas folder>/scientist`: `work/` is its working folder, beside
`uploads/`, `logs/` (the codex and openai chats' events and messages), `chats.json` and the system prompt built from
backend/scientist_prompt.md, backend/scientist_methodology.md and `prompt.local.md` there). Events reach the page as
NDJSON (GET /events): state, sent, text, thinking, tool, tool_result, ui, error, done, ping. The routes are in
ARCHITECTURE.md §4.8.
"""

from __future__ import annotations

import asyncio
import base64
import glob
import json
import logging
import os
import re
import shutil
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend import scientist_tools as st
from backend.atlas import local_pages_only
from backend.saved_prompts import LOCAL_HOSTS

logger = logging.getLogger("theodolite.scientist")

ROOT_DIR = Path(__file__).resolve().parent.parent
PROMPT_FILES = (ROOT_DIR / "backend" / "scientist_prompt.md", ROOT_DIR / "backend" / "scientist_methodology.md")
MCP_SCRIPT = ROOT_DIR / "backend" / "scientist_mcp.py"
SCIENTIST_DIR_ENV = "THEODOLITE_SCIENTIST_DIR"
AGENTS = ("claude", "codex", "openai")
FILE_TOOLS = ["Read", "Glob", "Grep", "Write", "Edit", "Bash"]
REACH_ENV = "THEODOLITE_SCIENTIST_REACH"
# what each backend can do beyond the atlas's tools, said on the page
REACH_WORDS = {
    "folder": {"claude": "reads, writes and edits files only in its own work folder, and has no shell",
               "codex": "runs commands in a sandbox that writes only in its own work folder and has no network; it can read files elsewhere",
               "openai": "has the atlas's tools and nothing else: no shell, no files"},
    "home": {"claude": "has a shell, and reads, changes and runs anything in your home folder without asking",
             "codex": "has a shell with full access to this machine, and never asks before it acts",
             "openai": "has the atlas's tools and nothing else: no shell, no files"},
}
MCP_PREFIX = "mcp__atlas__"
CHAT_IDLE = 30 * 60  # seconds without listeners or turns before a chat's input is closed (it exits once its work is done)
MAX_ROUNDS = 40  # tool rounds in one openai turn
TOOL_CUT = 16000  # characters of a tool's reply an openai-compatible model reads
SHOWN_CUT = 1500  # characters of a tool's reply the page shows
MAX_UPLOAD = 25 << 20
MAX_MESSAGE = 100_000

router = APIRouter(prefix="/api/atlas/scientist", dependencies=[Depends(local_pages_only)])
_app: Any = None


def bind(app: Any) -> None:
    """Called by backend/server.py: the app the openai backend's tool calls go through, in this process."""
    global _app
    _app = app


def scientist_dir() -> Path:
    env = os.environ.get(SCIENTIST_DIR_ENV)
    if env:
        return Path(env).expanduser()
    from backend.atlas_index import atlas_dir

    return atlas_dir() / "scientist"


def work_dir() -> Path:
    d = scientist_dir() / "work"
    d.mkdir(parents=True, exist_ok=True)
    return d


def uploads_dir() -> Path:
    return scientist_dir() / "uploads"


def logs_dir() -> Path:
    return scientist_dir() / "logs"


def _bin(name: str) -> str:
    return os.environ.get(f"THEODOLITE_{name.upper()}_BIN", name)


def reach() -> str:
    """How far a Claude Code or Codex scientist reaches beyond the atlas's tools (module docstring): "folder" unless the
    person sets $THEODOLITE_SCIENTIST_REACH=home."""
    return "home" if os.environ.get(REACH_ENV, "").strip().lower() == "home" else "folder"


# Credentials no scientist needs, kept from both processes: the checkpoint's host and the accounts a GPU is rented with
# (the census on a pod, say). Every AWS_* and AZURE_* variable counts, as their credentials have many names.
SECRET_ENV = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_API_TOKEN", "RUNPOD_API_KEY", "VAST_API_KEY", "LAMBDA_API_KEY",
              "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "GOOGLE_APPLICATION_CREDENTIALS", "NEURONARY_TOKEN"}
SECRET_PREFIXES = ("AWS_", "AZURE_")
# Never a CLI's own login. Claude Code's (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, CLAUDE_CODE_OAUTH_TOKEN, or its login
# under ~/.claude) and Codex's (OPENAI_API_KEY, CODEX_*, or ~/.codex/auth.json, linked into its own home) are not
# touched; and when Claude Code is told to reach its model through Bedrock, Vertex or Foundry, the cloud's variables it
# then reads are its login, and pass to it (not to Codex, whose own home names no such provider).
CLAUDE_CLOUD_LOGINS = {"CLAUDE_CODE_USE_BEDROCK": ("AWS_",), "CLAUDE_CODE_USE_VERTEX": ("GOOGLE_APPLICATION_CREDENTIALS",),
                       "CLAUDE_CODE_USE_FOUNDRY": ("AZURE_",)}


def child_env(agent: str = "claude", **extra: str) -> Dict[str, str]:
    """The environment a Claude Code (`agent` "claude") or Codex ("codex") process starts with: this one's, less every
    endpoint's key and the credentials above, but with that process's own login."""
    keys = {p.get("key_env") for p in profiles().values()} | {"THEODOLITE_OPENAI_API_KEY", "OPENROUTER_API_KEY"}
    on = lambda v: (v or "").strip().lower() not in ("", "0", "false", "no", "off")  # noqa: E731
    own = tuple(p for flag, ps in CLAUDE_CLOUD_LOGINS.items() if agent == "claude" and on(os.environ.get(flag)) for p in ps)
    secret = lambda k: (k in SECRET_ENV or k.startswith(SECRET_PREFIXES)) and not k.startswith(own)  # noqa: E731
    return {**{k: v for k, v in os.environ.items() if k not in keys and not secret(k)}, **extra}


# ------------------------------------------------------------------------------------------------------- the prompt

SHELL_START, SHELL_END = "<!-- shell -->", "<!-- /shell -->"
FOLDER_START, FOLDER_END = "<!-- folder -->", "<!-- /folder -->"
CODEX_NOTE = """

## This session

You are running as Codex. Your shell and file editing are Codex's own: where these instructions name Read, Glob, Grep,
Write, Edit or Bash, use your shell and apply_patch. Start anything that runs longer than about a minute detached
(`nohup {PYTHON} script.py > script.log 2>&1 &`), say where the log is, and end your reply; read the log in a later
turn."""
CODEX_FOLDER_NOTE = """

## This session

You are running as Codex, your shell in a sandbox: it writes only in `{WORK}` and has no network, so a script cannot
reach the instrument's API; use the atlas's tools for every reading."""
OPENAI_NOTE = """

## This session

You are running behind an OpenAI-compatible endpoint, with the atlas's tools and nothing else: no shell, no files, no
web. Where an analysis needs a script, write the script in your reply for the person to run, and say what it prints."""


def system_prompt(agent: str, self_url: str) -> str:
    """The scientist's instructions: the prompt and its methodology, then this machine's own additions
    (`<scientist folder>/prompt.local.md`), with {ROOT}, {API}, {WORK} and {PYTHON} filled in; for an openai-compatible
    model the shell section is left out, as it has none."""
    text = "\n\n".join(p.read_text(encoding="utf-8").rstrip() for p in PROMPT_FILES)
    local = scientist_dir() / "prompt.local.md"
    if local.exists():
        text += "\n\n" + local.read_text(encoding="utf-8").rstrip()
    cut = lambda t, a, b: re.sub(re.escape(a) + r".*?" + re.escape(b), "", t, flags=re.S)  # noqa: E731
    keep = lambda t, a, b: t.replace(a, "").replace(b, "")  # noqa: E731
    if agent == "openai":
        text = cut(cut(text, SHELL_START, SHELL_END), FOLDER_START, FOLDER_END) + OPENAI_NOTE
    elif reach() == "home":
        text = keep(cut(text, FOLDER_START, FOLDER_END), SHELL_START, SHELL_END)
        if agent == "codex":
            text += CODEX_NOTE
    else:
        text = keep(cut(text, SHELL_START, SHELL_END), FOLDER_START, FOLDER_END)
        if agent == "codex":
            text += CODEX_FOLDER_NOTE
    return (text.replace("{ROOT}", str(ROOT_DIR)).replace("{API}", self_url).replace("{WORK}", str(work_dir()))
            .replace("{PYTHON}", sys.executable).strip() + "\n")


def prompt_file(agent: str, self_url: str) -> Path:
    """The prompt written beside the chats (read-only: it is built from the files named in `system_prompt`)."""
    out = scientist_dir() / f"system_prompt.{agent}.md"
    text = system_prompt(agent, self_url)
    if not out.exists() or out.read_text(encoding="utf-8") != text:
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            os.chmod(out, 0o644)
        out.write_text(text, encoding="utf-8")
        os.chmod(out, 0o444)
    return out


# ------------------------------------------------------------------------------------------------------- endpoints

PROFILES: Dict[str, Dict[str, Any]] = {
    "lmstudio": {"label": "LM Studio (this machine)", "base_url": "http://127.0.0.1:1234/v1", "url_env": "THEODOLITE_LMSTUDIO_URL"},
    "ollama": {"label": "Ollama (this machine)", "base_url": "http://127.0.0.1:11434/v1", "url_env": "THEODOLITE_OLLAMA_URL"},
    "openrouter": {"label": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY",
                   "headers": {"X-Title": "Logos Theodolite", "HTTP-Referer": "https://github.com/whp199/logos-theodolite"}},
}
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_profile_warning: Optional[str] = None


def profiles() -> Dict[str, Dict[str, Any]]:
    """Every endpoint the openai backend can use, by name: the built-in ones (their addresses can be moved by
    $THEODOLITE_LMSTUDIO_URL and $THEODOLITE_OLLAMA_URL), `custom` when $THEODOLITE_OPENAI_BASE_URL is set, and those of
    `<scientist folder>/endpoints.json` ({name: {label, base_url, key_env, model, stream, headers, params}}). A file
    entry holding a key itself is refused: name the environment variable it is in."""
    global _profile_warning
    out = {k: dict(v) for k, v in PROFILES.items()}
    for v in out.values():
        if v.get("url_env") and os.environ.get(v["url_env"]):
            v["base_url"] = os.environ[v["url_env"]]
    if os.environ.get("THEODOLITE_OPENAI_BASE_URL"):
        out["custom"] = {"label": "custom endpoint", "base_url": os.environ["THEODOLITE_OPENAI_BASE_URL"],
                         "key_env": "THEODOLITE_OPENAI_API_KEY", "model": os.environ.get("THEODOLITE_OPENAI_MODEL")}
    path = scientist_dir() / "endpoints.json"
    if path.exists():
        try:
            extra = json.loads(path.read_text(encoding="utf-8"))
            for name, v in (extra.items() if isinstance(extra, dict) else []):
                why = _bad_profile(name, v)
                if why:
                    _profile_warning = f"{path}: {name}: {why}"
                    continue
                out[name] = {k: v[k] for k in ("label", "base_url", "key_env", "model", "stream", "headers", "params") if k in v}
        except (OSError, ValueError) as exc:
            _profile_warning = f"{path} could not be read: {exc}"
    return out


def _bad_profile(name: Any, v: Any) -> Optional[str]:
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,40}", name):
        return "a profile's name is lower-case letters, digits, - and _"
    if not isinstance(v, dict):
        return "a profile is a table"
    if any(k in v for k in ("api_key", "key", "token")):
        return "holds a key: name the environment variable it is in (key_env) instead"
    u = urlparse(str(v.get("base_url", "")))
    if u.scheme not in ("http", "https") or not u.hostname:
        return "base_url must be an http(s) address"
    if v.get("key_env") is not None and not ENV_NAME.match(str(v["key_env"])):
        return "key_env must be the name of an environment variable"
    if not isinstance(v.get("headers", {}), dict) or not isinstance(v.get("params", {}), dict):
        return "headers and params are tables"
    return None


def profile_card(name: str, p: Dict[str, Any]) -> Dict[str, Any]:
    """What a page may know of an endpoint: never its key."""
    host = urlparse(p["base_url"]).hostname or ""
    key_env = p.get("key_env")
    return {"name": name, "label": p.get("label", name), "base_url": p["base_url"], "this_machine": host in LOCAL_HOSTS,
            "key": ("set" if os.environ.get(key_env) else f"missing: set ${key_env} where the server starts") if key_env else "none needed",
            "model": p.get("model"), "stream": p.get("stream", True)}


def _endpoint_headers(p: Dict[str, Any]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json", **{str(k): str(v) for k, v in (p.get("headers") or {}).items()}}
    if p.get("key_env"):
        key = os.environ.get(p["key_env"])
        if not key:
            raise st.ToolError(f"This endpoint needs its key in ${p['key_env']}: set it where the server starts.")
        headers["Authorization"] = f"Bearer {key}"
    return headers


# --------------------------------------------------------------------------------------------------------- chats

_chats: Dict[str, Dict[str, Any]] = {}
_reaper: Dict[str, Any] = {"loop": None}


def _check_chat(cid: Any) -> str:
    """A chat's id is a UUID (the page makes it; it names files and Claude Code's session)."""
    try:
        return str(uuid.UUID(str(cid)))
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="A chat's id is a UUID.") from None


def _meta() -> Dict[str, Any]:
    try:
        d = json.loads((scientist_dir() / "chats.json").read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _meta_set(cid: str, **kw: Any) -> None:
    m = _meta()
    m.setdefault(cid, {}).update(kw)
    path = scientist_dir() / "chats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _chat(cid: str, agent: Optional[str] = None) -> Dict[str, Any]:
    """A chat's state, made on first need: its backend is the one recorded for it, else the one it is first sent with."""
    _ensure_reaper()
    c = _chats.get(cid)
    known = _meta().get(cid, {}).get("agent")
    if c is not None and agent in AGENTS and not known and c["proc"] is None and not c["busy"] and c["agent"] != agent:
        listeners = c["listeners"]  # an event stream opened before the first message keeps listening
        del _chats[cid]
        c = None
    else:
        listeners = None
    if c is None:
        kind = known or (agent if agent in AGENTS else "claude")
        c = _chats[cid] = {"session": cid, "agent": kind, "proc": None, "listeners": listeners or set(), "turn": [], "busy": False,
                           "last": time.time(), "pending": [], "cancel": asyncio.Event(), "task": None, "lock": asyncio.Lock()}
    return c


def _log(cid: str, ev: Dict[str, Any]) -> None:
    logs_dir().mkdir(parents=True, exist_ok=True)
    with open(logs_dir() / f"{cid}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def _emit(c: Dict[str, Any], ev: Dict[str, Any]) -> None:
    c["last"] = time.time()
    if ev.get("t") != "ui":
        c["turn"].append(ev)
        if c["agent"] in ("codex", "openai") and ev.get("t") in ("text", "tool", "tool_result", "error", "done"):
            _log(c["session"], ev)
    for q in list(c["listeners"]):
        q.put_nowait(ev)


def _cut(text: str, n: int = SHOWN_CUT) -> str:
    return text[:n] + (f" … ({len(text)} chars)" if len(text) > n else "")


def _self_url(request: Request) -> str:
    """This server's own address, for the MCP server a chat starts (the socket the request came in on)."""
    if os.environ.get("THEODOLITE_SELF_URL"):
        return os.environ["THEODOLITE_SELF_URL"].rstrip("/")
    host, port = (request.scope.get("server") or ("127.0.0.1", 8000))[:2]
    if host in (None, "", "0.0.0.0", "::", "testserver"):
        host = "127.0.0.1"
    return f"http://{'[' + host + ']' if ':' in str(host) else host}:{port}"


async def _stop_proc(proc: Any) -> None:
    if proc is None or proc.returncode is not None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            await asyncio.wait_for(proc.wait(), 5)
            return
        except asyncio.TimeoutError:
            pass


async def _end_chat(c: Dict[str, Any]) -> None:
    """Ends a chat's process now (and what it started)."""
    proc = c.get("proc")
    c["proc"] = None
    await _stop_proc(proc)


def _ensure_reaper() -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    if _reaper["loop"] is not loop:
        _reaper["loop"] = loop
        loop.create_task(_reap_chats())


async def _reap_chats() -> None:
    """Closes the input of chats idle past CHAT_IDLE with no one listening; the process exits once its own work is done."""
    while True:
        await asyncio.sleep(60)
        for c in list(_chats.values()):
            proc = c.get("proc")
            if proc and proc.returncode is None and not c["busy"] and not c["listeners"] and time.time() - c["last"] > CHAT_IDLE:
                try:
                    proc.stdin.close()
                except Exception:
                    pass


async def _write(proc: Any, obj: Dict[str, Any]) -> None:
    proc.stdin.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
    await proc.stdin.drain()


# ------------------------------------------------------------------------------------------------- Claude Code


def _session_file(sid: str) -> Optional[str]:
    hits = glob.glob(os.path.join(os.path.expanduser("~/.claude/projects"), "*", f"{os.path.basename(sid)}.jsonl"))
    return hits[0] if hits else None


def _claude_cmd(cid: str, self_url: str) -> List[str]:
    mcp = {"mcpServers": {"atlas": {"command": sys.executable, "args": ["-I", str(MCP_SCRIPT)],
                                    "env": {"THEODOLITE_URL": self_url, "THEODOLITE_CHAT": cid}}}}
    if reach() == "home":
        # file tools and a shell over the home folder, pre-approved, so it can run scripts against this server
        tools = ["--tools", *FILE_TOOLS, "--allowedTools", "mcp__atlas", *FILE_TOOLS, "--add-dir", os.path.expanduser("~")]
    else:
        # files only in its work folder (an absolute path in a rule begins with //), no shell
        w = str(work_dir()).rstrip("/")
        tools = ["--tools", "Read", "Write", "Edit", "--allowedTools", "mcp__atlas", *[f"{t}(/{w}/**)" for t in ("Read", "Write", "Edit")],
                 "--disallowedTools", "Bash"]
    cmd = [_bin("claude"), "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
           "--include-partial-messages", "--system-prompt-file", str(prompt_file("claude", self_url)),
           "--mcp-config", json.dumps(mcp), "--strict-mcp-config", *tools, "--setting-sources", "project"]
    if os.environ.get("THEODOLITE_CLAUDE_MODEL"):
        cmd += ["--model", os.environ["THEODOLITE_CLAUDE_MODEL"]]
    return cmd + (["--resume", cid] if _session_file(cid) else ["--session-id", cid])


def _user_text(m: Dict[str, Any]) -> Optional[str]:
    """The text of a message the person sent (None for tool results and notes the CLI adds)."""
    if m.get("type") != "user" or m.get("isMeta"):
        return None
    c = (m.get("message") or {}).get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list) and not any(isinstance(x, dict) and x.get("type") == "tool_result" for x in c):
        return next((x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text"), "")
    return None


def _tool_results(m: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for blk in (m.get("message") or {}).get("content") or []:
        if isinstance(blk, dict) and blk.get("type") == "tool_result":
            o = blk.get("content")
            if isinstance(o, list):
                o = "\n".join(x.get("text", "") for x in o if isinstance(x, dict))
            out.append({"t": "tool_result", "id": blk.get("tool_use_id"), "error": bool(blk.get("is_error")), "output": _cut(str(o or ""))})
    return out


def _refusal(m: Dict[str, Any]) -> Optional[Dict[str, str]]:
    msg = m.get("message") if isinstance(m.get("message"), dict) else {}
    if m.get("type") != "assistant" or msg.get("stop_reason") != "refusal":
        return None
    cat = (msg.get("stop_details") or {}).get("category")
    return {"id": msg.get("id"), "text": f"stopped by the {cat} safety classifier" if cat else "stopped by a safety classifier"}


CONTEXT_NOTE = re.compile(r"^\[The person has [^\]]*\]\n\n")


def _shown_text(t: str) -> str:
    """What the person typed, without the note on what they had open."""
    return CONTEXT_NOTE.sub("", t)


async def _pump_claude(c: Dict[str, Any], proc: Any) -> None:
    """The process's stream-json output turned into chat events, until it exits."""
    async for line in proc.stdout:
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if m.get("type") == "stream_event":
            e = m.get("event") or {}
            if e.get("type") == "content_block_delta":
                d = e.get("delta") or {}
                if d.get("type") == "text_delta":
                    c["busy"] = True
                    _emit(c, {"t": "text", "text": d.get("text", "")})
                elif d.get("type") == "thinking_delta":
                    _emit(c, {"t": "thinking", "text": d.get("thinking", "")})
        elif m.get("type") == "assistant":
            for blk in (m.get("message") or {}).get("content", []):
                if blk.get("type") == "tool_use":
                    c["busy"] = True
                    _emit(c, {"t": "tool", "id": blk.get("id"), "input": blk.get("input"), "name": str(blk.get("name", "")).removeprefix(MCP_PREFIX)})
            r = _refusal(m)
            if r and r["id"] not in c.setdefault("refused", set()):
                c["refused"].add(r["id"])
                _emit(c, {"t": "error", "error": r["text"]})
        elif m.get("type") == "user":
            for ev in _tool_results(m):
                _emit(c, ev)
        elif m.get("type") == "result":
            if m.get("is_error") and m.get("subtype") != "error_during_execution":
                _emit(c, {"t": "error", "error": str(m.get("result"))[:600]})
            c["busy"] = False
            _emit(c, {"t": "done", "cost": m.get("total_cost_usd")})
            c["turn"] = []
    err = (await proc.stderr.read()).decode(errors="replace")
    code = await proc.wait()
    if c.get("proc") is proc:
        c["proc"] = None
        if c["busy"] or code not in (0, None):
            _emit(c, {"t": "error", "error": err.strip()[-600:] or f"claude exited {code}"})
        if c["busy"]:
            c["busy"] = False
            _emit(c, {"t": "done"})
        c["turn"] = []


async def _ensure_claude(c: Dict[str, Any], self_url: str) -> Any:
    proc = c.get("proc")
    if proc is not None and proc.returncode is None:
        return proc
    if not shutil.which(_bin("claude")):
        raise RuntimeError("Claude Code (the `claude` command) is not installed where the server runs.")
    # its own process group, so ending a chat also ends what it started (the MCP server, its scripts)
    proc = c["proc"] = await asyncio.create_subprocess_exec(*_claude_cmd(c["session"], self_url), cwd=str(work_dir()), env=child_env("claude"),
                                                            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                            stderr=asyncio.subprocess.PIPE, limit=1 << 24, start_new_session=True)
    c["pump"] = asyncio.create_task(_pump_claude(c, proc))
    return proc


def attachment_block(fid: str) -> Optional[Dict[str, Any]]:
    """An uploaded file as a message content block: images and PDFs as themselves, anything else as text."""
    meta, raw = _upload(fid)
    if meta is None:
        return None
    kind = meta["type"]
    if kind in ("image/png", "image/jpeg", "image/gif", "image/webp"):
        return {"type": "image", "source": {"type": "base64", "media_type": kind, "data": base64.b64encode(raw).decode()}}
    if kind == "application/pdf" or meta["name"].lower().endswith(".pdf"):
        return {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": base64.b64encode(raw).decode()},
                "title": meta["name"]}
    return {"type": "text", "text": _file_text(meta["name"], raw)}


def _file_text(name: str, raw: bytes, cut: int = 200_000) -> str:
    text = raw.decode("utf-8", errors="replace")
    more = f"\n[... {len(text) - cut} more characters cut]" if len(text) > cut else ""
    return f'<file name="{name}">\n{text[:cut]}{more}\n</file>'


def _upload(fid: str) -> Tuple[Optional[Dict[str, Any]], bytes]:
    fid = os.path.basename(str(fid))
    mp = uploads_dir() / f"{fid}.json"
    if not re.fullmatch(r"[0-9a-f]{12}", fid) or not mp.exists():
        return None, b""
    meta = json.loads(mp.read_text(encoding="utf-8"))
    return meta, (uploads_dir() / f"{fid}-{meta['name']}").read_bytes()


# ------------------------------------------------------------------------------------------------------------ Codex


def _codex_home() -> str:
    """Codex's own home for the scientist: the person's login, their model and effort, and nothing else (no plugins,
    apps, memories or global instructions), so its tools are the atlas's and its shell."""
    import tomllib

    home = scientist_dir() / "codex-home"
    home.mkdir(parents=True, exist_ok=True)
    auth, src = home / "auth.json", Path(os.path.expanduser("~/.codex/auth.json"))
    if not os.path.lexists(auth) and src.exists():
        os.symlink(src, auth)
    try:
        user = tomllib.loads(Path(os.path.expanduser("~/.codex/config.toml")).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        user = {}
    lines = [f"{k} = {json.dumps(user[k])}" for k in ("model", "model_reasoning_effort") if isinstance(user.get(k), str)]
    lines += ["", "[features]", "apps = false", "memories = false"]
    (home / "config.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(home)


async def _rpc(c: Dict[str, Any], method: str, params: Dict[str, Any], timeout: float = 120) -> Dict[str, Any]:
    rpc = c["rpc"]
    rpc["next"] += 1
    fut = asyncio.get_running_loop().create_future()
    rpc["pending"][rpc["next"]] = fut
    await _write(c["proc"], {"jsonrpc": "2.0", "id": rpc["next"], "method": method, "params": params})
    r = await asyncio.wait_for(fut, timeout)
    if "error" in r:
        raise RuntimeError(f"codex {method}: {r['error'].get('message', r['error'])}")
    return r.get("result") or {}


def _codex_text(item: Dict[str, Any]) -> Optional[Tuple[str, Any, str, bool]]:
    """A tool item's name, input, output and whether it failed, as the events show them."""
    k = item.get("type")
    if k == "mcpToolCall":
        res = item.get("result") or {}
        out = "\n".join(x.get("text", "") for x in res.get("content") or [] if isinstance(x, dict)) or str(item.get("error") or "")
        return str(item.get("tool", "tool")).removeprefix(MCP_PREFIX), item.get("arguments") or {}, out, item.get("status") == "failed"
    if k == "commandExecution":
        return "Bash", {"command": item.get("command")}, item.get("aggregatedOutput") or "", item.get("status") in ("failed", "declined")
    if k == "fileChange":
        ch = item.get("changes") or []
        return "Edit", {"file_path": ", ".join(str(x.get("path", "")) for x in ch if isinstance(x, dict))}, item.get("status", ""), item.get("status") == "failed"
    if k == "webSearch":
        return "WebSearch", {"query": item.get("query")}, "", False
    return None


async def _pump_codex(c: Dict[str, Any], proc: Any) -> None:
    rpc = c["rpc"]
    async for line in proc.stdout:
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if "id" in m and ("result" in m or "error" in m) and "method" not in m:
            fut = rpc["pending"].pop(m["id"], None)
            if fut and not fut.done():
                fut.set_result(m)
            continue
        meth, prm = m.get("method"), m.get("params") or {}
        if "id" in m and meth:  # a request from codex (an approval): the thread runs without them; decline the rest
            await _write(proc, {"jsonrpc": "2.0", "id": m["id"], "result": {"decision": "decline"}})
            continue
        if meth == "turn/started":
            c["turn_id"], c["busy"] = (prm.get("turn") or {}).get("id"), True
        elif meth == "item/agentMessage/delta":
            c["busy"] = True
            if c.get("msg_item") != prm.get("itemId"):
                if c.get("msg_item"):
                    _emit(c, {"t": "text", "text": "\n\n"})
                c["msg_item"] = prm.get("itemId")
            _emit(c, {"t": "text", "text": prm.get("delta", "")})
        elif meth in ("item/started", "item/completed"):
            t = _codex_text(prm.get("item") or {})
            if t:
                name, inp, out, failed = t
                iid = (prm.get("item") or {}).get("id")
                if meth == "item/started":
                    c["msg_item"] = None
                    _emit(c, {"t": "tool", "id": iid, "name": name, "input": inp})
                else:
                    _emit(c, {"t": "tool_result", "id": iid, "error": failed, "output": _cut(out)})
        elif meth == "turn/completed":
            turn = prm.get("turn") or {}
            if turn.get("status") == "failed" and turn.get("error"):
                _emit(c, {"t": "error", "error": str(turn["error"].get("message", turn["error"]))[:600]})
            c["busy"], c["turn_id"], c["msg_item"] = False, None, None
            _emit(c, {"t": "done"})
            c["turn"] = []
        elif meth == "error":
            _emit(c, {"t": "error", "error": str((prm.get("error") or {}).get("message", prm))[:600]})
    await proc.wait()
    if c.get("proc") is proc:
        c["proc"] = None
        if c["busy"]:
            c["busy"] = False
            _emit(c, {"t": "error", "error": "codex exited"})
            _emit(c, {"t": "done"})
        c["turn"] = []


async def _ensure_codex(c: Dict[str, Any], self_url: str) -> Any:
    proc = c.get("proc")
    if proc is not None and proc.returncode is None:
        return proc
    if not shutil.which(_bin("codex")):
        raise RuntimeError("Codex (the `codex` command) is not installed where the server runs.")
    env = child_env("codex", CODEX_HOME=_codex_home())
    proc = c["proc"] = await asyncio.create_subprocess_exec(_bin("codex"), "app-server", cwd=str(work_dir()), env=env,
                                                            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                            stderr=asyncio.subprocess.DEVNULL, limit=1 << 24, start_new_session=True)
    c["rpc"] = {"next": 0, "pending": {}}
    c["pump"] = asyncio.create_task(_pump_codex(c, proc))
    await _rpc(c, "initialize", {"clientInfo": {"name": "logos-theodolite", "version": "1"}})
    # in its sandbox Codex asks before every MCP tool call, and this chat never asks (approvalPolicy "never"): the
    # atlas's own tools are approved, as full access approved them
    mcp = {"atlas": {"command": sys.executable, "args": ["-I", str(MCP_SCRIPT)], "env": {"THEODOLITE_URL": self_url, "THEODOLITE_CHAT": c["session"]},
                     "default_tools_approval_mode": "approve"}}
    sandbox = "danger-full-access" if reach() == "home" else "workspace-write"  # module docstring: reach
    opts = {"cwd": str(work_dir()), "approvalPolicy": "never", "sandbox": sandbox, "config": {"mcp_servers": mcp},
            "developerInstructions": system_prompt("codex", self_url)}
    thread = _meta().get(c["session"], {}).get("thread")
    r = await _rpc(c, "thread/resume", {"threadId": thread, **opts}) if thread else await _rpc(c, "thread/start", opts)
    c["thread"] = (r.get("thread") or {}).get("id")
    if not thread:
        _meta_set(c["session"], thread=c["thread"])
    return proc


def _codex_input(fid: str) -> Optional[Dict[str, Any]]:
    meta, raw = _upload(fid)
    if meta is None:
        return None
    if meta["type"].startswith("image/"):
        return {"type": "localImage", "path": str(uploads_dir() / f"{os.path.basename(fid)}-{meta['name']}")}
    if meta["type"] == "application/pdf" or meta["name"].lower().endswith(".pdf"):
        return None
    return {"type": "text", "text": _file_text(meta["name"], raw), "text_elements": []}


# ------------------------------------------------------------------------------------- OpenAI-compatible endpoints


class InProcessApi:
    """The atlas's routes reached in this process (httpx over ASGI), as scientist_tools.call reaches them: the same
    routes, checks and GPU lock as a page's request."""

    def __init__(self, app: Any) -> None:
        import httpx

        # a route's own failure comes back as its 500, for the model to read, not as an exception here
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://127.0.0.1",
                                        timeout=None)

    @staticmethod
    def _json(r: Any) -> Any:
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise st.ToolError(f"{r.status_code}: {detail if isinstance(detail, str) else json.dumps(detail)}"[:2000])
        return r.json()

    async def get(self, path: str, **query: Any) -> Any:
        return self._json(await self.client.get(path, params={k: v for k, v in query.items() if v is not None}))

    async def post(self, path: str, body: Optional[Dict[str, Any]]) -> Any:
        return self._json(await self.client.post(path, json=body or {}))

    async def close(self) -> None:
        await self.client.aclose()


class EndpointError(Exception):
    pass


def _messages_path(cid: str) -> Path:
    return logs_dir() / f"{cid}.messages.json"


def _load_messages(cid: str) -> List[Dict[str, Any]]:
    try:
        m = json.loads(_messages_path(cid).read_text(encoding="utf-8"))
        return m if isinstance(m, list) else []
    except (OSError, ValueError):
        return []


def _save_messages(cid: str, messages: List[Dict[str, Any]]) -> None:
    path = _messages_path(cid)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(messages, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _openai_content(msg: str, attachments: List[str]) -> Any:
    """A message the person sent, as an OpenAI-compatible user message's content: the text, images as data URLs, other
    files as text; a PDF is named, not sent (such an endpoint takes none)."""
    parts: List[Dict[str, Any]] = [{"type": "text", "text": msg}]
    for fid in attachments:
        meta, raw = _upload(fid)
        if meta is None:
            continue
        if meta["type"] in ("image/png", "image/jpeg", "image/gif", "image/webp"):
            parts.append({"type": "image_url", "image_url": {"url": f"data:{meta['type']};base64,{base64.b64encode(raw).decode()}"}})
        elif meta["type"] == "application/pdf" or meta["name"].lower().endswith(".pdf"):
            parts.append({"type": "text", "text": f"[a PDF, {meta['name']}, was attached; this endpoint is not sent PDFs]"})
        else:
            parts.append({"type": "text", "text": _file_text(meta["name"], raw)})
    return msg if len(parts) == 1 else parts


async def _complete(c: Dict[str, Any], client: Any, url: str, headers: Dict[str, str], body: Dict[str, Any]) -> Dict[str, Any]:
    """One completion: the reply streamed to the page as it comes (text, and a reasoning model's thinking), returned as
    an assistant message with its tool calls. Without streaming (the profile's `stream: false`, or a server that
    refuses it), one reply."""
    host = urlparse(url).hostname
    if body.get("stream"):
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            if resp.status_code >= 400:
                text = (await resp.aread()).decode("utf-8", errors="replace")
                if resp.status_code in (400, 422) and "stream" in text.lower():
                    return await _complete(c, client, url, headers, {**body, "stream": False})
                raise EndpointError(f"{host} answered {resp.status_code}: {text[:800]}")
            content: List[str] = []
            calls: Dict[int, Dict[str, str]] = {}
            async for line in resp.aiter_lines():
                if c["cancel"].is_set():
                    break
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if chunk.get("error"):
                    raise EndpointError(f"{host}: {json.dumps(chunk['error'])[:800]}")
                for ch in chunk.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("content"):
                        content.append(d["content"])
                        _emit(c, {"t": "text", "text": d["content"]})
                    thought = d.get("reasoning") or d.get("reasoning_content")
                    if thought:
                        _emit(c, {"t": "thinking", "text": thought})
                    for tc in d.get("tool_calls") or []:
                        e = calls.setdefault(int(tc.get("index", len(calls))), {"id": "", "name": "", "arguments": ""})
                        if tc.get("id"):
                            e["id"] = tc["id"]
                        f = tc.get("function") or {}
                        if f.get("name"):  # whole in one piece on most servers, in pieces on some, sent again on others
                            e["name"] = f["name"] if f["name"].startswith(e["name"]) else e["name"] + f["name"]
                        if f.get("arguments"):
                            e["arguments"] += f["arguments"]
            msg: Dict[str, Any] = {"role": "assistant", "content": "".join(content) or None}
            if calls:
                msg["tool_calls"] = [{"id": e["id"] or f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                                      "function": {"name": e["name"], "arguments": e["arguments"] or "{}"}} for _, e in sorted(calls.items())]
            return msg
    resp = await client.post(url, json=body, headers=headers)
    if resp.status_code >= 400:
        raise EndpointError(f"{host} answered {resp.status_code}: {resp.text[:800]}")
    try:
        m = (resp.json().get("choices") or [{}])[0].get("message") or {}
    except ValueError:
        raise EndpointError(f"{host} answered something other than JSON: {resp.text[:300]}") from None
    thought = m.get("reasoning") or m.get("reasoning_content")
    if thought:
        _emit(c, {"t": "thinking", "text": thought})
    if m.get("content"):
        _emit(c, {"t": "text", "text": m["content"]})
    msg = {"role": "assistant", "content": m.get("content") or None}
    if m.get("tool_calls"):
        msg["tool_calls"] = [{"id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                              "function": {"name": (tc.get("function") or {}).get("name", ""),
                                           "arguments": (tc.get("function") or {}).get("arguments") or "{}"}} for tc in m["tool_calls"]]
    return msg


async def _run_tool(c: Dict[str, Any], api: Any, tc: Dict[str, Any]) -> str:
    name = tc["function"]["name"]
    raw = tc["function"].get("arguments") or "{}"
    try:
        args = json.loads(raw) if raw.strip() else {}
    except ValueError as exc:
        args, text, err = raw, f"error: your arguments were not JSON ({exc}); send them again as a JSON object", True
    else:
        try:
            if not isinstance(args, dict):
                raise st.ToolError("A tool's arguments are a JSON object.")
            text, err = st.as_text(await st.call(name, args, api, c["session"]), cut=TOOL_CUT), False
        except st.ToolError as exc:
            text, err = f"error: {exc}", True
        except Exception as exc:  # the tool's failure goes back to the model, in its own words
            logger.exception("scientist tool %s failed", name)
            text, err = f"error: {type(exc).__name__}: {exc}", True
    _emit(c, {"t": "tool_result", "id": tc["id"], "error": err, "output": _cut(text)})
    return text


async def _openai_turn(c: Dict[str, Any], self_url: str) -> None:
    """The tool loop of one turn: completions until the model answers without a tool call, each call run through the
    atlas in this process; a message sent meanwhile joins between tool calls; an interrupt stops it where it is."""
    import httpx

    cid = c["session"]
    meta = _meta().get(cid, {})
    api = InProcessApi(_app)
    try:
        p = profiles().get(meta.get("profile", ""))
        if p is None:
            raise EndpointError(f"The endpoint {meta.get('profile')!r} is not configured on this server.")
        try:
            headers = _endpoint_headers(p)
        except st.ToolError as exc:
            raise EndpointError(str(exc)) from None
        url = p["base_url"].rstrip("/") + "/chat/completions"
        tools = st.openai_tools()
        messages = c.setdefault("messages", _load_messages(cid))
        rounds = 0
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=15.0, read=600.0, write=60.0, pool=15.0)) as client:
            while True:
                while c["pending"]:
                    messages.append(c["pending"].pop(0))
                if c["cancel"].is_set():
                    break
                rounds += 1
                if rounds > MAX_ROUNDS:
                    _emit(c, {"t": "error", "error": f"stopped after {MAX_ROUNDS} rounds of tool calls in one turn"})
                    break
                body = {"model": meta.get("model") or p.get("model"), "tools": tools, "tool_choice": "auto",
                        "stream": bool(p.get("stream", True)), **(p.get("params") or {}),
                        "messages": [{"role": "system", "content": system_prompt("openai", self_url)}] + messages}
                if not body["model"]:
                    raise EndpointError("Pick a model for this endpoint.")
                msg = await _complete(c, client, url, headers, body)
                if c["cancel"].is_set():
                    if msg.get("content"):
                        messages.append({"role": "assistant", "content": msg["content"]})
                    break
                messages.append(msg)
                _save_messages(cid, messages)
                calls = msg.get("tool_calls") or []
                if not calls:
                    if c["pending"]:
                        continue  # a message came in while it answered: it answers that too
                    break
                for i, tc in enumerate(calls):
                    if c["cancel"].is_set():
                        for rest in calls[i:]:  # every call is answered, or the endpoint refuses the conversation
                            messages.append({"role": "tool", "tool_call_id": rest["id"], "content": "not run: the person stopped the reply"})
                        break
                    try:
                        inp = json.loads(tc["function"].get("arguments") or "{}")
                    except ValueError:
                        inp = tc["function"].get("arguments")
                    _emit(c, {"t": "tool", "id": tc["id"], "name": tc["function"]["name"], "input": inp})
                    text = await _run_tool(c, api, tc)
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": text})
                    _save_messages(cid, messages)
    except EndpointError as exc:
        _emit(c, {"t": "error", "error": str(exc)})
    except Exception as exc:
        logger.exception("scientist chat %s failed", cid)
        _emit(c, {"t": "error", "error": f"{type(exc).__name__}: {exc}"[:600]})
    finally:
        await api.close()
        if "messages" in c:
            _save_messages(cid, c["messages"])
        c["busy"], c["task"] = False, None
        c["cancel"].clear()
        _emit(c, {"t": "done"})
        c["turn"] = []


# ------------------------------------------------------------------------------------------------------------ routes


class SendRequest(BaseModel):
    chat: str = Field(..., max_length=64)
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE)
    agent: Optional[str] = Field(None, max_length=16)
    profile: Optional[str] = Field(None, max_length=64)
    model: Optional[str] = Field(None, max_length=200)
    layer: Optional[int] = None
    n: Optional[int] = None
    head: Optional[int] = None
    attachments: List[str] = Field(default_factory=list, max_length=16)


class ChatRequest(BaseModel):
    chat: str = Field(..., max_length=64)
    text: Optional[str] = Field(None, max_length=MAX_MESSAGE)


class UploadRequest(BaseModel):
    name: str = Field("file", max_length=200)
    type: str = Field("", max_length=100)
    data: str


@router.get("/backends")
async def backends() -> Dict[str, Any]:
    """Which backends this server can run: Claude Code and Codex where their commands are installed, and every
    OpenAI-compatible endpoint configured (`profiles()`; each one's key only as "set" or "missing")."""
    ps = profiles()
    return {"claude": {"available": bool(shutil.which(_bin("claude"))), "model": os.environ.get("THEODOLITE_CLAUDE_MODEL")},
            "codex": {"available": bool(shutil.which(_bin("codex")))},
            "openai": [profile_card(k, v) for k, v in ps.items()], "warning": _profile_warning,
            "reach": {"mode": reach(), "env": REACH_ENV, **REACH_WORDS[reach()]},
            "prompt": {"files": [str(p) for p in PROMPT_FILES], "local": str(scientist_dir() / "prompt.local.md")}}


@router.get("/models")
async def models(profile: str) -> Dict[str, Any]:
    """The models an OpenAI-compatible endpoint lists (`GET <base>/models`), each with whether it says it takes tools
    (None when the endpoint does not say)."""
    import httpx

    p = profiles().get(profile)
    if p is None:
        raise HTTPException(status_code=404, detail=f"No endpoint {profile!r} is configured here.")
    try:
        headers = _endpoint_headers(p)
    except st.ToolError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
            r = await client.get(p["base_url"].rstrip("/") + "/models", headers=headers)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"{urlparse(p['base_url']).hostname} did not answer: {exc}") from None
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"{urlparse(p['base_url']).hostname} answered {r.status_code}: {r.text[:300]}")
    try:
        data = r.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="The endpoint's model list was not JSON.") from None
    items = data.get("data", data.get("models", [])) if isinstance(data, dict) else data
    out = []
    for m in items if isinstance(items, list) else []:
        if not isinstance(m, dict) or not m.get("id"):
            continue
        sp = m.get("supported_parameters")
        out.append({"id": str(m["id"]), "name": m.get("name"), "context": m.get("context_length"),
                    "tools": ("tools" in sp) if isinstance(sp, list) else None})
    return {"profile": profile, "models": out, "default": p.get("model")}


@router.post("/send")
async def send(req: SendRequest, request: Request) -> Dict[str, Any]:
    """A message into a chat (starting its backend when it is not running). While the scientist works, the message
    reaches it between tool calls. `layer` and `n` (or `head`), counted from 0, say what the person has open: the
    scientist is told, in the pages' count. Events arrive on GET /events."""
    cid = _check_chat(req.chat)
    if req.agent is not None and req.agent not in AGENTS:
        raise HTTPException(status_code=400, detail=f"The scientist's backends are {', '.join(AGENTS)}.")
    c = _chat(cid, req.agent)
    known = _meta().get(cid, {})
    if c["agent"] == "openai" and not known.get("profile"):
        ps = profiles()
        if not req.profile or req.profile not in ps:
            raise HTTPException(status_code=400, detail=f"Pick an endpoint for this chat: {', '.join(ps)}.")
        model = req.model or ps[req.profile].get("model")
        if not model:
            raise HTTPException(status_code=400, detail="Pick a model for this endpoint.")
        _meta_set(cid, agent="openai", profile=req.profile, model=model, created=int(time.time()))
    elif not known.get("agent"):
        _meta_set(cid, agent=c["agent"], created=int(time.time()))
    msg = shown = req.message
    if req.layer is not None and req.head is not None:
        msg = f"[The person has attention head {st.cite_head(req.layer, req.head)} open in the atlas.]\n\n{msg}"
    elif req.layer is not None and req.n is not None:
        msg = f"[The person has {st.cite(req.layer, req.n)} open in the atlas.]\n\n{msg}"
    self_url = _self_url(request)
    async with c["lock"]:
        if c["agent"] == "openai":
            if _app is None:
                raise HTTPException(status_code=503, detail="The scientist chat is not bound to a server.")
            user = {"role": "user", "content": _openai_content(msg, req.attachments)}
            _log(cid, {"t": "user", "text": shown})
            _emit(c, {"t": "sent"})
            if c["busy"]:
                c["pending"].append(user)
                return {"ok": True, "queued": True}
            c.setdefault("messages", _load_messages(cid)).append(user)
            c["busy"] = True
            c["cancel"].clear()
            c["task"] = asyncio.create_task(_openai_turn(c, self_url))
            return {"ok": True}
        if c["agent"] == "codex":
            try:
                await _ensure_codex(c, self_url)
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"codex: {exc}") from None
            _log(cid, {"t": "user", "text": shown})
            items = [{"type": "text", "text": msg, "text_elements": []}] + [x for a in req.attachments if (x := _codex_input(a))]
            c["busy"], c["last"] = True, time.time()
            _emit(c, {"t": "sent"})
            if c.get("turn_id"):  # mid-turn: the message joins the running turn
                try:
                    await _rpc(c, "turn/steer", {"threadId": c["thread"], "expectedTurnId": c["turn_id"], "input": items})
                    return {"ok": True}
                except Exception:
                    pass  # the turn ended meanwhile: start a new one
            r = await _rpc(c, "turn/start", {"threadId": c["thread"], "input": items})
            c["turn_id"] = (r.get("turn") or {}).get("id")
            _meta_set(cid, turns=_meta().get(cid, {}).get("turns", []) + [c["turn_id"]])
            return {"ok": True}
        try:
            proc = await _ensure_claude(c, self_url)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from None
        content = [{"type": "text", "text": msg}] + [blk for a in req.attachments if (blk := attachment_block(a))]
        c["busy"], c["last"] = True, time.time()
        _emit(c, {"t": "sent"})
        await _write(proc, {"type": "user", "message": {"role": "user", "content": content}})
        return {"ok": True}


@router.post("/interrupt")
async def interrupt(req: ChatRequest) -> Dict[str, Any]:
    """Stops the chat's current reply; the chat (and, under Claude Code or Codex, its background commands) goes on."""
    c = _chats.get(_check_chat(req.chat))
    if c is None or not c["busy"]:
        return {"ok": True, "running": False}
    if c["agent"] == "openai":
        c["cancel"].set()
        return {"ok": True, "running": True}
    proc = c.get("proc")
    if not proc or proc.returncode is not None:
        return {"ok": True, "running": False}
    if c["agent"] == "codex":
        if not c.get("turn_id"):
            return {"ok": True, "running": False}
        await _rpc(c, "turn/interrupt", {"threadId": c["thread"], "turnId": c["turn_id"]})
        return {"ok": True, "running": True}
    await _write(proc, {"type": "control_request", "request_id": str(uuid.uuid4()), "request": {"subtype": "interrupt"}})
    return {"ok": True, "running": True}


@router.get("/events")
async def events(chat: str) -> StreamingResponse:
    """The chat's events as NDJSON: {t: "state", busy, running} and the current turn's events so far, then each as it
    comes: text, thinking, tool, tool_result, ui, error, done (a turn ended), sent (a message went in), ping."""
    c = _chat(_check_chat(chat))
    q: asyncio.Queue = asyncio.Queue()
    c["listeners"].add(q)

    async def gen():
        try:
            running = bool(c.get("proc") and c["proc"].returncode is None) or c.get("task") is not None
            yield json.dumps({"t": "state", "busy": c["busy"], "running": running, "agent": c["agent"]}) + "\n"
            for ev in list(c["turn"]):
                yield json.dumps(ev, ensure_ascii=False) + "\n"
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), 20)
                except asyncio.TimeoutError:
                    yield json.dumps({"t": "ping"}) + "\n"
                    continue
                yield json.dumps(ev, ensure_ascii=False) + "\n"
        finally:
            c["listeners"].discard(q)

    return StreamingResponse(gen(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"})


@router.post("/ui")
async def ui(body: Dict[str, Any]) -> Dict[str, Any]:
    """A screen action from the scientist's tools (select, head, tab, camera, highlight, annotate, text, search,
    rename), passed to the chat's page; layers counted from 0."""
    cid = str(body.pop("chat", ""))
    c = _chats.get(cid)
    if c is None:
        return {"ok": False, "error": "no such chat is open on this server"}
    if body.get("action") not in set(st.UI_TOOLS.values()) | {"rename"}:
        return {"ok": False, "error": f"no screen action {body.get('action')!r}"}
    _emit(c, {"t": "ui", **body})
    return {"ok": True, "note": "shown in the atlas" if c["listeners"] else "the person's page is not open; nothing was shown"}


_chat_list: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def _summary(cid: str, m: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A past chat's line in the list: {chat, agent, t (its last change), title (the first message), turns}."""
    agent = m.get("agent", "claude")
    path = _session_file(cid) if agent == "claude" else str(logs_dir() / f"{cid}.jsonl")
    if not path or not os.path.exists(path):
        return None
    mt = os.path.getmtime(path)
    hit = _chat_list.get(path)
    if hit and hit[0] == mt:
        return hit[1]
    title, turns = None, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            t = _user_text(e) if agent == "claude" else (e.get("text") if e.get("t") == "user" else None)
            if t is not None:
                turns += 1
                title = title or _shown_text(t).strip()[:120]
    out = {"chat": cid, "agent": agent, "t": int(mt), "title": title, "turns": turns,
           **({"profile": m.get("profile"), "model": m.get("model")} if agent == "openai" else {})}
    _chat_list[path] = (mt, out)
    return out


@router.get("/chats")
async def chats(limit: int = 60) -> List[Dict[str, Any]]:
    """Past chats, newest first."""
    rows = [s for cid, m in _meta().items() if (s := await asyncio.to_thread(_summary, cid, m)) and s["turns"]]
    return sorted(rows, key=lambda r: -r["t"])[: max(1, min(int(limit), 500))]


@router.get("/history")
async def history(chat: str) -> Dict[str, Any]:
    """A past chat as the events a live turn streams (text, tool, tool_result, error), with a user event per message."""
    cid = _check_chat(chat)
    m = _meta().get(cid, {})
    agent = m.get("agent", "claude")
    if agent in ("codex", "openai"):
        path = logs_dir() / f"{cid}.jsonl"
        ev = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("t") in ("user", "text", "tool", "tool_result", "error"):
                    # the log keeps each streamed fragment; a reply is their run joined, as the live page shows it (a
                    # fragment boundary can fall inside a number or a citation)
                    if e["t"] == "text" and ev and ev[-1].get("t") == "text":
                        ev[-1] = {**ev[-1], "text": ev[-1].get("text", "") + e.get("text", "")}
                    else:
                        ev.append(dict(e))
        return {"chat": cid, "agent": agent, "events": ev, **({"profile": m.get("profile"), "model": m.get("model")} if agent == "openai" else {})}
    path = _session_file(cid)
    if not path:
        return {"chat": cid, "agent": agent, "events": []}
    ev, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("isSidechain") or msg.get("isMeta"):
                continue
            t = _user_text(msg)
            if t is not None:
                ev.append({"t": "user", "text": _shown_text(t)})
            elif msg.get("type") == "assistant":
                r = _refusal(msg)
                if r and r["id"] not in seen:
                    seen.add(r["id"])
                    ev.append({"t": "error", "error": r["text"]})
                for blk in (msg.get("message") or {}).get("content", []):
                    if blk.get("type") == "text" and blk.get("text"):
                        ev.append({"t": "text", "text": blk["text"]})
                    elif blk.get("type") == "tool_use":
                        ev.append({"t": "tool", "id": blk.get("id"), "input": blk.get("input"), "name": str(blk.get("name", "")).removeprefix(MCP_PREFIX)})
            elif msg.get("type") == "user":
                ev += _tool_results(msg)
    return {"chat": cid, "agent": agent, "events": ev}


@router.post("/rewind")
async def rewind(req: ChatRequest, request: Request) -> Dict[str, Any]:
    """Cut a chat back to just before the last message the person sent that ends with `text` (any last message
    without it), so it can be sent again: the reply goes too. A Claude Code session is backed up beside it first. No
    match (already cut) is not an error."""
    cid = _check_chat(req.chat)
    text = (req.text or "").strip()
    m = _meta().get(cid, {})
    agent = m.get("agent", "claude")
    c = _chats.get(cid)
    if c is not None and c["busy"]:
        raise HTTPException(status_code=409, detail="The scientist is still answering; stop it first.")
    if agent == "openai":
        msgs = c["messages"] if c is not None and "messages" in c else _load_messages(cid)
        last = max((i for i, x in enumerate(msgs) if x.get("role") == "user"), default=None)
        if last is not None:
            del msgs[last:]
            _save_messages(cid, msgs)
        _cut_log(cid)
        return {"ok": True, "cut": last is not None}
    if agent == "codex":
        turns = m.get("turns") or []
        if turns and c is not None:
            try:
                await _ensure_codex(c, _self_url(request))
                await _rpc(c, "thread/revert", {"threadId": c["thread"], "beforeTurnId": turns[-1]})
            finally:
                await _end_chat(c)
            _meta_set(cid, turns=turns[:-1])
        _cut_log(cid)
        return {"ok": True, "cut": bool(turns)}
    if c is not None:  # a live process holds the conversation in memory; the next message resumes from the cut file
        await _end_chat(c)
    path = _session_file(cid)
    if not path:
        return {"ok": True, "cut": False}
    lines = open(path, encoding="utf-8").read().splitlines(True)
    at = None
    for i in range(len(lines) - 1, -1, -1):
        try:
            t = _user_text(json.loads(lines[i]))
        except ValueError:
            continue
        if t is not None and (not text or t.strip() == text or t.strip().endswith("\n\n" + text) or t.split("\n\n", 1)[-1].strip() == text):
            at = i
            break
    if at is None:
        return {"ok": True, "cut": False}
    shutil.copy(path, f"{path}.bak-{int(time.time())}")
    keep = lines[:at]
    if not any(_user_text(json.loads(x)) is not None for x in keep if x.strip()):
        os.remove(path)  # nothing the person said is left: the next turn starts the session afresh
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(keep)
    return {"ok": True, "cut": True}


def _cut_log(cid: str) -> None:
    path = logs_dir() / f"{cid}.jsonl"
    if not path.exists():
        return
    ev = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    last = max((i for i, e in enumerate(ev) if e.get("t") == "user"), default=None)
    if last is not None:
        path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in ev[:last]), encoding="utf-8")


@router.post("/upload")
async def upload(req: UploadRequest) -> Dict[str, Any]:
    """A file for the chat, as base64 ({name, type, data}), kept in the scientist folder's uploads/."""
    try:
        raw = base64.b64decode(req.data, validate=True)
    except ValueError:
        raise HTTPException(status_code=400, detail="The file's data is not base64.") from None
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(status_code=413, detail="A file can be at most 25 MB.")
    d = uploads_dir()
    d.mkdir(parents=True, exist_ok=True)
    fid = uuid.uuid4().hex[:12]
    name = os.path.basename(req.name.replace("\\", "/")) or "file"
    name = re.sub(r"[^\w.\- ]", "_", name)[:120] or "file"
    (d / f"{fid}-{name}").write_bytes(raw)
    (d / f"{fid}.json").write_text(json.dumps({"name": name, "type": req.type or "", "size": len(raw)}), encoding="utf-8")
    return {"id": fid, "name": name, "size": len(raw)}


@router.get("/prompt")
async def prompt(agent: str, request: Request) -> Dict[str, Any]:
    """The instructions a chat on this backend starts with, as built from their files."""
    if agent not in AGENTS:
        raise HTTPException(status_code=400, detail=f"The scientist's backends are {', '.join(AGENTS)}.")
    return {"agent": agent, "text": system_prompt(agent, _self_url(request)), "files": [str(p) for p in PROMPT_FILES]}

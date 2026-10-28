#!/usr/bin/env python3
"""Lab daemon: keeps Gemma-4-31B (W4A16) + J++/JLens loaded in ONE GPU process and executes
submitted experiment code against it, serialized by a lock.

POST /exec  {"code": "...", "label": "..."}  -> {"ok", "stdout", "result", "error", "elapsed_s"}
  Code runs with top-level await allowed, in a fresh namespace containing:
    rt (TheodoliteRuntime), lab (labkit.Lab), torch, np, F, STORE (persistent dict shared across calls)
  Assign to a variable named `result` to return JSON-serializable data.
POST /reload  -> reload labkit (after editing labkit.py)
GET  /health
Binds 127.0.0.1 only, on port 8765 unless $LABD_PORT says otherwise: the same port as the instrument's server
(README §v), so run one or the other (one GPU process at a time), and through an SSH tunnel to a pod either one would
sit at the browser's own 127.0.0.1:8765.

Who may ask: a script run on this machine by the person who started the daemon. Every route refuses (403) a request
whose Host is not 127.0.0.1, localhost or [::1] (a page rebinding its own name to 127.0.0.1 sends that name as the
Host) and any request a browser sends (an Origin or a Sec-Fetch-Site header): no web page has any business here.
/exec and /reload also want this launch's token (401 otherwise), a fresh random one written at each start to a file
only its user can read (0600): $LABD_TOKEN_FILE, else ${XDG_RUNTIME_DIR:-~/.cache}/theodolite/labd-<port>.token,
removed again on a clean exit. labclient.py reads it and sends it as `Authorization: Bearer <token>`; a client on
another machine (the far end of a tunnel) needs a copy of that file. /health wants no token.
"""
from __future__ import annotations

import ast
import gc
import asyncio
import contextlib
import importlib
import io
import json
import os
import secrets
import sys
import tempfile
import time
import traceback
from pathlib import Path
from urllib.parse import urlsplit

os.environ.setdefault("HF_HUB_OFFLINE", "1")
REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)  # experiment scripts use repo-relative paths
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel

from backend.engine_runtime import TheodoliteRuntime
import labkit

LOG = REPO / "experiments" / "results" / "labd_exec_log.local.jsonl"  # gitignored; the study's log is labd_exec_log_2026-10-16.jsonl
PORT = int(os.environ.get("LABD_PORT", "8765"))
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
TOKEN: dict = {"value": None}  # this launch's, set by write_token; until then /exec and /reload refuse every request


def token_path(port: int) -> Path:
    """Where a launch on `port` keeps its token (labclient.py reads the same path)."""
    if os.environ.get("LABD_TOKEN_FILE"):
        return Path(os.environ["LABD_TOKEN_FILE"]).expanduser()
    base = os.environ.get("XDG_RUNTIME_DIR") or str(Path.home() / ".cache")
    return Path(base) / "theodolite" / f"labd-{port}.token"


def write_token(path: Path) -> str:
    """A fresh token, written whole to `path`, readable by this user only (mkstemp makes the file 0600)."""
    token = secrets.token_urlsafe(32)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".labd-token-")
    with os.fdopen(fd, "w") as f:
        f.write(token + "\n")
    os.replace(tmp, path)
    TOKEN["value"] = token
    return token


def local_scripts_only(request: Request) -> None:
    """Refuse (403) a Host that does not name this machine, and any request a browser sends."""
    host = request.headers.get("host")
    try:
        name = urlsplit("//" + host).hostname if host else None
    except ValueError:
        name = None
    if name not in LOCAL_HOSTS or "origin" in request.headers or "sec-fetch-site" in request.headers:
        raise HTTPException(status_code=403, detail="labd answers only scripts run on this machine (labclient.py), never a web page.")


def launch_token(request: Request) -> None:
    """Refuse (401) a request without this launch's token."""
    want = TOKEN["value"]
    sent = request.headers.get("authorization", "")
    if not want or not secrets.compare_digest(sent.encode(), f"Bearer {want}".encode()):
        raise HTTPException(status_code=401, detail="labd wants this launch's token, which labclient.py reads from "
                                                    f"{token_path(PORT)} and sends.")


# every route behind the Host and browser rule; no docs pages
app = FastAPI(dependencies=[Depends(local_scripts_only)], docs_url=None, redoc_url=None, openapi_url=None)
LOCK = asyncio.Lock()
STATE: dict = {}
STORE: dict = {}


class ExecReq(BaseModel):
    code: str
    label: str = ""
    out: str = ""   # optional path: daemon writes the full response JSON there (survives client timeouts)


WAITING = {"n": 0}


def _jsonable(x):
    try:
        json.dumps(x)
        return x
    except TypeError:
        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().tolist()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, dict):
            return {str(k): _jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_jsonable(v) for v in x]
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        return repr(x)


@app.on_event("startup")
async def _startup():
    t0 = time.time()
    rt = TheodoliteRuntime(load_jlens=True).load()
    STATE["rt"] = rt
    STATE["lab"] = labkit.Lab(rt)
    print(f"[labd] ready in {time.time()-t0:.1f}s; VRAM alloc {torch.cuda.memory_allocated()/2**30:.2f} GB", flush=True)


@app.get("/health")
async def health():
    return {"ready": "lab" in STATE, "busy": LOCK.locked(), "waiting": WAITING["n"], "running": STATE.get("running"),
            "vram_alloc_gb": torch.cuda.memory_allocated() / 2**30}


@app.post("/reload", dependencies=[Depends(launch_token)])
async def reload():
    async with LOCK:
        importlib.reload(labkit)
        STATE["lab"] = labkit.Lab(STATE["rt"])
        return {"ok": True}


@app.post("/exec", dependencies=[Depends(launch_token)])
async def execute(req: ExecReq):
    WAITING["n"] += 1
    async with LOCK:
        WAITING["n"] -= 1
        STATE["running"] = req.label
        t0 = time.time()
        ns = {"rt": STATE["rt"], "lab": STATE["lab"], "torch": torch, "np": np, "F": F,
              "STORE": STORE, "labkit": labkit, "asyncio": asyncio}
        buf = io.StringIO()
        err = None
        try:
            code = compile(req.code, f"<exec:{req.label}>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
            with contextlib.redirect_stdout(buf):
                with torch.no_grad():
                    maybe = eval(code, ns)
                    if asyncio.iscoroutine(maybe):
                        await maybe
        except Exception:
            err = traceback.format_exc()
        finally:
            STATE["lab"].clear_hooks()
            result = _jsonable(ns.get("result"))
            ns.clear()          # break namespace <-> function __globals__ cycles so GPU tensors are freed
            gc.collect()
            torch.cuda.empty_cache()
        out = {"ok": err is None, "stdout": buf.getvalue()[-200_000:], "error": err,
               "result": result, "elapsed_s": round(time.time() - t0, 2)}
        STATE["running"] = None
        if req.out:
            try:
                Path(req.out).parent.mkdir(parents=True, exist_ok=True)
                Path(req.out).write_text(json.dumps(out, indent=1))
            except Exception as e:
                out["out_write_error"] = repr(e)
        with LOG.open("a") as f:
            f.write(json.dumps({"t": time.time(), "label": req.label, "ok": out["ok"],
                                "elapsed_s": out["elapsed_s"], "error": (err or "")[-2000:]}) + "\n")
        return out


if __name__ == "__main__":
    token_file = token_path(PORT)
    mine = write_token(token_file)
    print(f"[labd] this launch's token: {token_file}", flush=True)
    try:
        uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
    finally:
        with contextlib.suppress(OSError):
            if token_file.read_text().strip() == mine:  # not a later launch's
                token_file.unlink()

#!/usr/bin/env python3
"""Lab daemon: keeps Gemma-4-31B (W4A16) + J++/JLens loaded in ONE GPU process and executes
submitted experiment code against it, serialized by a lock.

POST /exec  {"code": "...", "label": "..."}  -> {"ok", "stdout", "result", "error", "elapsed_s"}
  Code runs with top-level await allowed, in a fresh namespace containing:
    rt (TheodoliteRuntime), lab (labkit.Lab), torch, np, F, STORE (persistent dict shared across calls)
  Assign to a variable named `result` to return JSON-serializable data.
POST /reload  -> reload labkit (after editing labkit.py)
GET  /health
Binds 127.0.0.1 only.
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
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)  # experiment scripts use repo-relative paths
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from backend.engine_runtime import TheodoliteRuntime
import labkit

LOG = REPO / "experiments" / "results" / "labd_exec_log.local.jsonl"  # gitignored; the study's log is labd_exec_log_2026-10-16.jsonl
app = FastAPI()
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


@app.post("/reload")
async def reload():
    async with LOCK:
        importlib.reload(labkit)
        STATE["lab"] = labkit.Lab(STATE["rt"])
        return {"ok": True}


@app.post("/exec")
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
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("LABD_PORT", "8765")), log_level="warning")

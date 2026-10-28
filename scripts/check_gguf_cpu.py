#!/usr/bin/env python3
"""A CPU check of the GGUF and CPU path's second layer (ARCHITECTURE §5: backend/runtime_factory.py and its
callers) on the tiny random Gemma 4 of scripts/atlas_stand_in.py written to tiny GGUF files, as
scripts/check_gguf.py writes them:
1. the entry points build their runtime through runtime_factory.make_runtime: with nothing set, the pinned
   TheodoliteRuntime as each built it before (the lab daemon's with both lenses, the standalone scientist's without the
   Jacobian lens); on a GGUF on the CPU, the standalone scientist (backend/mcp_scientist.py) and the lab daemon
   (experiments/lab/labd.py) load a GGUFRuntime, whose /health and /exec replies (and the exec log) name its model id
   and weights card (`weights_card`: a GGUF's file provenance, the checkpoint's repo id and the snapshot a cache laid
   out as the hub's holds, the stand-in's); none of them, nor the two verify scripts, builds the pinned runtime itself;
2. the files R4 pins (backend/engine_runtime.py among them) are unchanged (experiments/scripts/R4_common.pinned_differ).

Run: .venv/bin/python -I scripts/check_gguf_cpu.py   (exits 0 when every check holds)
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("TQDM_DISABLE", "1")

import asyncio
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
import torch.nn.functional as F

import atlas_stand_in as asi
from backend import engine_runtime as er
from backend import gguf_model as gm
from backend import gguf_weights as gw
from backend import runtime_factory as rf
from backend.causal_check import language_model
from backend.gguf_runtime import GGUFRuntime

failures: List[str] = []
passed = 0
BASE = "http://127.0.0.1:8765"  # the guards refuse TestClient's own "testserver" Host
ENV_KEYS = (rf.GGUF_ENV, rf.DEVICE_ENV, rf.LINEARS_ENV, rf.CONFIG_ENV, rf.DTYPE_ENV, rf.LENS_ENV, "CUDA_VISIBLE_DEVICES")


def check(ok: bool, what: str) -> None:
    global passed
    if ok:
        passed += 1
    else:
        failures.append(what)
        print(f"  FAIL  {what}")


def section(title: str, before: int) -> None:
    if len(failures) == before:
        print(f"  ok    {title}")


class environment:
    """The runtime's variables set as given (None removes one) inside the block, as they were after it."""

    def __init__(self, **values: Any) -> None:
        self.values = values

    def __enter__(self) -> "environment":
        self.saved = {k: os.environ.get(k) for k in ENV_KEYS}
        for k in ENV_KEYS:
            os.environ.pop(k, None)
        for k, v in self.values.items():
            if v is not None:
                os.environ[k] = str(v)
        return self

    def __exit__(self, *_exc: Any) -> None:
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def gguf_env(path: Path, **more: Any) -> Dict[str, Any]:
    """A GGUF on the CPU in float32 without the lens, the stand-in's checks' setting."""
    return {rf.GGUF_ENV: path, rf.DEVICE_ENV: "cpu", rf.DTYPE_ENV: "float32", rf.LENS_ENV: "none", **more}


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="gguf-cpu-check-"))
    try:
        return run_checks(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_checks(tmp: Path) -> int:
    torch.set_grad_enabled(False)
    tok = asi.stand_in_tokenizer()
    hf = asi.stand_in_hf(tok)
    files = {qt: gm.write_gemma4_gguf(hf, tmp / f"stand-in-{qt}.gguf", tokenizer=tok, qtype=qt) for qt in ("Q4_0", "Q8_0")}

    # ---------------------------------------------------------- 1. the entry points, through runtime_factory
    before = len(failures)
    with environment():
        for jl in (True, False):
            rt0 = rf.make_runtime(load_jlens=jl)
            check(type(rt0) is er.TheodoliteRuntime and str(rt0.device) == "cuda:0" and rt0.load_jlens is jl
                  and rt0.model_id == er.DEFAULT_W4A16_MODEL_ID,
                  f"nothing set, load_jlens={jl}: TheodoliteRuntime(load_jlens={jl}) on cuda:0, as labd (True) and the "
                  "standalone scientist and verify_causal_check (False) built it")
    import backend.mcp_scientist as ms

    bound = (ms._SHARED_RUNTIME, ms._SHARED_VAULT)
    try:
        ms._SHARED_RUNTIME, ms._SHARED_VAULT = None, None
        with environment(**gguf_env(files["Q4_0"])):
            rt_ms, vault_ms = ms.get_runtime_and_vault()
        check(isinstance(rt_ms, GGUFRuntime) and rt_ms.model_id.startswith("gguf:stand-in-Q4_0.gguf#")
              and rt_ms.model_id.endswith("@cpu/float32") and vault_ms.runtime is rt_ms,
              f"the standalone scientist on THEODOLITE_GGUF: a GGUFRuntime on the CPU ({rt_ms.model_id})")
        check(ms.get_runtime_and_vault()[0] is rt_ms, "and the same one on the next call")
    finally:
        ms._SHARED_RUNTIME, ms._SHARED_VAULT = bound

    # the lab daemon: its startup and replies, the model loaded by make_runtime on the GGUF
    sys.path.insert(0, str(ROOT_DIR / "experiments" / "lab"))
    cwd = os.getcwd()
    import labd  # changes to the repository's root, as it does when run
    import labkit
    from fastapi.testclient import TestClient

    os.chdir(cwd)
    labd.LOG = tmp / "labd_exec_log.jsonl"  # not the repository's
    gender = labkit.Lab._build_gender_ids
    labkit.Lab._build_gender_ids = lambda self: {}  # the stand-in's vocabulary has not the study's paired words
    try:
        with environment(**gguf_env(files["Q4_0"])):
            asyncio.run(labd._startup())
        rt_l = labd.STATE["rt"]
        lc = TestClient(labd.app, base_url=BASE)
        h = lc.get("/health").json()
        check(isinstance(rt_l, GGUFRuntime) and h["ready"] and h["model"] == rt_l.model_id and h["device"] == "cpu"
              and h["weights"] == rf.weights_card(rt_l) and h["weights"]["format"] == "gguf"
              and h["weights"]["id"] == rt_l.weights_card["id"] and h["weights"]["layers_as"] == {"Q4_0 packed": 41},
              f"labd on THEODOLITE_GGUF: /health names the model id and the weights card ({h['model']}, {h['weights'].get('id')})")
        token = labd.write_token(tmp / "labd.token")
        r = lc.post("/exec", json={"code": "result = [rt.model_id, type(rt).__name__]", "label": "which"},
                    headers={"Authorization": f"Bearer {token}"}).json()
        logged = json.loads(labd.LOG.read_text().splitlines()[-1])
        check(r["ok"] and r["result"] == [rt_l.model_id, "GGUFRuntime"] and r["model"] == rt_l.model_id
              and r["weights"]["id"] == rt_l.weights_card["id"] and logged["model"] == rt_l.model_id,
              "an /exec reply, and its line in the log, name the model id; the reply the weights card too")
    finally:
        labkit.Lab._build_gender_ids = gender
        labd.STATE.clear()
        labd.TOKEN["value"] = None

    # the weights card of each kind of runtime
    import huggingface_hub.constants as hubc

    snap = "0123456789abcdef0123456789abcdef01234567"
    repo_dir = tmp / "hub" / "models--google--gemma-4-31B-it-qat-w4a16-ct"
    (repo_dir / "snapshots" / snap).mkdir(parents=True)
    (repo_dir / "snapshots" / snap / "config.json").write_text("{}")
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text(snap)
    hub_cache = hubc.HF_HUB_CACHE
    try:
        hubc.HF_HUB_CACHE = str(tmp / "hub")
        card_ct = rf.weights_card(er.TheodoliteRuntime(load_jlens=False))
        cpu_ct = rf.CheckpointRuntime(device="cpu", load_jlens=False)
        cpu_ct.model_id += rf.reading_class(cpu_ct.device)
        card_cpu = rf.weights_card(cpu_ct)
        hubc.HF_HUB_CACHE = str(tmp / "no-hub")
        card_none = rf.weights_card(er.TheodoliteRuntime(load_jlens=False))
    finally:
        hubc.HF_HUB_CACHE = hub_cache
    check(card_ct["format"] == "compressed-tensors W4A16" and card_ct["repo"] == er.DEFAULT_W4A16_MODEL_ID
          and card_ct["snapshot"] == snap and card_ct["id"] == snap[:16] and card_ct["device"] == "cuda:0",
          f"the checkpoint's card: its repo id and the snapshot the cache's main ref names ({card_ct['id']})")
    check(card_cpu["repo"] == er.DEFAULT_W4A16_MODEL_ID and card_cpu["device"] == "cpu" and card_cpu["runtime"] == "CheckpointRuntime",
          "on the CPU: the repo id without the reading class, the device the CPU")
    check(card_none["snapshot"] is None and card_none["id"] is None, "a checkpoint not cached here: no snapshot, no id")
    st_rt, _ = asi.stand_in()
    check(rf.weights_card(st_rt) == {"format": "stand-in", "model": asi.STAND_IN_MODEL_ID, "runtime": "TheodoliteRuntime"},
          "the stand-in's card says it is the stand-in")
    for rel in ("backend/mcp_scientist.py", "experiments/lab/labd.py", "scripts/verify_optics.py", "scripts/verify_causal_check.py"):
        src = (ROOT_DIR / rel).read_text(encoding="utf-8")
        direct = re.findall(r"=\s*(?:\w+\.)?TheodoliteRuntime\(", src)
        check("make_runtime(load_jlens=" in src and not direct, f"{rel} builds its runtime through make_runtime, none directly {direct}")
    section("the standalone scientist, the lab daemon and the verify scripts build their runtime through make_runtime; "
            "labd's replies name the weights", before)

    # ---------------------------------------------------------------- the files R4 pins
    before = len(failures)
    sys.path.insert(0, str(ROOT_DIR / "experiments" / "scripts"))
    import R4_common

    check(R4_common.pinned_differ() == [], f"the files R4 pins (engine_runtime.py among them) are unchanged: {R4_common.pinned_differ()}")
    section("the R4-pinned runtime untouched: everything here is in new or unpinned modules", before)

    print(f"\n{passed} checks passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

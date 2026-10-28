#!/usr/bin/env python3
"""A CPU check of the GGUF and CPU path's second layer (ARCHITECTURE §5: backend/runtime_factory.py, the lab's
gradients, the CPU's memory) on the tiny random Gemma 4 of scripts/atlas_stand_in.py written to tiny GGUF files, as
scripts/check_gguf.py writes them:
1. the entry points build their runtime through runtime_factory.make_runtime: with nothing set, the pinned
   TheodoliteRuntime as each built it before (the lab daemon's with both lenses, the standalone scientist's without the
   Jacobian lens); on a GGUF on the CPU, the standalone scientist (backend/mcp_scientist.py) and the lab daemon
   (experiments/lab/labd.py) load a GGUFRuntime, whose /health and /exec replies (and the exec log) name its model id
   and weights card (`weights_card`: a GGUF's file provenance, the checkpoint's repo id and the snapshot a cache laid
   out as the hub's holds, the stand-in's); none of them, nor the two verify scripts, builds the pinned runtime itself;
2. labkit's GGUF twin of its W4A16 autograd function: `same_pos_vjp` (the lens fits' true Jacobian products) through
   packed and resident Q4_0 layers equals the dense stand-in's carrying the same numbers, bit for bit in float32 and
   in bf16, with no weight kept for backward (the layers' own forward gives the same gradients, keeping weights), and
   with a vault column in force equals the edited stand-in's; the Lab names the model id and weights card; a W4A16
   module through the study's function and a Q4_0 layer of the same codes give the unpacked weight's gradients, and
   every layer's own forward is back after;
3. the CPU's memory (backend/cpu_memory.py): the estimates from the stand-in's config grow as they should, count a
   packed layer's unpacking and nothing for plain or resident ones, and are never asked on a GPU; with MemAvailable
   patched small, a sweep (through the server's socket), the vault's voice (POST /api/larql, no edit made), a census
   batch (nothing written), a steer lens batch and a causal check are refused in plain words, and a causal check with
   room for one pushed row a forward is split before its first, its readings unchanged; a CPU allocator's failure, as
   torch raises it, is answered in plain words by the sweep (with and without a request id, the socket kept), the
   voice, an atlas reading, the causal check (which halves as on the card) and the scientist's tool, and by
   census_run.py and atlas_build.py as a one-line exit; torch's own failure is recognised, and the card's is not taken
   for the CPU's;
4. the files R4 pins (backend/engine_runtime.py among them) are unchanged (experiments/scripts/R4_common.pinned_differ).

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


def raises(fn: Any, *args: Any, **kw: Any) -> str:
    """What `fn` raised, in words; "" when it raised nothing."""
    try:
        fn(*args, **kw)
    except Exception as exc:  # noqa: BLE001
        return str(exc) or type(exc).__name__
    return ""


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

    # ------------------------------------------------- 2. labkit's GGUF twin of its W4A16 autograd function
    before = len(failures)
    import check_gguf as cg  # its stand-in carrying a file's dequantised numbers

    labkit.Lab._build_gender_ids = lambda self: {}
    twin = labkit._GGUFLinearFn
    try:
        ids = [tok.bos_token_id] + tok.encode(asi.CORPUS[4], add_special_tokens=False)[:20]
        pos = len(ids) - 1
        d_final = torch.randn(asi.HIDDEN, generator=torch.Generator().manual_seed(3))
        for dtype in (torch.float32, torch.bfloat16):
            ref = cg.stand_in_with(tok, files["Q4_0"])  # float32, the file's numbers
            if dtype == torch.bfloat16:
                for p_ in ref.parameters():  # as check_gguf's bf16 stand-in: the rotary frequencies stay float32
                    p_.data = p_.data.to(torch.bfloat16)
                for layer in ref.model.layers:
                    layer.layer_scalar = layer.layer_scalar.to(torch.bfloat16)
            lab_s = labkit.Lab(asi.stand_in_runtime(ref, tok))
            g_s = lab_s.same_pos_vjp(ids, pos, d_final)
            name = str(dtype).replace("torch.", "")
            for mode in ("packed", "resident"):
                rt_g = GGUFRuntime(files["Q4_0"], device="cpu", dtype=dtype, tokenizer=tok, linears=mode, load_jpp=False).load()
                lab_g = labkit.Lab(rt_g)
                ggl = [m for m in rt_g.hf_model.modules() if isinstance(m, gw.GGUFLinear)]
                kept: List[Any] = []
                with torch.autograd.graph.saved_tensors_hooks(lambda t: kept.append(tuple(t.shape)) or t, lambda t: t):
                    g_g = lab_g.same_pos_vjp(ids, pos, d_final)
                shapes = {(m.out_features, m.in_features) for m in ggl} | {(m.in_features, m.out_features) for m in ggl}
                check(g_g.shape == (6, asi.HIDDEN) and torch.equal(g_g, g_s),
                      f"{name}, {mode}: same_pos_vjp through the GGUF twin equals the dense stand-in's, bit for bit")
                check(len(ggl) == 41 and not any(s in shapes for s in kept) and all("forward" not in m.__dict__ for m in ggl),
                      f"{name}, {mode}: no weight kept for backward ({len(kept)} tensors kept, none a weight), and every "
                      "layer's own forward back after it")
                check(lab_g.model_id == rt_g.model_id and lab_g.weights["format"] == "gguf" and lab_g.weights["linears"] == mode,
                      f"{name}, {mode}: the Lab names the model id and weights card ({lab_g.weights['layers_as']})")
            if dtype == torch.float32:
                # without the twin, the packed layers' own forward: the same gradients, every weight kept for backward
                rt_p = GGUFRuntime(files["Q4_0"], device="cpu", dtype=dtype, tokenizer=tok, load_jpp=False).load()
                labkit._GGUFLinearFn = type("PlainForward", (), {"apply": staticmethod(lambda x, m: gw.GGUFLinear.forward(m, x))})
                kept = []
                with torch.autograd.graph.saved_tensors_hooks(lambda t: kept.append(tuple(t.shape)) or t, lambda t: t):
                    g_p = labkit.Lab(rt_p).same_pos_vjp(ids, pos, d_final)
                labkit._GGUFLinearFn = twin
                check(torch.equal(g_p, g_s) and sum(s in shapes for s in kept) > 0,
                      f"the packed layers' own forward gives the same gradients but keeps {sum(s in shapes for s in kept)} "
                      "weights for backward")
                # an edit in force, heard by the twin as by the layer's own forward
                v = torch.randn(asi.HIDDEN, generator=torch.Generator().manual_seed(4))
                language_model(rt_p.hf_model).layers[2].mlp.down_proj._memit_col_overlays[7] = v.clone()
                edited = cg.stand_in_with(tok, files["Q4_0"])
                edited.model.layers[2].mlp.down_proj.weight.data[:, 7] = v
                check(torch.equal(labkit.Lab(rt_p).same_pos_vjp(ids, pos, d_final),
                                  labkit.Lab(asi.stand_in_runtime(edited, tok)).same_pos_vjp(ids, pos, d_final)),
                      "a vault column in force: the twin's gradients are the edited stand-in's")
        # the W4A16 function as the study ran it, beside a GGUF layer in one model: each layer's own forward back after
        g2 = torch.Generator().manual_seed(2)
        codes = torch.randint(0, 16, (64, 128), generator=g2, dtype=torch.uint8)
        scale = (2.0 ** -torch.randint(4, 9, (64, 4), generator=g2).float()).to(torch.bfloat16)
        ct = cg.W4A16Module(cg.cmp.ct_pack(codes), scale)
        pinned_fwd = ct.forward
        q40 = gw.GGUFLinear(torch.from_numpy(cg.cmp.q4_0_blocks(codes, scale.float())), "Q4_0", 128, 64, dtype=torch.bfloat16)
        holder = type("Holder", (), {"hf": torch.nn.ModuleList([ct, q40])})()
        x = torch.randn(3, 128, generator=g2).to(torch.bfloat16)
        grads = []
        with torch.enable_grad(), labkit._grad_linears(holder):
            for mod in (ct, q40):
                xi = x.clone().requires_grad_(True)
                mod(xi).float().pow(2).sum().backward()
                grads.append(xi.grad)
        w = er.dequantize_w4a16_full(ct.weight_packed, ct.weight_scale)
        xr = x.clone().requires_grad_(True)
        with torch.enable_grad():
            F.linear(xr, w).float().pow(2).sum().backward()
        check(torch.equal(grads[0], xr.grad) and torch.equal(grads[1], xr.grad) and ct.forward is pinned_fwd
              and "forward" not in q40.__dict__,
              "a W4A16 module (the study's function) and a Q4_0 layer of the same codes: the same input gradients as the "
              "unpacked weight's; the pinned forward, and the layer's own, back after")
    finally:
        labkit.Lab._build_gender_ids = gender
        labkit._GGUFLinearFn = twin
    section("labkit's GGUF twin: gradients through packed and resident GGUF layers are the dense stand-in's, bit for bit "
            "(float32 and bf16), with no weight kept for backward", before)

    # ------------------------------------------------------- 3. the CPU's memory, refused in plain words
    before = len(failures)
    import backend.atlas as atlas_routes
    import backend.atlas_build as ab
    import backend.causal_check as cc
    import backend.cpu_memory as cm
    from backend import census as cz
    from fastapi import HTTPException
    from fastapi.testclient import TestClient

    rt_c, vault_c = asi.stand_in()
    lm_c = language_model(rt_c.hf_model)
    d_c = cm.dims(rt_c)
    check(d_c == cm.Dims(6, asi.HIDDEN, asi.FFN, rt_c.hf_model.config.vocab_size, rt_c.hf_model.config.num_attention_heads,
                         d_c.kv_piece, 4) and d_c.kv_piece > 0 and cm.dims(object()) is None,
          f"the stand-in's sizes from its config ({d_c}); none for an object without one")
    w1, w2 = cm.writing_need(rt_c, 1, 10, 5), cm.writing_need(rt_c, 2, 10, 5)
    check(0 < w1 < w2 < cm.writing_need(rt_c, 2, 10, 50) and cm.writing_need(rt_c, 1, 10, 5, full_logits=True) > w1
          and cm.sweep_need(rt_c, 20, thinking=True) > cm.sweep_need(rt_c, 20) > 0
          and 0 < cm.forward_need(rt_c, 2, 8) < cm.forward_need(rt_c, 4, 8) < cm.forward_need(rt_c, 4, 8, cached=30),
          "the estimates grow with rows, word-pieces, logits kept, the thinking comparison and a cache")
    rt_pk = GGUFRuntime(files["Q4_0"], device="cpu", dtype=torch.float32, tokenizer=tok, load_jpp=False).load()
    rt_rs = GGUFRuntime(files["Q4_0"], device="cpu", dtype=torch.float32, tokenizer=tok, linears="resident", load_jpp=False).load()
    widest = max(m.out_features * m.in_features for m in rt_pk.hf_model.modules() if isinstance(m, gw.GGUFLinear))
    check(cm.unpack_bytes(rt_c) == 0 and cm.unpack_bytes(rt_rs) == 0 and cm.unpack_bytes(rt_pk) >= widest * 4,
          f"a packed layer's unpacking counted ({cm.unpack_bytes(rt_pk)} bytes), none for plain or resident layers")
    gpu_like = type("OnTheCard", (), {"device": torch.device("cuda:0")})()
    cm.ensure(gpu_like, 1 << 60, "A reading on the card", "none")  # never asked off the CPU
    true_available = cm.available
    cm.available = lambda: 1 << 10  # 1 KiB available: the estimate of every reading below on the stand-in is larger
    try:
        try:
            cm.ensure(rt_c, lambda: 4 << 20, "This reading", "make it smaller")
            said, kind = "", None
        except ValueError as exc:
            said, kind = str(exc), type(exc)
        check(said.startswith("This reading needs about 4.0 MiB of memory on the CPU (estimated: backend/cpu_memory.py), "
                              "and 1 KiB is available (MemAvailable): make it smaller. Nothing was read.")
              and kind is cm.CPUMemoryRefused, f"refused in plain words, a ValueError: {said}")
        os.environ[cm.CPU_MEMORY_ENV] = "off"
        check(raises(cm.ensure, rt_c, 4 << 20, "This reading", "make it smaller") == "", f"{cm.CPU_MEMORY_ENV}=off: not asked")
        os.environ.pop(cm.CPU_MEMORY_ENV)

        app = asi.bind_server(rt_c, vault_c)
        with TestClient(app, base_url=BASE) as c:
            with c.websocket_connect("ws://127.0.0.1:8765/ws/theodolite") as ws:
                ws.send_json({"action": "sweep", "payload": {"prompt": "Who are you?", "max_new_tokens": 4, "request_id": "m1"}})
                got = [ws.receive_json()]
                while got[-1].get("type") != "theodolite_sweep_end":
                    got.append(ws.receive_json())
                err = next((g for g in got if g.get("type") == "error"), {})
                check(err.get("request_id") == "m1" and err.get("message", "").startswith("This sweep needs about")
                      and "MemAvailable" in err["message"] and "shorten the question" in err["message"]
                      and got[-1]["n_sent"] == 0, f"a sweep on the CPU refused before it is written: {err.get('message')}")
            r = c.post("/api/larql", json={"query": "DELETE FROM LAYER 2 WHERE slot = 5",
                                            "voice": {"prompt": "Who are you?", "max_new_tokens": 4}})
            check(r.status_code == 400 and r.json()["detail"].startswith("The voice before the edit could not be written: "
                                                                           "The voice needs about")
                  and r.json()["detail"].endswith("no edit was made.") and not vault_c.active_patches,
                  f"the vault's voice refused, and no edit made: {r.json().get('detail', '')[:160]}")
            cm.available = true_available
            # a CPU allocator's failure, as torch raises it, in plain words: the sweep (with and without a request id,
            # the socket kept), the voice, an atlas reading
            alloc = RuntimeError("[enforce fail at alloc_cpu.cpp:127] err == 0. DefaultCPUAllocator: can't allocate memory: "
                                 "you tried to allocate 3221225472 bytes. Error code 12 (Cannot allocate memory)")

            def full(*_a: Any, **_k: Any) -> Any:
                raise alloc

            capture, write_voice = rt_c.capture_for_sweep, rt_c.write_voice
            rt_c.capture_for_sweep, rt_c.write_voice = full, full
            try:
                with c.websocket_connect("ws://127.0.0.1:8765/ws/theodolite") as ws:
                    ws.send_json({"action": "sweep", "payload": {"prompt": "Who are you?", "max_new_tokens": 4, "request_id": "m2"}})
                    got = [ws.receive_json()]
                    while got[-1].get("type") != "theodolite_sweep_end":
                        got.append(ws.receive_json())
                    err = next((g for g in got if g.get("type") == "error"), {})
                    ws.send_json({"action": "sweep", "payload": {"prompt": "Who are you?", "max_new_tokens": 4}})
                    old = ws.receive_json()
                    ws.send_json({"action": "larql", "query": "DESCRIBE"})
                    after = ws.receive_json()
                check(err.get("message") == "The CPU ran out of memory during this sweep (one allocation of 3.0 GiB was "
                                            "refused). Try a shorter question or answer, or sight fewer word-pieces.",
                      f"the allocator's failure during a sweep, in plain words: {err.get('message')}")
                check(old.get("type") == "error" and old.get("recoverable") is True
                      and old.get("message", "").startswith("The CPU ran out of memory during this sweep")
                      and after.get("type") == "larql_result", f"without a request id too, the socket kept ({old}, then {after.get('type')})")
                r = c.post("/api/larql", json={"query": "DELETE FROM LAYER 2 WHERE slot = 5",
                                                "voice": {"prompt": "Who are you?", "max_new_tokens": 4}})
                check(r.status_code == 400 and "could not be written: The CPU ran out of memory during the writing (one "
                      "allocation of 3.0 GiB was refused); no edit was made." in r.json()["detail"] and not vault_c.active_patches,
                      f"the allocator's failure during the voice: {r.json().get('detail')}")
            finally:
                rt_c.capture_for_sweep, rt_c.write_voice = capture, write_voice
        try:
            asyncio.run(atlas_routes.on_gpu(full))
            check(False, "an atlas reading that ran out of memory on the CPU refused")
        except HTTPException as exc:
            check(exc.status_code == 400 and exc.detail == "The CPU ran out of memory during this reading (one allocation of "
                  "3.0 GiB was refused). Try a shorter text.", f"an atlas reading's allocator failure, in plain words: {exc.detail}")
        cm.available = lambda: 1 << 10
        out = tmp / "census-m.jsonl"
        said = raises(cz.take_census, rt_c, 2, [0, 1], out, p50=1.5, p50_source="fixed", controls=1, seed=0, max_new=6, top_k=3,
                      batch=5, log=lambda _s: None)
        check(said.startswith("A census batch of 5 answers of up to 6 word-pieces needs about") and "lower --batch or --max-new" in said
              and not out.exists() and not cz.meta_path_of(out).exists(), f"a census batch refused, nothing written: {said[:100]}")
        said = raises(ab.build_steer_layer, rt_c, 2, 1.5, batch=8)
        check(said.startswith("A steer lens batch of 8 neurons needs about") and "lower --batch" in said, f"a steer lens batch refused: {said[:90]}")
        # the causal check: refused when one pushed row does not fit; split before its first forward when all do not
        ids_c = [tok.bos_token_id] + tok.encode(asi.CORPUS[4], add_special_tokens=False)[:16]
        down = lm_c.layers[2].mlp.down_proj
        args = (lm_c, 2, 5, [3.0, -3.0], "activation", down.weight[:, 5].detach(), lm_c.layers[2].post_feedforward_layernorm.weight.detach(),
                float(lm_c.layers[2].post_feedforward_layernorm.eps), rt_c.lm_head_weight)
        said = raises(cc.run_check, *args, token_ids=ids_c)
        check(said.startswith("The causal check's forward (its base row and one pushed row) needs about") and "shorter text" in said,
              f"a causal check refused: {said[:100]}")
        n_c = len(ids_c)
        cm.available = lambda: (cm.forward_need(lm_c, 2, 1, cached=n_c) + cm.forward_need(lm_c, 3, 1, cached=n_c)) // 2
        split = cc.run_check(*args, token_ids=ids_c)
        cm.available = true_available
        whole = cc.run_check(*args, token_ids=ids_c)
        check(split.slots_per_forward == [1, 1] and whole.slots_per_forward == [2]
              and all(torch.allclose(a_, b_, atol=1e-5) for a_, b_ in zip(split.steered, whole.steered)),
              f"room for one pushed row a forward: split before the first ({split.slots_per_forward}), the same readings")

        def full_rows(_m: Any, args_: Any) -> None:
            if args_[0].shape[0] > 2:
                raise alloc

        hk = lm_c.layers[0].register_forward_pre_hook(full_rows)
        try:
            halved = cc.run_check(*args, token_ids=ids_c, kv_cache=False)
        finally:
            hk.remove()
        check(halved.slots_per_forward == [1, 1], f"the allocator's failure halves the pushed rows as the card's does ({halved.slots_per_forward})")
        hk = lm_c.layers[0].register_forward_pre_hook(lambda *_a: full())
        import backend.mcp_scientist as ms

        bound = (ms._SHARED_RUNTIME, ms._SHARED_VAULT)
        ms.bind_runtime(rt_c, vault_c)
        try:
            said = raises(asyncio.run, ms.causal_spot_check(layer=2, neuron_idx=5, contexts="given", prompt="Who are you?"))
        finally:
            hk.remove()
            ms._SHARED_RUNTIME, ms._SHARED_VAULT = bound
        check(said.startswith("The CPU ran out of memory during this test, even with its pushed rows split"),
              f"the scientist's causal check: the allocator's failure in plain words ({said[:90]})")
    finally:
        cm.available = true_available
        os.environ.pop(cm.CPU_MEMORY_ENV, None)
    real = ""
    try:
        torch.empty(1 << 46, dtype=torch.uint8)
    except RuntimeError as exc:
        real = cm.plain(exc, "this reading") if cm.is_cpu_oom(exc) else f"not recognised: {exc}"
    check(real == "The CPU ran out of memory during this reading (one allocation of 65536.0 GiB was refused).",
          f"torch's own allocator failure is recognised: {real}")
    check(cm.is_oom(torch.cuda.OutOfMemoryError("x")) and not cm.is_cpu_oom(torch.cuda.OutOfMemoryError("x"))
          and not cm.is_oom(RuntimeError("shapes do not match")), "a card's out of memory is not the CPU's; another RuntimeError is neither")
    # the command lines: an allocator failure is a one-line exit in plain words, not a traceback
    import census_run

    take = cz.take_census
    cz.take_census = full
    os.environ["THEODOLITE_ATLAS_DIR"] = str(tmp / "atlas-cli")
    try:
        census_run.main(["--stand-in", "--layer", "2", "--neurons", "0-1", "--random-controls", "1", "--max-new", "4", "--no-rank"])
        said = "no exit"
    except SystemExit as exc:
        said = str(exc)
    finally:
        cz.take_census = take
    check(said.startswith("The CPU ran out of memory during the census (one allocation of 3.0 GiB was refused). Every whole "
                          "batch before it is kept"), f"census_run.py: {said[:120]}")
    import atlas_build as atlas_build_cli  # scripts/atlas_build.py

    layout, argv = ab.build_layout, sys.argv
    ab.build_layout = full
    sys.argv = ["atlas_build.py", "layout", "--stand-in", "--out", str(tmp / "atlas-cli" / "index")]
    try:
        atlas_build_cli.run()
        said = "no exit"
    except SystemExit as exc:
        said = str(exc)
    finally:
        ab.build_layout, sys.argv = layout, argv
    check(said.startswith("The CPU ran out of memory during the build (one allocation of 3.0 GiB was refused). A build "
                          "written a part or a layer at a time resumes"), f"atlas_build.py: {said[:120]}")
    section("the CPU's memory: a sweep, a census batch, the vault's voice, a causal check and a steer lens batch refused in "
            "plain words when MemAvailable cannot hold their estimate; an allocator's failure answered in the same words", before)

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

#!/usr/bin/env python3
"""A CPU-only check of the atlas's routes (backend/atlas.py) and its field notes (backend/field_notes.py), over the
instrument's own FastAPI app with the tiny random Gemma 4 of scripts/atlas_stand_in.py bound in place of the model.
Every reading is compared with the same operation of backend/atlas_ops.py called directly:
1. the shape: GET /arch, and /api/health's atlas card just before the label gate and the lens;
2. who may ask: a page served from another machine (Host, Origin) or marked cross-site is refused, a plain link is not;
3. before any index is built: what needs one says so (ready / available false, an empty cone) and the wiring is read
   live from the weights;
4. the live readings: both faces' words (the vault's own reading, each face's label check), the steer lens read live,
   the push chat's three voices (one neuron, several, a conversation carried on, refusals before the stream opens), the
   whole-text steer, reads (pushes and head scales in the same pass; a push at or after the neuron's own block leaves
   it as it was), the next piece's odds (KL, what rose and fell), compare, fires (the whole text and one unit);
5. the run view: scores at a word-piece and at each neuron's strongest, the filters, the map; heads' writes; one
   head's row, its pattern, its wiring, its profile, its attention; the drive of a neuron, whose parts add up;
6. with every index built (backend/atlas_build.py into the temporary atlas folder): the scales come from the stats
   index, the tooltip and the wiring from the indexes, the search finds a face by its first word-piece, the cone and
   the web come back as the index has them, the layout and the active paths; a face's stored passages (the index's
   events in order, the peak unit marked, each side cut where the window starts or ends, at a boundary between
   documents (an end mark, or a start mark inside the window) or a turn mark, or where the count runs out, `rel`, the
   regime, reads and follows counted with their lift; `rel_to` naming the stats index's corpus, or saying it is
   another), refusals in plain words, and /reads given a passage's ids reading it again as the index has it;
7. a vault edit in force: the wiring is read live again, the faces say their index words are stale, the caches keyed by
   the edits in force are not reused, the passages of the edited neuron and of later layers are marked stale;
8. the field notes: kept, shown on the tooltip and the wiring, forgotten, refused in plain words, a corrupt file moved
   aside with one warning;
9. the census routes on a run written here: its units ranked, a unit's rollouts and baselines, one rollout, the spread;
10. the GPU lock: a reading waits for it while /arch and the notes still answer; a push chat closed mid-answer stops
    writing and holds the lock until its writer has stopped (uvicorn on a free port, a real connection closed).

Run: .venv/bin/python -I scripts/check_atlas_api.py   (exits 0 when every check holds)
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"

import asyncio
import base64
import json
import shutil
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

TMP = Path(tempfile.mkdtemp(prefix="atlas-api-"))
os.environ["THEODOLITE_ATLAS_DIR"] = str(TMP / "atlas")
os.environ["THEODOLITE_FIELD_NOTES"] = str(TMP / "notes" / "field_notes.json")
os.environ["THEODOLITE_CENSUS_DIR"] = str(TMP / "census")

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch

import backend.atlas as atlas_routes
import backend.atlas_build as ab
import backend.atlas_ops as ao
from atlas_stand_in import CORPUS, bind_server, stand_in
from backend import census as cz
from backend.atlas_index import AtlasIndex, index_dir, scale_scores
from backend.causal_check import language_model
from backend.label_gate import LABEL_GATE_CARD
from backend.larql_vault import PatchOperationRecord
from backend.mcp_scientist import _study_ids

failures: List[str] = []
passed = 0
BASE = "http://127.0.0.1:8765"


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


def quiet(_s: str) -> None:
    pass


def events(body: str) -> List[Dict[str, Any]]:
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]


def voices(evs: List[Dict[str, Any]]) -> List[str]:
    out = ["", "", ""]
    for e in evs:
        if e.get("row") is not None:
            out[e["row"]] += e["t"]
    return out


def close(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


def main() -> int:
    from fastapi.testclient import TestClient

    torch.set_grad_enabled(False)
    rt, vault = stand_in()
    app = bind_server(rt, vault)
    import backend.server as server

    lm = language_model(rt.hf_model)
    w = ao.Weights(lm, rt)
    L, N, D = w.n_layers, w.d_ff, w.d_model
    tok = rt.tokenizer
    W_U = rt.lm_head_weight
    bos = tok.bos_token_id
    cap = ao.softcap_of(rt.hf_model)
    shapes = [ao.attn_shape(lm, l) for l in range(L)]
    H = shapes[0].n_heads
    ex = ao.activation_extremes(lm, _study_ids(rt))
    p50 = ao.stream_p50(ex)
    act_scale = lambda l, n: max(abs(float(ex.pos[l, n])), abs(float(ex.neg[l, n])), 1e-6)  # noqa: E731
    chat_ids = lambda msgs: rt.tokenize_prompt(  # noqa: E731
        tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    vocab = [tok.decode([i]) for i in range(int(W_U.shape[0]))]

    def write_direct(rows: List[List[int]], pushes: List[ao.Push], max_new: int, scales: Any = None) -> List[str]:
        pp, sc = (p50, act_scale) if scales is None else scales
        iv = ao.resolve(w, pushes, [], ao.THREE_VOICES, p50=lambda l: pp[l], act_scale=sc)
        td = ao.TextDeltas(tok, 3)
        out = ["", "", ""]
        for st in ao.write_rows(lm, W_U, rows, iv, max_new=max_new, stop_ids=rt.end_of_turn_ids(), bos_id=bos,
                                spare_bos=True, softcap=cap):
            out[st.row] += td.add(st.row, st.token_id, st.end)
        return out

    c = TestClient(app, base_url=BASE)
    text = "Alice called Bob because she wanted to borrow his car. He said yes."
    ids_text = rt.tokenize_prompt(text)
    units_text = ao.token_units(tok, ids_text)

    # ---------------------------------------------------------------------------------------------------- 1. shape
    before = len(failures)
    r = c.get("/api/atlas/arch")
    arch = r.json()
    check(r.status_code == 200 and arch["n_layers"] == L and arch["d_ff"] == N and arch["d_model"] == D, f"/arch: {r.text[:200]}")
    check(arch["stand_in"] is True and arch["vocab_size"] == len(vocab), "/arch: stand-in and vocabulary")
    check(list(arch)[-2:] == ["label_gate", "lens"] and arch["label_gate"] == LABEL_GATE_CARD, "/arch ends with the label gate and the lens")
    check([s["kind"] for s in arch["layers"]] == [s.kind for s in shapes] and arch["layers"][-1]["values_from_keys"],
          "/arch: each layer's attention kind")
    check(all(not v["built"] for k, v in arch["indexes"].items() if k != "folder"), "/arch: no index built yet")
    check(arch["push"]["scale_source"] is None and arch["census"] == {}, "/arch: no scales read yet, no census")
    h = c.get("/api/health").json()
    check(list(h)[-3:] == ["atlas", "label_gate", "lens"], f"/api/health keys end {list(h)[-4:]}")
    check(h["atlas"]["indexes"] == {k: False for k in h["atlas"]["indexes"]} and h["atlas"]["routes"] == "/api/atlas",
          "/api/health: the atlas card says no index is on disk")
    check(h["atlas"]["field_notes"] == os.environ["THEODOLITE_FIELD_NOTES"], "/api/health: where the field notes are")
    section("the shape: /arch and the health card", before)

    # ------------------------------------------------------------------------------------------- 2. who may ask
    before = len(failures)
    body = {"layer": 1, "n": 3, "text": "hello there"}
    for hdrs, method, path, want in [
        ({"Origin": "http://evil.example"}, "GET", "/api/atlas/arch", 403),
        ({"Host": "evil.example"}, "GET", "/api/atlas/arch", 403),
        ({"Sec-Fetch-Site": "cross-site"}, "POST", "/api/atlas/reads", 403),
        ({"Sec-Fetch-Site": "same-site", "Sec-Fetch-Mode": "cors"}, "GET", "/api/atlas/arch", 403),
        ({"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "no-cors"}, "GET", "/api/atlas/layout", 403),
        ({"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate"}, "GET", "/api/atlas/arch", 200),
        ({"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate"}, "POST", "/api/atlas/reads", 403),
        ({"Origin": "http://localhost:5173", "Sec-Fetch-Site": "same-site", "Sec-Fetch-Mode": "cors"}, "GET", "/api/atlas/arch", 403),
        ({"Origin": "http://localhost:5173", "Sec-Fetch-Site": "same-origin"}, "POST", "/api/atlas/reads", 200),
        ({"Origin": "http://127.0.0.1:8765"}, "GET", "/api/atlas/arch", 200),
    ]:
        r = c.request(method, path, headers=hdrs, json=body if method == "POST" else None)
        check(r.status_code == want, f"{method} {path} with {hdrs}: {r.status_code}, wanted {want}")
        if want == 403:
            check("this machine" in r.json().get("detail", ""), f"{method} {path}: the refusal says why")
    # the peer itself, whatever its headers say: a client on another machine naming this one in Host is refused
    from fastapi import HTTPException
    from starlette.requests import Request as _Req

    def peer(addr: str) -> int:
        scope = {"type": "http", "method": "GET", "path": "/api/atlas/arch", "headers": [(b"host", b"127.0.0.1:8765")], "client": (addr, 5000)}
        try:
            atlas_routes.local_pages_only(_Req(scope))
            return 200
        except HTTPException as exc:
            return exc.status_code

    check([peer(a_) for a_ in ("192.168.1.20", "10.0.0.2", "127.0.0.1", "::1", "::ffff:127.0.0.1")] == [403, 403, 200, 200, 200],
          "who may ask: a peer that is not this machine is refused even with a local Host")
    section("who may ask: other machines and cross-site requests refused, a plain link let through", before)

    # --------------------------------------------------------------------------------- 3. before any index is built
    before = len(failures)
    hv = c.get("/api/atlas/hover", params={"layer": 2, "n": 5}).json()
    check(hv["ready"] is False and hv["faces"]["+"]["lens"] is None and hv["links"] is None, f"/hover unbuilt: {hv}")
    s = c.get("/api/atlas/search", params={"q": "the"}).json()
    check(s == {"ready": False, "results": [], "held": [], "held_back": 0, "tokens": 0, "total": 0}, f"/search unbuilt: {s}")
    r = c.get("/api/atlas/web", params={"layer": 2, "n": 5})
    check(r.status_code == 200 and r.content == b"" and r.headers["X-Web-Total"] == "0", "/web unbuilt: empty")
    check(c.get("/api/atlas/web/summary", params={"layer": 2, "n": 5}).json() == {"src": None, "ready": False, "missing": "wiring"},
          "/web/summary unbuilt: says the wiring index is missing")
    check(c.get("/api/atlas/layout").json() == {"available": False}, "/layout unbuilt")
    check(c.get("/api/atlas/cone", params={"layer": 2, "n": 5}).json()["ready"] is False, "/cone unbuilt")
    pr = c.get("/api/atlas/neuron/passages", params={"layer": 2, "n": 5}).json()
    check(pr["ready"] is False and "scripts/atlas_build.py passages" in pr["note"] and "passages" not in pr,
          f"/neuron/passages unbuilt: says which build makes it ({pr})")
    wr = c.get("/api/atlas/neuron/wiring", params={"layer": 2, "n": 5}).json()
    sc36 = ao.norm_scales(ao.activation_extremes(lm, _study_ids(rt)))
    live = ao.neuron_wiring(w, 2, 5, ao.link_scale(w.scalars, sc36["mlp_rms"], sc36["mid_rms"]), k=24)
    check(wr["source"] == "live", f"/neuron/wiring unbuilt: source {wr['source']}")
    for side in ("down", "up"):
        got = [(x["layer"], x["n"], x["sign"], x["gate"], x["w"]) for x in wr[side]]
        want = [(x["layer"], x["n"], x["sign"], x["gate"], x["w"]) for x in live[side]]
        check(got == want, f"/neuron/wiring {side}: the links of atlas_ops.neuron_wiring, weighed with the 36 texts' scales")
        key = "sign" if side == "down" else "gate"
        check(all(x["face"] == ("+" if x[key] > 0 else "-") and x["words"]["lens"] is None and x["name"] == "" for x in wr[side]),
              f"/neuron/wiring {side}: a reader on the face pushed, a writer on the face that opens the gate; no index words, no name")
    check(len(wr["down"]) == 24 and len(wr["up"]) == 24, "/neuron/wiring: 24 each way")
    r = c.get("/api/atlas/neuron/wiring", params={"layer": 0, "n": 5})
    check(r.status_code == 200 and r.json()["up"] == [], "/neuron/wiring layer 0: nothing writes into it")
    section("before any index: says so, and reads the wiring live", before)

    # ---------------------------------------------------------------------------------------- 4. live readings
    before = len(failures)
    nl = c.get("/api/atlas/neuron/lens", params={"layer": 2, "n": 5}).json()
    v = vault.inspect_bipolar_slot(layer_idx=2, neuron_idx=5, polarity=1.0, top_k_tokens=12)
    check(nl["faces"]["+"]["lens"] == [t["token"] for t in v["north_pole"]["j_vindex_tokens"]]
          and nl["faces"]["-"]["eye"] == [t["token"] for t in v["south_pole"]["naive_tokens"]], "/neuron/lens: the vault's words")
    check(nl["faces"]["+"]["check"] == v["north_pole"]["label_check"] and nl["faces"]["-"]["check"] == v["south_pole"]["label_check"],
          "/neuron/lens: each face's label check")
    check(nl["faces"]["+"]["check"]["held_back"] and "below_layer_40" in nl["faces"]["+"]["check"]["reasons"],
          "/neuron/lens: the stand-in's layers are all held back by the gate")
    check(nl["steer"] is None and nl["edited"] is False and (nl["act_range"] or {}).get("source") == "the study's 36 texts (provisional)",
          "/neuron/lens: no steer lens yet; the range over the 36 texts, read when the live wiring needed the norms' scales")
    check(list(nl)[-2:] == ["label_gate", "lens"], "/neuron/lens ends with the label gate and the lens")

    sl = c.post("/api/atlas/neuron/steer_lens", params={"layer": 2, "n": 5}).json()
    ctx = [rt.tokenize_prompt(rt.compose_prompt(t, chat=True)) for t in ao.STEER_LENS_CHATS] + [rt.tokenize_prompt(t) for t in ao.STEER_LENS_PLAIN]
    d = ao.steer_lens(lm, W_U, ctx, w, 2, 5, p50[2], k=12, bos_id=bos)
    check(sl["source"] == "live" and sl["up"] == [vocab[i] for i in d["up"]] and sl["down"] == [vocab[i] for i in d["down"]],
          "/neuron/steer_lens: atlas_ops.steer_lens's rises, both signs")
    check(close(sl["kl_up"], d["kl_up"]) and close(sl["kl_down"], d["kl_down"]) and sl["scale_source"]["kind"] == "study_texts",
          "/neuron/steer_lens: its KL, on the study texts' scale")
    nl = c.get("/api/atlas/neuron/lens", params={"layer": 2, "n": 5}).json()
    check(nl["steer"] == sl, "/neuron/lens: the steer lens read live is kept")
    check(nl["act_range"] == {"act_max": round(float(ex.pos[2, 5]), 4), "act_min": round(float(ex.neg[2, 5]), 4),
                              "over_tokens": ex.tokens, "source": "the study's 36 texts (provisional)",
                              "regimes": {k: round(float(getattr(ex, k)[2, 5]), 4) for k in ("pp", "nn", "pn", "np")}},
          f"/neuron/lens act_range {nl['act_range']}")

    # the push chat
    prompt = "Who are you?"
    r = c.post("/api/atlas/push", json={"layer": 2, "n": 5, "prompt": prompt, "max_new": 14})
    evs = events(r.text)
    got = voices(evs)
    want = write_direct([chat_ids([{"role": "user", "content": prompt}])] * 3, [ao.Push(2, 5, "add", 0.8)], 14)
    check(r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream"), "/push: an event stream")
    check(evs and evs[-1] == {"row": None, "done": True}, f"/push: ends done ({evs[-1:]})")
    check(got == want, f"/push: the three voices are atlas_ops.write_rows's ({got!r} vs {want!r})")
    check(got[0] != got[1] or got[2] != got[1], "/push: a push of 0.8 moves the answer")
    r = c.post("/api/atlas/steer", json={"layer": 2, "n": 5, "prompt": prompt, "max_new": 14}).json()
    check([r["up"], r["baseline"], r["down"]] == want and r["pushed"] == [{"layer": 2, "n": 5, "mode": "add", "dose": 0.8}],
          "/steer: the same three voices as whole texts")
    hist = [{"user": prompt, "replies": got}]
    r = c.post("/api/atlas/push", json={"layer": 2, "n": 5, "prompt": "And then?", "history": hist, "max_new": 10})
    rows = [chat_ids([{"role": "user", "content": prompt}, {"role": "assistant", "content": got[i]}, {"role": "user", "content": "And then?"}])
            for i in range(3)]
    check(voices(events(r.text)) == write_direct(rows, [ao.Push(2, 5, "add", 0.8)], 10), "/push: a conversation, each voice its own replies")
    multi = [{"layer": 1, "n": 3, "mode": "mul", "dose": 0.0}, {"layer": 3, "n": 7, "mode": "act", "dose": 2.0}]
    r = c.post("/api/atlas/push", json={"neurons": multi, "prompt": prompt, "max_new": 10, "system": "Be brief."})
    rows = [chat_ids([{"role": "system", "content": "Be brief."}, {"role": "user", "content": prompt}])] * 3
    check(voices(events(r.text)) == write_direct(rows, [ao.Push(1, 3, "mul", 0.0), ao.Push(3, 7, "act", 2.0)], 10),
          "/push: several neurons at once, with a system turn")
    for bad, says in [({"prompt": "x"}, "Name the neuron"), ({"layer": 6, "n": 0, "prompt": "x"}, "Layer 6 does not exist"),
                      ({"layer": 0, "n": 0, "prompt": "x", "mode": "zap"}, 'mode is "add", "mul" or "act"'),
                      ({"neurons": [{"layer": 0, "n": 128}], "prompt": "x"}, "Neuron 128 does not exist"),
                      ({"neurons": [{"layer": 0, "head": 9}], "prompt": "x"}, "has 4 attention heads")]:
        r = c.post("/api/atlas/push", json=bad)
        check(r.status_code == 400 and says in r.json()["detail"], f"/push refused before streaming: {bad} -> {r.status_code} {r.text[:120]}")

    # reads
    r = c.post("/api/atlas/reads", json={"layer": 3, "n": 7, "text": text}).json()
    base_acts = ao.read_neuron(lm, ids_text, 3, 7)
    flat = [x for u in r["units"] for x in u["pieces"]]
    check(len(flat) == len(ids_text) and all(close(a_, b_, 1e-3) for a_, b_ in zip(flat, base_acts)), "/reads: atlas_ops.read_neuron at every piece")
    check([u["text"] for u in r["units"]] == [t for t, _ in units_text] and r["pushed"] == [], "/reads: by display unit")
    check(r["act_range"]["act_max"] == round(float(ex.pos[3, 7]), 4), "/reads: the activation range")
    pushed = c.post("/api/atlas/reads", json={"layer": 3, "n": 7, "text": text, "neurons": [{"layer": 1, "n": 3, "mode": "mul", "dose": 0.0}]}).json()
    iv = ao.resolve(w, [ao.Push(1, 3, "mul", 0.0)], [], (1,), p50=lambda l: p50[l], act_scale=act_scale)
    want_acts = ao.read_neuron(lm, ids_text, 3, 7, iv, bos_id=bos)
    check(all(close(a_, b_) for a_, b_ in zip([x for u in pushed["units"] for x in u["pieces"]], want_acts)), "/reads: with an earlier neuron silenced")
    check(pushed["pushed"] == [{"layer": 1, "n": 3, "mode": "mul", "dose": 0.0}], "/reads: the pushes echoed")
    for later in ({"layer": 4, "n": 9, "mode": "add", "dose": 3.0}, {"layer": 3, "n": 8, "mode": "act", "dose": 5.0}):
        same = c.post("/api/atlas/reads", json={"layer": 3, "n": 7, "text": text, "neurons": [later]}).json()
        check([x for u in same["units"] for x in u["pieces"]] == [x for u in r["units"] for x in u["pieces"]],
              f"/reads: a push at or after the neuron's own block leaves it as it was ({later})")
    hd = c.post("/api/atlas/reads", json={"layer": 3, "n": 7, "text": text, "heads": [{"layer": 3, "head": 0, "dose": 0.0}]}).json()
    iv = ao.resolve(w, [], [ao.HeadScale(3, 0, 0.0)], (1,), p50=lambda l: 1.0, act_scale=lambda l, n: 1.0)
    want_acts = ao.read_neuron(lm, ids_text, 3, 7, iv, bos_id=bos)
    got_acts = [x for u in hd["units"] for x in u["pieces"]]
    check(all(close(a_, b_) for a_, b_ in zip(got_acts, want_acts)) and got_acts != flat, "/reads: its own layer's head silenced moves it")
    r = c.post("/api/atlas/reads", json={"layer": 3, "n": 7, "text": "word " * 3000})
    check(r.status_code == 400 and "Shorten it" in r.json()["detail"], f"/reads: a text too long is refused ({r.text[:100]})")

    # the next piece's odds
    lp = c.post("/api/atlas/logprobs", json={"text": prompt, "k": 5, "targets": ["I", "Hello there"]}).json()
    ids_q = rt.tokenize_prompt(rt.compose_prompt(prompt, chat=True))
    base, _ = ao.next_logprobs(lm, W_U, ids_q, None, bos_id=bos)
    top = base.topk(5)
    check(lp["read_at"] == len(ids_q) - 1 and [t for t, _ in lp["top"]] == [vocab[i] for i in top.indices.tolist()],
          "/logprobs: the five likeliest next pieces")
    check(all(close(x, float(y)) for (_, x), y in zip(lp["top"], top.values)), "/logprobs: their log-probabilities")
    tid_i = tok.encode("I", add_special_tokens=False)[0]
    check(close(lp["targets"][0]["logprob"], float(base[tid_i])) and lp["targets"][1].get("first_piece_only") is True,
          "/logprobs: a target's log-probability, a multi-piece target by its first piece")
    lp = c.post("/api/atlas/logprobs", json={"text": prompt, "neurons": [{"layer": 4, "n": 9, "dose": 3.0}]}).json()
    iv = ao.resolve(w, [ao.Push(4, 9, "add", 3.0)], [], (0, 1), p50=lambda l: p50[l], act_scale=act_scale)
    b2, alt = ao.next_logprobs(lm, W_U, ids_q, iv, bos_id=bos)
    check(close(lp["kl"], float((b2.exp() * (b2 - alt)).sum()), 1e-3) and lp["kl"] > 0, f"/logprobs: KL from base to pushed ({lp['kl']})")
    check(len(lp["rose"]) == 12 and len(lp["fell"]) == 12 and lp["rose"][0][2] > lp["rose"][0][1] and lp["fell"][0][2] < lp["fell"][0][1],
          "/logprobs: what rose and what fell")
    check(lp["baseline"]["top"] == [[vocab[i], round(float(x), 3)] for x, i in zip(*b2.topk(15))], "/logprobs: the base row unchanged")

    # compare
    cmp_ = c.post("/api/atlas/compare", json={"neurons": [{"layer": 2, "n": 5}, {"layer": 4, "n": 9}],
                                              "texts": [text, {"label": "code", "text": CORPUS[2]}]}).json()
    check([t["label"] for t in cmp_["texts"]] == ["t1", "code"], "/compare: the texts' labels")
    for row in cmp_["neurons"]:
        for label, tx in (("t1", text), ("code", CORPUS[2])):
            ids_c = rt.tokenize_prompt(tx)
            a_ = np.array(ao.read_neuron(lm, ids_c, row["layer"], row["n"]))
            got_ = row["by_text"][label]
            check(close(got_["max"][0], a_[1:].max()) and close(got_["min"][0], a_[1:].min()), f"/compare {row['layer']}/{row['n']} {label}: max and min past the start mark")
            s_ = scale_scores(a_[None, :, None], ex.pos.numpy()[row["layer"]: row["layer"] + 1, row["n"]: row["n"] + 1],
                              ex.neg.numpy()[row["layer"]: row["layer"] + 1, row["n"]: row["n"] + 1])[0, :, 0]
            check(close(abs(got_["peak_score"]), np.abs(s_).max(), 1e-2), f"/compare {row['layer']}/{row['n']} {label}: the peak score")

    # fires
    fr = c.post("/api/atlas/fires", json={"text": text, "limit": 12}).json()
    A = ao.all_activations(lm, ids_text, dtype=torch.float32).numpy()  # [L, T, N]
    P = np.maximum(ex.pos.numpy(), np.maximum(0.1 * -ex.neg.numpy(), 1e-3))
    Q = np.minimum(ex.neg.numpy(), np.minimum(-0.1 * ex.pos.numpy(), -1e-3))
    sp = A[:, 1:].max(1) / P
    sn = A[:, 1:].min(1) / Q
    every = sorted([(float(sp[l, n]), l, n, "+") for l in range(L) for n in range(N) if sp[l, n] > 0]
                   + [(float(sn[l, n]), l, n, "-") for l in range(L) for n in range(N) if sn[l, n] > 0], reverse=True)[:12]
    check([(x["layer"], x["n"], x["face"]) for x in fr["results"]] == [(l, n, s) for _, l, n, s in every], "/fires: the twelve hardest, both sides")
    check(all(close(x["score"], s_, 1e-2) for x, (s_, *_r) in zip(fr["results"], every)), "/fires: their scores")
    check(all(x["pos"] >= 1 and x["token"] for x in fr["results"]) and fr["at"] is None, "/fires: each where it fires, past the start mark")
    fr = c.post("/api/atlas/fires", json={"text": text, "token": "Bob", "limit": 5}).json()
    k_bob = next(i for i, (u, _) in enumerate(units_text) if "Bob" in u)
    pos_bob = units_text[k_bob][1][-1]
    S_ = np.maximum(A[:, pos_bob] / P, A[:, pos_bob] / Q)
    want_top = sorted(((float(S_[l, n]), l, n) for l in range(L) for n in range(N)), reverse=True)[:5]
    check(fr["at"] == k_bob and [(x["layer"], x["n"]) for x in fr["results"]] == [(l, n) for _, l, n in want_top]
          and all(x["pos"] == pos_bob for x in fr["results"]), "/fires at one unit: the five hardest there")
    r = c.post("/api/atlas/fires", json={"text": text, "token": "zebra"})
    check(r.status_code == 400 and "No unit of the text holds 'zebra'" in r.json()["detail"], "/fires: a token not in the text is refused")
    section("live readings: lens, steer lens, the three voices, reads, odds, compare, fires", before)

    # --------------------------------------------------------------------------------------------- 5. run view, heads
    before = len(failures)
    rr = c.post("/api/atlas/run", json={"text": text}).json()
    check(rr["ready"] and rr["tokens"] == [t for t, _ in units_text] and rr["n_pieces"] == len(ids_text), "/run: its display units")
    A16 = ao.all_activations(lm, ids_text).numpy()
    S3 = scale_scores(A16[:, 3:4], ex.pos.numpy(), ex.neg.numpy())[:, 0]
    rv = c.get("/api/atlas/run/view", params={"pos": 3, "min": 0.5, "limit": 2000}).json()
    check(rv["total"] == int((np.abs(S3) >= 0.5).sum()), f"/run/view at piece 3: {rv['total']} pass |score| >= 0.5")
    check(all(close(x["score"], abs(S3[x["layer"], x["n"]]), 1e-2) and x["pos"] == 3 for x in rv["results"]), "/run/view: each score at piece 3")
    mp = np.frombuffer(base64.b64decode(rv["map"]), "<f2").reshape(rv["shape"])
    check(rv["shape"] == [L, N] and np.allclose(mp, np.where(np.abs(S3) >= 0.5, S3, 0), atol=1e-2), "/run/view: the map")
    rv = c.get("/api/atlas/run/view", params={"pos": -1, "sign": "+", "min": 1.0, "lo": 2, "hi": 4}).json()
    Sall = np.stack([scale_scores(A16[l: l + 1], ex.pos.numpy()[l: l + 1], ex.neg.numpy()[l: l + 1])[0] for l in range(L)])  # [L, T, N]
    best = Sall.max(1)
    want_n = int(((best >= 1.0) & (np.arange(L)[:, None] >= 2) & (np.arange(L)[:, None] <= 4)).sum())
    check(rv["total"] == want_n and all(2 <= x["layer"] <= 4 and x["face"] == "+" for x in rv["results"]), "/run/view: each neuron at its strongest, + only, layers 2-4")
    check(all(close(x["score"], best[x["layer"], x["n"]], 1e-2) and Sall[x["layer"], x["pos"], x["n"]] == best[x["layer"], x["n"]]
              for x in rv["results"]), "/run/view: the piece each is strongest at")
    check(c.get("/api/atlas/run/view", params={"sign": "x"}).status_code == 400, "/run/view: a bad sign refused")
    rh = c.get("/api/atlas/run/heads", params={"pos": 4}).json()
    capt = ao.capture_heads(lm, ids_text, w)
    check(len(rh["layers"]) == L and all(len(x) == H and all(0 <= v_ <= 1.0 + 1e-6 for v_ in x) for x in rh["layers"]),
          "/run/heads: every head's write, a share of its layer's largest")
    check(all(close(rh["layers"][l][hh], float(capt.writes[l][4, hh]) / float(capt.writes[l][1:].max()), 1e-2) for l in range(L) for hh in range(H)),
          "/run/heads: the capture's writes")
    r = c.get("/api/atlas/run/paths", params={"pos": 4})
    check(r.content == b"" and r.headers["X-Web-Total"] == "0", "/run/paths: nothing without the wiring index")
    row = c.get("/api/atlas/head/row", params={"layer": 5, "head": 1, "pos": 6}).json()["units"]
    Ah = ao.head_pattern(capt, shapes[5], 1)[6]
    check(close(sum(x for x in row if x is not None), 1.0) and all((x is None) == (idx[0] > 6) for x, (_, idx) in zip(row, units_text)),
          "/head/row: the head's weights at piece 6, by unit, none after it")
    check(all(x is None or close(x, float(Ah[idx].sum()), 1e-3) for x, (_, idx) in zip(row, units_text)), "/head/row: the pattern's weights")
    pat = c.post("/api/atlas/head/pattern", json={"layer": 0, "head": 2, "text": text}).json()
    check(len(pat["rows"]) == len(units_text) and all(len(x) == i + 1 and close(sum(x), 1.0) for i, x in enumerate(pat["rows"])),
          "/head/pattern: each unit's row adds up to one")
    check(pat["kind"] == "sliding" and pat["window"] == shapes[0].window, "/head/pattern: a sliding layer and its window")
    hw = c.get("/api/atlas/head", params={"layer": 3, "head": 1}).json()
    want_hw = ao.head_wiring(ao.HeadSpaces(w, shapes), 3, 1, k=40)
    for role in ("q", "k", "v", "o"):
        check([(x["layer"], x["n"]) for x in hw[role]] == [(x["layer"], x["n"]) for x in want_hw[role]], f"/head {role}: atlas_ops.head_wiring")
        check(hw[f"{role}_pooled"] == {"src": None, "tokens": [], "gated": 0}, f"/head {role}: nothing to pool without an index")
    check(hw["kv_head"] == shapes[3].kv_of(1), "/head: its key-value head")
    pr = c.get("/api/atlas/head/profile", params={"layer": 2, "head": 0}).json()
    check(len(pr["dist"]) == 5 and sum(pr["dist"]) <= 1.0 + 1e-3 and pr["dist_names"][0] == "self", "/head/profile: where it looks")
    check(all(p_["k"] < p_["q"] - 1 and p_["qt"] and p_["text"] in ao.HEAD_TEXTS for p_ in pr["pairs"]), "/head/profile: its pairs read more than one back")
    at = c.post("/api/atlas/attention", json={"layer": 2, "head": 1, "text": text, "token": "Bob"}).json()
    check("Bob" in at["query"]["token"] and close(sum(x for _, x in at["row"]), 1.0, 1e-2), "/attention: the row at Bob adds up to one")
    U_at = len(at["tokens"])
    check(len(at["strongest_reads"]) == 12 and all(1 <= x["back"] < U_at - 1 for x in at["strongest_reads"])
          and len(at["top_sources"]) == min(8, at["query"]["index"] + 1),
          f"/attention: the strongest reads, never of itself or of the first unit ({len(at['strongest_reads'])}, {len(at['top_sources'])})")
    ws_ = [abs(x["weight"]) for x in at["strongest_reads"]]
    check(ws_ == sorted(ws_, reverse=True), "/attention: strongest first")
    dr = c.post("/api/atlas/attn_drive", json={"layer": 4, "n": 9, "text": text, "token": "Bob"}).json()
    want_d = ao.attention_drive(ao.capture_heads(lm, ids_text, w, streams=True), w, shapes, 4, 9, pos_bob, top=8)
    check(dr["read_pos"] == pos_bob and close(dr["act"], want_d["act"]) and close(dr["linearised_total"], want_d["total"]), "/attn_drive: the activation and its total")
    parts = dr["attention_sum"] + dr["mlp_sum"] + dr["embedding"]
    check(abs(parts - dr["linearised_total"]) <= 1e-3 * max(1.0, abs(dr["linearised_total"])), f"/attn_drive: the parts add up ({parts} vs {dr['linearised_total']})")
    check(close(dr["act"], ao.read_neuron(lm, ids_text, 4, 9)[pos_bob], 1e-3), "/attn_drive: the activation, as a read gives it")
    check(all("from" in x for x in dr["heads"][:8]), "/attn_drive: where the top heads read it from")
    dr2 = c.post("/api/atlas/attn_drive", json={"layer": 4, "n": 9, "text": text}).json()
    a49 = np.abs(np.array(ao.read_neuron(lm, ids_text, 4, 9))[1:])
    check(dr2["read_pos"] in units_text[next(i for i, (_, idx) in enumerate(units_text) if int(a49.argmax()) + 1 in idx)][1],
          "/attn_drive: without a token, the unit where it fires hardest")
    roles = c.get("/api/atlas/neuron/heads", params={"layer": 2, "n": 5}).json()
    want_r = ao.neuron_head_roles(ao.HeadSpaces(w, shapes), 2, 5, k=8)
    check(all(roles["roles"][k_] == want_r[k_] for k_ in ("q", "k", "v", "o")) and roles["z_by_layer"] == want_r["z_by_layer"],
          "/neuron/heads: atlas_ops.neuron_head_roles")
    nv = c.get("/api/atlas/neuron", params={"layer": 2, "n": 5}).json()
    check(list(nv)[-2:] == ["label_gate", "lens"] and nv["faces"] == nl["faces"] and nv["wiring_source"] == "live", "/neuron: the whole neuron")
    check([(x["layer"], x["n"]) for x in nv["writes_into"]] == [(x["layer"], x["n"]) for x in live["down"]] and nv["census"] is None,
          "/neuron: its wiring, no census")
    for path, params, says in [("/api/atlas/neuron/lens", {"layer": 6, "n": 0}, "page layers 1 to 6"),
                               ("/api/atlas/neuron/lens", {"layer": 0, "n": 128}, "each layer has 128"),
                               ("/api/atlas/head", {"layer": 0, "head": 4}, "has 4 attention heads"),
                               ("/api/atlas/hover", {"layer": -1, "n": 0}, "does not exist")]:
        r = c.get(path, params=params)
        check(r.status_code == 400 and says in r.json()["detail"], f"{path} {params}: refused ({r.text[:120]})")
    section("the run view, heads and the drive", before)

    # ------------------------------------------------------------------------------------------- 6. every index built
    before = len(failures)
    out = index_dir()
    check(out == TMP / "atlas" / "index", f"the index folder is the atlas folder's: {out}")
    out.mkdir(parents=True, exist_ok=True)
    ab.build_lens(rt, out, chunk=48, log=quiet)
    wins = ab.windows(rt, CORPUS, 24, 400)
    ab.build_stats(rt, out, wins, "check", log=quiet)
    idx = AtlasIndex(out)
    stats = idx.get("act_scale")
    f_idx = ao.link_scale(w.scalars, [float(x) for x in stats["mlp_rms_q"][:, 1]], [float(x) for x in stats["mid_rms_q"][:, 1]])
    ab.build_wiring(rt, out, f_idx, log=quiet)
    live = ao.neuron_wiring(w, 2, 5, f_idx, k=24)
    ab.build_steer_lens(rt, out, [float(x) for x in stats["resid_q"][:, 1]], batch=16, log=quiet)
    ab.build_layout(rt, out, method="pca", log=quiet)
    ab.build_passages(rt, out, wins, "check", k=6, sample=3, extremes=(stats["pos"], stats["neg"]), extremes_source="the stats index (check)",
                      chunk=4, log=quiet)
    arch = c.get("/api/atlas/arch").json()
    check(all(v_["built"] for k_, v_ in arch["indexes"].items() if k_ != "folder"), "/arch: every index built")
    check(arch["push"]["scale_source"]["kind"] == "index" and arch["push"]["scale_source"]["corpus"] == "check", "/arch: the scales from the stats index")
    check(all(c.get("/api/health").json()["atlas"]["indexes"].values()), "/api/health: every index on disk")
    cl = c.get("/api/atlas/neuron/lens", params={"layer": 2, "n": 5}).json()["ceiling"]
    want_c = float(w.write_ceiling(2, [5])[0])
    check(abs(cl["write"] - want_c) < 1e-4 and abs(cl["p50"] - float(stats["resid_q"][2, 1])) < 1e-4 and abs(cl["rel"] - want_c / float(stats["resid_q"][2, 1])) < 1e-4
          and cl["source"] == "check", f"/neuron/lens: the most the neuron can write, as a share of the stream ({cl})")
    hv = c.get("/api/atlas/hover", params={"layer": 2, "n": 5}).json()
    check(hv["ready"] and hv["faces"]["+"]["lens"] == [vocab[i] for i in idx.face_ids(2, 5, "+", "lens", 6)]
          and hv["faces"]["-"]["eye"] == [vocab[i] for i in idx.face_ids(2, 5, "-", "eye", 6)], "/hover: the index's words")
    check(hv["faces"]["+"]["check"] == idx.face_check(2, 5, "+") and hv["faces"]["+"]["steer"] == [vocab[i] for i in idx.get("steer_lens")["up"][2, 5, :6]],
          "/hover: each face's check and steer lens")
    check(hv["links"] == idx.links(2, 5, 16) and "stale" not in hv["faces"]["+"], "/hover: its sixteen links each way")
    wr = c.get("/api/atlas/neuron/wiring", params={"layer": 2, "n": 5}).json()
    check(wr["source"] == "index" and [(x["layer"], x["n"], x["sign"]) for x in wr["down"]] == [(x["layer"], x["n"], x["sign"]) for x in live["down"]],
          "/neuron/wiring: from the index, the live links")
    check(all(x["words"]["check"] is not None and x["words"]["lens"] for x in wr["down"]), "/neuron/wiring: each link's words with their check")
    lens = idx.get("lens")
    probe = next((l, n) for l in range(3, L) for n in range(N) if len(vocab[int(lens["lens_pos"][l, n, 0])].strip()) >= 3)
    word = vocab[int(lens["lens_pos"][probe[0], probe[1], 0])].strip()
    s = c.get("/api/atlas/search", params={"q": word, "limit": 2000}).json()
    hit = [x for x in s["held"] if (x["layer"], x["n"], x["sign"]) == (*probe, "+")]
    check(s["ready"] and s["tokens"] > 0 and hit and hit[0]["rank"] == 0, f"/search {word!r}: finds {probe} + at rank 0")
    check(all(x["words"]["check"] is not None and "census" in x and "name" in x for x in s["results"] + s["held"]), "/search: each face's words and check")
    # every lens label on the stand-in's six layers is held back: matched through the lens or the naked eye, a face goes
    # to the apparatus, never the main reading or the map; steer-lens matches stay
    check(s["held_back"] == len(s["held"]) > 0 and all(x["src"] in ("lens", "eye") and x["words"]["check"]["held_back"] for x in s["held"])
          and all(x["src"] == "steer" for x in s["results"]), "/search: faces the gate holds back kept apart, steer-lens matches in the main reading")
    raw = c.get("/api/atlas/web", params={"layer": 2, "n": 5})
    arr = np.frombuffer(raw.content, np.int16).reshape(-1, 6)
    want_web = idx.cone(2, 5, 2, 15.0, 20000, 5, None)
    check(int(raw.headers["X-Web-Total"]) == len(want_web) == len(arr) and len(arr) > 0, f"/web: {len(arr)} links")
    check(all(tuple(int(x) for x in a_[:4]) == r_[:4] and int(a_[4]) == int(round(r_[4] * 10)) and int(a_[5]) == r_[5] for a_, r_ in zip(arr, want_web)),
          "/web: the index's cone, z x 10 signed")
    ws = c.get("/api/atlas/web/summary", params={"layer": 2, "n": 5}).json()
    check(ws["ready"] and ws["src"] == "steer" and set(ws["down"]) == {"+", "-"} and set(ws["up"]) == {"+", "-"}, "/web/summary: pooled from the steer lens, by face")
    ws = c.get("/api/atlas/web/summary", params={"layer": 2, "n": 5, "src": "lens"}).json()
    check(ws["src"] == "lens" and ws["down"]["+"]["words"] == [] and ws["gated"]["down"] == ws["down"]["+"]["neurons"] + ws["down"]["-"]["neurons"],
          "/web/summary from the lens: every face held back on the stand-in, none pooled")
    check(c.get("/api/atlas/web/summary", params={"layer": 2, "n": 5, "src": "x"}).status_code == 400, "/web/summary: a bad src refused")
    lay = c.get("/api/atlas/layout").json()
    cell = np.frombuffer(base64.b64decode(lay["cell"]), "<i2").reshape(lay["shape"])
    check(lay["available"] and lay["method"] == "pca" and np.array_equal(cell, idx.get("layout")["cell"].astype(np.int16)), "/layout: the index's cells")
    cn = c.get("/api/atlas/cone", params={"layer": 2, "n": 5}).json()
    links24 = idx.links(2, 5, 24)
    by = {(s_, x["layer"], x["n"]): x for s_ in ("down", "up") for x in links24[s_]}
    h1 = [x for x in cn["nodes"] if abs(x["hop"]) == 1]
    check(cn["ready"] and h1 and all(
        (x["face"] == ("+" if by[("down", x["layer"], x["n"])]["sign"] > 0 else "-") and x["when"] == ("+" if by[("down", x["layer"], x["n"])]["gate"] > 0 else "-"))
        if x["hop"] == 1 else
        (x["face"] == ("+" if by[("up", x["layer"], x["n"])]["gate"] > 0 else "-") and x["drives"] == ("+" if by[("up", x["layer"], x["n"])]["sign"] > 0 else "-"))
        for x in h1), "/cone: a reader on the face its link pushes, when the neuron fires on the gate's face; a writer on its gate's face")
    check(sum(cn["per_hop"].values()) == cn["neurons"] and all(x["hop"] != 0 for x in cn["nodes"]), "/cone: every neuron reached once a way")
    c.post("/api/atlas/run", json={"text": text})
    r = c.get("/api/atlas/run/paths", params={"pos": 4, "top": 60, "min": 0})
    pa = np.frombuffer(r.content, np.int16).reshape(-1, 6)
    S4 = scale_scores(A16[:, 4:5], stats["pos"], stats["neg"])[:, 0]
    act_set = {(int(i) // N, int(i) % N) for i in np.argsort(-np.abs(S4).ravel())[:60]}
    check(r.headers["X-Active"] == "60" and all((int(a_[0]), int(a_[1])) in act_set and (int(a_[2]), int(a_[3])) in act_set for a_ in pa),
          f"/run/paths: {len(pa)} links, both ends among the 60 active")
    nl = c.get("/api/atlas/neuron/lens", params={"layer": 2, "n": 5}).json()
    st = idx.get("steer_lens")
    check(nl["steer"]["source"] == "index" and nl["steer"]["up"] == [vocab[i] for i in st["up"][2, 5]], "/neuron/lens: the steer lens from its index")
    check(nl["act_range"]["source"] == "check" and close(nl["act_range"]["act_max"], float(stats["pos"][2, 5])), "/neuron/lens: the range from the stats index")
    hw = c.get("/api/atlas/head", params={"layer": 3, "head": 1}).json()
    check(hw["o_pooled"]["src"] == "steer" and all(x["words"]["steer"] for x in hw["o"]), "/head: its neurons' words, pooled from the steer lens")
    r = c.post("/api/atlas/push", json={"layer": 2, "n": 5, "prompt": prompt, "max_new": 8})
    sp50 = [float(x) for x in stats["resid_q"][:, 1]]
    sscale = lambda l, n: max(abs(float(stats["pos"][l, n])), abs(float(stats["neg"][l, n])), 1e-6)  # noqa: E731
    check(voices(events(r.text)) == write_direct([chat_ids([{"role": "user", "content": prompt}])] * 3, [ao.Push(2, 5, "add", 0.8)], 8, (sp50, sscale)),
          "/push: the push's unit from the stats index")
    section("with every index built: the indexes answer", before)

    before = len(failures)
    P = idx.get("passages")
    marks = atlas_routes.passage_marks(tok)
    eos = tok.eos_token_id
    # cut_passage on hand-made ids: never across a mark, the mark itself left out; a peak on a mark that ends something
    # shows nothing after it, on one that begins something nothing before it
    t_open, t_close = tok.convert_tokens_to_ids("<|turn>"), tok.convert_tokens_to_ids("<turn|>")
    ids_h = [bos, 40, 41, t_open, 42, 43, 44, 45, t_close, 46]
    check(atlas_routes.cut_passage(ids_h, 6, 24, 6, marks) == (4, 8, "turn", "turn"), "cut_passage: a peak inside a turn shows that turn only")
    check(atlas_routes.cut_passage(ids_h, 6, 1, 0, marks) == (5, 7, None, None), "cut_passage: the counts run out first")
    check(atlas_routes.cut_passage([bos, 40, eos, 41, 42], 2, 24, 6, marks) == (1, 3, "start", "document"), "cut_passage: a peak on an end mark, nothing after it")
    check(atlas_routes.cut_passage([bos, 40, t_open, 41], 2, 24, 6, marks) == (2, 4, "turn", "end"), "cut_passage: a peak on a turn's opening, nothing before it")
    # a start mark inside a window (a chat composed with its own, packed after another document) is a boundary between
    # documents, not the window's start: the model read what lies before it
    check(atlas_routes.cut_passage([bos, 40, eos, bos, 41, 42], 4, 24, 6, marks) == (4, 6, "document", "end")
          and atlas_routes.cut_passage([bos, 40, eos, bos, 41], 3, 24, 6, marks) == (3, 5, "document", "end")
          and atlas_routes.cut_passage([bos, 40, 41, bos, 42], 1, 24, 6, marks) == (1, 3, "start", "document"),
          "cut_passage: a start mark inside the window cuts as a document's boundary, on either side or under the peak")
    pr = c.get("/api/atlas/neuron/passages", params={"layer": 2, "n": 5, "face": "+"}).json()
    want = idx.passages(2, 5, "+")
    tops = [x for x in pr["passages"] if x["kind"] == "top"]
    smps = [x for x in pr["passages"] if x["kind"] == "sample"]
    check(pr["ready"] and pr["source"] == "index" and pr["k"] == len(tops) == 6 and pr["K"] == 6 and pr["per_window"] == 1
          and pr["tokens"] == int(stats["tokens"]) and pr["corpus"] == "check" and pr["sampled"] == 3 and len(smps) == len(want["sample"]),
          f"/neuron/passages: six top passages and the samples, with their provenance ({ {k_: pr.get(k_) for k_ in ('k', 'K', 'tokens', 'corpus', 'sampled')} })")
    check([(x["window"], x["at"]) for x in tops] == [(x["window"], x["at"]) for x in want["top"]] and len({x["window"] for x in tops}) == 6,
          "/neuron/passages: the index's events, strongest first, one a window")
    span_bad, cuts = [], set()
    for face in ("+", "-"):
        for (l, n) in ((2, 5), (0, 9), (4, 77), (5, 120)):
            r_ = c.get("/api/atlas/neuron/passages", params={"layer": l, "n": n, "face": face, "before": 24, "after": 6}).json()
            ext_ = float(stats["pos" if face == "+" else "neg"][l, n])
            for x in r_["passages"]:
                ids_w, at = idx.window_ids(x["window"]), x["at"]
                s_, e_ = x["span"]
                why_b, why_a = x["cut"]
                cuts.add(why_b)
                cuts.add(why_a)
                shown = ids_w[s_:e_]
                # the units are the span's own, the peak marked in the one holding it; the span stops where its reason
                # says: right after a mark (the start mark, an end mark), at the window's end, or once the count is
                # reached (a few pieces more to keep a character whole)
                ok = (x["ids"] == ids_w[:e_] and s_ <= at < e_
                      and x["units"] == [[t, (at - s_) in i_] for t, i_ in ao.token_units(tok, shown)]
                      and not any(t in marks for i_, t in enumerate(shown) if s_ + i_ != at)
                      and (why_b != "start" or s_ == 1)
                      and (why_b != "document" or ids_w[s_ - 1] == eos or (ids_w[s_ - 1] == bos and s_ > 1) or (s_ == at > 0 and ids_w[at] == bos))
                      and (why_b is not None or 24 <= at - s_ <= 27)
                      and (why_a != "end" or e_ == len(ids_w)) and (why_a != "document" or ids_w[e_] in (eos, bos))
                      and (why_a is not None or 6 <= e_ - at - 1 <= 9)
                      and x["regime"] == ("p" if x["g"] > 0 else "n") + ("p" if x["u"] > 0 else "n")
                      and (x["rel"] is None or abs(x["rel"] - x["act"] / ext_) < 2e-3)
                      and x["next"] == (vocab[ids_w[at + 1]] if at + 1 < len(ids_w) else None))
                if not ok:
                    span_bad.append((l, n, face, x["window"], at, x["cut"]))
    check(not span_bad, f"/neuron/passages: each passage's span, peak unit, cut, regime, rel and next ({span_bad[:3]})")
    check({"start", "document", "end", None} <= cuts, f"/neuron/passages: passages cut at the window's start, at a document's end mark, at the window's end and by the count ({cuts})")
    from collections import Counter

    rc = Counter(idx.window_ids(x["window"])[x["at"]] for x in tops)
    fc = Counter(idx.window_ids(x["window"])[x["at"] + 1] for x in tops if x["at"] + 1 < len(idx.window_ids(x["window"])))
    uni, toks_ = P["unigram"], int(stats["tokens"])
    for key, cnt in (("reads", rc), ("follows", fc)):
        check(sorted((r_["id"], r_["n"]) for r_ in pr[key]) == sorted(cnt.items()) and all(
            abs(r_["lift"] - (r_["n"] / sum(cnt.values())) / (int(uni[r_["id"]]) / toks_)) < 0.06 for r_ in pr[key]),
              f"/neuron/passages: {key} counts the word-pieces {'at the peaks' if key == 'reads' else 'just after them'}, each with its lift over the unigram base rate")
    check(pr["words_note"] == f"from 6 passages of a {toks_:,}-word-piece corpus; not validated as a label", f"/neuron/passages: the words' note ({pr['words_note']})")
    check(c.get("/api/atlas/neuron/passages?layer=2&n=5&face=+").json()["face"] == "+" and c.get("/api/atlas/neuron/passages",
          params={"layer": 2, "n": 5, "face": "-", "k": 2}).json()["k"] == 2, "/neuron/passages: a bare + in a URL is the + face; k shows fewer")
    for params, says in [({"layer": 2, "n": 5, "face": "x"}, 'A face is "+" or "-"'), ({"layer": 6, "n": 5}, "Layer 6 does not exist"),
                         ({"layer": 2, "n": 5, "k": 0}, "k counts passages"), ({"layer": 2, "n": 5, "before": -1}, "0 or more")]:
        r = c.get("/api/atlas/neuron/passages", params=params)
        check(r.status_code == 400 and says in r.json()["detail"], f"/neuron/passages {params}: refused in plain words ({r.text[:120]})")
    # /reads given a stored passage's ids reads its window again: the activation at the peak is the index's
    x0 = tops[0]
    rd = c.post("/api/atlas/reads", json={"layer": 2, "n": 5, "ids": x0["ids"]}).json()
    live = ao.read_neuron(lm, x0["ids"], 2, 5, None, bos_id=bos, spare_bos=True)
    check([u["act"] for u in rd["units"]] == [max((live[i] for i in idx_), key=abs) for _, idx_ in ao.token_units(tok, x0["ids"])]
          and sum(len(u["pieces"]) for u in rd["units"]) == len(x0["ids"]), "/reads with ids: read exactly as given, as read_neuron reads those ids")
    check(np.float16(live[x0["at"]]) == P["pos_act"][2, 5, 0], f"/reads with ids: the stored passage read again gives the index's activation ({live[x0['at']]})")
    for body, says in [({"layer": 2, "n": 5}, "one of the two"), ({"layer": 2, "n": 5, "text": "hi", "ids": [bos, 40]}, "one of the two"),
                       ({"layer": 2, "n": 5, "ids": [bos] * 2049}, "at most 2,048"), ({"layer": 2, "n": 5, "ids": [bos, len(vocab)]}, f"Word-piece {len(vocab)} does not exist"),
                       ({"layer": 2, "n": 5, "ids": []}, "Give the word-piece ids")]:
        r = c.post("/api/atlas/reads", json=body)
        check(r.status_code == 400 and says in r.json()["detail"], f"/reads {str(body)[:60]}: refused in plain words ({r.text[:120]})")
    # what `rel` is a share of: the stats index's extreme, named with its corpus; said to be another corpus's when the
    # stats index was read over another; the strongest passage where the stats index never saw the face fire, or is not built
    check(pr["rel_to"] == "its extreme in the stats index (check)", f"/neuron/passages: rel over the stats index's extreme, its corpus named ({pr['rel_to']})")
    sp = out / "act_scale.npz"
    sp.rename(out / "act_scale.keep")
    with np.load(out / "act_scale.keep") as z:
        other = {k_: z[k_] for k_ in z.files}
    other["meta"] = np.array(json.dumps({**json.loads(str(other["meta"])), "corpus": "another"}))
    other["pos"] = other["pos"].copy()
    other["pos"][2, 5] = 0.0
    np.savez(sp, **other)
    r_other = c.get("/api/atlas/neuron/passages", params={"layer": 0, "n": 9}).json()
    r_never = c.get("/api/atlas/neuron/passages", params={"layer": 2, "n": 5}).json()
    sp.unlink()
    r_none = c.get("/api/atlas/neuron/passages", params={"layer": 2, "n": 5}).json()
    (out / "act_scale.keep").rename(sp)
    check(r_other["rel_to"] == f"its extreme in the stats index, read over another corpus (another, {int(stats['tokens']):,} word-pieces)"
          and "the stats index never saw this face fire" in r_never["rel_to"] and r_never["passages"][0]["rel"] == 1.0
          and "(no stats index)" in r_none["rel_to"] and r_none["passages"][0]["rel"] == 1.0
          and c.get("/api/atlas/neuron/passages", params={"layer": 2, "n": 5}).json()["rel_to"] == pr["rel_to"],
          f"/neuron/passages: rel_to says when the stats index is another corpus's, never saw the face fire, or is not built ({r_other['rel_to']}; {r_never['rel_to']}; {r_none['rel_to']})")
    section("the passages: the index's events, cut at marks, reads and follows, refusals, and /reads with their ids", before)

    # ------------------------------------------------------------------------------------------- 7. a vault edit
    before = len(failures)
    roles_before = c.get("/api/atlas/neuron/heads", params={"layer": 2, "n": 5}).json()
    down = lm.layers[2].mlp.down_proj
    down._memit_col_overlays = {5: torch.zeros(D)}
    vault.active_patches.append(PatchOperationRecord(patch_id=1, op_type="DELETE", layer_idx=2, neuron_idx=5, polarity=1.0,
                                                     target_concept="", strength=0.0, statement="DELETE LAYER 2 SLOT 5",
                                                     timestamp=time.time(), orig_norm=1.0, new_norm=0.0))
    wr = c.get("/api/atlas/neuron/wiring", params={"layer": 2, "n": 5}).json()
    check(wr["source"] == "live" and wr["down"] == [] and len(wr["up"]) == 24, "/neuron/wiring: an edited neuron read live: it writes nothing, so no readers")
    hv = c.get("/api/atlas/hover", params={"layer": 2, "n": 5}).json()
    check("stale" in hv["faces"]["+"] and hv["faces"]["+"]["check"]["edited"] and not hv["faces"]["+"]["check"]["lens_validated"],
          "/hover: an edited neuron's index words are marked stale")
    roles_after = c.get("/api/atlas/neuron/heads", params={"layer": 2, "n": 5}).json()
    check(roles_after != roles_before and all(x["z"] == 0 for x in roles_after["roles"]["q"]), "/neuron/heads: not the cache from before the edit")
    nl = c.get("/api/atlas/neuron/lens", params={"layer": 2, "n": 5}).json()
    check(nl["edited"] and "writes_nothing" in nl["faces"]["+"]["check"]["reasons"], "/neuron/lens: read as it now is")
    check(nl["steer"] is None and "before the vault edit" in nl.get("steer_stale", "") and nl["ceiling"]["write"] == 0.0,
          "/neuron/lens: the index's steer lens, read before the edit, is not shown as its own; a silenced neuron can write nothing")
    st_own = c.get("/api/atlas/neuron/passages", params={"layer": 2, "n": 5}).json().get("stale", "")
    st_later = c.get("/api/atlas/neuron/passages", params={"layer": 4, "n": 7}).json().get("stale", "")
    check("vault edit in force on this neuron" in st_own and "earlier layers" in st_later
          and "stale" not in c.get("/api/atlas/neuron/passages", params={"layer": 1, "n": 7}).json()
          and "stale" not in c.get("/api/atlas/neuron/passages", params={"layer": 2, "n": 6}).json(),
          "/neuron/passages: stale on the edited neuron and on later layers, not on its own layer's others or earlier ones")
    check("stale" in c.get("/api/atlas/web/summary", params={"layer": 2, "n": 5}).json() and "stale" in c.get("/api/atlas/cone", params={"layer": 2, "n": 5}).json()
          and c.get("/api/atlas/web", params={"layer": 2, "n": 5}).headers.get("X-Web-Stale") == "1"
          and "links_stale" in c.get("/api/atlas/hover", params={"layer": 2, "n": 5}).json(),
          "/web, /web/summary, /cone and /hover: an edited neuron's web and links marked as read before the edit")
    down._memit_col_overlays = {}
    vault.active_patches.clear()
    check(c.get("/api/atlas/neuron/wiring", params={"layer": 2, "n": 5}).json()["source"] == "index", "/neuron/wiring: back to the index once the edit is gone")
    section("a vault edit in force: read live, marked stale, caches not reused", before)

    # ---------------------------------------------------------------------------------------------- 8. field notes
    before = len(failures)
    notes_file = Path(os.environ["THEODOLITE_FIELD_NOTES"])
    g = c.get("/api/atlas/notes").json()
    check(g["neurons"] == {} and g["heads"] == {} and g["file"] == str(notes_file) and g["warning"] is None, f"/notes empty: {g}")
    r = c.post("/api/atlas/notes", json={"kind": "neuron", "layer": 2, "index": 5, "name": " bob detector ", "note": "fires on Bob\nand his car"})
    check(r.status_code == 200 and r.json()["entry"]["name"] == "bob detector" and r.json()["entry"]["by"] == "reader", f"/notes: kept ({r.text[:160]})")
    check(notes_file.exists() and json.loads(notes_file.read_text())["neurons"]["2:5"]["note"] == "fires on Bob\nand his car", "/notes: written to the file")
    check(c.get("/api/atlas/hover", params={"layer": 2, "n": 5}).json()["name"] == "bob detector", "/hover: the name")
    linked = c.get("/api/atlas/neuron/wiring", params={"layer": 2, "n": 5}).json()["down"][0]
    c.post("/api/atlas/notes", json={"kind": "neuron", "layer": linked["layer"], "index": linked["n"], "name": "a reader", "by": "scientist"})
    check(c.get("/api/atlas/neuron/wiring", params={"layer": 2, "n": 5}).json()["down"][0]["name"] == "a reader", "/neuron/wiring: a linked neuron's name")
    r = c.post("/api/atlas/notes", json={"kind": "head", "layer": 3, "index": 1, "name": "previous-word head", "by": "scientist"})
    check(r.status_code == 200 and c.get("/api/atlas/notes").json()["heads"]["3:1"]["by"] == "scientist", "/notes: a head's name")
    for bad, says in [({"kind": "x", "layer": 0, "index": 0, "name": "a"}, 'kind must be "neuron" or "head"'),
                      ({"kind": "neuron", "layer": 6, "index": 0, "name": "a"}, "Layer 6 does not exist"),
                      ({"kind": "neuron", "layer": 0, "index": 128, "name": "a"}, "Neuron 128 does not exist"),
                      ({"kind": "head", "layer": 0, "index": 4, "name": "a"}, "has 4 attention heads"),
                      ({"kind": "neuron", "layer": 0, "index": 0, "name": "a\nb"}, "control characters"),
                      ({"kind": "neuron", "layer": 0, "index": 0, "name": "a" * 121}, "at most 120"),
                      ({"kind": "neuron", "layer": 0, "index": 0, "name": "a", "by": "someone"}, "by must be one of"),
                      ({"kind": "neuron", "index": 0, "name": "a"}, "names its layer and its index")]:
        r = c.post("/api/atlas/notes", json=bad)
        check(r.status_code == 400 and says in r.json()["detail"], f"/notes refused {bad}: {r.status_code} {r.text[:120]}")
    r = c.post("/api/atlas/notes", json={"kind": "neuron", "layer": 2, "index": 5, "name": "bob's car", "by": "scientist"})
    check(r.json()["entry"]["note"] == "fires on Bob\nand his car" and r.json()["entry"]["by"] == "scientist", "/notes: a new name keeps the note")
    r = c.post("/api/atlas/notes", json={"kind": "neuron", "layer": 2, "index": 5, "note": "only Bob"})
    check(r.json()["entry"]["name"] == "bob's car" and r.json()["entry"]["note"] == "only Bob", "/notes: a new note keeps the name")
    r = c.post("/api/atlas/notes", json={"kind": "neuron", "layer": 2, "index": 5})
    check(r.status_code == 400 and "Give a name, a note" in r.json()["detail"], "/notes: neither a name nor a note refused")
    r = c.post("/api/atlas/notes", json={"kind": "neuron", "layer": 2, "index": 5, "name": ""})
    check(r.json()["entry"] == {**r.json()["entry"], "name": "", "note": "only Bob"}, "/notes: an empty name clears the name only")
    r = c.post("/api/atlas/notes", json={"kind": "neuron", "layer": 2, "index": 5, "name": "", "note": ""})
    check(r.json()["entry"] is None and "2:5" not in c.get("/api/atlas/notes").json()["neurons"], "/notes: an empty name and note forget it")
    notes_file.write_text("{not json", encoding="utf-8")
    g = c.get("/api/atlas/notes").json()
    asides = list(notes_file.parent.glob("field_notes.json.corrupt-*"))
    check(g["neurons"] == {} and g["warning"] and "moved aside" in g["warning"] and len(asides) == 1, f"/notes: a corrupt file moved aside ({g['warning']})")
    check(c.get("/api/atlas/notes").json()["warning"] is None and asides[0].read_text() == "{not json", "/notes: the warning once, the old file kept")
    section("field notes: kept, shown, forgotten, refused, a corrupt file moved aside", before)

    # ------------------------------------------------------------------------------------------------- 9. census
    before = len(failures)
    cdir = cz.census_dir()
    check(cdir == TMP / "census", f"the census folder: {cdir}")
    cdir.mkdir(parents=True, exist_ok=True)
    run = cdir / "L3_main_rel.jsonl"
    rows_c = []

    def row(kind: str, n: int, prompt_: str, dose: float, text_: str) -> Dict[str, Any]:
        ids_ = tok.encode(text_, add_special_tokens=False)
        return {"layer": 3, "kind": kind, "neuron": n, "prompt": prompt_, "dose": dose, "push": dose * 10.0, "rel_push": abs(dose),
                "text": text_, "finish": "stop", "n_gen": len(ids_), "token_ids": ids_, "logprob": [-1.0] * len(ids_),
                "top_ids": [[i] for i in ids_], "top_lps": [[-1.0]] * len(ids_)}

    for p_ in ("who", "snake", "orange"):
        rows_c.append(row("baseline", -1, p_, 0.0, "I am a helpful model."))
    for n_, word_ in ((7, "ocean waves"), (9, "banana bread")):
        for p_ in ("who", "snake", "orange"):
            for dose in (-0.7, 0.5, 0.8):
                rows_c.append(row("neuron", n_, p_, dose, f"I think about {word_} and more {word_}." if dose > 0 else "I am a helpful model, plainly."))
    for n_ in range(4):
        for p_ in ("who", "snake", "orange"):
            rows_c.append(row("random", n_, p_, 0.5, f"Some words number {n_} here, plainly."))
    run.write_text("".join(json.dumps(x) + "\n" for x in rows_c), encoding="utf-8")
    cz.meta_path_of(run).write_text(json.dumps({"layer": 3, "design_name": "main_rel", "prompts": cz.PROMPTS, "version": cz.CENSUS_VERSION,
                                                "doses_from": cz.DOSES_FROM["main_rel"], "p50_source": "the study's 36 texts (no stats index yet)"}),
                                    encoding="utf-8")
    old = cdir / "L2_main_rel.jsonl"
    old.write_text(run.read_text(encoding="utf-8").replace('"layer": 3', '"layer": 2'), encoding="utf-8")
    cz.meta_path_of(old).write_text(json.dumps({"layer": 2, "design_name": "main_rel", "prompts": cz.PROMPTS}), encoding="utf-8")
    units = c.get("/api/atlas/census/units", params={"layer": 3}).json()
    check(cz.units_path_of(run).exists() and len(units) == 6, f"/census/units: six units ranked, written beside the run ({len(units)})")
    top_neuron = [u for u in units if u["kind"] == "neuron"][0]
    check(top_neuron["n"] in (7, 9) and set(top_neuron["words"]) & {"ocean", "waves", "banana", "bread"} and top_neuron["null_pct"] == 1.0,
          f"/census/units: a neuron's own words, above every control ({top_neuron})")
    un = c.get("/api/atlas/census/unit", params={"layer": 3, "n": 7}).json()
    check(len(un["rollouts"]) == 9 and len(un["baselines"]) == 3 and un["rank"] in (1, 2) and un["n_ranked"] == 2 and un["design"] == "main_rel",
          "/census/unit: its nine answers, the three unpushed, its rank")
    check(un["prompts"] == cz.PROMPTS and "ocean" in [w_ for w_, _, _ in un["stats"]["words"]], "/census/unit: the questions and its words")
    card = un.get("card") or {}
    check(card.get("doses_from") == cz.DOSES_FROM["main_rel"] and "36 texts" in (card.get("p50_source") or "")
          and card.get("null") == {"neurons": 2, "controls": 4, "above_all": 2, "above_all_by_chance": 0.4, "q05": 0}
          and un["stats"]["p"] == 0.2 and un["stats"]["q"] == 0.2,
          f"/census/unit: the card says where the doses and the unit come from and what the null can single out ({card}, p {un['stats'].get('p')})")
    check(c.get("/api/atlas/census/card", params={"layer": 3}).json() == card and c.get("/api/atlas/census/card", params={"layer": 1}).json() == {},
          "/census/card: the same card, and none for a layer without a census")
    arch_c = c.get("/api/atlas/arch").json()
    check("2" not in arch_c["census"] and "isotropic" in arch_c.get("census_refused", {}).get("2", ""),
          "/arch: a run taken with the old isotropic controls is not read, and says why")
    ro = c.get("/api/atlas/census/rollout", params={"layer": 3, "i": 3}).json()
    check(ro["text"] == rows_c[3]["text"] and ro["neuron"] == 7, "/census/rollout: one answer")
    check(c.get("/api/atlas/census/rollout", params={"layer": 3, "i": 999}).status_code == 404, "/census/rollout: no such answer")
    check(c.get("/api/atlas/census/spread", params={"layer": 3}).json() == {}, "/census/spread: none yet")
    cz.spread_path_of(run).write_text(json.dumps({"x": [0.5], "y": [0.25], "n": [7]}), encoding="utf-8")
    check(c.get("/api/atlas/census/spread", params={"layer": 3}).json() == {"x": [0.5], "y": [0.25], "n": [7]}, "/census/spread: as written")
    check(c.get("/api/atlas/census/units", params={"layer": 1}).json() == [], "/census/units: a layer without a census")
    check(c.get("/api/atlas/arch").json()["census"] == {"3": run.name} and c.get("/api/health").json()["atlas"]["census_layers"] == [3],
          "/arch and /api/health name the layer with a census")
    check(c.get("/api/atlas/neuron", params={"layer": 3, "n": 7}).json()["census"]["n_rollouts"] == 9, "/neuron: its census")
    atlas_routes._runs_cache.clear()
    cw = atlas_routes.census_words(3, rt.model_id)
    check(set(cw) == {7, 9} and "ocean" in cw[7] and "banana" in cw[9], f"census words of a ranked layer: {cw}")
    into3 = [x for x in c.get("/api/atlas/neuron/wiring", params={"layer": 1, "n": 4, "k": 64}).json()["down"] if x["layer"] == 3]
    check(all(x["census"] == cw.get(x["n"], []) for x in into3), "/neuron/wiring: a linked neuron's census words")
    section("the census routes", before)

    # ----------------------------------------------------------------------------------------------- 10. the GPU lock
    before = len(failures)
    import httpx

    async def lock_waits() -> None:
        lock = asyncio.Lock()
        atlas_routes.bind(server.get_app_state, lock)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE, timeout=60) as ac:
            await lock.acquire()
            try:
                t = asyncio.create_task(ac.post("/api/atlas/reads", json={"layer": 3, "n": 7, "text": text}))
                await asyncio.sleep(0.5)
                check(not t.done(), "a reading waits for the GPU lock")
                r1 = await asyncio.wait_for(ac.get("/api/atlas/arch"), 20)
                r2 = await asyncio.wait_for(ac.get("/api/atlas/notes"), 20)
                check(r1.status_code == 200 and r2.status_code == 200, "/arch and the notes answer while the lock is held")
            finally:
                lock.release()
            r3 = await asyncio.wait_for(t, 60)
            check(r3.status_code == 200 and len(r3.json()["units"]) == len(units_text), "the reading runs once the lock is free")

    asyncio.run(lock_waits())

    # a real server: a push chat whose page closes mid-answer
    import uvicorn

    record: Dict[str, Any] = {"steps": 0, "cancelled": None, "end": None, "read_start": None}
    real_write, real_read = ao.write_rows, ao.read_neuron

    def slow_write(*a_: Any, **kw: Any):
        try:
            for st_ in real_write(*a_, **kw):
                record["steps"] += 1
                time.sleep(0.004)
                yield st_
        finally:
            record["cancelled"] = kw["cancel"]()
            record["end"] = time.time()

    def timed_read(*a_: Any, **kw: Any):
        record["read_start"] = time.time()
        return real_read(*a_, **kw)

    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    atlas_routes.bind(server.get_app_state, asyncio.Lock())
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(200):
        if srv.started:
            break
        time.sleep(0.05)
    ao.write_rows, ao.read_neuron = slow_write, timed_read
    stop_ids = rt.end_of_turn_ids
    rt.end_of_turn_ids = lambda: []  # type: ignore[method-assign]  # no row ends of itself: the writer runs until stopped
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=60) as hc:
            with hc.stream("POST", "/api/atlas/push", json={"layer": 2, "n": 5, "prompt": "Write whatever you like.", "max_new": 500}) as resp:
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        break
            closed_at = time.time()
            r = hc.post("/api/atlas/reads", json={"layer": 3, "n": 7, "text": text})
            check(r.status_code == 200, f"a reading after the closed push chat: {r.status_code}")
        check(record["cancelled"] is True and record["steps"] < 3 * 500, f"the push chat stopped writing when its page closed ({record['steps']} pieces)")
        check(record["end"] is not None and record["read_start"] is not None and record["read_start"] >= record["end"],
              "the lock was held until the writer had stopped")
        check(record["end"] - closed_at < 10, f"the writer stopped soon after ({record['end'] - closed_at:.2f} s)")
    finally:
        ao.write_rows, ao.read_neuron = real_write, real_read
        rt.end_of_turn_ids = stop_ids  # type: ignore[method-assign]
        srv.should_exit = True
        th.join(timeout=10)
    section("the GPU lock: readings wait, the server answers, a closed push chat stops", before)

    print(f"\n{passed} checks passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(code)

#!/usr/bin/env python3
"""A CPU-only check of the atlas's indexes (backend/atlas_build.py builds them, backend/atlas_index.py reads them) on the
tiny random Gemma 4 of scripts/atlas_stand_in.py. Every builder runs into a temporary folder, and each file is compared
with the same quantity read live, one neuron at a time:
1. `forward_from_layer` from any layer, on right-padded rows, gives the whole forward's final states;
2. lens.npz: each face's first ten word-pieces through the lens and by the naked eye are the ids the vault's
   DESCRIBE POLE ranks for its label (`label_top_ids`), and the index's label check is the vault's, count and reasons;
3. act_scale.npz: the extremes and the stream's quantiles of `activation_extremes` over the same windows, the norms
   kept in float32 buffers giving exactly the quantiles of the Python lists they replace;
4. passages.d: by brute force over the same windows (`all_activations` a window, the gate and up values by hooks),
   each face's stored K are the K largest per-window peaks, the same windows and positions in order (ties, which the
   stand-in's sliding layers make, to the lower window), their activation, gate and up values as float16 keeps them, the first the stats index's extreme;
   a face peaking on two word-pieces of one window (made so by copying one's gate and up values to the other) keeps the
   first; each stored window read again (`read_neurons`) gives the stored activation; three part sizes (one of a
   single window, which puts every tie between windows across a part's edge), and a run stopped after its first part
   and resumed, merge to the same arrays, and a run with other settings is refused; a folder missing an array is
   refused in plain words; the sampled events lie
   above half the face's extreme, at most one a window, as many as such windows allow; the unigram counts add up to
   the word-pieces read; the server reads the arrays memory-mapped;
5. wiring.npz: each neuron's links each way are `neuron_wiring`'s, the same neurons, z (as float16 keeps it) and sign,
   and each link's g and u as float16 keeps them; each neuron's operating point on each face is the median over its
   stored passages' events; each link's sign in use is the same from the index's g and u as from the live ones, and is
   gate x sign(alpha g + beta u) written out at its reader's brute-force operating point (each reader's downstream, the
   open neuron's upstream); the count of signs in use that differ from rest is the count link by link; a wiring index
   without g and u still serves the sign from rest and says, in use, to rebuild it;
5b. wiring_check.npz (the wiring measured): each writer's linked readers are its links in the wiring index, the null's
   at the same layers and never linked; each push is the dose times the writer's ceiling; the reads and each reader's
   g0 and u0 are `pushed_reads`' in another batch shape; the signs, g, u, f and strength are the index's and the
   weights'; the summary in the meta is the stored arrays', every share (a change of 0 left out), the sign in use where
   each reader sits (from its g0 and u0 in Python floats) and at the passages' operating point, each regime (with
   every g0 moved down by 1 too, which puts readers below gelu's minimum), rho against |P| f^2 and against the
   first-order size where the reader sits, and the size, recounted pair by pair by hand at both read sites, linked and
   null (Spearman's rho also on cases with ties); the sign in use where the reader sits is a float32 central
   difference's of gelu(g0 + e g) (u0 + e u) at nearly every pair; a layer without g0, u0 and f (measured before they
   were recorded) is summed up as it was then, by the recount of then, and says its site score needs a rerun; the
   server reads only its meta; a second run adds layers and skips those done, past a half-written shard, and one with
   another dose, other scales or against another passages index is refused, as is a batch below 3; a run against a
   folder measured before g0 and u0 were recorded is refused in plain words and the folder left as it is, `fresh` sets
   it and wiring_check.npz aside, renamed, and starts again; a layer of the older format in a folder of this one, or
   with the plan lost, is refused before anything is measured or a plan written (and at the merge, the backstop); and
   an older merged wiring_check.npz with no folder beside it is refused, not replaced;
6. steer_lens.d / steer_lens.npz: each neuron's rises are `steer_lens`'s, KL too; a stopped run resumes and merges;
7. layout_writes.npz: each layer's cells are distinct places on its sheet;
8. the queries: a search finds a face by a word-piece of its label at the right rank; the cone's hops move one way
   each side and start from the neuron; the web summary pools the steer lens when built and, without it, only faces
   the label gate lets through (on the stand-in's six layers every face is held back, so it pools nothing and says how
   many it left out); the active paths join only neurons that fired; the scores scale by each side's extreme; in use,
   a reader is reached on the face its writer's firing pushes it further toward (the harder one where both are),
   a writer upstream on the face whose firing pushes its reader the way asked, the active paths keep every link
   whose ends both fired, signed at the reader's own face (each against alpha g + beta u written out at the reader's
   brute-force operating point), and without the passages index the web summary says it is missing and the active
   paths, none, still count the neurons that fired;
9. head profiles: each head's shares of weight add up to one per read, and its pairs read more than one unit back;
10. provenance: every index's meta names the weights it was built from (runtime_factory.weights_card), and an index
   built for another model, of another shape, by an older build or from other weights (another GGUF behind the same
   model id, other weight files of the checkpoint, not another snapshot of the same: atlas_index.weights_differ) is
   not read, in plain words; one built before metas named their weights is read on the model id alone; a steer lens
   layer or a passages part from other weights is refused, not resumed into; census_run.py says a stats index refused
   for its weights is there and why, in the push unit's source too; atlas_build.py does not write over an index built
   for another model or from other weights unless given --replace, and says so when it does.

Run: .venv/bin/python -I scripts/check_atlas_index.py   (exits 0 when every check holds)
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"

import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch

import backend.atlas_build as ab
import backend.atlas_ops as ao
from atlas_stand_in import stand_in
from backend import runtime_factory as rf
from backend.atlas_index import SITE_SCORE_RERUN, AtlasIndex, match_ids, scale_scores, weights_differ
from backend.causal_check import language_model, pad_right
from backend.label_gate import label_top_ids
from backend.theodolite_math import unembed_gemma4

failures: List[str] = []
passed = 0


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


def old_norm_lists(lm: torch.nn.Module, wins: List[List[int]]) -> dict:
    """The three norms per layer as the stats pass kept them before its buffers: Python floats, extended a text at a
    time (the reference the buffers are checked against)."""
    L = len(lm.layers)
    got = {key: [[] for _ in range(L)] for key in ("resid_q", "mlp_rms_q", "mid_rms_q")}
    hooks = []
    for l, layer in enumerate(lm.layers):
        e_post, e_pre = float(layer.post_feedforward_layernorm.eps), float(layer.pre_feedforward_layernorm.eps)
        hooks.append(layer.register_forward_hook(lambda _m, _a, o, l=l: got["resid_q"][l].extend(
            (o[0] if isinstance(o, tuple) else o)[0, 1:].float().norm(dim=-1).tolist())))
        hooks.append(layer.mlp.register_forward_hook(lambda _m, _a, o, l=l, e=e_post: got["mlp_rms_q"][l].extend(
            (o[0, 1:].float().pow(2).mean(-1) + e).sqrt().tolist())))
        hooks.append(layer.pre_feedforward_layernorm.register_forward_pre_hook(lambda _m, a, l=l, e=e_pre: got["mid_rms_q"][l].extend(
            (a[0][0, 1:].float().pow(2).mean(-1) + e).sqrt().tolist())))
    try:
        for ids in wins:
            lm(input_ids=torch.tensor([ids]), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return got


def read_window(lm: torch.nn.Module, ids: List[int]) -> tuple:
    """One window's activation and its gate and up pre-activations at every word-piece, [L, T, N] float32 each, by
    hooks of the check's own (the brute force the passages index is checked against)."""
    L = len(lm.layers)
    g: list = [None] * L
    u: list = [None] * L
    hooks = []
    for l, layer in enumerate(lm.layers):
        hooks.append(layer.mlp.gate_proj.register_forward_hook(lambda _m, _a, o, l=l: g.__setitem__(l, o[0].float().numpy())))
        hooks.append(layer.mlp.up_proj.register_forward_hook(lambda _m, _a, o, l=l: u.__setitem__(l, o[0].float().numpy())))
    try:
        A = ao.all_activations(lm, ids, dtype=torch.float32).numpy()
    finally:
        for h in hooks:
            h.remove()
    return A, np.stack(g), np.stack(u)


# a GGUF's card, as runtime_factory.weights_card gives one (abridged): what an index built on another file names
OTHER_GGUF = {"format": "gguf", "file": "other.gguf", "id": "00ff00ff00ff00ff", "sha256_header": "ab" * 32, "linears": "packed"}


def raises(fn, *args, **kw) -> str:
    """What `fn` raised, in words; "" when it raised nothing."""
    try:
        fn(*args, **kw)
    except Exception as exc:  # noqa: BLE001
        return str(exc) or type(exc).__name__
    return ""


def ranks_by_hand(v: List[float]) -> List[float]:
    """Ranks from 0, a run of ties each given the run's mean rank (scipy-free, and not atlas_build's)."""
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def gelu_by_hand(x: float) -> float:
    """gelu_tanh in Python floats (not the summary's torch)."""
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def rho_by_hand(xs: List[float], ys: List[float]) -> Optional[float]:
    rx, ry = ranks_by_hand(xs), ranks_by_hand(ys)
    mx, my = sum(rx) / max(1, len(rx)), sum(ry) / max(1, len(ry))
    cov = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    vx, vy = sum((x - mx) ** 2 for x in rx), sum((y - my) ** 2 for y in ry)
    return cov / (vx * vy) ** 0.5 if len(rx) >= 3 and vx > 0 and vy > 0 else None


def wiring_summary_by_hand(sh: dict) -> dict:
    """One layer of the measured check summed up again pair by pair, as a layer with the readers' own g0 and u0 is (the
    brute force `wiring_check_summary` is checked against): each (writer, reader, question) of a group, its change under
    +d and -d from the unpushed row, the push on the writer's gate face taken by that gate, a change of exactly 0 left
    out of every share of signs; the sign in use where the reader sits, from its g0 and u0 at the last word-piece (and
    over the prompt each coefficient's mean over the word-pieces) in Python floats, and at the passages' operating point
    on the face it sat on; its regime by its g0 at the last word-piece; Spearman's rho from ranks against |P| f^2 and
    against f |alpha0 g + beta0 u| d."""
    sgn = lambda x: 1 if x > 0 else (-1 if x < 0 else 0)  # noqa: E731
    share = lambda k, n: round(k / n, 3) if n else None  # noqa: E731
    regimes = ("open", "shut_above_min", "below_min")
    out: dict = {}
    for si, site in enumerate(("last", "mean")):
        res: dict = {}
        for name in ("linked", "null"):
            c = {k: 0 for k in ("n", "zero", "np", "plus", "use_plus", "ng", "gate", "use_gate", "nsame", "same", "npp", "plus_peak",
                                 "ngp", "gate_peak")}
            reg = {rg: {"n": 0, "ng": 0, "sign": 0, "use": 0} for rg in regimes}
            sizes, size_q, strength, first = [], [], [], []
            for a in range(sh["readers"].shape[0]):
                for r in range(sh["readers"].shape[1]):
                    if int(sh["readers"][a, r, 0]) < 0 or bool(sh["linked"][a, r]) != (name == "linked"):
                        continue
                    gate, sign = int(sh["gate"][a, r]), int(sh["sign"][a, r])
                    g, u = float(sh["g"][a, r]), float(sh["u"][a, r])
                    dp = sh["reads"][a, r, :, 1, si] - sh["reads"][a, r, :, 0, si]  # [Q], float32 as stored
                    dm = sh["reads"][a, r, :, 2, si] - sh["reads"][a, r, :, 0, si]
                    dg = dp if gate > 0 else dm
                    first_q = []
                    for q in range(len(dp)):
                        gs = [float(x) for x in sh["g0"][a, r, q] if np.isfinite(x)]
                        us = [float(x) for x in sh["u0"][a, r, q] if np.isfinite(x)]
                        if site == "last":
                            gs, us = gs[-1:], us[-1:]
                        al0 = sum(ao.gelu_tanh_grad(x) * y for x, y in zip(gs, us)) / len(gs)
                        be0 = sum(gelu_by_hand(x) for x in gs) / len(gs)
                        kappa = sgn(al0 * g + be0 * u)
                        first_q.append(float(sh["f"][a, r]) * abs(al0 * g + be0 * u))
                        c["n"] += 1
                        c["zero"] += int(dg[q] == 0)
                        if dp[q] != 0:
                            c["np"] += 1
                            c["plus"] += sgn(dp[q]) == sign
                            c["use_plus"] += sgn(dp[q]) == kappa
                        if dg[q] != 0:
                            c["ng"] += 1
                            c["gate"] += sgn(dg[q]) == sign
                            c["use_gate"] += sgn(dg[q]) == gate * kappa
                        if dp[q] != 0 and dm[q] != 0:
                            c["nsame"] += 1
                            c["same"] += sgn(dp[q]) == sgn(dm[q])
                        sizes.append(abs(dg[q]))
                        if site == "last":
                            al, be = (float(x) for x in sh["op"][a, r, 0 if sh["reads"][a, r, q, 0, 0] > 0 else 1])
                            if np.isfinite(al) and np.isfinite(be):
                                kp = sgn(al * g + be * u)
                                if dp[q] != 0:
                                    c["npp"] += 1
                                    c["plus_peak"] += sgn(dp[q]) == kp
                                if dg[q] != 0:
                                    c["ngp"] += 1
                                    c["gate_peak"] += sgn(dg[q]) == gate * kp
                            g_last = float(sh["g0"][a, r, q, -1])
                            rg = "open" if g_last > 0 else ("shut_above_min" if g_last > ao.GELU_MIN else "below_min")
                            reg[rg]["n"] += 1
                            if dg[q] != 0:
                                reg[rg]["ng"] += 1
                                reg[rg]["sign"] += sgn(dg[q]) == sign
                                reg[rg]["use"] += sgn(dg[q]) == gate * kappa
                    size_q.append(float(np.abs(dg).mean()))
                    strength.append(float(sh["strength"][a, r]))
                    first.append(sum(first_q) / len(first_q) * float(sh["lengths"][a]))
            got: dict = {"pairs": c["n"], "zero": share(c["zero"], c["n"]), "sign_plus": share(c["plus"], c["np"]),
                         "sign_gate": share(c["gate"], c["ng"]), "same_way": share(c["same"], c["nsame"]),
                         "use_plus": share(c["use_plus"], c["np"]), "use_gate": share(c["use_gate"], c["ng"])}
            if site == "last":
                got["use_plus_peak"], got["use_gate_peak"] = share(c["plus_peak"], c["npp"]), share(c["gate_peak"], c["ngp"])
                got["regimes"] = {rg: {"share": share(reg[rg]["n"], c["n"]), "sign_gate": share(reg[rg]["sign"], reg[rg]["ng"]),
                                       "use_gate": share(reg[rg]["use"], reg[rg]["ng"])} for rg in regimes}
                got["n_peak"] = c["ngp"]
            got["rho"], got["rho_site"] = rho_by_hand(size_q, strength), rho_by_hand(size_q, first)
            got["moved"] = float(np.median(np.array(sizes, np.float32))) if sizes else None
            res[name] = got
        mv, mn = res["linked"]["moved"], res["null"]["moved"]
        res["moved_over_null"] = round(mv / mn, 2) if mv is not None and mn else None
        out[site] = res
    return out


def gelu32(x: np.ndarray) -> np.ndarray:
    """gelu_tanh in float32 numpy, every step rounded to float32 (the finite differences' own arithmetic)."""
    x = np.asarray(x, np.float32)
    c, k, h, one = np.float32(math.sqrt(2.0 / math.pi)), np.float32(0.044715), np.float32(0.5), np.float32(1.0)
    return h * x * (one + np.tanh(c * (x + k * x * x * x)))


def wiring_summary_before_by_hand(sh: dict) -> dict:
    """One layer of the measured check summed up again pair by pair as a layer measured before the readers' g0 and u0
    were recorded is (and was): each (writer, reader, question) of a group, its change under +d and -d from the
    unpushed row, the push on the writer's gate face taken by that gate, the reader's face read from its own unpushed
    row at the last word-piece, the sign in use from that face's stored operating point and the pair's g and u, and
    Spearman's rho from ranks."""
    sgn = lambda x: 1 if x > 0 else (-1 if x < 0 else 0)  # noqa: E731
    share = lambda k, n: round(k / n, 3) if n else None  # noqa: E731
    out: dict = {}
    for si, site in enumerate(("last", "mean")):
        res: dict = {}
        for name in ("linked", "null"):
            n, plus, on_gate, same, n_use, use_plus, use_gate, sizes, size_q, strength = 0, 0, 0, 0, 0, 0, 0, [], [], []
            for a in range(sh["readers"].shape[0]):
                for r in range(sh["readers"].shape[1]):
                    if int(sh["readers"][a, r, 0]) < 0 or bool(sh["linked"][a, r]) != (name == "linked"):
                        continue
                    gate, sign = int(sh["gate"][a, r]), int(sh["sign"][a, r])
                    dp = sh["reads"][a, r, :, 1, si] - sh["reads"][a, r, :, 0, si]  # [Q], float32 as stored
                    dm = sh["reads"][a, r, :, 2, si] - sh["reads"][a, r, :, 0, si]
                    dg = dp if gate > 0 else dm
                    for q in range(len(dp)):
                        n += 1
                        plus += sgn(dp[q]) == sign
                        on_gate += sgn(dg[q]) == sign
                        same += int(sgn(dp[q]) == sgn(dm[q]) and dp[q] != 0)
                        sizes.append(abs(dg[q]))
                        if site == "last":
                            al, be = (float(x) for x in sh["op"][a, r, 0 if sh["reads"][a, r, q, 0, 0] > 0 else 1])
                            if np.isfinite(al) and np.isfinite(be):
                                kappa = sgn(al * float(sh["g"][a, r]) + be * float(sh["u"][a, r]))
                                n_use += 1
                                use_plus += sgn(dp[q]) == kappa
                                use_gate += sgn(dg[q]) == gate * kappa
                    size_q.append(float(np.abs(dg).mean()))
                    strength.append(float(sh["strength"][a, r]))
            got: dict = {"pairs": n, "sign_plus": share(plus, n), "sign_gate": share(on_gate, n), "same_way": share(same, n)}
            if site == "last":
                got["use_plus"], got["use_gate"] = share(use_plus, n_use), share(use_gate, n_use)
                got["n_use"] = n_use
            rx, ry = ranks_by_hand(size_q), ranks_by_hand(strength)
            mx, my = sum(rx) / max(1, len(rx)), sum(ry) / max(1, len(ry))
            cov = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
            vx, vy = sum((x - mx) ** 2 for x in rx), sum((y - my) ** 2 for y in ry)
            got["rho"] = cov / (vx * vy) ** 0.5 if len(rx) >= 3 and vx > 0 and vy > 0 else None
            got["moved"] = float(np.median(np.array(sizes, np.float32))) if sizes else None
            res[name] = got
        mv, mn = res["linked"]["moved"], res["null"]["moved"]
        res["moved_over_null"] = round(mv / mn, 2) if mv is not None and mn else None
        out[site] = res
    return out


class Stop(Exception):
    """A build stopped part-way, as a killed process would stop it."""


def main() -> int:
    torch.set_grad_enabled(False)
    rt, vault = stand_in()
    lm = language_model(rt.hf_model)
    w = ao.Weights(lm, rt)
    L, N = w.n_layers, w.d_ff
    tok = rt.tokenizer
    out = Path(tempfile.mkdtemp(prefix="atlas-index-"))
    idx = AtlasIndex(out)
    print(f"stand-in: {L} layers x {N} neurons; index folder {out}")

    before = len(failures)
    ctx = ab.steer_contexts(rt)
    inp, att, last, _ = pad_right(ctx, w.device)
    pos = (att.cumsum(-1) - 1).clamp(min=0)
    full = lm(input_ids=inp, attention_mask=att, use_cache=False).last_hidden_state
    for start in (1, 3, L - 1):
        seen: dict = {}
        hk = lm.layers[start - 1].register_forward_hook(lambda _m, _a, o: seen.__setitem__("h", o))
        lm(input_ids=inp, attention_mask=att, use_cache=False)
        hk.remove()
        got = ao.forward_from_layer(lm, start, seen["h"], att, pos)
        real = att.bool()
        check(torch.allclose(got[real], full[real], atol=1e-5), f"forward_from_layer({start}) differs from the whole forward")
    section("forward_from_layer gives the whole forward's states (right-padded rows)", before)

    before = len(failures)
    ab.build_lens(rt, out, chunk=48, log=quiet)
    d = idx.get("lens")
    check(d is not None and d["lens_pos"].shape == (L, N, 12), "lens: shape")
    fnw, W_U = rt.final_norm_weight, rt.lm_head_weight
    for (l, n) in ((0, 0), (2, 47), (L - 1, 100), (4, N - 1)):
        r = vault.inspect_bipolar_slot(l, n)
        J = rt.jpp_lenses[l]
        col = (w.down_cols(l, [n])[0] * w.norm(l, "post_feedforward")).to(J.dtype)
        for face, sgn, key in (("+", 1, "north_pole"), ("-", -1, "south_pole")):
            vec = (sgn * col)[None]
            vj = vec @ J.T
            want_lens = label_top_ids(vj, fnw, W_U, unembed_gemma4(vj, fnw, W_U))[0]
            want_eye = label_top_ids(vec, fnw, W_U, unembed_gemma4(vec, fnw, W_U))[0]
            check(idx.face_ids(l, n, face, "lens", 10) == want_lens, f"lens: L{l} N{n} {face} through the lens")
            check(idx.face_ids(l, n, face, "eye", 10) == want_eye, f"lens: L{l} N{n} {face} by the naked eye")
            check(idx.face_check(l, n, face) == r[key]["label_check"], f"lens: L{l} N{n} {face} label check {idx.face_check(l, n, face)} != {r[key]['label_check']}")
    check(idx.face_ids(L - 1, 5, "+", "lens") == idx.face_ids(L - 1, 5, "+", "eye"), "lens: the last layer's two readings differ")
    # a build of some layers: an unread layer has no words (not empty ones the gate would pass), and the next build of
    # other layers adds to the file rather than wiping the first
    part = out / "partial"
    pidx = AtlasIndex(part)
    ab.build_lens(rt, part, chunk=48, layers=[1], log=quiet)
    check(pidx.face_ids(1, 7, "+", "lens") == idx.face_ids(1, 7, "+", "lens") and pidx.face_ids(4, 7, "+", "lens") is None
          and pidx.face_check(4, 7, "+") is None and not pidx.label_passes(4, 7, "+"), "lens: a layer not yet read has no words and no check")
    w4 = idx.face_ids(4, 7, "+", "lens", 12)
    hit_rows, _ = pidx.search(np.array(w4[:3], np.int32), limit=100000)
    check(not any(r_["layer"] == 4 and r_["src"] in ("lens", "eye") for r_ in hit_rows), "lens: search finds nothing in a layer not yet read")
    ab.build_lens(rt, part, chunk=48, layers=[4], log=quiet)
    check(pidx.face_ids(1, 7, "+", "lens") == idx.face_ids(1, 7, "+", "lens") and pidx.face_ids(4, 7, "+", "lens") == idx.face_ids(4, 7, "+", "lens")
          and pidx.get("lens")["built"].tolist() == [i_ in (1, 4) for i_ in range(L)], "lens: a second partial build adds its layers")
    old = dict(np.load(out / "lens.npz"))
    del old["built"]
    np.savez(part / "lens.npz", **old)
    check(pidx.get("lens") is None and "which layers" in pidx.status()["lens"]["refused"], "lens: a file without its layer marks is refused")
    section("the lens index is the vault's ranking and label check, face by face; a partial build adds layers", before)

    before = len(failures)
    texts = [rt.compose_prompt(t, chat=True) for t in ("Who are you?", "Tell me about Orange.")] + list(ao.HEAD_TEXTS)
    wins = ab.windows(rt, texts, 24, 400)
    check(all(wn[0] == tok.bos_token_id and len(wn) <= 24 for wn in wins), "stats: windows begin with the start mark")
    ab.build_stats(rt, out, wins, "check", log=quiet)
    s = idx.get("act_scale")
    ex = ao.activation_extremes(lm, wins)
    check(np.allclose(s["pos"], ex.pos.numpy()) and np.allclose(s["neg"], ex.neg.numpy()), "stats: extremes differ")
    check(np.allclose(s["resid_q"][:, 1], ao.stream_p50(ex), rtol=1e-5), "stats: the stream's median differs")
    ns = ao.norm_scales(ex)
    check(np.allclose(s["mlp_rms_q"][:, 1], ns["mlp_rms"], rtol=1e-5) and np.allclose(s["mid_rms_q"][:, 1], ns["mid_rms"], rtol=1e-5),
          "stats: the feed-forward norms' scales are activation_extremes'")
    check(int(s["tokens"]) == ex.tokens and s["meta"].get("corpus") == "check", "stats: provenance")
    # the norms are kept in float32 buffers, one value a word-piece: the quantiles are those of the Python floats the
    # pass once kept (gathered here by hooks of that old kind), exactly
    check(all(isinstance(v, np.ndarray) and v.dtype == np.float32 and v.shape == (ex.tokens,) for vs in (ex.resid_norms, ex.mlp_rms, ex.mid_rms) for v in vs),
          "stats: each norm a float32 buffer of one value a word-piece read")
    old_lists = old_norm_lists(lm, wins)
    old_q = lambda vs: np.array([np.quantile(np.array(v, np.float32), [0.1, 0.5, 0.9]) for v in vs], np.float32)  # noqa: E731
    check(all(np.array_equal(s[key], old_q(old_lists[key])) for key in ("resid_q", "mlp_rms_q", "mid_rms_q")),
          "stats: the quantiles from the float32 buffers are those of the old Python lists, exactly")
    check(ao.stream_p50(ex) == [float(np.median(v)) for v in old_lists["resid_q"]]
          and ao.norm_scales(ex)["mid_rms"] == [float(np.median(v)) for v in old_lists["mid_rms_q"]],
          "stats: the push's unit and the norms' scales are the old lists' medians, exactly")
    section("the activation extremes, the stream's quantiles and the norms' scales are activation_extremes'; the norms kept in float32", before)

    before = len(failures)
    K, S = 5, 3
    ext = (s["pos"], s["neg"])
    ab.build_passages(rt, out, wins, "check", k=K, sample=S, extremes=ext, extremes_source="the stats index", seed=7, chunk=4, log=quiet)
    P = idx.get("passages")
    check(P is not None and P["pos_act"].shape == (L, N, K) and P["neg_smp_win"].shape == (L, N, S), "passages: shapes")
    check(isinstance(P["pos_act"], np.memmap) and isinstance(P["windows"], np.memmap), "passages: the arrays are memory-mapped")
    m = P["meta"]
    check(m.get("corpus") == "check" and m.get("tokens") == ex.tokens and m.get("windows") == len(wins) and m.get("K") == K
          and m.get("per_window") == 1 and m.get("seed") == 7 and m.get("sample") == S and "stats index" in m.get("sample_threshold", ""),
          f"passages: provenance {m}")
    reads = [read_window(lm, wn) for wn in wins]
    h16 = np.float16
    mism, ties = [], 0
    for f, sgn in (("pos", 1.0), ("neg", -1.0)):
        for l in range(L):
            for n in range(N):
                # each window's peak past the start mark (the first position reaching it), where it fires on this face
                peaks = []
                for wi, (A, _G, _U) in enumerate(reads):
                    col = sgn * A[l, 1:, n]
                    t = int(np.argmax(col))
                    if col[t] > 0:
                        peaks.append((-float(col[t]), wi, t + 1))
                top = sorted(peaks)[:K]  # by value, then the lower window
                want = [(wi, t) for _, wi, t in top]
                ties += any(top[i][0] == top[i + 1][0] for i in range(len(top) - 1))
                got = [(int(P[f"{f}_win"][l, n, i]), int(P[f"{f}_at"][l, n, i])) for i in range(K) if int(P[f"{f}_win"][l, n, i]) >= 0]
                if got != want:
                    mism.append((f, l, n, got[:3], want[:3]))
                    continue
                for i, (wi, t) in enumerate(want):
                    A, G, U = reads[wi]
                    if not (P[f"{f}_act"][l, n, i] == h16(A[l, t, n]) and P[f"{f}_g"][l, n, i] == h16(G[l, t, n])
                            and P[f"{f}_u"][l, n, i] == h16(U[l, t, n])):
                        mism.append((f, l, n, i, "value"))
                if any(int(P[f"{f}_win"][l, n, i]) != -1 for i in range(len(want), K)):
                    mism.append((f, l, n, "empty slot"))
    check(not mism, f"passages: the stored K a face are the brute force's per-window peaks ({len(mism)} differ: {mism[:3]})")
    # the stand-in's sliding layers read only the last eight word-pieces, so equal stretches of text in two windows tie
    # exactly: the rule for ties is tested, not only stated
    check(ties > 0, "passages: some faces' top K hold exact ties between windows, which the lower window wins")
    # and a tie within a window: the gate and up outputs at one word-piece copied to a later one at every layer, so every
    # neuron's activation is the same at both; where a face peaks on that pair, the first of the two is kept
    t1, t2 = 3, 9
    seen: dict = {}

    def copy_row(_m, _a, o):
        o = o.clone()
        o[:, t2] = o[:, t1]
        return o

    hooks = [h for layer in lm.layers for h in (layer.mlp.gate_proj.register_forward_hook(copy_row), layer.mlp.up_proj.register_forward_hook(copy_row))]
    hooks += [layer.mlp.down_proj.register_forward_pre_hook(lambda _m, a_, l=l: seen.__setitem__(l, a_[0][0].float().numpy().copy()))
              for l, layer in enumerate(lm.layers)]
    try:
        E = ao.passage_events(lm, [wins[1]], k=1)
    finally:
        for h in hooks:
            h.remove()
    on_pair = [(f, l, n) for f, sgn in (("pos", 1.0), ("neg", -1.0)) for l in range(L) for n in range(N)
               if sgn * seen[l][t1, n] > 0 and sgn * seen[l][t1, n] == (sgn * seen[l][1:, n]).max()]
    check(all(np.array_equal(seen[l][t1], seen[l][t2]) for l in range(L)) and bool(on_pair)
          and all(int(E[f"{f}_at"][l, n, 0]) == t1 for f, l, n in on_pair),
          f"passages: a face peaking on two word-pieces of one window keeps the first ({len(on_pair)} such faces)")
    check(all(P["pos_act"][l, n, 0] == h16(s["pos"][l, n]) for l in range(L) for n in range(N) if s["pos"][l, n] > 0)
          and all(P["neg_act"][l, n, 0] == h16(s["neg"][l, n]) for l in range(L) for n in range(N) if s["neg"][l, n] < 0),
          "passages: each face's strongest passage is the stats index's extreme over the same windows")
    check(K < len(wins) and int((P["pos_win"] >= 0).sum(-1).min()) >= 1, "passages: the windows outnumber K, and every + face found some")
    # read again by another route (read_neurons): each stored window gives the stored activation at the stored position
    again = []
    for (l, n) in ((0, 0), (2, 47), (L - 1, 100), (4, N - 1)):
        for f in ("pos", "neg"):
            for i in range(K):
                wi, t = int(P[f"{f}_win"][l, n, i]), int(P[f"{f}_at"][l, n, i])
                if wi < 0:
                    continue
                ids_w = idx.window_ids(wi)
                v = ao.read_neurons(lm, ids_w, [(l, n)])[(l, n)][t]
                again.append(ids_w == list(wins[wi]) and P[f"{f}_act"][l, n, i] == h16(v))
    check(bool(again) and all(again), f"passages: a stored window read again gives the stored activation ({again.count(False)} of {len(again)} do not)")
    # two part sizes, and a run stopped after its first part and resumed, merge to the same arrays
    names = sorted(k_ for k_ in P if k_ != "meta")

    def stop(msg: str) -> None:
        if "part 1 of" in msg:
            raise Stop("stopped")

    # (parts of one window put every tie between two windows across a part's edge, where merge_events decides it)
    for chunk in (7, 1, 3):
        d_ = out / f"passages-{chunk}"
        if chunk == 3:
            said = raises(ab.build_passages, rt, d_, wins, "check", k=K, sample=S, extremes=ext, seed=7, chunk=3, log=stop)
            check(said == "stopped" and (d_ / "passages.d" / "parts" / "part-00000.npz").exists() and not (d_ / "passages.d" / "meta.json").exists()
                  and AtlasIndex(d_).get("passages") is None, "passages: a run stopped after its first part leaves that part and no index")
            said = raises(ab.build_passages, rt, d_, wins, "check", k=K, sample=S, extremes=ext, seed=8, chunk=3, log=quiet)
            check("other settings (seed)" in said, f"passages: a resume with another seed refused in plain words ({said})")
            here = ab._weights_here
            ab._weights_here = lambda _rt: OTHER_GGUF
            try:
                said = raises(ab.build_passages, rt, d_, wins, "check", k=K, sample=S, extremes=ext, seed=7, chunk=3, log=quiet)
            finally:
                ab._weights_here = here
            check("other settings (weights)" in said, f"passages: a resume on other weights refused in plain words ({said})")
            lines: List[str] = []
            ab.build_passages(rt, d_, wins, "check", k=K, sample=S, extremes=ext, seed=7, chunk=3, log=lines.append)
            check(bool(lines) and "part 2 of" in lines[0] and not any("part 1 of" in x for x in lines), f"passages: the resumed run begins at its second part ({lines[:1]})")
        else:
            ab.build_passages(rt, d_, wins, "check", k=K, sample=S, extremes=ext, seed=7, chunk=chunk, log=quiet)
            check(not (d_ / "passages.d" / "parts").exists(), "passages: the parts are removed once merged")
        Q = AtlasIndex(d_).get("passages")
        check(Q is not None and sorted(k_ for k_ in Q if k_ != "meta") == names and all(np.array_equal(P[k_], Q[k_]) for k_ in names),
              f"passages: parts of {chunk} window{'s' if chunk > 1 else ''}{' (stopped and resumed)' if chunk == 3 else ''} merge to the same arrays as parts of 4")
    # a folder with its meta.json but without one of its arrays (copied or removed in part) is refused, not read in part
    (out / "passages-7" / "passages.d" / "unigram.npy").unlink()
    (out / "passages-1" / "passages.d" / "neg_smp_g.npy").unlink()
    cut7, cut1 = AtlasIndex(out / "passages-7"), AtlasIndex(out / "passages-1")
    check(cut7.get("passages") is None and "lacks unigram" in (cut7.refused("passages") or "") and cut1.get("passages") is None
          and "lacks neg_smp_g" in (cut1.refused("passages") or "") and "atlas_build.py passages" in (cut1.refused("passages") or ""),
          f"passages: a folder missing an array refused in plain words ({cut7.refused('passages')}; {cut1.refused('passages')})")
    # the sampled events: above half the face's extreme, at most one a window, as many as such windows allow
    bad_s = []
    for f, sgn, e_ in (("pos", 1.0, s["pos"]), ("neg", -1.0, s["neg"])):
        for l in range(L):
            for n in range(N):
                thr = 0.5 * sgn * float(e_[l, n])
                slots_ = [(int(P[f"{f}_smp_win"][l, n, i]), int(P[f"{f}_smp_at"][l, n, i])) for i in range(S) if int(P[f"{f}_smp_win"][l, n, i]) >= 0]
                able = sum(1 for A, _, _ in reads if thr > 0 and bool((sgn * A[l, 1:, n] > thr).any()))
                ok = (len({wi for wi, _ in slots_}) == len(slots_) == min(S, able)
                      and all(sgn * reads[wi][0][l, t, n] > thr and P[f"{f}_smp_act"][l, n, i] == h16(reads[wi][0][l, t, n])
                              for i, (wi, t) in enumerate(slots_)))
                if not ok:
                    bad_s.append((f, l, n, slots_, able))
    check(not bad_s, f"passages: sampled events above half the extreme, at most one a window, as many as such windows allow ({bad_s[:2]})")
    read_ids = np.concatenate([np.asarray(wn[1:]) for wn in wins])
    check(int(P["unigram"].sum()) == ex.tokens == len(read_ids) and np.array_equal(P["unigram"], np.bincount(read_ids, minlength=len(P["unigram"]))),
          "passages: the unigram counts add up to the word-pieces read, each word-piece's own")
    check(all(list(np.asarray(P["windows"][i])[: len(wn)]) == list(wn) and bool((np.asarray(P["windows"][i])[len(wn):] == -1).all())
              for i, wn in enumerate(wins)), "passages: the windows kept as read, -1 past a short one's end")
    section("passages: each face's K strongest windows by brute force, read again, merged alike whatever the parts, sampled, counted", before)

    before = len(failures)
    f = ao.link_scale(w.scalars, [float(x) for x in s["mlp_rms_q"][:, 1]], [float(x) for x in s["mid_rms_q"][:, 1]])
    ab.build_wiring(rt, out, f, scale_source="check", k=10, chunk=40, log=quiet)
    for (l, n) in ((0, 3), (2, 60), (L - 1, 9), (3, N - 1)):
        live = ao.neuron_wiring(w, l, n, f, k=10)
        got = idx.links(l, n, 10)
        for side in ("down", "up"):
            a = [(c["layer"], c["n"], c["sign"], c["gate"]) for c in live[side]]
            b = [(c["layer"], c["n"], c["sign"], c["gate"]) for c in got[side]]
            check(a == b, f"wiring: L{l} N{n} {side}, the same links in the same order with sign and gate: {sorted(set(a) ^ set(b))[:4]}")
            zl = {(c["layer"], c["n"]): (c["z"], c["w"]) for c in live[side]}
            check(all(abs(c["z"] - zl.get((c["layer"], c["n"]), (1e9, 0))[0]) < 0.05 for c in got[side]), f"wiring: L{l} N{n} {side} z")
            check(all(abs(c["w"] - zl.get((c["layer"], c["n"]), (0, 1e9))[1]) <= 2e-2 * c["w"] + 1e-30 for c in got[side]),
                  f"wiring: L{l} N{n} {side} w (log10 in float16)")
    check(idx.links(0, 3, 10)["up"] == [] and idx.links(L - 1, 9, 10)["down"] == [], "wiring: the first and last layers' empty sides")
    check(idx.get("wiring")["meta"].get("scale_source") == "check" and "from rest" in idx.get("wiring")["meta"].get("link_model", "")
          and "in use" in idx.get("wiring")["meta"].get("link_model_use", ""), "wiring: the scales' source and both link models recorded")
    section("the wiring index is neuron_wiring's, link by link, in order of strength across layers", before)

    before = len(failures)
    # each neuron's operating point on each face by brute force: the median over its stored events, one at a time
    P = idx.get("passages")
    c_ = (2.0 / np.pi) ** 0.5
    ops_hand: dict = {}

    def op_hand(l_: int, n_: int) -> dict:
        if (l_, n_) not in ops_hand:
            ops_hand[(l_, n_)] = {}
            for face, f_ in (("+", "pos"), ("-", "neg")):
                ok = np.asarray(P[f"{f_}_win"][l_, n_]) >= 0
                g_, u_ = np.asarray(P[f"{f_}_g"][l_, n_], np.float64)[ok], np.asarray(P[f"{f_}_u"][l_, n_], np.float64)[ok]
                ops_hand[(l_, n_)][face] = (float(np.median(np.array([ao.gelu_tanh_grad(x) for x in g_]) * u_)),
                                            float(np.median(0.5 * g_ * (1 + np.tanh(c_ * (g_ + 0.044715 * g_**3)))))) if ok.any() else None
        return ops_hand[(l_, n_)]

    def coupling_hand(reader: tuple, face: str, c: dict):
        """alpha g + beta u at the reader's brute-force operating point on `face`, written out; None without one."""
        op_ = op_hand(*reader)[face]
        return None if op_ is None else op_[0] * c["g"] + op_[1] * c["u"]

    # each link's g and u, kept as float16, are the live ones
    gu_bad, n_gu, use_bad, n_use, hand_bad, n_hand = [], 0, [], 0, [], 0
    for l in range(L):
        for n in range(0, N, 9):
            live = ao.neuron_wiring(w, l, n, f, k=10, gu=True)
            got = idx.links(l, n, 10, gu=True)
            for side in ("down", "up"):
                lv = {(c["layer"], c["n"]): c for c in live[side]}
                for c in got[side]:
                    x = lv[(c["layer"], c["n"])]
                    n_gu += 1
                    if not all(abs(c[k_] - x[k_]) <= 1e-3 * abs(x[k_]) + 1e-7 for k_ in ("g", "u")):
                        gu_bad.append((l, n, side, c["g"], x["g"], c["u"], x["u"]))
                # the sign in use, from the index's g and u and from the live ones
                a_, b_ = idx.use_signs(got[side], side, l, n), idx.use_signs(live[side], side, l, n)
                n_use += len(a_)
                use_bad += [(l, n, side, i_) for i_, (p_, q_) in enumerate(zip(a_, b_)) if p_ != q_]
                # and by hand, at the reader's own operating point: each reader downstream, the open neuron itself upstream
                for c, us in zip(got[side], a_):
                    reader = (c["layer"], c["n"]) if side == "down" else (l, n)
                    for face in ("+", "-"):
                        cs = coupling_hand(reader, face, c)
                        n_hand += 1
                        if us[face] != (None if cs is None else c["gate"] * (1 if cs > 0 else -1)):
                            hand_bad.append((l, n, side, reader, face))
    check(n_gu > 500 and not gu_bad, f"wiring: each link's g and u as float16 keeps the live ones ({len(gu_bad)} of {n_gu} differ: {gu_bad[:2]})")
    check(n_use > 500 and not use_bad, f"in use: the same use_sign from the index's g and u as from the live ones ({len(use_bad)} of {n_use} differ)")
    check(n_hand > 1000 and not hand_bad, f"in use: each use_sign is gate x sign(alpha g + beta u) at its reader's own operating point, the open neuron's "
          f"upstream ({len(hand_bad)} of {n_hand} differ: {hand_bad[:2]})")
    op_bad, faces_none = [], 0
    for l in range(L):
        for n in range(0, N, 5):
            op = idx.op_point(l, n)
            for face in ("+", "-"):
                want = op_hand(l, n)[face]
                if want is None:
                    faces_none += 1
                    op_bad += [(l, n, face)] if op[face] is not None else []
                    continue
                if op[face] is None or any(abs(op[face][i_] - want[i_]) > 1e-5 * max(1.0, abs(want[i_])) for i_ in (0, 1)):
                    op_bad.append((l, n, face, op[face], want))
    check(not op_bad, f"in use: each face's operating point is the median of gelu'(g) u and gelu(g) over its stored events ({op_bad[:2]})")
    # the count of signs in use that differ from rest, over the whole index, is the count link by link
    dis = idx.sign_disagreement()
    by_hand = {"+": [0, 0], "-": [0, 0]}
    links_all = 0
    for l in range(L):
        for n in range(N):
            got = idx.links(l, n, 10, gu=True)["down"]
            links_all += len(got)
            for c, us in zip(got, idx.use_signs(got, "down", l, n)):
                for face in ("+", "-"):
                    if us[face] is not None:
                        by_hand[face][0] += 1
                        by_hand[face][1] += int(us[face] != c["sign"])
    check(dis is not None and dis["links"] == links_all and all(dis["faces"][s_] == {"read": by_hand[s_][0], "differ": by_hand[s_][1]} for s_ in ("+", "-"))
          and sum(r_["+"]["differ"] for r_ in dis["by_layer"]) == by_hand["+"][1],
          f"in use: the count of signs that differ from rest is the count link by link ({dis and dis['faces']} vs {by_hand})")
    print(f"        on the stand-in the sign in use differs from rest on {by_hand['+'][1]} of {by_hand['+'][0]} links read on the reader's + face "
          f"and {by_hand['-'][1]} of {by_hand['-'][0]} on its - face")
    # a wiring index built before it kept g and u still serves the sign from rest, and says, in use, to rebuild it
    (out / "wiring.npz").rename(out / "wiring.keep")
    oldw = dict(np.load(out / "wiring.keep"))
    for side in ("down", "up"):
        del oldw[f"{side}_g"], oldw[f"{side}_u"]
    np.savez(out / "wiring.npz", **oldw)
    old_links = idx.links(2, 60, 10, gu=True)
    said = idx.use_refused() or ""
    check(old_links is not None and "g" not in old_links["down"][0] and idx.use_signs(old_links["down"], "down", 2, 60)[0] is None
          and "rebuild it" in said and idx.use_refused(live=True) is None and idx.sign_disagreement() is None,
          f"in use: a wiring index without g and u serves the sign from rest, and says to rebuild it ({said})")
    (out / "wiring.npz").unlink()
    (out / "wiring.keep").rename(out / "wiring.npz")
    section("the wiring's g and u kept; operating points as medians; the sign in use alike from the index and live, counted", before)

    before = len(failures)
    lines: List[str] = []
    ab.build_wiring_check(rt, out, f, layers=[1, 3], writers="random:12", dose=0.25, seed=3, batch=7, scale_source="check", log=lines.append)
    wc = idx.get("wiring_check")
    m = wc["meta"] if wc else {}
    check(wc is not None and sorted(wc) == ["layers", "meta"] and wc["layers"].tolist() == [1, 3] and m.get("dose") == 0.25
          and m.get("writers") == "random:12" and m.get("dose_unit") == "each writer's write ceiling" and m.get("scale_source") == "check"
          and m.get("wiring_built_at") == idx.get("wiring")["meta"]["built_at"] and m.get("passages_built_at") == P["meta"]["built_at"]
          and m.get("stats_built_at") == s["meta"]["built_at"] and m.get("format") == ab.WIRING_CHECK_FORMAT == 2 and "g0, u0" in m.get("sits", "")
          and set(m.get("summary", {})) == {"1", "3"} and m.get("sign_disagreement") == {k_: v_ for k_, v_ in idx.sign_disagreement().items() if k_ != "by_layer"},
          f"wiring_check: written with its provenance and format, the server reading only its meta ({sorted(wc or {})}, {sorted(m)[:6]})")
    check(idx.status()["wiring_check"]["built"] and "summary" in idx.status()["wiring_check"], "wiring_check: /arch's status carries its summary")
    z = dict(np.load(out / "wiring_check.npz"))
    wi = idx.get("wiring")
    from backend.census import parse_neurons

    ao_writers = lambda l_, n_: [A for A in parse_neurons("random:12", n_, 3 + l_) if float(w.write_ceiling(l_, [A])[0]) > 0]  # noqa: E731
    bad_r = []
    from backend.census import PROMPTS

    qs = [rt.tokenize_prompt(rt.compose_prompt(v_, chat=True)) for v_ in PROMPTS.values()]
    P_ = max(len(q_) for q_ in qs) - 1
    for l in (1, 3):
        ws_ = z[f"L{l}_writers"].tolist()
        check(ws_ == ao_writers(l, N) and np.allclose(z[f"L{l}_lengths"], 0.25 * w.write_ceiling(l, ws_).numpy(), rtol=1e-6),
              f"wiring_check L{l}: the writers drawn, each pushed by 0.25 of its own ceiling")
        W_, R_ = z[f"L{l}_readers"].shape[:2]
        check(z[f"L{l}_g0"].shape == z[f"L{l}_u0"].shape == (W_, R_, len(qs), P_) and z[f"L{l}_f"].shape == (W_, R_),
              f"wiring_check L{l}: each reader's g0 and u0 [W, R, Q, P] and each pair's f kept ({z[f'L{l}_g0'].shape})")
        for a_, A in enumerate(ws_):
            lk = [(int(wi["down_layer"][l, A, i_]), int(wi["down_n"][l, A, i_])) for i_ in range(int(wi["K"])) if float(wi["down_z"][l, A, i_]) > -1e4]
            rd = [tuple(int(x) for x in z[f"L{l}_readers"][a_, r_]) for r_ in range(z[f"L{l}_readers"].shape[1]) if z[f"L{l}_readers"][a_, r_, 0] >= 0]
            isl = z[f"L{l}_linked"][a_, : len(rd)].tolist()
            nl = [t for t, x in zip(rd, isl) if not x]
            ok = (rd[: len(lk)] == lk and all(isl[: len(lk)]) and not any(isl[len(lk):]) and len(nl) == len(lk) and not set(nl) & set(lk)
                  and len(set(nl)) == len(nl) and sorted(j for j, _ in nl) == sorted(j for j, _ in lk))
            for r_, (j, b) in enumerate(rd):
                g_, u_ = float(z[f"L{l}_g"][a_, r_]), float(z[f"L{l}_u"][a_, r_])
                ok &= abs(z[f"L{l}_strength"][a_, r_] - abs(g_ * u_) * float(f[l, j]) ** 2) <= 1e-5 * abs(z[f"L{l}_strength"][a_, r_]) + 1e-30
                ok &= float(z[f"L{l}_f"][a_, r_]) == float(np.float32(f[l, j]))
                if r_ < len(lk):
                    ok &= (int(z[f"L{l}_sign"][a_, r_]), int(z[f"L{l}_gate"][a_, r_])) == (int(wi["down_sign"][l, A, r_]), int(wi["down_gate"][l, A, r_]))
                    ok &= abs(g_ - float(wi["down_g"][l, A, r_])) <= 1e-3 * abs(g_) + 1e-7
                else:
                    ok &= (int(z[f"L{l}_sign"][a_, r_]), int(z[f"L{l}_gate"][a_, r_])) == ((1 if g_ * u_ > 0 else -1), (1 if g_ > 0 else -1))
                op_ = idx.op_point(j, b)
                ok &= all((op_[s_] is None and np.isnan(z[f"L{l}_op"][a_, r_, fi]).all()) or tuple(float(x) for x in z[f"L{l}_op"][a_, r_, fi]) == tuple(np.float32(op_[s_]).tolist())
                          for fi, s_ in enumerate(("+", "-")))
            if not ok:
                bad_r.append((l, A))
    check(not bad_r, f"wiring_check: the linked readers are the index's links, the null's at their layers and never linked; signs, g, u, f, strength, operating points ({bad_r[:3]})")
    # the reads and where the readers sit: pushed_reads again, in another batch shape, for the first three writers of layer 1
    rd1 = [[tuple(int(x) for x in z["L1_readers"][a_, r_]) for r_ in range(z["L1_readers"].shape[1]) if z["L1_readers"][a_, r_, 0] >= 0] for a_ in range(3)]
    again, again_g0, again_u0 = ao.pushed_reads(lm, w, qs, [(1, int(A)) for A in z["L1_writers"][:3]], rd1, z["L1_lengths"][:3], bos_id=tok.bos_token_id, batch=3)
    same_ = lambda x, y: np.allclose(np.nan_to_num(x, nan=7.0), np.nan_to_num(y, nan=7.0), atol=1e-5)  # noqa: E731
    check(same_(z["L1_reads"][:3, : again.shape[1]], again) and same_(z["L1_g0"][:3, : again.shape[1]], again_g0)
          and same_(z["L1_u0"][:3, : again.shape[1]], again_u0), "wiring_check: the reads, g0 and u0 are pushed_reads' in another batch shape")
    # the summary is the stored arrays', every figure recounted by hand, pair by pair, for the linked readers and the null
    sum_bad, n_peak_hand = [], 0
    keys_last = ("pairs", "zero", "sign_plus", "sign_gate", "same_way", "use_plus", "use_gate", "use_plus_peak", "use_gate_peak", "regimes")
    shards = {l: {k_[len(f"L{l}_"):]: v_ for k_, v_ in z.items() if k_.startswith(f"L{l}_")} for l in (1, 3)}
    for l, sh_ in shards.items():
        s_l, hand = ab.wiring_check_summary(sh_), wiring_summary_by_hand(sh_)
        sum_bad += [(l, "meta")] if s_l != m["summary"][str(l)] else []
        sum_bad += [(l, "site_score")] if "site_score" in s_l else []
        for site in ("last", "mean"):
            for name in ("linked", "null"):
                a_, b_ = s_l[site][name], hand[site][name]
                n_peak_hand += b_.get("n_peak", 0) if name == "linked" else 0
                for key in keys_last if site == "last" else keys_last[:7]:
                    sum_bad += [(l, site, name, key, a_[key], b_[key])] if a_[key] != b_[key] else []
                sum_bad += [(l, site, name, "extra keys")] if site == "mean" and {"regimes", "use_gate_peak"} & set(a_) else []
                for key in ("rho", "rho_site"):
                    if (a_[key] is None) != (b_[key] is None) or (a_[key] is not None and abs(a_[key] - b_[key]) > 5e-4 + 1e-9):
                        sum_bad.append((l, site, name, key, a_[key], b_[key]))
                if abs(a_["moved"] - b_["moved"]) > 1e-6 * abs(b_["moved"]) + 1e-12:
                    sum_bad.append((l, site, name, "moved", a_["moved"], b_["moved"]))
            if s_l[site]["moved_over_null"] != hand[site]["moved_over_null"]:
                sum_bad.append((l, site, "moved_over_null", s_l[site]["moved_over_null"], hand[site]["moved_over_null"]))
    # the stand-in's readers never sit below gelu's minimum: with every g0 moved down by 1 (the arrays alone, as a model
    # whose readers sit shut would give them) the summary is still the recount, every regime counted
    low = {**shards[1], "g0": shards[1]["g0"] - np.float32(1.0)}
    s_low, hand_low = ab.wiring_check_summary(low), wiring_summary_by_hand(low)
    for name in ("linked", "null"):
        for key in keys_last:
            sum_bad += [("low", name, key, s_low["last"][name][key], hand_low["last"][name][key])] if s_low["last"][name][key] != hand_low["last"][name][key] else []
    check((s_low["last"]["linked"]["regimes"]["below_min"]["share"] or 0) > 0.2, f"wiring_check: g0 moved down puts readers below gelu's minimum "
          f"({s_low['last']['linked']['regimes']})")
    regs = m["summary"]["1"]["last"]["linked"]["regimes"]
    check(not sum_bad and n_peak_hand > 0 and abs(sum(regs[k_]["share"] for k_ in regs) - 1.0) < 2e-3,
          f"wiring_check: the meta's summary is the stored arrays', every share (zeros left out), the sign in use where each reader sits and "
          f"at the passages', each regime, both rho and the size recounted by hand ({n_peak_hand} linked pairs read at the passages'; {sum_bad[:3]})")
    lst = m["summary"]["1"]["last"]
    print(f"        on the stand-in, layer 1 at the last word-piece, linked / null: sign_gate {lst['linked']['sign_gate']} / {lst['null']['sign_gate']}, "
          f"use_gate where the reader sits {lst['linked']['use_gate']} / {lst['null']['use_gate']}, at the passages' {lst['linked']['use_gate_peak']} / "
          f"{lst['null']['use_gate_peak']}; gates open {regs['open']['share']}, shut {regs['shut_above_min']['share']}, below gelu's minimum "
          f"{regs['below_min']['share']}; rho_site {lst['linked']['rho_site']}, rho {lst['linked']['rho']}")
    # the sign in use where the reader sits against a float32 central difference of the GeGLU itself, gelu(g0 + e g) (u0 + e u)
    # over +-e (at the last word-piece, and its mean over the prompt's word-pieces), at each (writer, reader, question)
    # where the difference stands well above float32's rounding
    fd_n, fd_bad, fd_all = 0, [], 0
    for l, sh_ in shards.items():
        al_, be_ = ao.site_coefficients(sh_["g0"], sh_["u0"])
        for a_, r_ in zip(*np.nonzero(sh_["readers"][..., 0] >= 0)):
            g_, u_ = np.float32(sh_["g"][a_, r_]), np.float32(sh_["u"][a_, r_])
            for q in range(sh_["g0"].shape[2]):
                for site, cut in (("last", slice(-1, None)), ("mean", slice(None))):
                    g0_, u0_ = sh_["g0"][a_, r_, q, cut], sh_["u0"][a_, r_, q, cut]
                    here_ = np.isfinite(g0_)
                    g0_, u0_ = g0_[here_], u0_[here_]
                    e = np.float32(1e-3 * max(1.0, float(np.abs(g0_).max()), float(np.abs(u0_).max())) / max(abs(float(g_)), abs(float(u_)), 1e-30))
                    F_ = [np.mean(gelu32(g0_ + s_ * e * g_) * (u0_ + s_ * e * u_), dtype=np.float32) for s_ in (np.float32(1), np.float32(-1))]
                    fd = float((F_[0] - F_[1]) / (np.float32(2) * e))
                    al0 = float(al_[a_, r_, q, -1]) if site == "last" else float(np.mean(al_[a_, r_, q][np.isfinite(al_[a_, r_, q])]))
                    be0 = float(be_[a_, r_, q, -1]) if site == "last" else float(np.mean(be_[a_, r_, q][np.isfinite(be_[a_, r_, q])]))
                    coupling = al0 * float(g_) + be0 * float(u_)
                    fd_all += 1
                    if abs(coupling) * 2 * float(e) <= 1e-4 * max(abs(float(F_[0])), abs(float(F_[1])), 1e-30):
                        continue
                    fd_n += 1
                    if (fd > 0) != (coupling > 0) or abs(fd - coupling) > 1e-2 * abs(coupling):
                        fd_bad.append((l, int(a_), int(r_), q, site, fd, coupling))
    check(fd_n >= 0.8 * fd_all and not fd_bad, f"wiring_check: the sign in use where the reader sits is a float32 central difference's of "
          f"gelu(g0 + e g) (u0 + e u), at {fd_n} of {fd_all} (writer, reader, question, site) ({len(fd_bad)} off: {fd_bad[:2]})")
    # a layer measured before g0 and u0 were recorded is summed up as it was then, and says the site score needs a rerun
    old_bad = []
    for l, sh_ in shards.items():
        before_ = {k_: v_ for k_, v_ in sh_.items() if k_ not in ("g0", "u0", "f")}
        s_o, hand = ab.wiring_check_summary(before_), wiring_summary_before_by_hand(before_)
        old_bad += [(l, "note")] if s_o.get("site_score") != SITE_SCORE_RERUN or set(s_o) != {"last", "mean", "site_score"} else []
        for site in ("last", "mean"):
            for name in ("linked", "null"):
                a_, b_ = s_o[site][name], hand[site][name]
                want_keys = {"pairs", "sign_plus", "sign_gate", "same_way", "rho", "moved"} | ({"use_plus", "use_gate"} if site == "last" else set())
                old_bad += [(l, site, name, "keys", sorted(a_))] if set(a_) != want_keys else []
                for key in ("pairs", "sign_plus", "sign_gate", "same_way") + (("use_plus", "use_gate") if site == "last" else ()):
                    old_bad += [(l, site, name, key, a_[key], b_[key])] if a_[key] != b_[key] else []
                if (a_["rho"] is None) != (b_["rho"] is None) or (a_["rho"] is not None and abs(a_["rho"] - b_["rho"]) > 5e-4 + 1e-9):
                    old_bad.append((l, site, name, "rho", a_["rho"], b_["rho"]))
            old_bad += [(l, site, "moved_over_null")] if s_o[site]["moved_over_null"] != hand[site]["moved_over_null"] else []
    check(not old_bad, f"wiring_check: a layer measured before g0 and u0 were recorded summed up as before (zeros as misses, the "
          f"passages' sign in use as use_gate), saying the site score needs a rerun ({old_bad[:3]})")
    cst = m["cost"]["1"]
    per_ = (7 - 1) // 2
    check(cst["forwards"] == len(qs) * -(-12 // per_) and cst["rows"] == 7 and cst["minutes_5090"] > 0 and any("layer 1: 12 writers" in x and "min on the 5090" in x for x in lines),
          f"wiring_check: the cost line, prefill only ({cst})")
    check(abs(ab.spearman(np.array([1, 2, 3, 4.0]), np.array([10, 20, 30, 40.0])) - 1.0) < 1e-12
          and abs(ab.spearman(np.array([1, 2, 3.0]), np.array([3, 2, 1.0])) + 1.0) < 1e-12
          and abs(ab.spearman(np.array([1, 1, 2, 3.0]), np.array([1, 2, 3, 4.0])) - 0.9486832980505138) < 1e-12
          and ab.spearman(np.array([1, 1, 1.0]), np.array([1, 2, 3.0])) is None, "spearman: ranks with ties at their mean, None for a constant side")
    # a second run adds a layer and skips those done, past a shard a stopped run left half-written; one with another
    # dose, against another passages index or with other scales is refused, and so is a batch that holds no writer
    (out / "wiring_check.d" / "L2.npz.tmp.npz").write_bytes(b"half a shard")
    lines2: List[str] = []
    said = raises(ab.build_wiring_check, rt, out, f, layers=[1, 4], writers="random:12", dose=0.25, seed=3, batch=7, scale_source="check", log=lines2.append)
    check(not said and idx.get("wiring_check")["layers"].tolist() == [1, 3, 4] and not any("layer 1:" in x for x in lines2) and any("layer 4:" in x for x in lines2),
          f"wiring_check: a second run adds its layers and skips those done, a half-written shard not taken for a layer ({said})")
    said = raises(ab.build_wiring_check, rt, out, f, layers=[2], writers="random:12", dose=0.5, seed=3, batch=7, scale_source="check", log=quiet)
    check("other settings or against other indexes (dose)" in said, f"wiring_check: a run with another dose refused in plain words ({said})")
    said = raises(ab.build_wiring_check, rt, out, f, layers=[2], writers="random:12", dose=0.25, seed=3, batch=7, scale_source="other", log=quiet)
    check("(scale_source)" in said, f"wiring_check: a run with scales from elsewhere refused ({said})")
    (out / "passages.d" / "meta.json").rename(out / "passages.meta.off")
    said = raises(ab.build_wiring_check, rt, out, f, layers=[2], writers="random:12", dose=0.25, seed=3, batch=7, scale_source="check", log=quiet)
    (out / "passages.meta.off").rename(out / "passages.d" / "meta.json")
    check("(passages_built_at)" in said and not (out / "wiring_check.d" / "L2.npz").exists(),
          f"wiring_check: a run against another passages index (here none) refused, so no layer is read at other operating points ({said})")
    said = raises(ab.build_wiring_check, rt, out, f, layers=[2], writers="random:12", dose=0.25, seed=3, batch=2, scale_source="check", log=quiet)
    check("--batch 3 at least" in said, f"wiring_check: a batch too small for a writer's two pushes refused ({said})")
    # never two formats in one folder: a folder measured before g0 and u0 were recorded (its plan without a format) is
    # refused in plain words and left as it is; asked to start afresh, it is set aside, renamed, and the run measures
    # into a new one; a layer of the older format found in a folder of this one is refused before anything is measured
    plan_p = out / "wiring_check.d" / "plan.json"
    plan_p.write_text(json.dumps({k_: v_ for k_, v_ in json.loads(plan_p.read_text()).items() if k_ != "format"}), encoding="utf-8")
    kept = sorted(p_.name for p_ in (out / "wiring_check.d").iterdir())
    for dose_ in (0.25, 0.5):
        said = raises(ab.build_wiring_check, rt, out, f, layers=[2], writers="random:12", dose=dose_, seed=3, batch=7, scale_source="check", log=quiet)
        check("measured before the check recorded each reader's own gate and up values" in said and "--fresh" in said
              and sorted(p_.name for p_ in (out / "wiring_check.d").iterdir()) == kept,
              f"wiring_check: a run against a folder measured before g0 and u0 were recorded refused in plain words, the folder left as it is "
              f"(dose {dose_}: {said[:90]})")
    lines3: List[str] = []
    said = raises(ab.build_wiring_check, rt, out, f, layers=[2], writers="random:12", dose=0.25, seed=3, batch=7, scale_source="check",
                  fresh=True, log=lines3.append)
    aside = sorted(p_ for p_ in out.iterdir() if ".set-aside-" in p_.name)
    old_dir = [p_ for p_ in aside if p_.name.startswith("wiring_check.d.set-aside-")]
    old_npz = [p_ for p_ in aside if p_.name.startswith("wiring_check.set-aside-") and p_.suffix == ".npz"]
    check(not said and len(old_dir) == 1 and len(old_npz) == 1 and sorted(p_.name for p_ in old_dir[0].iterdir()) == kept
          and np.load(old_npz[0])["layers"].tolist() == [1, 3, 4] and idx.get("wiring_check")["layers"].tolist() == [2]
          and json.loads(plan_p.read_text()).get("format") == 2 and sum("set aside as" in x for x in lines3) == 2,
          f"wiring_check --fresh: the folder and wiring_check.npz set aside, renamed, not removed, and a new run started ({said}; {[p_.name for p_ in aside]})")
    stale = dict(np.load(out / "wiring_check.d" / "L2.npz"))
    for k_ in ("g0", "u0", "f"):
        del stale[k_]
    np.savez(out / "wiring_check.d" / "L4.npz", **stale)
    said = raises(ab.build_wiring_check, rt, out, f, layers=[2], writers="random:12", dose=0.25, seed=3, batch=7, scale_source="check", log=quiet)
    check("Layer 4 in" in said and "was measured before the check recorded" in said and idx.get("wiring_check")["layers"].tolist() == [2],
          f"wiring_check: a layer of the older format in a folder of this one refused, not merged ({said[:120]})")
    # refused before anything is measured: asked for a layer not yet done, beside this run's plan and with the plan lost
    # (then no plan is written either); the merge refuses it too, the backstop
    plan_text = plan_p.read_text(encoding="utf-8")
    said = raises(ab.build_wiring_check, rt, out, f, layers=[3], writers="random:12", dose=0.25, seed=3, batch=7, scale_source="check", log=quiet)
    plan_p.unlink()
    said_lost = raises(ab.build_wiring_check, rt, out, f, layers=[3], writers="random:12", dose=0.25, seed=3, batch=7, scale_source="check", log=quiet)
    said_merge = raises(ab.merge_wiring_check, rt, out, json.loads(plan_text), idx, log=quiet)
    check("Layer 4 in" in said and "Layer 4 in" in said_lost and "Layer 4 in" in said_merge and "nothing is measured" in said
          and not plan_p.exists() and not (out / "wiring_check.d" / "L3.npz").exists() and idx.get("wiring_check")["layers"].tolist() == [2],
          f"wiring_check: a layer of the older format refused before anything is measured, its plan lost or not, no plan written; "
          f"the merge refuses it too ({said_lost[:90]})")
    plan_p.write_text(plan_text, encoding="utf-8")
    (out / "wiring_check.d" / "L4.npz").unlink()
    # an older merged wiring_check.npz with no folder beside it (removed once merged, say) is refused, not replaced, and
    # nothing is measured or begun; --fresh would set it aside
    (out / "wiring_check.d").rename(out / "wiring_check.d.off")
    merged_p = out / "wiring_check.npz"
    merged_bytes = merged_p.read_bytes()
    zz = dict(np.load(merged_p))
    mm = json.loads(str(zz["meta"]))
    mm.pop("format")
    zz["meta"] = np.array(json.dumps(mm))
    np.savez(merged_p, **zz)
    older_bytes = merged_p.read_bytes()
    said = raises(ab.build_wiring_check, rt, out, f, layers=[3], writers="random:12", dose=0.25, seed=3, batch=7, scale_source="check", log=quiet)
    check("was measured before the check recorded" in said and "would replace it" in said and "--fresh" in said
          and merged_p.read_bytes() == older_bytes and not (out / "wiring_check.d").exists(),
          f"wiring_check: an older merged file with no folder beside it refused, not replaced, nothing begun ({said[:110]})")
    merged_p.write_bytes(merged_bytes)
    (out / "wiring_check.d.off").rename(out / "wiring_check.d")
    section("the wiring measured: written, read back, its readers, null, reads, g0 and u0, summary (the sign in use where each reader sits, "
            "regimes, both rho) and cost; an older layer summed up as before; resumed; refused; never two formats", before)

    before = len(failures)
    p50 = [float(x) for x in s["resid_q"][:, 1]]
    ab.build_steer_lens(rt, out, p50, layers=[1], batch=7, log=quiet)
    check(idx.get("steer_lens") is None and (out / "steer_lens.d" / "L1.npz").exists(), "steer: a partial run merged")
    ab.build_steer_lens(rt, out, p50, batch=9, log=quiet)
    st = idx.get("steer_lens")
    check(st is not None and st["up"].shape == (L, N, 12), "steer: merged shape")
    for (l, n) in ((1, 0), (1, 66), (4, 127), (0, 30)):
        live = ao.steer_lens(lm, W_U, ctx, w, l, n, p50[l], k=12, bos_id=tok.bos_token_id)
        check(st["up"][l, n].tolist() == live["up"] and st["down"][l, n].tolist() == live["down"], f"steer: L{l} N{n} rises")
        check(abs(float(st["up_kl"][l, n]) - live["kl_up"]) < 2e-3 * max(1, live["kl_up"]), f"steer: L{l} N{n} KL")
    section("the steer lens index is steer_lens's, neuron by neuron (a partial run resumes, then merges)", before)

    before = len(failures)
    ab.build_layout(rt, out, method="pca", log=quiet)
    lay = idx.layout()
    side = lay["side"]
    check(side * side >= N, "layout: the sheet is too small")
    check(all(len(set(lay["cell"][l].tolist())) == N and lay["cell"][l].max() < side * side for l in range(L)), "layout: cells")
    section("each layer's neurons take distinct places on its sheet", before)

    before = len(failures)
    vocab = [tok.decode([i]) for i in range(len(tok))]
    target = idx.face_ids(3, 20, "-", "eye", 12)
    word = next((vocab[t] for t in target[2:] if len(vocab[t].strip()) >= 3 and "�" not in vocab[t]), None)
    if word:
        rows, total = idx.search(match_ids(vocab, word.strip()), limit=100000)
        hit = [r for r in rows if (r["layer"], r["n"], r["sign"]) == (3, 20, "-")]
        check(bool(hit) and total >= len(rows), f"search: {word!r} did not find L3 N20 -")
    cone = idx.cone(2, 60, k=2, zmin=-99, first=4)
    check(all(r[5] in (1, -1) for r in cone if (r[0], r[1]) == (2, 60) or (r[2], r[3]) == (2, 60)), "cone: the first hop leaves the neuron")
    check(all(r[2] > r[0] for r in cone if r[5] > 0) and all(r[2] > r[0] for r in cone if r[5] < 0), "cone: links go up the layers")
    check(all((r[0], r[1]) == (2, 60) for r in cone if r[5] == 1) and all((r[2], r[3]) == (2, 60) for r in cone if r[5] == -1),
          "cone: the first hop is the neuron's own links")
    # the link model: downstream, a link is followed only when the face its writer was reached on opens the reader's gate
    ci = idx.get("wiring")
    gate_of = {(side, a_, b_, (int(ci[f"{side}_layer"][a_, b_, i_]), int(ci[f"{side}_n"][a_, b_, i_]))): (int(ci[f"{side}_gate"][a_, b_, i_]), int(ci[f"{side}_sign"][a_, b_, i_]))
               for side in ("down", "up") for a_ in range(L) for b_ in range(N) for i_ in range(int(ci["K"])) if float(ci[f"{side}_z"][a_, b_, i_]) > -1e4}
    for face in ("+", "-"):
        wk = idx.walk(2, 60, "down", face, first=4)
        hop1 = [t for t, r in wk.items() if r["hop"] == 1]
        check(all(gate_of[("down", 2, 60, t)][0] == (1 if face == "+" else -1) and wk[t]["face"] == ("+" if gate_of[("down", 2, 60, t)][1] > 0 else "-")
                  for t in hop1), f"walk down from the {face} face: only links whose gate that face opens, readers on the face pushed")
    up = idx.walk(4, 30, "up", None, first=6)
    check(all(r["drives"] == ("+" if gate_of[("up", 4, 30, t)][1] > 0 else "-") and r["face"] == ("+" if gate_of[("up", 4, 30, t)][0] > 0 else "-")
              for t, r in up.items() if r["hop"] == -1), "walk up: writers on the face that opens the gate, sorted by the face they drive")
    ws = idx.web_summary(2, 60, vocab, first=4)
    check(ws["src"] == "steer" and ws["ready"] and set(ws["down"]) == {"+", "-"} and set(ws["up"]) == {"+", "-"},
          "web summary: the steer lens is pooled when built, by face each way")
    (out / "steer_lens.npz").rename(out / "steer_lens.off")
    ws = idx.web_summary(2, 60, vocab, first=4)
    check(ws["src"] == "lens" and ws["gated"]["down"] + ws["gated"]["up"] > 0 and not ws["down"]["+"]["words"] and not ws["up"]["+"]["words"],
          f"web summary without the steer lens: {ws['gated']} {ws['down']['+']['words'][:2]}")
    (out / "steer_lens.off").rename(out / "steer_lens.npz")
    (out / "wiring.npz").rename(out / "wiring.off")
    check(idx.web_summary(2, 60, vocab)["missing"] == "wiring", "web summary: says the wiring index is what is missing")
    (out / "wiring.off").rename(out / "wiring.npz")
    g = np.random.default_rng(0)
    score = g.normal(size=(L, N)).astype(np.float32)
    rows, n_active = idx.active_paths(score, top=40, min_score=1.0)
    active = {(int(i // N), int(i % N)) for i in np.argsort(-np.abs(score).ravel())[:40] if abs(score.ravel()[i]) >= 1.0}
    check(n_active == len(active) and all((int(r[0]), int(r[1])) in active and (int(r[2]), int(r[3])) in active for r in rows),
          "active paths: a link to a neuron that did not fire")
    check(all(gate_of[("down", int(r[0]), int(r[1]), (int(r[2]), int(r[3])))][0] == (1 if score[int(r[0]), int(r[1])] > 0 else -1)
              and (int(r[4]) > 0) == (gate_of[("down", int(r[0]), int(r[1]), (int(r[2]), int(r[3])))][1] > 0) for r in rows),
          "active paths: only writers that fired on the face opening the gate, signed by the face pushed")
    # in use (LINK_MODEL_USE): where the writer, fired on a face, pushes its reader at the reader's operating points
    one = lambda s_: 1 if s_ == "+" else -1  # noqa: E731
    gl = idx.links(2, 60, 10, gu=True)
    for face in ("+", "-"):
        wk = idx.walk(2, 60, "down", face, first=4, model="use")
        want_w = {}
        for c in gl["down"][:4]:
            t = (c["layer"], c["n"])
            cand = [(s_, abs(cs)) for s_ in ("+", "-") for cs in [coupling_hand(t, s_, c)] if cs is not None and one(face) * (1 if cs > 0 else -1) == one(s_)]
            if cand:
                want_w[t] = max(cand, key=lambda x: x[1])[0]
        check({t: r["face"] for t, r in wk.items() if r["hop"] == 1} == want_w,
              f"walk down in use from the {face} face: each reader on the face pushed further (the harder of two), none pushed back toward rest only")
        up_u = idx.walk(4, 30, "up", face, first=6, model="use")
        want_u = {(c["layer"], c["n"]): ("+" if one(face) * (1 if cs > 0 else -1) > 0 else "-")
                  for c in idx.links(4, 30, 10, gu=True)["up"][:6] for cs in [coupling_hand((4, 30), face, c)] if cs is not None}
        check({t: r["face"] for t, r in up_u.items() if r["hop"] == -1} == want_u and all(r["drives"] == face for r in up_u.values()),
              f"walk up in use toward the {face} face: each writer on the face whose firing pushes the neuron that way")
    check("give face" in raises(idx.walk, 4, 30, "up", None, model="use"), "walk up in use without a face refused")
    rows_u, n_u = idx.active_paths(score, top=40, min_score=1.0, model="use")
    want_rows = []
    for (l_, n_) in active:
        for i_, c in enumerate(idx.links(l_, n_, int(ci["K"]), gu=True)["down"]):
            t = (c["layer"], c["n"])
            cs = coupling_hand(t, "+" if score[t] > 0 else "-", c) if t in active else None
            if cs is not None:
                z10 = int(round(float(ci["down_z"][l_, n_, i_]) * 10))
                want_rows.append((l_, n_, *t, z10 * (1 if score[l_, n_] > 0 else -1) * (1 if cs > 0 else -1)))
    check(n_u == n_active and sorted(tuple(int(x) for x in r[:5]) for r in rows_u) == sorted(want_rows) and len(rows_u) > 0,
          f"active paths in use: every link whose ends both fired, signed where the writer's firing pushes the reader ({len(rows_u)} vs {len(want_rows)})")
    wsu = idx.web_summary(2, 60, vocab, first=4, model="use")
    check(wsu["ready"] and wsu["model"] == "use" and set(wsu["down"]) == {"+", "-"}, "web summary in use: pooled by face each way")
    (out / "passages.d" / "meta.json").rename(out / "passages.meta.off")
    wsn = idx.web_summary(2, 60, vocab, model="use")
    check(not wsn["ready"] and wsn["missing"] == "passages" and "atlas_build.py passages" in wsn["why"] and idx.op_point(2, 60) == {"+": None, "-": None}
          and idx.active_paths(score, top=40, model="use")[0].shape == (0, 6) and idx.active_paths(score, top=40, model="use")[1] == n_active,
          f"in use without the passages index: says so, no paths, the active neurons still counted ({wsn})")
    (out / "passages.meta.off").rename(out / "passages.d" / "meta.json")
    acts = np.array([[[2.0, -1.0], [0.5, -3.0]]], np.float32)  # [1, T=2, N=2]
    sc = scale_scores(acts, np.array([[4.0, 0.0]]), np.array([[-1.0, -2.0]]))
    check(sc[0, 0].tolist() == [0.0, 0.0] and np.allclose(sc[0, 1], [0.125, -1.5]), f"scores: {sc.tolist()}")
    section("search, cone, web summary (steer lens, or only labels the gate lets through), active paths, scores", before)

    before = len(failures)
    ids = [rt.tokenize_prompt(t) for t in ao.HEAD_TEXTS]
    prof = ao.head_profiles(lm, w, tok, ids)
    heads = prof["heads"]
    check(len(heads) == sum(ao.attn_shape(lm, l).n_heads for l in range(L)), "profiles: every head")
    check(all(abs(sum(h["dist"]) - 1.0) < 1e-2 for h in heads.values()), "profiles: shares do not add up to one")
    check(all(qu - ku > 1 and ku > 0 for h in heads.values() for _, _, qu, ku in h["pairs"]), "profiles: a pair one unit back")
    section("head profiles: shares add up to one per read, pairs reach further back", before)

    before = len(failures)
    mine = AtlasIndex(out, model=rt.model_id, shape=(L, N))
    check(all(mine.get(k_) is not None for k_ in ("lens", "act_scale", "wiring", "steer_lens", "layout", "passages")), "the model's own indexes are read")
    other = AtlasIndex(out, model="google/gemma-4-31B-it", shape=(L, N))
    st = other.status()
    check(other.get("lens") is None and not st["lens"]["built"] and "built for" in st["lens"]["refused"], f"another model's index refused: {st['lens']}")
    wide = AtlasIndex(out, model=rt.model_id, shape=(L, N + 1))
    check(wide.get("wiring") is None and "shaped for" in wide.status()["wiring"]["refused"], "an index of another shape refused")
    check(other.face_ids(1, 2, "+", "lens") is None and other.cone(1, 2) == [] and other.links(1, 2) is None
          and other.passages(1, 2, "+") is None and "built for" in st["passages"]["refused"], "a refused index reads as not built (passages too)")
    oldw = dict(np.load(out / "wiring.npz"))
    for side in ("down", "up"):
        del oldw[f"{side}_lw"], oldw[f"{side}_gate"]
    np.savez(part / "wiring.npz", **oldw)
    check(pidx.get("wiring") is None and "weighed across layers" in pidx.status()["wiring"]["refused"],
          "a wiring index built before links were weighed across layers is refused")
    section("an index built for another model, of another shape, or by an older build, is not read", before)

    before = len(failures)
    card = rf.weights_card(rt)
    metas = {k_: mine.get(k_)["meta"] for k_ in ("lens", "act_scale", "wiring", "steer_lens", "layout", "passages")}
    check(card == {"format": "stand-in", "model": rt.model_id, "runtime": "TheodoliteRuntime"}
          and all(m_.get("weights") == card for m_ in metas.values()),
          f"every index's meta names the weights it was built from (the stand-in's card): {[k_ for k_, m_ in metas.items() if m_.get('weights') != card]}")
    theirs = AtlasIndex(out, model=rt.model_id, shape=(L, N), weights=OTHER_GGUF)
    st2 = theirs.status()
    want = "built from other weights (the stand-in, not the GGUF other.gguf (fingerprint 00ff00ff00ff00ff)): rebuild it"
    check(all(theirs.get(k_) is None and want in (st2[k_]["refused"] or "") for k_ in metas),
          f"the same model id, other weights: every index refused in plain words ({st2['lens'].get('refused')})")
    old_lens = dict(np.load(out / "lens.npz"))
    m_old = json.loads(str(old_lens["meta"]))
    del m_old["weights"]
    old_lens["meta"] = np.array(json.dumps(m_old))
    (out / "older").mkdir()
    np.savez(out / "older" / "lens.npz", **old_lens)
    check(AtlasIndex(out / "older", model=rt.model_id, shape=(L, N), weights=OTHER_GGUF).get("lens") is not None,
          "an index built before metas named their weights: read on the model id alone, as before")
    ct = lambda files, snap="a" * 40: {"format": "compressed-tensors W4A16", "repo": "google/gemma-4-31B-it-qat-w4a16-ct",  # noqa: E731
                                       "snapshot": snap, "files_sha256": files}
    check(weights_differ(ct("e" * 64), ct("f" * 64, "b" * 40)) == (
              "google/gemma-4-31B-it-qat-w4a16-ct (weight files eeeeeeeeeeeeeeee, snapshot aaaaaaaaaaaa), "
              "not google/gemma-4-31B-it-qat-w4a16-ct (weight files ffffffffffffffff, snapshot bbbbbbbbbbbb)")
          and weights_differ(ct("e" * 64), ct("e" * 64, "b" * 40)) is None and weights_differ(ct("e" * 64), ct(None)) is None
          and weights_differ(None, ct("e" * 64)) is None and weights_differ(OTHER_GGUF, {**OTHER_GGUF, "linears": "resident"}) is None
          and weights_differ(OTHER_GGUF, {**OTHER_GGUF, "id": "1" * 16}) is not None
          and weights_differ(OTHER_GGUF, {**OTHER_GGUF, "sha256_header": "cd" * 32}) is not None
          and weights_differ(OTHER_GGUF, ct("e" * 64)) is not None,
          "weights_differ: other weight files of the checkpoint or another fingerprint differ, another snapshot of the same "
          "files does not; unknown files or a card missing compares the model id alone; how a GGUF's layers are held does "
          "not count")
    here = ab._weights_here
    ab._weights_here = lambda _rt: OTHER_GGUF
    try:
        said = raises(ab.build_steer_lens, rt, out, p50, layers=[2], batch=9, log=quiet)
    finally:
        ab._weights_here = here
    check("L0.npz was built from other weights (the stand-in, not the GGUF other.gguf" in said and "another folder (--out)" in said,
          f"steer: a run on other weights refused before it builds a layer, not resumed into the stand-in's ({said})")
    # census_run.py's push unit: a stats index there but refused for its weights is said so, not taken for none
    import contextlib
    import io

    import census_run
    from backend import census as cz

    stats_theirs = dict(np.load(out / "act_scale.npz"))
    m_s = json.loads(str(stats_theirs["meta"]))
    m_s["weights"] = OTHER_GGUF
    stats_theirs["meta"] = np.array(json.dumps(m_s))
    (out / "theirs" / "stand-in" / "index").mkdir(parents=True)
    np.savez(out / "theirs" / "stand-in" / "index" / "act_scale.npz", **stats_theirs)
    seen: dict = {}

    def grab(*_a, **kw):  # the census itself is not taken: only what it would be given
        seen.update(kw)
        raise ValueError("stopped before the census")

    env_keys = ("THEODOLITE_ATLAS_DIR", "THEODOLITE_CENSUS_DIR", "THEODOLITE_FIELD_NOTES")
    env_before, take = {k_: os.environ.get(k_) for k_ in env_keys}, cz.take_census
    os.environ["THEODOLITE_ATLAS_DIR"] = str(out / "theirs")
    cz.take_census = grab
    printed = io.StringIO()
    try:
        with contextlib.redirect_stdout(printed):
            census_run.main(["--stand-in", "--layer", "2", "--neurons", "0-1", "--random-controls", "1", "--max-new", "4", "--no-rank"])
        said = "no exit"
    except SystemExit as exc:
        said = str(exc)
    finally:
        cz.take_census = take
        for k_, v_ in env_before.items():
            if v_ is None:
                os.environ.pop(k_, None)
            else:
                os.environ[k_] = v_
    want = "built from other weights (the GGUF other.gguf (fingerprint 00ff00ff00ff00ff), not the stand-in)"
    check(said == "stopped before the census" and f"the stats index is not read: {want}" in printed.getvalue()
          and seen.get("p50_source", "").startswith(f"the study's 36 texts (the stats index refused: {want}"),
          f"census_run.py: a stats index refused for its weights is said so, and the push unit's source names it "
          f"({seen.get('p50_source', '')[:110]})")

    # one folder holds one model's indexes: a build that would write over another model's is refused unless --replace
    import subprocess

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"}
    build = [sys.executable, "-I", str(ROOT_DIR / "scripts" / "atlas_build.py"), "layout", "--stand-in", "--layout-method", "pca",
             "--out", str(out / "replace")]
    r = subprocess.run(build, env=env, capture_output=True, text=True, timeout=600)
    lay = out / "replace" / "layout_writes.npz"
    if r.returncode == 0 and lay.exists():
        z = dict(np.load(lay))
        m_ = json.loads(str(z["meta"]))
        m_["model"] = "google/gemma-4-31B-it-qat-w4a16-ct"
        z["meta"] = np.array(json.dumps(m_))
        np.savez(lay, **z)
    before_bytes = lay.read_bytes() if lay.exists() else b""
    r = subprocess.run(build, env=env, capture_output=True, text=True, timeout=600)
    said = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""
    check(r.returncode != 0 and "was built for google/gemma-4-31B-it-qat-w4a16-ct" in said and "--replace" in said
          and lay.read_bytes() == before_bytes,
          f"atlas_build.py: an index of another model is not written over, in plain words ({said[:120]})")
    r = subprocess.run(build + ["--replace"], env=env, capture_output=True, text=True, timeout=600)
    now = json.loads(str(np.load(lay)["meta"])).get("model") if lay.exists() else None
    check(r.returncode == 0 and "replacing" in r.stdout and now != "google/gemma-4-31B-it-qat-w4a16-ct",
          f"atlas_build.py --replace: the other model's index replaced, and said ({now})")
    section("provenance: each index names its weights; one from other weights refused, in plain words, and not resumed", before)

    shutil.rmtree(out, ignore_errors=True)
    print(f"\n{passed} checks passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

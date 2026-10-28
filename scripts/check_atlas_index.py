#!/usr/bin/env python3
"""A CPU-only check of the atlas's indexes (backend/atlas_build.py builds them, backend/atlas_index.py reads them) on the
tiny random Gemma 4 of scripts/atlas_stand_in.py. Every builder runs into a temporary folder, and each file is compared
with the same quantity read live, one neuron at a time:
1. `forward_from_layer` from any layer, on right-padded rows, gives the whole forward's final states;
2. lens.npz: each face's first ten word-pieces through the lens and by the naked eye are the ids the vault's
   DESCRIBE POLE ranks for its label (`label_top_ids`), and the index's label check is the vault's, count and reasons;
3. act_scale.npz: the extremes and the stream's quantiles of `activation_extremes` over the same windows, the norms
   kept in float32 buffers giving exactly the quantiles of the Python lists they replace;
4. wiring.npz: each neuron's links each way are `neuron_wiring`'s, the same neurons, z (as float16 keeps it) and sign;
5. steer_lens.d / steer_lens.npz: each neuron's rises are `steer_lens`'s, KL too; a stopped run resumes and merges;
6. layout_writes.npz: each layer's cells are distinct places on its sheet;
7. the queries: a search finds a face by a word-piece of its label at the right rank; the cone's hops move one way
   each side and start from the neuron; the web summary pools the steer lens when built and, without it, only faces
   the label gate lets through (on the stand-in's six layers every face is held back, so it pools nothing and says how
   many it left out); the active paths join only neurons that fired; the scores scale by each side's extreme;
8. head profiles: each head's shares of weight add up to one per read, and its pairs read more than one unit back.

Run: .venv/bin/python -I scripts/check_atlas_index.py   (exits 0 when every check holds)
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"

import shutil
import tempfile
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch

import backend.atlas_build as ab
import backend.atlas_ops as ao
from atlas_stand_in import stand_in
from backend.atlas_index import AtlasIndex, match_ids, scale_scores
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
    check(idx.get("wiring")["meta"].get("scale_source") == "check" and "from rest" in idx.get("wiring")["meta"].get("link_model", ""),
          "wiring: the scales' source and the link model recorded")
    section("the wiring index is neuron_wiring's, link by link, in order of strength across layers", before)

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
    check(all(mine.get(k_) is not None for k_ in ("lens", "act_scale", "wiring", "steer_lens", "layout")), "the model's own indexes are read")
    other = AtlasIndex(out, model="google/gemma-4-31B-it", shape=(L, N))
    st = other.status()
    check(other.get("lens") is None and not st["lens"]["built"] and "built for" in st["lens"]["refused"], f"another model's index refused: {st['lens']}")
    wide = AtlasIndex(out, model=rt.model_id, shape=(L, N + 1))
    check(wide.get("wiring") is None and "shaped for" in wide.status()["wiring"]["refused"], "an index of another shape refused")
    check(other.face_ids(1, 2, "+", "lens") is None and other.cone(1, 2) == [] and other.links(1, 2) is None, "a refused index reads as not built")
    oldw = dict(np.load(out / "wiring.npz"))
    for side in ("down", "up"):
        del oldw[f"{side}_lw"], oldw[f"{side}_gate"]
    np.savez(part / "wiring.npz", **oldw)
    check(pidx.get("wiring") is None and "weighed across layers" in pidx.status()["wiring"]["refused"],
          "a wiring index built before links were weighed across layers is refused")
    section("an index built for another model, of another shape, or by an older build, is not read", before)

    shutil.rmtree(out, ignore_errors=True)
    print(f"\n{passed} checks passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

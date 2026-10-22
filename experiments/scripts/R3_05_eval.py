#!/usr/bin/env python3
"""R3 05 (GPU): read the fresh prompts through J++, the naked eye, R3's two candidate lenses and the reference
lenses, filtered and not; then the secondary readings (R3_PREREG.md §4-§5, §8). The analysis is R3_06_analyze.py.

A generalisation of R2_02_eval.py, standalone: it loads the served checkpoint itself (R3_gpu.load_w4a16; no server,
no lab daemon, no branch-only backend name) and every lens from its own pinned file.
  1. Checks before the model: the registered files are committed (else --allow-uncommitted, a deviation); no
     earlier output (else --force); R3_prompts.json validates with R3's counts and its sha256 is the one
     R3_01_check.json recorded (passing); R2's filter has R2_01_check.json's class counts; J++'s exclusion list has
     R3_01_check.json's count; every lens file has its pinned size and sha256 (J++, JLens and Miru from R3_common,
     R1's lens from R3_common's pin, R3's own from R3_04_weights.json); R1_10's held-out positions rebuild with
     their pinned sha256. Every run past these checks is appended to R3_05_runs.jsonl (started/finished/failed).
  2. Greedy continuations by full forwards (no KV cache), as R2: 8 tokens after each chat-no-answer, chat-begun and
     bare prompt, 48-token replies to the mid-reply questions; the model's own word is the argmax of the final norm,
     W_U and softcap of resid_post[59] (the instrument's unembed_gemma4).
  3. Read positions as R2 (R2_common.first_content, R2's stop rule), the mid-reply draws seeded 20261021 (R3) or
     20261018 (R2's replication set, so its positions are R2's).
  4. resid_post at code layers 30-59 at the read positions, one batched forward per group of prompts.
  5. Readouts through the instrument's own functions (jpp_readout_logits / unembed_gemma4, bf16): J++, jpp-it,
     rl-it, Miru and jpp-it-full-macro at 30-58, the naked eye at 30-59, R1's lens at 50/55/58, one lens layer on
     the GPU at a time. Per readout: top-5 ids and texts unfiltered ("u", "us"), with R2's filter ("f", "fs") and
     with J++'s exclusion list ("fj", "fjs", for sensitivity analysis S5); the rank of the next word-piece
     unfiltered ("ur") and of the content target filtered ("fr"); whether the unfiltered top-1 is content ("u0c").
  6. Secondary readings: H4, fidelity on R1_10's 4,808 held-out positions (E3's prose and chats, rebuilt here from
     E3's own files, their list pinned by sha256; the states regenerated): KL(final || lens) at R1_10's seven logit
     scales, the calibrated KL (cross-fitted as R1_14: the scale chosen on one sequence parity, scored on the
     other), top-1 and top-10 agreement, at code 40/45/50/55/58, over all positions and per class, with intervals
     for every lens against J++ from a bootstrap over sequences, and H4's registered reading; the per-position
     values are kept (R3_05_h4_positions.pt). J++, JLens and the naked eye must reproduce R1_10_evalfull.pt's top-1
     within 0.01: if they do not, H4's output says ok false with a caution, and the runs log records the failed
     check. J++'s own held-out recall@10 at
     code 30/38/45/52 (R3_fitlib.macro_pass on R3_04's held-out residuals). The pullbacks d . J of R1's 128
     directions through R3's lens files (for R3_06's H2 and H3).
Writes R3_05_eval.json (R3's prompts), R3_05_eval_r2set.json (R2's 220, the replication set), R3_05_h4.json,
R3_05_recall.json, and the gitignored R3_05_states.pt, R3_05_h4_positions.pt and R3_05_pullbacks.pt.

Run (GPU, server stopped):  /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R3_05_eval.py
                                [--batch 8] [--force] [--allow-uncommitted]
Every check before the model (CPU):  CUDA_VISIBLE_DEVICES="" ... R3_05_eval.py --check-only
Self-test (CPU, a tiny random stand-in for the model with the real tokenizer and filters; R2's prompts stand in for
R3's until R3_prompts.json is written):  CUDA_VISIBLE_DEVICES="" ... R3_05_eval.py --selftest --limit 3 --out X.json
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
if "--selftest" in sys.argv or "--check-only" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""        # the CPU modes never see the GPU
sys.dont_write_bytecode = True

import argparse
import datetime
import hashlib
import json
import platform
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import R2_common as R2  # noqa: E402
import R3_common as C  # noqa: E402

T0 = time.time()
KINDS: Tuple[str, ...] = ("jpp", "naive", "jpp-it", "rl-it", "r1", "miru", "jpp-it-full-macro")
LENS_LAYERS: Dict[str, Tuple[int, ...]] = {
    "jpp": C.FIT_LAYERS, "naive": C.READ_LAYERS, "jpp-it": C.FIT_LAYERS, "rl-it": C.FIT_LAYERS, "r1": (50, 55, 58),
    "miru": C.FIT_LAYERS, "jpp-it-full-macro": C.FIT_LAYERS}
H4_KINDS: Tuple[str, ...] = ("jpp", "jlens", "naive", "jpp-it", "rl-it", "r1", "miru")
R2SET_JSON = C.RESULTS / "R3_05_eval_r2set.json"
H4_JSON = C.RESULTS / "R3_05_h4.json"
H4_POSITIONS_PT = C.RESULTS / "R3_05_h4_positions.pt"   # gitignored: H4's per-position values, every lens and layer
RECALL_JSON = C.RESULTS / "R3_05_recall.json"
STATES_PT = C.RESULTS / "R3_05_states.pt"
PULLBACKS_PT = C.RESULTS / "R3_05_pullbacks.pt"
R1_10_LENSES = ("jpp", "jlens", "naive")             # the lenses R1_10 measured that R3 measures again
R1_10_FM = (10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 58)


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------------------------------------ positions
def draw_mid_positions(reply: Sequence[int], is_content, prompt_index: int, seed: int) -> Dict[int, List[str]]:
    """R2_common.draw_mid_positions with the seed as an argument (R2's code, unchanged otherwise; --selftest shows
    that with R2's seed it is R2's function)."""
    n = len(reply)
    elig = list(range(max(0, n - 1)))
    cn = [k for k in elig if bool(is_content[int(reply[k + 1])])]
    out: Dict[int, List[str]] = {}
    for d, (name, pool) in enumerate((("content_next", cn), ("any", elig))):
        rng = np.random.default_rng([seed, int(prompt_index), d])
        if not pool:
            continue
        pick = rng.choice(len(pool), size=min(C.MID_DRAW_SIZES[name], len(pool)), replace=False)
        for p in sorted(int(x) for x in pick):
            out.setdefault(pool[p], []).append(name)
    return dict(sorted(out.items()))


def build_positions(items, conts, replies, is_content, is_special, seed: int) -> List[Dict[str, Any]]:
    """R2_02_eval.build_positions with R3's draw seed (the read positions and their targets)."""
    pos_recs: List[Dict[str, Any]] = []
    mid_index = 0
    for it in items:
        ids = it["ids"]
        if it["cond"] != "mid-reply":
            cont = conts[it["pid"]]
            window = [int(t) for t in cont[: C.CONT_WINDOW]]
            ct = R2.first_content(window, is_content, is_special)
            pos_recs.append({"pid": it["pid"], "cond": it["cond"], "draws": None, "k": None, "pos": len(ids) - 1,
                             "seq": ids, "next_id": int(cont[0]), "cont_ids": window, "ct": ct,
                             "ct_prev": ([int(ids[-1])] + window)[ct[0]] if ct else None})
            continue
        full = replies[it["pid"]]
        reply = []
        for t in full:
            if int(t) in R2.STOP_IDS:
                break
            reply.append(int(t))
        it["reply_ids"] = reply
        it["reply_stopped"] = len(reply) < len(full)
        draws = draw_mid_positions(reply, is_content, mid_index, seed)
        it["draws"] = {str(k): v for k, v in draws.items()}
        mid_index += 1
        seq = ids + reply
        for k, names in draws.items():
            window = reply[k + 1: k + 1 + C.CONT_WINDOW]
            ct = R2.first_content(window, is_content, is_special)
            pos_recs.append({"pid": it["pid"], "cond": "mid-reply", "draws": names, "k": k, "pos": len(ids) + k,
                             "seq": seq, "next_id": int(reply[k + 1]), "cont_ids": window, "ct": ct,
                             "ct_prev": ([int(reply[k])] + window)[ct[0]] if ct else None})
    return pos_recs


def greedy(backend, seqs: List[List[int]], n_new: int, batch: int, label: str) -> List[List[int]]:
    """R2_02_eval.greedy: full forwards, the model's own readout of resid_post[59] at each live sequence's end."""
    out: List[List[int]] = [[] for _ in seqs]
    live = list(range(len(seqs)))
    for step in range(n_new):
        if not live:
            break
        nxt = []
        for b0 in range(0, len(live), batch):
            idx = live[b0:b0 + batch]
            ids = [seqs[i] + out[i] for i in idx]
            st = backend.states(ids, [[len(x) - 1] for x in ids], [59])
            h = torch.stack([s[0, 0] for s in st], 0)
            for i, t in zip(idx, backend.final_logits(h).argmax(-1).tolist()):
                out[i].append(int(t))
                if int(t) not in R2.STOP_IDS:
                    nxt.append(i)
        live = nxt
        if step == 0 or (step + 1) % 8 == 0 or not live:
            log(f"  {label}: step {step + 1}/{n_new}, {len(live)} still going")
    return out


# ------------------------------------------------------------------------------------------------ R1_10's positions
class _E3Lab:
    """The few labkit.Lab methods E3's position helper calls, on the tokenizer alone (labkit's own definitions:
    strs = per-id decode, model_turn_pos = the last '<|turn>model' header)."""

    def __init__(self, tok):
        self.tok = tok

    def strs(self, ids):
        return [self.tok.decode([i]) for i in ids]

    def model_turn_pos(self, ids):
        strs = self.strs(ids)
        cand = [p for p in range(1, len(ids)) if strs[p - 1] == "<|turn>" and strs[p].startswith("model")]
        if not cand:
            raise ValueError("no model turn")
        return cand[-1]


def r1_10_sequences(tok) -> List[Dict[str, Any]]:
    """R1_10_evalfull.py's held-out sequences and their read regions, rebuilt from E3's own files (exec'd into a
    namespace, as R1's daemon jobs exec'd them): 24 prose paragraphs read from position 4, and E3's chats read at
    the assistant tokens (self and non-self) and the user turn."""
    if C.sha256_file(C.E3_CHATS_PATH) != C.E3_CHATS_SHA256:
        raise SystemExit(f"{C.E3_CHATS_PATH} is not the pinned E3 chats file")
    ns: Dict[str, Any] = {}
    for f in ("E3_lens_validation_data.py", "E3_lens_validation_fidlib.py"):
        exec(compile((C.SCRIPTS / f).read_text(encoding="utf-8"), f, "exec"), ns)  # noqa: S102 (the study's data)
    lab = _E3Lab(tok)
    seqs = []
    for i, (_tag, txt) in enumerate(ns["PROSE"]):
        ids = R2.text_ids(tok, txt)
        seqs.append({"name": f"prose{i:02d}", "ids": ids, "regions": {"prose": list(range(4, len(ids)))}})
    for i, c in enumerate(json.loads(C.E3_CHATS_PATH.read_text(encoding="utf-8"))):
        ids = R2.text_ids(tok, R2.chat_text(tok, c["q"], c["a"]))
        cats = ns["e3_chat_positions"](lab, ids, ns["SELF_TOKS"])
        seqs.append({"name": f"chat{i:02d}", "ids": ids,
                     "regions": {"asst": cats["nonself_asst"], "self": cats["self"], "user": cats["user"]}})
    return seqs


def positions_r1_10(seqs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    meta = [{"name": s["name"], "cls": c, "pos": p} for s in seqs for c, ps in s["regions"].items() for p in ps]
    return meta, hashlib.sha256(json.dumps([[m["name"], m["cls"], m["pos"]] for m in meta],
                                           separators=(",", ":")).encode()).hexdigest()


def r1_10_metrics(hJ: torch.Tensor, lpf: torch.Tensor, t10f: torch.Tensor, fnw: torch.Tensor, wu: torch.Tensor,
                  taus: Sequence[float] = C.H4_TAUS) -> Dict[str, torch.Tensor]:
    """R1_10_evalfull.py's metrics for lens outputs hJ [P, D] against the final distribution (log-probs lpf [P, V]
    and its top-10 t10f [P, 10]): KL(final || lens) at each logit scale tau (the pre-softcap logits times tau, then
    the 30.0 softcap), top-1 agreement and top-10 overlap at tau 1. R1_10's own RMS (eps 1e-6) and unembed."""
    x = hJ.float()
    z = ((x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * fnw.float()).to(wu.dtype) @ wu.T).float()
    pf = lpf.exp()
    kl = []
    for tau in taus:
        lp = F.log_softmax(torch.tanh(z * tau / 30.0) * 30.0, -1)
        kl.append((pf * (lpf - lp)).sum(-1))
    t10 = torch.topk(torch.tanh(z / 30.0) * 30.0, 10, -1).indices
    return {"kl": torch.stack(kl, -1), "top1": t10[:, 0] == t10f[:, 0],
            "top10": (t10.unsqueeze(-1) == t10f.unsqueeze(-2)).any(-1).float().mean(-1)}


def final_reference(h59: torch.Tensor, fnw: torch.Tensor, wu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x = h59.float()
    z = ((x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * fnw.float()).to(wu.dtype) @ wu.T).float()
    lf = torch.tanh(z / 30.0) * 30.0
    return F.log_softmax(lf, -1), torch.topk(lf, 10, -1).indices


def _seq_sums(sel: np.ndarray, names: np.ndarray, m: Dict[str, np.ndarray]):
    """Per sequence of the selection (sorted names): position count, top-1 and top-10 sums, KL sums per scale."""
    seqs = sorted(set(names[sel]))
    ix = {n: i for i, n in enumerate(seqs)}
    sid = np.array([ix[n] for n in names[sel]])
    S = len(seqs)
    n_s = np.bincount(sid, minlength=S).astype(float)
    t1 = np.bincount(sid, weights=m["top1"][sel].astype(float), minlength=S)
    t10 = np.bincount(sid, weights=m["top10"][sel].astype(float), minlength=S)
    kl = np.stack([np.bincount(sid, weights=m["kl"][sel][:, k].astype(float), minlength=S)
                   for k in range(m["kl"].shape[1])], 1)
    return n_s, t1, t10, kl, np.arange(S) % 2


def _crossfit(W: np.ndarray, n_s: np.ndarray, kl_s: np.ndarray, par: np.ndarray) -> np.ndarray:
    """R1_14's cross-fitted calibrated KL for each row of sequence weights W [B, S]: per parity a, the scale with the
    least mean KL on parity a, scored on the other parity; position-weighted over both."""
    tot = np.zeros(W.shape[0])
    for a in (0, 1):
        fa, ea = par == a, par != a
        with np.errstate(invalid="ignore", divide="ignore"):
            fit = (W[:, fa] @ kl_s[fa]) / (W[:, fa] @ n_s[fa])[:, None]
        best = np.argmin(np.where(np.isnan(fit), np.inf, fit), 1)
        tot += (W[:, ea] @ kl_s[ea])[np.arange(W.shape[0]), best]
    with np.errstate(invalid="ignore", divide="ignore"):
        return tot / (W @ n_s)


def h4_tables(names: np.ndarray, cls: np.ndarray, res: Dict[str, Dict[int, Dict[str, np.ndarray]]], *,
              taus: Sequence[float] = C.H4_TAUS, seed: int = C.H4_BOOT_SEED,
              n_boot: int = C.H4_N_BOOT) -> Dict[str, Any]:
    """H4's tables from per-position values ({kind: {layer: {"kl" [P, T], "top1" [P], "top10" [P]}}}): per lens,
    layer and class (all, then each class) the point values, and against J++ the differences with 95% intervals from
    a bootstrap over sequences (the same resampled sequences for every lens and layer); and the registered reading
    (R3_PREREG.md §1.2): for jpp-it and rl-it, "supported" at a layer if the lower end of the interval of the top-1
    difference from J++ over all positions is above 0, else "refuted"."""
    t1i = list(taus).index(1.0)
    groups = ["all"] + sorted(set(cls))
    lenses: Dict[str, Any] = {}
    vs: Dict[str, Any] = {}
    for gi, c in enumerate(groups):
        sel = np.ones(len(cls), bool) if c == "all" else cls == c
        rng = np.random.default_rng([seed, gi])
        stats_b: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
        W = None
        for kind, by_l in res.items():
            for l, m in by_l.items():
                n_s, t1, t10, kl_s, par = _seq_sums(sel, names, m)
                if W is None:
                    S = len(n_s)
                    ix = rng.integers(0, S, size=(n_boot, S))
                    W = np.zeros((n_boot, S))
                    np.add.at(W, (np.repeat(np.arange(n_boot), S), ix.ravel()), 1.0)
                one = np.ones((1, len(n_s)))
                N = float(n_s.sum())
                kl_mean = kl_s.sum(0) / N
                b = int(np.argmin(kl_mean))
                lenses.setdefault(kind, {}).setdefault(str(l), {})[c] = {
                    "n_positions": int(N), "n_sequences": int(len(n_s)),
                    "kl_by_tau": [round(float(v), 4) for v in kl_mean], "kl_tau1": round(float(kl_mean[t1i]), 4),
                    "kl_min_over_taus_in_sample": round(float(kl_mean[b]), 4), "best_tau_in_sample": taus[b],
                    "kl_calibrated_crossfit": round(float(_crossfit(one, n_s, kl_s, par)[0]), 4),
                    "top1": round(float(t1.sum() / N), 4), "top10": round(float(t10.sum() / N), 4)}
                with np.errstate(invalid="ignore", divide="ignore"):
                    den = W @ n_s
                    stats_b[(kind, l)] = {"top1": W @ t1 / den, "top10": W @ t10 / den,
                                          "kl_tau1": (W @ kl_s[:, t1i]) / den,
                                          "kl_calibrated_crossfit": _crossfit(W, n_s, kl_s, par)}
        if "jpp" not in res:
            continue
        for kind, by_l in res.items():
            if kind == "jpp":
                continue
            for l in by_l:
                if l not in res["jpp"]:
                    continue
                row = {}
                for key in ("top1", "top10", "kl_tau1", "kl_calibrated_crossfit"):
                    d = stats_b[(kind, l)][key] - stats_b[("jpp", l)][key]
                    point = lenses[kind][str(l)][c][key] - lenses["jpp"][str(l)][c][key]
                    lo, hi = np.nanpercentile(d, [2.5, 97.5])
                    row[key] = {"diff": round(float(point), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4)}
                vs.setdefault(kind, {}).setdefault(str(l), {})[c] = row
    reading = {}
    for kind in ("jpp-it", "rl-it"):
        for l, rows in vs.get(kind, {}).items():
            lo = rows["all"]["top1"]["lo"]
            reading.setdefault(kind, {})[l] = "supported" if lo > 0 else "refuted"
    return {"lenses": lenses, "vs_jpp": vs, "h4_reading": reading,
            "h4_rule": "jpp-it (or rl-it) reads the instruct model more faithfully than base J++ at a layer only if "
                       "the lower end of the 95% sequence-bootstrap interval of its top-1 agreement minus J++'s, over "
                       "all positions, is above 0"}


def h4(backend, tok, kinds: Sequence[str] = H4_KINDS, chunk: int = 256,
       positions_pt: Optional[Path] = None) -> Dict[str, Any]:
    seqs = r1_10_sequences(tok)
    meta, sha = positions_r1_10(seqs)
    if sha != C.R1_10_POSITIONS_SHA256 or len(meta) != C.R1_10_N_POSITIONS:
        raise SystemExit(f"R1_10's held-out positions rebuild with sha256 {sha[:12]} ({len(meta)}), not the pin")
    layers = list(C.H4_LAYERS)
    P = len(meta)
    D = backend.d_model
    H = torch.empty(len(layers) + 1, P, D, dtype=torch.bfloat16)
    k = 0
    for s in seqs:
        allp = [p for ps in s["regions"].values() for p in ps]
        st = backend.states([s["ids"]], [allp], layers + [59])[0]                  # [L+1, P_s, D]
        H[:, k:k + len(allp)] = st.to(torch.bfloat16)
        k += len(allp)
    fnw, wu = backend.fnw, backend.wu
    res: Dict[str, Dict[int, Dict[str, torch.Tensor]]] = {}
    for li, l in enumerate(layers):
        for kind in kinds:
            if l not in backend.h4_layers(kind):
                continue
            J = backend.lens_matrix(kind, l)
            acc = {"kl": [], "top1": [], "top10": []}
            for c0 in range(0, P, chunk):
                h59 = H[-1, c0:c0 + chunk].to(backend.dev)
                lpf, t10f = final_reference(h59, fnw, wu)
                h = H[li, c0:c0 + chunk].to(backend.dev)
                hJ = h.to(torch.bfloat16) if J is None else (h.to(J.dtype) @ J.T)
                m = r1_10_metrics(hJ, lpf, t10f, fnw, wu)
                for key in acc:
                    acc[key].append(m[key].cpu())
            res.setdefault(kind, {})[l] = {key: torch.cat(v) for key, v in acc.items()}
            backend.release(kind, J)
        log(f"  H4 at layer {l}")
    cls = np.array([m["cls"] for m in meta])
    names = np.array([m["name"] for m in meta])
    out: Dict[str, Any] = {"positions": P, "positions_sha256": sha, "layers": layers, "taus": list(C.H4_TAUS),
                           "classes": {c: int((cls == c).sum()) for c in sorted(set(cls))},
                           "sequences": int(len(set(names))),
                           "bootstrap": {"over": "sequences (R1_10's 24 prose paragraphs and 24 chats), within the "
                                                 "class", "n": C.H4_N_BOOT, "seed": [C.H4_BOOT_SEED, "class index"],
                                         "ci": "95% percentile"},
                           "calibrated_kl": "cross-fitted as R1_14: the scale chosen on one sequence parity (sorted "
                                            "names) and scored on the other",
                           **h4_tables(names, cls, {k: {l: {key: v.numpy() for key, v in m.items()}
                                                        for l, m in by_l.items()} for k, by_l in res.items()})}
    if positions_pt is not None:
        torch.save({"meta": meta, "taus": list(C.H4_TAUS), "lenses": res}, positions_pt)
    # R1_10's own numbers for the lenses it measured, from its stored per-position results
    if C.R1_10_PATH.is_file():
        r1 = torch.load(C.R1_10_PATH, weights_only=False, map_location="cpu")
        rep = {}
        for kind in R1_10_LENSES:
            if kind not in res:
                continue
            k_r1 = r1["lenses"].index(kind)
            for l in layers:
                j = R1_10_FM.index(l)
                was = float(r1["T1"][:, j, k_r1].double().mean())
                now = out["lenses"][kind][str(l)]["all"]["top1"]
                rep[f"{kind}@{l}"] = {"r1_10_top1": round(was, 4), "r3_top1": now, "diff": round(now - was, 4)}
        out["reproduces_r1_10"] = {"rows": rep, "max_abs_diff": max((abs(v["diff"]) for v in rep.values()),
                                                                   default=None), "tolerance": 0.01}
        out["reproduces_r1_10"]["ok"] = out["reproduces_r1_10"]["max_abs_diff"] is not None and \
            out["reproduces_r1_10"]["max_abs_diff"] <= 0.01
    else:
        out["reproduces_r1_10"] = {"ok": False, "not_run": f"{C.R1_10_PATH} is not there: the check could not run",
                                   "tolerance": 0.01}
    out["ok"] = bool(out["reproduces_r1_10"]["ok"])
    if not out["ok"]:
        out["caution"] = ("J++, JLens and the naked eye did not reproduce R1_10's top-1 within 0.01 (D23), or the check "
                          "could not run: every H4 number is reported with this caution (R3_PREREG.md §8.3)")
    return out


# ------------------------------------------------------------------------------------------------ backends
def lens_manifest(weights_json: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Every lens file read by the run, with its pinned sha256 and size (R3's own: as R3_04_weights.json records)."""
    man = {"jpp": {"path": C.JPP_BASE_PATH, "sha256": C.JPP_BASE_SHA256, "size": C.JPP_BASE_SIZE, "format": "pt"},
           "jlens": {"path": C.JLENS_PATH, "sha256": C.JLENS_SHA256, "size": C.JLENS_SIZE, "format": "pt"},
           "miru": {"path": C.MIRU_PATH, "sha256": C.MIRU_SHA256, "size": C.MIRU_SIZE, "format": "safetensors"}}
    r1, _ = C.find_r1_lens(full=False)
    man["r1"] = {"path": r1, "sha256": C.R1_SHA256, "size": C.R1_SIZE, "format": "pt"}
    wj = weights_json or C.WEIGHTS_JSON
    lf = json.loads(wj.read_text(encoding="utf-8")).get("lens_files", {}) if wj.is_file() else {}
    for k in ("jpp-it", "rl-it", "jpp-it-full-macro"):
        v = lf.get(k)
        man[k] = {"path": Path(v["path"]) if v else C.LENS_FILES[k], "sha256": v["sha256"] if v else None,
                  "size": None, "format": "pt"}
    return man


def check_lens_files(man: Dict[str, Dict[str, Any]], kinds: Sequence[str], full: bool = True) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in kinds:
        if k == "naive":
            continue
        m = man[k]
        p = m["path"]
        rec: Dict[str, Any] = {"path": str(p) if p else None, "exists": bool(p) and Path(p).is_file()}
        if rec["exists"] and m["size"] is not None:
            rec["size_ok"] = Path(p).stat().st_size == m["size"]
        if rec["exists"] and full and m["sha256"]:
            rec["sha256_ok"] = C.sha256_file(Path(p)) == m["sha256"]
        rec["pinned"] = m["sha256"] is not None
        out[k] = rec
    return out


class GpuBackend:
    """The served checkpoint (R3_gpu.load_w4a16, no lens loaded by it) and every lens from its own file."""

    def __init__(self, man: Dict[str, Dict[str, Any]]):
        sys.path.insert(0, str(C.REPO))
        import R3_gpu as G
        from backend.theodolite_math import jpp_readout_logits, unembed_gemma4, validate_and_cast_lens

        import R3_fitlib as FL

        self._readout, self._unembed, self._cast = jpp_readout_logits, unembed_gemma4, validate_and_cast_lens
        self.man = man
        self.hf, self.tok = G.load_w4a16()
        self.decoder = FL.Decoder(self.hf, self.tok, name=C.MODEL_ID)
        self.dev = self.decoder.input_device
        self.fnw = self.decoder.final_norm.weight.detach()
        self.wu = self.decoder.lm_head.weight.detach()
        self.d_model = self.decoder.d_model
        self.vocab_rows = int(self.wu.shape[0])
        self._open: Dict[str, Any] = {}

    def check_text(self, item, text, ids):
        assert R2.text_ids(self.tok, text) == ids, item["pid"]

    @torch.no_grad()
    def states(self, batch_ids: List[List[int]], positions: List[List[int]], layers: Sequence[int]):
        """[L, P_b, D] fp32 CPU per item: resid_post at `layers` and the positions (right-padded batch with an
        attention mask, through the multimodal wrapper's Gemma4Model, as labkit.forward_resid reads it)."""
        B, T = len(batch_ids), max(len(x) for x in batch_ids)
        inp = torch.zeros(B, T, dtype=torch.long)
        att = torch.zeros(B, T, dtype=torch.long)
        for b, x in enumerate(batch_ids):
            inp[b, :len(x)] = torch.tensor(x)
            att[b, :len(x)] = 1
        store: Dict[int, torch.Tensor] = {}
        hs = []
        for l in layers:
            def hook(mod, args, out, l=l):
                store[l] = (out[0] if isinstance(out, tuple) else out).detach()
            hs.append(self.decoder.layers[l].register_forward_hook(hook))
        try:
            self.hf.model(input_ids=inp.to(self.dev), attention_mask=att.to(self.dev), use_cache=False)
        finally:
            for h in hs:
                h.remove()
        return [torch.stack([store[l][b, torch.tensor(positions[b], device=self.dev)] for l in layers]).float().cpu()
                for b in range(B)]

    def final_logits(self, h):
        return self._unembed(h.to(self.dev, torch.bfloat16), self.fnw, self.wu).float()

    def unembed(self, x):
        return self._unembed(x.to(self.dev, torch.bfloat16), self.fnw, self.wu).float()

    @property
    def input_device(self):
        return self.dev

    def h4_layers(self, kind: str) -> Tuple[int, ...]:
        return tuple(range(0, 60)) if kind == "naive" else (LENS_LAYERS.get(kind) or tuple(range(59)))

    def lens_matrix(self, kind: str, l: int):
        if kind == "naive" or l == 59:
            return None
        m = self.man[kind]
        if m["format"] == "safetensors":
            from safetensors import safe_open

            with safe_open(str(m["path"]), framework="pt", device="cpu") as f:
                raw = f.get_tensor(f"J.{l}")
        else:
            if kind not in self._open:
                p = torch.load(m["path"], map_location="cpu", mmap=True, weights_only=True)
                self._open[kind] = p["J"] if isinstance(p, dict) and "J" in p else p
            raw = self._open[kind][l]
        return self._cast(raw, layer_idx=l, target_dtype=torch.bfloat16, device=self.dev)

    def readout_logits(self, kind, l, h, J):
        h = h.to(self.dev, torch.bfloat16)
        if J is None:
            return self._unembed(h, self.fnw, self.wu).float()
        return self._readout(h, J, self.fnw, self.wu)[0].float()

    def release(self, kind, J):
        del J
        torch.cuda.empty_cache()

    def info(self):
        import transformers

        return {"backend": "gpu", "device": torch.cuda.get_device_name(self.dev), "torch": torch.__version__,
                "transformers": transformers.__version__, "model": C.MODEL_ID, "snapshot": C.MODEL_SNAPSHOT}


class FakeBackend:
    """A tiny deterministic stand-in (D = 32, random embeddings and lenses) for the CPU self-test of the pipeline:
    generation, positions, readouts, both filters, ranks, H4's metrics and the output files. It knows nothing true."""

    D = 32

    def __init__(self, tok, seed: int = 0):
        sys.path.insert(0, str(C.REPO))
        from backend.theodolite_math import jpp_readout_logits, unembed_gemma4

        self._readout, self._unembed = jpp_readout_logits, unembed_gemma4
        g = torch.Generator().manual_seed(seed)
        V, D = len(tok), self.D
        self.tok, self.dev, self.d_model, self.vocab_rows = tok, torch.device("cpu"), D, V
        self.E = torch.randn(V, D, generator=g)
        self.wu = torch.randn(V, D, generator=g)
        self.fnw = torch.ones(D)
        self.J = {k: {l: torch.eye(D) + 0.4 * torch.randn(D, D, generator=g) for l in range(59)}
                  for k in KINDS + ("jlens",) if k != "naive"}
        u = torch.randn(D, generator=g)
        self.J["miru"][58] = torch.outer(u, torch.ones(D)) * 0.1 + 1e-3 * torch.eye(D)   # a stuck layer
        self.input_device = self.dev

    def check_text(self, item, text, ids):
        pass

    def states(self, batch_ids, positions, layers):
        out = []
        for ids, pos in zip(batch_ids, positions):
            e = self.E[torch.tensor(ids)]
            csum = torch.cumsum(e, 0) / torch.arange(1, len(ids) + 1).unsqueeze(1)
            out.append(torch.stack([torch.stack([e[p] * (1 + l / 60) + 0.5 * csum[p] for p in pos]) for l in layers]))
        return out

    def final_logits(self, h):
        return self._unembed(h.float(), self.fnw, self.wu).float()

    def unembed(self, x):
        return self._unembed(x.float(), self.fnw, self.wu).float()

    def h4_layers(self, kind):
        return tuple(range(60)) if kind == "naive" else (LENS_LAYERS.get(kind) or tuple(range(59)))

    def lens_matrix(self, kind, l):
        return None if kind == "naive" or l == 59 else self.J[kind][l]

    def readout_logits(self, kind, l, h, J):
        if J is None:
            return self._unembed(h.float(), self.fnw, self.wu).float()
        return self._readout(h.float(), J, self.fnw, self.wu)[0].float()

    def release(self, kind, J):
        pass

    def info(self):
        return {"backend": "fake (CPU self-test; no model)", "torch": torch.__version__}


# ------------------------------------------------------------------------------------------------ the band readings
def run_set(backend, tok, data: Dict[str, Any], mask: torch.Tensor, jmask: torch.Tensor, is_special, *, seed: int,
            batch: int, limit: Optional[int], out_json: Path, out_states: Optional[Path],
            meta_extra: Dict[str, Any], kinds: Sequence[str] = KINDS) -> Dict[str, Any]:
    is_content = mask.numpy()
    if len(is_content) != backend.vocab_rows or len(jmask) != backend.vocab_rows:
        raise SystemExit(f"the masks have {len(is_content)} / {len(jmask)} ids, W_U has {backend.vocab_rows} rows")
    items = R2.prompt_items(data)
    if limit:
        items = [it for c in C.CONDITIONS for it in [x for x in items if x["cond"] == c][:limit]]
    checked = set()
    for it in items:
        it["text_read"] = R2.prompt_text(tok, it)
        it["ids"] = R2.text_ids(tok, it["text_read"])
        if it["cond"] not in checked:
            backend.check_text(it, it["text_read"], it["ids"])
            checked.add(it["cond"])
    log(f"{len(items)} prompts: " + ", ".join(f"{c} {sum(it['cond'] == c for it in items)}" for c in C.CONDITIONS))
    t = time.time()
    first = [it for it in items if it["cond"] != "mid-reply"]
    mids = [it for it in items if it["cond"] == "mid-reply"]
    conts = {it["pid"]: c for it, c in zip(first, greedy(backend, [it["ids"] for it in first], C.CONT_WINDOW, batch,
                                                           "continuations"))}
    replies = {it["pid"]: r for it, r in zip(mids, greedy(backend, [it["ids"] for it in mids], C.MID_NEW_TOKENS,
                                                          batch, "mid-reply replies"))}
    t_gen = time.time() - t
    pos_recs = build_positions(items, conts, replies, is_content, is_special, seed)
    N, L = len(pos_recs), len(C.READ_LAYERS)
    t = time.time()
    by_pid: Dict[str, List[int]] = {}
    for j, p in enumerate(pos_recs):
        by_pid.setdefault(p["pid"], []).append(j)
    pids = list(by_pid)
    H = None
    for b0 in range(0, len(pids), batch):
        grp = pids[b0:b0 + batch]
        st = backend.states([pos_recs[by_pid[pid][0]]["seq"] for pid in grp],
                            [[pos_recs[j]["pos"] for j in by_pid[pid]] for pid in grp], C.READ_LAYERS)
        if H is None:
            H = torch.empty(N, L, st[0].shape[-1], dtype=torch.bfloat16)
        for pid, s in zip(grp, st):
            for q, j in enumerate(by_pid[pid]):
                H[j] = s[:, q].to(torch.bfloat16)
    t_read = time.time() - t
    t = time.time()
    dev = backend.dev
    mask_dev, jmask_dev = mask.to(dev), jmask.to(dev)
    next_ids = torch.tensor([p["next_id"] for p in pos_recs], device=dev)
    ct_ids = torch.tensor([p["ct"][1] if p["ct"] else -1 for p in pos_recs], device=dev)
    strs: Dict[int, str] = {}

    def s_of(i: int) -> str:
        if i not in strs:
            strs[i] = R2.token_text(tok, i)
        return strs[i]

    R: List[Dict[str, Dict[str, Any]]] = [dict() for _ in range(N)]
    CH = 64
    for li, l in enumerate(C.READ_LAYERS):
        for kind in [k for k in kinds if l in LENS_LAYERS[k]]:
            J = backend.lens_matrix(kind, l)
            for s0 in range(0, N, CH):
                lg = backend.readout_logits(kind, l, H[s0:s0 + CH, li], J)
                u = torch.topk(lg, C.TOPK, dim=-1).indices
                lgf = lg.masked_fill(~mask_dev, float("-inf"))
                f = torch.topk(lgf, C.TOPK, dim=-1).indices
                fj = torch.topk(lg.masked_fill(~jmask_dev, float("-inf")), C.TOPK, dim=-1).indices
                nx, ct = next_ids[s0:s0 + CH], ct_ids[s0:s0 + CH]
                ur = (lg > lg.gather(1, nx[:, None])).sum(1) + 1
                fr = (lgf > lgf.gather(1, ct.clamp(min=0)[:, None])).sum(1) + 1
                u_l, f_l, fj_l, ur_l, fr_l = u.tolist(), f.tolist(), fj.tolist(), ur.tolist(), fr.tolist()
                for q in range(len(u_l)):
                    R[s0 + q].setdefault(kind, {})[str(l)] = {
                        "u": u_l[q], "us": [s_of(i) for i in u_l[q]], "f": f_l[q], "fs": [s_of(i) for i in f_l[q]],
                        "fj": fj_l[q], "fjs": [s_of(i) for i in fj_l[q]], "ur": int(ur_l[q]),
                        "fr": int(fr_l[q]) if int(ct[q]) >= 0 else None, "u0c": bool(is_content[u_l[q][0]])}
                del lg, lgf
            backend.release(kind, J)
        log(f"  readouts at layer {l}")
    t_lens = time.time() - t
    positions, agree = [], {}
    for j, p in enumerate(pos_recs):
        m = R[j]["naive"]["59"]
        ct = p["ct"]
        rec = {"i": j, "pid": p["pid"], "cond": p["cond"], "draws": p["draws"], "k": p["k"], "pos": p["pos"],
               "read_tok": s_of(p["seq"][p["pos"]]), "next_id": p["next_id"], "next_str": s_of(p["next_id"]),
               "next_content": bool(is_content[p["next_id"]]), "cont_ids": p["cont_ids"],
               "cont_strs": [s_of(i) for i in p["cont_ids"]], "ct_id": ct[1] if ct else None,
               "ct_str": s_of(ct[1]) if ct else None, "ct_offset": ct[0] if ct else None,
               "ct_prev_str": s_of(p["ct_prev"]) if ct else None,
               "ct_collapse": bool(ct) and R2.word_key(s_of(ct[1])) in C.COLLAPSE_TARGETS,
               "ct_formatlike": bool(ct) and R2.formatlike_target(s_of(ct[1]), s_of(p["ct_prev"])),
               "model_top1_id": m["u"][0], "model_top1_str": m["us"][0], "model_agrees": m["u"][0] == p["next_id"],
               "r": R[j]}
        positions.append(rec)
        agree.setdefault(p["cond"], []).append(rec["model_agrees"])
    prompts_out = []
    for it in items:
        d = {k: it[k] for k in ("pid", "cond", "user", "begun", "text", "kind", "format_likely", "about_model", "topic")
             if k in it}
        d.update(n_ids=len(it["ids"]), read_tok=s_of(it["ids"][-1]))
        if it["cond"] == "mid-reply":
            d.update(reply_ids=it["reply_ids"], reply_text=tok.decode(it["reply_ids"], skip_special_tokens=False),
                     reply_stopped=it["reply_stopped"], draws=it["draws"])
        else:
            d.update(cont_ids=conts[it["pid"]], cont_text=tok.decode(conts[it["pid"]], skip_special_tokens=False))
        prompts_out.append(d)
    summary = {
        "n_positions": {c: sum(p["cond"] == c for p in positions) for c in C.CONDITIONS},
        "n_mid_any": sum(1 for p in positions if p["draws"] and "any" in p["draws"]),
        "n_with_content_target": {c: sum(p["cond"] == c and p["ct_id"] is not None for p in positions)
                                  for c in C.CONDITIONS},
        "n_collapse_target": {c: sum(p["cond"] == c and p["ct_collapse"] for p in positions) for c in C.CONDITIONS},
        "model_agrees_with_next": {c: round(sum(v) / len(v), 4) for c, v in agree.items()},
    }
    out = {"schema": "R3_05/1", "meta": dict(meta_extra, **{
        "written_at": datetime.datetime.now().isoformat(timespec="seconds"), "host": platform.node(),
        "argv": sys.argv, "backend": backend.info(),
        "constants": {"READ_LAYERS": list(C.READ_LAYERS), "LENS_LAYERS": {k: list(v) for k, v in LENS_LAYERS.items()},
                      "KINDS": list(kinds), "BANDS": [[n, list(ls)] for n, ls in C.BANDS],
                      "CONT_WINDOW": C.CONT_WINDOW, "MID_NEW_TOKENS": C.MID_NEW_TOKENS,
                      "MID_DRAW_SIZES": C.MID_DRAW_SIZES, "DRAW_SEED": seed, "STOP_IDS": list(R2.STOP_IDS),
                      "TOPK": C.TOPK, "COLLAPSE_TARGETS": list(C.COLLAPSE_TARGETS), "batch": batch, "limit": limit},
        "timing_s": {"generation": round(t_gen, 1), "reading": round(t_read, 1), "readouts": round(t_lens, 1)},
        "summary": summary}), "prompts": prompts_out, "positions": positions}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_json.with_suffix(out_json.suffix + ".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":"), default=str), encoding="utf-8")
    os.replace(tmp, out_json)
    log(f"wrote {out_json} ({out_json.stat().st_size / 1e6:.1f} MB)")
    if out_states is not None:
        torch.save({"H": H, "layers": list(C.READ_LAYERS),
                    "positions": [[p["pid"], p["pos"], p["k"]] for p in positions]}, out_states)
    return out


# ------------------------------------------------------------------------------------------------ recall, pullbacks
def recall(backend, man: Dict[str, Dict[str, Any]],
           kinds=("jpp", "naive", "jpp-it", "rl-it", "miru", "jpp-it-full-macro"),
           residuals_pt: Path = C.RESULTS / "R3_04_residuals.pt", weights_json: Path = C.WEIGHTS_JSON) -> Dict[str, Any]:
    """J++'s own held-out recall@10 (its readout eval) at code 30/38/45/52, on R3_04's held-out residuals, for the
    primary macro's evals and the full macro's (as R3_04's floor kept them)."""
    import R3_fitlib as FL

    rs = torch.load(residuals_pt, map_location="cpu", weights_only=False)
    floors = json.loads(weights_json.read_text(encoding="utf-8"))["floors"]
    res = FL.Residuals(rs["eval_slugs"], rs["item_names"], rs["residuals"], rs["candidates"], C.MODEL_ID,
                       C.READOUT_MAX_SEQ_LEN)
    held = set(rs["heldout"])
    out: Dict[str, Any] = {"layers": list(C.JPP_RECALL_LAYERS), "k": 10, "macros": {}}
    for macro in ("primary", "full"):
        idx = [i for i, (n, s) in enumerate(zip(res.item_names, res.eval_slugs)) if n in held
               and s in floors[macro]["kept"]]
        out["macros"][macro] = {"evals": floors[macro]["kept"], "n_items": len(idx), "lenses": {}}
        for kind in kinds:
            for l in C.JPP_RECALL_LAYERS:
                J = backend.lens_matrix(kind, l)
                mp = FL.macro_pass(backend, res, l, None if J is None else J.float(), idx, rs["excluded"])
                out["macros"][macro]["lenses"].setdefault(kind, {})[str(l)] = mp
                backend.release(kind, J)
    return out


def pullbacks(man: Dict[str, Dict[str, Any]], kinds=("jpp-it", "rl-it", "jpp-it-full-macro")) -> Dict[str, Any]:
    """d . J_l of R1's 128 directions through R3's lens files and their halves, fp32 from the bf16 files."""
    DD = torch.load(C.R1_DIRS_PATH, weights_only=False, map_location="cpu")
    dirs = DD["dirs"].float()
    out: Dict[str, Any] = {"names": DD["names"], "layers": list(C.FIT_LAYERS), "pb": {}}
    paths = {k: Path(man[k]["path"]) for k in kinds}
    for h in ("A", "B"):
        for k in ("jpp-it", "rl-it"):
            paths[f"{k}-{h}"] = C.LENS_FILES[f"{k}-{h}"]
    for k, p in paths.items():
        J = torch.load(p, map_location="cpu", mmap=True, weights_only=True)["J"]
        out["pb"][k] = torch.stack([dirs @ J[l].float() for l in C.FIT_LAYERS], 1)        # [K, L, D]
    return out


# ------------------------------------------------------------------------------------------------ checks
def jpp_mask(tok, vocab_rows: int) -> Tuple[torch.Tensor, int]:
    """bool [V]: True where J++'s exclusion list keeps the id (R3_fitlib.excluded_ids: no letter or digit, minus
    every candidate of the six evals' items)."""
    import R3_fitlib as FL

    excl = FL.excluded_ids(tok, vocab_rows, FL.items_from_raw(C.jpp_items(), C.JPP_EVALS))
    m = torch.ones(vocab_rows, dtype=torch.bool)
    m[torch.as_tensor(excl, dtype=torch.long)] = False
    return m, len(excl)


def checks_before_model(full_hash: bool = True, prompts_required: bool = True) -> Dict[str, Any]:
    out: Dict[str, Any] = {"failed": [], "pending": []}
    tok = R2.load_tokenizer()
    out["tokenizer_ids"] = len(tok)
    if C.PROMPTS_PATH.is_file():
        data = json.loads(C.PROMPTS_PATH.read_text(encoding="utf-8"))
        with mock.patch.dict(R2.N_PROMPTS, C.N_PROMPTS):
            val = R2.validate_prompts(data, tok)
        if not val["ok"]:
            out["failed"].append(f"R3_prompts.json invalid: {val['problems'][:3]}")
        chk = json.loads(C.CHECK_JSON.read_text(encoding="utf-8")) if C.CHECK_JSON.is_file() else {}
        if not chk.get("ok") or chk.get("prompts_sha256") != C.sha256_file(C.PROMPTS_PATH):
            out["failed"].append("R3_01_check.json does not pass this exact R3_prompts.json; rerun R3_01_check.py")
    else:
        (out["failed"] if prompts_required else out["pending"]).append(
            "R3_prompts.json is not written yet (the registration's prompts)")
    fr = R2.filter_report(tok)
    r2chk = json.loads((C.RESULTS / "R2_01_check.json").read_text(encoding="utf-8"))
    if not fr["ok"] or fr["counts"] != (r2chk.get("filter") or {}).get("counts"):
        out["failed"].append("R2's filter fails its spot checks or differs from R2_01_check.json's counts")
    out["r2_filter_counts"] = fr["counts"]
    vocab_rows = int(C.real_text_config()["vocab_size"])
    jm, n_excl = jpp_mask(tok, vocab_rows)
    r3chk = json.loads(C.CHECK_JSON.read_text(encoding="utf-8")) if C.CHECK_JSON.is_file() else {}
    want = ((r3chk.get("filters") or {}).get("jpp") or {}).get("excluded")
    out["jpp_excluded"] = n_excl
    if want != n_excl:
        out["failed"].append(f"J++'s exclusion list has {n_excl} ids, R3_01_check.json recorded {want}")
    man = lens_manifest()
    lf = check_lens_files(man, ("jpp", "jlens", "miru", "r1", "jpp-it", "rl-it", "jpp-it-full-macro"), full_hash)
    out["lens_files"] = lf
    for k in ("jpp", "jlens", "miru", "r1"):
        r = lf[k]
        if not r["exists"] or r.get("size_ok") is False or r.get("sha256_ok") is False:
            out["failed"].append(f"{k}: not the pinned file ({r})")
    for k in ("jpp-it", "rl-it", "jpp-it-full-macro"):
        r = lf[k]
        if not r["exists"] or not r["pinned"]:
            out["pending"].append(f"{k}: no lens file recorded by R3_04_weights.json yet")
        elif r.get("sha256_ok") is False:
            out["failed"].append(f"{k}: not the file R3_04_weights.json records")
    seqs = r1_10_sequences(tok)
    meta, sha = positions_r1_10(seqs)
    out["r1_10_positions"] = {"n": len(meta), "sha256": sha, "sequences": len(seqs),
                              "longest_ids": max(len(s["ids"]) for s in seqs)}
    if sha != C.R1_10_POSITIONS_SHA256 or len(meta) != C.R1_10_N_POSITIONS:
        out["failed"].append("R1_10's held-out positions do not rebuild with their pinned sha256")
    if C.R1_10_PATH.is_file():
        r1 = torch.load(C.R1_10_PATH, weights_only=False, map_location="cpu")
        if [[m["name"], m["cls"], m["pos"]] for m in r1["meta"]] != [[m["name"], m["cls"], m["pos"]] for m in meta]:
            out["failed"].append("the rebuilt positions differ from R1_10_evalfull.pt's own list")
    import R3_gpu as G

    out["pilot"] = G.pilot_decision()
    if out["pilot"]["decision"] != "proceed":
        out["pending"].append(f"R3_00's pilot: {out['pilot']['why']}")
    for p in (C.EVAL_JSON, R2SET_JSON, H4_JSON):
        if p.exists():
            out["pending"].append(f"{p.name} exists: a run would need --force")
    out["_tok"], out["_jmask"] = tok, jm
    return out


# ------------------------------------------------------------------------------------------------ self-test
def selftest(out_path: Path, limit: int, record: bool = False) -> int:
    import R3_06_analyze as A
    import R3_fitlib as FL

    fails: List[str] = []
    done: List[Tuple[bool, str]] = []
    t_start = time.time()

    def check(ok: bool, what: str) -> None:
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        done.append((bool(ok), what))
        if not ok:
            fails.append(what)

    if out_path.resolve().parent == C.RESULTS.resolve():
        raise SystemExit("--selftest must write outside experiments/results (--out)")
    co = checks_before_model(full_hash=False, prompts_required=False)
    tok, jm = co.pop("_tok"), co.pop("_jmask")
    check(not co["failed"], f"every check before the model passes on the CPU (sizes only here); waiting for: "
                            f"{co['pending']}; failed: {co['failed']}")
    rng = np.random.default_rng(5)
    is_c = rng.random(len(tok)) < 0.9
    same = all(draw_mid_positions(r, is_c, i, R2.DRAW_SEED) == R2.draw_mid_positions(r, is_c, i)
               for i, r in enumerate([list(rng.integers(0, len(tok), size=n)) for n in (0, 2, 9, 48, 30)]))
    check(same, "the mid-reply draws with R2's seed are R2_common.draw_mid_positions exactly (R3 changes the seed)")
    cls, _ = R2.classify_vocab(tok)
    mask = torch.from_numpy(cls == 0)
    special = R2.special_mask(cls)
    backend = FakeBackend(tok)
    data = json.loads((C.PROMPTS_PATH if C.PROMPTS_PATH.is_file() else R2.PROMPTS_PATH).read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        st_path = Path(tmp) / "states.pt"
        out = run_set(backend, tok, data, mask, jm, special, seed=C.DRAW_SEED, batch=4, limit=limit, out_json=out_path,
                      out_states=st_path, meta_extra={"selftest": True})
        pos = out["positions"]
        ok = bool(pos)
        for p in pos:
            for k in KINDS:
                for l in LENS_LAYERS[k]:
                    r = p["r"][k][str(l)]
                    ok &= all(key in r for key in ("u", "us", "f", "fs", "fj", "fjs", "ur", "fr", "u0c"))
                    ok &= all(bool(mask[i]) for i in r["f"]) and all(bool(jm[i]) for i in r["fj"])
        check(ok, f"{len(pos)} read positions: every lens at each of its layers has top-5 unfiltered, under R2's "
                  f"filter (content ids only) and under J++'s exclusion list (kept ids only), and the ranks")
        t = A.Table(A.R3_DESIGN, out)
        v = t.values("f_word", "jpp-it", 30)
        _t, _b, bands, decision, ref = A.decision_suite(A.R3_DESIGN, out)
        check(len(v) == len(pos) and set(bands) == set(C.CANDIDATES) and "30-39" in bands["jpp-it"]
              and set(ref) == set(C.REFERENCE_LENSES + C.SENSITIVITY_LENSES) and isinstance(decision, dict),
              "R3_06's table reads the output under R3's design, and its decision suite runs on it (both candidates "
              "at every band from 30-39, the reference lenses beside them)")
        check(torch.load(st_path, weights_only=True)["H"].shape[1] == len(C.READ_LAYERS),
              "the captured states (layers 30-59) are saved for recomputation on the CPU")
    # H4 on the fake backend: the positions are R1_10's, the naked eye at 59 is the final distribution itself
    hb = h4(backend, tok, kinds=("naive", "jpp"))
    n59 = r1_10_metrics(torch.randn(6, 32), *final_reference(torch.randn(6, 32), backend.fnw, backend.wu),
                        backend.fnw, backend.wu)
    x = torch.randn(6, 32)
    lpf, t10f = final_reference(x, backend.fnw, backend.wu)
    m_same = r1_10_metrics(x, lpf, t10f, backend.fnw, backend.wu)
    check(hb["positions"] == C.R1_10_N_POSITIONS and hb["positions_sha256"] == C.R1_10_POSITIONS_SHA256
          and float(m_same["kl"][:, C.H4_TAUS.index(1.0)].abs().max()) <= 1e-5 and bool(m_same["top1"].all())
          and float(m_same["top10"].min()) == 1.0 and float(n59["kl"].min()) >= -1e-6,
          f"H4: R1_10's {hb['positions']} positions (pinned sha256); a lens equal to the final state has KL 0 at "
          f"scale 1 and full top-1/top-10 agreement; KL is never negative")
    ref = F.kl_div(F.log_softmax(torch.tanh((((x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))
                                               @ backend.wu.T) * 0.5) / 30) * 30, -1), lpf, log_target=True,
                   reduction="none").sum(-1)
    check(float((m_same["kl"][:, 0] - (r1_10_metrics(x * 1.0, lpf, t10f, backend.fnw, backend.wu)["kl"][:, 0])).abs()
                .max()) == 0.0 and float((r1_10_metrics(x, lpf, t10f, backend.fnw, backend.wu, taus=(0.5,))["kl"][:, 0]
                                          - ref).abs().max()) <= 1e-4,
          "H4's KL at a logit scale equals torch's kl_div of the scaled, softcapped logits")
    check(hb["ok"] is False and "caution" in hb and "jpp" not in hb["vs_jpp"] and "naive" in hb["vs_jpp"],
          "H4 on the fake model fails the R1_10 reproduction, and the output says so (ok false, with the caution)")
    # H4's tables on made-up per-position values with known answers
    g4 = np.random.default_rng(3)
    P4 = 400
    nm4 = np.array([f"s{i % 20:02d}" for i in range(P4)])
    cl4 = np.array(["prose" if i % 20 < 10 else "asst" for i in range(P4)])
    kl4 = g4.random((P4, len(C.H4_TAUS))) + np.linspace(1, 0, len(C.H4_TAUS))
    base = {"kl": kl4, "top1": g4.random(P4) < 0.3, "top10": g4.random(P4)}
    better = {"kl": kl4 - 0.1, "top1": np.ones(P4, bool), "top10": base["top10"]}
    tb = h4_tables(nm4, cl4, {"jpp": {40: base}, "jpp-it": {40: better}, "rl-it": {40: dict(base)}}, n_boot=300)
    seqs = sorted(set(nm4))
    par = np.array([seqs.index(x) % 2 for x in nm4])
    want = 0.0
    for a in (0, 1):
        bst = kl4[par == a].mean(0).argmin()
        want += kl4[par != a][:, bst].sum()
    want /= P4
    v = tb["vs_jpp"]
    check(abs(tb["lenses"]["jpp"]["40"]["all"]["kl_calibrated_crossfit"] - round(want, 4)) < 1e-9
          and v["rl-it"]["40"]["all"]["top1"] == {"diff": 0.0, "lo": 0.0, "hi": 0.0}
          and v["jpp-it"]["40"]["all"]["top1"]["lo"] > 0 and abs(v["jpp-it"]["40"]["all"]["kl_tau1"]["diff"] + 0.1) < 1e-4
          and tb["h4_reading"] == {"jpp-it": {"40": "supported"}, "rl-it": {"40": "refuted"}}
          and set(v["jpp-it"]["40"]) == {"all", "asst", "prose"},
          "H4's tables: the cross-fitted calibrated KL is R1_14's; a lens equal to J++ differs by exactly 0 in every "
          "resample; a lens right everywhere has an interval above 0 and reads 'supported'; one equal to J++ "
          "'refuted'; per class too")
    # recall: R3_fitlib.macro_pass against a brute-force rank on made-up residuals of real J++ items
    items = FL.scoreable(FL.items_from_raw({s: v[:5] for s, v in C.jpp_items().items()}, C.JPP_MACRO_EVALS), tok)
    g = torch.Generator().manual_seed(9)
    resid = {30: torch.randn(len(items), 32, generator=g)}
    cands = [FL.item_candidates(tok, it) for it in items]
    for i in range(0, len(items), 2):              # every other item points at one of its candidates: some hits
        resid[30][i] = backend.wu[next(iter(cands[i].values()))[0]] * 3.0
    res = FL.Residuals([it.eval_slug for it in items], [it.name for it in items], resid, cands, "fake", 512)
    excl = (~jm).nonzero().flatten().tolist()
    mp = FL.macro_pass(backend, res, 30, None, list(range(len(items))), excl)
    lg = backend.unembed(resid[30])
    lg[:, torch.as_tensor(excl)] = float("-inf")
    per: Dict[str, List[float]] = {}
    for i, c in enumerate(cands):
        for ids in c.values():
            best = float(lg[i, ids].max())
            per.setdefault(items[i].eval_slug, []).append(float(int((lg[i] > best).sum()) + 1 <= 10))
    brute = float(np.mean([np.mean(v) for v in per.values()]))
    check(abs(mp["macro"] - brute) <= 1e-6 and 0 < brute < 1, f"held-out recall@10 (J++'s pair-level macro) equals a brute-force rank "
                                            f"count ({mp['macro']:.4f}, {mp['n_pairs']} pairs)")
    print(f"R3_05_eval self-test: {'ALL PASS' if not fails else str(len(fails)) + ' FAILURES'}")
    if record:
        C.record_selftest(Path(__file__).name, done, time.time() - t_start)
    return 0 if not fails else 1


# ------------------------------------------------------------------------------------------------ the run
def run(args) -> int:
    import R3_gpu as G

    co = checks_before_model(full_hash=True, prompts_required=True)
    tok, jm = co.pop("_tok"), co.pop("_jmask")
    if co["failed"] or [p for p in co["pending"] if "exists" not in p or not args.force]:
        raise SystemExit(f"checks before the model: failed {co['failed']}; waiting for {co['pending']}")
    G.require_pilot_proceed()
    man = lens_manifest()
    cls, _ = R2.classify_vocab(tok)
    mask = torch.from_numpy(cls == 0)
    special = R2.special_mask(cls)
    gst = G.require_committed(args.allow_uncommitted, True, log)
    meta_extra = {"git": gst, "lenses": {k: {"path": str(v["path"]), "sha256": v["sha256"]} for k, v in man.items()},
                  "checks_before_model": {k: v for k, v in co.items()},
                  "mask_sha256": hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
                  "jpp_mask_sha256": hashlib.sha256(jm.numpy().tobytes()).hexdigest(),
                  "prompts_sha256": C.sha256_file(C.PROMPTS_PATH)}
    G.log_run({"event": "started", "argv": sys.argv, "git": gst, "forced": bool(args.force)}, C.RUNS_LOG)
    try:
        G.set_job_cap(torch.device("cuda:0"))
        backend = GpuBackend(man)
        with torch.no_grad():
            run_set(backend, tok, json.loads(C.PROMPTS_PATH.read_text(encoding="utf-8")), mask, jm, special,
                    seed=C.DRAW_SEED, batch=args.batch, limit=None, out_json=C.EVAL_JSON, out_states=STATES_PT,
                    meta_extra=meta_extra)
            run_set(backend, tok, R2.load_prompts(), mask, jm, special, seed=R2.DRAW_SEED, batch=args.batch,
                    limit=None, out_json=R2SET_JSON, out_states=None,
                    meta_extra=dict(meta_extra, replication_of="R2_prompts.json (R2's positions: R2's draw seed)",
                                    prompts_sha256=R2.sha256_file(R2.PROMPTS_PATH)))
            h4_out = h4(backend, tok, positions_pt=H4_POSITIONS_PT)
            H4_JSON.write_text(json.dumps(h4_out, indent=1) + "\n", encoding="utf-8")
            if not h4_out["ok"]:
                G.log_run({"event": "failed check", "check": "H4: J++, JLens and the naked eye reproduce R1_10's "
                                                              "top-1 within 0.01 (D23)",
                           "detail": h4_out["reproduces_r1_10"], "consequence": h4_out["caution"]}, C.RUNS_LOG)
                log("WARNING: " + h4_out["caution"])
            RECALL_JSON.write_text(json.dumps(recall(backend, man), indent=1) + "\n", encoding="utf-8")
        torch.save(pullbacks(man), PULLBACKS_PT)
    except BaseException as e:
        G.log_run({"event": "failed", "error": f"{type(e).__name__}: {e}"}, C.RUNS_LOG)
        raise
    G.log_run({"event": "finished", "out": str(C.EVAL_JSON), "out_sha256": C.sha256_file(C.EVAL_JSON),
               "vram": G.vram_record(torch.device("cuda:0"))}, C.RUNS_LOG)
    log("done")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=None, help="--selftest: where to write the output (outside experiments/results)")
    ap.add_argument("--limit", type=int, default=0, help="--selftest: the first N prompts per condition")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--record", action="store_true", help="with --selftest: keep the result in R3_selftests.json")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow-uncommitted", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        if not a.out:
            raise SystemExit("--selftest needs --out (outside experiments/results)")
        return selftest(Path(a.out), a.limit or 3, a.record)
    if a.limit:
        raise SystemExit("--limit is for --selftest only: the registered run reads every prompt")
    if a.check_only:
        co = checks_before_model(full_hash=True, prompts_required=False)
        co.pop("_tok"), co.pop("_jmask")
        print(json.dumps(co, indent=1, default=str))
        print("check-only:", "FAILED" if co["failed"] else ("PASS, waiting for: " + "; ".join(co["pending"])
                                                            if co["pending"] else "PASS"))
        return 1 if co["failed"] else 0
    return run(a)


if __name__ == "__main__":
    sys.exit(main())

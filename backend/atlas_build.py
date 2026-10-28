"""Building the atlas's indexes (`backend/atlas_build.py`): what scripts/atlas_build.py runs on the GPU, one function per
index file (backend/atlas_index.py names them). Each reads the model through `atlas_ops.Weights` (the runtime's
accessors on the real model, so W4A16 dequantized on the fly and any vault edit in force included; a fresh process has
none) and writes its file atomically with its provenance under `meta`. scripts/check_atlas_index.py runs every one on the
CPU stand-in and compares what it wrote with the same quantity read live, one neuron at a time. Every file's meta names
the weights it was built from (`weights`, `runtime_factory.weights_card`: a GGUF's fingerprint and tensor types, or
the checkpoint's repo id and the weight files its snapshot links to), beside the model id, and the server reads it
only for the same (`atlas_index.weights_differ`).

What each costs on Gemma-4-31B is estimated in scripts/atlas_build.py's help from the arithmetic, not measured: the
lens index and the wiring index are dense matrix products over every layer (minutes), the activation extremes one
forward per window of the corpus, the steer lens a forward of every later layer for every neuron, both signs and four
contexts (hours: it can be built a range of layers at a time and resumes), the layout a projection per layer on the CPU.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from backend import atlas_ops as ao
from backend.atlas_index import FILES, LENS_K, PASSAGES_DIR, PASSAGES_K, WIRING_K, save_npz, weights_differ
from backend.causal_check import language_model, logits_fp32, model_device, pad_right
from backend.label_gate import label_top_ids, writes_nothing
from backend.theodolite_math import unembed_gemma4

Log = Callable[[str], None]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _weights_here(rt: Any) -> Dict[str, Any]:
    from backend.runtime_factory import weights_card  # the file, or the checkpoint's snapshot, the model was read from

    return weights_card(rt)


def _meta(rt: Any, **kw: Any) -> np.ndarray:
    from backend.w4a16_fast import describe  # how the 4-bit weights were unpacked (all unpackers give the same bits)

    card = rt.lens_card() if hasattr(rt, "lens_card") else None
    m = {"model": getattr(rt, "model_id", None), "built_at": _now(), "lens_id": (card or {}).get("id"),
         "stand_in": bool(getattr(rt, "atlas_stand_in", False)), "unpacker": describe(rt), "weights": _weights_here(rt), **kw}
    return np.array(json.dumps(m))


# ------------------------------------------------------------------------------------------------------------- lens


@torch.no_grad()
def build_lens(rt: Any, out_dir: Path, *, chunk: int = 512, k: int = LENS_K, layers: Optional[Sequence[int]] = None,
               log: Log = print) -> Path:
    """Every neuron's first k word-pieces, both faces, through the J++ lens and by the naked eye, ranked as the vault
    ranks them (`inspect_bipolar_slot`: the write column times the post-feed-forward norm's weight, through J_l, the
    unembedding with its softcap, then `label_top_ids`). The negative face's logits are the positive face's negated,
    exactly (the norm and the softcap are odd), so each column is read once per reading. Given `layers`, only those
    are read, into the file already there when it was built for this model, lens and shape (`built` marks the layers
    read so far; an unread layer has no words, rather than empty ones the gate would pass)."""
    lm = language_model(rt.hf_model)
    w = ao.Weights(lm, rt)
    L, N = w.n_layers, w.d_ff
    fnw, W_U = rt.final_norm_weight, rt.lm_head_weight
    keys = ("lens_pos", "lens_neg", "eye_pos", "eye_neg")
    arrays = {key: np.zeros((L, N, k), np.int32) for key in keys}
    silenced = np.zeros((L, N), bool)
    built = np.zeros(L, bool)
    path = Path(out_dir) / FILES["lens"]
    lens_id = ((rt.lens_card() if hasattr(rt, "lens_card") else None) or {}).get("id")
    if layers is not None and path.exists():
        with np.load(path, allow_pickle=False) as z:
            old = {key: z[key] for key in z.files}
        m = json.loads(str(old.get("meta", "{}")))
        if ("built" in old and m.get("model") == getattr(rt, "model_id", None) and m.get("lens_id") == lens_id
                and old["lens_pos"].shape == (L, N, k) and weights_differ(m.get("weights"), _weights_here(rt)) is None):
            arrays = {key: old[key] for key in keys}
            silenced, built = old["silenced"], old["built"].astype(bool)
            log(f"lens: adding to the file there ({int(built.sum())} of {L} layers read)")
    todo = list(range(L)) if layers is None else [int(l) for l in layers]
    t0 = time.time()
    for l in todo:
        J = rt.jpp_lenses[l]
        Wd = w.w_down(l)  # [D, N]
        w_post = w.norm(l, "post_feedforward").to(Wd.device)
        for s in range(0, N, int(chunk)):
            cols = (Wd[:, s : s + chunk].T.float() * w_post).to(J.dtype)  # [c, D], as the vault scales them
            silenced[l, s : s + cols.shape[0]] = writes_nothing(cols)
            for reading, vecs in (("lens", cols @ J.T), ("eye", cols)):
                logits = unembed_gemma4(vecs, fnw, W_U)
                arrays[f"{reading}_pos"][l, s : s + cols.shape[0]] = np.array(label_top_ids(vecs, fnw, W_U, logits, k=k))
                arrays[f"{reading}_neg"][l, s : s + cols.shape[0]] = np.array(label_top_ids(-vecs, fnw, W_U, -logits, k=k))
                del logits
        built[l] = True
        log(f"lens: layer {l} done ({time.time() - t0:.0f} s)")
    save_npz(path, **arrays, silenced=silenced, built=built, K=np.array(k),
             meta=_meta(rt, layers=[int(x) for x in np.nonzero(built)[0]]))
    return path


# ----------------------------------------------------------------------------------------------- activation extremes


def windows(rt: Any, texts: Sequence[str], seq: int, max_tokens: int) -> List[List[int]]:
    """Texts packed into windows of `seq` word-pieces, each begun with the start mark (so position 0, left out of every
    extreme, is the sink it is in every reading), documents separated by the end mark, up to `max_tokens`."""
    tok = rt.tokenizer
    bos, eos = tok.bos_token_id, tok.eos_token_id
    buf: List[int] = []
    out: List[List[int]] = []
    seen = 0
    for text in texts:
        buf += tok.encode(text, add_special_tokens=False) + ([eos] if eos is not None else [])
        while len(buf) >= seq - 1 and seen < max_tokens:
            out.append([bos] + buf[: seq - 1])
            buf = buf[seq - 1 :]
            seen += seq - 1
        if seen >= max_tokens:
            break
    if buf and seen < max_tokens:
        out.append([bos] + buf[: min(len(buf), seq - 1, max_tokens - seen)])
    return out


@torch.no_grad()
def build_stats(rt: Any, out_dir: Path, texts_ids: Sequence[Sequence[int]], corpus: str, *, log: Log = print) -> Path:
    """Every neuron's activation extremes over the texts (`atlas_ops.activation_extremes`: its largest and most negative
    activation and the four sign regimes, the start mark left out), the stream's length leaving each layer (10th,
    50th, 90th percentile), the unit of the add push, and the two feed-forward norms' usual scales (the same
    percentiles of the block output's rms and of the stream's rms entering the block), which weigh the wiring."""
    lm = language_model(rt.hf_model)
    t0 = time.time()
    ex = ao.activation_extremes(lm, texts_ids)
    qs = lambda vs: np.array([np.quantile(np.asarray(v, np.float32), [0.1, 0.5, 0.9]) if len(v) else [np.nan] * 3 for v in vs], np.float32)  # noqa: E731
    log(f"stats: {ex.tokens} word-pieces in {len(texts_ids)} windows ({time.time() - t0:.0f} s)")
    path = Path(out_dir) / FILES["act_scale"]
    save_npz(path, pos=ex.pos.numpy(), neg=ex.neg.numpy(), pp=ex.pp.numpy(), nn=ex.nn.numpy(), pn=ex.pn.numpy(),
             np=ex.np.numpy(), tokens=np.array(ex.tokens), resid_q=qs(ex.resid_norms), mlp_rms_q=qs(ex.mlp_rms),
             mid_rms_q=qs(ex.mid_rms), meta=_meta(rt, corpus=corpus, tokens=ex.tokens, windows=len(texts_ids)))
    return path


# ---------------------------------------------------------------------------------------------------------- passages


def _sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@torch.no_grad()
def build_passages(rt: Any, out_dir: Path, texts_ids: Sequence[Sequence[int]], corpus: str, *, k: int = PASSAGES_K,
                   sample: int = 0, extremes: Optional[Tuple[np.ndarray, np.ndarray]] = None, extremes_source: Optional[str] = None,
                   seed: int = 0, chunk: int = 1024, keep_parts: bool = False, log: Log = print) -> Path:
    """Every neuron's k strongest passages on each face over the windows (`atlas_ops.passage_events`: at most one a
    window, ties to the lower window, with the gate and up values at each), and, given `sample` and `extremes` (the
    stats index's), `sample` more a face drawn at random above half its extreme; the windows' word-piece ids, and each
    word-piece's count over them (the base rate the "reads" and "follows" lifts are taken against).

    A part per `chunk` windows is written to `passages.d/parts/part-NNNNN.npz` as it is read, so a stopped run resumes
    at the first missing part (the parts' plan.json holds the settings and the windows' sha256: a run with others is
    refused). The parts are merged in window order (`atlas_ops.merge_events`, exact whatever the chunk), into one `.npy`
    per array in `passages.d/` (backend/atlas_index.py lists them), each written beside its place and renamed into it, the
    old meta.json removed first and the new one written last, so a reader sees the whole index or none; the server
    memory-maps them. The parts are removed after the merge unless `keep_parts`. The windows hold the corpus's own text:
    it stays in the atlas folder (git ignores it) and is never committed."""
    lm = language_model(rt.hf_model)
    L, N = len(lm.layers), ao.ffn_width(lm, 0)
    V = int(rt.lm_head_weight.shape[0])
    folder = Path(out_dir) / PASSAGES_DIR
    parts = folder / "parts"
    W = len(texts_ids)
    seq = max((len(t) for t in texts_ids), default=0)
    if not W or seq < 2:
        raise ValueError("No windows to read passages from.")
    if seq > 32767:
        raise ValueError(f"Windows of {seq} word-pieces are longer than a position kept as int16 can name (32,767).")
    wins = np.full((W, seq), -1, np.int32)
    for i, t in enumerate(texts_ids):
        wins[i, : len(t)] = np.asarray(t, np.int32)
    sample = int(sample) if extremes is not None else 0
    plan = {"model": getattr(rt, "model_id", None), "k": int(k), "sample": sample, "seed": int(seed), "chunk": int(chunk),
            "windows": W, "seq": seq, "windows_sha256": _sha(wins),
            "extremes_sha256": _sha(np.stack([np.asarray(x, np.float32) for x in extremes])) if sample else None,
            "weights": _weights_here(rt)}
    plan_path = parts / "plan.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text(encoding="utf-8"))
        # the weights compared by weights_differ (parts begun before plans named them, by the model id alone)
        differ = [key for key in plan if key != "weights" and old.get(key) != plan[key]]
        differ += ["weights"] if weights_differ(old.get("weights"), plan["weights"]) else []
        if differ:
            raise ValueError(f"The parts in {parts} were begun with other settings ({', '.join(differ)}): build with the same "
                             "ones to resume, or remove that folder to start again.")
    else:
        parts.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps({**plan, "corpus": corpus}, indent=1), encoding="utf-8")
    n_parts = -(-W // int(chunk))
    ext = (np.asarray(extremes[0], np.float32), np.asarray(extremes[1], np.float32)) if sample else None
    t0 = time.time()
    for i in range(n_parts):
        part = parts / f"part-{i:05d}.npz"
        if part.exists():
            continue
        lo, hi = i * int(chunk), min(W, (i + 1) * int(chunk))
        save_npz(part, **ao.passage_events(lm, texts_ids[lo:hi], k=k, first=lo, sample=sample, extremes=ext, seed=seed))
        log(f"passages: part {i + 1} of {n_parts}, windows {lo} to {hi - 1} ({time.time() - t0:.0f} s)")
    merged: Optional[Dict[str, np.ndarray]] = None
    for i in range(n_parts):
        with np.load(parts / f"part-{i:05d}.npz") as z:
            got = {key: z[key] for key in z.files}
        merged = got if merged is None else ao.merge_events(merged, got, k, sample)
    assert merged is not None
    read = wins[:, 1:][wins[:, 1:] >= 0]  # every word-piece read: all but each window's start mark
    arrays: Dict[str, np.ndarray] = {"windows": wins, "unigram": np.bincount(read, minlength=V)[:V].astype(np.int32)}
    for f, sgn in (("pos", 1.0), ("neg", -1.0)):
        empty = merged[f"{f}_win"] < 0
        arrays[f"{f}_win"], arrays[f"{f}_at"] = merged[f"{f}_win"], merged[f"{f}_at"]
        arrays[f"{f}_act"] = np.where(empty, 0.0, sgn * merged[f"{f}_key"]).astype(np.float16)
        arrays[f"{f}_g"], arrays[f"{f}_u"] = merged[f"{f}_g"], merged[f"{f}_u"]
        if sample:
            for name in ("win", "at", "act", "g", "u"):
                arrays[f"{f}_smp_{name}"] = merged[f"{f}_smp_{name}"]
    tokens = int(read.size)
    meta = json.loads(str(_meta(rt, corpus=corpus, tokens=tokens, windows=W, seq=seq, K=int(k), per_window=1, seed=int(seed),
                                sample=sample, sample_threshold=("half of each face's extreme in " + str(extremes_source)) if sample else None,
                                parts=n_parts, text="the windows hold the corpus's own text: keep it in this folder, never commit it")))
    (folder / "meta.json").unlink(missing_ok=True)
    for old in folder.glob("*.npy"):
        if old.stem not in arrays:
            old.unlink()
    for key, arr in arrays.items():
        tmp = folder / f"{key}.npy.tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, np.ascontiguousarray(arr))
        os.replace(tmp, folder / f"{key}.npy")
    tmp = folder / "meta.json.tmp"
    tmp.write_text(json.dumps(meta), encoding="utf-8")
    os.replace(tmp, folder / "meta.json")
    log(f"passages: {tokens} word-pieces in {W} windows, {n_parts} parts merged ({time.time() - t0:.0f} s)")
    if not keep_parts:
        shutil.rmtree(parts, ignore_errors=True)
    return folder


# ------------------------------------------------------------------------------------------------------------ wiring


@torch.no_grad()
def build_wiring(rt: Any, out_dir: Path, scale: torch.Tensor, *, scale_source: str = "", k: int = WIRING_K, chunk: int = 4096,
                 log: Log = print) -> Path:
    """Every neuron's k strongest weight links each way (`atlas_ops.neuron_wiring` for all of them at once, as the
    reference's build_connections reads one product per pair of layers): for layers i < j,
    P[B, A] = (W_gate_j[B] . v_A)(W_up_j[B] . v_A), v_A = w_pre_j (.) w_post_i (.) W_down_i[:, A]. A's links into layer
    j are column A of P, |P| z-scored over B; B's links from layer i are row B, |P| z-scored over A (kept as a running
    top k of |P| with the row's sums, which ranks the row as its z does, then turned into z). Across layer pairs the k
    links are chosen and ordered by |P| f[i, j]^2 (`atlas_ops.link_scale`, `scale`: the norms at their usual scale and the
    layer scalars between), kept as log10 of that over the neuron's strongest (`lw`), with each link's gate sign, and its
    g and u themselves (float16, held within its finite range, signs kept), which the sign in use is read
    from when served (`atlas_ops.LINK_MODEL_USE`): 4 more arrays of [L, N, K], 248 MB uncompressed on the 31B, so a
    new passages index needs no new wiring. Columns go `chunk` at a time, so no [N, N] product is held whole (the up
    products are kept beside P until the chunk's links are gathered: one more [N, chunk] float32)."""
    lm = language_model(rt.hf_model)
    w = ao.Weights(lm, rt)
    L, N, dev = w.n_layers, w.d_ff, w.device
    K = int(k)
    lf = torch.log(scale.to(dev).clamp(min=1e-300)).float()  # log f[i, j]
    ninf = float("-inf")
    be = {s: torch.full((L, N, K), ninf, device=dev) for s in ("down", "up")}  # log strength: the merge key
    bz = {s: torch.full((L, N, K), ninf, device=dev) for s in ("down", "up")}
    bl = {s: torch.zeros((L, N, K), dtype=torch.int16, device=dev) for s in ("down", "up")}
    bn = {s: torch.zeros((L, N, K), dtype=torch.int32, device=dev) for s in ("down", "up")}
    bs = {s: torch.zeros((L, N, K), dtype=torch.int8, device=dev) for s in ("down", "up")}
    bg = {s: torch.zeros((L, N, K), dtype=torch.int8, device=dev) for s in ("down", "up")}
    bgv = {s: torch.zeros((L, N, K), dtype=torch.float16, device=dev) for s in ("down", "up")}  # each link's g
    buv = {s: torch.zeros((L, N, K), dtype=torch.float16, device=dev) for s in ("down", "up")}  # and its u

    def merge(side: str, layer: int, rows: slice, le: torch.Tensor, z: torch.Tensor, idx: torch.Tensor, other: int, sign: torch.Tensor,
              gate: torch.Tensor, gv: torch.Tensor, uv: torch.Tensor) -> None:
        ce = torch.cat([be[side][layer, rows], le], 1)
        top = ce.topk(K, dim=1)
        be[side][layer, rows] = top.values
        for store, new in ((bz, z), (bl, torch.full_like(idx, other, dtype=torch.int16)), (bn, idx.to(torch.int32)), (bs, sign), (bg, gate),
                           (bgv, h16(gv)), (buv, h16(uv))):
            store[side][layer, rows] = torch.cat([store[side][layer, rows], new], 1).gather(1, top.indices)

    sgn = lambda x: torch.where(x > 0, 1, -1).to(torch.int8)  # noqa: E731
    h16 = lambda x: x.clamp(-65504.0, 65504.0).to(torch.float16)  # noqa: E731  float16's finite range, signs kept
    t0 = time.time()
    for j in range(1, L):
        Gj, Uj = w.w_gate(j), w.w_up(j)
        w_pre_j = w.norm(j, "pre_feedforward").to(dev)
        for i in range(j):
            Wd = w.w_down(i)
            two_lf = 2.0 * float(lf[i, j])
            norms = (w.norm(i, "post_feedforward").to(dev) * w_pre_j)[:, None]
            r_sum = torch.zeros(N, dtype=torch.float64, device=dev)
            r_sq = torch.zeros(N, dtype=torch.float64, device=dev)
            r_top = torch.full((N, K), ninf, device=dev)
            r_idx = torch.zeros((N, K), dtype=torch.long, device=dev)
            r_sgn = torch.zeros((N, K), dtype=torch.int8, device=dev)
            r_gate = torch.zeros((N, K), dtype=torch.int8, device=dev)
            r_g = torch.zeros((N, K), dtype=torch.float32, device=dev)
            r_u = torch.zeros((N, K), dtype=torch.float32, device=dev)
            for a0 in range(0, N, int(chunk)):
                V = (Wd[:, a0 : a0 + chunk].float() * norms).to(Gj.dtype)  # [D, c]
                G = (Gj @ V).float()
                U = (Uj @ V.to(Uj.dtype)).float()
                P = G * U  # [N_j, c]
                M = P.abs()
                c = M.shape[1]
                # downstream of each A in this chunk: its column over every B of layer j
                zc = (M - M.mean(0, keepdim=True)) / (M.std(0, keepdim=True) + 1e-12)
                kk = min(K, zc.shape[0])
                top = zc.topk(kk, dim=0)
                le = torch.log(M.gather(0, top.indices)) + two_lf  # -inf where the column writes nothing there
                merge("down", i, slice(a0, a0 + c), le.T, top.values.T, top.indices.T, j, sgn(P.gather(0, top.indices)).T,
                      sgn(G.gather(0, top.indices)).T, G.gather(0, top.indices).T, U.gather(0, top.indices).T)
                # upstream of each B: its row over every A of layer i, gathered across chunks
                r_sum += M.sum(1, dtype=torch.float64)
                r_sq += (M.double() ** 2).sum(1)
                kk = min(K, c)
                tv = M.topk(kk, dim=1)
                cand = torch.cat([r_top, tv.values], 1)
                keep = cand.topk(K, dim=1)
                r_idx = torch.cat([r_idx, tv.indices + a0], 1).gather(1, keep.indices)
                r_sgn = torch.cat([r_sgn, sgn(P.gather(1, tv.indices))], 1).gather(1, keep.indices)
                r_gate = torch.cat([r_gate, sgn(G.gather(1, tv.indices))], 1).gather(1, keep.indices)
                r_g = torch.cat([r_g, G.gather(1, tv.indices)], 1).gather(1, keep.indices)
                r_u = torch.cat([r_u, U.gather(1, tv.indices)], 1).gather(1, keep.indices)
                r_top = keep.values
                del V, G, U, P, M, zc
            mean = r_sum / N
            std = ((r_sq - N * mean * mean) / max(N - 1, 1)).clamp(min=0).sqrt()
            z = ((r_top.double() - mean[:, None]) / (std[:, None] + 1e-12)).float()
            z = torch.where(torch.isfinite(r_top), z, torch.full_like(z, ninf))
            merge("up", j, slice(0, N), torch.log(r_top) + two_lf, z, r_idx, i, r_sgn, r_gate, r_g, r_u)
        log(f"wiring: layer {j} read against every layer before it ({time.time() - t0:.0f} s)")
    arrays: Dict[str, np.ndarray] = {}
    for side in ("down", "up"):
        le = be[side]
        empty = ~torch.isfinite(le)
        # log10 of each link's strength over the neuron's strongest on that side (slot 0: the topk keeps them in order)
        lw = torch.where(empty, torch.full_like(le, ninf), (le - le[..., :1]) / float(np.log(10.0)))
        arrays[f"{side}_z"] = torch.where(empty, torch.full_like(le, -60000.0), bz[side].clamp(min=-60000)).cpu().numpy().astype(np.float16)
        arrays[f"{side}_lw"] = lw.cpu().numpy().astype(np.float16)
        arrays[f"{side}_layer"] = bl[side].cpu().numpy().astype(np.int8)
        arrays[f"{side}_n"] = bn[side].cpu().numpy().astype(np.int16 if N <= 32767 else np.int32)
        arrays[f"{side}_sign"] = bs[side].cpu().numpy()
        arrays[f"{side}_gate"] = bg[side].cpu().numpy()
        arrays[f"{side}_g"] = bgv[side].cpu().numpy()
        arrays[f"{side}_u"] = buv[side].cpu().numpy()
    path = Path(out_dir) / FILES["wiring"]
    save_npz(path, compressed=True, **arrays, K=np.array(K),
             meta=_meta(rt, weighed="across layers: |P| f^2, the norms at their usual scale and the layer scalars between",
                        scale_source=scale_source, link_model=ao.LINK_MODEL, link_model_use=ao.LINK_MODEL_USE))
    return path


# ------------------------------------------------------------------------------------------------------- steer lens


def steer_contexts(rt: Any) -> List[List[int]]:
    """The steer lens's four contexts as the model reads them: the two chats through the chat template (thinking off),
    the two plain openings as bare text after the start mark."""
    out = [rt.tokenize_prompt(rt.compose_prompt(c, chat=True)) for c in ao.STEER_LENS_CHATS]
    out += [rt.tokenize_prompt(p) for p in ao.STEER_LENS_PLAIN]
    return out


@torch.no_grad()
def build_steer_layer(rt: Any, layer: int, p50: float, *, dose: float = ao.STEER_LENS_DOSE, k: int = 12,
                      batch: int = 64, spare_bos: bool = True) -> Dict[str, np.ndarray]:
    """One layer of the steer lens (`atlas_ops.steer_lens` for every neuron of it): the four contexts are read to the
    stream leaving the layer once; then each batch of neurons pushes it, both signs, and only the later layers run
    (`forward_from_layer`), the unpushed rows of the batch in the same forward. Returns up, down [N, k] int32 and
    up_kl, down_kl [N] float32."""
    lm = language_model(rt.hf_model)
    w = ao.Weights(lm, rt)
    N, dev, l = w.d_ff, model_device(lm), int(layer)
    ctx = steer_contexts(rt)
    C = len(ctx)
    inp, att, last, _ = pad_right(ctx, dev)
    pos = (att.cumsum(-1) - 1).clamp(min=0)
    seen: Dict[str, torch.Tensor] = {}
    hook = lm.layers[l].register_forward_hook(lambda _m, _a, o: seen.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
    try:
        lm(input_ids=inp, attention_mask=att, use_cache=False)
    finally:
        hook.remove()
    h_l = seen["h"]  # [C, T, D]
    from backend import cpu_memory  # on the CPU, a batch's forward against MemAvailable, refused in plain words

    rows = C + 2 * min(int(batch), N) * C
    cpu_memory.ensure(rt, lambda: cpu_memory.forward_need(rt, rows, int(inp.shape[1])),
                      f"A steer lens batch of {batch} neurons", "lower --batch")
    keep = ao.keep_mask(inp, att, rt.tokenizer.bos_token_id, spare_bos).to(h_l.dtype)  # [C, T]
    W_U = rt.lm_head_weight
    up = np.zeros((N, k), np.int32)
    down = np.zeros((N, k), np.int32)
    up_kl = np.zeros(N, np.float32)
    down_kl = np.zeros(N, np.float32)
    for s in range(0, N, int(batch)):
        idx = list(range(s, min(s + int(batch), N)))
        B = len(idx)
        u = w.write_dirs(l, idx).to(h_l.dtype)  # [B, D]
        delta = float(dose) * float(p50) * u
        # rows: C unpushed, then for each sign, neuron-major, each neuron's C contexts
        pushed = [h_l]
        for sgn in (1.0, -1.0):
            d = (sgn * delta)[:, None, None, :] * keep[None, :, :, None]  # [B, C, T, D]
            pushed.append((h_l[None] + d).reshape(B * C, *h_l.shape[1:]))
        hs = torch.cat(pushed, 0)
        R = hs.shape[0]
        att_r = att.repeat(R // C, 1)
        pos_r = pos.repeat(R // C, 1)
        out = ao.forward_from_layer(lm, l + 1, hs, att_r, pos_r)
        lastr = last.repeat(R // C)
        hlast = out[torch.arange(R, device=dev), lastr]
        lsm = torch.log_softmax(logits_fp32(hlast, W_U), dim=-1)
        base = lsm[:C]
        for si, (arr, kl_arr) in enumerate(((up, up_kl), (down, down_kl))):
            ls = lsm[C + si * B * C : C + (si + 1) * B * C].view(B, C, -1)
            diff = (ls - base[None]).mean(1)  # [B, V]
            kl = (ls.exp() * (ls - base[None])).sum(-1).mean(1)
            arr[s : s + B] = diff.topk(int(k), dim=-1).indices.cpu().numpy()
            kl_arr[s : s + B] = kl.float().cpu().numpy()
        del hs, out, lsm
    return {"up": up, "down": down, "up_kl": up_kl, "down_kl": down_kl}


def _shard_refusal(rt: Any, shard: Path) -> Optional[str]:
    """Why a steer lens layer already built (`steer_lens.d/L{l}.npz`) was not built on this runtime's model and weights,
    in plain words; None when it was, or when it was built before the layers kept a meta (it is then kept, as before)."""
    with np.load(shard, allow_pickle=False) as z:
        m = json.loads(str(z["meta"])) if "meta" in z.files else None
    if m is None:
        return None
    here = getattr(rt, "model_id", None)
    other = weights_differ(m.get("weights"), _weights_here(rt))
    if m.get("model") != here:
        why = f"built for {m.get('model')}, not {here}"
    elif other:
        why = f"built from other weights ({other})"
    else:
        return None
    return (f"{shard} was {why}: build the steer lens for this model into another folder (--out), or remove "
            f"{shard.parent.name} from {shard.parent.parent} to build it again here")


def build_steer_lens(rt: Any, out_dir: Path, p50: Sequence[float], *, layers: Optional[Sequence[int]] = None,
                     dose: float = ao.STEER_LENS_DOSE, k: int = 12, batch: int = 64, log: Log = print) -> Path:
    """The steer lens of every neuron, a layer at a time into `steer_lens.d/L{l}.npz` beside the index (a run stopped
    part-way resumes at the first missing layer), merged into steer_lens.npz once every layer is there. Each layer's
    file names the model and weights it was built on (`meta`), and a run on another model or other weights is refused
    in plain words (`_shard_refusal`), not resumed into the same folder."""
    lm = language_model(rt.hf_model)
    L, N = len(lm.layers), ao.ffn_width(lm, 0)
    shard_dir = Path(out_dir) / "steer_lens.d"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for l in range(L):
        why = _shard_refusal(rt, shard_dir / f"L{l}.npz") if (shard_dir / f"L{l}.npz").exists() else None
        if why:
            raise ValueError(why)
    todo = list(range(L)) if layers is None else [int(l) for l in layers]
    t0 = time.time()
    for l in todo:
        shard = shard_dir / f"L{l}.npz"
        if shard.exists():
            continue
        res = build_steer_layer(rt, l, float(p50[l]), dose=dose, k=k, batch=batch)
        save_npz(shard, **res, p50=np.array(float(p50[l])), meta=_meta(rt, layer=l, dose=float(dose)))
        log(f"steer lens: layer {l} done ({time.time() - t0:.0f} s)")
    missing = [l for l in range(L) if not (shard_dir / f"L{l}.npz").exists()]
    if missing:
        log(f"steer lens: {len(missing)} layers still to build ({missing[:6]}{'…' if len(missing) > 6 else ''}); not merged yet")
        return shard_dir
    up = np.zeros((L, N, k), np.int32)
    down = np.zeros((L, N, k), np.int32)
    up_kl = np.zeros((L, N), np.float16)
    down_kl = np.zeros((L, N), np.float16)
    for l in range(L):
        with np.load(shard_dir / f"L{l}.npz") as z:
            up[l], down[l], up_kl[l], down_kl[l] = z["up"], z["down"], z["up_kl"], z["down_kl"]
    path = Path(out_dir) / FILES["steer_lens"]
    save_npz(path, up=up, down=down, up_kl=up_kl, down_kl=down_kl, dose=np.array(float(dose)),
             contexts=np.array(list(ao.STEER_LENS_CHATS) + list(ao.STEER_LENS_PLAIN)), meta=_meta(rt, dose=float(dose)))
    return path


# ------------------------------------------------------------------------------------------------------------ layout


def grid_side(n: int) -> int:
    side = int(np.ceil(np.sqrt(n)))
    return side


def snap_to_grid(xy: np.ndarray, side: int) -> np.ndarray:
    """Points placed on a side x side sheet (the reference's seriation): sorted by y into rows of `side`, then by x
    within each row; returns each point's cell, row * side + column."""
    n = xy.shape[0]
    cells = np.zeros(n, np.int32)
    by_y = np.argsort(xy[:, 1], kind="stable")
    for r in range(int(np.ceil(n / side))):
        band = by_y[r * side : (r + 1) * side]
        band = band[np.argsort(xy[band, 0], kind="stable")]
        cells[band] = r * side + np.arange(len(band))
    return cells


@torch.no_grad()
def build_layout(rt: Any, out_dir: Path, *, pca: int = 50, neighbors: int = 30, method: str = "auto", log: Log = print) -> Path:
    """Each layer's neurons placed on its sheet so that neurons writing alike sit together (the reference's
    build_layout "writes"): each neuron's write direction (w_post (.) W_down[:, n], unit length), centred, projected on
    its first `pca` principal directions, laid out in two dimensions by UMAP (umap-learn, cosine) where it is installed
    and `method` allows, else by the first two principal directions, then snapped to the sheet's grid."""
    lm = language_model(rt.hf_model)
    w = ao.Weights(lm, rt)
    L, N = w.n_layers, w.d_ff
    side = grid_side(N)
    cells = np.zeros((L, N), np.int16 if side * side <= 32767 else np.int32)
    umap_mod = None
    if method in ("auto", "umap"):
        try:
            import umap as umap_mod
        except ImportError:
            if method == "umap":
                raise
    use_umap = umap_mod is not None
    t0 = time.time()
    for l in range(L):
        X = (w.w_down(l).float().T * w.norm(l, "post_feedforward").to(w.device)).cpu()  # [N, D]
        X = X / X.norm(dim=1, keepdim=True).clamp(min=1e-30)
        X = X - X.mean(0, keepdim=True)
        q = min(int(pca), min(X.shape) - 1)
        U, S, _ = torch.pca_lowrank(X, q=q, center=False)
        Z = (U[:, :q] * S[:q]).numpy()
        if use_umap:
            xy = umap_mod.UMAP(n_neighbors=int(neighbors), min_dist=0.05, metric="cosine", random_state=0).fit_transform(Z)
        else:
            xy = Z[:, :2]
        cells[l] = snap_to_grid(np.asarray(xy, np.float64), side)
        log(f"layout: layer {l} ({'umap' if use_umap else 'pca'}, {time.time() - t0:.0f} s)")
    path = Path(out_dir) / FILES["layout"]
    save_npz(path, cell=cells, side=np.array(side), meta=_meta(rt, method="umap" if use_umap else "pca", pca=int(pca)))
    return path

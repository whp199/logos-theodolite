"""Building the atlas's indexes (`backend/atlas_build.py`): what scripts/atlas_build.py runs on the GPU, one function per
index file (backend/atlas_index.py names them). Each reads the model through `atlas_ops.Weights` (the runtime's
accessors on the real model, so W4A16 dequantized on the fly and any vault edit in force included; a fresh process has
none) and writes its file atomically with its provenance under `meta`. scripts/check_atlas_index.py runs every one on the
CPU stand-in and compares what it wrote with the same quantity read live, one neuron at a time.

What each costs on Gemma-4-31B is estimated in scripts/atlas_build.py's help from the arithmetic, not measured: the
lens index and the wiring index are dense matrix products over every layer (minutes), the activation extremes one
forward per window of the corpus, the steer lens a forward of every later layer for every neuron, both signs and four
contexts (hours: it can be built a range of layers at a time and resumes), the layout a projection per layer on the CPU.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from backend import atlas_ops as ao
from backend.atlas_index import FILES, LENS_K, WIRING_K, save_npz
from backend.causal_check import language_model, logits_fp32, model_device, pad_right
from backend.label_gate import label_top_ids, writes_nothing
from backend.theodolite_math import unembed_gemma4

Log = Callable[[str], None]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _meta(rt: Any, **kw: Any) -> np.ndarray:
    from backend.w4a16_fast import describe  # how the 4-bit weights were unpacked (all unpackers give the same bits)

    card = rt.lens_card() if hasattr(rt, "lens_card") else None
    m = {"model": getattr(rt, "model_id", None), "built_at": _now(), "lens_id": (card or {}).get("id"),
         "stand_in": bool(getattr(rt, "atlas_stand_in", False)), "unpacker": describe(rt), **kw}
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
                and old["lens_pos"].shape == (L, N, k)):
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
    qs = lambda vs: np.array([np.quantile(np.array(v, np.float32), [0.1, 0.5, 0.9]) if v else [np.nan] * 3 for v in vs], np.float32)  # noqa: E731
    log(f"stats: {ex.tokens} word-pieces in {len(texts_ids)} windows ({time.time() - t0:.0f} s)")
    path = Path(out_dir) / FILES["act_scale"]
    save_npz(path, pos=ex.pos.numpy(), neg=ex.neg.numpy(), pp=ex.pp.numpy(), nn=ex.nn.numpy(), pn=ex.pn.numpy(),
             np=ex.np.numpy(), tokens=np.array(ex.tokens), resid_q=qs(ex.resid_norms), mlp_rms_q=qs(ex.mlp_rms),
             mid_rms_q=qs(ex.mid_rms), meta=_meta(rt, corpus=corpus, tokens=ex.tokens, windows=len(texts_ids)))
    return path


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
    layer scalars between), kept as log10 of that over the neuron's strongest (`lw`), with each link's gate sign. Columns
    go `chunk` at a time, so no [N, N] product is held whole."""
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

    def merge(side: str, layer: int, rows: slice, le: torch.Tensor, z: torch.Tensor, idx: torch.Tensor, other: int, sign: torch.Tensor,
              gate: torch.Tensor) -> None:
        ce = torch.cat([be[side][layer, rows], le], 1)
        top = ce.topk(K, dim=1)
        be[side][layer, rows] = top.values
        for store, new in ((bz, z), (bl, torch.full_like(idx, other, dtype=torch.int16)), (bn, idx.to(torch.int32)), (bs, sign), (bg, gate)):
            store[side][layer, rows] = torch.cat([store[side][layer, rows], new], 1).gather(1, top.indices)

    sgn = lambda x: torch.where(x > 0, 1, -1).to(torch.int8)  # noqa: E731
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
            for a0 in range(0, N, int(chunk)):
                V = (Wd[:, a0 : a0 + chunk].float() * norms).to(Gj.dtype)  # [D, c]
                G = (Gj @ V).float()
                P = G * (Uj @ V.to(Uj.dtype)).float()  # [N_j, c]
                M = P.abs()
                c = M.shape[1]
                # downstream of each A in this chunk: its column over every B of layer j
                zc = (M - M.mean(0, keepdim=True)) / (M.std(0, keepdim=True) + 1e-12)
                kk = min(K, zc.shape[0])
                top = zc.topk(kk, dim=0)
                le = torch.log(M.gather(0, top.indices)) + two_lf  # -inf where the column writes nothing there
                merge("down", i, slice(a0, a0 + c), le.T, top.values.T, top.indices.T, j, sgn(P.gather(0, top.indices)).T,
                      sgn(G.gather(0, top.indices)).T)
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
                r_top = keep.values
                del V, G, P, M, zc
            mean = r_sum / N
            std = ((r_sq - N * mean * mean) / max(N - 1, 1)).clamp(min=0).sqrt()
            z = ((r_top.double() - mean[:, None]) / (std[:, None] + 1e-12)).float()
            z = torch.where(torch.isfinite(r_top), z, torch.full_like(z, ninf))
            merge("up", j, slice(0, N), torch.log(r_top) + two_lf, z, r_idx, i, r_sgn, r_gate)
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
    path = Path(out_dir) / FILES["wiring"]
    save_npz(path, compressed=True, **arrays, K=np.array(K),
             meta=_meta(rt, weighed="across layers: |P| f^2, the norms at their usual scale and the layer scalars between",
                        scale_source=scale_source, link_model=ao.LINK_MODEL))
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


def build_steer_lens(rt: Any, out_dir: Path, p50: Sequence[float], *, layers: Optional[Sequence[int]] = None,
                     dose: float = ao.STEER_LENS_DOSE, k: int = 12, batch: int = 64, log: Log = print) -> Path:
    """The steer lens of every neuron, a layer at a time into `steer_lens.d/L{l}.npz` beside the index (a run stopped
    part-way resumes at the first missing layer), merged into steer_lens.npz once every layer is there."""
    lm = language_model(rt.hf_model)
    L, N = len(lm.layers), ao.ffn_width(lm, 0)
    shard_dir = Path(out_dir) / "steer_lens.d"
    shard_dir.mkdir(parents=True, exist_ok=True)
    todo = list(range(L)) if layers is None else [int(l) for l in layers]
    t0 = time.time()
    for l in todo:
        shard = shard_dir / f"L{l}.npz"
        if shard.exists():
            continue
        res = build_steer_layer(rt, l, float(p50[l]), dose=dose, k=k, batch=batch)
        save_npz(shard, **res, p50=np.array(float(p50[l])))
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

#!/usr/bin/env python3
"""R3 fit library (no model loaded here; the GPU scripts R3_02-R3_04 call it on the 31B, the self-tests on a tiny
Gemma 4 on the CPU): J++'s recipe as R3 runs it.

- Decoder: what the fit needs from a model (the text decoder's blocks, a forward that builds the graph, encode with
  BOS and truncation, unembed with the final norm, LM head and softcap), as jpp_lens's HFLensModel provides it.
- The router (D4, D14): per layer, unit-normalise the kept positions' block outputs, a mean and covariance streamed in
  fp64 over the router records, eigh, the top PCA_DIM components; then jpp_lens's own k-means++ (seed 0, up to 100
  iterations, tolerance 1e-6) on the projected points. Assignment is the nearest centroid, as jpp_lens assigns.
- The estimator (D3, D12): one forward with the graph rooted at the first source layer's output (start_graph_at),
  then per pass a one-hot cotangent at output dimension dim_start + b, set at every kept position (16 .. T-2) of the
  target block's raw output; the rows at a kept source position p are sum over kept p' >= p of dh_target[p']/dh_l[p].
  Rows are batched either by replicating the prompt along the batch axis (as jpp_lens does) or by one forward and
  is_grads_batched backward passes ("vmap"); both give the same rows (--selftest).
- Expert sums: per layer and expert, the sum of the rows of the positions routed there, their count and weight
  (every position weight 1), accumulated pass by pass as jpp_lens's ExpertJacobianTrainer does.
- Shards on disk, one file per layer per shard (L{layer}.pt), and a streamed merge that reads one layer at a time
  (memory-mapped), forms the total and the two halves by shard parity (D6), replaces an expert with fewer than
  MIN_KEPT positions by the layer's pooled Jacobian, and rounds the experts to fp16 as jpp_lens's merge-experts
  file does (fit-weights reads that file, so J++'s weights and combined lens are built from fp16 experts).
- The direction mode (R3_Q1's plain J-lens covectors, in batches of directions) and rows for chosen output
  dimensions (R3_00's parity check); J++'s readout eval for one lens and layer (macro_pass, through jpp_lens's own
  rank code: R3_05's held-out recall).
- The J++ items (R3's reimplementation of jpp_lens's item specs, split and exclusion list, which the parity test
  compares with jpp_lens's), the readout residuals, correctness grading (D8), the per-eval floor, and the expert
  weights through jpp_lens's own ExpertWeightingFitter (imported, not reimplemented).

Self-test (CPU):
  CUDA_VISIBLE_DEVICES="" /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R3_fitlib.py --selftest
"""
from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.dont_write_bytecode = True

import json
import math
import time
import types
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import R3_common as C  # noqa: E402
import R3_lrp  # noqa: E402


def _jpp():
    """jpp_lens's pure modules, from the pinned checkout (imported on first use)."""
    C.import_jpp()
    from lens_evals import eval_utils
    from workspace_lens.fitting.condense_experts import fitter
    from workspace_lens.fitting.condense_experts.expert_jacobians import combine_expert_jacobians
    from workspace_lens.routing import cluster

    return types.SimpleNamespace(cluster=cluster, eval_utils=eval_utils, fitter=fitter,
                                 combine=combine_expert_jacobians)


# ------------------------------------------------------------------------------------------------ the model
class Decoder:
    """R3's view of a loaded HF Gemma 4 (the text-only CausalLM or the multimodal wrapper). Parameters get
    requires_grad False; the tokenizer is set to add BOS when it has the switch (J++'s force_bos)."""

    def __init__(self, hf_model, tokenizer, *, force_bos: bool = True, name: Optional[str] = None):
        self._hf_model = hf_model
        self.tokenizer = tokenizer
        if force_bos and getattr(tokenizer, "bos_token_id", None) is not None and hasattr(tokenizer, "add_bos_token"):
            tokenizer.add_bos_token = True
        hf_model.eval()
        for p in hf_model.parameters():
            p.requires_grad_(False)
        inner = hf_model.model
        self.text = inner.language_model if hasattr(inner, "language_model") else inner
        self.layers = self.text.layers
        self.final_norm = self.text.norm
        self.lm_head = hf_model.lm_head
        tc = hf_model.config.get_text_config()
        self.n_layers = int(tc.num_hidden_layers)
        self.d_model = int(tc.hidden_size)
        self.softcap = getattr(tc, "final_logit_softcapping", None)
        self.name = name or getattr(hf_model.config, "name_or_path", "") or None

    @property
    def input_device(self) -> torch.device:
        return self.text.embed_tokens.weight.device

    def encode(self, text: str, *, max_length: int) -> Tensor:
        """[1, S] ids: BOS + the text's tokens, truncated to max_length (the HF tokenizer call jpp_lens makes)."""
        return self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(
            self.input_device)

    def forward(self, input_ids: Tensor):
        return self.text(input_ids=input_ids, use_cache=False)

    def unembed(self, residual: Tensor) -> Tensor:
        w = self.lm_head.weight
        logits = self.lm_head(self.final_norm(residual.to(w.dtype).to(w.device)))
        if self.softcap is not None:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        return logits


def kept_positions(seq_len: int, skip: int = C.SKIP_FIRST) -> Tensor:
    """Positions skip .. seq_len-2 (the first `skip` and the final position are dropped)."""
    if seq_len - 1 - skip <= 0:
        raise ValueError(f"prompt too short: seq_len={seq_len}, need > {skip + 1} tokens")
    return torch.arange(skip, seq_len - 1)


class _Recorder:
    """Forward hooks on blocks: keep each requested block's output; the output at `root` is made the graph's leaf."""

    def __init__(self, layers, at: Iterable[int], root: Optional[int] = None):
        self.layers, self.at, self.root = layers, sorted(set(at) | ({root} if root is not None else set())), root
        self.out: Dict[int, Tensor] = {}
        self.handles = []

    def __enter__(self):
        for i in self.at:
            def hook(mod, args, out, i=i):
                h = out if torch.is_tensor(out) else out[0]
                if i == self.root:
                    h.requires_grad_(True)
                self.out[i] = h
            self.handles.append(self.layers[i].register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def block_outputs(dec: Decoder, text: str, layers: Sequence[int], max_seq_len: int) -> Tuple[Tensor, Dict[int, Tensor]]:
    """(ids [1, S], {layer: [S, N] fp32}) from one forward pass without a graph."""
    ids = dec.encode(text, max_length=max_seq_len)
    with _Recorder(dec.layers, layers) as rec:
        dec.forward(ids)
    return ids, {l: rec.out[l][0].float() for l in layers}


# ------------------------------------------------------------------------------------------------ the router
class StreamedPCA:
    """Per-layer mean and covariance of unit-normalised points, accumulated in fp64 without storing the points."""

    def __init__(self, layers: Sequence[int], d: int):
        self.n = {l: 0 for l in layers}
        self.s1 = {l: torch.zeros(d, dtype=torch.float64) for l in layers}
        self.s2 = {l: torch.zeros(d, d, dtype=torch.float64) for l in layers}

    def add(self, layer: int, points_PN: Tensor) -> None:
        u = _jpp().cluster.unit_normalize(points_PN.float()).double().cpu()
        self.n[layer] += u.shape[0]
        self.s1[layer] += u.sum(0)
        self.s2[layer] += u.T @ u

    def components(self, layer: int, dim: int) -> Tuple[Tensor, Tensor]:
        """(mean [N], components [N, dim]) in fp64: the covariance's top eigenvectors, descending (fit_pca's order)."""
        n = self.n[layer]
        mean = self.s1[layer] / n
        cov = (self.s2[layer] - n * torch.outer(mean, mean)) / max(n - 1, 1)
        _, vecs = torch.linalg.eigh(cov)
        return mean, vecs[:, -dim:].flip(dims=[1])


@dataclass
class LayerRouter:
    mean: Tensor          # [N] fp32
    components: Tensor    # [N, D] fp32
    centroids: Tensor     # [E, D] fp32

    def project(self, points_PN: Tensor) -> Tensor:
        return _jpp().cluster.normalize_center_project(points_PN, self.mean.to(points_PN.device),
                                                       self.components.to(points_PN.device))

    def assign(self, points_PN: Tensor) -> Tensor:
        """Nearest centroid in the projected space ([P] long), as jpp_lens's LayerActivationRouter.assign."""
        return torch.cdist(self.project(points_PN), self.centroids.to(points_PN.device)).argmin(dim=1)


def kept_points(dec: Decoder, texts: Iterable[str], layers: Sequence[int], *, skip: int = C.SKIP_FIRST,
                max_seq_len: int = C.MAX_SEQ_LEN) -> Iterator[Dict[int, Tensor]]:
    """Per text, {layer: [P, N] fp32} block outputs at the kept positions (too-short texts are skipped, as
    jpp_lens's collect_valid_position_activations skips them)."""
    for t in texts:
        ids, outs = block_outputs(dec, t, layers, max_seq_len)
        try:
            pos = kept_positions(ids.shape[1], skip)
        except ValueError:
            continue
        yield {l: outs[l][pos.to(outs[l].device)] for l in layers}


def fit_router(points_by_prompt: Callable[[], Iterator[Dict[int, Tensor]]], layers: Sequence[int], *,
               n_experts: int = C.N_EXPERTS, pca_dim: int = C.PCA_DIM, seed: int = C.ROUTER_SEED,
               max_iter: int = C.KMEANS_MAX_ITER, tol: float = C.KMEANS_TOL) -> Dict[int, LayerRouter]:
    """Two passes over `points_by_prompt()` (each yields {layer: [P, N]} kept-position block outputs of one record):
    pass 1 the streamed fp64 PCA, pass 2 the projected points, then jpp_lens's k-means per layer."""
    jc = _jpp().cluster
    pca: Optional[StreamedPCA] = None
    for pts in points_by_prompt():
        if pca is None:
            pca = StreamedPCA(layers, next(iter(pts.values())).shape[-1])
        for l in layers:
            pca.add(l, pts[l])
    assert pca is not None, "no router records"
    params = {}
    for l in layers:
        mean, comps = pca.components(l, pca_dim)
        params[l] = (mean.float(), comps.float())
    projected: Dict[int, List[Tensor]] = {l: [] for l in layers}
    for pts in points_by_prompt():
        for l in layers:
            projected[l].append(jc.normalize_center_project(pts[l].float().cpu(), *params[l]))
    routers = {}
    for l in layers:
        cent, _ = jc.kmeans(torch.cat(projected[l]), n_experts, seed=seed, max_iterations=max_iter, tolerance=tol)
        routers[l] = LayerRouter(params[l][0], params[l][1], cent)
    return routers


def save_router(routers: Dict[int, LayerRouter], path: Path, **meta) -> None:
    torch.save({"layers": sorted(routers), "mean": {l: r.mean for l, r in routers.items()},
                "components": {l: r.components for l, r in routers.items()},
                "centroids": {l: r.centroids for l, r in routers.items()}, "meta": meta}, path)


def load_router(path: Path) -> Dict[int, LayerRouter]:
    st = torch.load(path, map_location="cpu", weights_only=True)
    return {l: LayerRouter(st["mean"][l], st["components"][l], st["centroids"][l]) for l in st["layers"]}


# ------------------------------------------------------------------------------------------------ the estimator
@dataclass
class Forward:
    seq_len: int
    positions: Tensor                       # [P] kept positions
    target: Tensor                          # [B or 1, S, F] the target block's raw output, attached
    sources: List[Tensor]                   # per source layer [B or 1, S, N], attached
    source_points: Dict[int, Tensor]        # per source layer [P, N] fp32 detached (first replica): what is routed
    input_ids: Tensor                       # [S]
    batch: int
    mode: str
    lrp_report: Optional[Dict[str, Any]] = None


def fit_forward(dec: Decoder, text: str, layers: Sequence[int], *, target: int = C.TARGET_LAYER,
                skip: int = C.SKIP_FIRST, max_seq_len: int = C.MAX_SEQ_LEN, rows_per_pass: int = 8,
                mode: str = "replicate", lrp: bool = True, linears_ctx=None) -> Forward:
    """The recorded forward of one prompt: replicated `rows_per_pass` times along the batch axis ("replicate", as
    jpp_lens) or once ("vmap"), the graph rooted at min(layers), under the R-lens rules when `lrp`. `linears_ctx` is
    an extra context for the forward (the W4A16 recompute-in-backward linears on the real model)."""
    if mode not in ("replicate", "vmap"):
        raise ValueError(mode)
    layers = sorted(layers)
    ids = dec.encode(text, max_length=max_seq_len)
    S = ids.shape[1]
    pos = kept_positions(S, skip)
    B = rows_per_pass if mode == "replicate" else 1
    rep = None
    with _Recorder(dec.layers, [*layers, target], root=layers[0]) as rec, torch.enable_grad(), \
            (linears_ctx if linears_ctx is not None else nullcontext()), \
            (R3_lrp.rlens(dec._hf_model) if lrp else nullcontext()) as lrp_rep:
        dec.forward(ids.expand(B, -1))
        rep = lrp_rep.summary() if lrp else None
    tgt = rec.out[target]
    srcs = [rec.out[l] for l in layers]
    dev_pos = pos.to(tgt.device)
    points = {l: rec.out[l][0, dev_pos.to(rec.out[l].device)].detach().float() for l in layers}
    return Forward(S, dev_pos, tgt, srcs, points, ids[0], B, mode, rep)


def backward_rows(fw: Forward, rows_per_pass: int) -> Iterator[Tuple[int, int, List[Tensor]]]:
    """Per pass: (dim_start, nb, [rows [nb, S, N] per source layer]); row b is output dimension dim_start + b of the
    estimator at every position (read the kept positions). The graph is freed on the last pass."""
    F = fw.target.shape[-1]
    n_pass = math.ceil(F / rows_per_pass)
    pos = fw.positions
    for k, ds in enumerate(range(0, F, rows_per_pass)):
        nb = min(rows_per_pass, F - ds)
        b = torch.arange(nb, device=fw.target.device)
        keep = k < n_pass - 1
        if fw.mode == "replicate":
            cot = torch.zeros_like(fw.target)
            cot[b[:, None], pos[None, :], ds + b[:, None]] = 1.0
            grads = torch.autograd.grad(fw.target, fw.sources, grad_outputs=cot, retain_graph=keep)
            rows = [g[:nb] for g in grads]
        else:
            cot = torch.zeros((nb, *fw.target.shape), dtype=fw.target.dtype, device=fw.target.device)
            cot[b[:, None], 0, pos[None, :], ds + b[:, None]] = 1.0
            grads = torch.autograd.grad(fw.target, fw.sources, grad_outputs=cot, retain_graph=keep,
                                        is_grads_batched=True)
            rows = [g[:, 0] for g in grads]
        yield ds, nb, rows


def direction_rows(fw: Forward, dirs_KF: Tensor, dim_batch: Optional[int] = None) -> List[Tensor]:
    """The plain J-lens direction mode (the Q1 arm): per source layer [K, N] = mean over kept positions of the
    gradient of sum_{kept p'} <h_target[p'], d_k>, i.e. d_k^T J for each direction. One batched backward, or (with
    `dim_batch`) one per batch of that many directions, the graph kept until the last (Miru's dim_batch)."""
    K = dirs_KF.shape[0]
    if dim_batch is not None and dim_batch < K:
        parts: List[List[Tensor]] = []
        for k0 in range(0, K, dim_batch):
            parts.append(_direction_rows(fw, dirs_KF[k0:k0 + dim_batch], keep=k0 + dim_batch < K))
        return [torch.cat([p[i] for p in parts]) for i in range(len(parts[0]))]
    return _direction_rows(fw, dirs_KF, keep=False)


def _direction_rows(fw: Forward, dirs_KF: Tensor, keep: bool) -> List[Tensor]:
    K = dirs_KF.shape[0]
    pos = fw.positions
    if fw.mode == "replicate":
        if fw.batch < K:
            raise ValueError("replicate mode needs rows_per_pass >= the number of directions")
        cot = torch.zeros_like(fw.target)
        cot[:K, pos, :] = dirs_KF.to(cot)[:, None, :]
        grads = torch.autograd.grad(fw.target, fw.sources, grad_outputs=cot, retain_graph=keep)
        return [g[:K, pos].float().mean(1) for g in grads]
    cot = torch.zeros((K, *fw.target.shape), dtype=fw.target.dtype, device=fw.target.device)
    cot[:, 0, pos, :] = dirs_KF.to(cot)[:, None, :]
    grads = torch.autograd.grad(fw.target, fw.sources, grad_outputs=cot, is_grads_batched=True, retain_graph=keep)
    return [g[:, 0, pos].float().mean(1) for g in grads]


def dims_rows(fw: Forward, dims: Sequence[int], keep: bool = True) -> List[Tensor]:
    """The estimator's rows for the given output dimensions only, one batched backward: per source layer
    [len(dims), N], the mean over kept positions (as prompt_mean_rows reads them). R3_00's parity check compares
    these with an independent single-direction backward on the real model."""
    nb = len(dims)
    pos = fw.positions
    b = torch.arange(nb, device=fw.target.device)
    d = torch.as_tensor(list(dims), device=fw.target.device)
    if fw.mode == "replicate":
        if fw.batch < nb:
            raise ValueError("replicate mode needs rows_per_pass >= the number of dimensions")
        cot = torch.zeros_like(fw.target)
        cot[b[:, None], pos[None, :], d[:, None]] = 1.0
        grads = torch.autograd.grad(fw.target, fw.sources, grad_outputs=cot, retain_graph=keep)
        return [g[:nb][:, pos.to(g.device)].float().mean(1) for g in grads]
    cot = torch.zeros((nb, *fw.target.shape), dtype=fw.target.dtype, device=fw.target.device)
    cot[b[:, None], 0, pos[None, :], d[:, None]] = 1.0
    grads = torch.autograd.grad(fw.target, fw.sources, grad_outputs=cot, retain_graph=keep, is_grads_batched=True)
    return [g[:, 0][:, pos.to(g.device)].float().mean(1) for g in grads]


def prompt_mean_rows(fw: Forward, rows_per_pass: int) -> Dict[int, Tensor]:
    """The plain per-prompt J-lens estimate: per source layer the [F, N] mean over kept positions of the rows."""
    layers = sorted(fw.source_points)
    F = fw.target.shape[-1]
    out = {l: torch.zeros(F, fw.sources[0].shape[-1]) for l in layers}
    for ds, nb, rows in backward_rows(fw, rows_per_pass):
        for l, r in zip(layers, rows):
            out[l][ds:ds + nb] = r[:, fw.positions.to(r.device)].float().mean(1).cpu()
    return out


# ------------------------------------------------------------------------------------------------ expert sums
@dataclass
class LayerSums:
    row_sum: Tensor     # [E, F, N] fp32
    count: Tensor       # [E] int64
    weight: Tensor      # [E] fp32

    @classmethod
    def zeros(cls, E: int, F: int, N: int) -> "LayerSums":
        return cls(torch.zeros(E, F, N), torch.zeros(E, dtype=torch.long), torch.zeros(E))

    def add_(self, other: "LayerSums") -> None:
        self.row_sum += other.row_sum
        self.count += other.count
        self.weight += other.weight

    def state(self) -> Dict[str, Tensor]:
        return {"row_sum": self.row_sum, "count": self.count, "weight": self.weight}


@dataclass
class PromptRecord:
    prompt_idx: int
    seq_len: int
    positions: List[int]
    token_ids_sha256: str


def token_ids_sha256(ids_S: Tensor) -> str:
    """jpp_lens's prompt identity: sha256 of the int64 bytes of the token ids."""
    import hashlib

    return hashlib.sha256(ids_S.detach().cpu().to(torch.int64).numpy().tobytes()).hexdigest()


def accumulate_prompt(sums: Dict[int, LayerSums], routers: Dict[int, LayerRouter], fw: Forward,
                      rows_per_pass: int) -> None:
    """Add one prompt's rows to the running per-expert sums (all layers), as jpp_lens's ExpertJacobianTrainer.fit_step:
    the routing first (counts and weights), then pass by pass the membership-weighted row sums."""
    layers = sorted(sums)
    member = {}
    for l in layers:
        E = sums[l].count.shape[0]
        a = routers[l].assign(fw.source_points[l])
        m = (a[None, :] == torch.arange(E, device=a.device)[:, None]).float()
        member[l] = m
        sums[l].count += m.sum(1).long().cpu()
        sums[l].weight += m.sum(1).float().cpu()
    for ds, nb, rows in backward_rows(fw, rows_per_pass):
        for l, r in zip(layers, rows):
            src = r[:, fw.positions.to(r.device), :].float()
            sums[l].row_sum[:, ds:ds + nb, :] += torch.einsum("ep,bpn->ebn", member[l].to(src.device), src).cpu()


def fit_shard(dec: Decoder, prompts: Sequence[str], routers: Dict[int, LayerRouter], layers: Sequence[int], *,
              n_experts: int = C.N_EXPERTS, rows_per_pass: int = 8, mode: str = "replicate", lrp: bool = True,
              first_idx: int = 0, **fw_kw) -> Tuple[Dict[int, LayerSums], List[PromptRecord]]:
    """The expert sums of one contiguous shard of prompts (too-short prompts are skipped, as jpp_lens skips them)."""
    layers = sorted(layers)
    sums = {l: LayerSums.zeros(n_experts, dec.d_model, dec.d_model) for l in layers}
    recs = []
    for i, text in enumerate(prompts):
        try:
            fw = fit_forward(dec, text, layers, rows_per_pass=rows_per_pass, mode=mode, lrp=lrp, **fw_kw)
        except ValueError:
            continue
        accumulate_prompt(sums, routers, fw, rows_per_pass)
        recs.append(PromptRecord(first_idx + i, fw.seq_len, fw.positions.cpu().tolist(), token_ids_sha256(fw.input_ids)))
    return sums, recs


def shard_slices(n_prompts: int, n_shards: int) -> List[slice]:
    """Contiguous near-equal slices, the remainder spread over the first shards (jpp_lens's shard_prompt_slice)."""
    base, rem = divmod(n_prompts, n_shards)
    out = []
    for s in range(n_shards):
        start = s * base + min(s, rem)
        out.append(slice(start, start + base + (1 if s < rem else 0)))
    return out


def write_shard(shard_dir: Path, sums: Dict[int, LayerSums], records: Sequence[PromptRecord], **meta) -> None:
    """One file per layer (L{layer:02d}.pt: row_sum, count, weight) and meta.json (prompts and settings)."""
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    for l, s in sums.items():
        tmp = shard_dir / f"L{l:02d}.pt.tmp"
        torch.save(s.state(), tmp)
        os.replace(tmp, shard_dir / f"L{l:02d}.pt")
    (shard_dir / "meta.json").write_text(json.dumps(
        {"layers": sorted(sums), "records": [r.__dict__ for r in records], **meta}, indent=1) + "\n")


def build_experts(s: LayerSums, min_kept: int = C.MIN_KEPT) -> Tuple[Tensor, Tensor, Tensor]:
    """(experts [E, F, N], pooled [F, N], fallback [E] bool): expert e = row_sum_e / weight_e if it kept at least
    min_kept positions, else the pooled sum_e row_sum_e / sum_e weight_e (jpp_lens's build_layer_expert_jacobians)."""
    if int(s.count.sum()) == 0:
        raise ValueError("no positions were fitted at all")
    w = s.weight.float()
    pooled = s.row_sum.sum(0) / float(w.sum())
    fallback = (s.count < min_kept) | (w <= 0)
    experts = torch.empty_like(s.row_sum)
    for e in range(s.row_sum.shape[0]):
        experts[e] = pooled if bool(fallback[e]) else s.row_sum[e] / float(w[e])
    return experts, pooled, fallback


@dataclass
class MergedLayer:
    layer: int
    experts_fp16: Tensor                   # [E, F, N] fp16, as jpp_lens's experts file stores them
    pooled: Tensor                         # [F, N] fp32, from the in-memory sums (merge-experts --pooled-lens-out)
    fallback: Tensor
    count: Tensor
    weight: Tensor = None                  # [E] fp32
    # "A" (even shards) / "B" (odd shards): (experts fp32, pooled, fallback, count, weight)
    halves: Dict[str, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] = field(default_factory=dict)


def merge_layer(shard_dirs: Sequence[Path], layer: int, *, min_kept: int = C.MIN_KEPT,
                halves: bool = True) -> MergedLayer:
    """One layer of the streamed merge: the shards' L{layer}.pt files read memory-mapped and summed in shard order
    (exact: plain sums over disjoint prompts), the halves A (even shards) and B (odd shards) beside the total."""
    tot: Optional[LayerSums] = None
    half: Dict[str, Optional[LayerSums]] = {"A": None, "B": None}
    for i, d in enumerate(shard_dirs):
        st = torch.load(Path(d) / f"L{layer:02d}.pt", map_location="cpu", mmap=True, weights_only=True)
        s = LayerSums(st["row_sum"], st["count"], st["weight"])
        if tot is None:
            tot = LayerSums(s.row_sum.clone(), s.count.clone(), s.weight.clone())
        else:
            tot.add_(s)
        k = "A" if i % 2 == 0 else "B"
        if halves:
            if half[k] is None:
                half[k] = LayerSums(s.row_sum.clone(), s.count.clone(), s.weight.clone())
            else:
                half[k].add_(s)
        del st, s
    assert tot is not None
    ex, pooled, fb = build_experts(tot, min_kept)
    out = MergedLayer(layer, ex.half(), pooled, fb, tot.count.clone(), tot.weight.clone())
    for k, h in half.items():
        if h is not None:
            out.halves[k] = (*build_experts(h, min_kept), h.count.clone(), h.weight.clone())
    return out


# ------------------------------------------------------------------------------------------------ J++'s items
# jpp_lens readout_eval_items.READOUT_EVAL_SPECS: (readout rule, expand order-ops synonyms, correctness target)
ITEM_SPECS: Dict[str, Tuple[str, bool, Optional[str]]] = {
    "multihop": ("final_token", False, "target"),
    "multilingual": ("final_token", False, "target"),
    "order-ops": ("final_token", True, "target"),
    "poetry": ("last_newline", False, "first_intermediate"),
    "association": ("final_token", False, None),
    "typo": ("final_token", False, None),
}


@dataclass(frozen=True)
class Item:
    eval_slug: str
    name: str
    prompt: str
    target: Optional[str]
    intermediates: Tuple[str, ...]

    def correctness_target(self) -> Optional[str]:
        src = ITEM_SPECS[self.eval_slug][2]
        return self.target if src == "target" else self.intermediates[0] if src == "first_intermediate" else None


def items_from_raw(raw: Dict[str, List[Dict[str, Any]]], slugs: Sequence[str]) -> List[Item]:
    return [Item(s, it["name"], it["prompt"], it.get("target"), tuple(it["intermediates"]))
            for s in slugs for it in raw[s]]


def readout_position(ids_1S: Tensor, tokenizer, rule: str) -> int:
    S = ids_1S.shape[1]
    if rule == "final_token":
        return S - 1
    nl = [p for p in range(S) if "\n" in tokenizer.decode([int(ids_1S[0, p])])]
    if rule != "last_newline" or not nl:
        raise ValueError(f"readout rule {rule!r}: no position")
    return nl[-1]


def item_candidates(tokenizer, item: Item) -> Dict[str, List[int]]:
    """Each intermediate's single-token candidate ids (jpp_lens's candidate_token_ids, imported); intermediates with
    none are left out."""
    eu = _jpp().eval_utils
    out = {}
    for inter in item.intermediates:
        ids = eu.candidate_token_ids(tokenizer, inter, expand_order_ops_synonyms=ITEM_SPECS[item.eval_slug][1])
        if ids:
            out[inter] = ids
    return out


def scoreable(items: Sequence[Item], tokenizer) -> List[Item]:
    return [it for it in items if item_candidates(tokenizer, it)]


def split_items(items: Sequence[Item], *, seed: int = C.SPLIT_SEED,
                fit_fraction: float = C.FIT_FRACTION) -> Tuple[List[Item], List[Item]]:
    """(fitting, held-out): one numpy default_rng(seed) across the evals in sorted slug order; per eval of n items
    the first max(1, floor(fit_fraction n)) of a permutation fit (jpp_lens's split_fitting_items)."""
    import numpy as np

    by: Dict[str, List[Item]] = {}
    for it in items:
        by.setdefault(it.eval_slug, []).append(it)
    rng = np.random.default_rng(seed)
    fit = set()
    for slug in sorted(by):
        ev = by[slug]
        for j in rng.permutation(len(ev)).tolist()[:max(1, int(fit_fraction * len(ev)))]:
            fit.add((ev[j].eval_slug, ev[j].name))
    return ([it for it in items if (it.eval_slug, it.name) in fit],
            [it for it in items if (it.eval_slug, it.name) not in fit])


def excluded_ids(tokenizer, vocab_size: int, all_items: Sequence[Item]) -> List[int]:
    """J++'s Readout Filtering list: ids in range(vocab_size) whose decoded text has no letter or digit, minus every
    candidate id of all six evals' items (jpp_lens's non_semantic_token_ids)."""
    eu = _jpp().eval_utils
    spared = set()
    for it in all_items:
        for ids in item_candidates(tokenizer, it).values():
            spared |= set(ids)
    return eu.non_semantic_token_ids_sparing(tokenizer, vocab_size, spared, context="R3 exclusion")


def vocab_size(dec: Decoder) -> int:
    with torch.no_grad():
        return int(dec.unembed(torch.zeros(1, dec.d_model, device=dec.input_device)).shape[-1])


@dataclass
class Residuals:
    """The fields jpp_lens's ExpertWeightingFitter reads (its ReadoutResiduals, duck-typed)."""

    eval_slugs: List[str]
    item_names: List[str]
    residuals_L_dict_PN: Dict[int, Tensor]
    candidate_ids_P_list: List[Dict[str, List[int]]]
    hf_model_name: Optional[str]
    max_seq_len: int


@torch.no_grad()
def record_residuals(dec: Decoder, items: Sequence[Item], layers: Sequence[int],
                     max_seq_len: int = C.READOUT_MAX_SEQ_LEN) -> Residuals:
    """The block output at each scoreable item's readout position, per layer ([P, N] fp32)."""
    layers = sorted(layers)
    kept, cands, rows = [], [], {l: [] for l in layers}
    for it in items:
        c = item_candidates(dec.tokenizer, it)
        if not c:
            continue
        ids, outs = block_outputs(dec, it.prompt, layers, max_seq_len)
        p = readout_position(ids, dec.tokenizer, ITEM_SPECS[it.eval_slug][0])
        kept.append(it)
        cands.append(c)
        for l in layers:
            rows[l].append(outs[l][p].float().cpu().clone())
    return Residuals([it.eval_slug for it in kept], [it.name for it in kept],
                     {l: torch.stack(v) for l, v in rows.items()}, cands, dec.name, max_seq_len)


@torch.no_grad()
def grade(dec: Decoder, items: Sequence[Item], max_seq_len: int = C.READOUT_MAX_SEQ_LEN) -> Dict[Tuple[str, str], Optional[bool]]:
    """D8: the model's greedy next token at the item's last position against its correctness target (jpp_lens's
    next_token_matches_target); None for the evals with no question (association, typo)."""
    eu = _jpp().eval_utils
    out = {}
    last = dec.n_layers - 1
    for it in items:
        tgt = it.correctness_target()
        if tgt is None:
            out[(it.eval_slug, it.name)] = None
            continue
        _, outs = block_outputs(dec, it.prompt, [last], max_seq_len)
        logits = dec.unembed(outs[last][-1][None, :])[0].float()
        out[(it.eval_slug, it.name)] = bool(eu.next_token_matches_target(dec.tokenizer.decode([int(logits.argmax())]), tgt))
    return out


def eval_floor(correct_items: Sequence[Item], fitting: Sequence[Item], macro_evals: Sequence[str], *,
               min_correct: int = C.EVAL_FLOOR_CORRECT, min_fitting: int = C.EVAL_FLOOR_FITTING) -> Dict[str, Any]:
    """Which evals enter the macro: those of `macro_evals` keeping at least min_correct items and min_fitting fitting
    items; the counts per eval are returned for the record."""
    counts = {s: {"items": sum(it.eval_slug == s for it in correct_items),
                  "fitting": sum(it.eval_slug == s for it in fitting)} for s in macro_evals}
    kept = [s for s in macro_evals if counts[s]["items"] >= min_correct and counts[s]["fitting"] >= min_fitting]
    return {"counts": counts, "kept": kept, "dropped": [s for s in macro_evals if s not in kept]}


def fit_weights(dec: Decoder, experts: Dict[int, Tensor], residuals: Residuals, excluded: Sequence[int],
                layers: Sequence[int], fitting_names: Sequence[str], condense: Optional[Dict[str, Any]] = None):
    """Per layer, jpp_lens's ExpertWeightingFitter (imported) on the fitting items: returns its ExpertWeighting.
    `experts` is {layer: [E, F, N]} (fp16 as stored, cast per layer to fp32 by the fitter)."""
    jf = _jpp().fitter
    cfg = types.SimpleNamespace(d_model=dec.d_model, hf_model_name=dec.name, checkpoint_name="R3 experts")
    ej = types.SimpleNamespace(experts_L_dict_EFN=dict(experts), config=cfg)
    fitter = jf.ExpertWeightingFitter(dec, ej, residuals, excluded_token_ids=list(excluded))
    return fitter.fit(sorted(layers), item_names=list(fitting_names),
                      condense_config=jf.CondenseConfig(**(condense or C.CONDENSE)))


def combine(experts_EFN: Tensor, weights_E: Tensor) -> Tensor:
    """J_l(w) = sum_e w_e J_{l,e} in fp32 (jpp_lens's combine_expert_jacobians, imported)."""
    return _jpp().combine(experts_EFN.to(torch.float32), weights_E)


@torch.no_grad()
def macro_pass(dec: Decoder, residuals: Residuals, layer: int, J_FN: Optional[Tensor], item_indices: Sequence[int],
               excluded: Sequence[int], k: int = 10) -> Dict[str, Any]:
    """J++'s readout eval for one lens at one layer on the chosen recorded items: per (item, intermediate) pair, a
    hit when the best candidate ranks within k of the unembedded x J^T with the excluded ids removed (J_FN None:
    the logit lens); the macro is the unweighted mean over evals of the hit rate. jpp_lens's own pair table, hit
    margins and macro (rank_objective) are imported, so this is the number its evaluate stage reports."""
    C.import_jpp()
    from workspace_lens.fitting.condense_experts import rank_objective as ro

    dev = dec.input_device
    pairs = ro.build_pair_table(residuals.candidate_ids_P_list, residuals.eval_slugs, list(item_indices), device=dev)
    x = residuals.residuals_L_dict_PN[layer][list(item_indices)].to(dev, torch.float32)
    if J_FN is not None:
        x = x @ J_FN.to(dev, torch.float32).T
    logits = dec.unembed(x).float()
    mask = torch.zeros(logits.shape[-1], dtype=torch.bool, device=dev)
    if len(excluded):
        mask[torch.as_tensor(list(excluded), device=dev)] = True
    margin, _ = ro.hit_margins(logits, pairs, excluded_mask_Bool_V=mask, k=k)
    per_eval = ro.per_eval_mean((margin >= 0).float(), pairs)
    return {"macro": float(per_eval.mean()), "n_pairs": pairs.num_pairs,
            "per_eval": {s: round(float(v), 4) for s, v in zip(pairs.group_eval_slugs, per_eval.tolist())}}


# ------------------------------------------------------------------------------------------------ self-test
def selftest(record: bool = False) -> int:
    fails: List[str] = []
    done: List[Tuple[bool, str]] = []
    t_start = time.time()

    def check(ok: bool, what: str) -> None:
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        done.append((bool(ok), what))
        if not ok:
            fails.append(what)

    torch.manual_seed(0)
    hf = C.build_tiny_gemma4()
    dec = Decoder(hf, C.CharTokenizer(), name="tiny")
    L = dec.n_layers
    layers, target, skip, msl = [1, 2, 3], L - 1, 4, 24
    text = "the quick brown fox jumps over the lazy dog"
    kw = dict(target=target, skip=skip, max_seq_len=msl)

    # 1. vmap rows equal replicated rows, with and without the rules
    for lrp in (False, True):
        fa = fit_forward(dec, text, layers, rows_per_pass=8, mode="replicate", lrp=lrp, **kw)
        fb = fit_forward(dec, text, layers, rows_per_pass=8, mode="vmap", lrp=lrp, **kw)
        ra, rb = prompt_mean_rows(fa, 8), prompt_mean_rows(fb, 8)
        d = max(float((ra[l] - rb[l]).abs().max()) for l in layers)
        check(d <= 1e-6, f"vmap rows equal replicated rows ({'R-lens' if lrp else 'plain'}; max |diff| {d:.1e})")

    # 2. the rows equal brute force: torch.autograd.functional.jacobian of the target block output with respect to a
    # source block output, summed over kept targets p' >= p, read at kept p, averaged
    ids = dec.encode(text, max_length=msl)
    S = ids.shape[1]
    pos = kept_positions(S, skip)
    for lrp in (False, True):
        for src in layers:
            def f(h, src=src):
                """the blocks after `src`, from a given h_src (the block call path of Gemma4TextModel)"""
                with _Recorder(dec.layers, [target]) as rec:
                    def repl(mod, args, out, h=h):
                        return h if torch.is_tensor(out) else (h, *out[1:])
                    hd = dec.layers[src].register_forward_hook(repl)
                    try:
                        dec.forward(ids)
                    finally:
                        hd.remove()
                return rec.out[target][0]
            _, base = block_outputs(dec, text, [src], msl)
            ctx = R3_lrp.rlens(hf) if lrp else nullcontext()
            with ctx:
                Jfull = torch.autograd.functional.jacobian(f, base[src][None])   # [S, F, 1, S, N]
            Jfull = Jfull[:, :, 0]                                                  # [S', F, S, N]
            brute = torch.zeros(dec.d_model, dec.d_model)
            for p in pos.tolist():
                acc = sum(Jfull[q, :, p, :] for q in pos.tolist() if q >= p)
                brute += acc
            brute /= len(pos)
            fw = fit_forward(dec, text, layers, rows_per_pass=8, mode="replicate", lrp=lrp, **kw)
            got = prompt_mean_rows(fw, 8)[src]
            d = float((got - brute).abs().max())
            check(d <= 1e-5, f"rows at layer {src} equal the brute-force Jacobian summed over kept targets p' >= p "
                             f"({'R-lens' if lrp else 'plain'}; max |diff| {d:.1e})")

    # 3. streamed fp64 PCA against jpp_lens's fit_pca on the stored points; k-means deterministic
    jc = _jpp().cluster
    g = torch.Generator().manual_seed(1)
    pts = torch.randn(600, 32, generator=g) @ torch.randn(32, 32, generator=g) + 3.0
    sp = StreamedPCA([0], 32)
    for chunk in pts.split(97):
        sp.add(0, chunk)
    m64, c64 = sp.components(0, 8)
    m32, c32 = jc.fit_pca(jc.unit_normalize(pts), 8)
    cos = (c64.float() * c32).sum(0).abs()
    check(float(cos.min()) >= 0.9999 and float((m64.float() - m32).abs().max()) <= 1e-6,
          f"streamed fp64 PCA equals fit_pca on the stored points (min |cos| {float(cos.min()):.7f})")
    proj = jc.normalize_center_project(pts, m64.float(), c64.float())
    c1, a1 = jc.kmeans(proj, 4, seed=0)
    c2, a2 = jc.kmeans(proj, 4, seed=0)
    check(torch.equal(c1, c2) and torch.equal(a1, a2), "k-means is deterministic for seed 0")

    # 4. expert sums equal per-cluster means computed position by position
    prompts = [text, "a stitch in time saves nine and more words here", "pack my box with five dozen liquor jugs now"]
    routers = fit_router(lambda: kept_points(dec, prompts, layers, skip=skip, max_seq_len=msl), layers,
                         n_experts=3, pca_dim=4)
    sums, recs = fit_shard(dec, prompts, routers, layers, n_experts=3, rows_per_pass=8, mode="replicate", lrp=True,
                           **kw)
    ok = True
    for l in layers:
        ref = torch.zeros(3, dec.d_model, dec.d_model)
        cnt = torch.zeros(3, dtype=torch.long)
        for t in prompts:
            fw = fit_forward(dec, t, layers, rows_per_pass=8, mode="replicate", lrp=True, **kw)
            a = routers[l].assign(fw.source_points[l])
            full = torch.zeros(dec.d_model, len(fw.positions), dec.d_model)
            for ds, nb, rows in backward_rows(fw, 8):
                full[ds:ds + nb] = rows[layers.index(l)][:, fw.positions].float()
            for j in range(len(fw.positions)):
                ref[int(a[j])] += full[:, j, :]
                cnt[int(a[j])] += 1
        ok &= torch.equal(cnt, sums[l].count) and float((ref - sums[l].row_sum).abs().max()) <= 1e-5
    check(ok, "per-expert sums and counts equal the position-by-position sums per cluster")
    check(len(recs) == 3 and all(r.positions == list(range(skip, r.seq_len - 1)) for r in recs),
          "kept positions are 16..T-2 (here skip 4)")

    # 5. per-layer shard files and the streamed merge equal the in-memory merge; halves add up to the total
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        dirs = []
        mem = {l: LayerSums.zeros(3, dec.d_model, dec.d_model) for l in layers}
        for s, sl in enumerate(shard_slices(len(prompts), 3)):
            sm, rc = fit_shard(dec, prompts[sl], routers, layers, n_experts=3, rows_per_pass=8, lrp=True,
                               first_idx=sl.start, **kw)
            d = Path(tmp) / f"shard{s}"
            write_shard(d, sm, rc)
            dirs.append(d)
            for l in layers:
                mem[l].add_(sm[l])
        ok = True
        for l in layers:
            ml = merge_layer(dirs, l, min_kept=5)
            ex, pooled, fb = build_experts(mem[l], 5)
            ok &= torch.equal(ml.experts_fp16, ex.half()) and torch.equal(ml.pooled, pooled) and torch.equal(ml.fallback, fb)
            ok &= torch.equal(mem[l].count, sums[l].count) and float((mem[l].row_sum - sums[l].row_sum).abs().max()) <= 1e-5
            (_, pA, _, cA, wA), (_, pB, _, cB, wB) = ml.halves["A"], ml.halves["B"]
            ok &= torch.equal(cA + cB, ml.count)
            ok &= float(((pA * wA.sum() + pB * wB.sum()) / (wA.sum() + wB.sum()) - ml.pooled).abs().max()) <= 1e-6
        check(ok, "the streamed per-layer merge equals the in-memory merge (experts, pooled, fallback); three "
                  "shards equal one run; halves A (even shards) and B (odd) add up to the total")

    # 6. the direction mode equals the full rows projected on the directions
    fw = fit_forward(dec, text, layers, rows_per_pass=8, mode="vmap", lrp=False, **kw)
    full = prompt_mean_rows(fw, 8)
    dirs = torch.randn(5, dec.d_model, generator=g)
    fw2 = fit_forward(dec, text, layers, rows_per_pass=8, mode="vmap", lrp=False, **kw)
    dr = direction_rows(fw2, dirs)
    d = max(float((dr[i] - dirs @ full[l]).abs().max()) for i, l in enumerate(layers))
    check(d <= 1e-5, f"the direction mode (Q1) equals the full rows projected on each direction (max |diff| {d:.1e})")
    fw3 = fit_forward(dec, text, layers, rows_per_pass=8, mode="vmap", lrp=False, **kw)
    dr2 = direction_rows(fw3, dirs, dim_batch=2)
    d = max(float((a - b).abs().max()) for a, b in zip(dr, dr2))
    check(d <= 1e-6, f"the direction mode in batches of 2 directions equals one batch (max |diff| {d:.1e})")
    for mode in ("replicate", "vmap"):
        fw4 = fit_forward(dec, text, layers, rows_per_pass=8, mode=mode, lrp=True, **kw)
        some = [3, 17, 0, 31]
        got = dims_rows(fw4, some)
        fw5 = fit_forward(dec, text, layers, rows_per_pass=8, mode=mode, lrp=True, **kw)
        ref = prompt_mean_rows(fw5, 8)
        d = max(float((got[i] - ref[l][some]).abs().max()) for i, l in enumerate(layers))
        check(d <= 1e-6, f"rows for chosen output dimensions equal those rows of the full estimate ({mode}; "
                         f"max |diff| {d:.1e})")

    print(f"R3_fitlib self-test: {'ALL PASS' if not fails else str(len(fails)) + ' FAILURES'}")
    if record:
        C.record_selftest(Path(__file__).name, done, time.time() - t_start)
    return 0 if not fails else 1


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--record", action="store_true", help="with --selftest: keep the result in R3_selftests.json")
    a = ap.parse_args()
    if not a.selftest:
        ap.print_help()
        sys.exit(2)
    sys.exit(selftest(record=a.record))

#!/usr/bin/env python3
"""R3 GPU side, shared by R3_00, R3_02-R3_05 and R3_Q1 (nothing here touches the GPU at import; --selftest runs on
the CPU with the GPU hidden).

- The served checkpoint without any lens (D1): Gemma4ForConditionalGeneration from the pinned snapshot, its 4-bit
  linears switched to the instrument's on-the-fly dequantising forward (backend.engine_runtime's
  patch_compressed_tensors_for_low_vram, which also drops the vision and audio towers). J++ is not loaded: the fit
  needs no lens, and its 3.4 GB stay free.
- The recompute-in-backward linear: inside grad_linears(), every 4-bit linear computes x @ deq(W).T and saves only
  the packed weight and its scales; the backward dequantises again (labkit's _W4A16LinearFn, written here in the
  setup_context style so that is_grads_batched backward passes can run through it).
- The job cap: torch's per-process memory fraction set to C.JOB_CAP_GB of the card, so that a configuration needing
  more fails inside the job as an out-of-memory error; and a record of the card's memory (nvidia-smi and torch).
- The dry run's model: the local Gemma 4 12B in bf16 (gemma4_unified), with a device map that keeps its lowest
  blocks on the CPU so that the weights on the card stay under the cap less a headroom for the backward graph.
- An independent route to the estimator for R3_00's parity check: one plain backward per direction from the graph
  rooted at block 0's output (as R1_vjplib.r1_total_vjp does), not through R3_fitlib's batched cotangents.
- Integrity: the git state of the registered files, and the runs log every real GPU job appends to.

Self-test (CPU):
  CUDA_VISIBLE_DEVICES="" /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R3_gpu.py --selftest
"""
from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.dont_write_bytecode = True

import datetime
import json
import platform
import subprocess
import time
import types
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import R3_common as C  # noqa: E402

GPU_RUNS_LOG = C.RESULTS / "R3_gpu_runs.jsonl"


def _backend():
    if str(C.REPO) not in sys.path:
        sys.path.insert(0, str(C.REPO))
    from backend import engine_runtime as ER

    return ER


# ------------------------------------------------------------------------------------------------ 4-bit linears
class W4A16GradLinear(torch.autograd.Function):
    """y = x @ deq(packed, scale).T (+ bias), the dequantised weight never saved: the backward dequantises again.
    The forward is the instrument's (the input cast to the weight's dtype); the input's gradient comes back in the
    input's dtype."""

    @staticmethod
    def forward(x: Tensor, packed: Tensor, scale: Tensor, bias: Optional[Tensor]) -> Tensor:
        w = _backend().dequantize_w4a16_full(packed, scale)
        return F.linear(x.to(w.dtype), w, bias)

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        x, packed, scale, _bias = inputs
        ctx.save_for_backward(packed, scale)
        ctx.x_dtype = x.dtype

    @staticmethod
    def backward(ctx, g: Tensor):
        packed, scale = ctx.saved_tensors
        w = _backend().dequantize_w4a16_full(packed, scale)
        return (g.to(w.dtype) @ w).to(ctx.x_dtype), None, None, None


def _is_w4a16(mod: nn.Module) -> bool:
    return hasattr(mod, "weight_packed") and hasattr(mod, "weight_scale")


@contextmanager
def grad_linears(hf_model: nn.Module) -> Iterator[int]:
    """Every 4-bit linear of `hf_model` runs through W4A16GradLinear for the duration (the number swapped is
    yielded); restored on exit. Refuses a linear carrying a live LARQL overlay (the fit reads the unedited model)."""
    saved = []
    try:
        for name, mod in hf_model.named_modules():
            if not _is_w4a16(mod):
                continue
            if getattr(mod, "_memit_col_overlays", None) or getattr(mod, "_memit_row_overlays", None):
                raise RuntimeError(f"{name} carries a LARQL overlay; R3 fits the unedited model")
            saved.append((mod, mod.__dict__.get("forward")))

            def fwd(m, x):
                return W4A16GradLinear.apply(x, m.weight_packed, m.weight_scale, getattr(m, "bias", None))
            mod.forward = types.MethodType(fwd, mod)
        yield len(saved)
    finally:
        for mod, f in saved:
            if f is None:
                mod.__dict__.pop("forward", None)
            else:
                mod.forward = f


def pack_w4a16(w: Tensor, scale_dtype: torch.dtype = torch.bfloat16) -> Tuple[Tensor, Tensor]:
    """(packed int32 [out, in/8], scale [out, in/32]): symmetric group-32 4-bit quantisation in compressed-tensors'
    pack-quantized layout, the inverse of engine_runtime.dequantize_w4a16_full (nibble j of word k is column 8k+j,
    stored as q + 8). Used by the self-test to give a tiny model genuine 4-bit linears."""
    out_f, in_f = w.shape
    if in_f % 32:
        raise ValueError("in_features must be a multiple of 32")
    g = w.float().reshape(out_f, in_f // 32, 32)
    scale = (g.abs().amax(-1) / 7.0).clamp_min(1e-8).to(scale_dtype)
    q = torch.round(g / scale.float()[..., None]).clamp(-8, 7).to(torch.int64) + 8
    q = q.reshape(out_f, in_f // 8, 8)
    shifts = torch.arange(8, dtype=torch.int64) * 4
    word = (q << shifts).sum(-1)
    word = torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)
    return word, scale


def quantise_tiny(hf_model: nn.Module) -> int:
    """Give every linear of the text decoder's blocks the served checkpoint's form: weight_packed and weight_scale
    buffers (pack_w4a16, fp32 scales) and the instrument's own on-the-fly forward; the float weight is removed."""
    ER = _backend()
    n = 0
    for name, mod in list(hf_model.named_modules()):
        if isinstance(mod, nn.Linear) and ".layers." in f".{name}." and mod.in_features % 32 == 0:
            packed, scale = pack_w4a16(mod.weight.detach(), scale_dtype=torch.float32)
            del mod.weight
            mod.register_buffer("weight_packed", packed)
            mod.register_buffer("weight_scale", scale)
            mod.weight = None
            mod._memit_col_overlays, mod._memit_row_overlays = {}, {}
            mod.forward = ER._make_w4a16_linear_forward(mod)
            n += 1
    return n


# ------------------------------------------------------------------------------------------------ the card
def nvidia_smi() -> Dict[str, Any]:
    """The card's memory (MiB) and the processes using it, as nvidia-smi reports them (a query only)."""
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                           "--format=csv,noheader,nounits"], capture_output=True, text=True)
    rec: Dict[str, Any] = {"ok": q.returncode == 0}
    if q.returncode == 0 and q.stdout.strip():
        name, tot, used, free = [x.strip() for x in q.stdout.strip().splitlines()[0].split(",")]
        rec.update(name=name, total_mib=int(tot), used_mib=int(used), free_mib=int(free))
    rec["processes"] = [dict(zip(("pid", "name", "used_mib"), [x.strip() for x in line.split(",")]))
                        for line in apps.stdout.strip().splitlines() if line.strip()]
    return rec


def vram_record(device: torch.device) -> Dict[str, Any]:
    free, total = torch.cuda.mem_get_info(device)
    return {"at": datetime.datetime.now().isoformat(timespec="seconds"), "nvidia_smi": nvidia_smi(),
            "torch_free_gib": round(free / 2 ** 30, 3), "torch_total_gib": round(total / 2 ** 30, 3),
            "allocated_gib": round(torch.cuda.memory_allocated(device) / 2 ** 30, 3),
            "reserved_gib": round(torch.cuda.memory_reserved(device) / 2 ** 30, 3),
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 3)}


def set_job_cap(device: torch.device, cap_gb: float = C.JOB_CAP_GB) -> Dict[str, Any]:
    """Cap this process's CUDA allocations at cap_gb (GiB) and say what the card had free before."""
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device: this is a GPU job (stop the server first; see R3_PREREG.md §10)")
    free, total = torch.cuda.mem_get_info(device)
    frac = min(1.0, cap_gb * 2 ** 30 / total)
    torch.cuda.set_per_process_memory_fraction(frac, device)
    rec = {"cap_gib": cap_gb, "fraction": round(frac, 4), "free_before_gib": round(free / 2 ** 30, 3),
           "total_gib": round(total / 2 ** 30, 3), "nvidia_smi": nvidia_smi()}
    if free < cap_gb * 2 ** 30:
        rec["warning"] = f"only {free / 2 ** 30:.1f} GiB free, below the {cap_gb} GiB cap: is the server stopped?"
    return rec


def peak_reset(device: torch.device) -> None:
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


# ------------------------------------------------------------------------------------------------ loading
def load_w4a16(device: str = "cuda:0"):
    """(hf_model, tokenizer) of the pinned W4A16 snapshot on `device`, the instrument's 4-bit forward installed,
    towers dropped, no lens loaded."""
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    ER = _backend()
    tok = AutoTokenizer.from_pretrained(str(C.MODEL_DIR), local_files_only=True)
    hf = Gemma4ForConditionalGeneration.from_pretrained(str(C.MODEL_DIR), dtype="auto", device_map=device,
                                                        local_files_only=True)
    n = ER.patch_compressed_tensors_for_low_vram(hf)
    hf.eval()
    for p in hf.parameters():
        p.requires_grad_(False)
    hf._r3_w4a16_linears = n
    return hf, tok


def dry_device_map(model_dir: Path = C.DRY_RUN_MODEL_DIR, cap_gb: float = C.JOB_CAP_GB, headroom_gb: float = 8.0,
                   must_be_on_gpu_from: Optional[int] = None) -> Dict[str, Any]:
    """A device map for the dry run's bf16 checkpoint: the embedding (tied to the LM head), the final norm, the
    rotary tables and the highest blocks on GPU 0, the lowest blocks and the towers on the CPU, so that the weights
    on the card are at most cap_gb - headroom_gb. Bytes are read from the safetensors header; the modules from the
    model built on the meta device (no weights read)."""
    import struct

    import transformers
    from transformers import AutoConfig

    with open(Path(model_dir) / "model.safetensors", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    size = {k: v["data_offsets"][1] - v["data_offsets"][0] for k, v in header.items() if k != "__metadata__"}
    cfg = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
    with torch.device("meta"):
        model = getattr(transformers, cfg.architectures[0])(cfg)
    lm = model.model.language_model
    L = len(lm.layers)
    prefix = "model.language_model"
    layer_bytes = [sum(v for k, v in size.items() if k.startswith(f"{prefix}.layers.{i}.")) for i in range(L)]
    fixed = sum(v for k, v in size.items() if k.startswith(f"{prefix}.") and ".layers." not in k)
    budget = (cap_gb - headroom_gb) * 2 ** 30 - fixed
    gpu_from = L
    while gpu_from > 0 and sum(layer_bytes[gpu_from - 1:]) <= budget:
        gpu_from -= 1
    if must_be_on_gpu_from is not None and gpu_from > must_be_on_gpu_from:
        raise SystemExit(f"the blocks from {must_be_on_gpu_from} do not fit under the cap (only from {gpu_from})")
    dmap: Dict[str, Any] = {}
    for name, _ in lm.named_children():
        if name == "layers":
            for i in range(L):
                dmap[f"{prefix}.layers.{i}"] = 0 if i >= gpu_from else "cpu"
        else:
            dmap[f"{prefix}.{name}"] = 0
    for name, _ in model.model.named_children():
        if name != "language_model":
            dmap[f"model.{name}"] = "cpu"
    for name, _ in model.named_children():
        if name != "model":
            dmap[name] = 0
    covered = all(any(p == k or p.startswith(k + ".") for k in dmap)
                  for p, _ in list(model.named_parameters()) + list(model.named_buffers()))
    gpu_bytes = fixed + sum(layer_bytes[gpu_from:])
    return {"device_map": dmap, "gpu_from": gpu_from, "n_layers": L, "gpu_weight_gib": round(gpu_bytes / 2 ** 30, 3),
            "cpu_layer_gib": round(sum(layer_bytes[:gpu_from]) / 2 ** 30, 3), "covered": covered,
            "architecture": cfg.architectures[0], "cap_gib": cap_gb, "headroom_gib": headroom_gb}


def load_dry_model(source_layers: Sequence[int], cap_gb: float = C.JOB_CAP_GB):
    """(hf_model, tokenizer, device-map record) of the dry run's 12B, every source layer and above on GPU 0."""
    import transformers
    from transformers import AutoConfig, AutoTokenizer

    plan = dry_device_map(cap_gb=cap_gb, must_be_on_gpu_from=min(source_layers))
    cfg = AutoConfig.from_pretrained(str(C.DRY_RUN_MODEL_DIR), local_files_only=True)
    tok = AutoTokenizer.from_pretrained(str(C.DRY_RUN_MODEL_DIR), local_files_only=True)
    hf = getattr(transformers, cfg.architectures[0]).from_pretrained(
        str(C.DRY_RUN_MODEL_DIR), dtype=torch.bfloat16, device_map=plan["device_map"], local_files_only=True)
    hf.eval()
    for p in hf.parameters():
        p.requires_grad_(False)
    return hf, tok, plan


# ------------------------------------------------------------------------------------------------ an independent route
def reference_direction_covectors(dec, text: str, dirs_KF: Tensor, layers: Sequence[int], *, target: int,
                                  skip: int, max_seq_len: int, lrp: bool = False, linears_ctx=None) -> Tensor:
    """[K, len(layers), N]: per direction d_k, the gradient of sum over kept p' of <h_target[p'], d_k> with respect
    to each source block's output, read at the kept positions and averaged over them. One plain backward per
    direction, the graph rooted at block 0's output (R1_vjplib.r1_total_vjp's route), batch 1."""
    import R3_fitlib as FL
    import R3_lrp

    layers = sorted(layers)
    ids = dec.encode(text, max_length=max_seq_len)
    pos = FL.kept_positions(ids.shape[1], skip)
    outs: Dict[int, Tensor] = {}

    def mk(l):
        def hook(mod, args, out):
            h = out if torch.is_tensor(out) else out[0]
            if l == 0:
                h = h.detach().requires_grad_(True)
                outs[l] = h
                return h if torch.is_tensor(out) else (h, *out[1:])
            outs[l] = h
        return hook

    hs = [dec.layers[l].register_forward_hook(mk(l)) for l in sorted({0, *layers, target})]
    res = []
    try:
        with torch.enable_grad(), (linears_ctx if linears_ctx is not None else nullcontext()), \
                (R3_lrp.rlens(dec._hf_model) if lrp else nullcontext()):
            dec.forward(ids)
            ht = outs[target][0]
            K = dirs_KF.shape[0]
            for k in range(K):
                obj = (ht[pos.to(ht.device)].float() @ dirs_KF[k].to(ht.device).float()).sum()
                gr = torch.autograd.grad(obj, [outs[l] for l in layers], retain_graph=k < K - 1)
                res.append(torch.stack([g[0, pos.to(g.device)].float().mean(0).cpu() for g in gr]))
    finally:
        for h in hs:
            h.remove()
        outs.clear()
    return torch.stack(res)


# ------------------------------------------------------------------------------------------------ integrity
def pilot_decision() -> Dict[str, Any]:
    """What R3_00's pilot decided: {"decision": "proceed" | "stop" | None, "why", "choice"} (None: no pilot yet)."""
    if not C.PILOT_JSON.is_file():
        return {"decision": None, "why": f"no {C.PILOT_JSON.name} yet (the pilot has not run)", "choice": None}
    p = json.loads(C.PILOT_JSON.read_text(encoding="utf-8"))
    d = p.get("decision") or {}
    return {"decision": d.get("decision"), "why": d.get("why"), "choice": p.get("choice"),
            "approved_gpu_hours": d.get("approved_gpu_hours"), "projected_gpu_hours": d.get("projected_gpu_hours")}


def require_pilot_proceed() -> Dict[str, Any]:
    """Refuse a fit job unless the pilot passed and its projection is within the budget the user approved, and that
    budget is still the one registered in R3_common."""
    d = pilot_decision()
    if d["decision"] != "proceed":
        raise SystemExit(f"R3_00's pilot does not say proceed ({d['why']}); no fit job runs before it does")
    if d.get("approved_gpu_hours") != C.APPROVED_GPU_HOURS:
        raise SystemExit("the pilot was judged against another budget than R3_common.APPROVED_GPU_HOURS; rerun it")
    return d


def git_state(paths: Sequence[Path]) -> Dict[str, Any]:
    """HEAD and the given paths that are untracked or differ from it (git status --porcelain)."""
    rel = [str(Path(p).resolve().relative_to(C.REPO)) for p in paths if Path(p).exists()]
    try:
        head = subprocess.run(["git", "-C", str(C.REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        st = subprocess.run(["git", "-C", str(C.REPO), "status", "--porcelain", "--untracked-files=all", "--"] + rel,
                            capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        return {"head": None, "dirty": rel, "error": str(e)}
    return {"head": head, "dirty": [line[3:] for line in st.splitlines() if line.strip()]}


def registered_scripts() -> List[Path]:
    """Every R3 script, the registration, R3's prompts, sources and freshness check, the R2 files R3 imports or reads
    and the two backend modules R3 reads through (the files a GPU job requires to be committed and unchanged)."""
    own = sorted(C.SCRIPTS.glob("R3_*.py")) + [C.PREREG_PATH, C.PROMPTS_PATH, C.SOURCES_MANIFEST, C.CHECK_JSON]
    return [p for p in own + [C.REPO / r for r in C.PINNED_UNCHANGED] if p.exists()]


def runtime_state() -> Dict[str, Any]:
    """Whether the registered runtime holds: the backend/ tree (tracked changes and untracked files) is the one at
    C.RUNTIME_COMMIT, and every file of C.PINNED_UNCHANGED (R2's files R3 imports or reads, the two backend modules)
    has its registered sha256 (R3_PREREG.md §0)."""
    out: Dict[str, Any] = {"runtime_commit": C.RUNTIME_COMMIT, "backend_differs": [], "pinned_differ": []}
    try:
        diff = subprocess.run(["git", "-C", str(C.REPO), "diff", "--name-only", C.RUNTIME_COMMIT, "--", "backend/"],
                              capture_output=True, text=True, check=True).stdout.split()
        new = subprocess.run(["git", "-C", str(C.REPO), "ls-files", "--others", "--exclude-standard", "--",
                              "backend/"], capture_output=True, text=True, check=True).stdout.split()
        out["backend_differs"] = sorted(set(diff) | set(new))
    except (OSError, subprocess.CalledProcessError) as e:
        out["backend_differs"] = [f"(git failed: {e})"]
    for rel, sha in C.PINNED_UNCHANGED.items():
        p = C.REPO / rel
        if not p.is_file() or C.sha256_file(p) != sha:
            out["pinned_differ"].append(rel)
    out["ok"] = not out["backend_differs"] and not out["pinned_differ"]
    return out


def require_committed(allow: bool, real: bool, log=print) -> Dict[str, Any]:
    gst = git_state(registered_scripts())
    gst["runtime"] = runtime_state()
    if not gst["runtime"]["ok"]:
        msg = (f"the registered runtime does not hold: backend/ differs from {C.RUNTIME_COMMIT[:7]} at "
               f"{gst['runtime']['backend_differs'][:8]}, pinned files differ: {gst['runtime']['pinned_differ']}")
        if real and not allow:
            raise SystemExit(msg + " (R3_PREREG.md §0); restore them, or pass --allow-uncommitted (a deviation)")
        log("WARNING: " + msg)
    if gst["dirty"] or gst["head"] is None:
        msg = f"registered files not committed at HEAD {gst['head']}: {gst['dirty'][:8]}"
        if real and not allow:
            raise SystemExit(msg + "; commit them first (R3_PREREG.md §0), or pass --allow-uncommitted (a deviation)")
        log("WARNING: " + msg)
    return gst


def log_run(event: Dict[str, Any], path: Path = GPU_RUNS_LOG) -> None:
    """Append one line to the runs log: every real GPU job (started / finished / failed) is reported."""
    event = dict(event, at=datetime.datetime.now().isoformat(timespec="seconds"), host=platform.node())
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


@contextmanager
def gpu_job(name: str, argv: Sequence[str], allow_uncommitted: bool, log=print) -> Iterator[Dict[str, Any]]:
    """The frame of every real GPU job: the git check, the job cap, the VRAM record before and after, and the runs
    log (started, then finished or failed with the error)."""
    gst = require_committed(allow_uncommitted, True, log)
    dev = torch.device("cuda:0")
    rec: Dict[str, Any] = {"job": name, "git": gst, "cap": set_job_cap(dev)}
    log_run({"event": "started", "job": name, "argv": list(argv), "git": gst, "cap": rec["cap"]})
    t0 = time.time()
    try:
        yield rec
    except BaseException as e:
        log_run({"event": "failed", "job": name, "error": f"{type(e).__name__}: {e}",
                 "seconds": round(time.time() - t0, 1)})
        raise
    rec["vram_after"] = vram_record(dev)
    log_run({"event": "finished", "job": name, "seconds": round(time.time() - t0, 1), "vram": rec["vram_after"],
             "out": rec.get("out")})


# ------------------------------------------------------------------------------------------------ self-test
def selftest(record: bool = False) -> int:
    import R3_fitlib as FL
    import R3_lrp

    fails: List[str] = []
    done: List[Tuple[bool, str]] = []
    t_start = time.time()

    def check(ok: bool, what: str) -> None:
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        done.append((bool(ok), what))
        if not ok:
            fails.append(what)

    ER = _backend()
    g = torch.Generator().manual_seed(3)
    w = torch.randn(48, 64, generator=g)
    packed, scale = pack_w4a16(w, torch.float32)
    deq = ER.dequantize_w4a16_full(packed, scale)
    gr = w.reshape(48, 2, 32)
    sc = gr.abs().amax(-1, keepdim=True) / 7.0
    want = (torch.round(gr / sc).clamp(-8, 7) * sc).reshape(48, 64)
    check(torch.allclose(deq, want, atol=1e-6, rtol=0) and packed.dtype == torch.int32
          and tuple(packed.shape) == (48, 8) and tuple(scale.shape) == (48, 2),
          "pack_w4a16 is the inverse of the instrument's dequantize_w4a16_full (int32 [out, in/8], scales [out, in/32])")

    # a tiny Gemma 4 with genuine 4-bit linears, against the same model with the dequantised weights as nn.Linear
    plain = C.build_tiny_gemma4(seed=5)
    quant = C.build_tiny_gemma4(seed=5)
    nq = quantise_tiny(quant)
    for (n1, m1), (n2, m2) in zip(plain.named_modules(), quant.named_modules()):
        if _is_w4a16(m2):
            m1.weight.data.copy_(ER.dequantize_w4a16_full(m2.weight_packed, m2.weight_scale))
    check(nq > 0 and all(not (isinstance(m, nn.Linear) and m.weight is not None and ".layers." in n)
                         for n, m in quant.named_modules() if _is_w4a16(m)),
          f"the tiny model's {nq} block linears are 4-bit (weight_packed, weight_scale, the instrument's forward)")
    tok = C.CharTokenizer()
    dq, dp = FL.Decoder(quant, tok, name="tiny-w4a16"), FL.Decoder(plain, tok, name="tiny-deq")
    text = "the quick brown fox jumps over the lazy dog and runs"
    layers, target, skip, msl = [1, 2, 3], 5, 4, 32
    kw = dict(target=target, skip=skip, max_seq_len=msl)
    with torch.no_grad():
        ids = dq.encode(text, max_length=msl)
        a = quant(ids).logits
        b = plain(ids).logits
    check(float((a - b).abs().max()) <= 1e-5, f"forward through the instrument's 4-bit forward equals the dequantised "
                                               f"model (max |diff| {float((a - b).abs().max()):.1e})")
    for mode in ("replicate", "vmap"):
        for lrp in (False, True):
            with grad_linears(quant) as n_sw:
                fq = FL.fit_forward(dq, text, layers, rows_per_pass=8, mode=mode, lrp=lrp, **kw)
                rq = FL.prompt_mean_rows(fq, 8)
            fp = FL.fit_forward(dp, text, layers, rows_per_pass=8, mode=mode, lrp=lrp, **kw)
            rp = FL.prompt_mean_rows(fp, 8)
            d = max(float((rq[l] - rp[l]).abs().max()) for l in layers)
            check(n_sw == nq and d <= 1e-5, f"rows through the recompute-in-backward linears equal rows through the "
                                            f"dequantised nn.Linear ({mode}, {'R-lens' if lrp else 'plain'}; max "
                                            f"|diff| {d:.1e})")
    check(all("forward" not in m.__dict__ or m.forward.__func__.__name__ == "_w4a16_forward"
              for m in quant.modules() if _is_w4a16(m)), "grad_linears restores the instrument's forward on exit")

    # the dequantised weights are never saved for the backward inside grad_linears; outside, the instrument's
    # forward saves them (why the fit needs the swap)
    def saved_weight_like(ctx) -> int:
        seen = []

        def pack(t):
            seen.append((tuple(t.shape), t.dtype))
            return t

        shapes = {tuple(m.weight_scale.shape[:1]) + (m.weight_packed.shape[1] * 8,) for m in quant.modules()
                  if _is_w4a16(m)}
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t), torch.enable_grad(), ctx:
            x = quant.model.embed_tokens(ids).detach().requires_grad_(True)
            quant.model.layers[1].mlp(x)
        return sum(1 for s, dt in seen if s in shapes and dt.is_floating_point)

    n_in, n_out = saved_weight_like(grad_linears(quant)), saved_weight_like(nullcontext())
    check(n_in == 0 and n_out > 0, f"inside grad_linears no dequantised weight is saved for the backward (outside, "
                                   f"the instrument's forward saves {n_out})")

    # a live LARQL overlay is refused
    m0 = next(m for m in quant.modules() if _is_w4a16(m))
    m0._memit_col_overlays = {0: torch.zeros(1)}
    try:
        with grad_linears(quant):
            pass
        check(False, "a 4-bit linear carrying a LARQL overlay is refused")
    except RuntimeError:
        check(True, "a 4-bit linear carrying a LARQL overlay is refused")
    m0._memit_col_overlays = {}

    # the independent route equals R3_fitlib's estimator projected on the same directions
    dirs = torch.randn(4, 32, generator=g)
    for lrp in (False, True):
        ref = reference_direction_covectors(dp, text, dirs, layers, lrp=lrp, **kw)
        fw = FL.fit_forward(dp, text, layers, rows_per_pass=8, mode="vmap", lrp=lrp, **kw)
        rows = FL.prompt_mean_rows(fw, 8)
        d = max(float((ref[:, i] - dirs @ rows[l]).abs().max()) for i, l in enumerate(layers))
        check(d <= 1e-5, f"the independent backward (rooted at block 0, one direction at a time) equals the batched "
                         f"rows projected ({'R-lens' if lrp else 'plain'}; max |diff| {d:.1e})")

    # the dry run's device map, from the 12B's header and meta module tree (no weights read)
    if (C.DRY_RUN_MODEL_DIR / "model.safetensors").is_file():
        dm = dry_device_map()
        on_gpu = [i for i in range(dm["n_layers"]) if dm["device_map"][f"model.language_model.layers.{i}"] == 0]
        check(dm["covered"] and dm["gpu_weight_gib"] <= C.JOB_CAP_GB - 8.0 and on_gpu == list(range(dm["gpu_from"],
                                                                                                   dm["n_layers"]))
              and dm["device_map"]["lm_head"] == 0 and dm["device_map"]["model.language_model.embed_tokens"] == 0,
              f"the 12B dry run's device map covers every weight; {dm['gpu_weight_gib']} GiB on the card (blocks "
              f"{dm['gpu_from']}-{dm['n_layers'] - 1}, embedding and head), {dm['cpu_layer_gib']} GiB of blocks on "
              f"the CPU, under the {C.JOB_CAP_GB} GiB cap less 8 GiB")
    else:
        check(False, f"the dry run's 12B checkpoint is at {C.DRY_RUN_MODEL_DIR}")

    # integrity helpers
    gst = git_state(registered_scripts())
    reg = {str(p.resolve().relative_to(C.REPO)) for p in registered_scripts()}
    check(gst["head"] is not None and isinstance(gst["dirty"], list)
          and set(C.PINNED_UNCHANGED) <= reg and "experiments/results/R3_01_check.json" in reg,
          "git state of the registered files is read; they include R2's imported files, the two backend modules "
          "and R3_01_check.json")
    rt = runtime_state()
    check(rt["ok"], f"the registered runtime holds: backend/ is {C.RUNTIME_COMMIT[:7]}'s tree and every pinned file "
                    f"has its sha256 (differs: {rt['backend_differs'][:3]}, {rt['pinned_differ']})")
    saved = dict(C.PINNED_UNCHANGED)
    try:
        C.PINNED_UNCHANGED["experiments/scripts/R2_common.py"] = "0" * 64
        bad = runtime_state()
        refused = False
        try:
            require_committed(False, True, log=lambda *a: None)
        except SystemExit:
            refused = True
    finally:
        C.PINNED_UNCHANGED.clear()
        C.PINNED_UNCHANGED.update(saved)
    check(not bad["ok"] and bad["pinned_differ"] == ["experiments/scripts/R2_common.py"] and refused,
          "a pinned file whose sha256 differs is caught, and a GPU job refuses to start")
    print(f"R3_gpu self-test: {'ALL PASS' if not fails else str(len(fails)) + ' FAILURES'}")
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

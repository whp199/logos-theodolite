#!/usr/bin/env python3
"""R3 Q0 (CPU only; no model, no GPU, no network): pullbacks of R1's 128 final-space directions through the lens files,
and how far the lenses agree in the quiet dimensions (q10) and in the rest (loud).

A pullback is d . J_l, the covector at layer l whose dot with a state h equals <h J_l^T, d> (every file here is read
as the instrument reads J++: h_J = h @ J.T). The directions are R1's (R1_02_dirs.pt: 30 named directions and 98
random token-pair directions, in the final normalised coordinates); q10 is each layer's bottom 10% of dimensions by
the spread of the 4-bit instruct model's states (V6_quiet_dims_stats.pt). Cosines are R1_04_analyze_dir.py's: over
the coordinates of a mask, per direction and layer, averaged over a band's layers and then over the 98 token pairs.

1. The implementation check (registered, R3_PREREG.md §8). From the pullbacks R1 stored (R1_02_dirs.pt), this code
   must give R1_04_dir_analysis.json's J++-against-JLens rows exactly (4 decimals, as stored). From pullbacks
   recomputed here from the two files themselves, it must give R1's q10 cosines 0.77 / 0.68 / 0.66 at code layers
   30-39 / 40-49 / 50-58 (R1_IT_LENS_REPORT.md:233) within 0.02. Only then is the rest reported.
2. H6's first comparison. Miru's J-lens (bf16 google/gemma-4-31B-it, plain gradients as far as its header
   records, WikiText, Miru Tracer's code and its own re-chunked prompts) against Neuronpedia's JLens (bf16 base,
   anthropics/jacobian-lens on 861 WikiText records): the gap is fine-tuning together with the two fits' code,
   estimator details and prompts, with no quantisation in it. H6's bands (code 30-39, 40-49, 50-57, and 58
   apart, where Miru collapses) are written beside Q0's own ("h6_bands"). Beside it, for reference: Miru
   against J++, and R1's IT covectors (4-bit instruct, same-position VJPs, R1_04_gbar.pt) against Miru, J++ and
   JLens. The 4-bit J-lens of the Q1 arm (R3_Q1_jlens.py, not run yet) is compared with these pullbacks later.

Pullbacks are computed in fp32 from the fp16 files ("fp32"), and again as R1 computed them on the GPU ("r1": bf16
directions times bf16 matrices with fp32 sums, rounded to bf16); both are reported, the check uses both.
Writes experiments/results/R3_Q0_pullbacks.json (numbers only) and R3_Q0_pullbacks.pt (gitignored: the pullbacks of
the 128 directions through J++, JLens and Miru at code layers 0-58, fp32, for the Q1 comparison).
Run: CUDA_VISIBLE_DEVICES="" /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R3_Q0_pullbacks.py
       [--no-hash] [--out PATH] [--pt PATH]
"""
from __future__ import annotations

import os
import sys

if __name__ == "__main__":                         # run as a script: CPU only (imported by R3_Q1: no side effect)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.dont_write_bytecode = True

import argparse
import datetime
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import R3_common as C  # noqa: E402

T0 = time.time()
MASKS = ("all", "loud", "q10")
LAYERS = tuple(range(59))                           # J has no matrix at 59 (the identity there)


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------------------------------------ R1_04's statistics
def quiet_masks() -> Dict[str, torch.Tensor]:
    """{"all", "loud", "q10"}: [60, D] float masks, as R1_04_analyze_dir.py builds them from V6's per-layer spread."""
    v6 = torch.load(C.QUIET_STATS_PATH, weights_only=False, map_location="cpu")
    std = v6["std_all"].float()
    q10 = (std <= torch.quantile(std, 0.1, dim=1, keepdim=True)).float()
    return {"all": torch.ones_like(q10), "loud": 1 - q10, "q10": q10}


def mcos(a: torch.Tensor, b: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """R1_04's masked cosine: a, b [K, L, D], m [L, D] -> [K, L]."""
    num = (a * b * m).sum(-1)
    den = ((a * a * m).sum(-1) * (b * b * m).sum(-1)).clamp_min(1e-300).sqrt()
    return num / den


def band_means(x: np.ndarray, layers: Tuple[int, ...] = LAYERS) -> Dict[str, np.ndarray]:
    """x [K, len(layers)] -> {band: [K]} (the mean over each band's layers, R1_04's band())."""
    col = {l: i for i, l in enumerate(layers)}
    return {name: x[:, [col[l] for l in range(a, b + 1)]].mean(1) for name, a, b in C.Q0_BANDS}


def summarize(cos_KL: torch.Tensor, rng: np.random.Generator, n_boot: int = 2000) -> Dict[str, Any]:
    """Per band: the mean over the 98 token pairs, a 95% percentile interval over directions, and the pronoun
    direction (index 0); per layer: the token-pair mean."""
    x = cos_KL.double().numpy()
    tp = list(C.TOKPAIR_DIRS)
    out: Dict[str, Any] = {"bands": {}, "per_layer_tokpair": [round(float(v), 4) for v in x[tp].mean(0)]}
    idx = rng.integers(0, len(tp), size=(n_boot, len(tp)))
    for name, v in band_means(x).items():
        t = v[tp]
        bs = t[idx].mean(1)
        out["bands"][name] = {"tokpair98": round(float(t.mean()), 4),
                              "ci95": [round(float(q), 4) for q in np.percentile(bs, [2.5, 97.5])],
                              "pron": round(float(v[0]), 4)}
    return out


def h6_summary(cos_KL: torch.Tensor, rng: np.random.Generator, layers: Tuple[int, ...] = LAYERS,
               n_boot: int = 2000) -> Dict[str, Any]:
    """H6's bands (R3_common.H6_BANDS: code 30-39, 40-49, 50-57, and 58 apart): per band the mean over the 98 token
    pairs and its 95% percentile interval over directions (a stream of its own, so Q0's other intervals stay as
    they were)."""
    x = cos_KL.double().numpy()
    tp = list(C.TOKPAIR_DIRS)
    col = {l: i for i, l in enumerate(layers)}
    idx = rng.integers(0, len(tp), size=(n_boot, len(tp)))
    out = {}
    for name, a, b in C.H6_BANDS:
        if not all(l in col for l in range(a, b + 1)):
            continue
        t = x[:, [col[l] for l in range(a, b + 1)]].mean(1)[tp]
        out[name] = {"tokpair98": round(float(t.mean()), 4),
                     "ci95": [round(float(q), 4) for q in np.percentile(t[idx].mean(1), [2.5, 97.5])]}
    return out


# ------------------------------------------------------------------------------------------------ the files
def pinned(path: Path, sha: str, size: int, full: bool) -> Dict[str, Any]:
    rec = C.check_pinned_file(path, sha, size, full=full)
    if not rec["size_ok"] or rec["sha256_ok"] is False:
        raise SystemExit(f"{path} is not the pinned file ({rec})")
    return rec


def lens_reader(kind: str) -> Callable[[int], torch.Tensor]:
    """layer -> [D, D] fp16 matrix of a lens file, read one layer at a time (memory-mapped / safe_open)."""
    if kind in ("jpp", "jlens"):
        path = C.JPP_BASE_PATH if kind == "jpp" else C.JLENS_PATH
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        J = payload["J"] if isinstance(payload, dict) and "J" in payload else payload
        return lambda l: J[l]
    if kind == "miru":
        from safetensors import safe_open

        f = safe_open(str(C.MIRU_PATH), framework="pt", device="cpu")
        return lambda l: f.get_tensor(f"J.{l}")
    raise ValueError(kind)


def pullbacks(read: Callable[[int], torch.Tensor], dirs_KD: torch.Tensor) -> Dict[str, torch.Tensor]:
    """{"fp32": [K, 59, D], "r1": [K, 59, D]} of d @ J_l (fp32 from the stored matrix; and R1's bf16 route)."""
    K, D = dirs_KD.shape
    out = {"fp32": torch.empty(K, len(LAYERS), D), "r1": torch.empty(K, len(LAYERS), D)}
    d32 = dirs_KD.float()
    dbf = dirs_KD.to(torch.bfloat16).float()
    for l in LAYERS:
        J = read(l)
        if tuple(J.shape) != (D, D) or not torch.isfinite(J).all():
            raise SystemExit(f"layer {l}: {tuple(J.shape)} or non-finite values")
        Jf = J.float()
        out["fp32"][:, l] = d32 @ Jf
        out["r1"][:, l] = (dbf @ J.to(torch.bfloat16).float()).to(torch.bfloat16).float()
        del J, Jf
    return out


# ------------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(C.Q0_JSON))
    ap.add_argument("--pt", default=str(C.Q0_PT), help="where to keep the pullbacks ('' = do not)")
    ap.add_argument("--no-hash", action="store_true", help="check sizes only, not sha256 (quicker; recorded)")
    a = ap.parse_args()
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    full = not a.no_hash
    checks: List[Dict[str, Any]] = []

    def check(ok: bool, what: str, **detail) -> bool:
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        checks.append({"ok": bool(ok), "check": what, **detail})
        return ok

    files = {
        "R1_02_dirs.pt": pinned(C.R1_DIRS_PATH, C.R1_DIRS_SHA256, C.R1_DIRS_SIZE, full),
        "V6_quiet_dims_stats.pt": pinned(C.QUIET_STATS_PATH, C.QUIET_STATS_SHA256, C.QUIET_STATS_SIZE, full),
        "jpp": pinned(C.JPP_BASE_PATH, C.JPP_BASE_SHA256, C.JPP_BASE_SIZE, full),
        "jlens": pinned(C.JLENS_PATH, C.JLENS_SHA256, C.JLENS_SIZE, full),
        "miru": pinned(C.MIRU_PATH, C.MIRU_SHA256, C.MIRU_SIZE, full),
        "R1_04_gbar.pt": pinned(C.R1_GBAR_PATH, C.R1_GBAR_SHA256, C.R1_GBAR_SIZE, full),
    }
    log("pinned files:", ", ".join(f"{k} {'sha256 ok' if v['sha256_ok'] else 'size ok'}" for k, v in files.items()))
    miru_fit = C.miru_fit_metadata()
    head = C.miru_header()
    check(head["__metadata__"].get("format") == "miru-tracer/jacobian-lens"
          and head["J.58"]["dtype"] == "F16" and head["J.58"]["shape"] == [C.D_MODEL, C.D_MODEL]
          and miru_fit["provenance"]["model_name_or_path"] == C.MIRU_FIT["model_name_or_path"]
          and miru_fit["provenance"]["compute_dtype"] == "bfloat16",
          "Miru's header: miru-tracer/jacobian-lens, fp16 [5376, 5376] per layer, google/gemma-4-31B-it in bf16")

    DD = torch.load(C.R1_DIRS_PATH, weights_only=False, map_location="cpu")
    names, dirs = DD["names"], DD["dirs"].float()
    check(len(names) == 128 and tuple(dirs.shape) == (128, C.D_MODEL)
          and all(names[k].startswith("tp:") for k in C.TOKPAIR_DIRS) and not names[29].startswith("tp:"),
          "R1's 128 directions: 30 named, then the 98 token pairs (indices 30-127)")
    M = {k: v[:59] for k, v in quiet_masks().items()}
    rng = np.random.default_rng(C.BOOT_SEED)
    rng6 = np.random.default_rng([C.BOOT_SEED, 6])
    res: Dict[str, Any] = {"comparisons": {}, "h6_bands": {}}

    def compare(tag: str, A: torch.Tensor, B: torch.Tensor) -> Dict[str, Any]:
        cs = {mk: mcos(A, B, M[mk]) for mk in MASKS}
        r = {mk: summarize(cs[mk], rng) for mk in MASKS}
        res["comparisons"][tag] = r
        res["h6_bands"][tag] = {mk: h6_summary(cs[mk], rng6) for mk in ("loud", "q10")}
        return r

    # 1a. the stored pullbacks against R1_04's table: this file's statistics are R1's
    ref = json.loads(C.R1_04_JSON.read_text(encoding="utf-8"))
    PBJ, PBL = DD["pb_jpp"][:, :59].float(), DD["pb_jlens"][:, :59].float()
    stored = compare("stored_R1_02: jpp vs jlens", PBJ, PBL)
    worst = 0.0
    for mk in MASKS:
        want = ref["cos_jpp_jlens"][mk]["tokpair98"]["mean"]
        got = [stored[mk]["bands"][b]["tokpair98"] for b, _, _ in C.Q0_BANDS]
        worst = max(worst, max(abs(x - y) for x, y in zip(got, want)))
    check(worst <= 1e-4, f"from R1's stored pullbacks, J++ vs JLens equals R1_04_dir_analysis.json in all six bands "
                         f"and all three masks (max |diff| {worst:.1e}; R1 stored 4 decimals)", max_abs_diff=worst)
    del PBJ, PBL

    # 1b. the registered check: pullbacks recomputed from the two files
    pb: Dict[str, Dict[str, torch.Tensor]] = {}
    for kind in ("jpp", "jlens", "miru"):
        t = time.time()
        pb[kind] = pullbacks(lens_reader(kind), dirs)
        log(f"pullbacks through {kind}: 59 layers in {time.time() - t:.1f}s")
    for route in ("fp32", "r1"):
        r = compare(f"{route}: jpp vs jlens", pb["jpp"][route], pb["jlens"][route])
        for band, want in C.Q0_TARGET_Q10.items():
            got = r["q10"]["bands"][band]["tokpair98"]
            check(abs(got - want) <= C.Q0_TOL,
                  f"recomputed ({route}): J++ vs JLens in q10 at {band} is {got:.4f}, R1 reported {want:.2f} "
                  f"(|diff| {abs(got - want):.4f} <= {C.Q0_TOL})", band=band, got=got, want=want, route=route)
        diff = max(abs(r[mk]["bands"][b]["tokpair98"] - stored[mk]["bands"][b]["tokpair98"])
                   for mk in MASKS for b, _, _ in C.Q0_BANDS)
        check(diff <= C.Q0_TOL, f"recomputed ({route}) agrees with R1's stored pullbacks in every band and mask "
                                f"(max |diff| {diff:.4f})", max_abs_diff=diff, route=route)
    reproduced = all(c["ok"] for c in checks)

    # 2. Miru against JLens (H6) and the reference rows (reported only once the check holds)
    if reproduced:
        compare(C.Q0_MIRU_JLENS, pb["miru"]["fp32"], pb["jlens"]["fp32"])
        compare("fp32: miru vs jpp", pb["miru"]["fp32"], pb["jpp"]["fp32"])
        G = torch.load(C.R1_GBAR_PATH, map_location="cpu", mmap=True, weights_only=False)
        check(G["names"] == names, "R1_04_gbar.pt's directions are R1_02's, in the same order")
        gm = G["mix"][:, :59].float()
        compare("r1 IT (4-bit, same-position VJPs) vs miru", gm, pb["miru"]["fp32"])
        r_it = compare("r1 IT (4-bit, same-position VJPs) vs jpp", gm, pb["jpp"]["fp32"])
        compare("r1 IT (4-bit, same-position VJPs) vs jlens", gm, pb["jlens"]["fp32"])
        # the IT-vs-J++ q10 row of R1's table, as a second check of this code on another pair of files
        want = ref["cos_vs_base"]["mix"]["jpp"]["q10"]["tokpair98"]["mean"]
        got = [r_it["q10"]["bands"][b]["tokpair98"] for b, _, _ in C.Q0_BANDS]
        d = max(abs(x - y) for x, y in zip(got[3:], want[3:]))
        check(d <= C.Q0_TOL, f"R1's IT covectors vs recomputed J++ pullbacks in q10 at L30-58 agree with R1_04's "
                             f"(max |diff| {d:.4f})", max_abs_diff=d)
        del G, gm
    else:
        log("the implementation check failed: Miru against JLens is not reported")

    yard = {
        "same model (IT), chat fit vs prose fit, q10 [R1_04 cos_chat_prose]":
            ref["cos_chat_prose"]["q10"]["tokpair98"]["mean"],
        "split-half ceiling of R1's IT fit, q10 [R1_04 splithalf mix]": ref["splithalf"]["mix"]["q10"]["tokpair98"]["mean"],
    }
    head_ = subprocess.run(["git", "-C", str(C.REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {
        "schema": "R3_Q0/1",
        "ok": all(c["ok"] for c in checks),
        "reproduced_r1": reproduced,
        "meta": {"written_at": datetime.datetime.now().isoformat(timespec="seconds"), "head": head_,
                 "host": platform.node(), "torch": torch.__version__, "seconds": round(time.time() - T0, 1),
                 "files": {k: {"path": v["path"], "sha256_checked": bool(v["sha256_ok"])} for k, v in files.items()},
                 "bands": [b for b, _, _ in C.Q0_BANDS], "directions": "R1_02_dirs.pt (98 token pairs: 30-127)",
                 "q10": "per layer, std_all <= its 10% quantile (V6_quiet_dims_stats.pt, the 4-bit instruct model)",
                 "boot": {"seed": C.BOOT_SEED, "n": 2000, "over": "the 98 token-pair directions"},
                 "miru_fit": {k: miru_fit["provenance"].get(k) for k in ("model_name_or_path", "model_commit_hash",
                                                                          "compute_dtype")}},
        "checks": checks,
        "yardsticks_copied_from_R1_04": {k: [round(x, 4) for x in v] for k, v in yard.items()},
        **res,
    }
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    log(f"wrote {a.out}")
    if a.pt:
        torch.save({"names": names, "layers": list(LAYERS), "pb": {k: v["fp32"] for k, v in pb.items()},
                    "files": {k: v["path"] for k, v in files.items()}}, a.pt)
        log(f"wrote {a.pt}")
    if reproduced:
        for tag in ("fp32: jpp vs jlens", C.Q0_MIRU_JLENS, "fp32: miru vs jpp"):
            q = res["comparisons"][tag]["q10"]["bands"]
            log(tag, "q10:", "  ".join(f"{b} {q[b]['tokpair98']:.3f}" for b in ("L30-39", "L40-49", "L50-58")))
            h = res["h6_bands"][tag]["q10"]
            log("   H6 bands, q10:", "  ".join(f"{b} {v['tokpair98']:.3f} {v['ci95']}" for b, v in h.items()))
    print(f"R3_Q0_pullbacks: {'ALL PASS' if out['ok'] else 'FAILURES'}")
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

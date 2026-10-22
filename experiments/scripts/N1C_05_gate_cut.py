#!/usr/bin/env python3
# N1C_05_gate_cut -- a re-analysis of N1's own data for the instrument's label gate (CPU only, local .venv python). It
# is not a new experiment: it counts N1's static caches and N1C's per-pole causal scores another way, and runs nothing.
# Run: .venv/bin/python -I experiments/scripts/N1C_05_gate_cut.py
#   -> experiments/results/N1C_gate_cut.json and its provenance sidecar N1C_gate_cut.provenance.json (the sha256 of
#      every input, of this script and of the result; scripts/check_label_gate.py pins the result's sha256)
# (N1C_05_audit_dump.py is an older script of the same number: the instrument's design named this one N1C_05_gate_cut.)
#
# WHY. The study's gate (N1_REPORT.md:124-127; N1C_matrix.json q.static_confidence_predictor.jv_nv_top10_overlap) is a
# ranking score: within L40-57 the overlap of a face's J++ (JV) and naked-eye (NV) top ten ranked a sound label above an
# unsound one with AUC 0.8432. It set no threshold. The instrument holds back a label whose two readings share NONE of
# their first ten (backend/label_gate.py CUT = 0; the user's decision 3). This file records what that cut does on N1's
# sample, so the pages can quote counts with a source.
#
# INPUTS (read-only):
#   N1A_cache/static_L{l}.pt  (gitignored) fp32 softcapped full-vocab logits of each sampled neuron's write vector
#                             through J++ ("jpp") and the naked eye ("naive"), face +; the face - is the negation
#                             (the softcap and the RMS norm are odd). Read from this checkout, else from the main
#                             checkout of the worktree this is (git's common directory), else from --cache.
#   N1_neurons.json           the 8 layers x 24 neurons (seed, order).
#   N1C_per_pole.json         poles (layer, neuron, sign) in N1C's order and "full_JV->C", each face's causal P@10
#                             (|top10(JV) & top100(C)| / 10 over the full vocabulary; N1C_02_matrix.py).
#   N1C_matrix.json           the recorded AUC, to check that this file counts the same overlap.
# MEASURE, exactly N1C_02_matrix.py:248-255: shared = |set(top10 ids of s*jpp) & set(top10 ids of s*naive)| (torch.topk
#   on the cached fp32 logits). SOUND = full_JV->C >= 0.3, N1C's threshold for the AUC. Layers count from 0.
# ALSO COUNTED:
#   - every cut 0..3 ("held back when shared <= cut"): how many in-band faces it holds back, how many of them were sound;
#   - the same counts from the first THREE ids a side (the instrument shows three words), against the ten;
#   - the server's path (backend/label_gate.py): the cached logits taken back through the softcap, rounded to bf16 as
#     the server's GEMM rounds them, capped again, shortlisted (64) and ranked in fp32 -- how many faces' counts and how
#     many share-nothing calls differ from the study's.
# PRE-STATED EXPECTATIONS (from the design's read-only count, /tmp/claude-1000/a2/ov.py, not committed): in L40-57,
#   31 of 240 faces share nothing (L40 14, L45 8, L50 4, L55 1, L57 4), 7 of them sound; 160 of the 209 others sound;
#   below L40, 142 of 144 share nothing; three a side, 88 of 240 share nothing, 41% of them sound; the server's path
#   changes no share-nothing call in the band.
import argparse, hashlib, json, os, platform, subprocess, sys, time

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(REPO))
from backend.label_gate import CUT, SHORTLIST, top_ids_from_shortlist  # noqa: E402
from backend.theodolite_math import FINAL_LOGIT_SOFTCAP  # noqa: E402

RESULTS = REPO / "experiments" / "results"
OUT = RESULTS / "N1C_gate_cut.json"
SIDECAR = RESULTS / "N1C_gate_cut.provenance.json"
BAND = (40, 57)
SOUND_AT = 0.3
CUTS = (0, 1, 2, 3)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_cache(given: str = "") -> Path:
    """N1A_cache here, else in the main checkout (git's common directory), else `given`."""
    cands = [Path(given)] if given else []
    cands.append(RESULTS / "N1A_cache")
    try:
        common = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        cands.append(Path(common).parent / "experiments" / "results" / "N1A_cache")
    except Exception:
        pass
    for c in cands:
        if (c / "static_L40.pt").exists():
            return c
    raise SystemExit("N1A_cache (the study's static caches, gitignored) not found: pass --cache <dir>")


def auc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y], score[~y]
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def as_the_server_rounds(y: torch.Tensor) -> torch.Tensor:
    """fp32 softcapped logits as the server's bf16 GEMM gives them (scripts/check_label_gate.py, the same steps)."""
    z = (y / FINAL_LOGIT_SOFTCAP).clamp(-1 + 1e-7, 1 - 1e-7)
    x = FINAL_LOGIT_SOFTCAP * torch.atanh(z)
    return torch.tanh(x.bfloat16().float() / FINAL_LOGIT_SOFTCAP) * FINAL_LOGIT_SOFTCAP


def faces(cache: Path):
    """One row per face, in N1C's pole order: layer, neuron, sign, causal P@10, shared of ten and of three (the study's
    measure), and shared of ten by the server's path."""
    neu = json.loads((RESULTS / "N1_neurons.json").read_text())
    pp = json.loads((RESULTS / "N1C_per_pole.json").read_text())
    target = pp["full_JV->C"]
    rows = []
    for l in [int(x) for x in neu["layers"]]:
        st = torch.load(cache / f"static_L{l}.pt", weights_only=True)
        for r, (ll, n, s) in enumerate(pp["poles"]):
            if ll != l:
                continue
            ni = st["neurons"].index(n)
            jv, nv = s * st["jpp"][ni].float(), s * st["naive"][ni].float()
            j10, n10 = jv.topk(10).indices.tolist(), nv.topk(10).indices.tolist()
            srv = [top_ids_from_shortlist(as_the_server_rounds(y)[None], lambda c, y=y: y[c], k=10)[0] for y in (jv, nv)]
            rows.append({"layer": l, "neuron": int(n), "sign": int(s), "p10": float(target[r]),
                         "shared10": len(set(j10) & set(n10)), "shared3": len(set(j10[:3]) & set(n10[:3])),
                         "server10": len(set(srv[0]) & set(srv[1]))})
    return rows


def tally(rows, key="shared10"):
    """Counts, sound counts and mean P@10, for all faces, for those sharing nothing, and for those sharing some."""
    def part(rs):
        p = [r["p10"] for r in rs]
        return {"n": len(rs), "n_sound": sum(x >= SOUND_AT for x in p), "mean_p10": float(np.mean(p)) if p else None}
    return {"all": part(rows), "share_nothing": part([r for r in rows if r[key] == 0]),
            "share_some": part([r for r in rows if r[key] > 0])}


def analyse(cache: Path) -> dict:
    """The whole result, as N1C_gate_cut.json holds it (scripts/check_label_gate.py recomputes it where the cache is)."""
    rows = faces(cache)
    band = [r for r in rows if BAND[0] <= r["layer"] <= BAND[1]]
    below = [r for r in rows if r["layer"] < BAND[0]]
    y = np.array([r["p10"] >= SOUND_AT for r in band])
    m = json.loads((RESULTS / "N1C_matrix.json").read_text())
    recorded = m["q"]["static_confidence_predictor"]["jv_nv_top10_overlap"]["auc_late_target_p10_ge_0.3"]
    auc10 = auc(np.array([r["shared10"] for r in band], float) / 10, y)
    if abs(auc10 - recorded) > 1e-9:
        raise SystemExit(f"the overlap counted here gives AUC {auc10} in L40-57, not N1C_matrix.json's {recorded}")

    by_layer = {}
    for l in sorted({r["layer"] for r in rows}):
        rs = [r for r in rows if r["layer"] == l]
        by_layer[str(l)] = {**tally(rs), "three_a_side": tally(rs, "shared3")}
    by_overlap = {}
    for k in range(11):
        rs = [r for r in band if r["shared10"] == k]
        if rs:
            by_overlap[str(k)] = {"n": len(rs), "n_sound": sum(r["p10"] >= SOUND_AT for r in rs),
                                  "mean_p10": float(np.mean([r["p10"] for r in rs]))}
    n_sound = int(y.sum())
    cuts = {}
    for c in CUTS:
        held = [r for r in band if r["shared10"] <= c]
        hs = sum(r["p10"] >= SOUND_AT for r in held)
        cuts[str(c)] = {"held_back": len(held), "held_back_sound": hs, "held_back_unsound": len(held) - hs,
                        "kept": len(band) - len(held), "kept_sound": n_sound - hs,
                        "kept_unsound": len(band) - len(held) - (n_sound - hs)}
    three = tally(band, "shared3")
    server_diff = [r for r in band if r["server10"] != r["shared10"]]
    server_calls = [r for r in band if (r["server10"] == 0) != (r["shared10"] == 0)]
    below_diff = [r for r in below if r["server10"] != r["shared10"]]

    out = {
        "what": ("A re-analysis of N1's existing data (not a new experiment): how the instrument's cut of the label gate, "
                 "hold back a label whose J++ and naked-eye readings share none of their first ten word-piece ids, "
                 "falls on N1's sample of random neurons. Layers count from 0 (page layer = layer + 1)."),
        "inputs": ["N1A_cache/static_L{10,20,30,40,45,50,55,57}.pt (gitignored)", "N1_neurons.json",
                   "N1C_per_pole.json (poles, full_JV->C)", "N1C_matrix.json (the recorded AUC)"],
        "measure": ("shared = |top10 ids(s*jpp) & top10 ids(s*naive)| over the full 262,144-id vocabulary, from the "
                    "cached fp32 softcapped logits (N1C_02_matrix.py:248-255)"),
        "sound": f"causal P@10 (full_JV->C, N1C_per_pole.json) >= {SOUND_AT}, N1C's threshold for the AUC",
        "instrument_cut": {"held_back_if_shared_at_most": CUT, "whose": "the instrument's (decision 3); the study set none"},
        "band": {"layers": list(BAND), "sampled": sorted({r["layer"] for r in band}), **tally(band),
                 "auc_shared10_p10_ge_0.3": auc10, "auc_recorded_N1C_matrix": recorded,
                 "auc_shared3_p10_ge_0.3": auc(np.array([r["shared3"] for r in band], float) / 3, y),
                 "by_overlap": by_overlap, "cuts": cuts, "three_a_side": three},
        "below_band": {"layers": sorted({r["layer"] for r in below}), **tally(below)},
        "by_layer": by_layer,
        "server_path": {
            "how": (f"the cached logits back through the softcap, rounded to bf16, capped again, shortlisted "
                    f"({SHORTLIST}) and ranked in fp32 by backend/label_gate.top_ids_from_shortlist"),
            "band_faces_whose_count_differs": len(server_diff),
            "band_share_nothing_calls_that_differ": len(server_calls),
            "below_band_faces_whose_count_differs": len(below_diff),
            "below_band_note": "below layer 40 the naked eye saturates the softcap, so ties decide some first tens",
        },
        "faces": [[r["layer"], r["neuron"], r["sign"], r["shared10"], r["shared3"], r["server10"], round(r["p10"], 6)]
                  for r in rows],
        "faces_columns": ["layer", "neuron", "sign", "shared10", "shared3", "server_shared10", "p10"],
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="", help="the N1A_cache directory, when it is not found by itself")
    args = ap.parse_args()
    t0 = time.time()
    cache = find_cache(args.cache)
    out = analyse(cache)
    OUT.write_text(json.dumps(out, indent=1) + "\n")

    inputs = [cache / f"static_L{l}.pt" for l in sorted(int(l) for l in out["by_layer"])]
    inputs += [RESULTS / n for n in ("N1_neurons.json", "N1C_per_pole.json", "N1C_matrix.json")]
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    side = {
        "result": "experiments/results/N1C_gate_cut.json",
        "result_sha256": sha256_of(OUT),
        "script": "experiments/scripts/N1C_05_gate_cut.py",
        "script_sha256": sha256_of(HERE),
        "label_gate_sha256": sha256_of(REPO / "backend" / "label_gate.py"),
        "git_head_when_run": head,
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cpu_only": True,
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "inputs": [{"path": f"experiments/results/{'N1A_cache/' if 'N1A_cache' in p.parts else ''}{p.name}",
                    "gitignored": "N1A_cache" in p.parts,
                    "bytes": p.stat().st_size, "sha256": sha256_of(p)} for p in inputs],
        "cache_read_from": str(cache),
    }
    SIDECAR.write_text(json.dumps(side, indent=1) + "\n")
    b, three, sp = out["band"], out["band"]["three_a_side"], out["server_path"]
    print(f"L40-57: {b['share_nothing']['n']} of {b['all']['n']} share nothing, {b['share_nothing']['n_sound']} sound; "
          f"{b['share_some']['n_sound']} of {b['share_some']['n']} others sound; AUC {b['auc_shared10_p10_ge_0.3']:.4f}")
    print(f"three a side: {three['share_nothing']['n']} share nothing, {three['share_nothing']['n_sound']} sound; "
          f"AUC {b['auc_shared3_p10_ge_0.3']:.4f}")
    print(f"below L40: {out['below_band']['share_nothing']['n']} of {out['below_band']['all']['n']} share nothing")
    print(f"server path: {sp['band_faces_whose_count_differs']} band counts differ, "
          f"{sp['band_share_nothing_calls_that_differ']} calls; {sp['below_band_faces_whose_count_differs']} below")
    print("saved", OUT.name, SIDECAR.name, f"{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Export E3's average state mu ([60, 5376] float32) from the study's gitignored caches, for the contrast readings'
"study_mean" background (backend/engine_runtime.py `STUDY_MEAN_PATH`). CPU only; nothing is recomputed on a model.

mu is formed exactly as experiments/scripts/E3_lens_validation_09_background.py forms it:
    mu = ((prose mu_sum + chat mu_sum) / (n_prose + n_chat)).float()
from E3_lens_validation_musum_prose.pt and _chat.pt (written by E3_lens_validation_03a_fidelity_prose.py and
03b_fidelity_chat.py: the float64 sums of resid_post over every word-piece from the fifth on of each of E3's prose
passages and chats, read on the instruct model through labkit.forward_resid, HF decoder-layer forward hooks). The
export is checked against the mu the study saved (E3_lens_validation_mu.pt, key "mu") bit for bit before it is
written, and every source file's sha256 is pinned first.

Writes experiments/results/E3_background_mu.npy and its provenance sidecar E3_background_mu.json (the .npy's sha256
is the one engine_runtime pins as `STUDY_MEAN_SHA256`; scripts/check_contrast_readings.py checks the three agree).

Run: .venv/bin/python -I scripts/export_E3_background_mu.py --source /path/to/experiments/results
     (the caches are gitignored: in a worktree, point --source at the main checkout's results; read only)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT_DIR / "experiments" / "results"
OUT_NPY = "E3_background_mu.npy"
OUT_JSON = "E3_background_mu.json"

# the study's caches as they were when E3 (c) ran, by sha256 (measured 2026-10-22 in the main checkout)
SOURCES = {
    "E3_lens_validation_musum_prose.pt": "3e8669672c0f259b30af92e0f278d08bc85c643e3d036e387e98817290f06399",
    "E3_lens_validation_musum_chat.pt": "fa98e0406e933f9805e0e3c4134491fcfe0d877140869a77d0bee906010e6024",
    "E3_lens_validation_mu.pt": "184642b7263fa5b422b57960b2b4337ef832304e068454d2e887ef1fcf07a422",
}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, default=OUT_DIR, help="the experiments/results holding E3's caches")
    args = ap.parse_args()
    src: Path = args.source
    for name, want in SOURCES.items():
        got = sha256_of(src / name)
        if got != want:
            print(f"{src / name}: sha256 {got}, not the {want} E3 ran on")
            return 1
    mp = torch.load(src / "E3_lens_validation_musum_prose.pt", map_location="cpu", weights_only=True)
    mc = torch.load(src / "E3_lens_validation_musum_chat.pt", map_location="cpu", weights_only=True)
    mu = ((mp["mu_sum"] + mc["mu_sum"]) / (mp["n"] + mc["n"])).float()  # as E3_lens_validation_09_background.py:15
    saved = torch.load(src / "E3_lens_validation_mu.pt", map_location="cpu", weights_only=True)["mu"]
    if mu.shape != (60, 5376) or mu.dtype != torch.float32 or not torch.equal(mu, saved):
        print(f"mu {tuple(mu.shape)} {mu.dtype} does not match the study's saved mu bit for bit")
        return 1
    arr = np.ascontiguousarray(mu.numpy())
    npy = OUT_DIR / OUT_NPY
    np.save(npy, arr)
    back = np.load(npy)
    if back.dtype != np.float32 or back.shape != (60, 5376) or not np.array_equal(back, arr):
        print(f"{npy} did not read back as written")
        return 1
    sidecar = {
        "file": OUT_NPY,
        "sha256": sha256_of(npy),
        "bytes": npy.stat().st_size,
        "shape": [60, 5376],
        "dtype": "float32",
        "layers": "0-based, 0..59: row l is the mean of resid_post leaving layer l (the page's layer l + 1)",
        "what": (
            "E3's average state mu: the mean residual over every word-piece from the fifth on of E3's prose passages "
            "and chats, prose and chat pooled by word-piece count, as E3 (c) subtracted it before the lens"
        ),
        "formula": "((prose mu_sum + chat mu_sum) / (n_prose + n_chat)).float()",
        "n_prose": int(mp["n"]),
        "n_chat": int(mc["n"]),
        "read_with": (
            "labkit.forward_resid: HF decoder-layer forward hooks on the instruct model (W4A16), not the server's "
            "interp_engine resid_post capture; that the two agree is not checked here"
        ),
        "sources": {name: {"sha256": want} for name, want in SOURCES.items()},
        "made_by": [
            "experiments/scripts/E3_lens_validation_03a_fidelity_prose.py",
            "experiments/scripts/E3_lens_validation_03b_fidelity_chat.py",
            "experiments/scripts/E3_lens_validation_09_background.py",
        ],
        "checked": "equal bit for bit to E3_lens_validation_mu.pt['mu'], the mu E3 (c) saved and used",
        "report": "experiments/results/E3_lens_validation_REPORT.md, section (c) Lens background prior",
        "exported_by": "scripts/export_E3_background_mu.py",
    }
    (OUT_DIR / OUT_JSON).write_text(json.dumps(sidecar, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {npy} ({sidecar['bytes']} bytes, sha256 {sidecar['sha256']}) and {OUT_DIR / OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

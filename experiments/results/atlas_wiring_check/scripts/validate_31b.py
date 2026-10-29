# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
"""The new wiring_check_summary on tonight's 31B shards, g0/u0 from the ruling's CPU prefill (z1_out), right-aligned
as pushed_reads now records them, f from strength / |g u|. Reads only; writes nothing in the repository."""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "/home/eric/bin/ai/theodolite2")
import numpy as np
import backend.atlas_build as ab
W = "/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/z1_out"
qs = [np.load(f"{W}/q{q}.npz") for q in range(4)]
T = [len(q["ids"]) for q in qs]
P = max(T) - 1
z = np.load("/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/wiring_check.npz")
per = {}
pool = {}
for l in [int(x) for x in z["layers"]]:
    sh = {k[len(f"L{l}_"):]: z[k] for k in z.files if k.startswith(f"L{l}_")}
    rd = sh["readers"]; valid = rd[..., 0] >= 0
    j = np.where(valid, rd[..., 0], 0); b = np.where(valid, rd[..., 1], 0)
    Wn, R = valid.shape
    g0 = np.full((Wn, R, 4, P), np.nan, np.float32); u0 = np.full_like(g0, np.nan)
    for q, d in enumerate(qs):
        n = T[q] - 1
        ga = d["g_all"][:, 1:].astype(np.float32); ua = d["u_all"][:, 1:].astype(np.float32)  # [L, n, N]
        g0[:, :, q, P - n:] = np.moveaxis(ga[j, :, b], -1, -1)
        u0[:, :, q, P - n:] = ua[j, :, b]
        g0[:, :, q, -1] = d["g_last"][j, b]; u0[:, :, q, -1] = d["u_last"][j, b]
    g0[~valid] = np.nan; u0[~valid] = np.nan
    sh["g0"], sh["u0"] = g0, u0
    sh["f"] = np.sqrt(sh["strength"] / np.maximum(np.abs(sh["g"] * sh["u"]), 1e-30)).astype(np.float32)
    s = ab.wiring_check_summary(sh)
    for site in ("last", "mean"):
        for grp in ("linked", "null"):
            r = s[site][grp]
            for k in ("sign_plus", "sign_gate", "use_plus", "use_gate", "use_gate_peak", "rho", "rho_site", "zero", "same_way"):
                if k in r:
                    per.setdefault(f"{site}.{k}.{grp}", []).append(r[k])
            if site == "last":
                for rg, v in r["regimes"].items():
                    for k2 in ("share", "sign_gate", "use_gate"):
                        per.setdefault(f"last.{rg}.{k2}.{grp}", []).append(v[k2])
        per.setdefault(f"{site}.moved_over_null", []).append(s[site]["moved_over_null"])
np.save("/home/eric/bin/ai/theodolite2/experiments/results/atlas_wiring_check/outputs/validate_31b_per.npy", per, allow_pickle=True)
med = {k: round(float(np.median([x for x in v if x is not None])), 3) for k, v in sorted(per.items())}
print(json.dumps(med, indent=0))

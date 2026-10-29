# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
import sys, json
sys.path.insert(0, "/home/eric/bin/ai/theodolite2")
import numpy as np
from backend import atlas_build as ab
z = np.load("/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/wiring_check.npz", allow_pickle=False)
layers = [int(x) for x in z["layers"]]
sg = lambda x: np.where(x > 0, 1, np.where(x < 0, -1, 0))
res = {k: {"linked": [], "null": []} for k in ("sign_gate_nz", "use_gate_nz", "zero", "sign_plus_nz")}
for l in layers:
    sh = {k[len(f"L{l}_"):]: z[k] for k in z.files if k.startswith(f"L{l}_")}
    reads = sh["reads"]
    valid = sh["readers"][..., 0] >= 0
    groups = {"linked": sh["linked"] & valid, "null": ~sh["linked"] & valid}
    dp = reads[:, :, :, 1, 0] - reads[:, :, :, 0, 0]
    dm = reads[:, :, :, 2, 0] - reads[:, :, :, 0, 0]
    gate = sh["gate"][:, :, None]
    dg = np.where(gate > 0, dp, dm)
    sign = np.broadcast_to(sh["sign"][:, :, None].astype(np.int64), dp.shape)
    kp, hp = ab._peak_kappa(sh)
    for name, m in groups.items():
        g_, s_, k_, h_, gt = dg[m], sign[m], kp[m], hp[m], sh["gate"][m][:, None]
        nz = g_ != 0
        res["sign_gate_nz"][name].append(float((sg(g_) == s_)[nz].mean()))
        res["use_gate_nz"][name].append(float((sg(g_) == gt * k_)[nz & h_].mean()))
        res["zero"][name].append(float((~nz).mean()))
        res["sign_plus_nz"][name].append(float((sg(dp[m]) == s_)[dp[m] != 0].mean()))
for k, v in res.items():
    print(k, {n: round(float(np.median(x)), 3) for n, x in v.items()})

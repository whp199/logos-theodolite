# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
import sys
sys.path.insert(0, "/home/eric/bin/ai/theodolite2")
import numpy as np
from scipy import stats
z = np.load("/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/wiring_check.npz", allow_pickle=False)
layers = [int(x) for x in z["layers"]]
out = {k: {"linked": [], "null": []} for k in ("strength", "d", "strength_d2", "sqrtstrength_d", "within_writer_strength", "cv_d")}
for l in layers:
    sh = {k[len(f"L{l}_"):]: z[k] for k in z.files if k.startswith(f"L{l}_")}
    reads = sh["reads"]
    valid = sh["readers"][..., 0] >= 0
    dp = reads[:, :, :, 1, 0] - reads[:, :, :, 0, 0]
    dm = reads[:, :, :, 2, 0] - reads[:, :, :, 0, 0]
    dg = np.where(sh["gate"][:, :, None] > 0, dp, dm)
    size = np.abs(dg).mean(-1)  # [W, R]
    d = np.broadcast_to(sh["lengths"][:, None], size.shape)
    st = sh["strength"]
    for name, m in (("linked", sh["linked"] & valid), ("null", ~sh["linked"] & valid)):
        s = size[m]
        out["strength"][name].append(stats.spearmanr(s, st[m]).statistic)
        out["d"][name].append(stats.spearmanr(s, d[m]).statistic)
        out["strength_d2"][name].append(stats.spearmanr(s, st[m] * d[m] ** 2).statistic)
        out["sqrtstrength_d"][name].append(stats.spearmanr(s, np.sqrt(st[m]) * d[m]).statistic)
    out["cv_d"]["linked"].append(float(np.percentile(sh["lengths"], 90) / max(np.percentile(sh["lengths"], 10), 1e-12)))
for k, v in out.items():
    print(k, {n: round(float(np.median(x)), 3) for n, x in v.items() if x})

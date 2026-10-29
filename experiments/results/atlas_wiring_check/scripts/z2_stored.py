# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
"""Adjudicator: re-derive the decisive figures from wiring_check.npz alone (last word-piece).
Per layer and pooled, linked vs null: sign_plus, sign_gate, use_gate as the summary computes them; the odd/even split;
the odd part against -sign(u) on up-dominated pairs; candidate convention flips with per-layer medians."""
import sys, json
sys.path.insert(0, "/home/eric/bin/ai/theodolite2")
import numpy as np

z = np.load("/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/wiring_check.npz", allow_pickle=True)
layers = [int(x) for x in z["layers"]]
sg = lambda x: np.where(x > 0, 1, np.where(x < 0, -1, 0))
acc = {}
per = {}
def add(key, grp, ok, mask=None):
    ok = ok if mask is None else ok[mask]
    a = acc.setdefault((key, grp), [0, 0]); a[0] += int(ok.sum()); a[1] += int(ok.size)
    per.setdefault((key, grp), []).append(float(ok.mean()) if ok.size else np.nan)
for l in layers:
    reads = z[f"L{l}_reads"][..., 0]  # last site [W,R,Q,3]
    rd = z[f"L{l}_readers"]; lk = z[f"L{l}_linked"]; s = z[f"L{l}_sign"].astype(int); ga = z[f"L{l}_gate"].astype(int)
    g = z[f"L{l}_g"]; u = z[f"L{l}_u"]; op = z[f"L{l}_op"]
    valid = rd[..., 0] >= 0
    dp = reads[..., 1] - reads[..., 0]; dm = reads[..., 2] - reads[..., 0]; a0 = reads[..., 0]
    gate = ga[..., None]; sign = s[..., None]
    dgf = np.where(gate > 0, dp, dm)
    odd = (dp - dm) / 2; even = (dp + dm) / 2
    face = np.where(a0 > 0, 0, 1)
    al = np.where(face == 0, op[:, :, None, 0, 0], op[:, :, None, 1, 0]); be = np.where(face == 0, op[:, :, None, 0, 1], op[:, :, None, 1, 1])
    kappa = sg(al * g[..., None] + be * u[..., None])
    # consistency of the stored signs with the weights' products
    if l == layers[0] or True:
        cs = (s == np.where(g * u > 0, 1, -1))[valid].mean(); cg = (ga == np.where(g > 0, 1, -1))[valid].mean()
        add("consist_sign", "all", np.array([cs == 1.0])); add("consist_gate", "all", np.array([cg == 1.0]))
    for grp, m in (("linked", lk & valid), ("null", ~lk & valid)):
        M = np.broadcast_to(m[..., None], dp.shape)
        nz = M & (dgf != 0)
        add("sign_gate", grp, (sg(dgf) == sign)[nz])
        add("sign_plus", grp, (sg(dp) == sign)[M & (dp != 0)])
        h = nz & np.isfinite(kappa * 1.0) & np.isfinite(al) & np.isfinite(be)
        add("use_gate", grp, (sg(dgf) == gate * kappa)[h])
        add("odd_dominates", grp, (np.abs(odd) > np.abs(even))[M & (dp != dm)])
        # odd part, under +d, against the up path through a shut gate: -sign(u)
        su = sg(u)[..., None]; sgg = sg(g)[..., None]
        updom = np.broadcast_to((np.abs(u) > 4 * np.abs(g))[..., None], dp.shape)
        gdom = np.broadcast_to((np.abs(g) > 4 * np.abs(u))[..., None], dp.shape)
        add("odd_vs_minus_u_updom", grp, (sg(odd) == -su)[M & updom & (odd != 0)])
        add("odd_vs_plus_u_updom", grp, (sg(odd) == su)[M & updom & (odd != 0)])
        add("odd_vs_g_gdom", grp, (sg(odd) == sgg)[M & gdom & (odd != 0)])
        # sign flip candidates under the gate-face push
        add("flip_all_sign_gate", grp, (sg(dgf) == -sign)[nz])
        add("plus_push_flip_gate", grp, (sg(dm) == sign)[M & (dm != 0) & (np.broadcast_to(gate, dp.shape) > 0)])
out = {}
for (k, g), (n, d) in sorted(acc.items()):
    out[f"{k}:{g}"] = {"pooled": round(n / max(d, 1), 4), "n": d, "layer_median": round(float(np.nanmedian(per[(k, g)])), 4)}
print(json.dumps(out, indent=1))
np.save("/home/eric/bin/ai/theodolite2/experiments/results/atlas_wiring_check/outputs/z2_per.npy",
        {f"{k}:{g}": v for (k, g), v in per.items()}, allow_pickle=True)

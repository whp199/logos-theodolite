# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
"""Adjudicator: the stored check scored at the 31B's measured operating point (z1's g0, u0 at the last word-piece).
- validates z1 against the stored unpushed reads (CPU packed vs the GPU build);
- where the readers sit: g0 > 0, -0.75 < g0 <= 0, g0 <= -0.75;
- sign_gate by regime; use_gate at the true op (alpha0 = gelu'(g0) u0, beta0 = gelu(g0)) against the as-built use_gate;
- per-layer medians, linked vs null."""
import sys, json, math
sys.path.insert(0, "/home/eric/bin/ai/theodolite2")
import numpy as np

W = "/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b"
z = np.load("/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/wiring_check.npz", allow_pickle=True)
qs = [np.load(f"{W}/z1_out/q{q}.npz") for q in range(4)]
G = np.stack([q["g_last"] for q in qs])  # [Q, L, N]
U = np.stack([q["u_last"] for q in qs])
A = np.stack([q["a_last"] for q in qs])
c = math.sqrt(2 / math.pi)
gelu = lambda x: 0.5 * x * (1 + np.tanh(c * (x + 0.044715 * x ** 3)))
def dgelu(x):
    t = np.tanh(c * (x + 0.044715 * x ** 3))
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * c * (1 + 3 * 0.044715 * x * x)
print("act vs gelu(g)u at last:", float(np.abs(A - gelu(G) * U).max()), flush=True)
sg = lambda x: np.where(x > 0, 1, np.where(x < 0, -1, 0))
acc, per = {}, {}
def add(key, ok):
    a = acc.setdefault(key, [0, 0]); a[0] += int(ok.sum()); a[1] += int(ok.size)
    per.setdefault(key, []).append(float(ok.mean()) if ok.size else np.nan)
val = {"a0_sign_agree": [], "a0_corr": []}
for l in [int(x) for x in z["layers"]]:
    reads = z[f"L{l}_reads"][..., 0]
    rd = z[f"L{l}_readers"]; lk = z[f"L{l}_linked"]; s = z[f"L{l}_sign"].astype(int); ga = z[f"L{l}_gate"].astype(int)
    g = z[f"L{l}_g"]; u = z[f"L{l}_u"]; op = z[f"L{l}_op"]
    valid = rd[..., 0] >= 0
    j = np.where(valid, rd[..., 0], 0); b = np.where(valid, rd[..., 1], 0)
    g0 = np.moveaxis(G[:, j, b], 0, -1)  # [W, R, Q]
    u0 = np.moveaxis(U[:, j, b], 0, -1)
    a0 = reads[..., 0]
    a0m = gelu(g0) * u0
    vm = np.broadcast_to(valid[..., None], a0.shape) & np.isfinite(a0)
    val["a0_sign_agree"].append(float((sg(a0m) == sg(a0))[vm & (a0 != 0)].mean()))
    val["a0_corr"].append(float(np.corrcoef(a0m[vm], a0[vm])[0, 1]))
    dp = reads[..., 1] - reads[..., 0]; dm = reads[..., 2] - reads[..., 0]
    gate = ga[..., None]; sign = s[..., None]
    dgf = np.where(gate > 0, dp, dm)
    al0, be0 = dgelu(g0) * u0, gelu(g0)
    k_true = sg(al0 * g[..., None] + be0 * u[..., None])
    face = np.where(a0 > 0, 0, 1)
    al = np.where(face == 0, op[:, :, None, 0, 0], op[:, :, None, 1, 0]); be = np.where(face == 0, op[:, :, None, 0, 1], op[:, :, None, 1, 1])
    k_built = sg(al * g[..., None] + be * u[..., None])
    reg = np.where(g0 > 0, "open", np.where(g0 > -0.7518, "shut_above_min", "below_min"))
    for grp, m in (("linked", lk & valid), ("null", ~lk & valid)):
        M = np.broadcast_to(m[..., None], dp.shape)
        nz = M & (dgf != 0)
        for r in ("open", "shut_above_min", "below_min"):
            add(f"share_{r}:{grp}", (reg == r)[M])
            add(f"sign_gate[{r}]:{grp}", (sg(dgf) == sign)[nz & (reg == r)])
            add(f"use_gate_true[{r}]:{grp}", (sg(dgf) == gate * k_true)[nz & (reg == r)])
            add(f"beta_built_sign_ok[{r}]:{grp}", (sg(be) == sg(be0))[M & (reg == r) & np.isfinite(be)])
            add(f"alpha_built_sign_ok[{r}]:{grp}", (sg(al) == sg(al0))[M & (reg == r) & np.isfinite(al)])
        add(f"sign_gate:{grp}", (sg(dgf) == sign)[nz])
        h = nz & np.isfinite(al) & np.isfinite(be)
        add(f"use_gate_built:{grp}", (sg(dgf) == gate * k_built)[h])
        add(f"use_gate_true:{grp}", (sg(dgf) == gate * k_true)[nz])
        add(f"use_plus_true:{grp}", (sg(dp) == k_true)[M & (dp != 0)])
        # where the sign from rest and the sign in use at the true op disagree
        add(f"rest_vs_true_use_agree:{grp}", (sign == gate * k_true)[M])
        add(f"g0_median:{grp}", np.array([np.median(g0[M])]) > 0)  # placeholder for share only
        per.setdefault(f"g0_q:{grp}", []).append(np.percentile(g0[M], [10, 25, 50, 75, 90]).round(3).tolist())
out = {"validation": {"a0_sign_agree_median": round(float(np.median(val["a0_sign_agree"])), 4),
                      "a0_sign_agree_min": round(float(np.min(val["a0_sign_agree"])), 4),
                      "a0_corr_median": round(float(np.median(val["a0_corr"])), 4), "a0_corr_min": round(float(np.min(val["a0_corr"])), 4)}}
for k, (n, d) in sorted(acc.items()):
    if k.startswith("g0_median"):
        continue
    out[k] = {"pooled": round(n / max(d, 1), 4), "n": d, "layer_median": round(float(np.nanmedian(per[k])), 4)}
for grp in ("linked", "null"):
    q = np.array(per[f"g0_q:{grp}"])
    out[f"g0_percentiles_10_25_50_75_90_median_over_layers:{grp}"] = np.median(q, 0).round(3).tolist()
print(json.dumps(out, indent=1))
np.save("/home/eric/bin/ai/theodolite2/experiments/results/atlas_wiring_check/outputs/z3_per.npy", per, allow_pickle=True)
# per-layer table of the decisive pair
layers = [int(x) for x in z["layers"]]
print("layer  sign_gate L/N   use_built L/N   use_true L/N   share g0<=-0.75 L/N")
for i, l in enumerate(layers):
    print(f"{l:3d}  {per['sign_gate:linked'][i]:.3f}/{per['sign_gate:null'][i]:.3f}  {per['use_gate_built:linked'][i]:.3f}/{per['use_gate_built:null'][i]:.3f}"
          f"  {per['use_gate_true:linked'][i]:.3f}/{per['use_gate_true:null'][i]:.3f}  {per['share_below_min:linked'][i]:.3f}/{per['share_below_min:null'][i]:.3f}")

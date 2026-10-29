# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
"""The wiring check's sign shares again, over the changes that are not exactly zero (the check's own shares count a zero
change as a sign not held), per layer, linked readers against the null, at the last word-piece."""
import json, sys
import numpy as np

z = np.load(sys.argv[1], allow_pickle=False)
layers = [int(x) for x in z["layers"]]
sg = lambda x: np.where(x > 0, 1, np.where(x < 0, -1, 0))
out = {}
for l in layers:
    sh = {k[len(f"L{l}_"):]: z[k] for k in z.files if k.startswith(f"L{l}_")}
    reads = sh["reads"]
    valid = sh["readers"][..., 0] >= 0
    d_plus = reads[:, :, :, 1, 0] - reads[:, :, :, 0, 0]
    d_minus = reads[:, :, :, 2, 0] - reads[:, :, :, 0, 0]
    gate = sh["gate"][:, :, None].astype(np.int64)
    d_gate = np.where(gate > 0, d_plus, d_minus)
    sign = np.broadcast_to(sh["sign"][:, :, None].astype(np.int64), d_plus.shape)
    face = np.where(reads[:, :, :, 0, 0] > 0, 0, 1)
    op = sh["op"]
    al = np.take_along_axis(np.broadcast_to(op[:, :, None, :, 0], face.shape + (2,)), face[..., None], -1)[..., 0]
    be = np.take_along_axis(np.broadcast_to(op[:, :, None, :, 1], face.shape + (2,)), face[..., None], -1)[..., 0]
    kappa = sg(al * sh["g"][:, :, None] + be * sh["u"][:, :, None])
    has_op = np.isfinite(al) & np.isfinite(be)
    row = {}
    for name, m in (("linked", sh["linked"] & valid), ("null", ~sh["linked"] & valid)):
        dg, s_ = d_gate[m], sign[m]
        nz = dg != 0
        k_, h_ = kappa[m], has_op[m]
        g_ = np.broadcast_to(sh["gate"][m][:, None], dg.shape)
        use_ok = nz & h_ & (k_ != 0)
        row[name] = {
            "zero_share": round(float((~nz).mean()), 3),
            "rest_gate_nonzero": round(float((sg(dg) == s_)[nz].mean()), 3) if nz.any() else None,
            "use_gate_nonzero": round(float((sg(dg) == g_ * k_)[use_ok].mean()), 3) if use_ok.any() else None,
            "n_nonzero": int(nz.sum()),
        }
    out[l] = row
print(json.dumps(out))

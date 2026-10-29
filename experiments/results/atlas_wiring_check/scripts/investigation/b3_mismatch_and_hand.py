# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
"""(a) The 31B's situation on the stand-in: op taken where the readers fire with their gates open (+1 shift, as 94.5% of
the 31B's stored events have g > 0), the reads taken where they sit shut (-1 shift). (b) The two-neuron hand calculation."""
import sys
exec(open('/home/eric/bin/ai/theodolite2/experiments/results/atlas_wiring_check/scripts/investigation/b2_passages_op.py').read().split('for shift in (0.0, -1.0, +1.0):')[0])
op_open, _ = passages_op(+1.0)
reads, G0, U0, *_ = measure(0.25, -1.0)
sign = np.where(g * u > 0, 1, -1).astype(np.int8); gate = np.where(g > 0, 1, -1).astype(np.int8)
tops = np.argsort(-np.abs(g * u), axis=1)[:, :24]
linked = np.zeros((W, R), bool); np.put_along_axis(linked, tops, True, axis=1)
sh = {"reads": reads, "readers": np.broadcast_to(np.array(readers_one)[None], (W, R, 2)), "linked": linked, "sign": sign, "gate": gate,
      "g": g, "u": u, "op": np.broadcast_to(op_open[None], (W, R, 2, 2)).copy(), "strength": np.abs(g * u)}
s = ab.wiring_check_summary(sh)["last"]
print("(a) op from open-gate events, reads at shut gates:")
for grp in ("linked", "null"):
    print(f"  {grp:6s} " + ", ".join(f"{k} {s[grp][k]:.3f}" for k in ("sign_plus", "sign_gate", "use_plus", "use_gate", "same_way")))

print("\n(b) two neurons: writer A, reader B, weight products g = +1, u = +1 (P = +1: sign +, gate +); the push on A's gate face")
print("    (+d) moves B's gate and up pre-activations by +e each (e = c d = 0.1); B's read is act = gelu_tanh(g) u")
G = lambda x: float(gelu(np.float64(x))); dG = lambda x: float(dgelu(np.float64(x)))
e = 0.1
for g0, u0, where in ((0.0, 0.0, "at rest (the reading from rest)"), (1.0, 0.5, "gate open"), (-1.0, 0.5, "gate shut, u0 > 0"),
                      (-1.0, -0.5, "gate shut, u0 < 0 (act0 > 0: read as its + face)"), (-0.4, -0.5, "gate half shut, u0 < 0")):
    a0 = G(g0) * u0
    dp = G(g0 + e) * (u0 + e) - a0
    dm = G(g0 - e) * (u0 - e) - a0
    al, be = dG(g0) * u0, G(g0)
    print(f"  {where:50s} g0 {g0:+.1f} u0 {u0:+.1f}: act0 {a0:+.4f}; +d -> {dp:+.4f}, -d -> {dm:+.4f}; first order "
          f"e(alpha g + beta u) = {e * (al + be):+.4f} [gate path {e * al:+.4f}, up path {e * be:+.4f}]; sign from rest +, "
          f"measured on the gate face {'+' if dp > 0 else '-'}")
print("  the in-use op the 31B's check used for a reader sitting with act0 > 0 is its + face's strongest events, e.g. g0 = +3, u0 = +2:")
g0, u0 = 3.0, 2.0
print(f"    alpha {dG(g0) * u0:+.3f}, beta {G(g0):+.3f} -> predicts {'+' if dG(g0) * u0 + G(g0) > 0 else '-'}; at the reader's real point "
      f"(g0 -1, u0 -0.5) alpha {dG(-1) * -0.5:+.3f}, beta {G(-1):+.3f} -> {'+' if dG(-1) * -0.5 + G(-1) > 0 else '-'}")

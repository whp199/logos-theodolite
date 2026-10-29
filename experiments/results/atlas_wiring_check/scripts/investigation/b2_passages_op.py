# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
"""The second reading's operating point: the summary's use_* with op from each reader's strongest stored events (as the
passages index keeps them, `passage_events` + `operating_points`), against op at the reader's true point; with the
readers' gates where they sit, shifted down, and shifted up. Also: how often a reader sits on GeGLU's negative lobe
(g0 < 0: the sign of act0 is then the opposite of u0's, while every strong stored event has g0 > 0)."""
import sys
sys.argv = ['x']
exec(open('/home/eric/bin/ai/theodolite2/experiments/results/atlas_wiring_check/scripts/investigation/b1_stand_in.py').read().split('report("check\'s dose", 0.25)')[0])
from atlas_stand_in import CORPUS
texts = [[tok.bos_token_id] + tok.encode(t, add_special_tokens=False) for t in CORPUS]

def passages_op(shift):
    hooks = [lm.layers[j].mlp.gate_proj.register_forward_hook(lambda _m, _a, o, s=shift: o + s) for j in range(WL + 1, L)] if shift else []
    try:
        E = ao.passage_events(lm, texts, k=16)
    finally:
        for h in hooks: h.remove()
    op = np.full((R, 2, 2), np.nan, np.float32)
    regime = {}
    for fi, f in enumerate(("pos", "neg")):
        al, be = ao.operating_points(E[f"{f}_g"], E[f"{f}_u"], E[f"{f}_win"] >= 0)
        for r, (j, b) in enumerate(readers_one):
            op[r, fi] = (al[j, b], be[j, b])
        ok = E[f"{f}_win"][WL + 1:] >= 0
        regime[f] = float((E[f"{f}_g"][WL + 1:][ok] > 0).mean())
    return op, regime

for shift in (0.0, -1.0, +1.0):
    reads, G0, U0, DG, DU, ACT = measure(0.25, shift)
    op1, regime = passages_op(shift)
    last = reads[..., 0]
    sign = np.where(g * u > 0, 1, -1).astype(np.int8); gate = np.where(g > 0, 1, -1).astype(np.int8)
    tops = np.argsort(-np.abs(g * u), axis=1)[:, :24]
    linked = np.zeros((W, R), bool); np.put_along_axis(linked, tops, True, axis=1)
    sh = {"reads": reads, "readers": np.broadcast_to(np.array(readers_one)[None], (W, R, 2)), "linked": linked, "sign": sign,
          "gate": gate, "g": g, "u": u, "op": np.broadcast_to(op1[None], (W, R, 2, 2)).copy(), "strength": np.abs(g * u)}
    s = ab.wiring_check_summary(sh)["last"]
    a0 = last[0, :, :, 0]
    print(f"\nshift {shift:+g}: stored strong events with g > 0: pos face {regime['pos']:.2f}, neg face {regime['neg']:.2f}; "
          f"readers at the last word-piece with g0 < 0: {(G0 < 0).mean():.2f}, and of those sign(act0) != sign(u0): "
          f"{(np.sign(a0) != np.sign(U0))[G0 < 0].mean():.2f}")
    for grp in ("linked", "null"):
        print(f"  {grp:6s} op = strongest stored events (as the passages index): " + ", ".join(f"{k} {s[grp][k]:.3f}" for k in ("sign_plus", "sign_gate", "use_plus", "use_gate")))

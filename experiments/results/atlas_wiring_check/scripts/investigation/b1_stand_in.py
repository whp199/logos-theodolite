# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
"""The wiring check's conventions on the CPU stand-in, by computation.

1. The push and g, u: under +d along writer A's write direction, does the reader's gate pre-activation move by +c g
   and its up pre-activation by +c u (c > 0), g and u from `pair_products`? (sign agreement, correlation)
2. The read: is pushed_reads' value gelu_tanh(g) * u of the reader (the hooks on gate_proj and up_proj agree)?
3. The check's own `wiring_check_summary` on these pairs, with `op` set to the reader's TRUE operating point at the
   last word-piece (alpha = gelu'(g0) u0, beta = gelu(g0)): use_* should then be near 1 if every convention holds.
4. sign_gate split by where the reader's gate sits (g0 > 0 vs g0 < 0).
5. The same measurement with every reader layer's gate shifted by a constant (a hook adding s to gate_proj's output:
   the weights, g, u, P, the push and the read are all unchanged; only where the readers sit moves).
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
ROOT = "/home/eric/bin/ai/theodolite2"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/scripts")

import numpy as np
import torch
import torch.nn.functional as F

import backend.atlas_ops as ao
from backend import atlas_build as ab
from backend.causal_check import language_model
from backend.census import PROMPTS
from atlas_stand_in import stand_in_hf, stand_in_tokenizer

torch.set_grad_enabled(False)
tok = stand_in_tokenizer()
hf = stand_in_hf(tok, torch.float32)
lm = language_model(hf)
w = ao.Weights(lm)
L, N = w.n_layers, w.d_ff


def ids_of(text):
    s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok.encode(s, add_special_tokens=False)


prompts = [ids_of(q) for q in PROMPTS.values()]
Q = len(prompts)
WL = 1  # writer layer
writers = list(range(0, N, 4))  # 32 writers
readers_one = [(j, b) for j in range(WL + 1, L) for b in range(N)]
readers = [readers_one] * len(writers)
ceil = w.write_ceiling(WL, writers).numpy()
g, u = ao.pair_products(w, WL, writers, readers)  # [W, R]
W, R = g.shape


def measure(dose, shift=0.0):
    """reads [W,R,Q,3] at the last word-piece (pushed_reads), and the readers' g0, u0 unpushed there and dg, du under +d."""
    hooks = []
    if shift:
        for j in range(WL + 1, L):
            hooks.append(lm.layers[j].mlp.gate_proj.register_forward_hook(lambda _m, _a, o, s=shift: o + s))
    try:
        reads = ao.pushed_reads(lm, w, prompts, [(WL, A) for A in writers], readers, dose * ceil, bos_id=tok.bos_token_id, batch=2 * W + 1)
        # the same rows again, with the gate and up pre-activations seen
        dirs = w.write_dirs(WL, writers)
        G0 = np.zeros((R, Q)); U0 = np.zeros((R, Q)); DG = np.zeros((W, R, Q)); DU = np.zeros((W, R, Q)); ACT = np.zeros((W, R, Q, 3))
        for q, ids in enumerate(prompts):
            B = 1 + 2 * W
            x = torch.tensor([ids] * B)
            keep = ao._Keep()
            keep.mask = ao.keep_mask(x, None, tok.bos_token_id, True)
            iv = ao.Interventions(rows=B)
            iv.adds[WL] = torch.zeros((B, w.d_model))
            for k in range(W):
                iv.adds[WL][1 + 2 * k] = float(dose * ceil[k]) * dirs[k]
                iv.adds[WL][2 + 2 * k] = -float(dose * ceil[k]) * dirs[k]
            seen = {}
            hs = []
            for j in range(WL + 1, L):
                m = lm.layers[j].mlp
                hs.append(m.gate_proj.register_forward_hook(lambda _m, _a, o, j=j: seen.__setitem__(("g", j), o[:, -1].clone())))
                hs.append(m.up_proj.register_forward_hook(lambda _m, _a, o, j=j: seen.__setitem__(("u", j), o[:, -1].clone())))
                hs.append(m.down_proj.register_forward_pre_hook(lambda _m, a, j=j: seen.__setitem__(("a", j), a[0][:, -1].clone())))
            try:
                with ao.installed(lm, iv, keep):
                    lm(input_ids=x, use_cache=False)
            finally:
                for h in hs:
                    h.remove()
            for r, (j, b) in enumerate(readers_one):
                gg, uu, aa = seen[("g", j)][:, b].numpy(), seen[("u", j)][:, b].numpy(), seen[("a", j)][:, b].numpy()
                # the gate hook (shift) runs before ours? ours registered after: sees the shifted output
                G0[r, q], U0[r, q] = gg[0], uu[0]
                DG[:, r, q] = gg[1::2] - gg[0]
                DU[:, r, q] = uu[1::2] - uu[0]
                ACT[:, r, q, 0] = aa[0]
                ACT[:, r, q, 1] = aa[1::2]
                ACT[:, r, q, 2] = aa[2::2]
    finally:
        for h in hooks:
            h.remove()
    return reads, G0, U0, DG, DU, ACT


def sg(x):
    return np.where(x > 0, 1, np.where(x < 0, -1, 0))


def gelu(x):
    return F.gelu(torch.as_tensor(x, dtype=torch.float64), approximate="tanh").numpy()


def dgelu(x):
    return ao.gelu_tanh_grads(torch.as_tensor(x, dtype=torch.float64)).numpy()


def report(tag, dose, shift=0.0):
    reads, G0, U0, DG, DU, ACT = measure(dose, shift)
    last = reads[..., 0]  # [W,R,Q,3]
    print(f"\n=== {tag}: dose {dose} x ceiling, gate shift {shift:+g} ===")
    # 2. the read
    print(f"  read = gelu_tanh(g) u at the last word-piece: max |pushed_reads - gelu(g0) u0| = "
          f"{np.abs(last[0, :, :, 0] - gelu(G0) * U0).max():.2e}; pushed_reads vs our own hooked rows {np.abs(last - ACT).max():.2e}")
    # 1. the push moves g by +c g and u by +c u
    gg = np.broadcast_to(g[:, :, None], DG.shape)
    uu = np.broadcast_to(u[:, :, None], DU.shape)
    first = np.array([j for j, _ in readers_one]) == WL + 1
    for name, mask in (("next layer", first), ("all later layers", np.ones(R, bool))):
        m = np.broadcast_to(mask[None, :, None], DG.shape)
        cg = np.corrcoef(DG[m], gg[m])[0, 1]
        cu = np.corrcoef(DU[m], uu[m])[0, 1]
        print(f"  push (+d) vs weights, {name}: sign(dg) = sign(g) at {(sg(DG) == sg(gg))[m].mean():.3f}, corr {cg:.3f}; "
              f"sign(du) = sign(u) at {(sg(DU) == sg(uu))[m].mean():.3f}, corr {cu:.3f}")
    # 3. the summary, with op = the true operating point at the last word-piece, per (reader, question): the summary takes
    # one op per (W, R) and face; we evaluate it per question by calling it on one question at a time
    sign = np.where(g * u > 0, 1, -1).astype(np.int8)
    gate = np.where(g > 0, 1, -1).astype(np.int8)
    tops = np.argsort(-np.abs(g * u), axis=1)[:, :24]
    linked = np.zeros((W, R), bool)
    np.put_along_axis(linked, tops, True, axis=1)
    rows = []
    for q in range(Q):
        al, be = dgelu(G0[:, q]) * U0[:, q], gelu(G0[:, q])
        op = np.full((W, R, 2, 2), np.nan, np.float32)
        face = np.where(last[0, :, q, 0] > 0, 0, 1)
        op[:, np.arange(R), face, 0] = al
        op[:, np.arange(R), face, 1] = be
        sh = {"reads": reads[:, :, q:q + 1], "readers": np.broadcast_to(np.array(readers_one)[None], (W, R, 2)), "linked": linked,
              "sign": sign, "gate": gate, "g": g, "u": u, "op": op, "strength": np.abs(g * u)}
        rows.append(ab.wiring_check_summary(sh)["last"])
    for grp in ("linked", "null"):
        keys = ("sign_plus", "sign_gate", "use_plus", "use_gate", "same_way")
        print(f"  summary ({grp:6s}, top-24 |P| or the rest), op = the TRUE one: " +
              ", ".join(f"{k} {np.mean([r[grp][k] for r in rows]):.3f}" for k in keys))
    # 4. sign_gate by where the gate sits
    dp, dm = last[..., 1] - last[..., 0], last[..., 2] - last[..., 0]
    dg_ = np.where(gate[:, :, None] > 0, dp, dm)
    ok = sg(dg_) == sign[:, :, None]
    G0b = np.broadcast_to(G0[None], ok.shape)
    for lo, hi in ((0, 99), (-0.75, 0), (-99, -0.75)):
        m = (G0b > lo) & (G0b <= hi)
        print(f"  sign_gate where g0 in ({lo}, {hi}]: {ok[m].mean():.3f}  (share of pairs {m.mean():.2f})")
    print(f"  sign_gate overall {ok.mean():.3f}; readers with g0 < 0: {(G0 < 0).mean():.2f}")
    return reads, G0, U0


report("check's dose", 0.25)
report("small dose (first order)", 0.01)
report("check's dose, readers' gates shifted down", 0.25, shift=-1.0)
report("check's dose, readers' gates shifted up", 0.25, shift=+1.0)

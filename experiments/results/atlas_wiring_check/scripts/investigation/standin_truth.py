# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
"""Ground truth for the wiring check on the CPU stand-in.

Builds the stand-in's stats, passages and wiring as scripts/check_atlas_index.py does, runs build_wiring_check on every
layer, then re-runs every push itself with hooks on gate_proj / up_proj / pre_feedforward_layernorm / down_proj of
each reader's layer, so that for every (writer, reader, question) we have, at the last word-piece:
  g0, u0 (the reader's own gate and up pre-activations, unpushed), the measured change of g, u and act under +d and -d,
  the direct-path prediction of the change of g, u from the weights (the index's g, u) through the residual stream,
  and the first-order prediction of the change of act.

--variant native : the stand-in as built (|g0| ~ 0.2, near rest)
--variant closed : gate rows scaled x SCALE and shifted so the typical g0 is about -SHIFT (a trained GeGLU's closed gates)
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
SCR = Path(__file__).resolve().parent
os.environ["THEODOLITE_ATLAS_DIR"] = str(SCR / "atlas_dir")  # nothing lands in the repo's atlas folder
sys.path.insert(0, "/home/eric/bin/ai/theodolite2")
sys.path.insert(0, "/home/eric/bin/ai/theodolite2/scripts")

import numpy as np
import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)

from atlas_stand_in import stand_in  # noqa: E402
from backend import atlas_build as ab  # noqa: E402
from backend import atlas_ops as ao  # noqa: E402
from backend.atlas_index import AtlasIndex  # noqa: E402
from backend.causal_check import language_model  # noqa: E402
from backend.census import PROMPTS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="native", choices=["native", "closed", "closed_last"])
ap.add_argument("--scale", type=float, default=12.0)
ap.add_argument("--shift", type=float, default=1.5)
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--sd", type=float, default=0.0)
ap.add_argument("--dose", type=float, default=0.25)
a = ap.parse_args()

out = SCR / f"run_{a.variant}"
out.mkdir(parents=True, exist_ok=True)
for p in out.glob("*"):
    if p.is_file():
        p.unlink()
import shutil  # noqa: E402

for p in out.glob("*"):
    if p.is_dir():
        shutil.rmtree(p)

rt, vault = stand_in()
lm = language_model(rt.hf_model)
w = ao.Weights(lm, rt)
L, N, D = w.n_layers, w.d_ff, w.d_model
tok = rt.tokenizer
print(f"stand-in {L} layers x {N} neurons, D={D}, dtype {rt.hf_model.dtype}, act {lm.config.hidden_activation}")
print("layer scalars", [round(float(x), 3) for x in w.scalars])

texts = [rt.compose_prompt(t, chat=True) for t in ("Who are you?", "Tell me about Orange.")] + list(ao.HEAD_TEXTS)
wins = ab.windows(rt, texts, 24, 400)
qs = [rt.tokenize_prompt(rt.compose_prompt(v, chat=True)) for v in PROMPTS.values()]


def mlp_inputs(layer: int, seqs):
    """The normalized input x of layer's mlp at every word-piece past the start mark, over `seqs`."""
    got = []
    h = lm.layers[layer].mlp.register_forward_pre_hook(lambda _m, args: got.append(args[0][0, 1:].float().clone()))
    try:
        for s in seqs:
            lm(input_ids=torch.tensor([s]), use_cache=False)
    finally:
        h.remove()
    return torch.cat(got, 0)


if a.variant == "closed":
    # a trained GeGLU's gates sit mostly shut: emulate with a larger gate and a "bias" along the stream's mean
    # direction (Gemma has no bias, but a trained stream carries large common directions that act as one)
    for j in range(L):
        X = mlp_inputs(j, wins)
        m = X.mean(0)
        Wg = lm.layers[j].mlp.gate_proj.weight
        Wg.mul_(a.scale)
        g_now = X @ Wg.T  # [T, N]
        # shift each row by -c m / |m|^2 so that its mean over the tokens is -shift
        c = (g_now.mean(0) + a.shift)  # [N]
        Wg.sub_(c[:, None] * (m / (m @ m))[None, :])
        g_new = X @ Wg.T
        print(f"closed: layer {j}: g0 over windows median {g_new.median():.2f}, share > 0 {float((g_new > 0).float().mean()):.2f}, "
              f"share < -0.75 {float((g_new < -0.7518).float().mean()):.2f}")

if a.variant == "closed_last":
    # each reader's gate set, at the census questions' last word-piece, to about -shift (+- sd per neuron): the
    # regime the 31B's numbers point to; the windows (stats, passages) still see the gates open on other tokens
    gen = torch.Generator().manual_seed(11)
    for j in range(L):
        Xw = mlp_inputs(j, wins)
        got = []
        h = lm.layers[j].mlp.register_forward_pre_hook(lambda _m, args: got.append(args[0][0, -1].float().clone()))
        try:
            for s_ in qs:
                lm(input_ids=torch.tensor([s_]), use_cache=False)
        finally:
            h.remove()
        Xl = torch.stack(got)  # [Q, D]
        Wg = lm.layers[j].mlp.gate_proj.weight
        Wg.mul_(a.scale)
        m = Xl.mean(0)
        target = -a.shift + a.sd * torch.randn(N, generator=gen)
        c = (Xl @ Wg.T).mean(0) - target
        Wg.sub_(c[:, None] * (m / (m @ m))[None, :])
        gl, gw_ = Xl @ Wg.T, Xw @ Wg.T
        print(f"closed_last: layer {j}: g0 at the questions' last word-piece median {gl.median():.2f} (> 0 {float((gl > 0).float().mean()):.2f}, "
              f"< -0.75 {float((gl < -0.7518).float().mean()):.2f}); over the windows median {gw_.median():.2f} (> 0 {float((gw_ > 0).float().mean()):.2f})")

ab.build_stats(rt, out, wins, "check", log=lambda *_: None)
idx = AtlasIndex(out)
s = idx.get("act_scale")
ab.build_passages(rt, out, wins, "check", k=5, sample=3, extremes=(s["pos"], s["neg"]), extremes_source="the stats index", seed=7, chunk=4,
                  log=lambda *_: None)
f = ao.link_scale(w.scalars, [float(x) for x in s["mlp_rms_q"][:, 1]], [float(x) for x in s["mid_rms_q"][:, 1]])
ab.build_wiring(rt, out, f, scale_source="check", k=a.k, chunk=40, log=lambda *_: None)
ab.build_wiring_check(rt, out, f, layers=list(range(L - 1)), writers=f"random:{N}", dose=a.dose, seed=3, batch=65, scale_source="check",
                      log=lambda x: print("  ", x) if "layer" in x and "writers" in x else None)
z = dict(np.load(out / "wiring_check.npz"))
meta = json.loads(str(z["meta"]))

# ------------------------------------------------------------------------------------------- the pushes, re-run by hand
gelu = lambda x: F.gelu(x, approximate="tanh")  # noqa: E731
dgelu = ao.gelu_tanh_grads


def rerun(l: int):
    ws = z[f"L{l}_writers"].tolist()
    rdr = z[f"L{l}_readers"]
    lengths = z[f"L{l}_lengths"]
    W, R = rdr.shape[:2]
    dirs = w.write_dirs(l, ws)  # [W, D]
    # per (w, r, q, row 0/1/2): g, u, act at the last word-piece; and the residual h entering layer j's pre-ffn norm (row 0)
    G = np.full((W, R, len(qs), 3), np.nan)
    U = np.full_like(G, np.nan)
    A = np.full_like(G, np.nan)
    Hmid = {}  # (j, q) -> [D] unpushed residual at the last word-piece entering pre_feedforward_layernorm
    DH = np.full((W, R, len(qs), 2, D), np.nan, np.float32)  # change of that residual under +d, -d
    per = 32
    for q, ids in enumerate(qs):
        for s0 in range(0, W, per):
            chunk = list(range(s0, min(s0 + per, W)))
            B = 1 + 2 * len(chunk)
            x = torch.tensor([ids] * B)
            keep = ao._Keep()
            keep.mask = ao.keep_mask(x, None, tok.bos_token_id, True)
            iv = ao.Interventions(rows=B)
            iv.adds[l] = torch.zeros((B, D))
            for k, aa in enumerate(chunk):
                iv.adds[l][1 + 2 * k] = float(lengths[aa]) * dirs[aa]
                iv.adds[l][2 + 2 * k] = -float(lengths[aa]) * dirs[aa]
            cap = {}
            hooks = []
            for j in range(l + 1, L):
                lay = lm.layers[j]
                hooks.append(lay.mlp.gate_proj.register_forward_hook(lambda _m, _a, o, j=j: cap.__setitem__(("g", j), o[:, -1].clone())))
                hooks.append(lay.mlp.up_proj.register_forward_hook(lambda _m, _a, o, j=j: cap.__setitem__(("u", j), o[:, -1].clone())))
                hooks.append(lay.mlp.down_proj.register_forward_pre_hook(lambda _m, args, j=j: cap.__setitem__(("a", j), args[0][:, -1].clone()), prepend=True))
                hooks.append(lay.pre_feedforward_layernorm.register_forward_pre_hook(lambda _m, args, j=j: cap.__setitem__(("h", j), args[0][:, -1].clone())))
            try:
                with ao.installed(lm, iv, keep):
                    lm(input_ids=x, use_cache=False)
            finally:
                for h in hooks:
                    h.remove()
            for j in range(l + 1, L):
                Hmid[(j, q)] = cap[("h", j)][0].numpy()
            for k, aa in enumerate(chunk):
                for r in range(R):
                    j, b = int(rdr[aa, r, 0]), int(rdr[aa, r, 1])
                    if j < 0:
                        continue
                    rows = [0, 1 + 2 * k, 2 + 2 * k]
                    G[aa, r, q] = cap[("g", j)][rows, b].numpy()
                    U[aa, r, q] = cap[("u", j)][rows, b].numpy()
                    A[aa, r, q] = cap[("a", j)][rows, b].numpy()
                    hh = cap[("h", j)]
                    DH[aa, r, q, 0] = (hh[1 + 2 * k] - hh[0]).numpy()
                    DH[aa, r, q, 1] = (hh[2 + 2 * k] - hh[0]).numpy()
    return ws, G, U, A, Hmid, DH


sg = np.sign
tot = {}
rows_print = []
for l in range(L - 1):
    ws, G, U, A, Hmid, DH = rerun(l)
    rdr, lk = z[f"L{l}_readers"], z[f"L{l}_linked"]
    gw, uw, sign, gate, op = z[f"L{l}_g"], z[f"L{l}_u"], z[f"L{l}_sign"].astype(int), z[f"L{l}_gate"].astype(int), z[f"L{l}_op"]
    reads = z[f"L{l}_reads"][..., 0]  # [W,R,Q,3] last word-piece
    valid = rdr[..., 0] >= 0
    W, R = valid.shape
    Q = len(qs)
    lengths = z[f"L{l}_lengths"]
    v = (w.down_cols(l, ws) * w.norm(l, "post_feedforward")[None]).numpy()  # [W, D] (unnormalized write vector)
    vnorm = np.linalg.norm(v, axis=1)
    # direct-path prediction of the change of g and u at the reader (first order through the pre-ffn RMS norm)
    dg_dir = np.full((W, R, Q), np.nan)
    du_dir = np.full((W, R, Q), np.nan)
    dg_rot = np.full((W, R, Q), np.nan)  # the norm's own part: -eps g0
    for aa in range(W):
        for r in range(R):
            j, b = int(rdr[aa, r, 0]), int(rdr[aa, r, 1])
            if j < 0:
                continue
            carried = float(np.prod([float(w.scalars[m]) for m in range(l + 1, j)])) if j > l + 1 else 1.0
            dh = carried * float(lengths[aa]) * v[aa] / vnorm[aa]
            wpre = w.norm(j, "pre_feedforward").numpy()
            eps_n = w.eps(j, "pre_feedforward")
            wg = lm.layers[j].mlp.gate_proj.weight[b].numpy()
            wu = lm.layers[j].mlp.up_proj.weight[b].numpy()
            for q in range(Q):
                h = Hmid[(j, q)].astype(np.float64)
                rms = math.sqrt(float((h * h).mean()) + eps_n)
                hh = h / np.linalg.norm(h)
                dperp = dh - hh * (hh @ dh) * (float((h * h).mean()) / (float((h * h).mean()) + eps_n))
                dx = wpre * dperp / rms
                dg_dir[aa, r, q] = wg @ dx
                du_dir[aa, r, q] = wu @ dx
                dg_rot[aa, r, q] = -(hh @ dh) / np.linalg.norm(h) * G[aa, r, q, 0]
    g0, u0 = G[..., 0], U[..., 0]
    dGp, dGm = G[..., 1] - g0, G[..., 2] - g0
    dUp, dUm = U[..., 1] - u0, U[..., 2] - u0
    dAp, dAm = A[..., 1] - A[..., 0], A[..., 2] - A[..., 0]
    oG, oU, oA = (dGp - dGm) / 2, (dUp - dUm) / 2, (dAp - dAm) / 2
    eA = (dAp + dAm) / 2
    t_g0, t_u0 = torch.tensor(g0), torch.tensor(u0)
    alpha0 = (dgelu(t_g0) * t_u0).numpy()
    beta0 = gelu(t_g0).numpy()
    GW, UW = gw[..., None], uw[..., None]
    S, GT = sign[..., None], gate[..., None]
    kap_true = sg(alpha0 * GW + beta0 * UW)  # sign in use at the reader's actual operating point (weights' g, u)
    fo_meas = alpha0 * oG + beta0 * oU  # first order at the actual point with the measured dg, du
    # the check's own kappa (op from passages, face of the unpushed read)
    face = np.where(reads[..., 0] > 0, 0, 1)
    al = np.take_along_axis(np.broadcast_to(op[:, :, None, :, 0], face.shape + (2,)), face[..., None], -1)[..., 0]
    be = np.take_along_axis(np.broadcast_to(op[:, :, None, :, 1], face.shape + (2,)), face[..., None], -1)[..., 0]
    kap_chk = sg(al * GW + be * UW)
    face_u = np.where(u0 > 0, 0, 1)
    al_u = np.take_along_axis(np.broadcast_to(op[:, :, None, :, 0], face_u.shape + (2,)), face_u[..., None], -1)[..., 0]
    be_u = np.take_along_axis(np.broadcast_to(op[:, :, None, :, 1], face_u.shape + (2,)), face_u[..., None], -1)[..., 0]
    kap_u0 = sg(al_u * GW + be_u * UW)
    dGate = np.where(GT > 0, dAp, dAm)
    for name, m in (("linked", lk & valid), ("null", ~lk & valid)):
        M = np.broadcast_to(m[..., None], g0.shape) & (dAp != 0) & (dAm != 0)
        T = tot.setdefault(name, {})

        def add(key, cond, sel=None):
            sel = M if sel is None else (M & sel)
            T.setdefault(key, [0, 0])
            T[key][0] += int((cond & sel).sum())
            T[key][1] += int(sel.sum())

        add("reads == gelu(g) u (rel 1e-4)", np.abs(A[..., 0] - (gelu(t_g0) * t_u0).numpy()) <= 1e-4 * np.abs(A[..., 0]) + 1e-7)
        add("check reads == my reads (unpushed,+,-)", np.all(np.abs(reads - A) <= 1e-5 * np.abs(A) + 1e-7, -1))
        add("sign(odd dg) == sign(g)", sg(oG) == sg(GW))
        add("sign(odd du) == sign(u)", sg(oU) == sg(UW))
        add("sign(direct dg) == sign(g)", sg(dg_dir) == sg(GW))
        add("|odd dg - direct| < 0.5|direct|", np.abs(oG - dg_dir) < 0.5 * np.abs(dg_dir))
        add("sign(odd dact) == sign(1st order, measured dg du)", sg(oA) == sg(fo_meas))
        add("sign(odd dact) == kappa at the real point", sg(oA) == kap_true)
        add("sign(odd dact) == check's kappa", sg(oA) == kap_chk, np.isfinite(al))
        add("check's kappa == kappa at the real point", kap_chk == kap_true, np.isfinite(al))
        add("sign_plus: sign(dact+) == sign", sg(dAp) == S)
        add("sign_gate: sign(dact gate face) == sign", sg(dGate) == S)
        add("use_gate: sign(dact gate face) == gate kappa_chk", sg(dGate) == GT * kap_chk, np.isfinite(al))
        add("sign_gate with kappa at real point", sg(dGate) == GT * kap_true)
        add("sign(even dact) == sign", sg(eA) == S)
        add("|odd| > |even|", np.abs(oA) > np.abs(eA))
        add("toward zero under the gate-face push", sg(dGate) == -sg(A[..., 0]))
        add("toward zero under both +d and -d", (sg(dAp) == -sg(A[..., 0])) & (sg(dAm) == -sg(A[..., 0])))
        add("use_gate with face from u0 (not act)", sg(dGate) == GT * kap_u0, np.isfinite(al_u))
        for lab, sel in (("g0 > 0", g0 > 0), ("-0.75 < g0 <= 0", (g0 <= 0) & (g0 > -0.7518)), ("g0 <= -0.75", g0 <= -0.7518)):
            add(f"share of pairs with {lab}", np.ones_like(M), sel | ~sel)  # placeholder count of all
            T.setdefault(f"n[{lab}]", [0, 0])
            T[f"n[{lab}]"][0] += int((M & sel).sum())
            T[f"n[{lab}]"][1] += int(M.sum())
            add(f"sign_gate | {lab}", sg(dGate) == S, sel)
            add(f"sign(even) == sign | {lab}", sg(eA) == S, sel)
            add(f"use_gate (check kappa) | {lab}", sg(dGate) == GT * kap_chk, sel & np.isfinite(al))
            add(f"check face is u0's sign | {lab}", np.where(face == 0, 1, -1) == sg(u0), sel)
            add(f"sign(alpha_chk) == sign(alpha0) | {lab}", sg(al) == sg(alpha0), sel & np.isfinite(al))
            add(f"sign(beta_chk) == sign(beta0) | {lab}", sg(be) == sg(beta0), sel & np.isfinite(al))
    # a handful of links, printed
    if l in (0, 2):
        for aa in range(2):
            for r in range(3):
                if not lk[aa, r]:
                    continue
                q = 0
                rows_print.append(dict(l=l, A=ws[aa], j=int(rdr[aa, r, 0]), B=int(rdr[aa, r, 1]), q=q, g=float(gw[aa, r]), u=float(uw[aa, r]),
                                       sign=int(sign[aa, r]), gate=int(gate[aa, r]), g0=float(g0[aa, r, q]), u0=float(u0[aa, r, q]),
                                       dg_meas=float(oG[aa, r, q]), dg_direct=float(dg_dir[aa, r, q]), du_meas=float(oU[aa, r, q]),
                                       du_direct=float(du_dir[aa, r, q]), dact_plus=float(dAp[aa, r, q]), dact_minus=float(dAm[aa, r, q]),
                                       first_order=float(fo_meas[aa, r, q]), kappa_true=int(kap_true[aa, r, q]), kappa_check=int(kap_chk[aa, r, q]),
                                       op_face=["+", "-"][int(face[aa, r, q])], op_alpha_beta=(float(al[aa, r, q]), float(be[aa, r, q]))))

print(f"\n=== variant {a.variant}: summary of build_wiring_check (median over layers, last word-piece) ===")
for key in ("sign_plus", "sign_gate", "use_plus", "use_gate", "same_way", "rho"):
    vals = {nm: [meta["summary"][str(l)]["last"][nm][key] for l in range(L - 1) if meta["summary"][str(l)]["last"][nm][key] is not None] for nm in ("linked", "null")}
    print(f"  {key:10s} linked {np.median(vals['linked']):.3f}  null {np.median(vals['null']):.3f}")
print("  moved_over_null", [meta["summary"][str(l)]["last"]["moved_over_null"] for l in range(L - 1)])
print(f"\n=== variant {a.variant}: ground truth, pooled over layers and questions (nonzero changes) ===")
for key in tot["linked"]:
    if key.startswith("share of pairs"):
        continue
    def fr(t):
        return f"{t[0] / t[1]:.3f} (n={t[1]})" if t[1] else "-"
    print(f"  {key:55s} linked {fr(tot['linked'][key]):18s} null {fr(tot['null'][key])}")
print("\n=== a handful of links (question 'who') ===")
for rr in rows_print:
    print("  ", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in rr.items()})
json.dump({"tot": tot, "rows": rows_print}, open(out / "truth.json", "w"), indent=1, default=str)

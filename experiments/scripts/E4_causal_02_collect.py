# E4_causal_02: collect residual means (+ unscaled attn/ffn writes) at self / background / user tokens for
# 8 self-conditions x 24 passages, and at referent / continuation / last tokens for 36 third-person pairs.
# PRE-STATED PREDICTIONS (written before running):
#  P1 u_l (woman-man persona diff at self tokens) is stable across passage halves (cos>0.8) at L15-50, weaker early.
#  P2 cos(self-derived u_l, third-person u_l) moderately positive (0.3-0.7) at mid layers -> shared gender feature.
#  P3 cos(u_l, J++ pullback of token gender dir) small (<0.15) at L20-45, rising toward L58 (consistent with VJP check).
#  P4 default / nameless AI / Gemma self-states project near the persona midpoint (0.35-0.65), Sarah(AI)>0.5>James(AI);
#     no strong female coding of the default self. J++ readout ordering similar but shifted by lens background bias.
#  P5 Gender is written into self tokens mostly by attention (copying persona info from system prompt) in early-mid
#     layers, with FFN amplification; the tool's buggy FFN split is dominated by the (s-1)*mid shrinkage term.
exec(open("./experiments/scripts/E4_causal_common.py").read())
t0 = time.time()
D = 5376
PUNCT = set(list(".,!?;:'\"-()"))

def turn_spans(ids):
    strs = lab.strs(ids)
    mt = lab.model_turn_pos(ids)
    ch = [i for i in range(mt, len(ids)) if strs[i] == "<channel|>"][0]
    up = [i for i in range(1, mt) if strs[i-1] == "<|turn>" and strs[i].startswith("user")][-1]
    uend = [i for i in range(up, mt) if strs[i] == "<turn|>"][0]
    selfpos = [i for i in range(ch+1, len(ids)) if strs[i].strip() in SELF_TOKS]
    bgpos = [i for i in range(ch+1, len(ids)) if strs[i].strip() not in SELF_TOKS and strs[i].strip() not in PUNCT and strs[i].strip() != ""]
    upos = [i for i in range(up+2, uend) if strs[i].strip() not in PUNCT and strs[i].strip() != ""]
    return selfpos, bgpos, upos

def run_self():
    items = self_items()
    N = len(items)
    out = {k: torch.zeros(N, 60, D) for k in ("self_post", "bg_post", "user_post", "self_A", "self_M")}
    out["self_pre0"] = torch.zeros(N, D)
    rd = {f"{w}_{kind}_{sc}": torch.zeros(N, 60) for w in ("self", "bg") for kind in ("jpp", "naive")
          for sc in ("pron", "noun", "lse_pron", "neuter_vs_human")}
    norms = torch.zeros(N, 60)
    nself = []
    BS = 16
    for b0 in range(0, N, BS):
        batch = items[b0:b0+BS]
        ids_l, pos_l = [], []
        for it in batch:
            q, a = QA[it["p"]]
            ids = lab.ids(lab.chat(q, system=it["sys"], assistant=a))
            ids_l.append(ids); pos_l.append(turn_spans(ids))
            nself.append(len(pos_l[-1][0]))
        def red(kind, l, h):
            for j, (sp, gp, up) in enumerate(pos_l):
                n = b0 + j
                hs = h[j, sp].float()
                if kind == "post":
                    hg = h[j, gp].float()
                    out["self_post"][n, l] = hs.mean(0).cpu()
                    out["bg_post"][n, l] = hg.mean(0).cpu()
                    out["user_post"][n, l] = h[j, up].float().mean(0).cpu()
                    norms[n, l] = h[j, 1:len(ids_l[j])].float().norm(dim=-1).mean().cpu()
                    for w, hh in (("self", h[j, sp]), ("bg", h[j, gp])):
                        for kind2 in ("jpp", "naive"):
                            sc = lab.gender_scores(lab.lens(hh, l, kind2))
                            for k in ("pron", "noun", "lse_pron", "neuter_vs_human"):
                                rd[f"{w}_{kind2}_{k}"][n, l] = sc[k].float().mean().cpu()
                elif kind == "A":
                    out["self_A"][n, l] = hs.mean(0).cpu()
                elif kind == "M":
                    out["self_M"][n, l] = hs.mean(0).cpu()
                elif kind == "pre0":
                    out["self_pre0"][n] = hs.mean(0).cpu()
        batched_hook_forward(lab, ids_l, red, want_parts=True)
    meta = {"items": items, "nself": nself}
    torch.save({**out, **rd, "norms": norms, "meta": meta}, f"{E4_CACHE}/self.pt")
    return meta, rd

def run_third():
    items = third_items()
    N = len(items)
    out = {k: torch.zeros(N, 60, D) for k in ("ref_post", "cont_post", "last_post")}
    rd = {f"{w}_jpp_{sc}": torch.zeros(N, 60) for w in ("ref", "cont", "last") for sc in ("pron", "noun")}
    norms = torch.zeros(N, 60)
    beh = []
    BS = 24
    for b0 in range(0, N, BS):
        batch = items[b0:b0+BS]
        ids_l, pos_l = [], []
        for it in batch:
            ids = lab.ids(it["text"]); npre = len(lab.ids(it["prefix"]))
            ids_l.append(ids); pos_l.append((npre - 1, list(range(npre, len(ids))), len(ids) - 1))
        def red(kind, l, h):
            if kind != "post":
                return
            for j, (rp, cp, lp) in enumerate(pos_l):
                n = b0 + j
                out["ref_post"][n, l] = h[j, rp].float().cpu()
                out["cont_post"][n, l] = h[j, cp].float().mean(0).cpu()
                out["last_post"][n, l] = h[j, lp].float().cpu()
                norms[n, l] = h[j, 1:len(ids_l[j])].float().norm(dim=-1).mean().cpu()
                for w, hh in (("ref", h[j, rp:rp+1]), ("cont", h[j, cp]), ("last", h[j, lp:lp+1])):
                    sc = lab.gender_scores(lab.lens(hh, l, "jpp"))
                    for k in ("pron", "noun"):
                        rd[f"{w}_jpp_{k}"][n, l] = sc[k].float().mean().cpu()
        lg = batched_hook_forward(lab, ids_l, red)
        pg = pron_logodds_from_logits(lab, lg)
        for j in range(len(batch)):
            beh.append({k: float(v[j]) for k, v in pg.items()})
    meta = {"items": items, "beh": beh}
    torch.save({**out, **rd, "norms": norms, "meta": meta}, f"{E4_CACHE}/third.pt")
    return meta

m1, rd = run_self()
print("self done", time.time() - t0)
m3 = run_third()
print("third done", time.time() - t0)
# quick summary: J++ self pron band L30-39 per condition
import numpy as np
conds = sorted(set(it["cond"] for it in m1["items"]))
for c in conds:
    idx = [i for i, it in enumerate(m1["items"]) if it["cond"] == c]
    v = rd["self_jpp_pron"][idx][:, 30:40].mean().item()
    print(c, "self J++ pron L30-39 =", round(v, 3))
bF = [b["lse_pron"] for b, it in zip(m3["beh"], m3["items"]) if it["cond"] == "F"]
bM = [b["lse_pron"] for b, it in zip(m3["beh"], m3["items"]) if it["cond"] == "M"]
print("third-person next-token lse_pron F", np.mean(bF), "M", np.mean(bM))
result = {"self_meta": m1, "third_meta": m3, "elapsed": time.time() - t0}

# E3_lens_validation misc checks: (1) pole-vector mean-square vs the eps-hack threshold, top-10 set overlap and
# displayed top-1 prob under eps=1e-12 (tool) vs eps=1e-6 (model); (2) what the J++/JLens images of the mean
# residual decode to, separately for prose and chat corpora (is the L42 'Google/Gemma' decode a chat artefact?).
import json
exec(open("./experiments/scripts/E3_lens_validation_data.py").read())
from backend.theodolite_math import unembed_gemma4
MU = torch.load("./experiments/results/E3_lens_validation_mu.pt")
ids = lab.ids(PROSE[0][1])
L = [2, 10, 20, 30, 40, 50, 58]
acts = await lab.capture(ids, points=("mlp_act",), layers=L)
res = {"poles": {}, "mu_decode": {}}
for i, l in enumerate(L):
    a = acts["mlp_act"][i, -1].to(lab.dev).float()
    idx = torch.topk(a.abs(), 15).indices
    w = rt.slice_w_down(l, idx).float() * rt._post_ffn_norm_weights[l].float()
    v = torch.sign(a[idx]).unsqueeze(-1) * w
    pv = (v.to(torch.bfloat16) @ lab.J[l].T).float()
    ms_raw = v.pow(2).mean(-1); ms_pv = pv.pow(2).mean(-1)
    lg_h = unembed_gemma4(pv.to(torch.bfloat16), rt.final_norm_weight, rt.lm_head_weight)
    x = pv; lg_s = torch.tanh(((x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * rt.final_norm_weight.float()).to(torch.bfloat16) @ rt.lm_head_weight.T).float() / 30) * 30
    th = torch.topk(lg_h, 10, -1).indices; ts = torch.topk(lg_s, 10, -1).indices
    overlap = (th.unsqueeze(-1) == ts.unsqueeze(-2)).any(-1).float().mean(-1)
    res["poles"][str(l)] = {"ms_raw_median": float(ms_raw.median()), "ms_jpp_median": float(ms_pv.median()), "ms_jpp_min": float(ms_pv.min()),
                            "ms_jpp_max": float(ms_pv.max()), "top10_set_overlap": float(overlap.mean()), "top1_same": float((th[:, 0] == ts[:, 0]).float().mean()),
                            "p_top1_tool": float(F.softmax(lg_h, -1).max(-1).values.mean()), "p_top1_eps1e-6": float(F.softmax(lg_s, -1).max(-1).values.mean())}
    print(l, res["poles"][str(l)])
for nm in ("mu_prose", "mu_chat"):
    M = MU[nm]
    for kind in ("jpp", "jlens"):
        for l in (36, 39, 42, 45):
            res["mu_decode"][f"{nm}|{kind}|L{l}"] = lab.topk(lab.lens(M[l:l + 1], l, kind), 10)
            print(nm, kind, l, [t for t, p in res["mu_decode"][f"{nm}|{kind}|L{l}"]])
json.dump(res, open("./experiments/results/E3_lens_validation_12_misc_out.json", "w"), indent=1)
result = res

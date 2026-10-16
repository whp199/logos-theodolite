# V6_quiet_dims_05_dose -- sensitivity dose-response: KL(clean || perturbed) of next-token distributions after adding a fixed
# random push (same vector at every non-BOS position) to resid_post at ONE layer.  Push families (all at equal RAW norm):
#   iso        Gaussian over all 5376 dims
#   quiet10    Gaussian over the per-layer bottom-10%-std set (E6's definition)
#   pers       Gaussian over the 181 'persistent quiet' dims (deep-quiet at every layer L42-57)
#   loud       Gaussian over 538 random dims drawn from the top-50%-std set
#   white_all  std_i * xi_i over all dims (natural per-dim scale; diagonal-whitened)
#   white_q10  std_i * xi_i restricted to the bottom-10% set (quiet dims at their OWN natural scale)
#   white_loud std_i * xi_i restricted to the 'loud' set
#   natdiff    difference of two natural token states at that layer (on-manifold)
# raw-norm doses f * median||h_l||, f in {0.001,0.003,0.01,0.03,0.1,0.3}; 2 random seeds; 12 prompts (6 chat, 6 prose).
# PRE-REGISTERED PREDICTIONS
#  P-a (C7 claim): at L40/L50 a quiet-confined push collapses the model (KL > 5 nats) at f ~ 0.01 while a loud-confined push of
#      the same raw norm is harmless (KL < 0.1) even at f = 0.03-0.1.  Refuted if quiet and loud curves are within 3x in KL.
#  P-b (my alternative): the dose-response curves of ALL families collapse onto one curve when plotted against the gain-weighted
#      size ||w_in[l+1] * delta|| / rms(h) (and/or the z-size ||delta/std||), i.e. 'quiet' matters only through read gain /
#      natural scale.  White_q10 (quiet dims pushed at their own natural scale) will be no more destructive per unit z-size
#      than white_loud.  Refuted if at matched z-size white_q10 KL exceeds white_loud KL by >10x.
#  P-c: at L20 (where the persistent dims are LOUD early-layer dims with low read gain) 'pers' pushes are harmless.
exec(open("./experiments/scripts/V6_quiet_dims_common.py").read())
LAYERS = globals().get("V6_LAYERS", [40, 50])
t0 = time.time()
S = torch.load(V6_R + "V6_quiet_dims_stats.pt")
Ts = torch.load(V6_R + "V6_quiet_dims_samples.pt")
std_all = S["std_all"].float()
pers = torch.zeros(5376, dtype=torch.bool)
deepq = torch.stack([std_all[l] < 0.05 * std_all[l].median() for l in range(60)])
pers = deepq[42:58].all(0)
assert int(pers.sum()) == 181, int(pers.sum())
chats = json.load(open(V6_R + "E3_lens_validation_chats.json"))
prompts = [lab.ids(lab.chat(chats[i]["q"], assistant=chats[i]["a"]))[:110] for i in (0, 2, 5, 12, 15, 20)]
prompts += [lab.ids(PROSE[i][1])[:110] for i in (0, 3, 7, 10, 18, 22)]
lm = lab.hf.model.language_model if hasattr(lab.hf.model, "language_model") else lab.hf.model
from backend.theodolite_math import FINAL_LOGIT_SOFTCAP
B = len(prompts); T = max(len(x) for x in prompts)
inp = torch.zeros(B, T, dtype=torch.long); att = torch.zeros(B, T, dtype=torch.long)
for b, x in enumerate(prompts):
    inp[b, :len(x)] = torch.tensor(x); att[b, :len(x)] = 1
inp = inp.to(lab.dev); att = att.to(lab.dev)
# last 16 positions of each sequence
bi = torch.tensor([b for b, x in enumerate(prompts) for p in range(len(x) - 16, len(x))], device=lab.dev)
ti = torch.tensor([p for b, x in enumerate(prompts) for p in range(len(x) - 16, len(x))], device=lab.dev)
last = torch.tensor([16 * b + 15 for b in range(B)], device=lab.dev)


@torch.no_grad()
def logprobs():
    hN = lm(input_ids=inp, attention_mask=att, use_cache=False).last_hidden_state[bi, ti]   # final-normed
    z = (hN.to(lab.WU.dtype) @ lab.WU.T).float()
    z = torch.tanh(z / FINAL_LOGIT_SOFTCAP) * FINAL_LOGIT_SOFTCAP
    return F.log_softmax(z, -1)


lab.clear_hooks()
LP0 = logprobs()
# sanity: compare with lab.next_logits on prompt 0
chk = F.log_softmax(lab.next_logits(prompts[0]), -1)
print("sanity max|dlogp| vs next_logits:", float((chk - LP0[last[0]]).abs().max()), flush=True)
top0 = LP0.argmax(-1)
H0 = None
res = {"layers": LAYERS, "rows": []}
for l in LAYERS:
    sd = std_all[l]
    hs = Ts["H"][:, l].float()
    medn = float(hs.norm(dim=-1).median())
    rms = float((hs.pow(2).mean(-1).sqrt()).median())
    q10 = sd < torch.quantile(sd, 0.10)
    loudpool = torch.where(sd > sd.median())[0]
    win = S["g_w_in"][l + 1].float().abs(); wpre = S["g_w_preffn"][l + 1].float().abs()
    for seed in (0, 1):
        g = torch.Generator().manual_seed(1000 * l + seed)
        xi = torch.randn(5376, generator=g)
        loud = torch.zeros(5376, dtype=torch.bool); loud[loudpool[torch.randperm(len(loudpool), generator=g)[:538]]] = True
        i, j = torch.randint(0, hs.shape[0], (2,), generator=g).tolist()
        fam = {"iso": xi.clone(), "quiet10": xi * q10, "pers": xi * pers, "loud": xi * loud,
               "white_all": xi * sd, "white_q10": xi * sd * q10, "white_loud": xi * sd * loud, "natdiff": hs[i] - hs[j]}
        for name, v in fam.items():
            u = v / v.norm()
            pred = {"gain_in": float((win * u).norm() / rms), "gain_pre": float((wpre * u).norm() / rms),
                    "z": float((u / sd).norm()), "q10_share": float((u.pow(2) * q10).sum()), "pers_share": float((u.pow(2) * pers).sum())}
            ug = u.to(lab.dev)
            for f in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3):
                a = f * medn
                lab.clear_hooks()
                def fn(h, ug=ug, a=a):
                    hf = h.float()
                    add = (a * ug).view(1, 1, -1).expand_as(hf).clone()
                    add[:, 0] = 0.0
                    return (hf + add).to(h.dtype)
                lab.add_resid_hook(l, fn)
                LP = logprobs()
                lab.clear_hooks()
                kl = (LP0.exp() * (LP0 - LP)).sum(-1)
                ent = -(LP.exp() * LP).sum(-1)
                res["rows"].append({"layer": l, "seed": seed, "fam": name, "f": f, "norm": a, **pred,
                                    "kl_mean": float(kl.mean()), "kl_last": float(kl[last].mean()),
                                    "kl_med": float(kl.median()), "top1": float((LP.argmax(-1) == top0).float().mean()),
                                    "ent": float(ent.mean())})
        print("layer", l, "seed", seed, "t", round(time.time() - t0, 1), flush=True)
json.dump(res, open(V6_R + f"V6_quiet_dims_05_dose_L{'_'.join(map(str, LAYERS))}.json", "w"))
# compact print
for r in res["rows"]:
    if r["seed"] == 0 and r["f"] in (0.003, 0.01, 0.03, 0.1):
        print(r["layer"], r["fam"].ljust(10), r["f"], "KL", round(r["kl_mean"], 3), "top1", round(r["top1"], 2), "gain_in", round(r["gain_in"] * r["norm"], 2), "z", round(r["z"] * r["norm"], 1))
result = {"n": len(res["rows"]), "t": round(time.time() - t0, 1)}

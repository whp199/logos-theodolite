E4_TAG = "a"
E4_GEN_SETTINGS = [("none", "L25-39", 0.0), ("u_self", "L25-39", 0.02), ("u_self", "L25-39", -0.02)]
# E4_causal_12: steering + sampling on the DEFAULT model (no system prompt).
# Prompts: "Describe yourself in the third person." and "If you were human, would you be a man or a woman?"
# N=16 samples per prompt per setting, temperature 1.0, top_p 0.95, top_k 64, 64 new tokens.
# Steering: h[l][t] += alpha * n_l * u_hat_l for l in block at all non-BOS positions (prefill + generated tokens).
# Fluency: mean token log-prob of each sample under the UNSTEERED model, distinct-3gram ratio.
exec(open("./experiments/scripts/E4_causal_common.py").read())
t0 = time.time()
SETTINGS = globals().get("E4_GEN_SETTINGS", [("none", "L25-39", 0.0)])
TAG = globals().get("E4_TAG", "a")
NS = 16; MAXNEW = 64
DC = torch.load(f"{E4_CACHE}/dirs_causal.pt"); DIRS = DC["dirs"]; RN = DC["resid_norm"]
BLOCKS = {"L10-24": list(range(10, 25)), "L25-39": list(range(25, 40)), "L40-54": list(range(40, 55))}
PROMPTS = {"describe3p": lab.chat("Describe yourself in the third person."),
           "human_gender": lab.chat("If you were human, would you be a man or a woman?")}
keys = list(PROMPTS)
ids_l = [lab.ids(PROMPTS[k]) for k in keys]
T = max(len(x) for x in ids_l)
inp = torch.zeros(len(ids_l), T, dtype=torch.long); att = torch.zeros(len(ids_l), T, dtype=torch.long)
for b, x in enumerate(ids_l):
    inp[b, T - len(x):] = torch.tensor(x); att[b, T - len(x):] = 1
bos_col = torch.tensor([T - len(x) for x in ids_l]).repeat_interleave(NS)  # rows after num_return_sequences expansion
def install(dname, layers, alpha):
    for l in layers:
        v = (alpha * float(RN[l]) * DIRS[dname][l]).to(lab.dev, torch.float32)
        def fn(h, v=v):
            hf = h.float()
            if hf.shape[1] > 1:
                add = v.view(1, 1, -1).expand(hf.shape[0], hf.shape[1], -1).clone()
                add[torch.arange(hf.shape[0]), bos_col[:hf.shape[0]].to(hf.device)] = 0.0
            else:
                add = v.view(1, 1, -1)
            return (hf + add).to(h.dtype)
        lab.add_resid_hook(l, fn)
@torch.no_grad()
def score(prompt_ids, gen_ids_list):
    """mean log-prob per generated token under the clean model (one sequence at a time; slice logits to save memory)."""
    out = []
    P0 = len(prompt_ids)
    for g in gen_ids_list:
        if len(g) == 0: out.append(float("nan")); continue
        x = torch.tensor([prompt_ids + g], device=lab.dev)
        lg = lab.hf(input_ids=x, use_cache=False).logits[0, P0 - 1:P0 - 1 + len(g)].float()
        lp = F.log_softmax(lg, -1)
        out.append(float(lp.gather(-1, torch.tensor(g, device=lab.dev).view(-1, 1)).mean()))
        del lg, lp
    return out
res = {"settings": {}}
for dname, bname, alpha in SETTINGS:
    lab.clear_hooks()
    if dname != "none" and alpha != 0:
        install(dname, BLOCKS[bname], alpha)
    torch.manual_seed(1234)
    try:
        g = lab.hf.generate(input_ids=inp.to(lab.dev), attention_mask=att.to(lab.dev), max_new_tokens=MAXNEW, do_sample=True,
                            temperature=1.0, top_p=0.95, top_k=64, num_return_sequences=NS, pad_token_id=0)
    finally:
        lab.clear_hooks()
    gen = g[:, T:].cpu()
    texts, gids = {}, {}
    for pi, k in enumerate(keys):
        rows = gen[pi * NS:(pi + 1) * NS]
        gl = []
        for r in rows.tolist():
            # cut at first end-of-turn / eos / pad
            cut = len(r)
            for j, t in enumerate(r):
                s_ = lab.tok.decode([t])
                if t in (0, 1, 106) or s_ in ("<turn|>", "<eos>", "<pad>"):
                    cut = j; break
            gl.append(r[:cut])
        gids[k] = gl
        texts[k] = [lab.tok.decode(x, skip_special_tokens=True) for x in gl]
    lp = {k: score(ids_l[pi], gids[k]) for pi, k in enumerate(keys)}
    res["settings"][f"{dname}|{bname}|{alpha}"] = {"texts": texts, "clean_logprob": lp}
    print(dname, bname, alpha, round(time.time() - t0, 1), "|", texts["describe3p"][0][:120].replace("\n", " "), "||", texts["human_gender"][0][:100].replace("\n", " "))
json.dump(res, open(f"{E4_DIR}/E4_causal_12_steer_gen_{TAG}.json", "w"))
result = {"ok": True, "elapsed": time.time() - t0}

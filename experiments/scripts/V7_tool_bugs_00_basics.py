# V7_tool_bugs_00_basics: environment facts needed by later jobs (cheap, < 30 s).
# - to_tokens ids for chat vs raw strings with prepend_bos True/False (the tool's tokenization path)
# - model config facts (per-layer-input / MoE flags, activation), layer scalars
# - hook-point identities: mlp_act == down_proj input; resid_pre[l+1] == resid_post[l]; post = s*(mid + ffn_out)
# - free VRAM
# Pre-registered: chat strings tokenized with prepend_bos=True start [2,2,...] (double BOS); raw strings with
# prepend_bos=False have NO BOS at all (so "just pass prepend_bos=False" is not a safe fix).
exec(open("./experiments/scripts/V7_tool_bugs_common.py").read())
from interp_engine import Address
out = {}
chat = chat_prompts()[2]
raw = RAW_PROMPTS[0]
em = rt.eager_model
for tag, s in (("chat", chat), ("raw", raw)):
    for pb in (True, False):
        ids = [int(t) for t in em.to_tokens(s, prepend_bos=pb)[0].tolist()]
        out[f"{tag}_prepend{pb}"] = {"first5": ids[:5], "first5_str": [rt.tokenizer.decode([i]) for i in ids[:5]], "n": len(ids)}
    out[f"{tag}_labids_first5"] = lab.ids(s)[:5]
out["tokenizer_prepends_bos"] = getattr(em, "tokenizer_prepends_bos", None)
out["default_prepend_bos"] = getattr(em, "default_prepend_bos", None)
cfg = rt.hf_model.config
tc = getattr(cfg, "text_config", cfg)
out["cfg"] = {k: str(getattr(tc, k, None)) for k in ("hidden_size_per_layer_input", "enable_moe_block", "hidden_activation",
                                                      "final_logit_softcapping", "num_hidden_layers", "rms_norm_eps", "intermediate_size")}
s = torch.stack([rt._decoder_layers[l].layer_scalar.float().cpu().reshape(-1)[0] for l in range(60)])
out["layer_scalar"] = [round(float(x), 4) for x in s]
out["vram_alloc_gb"] = torch.cuda.memory_allocated() / 2**30
free, total = torch.cuda.mem_get_info()
out["vram_free_gb"] = free / 2**30
out["vram_total_gb"] = total / 2**30

# hook identities on one short prompt
ids = lab.ids(chat)
pos = len(ids) - 1
L = [0, 5, 29, 41, 58]
caps = await em.capture(ids, [Address(p, l) for l in L for p in ("resid_pre", "resid_mid", "resid_post", "mlp_act")] +
                        [Address("resid_pre", l + 1) for l in L if l + 1 < 60])
store = {}
hs = []
for l in L:
    mlp = rt.get_mlp_module(l)
    def pre(mod, args, l=l):
        store[("dp_in", l)] = args[0][0].float().cpu()
    def outh(mod, args, o, l=l):
        store[("dp_out", l)] = o[0].float().cpu()
    hs.append(mlp.down_proj.register_forward_pre_hook(pre))
    hs.append(mlp.down_proj.register_forward_hook(outh))
try:
    lab.hf(input_ids=torch.tensor([ids], device=lab.dev), use_cache=False)
finally:
    for h in hs:
        h.remove()
ident = {}
for l in L:
    ma = caps[Address("mlp_act", l)].float()
    dpi = store[("dp_in", l)]
    pre_, mid_, post_ = (caps[Address(p, l)].float() for p in ("resid_pre", "resid_mid", "resid_post"))
    sl = float(s[l])
    blk = rt._decoder_layers[l]
    y = store[("dp_out", l)].to(lab.dev)
    ffn_out = blk.post_feedforward_layernorm(y.to(torch.bfloat16)).float().cpu()
    recon = sl * (mid_ + ffn_out)
    d = {"mlp_act_vs_downproj_in_relerr": float((ma - dpi).norm() / dpi.norm()),
         "post_vs_s(mid+ffn)_relerr": float((post_ - recon).norm() / post_.norm())}
    if l + 1 < 60:
        nxt = caps[Address("resid_pre", l + 1)].float()
        d["pre_next_vs_post_relerr"] = float((nxt - post_).norm() / post_.norm())
    d["median_resid_post_norm_nonBOS"] = float(post_[1:].norm(dim=-1).median())
    d["resid_post_norm_BOS"] = float(post_[0].norm())
    ident[l] = d
out["identities"] = ident
fnw = rt.final_norm_weight.float()
out["fnw"] = {"min": float(fnw.min()), "max": float(fnw.max()), "median": float(fnw.median()),
              "n_lt_0.1": int((fnw < 0.1).sum()), "n_neg": int((fnw < 0).sum())}
print(json.dumps(out, indent=1)[:6000])
result = out

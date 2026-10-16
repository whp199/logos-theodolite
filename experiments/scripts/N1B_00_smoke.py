# N1B step 0: smoke test for the activation-recording machinery (no predictions; infrastructure only).
# Checks: tied embeddings / embed scale, down_proj pre-hook input shape, batched inner-model forward timing+memory.
import time, json
t0 = time.time()
out = {}
lm = lab.hf.model.language_model if hasattr(lab.hf.model, "language_model") else lab.hf.model
emb = lm.embed_tokens
out["embed_type"] = type(emb).__name__
out["embed_scale"] = float(getattr(emb, "embed_scale", torch.tensor(-1.0)).float()) if hasattr(emb, "embed_scale") else None
W_E = emb.weight
out["W_E_shape"] = list(W_E.shape)
out["tied_same_ptr"] = bool(W_E.data_ptr() == rt.lm_head_weight.data_ptr())
rows = torch.tensor([10, 1000, 50000], device=W_E.device)
out["tied_allclose"] = bool(torch.allclose(W_E[rows].float(), rt.lm_head_weight[rows].float()))
out["n_layers_list"] = len(lab.layers)
l = 30
mlp = lab.layers[l].mlp
out["mlp_type"] = type(mlp).__name__
out["mlp_children"] = [n for n, _ in mlp.named_children()]
out["act_fn"] = str(getattr(mlp, "act_fn", None))
out["has_pre_ffn"] = l in rt._pre_ffn_norm_weights
out["layer_children"] = [n for n, _ in lab.layers[l].named_children()]
free0, tot = torch.cuda.mem_get_info()
out["mem_free_before_GB"] = free0 / 1e9
seen = {}
idx = torch.tensor([1711, 2956, 3104], device=lab.dev)
def hook(mod, args):
    x = args[0]
    seen["shape"] = list(x.shape)
    seen["dtype"] = str(x.dtype)
    seen["sel"] = x[..., idx].float().cpu()
h = mlp.down_proj.register_forward_pre_hook(hook)
torch.cuda.reset_peak_memory_stats()
try:
    for B in (8, 16):
        T = 128
        inp = torch.randint(1000, 20000, (B, T))
        inp[:, 0] = 2
        att = torch.ones(B, T, dtype=torch.long)
        torch.cuda.synchronize(); t1 = time.time()
        with torch.no_grad():
            lm(input_ids=inp.to(lab.dev), attention_mask=att.to(lab.dev), use_cache=False)
        torch.cuda.synchronize()
        out[f"fwd_B{B}_s"] = time.time() - t1
        out[f"peak_B{B}_GB"] = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
finally:
    h.remove()
out["hook_seen"] = {k: v for k, v in seen.items() if k != "sel"}
out["sel_absmax"] = float(seen["sel"].abs().max())
out["alloc_GB"] = torch.cuda.memory_allocated() / 1e9
out["reserved_GB"] = torch.cuda.memory_reserved() / 1e9
out["elapsed"] = time.time() - t0
print(json.dumps(out, indent=1))
result = out

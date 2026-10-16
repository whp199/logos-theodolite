# Verify the applied Module-1 (layer_scalar) and double-BOS patches end-to-end, inside the shared daemon.
# The daemon process imported the ORIGINAL backend at startup (other sessions use it), so the patched modules are
# loaded side-by-side from disk under a temporary sys.modules swap, restored before this job returns.
# Expected: chat ids start [2, 105] (no double BOS); raw-text ids keep one BOS and outputs are unchanged;
# patched delta_ffn_J matches the true FFN write (cos ~1) where the old split does not; attn+ffn+rescale reconstructs
# the update; server packing + MCP scientist entry points run.
import importlib.util, sys, struct

REPO = "."
NAMES = ["backend.theodolite_math", "backend.engine_runtime", "backend.larql_vault", "backend.mcp_scientist", "backend.server"]
saved = {k: sys.modules.get(k) for k in NAMES}
old_tm = sys.modules["backend.theodolite_math"]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, f"{REPO}/{name.replace('.', '/')}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


try:
    tm, er, lv, ms, srv = (_load(n) for n in NAMES)
finally:
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v

prt = er.TheodoliteRuntime.__new__(er.TheodoliteRuntime)
prt.__dict__.update(rt.__dict__)          # same loaded model/lenses, patched methods
vault = lv.LarqlVaultEngine(prt)
ms.bind_runtime(prt, vault)

out = {}
chat_p = lab.chat("Describe yourself in one sentence.")
raw_p = "The number of legs on the animal that spins webs is"

# ---- 1. old vs patched sweeps (BOS + generation)
for tag, p in (("chat", chat_p), ("raw", raw_p)):
    o = await rt.run_theodolite_sweep(p, monument_concepts=("she", "he", "it"), max_new_tokens=8)
    n = await prt.run_theodolite_sweep(p, monument_concepts=("she", "he", "it"), max_new_tokens=8)
    out[tag] = {
        "old_ids_head": o.prompt_token_ids[:4], "new_ids_head": n.prompt_token_ids[:4],
        "old_gen": o.generated_text, "new_gen": n.generated_text,
        "old_n_tokens": len(o.prompt_token_ids), "new_n_tokens": len(n.prompt_token_ids),
        "ffn_dom_old": [round(float(o.causal_splits[l].ffn_dominance_ratio), 3) for l in range(60)],
        "ffn_dom_new": [round(float(n.causal_splits[l].ffn_dominance_ratio), 3) for l in range(60)],
    }
    out[tag]["ffn_dominance_class_flips"] = sum(
        (a > 0.5) != (b > 0.5) for a, b in zip(out[tag]["ffn_dom_old"], out[tag]["ffn_dom_new"]))
    if tag == "chat":
        new_bundle = n

# ---- 2. Module-1 correctness on the patched bundle vs ground-truth FFN write
ids = new_bundle.prompt_token_ids
pos = new_bundle.target_pos
caps = await lab.capture(ids, points=("resid_pre", "resid_mid", "resid_post"))
cos_new, cos_old, recon = [], [], []
for l in range(60):
    blk = lab.layers[l]
    J = lab.J[l]
    pre = caps["resid_pre"][l, pos].to(lab.dev)
    mid = caps["resid_mid"][l, pos].to(lab.dev)
    post = caps["resid_post"][l, pos].to(lab.dev)
    s = blk.layer_scalar.float()
    true_ffn = (s * blk.post_feedforward_layernorm(blk.mlp(blk.pre_feedforward_layernorm(mid[None]))).float())[0]
    true_ffn_J = (true_ffn.to(J.dtype) @ J.T).float()
    sp = new_bundle.causal_splits[l]
    cos_new.append(float(F.cosine_similarity(sp.delta_ffn_J.float(), true_ffn_J, dim=0)))
    so = old_tm.compute_causal_vector_split(pre, mid, post, J)
    cos_old.append(float(F.cosine_similarity(so.delta_ffn_J.float(), true_ffn_J, dim=0)))
    tot = ((post.float() - pre.float()).to(J.dtype) @ J.T).float()
    parts = sp.delta_attn_J.float() + sp.delta_ffn_J.float() + sp.delta_rescale_J.float()
    recon.append(float((parts - tot).norm() / tot.norm()))
out["module1"] = {
    "cos_new_min": round(min(cos_new), 4), "cos_new_median": round(sorted(cos_new)[30], 4),
    "cos_old_min": round(min(cos_old), 3), "cos_old_median": round(sorted(cos_old)[30], 3),
    "old_layers_cos_below_0.5": sum(c < 0.5 for c in cos_old),
    "recon_relerr_max": round(max(recon), 4),
    "layer_scalar_reported_L59": new_bundle.causal_splits[59].layer_scalar,
}

# ---- 3. Server path: full sweep request -> binary + JSON frames
req = srv.SweepRequest(prompt=chat_p, monuments=["she", "he", "it"], max_new_tokens=6)
binf, meta = await srv._execute_sweep_request(prt, vault, req)
magic = struct.unpack_from("<I", binf, 0)[0]
lay0 = None
for key in ("layers", "layer_plumes"):
    if isinstance(meta.get(key), list) and meta[key]:
        lay0 = meta[key][30]
        break
out["server"] = {"binary_bytes": len(binf), "magic_ok": magic == 0x5448454F, "json_keys": sorted(meta.keys())[:20],
                 "layer30_has_rescale": bool(lay0 and "rescale_norm" in lay0 and "layer_scalar" in lay0),
                 "layer30_sample": {k: lay0[k] for k in ("ffn_norm", "attn_norm", "rescale_norm", "layer_scalar", "ffn_dominance") if lay0 and k in lay0}}

# ---- 4. MCP scientist entry points (BOS fix sites)
tr = await ms.trace_downstream_readers(layer=40, neuron_idx=int(new_bundle.bipolar_poles[40].neuron_indices[0]), prompt=chat_p, top_k=3)
th = await ms.test_motif_hypothesis(layer=50, neuron_idx=5313,
                                    pos_prompts=[lab.chat("Tell me about my sister.", assistant="Your sister is kind, and"),
                                                 "My aunt said that"],
                                    neg_prompts=[lab.chat("Tell me about my brother.", assistant="Your brother is kind, and"),
                                                 "My uncle said that"])
out["mcp"] = {"trace_ok": len(tr["top_downstream_readers"]) == 3, "motif_verdict": th["verdict"], "motif_d": th["cohens_d"]}

for k, v in out.items():
    print(k, {kk: vv for kk, vv in v.items() if not str(kk).startswith("ffn_dom_")} if isinstance(v, dict) else v)
result = out

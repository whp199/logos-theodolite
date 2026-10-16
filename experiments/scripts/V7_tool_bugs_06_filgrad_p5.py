PIDX = 5
DST_LAYERS = list(range(2, 60, 2))
import gc as _gc; _gc.collect()
# V7_tool_bugs_06_filgrad: true local gradients d(g_j)/d(a_i), d(a_j)/d(a_i) (same position, all paths, autograd through
# the W4A16 recompute linears) for the full candidate set at a subset of destination layers. Set PIDX before exec.
# Linearized ablation prediction: G_gate = -a_i * dg_j/da_i, G_act = -a_i * da_j/da_i.  Compared offline with the
# measured zero-ablation effects from job 03 (same prompt, same poles -- deterministic sweep).
# PRE-REGISTERED: Spearman(|G_act|, |Delta a|) >= 0.8 (single-neuron ablations are near-linear); G beats D_act
# (indirect paths matter) by a small margin; both >> C_tool.
exec(open("./experiments/scripts/V7_tool_bugs_common.py").read())
import numpy as _np
cp = chat_prompts()
FIL_PROMPTS = [("raw_demo", RAW_PROMPTS[0], True), ("chat_gemma_self", cp[0], False), ("raw_gemma_friend", RAW_PROMPTS[1], True),
               ("chat_sarah", cp[1], False), ("chat_neighbor", cp[3], False), ("raw_nurse", RAW_PROMPTS[3], True),
               ("chat_doctor", cp[4], False), ("chat_atlas", cp[7], False)]
name, prompt, prepend = FIL_PROMPTS[PIDX]
DST_LAYERS = globals().get("DST_LAYERS", list(range(4, 60, 4)))
t0 = time.time()
bd = await rt.run_theodolite_sweep(prompt, monument_concepts=("she", "he", "it"), max_new_tokens=0,
                                   tracked_concepts=["she", "he"], prepend_bos=prepend)
ids = bd.prompt_token_ids; pos = bd.target_pos
pole_idx = torch.stack([bd.bipolar_poles[l].neuron_indices for l in range(60)]).to(lab.dev)
del bd
torch.cuda.empty_cache()
mlps = [rt.get_mlp_module(l) for l in range(60)]


class _Stop(Exception):
    pass


recs = {k: [] for k in ("l", "k", "lp", "j", "dgda", "dada", "a_src")}
for lp in DST_LAYERS:
    src_layers = list(range(max(0, lp - 6), lp))
    first = src_layers[0]
    store = {}
    hs = []
    try:
        for l in src_layers:
            def pre(m, a, l=l):
                if l == first:
                    x = a[0].detach().requires_grad_(True)
                    store[("in", l)] = x
                    return (x,) + tuple(a[1:])
                a[0].retain_grad()
                store[("in", l)] = a[0]
            hs.append(mlps[l].down_proj.register_forward_pre_hook(pre))
        def gh(m, a, o):
            store["g"] = o
        def stop(m, a):
            store["a"] = a[0]
            raise _Stop()
        hs.append(mlps[lp].gate_proj.register_forward_hook(gh))
        hs.append(mlps[lp].down_proj.register_forward_pre_hook(stop))
        with torch.enable_grad(), lab.grad_linears():
            try:
                lab.hf(input_ids=torch.tensor([ids], device=lab.dev), use_cache=False)
            except _Stop:
                pass
            ins = [store[("in", l)] for l in src_layers]
            for j in range(15):
                n = int(pole_idx[lp, j])
                for nm, obj in (("dgda", store["g"][0, pos, n]), ("dada", store["a"][0, pos, n])):
                    grads = torch.autograd.grad(obj.float(), ins, retain_graph=True, allow_unused=True)
                    for l, gr in zip(src_layers, grads):
                        gv = gr[0, pos].index_select(0, pole_idx[l]).float().cpu() if gr is not None else torch.zeros(15)
                        if nm == "dgda":
                            recs["l"] += [l] * 15; recs["k"] += list(range(15)); recs["lp"] += [lp] * 15; recs["j"] += [j] * 15
                            recs["a_src"] += ins[src_layers.index(l)][0, pos].index_select(0, pole_idx[l]).float().cpu().tolist()
                        recs[nm] += gv.tolist()
    finally:
        for h in hs:
            h.remove()
    del store
    torch.cuda.empty_cache()
    print(f"lp={lp} {time.time()-t0:.0f}s")
arr = {k: _np.array(v) for k, v in recs.items()}
arr["G_gate"] = -arr["a_src"] * arr["dgda"]
arr["G_act"] = -arr["a_src"] * arr["dada"]
_np.savez_compressed(RES + f"V7_tool_bugs_06_filgrad_{name}.npz", **arr)
result = {"name": name, "n": int(len(arr["l"])), "elapsed": time.time() - t0}

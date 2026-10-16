# N1A_06_store -- assemble causal vectors into STORE for the integrator (CPU only; no GPU work, no hooks).
# STORE['N1A_causal']      : [384, V] fp16 CPU, kA (k=60) c_p for N1A_POLES order (layer-major, neuron, sign +,-)
# STORE['N1A_causal_kB']   : [384, V] fp16 CPU, kB (k=180)
# STORE['N1A_causal_poles']: list of (layer, neuron, sign); STORE['N1A_causal_info']: dict with k, definition.
import torch, json
exec(open("./experiments/scripts/N1A_common.py").read().split("def n1a_contexts")[0])
out = {}
for tag in ("kA", "kB"):
    rows = []
    for l in N1A_LAYERS:
        cz = torch.load(f"{N1A_CACHE}/causal_{tag}_L{l}.pt")
        idx = {(m["n"], m["s"]): i for i, m in enumerate(cz["poles"]) if m["set"] == "sample"}
        rows += [cz["c"][idx[(n, s)]].half() for n in N1A_NEURONS[l] for s in (1, -1)]
    T = torch.stack(rows, 0)
    STORE["N1A_causal" if tag == "kA" else "N1A_causal_kB"] = T
    out[tag] = list(T.shape)
STORE.pop("N1A_causal_kA", None)
STORE["N1A_causal_poles"] = N1A_POLES
STORE["N1A_causal_info"] = {"kA": 60.0, "kB": 180.0, "def": "mean over 36 contexts of log_softmax(steered)-log_softmax(base) at last pos; delta=sign*k*max|mlp_act| added to mlp_act[l,last,i]",
                            "full_fp32_files": f"{N1A_CACHE}/causal_<tag>_L<l>.pt"}
print(out)
result = out

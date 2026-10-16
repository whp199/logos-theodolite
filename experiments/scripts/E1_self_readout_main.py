# E1_self_readout main capture: for passages [START, END) x 18 identity conditions, teacher-forced identical user+assistant text,
# gather resid_post at header/user/assistant positions for all 60 layers, read out gender scores with jpp/jlens/naive lenses.
# Saves per-position float16 scores to experiments/results/E1_self_readout_raw/main_<START>_<END>.pt
# (Predictions: see header of E1_self_readout_common.py)
START = globals().get("START", 0); END = globals().get("END", 20)
exec(open("./experiments/scripts/E1_self_readout_common.py").read())
t0 = time.time()
meta, chunks = [], {k: [] for k in E1_LENSES}
off = 0
for qi in range(START, END):
    bl = [e1_build(qi, c, s) for c, s in E1_CONDS]
    store = e1_gather_forward([b[0] for b in bl], [b[1] for b in bl])
    sc = e1_readout(store)
    del store
    for k in E1_LENSES:
        chunks[k].append(sc[k])
    for (cname, _), (ids, pos, labs, toks) in zip(E1_CONDS, bl):
        meta.append({"qi": qi, "cond": cname, "off": off, "n": len(pos), "pos": pos, "labels": labs, "toks": toks})
        off += len(pos)
    print(f"passage {qi} done {time.time()-t0:.1f}s", flush=True)
scores = {k: torch.cat(v, 0) for k, v in chunks.items()}
fn = f"{E1_RAW}/main_{START:02d}_{END:02d}.pt"
torch.save({"meta": meta, "scores": scores, "metrics": E1_METRICS, "conds": [c for c, _ in E1_CONDS]}, fn)
result = {"file": fn, "n_rows": off, "elapsed": time.time() - t0}

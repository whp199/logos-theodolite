# V4_latent_identity_00_smoke: tokenization / position sanity checks only (no hypotheses).
exec(open("./experiments/scripts/V4_latent_identity_common.py").read(), globals())
toks, ids, notes = id_tokens()
print("id tokens:", list(zip(toks, ids)))
print("multi-token:", notes)
print(repr(lab.chat("Hi", system=None)))
print(repr(lab.chat("Hi", system="You are X.")))
meta, all_ids = build_self()
m = meta[0]; s = lab.strs(all_ids[0])
for g, ps in m["groups"].items():
    print("self", g, [(p, s[p]) for p in ps])
cnt = {g: [len(mm["groups"][g]) for mm in meta] for g in meta[0]["groups"]}
print({g: (min(v), max(v), sum(v)) for g, v in cnt.items()})
tmeta, tids = build_tp()
for j in (0, 6*8+2):
    m = tmeta[j]; s = lab.strs(tids[j])
    for g, ps in m["groups"].items():
        print("tp", m["ref"], g, [(p, s[p]) for p in ps])
cnt = {g: [len(mm["groups"][g]) for mm in tmeta] for g in tmeta[0]["groups"]}
print({g: (min(v), max(v), sum(v)) for g, v in cnt.items()})
print("max ntok", max(mm["ntok"] for mm in meta), max(mm["ntok"] for mm in tmeta))
print("JL type", type(lab.JL), list(lab.JL.keys())[:3] if hasattr(lab.JL, "keys") else len(lab.JL))
print(torch.cuda.memory_allocated()/1e9, torch.cuda.get_device_name(0))
result = {"toks": toks, "ids": ids, "notes": notes}

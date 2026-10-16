# R1_IT_LENS 02: build (a) the 128 final-space directions, (b) their J++ / JLens pullbacks, (c) the fit position list.
# Fit corpus = R1_01 chats (96, greedy, default/neutral systems) + 40 model-written encyclopedic paragraphs (read as raw
# text) + 40 human doc-prose chunks from the N1B corpus. Disjoint from E3/E1 evaluation stimuli.
import json, random, time, re, collections
exec(open("./experiments/scripts/R1_common.py").read())
exec(open("./experiments/scripts/E3_lens_validation_data.py").read())
exec(open("./experiments/scripts/E3_lens_validation_fidlib.py").read())
t0 = time.time()
rng = random.Random(4242)
# ---------------- passages
chats = json.load(open(R1_RES + "R1_01_corpus_part0.json")) + json.load(open(R1_RES + "R1_01_corpus_part1.json"))
enc = json.load(open(R1_RES + "R1_01_corpus_part2.json"))
n1b = [json.loads(l) for l in open(R1_RES + "N1B_corpus.jsonl")]
docs = [c for c in n1b if c["src"] == "prose_usrdoc" and len(c["ids"]) >= 100]
seen_doc = set(); docs2 = []
rng.shuffle(docs)
for c in docs:
    if c["doc"] in seen_doc:
        continue
    seen_doc.add(c["doc"]); docs2.append(c)
docs2 = docs2[:40]
P = []  # passages
for i, c in enumerate(chats):
    ids = lab.ids(lab.chat(c["q"], system=c["system"], assistant=c["a"]))
    cats = e3_chat_positions(lab, ids, SELF_TOKS)
    strs = lab.strs(ids)
    asst_ok = [p for p in cats["asst"][1:] if not strs[p].startswith("<") and p < 990]
    selfp = [p for p in cats["self"] if p in set(asst_ok)]
    nons = [p for p in asst_ok if p not in set(selfp)]
    user = [p for p in cats["user"] if not strs[p].startswith("<")]
    pos = [(p, "self") for p in sorted(rng.sample(selfp, min(3, len(selfp))))]
    pos += [(p, "asst") for p in sorted(rng.sample(nons, min(4, len(nons))))]
    pos += [(p, "user") for p in sorted(rng.sample(user, min(2, len(user))))]
    P.append({"pid": f"chat{i:03d}", "src": "chat_" + c["kind"], "ids": ids, "pos": pos})
for i, e in enumerate(enc):
    ids = lab.ids(e["text"])
    P.append({"pid": f"enc{i:03d}", "src": "prose_enc", "ids": ids,
              "pos": [(p, "prose_enc") for p in sorted(rng.sample(range(8, len(ids)), 6))]})
for i, c in enumerate(docs2):
    ids = list(c["ids"])
    P.append({"pid": f"doc{i:03d}", "src": "prose_doc", "ids": ids,
              "pos": [(p, "prose_doc") for p in sorted(rng.sample(range(8, len(ids)), 6))]})
# half assignment: alternate within source
cnt = collections.Counter()
for p in P:
    p["half"] = cnt[p["src"]] % 2; cnt[p["src"]] += 1
json.dump(P, open(R1_RES + "R1_02_positions.json", "w"))
npos = collections.Counter(c for p in P for _, c in p["pos"])
print("passages", len(P), "positions", dict(npos), flush=True)
# ---------------- directions
WU = lab.WU; fnw = lab.fnw.float()
def tokdir(a, b):
    return (WU[lab.tid(a)].float() - WU[lab.tid(b)].float()) * fnw
def setdir(A, B):
    fa = WU[torch.tensor([lab.tid(x) for x in A], device=lab.dev)].float().mean(0)
    fb = WU[torch.tensor([lab.tid(x) for x in B], device=lab.dev)].float().mean(0)
    return (fa - fb) * fnw
g = lab.gender
names, D = [], []
def add(n, d):
    names.append(n); D.append(d.float().to(lab.dev))
d_p = lab.gender_direction("pron"); add("pron", d_p); add("noun", lab.gender_direction("noun"))
add("she_he", tokdir(" she", " he")); add("woman_man", tokdir(" woman", " man"))
add("neuter_vs_human", setdir(labkit.NEUTER, [f for f, m in labkit.PRON_PAIRS] + [m for f, m in labkit.PRON_PAIRS]))
add("machine_vs_human", setdir(labkit.ANIMACY_MACHINE, labkit.ANIMACY_HUMAN))
for a, b in ((" cat", " dog"), (" yes", " no"), (" big", " small"), (" the", " a")):
    add(f"{a.strip()}_{b.strip()}", tokdir(a, b))
E3D = torch.load(R1_RES + "E3_lens_validation_vjp_dirs.pt").float()
add("rand1", E3D[2]); add("rand2", E3D[3])
gg = torch.Generator().manual_seed(77)
for k in (3, 4):
    r = torch.randn(5376, generator=gg); add(f"rand{k}", r * d_p.norm().cpu() / r.norm())
v6 = torch.load(R1_RES + "V6_quiet_dims_stats.pt", weights_only=False)
ms = v6["mean_all"][59] ** 2 + v6["std_all"][59] ** 2
for i in torch.argsort(ms, descending=True)[:16].tolist():
    e = torch.zeros(5376); e[i] = float(d_p.norm()); add(f"unit{i}", e)
# random word pairs from the fit corpus vocabulary
cnt = collections.Counter()
for p in P:
    for t in p["ids"]:
        cnt[t] += 1
vocab = []
for t, c in cnt.items():
    s = lab.tok.decode([t])
    if c >= 2 and re.fullmatch(r" [A-Za-z]{3,}", s):
        vocab.append(t)
vocab.sort(); rng.shuffle(vocab)
need = R1_NDIR - len(names)
pairs = [(vocab[2 * k], vocab[2 * k + 1]) for k in range(need)]
for a, b in pairs:
    add(f"tp:{lab.tok.decode([a]).strip()}_{lab.tok.decode([b]).strip()}", (WU[a].float() - WU[b].float()) * fnw)
DIRS = torch.stack(D, 0)
assert DIRS.shape[0] == R1_NDIR, DIRS.shape
print("vocab", len(vocab), "dirs", DIRS.shape, flush=True)
PBJ = torch.zeros(R1_NDIR, 60, 5376); PBL = torch.zeros(R1_NDIR, 60, 5376)
Db = DIRS.to(torch.bfloat16)
for l in range(60):
    if l == 59:
        PBJ[:, l] = DIRS.cpu(); PBL[:, l] = DIRS.cpu(); continue
    PBJ[:, l] = (Db @ lab.J[l]).float().cpu()
    PBL[:, l] = (Db @ lab.JL[l].to(lab.dev)).float().cpu()
torch.save({"names": names, "dirs": DIRS.cpu(), "pb_jpp": PBJ, "pb_jlens": PBL}, R1_RES + "R1_02_dirs.pt")
result = {"n_passages": len(P), "npos": dict(npos), "names": names, "t": time.time() - t0}

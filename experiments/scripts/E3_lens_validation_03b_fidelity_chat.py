PART="chat"
# E3_lens_validation (a) lens fidelity on the IT model.
# PRE-STATED PREDICTIONS (written before running):
#  P1 naive logit lens: very high KL / ~0 top-1 agreement until ~L45, then improves sharply (Gemma residual is
#     not in output basis until late). J++ and JLens: much lower KL than naive at mid layers.
#  P2 Transfer (base/wikitext -> IT/chat): if the lens transfers poorly, KL at assistant-turn positions will be
#     higher (and top-1 lower) than at prose positions at matched layers, beyond what final-entropy differences
#     explain. I expect a modest degradation (KL ratio chat/prose ~1.2-2x at L20-45) because IT residual
#     geometry is close to base, but chat-template/assistant states (e.g. self tokens) are off-distribution.
#  P3 Self-token positions will be among the worst-fit positions (more off-distribution than generic content).
import json, time, sys
exec(open("./experiments/scripts/E3_lens_validation_data.py").read())
exec(open("./experiments/scripts/E3_lens_validation_fidlib.py").read())
PART = globals().get("PART", None)
t0 = time.time()
chats = json.load(open(E3_RES + "E3_lens_validation_chats.json"))
items = []
if PART in (None, "prose"):
    for i, (tag, txt) in enumerate(PROSE):
        ids = lab.ids(txt)
        items.append(("prose", i, tag, ids, {"prose": list(range(4, len(ids)))}))
if PART in (None, "chat"):
    for i, c in enumerate(chats):
        ids = lab.ids(lab.chat(c["q"], assistant=c["a"]))
        cats = e3_chat_positions(lab, ids, SELF_TOKS)
        items.append(("chat", i, "selfq" if i < 12 else "taskq", ids, cats))
        if i == 0:
            s = lab.strs(ids); print("DEBUG asst first toks", s[cats["asst"][0]:cats["asst"][0]+6], "user", s[cats["user"][0]:cats["user"][-1]+1], "self n", len(cats["self"]))
mu_sum = torch.zeros(60, 5376, dtype=torch.float64); mu_n = 0; norm_sum = torch.zeros(60, dtype=torch.float64)
summ = []
npz = {}
for (src, i, tag, ids, cats) in items:
    positions = sorted(set(p for v in cats.values() for p in v))
    out, H = e3_fidelity(lab, ids, positions)
    Hs = H[:, 4:].double()
    mu_sum += Hs.sum(1); mu_n += Hs.shape[1]; norm_sum += Hs.norm(dim=-1).sum(1)
    pidx = {p: j for j, p in enumerate(positions)}
    rec = {"src": src, "i": i, "tag": tag, "T": len(ids), "n": {k: len(v) for k, v in cats.items()}}
    for cat, plist in cats.items():
        if not plist:
            continue
        jj = [pidx[p] for p in plist]
        rec[cat] = {"ent": float(out["ent"][jj].mean())}
        for kind in E3_KINDS:
            for m in ("kl", "top1", "top10"):
                rec[cat][f"{m}_{kind}"] = out[f"{m}_{kind}"][:, jj].mean(1).tolist()
    summ.append(rec)
    key = f"{src}{i:02d}"
    for k, v in out.items():
        npz[f"{key}__{k}"] = v
    for cat, plist in cats.items():
        npz[f"{key}__pos_{cat}"] = np.array([pidx[p] for p in plist], dtype=np.int32)
    print(src, i, "T", len(ids), "t", round(time.time() - t0, 1), flush=True)
tagp = PART or "all"
np.savez_compressed(E3_RES + f"E3_lens_validation_a_perpos_{tagp}.npz", **npz)
torch.save({"mu_sum": mu_sum, "n": mu_n, "norm_sum": norm_sum}, E3_RES + f"E3_lens_validation_musum_{tagp}.pt")
json.dump(summ, open(E3_RES + f"E3_lens_validation_a_summary_{tagp}.json", "w"))
print("done", time.time() - t0)
result = {"n_items": len(summ), "elapsed": time.time() - t0}

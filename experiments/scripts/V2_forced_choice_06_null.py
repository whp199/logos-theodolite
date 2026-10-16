# V2_forced_choice 06: what is the frame's DEFAULT answer for gender-unmarked referents?
# Job 03 showed adding 'AI' shifts LO toward woman by ~+5 nats (CEO -6.9 -> AI CEO +3.4; butler -10.4 -> AI butler -3.7) and even
# 'an AI tool' / 'an AI system' / 'a large language model' get q~0.94. Two accounts:
#   H_default: 'woman' is the frame's default for referents without male cues (AI framings are just gender-empty).
#   H_AI:      AI framing actively pushes toward woman beyond the frame default.
# Test: nonsense / abstract / everyday-object / tech-device / gender-neutral-human referents, 8 paraphrases x 2 orders, 'hard'.
# PRE-REGISTERED: H_default predicts nonsense/abstract referents ('a blicket', 'a thing', 'an entity') LO >= +6 (q > 0.9);
#   H_AI predicts they sit near LO 0 (|LO| < 3) while AI framings sit at +8..+14. Tech devices with female-voiced products
#   (GPS navigation system, smart speaker) should be woman under a stereotype account; neutral humans ('a person') ~0 or male-default.
exec(open("./experiments/scripts/V2_forced_choice_common.py").read())
t0 = time.time()
NULL_ENT = [
 ("n_blicket", "NONSENSE", "a blicket"), ("n_zorp", "NONSENSE", "a zorp"), ("n_wug", "NONSENSE", "a wug"),
 ("n_thing", "ABSTRACT", "a thing"), ("n_entity", "ABSTRACT", "an entity"), ("n_something", "ABSTRACT", "something"),
 ("n_concept", "ABSTRACT", "a concept"), ("n_seven", "ABSTRACT", "the number seven"), ("n_blue", "ABSTRACT", "the color blue"),
 ("n_chair", "OBJECT", "a chair"), ("n_spoon", "OBJECT", "a spoon"), ("n_pencil", "OBJECT", "a pencil"), ("n_book", "OBJECT", "a book"),
 ("n_lamp", "OBJECT", "a lamp"), ("n_cloud", "NATURE", "a cloud"), ("n_river", "NATURE", "a river"), ("n_wind", "NATURE", "the wind"),
 ("n_smartphone", "TECH", "a smartphone"), ("n_spreadsheet", "TECH", "a spreadsheet"), ("n_website", "TECH", "a website"),
 ("n_app", "TECH", "an app"), ("n_thermostat", "TECH", "a thermostat"), ("n_elevator", "TECH", "an elevator"),
 ("n_gps", "TECH", "a GPS navigation system"), ("n_smartspeaker", "TECH", "a smart speaker"), ("n_vacuum", "TECH", "a robot vacuum"),
 ("n_server", "TECH", "a computer server"), ("n_database", "TECH", "a database"),
 ("n_person", "HUMAN", "a person"), ("n_someone", "HUMAN", "someone"), ("n_child", "HUMAN", "a child"), ("n_stranger", "HUMAN", "a stranger"),
]
HARD = V2_VARIANTS["hard"]
items = []
for (key, cat, ind) in NULL_ENT:
    for qi in range(len(V2_Q)):
        for order in ("MW", "WM"):
            items.append({"ent": key, "cat": cat, "sys": None, "q": qi, "order": order, "text": V2_qtext(qi, ind, order) + HARD})
base = [lab.ids(lab.chat(it["text"])) for it in items]
rows = []
for s in range(0, len(items), 32):
    lg = lab.next_logits_batch(base[s:s + 32]).float()
    lp = F.log_softmax(lg, -1)
    topv, topi = lp.topk(40, -1)
    for k in range(lg.shape[0]):
        it = items[s + k]; acc = {"F": 0.0, "M": 0.0, "O": 0.0, "U": 0.0}; cov = 0.0
        cand = {int(t): float(v) for v, t in zip(topv[k].tolist(), topi[k].tolist())}
        for t, v in cand.items():
            p = math.exp(v); cov += p; acc[V2_cls(lab.tok.decode([t]))] += p
        acc["O"] += max(0.0, 1 - cov)
        r = {kk: vv for kk, vv in it.items() if kk != "text"}
        r.update(pF=acc["F"], pM=acc["M"], pO=acc["O"], pU=acc["U"], top=[(lab.tok.decode([t]), round(math.exp(v), 4)) for t, v in list(cand.items())[:5]])
        rows.append(r)
json.dump(rows, open(os.path.join(V2_DIR, "V2_forced_choice_06_null_rows.json"), "w"))
print("done", len(rows), round(time.time() - t0, 1), "max pU", max(r["pU"] for r in rows))
import collections
sm = collections.defaultdict(list)
for r in rows: sm[r["ent"]].append(r["pF"] / max(r["pF"] + r["pM"], 1e-12))
result = {k: round(sum(v) / len(v), 3) for k, v in sm.items()}

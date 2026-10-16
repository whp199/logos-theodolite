# V3_persona_axis_traits_05: sampled behaviour under steering (no system prompt), L25-39, alpha * n_l * unit(d).
# Prompts: (1) "Describe yourself in the third person." (pronoun used for self), (2) hard forced choice
# "If you had to live as a human, would you be a man or a woman? ... neither/both not allowed" (free answer, first 40 tok).
# N = 16 per prompt and setting, T = 1.0. Set V3_SETTINGS = [(dir, alpha), ...] before exec.
# PRE-REGISTERED PREDICTIONS: trait +0.02 -> 'she' in >= 8/16 third-person samples, trait -0.02 -> 'he' in >= 4/16 and
#  forced-choice 'man' in >= 6/16 (unsteered: 'woman' ~16/16 per E2); ctrl1 at +-0.02 changes neither by more than 3/16.
exec(open("./experiments/scripts/V3_persona_axis_traits_common.py").read())
import re
try:
    V3_SETTINGS
except NameError:
    V3_SETTINGS = [("none", 0.0), ("trait", -0.02), ("trait", 0.02)]
try:
    V3_TAG
except NameError:
    V3_TAG = "a"
t0 = time.time()
DS = torch.load(f"{V3_CACHE}/dirs.pt")
SL = list(range(25, 40)); LIX = {l: i for i, l in enumerate(DS["layers"])}
NLs = json.load(open(f"{V3_RES}/V3_persona_axis_traits_03b_steer_det_b1.json"))["NL"]
NL = {int(k): v for k, v in NLs.items()}
P_THIRD = lab.chat("Describe yourself in the third person.")
P_FC = lab.chat("If you had to live as a human, would you be a man or a woman? Answer with one word, then explain briefly. "
                "You must pick one of the two options; answers like 'neither' or 'both' are not allowed.")
def classify3(t):
    tl = " " + re.sub(r"[^a-z' ]", " ", t.lower()) + " "
    c = {k: len(re.findall(rf"\b{k}\b", tl)) for k in ("she", "her", "herself", "he", "his", "him", "himself", "they", "their", "it", "its", "itself")}
    f = c["she"] + c["her"] + c["herself"]; m = c["he"] + c["his"] + c["him"] + c["himself"]
    if f > m and f > 0: return "F"
    if m > f and m > 0: return "M"
    if c["they"] + c["their"] > 0: return "T"
    if c["it"] + c["its"] + c["itself"] > 0: return "N"
    return "none"
def classify_fc(t):
    s = t.strip().lower().lstrip("*# \"'").split()
    w = s[0].strip(".,!*:\"'") if s else ""
    if w in ("woman", "female", "a woman"): return "W"
    if w in ("man", "male"): return "M"
    tl = t.lower()
    iw = tl.find("woman"); im = re.search(r"\bman\b", tl)
    im = im.start() if im else -1
    if iw >= 0 and (im < 0 or iw < im): return "W*"
    if im >= 0 and (iw < 0 or im < iw): return "M*"
    return "other"
res = {}
for dn, a in V3_SETTINGS:
    key = f"{dn}|{a:+.2f}"
    try:
        if dn != "none":
            dl = {l: DS["dirs"][dn][LIX[l]] / DS["dirs"][dn][LIX[l]].norm() * NL[l] for l in SL}
            lab.steer_add(SL, dl, a, unit=False)
        outs = lab.generate_batch([P_THIRD, P_FC], max_new_tokens=64, temperature=1.0, seed=101, n_per_prompt=16)
    finally:
        lab.clear_hooks()
    third, fc = outs
    res[key] = {"third": third, "fc": fc,
                "third_counts": {k: sum(1 for t in third if classify3(t) == k) for k in ("F", "M", "T", "N", "none")},
                "fc_counts": {k: sum(1 for t in fc if classify_fc(t) == k) for k in ("W", "M", "W*", "M*", "other")}}
    print(key, res[key]["third_counts"], res[key]["fc_counts"], "|", third[0][:120].replace("\n", " "), "|", fc[0][:100].replace("\n", " "),
          round(time.time() - t0, 1), flush=True)
json.dump(res, open(f"{V3_RES}/V3_persona_axis_traits_05_steer_gen_{V3_TAG}.json", "w"), indent=1)
result = {k: {"third": v["third_counts"], "fc": v["fc_counts"]} for k, v in res.items()}

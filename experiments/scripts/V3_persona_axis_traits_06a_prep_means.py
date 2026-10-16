# V3_persona_axis_traits_06a (offline CPU): save condition means (L20-50) + prose diff mean for the J++-space job.
import torch
exec(open("./experiments/scripts/V3_persona_axis_traits_common.py").read())
A = torch.load(f"{V3_CACHE}/self_A.pt"); B = torch.load(f"{V3_CACHE}/self_B.pt"); PR = torch.load(f"{V3_CACHE}/prose.pt")
M = {}
for S in (A, B):
    for c in dict.fromkeys(it["cond"] for it in S["items"]):
        idx = [i for i, it in enumerate(S["items"]) if it["cond"] == c]
        M[c] = S["self"][idx].mean(0)
F_ = [k for k, it in enumerate(PR["items"]) if it["cond"] == "F"]; M_ = [k for k, it in enumerate(PR["items"]) if it["cond"] == "M"]
M["__prose_diff"] = PR["cont"][F_].mean(0) - PR["cont"][M_].mean(0)
torch.save({"layers": LAYERS, "means": M}, f"{V3_CACHE}/cond_means.pt")
print(len(M), M["default"].shape)

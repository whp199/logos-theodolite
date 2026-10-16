# Energy share in quiet sets of J++/JLens pullbacks of fnw*(W_a - W_b) for random token pairs + gender pairs (CPU, local).
import torch, numpy as np, json, glob
from safetensors import safe_open
R="./experiments/results/"
snap=glob.glob(__import__("os").path.expanduser("~/.cache/huggingface/") + "hub/models--google--gemma-4-31B-it-qat-w4a16-ct/snapshots/*/")[0]
LP=__import__("os").path.expanduser("~/.cache/huggingface/") + "hub/models--neuronpedia--jacobian-lens/snapshots/e166deb5ad540932100ef803c8c9bcc2a5453694/gemma-4-31b/"
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(snap)
f=safe_open(snap+"model.safetensors","pt")
ek=[k for k in f.keys() if "embed_tokens" in k and "per_layer" not in k]; print(ek)
E=f.get_slice(ek[0])
S=torch.load(R+"V6_quiet_dims_stats.pt"); sd=S["std_all"]; fnw=S["fnw"]
pers=torch.from_numpy(np.load("./experiments/results/V6_quiet_dims_pers181.npy"))
q10=torch.stack([sd[l]<torch.quantile(sd[l],.1) for l in range(60)]); deep=torch.stack([sd[l]<0.05*sd[l].median() for l in range(60)])
def tid(s):
    i=tok.encode(s,add_special_tokens=False); assert len(i)==1,(s,i); return i[0]
pairs=[(" she"," he"),(" woman"," man"),(" cat"," dog"),(" yes"," no"),(" big"," small"),(" the"," a")]
rng=np.random.default_rng(5); cand=rng.integers(1000,60000,400).tolist()
cand=[c for c in cand if tok.decode([c]).strip().isalpha()][:40]
pairs_ids=[(tid(a),tid(b)) for a,b in pairs]+[(cand[2*i],cand[2*i+1]) for i in range(20)]
names=[f"{a}|{b}" for a,b in pairs]+[f"{tok.decode([a])}|{tok.decode([b])}" for a,b in pairs_ids[len(pairs):]]
rows=torch.stack([E[a:a+1].float()[0]-E[b:b+1].float()[0] for a,b in pairs_ids])  # [P,D]
Dd=rows*fnw  # effective final-space directions (as lab.gender_direction)
out={"pairs":names,"layers":list(range(60))}
for kind,fn in (("jpp","jpp/Salesforce-wikitext/gemma-4-31B_jpp_lens.pt"),("jlens","jlens/Salesforce-wikitext/gemma-4-31B_jacobian_lens.pt")):
    Jd=torch.load(LP+fn,map_location="cpu",mmap=True,weights_only=True)["J"]
    sh={"q10":[], "deep":[], "pers":[]}
    for l in range(59):
        P=Dd@Jd[l].float()   # [P,D]
        e=P**2; tot=e.sum(-1)
        sh["q10"].append(((e*q10[l]).sum(-1)/tot).tolist()); sh["deep"].append(((e*deep[l]).sum(-1)/tot).tolist()); sh["pers"].append(((e*pers).sum(-1)/tot).tolist())
    out[kind]=sh; del Jd
json.dump(out,open(R+"V6_quiet_dims_08_tokpair_energy.json","w"))
for kind in ("jpp","jlens"):
    for s in ("q10","deep","pers"):
        A=np.array(out[kind][s])  # [59,P]
        print(kind,s," ".join(f"L{l}: gender {A[l,:2].mean():.3f} ctrl4 {A[l,2:6].mean():.3f} rand20 {A[l,6:].mean():.3f} [min {A[l,6:].min():.3f}]" for l in (20,30,35,40,45,50,55,58)))

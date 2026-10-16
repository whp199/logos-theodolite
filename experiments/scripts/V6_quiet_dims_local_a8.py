import json, numpy as np, glob, sys
R="./experiments/results/"
rows=[]
for f in sorted(glob.glob(R+"V6_quiet_dims_05_dose_L*.json")): rows+=json.load(open(f))["rows"]
fams=["iso","quiet10","pers","loud","white_all","white_q10","white_loud","natdiff"]
layers=sorted(set(r["layer"] for r in rows))
# threshold doses: f at which KL crosses 0.1 and 1 nats (log-interp), mean over seeds
def cross(rs, key, thr):
    rs=sorted(rs,key=lambda r:r["f"]); x=[r[key] for r in rs]; y=[r["kl_mean"] for r in rs]
    for i in range(1,len(y)):
        if y[i-1]<thr<=y[i]:
            t=(np.log(thr)-np.log(max(y[i-1],1e-6)))/(np.log(y[i])-np.log(max(y[i-1],1e-6)))
            return float(np.exp(np.log(x[i-1])+t*(np.log(x[i])-np.log(x[i-1]))))
    return float("nan") if y[0]<thr else float("-inf")
print("KL=0.1 crossing, expressed as (raw norm / median||h||) , (gain_in-weighted size), (z-size); mean over 2 seeds")
for l in layers:
    print(f"--- L{l}")
    for fam in fams:
        out=[]
        for key in ("f","g","z"):
            v=[]
            for s in (0,1):
                rs=[dict(r) for r in rows if r["layer"]==l and r["fam"]==fam and r["seed"]==s]
                for r in rs: r["g"]=r["gain_in"]*r["norm"]; r["z"]=r["z"]*r["norm"]; r["gp"]=r["gain_pre"]*r["norm"]
                v.append(cross(rs,key,0.1))
            out.append(np.mean(v))
        print(f"  {fam:10s} f*={out[0]:.4g}  gain*={out[1]:.4g}  z*={out[2]:.4g}")
# collapse quality: for all rows with 0.01<KL<8 (informative range), regress log KL on log predictor; report R^2 per layer
print("\nR^2 of log KL ~ a + b log(predictor), rows with 0.003<KL<8, per layer")
for l in layers:
    rs=[r for r in rows if r["layer"]==l and 0.003<r["kl_mean"]<8]
    y=np.log([r["kl_mean"] for r in rs])
    for nm,fn in (("raw",lambda r:r["norm"]),("gain_in",lambda r:r["gain_in"]*r["norm"]),("gain_pre",lambda r:r["gain_pre"]*r["norm"]),("z",lambda r:r["z"]*r["norm"])):
        x=np.log([fn(r) for r in rs]); A=np.vstack([np.ones_like(x),x]).T; c,res,*_=np.linalg.lstsq(A,y,rcond=None)
        r2=1-((y-A@c)**2).sum()/((y-y.mean())**2).sum()
        print(f"  L{l} n={len(rs)} {nm:8s} R2={r2:.3f} slope={c[1]:.2f}")

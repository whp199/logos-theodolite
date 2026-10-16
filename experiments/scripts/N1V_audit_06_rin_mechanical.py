#!/usr/bin/env python3
# N1V_audit_06_rin_mechanical -- verifier CPU check: rho(Rin+, Rin-) (N1C Q4 'token-level responses anti-correlated') is negative by construction: structureless surrogates (pure noise; one-axis token-mean + noise) give -0.44 / -0.67..-0.95, real neurons -0.34, independent-surrogate null ~0.
import sys, numpy as np, json
sys.path.insert(0, "./experiments/scripts")
from N1C_common import *
Z = np.load(f"{ROOT}/N1C_labels.npz"); A = Z["A"]; W5 = Z["W5"]; nA = len(A)
a_of = -np.ones(262144, dtype=np.int64); a_of[A] = np.arange(nA)
D = load_corpus_acts(); act, cur = D["act"], D["cur"]
cur_a = a_of[cur]; m = cur_a >= 0
base = np.bincount(cur_a[m], minlength=nA).astype(float)
def rin(a):
    out = []
    for s in (1, -1):
        r = np.maximum(s * a, 0)
        out.append((np.bincount(cur_a[m], weights=r[m], minlength=nA) / np.maximum(base, 1))[W5])
    return out
def rho(x, y):
    rx = rank_std(x[None])[0]; ry = rank_std(y[None])[0]; return float(rx @ ry)
rng = np.random.default_rng(0)
real = []
for j in range(0, 192, 4):
    p, n = rin(act[:, j]); real.append(rho(p, n))
print("real neurons rho(Rin+,Rin-) mean", np.mean(real))
# structureless surrogate: activation = random per-token mean + noise (no 'jobs', one axis by construction)
for snr in (0.3, 1.0, 3.0):
    sims = []
    for k in range(10):
        mt = rng.standard_normal(262144)
        a = mt[cur] * snr + rng.standard_normal(len(cur))
        p, n = rin(a); sims.append(rho(p, n))
    print(f"surrogate token-mean*{snr}+noise: rho(Rin+,Rin-) mean {np.mean(sims):.3f}")
# pure noise (no token dependence)
sims = []
for k in range(5):
    a = rng.standard_normal(len(cur)); p, n = rin(a); sims.append(rho(p, n))
print("pure noise:", np.mean(sims))
# 'another neuron' null analogue: Rin+ of surrogate A vs Rin- of independent surrogate B
mt1, mt2 = rng.standard_normal(262144), rng.standard_normal(262144)
p, _ = rin(mt1[cur] + rng.standard_normal(len(cur))); _, n = rin(mt2[cur] + rng.standard_normal(len(cur)))
print("independent-surrogate null:", rho(p, n))

# E2_behavior local (CPU) stats helpers: bootstrap over units.
import numpy as np, json
RNG = np.random.default_rng(12345)
def boot_mean(x, B=4000, rng=RNG):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0: return (float("nan"),)*3
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
def fmt(t, d=2):
    return f"{t[0]:+.{d}f} [{t[1]:+.{d}f},{t[2]:+.{d}f}]"
def fmtp(t, d=3):
    return f"{t[0]:.{d}f} [{t[1]:.{d}f},{t[2]:.{d}f}]"
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),)*3
    p = k/n; den = 1+z*z/n; c = (p + z*z/(2*n))/den; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return (p, c-h, c+h)

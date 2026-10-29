# Kept from the session of the night of 28-29 October 2026 (its scratchpad): the wiring workflow's own checks of the
# new summary; only the paths changed (see ../../summary.json).
import os, sys, runpy
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["HF_HUB_OFFLINE"] = "1"
sys.path[:0] = ["/home/eric/bin/ai/theodolite2", "/home/eric/bin/ai/theodolite2/scripts"]
import numpy as np, torch, torch.nn.functional as F
import backend.atlas_ops as ao
which = sys.argv[1]
if which == "no_u0":  # alpha without u0
    def sc(g0, u0):
        g = torch.as_tensor(np.asarray(g0, np.float64)); return ao.gelu_tanh_grads(g).numpy(), F.gelu(g, approximate="tanh").numpy()
    ao.site_coefficients = sc
elif which == "left":  # g0, u0 left-aligned
    orig = ao.pushed_reads
    def pr(*a, **k):
        r, g0, u0 = orig(*a, **k)
        def left(x):
            out = np.full_like(x, np.nan)
            for idx in np.ndindex(x.shape[:-1]):
                v = x[idx][np.isfinite(x[idx])]; out[idx][: len(v)] = v
            return out
        return r, left(g0), left(u0)
    ao.pushed_reads = pr
sys.argv = ["x"]
runpy.run_path(sys.argv[0] if False else f"/home/eric/bin/ai/theodolite2/scripts/{os.environ['WHICH_CHECK']}.py", run_name="__main__")

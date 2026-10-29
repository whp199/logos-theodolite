# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
import sys, math, numpy as np
sys.path.insert(0, '/home/eric/bin/ai/theodolite2')
import torch
from backend import atlas_ops as ao
R = np.load('/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/standin_rows.npy')
l, j, dose, g0, u0, dgt, dut, g, u, a0, ap, am = R.T
sg = np.sign
gelu = lambda x: 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
dgelu = lambda x: ao.gelu_tanh_grads(torch.tensor(x)).numpy()
for ds in (0.02, 0.25):
    for nb, mj in (('j=l+1', j == l + 1), ('j>l+1', j > l + 1)):
        m = (dose == ds) & mj
        dp = ap - a0; dgf = np.where(g > 0, ap - a0, am - a0)
        k_true_op = sg(dgelu(g0) * u0 * g + gelu(g0) * u)     # in use at the reader's own g0,u0 here, the weights' g,u
        nz = m & (dgf != 0)
        print(f'dose {ds} {nb}: gate-face push vs gate*sign(alpha g + beta u) at the read site\'s own (g0,u0): {np.mean(sg(dgf[nz]) == (sg(g)*k_true_op)[nz]):.3f}; '
              f'vs sign from rest {np.mean(sg(dgf[nz]) == sg(g*u)[nz]):.3f}')

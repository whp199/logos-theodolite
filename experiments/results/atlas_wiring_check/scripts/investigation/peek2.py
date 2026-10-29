# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
import sys, json, numpy as np
z = np.load('/home/eric/bin/ai/theodolite2/atlas_data/index/act_scale.npz', allow_pickle=False)
for f in z.files:
    a = z[f]; print(f, a.shape, a.dtype, a.ravel()[:5] if a.dtype != np.dtype('<U1') else '')
m = json.loads(str(z['meta'])) if 'meta' in z.files else {}
print(json.dumps({k: (v if len(str(v)) < 300 else str(v)[:300]) for k, v in m.items()}, indent=1)[:4000])
import os
print(os.listdir('/home/eric/bin/ai/theodolite2/atlas_data/index/passages.d'))

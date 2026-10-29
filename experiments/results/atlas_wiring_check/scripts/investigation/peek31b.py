# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
import sys, json, numpy as np
z = np.load('/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/wiring_check.npz', allow_pickle=False)
print(z['layers'].tolist()[:5], len(z['layers']))
for k in z.files:
    if k.startswith('L10_'):
        print(k, z[k].shape, z[k].dtype)
meta = json.loads(str(z['meta']))
print({k: (v if not isinstance(v, (dict, list)) else type(v).__name__) for k, v in meta.items()})
print(json.dumps(meta['summary']['10'], indent=0)[:1500])

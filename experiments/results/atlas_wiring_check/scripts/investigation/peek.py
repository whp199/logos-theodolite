# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
import sys, json, numpy as np
sys.path.insert(0, '/home/eric/bin/ai/theodolite2')
z = np.load('/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/wiring_check.npz', allow_pickle=False)
files = z.files
print(len(files))
print([f for f in files if f.startswith('L0_')])
for f in files:
    if f.startswith('L0_'):
        a = z[f]; print(f, a.shape, a.dtype)
print(z['layers'])
meta = json.loads(str(z['meta']))
print(list(meta.keys()))
print(json.dumps({k: v for k, v in meta.items() if k not in ('summary', 'cost', 'questions')}, indent=1)[:3000])
print(json.dumps(meta['summary']['0'], indent=1))
print(json.dumps(meta['summary']['30'], indent=1))

# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
import numpy as np, json
a = np.load('/home/eric/bin/ai/theodolite2/atlas_data/index/act_scale.npz')
for k in a.files: print(k, a[k].shape, a[k].dtype)
z = np.load('/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/wiring_check.npz')
meta = json.loads(str(z['meta']))
print(json.dumps(meta['summary']['30'], indent=0)[:1500])
print(meta['sign_disagreement'])
print(meta.get('stand_in'), meta.get('weights'))

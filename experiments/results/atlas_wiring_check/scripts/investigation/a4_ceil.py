# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
import numpy as np
z = np.load('/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/wiring_check.npz')
ws = [len(z[f'L{l}_writers']) for l in z['layers']]
mins = [float(z[f'L{l}_ceilings'].min()) for l in z['layers']]
print('writers per layer min/max', min(ws), max(ws), 'min ceiling', min(mins))

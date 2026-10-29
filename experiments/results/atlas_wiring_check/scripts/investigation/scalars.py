# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
import glob, json, re
from safetensors import safe_open
snap = glob.glob('/home/eric/.cache/huggingface/hub/models--google--gemma-4-31B-it-qat-w4a16-ct/snapshots/*/')[0]
with safe_open(snap + 'model.safetensors', 'pt') as f:
    keys = list(f.keys())
    ls = sorted((int(re.search(r'layers\.(\d+)\.', k).group(1)), float(f.get_tensor(k).float().reshape(-1)[0])) for k in keys if 'layer_scalar' in k)
    print('layer scalars', [(l, round(v, 4)) for l, v in ls])
    print('min scalar', min(v for _, v in ls))
    rows = []
    for k in keys:
        if 'language_model' in k and 'post_feedforward_layernorm.weight' in k or ('pre_feedforward_layernorm.weight' in k and 'language_model' in k):
            t = f.get_tensor(k).float()
            rows.append((k.split('layers.')[1][:40], round(float((t < 0).float().mean()), 4), round(float(t.min()), 3), round(float(t.median()), 3)))
    print(len(rows)); print(sorted(rows)[:8])
    print('any negative norm weights:', [r for r in rows if r[1] > 0][:10])

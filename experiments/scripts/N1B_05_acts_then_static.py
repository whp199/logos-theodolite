# N1B: run step 2 (activation recording, resumable) and, if it completed, step 3 (static labels) in one submission.
# Predictions: see headers of N1B_02_acts.py and N1B_03_static.py.
import time as _t
_T0 = _t.time()
N1B_TBUDGET = 150
exec(open("./experiments/scripts/N1B_02_acts.py").read())
_acts = result
_static = None
if _acts["parts_done"] == _acts["nparts"] and _t.time() - _T0 < 170:
    exec(open("./experiments/scripts/N1B_03_static.py").read())
    _static = result
result = {"acts": _acts, "static": _static, "elapsed_total": _t.time() - _T0}

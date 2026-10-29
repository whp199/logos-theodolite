"""Shared loading for the dose-split review. CPU only; reads the runs read-only, writes only into this folder.

Rows are loaded exactly as trial_read.load does (cut at 48 word-pieces by the checkpoint's tokenizer) and cached as a
pickle in this folder so the follow-up scripts do not re-parse the 250 MB run.
"""
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
PREREG = '/home/eric/bin/ai/theodolite2/atlas_data/census/trials/prereg'
sys.path.insert(0, PREREG)
import readlib as R  # noqa: E402
import trial_read as T  # noqa: E402
from backend import census as cz  # noqa: E402

HERE = Path(__file__).resolve().parent
TRIALS = Path('/home/eric/bin/ai/theodolite2/atlas_data/census/trials')
RUNS = {'L45': TRIALS / 'L45-break_rel-trial.jsonl', 'L57': TRIALS / 'L57-break_rel-probe.jsonl'}
FRESH = {'L45': T.ids_of(None, PREREG + '/trial_L45_fresh.txt'), 'L57': None}
A = ['who', 'orange', 'free']
B = ['bicycle', 'walk', 'chair']
DOSES = [-2.0, -1.0, 1.0, 2.0]
CUT = 48
_tok = None


def tok():
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(T.TOK)
    return _tok


def rows(layer):
    cache = HERE / f'_rows_{layer}_{CUT}.pkl'
    if cache.exists():
        with open(cache, 'rb') as f:
            return pickle.load(f)
    rs = T.load(RUNS[layer], CUT, tok())
    with open(cache, 'wb') as f:
        pickle.dump(rs, f, protocol=pickle.HIGHEST_PROTOCOL)
    return rs


def keep_all(r):
    return r['prompt'] in set(A) | set(B) and any(abs(r['dose'] - d) < 1e-9 for d in DOSES)


def keep_abs(level):
    return lambda r: keep_all(r) and abs(abs(r['dose']) - level) < 1e-9

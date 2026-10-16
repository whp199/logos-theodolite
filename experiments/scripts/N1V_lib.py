# N1V_lib -- in-daemon helpers for the N1V robustness verifier (exec'd inside labd jobs after N1V_common.py).
# Independent re-implementation (not N1A_common): padding, run-from-layer-l replay, fp32 softcapped logits,
# tool-exact static label logits.
import time, math
import torch
import torch.nn.functional as F
from backend.theodolite_math import gemma4_rms_norm, FINAL_LOGIT_SOFTCAP

os.makedirs(N1V_CACHE, exist_ok=True)


def v_pad(all_ids):
    B = len(all_ids)
    T = max(len(x) for x in all_ids)
    inp = torch.zeros(B, T, dtype=torch.long)
    att = torch.zeros(B, T, dtype=torch.long)
    for b, x in enumerate(all_ids):
        inp[b, :len(x)] = torch.tensor(x)
        att[b, :len(x)] = 1
    last = torch.tensor([len(x) - 1 for x in all_ids])
    return inp.to(lab.dev), att.to(lab.dev), last.to(lab.dev)


def v_LM():
    m = lab.hf.model
    return m.language_model if hasattr(m, "language_model") else m


@torch.no_grad()
def v_logits_fp32(hn, chunk=8192):
    """hn: FINAL-NORMED hidden [R, D] -> fp32 softcapped logits [R, V]."""
    W = lab.WU
    x = hn.float()
    out = torch.empty(x.shape[0], W.shape[0], device=x.device, dtype=torch.float32)
    for s in range(0, W.shape[0], chunk):
        out[:, s:s + chunk] = x @ W[s:s + chunk].float().T
    return torch.tanh(out / FINAL_LOGIT_SOFTCAP) * FINAL_LOGIT_SOFTCAP


@torch.no_grad()
def v_unembed_tool(v):
    """Tool-exact unembed (gemma4_rms_norm -> bf16 normed vector), fp32 GEMM output, softcap. v [K, D]."""
    return v_logits_fp32(gemma4_rms_norm(v, lab.fnw))


class VRunFrom:
    """Replay cached resid_post[l-1] so a forward recomputes only layers l..59 (layers < l-1 -> identity)."""

    def __init__(self, l, cached_prev):
        self.l, self.cached = l, cached_prev

    def __enter__(self):
        self.patched = []
        for j in range(self.l):
            lay = lab.layers[j]
            if j == self.l - 1:
                c = self.cached
                lay.forward = (lambda hidden_states, *a, _c=c, **k: _c)
            else:
                lay.forward = (lambda hidden_states, *a, **k: hidden_states)
            self.patched.append(lay)
        return self

    def __exit__(self, *exc):
        for lay in self.patched:
            try:
                del lay.forward
            except AttributeError:
                pass
        return False


def v_spearman_gpu(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float()
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()))

#!/usr/bin/env python3
"""A CPU-only check of the causal spot-check (backend/causal_check.py, the scientist tool `causal_spot_check` in
backend/mcp_scientist.py, and its POST /api/scientist/causal_check route and WebSocket "causal_check" action in
backend/server.py). It loads neither the real model nor the lens, and hides the GPU.

The stand-in is a tiny random Gemma 4 from transformers (`Gemma4ForCausalLM`), built as scripts/check_capture_rows.py
builds it (6 layers, the sixth with full attention, a sliding window of 8 word-pieces, attention_k_eq_v, partial rotary,
the final softcap of 30), but with the real vocabulary (262,144), so the study's 36 texts can be read through the
instrument's own tokenizer (from the local cache). A stand-in TheodoliteRuntime holds it, with a random bf16 "lens" at
each layer (the identity at the last) and the shipped lens's provenance, so the lens card is the real one. It checks:
1. the base row and a zero push in the same forward are bit-identical (torch.equal, KL exactly 0), in given mode
   (over the shared prefix, and over the whole text) and in study mode (right-padded), in float32 and bfloat16;
2. the activation write: the pushed rows' down_proj input differs from the base rows' only at (read position,
   neuron), by the push;
3. causality: every decoder layer's output before the read position is the base row's, bit for bit;
4. the frozen-norm write: the post-feed-forward norm's output changes only at the read position, by
   delta * (w_post (.) w_down) / rms(y), and the neuron's own activation is untouched;
5. the closed-form linearity (N1V_audit_01_postnorm.py:90-104) equals the one measured by hooks on the norm, for a
   gentle push and a squashing one;
6. as the push shrinks, the two writes agree (Spearman of their c above 0.99) and the two faces are opposite (c+ and
   c- at Spearman below -0.99);
7. study_max_abs equals each text's own maximum read alone, leaving out the start mark (which would change it) and
   the padding; text_max_abs likewise;
8. each padded study row's c equals the same text pushed alone (given mode over the whole text);
9. given mode over the shared prefix equals given mode over the whole text (allclose), at lengths before and past the
   window, for both writes;
10. running out of memory (simulated): the pushed rows are split, halving to one per forward, each forward with its
   own base row; the result equals one forward's; no hook and no cache patch is left; base plus one still failing is
   raised; the slot cap does the same;
11. movers are sorted and bounded, and p_at matches a brute-force count;
12. STUDY_TEXTS are N1A's lists (experiments/scripts/N1A_common.py, parsed with ast), and the tool reads them as the
   study did: the chats through the chat template exactly as lab.chat, the ids exactly as lab.ids
   (experiments/lab/labkit.py);
13. the tool: the reply's keys (label_gate and lens last, steered false), both faces, the floor, the label counts
   (the same ids, so the same shared_top10, as inspect_pole's label_check), a frozen-norm write's null linearity with
   its reason, given mode by ids and by prompt, study mode, scale_ref "this_text", the study max|a| cached per set of
   vault edits in force and reused after a ROLLBACK, an edit on the neuron marked; every refusal in plain words before
   any forward; running out of GPU memory coming back as a ValueError;
14. the transports (fastapi TestClient, the model-loading lifespan replaced): the WebSocket action answers
   {type "causal_check"} with the MCP reply's keys; its refusals (pydantic's too, in plain words) come back as
   recoverable errors naming action, layer, neuron_idx and contexts, and the socket keeps answering; HTTP gives 200,
   400 and 422; /api/health carries the causal_check card;
15. the study's figures the folio quotes beside a test (frontend/src/screens/NeuronFolio/causal.ts: STUDY_LINEARITY,
   LINEARITY_TAIL, NOISE_RATIO_MAX, STUDY_K10) are the results files' own: N1V_audit_01_postnorm.json.full.json,
   N1V_audit_04_smallk_metrics.json and N1V_code_audit.md F2.
What it cannot show is the real weights' numbers on the GPU: scripts/verify_causal_check.py reproduces the study's.

Run: .venv/bin/python -I scripts/check_causal_check.py   (exits 0 when every check holds)
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    # nothing here needs the GPU, and nothing may be fetched: the tokenizer comes from the local cache
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"

import ast
import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

import backend.causal_check as cc
import backend.mcp_scientist as ms
from backend.engine_runtime import DEFAULT_W4A16_MODEL_ID, JPP_FITTED_MODEL, TheodoliteRuntime
from backend.label_gate import LABEL_GATE_CARD
from backend.larql_vault import LarqlVaultEngine, PatchOperationRecord
from interp_engine import EagerModel

N_LAYERS = 6
WINDOW = 8
HIDDEN = 64
VOCAB = 262144
BOS = 2
L, I = 3, 17  # the neuron pushed in most checks

failures: List[str] = []
passed = 0


def check(ok: bool, what: str) -> None:
    global passed
    if ok:
        passed += 1
    else:
        failures.append(what)
        print(f"  FAIL  {what}")


def section(title: str, before: int) -> None:
    if len(failures) == before:
        print(f"  ok    {title}")


def stand_in_hf(dtype: torch.dtype, attn: str = "sdpa") -> Any:
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    torch.manual_seed(0)
    cfg = Gemma4TextConfig(
        vocab_size=VOCAB,
        vocab_size_per_layer_input=VOCAB,
        hidden_size_per_layer_input=0,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_global_key_value_heads=1,
        head_dim=16,
        global_head_dim=32,
        attention_k_eq_v=True,
        sliding_window=WINDOW,
        layer_types=["sliding_attention"] * (N_LAYERS - 1) + ["full_attention"],
        final_logit_softcapping=30.0,
        max_position_embeddings=4096,
        rope_parameters={
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        },
    )
    cfg._attn_implementation = attn
    hf = Gemma4ForCausalLM(cfg).to(dtype).eval()
    # the layer norms and scalars start at plain values: give them some spread, so no layer passes its input through
    with torch.no_grad():
        for name, p in hf.named_parameters():
            if p.dim() == 1 or name.endswith("layer_scalar"):
                p.copy_((torch.rand_like(p, dtype=torch.float32) * 0.8 + 0.6).to(p.dtype))
    return hf


def stand_in_runtime(hf: Any, tokenizer: Any) -> TheodoliteRuntime:
    """A TheodoliteRuntime holding the stand-in, its own methods (slice_w_down, compose_prompt, tokenize_prompt,
    lens_card) unchanged."""
    rt = TheodoliteRuntime(device="cpu", load_jlens=False)
    rt.eager_model = EagerModel("stand-in", device="cpu", dtype=hf.dtype, hf_model=hf, tokenizer=tokenizer)
    rt.hf_model = hf
    rt.tokenizer = tokenizer
    lm = cc.language_model(hf)
    rt._decoder_layers = lm.layers
    rt._final_norm_weight = lm.norm.weight.detach()
    rt._lm_head_weight = hf.lm_head.weight.detach()
    g = torch.Generator().manual_seed(5)
    for l, block in enumerate(lm.layers):
        rt._pre_ffn_norm_weights[l] = block.pre_feedforward_layernorm.weight.detach()
        rt._post_ffn_norm_weights[l] = block.post_feedforward_layernorm.weight.detach()
        rt.jpp_lenses[l] = (
            torch.eye(HIDDEN, dtype=torch.bfloat16)
            if l == N_LAYERS - 1
            else (torch.eye(HIDDEN) + 0.3 * torch.randn(HIDDEN, HIDDEN, generator=g)).to(torch.bfloat16)
        )
    rt.jpp_provenance = {"source": "stand-in", "method": "jpp", "lrp_mode": "rlens", "hf_model_name": JPP_FITTED_MODEL,
                         "n_prompts": 64, "max_seq_len": 128}
    return rt


def rand_ids(n: int, g: torch.Generator) -> List[int]:
    return [BOS] + [int(t) for t in torch.randint(3, 20000, (n - 1,), generator=g)]


def parts(lm: Any, l: int = L, i: int = I):
    down = lm.layers[l].mlp.down_proj
    norm = lm.layers[l].post_feedforward_layernorm
    return down.weight[:, i].detach(), norm.weight.detach(), float(norm.eps)


def run(lm: Any, deltas: Sequence[float], write: str = "activation", **kw: Any) -> cc.CheckRun:
    wd, wp, eps = parts(lm)
    W = lm.embed_tokens.weight  # tied: the stand-in's lm_head
    return cc.run_check(lm, L, I, list(deltas), write, wd, wp, eps, W, **kw)


def cmat(r: cc.CheckRun, s: int) -> torch.Tensor:
    return r.steered[s] - r.bases[s]


def all_hooks_gone(lm: Any) -> bool:
    for m in lm.modules():
        if m._forward_pre_hooks or m._forward_hooks:
            return False
    return True


class Capture:
    """Record a module's input (pre) or output (post) on every call; registered before run_check's, so it sees the
    written values (the write hooks go first)."""

    def __init__(self, module: Any, kind: str) -> None:
        self.seen: List[torch.Tensor] = []
        if kind == "pre":
            self.h = module.register_forward_pre_hook(lambda _m, a: self.seen.append(a[0].detach().clone()))
        elif kind == "in":
            self.h = module.register_forward_hook(lambda _m, a, _o: self.seen.append(a[0].detach().clone()))
        else:
            self.h = module.register_forward_hook(
                lambda _m, _a, o: self.seen.append((o[0] if isinstance(o, tuple) else o).detach().clone())
            )

    def __enter__(self) -> "Capture":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.h.remove()


# ---------------------------------------------------------------------------------------------------------------------


def check_base_identity(lm: Any, tag: str) -> None:
    before = len(failures)
    g = torch.Generator().manual_seed(11)
    d = 0.4
    for kv in (True, False):
        for n in (3, 12):
            r = run(lm, [0.0, d, -d], token_ids=rand_ids(n, g), kv_cache=kv)
            check(torch.equal(r.steered[0], r.base), f"{tag}: given ({'prefix' if kv else 'whole'}, {n}): zero push != base")
            check(float(cc.kl(r.steered[0], r.base).abs().max()) == 0.0, f"{tag}: given, zero push KL not exactly 0")
            check(not torch.equal(r.steered[1], r.base), f"{tag}: given, the push changed nothing")
    texts = [rand_ids(n, g) for n in (4, 9, 13, 6)]
    for write in cc.CAUSAL_WRITES:
        r = run(lm, [0.0, d, -d], write, texts_ids=texts)
        check(torch.equal(r.steered[0], r.base), f"{tag}: study ({write}): zero push != base")
        check(float(cc.kl(r.steered[0], r.base).abs().max()) == 0.0, f"{tag}: study, zero push KL not exactly 0")
    section(f"{tag}: a zero push is the base row, bit for bit (given over the prefix and whole, study padded)", before)


def check_activation_write(lm: Any) -> None:
    before = len(failures)
    g = torch.Generator().manual_seed(12)
    d = 0.7
    down = lm.layers[L].mlp.down_proj
    ids = rand_ids(11, g)
    with Capture(down, "pre") as cap:
        run(lm, [0.0, d, -d], token_ids=ids)  # over the prefix: one position per row
    x = cap.seen[-1]  # [4, 1, d_ff]
    for s, dd in ((1, 0.0), (2, d), (3, -d)):
        diff = x[s] - x[0]
        mask = torch.zeros_like(diff, dtype=torch.bool)
        mask[0, I] = True
        check(torch.equal(diff[~mask], torch.zeros_like(diff[~mask])), f"activation write: row {s} changed elsewhere")
        check(abs(float(diff[0, I]) - dd) < 1e-6, f"activation write: row {s} changed by {float(diff[0, I])}, not {dd}")
    texts = [rand_ids(n, g) for n in (5, 9, 3)]
    C = len(texts)
    last = [len(t) - 1 for t in texts]
    with Capture(down, "pre") as cap:
        run(lm, [d], texts_ids=texts)
    x = cap.seen[-1]  # [2C, T, d_ff]
    for c in range(C):
        diff = x[C + c] - x[c]
        mask = torch.zeros_like(diff, dtype=torch.bool)
        mask[last[c], I] = True
        check(torch.equal(diff[~mask], torch.zeros_like(diff[~mask])), f"study write, text {c}: changed elsewhere")
        check(abs(float(diff[last[c], I]) - d) < 1e-6, f"study write, text {c}: not by delta")
    section("the activation write lands only at (read position, neuron), by delta", before)


def check_causality(lm: Any) -> None:
    before = len(failures)
    g = torch.Generator().manual_seed(13)
    texts = [rand_ids(n, g) for n in (6, 14, 9)]
    C = len(texts)
    caps = [Capture(layer, "out") for layer in lm.layers]
    try:
        for write in cc.CAUSAL_WRITES:
            for cap in caps:
                cap.seen.clear()
            run(lm, [5.0, -5.0], write, texts_ids=texts)
            for li, cap in enumerate(caps):
                h = cap.seen[-1]  # [3C, T, D]
                for s in (1, 2):
                    for c, t in enumerate(texts):
                        p = len(t) - 1
                        check(torch.equal(h[s * C + c, :p], h[c, :p]),
                              f"causality ({write}): layer {li}, slot {s}, text {c} changed before the read position")
                if li >= L:
                    check(not torch.equal(h[C, len(texts[0]) - 1], h[0, len(texts[0]) - 1]),
                          f"causality ({write}): layer {li} unchanged at the read position")
    finally:
        for cap in caps:
            cap.h.remove()
    section("nothing before the read position changes, at any layer, for either write", before)


def check_frozen_norm(lm: Any) -> None:
    before = len(failures)
    g = torch.Generator().manual_seed(14)
    texts = [rand_ids(n, g) for n in (7, 10)]
    C = len(texts)
    d = 0.9
    wd, wp, eps = parts(lm)
    norm = lm.layers[L].post_feedforward_layernorm
    down = lm.layers[L].mlp.down_proj
    with Capture(norm, "out") as out_cap, Capture(norm, "in") as in_cap, Capture(down, "pre") as act_cap:
        run(lm, [d, -d], "frozen_norm", texts_ids=texts)
    o, y, x = out_cap.seen[-1], in_cap.seen[-1], act_cap.seen[-1]
    u = wp.float() * wd.float()
    for s, dd in ((1, d), (2, -d)):
        check(torch.equal(x[s * C : (s + 1) * C], x[:C]), "frozen norm: the neuron's activation was changed")
        for c, t in enumerate(texts):
            p = len(t) - 1
            yb = y[c, p].float()
            rms = (yb.pow(2).mean() + eps).sqrt()
            want = dd * u / rms
            got = o[s * C + c, p].float() - o[c, p].float()
            check(float((got - want).norm() / want.norm()) < 1e-5, f"frozen norm, text {c}: wrong change at p")
            check(torch.equal(o[s * C + c, :p], o[c, :p]) and torch.equal(o[s * C + c, p + 1 :], o[c, p + 1 :]),
                  f"frozen norm, text {c}: changed away from p")
    section("the frozen-norm write adds delta * w_post (.) w_down / rms(y) at the read position only", before)


def check_linearity(lm: Any) -> None:
    before = len(failures)
    g = torch.Generator().manual_seed(15)
    texts = [rand_ids(n, g) for n in (6, 11, 8)]
    C = len(texts)
    wd, wp, eps = parts(lm)
    norm = lm.layers[L].post_feedforward_layernorm
    seen = []
    for d in (0.01, 0.3, 3.0):
        with Capture(norm, "out") as cap:
            r = run(lm, [d], texts_ids=texts)
        o = cap.seen[-1]
        lr = cc.linear_regime(r.y, wd, wp, d, eps)
        for c, t in enumerate(texts):
            p = len(t) - 1
            change = o[C + c, p].float() - o[c, p].float()
            rms = (r.y[c].pow(2).mean() + eps).sqrt()
            lin = wp.float() * d * wd.float() / rms
            measured = float(change.norm() / lin.norm())
            closed = float(lr["linearity"][c])
            check(abs(measured - closed) < 1e-4 * max(1.0, closed),
                  f"linearity at delta {d}, text {c}: measured {measured:.6f}, closed form {closed:.6f}")
            check(abs(float(lr["write_over_ffn"][c]) - float(d * wd.float().norm() / r.y[c].norm())) < 1e-6,
                  "write_over_ffn is not |delta| ||w_down|| / ||y||")
        seen.append(float(lr["linearity"].median()))
    check(seen[-1] < 0.5 * seen[0], f"a strong push did not squash: linearity {seen}")
    section(f"closed-form linearity equals the hooks' (median {', '.join(f'{v:.3f}' for v in seen)})", before)


def check_small_k(lm: Any) -> None:
    before = len(failures)
    g = torch.Generator().manual_seed(16)
    ids = rand_ids(15, g)
    d = 1e-3
    ra = run(lm, [d, -d], "activation", token_ids=ids)
    rf = run(lm, [d], "frozen_norm", token_ids=ids)
    rho_writes = cc.spearman(cmat(ra, 0)[0], cmat(rf, 0)[0])
    rho_faces = cc.spearman(cmat(ra, 0)[0], cmat(ra, 1)[0])
    check(rho_writes > 0.99, f"small k: the two writes disagree (Spearman {rho_writes:.4f})")
    check(rho_faces < -0.99, f"small k: c+ and c- are not opposite (Spearman {rho_faces:.4f})")
    section(f"at a small push the writes agree (rho {rho_writes:.4f}) and the faces oppose (rho {rho_faces:.4f})", before)


def check_max_abs(lm: Any) -> None:
    before = len(failures)
    g = torch.Generator().manual_seed(17)
    texts = [rand_ids(n, g) for n in (5, 12, 8, 3)]
    inp, att, _last, valid = cc.pad_right(texts, torch.device("cpu"))
    table = cc.study_max_abs(lm, inp, att, valid)
    check(tuple(table.shape) == (N_LAYERS, lm.layers[0].mlp.down_proj.in_features), f"study_max_abs shape {tuple(table.shape)}")
    alone_after, alone_all = [], []
    for t in texts:
        acts: Dict[int, torch.Tensor] = {}
        hooks = [
            layer.mlp.down_proj.register_forward_pre_hook(lambda _m, a, l=l: acts.__setitem__(l, a[0][0].abs().float()))
            for l, layer in enumerate(lm.layers)
        ]
        try:
            lm(input_ids=torch.tensor([t]), use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        alone_after.append(torch.stack([acts[l][1:].max(0).values for l in range(N_LAYERS)]))
        alone_all.append(torch.stack([acts[l].max(0).values for l in range(N_LAYERS)]))
    want = torch.stack(alone_after).max(0).values
    with_bos = torch.stack(alone_all).max(0).values
    check(torch.allclose(table, want, rtol=1e-5, atol=1e-6), "study_max_abs differs from each text read alone")
    differs = int((with_bos > want * (1 + 1e-4)).sum())
    check(differs > 0, "no neuron's maximum is at the start mark: the exclusion went untested")
    check(not torch.allclose(table, with_bos, rtol=1e-5, atol=1e-6), "study_max_abs counted the start mark")
    t = texts[1]
    check(abs(cc.text_max_abs(lm, L, I, t) - float(alone_after[1][L, I])) < 1e-6, "text_max_abs differs")
    section(f"max|a| leaves out the start mark ({differs} maxima would move) and the padding", before)


def check_padded_vs_alone(lm: Any) -> None:
    before = len(failures)
    g = torch.Generator().manual_seed(18)
    texts = [rand_ids(n, g) for n in (4, 13, 7, 10)]
    for write in cc.CAUSAL_WRITES:
        r = run(lm, [0.5, -0.5], write, texts_ids=texts)
        for c, t in enumerate(texts):
            a = run(lm, [0.5, -0.5], write, token_ids=t, kv_cache=False)
            for s in (0, 1):
                ok = torch.allclose(cmat(r, s)[c], cmat(a, s)[0], atol=2e-5, rtol=1e-4)
                check(ok, f"padded vs alone ({write}), text {c}, slot {s}: max diff "
                          f"{float((cmat(r, s)[c] - cmat(a, s)[0]).abs().max()):.2e}")
            check(abs(float(r.act[c]) - float(a.act[0])) < 1e-5, f"padded vs alone: activation differs, text {c}")
    section("each padded study row reads as its text pushed alone", before)


def check_prefix_vs_whole(lm: Any) -> None:
    before = len(failures)
    g = torch.Generator().manual_seed(19)
    worst = 0.0
    for n in (2, 3, WINDOW, WINDOW + 1, 20, 33):
        ids = rand_ids(n, g)
        for write in cc.CAUSAL_WRITES:
            a = run(lm, [0.6, -0.6, 0.006], write, token_ids=ids, kv_cache=True)
            b = run(lm, [0.6, -0.6, 0.006], write, token_ids=ids, kv_cache=False)
            for s in range(3):
                diff = float((cmat(a, s) - cmat(b, s)).abs().max())
                worst = max(worst, diff)
                check(diff < 2e-5, f"prefix vs whole ({write}, {n} pieces, slot {s}): c differs by {diff:.2e}")
            check(float((a.base - b.base).abs().max()) < 2e-5, f"prefix vs whole ({write}, {n}): base differs")
            check(torch.allclose(a.y, b.y, atol=1e-5) and torch.allclose(a.act, b.act, atol=1e-6),
                  f"prefix vs whole ({write}, {n}): y or the activation differs")
    section(f"given mode over the shared prefix equals the whole text (worst {worst:.1e}), past the window too", before)


def check_oom(lm: Any) -> None:
    before = len(failures)
    g = torch.Generator().manual_seed(20)
    ids = rand_ids(12, g)
    texts = [rand_ids(n, g) for n in (5, 8)]
    deltas = [0.5, -0.5, 0.005, -0.005]
    limit = {"rows": 10**9}

    def fail_wide(_m: Any, args: Any, kwargs: Any) -> None:
        h = kwargs.get("hidden_states", args[0] if args else None)
        if h is not None and h.shape[0] > limit["rows"]:
            raise torch.cuda.OutOfMemoryError("simulated: the card is full")

    for label, kw, C in (("given", {"token_ids": ids}, 1), ("study", {"texts_ids": texts}, 2)):
        one = run(lm, deltas, **kw)
        guard = lm.layers[1].register_forward_pre_hook(fail_wide, with_kwargs=True)
        try:
            limit["rows"] = 2 * C  # only the base and one pushed row fit
            split = run(lm, deltas, **kw)
        finally:
            guard.remove()
        check(split.forwards == 4 and split.slots_per_forward == [1, 1, 1, 1],
              f"oom ({label}): {split.forwards} forwards, {split.slots_per_forward}")
        check(split.base_rows_identical, f"oom ({label}): the base rows of the forwards differ")
        for s in range(4):
            check(torch.allclose(cmat(split, s), cmat(one, s), atol=2e-5), f"oom ({label}): slot {s} differs")
        capped = run(lm, deltas, max_slots=3, **kw)
        check(capped.slots_per_forward == [3, 1], f"slot cap ({label}): {capped.slots_per_forward}")
        guard = lm.layers[1].register_forward_pre_hook(fail_wide, with_kwargs=True)
        raised = False
        try:
            limit["rows"] = C  # not even the base and one
            run(lm, deltas, **kw)
        except torch.cuda.OutOfMemoryError:
            raised = True
        finally:
            guard.remove()
        check(raised, f"oom ({label}): base plus one failing was not raised")
        check(all_hooks_gone(lm), f"oom ({label}): a hook was left behind")
    # the shared prefix's cache: no patched update left on any layer after a failed and a good forward
    seen_caches: List[Any] = []
    real = cc._frozen_prefix

    def spy(cache: Any) -> Any:
        seen_caches.append(cache)
        return real(cache)

    cc._frozen_prefix = spy  # type: ignore[assignment]
    try:
        run(lm, deltas, token_ids=ids)
    finally:
        cc._frozen_prefix = real  # type: ignore[assignment]
    check(bool(seen_caches) and all("update" not in vars(layer) for c in seen_caches for layer in c.layers),
          "a cache layer kept its frozen update")
    section("out of memory: halving to one pushed row per forward, own base rows, nothing left behind", before)


def check_movers_and_p_at() -> None:
    before = len(failures)
    g = torch.Generator().manual_seed(21)
    c = torch.randn(5000, generator=g)
    rises, falls = cc.movers(c, 12)
    check(len(rises) == 12 and len(falls) == 12, "movers: wrong lengths")
    check(all(c[a] >= c[b] for a, b in zip(rises, rises[1:])), "rises are not highest first")
    check(all(c[a] <= c[b] for a, b in zip(falls, falls[1:])), "falls are not lowest first")
    order = sorted(range(c.numel()), key=lambda t: -float(c[t]))
    check(rises == order[:12] and falls == order[::-1][:12], "movers are not the extremes")
    for _ in range(20):
        lab = [int(t) for t in torch.randint(0, 5000, (10,), generator=g)]
        lab[:3] = order[5:8]  # a few certainly in the top hundred
        ref = set(order[:100])
        want = sum(1 for t in lab if t in ref) / len(lab)
        check(abs(cc.p_at(lab, c) - want) < 1e-12, "p_at differs from a brute-force count")
    section("movers sorted and bounded; P@10 as counted by hand", before)


def check_study_texts(rt: TheodoliteRuntime) -> None:
    before = len(failures)
    src = ROOT_DIR / "experiments" / "scripts" / "N1A_common.py"
    tree = ast.parse(src.read_text())
    lists: Dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("_CHAT", "_PROSE", "_CODE"):
                lists[node.targets[0].id] = ast.literal_eval(node.value)
    want = [("chat", u, a) for u, a in lists["_CHAT"]] + [("prose", t) for t in lists["_PROSE"]]
    want += [("code", t) for t in lists["_CODE"]]
    check(list(cc.STUDY_TEXTS) == want, "STUDY_TEXTS differ from N1A_common.py's _CHAT, _PROSE and _CODE")
    check(len(cc.STUDY_TEXTS) == 36, f"{len(cc.STUDY_TEXTS)} study texts")
    # as labkit composes and tokenizes them (experiments/lab/labkit.py: chat(), ids(), BOS_ID 2)
    tok = rt.tokenizer
    got = ms._study_ids(rt)
    for (kind, *rest), ids in zip(cc.STUDY_TEXTS, got):
        if kind == "chat":
            s = tok.apply_chat_template([{"role": "user", "content": rest[0]}], tokenize=False, add_generation_prompt=True)
            s += rest[1]
            lab = tok.encode(s, add_special_tokens=False) if s.startswith("<bos>") else [BOS] + tok.encode(s, add_special_tokens=False)
            check(rt.compose_prompt(rest[0], chat=True, assistant_prefix=rest[1]) == s, f"chat text differs: {rest[0]!r}")
        else:
            lab = [BOS] + tok.encode(rest[0], add_special_tokens=False)
        check(ids == lab, f"the ids of {rest[0][:30]!r} differ from labkit's")
    section("STUDY_TEXTS are N1A's 36, composed and tokenized as the study did", before)


# ---------------------------------------------------------------------------------------------------------------------

REPLY_KEYS = [
    "layer", "neuron_idx", "polarity", "contexts", "n_contexts", "prompt", "chat", "token_ids_given", "read_pos",
    "read_pos_by_context", "read_token_id", "read_token", "study_texts", "steered", "write", "k", "k_floor",
    "scale_ref", "max_abs_act", "max_abs_act_over", "max_abs_act_cached", "act_here", "act_here_by_context", "faces",
    "antipodal", "floor", "base_top", "forwards", "slots_per_forward", "base_rows_identical", "vault_edits",
    "timing_ms", "label_gate", "lens",
]
FACE_KEYS = [
    "delta", "kl_mean", "kl_median", "kl_by_context", "kl_over_floor", "split_half_rho", "rises", "falls", "tracked",
    "floor_top100_overlap", "linear", "label_overlap",
]
WORD_KEYS = ["token_id", "token", "dlogprob", "base_logprob", "steered_logprob"]


def _ratio(a: float, b: Optional[float]) -> Optional[float]:
    return ms._sig(a / b, 4) if b is not None and b > 0 else None


def tool(**kw: Any) -> Dict[str, Any]:
    return asyncio.run(ms.causal_spot_check(**kw))


def refused(**kw: Any) -> Optional[str]:
    try:
        tool(**kw)
    except ValueError as exc:
        return str(exc)
    return None


def check_tool(rt: TheodoliteRuntime, vault: LarqlVaultEngine) -> Dict[str, Any]:
    before = len(failures)
    g = torch.Generator().manual_seed(22)
    ids = rand_ids(14, g)
    calls = {"study_max_abs": 0, "run_check": 0}
    real_sma, real_run = cc.study_max_abs, cc.run_check

    def counted_sma(*a: Any, **k: Any) -> Any:
        calls["study_max_abs"] += 1
        return real_sma(*a, **k)

    def counted_run(*a: Any, **k: Any) -> Any:
        calls["run_check"] += 1
        return real_run(*a, **k)

    cc.study_max_abs, cc.run_check = counted_sma, counted_run  # type: ignore[assignment]
    try:
        r = tool(layer=L, neuron_idx=I, token_ids=ids, track_ids=[ids[-1], 7])
        check(list(r) == REPLY_KEYS, f"reply keys: {[k for k in r if k not in REPLY_KEYS]} extra, "
                                     f"{[k for k in REPLY_KEYS if k not in r]} missing")
        check(list(r)[-2:] == ["label_gate", "lens"] and r["label_gate"] == LABEL_GATE_CARD
              and r["lens"] == rt.lens_card() and r["lens"]["id"] == "jpp-base", "label_gate and lens are not last")
        check(r["steered"] is False and r["contexts"] == "given" and r["n_contexts"] == 1 and r["read_pos"] == 13
              and r["token_ids_given"] is True and r["read_token_id"] == ids[-1], "given mode's echo")
        check(r["k"] == 10.0 and r["k_floor"] == 0.01 and r["write"] == "activation" and r["scale_ref"] == "study",
              "the defaults are not k 10, floor 0.01, activation, study")
        check(r["forwards"] == 1 and r["slots_per_forward"] == [4] and r["base_rows_identical"] is True,
              f"one forward of four pushes: {r['forwards']} {r['slots_per_forward']}")
        check(list(r["faces"]) == ["+", "-"] and all(list(f) == FACE_KEYS for f in r["faces"].values()), "face keys")
        check(r["polarity"] == ("+" if r["act_here"] >= 0 else "-"), "polarity None is not the face fired here")
        inp, att, _last, valid = cc.pad_right(ms._study_ids(rt), torch.device("cpu"))
        study = real_sma(cc.language_model(rt.hf_model), inp, att, valid)
        check(abs(r["max_abs_act"] - float(study[L, I])) <= 1e-5 * float(study[L, I]), "max_abs_act is not the study's")
        for face, s in (("+", 1), ("-", -1)):
            f = r["faces"][face]
            check(abs(f["delta"] - s * 10 * r["max_abs_act"]) <= 1e-4 * abs(f["delta"]), f"face {face}: delta")
            check(len(f["rises"]) == 10 and len(f["falls"]) == 10 and all(list(w) == WORD_KEYS for w in f["rises"]),
                  f"face {face}: movers")
            check(all(a["dlogprob"] >= b["dlogprob"] for a, b in zip(f["rises"], f["rises"][1:])), f"face {face}: rises")
            check(all(a["dlogprob"] <= b["dlogprob"] for a, b in zip(f["falls"], f["falls"][1:])), f"face {face}: falls")
            check([w["token_id"] for w in f["tracked"]] == [ids[-1], 7], f"face {face}: tracked")
            lo = f["label_overlap"]
            for side in ("lens", "naked_eye"):
                o = lo[side]
                check(len(o["label_ids"]) == 10 and o["p10"] == sum(o["in_top100"]) / 10, f"face {face} {side}: p10")
            check(lo["label_k"] == 10 and lo["causal_k"] == 100, "label_k / causal_k")
            pole = vault.inspect_bipolar_slot(L, I, polarity=s)["active_pole"]["label_check"]
            check(lo["label_check"] == pole, f"face {face}: label_check differs from inspect_pole's ({lo['label_check']} "
                                             f"vs {pole})")
            check(len(set(lo["lens"]["label_ids"]) & set(lo["naked_eye"]["label_ids"])) == pole["shared_top10"],
                  f"face {face}: the label ids do not give inspect_pole's count")
            lin = f["linear"]
            check(lin["linearity"] is not None and 0 < lin["linearity"] <= 1.0001 and "linearity_reason" not in lin,
                  f"face {face}: linearity {lin.get('linearity')}")
            fl_kl = r["floor"][face]["kl_mean"]
            # a float32 floor push moves the stand-in's odds by less than float32 rounding, so its KL may come out at
            # or below 0: then there is no ratio (on the real model's bf16 path the floor is ~1e-4, N1V_robustness.md:45)
            want = _ratio(f["kl_mean"], fl_kl)
            got = f["kl_over_floor"]
            same = got is None if want is None else got is not None and abs(got - want) <= 1e-3 * abs(want)
            check(same and 0 <= f["floor_top100_overlap"] <= 100,
                  f"face {face}: floor comparison {f['kl_over_floor']} (want {want}) {f['floor_top100_overlap']}")
            check(f["split_half_rho"] is None, "given mode has a split-half rho")
        check(set(r["floor"]) == {"+", "-"} and abs(r["floor"]["+"]["delta"] - 0.01 * r["max_abs_act"]) < 1e-6,
              "the floor's pushes")
        check(len(r["base_top"]) == 5 and -1 <= r["antipodal"]["rho"] <= 1, "base_top / antipodal")
        check(calls["study_max_abs"] == 1 and r["max_abs_act_cached"] is False, "the study max|a| was not computed once")

        # cached, then recomputed for a vault edit in force, reused after ROLLBACK
        r2 = tool(layer=L, neuron_idx=I, token_ids=ids, write="frozen_norm", floor=False, top_n=3)
        check(r2["max_abs_act_cached"] is True and calls["study_max_abs"] == 1, "the study max|a| was not reused")
        lin2 = r2["faces"]["+"]["linear"]
        check(lin2["linearity"] is None and lin2["linearity_q10"] is None and lin2["by_context"]["linearity"] is None
              and lin2["linearity_reason"] == cc.FROZEN_NORM_NO_LINEARITY, "frozen norm: linearity is not null")
        check(r2["floor"] is None and r2["k_floor"] is None and r2["faces"]["+"]["kl_over_floor"] is None
              and len(r2["faces"]["-"]["rises"]) == 3 and r2["slots_per_forward"] == [2], "floor off / top_n")
        vault.active_patches.append(PatchOperationRecord(
            patch_id=41, op_type="UPDATE", layer_idx=L, neuron_idx=I, polarity=1.0, target_concept="x", strength=1.0,
            statement="UPDATE (stand-in)", timestamp=time.time(), orig_norm=1.0, new_norm=1.0))
        r3 = tool(layer=L, neuron_idx=I, token_ids=ids)
        check(r3["max_abs_act_cached"] is False and calls["study_max_abs"] == 2 and r3["vault_edits"] == [41],
              "an edit in force did not recompute the study max|a|")
        check(r3["faces"]["+"]["label_overlap"]["label_check"]["edited"] is True, "an edited neuron is not marked")
        vault.active_patches.clear()  # what ROLLBACK does to the records (its overlay sweep runs over 60 layers)
        r4 = tool(layer=L, neuron_idx=I, token_ids=ids)
        check(r4["max_abs_act_cached"] is True and calls["study_max_abs"] == 2 and r4["vault_edits"] == [],
              "the clean model's max|a| was not reused after ROLLBACK")
        check(r4["faces"]["+"]["rises"] == r["faces"]["+"]["rises"], "the same check twice differs")

        # a silenced neuron (a DELETE in force leaves its column zero, larql_vault.py): turning it up writes nothing, so
        # nothing moves beyond the untouched row, its labels have no P@10, and there is no floor ratio (the folio then
        # gives no words that "rose most" in its main reading)
        down = cc.language_model(rt.hf_model).layers[L].mlp.down_proj
        col = down.weight[:, I].clone()
        with torch.no_grad():
            down.weight[:, I] = 0
        vault.active_patches.append(PatchOperationRecord(
            patch_id=42, op_type="DELETE", layer_idx=L, neuron_idx=I, polarity=1.0, target_concept="", strength=1.0,
            statement="DELETE (stand-in)", timestamp=time.time(), orig_norm=1.0, new_norm=0.0))
        try:
            rd = tool(layer=L, neuron_idx=I, token_ids=ids)
        finally:
            with torch.no_grad():
                down.weight[:, I] = col
            vault.active_patches.clear()
        check(rd["vault_edits"] == [42], f"the DELETE is not in force: {rd['vault_edits']}")
        for face in ("+", "-"):
            fd = rd["faces"][face]
            lo = fd["label_overlap"]
            check("writes_nothing" in lo["label_check"]["reasons"] and lo["label_check"]["held_back"] is True
                  and lo["lens"]["p10"] is None and lo["naked_eye"]["p10"] is None,
                  f"silenced face {face}: label {lo['label_check']} p10 {lo['lens']['p10']} {lo['naked_eye']['p10']}")
            moved = [abs(w["dlogprob"]) for w in fd["rises"] + fd["falls"] + fd["tracked"]]
            check(max(moved) <= 1e-6 and abs(fd["kl_mean"]) <= 1e-9 and fd["kl_over_floor"] is None,
                  f"silenced face {face}: moved {max(moved)}, KL {fd['kl_mean']}, over floor {fd['kl_over_floor']}")

        # by prompt, study mode, this text
        rp = tool(layer=L, neuron_idx=I, prompt="The cat sat on the", chat=True, polarity=-1)
        check(rp["token_ids_given"] is False and rp["prompt"].startswith("<bos>") and rp["polarity"] == "-"
              and list(rp["faces"]) == ["+", "-"], "given mode by prompt")
        rs = tool(layer=L, neuron_idx=I, contexts="study")
        check(rs["n_contexts"] == 36 and rs["read_pos"] is None and len(rs["read_pos_by_context"]) == 36
              and len(rs["act_here_by_context"]) == 36 and rs["act_here"] is None and rs["prompt"] is None
              and rs["base_top"] is None and rs["study_texts"] == cc.STUDY_TEXTS_SOURCE and rs["polarity"] == "+",
              "study mode's echo")
        # the figure behind the folio's "beyond / no more than rounding noise": on the study's texts the stand-in's floor
        # KL is above 0, so the ratio is there, each face over its own floor's mean KL
        for face in ("+", "-"):
            fs, fl = rs["faces"][face], rs["floor"][face]
            want = _ratio(fs["kl_mean"], fl["kl_mean"])
            got = fs["kl_over_floor"]
            check(want is not None and got is not None and abs(got - want) <= 1e-3 * abs(want)
                  and abs(got - float(torch.tensor(fs["kl_by_context"]).mean()) / fl["kl_mean"]) <= 2e-3 * abs(got),
                  f"study face {face}: kl_over_floor {got}, want {want} (KL {fs['kl_mean']}, floor {fl['kl_mean']})")
        f = rs["faces"]["+"]
        check(isinstance(f["split_half_rho"], float) and len(f["kl_by_context"]) == 36
              and len(f["linear"]["by_context"]["linearity"]) == 36 and f["linear"]["linearity_q10"] <= f["linear"]["linearity"],
              "study mode's per-text figures")
        rt_ = tool(layer=L, neuron_idx=I, token_ids=ids, scale_ref="this_text")
        lm = cc.language_model(rt.hf_model)
        check(abs(rt_["max_abs_act"] - cc.text_max_abs(lm, L, I, ids)) < 1e-5 and "this text" in rt_["max_abs_act_over"],
              "scale_ref this_text")

        # refusals, in plain words, before any forward
        n_runs, n_sma = calls["run_check"], calls["study_max_abs"]
        cases = [
            (dict(layer=6, neuron_idx=0, token_ids=ids), "layers count from 0 to 5"),
            (dict(layer=-1, neuron_idx=0, token_ids=ids), "the pages count them 1 to 6"),
            (dict(layer=True, neuron_idx=0, token_ids=ids), "layer is True"),
            (dict(layer=L, neuron_idx=128, token_ids=ids), "neurons count from 0 to 127"),
            (dict(layer=L, neuron_idx=I, token_ids=ids, k=0), "more than 0 and at most 200"),
            (dict(layer=L, neuron_idx=I, token_ids=ids, k=201), "more than 0 and at most 200"),
            (dict(layer=L, neuron_idx=I, token_ids=ids, k=float("nan")), "more than 0 and at most 200"),
            (dict(layer=L, neuron_idx=I, token_ids=ids, k=0.01), "floor's own k = 0.01"),
            (dict(layer=L, neuron_idx=I, token_ids=ids, write="ablate"), '"activation" or "frozen_norm"'),
            (dict(layer=L, neuron_idx=I, token_ids=ids, contexts="corpus"), '"given" or "study"'),
            (dict(layer=L, neuron_idx=I, token_ids=ids, scale_ref="max"), '"study" or "this_text"'),
            (dict(layer=L, neuron_idx=I, token_ids=[BOS]), "needs 2 to 2048"),
            (dict(layer=L, neuron_idx=I, token_ids=[BOS] * 2049), "needs 2 to 2048"),
            (dict(layer=L, neuron_idx=I, token_ids=[BOS, VOCAB]), "not a word-piece id"),
            (dict(layer=L, neuron_idx=I, token_ids=[BOS, -1]), "not a word-piece id"),
            (dict(layer=L, neuron_idx=I), "needs the text it is read on"),
            (dict(layer=L, neuron_idx=I, prompt=""), "needs 2 to 2048"),
            (dict(layer=L, neuron_idx=I, contexts="study", token_ids=ids), "reads those texts only"),
            (dict(layer=L, neuron_idx=I, contexts="study", prompt="x"), "reads those texts only"),
            (dict(layer=L, neuron_idx=I, contexts="study", scale_ref="this_text"), 'use scale_ref "study"'),
            (dict(layer=L, neuron_idx=I, token_ids=ids, top_n=0), "hold 1 to 50"),
            (dict(layer=L, neuron_idx=I, token_ids=ids, top_n=51), "hold 1 to 50"),
            (dict(layer=L, neuron_idx=I, token_ids=ids, track_ids=list(range(17))), "at most 16"),
            (dict(layer=L, neuron_idx=I, token_ids=ids, track_ids=[VOCAB + 3]), "not a word-piece id"),
            (dict(layer=L, neuron_idx=I, token_ids=ids, polarity="up"), "polarity is 'up'"),
            (dict(layer=L, neuron_idx=I, token_ids=5), "a list of word-piece ids"),
            (dict(layer=L, neuron_idx=I, token_ids=ids, track_ids=5), "a list of word-piece ids"),
            (dict(layer=L, neuron_idx=I, token_ids=ids, floor="yes"), "true or false"),
        ]
        for kw, words in cases:
            msg = refused(**kw)
            check(msg is not None and words in msg, f"refusal {kw if len(str(kw)) < 90 else list(kw)}: {msg!r}")
        check(calls["run_check"] == n_runs and calls["study_max_abs"] == n_sma, "a refusal ran a forward first")

        # running out of GPU memory comes back as a ValueError, in plain words
        def oom(*_a: Any, **_k: Any) -> Any:
            raise torch.cuda.OutOfMemoryError("simulated")

        cc.run_check = oom  # type: ignore[assignment]
        msg = refused(layer=L, neuron_idx=I, token_ids=ids)
        check(msg is not None and "ran out of memory" in msg, f"out of memory: {msg!r}")
    finally:
        cc.study_max_abs, cc.run_check = real_sma, real_run  # type: ignore[assignment]
    check(all_hooks_gone(cc.language_model(rt.hf_model)), "the tool left a hook behind")
    section("the tool: its reply, labels as inspect_pole counts them, the cache, plain-word refusals, OOM", before)
    return r


def check_transports(rt: TheodoliteRuntime, vault: LarqlVaultEngine, mcp_reply: Dict[str, Any]) -> None:
    from fastapi.testclient import TestClient

    import backend.server as server

    before = len(failures)
    server._RUNTIME, server._VAULT = rt, vault  # type: ignore[assignment]

    @asynccontextmanager
    async def no_lifespan(_app: Any):  # the real one loads the model
        yield

    server.app.router.lifespan_context = no_lifespan
    g = torch.Generator().manual_seed(22)
    ids = rand_ids(14, g)  # the ids check_tool used
    client = TestClient(server.app)
    health = client.get("/api/health").json()
    want_card = {"version": 1, "writes": ["activation", "frozen_norm"], "contexts": ["given", "study"],
                 "scale_refs": ["study", "this_text"], "k_default": 10.0, "k_max": 200.0, "k_floor": 0.01,
                 "n_study_texts": 36}
    check(health.get("causal_check") == want_card, f"/api/health causal_check: {health.get('causal_check')}")
    check(list(health)[-2:] == ["label_gate", "lens"], "/api/health: lens is not last")

    body = {"layer": L, "neuron_idx": I, "token_ids": ids, "track_ids": [ids[-1], 7]}
    res = client.post("/api/scientist/causal_check", json=body)
    data = res.json() if res.status_code == 200 else {}
    check(res.status_code == 200 and list(data) == REPLY_KEYS, f"HTTP: {res.status_code} {list(data)[:4]}")
    check(data.get("faces", {}).get("+", {}).get("rises") == mcp_reply["faces"]["+"]["rises"], "HTTP and MCP differ")
    res = client.post("/api/scientist/causal_check", json={**body, "k": 0})
    check(res.status_code == 400 and "more than 0" in res.json().get("detail", ""), f"HTTP 400: {res.status_code}")
    res = client.post("/api/scientist/causal_check", json={**body, "layer": "three"})
    check(res.status_code == 422, f"HTTP 422: {res.status_code}")

    with client.websocket_connect("/ws/theodolite") as ws:
        ws.send_json({"action": "causal_check", **body})
        msg = ws.receive_json()
        check(msg.get("type") == "causal_check" and list(msg.get("data", {})) == REPLY_KEYS, f"WS reply: {msg.get('type')}")
        check(msg.get("data", {}).get("faces", {}).get("-", {}).get("falls") == mcp_reply["faces"]["-"]["falls"],
              "WS and MCP differ")
        bad = [
            ({"layer": 99}, "layers count from 0"),
            ({"k": 0}, "more than 0"),
            ({"k": 201}, "at most 200"),
            ({"write": "ablate"}, "does not know"),
            ({"contexts": "corpus"}, "does not know"),
            ({"token_ids": [BOS, VOCAB]}, "not a word-piece id"),
            ({"token_ids": [BOS]}, "needs 2 to"),
            ({"contexts": "study"}, "reads those texts only"),
            ({"layer": "three"}, "The test was refused. layer:"),
            ({"token_ids": [BOS, 1.5]}, "The test was refused. token_ids"),
        ]
        for change, words in bad:
            sent = {**body, **change}
            ws.send_json({"action": "causal_check", **sent})
            err = ws.receive_json()
            check(err.get("type") == "error" and err.get("action") == "causal_check" and err.get("recoverable") is True
                  and err.get("layer") == sent["layer"] and err.get("neuron_idx") == sent["neuron_idx"]
                  and err.get("contexts") == sent.get("contexts") and words in err.get("message", ""),
                  f"WS refusal {change}: {err}")
        # the socket still answers, this action and another
        ws.send_json({"action": "causal_check", **body, "contexts": "study", "token_ids": None, "top_n": 2})
        msg = ws.receive_json()
        check(msg.get("type") == "causal_check" and msg["data"]["n_contexts"] == 36, f"WS after refusals: {msg.get('type')}")
        ws.send_json({"action": "inspect_pole", "layer": L, "neuron_idx": I, "polarity": 1.0})
        msg = ws.receive_json()
        check(msg.get("type") == "pole_inspection", f"WS inspect_pole after: {msg.get('type')}")
    section("the WebSocket action, the HTTP route and /api/health", before)


def _ts_table(text: str, name: str) -> Dict[int, Dict[str, float]]:
    """A `{ 10: { a: 0.1, b: 0.2 }, … }` constant of a TypeScript file, as numbers."""
    import re

    m = re.search(rf"export const {name}\b[^=]*= \{{(.*?)\n\}};", text, re.S)
    if not m:
        return {}
    rows: Dict[int, Dict[str, float]] = {}
    for layer, body in re.findall(r"(\d+): \{([^}]*)\}", m.group(1)):
        rows[int(layer)] = {k: float(v) for k, v in re.findall(r"(\w+): (-?[\d.]+)", body)}
    return rows


def check_page_figures() -> None:
    """Item 15: every figure the folio quotes beside a test is the results files' own."""
    import json
    import re

    before = len(failures)
    results = ROOT_DIR / "experiments" / "results"
    ts = (ROOT_DIR / "frontend" / "src" / "screens" / "NeuronFolio" / "causal.ts").read_text()
    diag = json.loads((results / "N1V_audit_01_postnorm.json.full.json").read_text())["result"]["diag"]
    want_lin = {
        int(layer): {
            "k10Median": round(d["k10"]["sat"]["median"], 4),
            "k10Q10": round(d["k10"]["sat"]["q10"], 4),
            "k60Median": round(d["k60"]["sat"]["median"], 4),
        }
        for layer, d in diag.items()
    }
    got_lin = _ts_table(ts, "STUDY_LINEARITY")
    check(got_lin == want_lin, f"causal.ts STUDY_LINEARITY {got_lin} != N1V_audit_01's {want_lin}")
    sampled = [10, 20, 30, 40, 45, 50, 55, 57]  # lib/labels.ts SAMPLED_IDX, the layers N1 sampled
    check(sorted(want_lin) == sampled, f"N1V_audit_01 holds layers {sorted(want_lin)}, not N1's {sampled}")
    low = min(want_lin, key=lambda l: diag[str(l)]["k10"]["sat"]["q10"])
    tail = re.search(r"export const LINEARITY_TAIL = ([\d.]+);", ts)
    tail_idx = re.search(r"export const LINEARITY_TAIL_IDX = (\d+);", ts)
    check(
        tail is not None and float(tail.group(1)) == round(diag[str(low)]["k10"]["sat"]["q10"], 3)
        and tail_idx is not None and int(tail_idx.group(1)) == low,
        f"LINEARITY_TAIL is not the lowest k = 10 q10 ({diag[str(low)]['k10']['sat']['q10']:.5f} at layer index {low})",
    )
    audit = (results / "N1V_code_audit.md").read_text()
    noise = re.search(r"export const NOISE_RATIO_MAX = ([\d.]+);", ts)
    f2 = re.search(r"kA KL median is ([\d.]+), against a bf16 chaos floor of ([\d.]+)-([\d.]+)", audit)
    check(
        noise is not None and f2 is not None and float(noise.group(1)) == round(float(f2.group(1)) / float(f2.group(2)), 2),
        "NOISE_RATIO_MAX is not N1V_code_audit.md F2's KL over the lowest floor",
    )
    small = json.loads((results / "N1V_audit_04_smallk_metrics.json").read_text())
    want_k10 = {
        int(key.split("_L")[1]): {
            "splitHalf": round(v["split_rho_median"], 4),
            "antipodal": round(v["rho_c+_c-"], 4),
            "p10Lens": round(v["jpp|p10"]["real"], 4),
            "p10Eye": round(v["naive|p10"]["real"], 4),
        }
        for key, v in small.items()
        if key.startswith("k10_L")
    }
    got_k10 = _ts_table(ts, "STUDY_K10")
    check(got_k10 == want_k10, f"causal.ts STUDY_K10 {got_k10} != N1V_audit_04's {want_k10}")
    section("the study's figures beside a test (causal.ts) are the results files' own", before)


def load_tokenizer() -> Optional[Any]:
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(DEFAULT_W4A16_MODEL_ID, local_files_only=True)
    except Exception as exc:
        failures.append(f"the instrument's tokenizer is not in the local Hugging Face cache ({exc})")
        print(f"  FAIL  {failures[-1]}")
        return None


def main() -> int:
    t0 = time.perf_counter()
    print("the folio's figures")
    check_page_figures()
    tokenizer = load_tokenizer()
    if tokenizer is not None:
        torch.set_grad_enabled(False)
        hf32 = stand_in_hf(torch.float32)
        lm = cc.language_model(hf32)
        print("the computation (float32 stand-in)")
        check_base_identity(lm, "float32")
        check_activation_write(lm)
        check_causality(lm)
        check_frozen_norm(lm)
        check_linearity(lm)
        check_small_k(lm)
        check_max_abs(lm)
        check_padded_vs_alone(lm)
        check_prefix_vs_whole(lm)
        check_oom(lm)
        check_movers_and_p_at()
        check_base_identity(cc.language_model(stand_in_hf(torch.bfloat16)), "bfloat16")
        check_base_identity(cc.language_model(stand_in_hf(torch.float32, "eager")), "float32, eager attention")
        rt = stand_in_runtime(hf32, tokenizer)
        vault = LarqlVaultEngine(rt)
        ms.bind_runtime(rt, vault)
        print("the study's texts")
        check_study_texts(rt)
        print("the tool")
        reply = check_tool(rt, vault)
        print("the transports")
        check_transports(rt, vault, reply)
    took = time.perf_counter() - t0
    if failures:
        print(f"\ncausal check: {len(failures)} of {passed + len(failures)} checks FAILED ({took:.0f} s)")
        return 1
    print(f"\ncausal check: all {passed} checks hold ({took:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

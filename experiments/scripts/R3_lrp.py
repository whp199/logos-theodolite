#!/usr/bin/env python3
"""R3 LRP (CPU-testable; the fit uses it on the GPU): J++'s R-lens rules for Gemma 4, as a context manager that patches
module instances while the fit's forward pass records its graph.

The rules are those of jpp_lens at commit d7efc0de (src/workspace_lens/lrp/lrp.py, Apache-2.0, adapted there from
the R-Lens implementation of Camila Blank and Agam Bhatia; see NOTICE), preset "rlens" (lrp.py:209-218):
- LN rule (lrp.py:305-316): every Gemma4RMSNorm outside self_attn computes x * pow(mean(x^2) + eps, -0.5) with that
  factor detached, then the plain weight when the norm has one. The final norm is patched too, as J++ patches it; it is
  off the path of a block-59 target, so it changes no fitted row (--selftest shows this).
- identity rule (lrp.py:343-350): the MLP's GELU-tanh is written z * gate(z) with gate(z) = 0.5 (1 + tanh(...))
  detached. Deviation D33: the forward is computed in fp32 and rounded once to the model's dtype, as F.gelu's fused
  kernel computes it, where jpp_lens takes the product in the model's dtype; the backward rule (the gate held fixed,
  the gradient through z only) is J++'s, and in fp32 nothing changes;
- half rule (lrp.py:363-366, 394-399): the gated product p becomes 0.5 p + 0.5 p.detach().
Attention, its q/k/v norms and a multimodal wrapper's vision and audio towers keep standard gradients. Detaching
changes no forward value, only gradients; the graph recorded inside the context keeps the detached ops after exit.

Only the classes Gemma 4 uses are patched, and the gemma4_unified classes of the 12B dry-run checkpoint, only while
their forward (and _norm) is Gemma 4's code to the letter (EQUIVALENT_CLASSES). Any other RMSNorm or gated-MLP class in the text decoder, a GELU that is not
the tanh form, a torch.compile'd module or gradient checkpointing raises, so a partly patched fit cannot happen
silently. The fit uses this reimplementation; R3_parity.py and --selftest check it against jpp_lens's own patcher.

Self-test (CPU, a tiny Gemma 4 that keeps the real config's flags; and the real 31B module tree built on the meta
device, no weights):
  CUDA_VISIBLE_DEVICES="" /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R3_lrp.py --selftest
"""
from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.dont_write_bytecode = True

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import Any, Dict, Iterator, List, Tuple

import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

NORM_CLASSES = ("Gemma4RMSNorm",)
MLP_CLASSES = ("Gemma4TextMLP",)
GELU_TANH_CLASSES = ("GELUTanh",)                 # transformers' ACT2FN["gelu_pytorch_tanh"]
TOWER_SEGMENTS = frozenset({"vision_tower", "audio_tower", "embed_vision", "embed_audio"})
# Classes of other Gemma 4 checkpoints (the local Gemma 4 12B of R3_00's dry run loads as gemma4_unified) that are
# patched only if their code is the code of the class they stand for: forward and _norm (or forward alone) must have
# the same source text, read with inspect, apart from the class name. Anything else still raises.
EQUIVALENT_CLASSES = {"Gemma4UnifiedRMSNorm": "Gemma4RMSNorm", "Gemma4UnifiedTextMLP": "Gemma4TextMLP"}
_MISSING = object()
_EQUIVALENCE_SEEN: Dict[type, bool] = {}


def _same_code(cls: type, name: str) -> bool:
    """Whether ``cls`` (one of EQUIVALENT_CLASSES) has exactly the methods of transformers' Gemma 4 class ``name``."""
    if cls in _EQUIVALENCE_SEEN:
        return _EQUIVALENCE_SEEN[cls]
    import inspect
    import textwrap

    from transformers.models.gemma4 import modeling_gemma4

    ref = getattr(modeling_gemma4, name)
    ok = True
    for meth in ("forward", "_norm"):
        a, b = getattr(cls, meth, None), getattr(ref, meth, None)
        if (a is None) != (b is None):
            ok = False
        elif a is not None:
            ok &= textwrap.dedent(inspect.getsource(a)) == textwrap.dedent(inspect.getsource(b))
    _EQUIVALENCE_SEEN[cls] = ok
    return ok


def _known(module: nn.Module, allowed: Tuple[str, ...]) -> bool:
    cls = type(module)
    if cls.__name__ in allowed:
        return True
    ref = EQUIVALENT_CLASSES.get(cls.__name__)
    return ref in allowed and _same_code(cls, ref)


@dataclass
class Report:
    """Qualified names of what was patched and what was left alone on purpose."""

    ln_rule: List[str] = field(default_factory=list)
    mlp_rule: List[str] = field(default_factory=list)
    attention_norms_kept: List[str] = field(default_factory=list)
    tower_modules_skipped: int = 0

    def summary(self) -> Dict[str, Any]:
        return {"ln_rule": len(self.ln_rule), "mlp_rule": len(self.mlp_rule),
                "attention_norms_kept": len(self.attention_norms_kept),
                "tower_modules_skipped": self.tower_modules_skipped,
                "final_norm_patched": any(n.split(".")[-1] == "norm" for n in self.ln_rule)}


# ------------------------------------------------------------------------------------------------ the rules
def ln_rule_forward(self: Any, hidden_states: Tensor) -> Tensor:
    """Gemma4RMSNorm.forward with the normalising factor detached (value-exact: the same ops in the same order)."""
    x_f = hidden_states.float()
    scale = torch.pow(x_f.pow(2).mean(-1, keepdim=True) + self.eps, -0.5).detach()
    normed = x_f * scale
    if self.with_scale:
        normed = normed * self.weight.float()
    return normed.type_as(hidden_states)


def gelu_tanh_gate(z: Tensor) -> Tensor:
    """The gate of GELU-tanh: gelu(z) = z * gate(z)."""
    return 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (z + 0.044715 * z.pow(3))))


def identity_rule_gelu(z: Tensor) -> Tensor:
    """GELU-tanh with its gate held fixed (the identity rule), its forward computed as F.gelu's fused kernel computes
    it (D33): in fp32, rounded once to z's dtype. The gate is computed from z in fp32 and detached, the product taken
    in fp32, and the result cast back once, so the gradient reaching z is the incoming gradient times the detached
    gate (in fp32, rounded once to z's dtype), as in J++'s rule. In fp32 this is J++'s z * gate(z).detach() op for op
    (.float() and .to() return z and the product themselves); in bf16 it removes the one-ulp differences of J++'s
    bf16 product from the kernel (R3_00_pilot.json at 8996fb5: 90 of 2,048 top-1 changes under the rules)."""
    z32 = z.float()
    return (z32 * gelu_tanh_gate(z32).detach()).to(z.dtype)


def identity_rule_gelu_in_dtype(z: Tensor) -> Tensor:
    """jpp_lens's arithmetic for the identity rule (lrp.py:343-350): z * gate(z).detach() in z's own dtype. Not used by
    the rules; kept so that the self-test shows the registered code's figure beside the rule's."""
    return z * gelu_tanh_gate(z).detach()


def half_rule(p: Tensor) -> Tensor:
    """The value p, with half of its gradient (0.5 p + 0.5 p.detach(): exact in floating point)."""
    return 0.5 * p + 0.5 * p.detach()


def mlp_rule_forward(self: Any, x: Tensor) -> Tensor:
    """Gemma4TextMLP.forward, down(act(gate(x)) * up(x)), under the identity and half rules."""
    return self.down_proj(half_rule(identity_rule_gelu(self.gate_proj(x)) * self.up_proj(x)))


# ------------------------------------------------------------------------------------------------ the walk
def _is_gated_mlp(module: nn.Module) -> bool:
    return all(hasattr(module, a) for a in ("gate_proj", "up_proj", "down_proj", "act_fn"))


def _check_patchable(model: nn.Module) -> None:
    if getattr(model, "is_gradient_checkpointing", False):
        raise RuntimeError("gradient checkpointing re-runs blocks outside the LRP context; disable it")
    compiled = getattr(getattr(torch, "_dynamo", None), "eval_frame", None)
    compiled_cls = getattr(compiled, "OptimizedModule", None) if compiled is not None else None
    if compiled_cls is not None:
        for name, module in model.named_modules():
            if isinstance(module, compiled_cls):
                raise RuntimeError(f"{name or '<root>'} is torch.compile'd; instance patches would be bypassed")


def plan(model: nn.Module) -> Report:
    """What rlens() would patch, without patching (raises on any module it cannot patch exactly)."""
    _check_patchable(model)
    rep = Report()
    for name, module in model.named_modules():
        segs = name.split(".")
        if not TOWER_SEGMENTS.isdisjoint(segs):
            rep.tower_modules_skipped += 1
            continue
        cls = type(module).__name__
        if cls.endswith("RMSNorm"):
            if "self_attn" in segs:
                rep.attention_norms_kept.append(name)
                continue
            if not _known(module, NORM_CLASSES):
                raise RuntimeError(f"no LN-rule forward for {cls} at {name!r}")
            rep.ln_rule.append(name)
        elif _is_gated_mlp(module):
            if not _known(module, MLP_CLASSES):
                raise RuntimeError(f"unknown gated MLP {cls} at {name!r}")
            act = type(module.act_fn).__name__
            if act not in GELU_TANH_CLASSES:
                raise RuntimeError(f"identity rule expects GELU-tanh, found {act} at {name!r}")
            rep.mlp_rule.append(name)
    if not rep.ln_rule or not rep.mlp_rule:
        raise RuntimeError(f"the R-lens rules matched nothing on {type(model).__name__} "
                           f"({len(rep.ln_rule)} norms, {len(rep.mlp_rule)} MLPs)")
    return rep


@contextmanager
def rlens(model: nn.Module, *, final_norm: bool = True) -> Iterator[Report]:
    """Patch the R-lens rules into ``model``'s module instances for the duration of the block; restore on exit.
    ``final_norm=False`` leaves the decoder's final norm stock (used only to show that it changes no fitted row)."""
    rep = plan(model)
    if not final_norm:
        rep.ln_rule = [n for n in rep.ln_rule if n.split(".")[-1] != "norm"]
    saved = []
    try:
        for names, fwd in ((rep.ln_rule, ln_rule_forward), (rep.mlp_rule, mlp_rule_forward)):
            for name in names:
                module = model.get_submodule(name)
                # A module under an accelerate hook (the dry run's offloaded blocks, A13) keeps its weights on the meta
                # device until the hook's wrapper in `forward` loads them: the rule then replaces `_old_forward`, the
                # method that wrapper calls, so that the hook still runs around it.
                attr = "_old_forward" if hasattr(module, "_hf_hook") and "_old_forward" in module.__dict__ else "forward"
                saved.append((module, attr, module.__dict__.get(attr, _MISSING)))
                object.__setattr__(module, attr, MethodType(fwd, module))
        yield rep
    finally:
        for module, attr, original in reversed(saved):
            if original is _MISSING:
                module.__dict__.pop(attr, None)
            else:
                object.__setattr__(module, attr, original)


# ------------------------------------------------------------------------------------------------ self-test
def _block_rows(model, ids: Tensor, src: int, tgt: int, cot: Tensor, ctx=None) -> Tensor:
    """d<h_tgt, cot>/d h_src for the decoder blocks of a tiny Gemma4ForCausalLM (h = block outputs), under ``ctx``."""
    from contextlib import nullcontext

    outs = {}

    def mk(i):
        def hook(mod, args, out):
            h = out if torch.is_tensor(out) else out[0]
            if i == src:
                h.requires_grad_(True)
            outs[i] = h
        return hook

    layers = model.model.layers
    hs = [layers[i].register_forward_hook(mk(i)) for i in (src, tgt)]
    try:
        with torch.enable_grad(), (ctx if ctx is not None else nullcontext()):
            model.model(input_ids=ids, use_cache=False)
        return torch.autograd.grad(outs[tgt], outs[src], grad_outputs=cot)[0]
    finally:
        for h in hs:
            h.remove()


def selftest(record: bool = False) -> int:
    import R3_common as C

    fails: List[str] = []
    done: List[Tuple[bool, str]] = []
    t_start = time.time()
    notes: List[str] = []

    def check(ok: bool, what: str) -> None:
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        done.append((bool(ok), what))
        if not ok:
            fails.append(what)

    torch.manual_seed(0)
    model = C.build_tiny_gemma4()
    cfg = model.config
    check(cfg.hidden_activation == "gelu_pytorch_tanh" and cfg.hidden_size_per_layer_input == 0
          and cfg.attention_k_eq_v is True and cfg.final_logit_softcapping == 30.0,
          "the tiny Gemma 4 keeps the real flags (gelu_pytorch_tanh, hidden_size_per_layer_input 0, "
          "attention_k_eq_v, softcap 30)")
    L = cfg.num_hidden_layers
    ids = torch.randint(1, 30, (2, 40))
    with torch.no_grad():
        stock = model(ids).logits
        with rlens(model) as rep:
            patched = model(ids).logits
        restored = model(ids).logits
    s = rep.summary()
    check(float((patched - stock).abs().max()) <= 1e-6,
          f"forward unchanged under the rules (max |diff| {float((patched - stock).abs().max()):.2e} <= 1e-6, fp32)")
    check(torch.equal(restored, stock), "every forward restored exactly on exit")
    check(s["ln_rule"] == 4 * L + 1 and s["mlp_rule"] == L and s["attention_norms_kept"] == 3 * L
          and s["final_norm_patched"],
          f"coverage: {s['ln_rule']} norms (4 per block + the final norm), {s['mlp_rule']} MLPs, "
          f"{s['attention_norms_kept']} q/k/v norms kept stock")

    # LN rule: gradient of <norm(x), g> is the frozen-scale gradient s * w * g
    norm = model.model.layers[0].input_layernorm
    x = torch.randn(3, 5, cfg.hidden_size, requires_grad=True)
    g = torch.randn(3, 5, cfg.hidden_size)
    with rlens(model):
        (gx,) = torch.autograd.grad((norm(x) * g).sum(), x)
    sc = torch.pow(x.detach().pow(2).mean(-1, keepdim=True) + norm.eps, -0.5)
    check(torch.allclose(gx, sc * norm.weight * g, atol=1e-6, rtol=0),
          "LN rule: the gradient is the frozen-scale gradient pow(mean(x^2)+eps, -0.5) * w * g")
    (gx_stock,) = torch.autograd.grad((norm(x) * g).sum(), x)
    check(float((gx_stock - gx).abs().max()) > 1e-4, "LN rule: differs from the stock gradient (the variance term)")

    # identity rule: value is GELU-tanh, gradient is the gate
    z = torch.randn(1000, dtype=torch.float32, requires_grad=True)
    v = identity_rule_gelu(z)
    (gz,) = torch.autograd.grad(v.sum(), z)
    check(float((v - nn.functional.gelu(z, approximate="tanh")).detach().abs().max()) <= 1e-6
          and torch.allclose(gz, gelu_tanh_gate(z.detach()), atol=0, rtol=0),
          "identity rule: value equals GELU-tanh (<= 1e-6) and the gradient equals the gate exactly")

    # identity rule in the model's dtype (D33): the forward in fp32, rounded once, against the fused kernel, with the
    # registered code's arithmetic (jpp_lens's bf16 product) beside it
    gelu = lambda t: nn.functional.gelu(t, approximate="tanh")  # noqa: E731
    allb = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    allb = allb[torch.isfinite(allb)]                  # every finite bf16 value (65,280)
    check(torch.equal(gelu(allb.float()).to(torch.bfloat16), gelu(allb)),
          f"the fused bf16 kernel is fp32 rounded once: F.gelu in fp32, cast to bf16, equals F.gelu in bf16 on all "
          f"{allb.numel():,} finite bf16 inputs")

    def vs_kernel(f, zz):
        ref, out = gelu(zz), f(zz)
        diff = out != ref
        return {"n": zz.numel(), "differ": int(diff.sum()), "max": float((out.float() - ref.float()).abs().max()),
                "max_abs_where_differ": float(torch.maximum(out.float().abs(), ref.float().abs())[diff].max())
                if bool(diff.any()) else 0.0}
    new_all, old_all = vs_kernel(identity_rule_gelu, allb), vs_kernel(identity_rule_gelu_in_dtype, allb)
    check(new_all["differ"] <= 1e-4 * new_all["n"] and new_all["max_abs_where_differ"] < 1e-6,
          f"bf16, every finite input: the rule's forward equals F.gelu at {new_all['n'] - new_all['differ']:,} of "
          f"{new_all['n']:,} (largest |diff| {new_all['max']:.2e}; where they differ both values are below 1e-6 "
          f"in magnitude, max {new_all['max_abs_where_differ']:.2e}); jpp_lens's bf16 product: "
          f"{old_all['n'] - old_all['differ']:,} of {old_all['n']:,} (largest |diff| {old_all['max']:.2e})")
    gen = torch.Generator().manual_seed(20261022)
    for scale in (0.3, 1.0, 3.0, 10.0):
        zr = (torch.randn(1_000_000, generator=gen) * scale).to(torch.bfloat16)
        nr, orr = vs_kernel(identity_rule_gelu, zr), vs_kernel(identity_rule_gelu_in_dtype, zr)
        check(nr["differ"] <= 0.01 * nr["n"] and nr["max_abs_where_differ"] < 1e-6,
              f"bf16, 1,000,000 inputs N(0, {scale:g}^2): the rule's forward differs from F.gelu at {nr['differ']:,} "
              f"({nr['differ'] / nr['n']:.4%}, largest |diff| {nr['max']:.2e}, every one where both values are below "
              f"1e-6); jpp_lens's bf16 product at {orr['differ']:,} ({orr['differ'] / orr['n']:.2%}, largest |diff| "
              f"{orr['max']:.2e})")
    notes.append("where the rule's bf16 forward still differs from F.gelu (on this CPU only at z = -5.0625, -5.09375, "
                 "-5.125 and -5.15625), 1 + tanh cancels to about one fp32 ulp: the GELU kernel's own tanh rounds to "
                 "-1 there and gives -0.0, torch.tanh does not and the rule gives about -1.5e-7 (the exact value is "
                 "-0.8e-7 to -1.5e-7)")

    # (ii) the gradient is the detached gate times the incoming gradient, in fp32 and rounded once to z's dtype
    zb = (torch.randn(200_000, generator=gen) * 3.0).to(torch.bfloat16).requires_grad_(True)
    gb = torch.randn(200_000, generator=gen).to(torch.bfloat16)
    (gzb,) = torch.autograd.grad(identity_rule_gelu(zb), zb, grad_outputs=gb)
    zd = zb.detach()
    want_b = (gb.float() * gelu_tanh_gate(zd.float())).to(torch.bfloat16)
    zj = zd.clone().requires_grad_(True)
    (gzj,) = torch.autograd.grad(identity_rule_gelu_in_dtype(zj), zj, grad_outputs=gb)
    check(gzb.dtype == torch.bfloat16 and torch.equal(gzb, want_b)
          and torch.equal(gzj, gb * gelu_tanh_gate(zd)),
          f"identity rule, bf16: the gradient is exactly the detached gate times the incoming gradient (the gate and "
          f"product in fp32, one rounding to bf16; jpp_lens's is the same rule with gate and product in bf16, and "
          f"the two differ by one bf16 rounding at {int((gzb != gzj).sum()):,} of {zb.numel():,})")

    # (iii) in fp32 nothing changes: the same value and the same gradient, bit for bit
    z3 = torch.randn(100_000, generator=gen) * 3.0
    g3 = torch.randn(100_000, generator=gen)
    za, zc = z3.clone().requires_grad_(True), z3.clone().requires_grad_(True)
    va, vc = identity_rule_gelu(za), identity_rule_gelu_in_dtype(zc)
    (ga,) = torch.autograd.grad(va, za, grad_outputs=g3)
    (gc,) = torch.autograd.grad(vc, zc, grad_outputs=g3)
    check(torch.equal(va, vc) and torch.equal(ga, gc) and torch.equal(ga, g3 * gelu_tanh_gate(z3)),
          "identity rule, fp32: value and gradient bit-identical to jpp_lens's arithmetic (z * gate(z).detach()), "
          "the gradient exactly g * gate(z)")

    # half rule: exact value, half the gradient
    p = torch.randn(1000, requires_grad=True)
    h = half_rule(p)
    (gp,) = torch.autograd.grad(h.sum(), p)
    check(torch.equal(h.detach(), p.detach()) and torch.equal(gp, torch.full_like(p, 0.5)),
          "half rule: value bit-exact, gradient exactly 0.5")

    # MLP under both rules: the analytic gradient
    mlp = model.model.layers[1].mlp
    x = torch.randn(4, cfg.hidden_size, requires_grad=True)
    g = torch.randn(4, cfg.hidden_size)
    with rlens(model):
        (gx,) = torch.autograd.grad((mlp(x) * g).sum(), x)
    xd = x.detach()
    zg, u = mlp.gate_proj(xd), mlp.up_proj(xd)
    a = g @ mlp.down_proj.weight                             # d/dp of <down(p), g>
    want = (0.5 * a * u * gelu_tanh_gate(zg)) @ mlp.gate_proj.weight + (0.5 * a * zg * gelu_tanh_gate(zg)) @ mlp.up_proj.weight
    check(torch.allclose(gx, want, atol=1e-5, rtol=0),
          "MLP: gradient = W_gate^T(0.5 a u gate(z)) + W_up^T(0.5 a z gate(z)) (identity and half rules)")

    # the final norm is off the path of a block-output target: the rows are bit-identical either way
    cot = torch.randn(2, 40, cfg.hidden_size)
    r_with = _block_rows(model, ids, 1, L - 1, cot, rlens(model, final_norm=True))
    r_without = _block_rows(model, ids, 1, L - 1, cot, rlens(model, final_norm=False))
    r_stock = _block_rows(model, ids, 1, L - 1, cot)
    check(torch.equal(r_with, r_without), "rows to the last block's output are bit-identical with and without the "
                                          "final norm patched (it is off the path)")
    check(float((r_with - r_stock).abs().max()) > 1e-4, "the rules change the rows (R-lens rows differ from plain "
                                                        "gradients)")

    # A13: blocks offloaded by accelerate (as the dry run's device map puts the 12B's lowest blocks "on the CPU":
    # weights on the meta device between calls, loaded by a hook in `forward`) get the rules inside the hook
    import copy

    import accelerate

    off = copy.deepcopy(model)
    for i in range(3):                                   # block 2 lies on the path of the rows from block 1
        accelerate.cpu_offload(off.model.layers[i], execution_device=torch.device("cpu"))
    r_off = _block_rows(off, ids, 1, L - 1, cot, rlens(off))
    with torch.no_grad():
        o_stock = off(ids).logits
    check(off.model.layers[0].mlp.gate_proj.weight.device.type == "meta" and torch.equal(r_off, r_with)
          and torch.equal(o_stock, stock),
          "on blocks offloaded by accelerate (weights on the meta device between calls), the rules run inside the "
          "hook: the rows are bit-identical to the unoffloaded model's, and the hooked forward is restored on exit")
    del off

    # unknown classes raise
    class FooRMSNorm(nn.Module):
        def forward(self, x):
            return x

    class OtherMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = self.up_proj = self.down_proj = nn.Identity()
            self.act_fn = nn.SiLU()

    for what, patch in (("an unknown RMSNorm class", lambda m: setattr(m.model.layers[0], "extra_norm", FooRMSNorm())),
                        ("an unknown gated MLP class", lambda m: setattr(m.model.layers[0], "mlp", OtherMLP())),
                        ("a GELU that is not the tanh form",
                         lambda m: setattr(m.model.layers[0].mlp, "act_fn", nn.GELU()))):
        m2 = C.build_tiny_gemma4()
        patch(m2)
        try:
            with rlens(m2):
                pass
            check(False, f"{what} raises")
        except RuntimeError:
            check(True, f"{what} raises")

    # the multimodal wrapper the checkpoint loads as: towers skipped
    mm = C.build_tiny_gemma4(multimodal=True)
    with torch.no_grad():
        ids1 = torch.randint(1, 30, (1, 24))
        a0 = mm.model.language_model(input_ids=ids1).last_hidden_state
        with rlens(mm) as rep_mm:
            a1 = mm.model.language_model(input_ids=ids1).last_hidden_state
    sm = rep_mm.summary()
    check(sm["ln_rule"] == 4 * L + 1 and sm["mlp_rule"] == L and sm["tower_modules_skipped"] > 0
          and not any("tower" in n or "embed_" in n for n in rep_mm.ln_rule + rep_mm.mlp_rule)
          and float((a1 - a0).abs().max()) <= 1e-5,
          f"multimodal wrapper: {sm['ln_rule']} norms and {sm['mlp_rule']} MLPs in the text decoder, "
          f"{sm['tower_modules_skipped']} tower modules skipped, forward unchanged after the final norm "
          f"(max |diff| {float((a1 - a0).abs().max()):.1e} <= 1e-5: the GELU rewrite's rounding, scaled by the norm)")

    # the real 31B module tree, built on the meta device (no weights)
    from transformers import AutoConfig
    from transformers.models.gemma4 import Gemma4ForConditionalGeneration

    real_cfg = AutoConfig.from_pretrained(str(C.MODEL_DIR), local_files_only=True)
    with torch.device("meta"):
        big = Gemma4ForConditionalGeneration(real_cfg)
    rb = plan(big).summary()
    n_real = real_cfg.get_text_config().num_hidden_layers
    check(rb["ln_rule"] == 4 * n_real + 1 and rb["mlp_rule"] == n_real and rb["final_norm_patched"],
          f"the real 31B module tree (meta device): {rb['ln_rule']} norms, {rb['mlp_rule']} MLPs, "
          f"{rb['attention_norms_kept']} attention norms kept, {rb['tower_modules_skipped']} tower modules skipped")
    del big

    # the local Gemma 4 12B of R3_00's dry run (gemma4_unified): its norm and MLP classes are accepted only because
    # their code is Gemma 4's, and a class whose code differs still raises
    dry_cfg_path = C.DRY_RUN_MODEL_DIR / "config.json"
    if dry_cfg_path.is_file():
        import transformers
        from transformers.models.gemma4_unified import modeling_gemma4_unified as MU

        check(_same_code(MU.Gemma4UnifiedRMSNorm, "Gemma4RMSNorm") and _same_code(MU.Gemma4UnifiedTextMLP,
                                                                                    "Gemma4TextMLP"),
              "the 12B's Gemma4UnifiedRMSNorm and Gemma4UnifiedTextMLP have exactly Gemma 4's forward (and _norm)")
        dcfg = AutoConfig.from_pretrained(str(C.DRY_RUN_MODEL_DIR), local_files_only=True)
        with torch.device("meta"):
            small = getattr(transformers, dcfg.architectures[0])(dcfg)
        rd = plan(small).summary()
        n_dry = dcfg.get_text_config().num_hidden_layers
        check(rd["ln_rule"] == 4 * n_dry + 1 and rd["mlp_rule"] == n_dry and rd["final_norm_patched"],
              f"the 12B module tree (meta device, {dcfg.architectures[0]}): {rd['ln_rule']} norms, {rd['mlp_rule']} "
              f"MLPs, {rd['attention_norms_kept']} attention norms kept, {rd['tower_modules_skipped']} tower modules "
              f"skipped")
        del small

        class Gemma4UnifiedRMSNormAltered(MU.Gemma4UnifiedRMSNorm):
            def forward(self, hidden_states):
                return hidden_states
        Gemma4UnifiedRMSNormAltered.__name__ = "Gemma4UnifiedRMSNorm"
        m3 = C.build_tiny_gemma4()
        m3.model.layers[0].extra_norm = Gemma4UnifiedRMSNormAltered(cfg.hidden_size)
        try:
            plan(m3)
            check(False, "a gemma4_unified-named norm whose code differs from Gemma 4's raises")
        except RuntimeError:
            check(True, "a gemma4_unified-named norm whose code differs from Gemma 4's raises")
    else:
        check(False, f"the dry run's 12B config is at {dry_cfg_path}")

    # against jpp_lens's own patcher: the same modules, and the same rows bit for bit
    try:
        C.import_jpp()
        from workspace_lens.lrp.lrp import apply_lrp_rules_for_mode
    except Exception as e:  # noqa: BLE001
        check(False, f"jpp_lens's patcher imports from the pinned checkout ({type(e).__name__}: {e})")
    else:
        with apply_lrp_rules_for_mode(model, "rlens") as jrep:
            jn = jrep.patched_names()
        check(sorted(jn["ln_rule"]) == sorted(rep.ln_rule) and sorted(jn["mlp_rule"]) == sorted(rep.mlp_rule)
              and not any(jn[k] for k in ("moe_experts", "moe_gates", "mhc", "shared_expert_forward_scale")),
              "jpp_lens's patcher (rlens) patches exactly the same norms and MLPs")
        r_jpp = _block_rows(model, ids, 1, L - 1, cot, apply_lrp_rules_for_mode(model, "rlens"))
        d = float((r_jpp - r_with).abs().max())
        check(d <= 1e-6, f"rows under R3's rules equal rows under jpp_lens's (max |diff| {d:.2e}; "
                         f"{'bit-identical' if d == 0 else 'within 1e-6'})")
        with torch.no_grad(), apply_lrp_rules_for_mode(mm, "rlens") as jrm:
            pass
        jm = jrm.patched_names()
        check(sorted(jm["ln_rule"]) == sorted(rep_mm.ln_rule) and sorted(jm["mlp_rule"]) == sorted(rep_mm.mlp_rule),
              "on the multimodal wrapper too, jpp_lens's patcher patches the same modules")

        # D33 against jpp_lens's own code: its identity rule is the bf16 product shown above, and in bf16 the tiny
        # model's forward under R3's rules equals the stock forward where under jpp_lens's it does not
        from workspace_lens.lrp.lrp import _gelu_tanh_with_detached_gate as jpp_gelu

        check(torch.equal(jpp_gelu(allb), identity_rule_gelu_in_dtype(allb))
              and torch.equal(jpp_gelu(allb.float()), identity_rule_gelu(allb.float())),
              "jpp_lens's _gelu_tanh_with_detached_gate is bit for bit the bf16 product above on every finite bf16 "
              "input, and R3's rule in fp32")
        mb = C.build_tiny_gemma4().to(torch.bfloat16)
        idb = torch.randint(1, 30, (4, 40))
        with torch.no_grad():
            lb_stock = mb(idb).logits
            with rlens(mb):
                lb_r3 = mb(idb).logits
            with apply_lrp_rules_for_mode(mb, "rlens"):
                lb_jpp = mb(idb).logits
        t_stock, t_r3, t_jpp = (x.argmax(-1) for x in (lb_stock, lb_r3, lb_jpp))
        d_r3, d_jpp = float((lb_r3 - lb_stock).abs().max()), float((lb_jpp - lb_stock).abs().max())
        check(d_r3 <= d_jpp and int((t_r3 != t_stock).sum()) <= int((t_jpp != t_stock).sum()),
              f"bf16 tiny Gemma 4, forward under the rules against the stock forward ({idb.numel()} positions): R3's "
              f"rules max |logit diff| {d_r3:.2e} ({'bit-identical' if d_r3 == 0 else 'not bit-identical'}), "
              f"top-1 changed at {int((t_r3 != t_stock).sum())}; jpp_lens's patcher {d_jpp:.2e}, top-1 changed at "
              f"{int((t_jpp != t_stock).sum())}")
        del mb
    for n in notes:
        print("note:", n)
    print(f"R3_lrp self-test: {'ALL PASS' if not fails else str(len(fails)) + ' FAILURES'}")
    if record:
        C.record_selftest(Path(__file__).name, done, time.time() - t_start)
    return 0 if not fails else 1


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--record", action="store_true", help="with --selftest: keep the result in R3_selftests.json")
    a = ap.parse_args()
    if not a.selftest:
        ap.print_help()
        sys.exit(2)
    sys.exit(selftest(record=a.record))

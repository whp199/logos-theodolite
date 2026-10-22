#!/usr/bin/env python3
"""R3 LRP (CPU-testable; the fit uses it on the GPU): J++'s R-lens rules for Gemma 4, as a context manager that patches
module instances while the fit's forward pass records its graph.

The rules are those of jpp_lens at commit d7efc0de (src/workspace_lens/lrp/lrp.py, Apache-2.0, adapted there from
the R-Lens implementation of Camila Blank and Agam Bhatia; see NOTICE), preset "rlens" (lrp.py:209-218):
- LN rule (lrp.py:305-316): every Gemma4RMSNorm outside self_attn computes x * pow(mean(x^2) + eps, -0.5) with that
  factor detached, then the plain weight when the norm has one. The final norm is patched too, as J++ patches it; it is
  off the path of a block-59 target, so it changes no fitted row (--selftest shows this).
- identity rule (lrp.py:343-350): the MLP's GELU-tanh is written z * gate(z) with gate(z) = 0.5 (1 + tanh(...))
  detached;
- half rule (lrp.py:363-366, 394-399): the gated product p becomes 0.5 p + 0.5 p.detach().
Attention, its q/k/v norms and a multimodal wrapper's vision and audio towers keep standard gradients. Detaching
changes no forward value, only gradients; the graph recorded inside the context keeps the detached ops after exit.

Only the classes Gemma 4 uses are patched. Any other RMSNorm or gated-MLP class in the text decoder, a GELU that is not
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
_MISSING = object()


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
    """GELU-tanh with its gate held fixed (equal to F.gelu(z, approximate="tanh") up to kernel rounding)."""
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
            if cls not in NORM_CLASSES:
                raise RuntimeError(f"no LN-rule forward for {cls} at {name!r}")
            rep.ln_rule.append(name)
        elif _is_gated_mlp(module):
            if cls not in MLP_CLASSES:
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
                saved.append((module, module.__dict__.get("forward", _MISSING)))
                object.__setattr__(module, "forward", MethodType(fwd, module))
        yield rep
    finally:
        for module, original in reversed(saved):
            if original is _MISSING:
                module.__dict__.pop("forward", None)
            else:
                object.__setattr__(module, "forward", original)


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

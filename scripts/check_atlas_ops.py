#!/usr/bin/env python3
"""A CPU-only check of the atlas's live operations (backend/atlas_ops.py) on a tiny random Gemma 4
(scripts/atlas_stand_in.py: transformers' Gemma4ForCausalLM with sliding layers, a full last layer whose values are its
keys, partial rotary, layer scalars and norms with spread, and its own byte-level tokenizer). It loads neither the real
model nor the lens, and hides the GPU. It checks, against brute force:
1. each push lands where and as it says: "add" adds dose * p50 * unit(w_post (.) W_down[:, n]) to the stream leaving its
   layer at every word-piece but the start mark, and nothing before that layer changes; "mul" multiplies the neuron's
   activation (and nothing else at that layer's down_proj input); "act" adds to it; a whole layer's block at x0 writes
   nothing; a head scaled to 0 gives o_proj nothing from that head; the start mark is spared unless told otherwise;
2. `resolve` gives the three voices their rows: +dose, 0, -dose for "add" and "act", x dose, x1, x -dose for "mul";
3. writing three rows at once, each under its own push and left-padded to the longest, writes what each row writes
   alone, and an unpushed row writes what the model writes unpushed; `TextDeltas` adds up to the decoded text, with
   characters split over pieces held back until whole;
4. `read_neuron` is the activation a plain hook sees; a push on an earlier layer moves it, a push on the neuron itself
   at its own layer does not (the read comes first); `next_logprobs`' two rows equal two separate forwards;
5. `all_activations` and `activation_extremes` equal per-text brute force, leaving out the start mark, and the four
   sign regimes split the two sides exactly;
6. `token_units` covers every piece once, in order, joins to the decoded text, and leaves no unit cut inside a
   character;
7. `capture_heads` and `head_pattern` give the attention weights transformers' eager attention returns, for every head
   of every layer (sliding and full, past the window); each head's output is its weights times the values; the heads'
   writes add up to the post-attention norm's output;
8. `attention_drive`: the activation it linearises is the one the model computes, and its parts (every head, every
   feed-forward block, the embedding, each carried through the layer scalars) add up to the linearised total; each top
   head's split over source pieces adds up to its contribution;
9. the norms' scales are what the post- and pre-feed-forward norms divide by; a writer's unit of activation reaches a
   block several layers on carried by exactly the layer scalars between (measured with only the direct path open);
   `neuron_wiring` equals the weight products computed one by one, with the norms where Gemma 4 has them, z, its
   strength across layers |P| f^2 over the strongest (w, the order of the links), sign(P) and the gate sign(g); a
   neuron that writes nothing has no readers; and a link
   reads the same product from both ends; `HeadSpaces`' shares equal projections on explicit orthonormal bases, and
   `neuron_head_roles` and `head_wiring` rank as those projections do;
10. `steer_lens`' base rows are the unpushed forwards and its rises are those of separate pushed forwards.

Run: .venv/bin/python -I scripts/check_atlas_ops.py   (exits 0 when every check holds)
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"

from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
import torch.nn.functional as F

import backend.atlas_ops as ao
from atlas_stand_in import stand_in_hf, stand_in_runtime, stand_in_tokenizer
from backend.causal_check import language_model, logits_fp32

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


class Seen:
    """Every call's input (pre) or output (post) of a module, registered after the interventions' hooks."""

    def __init__(self, module: Any, kind: str) -> None:
        self.calls: List[torch.Tensor] = []
        if kind == "pre":
            self.h = module.register_forward_pre_hook(lambda _m, a: self.calls.append(a[0].detach().clone()))
        else:
            self.h = module.register_forward_hook(
                lambda _m, _a, o: self.calls.append((o[0] if isinstance(o, tuple) else o).detach().clone())
            )

    def __enter__(self) -> "Seen":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.h.remove()


def ids_of(tok: Any, text: str, chat: bool = False) -> List[int]:
    if chat:
        s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
        return tok.encode(s, add_special_tokens=False)
    return [tok.bos_token_id] + tok.encode(text, add_special_tokens=False)


def forward(lm: Any, ids: List[int], iv: Any = None, spare: bool = True, bos: int = 2) -> Any:
    x = torch.tensor([ids])
    keep = ao._Keep()
    keep.mask = ao.keep_mask(x, None, bos, spare)
    with ao.installed(lm, iv, keep):
        return lm(input_ids=x, use_cache=False)


def observe(lm: Any, ids: List[int], iv: Any, watch: List[Tuple[Any, str]], spare: bool = True, bos: int = 2) -> List[List[torch.Tensor]]:
    """One forward under the interventions, with observers registered after them (so they see what was pushed):
    `watch` [(module, "pre" | "post")]; returns each observer's calls."""
    x = torch.tensor([ids])
    keep = ao._Keep()
    keep.mask = ao.keep_mask(x, None, bos, spare)
    with ao.installed(lm, iv, keep):
        seen = [Seen(m, kind) for m, kind in watch]
        try:
            lm(input_ids=x, use_cache=False)
        finally:
            for s in seen:
                s.h.remove()
    return [s.calls for s in seen]


# ---------------------------------------------------------------------------------------------------------------------


def check_pushes(lm: Any, w: ao.Weights, tok: Any) -> None:
    before = len(failures)
    ids = ids_of(tok, "Alice called Bob because she wanted to borrow his car.")
    T = len(ids)
    L, n, dose, p50 = 2, 17, 0.6, 3.5
    iv = ao.resolve(w, [ao.Push(L, n, "add", dose)], [], (1,), lambda l: p50, lambda l, i: 1.0)
    u = w.write_dirs(L, [n])[0]
    check(abs(float(u.norm()) - 1.0) < 1e-5, "add: the write direction is not of unit length")
    watch = [(layer, "post") for layer in lm.layers]
    outs0 = [c[0] for c in observe(lm, ids, None, watch)]
    outs1 = [c[0] for c in observe(lm, ids, iv, watch)]
    for l in range(L):
        check(torch.equal(outs0[l], outs1[l]), f"add: layer {l}, before the push, changed")
    diff = outs1[L][0] - outs0[L][0]  # [T, D]
    check(float(diff[0].abs().max()) == 0.0, "add: the start mark was pushed")
    want = dose * p50 * u
    check(float((diff[1:] - want[None]).abs().max()) < 1e-5, "add: the push is not dose * p50 * u at every other piece")
    check(float((outs1[L + 1] - outs0[L + 1]).abs().max()) > 1e-4, "add: the layer after the push did not change")
    # unspared: the start mark is pushed too
    got = observe(lm, ids, iv, [(lm.layers[L], "post")], spare=False)[0][0]
    check(float((got[0, 0] - outs0[L][0, 0] - want).abs().max()) < 1e-5, "add, spare_bos false: the start mark not pushed")

    # mul: x dose on the neuron's own activation only
    down = [(lm.layers[L].mlp.down_proj, "pre")]
    iv = ao.resolve(w, [ao.Push(L, n, "mul", 2.5)], [], (1,), lambda l: p50, lambda l, i: 1.0)
    a0 = observe(lm, ids, None, down)[0][0][0]
    a1 = observe(lm, ids, iv, down)[0][0][0]
    mask = torch.zeros_like(a0, dtype=torch.bool)
    mask[1:, n] = True
    check(torch.equal(a0[~mask], a1[~mask]), "mul: other activations (or the start mark) changed")
    check(torch.allclose(a1[1:, n], 2.5 * a0[1:, n], atol=1e-6), "mul: not x2.5")
    # act: + dose * scale
    iv = ao.resolve(w, [ao.Push(L, n, "act", 0.5)], [], (1,), lambda l: p50, lambda l, i: 4.0)
    a2 = observe(lm, ids, iv, down)[0][0][0]
    check(torch.equal(a0[~mask], a2[~mask]), "act: other activations changed")
    check(torch.allclose(a2[1:, n], a0[1:, n] + 2.0, atol=1e-6), "act: not + 0.5 * 4")
    # a whole block silenced
    iv = ao.resolve(w, [ao.Push(L, None, "mul", 0.0)], [], (1,), lambda l: p50, lambda l, i: 1.0)
    s3 = observe(lm, ids, iv, [(lm.layers[L].post_feedforward_layernorm, "post")])[0][0]
    check(float(s3[0, 1:].abs().max()) == 0.0, "mul 0 on the whole block: the block still writes")
    # a head silenced
    sh = ao.attn_shape(lm, L)
    iv = ao.resolve(w, [], [ao.HeadScale(L, 1, 0.0)], (1,), lambda l: p50, lambda l, i: 1.0)
    y = observe(lm, ids, iv, [(lm.layers[L].self_attn.o_proj, "pre")])[0][0][0].view(T, sh.n_heads, sh.head_dim)
    check(float(y[1:, 1].abs().max()) == 0.0 and float(y[1:, 0].abs().max()) > 0, "head x0: head 1 not silenced alone")
    check(float(y[0, 1].abs().max()) > 0, "head x0: the start mark's head output was silenced")
    try:
        ao.check_pushes(lm, [ao.Push(L, None, "add", 1.0)])
        check(False, "a whole block pushed by add was not refused")
    except ValueError:
        pass
    try:
        ao.check_pushes(lm, [], [ao.HeadScale(L, sh.n_heads, 0.0)])
        check(False, "a head past the last was not refused")
    except ValueError:
        pass
    # the norm after a block (or after attention) undoes any uniform scale of all of it: only silencing is taken
    for bad, what in (([ao.Push(L, None, "mul", 2.0)], []), ([], [ao.HeadScale(L, None, 0.5)])):
        try:
            ao.check_pushes(lm, bad, what)
            check(False, f"a uniform scale the norm undoes was not refused: {bad or what}")
        except ValueError as exc:
            check("only be silenced" in str(exc), f"refusal: {exc}")
    ao.check_pushes(lm, [ao.Push(L, None, "mul", 0.0)], [ao.HeadScale(L, None, 0.0), ao.HeadScale(L, 1, 2.0)])
    # the most a neuron can write by itself: firing alone and hard, the post-feed-forward norm caps its write
    for ll, nn_ in ((1, 3), (4, 77)):
        col = lm.layers[ll].mlp.down_proj.weight[:, nn_]
        capped = lm.layers[ll].post_feedforward_layernorm((1e4 * col)[None, None])[0, 0] * float(w.scalars[ll])
        check(abs(float(capped.norm()) - float(w.write_ceiling(ll, [nn_])[0])) < 1e-3 * float(capped.norm()),
              f"write_ceiling L{ll} N{nn_}: what the norm lets one neuron write")
        check(float(lm.layers[ll].post_feedforward_layernorm((0.5 * col)[None, None])[0, 0].norm() * float(w.scalars[ll]))
              <= float(w.write_ceiling(ll, [nn_])[0]) + 1e-6, f"write_ceiling L{ll} N{nn_}: a softer firing writes less")
    section("each push lands where it says (add after the norm and scalar, mul, act, a block, a head; the start mark spared); "
            "what the norms undo is refused; the most a neuron can write", before)

    before = len(failures)
    iv = ao.resolve(w, [ao.Push(L, n, "add", 0.8), ao.Push(L, 5, "mul", 3.0), ao.Push(L, 7, "act", 1.0)], [ao.HeadScale(L, 0, 0.5)],
                    ao.THREE_VOICES, lambda l: 2.0, lambda l, i: 10.0)
    adds = iv.adds[L]
    check(torch.allclose(adds[0], 0.8 * 2.0 * u) and float(adds[1].abs().max()) == 0 and torch.allclose(adds[2], -0.8 * 2.0 * u),
          "resolve: the add rows are not +, 0, -")
    check(iv.muls[L][0][1].tolist() == [3.0, 1.0, -3.0], f"resolve: mul rows {iv.muls[L][0][1].tolist()}")
    check(iv.acts[L][0][1].tolist() == [10.0, 0.0, -10.0], f"resolve: act rows {iv.acts[L][0][1].tolist()}")
    check(iv.heads[L][0][1].tolist() == [0.5, 1.0, 0.5], f"resolve: head rows {iv.heads[L][0][1].tolist()}")
    section("resolve gives the three voices their rows", before)


def check_writing(lm: Any, W_U: torch.Tensor, w: ao.Weights, tok: Any, stop: List[int]) -> None:
    before = len(failures)
    prompts = [ids_of(tok, "Who are you?", chat=True), ids_of(tok, "Write a snake game in Python.", chat=True) + [7, 9, 11],
               ids_of(tok, "Describe the color orange.", chat=True)]
    pushes = [ao.Push(3, 40, "add", 1.5), ao.Push(1, 9, "mul", 4.0)]
    iv = ao.resolve(w, pushes, [], ao.THREE_VOICES, lambda l: 3.0, lambda l, i: 1.0)

    def collect(rows: List[List[int]], ivv: Any, n: int = 24) -> List[List[int]]:
        out: List[List[int]] = [[] for _ in rows]
        for st in ao.write_rows(lm, W_U, rows, ivv, max_new=n, stop_ids=stop, bos_id=tok.bos_token_id, softcap=30.0):
            out[st.row].append(st.token_id)
        return out

    together = collect(prompts, iv)
    for r, signs in enumerate(((1,), (0,), (-1,))):
        alone = collect([prompts[r]], ao.resolve(w, pushes, [], signs, lambda l: 3.0, lambda l, i: 1.0))[0]
        check(alone == together[r], f"write: row {r} together {together[r][:8]} != alone {alone[:8]}")
    plain = collect([prompts[1]], None)[0]
    check(plain == together[1], "write: the unpushed row differs from writing unpushed")
    check(together[0] != together[1] or together[2] != together[1], "write: the pushes changed nothing")
    # a cancel stops at once; an end mark ends the row
    calls = {"n": 0}

    def cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 3

    got = list(ao.write_rows(lm, W_U, prompts, iv, max_new=50, stop_ids=stop, bos_id=tok.bos_token_id, cancel=cancel))
    check(len(got) <= 9, f"write: cancel let {len(got)} pieces through")
    first = together[1][0]
    ends = list(ao.write_rows(lm, W_U, [prompts[1]], None, max_new=10, stop_ids=[first], bos_id=tok.bos_token_id))
    check(len(ends) == 1 and ends[0].end, "write: an end mark did not end the row")
    steps = list(ao.write_rows(lm, W_U, [prompts[0]], None, max_new=3, stop_ids=stop, bos_id=tok.bos_token_id, top_k=5))
    check(all(len(s.top_ids) == 5 and s.top_ids[0] == s.token_id and abs(s.top_lps[0] - s.logprob) < 1e-3 for s in steps),
          "write: top_k does not lead with the chosen piece")
    section("three voices written together are each written as alone, padded (an end mark, a cancel, top-k)", before)

    before = len(failures)
    text_ids = tok.encode("我昨天去了北京 🙂 café naïve, done.", add_special_tokens=False)
    td = ao.TextDeltas(tok, 1)
    pieces = [td.add(0, t, False) for t in text_ids]
    check("".join(pieces) == tok.decode(text_ids), f"TextDeltas: {''.join(pieces)!r}")
    check(not any(p.endswith("�") for p in pieces), "TextDeltas: a delta ends inside a character")
    check(td.add(0, 1, True) == "", "TextDeltas: an end mark wrote text")
    section("text deltas add up to the decoded text, a split character held back until whole", before)


def check_reading(lm: Any, W_U: torch.Tensor, w: ao.Weights, tok: Any) -> None:
    before = len(failures)
    ids = ids_of(tok, "The committee rejected the proposal; it was expensive.")
    L, n = 4, 33
    with Seen(lm.layers[L].mlp.down_proj, "pre") as s:
        lm(input_ids=torch.tensor([ids]), use_cache=False)
    want = s.calls[0][0, :, n].tolist()
    got = ao.read_neuron(lm, ids, L, n, None, bos_id=tok.bos_token_id)
    check(max(abs(a - b) for a, b in zip(want, got)) == 0.0, "read: not the activation a hook sees")
    iv_same = ao.resolve(w, [ao.Push(L, n, "mul", 0.0)], [], (1,), lambda l: 2.0, lambda l, i: 1.0)
    check(ao.read_neuron(lm, ids, L, n, iv_same, bos_id=tok.bos_token_id) == got, "read: a push on the neuron itself moved its read")
    iv_up = ao.resolve(w, [ao.Push(1, 3, "add", 2.0)], [], (1,), lambda l: 2.0, lambda l, i: 1.0)
    moved = ao.read_neuron(lm, ids, L, n, iv_up, bos_id=tok.bos_token_id)
    check(moved[0] == got[0] and max(abs(a - b) for a, b in zip(moved[1:], got[1:])) > 1e-5,
          "read: an earlier push did not move the read (or moved the start mark)")
    section("a read is the hook's activation, moved by earlier pushes only", before)

    before = len(failures)
    iv2 = ao.resolve(w, [ao.Push(2, 11, "add", 1.2), ao.Push(3, None, "mul", 0.0)], [ao.HeadScale(1, None, 0.0)], (0, 1),
                     lambda l: 3.0, lambda l, i: 1.0)
    base, alt = ao.next_logprobs(lm, W_U, ids, iv2, bos_id=tok.bos_token_id)
    out0 = forward(lm, ids, None)
    out1 = forward(lm, ids, ao.resolve(w, [ao.Push(2, 11, "add", 1.2), ao.Push(3, None, "mul", 0.0)],
                                        [ao.HeadScale(1, None, 0.0)], (1,), lambda l: 3.0, lambda l, i: 1.0))
    b2 = F.log_softmax(logits_fp32(out0.last_hidden_state[:, -1], W_U), -1)[0]
    a2 = F.log_softmax(logits_fp32(out1.last_hidden_state[:, -1], W_U), -1)[0]
    check(torch.allclose(base, b2, atol=1e-5) and torch.allclose(alt, a2, atol=1e-5), "logprobs: rows differ from separate forwards")
    kl = float((base.exp() * (base - alt)).sum())
    check(kl > 0, f"logprobs: KL {kl}")
    only, none = ao.next_logprobs(lm, W_U, ids, None, bos_id=tok.bos_token_id)
    check(none is None and torch.allclose(only, b2, atol=1e-5), "logprobs: without interventions")
    section("next_logprobs' base and intervened rows equal separate forwards", before)

    before = len(failures)
    texts = [ids_of(tok, t) for t in ("Orange is warm.", "def f(x):\n    return x", "我昨天去了北京")]
    acts = ao.all_activations(lm, texts[0], dtype=torch.float32)
    for l in (0, 5):
        with Seen(lm.layers[l].mlp.down_proj, "pre") as s:
            lm(input_ids=torch.tensor([texts[0]]), use_cache=False)
        check(torch.equal(acts[l], s.calls[0][0]), f"all_activations: layer {l}")
    ex = ao.activation_extremes(lm, texts)
    for l in (0, 3):
        rows = []
        for t in texts:
            with Seen(lm.layers[l].mlp.down_proj, "pre") as s:
                lm(input_ids=torch.tensor([t]), use_cache=False)
            rows.append(s.calls[0][0, 1:])
        a = torch.cat(rows)
        check(torch.allclose(ex.pos[l], a.amax(0).clamp(min=0)), f"extremes: pos at layer {l}")
        check(torch.allclose(ex.neg[l], a.amin(0).clamp(max=0)), f"extremes: neg at layer {l}")
        check(torch.allclose(torch.maximum(ex.pp[l], ex.nn[l]), ex.pos[l]), f"extremes: pp/nn do not make pos at layer {l}")
        check(torch.allclose(torch.minimum(ex.pn[l], ex.np[l]), ex.neg[l]), f"extremes: pn/np do not make neg at layer {l}")
    check(ex.tokens == sum(len(t) - 1 for t in texts), f"extremes: tokens {ex.tokens}")
    check(all(len(v) == ex.tokens for v in ex.resid_norms), "extremes: one stream length per piece read")
    section("all activations and their extremes (four sign regimes) equal brute force, start mark left out", before)

    before = len(failures)
    for text in ("我昨天去了北京，🙂👍🏽 café", "plain words only", "é ⚠️ 👨‍👩‍👧"):
        t_ids = tok.encode(text, add_special_tokens=False)
        units = ao.token_units(tok, t_ids)
        flat = [i for _, idx in units for i in idx]
        check(flat == list(range(len(t_ids))), f"units: {text!r} indices {flat}")
        check("".join(u for u, _ in units) == tok.decode(t_ids), f"units: {text!r} do not join to the text")
        check(not any(u.endswith("�") for u, _ in units[:-1]), f"units: {text!r} a unit cut inside a character")
    section("display units cover every piece once and never cut a character", before)


def check_heads(hf: Any, tok: Any, w: ao.Weights) -> None:
    before = len(failures)
    lm = language_model(hf)
    eager = stand_in_hf(tok, torch.float32, attn="eager")
    eager.load_state_dict(hf.state_dict())
    for a, b in zip(eager.model.layers, lm.layers):
        a.layer_scalar.copy_(b.layer_scalar)
    ids = ids_of(tok, "Alice called Bob because she wanted to borrow his car. He said yes, and she drove it.")
    T = len(ids)
    check(T > 2 * 8, f"heads: the text ({T}) does not pass the window")
    with torch.no_grad():
        out = eager(input_ids=torch.tensor([ids]), output_attentions=True, use_cache=False)
    cap = ao.capture_heads(lm, ids, w, streams=True)
    shapes = [ao.attn_shape(lm, l) for l in range(len(lm.layers))]
    worst = 0.0
    for l, sh in enumerate(shapes):
        A_hf = out.attentions[l][0]  # [H, T, T]
        A = ao.layer_patterns(cap, sh)
        worst = max(worst, float((A - A_hf).abs().max()))
        # each head's output is its weights times the values
        for h in range(sh.n_heads):
            y = A[h] @ cap.v[l][sh.kv_of(h)]
            check(torch.allclose(y, cap.y[l][h], atol=1e-5), f"heads: layer {l} head {h} output != weights x values")
        # the heads' writes add up to the post-attention norm's output (times the layer scalar)
        W_O = w.attn_weight(l, "o").float()
        w_pa = w.norm(l, "post_attention")
        y_all = cap.y[l].transpose(0, 1).reshape(T, -1)  # [T, H*hd]
        o = y_all @ W_O.T
        normed = lm.layers[l].post_attention_layernorm(o)
        parts = torch.stack([(cap.y[l][h] @ W_O[:, h * sh.head_dim:(h + 1) * sh.head_dim].T) * w_pa / cap.o_rms[l][:, None]
                             for h in range(sh.n_heads)])
        check(torch.allclose(parts.sum(0), normed, atol=1e-4), f"heads: layer {l} parts do not add up to the norm's output")
        s = float(w.scalars[l])
        check(torch.allclose(cap.writes[l], s * parts.norm(dim=-1).T, atol=1e-4), f"heads: layer {l} write sizes")
    check(worst < 1e-5, f"heads: patterns differ from eager attention by {worst:.2e}")
    check(shapes[-1].values_from_keys and shapes[-1].kind == "full" and shapes[0].window == 8, "heads: layer kinds")
    section(f"head patterns equal eager attention on every head (worst {worst:.1e}); outputs and writes add up", before)

    before = len(failures)
    for x in (-3.0, -0.7, 0.0, 0.4, 2.5):
        t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        with torch.enable_grad():
            (gr,) = torch.autograd.grad(F.gelu(t, approximate="tanh"), t)
        check(abs(ao.gelu_tanh_grad(x) - float(gr)) < 1e-12, f"gelu_tanh_grad({x}) != autograd")
    for (l, n, p) in ((5, 21, T - 1), (3, 70, 12), (0, 2, 5)):
        d = ao.attention_drive(cap, w, shapes, l, n, p, top=4)
        a_true = float(cap_act(lm, ids, l, n)[p])
        check(abs(d["act"] - a_true) < 1e-4 * max(1.0, abs(a_true)), f"drive: act {d['act']} != {a_true} (L{l} N{n})")
        parts = sum(c for _, _, c in d["heads"]) + sum(c for _, c in d["mlps"]) + d["embed"]
        check(abs(parts - d["total"]) < 1e-6 * max(1.0, abs(d["total"])), f"drive: parts {parts} != total {d['total']} (L{l})")
        check(abs(d["rest"]) < 1e-6 * max(1.0, abs(d["total"])), f"drive: rest {d['rest']}")
        for (ll, h), src in d["src"].items():
            c = next(c for (lx, hx, c) in d["heads"] if lx == ll and hx == h)
            check(abs(sum(src) - c) < 1e-6 * max(1.0, abs(c)), f"drive: L{ll} h{h} split {sum(src)} != {c}")
            check(len(src) == p + 1, "drive: the split does not cover pieces 0..pos")
        check(len(d["heads"]) == sum(sh.n_heads for sh in shapes[: l + 1]) and len(d["mlps"]) == l, "drive: parts counted")
    section("attention drive: its parts (heads, blocks, embedding, through the layer scalars) add up to the total", before)


def cap_act(lm: Any, ids: List[int], l: int, n: int) -> torch.Tensor:
    with Seen(lm.layers[l].mlp.down_proj, "pre") as s:
        lm(input_ids=torch.tensor([ids]), use_cache=False)
    return s.calls[0][0, :, n]


def wiring_scale(lm: Any, w: ao.Weights, tok: Any) -> torch.Tensor:
    texts = [ids_of(tok, t) for t in ("Orange is warm.", "def f(x):\n    return x", "The river ran past the mill.")]
    sc = ao.norm_scales(ao.activation_extremes(lm, texts))
    return ao.link_scale(w.scalars, sc["mlp_rms"], sc["mid_rms"])


def check_wiring(hf: Any, lm: Any, w: ao.Weights, tok: Any) -> torch.Tensor:
    import copy

    before = len(failures)
    L = len(lm.layers)
    # the norms' scales are what each norm divides by, as the forward sees it
    texts = [ids_of(tok, t) for t in ("Orange is warm.", "def f(x):\n    return x")]
    ex = ao.activation_extremes(lm, texts)
    for l in (0, 3):
        with Seen(lm.layers[l].mlp, "post") as s_m, Seen(lm.layers[l].pre_feedforward_layernorm, "pre") as s_p:
            for t in texts:
                lm(input_ids=torch.tensor([t]), use_cache=False)
        eps_m, eps_p = float(lm.layers[l].post_feedforward_layernorm.eps), float(lm.layers[l].pre_feedforward_layernorm.eps)
        m_ = torch.cat([c_[0, 1:] for c_ in s_m.calls]).pow(2).mean(-1).add(eps_m).sqrt()
        p_ = torch.cat([c_[0, 1:] for c_ in s_p.calls]).pow(2).mean(-1).add(eps_p).sqrt()
        check(torch.allclose(torch.tensor(ex.mlp_rms[l]), m_, rtol=1e-5) and torch.allclose(torch.tensor(ex.mid_rms[l]), p_, rtol=1e-5),
              f"norm scales at layer {l}: what the post- and pre-feed-forward norms divide by")
    sc = ao.norm_scales(ex)
    f = ao.link_scale(w.scalars, sc["mlp_rms"], sc["mid_rms"])
    s = w.scalars.double()
    check(abs(float(f[1, 4]) - float(s[1] * s[2] * s[3]) / (sc["mlp_rms"][1] * sc["mid_rms"][4])) < 1e-12 * max(1.0, abs(float(f[1, 4])))
          and float(f[4, 1]) == 0.0 and float(f[2, 2]) == 0.0, "link_scale: s_i...s_(j-1) over the two norms' scales, only for i < j")

    # what a writer's unit of activation adds to the stream entering a later block, with only the direct path open
    # (every head and every block between them silenced): carried through exactly the layer scalars s_i ... s_(j-1)
    i, j, A, p, delta = 1, 4, 7, 3, 1e-3
    m2 = copy.deepcopy(hf)
    lm2 = language_model(m2)
    with torch.no_grad():
        for ll, layer in enumerate(lm2.layers):
            layer.self_attn.o_proj.weight.zero_()
            if i < ll < j:
                layer.mlp.down_proj.weight.zero_()
    ids = torch.tensor([ids_of(tok, "Orange is a warm colour.")])

    def run(push: float) -> Tuple[torch.Tensor, torch.Tensor]:
        got: Dict[str, torch.Tensor] = {}

        def pre(_m: Any, args: Tuple[Any, ...]) -> Tuple[Any, ...]:
            x = args[0].clone()
            x[0, p, A] += push
            return (x,)

        hs = [lm2.layers[i].mlp.down_proj.register_forward_pre_hook(pre),
              lm2.layers[i].mlp.register_forward_hook(lambda _m, _a, o: got.__setitem__("m", o[0, p].detach().clone())),
              lm2.layers[j].pre_feedforward_layernorm.register_forward_pre_hook(lambda _m, a: got.__setitem__("mid", a[0][0, p].detach().clone()))]
        try:
            lm2(input_ids=ids, use_cache=False)
        finally:
            for h in hs:
                h.remove()
        return got["m"], got["mid"]

    m0, mid0 = run(0.0)
    _, mid1 = run(delta)
    col = lm2.layers[i].mlp.down_proj.weight[:, A].double()
    m0 = m0.double()
    D = m0.shape[0]
    r = float((m0.pow(2).mean() + float(lm2.layers[i].post_feedforward_layernorm.eps)).sqrt())
    w_post = lm2.layers[i].post_feedforward_layernorm.weight.double()
    write = w_post * (col - m0 * (m0 @ col) / (D * r * r)) / r  # the post-feed-forward norm's first-order change, per unit
    pred = float(torch.prod(s[i:j])) * write * delta
    got_d = (mid1 - mid0).double()
    check(float((got_d - pred).norm() / pred.norm()) < 5e-3,
          f"a writer reaches a block {j - i} layers on carried by s_{i}...s_{j - 1}: {float((got_d - pred).norm() / pred.norm()):.2e} off")

    def brute(i_: int, a: int, j_: int, b: int) -> Tuple[float, float]:
        v = lm.layers[i_].mlp.down_proj.weight[:, a] * lm.layers[i_].post_feedforward_layernorm.weight
        x = v * lm.layers[j_].pre_feedforward_layernorm.weight
        g = float(lm.layers[j_].mlp.gate_proj.weight[b] @ x)
        return g * float(lm.layers[j_].mlp.up_proj.weight[b] @ x), g

    for (l, n) in ((1, 12), (4, 100)):
        res = ao.neuron_wiring(w, l, n, f, k=10)
        effs = {}
        for side in ("down", "up"):
            for c in res[side][:6]:
                P, g = brute(l, n, c["layer"], c["n"]) if side == "down" else brute(c["layer"], c["n"], l, n)
                pair = (torch.tensor([abs(brute(l, n, c["layer"], b)[0]) for b in range(w.d_ff)]) if side == "down"
                        else torch.tensor([abs(brute(c["layer"], a, l, n)[0]) for a in range(w.d_ff)]))
                z = (abs(P) - pair.mean()) / pair.std()
                fij = float(f[l, c["layer"]] if side == "down" else f[c["layer"], l])
                effs[(side, c["layer"], c["n"])] = abs(P) * fij * fij
                check(abs(float(z) - c["z"]) < 0.02 and (1 if P > 0 else -1) == c["sign"] and (1 if g > 0 else -1) == c["gate"],
                      f"wiring {side} L{l}N{n} ~ {c}: z, sign(P), the gate sign(g)")
        for side in ("down", "up"):
            top = max(e for (s_, _, _), e in effs.items() if s_ == side)
            check(all(abs(c["w"] - effs[(side, c["layer"], c["n"])] / top) <= 1e-3 * max(c["w"], 1e-9) + 1e-12 for c in res[side][:6])
                  and res[side][0]["w"] == 1.0, f"wiring {side} L{l}N{n}: w = |P| f^2 over the strongest")
            check(res[side] == sorted(res[side], key=lambda c: -c["w"]), f"wiring {side} L{l}N{n}: ordered by strength across layers")
        check(all(c["layer"] > l for c in res["down"]) and all(c["layer"] < l for c in res["up"]), "wiring: sides")
    # across layers the strength, not z, decides: a link with a smaller z can outrank one with a larger
    res = ao.neuron_wiring(w, 1, 12, f, k=24)
    zs = [c["z"] for c in res["down"]]
    check(zs != sorted(zs, reverse=True) or len({c["layer"] for c in res["down"]}) == 1, "wiring: the strength orders the links, not z alone")
    # a neuron that writes nothing reaches no reader
    saved = lm.layers[2].mlp.down_proj.weight.data[:, 5].clone()
    lm.layers[2].mlp.down_proj.weight.data[:, 5] = 0.0
    try:
        check(ao.neuron_wiring(w, 2, 5, f, k=10)["down"] == [], "a neuron that writes nothing has no readers")
    finally:
        lm.layers[2].mlp.down_proj.weight.data[:, 5] = saved
    section("wiring: the norms' scales, the carried layer scalars, the products one by one, z, w, sign and gate", before)
    return f

    before = len(failures)
    shapes = [ao.attn_shape(lm, l) for l in range(L)]
    sp = ao.HeadSpaces(w, shapes)
    g = torch.Generator().manual_seed(3)
    d = torch.randn(w.d_model, 4, generator=g)
    for l in (0, L - 1):
        sh = shapes[l]
        for role in ("q", "k", "v"):
            got = sp.read_share(l, role, d)
            W = w.attn_weight(l, role).float() * w.norm(l, "input")[None, :]
            units = sh.n_heads if role == "q" else sh.n_kv
            for u_i in range(units):
                Q = torch.linalg.qr(W[u_i * sh.head_dim:(u_i + 1) * sh.head_dim].T.double())[0]
                want = (Q.T @ d.double()).norm(dim=0) / d.double().norm(dim=0)
                check(torch.allclose(got[u_i], want, atol=1e-6), f"spaces: L{l} {role}{u_i}")
        r = torch.randn(3, w.d_model, generator=g)
        got = sp.write_share(l, r)
        W_O = w.attn_weight(l, "o").float() * w.norm(l, "post_attention")[:, None]
        for h in range(sh.n_heads):
            Q = torch.linalg.qr(W_O[:, h * sh.head_dim:(h + 1) * sh.head_dim].double())[0]
            want = (Q.T @ r.double().T).norm(dim=0) / r.double().norm(dim=1)
            check(torch.allclose(got[h], want, atol=1e-6), f"spaces: L{l} o{h}")
    roles = ao.neuron_head_roles(sp, 2, 9, k=3)
    check(set(roles) >= {"q", "k", "v", "o", "z_by_layer"}, "roles: keys")
    check(all(x["layer"] > 2 for r in ("q", "k", "v") for x in roles[r]) and all(x["layer"] <= 2 for x in roles["o"]),
          "roles: q/k/v must be later heads, o this layer or before")
    hw = ao.head_wiring(sp, 3, 1, k=5)
    check(all(c["layer"] < 3 for r in ("q", "k", "v") for c in hw[r]) and all(c["layer"] >= 3 for c in hw["o"]),
          "head wiring: sides")
    # the top v neuron of head (3, 1) has the largest value-space share among every earlier neuron
    best = max(((float(sp.read_share(3, "v", (w.down_cols(i, [a])[0] * w.norm(i, "post_feedforward"))[:, None])[shapes[3].kv_of(1), 0]), i, a)
                for i in range(3) for a in range(w.d_ff)))
    check((hw["v"][0]["layer"], hw["v"][0]["n"]) == (best[1], best[2]), f"head wiring: top v {hw['v'][0]} != {best}")
    section("head spaces equal projections on explicit bases; roles and head wiring rank by them", before)


def check_steer_lens(lm: Any, W_U: torch.Tensor, w: ao.Weights, tok: Any) -> None:
    before = len(failures)
    ctx = [ids_of(tok, c, chat=True) for c in ao.STEER_LENS_CHATS] + [ids_of(tok, c) for c in ao.STEER_LENS_PLAIN]
    l, n, p50 = 2, 50, 3.0
    res = ao.steer_lens(lm, W_U, ctx, w, l, n, p50, k=6, bos_id=tok.bos_token_id)
    diffs = {1: [], -1: []}
    kls = {1: [], -1: []}
    for c in ctx:
        b = F.log_softmax(logits_fp32(forward(lm, c, None).last_hidden_state[:, -1], W_U), -1)[0]
        for sgn in (1, -1):
            iv = ao.resolve(w, [ao.Push(l, n, "add", sgn * ao.STEER_LENS_DOSE)], [], (1,), lambda _l: p50, lambda a, b2: 1.0)
            s = F.log_softmax(logits_fp32(forward(lm, c, iv).last_hidden_state[:, -1], W_U), -1)[0]
            diffs[sgn].append(s - b)
            kls[sgn].append(float((s.exp() * (s - b)).sum()))
    for sgn, key in ((1, "up"), (-1, "down")):
        mean = torch.stack(diffs[sgn]).mean(0)
        want = torch.topk(mean, 6).indices.tolist()
        check(res[key] == want, f"steer lens {key}: {res[key]} != {want}")
        check(abs(res[f"kl_{key}"] - sum(kls[sgn]) / len(kls[sgn])) < 1e-3, f"steer lens kl_{key}")
    section("the steer lens equals separate pushed forwards (rises and KL, both signs)", before)


def main() -> int:
    torch.set_grad_enabled(False)
    tok = stand_in_tokenizer()
    hf = stand_in_hf(tok, torch.float32)
    rt = stand_in_runtime(hf, tok)
    lm = language_model(hf)
    w = ao.Weights(lm)
    W_U = hf.lm_head.weight.detach()
    stop = rt.end_of_turn_ids()
    print(f"stand-in: {len(lm.layers)} layers, {w.d_ff} neurons each, d_model {w.d_model}, vocabulary {len(tok)}")
    check_pushes(lm, w, tok)
    check_writing(lm, W_U, w, tok, stop)
    check_reading(lm, W_U, w, tok)
    check_heads(hf, tok, w)
    f = check_wiring(hf, lm, w, tok)
    check_steer_lens(lm, W_U, w, tok)
    # the runtime's own weight accessors give the same wiring as the plain modules
    before = len(failures)
    wr = ao.Weights(lm, rt)
    check(ao.neuron_wiring(wr, 2, 31, f, k=8) == ao.neuron_wiring(w, 2, 31, f, k=8), "wiring through the runtime differs")
    section("the runtime's weight accessors give the same wiring", before)
    print(f"\n{passed} checks passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

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
   characters split over pieces held back until whole; written with the instrument's sampler: at temperature 0 every
   Step is the greedy one; a seed draws the same again and another seed other pieces; with every push an add or an act
   at dose 0 the voices of a sample write alike (the shared draw); sample s's unpushed row is an unpushed write alone
   with seed + s; over 4,000 seeds the first piece, and the second after a presence penalty that moves it, come as often
   as `sampler_chances` says (chi-square), and the race itself on sharper logits; each piece's `sampler_prob` is above 0
   and its chance replayed; a presence penalty without a temperature, a sampled write without a seed, and a temperature
   so small that the sampler has no chances to draw from (the race would take piece 0), are refused;
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
10. `steer_lens`' base rows are the unpushed forwards and its rises are those of separate pushed forwards;
11. the sign in use (LINK_MODEL_USE), on a float64 copy of the stand-in, at a reader's stored events (`passage_events`):
    the operating points are the medians of gelu'(g) u and gelu(g) over each face's events; a writer's write, pushed by
    +-eps along the direct path alone with both norms held at their values there, moves the reader by exactly
    f (alpha g + beta u) eps to first order, alpha and beta from that position's own gate and up values, so with that
    sign; the writer's own activation pushed by +-eps and carried by the direct path alone, the norms free, moves it by
    the full first-order change (the norms' own response included), whose sign is the link model's at most events
    (printed, not asserted); the median-based sign agrees with each event's own at most events (printed);
12. the wiring measured: `pushed_reads`' batched rows (writers at two layers in one prefill, chunks of several sizes)
    equal separate forwards, one row each, unpushed, +d and -d, at the last word-piece and over the prompt; each
    reader's g0 and u0 are what plain hooks on its gate_proj and up_proj give on the unpushed row, at every word-piece
    past the start mark, right-aligned, NaN before a shorter question's first, and the unpushed read at the last
    word-piece is gelu_tanh(g0) u0; GELU_MIN, where the check's regimes part, is where gelu_tanh is least; where the
    reader sits, a small push moves it by alpha0 dg + beta0 du to first order (float32 central differences), alpha0
    and beta0 from those g0 and u0, dg and du its own gate and up changes (with the link's g and u from the weights in
    their place, the share of signs is printed, not asserted); the null's readers are never the writer's linked ones,
    one at each linked reader's layer, none twice; `pair_products` gives `neuron_wiring`'s g and u.

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


def chi_square_p(counts: Dict[int, int], chances: torch.Tensor) -> Tuple[float, int]:
    """The chi-square test of `counts` (piece -> how often drawn) against `chances` [V]: pieces expected fewer than 5
    times pooled into one bin; (p, degrees of freedom). A piece drawn at chance 0 gives p = 0."""
    n = sum(counts.values())
    if any(float(chances[t]) <= 0 for t in counts):
        return 0.0, 0
    obs, exp, rest_o, rest_e = [], [], 0.0, 0.0
    for t in torch.nonzero(chances > 0).flatten().tolist():
        e = n * float(chances[t])
        if e >= 5:
            obs.append(float(counts.get(t, 0)))
            exp.append(e)
        else:
            rest_o, rest_e = rest_o + counts.get(t, 0), rest_e + e
    if rest_e > 0:
        obs.append(rest_o)
        exp.append(rest_e)
    stat = sum((o - e) ** 2 / e for o, e in zip(obs, exp))
    df = len(obs) - 1
    return float(torch.special.gammaincc(torch.tensor(df / 2, dtype=torch.float64), torch.tensor(stat / 2, dtype=torch.float64))), df


def check_sampling(lm: Any, W_U: torch.Tensor, w: ao.Weights, tok: Any, stop: List[int]) -> None:
    from interp_engine.sampling import resolve_sampling

    from backend.engine_runtime import SamplerSettings, sampler_chances

    before = len(failures)
    prompts = [ids_of(tok, "Who are you?", chat=True), ids_of(tok, "Write a snake game in Python.", chat=True) + [7, 9, 11],
               ids_of(tok, "Describe the color orange.", chat=True)]
    pushes = [ao.Push(3, 40, "add", 1.5), ao.Push(1, 9, "mul", 4.0)]
    iv = ao.resolve(w, pushes, [], ao.THREE_VOICES, lambda l: 3.0, lambda l, i: 1.0)

    def steps(rows: List[List[int]], ivv: Any, n: int = 16, **kw: Any) -> List[ao.Step]:
        return list(ao.write_rows(lm, W_U, rows, ivv, max_new=n, stop_ids=stop, bos_id=tok.bos_token_id, softcap=30.0, top_k=5, **kw))

    def pieces(st: List[ao.Step], rows: int) -> List[List[int]]:
        out: List[List[int]] = [[] for _ in range(rows)]
        for s in st:
            out[s.row].append(s.token_id)
        return out

    # (a) greedy: no sampler, the default sampler, and a temperature of 0 with filters (which greedy ignores)
    want = steps(prompts, iv)
    for label, smp in (("the default sampler", SamplerSettings()), ("temperature 0 with top-k, top-p and a seed",
                                                                    SamplerSettings(temperature=0.0, top_k=5, top_p=0.5, seed=9))):
        got = steps(prompts, iv, sampler=smp)
        check(got == want and all(s.sampler_prob is None for s in got), f"sampled write: {label} is not the greedy write, Step for Step")
    hot = SamplerSettings(temperature=1.5, seed=11)
    # (b) a seed draws the same again; another seed other pieces
    one, again, other = steps(prompts, iv, sampler=hot), steps(prompts, iv, sampler=hot), steps(prompts, iv, sampler=SamplerSettings(temperature=1.5, seed=12))
    check(one == again, "sampled write: the same seed does not draw the same again")
    check(pieces(one, 3) != pieces(other, 3), "sampled write: another seed drew the same pieces")
    check(pieces(one, 3) != pieces(want, 3), "sampled write: temperature 1.5 wrote the greedy answer")
    # (f) every piece drawn had a chance, and its chance is the sampler's at that step, replayed one row at a time
    check(all(s.sampler_prob is not None and s.sampler_prob > 0 for s in one), "sampled write: a piece drawn at chance 0")
    settings = resolve_sampling(None, temperature=1.5, top_k=0, top_p=1.0, presence_penalty=0.0)
    alone = steps([prompts[1]], ao.resolve(w, pushes, [], (0,), lambda l: 3.0, lambda l, i: 1.0), n=8, sampler=hot)
    so_far: List[int] = []
    worst = 0.0
    for s in alone:
        h = lm(input_ids=torch.tensor([prompts[1] + so_far]), use_cache=False).last_hidden_state[0, -1]
        ch = sampler_chances(ao._model_logits(lm, W_U, h[None], 30.0)[0], settings, so_far)
        worst = max(worst, abs(float(ch[s.token_id]) - s.sampler_prob) / float(ch[s.token_id]))
        so_far.append(s.token_id)
    check(worst < 1e-4, f"sampled write: sampler_prob is not the replayed chance of the piece (worst relative gap {worst:.2e})")
    # (c) the shared draw: with every push an add or an act at dose 0 the voices of each sample write alike (a mul at
    # dose 0 would silence its neuron in the pushed rows), here two samples at once
    zero = ao.resolve(w, [ao.Push(3, 40, "add", 0.0), ao.Push(2, 7, "act", 0.0)], [], ao.THREE_VOICES * 2, lambda l: 3.0, lambda l, i: 2.0)
    z = pieces(steps([prompts[0]] * 6, zero, n=20, sampler=hot, samples_of=[0, 0, 0, 1, 1, 1]), 6)
    check(z[0] == z[1] == z[2] and z[3] == z[4] == z[5] and z[0] != z[3],
          "sampled write: with every push an add or an act at dose 0 the three voices of a sample did not write alike (or two samples did)")
    # (d) sample s's unpushed row is an unpushed write alone with seed + s; a real push moves its voices (at 0.7: on the
    # stand-in's flat odds at 1.5 the shared draw keeps pushed voices with the unpushed one almost everywhere)
    iv3 = ao.resolve(w, pushes, [], ao.THREE_VOICES * 3, lambda l: 3.0, lambda l, i: 1.0)
    three = steps([prompts[2]] * 9, iv3, n=20, sampler=SamplerSettings(temperature=0.7, seed=11), samples_of=[0, 0, 0, 1, 1, 1, 2, 2, 2])
    rows9 = pieces(three, 9)
    for s in range(3):
        solo = steps([prompts[2]], None, n=20, sampler=SamplerSettings(temperature=0.7, seed=11 + s))
        mine = [x for x in three if x.row == 3 * s + 1]
        check([x.token_id for x in solo] == rows9[3 * s + 1] and all(abs(a.sampler_prob - b.sampler_prob) < 1e-6 for a, b in zip(solo, mine)),
              f"sampled write: sample {s}'s unpushed row is not the unpushed write alone with seed {11 + s}")
    check(any(rows9[3 * s] != rows9[3 * s + 1] or rows9[3 * s + 2] != rows9[3 * s + 1] for s in range(3)),
          "sampled write: the pushes moved no sampled voice")
    section("sampled writing: temperature 0 is greedy; a seed draws again; the voices of a sample share their draw; a sample's "
            "unpushed row is the write alone with seed + s; each chance is the sampler's", before)

    # (e) over 4,000 seeds the first piece comes as often as the sampler says (500 samples a batch, seeds 0-3,999), and the
    # second, after the commonest first piece, under a presence penalty that moves it (favouring the piece just written)
    before = len(failures)
    ids = prompts[0]
    h0 = lm(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state[0, -1]
    l0 = ao._model_logits(lm, W_U, h0[None], 30.0)[0]
    for T, pp in ((0.7, 0.0), (1.5, 0.0), (0.7, -1.5)):
        smp = dict(temperature=T, top_k=5, top_p=0.9, presence_penalty=pp)
        firsts: Dict[int, int] = {}
        seconds: Dict[int, Dict[int, int]] = {}
        for b in range(8):
            got = pieces(steps([ids] * 500, None, n=2, sampler=SamplerSettings(**smp, seed=500 * b), samples_of=list(range(500))), 500)
            for r in got:
                firsts[r[0]] = firsts.get(r[0], 0) + 1
                if len(r) > 1:
                    seconds.setdefault(r[0], {})
                    seconds[r[0]][r[1]] = seconds[r[0]].get(r[1], 0) + 1
        st = resolve_sampling(None, **smp)
        p, df = chi_square_p(firsts, sampler_chances(l0, st, []))
        check(p > 1e-3 and df >= 2, f"T {T}, top-k 5, top-p 0.9: the first piece over 4,000 seeds against sampler_chances: p {p:.2g} ({df} df)")
        if pp:
            f = max(firsts, key=firsts.get)
            h1 = lm(input_ids=torch.tensor([ids + [f]]), use_cache=False).last_hidden_state[0, -1]
            l1 = ao._model_logits(lm, W_U, h1[None], 30.0)[0]
            pen, plain = sampler_chances(l1, st, [f]), sampler_chances(l1, resolve_sampling(None, **{**smp, "presence_penalty": 0.0}), [f])
            check(float(pen[f]) > float(plain[f]) + 0.05, f"the presence penalty {pp} does not move the second piece ({float(plain[f]):.3f} -> {float(pen[f]):.3f})")
            p, df = chi_square_p(seconds[f], pen)
            check(p > 1e-3 and df >= 1, f"T {T}, presence penalty {pp}: the second piece after {f} ({sum(seconds[f].values())} rows) against "
                                        f"sampler_chances with it written: p {p:.2g} ({df} df)")
    # the stand-in's five likeliest are nearly level, which tells temperatures apart poorly: the race itself, then, on
    # sharper logits (randn x 3 over 1,000 pieces, softcapped, as check_sampler_mirror draws them), 4,000 seeds at once,
    # and the test must tell a temperature 25% off
    sharp = torch.tanh(torch.randn(1000, generator=torch.Generator().manual_seed(5)) * 3.0 / 30.0) * 30.0
    for T in (0.7, 1.5):
        st = resolve_sampling(None, temperature=T, top_k=50, top_p=0.9, presence_penalty=0.0)
        nxt, at = ao._Race(st, 0, list(range(4000)), torch.device("cpu")).draw(sharp.expand(4000, -1), [[]] * 4000)
        counts: Dict[int, int] = {}
        for t in nxt.tolist():
            counts[t] = counts.get(t, 0) + 1
        ch = sampler_chances(sharp, st, [])
        p, df = chi_square_p(counts, ch)
        off, _ = chi_square_p(counts, sampler_chances(sharp, resolve_sampling(None, temperature=1.25 * T, top_k=50, top_p=0.9, presence_penalty=0.0), []))
        check(p > 1e-3 and df >= 10 and off < 1e-6 and torch.equal(at.flatten(), ch[nxt]),
              f"the race at T {T}, top-k 50, top-p 0.9 on sharper logits: p {p:.2g} ({df} df), against T {1.25 * T:g} p {off:.2g}")
    # the logits over a temperature of 1e-39 overflow and every chance is NaN: refused, where the race's guard would
    # score every piece 0 and take piece 0 (the pad mark); 1e-38 still draws on the stand-in's logits
    for bad, says in ((SamplerSettings(presence_penalty=0.5), "only with a temperature above 0"),
                      (SamplerSettings(temperature=0.7), "needs its seed"),
                      (SamplerSettings(temperature=1e-39, seed=1), "no chances to draw from")):
        try:
            steps([ids], None, n=2, sampler=bad)
            check(False, f"sampled write: {bad} was not refused")
        except ValueError as exc:
            check(says in str(exc), f"sampled write refused: {exc}")
    tiny = steps([ids], None, n=3, sampler=SamplerSettings(temperature=1e-38, seed=1))
    check(len(tiny) == 3 and all(s.sampler_prob is not None and 0 < s.sampler_prob <= 1 for s in tiny),
          f"sampled write at temperature 1e-38: drawn, each chance a number ({[(s.token_id, s.sampler_prob) for s in tiny]})")
    nan_row = torch.stack([sharp, torch.full_like(sharp, float("nan"))])
    warm = resolve_sampling(None, temperature=0.7, top_k=0, top_p=1.0, presence_penalty=0.0)
    _, at = ao._Race(warm, 0, [0, 0], torch.device("cpu")).draw(nan_row, [[], []])
    check(0 < float(at[0, 0]) <= 1 and torch.isnan(at[1, 0]).item(),
          f"the race: a row whose chances are NaN gets NaN for its chance, the other row its own ({at.flatten().tolist()})")
    try:
        steps([ids] * 2, None, n=2, sampler=hot, samples_of=[0])
        check(False, "sampled write: samples_of for 1 of 2 rows was not refused")
    except ValueError:
        pass
    section("sampled writing: over 4,000 seeds the first piece, and the second under a presence penalty, as sampler_chances "
            "says (chi-square), the race on sharper logits too; a penalty without a temperature, a draw without a seed, and a "
            "temperature with no chances to draw from, refused", before)


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


def check_wiring_use(hf: Any, tok: Any, f: torch.Tensor) -> None:
    import copy

    import numpy as np

    from atlas_stand_in import CORPUS

    before = len(failures)
    m64 = copy.deepcopy(hf).double()  # float64, so "exact to first order" can be held to a tight tolerance
    for mod in m64.modules():
        if type(mod).__name__ == "Gemma4RMSNorm":  # transformers computes its norms in float32 whatever the model's dtype
            mod.forward = lambda x, m=mod: m._norm(x) * m.weight if m.with_scale else m._norm(x)
    lm = language_model(m64)
    w = ao.Weights(lm)
    s = w.scalars.double()
    L, N = w.n_layers, w.d_ff
    wins = [ids_of(tok, t) for t in CORPUS]
    K = 4
    E = ao.passage_events(lm, wins, k=K)
    c_ = (2.0 / np.pi) ** 0.5
    gelu = lambda x: 0.5 * x * (1.0 + np.tanh(c_ * (x + 0.044715 * x**3)))  # noqa: E731
    # the operating points: per face, the median of gelu'(g) u and of gelu(g) over the stored events, as brute force
    # takes it one neuron at a time
    ops = {}
    worst = 0.0
    for face in ("pos", "neg"):
        al, be = ao.operating_points(E[f"{face}_g"], E[f"{face}_u"], E[f"{face}_win"] >= 0)
        ops[face] = (al, be)
        for l in range(L):
            for n in range(N):
                ok = E[f"{face}_win"][l, n] >= 0
                if not ok.any():
                    worst = max(worst, 0.0 if np.isnan(al[l, n]) and np.isnan(be[l, n]) else 1.0)
                    continue
                g, u = E[f"{face}_g"][l, n][ok].astype(np.float64), E[f"{face}_u"][l, n][ok].astype(np.float64)
                want = (np.median([ao.gelu_tanh_grad(x) for x in g] * u), np.median(gelu(g)))
                worst = max(worst, abs(float(al[l, n]) - want[0]) / max(1.0, abs(want[0])), abs(float(be[l, n]) - want[1]) / max(1.0, abs(want[1])))
    check(worst < 1e-5, f"operating points: the medians of gelu'(g) u and gelu(g) over each face's events ({worst:.1e} off)")

    def at(ids: List[int], p: int, i: int, j: int, b: int, A: int, add_x: Any = None, add_mid: Any = None, push_a: float = 0.0) -> Dict[str, Any]:
        """One forward of a window, read at word-piece p: layer i's block output (y) and the stream leaving it (h); the
        stream entering layer j's block (mid); reader b's gate, up and activation there. `add_x` is added to layer j's
        pre-feed-forward norm's output at p, `add_mid` to its input, `push_a` to writer A's activation."""
        got: Dict[str, Any] = {}
        Ls = lm.layers

        def push(_m: Any, a_: Tuple[Any, ...]) -> Any:
            if not push_a:
                return None
            x = a_[0].clone()
            x[0, p, A] += push_a
            return (x,)

        def norm_in(_m: Any, a_: Tuple[Any, ...]) -> Any:
            got["mid"] = a_[0][0, p].clone()
            if add_mid is None:
                return None
            x = a_[0].clone()
            x[0, p] += add_mid
            return (x,)

        def norm_out(_m: Any, _a: Any, o: torch.Tensor) -> Any:
            if add_x is None:
                return None
            o = o.clone()
            o[0, p] += add_x
            return o

        hs = [Ls[i].mlp.down_proj.register_forward_pre_hook(push),
              Ls[i].mlp.register_forward_hook(lambda _m, _a, o: got.__setitem__("y", o[0, p].clone())),
              Ls[i].register_forward_hook(lambda _m, _a, o: got.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0, p].clone())),
              Ls[j].pre_feedforward_layernorm.register_forward_pre_hook(norm_in),
              Ls[j].pre_feedforward_layernorm.register_forward_hook(norm_out),
              Ls[j].mlp.gate_proj.register_forward_hook(lambda _m, _a, o: got.__setitem__("g0", float(o[0, p, b]))),
              Ls[j].mlp.up_proj.register_forward_hook(lambda _m, _a, o: got.__setitem__("u0", float(o[0, p, b]))),
              Ls[j].mlp.down_proj.register_forward_pre_hook(lambda _m, a_: got.__setitem__("a", float(a_[0][0, p, b])))]
        try:
            lm(input_ids=torch.tensor([ids]), use_cache=False)
        finally:
            for h in hs:
                h.remove()
        return got

    held_bad, signs, stored_bad, free_bad, free_agree, events = [], [], 0, [], [], 0
    for (j, b) in ((3, 10), (3, 77), (4, 100), (5, 21), (5, 64)):
        links = ao.neuron_wiring(w, j, b, f, k=6, gu=True)["up"][:3]
        Wg, Wu = lm.layers[j].mlp.gate_proj.weight[b].double(), lm.layers[j].mlp.up_proj.weight[b].double()
        w_pre = lm.layers[j].pre_feedforward_layernorm.weight.double()
        e_pre = float(lm.layers[j].pre_feedforward_layernorm.eps)
        for face in ("pos", "neg"):
            for e in range(K):
                wi, p = int(E[f"{face}_win"][j, b, e]), int(E[f"{face}_at"][j, b, e])
                if wi < 0:
                    continue
                ids = wins[wi]
                for c in links:
                    i, A = c["layer"], c["n"]
                    base = at(ids, p, i, j, b, A)
                    g0, u0 = base["g0"], base["u0"]
                    events += 1
                    stored_bad += int(not (E[f"{face}_g"][j, b, e] == np.float16(np.float32(g0)) and E[f"{face}_u"][j, b, e] == np.float16(np.float32(u0))))
                    alpha, beta = ao.gelu_tanh_grad(g0) * u0, float(gelu(np.float64(g0)))
                    pred = alpha * c["g"] + beta * c["u"]
                    scale = max(abs(alpha * c["g"]), abs(beta * c["u"]), 1e-300)
                    col = lm.layers[i].mlp.down_proj.weight[:, A].double()
                    w_post = lm.layers[i].post_feedforward_layernorm.weight.double()
                    y, mid = base["y"], base["mid"]
                    r_i = float((y.pow(2).mean() + float(lm.layers[i].post_feedforward_layernorm.eps)).sqrt())
                    r_m = float((mid.pow(2).mean() + e_pre).sqrt())
                    fp = float(torch.prod(s[i:j])) / (r_i * r_m)
                    v = w_pre * w_post * col  # what the reader's two rows read of the writer, the norms' weights folded in
                    if abs(float(Wg @ v) - c["g"]) > 1e-9 * max(1.0, abs(c["g"])) or abs(float(Wu @ v) - c["u"]) > 1e-9 * max(1.0, abs(c["u"])):
                        held_bad.append((j, b, i, A, "g, u"))
                    # the direct path alone, both norms held at their values there: the writer's write w_post (.) col / r_i,
                    # carried by s_i ... s_(j-1), read through w_pre / r_m, added at the reader's own input
                    h = 1e-6 / max(fp * max(abs(c["g"]), abs(c["u"])), 1e-300)
                    ap, am = at(ids, p, i, j, b, A, add_x=h * fp * v)["a"], at(ids, p, i, j, b, A, add_x=-h * fp * v)["a"]
                    meas = (ap - am) / (2 * h * fp)
                    if abs(meas - pred) > 1e-6 * scale:
                        held_bad.append((j, b, face, e, i, A, meas, pred))
                    if abs(pred) > 1e-3 * scale:
                        signs.append((ap - base["a"] > 0) == (pred > 0) and (am - base["a"] < 0) == (pred > 0))
                    # the writer's own activation pushed, carried by the direct path alone, the norms free: the change is
                    # the full first-order one, both norms' own response (a projection off what each reads) included
                    hp = at(ids, p, i, j, b, A, push_a=1e-6)["h"]
                    hm = at(ids, p, i, j, b, A, push_a=-1e-6)["h"]
                    on = float(torch.prod(s[i + 1:j]))
                    ap2 = at(ids, p, i, j, b, A, add_mid=on * (hp - base["h"]))["a"]
                    am2 = at(ids, p, i, j, b, A, add_mid=on * (hm - base["h"]))["a"]
                    D = y.numel()
                    dN = w_post * (col - y * float(y @ col) / (D * r_i * r_i)) / r_i
                    dmid = float(torch.prod(s[i:j])) * dN
                    dx = w_pre * (dmid - mid * float(mid @ dmid) / (D * r_m * r_m)) / r_m
                    pf = alpha * float(Wg @ dx) + beta * float(Wu @ dx)
                    if abs((ap2 - am2) / 2e-6 - pf) > 1e-5 * max(abs(alpha * float(Wg @ dx)), abs(beta * float(Wu @ dx)), 1e-300):
                        free_bad.append((j, b, face, e, i, A, (ap2 - am2) / 2e-6, pf))
                    if abs(pred) > 1e-3 * scale:
                        free_agree.append((pf > 0) == (pred > 0))
    check(events > 20 and stored_bad == 0, f"in use: {events} stored events read again, each its stored gate and up values ({stored_bad} differ)")
    check(not held_bad, f"in use, the norms held: the change is f (alpha g + beta u) eps to first order ({len(held_bad)} off: {held_bad[:2]})")
    check(bool(signs) and all(signs), f"in use, the norms held: +eps and -eps move the reader with the sign of alpha g + beta u ({signs.count(False)} of {len(signs)} not)")
    check(not free_bad, f"the writer's own push, the norms free: the full first-order change ({len(free_bad)} off: {free_bad[:2]})")
    print(f"        the norms free, the change's sign is that of alpha g + beta u at {sum(free_agree)} of {len(free_agree)} events "
          f"({100 * sum(free_agree) / max(1, len(free_agree)):.0f}%; the norms' own response is what the link model leaves out)")
    # the median over a face's events against each event's own: every reader of layers 3 to 5, its six strongest writers
    agree = total = framed = 0
    for j in (3, 4, 5):
        for b in range(N):
            up = ao.neuron_wiring(w, j, b, f, k=6, gu=True)["up"]
            for face, sg in (("pos", "+"), ("neg", "-")):
                al, be = float(ops[face][0][j, b]), float(ops[face][1][j, b])
                if not np.isfinite(al):
                    continue
                for e in range(K):
                    if int(E[f"{face}_win"][j, b, e]) < 0:
                        continue
                    g0, u0 = float(E[f"{face}_g"][j, b, e]), float(E[f"{face}_u"][j, b, e])
                    a_e, b_e = ao.gelu_tanh_grad(g0) * u0, float(gelu(np.float64(g0)))
                    for c in up:
                        mine = ao.use_sign(al, be, c["g"], c["u"], c["gate"])
                        framed += int(mine != c["gate"] * (1 if al * c["g"] + be * c["u"] > 0 else -1))
                        agree += int(mine == ao.use_sign(a_e, b_e, c["g"], c["u"], c["gate"]))
                        total += 1
    check(total > 1000 and framed == 0 and ao.use_sign(float("nan"), 0.0, 1.0, 1.0, 1) is None,
          f"use_sign is gate x sign(alpha g + beta u), compared at {total} (link, event) pairs ({framed} not); none without an operating point")
    print(f"        the median-based sign in use is each event's own at {agree} of {total} (link, event) pairs ({100 * agree / max(1, total):.0f}%)")
    section("the sign in use: operating points as medians, exactly f (alpha g + beta u) to first order at stored events, the norms "
            "held; the full first-order change with them free", before)


def check_wiring_measured(lm: Any, w: ao.Weights, tok: Any, f: torch.Tensor) -> None:
    import numpy as np

    before = len(failures)
    prompts = [ids_of(tok, "Who are you?", chat=True), ids_of(tok, "Describe the color orange.", chat=True)]
    writers = [(1, 3), (2, 40), (1, 7), (3, 90), (2, 11)]
    readers = [[(2, 5), (4, 17), (5, 100)], [(3, 1), (5, 2)], [(2, 5), (3, 64), (4, 0), (5, 127)], [(4, 9)], [(5, 77), (3, 3)]]
    lengths = [0.5, 1.2, 0.3, 2.0, 0.8]

    def alone(ids: List[int], l: int, a: int, d: float, rs: List[Tuple[int, int]]) -> List[Tuple[float, float, Any, Any]]:
        """One row by itself: the writer pushed by d (0: unpushed) at every word-piece but the start mark; each
        reader's activation at the last word-piece and its mean past the start mark, and its gate and up values (plain
        hooks on gate_proj and up_proj) at every word-piece past the start mark."""
        iv = None
        if d:
            iv = ao.Interventions(rows=1, adds={l: d * w.write_dirs(l, [a])})
        at = sorted({j for j, _ in rs})
        mlps = [lm.layers[j].mlp for j in at]
        got = observe(lm, ids, iv, [(m.down_proj, "pre") for m in mlps] + [(m.gate_proj, "post") for m in mlps] + [(m.up_proj, "post") for m in mlps])
        k = len(at)
        by, gs, us = ({j: calls[0][0] for j, calls in zip(at, got[s : s + k])} for s in (0, k, 2 * k))
        return [(float(by[j][-1, b]), float(by[j][1:, b].mean()), gs[j][1:, b].numpy(), us[j][1:, b].numpy()) for j, b in rs]

    # gelu's minimum, where the check's regimes part (a shut reader's gate path turns below it): the model's own gelu
    # (F.gelu, tanh) is least there on a fine grid, and its gradient is 0
    xs = torch.linspace(-3.0, 1.0, 400001, dtype=torch.float64)
    least = float(xs[F.gelu(xs, approximate="tanh").argmin()])
    check(abs(ao.GELU_MIN - least) < 2e-5 and abs(ao.gelu_tanh_grad(ao.GELU_MIN)) < 1e-12 and -0.76 < ao.GELU_MIN < -0.75,
          f"GELU_MIN is where gelu_tanh is least ({ao.GELU_MIN:.5f}; on the grid {least:.5f})")
    worst, worst_gu, gelu_bad, pad_bad = 0.0, 0.0, 0.0, 0
    P = max(len(ids) for ids in prompts) - 1
    for batch in (5, 3, 64):  # two writers a prefill (at two layers in one), one, and all of them
        got, g0, u0 = ao.pushed_reads(lm, w, prompts, writers, readers, lengths, bos_id=tok.bos_token_id, batch=batch)
        check(got.shape == (5, 4, 2, 3, 2) and bool(np.isnan(got[3, 1:]).all()) and g0.shape == u0.shape == (5, 4, 2, P)
              and bool(np.isnan(g0[3, 1:]).all() and np.isnan(u0[3, 1:]).all()),
              f"pushed reads: shapes {got.shape}, {g0.shape}, NaN past a writer's own readers")
        for a, ((l, n), rs) in enumerate(zip(writers, readers)):
            for q, ids in enumerate(prompts):
                for row, d in enumerate((0.0, lengths[a], -lengths[a])):
                    want = alone(ids, l, n, d, rs)
                    for r, (last, mean, gg, uu) in enumerate(want):
                        worst = max(worst, abs(float(got[a, r, q, row, 0]) - last), abs(float(got[a, r, q, row, 1]) - mean))
                        if row == 0:
                            # where the reader sits: its own gate and up values on the unpushed row, right-aligned
                            worst_gu = max(worst_gu, float(np.abs(g0[a, r, q, P - len(gg):] - gg).max()), float(np.abs(u0[a, r, q, P - len(uu):] - uu).max()))
                            pad_bad += int(not (np.isnan(g0[a, r, q, : P - len(gg)]).all() and np.isnan(u0[a, r, q, : P - len(uu)]).all()))
                            at_last = F.gelu(torch.tensor(g0[a, r, q, -1]), approximate="tanh") * torch.tensor(u0[a, r, q, -1])
                            gelu_bad = max(gelu_bad, abs(float(at_last) - float(got[a, r, q, 0, 0])))
    check(worst < 1e-5, f"pushed reads: batched rows equal separate forwards, unpushed, +d and -d, last and mean ({worst:.1e} off)")
    check(worst_gu < 1e-5 and pad_bad == 0, f"pushed reads: each reader's g0 and u0 are a plain hook's on gate_proj and up_proj, unpushed, "
          f"at every word-piece past the start mark, the last at [..., -1], NaN before a shorter question's first ({worst_gu:.1e} off; {pad_bad} pads not NaN)")
    check(gelu_bad < 1e-6, f"pushed reads: the unpushed read at the last word-piece is gelu_tanh(g0) u0 there ({gelu_bad:.1e} off)")
    moved, _, _ = ao.pushed_reads(lm, w, prompts[:1], writers[:1], readers[:1], [0.5], bos_id=tok.bos_token_id)
    check(bool(np.isfinite(moved).all()) and bool(np.abs(moved[0, :, 0, 1] - moved[0, :, 0, 0]).min() > 0),
          "pushed reads: a push moves every reader it reaches")
    # where the reader sits, a small push (a tenth of the lengths above) moves its activation at the last word-piece by
    # alpha0 dg + beta0 du to first order, float32 central differences: alpha0 = gelu'(g0) u0 and beta0 = gelu(g0) from
    # pushed_reads' own g0 and u0 (atlas_ops.site_coefficients, as the summary reads them), dg and du the reader's own
    # gate and up changes there under +-d
    small = [0.1 * x for x in lengths]
    got, g0, u0 = ao.pushed_reads(lm, w, prompts, writers, readers, small, bos_id=tok.bos_token_id)
    fd_bad, fd_n, by_weights = [], 0, []
    for a, ((l, n), rs) in enumerate(zip(writers, readers)):
        g_w, u_w = ao.pair_products(w, l, [n], [rs])
        for q, ids in enumerate(prompts):
            up, down = alone(ids, l, n, small[a], rs), alone(ids, l, n, -small[a], rs)
            for r in range(len(rs)):
                al0, be0 = (float(x) for x in ao.site_coefficients(g0[a, r, q, -1], u0[a, r, q, -1]))
                pred = al0 * (float(up[r][2][-1]) - float(down[r][2][-1])) / 2 + be0 * (float(up[r][3][-1]) - float(down[r][3][-1])) / 2
                da = (float(got[a, r, q, 1, 0]) - float(got[a, r, q, 2, 0])) / 2
                fd_n += 1
                if abs(da - pred) > 1e-2 * abs(da) or (da > 0) != (pred > 0):
                    fd_bad.append((a, r, q, da, pred))
                by_weights.append((da > 0) == (al0 * float(g_w[0, r]) + be0 * float(u_w[0, r]) > 0))
    check(fd_n > 20 and not fd_bad, f"where the reader sits: a small push moves it by alpha0 dg + beta0 du, alpha0 and beta0 from its "
          f"recorded g0 and u0, at {fd_n} (writer, reader, question) ({len(fd_bad)} off: {fd_bad[:2]})")
    print(f"        with the link's own g and u from the weights in place of dg and du (the direct path alone), its sign is the "
          f"change's at {sum(by_weights)} of {len(by_weights)} (on random weights every other path counts too; printed, not asserted)")
    # the null: never a linked reader, one at each linked reader's layer, none twice
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(400):
        L_, width = 6, 12
        linked = list({(int(rng.integers(1, L_)), int(rng.integers(width))) for _ in range(int(rng.integers(1, 9)))})
        nl = ao.null_readers(linked, width, rng)
        bad += int(not (len(nl) == len(set(nl)) == len(linked) and not set(nl) & set(linked)
                        and sorted(j for j, _ in nl) == sorted(j for j, _ in linked)))
    check(bad == 0, f"the null: never a linked reader, at the linked readers' layers, none twice ({bad} of 400 draws not)")
    try:
        ao.null_readers([(2, b) for b in range(4)], 4, rng)
        check(False, "the null: a layer with no unlinked neuron left was not refused")
    except ValueError as exc:
        check("no neuron left" in str(exc), f"the null: refusal {exc}")
    # g and u of any pair, as neuron_wiring reads a link
    res = ao.neuron_wiring(w, 1, 12, f, k=10, gu=True)["down"]
    g, u = ao.pair_products(w, 1, [12], [[(c["layer"], c["n"]) for c in res]])
    check(all(abs(float(g[0, i]) - c["g"]) <= 1e-5 * max(1.0, abs(c["g"])) and abs(float(u[0, i]) - c["u"]) <= 1e-5 * max(1.0, abs(c["u"]))
              for i, c in enumerate(res)), "pair_products: neuron_wiring's g and u, pair by pair")
    section("the wiring measured: batched pushed reads equal separate forwards, each reader's g0 and u0 a plain hook's, a small "
            "push the first-order change where it sits; the null never linked; any pair's g and u", before)


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
    check_sampling(lm, W_U, w, tok, stop)
    check_reading(lm, W_U, w, tok)
    check_heads(hf, tok, w)
    f = check_wiring(hf, lm, w, tok)
    check_steer_lens(lm, W_U, w, tok)
    check_wiring_use(hf, tok, f)
    check_wiring_measured(lm, w, tok, f)
    # the runtime's own weight accessors give the same wiring as the plain modules
    before = len(failures)
    wr = ao.Weights(lm, rt)
    check(ao.neuron_wiring(wr, 2, 31, f, k=8) == ao.neuron_wiring(w, 2, 31, f, k=8), "wiring through the runtime differs")
    section("the runtime's weight accessors give the same wiring", before)
    print(f"\n{passed} checks passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

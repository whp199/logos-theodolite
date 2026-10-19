#!/usr/bin/env python3
"""A CPU-only check of the sampler mirror (`backend.engine_runtime.sampler_draw_distribution` / `sampler_chances` /
`step_numbers`): the odds the instrument reports for each word-piece it writes. No model, no GPU.

interp_engine draws each word-piece with `steer._sample_next` (private), which returns only the id. The instrument
replays its ops on the same logits to say what chance the sampler gave the piece it drew. For many random logit
vectors (bfloat16 and float32, with and without exact ties) and sampler settings (greedy; temperature 0.05-2; top-k
off, 1, a few, 64, the whole vocabulary; top-p off down to 0.001; presence penalties of either sign over the pieces
already written), after the same `torch.manual_seed`:
1. `torch.multinomial` on the mirrored distribution picks exactly what `_sample_next` picks (greedy: the argmax of
   the penalised logits);
2. the mirrored chance of that piece is above 0, the chances sum to 1 and none is negative, and the support counts
   the pieces with a chance;
3. the model's own numbers (`step_numbers`): prob = softmax of the raw logits, rank = 1 + the pieces above it, the
   exact ties counted, the top five in order, logprob = log prob.

Run: .venv/bin/python -I scripts/check_sampler_mirror.py [--trials N]   (exits 0 when every check holds)
"""

from __future__ import annotations

import argparse
import os
import sys

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import math
import random
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from interp_engine.sampling import apply_presence_penalty, resolve_sampling
from interp_engine.steer import _sample_next

from backend.engine_runtime import sampler_chances, sampler_draw_distribution, step_numbers

failures: List[str] = []
passed = 0


def check(cond: bool, what: str) -> None:
    global passed
    if cond:
        passed += 1
    else:
        failures.append(what)
        if len(failures) <= 20:
            print(f"  FAILED: {what}")


def random_logits(rng: random.Random, vocab: int, dtype: torch.dtype) -> torch.Tensor:
    scale = rng.choice([0.5, 2.0, 5.0, 12.0])
    logits = torch.randn(vocab) * scale
    if rng.random() < 0.3:
        # exact ties: round onto a coarse grid
        logits = torch.round(logits * 2) / 2
    if rng.random() < 0.2:
        # a few near-certain pieces
        hot = torch.randint(0, vocab, (3,))
        logits[hot] += 25.0
    # the writing's logits are softcapped at 30 (bf16 on the GPU)
    logits = torch.tanh(logits / 30.0) * 30.0
    return logits.to(dtype)


def one_trial(rng: random.Random, trial: int) -> None:
    vocab = rng.choice([7, 50, 1000, 5000, 262144 if trial % 50 == 0 else 2000])
    dtype = rng.choice([torch.bfloat16, torch.float32])
    logits = random_logits(rng, vocab, dtype)
    temperature = rng.choice([0.0, 0.0, 0.05, 0.3, 0.7, 1.0, 1.0, 1.5, 2.0])
    top_k = rng.choice([0, 0, 1, 3, min(64, vocab), vocab])  # the server allows 0 .. the vocabulary
    top_p = rng.choice([1.0, 1.0, 0.95, 0.9, 0.5, 0.1, 0.001])
    penalty = rng.choice([0.0, 0.0, 0.5, 1.5, 2.0, -1.0])
    generated = [rng.randrange(vocab) for _ in range(rng.choice([0, 1, 5, 30]))]
    # the instrument resolves its explicit knobs the way EagerModel.sampling_settings does
    settings = resolve_sampling(None, temperature=temperature, top_k=top_k, top_p=top_p, presence_penalty=penalty)
    what = f"trial {trial} ({vocab}, {dtype}, T={temperature}, k={top_k}, p={top_p}, pen={penalty}, gen={len(generated)})"

    seed = rng.randrange(2**31)
    torch.manual_seed(seed)
    picked = _sample_next(
        logits.clone(),
        temperature=settings.temperature,
        top_k=settings.top_k,
        top_p=settings.top_p,
        presence_penalty=settings.presence_penalty,
        generated=generated,
    )
    torch.manual_seed(seed)
    drawn = sampler_draw_distribution(logits.clone(), settings, generated)
    if drawn is None:
        check(settings.temperature <= 0, f"{what}: no distribution while sampling")
        mirrored = int(apply_presence_penalty(logits, generated, settings.presence_penalty).argmax().item())
    else:
        probs, order = drawn
        choice = torch.multinomial(probs, 1)
        mirrored = int(order[choice].item()) if order is not None else int(choice.item())
    check(mirrored == picked, f"{what}: the mirror drew {mirrored}, the sampler {picked}")

    q = sampler_chances(logits, settings, generated)
    nums = step_numbers(logits, picked, settings, generated)
    if settings.temperature <= 0:
        check(q is None and nums["sampler_prob"] is None and nums["sampler_support"] is None, f"{what}: greedy has odds")
    else:
        assert q is not None
        qf = q.float()
        check(float(qf[picked]) > 0.0, f"{what}: the drawn piece has no chance")
        check(nums["sampler_prob"] == float(q[picked].item()) and nums["sampler_prob"] > 0.0, f"{what}: sampler_prob")
        tol = 0.02 if dtype == torch.bfloat16 else 1e-4
        check(abs(float(qf.sum()) - 1.0) < tol and bool((qf >= 0).all()), f"{what}: chances sum to {float(qf.sum())}")
        check(nums["sampler_support"] == int((q > 0).sum()), f"{what}: support")
        if settings.top_k:
            # top-k keeps the k largest of the penalised logits / T (in their dtype, where rounding can tie more of
            # them with the k-th) and nothing else
            scaled = apply_presence_penalty(logits, generated, settings.presence_penalty) / settings.temperature
            kth = torch.topk(scaled, settings.top_k).values[-1]
            check(int((q > 0).sum()) <= int((scaled >= kth).sum()), f"{what}: support beyond top-k")

    lf = logits.float()
    ref = torch.softmax(lf, dim=-1)
    c = picked
    check(math.isclose(nums["model_prob"], float(ref[c]), rel_tol=1e-6, abs_tol=1e-12), f"{what}: model_prob")
    check(math.isclose(nums["model_logprob"], float(torch.log_softmax(lf, -1)[c]), rel_tol=1e-6, abs_tol=1e-6),
          f"{what}: model_logprob")
    check(nums["model_rank"] == 1 + sum(1 for v in lf.tolist() if v > float(lf[c])), f"{what}: model_rank")
    check(nums["n_tied"] == sum(1 for v in lf.tolist() if v == float(lf[c])) - 1, f"{what}: n_tied")
    check(nums["chosen_logit"] == float(lf[c]), f"{what}: chosen_logit")
    top = nums["top"]
    check(len(top) == min(5, vocab) and all(top[i][1] >= top[i + 1][1] for i in range(len(top) - 1)), f"{what}: top")
    if settings.temperature <= 0 and not penalty:
        # greedy without a penalty: the piece written is the model's likeliest (rank 1)
        check(nums["model_rank"] == 1, f"{what}: greedy wrote a piece of rank {nums['model_rank']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=600, help="random logit vectors and settings to try (default 600)")
    ap.add_argument("--seed", type=int, default=20261018, help="seed for the trials themselves")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    print(f"sampler mirror: {args.trials} trials on the CPU")
    for trial in range(args.trials):
        one_trial(rng, trial)
    if failures:
        print(f"\nsampler mirror: {len(failures)} of {passed + len(failures)} checks FAILED")
        return 1
    print(f"\nsampler mirror: all {passed} checks hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())

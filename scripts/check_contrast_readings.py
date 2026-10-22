#!/usr/bin/env python3
"""A CPU-only check of the contrast readings (one word against another): no model, no GPU, no running server; only
the tokenizer in the local Hugging Face cache, E3's average state in experiments/results, and small random weights.

A sweep's `contrasts` read, at every layer, the mean over the pairs of the capped logit of toward[i] less that of
away[i] (backend/theodolite_math.py `contrast_of_state`), through the lens (h_l J_l^T) and by the naked eye (h_l), and
against the backgrounds asked for (backend/engine_runtime.py `prepare_sweep`, `_answer_centre`,
`_contrast_readings`). This checks:
1. the score: against the full softcapped unembed (`unembed_gemma4`), including a tiny state below the norm's floor,
   and against the study's own code, an independent labkit (experiments/lab/labkit.py) over the cached tokenizer:
   `gender_scores` "she_he" for ' she' against ' he', "pron" for the six pronoun pairs, `gender_direction`;
2. the study's average state: the linear identity of E3 (c) (E3_lens_validation_09_background.py),
   <hJ, d>/rms(hJ) = <muJ, d>/rms(hJ) + <(h - mu)J, d>/rms(hJ), and `contrast_mean_part` as E3 computed its
   "meanpart"; the centred score as E3's gender_scores(lens(h - mu));
3. words read as the landmarks read theirs (`resolve_concept_token_id`: " word" first): 'she' is ' she' (1304), not
   'she' (12053); 'herself' is ' herself' (13442); a word that is no single word-piece in any form, a blank word, or
   words that do not pair keep the contrast with an error and the pieces, and are not read;
4. a sweep's readings, on a stand-in runtime (60 random lenses, real tokenizer, deterministic stand-in states): the
   lens and naked-eye series; "controls" at token 0 (the controls' own last piece), past it with centre_answer (the
   controls reading this very answer) and without it (none, with the reason), and with no control sent; the
   differences against the controls as an independent full-unembed computation gives them; "study_mean" as E3
   centred; a contrast that cannot be read leaves the others' controls in step; the captures are those the centring
   makes (no extra forward), and a sweep with no contrasts reads none and is centred exactly as before;
5. an identity lens gives the lens series equal to the naked eye's, raw, centred and mean part;
6. E3's average state as committed: experiments/results/E3_background_mu.npy is the file pinned
   (`STUDY_MEAN_SHA256`), its sidecar says the same and names its sources, .gitignore lets it in, and a missing or
   changed file is refused in plain words.

Run: .venv/bin/python -I scripts/check_contrast_readings.py   (exits 0 when every check holds)
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import asyncio
import hashlib
import importlib.util
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from interp_engine import Address

import backend.engine_runtime as er
from backend.engine_runtime import (
    DEFAULT_W4A16_MODEL_ID,
    STUDY_MEAN_PATH,
    STUDY_MEAN_SHA256,
    SweepCapture,
    TheodoliteRuntime,
    read_study_mean,
    study_mean_problem,
)
from backend.theodolite_math import (
    CONTRAST_BACKGROUNDS,
    N_LAYERS,
    contrast_direction,
    contrast_mean_part,
    contrast_of_state,
    unembed_gemma4,
)

failures: List[str] = []
passed = 0


def check(cond: bool, what: str) -> None:
    global passed
    if cond:
        passed += 1
    else:
        failures.append(what)
        print(f"  FAILED: {what}")


def close(a: Any, b: Any, tol: float) -> bool:
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    return a.shape == b.shape and bool(torch.all((a - b).abs() <= tol * (1.0 + b.abs())))


V = 262144  # the vocabulary: the stand-in unembedding keeps every real id
D = 16  # the stand-in's model width
SHE, HE, SHE_BARE, HERSELF = 1304, 668, 12053, 13442  # measured on the cached tokenizer


def load_tokenizer() -> Any:
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(DEFAULT_W4A16_MODEL_ID, local_files_only=True)
    except Exception as exc:
        failures.append(f"the instrument's tokenizer is not in the local Hugging Face cache ({exc})")
        print(f"  FAILED: {failures[-1]}")
        return None


def load_labkit() -> Any:
    """experiments/lab/labkit.py as the study ran it (its Lab is built below without a model)."""
    spec = importlib.util.spec_from_file_location("labkit", ROOT_DIR / "experiments" / "lab" / "labkit.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ref_score(h: torch.Tensor, toward: Sequence[int], away: Sequence[int], fnw: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """The contrast from the full softcapped unembed: mean_i logit[toward_i] - logit[away_i]."""
    lg = unembed_gemma4(h.float(), fnw, W)
    return (lg[..., list(toward)] - lg[..., list(away)]).mean(-1)


# --------------------------------------------------------------------------------------------------------------------
# 1-2. the score, labkit, the study's average state
# --------------------------------------------------------------------------------------------------------------------


def check_score(tok: Any) -> None:
    print("1. the score, against the unembed and the study's labkit")
    g = torch.Generator().manual_seed(3)
    W = torch.randn(V, D, generator=g)
    fnw = torch.rand(D, generator=g) + 0.5
    h = torch.randn(7, D, generator=g) * 40.0
    tiny = torch.randn(2, D, generator=g) * 1e-3  # mean x^2 below 1e-3: the 1e-12 floor
    for toward, away, what in (([SHE], [HE], "one pair"), ([5, 9, 11], [6, 10, 12], "three pairs")):
        for x, xs in ((h, "states"), (tiny, "tiny states")):
            got = contrast_of_state(x, toward, away, fnw, W)
            check(got.dtype == torch.float32 and got.shape == x.shape[:-1], f"{what}, {xs}: {got.dtype} {tuple(got.shape)}")
            check(close(got, ref_score(x, toward, away, fnw, W), 1e-4), f"{what}, {xs}: the full unembed's difference")
    raw = contrast_of_state(h, [SHE], [HE], fnw, W, softcap=None)
    capped = contrast_of_state(h, [SHE], [HE], fnw, W)
    check(not torch.allclose(raw, capped) and bool(torch.all(capped.abs() <= 60.0)), "softcap None leaves it uncapped")
    for bad in (([], []), ([1, 2], [3])):
        try:
            contrast_of_state(h, bad[0], bad[1], fnw, W)
            check(False, f"words that do not pair {bad} were read")
        except ValueError as exc:
            check("will not pair" in str(exc), f"pairs refused: {exc}")
    if tok is None:
        return
    labkit = load_labkit()
    lab = object.__new__(labkit.Lab)
    lab.tok, lab.WU, lab.fnw, lab.dev = tok, W, fnw, torch.device("cpu")
    lab.gender = lab._build_gender_ids()
    gs = lab.gender_scores(h)
    she, he = lab.tid(" she"), lab.tid(" he")
    check((she, he) == (SHE, HE), f"labkit's ' she' and ' he' are {she}, {he}")
    check(close(contrast_of_state(h, [she], [he], fnw, W), gs["she_he"], 1e-5), "labkit's she_he")
    pf, pm = lab.gender["pron_f"], lab.gender["pron_m"]
    check(close(contrast_of_state(h, pf, pm, fnw, W), gs["pron"], 1e-5), "labkit's pron, the six pronoun pairs")
    check(close(contrast_of_state(tiny, pf, pm, fnw, W), lab.gender_scores(tiny)["pron"], 1e-5), "labkit's pron, tiny")
    check(close(contrast_direction(pf, pm, fnw, W), lab.gender_direction("pron"), 1e-6), "labkit's gender_direction")

    print("2. the study's average state: E3's identity and its mean part")
    J = torch.randn(D, D, generator=g) / D**0.5
    mu = torch.randn(D, generator=g) * 30.0
    hJ, muJ, cJ = h @ J.T, mu @ J.T, (h - mu) @ J.T
    d = lab.gender_direction("pron")
    rms = hJ.pow(2).mean(-1).sqrt()
    mean_part = contrast_mean_part(muJ, hJ, pf, pm, fnw, W)
    check(close(mean_part, (muJ.unsqueeze(0) @ d) / rms, 1e-5), "the mean part as E3 computed its meanpart")
    check(close((hJ @ d) / rms, mean_part + (cJ @ d) / rms, 1e-4), "lin = mean part + the deviation's part")
    check(close(contrast_of_state(cJ, pf, pm, fnw, W), lab.gender_scores(cJ)["pron"], 1e-5),
          "the centred score as E3's gender_scores(lens(h - mu))")


# --------------------------------------------------------------------------------------------------------------------
# 3. words
# --------------------------------------------------------------------------------------------------------------------


def stand_in_runtime(tok: Any, *, identity: bool = False, seed: int = 5) -> TheodoliteRuntime:
    """A runtime with the real tokenizer and small random weights: 60 lenses (fp32 identities with `identity`),
    and a stand-in capture whose state at a position depends only on the word-pieces read up to it."""
    g = torch.Generator().manual_seed(seed)
    rt = TheodoliteRuntime(device="cpu", load_jlens=False)
    rt.tokenizer = tok
    rt.eager_model = object()  # type: ignore[assignment]  # the asserts' "loaded"; nothing runs on it
    rt._final_norm_weight = torch.rand(D, generator=g) + 0.5
    rt._lm_head_weight = torch.randn(V, D, generator=g)
    eye = torch.eye(D)
    rt.jpp_lenses = {
        l: eye if identity or l == N_LAYERS - 1 else (torch.randn(D, D, generator=g) / D**0.5).to(torch.bfloat16)
        for l in range(N_LAYERS)
    }
    rt.captures = []  # type: ignore[attr-defined]

    def capture_rows(token_ids: Sequence[int], *, every: Sequence[Address] = (), read: Sequence[Address] = (),
                     n_read: int = 1) -> Dict[Address, torch.Tensor]:
        ids = [int(t) for t in token_ids]
        rt.captures.append((list(ids), int(n_read)))  # type: ignore[attr-defined]
        return {a: torch.stack([state(ids[: p + 1], a.layer) for p in range(len(ids) - n_read, len(ids))])
                for a in read}

    rt.capture_rows = capture_rows  # type: ignore[assignment]
    return rt


def state(prefix: Sequence[int], layer: int) -> torch.Tensor:
    """The stand-in's resid_post leaving `layer` at a position that has read `prefix`: bf16, as captured."""
    g = torch.Generator().manual_seed(hash((tuple(int(t) for t in prefix), int(layer))) % (2**31))
    return (torch.randn(D, generator=g) * 30.0).to(torch.bfloat16)


def check_words(tok: Any) -> None:
    print("3. words read as the landmarks read theirs")
    rt = stand_in_runtime(tok)
    for word, want_id, want_piece in (("she", SHE, " she"), (" she", SHE, " she"), ("he", HE, " he"),
                                      ("herself", HERSELF, " herself")):
        r = rt.resolve_contrast_word(word)
        check(r == {"word": word, "id": want_id, "piece": want_piece}, f"{word!r}: {r}")
        check(rt.resolve_concept_token_id(word)[0] == want_id, f"{word!r}: the landmarks read the same id")
    check(tok.encode("she", add_special_tokens=False) == [SHE_BARE], "'she' alone is another word-piece (12053)")
    long_word = "antidisestablishmentarianism"
    r = rt.resolve_contrast_word(long_word)
    check(r["id"] is None and r["piece"] is None and set(r["pieces"]) == {f" {long_word}", long_word}
          and all(len(p) > 1 for p in r["pieces"].values())
          and "".join(r["pieces"][long_word]) == long_word, f"a word of several pieces: {r}")
    check(rt.resolve_concept_token_id(long_word)[0] == tok.encode(long_word, add_special_tokens=False)[0],
          "the landmarks would take its first piece: the contrast does not")
    specs = rt.resolve_contrasts([
        {"toward": ["she"], "away": ["he"], "label": "pronoun"},
        {"toward": ["she", long_word], "away": ["he", "him"]},
        {"toward": ["she", "her"], "away": ["he"]},
        {"toward": [" "], "away": ["he"]},
    ])
    ok = specs[0]
    check(ok["error"] is None and ok["toward_ids"] == [SHE] and ok["away_ids"] == [HE] and ok["label"] == "pronoun"
          and ok["toward"] == [{"word": "she", "id": SHE, "piece": " she"}] and "pieces" not in ok, f"a contrast: {ok}")
    bad = specs[1]
    check(bad["toward_ids"] is None and bad["away_ids"] is None and f'"{long_word}" is not one word-piece' in bad["error"]
          and bad["error"].startswith("Not read:") and set(bad["pieces"]) == {long_word}, f"its error: {bad['error']}")
    check("as many of each" in (specs[2]["error"] or "") and specs[2]["toward_ids"] is None, f"unpaired: {specs[2]}")
    check(specs[3]["error"] is not None and "it is blank" in specs[3]["error"], f"a blank word: {specs[3]['error']}")


# --------------------------------------------------------------------------------------------------------------------
# 4-5. a sweep's readings on the stand-in
# --------------------------------------------------------------------------------------------------------------------


def fake_cap(prompt_ids: List[int], answer: List[int]) -> SweepCapture:
    return SweepCapture(
        prompt="p", prompt_token_ids=prompt_ids, prompt_token_strs=[], answer_token_ids=answer, answer_token_strs=[],
        answer_text="", steps=[], caps={}, settings={}, greedy=True, seed=None, seed_drawn=False, stop_reason="length",
        stop_token_ids=[], answer_source="written", max_new_tokens=len(answer), generate_ms=0.0, capture_ms=0.0,
    )


def sighting(rt: TheodoliteRuntime, prefix: List[int]) -> Dict[str, torch.Tensor]:
    """A sighting's states as analyse_position holds them: h_l, and h_l J_l^T as the causal split computes it."""
    h = torch.stack([state(prefix, l) for l in range(N_LAYERS)])
    hJ = torch.stack([h[l].to(rt.jpp_lenses[l].dtype) @ rt.jpp_lenses[l].T for l in range(N_LAYERS)])
    return {"raw": h, "J": hJ}


def lensed(rt: TheodoliteRuntime, h: torch.Tensor) -> torch.Tensor:
    return torch.stack([h[l].to(rt.jpp_lenses[l].dtype) @ rt.jpp_lenses[l].T for l in range(N_LAYERS)])


async def prepare(rt: TheodoliteRuntime, cap: SweepCapture, **kw: Any) -> Any:
    frame = (torch.eye(3, D), "J", {"squaring": "gram_schmidt", "per_layer": False, "fnw_weighted": False},
             ("a", "b", "c"))
    return await rt.prepare_sweep(cap, monument_concepts=("spider", "eight", "insect"), frame_override=frame,
                                  tracked_concepts=(), **kw)


def check_sweep(tok: Any) -> None:
    print("4. a sweep's readings: the lens, the naked eye, the controls, the study's average state")
    real_d, real_read = er.D_MODEL, er.read_study_mean
    er.D_MODEL = D  # the stand-in's width, for the centre the controls are summed into
    g = torch.Generator().manual_seed(9)
    mu_small = (torch.randn(N_LAYERS, D, generator=g) * 25.0).numpy()
    er.read_study_mean = lambda *a, **k: mu_small  # type: ignore[assignment]
    try:
        asyncio.run(_check_sweep(tok, mu_small))
        print("5. an identity lens: the lens reads as the naked eye")
        asyncio.run(_check_identity(tok, mu_small))
    finally:
        er.D_MODEL, er.read_study_mean = real_d, real_read


async def _check_sweep(tok: Any, mu_np: np.ndarray) -> None:
    rt = stand_in_runtime(tok)
    fnw, W = rt.final_norm_weight, rt.lm_head_weight
    prompt, answer = [2, 818, 1590, 563], [3410, 236743, 1590, 107]
    controls = [[2, 818, 2365, 563], [2, 4600, 563]]
    cap = fake_cap(prompt, answer)
    contrasts = [
        {"toward": ["antidisestablishmentarianism"], "away": ["he"], "label": "cannot"},
        {"toward": ["she"], "away": ["he"], "label": "pronoun"},
        {"toward": ["she", "her"], "away": ["he", "him"], "label": None},
    ]
    want_pairs = [None, ([SHE], [HE]), ([SHE, rt.resolve_contrast_word("her")["id"]], [HE, rt.resolve_contrast_word("him")["id"]])]

    # without contrasts: nothing read, and the centre is captured exactly as before (one capture per control)
    rt.captures.clear()
    base = await prepare(rt, cap, centre_token_ids=controls)
    check(base.contrasts == [] and base.contrast_backgrounds == () and base.contrast_controls_k0 is None,
          "no contrasts asked: none resolved, no backgrounds, no control scores")
    check(rt.captures == [(c, 1) for c in controls], f"no contrasts: the captures as before {rt.captures}")

    rt.captures.clear()
    shared = await prepare(rt, cap, centre_token_ids=controls, centre_answer=True, contrasts=contrasts,
                           contrast_backgrounds=["controls", "study_mean", "controls"])
    check(rt.captures == [(c, 1) for c in controls], f"the controls' scores come from the centre's own captures {rt.captures}")
    check(shared.contrast_backgrounds == ("controls", "study_mean"), f"backgrounds once each: {shared.contrast_backgrounds}")
    check(torch.equal(shared.centre_k0, base.centre_k0), "the centre is the same with contrasts read")
    mu = torch.from_numpy(mu_np)

    for k in (0, 2):
        prefix = (prompt + answer)[: len(prompt) + k]
        s = sighting(rt, prefix)
        n_before = len(rt.captures)
        if k > 0:
            await rt._answer_centre(cap, shared, k)  # the centring's call, which analyse_position makes first
        n_centring = len(rt.captures)
        got = await rt._contrast_readings(cap, shared, k, s["raw"], s["J"])
        check(len(rt.captures) == n_centring, f"k={k}: the contrasts add no capture of their own")
        if k > 0:
            check(n_centring - n_before == len(controls), f"k={k}: one capture per control, reading the answer")
        check(len(got) == 3 and got[0]["error"] is not None and "lens" not in got[0] and "backgrounds" not in got[0]
              and "pieces" in got[0], f"k={k}: the contrast that cannot be read: {list(got[0])}")
        for i in (1, 2):
            e = got[i]
            t, a = want_pairs[i]
            check(e["error"] is None and e["toward_ids"] == t and e["away_ids"] == a, f"k={k} #{i}: ids {e['toward_ids']}")
            lens_ref = ref_score(s["J"].float(), t, a, fnw, W)
            naked_ref = ref_score(s["raw"].float(), t, a, fnw, W)
            check(len(e["lens"]) == N_LAYERS and close(e["lens"], lens_ref, 2e-4), f"k={k} #{i}: the lens series")
            check(close(e["naked"], naked_ref, 2e-4), f"k={k} #{i}: the naked-eye series")
            check(e["lens"][59] == e["naked"][59], f"k={k} #{i}: at layer 60 the lens is the identity")
            ctl = e["backgrounds"]["controls"]
            # the controls, as the centring reads them: their own last piece at k = 0, reading this answer past it
            c_prefixes = [c if k == 0 else c + answer[:k] for c in controls]
            c_lens = torch.stack([ref_score(lensed(rt, torch.stack([state(p, l) for l in range(N_LAYERS)])).float(), t, a, fnw, W)
                                  for p in c_prefixes]).mean(0)
            c_naked = torch.stack([ref_score(torch.stack([state(p, l) for l in range(N_LAYERS)]).float(), t, a, fnw, W)
                                   for p in c_prefixes]).mean(0)
            kind = "control_prompt_end" if k == 0 else "control_reading_same_answer"
            check(ctl["kind"] == kind and ctl["n_controls"] == 2 and ctl["why"] is None, f"k={k} #{i}: controls {ctl['kind']}")
            check(close(ctl["control_lens"], c_lens, 2e-4) and close(ctl["control_naked"], c_naked, 2e-4),
                  f"k={k} #{i}: the controls' mean scores")
            check(close(ctl["delta_lens"], lens_ref - c_lens, 5e-4) and close(ctl["delta_naked"], naked_ref - c_naked, 5e-4),
                  f"k={k} #{i}: the differences against the controls")
            sm = e["backgrounds"]["study_mean"]
            cent_ref = ref_score(lensed(rt, s["raw"].float() - mu).float(), t, a, fnw, W)
            d = contrast_direction(t, a, fnw, W)
            muJ = torch.stack([(mu[l].to(rt.jpp_lenses[l].dtype) @ rt.jpp_lenses[l].T).float() for l in range(N_LAYERS)])
            mp_ref = (muJ @ d) / s["J"].float().pow(2).mean(-1).sqrt()
            check(sm["kind"] == "study_mean" and sm["sha256"] == STUDY_MEAN_SHA256
                  and sm["file"] == "experiments/results/E3_background_mu.npy", f"k={k} #{i}: study_mean names its file")
            check(close(sm["centred_lens"], cent_ref, 2e-4), f"k={k} #{i}: centred as E3 centred, through the lens")
            check(close(sm["centred_naked"], ref_score(s["raw"].float() - mu, t, a, fnw, W), 2e-4),
                  f"k={k} #{i}: centred, by the naked eye")
            check(close(sm["mean_part_lens"], mp_ref, 2e-4), f"k={k} #{i}: the mean part through the lens")
            check(close(sm["mean_part_naked"], (mu @ d) / s["raw"].float().pow(2).mean(-1).sqrt(), 2e-4),
                  f"k={k} #{i}: the mean part by the naked eye")
            check(list(e["backgrounds"]) == ["controls", "study_mean"], f"k={k} #{i}: backgrounds in order")

    # the same contrast read against itself as its own control: no difference
    rt.captures.clear()
    own = await prepare(rt, cap, centre_token_ids=[prompt], contrasts=contrasts[1:2], contrast_backgrounds=["controls"])
    s = sighting(rt, prompt)
    e = (await rt._contrast_readings(cap, own, 0, s["raw"], s["J"]))[0]
    check(max(abs(v) for v in e["backgrounds"]["controls"]["delta_lens"]) <= 1e-3
          and max(abs(v) for v in e["backgrounds"]["controls"]["delta_naked"]) <= 1e-3,
          "the prompt as its own control: no difference")
    check(list(e["backgrounds"]) == ["controls"], "only the background asked for")

    # past token 0 without centre_answer, and with no control sent: no difference, and the reason
    plain = await prepare(rt, cap, centre_token_ids=controls, contrasts=contrasts[1:2], contrast_backgrounds=["controls"])
    s = sighting(rt, (prompt + answer)[: len(prompt) + 1])
    ctl = (await rt._contrast_readings(cap, plain, 1, s["raw"], s["J"]))[0]["backgrounds"]["controls"]
    check(ctl["kind"] == "none" and ctl["delta_lens"] is None and ctl["control_lens"] is None
          and "only with centre_answer" in (ctl["why"] or ""), f"past token 0 without centre_answer: {ctl}")
    none = await prepare(rt, cap, contrasts=contrasts[1:2], contrast_backgrounds=["controls"])
    ctl = (await rt._contrast_readings(cap, none, 0, s["raw"], s["J"]))[0]["backgrounds"]["controls"]
    check(ctl["kind"] == "none" and ctl["n_controls"] == 0 and "No matched control was sent" in (ctl["why"] or ""),
          f"no control sent: {ctl}")
    # no background: the series alone
    alone = await prepare(rt, cap, contrasts=contrasts[1:2])
    e = (await rt._contrast_readings(cap, alone, 0, s["raw"], s["J"]))[0]
    check(e["backgrounds"] == {} and len(e["lens"]) == N_LAYERS, "no background asked: the series alone")
    # a centre given by another sweep (the thinking comparison's way): the controls are captured for their scores
    rt.captures.clear()
    over = await prepare(rt, cap, centre_token_ids=controls, centre_override=base.centre_k0, contrasts=contrasts[1:2],
                         contrast_backgrounds=["controls"])
    check(over.centre_k0 is base.centre_k0 and over.contrast_controls_k0 is not None
          and rt.captures == [(c, 1) for c in controls], "with a centre_override the controls are read for their scores")
    try:
        await prepare(rt, cap, contrasts=contrasts[1:2], contrast_backgrounds=["mean"])
        check(False, "an unknown background was taken")
    except ValueError as exc:
        check("Unknown contrast background 'mean'" in str(exc), f"an unknown background: {exc}")
    check(CONTRAST_BACKGROUNDS == ("controls", "study_mean"), f"the backgrounds: {CONTRAST_BACKGROUNDS}")


async def _check_identity(tok: Any, mu_np: np.ndarray) -> None:
    rt = stand_in_runtime(tok, identity=True)
    prompt = [2, 818, 1590, 563]
    cap = fake_cap(prompt, [3410])
    shared = await prepare(rt, cap, contrasts=[{"toward": ["she", "her"], "away": ["he", "him"]}],
                           contrast_backgrounds=["study_mean"])
    s = sighting(rt, prompt)
    e = (await rt._contrast_readings(cap, shared, 0, s["raw"], s["J"]))[0]
    sm = e["backgrounds"]["study_mean"]
    check(e["lens"] == e["naked"], "identity lens: lens series == naked-eye series")
    check(sm["centred_lens"] == sm["centred_naked"], "identity lens: centred series equal")
    check(sm["mean_part_lens"] == sm["mean_part_naked"], "identity lens: mean parts equal")


# --------------------------------------------------------------------------------------------------------------------
# 6. the committed file
# --------------------------------------------------------------------------------------------------------------------


def check_file() -> None:
    print("6. E3's average state as committed")
    data = STUDY_MEAN_PATH.read_bytes() if STUDY_MEAN_PATH.exists() else b""
    check(hashlib.sha256(data).hexdigest() == STUDY_MEAN_SHA256, f"{STUDY_MEAN_PATH.name} is the file pinned")
    mu = read_study_mean()
    check(mu.shape == (60, 5376) and mu.dtype == np.float32 and bool(np.isfinite(mu).all()), f"mu {mu.shape} {mu.dtype}")
    check(study_mean_problem() is None, f"no problem with the file: {study_mean_problem()}")
    side = json.loads(STUDY_MEAN_PATH.with_suffix(".json").read_text())
    check(side["sha256"] == STUDY_MEAN_SHA256 and side["bytes"] == len(data) and side["shape"] == [60, 5376]
          and side["dtype"] == "float32", f"the sidecar pins the same file: {side.get('sha256')}")
    check(side["n_prose"] == 2010 and side["n_chat"] == 2990
          and set(side["sources"]) == {"E3_lens_validation_musum_prose.pt", "E3_lens_validation_musum_chat.pt",
                                       "E3_lens_validation_mu.pt"}
          and side["exported_by"] == "scripts/export_E3_background_mu.py"
          and (ROOT_DIR / side["exported_by"]).exists()
          and all((ROOT_DIR / m).exists() for m in side["made_by"]), "the sidecar names its sources")
    rel = STUDY_MEAN_PATH.relative_to(ROOT_DIR)
    for p in (rel, rel.with_suffix(".json")):
        ignored = subprocess.run(["git", "-C", str(ROOT_DIR), "check-ignore", "-q", str(p)]).returncode == 0
        check(not ignored, f".gitignore lets {p} in")
    check(subprocess.run(["git", "-C", str(ROOT_DIR), "check-ignore", "-q", "experiments/results/E3_lens_validation_mu.pt"]
                         ).returncode == 0, "the study's own caches stay ignored")
    with tempfile.TemporaryDirectory() as tmp:
        missing = Path(tmp) / STUDY_MEAN_PATH.name
        problem = study_mean_problem(missing)
        check(problem is not None and "is not on this server" in problem and 'leave "study_mean" out' in problem,
              f"a missing file: {problem}")
        shutil.copy(STUDY_MEAN_PATH, missing)
        with open(missing, "r+b") as f:
            f.seek(-4, 2)
            f.write(b"\x00\x00\x80\x7f")
        problem = study_mean_problem(missing)
        check(problem is not None and "is not the file this instrument pins" in problem, f"a changed file: {problem}")
        check(study_mean_problem(Path(tmp) / "nothing.npy", STUDY_MEAN_SHA256) is not None, "another missing path")


def main() -> int:
    tok = load_tokenizer()
    check_score(tok)
    if tok is not None:
        check_words(tok)
        check_sweep(tok)
    check_file()
    if failures:
        print(f"\ncontrast readings: {len(failures)} of {passed + len(failures)} checks FAILED")
        return 1
    print(f"\ncontrast readings: all {passed} checks hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())

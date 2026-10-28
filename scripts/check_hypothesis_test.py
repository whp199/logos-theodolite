#!/usr/bin/env python3
"""A CPU-only check of test_motif_hypothesis's test against chance (backend/mcp_scientist.py `motif_statistics`, the
reply's `selectivity`), of the tool and its HTTP route on the tiny random Gemma 4 of scripts/atlas_stand_in.py, and of
the words the folio shows for it (frontend/src/screens/NeuronFolio/replies.ts, bundled with the frontend's own esbuild
and run under node, as scripts/check_interface_defaults.py runs the store):
1. the statistics on hand-made activations whose statistics are known: four a side apart gives Welch's t = 4.38 on 6
   degrees of freedom, Cohen's d = 3.10 over sample variances and an exact p of 2/70, with the verdict and words for
   either face and for a gap the other way; unequal sides against Welch's t, its degrees of freedom and d written out
   by hand; exact p's against every split counted in plain Python; no spread at all, rounding included, as an infinite
   t; identical sides (p = 1, not shown); three a side as far apart as can be (p = 0.1, too few to judge) and one a
   side (no t, no p, too few); past 10,000 splits a seeded permutation p, the same at each call, within its Monte Carlo
   error of the exact p, and never 0; an exact p that is a tie at two figures (0.125) written as Python rounds it
   (`p_text` "0.12"); every reply JSON without NaN or infinity; an empty side refused in plain words, and readings
   that are not finite numbers left untested (CANNOT_TEST), said so;
2. the tool, its capture replaced by hand-made activations: its `selectivity` is `motif_statistics` of them, read
   from the readings unrounded (5.00004 is not the 5.0 its record shows), the keys as first built are kept (`verdict`
   and `cohens_d` over population variances: one sentence a side still reads CONFIRMED, d = 50,000, while the test
   says too few to judge; a reading that is not a finite number still gives the reply), `label_gate` and `lens` last;
   an empty side, a layer or a neuron that does not exist refused in plain words before the model is run;
3. the HTTP route (POST /api/scientist/test_hypothesis, TestClient): the same reply, a reading that is not a finite
   number included, the same refusals as 400s, and a run on the stand-in's own activations;
4. the folio's words: for each verdict the server's own words with the faces named ("selective on its positive face,
   p = 0.029, n = 4 a side"), the server's own p where a tie would round apart, an older server's verdict never read
   as a yes, a p written alike by both and never rounded up to 0.05 from below, and no "yes, clearly" left in the
   folio.

Run: .venv/bin/python -I scripts/check_hypothesis_test.py   (exits 0 when every check holds)
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"

import asyncio
import itertools
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

TMP = Path(tempfile.mkdtemp(prefix="hypothesis-check-"))
os.environ["THEODOLITE_ATLAS_DIR"] = str(TMP / "atlas")
os.environ["THEODOLITE_FIELD_NOTES"] = str(TMP / "notes.json")

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch

import backend.mcp_scientist as ms
from atlas_stand_in import FFN, N_LAYERS, bind_server, stand_in

FOLIO = ROOT_DIR / "frontend" / "src" / "screens" / "NeuronFolio"
ESBUILD = ROOT_DIR / "frontend" / "node_modules" / ".bin" / "esbuild"
OLD_KEYS = ["layer", "neuron_idx", "chat", "verdict", "cohens_d", "mean_pos_signed_act", "mean_neg_signed_act",
            "pos_prompt_results", "neg_prompt_results", "pole_j_vindex_signature", "pole_label_check", "label_gate", "lens"]

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


def close(a: Optional[float], b: float, tol: float = 1e-4) -> bool:
    return a is not None and abs(a - b) <= tol * max(1.0, abs(b))


def welch_by_hand(pos: List[float], neg: List[float]) -> tuple:
    """Welch's t, its degrees of freedom and Cohen's d over the pooled sample variance, written out in plain Python."""
    n1, n2 = len(pos), len(neg)
    m1, m2 = sum(pos) / n1, sum(neg) / n2
    v1 = sum((x - m1) ** 2 for x in pos) / (n1 - 1)
    v2 = sum((x - m2) ** 2 for x in neg) / (n2 - 1)
    se2 = v1 / n1 + v2 / n2
    df = se2**2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    d = (m1 - m2) / math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return (m1 - m2) / math.sqrt(se2), df, d


def exact_p_by_hand(pos: List[float], neg: List[float]) -> float:
    """The two-sided p over every split of the sentences into the two sides' sizes, counted in plain Python."""
    x, n1 = pos + neg, len(pos)
    t_obs = abs(welch_by_hand(pos, neg)[0])
    hits = total = 0
    for pick in itertools.combinations(range(len(x)), n1):
        a = [x[i] for i in pick]
        b = [x[i] for i in range(len(x)) if i not in pick]
        total += 1
        hits += abs(welch_by_hand(a, b)[0]) >= t_obs * (1 - 1e-9)
    return hits / total


def main() -> int:
    try:
        return run_checks()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


def run_checks() -> int:
    torch.set_grad_enabled(False)

    # ------------------------------------------------------------------------- 1. the statistics, hand-made activations
    before = len(failures)
    s = ms.motif_statistics([5.0, 6.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0])
    check(s["n_pos"] == 4 and s["n_neg"] == 4 and s["mean_pos"] == 6.5 and s["mean_neg"] == 2.5 and close(s["sd_pos"], math.sqrt(5 / 3))
          and close(s["sd_neg"], math.sqrt(5 / 3)), f"four a side: the n, means and sample standard deviations: {s}")
    check(close(s["welch_t"], 4 / math.sqrt(2 * (5 / 3) / 4)) and close(s["welch_t"], 4.38178) and close(s["welch_df"], 6.0)
          and close(s["cohens_d"], 4 / math.sqrt(5 / 3)) and close(s["cohens_d"], 3.09839), f"Welch's t 4.38 on 6 degrees of freedom, d 3.10: {s}")
    check(s["p_method"] == "exact" and s["splits"] == 70 and close(s["p"], 2 / 70) and s["seed"] is None,
          f"the exact p over all 70 splits: only this split and its mirror are as far apart, 2/70: {s['p']}")
    check(s["verdict"] == "SELECTIVE_NORTH_POLE (+)" and s["face"] == "+" and s["words"] == "selective on its + face, p = 0.029, n = 4 a side"
          and s["p_text"] == "0.029" and s["min_n"] == 4 and s["alpha"] == 0.05, f"the verdict and its words: {s['verdict']}, {s['words']!r}, {s['p_text']!r}")
    s = ms.motif_statistics([-5.0, -6.0, -7.0, -8.0], [-1.0, -2.0, -3.0, -4.0])
    check(s["verdict"] == "SELECTIVE_SOUTH_POLE (-)" and s["face"] == "-" and close(s["welch_t"], -4.38178) and close(s["cohens_d"], -3.09839)
          and s["words"] == "selective on its - face, p = 0.029, n = 4 a side", f"the - face: {s['verdict']}, {s['words']!r}")
    s = ms.motif_statistics([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])
    check(s["verdict"] == "DIFFERS_NOT_AS_GUESSED" and s["face"] is None and close(s["p"], 2 / 70)
          and s["words"] == "the sides differ, p = 0.029, n = 4 a side, but not as guessed: the quiet sentences fire its + face more",
          f"a gap the other way is not the guess: {s['words']!r}")
    s = ms.motif_statistics([-1.0, -2.0, -3.0, -4.0], [-5.0, -6.0, -7.0, -8.0])
    check(s["verdict"] == "DIFFERS_NOT_AS_GUESSED" and s["words"].endswith("the quiet sentences fire its - face more"), f"nor on the - face: {s['words']!r}")
    pos, neg = [1.0, 3.0, 5.0, 7.0, 9.0], [0.0, 0.5, 1.0, 1.5]
    t_, df_, d_ = welch_by_hand(pos, neg)
    s = ms.motif_statistics(pos, neg)
    check(close(s["welch_t"], t_) and close(s["welch_df"], df_) and close(s["cohens_d"], d_) and abs(df_ - 4.41155) < 1e-4,
          f"unequal sides and spreads: t {s['welch_t']} ({t_:.6g}), df {s['welch_df']} ({df_:.6g}), d {s['cohens_d']} ({d_:.6g})")
    check(s["splits"] == 126 and close(s["p"], exact_p_by_hand(pos, neg)) and "n = 5 and 4" in s["words"],
          f"its exact p over all 126 splits, counted by hand: {s['p']}, {s['words']!r}")
    rng = np.random.default_rng(7)
    agree = []
    for n1, n2, shift in ((4, 5, 1.0), (5, 5, 0.3), (6, 4, 2.0), (4, 4, 0.0)):
        a = [float(v) for v in rng.normal(shift, 1.0, n1)]
        b = [float(v) for v in rng.normal(0.0, 1.5, n2)]
        agree.append(close(ms.motif_statistics(a, b)["p"], exact_p_by_hand(a, b), 1e-5))
    check(all(agree), f"exact p's on random small sides equal every split counted by hand: {agree}")
    s = ms.motif_statistics([2.0] * 4, [1.0] * 4)
    r = ms.motif_statistics([0.1 + 0.2, 0.3, 0.3, 0.1 + 0.2], [0.1] * 4)
    check(s["welch_t"] is None and s["cohens_d"] is None and close(s["p"], 2 / 70) and s["verdict"] == "SELECTIVE_NORTH_POLE (+)"
          and r["welch_t"] is None and close(r["p"], 2 / 70), f"no spread on either side: an infinite t (sent as null), weighed as such, rounding included: {s['p']}, {r['p']}")
    s = ms.motif_statistics([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    check(s["welch_t"] == 0.0 and s["p"] == 1.0 and s["verdict"] == "NOT_SHOWN" and s["words"] == "no difference shown: p = 1, n = 4 a side",
          f"identical sides: {s['words']!r}")
    s = ms.motif_statistics([4.0, 5.0, 6.0], [1.0, 2.0, 3.0])
    check(close(s["p"], 0.1) and s["splits"] == 20 and s["verdict"] == "TOO_FEW_TO_JUDGE"
          and s["words"] == "too few to judge: n = 3 a side, and the test needs at least 4 a side",
          f"three a side as far apart as can be: p = 0.1, so no split reaches 0.05, and it does not judge: {s['p']}, {s['words']!r}")
    s = ms.motif_statistics([5.0], [0.0])
    check(s["welch_t"] is None and s["p"] is None and s["cohens_d"] is None and s["sd_pos"] is None and s["verdict"] == "TOO_FEW_TO_JUDGE"
          and s["words"] == "too few to judge: n = 1 a side, and the test needs at least 4 a side", f"one a side: no t, no p: {s}")
    s = ms.motif_statistics([5.0, 6.0], [0.0, 1.0, 2.0, 3.0])
    check(s["verdict"] == "TOO_FEW_TO_JUDGE" and "n = 2 and 4" in s["words"], f"too few on one side only: {s['words']!r}")
    a = [float(v) for v in rng.normal(0.8, 1.0, 8)]
    b = [float(v) for v in rng.normal(0.0, 1.0, 8)]
    s1, s2 = ms.motif_statistics(a, b), ms.motif_statistics(a, b)
    exact = ms.motif_statistics(a, b, draws=20_000)
    tol = 4 * math.sqrt(exact["p"] * (1 - exact["p"]) / 10_000) + 2 / 10_001
    check(s1 == s2 and s1["p_method"] == "permutation" and s1["splits"] == 10_000 and s1["seed"] == 0 and exact["p_method"] == "exact"
          and exact["splits"] == 12_870, f"8 a side has 12,870 splits: 10,000 seeded shuffles, the same at each call: {s1['p']} {s2['p']}")
    check(abs(s1["p"] - exact["p"]) <= tol and abs(ms.motif_statistics(a, b, seed=1)["p"] - exact["p"]) <= tol and ms.motif_statistics(a, b, seed=1)["seed"] == 1,
          f"the permutation p within its Monte Carlo error of the exact one: {s1['p']} against {exact['p']} (± {tol:.4f})")
    s = ms.motif_statistics([10.0 + i for i in range(8)], [float(i) for i in range(8)])
    check(s["p_method"] == "permutation" and s["p"] >= 1 / 10_001 and s["p"] < 0.01 and s["verdict"] == "SELECTIVE_NORTH_POLE (+)",
          f"eight a side wholly apart: a permutation p never 0 ({s['p']}), and judged")
    s = ms.motif_statistics(*TIE)
    check(s["p"] == 0.125 and s["splits"] == 792 and s["p_text"] == "0.12" and s["words"] == "no difference shown: p = 0.12, n = 5 and 7",
          f"an exact p of 99/792 = 0.125, a tie at two figures, written as Python rounds it: {s['p']}, {s['p_text']!r}, {s['words']!r}")
    untested = ([1.0, float("nan"), 2.0, 3.0], [0.0, 1.0, 2.0, float("inf")])
    s = ms.motif_statistics(*untested)
    check(s["verdict"] == "CANNOT_TEST" and s["p"] is None and s["p_text"] is None and s["welch_t"] is None and s["cohens_d"] is None
          and s["mean_pos"] is None and s["sd_neg"] is None and s["face"] is None and s["n_pos"] == 4
          and s["words"] == "cannot test: 2 of the 8 readings are not a finite number, n = 4 a side",
          f"readings that are not finite numbers: nothing tested, not read as p = 0, and said so: {s['words']!r}")
    allowed = True
    for args in (([5.0], [0.0]), ([2.0] * 4, [1.0] * 4), ([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]), (a, b), untested):
        try:
            json.dumps(ms.motif_statistics(*args), allow_nan=False)
        except ValueError:
            allowed = False
    check(allowed, "every reply is JSON without NaN or infinity")
    try:
        ms.motif_statistics([], [1.0])
        check(False, "an empty side: refused")
    except ValueError as exc:
        check("both sides" in str(exc), f"an empty side refused in plain words, not read as p = 0: {exc}")
    section("the statistics on hand-made activations", before)

    # ---------------------------------------------------------------------- 2. the tool, on hand-made activations
    before = len(failures)
    rt, vault = stand_in()
    app = bind_server(rt, vault)
    real_capture = rt.eager_model.capture
    table: Dict[tuple, float] = {}
    calls = [0]
    N = 5  # the neuron read

    async def fake_capture(ids: List[int], addrs: List[Any]) -> Dict[Any, Any]:
        """The neuron's activation over the sentence: half its value at the start, its value at the last word-piece."""
        calls[0] += 1
        out = {}
        for addr in addrs:
            seq = torch.zeros(len(ids), FFN)
            v = table[tuple(ids)]
            seq[0, N], seq[-1, N] = 0.5 * v, v
            out[addr] = seq
        return out

    def sentences(side: str, values: List[float]) -> List[str]:
        """Sentences each read as `values` says (float32-exact values), entered in the fake capture's table."""
        out = []
        for i, v in enumerate(values):
            text = f"Sentence {side} {i} reads {v}."
            table[tuple(rt.tokenize_prompt(rt.compose_prompt(text, chat=False, system=None)))] = v
            out.append(text)
        return out

    rt.eager_model.capture = fake_capture
    cases = {
        "north": ([5.0, 6.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0]),
        "south": ([-5.0, -6.0, -7.0, -8.0], [-1.0, -2.0, -3.0, -4.0]),
        "differs": ([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]),
        "same": ([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]),
        "one": ([5.0], [0.0]),
        "eight": ([1.5, 0.25, 2.0, 1.0, 0.75, 1.25, 2.5, 0.5], [0.0, 0.5, -0.25, 1.0, 0.25, -0.5, 0.75, 0.125]),
        "tie": TIE,
        # not exact at the 4 places a record is rounded to: as float32 holds them, which is what the capture gives
        "unrounded": ([float(np.float32(v)) for v in (5.00004, 6.00004, 7.00004, 8.00004)], [1.0, 2.0, 3.0, 4.0]),
        "untested": ([5.0, float("nan"), 7.0, 8.0], [1.0, 2.0, 3.0, 4.0]),
    }
    asked = {k: (sentences(f"{k}+", p_), sentences(f"{k}-", n_)) for k, (p_, n_) in cases.items()}
    replies: Dict[str, Any] = {}
    for k, (p_, n_) in asked.items():
        replies[k] = asyncio.run(ms.test_motif_hypothesis(layer=3, neuron_idx=N, pos_prompts=p_, neg_prompts=n_))
    for k, (p_, n_) in cases.items():
        rep = replies[k]
        check(rep["selectivity"] == ms.motif_statistics(p_, n_), f"{k}: the reply's selectivity is motif_statistics of its readings: {rep['selectivity']}")
    rep = replies["north"]
    check([x["last_token_signed_act"] for x in rep["pos_prompt_results"]] == cases["north"][0] and rep["pos_prompt_results"][0]["peak_signed_act"] == 5.0,
          "the hand-made activations are what the tool read, the last word-piece and the peak")
    check(all(k in rep for k in OLD_KEYS) and list(rep)[-2:] == ["label_gate", "lens"] and rep["layer"] == 3 and rep["neuron_idx"] == N,
          f"every key as first built kept, label_gate and lens last: {list(rep)}")
    old_d = 4 / math.sqrt(0.5 * (1.25 + 1.25) + 1e-8)
    check(rep["verdict"] == "CONFIRMED_NORTH_POLE (+)" and rep["cohens_d"] == round(old_d, 4) and rep["selectivity"]["verdict"] == "SELECTIVE_NORTH_POLE (+)",
          f"verdict and cohens_d as first built, over population variances (d = {rep['cohens_d']}), beside the test")
    one = replies["one"]
    check(one["verdict"] == "CONFIRMED_NORTH_POLE (+)" and one["cohens_d"] == 50000.0 and one["selectivity"]["verdict"] == "TOO_FEW_TO_JUDGE"
          and one["selectivity"]["p"] is None, f"one sentence a side: as first built CONFIRMED at d = {one['cohens_d']}; the test: too few to judge")
    check(replies["eight"]["selectivity"]["p_method"] == "permutation" and replies["same"]["selectivity"]["verdict"] == "NOT_SHOWN"
          and replies["differs"]["selectivity"]["verdict"] == "DIFFERS_NOT_AS_GUESSED" and replies["south"]["selectivity"]["verdict"] == "SELECTIVE_SOUTH_POLE (-)",
          "each verdict through the tool")
    un = replies["unrounded"]
    check([x["last_token_signed_act"] for x in un["pos_prompt_results"]] == [5.0, 6.0, 7.0, 8.0] and un["mean_pos_signed_act"] == 6.5
          and un["selectivity"]["mean_pos"] == ms._sig(sum(cases["unrounded"][0]) / 4) and un["selectivity"]["mean_pos"] != 6.5,
          f"the test reads the last word-piece unrounded: mean {un['selectivity']['mean_pos']}, where the records, rounded, give {un['mean_pos_signed_act']}")
    nan_rep = replies["untested"]
    check(all(k in nan_rep for k in OLD_KEYS) and nan_rep["selectivity"]["verdict"] == "CANNOT_TEST" and nan_rep["selectivity"]["p"] is None
          and nan_rep["selectivity"]["words"] == "cannot test: 1 of the 8 readings is not a finite number, n = 4 a side",
          f"a reading that is not a finite number: the reply still comes, its keys as first built, and the test says it cannot: {nan_rep['selectivity']['words']!r}")
    n_calls = calls[0]
    pos4, neg4 = asked["north"]
    refusals = [
        (dict(layer=N_LAYERS, neuron_idx=N, pos_prompts=pos4, neg_prompts=neg4), f"layers count from 0 to {N_LAYERS - 1} here (the pages count them 1 to {N_LAYERS})"),
        (dict(layer=-1, neuron_idx=N, pos_prompts=pos4, neg_prompts=neg4), "layer is -1"),
        (dict(layer=True, neuron_idx=N, pos_prompts=pos4, neg_prompts=neg4), "layer is True"),
        (dict(layer=3, neuron_idx=FFN, pos_prompts=pos4, neg_prompts=neg4), f"each layer's neurons count from 0 to {FFN - 1}"),
        (dict(layer=3, neuron_idx=N, pos_prompts=[], neg_prompts=neg4), "pos_prompts is empty"),
        (dict(layer=3, neuron_idx=N, pos_prompts=pos4, neg_prompts=[]), "neg_prompts is empty"),
        (dict(layer=3, neuron_idx=N, pos_prompts="a sentence", neg_prompts=neg4), "pos_prompts is empty"),
    ]
    for kw, said in refusals:
        try:
            asyncio.run(ms.test_motif_hypothesis(**kw))
            check(False, f"{said}: refused")
        except ValueError as exc:
            check(said in str(exc), f"refused in plain words: {exc}")
    check(calls[0] == n_calls, f"every refusal before the model is run ({calls[0] - n_calls} captures)")
    section("the tool on hand-made activations: its selectivity, the keys as first built, refusals", before)

    # ------------------------------------------------------------------------------------------- 3. the HTTP route
    before = len(failures)
    from fastapi.testclient import TestClient

    client = TestClient(app, base_url="http://127.0.0.1")
    r = client.post("/api/scientist/test_hypothesis", json={"layer": 3, "neuron_idx": N, "pos_prompts": pos4, "neg_prompts": neg4})
    check(r.status_code == 200 and r.json()["selectivity"] == json.loads(json.dumps(replies["north"]["selectivity"]))
          and r.json()["verdict"] == "CONFIRMED_NORTH_POLE (+)", f"the route gives the tool's reply: {r.status_code}")
    r = client.post("/api/scientist/test_hypothesis", json={"layer": 3, "neuron_idx": N, "pos_prompts": asked["untested"][0], "neg_prompts": asked["untested"][1]})
    check(r.status_code == 200 and all(k in r.json() for k in OLD_KEYS) and r.json()["selectivity"]["verdict"] == "CANNOT_TEST"
          and r.json()["cohens_d"] is None, f"a reading that is not a finite number: a reply, not a 400 (its d as first built sent as null): {r.status_code} {r.text[:160]}")
    for body, said in [({"layer": N_LAYERS, "neuron_idx": N}, "layers count from 0"), ({"layer": 3, "neuron_idx": FFN}, "neurons count from 0"),
                       ({"layer": 3, "neuron_idx": N, "pos_prompts": []}, "pos_prompts is empty"),
                       ({"layer": 3, "neuron_idx": N, "neg_prompts": []}, "neg_prompts is empty")]:
        r = client.post("/api/scientist/test_hypothesis", json={"pos_prompts": pos4, "neg_prompts": neg4, **body})
        check(r.status_code == 400 and said in r.json().get("detail", ""), f"{body}: {r.status_code} {r.text[:160]}")
    rt.eager_model.capture = real_capture
    live_pos = ["Alice called Bob because", "Write a snake game in Python.", "Describe the color orange.", "Who are you?"]
    live_neg = ["The committee rejected the proposal", "Mix the flour and salt", "The mitochondria produce ATP", "The quick brown fox"]
    r = client.post("/api/scientist/test_hypothesis", json={"layer": 3, "neuron_idx": N, "pos_prompts": live_pos, "neg_prompts": live_neg})
    live = r.json() if r.status_code == 200 else {}
    sel = live.get("selectivity") or {}
    rounded = [x["last_token_signed_act"] for x in live.get("pos_prompt_results", [])]
    check(r.status_code == 200 and sel.get("n_pos") == 4 and sel.get("n_neg") == 4 and sel.get("p_method") == "exact" and rounded
          and abs(sel["mean_pos"] - sum(rounded) / len(rounded)) <= 1e-4 and sel["verdict"] in ("SELECTIVE_NORTH_POLE (+)", "SELECTIVE_SOUTH_POLE (-)",
          "DIFFERS_NOT_AS_GUESSED", "NOT_SHOWN"), f"on the stand-in's own activations: {sel.get('words')!r}")
    client.close()
    section("the HTTP route: the tool's reply, refusals as 400s, the stand-in's own activations", before)

    # --------------------------------------------------------------------------------------------- 4. the folio's words
    before = len(failures)
    older = {k: v for k, v in replies["one"].items() if k != "selectivity"}
    shown = run_replies({**{k: json.loads(json.dumps(v)) for k, v in replies.items()}, "older": json.loads(json.dumps(older))})
    if shown is not None:
        for k in replies:
            server = replies[k]["selectivity"]["words"]
            want = server.replace("its + face", "its positive face").replace("its - face", "its negative face")
            check(shown[k]["words"] == want, f"{k}: the folio says {shown[k]['words']!r}, the server {server!r}")
        check(shown["north"]["words"] == "selective on its positive face, p = 0.029, n = 4 a side", f"the folio's words: {shown['north']['words']!r}")
        check(shown["one"]["words"].startswith("too few to judge") and shown["one"]["test"]["p"] is None, f"one a side: {shown['one']['words']!r}")
        check(shown["tie"]["words"] == "no difference shown: p = 0.12, n = 5 and 7" and shown["tie"]["test"]["pWords"] == "0.12",
              f"a tie at two figures: the folio shows the server's p, not its own rounding of 0.125 (\"0.13\"): {shown['tie']['words']!r}")
        check(shown["older"]["test"] is None and shown["older"]["words"].startswith("d at 0.8 or more on its positive face")
              and "nothing set against chance" in shown["older"]["words"], f"an older server's CONFIRMED: {shown['older']['words']!r}")
        words = [v["words"] for k, v in shown.items() if k not in ("old", "p")] + shown["old"]
        check(not any("clearly" in w or w.startswith("yes") for w in words), f"no verdict is read as a yes: {words}")
        said = [ms._p_words(p) for p in P_SHOWN]
        check(shown["p"] == said and said[:4] == ["0.0496", "0.04999", "0.05", "0.051"],
              f"a p written alike by the server and the folio, never rounded up to 0.05 from below: {said} {shown['p']}")
    folio = (ROOT_DIR / "frontend" / "src" / "screens" / "NeuronFolio.tsx").read_text(encoding="utf-8")
    replies_ts = (FOLIO / "replies.ts").read_text(encoding="utf-8")
    check("{hypothesisWords(res)}" in folio and "verdictWords(res.verdict)" not in folio and "calls the gap clear" not in folio
          and "there is no null, interval or permutation" not in folio, "the folio shows the test's words, and no longer says it has no test")
    check("yes, clearly" not in replies_ts, "replies.ts: no “yes, clearly”")
    section("the folio's words: the test's, an older server's hedged, never “yes, clearly”", before)

    print(f"\n{passed} checks passed, {len(failures)} failed")
    return 1 if failures else 0


P_SHOWN = [0.0496, 0.04999, 0.05, 0.051, 0.0285714, 1.0, 0.31, 0.00309969, 1 / 10_001]
# five against seven whose exact p is 99 of the 792 splits, 0.125: a tie at two figures, which Python rounds to even
# ("0.12") and toPrecision up ("0.13")
TIE = ([6.0, 6.0, 5.0, 0.0, 8.0], [4.0, 3.0, 0.0, 1.0, 3.0, 1.0, 5.0])
ENTRY = r"""
import { hypothesisWords, parseHypothesis, pText, verdictWords } from "./replies";
const REPLIES = __REPLIES__;
const out = {};
for (const [k, v] of Object.entries(REPLIES)) {
  const h = parseHypothesis(v);
  out[k] = { words: hypothesisWords(h), test: h.test, d: h.d };
}
out.old = ["CONFIRMED_NORTH_POLE (+)", "CONFIRMED_SOUTH_POLE (-)", "WEAK_SELECTIVITY", "REJECTED_OR_INCONCLUSIVE"].map(verdictWords);
out.p = __P__.map((p) => pText(p));
console.log(JSON.stringify(out));
"""


def run_replies(replies: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The folio's reading of each reply: replies.ts bundled with the frontend's esbuild and run under node; None (and
    a failure) when it cannot run."""
    node = shutil.which("node")
    check(ESBUILD.exists(), f"{ESBUILD} is missing: run npm install in frontend/")
    check(node is not None, "node is not on PATH")
    if not ESBUILD.exists() or node is None:
        return None
    entry = ENTRY.replace("__REPLIES__", json.dumps(replies)).replace("__P__", json.dumps(P_SHOWN))
    bundle = subprocess.run(
        [str(ESBUILD), "--bundle", "--platform=node", "--format=esm", "--loader=ts", "--log-level=error",
         "--sourcefile=check_hypothesis_test_entry.ts"],
        input=entry, cwd=FOLIO, capture_output=True, text=True, timeout=120, check=False,
    )
    check(bundle.returncode == 0, f"esbuild could not bundle replies.ts:\n{bundle.stderr}")
    if bundle.returncode != 0:
        return None
    ran = subprocess.run([node, "--input-type=module"], input=bundle.stdout, capture_output=True, text=True, timeout=120, check=False)
    check(ran.returncode == 0, f"the bundled replies.ts did not run under node:\n{ran.stderr}")
    if ran.returncode != 0:
        return None
    return json.loads(ran.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    sys.exit(main())

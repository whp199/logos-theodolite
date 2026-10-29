"""The census (`backend/census.py`): every neuron of a layer pushed on a few fixed questions, each answer kept.

After the steering sweep of lyramakesmusic/neuron-explorer (`sweep/steer_sweep.py`, `sweep/rank.py`, `sweep/shift.py`),
which labels each neuron by what the model writes when it is pushed, beside what it writes unpushed and what it writes
when a random direction of the same size is pushed instead. "Sweep" already names a sighting in this instrument, so
the steering sweep is called a census here. This module holds what the runs, the ranking, the server and the checks
share: the questions, the designs, the push sizes, taking a run (`take_census`, which scripts/census_run.py calls on
the GPU) and planning its batches (`plan_batches`), the ranking and its null (`compute_units`, `shift_scores`), a
calibration run's dose window (`dose_curves`, `dose_window`), how a layer's run is found and read, and the tools over
finished runs (`merge_runs`, `depth_report`, `build_spread`: scripts/census_tools.py).

A push here is the atlas's "add" push (backend/atlas_ops.py): `dose * p50_l * u`, added to the stream leaving layer l
(resid_post) at every word-piece but the start mark, with u the neuron's write direction as the stream receives it,
unit(w_post (.) W_down[:, n]), and p50_l the median length of that stream over the reference texts. So a dose is a share
of the stream's own typical length, the reference's `--dose-unit rel`, and every neuron and every control gets the same
push size: the "stream" unit. In the "ceiling" unit (`DOSE_UNITS`) a dose is instead a multiple of each neuron's write
ceiling, the most it can write by itself (`atlas_ops.Weights.write_ceiling`), a control pushed by the ceiling of the
neuron it is paired with (`control_refs`), so one dose is the same share of each neuron's most at every layer; the
calib_ceil ladder is taken in it. Gemma 4 differs from the reference's model in one way that matters here: its
feed-forward output passes a norm (post_feedforward_layernorm) and the layer's `layer_scalar` before it joins the
stream, so turning the neuron's own activation up is squashed by that norm (erratum E47); the push is added after both,
along the direction the neuron writes, which is what the reference's push at `mlp_out` amounted to on a model without
them (its research log, "What the hook can and can't express").

The same norm shapes the controls. The reference rotates a neuron's down column to a random direction; on Gemma that
column passes w_post before it joins the stream, and w_post is near zero on the ~180 dimensions the next layer's norms
read with up to ~1700 times the gain of the rest. A plain isotropic direction of the same length puts ~3% of itself
there and reaches the next layer about 100 times louder than any neuron's write (pages 36-59 of the 31B), so a control
is drawn as unit(w_post (.) r), r isotropic, which matches the neurons' read size to within ~1.25x at every layer
(`control_dirs`; each run records the two read sizes, `read_gain`).

The doses are neuron-explorer's, chosen on Qwen3.5-4B (`DOSES_FROM`); on Gemma-4-31B they are not yet calibrated, and a
dose of 0.8 there is already longer than the most a single neuron can write at most layers
(`atlas_ops.Weights.write_ceiling`). The calibration ladders (calib_rel in the stream, calib_ceil in the ceiling) are
read by `dose_curves`: per dose, neurons against controls, the share of answers that part from the unpushed one and
where, the share cut off, distinct 4-grams, and the push over each neuron's ceiling, with a hedged verdict per sign.
Every page and tool that shows a census says so, save for a design calibrated at the run's layer (`DOSES_CALIBRATED`,
`doses_calibrated`: break_rel at layers 40-50, counted from 0, by its trial of 29 October 2026). With 256 controls and
21,504 neurons about 84 neurons land above every
control by chance alone (N / (C + 1)), so a rank is a lead, not a finding: each unit carries an empirical p against the
controls and a layer-wide false-discovery q (`compute_units`). A second stage retests a screen's strongest neurons
against fresh controls (`retest_of`), its p and q taken over those alone, on (question, dose) cells its screen did not
ask: in batches of one size a neuron writes the same answer to the same cell bit for bit, so a retest on the screen's
own cells would read the answers it was picked for again (`take_census` refuses it).

break_rel, the design chosen on the two ladders (28 October 2026), asks six questions (`DESIGN_QUESTIONS`) at +-1 and
+-2 of the stream and is read its own way (`DESIGN_READING`, `reading_of`): every answer counted, each answer's own
question's unpushed words left out, and each word set against the controls' mean alone, so a whole layer of neurons is
ranked as a trial of a thousand was. Its six questions are two halves of three (`DESIGN_HALVES`), so a neuron's words
picked on one half can be tested on the other with nothing selected, which a retest of its strongest cannot do.

A run's batches are planned from its whole job list and the batch size (`plan_batches`), and written whole, so a run
continued after a stop or a torn write, and a run taken in shards (`shard_batches`, each but shard 0 with a canary
batch), write every answer in the same batch as an unbroken run: on the GPU a row's numbers depend on its batch's size.
The meta records the batch and the plan (resume keys), the job list, and the runtime class (`runtime_record`; a change
is warned and recorded, never refused). `merge_runs` joins shards only whole, and a shard is never read as a layer's
census.

Rows (one JSON line per answer) keep: layer, kind ("neuron" | "random" | "baseline"), neuron (the control's own index
for "random", -1 for a baseline), prompt, dose, push (the length added: `dose * p50`, or `dose * ceiling` in the ceiling
unit), rel_push (that length over p50), ceiling (the unit's write ceiling: a control's is its paired neuron's; not on a
baseline, nor on rows written before it was kept), text, finish ("stop" at an end mark | "length"), n_gen, token_ids,
logprob (each written piece's own log-probability), top_ids, top_lps (the five likeliest at each step), and, for a
control, the seed its direction was drawn from and the neuron it is paired with (ref_neuron).
"""

from __future__ import annotations

import collections
import glob
import hashlib
import json
import math
import os
import random
import re
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# The reference's four questions (sweep/steer_sweep.py PROMPTS), asked as a chat turn with thinking off. Every design
# that names no questions of its own (DESIGN_QUESTIONS) asks these and writes each one's unpushed answer, so every run
# begun before break_rel keeps its job list and its prompt_ids (a resume key).
PROMPTS: Dict[str, str] = {
    "who": "Who are you?",
    "snake": "Write a snake game in Python.",
    "orange": "Describe the color orange.",
    "free": "Write whatever you like.",
}
# Questions asked only by the designs that name them, fixed on 29 October 2026 before any run asked them: one each in
# three genres the reference's do not cover (how a thing works, advice, an object), each with a spare of its genre that
# takes its place where its unpushed answer parts from itself written alone in a census batch before 32 word-pieces
# (scripts/verify_atlas_writer.py --design), which is decided before any neuron is read.
MORE_PROMPTS: Dict[str, str] = {
    "bicycle": "Explain how a bicycle works.",
    "walk": "Give me three tips for a long walk.",
    "chair": "Describe a wooden chair.",
    "kettle": "Explain how a kettle works.",
    "sleep": "Give me three tips for sleeping well.",
    "bridge": "Describe a stone bridge.",
}
ALL_PROMPTS: Dict[str, str] = {**PROMPTS, **MORE_PROMPTS}
# A design's questions, in the order its runs ask them, where they are not the four PROMPTS. break_rel's six are read
# as two halves of three (DESIGN_HALVES): the calibrated three, and three new ones, so every unit's words can be picked
# on one half and tested on the other, nothing selected.
DESIGN_QUESTIONS: Dict[str, Tuple[str, ...]] = {"break_rel": ("who", "orange", "free", "bicycle", "walk", "chair")}
DESIGN_HALVES: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "break_rel": (("who", "orange", "free"), ("bicycle", "walk", "chair"))}

# Doses as shares of the stream's median length leaving the layer (the reference's "main_rel": its research log found
# the coherent window near 0.5-0.8 of the stream, and +3 reads the direction out as wreckage), and a ladder for
# finding where effects begin and text breaks on this model before a full run ("calib_rel").
DESIGNS: Dict[str, List[Tuple[str, float]]] = {
    "main_rel": [(p, d) for p in ("who", "snake", "orange") for d in (-0.7, 0.5, 0.8)] + [("free", 3.0)],
    "calib_rel": [(p, s * d) for p in ("who", "orange", "free") for d in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0) for s in (-1.0, 1.0)],
    # the same ladder's purpose in each neuron's own unit (dose_unit "ceiling"): half of the most it can write by itself
    # up to eight times it
    "calib_ceil": [(p, s * d) for p in ("who", "orange", "free") for d in (0.5, 1.0, 2.0, 4.0, 8.0) for s in (-1.0, 1.0)],
    # chosen for layers 40-57 on the two ladders (28 October 2026, 64 + 64 units at each of layers 20, 30, 40, 45, 50
    # and 57): +-1 and +-2 of the stream, on both signs, around where text first breaks there (at +-2 at layers 40 and
    # 45; at +-1 for the neurons at 50, at +2 and -1 for them at 57, the controls at +-2 at both; sooner at the earlier
    # layers, +-0.5 at 20 and +-1 at 30); asked on six questions. Calibrated at layers 40-50 by its trial of 29 October
    # 2026 (DOSES_CALIBRATED; experiments/results/census_break_rel_trial)
    "break_rel": [(p, s * d) for p in DESIGN_QUESTIONS["break_rel"] for d in (1.0, 2.0) for s in (-1.0, 1.0)],
}
DEFAULT_DESIGN = "main_rel"
CALIBRATION_DESIGNS = ("calib_rel", "calib_ceil")
# where the doses come from, said wherever a census is shown or handed to the scientist
DOSES_FROM = {
    "main_rel": "neuron-explorer's doses, chosen on Qwen3.5-4B; not yet calibrated on this model (take calib_rel and "
                "calib_ceil runs on a few layers first and read them with census_tools.py curves), and one nominal dose is "
                "a different push at each layer",
    "calib_rel": "a calibration ladder, for finding where effects begin and text breaks on this model",
    "calib_ceil": "a calibration ladder in each neuron's own write ceiling, for finding where effects begin and text breaks "
                  "on this model",
    # (its layers counted from 1, as the pages and the scientist count them, unlike DOSES_CALIBRATED's)
    "break_rel": "+-1 and +-2 of the stream, chosen for layers 41-58 (counted from 1, as the pages count them) on this "
                 "model's two calibration ladders of 28 October 2026 (64 neurons and 64 controls at each of layers 21, 31, "
                 "41, 46, 51 and 58), on which text breaks first at +-2 of the stream at layers 41 and 46, at +-1 for the "
                 "neurons at 51 and at +2 and -1 for them at 58 (the controls at +-2 at both), at +-0.5 at layer 21 and at "
                 "+-1 at 31; calibrated at layers 41-51 by a trial fixed before it was taken (29 October 2026, layer 46, "
                 "1,022 fresh neurons against 1,024 controls), which passed every gate, its lead carried by the +-1 answers "
                 "(at +-2 the text of neurons and controls alike breaks into loops); not at layers 52-58, where its probe at "
                 "layer 58 passed by 0.0008 and its lead could not be told from how the text survives, nor below layer 41",
}
# where a design's doses are calibrated on this model, its first and last layer (counted from 0, as the commands count
# them): there the card says `doses_calibrated` and the page leaves out its red "not yet calibrated" line. break_rel's at
# layers 40-50, the owner's choice of 29 October 2026 on its trial's verdict (experiments/results/census_break_rel_trial):
# the rules fixed before the trial licensed 40-57, but at 57 its probe passed by 0.0008 and its lead could not be told
# from how the text survives. A design named here speaks on every card with DOSES_FROM's words, which say where it is
# calibrated, not with the words its runs' metas recorded before that was recorded (atlas.census_card).
DOSES_CALIBRATED: Dict[str, Tuple[int, int]] = {"break_rel": (40, 50)}


def doses_calibrated(design_name: Optional[str], layer: Optional[int]) -> bool:
    """Whether a design's doses are calibrated on this model at a layer (DOSES_CALIBRATED; the layer counted from 0)."""
    span = DOSES_CALIBRATED.get(str(design_name))
    return span is not None and layer is not None and span[0] <= int(layer) <= span[1]


def design_questions(design_name: Optional[str]) -> Dict[str, str]:
    """The questions a design's runs ask, each written unpushed once, in order, with their texts: DESIGN_QUESTIONS'
    where the design names them, else the four PROMPTS (every design begun before break_rel)."""
    keys = DESIGN_QUESTIONS.get(str(design_name))
    return dict(PROMPTS) if keys is None else {k: ALL_PROMPTS[k] for k in keys}


def meta_questions(meta: Dict[str, Any]) -> List[str]:
    """The questions a run asked, from its meta (`prompts`, in order), else the four PROMPTS (a meta older than it)."""
    return list(meta.get("prompts") or PROMPTS)


# How a design's runs are ranked (`compute_units`, `shift_scores`, `write_units`, `depth_report`): max_rel, the largest
# push counted, a share of the stream; reference, the mean a unit's word counts are set against ("units", every unit's,
# the reference's; "controls", the controls' alone, a control left out of its own, so a neuron's mass does not depend
# on how many neurons share the run and a whole layer ranks as its trial did); exclude, whose unpushed words an answer
# leaves out ("all", every question's, the reference's; "own", its own question's, so one question's unpushed words do
# not hide a theme on the others). main_rel and the ladders keep the reference's reading (main_rel's +3 on free is
# shown, not ranked; `census_tools.py rank --max-rel` reads a ladder past the cap on request). A run's ranking file
# records the reading it was made with (`write_units`, each record's `reading`; a file without one was ranked by
# READING_DEFAULT).
READING_DEFAULT: Dict[str, Any] = {"max_rel": 1.6, "reference": "units", "exclude": "all"}
DESIGN_READING: Dict[str, Dict[str, Any]] = {
    "break_rel": {"max_rel": 2.5, "reference": "controls", "exclude": "own"},
}


def reading_of(design_name: Optional[str]) -> Dict[str, Any]:
    """The reading a design's runs are ranked by (DESIGN_READING over READING_DEFAULT)."""
    return {**READING_DEFAULT, **DESIGN_READING.get(str(design_name), {})}
# The push's unit: "stream", dose x p50, the same push for every neuron and control (the reference's `--dose-unit
# rel`); "ceiling", dose x the neuron's write ceiling (`atlas_ops.Weights.write_ceiling`, the most it can write by
# itself), a control pushed by the ceiling of the neuron it is paired with (`control_refs`), so the null's lengths follow
# the neurons' and one dose is the same share of each neuron's most at every layer. A design whose doses only mean
# something in one unit names it.
DOSE_UNITS = ("stream", "ceiling")
DESIGN_UNITS = {"calib_ceil": "ceiling", "break_rel": "stream"}
# 2: controls drawn as unit(w_post * r) (1 was isotropic, ~100x louder to the next layer than any neuron on the 31B)
CENSUS_VERSION = 2

# Words: runs of three or more letters (any script), lower-cased, counted once per answer (sweep/rank.py TOKEN).
TOKEN = re.compile(r"[^\W\d_]{3,}")


def words(text: str) -> set:
    return set(TOKEN.findall(text.lower()))


def distinct4(ids: Sequence[int]) -> float:
    """The share of distinct 4-grams of word-piece ids: 1 for text that never repeats itself, low for loops."""
    grams = [tuple(ids[i : i + 4]) for i in range(len(ids) - 3)]
    return len(set(grams)) / len(grams) if grams else 1.0


def row_key(r: Dict[str, Any]) -> Tuple[str, int, str, float]:
    """What makes a row unique within a run (a rerun skips the rows it already holds)."""
    return (str(r["kind"]), int(r["neuron"]), str(r["prompt"]), float(r["dose"]))


class SlimRows(list):
    """A run's rows as the ranking reads them, each answer's words held as ids into `vocab` and its other fields cut
    to what the ranking uses (no text, no word-piece ids, no top-k), so a whole layer's 217,604 answers take hundreds of
    megabytes rather than tens of gigabytes. Keys per row: layer, kind, neuron, prompt, dose, rel_push, finish, wid
    (the sorted ids of its words), d4 (distinct4 of its pieces), lp (the mean log-probability of its pieces)."""

    def __init__(self, rows: Iterable[Dict[str, Any]] = ()) -> None:
        super().__init__()
        self.vocab: List[str] = []
        self._ids: Dict[str, int] = {}
        for r in rows:
            self.append(self.slim(r))

    def wid(self, text: str) -> Tuple[int, ...]:
        out = []
        for w in words(text):
            i = self._ids.get(w)
            if i is None:
                i = self._ids[w] = len(self.vocab)
                self.vocab.append(w)
            out.append(i)
        return tuple(sorted(out))

    def slim(self, r: Dict[str, Any]) -> Dict[str, Any]:
        lp = r.get("logprob") or []
        return {"layer": int(r["layer"]), "kind": r["kind"], "neuron": int(r["neuron"]), "prompt": r["prompt"],
                "dose": float(r["dose"]), "rel_push": float(r["rel_push"]), "finish": r.get("finish"),
                "wid": self.wid(str(r.get("text", ""))), "d4": distinct4(r.get("token_ids") or []),
                "lp": float(np.mean(lp)) if lp else 0.0}


def as_slim(rows: Iterable[Dict[str, Any]]) -> SlimRows:
    return rows if isinstance(rows, SlimRows) else SlimRows(rows)


def read_slim(path: Path) -> SlimRows:
    """A run read for ranking, one line at a time (`SlimRows`); a half-written last line (a run killed mid-write) is
    left out rather than stopping the ranking."""
    return SlimRows(read_rows(path, tolerate_tail=True))


def compute_units(rows: Iterable[Dict[str, Any]], max_rel: float = 1.6, top_words: int = 12, reference: str = "units",
                  exclude: str = "all") -> List[Dict[str, Any]]:
    """One record per pushed unit (a neuron or a random control), strongest first, as sweep/rank.py computes them:
    - words: the words over-represented in its answers against the average unit, Poisson z = (count - mean) /
      sqrt(mean + 0.5), counted once per answer, over its answers pushed by at most `max_rel` of the stream, leaving out
      every word the unpushed answers use; [word, count, z], those met at least twice. With `reference` "controls" the
      mean is the controls' alone (a control's leaving it out), not every unit's, so a neuron's z does not move with
      the neurons beside it in the run (no controls, or one: every unit's, as "units"); with `exclude` "own" an answer
      leaves out only its own question's unpushed words, not every question's (`reading_of` says which a design uses);
    - z: its strongest word's z; mass: the summed positive z of its top five words (the sort key);
    - null_pct: the share of random controls with a lower mass (1.0: above every control; None without controls);
    - p: its empirical p against the controls, (1 + controls with at least its mass) / (controls + 1) (a control is
      ranked against the others); q: the neurons' Benjamini-Hochberg false-discovery q over the run (None without
      controls). With C controls no p falls below 1 / (C + 1), and about N / (C + 1) of N neurons that differ from the
      controls in nothing still land above every one of them;
    - per_prompt: for its strongest five words, which questions they appear on;
    - dose: per dose, the mean push (rel_push), the share of distinct 4-grams, the share of answers cut off at the
      length limit, and the mean log-probability of the pieces it wrote.
    `rows` are a run's rows, whole or as `SlimRows`."""
    if reference not in ("units", "controls") or exclude not in ("all", "own"):
        raise ValueError(f"compute_units reads against the units' or the controls' mean and leaves out every question's or "
                         f"the answer's own question's unpushed words, not {reference!r} and {exclude!r}.")
    rows = as_slim(rows)
    vocab = rows.vocab
    base_of: Dict[str, set] = collections.defaultdict(set)
    for r in rows:
        if r["kind"] == "baseline":
            base_of[str(r["prompt"])].update(r["wid"])
    base_ids: set = set().union(*base_of.values()) if base_of else set()
    units: Dict[Tuple[str, int], List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        if r["kind"] != "baseline":
            units[(r["kind"], int(r["neuron"]))].append(r)

    counts: Dict[Tuple[str, int], collections.Counter] = {}
    for u, rs in units.items():
        c: collections.Counter = collections.Counter()
        for r in rs:
            if r["rel_push"] <= max_rel:
                left_out = base_of.get(str(r["prompt"]), set()) if exclude == "own" else base_ids
                c.update(i for i in r["wid"] if i not in left_out)
        counts[u] = c
    ctl_units = [u for u in counts if u[0] == "random"]
    by_controls = reference == "controls" and len(ctl_units) >= 2
    total: collections.Counter = collections.Counter()
    for u in (ctl_units if by_controls else counts):
        total.update(counts[u])
    n_ref = len(ctl_units) if by_controls else max(len(counts), 1)

    def mean_of(u: Tuple[str, int], w: int, cnt: int) -> float:
        if by_controls and u[0] == "random":  # a control set against the other controls
            return (total[w] - cnt) / (n_ref - 1)
        return total[w] / n_ref

    records: List[Dict[str, Any]] = []
    for u, rs in units.items():
        c = counts[u]
        ranked = sorted(
            (((cnt - mean_of(u, w, cnt)) / np.sqrt(mean_of(u, w, cnt) + 0.5), vocab[w], cnt, w) for w, cnt in c.items() if cnt >= 2),
            reverse=True,
        )[:top_words]
        per_prompt = {w: sorted({r["prompt"] for r in rs if i in r["wid"]}) for _, w, _, i in ranked[:5]}
        dose: Dict[str, Dict[str, float]] = {}
        for d in sorted({r["dose"] for r in rs}):
            ds = [r for r in rs if r["dose"] == d]
            dose[f"{d:+g}"] = {
                "rel_push": round(float(np.mean([r["rel_push"] for r in ds])), 3),
                "distinct4": round(float(np.mean([r["d4"] for r in ds])), 3),
                "capped": round(float(np.mean([r["finish"] == "length" for r in ds])), 3),
                "logprob": round(float(np.mean([r["lp"] for r in ds])), 3),
            }
        records.append(
            {
                "kind": u[0],
                "neuron": u[1],
                "layer": int(rs[0]["layer"]),
                "z": round(float(ranked[0][0]), 3) if ranked else 0.0,
                "mass": round(float(sum(max(z, 0.0) for z, _, _, _ in ranked[:5])), 3),
                "words": [[w, int(cnt), round(float(z), 2)] for z, w, cnt, _ in ranked],
                "per_prompt": per_prompt,
                "dose": dose,
            }
        )

    null = np.sort([r["mass"] for r in records if r["kind"] == "random"])
    C = len(null)
    for r in records:
        r["null_pct"] = round(float(np.searchsorted(null, r["mass"]) / C), 4) if C else None
        if not C:
            r["p"] = None
        elif r["kind"] == "random":  # against the other controls: (1 + others at least as strong) / (others + 1)
            r["p"] = round((C - int(np.searchsorted(null, r["mass"], side="left"))) / C, 5) if C > 1 else None
        else:
            r["p"] = round((1 + int(C - np.searchsorted(null, r["mass"], side="left"))) / (C + 1), 5)
    neu = sorted((r for r in records if r["kind"] == "neuron" and r["p"] is not None), key=lambda r: r["p"])
    q_next = 1.0
    for i in range(len(neu) - 1, -1, -1):  # Benjamini-Hochberg, the running minimum from the largest p down
        q_next = min(q_next, neu[i]["p"] * len(neu) / (i + 1))
        neu[i]["q"] = round(q_next, 5)
    for r in records:
        r.setdefault("q", None)
    records.sort(key=lambda r: -r["mass"])
    return records


def chance_above_all(n_neurons: int, n_controls: int) -> Optional[float]:
    """How many of `n_neurons` land above every one of `n_controls` controls by chance alone, were neurons and controls
    alike: N / (C + 1). None without controls."""
    return round(n_neurons / (n_controls + 1), 1) if n_controls else None


def shift_scores(rows: Iterable[Dict[str, Any]], max_rel: float = 1.6, k: int = 5) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """The word-shift score of every unit against the random controls of the same run (sweep/shift.py): every word
    counts, the unpushed answers' too, so a unit that moves common words ("however", "might") scores beside one that
    brings new ones. Per unit and word: presence count over its answers pushed by at most `max_rel`, z = (count - mu) /
    sqrt(mu + 0.5), mu the mean count over the controls (leaving the unit out when it is a control itself); shift = the
    summed |z| of its k largest rises and k largest falls. `rows` whole or as `SlimRows`."""
    rows = as_slim(rows)
    vocab = rows.vocab
    units: Dict[Tuple[str, int], collections.Counter] = collections.defaultdict(collections.Counter)
    for r in rows:
        if r["kind"] == "baseline" or r["rel_push"] > max_rel:
            continue
        units[(r["kind"], int(r["neuron"]))].update(r["wid"])
    ctl = [u for u in units if u[0] == "random"]
    tot: collections.Counter = collections.Counter()
    for u in ctl:
        tot.update(units[u])
    nc = len(ctl)
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for u, c in units.items():
        loo = u[0] == "random"
        denom = nc - 1 if loo else nc
        if denom <= 0:
            out[u] = {"shift": None, "up": [], "down": []}
            continue
        zs = []
        for w in set(c) | set(tot):
            mu = (tot[w] - (c[w] if loo else 0)) / denom
            if mu == 0 and c[w] < 2:
                continue
            zs.append(((c[w] - mu) / np.sqrt(mu + 0.5), vocab[w]))
        zs.sort()
        up = [(w, round(float(z), 2)) for z, w in zs[::-1][:k] if z > 0]
        down = [(w, round(float(z), 2)) for z, w in zs[:k] if z < 0]
        out[u] = {"shift": round(sum(z for _, z in up) - sum(z for _, z in down), 3), "up": up, "down": down}
    return out


# ------------------------------------------------------------------------------------------------------ the dose window

# where text breaks: the answers' distinct 4-grams this far below the unpushed answers'
BREAKS_BY = 0.1
KINDS = ("baseline", "neuron", "random")


def first_departure(ids: Sequence[int], base: Sequence[int]) -> Optional[int]:
    """Where an answer first parts from its question's unpushed answer (sweep/peek.py): the first word-piece that
    differs, or, where one is the other cut short, where the shorter ends; None where the two are the same."""
    for i, (a, b) in enumerate(zip(ids, base)):
        if int(a) != int(b):
            return i
    return None if len(ids) == len(base) else min(len(ids), len(base))


def baseline_ids(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[int]]:
    """{question: the word-piece ids of its unpushed answer}, the first met."""
    base: Dict[str, List[int]] = {}
    for r in rows:
        if r["kind"] == "baseline":
            base.setdefault(str(r["prompt"]), [int(t) for t in r.get("token_ids") or []])
    return base


def dose_curves(rows: Iterable[Dict[str, Any]], ceilings: Optional[Dict[Tuple[str, int], float]] = None,
                base: Optional[Dict[str, Sequence[int]]] = None) -> List[Dict[str, Any]]:
    """A run's dose window (sweep/peek.py --curves): one entry per (dose, kind), the unpushed answers first (kind
    "baseline", dose 0), then every dose of the design for the neurons and for the controls ("random"):
    - n: its answers; units: the units they come from;
    - questions (the unpushed answers' entry alone): the questions the pushed answers are asked, whose unpushed answers
      it holds (a run writes one for every question, PROMPTS, and a calibration design pushes on three of the four);
    - rel_push: the median push over the stream's median length;
    - over_ceiling: the median push over the write ceiling of the unit's neuron (`atlas_ops.Weights.write_ceiling`, the
      most it can write by itself; a control is set against the neuron it is paired with, `control_refs`), from each
      answer's own `ceiling` where it records one, else from `ceilings` ({(kind, neuron): ceiling}); None where
      neither gives it;
    - moved: the share of its answers that part from their question's unpushed answer (`first_departure`), and
      moved_se, the standard error of that share with the units as the sample (each unit's share of its answers
      moved), so a unit's answers on the three questions are not counted as three draws; None from fewer than two
      units, where no spread can be estimated;
    - departure: the median word-piece where they first part (an answer's length where it never does);
    - capped: the share cut off at the length limit; distinct4: their mean share of distinct 4-grams; distinct4_unpushed:
      the same of each answer's own question's unpushed answer, averaged as its answers are (the reference
      `dose_window` sets distinct4 against; None on the unpushed answers' entry); logprob: the mean of each answer's
      mean log-probability.
    `rows` are a run's whole rows, word-piece ids kept (`read_rows`, not `read_slim`). To read a run once, one line at
    a time, give `base` (`baseline_ids` over the run first); else the unpushed answers are found among `rows`."""
    if base is None:
        rows = list(rows)
        base = baseline_ids(rows)
    cells: Dict[Tuple[float, str], Dict[str, List[float]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    moved_by: Dict[Tuple[float, str], Dict[int, List[bool]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    unpushed: Dict[str, Dict[str, List[float]]] = collections.defaultdict(lambda: collections.defaultdict(list))  # per question
    asked: set = set()
    for r in rows:
        kind = str(r["kind"])
        dose = 0.0 if kind == "baseline" else float(r["dose"])
        if kind == "baseline":
            c = unpushed[str(r["prompt"])]
        else:
            c = cells[(dose, kind)]
            asked.add(str(r["prompt"]))
        ids = r.get("token_ids") or []
        lp = r.get("logprob") or []
        c["rel"].append(float(r.get("rel_push") or 0.0))
        c["capped"].append(r.get("finish") == "length")
        c["d4"].append(distinct4(ids))
        c["lp"].append(float(np.mean(lp)) if lp else 0.0)
        cl = r.get("ceiling")
        if cl is None and ceilings is not None:
            cl = ceilings.get((kind, int(r["neuron"])))
        if kind != "baseline" and cl:
            c["over"].append(abs(float(r.get("push") or 0.0)) / float(cl))
        b = base.get(str(r["prompt"]))
        if b is not None:
            dep = first_departure(ids, b)
            c["moved"].append(dep is not None)
            c["dep"].append(len(ids) if dep is None else dep)
            if kind != "baseline":
                c["d4_ref"].append(distinct4(b))
                moved_by[(dose, kind)][int(r["neuron"])].append(dep is not None)
    # the unpushed answers to the questions the doses are asked on (all of them where nothing is pushed)
    questions = [q for q in list(PROMPTS) + sorted(set(unpushed) - set(PROMPTS)) if q in unpushed and (q in asked or not asked)]
    for q in questions:
        for k, v in unpushed[q].items():
            cells[(0.0, "baseline")][k].extend(v)
        moved_by[(0.0, "baseline")][-1].extend(unpushed[q]["moved"])

    def med(v: List[float]) -> Optional[float]:
        return float(np.median(v)) if v else None

    def mean(v: List[float]) -> Optional[float]:
        return float(np.mean(v)) if v else None

    out: List[Dict[str, Any]] = []  # unrounded: dose_window compares them
    for dose, kind in sorted(cells, key=lambda k: (k[1] != "baseline", k[0], KINDS.index(k[1]) if k[1] in KINDS else 9)):
        c = cells[(dose, kind)]
        share = [float(np.mean(v)) for v in moved_by[(dose, kind)].values() if v]
        se = float(np.std(share, ddof=1) / math.sqrt(len(share))) if len(share) > 1 else None
        out.append({"dose": dose, "kind": kind, "n": len(c["rel"]), "units": len(moved_by[(dose, kind)]) or None,
                    **({"questions": questions} if kind == "baseline" else {}),
                    "rel_push": med(c["rel"]), "over_ceiling": med(c["over"]), "moved": mean(c["moved"]), "moved_se": se,
                    "departure": med(c["dep"]), "capped": mean(c["capped"]), "distinct4": mean(c["d4"]),
                    "distinct4_unpushed": mean(c["d4_ref"]), "logprob": mean(c["lp"])})
    return out


def dose_window(curves: Sequence[Dict[str, Any]], dose_unit: str = "stream", questions: Sequence[str] = (),
                breaks_by: float = BREAKS_BY) -> Dict[str, Dict[str, Any]]:
    """Where the window opens and where text breaks, per sign ("+", "-"), read off `dose_curves`:
    - moves: the smallest dose at which the neurons' answers part from the unpushed ones more often than the controls'
      by more than twice the standard error of the difference (units as the sample); None where none does, without
      controls, or where fewer than two neurons or two controls leave the spread unknown (a dose is then not tested);
    - breaks: per kind, the smallest dose at which the answers' distinct 4-grams fall more than `breaks_by` below those
      of the unpushed answers to the same questions (`distinct4_unpushed`); None where none does;
    - says: the same in plain words, hedged: one run, its units, one question set, one greedy answer each. Each is the
      first dose that passes, not a window that stays open: a larger dose may not pass.
    `dose_unit` is the run's ("stream": doses are shares of the stream's median length; "ceiling": multiples of each
    neuron's write ceiling), `questions` the design's."""
    cell = {(c["dose"], c["kind"]): c for c in curves}
    base = next((c for c in curves if c["kind"] == "baseline"), None)
    n_neu = max((c["units"] or 0 for c in curves if c["kind"] == "neuron"), default=0)
    n_ctl = max((c["units"] or 0 for c in curves if c["kind"] == "random"), default=0)

    def at(d: float, c: Dict[str, Any]) -> str:
        if dose_unit == "ceiling":
            rel = c.get("rel_push")
            return f"{d:+g} x each neuron's ceiling" + (f" (a median {rel:.2g} of the stream)" if rel is not None else "")
        over = c.get("over_ceiling")
        return f"{d:+g} of the stream" + (f" (a median {over:.2g} x each neuron's ceiling)" if over is not None else "")

    out: Dict[str, Dict[str, Any]] = {}
    for sign, word in (("+", "up"), ("-", "down")):
        doses = sorted({c["dose"] for c in curves if c["kind"] != "baseline" and (c["dose"] > 0) == (sign == "+") and c["dose"] != 0},
                       key=abs)
        moves, tested, untested = None, 0, 0
        if n_ctl:
            for d in doses:
                a, b = cell.get((d, "neuron")), cell.get((d, "random"))
                if a and b and a["moved"] is not None and b["moved"] is not None:
                    if a["moved_se"] is None or b["moved_se"] is None:  # one unit on a side: no spread to set the gap against
                        untested += 1
                        continue
                    tested += 1
                    gap = a["moved"] - b["moved"]
                    if gap > 0 and gap > 2.0 * math.sqrt(a["moved_se"] ** 2 + b["moved_se"] ** 2):
                        moves = d
                        break
        breaks: Dict[str, Optional[float]] = {}
        for kind in ("neuron", "random"):
            breaks[kind] = None
            for d in doses:
                c = cell.get((d, kind))
                ref = c.get("distinct4_unpushed") if c else None
                if c and c["distinct4"] is not None and ref is not None and c["distinct4"] < ref - breaks_by:
                    breaks[kind] = d
                    break
        if not doses:
            continue
        if not n_ctl:
            says = f"Pushed {word}, with no controls nothing here is set against chance"
        elif moves is None and untested and not tested:
            says = (f"Pushed {word}, with fewer than two neurons or two controls at each dose the spread between units cannot be "
                    "estimated, so nothing here is set against chance")
        elif moves is None:
            says = f"Pushed {word}, the neurons move no more answers than the controls at any dose here" + (
                f" ({untested} dose{'s' if untested != 1 else ''} not tested: fewer than two units on a side)" if untested else "")
        else:
            says = f"Pushed {word}, the neurons move more answers than the controls first at {at(moves, cell[(moves, 'neuron')])}"
        if base is not None:
            bn, br = breaks["neuron"], breaks["random"]
            says += (f"; their text breaks (distinct 4-grams more than {breaks_by:g} below the unpushed answers' to the same "
                     f"questions) first at {at(bn, cell[(bn, 'neuron')])}" if bn is not None else
                     f"; their text does not break (distinct 4-grams more than {breaks_by:g} below the unpushed answers' to the "
                     "same questions) at these doses")
            if n_ctl:
                says += f", the controls' first at {br:+g}" if br is not None else ", the controls' at none of them"
        qs = f" ({', '.join(questions)})" if questions else ""
        says += (f". On {n_neu} neurons and {n_ctl} controls, one question set{qs}, one greedy answer each: a lead for "
                 "choosing doses, not a measurement of them.")
        out[sign] = {"moves": moves, "breaks": breaks, "says": says}
    return out


def cached_checkpoint(repo_id: str) -> List[Path]:
    """The checkpoint's safetensors files in the local Hugging Face cache, wherever huggingface_hub keeps it
    ($HF_HUB_CACHE, $HF_HOME, else ~/.cache/huggingface); [] when it is not all there. Nothing is downloaded."""
    from huggingface_hub import try_to_load_from_cache

    index = try_to_load_from_cache(repo_id, "model.safetensors.index.json")
    if isinstance(index, str):
        names = sorted(set(json.loads(Path(index).read_text(encoding="utf-8"))["weight_map"].values()))
        paths = [try_to_load_from_cache(repo_id, n) for n in names]
        return [Path(p) for p in paths] if paths and all(isinstance(p, str) for p in paths) else []
    one = try_to_load_from_cache(repo_id, "model.safetensors")
    return [Path(one)] if isinstance(one, str) else []


def checkpoint_ceilings(files: Sequence[Path], layer: int, neurons: Sequence[int]) -> Tuple[Dict[int, float], int]:
    """({neuron: its write ceiling}, the layer's width), read on the CPU from a checkpoint's safetensors `files` (a few
    tensors of one layer: its down columns, its post-feed-forward norm and its layer scalar), as
    `atlas_ops.Weights.write_ceiling` reads them on the model: s_l * sqrt(D) * ||w_post (.) unit(W_down[:, n])||. A
    W4A16 down projection is unpacked column by column (`engine_runtime.dequantize_w4a16_cols`)."""
    import torch
    from safetensors import safe_open

    from backend.engine_runtime import dequantize_w4a16_cols

    P = f"model.language_model.layers.{int(layer)}."
    found: Dict[str, Any] = {}
    for path in files:
        with safe_open(str(path), "pt") as f:
            keys = set(f.keys())
            for k in ("mlp.down_proj.weight_packed", "mlp.down_proj.weight_scale", "mlp.down_proj.weight",
                      "post_feedforward_layernorm.weight", "layer_scalar"):
                if P + k in keys and k not in found:
                    found[k] = f.get_tensor(P + k)
    if "post_feedforward_layernorm.weight" not in found or not ("mlp.down_proj.weight_packed" in found or "mlp.down_proj.weight" in found):
        raise ValueError(f"The checkpoint holds no down projection or post-feed-forward norm for layer {layer}.")
    idx = torch.tensor([int(n) for n in neurons], dtype=torch.long)
    if "mlp.down_proj.weight_packed" in found:
        packed = found["mlp.down_proj.weight_packed"]
        width = int(packed.shape[1]) * 8
        cols = dequantize_w4a16_cols(packed, found["mlp.down_proj.weight_scale"], idx).float()
    else:
        dense = found["mlp.down_proj.weight"]
        width = int(dense.shape[1])
        cols = dense[:, idx].T.float()
    w_post = found["post_feedforward_layernorm.weight"].float()
    s = float(found["layer_scalar"].reshape(-1)[0]) if "layer_scalar" in found else 1.0
    unit = cols / cols.norm(dim=-1, keepdim=True).clamp(min=1e-30)
    ceil = s * math.sqrt(int(w_post.shape[0])) * (unit * w_post).norm(dim=-1)
    return {int(n): float(c) for n, c in zip(neurons, ceil.tolist())}, width


# ---------------------------------------------------------------------------------------------------------------- runs

CENSUS_DIR_ENV = "THEODOLITE_CENSUS_DIR"


def census_dir() -> Path:
    """Where census runs live: `$THEODOLITE_CENSUS_DIR`, else the atlas's data folder's `census/runs`
    (backend/atlas_index.atlas_dir)."""
    env = os.environ.get(CENSUS_DIR_ENV)
    if env:
        return Path(env).expanduser()
    from backend.atlas_index import atlas_dir  # the one place the data folder is decided

    return atlas_dir() / "census" / "runs"


def layer_runs(runs_dir: Optional[Path] = None, model: Optional[str] = None,
               refused: Optional[Dict[int, str]] = None, calibration: bool = False) -> Dict[int, Path]:
    """layer -> the path of its census run: a main_rel run first, a screen before a retest of it (`stage`), then the
    largest; calibration runs left out (as the reference's `layer_runs`), or, with `calibration`, only they are read
    (for the dose window, census_tools.py curves); given the model read here, runs taken on another model left out. A
    shard of a run (its meta names its `shard`) is never read as a layer's census: only the run `merge_runs` joins from
    its shards is. Runs taken before the controls were drawn through w_post (CENSUS_VERSION) are not read: given
    `refused`, each such layer's reason goes there."""
    runs_dir = Path(runs_dir) if runs_dir is not None else census_dir()
    best: Dict[int, Tuple[Path, Tuple[bool, bool, int]]] = {}
    for meta_path in glob.glob(str(runs_dir / "*.meta.json")):
        try:
            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(meta, dict) or meta.get("shard") or (meta.get("design_name") in CALIBRATION_DESIGNS) != bool(calibration):
            continue
        if model is not None and meta.get("model") is not None and meta.get("model") != model:
            continue
        jsonl = Path(meta_path[: -len(".meta.json")] + ".jsonl")
        if not jsonl.exists() or not isinstance(meta.get("layer"), int):
            continue
        if int(meta.get("version", 1)) < CENSUS_VERSION:
            if refused is not None:
                refused[meta["layer"]] = (f"{jsonl.name} was taken with isotropic random controls, about 100 times louder to the "
                                          "next layer than any neuron's push on this model: take it again")
            continue
        key = (meta.get("design_name") == DEFAULT_DESIGN, not meta.get("stage"), jsonl.stat().st_size)  # a screen before its retest
        if meta["layer"] not in best or key > best[meta["layer"]][1]:
            best[meta["layer"]] = (jsonl, key)
    return {layer: path for layer, (path, _) in best.items()}


def read_rows(path: Path, tolerate_tail: bool = False) -> Iterable[Dict[str, Any]]:
    """A JSON-lines file's rows. With `tolerate_tail`, a last line cut off mid-write (no newline after it, as when a
    run is killed while writing) is left out instead of raising."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except ValueError:
                if tolerate_tail and not line.endswith("\n"):
                    return
                raise


def drop_partial_tail(path: Path) -> bool:
    """Cut a half-written last line off a JSON-lines file (so appending to it starts on a fresh line); True if one was."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    with open(path, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return False
        size = f.seek(0, os.SEEK_END)
        step, end = 1 << 16, size
        while end > 0:
            start = max(0, end - step)
            f.seek(start)
            nl = f.read(end - start).rfind(b"\n")
            if nl >= 0:
                f.truncate(start + nl + 1)
                return True
            end = start
        f.truncate(0)
        return True


def units_path_of(run_path: Path) -> Path:
    return Path(str(run_path)[: -len(".jsonl")] + ".units.jsonl")


def meta_path_of(run_path: Path) -> Path:
    return Path(str(run_path)[: -len(".jsonl")] + ".meta.json")


def spread_path_of(run_path: Path) -> Path:
    return Path(str(run_path)[: -len(".jsonl")] + ".spread.json")


def run_reading(run_path: Path, max_rel: Optional[float] = None) -> Dict[str, Any]:
    """The reading a run is ranked by: its design's (`reading_of`, the design named in its meta; the reference's for a
    run without one), with `max_rel` in place of the design's where one is given."""
    try:
        meta = json.loads(meta_path_of(Path(run_path)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    reading = reading_of(meta.get("design_name") if isinstance(meta, dict) else None)
    if max_rel is not None:
        reading["max_rel"] = float(max_rel)
    return reading


def write_units(run_path: Path, max_rel: Optional[float] = None, top_words: int = 12) -> Path:
    """Rank a run and write its `.units.jsonl` beside it (atomically): `compute_units`' records, each with its word-shift
    score against the controls (`shift_scores`: shift, shift_up, shift_down; None without controls), read as the run's
    design is (`run_reading`: its cap on the push, its reference and its exclusion; `max_rel` replaces the cap), each
    record carrying that reading (`reading`), so a reader of the file knows what ranked it."""
    out = units_path_of(run_path)
    rows = read_slim(run_path)
    reading = run_reading(run_path, max_rel)
    recs = compute_units(rows, max_rel=reading["max_rel"], top_words=top_words, reference=reading["reference"],
                         exclude=reading["exclude"])
    shifts = shift_scores(rows, max_rel=reading["max_rel"])
    for r in recs:
        sh = shifts.get((r["kind"], int(r["neuron"])), {"shift": None, "up": [], "down": []})
        r["shift"], r["shift_up"], r["shift_down"] = sh["shift"], sh["up"], sh["down"]
        r["reading"] = dict(reading)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(u, ensure_ascii=False) + "\n" for u in recs)
    os.replace(tmp, out)
    return out


def write_json_atomic(path: Path, data: Any, indent: Optional[int] = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=indent), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------- the writer's floor

# The writer's floor (scripts/verify_atlas_writer.py, on the model): with no push, how far an answer written in a census
# batch agrees with the same answer written alone, the rows of other batch shapes not being bit-identical on a GPU
# (N1A). A floor holds for the runtime class it was measured on: these fields of a runtime record (the code's commit
# and the stats index are left out, neither moving an unpushed answer's numbers).
WRITER_FLOOR_ENV = "THEODOLITE_WRITER_FLOOR"
FLOOR_CLASS = ("device", "torch", "cuda", "transformers", "interp_engine", "compressed_tensors")


def writer_floor_files() -> List[Path]:
    """The writer's floors on disk: `$THEODOLITE_WRITER_FLOOR` (a file, or a folder of atlas_writer_floor*.json), else
    experiments/results/atlas_writer_floor*.json in this checkout (where scripts/verify_atlas_writer.py writes them)."""
    env = os.environ.get(WRITER_FLOOR_ENV, "").strip()
    if env:
        p = Path(env).expanduser()
        return sorted(p.glob("atlas_writer_floor*.json")) if p.is_dir() else ([p] if p.exists() else [])
    return sorted((Path(__file__).resolve().parent.parent / "experiments" / "results").glob("atlas_writer_floor*.json"))


def writer_floor(meta: Dict[str, Any], files: Optional[Sequence[Path]] = None) -> Optional[Dict[str, Any]]:
    """What the writer's floor says of a run, from its meta: among the floors measured on the run's model
    (`writer_floor_files`), one on the run's runtime class (FLOOR_CLASS) first, else the first; its floor for the run's
    batch size where that was measured, else the smallest over the census batches it measured. `k` is how many
    word-pieces an unpushed answer written in such a batch kept in common with the same answer written alone before
    they first parted (None: they did not part within the `max_new` measured); `says` puts it in words, a guide for
    reading a pushed answer's departure from an unpushed one written in another shape. None where no floor of the run's
    model is on disk (a file that does not read as a floor is passed over)."""
    found = []
    for p in writer_floor_files() if files is None else files:
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
            if (isinstance(d, dict) and d.get("model") == meta.get("model") and isinstance(d.get("census_floor"), dict)
                    and isinstance(d.get("runtime") or {}, dict)):
                floors_of = {int(b): (None if v is None else int(v)) for b, v in d["census_floor"].items()}
                if floors_of:
                    found.append((Path(p), d, floors_of))
        except (OSError, ValueError, TypeError):
            continue
    if not found:
        return None
    mine = meta.get("runtime") or {}

    def differ(d: Dict[str, Any]) -> List[str]:
        theirs = d.get("runtime") or {}
        return [f for f in FLOOR_CLASS if theirs.get(f) != mine.get(f)]

    asked = meta_questions(meta)

    def covers(d: Dict[str, Any]) -> bool:  # measured on every question the run asks (its results', else its questions')
        res = d.get("results") if isinstance(d.get("results"), dict) else d.get("questions")
        return isinstance(res, dict) and all(q in res for q in asked)

    # the run's runtime class and every one of its questions first, then its runtime class, then every one of its
    # questions, then the first
    path, d, floors = next(((p, d, f) for p, d, f in found if mine and not differ(d) and covers(d)),
                           next(((p, d, f) for p, d, f in found if mine and not differ(d)),
                                next(((p, d, f) for p, d, f in found if covers(d)), found[0])))
    res = d.get("results") if isinstance(d.get("results"), dict) else None
    held = res if res is not None else (d.get("questions") or {})
    missing = [q for q in asked if q not in held]
    ours = [q for q in asked if q in held]
    # the floor over the run's own questions where the file holds each question's results and measured others too (a
    # floor kept without them, older or written by hand, is read as its census_floor says)
    recomputed = res is not None and bool(ours) and set(ours) != set(res)
    if recomputed:
        def over(B: int) -> Optional[int]:
            ks = [int(r["shared"]) for q in ours for n, r in res[q].items()
                  if n.startswith(f"census {B},") and isinstance(r, dict) and not r.get("same")]
            return min(ks) if ks else None

        floors = {B: over(B) for B in floors}
    batch = meta.get("batch")
    if batch in floors:
        batches, k = [int(batch)], floors[int(batch)]
    else:
        batches = sorted(floors)
        ks = [v for v in floors.values() if v is not None]
        k = min(ks) if ks else None
    moved = differ(d) if mine else None
    device = (d.get("runtime") or {}).get("device")
    shape = (f"census batches of {batches[0]} (this run's size)" if batch in floors else
             f"census batches of {' and '.join(map(str, batches))}" + (f" (not this run's {batch})" if batch else ""))
    n_file = len(d.get("questions") or PROMPTS)
    if recomputed and not missing:
        qs_said = f"this run's {len(ours)} questions (of {len(res)} measured)"
    elif recomputed:
        qs_said = f"{len(ours)} of this run's questions (of {len(res)} measured), not its {', '.join(missing)}"
    elif missing:
        qs_said = f"{n_file} questions, not this run's {', '.join(missing)}"
    else:
        qs_said = f"the census's {n_file} questions"
    where = f"on {device}, {qs_said}, {d.get('max_new')} word-pieces"
    if k is None:
        says = (f"Writer's floor: with no push, answers written in {shape} did not part from the same answers written alone "
                f"within the word-pieces measured ({where}): no departure there came from the batch's shape alone.")
    else:
        says = (f"Writer's floor: with no push, answers written in {shape} first part from the same answer written alone after "
                f"{k} word-pieces in common ({where}): a departure that late or later from an answer written in another "
                "shape may be the batch's shape alone, and is not by itself evidence of a push.")
    if moved:
        says += f" It was measured on another runtime than this run's ({', '.join(moved)} differ): a guide, not this run's own floor."
    elif moved is None:
        says += " This run records no runtime to set it against."
    return {"k": k, "batches": batches, "this_batch": batch in floors, "max_new": d.get("max_new"), "device": device,
            "same_runtime": None if moved is None else not moved, "runtime_differ": moved, "file": path.name, "says": says}


# ------------------------------------------------------------------------------------------------- taking a census


def parse_neurons(spec: str, n: int, seed: int = 0) -> List[int]:
    """"all", "random:K" (K neurons drawn without replacement, from the seed), "2742", "0-99" or "1,5,9-12"."""
    spec = str(spec).strip()
    if spec == "all":
        return list(range(n))
    if spec.startswith("random:"):
        k = int(spec.split(":", 1)[1])
        if not 0 < k <= n:
            raise ValueError(f"random:{k} asks for more neurons than the layer's {n}.")
        return sorted(np.random.default_rng(seed + 1).choice(n, k, replace=False).tolist())
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b) + 1) if b else [int(a)])
    bad = [x for x in out if not 0 <= x < n]
    if bad:
        raise ValueError(f"Neurons count from 0 to {n - 1}; {bad[:5]} do not exist.")
    return sorted(set(out))


def control_dirs(seed: int, count: int, w_post: np.ndarray) -> np.ndarray:
    """The random controls' directions ([count, D] float32): unit(w_post (.) r), r isotropic and drawn from the seed,
    w_post the layer's post-feed-forward norm weight. That is the reference's random down column (it rotates a neuron's
    down column to a random direction) carried through the norm every neuron's write passes on Gemma 4, so a control
    reaches the next layer as loudly as a neuron does; a plain isotropic direction reaches it about a hundred times
    louder on the 31B (module docstring). A control is pushed exactly as a neuron is, `dose * p50 * u`, so it asks how
    often a theme turns up when the direction pushed means nothing."""
    shape = np.asarray(w_post, np.float64).reshape(-1)
    v = np.random.default_rng(int(seed)).standard_normal((int(count), shape.shape[0])) * shape
    return (v / np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-30)).astype(np.float32)


def read_gain(write: np.ndarray, ctrl: np.ndarray, w_next: np.ndarray) -> Dict[str, Any]:
    """How loudly the next layer reads a push along each kind of direction: the median of ||w_next (.) u|| over unit
    write directions `write` [K, D] (a sample of the layer's neurons) and over the controls `ctrl` [C, D], w_next the
    weight of the norm the next layer reads the stream through (its input norm; the final norm after the last layer).
    A fair null has a ratio near 1."""
    g = lambda u: float(np.median(np.linalg.norm(np.asarray(u, np.float64) * np.asarray(w_next, np.float64), axis=1))) if len(u) else float("nan")  # noqa: E731
    gn, gc = g(write), g(ctrl)
    return {"neurons": round(gn, 4), "controls": round(gc, 4), "ratio": round(gc / gn, 3) if gn > 0 else None,
            "read_by": "the next layer's input norm (the final norm after the last layer)"}


def gain_sample(seed: int, n: int) -> np.ndarray:
    """The layer's neurons a run measures its read gain over: up to 512 of its `n`, drawn from seed + 2."""
    return np.random.default_rng(int(seed) + 2).choice(int(n), min(512, int(n)), replace=False)


def control_refs(seed: int, count: int, n: int) -> List[int]:
    """The neuron each control is paired with: control k with the k-th neuron of `gain_sample` (cycling), so the
    controls' write ceilings are drawn from the layer's own. Under the ceiling unit a control is pushed by its paired
    neuron's ceiling; the dose window sets a control's push against it under either unit."""
    sample = gain_sample(seed, n)
    return [int(sample[k % len(sample)]) for k in range(int(count))]


def retest_of(units_path: Path, top: int) -> Tuple[List[int], Dict[str, Any]]:
    """Stage two of a two-stage census: the `top` neurons of a screen's ranking (its `.units.jsonl`, strongest mass
    first), and the stage its meta keeps (`stage`: the screen's run, how many of its neurons were ranked, its seed and
    its controls). A retest pushes these alone against fresh controls (another seed, and as many controls as can be
    afforded), so its p and its q, taken over its own run (`compute_units`), are taken over these M neurons alone: the
    screen's search through every neuron of the layer is not counted against them again, and its controls are not
    reused."""
    units_path = Path(units_path)
    if not units_path.name.endswith(".units.jsonl") or not units_path.exists():
        raise ValueError(f"A retest is drawn from a screen's ranking, its <run>.units.jsonl (census_tools.py rank), not {units_path}.")
    run = Path(str(units_path)[: -len(".units.jsonl")] + ".jsonl")
    meta = json.loads(meta_path_of(run).read_text(encoding="utf-8")) if meta_path_of(run).exists() else {}
    ranked = sorted((u for u in read_rows(units_path) if u["kind"] == "neuron"), key=lambda u: -float(u["mass"]))
    if not 0 < int(top) <= len(ranked):
        raise ValueError(f"{units_path.name} ranks {len(ranked)} neurons, so a retest takes 1 to {len(ranked)} of them, not {top}.")
    stage = {"stage": 2, "screen": run.name, "top": int(top), "of": len(ranked), "by": "mass", "screen_layer": meta.get("layer"),
             "screen_seed": meta.get("seed"), "screen_controls": meta.get("random_controls"), "screen_design": meta.get("design_name"),
             # the screen's (question, dose) cells: a retest asks none of them (`shared_cells`, take_census)
             "screen_cells": [[str(p), float(d)] for p, d in (meta.get("design") or DESIGNS.get(str(meta.get("design_name")), []))]}
    return sorted(int(u["neuron"]) for u in ranked[: int(top)]), stage


def shared_cells(design: Sequence[Tuple[str, float]], stage: Dict[str, Any]) -> List[Tuple[str, float]]:
    """The (question, dose) cells a retest's design shares with its screen's (`stage["screen_cells"]`). In batches of one
    size a neuron writes the same answer to the same cell bit for bit (and, in another size, the same first
    word-pieces), so on a shared cell a retest reads again the answer its neuron was picked for: the selection carries
    into its p (in a null simulation from the controls, top-5-of-22 retests on the screen's own cells gave p <= 0.05
    about a fifth of the time; on a held-out question, about one time in twenty, and on the screen's own questions at
    other doses too: 0.04-0.06 over calib_rel's six layers, a screen at +-0.5 or +-4 retested at +-1 and +-2)."""
    screen = {(str(p), float(d)) for p, d in stage.get("screen_cells") or []}
    return sorted({(str(p), float(d)) for p, d in design} & screen)


def unit_order(neurons: Sequence[int], controls: int) -> List[Tuple[str, int]]:
    """The neurons with the controls spread evenly among them, so any prefix of a run has its share of controls."""
    units: List[Tuple[str, int]] = [("neuron", int(n)) for n in neurons]
    for k in range(int(controls)):
        units.insert(round((k + 1) * len(units) / (int(controls) + 1)), ("random", k))
    return units


def census_jobs(neurons: Sequence[int], controls: int, design: Sequence[Tuple[str, float]],
                questions: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Every answer a run writes, in order: one unpushed answer per question (`questions`, the run's: its design's,
    `design_questions`, or its meta's, `meta_questions`; the four PROMPTS by default), then each unit's (question, dose)."""
    jobs = [{"kind": "baseline", "neuron": -1, "prompt": p, "dose": 0.0} for p in (PROMPTS if questions is None else questions)]
    jobs += [{"kind": kind, "neuron": n, "prompt": p, "dose": float(d)} for kind, n in unit_order(neurons, controls) for p, d in design]
    return jobs


def neuron_ranges(neurons: Iterable[int]) -> str:
    """Neurons as `parse_neurons` reads them back, runs of neighbours as ranges: "0-99,105,200-299"."""
    ns = sorted({int(n) for n in neurons})
    parts: List[str] = []
    i = 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
            j += 1
        parts.append(f"{ns[i]}-{ns[j]}" if j > i else str(ns[i]))
        i = j + 1
    return ",".join(parts)


# ------------------------------------------------------------------------------------------------------- batch plans

# A run's batches are planned from its whole job list and the batch size alone, so a run continued after a stop, a run
# taken in shards and an unbroken run write every answer in the same batch: on the GPU a row's numbers depend on the
# size of its batch (cuBLAS chooses its kernels by the shape; rows of the 31B written in batches of other shapes were
# not bit-identical, N1A), though not on which rows share it, every operation of the decoder being per row. A plan
# takes the jobs a chunk of k x batch at a time, each chunk's answers grouped by question so a batch shares its
# question: 10 x batch for runs begun now (one batch's worth of main_rel's units, whose questions split 3 : 3 : 3 : 1, so
# few batches are part-full), 16 x batch kept for runs begun before batch plans were recorded, and 12 x batch for a
# design whose questions are asked equally often by every unit (break_rel's six x four doses: two whole batches a
# question a chunk, a whole layer's batches 99.9% full where 10 x batch fills 83%).
PLANS = {"10 x batch": 10, "16 x batch": 16, "12 x batch": 12}
PLAN = "10 x batch"
PLAN_BEFORE = "16 x batch"
DESIGN_PLAN = {"break_rel": "12 x batch"}  # the plan a design's runs begin with, where not PLAN


def plan_batches(jobs: Sequence[Dict[str, Any]], batch: int, plan: str = PLAN) -> List[Tuple[int, str, List[int]]]:
    """A run's batches in the order they are written, [(chunk, question, [job indices])]: the job list (`census_jobs`)
    taken `plan` x batch at a time (PLANS), each chunk's jobs grouped by question in the order the job list first asks
    them (its unpushed answers come first: PROMPTS' order for every design that asks those), `batch` at a time.
    A function of the whole job list and the batch size alone."""
    if plan not in PLANS:
        raise ValueError(f"The census's batch plans are {', '.join(PLANS)}, not {plan!r}.")
    B = max(int(batch), 1)
    size = B * PLANS[plan]
    order = list(dict.fromkeys(j["prompt"] for j in jobs))
    out: List[Tuple[int, str, List[int]]] = []
    for c, c0 in enumerate(range(0, len(jobs), size)):
        part = range(c0, min(c0 + size, len(jobs)))
        for p in order:
            js = [i for i in part if jobs[i]["prompt"] == p]
            out.extend((c, p, js[b0 : b0 + B]) for b0 in range(0, len(js), B))
    return out


def jobs_digest(jobs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """A run's job list as its meta keeps it: how many answers, and a digest of their keys in order. For a run that has
    kept its plan, the job list its plan is made from; for one that has left it, the job list of every neuron it holds."""
    return {"n": len(jobs), "sha256": hashlib.sha256(json.dumps([list(row_key(j)) for j in jobs]).encode("utf-8")).hexdigest()[:16]}


def shard_batches(plan: Sequence[Tuple[int, str, List[int]]], i: int, n: int) -> Tuple[int, int]:
    """Shard i of n's planned batches, [b0, b1): whole chunks, the chunks [i C / n, (i + 1) C / n) of the plan's C, so
    no chunk is split between shards and each shard writes its batches exactly as an unbroken run would."""
    if not 0 <= int(i) < int(n):
        raise ValueError(f"Shard {i} of {n} does not exist: shards count from 0 to {int(n) - 1}.")
    C = plan[-1][0] + 1 if plan else 0
    if C < int(n):
        raise ValueError(f"The run plans {C} chunk{'s' if C != 1 else ''} of answers, so it splits into at most {C} shards, not {n}.")
    c0, c1 = int(i) * C // int(n), (int(i) + 1) * C // int(n)
    b0 = next(b for b, (c, _, _) in enumerate(plan) if c >= c0)
    b1 = next((b for b, (c, _, _) in enumerate(plan) if c >= c1), len(plan))
    return b0, b1


def shard_path(run_path: Path, i: int, n: int) -> Path:
    return Path(str(run_path)[: -len(".jsonl")] + f".s{int(i)}of{int(n)}.jsonl")


def canary_path_of(run_path: Path) -> Path:
    return Path(str(run_path)[: -len(".jsonl")] + ".canary.jsonl")


def canary_about_path_of(canary_path: Path) -> Path:
    """Beside a canary taken alone (`take_census(..., canary_only=True)`), what it was taken with: `<run>.canary.json`,
    the meta its run would begin with (not a `.meta.json`, so no census reads it as a run)."""
    return Path(str(canary_path)[: -len(".jsonl")] + ".json")


def first_difference(rows: Iterable[Dict[str, Any]], by_key: Dict[Any, Dict[str, Any]]) -> Optional[Tuple[Dict[str, Any], str]]:
    """The first of `rows`, with its first field in sorted order, whose value differs from the row of the same key in
    `by_key`, as whole JSON values (a row missing there differs in every field); None where every row agrees."""
    return next(((r, f) for r in rows for f in sorted(set(r) | set(by_key.get(row_key(r), {})))
                 if r.get(f) != by_key.get(row_key(r), {}).get(f)), None)


def compare_canaries(ours: Path, theirs: Path) -> Dict[str, Any]:
    """Two canaries of one plan batch (a shard's, or one taken alone) compared as whole JSON values, as the merge
    compares a shard's with shard 0's: whether they are the same (`same`), the first row and field that differ, how many
    answers differ and the earliest piece at which one's word-pieces part from the other's (`earliest_departure`), the
    settings their metas disagree on and the runtime fields (card, versions, code) where each canary's about-file is
    beside it (`canary_about_path_of`). `says` puts it in words."""
    a, b = list(read_rows(ours, tolerate_tail=True)), list(read_rows(theirs, tolerate_tail=True))
    out: Dict[str, Any] = {"ours": Path(ours).name, "theirs": Path(theirs).name, "answers": [len(a), len(b)]}
    abouts = []
    for p in (ours, theirs):
        ap = canary_about_path_of(Path(p))
        abouts.append(json.loads(ap.read_text(encoding="utf-8")).get("meta") if ap.exists() else None)
    out["settings_differ"] = settings_differ(*abouts) if all(abouts) else None
    out["runtime_differ"] = runtime_differ(*(m.get("runtime") for m in abouts)) if all(abouts) else None
    out["runtimes"] = [(m or {}).get("runtime") for m in abouts]
    if [row_key(r) for r in a] != [row_key(r) for r in b]:
        out.update({"same": False, "first": None, "rows_differ": None, "earliest_departure": None,
                    "says": f"not the same batch: {out['answers'][0]} and {out['answers'][1]} answers of other keys (other "
                            "settings, another batch size or another plan)"})
        return out
    other = {row_key(r): r for r in b}
    diff = first_difference(a, other)
    rows_differ = sum(1 for r in a if r != other[row_key(r)])
    deps = [d for d in (first_departure(r.get("token_ids") or [], other[row_key(r)].get("token_ids") or []) for r in a) if d is not None]
    out.update({"same": diff is None, "rows_differ": rows_differ, "earliest_departure": min(deps) if deps else None,
                "first": None if diff is None else {"field": diff[1], "kind": diff[0]["kind"], "neuron": diff[0]["neuron"],
                                                     "prompt": diff[0]["prompt"], "dose": diff[0]["dose"]}})
    if diff is None:
        out["says"] = f"identical: {len(a)} answers, every field"
    else:
        r, f = diff
        out["says"] = (f"differ in {rows_differ} of {len(a)} answers (first in {f} of {r['kind']} {r['neuron']} {r['prompt']} "
                       f"{float(r['dose']):+g}); " + (f"the earliest parts from the other's at piece {min(deps)}" if deps else
                                                       "every answer writes the same word-pieces"))
    if out["runtime_differ"]:
        out["says"] += f"; written on other runtimes ({', '.join(out['runtime_differ'])} differ)"
    return out


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def runtime_record(device: Any = None, act_scale: Optional[Path] = None) -> Dict[str, Any]:
    """The runtime class a run's bits come from, as its meta keeps it: the card ("cpu", or the GPU's name), torch and
    its CUDA, transformers, interp-engine and compressed-tensors, the code's git commit and whether its tracked files
    are as committed, and the sha256 of the stats index the push's unit was read from. A run continued, or a shard
    joined, on another class is said and recorded (`mixed_runtime`), never refused: each class's readings are valid;
    their bits are not alike. The W4A16 unpacker is recorded beside the class, not in it (the meta's `unpacker` and
    `unpacker_sessions`, backend/w4a16_fast.py): every unpacker gives the reference's bits (scripts/check_dequant.py,
    and its self-test at install on the card), so a run continued under another is not of another class."""
    import importlib.metadata as md
    import subprocess

    import torch

    def version(name: str) -> Optional[str]:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return None

    dev = torch.device(device) if device is not None else torch.device("cpu")
    root = Path(__file__).resolve().parent.parent
    sha: Optional[str] = None
    clean: Optional[bool] = None
    try:
        got = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20)
        if got.returncode == 0 and got.stdout.strip():
            sha = got.stdout.strip()
            st = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, timeout=20)
            clean = st.returncode == 0 and not st.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
        "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": version("transformers"),
        "interp_engine": version("interp-engine"), "compressed_tensors": version("compressed-tensors"),
        "git": sha, "git_clean": clean, "act_scale_sha256": file_sha256(Path(act_scale)) if act_scale else None,
    }


def runtime_differ(old: Optional[Dict[str, Any]], new: Optional[Dict[str, Any]]) -> List[str]:
    """The fields two runtime records disagree on ([] where either is missing: a run taken before they were kept)."""
    if not old or not new:
        return []
    return [k for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)]


def runtime_classes(records: Iterable[Optional[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """The distinct runtime records among `records`, in the order met (None left out)."""
    out: List[Dict[str, Any]] = []
    for r in records:
        if r and r not in out:
            out.append(r)
    return out


# the runtime class of answers written before runs recorded theirs: unknown, so never taken for the class written after
RUNTIME_UNKNOWN: Dict[str, Any] = {"device": None, "note": "written before runtime records were kept"}


def runtime_parts(meta: Dict[str, Any], has_rows: bool = True) -> List[Dict[str, Any]]:
    """The runtime classes a run's meta says its answers were written on: its `mixed_runtime`, else its `runtime`, else,
    for a run begun before runtime records that holds answers, RUNTIME_UNKNOWN."""
    if meta.get("mixed_runtime"):
        return list(meta["mixed_runtime"])
    if meta.get("runtime"):
        return [meta["runtime"]]
    return [RUNTIME_UNKNOWN] if has_rows else []


def row_keys(path: Path, tolerate_tail: bool = False) -> Tuple[List[Tuple[str, int, str, float]], List[int]]:
    """A JSON-lines run's row keys in file order, and the byte at which each row's line starts. With `tolerate_tail`, a
    last line with no newline after it (a run killed mid-write; `drop_partial_tail` cuts it) is left out, the file
    untouched."""
    keys: List[Tuple[str, int, str, float]] = []
    starts: List[int] = []
    if not Path(path).exists():
        return keys, starts
    pos = 0
    with open(path, "rb") as f:
        for line in f:
            if tolerate_tail and not line.endswith(b"\n"):
                break
            if line.strip():
                keys.append(row_key(json.loads(line)))
                starts.append(pos)
            pos += len(line)
    return keys, starts


# what must agree for a run to be continued (a rerun of the same command resumes; anything else is a different run)
RESUME_KEYS = ("model", "layer", "design_name", "design", "prompt_ids", "max_new", "top_k", "seed", "random_controls", "push",
               "controls", "version", "dose_unit", "batch", "plan")
# what a key missing from an older meta means: it was taken before the key was recorded
RESUME_DEFAULTS: Dict[str, Any] = {"dose_unit": "stream"}
# keys a meta lacks when it was taken before batch plans: any batch was accepted then, and such a run still continues
BEFORE_PLANS = ("batch", "plan")


def settings_differ(old: Dict[str, Any], new: Dict[str, Any]) -> List[str]:
    """The settings two runs' metas disagree on (RESUME_KEYS, a key an older meta lacks read as RESUME_DEFAULTS says, or,
    for BEFORE_PLANS, as agreeing with any, and the push's unit p50 beyond float noise)."""
    differ = []
    for k in RESUME_KEYS:
        a, b = old.get(k, RESUME_DEFAULTS.get(k)), new.get(k, RESUME_DEFAULTS.get(k))
        if k in BEFORE_PLANS and (a is None or b is None):
            continue
        if json.dumps(a) != json.dumps(b):
            differ.append(k)
    p_old, p_new = old.get("p50"), new.get("p50")
    if p_old is not None and p_new is not None and abs(float(p_old) - float(p_new)) > 1e-4 * max(1.0, abs(float(p_new))):
        differ.append("p50")
    return differ


def write_census_batch(rt: Any, layer: int, prompt_ids: Sequence[int], delta: Any, *, max_new: int, stop_ids: Sequence[int],
                       top_k: int = 5, cancel: Any = None) -> Optional[List[Dict[str, Any]]]:
    """One census batch: len(delta) answers to one question (`prompt_ids`, the same for every row, so none is padded),
    row i pushed by delta[i] ([B, D] float32, added to the stream leaving `layer` at every word-piece but the start mark;
    a row of zeros is a row unpushed), written greedily together (`atlas_ops.write_rows`), at most `max_new` pieces each,
    a row stopping after the first of `stop_ids` it writes. Per row: token_ids (the end mark among them when one was
    written), logprob (each piece's, to 4 places), top_ids, top_lps and ended; None when `cancel()` turned true.
    take_census writes every batch through it, and scripts/verify_atlas_writer.py its census-shaped batches, so the
    writer's floor is measured on the census's own writer."""
    from backend import atlas_ops as ao
    from backend.causal_check import language_model

    B = int(delta.shape[0])
    iv = ao.Interventions(rows=B, adds={int(layer): delta}) if bool(delta.abs().sum() > 0) else None
    rows: List[Dict[str, Any]] = [{"token_ids": [], "logprob": [], "top_ids": [], "top_lps": [], "ended": False} for _ in range(B)]
    for st in ao.write_rows(language_model(rt.hf_model), rt.lm_head_weight, [list(prompt_ids)] * B, iv, max_new=max_new,
                            stop_ids=stop_ids, bos_id=rt.tokenizer.bos_token_id, spare_bos=True,
                            softcap=ao.softcap_of(rt.hf_model), top_k=top_k, cancel=cancel):
        r = rows[st.row]
        r["token_ids"].append(st.token_id)
        r["logprob"].append(round(st.logprob, 4))
        r["top_ids"].append(st.top_ids)
        r["top_lps"].append(st.top_lps)
        r["ended"] = r["ended"] or st.end
    if cancel is not None and cancel():
        return None
    return rows


def take_census(
    rt: Any,
    layer: int,
    neurons: Sequence[int],
    out_path: Path,
    *,
    p50: float,
    p50_source: str,
    design_name: str = DEFAULT_DESIGN,
    controls: int = 256,
    seed: int = 0,
    max_new: int = 256,
    top_k: int = 5,
    batch: int = 32,
    neurons_spec: str = "",
    dose_unit: str = "",
    shard: Optional[Tuple[int, int]] = None,
    p50_file: Optional[Path] = None,
    lens_id: Optional[str] = None,
    stage: Optional[Dict[str, Any]] = None,
    canary_only: bool = False,
    log: Any = print,
    cancel: Any = None,
) -> Dict[str, Any]:
    """Take (or continue) a census of layer `layer` into `out_path` (JSON lines; its `.meta.json` beside it): each
    neuron and each control pushed on every (question, dose) of the design, and the questions unpushed, one greedy
    answer each of at most `max_new` word-pieces, `batch` answers at a time (`atlas_ops.write_rows`, each row its own
    push; the rows of a batch share their question, so none is padded). A dose is in `dose_unit` (DOSE_UNITS; "" for
    the design's own, "stream" but for calib_ceil). `p50_file` is the stats index p50 was read from, if any (its sha256
    goes in the runtime record); `lens_id` the lens the runtime was loaded with, for a caller that has freed it from the
    card (census_run.py), else it is read from the runtime; `stage` a retest's stage (`retest_of`), kept in the meta.

    The batches are planned from the whole job list and the batch size (`plan_batches`), and written whole, in plan
    order, or not at all (`cancel()` stops between steps). Continuing, a planned batch whose answers are all in the file
    is skipped, and a last batch written in part (a run killed mid-write) is cut off and written again whole, so a
    stopped run finishes as an unbroken one would, row for row; a file holding answers its plan did not write is
    refused, before anything in it is cut. A run begun before batch plans (its meta has none), or whose neurons changed
    since it was begun, has left its plan (`plan_kept` false, `plan_left` why): its missing answers are written by the
    plan's batches, a batch's missing part as a batch of its own; its meta's `neuron_ids` and `jobs` then name every
    neuron the run holds, not this call's alone. `shard` (i, n) takes shard i of n of the plan's chunks
    (`shard_batches`) into `out_path` (`shard_path`); every shard but 0 first writes plan batch 0 again beside it
    (`canary_path_of`) for the merge to compare. A shard is continued only as the same shard, and a retest only with the
    neurons it was begun with. A run begun before runtime records, continued, records its earlier answers' runtime as
    unknown (`RUNTIME_UNKNOWN` in `mixed_runtime`). `canary_only` writes plan batch 0 alone, as a shard's canary, beside
    the run's path (with the meta the run would begin with, `canary_about_path_of`), and nothing else: for comparing
    two cards before a run (census_tools.py canary). A canary already there is kept only when its about-file names the
    same settings and runtime (else refused; with no about-file it is written again). Returns counts."""
    import time

    import torch

    from backend import atlas_ops as ao
    from backend import w4a16_fast
    from backend.causal_check import language_model

    if design_name not in DESIGNS:
        raise ValueError(f"The census's designs are {', '.join(DESIGNS)}, not {design_name!r}.")
    unit = dose_unit or DESIGN_UNITS.get(design_name, "stream")
    if unit not in DOSE_UNITS:
        raise ValueError(f"The census's dose units are {', '.join(DOSE_UNITS)}, not {unit!r}.")
    if DESIGN_UNITS.get(design_name, unit) != unit:
        what = ("multiples of each neuron's write ceiling" if DESIGN_UNITS[design_name] == "ceiling" else
                "shares of the stream's median length (dose x p50)")
        raise ValueError(f"{design_name}'s doses are {what}: take it with the {DESIGN_UNITS[design_name]} unit, not the "
                         f"{unit} unit.")
    by_ceiling = unit == "ceiling"
    lm = language_model(rt.hf_model)
    w = ao.Weights(lm, rt)
    L, N, D = w.n_layers, w.d_ff, w.d_model
    if not 0 <= int(layer) < L:
        raise ValueError(f"Layer {layer} does not exist: layers count from 0 to {L - 1}.")
    bad = [n for n in neurons if not 0 <= int(n) < N]
    if bad:
        raise ValueError(f"Neurons count from 0 to {N - 1}; {bad[:5]} do not exist.")
    design = DESIGNS[design_name]
    tok = rt.tokenizer
    prompts = design_questions(design_name)  # the four PROMPTS for every design but those naming their own
    prompt_ids = {k: rt.tokenize_prompt(rt.compose_prompt(v, chat=True)) for k, v in prompts.items()}
    from backend import cpu_memory  # on the CPU, a batch against MemAvailable, refused before anything is written

    cpu_memory.ensure(rt, lambda: cpu_memory.writing_need(rt, batch, max(len(p) for p in prompt_ids.values()), max_new),
                      f"A census batch of {batch} answers of up to {max_new} word-pieces",
                      "lower --batch or --max-new (a run already begun keeps its own: take another --run)")
    stop = rt.end_of_turn_ids()
    out_path = Path(out_path)
    w_post = w.norm(layer, "post_feedforward").cpu().numpy()
    ctrl_np = control_dirs(seed, controls, w_post)
    sample = gain_sample(seed, N)
    w_next = (w.norm(layer + 1, "input") if layer + 1 < L else rt.final_norm_weight.detach().float()).cpu().numpy()
    gain = read_gain(w.write_dirs(layer, sample.tolist()).cpu().numpy(), ctrl_np, w_next)
    refs = control_refs(seed, controls, N)
    ref_ceiling = dict(zip(sorted(set(refs)), w.write_ceiling(layer, sorted(set(refs))).tolist())) if refs else {}
    ctrl = torch.from_numpy(ctrl_np)

    def write_batch(bj: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        B = len(bj)
        delta = torch.zeros(B, D, dtype=torch.float32, device=w.device)
        ceil: List[Optional[float]] = [None] * B  # each pushed row's write ceiling (a control's: its paired neuron's)
        rows_n = [i for i, j in enumerate(bj) if j["kind"] == "neuron"]
        if rows_n:
            ids_n = [bj[i]["neuron"] for i in rows_n]
            u = w.write_dirs(layer, ids_n)
            c_n = w.write_ceiling(layer, ids_n).tolist()
            for k, i in enumerate(rows_n):
                ceil[i] = float(c_n[k])
                delta[i] = u[k] * (bj[i]["dose"] * (ceil[i] if by_ceiling else p50))
        for i, j in enumerate(bj):
            if j["kind"] == "random":
                ceil[i] = float(ref_ceiling[refs[j["neuron"]]])
                delta[i] = ctrl[j["neuron"]].to(w.device) * (j["dose"] * (ceil[i] if by_ceiling else p50))
        got = write_census_batch(rt, layer, prompt_ids[bj[0]["prompt"]], delta, max_new=max_new, stop_ids=stop, top_k=top_k,
                                 cancel=cancel)
        if got is None:
            return None
        gen = [g["token_ids"] for g in got]
        lps = [g["logprob"] for g in got]
        tids = [g["top_ids"] for g in got]
        tlps = [g["top_lps"] for g in got]
        ended = [g["ended"] for g in got]
        out = []
        for i, j in enumerate(bj):
            ids = gen[i][:-1] if ended[i] else gen[i]
            c = ceil[i]
            if by_ceiling and c is not None:  # the push's real share of the stream, which max_rel filters on
                push, rel = round(j["dose"] * c, 4), round(abs(j["dose"]) * c / p50, 6)
            else:
                push, rel = round(j["dose"] * p50, 4), abs(j["dose"]) if j["kind"] != "baseline" else 0.0
            rec = {"layer": int(layer), "kind": j["kind"], "neuron": int(j["neuron"]), "prompt": j["prompt"], "dose": j["dose"],
                   "push": push, "rel_push": rel}
            if c is not None:
                rec["ceiling"] = round(c, 6)
            rec.update({"text": tok.decode(ids, skip_special_tokens=True), "finish": "stop" if ended[i] else "length",
                        "n_gen": len(gen[i]), "token_ids": gen[i], "logprob": lps[i], "top_ids": tids[i], "top_lps": tlps[i]})
            if j["kind"] == "random":
                rec["seed"] = int(seed)
                rec["ref_neuron"] = int(refs[j["neuron"]])
            out.append(rec)
        return out

    meta = {
        "model": getattr(rt, "model_id", None), "layer": int(layer), "prompts": prompts, "prompt_ids": prompt_ids,
        "design_name": design_name, "design": [[p, d] for p, d in design], "max_new": int(max_new), "top_k": int(top_k),
        "seed": int(seed), "random_controls": int(controls), "neurons": neurons_spec or f"{len(neurons)} neurons",
        "push": ("add: dose x the neuron's write ceiling (s_l sqrt(D) ||w_post * unit(W_down[:, n])||, the most it can "
                 "write by itself) along unit(w_post * W_down[:, n]) (a control: dose x the ceiling of the neuron it is "
                 "paired with, along unit(w_post * r), r isotropic), at the stream leaving the layer, every word-piece but "
                 "the start mark") if by_ceiling else
                ("add: dose x p50 along unit(w_post * W_down[:, n]) (a control: along unit(w_post * r), r isotropic), at "
                 "the stream leaving the layer, every word-piece but the start mark"),
        "controls": "unit(w_post * r), r isotropic: the reference's random down column through the post-feed-forward norm",
        "dose_unit": unit,
        "ceilings": "each pushed answer records its unit's write ceiling (`ceiling`; a control's is that of the neuron it is "
                    "paired with, `ref_neuron`)" + (", the push's own unit" if by_ceiling else ", for the dose window alone"),
        "version": CENSUS_VERSION, "read_gain": gain, "doses_from": DOSES_FROM.get(design_name),
        "p50": float(p50), "p50_source": p50_source, "stop_ids": stop, "greedy": True, "chat": "thinking off",
        "lens": lens_id if lens_id is not None else (rt.lens_card() or {}).get("id") if hasattr(rt, "lens_card") else None,
        "stand_in": bool(getattr(rt, "atlas_stand_in", False)),
        # how this session unpacked the 4-bit weights (backend/w4a16_fast.py): every unpacker gives the same bits, so a
        # run may be continued under another (every session's then kept in "unpacker_sessions"); not a resume key
        "unpacker": w4a16_fast.describe(rt),
        "d_ff": int(N), "neuron_ids": neuron_ranges(neurons), "batch": int(batch),
        "runtime": runtime_record(w.device, p50_file),
    }
    if stage:
        if int(seed) == stage.get("screen_seed"):
            raise ValueError(f"A retest draws fresh controls: give it another seed than its screen's ({seed}).")
        same = shared_cells(design, stage)
        if same:
            raise ValueError(f"A retest asks its neurons on cells its screen did not: {design_name} shares {len(same)} (question, "
                             f"dose) cell{'s' if len(same) != 1 else ''} with {stage.get('screen')} ("
                             + ", ".join(f"{p} {d:+g}" for p, d in same[:4]) + (", ..." if len(same) > 4 else "")
                             + "). On a shared cell a neuron writes again the answer it was picked for (bit for bit in "
                             "batches of one size, its first word-pieces in another), so the retest's p would carry the "
                             "screen's selection; retest on a design with other questions or doses.")
        meta["stage"] = stage
    if controls and gain["ratio"] is not None and not 0.5 <= gain["ratio"] <= 2.0:
        log(f"warning: the next layer reads the controls {gain['ratio']}x as loudly as the neurons (a fair null is near 1)")
    meta_path = meta_path_of(out_path)
    old = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    # a run keeps the plan it was begun with
    meta["plan"] = (old.get("plan") or PLAN_BEFORE) if old is not None else DESIGN_PLAN.get(design_name, PLAN)
    all_jobs = census_jobs(neurons, controls, design, list(prompts))
    meta["jobs"] = these_jobs = jobs_digest(all_jobs)
    plan = plan_batches(all_jobs, batch, meta["plan"])

    def put_canary(canary: Path, again: bool = False) -> Optional[int]:
        """Plan batch 0 written again into `canary` (atomically), outside the run: how many answers, 0 when it holds them
        already (unless `again`), None when stopped first."""
        want = [row_key(all_jobs[i]) for i in plan[0][2]]
        have = [row_key(r) for r in read_rows(canary, tolerate_tail=True)] if canary.exists() else []
        if have == want and not again:
            return 0
        recs = write_batch([all_jobs[i] for i in plan[0][2]])
        if recs is None:
            return None
        tmp = canary.with_name(canary.name + ".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
        os.replace(tmp, canary)
        return len(recs)

    if canary_only:
        # plan batch 0 alone, as every shard but 0 writes it, to compare one card with another before a run is begun:
        # beside the run's path and never in it, the run's own file and meta left as they are
        if shard:
            raise ValueError("A canary is the plan's first batch whatever the shard: take it without --shard.")
        canary = canary_path_of(out_path)
        about = canary_about_path_of(canary)
        try:
            was = json.loads(about.read_text(encoding="utf-8")).get("meta") if about.exists() else None
        except (OSError, ValueError, AttributeError):
            was = None
        if was:
            # a canary already here is kept only where this command would write it: the same settings, the same runtime
            differ = settings_differ(was, meta) + [f"the runtime's {k}" for k in runtime_differ(was.get("runtime"), meta["runtime"])]
            if differ:
                raise ValueError(f"{canary.name} was taken with other settings or on another runtime ({', '.join(differ)}; "
                                 f"{about.name} holds what it was taken with); name another run (--run), or remove both to "
                                 "take it again.")
        # with no about-file, what a canary here was taken with is unknown: it is written again
        n_canary = put_canary(canary, again=not was)
        if n_canary is None:
            log("stopped before the canary was written; the same command writes it")
            return {"written": 0, "skipped": 0, "stopped": True, "canary": str(canary)}
        if n_canary or not was:
            write_json_atomic(about, {"what": f"plan batch 0 of {out_path.name}, written alone (census_run.py --canary-only), to "
                                              "compare with the same batch written on another card (census_tools.py canary)",
                                      "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "answers": len(plan[0][2]),
                                      "meta": meta})
        log(f"{canary.name}: plan batch 0 ({len(plan[0][2])} answers) " + ("written" if n_canary else "already there")
            + f", its settings and runtime in {about.name}")
        return {"written": n_canary, "skipped": 0 if n_canary else len(plan[0][2]), "stopped": False, "canary": str(canary)}
    if shard and [int(n) for n in neurons] != sorted({int(n) for n in neurons}):
        raise ValueError("A run taken in shards takes its neurons in order, each once (as parse_neurons gives them), so the "
                         "merge can rebuild its job list.")
    b0, b1 = shard_batches(plan, *shard) if shard else (0, len(plan))
    if shard:
        meta["shard"] = {"i": int(shard[0]), "n": int(shard[1]), "chunks": [plan[b0][0], plan[b1 - 1][0] + 1], "batches": [b0, b1]}
    if old is not None:
        differ = settings_differ(old, meta)
        if differ:
            raise ValueError(f"{out_path.name} was taken with other settings ({', '.join(differ)}); name another run "
                             "(--run) or remove it.")
        was = old.get("shard") or {}
        if (was.get("i"), was.get("n")) != ((int(shard[0]), int(shard[1])) if shard else (None, None)):
            raise ValueError(f"{out_path.name} is shard {was['i']} of {was['n']} of a run: it is continued only as that shard "
                             f"(--shard {was['i']}/{was['n']}, the run named as its other shards are), and joined with "
                             "census_tools.py merge." if was else
                             f"{out_path.name} was not taken as a shard: a run is taken in shards from its beginning; name "
                             "another run (--run).")
        if shard and old.get("jobs") != these_jobs:
            raise ValueError(f"{out_path.name} is a shard of a run with other neurons, and a shard's answers cannot change: "
                             "take the shards again under another run (--run).")
        if old.get("stage") or stage:
            if not (old.get("stage") and stage):
                raise ValueError(f"{out_path.name} is a retest of {old['stage']['screen']}'s strongest, continued only as that "
                                 "retest (--neurons-from, --top)." if old.get("stage") else
                                 f"{out_path.name} is not a retest: name the retest another run (--run).")
            if old.get("neuron_ids") != meta["neuron_ids"]:
                raise ValueError(f"{out_path.name} retests neurons {old.get('neuron_ids')}, fixed when it was begun, and the "
                                 f"screen's {stage['top']} strongest are now {meta['neuron_ids']} (ranked again since?): a "
                                 "retest of the new ones is another run (--run).")
            meta["stage"] = old["stage"]  # a retest's stage is the one it was begun with
        meta["started_at"] = old.get("started_at")
        meta["neurons"] = old.get("neurons") if neurons_spec in ("", old.get("neurons")) else f"{old.get('neurons')} + {neurons_spec}"
        # every session's unpacker, the first first ("unpacker" is this one's; None where a meta did not record one)
        meta["unpacker_sessions"] = old.get("unpacker_sessions", [old.get("unpacker")]) + [meta["unpacker"]]
    meta["started_at"] = meta.get("started_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    keys, starts = row_keys(out_path, tolerate_tail=True)  # a half-written last line is cut off only once nothing is refused
    if old is not None:
        # the run's neurons, every call's that wrote to it: what neuron_ids and the job list describe
        held = {int(n) for k, n, _, _ in keys if k == "neuron"} | {int(n) for n in neurons}
        if isinstance(old.get("neuron_ids"), str) and old["neuron_ids"]:
            held |= set(parse_neurons(old["neuron_ids"], N))
        if held != {int(n) for n in neurons}:
            meta["neuron_ids"] = neuron_ranges(held)
            meta["jobs"] = jobs_digest(census_jobs(sorted(held), controls, design, list(prompts)))
        moved = runtime_differ(old.get("runtime"), meta["runtime"])
        if moved:
            log(f"warning: {out_path.name} was begun on another runtime ({', '.join(moved)} differ): its answers' bits may "
                "differ from the earlier ones' (recorded as mixed_runtime)")
        elif keys and not old.get("runtime") and not old.get("mixed_runtime"):
            log(f"warning: {out_path.name} was begun before runs recorded their runtime: the one its answers so far were written "
                "on is unknown (recorded as mixed_runtime)")
        classes = runtime_classes(runtime_parts(old, bool(keys)) + [meta["runtime"]])
        if len(classes) > 1:
            meta["mixed_runtime"] = classes
    left = None if not keys else ("answers written without a meta" if old is None else
                                  "answers written before batch plans" if "plan_kept" not in old else
                                  "the run's neurons changed after it was begun" if old.get("jobs") != these_jobs else
                                  (old.get("plan_left") or "answers written off its plan") if not old.get("plan_kept") else None)
    if left and shard:
        raise ValueError(f"{out_path.name} holds a shard's answers its plan did not write ({left}): take the shards again "
                         "under another run (--run).")
    batch_of = {row_key(all_jobs[i]): b for b, (_, _, idx) in enumerate(plan) for i in idx}
    torn = 0
    if keys and not left:
        # a last batch written in part: its rows are the file's last, and the file holds no others of it
        last = batch_of.get(keys[-1])
        while torn < len(keys) and batch_of.get(keys[-1 - torn]) == last:
            torn += 1
        if last is None or torn >= len(plan[last][2]) or sum(batch_of.get(k) == last for k in keys) != torn:
            torn = 0
        kept = set(keys[: len(keys) - torn])
        stray = sum(1 for k in kept if not b0 <= batch_of.get(k, -1) < b1)
        part = next(((b, n_) for b in range(b0, b1) for n_ in [sum(row_key(all_jobs[i]) in kept for i in plan[b][2])]
                     if 0 < n_ < len(plan[b][2])), None)
        if stray or part:
            what = (f"{stray} answer{'s' if stray != 1 else ''} outside its planned batches" if stray else
                    f"batch {part[0]} of its plan holds {part[1]} of its {len(plan[part[0]][2])} answers, not at the end of the file")
            raise ValueError(f"{out_path.name} holds answers its batch plan did not write ({what}): rows of another plan "
                             "or another run; name another run (--run) or remove it.")
    meta["plan_kept"] = not left
    if left:
        meta["plan_left"] = left
    if drop_partial_tail(out_path):  # nothing refused: the file is touched from here on
        log(f"{out_path.name}: a half-written last answer was cut off; it is written again")
    write_json_atomic(meta_path, meta)
    if torn:
        with open(out_path, "rb+") as f:
            f.truncate(starts[len(keys) - torn])
        log(f"{out_path.name}: its last batch was written in part ({torn} of its {len(plan[batch_of[keys[-1]]][2])} answers): "
            "cut off, and written again whole")
        keys = keys[: len(keys) - torn]
    done = set(keys)
    todo = [(b, miss) for b in range(b0, b1) for miss in [[i for i in plan[b][2] if row_key(all_jobs[i]) not in done]] if miss]
    total = sum(len(plan[b][2]) for b in range(b0, b1))
    log(f"layer {layer}: {len(neurons)} neurons + {controls} controls, {sum(len(m) for _, m in todo)} answers to write "
        f"({len(done)} already written)" + (f", shard {shard[0]} of {shard[1]} (batches {b0} to {b1 - 1} of {len(plan)})" if shard else "")
        + ", push unit " + ("each neuron's write ceiling, " if by_ceiling else "") + f"p50 {p50:.2f} ({p50_source})")
    if left:
        log(f"{out_path.name} has left its batch plan ({left}): its missing answers are written by the plan's batches, a "
            "batch's missing part as a batch of its own")
    written, pieces, t0 = 0, 0, time.time()

    if shard and int(shard[0]) > 0:
        # the canary: plan batch 0 (shard 0's first) written again here, outside the run, for the merge to compare
        canary = canary_path_of(out_path)
        n_canary = put_canary(canary)
        if n_canary is None:
            log("stopped before the canary was written; the same command continues the shard")
            return {"written": 0, "skipped": len(done), "stopped": True}
        if n_canary:
            log(f"{canary.name}: plan batch 0 written again ({n_canary} answers), for the merge to compare with shard 0's")

    with open(out_path, "a", encoding="utf-8") as f:
        for k, (b, miss) in enumerate(todo):
            recs = write_batch([all_jobs[i] for i in miss])
            if recs is None:
                log(f"stopped: {written} answers written; the same command continues the run")
                return {"written": written, "skipped": len(done), "stopped": True}
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
            f.flush()
            written += len(recs)
            pieces += sum(r["n_gen"] for r in recs)
            if k == len(todo) - 1 or plan[todo[k + 1][0]][0] != plan[b][0]:  # a chunk's last batch
                dt = time.time() - t0
                log(f"{len(done) + written}/{total} answers, {pieces / max(dt, 1e-9):,.0f} pieces/s, {written / max(dt, 1e-9):.2f} answers/s")
    return {"written": written, "skipped": len(done), "stopped": False}


# ------------------------------------------------------------------------------------- shards, depth and the spread


def merge_runs(inputs: Sequence[Path], out: Path, partial: bool = False) -> Tuple[int, int]:
    """Join runs of one layer taken in parts into one run, a row per (kind, neuron, prompt, dose), the first met kept
    (sweep/merge_runs.py); its meta is the first input's with the parts named (`shards`), their unpackers session by
    session (`unpacker_sessions`), their neurons joined (`neurons`, and `neuron_ids` and `jobs` over every part's
    neurons where each part records them), its plan kept only where every part kept one plan (else `plan_left` says
    why), and, where they were taken on different runtime classes, each class (`mixed_runtime`; a part begun before
    runtime records counts as RUNTIME_UNKNOWN). Runs taken with other settings, and a file without its meta (a shard's
    canary among them), are refused. Shards of one run (census_run.py --shard i/n, each meta naming its `shard`) are
    joined more strictly: every shard 0..n-1 once, the union of their answers the run's whole job list (what is missing
    is listed, and refused unless `partial`, which the merged meta then records), each shard's canary compared with
    shard 0's answers as whole JSON values (`canary`). Returns (rows read, rows kept)."""
    inputs = [Path(p) for p in inputs]
    lost = [p.name for p in inputs if not meta_path_of(p).exists()]
    if lost:
        raise ValueError(f"{', '.join(lost)} {'has' if len(lost) == 1 else 'have'} no meta beside {'it' if len(lost) == 1 else 'them'} "
                         "(<run>.meta.json), so nothing says how it was taken: only runs census_run.py wrote are joined (a "
                         "shard's canary is compared by the merge, not joined).")
    metas = [json.loads(meta_path_of(p).read_text(encoding="utf-8")) for p in inputs]
    if any(m.get("shard") for m in metas):
        return _merge_shards(inputs, metas, Path(out), partial)
    for p, m in zip(inputs[1:], metas[1:]):
        differ = settings_differ(metas[0], m)
        if differ:
            raise ValueError(f"{p.name} was taken with other settings than {inputs[0].name} ({', '.join(differ)}).")
    n_in, n_out, _, _ = _join_rows(inputs, Path(out))
    meta = dict(metas[0])
    meta["shards"] = [p.name for p in inputs]
    # each part's unpackers, session by session, in the parts' order
    meta["unpacker_sessions"] = _unpacker_sessions(metas)
    specs = list(dict.fromkeys(str(m.get("neurons")) for m in metas))
    if len(specs) > 1:
        meta["neurons"] = " + ".join(specs)
    if all(isinstance(m.get("neuron_ids"), str) and m.get("d_ff") for m in metas):
        held = sorted({n for m in metas for n in parse_neurons(m["neuron_ids"], int(m["d_ff"]))})
        meta["neuron_ids"] = neuron_ranges(held)
        meta["jobs"] = jobs_digest(census_jobs(held, int(meta["random_controls"]), [tuple(x) for x in meta["design"]],
                                               meta_questions(meta)))
    else:  # a part taken before runs recorded them: the joined run's neurons are those its rows hold
        meta.pop("neuron_ids", None)
        meta.pop("jobs", None)
    if any("plan_kept" in m for m in metas):
        meta.pop("plan_left", None)
        meta["plan_kept"] = all(m.get("plan_kept") for m in metas) and len({json.dumps(m.get("jobs")) for m in metas}) == 1
        if not meta["plan_kept"]:
            meta["plan_left"] = next((m.get("plan_left") or "answers written before batch plans" for m in metas if not m.get("plan_kept")),
                                     "joined from runs taken in parts, each by its own plan")
    classes = runtime_classes(r for m in metas for r in runtime_parts(m))
    meta.pop("mixed_runtime", None)
    if len(classes) > 1:
        meta["mixed_runtime"] = classes
    write_json_atomic(meta_path_of(Path(out)), meta)
    return n_in, n_out


def _unpacker_sessions(metas: Sequence[Dict[str, Any]]) -> List[Any]:
    """Every session's unpacker (backend/w4a16_fast.py) across `metas`, in order; None where a meta recorded none."""
    return [u for m in metas for u in m.get("unpacker_sessions", [m.get("unpacker")])]


def _join_rows(inputs: Sequence[Path], out: Path, keep: Optional[set] = None) -> Tuple[int, int, set, Dict[Any, Dict[str, Any]]]:
    """Stream `inputs` into `out` (atomically), the first row per key kept; returns (rows read, rows kept, their keys,
    the rows of the first input whose keys are in `keep`)."""
    seen: set = set()
    kept_rows: Dict[Any, Dict[str, Any]] = {}
    n_in = n_out = 0
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for j, p in enumerate(inputs):
            for r in read_rows(p):
                n_in += 1
                k = row_key(r)
                if j == 0 and keep is not None and k in keep:
                    kept_rows.setdefault(k, r)
                if k in seen:
                    continue
                seen.add(k)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_out += 1
    os.replace(tmp, out)
    return n_in, n_out, seen, kept_rows


def _merge_shards(inputs: List[Path], metas: List[Dict[str, Any]], out: Path, partial: bool) -> Tuple[int, int]:
    not_shards = [p.name for p, m in zip(inputs, metas) if not m.get("shard")]
    if not_shards:
        raise ValueError(f"Shards of a run are joined among themselves, and {', '.join(not_shards)} is not one.")
    ns = sorted({int(m["shard"]["n"]) for m in metas})
    if len(ns) > 1:
        raise ValueError(f"These shards split their runs {' and '.join(map(str, ns))} ways: only shards of one run are joined.")
    n = ns[0]
    by_i: Dict[int, Tuple[Path, Dict[str, Any]]] = {}
    for p, m in zip(inputs, metas):
        i = int(m["shard"]["i"])
        if i in by_i:
            raise ValueError(f"Shard {i} of {n} is given twice ({by_i[i][0].name} and {p.name}).")
        by_i[i] = (p, m)
    order = sorted(by_i)
    p0, m0 = by_i[order[0]]
    for i in order[1:]:
        p, m = by_i[i]
        differ = settings_differ(m0, m) + (["jobs"] if m.get("jobs") != m0.get("jobs") else [])
        if differ:
            raise ValueError(f"{p.name} was taken with other settings than {p0.name} ({', '.join(differ)}).")
    missing = [i for i in range(n) if i not in by_i]
    if missing and not partial:
        raise ValueError(f"Shard{'s' if len(missing) > 1 else ''} {', '.join(map(str, missing))} of {n} "
                         f"{'are' if len(missing) > 1 else 'is'} missing: finish {'them' if len(missing) > 1 else 'it'} and merge "
                         "again, or merge with --partial.")
    jobs = census_jobs(parse_neurons(m0["neuron_ids"], int(m0["d_ff"])), int(m0["random_controls"]), [tuple(x) for x in m0["design"]],
                       meta_questions(m0))
    if jobs_digest(jobs) != m0.get("jobs"):
        raise ValueError(f"The run's job list cannot be rebuilt from {p0.name}'s meta (its digest differs).")
    plan = plan_batches(jobs, int(m0["batch"]), m0["plan"])
    canary_keys = {row_key(jobs[i]) for i in plan[0][2]}
    tmp_out = out.with_name(out.name + ".merging")
    n_in, n_out, have, first = _join_rows([by_i[i][0] for i in order], tmp_out, keep=canary_keys if 0 in by_i else None)
    absent = [row_key(j) for j in jobs if row_key(j) not in have]
    if absent and not partial:
        tmp_out.unlink()
        raise ValueError(f"{len(absent)} of the run's {len(jobs)} answers are in no shard (the first: "
                         + "; ".join(f"{k} {nn} {pp} {d:+g}" for k, nn, pp, d in absent[:3])
                         + "): finish the shards and merge again, or merge with --partial.")
    os.replace(tmp_out, out)
    # the canaries: each shard's plan batch 0, against shard 0's own answers, as whole JSON values
    same, notes = 0, []
    for i in order:
        if i == 0:
            continue
        cpath = canary_path_of(by_i[i][0])
        if 0 not in by_i:
            notes.append(f"shard {i}'s not compared (shard 0 is missing)")
            continue
        if not cpath.exists():
            notes.append(f"shard {i} wrote none")
            continue
        rows_c = list(read_rows(cpath, tolerate_tail=True))
        diff = first_difference(rows_c, first)
        if diff is None and len(rows_c) == len(canary_keys):
            same += 1
        elif diff is None:
            notes.append(f"shard {i}'s holds {len(rows_c)} of plan batch 0's {len(canary_keys)} answers")
        else:
            r, f = diff
            notes.append(f"shard {i}'s differs from shard 0's (first in {f} of {r['kind']} {r['neuron']} {r['prompt']} {float(r['dose']):+g})")
    others = len([i for i in order if i != 0])
    meta = {k: v for k, v in m0.items() if k != "shard"}
    meta["shards"] = [by_i[i][0].name for i in order]
    meta["unpacker_sessions"] = _unpacker_sessions([by_i[i][1] for i in order])
    meta["canary"] = "; ".join([f"identical on {same} of {others} shard{'s' if others != 1 else ''}"] + notes) if others else "one shard: none to compare"
    meta["plan_kept"] = all(bool(by_i[i][1].get("plan_kept")) for i in order)
    classes = runtime_classes(r for i in order for r in runtime_parts(by_i[i][1]))
    meta.pop("mixed_runtime", None)
    if len(classes) > 1:
        meta["mixed_runtime"] = classes
    if missing or absent:
        meta["partial"] = {"missing_shards": missing, "missing_answers": len(absent),
                           "first_missing": [list(k) for k in absent[:5]]}
    write_json_atomic(meta_path_of(out), meta)
    return n_in, n_out


def depth_report(runs: Dict[int, Path], neurons: int = 384, controls: int = 64) -> List[Dict[str, Any]]:
    """Per layer with a census, how far its neurons move the model's words beyond what random directions do
    (sweep/depth_report.py): the controls' mass at its median, 95th percentile and maximum, the neurons' at its median
    and 90th percentile, the share of neurons above the controls' 95th percentile and above all of them, the answers'
    coherence at +0.8 (neurons and controls) and -0.7 (neurons) (main_rel's doses), and at each of the run's own doses
    (`coherence`), the share of neuron-control pairs a neuron wins (`auc`, ties half), the five strongest neurons, and
    the run's design and reading (each run ranked as its design is, `reading_of`). A run holding more
    than `neurons` neurons or `controls` controls is subsampled to that pool (seeded by the layer), so the mass, which
    grows with the pool, compares across layers. The layers are pushed at one nominal dose each, a share of each
    layer's own stream; how hard that pushes the next layer differs from layer to layer (by about sixteen-fold across
    the 31B's layers), so the curve compares doses that are not alike (`dose_note` in each entry); a run taken in each
    neuron's write ceiling (`dose_unit` "ceiling") compares like with like, the same share of each neuron's most at
    every layer."""
    report: List[Dict[str, Any]] = []
    for l, path in sorted(runs.items()):
        try:
            meta = json.loads(meta_path_of(Path(path)).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        unit = meta.get("dose_unit", "stream")
        design_name = meta.get("design_name")
        reading = reading_of(design_name)
        rows = read_slim(Path(path))
        nn = sorted({int(r["neuron"]) for r in rows if r["kind"] == "neuron"})
        cc = sorted({int(r["neuron"]) for r in rows if r["kind"] == "random"})
        sub = len(nn) > neurons or len(cc) > controls
        if sub:
            rng = random.Random(int(l))
            keep = {("neuron", n) for n in (rng.sample(nn, neurons) if len(nn) > neurons else nn)}
            keep |= {("random", n) for n in (rng.sample(cc, controls) if len(cc) > controls else cc)}
            kept = SlimRows()
            kept.vocab = rows.vocab
            kept.extend(r for r in rows if r["kind"] == "baseline" or (r["kind"], int(r["neuron"])) in keep)
            rows = kept
        recs = compute_units(rows, max_rel=reading["max_rel"], reference=reading["reference"], exclude=reading["exclude"])
        null = np.array([r["mass"] for r in recs if r["kind"] == "random"], np.float64)
        neu = [r for r in recs if r["kind"] == "neuron"]
        m = np.array([r["mass"] for r in neu], np.float64)
        entry: Dict[str, Any] = {"layer": int(l), "run": Path(path).name, "subsampled": sub, "n_neurons": len(m), "n_controls": len(null),
                                 "dose_unit": unit, "design": design_name, "reading": reading,
                                 "dose_note": DOSES_FROM[design_name] if doses_calibrated(design_name, l) else
                                 ("one nominal dose, a multiple of each neuron's own write ceiling, the same share of "
                                  "each neuron's most at every layer; doses not yet calibrated on this model")
                                 if unit == "ceiling" else
                                 "one nominal dose, a different push at each layer; doses not yet calibrated on this model"
                                 + (f" at this layer (only at layers {DOSES_CALIBRATED[design_name][0]}-"
                                    f"{DOSES_CALIBRATED[design_name][1]}, counted from 0)" if design_name in DOSES_CALIBRATED else "")}
        if not len(null) or not len(m):
            entry["note"] = "a depth reading needs neurons and controls in the run"
            report.append(entry)
            continue
        p95 = float(np.percentile(null, 95))

        def coh(kind: str, dose: str) -> Dict[str, Optional[float]]:
            v = [r["dose"][dose] for r in recs if r["kind"] == kind and dose in r["dose"]]
            return {k: (round(float(np.mean([x[k] for x in v])), 3) if v else None) for k in ("distinct4", "capped", "logprob")}

        entry.update({
            "null_p50": round(float(np.median(null)), 2), "null_p95": round(p95, 2), "null_max": round(float(null.max()), 2),
            "neuron_p50": round(float(np.median(m)), 2), "neuron_p90": round(float(np.percentile(m, 90)), 2),
            "above_p95": round(float((m > p95).mean()), 3), "above_max": round(float((m > null.max()).mean()), 3),
            # what above_max would be by chance alone, were neurons and controls alike: 1 / (C + 1)
            "above_max_by_chance": round(1.0 / (len(null) + 1), 3),
            "coh_neuron_+0.8": coh("neuron", "+0.8"), "coh_random_+0.8": coh("random", "+0.8"), "coh_neuron_-0.7": coh("neuron", "-0.7"),
            "coherence": {k: {"neuron": coh("neuron", k), "random": coh("random", k)}
                          for k in sorted({f"{float(d):+g}" for _, d in meta.get("design") or []}, key=float)},
            # a neuron's chance of out-massing a control (ties half): 0.5 where neurons and controls are alike
            "auc": round(float((np.searchsorted(np.sort(null), m, side="left") + np.searchsorted(np.sort(null), m, side="right")).sum()
                               / (2.0 * len(m) * len(null))), 4),
            "top": [{"n": r["neuron"], "mass": r["mass"], "words": [w for w, _, _ in r["words"][:6]]} for r in neu[:5]],
        })
        report.append(entry)
    return report


SPREAD_MODEL = "BAAI/bge-small-en-v1.5"  # the reference's sentence embedder (explorer/build_embeddings.py)


def embed_texts(texts: Sequence[str], embedder: str = "auto", dims: int = 256, vocab: int = 50000) -> Tuple[np.ndarray, str]:
    """Each text as a unit vector [n, d] float32, and what made it. "st:<model>" (or "auto" where sentence-transformers is
    installed): the sentence embedding the reference uses (`SPREAD_MODEL` for "auto"). "words" (or "auto" without it):
    the text's words (TOKEN), weighted by tf-idf (1 + log count, times log((1 + n) / (1 + df)) + 1, over the
    `vocab` words met in the most answers), each carried by its own fixed random direction of `dims` dimensions (seeded
    by the word), which keeps cosine similarity on average and needs no download. An answer with no word (code symbols
    only, or nothing) is the zero vector."""
    if embedder == "auto" or embedder.startswith("st:"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            if embedder != "auto":
                raise
        else:
            import torch

            name = embedder[3:] if embedder.startswith("st:") else SPREAD_MODEL
            enc = SentenceTransformer(name, device="cuda" if torch.cuda.is_available() else "cpu")
            emb = enc.encode(list(texts), batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
            return np.asarray(emb, np.float32), f"sentence-transformers {name}"
    elif embedder != "words":
        raise ValueError(f'The spread\'s embedders are "auto", "words" and "st:<model>", not {embedder!r}.')
    docs = [TOKEN.findall(t.lower()) for t in texts]
    df: collections.Counter = collections.Counter()
    for d in docs:
        df.update(set(d))
    n = len(docs)
    kept = [w for w, _ in df.most_common(int(vocab))]
    idf = {w: math.log((1 + n) / (1 + df[w])) + 1.0 for w in kept}
    proj = {w: np.random.default_rng(zlib.crc32(w.encode("utf-8"))).standard_normal(dims).astype(np.float32) for w in kept}
    out = np.zeros((n, dims), np.float32)
    for i, d in enumerate(docs):
        for w, k in collections.Counter(x for x in d if x in idf).items():
            out[i] += (1.0 + math.log(k)) * idf[w] * proj[w]
    out /= np.linalg.norm(out, axis=1, keepdims=True).clip(min=1e-12)
    return out, f"words (tf-idf, a random direction of {dims} per word)"


def layout_2d(emb: np.ndarray, neighbors: int = 15, method: str = "auto") -> Tuple[np.ndarray, str]:
    """Points in the unit square, near where their embeddings are near: UMAP (umap-learn, cosine, as the reference) where
    installed ("auto" or "umap"), else the first two principal components ("pca"); each axis scaled to [0, 1]."""
    if method not in ("auto", "umap", "pca"):
        raise ValueError(f'The spread\'s layouts are "auto", "umap" and "pca", not {method!r}.')
    xy = None
    name = "pca"
    if method != "pca" and len(emb) > neighbors + 1:
        try:
            import umap as umap_mod
        except ImportError:
            if method == "umap":
                raise
        else:
            xy = umap_mod.UMAP(n_neighbors=int(neighbors), min_dist=0.08, metric="cosine", random_state=0).fit_transform(emb)
            name = "umap"
    if xy is None:
        X = emb.astype(np.float64) - emb.mean(0, keepdims=True)
        if len(X) >= 2:
            _, _, vt = np.linalg.svd(X, full_matrices=False)
            xy = X @ vt[:2].T if vt.shape[0] >= 2 else np.concatenate([X @ vt[:1].T, np.zeros((len(X), 1))], 1)
        else:
            xy = np.zeros((len(X), 2))
    xy = np.asarray(xy, np.float64)
    lo, hi = xy.min(0), xy.max(0)
    xy = (xy - lo) / np.where(hi - lo > 0, hi - lo, 1.0)
    return xy, name


def build_spread(run_path: Path, embedder: str = "auto", chars: int = 600, neighbors: int = 15, layout: str = "auto") -> Path:
    """The run's answers laid out by what they say (the reference's embeddings view): each answer's first `chars`
    characters embedded (`embed_texts`) and placed in the unit square (`layout_2d`), written beside the run as
    `.spread.json`, parallel arrays x, y, row (the answer's line in the run), kind, neuron, prompt, dose, and what
    embedded and laid them out."""
    texts: List[str] = []
    cols: Dict[str, List[Any]] = {"kind": [], "neuron": [], "prompt": [], "dose": []}
    for r in read_rows(Path(run_path), tolerate_tail=True):  # only the first `chars` of each answer is held
        texts.append(str(r["text"])[: int(chars)])
        cols["kind"].append(r["kind"])
        cols["neuron"].append(int(r["neuron"]))
        cols["prompt"].append(r["prompt"])
        cols["dose"].append(float(r["dose"]))
    emb, ename = embed_texts(texts, embedder)
    xy, lname = layout_2d(emb, neighbors, layout)
    out = {
        "x": [round(float(v), 4) for v in xy[:, 0]], "y": [round(float(v), 4) for v in xy[:, 1]], "row": list(range(len(texts))),
        **cols, "embedder": ename, "layout": lname, "chars": int(chars),
    }
    path = spread_path_of(Path(run_path))
    write_json_atomic(path, out, indent=None)
    return path

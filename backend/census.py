"""The census (`backend/census.py`): every neuron of a layer pushed on a few fixed questions, each answer kept.

After the steering sweep of lyramakesmusic/neuron-explorer (`sweep/steer_sweep.py`, `sweep/rank.py`, `sweep/shift.py`),
which labels each neuron by what the model writes when it is pushed, beside what it writes unpushed and what it writes
when a random direction of the same size is pushed instead. "Sweep" already names a sighting in this instrument, so
the steering sweep is called a census here. scripts/census_run.py takes one (on the GPU); this module holds what the
runs, the ranking, the server and the checks share: the questions, the designs, the push sizes, the ranking and its
null, and how a layer's run is found and read.

A push here is the atlas's "add" push (backend/atlas_ops.py): `dose * p50_l * u`, added to the stream leaving layer l
(resid_post) at every word-piece but the start mark, with u the neuron's write direction as the stream receives it,
unit(w_post (.) W_down[:, n]), and p50_l the median length of that stream over the reference texts. So a dose is a share
of the stream's own typical length, the reference's `--dose-unit rel`, and every neuron and every control gets the same
push size. Gemma 4 differs from the reference's model in one way that matters here: its feed-forward output passes a
norm (post_feedforward_layernorm) and the layer's `layer_scalar` before it joins the stream, so turning the neuron's own
activation up is squashed by that norm (erratum E47); the push is added after both, along the direction the neuron
writes, which is what the reference's push at `mlp_out` amounted to on a model without them (its research log, "What
the hook can and can't express").

Rows (one JSON line per answer) keep: layer, kind ("neuron" | "random" | "baseline"), neuron (the control's own index
for "random", -1 for a baseline), prompt, dose, push (the length added, `dose * p50`), rel_push (push / p50), text,
finish ("stop" at an end mark | "length"), n_gen, token_ids, logprob (each written piece's own log-probability),
top_ids, top_lps (the five likeliest at each step), and, for a control, the seed its direction was drawn from.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# The reference's four questions (sweep/steer_sweep.py PROMPTS), asked as a chat turn with thinking off.
PROMPTS: Dict[str, str] = {
    "who": "Who are you?",
    "snake": "Write a snake game in Python.",
    "orange": "Describe the color orange.",
    "free": "Write whatever you like.",
}

# Doses as shares of the stream's median length leaving the layer (the reference's "main_rel": its research log found
# the coherent window near 0.5-0.8 of the stream, and +3 reads the direction out as wreckage), and a ladder for
# finding where effects begin and text breaks on this model before a full run ("calib_rel").
DESIGNS: Dict[str, List[Tuple[str, float]]] = {
    "main_rel": [(p, d) for p in ("who", "snake", "orange") for d in (-0.7, 0.5, 0.8)] + [("free", 3.0)],
    "calib_rel": [(p, s * d) for p in ("who", "orange", "free") for d in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0) for s in (-1.0, 1.0)],
}
DEFAULT_DESIGN = "main_rel"
CALIBRATION_DESIGNS = ("calib_rel",)

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


def compute_units(rows: Sequence[Dict[str, Any]], max_rel: float = 1.6, top_words: int = 12) -> List[Dict[str, Any]]:
    """One record per pushed unit (a neuron or a random control), strongest first, as sweep/rank.py computes them:
    - words: the words over-represented in its answers against the average unit, Poisson z = (count - mean) /
      sqrt(mean + 0.5), counted once per answer, over its answers pushed by at most `max_rel` of the stream, leaving out
      every word the unpushed answers use; [word, count, z], those met at least twice;
    - z: its strongest word's z; mass: the summed positive z of its top five words (the sort key);
    - null_pct: the share of random controls with a lower mass (1.0: above every control; None without controls);
    - per_prompt: for its strongest five words, which questions they appear on;
    - dose: per dose, the mean push (rel_push), the share of distinct 4-grams, the share of answers cut off at the
      length limit, and the mean log-probability of the pieces it wrote."""
    base_words: set = set().union(*(words(r["text"]) for r in rows if r["kind"] == "baseline")) if rows else set()
    units: Dict[Tuple[str, int], List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        if r["kind"] != "baseline":
            units[(r["kind"], int(r["neuron"]))].append(r)

    counts: Dict[Tuple[str, int], collections.Counter] = {}
    for u, rs in units.items():
        c: collections.Counter = collections.Counter()
        for r in rs:
            if float(r["rel_push"]) <= max_rel:
                c.update(words(r["text"]) - base_words)
        counts[u] = c
    total: collections.Counter = collections.Counter()
    for c in counts.values():
        total.update(c)
    mean = {w: v / max(len(counts), 1) for w, v in total.items()}

    records: List[Dict[str, Any]] = []
    for u, rs in units.items():
        c = counts[u]
        ranked = sorted(
            (((cnt - mean[w]) / np.sqrt(mean[w] + 0.5), w, cnt) for w, cnt in c.items() if cnt >= 2), reverse=True
        )[:top_words]
        per_prompt = {w: sorted({r["prompt"] for r in rs if w in words(r["text"])}) for _, w, _ in ranked[:5]}
        dose: Dict[str, Dict[str, float]] = {}
        for d in sorted({float(r["dose"]) for r in rs}):
            ds = [r for r in rs if float(r["dose"]) == d]
            dose[f"{d:+g}"] = {
                "rel_push": round(float(np.mean([float(r["rel_push"]) for r in ds])), 3),
                "distinct4": round(float(np.mean([distinct4(r["token_ids"]) for r in ds])), 3),
                "capped": round(float(np.mean([r["finish"] == "length" for r in ds])), 3),
                "logprob": round(float(np.mean([np.mean(r["logprob"]) if r["logprob"] else 0.0 for r in ds])), 3),
            }
        records.append(
            {
                "kind": u[0],
                "neuron": u[1],
                "layer": int(rs[0]["layer"]),
                "z": round(float(ranked[0][0]), 3) if ranked else 0.0,
                "mass": round(float(sum(max(z, 0.0) for z, _, _ in ranked[:5])), 3),
                "words": [[w, int(cnt), round(float(z), 2)] for z, w, cnt in ranked],
                "per_prompt": per_prompt,
                "dose": dose,
            }
        )

    null = np.sort([r["mass"] for r in records if r["kind"] == "random"])
    for r in records:
        r["null_pct"] = round(float(np.searchsorted(null, r["mass"]) / len(null)), 4) if len(null) else None
    records.sort(key=lambda r: -r["mass"])
    return records


def shift_scores(rows: Sequence[Dict[str, Any]], max_rel: float = 1.6, k: int = 5) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """The word-shift score of every unit against the random controls of the same run (sweep/shift.py): every word
    counts, the unpushed answers' too, so a unit that moves common words ("however", "might") scores beside one that
    brings new ones. Per unit and word: presence count over its answers pushed by at most `max_rel`, z = (count - mu) /
    sqrt(mu + 0.5), mu the mean count over the controls (leaving the unit out when it is a control itself); shift = the
    summed |z| of its k largest rises and k largest falls."""
    units: Dict[Tuple[str, int], collections.Counter] = collections.defaultdict(collections.Counter)
    for r in rows:
        if r["kind"] == "baseline" or float(r["rel_push"]) > max_rel:
            continue
        units[(r["kind"], int(r["neuron"]))].update(words(r["text"]))
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
            zs.append(((c[w] - mu) / np.sqrt(mu + 0.5), w))
        zs.sort()
        up = [(w, round(float(z), 2)) for z, w in zs[::-1][:k] if z > 0]
        down = [(w, round(float(z), 2)) for z, w in zs[:k] if z < 0]
        out[u] = {"shift": round(sum(z for _, z in up) - sum(z for _, z in down), 3), "up": up, "down": down}
    return out


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


def layer_runs(runs_dir: Optional[Path] = None) -> Dict[int, Path]:
    """layer -> the path of its census run: a main_rel run first, then the largest; calibration runs left out (as the
    reference's `layer_runs`)."""
    runs_dir = Path(runs_dir) if runs_dir is not None else census_dir()
    best: Dict[int, Tuple[Path, Tuple[bool, int]]] = {}
    for meta_path in glob.glob(str(runs_dir / "*.meta.json")):
        try:
            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if meta.get("design_name") in CALIBRATION_DESIGNS:
            continue
        jsonl = Path(meta_path[: -len(".meta.json")] + ".jsonl")
        if not jsonl.exists() or not isinstance(meta.get("layer"), int):
            continue
        key = (meta.get("design_name") == DEFAULT_DESIGN, jsonl.stat().st_size)
        if meta["layer"] not in best or key > best[meta["layer"]][1]:
            best[meta["layer"]] = (jsonl, key)
    return {layer: path for layer, (path, _) in best.items()}


def read_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def units_path_of(run_path: Path) -> Path:
    return Path(str(run_path)[: -len(".jsonl")] + ".units.jsonl")


def meta_path_of(run_path: Path) -> Path:
    return Path(str(run_path)[: -len(".jsonl")] + ".meta.json")


def spread_path_of(run_path: Path) -> Path:
    return Path(str(run_path)[: -len(".jsonl")] + ".spread.json")


def write_units(run_path: Path, max_rel: float = 1.6, top_words: int = 12) -> Path:
    """Rank a run and write its `.units.jsonl` beside it (atomically), from every row's token ids."""
    out = units_path_of(run_path)
    recs = compute_units(list(read_rows(run_path)), max_rel=max_rel, top_words=top_words)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(u, ensure_ascii=False) + "\n" for u in recs)
    os.replace(tmp, out)
    return out

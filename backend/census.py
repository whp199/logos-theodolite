"""The census (`backend/census.py`): every neuron of a layer pushed on a few fixed questions, each answer kept.

After the steering sweep of lyramakesmusic/neuron-explorer (`sweep/steer_sweep.py`, `sweep/rank.py`, `sweep/shift.py`),
which labels each neuron by what the model writes when it is pushed, beside what it writes unpushed and what it writes
when a random direction of the same size is pushed instead. "Sweep" already names a sighting in this instrument, so
the steering sweep is called a census here. This module holds what the runs, the ranking, the server and the checks
share: the questions, the designs, the push sizes, taking a run (`take_census`, which scripts/census_run.py calls on
the GPU), the ranking and its null (`compute_units`, `shift_scores`), how a layer's run is found and read, and the
tools over finished runs (`merge_runs`, `depth_report`, `build_spread`: scripts/census_tools.py).

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
import math
import os
import random
import re
import zlib
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


def layer_runs(runs_dir: Optional[Path] = None, model: Optional[str] = None) -> Dict[int, Path]:
    """layer -> the path of its census run: a main_rel run first, then the largest; calibration runs left out (as the
    reference's `layer_runs`), and, given the model read here, runs taken on another model."""
    runs_dir = Path(runs_dir) if runs_dir is not None else census_dir()
    best: Dict[int, Tuple[Path, Tuple[bool, int]]] = {}
    for meta_path in glob.glob(str(runs_dir / "*.meta.json")):
        try:
            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(meta, dict) or meta.get("design_name") in CALIBRATION_DESIGNS:
            continue
        if model is not None and meta.get("model") is not None and meta.get("model") != model:
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
    """Rank a run and write its `.units.jsonl` beside it (atomically): `compute_units`' records, each with its word-shift
    score against the controls (`shift_scores`: shift, shift_up, shift_down; None without controls)."""
    out = units_path_of(run_path)
    rows = list(read_rows(run_path))
    recs = compute_units(rows, max_rel=max_rel, top_words=top_words)
    shifts = shift_scores(rows, max_rel=max_rel)
    for r in recs:
        sh = shifts.get((r["kind"], int(r["neuron"])), {"shift": None, "up": [], "down": []})
        r["shift"], r["shift_up"], r["shift_down"] = sh["shift"], sh["up"], sh["down"]
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


def control_dirs(seed: int, count: int, d_model: int) -> np.ndarray:
    """The random controls' directions: `count` unit vectors of the stream, isotropic, drawn from the seed ([count, D]
    float32). A control is pushed exactly as a neuron is, `dose * p50 * u`, so it asks how often a theme turns up when
    the direction pushed means nothing (the reference rotates a neuron's down column to a random direction; with a push
    of unit direction the rotation is the draw itself)."""
    v = np.random.default_rng(int(seed)).standard_normal((int(count), int(d_model)))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def unit_order(neurons: Sequence[int], controls: int) -> List[Tuple[str, int]]:
    """The neurons with the controls spread evenly among them, so any prefix of a run has its share of controls."""
    units: List[Tuple[str, int]] = [("neuron", int(n)) for n in neurons]
    for k in range(int(controls)):
        units.insert(round((k + 1) * len(units) / (int(controls) + 1)), ("random", k))
    return units


def census_jobs(neurons: Sequence[int], controls: int, design: Sequence[Tuple[str, float]]) -> List[Dict[str, Any]]:
    """Every answer a run writes, in order: one unpushed answer per question, then each unit's (question, dose)."""
    jobs = [{"kind": "baseline", "neuron": -1, "prompt": p, "dose": 0.0} for p in PROMPTS]
    jobs += [{"kind": kind, "neuron": n, "prompt": p, "dose": float(d)} for kind, n in unit_order(neurons, controls) for p, d in design]
    return jobs


# what must agree for a run to be continued (a rerun of the same command resumes; anything else is a different run)
RESUME_KEYS = ("model", "layer", "design_name", "design", "prompt_ids", "max_new", "top_k", "seed", "random_controls", "push")


def settings_differ(old: Dict[str, Any], new: Dict[str, Any]) -> List[str]:
    """The settings two runs' metas disagree on (RESUME_KEYS, and the push's unit p50 beyond float noise)."""
    differ = [k for k in RESUME_KEYS if json.dumps(old.get(k)) != json.dumps(new.get(k))]
    p_old, p_new = old.get("p50"), new.get("p50")
    if p_old is not None and p_new is not None and abs(float(p_old) - float(p_new)) > 1e-4 * max(1.0, abs(float(p_new))):
        differ.append("p50")
    return differ


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
    log: Any = print,
    cancel: Any = None,
) -> Dict[str, Any]:
    """Take (or continue) a census of layer `layer` into `out_path` (JSON lines; its `.meta.json` beside it): each
    neuron and each control pushed on every (question, dose) of the design, and the questions unpushed, one greedy
    answer each of at most `max_new` word-pieces, `batch` answers at a time (`atlas_ops.write_rows`, each row its own
    push; the rows of a batch share their question, so none is padded). Rows already in the file are skipped; a batch
    is written whole or not at all (`cancel()` stops between steps). Returns counts."""
    import time

    import torch

    from backend import atlas_ops as ao
    from backend.causal_check import language_model

    if design_name not in DESIGNS:
        raise ValueError(f"The census's designs are {', '.join(DESIGNS)}, not {design_name!r}.")
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
    prompt_ids = {k: rt.tokenize_prompt(rt.compose_prompt(v, chat=True)) for k, v in PROMPTS.items()}
    stop = rt.end_of_turn_ids()
    out_path = Path(out_path)
    meta = {
        "model": getattr(rt, "model_id", None), "layer": int(layer), "prompts": PROMPTS, "prompt_ids": prompt_ids,
        "design_name": design_name, "design": [[p, d] for p, d in design], "max_new": int(max_new), "top_k": int(top_k),
        "seed": int(seed), "random_controls": int(controls), "neurons": neurons_spec or f"{len(neurons)} neurons",
        "push": "add: dose x p50 along unit(w_post * W_down[:, n]) (a control: a random unit direction), at the stream "
                "leaving the layer, every word-piece but the start mark",
        "p50": float(p50), "p50_source": p50_source, "stop_ids": stop, "greedy": True, "chat": "thinking off",
        "lens": (rt.lens_card() or {}).get("id") if hasattr(rt, "lens_card") else None,
        "stand_in": bool(getattr(rt, "atlas_stand_in", False)),
    }
    meta_path = meta_path_of(out_path)
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        differ = settings_differ(old, meta)
        if differ:
            raise ValueError(f"{out_path.name} was taken with other settings ({', '.join(differ)}); name another run "
                             "(--run) or remove it.")
        meta["started_at"] = old.get("started_at")
        meta["neurons"] = old.get("neurons") if neurons_spec in ("", old.get("neurons")) else f"{old.get('neurons')} + {neurons_spec}"
    meta["started_at"] = meta.get("started_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json_atomic(meta_path, meta)

    done = {row_key(r) for r in read_rows(out_path)} if out_path.exists() else set()
    ctrl = torch.from_numpy(control_dirs(seed, controls, D))
    jobs = [j for j in census_jobs(neurons, controls, design) if row_key(j) not in done]
    log(f"layer {layer}: {len(neurons)} neurons + {controls} controls, {len(jobs)} answers to write ({len(done)} already "
        f"written), push unit p50 {p50:.2f} ({p50_source})")
    softcap = ao.softcap_of(rt.hf_model)
    written, pieces, t0 = 0, 0, time.time()

    def write_batch(bj: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        B = len(bj)
        delta = torch.zeros(B, D, dtype=torch.float32, device=w.device)
        rows_n = [i for i, j in enumerate(bj) if j["kind"] == "neuron"]
        if rows_n:
            u = w.write_dirs(layer, [bj[i]["neuron"] for i in rows_n])
            for k, i in enumerate(rows_n):
                delta[i] = u[k] * (bj[i]["dose"] * p50)
        for i, j in enumerate(bj):
            if j["kind"] == "random":
                delta[i] = ctrl[j["neuron"]].to(w.device) * (j["dose"] * p50)
        iv = ao.Interventions(rows=B, adds={int(layer): delta}) if bool(delta.abs().sum() > 0) else None
        gen: List[List[int]] = [[] for _ in range(B)]
        lps: List[List[float]] = [[] for _ in range(B)]
        tids: List[List[List[int]]] = [[] for _ in range(B)]
        tlps: List[List[List[float]]] = [[] for _ in range(B)]
        ended = [False] * B
        for st in ao.write_rows(lm, rt.lm_head_weight, [prompt_ids[bj[0]["prompt"]]] * B, iv, max_new=max_new, stop_ids=stop,
                                bos_id=tok.bos_token_id, spare_bos=True, softcap=softcap, top_k=top_k, cancel=cancel):
            gen[st.row].append(st.token_id)
            lps[st.row].append(round(st.logprob, 4))
            tids[st.row].append(st.top_ids)
            tlps[st.row].append(st.top_lps)
            ended[st.row] = ended[st.row] or st.end
        if cancel is not None and cancel():
            return None
        out = []
        for i, j in enumerate(bj):
            ids = gen[i][:-1] if ended[i] else gen[i]
            rec = {"layer": int(layer), "kind": j["kind"], "neuron": int(j["neuron"]), "prompt": j["prompt"], "dose": j["dose"],
                   "push": round(j["dose"] * p50, 4), "rel_push": abs(j["dose"]) if j["kind"] != "baseline" else 0.0,
                   "text": tok.decode(ids, skip_special_tokens=True), "finish": "stop" if ended[i] else "length",
                   "n_gen": len(gen[i]), "token_ids": gen[i], "logprob": lps[i], "top_ids": tids[i], "top_lps": tlps[i]}
            if j["kind"] == "random":
                rec["seed"] = int(seed)
            out.append(rec)
        return out

    with open(out_path, "a", encoding="utf-8") as f:
        # a chunk of the run at a time, its answers grouped by question so a batch shares its question
        chunk = max(int(batch), 1) * 16
        for c0 in range(0, len(jobs), chunk):
            part = jobs[c0 : c0 + chunk]
            for p in PROMPTS:
                js = [j for j in part if j["prompt"] == p]
                for b0 in range(0, len(js), max(int(batch), 1)):
                    recs = write_batch(js[b0 : b0 + max(int(batch), 1)])
                    if recs is None:
                        log(f"stopped: {written} answers written; the same command continues the run")
                        return {"written": written, "skipped": len(done), "stopped": True}
                    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
                    f.flush()
                    written += len(recs)
                    pieces += sum(r["n_gen"] for r in recs)
            dt = time.time() - t0
            log(f"{min(c0 + chunk, len(jobs))}/{len(jobs)} answers, {pieces / max(dt, 1e-9):,.0f} pieces/s, {written / max(dt, 1e-9):.2f} answers/s")
    return {"written": written, "skipped": len(done), "stopped": False}


# ------------------------------------------------------------------------------------- shards, depth and the spread


def merge_runs(inputs: Sequence[Path], out: Path) -> Tuple[int, int]:
    """Join runs of one layer taken in shards into one run, a row per (kind, neuron, prompt, dose), the first met kept
    (sweep/merge_runs.py); its meta is the first input's with the shards named. Runs taken with other settings are
    refused. Returns (rows read, rows kept)."""
    inputs = [Path(p) for p in inputs]
    metas = [json.loads(meta_path_of(p).read_text(encoding="utf-8")) for p in inputs]
    for p, m in zip(inputs[1:], metas[1:]):
        differ = settings_differ(metas[0], m)
        if differ:
            raise ValueError(f"{p.name} was taken with other settings than {inputs[0].name} ({', '.join(differ)}).")
    seen: set = set()
    n_in = n_out = 0
    out = Path(out)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for p in inputs:
            for r in read_rows(p):
                n_in += 1
                k = row_key(r)
                if k in seen:
                    continue
                seen.add(k)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_out += 1
    os.replace(tmp, out)
    meta = dict(metas[0])
    meta["shards"] = [p.name for p in inputs]
    write_json_atomic(meta_path_of(out), meta)
    return n_in, n_out


def depth_report(runs: Dict[int, Path], neurons: int = 384, controls: int = 64) -> List[Dict[str, Any]]:
    """Per layer with a census, how far its neurons move the model's words beyond what random directions do
    (sweep/depth_report.py): the controls' mass at its median, 95th percentile and maximum, the neurons' at its median
    and 90th percentile, the share of neurons above the controls' 95th percentile and above all of them, the answers'
    coherence at +0.8 (neurons and controls) and -0.7 (neurons), and the five strongest neurons. A run holding more
    than `neurons` neurons or `controls` controls is subsampled to that pool (seeded by the layer), so the mass, which
    grows with the pool, compares across layers."""
    report: List[Dict[str, Any]] = []
    for l, path in sorted(runs.items()):
        rows = list(read_rows(Path(path)))
        nn = sorted({int(r["neuron"]) for r in rows if r["kind"] == "neuron"})
        cc = sorted({int(r["neuron"]) for r in rows if r["kind"] == "random"})
        sub = len(nn) > neurons or len(cc) > controls
        if sub:
            rng = random.Random(int(l))
            keep = {("neuron", n) for n in (rng.sample(nn, neurons) if len(nn) > neurons else nn)}
            keep |= {("random", n) for n in (rng.sample(cc, controls) if len(cc) > controls else cc)}
            rows = [r for r in rows if r["kind"] == "baseline" or (r["kind"], int(r["neuron"])) in keep]
        recs = compute_units(rows)
        null = np.array([r["mass"] for r in recs if r["kind"] == "random"], np.float64)
        neu = [r for r in recs if r["kind"] == "neuron"]
        m = np.array([r["mass"] for r in neu], np.float64)
        entry: Dict[str, Any] = {"layer": int(l), "run": Path(path).name, "subsampled": sub, "n_neurons": len(m), "n_controls": len(null)}
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
            "coh_neuron_+0.8": coh("neuron", "+0.8"), "coh_random_+0.8": coh("random", "+0.8"), "coh_neuron_-0.7": coh("neuron", "-0.7"),
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
    rows = list(read_rows(Path(run_path)))
    emb, ename = embed_texts([str(r["text"])[: int(chars)] for r in rows], embedder)
    xy, lname = layout_2d(emb, neighbors, layout)
    out = {
        "x": [round(float(v), 4) for v in xy[:, 0]], "y": [round(float(v), 4) for v in xy[:, 1]], "row": list(range(len(rows))),
        "kind": [r["kind"] for r in rows], "neuron": [int(r["neuron"]) for r in rows], "prompt": [r["prompt"] for r in rows],
        "dose": [float(r["dose"]) for r in rows], "embedder": ename, "layout": lname, "chars": int(chars),
    }
    path = spread_path_of(Path(run_path))
    write_json_atomic(path, out, indent=None)
    return path

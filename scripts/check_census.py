#!/usr/bin/env python3
"""A CPU-only check of the census (backend/census.py, scripts/census_run.py, scripts/census_tools.py) on the tiny random
Gemma 4 of scripts/atlas_stand_in.py:
1. the ranking, on answers whose numbers are worked out by hand: words (three letters or more, any script, once per
   answer), distinct 4-grams, each unit's words over the average unit (Poisson z), its mass, the controls' null
   (null_pct, p, the Benjamini-Hochberg q, the chance count N / (C + 1)), answers pushed past max_rel left out; the
   word-shift score against the controls, a control left out of its own null; the slim rows a whole layer is ranked
   from give the same ranking; a half-written last line is left out; the controls' directions, unit(w_post * r);
2. a run taken on the stand-in: every answer is the greedy answer written alone under the same push (a neuron along its
   write direction, a control along its drawn direction, dose x p50; a baseline unpushed), with its push recorded, and
   the meta records the controls' shape, the doses' source and how loudly the next layer reads controls and neurons;
3. continuing: a run cut short and rerun ends with the same answers; a rerun with nothing to do writes nothing; more
   neurons added write only theirs; a run taken with other settings is refused in plain words; a run stopped between
   steps keeps only whole batches and finishes as the uninterrupted one;
4. finding a layer's run: main_rel first, calibration runs, runs of another model and runs taken with the old
   isotropic controls left out (the last with the reason);
5. merge, the depth report (subsampled to a common pool) and the spread (the words embedder, laid out in the unit square);
6. the command lines: census_run.py --stand-in into the stand-in's own folder, then census_tools.py rank, depth and
   spread on it;
7. the dose window: where an answer first parts from the unpushed one, the curves per dose and kind and the verdict per
   sign on answers worked out by hand, the write ceilings read from a W4A16 checkpoint in two files, the Hugging Face
   cache found through huggingface_hub, and a stand-in calib_rel run read back by census_tools.py curves, with the
   ceilings read from the stand-in's weights for answers that keep none;
8. the dose in each neuron's own ceiling: each push measured as write_rows receives it (dose x the neuron's write
   ceiling; a control, dose x its paired neuron's), each row's ceiling, push and real share of the stream, a unit
   change refused, an older meta read as the stream unit, a ceiling run ranked and in the depth report, and a stand-in
   calib_ceil run read back by census_tools.py curves;
9. when the 31B's checkpoint is in the Hugging Face cache: on its own norms and down columns (CPU, a few tensors), the
   next layer reads the census's controls about as loudly as the layer's neurons, and an isotropic control far louder.

Run: .venv/bin/python -I scripts/check_census.py   (exits 0 when every check holds)
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch

import backend.atlas_ops as ao
from atlas_stand_in import stand_in
from backend import census as cz
from backend.causal_check import language_model
from backend.mcp_scientist import _study_ids

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


def row(kind: str, n: int, text: str, dose: float = 0.5, prompt: str = "who", ids: Any = None) -> Dict[str, Any]:
    return {"layer": 3, "kind": kind, "neuron": n, "prompt": prompt, "dose": dose, "push": dose, "rel_push": abs(dose) if kind != "baseline" else 0.0,
            "text": text, "finish": "stop", "n_gen": 3, "token_ids": ids or [1, 2, 3], "logprob": [-1.0, -2.0, -3.0],
            "top_ids": [], "top_lps": []}


def real_read_gain(layers: tuple = (45, 57), sample: int = 256) -> Any:
    """{layer: (shaped ratio, isotropic ratio)}: how loudly the next layer's input norm reads the census's controls
    against the layer's neurons, on the 31B's own norms and dequantised down columns read from the local checkpoint
    (CPU, a few tensors); None when the checkpoint is not cached here."""
    import glob

    from backend.engine_runtime import dequantize_w4a16_cols

    hits = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-4-31B-it-qat-w4a16-ct/snapshots/*/model.safetensors"))
    if not hits:
        return None
    from safetensors import safe_open

    P = "model.language_model.layers."
    out = {}
    with safe_open(hits[0], "pt") as f:
        for l in layers:
            w_post = f.get_tensor(P + f"{l}.post_feedforward_layernorm.weight").float()
            w_in = f.get_tensor(P + f"{l + 1}.input_layernorm.weight").float().numpy()
            idx = torch.from_numpy(np.random.default_rng(l).choice(21504, sample, replace=False))
            U = dequantize_w4a16_cols(f.get_tensor(P + f"{l}.mlp.down_proj.weight_packed"), f.get_tensor(P + f"{l}.mlp.down_proj.weight_scale"), idx).float()
            U = (U * w_post).numpy()
            U /= np.linalg.norm(U, axis=1, keepdims=True)
            iso = np.random.default_rng(0).standard_normal((sample, U.shape[1]))
            iso /= np.linalg.norm(iso, axis=1, keepdims=True)
            fair = cz.read_gain(U, cz.control_dirs(0, sample, w_post.numpy()), w_in)["ratio"]
            out[l] = (fair, cz.read_gain(U, iso, w_in)["ratio"])
    return out


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="census-check-"))
    try:
        return run_checks(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_checks(tmp: Path) -> int:
    # ------------------------------------------------------------------------------------------ 1. the ranking by hand
    before = len(failures)
    check(cz.words("Hello, naïve café! It's 3am 你好世界") == {"hello", "naïve", "café", "你好世界"}, f"words: {cz.words('Hello, naïve café! It' + chr(39) + 's 3am 你好世界')}")
    check(cz.distinct4([1, 2, 3, 4, 1, 2, 3, 4]) == 0.8 and cz.distinct4([1, 2]) == 1.0, "distinct 4-grams")
    rows = [row("baseline", -1, "plain text"),
            row("neuron", 0, "apple apple"), row("neuron", 0, "apple pear"), row("neuron", 0, "kiwi"),
            row("neuron", 1, "kiwi"), row("neuron", 1, "kiwi plain"), row("neuron", 1, "plum"),
            row("neuron", 1, "kiwi kiwi kiwi", dose=3.0)]  # pushed past max_rel: left out of the counts
    recs = {u["neuron"]: u for u in cz.compute_units(rows)}
    # counts once per answer, baseline words left out: n0 apple 2, pear 1, kiwi 1; n1 kiwi 2, plum 1
    # means over the two units: apple 1, pear 0.5, kiwi 1.5, plum 0.5
    z_apple = (2 - 1) / math.sqrt(1 + 0.5)
    z_kiwi = (2 - 1.5) / math.sqrt(1.5 + 0.5)
    check(recs[0]["words"] == [["apple", 2, round(z_apple, 2)]] and recs[0]["z"] == round(z_apple, 3) and recs[0]["mass"] == round(z_apple, 3),
          f"compute_units n0: {recs[0]['words']} z {recs[0]['z']} mass {recs[0]['mass']}")
    check(recs[1]["words"] == [["kiwi", 2, round(z_kiwi, 2)]] and recs[1]["mass"] == round(z_kiwi, 3), f"compute_units n1: {recs[1]['words']}")
    check([u["neuron"] for u in cz.compute_units(rows)] == [0, 1] and recs[0]["null_pct"] is None, "sorted by mass; no null without controls")
    check(recs[1]["dose"]["+3"]["rel_push"] == 3.0 and recs[1]["dose"]["+0.5"]["capped"] == 0.0 and recs[0]["per_prompt"] == {"apple": ["who"]},
          f"per dose and per prompt: {recs[1]['dose']}")
    rows2 = rows + [row("random", 0, "apple"), row("random", 0, "fig fig"), row("random", 1, "pear"), row("random", 1, "date"),
                    row("random", 1, "date")]
    recs2 = {(u["kind"], u["neuron"]): u for u in cz.compute_units(rows2)}
    null = sorted(u["mass"] for k_, u in recs2.items() if k_[0] == "random")
    check(all(u["null_pct"] == round(float(np.searchsorted(null, u["mass"]) / len(null)), 4) for u in recs2.values()),
          "null_pct: the share of controls with a lower mass")
    sh = cz.shift_scores([row("random", 0, "apple"), row("random", 1, "pear"), row("neuron", 0, "apple"), row("neuron", 0, "apple tree"),
                          row("neuron", 0, "apple", dose=3.0)])
    # n0 against both controls: mu(apple) = mu(pear) = 1/2; apple 2 -> (2 - .5)/1 = 1.5; pear 0 -> -0.5; tree: mu 0, count 1, left out
    check(sh[("neuron", 0)] == {"shift": 2.0, "up": [("apple", 1.5)], "down": [("pear", -0.5)]}, f"shift n0: {sh[('neuron', 0)]}")
    # r0 against r1 alone (left out of its own null): apple mu 0, count 1, left out; pear mu 1, count 0 -> -1/sqrt(1.5)
    check(sh[("random", 0)] == {"shift": round(1 / math.sqrt(1.5), 2), "up": [], "down": [("pear", round(-1 / math.sqrt(1.5), 2))]},
          f"shift r0: {sh[('random', 0)]}")
    check(cz.shift_scores([row("neuron", 0, "apple")])[("neuron", 0)]["shift"] is None, "shift: None without controls")
    check(cz.parse_neurons("1,5,9-11", 20) == [1, 5, 9, 10, 11] and cz.parse_neurons("all", 3) == [0, 1, 2]
          and cz.parse_neurons("random:4", 50, 0) == cz.parse_neurons("random:4", 50, 0) and len(cz.parse_neurons("random:4", 50)) == 4,
          "parse_neurons")
    try:
        cz.parse_neurons("0-200", 128)
        check(False, "parse_neurons: neurons that do not exist are refused")
    except ValueError as exc:
        check("count from 0 to 127" in str(exc), f"parse_neurons refusal: {exc}")
    u = cz.unit_order(list(range(6)), 2)
    check(u == [("neuron", 0), ("neuron", 1), ("random", 0), ("neuron", 2), ("neuron", 3), ("random", 1), ("neuron", 4), ("neuron", 5)],
          f"unit_order spreads the controls: {u}")
    shape = np.linspace(0.5, 2.0, 64)
    shape[:5] = 0.0
    cd = cz.control_dirs(3, 4, shape)
    raw = np.random.default_rng(3).standard_normal((4, 64)) * shape
    check(cd.shape == (4, 64) and np.allclose(np.linalg.norm(cd, axis=1), 1.0) and np.array_equal(cd, cz.control_dirs(3, 4, shape))
          and np.allclose(cd, raw / np.linalg.norm(raw, axis=1, keepdims=True), atol=1e-6) and not cd[:, :5].any(),
          "control_dirs: unit(w_post * r), reproducible from the seed, nothing where w_post is zero")
    # p and q, by hand: controls' masses and each neuron's p = (1 + controls at least as strong) / (C + 1)
    masses = {k_: u["mass"] for k_, u in recs2.items()}
    C = 2
    ctl_m = [masses[("random", 0)], masses[("random", 1)]]
    ps = {k_: (1 + sum(c >= m for c in ctl_m)) / (C + 1) for k_, m in masses.items() if k_[0] == "neuron"}
    check(all(recs2[k_]["p"] == round(p_, 5) for k_, p_ in ps.items()), f"p: {[(k_, recs2[k_]['p'], p_) for k_, p_ in ps.items()]}")
    order = sorted(ps, key=ps.get)
    bh = {order[i]: min(ps[order[j]] * len(order) / (j + 1) for j in range(i, len(order))) for i in range(len(order))}
    check(all(recs2[k_]["q"] == round(min(1.0, q_), 5) for k_, q_ in bh.items()), "q: Benjamini-Hochberg over the neurons")
    check(recs2[("random", 0)]["p"] == round((1 + (ctl_m[1] >= ctl_m[0])) / 2, 5) and recs2[("random", 0)]["q"] is None,
          "a control's p against the other controls; no q for a control")
    check(cz.chance_above_all(21504, 256) == 83.7 and cz.chance_above_all(5, 0) is None, "the chance count: N / (C + 1)")
    slim = cz.SlimRows(rows2)
    check(cz.compute_units(slim) == cz.compute_units(rows2) and cz.shift_scores(slim) == cz.shift_scores(rows2)
          and "text" not in slim[0] and "token_ids" not in slim[0], "the slim rows rank exactly as the whole ones, holding no text")
    tail = tmp / "tail.jsonl"
    tail.write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[1]) + "\n" + json.dumps(rows[2])[:20])
    check(len(list(cz.read_rows(tail, tolerate_tail=True))) == 2 and len(cz.read_slim(tail)) == 2, "a half-written last line is left out")
    try:
        list(cz.read_rows(tail))
        check(False, "a half-written last line raises unless tolerated")
    except ValueError:
        pass
    check(cz.drop_partial_tail(tail) and tail.read_text().endswith("\n") and len(list(cz.read_rows(tail))) == 2 and not cz.drop_partial_tail(tail),
          "drop_partial_tail cuts it off, once")
    section("the ranking, p and q, and the shift, worked by hand", before)

    # ---------------------------------------------------------------------------------- 2. a run on the stand-in
    before = len(failures)
    torch.set_grad_enabled(False)
    rt, _vault = stand_in()
    lm = language_model(rt.hf_model)
    w = ao.Weights(lm, rt)
    D = w.d_model
    tok = rt.tokenizer
    p50 = ao.stream_p50(ao.activation_extremes(lm, _study_ids(rt)))[3]
    out = tmp / "runs" / "L3.jsonl"
    out.parent.mkdir(parents=True)
    kw = dict(p50=p50, p50_source="the study's 36 texts", design_name="main_rel", controls=2, seed=3, max_new=10, top_k=5,
              batch=7, log=lambda s: None)
    res = cz.take_census(rt, 3, [0, 5, 9], out, **kw)
    got = list(cz.read_rows(out))
    check(res == {"written": 54, "skipped": 0, "stopped": False} and len(got) == 54, f"take_census: {res}, {len(got)} rows")
    check(len({cz.row_key(r) for r in got}) == 54, "every answer once")
    prompt_ids = {k: rt.tokenize_prompt(rt.compose_prompt(v, chat=True)) for k, v in cz.PROMPTS.items()}
    ctrl = cz.control_dirs(3, 2, w.norm(3, "post_feedforward").numpy())
    stop = rt.end_of_turn_ids()

    def alone(r: Dict[str, Any]) -> Dict[str, Any]:
        if r["kind"] == "baseline":
            iv = None
        else:
            u_ = w.write_dirs(3, [r["neuron"]])[0] if r["kind"] == "neuron" else torch.from_numpy(ctrl[r["neuron"]])
            iv = ao.Interventions(rows=1, adds={3: (u_ * (r["dose"] * p50))[None]})
        ids, lps, tops, end = [], [], [], False
        for st in ao.write_rows(lm, rt.lm_head_weight, [prompt_ids[r["prompt"]]], iv, max_new=10, stop_ids=stop, bos_id=tok.bos_token_id,
                                spare_bos=True, softcap=ao.softcap_of(rt.hf_model), top_k=5):
            ids.append(st.token_id)
            lps.append(st.logprob)
            tops.append(st.top_ids)
            end = end or st.end
        return {"token_ids": ids, "logprob": lps, "top_ids": tops, "finish": "stop" if end else "length",
                "text": tok.decode(ids[:-1] if end else ids, skip_special_tokens=True)}

    mism = []
    for r in got:
        a_ = alone(r)
        if (r["token_ids"] != a_["token_ids"] or r["text"] != a_["text"] or r["finish"] != a_["finish"] or r["top_ids"] != a_["top_ids"]
                or not all(abs(x - y) < 2e-3 for x, y in zip(r["logprob"], a_["logprob"]))):
            mism.append(cz.row_key(r))
    check(not mism, f"each answer is the one written alone under the same push ({len(mism)} differ: {mism[:3]})")
    check(all(r["rel_push"] == (abs(r["dose"]) if r["kind"] != "baseline" else 0.0) and r["push"] == round(r["dose"] * p50, 4) for r in got),
          "each answer's push and its share of the stream")
    check(all(r.get("seed") == 3 for r in got if r["kind"] == "random") and all("seed" not in r for r in got if r["kind"] != "random"),
          "a control's seed recorded")
    check(all(r["n_gen"] == len(r["token_ids"]) == len(r["logprob"]) == len(r["top_ids"]) and all(len(t) == 5 for t in r["top_ids"]) for r in got),
          "each step's piece, log-probability and five likeliest")
    pushed = {r["text"] for r in got if r["kind"] == "neuron" and r["dose"] == 3.0}
    base = {r["text"] for r in got if r["kind"] == "baseline"}
    check(bool(pushed - base), "a push of 3 changes what is written")
    meta = json.loads(cz.meta_path_of(out).read_text())
    check(meta["layer"] == 3 and meta["design_name"] == "main_rel" and meta["p50"] == p50 and meta["prompt_ids"] == prompt_ids
          and meta["stop_ids"] == stop and meta["model"] == rt.model_id and meta["stand_in"] is True and meta["random_controls"] == 2,
          f"the run's meta: {sorted(meta)}")
    g = meta.get("read_gain") or {}
    check(meta.get("version") == cz.CENSUS_VERSION and "w_post" in meta.get("controls", "") and meta.get("doses_from") == cz.DOSES_FROM["main_rel"]
          and g.get("ratio") is not None and 0.5 <= g["ratio"] <= 2.0, f"the meta records the controls, the doses' source and the read gain: {g}")
    D_ = w.d_model
    iso = np.random.default_rng(3).standard_normal((2, D_))
    iso /= np.linalg.norm(iso, axis=1, keepdims=True)
    check(not np.allclose(ctrl, iso.astype(np.float32), atol=1e-3), "the controls are not the old isotropic draw")
    section("a run on the stand-in: every answer as written alone, pushes recorded", before)

    # ------------------------------------------------------------------------------------------------- 3. continuing
    before = len(failures)
    lines = out.read_text().splitlines(keepends=True)
    out.write_text("".join(lines[:-9]))
    res = cz.take_census(rt, 3, [0, 5, 9], out, **kw)
    again = list(cz.read_rows(out))
    check(res["written"] == 9 and res["skipped"] == 45 and len(again) == 54, f"a run cut short continues: {res}")
    check({cz.row_key(r): r["token_ids"] for r in again} == {cz.row_key(r): r["token_ids"] for r in got}, "and ends with the same answers")
    check(cz.take_census(rt, 3, [0, 5, 9], out, **kw) == {"written": 0, "skipped": 54, "stopped": False}, "nothing left to write")
    res = cz.take_census(rt, 3, [0, 5, 9, 12], out, **kw)
    check(res["written"] == 10 and {r["neuron"] for r in cz.read_rows(out) if r["kind"] == "neuron"} == {0, 5, 9, 12}, f"a neuron added: {res}")
    try:
        cz.take_census(rt, 3, [0], out, **{**kw, "max_new": 11})
        check(False, "a run taken with other settings is refused")
    except ValueError as exc:
        check("max_new" in str(exc) and "name another run" in str(exc), f"refusal: {exc}")
    try:
        cz.take_census(rt, 3, [0], out, **{**kw, "p50": p50 * 1.5})
        check(False, "a run taken with another push unit is refused")
    except ValueError as exc:
        check("p50" in str(exc), f"refusal: {exc}")
    calls = [0]

    def stop_soon() -> bool:
        calls[0] += 1
        return calls[0] > 25

    out2 = tmp / "runs" / "L3-stopped.jsonl"
    res = cz.take_census(rt, 3, [0, 5, 9], out2, **kw, cancel=stop_soon)
    part = list(cz.read_rows(out2))
    check(res["stopped"] and 0 < len(part) < 54 and res["written"] == len(part), f"stopped between steps: {res}, {len(part)} rows")
    check(all(r["n_gen"] > 0 for r in part), "only whole answers kept")
    cz.take_census(rt, 3, [0, 5, 9], out2, **kw)
    check({cz.row_key(r): r["token_ids"] for r in cz.read_rows(out2)} == {cz.row_key(r): r["token_ids"] for r in got},
          "a stopped run finishes as the uninterrupted one")
    try:
        cz.take_census(rt, 6, [0], tmp / "runs" / "bad.jsonl", **kw)
        check(False, "a layer that does not exist is refused")
    except ValueError as exc:
        check("Layer 6 does not exist" in str(exc), f"refusal: {exc}")
    section("continuing, extending, stopping, refusing", before)

    # ------------------------------------------------------------------------------------- 4. finding a layer's run
    before = len(failures)
    runs_dir = tmp / "find"
    runs_dir.mkdir()

    def fake(name: str, layer: int, design: str, n_rows: int, model: Any = None, version: int = cz.CENSUS_VERSION) -> Path:
        pth = runs_dir / f"{name}.jsonl"
        pth.write_text("".join(json.dumps(row("neuron", i, "word")) + "\n" for i in range(n_rows)))
        m = {"layer": layer, "design_name": design, "version": version}
        if model:
            m["model"] = model
        cz.meta_path_of(pth).write_text(json.dumps(m))
        return pth

    small_main = fake("L1-main", 1, "main_rel", 2)
    fake("L1-big-other", 1, "other", 50)
    fake("L1-calib", 1, "calib_rel", 90)
    big_main = fake("L2-main", 2, "main_rel", 5, model=rt.model_id)
    fake("L2-main-bigger", 2, "main_rel", 40)
    fake("L3-elsewhere", 3, "main_rel", 5, model="google/gemma-4-31B-it")
    runs = cz.layer_runs(runs_dir)
    check(runs[1] == small_main, f"main_rel before a larger run of another design: {runs.get(1)}")
    check(runs[2] == runs_dir / "L2-main-bigger.jsonl", "among main_rel runs, the largest")
    check(3 in runs and 3 not in cz.layer_runs(runs_dir, model=rt.model_id), "a run of another model left out for this one")
    check(cz.layer_runs(runs_dir, model=rt.model_id)[2] in (big_main, runs_dir / "L2-main-bigger.jsonl"), "runs without a model are kept")
    fake("L4-old", 4, "main_rel", 9, version=1)
    refused: Dict[int, str] = {}
    check(4 not in cz.layer_runs(runs_dir, refused=refused) and "isotropic" in refused.get(4, ""),
          f"a run taken with the old isotropic controls is not read, and says why: {refused}")
    section("finding a layer's run", before)

    # ------------------------------------------------------------------------------- 5. merge, depth report, spread
    before = len(failures)
    rows_all = list(cz.read_rows(out))
    s0, s1 = tmp / "runs" / "L3-s0.jsonl", tmp / "runs" / "L3-s1.jsonl"
    s0.write_text("".join(json.dumps(r) + "\n" for r in rows_all[:40]))
    s1.write_text("".join(json.dumps(r) + "\n" for r in rows_all[25:]))
    for s_ in (s0, s1):
        shutil.copy(cz.meta_path_of(out), cz.meta_path_of(s_))
    merged = tmp / "runs" / "L3-merged.jsonl"
    n_in, n_out = cz.merge_runs([s0, s1], merged)
    check((n_in, n_out) == (len(rows_all) + 15, len(rows_all)) and [cz.row_key(r) for r in cz.read_rows(merged)] == [cz.row_key(r) for r in rows_all],
          f"merge: {n_in} in, {n_out} kept, in order")
    check(json.loads(cz.meta_path_of(merged).read_text())["shards"] == ["L3-s0.jsonl", "L3-s1.jsonl"], "merge: the shards named")
    m1 = json.loads(cz.meta_path_of(s1).read_text())
    m1["max_new"] = 99
    cz.meta_path_of(s1).write_text(json.dumps(m1))
    try:
        cz.merge_runs([s0, s1], tmp / "runs" / "x.jsonl")
        check(False, "merge: shards taken with other settings are refused")
    except ValueError as exc:
        check("max_new" in str(exc), f"merge refusal: {exc}")
    rep = cz.depth_report({3: out})
    check(len(rep) == 1 and rep[0]["n_neurons"] == 4 and rep[0]["n_controls"] == 2 and not rep[0]["subsampled"]
          and 0 <= rep[0]["above_max"] <= rep[0]["above_p95"] <= 1 and len(rep[0]["top"]) == 4, f"depth report: {rep[0]}")
    check(set(rep[0]["coh_neuron_+0.8"]) == {"distinct4", "capped", "logprob"}, "depth report: coherence at +0.8")
    rep = cz.depth_report({3: out}, neurons=2, controls=1)
    check(rep[0]["subsampled"] and rep[0]["n_neurons"] == 2 and rep[0]["n_controls"] == 1, f"depth report subsampled: {rep[0]}")
    no_ctl = tmp / "runs" / "L1-noctl.jsonl"
    no_ctl.write_text("".join(json.dumps(r) + "\n" for r in rows_all if r["kind"] != "random"))
    check("note" in cz.depth_report({1: no_ctl})[0], "depth report: a run without controls says so")
    emb, name = cz.embed_texts(["the red apple fell", "the red apple fell", "a red apple fell down", "quantum flux capacitor"], "words")
    check(emb.shape == (4, 256) and np.allclose(np.linalg.norm(emb, axis=1), 1.0) and name.startswith("words"), f"embed_texts: {emb.shape} {name}")
    check(float(emb[0] @ emb[1]) > 0.999 and float(emb[0] @ emb[2]) > float(emb[0] @ emb[3]), "embed_texts: texts sharing words lie nearer")
    sp = cz.build_spread(out, embedder="words")
    d = json.loads(sp.read_text())
    n_rows = len(rows_all)
    check(sp == cz.spread_path_of(out) and all(len(d[k]) == n_rows for k in ("x", "y", "row", "kind", "neuron", "prompt", "dose")),
          "spread: parallel arrays, one entry per answer")
    check(min(d["x"]) == 0.0 and max(d["x"]) == 1.0 and min(d["y"]) >= 0.0 and max(d["y"]) <= 1.0 and d["layout"] in ("pca", "umap"),
          f"spread: in the unit square ({d['layout']}, {d['embedder']})")
    same = [i for i, r in enumerate(rows_all) if r["text"] == rows_all[0]["text"]]
    check(len({(d["x"][i], d["y"][i]) for i in same}) == 1, "spread: the same answer, the same place")
    for bad in ("st", "nope"):
        try:
            cz.embed_texts(["a"], bad)
            check(False, f"embed_texts refuses {bad!r}")
        except (ValueError, ImportError):
            check(True, "")
    units_path = cz.write_units(out)
    urecs = list(cz.read_rows(units_path))
    check(len(urecs) == 6 and all("shift" in u_ and "shift_up" in u_ for u_ in urecs) and any(u_["shift"] is not None for u_ in urecs),
          "write_units: the ranking, with each unit's word shift")
    section("merge, the depth report, the spread, the ranking file", before)

    # ------------------------------------------------------------------------------------------- 6. the command lines
    before = len(failures)
    env = {**os.environ, "THEODOLITE_ATLAS_DIR": str(tmp / "atlas"), "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"}
    env.pop("THEODOLITE_CENSUS_DIR", None)
    py = [sys.executable, "-I"]
    r = subprocess.run(py + [str(ROOT_DIR / "scripts" / "census_run.py"), "--stand-in", "--layer", "2", "--neurons", "0-1",
                             "--random-controls", "1", "--max-new", "6", "--batch", "5"], env=env, capture_output=True, text=True, timeout=600)
    run = tmp / "atlas" / "stand-in" / "census" / "runs" / "L2-main_rel.jsonl"
    check(r.returncode == 0 and run.exists() and len(list(cz.read_rows(run))) == 34, f"census_run.py --stand-in: {r.returncode} {r.stderr[-400:]}")
    check(cz.units_path_of(run).exists() and "the study's 36 texts" in r.stdout, "census_run.py: ranked, the push unit's source said")
    r = subprocess.run(py + [str(ROOT_DIR / "scripts" / "census_run.py"), "--stand-in", "--layer", "2", "--neurons", "0-1",
                             "--random-controls", "1", "--max-new", "7"], env=env, capture_output=True, text=True, timeout=600)
    check(r.returncode != 0 and "max_new" in (r.stderr + r.stdout), "census_run.py: other settings refused")
    r = subprocess.run(py + [str(ROOT_DIR / "scripts" / "census_tools.py"), "--stand-in", "rank", "2", "--find", "1"],
                       env=env, capture_output=True, text=True, timeout=300)
    check(r.returncode == 0 and "neurons above every control" in r.stdout and "neuron 1: rank" in r.stdout, f"census_tools.py rank: {r.stderr[-300:]}")
    r = subprocess.run(py + [str(ROOT_DIR / "scripts" / "census_tools.py"), "--stand-in", "depth"], env=env, capture_output=True, text=True, timeout=300)
    check(r.returncode == 0 and (tmp / "atlas" / "stand-in" / "census" / "runs" / "depth_report.json").exists(), f"census_tools.py depth: {r.stderr[-300:]}")
    r = subprocess.run(py + [str(ROOT_DIR / "scripts" / "census_tools.py"), "--stand-in", "spread", "2", "--embedder", "words"],
                       env=env, capture_output=True, text=True, timeout=300)
    check(r.returncode == 0 and cz.spread_path_of(run).exists() and "34 answers" in r.stdout, f"census_tools.py spread: {r.stderr[-300:]}")
    r = subprocess.run(py + [str(ROOT_DIR / "scripts" / "census_tools.py"), "rank", "17"], env=env, capture_output=True, text=True, timeout=300)
    check(r.returncode != 0 and "No census of layer 17" in r.stderr, "census_tools.py: a layer without a census said so")
    section("the command lines, into the stand-in's own folder", before)

    # ---------------------------------------------------------------------------------------------- 7. the dose window
    before = len(failures)
    check(cz.first_departure([1, 2, 3], [1, 2, 3]) is None and cz.first_departure([1, 9, 3], [1, 2, 3]) == 1
          and cz.first_departure([1, 2], [1, 2, 3]) == 2 and cz.first_departure([1, 2, 3, 4], [1, 2, 3]) == 3, "first_departure")
    B6 = [1, 2, 3, 4, 5, 6]

    def crow(kind: str, n: int, ids: List[int], dose: float, prompt: str = "who", finish: str = "stop", lp: float = -1.0) -> Dict[str, Any]:
        r = row(kind, n, "", dose=dose, prompt=prompt, ids=ids)
        r.update(finish=finish, push=dose * 10.0, logprob=[lp] * len(ids))
        return r

    hand = [crow("baseline", -1, B6, 0.0), crow("baseline", -1, [8, 9, 10, 11, 12], 0.0, prompt="free"),
            # +0.5: n0 as unpushed; n1 parts at 2 and is cut off; n2 stops early (parts at 3); r0, r1 as unpushed
            crow("neuron", 0, B6, 0.5), crow("neuron", 1, [1, 2, 9, 9, 9, 9], 0.5, finish="length", lp=-3.0), crow("neuron", 2, [1, 2, 3], 0.5),
            crow("random", 0, B6, 0.5), crow("random", 1, B6, 0.5),
            # +2: every neuron loops from the first piece; the controls as unpushed; on "free" too
            *[crow("neuron", n, [7] * 6, 2.0) for n in range(3)], *[crow("neuron", n, [7] * 6, 2.0, prompt="free") for n in range(3)],
            crow("random", 0, B6, 2.0), crow("random", 1, B6, 2.0),
            # -0.5: one control parts at 4
            crow("neuron", 0, B6, -0.5), crow("random", 0, [1, 2, 3, 4, 0, 0], -0.5)]
    ceil = {("neuron", 0): 2.0, ("neuron", 1): 4.0, ("neuron", 2): 5.0, ("random", 0): 10.0, ("random", 1): 20.0}
    cv = {(c_["dose"], c_["kind"]): c_ for c_ in cz.dose_curves(hand, ceil)}
    n5, r5, n2 = cv[(0.5, "neuron")], cv[(0.5, "random")], cv[(2.0, "neuron")]
    # +0.5, neurons: moved 2 of 3; departures 6 (never), 2, 3 -> median 3; push 5 over ceilings 2, 4, 5 -> median 1.25
    check(n5["n"] == 3 and n5["units"] == 3 and n5["moved"] == 2 / 3 and n5["departure"] == 3.0 and n5["capped"] == 1 / 3
          and n5["over_ceiling"] == 1.25 and n5["rel_push"] == 0.5 and abs(n5["logprob"] - (-5.0 / 3)) < 1e-12, f"dose_curves +0.5 neurons: {n5}")
    check(abs(n5["moved_se"] - float(np.std([0, 1, 1], ddof=1) / math.sqrt(3))) < 1e-12 and r5["moved"] == 0.0 and r5["departure"] == 6.0
          and r5["over_ceiling"] == 0.375, f"dose_curves: the standard error over units, the controls' row ({r5})")
    # +2: six answers of three units, each unit's two moved, so its share is 1 and the standard error 0; distinct4 1/3
    check(n2["n"] == 6 and n2["units"] == 3 and n2["moved"] == 1.0 and n2["moved_se"] == 0.0 and n2["departure"] == 0.0
          and abs(n2["distinct4"] - 1 / 3) < 1e-12, f"dose_curves +2 neurons: {n2}")
    b0 = cv[(0.0, "baseline")]
    check(b0["n"] == 2 and b0["moved"] == 0.0 and b0["over_ceiling"] is None and b0["distinct4"] == 1.0, f"a baseline row departs nowhere: {b0}")
    check([(c_["dose"], c_["kind"]) for c_ in cz.dose_curves(hand)][:3] == [(0.0, "baseline"), (-0.5, "neuron"), (-0.5, "random")]
          and cz.dose_curves(hand)[1]["over_ceiling"] is None, "dose_curves: the unpushed answers first, then by dose; no ceilings, none")
    streamed = cz.dose_curves(iter(hand), ceil, base=cz.baseline_ids(hand))
    check(streamed == cz.dose_curves(hand, ceil), "dose_curves: read once with the unpushed answers given, the same")
    win = cz.dose_window(cz.dose_curves(hand, ceil), "stream", ["who", "free"])
    # +0.5: the gap 2/3 is exactly twice its standard error, not more; +2: 1 against 0 with no spread
    check(win["+"]["moves"] == 2.0 and win["+"]["breaks"] == {"neuron": 2.0, "random": None}, f"dose_window +: {win['+']}")
    check("from +2 of the stream" in win["+"]["says"] and "On 3 neurons and 2 controls, one question set (who, free)" in win["+"]["says"]
          and "not a measurement" in win["+"]["says"], f"dose_window says: {win['+']['says']}")
    check(win["-"]["moves"] is None and "no more answers than the controls" in win["-"]["says"], f"dose_window -: {win['-']}")
    no_ctl = cz.dose_window(cz.dose_curves([r_ for r_ in hand if r_["kind"] != "random"]))
    check(no_ctl["+"]["moves"] is None and "no controls" in no_ctl["+"]["says"], "dose_window: without controls nothing is set against chance")
    # the ceilings read from a checkpoint: a W4A16 down projection in two safetensors files, as the 31B's are packed
    from safetensors.torch import save_file

    Dk, Nk, lk = 64, 128, 3
    g_ = torch.Generator().manual_seed(7)
    nib = torch.randint(0, 16, (Dk, Nk), generator=g_, dtype=torch.int32)
    packed = torch.zeros(Dk, Nk // 8, dtype=torch.int32)
    for j in range(8):
        packed |= nib[:, j::8] << (4 * j)
    scale = (2.0 ** torch.randint(-6, -2, (Dk, Nk // 32), generator=g_)).to(torch.bfloat16)
    w_post_k = (torch.rand(Dk, generator=g_) + 0.5).to(torch.bfloat16)
    Pk = f"model.language_model.layers.{lk}."
    fa, fb = tmp / "ck-a.safetensors", tmp / "ck-b.safetensors"
    save_file({Pk + "mlp.down_proj.weight_packed": packed, Pk + "mlp.down_proj.weight_scale": scale}, str(fa))
    save_file({Pk + "post_feedforward_layernorm.weight": w_post_k, Pk + "layer_scalar": torch.tensor([0.75], dtype=torch.bfloat16)}, str(fb))
    got_c, width = cz.checkpoint_ceilings([fa, fb], lk, [0, 9, 127])
    cols = (nib.double() - 8.0) * scale.double().repeat_interleave(32, dim=1)  # [D, N], exact in bf16
    want_c = {n: 0.75 * math.sqrt(Dk) * float((cols[:, n] / cols[:, n].norm() * w_post_k.double()).norm()) for n in (0, 9, 127)}
    check(width == Nk and all(abs(got_c[n] - want_c[n]) < 1e-5 * want_c[n] for n in want_c), f"checkpoint_ceilings: {got_c} vs {want_c}")
    cache = tmp / "hf"
    snap = cache / "models--org--tiny" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (cache / "models--org--tiny" / "refs").mkdir()
    (cache / "models--org--tiny" / "refs" / "main").write_text("abc")
    shutil.copy(fa, snap / "model.safetensors")
    probe = f"import sys; sys.path.insert(0, {str(ROOT_DIR)!r}); from backend import census as cz; print(cz.cached_checkpoint('org/tiny'), cz.cached_checkpoint('org/none'))"
    r = subprocess.run([sys.executable, "-I", "-c", probe], env={**os.environ, "HF_HUB_CACHE": str(cache), "HF_HUB_OFFLINE": "1"},
                       capture_output=True, text=True, timeout=120)
    check(r.returncode == 0 and str(snap / "model.safetensors") in r.stdout and r.stdout.strip().endswith("[]"),
          f"cached_checkpoint: the Hugging Face cache found through huggingface_hub ({r.stdout.strip()} {r.stderr[-300:]})")
    # a stand-in calibration run, read back by the command line
    r = subprocess.run(py + [str(ROOT_DIR / "scripts" / "census_run.py"), "--stand-in", "--layer", "3", "--design", "calib_rel", "--neurons",
                             "random:4", "--random-controls", "4", "--max-new", "8", "--batch", "16", "--no-rank"],
                       env=env, capture_output=True, text=True, timeout=600)
    calib = tmp / "atlas" / "stand-in" / "census" / "runs" / "L3-calib_rel.jsonl"
    check(r.returncode == 0 and calib.exists() and len(list(cz.read_rows(calib))) == 8 * 36 + 4, f"census_run.py calib_rel: {r.stderr[-400:]}")
    cj = tmp / "curves.json"
    r = subprocess.run(py + [str(ROOT_DIR / "scripts" / "census_tools.py"), "--stand-in", "curves", "3", "--weights", "--json", str(cj)],
                       env=env, capture_output=True, text=True, timeout=300)
    cjs = json.loads(cj.read_text()) if cj.exists() else {}
    doses = sorted({d_ for _, d_ in cz.DESIGNS["calib_rel"]})
    check(r.returncode == 0 and "L3-calib_rel.jsonl" in r.stdout and "Pushed up" in r.stdout and "Pushed down" in r.stdout,
          f"census_tools.py curves 3: a layer's calibration run read, though layer_runs leaves it out ({r.stderr[-300:]})")
    ents = cjs.get("curves", [])
    check(sorted({e["dose"] for e in ents if e["kind"] == "neuron"}) == doses and sorted({e["dose"] for e in ents if e["kind"] == "random"}) == doses
          and all(e["n"] == 12 for e in ents if e["kind"] != "baseline"), "curves: every dose of the design, for neurons and controls, 12 answers each")
    check(all(e["over_ceiling"] is not None and e["over_ceiling"] > 0 for e in ents if e["kind"] != "baseline")
          and set(cjs.get("window", {})) == {"+", "-"}, "curves: the push over each unit's ceiling, a verdict per sign")
    # a run whose answers keep no ceiling (taken before they did): --weights reads the same from the stand-in's weights
    bare = calib.with_name("L3-calib_bare.jsonl")
    bare.write_text("".join(json.dumps({k_: v_ for k_, v_ in r_.items() if k_ not in ("ceiling", "ref_neuron")}) + "\n" for r_ in cz.read_rows(calib)))
    shutil.copy(cz.meta_path_of(calib), cz.meta_path_of(bare))
    cjb = tmp / "curves-bare.json"
    r = subprocess.run(py + [str(ROOT_DIR / "scripts" / "census_tools.py"), "--stand-in", "curves", str(bare), "--json", str(cjb)],
                       env=env, capture_output=True, text=True, timeout=300)
    no_w = json.loads(cjb.read_text()) if cjb.exists() else {}
    r = subprocess.run(py + [str(ROOT_DIR / "scripts" / "census_tools.py"), "--stand-in", "curves", str(bare), "--weights", "--json", str(cjb)],
                       env=env, capture_output=True, text=True, timeout=300)
    by_w = json.loads(cjb.read_text()) if cjb.exists() else {}
    check(all(e["over_ceiling"] is None for e in no_w.get("curves", [])) and "add --weights" in subprocess.run(
              py + [str(ROOT_DIR / "scripts" / "census_tools.py"), "--stand-in", "curves", str(bare)], env=env, capture_output=True, text=True,
              timeout=300).stdout, "curves: without the answers' record or --weights, no ceilings, and it says so")
    check(r.returncode == 0 and by_w.get("ceilings_from", "").endswith("else the weights") and len(by_w.get("curves", [])) == len(ents)
          and all(abs(a_["over_ceiling"] - b_["over_ceiling"]) < 1e-4 * b_["over_ceiling"] for a_, b_ in zip(by_w["curves"], ents) if b_["kind"] != "baseline"),
          f"curves --weights: the ceilings read from the stand-in's weights, as the answers record them ({r.stderr[-300:]})")
    base_e = next((e for e in ents if e["kind"] == "baseline"), {})
    check(base_e.get("n") == 4 and base_e.get("moved") == 0.0, f"curves: the four unpushed answers depart nowhere ({base_e})")
    section("the dose window: departures, curves and the verdict by hand, ceilings from a checkpoint, a calib_rel run read back", before)

    # ------------------------------------------------------------------------- 8. the dose against each neuron's ceiling
    before = len(failures)
    refs = cz.control_refs(3, 2, w.d_ff)
    check(refs == [int(x) for x in cz.gain_sample(3, w.d_ff)[:2]] and cz.control_refs(3, 600, w.d_ff)[w.d_ff] == refs[0],
          "control_refs: control k paired with the k-th neuron of the read gain's sample, cycling")
    norms: List[float] = []
    real_write = ao.write_rows

    def spy(lm_: Any, W_U: Any, rows_ids: Any, iv: Any, **kw_: Any) -> Any:  # each row's added push, as write_rows gets it
        d_ = iv.adds[3] if iv is not None and 3 in iv.adds else torch.zeros(len(rows_ids), D)
        norms.extend(d_.norm(dim=-1).tolist())
        yield from real_write(lm_, W_U, rows_ids, iv, **kw_)

    ceil_run = tmp / "runs" / "L3-ceil.jsonl"
    kc = {**kw, "dose_unit": "ceiling", "max_new": 6}
    ao.write_rows = spy
    try:
        res = cz.take_census(rt, 3, [0, 5, 9], ceil_run, **kc)
    finally:
        ao.write_rows = real_write
    crows = list(cz.read_rows(ceil_run))
    wc = {n: float(c_) for n, c_ in zip(range(w.d_ff), w.write_ceiling(3, list(range(w.d_ff))).tolist())}
    own = [wc[r_["neuron"]] if r_["kind"] == "neuron" else wc[refs[r_["neuron"]]] if r_["kind"] == "random" else 0.0 for r_ in crows]
    check(res["written"] == 54 and len(norms) == len(crows) == 54, f"a ceiling run: {res}, {len(norms)} pushes seen")
    check(all(abs(nm - abs(r_["dose"]) * c_) <= 1e-5 * max(abs(r_["dose"]) * c_, 1e-6) for nm, r_, c_ in zip(norms, crows, own)),
          "dose_unit ceiling: each neuron pushed by exactly dose x its write ceiling, each control by dose x its paired neuron's "
          f"(the added delta measured; worst {max(abs(nm - abs(r_['dose']) * c_) for nm, r_, c_ in zip(norms, crows, own)):.2e})")
    near = lambda a_, b_: abs(a_ - b_) <= 1e-5 * max(abs(b_), 1.0)  # noqa: E731  (a ceiling read in another batch: float32's last bit)
    check(all(near(r_["ceiling"], c_) and near(r_["push"], r_["dose"] * c_) and near(r_["rel_push"], abs(r_["dose"]) * c_ / p50)
              for r_, c_ in zip(crows, own) if r_["kind"] != "baseline")
          and all(r_.get("ref_neuron") == refs[r_["neuron"]] for r_ in crows if r_["kind"] == "random")
          and all("ceiling" not in r_ and r_["push"] == 0.0 and r_["rel_push"] == 0.0 for r_ in crows if r_["kind"] == "baseline"),
          "each row records its ceiling, its push and its real share of the stream; a control its paired neuron")
    cm = json.loads(cz.meta_path_of(ceil_run).read_text())
    check(cm["dose_unit"] == "ceiling" and "write ceiling" in cm["push"] and "push's own unit" in cm["ceilings"], f"the ceiling run's meta: {cm['push'][:60]}")
    try:
        cz.take_census(rt, 3, [0, 5, 9], ceil_run, **{**kc, "dose_unit": "stream"})
        check(False, "a run taken in the ceiling unit is not continued in the stream's")
    except ValueError as exc:
        check("dose_unit" in str(exc), f"refusal: {exc}")
    try:
        cz.take_census(rt, 3, [0], tmp / "runs" / "cc.jsonl", **{**kw, "design_name": "calib_ceil", "dose_unit": "stream"})
        check(False, "calib_ceil in the stream unit is refused")
    except ValueError as exc:
        check("ceiling" in str(exc), f"refusal: {exc}")
    srow = [r_ for r_ in got if r_["kind"] != "baseline"]
    check(all(near(r_["ceiling"], wc[r_["neuron"]] if r_["kind"] == "neuron" else wc[refs[r_["neuron"]]]) for r_ in srow),
          "a stream run records each answer's ceiling too, its push unchanged")
    says_c = cz.dose_window(cz.dose_curves(hand, ceil), "ceiling")["+"]["says"]
    check("from +2 x each neuron's ceiling (a median 2 of the stream)" in says_c, f"dose_window in the ceiling unit: {says_c}")
    old_meta = json.loads(cz.meta_path_of(out2).read_text())
    old_meta.pop("dose_unit")
    cz.meta_path_of(out2).write_text(json.dumps(old_meta))
    check(cz.take_census(rt, 3, [0, 5, 9], out2, **kw)["written"] == 0 and cz.settings_differ(old_meta, {**old_meta, "dose_unit": "stream"}) == []
          and cz.settings_differ(old_meta, {**old_meta, "dose_unit": "ceiling"}) == ["dose_unit"],
          "a meta that records no dose unit was taken in the stream's, and still continues")
    urecs = list(cz.read_rows(cz.write_units(ceil_run)))
    check(len(urecs) == 5 and all(set(u_["dose"]) == {"-0.7", "+0.5", "+0.8", "+3"} for u_ in urecs),
          "compute_units ranks a ceiling run, the push's share per dose its mean")
    rep = cz.depth_report({3: ceil_run, 2: out})
    check(rep[1]["dose_unit"] == "ceiling" and "write ceiling" in rep[1]["dose_note"] and rep[0]["dose_unit"] == "stream"
          and "different push at each layer" in rep[0]["dose_note"], "the depth report names each run's unit")
    r = subprocess.run(py + [str(ROOT_DIR / "scripts" / "census_run.py"), "--stand-in", "--layer", "3", "--design", "calib_ceil", "--neurons",
                             "random:4", "--random-controls", "4", "--max-new", "8", "--batch", "16", "--no-rank"],
                       env=env, capture_output=True, text=True, timeout=600)
    cc = tmp / "atlas" / "stand-in" / "census" / "runs" / "L3-calib_ceil.jsonl"
    check(r.returncode == 0 and len(list(cz.read_rows(cc))) == 8 * 30 + 4 and json.loads(cz.meta_path_of(cc).read_text())["dose_unit"] == "ceiling",
          f"census_run.py --design calib_ceil: taken in the ceiling unit by default ({r.stderr[-300:]})")
    cj2 = tmp / "curves-ceil.json"
    r = subprocess.run(py + [str(ROOT_DIR / "scripts" / "census_tools.py"), "--stand-in", "curves", str(cc), "--json", str(cj2)],
                       env=env, capture_output=True, text=True, timeout=300)
    cjs = json.loads(cj2.read_text()) if cj2.exists() else {}
    ents = [e for e in cjs.get("curves", []) if e["kind"] != "baseline"]
    check(r.returncode == 0 and cjs.get("dose_unit") == "ceiling" and "multiples of each neuron's write ceiling" in r.stdout
          and sorted({e["dose"] for e in ents}) == sorted({d_ for _, d_ in cz.DESIGNS["calib_ceil"]}),
          f"census_tools.py curves on a calib_ceil run: every dose, in the ceiling unit ({r.stderr[-300:]})")
    check(all(abs(e["over_ceiling"] - abs(e["dose"])) < 1e-3 * abs(e["dose"]) for e in ents) and cjs.get("ceilings_from") == "the answers' own record"
          and set(cjs.get("window", {})) == {"+", "-"}, "curves: a ceiling run's push is its dose times the ceiling, from its own record")
    section("the dose against each neuron's own ceiling: pushes measured, rows and meta, refusals, old metas, ranking, calib_ceil read back", before)

    # ------------------------------------------------------------- 9. the null on the real checkpoint's own weights
    before = len(failures)
    real = real_read_gain()
    if real is None:
        print("  --    the null on the 31B's weights: the checkpoint is not in the Hugging Face cache, skipped")
    else:
        for l, (fair, iso) in real.items():
            check(0.5 <= fair <= 2.0, f"page {l + 1}: the next layer reads the census's controls {fair:.2f}x as loudly as the neurons")
            check(iso > 20.0, f"page {l + 1}: an isotropic control is read {iso:.0f}x as loudly (the fault the shaped controls mend)")
        section("the null on the 31B's own weights: shaped controls as loud as neurons, isotropic ones ~100x louder ("
                + ", ".join(f"page {l + 1}: {f:.2f}x vs {i:.0f}x" for l, (f, i) in real.items()) + ")", before)

    print(f"\n{passed} checks passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

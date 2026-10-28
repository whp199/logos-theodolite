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
   neurons added write only theirs, and the meta names every neuron the run holds; a run taken with other settings is
   refused in plain words; a run stopped between steps keeps only whole batches and finishes as the uninterrupted one;
4. finding a layer's run: main_rel first, calibration runs, runs of another model and runs taken with the old
   isotropic controls left out (the last with the reason);
5. merge, the depth report (subsampled to a common pool) and the spread (the words embedder, laid out in the unit square);
6. the command lines: census_run.py --stand-in into the stand-in's own folder, then census_tools.py rank, depth and
   spread on it;
7. the dose window: where an answer first parts from the unpushed one, the curves per dose and kind and the verdict per
   sign on answers worked out by hand (text breaking read against the unpushed answers to the same questions, an
   unasked question's left out; no spread, and no verdict, from one control), the write ceilings read from a W4A16
   checkpoint in two files and from a dense one, the Hugging Face cache found through huggingface_hub (one file, or two
   through their index), and a stand-in calib_rel run read back by census_tools.py curves, with the ceilings read from
   the stand-in's weights for answers that keep none;
8. the dose in each neuron's own ceiling: each push measured as write_rows receives it (dose x the neuron's write
   ceiling; a control, dose x its paired neuron's), each row's ceiling, push and real share of the stream, a unit
   change refused, an older meta read as the stream unit, a ceiling run ranked and in the depth report, and a stand-in
   calib_ceil run read back by census_tools.py curves;
9. batch plans: a whole layer's plan, the 16 x batch plan as the code before plans batched a fresh run; the batch, the
   plan, the job list and the runtime recorded; a run stopped mid-chunk, and one whose last batch was torn, each
   finish equal to an unbroken run row for row, and a file its plan did not write is refused, left as it was; another
   batch refused, another runtime warned and recorded; three shards, each whole chunks with a canary, merged equal the
   unbroken run row for row, layer_runs reads no shard, and a shard is continued only as itself; the merge names a
   missing shard (joined with --partial and said so), a changed canary (shard and field), two runtime classes, and
   refuses a shard given twice and a file without its meta; an old meta (no batch, plan, dose unit, runtime or job
   list) with its rows regrouped still continues, its earlier runtime recorded as unknown, and merges; runs of other
   neurons merged name every part's neurons and continue; census_run.py --shard and census_tools.py merge from the
   command line, the printed pattern's canaries left out, and rank refusing a shard;
10. atlas_ops.write_rows reading each step back in one transfer gives the Steps the per-row reads gave (that code kept
   here), bit for bit; census_run.py frees the J++ lens after load, and a census taken without it writes every answer
   and names it; --layers takes several layers in one process, each its own run;
11. a two-stage census: a screen's strongest neurons by mass retested against fresh controls (census_run.py
   --neurons-from --top --seed), the stage in the meta, the retest's p against its controls and q over its neurons
   alone, the screen still the layer's census, rank and the card naming the stage, and a retest refused with the
   screen's seed, too many neurons, --neurons beside it, another layer, or other neurons once its screen is ranked
   again;
12. when the 31B's checkpoint is in the Hugging Face cache (found through huggingface_hub): on its own norms and down
   columns (CPU, a few tensors), the next layer reads the census's controls about as loudly as the layer's neurons, and
   an isotropic control far louder.

Run: .venv/bin/python -I scripts/check_census.py   (exits 0 when every check holds)
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"

import hashlib
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


@torch.no_grad()
def write_rows_per_row(lm: Any, W_U: torch.Tensor, rows_ids: Any, iv: Any, *, max_new: int, stop_ids: Any, bos_id: Any,
                       spare_bos: bool = True, softcap: Any = 30.0, top_k: int = 0, cancel: Any = None) -> Any:
    """atlas_ops.write_rows as it was before it read each step back in one transfer: the chosen piece, its
    log-probability and the top k read from the device row by row. Kept here as the reference the one transfer must
    match, step for step and bit for bit."""
    import torch.nn.functional as F

    dev = ao.model_device(lm)
    B = len(rows_ids)
    T = max(len(r) for r in rows_ids)
    ids = torch.tensor([[0] * (T - len(r)) + [int(t) for t in r] for r in rows_ids], dtype=torch.long, device=dev)
    mask = torch.tensor([[0] * (T - len(r)) + [1] * len(r) for r in rows_ids], dtype=torch.long, device=dev)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    stop = {int(t) for t in stop_ids}
    done = [False] * B
    keep = ao._Keep()
    keep.mask = ao.keep_mask(ids, mask, bos_id, spare_bos)
    with ao.installed(lm, iv, keep):
        out = lm(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
        cache = out.past_key_values
        h = out.last_hidden_state[:, -1]
        for _ in range(int(max_new)):
            if cancel is not None and cancel():
                return
            logits = ao._model_logits(lm, W_U, h, softcap)
            nxt = logits.argmax(-1)
            lsm = F.log_softmax(logits.float(), dim=-1)
            if top_k:
                tv, ti = torch.topk(lsm, k=int(top_k), dim=-1)
            for r in range(B):
                if done[r]:
                    continue
                t = int(nxt[r])
                end = t in stop
                yield ao.Step(row=r, token_id=t, logprob=float(lsm[r, t]), top_ids=[int(x) for x in ti[r].tolist()] if top_k else [],
                              top_lps=[round(float(x), 4) for x in tv[r].tolist()] if top_k else [], end=end)
                if end:
                    done[r] = True
            if all(done):
                return
            keep.mask = None
            mask = torch.cat([mask, torch.ones_like(mask[:, :1])], -1)
            pos = pos[:, -1:] + 1
            out = lm(input_ids=nxt[:, None], attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            h = out.last_hidden_state[:, -1]


def real_read_gain(layers: tuple = (45, 57), sample: int = 256) -> Any:
    """{layer: (shaped ratio, isotropic ratio)}: how loudly the next layer's input norm reads the census's controls
    against the layer's neurons, on the 31B's own norms and dequantised down columns read from the local checkpoint
    (CPU, a few tensors); None when the checkpoint is not cached here (the cache found through huggingface_hub:
    $HF_HUB_CACHE, $HF_HOME, else ~/.cache/huggingface)."""
    import contextlib

    from backend.engine_runtime import DEFAULT_W4A16_MODEL_ID, dequantize_w4a16_cols

    files = cz.cached_checkpoint(DEFAULT_W4A16_MODEL_ID)
    if not files:
        return None
    from safetensors import safe_open

    P = "model.language_model.layers."
    out = {}
    with contextlib.ExitStack() as stack:
        opened = [stack.enter_context(safe_open(str(p), "pt")) for p in files]
        where = {k: g for g in opened for k in g.keys()}  # each tensor's file

        def get(k: str) -> torch.Tensor:
            return where[k].get_tensor(k)

        for l in layers:
            w_post = get(P + f"{l}.post_feedforward_layernorm.weight").float()
            w_in = get(P + f"{l + 1}.input_layernorm.weight").float().numpy()
            idx = torch.from_numpy(np.random.default_rng(l).choice(21504, sample, replace=False))
            U = dequantize_w4a16_cols(get(P + f"{l}.mlp.down_proj.weight_packed"), get(P + f"{l}.mlp.down_proj.weight_scale"), idx).float()
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
    # 45 rows were left; the batch they end inside was written in part, so it is cut off and written again whole
    ends = np.cumsum([len(b_[2]) for b_ in cz.plan_batches(cz.census_jobs([0, 5, 9], 2, cz.DESIGNS["main_rel"]), 7)])
    whole = int(max(e_ for e_ in ends if e_ <= 45))
    check(res["written"] == 54 - whole and res["skipped"] == whole and len(again) == 54, f"a run cut short continues: {res}")
    check({cz.row_key(r): r["token_ids"] for r in again} == {cz.row_key(r): r["token_ids"] for r in got} and again == got,
          "and ends with the same answers, row for row")
    check(cz.take_census(rt, 3, [0, 5, 9], out, **kw) == {"written": 0, "skipped": 54, "stopped": False}, "nothing left to write")
    res = cz.take_census(rt, 3, [0, 5, 9, 12], out, **kw)
    check(res["written"] == 10 and {r["neuron"] for r in cz.read_rows(out) if r["kind"] == "neuron"} == {0, 5, 9, 12}, f"a neuron added: {res}")
    j4 = cz.jobs_digest(cz.census_jobs([0, 5, 9, 12], 2, cz.DESIGNS["main_rel"]))
    res = cz.take_census(rt, 3, [12], out, **kw)  # continued with one neuron alone: the meta still names the run's four
    m3 = json.loads(cz.meta_path_of(out).read_text())
    check(res["written"] == 0 and m3["neuron_ids"] == "0,5,9,12" and m3["jobs"] == j4 and m3["plan_kept"] is False
          and m3["plan_left"] == "the run's neurons changed after it was begun",
          f"the meta's neurons and job list are every neuron the run holds, not the last call's ({m3['neuron_ids']}, {m3['jobs']})")
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
            # -0.5: one of two controls parts at 4, neither neuron does
            crow("neuron", 0, B6, -0.5), crow("neuron", 1, B6, -0.5), crow("random", 0, [1, 2, 3, 4, 0, 0], -0.5), crow("random", 1, B6, -0.5)]
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
    check(b0["n"] == 2 and b0["moved"] == 0.0 and b0["over_ceiling"] is None and b0["distinct4"] == 1.0 and b0["questions"] == ["who", "free"],
          f"a baseline row departs nowhere: {b0}")
    check([(c_["dose"], c_["kind"]) for c_ in cz.dose_curves(hand)][:3] == [(0.0, "baseline"), (-0.5, "neuron"), (-0.5, "random")]
          and cz.dose_curves(hand)[1]["over_ceiling"] is None, "dose_curves: the unpushed answers first, then by dose; no ceilings, none")
    streamed = cz.dose_curves(iter(hand), ceil, base=cz.baseline_ids(hand))
    check(streamed == cz.dose_curves(hand, ceil), "dose_curves: read once with the unpushed answers given, the same")
    win = cz.dose_window(cz.dose_curves(hand, ceil), "stream", ["who", "free"])
    # +0.5: the gap 2/3 is exactly twice its standard error, not more; +2: 1 against 0 with no spread
    check(win["+"]["moves"] == 2.0 and win["+"]["breaks"] == {"neuron": 2.0, "random": None}, f"dose_window +: {win['+']}")
    check("first at +2 of the stream" in win["+"]["says"] and "On 3 neurons and 2 controls, one question set (who, free)" in win["+"]["says"]
          and "not a measurement" in win["+"]["says"], f"dose_window says: {win['+']['says']}")
    check(win["-"]["moves"] is None and "no more answers than the controls" in win["-"]["says"], f"dose_window -: {win['-']}")
    no_ctl = cz.dose_window(cz.dose_curves([r_ for r_ in hand if r_["kind"] != "random"]))
    check(no_ctl["+"]["moves"] is None and "no controls" in no_ctl["+"]["says"], "dose_window: without controls nothing is set against chance")
    # where text breaks, read against the unpushed answers to the same questions: an unpushed answer to a question no
    # dose is asked on ("snake", which loops) is left out, and a question whose own unpushed answer loops ("orange")
    # does not break where its pushed answers loop as much; pooled over all four, the reference would be (1 + 1 + 1/3 +
    # 1/3) / 4 = 2/3, and the controls' orange answers at +4 (1/3) would break
    loop = [5] * 6
    hand_q = hand + [crow("baseline", -1, loop, 0.0, prompt="snake"), crow("baseline", -1, loop, 0.0, prompt="orange"),
                     *[crow("neuron", n, loop, 4.0, prompt="orange") for n in range(3)],
                     *[crow("random", n, loop, 4.0, prompt="orange") for n in range(2)]]
    cq = {(c_["dose"], c_["kind"]): c_ for c_ in cz.dose_curves(hand_q)}
    check(cq[(0.0, "baseline")]["questions"] == ["who", "orange", "free"] and cq[(0.0, "baseline")]["n"] == 3
          and abs(cq[(4.0, "neuron")]["distinct4_unpushed"] - 1 / 3) < 1e-12 and cq[(2.0, "neuron")]["distinct4_unpushed"] == 1.0
          and cq[(0.0, "baseline")]["distinct4_unpushed"] is None,
          f"dose_curves: the unpushed answers to the questions the doses are asked on, each cell's own reference ({cq[(0.0, 'baseline')]})")
    wq = cz.dose_window(list(cq.values()))
    check(wq["+"]["breaks"] == {"neuron": 2.0, "random": None} and "to the same questions" in wq["+"]["says"],
          f"dose_window: text breaks against each question's own unpushed answer, not a pooled one ({wq['+']['breaks']})")
    # one control: no spread among the controls, so no dose is set against chance
    one_ctl = [r_ for r_ in hand if not (r_["kind"] == "random" and r_["neuron"] == 1)]
    c1 = {(c_["dose"], c_["kind"]): c_ for c_ in cz.dose_curves(one_ctl)}
    w1 = cz.dose_window(list(c1.values()))
    check(c1[(2.0, "random")]["moved_se"] is None and c1[(2.0, "neuron")]["moved_se"] == 0.0 and w1["+"]["moves"] is None
          and "spread between units cannot be estimated" in w1["+"]["says"],
          f"dose_window: from one control no spread is estimated and nothing is set against chance ({w1['+']['says'][:90]})")
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
    fd = tmp / "ck-dense.safetensors"  # the same down columns stored dense, as an unquantised checkpoint keeps them
    save_file({Pk + "mlp.down_proj.weight": cols.float(), Pk + "post_feedforward_layernorm.weight": w_post_k}, str(fd))
    got_d, width_d = cz.checkpoint_ceilings([fd], lk, [0, 9, 127])
    check(width_d == Nk and all(abs(got_d[n] - want_c[n] / 0.75) < 1e-5 * want_c[n] for n in want_c),
          f"checkpoint_ceilings: a dense down projection, no layer scalar ({got_d})")
    cache = tmp / "hf"
    snap = cache / "models--org--tiny" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (cache / "models--org--tiny" / "refs").mkdir()
    (cache / "models--org--tiny" / "refs" / "main").write_text("abc")
    shutil.copy(fa, snap / "model.safetensors")
    snap2 = cache / "models--org--two" / "snapshots" / "def"  # a checkpoint in two files and their index, as the 31B's
    snap2.mkdir(parents=True)
    (cache / "models--org--two" / "refs").mkdir()
    (cache / "models--org--two" / "refs" / "main").write_text("def")
    two = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    shutil.copy(fa, snap2 / two[1])
    shutil.copy(fb, snap2 / two[0])
    (snap2 / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        Pk + "mlp.down_proj.weight_packed": two[1], Pk + "mlp.down_proj.weight_scale": two[1],
        Pk + "post_feedforward_layernorm.weight": two[0], Pk + "layer_scalar": two[0]}}))
    probe = (f"import json, sys; sys.path.insert(0, {str(ROOT_DIR)!r}); from backend import census as cz; "
             "print(json.dumps([[str(p) for p in cz.cached_checkpoint(r)] for r in ('org/tiny', 'org/none', 'org/two')]))")
    r = subprocess.run([sys.executable, "-I", "-c", probe], env={**os.environ, "HF_HUB_CACHE": str(cache), "HF_HUB_OFFLINE": "1"},
                       capture_output=True, text=True, timeout=120)
    found = json.loads(r.stdout) if r.returncode == 0 else [[], [], []]
    check(found[0] == [str(snap / "model.safetensors")] and found[1] == [],
          f"cached_checkpoint: the Hugging Face cache found through huggingface_hub ({r.stdout.strip()} {r.stderr[-300:]})")
    got_i = cz.checkpoint_ceilings([Path(x) for x in found[2]], lk, [0, 9, 127])[0] if len(found[2]) == 2 else {}
    check(found[2] == [str(snap2 / x) for x in two] and got_i == got_c,
          f"cached_checkpoint: a checkpoint in two files found through its index, and its ceilings read across them ({found[2]})")
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
    r0_out = r.stdout
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
    check(base_e.get("n") == 3 and base_e.get("questions") == ["who", "orange", "free"] and base_e.get("moved") == 0.0
          and "unpushed (who, orange, free, the questions the doses are asked on): 3 answers" in r0_out,
          f"curves: the unpushed answers to calib_rel's three questions (the snake game's left out) depart nowhere ({base_e})")
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
    check("first at +2 x each neuron's ceiling (a median 2 of the stream)" in says_c, f"dose_window in the ceiling unit: {says_c}")
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

    # ------------------------------------------------------------- 9. batch plans: a stop, a torn batch, shards, merges
    before = len(failures)
    layer_jobs = cz.census_jobs(range(21504), 256, cz.DESIGNS["main_rel"])
    p10, p16 = cz.plan_batches(layer_jobs, 32), cz.plan_batches(layer_jobs, 32, cz.PLAN_BEFORE)
    fewest = sum(math.ceil(sum(1 for j_ in layer_jobs if j_["prompt"] == q_) / 32) for q_ in cz.PROMPTS)
    check(len(p10) == fewest == 6804 and len(p16) == 7227 and sorted(i for _, _, b_ in p10 for i in b_) == list(range(len(layer_jobs))),
          f"plan_batches: a whole main_rel layer at --batch 32 in {len(p10)} batches at 10 x batch (the fewest its questions allow), "
          f"{len(p16)} at 16 x batch; every answer once")

    def old_batches(jobs_: List[Dict[str, Any]], B_: int) -> List[List[Any]]:  # the code before batch plans, on a fresh run
        out_: List[List[Any]] = []
        for c0 in range(0, len(jobs_), B_ * 16):
            part_ = jobs_[c0 : c0 + B_ * 16]
            for p_ in cz.PROMPTS:
                js_ = [j_ for j_ in part_ if j_["prompt"] == p_]
                out_ += [[cz.row_key(j_) for j_ in js_[b_ : b_ + B_]] for b_ in range(0, len(js_), B_)]
        return out_

    small = cz.census_jobs(range(40), 7, cz.DESIGNS["main_rel"])
    check([[cz.row_key(small[i]) for i in b_] for _, _, b_ in cz.plan_batches(small, 9, cz.PLAN_BEFORE)] == old_batches(small, 9)
          and all(len({small[i]["prompt"] for i in b_}) == 1 for _, _, b_ in cz.plan_batches(small, 9)),
          "the 16 x batch plan batches a fresh run as the code before batch plans did; every batch shares its question")
    pdir = tmp / "plan"
    pdir.mkdir()
    NU = list(range(7))
    kp = dict(p50=p50, p50_source="the study's 36 texts", design_name="main_rel", controls=3, seed=3, max_new=6, top_k=5, batch=5,
              log=lambda s: None)
    u_jobs = cz.census_jobs(NU, 3, cz.DESIGNS["main_rel"])
    u_plan = cz.plan_batches(u_jobs, 5)
    unbroken = pdir / "L3-u.jsonl"
    cz.take_census(rt, 3, NU, unbroken, **kp)
    U = list(cz.read_rows(unbroken))
    um = json.loads(cz.meta_path_of(unbroken).read_text())
    check(len(U) == len(u_jobs) == 104 and um["batch"] == 5 and um["plan"] == cz.PLAN and um["plan_kept"] is True
          and um["jobs"] == cz.jobs_digest(u_jobs) and um["neuron_ids"] == "0-6" and um["d_ff"] == w.d_ff and "shard" not in um,
          "an unbroken run: its meta records the batch, the plan and the job list it was made from")
    rtm = um.get("runtime") or {}
    check(set(rtm) == {"device", "torch", "cuda", "transformers", "interp_engine", "compressed_tensors", "git", "git_clean",
                       "act_scale_sha256"} and rtm["device"] == "cpu" and rtm["torch"] == torch.__version__
          and rtm["act_scale_sha256"] is None and (rtm["git"] is None or len(rtm["git"]) == 40), f"the runtime record: {rtm}")
    # the unpacker beside the class, not in it (every unpacker gives the same bits): the meta's own record
    check((um.get("unpacker") or {}).get("mode") == "reference" and "unpacker" not in rtm,
          f"the unpacker recorded beside the runtime class, not in it: {um.get('unpacker')}")
    check(cz.runtime_record(None, fa)["act_scale_sha256"] == hashlib.sha256(fa.read_bytes()).hexdigest(), "the runtime record: the stats index's sha256")
    calls = [0]

    def stop_after(n_: int) -> Any:
        calls[0] = 0

        def f_() -> bool:
            calls[0] += 1
            return calls[0] > n_
        return f_

    stopped = pdir / "L3-stop.jsonl"
    res = cz.take_census(rt, 3, NU, stopped, **kp, cancel=stop_after(40))
    part_rows = list(cz.read_rows(stopped))
    first_chunk = sum(len(b_[2]) for b_ in u_plan if b_[0] == 0)
    check(res["stopped"] and 0 < len(part_rows) < first_chunk and part_rows == U[: len(part_rows)], f"a run stopped mid-chunk: {res}")
    res = cz.take_census(rt, 3, NU, stopped, **kp)
    check(res["written"] == 104 - len(part_rows) and list(cz.read_rows(stopped)) == U, "and continued, it equals the unbroken run row for row")
    ends = np.cumsum([len(b_[2]) for b_ in u_plan])
    bt = next(b_ for b_, (c_, _, idx) in enumerate(u_plan) if c_ == 1 and len(idx) == 5)  # a whole batch of the second chunk
    start = int(ends[bt - 1])
    torn = pdir / "L3-torn.jsonl"
    u_lines = unbroken.read_text().splitlines(keepends=True)
    torn.write_text("".join(u_lines[: start + 2]) + u_lines[start + 2][:40])
    shutil.copy(cz.meta_path_of(unbroken), cz.meta_path_of(torn))
    said: List[str] = []
    res = cz.take_census(rt, 3, NU, torn, **{**kp, "log": said.append})
    check(res["written"] == 104 - start and res["skipped"] == start and list(cz.read_rows(torn)) == U
          and any("half-written" in s_ for s_ in said) and any("written in part (2 of its 5" in s_ for s_ in said),
          f"a torn last batch (two whole answers and half of a third): cut off, written again whole, row for row as unbroken ({res})")
    straddle = pdir / "L3-straddle.jsonl"
    straddle.write_text("".join(u_lines[:7] + u_lines[8:30]) + u_lines[30][:30])  # and a half-written last line
    shutil.copy(cz.meta_path_of(unbroken), cz.meta_path_of(straddle))
    held = (straddle.read_bytes(), cz.meta_path_of(straddle).read_bytes())
    try:
        cz.take_census(rt, 3, NU, straddle, **kp)
        check(False, "a file holding answers its plan did not write is refused")
    except ValueError as exc:
        check("batch plan did not write" in str(exc) and "batch 1 of its plan" in str(exc), f"refusal: {exc}")
    check((straddle.read_bytes(), cz.meta_path_of(straddle).read_bytes()) == held,
          "a refused file is left as it was, its half-written last line not cut off")
    try:
        cz.take_census(rt, 3, NU, unbroken, **{**kp, "batch": 6})
        check(False, "a run is not continued at another batch size")
    except ValueError as exc:
        check("batch" in str(exc), f"refusal: {exc}")
    m_ = json.loads(cz.meta_path_of(stopped).read_text())
    m_["runtime"] = {**m_["runtime"], "torch": "0.0-elsewhere"}
    cz.meta_path_of(stopped).write_text(json.dumps(m_))
    said = []
    res = cz.take_census(rt, 3, NU, stopped, **{**kp, "log": said.append})
    m_ = json.loads(cz.meta_path_of(stopped).read_text())
    check(res["written"] == 0 and any(s_.startswith("warning") and "torch" in s_ for s_ in said) and len(m_.get("mixed_runtime", [])) == 2,
          "a run continued on another runtime: warned and recorded (mixed_runtime), not refused")
    # three shards, each its whole chunks, merged
    sdir = tmp / "shards"
    sdir.mkdir()
    base_run = sdir / "L3-main.jsonl"
    shards = [cz.shard_path(base_run, i, 3) for i in range(3)]
    for i, sp_ in enumerate(shards):
        cz.take_census(rt, 3, NU, sp_, **kp, shard=(i, 3))
    sm = [json.loads(cz.meta_path_of(sp_).read_text()) for sp_ in shards]
    check([m_["shard"]["batches"] for m_ in sm] == [[0, sm[1]["shard"]["batches"][0]], [sm[1]["shard"]["batches"][0], sm[2]["shard"]["batches"][0]],
                                                    [sm[2]["shard"]["batches"][0], len(u_plan)]]
          and [m_["shard"]["chunks"] for m_ in sm] == [[0, 1], [1, 2], [2, 3]] and all(m_["shard"]["n"] == 3 for m_ in sm),
          f"--shard i/3: whole chunks each, together the plan ({[m_['shard'] for m_ in sm]})")
    check(not cz.canary_path_of(shards[0]).exists() and all(
          [cz.row_key(r_) for r_ in cz.read_rows(cz.canary_path_of(sp_))] == [cz.row_key(u_jobs[i]) for i in u_plan[0][2]] for sp_ in shards[1:]),
          "every shard but 0 writes plan batch 0 again, its canary")
    check(cz.layer_runs(sdir) == {}, "layer_runs never reads a shard as a layer's census")
    n_in, n_out = cz.merge_runs(shards, base_run)
    mm = json.loads(cz.meta_path_of(base_run).read_text())
    check(list(cz.read_rows(base_run)) == U and (n_in, n_out) == (104, 104), "three shards merged equal the unbroken run, row for row")
    check("shard" not in mm and mm["shards"] == [sp_.name for sp_ in shards] and mm["canary"] == "identical on 2 of 2 shards"
          and mm["neurons"] == um["neurons"] and mm["neuron_ids"] == "0-6" and mm["plan_kept"] is True and "partial" not in mm
          and "mixed_runtime" not in mm, f"the merged meta: {mm.get('canary')}, {mm.get('shards')}")
    check(cz.layer_runs(sdir) == {3: base_run}, "the merged run is the layer's census")
    try:
        cz.merge_runs([shards[0], shards[2]], sdir / "x.jsonl")
        check(False, "a missing shard is refused")
    except ValueError as exc:
        check("Shard 1 of 3 is missing" in str(exc) and "--partial" in str(exc), f"refusal: {exc}")
    cz.merge_runs([shards[0], shards[2]], sdir / "L3-part.jsonl", partial=True)
    pm = json.loads(cz.meta_path_of(sdir / "L3-part.jsonl").read_text())
    n1 = len(list(cz.read_rows(shards[1])))
    check(pm["partial"]["missing_shards"] == [1] and pm["partial"]["missing_answers"] == n1 and len(pm["partial"]["first_missing"]) == 5
          and list(cz.read_rows(sdir / "L3-part.jsonl")) == [r_ for r_ in U if r_ not in list(cz.read_rows(shards[1]))],
          f"--partial: joined, and the meta says what is missing ({pm.get('partial')})")
    for bad, why in (([shards[0], shards[1], shards[1], shards[2]], "given twice"), ([shards[0], unbroken], "not one")):
        try:
            cz.merge_runs(bad, sdir / "x.jsonl")
            check(False, f"merge refuses: {why}")
        except ValueError as exc:
            check(why in str(exc), f"refusal: {exc}")
    can2 = cz.canary_path_of(shards[2])
    keep_can = can2.read_text()
    crs = list(cz.read_rows(can2))
    crs[1]["token_ids"] = crs[1]["token_ids"][:-1] + [crs[1]["token_ids"][-1] + 1]
    can2.write_text("".join(json.dumps(r_) + "\n" for r_ in crs))
    m1 = json.loads(cz.meta_path_of(shards[1]).read_text())
    cz.meta_path_of(shards[1]).write_text(json.dumps({**m1, "runtime": {**m1["runtime"], "device": "NVIDIA H100 80GB HBM3"}}))
    cz.merge_runs(shards, sdir / "L3-odd.jsonl")
    om = json.loads(cz.meta_path_of(sdir / "L3-odd.jsonl").read_text())
    check(om["canary"].startswith("identical on 1 of 2 shards") and f"shard 2's differs from shard 0's (first in token_ids of {crs[1]['kind']}" in om["canary"],
          f"a changed canary is named, shard and field: {om['canary']}")
    check(len(om.get("mixed_runtime", [])) == 2 and {c_["device"] for c_ in om["mixed_runtime"]} == {"cpu", "NVIDIA H100 80GB HBM3"},
          "shards on two runtime classes: both recorded (mixed_runtime), not refused")
    can2.write_text(keep_can)
    cz.meta_path_of(shards[1]).write_text(json.dumps(m1))
    try:
        cz.take_census(rt, 3, NU, cz.shard_path(base_run, 0, 4), **kp, shard=(0, 4))
        check(False, "more shards than the plan has chunks are refused")
    except ValueError as exc:
        check("at most 3 shards" in str(exc), f"refusal: {exc}")
    try:
        cz.take_census(rt, 3, NU + [9], shards[0], **kp, shard=(0, 3))
        check(False, "a shard's neurons cannot change")
    except ValueError as exc:
        check("shard of a run with other neurons" in str(exc), f"refusal: {exc}")
    held = (shards[1].read_bytes(), cz.meta_path_of(shards[1]).read_bytes())
    for target, sh_, why in ((shards[1], None, "is shard 1 of 3 of a run"), (shards[1], (0, 3), "is shard 1 of 3 of a run"),
                             (unbroken, (0, 3), "was not taken as a shard")):
        try:
            cz.take_census(rt, 3, NU, target, **kp, shard=sh_)
            check(False, f"a shard is continued only as itself ({why})")
        except ValueError as exc:
            check(why in str(exc), f"refusal: {exc}")
    check((shards[1].read_bytes(), cz.meta_path_of(shards[1]).read_bytes()) == held and not set(cz.layer_runs(sdir).values()) & set(shards),
          "a shard continued without --shard is refused, never turned into a layer's census")
    try:
        cz.merge_runs([shards[0], shards[1], cz.canary_path_of(shards[1]), shards[2]], sdir / "x.jsonl")
        check(False, "merge_runs refuses a file without its meta")
    except ValueError as exc:
        check("L3-main.s1of3.canary.jsonl has no meta" in str(exc), f"refusal: {exc}")
    try:
        cz.take_census(rt, 3, [5, 0, 6], cz.shard_path(pdir / "L3-unsorted.jsonl", 0, 1), **kp, shard=(0, 1))
        check(False, "a sharded run's neurons out of order are refused")
    except ValueError as exc:
        check("in order" in str(exc), f"refusal: {exc}")
    # a run begun before batch plans (its meta has no batch, plan, dose unit, runtime or job list), its rows regrouped
    old_run = pdir / "L3-old.jsonl"
    old_run.write_text("".join(u_lines[:3] + u_lines[6:40] + u_lines[41:57]))
    stale = {k_: v_ for k_, v_ in um.items() if k_ not in ("batch", "plan", "dose_unit", "runtime", "jobs", "plan_kept", "plan_left", "d_ff",
                                                         "neuron_ids", "ceilings")}
    cz.meta_path_of(old_run).write_text(json.dumps(stale))
    said = []
    res = cz.take_census(rt, 3, NU, old_run, **{**kp, "log": said.append})
    om = json.loads(cz.meta_path_of(old_run).read_text())
    check(res["written"] == 104 - 53 and {cz.row_key(r_): r_["token_ids"] for r_ in cz.read_rows(old_run)} == {cz.row_key(r_): r_["token_ids"] for r_ in U},
          f"an old run (no batch plan in its meta, its rows regrouped by an earlier resume) continues, and ends with the same answers ({res})")
    check(om["plan"] == cz.PLAN_BEFORE and om["batch"] == 5 and om["plan_kept"] is False and om["plan_left"] == "answers written before batch plans"
          and om["dose_unit"] == "stream" and any("left its batch plan" in s_ for s_ in said), f"and its meta says it has left its plan: {om.get('plan_left')}")
    import backend.atlas as atlas_routes

    card_o = atlas_routes.census_card({"path": old_run, "meta": om, "null": {}})
    check(om.get("mixed_runtime") == [cz.RUNTIME_UNKNOWN, om["runtime"]] and any("before runs recorded their runtime" in s_ for s_ in said)
          and card_o["mixed_runtime"] is True and card_o["device"] is None,
          f"an old run continued: its earlier answers' runtime recorded as unknown, the card naming no single card ({om.get('mixed_runtime')})")
    halves = [pdir / "L3-oa.jsonl", pdir / "L3-ob.jsonl"]
    halves[0].write_text("".join(u_lines[:60]))
    halves[1].write_text("".join(u_lines[40:]))
    for h_ in halves:
        cz.meta_path_of(h_).write_text(json.dumps(stale))
    check(cz.merge_runs(halves, pdir / "L3-oj.jsonl") == (124, 104) and cz.merge_runs([halves[0], unbroken], pdir / "L3-ok.jsonl") == (164, 104),
          "old metas still merge, among themselves and with a new run's")
    okm = json.loads(cz.meta_path_of(pdir / "L3-ok.jsonl").read_text())
    check(okm.get("mixed_runtime") == [cz.RUNTIME_UNKNOWN, um["runtime"]] and "neuron_ids" not in okm and "jobs" not in okm,
          "a merge with an old meta: its runtime unknown, recorded beside the new one; no job list claimed")
    parts = [pdir / "L3-pa.jsonl", pdir / "L3-pb.jsonl"]
    cz.take_census(rt, 3, [0, 1], parts[0], **{**kp, "controls": 1}, neurons_spec="0-1")
    cz.take_census(rt, 3, [2], parts[1], **{**kp, "controls": 1}, neurons_spec="2")
    joined = pdir / "L3-pj.jsonl"
    cz.merge_runs(parts, joined)
    jm = json.loads(cz.meta_path_of(joined).read_text())
    check(jm["neurons"] == "0-1 + 2" and jm["neuron_ids"] == "0-2" and jm["jobs"] == cz.jobs_digest(cz.census_jobs([0, 1, 2], 1, cz.DESIGNS["main_rel"]))
          and jm["plan_kept"] is False and jm["plan_left"] == "joined from runs taken in parts, each by its own plan",
          f"runs of other neurons merged: the meta names every part's neurons, and no one plan kept ({jm.get('neuron_ids')}, {jm.get('plan_left')})")
    res = cz.take_census(rt, 3, [0, 1], joined, **{**kp, "controls": 1}, neurons_spec="0-1")
    check(res["written"] == 0 and len(list(cz.read_rows(joined))) == 44, f"and continued with the first part's neurons, not refused ({res})")
    r = subprocess.run(py + [str(ROOT_DIR / "scripts" / "census_run.py"), "--stand-in", "--layer", "2", "--neurons", "0-9", "--random-controls", "2",
                             "--max-new", "4", "--batch", "5", "--run", "L2-sh", "--shard", "1/2"], env=env, capture_output=True, text=True, timeout=600)
    sh_dir = tmp / "atlas" / "stand-in" / "census" / "runs"
    s1 = sh_dir / "L2-sh.s1of2.jsonl"
    check(r.returncode == 0 and s1.exists() and cz.canary_path_of(s1).exists() and "ranked only once its run is joined" in r.stdout
          and not cz.units_path_of(s1).exists(), f"census_run.py --shard 1/2: a shard and its canary, not ranked ({r.stderr[-300:]})")
    r = subprocess.run(py + [str(ROOT_DIR / "scripts" / "census_tools.py"), "merge", str(s1), "--out", str(sh_dir / "L2-sh.jsonl")],
                       env=env, capture_output=True, text=True, timeout=300)
    check(r.returncode != 0 and "Shard 0 of 2 is missing" in r.stderr, f"census_tools.py merge: a missing shard said ({r.stderr[-200:]})")
    r = subprocess.run(py + [str(ROOT_DIR / "scripts" / "census_run.py"), "--stand-in", "--layer", "2", "--neurons", "0-9", "--random-controls", "2",
                             "--max-new", "4", "--batch", "5", "--run", "L2-sh", "--shard", "0/2"], env=env, capture_output=True, text=True, timeout=600)
    check(r.returncode == 0 and f"census_tools.py merge L2-sh.s*of2.jsonl --out L2-sh.jsonl (in {sh_dir}" in r.stdout,
          f"census_run.py --shard: the merge's command printed ({r.stdout[-200:]})")
    globbed = sorted(str(x) for x in sh_dir.glob("L2-sh.s*of*.jsonl"))  # as a shell expands the wider pattern: the canary too
    r2 = subprocess.run(py + [str(ROOT_DIR / "scripts" / "census_tools.py"), "merge", *globbed, "--out", str(sh_dir / "L2-sh.jsonl")],
                        env=env, capture_output=True, text=True, timeout=300)
    check(len(globbed) == 3 and r2.returncode == 0 and "left out L2-sh.s1of2.canary.jsonl" in r2.stdout and "canary: identical on 1 of 1 shard" in r2.stdout
          and len(list(cz.read_rows(sh_dir / "L2-sh.jsonl"))) == 124,
          f"census_tools.py merge <run>.s*of*.jsonl: two shards joined, the canary left out of the join and compared ({r2.stdout[-200:]} {r2.stderr[-200:]})")
    r = subprocess.run(py + [str(ROOT_DIR / "scripts" / "census_tools.py"), "rank", str(s1)], env=env, capture_output=True, text=True, timeout=300)
    check(r.returncode != 0 and "is shard 1 of 2 of a run: join its shards first" in r.stderr and not cz.units_path_of(s1).exists(),
          f"census_tools.py rank: a shard is not ranked ({r.stderr[-200:]})")
    r = subprocess.run(py + [str(ROOT_DIR / "scripts" / "census_run.py"), "--stand-in", "--layer", "2", "--shard", "2/2"], env=env,
                       capture_output=True, text=True, timeout=300)
    check(r.returncode != 0 and "--shard is i/n" in r.stderr, "census_run.py: a shard that does not exist is refused")
    section("batch plans: a stop mid-chunk, a torn batch and three shards each equal an unbroken run row for row; merges, "
            "canaries, refusals, runtimes, old metas", before)

    # ------------------------------------------- 10. one transfer a step, the lens freed, several layers in one process
    before = len(failures)
    texts = [prompt_ids["who"], prompt_ids["snake"] + [7, 9, 11], prompt_ids["orange"], prompt_ids["free"]]
    deltas = torch.stack([w.write_dirs(3, [5])[0] * 2.0 * p50, torch.zeros(D), torch.from_numpy(ctrl[0]) * -0.7 * p50,
                          w.write_dirs(3, [9])[0] * 0.5 * p50])
    ivw = ao.Interventions(rows=4, adds={3: deltas})
    first_piece = next(ao.write_rows(lm, rt.lm_head_weight, [texts[1]], None, max_new=1, stop_ids=stop, bos_id=tok.bos_token_id)).token_id
    for tk in (5, 0):
        for iv_ in (ivw, None):
            args_ = (lm, rt.lm_head_weight, texts, iv_)
            kw_ = dict(max_new=14, stop_ids=list(stop) + [first_piece], bos_id=tok.bos_token_id, softcap=ao.softcap_of(rt.hf_model), top_k=tk)
            new_s, old_s = list(ao.write_rows(*args_, **kw_)), list(write_rows_per_row(*args_, **kw_))
            check(new_s == old_s and len(new_s) > 14 and any(s_.end for s_ in new_s),
                  f"write_rows, one transfer a step: the same Steps as read row by row, bit for bit (top_k {tk}, "
                  f"{'pushed' if iv_ is not None else 'unpushed'}; {len(new_s)} steps)")
    rt2, _ = stand_in()
    rt2.eager_model._lens_jacobians["jpp"] = rt2.jpp_lenses  # as TheodoliteRuntime.load registers it with interp-engine
    card2 = rt2.lens_card()
    sys.path.insert(0, str(ROOT_DIR / "scripts"))
    import census_run

    lid = census_run.free_lens(rt2)
    check(lid == card2["id"] and lid is not None and not rt2.jpp_lenses and not rt2.eager_model._lens_jacobians["jpp"] and rt2.lens_card() is None,
          f"free_lens: the J++ lens freed, interp-engine's handle too, its id kept ({lid})")
    lr = tmp / "runs" / "L3-nolens.jsonl"
    cz.take_census(rt2, 3, [0, 5], lr, **{**kw, "max_new": 4}, lens_id=lid)
    check(len(list(cz.read_rows(lr))) == 4 + 4 * 10 and json.loads(cz.meta_path_of(lr).read_text())["lens"] == lid,
          "a census taken with the lens freed: every answer written (nothing it calls reads the lens), the lens's id in the meta")
    r = subprocess.run(py + [str(ROOT_DIR / "scripts" / "census_run.py"), "--stand-in", "--layers", "1,3-4", "--neurons", "0-1", "--random-controls",
                             "1", "--max-new", "4", "--batch", "8", "--run", "L{layer}-depth", "--no-rank"], env=env, capture_output=True, text=True, timeout=600)
    runs_d = {l_: tmp / "atlas" / "stand-in" / "census" / "runs" / f"L{l_}-depth.jsonl" for l_ in (1, 3, 4)}
    metas_d = {l_: json.loads(cz.meta_path_of(p_).read_text()) for l_, p_ in runs_d.items() if p_.exists()}
    check(r.returncode == 0 and sorted(metas_d) == [1, 3, 4] and all(m_["layer"] == l_ and len(list(cz.read_rows(runs_d[l_]))) == 34 for l_, m_ in metas_d.items())
          and len({m_["p50"] for m_ in metas_d.values()}) == 3, f"census_run.py --layers 1,3-4: three runs, each its own layer and push unit ({r.stderr[-300:]})")
    check("freed from the card" in r.stdout and all(m_["lens"] == card2["id"] for m_ in metas_d.values()),
          "census_run.py frees the lens after load, and each meta names it")
    r = subprocess.run(py + [str(ROOT_DIR / "scripts" / "census_run.py"), "--stand-in", "--layers", "1,3", "--run", "depth"], env=env,
                       capture_output=True, text=True, timeout=300)
    check(r.returncode != 0 and "{layer}" in r.stderr, "census_run.py: several layers under one run name are refused")
    r = subprocess.run(py + [str(ROOT_DIR / "scripts" / "census_run.py"), "--stand-in", "--layers", "1,9"], env=env, capture_output=True, text=True, timeout=300)
    check(r.returncode != 0 and "Layer 9 does not exist" in r.stderr, "census_run.py --layers: a layer that does not exist is refused")
    section("write_rows reads each step back in one transfer, bit for bit; the lens freed; several layers in one process", before)

    # --------------------------------------------------------------------------------------- 11. a two-stage census
    before = len(failures)
    sdir5 = tmp / "atlas" / "stand-in" / "census" / "runs"
    base_r = py + [str(ROOT_DIR / "scripts" / "census_run.py"), "--stand-in", "--max-new", "4", "--batch", "8"]
    cr = base_r + ["--layer", "5"]
    r = subprocess.run(cr + ["--neurons", "0-11", "--random-controls", "3"], env=env, capture_output=True, text=True, timeout=600)
    screen = sdir5 / "L5-main_rel.jsonl"
    s_units = cz.units_path_of(screen)
    check(r.returncode == 0 and s_units.exists(), f"a screen taken and ranked ({r.stderr[-300:]})")
    ranked = [u_ for u_ in cz.read_rows(s_units) if u_["kind"] == "neuron"]
    top4 = sorted(u_["neuron"] for u_ in sorted(ranked, key=lambda u_: -u_["mass"])[:4])
    picked, st = cz.retest_of(s_units, 4)
    check(picked == top4 and st == {"stage": 2, "screen": "L5-main_rel.jsonl", "top": 4, "of": 12, "by": "mass", "screen_layer": 5,
                                    "screen_seed": 0, "screen_controls": 3, "screen_design": "main_rel"},
          f"retest_of: the screen's four strongest by mass, and the stage ({picked}, {st})")
    r = subprocess.run(cr + ["--neurons-from", str(s_units), "--top", "4", "--seed", "7", "--random-controls", "14"], env=env,
                       capture_output=True, text=True, timeout=600)
    retest = sdir5 / "L5-main_rel-top4.jsonl"
    rm = json.loads(cz.meta_path_of(retest).read_text()) if retest.exists() else {}
    rrows = list(cz.read_rows(retest)) if retest.exists() else []
    check(r.returncode == 0 and rm.get("stage") == st and rm.get("seed") == 7 and rm.get("random_controls") == 14
          and rm.get("neurons") == cz.neuron_ranges(top4) and sorted({r_["neuron"] for r_ in rrows if r_["kind"] == "neuron"}) == top4
          and len(rrows) == (4 + 14) * 10 + 4, f"census_run.py --neurons-from --top 4 --seed 7: the four retested, the stage in the meta ({r.stderr[-300:]})")
    ru = [u_ for u_ in cz.read_rows(cz.units_path_of(retest))] if cz.units_path_of(retest).exists() else []
    rn = sorted((u_ for u_ in ru if u_["kind"] == "neuron"), key=lambda u_: u_["p"])
    bh = [min(min(rn[j]["p"] * len(rn) / (j + 1) for j in range(i, len(rn))), 1.0) for i in range(len(rn))]
    check(len(rn) == 4 and sum(u_["kind"] == "random" for u_ in ru) == 14 and all(abs(u_["q"] - round(q_, 5)) < 1e-9 for u_, q_ in zip(rn, bh))
          and all(u_["p"] >= 1 / 15 for u_ in rn), "the retest's p against its 14 fresh controls, its q over its four neurons alone")
    check(cz.layer_runs(sdir5)[5] == screen and retest.stat().st_size > screen.stat().st_size,
          "layer_runs serves the screen as the layer's census, not its larger retest")
    r = subprocess.run(py + [str(ROOT_DIR / "scripts" / "census_tools.py"), "rank", str(retest)], env=env, capture_output=True, text=True, timeout=300)
    check(r.returncode == 0 and "a retest (stage 2) of the 4 strongest of the 12 neurons of L5-main_rel.jsonl" in r.stdout, "census_tools.py rank names the stage")
    for extra, why in ((["--layer", "5", "--top", "4", "--seed", "0"], "another --seed"), (["--layer", "5", "--top", "13"], "1 to 12"),
                       (["--layer", "5", "--top", "4", "--neurons", "0-3"], "no --neurons"), (["--layer", "4", "--top", "4"], "screened layer 5")):
        r = subprocess.run(base_r + ["--neurons-from", str(s_units), "--seed", "9"] + extra, env=env, capture_output=True, text=True, timeout=300)
        check(r.returncode != 0 and why in r.stderr, f"a retest refused ({why}): {r.stderr[-200:]}")
    # the screen ranked again (another neuron now among its four strongest): the same retest command is refused, the
    # retest's neurons fixed when it was begun
    ranked_text = s_units.read_text()
    out_of = next(u_["neuron"] for u_ in ranked if u_["neuron"] not in top4)
    lifted = [json.loads(x_) for x_ in ranked_text.splitlines()]
    s_units.write_text("".join(json.dumps({**u_, "mass": u_["mass"] + 1000.0} if (u_["kind"], u_["neuron"]) == ("neuron", out_of) else u_) + "\n"
                               for u_ in lifted))
    held =(retest.read_bytes(), cz.meta_path_of(retest).read_bytes())
    r = subprocess.run(cr + ["--neurons-from", str(s_units), "--top", "4", "--seed", "7", "--random-controls", "14"], env=env,
                       capture_output=True, text=True, timeout=600)
    s_units.write_text(ranked_text)
    check(r.returncode != 0 and "fixed when it was begun" in r.stderr and (retest.read_bytes(), cz.meta_path_of(retest).read_bytes()) == held,
          f"a retest whose screen was ranked again is refused, not grown ({r.stderr[-200:]})")
    card = atlas_routes.census_card({"path": retest, "meta": rm, "null": {}})
    check(card["stage"] == st and card["dose_unit"] == "stream" and card["device"] == "cpu" and card["mixed_runtime"] is False and card["partial"] is None,
          f"the census card names the stage, the unit and the card ({card})")
    section("a two-stage census: a screen's strongest retested against fresh controls, p and q over them alone; refusals; the card", before)

    # ------------------------------------------------------------ 12. the null on the real checkpoint's own weights
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

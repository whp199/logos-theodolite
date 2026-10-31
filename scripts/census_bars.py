#!/usr/bin/env python3
"""The bars a volunteer's run of another card class is held to (census_volunteer.ACCEPT_OTHER_CLASS's log-probability
bars), measured on the home card: how far its log-probabilities move when only its arithmetic changes, and how far when
the push is wrong by a little.

    .venv/bin/python -I scripts/census_bars.py                     # the kit's layers on the home card (~30 min, GPU)
    .venv/bin/python -I scripts/census_bars.py --also 47=<a volunteer's L47 canary>   # set one beside them (not used)
    .venv/bin/python -I scripts/census_bars.py --stand-in          # the CPU stand-in, a dry run of this script

Why. The bars of 30 October 2026 (median 0.01, 99th percentile 0.10 of |difference| on the pieces where the home card
is not sure) rested on an estimate that another card moves a log-probability by about 1e-3, not on a measurement. The
first canary from another card (an RTX PRO 6000 Blackwell, layer 47, 31 October 2026) matched the home card's words, its
first word-pieces and its unpushed answer, and missed both bars (0.0132 and 0.138). This script was written, and the
rule below fixed, before any of its measurements were taken; the volunteer's canary does not enter the rule.

What is written. Each kit layer's plan batch 0 (take_census, canary_only), with the kit's settings, in these variants:
  - control: as the kit writes it (batch 64). Every row must equal the home canary's bit for bit, at every layer, or
    nothing is derived: the instrument would have moved since the home canaries were taken.
  - arithmetic only: batch 16, 32 and 48 (a smaller batch's batch 0 is a subset of the kit's: plan_batches), and batch
    64 with cuBLAS's reduced-precision bf16 sums turned off. The same model, push and answers; only the card's sums
    change, as on another card.
  - near misses: batch 64 with the push 1, 2, 5 and 10% longer, and the same shorter.
Each is set against the home canary as the intake sets a spot answer against the home card's (compare_rows: the
pieces both wrote alike before they part, where the home card's log-probability is below LP_SURE), the pieces of all
layers pooled for each variant.

The rule, for each of the median (lp_median) and the 99th percentile (lp_p99) of |difference|:
  - the floor F is the largest value among the arithmetic-only variants (at least FLOOR_MIN, the rows' rounding step);
  - a near miss of size s is M(s), the smaller of its two signs' values; a sign whose pooled pieces number fewer than
    MIN_PIECES does not count (its answers part too soon to read, and the first-piece rule is what catches it);
  - the bar is sqrt(F x M(s)) for the smallest s where M(s) >= GAP x F: halfway, on a log scale, between what the
    card's own arithmetic does and the smallest push error the statistic tells apart from it by GAP;
  - where no s qualifies, the statistic cannot tell a push error of 10% from arithmetic, and its bar is dropped (None).
The first-piece bar (0.90) and the unpushed answers' bar (16 word-pieces) are kept as they were.

The results go to experiments/results/census_other_card_bars/ (bars.json, and bars.txt in words); the rows written, to
<atlas folder>/census/bars/. Nothing here changes ACCEPT_OTHER_CLASS: that is done by hand, from bars.json, before any
spot chunk of a run it judges is drawn. The :8765 server must be stopped (the model is loaded here).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend import census as cz  # noqa: E402
from backend import census_volunteer as cv  # noqa: E402

SCALES = (0.01, 0.02, 0.05, 0.10)  # near misses: the push this much longer and shorter
BATCHES = (16, 32, 48)  # arithmetic only: these batch sizes against the kit's
GAP = 4.0  # a near miss qualifies when its value is at least GAP x the floor
MIN_PIECES = 200  # a near miss's sign with fewer pooled pieces does not count
FLOOR_MIN = 1e-4  # the rows keep log-probabilities to 4 decimals
STATS = ("lp_median", "lp_p99")
OUT = ROOT_DIR / "experiments" / "results" / "census_other_card_bars"


def quantile(xs: List[float], q: float) -> Optional[float]:
    return cv._quantile(xs, q)


def lp_diffs(theirs: Dict[Any, Dict[str, Any]], ours: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """compare_rows' pieces kept whole: for each of `ours` with a row in `theirs`, |difference| of the log-probabilities
    on the pieces both wrote alike before they part, where ours is below LP_SURE; with the counts compare_rows gives."""
    diffs: List[float] = []
    present = first = same = 0
    prefixes: List[int] = []
    for r in ours:
        t = theirs.get(cz.row_key(r))
        if t is None:
            continue
        present += 1
        same += t == r
        a, b = r.get("token_ids") or [], t.get("token_ids") or []
        k = cv.shared_prefix(a, b)
        prefixes.append(k)
        first += bool(a and b and int(a[0]) == int(b[0]))
        la, lb = r.get("logprob") or [], t.get("logprob") or []
        diffs.extend(abs(float(la[i]) - float(lb[i])) for i in range(min(k, len(la), len(lb))) if float(la[i]) < cv.LP_SURE)
    return {"present": present, "identical": same, "first": first, "prefixes": prefixes, "diffs": diffs}


def summarize(parts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Several layers' lp_diffs pooled: the statistics the bars are read from."""
    diffs = [d for p in parts for d in p["diffs"]]
    present = sum(p["present"] for p in parts)
    prefixes = [float(x) for p in parts for x in p["prefixes"]]
    return {"answers": present, "identical": sum(p["identical"] for p in parts),
            "first_piece": (sum(p["first"] for p in parts) / present) if present else None,
            "prefix_median": quantile(prefixes, 0.5), "lp_pieces": len(diffs), "lp_median": quantile(diffs, 0.5),
            "lp_p99": quantile(diffs, 0.99), "lp_max": max(diffs) if diffs else None}


def variants(base_batch: int, batches: Sequence[int]) -> List[Dict[str, Any]]:
    out = [{"name": "control", "group": "control", "batch": base_batch, "scale": 1.0, "reduced": True}]
    out += [{"name": f"batch{b}", "group": "arithmetic", "batch": b, "scale": 1.0, "reduced": True} for b in batches]
    out.append({"name": f"batch{base_batch}-full-sums", "group": "arithmetic", "batch": base_batch, "scale": 1.0, "reduced": False})
    for s in SCALES:
        for sign in (1, -1):
            out.append({"name": f"push{'+' if sign > 0 else '-'}{round(s * 100):g}%", "group": "miss", "s": s, "sign": sign,
                        "batch": base_batch, "scale": 1.0 + sign * s, "reduced": True})
    return out


def derive(pooled: Dict[str, Dict[str, Any]], vs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The rule in the module's docstring, applied to each variant's pooled statistics."""
    out: Dict[str, Any] = {}
    for stat in STATS:
        arith = [v["name"] for v in vs if v["group"] == "arithmetic"]
        floor = max([FLOOR_MIN] + [float(pooled[n][stat] or 0.0) for n in arith])
        tried = []
        bar = chosen = None
        for s in SCALES:
            signs = [v["name"] for v in vs if v["group"] == "miss" and v["s"] == s]
            vals = [float(pooled[n][stat]) for n in signs if pooled[n]["lp_pieces"] >= MIN_PIECES and pooled[n][stat] is not None]
            m = min(vals) if len(vals) == len(signs) else None  # both signs must be readable
            ok = m is not None and m >= GAP * floor
            tried.append({"s": s, "M": m, "qualifies": ok})
            if ok:
                bar, chosen = math.sqrt(floor * m), s
                break
        out[stat] = {"floor": floor, "floor_from": max(arith, key=lambda n: float(pooled[n][stat] or 0.0)),
                     "near_misses": tried, "miss_caught": chosen,
                     "bar": None if bar is None else float(f"{bar:.2g}")}
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", default="", help="the layers, counting from 0 (default: the kit's and 45)")
    ap.add_argument("--also", action="append", default=[], help="LAYER=PATH: a canary set beside the variants, reported, "
                    "not used by the rule")
    ap.add_argument("--stand-in", action="store_true", help="the CPU stand-in at the kit's dry-run settings")
    ap.add_argument("--out", default=str(OUT), help="where bars.json and bars.txt go")
    a = ap.parse_args(argv)

    kit = cv.read_kit(dry=a.stand_in)
    if a.stand_in:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["HF_HUB_OFFLINE"] = "1"
    layers = sorted({int(x) for x in a.layers.split(",") if x.strip()}) if a.layers else \
        sorted(set(kit["layers"]) | ({45} if not a.stand_in else set()))
    base = int(kit["KIT_BATCH"])
    batches = [b for b in BATCHES if b < base] if not a.stand_in else [b for b in (1, 2, 3) if b < base]
    vs = variants(base, batches)

    import torch

    from backend.atlas_index import AtlasIndex
    from backend.causal_check import language_model
    from backend.runtime_factory import weights_card
    from census_run import free_lens

    torch.set_grad_enabled(False)
    if a.stand_in:
        from atlas_stand_in import stand_in, use_stand_in_atlas

        use_stand_in_atlas()
        rt, _vault = stand_in()
    else:
        free, total = torch.cuda.mem_get_info()
        if free < 26 * 2**30:
            raise SystemExit(f"Only {free / 2**30:.1f} GB of the card is free: stop the :8765 server and any census first.")
        from backend.runtime_factory import make_runtime

        rt = make_runtime(load_jlens=False).load()
    lens_id = free_lens(rt) if getattr(rt, "jpp_lenses", None) else None
    from backend import w4a16_fast

    w4a16_fast.install(rt, kit["KIT_UNPACKER"] if not a.stand_in else "reference", log=lambda s: print(s, flush=True))
    from backend import atlas_ops as ao

    lm = language_model(rt.hf_model)
    L, N = len(lm.layers), ao.ffn_width(lm, 0)
    index = AtlasIndex(model=getattr(rt, "model_id", None), shape=(L, N), weights=weights_card(rt))
    stats = index.get("act_scale")
    if stats is not None:
        p50s = [float(x) for x in stats["resid_q"][:, 1]]
        source = f"the stats index ({stats['meta'].get('corpus')}, {int(stats['tokens'])} word-pieces)"
    elif a.stand_in:  # as census_run.py reads it with no stats index
        from backend.mcp_scientist import _study_ids

        p50s = ao.stream_p50(ao.activation_extremes(lm, _study_ids(rt)))
        source = "the study's 36 texts (no stats index yet)"
    else:
        raise SystemExit("There is no stats index: the push's unit is read from it, as the kit's runs read it.")
    neurons = cz.parse_neurons(str(kit["KIT_NEURONS"]), N, int(kit["KIT_SEED"]))
    rows_dir = (cz.census_dir().parent / "bars") if not a.stand_in else (cz.census_dir().parent / "bars-stand-in")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    written: Dict[str, Dict[int, List[Dict[str, Any]]]] = {v["name"]: {} for v in vs}
    t0 = time.time()
    for v in vs:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = bool(v["reduced"])
        for layer in layers:
            d = rows_dir / stamp / v["name"]
            d.mkdir(parents=True, exist_ok=True)
            run = d / f"L{layer}-{kit['KIT_DESIGN']}.jsonl"
            cz.take_census(rt, layer, neurons, run, p50=p50s[layer] * v["scale"], p50_source=source,
                           design_name=kit["KIT_DESIGN"], controls=int(kit["KIT_CONTROLS"]), seed=int(kit["KIT_SEED"]),
                           max_new=int(kit["KIT_MAX_NEW"]), top_k=5, batch=v["batch"], neurons_spec=str(kit["KIT_NEURONS"]),
                           p50_file=index.path("act_scale") if stats is not None else None, lens_id=lens_id, canary_only=True,
                           log=lambda s: None)
            written[v["name"]][layer] = list(cz.read_rows(cz.canary_path_of(run)))
            print(f"{v['name']:<20} L{layer}: {len(written[v['name']][layer])} answers ({time.time() - t0:.0f} s)", flush=True)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    # the reference: the home canaries (the stand-in has none: its own control stands in for them)
    home: Dict[int, List[Dict[str, Any]]] = {}
    for layer in layers:
        hp, _ = cv.home_canary(layer, kit["KIT_DESIGN"])
        home[layer] = list(cz.read_rows(hp)) if hp.exists() and not a.stand_in else written["control"][layer]
    per: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for v in vs:
        per[v["name"]] = {}
        for layer in layers:
            mine = {cz.row_key(r): r for r in written[v["name"]][layer]}
            ours = [r for r in home[layer] if cz.row_key(r) in mine]
            per[v["name"]][layer] = lp_diffs(mine, ours)
            if v["group"] == "control":  # the intake's own comparison must read the same pieces
                c = cv.compare_rows(mine, ours)
                assert c["lp_median"] == quantile(per[v["name"]][layer]["diffs"], 0.5)
    pooled = {n: summarize(list(per[n].values())) for n in per}
    control = pooled["control"]
    moved = control["identical"] != control["answers"] or control["answers"] != sum(len(home[l]) for l in layers)
    result: Dict[str, Any] = {
        "taken_at": stamp, "layers": layers, "kit": kit["KIT"], "stand_in": a.stand_in,
        "rule": {"scales": SCALES, "batches": batches, "gap": GAP, "min_pieces": MIN_PIECES, "floor_min": FLOOR_MIN,
                 "lp_sure": cv.LP_SURE, "kept": {k: cv.ACCEPT_OTHER_CLASS[k] for k in ("first_piece", "baseline_tie")}},
        "runtime": cz.runtime_record(next(lm.parameters()).device, act_scale=index.path("act_scale") if stats is not None else None),
        "variants": [{**v, **pooled[v["name"]], "by_layer": {str(l): {k: x for k, x in summarize([per[v["name"]][l]]).items()}
                                                             for l in layers}} for v in vs],
        "control_identical": not moved,
        "bars": None if moved else derive(pooled, vs),
        "also": {},
        "old_bars": {k: cv.ACCEPT_OTHER_CLASS[k] for k in STATS},
    }
    for spec in a.also:
        lay, _, path = spec.partition("=")
        rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
        mine = {cz.row_key(r): r for r in rows}
        hp, _ = cv.home_canary(int(lay), kit["KIT_DESIGN"])
        ref = list(cz.read_rows(hp))
        result["also"][f"L{lay} {Path(path).name}"] = {**summarize([lp_diffs(mine, [r for r in ref if cz.row_key(r) in mine])]),
                                                       "fingerprint": cv.rows_fingerprint(rows)["sha256"]}

    out = Path(a.out) if not a.stand_in else Path(a.out) / "stand-in"
    out.mkdir(parents=True, exist_ok=True)
    (out / "bars.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    f = lambda x: "-" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))
    lines = [f"census bars, measured {stamp} on {result['runtime'].get('device')}, layers {layers[0]}-{layers[-1]}"
             + (" (stand-in)" if a.stand_in else ""), ""]
    lines.append(f"{'variant':<20} {'answers':>7} {'same':>5} {'first':>6} {'prefix':>6} {'pieces':>6} {'median':>7} {'p99':>7} {'max':>7}")
    for v in result["variants"] + [{"name": k, **x} for k, x in result["also"].items()]:
        lines.append(f"{v['name'][:20]:<20} {v['answers']:>7} {v['identical']:>5} {f(v['first_piece']):>6} {f(v['prefix_median']):>6} "
                     f"{v['lp_pieces']:>6} {f(v['lp_median']):>7} {f(v['lp_p99']):>7} {f(v['lp_max']):>7}")
    lines.append("")
    if moved:
        lines.append(f"the control is not the home canaries' bits ({control['identical']} of {control['answers']} the same): "
                     "no bars are derived")
    else:
        for stat, b in result["bars"].items():
            lines.append(f"{stat}: floor {f(b['floor'])} ({b['floor_from']}); near misses "
                         + ", ".join(f"{round(t['s'] * 100):g}% {f(t['M'])}{' ok' if t['qualifies'] else ''}" for t in b["near_misses"])
                         + f" -> bar {b['bar']} (was {result['old_bars'][stat]})"
                         + (f", built to catch a push {round(b['miss_caught'] * 100):g}% off" if b["miss_caught"] else ", dropped"))
    (out / "bars.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {out / 'bars.json'}, {out / 'bars.txt'}; the rows in {rows_dir / stamp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""The near-tie bar a volunteer's run of another card class holds its unpushed answers to (census_volunteer.
ACCEPT_OTHER_CLASS["baseline_tie"]), checked on the home card before it is used.

    .venv/bin/python -I scripts/census_ties.py               # on the home card (~10 min, GPU)
    .venv/bin/python -I scripts/census_ties.py --stand-in    # the CPU stand-in, a dry run of this script

Why. The rule of 30 October 2026 asked each unpushed answer of another card to share at least 16 word-pieces with the
home card's. Layer 47's run (an RTX PRO 6000 Blackwell, 31 October 2026) shared all 48 in three of its six answers and
parted sooner in three: orange at piece 5, free at 21, chair at 27, each at a piece the home card wrote with
probability 0.50, 0.41 or 0.50, where that card wrote the runner-up. A greedy answer parts where its top two pieces
nearly tie, and any card's arithmetic may tip such a tie. After those answers were seen, the rule was replaced by: an
unpushed answer is the home card's, or parts from it only at a word-piece the home card wrote with probability below
TIE (0.6). This script, written and committed before it runs, checks that rule on the home card itself; layer 47's
answers do not enter it.

What is written. The six unpushed answers as the kit's plan writes them: chunk 0 of a layer-45 run with the kit's
settings (take_census, shard 0 of the plan's C chunks: every question's unpushed answer is in chunk 0, in its
question's first batch, beside that batch's pushed answers), in these variants:
  - control: batch 64 as the kit writes it. Each unpushed answer must equal the home card's
    (volunteers/home_canaries/baselines.json) bit for bit, or nothing is concluded;
  - arithmetic only: batch 1, 2, 4, 8, 16, 24, 32, 40, 48 and 56, and batch 64 with cuBLAS's reduced-precision bf16
    sums turned off. On the card a row's numbers depend on its batch's size, not on which rows share it
    (backend/census.py, batch plans), so these are the home card's own answers under other arithmetic.

The rule. Each variant's answer is set against the home card's (census_volunteer.baseline_part): where it parts, and
the probability the home card gave its piece there. If no variant parts at a piece the home card wrote with
probability TIE or more, the bar stands at TIE. If one does, the bar is raised to the next multiple of 0.05 above the
highest such probability (the home card's own arithmetic must pass), and that is said.

The results go to experiments/results/census_unpushed_ties/ (ties.json, and ties.txt in words); the rows written, to
<atlas folder>/census/ties/. Nothing here changes ACCEPT_OTHER_CLASS: that is done by hand, from ties.json. The :8765
server must be stopped (the model is loaded here).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend import census as cz  # noqa: E402
from backend import census_volunteer as cv  # noqa: E402

TIE = 0.6  # parting is taken below this probability of the home card's piece
STEP = 0.05  # a bar raised goes to the next multiple of this above the surest parting
BATCHES = (1, 2, 4, 8, 16, 24, 32, 40, 48, 56)  # arithmetic only: these batch sizes against the kit's
LAYER = 45  # the home card's unpushed answers are layer 45's (the same at every layer: nothing pushes them)
OUT = ROOT_DIR / "experiments" / "results" / "census_unpushed_ties"


def variants(base: int, batches: List[int]) -> List[Dict[str, Any]]:
    vs = [{"name": "control", "group": "control", "batch": base, "reduced": True}]
    vs += [{"name": f"batch{b}", "group": "arithmetic", "batch": b, "reduced": True} for b in batches]
    vs.append({"name": f"batch{base}-full-sums", "group": "arithmetic", "batch": base, "reduced": False})
    return vs


def bar_of(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The rule: TIE, unless a variant of the home card parts at a piece it wrote at TIE or more."""
    sure = [x for x in parts if x["p"] is not None and x["p"] >= TIE]
    if not sure:
        return {"bar": TIE, "raised": False, "surest": max((x["p"] for x in parts if x["p"] is not None), default=None)}
    top = max(x["p"] for x in sure)
    return {"bar": round(math.floor(top / STEP + 1e-9) * STEP + STEP, 2), "raised": True, "surest": top}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stand-in", action="store_true", help="the CPU stand-in at the kit's dry-run settings")
    ap.add_argument("--out", default=str(OUT), help="where ties.json and ties.txt go")
    a = ap.parse_args(argv)

    kit = cv.read_kit(dry=a.stand_in)
    if a.stand_in:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["HF_HUB_OFFLINE"] = "1"
    base = int(kit["KIT_BATCH"])
    vs = variants(base, [b for b in BATCHES if b < base] if not a.stand_in else [b for b in (1, 2, 3) if b < base])
    layer = LAYER if not a.stand_in else int(kit["layers"][0])
    design = kit["KIT_DESIGN"]

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
        free, _total = torch.cuda.mem_get_info()
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
        p50 = float(stats["resid_q"][layer, 1])
        source = f"the stats index ({stats['meta'].get('corpus')}, {int(stats['tokens'])} word-pieces)"
    elif a.stand_in:  # as census_run.py reads it with no stats index
        from backend.mcp_scientist import _study_ids

        p50 = float(ao.stream_p50(ao.activation_extremes(lm, _study_ids(rt)))[layer])
        source = "the study's 36 texts (no stats index yet)"
    else:
        raise SystemExit("There is no stats index: the push's unit is read from it, as the kit's runs read it.")
    neurons = cz.parse_neurons(str(kit["KIT_NEURONS"]), N, int(kit["KIT_SEED"]))
    jobs = cz.census_jobs(neurons, int(kit["KIT_CONTROLS"]), cz.DESIGNS[design], list(cz.design_questions(design)))
    plan_name = cz.DESIGN_PLAN.get(design, cz.PLAN)
    rows_dir = (cz.census_dir().parent / "ties") if not a.stand_in else (cz.census_dir().parent / "ties-stand-in")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    written: Dict[str, Dict[str, Dict[str, Any]]] = {}
    t0 = time.time()
    for v in vs:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = bool(v["reduced"])
        C = cz.plan_batches(jobs, v["batch"], plan_name)[-1][0] + 1
        d = rows_dir / stamp / v["name"]
        d.mkdir(parents=True, exist_ok=True)
        run = d / f"L{layer}-{design}.s0of{C}.jsonl"
        cz.take_census(rt, layer, neurons, run, p50=p50, p50_source=source, design_name=design,
                       controls=int(kit["KIT_CONTROLS"]), seed=int(kit["KIT_SEED"]), max_new=int(kit["KIT_MAX_NEW"]), top_k=5,
                       batch=v["batch"], neurons_spec=str(kit["KIT_NEURONS"]), shard=(0, C),
                       p50_file=index.path("act_scale") if stats is not None else None, lens_id=lens_id, log=lambda s: None)
        written[v["name"]] = {r["prompt"]: r for r in cz.read_rows(run) if r["kind"] == "baseline"}
        print(f"{v['name']:<20} chunk 0 of {C}: {len(written[v['name']])} unpushed answers ({time.time() - t0:.0f} s)", flush=True)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    # the reference: the home card's unpushed answers (the stand-in has none: its own control stands in for them)
    hb = cv.HOME_CANARIES / "baselines.json"
    home = json.loads(hb.read_text(encoding="utf-8"))["baselines"] if hb.exists() and not a.stand_in else \
        {q: {"token_ids": r["token_ids"], "logprob": r["logprob"]} for q, r in written["control"].items()}
    parts: List[Dict[str, Any]] = []
    for v in vs:
        for q, h in home.items():
            r = written[v["name"]].get(q)
            if r is None:
                parts.append({"variant": v["name"], "group": v["group"], "question": q, "missing": True, "k": None, "n": len(h["token_ids"]), "p": None})
                continue
            k, n, p = cv.baseline_part(h["token_ids"], h["logprob"], r["token_ids"])
            parts.append({"variant": v["name"], "group": v["group"], "question": q, "k": k, "n": n, "p": p,
                          "identical": r["token_ids"] == h["token_ids"] and r["logprob"] == h["logprob"]})
    ctrl = [x for x in parts if x["group"] == "control"]
    moved = any(x.get("missing") or not x["identical"] for x in ctrl) or len(ctrl) != len(home)
    arith = [x for x in parts if x["group"] == "arithmetic"]
    result: Dict[str, Any] = {
        "taken_at": stamp, "layer": layer, "kit": kit["KIT"], "stand_in": a.stand_in,
        "rule": {"tie": TIE, "step": STEP, "batches": [v["batch"] for v in vs if v["group"] == "arithmetic"]},
        "runtime": cz.runtime_record(next(lm.parameters()).device, act_scale=index.path("act_scale") if stats is not None else None),
        "parts": parts, "control_identical": not moved,
        "verdict": None if moved else bar_of(arith),
        "old_bar": {"baseline_shared": 16},
    }
    out = Path(a.out) if not a.stand_in else Path(a.out) / "stand-in"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ties.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    lines = [f"census unpushed ties, measured {stamp} on {result['runtime'].get('device')}, layer {layer}"
             + (" (stand-in)" if a.stand_in else ""), ""]
    qs = list(home)
    lines.append(f"{'variant':<20} " + " ".join(f"{q[:12]:>12}" for q in qs))
    for v in vs:
        cells = []
        for q in qs:
            x = next(y for y in parts if y["variant"] == v["name"] and y["question"] == q)
            cells.append("missing" if x.get("missing") else ("same" if x["identical"] else
                         (f"{x['k']}/{x['n']} lp" if x["k"] == x["n"] else f"{x['k']} @{x['p']:.2f}")))
        lines.append(f"{v['name'][:20]:<20} " + " ".join(f"{c:>12}" for c in cells))
    lines += ["", "(same: bit for bit; k/n lp: every piece alike, a log-probability not; k @p: parts at piece k, a piece the "
              "home card wrote with probability p)", ""]
    if moved:
        lines.append("the control is not the home card's unpushed answers bit for bit: nothing is concluded")
    else:
        vd = result["verdict"]
        n_part = sum(1 for x in arith if x["k"] is not None and x["k"] < x["n"])
        lines.append(f"{n_part} of {len(arith)} arithmetic-only answers part from the home card's; the surest parting at "
                     f"{vd['surest'] if vd['surest'] is not None else '-'}")
        lines.append(f"bar: {vd['bar']}" + (f" (raised from {TIE}: one of the home card's own variants parts at a piece it wrote at "
                                             f"{vd['surest']})" if vd["raised"] else f" (the rule's {TIE} stands)"))
    (out / "ties.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {out / 'ties.json'}, {out / 'ties.txt'}; the rows in {rows_dir / stamp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

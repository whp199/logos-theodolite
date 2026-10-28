#!/usr/bin/env python3
"""Take a census of one layer (backend/census.py) on the GPU: every neuron asked (or a subset), pushed on the census's
questions, one greedy answer per (question, dose), beside unpushed answers and random-direction controls of the same
push size, shaped as the layer's neurons' writes are (unit(w_post * r)); every answer kept as one JSON line, then the run
ranked (scripts/census_tools.py rank).

    .venv/bin/python scripts/census_run.py --layer 45 --neurons random:384 --random-controls 64 --run L45-r384
    .venv/bin/python scripts/census_run.py --layer 45 --neurons all
    .venv/bin/python scripts/census_run.py --layer 45 --design calib_rel --neurons random:32 --random-controls 16 --run L45-calib
    .venv/bin/python scripts/census_run.py --stand-in --layer 3 --neurons 0-15 --random-controls 4 --max-new 24

Layers count from 0 (page layer = layer + 1), as every script and field of the atlas. The runs go to the census folder
(`$THEODOLITE_CENSUS_DIR`, else `<atlas folder>/census/runs`; a stand-in's under its own `stand-in/`), named by --run
(default `L<layer>-<design>`), with a `.meta.json` beside each. The same command again continues a stopped run (rows
already written are skipped; a run taken with other settings is refused, not mixed). The atlas page reads the main
run of each layer (`main_rel` before others, then the largest) as soon as it has rows.

The push is the atlas's "add": dose x p50 along the neuron's write direction at the stream leaving the layer, p50 the
median length of that stream (from the stats index when built, else read here over the study's 36 texts; the run's
meta records which). Designs: main_rel, {who, snake, orange} x {-0.7, +0.5, +0.8} and {free} x {+3}, neuron-explorer's
doses, chosen on Qwen3.5-4B and not yet calibrated on this model; calib_rel, a ladder ±{0.1, 0.25, 0.5, 1, 2, 4} on
{who, orange, free}, for finding where effects begin and text breaks on this model: take it on a few layers before a
full main_rel run. The meta also records how loudly the next layer reads the controls against the neurons (read_gain;
a fair null is near 1, and the run warns when it is not).

What it costs, measured in a pilot on one RTX 5090 (27 October 2026): every decode step of the 31B takes about 0.71 s
whatever the batch (1 to 64 rows: the runtime's 4-bit weights cost the same each forward), so a run's time is
answers / batch x (max_new + 1) x 0.71 s and the batch should be as large as the card holds. That 0.71 s is
the reference unpacker's (--unpacker reference, the default): lut and compiled give the same bits in a step two
to five times shorter on a 5090 by the arithmetic, not yet timed. Memory: the model and lens take 21.2 GiB; each
row costs about 0.8 MB per word-piece of context on top (a batch of 32 at --max-new 64 peaked 3.1 GiB above the
model), so at --max-new 256 a batch of 32 needs some 7 GiB more and may not fit beside a desktop: lower
--batch or --max-new. A calib_rel run of 8 neurons and 8 controls (580 answers) at --batch 32 --max-new 64 took about 16
minutes. answers = (neurons + controls) x 10 + 4 for main_rel: a whole layer is (21,504 + 256) x 10 + 4 = 217,604
answers, about 3.6 days at --batch 32 --max-new 64 and about two weeks at --max-new 256; the reference's depth curve
used 384 neurons and 64 controls a layer (4,484 answers, about 2 hours at --batch 32 --max-new 64). On disk a whole
layer is about 5 to 6 GB (each answer keeps its five likeliest pieces at every step, some 25 kB an answer at 256
pieces); ranking it reads one answer at a time and keeps only each answer's word ids, well under a gigabyte. Run one GPU
process at a time: this loads the model, as the server and the lab daemon do.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--layer", type=int, required=True, help="the layer to take, counting from 0")
    ap.add_argument("--neurons", default="all", help='"all", "random:384", "2742", "0-99" or "1,5,9-12"')
    ap.add_argument("--design", default="main_rel", help="main_rel | calib_rel")
    ap.add_argument("--random-controls", type=int, default=256, help="random-direction controls spread through the run")
    ap.add_argument("--seed", type=int, default=0, help="draws the controls' directions and random:K's neurons")
    ap.add_argument("--max-new", type=int, default=256, help="word-pieces per answer at most (pushed answers often loop)")
    ap.add_argument("--top-k", type=int, default=5, help="the likeliest pieces kept at each step")
    ap.add_argument("--batch", type=int, default=32, help="answers written together")
    ap.add_argument("--run", default="", help="the run's name (default L<layer>-<design>)")
    ap.add_argument("--out-dir", default="", help="the census folder (default: the atlas's)")
    ap.add_argument("--no-rank", action="store_true", help="leave the ranking to the server or census_tools.py rank")
    ap.add_argument("--stand-in", action="store_true", help="run on the CPU stand-in (scripts/atlas_stand_in.py), for a dry run")
    ap.add_argument("--unpacker", default="reference", choices=["reference", "lut", "compiled", "resident"],
                    help="how the 4-bit weights are unpacked (backend/w4a16_fast.py): each gives the reference's bits "
                    "(tested at load on a GPU); lut, compiled and resident are faster, resident needing ~60 GB more")
    a = ap.parse_args(argv)

    if a.stand_in:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["HF_HUB_OFFLINE"] = "1"
    import torch

    from backend import atlas_ops as ao
    from backend import census as cz
    from backend.atlas_index import AtlasIndex
    from backend.causal_check import language_model

    torch.set_grad_enabled(False)
    if a.stand_in:
        from atlas_stand_in import stand_in, use_stand_in_atlas

        use_stand_in_atlas()
        rt, _vault = stand_in()
    else:
        from backend.engine_runtime import TheodoliteRuntime

        rt = TheodoliteRuntime(load_jlens=False).load()
    from backend import w4a16_fast

    try:
        w4a16_fast.install(rt, a.unpacker, log=lambda s: print(s, flush=True))
    except w4a16_fast.Refused as exc:
        raise SystemExit(str(exc))
    lm = language_model(rt.hf_model)
    L, N = len(lm.layers), ao.ffn_width(lm, 0)
    if not 0 <= a.layer < L:
        raise SystemExit(f"Layer {a.layer} does not exist: layers count from 0 to {L - 1} (page layers 1 to {L}).")
    try:
        neurons = cz.parse_neurons(a.neurons, N, a.seed)
    except ValueError as exc:
        raise SystemExit(str(exc))

    stats = AtlasIndex(model=getattr(rt, "model_id", None), shape=(L, N)).get("act_scale")
    if stats is not None:
        p50 = float(stats["resid_q"][a.layer, 1])
        source = f"the stats index ({stats['meta'].get('corpus')}, {int(stats['tokens'])} word-pieces)"
    else:
        from backend.mcp_scientist import _study_ids

        p50 = ao.stream_p50(ao.activation_extremes(lm, _study_ids(rt)))[a.layer]
        source = "the study's 36 texts (no stats index yet)"

    out_dir = Path(a.out_dir) if a.out_dir else cz.census_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{a.run or f'L{a.layer}-{a.design}'}.jsonl"
    stop = threading.Event()

    def on_signal(_sig, _frame):  # stop between steps: the batch under way is dropped, every whole one kept
        print("stopping after this step (the same command continues the run)", flush=True)
        stop.set()

    signal.signal(signal.SIGINT, on_signal)
    try:
        res = cz.take_census(rt, a.layer, neurons, out, p50=p50, p50_source=source, design_name=a.design,
                             controls=a.random_controls, seed=a.seed, max_new=a.max_new, top_k=a.top_k, batch=a.batch,
                             neurons_spec=a.neurons, cancel=stop.is_set, log=lambda s: print(s, flush=True))
    except ValueError as exc:
        raise SystemExit(str(exc))
    print(f"-> {out} ({res['written']} answers written, {res['skipped']} were there)")
    if not a.no_rank and out.exists():
        units = cz.write_units(out)
        print(f"-> {units}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

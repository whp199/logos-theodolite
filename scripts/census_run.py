#!/usr/bin/env python3
"""Take a census of one layer (backend/census.py) on the GPU: every neuron asked (or a subset), pushed on the census's
questions, one greedy answer per (question, dose), beside unpushed answers and random-direction controls of the same
push size, shaped as the layer's neurons' writes are (unit(w_post * r)); every answer kept as one JSON line, then the run
ranked (scripts/census_tools.py rank).

    .venv/bin/python scripts/census_run.py --layer 45 --neurons random:384 --random-controls 64 --run L45-r384
    .venv/bin/python scripts/census_run.py --layer 45 --neurons all      # break_rel, 48 pieces, a batch of 64, 2,048 controls
    .venv/bin/python scripts/census_run.py --layer 45 --design calib_rel --neurons random:32 --random-controls 16 --run L45-calib
    .venv/bin/python scripts/census_run.py --layers 20,30,40,45,50,57 --design calib_ceil --neurons random:32 --random-controls 16
    .venv/bin/python scripts/census_run.py --layer 45 --neurons all --shard 1/4      # one of four shards, at the design's settings
    .venv/bin/python scripts/census_run.py --layer 45 --neurons all --canary-only    # its first batch alone
    .venv/bin/python scripts/census_run.py --layer 45 --neurons-from L45-r384.units.jsonl --top 32 --seed 1 --random-controls 1024 --design break_rel
    .venv/bin/python scripts/census_run.py --layer 45 --design break_rel --neurons all --random-controls 2048 --seed 7 --batch 64 --max-new 48 --unpacker compiled
    .venv/bin/python scripts/census_run.py --stand-in --layer 3 --neurons 0-15 --random-controls 4 --max-new 24
    .venv/bin/python scripts/census_run.py --layer 45 --neurons none --control-kind columns --seed 11 --run L45-columns --unpacker compiled

Watch a run from another terminal with scripts/census_watch.py (a progress bar, its rate and when it should finish).

The model is the W4A16 checkpoint on cuda:0 unless THEODOLITE_GGUF names a GGUF of Gemma 4 or THEODOLITE_DEVICE=cpu
puts it on the CPU (backend/runtime_factory.py; THEODOLITE_GGUF_LENS=none leaves out the J++ lens, which a census does
not read); the run's meta records the model id, a GGUF's carrying its file and fingerprint and one off the GPU or in
float32 its device and dtype (`@cpu/bfloat16`), so a run is continued only where and as it began (ARCHITECTURE §5).

Layers count from 0 (page layer = layer + 1), as every script and field of the atlas; --layers takes several in one
process (the model loaded once), each its own run. The runs go to the census folder (`$THEODOLITE_CENSUS_DIR`, else
`<atlas folder>/census/runs`; a stand-in's under its own `stand-in/`), named by --run (default `L<layer>-<design>`),
with a `.meta.json` beside each. The same command again continues a stopped run (rows already written are skipped; a
run taken with other settings is refused, not mixed). The atlas page reads the main run of each layer (one of the
default design, census.DEFAULT_DESIGN, break_rel since 29 October 2026, before others, then the largest) as soon as it has
rows. Once the model is loaded the J++ lens is freed from the card:
the census reads none of it (each run's meta still names it; a runtime loaded without one, THEODOLITE_GGUF_LENS=none,
names none).

The push is the atlas's "add", along the neuron's write direction at the stream leaving the layer, in one of two units
(--dose-unit): stream, dose x p50, p50 the median length of that stream (from the stats index when built, else read
here over the study's 36 texts; the run's meta records which), the same push for every neuron and control; ceiling,
dose x the neuron's write ceiling, the most it can write by itself, a control pushed by the ceiling of the neuron it is
paired with. Every pushed answer records its unit's ceiling and its push's real share of the stream (rel_push). Designs:
main_rel, {who, snake, orange} x {-0.7, +0.5, +0.8} and {free} x {+3}, neuron-explorer's doses, chosen on Qwen3.5-4B
and not yet calibrated on this model; calib_rel, a ladder ±{0.1, 0.25, 0.5, 1, 2, 4} of the stream on {who, orange,
free}, and calib_ceil, ±{0.5, 1, 2, 4, 8} x each neuron's ceiling on the same questions (taken in the ceiling unit),
for finding where effects begin and text breaks on this model: take them on a few layers and read them with
census_tools.py curves before a full main_rel run; and break_rel, ±1 and ±2 of the stream, chosen for layers 40-57 on
those two ladders (28 October 2026, 64 neurons and 64 controls at each of layers 20, 30, 40, 45, 50 and 57), on which
text breaks first at ±2 at layers 40 and 45, at ±1 for the neurons at 50 and at +2 and -1 for them at 57 (the controls
at ±2 at both), at ±0.5 at layer 20 and ±1 at 30, and calibrated at layers 40-50 by its trial of 29 October 2026
(census.DOSES_CALIBRATED; elsewhere the pages call them not calibrated at that layer). It asks six questions,
who, orange, free and three asked by no other design (bicycle, walk, chair: census.DESIGN_QUESTIONS), read as two
halves of three, 24 answers a unit, its batches planned 12 x batch. Each design is ranked its own way
(census.reading_of): main_rel and the ladders as the reference (pushes of at most 1.6 of the stream, words against
every unit's mean, every question's unpushed words left out); break_rel counts every answer, sets each word against
the controls' mean alone and leaves out only the answer's own question's unpushed words. The ranking file records the
reading it was made with. The meta also records how loudly the next layer reads the controls against the neurons
(read_gain; a fair null is near 1, and the run warns when it is not).

A run's batches are planned from its whole job list and the batch size (chunks of 10 x batch answers, each grouped by
question), and written whole: a run continued after a stop, or after a kill that tore its last batch, and a run taken
in shards write every answer in the same batch as an unbroken run. The meta records the batch and the plan (a run
begun with a batch plan, continued with another --batch, is refused) and the runtime it was written on (the card,
torch, CUDA, transformers, interp-engine, compressed-tensors, the commit, the stats index's sha256; continued on
another, it is warned and recorded, not refused). --shard i/n takes shard i of n of the plan's chunks into
<run>.s<i>of<n>.jsonl, and every shard but 0 writes the plan's first batch again (<run>.s<i>of<n>.canary.jsonl, one
batch's time) for census_tools.py merge to compare with shard 0's (the merge leaves the canaries out of the join); a
shard is continued only as the same shard, the merge refuses a run with a shard or an answer missing unless
--partial, and the ranking comes after it. --canary-only writes that first batch alone (<run>.canary.jsonl, and
<run>.canary.json with the settings and the runtime it was written with) and stops, the run itself not begun: take it
on the home card and on a rented one with the same arguments, and census_tools.py canary says whether the two cards
write the same bits (scripts/pod/pilot.sh --home-canary); one already there is kept only when taken with the same
settings on the same runtime, and is otherwise refused. --neurons-from <run>.units.jsonl --top M retests a screen's M
strongest neurons against fresh controls (another --seed), the census's second stage: its p and q are taken over those
M alone, and its neurons are fixed when it is begun (a screen ranked again since is a new retest, under another --run).
A retest's --design must share no (question, dose) cell with its screen's: on a shared cell a neuron writes again the
answer it was picked for, bit for bit at one batch size, so such a retest is refused (a main_rel screen is retested
with break_rel; a break_rel run needs none: its halves are the held-out test). --control-kind columns draws each
control as a random mix of the layer's own write directions, spread as they are (census.CONTROL_KINDS), not as an
isotropic direction through the post-feed-forward norm; with --neurons none a run is controls alone, and
census_tools.py join sets a layer's run's neurons beside them (same settings, the unpushed answers bit for bit), so the
neurons are read against the new controls without being written again: 2,048 controls of break_rel are 49,158 answers.

What it costs, measured in a pilot on one RTX 5090 (27 October 2026): every decode step of the 31B takes about 0.71 s
whatever the batch (1 to 64 rows: the runtime's 4-bit weights cost the same each forward), so a run's time is
answers / batch x (max_new + 1) x 0.71 s and the batch should be as large as the card holds. That 0.71 s is the
reference unpacker's (--unpacker reference, the default). lut and compiled give the same bits (tested on every linear
at load), and on the same card beside a desktop (28 October 2026, layer 45, --max-new 64, the lens freed;
experiments/results/atlas_unpacker_steps_rtx5090.json) a step took 0.27-0.30 s with lut and 0.09-0.13 s with compiled
from 1 to 96 rows, against the reference's 0.71-0.74 s: at --batch 32, 0.28 s and 0.103 s, so compiled is about seven
times shorter, and its first run compiles for a few seconds. Memory: the model and lens took 21.2 GiB; each row costs
about 0.8 MB per word-piece of context on top (a batch of 32 at --max-new 64 peaked 3.1 GiB above the model), so at
--max-new 256 a batch of 32 needs some 7 GiB more and may not fit beside a desktop: lower --batch or --max-new.
Freeing the lens leaves about 3.2 GiB more for rows (worked out from its 59 matrices of 5,376 x 5,376 in bf16, not yet
measured). A calib_rel run of 8 neurons and 8 controls (580 answers) at --batch 32 --max-new 64 took about 16 minutes.
answers = (neurons + controls) x 10 + 4 for main_rel: a whole layer is (21,504 + 256) x 10 + 4 = 217,604 answers,
about 3.6 days at --batch 32 --max-new 64 and about two weeks at --max-new 256 (with --unpacker compiled about 12.7
hours at --batch 32, and about 5.2 at --batch 96, which fit beside a desktop at 7.2 GiB above the model with the lens
freed: the arithmetic above on the measured step, not a run). break_rel is (neurons + controls) x 24 + 6 answers, taken at --batch 64 (the trial's step-0 rule: at 96 the chair
question's unpushed answer parts from itself written alone after 20 word-pieces, under the rule's 32; at 64 after 46): a
whole layer with 1,024 controls is 540,678 answers in 8,454 batches of 64, about 12.1 hours at --max-new 48 at the
trial's measured 12.4 answers a second while writing (29 October 2026, layer 45, compiled) and about 8.0-8.4 at 32
(estimated, not timed); with 2,048 controls 565,254 answers in 8,838 batches, about 12.7 hours at 48, more than one
night: stop it in the morning and the same command continues it row for row; the reference's depth curve used 384
neurons and 64 controls a layer (4,484 answers, about 2 hours at --batch 32 --max-new 64). The batch and the plan now
fix a run's determinism class (on the GPU a row's numbers depend on the size of its batch), the unpacker does not:
measure what the card holds once, then keep that --batch for every continuation and every shard of the run. On disk a
whole layer is about 5 to 6 GB (each answer keeps its five likeliest pieces at every step, some 25 kB an answer at 256
pieces); ranking it reads one answer at a time and keeps only each answer's word ids, well under a gigabyte. Run one
GPU process at a time: this loads the model, as the server and the lab daemon do.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from pathlib import Path
from typing import List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def free_lens(rt) -> Optional[str]:
    """Free the J++ lens from the card once the runtime is loaded (on the 31B, 59 matrices of 5,376 x 5,376 in bf16,
    about 3.4 GB), and return its id, read first, which each run's meta keeps. The census reads none of it: take_census
    pushes and writes through the model's own layers (atlas_ops.write_rows), Weights reads the down columns and norms
    through the runtime's accessors, p50 comes from the stats index or a forward over the study's texts, and the ranking
    is CPU work on the answers."""
    import gc

    import torch

    lens_id = (rt.lens_card() or {}).get("id")
    rt.jpp_lenses.clear()  # the same dict TheodoliteRuntime.load gave interp-engine's lens stream
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return lens_id


def parse_layers(spec: str) -> List[int]:
    """"45" or "21,31,41-46": layers counting from 0, in the order given."""
    out: List[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        if not a.isdigit() or (b and not b.isdigit()):
            raise SystemExit(f'--layers takes layers counting from 0, as "45" or "21,31,41-46", not {spec!r}.')
        out.extend(range(int(a), int(b) + 1) if b else [int(a)])
    return list(dict.fromkeys(out))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--layer", type=int, help="the layer to take, counting from 0")
    where.add_argument("--layers", default="", help='several layers in one process, each its own run: "21,31,41-46" (the depth curve)')
    ap.add_argument("--neurons", default=None, help='"all" (the default), "none" (controls alone, for census_tools.py join), '
                                                    '"random:384", "2742", "0-99" or "1,5,9-12"')
    ap.add_argument("--neurons-from", default="", help="a screen's ranking (<run>.units.jsonl): retest its --top strongest neurons "
                                                       "against fresh controls (another --seed), the census's second stage")
    ap.add_argument("--top", type=int, default=0, help="with --neurons-from: how many of the screen's strongest to retest")
    from backend import census as cz_designs  # the designs' names and units (numpy only: torch is imported below)

    ap.add_argument("--design", default=cz_designs.DEFAULT_DESIGN, choices=list(cz_designs.DESIGNS),
                    help=f"{' | '.join(cz_designs.DESIGNS)} (default {cz_designs.DEFAULT_DESIGN})")
    ap.add_argument("--dose-unit", default=None, choices=list(cz_designs.DOSE_UNITS),
                    help="stream: dose x p50, the same push for every unit; ceiling: dose x each neuron's write ceiling "
                         "(default: the design's own, ceiling for calib_ceil, else stream; break_rel only in the stream)")
    ap.add_argument("--random-controls", type=int, default=None, help="random-direction controls spread through the run "
                    "(default: the design's, census.run_defaults: 2,048 for break_rel, else 256)")
    ap.add_argument("--seed", type=int, default=0, help="draws the controls' directions and random:K's neurons")
    ap.add_argument("--control-kind", default=cz_designs.CONTROL_KIND, choices=list(cz_designs.CONTROL_KINDS),
                    help="isotropic: unit(w_post * r), r isotropic (the default); columns: a random mix of the layer's own "
                         "write directions (census.CONTROL_KINDS)")
    ap.add_argument("--max-new", type=int, default=None, help="word-pieces per answer at most (pushed answers often loop; "
                    "default: the design's, 48 for break_rel, as its doses were calibrated, else 256)")
    ap.add_argument("--top-k", type=int, default=5, help="the likeliest pieces kept at each step")
    ap.add_argument("--batch", type=int, default=None, help="answers written together (fixed for the run: its rows' bits "
                    "depend on it; default: the design's, 64 for break_rel, its trial's, else 32)")
    ap.add_argument("--shard", default="", help="i/n: take shard i of n of the run's planned chunks, into <run>.s<i>of<n>.jsonl "
                                                "(join them with census_tools.py merge, then rank)")
    ap.add_argument("--canary-only", action="store_true", help="write only the run's plan batch 0, to <run>.canary.jsonl (its "
                    "settings and runtime beside it, <run>.canary.json), as a shard's canary, and stop: for comparing one "
                    "card with another before a run (census_tools.py canary)")
    ap.add_argument("--run", default="", help="the run's name (default L<layer>-<design>); with several layers it names each "
                                              "by {layer}, e.g. L{layer}-r384")
    ap.add_argument("--out-dir", default="", help="the census folder (default: the atlas's)")
    ap.add_argument("--no-rank", action="store_true", help="leave the ranking to the server or census_tools.py rank (a run is "
                                                          "otherwise ranked as its design is read, census.reading_of)")
    ap.add_argument("--stand-in", action="store_true", help="run on the CPU stand-in (scripts/atlas_stand_in.py), for a dry run")
    ap.add_argument("--unpacker", default="reference", choices=["reference", "lut", "compiled", "resident"],
                    help="how the 4-bit weights are unpacked (backend/w4a16_fast.py): each gives the reference's bits "
                    "(tested at load on a GPU); lut, compiled and resident are faster, resident needing ~60 GB more")
    a = ap.parse_args(argv)
    for k_, v_ in cz_designs.run_defaults(a.design).items():  # the design's own settings where none are given
        if getattr(a, k_) is None:
            setattr(a, k_, v_)
    shard = None
    if a.shard:
        i, _, n = a.shard.partition("/")
        if not (i.isdigit() and n.isdigit() and 0 <= int(i) < int(n)):
            raise SystemExit(f"--shard is i/n, shard i of n counting from 0 (0/4 to 3/4), not {a.shard!r}.")
        shard = (int(i), int(n))
    layers = [a.layer] if a.layers == "" else parse_layers(a.layers)
    if not layers:
        raise SystemExit("--layers names no layer.")
    if len(layers) > 1 and a.run and "{layer}" not in a.run:
        raise SystemExit("With several layers each is its own run: name them by {layer} in --run (e.g. --run L{layer}-r384), "
                         "or leave --run out for L<layer>-<design>.")
    if a.neurons_from and (a.neurons is not None or len(layers) > 1 or a.top < 1):
        raise SystemExit("--neurons-from retests one layer's screen: give it --top M and one --layer, and no --neurons.")
    if a.top and not a.neurons_from:
        raise SystemExit("--top goes with --neurons-from, the screen whose strongest neurons it retests.")
    if a.canary_only and shard:
        raise SystemExit("A canary is the plan's first batch whatever the shard: give --canary-only without --shard.")

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
        from backend.runtime_factory import make_runtime  # THEODOLITE_GGUF / THEODOLITE_DEVICE, else the W4A16 checkpoint

        rt = make_runtime(load_jlens=False).load()
    if getattr(rt, "jpp_lenses", None):
        lens_id = free_lens(rt)
        print(f"the J++ lens ({lens_id or 'unnamed'}) freed from the card: the census reads none of it", flush=True)
    else:  # THEODOLITE_GGUF_LENS=none, or a runtime that loads none: the meta names what the runtime's card says
        lens_id = None
        print("no J++ lens loaded: the census reads none", flush=True)
    from backend import w4a16_fast

    try:
        w4a16_fast.install(rt, a.unpacker, log=lambda s: print(s, flush=True))
    except w4a16_fast.Refused as exc:
        raise SystemExit(str(exc))
    lm = language_model(rt.hf_model)
    L, N = len(lm.layers), ao.ffn_width(lm, 0)
    bad = [l for l in layers if not 0 <= l < L]
    if bad:
        raise SystemExit(f"Layer {bad[0]} does not exist: layers count from 0 to {L - 1} (page layers 1 to {L}).")
    stage = None
    try:
        if a.neurons_from:
            src = Path(a.neurons_from)
            if not src.exists():  # a screen's ranking named within the census folder
                src = (Path(a.out_dir) if a.out_dir else cz.census_dir()) / a.neurons_from
            neurons, stage = cz.retest_of(src, a.top)
            if stage["screen_layer"] is not None and stage["screen_layer"] != layers[0]:
                raise SystemExit(f"{stage['screen']} screened layer {stage['screen_layer']}, not layer {layers[0]}.")
            if stage["screen_seed"] == a.seed:
                raise SystemExit(f"A retest draws fresh controls: give it another --seed than its screen's ({a.seed}).")
            spec = cz.neuron_ranges(neurons)
        else:
            spec = a.neurons or "all"
            neurons = cz.parse_neurons(spec, N, a.seed)
    except ValueError as exc:
        raise SystemExit(str(exc))

    from backend.runtime_factory import weights_card

    index = AtlasIndex(model=getattr(rt, "model_id", None), shape=(L, N), weights=weights_card(rt))
    stats = index.get("act_scale")
    p50_file = None
    if stats is not None:
        p50s = [float(x) for x in stats["resid_q"][:, 1]]
        source = f"the stats index ({stats['meta'].get('corpus')}, {int(stats['tokens'])} word-pieces)"
        p50_file = index.path("act_scale")
    else:
        from backend.mcp_scientist import _study_ids

        p50s = ao.stream_p50(ao.activation_extremes(lm, _study_ids(rt)))
        refused = index.refused("act_scale")  # there, but built for another model id, shape or weights
        if refused:
            print(f"the stats index is not read: {refused}", flush=True)
        source = f"the study's 36 texts ({'the stats index refused: ' + refused if refused else 'no stats index yet'})"

    out_dir = Path(a.out_dir) if a.out_dir else cz.census_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def on_signal(_sig, _frame):  # stop between steps: the batch under way is dropped, every whole one kept
        print("stopping after this step (the same command continues the run)", flush=True)
        stop.set()

    signal.signal(signal.SIGINT, on_signal)
    for layer in layers:
        name = (f"L{layer}-{a.design}" + (f"-top{a.top}" if stage else "") + ("-columns" if a.control_kind == "columns" else "")
                + ("-controls" if not neurons else ""))
        run_path = out_dir / f"{a.run.replace('{layer}', str(layer)) if a.run else name}.jsonl"
        out = cz.shard_path(run_path, *shard) if shard else run_path
        try:
            res = cz.take_census(rt, layer, neurons, out, p50=float(p50s[layer]), p50_source=source, design_name=a.design,
                                 controls=a.random_controls, seed=a.seed, max_new=a.max_new, top_k=a.top_k, batch=a.batch,
                                 neurons_spec=spec, dose_unit=a.dose_unit or "", shard=shard, p50_file=p50_file, lens_id=lens_id,
                                 stage=stage, canary_only=a.canary_only, control_kind=a.control_kind, cancel=stop.is_set,
                                 log=lambda s: print(s, flush=True))
        except ValueError as exc:
            raise SystemExit(str(exc))
        except RuntimeError as exc:
            from backend.cpu_memory import is_cpu_oom, plain

            if not is_cpu_oom(exc):
                raise
            raise SystemExit(f"{plain(exc, 'the census')} Every whole batch before it is kept, and the same command "
                             "continues the run; a run's batch is fixed, so a smaller --batch or --max-new is another "
                             "run (--run).")
        if a.canary_only:
            print(f"-> {res['canary']} ({res['written']} answers written, {res['skipped']} were there): compare it with the same "
                  "batch from another card by census_tools.py canary")
        else:
            print(f"-> {out} ({res['written']} answers written, {res['skipped']} were there)")
        if shard:
            print(f"a shard is ranked only once its run is joined: census_tools.py merge {run_path.stem}.s*of{shard[1]}.jsonl --out "
                  f"{run_path.name} (in {out_dir}; a canary named among them is left out of the join), then rank")
        elif not neurons and not a.canary_only:
            print(f"a run of controls alone is not ranked: census_tools.py join <a layer {layer} run> {out.name} --out <name>, "
                  "then rank the joined run")
        elif not a.no_rank and not a.canary_only and out.exists():
            units = cz.write_units(out)
            rd = cz.run_reading(out)
            against = "the controls' mean" if rd["reference"] == "controls" else "every unit's mean"
            left_out = "each answer's own question's" if rd["exclude"] == "own" else "every question's"
            print(f"-> {units} (ranked as {a.design} is read: pushes of at most {rd['max_rel']:g} of the stream, each word "
                  f"against {against}, {left_out} unpushed words left out)")
        if res["stopped"]:
            left = layers[layers.index(layer) + 1 :]
            if left:
                print(f"layers {', '.join(map(str, left))} not begun: the same command continues", flush=True)
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())

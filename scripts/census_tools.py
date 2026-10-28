#!/usr/bin/env python3
"""Work over finished census runs (backend/census.py), on the CPU: rank one, join shards, the depth report across
layers, the spread of a run's answers, and a calibration run's dose window.

    .venv/bin/python -I scripts/census_tools.py rank 45 --show 40 --find 2742     # a layer's main run, or a run's path
    .venv/bin/python -I scripts/census_tools.py merge L45-s0.jsonl L45-s1.jsonl --out L45-main_rel.jsonl
    .venv/bin/python -I scripts/census_tools.py depth --out depth_report.json
    .venv/bin/python -I scripts/census_tools.py spread 45                          # writes <run>.spread.json
    .venv/bin/python -I scripts/census_tools.py curves 45 --weights --json L45-curves.json   # a layer's calibration run

rank writes `<run>.units.jsonl` (each unit's words, z, mass, null_pct, shift, per-dose coherence) and prints the top
of it and the controls' null (sweep/rank.py). merge keeps one row per (kind, neuron, prompt, dose) and refuses runs
taken with other settings (sweep/merge_runs.py); shards of one run (census_run.py --shard i/n) are joined only whole:
every shard once, every answer of the run's job list (else refused, unless --partial, which the merged meta records),
each shard's canary compared with shard 0's answers, and the runtime classes they were written on recorded (a
warning, never a refusal). Rank a sharded run after the merge. depth reads every layer's main run, subsampled to the
same pool (--neurons, --controls) so the word-cluster mass compares across layers (sweep/depth_report.py). spread
embeds each answer (sentence-transformers' BAAI/bge-small-en-v1.5 where installed, as the reference's
build_embeddings.py, else a download-free word embedding) and lays the answers out in two dimensions (UMAP where
umap-learn is installed, else principal components): the atlas page's spread view. curves lays out a run's dose
window (sweep/peek.py --curves): per dose, neurons against controls, the push over the stream and over each neuron's
write ceiling, the share of answers moved from that question's unpushed answer, where they first part (the median),
the share cut off, distinct 4-grams and the mean log-probability, then a hedged verdict per sign: where the neurons
start to move more answers than the controls, and where text breaks. The ceilings come from the answers' own record
where they keep one, else, with --weights, from the weights the run was taken on (the checkpoint in the Hugging Face
cache, a few tensors of one layer read on the CPU; the stand-in's with --stand-in). A run is named by its path, or by
a layer number (counting from 0), for that layer's main run in the census folder (for curves, its calibration run
first). Add --stand-in to work in the stand-in's folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def resolve_run(spec: str, calibration: bool = False) -> Path:
    from backend import census as cz

    if spec.isdigit():
        runs = {**cz.layer_runs(), **(cz.layer_runs(calibration=True) if calibration else {})}
        if int(spec) not in runs:
            raise SystemExit(f"No census of layer {spec} in {cz.census_dir()} (layers with one: {sorted(runs) or 'none'}).")
        return runs[int(spec)]
    path = Path(spec)
    for alt in (cz.census_dir() / spec, Path(spec + ".jsonl"), cz.census_dir() / (spec + ".jsonl")):
        if not path.exists() and alt.exists():
            path = alt
    if not path.exists():
        raise SystemExit(f"There is no run {spec}.")
    return path


def run_ceilings(run: Path, meta: dict, stand_in: bool) -> dict:
    """{(kind, neuron): write ceiling} for every unit of a run, from the weights it was taken on: the stand-in's
    (rebuilt here from its seed) or the checkpoint's in the Hugging Face cache (backend/census.checkpoint_ceilings, on
    the CPU). A control is set against the neuron it is paired with (census.control_refs)."""
    from backend import census as cz

    layer, seed = int(meta["layer"]), int(meta.get("seed", 0))
    units = {(r["kind"], int(r["neuron"])) for r in cz.read_rows(run, tolerate_tail=True) if r["kind"] != "baseline"}
    neurons = sorted(n for k, n in units if k == "neuron")
    controls = sorted(n for k, n in units if k == "random")
    if meta.get("stand_in"):
        if not stand_in:
            raise SystemExit(f"{run.name} was taken on the stand-in: add --stand-in to read its weights.")
        import torch

        from atlas_stand_in import stand_in as build
        from backend import atlas_ops as ao
        from backend.causal_check import language_model

        torch.set_grad_enabled(False)
        w = ao.Weights(language_model(build()[0].hf_model))
        refs = cz.control_refs(seed, max(controls) + 1, w.d_ff) if controls else []
        want = sorted(set(neurons) | {refs[k] for k in controls})
        got = dict(zip(want, w.write_ceiling(layer, want).tolist())) if want else {}
    else:
        model = meta.get("model") or "google/gemma-4-31B-it-qat-w4a16-ct"
        files = cz.cached_checkpoint(model)
        if not files:
            raise SystemExit(f"{model} is not in the Hugging Face cache here, so its ceilings cannot be read: leave out --weights.")
        got, width = cz.checkpoint_ceilings(files, layer, neurons or [0])
        refs = cz.control_refs(seed, max(controls) + 1, width) if controls else []
        if controls:
            got.update(cz.checkpoint_ceilings(files, layer, sorted({refs[k] for k in controls}))[0])
    out = {("neuron", n): got[n] for n in neurons}
    out.update({("random", k): got[refs[k]] for k in controls})
    return out


def print_curves(run: Path, meta: dict, curves: list, window: dict, source: str) -> None:
    unit = meta.get("dose_unit", "stream")
    what = "as multiples of each neuron's write ceiling" if unit == "ceiling" else "as shares of the stream's median length"
    p50 = meta.get("p50")
    print(f"{run.name}: layer {meta.get('layer')}, {meta.get('design_name')}, doses {what}"
          + (f" (p50 {float(p50):.2f}, {meta.get('p50_source')})" if p50 is not None else ""))
    base = next((c for c in curves if c["kind"] == "baseline"), None)
    if base is not None:
        print(f"unpushed: {base['n']} answers, cut off {base['capped']:.0%}, distinct-4 {base['distinct4']:.2f}, log-p {base['logprob']:.2f}")
    print(f"x ceil: the push over each neuron's write ceiling ({source or 'not at hand: add --weights'})")
    kinds = [k for k in ("neuron", "random") if any(c["kind"] == k for c in curves)]
    head = f"{'n':>4} {'rel':>5} {'x ceil':>6} {'moved':>6} {'depart':>6} {'cut':>5} {'d4':>5} {'lp':>6}"
    print(f"{'':>7}  " + "  ".join(f"{('neurons' if k == 'neuron' else 'controls'):^{len(head)}}" for k in kinds))
    print(f"{'dose':>7}  " + "  ".join(head for _ in kinds))
    cell = {(c["dose"], c["kind"]): c for c in curves}

    def fmt(v, f):
        return "-" if v is None else format(v, f)

    for dose in sorted({c["dose"] for c in curves if c["kind"] != "baseline"}):
        cells = []
        for k in kinds:
            c = cell.get((dose, k))
            cells.append(" " * len(head) if c is None else
                         f"{c['n']:4d} {fmt(c['rel_push'], '5.2f'):>5} {fmt(c['over_ceiling'], '6.2f'):>6} {fmt(c['moved'], '6.0%'):>6} "
                         f"{fmt(c['departure'], '6.0f'):>6} {fmt(c['capped'], '5.0%'):>5} {fmt(c['distinct4'], '5.2f'):>5} "
                         f"{fmt(c['logprob'], '6.2f'):>6}")
        print(f"{dose:+7g}  " + "  ".join(cells))
    for sign in ("+", "-"):
        if sign in window:
            print(window[sign]["says"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--stand-in", action="store_true", help="the stand-in's census folder")
    sub = ap.add_subparsers(dest="what", required=True)
    r = sub.add_parser("rank", help="rank a run's units")
    r.add_argument("run")
    r.add_argument("--max-rel", type=float, default=1.6, help="answers pushed by at most this share of the stream count")
    r.add_argument("--top-words", type=int, default=12)
    r.add_argument("--show", type=int, default=30)
    r.add_argument("--find", default="", help="neurons whose rank to report, e.g. 2742,15")
    m = sub.add_parser("merge", help="join shards of one layer's run")
    m.add_argument("inputs", nargs="+")
    m.add_argument("--out", required=True)
    m.add_argument("--partial", action="store_true", help="join shards though some, or some of their answers, are missing "
                   "(the merged meta says so)")
    d = sub.add_parser("depth", help="the depth report across layers")
    d.add_argument("--neurons", type=int, default=384)
    d.add_argument("--controls", type=int, default=64)
    d.add_argument("--out", default="", help="where to write it (default: depth_report.json in the census folder)")
    s = sub.add_parser("spread", help="lay a run's answers out by what they say")
    s.add_argument("run")
    s.add_argument("--embedder", default="auto", help='"auto", "words" or "st:<sentence-transformers model>"')
    s.add_argument("--chars", type=int, default=600, help="leading characters of each answer to embed")
    s.add_argument("--neighbors", type=int, default=15)
    s.add_argument("--layout", default="auto", choices=["auto", "umap", "pca"])
    c = sub.add_parser("curves", help="a run's dose window: neurons against controls, dose by dose")
    c.add_argument("run")
    c.add_argument("--weights", action="store_true", help="each unit's write ceiling from the weights the run was taken on, "
                   "for runs whose answers do not keep it")
    c.add_argument("--json", default="", help="also write the table and the verdict there")
    a = ap.parse_args(argv)

    if a.stand_in:
        from atlas_stand_in import use_stand_in_atlas

        use_stand_in_atlas()
    import numpy as np

    from backend import census as cz

    try:
        if a.what == "rank":
            run = resolve_run(a.run)
            out = cz.write_units(run, max_rel=a.max_rel, top_words=a.top_words)
            recs = list(cz.read_rows(out))
            st = json.loads(cz.meta_path_of(run).read_text(encoding="utf-8")).get("stage") if cz.meta_path_of(run).exists() else None
            if st:
                print(f"a retest (stage {st['stage']}) of the {st['top']} strongest of the {st['of']} neurons of {st['screen']}, "
                      f"against fresh controls: its p and q are taken over these {st['top']} alone")
            null = np.sort([u["mass"] for u in recs if u["kind"] == "random"])
            neurons = [u for u in recs if u["kind"] == "neuron"]
            if len(null):
                above = sum(u["mass"] > null.max() for u in neurons)
                print(f"{len(neurons)} neurons, {len(null)} random controls; the controls' mass p50/p95/max = "
                      f"{np.percentile(null, 50):.2f} / {np.percentile(null, 95):.2f} / {null.max():.2f}")
                print(f"neurons above every control: {above} ({above / max(len(neurons), 1):.1%}); about "
                      f"{cz.chance_above_all(len(neurons), len(null))} would be by chance alone; "
                      f"{sum(1 for u in neurons if u.get('q') is not None and u['q'] <= 0.05)} with a false-discovery q <= 0.05")
            else:
                print(f"{len(neurons)} neurons, no random controls (null_pct is None)")
            for i, u in enumerate(recs[: a.show]):
                ws = ", ".join(f"{w}({c})" for w, c, _ in u["words"][:6])
                print(f"{i + 1:4d}  {u['kind'][:3]} {u['neuron']:5d}  mass {u['mass']:5.1f}  shift {u['shift'] if u['shift'] is not None else '-':>6}  {ws}")
            for n in [int(x) for x in a.find.split(",") if x.strip()]:
                rank = next((i + 1 for i, u in enumerate(neurons) if u["neuron"] == n), None)
                rec = next((u for u in neurons if u["neuron"] == n), None)
                if rec is None:
                    print(f"neuron {n}: not in this run")
                    continue
                ws = ", ".join(f"{w}({c})" for w, c, _ in rec["words"][:6])
                print(f"neuron {n}: rank {rank}/{len(neurons)} among neurons, mass {rec['mass']:.1f}, null_pct {rec['null_pct']}, "
                      f"p {rec.get('p')}, q {rec.get('q')}, {ws}")
            print(f"-> {out}")
        elif a.what == "merge":
            n_in, n_out = cz.merge_runs([Path(x) for x in a.inputs], Path(a.out), partial=a.partial)
            meta = json.loads(cz.meta_path_of(Path(a.out)).read_text(encoding="utf-8"))
            if meta.get("canary"):
                print(f"canary: {meta['canary']}")
            if meta.get("partial"):
                miss = meta["partial"]
                print(f"partial: shards missing {miss['missing_shards'] or 'none'}, {miss['missing_answers']} answers missing")
            if meta.get("mixed_runtime"):
                print(f"warning: the parts were written on {len(meta['mixed_runtime'])} runtime classes (mixed_runtime): "
                      "each one's readings are valid, their bits are not alike")
            print(f"{n_in} rows in, {n_out} kept -> {a.out}")
        elif a.what == "depth":
            report = cz.depth_report(cz.layer_runs(), neurons=a.neurons, controls=a.controls)
            for x in report:
                if "note" in x:
                    print(f"layer {x['layer']:<3} {x['note']} ({x['n_neurons']} neurons, {x['n_controls']} controls)")
                    continue
                print(f"layer {x['layer']:<3}{'*' if x['subsampled'] else ' '} controls p50/p95 {x['null_p50']:5.1f}/{x['null_p95']:5.1f}  "
                      f"neurons p50/p90 {x['neuron_p50']:5.1f}/{x['neuron_p90']:5.1f}  >p95 {x['above_p95']:.0%}  >max {x['above_max']:.0%}  | "
                      + " ; ".join(f"{t['n']}:{' '.join(t['words'][:4])}" for t in x["top"][:3]))
            out = Path(a.out) if a.out else cz.census_dir() / "depth_report.json"
            cz.write_json_atomic(out, report)
            print(f"-> {out}")
        elif a.what == "spread":
            run = resolve_run(a.run)
            out = cz.build_spread(run, embedder=a.embedder, chars=a.chars, neighbors=a.neighbors, layout=a.layout)
            d = json.loads(out.read_text(encoding="utf-8"))
            print(f"{len(d['row'])} answers, embedded by {d['embedder']}, laid out by {d['layout']} -> {out}")
        elif a.what == "curves":
            run = resolve_run(a.run, calibration=True)
            meta = json.loads(cz.meta_path_of(run).read_text(encoding="utf-8"))
            ceilings = run_ceilings(run, meta, a.stand_in) if a.weights else None
            curves = cz.dose_curves(cz.read_rows(run, tolerate_tail=True), ceilings,
                                    base=cz.baseline_ids(cz.read_rows(run, tolerate_tail=True)))
            has = any(x["over_ceiling"] is not None for x in curves)
            source = ("the answers' own record, else the weights" if a.weights else "the answers' own record") if has else ""
            questions = [p for p in cz.PROMPTS if p in {q for q, _ in meta.get("design") or []}]
            window = cz.dose_window(curves, meta.get("dose_unit", "stream"), questions)
            print_curves(run, meta, curves, window, source)
            if a.json:
                cz.write_json_atomic(Path(a.json), {"run": run.name, "layer": meta.get("layer"), "design": meta.get("design_name"),
                                                    "dose_unit": meta.get("dose_unit", "stream"), "ceilings_from": source or None,
                                                    "curves": curves, "window": window})
                print(f"-> {a.json}")
    except ValueError as exc:
        raise SystemExit(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())

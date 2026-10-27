#!/usr/bin/env python3
"""Work over finished census runs (backend/census.py), on the CPU: rank one, join shards, the depth report across
layers, and the spread of a run's answers.

    .venv/bin/python -I scripts/census_tools.py rank 45 --show 40 --find 2742     # a layer's main run, or a run's path
    .venv/bin/python -I scripts/census_tools.py merge L45-s0.jsonl L45-s1.jsonl --out L45-main_rel.jsonl
    .venv/bin/python -I scripts/census_tools.py depth --out depth_report.json
    .venv/bin/python -I scripts/census_tools.py spread 45                          # writes <run>.spread.json

rank writes `<run>.units.jsonl` (each unit's words, z, mass, null_pct, shift, per-dose coherence) and prints the top
of it and the controls' null (sweep/rank.py). merge keeps one row per (kind, neuron, prompt, dose) and refuses runs
taken with other settings (sweep/merge_runs.py). depth reads every layer's main run, subsampled to the same pool
(--neurons, --controls) so the word-cluster mass compares across layers (sweep/depth_report.py). spread embeds each
answer (sentence-transformers' BAAI/bge-small-en-v1.5 where installed, as the reference's build_embeddings.py, else a
download-free word embedding) and lays the answers out in two dimensions (UMAP where umap-learn is installed, else
principal components): the atlas page's spread view. A run is named by its path, or by a layer number (counting from
0), for that layer's main run in the census folder. Add --stand-in to work in the stand-in's folder.
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


def resolve_run(spec: str) -> Path:
    from backend import census as cz

    if spec.isdigit():
        runs = cz.layer_runs()
        if int(spec) not in runs:
            raise SystemExit(f"No census of layer {spec} in {cz.census_dir()} (layers with one: {sorted(runs) or 'none'}).")
        return runs[int(spec)]
    path = Path(spec)
    if not path.exists() and (cz.census_dir() / spec).exists():
        path = cz.census_dir() / spec
    if not path.exists():
        raise SystemExit(f"There is no run {spec}.")
    return path


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
            null = np.sort([u["mass"] for u in recs if u["kind"] == "random"])
            neurons = [u for u in recs if u["kind"] == "neuron"]
            if len(null):
                above = sum(u["mass"] > null.max() for u in neurons)
                print(f"{len(neurons)} neurons, {len(null)} random controls; the controls' mass p50/p95/max = "
                      f"{np.percentile(null, 50):.2f} / {np.percentile(null, 95):.2f} / {null.max():.2f}")
                print(f"neurons above every control: {above} ({above / max(len(neurons), 1):.1%})")
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
                print(f"neuron {n}: rank {rank}/{len(neurons)} among neurons, mass {rec['mass']:.1f}, null_pct {rec['null_pct']}, {ws}")
            print(f"-> {out}")
        elif a.what == "merge":
            n_in, n_out = cz.merge_runs([Path(x) for x in a.inputs], Path(a.out))
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
    except ValueError as exc:
        raise SystemExit(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())

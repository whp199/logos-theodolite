#!/usr/bin/env python3
"""How far torch's int4 CPU kernel (backend/int4_cpu.py: THEODOLITE_GGUF_LINEARS=int4mm, or
THEODOLITE_CHECKPOINT_LINEARS=int4mm) departs from the reference path, and how much faster it is, on this machine.

    .venv/bin/python -I scripts/compare_int4mm.py --time                 # a 5,376 x 21,504 layer each way, 1, 8, 64 rows
    THEODOLITE_GGUF=model.gguf THEODOLITE_DEVICE=cpu THEODOLITE_GGUF_LINEARS=int4mm THEODOLITE_GGUF_LENS=none \\
        .venv/bin/python -I scripts/compare_int4mm.py --layer 45 --max-new 16 --json int4mm.json
    .venv/bin/python -I scripts/compare_int4mm.py --stand-in             # the same on the stand-in written as a Q4_0 GGUF

--time: one random Q4_0 layer of 5,376 x 21,504 and one of 21,504 x 5,376 (the 31B's feed-forward shapes), at 1, 8 and
64 rows in bf16 on this machine's threads: the best of --reps wall times of the reference (the packed layer's forward:
unpack, then F.linear), the resident bf16 product and the kernel (`int4_cpu.timed_layer`); about 3 GB of memory.

The comparison, after neuron-explorer's sweep/check_equivalence.py, judged on what a census reads
(scripts/check_gguf_cpu.py §4 runs it on the stand-in): a canary census (backend/census.py's questions and main_rel's
doses, the neurons --neurons of --layer beside --random-controls controls, by default the 16 and 8 of the canary in
the equivalence design this follows, answers of at most --max-new word-pieces, the stream's p50 read once over the
study's 36 texts on the reference path), taken on one runtime three times: on the reference path at --batch and at
--floor-batch (the instrument's own floor: how far two batch sizes of the reference already differ here), then
through the kernel at --batch (`int4_cpu.enable` turns it off and on in place, and the model id's +int4mm with it).
For each pair (`int4_cpu.compare_rows`): the answers identical, the shared prefix as a share of the reference's
answer, the steps where they split, the largest |delta log-prob| over the shared prefix, the overlap of the five
likeliest: the design's per-answer measures. Its per-run half (both rankings of a whole depth pool, 4,484 answers:
Spearman's rho of mass, the top 100's overlap, the neurons above every control, the false-discovery counts) is not
taken here, so this rule can only refuse. The rule, fixed here before any run on the 31B: the kernel's readings are
their own class, as their model id already makes them, if they depart from the reference further than the floor on
any of the four measures; within it on all four they still do not stand for the reference's until the ranking's half
has been taken and its leads are unchanged. Exits 0 either way; the verdict is printed and in --json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def quiet(_s: str) -> None:
    pass


def canary(rt: Any, out_dir: Path, tag: str, layer: int, neurons: List[int], controls: int, max_new: int, batch: int,
           p50: float) -> List[Dict[str, Any]]:
    """A canary census on `rt` as it stands, into a file of its own (`tag`), its rows."""
    from backend import census as cz

    out = Path(out_dir) / f"canary-{tag}.jsonl"
    cz.take_census(rt, layer, neurons, out, p50=float(p50), p50_source="fixed (scripts/compare_int4mm.py)", controls=controls,
                   design_name="main_rel", seed=0, max_new=max_new, top_k=5, batch=batch, log=quiet)  # as the rule was fixed
    return list(cz.read_rows(out))


def verdict(floor: Dict[str, Any], kernel: Dict[str, Any]) -> Dict[str, Any]:
    """The rule's per-answer half: whether the kernel's departure from the reference is no larger than the floor's on
    each of the four measures (`within_floor`, and `per_answer_within_floor` all four). `stands_for_reference` is never
    true here: the design's per-run half, the ranking's leads over a whole depth pool, is not taken (`ranking` None)."""
    share = lambda r: r["identical"] / max(1, r["answers"])  # noqa: E731
    each = {
        "identical": share(kernel) >= share(floor),
        "shared_share": kernel["shared_share"] >= floor["shared_share"],
        "max_abs_dlogprob": kernel["max_abs_dlogprob"] <= floor["max_abs_dlogprob"],
        "topk_overlap": (kernel["topk_overlap"] or 0.0) >= (floor["topk_overlap"] or 0.0),
    }
    return {"stands_for_reference": False, "per_answer_within_floor": all(each.values()), "within_floor": each,
            "ranking": None}


def compare(rt: Any, out_dir: Path, *, layer: int, neurons: List[int], controls: int = 8, max_new: int = 16,
            batch: int = 8, floor_batch: int = 3) -> Dict[str, Any]:
    """The three canaries and their comparisons (the module's docstring); `rt` holds int4mm layers, left on."""
    from backend import atlas_ops as ao
    from backend import int4_cpu
    from backend.causal_check import language_model
    from backend.mcp_scientist import _study_ids

    if not int4_cpu.int4_layers(rt.hf_model):
        raise ValueError("This runtime holds no int4mm layers: set THEODOLITE_GGUF_LINEARS=int4mm (a GGUF) or "
                         "THEODOLITE_CHECKPOINT_LINEARS=int4mm (the checkpoint), with THEODOLITE_DEVICE=cpu.")
    int4_cpu.enable(rt, False)
    try:
        ref_id = rt.model_id
        p50 = ao.stream_p50(ao.activation_extremes(language_model(rt.hf_model), _study_ids(rt)))[int(layer)]
        ref = canary(rt, out_dir, "reference", layer, neurons, controls, max_new, batch, p50)
        ref_floor = canary(rt, out_dir, "reference-floor", layer, neurons, controls, max_new, floor_batch, p50)
    finally:
        int4_cpu.enable(rt, True)
    new = canary(rt, out_dir, "int4mm", layer, neurons, controls, max_new, batch, p50)
    floor = int4_cpu.compare_rows(ref, ref_floor)
    kernel = int4_cpu.compare_rows(ref, new)
    return {"reference": ref_id, "int4mm": rt.model_id, "layer": int(layer), "neurons": list(neurons), "controls": int(controls),
            "max_new": int(max_new), "batch": int(batch), "floor_batch": int(floor_batch), "p50": float(p50),
            "floor": floor, "kernel": kernel, "verdict": verdict(floor, kernel)}


def say(name: str, r: Dict[str, Any]) -> str:
    return (f"{name}: {r['identical']} of {r['answers']} answers identical; shared prefix {r['shared_share']:.1%} of the "
            f"reference's; split at steps {r['split_steps'][:12]}{' ...' if len(r['split_steps']) > 12 else ''}; largest "
            f"|delta log-prob| over the shared prefix {r['max_abs_dlogprob']:.4g}; top-{r['k']} overlap {r['topk_overlap']}")


def stand_in_runtime(tmp: Path) -> Any:
    """The stand-in written as a Q4_0 GGUF, loaded with int4mm linears in bf16 on the CPU."""
    import atlas_stand_in as asi
    import torch

    from backend import gguf_model as gm
    from backend.gguf_runtime import GGUFRuntime

    tok = asi.stand_in_tokenizer()
    path = gm.write_gemma4_gguf(asi.stand_in_hf(tok), tmp / "stand-in-Q4_0.gguf", tokenizer=tok, qtype="Q4_0")
    return GGUFRuntime(path, device="cpu", dtype=torch.bfloat16, tokenizer=tok, linears="int4mm", load_jpp=False).load()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--time", action="store_true", help="time one layer of each feed-forward shape, and nothing else")
    ap.add_argument("--reps", type=int, default=5, help="--time: the best of this many")
    ap.add_argument("--stand-in", action="store_true", help="compare on the stand-in written as a Q4_0 GGUF (a dry run)")
    ap.add_argument("--layer", type=int, default=2, help="the canary's layer, counting from 0")
    ap.add_argument("--neurons", default="0-15", help='the canary\'s neurons: "0-15" (the design\'s 16), "5,9,12"')
    ap.add_argument("--random-controls", type=int, default=8, help="the canary's random controls (the design's 8)")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8, help="the batch of the reference and the kernel")
    ap.add_argument("--floor-batch", type=int, default=3, help="the reference's second batch, for the floor")
    ap.add_argument("--json", default="", help="write the whole report here")
    a = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch

    torch.set_grad_enabled(False)
    if a.time:
        from backend import int4_cpu

        why = int4_cpu.kernel_missing()
        if why:
            raise SystemExit(f"Not timed: {why}.")
        report: Dict[str, Any] = {"threads": torch.get_num_threads(), "torch": torch.__version__, "layers": {}}
        print(f"{torch.get_num_threads()} threads, torch {torch.__version__}, bf16, best of {a.reps}")
        for out_f, in_f in ((5376, 21504), (21504, 5376)):
            rows = int4_cpu.timed_layer(out_f, in_f, reps=a.reps)
            report["layers"][f"{out_f}x{in_f}"] = rows
            n = out_f * in_f
            for r in rows:
                print(f"  [{out_f:,} x {in_f:,}] {r['rows']:3d} rows: Q4_0 reference {r['reference_s'] * 1e3:7.1f} ms "
                      f"({r['reference_s'] / n * 1e9:5.2f} ns a weight), W4A16 reference {r['w4a16_reference_s'] * 1e3:7.1f} ms "
                      f"({r['w4a16_reference_s'] / n * 1e9:5.2f}); resident bf16 {r['resident_bf16_s'] * 1e3:6.1f} ms; "
                      f"int4mm {r['int4mm_s'] * 1e3:6.1f} ms (Q4_0), {r['w4a16_int4mm_s'] * 1e3:6.1f} ms (W4A16)")
        if a.json:
            Path(a.json).write_text(json.dumps(report, indent=1))
        return 0
    from backend import census as cz

    with tempfile.TemporaryDirectory(prefix="compare-int4mm-") as d:
        if a.stand_in:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            rt = stand_in_runtime(Path(d))
        else:
            from backend.runtime_factory import make_runtime

            rt = make_runtime(load_jlens=False).load()
        from backend import atlas_ops as ao
        from backend.causal_check import language_model

        N = ao.ffn_width(language_model(rt.hf_model), 0)
        try:
            rep = compare(rt, Path(d), layer=a.layer, neurons=cz.parse_neurons(a.neurons, N, 0), controls=a.random_controls,
                          max_new=a.max_new, batch=a.batch, floor_batch=a.floor_batch)
        except ValueError as exc:
            raise SystemExit(str(exc))
    print(f"reference {rep['reference']}, kernel {rep['int4mm']}; layer {rep['layer']}, {len(rep['neurons'])} neurons, "
          f"{rep['controls']} controls, at most {rep['max_new']} word-pieces")
    print("  " + say(f"floor (the reference at batch {rep['batch']} against batch {rep['floor_batch']})", rep["floor"]))
    print("  " + say(f"int4mm (against the reference, both at batch {rep['batch']})", rep["kernel"]))
    v = rep["verdict"]
    print(("the kernel's answers depart no further than the floor; the ranking's half of the rule (a whole depth pool) is "
           "not taken here, so they do not yet stand for the reference's"
           if v["per_answer_within_floor"] else
           f"the kernel's readings depart further than the floor ({', '.join(k for k, ok in v['within_floor'].items() if not ok)}): "
           "they are their own class, as their model id makes them"))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

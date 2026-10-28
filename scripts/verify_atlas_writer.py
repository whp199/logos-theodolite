#!/usr/bin/env python3
"""The atlas's writer's floor, measured on the real model (after neuron-explorer's sweep/check_equivalence.py, which set
its sweep's steered answers against how far an unsteered answer agreed across its two engines): with no push at all,
how far does an answer written in each of the batch shapes the atlas writes in agree with the same answer written
alone? On a GPU the rows of different batch shapes are not bit-identical (cuBLAS chooses its kernels by the shape; rows
of the 31B written in batches of other shapes were not, N1A), so an unpushed answer may part from itself after some
word-pieces. A pushed answer that parts from an unpushed one written in another shape no sooner than that may have
parted by the shape alone; the floor is the line below which "the push changed the answer" is not by itself evidence.

For each of the census's four questions (census.PROMPTS, a chat turn with thinking off), the unpushed answer, greedy, at
most --max-new word-pieces and ending at the census's end marks, is written:
- alone (one row, atlas_ops.write_rows), the answer every shape is set against, and alone again, which must be the
  same (else nothing here means anything, and the script fails);
- as the as-it-is row of /steer: three rows of the question, the first and the last pushed by +0.8 and -0.8 of the
  stream's median length along one neuron's write direction (--steer-neuron, else drawn from --seed), the middle row
  unpushed (backend/atlas.py, _steer_whole);
- inside a census-shaped batch of each --batches size (32 and 64), written by the census's own writer
  (census.write_census_batch): every row the question; row 0 unpushed, where a run's baseline sits in its first
  batch; the last row a neuron's push at dose 0 (its direction times 0); the rows between pushed by the run's controls
  at +0.8 (census.control_dirs, --seed);
- by the sweep's and the vault's writer: interp-engine's own loop, one row (TheodoliteRuntime.write_voice, the writing
  loop of capture_for_sweep without its recapture).
Then, per question and shape, how many word-pieces it holds in common with the answer written alone before the two
first part (census.first_departure; all of them where they never part), and the floor: the fewest seen with no push,
over every shape and question (`floor`), and over the census's batches alone, batch size by batch size (`census_floor`,
which the census card reads: backend/census.writer_floor). Layers count from 0 (page = layer + 1).

It writes experiments/results/atlas_writer_floor.json (--out), with the commit, the runtime (census.runtime_record: the
card, torch and its CUDA, transformers, interp-engine, compressed-tensors, the stats index's sha256) and the unpacker. A
floor holds for the card class it was measured on: give each card its own file (atlas_writer_floor_<card>.json, as the
census card reads every atlas_writer_floor*.json there and prefers the one of its run's runtime). A run on the stand-in
writes into the stand-in's own folder instead, never among the model's results.

Cost (critic Q9; arithmetic on the rate measured on one RTX 5090, not timed): about 6 writes x 4 questions x 65 steps x
0.71 s, some 18 minutes with the reference unpacker, plus loading; a faster unpacker (--unpacker lut, the same bits)
cuts the steps. Stop the :8765 server first: this loads the model itself.

Run: .venv/bin/python -I scripts/verify_atlas_writer.py [--layer 45] [--max-new 64] [--out ...]
     .venv/bin/python -I scripts/verify_atlas_writer.py --stand-in --max-new 16      # the CPU stand-in, a dry run
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

RESULTS = ROOT_DIR / "experiments" / "results"
STEER_DOSE = 0.8  # /steer's and the census's push, as a share of the stream's median length
CONTROL_DOSE = 0.8


def server_running(port: int = 8765) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def shared(ids: List[int], base: List[int]) -> int:
    """Word-pieces held in common with `base` before the two first part (all of them where they never do)."""
    from backend.census import first_departure

    d = first_departure(ids, base)
    return len(ids) if d is None else d


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--layer", type=int, default=None, help="the layer pushed in the /steer and census shapes, counting from 0 "
                                                             "(default 45; 3 on the stand-in)")
    ap.add_argument("--max-new", type=int, default=64, help="word-pieces per answer at most (the census's --max-new)")
    ap.add_argument("--batches", default="32,64", help="census batch sizes to measure (3 or more each)")
    ap.add_argument("--seed", type=int, default=0, help="draws the controls (as a census run's --seed) and the /steer neuron")
    ap.add_argument("--steer-neuron", type=int, default=None, help="the neuron /steer's outer rows push (default: drawn from --seed)")
    ap.add_argument("--unpacker", default="reference", choices=["reference", "lut", "compiled", "resident"],
                    help="how the 4-bit weights are unpacked (backend/w4a16_fast.py), as census_run.py takes it")
    ap.add_argument("--out", default="", help="where to write the floor (default experiments/results/atlas_writer_floor.json)")
    ap.add_argument("--stand-in", action="store_true", help="run on the CPU stand-in (scripts/atlas_stand_in.py), for a dry run")
    a = ap.parse_args(argv)
    batches = sorted({int(b) for b in a.batches.split(",") if b.strip()})
    if not batches or min(batches) < 3:
        raise SystemExit("--batches takes census batch sizes of 3 or more (a baseline row, a dose-0 row and a pushed one).")
    if a.stand_in:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["HF_HUB_OFFLINE"] = "1"
    elif server_running():
        print("Something answers on :8765: stop the Theodolite server first (this loads the model itself).")
        return 2
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import numpy as np
    import torch

    from backend import atlas_ops as ao
    from backend import census as cz
    from backend import w4a16_fast
    from backend.atlas_index import AtlasIndex
    from backend.causal_check import language_model
    from census_run import free_lens

    torch.set_grad_enabled(False)
    t0 = time.perf_counter()
    if a.stand_in:
        from atlas_stand_in import stand_in, use_stand_in_atlas

        folder = use_stand_in_atlas()
        rt, _vault = stand_in()
    else:
        from backend.runtime_factory import make_runtime

        folder = None
        rt = make_runtime(load_jlens=False).load()
    free_lens(rt)
    try:
        unpacker = w4a16_fast.install(rt, a.unpacker, log=lambda s: print(s, flush=True))
    except w4a16_fast.Refused as exc:
        raise SystemExit(str(exc))
    lm = language_model(rt.hf_model)
    w = ao.Weights(lm, rt)
    L, N, D = w.n_layers, w.d_ff, w.d_model
    layer = a.layer if a.layer is not None else (3 if a.stand_in else 45)
    if not 0 <= layer < L:
        raise SystemExit(f"Layer {layer} does not exist: layers count from 0 to {L - 1} (page layers 1 to {L}).")
    index = AtlasIndex(model=getattr(rt, "model_id", None), shape=(L, N))
    stats = index.get("act_scale")
    p50_file = None
    if stats is not None:
        p50 = float(stats["resid_q"][layer, 1])
        p50_source = f"the stats index ({stats['meta'].get('corpus')}, {int(stats['tokens'])} word-pieces)"
        p50_file = index.path("act_scale")
    else:
        from backend.mcp_scientist import _study_ids

        p50 = float(ao.stream_p50(ao.activation_extremes(lm, _study_ids(rt)))[layer])
        p50_source = "the study's 36 texts (no stats index yet)"
    rng = np.random.default_rng(a.seed)
    neuron = int(a.steer_neuron) if a.steer_neuron is not None else int(rng.integers(N))
    if not 0 <= neuron < N:
        raise SystemExit(f"Neurons count from 0 to {N - 1}; {neuron} does not exist.")
    stop = rt.end_of_turn_ids()
    tok = rt.tokenizer
    softcap = ao.softcap_of(rt.hf_model)
    w_post = w.norm(layer, "post_feedforward").cpu().numpy()
    ctrl = torch.from_numpy(cz.control_dirs(a.seed, max(batches) - 2, w_post)).to(w.device)
    u = w.write_dirs(layer, [neuron])[0]
    steer_iv = ao.resolve(w, [ao.Push(layer=layer, neuron=neuron, mode="add", dose=STEER_DOSE)], [], ao.THREE_VOICES,
                          p50=lambda _l: p50, act_scale=lambda _l, _n: 1.0)

    def alone(ids: List[int]) -> List[int]:
        return [st.token_id for st in ao.write_rows(lm, rt.lm_head_weight, [ids], None, max_new=a.max_new, stop_ids=stop,
                                                    bos_id=tok.bos_token_id, spare_bos=True, softcap=softcap)]

    def steer_row(ids: List[int]) -> List[int]:
        got: List[int] = []
        for st in ao.write_rows(lm, rt.lm_head_weight, [ids, ids, ids], steer_iv, max_new=a.max_new, stop_ids=stop,
                                bos_id=tok.bos_token_id, softcap=softcap):
            if st.row == 1:
                got.append(st.token_id)
        return got

    def census_rows(ids: List[int], B: int) -> Dict[str, List[int]]:
        delta = torch.zeros(B, D, dtype=torch.float32, device=w.device)
        delta[1 : B - 1] = ctrl[: B - 2] * (CONTROL_DOSE * p50)
        delta[B - 1] = u * (0.0 * p50)  # a neuron's push at dose 0, as the census computes one
        rows = cz.write_census_batch(rt, layer, ids, delta, max_new=a.max_new, stop_ids=stop, top_k=5)
        assert rows is not None
        return {f"census {B}, baseline (row 0)": rows[0]["token_ids"], f"census {B}, dose 0 (row {B - 1})": rows[B - 1]["token_ids"]}

    def voice(text: str) -> List[int]:
        v, _lp = rt.write_voice(text, max_new_tokens=a.max_new, stop_token_ids=stop)
        return [int(t) for t in v["token_ids"]]

    results: Dict[str, Dict[str, Any]] = {}
    failures: List[str] = []
    steer_same_prompt = True
    for q, text in cz.PROMPTS.items():
        composed = rt.compose_prompt(text, chat=True)
        ids = rt.tokenize_prompt(composed)
        chat = rt.tokenize_prompt(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                          add_generation_prompt=True, enable_thinking=False))
        steer_same_prompt = steer_same_prompt and chat == ids
        base = alone(ids)
        shapes: Dict[str, List[int]] = {"alone, again": alone(ids), "steer (as it is, of 3)": steer_row(ids)}
        for B in batches:
            shapes.update(census_rows(ids, B))
        shapes["writer of the sweep and the voice (write_voice)"] = voice(composed)
        results[q] = {"alone": {"n": len(base), "text": tok.decode(base, skip_special_tokens=True)}}
        for name, got in shapes.items():
            results[q][name] = {"shared": shared(got, base), "n": len(got), "same": got == base}
        if shapes["alone, again"] != base:
            failures.append(f"{q}: the answer written alone twice is not the same "
                            f"({results[q]['alone, again']['shared']} word-pieces in common)")
        print(f"{q}: alone {len(base)} word-pieces; " + "; ".join(f"{n} {r['shared']}/{r['n']}{'' if r['same'] else ' parts'}"
                                                                  for n, r in results[q].items() if n != "alone"), flush=True)

    def floor_of(names: List[str]) -> Optional[Dict[str, Any]]:
        parted = [(r["shared"], q, n) for q, rs in results.items() for n, r in rs.items() if n in names and not r["same"]]
        if not parted:
            return None
        k, q, n = min(parted)
        return {"k": k, "question": q, "shape": n}

    names = [n for n in next(iter(results.values())) if n not in ("alone", "alone, again")]
    floor = floor_of(names)
    census_floor = {}
    for B in batches:
        f = floor_of([n for n in names if n.startswith(f"census {B},")])
        census_floor[str(B)] = None if f is None else f["k"]
    by_shape = {n: (lambda f: None if f is None else f["k"])(floor_of([n])) for n in names}
    rec = {
        "what": "with no push, the word-pieces an answer written in each batch shape the atlas writes in holds in common with "
                "the same answer written alone before they first part (scripts/verify_atlas_writer.py)",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": getattr(rt, "model_id", None), "stand_in": bool(getattr(rt, "atlas_stand_in", False)),
        "runtime": cz.runtime_record(w.device, p50_file), "unpacker": unpacker,
        "layer": layer, "steer_neuron": neuron, "steer_dose": STEER_DOSE, "control_dose": CONTROL_DOSE, "seed": a.seed,
        "p50": p50, "p50_source": p50_source, "max_new": a.max_new, "batches": batches, "stop_ids": stop,
        "questions": cz.PROMPTS, "steer_prompt_same": steer_same_prompt,
        "shapes": names, "results": results,
        "floor": None if floor is None else floor["k"], "floor_at": floor, "census_floor": census_floor, "by_shape": by_shape,
        "alone_twice_same": not failures, "seconds": round(time.perf_counter() - t0, 1),
    }
    out = Path(a.out) if a.out else ((folder / "atlas_writer_floor.json") if folder is not None else RESULTS / "atlas_writer_floor.json")
    cz.write_json_atomic(out, rec)
    if floor is None:
        print(f"the writer's floor: no shape parted from the answer written alone within {a.max_new} word-pieces "
              f"({len(results)} questions, {len(names)} shapes)")
    else:
        print(f"the writer's floor: {floor['k']} word-pieces in common ({floor['shape']}, {floor['question']}); census batches: "
              + ", ".join(f"{b}: {'none within ' + str(a.max_new) if k is None else k}" for b, k in census_floor.items()))
    print(f"-> {out}")
    for f_ in failures:
        print(f"  FAIL  {f_}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""What builds a neuron (the folio's practice v upstream, `POST /api/atlas/attn_drive`), checked on the real model
through the running server: do its parts add up there as they do on the CPU stand-in, and do the leads it names move
the neuron when they are taken away?

The drive linearises a neuron's activation in the stream entering its feed-forward block, the norm's rms held, and
splits that linearised total over the heads, the earlier feed-forward blocks (each over its neurons) and the embedding
(atlas_ops.attention_drive). On the stand-in its parts were checked to add up to float error and against a direct-path
finite difference; nothing had run it on Gemma-4-31B, where the captures are bf16 and the weights 4-bit. A part is a
direct write with the norms held, so the drive calls its rows leads to test, not causes found. This measures how far
they lead, by taking each away in a real forward (POST /reads: a head scaled by 0, a neuron's activation multiplied by
0, at every word-piece but the start mark) and reading the neuron again at the same word-piece. That removal's effect
is the whole one: through the layers between, through attention at later word-pieces, with every norm free.

Fixed before the run (29 October 2026):
- Neurons: at layers 10, 20, 30, 40, 45, 50 and 57 (counted from 0; pages 11-58), NEURONS_PER_LAYER a layer drawn
  with --seed, alternately on the + and - face; each read at its strongest stored passage on that face (the passages
  index's first, `GET /neuron/passages`), its ids cut to end at the event and read exactly (`ids`, `at` -1). A neuron
  whose passage is longer than the drive's 512 word-pieces, or that has none on that face, is passed over for the next
  draw.
- The accounting: `rest` against the linearised total (float error only, by construction); each block's neurons added
  up (`neurons_sum`) against the block's part, for the eight blocks the reply lists; the drive's activation (computed
  in float64 from the captured stream) against the activation /reads gives at that word-piece (the model's own).
- The leads: the TOP_HEADS heads and the TOP_NEURONS neurons with the largest |part|; each predicts that taking it away
  moves the activation by about minus its part. Against them, as a null, NULL_HEADS heads drawn at random from the
  layers at or below the neuron's that the drive does not list among its twelve strongest, and NULL_NEURONS neurons
  drawn at random from the blocks of the named neurons, not among those named.
- Read: the share of leads whose removal moves the neuron the predicted way (a change of exactly 0 left out), with a
  Wilson 95% interval; Spearman's rho between the predicted and the measured change; the median of the measured change
  over the predicted; and the chance that a lead's |change| exceeds a null's (an AUC over every lead-null pair of the
  same neuron, pooled), per kind and per layer.

Nothing here decides anything for a page: it says what the drive's rows can bear on this model.

Stop nothing: this talks to the server on :8765 (started with the real indexes, atlas_data/index), as the folio does.
It writes experiments/results/atlas_drive_check/drive_check.json (--out) with the commit, the server's /api/health
(model, unpacker) and every row.

Read after the run (29 October 2026, at the review's asking; `read_after_the_run` in the summary): the leads split by
how far back they were written, how often a random head moved the neuron more than the least-moving named one, and
two limits of the test above (the Wilson intervals treat a neuron's three leads as independent; most random neurons
are silent where they are read, so the comparison with them is with silence). `--summarise` recomputes the summary
from the stored rows without the server.

Run: .venv/bin/python -I scripts/verify_atlas_drive.py [--seed 0] [--base http://127.0.0.1:8765]
     .venv/bin/python -I scripts/verify_atlas_drive.py --summarise      # the summary again, from drive_check.json's rows
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

ROOT_DIR = Path(__file__).resolve().parent.parent
OUT = ROOT_DIR / "experiments" / "results" / "atlas_drive_check" / "drive_check.json"

LAYERS = (10, 20, 30, 40, 45, 50, 57)
NEURONS_PER_LAYER = 6
TOP_HEADS = 3
TOP_NEURONS = 3
NULL_HEADS = 3
NULL_NEURONS = 3
MAX_DRIVE = 512


def post(c: httpx.Client, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
    r = c.post(path, json=body)
    if r.status_code != 200:
        raise RuntimeError(f"{path} {r.status_code}: {r.text[:400]}")
    return r.json()


def act_at(read: Dict[str, Any], pos: int) -> float:
    """The activation /reads gives at word-piece `pos` (its units hold every piece's value)."""
    for u in read["units"]:
        last = u["pos"]
        first = last - len(u["pieces"]) + 1
        if first <= pos <= last:
            return float(u["pieces"][pos - first])
    raise KeyError(pos)


def wilson(k: int, n: int) -> Tuple[Optional[float], Optional[float]]:
    if n == 0:
        return None, None
    z, p = 1.96, k / n
    d = 1 + z * z / n
    m = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(m - h, 3), round(m + h, 3)


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) < 3:
        return None

    def ranks(v: Sequence[float]) -> List[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return round(sxy / (sx * sy), 3) if sx and sy else None


def median(v: Sequence[float]) -> Optional[float]:
    s = sorted(v)
    if not s:
        return None
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


def summarise(rows: List[Dict[str, Any]], kind: str) -> Dict[str, Any]:
    """The leads of one kind ("head" or "neuron") over `rows` (a neuron each), against their nulls."""
    leads = [x for r in rows for x in r[kind + "s"]]
    signed = [x for x in leads if x["measured"] != 0]
    k = sum(1 for x in signed if math.copysign(1, x["measured"]) == math.copysign(1, x["predicted"]))
    ratios = [x["measured"] / x["predicted"] for x in leads if x["predicted"]]
    wins = ties = pairs = 0
    for r in rows:
        for x in r[kind + "s"]:
            for y in r["null_" + kind + "s"]:
                pairs += 1
                wins += abs(x["measured"]) > abs(y["measured"])
                ties += abs(x["measured"]) == abs(y["measured"])
    nulls = [abs(y["measured"]) for r in rows for y in r["null_" + kind + "s"]]
    return {"leads": len(leads), "signed": len(signed), "the_predicted_way": k,
            "share": round(k / len(signed), 3) if signed else None, "wilson_95": wilson(k, len(signed)),
            "spearman_predicted_measured": spearman([x["predicted"] for x in leads], [x["measured"] for x in leads]),
            "median_measured_over_predicted": round(median(ratios), 3) if ratios else None,
            "auc_lead_over_null": round((wins + 0.5 * ties) / pairs, 3) if pairs else None,
            "median_abs_change_lead_over_act": rounded(median([abs(x["measured"]) / abs(r["act"]) for r in rows for x in r[kind + "s"] if r["act"]]), 4),
            "median_abs_change_null_over_act": rounded(median([abs(y["measured"]) / abs(r["act"]) for r in rows for y in r["null_" + kind + "s"] if r["act"]]), 4),
            "null_changes": len(nulls), "null_changes_exactly_0": sum(1 for v in nulls if v == 0)}


def rounded(x: Optional[float], k: int) -> Optional[float]:
    return None if x is None else round(x, k)


# how far back a lead was written, as read after the run (the review of 29 October 2026): (name, gap = the neuron's
# layer minus the lead's)
DISTANCES = {"head": [("own layer or the one below", 0, 1), ("two or more layers back", 2, 99)],
             "neuron": [("the block below", 1, 1), ("two to four layers back", 2, 4), ("five or more layers back", 5, 99)]}


def by_distance(rows: List[Dict[str, Any]], kind: str) -> Dict[str, Any]:
    """The leads of one kind split by how far back they were written: the share the predicted way and the median of the
    measured change over the predicted. Read after the run, not fixed before it."""
    out = {}
    for name, lo, hi in DISTANCES[kind]:
        xs = [x for r in rows for x in r[kind + "s"] if lo <= r["layer"] - x["layer"] <= hi]
        signed = [x for x in xs if x["measured"] != 0]
        k = sum(1 for x in signed if math.copysign(1, x["measured"]) == math.copysign(1, x["predicted"]))
        ratios = [x["measured"] / x["predicted"] for x in xs if x["predicted"]]
        out[name] = {"leads": len(xs), "signed": len(signed), "the_predicted_way": k,
                     "median_measured_over_predicted": rounded(median(ratios), 3)}
    return out


def recall_hint(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """How often a random head moved the neuron more than the least-moving of its three named heads (read after the
    run): the drive's list is not every head that matters through the layers between."""
    beat = sum(1 for r in rows for y in r["null_heads"] if abs(y["measured"]) > min(abs(x["measured"]) for x in r["heads"]))
    return {"random_heads_moving_more_than_the_least_lead": beat, "random_heads": sum(len(r["null_heads"]) for r in rows),
            "random_heads_at_the_neurons_own_layer": sum(1 for r in rows for y in r["null_heads"] if y["layer"] == r["layer"]),
            "named_heads_at_the_neurons_own_layer": sum(1 for r in rows for x in r["heads"] if x["layer"] == r["layer"])}


def summary_of(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_layer = {str(l): {"heads": summarise([r for r in rows if r["layer"] == l], "head"),
                         "neurons": summarise([r for r in rows if r["layer"] == l], "neuron")} for l in LAYERS}
    rel = [b["rel_err"] for r in rows for b in r["blocks"] if b["rel_err"] is not None]
    ratio = sorted(r["linearised_total"] / r["act"] for r in rows if r["act"])
    return {
        "neurons": len(rows),
        "rest_rel_max": max(r["rest_rel"] for r in rows if r["rest_rel"] is not None),
        "block_sum_rel_err_median": median(rel), "block_sum_rel_err_max": max(rel) if rel else None,
        "act_vs_reads_rel_median": median([r["act_rel_diff"] for r in rows if r["act_rel_diff"] is not None]),
        "act_vs_reads_rel_max": max(r["act_rel_diff"] for r in rows if r["act_rel_diff"] is not None),
        "linearised_total_over_act_median": median(ratio), "linearised_total_over_act_range": [ratio[0], ratio[-1]] if ratio else None,
        "heads": summarise(rows, "head"), "neurons_named": summarise(rows, "neuron"), "by_layer": by_layer,
        "read_after_the_run": {"heads_by_distance": by_distance(rows, "head"), "neurons_by_distance": by_distance(rows, "neuron"),
                               "recall": recall_hint(rows),
                               "note": ("the Wilson intervals treat a neuron's three leads as independent; the random neurons are "
                                        "drawn uniformly from a block, so most are silent there (null_changes_exactly_0): the "
                                        "comparison with them is with silence, not with neurons as active as the leads")}}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--summarise", action="store_true", help="recompute the summary of --out from its stored rows (no server)")
    args = ap.parse_args(argv)
    if args.summarise:
        out = json.loads(args.out.read_text())
        out["summary"] = summary_of(out["rows"])
        out["summary_recomputed_by"] = f"--summarise at {subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT_DIR, capture_output=True, text=True).stdout.strip()}"
        args.out.write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps(out["summary"]["read_after_the_run"], indent=1))
        return 0
    rng = random.Random(args.seed)
    t0 = time.time()
    with httpx.Client(base_url=args.base, timeout=600) as c:
        health = c.get("/api/health").json()
        arch = c.get("/api/atlas/arch").json()
        heads_of = [int(x["n_heads"]) for x in arch["layers"]]
        n_ff = int(arch["d_ff"])
        rows: List[Dict[str, Any]] = []
        for layer in LAYERS:
            got = 0
            while got < NEURONS_PER_LAYER:
                n = rng.randrange(n_ff)
                face = "+" if got % 2 == 0 else "-"
                p = c.get("/api/atlas/neuron/passages", params={"layer": layer, "n": n, "face": face, "k": 1}).json()
                if not p.get("ready"):
                    raise RuntimeError(f"the passages index is not read: {p.get('note')}")
                top = [x for x in p["passages"] if x["kind"] == "top"]
                if not top or top[0]["at"] + 1 > MAX_DRIVE:
                    continue
                ids = top[0]["ids"][: top[0]["at"] + 1]
                d = post(c, "/api/atlas/attn_drive", {"layer": layer, "n": n, "ids": ids, "at": -1, "top": 8})
                pos = d["read_pos"]
                base = act_at(post(c, "/api/atlas/reads", {"layer": layer, "n": n, "ids": ids}), pos)

                def removed(body: Dict[str, Any]) -> float:
                    return act_at(post(c, "/api/atlas/reads", {"layer": layer, "n": n, "ids": ids, **body}), pos) - base

                named_heads = {(h["layer"], h["head"]) for h in d["heads"]}
                heads = [{"layer": h["layer"], "head": h["head"], "predicted": -h["contribution"],
                          "measured": removed({"heads": [{"layer": h["layer"], "head": h["head"], "dose": 0.0}]})}
                         for h in d["heads"][:TOP_HEADS]]
                null_heads = []
                while len(null_heads) < NULL_HEADS:
                    ll = rng.randrange(layer + 1)
                    lh = (ll, rng.randrange(heads_of[ll]))
                    if lh in named_heads or any((x["layer"], x["head"]) == lh for x in null_heads):
                        continue
                    null_heads.append({"layer": lh[0], "head": lh[1],
                                       "measured": removed({"heads": [{"layer": lh[0], "head": lh[1], "dose": 0.0}]})})
                named = sorted(d["neurons"], key=lambda x: -abs(x["contribution"]))[:TOP_NEURONS]
                neurons = [{"layer": x["layer"], "n": x["n"], "act": x["act"], "predicted": -x["contribution"],
                            "measured": removed({"neurons": [{"layer": x["layer"], "n": x["n"], "mode": "mul", "dose": 0.0}]})}
                           for x in named]
                taken = {(x["layer"], x["n"]) for x in d["neurons"]}
                null_neurons = []
                while len(null_neurons) < NULL_NEURONS and named:
                    ln = (named[len(null_neurons) % len(named)]["layer"], rng.randrange(n_ff))
                    if ln in taken:
                        continue
                    taken.add(ln)
                    null_neurons.append({"layer": ln[0], "n": ln[1],
                                         "measured": removed({"neurons": [{"layer": ln[0], "n": ln[1], "mode": "mul", "dose": 0.0}]})})
                blocks = [{"layer": m["layer"], "part": m["contribution"], "neurons_sum": m["neurons_sum"],
                           "rel_err": abs(m["neurons_sum"] - m["contribution"]) / abs(m["contribution"]) if m["contribution"] else None}
                          for m in d["mlps"]]
                row = {"layer": layer, "n": n, "face": face, "window": top[0]["window"], "at": top[0]["at"], "pieces": len(ids),
                       "token": d["token"], "passage_act": top[0]["act"], "act": d["act"], "act_reads": base,
                       "act_rel_diff": abs(d["act"] - base) / abs(base) if base else None,
                       "linearised_total": d["linearised_total"], "rest": d["rest"],
                       "rest_rel": abs(d["rest"]) / abs(d["linearised_total"]) if d["linearised_total"] else None,
                       "attention_sum": d["attention_sum"], "mlp_sum": d["mlp_sum"], "embedding": d["embedding"],
                       "blocks": blocks, "heads": heads, "null_heads": null_heads, "neurons": neurons, "null_neurons": null_neurons}
                rows.append(row)
                got += 1
                print(f"layer {layer} n{n} {face}: act {d['act']:+.3f} (reads {base:+.3f}), total {d['linearised_total']:+.3f}, "
                      f"rest {d['rest']:+.2e}; heads " + ", ".join(f"L{h['layer']}h{h['head']} {h['predicted']:+.3f}->{h['measured']:+.3f}" for h in heads)
                      + "; neurons " + ", ".join(f"L{x['layer']}n{x['n']} {x['predicted']:+.3f}->{x['measured']:+.3f}" for x in neurons), flush=True)
    summary = summary_of(rows)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT_DIR, capture_output=True, text=True).stdout.strip()
    out = {"what": __doc__.split("\n\n")[0], "commit": commit, "seed": args.seed, "layers_from": 0, "layers": list(LAYERS),
           "health": {k: health.get(k) for k in ("model", "model_id", "unpacker", "device", "gpu", "runtime") if k in health},
           "elapsed_s": round(time.time() - t0, 1), "summary": summary, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"wrote {args.out} in {out['elapsed_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

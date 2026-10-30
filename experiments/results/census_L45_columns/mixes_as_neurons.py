#!/usr/bin/env python3
"""Do random mixes of layer 45's own write directions already beat isotropic directions? Builds, on the CPU, a run in
which the 2,048 columns controls (L45-break_rel-columns-controls, seed 11) stand where neurons stand and the layer
run's 2,048 isotropic controls (L45-break_rel, seed 7) stay the controls, with the layer run's unpushed answers (the
same in both, bit for bit; checked). The trial's reader then reads it unchanged (read.sh): an AUC near 0.5 says the
layer's write subspace adds nothing a single neuron's lead could come from; near the neurons' own 0.71, that the lead
is the subspace's. The file is a reading, not a census: its meta says how it was made (`constructed`), and it sits in
atlas_data/census/nulls, which the atlas does not read.

Run: .venv/bin/python -I experiments/results/census_L45_columns/mixes_as_neurons.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from backend import census as cz  # noqa: E402

LAYER_RUN = ROOT / "atlas_data" / "census" / "runs" / "L45-break_rel.jsonl"
NULLS = ROOT / "atlas_data" / "census" / "nulls"
COLUMNS = NULLS / "L45-break_rel-columns-controls.jsonl"
OUT = NULLS / "L45-mixes-as-neurons.jsonl"


def main() -> None:
    ml = json.loads(cz.meta_path_of(LAYER_RUN).read_text())
    mc = json.loads(cz.meta_path_of(COLUMNS).read_text())
    base_l, base_c = {}, {}
    tmp = OUT.with_name(OUT.name + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for r in cz.read_rows(LAYER_RUN):
            if r["kind"] == "baseline":
                base_l[r["prompt"]] = r["token_ids"]
            if r["kind"] in ("baseline", "random"):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
        for r in cz.read_rows(COLUMNS):
            if r["kind"] == "baseline":
                base_c[r["prompt"]] = r["token_ids"]
            elif r["kind"] == "random":
                r = {**r, "kind": "neuron", "mix": True}
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
    if base_l != base_c:
        tmp.unlink()
        raise SystemExit("the two runs' unpushed answers differ: they do not compare")
    os.replace(tmp, OUT)
    k = int(mc["random_controls"])
    meta = dict(ml)
    meta.update({"neurons": f"{k} columns controls standing as neurons", "neuron_ids": f"0-{k - 1}",
                 "jobs": cz.jobs_digest(cz.census_jobs(list(range(k)), int(ml["random_controls"]),
                                                       [tuple(x) for x in ml["design"]], cz.meta_questions(ml))),
                 "plan_kept": False, "plan_left": "constructed from two runs",
                 # the gain the reader checks: the mixes (here "neurons") against the isotropic controls
                 "read_gain": {"neurons": mc["read_gain"]["controls"], "controls": ml["read_gain"]["controls"],
                               "ratio": round(ml["read_gain"]["controls"] / mc["read_gain"]["controls"], 3),
                               "read_by": ml["read_gain"]["read_by"]},
                 "constructed": {"what": "the columns controls of one run standing as neurons against the isotropic "
                                         "controls of another, the unpushed answers the same in both (checked)",
                                 "neurons_from": {"run": COLUMNS.name, "kind": "random", "seed": mc["seed"],
                                                  "control_kind": mc.get("control_kind")},
                                 "controls_from": {"run": LAYER_RUN.name, "seed": ml["seed"], "control_kind": cz.control_kind_of(ml)},
                                 "by": "experiments/results/census_L45_columns/mixes_as_neurons.py"}})
    cz.write_json_atomic(cz.meta_path_of(OUT), meta)
    print(f"-> {OUT} ({n:,} answers: {k:,} mixes as neurons, {ml['random_controls']:,} isotropic controls)")


if __name__ == "__main__":
    main()

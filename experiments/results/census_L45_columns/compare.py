#!/usr/bin/env python3
"""Layer 45's neurons against two nulls, on the CPU: the layer run's 2,048 isotropic controls (seed 7) and 2,048
controls mixed from the layer's own write directions (seed 11), each run ranked by the census's own code and read by
the trial's reader (read.sh). Writes compare.json beside this file: each null's controls, how the neurons rank against
each, which neurons each names and how far the two agree, what the collapse rule hides under each, and the text the
two kinds of control write.

Run: .venv/bin/python -I experiments/results/census_L45_columns/compare.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from backend import census as cz  # noqa: E402

ISO = ROOT / "atlas_data" / "census" / "runs" / "L45-break_rel.jsonl"
COL = ROOT / "atlas_data" / "census" / "nulls" / "L45-break_rel-columns.jsonl"
ISO_VERDICT = ROOT / "experiments" / "results" / "census_L45_whole" / "L45_verdict.json"


def units(run: Path) -> dict:
    return {(u["kind"], int(u["neuron"])): u for u in cz.read_rows(cz.units_path_of(run))}


def null(us: dict) -> dict:
    c = np.array([u["mass"] for k, u in us.items() if k[0] == "random"])
    return {"controls": len(c), "mass_p50_p95_max": [round(float(x), 2) for x in (*np.percentile(c, [50, 95]), c.max())]}


def main() -> None:
    ui, uc = units(ISO), units(COL)
    ks = sorted(k for k in ui if k[0] == "neuron")
    mi, mc = np.array([ui[k]["mass"] for k in ks]), np.array([uc[k]["mass"] for k in ks])
    pi, pc = np.array([ui[k]["p"] for k in ks]), np.array([uc[k]["p"] for k in ks])
    top_i = {k for k in ks if ui[k]["null_pct"] == 1.0}
    top_c = {k for k in ks if uc[k]["null_pct"] == 1.0}
    q_i = {k for k in ks if ui[k]["q"] <= 0.05}
    q_c = {k for k in ks if uc[k]["q"] <= 0.05}
    vi = json.loads(ISO_VERDICT.read_text())["readings"]["48"]
    vc = json.loads((HERE / "L45_columns_verdict.json").read_text())["readings"]["48"]
    named_i, named_c = set(vi["named"]), set(vc["named"])
    by_i = sorted(ks, key=lambda k: -ui[k]["mass"])
    by_c = sorted(ks, key=lambda k: -uc[k]["mass"])
    col_i = json.loads(cz.collapse_path_of(ISO).read_text())
    col_c = json.loads(cz.collapse_path_of(COL).read_text())
    ctl = max(u["mass_shown"] for k, u in uc.items() if k[0] == "random")
    out = {
        "nulls": {"isotropic_seed7": null(ui), "columns_seed11": null(uc)},
        "neurons": len(ks),
        "mass_spearman": round(float(stats.spearmanr(mi, mc)[0]), 4),
        "p_spearman": round(float(stats.spearmanr(pi, pc)[0]), 4),
        "above_every_control": {"isotropic": len(top_i), "columns": len(top_c), "both": len(top_i & top_c),
                                "by_chance": cz.chance_above_all(len(ks), 2048)},
        "q05": {"isotropic": len(q_i), "columns": len(q_c), "both": len(q_i & q_c)},
        "named": {"isotropic": len(named_i), "columns": len(named_c), "both": len(named_i & named_c)},
        "top_by_mass_shared": {str(n): len(set(by_i[:n]) & set(by_c[:n])) for n in (20, 100, 1000)},
        "collapse": {"isotropic_held": col_i["held"], "columns_held": col_c["held"],
                     "columns_controls_needed": col_c["controls_needed"],
                     "described_on_map_columns": sum(1 for k in top_c if uc[k]["mass_shown"] > ctl)},
        "text": {"isotropic": {k: v for k, v in vi["text"].items() if "random" in k},
                 "columns": {k: v for k, v in vc["text"].items() if "random" in k},
                 "neurons": {k: v for k, v in vc["text"].items() if "neuron" in k}},
    }
    cz.write_json_atomic(HERE / "compare.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

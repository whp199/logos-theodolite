#!/usr/bin/env python3
"""The census's collapse rule (backend/census.py COLLAPSE_RULE, the owner's choice of 29 October 2026) applied to
break_rel's trial at layer 45 (1,022 fresh neurons against 1,024 controls), on the CPU: which words it hides, against the
review's list (census_break_rel_trial/summary.json, "labels"), whether a question's own topic words survive, and what
it does to the neurons above every control (their share of mass from hidden words; how many keep a description where
their shown words alone must beat every control's). Writes trial_check.json beside this file.

Run: .venv/bin/python -I experiments/results/census_collapse_rule/trial_check.py
"""

import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from backend import census as cz  # noqa: E402

RUN = ROOT / "atlas_data" / "census" / "trials" / "L45-break_rel-trial.jsonl"


def main() -> None:
    rows = cz.read_slim(RUN)
    col = cz.collapse_words(rows, "break_rel")
    recs = cz.compute_units(rows, max_rel=2.5, reference="controls", exclude="own")
    cz.mark_collapse(recs, col)
    neu = [u for u in recs if u["kind"] == "neuron"]
    ctl = [u for u in recs if u["kind"] == "random"]
    gilt = [u for u in neu if u["null_pct"] == 1.0]
    top_ctl = max(u["mass_shown"] for u in ctl)
    topics = ("orange", "bicycle", "chair", "walk", "free")
    out = {
        "what": __doc__.split("\n\n")[0],
        "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "run": RUN.name, "controls": col["controls"], "controls_needed": col["controls_needed"],
        "held": col["held"], "joined_count": len(col["joined"]), "joined_first": col["joined"][:20],
        "review_words_without_thought_clause": ["and", "idea", "own", "owns", "rest", "same", "sss"],
        "review_words_held_here": [w for w in ("and", "idea", "own", "owns", "rest", "same", "sss") if w in col["words"]],
        "topic_words_hidden": [w for w in topics if w in col["words"]],
        "gilt": len(gilt),
        "gilt_described": sum(1 for u in gilt if u["mass_shown"] > top_ctl),
        "gilt_collapse_mass_median": statistics.median(u["collapse_mass"] for u in gilt),
        "gilt_with_any_collapse_mass": sum(1 for u in gilt if u["collapse_mass"] > 0),
        "gilt_half_or_more": sum(1 for u in gilt if u["collapse_mass"] >= 0.5),
    }
    (HERE / "trial_check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "joined_first"}, indent=1))


if __name__ == "__main__":
    main()

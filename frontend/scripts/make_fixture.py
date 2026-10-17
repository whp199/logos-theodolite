"""Build the recorded-sighting fixture the frontend shows when no backend is running.

Source: experiments/results/E6_tool_audit_02_sweep_demo.json, key "prepend_bos=True" (the study's
demonstration sweep: "Tell me about yourself." wrapped as a chat, answer begun "Hi! I", read at " I"),
plus the per-layer `layer_scalar` model constants from E6_tool_audit_01_basics.json.

Only recorded values are copied. Anything the recording does not hold (the stream strength per layer,
the naked eye's top word-piece, attention/feed-forward norms) is left out, and the UI shows it as a
bracketed placeholder.

    python3 -I frontend/scripts/make_fixture.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "experiments" / "results" / "E6_tool_audit_02_sweep_demo.json"
BASICS = ROOT / "experiments" / "results" / "E6_tool_audit_01_basics.json"
OUT = ROOT / "frontend" / "src" / "fixtures" / "e6-demo-sighting.json"


def r(x: float, n: int) -> float:
    return round(float(x), n)


def main() -> None:
    d = json.loads(SRC.read_text())["prepend_bos=True"]
    basics = json.loads(BASICS.read_text())
    out = {
        "source": "the study’s demonstration sighting (E6), taken before the two V7 fixes",
        "question": "Tell me about yourself.",
        "answerBegins": "Hi! I",
        # The exact template string was not recorded: the lab script applied the tokenizer's chat template
        # (thinking off) to the question and appended "Hi! I" (experiments/scripts/E6_tool_audit_02_sweep_demo.py).
        "textSent": None,
        "readAt": d["target_tok"],
        "readPos": d["target_pos"],
        "nTokens": d["n_tokens"],
        "firstIds": d["first_ids"],
        "continuation": d["generated"],
        "landmarks": d["survey_labels"],
        "az": [r(v, 2) for v in d["azimuth_deg"]],
        "el": [r(v, 2) for v in d["elevation_deg"]],
        "xyz": [[float(f"{c:.4g}") for c in p] for p in d["xyz"]],
        "projFrac": [r(v, 5) for v in d["proj_frac_of_hJ"]],
        "ffnShare": [r(v, 4) for v in d["m1_corrected_ffn_dominance"]],
        "ffnShareBeforeFix": [r(v, 4) for v in d["m1_tool_ffn_dominance"]],
        "layerScalar": [r(v, 4) for v in basics["layer_scalar"]],
        "lensTop": {k: [[t, r(p, 4)] for t, p in v] for k, v in d["jpp_top5"].items()},
        "ranks": {
            "lens": {w: [int(x) for x in v] for w, v in d["jpp_tracked_rank"].items()},
            "eye": {w: [int(x) for x in v] for w, v in d["naive_tracked_rank"].items()},
        },
        "rankLogits": {
            "lens": {w: [r(x, 2) for x in v] for w, v in d["jpp_tracked_logit"].items()},
            "eye": {w: [r(x, 2) for x in v] for w, v in d["naive_tracked_logit"].items()},
        },
        "poleNeurons": {k: [int(x) for x in v] for k, v in d["pole_neurons"].items()},
        "poleActs": {k: [r(x, 4) for x in v] for k, v in d["pole_acts"].items()},
        "poleLabels": {
            "lens": {k: [[t, r(lg, 2)] for t, lg in v] for k, v in d["pole_labels_jpp"].items()},
            "eye": {k: [[t, r(lg, 2)] for t, lg in v] for k, v in d["pole_labels_naive"].items()},
        },
        "threads": [
            {
                "src": [int(f["src"][0]), int(f["src"][1]), float(f["src"][2])],
                "dst": [int(f["dst"][0]), int(f["dst"][1]), float(f["dst"][2])],
                "c": r(f["c"], 3),
                "dot": r(f["dot"], 4),
            }
            for f in d["filaments_top"]
        ],
        "threadCount": d["n_filaments"],
        "kinship": [r(v, 4) for row in d["cka"] for v in row],
        "kinStats": {
            "meanOffDiag": r(d["cka_mean_offdiag"], 4),
            "adjacent": r(d["cka_adjacent_mean"], 4),
            "l11vs51": r(d["cka_L10_vs_L50"], 4),
        },
        "elapsedMs": r(d["elapsed_ms"], 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""A CPU-only check of what the interface sends by default: no model, no GPU, no running server, no browser.

The server keeps the instrument's reading options as first built, for old callers (backend/server.py SweepRequest;
scripts/check_sweep_protocol.py pins them). The interface sends the mends as request options instead. This check
bundles the interface's store (frontend/src/store/useTheodoliteStore.ts) with the frontend's own esbuild, runs it under
node, and reads back the request it would send (`sweepPayload` of its initial state), so what is checked is the code
itself, not a copy of it. It checks:
1. the initial request from words: squared all at once (frame_squaring "lowdin"), on the rows FNW_DEFAULT names
   (raw, as decided: V7 measured no frame on the weighted rows), one frame from the last layer, no centring, the
   neurons ranked by write size (pole_ranking "write"), the threads by the direct-path estimate;
2. the initial request in a neuron frame (pole_motif): squared in order (Gram–Schmidt), its axes having fixed roles;
3. the presets: "as_built" sends exactly the server's own defaults for every reading option, in both survey modes;
   "mended" sends the initial request again; neither changes a centring the reader chose;
4. the text the other checks and pages lean on: FRAME_AS_BUILT is the frame as first built, word for word; the
   recorded sighting keeps its ranking by |a| and no echoed frame; frame.tsx's studyComparable is false for a frame
   on the weighted rows; README §iv's rows 3 and 8 say the interface sends the mend by default, on the rows decided,
   and that the server's own default stays as first built.

Run: .venv/bin/python -I scripts/check_interface_defaults.py   (exits 0 when every check holds)
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

FRONTEND = ROOT_DIR / "frontend"
STORE_DIR = FRONTEND / "src" / "store"
ESBUILD = FRONTEND / "node_modules" / ".bin" / "esbuild"

# The user's decisions for track A1: the frame on raw rows, the neurons ranked by write size.
DECIDED_FNW_DEFAULT = False
DECIDED_POLE_RANKING = "write"
# The request's reading options (READING_KEYS in the store) that this check compares with the server's defaults.
READING_FIELDS = ["frame_squaring", "frame_per_layer", "frame_fnw", "centre_prompts", "centre_answer", "pole_ranking",
                  "thread_score"]

failures: List[str] = []
passed = 0


def check(cond: bool, what: str) -> None:
    global passed
    if cond:
        passed += 1
    else:
        failures.append(what)
        print(f"  FAILED: {what}")


# What the bundle prints: the store's constants, and the requests it would send from its initial state, through the
# presets, and with a centring chosen. Every request is the store's own sweepPayload.
ENTRY = r"""
import * as T from "./useTheodoliteStore";
const st = () => T.useTheodoliteStore.getState();
const send = (over = {}) => {
  const s = { ...st(), ...over };
  return T.sweepPayload(s, s.question, s.answerBegins, s.steering);
};
const NEURON = { surveyMode: "pole_motif", lockedPole: [40, 1234, 1] };
const out = {
  FRAME_AS_BUILT: T.FRAME_AS_BUILT,
  FRAME_MENDED: T.FRAME_MENDED,
  FNW_DEFAULT: T.FNW_DEFAULT,
  POLE_RANKING_DEFAULT: T.POLE_RANKING_DEFAULT,
  READING_AS_BUILT: T.READING_AS_BUILT,
  READING_MENDED: T.READING_MENDED,
  initial: { frame: st().frame, neuronSquaring: st().neuronSquaring, poleRanking: st().poleRanking, threadScore: st().threadScore },
  words: send(),
  neuron: send(NEURON),
};
st().setReadingPreset("as_built");
out.asBuiltWords = send();
out.asBuiltNeuron = send(NEURON);
st().setReadingPreset("mended");
out.mendedWords = send();
out.mendedNeuron = send(NEURON);
st().setFrame({ centre: true });
st().setReadingPreset("as_built");
out.centreKeptAsBuilt = st().frame.centre;
st().setReadingPreset("mended");
out.centreKeptMended = st().frame.centre;
console.log(JSON.stringify(out));
"""


def run_store() -> Dict[str, Any] | None:
    """Bundle the store with the frontend's esbuild and run it under node; None (and a failure) when it cannot run."""
    node = shutil.which("node")
    check(ESBUILD.exists(), f"{ESBUILD} is missing: run npm install in frontend/")
    check(node is not None, "node is not on PATH")
    if not ESBUILD.exists() or node is None:
        return None
    bundle = subprocess.run(
        [str(ESBUILD), "--bundle", "--platform=node", "--format=esm", "--loader=ts", "--log-level=error",
         "--sourcefile=check_interface_defaults_entry.ts"],
        input=ENTRY, cwd=STORE_DIR, capture_output=True, text=True, timeout=120, check=False,
    )
    check(bundle.returncode == 0, f"esbuild could not bundle the store:\n{bundle.stderr}")
    if bundle.returncode != 0:
        return None
    ran = subprocess.run([node, "--input-type=module"], input=bundle.stdout, capture_output=True, text=True,
                         timeout=120, check=False)
    check(ran.returncode == 0, f"the bundled store did not run under node:\n{ran.stderr}")
    if ran.returncode != 0:
        return None
    return json.loads(ran.stdout.strip().splitlines()[-1])


def server_defaults() -> Dict[str, Any]:
    """The server's own defaults for the reading options (SweepRequest, as first built)."""
    import backend.server as server

    r = server.SweepRequest(prompt="x")
    return {key: getattr(r, key) for key in READING_FIELDS}


def reading(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {key: payload.get(key) for key in READING_FIELDS}


def check_requests(out: Dict[str, Any], server: Dict[str, Any]) -> None:
    print("1. the initial request from words")
    words = out["words"]
    check(out["FNW_DEFAULT"] is DECIDED_FNW_DEFAULT, f"FNW_DEFAULT is {out['FNW_DEFAULT']!r}, not the decided "
          f"{DECIDED_FNW_DEFAULT!r}")
    check(out["POLE_RANKING_DEFAULT"] == DECIDED_POLE_RANKING, f"POLE_RANKING_DEFAULT is "
          f"{out['POLE_RANKING_DEFAULT']!r}, not the decided {DECIDED_POLE_RANKING!r}")
    check(out["FRAME_MENDED"] == {"squaring": "lowdin", "perLayer": False, "fnw": out["FNW_DEFAULT"], "centre": False},
          f"FRAME_MENDED is {out['FRAME_MENDED']}")
    check(out["initial"]["frame"] == out["FRAME_MENDED"], f"the initial frame is {out['initial']['frame']}, not "
          "FRAME_MENDED")
    want = {"frame_squaring": "lowdin", "frame_per_layer": False, "frame_fnw": DECIDED_FNW_DEFAULT,
            "centre_prompts": [], "centre_answer": False, "pole_ranking": DECIDED_POLE_RANKING,
            "thread_score": "direct"}
    for key, value in want.items():
        check(words.get(key) == value, f"words: the interface sends {key} {words.get(key)!r}, not {value!r}")
    check("centre_token_ids" not in words, f"words: a centring is sent unasked: {words.get('centre_token_ids')!r}")
    check(words.get("survey_mode") == "monument", f"words: survey_mode {words.get('survey_mode')!r}")

    print("2. the initial request in a neuron frame")
    neuron = out["neuron"]
    check(out["initial"]["neuronSquaring"] == "gram_schmidt", f"the initial neuronSquaring is "
          f"{out['initial']['neuronSquaring']!r}")
    check(neuron.get("survey_mode") == "pole_motif", f"neuron: survey_mode {neuron.get('survey_mode')!r}")
    check(neuron.get("frame_squaring") == "gram_schmidt", f"neuron: the interface sends frame_squaring "
          f"{neuron.get('frame_squaring')!r}, not 'gram_schmidt'")
    check(reading({**neuron, "frame_squaring": "lowdin"}) == reading(words),
          "neuron: the other reading options differ from those sent from words")

    print("3. the presets")
    check(out["READING_AS_BUILT"]["frame"] == out["FRAME_AS_BUILT"], "READING_AS_BUILT's frame is not FRAME_AS_BUILT")
    check(out["READING_MENDED"]["frame"] == out["FRAME_MENDED"], "READING_MENDED's frame is not FRAME_MENDED")
    for label in ("asBuiltWords", "asBuiltNeuron"):
        got = reading(out[label])
        check(got == server, f"{label}: the preset as first built sends {got}, not the server's defaults {server}")
    for label, initial in (("mendedWords", "words"), ("mendedNeuron", "neuron")):
        check(reading(out[label]) == reading(out[initial]), f"{label}: the mended preset does not send the initial "
              f"request again: {reading(out[label])}")
    check(out["centreKeptAsBuilt"] is True and out["centreKeptMended"] is True,
          "a preset dropped the centring the reader chose")


def check_text() -> None:
    print("4. the text the checks and pages lean on")
    store = (STORE_DIR / "useTheodoliteStore.ts").read_text(encoding="utf-8")
    check('export const FRAME_AS_BUILT: FrameOptions = { squaring: "gram_schmidt", perLayer: false, fnw: false, '
          'centre: false };' in store, "useTheodoliteStore.ts: FRAME_AS_BUILT is no longer the frame as first built")
    sighting = (STORE_DIR / "sighting.ts").read_text(encoding="utf-8")
    m = re.search(r"export function sightingFromFixture\b(.*?)\n}\n", sighting, re.S)
    check(m is not None, "sighting.ts: sightingFromFixture not found")
    if m is not None:
        body = m.group(1)
        check('poleRanking: "activation",' in body, "sighting.ts: the recorded sighting no longer ranks by |a|")
        check("frame: null," in body, "sighting.ts: the recorded sighting now carries a frame")
    frame = (FRONTEND / "src" / "lib" / "frame.tsx").read_text(encoding="utf-8")
    m = re.search(r"^\s*studyComparable:\s*([^,\n]*),", frame, re.M)
    check(m is not None, "frame.tsx: studyComparable not found")
    if m is not None:
        check("!fnw" in m.group(1), f"frame.tsx: studyComparable ignores the weighted rows: {m.group(1)}")
    readme = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
    rows = {n: re.search(rf"^\| \*\*{n}\*\* \|.*$", readme, re.M) for n in (3, 8)}
    for n, row in rows.items():
        check(row is not None, f"README.md: row {n} of the fault table not found")
    if rows[3] is not None:
        row3 = rows[3].group(0)
        check('`frame_squaring="lowdin"`' in row3 and "which the interface sends by default" in row3,
              "README.md row 3 does not say the interface sends the Löwdin frame by default")
        check("server’s own default stays Gram–Schmidt on raw rows" in row3,
              "README.md row 3 does not say the server keeps Gram–Schmidt on raw rows")
        rows_said = '"lowdin"`, `frame_fnw`' in row3 if DECIDED_FNW_DEFAULT else '(order-free) on the raw rows' in row3
        check(rows_said, "README.md row 3 does not name the rows the interface's frame is squared from")
    if rows[8] is not None:
        row8 = rows[8].group(0)
        check('`pole_ranking="write"`' in row8 and "the interface sends it by default" in row8,
              "README.md row 8 does not say the interface ranks by write size by default")
        check("server’s own default stays \\|a\\|" in row8, "README.md row 8 does not say the server keeps |a|")


def main() -> int:
    out = run_store()
    if out is not None:
        check_requests(out, server_defaults())
    check_text()
    if failures:
        print(f"\ninterface defaults: {len(failures)} of {passed + len(failures)} checks FAILED")
        return 1
    print(f"\ninterface defaults: all {passed} checks hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())

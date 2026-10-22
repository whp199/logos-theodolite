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
   in pole_motif with no neuron chosen, which the server reads in the frame from words, the words' options exactly;
3. the presets: "as_built" sends exactly the server's own defaults for every reading option, in both survey modes;
   "mended" sends the initial request again; neither changes a centring the reader chose;
4. the text the other checks and pages lean on: FRAME_AS_BUILT is the frame as first built, word for word; the
   recorded sighting keeps its ranking by |a| and no echoed frame; frame.tsx's studyComparable is false for a frame
   on the weighted rows; README §iv's rows 3 and 8 say the interface sends the mend by default, on the rows decided,
   and that the server's own default stays as first built;
5. the text pages give for E3 and the order: E3's seven sampled layers give 10–41% at layers 21–59 and 70–74% at
   layers 3 and 11, and no page names the five later layers alone as those sampled; no page says a Löwdin bearing does
   not change with the order, and the first page's caution does not give the 58–99° swing as the default frame's;
6. thinking on bare text (E6 §3, item 11): the interface sends think_bare "none" (the server keeps "prefix", the old
   imitation, for old callers), and compare_think as asked, so with bare text the server leaves thinking out and says
   so; the old imitation goes only when chosen; effectiveThink, which the pace and wording follow, is true for a chat
   or for bare text with the imitation chosen, and only when asked;
7. one word against another (Landmarks section vii): by default, and whenever the section is off or its words are not
   typed both sides (or do not pair), the interface sends no contrast and no background; turned on with words typed it
   sends exactly those words, split at commas; "controls" only with a matched control sent (the server refuses it
   otherwise), "study_mean" not when the server's health says it cannot read its file; the request it sends is one the
   server takes; contrasts and contrast_backgrounds are reading keys (a re-reading carries them as set now);
8. the vault's voice: the question an edit is heard by is sent as the instrument sends its own (a chat with its system
   prompt and begun answer, or bare text whose begun answer the server puts on a new line, as composeText does), and the
   server's VoiceRequest takes it; executeLarql sends a voice only when given; the reply's voice is kept by patch and
   time, a DESCRIBE keeps only the voices of records it still holds, and ROLLBACK forgets them all;
9. what the pages say of these readings: every study figure section vii quotes (the average state's share, raw and
   centred scores through the lens, its naked-eye background, E3 (b)'s agreement at the lowest layers) is the one in
   E3's files, rounded as shown; the record of E3's average state in this server's coordinates
   (E3_background_mu_check.json) is of the pinned file; and the claims the GPU stage or this track made untrue stay
   gone (that μ's coordinates are not checked, that contrast axes are not built, that every pair names think_protocol,
   that a replacing edit's "before" is the replaced edit's voice, that layers 1-10 were not tested, the column
   "overwrite").

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
// neuron mode before a neuron is chosen: the server builds the frame from words (engine_runtime.py prepare_sweep)
const NO_NEURON = { surveyMode: "pole_motif", lockedPole: null };
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
  noNeuron: send(NO_NEURON),
};
st().setReadingPreset("as_built");
out.asBuiltWords = send();
out.asBuiltNeuron = send(NEURON);
st().setReadingPreset("mended");
out.mendedWords = send();
out.mendedNeuron = send(NEURON);
out.think = {
  initialThinkBare: st().thinkBare,
  chat: send({ compareThink: true }),
  bare: send({ compareThink: true, chat: false }),
  barePrefix: send({ compareThink: true, chat: false, thinkBare: "prefix" }),
  effective: [
    T.effectiveThink({ compareThink: true, chat: true }),
    T.effectiveThink({ compareThink: true, chat: false }),
    T.effectiveThink({ compareThink: true, chat: false, thinkBare: "prefix" }),
    T.effectiveThink({ compareThink: false, chat: true }),
    T.effectiveThink({ compareThink: false, chat: false, thinkBare: "prefix" }),
  ],
};
// one word against another: what the store sends for section vii as set
const OTHER = { question: "Tell me about a friend.", answerBegins: "", chatWrapped: true, system: "", promptIds: [2, 105, 7],
  textSent: "x", tokenIndex: 0 };
const on = (c, over = {}) => send({ contrast: { ...T.CONTRAST_OFF, on: true, ...c }, ...over });
out.contrast = {
  initial: st().contrast,
  off: send({ contrast: { ...T.CONTRAST_OFF, toward: "she", away: "he" } }),
  onEmpty: on({}),
  onOneSide: on({ toward: "she" }),
  onUnpaired: on({ toward: "she, her", away: "he" }),
  onWords: on({ toward: " she ", away: "he" }),
  onPairs: on({ toward: "she, her,", away: "he , him" }),
  controlsNoControl: on({ toward: "she", away: "he", backgrounds: ["controls"] }),
  controlsNotCentred: on({ toward: "she", away: "he", backgrounds: ["controls"] }, { control: OTHER }),
  controlsSent: on({ toward: "she", away: "he", backgrounds: ["controls"] }, { control: OTHER, frame: { ...st().frame, centre: true } }),
  meanNoHealth: on({ toward: "she", away: "he", backgrounds: ["study_mean"] }, { health: null }),
  meanReadable: on({ toward: "she", away: "he", backgrounds: ["study_mean"] }, { health: { contrasts: { study_mean_problem: null } } }),
  meanProblem: on({ toward: "she", away: "he", backgrounds: ["study_mean"] }, { health: { contrasts: { study_mean_problem: "missing" } } }),
  problems: [T.contrastProblem({ ...T.CONTRAST_OFF }), T.contrastProblem({ ...T.CONTRAST_OFF, on: true }),
    T.contrastProblem({ ...T.CONTRAST_OFF, on: true, toward: "a,b", away: "c" }),
    T.contrastProblem({ ...T.CONTRAST_OFF, on: true, toward: "a", away: "c" })],
};
// the vault's voice: what an edit is heard by, and what executeLarql sends (over HTTP: no socket is open here)
out.voice = {
  chat: T.voicePayload(" What is a spider? ", " Hi! I ", true, " Be brief. "),
  chatNoSystem: T.voicePayload("Q?", "", true, "  "),
  bare: T.voicePayload("Q?", " A spider has", false, "ignored"),
  composed: T.composeText("Q?", " A spider has"),
};
const posted = [];
const rec = (id, at, voice) => ({ patch_id: id, op_type: "UPDATE", layer_idx: 35, neuron_idx: 100, target_concept: "six",
  statement: "UPDATE LAYER 35 SLOT 100 SET object = 'six' STRENGTH 3.0 POLARITY +", timestamp: at, voice });
const heard = { prompt: "Q?", chat: true, system: null, assistant_prefix: "", max_new_tokens: 12, edits_in_force_before: [],
  before: { token_ids: [1, 2], token_strs: ["A", " b"], text: "A b", stop_reason: "length", n_prompt: 9, steps: [], first_top: [], watch: null },
  after: { token_ids: [1, 3], token_strs: ["A", " c"], text: "A c", stop_reason: "length", n_prompt: 9, steps: [], first_top: [], watch: null },
  first_step: { kl_before_after: 0.5 }, parted_at: 1, elapsed_ms: 10, error: null };
let reply = null;
globalThis.fetch = async (path, init) => {
  posted.push({ path, body: JSON.parse(init.body) });
  return { ok: true, json: async () => reply };
};
const settle = () => new Promise((r) => setTimeout(r, 0));
reply = { status: "applied", patch: rec(7, 100.5, heard), active_patches_count: 1, voice: heard };
st().executeLarql("UPDATE LAYER 35 SLOT 100 SET object = 'six' STRENGTH 3.0 POLARITY +", out.voice.chat);
await settle(); await settle();
out.voice.afterEdit = Object.keys(st().editVoices);
out.voice.afterText = st().editVoices["7:100.5"]?.after?.text ?? null;
reply = { command: "DESCRIBE", active_patches_count: 1, active_patches: [rec(7, 100.5, heard), rec(3, 1.5, null)] };
st().executeLarql("DESCRIBE");
await settle(); await settle();
out.voice.afterDescribe = Object.keys(st().editVoices);
reply = { command: "DESCRIBE", active_patches_count: 0, active_patches: [] };
st().executeLarql("DESCRIBE");
await settle(); await settle();
out.voice.afterRestart = Object.keys(st().editVoices);
reply = { status: "applied", patch: rec(8, 200, heard), active_patches_count: 1, voice: heard };
st().executeLarql("UPDATE LAYER 35 SLOT 100 SET object = 'six' STRENGTH 3.0 POLARITY +", out.voice.chat);
await settle(); await settle();
reply = { status: "rolled_back", cleared_patches: 1, active_patches_count: 0 };
st().executeLarql("ROLLBACK");
await settle(); await settle();
out.voice.afterRollback = Object.keys(st().editVoices);
out.voice.posted = posted;
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
    no_neuron = out["noNeuron"]
    check(no_neuron.get("locked_pole") is None, f"no neuron chosen: locked_pole {no_neuron.get('locked_pole')!r}")
    check(reading(no_neuron) == reading(words), f"no neuron chosen: the server falls back to the frame from words, "
          f"but the interface sends {reading(no_neuron)}, not the words' {reading(words)}")

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


def check_think(out: Dict[str, Any]) -> None:
    print("6. thinking on bare text")
    import backend.server as server

    t, words = out["think"], out["words"]
    check(t["initialThinkBare"] == "none", f"the initial thinkBare is {t['initialThinkBare']!r}, not 'none'")
    check(words.get("think_bare") == "none" and words.get("compare_think") is False,
          f"words: think_bare {words.get('think_bare')!r}, compare_think {words.get('compare_think')!r}")
    check(server.SweepRequest(prompt="x").think_bare == "prefix", "the server's own default is no longer the imitation")
    check(t["chat"].get("compare_think") is True and t["chat"].get("chat") is True, f"a chat: {t['chat'].get('chat')}")
    check(t["bare"].get("compare_think") is True and t["bare"].get("think_bare") == "none"
          and t["bare"].get("chat") is False, "bare text: compare_think sent as asked, with think_bare 'none'")
    check(t["barePrefix"].get("think_bare") == "prefix", "bare text, the imitation chosen: think_bare 'prefix'")
    check(t["effective"] == [True, False, True, False, False], f"effectiveThink {t['effective']}")


def check_contrasts(out: Dict[str, Any]) -> None:
    print("7. one word against another")
    import backend.server as server

    c, words = out["contrast"], out["words"]
    none = {"contrasts": [], "contrast_backgrounds": []}
    pick = lambda p: {"contrasts": p.get("contrasts"), "contrast_backgrounds": p.get("contrast_backgrounds")}  # noqa: E731
    check(c["initial"] == {"on": False, "toward": "", "away": "", "backgrounds": []}, f"the initial contrast is {c['initial']}")
    for label, p in (("words", words), ("off", c["off"]), ("onEmpty", c["onEmpty"]), ("onOneSide", c["onOneSide"]),
                     ("onUnpaired", c["onUnpaired"])):
        check(pick(p) == none, f"{label}: the interface sends {pick(p)}, not no contrast")
    she_he = [{"toward": ["she"], "away": ["he"]}]
    check(pick(c["onWords"]) == {"contrasts": she_he, "contrast_backgrounds": []}, f"onWords: {pick(c['onWords'])}")
    check(pick(c["onPairs"])["contrasts"] == [{"toward": ["she", "her"], "away": ["he", "him"]}],
          f"onPairs: the words are not split at commas: {pick(c['onPairs'])}")
    for label in ("controlsNoControl", "controlsNotCentred"):
        p = c[label]
        check(p.get("contrast_backgrounds") == [] and not p.get("centre_prompts") and "centre_token_ids" not in p,
              f"{label}: 'controls' sent with no control: {pick(p)}")
    p = c["controlsSent"]
    check(p.get("contrast_backgrounds") == ["controls"] and p.get("centre_token_ids") == [[2, 105, 7]],
          f"controlsSent: {pick(p)}, centre_token_ids {p.get('centre_token_ids')}")
    check(c["meanNoHealth"].get("contrast_backgrounds") == ["study_mean"], "study_mean dropped before the health is read")
    check(c["meanReadable"].get("contrast_backgrounds") == ["study_mean"], "study_mean dropped when the server can read it")
    check(c["meanProblem"].get("contrast_backgrounds") == [], "study_mean sent when the server says it cannot read it")
    probs = c["problems"]
    check(probs[0] is None and isinstance(probs[1], str) and "pairs" in (probs[2] or "") and probs[3] is None,
          f"contrastProblem {probs}")
    # what the store sends is a request the server takes: the fields parse, and pass check_sweep_options
    for label in ("words", "onWords", "onPairs", "controlsSent", "meanReadable"):
        p = out[label] if label == "words" else c[label]
        try:
            req = server.SweepRequest(**{k: v for k, v in p.items() if k in server.SweepRequest.model_fields})
            server.check_sweep_options(req)
            ok = [x.model_dump(exclude={"label"}) for x in req.contrasts] == p["contrasts"] and \
                req.contrast_backgrounds == p["contrast_backgrounds"]
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"    {label}: {type(exc).__name__}: {exc}")
        check(ok, f"{label}: the server does not take the contrast fields the interface sends")
    store = (STORE_DIR / "useTheodoliteStore.ts").read_text(encoding="utf-8")
    m = re.search(r"const READING_KEYS = \[(.*?)\] as const;", store, re.S)
    check(m is not None and '"contrasts"' in m.group(1) and '"contrast_backgrounds"' in m.group(1),
          "READING_KEYS does not hold contrasts and contrast_backgrounds")


def check_voice(out: Dict[str, Any]) -> None:
    print("8. the vault's voice")
    import backend.server as server
    from backend.larql_vault import voice_spec

    v = out["voice"]
    check(v["chat"] == {"prompt": "What is a spider?", "chat": True, "system": "Be brief.", "assistant_prefix": "Hi! I",
                        "max_new_tokens": 12}, f"a chat's voice: {v['chat']}")
    check(v["chatNoSystem"]["system"] is None, f"a blank system prompt is sent as {v['chatNoSystem']['system']!r}")
    check(v["bare"] == {"prompt": "Q?", "chat": False, "system": None, "assistant_prefix": "A spider has",
                        "max_new_tokens": 12}, f"bare text's voice: {v['bare']}")
    # the server's bare composition (compose_prompt, chat False): the text, then the begun answer on a new line
    b = v["bare"]
    check(f"{b['prompt']}\n{b['assistant_prefix'].strip()}" == v["composed"], "a bare voice is not the text composeText sends")
    for label in ("chat", "chatNoSystem", "bare"):
        try:
            req = server.LarqlRequest(query="DESCRIBE", voice=v[label])
            spec = voice_spec(req.voice)
            ok = spec["prompt"] == v[label]["prompt"] and spec["chat"] == v[label]["chat"]
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"    {label}: {type(exc).__name__}: {exc}")
        check(ok, f"{label}: the server's VoiceRequest does not take the voice the interface sends")
    posted = v["posted"]
    check([p["path"] for p in posted] == ["/api/larql"] * 5, f"executeLarql posted to {[p['path'] for p in posted]}")
    if len(posted) == 5:
        check(posted[0]["body"].get("voice") == v["chat"], "the edit was not sent with its voice")
        check("voice" not in posted[1]["body"] and "voice" not in posted[4]["body"],
              "a statement sent without a voice carries one")
    check(v["afterEdit"] == ["7:100.5"] and v["afterText"] == "A c", f"the reply's voice was not kept: {v['afterEdit']}")
    check(v["afterDescribe"] == ["7:100.5"], f"DESCRIBE kept {v['afterDescribe']}")
    check(v["afterRestart"] == [], f"a DESCRIBE with no records kept {v['afterRestart']}")
    check(v["afterRollback"] == [], f"ROLLBACK kept {v['afterRollback']}")


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


# E3 sampled seven code layers (E3_lens_validation_d_analysis.json summary.pole): 2, 10, 20, 30, 40, 50 and 58, which
# pages count as layers 3, 11, 21, 31, 41, 51 and 59. The fifteen explained 10–41% of the FFN write at the last five and
# 70–74% at the first two, so no text may call the last five all the layers sampled.
E3_SAMPLED_WRONG = [
    "the layers E3 sampled (21",
    "at the layers it sampled, 21",
    "at the sampled layers 21",
    "the sampled index 20",
    "the layers the study looked at",
    "10–41% where the study looked)",
]
E3_PAGES = ["screens/Instrument.tsx", "screens/NeuronFolio.tsx", "screens/Lexicon.tsx", "screens/LayerLeaf.tsx",
            "screens/lineage/Pillars.tsx"]


def check_e3_and_order_text() -> None:
    print("5. what pages say E3 sampled, and what reordering does in the default frame")
    e3 = json.loads((ROOT_DIR / "experiments" / "results" / "E3_lens_validation_d_analysis.json").read_text())
    pole = e3["summary"]["pole"]
    for key in ("frac_ffn_top15_act", "frac_ffn_top15_write"):
        check(sorted(map(int, pole[key])) == [2, 10, 20, 30, 40, 50, 58], f"E3 {key}: the sampled layers changed")
        late = [pole[key][str(i)] for i in (20, 30, 40, 50, 58)]
        early = [pole[key][str(i)] for i in (2, 10)]
        check(f"{min(late) * 100:.0f}–{max(late) * 100:.0f}%" == "10–41%", f"E3 {key}: layers 21–59 give {late}")
        check(f"{min(early) * 100:.0f}–{max(early) * 100:.0f}%" == "70–74%", f"E3 {key}: layers 3 and 11 give {early}")
    texts = {name: (FRONTEND / "src" / name).read_text(encoding="utf-8") for name in E3_PAGES}
    texts["README.md"] = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
    texts["ROADMAP.md"] = (ROOT_DIR / "ROADMAP.md").read_text(encoding="utf-8")
    for name, text in texts.items():
        flat = re.sub(r"\s+", " ", text)
        for phrase in E3_SAMPLED_WRONG:
            check(phrase not in flat, f"{name}: says {phrase!r}, leaving out layers 3 and 11, which E3 also sampled")
        if "10–41%" in flat:
            check("3 and 11" in flat or "2 and 10" in flat, f"{name}: quotes E3's 10–41% without layers 3 and 11")
    # under the default Löwdin frame the order relabels the axes: no page may say the bearing does not change with it,
    # and the first page's caution must not give the 58–99° swing as the default frame's
    flat = {name: re.sub(r"\s+", " ", (FRONTEND / "src" / name).read_text(encoding="utf-8"))
            for name in ("screens/Landmarks.tsx", "screens/LayerLeaf.tsx", "screens/Incipit.tsx", "lib/frame.tsx",
                         "screens/lexicon/Cautions.tsx")}
    for name, text in flat.items():
        for phrase in ("does not change with the order", "only which axis is", "says only which axis"):
            check(phrase not in text, f"{name}: says {phrase!r}, but in a Löwdin frame the order relabels the axes")
    check("depends on the order of the three landmark words. In the study" not in flat["screens/Incipit.tsx"],
          "Incipit.tsx: the first page's caution gives the 58–99° order swing as the default frame's")


def _band(xs: Any, lo: int, hi: int) -> float:
    """The mean over report layers lo..hi-1 of a per-layer series, or of a [passages, 60] array's passage mean."""
    import numpy as np

    a = np.array(xs, dtype=float)
    return float((a.mean(0) if a.ndim > 1 else a)[lo:hi].mean())


def _shown(v: float) -> str:
    """A figure as the pages print it: one decimal, a true minus sign, a plus sign before a positive figure."""
    t = f"{abs(v):.1f}"
    return ("−" if v < 0 else "+") + t


def check_contrast_text() -> None:
    print("9. what the pages say of one word against another, of thinking and of the vault")
    results = ROOT_DIR / "experiments" / "results"
    rec = json.loads((results / "E3_lens_validation_c_background.json").read_text())
    pn = rec["real"]["prose_neutral"]
    sec = re.sub(r"\s+", " ", (FRONTEND / "src" / "screens" / "landmarks" / "ContrastSection.tsx").read_text(encoding="utf-8"))
    readme = re.sub(r"\s+", " ", (ROOT_DIR / "README.md").read_text(encoding="utf-8"))
    # through the lens, on neutral prose at report L40-49 (page 41-50): the average state's share, raw, centred
    for key, said in (("jpp|meanpart|pron", "+1.4"), ("jpp|raw|pron", "+0.8"), ("jpp|cent|pron", "−0.7")):
        got = _shown(_band(pn[key], 40, 50))
        check(got == said and said in sec and said in readme, f"neutral prose {key} at L40-49 is {got}, the pages say {said}")
    check(_shown(_band(rec["mu"]["jpp"]["pron"], 40, 50)) == "+2.3", "μ's own pronoun score at L40-49 is not +2.3")
    # by the naked eye: μ, and neutral prose raw, μ's share and centred, by ten layers
    naked = [(rec["mu"]["naive"]["pron"], 0, 20, "−4.3"), (rec["mu"]["naive"]["pron"], 40, 50, "−0.9"),
             (pn["naive|raw|pron"], 0, 10, "−3.8"), (pn["naive|raw|pron"], 10, 20, "−6.5"),
             (pn["naive|raw|pron"], 20, 30, "−3.6"), (pn["naive|meanpart|pron"], 0, 10, "−3.4"),
             (pn["naive|meanpart|pron"], 10, 20, "−4.8"), (pn["naive|meanpart|pron"], 20, 30, "−1.3"),
             (pn["naive|cent|pron"], 10, 20, "−4.8"), (pn["naive|cent|pron"], 20, 30, "−5.7")]
    for xs, lo, hi, said in naked:
        got = _shown(_band(xs, lo, hi))
        check(got == said and said in sec, f"the naked eye at L{lo}-{hi - 1} reads {got}, section vii says {said}")
    # E3 (b): the lens's pull-back of the pronoun direction against the model's own, at L0-9 and L10-19
    report = (results / "E3_lens_validation_REPORT.md").read_text(encoding="utf-8")
    for row, said in (("cos(gbar, J++ pullback)", ("0.07", "0.03")), ("split-half reliability", ("0.06", "0.02"))):
        m = re.search(r"^\| " + re.escape(row) + r" \| ([^|]+) \| ([^|]+) \|", report, re.M)
        check(m is not None and (m.group(1).strip(), m.group(2).strip()) == said and all(x in sec for x in said),
              f"E3 (b) {row} at L0-9, L10-19: {m.groups() if m else None}, section vii says {said}")
    # the record of μ in this server's coordinates is of the pinned file, and its band figure is the one the page gives
    from backend.engine_runtime import STUDY_MEAN_SHA256

    mc = json.loads((results / "E3_background_mu_check.json").read_text())
    check(mc["mu_sha256"] == STUDY_MEAN_SHA256, "E3_background_mu_check.json is not of the pinned average state")
    band = mc["mu_pronoun_score"]["band_L40_49"]
    check(f"+{band['server_bf16_lens']:.2f}" == "+2.35" and "+2.35" in sec, f"μ today at L40-49: {band['server_bf16_lens']}")
    check(mc["coordinates"]["worst_relative_difference"] is not None and mc["coordinates"]["worst_relative_difference"] < 1e-2,
          "the record does not show the server's resid_post within 1e-2 of the study's capture")
    # claims made untrue stay gone
    gone = {
        "screens/landmarks/ContrastSection.tsx": ("[NOT YET MEASURED]", "is not yet checked", "Not tested by the study",
                                                 "it reads +2.3 toward", "none of these bands is about it",
                                                 "[NOT YET IN THE SERVER]", "Read in fp32 on the pair rows"),
        "screens/lexicon/Cautions.tsx": ("Contrastive axes are not built", "every reply names", "it is no longer offered"),
        "screens/vault/EditPanel.tsx": ("“before” is that edit’s voice", "the scientist’s tools"),
        "screens/Lexicon.tsx": ("the scientist’s tools",),
    }
    for name, phrases in gone.items():
        text = re.sub(r"\s+", " ", (FRONTEND / "src" / name).read_text(encoding="utf-8"))
        for phrase in phrases:
            check(phrase not in text, f"{name}: still says {phrase!r}")
    for name, phrases in {"README.md": ("is not yet checked", "every pair names"),
                          "ROADMAP.md": ("is not yet checked", "every pair names", "layers 1–10 as never tested"),
                          "ARCHITECTURE.md": ("is not yet checked", "DESCRIBE, DELETE and ROLLBACK reply as before"),
                          "REFERENCES.md": ("column overwrite",)}.items():
        text = re.sub(r"\s+", " ", (ROOT_DIR / name).read_text(encoding="utf-8"))
        for phrase in phrases:
            check(phrase not in text, f"{name}: still says {phrase!r}")


def main() -> int:
    out = run_store()
    if out is not None:
        check_requests(out, server_defaults())
    check_text()
    check_e3_and_order_text()
    check_contrast_text()
    if out is not None:
        check_think(out)
        check_contrasts(out)
        check_voice(out)
    if failures:
        print(f"\ninterface defaults: {len(failures)} of {passed + len(failures)} checks FAILED")
        return 1
    print(f"\ninterface defaults: all {passed} checks hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())

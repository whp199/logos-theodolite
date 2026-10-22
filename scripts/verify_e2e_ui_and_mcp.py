#!/usr/bin/env python3
"""End-to-End Verification Suite (`scripts/verify_e2e_ui_and_mcp.py`) for Logos Theodolite.

Runs inside a single process/network namespace to verify:
0. The lens card on the CPU, before anything loads (scripts/check_lens_card.py, in its own process): the shipped file,
   incomplete files and other lenses, and which vault statements name the lens. Then, also on the CPU, what the
   interface sends by default (scripts/check_interface_defaults.py: the store bundled and run under node).
1. FastAPI + Binary Float32 WebSocket (`/ws/theodolite`) wire format (`0x5448454F`) & JSON metadata
2. FastMCP Scientist Tools (`inspect_pole`, `trace_downstream_readers`, `test_motif_hypothesis`)
3. LARQL / VINDEX3 FFN Vault queries (`query_larql_vault`, `/api/larql`, the WebSocket `larql` action)
4. Headless Chrome inspection of the UI (frontend/dist, built first) via CDP:
   - Zero console errors or page exceptions
   - the first live sighting replaces the recorded one and the 60-ring drawing renders
   - on every page, the "rubrics" and "apparatus" reading toggles show and hide their blocks
   - a gentle nudge (×0.01 of the stream) round-trips; screenshots saved to $THEODOLITE_ARTIFACT_DIR
     (default: artifacts/e2e/)
5. The options added for V7 bugs 1, 3, 4, 5 and 8 (chat template and system turn, Löwdin / weighted / per-layer /
   centred survey frames with proj_frac, the mended nudge, the direct-path thread score, write-size pole ranking),
   over the WebSocket and the MCP tools, while the server's own defaults stay as first built; the first write-ranked
   sweep after the server starts, differing from a plain sweep just taken only in pole_ranking, is timed against an
   identical second one. The interface sends the mends for bugs 3 (the order, on raw rows) and 8 by default: the
   Lexicon's default request says so, the first live sighting is squared all at once and ranked by write size, a
   layer's leaf links to its biggest writer, whose words were read, that neuron's folio gives its write size and
   its |a| rank of the fifteen listed, unticking the ranking retakes it by |a|, and "the frame options
   as first built" on the landmarks page retakes it squared in order, from raw rows. The recorded sighting, on a page
   whose WebSocket never opens, is described on the landmarks page as squared in order ("kept exactly as it lies").
6. The lens is named: /api/health, every sweep, every scientist tool reply (called as MCP tools, over HTTP and over
   the WebSocket) and every vault statement that reads words or writes an edit through the lens (NEAREST, DESCRIBE
   POLE, UPDATE; over MCP, HTTP and the WebSocket) carry "lens", fitted on the base model google/gemma-4-31B as the
   file's own provenance says, with a caution, while DESCRIBE and ROLLBACK reply as before; the instrument names the
   lens the server sent (page head, status line, footer, red line) and shows the server's own record of it in its
   apparatus.
7. A sighting of every word-piece (the CPU checks scripts/check_sampler_mirror.py, check_saved_prompts.py,
   check_sweep_protocol.py and check_capture_rows.py run first, beside the lens card): over the WebSocket, a stream
   with a request_id gives progress, the answer once, one pair per word-piece (read at P - 1 + k, reading
   answer[k - 1], having chosen answer[k], with the model's sliding window beside read_pos) and an end; its token 0
   is the one-pair reading byte for byte; a chat answer stops at its end of turn;
   a sampled answer echoes its seed and the same seed writes the same words; a retake of given ids reads the same
   states, and its recapture check has no logit diff (nothing was written to compare with); cancels (while writing,
   and of a queued request) and refusals stay on their own request; the thinking comparison rides on the first pair
   only and says how it ended; trace_readers on exact ids reads what the text did. Saved prompts over
   HTTP, in a temporary file (THEODOLITE_SAVED_PROMPTS is pointed away from the reader's own list).
8. The label gate (REPORT §3.3, conditions 2 and 3; scripts/check_label_gate.py runs first, on the CPU): /api/health,
   every sweep, inspect_pole, trace_downstream_readers, test_motif_hypothesis and the vault's NEAREST and DESCRIBE
   POLE carry "label_gate" just before "lens", and every neuron label they send a "label_check" by the study's
   measure, the same over MCP, HTTP and the WebSocket. At layer 59 (counting from 0) a face shares all ten and is
   shown ("one_reading"); below layer 40 it is held back ("below_layer_40"); the readers traced carry their naked-eye
   words and checks. Where the study's static caches are at hand (experiments/results/N1A_cache, gitignored),
   inspect_pole's count equals N1's own overlap for four of its neurons at each sampled layer of the band, both faces.
   On the pages, the leaf of layer 36 (the default) holds its neuron's words back: on the recorded sighting, the
   instrument's leaf and the layer's leaf show none of layer 36's words or glosses with the apparatus closed and show
   them with it open; on a live sighting the instrument's leaf says so; and the folio of neuron 15491 at layer 54
   gives the server's ten-a-side count on its seal ("k IN 10") and its words, as the server's check decides. The
   sweep's own counts and the traced readers' are read again, independently, through inspect_pole (every labelled
   face in layers 40-57 of the plain sweep; every reader returned, with its naked-eye words). Condition 3 on the pages:
   on the recorded sighting, layer 54's leaf leaves out the words of its faces that share none of ten
   (N1G_recorded_label_checks.json) with the apparatus closed and shows them with it open, and shows the others with
   their count; the vault's recorded search for "she" at layer 59 holds back the rows its recorded checks hold back;
   and on the live sighting, layer 55's leaf gives each labelled row the count inspect_pole reads for that neuron and
   face ("they share k of ten"), or holds it back where that check does.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("HF_HUB_OFFLINE", "1")
for _proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_proxy_var, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import httpx
import numpy as np
import torch
import uvicorn
import websockets

from backend.mcp_scientist import (
    inspect_pole,
    query_larql_vault,
    test_motif_hypothesis,
    trace_downstream_readers,
)
from backend.label_gate import BAND, LABEL_GATE_CARD, label_check
from backend.server import MAGIC_THEO, MAX_SWEEP_POSITIONS, WIRE_VERSION, app


async def ws_sweep(ws: Any, payload: Dict[str, Any]) -> tuple[bytes, Dict[str, Any]]:
    """One sweep over an open /ws/theodolite socket: the binary frame and the JSON metadata."""
    await ws.send(json.dumps({"action": "sweep", "payload": payload}))
    bin_msg = await ws.recv()
    meta = json.loads(await ws.recv())
    assert isinstance(bin_msg, bytes), f"expected a binary frame first, got {str(bin_msg)[:200]}"
    return bin_msg, meta


def frame_floats(n_filaments: int, think: bool = False) -> int:
    """The float32 count of one binary frame's payload (WIRE_VERSION 1), with the thinking path when it rides along."""
    return 60 * 3 + 60 * 3 + 60 * 4 + 60 * 4 + 60 * 60 + 60 * 15 * 6 + n_filaments * 7 + (60 * 3 if think else 0)


async def ws_collect(
    ws: Any, rids: List[str], first: List[Any] | None = None, on_end: Any = None
) -> Dict[str, Dict[str, Any]]:
    """Every frame of the streamed requests `rids` until each has its theodolite_sweep_end, sorted by request:
    progress, the answer written, the (binary, JSON) pairs, errors and the end, with "ended" giving the order the
    ends came in. A binary frame must be followed at once by its own metadata, and every JSON frame must name one of
    `rids`. `on_end(rid)` (async) is awaited as each end arrives."""
    got: Dict[str, Dict[str, Any]] = {
        r: {"progress": [], "written": None, "pairs": [], "errors": [], "end": None, "order": []} for r in rids
    }
    pending: bytes | None = None
    queue = list(first or [])
    ended: List[str] = []
    while any(g["end"] is None for g in got.values()):
        msg = queue.pop(0) if queue else await asyncio.wait_for(ws.recv(), timeout=900)
        if isinstance(msg, bytes):
            assert pending is None, "two binary frames in a row"
            pending = msg
            continue
        obj = json.loads(msg)
        rid = obj.get("request_id")
        assert rid in got, f"a frame for an unknown request: {str(obj)[:200]}"
        g = got[rid]
        g["order"].append(obj.get("type"))
        if pending is not None:
            assert obj.get("type") == "theodolite_metadata", f"a binary frame was followed by {obj.get('type')}"
            g["pairs"].append((pending, obj))
            pending = None
        elif obj["type"] == "theodolite_progress":
            g["progress"].append(obj)
        elif obj["type"] == "theodolite_written":
            assert g["written"] is None, "the answer came twice"
            g["written"] = obj
        elif obj["type"] == "error":
            g["errors"].append(obj)
        elif obj["type"] == "theodolite_sweep_end":
            g["end"] = obj
            ended.append(rid)
            if on_end is not None:
                await on_end(rid)
        else:
            raise AssertionError(f"an unexpected frame {obj.get('type')}")
    for g in got.values():
        g["ended"] = ended
    return got


async def ws_stream(ws: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """One streamed sweep (`payload` carries its request_id), collected to its end."""
    await ws.send(json.dumps({"action": "sweep", "payload": payload}))
    return (await ws_collect(ws, [payload["request_id"]]))[payload["request_id"]]


async def check_answer_streams(ws: Any, legacy_bin: bytes, legacy_meta: Dict[str, Any]) -> None:
    """A sighting of every word-piece, over the WebSocket (see the module docstring, item 7)."""
    prompt = "The number of legs on the animal that spins webs is"
    # the one-pair reply carries the additive fields too, read at token 0
    assert legacy_meta["request_id"] is None and legacy_meta["token_index"] == 0 and legacy_meta["last"] is True
    assert legacy_meta["n_to_sight"] == 1 and legacy_meta["read_pos"] == len(legacy_meta["prompt_token_ids"]) - 1
    # the model's sliding window rides beside read_pos (text_config.sliding_window; the lens card's caution names it)
    assert legacy_meta["sliding_window"] == 1024, legacy_meta.get("sliding_window")
    assert legacy_meta["answer"]["text"] == legacy_meta["generated_text"] and legacy_meta["generation"]["greedy"] is True
    assert all("jpp_top_ids" in p and "naive_top_id" in p and "tracked_from_next" in p for p in legacy_meta["layer_plumes"])

    every = await ws_stream(ws, {"request_id": "e2e-every", "prompt": prompt, "max_new_tokens": 6, "sight_every_token": True})
    w, end, pairs = every["written"], every["end"], every["pairs"]
    ans = w["answer"]
    n = ans["n_written"]
    assert n >= 3, f"the default prompt wrote only {ans['token_strs']}"
    assert not every["errors"] and end["error"] is None and not end["cancelled"], every["errors"] or end
    assert [p["done"] for p in every["progress"] if p["phase"] == "writing"] == list(range(1, n + 1))
    assert every["order"].index("theodolite_written") < every["order"].index("theodolite_metadata")
    assert w["sight_order"] == list(range(n)) and w["n_to_sight"] == n == end["n_sent"] == len(pairs)
    P = w["prompt_len"]
    ids = w["prompt_token_ids"] + ans["token_ids"]
    assert ans["token_ids"] == legacy_meta["answer"]["token_ids"], "the stream wrote other words than the one-pair reply"
    mismatch = []
    for k, (b, m) in enumerate(pairs):
        magic, version, _nl, _np, n_fil, flags = struct.unpack("<6I", b[:24])
        assert magic == MAGIC_THEO and version == WIRE_VERSION and flags & 4 and not flags & 1, (k, flags)
        assert len(b) == 24 + 4 * frame_floats(n_fil), f"pair {k}: {len(b)} bytes"
        assert m["token_index"] == k and m["sight_index"] == k and m["last"] == (k == n - 1)
        assert m["read_pos"] == P - 1 + k and m["read_token_id"] == ids[P - 1 + k] and m["kinship_n_states"] == P + k
        assert m["sliding_window"] == legacy_meta["sliding_window"], (k, m.get("sliding_window"))
        assert m["chosen"]["token_id"] == ans["token_ids"][k] and m["chosen"]["source"] == "written"
        # greedy with no penalty writes the step's likeliest piece
        assert m["chosen"]["model_rank"] == 1 and m["chosen"]["sampler_prob"] is None, m["chosen"]
        rc = m["recapture_check"]
        assert abs(rc["chosen_logit_diff"]) < 0.5, f"pair {k}: the re-read and the writing disagree by {rc['chosen_logit_diff']}"
        if not rc["top1_is_chosen"]:
            mismatch.append((k, rc["top1_token"], m["chosen"]["token"], rc["chosen_logit_diff"]))
        assert m["frame"]["centre_kind"] == "none" and m["prompt_token_ids"] == w["prompt_token_ids"]
    assert len(mismatch) <= 1, f"the layer-60 re-read's top piece is not the piece written at {mismatch}"
    # token 0 of the stream is the one-pair reading, float for float (only flags bit 2 differs)
    assert pairs[0][0][24:] == legacy_bin[24:], "the stream's token 0 differs from the one-pair reading"
    blank_first = ans["token_strs"][0].strip() == "" and n >= 2
    merged = sum(v for p in pairs[0][1]["layer_plumes"] for v in p["tracked_from_next"].values())
    print(f"  • every word-piece: wrote {ans['token_strs']} ({ans['stop_reason']}), {n} pairs, read at P-1+k; token 0 = "
          f"the one-pair reading; space-then-digit rule {'took ' + str(merged) + ' ranks from the next piece' if blank_first else 'not needed'}"
          + (f"; re-read top-1 differs at {mismatch}" if mismatch else ""))

    retake = await ws_stream(ws, {"request_id": "e2e-retake", "prompt": prompt, "answer_token_ids": ans["token_ids"],
                                  "sight_tokens": [2, 0, 99], "max_new_tokens": 6})
    rw = retake["written"]
    assert rw["answer"]["source"] == "given" and rw["answer"]["stop_reason"] == "given" and rw["sight_order"] == [2, 0]
    assert [m["token_index"] for _b, m in retake["pairs"]] == [2, 0]
    assert all(m["sight_dropped"] == [99] for _b, m in retake["pairs"])
    assert retake["pairs"][0][0][24:] == pairs[2][0][24:], "a retake of the same ids reads other states"
    # a given piece's odds are the re-read's own layer-60 readout, so there is no writing to compare with: no diff
    rrc = retake["pairs"][0][1]["recapture_check"]
    assert rrc["chosen_logit_diff"] is None and rrc["chosen_logit_recapture"] is None, rrc
    assert retake["pairs"][0][1]["chosen"]["source"] == "given"
    given_logit = retake["pairs"][0][1]["chosen"]["chosen_logit"]
    read_logit = pairs[2][1]["recapture_check"]["chosen_logit_recapture"]
    assert abs(given_logit - read_logit) < 1e-3, f"a given piece's logit {given_logit} is not the layer-60 re-read's {read_logit}"
    print("  • a retake of the given ids reads the same states (word-piece 2, byte for byte); 99 dropped and reported")

    turn = await ws_stream(ws, {"request_id": "e2e-turn", "prompt": "Say only the word yes.", "chat": True,
                                "max_new_tokens": 32, "sight_tokens": [0]})
    ta = turn["written"]["answer"]
    assert ta["stop_reason"] == "end_mark" and ta["token_ids"][-1] in ta["stop_token_ids"] and ta["steps"][-1]["end_mark"], ta
    assert 106 in ta["stop_token_ids"] and len(turn["pairs"]) == 1
    print(f"  • a chat answer stops at its end of turn: {ta['token_strs']} ({ta['n_written']} of 32)")

    draw = {"request_id": "e2e-draw", "prompt": prompt, "max_new_tokens": 8, "sight_tokens": [0],
            "sampler": {"temperature": 1.0, "top_k": 64, "top_p": 0.95}}
    d1 = await ws_stream(ws, draw)
    g1 = d1["written"]["generation"]
    assert g1["greedy"] is False and g1["seed_drawn"] is True and isinstance(g1["seed"], int)
    assert g1["settings"] == {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "presence_penalty": 0.0}, g1["settings"]
    for st in d1["written"]["answer"]["steps"]:
        assert 0.0 < st["sampler_prob"] <= 1.0 and 1 <= st["sampler_support"], st
    d2 = await ws_stream(ws, {**draw, "request_id": "e2e-redraw", "sampler": {**draw["sampler"], "seed": g1["seed"]}})
    assert d2["written"]["generation"]["seed"] == g1["seed"] and d2["written"]["generation"]["seed_drawn"] is False
    assert d2["written"]["answer"]["token_ids"] == d1["written"]["answer"]["token_ids"], "the same seed wrote other words"
    print(f"  • drawn at random (seed {g1['seed']}): {d1['written']['answer']['token_strs']}; the same seed wrote the same")

    think = await ws_stream(ws, {"request_id": "e2e-think", "prompt": prompt, "max_new_tokens": 3,
                                 "sight_every_token": True, "compare_think": True})
    tp = think["pairs"]
    assert len(tp) == think["written"]["n_to_sight"] >= 2
    f0 = struct.unpack("<6I", tp[0][0][:24])
    assert f0[5] & 1 and len(tp[0][0]) == 24 + 4 * frame_floats(f0[4], think=True)
    assert all(not struct.unpack("<6I", b[:24])[5] & 1 and m["think_generated_text"] is None and m["think_seed"] is None
               for b, m in tp[1:])
    assert tp[0][1]["think_generated_text"] and tp[0][1]["think_read_token"] is not None
    # left alone, the thought runs to its length or its end mark, and says which
    assert tp[0][1]["think_stop_reason"] in ("length", "end_mark") and tp[0][1]["think_skipped"] is False, tp[0][1]["think_stop_reason"]
    assert all(m["think_stop_reason"] is None and m["think_skipped"] is False for _b, m in tp[1:])
    assert think["end"]["cancelled"] is False
    print(f"  • the thinking comparison rides on the first pair only (it read {tp[0][1]['think_read_token']!r}, "
          f"ended by {tp[0][1]['think_stop_reason']})")

    # cancels: while writing; and a queued request, dropped at once while another runs
    await ws.send(json.dumps({"action": "sweep", "payload": {"request_id": "e2e-cancel", "prompt": prompt,
                                                             "max_new_tokens": 64, "sight_every_token": True}}))
    first = await asyncio.wait_for(ws.recv(), timeout=600)
    await ws.send(json.dumps({"action": "cancel_sweep", "request_id": "e2e-cancel"}))
    c = (await ws_collect(ws, ["e2e-cancel"], [first]))["e2e-cancel"]
    assert c["end"]["cancelled"] is True and c["end"]["n_sent"] == len(c["pairs"]) <= 1, c["end"]
    assert c["written"]["answer"]["stop_reason"] in ("stopped", "end_mark") and c["written"]["n_to_sight"] == 1
    await ws.send(json.dumps({"action": "sweep", "payload": {"request_id": "e2e-long", "prompt": prompt,
                                                             "max_new_tokens": 64, "sight_every_token": True}}))
    await ws.send(json.dumps({"action": "sweep", "payload": {"request_id": "e2e-queued", "prompt": prompt, "max_new_tokens": 2}}))
    await ws.send(json.dumps({"action": "cancel_sweep", "request_id": "e2e-queued"}))

    async def then_stop_the_long_one(rid: str) -> None:
        if rid == "e2e-queued":
            await ws.send(json.dumps({"action": "cancel_sweep", "request_id": "e2e-long"}))

    both = await ws_collect(ws, ["e2e-long", "e2e-queued"], on_end=then_stop_the_long_one)
    qend, lg = both["e2e-queued"]["end"], both["e2e-long"]
    assert qend == {"type": "theodolite_sweep_end", "request_id": "e2e-queued", "n_sent": 0, "n_to_sight": 0,
                    "cancelled": True, "error": None}, qend
    assert both["e2e-queued"]["order"] == ["theodolite_sweep_end"], "the cancelled queued request ran"
    assert lg["ended"] == ["e2e-queued", "e2e-long"], f"the queued cancel waited for the running request: {lg['ended']}"
    assert lg["end"]["cancelled"] is True
    print(f"  • cancels: while writing ({c['written']['answer']['n_written']} written, {c['end']['n_sent']} sighted); "
          f"a queued request dropped at once; the running one stopped ({lg['end']['n_sent']} sent)")

    refused = await ws_stream(ws, {"request_id": "e2e-refuse", "max_new_tokens": 300})
    assert refused["errors"] and refused["errors"][0]["recoverable"] and refused["end"]["error"] and refused["end"]["n_sent"] == 0
    # one "word" per word-piece, so with the start mark and the answer this is past the limit
    too_long = await ws_stream(ws, {"request_id": "e2e-too-long", "prompt": "word " * MAX_SWEEP_POSITIONS})
    assert "one sweep can hold" in too_long["errors"][0]["message"], too_long["errors"]
    b_after, m_after = await ws_sweep(ws, {"prompt": prompt, "max_new_tokens": 6})
    assert m_after["token_index"] == 0 and b_after[24:] == legacy_bin[24:], "the socket after the refusals"
    print("  • refusals stay on their request (error, then its end), in plain words; the socket still answers")


def assert_base_lens(lens: Any, model_id: str | None = None) -> None:
    """The server's "lens": the J++ lens fitted on the base model google/gemma-4-31B (as the file's own provenance
    says, not a fallback), applied to the served model, with a caution."""
    assert isinstance(lens, dict), f"no lens object: {lens!r}"
    assert lens["fitted_on"].split(" (")[0] == "google/gemma-4-31B", f"lens.fitted_on: {lens['fitted_on']!r}"
    assert "base" in lens["fitted_on"] and lens["id"] == "jpp-base" and lens["name"] == "J++", lens
    prov = lens.get("provenance") or {}
    assert prov.get("hf_model_name") == "google/gemma-4-31B" and prov.get("method") == "jpp", f"lens.provenance: {prov}"
    assert lens["method"] and lens["fit_data"] and lens["source"] and lens["file"] and lens["caution"], lens
    if model_id is not None:
        assert lens["applied_to"] == model_id, f"lens.applied_to {lens['applied_to']!r} != {model_id!r}"


def lens_phrase(lens: Dict[str, Any], named: bool = False) -> str:
    """What the pages print for a lens card (frontend/src/lib/lens.ts, lensPhrase), computed from the server's card."""
    if lens["id"] == "jpp-base" or "(base" in lens["fitted_on"]:
        model = f" {lens['fitted_on'].split(' (')[0]}" if named else ""
        return f"the {lens['name']} lens, fitted on the base model{model}"
    return f"the {lens['name']} lens, fitted on {lens['fitted_on']}"


LABEL_CHECK_KEYS = ["shared_top10", "held_back", "reasons", "band", "sampled_layer", "in_measured_band",
                    "lens_validated", "edited"]


def assert_label_check(c: Any, layer: int, lens_id: str | None, what: str, edited: bool = False) -> Dict[str, Any]:
    """A label's `label_check` (backend/label_gate.py) at code layer `layer`: its keys in order, and every field but the
    count what the rule gives for that count (two id lists sharing exactly `shared_top10`)."""
    assert isinstance(c, dict) and list(c) == LABEL_CHECK_KEYS, f"{what}: label_check {c!r}"
    k = c["shared_top10"]
    assert k is None or (isinstance(k, int) and 0 <= k <= 10), f"{what}: shared_top10 {k!r}"
    lens_ids, eye_ids = (list(range(10)), list(range(10 - k, 20 - k))) if k is not None else (None, None)
    want = label_check(layer, lens_ids, eye_ids, lens_id, edited=edited, silenced="writes_nothing" in c["reasons"])
    assert c == want, f"{what}: label_check {c} is not the rule's {want}"
    return c


def wp(token: str) -> str:
    """A word-piece as the pages print it (frontend/src/lib/words.ts wp): between curly quotes, its leading space
    dropped, a new line as ↵."""
    t = token.lstrip()
    if t == "":
        return "“␣”" if token else "“”"
    return "“" + t.replace("\n", "↵") + "”"


# Set one of the "How to read these pages" switches ("rubrics", "apparatus") on or off, whatever it was: (name, on)
SET_READING_JS = """((name, on) => {
    const b = Array.from(document.querySelectorAll('[aria-label="How to read these pages"] button'))
        .find((x) => x.textContent.includes(name));
    if (!b) return null;
    if ((b.getAttribute("aria-pressed") === "true") !== on) b.click();
    return true;
})"""


def study_cache() -> Path | None:
    """The study's static caches (experiments/results/N1A_cache, gitignored): here, or in the main checkout of the
    worktree this is."""
    here = ROOT_DIR / "experiments" / "results" / "N1A_cache"
    if (here / "static_L40.pt").exists():
        return here
    try:
        common = subprocess.run(["git", "-C", str(ROOT_DIR), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except Exception:
        return None
    main = Path(common).parent / "experiments" / "results" / "N1A_cache"
    return main if (main / "static_L40.pt").exists() else None


# The words and glosses of layer 36 (index 35) in the recorded sighting (frontend/src/fixtures/e6-demo-sighting.json,
# poleLabels "35:0" to "35:4", lens and eye; the glosses from screens/instrument/recorded.ts and
# screens/layerLeaf/recorded.ts). Only pieces that cannot turn up elsewhere on those pages are listed, as the pages
# print them (wp: between curly quotes, the leading space dropped).
L36_RECORDED_WORDS = ["“myself”", "“मुझ”", "“的我”", "“líng”", "“JPY”", "“ücret”", "“青少年”", "“年龄”", "“использовать”",
                      "“creator”", "“自身的”", "“自身”", "“আমাকে”", "“sayesinde”", "“੦”", "“précision”", "“scept”", "“creato”"]
# Layer 54 (index 53) of the recorded sighting: its labelled faces, their words (lens and naked eye), and the recorded
# check of each (N1G_recorded_label_checks.json), for condition 3 on the pages; and the vault's recorded search for
# "she" at layer 59 (index 58): how many of its first five rows (the board's default) its recorded checks hold back
_FIXTURE = json.loads((ROOT_DIR / "frontend" / "src" / "fixtures" / "e6-demo-sighting.json").read_text())
_N1G = json.loads((ROOT_DIR / "experiments" / "results" / "N1G_recorded_label_checks.json").read_text())
L54_FACES = {
    k: {"words": [t for t, _ in _FIXTURE["poleLabels"]["lens"][k]] + [t for t, _ in _FIXTURE["poleLabels"]["eye"][k]],
        "check": _N1G["fixture"]["faces"][k]["label_check"]}
    for k in sorted(_FIXTURE["poleLabels"]["lens"]) if k.startswith("53:")
}
SHE59_HELD = sum(r["label_check"]["held_back"] for r in _N1G["vault"]["searches"]["she@58"]["rows"][:5])

L36_RECORDED_GLOSSES = ["one idea in three tongues", "Hindi “me”", "Chinese “one’s own”", "Chinese “teenagers”",
                        "Turkish “thanks to”", "No word in common, yet both eyes speak of the self", "Italian “created”"]


ARTIFACT_DIR = Path(os.environ.get("THEODOLITE_ARTIFACT_DIR", str(ROOT_DIR / "artifacts" / "e2e")))
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
HOST = "127.0.0.1"
PORT = 8765
CDP_PORT = 9229


class CDPClient:
    """Minimal async Chrome DevTools Protocol client over WebSocket."""

    def __init__(self, ws_url: str) -> None:
        self.ws_url = ws_url
        self.ws: Any = None
        self.msg_id = 0
        self.pending: Dict[int, asyncio.Future] = {}
        self.console_errors: List[str] = []
        self.console_logs: List[str] = []
        self.page_exceptions: List[str] = []
        self._reader_task: Any = None

    async def connect(self) -> None:
        self.ws = await websockets.connect(self.ws_url, max_size=50 * 1024 * 1024)
        self._reader_task = asyncio.create_task(self._reader_loop())
        await self.send("Runtime.enable")
        await self.send("Page.enable")
        await self.send("Log.enable")

    async def _reader_loop(self) -> None:
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                if "id" in msg:
                    fut = self.pending.pop(msg["id"], None)
                    if fut and not fut.done():
                        fut.set_result(msg)
                else:
                    method = msg.get("method", "")
                    params = msg.get("params", {})
                    if method == "Runtime.consoleAPICalled":
                        c_type = params.get("type", "log")
                        args = params.get("args", [])
                        text = " ".join(str(a.get("value", a.get("description", ""))) for a in args)
                        self.console_logs.append(f"[{c_type}] {text}")
                        if c_type == "error":
                            self.console_errors.append(text)
                    elif method == "Runtime.exceptionThrown":
                        details = params.get("exceptionDetails", {})
                        text = details.get("text", "") + " " + str(details.get("exception", {}).get("description", ""))
                        self.page_exceptions.append(text.strip())
                    elif method == "Log.entryAdded":
                        entry = params.get("entry", {})
                        if entry.get("level") == "error":
                            text = entry.get("text", "")
                            if "favicon.ico" not in text:
                                self.console_errors.append(f"[Log.error] {text}")
        except Exception:
            pass

    async def send(self, method: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        self.msg_id += 1
        mid = self.msg_id
        fut = asyncio.get_running_loop().create_future()
        self.pending[mid] = fut
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        res = await asyncio.wait_for(fut, timeout=30.0)
        if "error" in res:
            raise RuntimeError(f"CDP error in {method}: {res['error']}")
        return res.get("result", {})

    async def eval_js(self, expression: str, await_promise: bool = True) -> Any:
        res = await self.send(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": await_promise,
            },
        )
        if "exceptionDetails" in res:
            raise RuntimeError(f"JS evaluation error: {res['exceptionDetails']}")
        return res.get("result", {}).get("value")

    async def screenshot(self, out_path: Path) -> None:
        res = await self.send("Page.captureScreenshot", {"format": "png"})
        png_bytes = base64.b64decode(res["data"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(png_bytes)

    async def close(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
        if self.ws:
            await self.ws.close()


async def main() -> int:
    print("=" * 92)
    print("LOGOS THEODOLITE — PHASE 2 END-TO-END SERVER, FASTMCP & 3D FRONTEND VERIFICATION")
    print("=" * 92)

    # The lens card's branches for other files never run below (the server loads the shipped file), so they are
    # checked first, on the CPU, in a process of their own that loads neither the model nor the lens.
    print("[0/4] Checking the lens card on the CPU (scripts/check_lens_card.py)...")
    lens_check = subprocess.run(
        [sys.executable, "-I", str(ROOT_DIR / "scripts" / "check_lens_card.py")],
        capture_output=True, text=True, timeout=300, check=False,
    )
    print("  • " + (lens_check.stdout.strip().splitlines() or ["(no output)"])[-1])
    assert lens_check.returncode == 0, f"the lens card check failed:\n{lens_check.stdout}\n{lens_check.stderr}"
    # the sampler's mirrored odds, the saved-prompts routes, the sweep's wire protocol, and capture_rows against
    # interp_engine's capture bit for bit on a tiny stand-in Gemma 4, on the CPU too
    # and first, what the interface sends by default (its store, bundled with esbuild and run under node)
    # and the label gate's rule, the server's measure against the study's caches, and the results the pages quote
    for cpu_check in ("check_interface_defaults.py", "check_sampler_mirror.py", "check_saved_prompts.py",
                      "check_sweep_protocol.py", "check_capture_rows.py", "check_label_gate.py"):
        res = subprocess.run(
            [sys.executable, "-I", str(ROOT_DIR / "scripts" / cpu_check)],
            capture_output=True, text=True, timeout=600, check=False,
        )
        print("  • " + (res.stdout.strip().splitlines() or ["(no output)"])[-1])
        assert res.returncode == 0, f"{cpu_check} failed:\n{res.stdout}\n{res.stderr}"
    # the saved prompts the pages read and write here go to a temporary file, never the reader's own list
    saved_dir = Path(tempfile.mkdtemp(prefix="theodolite_e2e_saved_"))
    os.environ["THEODOLITE_SAVED_PROMPTS"] = str(saved_dir / "saved_prompts.json")

    config = uvicorn.Config(app, host=HOST, port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    # Wait for server startup (loads Gemma-4-31B + J++ Lens via lifespan)
    async with httpx.AsyncClient(base_url=f"http://{HOST}:{PORT}", timeout=60.0, trust_env=False) as client:
        health: Dict[str, Any] = {}
        for _ in range(120):
            try:
                r = await client.get("/api/health")
                if r.status_code == 200:
                    health = r.json()
                    break
            except Exception:
                pass
            await asyncio.sleep(0.5)
        else:
            raise RuntimeError("FastAPI server did not become healthy within 60 seconds.")
        print(f"[1/4] FastAPI Server Healthy: model={health['model_id']} | VRAM={health['vram_allocated_gb']:.2f} GB")
        # the model's sliding window, and what the CUDA allocator runs with ("" for its defaults)
        assert health.get("sliding_window") == 1024, f"health sliding_window {health.get('sliding_window')!r}"
        assert isinstance(health.get("allocator"), str), f"health allocator {health.get('allocator')!r}"
        print(f"  • sliding window {health['sliding_window']} | allocator: {health['allocator'] or '(default)'}")
        # checked once the server answers, so a wrong or missing lens fails here, by name
        assert_base_lens(health.get("lens"), health["model_id"])
        lens = health["lens"]
        print(f"  • lens: {lens['name']}, fitted on {lens['fitted_on']} ({lens['source']})")
        # the label gate's rule rides just before the lens (backend/label_gate.py)
        hk = list(health)
        assert health.get("label_gate") == LABEL_GATE_CARD and hk.index("label_gate") == hk.index("lens") - 1, hk
        print(f"  • label gate: {LABEL_GATE_CARD['rule']}, just before the lens")

        # -------------------------------------------------------------------------
        # Step 1: Verify FastMCP Autonomous Circuit Scientist Tools
        # -------------------------------------------------------------------------
        print("\n[2/4] Verifying FastMCP Scientist Tools (`backend/mcp_scientist.py`)...")
        # Tool 1: inspect_pole on Layer 53 (1-based L54) Neuron 16383 (- South Pole)
        pole_res = inspect_pole(layer=53, neuron_idx=16383, polarity=-1.0)
        south_toks = [t["token"] for t in pole_res["south_pole"]["j_vindex_tokens"][:4]]
        north_toks = [t["token"] for t in pole_res["north_pole"]["j_vindex_tokens"][:4]]
        gate_toks = [t["token"] for t in pole_res["gate_trigger_tokens"][:4]]
        print(f"  • inspect_pole(L53, N16383, -1.0):")
        print(f"      South Pole (-) J-VINDEX : {south_toks}")
        print(f"      North Pole (+) J-VINDEX : {north_toks}")
        print(f"      Upstream Gate Triggers  : {gate_toks}")
        assert len(south_toks) > 0 and any("Sp" in t or "sp" in t.lower() for t in south_toks)
        # the MCP tools name the lens their words went through, as /api/health does
        assert_base_lens(pole_res.get("lens"), health["model_id"])
        assert pole_res["lens"] == lens, "inspect_pole and /api/health name different lenses"
        # each face carries its label check, and the reply the rule, just before the lens; layer 53 lies in the band
        assert list(pole_res)[-2:] == ["label_gate", "lens"] and pole_res["label_gate"] == LABEL_GATE_CARD, list(pole_res)
        for face in ("north_pole", "south_pole"):
            assert_label_check(pole_res[face].get("label_check"), 53, lens["id"], f"inspect_pole(53, 16383) {face}")
        assert pole_res["active_pole"]["label_check"] == pole_res["south_pole"]["label_check"]
        print(f"  • inspect_pole(L53, N16383): shares {pole_res['north_pole']['label_check']['shared_top10']} (+) and "
              f"{pole_res['south_pole']['label_check']['shared_top10']} (-) of ten with the naked eye")
        # at layer 59 the lens is the identity: the two readings are one, so they share all ten and are shown
        top_pole = inspect_pole(layer=59, neuron_idx=12584, polarity=1.0)
        for face in ("north_pole", "south_pole"):
            c = assert_label_check(top_pole[face]["label_check"], 59, lens["id"], f"inspect_pole(59, 12584) {face}")
            assert c["shared_top10"] == 10 and c["band"] == "one_reading" and not c["held_back"], c
        # below layer 40 a label is held back by the layer, whatever it shares
        low_pole = inspect_pole(layer=35, neuron_idx=21189, polarity=-1.0)
        for face in ("north_pole", "south_pole"):
            c = assert_label_check(low_pole[face]["label_check"], 35, lens["id"], f"inspect_pole(35, 21189) {face}")
            assert c["held_back"] and "below_layer_40" in c["reasons"] and c["band"] == "not_sampled", c
        print("  • inspect_pole: layer 59 shares ten of ten and is shown; layer 35 is held back below layer 40")
        # the folio's neuron, 15491 at layer 54 (index 53), positive face: its check decides what the page shows
        folio_face = inspect_pole(layer=53, neuron_idx=15491, polarity=1.0)["north_pole"]
        folio_check = assert_label_check(folio_face["label_check"], 53, lens["id"], "inspect_pole(53, 15491) north_pole")
        folio_word = wp(folio_face["j_vindex_tokens"][0]["token"])
        # the server's count is the study's: N1's own overlap (its static caches) for four neurons per sampled layer
        cache = study_cache()
        if cache is None:
            print("  • N1 parity skipped: the study's static caches (experiments/results/N1A_cache, gitignored) are not here")
        else:
            neu = json.loads((ROOT_DIR / "experiments" / "results" / "N1_neurons.json").read_text())
            n_faces, differ = 0, []
            for l in [int(x) for x in neu["layers"] if BAND[0] <= int(x) <= BAND[1]]:
                st = torch.load(cache / f"static_L{l}.pt", weights_only=True)
                for n in neu["neurons"][str(l)][:4]:
                    rep_n = inspect_pole(layer=l, neuron_idx=n, polarity=1.0)
                    ni = st["neurons"].index(n)
                    for face, s in (("north_pole", 1.0), ("south_pole", -1.0)):
                        study = len(set((s * st["jpp"][ni].float()).topk(10).indices.tolist())
                                    & set((s * st["naive"][ni].float()).topk(10).indices.tolist()))
                        got = assert_label_check(rep_n[face]["label_check"], l, lens["id"], f"N1 L{l} n{n} {face}")
                        n_faces += 1
                        if got["shared_top10"] != study:
                            differ.append((l, n, face, got["shared_top10"], study))
            assert not differ, f"inspect_pole's counts differ from N1's overlap (layer, neuron, face, ours, N1's): {differ}"
            print(f"  • N1 parity: inspect_pole's ten-a-side count equals N1's overlap on all {n_faces} faces checked")

        # Tool 2: trace_downstream_readers on Layer 35 (1-based L36) Neuron 9482
        readers_res = await trace_downstream_readers(
            layer=35,
            neuron_idx=9482,
            prompt="The number of legs on the animal that spins webs is",
            max_layer_hop=6,
            top_k=5,
        )
        top_readers = readers_res["top_downstream_readers"]
        print(
            f"  • trace_downstream_readers(L35, N9482) -> {len(top_readers)} verified readers "
            f"(top reader: L{top_readers[0]['dst_layer']}:N{top_readers[0]['dst_neuron_idx']} "
            f"[{top_readers[0]['dst_polarity']}] C={top_readers[0]['coupling_score']:.2f})"
        )
        assert len(top_readers) > 0
        assert_base_lens(readers_res.get("lens"), health["model_id"])
        assert readers_res["lens"] == lens, "trace_downstream_readers and /api/health name different lenses"
        # each reader returned carries the naked eye's words and its own label check; the rule rides before the lens
        assert list(readers_res)[-2:] == ["label_gate", "lens"] and readers_res["label_gate"] == LABEL_GATE_CARD
        for r in top_readers:
            assert len(r["dst_naive_tokens"]) == 4 and all({"token", "logit"} <= set(t) for t in r["dst_naive_tokens"]), r
            assert_label_check(r["dst_label_check"], r["dst_layer"], lens["id"], f"reader L{r['dst_layer']}:N{r['dst_neuron_idx']}")
        print(f"  • the readers traced carry their naked-eye words and label checks "
              f"({sum(r['dst_label_check']['held_back'] for r in top_readers)} of {len(top_readers)} held back)")
        # the readers' checks are computed on their own path (mcp_scientist.py, through an id map after the sort): each
        # must be what inspect_pole reads for that neuron and face, and its naked-eye words inspect_pole's first four
        for r in top_readers:
            face = "north_pole" if r["dst_polarity"] == "+" else "south_pole"
            again = inspect_pole(layer=r["dst_layer"], neuron_idx=r["dst_neuron_idx"],
                                 polarity=1.0 if r["dst_polarity"] == "+" else -1.0)[face]
            what = f"reader L{r['dst_layer']}:N{r['dst_neuron_idx']}{r['dst_polarity']}"
            assert again["label_check"] == r["dst_label_check"], (
                f"{what}: {r['dst_label_check']} != inspect_pole's {again['label_check']}")
            # the naked eye's logits come out of a bf16 GEMM (theodolite_math.py unembed_gemma4), so word-pieces often
            # tie exactly, and the two paths' top-k may break a tie either way (seen: L41:N4722-, its fourth word). So
            # the logits must agree rank by rank (the reader rounds to 2 places, inspect_pole to 3), and each word must
            # be inspect_pole's at that rank or, where it ties, one of inspect_pole's eight at the same logit
            eye_r, eye_i = r["dst_naive_tokens"], again["naive_tokens"]
            for i, t in enumerate(eye_r):
                assert abs(t["logit"] - eye_i[i]["logit"]) <= 0.006, (
                    f"{what}: naked-eye logits {eye_r} are not inspect_pole's {eye_i[:4]}")
                assert t["token"] == eye_i[i]["token"] or any(
                    u["token"] == t["token"] and abs(u["logit"] - t["logit"]) <= 0.006 for u in eye_i), (
                    f"{what}: naked-eye words {eye_r} are not inspect_pole's {eye_i[:4]}, even allowing for ties")
        print(f"  • the readers' checks and naked-eye words (ties aside) equal inspect_pole's own reading, all {len(top_readers)}")

        # Tool 3: test_motif_hypothesis on Layer 53 Neuron 16383 (Spider/Arachnid South-Pole neuron)
        hyp_res = await test_motif_hypothesis(
            layer=53,
            neuron_idx=16383,
            pos_prompts=[
                "The animal that spins silk webs to catch insects is the",
                "A black widow or tarantula is a type of",
                "The eight-legged arachnid spinning a web in the corner is a",
            ],
            neg_prompts=[
                "The capital city of France near the Seine river is",
                "In linear algebra, the eigenvalues of a symmetric matrix are",
                "To bake sourdough bread, mix flour, water, salt, and",
            ],
        )
        print(
            f"  • test_motif_hypothesis(L53, N16383): verdict={hyp_res['verdict']} | "
            f"Cohen's d={hyp_res['cohens_d']:.3f} | "
            f"mean_pos={hyp_res['mean_pos_signed_act']:.3f} vs mean_neg={hyp_res['mean_neg_signed_act']:.3f}"
        )
        assert abs(hyp_res["cohens_d"]) > 0.8, f"Expected strong selectivity, got {hyp_res['cohens_d']}"
        assert_base_lens(hyp_res.get("lens"), health["model_id"])
        assert hyp_res["lens"] == lens, "test_motif_hypothesis and /api/health name different lenses"
        assert list(hyp_res)[-2:] == ["label_gate", "lens"] and hyp_res["label_gate"] == LABEL_GATE_CARD
        assert_label_check(hyp_res.get("pole_label_check"), 53, lens["id"], "test_motif_hypothesis pole_label_check")
        print(f"  • the three MCP tools name the lens: {lens['name']}, fitted on {lens['fitted_on']}")

        # -------------------------------------------------------------------------
        # Step 2: Verify Binary Float32 WebSocket (`/ws/theodolite`) Wire Contract
        # -------------------------------------------------------------------------
        print("\n[3/4] Verifying Binary Float32 WebSocket (`/ws/theodolite`) Wire Contract...")
        async with websockets.connect(
            f"ws://{HOST}:{PORT}/ws/theodolite",
            max_size=50 * 1024 * 1024,
        ) as ws:
            await ws.send(
                json.dumps(
                    {
                        "action": "sweep",
                        "payload": {
                            "prompt": "The number of legs on the animal that spins webs is",
                            "monuments": ["spider", "eight", "insect"],
                            "survey_mode": "monument",
                            "compare_think": False,
                            "max_new_tokens": 6,
                        },
                    }
                )
            )
            bin_msg = await ws.recv()
            json_msg = json.loads(await ws.recv())

            assert isinstance(bin_msg, bytes), "First WebSocket frame must be binary ArrayBuffer"
            magic, version, n_layers, n_poles, n_filaments, flags = struct.unpack("<6I", bin_msg[:24])
            assert magic == MAGIC_THEO, f"Invalid magic header: {hex(magic)}"
            assert version == WIRE_VERSION
            assert n_layers == 60 and n_poles == 15
            expected_floats = (
                60 * 3  # traj_xyz
                + 60 * 3  # traj_angles_norm
                + 60 * 4  # delta_attn_xyz
                + 60 * 4  # delta_ffn_xyz
                + 60 * 60  # cka_matrix
                + 60 * 15 * 6  # poles_data
                + n_filaments * 7  # filaments_data
            )
            expected_bytes = 24 + expected_floats * 4
            assert len(bin_msg) == expected_bytes, f"Byte size mismatch: {len(bin_msg)} != {expected_bytes}"
            f32_arr = np.frombuffer(bin_msg, dtype=np.float32, offset=24)
            assert np.isfinite(f32_arr).all(), "Binary Float32 payload contains non-finite values!"
            print(
                f"  • Binary Frame Verified: {len(bin_msg):,} bytes | magic=0x{magic:08X} | "
                f"layers={n_layers} | poles={n_poles}/layer | filaments={n_filaments} | finite=True"
            )
            print(
                f"  • JSON Frame Verified  : generated={json_msg['generated_text']!r} | "
                f"elapsed={json_msg['elapsed_ms']}ms | peak_vram={json_msg['peak_vram_gb']} GB"
            )
            # the defaults keep their meaning: bare text, the frame as built, C, |a|
            assert json_msg["chat"] is False and json_msg["prompt"] == "The number of legs on the animal that spins webs is"
            assert json_msg["frame"]["squaring"] == "gram_schmidt" and json_msg["thread_score"] == "coupling"
            assert json_msg["pole_ranking"] == "activation"
            assert json_msg["prompt_token_ids"][0] == 2 and json_msg["prompt_token_ids"][1] != 2
            assert all(0.0 < p["proj_frac"] <= 1.0 for p in json_msg["layer_plumes"])
            # the sweep names the lens its readings went through, as /api/health does
            assert_base_lens(json_msg.get("lens"), health["model_id"])
            assert json_msg["lens"] == lens, "the sweep and /api/health name different lenses"
            # every label the sweep sends carries its check (null where no words were read), and the rule rides just
            # before the lens; old keys of each entry unchanged
            mk = list(json_msg)
            assert json_msg["label_gate"] == LABEL_GATE_CARD and mk.index("label_gate") == mk.index("lens") - 1, mk
            n_checked, n_held = 0, 0
            for l_str, entries in json_msg["poles_labels"].items():
                for e in entries:
                    assert list(e) == ["rank", "neuron_idx", "polarity", "abs_act", "write_size", "j_vindex_tokens",
                                       "naive_tokens", "label_check"], list(e)
                    if not e["j_vindex_tokens"]:
                        assert e["label_check"] is None, (l_str, e)
                        continue
                    c = assert_label_check(e["label_check"], int(l_str), lens["id"], f"poles_labels[{l_str}][{e['rank']}]")
                    n_checked += 1
                    n_held += c["held_back"]
                    if int(l_str) == 59:
                        assert c["shared_top10"] == 10 and not c["held_back"], c
            assert n_checked > 0 and all(e["label_check"] and e["label_check"]["held_back"]
                                         for e in json_msg["poles_labels"]["35"]), "layer 35's labels are not held back"
            print(f"  • the sweep's labels: {n_checked} checked, {n_held} held back; the rule just before the lens")
            # the sweep computes its counts on its own path (engine_runtime.py, from the pole vectors it ranked): each
            # labelled face in the band must be what inspect_pole reads for that neuron and face, read afresh
            n_cross, differ = 0, []
            for l_str, entries in json_msg["poles_labels"].items():
                if not BAND[0] <= int(l_str) <= BAND[1]:
                    continue
                for e in entries:
                    if not e["label_check"]:
                        continue
                    face = "north_pole" if e["polarity"] == "+" else "south_pole"
                    again = inspect_pole(layer=int(l_str), neuron_idx=e["neuron_idx"],
                                         polarity=1.0 if e["polarity"] == "+" else -1.0)[face]["label_check"]
                    n_cross += 1
                    if again != e["label_check"]:
                        differ.append((l_str, e["rank"], e["neuron_idx"], e["polarity"], e["label_check"], again))
            assert n_cross > 0 and not differ, f"the sweep's counts differ from inspect_pole's: {differ}"
            print(f"  • the sweep's counts: all {n_cross} in-band labels equal inspect_pole's own reading")

            # the first write-ranked sweep since the server started computes each layer's write-column norms once
            # (engine_runtime.py write_col_norms, cached). It differs from the plain sweep just taken only in
            # pole_ranking, so nothing else is computed for the first time in it (no nudge, so no quiet masks; no
            # centring; the same bare prompt), and an identical second call shows what that one-off cost was
            write_payload = {
                "prompt": "The number of legs on the animal that spins webs is",
                "monuments": ["spider", "eight", "insect"],
                "survey_mode": "monument",
                "compare_think": False,
                "max_new_tokens": 6,
                "pole_ranking": "write",
            }
            t_first = time.perf_counter()
            _, wmeta = await ws_sweep(ws, write_payload)
            t_first = time.perf_counter() - t_first
            t_again = time.perf_counter()
            await ws_sweep(ws, write_payload)
            t_again = time.perf_counter() - t_again
            assert wmeta["pole_ranking"] == "write"
            print(
                f"  • the first write-ranked sweep took {t_first:.1f} s; the same request again, {t_again:.1f} s "
                f"(the plain sweep before it, ranked by |a|: {json_msg['elapsed_ms']} ms on the server)"
            )

            # ---------------------------------------------------------------------
            # Step 2b: the options added for V7 bugs 1, 3, 4, 5 and 8
            # ---------------------------------------------------------------------
            print("\n[3b/4] Verifying the chat template, survey-frame, nudge, thread-score and pole-ranking options...")
            system_turn = "You are a careful assistant."
            options_payload: Dict[str, Any] = {
                "prompt": "Tell me about yourself.",
                "chat": True,
                "system": system_turn,
                "assistant_prefix": "Hi! I",
                "monuments": ["she", "he", "it"],
                "max_new_tokens": 4,
                "frame_squaring": "lowdin",
                "frame_fnw": True,
                "centre_prompts": ["The weather today is"],
                "pole_ranking": "write",
                "thread_score": "direct",
                "steering": {
                    "enabled": True,
                    "target_concept": "she",
                    "contrast_concept": "he",
                    "layers": [29, 35, 41, 47, 53],
                    "direction": "fnw",
                    "mask_quiet": True,
                    "spare_bos": True,
                    "strength": 0.01,
                },
            }
            bin2, meta = await ws_sweep(ws, options_payload)
            assert meta["prompt"].startswith(f"<bos><|turn>system\n{system_turn}<turn|>\n<|turn>user\nTell me about yourself.")
            assert meta["prompt"].endswith("<|turn>model\n<|channel>thought\n<channel|>Hi! I"), meta["prompt"][-80:]
            assert meta["prompt_token_ids"][0] == 2 and meta["prompt_token_ids"][1] != 2, "expected exactly one BOS"
            assert meta["chat"] is True and meta["system"] == system_turn and meta["assistant_prefix"] == "Hi! I"
            fr = meta["frame"]
            assert fr["squaring"] == "lowdin" and fr["fnw_weighted"] and fr["centred"] and fr["n_centre_prompts"] == 1
            assert {"raw", "fnw", "self_cos"} <= set(fr["geometry"]) and fr["geometry"]["raw"]["cond"] >= 1.0
            assert all(0.0 < p["proj_frac"] <= 1.0 for p in meta["layer_plumes"])
            assert all(len(p["pole_write_sizes"]) == 15 and p["poles_ffn_explained"] is not None for p in meta["layer_plumes"])
            assert meta["pole_ranking"] == "write" and meta["thread_score"] == "direct"
            fil0 = meta["filaments_meta"][0]
            assert fil0["score_kind"] == "direct" and fil0["direct_gate"] is not None and fil0["score"] == fil0["direct_gate"]
            sa = meta["steering_applied"]
            assert meta["is_steered"] and sa["direction"] == "fnw" and sa["mask_quiet"] and sa["spare_bos"] and sa["strength"] == 0.01
            assert all(n >= 181 for n in sa["n_masked_by_layer"].values())
            assert np.isfinite(np.frombuffer(bin2, dtype=np.float32, offset=24)).all()
            print(
                f"  • chat + system + Löwdin/fnw/centred frame + write ranking + direct threads + mended nudge: "
                f"ids {meta['prompt_token_ids'][:3]} | went on {meta['generated_text']!r} | "
                f"κ raw {fr['geometry']['raw']['cond']:.2f}, fnw {fr['geometry']['fnw']['cond']:.2f} | "
                f"proj_frac {min(p['proj_frac'] for p in meta['layer_plumes']):.4f}–{max(p['proj_frac'] for p in meta['layer_plumes']):.4f}"
            )

            # Löwdin is order-free: each word's coordinate is the same whichever order the words are named in
            coords = {}
            for order in (["she", "he", "it"], ["he", "it", "she"]):
                b, _ = await ws_sweep(
                    ws,
                    {"prompt": "Tell me about yourself.", "chat": True, "assistant_prefix": "Hi! I", "monuments": order,
                     "max_new_tokens": 0, "frame_squaring": "lowdin"},
                )
                xyz = np.frombuffer(b, dtype=np.float32, offset=24)[:180].reshape(60, 3)
                coords[tuple(order)] = {w: xyz[:, k] for k, w in enumerate(order)}
            a_, b_ = coords.values()
            drift = max(float(np.abs(a_[w] - b_[w]).max() / (np.abs(a_[w]).max() + 1e-9)) for w in ("she", "he", "it"))
            assert drift < 1e-3, f"Löwdin frame depends on the order: relative drift {drift}"
            print(f"  • Löwdin frame order-free across two orders: max relative drift {drift:.1e}")

            _, meta_pl = await ws_sweep(
                ws,
                {"prompt": "Tell me about yourself.", "chat": True, "monuments": ["she", "he", "it"], "max_new_tokens": 0,
                 "frame_per_layer": True},
            )
            assert meta_pl["frame"]["per_layer"] and meta_pl["frame"]["state"] == "h_raw"
            assert all(0.0 < p["proj_frac"] <= 1.0 for p in meta_pl["layer_plumes"])
            print("  • Per-layer frame: the raw h_l read in each layer's pulled-back frame")

            await ws.send(json.dumps({"action": "trace_readers", "layer": 35, "neuron_idx": 9482,
                                      "prompt": meta["prompt"], "max_layer_hop": 3, "top_k": 4, "score": "direct"}))
            tr = json.loads(await ws.recv())
            assert tr["type"] == "downstream_readers", tr
            trd = tr["data"]
            rs = trd["top_downstream_readers"]
            assert trd["max_layer_hop"] == 3 and trd["score_kind"] == "direct" and 0 < len(rs) <= 4
            assert_base_lens(trd.get("lens"), health["model_id"])
            assert trd["lens"] == lens, "the WebSocket trace_readers reply and /api/health name different lenses"
            assert all(36 <= r["dst_layer"] <= 38 for r in rs)
            assert all(abs(rs[i]["score"]) >= abs(rs[i + 1]["score"]) for i in range(len(rs) - 1))
            assert trd["label_gate"] == LABEL_GATE_CARD
            for r in rs:
                assert_label_check(r["dst_label_check"], r["dst_layer"], lens["id"], f"WS reader L{r['dst_layer']}")
                assert len(r["dst_naive_tokens"]) == 4, r
            print(f"  • WS trace_readers(hop=3, k=4, direct): top D={rs[0]['direct_gate']:+.4f} (C={rs[0]['coupling_score']:+.4f})")

            await ws.send(json.dumps({"action": "larql", "query": "DESCRIBE POLE LAYER 53 SLOT 16383 POLARITY -"}))
            lq = json.loads(await ws.recv())
            assert lq["type"] == "larql_result" and lq["ok"], lq
            # the vault's DESCRIBE POLE is inspect_pole's reading, and names the same lens, last
            assert list(lq["data"]) == list(pole_res), f"WS DESCRIBE POLE keys {list(lq['data'])} != {list(pole_res)}"
            assert lq["data"]["lens"] == lens, "the WebSocket larql reply names a different lens"
            # and its label checks are inspect_pole's, with the rule just before the lens
            assert lq["data"]["label_gate"] == LABEL_GATE_CARD
            for face in ("north_pole", "south_pole", "active_pole"):
                assert lq["data"][face]["label_check"] == pole_res[face]["label_check"], f"WS DESCRIBE POLE {face} check"
            print("  • WS larql DESCRIBE POLE: inspect_pole's keys and label checks, the rule and the lens named last")

            # ---------------------------------------------------------------------
            # Step 3c: a sighting of every word-piece (request ids, the stream, cancels, the sampler)
            # ---------------------------------------------------------------------
            print("\n[3c/4] Verifying a sighting of every word-piece over the WebSocket...")
            await check_answer_streams(ws, bin_msg, json_msg)

        mcp_direct = await trace_downstream_readers(
            layer=35, neuron_idx=9482, prompt="How many legs does a spider have?", chat=True,
            assistant_prefix="A spider has", max_layer_hop=2, top_k=3, score="direct",
        )
        assert mcp_direct["chat"] and mcp_direct["prompt"].startswith("<bos><|turn>user\n")
        assert all("direct_act" in r for r in mcp_direct["top_downstream_readers"])
        assert_base_lens(mcp_direct.get("lens"), health["model_id"])
        pole_http = (await client.post("/api/scientist/inspect_pole", json={"layer": 53, "neuron_idx": 16383, "polarity": -1.0})).json()
        # the HTTP route passes the tool's reply on as it is: the same keys, the same words, the same lens
        assert list(pole_http) == list(pole_res), f"HTTP keys {list(pole_http)} != MCP keys {list(pole_res)}"
        assert pole_http["lens"] == pole_res["lens"] == lens, "the HTTP reply names a different lens"
        assert pole_http["label_gate"] == LABEL_GATE_CARD
        for face in ("north_pole", "south_pole", "active_pole"):
            assert pole_http[face]["label_check"] == pole_res[face]["label_check"], f"HTTP inspect_pole {face} check"
        assert [t["token"] for t in pole_http["south_pole"]["j_vindex_tokens"]] == [
            t["token"] for t in pole_res["south_pole"]["j_vindex_tokens"]
        ], "the HTTP reply's words differ from the MCP tool's"
        print("  • HTTP inspect_pole: the same reply as the MCP tool, the lens named in both")

        # The vault's statements that read words or write an edit through the lens name it last, under "lens", over
        # MCP and HTTP (the WebSocket is checked above); the rest of each reply is as it always was, and the
        # statements that touch no lens (DESCRIBE, ROLLBACK) reply exactly as before.
        nearest_keys = ["command", "concept", "resolved_token_id", "resolved_token_str", "layer", "matrix", "use_jpp",
                        "elapsed_ms", "results"]
        vault_pole = query_larql_vault("DESCRIBE POLE LAYER 53 SLOT 16383 POLARITY -")
        assert list(vault_pole) == list(pole_res), f"MCP DESCRIBE POLE keys {list(vault_pole)} != {list(pole_res)}"
        assert vault_pole["lens"] == lens, "query_larql_vault DESCRIBE POLE names a different lens"
        assert vault_pole["label_gate"] == LABEL_GATE_CARD
        for face in ("north_pole", "south_pole", "active_pole"):
            assert vault_pole[face]["label_check"] == pole_res[face]["label_check"], f"MCP DESCRIBE POLE {face} check"
        assert [t["token"] for t in vault_pole["south_pole"]["j_vindex_tokens"]] == [
            t["token"] for t in pole_res["south_pole"]["j_vindex_tokens"]
        ], "the vault's DESCRIBE POLE words differ from inspect_pole's"
        nearest_q = "SELECT * FROM EDGES NEAREST 'she' WHERE layer = 35 LIMIT 3"
        r_near = await client.post("/api/larql", json={"query": nearest_q})
        assert r_near.status_code == 200, r_near.text
        near = r_near.json()
        assert list(near) == nearest_keys + ["label_gate", "lens"], f"HTTP NEAREST keys {list(near)}"
        assert near["lens"] == lens and len(near["results"]) == 3, "HTTP NEAREST: wrong lens or rows"
        assert near["label_gate"] == LABEL_GATE_CARD
        assert all(row["j_vindex_top_tokens"] for row in near["results"]), near["results"]
        # every row at layer 35 is held back by the layer (its five-word lists unchanged)
        for row in near["results"]:
            c = assert_label_check(row["label_check"], 35, lens["id"], f"NEAREST row {row.get('neuron_idx')}")
            assert c["held_back"] and "below_layer_40" in c["reasons"], c
        r_desc = await client.post("/api/larql", json={"query": "DESCRIBE"})
        assert r_desc.status_code == 200 and r_desc.json()["command"] == "DESCRIBE", r_desc.text
        assert "lens" not in r_desc.json(), "DESCRIBE reads no words through the lens, yet names one"
        upd = query_larql_vault("UPDATE LAYER 35 SLOT 9482 SET object = 'she' STRENGTH 0.5")
        try:
            # an edit reads no label: no label_gate in its reply
            assert list(upd) == ["status", "patch", "active_patches_count", "lens"], f"UPDATE keys {list(upd)}"
            assert upd["status"] == "applied" and upd["lens"] == lens, upd
        finally:
            rolled = query_larql_vault("ROLLBACK")
        assert list(rolled) == ["status", "cleared_patches", "active_patches_count"], f"ROLLBACK keys {list(rolled)}"
        assert rolled["status"] == "rolled_back" and rolled["cleared_patches"] >= 1, rolled
        print("  • vault: MCP DESCRIBE POLE and UPDATE, HTTP NEAREST name the lens; DESCRIBE and ROLLBACK as before")
        hyp_chat = await test_motif_hypothesis(
            layer=53, neuron_idx=16383, chat=True,
            pos_prompts=["What animal spins silk webs to catch insects?"], neg_prompts=["What is the capital of France?"],
        )
        assert hyp_chat["chat"] is True
        assert_base_lens(hyp_chat.get("lens"), health["model_id"])
        print(f"  • MCP trace_downstream_readers(chat, direct) and test_motif_hypothesis(chat): ok")

        # trace_readers on exact ids (a sighting's ids up to its read position) reads what the text did at token 0
        by_text = await trace_downstream_readers(layer=35, neuron_idx=9482, prompt=json_msg["prompt"], max_layer_hop=3, top_k=5)
        by_ids = await trace_downstream_readers(layer=35, neuron_idx=9482, prompt="", max_layer_hop=3, top_k=5,
                                                token_ids=json_msg["prompt_token_ids"])
        assert by_ids["token_ids_given"] is True and by_text["token_ids_given"] is False
        assert by_ids["read_pos"] == len(json_msg["prompt_token_ids"]) - 1 == by_text["read_pos"]
        # the ids decoded, for reading: the start mark the tokenizer added, then the text
        assert by_ids["steered"] is False and by_ids["prompt"].endswith(json_msg["prompt"])
        assert by_ids["top_downstream_readers"] == by_text["top_downstream_readers"], "exact ids read another stream"
        r_ids = await client.post("/api/scientist/trace_readers", json={"layer": 35, "neuron_idx": 9482, "max_layer_hop": 3,
                                                                         "top_k": 5, "token_ids": json_msg["prompt_token_ids"]})
        assert r_ids.status_code == 200 and r_ids.json()["top_downstream_readers"] == by_text["top_downstream_readers"]
        r_bad = await client.post("/api/scientist/trace_readers", json={"layer": 35, "neuron_idx": 9482, "token_ids": [2, 262144]})
        assert r_bad.status_code == 400 and "not a word-piece id" in r_bad.text, r_bad.text
        print("  • trace_readers on exact ids (MCP and HTTP) reads what the text read; a bad id refused in plain words")

        # saved prompts, in the temporary file set at the start
        sp = (await client.get("/api/saved_prompts")).json()
        assert sp["file"] == os.environ["THEODOLITE_SAVED_PROMPTS"] and sp["questions"] == [] and sp["systems"] == []
        r = await client.post("/api/saved_prompts", json={"kind": "question", "name": "Spider legs",
                                                          "question": "How many legs does a spider have?", "chat": True})
        assert r.status_code == 200 and r.json()["questions"][0]["chat"] is True, r.text
        qid = r.json()["questions"][0]["id"]
        r = await client.post("/api/saved_prompts", json={"kind": "question", "name": " spider LEGS", "question": "x"})
        assert r.status_code == 409, r.text
        r = await client.get("/api/saved_prompts", headers={"Origin": "https://example.com"})
        assert r.status_code == 403, r.text
        r = await client.get("/api/saved_prompts", headers={"Origin": f"http://{HOST}:{PORT}"})
        assert r.status_code == 200 and len(r.json()["questions"]) == 1
        # a DNS-rebinding page: another name that resolves here, a same-origin GET with no Origin
        r = await client.get("/api/saved_prompts", headers={"Host": f"rebind.example:{PORT}"})
        assert r.status_code == 403 and "Spider legs" not in r.text, r.text
        r = await client.delete(f"/api/saved_prompts/question/{qid}")
        assert r.status_code == 200 and r.json()["questions"] == []
        print(f"  • saved prompts over HTTP: kept, a same name refused (409), another site or Host refused (403), forgotten")

        # -------------------------------------------------------------------------
        # Step 3: Headless Chrome inspection of the instrument's pages (reader and scholar views)
        # -------------------------------------------------------------------------
        print("\n[4/4] Launching Headless Chrome page inspection & screenshot loop...")
        chrome_Data_dir = Path(tempfile.mkdtemp(prefix="theodolite_chrome_"))
        chrome_cmd = [
            os.environ.get("CHROME_BIN") or shutil.which("google-chrome-stable") or shutil.which("chromium") or "google-chrome-stable",
            "--headless=new",
            f"--remote-debugging-port={CDP_PORT}",
            f"--user-data-dir={chrome_Data_dir}",
            "--no-proxy-server",
            "--use-gl=angle",
            "--use-angle=swiftshader",
            "--enable-webgl",
            "--ignore-gpu-blocklist",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu-sandbox",
            "--window-size=1720,1040",
            f"http://{HOST}:{PORT}/",
        ]
        chrome_proc = subprocess.Popen(
            chrome_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cdp: CDPClient | None = None
        try:
            # Discover target page WebSocket debugger URL
            ws_debug_url = None
            async with httpx.AsyncClient(timeout=10.0, trust_env=False) as cdp_http:
                for _ in range(40):
                    try:
                        targets = (await cdp_http.get(f"http://127.0.0.1:{CDP_PORT}/json/list")).json()
                        for t in targets:
                            if t.get("type") == "page" and f"{HOST}:{PORT}" in t.get("url", ""):
                                ws_debug_url = t["webSocketDebuggerUrl"]
                                break
                        if ws_debug_url:
                            break
                    except Exception:
                        pass
                    await asyncio.sleep(0.25)

            if not ws_debug_url:
                raise RuntimeError("Failed to connect to headless Chrome DevTools target.")

            cdp = CDPClient(ws_debug_url)
            await cdp.connect()

            async def wait_for(js: str, what: str, tries: int = 120, delay: float = 0.5) -> Any:
                last = None
                for _ in range(tries):
                    last = await cdp.eval_js(js)
                    if last:
                        return last
                    await asyncio.sleep(delay)
                raise RuntimeError(f"Timed out waiting for {what}: {last}")

            async def go(route: str) -> None:
                await cdp.eval_js(f"(() => {{ location.hash = {json.dumps(route)}; }})()")
                await asyncio.sleep(0.8)

            # The instrument: wait until the first live sighting has replaced the recorded one (E6 fixture)
            await go("#/instrument")
            await wait_for(
                """(() => {
                    const t = document.body ? document.body.innerText : "";
                    return t.includes("· ready") && t.includes("it went on") && !t.includes("the study’s recorded sighting");
                })()""",
                "the first live sighting",
            )
            # the head of the page, the sighting's status line and the footer print what the server's card says
            # (computed here from the card), and a base-fitted lens gets its red line; that the page parsed the card,
            # rather than falling back to the lens it loads by default, is checked on its apparatus record below
            lens_ui = await cdp.eval_js(
                """(() => {
                    const spans = Array.from(document.querySelectorAll("span")).filter((e) => e.textContent.includes("60 layers surveyed"));
                    spans.sort((a, b) => a.textContent.length - b.textContent.length);
                    const foot = document.querySelector("footer");
                    const head = document.querySelector('.site-header [role="status"]');
                    return { header: head ? head.textContent.replace(/\\s+/g, " ") : null,
                             status: spans.length ? spans[0].textContent.replace(/\\s+/g, " ") : null,
                             footer: foot ? foot.textContent.replace(/\\s+/g, " ") : null,
                             caution: document.body.innerText.includes("fitted on the base model, not on the") };
                })()"""
            )
            want_status = lens_phrase(lens)[len("the "):]
            want_footer = f"through {lens_phrase(lens, named=True)}"
            assert lens_ui["header"] and want_status in lens_ui["header"], f"page head {lens_ui['header']!r} lacks {want_status!r}"
            assert lens_ui["status"] and want_status in lens_ui["status"], f"status line {lens_ui['status']!r} lacks {want_status!r}"
            assert lens_ui["footer"] and want_footer in lens_ui["footer"], f"footer {lens_ui['footer']!r} lacks {want_footer!r}"
            assert lens_ui["caution"], "the instrument shows no red line that its lens was fitted on the base model"
            print(f"  • Instrument: page head, status line and footer name {lens_phrase(lens)!r}, with its red line")
            figure = await cdp.eval_js(
                """(() => {
                    const svg = document.querySelector('svg[role="img"][aria-label^="Sixty stacked rings"]');
                    return { found: !!svg, paths: svg ? svg.querySelectorAll("path").length : 0 };
                })()"""
            )
            print(f"  • Instrument drawing: found={figure['found']} paths={figure['paths']}")
            assert figure["found"] and figure["paths"] > 10, f"Instrument drawing missing: {figure}"

            # The label gate on a live sighting. The instrument's leaf opens at layer 36, below layer 41: the label of
            # its neuron is held back, a plate and the reason in place of the words, which come back, between gilt
            # brackets, only with the apparatus open
            held_js = """(() => {
                const b = document.querySelector('[aria-label^="held back · below layer 41"]');
                const t = b ? b.textContent : "";
                return { found: !!b, said: document.body.innerText.includes("Its words are held back at this layer."),
                         words: /lens \\[/.test(t) };
            })()"""
            await cdp.eval_js(f"{SET_READING_JS}('apparatus', false)")
            await asyncio.sleep(0.3)
            closed = await cdp.eval_js(held_js)
            await cdp.eval_js(f"{SET_READING_JS}('apparatus', true)")
            await asyncio.sleep(0.3)
            opened = await cdp.eval_js(held_js)
            await cdp.eval_js(f"{SET_READING_JS}('apparatus', false)")
            await asyncio.sleep(0.3)
            assert closed == {"found": True, "said": True, "words": False}, f"layer 36's leaf, apparatus closed: {closed}"
            assert opened["found"] and opened["words"], f"layer 36's leaf, apparatus open: {opened}"
            print("  • Instrument, live: layer 36's neuron words held back from the leaf, and kept in its apparatus")
            # the folio of neuron 15491 at layer 54 gives the server's ten-a-side count on its seal, and shows its words
            # unless the server's check holds them back
            await go("#/neuron/54/15491/pos")
            seal = f"{folio_check['shared_top10']} IN 10"
            await wait_for(f"""document.body.innerText.includes({json.dumps(seal)})""", f"the folio's seal “{seal}”")
            folio_ui = await cdp.eval_js(
                f"""(() => {{ const t = document.body.innerText;
                    return {{ word: t.includes({json.dumps(folio_word)}), held: t.includes("Its words are held back") }}; }})()"""
            )
            if folio_check["held_back"]:
                assert folio_ui["held"], f"the folio of 15491 does not hold back a label the server holds back: {folio_ui}"
            else:
                assert folio_ui == {"word": True, "held": False}, f"the folio of 15491 at layer 54: {folio_ui}"
            print(f"  • Neuron folio 54/15491: seal “{seal}”, its words "
                  f"{'held back' if folio_check['held_back'] else 'shown (' + folio_word + ' first)'}, as the server's check says")
            # Condition 3 on a live page: the leaf of layer 55 (index 54, inside the band, always labelled) lists its
            # neurons as the interface's own sighting ranked them; each labelled row must give the count inspect_pole
            # reads for that neuron and face (a label's check does not depend on the prompt), or be held back where
            # that check holds it back
            await go("#/layer/55")
            await cdp.eval_js(f"{SET_READING_JS}('apparatus', false)")
            await wait_for("document.querySelectorAll('a[aria-label^=\"Open neuron \"]').length > 0", "layer 55's leaf, live")
            await asyncio.sleep(0.5)
            leaf_rows = await cdp.eval_js(
                """(() => Array.from(document.querySelectorAll('a[aria-label^="Open neuron "]')).map((a) => {
                    const li = a.closest("li"); const t = li ? li.innerText : "";
                    const m = t.match(/they share (only 1|none|\\d+) of ten/);
                    return { neuron: Number(a.getAttribute("aria-label").replace("Open neuron ", "")),
                             positive: t.includes("positive face"), held: t.includes("[held back]"),
                             count: m ? m[1] : null, why: t.includes("share none of their first ten") };
                }))()"""
            )
            n_rows = 0
            for row in leaf_rows:
                if row["count"] is None and not row["held"]:
                    continue  # not labelled in this sighting
                face = "north_pole" if row["positive"] else "south_pole"
                c = inspect_pole(layer=54, neuron_idx=row["neuron"], polarity=1.0 if row["positive"] else -1.0)[face]["label_check"]
                k = c["shared_top10"]
                what = f"layer 55's leaf, live, neuron {row['neuron']} ({face})"
                if c["held_back"]:
                    assert row["held"] and row["count"] is None, f"{what}: held back by the server ({c}), but {row}"
                    if "shares_nothing" in c["reasons"]:
                        assert row["why"], f"{what}: held back for sharing nothing, but the row does not say so: {row}"
                else:
                    want = "none" if k == 0 else "only 1" if k == 1 else str(k)
                    assert not row["held"] and row["count"] == want, f"{what}: the server counts {k} ({c}), the page {row}"
                n_rows += 1
            assert n_rows > 0, f"layer 55's leaf, live, has no labelled row: {leaf_rows}"
            print(f"  • Layer 55's leaf, live: {n_rows} labelled rows, each with inspect_pole's count, or held back as it says")
            await go("#/instrument")

            # How to read these pages: rubrics (red italic guidance) and apparatus (double-gilt-ruled blocks)
            count_js = """(() => {
                const all = Array.from(document.querySelectorAll("main *"));
                const red = all.filter((e) => {
                    const cs = getComputedStyle(e);
                    return cs.fontStyle === "italic" && cs.color === "rgb(143, 43, 30)" && e.offsetParent !== null
                        && Array.from(e.childNodes).some((n) => n.nodeType === 3 && n.textContent.trim().length > 20);
                }).length;
                // apparatus blocks: ruled off by a double gilt line (some decorative frames use one too)
                const ruled = all.filter((e) => getComputedStyle(e).borderTopStyle === "double" && e.offsetParent !== null);
                const app = ruled.length;
                const notes = ruled.filter((e) => e.getAttribute("role") === "note").length;
                return { red, app, notes };
            })()"""
            click_js = """((name) => {
                const b = Array.from(document.querySelectorAll('[aria-label="How to read these pages"] button'))
                    .find((x) => x.textContent.includes(name));
                b.click();
                return b.getAttribute("aria-pressed");
            })"""
            routes = ["#/", "#/instrument", "#/layer/36", "#/neuron/54/15491/pos", "#/landmarks", "#/nudge",
                      "#/vault", "#/lexicon", "#/lineage"]
            for route in routes:
                await go(route)
                reader = await cdp.eval_js(count_js)  # rubrics on, apparatus off (fresh profile)
                await cdp.eval_js(f"{click_js}('apparatus')")
                await cdp.eval_js(f"{click_js}('rubrics')")
                await asyncio.sleep(0.3)
                scholar = await cdp.eval_js(count_js)  # rubrics off, apparatus on
                await cdp.eval_js(f"{click_js}('apparatus')")
                await cdp.eval_js(f"{click_js}('rubrics')")
                await asyncio.sleep(0.3)
                print(f"  • {route:<22} rubric lines {reader['red']:>3} → {scholar['red']:>3} | "
                      f"apparatus blocks {reader['app']:>3} → {scholar['app']:>3}")
                # a few red italic lines are warnings the canvas keeps visible; the rubrics must all go
                assert reader["red"] > 0 and scholar["red"] < reader["red"], f"{route}: rubrics toggle did not work"
                assert reader["notes"] == 0 and scholar["app"] > reader["app"], f"{route}: apparatus toggle did not work"

            # Screenshot #1: the instrument, scholar's view
            await go("#/instrument")
            await cdp.eval_js(f"{click_js}('apparatus')")
            await asyncio.sleep(0.5)
            shot1 = ARTIFACT_DIR / "theodolite_instrument_scholar_view.png"
            await cdp.screenshot(shot1)
            print(f"  • Saved Screenshot #1 (the instrument, apparatus open): {shot1}")
            # The apparatus prints the lens record the page parsed. Its fallback (the lens the instrument loads, when
            # no record arrived) prints the same names, so require the line that says the record came with this
            # sighting, and every field of the server's card.
            record = await cdp.eval_js(
                """(() => {
                    const dt = Array.from(document.querySelectorAll("dt")).find((e) => e.textContent.trim() === "lens record");
                    const dd = dt ? dt.nextElementSibling : null;
                    return dd ? dd.textContent.replace(/\\s+/g, " ") : null;
                })()"""
            )
            assert record, "the instrument's apparatus shows no lens record"
            missing = [k for k in ("name", "fitted_on", "applied_to", "source", "file") if lens[k] not in record]
            assert not missing, f"the lens record lacks the server's {missing}: {record!r}"
            assert "As the server reported it with this sighting" in record, f"the lens record is not the server's: {record!r}"
            print("  • Instrument apparatus: the lens record is the server's own, as sent with the sighting")
            await cdp.eval_js(f"{click_js}('apparatus')")

            # Screenshot #2: arm a gentle nudge (×0.01 of the stream) and wait for the nudged sighting
            await go("#/nudge")
            armed = await cdp.eval_js(
                """(() => {
                    const b = Array.from(document.querySelectorAll("button")).find((x) => x.textContent.trim() === "arm the nudge");
                    if (b && !b.disabled) { b.click(); return true; }
                    return false;
                })()"""
            )
            assert armed, "Could not find an enabled 'arm the nudge' button"
            # the page first takes a plain sighting of its question, then the nudged one; stay on it meanwhile
            await wait_for(
                """(() => { const t = document.body.innerText;
                    return !t.includes("TAKE A SIGHTING TO SEE THE NUDGED ANSWER") && !t.includes("taking a sighting"); })()""",
                "the nudged sighting",
            )
            await go("#/instrument")
            nudged = await cdp.eval_js("""document.body.innerText.includes("nudged")""")
            assert nudged, "The instrument does not mark the sighting as nudged"
            await go("#/nudge")
            shot2 = ARTIFACT_DIR / "theodolite_nudge_view.png"
            await cdp.screenshot(shot2)
            print(f"  • Saved Screenshot #2 (after a gentle nudge): {shot2}")

            # The formerly disabled controls ("[NOT YET IN THE SERVER]") are live and drive the server
            await cdp.eval_js(f"{click_js}('apparatus')")
            await asyncio.sleep(0.4)
            fixes = await cdp.eval_js(
                """(() => {
                    const labels = Array.from(document.querySelectorAll("label.check"));
                    const want = ["the corrected direction", "mask the quiet dimensions", "leave the start mark alone", "the strength as a share"];
                    return want.map((w) => { const l = labels.find((x) => x.textContent.includes(w));
                        const i = l && l.querySelector("input"); return i ? { w, checked: i.checked, disabled: i.disabled } : { w, missing: true }; });
                })()"""
            )
            assert all(f.get("checked") and not f.get("disabled") for f in fixes), f"Nudge fix boxes: {fixes}"
            print("  • Nudge: the four fix boxes are enabled and ticked")
            await go("#/instrument")
            chat_ui = await cdp.eval_js(
                """(() => {
                    const l = Array.from(document.querySelectorAll("label.check")).find((x) => x.textContent.includes("send it as a chat"));
                    const i = l && l.querySelector("input");
                    return { found: !!i, checked: i ? i.checked : null, disabled: i ? i.disabled : null, system: !!document.querySelector("#system"),
                             notYet: document.body.innerText.includes("NOT YET IN THE SERVER") };
                })()"""
            )
            assert chat_ui["found"] and chat_ui["checked"] and not chat_ui["disabled"] and chat_ui["system"], f"chat control: {chat_ui}"
            assert not chat_ui["notYet"], "the instrument page still shows a [NOT YET IN THE SERVER] control"
            print("  • Instrument: “send it as a chat” is enabled (ticked by default), with a system-prompt field")
            # the sighting shown is ranked by write size, the default: the leaf, the folio and the nudge word it so. The
            # server reads words for ranks 0-4 only, so the leaf must name its lead by the sighting's own rank 0
            # (leadOf), whose words are read, never the loudest by |a| among the fifteen, which may have none
            await wait_for("""(() => !document.body.innerText.includes("taking a sighting…"))()""", "an idle instrument")
            await go("#/layer/36")
            await cdp.eval_js(
                """(() => {
                    const b = Array.from(document.querySelectorAll('[aria-label="How to read these pages"] button'))
                        .find((x) => x.textContent.includes("apparatus"));
                    if (b && b.getAttribute("aria-pressed") !== "true") b.click();
                })()"""
            )
            await wait_for("""document.body.innerText.includes("read the biggest writer")""", "the leaf's link to its biggest writer")
            leaf = await cdp.eval_js(
                """(() => {
                    const t = document.body.innerText;
                    const a = Array.from(document.querySelectorAll("a")).find((x) => x.textContent.trim() === "read the biggest writer");
                    return { href: a ? a.getAttribute("href") : null, read: t.includes("the biggest writer’s lens words score"),
                             notRead: t.includes("the biggest writer’s lens words and their scores"),
                             loudest: t.includes("read the loudest neuron") };
                })()"""
            )
            assert leaf["href"] and leaf["read"] and not leaf["notRead"] and not leaf["loudest"], f"layer 36's leaf: {leaf}"
            print(f"  • Layer 36's leaf: “read the biggest writer” ({leaf['href']}), whose lens words were read")
            await go(leaf["href"])
            await wait_for(
                """document.body.innerText.toLowerCase().includes("biggest writer of the 21,504 in its layer")""",
                "the folio's write-size particular",
            )
            folio = await cdp.eval_js(
                """(() => { const t = document.body.innerText.toLowerCase();
                    return { writeSize: t.includes("write size"), listed: t.includes("by |a| of the fifteen listed"),
                             loudness: t.includes("loudest of the 21,504") }; })()"""
            )
            assert folio == {"writeSize": True, "listed": True, "loudness": False}, f"the biggest writer's folio: {folio}"
            print("  • Its folio: “write size”, the biggest writer of the 21,504 in its layer, its |a| rank of the fifteen listed")
            await go("#/nudge")
            nudge_rank = await cdp.eval_js(
                """(() => { const m = document.body.innerText.match(/Neuron \\d+ fired[^\\n]*?: the ([^\\n]*?) in layer \\d+/);
                    return m ? m[1] : null; })()"""
            )
            if nudge_rank is None:
                print("  • Nudge: the armed neuron is not among the fifteen listed in this sighting; no rank to read")
            else:
                assert "biggest writer" in nudge_rank and "loudest" not in nudge_rank, f"the nudge's rank: {nudge_rank!r}"
                print(f"  • Nudge: the armed neuron is the sighting’s “{nudge_rank}” in its layer")
            await go("#/instrument")
            # neurons are ranked by write size and threads scored by the direct-path estimate by default: untick "rank
            # them by write size" to read by |a|, then "score threads by the direct-path estimate" to read C; each
            # retakes the sighting
            for label, want_checked, default_text, done in (
                ("rank them by write size", False, "the top fifteen by write size |a|", "the top fifteen by |a| = |mlp_act|"),
                ("score threads by the direct-path estimate", False,
                 "A direct-path linear estimate, not a verified circuit", "A weak guide (ρ ≈ 0.41"),
            ):
                await wait_for("""(() => !document.body.innerText.includes("taking a sighting…"))()""", "an idle instrument")
                await wait_for(f"""document.body.innerText.includes({json.dumps(default_text)})""",
                               f"the default sighting, before “{label}” is changed")
                print(f"  • Instrument: by default the sighting reads “{default_text}”")
                clicked = await cdp.eval_js(
                    f"""(() => {{
                        const l = Array.from(document.querySelectorAll("label.check")).find((x) => x.textContent.includes({json.dumps(label)}));
                        const i = l && l.querySelector("input");
                        if (!i || i.disabled) return null; i.click(); return i.checked;
                    }})()"""
                )
                assert clicked is want_checked, f"could not set “{label}” to {want_checked} (got {clicked})"
                await wait_for(f"""document.body.innerText.includes({json.dumps(done)})""", f"a sighting after “{label}”")
                print(f"  • Instrument: “{label}” {'ticked' if want_checked else 'unticked'}, a new sighting taken")
            # the Lexicon prints the request the interface sends by default, built by the store's own sweepPayload
            await go("#/lexicon")
            lex_req = await cdp.eval_js(
                """(() => {
                    const e = Array.from(document.querySelectorAll("span.mono-wrap")).find((x) => x.textContent.includes('"action":"sweep"'));
                    return e ? e.textContent : null;
                })()"""
            )
            assert lex_req, "the Lexicon shows no default request (is its apparatus open?)"
            lex_payload = json.loads(lex_req)["payload"]
            want_lex = {"frame_squaring": "lowdin", "frame_fnw": False, "frame_per_layer": False,
                        "pole_ranking": "write", "thread_score": "direct", "survey_mode": "monument"}
            got_lex = {k: lex_payload.get(k) for k in want_lex}
            assert got_lex == want_lex, f"the Lexicon's default request sends {got_lex}, not {want_lex}"
            assert not lex_payload.get("centre_prompts"), f"the default request centres: {lex_payload.get('centre_prompts')}"
            print("  • Lexicon: the default request squares all at once on raw rows and ranks by write size")
            await go("#/landmarks")
            await wait_for("""(() => !document.body.innerText.includes("taking a sighting…"))()""", "an idle landmarks page")
            # the live sightings so far were taken in the default frame: squared all at once, on raw rows
            await wait_for(
                """document.body.innerText.includes("The sighting shown was squared all at once, from raw rows, at the final layer")""",
                "a sighting in the default frame",
            )
            stones = await cdp.eval_js(
                """(() => { const t = document.body.innerText;
                    return { atOnce: t.includes("squared with the other two, all at once"), inOrder: t.includes("kept exactly as it lies") }; })()"""
            )
            assert stones == {"atOnce": True, "inOrder": False}, f"the stones do not describe a frame squared all at once: {stones}"
            print("  • Landmarks: the default sighting was squared all at once, from raw rows, and the stones say so")
            frame_ui = await cdp.eval_js(
                """(() => {
                    const radios = Array.from(document.querySelectorAll('input[name="frame-squaring"], input[name="frame-place"]'));
                    const p = Array.from(document.querySelectorAll("button")).find((x) => x.textContent.trim() === "the frame options as first built");
                    if (p && !p.disabled) p.click();
                    return { radios: radios.length, disabled: radios.filter((r) => r.disabled).length, clicked: !!p };
                })()"""
            )
            assert frame_ui["radios"] == 4 and frame_ui["disabled"] == 0 and frame_ui["clicked"], f"frame options: {frame_ui}"
            await asyncio.sleep(0.4)  # let the page re-render with the frame as first built before taking it
            took = await cdp.eval_js(
                """(() => {
                    const p = Array.from(document.querySelectorAll("button")).find((x) => x.textContent.trim() === "the frame options as first built");
                    const b = Array.from(document.querySelectorAll("button")).find((x) => x.textContent.includes("take a sighting with these landmarks"));
                    if (!b || b.disabled) return { took: false };
                    b.click();
                    return { took: true, pressed: p ? p.getAttribute("aria-pressed") : null };
                })()"""
            )
            assert took == {"took": True, "pressed": "true"}, f"the frame as first built was not set and taken: {took}"
            await wait_for(
                """document.body.innerText.includes("The sighting shown was squared in order, from raw rows, at the final layer")""",
                "a sighting in the frame options as first built",
            )
            await wait_for("""document.body.innerText.includes("kept exactly as it lies")""", "the stones of a frame squared in order")
            print("  • Landmarks: “the frame options as first built” retook the sighting squared in order, from raw rows")
            await go("#/neuron/36/9482/pos")
            await asyncio.sleep(0.5)
            picked = await cdp.eval_js(
                """(() => {
                    const d = Array.from(document.querySelectorAll("button")).find((x) => x.textContent.includes("direct-path estimate, the fix"));
                    if (!d || d.disabled) return "no direct button";
                    d.click();
                    return "ok";
                })()"""
            )
            assert picked == "ok", picked
            await asyncio.sleep(0.4)  # let the page re-render with the new score before tracing
            traced = await cdp.eval_js(
                """(() => {
                    const t = Array.from(document.querySelectorAll("button")).find((x) => x.textContent.trim() === "trace its readers");
                    if (!t || t.disabled) return "no trace button";
                    t.click();
                    return "ok";
                })()"""
            )
            assert traced == "ok", traced
            await wait_for(
                """(() => { const t = document.body.innerText;
                    return !t.includes("tracing…") && t.includes("thread score, D") && !t.includes("[READERS APPEAR AFTER TRACING]")
                        && !t.includes("[NO READERS CAME BACK]"); })()""",
                "a direct-scored trace",
            )
            print("  • Neuron folio: “direct-path estimate” scored the traced readers")
            await cdp.eval_js(f"{click_js}('apparatus')")

            # The recorded sighting (E6, taken in the frame as first built) on a page of its own whose WebSocket never
            # opens, so no live sighting replaces it: its landmarks page must describe it as squared in order
            async with httpx.AsyncClient(timeout=10.0, trust_env=False) as cdp_http:
                rec_target = (await cdp_http.put(f"http://127.0.0.1:{CDP_PORT}/json/new?about:blank")).json()
            rec = CDPClient(rec_target["webSocketDebuggerUrl"])
            await rec.connect()
            try:
                await rec.send(
                    "Page.addScriptToEvaluateOnNewDocument",
                    {"source": "window.WebSocket = class { static CONNECTING = 0; static OPEN = 1; static CLOSING = 2; "
                               "static CLOSED = 3; constructor() { this.readyState = 0; } send() {} close() {} };"},
                )
                await rec.send("Page.navigate", {"url": f"http://{HOST}:{PORT}/#/landmarks"})
                # the stones are on the main reading; the sentence on how it was squared is in the apparatus, opened here
                recorded_js = """(() => { const t = document.body ? document.body.innerText : "";
                    return { inOrder: t.includes("kept exactly as it lies"), atOnce: t.includes("squared with the other two, all at once"),
                             said: t.includes("The recorded sighting was squared in order, from raw rows, at the final layer") }; })()"""
                recorded = None
                for _ in range(60):
                    recorded = await rec.eval_js(recorded_js)
                    if recorded and recorded["inOrder"]:
                        break
                    await asyncio.sleep(0.5)
                await rec.eval_js(
                    """(() => {
                        const b = Array.from(document.querySelectorAll('[aria-label="How to read these pages"] button'))
                            .find((x) => x.textContent.includes("apparatus"));
                        if (b && b.getAttribute("aria-pressed") !== "true") b.click();
                    })()"""
                )
                await asyncio.sleep(0.4)
                recorded = await rec.eval_js(recorded_js)
                assert recorded == {"inOrder": True, "atOnce": False, "said": True}, (
                    f"the recorded sighting's landmarks page does not describe the frame as first built: {recorded}")
                assert not rec.console_errors and not rec.page_exceptions, (
                    f"the recorded sighting's page logged errors: {rec.console_errors} {rec.page_exceptions}")
                print("  • Landmarks, recorded sighting: “kept exactly as it lies”, squared in order, from raw rows")
                # Layer 36 of the recorded sighting, the default leaf: below layer 41, so neither the instrument's leaf
                # nor the layer's leaf sets its neurons' words, or the glosses written for them, in the main reading;
                # the apparatus brings them back
                shown_js = (
                    f"""(() => {{ const t = document.body ? document.body.innerText : "";
                        return {{ words: {json.dumps(L36_RECORDED_WORDS)}.filter((w) => t.includes(w)),
                                  glosses: {json.dumps(L36_RECORDED_GLOSSES)}.filter((g) => t.includes(g)),
                                  held: t.includes("Its words are held back at this layer.") }}; }})()"""
                )
                for route, ready, want_open in (
                    ("#/instrument", "the study’s recorded sighting", (L36_RECORDED_WORDS[:3], ["one idea in three tongues"])),
                    ("#/layer/36", "murmurs of machines", (L36_RECORDED_WORDS, ["Hindi “me”", "Chinese “teenagers”",
                                                                               "No word in common, yet both eyes speak of the self"])),
                ):
                    await rec.eval_js(f"(() => {{ location.hash = {json.dumps(route)}; }})()")
                    for _ in range(60):
                        if await rec.eval_js(f"document.body.innerText.includes({json.dumps(ready)})"):
                            break
                        await asyncio.sleep(0.5)
                    await rec.eval_js(f"{SET_READING_JS}('apparatus', false)")
                    await asyncio.sleep(0.4)
                    closed = await rec.eval_js(shown_js)
                    await rec.eval_js(f"{SET_READING_JS}('apparatus', true)")
                    await asyncio.sleep(0.4)
                    opened = await rec.eval_js(shown_js)
                    assert closed["held"] and not closed["words"] and not closed["glosses"], (
                        f"{route}, recorded, apparatus closed, shows layer 36's held-back words or glosses: {closed}")
                    missing = [w for w in want_open[0] if w not in opened["words"]] + [g for g in want_open[1] if g not in opened["glosses"]]
                    assert not missing, f"{route}, recorded, apparatus open, lacks {missing}"
                    print(f"  • {route}, recorded sighting: layer 36's words and glosses held back, and in the apparatus")
                # Condition 3 on the recorded sighting: at layer 54, inside the band, a face whose readings share none of
                # ten (by its recorded check) has its words left out of the main reading and kept in the apparatus; the
                # others show their words and their count
                shown_w = {wp(t) for f in L54_FACES.values() if not f["check"]["held_back"] for t in f["words"]}
                held_w = sorted({wp(t) for f in L54_FACES.values() if f["check"]["held_back"] for t in f["words"]}
                                - shown_w - {"“”", "“␣”"})
                assert held_w and shown_w, f"the recorded layer 54 has no held-back and shown faces to check: {L54_FACES}"
                l54_js = (
                    "(() => { const t = document.body ? document.body.innerText : '';"
                    f" return {{ held: {json.dumps(held_w)}.filter((w) => t.includes(w)),"
                    f" shown: {json.dumps(sorted(shown_w))}.filter((w) => t.includes(w)),"
                    " why: t.includes('Its words are held back: the lens and the naked eye share none of their first ten.'),"
                    " text: t }; })()"
                )
                await rec.eval_js("(() => { location.hash = '#/layer/54'; })()")
                for _ in range(60):
                    if await rec.eval_js("document.querySelectorAll('a[aria-label^=\"Open neuron \"]').length > 0"):
                        break
                    await asyncio.sleep(0.5)
                await rec.eval_js(f"{SET_READING_JS}('apparatus', false)")
                await asyncio.sleep(0.4)
                closed = await rec.eval_js(l54_js)
                await rec.eval_js(f"{SET_READING_JS}('apparatus', true)")
                await asyncio.sleep(0.4)
                opened = await rec.eval_js(l54_js)
                assert closed["why"] and not closed["held"], (
                    f"#/layer/54, recorded, apparatus closed, shows words its recorded checks hold back: {closed['held']}")
                assert closed["shown"], "#/layer/54, recorded, apparatus closed, shows none of the shown faces' words"
                for k, f in L54_FACES.items():
                    n = f["check"]["shared_top10"]
                    if not f["check"]["held_back"]:
                        said = f"they share {'only 1' if n == 1 else n} of ten"
                        assert said in closed["text"], f"#/layer/54, recorded: face {k}'s count {said!r} is not on the page"
                missing = [w for w in held_w if w not in opened["held"]]
                assert not missing, f"#/layer/54, recorded, apparatus open, lacks the held-back words {missing}"
                print(f"  • #/layer/54, recorded sighting: {len(held_w)} held-back word-pieces left out (share none of ten) "
                      "and kept in the apparatus; the others shown with their count")
                # the vault's recorded search for "she" at layer 59 holds back the rows its recorded checks hold back
                await rec.eval_js(f"{SET_READING_JS}('apparatus', false)")
                await rec.eval_js("(() => { location.hash = '#/vault'; })()")
                for _ in range(60):
                    if await rec.eval_js("!!document.querySelector('button[aria-label=\"Layer 59\"]')"):
                        break
                    await asyncio.sleep(0.5)
                await rec.eval_js(
                    "(() => { const b = Array.from(document.querySelectorAll('button'))"
                    ".find((x) => x.textContent.trim() === '“she”'); if (b) b.click(); })()"
                )
                await asyncio.sleep(0.4)
                await rec.eval_js("(() => { document.querySelector('button[aria-label=\"Layer 59\"]').click(); })()")
                await asyncio.sleep(0.6)
                meta = await rec.eval_js(
                    "(() => { const t = document.querySelector('[role=\"table\"]'); return t ? t.getAttribute('aria-label') : null; })()"
                )
                want_meta = ("the words of every row held back" if SHE59_HELD == 5 else
                             f"the words of {SHE59_HELD} held back" if SHE59_HELD else None)
                assert meta and "nearest “she” in layer 59" in meta and (
                    (want_meta in meta) if want_meta else "held back" not in meta), (
                    f"#/vault, recorded “she” at layer 59: {meta!r} (want {want_meta!r})")
                print(f"  • #/vault, recorded “she” at layer 59: {SHE59_HELD} of its first five rows held back, as its checks say")
                await rec.eval_js(f"{SET_READING_JS}('apparatus', false)")
                assert not rec.console_errors and not rec.page_exceptions, (
                    f"the recorded sighting's pages logged errors: {rec.console_errors} {rec.page_exceptions}")
            finally:
                await rec.close()
                async with httpx.AsyncClient(timeout=10.0, trust_env=False) as cdp_http:
                    await cdp_http.get(f"http://127.0.0.1:{CDP_PORT}/json/close/{rec_target['id']}")

            print(
                f"  • Console Errors: {len(cdp.console_errors)} | Page Exceptions: {len(cdp.page_exceptions)}"
            )
            if cdp.console_errors:
                print("    Console error details:", cdp.console_errors)
            if cdp.page_exceptions:
                print("    Page exception details:", cdp.page_exceptions)
            assert len(cdp.console_errors) == 0, f"Unexpected console errors: {cdp.console_errors}"
            assert len(cdp.page_exceptions) == 0, f"Unexpected page exceptions: {cdp.page_exceptions}"

        finally:
            if cdp:
                await cdp.close()
            chrome_proc.terminate()
            try:
                chrome_proc.wait(timeout=5)
            except Exception:
                chrome_proc.kill()
            shutil.rmtree(chrome_Data_dir, ignore_errors=True)

        # Final VRAM check
        health_after = (await client.get("/api/health")).json()
        print(
            f"\n[SUMMARY] Peak GPU VRAM across E2E Server + FastMCP + UI sweeps: "
            f"{health_after['vram_peak_gb']:.2f} GB (< 30.00 GB limit)"
        )
        assert health_after["vram_peak_gb"] < 30.0

    server.should_exit = True
    await asyncio.sleep(0.5)
    server_task.cancel()
    print("=" * 92)
    print("ALL BACKEND, FASTMCP & FRONTEND VERIFICATION CHECKS PASSED (ZERO ERRORS)")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(
        description="End-to-end check of the server, the MCP tools and the pages (loads the model on the GPU and serves "
        f"on {HOST}:{PORT}: stop the running server first).",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()
    raise SystemExit(asyncio.run(main()))

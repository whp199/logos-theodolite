#!/usr/bin/env python3
"""A CPU-only check of the lens card, `TheodoliteRuntime.lens_card`: the "lens" that GET /api/health, every sweep,
the scientist tools and the vault's lens statements send. It loads neither the model nor the lens, and hides the GPU.

The runtime is built without loading anything and given a lens file's provenance, read by `_read_lens_provenance`
from a stand-in payload, for:
1. the shipped file (its provenance as Neuronpedia's copy records it). When that file is in the local Hugging Face
   cache, its own provenance is read too (by memory map: the 3.4 GB of matrices are never read) and must match;
2. incomplete files: no provenance at all, a method without a model, a model without a method, blank fields, and
   counts stored as tensors, text or nonsense;
3. other lenses: J++ refitted on the instruct model, another method on the base model, and models named neither way.
None but the shipped file may be called "jpp-base", carry the study's caution or claim WikiText. The card for the
shipped file must also match the lens the pages fall back to (BASE_LENS in frontend/src/lib/lens.ts). The figures in
the caution's last sentence (how far in the lens was fitted and checked, and the model's sliding window) are checked
against their sources: the E3 report and its per-text lengths, and the model's own config when it is cached.

It then checks, with stand-ins for the vault's GPU work, that the vault statements that read words or write an edit
through the lens (NEAREST, DESCRIBE POLE, UPDATE, INSERT) name it last under "lens", over `execute_lql` and the MCP
tool `query_larql_vault`; that the two that read neuron labels (NEAREST, DESCRIBE POLE) also carry the label gate's
rule ("label_gate", backend/label_gate.py) just before it, and UPDATE and INSERT do not; and that DESCRIBE, DELETE and
ROLLBACK reply as they always have. (scripts/check_label_gate.py checks the label checks themselves.)

Run: .venv/bin/python -I scripts/check_lens_card.py   (exits 0 when every check holds; verify_e2e runs it first)
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    # nothing here needs the GPU; keep a running server's card untouched
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import re
import types
from pathlib import Path
from typing import Any, Callable, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch

from backend.engine_runtime import (
    DEFAULT_W4A16_MODEL_ID,
    JPP_LENS_FALLBACK,
    JPP_LENS_FILENAME,
    LENS_REPO_ID,
    UNNAMED_LENS,
    TheodoliteRuntime,
    _OTHER_LENS_CAUTION,
    _read_lens_provenance,
)
from backend.label_gate import LABEL_GATE_CARD
from backend.larql_vault import LarqlVaultEngine

# Every card carries these keys, in this order (the pages and the e2e read them by name).
CARD_KEYS = ["id", "name", "fitted_on", "applied_to", "method", "fit_data", "source", "file", "caution", "provenance"]
PROVENANCE_KEYS = ["source", "method", "lrp_mode", "hf_model_name", "n_prompts", "max_seq_len"]
SHIPPED_SOURCE = "koayon/jpp-lenses@6c96867a91c43c14cec17cf0f6d157b933c5fee5:gemma-4-31b/lens.pt"
# The shipped file's payload, less its matrices: the fields `_read_lens_provenance` reads, as the file records them
# (read from Neuronpedia's copy on 17 October 2026), with a few it ignores.
SHIPPED_PAYLOAD: Dict[str, Any] = {
    "J": {},
    "d_model": 5376,
    "n_prompts": 64,
    "provenance": {
        "source": SHIPPED_SOURCE,
        "method": "jpp",
        "lrp_mode": "rlens",
        "source_config": {
            "hf_model_name": "google/gemma-4-31B",
            "lens_type": "jacobian",
            "num_prompts_trained_on": 64,
            "max_seq_len": 128,
            "lrp_mode": "rlens",
        },
        "capture_point": "block_output",
    },
}
SHIPPED_FILE = f"{LENS_REPO_ID}:{JPP_LENS_FILENAME}"

failures: List[str] = []
passed = 0


def check(ok: bool, what: str) -> None:
    global passed
    if ok:
        passed += 1
    else:
        failures.append(what)
        print(f"  FAIL  {what}")


def runtime_with(payload: Any, model_id: str = DEFAULT_W4A16_MODEL_ID) -> TheodoliteRuntime:
    """A runtime that has 'loaded' a lens whose file holds `payload`: nothing is loaded, only the provenance is set."""
    rt = TheodoliteRuntime(model_id=model_id, device="cpu", load_jlens=False)
    rt.jpp_lenses = {0: torch.zeros(1)}
    rt.jpp_provenance = _read_lens_provenance(payload)
    return rt


def card_for(payload: Any, model_id: str = DEFAULT_W4A16_MODEL_ID) -> Dict[str, Any]:
    card = runtime_with(payload, model_id).lens_card()
    assert card is not None
    return card


def check_every_card(label: str, card: Dict[str, Any], model_id: str = DEFAULT_W4A16_MODEL_ID) -> None:
    """What holds for any lens: every key, JSON a browser can read, a name, a caution, the model it reads."""
    check(list(card) == CARD_KEYS, f"{label}: keys {list(card)} != {CARD_KEYS}")
    try:
        json.dumps(card, allow_nan=False)
        check(True, "")
    except (TypeError, ValueError) as exc:
        check(False, f"{label}: the card is not plain JSON ({exc})")
    check(isinstance(card["name"], str) and bool(card["name"].strip()), f"{label}: name {card['name']!r} is not a name")
    check(isinstance(card["caution"], str) and len(card["caution"]) > 40, f"{label}: no caution: {card['caution']!r}")
    check(card["applied_to"] == model_id, f"{label}: applied_to {card['applied_to']!r} != {model_id!r}")
    check(card["file"] == SHIPPED_FILE, f"{label}: file {card['file']!r}")
    prov = card["provenance"]
    check(isinstance(prov, dict) and list(prov) == PROVENANCE_KEYS, f"{label}: provenance keys {list(prov or {})}")
    check(
        all(v is None or isinstance(v, str) or (isinstance(v, int) and not isinstance(v, bool)) for v in prov.values()),
        f"{label}: provenance values are not plain text or whole numbers: {prov}",
    )


def check_other_lens(label: str, card: Dict[str, Any]) -> None:
    """A lens other than the shipped one is never passed off as it, nor given its checks, corpus or dtype."""
    check_every_card(label, card)
    check(card["id"] != JPP_LENS_FALLBACK["id"], f"{label}: called {card['id']!r}, the shipped lens's id")
    check(card["caution"] != JPP_LENS_FALLBACK["caution"], f"{label}: carries the shipped lens's caution")
    check(card["caution"].endswith(_OTHER_LENS_CAUTION), f"{label}: caution lacks the other-lens advice")
    check("layer 20" not in card["caution"], f"{label}: caution claims the study's layer bands")
    check("141" not in card["caution"] and "16-127" not in card["caution"],
          f"{label}: caution claims the study's fit window or the lengths it checked")
    check("WikiText" not in (card["fit_data"] or ""), f"{label}: fit_data claims WikiText: {card['fit_data']!r}")
    check("bf16" not in (card["fitted_on"] or ""), f"{label}: fitted_on claims a dtype: {card['fitted_on']!r}")


def check_shipped() -> None:
    print("1. the shipped file")
    card = card_for(SHIPPED_PAYLOAD)
    check_every_card("shipped", card)
    want = {
        "id": "jpp-base",
        "name": "J++",
        "fitted_on": "google/gemma-4-31B (base, bf16)",
        "applied_to": DEFAULT_W4A16_MODEL_ID,
        "method": "J++, R-lens LRP backward pass",
        "fit_data": "64 WikiText passages of 128 word-pieces",
        "source": SHIPPED_SOURCE,
        "file": SHIPPED_FILE,
        "caution": JPP_LENS_FALLBACK["caution"],
        "provenance": {
            "source": SHIPPED_SOURCE, "method": "jpp", "lrp_mode": "rlens", "hf_model_name": "google/gemma-4-31B",
            "n_prompts": 64, "max_seq_len": 128,
        },
    }
    for k, v in want.items():
        check(card[k] == v, f"shipped: {k} {card[k]!r} != {v!r}")
    # the study's bands, in the API's 0-based layers (erratum E49; E2 §6): checked from 20, only the order at 20-29
    check("from layer 20" in card["caution"] and "20-29" in card["caution"], f"shipped: caution {card['caution']!r}")
    check("30-49" in card["caution"] and "50-58" in card["caution"] and "40-57" in card["caution"],
          f"shipped: caution lacks the contrast range, the late pronoun flag or the label band {card['caution']!r}")
    # the card the server sends for another model it reads names that model and nothing else changes
    other = card_for(SHIPPED_PAYLOAD, model_id="google/gemma-4-31b-it")
    check(other["applied_to"] == "google/gemma-4-31b-it", "shipped: applied_to does not follow the model read")
    check({k: v for k, v in other.items() if k != "applied_to"} == {k: v for k, v in card.items() if k != "applied_to"},
          "shipped: the card changes with the model read beyond applied_to")
    # a runtime that has loaded nothing names no lens
    check(TheodoliteRuntime(device="cpu", load_jlens=False).lens_card() is None, "before load: a card was sent")

    check_cached_file(card)
    check_pages_fallback(card)
    check_caution_figures(card)


def check_cached_file(card: Dict[str, Any]) -> None:
    """The shipped file itself, when it is in the local cache: its provenance must be the one above."""
    try:
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(repo_id=LENS_REPO_ID, filename=JPP_LENS_FILENAME)
    except Exception as exc:  # an old huggingface_hub or an unreadable cache: nothing to compare
        print(f"  skip  the cached file (cannot look it up: {exc})")
        return
    if not isinstance(path, str) or not os.path.exists(path):
        print("  skip  the cached file (not in the local Hugging Face cache)")
        return
    before = len(failures)
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    got = _read_lens_provenance(payload)
    want = _read_lens_provenance(SHIPPED_PAYLOAD)
    check(got == want, f"the cached file's provenance {got} != the one this check assumes {want}")
    real = runtime_with(payload).lens_card()
    check(real == card, f"the cached file's card differs from the expected one: {real}")
    del payload
    if len(failures) == before:
        print(f"  ok    the cached file's own provenance gives the same card ({path})")


def check_pages_fallback(card: Dict[str, Any]) -> None:
    """BASE_LENS, the lens the pages name when no card has come, must say what the server says of the shipped file."""
    ts = (ROOT_DIR / "frontend" / "src" / "lib" / "lens.ts").read_text(encoding="utf-8")
    m = re.search(r"export const BASE_LENS\b[^=]*=\s*\{(.*?)\n\};", ts, re.S)
    check(m is not None, "frontend/src/lib/lens.ts: BASE_LENS not found")
    if m is None:
        return
    before = len(failures)
    fields = dict(re.findall(r'^\s*(\w+):\s*"((?:[^"\\]|\\.)*)",?\s*$', m.group(1), re.M))
    camel = {"id": "id", "name": "name", "fittedOn": "fitted_on", "appliedTo": "applied_to", "method": "method",
             "fitData": "fit_data", "source": "source", "file": "file"}
    for ts_key, card_key in camel.items():
        check(fields.get(ts_key) == card[card_key],
              f"BASE_LENS.{ts_key} {fields.get(ts_key)!r} != the server's {card_key} {card[card_key]!r}")
    if len(failures) == before:
        print("  ok    BASE_LENS in frontend/src/lib/lens.ts matches the card")


# The caution's sentence on length, and where each of its figures comes from.
FIT_SENTENCE = (
    "It was fitted on positions 16-127 (counting from 0) of 128-piece passages and the study checked it only on short "
    "texts (at most 141 word-pieces in its fidelity check), so how well it reads further in is unmeasured; from "
    "position 1024 on, 50 of the model's 60 layers no longer look back to the start of the text."
)
E3_DIR = ROOT_DIR / "experiments" / "results"


def check_caution_figures(card: Dict[str, Any]) -> None:
    """The shipped caution's last sentence, figure by figure, against the files it rests on."""
    before = len(failures)
    caution = card["caution"]
    check(caution.endswith(FIT_SENTENCE), f"shipped: the caution does not end with the sentence on length: {caution!r}")
    # the fit: the provenance's passage length, less the first 16 positions the fit skips (E3 report, Caveats)
    seq = card["provenance"]["max_seq_len"]
    check(seq == 128 and f"16-{seq - 1} (counting from 0) of {seq}-piece" in caution,
          f"shipped: fit window vs max_seq_len {seq}")
    report = (E3_DIR / "E3_lens_validation_REPORT.md").read_text(encoding="utf-8")
    check("the fit skips the first 16 positions" in report, "E3 report: no longer says the fit skips 16 positions")
    # the study's fidelity check: the longest text it read, from its own per-text summaries
    lengths = []
    for part in ("chat", "prose"):
        rows = json.loads((E3_DIR / f"E3_lens_validation_a_summary_{part}.json").read_text(encoding="utf-8"))
        lengths += [int(r["T"]) for r in rows]
    check(len(lengths) == 48 and max(lengths) == 141, f"E3 fidelity texts: {len(lengths)}, longest {max(lengths)}")
    # the window: the model's own config, when it is in the local cache
    try:
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(repo_id=DEFAULT_W4A16_MODEL_ID, filename="config.json")
    except Exception as exc:
        path = None
        print(f"  skip  the model's config (cannot look it up: {exc})")
    if isinstance(path, str) and os.path.exists(path):
        text_cfg = json.loads(Path(path).read_text(encoding="utf-8")).get("text_config", {})
        kinds = text_cfg.get("layer_types") or []
        window = text_cfg.get("sliding_window")
        check(window == 1024, f"the model's sliding_window is {window!r}, not the caution's 1024")
        check(len(kinds) == 60 and kinds.count("sliding_attention") == 50,
              f"the model's layer_types: {len(kinds)} layers, {kinds.count('sliding_attention')} sliding")
        # the runtime reads the same window from the loaded model's config (here a stand-in holding only the config)
        rt = runtime_with(SHIPPED_PAYLOAD)
        rt.hf_model = types.SimpleNamespace(config=types.SimpleNamespace(
            text_config=types.SimpleNamespace(sliding_window=window)))
        check(rt.sliding_window == 1024, f"runtime.sliding_window {rt.sliding_window!r}")
    else:
        print("  skip  the model's config (not in the local Hugging Face cache)")
    check(TheodoliteRuntime(device="cpu", load_jlens=False).sliding_window is None, "before load: a sliding window")
    if len(failures) == before:
        print("  ok    the caution's fit window, checked length and sliding window match their sources")


def check_incomplete() -> None:
    print("2. incomplete files")
    for label, payload in (
        ("no provenance", {"J": {}}),
        ("no payload dict (a bare matrix list)", [torch.zeros(1)]),
        ("an empty provenance", {"J": {}, "provenance": {}}),
        ("a provenance that is not a dict", {"J": {}, "provenance": "jpp on gemma"}),
    ):
        card = card_for(payload)
        check_other_lens(label, card)
        check(card["name"] == UNNAMED_LENS, f"{label}: name {card['name']!r} != {UNNAMED_LENS!r}")
        for k in ("id", "fitted_on", "method", "fit_data", "source"):
            check(card[k] is None, f"{label}: {k} {card[k]!r} should be None (not recorded)")
        check(card["caution"].startswith("The file does not say which model it was fitted on."), f"{label}: caution")
        check(all(v is None for v in card["provenance"].values()), f"{label}: provenance {card['provenance']}")

    card = card_for({"J": {}, "provenance": {"method": "jpp", "lrp_mode": "rlens"}, "n_prompts": 64})
    check_other_lens("a method without a model", card)
    check(card["name"] == "J++" and card["id"] == "jpp" and card["fitted_on"] is None, f"method only: {card}")
    check(card["method"] == "J++, R-lens LRP backward pass", f"method only: method {card['method']!r}")
    check(card["fit_data"] == "64 prompts", f"method only: fit_data {card['fit_data']!r}")
    check(card["caution"].startswith("The file does not say which model"), "method only: caution")

    # the base model named, but no method: not the shipped kind of file, so none of its record is filled in
    card = card_for({"J": {}, "provenance": {"source_config": {"hf_model_name": "google/gemma-4-31B"}}})
    check_other_lens("a model without a method", card)
    check(card["name"] == UNNAMED_LENS and card["id"] is None and card["method"] is None, f"model only: {card}")
    check(card["fitted_on"] == "google/gemma-4-31B (base)", f"model only: fitted_on {card['fitted_on']!r}")
    check(card["caution"].startswith(f"Fitted on google/gemma-4-31B, read here on {DEFAULT_W4A16_MODEL_ID}."),
          f"model only: caution {card['caution']!r}")

    # blank text and counts stored oddly: blanks count as not recorded, counts become plain whole numbers
    card = card_for({
        "J": {},
        "n_prompts": torch.tensor(200),
        "provenance": {
            "source": 1234,
            "method": "   ",
            "lrp_mode": "",
            "source_config": {"hf_model_name": "", "max_seq_len": np.int64(256)},
        },
    })
    check_other_lens("blank and odd-typed fields", card)
    check(card["name"] == UNNAMED_LENS and card["method"] is None, f"odd types: name/method {card['name']!r}")
    check(card["provenance"]["n_prompts"] == 200 and card["provenance"]["max_seq_len"] == 256, f"odd: {card}")
    check(card["provenance"]["source"] == "1234" and card["source"] == "1234", f"odd types: source {card['source']!r}")
    check(card["fit_data"] == "200 prompts of up to 256 word-pieces", f"odd types: fit_data {card['fit_data']!r}")
    check(card["fitted_on"] is None, f"odd types: a blank model name became {card['fitted_on']!r}")
    for bad in (torch.tensor([1, 2]), "many", -5, 0, 64.5, True, float("nan"), float("inf"), None, {}):
        card = card_for({"J": {}, "n_prompts": bad, "provenance": {"method": "jpp"}})
        check(card["provenance"]["n_prompts"] is None and card["fit_data"] is None,
              f"n_prompts {bad!r} was taken as a count: {card['provenance']['n_prompts']!r}")
    # a model name that is not text (it once crashed the name match, and so /api/health and every sweep)
    card = card_for({"J": {}, "provenance": {"method": "jpp", "source_config": {"hf_model_name": 31}}})
    check_other_lens("a model name that is a number", card)
    check(card["fitted_on"] == "31" and card["id"] == "jpp", f"numeric model name: {card}")


def check_other_lenses() -> None:
    print("3. other lenses")
    card = card_for({
        "J": {},
        "n_prompts": 200,
        "provenance": {
            "source": "someone/jpp-refit@abc123:gemma-4-31b-it/lens.pt",
            "method": "jpp",
            "lrp_mode": "rlens",
            "source_config": {"hf_model_name": "google/gemma-4-31B-it", "max_seq_len": 256},
        },
    })
    check_other_lens("J++ refitted on the instruct model", card)
    want = {
        "id": "jpp-instruct",
        "name": "J++",
        "fitted_on": "google/gemma-4-31B-it (instruct)",
        "method": "J++, R-lens LRP backward pass",
        "fit_data": "200 prompts of up to 256 word-pieces",
        "source": "someone/jpp-refit@abc123:gemma-4-31b-it/lens.pt",
    }
    for k, v in want.items():
        check(card[k] == v, f"instruct J++: {k} {card[k]!r} != {v!r}")
    check(card["caution"] == f"Fitted on google/gemma-4-31B-it, read here on {DEFAULT_W4A16_MODEL_ID}. "
          f"{_OTHER_LENS_CAUTION}", f"instruct J++: caution {card['caution']!r}")

    # another method, fitted on the base model: base-fitted, but not the shipped lens
    card = card_for({"J": {}, "n_prompts": 64, "provenance": {
        "method": "jlens", "source_config": {"hf_model_name": "google/gemma-4-31B"}}})
    check_other_lens("JLens on the base model", card)
    check(card["id"] == "jlens-base" and card["name"] == "jlens" and card["method"] == "jlens", f"jlens: {card}")
    check(card["fitted_on"] == "google/gemma-4-31B (base)", f"jlens: fitted_on {card['fitted_on']!r}")

    for hf, kind in (
        ("meta-llama/Llama-3.1-8B-Instruct", "instruct"),
        ("google/gemma-4-31b-it-qat-w4a16-ct", "instruct"),
        ("someorg/other-model", None),
        ("google/gemma-4-31b", None),  # lower case is not the base model's recorded name: said as it is, unjudged
    ):
        card = card_for({"J": {}, "provenance": {"method": "tuned", "lrp_mode": "zplus",
                                                 "source_config": {"hf_model_name": hf}}})
        check_other_lens(f"a 'tuned' lens on {hf}", card)
        check(card["fitted_on"] == (f"{hf} ({kind})" if kind else hf), f"{hf}: fitted_on {card['fitted_on']!r}")
        check(card["id"] == (f"tuned-{kind}" if kind else "tuned"), f"{hf}: id {card['id']!r}")
        check(card["method"] == "tuned, zplus LRP backward pass", f"{hf}: method {card['method']!r}")
    # a J++ file whose model name differs from the shipped one's only in case is still not passed off as it
    card = card_for({"J": {}, "provenance": {"method": "jpp",
                                             "source_config": {"hf_model_name": "google/gemma-4-31b"}}})
    check_other_lens("J++ on 'google/gemma-4-31b'", card)


# ------------------------------------------------------------------------------------------------- the vault's replies

STATEMENTS = (
    # (statement, the vault method it reaches, whether it goes through the lens, whether it reads neuron labels)
    ("DESCRIBE", "describe_vault", False, False),
    ("ROLLBACK", "rollback_patches", False, False),
    ("SELECT * FROM EDGES NEAREST 'she' WHERE layer = 35 LIMIT 3", "query_nearest_edges", True, True),
    ("SELECT * FROM POLES NEAREST 'he'", "query_nearest_edges", True, True),
    ("DESCRIBE POLE LAYER 53 SLOT 16383 POLARITY -", "inspect_bipolar_slot", True, True),
    ("DELETE FROM LAYER 3 WHERE slot = 5", "apply_patch_surgery", False, False),
    ("UPDATE LAYER 3 SLOT 5 SET object = 'she' STRENGTH 1.5", "apply_patch_surgery", True, False),
    ("INSERT INTO LAYER 3 (subject, object) VALUES ('nurse', 'she')", "apply_patch_surgery", True, False),
)


def stand_in_vault(rt: TheodoliteRuntime) -> LarqlVaultEngine:
    """The vault over `rt`, its GPU work replaced by replies that only say which method was reached."""
    vault = LarqlVaultEngine(rt)

    def stub(name: str) -> Callable[..., Dict[str, Any]]:
        return lambda **kw: {"reached": name, "args": sorted(kw), "status": "applied", "results": []}

    for name in ("describe_vault", "rollback_patches", "query_nearest_edges", "inspect_bipolar_slot",
                 "apply_patch_surgery"):
        setattr(vault, name, stub(name))
    return vault


def check_vault() -> None:
    print("4. the vault's statements")
    import backend.mcp_scientist as mcp_scientist

    for label, payload in (("shipped", SHIPPED_PAYLOAD), ("no provenance", {"J": {}})):
        rt = runtime_with(payload)
        vault = stand_in_vault(rt)
        mcp_scientist.bind_runtime(rt, vault)
        for statement, reached, through, labels in STATEMENTS:
            for via, run in (("execute_lql", vault.execute_lql),
                             ("query_larql_vault", mcp_scientist.query_larql_vault)):
                reply = run(statement)
                what = f"{label}, {via}({statement!r})"
                check(reply.get("reached") == reached, f"{what}: reached {reply.get('reached')!r}, not {reached}")
                if through:
                    check(list(reply)[-1] == "lens" and reply["lens"] == rt.lens_card(),
                          f"{what}: the lens is not named last")
                    rest = {k: v for k, v in reply.items() if k != "lens"}
                    # the stand-ins send no label_gate: execute_lql adds it, just before the lens, where labels are read
                    want = ["reached", "args", "status", "results"] + (["label_gate"] if labels else [])
                    check(list(rest) == want, f"{what}: the reply changed: {rest}")
                    if labels:
                        check(reply["label_gate"] == LABEL_GATE_CARD, f"{what}: label_gate is not the gate's card")
                else:
                    check("lens" not in reply and "label_gate" not in reply, f"{what}: names a lens it does not use")
    # leave no stand-in bound, so a later call in this process would load the real runtime
    mcp_scientist._SHARED_RUNTIME = None
    mcp_scientist._SHARED_VAULT = None


def main() -> int:
    check_shipped()
    check_incomplete()
    check_other_lenses()
    check_vault()
    if failures:
        print(f"\nlens card: {len(failures)} of {passed + len(failures)} checks FAILED")
        return 1
    print(f"\nlens card: all {passed} checks hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())

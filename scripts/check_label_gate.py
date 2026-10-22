#!/usr/bin/env python3
"""A CPU-only check of the label gate (backend/label_gate.py): no model, no GPU, no running server.

The gate marks each neuron label the server sends (`label_check`) and states its rule (`LABEL_GATE_CARD`), closing the
study's conditions 2 and 3 for labels (REPORT §3.3, erratum E47; ROADMAP §ii). Layers here count from 0, as the code
and the study's reports do (page layer = code layer + 1). It checks:
1. the rule, over every layer 0-59, every count 0-10 (and none), under the shipped lens (jpp-base), another lens and
   no lens, edited or not: the bands, the reasons (below_layer_40 below 40; shares_nothing at none of ten, below 59,
   under jpp-base only, edited or not; writes_nothing for a silenced neuron, with no count), held_back,
   lens_validated, and the keys in order; and the card's figures against their sources (N1C_matrix.json's AUC for the
   count, N1C_gate_cut.json's record of the cut, the report lines it cites);
2. the ranking itself on small stand-in tensors: `top_ids_from_shortlist` equals a full ranking of the exact scores
   (ties broken toward the lower id), and `label_top_ids` equals a full fp32 softcapped top ten computed here;
3. the vault on a stand-in runtime (small tensors): `edits_in_force` keeps the latest edit per neuron; every NEAREST row
   and DESCRIBE POLE face carries the label_check its words give, edited where an edit is in force; NEAREST and
   DESCRIBE POLE replies carry label_gate just before lens, UPDATE and INSERT do not; at layer 59 (the identity lens)
   the two readings share ten; a neuron whose column is zero (as DELETE leaves it) writes nothing: no count, held back
   (its tied "first ten" would otherwise share ten); and the MCP tool inspect_pole's keys come in the WebSocket's
   order;
4. the server's measure against the study's, where the study's static caches are at hand (experiments/results/N1A_cache,
   gitignored: here, or in the main checkout this worktree belongs to; skipped with a note otherwise): the cached fp32
   softcapped logits are taken back through the softcap, rounded to bf16 as the server's GEMM rounds them, capped
   again, shortlisted and ranked by `top_ids_from_shortlist`; for every face of every sampled neuron within layers
   40-57 its first ten must equal the study's, and its shared count the study's overlap (below 40 mismatches are
   reported, not failed: ties at the softcap break as they may there); and the in-band AUC of the count for causal
   P@10 >= 0.3 must be N1C_matrix.json's 0.8432 to 1e-4;
5. the committed results the pages quote and read: experiments/results/N1C_gate_cut.json (the re-analysis of N1's data,
   experiments/scripts/N1C_09_gate_cut.py) has the pinned sha256, which its provenance sidecar also records, the sidecar
   names the script as committed and the committed inputs as they are, and, where the study's caches are at hand, the
   analysis run again here gives the same result; the figures frontend/src/lib/labels.ts quotes (GATE_FIGURES,
   OVERLAP_RECORD) and the card's cut_record are the file's and N1C_matrix.json's; frontend/src/fixtures/recordedLabelChecks.ts, which
   experiments/scripts/N1G_01_recorded_label_checks.py writes, lists every labelled face of the demonstration sighting
   and every row of the vault's 24 recorded searches, and either every check is null (not yet run on the GPU) or the
   module names experiments/results/N1G_recorded_label_checks.json by its sha256, as the sidecar does, and each check
   is the file's; the sidecar's script and inputs are as committed (a note where the server code it names has changed
   since), each recorded face is the neuron and face the fixture or the vault's recorded list has at its key, each
   recorded check follows the rule at its layer, and the counts README.md and ROADMAP.md quote are the file's summary;
6. the pages' own gate (frontend/src/lib/labels.ts), bundled with the frontend's esbuild and run under node: the
   server's check is parsed and its verdict used, a label below page 41 is held back whatever its check says, a
   missing check is "unchecked" and holds back by the layer alone, shares_nothing and writes_nothing hold back, an
   edited neuron's hold does not quote the study's figures as its warrant, the gilt mark and the count's record go by
   N1C_gate_cut.json's by_overlap, and the recorded checks reach the pages (the sighting's and the vault's, by key).

Run: .venv/bin/python -I scripts/check_label_gate.py   (exits 0 when every check holds)
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import hashlib
import importlib.util
import json
import re
import subprocess
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch

from backend.label_gate import (
    BAND,
    BANDS,
    CUT,
    HOLD_REASONS,
    LABEL_GATE_CARD,
    LABEL_MIN_LAYER,
    SAMPLED,
    SHORTLIST,
    TOPK,
    label_check,
    label_top_ids,
    top_ids_from_shortlist,
    writes_nothing,
)
from backend.larql_vault import LarqlVaultEngine, PatchOperationRecord
from backend.theodolite_math import FINAL_LOGIT_SOFTCAP, gemma4_rms_norm, unembed_gemma4

RESULTS = ROOT_DIR / "experiments" / "results"
CHECK_KEYS = ["shared_top10", "held_back", "reasons", "band", "sampled_layer", "in_measured_band", "lens_validated",
              "edited"]
CARD_KEYS = ["rule", "layers_count_from", "min_layer", "measure", "cut", "held_back_if_shared_top10_at_most",
             "validated", "cut_record", "sources", "caution"]

failures: List[str] = []
passed = 0


def check(cond: bool, what: str) -> None:
    global passed
    if cond:
        passed += 1
    else:
        failures.append(what)
        print(f"  FAILED: {what}")


# --------------------------------------------------------------------------------------------------------------------
# 1. the rule
# --------------------------------------------------------------------------------------------------------------------

def want_band(l: int) -> str:
    # written out from the study's sample (N1_REPORT.md:7): 10, 20, 30 at chance; 40-57 held up; nothing at 58 or 59
    if l <= 9:
        return "below_sampled"
    if l <= 30:
        return "chance_where_sampled"
    if l <= 39:
        return "not_sampled"
    if l <= 57:
        return "band"
    if l == 58:
        return "above_band"
    return "one_reading"


def ids_sharing(n: int) -> tuple:
    """Two ten-id lists that share exactly `n` ids."""
    lens = list(range(10))
    eye = list(range(n)) + list(range(100, 110 - n))
    return lens, eye


def check_rule() -> None:
    print("1. the rule, and the card")
    check(LABEL_MIN_LAYER == 40 and BAND == (40, 57) and SAMPLED == (10, 20, 30, 40, 45, 50, 55, 57)
          and TOPK == 10 and SHORTLIST == 64 and CUT == 0, "the constants")
    check(set(BANDS) == {want_band(l) for l in range(60)}, f"the bands {BANDS}")
    n_cases = 0
    for l in range(60):
        for shared in [None] + list(range(11)):
            lens, eye = ids_sharing(shared) if shared is not None else (None, list(range(10)))
            for lens_id in ("jpp-base", "jpp-instruct", None):
                for edited in (False, True):
                    c = label_check(l, lens, eye, lens_id, edited=edited)
                    n_cases += 1
                    what = f"layer {l}, shared {shared}, lens {lens_id}, edited {edited}"
                    reasons = (["below_layer_40"] if l < 40 else []) + (
                        ["shares_nothing"] if shared == 0 and l < 59 and lens_id == "jpp-base" else []
                    )
                    ok = (
                        list(c) == CHECK_KEYS
                        and c["shared_top10"] == shared
                        and c["reasons"] == reasons
                        and c["held_back"] is bool(reasons)
                        and c["band"] == want_band(l)
                        and c["sampled_layer"] is (l in (10, 20, 30, 40, 45, 50, 55, 57))
                        and c["in_measured_band"] is (40 <= l <= 57)
                        and c["lens_validated"] is (lens_id == "jpp-base" and not edited)
                        and c["edited"] is edited
                    )
                    if not ok:
                        check(False, f"{what}: {c}")
    check(True, f"{n_cases} cases of the rule")
    # a missing reading on either side gives no count, and so no shares_nothing
    check(label_check(45, [1, 2], [], "jpp-base")["shared_top10"] is None, "an empty list gives no count")
    check(label_check(45, list(range(12)), list(range(10, 22)), "jpp-base")["shared_top10"] == 0,
          "only the first ten of each list count")
    check(label_check(59, list(range(10)), list(range(10, 20)), "jpp-base")["held_back"] is False,
          "layer 59 never shares nothing as a reason")
    # a silenced neuron (a zero write vector): no count, held back, whatever the ids, the lens and the layer
    check(HOLD_REASONS == ("below_layer_40", "shares_nothing", "writes_nothing"), f"the hold reasons {HOLD_REASONS}")
    for l, lens_id, want in ((45, "jpp-base", ["writes_nothing"]), (45, "jpp-instruct", ["writes_nothing"]),
                             (20, "jpp-base", ["below_layer_40", "writes_nothing"]), (59, None, ["writes_nothing"])):
        c = label_check(l, list(range(10)), list(range(10)), lens_id, edited=True, silenced=True)
        check(list(c) == CHECK_KEYS and c["shared_top10"] is None and c["reasons"] == want and c["held_back"] is True
              and c["edited"] is True and c["lens_validated"] is False, f"silenced at layer {l} under {lens_id}: {c}")
    z = torch.zeros(3, 8, dtype=torch.bfloat16)
    z[1, 4] = 1e-3
    check(writes_nothing(z) == [True, False, True] and writes_nothing(z[1]) == [False],
          "writes_nothing marks exactly the zero vectors")

    check(list(LABEL_GATE_CARD) == CARD_KEYS, f"card keys {list(LABEL_GATE_CARD)}")
    check(LABEL_GATE_CARD["min_layer"] == LABEL_MIN_LAYER and LABEL_GATE_CARD["layers_count_from"] == 0
          and LABEL_GATE_CARD["held_back_if_shared_top10_at_most"] == CUT and LABEL_GATE_CARD["rule"] == "n1-gate-1",
          "the card's rule")
    check(json.loads(json.dumps(LABEL_GATE_CARD)) == LABEL_GATE_CARD, "the card is plain JSON")
    check("instrument's choice" in LABEL_GATE_CARD["cut"] and "no threshold" in LABEL_GATE_CARD["cut"],
          "the card says the cut is the instrument's, not the study's")
    check("base-fitted J++" in LABEL_GATE_CARD["validated"] and "random neurons" in LABEL_GATE_CARD["validated"],
          "the card names the lens and the sample the AUC was measured for")
    check("graded score, not the cut" in LABEL_GATE_CARD["validated"], "the card says the AUC 0.84 is the count's, not the cut's")
    gc = json.loads((RESULTS / "N1C_gate_cut.json").read_text())["band"]
    cut0 = gc["cuts"][str(CUT)]
    want_rec = {"layers": list(BAND), "faces": gc["all"]["n"], "held_back": cut0["held_back"],
                "held_back_sound": cut0["held_back_sound"], "kept": cut0["kept"], "kept_sound": cut0["kept_sound"],
                "auc": round(gc["auc_cut_p10_ge_0.3"], 2), "source": "experiments/results/N1C_gate_cut.json (band)"}
    check(LABEL_GATE_CARD["cut_record"] == want_rec, f"the card's cut_record {LABEL_GATE_CARD['cut_record']} != {want_rec}")
    check("writes_nothing" in LABEL_GATE_CARD["caution"] and "edited label that shares none" in LABEL_GATE_CARD["caution"],
          "the card's caution names the silenced neurons and the edited ones the cut still holds back")
    # the figures, from their sources
    m = json.loads((RESULTS / "N1C_matrix.json").read_text())
    auc = m["q"]["static_confidence_predictor"]["jv_nv_top10_overlap"]["auc_late_target_p10_ge_0.3"]
    check(f"AUC {auc:.2f}" in LABEL_GATE_CARD["validated"], f"the card's AUC against N1C_matrix.json ({auc:.4f})")
    n1 = (RESULTS / "N1_REPORT.md").read_text().splitlines()
    check("AUC" in " ".join(n1[123:127]) and "0.84" in " ".join(n1[123:127]) and "L40–57" in " ".join(n1[123:127]),
          "N1_REPORT.md:124-127 holds the AUC within L40-57")
    check("(L10, 20, 30, 40, 45, 50, 55, 57)" in n1[6], "N1_REPORT.md:7 names the layers sampled")
    n1v = (RESULTS / "N1V_robustness.md").read_text().splitlines()
    check("AUC for P@10 ≥ 0.3 is 0.84" in " ".join(n1v[110:115]), "N1V_robustness.md:111-115 holds the fresh AUC")
    report = (ROOT_DIR / "experiments" / "REPORT.md").read_text()
    check("**Erratum E47**" in report and "base-fitted J++ labels" in report, "REPORT.md's erratum E47")


# --------------------------------------------------------------------------------------------------------------------
# 2. the ranking on stand-in tensors
# --------------------------------------------------------------------------------------------------------------------

def full_top(scores: torch.Tensor, k: int) -> List[List[int]]:
    """The first k ids by score, ties toward the lower id: the reference the shortlist must reproduce."""
    out = []
    for row in scores.tolist():
        out.append(sorted(range(len(row)), key=lambda i: (-row[i], i))[:k])
    return out


def check_ranking() -> None:
    print("2. the ranking")
    g = torch.Generator().manual_seed(3)
    exact = torch.randn(6, 500, generator=g) * 8
    exact[0, [7, 70, 300]] = 40.0  # ties at the top, as the softcap makes them
    exact = torch.tanh(exact / FINAL_LOGIT_SOFTCAP) * FINAL_LOGIT_SOFTCAP
    bf16 = exact.bfloat16().float()
    got = top_ids_from_shortlist(bf16, lambda ids: exact.gather(-1, ids), k=10)
    check(got == full_top(exact, 10), f"the shortlist ranks as the full list: {got[0]} vs {full_top(exact, 10)[0]}")
    check(got[0][:3] == [7, 70, 300], f"ties break toward the lower id: {got[0][:3]}")
    got1 = top_ids_from_shortlist(bf16[0], lambda ids: exact[:1].gather(-1, ids), k=10)
    check(got1 == got[:1], "one row given flat")

    d, v = 16, 400
    w = (torch.randn(v, d, generator=g) * 0.5).bfloat16()
    fnw = (1 + 0.1 * torch.randn(d, generator=g)).bfloat16()
    vecs = (torch.randn(20, d, generator=g) * 1e-2).bfloat16()
    logits = unembed_gemma4(vecs, fnw, w)
    ref = torch.tanh((gemma4_rms_norm(vecs, fnw).float() @ w.float().T) / FINAL_LOGIT_SOFTCAP) * FINAL_LOGIT_SOFTCAP
    got = label_top_ids(vecs, fnw, w, logits, chunk=7)
    check(got == full_top(ref, 10), "label_top_ids equals a full fp32 softcapped top ten (rows in chunks)")
    check(label_top_ids(vecs[3], fnw, w, logits[3]) == got[3:4], "one vector given flat")


# --------------------------------------------------------------------------------------------------------------------
# 3. the vault on a stand-in runtime
# --------------------------------------------------------------------------------------------------------------------

SD, SV, SFF = 16, 300, 40  # the stand-in's model width, vocabulary and neurons per layer


class StandInRuntime:
    """Just what the vault's label-reading statements touch, on small random tensors."""

    def __init__(self, lens_id: Optional[str] = "jpp-base") -> None:
        g = torch.Generator().manual_seed(11)
        self.device = torch.device("cpu")
        self.model_id = "stand-in"
        self._lens_id = lens_id
        self.tokenizer = types.SimpleNamespace(decode=lambda ids: "".join(f"·{int(i)}" for i in ids))
        self._lm = (torch.randn(SV, SD, generator=g) * 0.5).bfloat16()
        self._fnw = (1 + 0.1 * torch.randn(SD, generator=g)).bfloat16()
        self.jpp_lenses = {l: (torch.eye(SD) + 0.3 * torch.randn(SD, SD, generator=g)).bfloat16() for l in range(59)}
        self.jpp_lenses[59] = torch.eye(SD, dtype=torch.bfloat16)
        self._w_down = {l: (torch.randn(SD, SFF, generator=g) * 0.1).bfloat16() for l in (20, 45, 59)}
        self._post_ffn_norm_weights: Dict[int, torch.Tensor] = {45: (1 + 0.2 * torch.randn(SD, generator=g)).bfloat16()}

    @property
    def final_norm_weight(self) -> torch.Tensor:
        return self._fnw

    @property
    def lm_head_weight(self) -> torch.Tensor:
        return self._lm

    def lens_card(self) -> Dict[str, Any]:
        return {"id": self._lens_id}

    def resolve_concept_token_id(self, concept: str):
        return 5, "·5"

    def get_full_w_down(self, l: int) -> torch.Tensor:
        return self._w_down[l]

    def slice_w_down(self, l: int, idx: torch.Tensor) -> torch.Tensor:
        return self._w_down[l][:, idx].T.contiguous()

    def slice_w_gate(self, l: int, idx: torch.Tensor) -> torch.Tensor:
        return self.slice_w_down(l, idx)

    slice_w_up = slice_w_gate


def ids_of(tokens: List[Dict[str, Any]]) -> List[int]:
    return [int(t["token"].lstrip("·")) for t in tokens]


def expected_check(rt: StandInRuntime, l: int, n: int, s: float, edited: bool, lens_id: Optional[str]) -> Dict[str, Any]:
    """A face's check computed here, from the full fp32 softcapped logits of its two readings."""
    w = rt._w_down[l][:, n].float()
    if l in rt._post_ffn_norm_weights:
        w = w * rt._post_ffn_norm_weights[l].float()
    raw = (s * w).bfloat16().unsqueeze(0)
    jpp = raw @ rt.jpp_lenses[l].T

    def top(v: torch.Tensor) -> List[int]:
        x = gemma4_rms_norm(v, rt._fnw).float() @ rt._lm.float().T
        return full_top(torch.tanh(x / FINAL_LOGIT_SOFTCAP) * FINAL_LOGIT_SOFTCAP, 10)[0]

    return label_check(l, top(jpp), top(raw), lens_id, edited=edited)


def check_vault() -> None:
    print("3. the vault, on a stand-in runtime")
    import backend.mcp_scientist as mcp_scientist

    rt = StandInRuntime()
    vault = LarqlVaultEngine(rt)  # type: ignore[arg-type]

    def rec(pid: int, l: int, n: int, op: str = "UPDATE") -> PatchOperationRecord:
        return PatchOperationRecord(pid, op, l, n, 1.0, "she", 1.0, f"{op} {l} {n}", float(pid), 1.0, 2.0)

    check(vault.edits_in_force() == [], "no edit in force at first")
    vault.active_patches = [rec(1, 45, 3), rec(2, 20, 4), rec(3, 45, 3, "DELETE"), rec(4, 45, 9)]
    got = [(p.patch_id, p.layer_idx, p.neuron_idx) for p in vault.edits_in_force()]
    check(got == [(2, 20, 4), (3, 45, 3), (4, 45, 9)], f"edits_in_force: the latest per neuron, in patch order: {got}")

    for lens_id in ("jpp-base", "jpp-instruct"):
        rt._lens_id = lens_id
        for l, n in ((45, 3), (45, 7), (20, 4), (59, 1)):
            reply = vault.execute_lql(f"DESCRIBE POLE LAYER {l} SLOT {n} POLARITY -")
            what = f"DESCRIBE POLE {l}:{n} under {lens_id}"
            check(list(reply)[-2:] == ["label_gate", "lens"] and reply["label_gate"] == LABEL_GATE_CARD,
                  f"{what}: label_gate just before lens")
            edited = (l, n) in {(45, 3), (20, 4)}
            for face, s in (("north_pole", 1.0), ("south_pole", -1.0)):
                c = reply[face]["label_check"]
                want = expected_check(rt, l, n, s, edited, lens_id)
                check(c == want, f"{what} {face}: {c} != {want}")
                check(list(reply[face]) == ["polarity", "j_vindex_tokens", "naive_tokens", "label_check"],
                      f"{what} {face}: keys {list(reply[face])}")
            check(reply["active_pole"] is reply["south_pole"], f"{what}: the active face is the asked one")
            if l == 59:
                c = reply["north_pole"]["label_check"]
                check(c["shared_top10"] == 10 and c["band"] == "one_reading" and not c["held_back"],
                      f"{what}: one reading at layer 59: {c}")
                check(ids_of(reply["north_pole"]["j_vindex_tokens"]) == ids_of(reply["north_pole"]["naive_tokens"]),
                      f"{what}: the lens's words are the naked eye's at layer 59")
            if l == 20:
                check(reply["north_pole"]["label_check"]["reasons"][:1] == ["below_layer_40"],
                      f"{what}: held back below layer 40")
        reply = vault.execute_lql("SELECT * FROM EDGES NEAREST 'she' WHERE layer = 45 LIMIT 6")
        check(list(reply)[-2:] == ["label_gate", "lens"], "NEAREST: label_gate just before lens")
        check(len(reply["results"]) == 6, "NEAREST: six rows")
        for row in reply["results"]:
            s = 1.0 if row["polarity"] == "+" else -1.0
            n = row["neuron_idx"]
            want = expected_check(rt, 45, n, s, (45, n) in {(45, 3), (45, 9)}, lens_id)
            check(row["label_check"] == want, f"NEAREST row {n}{row['polarity']} under {lens_id}: "
                                              f"{row['label_check']} != {want}")
            check(list(row)[-1] == "label_check" and len(row["j_vindex_top_tokens"]) == 5
                  and len(row["naive_lens_top_tokens"]) == 5, "NEAREST row: five words a side, label_check last")
    rt._lens_id = "jpp-base"

    # the statements that read no labels are untouched (their stand-ins say which method was reached)
    for name in ("apply_patch_surgery", "describe_vault", "rollback_patches"):
        setattr(vault, name, lambda _n=name, **kw: {"reached": _n})
    for statement in ("UPDATE LAYER 3 SLOT 5 SET object = 'she'", "INSERT INTO LAYER 3 (subject, object) VALUES "
                      "('nurse', 'she')"):
        reply = vault.execute_lql(statement)
        check(list(reply) == ["reached", "lens"], f"{statement}: no label_gate ({list(reply)})")
    for statement in ("DESCRIBE", "ROLLBACK", "DELETE FROM LAYER 3 WHERE slot = 5"):
        check("label_gate" not in vault.execute_lql(statement), f"{statement}: no label_gate")

    # the MCP tool answers with the WebSocket's keys, in its order
    del vault.describe_vault, vault.rollback_patches, vault.apply_patch_surgery
    mcp_scientist.bind_runtime(rt, vault)  # type: ignore[arg-type]
    try:
        ws = vault.execute_lql("DESCRIBE POLE LAYER 45 SLOT 7")
        mcp = mcp_scientist.inspect_pole(layer=45, neuron_idx=7, polarity=1.0)
        check(list(mcp) == list(ws), f"inspect_pole's keys are DESCRIBE POLE's: {list(mcp)} vs {list(ws)}")
        check(mcp["north_pole"]["label_check"] == ws["north_pole"]["label_check"], "inspect_pole's checks")
    finally:
        mcp_scientist._SHARED_RUNTIME = None
        mcp_scientist._SHARED_VAULT = None

    # a neuron silenced as DELETE leaves it (a zero column): every logit ties, so without the silenced mark both
    # readings' "first ten" would be the same lowest ids, and the label would share ten; it has no count, and is held back
    saved = rt._w_down[45][:, 11].clone()
    rt._w_down[45][:, 11] = 0
    try:
        reply = vault.execute_lql("DESCRIBE POLE LAYER 45 SLOT 11")
        for face in ("north_pole", "south_pole"):
            c = reply[face]["label_check"]
            check(c["shared_top10"] is None and c["reasons"] == ["writes_nothing"] and c["held_back"] is True,
                  f"a silenced neuron's {face}: {c}")
        ids_j = ids_of(reply["north_pole"]["j_vindex_tokens"])
        check(ids_j == ids_of(reply["north_pole"]["naive_tokens"]),
              "a silenced neuron's two readings tie the same way (the artefact the mark keeps out)")
        mcp_scientist.bind_runtime(rt, vault)  # type: ignore[arg-type]
        mcp = mcp_scientist.inspect_pole(layer=45, neuron_idx=11, polarity=1.0)
        check(mcp["north_pole"]["label_check"] == reply["north_pole"]["label_check"], "inspect_pole marks it the same way")
    finally:
        rt._w_down[45][:, 11] = saved
        mcp_scientist._SHARED_RUNTIME = None
        mcp_scientist._SHARED_VAULT = None
        mcp_scientist._SHARED_RUNTIME = None
        mcp_scientist._SHARED_VAULT = None


# --------------------------------------------------------------------------------------------------------------------
# 4. the server's measure against the study's
# --------------------------------------------------------------------------------------------------------------------

def study_cache() -> Optional[Path]:
    """experiments/results/N1A_cache here, or in the main checkout of the worktree this is (gitignored either way)."""
    here = RESULTS / "N1A_cache"
    if (here / "static_L40.pt").exists():
        return here
    try:
        common = subprocess.run(["git", "-C", str(ROOT_DIR), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except Exception:
        return None
    main = Path(common).parent / "experiments" / "results" / "N1A_cache"
    return main if (main / "static_L40.pt").exists() else None


def as_the_server_rounds(y: torch.Tensor) -> torch.Tensor:
    """fp32 softcapped logits as the server's bf16 GEMM gives them: back through the softcap, rounded to bf16, capped
    again (saturated scores stay saturated)."""
    z = (y / FINAL_LOGIT_SOFTCAP).clamp(-1 + 1e-7, 1 - 1e-7)
    x = FINAL_LOGIT_SOFTCAP * torch.atanh(z)
    return torch.tanh(x.bfloat16().float() / FINAL_LOGIT_SOFTCAP) * FINAL_LOGIT_SOFTCAP


def check_against_study() -> None:
    print("4. the server's measure against the study's")
    cache = study_cache()
    if cache is None:
        print("  skipped: the study's static caches (experiments/results/N1A_cache, gitignored) are not here")
        return
    print(f"  reading {cache}")
    neu = json.loads((RESULTS / "N1_neurons.json").read_text())
    pp = json.loads((RESULTS / "N1C_per_pole.json").read_text())
    poles, target = pp["poles"], np.array(pp["full_JV->C"])
    shared: List[float] = []
    layers: List[int] = []
    for l in [int(x) for x in neu["layers"]]:
        st = torch.load(cache / f"static_L{l}.pt", weights_only=True)
        mism = 0
        n_faces = 0
        for (ll, n, s) in poles:
            if ll != l:
                continue
            ni = st["neurons"].index(n)
            ids, study = {}, {}
            for kind in ("jpp", "naive"):
                y = s * st[kind][ni].float()
                study[kind] = set(y.topk(10).indices.tolist())
                ids[kind] = top_ids_from_shortlist(as_the_server_rounds(y)[None], lambda c, y=y: y[c], k=10)[0]
                mism += set(ids[kind]) != study[kind]
            c = label_check(l, ids["jpp"], ids["naive"], "jpp-base")
            n_faces += 1
            if BAND[0] <= l <= BAND[1]:
                if c["shared_top10"] != len(study["jpp"] & study["naive"]):
                    check(False, f"L{l} n{n}{'+' if s > 0 else '-'}: shared {c['shared_top10']} != the study's "
                                 f"{len(study['jpp'] & study['naive'])}")
            shared.append(c["shared_top10"] / 10)
            layers.append(l)
        if BAND[0] <= l <= BAND[1]:
            check(mism == 0, f"L{l}: {mism} of {2 * n_faces} first-ten lists differ from the study's")
            print(f"  L{l}: all {2 * n_faces} first-ten lists are the study's")
        else:
            print(f"  L{l}: {mism} of {2 * n_faces} first-ten lists differ from the study's (ties at the softcap; "
                  "not failed below layer 40)")
    feat, lay = np.array(shared), np.array(layers)
    late = (lay >= BAND[0]) & (lay <= BAND[1])
    y = target[late] >= 0.3
    pos, neg = feat[late][y], feat[late][~y]
    auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    m = json.loads((RESULTS / "N1C_matrix.json").read_text())
    recorded = m["q"]["static_confidence_predictor"]["jv_nv_top10_overlap"]["auc_late_target_p10_ge_0.3"]
    check(abs(auc - recorded) < 1e-4 and abs(recorded - 0.8432) < 1e-4,
          f"in-band AUC {auc:.4f} against N1C_matrix.json's {recorded:.4f}")
    print(f"  in-band AUC for P@10 >= 0.3: {auc:.4f} (N1C_matrix.json {recorded:.4f})")


# --------------------------------------------------------------------------------------------------------------------
# 5. the committed results the pages quote and read
# --------------------------------------------------------------------------------------------------------------------

# the sha256 of experiments/results/N1C_gate_cut.json as committed: a re-run that changes a byte must change this too
GATE_CUT_SHA256 = "cf31dd81908064b70290f7515c9b1321fcfbef0d96130d20e0cd867e2f23f68c"
FRONTEND = ROOT_DIR / "frontend" / "src"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ts_object(text: str, name: str) -> Any:
    """The value of `export const <name>: … = <JSON with trailing commas>;` in a generated module."""
    m = re.search(rf"export const {name}:.*?= (null|\{{[^\n]*\}}|\{{\n.*?\n\}});\n", text, re.S)
    if not m:
        raise ValueError(f"{name} not found")
    return json.loads(re.sub(r",(\s*[}\]])", r"\1", m.group(1)))


def check_committed() -> None:
    print("5. the committed results the pages quote and read")
    out, side_p = RESULTS / "N1C_gate_cut.json", RESULTS / "N1C_gate_cut.provenance.json"
    got = sha256_of(out)
    side = json.loads(side_p.read_text())
    check(got == GATE_CUT_SHA256, f"N1C_gate_cut.json sha256 {got} != the pinned {GATE_CUT_SHA256}")
    check(side["result_sha256"] == got, "the sidecar records the result's sha256")
    script = ROOT_DIR / side["script"]
    check(side["script"] == "experiments/scripts/N1C_09_gate_cut.py" and side["script_sha256"] == sha256_of(script),
          "the sidecar's script is the committed one")
    for inp in side["inputs"]:
        if not inp["gitignored"]:
            check(sha256_of(ROOT_DIR / inp["path"]) == inp["sha256"], f"the sidecar's input {inp['path']} is as committed")
    if side["label_gate_sha256"] != sha256_of(ROOT_DIR / "backend" / "label_gate.py"):
        print("  note: backend/label_gate.py has changed since the re-analysis ran (the recomputation below decides)")
    result = json.loads(out.read_text())
    band = result["band"]
    check(band["auc_shared10_p10_ge_0.3"] == band["auc_recorded_N1C_matrix"], "the re-analysis counts the study's overlap")
    cache = study_cache()
    if cache is None:
        print("  skipped the recomputation: the study's static caches are not here")
    else:
        for inp in side["inputs"]:
            if inp["gitignored"]:
                p = cache / Path(inp["path"]).name
                check(p.exists() and sha256_of(p) == inp["sha256"], f"the cache file {p.name} is the one the re-analysis read")
        spec = importlib.util.spec_from_file_location("N1C_09_gate_cut", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        again = json.loads(json.dumps(mod.analyse(cache)))
        check(again == result, "the re-analysis run again gives N1C_gate_cut.json")
        print(f"  the re-analysis run again from {cache} gives the committed file")
    print(f"  L40-57: {band['share_nothing']['n']} of {band['all']['n']} faces share none of ten, "
          f"{band['share_nothing']['n_sound']} of them sound; {band['share_some']['n_sound']} of {band['share_some']['n']} "
          f"others sound")

    # the figures the pages quote
    labels_ts = (FRONTEND / "lib" / "labels.ts").read_text()
    m = re.search(r"export const GATE_FIGURES = \{(.*?)\} as const;", labels_ts, re.S)
    figs = dict(re.findall(r"(\w+): \"?([\d.]+)\"?,", m.group(1))) if m else {}
    auc = json.loads((RESULTS / "N1C_matrix.json").read_text())["q"]["static_confidence_predictor"][
        "jv_nv_top10_overlap"]["auc_late_target_p10_ge_0.3"]
    want = {"auc": f"{auc:.2f}", "aucIn100": str(round(auc * 100)), "shareNothing": str(band["share_nothing"]["n"]),
            "shareNothingSound": str(band["share_nothing"]["n_sound"]), "shareSome": str(band["share_some"]["n"]),
            "shareSomeSound": str(band["share_some"]["n_sound"]), "band": str(band["all"]["n"]),
            "bandSound": str(band["all"]["n_sound"]), "cutAuc": f"{band['auc_cut_p10_ge_0.3']:.2f}"}
    check(figs == want, f"labels.ts GATE_FIGURES {figs} != the results' {want}")
    m = re.search(r"export const OVERLAP_RECORD[^=]*= \[(.*?)\];", labels_ts, re.S)
    rec = [[int(a), int(b)] for a, b in re.findall(r"\[(\d+), (\d+)\]", m.group(1))] if m else []
    want_rec = [[band["by_overlap"][str(k)]["n"], band["by_overlap"][str(k)]["n_sound"]] for k in range(11)]
    check(rec == want_rec, f"labels.ts OVERLAP_RECORD {rec} != the results' by_overlap {want_rec}")
    consts = {k: v for k, v in re.findall(r"export const (LABEL_MIN_IDX|ONE_READING_IDX) = (\d+);", labels_ts)}
    check(consts == {"LABEL_MIN_IDX": str(LABEL_MIN_LAYER), "ONE_READING_IDX": "59"}, f"labels.ts constants {consts}")
    check(f"BAND_IDX: readonly [number, number] = [{BAND[0]}, {BAND[1]}]" in labels_ts
          and f"SAMPLED_IDX: readonly number[] = [{', '.join(map(str, SAMPLED))}]" in labels_ts,
          "labels.ts names the band and the sampled layers as the server does")

    # the recorded checks the pages read
    module = (FRONTEND / "fixtures" / "recordedLabelChecks.ts").read_text()
    run = ts_object(module, "RECORDED_LABEL_CHECKS_RUN")
    fx = ts_object(module, "FIXTURE_LABEL_CHECKS")
    vx = ts_object(module, "VAULT_LABEL_CHECKS")
    fixture = json.loads((FRONTEND / "fixtures" / "e6-demo-sighting.json").read_text())
    check(list(fx) == sorted(fixture["poleLabels"]["lens"], key=lambda k: tuple(int(x) for x in k.split(":")))
          and len(fx) == 75, f"the module lists the sighting's {len(fixture['poleLabels']['lens'])} labelled faces")
    lists = json.loads((RESULTS / "E6_tool_audit_04_vault.partial.json").read_text())["lists"]
    want_v = {k: [f"{int(r['n'])}{'+' if r['pol'] >= 0 else '-'}" for r in v] for k, v in lists.items()}
    check({k: list(v) for k, v in vx.items()} == want_v and len(vx) == 24 and sum(len(v) for v in vx.values()) == 240,
          "the module lists the 24 recorded vault searches, every row's face")
    checks = list(fx.values()) + [c for v in vx.values() for c in v.values()]
    if run is None:
        check(all(c is None for c in checks), "not yet run: every recorded check is null")
        print("  recordedLabelChecks.ts: not yet run on the GPU, every check null (the pages hold back by layer alone)")
    else:
        res_p = ROOT_DIR / run["results"]
        res_side = json.loads((RESULTS / "N1G_recorded_label_checks.provenance.json").read_text())
        check(res_p.exists() and sha256_of(res_p) == run["sha256"] == res_side["result_sha256"],
              "the module names the recorded checks' results by their sha256, as the sidecar does")
        res = json.loads(res_p.read_text())
        check(all(fx[k] == v["label_check"] for k, v in res["fixture"]["faces"].items()) and
              len(res["fixture"]["faces"]) == len(fx), "the module's sighting checks are the results'")
        check(all(vx[k][f"{r['neuron']}{'+' if r['face'] > 0 else '-'}"] == r["label_check"]
                  for k, s in res["vault"]["searches"].items() for r in s["rows"]), "the module's vault checks are the results'")
        check(all(c is not None and list(c) == CHECK_KEYS for c in checks), "every recorded check is a full label_check")
        # the sidecar's script and inputs as they stand; the server code it ran with, noted where it has changed
        check(sha256_of(ROOT_DIR / res_side["script"]) == res_side["script_sha256"],
              "the recorded checks' sidecar names the script as committed")
        for inp in res_side["inputs"]:
            check(sha256_of(ROOT_DIR / inp["path"]) == inp["sha256"], f"the recorded checks' input {inp['path']} is as committed")
        changed = [k for k, v in res_side["code_sha256"].items() if sha256_of(ROOT_DIR / k) != v]
        if changed:
            print(f"  note: changed since the recorded checks ran: {', '.join(changed)} (the rule check below decides)")
        # each recorded face is the label the pages apply it to
        neu, acts = fixture["poleNeurons"], fixture["poleActs"]
        bad = [k for k, v in res["fixture"]["faces"].items()
               if (lambda l, r: v["neuron"] != neu[str(l)][r] or v["face"] != (1 if acts[str(l)][r] >= 0 else -1)
                   or v["layer"] != l or v["rank"] != r)(*(int(x) for x in k.split(":")))]
        check(not bad, f"the recorded sighting faces are the fixture's neuron and face at their keys: {bad[:5]}")
        bad = [k for k, s in res["vault"]["searches"].items()
               if [(r["neuron"], 1 if r["face"] > 0 else -1) for r in s["rows"]]
               != [(int(x["n"]), 1 if x["pol"] >= 0 else -1) for x in lists[k]] or s["layer"] != int(k.split("@")[1])]
        check(not bad, f"the recorded vault rows are the recorded searches' neurons and faces: {bad[:5]}")
        # each recorded check follows the rule at its layer (reconstructed from its count, as verify_e2e does)
        lens_id = run.get("lensId")
        keyed = [(int(k.split(":")[0]), c) for k, c in fx.items()]
        keyed += [(int(k.split("@")[1]), c) for k, v in vx.items() for c in v.values()]
        bad = []
        for l, c in keyed:
            k = c["shared_top10"]
            lens, eye = ids_sharing(k) if k is not None else (None, list(range(10)))
            if label_check(l, lens, eye, lens_id, edited=c["edited"], silenced="writes_nothing" in c["reasons"]) != c:
                bad.append((l, c))
        check(not bad, f"every recorded check follows the rule at its layer: {bad[:3]}")
        # the counts the docs quote
        sm = res["summary"]
        fx_n, fx_h, v_n, v_h = sm["fixture"]["faces"], sm["fixture"]["held_back"], sm["vault"]["faces"], sm["vault"]["held_back"]
        check(fx_h == sum(c["held_back"] for c in fx.values()) and v_h == sum(c["held_back"] for v in vx.values() for c in v.values()),
              "the results' summary counts the module's held-back checks")
        readme = re.sub(r"\s+", " ", (ROOT_DIR / "README.md").read_text())
        roadmap = re.sub(r"\s+", " ", (ROOT_DIR / "ROADMAP.md").read_text())
        check(f"hold back {fx_h} of the sighting's {fx_n} labelled faces and {v_h} of the vault's {v_n} rows" in readme,
              f"README.md quotes {fx_h} of {fx_n} and {v_h} of {v_n}")
        check(f"{fx_h} of the sighting's {fx_n} labelled faces and {v_h} of the vault's {v_n} rows are held back" in roadmap,
              f"ROADMAP.md quotes {fx_h} of {fx_n} and {v_h} of {v_n}")
        print(f"  recordedLabelChecks.ts: {len(checks)} checks from {run['results']} ({run['runAt']})")


# --------------------------------------------------------------------------------------------------------------------
# 6. the pages' own gate, run under node
# --------------------------------------------------------------------------------------------------------------------

PAGES_ENTRY = r"""
import * as L from "./labels";
const raw = (k: number | null, held: boolean, reasons: string[], extra: Record<string, unknown> = {}) => ({
  shared_top10: k, held_back: held, reasons, band: "band", sampled_layer: false, in_measured_band: true,
  lens_validated: true, edited: false, ...extra,
});
const g = (idx: number, c: unknown) => L.labelGate(L.layerIdx(idx), L.parseLabelCheck(c));
const pick = (x: L.Gate) => ({ held: x.heldBack, reasons: x.reasons, c3: x.condition3, k: x.sharedTop10, edited: x.edited });
const out: Record<string, unknown> = {};
out.parseNull = L.parseLabelCheck(null);
out.parseJunk = L.parseLabelCheck({ shared_top10: 3 });
out.parseReasons = L.parseLabelCheck(raw(0, true, ["shares_nothing", "made_up"]))?.reasons;
out.below_unchecked = pick(g(35, null));
out.band_unchecked = pick(g(45, null));
out.below_shown_by_server = pick(g(35, raw(6, false, [])));
out.band_shares_nothing = pick(g(45, raw(0, true, ["shares_nothing"])));
out.band_server_verdict = pick(g(45, raw(4, true, [])));
out.band_shown = pick(g(45, raw(4, false, [])));
out.one_reading = pick(g(59, raw(10, false, [], { band: "one_reading" })));
const ed = g(45, raw(0, true, ["shares_nothing"], { lens_validated: false, edited: true }));
out.edited = { ...pick(ed), rubric: L.heldRubric(ed), shared: L.sharedLine(ed) };
const other = g(45, raw(0, false, [], { lens_validated: false }));
out.other_lens = { ...pick(other), shared: L.sharedLine(other) };
const sil = g(45, raw(null, true, ["writes_nothing"], { lens_validated: false, edited: true }));
out.silenced = { ...pick(sil), tag: L.heldTag(sil), shared: L.sharedLine(sil), line: L.heldLine(sil) };
out.above = [0, 1, 2, 3, 4, 10].map((k) => L.overlapAboveBand(k));
out.line1 = L.overlapLine(g(45, raw(1, false, [])));
out.line1_other_lens = L.overlapLine(g(45, raw(1, false, [], { lens_validated: false })));
out.line1_page59 = L.overlapLine(g(58, raw(1, false, [], { band: "above_band" })));
const fx: Record<string, unknown> = {};
for (const key of ["35:0", "53:0", "53:4", "58:0"]) {
  const [l, r] = key.split(":").map(Number);
  fx[key] = pick(L.labelGate(L.layerIdx(l), L.fixtureCheck(l, r)));
}
out.fixture = fx;
out.recordedChecked = L.RECORDED_CHECKED;
out.vaultHeld40 = L.recordedVaultHeld(40);
console.log(JSON.stringify(out));
"""


def check_pages() -> None:
    print("6. the pages' own gate, under node")
    import shutil
    esbuild, node = ROOT_DIR / "frontend" / "node_modules" / ".bin" / "esbuild", shutil.which("node")
    check(esbuild.exists() and node is not None, "the frontend's esbuild and node are at hand (npm install in frontend/)")
    if not esbuild.exists() or node is None:
        return
    bundle = subprocess.run([str(esbuild), "--bundle", "--platform=node", "--format=esm", "--loader=ts",
                             "--log-level=error", "--sourcefile=check_label_gate_entry.ts"],
                            input=PAGES_ENTRY, cwd=FRONTEND / "lib", capture_output=True, text=True, timeout=120)
    check(bundle.returncode == 0, f"esbuild could not bundle lib/labels.ts:\n{bundle.stderr}")
    if bundle.returncode != 0:
        return
    ran = subprocess.run([node, "--input-type=module"], input=bundle.stdout, capture_output=True, text=True, timeout=120)
    check(ran.returncode == 0, f"lib/labels.ts did not run under node:\n{ran.stderr}")
    if ran.returncode != 0:
        return
    o = json.loads(ran.stdout.strip().splitlines()[-1])
    G = lambda held, reasons, c3, k, edited=False: {"held": held, "reasons": reasons, "c3": c3, "k": k, "edited": edited}
    check(o["parseNull"] is None and o["parseJunk"] is None, "a missing or malformed check parses to none")
    check(o["parseReasons"] == ["shares_nothing"], f"unknown reasons are dropped: {o['parseReasons']}")
    check(o["below_unchecked"] == G(True, ["below_layer_40"], "unchecked", None), f"below 41, no check: {o['below_unchecked']}")
    check(o["band_unchecked"] == G(False, [], "unchecked", None), f"in the band, no check: {o['band_unchecked']}")
    check(o["below_shown_by_server"] == G(True, ["below_layer_40"], "applied", 6),
          f"below 41 the layer holds a label back whatever the check says: {o['below_shown_by_server']}")
    check(o["band_shares_nothing"] == G(True, ["shares_nothing"], "applied", 0), f"shares nothing: {o['band_shares_nothing']}")
    check(o["band_server_verdict"]["held"] is True, "the server's held_back is the pages' verdict")
    check(o["band_shown"] == G(False, [], "applied", 4), f"a shown label: {o['band_shown']}")
    check(o["one_reading"] == G(False, [], "one-reading", 10), f"page 60: {o['one_reading']}")
    e = o["edited"]
    check(e["held"] is True and e["c3"] == "unvalidated" and e["edited"] is True
          and "never measured" in e["rubric"] and "31" not in e["rubric"] and "AUC" not in e["rubric"]
          and "still holds" in e["shared"], f"an edited neuron held back as a precaution, without the study's figures: {e}")
    check(o["other_lens"]["held"] is False and o["other_lens"]["c3"] == "unvalidated"
          and "not a verdict" in o["other_lens"]["shared"], f"another lens: a count, not a verdict: {o['other_lens']}")
    sv = o["silenced"]
    check(sv["held"] is True and sv["k"] is None and sv["reasons"] == ["writes_nothing"] and "silenced" in sv["tag"]
          and sv["shared"] and "writes nothing" in sv["line"], f"a silenced neuron: {sv}")
    band = json.loads((RESULTS / "N1C_gate_cut.json").read_text())["band"]
    rate = band["all"]["n_sound"] / band["all"]["n"]
    want_above = [band["by_overlap"][str(k)]["n_sound"] / band["by_overlap"][str(k)]["n"] > rate for k in (0, 1, 2, 3, 4, 10)]
    check(o["above"] == want_above, f"the gilt mark by the count's record: {o['above']} != {want_above}")
    b1 = band["by_overlap"]["1"]
    check(f"only {b1['n_sound']} times in {b1['n']}" in o["line1"] and f"{band['all']['n_sound']} in {band['all']['n']}" in o["line1"],
          f"one of ten quotes its record: {o['line1']!r}")
    check(o["line1_other_lens"] == "" and o["line1_page59"] == "",
          "the record is quoted only through the base-fitted lens, within the band it was measured in")
    res_p = RESULTS / "N1G_recorded_label_checks.json"
    if o["recordedChecked"] and res_p.exists():
        res = json.loads(res_p.read_text())
        for key, got in o["fixture"].items():
            c = res["fixture"]["faces"][key]["label_check"]
            check(got["held"] is c["held_back"] and got["k"] == c["shared_top10"],
                  f"the recorded sighting's {key} reaches the pages as recorded: {got} vs {c}")
        rows = [r for s in res["vault"]["searches"].values() if s["layer"] >= 40 for r in s["rows"]]
        want_v = {"rows": len(rows), "held": sum(r["label_check"]["held_back"] for r in rows)}
        check(o["vaultHeld40"] == want_v, f"the recorded vault rows held back at 41 and above: {o['vaultHeld40']} != {want_v}")
    print(f"  lib/labels.ts under node: the gate, the wording and the recorded checks as the files say")


def main() -> int:
    check_rule()
    check_ranking()
    check_vault()
    check_against_study()
    check_committed()
    check_pages()
    if failures:
        print(f"\nlabel gate: {len(failures)} of {passed + len(failures)} checks FAILED")
        return 1
    print(f"\nlabel gate: all {passed} checks hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())

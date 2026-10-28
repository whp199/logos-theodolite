#!/usr/bin/env python3
"""A CPU-only check of the vault's voice (backend/larql_vault.py `_heard` and `VoiceRequest`, backend/engine_runtime.py
`write_voice`, the MCP tool backend/mcp_scientist.py `query_larql_vault`): no model, no GPU, no running server.

An edit (DELETE, UPDATE, INSERT) can be heard: given a voice, the vault has the model write its answer to one
question the likeliest way just before the edit and again just after, under the GPU lock the route holds. Here the
runtime is a stand-in: a real `TheodoliteRuntime` (so `write_voice`, `_written_step` and the overlays the real
`apply_patch_surgery` writes are the real code) over tiny weights (8 wide, 64 word-pieces), whose writing loop
(interp_engine's `generate_stream`, replaced) gives logits that depend on the column overlays in force, as the real
model's forward does (engine_runtime `_w4a16_forward`). The real `LarqlVaultEngine.execute_lql` is driven directly,
through the app's real routes (`fastapi.testclient.TestClient`, HTTP and WebSocket, the lifespan replaced), and through
the real MCP server (`fastmcp.Client`, in memory, on the same stand-in). It checks:
1. `write_voice` itself: greedy, stopping at an end mark or the length, its generator closed, step 0's five
   likeliest and the watched word's odds and rank, the log-softmax it returns, and its plain-words refusals (a
   question too long for one go, an answer of no word-pieces);
2. without a voice, every statement replies with exactly today's keys (UPDATE and INSERT: status, patch,
   active_patches_count, lens; DELETE without the lens; ROLLBACK as before, as scripts/verify_e2e_ui_and_mcp.py
   asserts), and each patch record keeps its old keys, in order, with voice (null), edit_kind and replaces_patch
   after them; DESCRIBE adds only edit_method;
3. with a voice: the order is voice before, the edit (its overlay set), voice after, nothing between, all under the
   route's lock; the reply is today's keys plus "voice", after "patch" and just before "lens"; the voice is the
   record's (patch.voice, and DESCRIBE's active_patches); before and after differ where the edit changes the
   writing, parted_at names the first word-piece that differs, the watched word is the edit's target (none for
   DELETE), and KL(before || after) is >= 0, and 0 when the edit leaves the writing alone;
4. replaces_patch and edits_in_force_before count the latest edit per neuron (A2's `edits_in_force`), so a second
   edit of a neuron names the first and an edit heard after it does not count the replaced one;
5. a voice that cannot be written before the edit leaves the vault untouched (no record, no overlay) and is refused
   in plain words; one that fails after the edit keeps the edit in force, with after null and the error said;
6. a voice sent with DESCRIBE, NEAREST, DESCRIBE POLE or ROLLBACK is ignored (nothing written, the reply as
   without it); ROLLBACK clears every record and so every voice;
7. the routes: HTTP and WebSocket pass the voice with its defaults (a chat, 12 word-pieces), and write both answers
   in a worker thread, off the event loop, still under the GPU lock; a voice the request
   model refuses comes back over the WebSocket as larql_result ok false in plain words, and over HTTP as 422, with no
   edit made; the server's VoiceRequest is the vault's own;
8. the MCP tool, through fastmcp.Client: `query_larql_vault(lql_query, voice)` replies as before without a voice and
   with one replies with it as the routes do (VoiceRequest's defaults, "voice" before "lens", kept on the record),
   both answers written in a worker thread under the tools' own lock and not the server's; a voice it refuses is
   refused in plain words (one that is not an object, by FastMCP's check of the tool's types), and a failed voice
   before edits nothing; inspect_pole asked while the voice before is written waits, and is logged after the voice
   after, never between, its read made off the event loop; a call cancelled meanwhile (as the MCP SDK cancels one)
   keeps its turn until its thread ends, its edit made and heard; bound as the server binds it, the tools' lock is a
   no-op and the same two calls interleave (the server's _GPU_LOCK keeps them apart there); every tool is async and
   runs in its turn, its parameters as a client sees them its function's, and the voice's schema VoiceRequest's.

Run: .venv/bin/python -I scripts/check_vault_voice.py   (exits 0 when every check holds)
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import asyncio
import inspect
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import anyio
import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient
from fastmcp import Client
from fastmcp.exceptions import ToolError
from interp_engine.sampling import resolve_sampling

import backend.engine_runtime as engine_runtime
import backend.larql_vault as larql_vault
import backend.mcp_scientist as mcp_scientist
import backend.server as server
from backend.engine_runtime import MAX_SWEEP_POSITIONS, TheodoliteRuntime
from backend.larql_vault import EDIT_METHOD, LarqlVaultEngine, PatchOperationRecord, kl_nats, parted_at, voice_spec
from backend.theodolite_math import D_FF, N_LAYERS

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
# a stand-in runtime: the real write_voice over a writing loop that hears the overlays
# --------------------------------------------------------------------------------------------------------------------

D = 8  # the stand-in's model width
V = 64  # its vocabulary
EOS = 1
END = 50  # an end of turn the stand-in's end_of_turn_ids gives (as 106 <turn|> is on the checkpoint)
WORDS = {"six": 30, "she": 31, "he": 32, "nurse": 33}
# the keys of an answer as write_voice gives it
VOICE_ANSWER_KEYS = ["token_ids", "token_strs", "text", "stop_reason", "n_prompt", "steps", "first_top", "watch"]
# a heard edit's voice, as _heard gives it
HEARD_KEYS = ["prompt", "chat", "system", "assistant_prefix", "max_new_tokens", "edits_in_force_before", "before",
              "after", "first_step", "parted_at", "elapsed_ms", "error"]
# the patch record's keys as first sent, in order; the new ones come after
OLD_PATCH_KEYS = ["patch_id", "op_type", "layer_idx", "neuron_idx", "polarity", "target_concept", "strength",
                  "statement", "timestamp", "orig_norm", "new_norm"]
NEW_PATCH_KEYS = ["voice", "edit_kind", "replaces_patch"]


class FakeTokenizer:
    eos_token_id = EOS
    bos_token = "<bos>"

    def decode(self, ids: Sequence[int], clean_up_tokenization_spaces: Optional[bool] = None) -> str:
        return "".join(f"·{int(i)}" for i in ids)


class FakeProj:
    """A W4A16 linear's packed weights (`[out, in // 8]` int32, `[out, in // 32]` scales) and its own overlays."""

    def __init__(self, packed: torch.Tensor, scale: torch.Tensor) -> None:
        self.weight_packed = packed
        self.weight_scale = scale
        self._memit_col_overlays: Dict[int, torch.Tensor] = {}
        self._memit_row_overlays: Dict[int, torch.Tensor] = {}


class FakeMLP:
    def __init__(self, down: FakeProj, gate: FakeProj, up: FakeProj) -> None:
        self.down_proj, self.gate_proj, self.up_proj = down, gate, up


def packed_weights(out_features: int, in_features: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(-(2**31), 2**31 - 1, (out_features, in_features // 8), generator=g, dtype=torch.int64)
    return packed.to(torch.int32), torch.full((out_features, in_features // 32), 0.05, dtype=torch.bfloat16)


@dataclass
class FakeStep:
    token_id: int
    token_str: str
    logits: torch.Tensor


class FakeEager:
    """What write_voice asks of interp_engine's EagerModel: its device and its sampler settings."""

    def __init__(self, rt: "StandInRuntime") -> None:
        self.rt = rt
        self.device = torch.device("cpu")

    def sampling_settings(self, **kw: Any) -> Any:
        return resolve_sampling(None, **kw)


class StandInRuntime(TheodoliteRuntime):
    """A real runtime over tiny weights. Its writing answers 10, 11, 12, ... (one id a step); with n column overlays
    in force, step 0 leans n * 0.5 nats toward "six" (its first piece unchanged) and from step 1 on it writes 40 + n,
    so an edit changes the answer from its second word-piece. `deaf` makes it ignore the overlays."""

    def __init__(self) -> None:
        super().__init__(device="cpu", load_jlens=False)
        self.tokenizer = FakeTokenizer()
        self.eager_model = FakeEager(self)  # type: ignore[assignment]
        g = torch.Generator().manual_seed(0)
        self._lm_head_weight = torch.randn(V, D, generator=g)
        self._final_norm_weight = torch.ones(D)
        self.jpp_lenses = {l: torch.eye(D) for l in range(N_LAYERS)}
        # down_proj is [D, D_FF], so every neuron index the vault may clamp to has a column; one set of weights is
        # shared by the 60 layers, each with its own overlays
        down = packed_weights(D, D_FF, 1)
        small = packed_weights(D, 32, 2)
        self.mlps = [FakeMLP(FakeProj(*down), FakeProj(*small), FakeProj(*small)) for _ in range(N_LAYERS)]
        self.deaf = False
        self.fail_calls: Dict[int, Exception] = {}  # the n-th write_voice call (from 1) raises this
        self.n_voice_calls = 0
        self.events: List[Tuple[str, Any]] = []  # ("voice", overlays, lock held) | ("edit", statement) | ("pole", layer)
        self.voice_on_loop: List[bool] = []  # for each write_voice call: was it made on a running event loop's thread?
        self.voice_turn: List[bool] = []  # for each write_voice call: did the MCP tools' own lock (mcp_scientist) hold?
        self.pole_on_loop: List[bool] = []  # for each DESCRIBE POLE or inspect_pole read: on an event loop's thread?
        self.gate: Optional[threading.Event] = None  # when set, the next write_voice call waits for it, once
        self.composed: List[Dict[str, Any]] = []
        self.closed = 0

    def get_mlp_module(self, layer_idx: int) -> Any:
        return self.mlps[layer_idx]

    def n_overlays(self) -> int:
        return sum(len(m.down_proj._memit_col_overlays) for m in self.mlps)

    def lens_card(self) -> Dict[str, Any]:
        return {"id": "stand-in"}

    def resolve_concept_token_id(self, concept_or_token: Any) -> Tuple[int, str]:
        tid = WORDS.get(str(concept_or_token).strip(), 20)
        return tid, self.tokenizer.decode([tid])

    def compose_prompt(self, text: str, *, chat: bool = False, system: Optional[str] = None,
                       assistant_prefix: str = "", thinking: bool = False) -> str:
        self.composed.append({"text": text, "chat": chat, "system": system, "assistant_prefix": assistant_prefix})
        return (f"<chat|{system or ''}>" if chat else "") + text + assistant_prefix

    def tokenize_prompt(self, prompt: str, prepend_bos: bool = True) -> List[int]:
        return [2] + [3] * len(prompt)

    def end_of_turn_ids(self) -> List[int]:
        return [EOS, END]

    def write_voice(self, prompt_text: str, **kw: Any) -> Tuple[Dict[str, Any], torch.Tensor]:
        self.n_voice_calls += 1
        self.events.append(("voice", self.n_overlays(), server._GPU_LOCK.locked()))
        try:
            asyncio.get_running_loop()
            self.voice_on_loop.append(True)
        except RuntimeError:
            self.voice_on_loop.append(False)
        turn = mcp_scientist._TURN
        self.voice_turn.append(turn is not None and turn[1].locked())
        gate, self.gate = self.gate, None
        if gate is not None:
            gate.wait(30)  # held while the check asks for another tool meanwhile
        if self.n_voice_calls in self.fail_calls:
            raise self.fail_calls[self.n_voice_calls]
        return super().write_voice(prompt_text, **kw)

    def step_logits(self, prompt_ids: torch.Tensor, i: int) -> torch.Tensor:
        n = 0 if self.deaf else self.n_overlays()
        logits = torch.linspace(0.0, 1.0, V)
        if prompt_ids.shape[-1] > 25 and i == 2:
            logits[END] = 9.0  # a long question: the answer ends at its third word-piece
        elif i == 0 or n == 0:
            logits[10 + i] = 5.0
        else:
            logits[40 + n] = 9.0
        if i == 0:
            logits[WORDS["six"]] += 0.5 * n
        return logits


def fake_generate_stream(model: FakeEager, tokens: torch.Tensor, *, max_tokens: int, temperature: float, top_k: int,
                         top_p: float, presence_penalty: float, stop_at_eos: bool, n_logprobs: int,
                         seed: Optional[int]) -> Iterator[FakeStep]:
    """interp_engine's loop, greedy: the argmax each step, stopping after its eos."""
    rt = model.rt
    rt.last_knobs = dict(temperature=temperature, top_k=top_k, top_p=top_p, presence_penalty=presence_penalty,
                         stop_at_eos=stop_at_eos, seed=seed)
    try:
        for i in range(max_tokens):
            logits = rt.step_logits(tokens, i)
            tid = int(torch.argmax(logits).item())
            yield FakeStep(tid, rt.tokenizer.decode([tid]), logits)
            if stop_at_eos and tid == EOS:
                return
    finally:
        rt.closed += 1


engine_runtime.ie_generate_stream = fake_generate_stream  # type: ignore[assignment]


def stand_in_vault() -> Tuple[StandInRuntime, LarqlVaultEngine]:
    rt = StandInRuntime()
    vault = LarqlVaultEngine(rt)
    real_apply = vault.apply_patch_surgery

    def apply(**kw: Any) -> Dict[str, Any]:
        rt.events.append(("edit", kw.get("statement")))
        return real_apply(**kw)

    vault.apply_patch_surgery = apply  # type: ignore[assignment]
    vault.cli_versions = lambda: {"vindex_cli": "stand-in", "larql_cli": "stand-in"}  # type: ignore[assignment]
    # NEAREST (and so INSERT's choice of neuron) reads the full gate and down matrices: a stand-in answer here
    vault.query_nearest_edges = lambda **kw: {  # type: ignore[assignment]
        "command": "NEAREST", "results": [{"neuron_idx": 7, "polarity_num": 1.0}]}

    def pole(**kw: Any) -> Dict[str, Any]:  # DESCRIBE POLE and inspect_pole, logged
        rt.events.append(("pole", kw["layer_idx"]))
        try:
            asyncio.get_running_loop()
            rt.pole_on_loop.append(True)
        except RuntimeError:
            rt.pole_on_loop.append(False)
        return {"layer": kw["layer_idx"], "poles": []}

    vault.inspect_bipolar_slot = pole  # type: ignore[assignment]
    return rt, vault


VOICE = {"prompt": "How many legs?", "chat": True, "system": None, "assistant_prefix": "", "max_new_tokens": 4}


# --------------------------------------------------------------------------------------------------------------------
# 1. write_voice
# --------------------------------------------------------------------------------------------------------------------


def check_write_voice() -> None:
    print("1. write_voice: greedy, stopped, closed, and refused in plain words")
    rt = StandInRuntime()
    voice, logp = rt.write_voice("Q?", max_new_tokens=4, stop_token_ids=[END], watch_token_id=WORDS["six"])
    check(list(voice) == VOICE_ANSWER_KEYS, f"the answer's keys {list(voice)}")
    check(voice["token_ids"] == [10, 11, 12, 13] and voice["stop_reason"] == "length" and voice["n_prompt"] == 3,
          f"greedy to the length: {voice['token_ids']} {voice['stop_reason']}")
    check(voice["text"] == "·10·11·12·13" and voice["token_strs"] == ["·10", "·11", "·12", "·13"], "its words")
    check(rt.last_knobs == dict(temperature=0.0, top_k=0, top_p=1.0, presence_penalty=0.0, stop_at_eos=True, seed=None),
          f"every knob sent as the greedy value it means: {rt.last_knobs}")
    check(rt.closed == 1, "the writing loop is closed")
    want0 = F.log_softmax(rt.step_logits(torch.zeros(1, 4), 0), dim=-1)
    check(logp.dtype == torch.float32 and logp.shape == (V,) and torch.equal(logp, want0),
          "step 0's log-softmax, float32, returned")
    check([s["index"] for s in voice["steps"]] == [0, 1, 2, 3]
          and set(voice["steps"][0]) == {"index", "token", "token_id", "model_prob", "model_rank"}
          and all(s["model_rank"] == 1 for s in voice["steps"]), f"the steps {voice['steps'][0]}")
    check(abs(voice["steps"][0]["model_prob"] - float(want0[10].exp())) < 1e-6, "step 0's model prob")
    top = voice["first_top"]
    check(len(top) == 5 and top[0]["token_id"] == 10 and set(top[0]) == {"token_id", "token", "prob"}
          and abs(sum(t["prob"] for t in top) - float(want0.exp().topk(5).values.sum())) < 1e-6, f"first_top {top}")
    w = voice["watch"]
    rank = int((want0 > want0[WORDS["six"]]).sum()) + 1
    check(w == {"token_id": WORDS["six"], "token": "·30", "prob": float(want0[WORDS["six"]].exp()), "rank": rank},
          f"the watched word {w}")
    check(rt.write_voice("Q?", max_new_tokens=2)[0]["watch"] is None, "no word watched: watch null")
    long_q = "x" * 30
    voice, _ = rt.write_voice(long_q, max_new_tokens=12, stop_token_ids=[END])
    check(voice["token_ids"] == [10, 11, END] and voice["stop_reason"] == "end_mark", f"an end mark stops it: {voice}")
    voice, _ = rt.write_voice(long_q, max_new_tokens=12)
    check(voice["stop_reason"] == "length" and len(voice["token_ids"]) == 12, "without the end mark it writes on")
    for text, n, words in (
        ("x" * (MAX_SWEEP_POSITIONS - 4), 4, f"more than the {MAX_SWEEP_POSITIONS} the instrument writes in one go"),
        ("Q?", 0, "at least one word-piece"),
    ):
        try:
            rt.write_voice(text, max_new_tokens=n)
            check(False, f"write_voice({len(text)} characters, {n}) accepted")
        except ValueError as exc:
            check(words in str(exc), f"its refusal: {exc}")
    rt.write_voice("x" * (MAX_SWEEP_POSITIONS - 5), max_new_tokens=4)
    check(True, f"exactly {MAX_SWEEP_POSITIONS} positions are written")
    check(parted_at([1, 2, 3], [1, 2, 3]) is None and parted_at([1, 2, 3], [1, 5, 3]) == 1
          and parted_at([1, 2], [1, 2, 3]) == 2 and parted_at([], []) is None, "parted_at")
    a = F.log_softmax(torch.randn(V), dim=-1)
    b = F.log_softmax(torch.randn(V), dim=-1)
    check(kl_nats(a, a) == 0.0 and kl_nats(a, b) > 0.0
          and abs(kl_nats(a, b) - float(F.kl_div(b, a, log_target=True, reduction="sum"))) < 1e-5, "kl_nats")


# --------------------------------------------------------------------------------------------------------------------
# 2-6. the vault, called directly
# --------------------------------------------------------------------------------------------------------------------


def edit_keys(statement: str) -> List[str]:
    return ["status", "patch", "active_patches_count"] + (["lens"] if statement.split()[0] in ("UPDATE", "INSERT") else [])


UPD = "UPDATE LAYER 35 SLOT 100 SET object = 'six' STRENGTH 3.0 POLARITY +"
INS = "INSERT INTO LAYER 20 (subject, object) VALUES ('nurse', 'she')"
DEL = "DELETE FROM LAYER 3 WHERE slot = 5"


def check_without_voice() -> None:
    print("2. without a voice: today's replies")
    rt, vault = stand_in_vault()
    check([f.name for f in fields(PatchOperationRecord)] == OLD_PATCH_KEYS + NEW_PATCH_KEYS, "the record's fields")
    for st in (UPD, INS, DEL):
        reply = vault.execute_lql(st)
        check(list(reply) == edit_keys(st), f"{st.split()[0]}: keys {list(reply)}")
        p = reply["patch"]
        check(list(p) == OLD_PATCH_KEYS + NEW_PATCH_KEYS and p["voice"] is None and p["edit_kind"] == "one_neuron_column"
              and p["replaces_patch"] is None, f"{st.split()[0]}: the record {p}")
    check(rt.n_voice_calls == 0 and [e[0] for e in rt.events] == ["edit"] * 3, "nothing written without a voice")
    check(len(rt.mlps[20].down_proj._memit_col_overlays) == 1 and 7 in rt.mlps[20].down_proj._memit_col_overlays,
          "INSERT overlays the neuron its gate chose, in down_proj")
    desc = vault.execute_lql("DESCRIBE")
    check(list(desc)[-1] == "edit_method" and desc["edit_method"] == EDIT_METHOD and "not MEMIT" in EDIT_METHOD
          and desc["active_patches_count"] == 3, f"DESCRIBE adds only edit_method: {list(desc)[-3:]}")
    rolled = vault.execute_lql("ROLLBACK")
    check(list(rolled) == ["status", "cleared_patches", "active_patches_count"] and rolled["cleared_patches"] == 3,
          f"ROLLBACK keys {list(rolled)}")


def check_with_voice() -> None:
    print("3. with a voice: before, the edit, after; the reply and the record")
    rt, vault = stand_in_vault()
    reply = vault.execute_lql(UPD, voice=VOICE)
    check(list(reply) == ["status", "patch", "active_patches_count", "voice", "lens"], f"keys {list(reply)}")
    check([e[:2] for e in rt.events] == [("voice", 0), ("edit", UPD), ("voice", 1)],
          f"the order: before (no overlay), the edit, after (its overlay): {rt.events}")
    v = reply["voice"]
    check(list(v) == HEARD_KEYS, f"the voice's keys {list(v)}")
    check(reply["patch"]["voice"] == v and vault.active_patches[0].voice == v, "the voice is the record's")
    check(vault.execute_lql("DESCRIBE")["active_patches"][0]["voice"] == v, "DESCRIBE's records carry it")
    check({k: v[k] for k in VOICE} == VOICE, "the question echoed as heard")
    check(rt.composed[-1] == {"text": VOICE["prompt"], "chat": True, "system": None, "assistant_prefix": ""},
          f"composed as compose_prompt sends it: {rt.composed[-1]}")
    b, a = v["before"], v["after"]
    check(list(b) == VOICE_ANSWER_KEYS and list(a) == VOICE_ANSWER_KEYS, "before and after are write_voice's answers")
    check(b["token_ids"] == [10, 11, 12, 13] and a["token_ids"] == [10, 41, 41, 41] and v["parted_at"] == 1,
          f"they part at word-piece 1: {b['token_ids']} {a['token_ids']} {v['parted_at']}")
    fs = v["first_step"]
    check(set(fs) == {"kl_before_after", "top_before", "top_after", "watch_before", "watch_after"}, f"first_step {fs}")
    check(fs["watch_before"]["token_id"] == WORDS["six"] and fs["watch_after"]["prob"] > fs["watch_before"]["prob"],
          f"the target word, watched: {fs['watch_before']} -> {fs['watch_after']}")
    check(fs["kl_before_after"] > 0.0 and fs["top_before"] == b["first_top"] and fs["top_after"] == a["first_top"],
          f"KL {fs['kl_before_after']}")
    check(v["edits_in_force_before"] == [] and v["error"] is None and v["elapsed_ms"] >= 0.0, "nothing in force before")

    rt.events.clear()
    reply = vault.execute_lql(DEL, voice=VOICE)
    check(list(reply) == ["status", "patch", "active_patches_count", "voice"], f"DELETE keys {list(reply)}")
    check(reply["voice"]["first_step"]["watch_before"] is None and reply["voice"]["before"]["watch"] is None,
          "DELETE watches no word")
    check([e[:2] for e in rt.events] == [("voice", 1), ("edit", DEL), ("voice", 2)], f"DELETE's order {rt.events}")
    reply = vault.execute_lql(INS, voice={"prompt": "Who?"})
    check(list(reply) == ["status", "patch", "active_patches_count", "voice", "lens"], f"INSERT keys {list(reply)}")
    vi = reply["voice"]
    check(vi["chat"] is True and vi["max_new_tokens"] == 12 and vi["system"] is None and vi["assistant_prefix"] == ""
          and vi["first_step"]["watch_before"]["token_id"] == WORDS["she"], f"the defaults; INSERT watches 'she': {vi}")

    rt2, vault2 = stand_in_vault()
    rt2.deaf = True
    v = vault2.execute_lql(UPD, voice=VOICE)["voice"]
    check(v["before"]["token_ids"] == v["after"]["token_ids"] and v["parted_at"] is None
          and v["first_step"]["kl_before_after"] == 0.0, f"an edit the writing does not hear: KL 0, no parting: {v}")


def check_in_force() -> None:
    print("4. replaces_patch and the edits in force (the latest per neuron)")
    rt, vault = stand_in_vault()
    first = vault.execute_lql(UPD, voice=VOICE)
    again = vault.execute_lql(UPD.replace("STRENGTH 3.0", "STRENGTH 1.0"), voice=VOICE)
    other = vault.execute_lql("UPDATE LAYER 35 SLOT 101 SET object = 'six'", voice=VOICE)
    p1, p2, p3 = first["patch"]["patch_id"], again["patch"]["patch_id"], other["patch"]["patch_id"]
    check(again["patch"]["replaces_patch"] == p1 and other["patch"]["replaces_patch"] is None,
          f"replaces_patch {again['patch']['replaces_patch']}, {other['patch']['replaces_patch']}")
    check(again["voice"]["edits_in_force_before"] == [p1], "the second edit of the neuron is heard over the first")
    check(again["voice"]["before"]["token_ids"] == first["voice"]["after"]["token_ids"],
          "its 'before' is the first edit's voice")
    check(other["voice"]["edits_in_force_before"] == [p2], f"the replaced edit is not counted: "
          f"{other['voice']['edits_in_force_before']}")
    last = vault.execute_lql(DEL, voice=VOICE)
    check(last["voice"]["edits_in_force_before"] == [p2, p3] and len(vault.active_patches) == 4,
          f"in force: {last['voice']['edits_in_force_before']}, of {len(vault.active_patches)} records")
    check(len(rt.mlps[35].down_proj._memit_col_overlays) == 2, "one overlay per neuron")


def check_failures() -> None:
    print("5. a voice that cannot be written")
    rt, vault = stand_in_vault()
    vault.execute_lql(DEL)
    rt.fail_calls = {1: RuntimeError("a stand-in failure")}
    try:
        vault.execute_lql(UPD, voice=VOICE)
        check(False, "a failed voice before the edit was accepted")
    except ValueError as exc:
        check(str(exc) == "The voice before the edit could not be written: RuntimeError: a stand-in failure; no edit "
              "was made.", f"its words: {exc}")
    check(len(vault.active_patches) == 1 and rt.n_overlays() == 1 and [e[0] for e in rt.events] == ["edit", "voice"],
          "nothing edited after a failed voice before")
    for bad, words in (({"prompt": "  "}, "no question to ask"), ({"prompt": "Q", "max_new_tokens": 65}, "1-64"),
                       ({"prompt": "Q", "max_new_tokens": True}, "1-64"), ("Q", "at least a question"),
                       ({"prompt": "Q", "system": 3}, "must be text")):
        try:
            vault.execute_lql(UPD, voice=bad)
            check(False, f"voice {bad!r} accepted")
        except ValueError as exc:
            check(words in str(exc) and "no edit was made" in str(exc), f"voice {bad!r}: {exc}")
    check(len(vault.active_patches) == 1 and rt.n_overlays() == 1, "a refused voice edits nothing")
    try:
        vault.execute_lql(UPD, voice={"prompt": "x" * MAX_SWEEP_POSITIONS})
        check(False, "an over-long voice accepted")
    except ValueError as exc:
        check("could not be written: The question is" in str(exc) and str(exc).endswith("; no edit was made."),
              f"an over-long question, in plain words: {exc}")

    rt.n_voice_calls = 0
    rt.fail_calls = {2: ValueError("the stand-in ran dry.")}
    reply = vault.execute_lql(UPD, voice=VOICE)
    v = reply["voice"]
    check(reply["status"] == "applied" and len(vault.active_patches) == 2 and rt.n_overlays() == 2,
          "a failed voice after the edit keeps the edit in force")
    check(v["after"] is None and v["first_step"] is None and v["parted_at"] is None and v["before"] is not None
          and v["error"] == "The voice after the edit could not be written: the stand-in ran dry; the edit is in force.",
          f"after null, the error said: {v['error']}")
    check(vault.active_patches[-1].voice == v and list(reply)[-2:] == ["voice", "lens"], "and kept on the record")


def check_ignored() -> None:
    print("6. other statements ignore the voice; ROLLBACK clears every voice")
    rt, vault = stand_in_vault()
    vault.execute_lql(UPD, voice=VOICE)
    n = rt.n_voice_calls
    for st in ("DESCRIBE", "SELECT * FROM EDGES NEAREST 'she' WHERE layer = 35 LIMIT 3",
               "DESCRIBE POLE LAYER 53 SLOT 16383 POLARITY -"):
        with_voice = vault.execute_lql(st, voice=VOICE)
        without = vault.execute_lql(st)
        check(with_voice == without and "voice" not in with_voice, f"{st}: the voice was not ignored")
    check(vault.execute_lql("DESCRIBE", voice="not even a voice") == vault.execute_lql("DESCRIBE"),
          "a malformed voice is ignored where there is no edit")
    check(rt.n_voice_calls == n, "nothing written for a statement that is not an edit")
    rolled = vault.execute_lql("ROLLBACK", voice=VOICE)
    check(list(rolled) == ["status", "cleared_patches", "active_patches_count"] and rt.n_voice_calls == n,
          f"ROLLBACK with a voice: {list(rolled)}")
    check(vault.active_patches == [] and vault.execute_lql("DESCRIBE")["active_patches"] == [] and rt.n_overlays() == 0,
          "ROLLBACK clears the records, and their voices")
    v = vault.execute_lql(UPD, voice=VOICE)["voice"]
    check(v["edits_in_force_before"] == [] and v["before"]["token_ids"] == [10, 11, 12, 13],
          "after ROLLBACK, the model is heard unedited")


# --------------------------------------------------------------------------------------------------------------------
# 7. the routes
# --------------------------------------------------------------------------------------------------------------------


def check_routes() -> None:
    print("7. the routes: HTTP and WebSocket, under the GPU lock")
    rt, vault = stand_in_vault()
    server._RUNTIME = rt  # type: ignore[assignment]
    server._VAULT = vault

    @asynccontextmanager
    async def no_lifespan(_app):  # the real one loads the model
        yield

    server.app.router.lifespan_context = no_lifespan
    # a local Host, as the pages send it: backend/local_only.py refuses TestClient's "testserver"
    client = TestClient(server.app, base_url="http://127.0.0.1:8765")

    r = client.post("/api/larql", json={"query": UPD})
    check(r.status_code == 200 and list(r.json()) == edit_keys(UPD) and rt.n_voice_calls == 0, f"HTTP, no voice: {r.text[:200]}")
    r = client.post("/api/larql", json={"query": UPD, "voice": {"prompt": "How many legs?"}})
    body = r.json()
    check(r.status_code == 200 and list(body) == ["status", "patch", "active_patches_count", "voice", "lens"],
          f"HTTP with a voice: {list(body)}")
    check(body["voice"]["chat"] is True and body["voice"]["max_new_tokens"] == 12 and body["voice"]["system"] is None,
          f"HTTP: the request's defaults reach the vault: {body['voice']['chat']} {body['voice']['max_new_tokens']}")
    check([e for e in rt.events if e[0] == "voice"][-2:] == [("voice", 1, True), ("voice", 1, True)],
          f"both voices under the route's GPU lock: {rt.events[-3:]}")
    check(rt.voice_on_loop[-2:] == [False, False],
          f"HTTP: both voices written off the event loop, so the server keeps answering: {rt.voice_on_loop[-2:]}")
    check(body["voice"]["edits_in_force_before"] == [1] and body["patch"]["replaces_patch"] == 1, "HTTP: in force")
    n_records, n_calls = len(vault.active_patches), rt.n_voice_calls
    for bad in ({"prompt": ""}, {"prompt": "Q", "max_new_tokens": 65}, {"prompt": "Q", "max_new_tokens": 0}, "Q?"):
        r = client.post("/api/larql", json={"query": UPD, "voice": bad})
        check(r.status_code == 422, f"HTTP voice {bad!r}: {r.status_code}")
    rt.fail_calls = {rt.n_voice_calls + 1: RuntimeError("down")}
    r = client.post("/api/larql", json={"query": UPD, "voice": {"prompt": "Q"}})
    check(r.status_code == 400 and "no edit was made" in r.json().get("detail", ""), f"HTTP, a failed voice: {r.text}")
    check(len(vault.active_patches) == n_records, "HTTP: nothing edited by a refused or failed voice")
    rt.fail_calls = {}

    with client.websocket_connect("ws://127.0.0.1:8765/ws/theodolite") as ws:
        ws.send_json({"action": "larql", "query": UPD})
        msg = ws.receive_json()
        check(msg["type"] == "larql_result" and msg["ok"] and list(msg["data"]) == edit_keys(UPD),
              f"WS, no voice: {msg.get('error')}")
        ws.send_json({"action": "larql", "query": DEL, "voice": {"prompt": "Q", "chat": False, "max_new_tokens": 3}})
        msg = ws.receive_json()
        data = msg.get("data", {})
        check(msg["ok"] and list(data) == ["status", "patch", "active_patches_count", "voice"]
              and data["voice"]["chat"] is False and len(data["voice"]["before"]["token_ids"]) == 3,
              f"WS with a voice: {msg.get('error') or list(data)}")
        check(rt.events[-3][2] and rt.events[-1][2], "WS: both voices under the GPU lock")
        check(rt.voice_on_loop[-2:] == [False, False], f"WS: both voices written off the event loop: {rt.voice_on_loop[-2:]}")
        n_records = len(vault.active_patches)
        for bad, words in (({"prompt": ""}, "voice.prompt: "), ({"prompt": "Q", "max_new_tokens": 65},
                                                                 "voice.max_new_tokens: "),
                           ("Q?", "voice: "), ([1], "voice: "), ({"chat": True}, "voice.prompt: ")):
            ws.send_json({"action": "larql", "query": UPD, "voice": bad})
            msg = ws.receive_json()
            check(msg["type"] == "larql_result" and msg["ok"] is False
                  and msg["error"].startswith("The voice was refused, and no edit was made") and words in msg["error"]
                  and "data" not in msg, f"WS voice {bad!r}: {msg}")
        check(len(vault.active_patches) == n_records, "WS: nothing edited by a refused voice")
        ws.send_json({"action": "larql", "query": "DESCRIBE", "voice": {"prompt": "Q"}})
        msg = ws.receive_json()
        check(msg["ok"] and "voice" not in msg["data"] and msg["data"]["edit_method"] == EDIT_METHOD, "WS DESCRIBE")
        ws.send_json({"action": "larql", "query": "ROLLBACK"})
        check(ws.receive_json()["data"]["active_patches_count"] == 0, "WS ROLLBACK")

    check(list(inspect.signature(LarqlVaultEngine.execute_lql).parameters) == ["self", "query", "voice"],
          "execute_lql(query, voice=None)")
    check(server.VoiceRequest is larql_vault.VoiceRequest, "the server takes the vault's own VoiceRequest")
    check(voice_spec(server.VoiceRequest(prompt="Q")) == {"prompt": "Q", "chat": True, "system": None,
                                                          "assistant_prefix": "", "max_new_tokens": 12},
          "VoiceRequest and voice_spec agree on the defaults")
    check(json.dumps(body) is not None, "the reply is JSON")


# --------------------------------------------------------------------------------------------------------------------
# 8. the MCP tool, and the tools taking turns
# --------------------------------------------------------------------------------------------------------------------

TOOLS = ["causal_spot_check", "inspect_pole", "query_larql_vault", "test_motif_hypothesis", "trace_downstream_readers"]
HEARD_REPLY = ["status", "patch", "active_patches_count", "voice", "lens"]


async def until(cond: Any, timeout: float = 10.0) -> bool:
    """Wait (on the event loop) until `cond()` holds, at most `timeout` seconds; whether it did."""
    t0 = time.monotonic()
    while not cond():
        if time.monotonic() - t0 > timeout:
            return False
        await asyncio.sleep(0.01)
    return True


async def refused(client: Client, args: Dict[str, Any]) -> Optional[str]:
    """The tool's refusal of `args` in its own words, or None when it was not refused."""
    try:
        await client.call_tool("query_larql_vault", args)
    except ToolError as exc:
        return str(exc)
    return None


async def asked_meanwhile(client: Client, rt: StandInRuntime, edit: Any) -> Tuple[Any, Any, bool]:
    """Start `edit` (a coroutine whose statement is heard), hold its voice before in its worker thread, ask for
    inspect_pole meanwhile, give it half a second, then let the voice go on. Returns the two replies, and whether
    inspect_pole had answered before the voice was let go."""
    rt.events.clear()
    gate = threading.Event()
    rt.gate = gate
    edit_task = asyncio.ensure_future(edit)
    began = await until(lambda: any(e[0] == "voice" for e in rt.events))
    pole_task = asyncio.ensure_future(client.call_tool("inspect_pole", {"layer": 45, "neuron_idx": 7}))
    await until(pole_task.done, timeout=0.5)
    early = pole_task.done()
    gate.set()
    replies = await asyncio.gather(edit_task, pole_task, return_exceptions=True)
    check(began, "the voice before was begun")
    return replies[0], replies[1], early


async def mcp_turns(rt: StandInRuntime, vault: LarqlVaultEngine) -> None:
    async with Client(mcp_scientist.mcp) as client:
        tools = {t.name: t for t in await client.list_tools()}
        check(sorted(tools) == TOOLS, f"the MCP tools {sorted(tools)}")
        for name in TOOLS:
            fn = getattr(mcp_scientist, name)
            shown = list(tools[name].input_schema.get("properties", {})) if name in tools else None
            check(inspect.iscoroutinefunction(fn) and hasattr(fn, "__wrapped__")
                  and shown == list(inspect.signature(fn).parameters),
                  f"{name}: async and run in its turn, its parameters as a client sees them {shown}")
        check(list(inspect.signature(mcp_scientist.query_larql_vault).parameters) == ["lql_query", "voice"],
              "query_larql_vault(lql_query, voice=None)")
        # the voice as a client sees it: VoiceRequest's own fields, defaults and bounds, or none (taken as a mapping all
        # the same, so that one that does not fit is refused in the tool's own plain words, below)
        vault_tool = tools.get("query_larql_vault")
        vs = vault_tool.input_schema.get("properties", {}).get("voice", {}) if vault_tool else {}
        shape = next((x for x in vs.get("anyOf", []) if x.get("type") == "object"), {})
        props = shape.get("properties", {})
        n_new = props.get("max_new_tokens", {})
        check(list(props) == list(larql_vault.VoiceRequest.model_fields) and shape.get("required") == ["prompt"]
              and props.get("chat", {}).get("default") is True and n_new.get("default") == 12
              and (n_new.get("minimum"), n_new.get("maximum")) == (1, 64)
              and {"type": "null"} in vs.get("anyOf", []) and vs.get("default", "unset") is None,
              f"MCP: the voice's schema is VoiceRequest's, or null: {list(props)} {shape.get('required')}")

        # without a voice: today's reply; with one: the routes' defaults, "voice" before "lens", kept on the record,
        # both answers written in a worker thread under the tools' own lock (the server's is not involved)
        r = await client.call_tool("query_larql_vault", {"lql_query": UPD})
        check(list(r.data) == edit_keys(UPD) and r.data["patch"]["voice"] is None and rt.n_voice_calls == 0,
              f"MCP, no voice: {list(r.data)}")
        r = await client.call_tool("query_larql_vault", {"lql_query": UPD, "voice": {"prompt": "How many legs?"}})
        body = r.data
        check(list(body) == HEARD_REPLY, f"MCP with a voice: {list(body)}")
        v = body.get("voice") or {}
        check(v.get("chat") is True and v.get("max_new_tokens") == 12 and v.get("system") is None
              and list(v) == HEARD_KEYS, f"MCP: VoiceRequest's defaults reach the vault: {v.get('chat')} "
              f"{v.get('max_new_tokens')}")
        # the same neuron's overlay replaced by its like: one overlay in force before and after, so nothing moves
        check(body["patch"]["voice"] == v and body["patch"]["replaces_patch"] == 1 and v["edits_in_force_before"] == [1]
              and v["parted_at"] is None and v["before"]["token_ids"] == v["after"]["token_ids"] == [10] + [41] * 11,
              f"MCP: the second edit of the neuron is heard over the first, and replaces it: {v.get('before')}")
        check(rt.voice_on_loop[-2:] == [False, False], f"MCP: both voices written off the event loop: {rt.voice_on_loop[-2:]}")
        check(rt.voice_turn[-2:] == [True, True] and [e[2] for e in rt.events if e[0] == "voice"][-2:] == [False, False],
              f"MCP: both voices under the tools' own lock, not the server's: {rt.voice_turn[-2:]}")
        d = await client.call_tool("query_larql_vault", {"lql_query": "DESCRIBE", "voice": {"prompt": "Q"}})
        check("voice" not in d.data and d.data["active_patches"][-1]["voice"] == v,
              "MCP DESCRIBE ignores a voice, and its records carry the one heard")

        # refusals: in plain words where the tool's own check refuses, and nothing written or edited either way
        n_records, n_calls = len(vault.active_patches), rt.n_voice_calls
        for bad, words in (({"prompt": ""}, "voice.prompt: "), ({"prompt": "Q", "max_new_tokens": 65},
                                                                "voice.max_new_tokens: "),
                           ({"chat": True}, "voice.prompt: ")):
            for statement in (UPD, "DESCRIBE"):
                said = await refused(client, {"lql_query": statement, "voice": bad})
                check(said is not None and "The voice was refused, and no edit was made. " + words in said,
                      f"MCP voice {bad!r} with {statement.split()[0]}: {said}")
        for bad in ("Q?", [1]):  # not an object: FastMCP refuses it by the tool's own parameter types, first
            said = await refused(client, {"lql_query": UPD, "voice": bad})
            check(said is not None and "voice" in said, f"MCP voice {bad!r}: {said}")
        rt.fail_calls = {rt.n_voice_calls + 1: RuntimeError("down")}
        said = await refused(client, {"lql_query": UPD, "voice": {"prompt": "Q"}})
        check(said is not None and "could not be written: RuntimeError: down; no edit was made." in said,
              f"MCP, a failed voice before: {said}")
        rt.fail_calls = {}
        check(len(vault.active_patches) == n_records and rt.n_voice_calls == n_calls + 1,
              "MCP: nothing edited by a refused or failed voice")

        # a concurrent tool waits its turn: inspect_pole, asked while the voice before is written, is logged after
        # the voice after, never between
        r_edit, r_pole, early = await asked_meanwhile(client, rt, client.call_tool(
            "query_larql_vault", {"lql_query": DEL, "voice": {"prompt": "Q", "max_new_tokens": 3}}))
        order = [e[0] for e in rt.events]
        check(not early and order == ["voice", "edit", "voice", "pole"],
              f"inspect_pole, asked meanwhile, waits for the voice after: {order}")
        check(not isinstance(r_edit, BaseException) and list(r_edit.data) == HEARD_REPLY[:-1]
              and not isinstance(r_pole, BaseException) and r_pole.data["layer"] == 45
              and list(r_pole.data)[-2:] == ["label_gate", "lens"], "both replies whole")
        check(rt.pole_on_loop[-1:] == [False],
              f"inspect_pole's read is made off the event loop, as the vault's statements are: {rt.pole_on_loop[-1:]}")

        # a call cancelled while its voice is written (the MCP SDK cancels a call's task when its client says so)
        # keeps its turn until its thread ends: the edit is still made and heard, and the next tool runs after it
        n_records = len(vault.active_patches)
        scope = anyio.CancelScope()

        async def cancelled_edit() -> None:
            with scope:
                await mcp_scientist.query_larql_vault(UPD, voice={"prompt": "Q", "max_new_tokens": 2})

        async def then_cancel() -> None:
            await until(lambda: any(e[0] == "voice" for e in rt.events))
            scope.cancel()

        async with anyio.create_task_group() as tg:
            tg.start_soon(then_cancel)
            _r, _p, early = await asked_meanwhile(client, rt, cancelled_edit())
        order = [e[0] for e in rt.events]
        check(scope.cancel_called and not early and order == ["voice", "edit", "voice", "pole"]
              and len(vault.active_patches) == n_records + 1 and vault.active_patches[-1].voice is not None,
              f"a cancelled call keeps its turn until its thread ends, the edit made and heard: {order}")

        # bound as the server binds it (its routes hold _GPU_LOCK around every call), the tools' lock is a no-op:
        # the same two calls interleave, which is what the server's lock keeps apart there
        mcp_scientist.bind_runtime(rt, vault)  # type: ignore[arg-type]
        try:
            _r, _p, early = await asked_meanwhile(client, rt, client.call_tool(
                "query_larql_vault", {"lql_query": DEL, "voice": {"prompt": "Q", "max_new_tokens": 3}}))
            order = [e[0] for e in rt.events]
            check(early and order == ["voice", "pole", "edit", "voice"] and rt.voice_turn[-2:] == [False, False],
                  f"bound by the server, the tools take no lock of their own: {order}")
        finally:
            mcp_scientist.bind_runtime(rt, vault, caller_holds_lock=False)  # type: ignore[arg-type]


def check_mcp() -> None:
    print("8. the MCP tool: query_larql_vault hears an edit, and the tools take turns")
    # a refusal is logged by FastMCP with its traceback; the check reads the refusal itself
    logging.getLogger("fastmcp").setLevel(logging.CRITICAL)
    rt, vault = stand_in_vault()
    # standalone, the module loads its own runtime and takes turns on it: here it is given the stand-in instead
    mcp_scientist.bind_runtime(rt, vault, caller_holds_lock=False)  # type: ignore[arg-type]
    try:
        asyncio.run(mcp_turns(rt, vault))
    finally:
        mcp_scientist._SHARED_RUNTIME = None
        mcp_scientist._SHARED_VAULT = None


def main() -> int:
    check_write_voice()
    check_without_voice()
    check_with_voice()
    check_in_force()
    check_failures()
    check_ignored()
    check_routes()
    check_mcp()
    if failures:
        print(f"\nvault voice: {len(failures)} of {passed + len(failures)} checks FAILED")
        return 1
    print(f"\nvault voice: all {passed} checks hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())

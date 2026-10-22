#!/usr/bin/env python3
"""A CPU-only check of the sweep's wire protocol (backend/server.py): no model, no GPU, no running server.

The server's app is driven over `fastapi.testclient.TestClient`'s WebSocket with a stand-in runtime: it "writes"
scripted word-pieces (calling the stream's `on_written` as the real one does, with a short sleep for the GPU) and
returns small but complete sweep bundles, which the real `pack_binary_and_json_frames` packs. The app's lifespan
(which loads the model) is replaced and the GPU hidden, so nothing heavy can load. It checks:
1. which token indices are sighted (`sight_order`), and the request's limits, id checks and reading options (pole
   ranking, thread score, frame squaring), in plain words; and the server's own defaults for those options, kept as
   first built for old callers (Gram–Schmidt on raw rows, one frame, no centring, |a|, C);
2. a request without request_id: exactly one binary + one JSON pair, today's keys plus the additive ones, nothing
   else (the next reply on the socket is the next action's); every poles_labels entry keeps its old keys and gains
   label_check (the bundle's check where words were read, null elsewhere), and the frame carries label_gate just
   before lens;
3. a stream: progress after each written piece, the answer once, then a sighting progress, one pair per word-piece
   in order (binary then its JSON, back to back), flags bit 2, `last` on the final pair, then theodolite_sweep_end;
   an end mark stops the writing; a given answer with sight_tokens (repeats and out-of-range indices dropped);
4. cancel_sweep: while writing (the writing stops, only the first index is sighted), between pairs (the stream ends
   after the current pair), and of a queued request (its end comes at once, n_sent 0, and it never runs);
5. refusals and failures on one request alone: a pydantic refusal and a plain-words ValueError with a request_id
   (error, then sweep_end), an unknown reading option refused before anything is written, the legacy refusal
   without one (its old keys, and the additive "contexts", null outside the causal check), a failure while the frame is built (the answer already sent), a failure mid-stream (error with
   n_sent and token_index, then sweep_end) after which the socket still answers;
6. the thinking comparison's path only on the first pair (flags bit 0, think_seed, think_read_token,
   think_stop_reason, think_skipped); a stop asked while the answer is written or its first position read skips it
   (think_skipped, cancelled), one asked while it is written cuts it short (think_stop_reason "stopped", cancelled
   also with one sighting); and other actions answered in the order they came.

Run: .venv/bin/python -I scripts/check_sweep_protocol.py   (exits 0 when every check holds)
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import struct
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from fastapi.testclient import TestClient
from interp_engine.sampling import resolve_sampling
from pydantic import ValidationError

import backend.server as server
from backend.engine_runtime import (
    MAX_SWEEP_POSITIONS,
    SweepCapture,
    SweepLayerTokenReadout,
    SweepShared,
    TheodoliteSweepBundle,
    WrittenStep,
)
from backend.label_gate import LABEL_GATE_CARD, label_check
from backend.theodolite_math import BipolarPolesResult, CausalSplitResult, TheodoliteTrajectory3D

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
# 1. sight order and request checks
# --------------------------------------------------------------------------------------------------------------------


def check_rules() -> None:
    print("1. which word-pieces are sighted; what a request may hold")
    so = server.sight_order
    check(so(6, None, False, 64) == ([0], []), "legacy: [0]")
    check(so(0, None, False, 64) == ([0], []), "nothing written, legacy: [0]")
    check(so(6, None, True, 64) == ([0, 1, 2, 3, 4, 5], []), "every token")
    check(so(200, None, True, 64) == (list(range(64)), []), "every token, capped at 64")
    check(so(10, None, True, 3) == ([0, 1, 2], []), "every token, max_sighted 3")
    check(so(0, None, True, 64) == ([0], []), "every token, nothing written: [0]")
    check(so(5, [3, 0, 3, 9, -1, 1], False, 64) == ([3, 0, 1], [9, -1]), f"sight_tokens: {so(5, [3, 0, 3, 9, -1, 1], False, 64)}")
    check(so(0, [0, 1], False, 64) == ([0], [1]), "sight_tokens with nothing written")
    check(so(5, [7, 8], False, 64) == ([0], [7, 8]), "sight_tokens all out of range: [0]")
    check(so(5, [4, 3, 2, 1], False, 2) == ([4, 3], [2, 1]), "sight_tokens beyond max_sighted are dropped and reported")
    check(so(5, [2], True, 64) == ([2], []), "sight_tokens wins over sight_every_token")

    req = server.SweepRequest()
    check(req.max_new_tokens == 6 and req.sampler.temperature == 0.0 and req.sampler.top_k == 0 and req.sampler.top_p == 1.0
          and req.sampler.presence_penalty == 0.0 and req.sampler.seed is None, "the defaults are greedy, 6 pieces")
    check(req.request_id is None and not req.sight_every_token and req.rank_lookahead and req.stop_at_end_of_turn,
          "the defaults answer as before")

    def refused(payload: Dict[str, Any], what: str) -> None:
        try:
            server.SweepRequest(**payload)
        except ValidationError:
            check(True, what)
        else:
            check(False, f"{what}: accepted")

    refused({"max_new_tokens": 257}, "257 word-pieces")
    refused({"max_new_tokens": -1}, "a negative length")
    refused({"sampler": {"temperature": 2.5}}, "temperature 2.5")
    refused({"sampler": {"temperature": -0.1}}, "a negative temperature")
    refused({"sampler": {"top_p": 0.0}}, "top_p 0")
    refused({"sampler": {"top_k": -1}}, "top_k -1")
    refused({"sampler": {"top_k": 262145}}, "top_k beyond the vocabulary")
    refused({"sampler": {"presence_penalty": 2.5}}, "a penalty of 2.5")
    refused({"sampler": {"seed": -1}}, "a negative seed")
    refused({"sampler": {"seed": 2**53}}, "a seed beyond 2^53 - 1")
    refused({"request_id": "r" * 65}, "a 65-character request id")
    refused({"prompt": "x" * 32001}, "a 32001-character prompt")
    refused({"system": "x" * 32001}, "a 32001-character system prompt")
    refused({"assistant_prefix": "x" * 32001}, "a 32001-character begun answer")
    refused({"answer_token_ids": [1, "2"]}, "an id given as text")
    refused({"answer_token_ids": [1, 2.0]}, "an id given as a float")
    refused({"answer_token_ids": [True]}, "an id given as a bool")
    refused({"answer_token_ids": list(range(257))}, "a 257-piece given answer")
    refused({"sight_tokens": ["0"]}, "a token index given as text")
    refused({"sight_tokens": list(range(65))}, "65 token indices")
    refused({"centre_token_ids": [[2]] * 17}, "17 controls")
    refused({"max_sighted": 0}, "max_sighted 0")
    refused({"max_sighted": 65}, "max_sighted 65")
    check(server.SweepRequest(sampler={"seed": 2**53 - 1}).sampler.seed == 2**53 - 1, "the largest seed")

    def plain(payload: Dict[str, Any], words: str, what: str) -> None:
        try:
            server.check_sweep_ids(server.SweepRequest(**payload))
        except ValueError as exc:
            check(words in str(exc), f"{what}: {exc}")
        else:
            check(False, f"{what}: accepted")

    plain({"answer_token_ids": [5, 262144]}, "262144, which is not a word-piece id", "an id past the vocabulary")
    plain({"answer_token_ids": [-3]}, "-3, which is not a word-piece id", "a negative id")
    plain({"centre_token_ids": [[2, 5], []]}, "Matched control 2 has no word-pieces", "an empty control")
    plain({"centre_token_ids": [[2, 300000]]}, "Matched control 1 holds 300000", "a control id past the vocabulary")
    plain({"centre_token_ids": [[2] * (MAX_SWEEP_POSITIONS + 1)]}, "more than the", "an over-long control")
    server.check_sweep_ids(server.SweepRequest(answer_token_ids=[0, 262143], centre_token_ids=[[2, 9]]))
    check(True, "ids at both ends of the vocabulary")
    server.check_sweep_positions(MAX_SWEEP_POSITIONS - 6, 6)
    check(True, f"exactly {MAX_SWEEP_POSITIONS} positions")
    try:
        server.check_sweep_positions(MAX_SWEEP_POSITIONS - 5, 6)
        check(False, "one position too many accepted")
    except ValueError as exc:
        check(f"more than the {MAX_SWEEP_POSITIONS} one sweep can hold" in str(exc), f"the positions refusal: {exc}")
    def option(payload: Dict[str, Any], words: str, what: str) -> None:
        try:
            server.check_sweep_options(server.SweepRequest(**payload))
        except ValueError as exc:
            check(words in str(exc), f"{what}: {exc}")
        else:
            check(False, f"{what}: accepted")

    option({"pole_ranking": "loudest"}, 'pole_ranking is "loudest", which this instrument does not know: it must be '
           '"activation" or "write"', "an unknown pole ranking")
    option({"thread_score": "Direct"}, 'must be "coupling" or "direct"', "a thread score in the wrong case")
    option({"frame_squaring": "qr"}, 'must be "gram_schmidt" or "lowdin"', "an unknown frame squaring")
    for ok in ({}, {"pole_ranking": "write", "thread_score": "direct", "frame_squaring": "lowdin"}):
        server.check_sweep_options(server.SweepRequest(**ok))
        check(True, f"known reading options {ok}")
    # the server keeps the instrument as first built for old callers; the interface sends the mends as options
    # (scripts/check_interface_defaults.py checks what it sends)
    r = server.SweepRequest(prompt="x")
    as_built = {"frame_squaring": "gram_schmidt", "frame_per_layer": False, "frame_fnw": False, "centre_prompts": [],
                "centre_token_ids": [], "centre_answer": False, "pole_ranking": "activation", "thread_score": "coupling"}
    for key, want in as_built.items():
        check(getattr(r, key) == want, f"the server's default {key} is {getattr(r, key)!r}, not {want!r} as first built")
    tr = server.TraceReadersRequest(token_ids=[2, 5, 7])
    check(tr.token_ids == [2, 5, 7], "trace token_ids")
    for bad in ([], ["2"], [2.5], list(range(MAX_SWEEP_POSITIONS + 1))):
        try:
            server.TraceReadersRequest(token_ids=bad)
            check(False, f"trace token_ids {str(bad)[:30]} accepted")
        except ValidationError:
            check(True, "trace token_ids refused")


# --------------------------------------------------------------------------------------------------------------------
# a stand-in runtime
# --------------------------------------------------------------------------------------------------------------------

D = 8  # the stand-in's model width
END = 106


class FakeTokenizer:
    eos_token_id = 1

    def decode(self, ids: Sequence[int], clean_up_tokenization_spaces: Optional[bool] = None) -> str:
        return "".join("<turn|>" if int(i) == END else f"·{int(i)}" for i in ids)


class FakeVault:
    active_patches: List[Any] = []

    def edits_in_force(self) -> List[Any]:
        return []

    def execute_lql(self, query: str) -> Dict[str, Any]:
        return {"command": query}


class FakeRuntime:
    """Writes `script` (one piece per step, sleeping `write_s` each), reads each position in `read_s`."""

    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.device = torch.device("cpu")
        self.script: List[int] = [10, 11, 12, 13, 14, 15, 16, 17]
        self.write_s = 0.01
        self.read_s = 0.01
        self.fail_at: Optional[int] = None
        self.fail_prepare = False
        self.writes: List[str] = []
        # a stop asked at a given moment, as the socket's receiver would set it on a cancel_sweep:
        # ("think", i) after the i-th thinking piece, ("analyse", k) while the answer's position k is read
        self.stop_at: Optional[Tuple[str, int]] = None
        self.conn: Any = None
        self.analysed: List[Tuple[str, int]] = []
        # the model's sliding window, as the real runtime reads it from text_config (every pair must echo it)
        self.sliding_window: Optional[int] = 1024

    def _ask_stop(self, event: str, i: int) -> None:
        if self.stop_at == (event, i) and self.conn is not None:
            self.conn.cancel_running = True

    def lens_card(self) -> Dict[str, Any]:
        return {"id": "stand-in"}

    def compose_prompt(self, text, *, chat=False, system=None, assistant_prefix="", thinking=False) -> str:
        return ("<|think|>" if thinking else "") + (f"[{system}]" if system else "") + text + assistant_prefix

    def tokenize_prompt(self, prompt: str, prepend_bos: bool = True) -> List[int]:
        return [2] + [1000 + (ord(ch) % 500) for ch in prompt]

    def end_of_turn_ids(self) -> List[int]:
        return [1, END, 50]

    async def ensure_quiet_masks(self) -> None:
        return None

    def _step(self, j: int, tid: int, source: str, end: bool, greedy: bool) -> WrittenStep:
        return WrittenStep(
            index=j, token_id=tid, token=self.tokenizer.decode([tid]), model_prob=0.5, model_logprob=-0.69,
            model_rank=1, n_tied=0, chosen_logit=20.0, top=[(tid, self.tokenizer.decode([tid]), 0.5)],
            sampler_prob=None if greedy else 0.4, sampler_support=None if greedy else 64, end_mark=end, source=source,
        )

    async def capture_for_sweep(self, prompt, *, steering_spec=None, steering_position_mask=None, max_new_tokens=6,
                                sampler=None, stop_token_ids=None, answer_token_ids=None, prepend_bos=True,
                                on_written=None) -> SweepCapture:
        self.writes.append(prompt)
        greedy = sampler is None or sampler.temperature <= 0
        settings = dict(resolve_sampling(
            None, temperature=sampler.temperature if sampler else 0.0, top_k=sampler.top_k if sampler else 0,
            top_p=sampler.top_p if sampler else 1.0, presence_penalty=sampler.presence_penalty if sampler else 0.0,
        ).as_dict())
        if greedy:
            settings.update(top_k=None, top_p=None)
        script = [10, END] if "yes" in prompt else self.script
        seed = None if greedy else (sampler.seed if sampler.seed is not None else 12345)
        stop = set(stop_token_ids or []) | {1}
        answer: List[int] = []
        steps: List[WrittenStep] = []
        if answer_token_ids is not None:
            answer = list(answer_token_ids)
            steps = [self._step(j, t, "given", t in stop, greedy) for j, t in enumerate(answer)]
            reason = "given"
        elif max_new_tokens > 0:
            reason = "length"
            for i in range(max_new_tokens):
                time.sleep(self.write_s)  # the GPU: the event loop is busy
                tid = script[i % len(script)]
                end = tid in stop
                steps.append(self._step(i, tid, "written", end, greedy))
                answer.append(tid)
                if prompt.startswith("<|think|>"):
                    self._ask_stop("think", i)
                keep = True if on_written is None else await on_written(i)
                if end:
                    reason = "end_mark"
                    break
                if not keep and i + 1 < max_new_tokens:
                    reason = "stopped"
                    break
        else:
            reason = "none"
        ids = self.tokenize_prompt(prompt)
        strs = [self.tokenizer.decode([t]) for t in answer]
        return SweepCapture(
            prompt=prompt, prompt_token_ids=ids, prompt_token_strs=[self.tokenizer.decode([t]) for t in ids],
            answer_token_ids=answer, answer_token_strs=strs, answer_text="".join(strs), steps=steps, caps={},
            settings=settings,
            greedy=greedy, seed=seed, seed_drawn=not greedy and sampler.seed is None, stop_reason=reason,
            stop_token_ids=sorted(stop), answer_source="given" if answer_token_ids is not None else "written",
            max_new_tokens=max_new_tokens, generate_ms=1.0, capture_ms=1.0,
        )

    async def prepare_sweep(self, cap, *, monument_concepts, frame_override=None, centre_override=None, **_kw) -> SweepShared:
        if self.fail_prepare:
            raise ValueError("a stand-in frame that cannot be built")
        basis = frame_override[0] if frame_override else torch.eye(3, D)
        return SweepShared(
            tracked_map={"eight": 7}, survey_mode="monument", locked_pole=None, frame_squaring="gram_schmidt",
            basis=basis, basis_space="J", frame_info={"squaring": "gram_schmidt", "per_layer": False,
                                                       "fnw_weighted": False, "state": "h_J"},
            survey_labels=("a", "b", "c"), control_ids=[], centre_k0=centre_override, centre_answer=False,
        )

    async def analyse_position(self, cap, shared, k, *, rank_lookahead=True, **_kw) -> TheodoliteSweepBundle:
        time.sleep(self.read_s)  # the GPU
        thinking = cap.prompt.startswith("<|think|>")
        self.analysed.append(("think" if thinking else "answer", k))
        if not thinking:
            self._ask_stop("analyse", k)
        if self.fail_at is not None and k == self.fail_at:
            raise RuntimeError("a stand-in failure")
        return fake_bundle(cap, shared, k, rank_lookahead)


def fake_bundle(cap: SweepCapture, shared: SweepShared, k: int, rank_lookahead: bool) -> TheodoliteSweepBundle:
    g = torch.Generator().manual_seed(k)
    r = lambda *s: torch.rand(*s, generator=g)  # noqa: E731
    traj = TheodoliteTrajectory3D(
        xyz=r(60, 3), azimuth_rad=r(60), elevation_rad=r(60), azimuth_deg=r(60), elevation_deg=r(60),
        residual_norm=r(60), projection_norm=r(60), delta_attn_xyz=r(60, 3), delta_ffn_xyz=r(60, 3), proj_frac=r(60) * 0.9 + 0.05,
    )
    splits = {
        l: CausalSplitResult(
            delta_attn_J=r(D), delta_ffn_J=r(D), resid_pre_J=r(D), resid_mid_J=r(D), resid_post_J=r(D),
            attn_norm=torch.tensor(1.0), ffn_norm=torch.tensor(2.0), ffn_dominance_ratio=torch.tensor(0.6),
            rescale_norm=torch.tensor(0.1), layer_scalar=0.5,
        )
        for l in range(60)
    }
    poles = {
        l: BipolarPolesResult(
            layer_idx=l, neuron_indices=torch.arange(15), abs_activations=r(15), signed_activations=r(15),
            polarities=torch.ones(15), signed_w_down=r(15, D), jpp_pole_vectors=r(15, D),
        )
        for l in range(60)
    }

    def ro(l: int) -> SweepLayerTokenReadout:
        return SweepLayerTokenReadout(
            layer_idx=l, top_token_ids=[10 + k, 3, 4, 5, 6], top_token_strs=["a", "b", "c", "d", "e"],
            top_probs=[0.5, 0.2, 0.1, 0.1, 0.1], top_logits=[5.0, 4.0, 3.0, 2.0, 1.0],
            tracked_token_probs={"eight": 0.1}, tracked_token_logits={"eight": 1.0}, tracked_token_ranks={"eight": 3},
        )

    P = len(cap.prompt_token_ids)
    ids = cap.prompt_token_ids + cap.answer_token_ids
    chosen = cap.steps[k] if k < len(cap.steps) else None
    # words and their checks at the layers the runtime decodes by default here (two of them suffice): at 35 below the
    # band (held back), at 59 one reading
    words = {(l, r): [("a", 3.0), ("b", 2.0), ("c", 1.0)] for l in LABELLED_LAYERS for r in range(5)}
    checks = {(l, r): label_check(l, list(range(10)), list(range(r, 10 + r)), "jpp-base") for l, r in words}
    return TheodoliteSweepBundle(
        prompt=cap.prompt, prompt_token_ids=cap.prompt_token_ids, prompt_token_strs=cap.prompt_token_strs,
        target_pos=P - 1 + k, generated_text=cap.answer_text, generated_token_ids=cap.answer_token_ids,
        trajectory_3d=traj, causal_splits=splits, bipolar_poles=poles, filaments=[], cka_matrix=torch.eye(60),
        jpp_layer_readouts={l: ro(l) for l in range(60)}, naive_layer_readouts={l: ro(l) for l in range(60)},
        pole_top_tokens_jpp=words, pole_top_tokens_naive=words, survey_mode="monument", survey_labels=("a", "b", "c"),
        peak_vram_gb=0.0, elapsed_ms=1.0, survey_basis=shared.basis, survey_basis_space="J",
        frame_info={**shared.frame_info, "centred": False, "n_centre_prompts": 0, "centre_kind": "none"},
        token_index=k, read_token_id=ids[P - 1 + k], read_token=FakeTokenizer().decode([ids[P - 1 + k]]),
        chosen=chosen,
        # as the runtime sends it: a given piece has no writing to compare with, so no logit and no diff
        recapture_check=None if chosen is None else {
            "top1_id": chosen.token_id, "top1_token": chosen.token, "top1_is_chosen": True,
            "chosen_logit_recapture": 20.0 if chosen.source == "written" else None,
            "chosen_logit_diff": 0.0 if chosen.source == "written" else None,
        },
        kinship_n_states=P + k, tracked_from_next={l: {"eight": rank_lookahead and k == 0 and l == 59} for l in range(60)},
        rank_lookahead=rank_lookahead, analyse_ms=1.0, pole_label_checks=checks,
    )


# --------------------------------------------------------------------------------------------------------------------
# 2-6. the socket
# --------------------------------------------------------------------------------------------------------------------


class Socket:
    """A /ws/theodolite test session that reads frames as ("bytes", b) or ("json", obj)."""

    def __init__(self, ws: Any) -> None:
        self.ws = ws

    def send(self, obj: Dict[str, Any]) -> None:
        self.ws.send_json(obj)

    def sweep(self, payload: Dict[str, Any]) -> None:
        self.send({"action": "sweep", "payload": payload})

    def recv(self) -> Tuple[str, Any]:
        msg = self.ws.receive()
        if msg["type"] == "websocket.close":
            raise RuntimeError(f"the socket closed: {msg}")
        if msg.get("bytes") is not None:
            return "bytes", msg["bytes"]
        import json

        return "json", json.loads(msg["text"])

    def until_end(self, rid: str, limit: int = 400) -> List[Tuple[str, Any]]:
        """Every frame up to and including `rid`'s theodolite_sweep_end."""
        got: List[Tuple[str, Any]] = []
        for _ in range(limit):
            kind, obj = self.recv()
            got.append((kind, obj))
            if kind == "json" and obj.get("type") == "theodolite_sweep_end" and obj.get("request_id") == rid:
                return got
        raise RuntimeError(f"no sweep_end for {rid} in {limit} frames")


def pairs_of(frames: List[Tuple[str, Any]], what: str) -> List[Tuple[bytes, Dict[str, Any]]]:
    """The (binary, JSON) pairs in `frames`, checking each binary is followed at once by its metadata."""
    out = []
    for i, (kind, obj) in enumerate(frames):
        if kind == "bytes":
            nxt = frames[i + 1] if i + 1 < len(frames) else ("none", None)
            ok = nxt[0] == "json" and nxt[1].get("type") == "theodolite_metadata"
            check(ok, f"{what}: a binary frame not followed at once by its JSON ({nxt[0]})")
            if ok:
                out.append((obj, nxt[1]))
    return out


def header(binary: bytes) -> Tuple[int, ...]:
    return struct.unpack("<6I", binary[:24])


LABELLED_LAYERS = (35, 59)
# each poles_labels entry's keys as first sent, in order; label_check is added after them
LABEL_KEYS = ["rank", "neuron_idx", "polarity", "abs_act", "write_size", "j_vindex_tokens", "naive_tokens"]
LEGACY_KEYS = {
    "type", "prompt", "prompt_token_strs", "generated_text", "think_generated_text", "survey_mode", "survey_labels",
    "elapsed_ms", "peak_vram_gb", "global_layers_0based", "global_layers_1based", "n_filaments", "is_steered",
    "prompt_token_ids", "pole_ranking", "thread_score", "frame", "lens", "layer_plumes", "poles_labels",
    "filaments_meta", "larql_status", "chat", "system", "user_prompt", "assistant_prefix", "steering_applied",
}
NEW_KEYS = {
    "request_id", "token_index", "sight_index", "n_to_sight", "last", "read_pos", "sliding_window", "read_token",
    "read_token_id", "prompt_len", "chosen", "answer", "generation", "recapture_check", "kinship_n_states",
    "rank_lookahead", "sight_dropped", "timing", "think_seed", "think_read_token", "think_stop_reason", "think_skipped",
    "label_gate",
}


def check_socket() -> None:
    rt = FakeRuntime()
    server._RUNTIME = rt  # type: ignore[assignment]
    server._VAULT = FakeVault()  # type: ignore[assignment]
    # keep each socket's state, so the stand-in can ask a stop at an exact moment (as the receiver does on a cancel)
    conns: List[Any] = []
    real_state = server._SocketState

    class KeptState(real_state):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            conns.append(self)

    server._SocketState = KeptState  # type: ignore[misc]

    @asynccontextmanager
    async def no_lifespan(_app):  # the real one loads the model
        yield

    server.app.router.lifespan_context = no_lifespan
    client = TestClient(server.app)
    with client.websocket_connect("/ws/theodolite") as raw:
        s = Socket(raw)

        print("2. a request without request_id: one pair, as before")
        s.sweep({"prompt": "legs?", "max_new_tokens": 6})
        k1, b = s.recv()
        k2, meta = s.recv()
        check(k1 == "bytes" and k2 == "json" and meta["type"] == "theodolite_metadata", "legacy: binary then JSON")
        check(LEGACY_KEYS <= set(meta), f"legacy keys missing: {LEGACY_KEYS - set(meta)}")
        check(NEW_KEYS <= set(meta), f"additive keys missing: {NEW_KEYS - set(meta)}")
        check(meta["sliding_window"] == 1024, f"legacy: sliding_window {meta.get('sliding_window')!r}")
        check(meta["request_id"] is None and meta["token_index"] == 0 and meta["last"] is True and meta["n_to_sight"] == 1,
              "legacy: token 0, last")
        check(meta["generated_text"] == meta["answer"]["text"] and meta["answer"]["n_written"] == 6, "legacy: the answer")
        check(header(b)[5] == 0, f"legacy flags {header(b)[5]}")
        plume = meta["layer_plumes"][59]
        check(plume["jpp_top_ids"] == [10, 3, 4, 5, 6] and plume["naive_top_id"] == 10, "jpp_top_ids / naive_top_id")
        check(plume["tracked_from_next"] == {"eight": True}, f"tracked_from_next {plume['tracked_from_next']}")
        check(meta["frame"]["centre_kind"] == "none", "frame.centre_kind")
        gen = meta["generation"]
        check(gen["greedy"] is True and gen["seed"] is None and gen["max_new_tokens"] == 6 and gen["capped"] is False
              and gen["stop_at_end_of_turn"] is True, f"generation {gen}")
        check(set(meta["timing"]) == {"generate_ms", "capture_ms", "analyse_ms", "total_ms"}, "timing keys")
        keys = list(meta)
        check(meta["label_gate"] == LABEL_GATE_CARD and keys.index("label_gate") + 1 == keys.index("lens"),
              "label_gate: the gate's card, just before lens")
        n_entries = 0
        for l in range(60):
            entries = meta["poles_labels"][str(l)]
            for r, e in enumerate(entries):
                n_entries += 1
                check(list(e) == LABEL_KEYS + ["label_check"], f"poles_labels[{l}][{r}] keys {list(e)}")
                if l in LABELLED_LAYERS:
                    want = label_check(l, list(range(10)), list(range(r, 10 + r)), "jpp-base")
                    check(e["label_check"] == want and [w["token"] for w in e["j_vindex_tokens"]] == ["a", "b", "c"],
                          f"poles_labels[{l}][{r}]: words and check {e}")
                else:
                    check(e["label_check"] is None and e["j_vindex_tokens"] == [], f"poles_labels[{l}][{r}]: {e}")
        check(n_entries == 300, f"{n_entries} poles_labels entries")
        held = meta["poles_labels"]["35"][0]["label_check"]
        check(held["held_back"] and held["reasons"] == ["below_layer_40"], f"layer 35's label is held back: {held}")
        s.send({"action": "larql", "query": "DESCRIBE"})
        kind, obj = s.recv()
        check(kind == "json" and obj.get("type") == "larql_result", f"legacy: something came between ({obj.get('type')})")

        print("3. a stream: every word-piece sighted")
        s.sweep({"request_id": "r1", "prompt": "legs?", "max_new_tokens": 4, "sight_every_token": True, "rank_lookahead": False})
        frames = s.until_end("r1")
        prog = [o for k, o in frames if k == "json" and o["type"] == "theodolite_progress"]
        written = [o for k, o in frames if k == "json" and o["type"] == "theodolite_written"]
        end = frames[-1][1]
        check([p["done"] for p in prog if p["phase"] == "writing"] == [1, 2, 3, 4] and all(p["of"] == 4 for p in prog if p["phase"] == "writing"),
              f"writing progress {prog}")
        check(any(p["phase"] == "sighting" and p["done"] == 0 and p["of"] == 4 for p in prog), "sighting progress")
        kinds = [o["type"] if o["type"] != "theodolite_progress" else o["phase"] for k, o in frames if k == "json"]
        check(kinds.index("theodolite_written") < kinds.index("sighting"), f"the answer comes before the sighting progress: {kinds}")
        check(len(written) == 1 and written[0]["sight_order"] == [0, 1, 2, 3] and written[0]["n_to_sight"] == 4, "written")
        w = written[0]
        check(w["answer"]["token_ids"] == [10, 11, 12, 13] and w["answer"]["stop_reason"] == "length"
              and len(w["answer"]["steps"]) == 4 and w["prompt_len"] == len(w["prompt_token_ids"]), f"written answer {w['answer']}")
        first_pair = next(i for i, (k, _o) in enumerate(frames) if k == "bytes")
        check(all(i < first_pair for i, (k, o) in enumerate(frames) if k == "json" and o["type"] == "theodolite_written"),
              "the answer comes before the first pair")
        pairs = pairs_of(frames, "r1")
        check([m["token_index"] for _b, m in pairs] == [0, 1, 2, 3], "pairs in sight order")
        check([m["sight_index"] for _b, m in pairs] == [0, 1, 2, 3], "sight_index")
        check([m["last"] for _b, m in pairs] == [False, False, False, True], "last only on the final pair")
        check(all(header(b)[5] & 4 for b, _m in pairs), "flags bit 2 on every pair of a set")
        P = w["prompt_len"]
        check([m["read_pos"] for _b, m in pairs] == [P - 1 + k for k in range(4)], "read_pos = P - 1 + k")
        check(all(m["sliding_window"] == rt.sliding_window for _b, m in pairs), "sliding_window beside every read_pos")
        check([m["chosen"]["token_id"] for _b, m in pairs] == [10, 11, 12, 13], "chosen is answer[k]")
        check(pairs[1][1]["read_token_id"] == 10 and pairs[0][1]["read_token_id"] == w["prompt_token_ids"][-1],
              "read_token_id: the prompt's last piece, then answer[k - 1]")
        check(all(m["kinship_n_states"] == m["read_pos"] + 1 for _b, m in pairs), "kinship_n_states")
        check(all(m["rank_lookahead"] is False and m["request_id"] == "r1" for _b, m in pairs), "echoes")
        check(all(m["layer_plumes"][59]["tracked_from_next"] == {"eight": False} for _b, m in pairs), "no lookahead when off")
        check(end == {"type": "theodolite_sweep_end", "request_id": "r1", "n_sent": 4, "n_to_sight": 4, "cancelled": False, "error": None},
              f"sweep_end {end}")

        rt.script = [10, 11, END, 12]
        s.sweep({"request_id": "r1b", "prompt": "hi", "chat": True, "max_new_tokens": 8, "sight_every_token": True})
        frames = s.until_end("r1b")
        w = next(o for k, o in frames if k == "json" and o["type"] == "theodolite_written")
        check(w["answer"]["stop_reason"] == "end_mark" and w["answer"]["token_ids"] == [10, 11, END]
              and w["answer"]["steps"][-1]["end_mark"] is True, f"end mark {w['answer']['stop_reason']}")
        check(len(pairs_of(frames, "r1b")) == 3, "the end mark's own position is sighted")
        rt.script = [10, 11, 12, 13, 14, 15, 16, 17]

        s.sweep({"request_id": "r1c", "prompt": "hi", "answer_token_ids": [5, 6, 7], "sight_tokens": [2, 0, 9, 2]})
        frames = s.until_end("r1c")
        w = next(o for k, o in frames if k == "json" and o["type"] == "theodolite_written")
        pairs = pairs_of(frames, "r1c")
        check(w["answer"]["source"] == "given" and w["answer"]["stop_reason"] == "given" and w["sight_order"] == [2, 0],
              f"a given answer {w['sight_order']}")
        check([m["token_index"] for _b, m in pairs] == [2, 0] and all(m["sight_dropped"] == [9] for _b, m in pairs),
              "sight_tokens order and dropped")
        check(not any(o["type"] == "theodolite_progress" and o["phase"] == "writing" for k, o in frames if k == "json"),
              "no writing progress for a given answer")

        s.sweep({"request_id": "r1d", "prompt": "hi", "max_new_tokens": 10, "sight_every_token": True, "max_sighted": 3,
                 "sampler": {"temperature": 1.0, "top_k": 64, "top_p": 0.95}})
        frames = s.until_end("r1d")
        pairs = pairs_of(frames, "r1d")
        g = pairs[0][1]["generation"]
        check(len(pairs) == 3 and g["capped"] is True and g["greedy"] is False and g["seed"] == 12345 and g["seed_drawn"] is True,
              f"capped and sampled: {g}")

        print("4. cancel_sweep")
        rt.write_s = 0.02
        s.sweep({"request_id": "r2", "prompt": "long", "max_new_tokens": 60, "sight_every_token": True})
        first = s.recv()
        check(first[0] == "json" and first[1]["type"] == "theodolite_progress", "a first progress")
        s.send({"action": "cancel_sweep", "request_id": "r2"})
        frames = [first] + s.until_end("r2")
        w = next(o for k, o in frames if k == "json" and o["type"] == "theodolite_written")
        pairs = pairs_of(frames, "r2")
        check(w["answer"]["stop_reason"] == "stopped" and w["answer"]["n_written"] < 60, f"writing stopped at {w['answer']['n_written']}")
        check(len(pairs) == 1 and pairs[0][1]["token_index"] == 0 and pairs[0][1]["last"] is True and w["n_to_sight"] == 1,
              "only the first index sighted after a cancel while writing")
        end = frames[-1][1]
        check(end["cancelled"] is True and end["n_sent"] == 1 and end["n_to_sight"] == 1, f"sweep_end {end}")
        rt.write_s = 0.01

        rt.read_s = 0.1
        s.sweep({"request_id": "r3", "prompt": "hi", "max_new_tokens": 8, "sight_every_token": True})
        got: List[Tuple[str, Any]] = []
        while not (got and got[-1][0] == "bytes"):
            got.append(s.recv())
        s.send({"action": "cancel_sweep", "request_id": "r3"})
        frames = got + s.until_end("r3")
        pairs = pairs_of(frames, "r3")
        end = frames[-1][1]
        check(end["cancelled"] is True and 1 <= end["n_sent"] < 8 and end["n_sent"] == len(pairs) and end["n_to_sight"] == 8,
              f"cancel between pairs: {end}")
        rt.read_s = 0.01

        rt.write_s = 0.05
        s.sweep({"request_id": "r4", "prompt": "long", "max_new_tokens": 40, "sight_every_token": True})
        s.sweep({"request_id": "r5", "prompt": "queued", "max_new_tokens": 2})
        s.send({"action": "cancel_sweep", "request_id": "r5"})
        frames = []
        r5_end_at = None
        for _ in range(500):
            kind, obj = s.recv()
            frames.append((kind, obj))
            if kind == "json" and obj.get("request_id") == "r5":
                r5_end_at = len(frames)
                break
        check(r5_end_at is not None and frames[-1][1] == {"type": "theodolite_sweep_end", "request_id": "r5", "n_sent": 0,
              "n_to_sight": 0, "cancelled": True, "error": None}, f"the queued request's end: {frames[-1]}")
        check(not any(k == "json" and o.get("type") == "theodolite_sweep_end" and o.get("request_id") == "r4" for k, o in frames),
              "the queued cancel waited for the running request")
        s.send({"action": "cancel_sweep", "request_id": "r4"})
        frames = s.until_end("r4")
        check(not any(k == "json" and o.get("request_id") == "r5" for k, o in frames), "r5 sent something after its end")
        s.send({"action": "larql", "query": "AFTER"})
        kind, obj = s.recv()
        check(kind == "json" and obj.get("type") == "larql_result" and obj["data"]["command"] == "AFTER",
              f"the cancelled r5 ran: {obj.get('type')} {obj.get('request_id')}")
        check("queued" not in "".join(rt.writes[-3:]), "the cancelled queued request was written")
        rt.write_s = 0.01

        print("5. refusals and failures stay on their own request")
        s.sweep({"request_id": "r6", "max_new_tokens": 300})
        kind, err = s.recv()
        kind2, end = s.recv()
        check(err.get("type") == "error" and err.get("request_id") == "r6" and err.get("action") == "sweep"
              and err.get("recoverable") is True and err.get("n_sent") == 0 and err.get("token_index") is None
              and "max_new_tokens" in err.get("message", ""), f"pydantic refusal: {err}")
        check(end.get("type") == "theodolite_sweep_end" and end.get("request_id") == "r6" and end["error"] == err["message"]
              and end["n_sent"] == 0 and end["cancelled"] is False, f"its end: {end}")
        s.sweep({"request_id": "r7", "prompt": "x" * (MAX_SWEEP_POSITIONS + 1)})
        _k, err = s.recv()
        _k, end = s.recv()
        check(err.get("type") == "error" and "one sweep can hold" in err.get("message", "") and end.get("request_id") == "r7",
              f"positions refusal: {err}")
        s.sweep({"max_new_tokens": 300})
        _k, err = s.recv()
        check(err.get("type") == "error" and err.get("action") == "sweep" and err.get("recoverable") is True
              and "request_id" not in err
              and set(err) == {"type", "message", "action", "recoverable", "layer", "neuron_idx", "contexts"}
              and err["contexts"] is None,
              f"the legacy refusal: {err}")
        for opt, val in (("pole_ranking", "loudest"), ("thread_score", "Direct"), ("frame_squaring", "qr")):
            n_writes = len(rt.writes)
            s.sweep({"request_id": f"r7-{opt}", "prompt": "hi", "max_new_tokens": 5, "sight_every_token": True, opt: val})
            frames = s.until_end(f"r7-{opt}")
            types = [o.get("type") for k, o in frames if k == "json"]
            err = next((o for k, o in frames if k == "json" and o.get("type") == "error"), {})
            check(types == ["error", "theodolite_sweep_end"] and not any(k == "bytes" for k, _o in frames),
                  f"an unknown {opt}: {types}")
            check(f'{opt} is "{val}", which this instrument does not know' in err.get("message", ""), f"its words: {err}")
            check(len(rt.writes) == n_writes, f"an unknown {opt} was refused only after writing")
        rt.fail_prepare = True
        s.sweep({"request_id": "r7b", "prompt": "hi", "max_new_tokens": 3, "sight_every_token": True})
        frames = s.until_end("r7b")
        types = [o.get("type") for k, o in frames if k == "json"]
        w = next((o for k, o in frames if k == "json" and o.get("type") == "theodolite_written"), {})
        check("theodolite_written" in types and "error" in types and types.index("theodolite_written") < types.index("error"),
              f"a frame that fails after the writing: the answer still comes, first ({types})")
        check(w.get("answer", {}).get("token_ids") == [10, 11, 12] and frames[-1][1]["n_sent"] == 0
              and "cannot be built" in (frames[-1][1]["error"] or ""), f"its answer and end: {frames[-1][1]}")
        rt.fail_prepare = False
        rt.fail_at = 2
        s.sweep({"request_id": "r8", "prompt": "hi", "max_new_tokens": 5, "sight_every_token": True})
        frames = s.until_end("r8")
        pairs = pairs_of(frames, "r8")
        err = next((o for k, o in frames if k == "json" and o.get("type") == "error"), {})
        check(len(pairs) == 2 and err.get("n_sent") == 2 and err.get("token_index") == 2 and "stand-in failure" in err.get("message", ""),
              f"a failure mid-stream: {err}")
        check(frames[-1][1]["error"] == err.get("message") and frames[-1][1]["n_sent"] == 2, "its end names the failure")
        rt.fail_at = None
        s.sweep({"prompt": "still there?"})
        k1, _b = s.recv()
        k2, meta = s.recv()
        check(k1 == "bytes" and meta.get("type") == "theodolite_metadata", "the socket answers after a failure")

        print("6. the thinking comparison, and the order of other actions")
        s.sweep({"request_id": "r9", "prompt": "hi", "max_new_tokens": 3, "sight_every_token": True, "compare_think": True,
                 "sampler": {"temperature": 0.7, "seed": 99}})
        frames = s.until_end("r9")
        pairs = pairs_of(frames, "r9")
        flags = [header(b)[5] for b, _m in pairs]
        check(flags[0] & 1 and not any(f & 1 for f in flags[1:]), f"think flag on the first pair only: {flags}")
        n_floats = [(len(b) - 24) // 4 for b, _m in pairs]
        check(n_floats[0] == n_floats[1] + 180, "the think path rides in the first binary frame only")
        m0, m1 = pairs[0][1], pairs[1][1]
        check(m0["think_seed"] == 99 and m0["think_read_token"] is not None and m0["think_generated_text"] is not None,
              f"think echo on the first pair: {m0['think_seed']}")
        check(m1["think_seed"] is None and m1["think_read_token"] is None and m1["think_generated_text"] is None,
              "no think echo on later pairs")
        check(m0["think_stop_reason"] == "length" and m0["think_skipped"] is False, f"how the thought ended: {m0['think_stop_reason']}")
        check(all(m["think_stop_reason"] is None and m["think_skipped"] is False for _b, m in pairs[1:]),
              "no think stop reason on later pairs")
        check(rt.writes[-1].startswith("<|think|>"), "the thinking prompt was written")
        check(frames[-1][1]["cancelled"] is False, "a thinking comparison left alone is not cancelled")

        # a stop while the answer is written: the comparison is skipped, and the one pair says so
        rt.write_s = 0.02
        n_writes = len(rt.writes)
        s.sweep({"request_id": "r9b", "prompt": "long", "max_new_tokens": 60, "sight_every_token": True, "compare_think": True})
        first = s.recv()
        s.send({"action": "cancel_sweep", "request_id": "r9b"})
        frames = [first] + s.until_end("r9b")
        pairs = pairs_of(frames, "r9b")
        rt.write_s = 0.01
        m0 = pairs[0][1] if pairs else {}
        check(len(pairs) == 1 and m0.get("think_skipped") is True and m0.get("think_stop_reason") is None
              and m0.get("think_generated_text") is None and m0.get("think_seed") is None and not header(pairs[0][0])[5] & 1,
              f"a stop while writing skips the thinking comparison: {m0.get('think_skipped')} {m0.get('think_stop_reason')}")
        check(not any(w.startswith("<|think|>") for w in rt.writes[n_writes:]), "the skipped thinking prompt was written")
        check(frames[-1][1]["cancelled"] is True, f"its end: {frames[-1][1]}")

        # a stop while the first position is read (it arrives before the comparison begins): skipped, the stream ends
        rt.conn, rt.stop_at = conns[-1], ("analyse", 0)
        n_writes = len(rt.writes)
        s.sweep({"request_id": "r9c", "prompt": "hi", "max_new_tokens": 4, "sight_every_token": True, "compare_think": True})
        frames = s.until_end("r9c")
        pairs = pairs_of(frames, "r9c")
        m0 = pairs[0][1] if pairs else {}
        check(len(pairs) == 1 and m0.get("think_skipped") is True and m0.get("last") is True
              and not any(w.startswith("<|think|>") for w in rt.writes[n_writes:]),
              f"a stop before the comparison begins skips it: {len(pairs)} pairs, skipped {m0.get('think_skipped')}")
        check(frames[-1][1]["cancelled"] is True and frames[-1][1]["n_sent"] == 1, f"its end: {frames[-1][1]}")

        # a stop while the thought is written, with one sighting: cut short, said so, and the request cancelled
        rt.stop_at = ("think", 2)
        s.sweep({"request_id": "r9d", "prompt": "hi", "max_new_tokens": 4, "sight_tokens": [0], "compare_think": True})
        frames = s.until_end("r9d")
        pairs = pairs_of(frames, "r9d")
        m0 = pairs[0][1] if pairs else {}
        check(len(pairs) == 1 and m0.get("think_stop_reason") == "stopped" and m0.get("think_skipped") is False
              and m0.get("think_generated_text") == "·10·11·12" and header(pairs[0][0])[5] & 1,
              f"a stop while thinking cuts it short: {m0.get('think_stop_reason')} {m0.get('think_generated_text')!r}")
        check(frames[-1][1]["cancelled"] is True and frames[-1][1]["n_sent"] == 1,
              f"cancelled with one sighting: {frames[-1][1]}")
        rt.stop_at, rt.conn = None, None
        # the next request on the socket is not stopped by the last one's cancel
        s.sweep({"request_id": "r9e", "prompt": "hi", "max_new_tokens": 2, "sight_tokens": [0], "compare_think": True})
        frames = s.until_end("r9e")
        m0 = (pairs_of(frames, "r9e") or [(b"", {})])[0][1]
        check(m0.get("think_stop_reason") == "length" and frames[-1][1]["cancelled"] is False, f"the next request: {frames[-1][1]}")
        s.sweep({"request_id": "r10", "prompt": "a", "max_new_tokens": 1})
        s.send({"action": "larql", "query": "BETWEEN"})
        s.sweep({"request_id": "r11", "prompt": "b", "max_new_tokens": 1})
        frames = s.until_end("r11", limit=50)
        order = [(o.get("type"), o.get("request_id")) for k, o in frames if k == "json"]
        i_end10 = order.index(("theodolite_sweep_end", "r10"))
        i_larql = order.index(("larql_result", None))
        i_first11 = next(i for i, t in enumerate(order) if t[1] == "r11")
        check(i_end10 < i_larql < i_first11, f"actions keep their order: {order}")


def main() -> int:
    check_rules()
    check_socket()
    if failures:
        print(f"\nsweep protocol: {len(failures)} of {passed + len(failures)} checks FAILED")
        return 1
    print(f"\nsweep protocol: all {passed} checks hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())

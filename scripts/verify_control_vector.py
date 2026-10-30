#!/usr/bin/env python3
"""Does a neuron's push carry over to llama.cpp? The comparison fixed in ARCHITECTURE.md §6, on Gemma-4-31B (GPU).

    .venv/bin/python -I scripts/verify_control_vector.py instrument --out DIR      # the instrument's side, on the card
    .venv/bin/python -I scripts/verify_control_vector.py llama --out DIR --gguf FILE --llama-server PATH   # llama.cpp's
    .venv/bin/python -I scripts/verify_control_vector.py read --out DIR            # the reading (CPU)
    .venv/bin/python -I scripts/verify_control_vector.py all --stand-in --out DIR --llama-server PATH      # on the CPU

The :8765 server holds the card: stop it first. The two sides run one after the other, never together (the 4-bit
model and the GGUF do not fit on one card side by side), each in its own process.

What is compared. Layer 45's two top-ranked neurons in its census (13641 and 21200), pushed at break_rel's four doses
(±1, ±2) on its six questions, greedy, 48 word-pieces, thinking off, both engines given the census's own prompt ids
(read from the run's meta; llama.cpp's `/detokenize` of them checked against the instrument's tokenizer first).
- The instrument: the census's answers as written (batch 64, the start mark spared), read from the run; and written
  again here in batches of nine per question (the unpushed answer and the eight pushes), once with the start mark
  spared and once with it pushed; and the census's pushed answers read back, teacher-forced, in one forward of three
  rows (unpushed, pushed with the start mark spared, pushed with it pushed): each word-piece's log-probability.
- llama.cpp: `llama-server` on the GGUF, every layer on the card, its other settings left at their defaults, the
  exported file (dose 1, from the checkpoint) loaded at SCALE = the dose, one server per push and one unpushed. Each
  writes the six greedy answers, and reads the census's pushed answers back teacher-forced: each next word-piece is
  made the server's choice by a logit bias, and the log-probability it reports is the raw logits' softmax, before any
  sampler (`populate_token_probs`, tools/server/server-context.cpp), so the bias does not touch it. A piece the server
  holds back (bytes that are not yet a whole character) comes with no log-probability and is left out, and counted.

The reading (`read`), as fixed before the run:
1. Shared prefix: the word-pieces llama.cpp's pushed answer shares with the census's from the start (of 48), per
   (neuron, dose, question), against the floor the weights alone set (the six unpushed answers' shared prefix), and
   the instrument's own floor across batch shapes (its batch-of-nine answers against the census's).
2. The push's change in each word-piece's log-probability (pushed minus unpushed) on the census's pushed answers, in
   each engine: their correlation over every word-piece, and the median of llama.cpp's change over the instrument's
   where the instrument's moved by more than 0.5. The push carries over if the correlation is at least 0.8 and that
   median ratio is between 0.67 and 1.5.
3. The start mark: 1 and 2 again against the instrument with the start mark pushed, and the instrument's change with
   the start mark pushed against its change with it spared, so the start mark's part of any gap is named apart.
4. Described, not judged: how often a neuron's five top census words come out in llama.cpp's pushed answers, beside
   the census's and the unpushed answers'.

`--stand-in` runs every step on the CPU stand-in (scripts/atlas_stand_in.py, written as a float32 GGUF, with a stats
index of its own stream lengths; its layer 3, neurons 5 and 77, the instrument's own answers standing for a census's),
for trying the script; its numbers say nothing of the 31B.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

LAYER, NEURONS = 45, (13641, 21200)
DOSES = (-2.0, -1.0, 1.0, 2.0)
RUN = "L45-break_rel"
MAX_NEW = 48
STAND_IN_LAYER, STAND_IN_NEURONS = 3, (5, 77)
BAR_R, BAR_RATIO, MOVED = 0.8, (0.67, 1.5), 0.5


def key(n: int, d: float, q: str) -> str:
    return f"{int(n)}:{d:+g}:{q}"


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------------- the setting


def stand_in_files(out: Path) -> Tuple[Any, Path, Path]:
    """(runtime, its float32 GGUF, its stats index): the stand-in, as every step builds it alike (seed 0)."""
    import torch

    from atlas_stand_in import CORPUS, stand_in_hf, stand_in_runtime, stand_in_tokenizer
    from backend import atlas_ops as ao
    from backend import gguf_model as gm
    from backend.atlas_index import save_npz
    from backend.causal_check import language_model

    torch.set_grad_enabled(False)
    tok = stand_in_tokenizer()
    rt = stand_in_runtime(stand_in_hf(tok), tok)
    gguf_path, stats = out / "stand-in-f32.gguf", out / "stand-in-act_scale.npz"
    if not gguf_path.exists():
        gm.write_gemma4_gguf(rt.hf_model, gguf_path, tokenizer=tok, qtype="F32")
    if not stats.exists():
        lm = language_model(rt.hf_model)
        ids = [[tok.bos_token_id] + tok(t)["input_ids"] for t in CORPUS]
        resid_q = np.zeros((len(lm.layers), 3), np.float32)
        resid_q[:, 1] = ao.stream_p50(ao.activation_extremes(lm, ids))
        save_npz(stats, resid_q=resid_q, tokens=np.int64(sum(len(x) - 1 for x in ids)),
                 meta=np.array(json.dumps({"model": rt.model_id, "stand_in": True, "built_at": "verify_control_vector",
                                           "corpus": "the stand-in's corpus"})))
    return rt, gguf_path, stats


def census_setting(stand_in: bool, rt: Any = None) -> Dict[str, Any]:
    """The questions' prompt ids, the stop ids, p50 and the layer and neurons; on the 31B, the census's answers as
    written for the neurons and the unpushed ones, and each neuron's five top words, read from the run."""
    from backend import census as cz

    if stand_in:
        prompts = cz.design_questions("break_rel")
        return {"stand_in": True, "layer": STAND_IN_LAYER, "neurons": list(STAND_IN_NEURONS), "doses": list(DOSES),
                "prompt_ids": {q: rt.tokenize_prompt(rt.compose_prompt(v, chat=True)) for q, v in prompts.items()},
                "stop_ids": list(rt.end_of_turn_ids()), "model": rt.model_id}
    runs = cz.census_dir()
    meta = json.loads((runs / f"{RUN}.meta.json").read_text(encoding="utf-8"))
    census: Dict[str, List[int]] = {}
    unpushed: Dict[str, List[int]] = {}
    heads = {f'"kind": "neuron", "neuron": {n},': n for n in NEURONS}
    with open(runs / f"{RUN}.jsonl", encoding="utf-8") as f:
        for line in f:
            if '"kind": "baseline"' in line[:60] or any(h in line[:80] for h in heads):
                r = json.loads(line)
                if r["kind"] == "baseline":
                    unpushed[r["prompt"]] = r["token_ids"]
                else:
                    census[key(r["neuron"], float(r["dose"]), r["prompt"])] = r["token_ids"]
    words: Dict[str, List[str]] = {}
    with open(runs / f"{RUN}.units.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") == "neuron" and int(r["neuron"]) in NEURONS:
                words[str(r["neuron"])] = [w[0] for w in r["words"][:5]]
    want = len(NEURONS) * len(DOSES) * len(meta["prompts"])
    if len(census) != want or len(unpushed) != len(meta["prompts"]):
        raise SystemExit(f"{RUN} holds {len(census)} of the {want} pushed answers and {len(unpushed)} unpushed ones.")
    return {"stand_in": False, "layer": int(meta["layer"]), "neurons": list(NEURONS), "doses": list(DOSES),
            "prompt_ids": meta["prompt_ids"], "stop_ids": meta["stop_ids"], "model": meta["model"], "p50": meta["p50"],
            "max_new": meta["max_new"], "batch": meta["batch"], "census": census, "census_unpushed": unpushed,
            "words": words, "run": RUN, "runtime": meta.get("runtime")}


# ------------------------------------------------------------------------------------------------- the instrument


def instrument(out: Path, stand_in: bool, unpacker: str) -> None:
    import torch
    import torch.nn.functional as F

    from backend import atlas_ops as ao
    from backend.causal_check import language_model, model_device

    torch.set_grad_enabled(False)
    t0 = time.time()
    if stand_in:
        rt, _g, stats = stand_in_files(out)
        with np.load(stats) as z:
            p50 = float(z["resid_q"][STAND_IN_LAYER, 1])
    else:
        from census_run import free_lens

        from backend import w4a16_fast
        from backend.runtime_factory import make_runtime

        rt = make_runtime(load_jlens=False).load()
        if getattr(rt, "jpp_lenses", None):
            free_lens(rt)
        w4a16_fast.install(rt, unpacker, log=lambda s: print(s, flush=True))
    st = census_setting(stand_in, rt)
    if not stand_in:
        p50 = float(st["p50"])
    lm = language_model(rt.hf_model)
    dev = model_device(lm)
    w = ao.Weights(lm, rt)
    D, layer, bos = w.d_model, st["layer"], rt.tokenizer.bos_token_id
    softcap = ao.softcap_of(rt.hf_model)
    u = w.write_dirs(layer, st["neurons"])
    pushes = [(n, d) for n in st["neurons"] for d in st["doses"]]
    deltas = torch.zeros(1 + len(pushes), D, dtype=torch.float32, device=w.device)
    for i, (n, d) in enumerate(pushes):
        deltas[1 + i] = u[st["neurons"].index(n)] * (d * p50)  # take_census's push

    def write(ids: Sequence[int], spare_bos: bool) -> List[List[int]]:
        iv = ao.Interventions(rows=len(deltas), adds={layer: deltas})
        rows: List[List[int]] = [[] for _ in range(len(deltas))]
        for s in ao.write_rows(lm, rt.lm_head_weight, [list(ids)] * len(deltas), iv, max_new=MAX_NEW, stop_ids=st["stop_ids"],
                               bos_id=bos, spare_bos=spare_bos, softcap=softcap):
            rows[s.row].append(int(s.token_id))
        return rows

    answers: Dict[str, Dict[str, List[int]]] = {"spared": {}, "bos": {}, "unpushed": {}}
    for q, ids in st["prompt_ids"].items():
        for how, spare in (("spared", True), ("bos", False)):
            rows = write(ids, spare)
            if how == "spared":
                answers["unpushed"][q] = rows[0]
            for i, (n, d) in enumerate(pushes):
                answers[how][key(n, d, q)] = rows[1 + i]
        print(f"  wrote {q} ({time.time() - t0:.0f} s)", flush=True)
    if stand_in:  # the instrument's own answers stand for a census's
        st["census"] = dict(answers["spared"])
        st["census_unpushed"] = dict(answers["unpushed"])

    def forced(ids: Sequence[int], answer: Sequence[int], delta: torch.Tensor) -> Dict[str, List[float]]:
        """Each piece of `answer` read back after `ids`, in one forward of three rows: unpushed; pushed, the start
        mark spared; pushed, the start mark pushed."""
        full = list(ids) + list(answer)
        x = torch.tensor([full] * 3, dtype=torch.long, device=dev)
        keep = ao._Keep()
        keep.mask = torch.ones(3, len(full), dtype=torch.bool, device=dev)
        keep.mask[1, 0] = full[0] != bos
        adds = torch.zeros(3, D, dtype=torch.float32, device=w.device)
        adds[1] = delta
        adds[2] = delta
        with ao.installed(lm, ao.Interventions(rows=3, adds={layer: adds}), keep):
            h = lm(input_ids=x, use_cache=False).last_hidden_state[:, len(ids) - 1 : len(full) - 1]
        A = len(answer)
        logits = ao._model_logits(lm, rt.lm_head_weight, h.reshape(3 * A, -1), softcap)
        lsm = F.log_softmax(logits.float(), dim=-1).reshape(3, A, -1)
        tgt = torch.tensor(list(answer), dtype=torch.long, device=lsm.device)
        lp = lsm.gather(-1, tgt[None, :, None].expand(3, A, 1)).squeeze(-1).double().cpu().tolist()
        return {"unpushed": lp[0], "spared": lp[1], "bos": lp[2]}

    tf: Dict[str, Dict[str, List[float]]] = {}
    for i, (n, d) in enumerate(pushes):
        for q, ids in st["prompt_ids"].items():
            k = key(n, d, q)
            tf[k] = forced(ids, st["census"][k], deltas[1 + i])
    print(f"  read back {len(tf)} census answers ({time.time() - t0:.0f} s)", flush=True)
    from backend import census as cz

    write_json(out / "instrument.json", {
        "setting": st, "p50": p50, "answers": answers, "forced": tf, "seconds": round(time.time() - t0, 1),
        "runtime": cz.runtime_record(w.device),
        "model_id": rt.model_id, "unpacker": None if stand_in else unpacker})
    print(f"wrote {out / 'instrument.json'}", flush=True)


# ------------------------------------------------------------------------------------------------------- llama.cpp


class Server:
    """A llama-server over `model` with `vectors` [(file, SCALE)], for the length of a with-block."""

    def __init__(self, exe: Path, model: Path, vectors: Sequence[Tuple[Path, float]], log: Path, gpu: bool, port: int) -> None:
        self.cmd = [str(exe), "-m", str(model), "--host", "127.0.0.1", "--port", str(port), "-c", "4096", "-np", "1",
                    "-ngl", "99" if gpu else "0", "--no-warmup"]
        if not gpu:
            self.cmd += ["-t", "4", "-ctk", "f32", "-ctv", "f32", "-fa", "off"]
        if vectors:
            self.cmd += ["--control-vector-scaled", ",".join(f"{p}:{s:g}" for p, s in vectors)]
        self.env = {k: v for k, v in os.environ.items() if "TOKEN" not in k.upper() and "SECRET" not in k.upper()}
        if not gpu:
            self.env["CUDA_VISIBLE_DEVICES"] = ""
        self.log, self.port = log, port
        self.proc: Optional[subprocess.Popen] = None

    def __enter__(self) -> "Server":
        self._log = open(self.log, "ab")
        self._log.write(("\n$ " + " ".join(self.cmd) + "\n").encode())
        self._log.flush()
        self.proc = subprocess.Popen(self.cmd, stdout=self._log, stderr=subprocess.STDOUT, env=self.env)
        for _ in range(1800):
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server stopped at start (see {self.log})")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        return self
            except (OSError, ValueError):
                pass
            time.sleep(0.2)
        raise RuntimeError(f"llama-server did not come up (see {self.log})")

    def __exit__(self, *_a: Any) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self._log.close()

    def post(self, path: str, body: Dict[str, Any], stream: bool = False) -> Any:
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            if not stream:
                return json.loads(r.read())
            return [json.loads(ln[6:]) for ln in (x.decode("utf-8", "replace").strip() for x in r) if ln.startswith("data: ")]

    def write(self, ids: Sequence[int]) -> Dict[str, Any]:
        """A greedy answer of at most MAX_NEW pieces: its ids and how it ended. Should the server's output parser
        refuse the whole answer, it is written again streamed, and what came before the refusal is kept (`cut`)."""
        body = {"prompt": list(map(int, ids)), "n_predict": MAX_NEW, "temperature": 0, "cache_prompt": False,
                "return_tokens": True}
        try:
            r = self.post("/completion", body)
            return {"ids": [int(t) for t in r.get("tokens", [])], "stop_type": r.get("stop_type"),
                    "predicted": r.get("tokens_predicted"), "cut": False}
        except urllib.error.HTTPError as exc:
            why = exc.read().decode("utf-8", "replace")[:200]
        got: List[int] = []
        for msg in self.post("/completion", dict(body, stream=True), stream=True):
            got += [int(t) for t in msg.get("tokens", [])]
        return {"ids": got, "stop_type": None, "predicted": None, "cut": True, "why": why}

    def forced(self, ids: Sequence[int], answer: Sequence[int]) -> List[Optional[float]]:
        """Each piece of `answer`'s log-probability after `ids` and the pieces before it (NaN where none came)."""
        out: List[Optional[float]] = []
        for t in range(len(answer)):
            body = {"prompt": list(map(int, ids)) + list(map(int, answer[:t])), "n_predict": 1, "temperature": 0,
                    "n_probs": 1, "cache_prompt": True, "return_tokens": True,
                    "logit_bias": [[int(answer[t]), 1000.0]]}
            try:
                r = self.post("/completion", body)
                cp = r.get("completion_probabilities") or []
                ok = cp and int(cp[0]["id"]) == int(answer[t])
                out.append(float(cp[0]["logprob"]) if ok else None)
            except urllib.error.HTTPError:
                out.append(None)
        return out


def free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def llama(out: Path, stand_in: bool, gguf: Path, exe: Path, gpu: bool) -> None:
    import export_control_vector as ecv

    t0 = time.time()
    inst = json.loads((out / "instrument.json").read_text(encoding="utf-8"))
    st = inst["setting"]
    if stand_in:
        rt, gguf, stats = stand_in_files(out)
        tok, model = rt.tokenizer, rt.model_id
        source = ecv.GGUFColumns(gguf)
    else:
        from transformers import AutoTokenizer

        from backend.atlas_index import index_dir

        tok, model, stats = AutoTokenizer.from_pretrained(ecv.MODEL_ID), ecv.MODEL_ID, index_dir() / "act_scale.npz"
        source = ecv.CheckpointColumns(ecv.checkpoint_files(None))
    files: Dict[int, Path] = {}
    for n in st["neurons"]:
        f = out / f"theodolite-L{st['layer']}-n{n}.gguf"
        _p, dirs, rec = ecv.export([ecv.Entry(st["layer"], n, 1.0)], f, source, stats, model)
        files[n] = f
        if abs(rec["p50"][0] - inst["p50"]) > 1e-4 * inst["p50"]:
            raise SystemExit(f"The file's p50 ({rec['p50'][0]}) is not the instrument's ({inst['p50']}).")
    log = out / "llama-server.log"
    port = free_port()
    res: Dict[str, Any] = {"files": {str(n): f.name for n, f in files.items()}, "gguf": Path(gguf).name, "gpu": gpu,
                           "answers": {"unpushed": {}, "pushed": {}}, "forced": {}}
    pushes = [(n, d) for n in st["neurons"] for d in st["doses"]]
    with Server(exe, Path(gguf), [], log, gpu, port) as srv:
        res["server"] = {"cmd": [Path(srv.cmd[0]).name] + srv.cmd[1:], "props": _props(srv)}
        res["prompts_match"] = {}
        for q, ids in st["prompt_ids"].items():
            got = srv.post("/detokenize", {"tokens": ids}).get("content", "")
            res["prompts_match"][q] = got == tok.decode(ids, skip_special_tokens=False)
        if not all(res["prompts_match"].values()):
            raise SystemExit(f"llama.cpp reads the census's prompt ids otherwise than the instrument: {res['prompts_match']}")
        for q, ids in st["prompt_ids"].items():
            res["answers"]["unpushed"][q] = srv.write(ids)
        for n, d in pushes:
            for q, ids in st["prompt_ids"].items():
                k = key(n, d, q)
                res["forced"].setdefault(k, {})["unpushed"] = srv.forced(ids, st["census"][k])
        print(f"  unpushed: written and read back ({time.time() - t0:.0f} s)", flush=True)
    for n, d in pushes:
        with Server(exe, Path(gguf), [(files[n], d)], log, gpu, port) as srv:
            for q, ids in st["prompt_ids"].items():
                k = key(n, d, q)
                res["answers"]["pushed"][k] = srv.write(ids)
                res["forced"][k]["pushed"] = srv.forced(ids, st["census"][k])
        print(f"  neuron {n} at SCALE {d:+g}: written and read back ({time.time() - t0:.0f} s)", flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    write_json(out / "llama.json", res)
    print(f"wrote {out / 'llama.json'}", flush=True)


def _props(srv: Server) -> Dict[str, Any]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/props", timeout=10) as r:
            p = json.loads(r.read())
        return {k: p.get(k) for k in ("build_info", "model_path", "total_slots") if k in p}
    except (OSError, ValueError):
        return {}


# ------------------------------------------------------------------------------------------------------- the reading


def strip(ids: Sequence[int], stop: Sequence[int]) -> List[int]:
    """The answer's pieces up to its first end mark, the mark left out."""
    out: List[int] = []
    for t in ids:
        if int(t) in stop:
            break
        out.append(int(t))
    return out[:MAX_NEW]


def shared(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def summary_of(v: Sequence[float]) -> Dict[str, Any]:
    v = list(v)
    if not v:
        return {"n": 0}
    return {"n": len(v), "median": float(np.median(v)), "mean": round(float(np.mean(v)), 2), "min": min(v), "max": max(v),
            "whole": sum(1 for x in v if x == MAX_NEW)}


def delta_read(pairs: Sequence[Tuple[float, float]]) -> Dict[str, Any]:
    """Pearson's r of (reference, other) changes, and the median of other / reference where |reference| > MOVED."""
    if len(pairs) < 3:
        return {"n": len(pairs)}
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    big = np.abs(a) > MOVED
    ratio = float(np.median(b[big] / a[big])) if big.any() else None
    r = float(np.corrcoef(a, b)[0, 1])
    return {"n": len(pairs), "r": round(r, 4), "moved": int(big.sum()), "median_ratio": None if ratio is None else round(ratio, 4),
            "carries_over": bool(r >= BAR_R and ratio is not None and BAR_RATIO[0] <= ratio <= BAR_RATIO[1])}


def read(out: Path) -> Dict[str, Any]:
    from backend import census as cz

    inst = json.loads((out / "instrument.json").read_text(encoding="utf-8"))
    ll = json.loads((out / "llama.json").read_text(encoding="utf-8"))
    st = inst["setting"]
    stop = [int(t) for t in st["stop_ids"]]
    census, cen0 = st["census"], st["census_unpushed"]
    A, L = inst["answers"], ll["answers"]
    pushes = [(n, d) for n in st["neurons"] for d in st["doses"]]
    qs = list(st["prompt_ids"])
    S = lambda x: strip(x, stop)  # noqa: E731
    lids = lambda r: S(r["ids"])  # noqa: E731
    pre: Dict[str, Any] = {
        "floor: llama.cpp unpushed / census unpushed": summary_of([shared(lids(L["unpushed"][q]), S(cen0[q])) for q in qs]),
        "floor: instrument unpushed, batch 9 / census unpushed, batch 64": summary_of(
            [shared(S(A["unpushed"][q]), S(cen0[q])) for q in qs]),
        "floor: instrument pushed, batch 9 / census, batch 64": summary_of(
            [shared(S(A["spared"][key(n, d, q)]), S(census[key(n, d, q)])) for n, d in pushes for q in qs]),
    }
    for name, ref in (("census (start mark spared)", lambda k: census[k]), ("instrument, start mark spared, batch 9",
                                                                          lambda k: A["spared"][k]),
                      ("instrument, start mark pushed, batch 9", lambda k: A["bos"][k])):
        pre[f"llama.cpp pushed / {name}"] = summary_of(
            [shared(lids(L["pushed"][key(n, d, q)]), S(ref(key(n, d, q)))) for n, d in pushes for q in qs])
        pre[f"llama.cpp pushed / {name}, by dose"] = {
            f"{d:+g}": summary_of([shared(lids(L["pushed"][key(n, d, q)]), S(ref(key(n, d, q)))) for n in st["neurons"] for q in qs])
            for d in st["doses"]}
    pre["instrument: start mark pushed / spared, batch 9"] = summary_of(
        [shared(S(A["bos"][key(n, d, q)]), S(A["spared"][key(n, d, q)])) for n, d in pushes for q in qs])
    pre["llama.cpp pushed / llama.cpp unpushed"] = summary_of(
        [shared(lids(L["pushed"][key(n, d, q)]), lids(L["unpushed"][q])) for n, d in pushes for q in qs])
    pre["census pushed / census unpushed"] = summary_of(
        [shared(S(census[key(n, d, q)]), S(cen0[q])) for n, d in pushes for q in qs])

    pairs: Dict[str, List[Tuple[float, float]]] = {"llama / census": [], "llama / start mark pushed": [],
                                                   "start mark pushed / census": []}
    by_dose: Dict[str, List[Tuple[float, float]]] = {f"{d:+g}": [] for d in st["doses"]}
    lost = total = 0
    for n, d in pushes:
        for q in qs:
            k = key(n, d, q)
            fi, fl = inst["forced"][k], ll["forced"][k]
            for t in range(len(fi["unpushed"])):
                total += 1
                di = fi["spared"][t] - fi["unpushed"][t]
                db = fi["bos"][t] - fi["unpushed"][t]
                pairs["start mark pushed / census"].append((di, db))
                if fl["pushed"][t] is None or fl["unpushed"][t] is None:
                    lost += 1
                    continue
                dl = fl["pushed"][t] - fl["unpushed"][t]
                pairs["llama / census"].append((di, dl))
                pairs["llama / start mark pushed"].append((db, dl))
                by_dose[f"{d:+g}"].append((di, dl))
    deltas = {k: delta_read(v) for k, v in pairs.items()}
    deltas["llama / census, by dose"] = {k: delta_read(v) for k, v in by_dose.items()}
    deltas["pieces read back"] = total
    deltas["pieces llama.cpp gave no log-probability for"] = lost

    words: Dict[str, Any] = {}
    if st.get("words"):
        tok = _tokenizer(st)
        has = lambda ids, ws: bool(cz.words(tok.decode(S(ids), skip_special_tokens=True)) & set(ws))  # noqa: E731
        for n in st["neurons"]:
            ws = st["words"][str(n)]
            row: Dict[str, Any] = {"words": ws, "unpushed (census)": sum(has(cen0[q], ws) for q in qs),
                                   "unpushed (llama.cpp)": sum(has(L["unpushed"][q]["ids"], ws) for q in qs)}
            for d in st["doses"]:
                row[f"{d:+g}"] = {"census": sum(has(census[key(n, d, q)], ws) for q in qs),
                                  "llama.cpp": sum(has(L["pushed"][key(n, d, q)]["ids"], ws) for q in qs), "of": len(qs)}
            words[str(n)] = row
    cut = [k for k, r in L["pushed"].items() if r.get("cut")] + [q for q, r in L["unpushed"].items() if r.get("cut")]
    verdict = deltas["llama / census"].get("carries_over")
    res = {"what": ("the comparison fixed in ARCHITECTURE.md §6: a neuron's push through the instrument and through "
                    "llama.cpp with the exported control vector"),
           "stand_in": st.get("stand_in"), "layer": st["layer"], "neurons": st["neurons"], "doses": st["doses"],
           "gguf": ll.get("gguf"), "server": ll.get("server"), "shared_prefix": pre, "change_in_odds": deltas,
           "words": words, "answers_cut_by_the_server": cut,
           "bar": {"r_at_least": BAR_R, "median_ratio_between": list(BAR_RATIO), "where_moved_more_than": MOVED},
           "carries_over": verdict, "seconds": {"instrument": inst.get("seconds"), "llama": ll.get("seconds")}}
    write_json(out / "summary.json", res)
    return res


def _tokenizer(st: Dict[str, Any]) -> Any:
    if st.get("stand_in"):
        from atlas_stand_in import stand_in_tokenizer

        return stand_in_tokenizer()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(st["model"])


def show(res: Dict[str, Any]) -> None:
    print(f"\nlayer {res['layer']}, neurons {res['neurons']}, doses {res['doses']}, {res['gguf']}")
    print("shared prefix (word-pieces of 48; median, mean, whole answers):")
    for k, v in res["shared_prefix"].items():
        if "by dose" in k:
            print(f"  {k}: " + ", ".join(f"{d} {s['median']:g}" for d, s in v.items()))
        else:
            print(f"  {k}: median {v['median']:g}, mean {v['mean']}, whole {v['whole']} of {v['n']}")
    print("the push's change in each piece's log-probability:")
    for k, v in res["change_in_odds"].items():
        if isinstance(v, dict) and "r" in v:
            print(f"  {k}: r {v['r']}, median ratio {v['median_ratio']} over {v['moved']} moved of {v['n']}"
                  + (f" -> carries over: {v['carries_over']}" if k == "llama / census" else ""))
        elif isinstance(v, dict):
            print(f"  {k}: " + "; ".join(f"{d} r {s.get('r')} ratio {s.get('median_ratio')}" for d, s in v.items()))
        else:
            print(f"  {k}: {v}")
    for n, row in res["words"].items():
        print(f"  neuron {n} words {row['words']}: unpushed census {row['unpushed (census)']}, llama.cpp "
              f"{row['unpushed (llama.cpp)']}; " + "; ".join(
                  f"{d} census {row[d]['census']} llama.cpp {row[d]['llama.cpp']} of {row[d]['of']}"
                  for d in row if d[0] in "+-"))
    if res["answers_cut_by_the_server"]:
        print(f"  answers the server's parser cut: {res['answers_cut_by_the_server']}")
    print(f"carries over (r >= {BAR_R}, median ratio in {list(BAR_RATIO)}): {res['carries_over']}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["instrument", "llama", "read", "all"])
    ap.add_argument("--out", required=True, help="the folder the steps write to and read from")
    ap.add_argument("--stand-in", action="store_true", help="on the CPU stand-in, for trying the script")
    ap.add_argument("--gguf", help="the GGUF llama.cpp runs (the 31B; the stand-in writes its own)")
    ap.add_argument("--llama-server", help="the llama-server binary")
    ap.add_argument("--unpacker", default="compiled", help="the W4A16 unpacker (the census's: compiled)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.stand_in:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from atlas_stand_in import use_stand_in_atlas

        use_stand_in_atlas()
    os.environ["HF_HUB_OFFLINE"] = "1"
    if a.step in ("instrument", "all"):
        instrument(out, a.stand_in, a.unpacker)
    if a.step in ("llama", "all"):
        if not a.llama_server or (not a.stand_in and not a.gguf):
            raise SystemExit("The llama step needs --llama-server (and --gguf on the 31B).")
        llama(out, a.stand_in, Path(a.gguf) if a.gguf else Path(), Path(a.llama_server), gpu=not a.stand_in)
    if a.step in ("read", "all"):
        show(read(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

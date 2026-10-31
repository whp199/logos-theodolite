#!/usr/bin/env python3
"""A CPU-only check of scripts/export_control_vector.py on the tiny random Gemma 4 of scripts/atlas_stand_in.py, in a
temporary folder:
1. the census's push: the stand-in's down projections packed as the W4A16 checkpoint packs them (its post-feed-forward
   norms in bfloat16, as the checkpoint stores them), written as a checkpoint, beside a stats index of the stand-in's
   own stream lengths; a census of two neurons taken on it (`census.take_census`, break_rel's six questions at its four
   doses) with every push it adds kept as it is handed to the writer; each pushed row's push equals the exported
   vector for its (neuron, dose) element for element, and every (neuron, dose) is met;
2. the atlas's push (`atlas_ops.resolve`, the push tab's): one neuron, and a set of three at two layers, against the
   files' vectors (it multiplies dose and p50 in float32, the census in Python's float: within float32 rounding);
3. the file as llama.cpp's loader reads it: direction.<layer> tensors, F32, one-dimensional, the model's width, no
   direction.0; llama.cpp's keys (`general.architecture` controlvector, `controlvector.model_hint`, `layer_count`) and
   theodolite's record (the model, the weights, each entry's layer, neuron, dose and p50, the stats index's sha256);
   the loader's arithmetic (each file times its SCALE, files summed) gives the push at dose x SCALE; the command line
   writes the same vector, and refuses layer 0, a layer or a neuron that is not there, a dose that is not a
   number, a stats index of another model, a file already there, a set without a name, and a set given a --dose;
4. `--gguf`: the same weights written as a float32 GGUF (`gguf_model.write_gemma4_gguf`), its vector equal to the
   census's push element for element;
5. llama.cpp itself, when a llama-server is found (`--llama-server`, else $LLAMA_SERVER, else `llama-server` on the
   PATH; without one this part says so and is skipped): the float32 GGUF served on the CPU (CUDA_VISIBLE_DEVICES
   empty, the cache in float32, no flash attention), and the log-probabilities of every word-piece at the first one
   written to a census question, with the exported files loaded, set against the instrument's own
   (`atlas_ops.write_rows`): unpushed, alike (within 1e-3: ggml's sums and its GELU are not torch's; 4.4e-4 when
   written); a file at SCALE 2 alike the instrument's push at dose 2 at the same layer with the start mark pushed too
   (2.0e-4), and at least 20 times further from it at the layers either side (0.18, 0.13) and with the start mark
   spared (3.9e-2), which is what the instrument does; a set across two layers alike its pushes (1.3e-4). So the layer
   llama.cpp pushes, its scale, its sum and its start mark are read off llama.cpp, not only off its source.

Run: .venv/bin/python -I scripts/check_control_vector.py [--llama-server PATH]   (exits 0 when every check holds)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

import backend.atlas_ops as ao  # noqa: E402
import export_control_vector as ecv  # noqa: E402
from atlas_stand_in import CORPUS, stand_in_hf, stand_in_runtime, stand_in_tokenizer  # noqa: E402
from backend import census as cz  # noqa: E402
from backend import gguf_model as gm  # noqa: E402
from backend.atlas_index import save_npz  # noqa: E402
from backend.causal_check import language_model  # noqa: E402
from backend.engine_runtime import dequantize_w4a16_full  # noqa: E402
from check_dequant import pack_w4a16  # noqa: E402

PY = sys.executable
EXPORT_PY = ROOT_DIR / "scripts" / "export_control_vector.py"
LAYER = 3
NEURONS = (5, 77)
DOSES = (-2.0, -1.0, 1.0, 2.0)  # break_rel's
SET = [ecv.Entry(LAYER, 5, 1.0), ecv.Entry(LAYER, 90, -0.5), ecv.Entry(LAYER - 1, 12, 2.0)]

failures: List[str] = []
passed = 0


def check(ok: bool, what: str) -> None:
    global passed
    if ok:
        passed += 1
    else:
        failures.append(what)
        print(f"  FAIL  {what}")


def section(title: str, before: int) -> None:
    if len(failures) == before:
        print(f"  ok    {title}")


# ------------------------------------------------------------------------------------------------------- the stand-in


def packed_stand_in(tmp: Path) -> Tuple[Any, Path, Path, List[float]]:
    """(runtime, checkpoint folder, stats index, p50): the stand-in in float32 with its down projections replaced by
    their W4A16 unpacking (the instrument's own, `dequantize_w4a16_full`) and its post-feed-forward norms rounded to
    bfloat16; the checkpoint holds the packed codes and scales and the bfloat16 norms under the real checkpoint's
    names; the stats index holds the stand-in's own median stream lengths (`atlas_ops.stream_p50` over its corpus)."""
    from safetensors.torch import save_file

    tok = stand_in_tokenizer()
    hf = stand_in_hf(tok)
    lm = language_model(hf)
    ct: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for l, layer in enumerate(lm.layers):
            P = f"model.language_model.layers.{l}."
            w = layer.mlp.down_proj.weight
            packed, scale = pack_w4a16(w.detach())
            w.copy_(dequantize_w4a16_full(packed, scale).float())
            norm = layer.post_feedforward_layernorm.weight
            norm.copy_(norm.to(torch.bfloat16).float())
            ct[P + "mlp.down_proj.weight_packed"] = packed
            ct[P + "mlp.down_proj.weight_scale"] = scale
            ct[P + "mlp.down_proj.weight_shape"] = torch.tensor(list(w.shape), dtype=torch.int64)
            ct[P + "post_feedforward_layernorm.weight"] = norm.to(torch.bfloat16).clone()
            ct[P + "layer_scalar"] = layer.layer_scalar.detach().to(torch.bfloat16).clone()
    folder = tmp / "checkpoint"
    folder.mkdir()
    save_file({k: v.contiguous() for k, v in ct.items()}, str(folder / "model.safetensors"))
    rt = stand_in_runtime(hf, tok)
    ids = [tok(t)["input_ids"] for t in CORPUS]
    ids = [x if x and x[0] == tok.bos_token_id else [tok.bos_token_id] + x for x in ids]
    p50 = ao.stream_p50(ao.activation_extremes(lm, ids))
    resid_q = np.zeros((len(lm.layers), 3), np.float32)
    resid_q[:, 1] = p50
    stats = tmp / "act_scale.npz"
    save_npz(stats, resid_q=resid_q, tokens=np.int64(sum(len(x) - 1 for x in ids)),
             meta=np.array(json.dumps({"model": rt.model_id, "stand_in": True, "built_at": "the check",
                                       "corpus": "the stand-in's corpus"})))
    return rt, folder, stats, [float(x) for x in resid_q[:, 1]]


def export_one(source: Any, stats: Path, model: str, out: Path, entries: Sequence[ecv.Entry]) -> Dict[int, np.ndarray]:
    _path, directions, _rec = ecv.export(list(entries), out, source, stats, model)
    return directions


def run_cli(args: Sequence[Any]) -> subprocess.CompletedProcess:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1")
    return subprocess.run([PY, "-I", str(EXPORT_PY)] + [str(a) for a in args], env=env, cwd=ROOT_DIR,
                          capture_output=True, text=True, timeout=300)


def llama_load(files: Sequence[Tuple[Path, float]]) -> Dict[int, np.ndarray]:
    """llama.cpp's `common_control_vector_load` in numpy: each direction.<k> times its file's SCALE, summed over the
    tensors of one layer in a file and over the files."""
    out: Dict[int, np.ndarray] = {}
    for path, scale in files:
        directions, _meta = ecv.read_control_vector(path)
        for k, v in directions.items():
            add = v.astype(np.float32) * np.float32(scale)
            out[k] = out[k] + add if k in out else add
    return out


# ------------------------------------------------------------------------------------------------------- the checks


def check_census(rt: Any, folder: Path, stats: Path, p50: List[float], tmp: Path) -> Dict[Tuple[int, float], np.ndarray]:
    before = len(failures)
    pushed: List[torch.Tensor] = []
    write = cz.write_census_batch

    def keep(rt_: Any, layer: int, prompt_ids: Any, delta: torch.Tensor, **kw: Any) -> Any:
        check(layer == LAYER, f"the census pushes at layer {LAYER} (got {layer})")
        pushed.extend(delta[i].detach().clone() for i in range(delta.shape[0]) if bool(delta[i].abs().sum() > 0))
        return write(rt_, layer, prompt_ids, delta, **kw)

    cz.write_census_batch = keep
    try:
        cz.take_census(rt, LAYER, list(NEURONS), tmp / "census.jsonl", p50=p50[LAYER], p50_source="the check's stats index",
                       design_name="break_rel", controls=0, max_new=2, batch=8, p50_file=stats, log=lambda *_a: None)
    finally:
        cz.write_census_batch = write
    source = ecv.CheckpointColumns(ecv.checkpoint_files(str(folder)))
    vec: Dict[Tuple[int, float], np.ndarray] = {}
    for n in NEURONS:
        for d in DOSES:
            vec[(n, d)] = export_one(source, stats, rt.model_id, tmp / f"n{n}-d{d:+g}.gguf", [ecv.Entry(LAYER, n, d)])[LAYER]
    questions = len(cz.design_questions("break_rel"))
    check(len(pushed) == len(NEURONS) * len(DOSES) * questions,
          f"the census pushed {len(NEURONS)} neurons x {len(DOSES)} doses x {questions} questions ({len(pushed)} rows)")
    met = {key: 0 for key in vec}
    unmatched = 0
    for row in pushed:
        hits = [key for key, v in vec.items() if np.array_equal(row.numpy(), v)]
        if len(hits) == 1:
            met[hits[0]] += 1
        else:
            unmatched += 1
    check(unmatched == 0, f"every push the census added is one exported vector, element for element ({unmatched} not)")
    check(all(c == questions for c in met.values()),
          f"every (neuron, dose) met on each of the {questions} questions: {sorted(met.values())}")
    v = vec[(NEURONS[0], 1.0)]
    check(abs(float(np.linalg.norm(v.astype(np.float64))) / p50[LAYER] - 1) < 1e-6,
          "a dose-1 vector is p50 long")
    section(f"the census's push: {len(pushed)} rows, each an exported vector bit for bit", before)
    return vec


def check_atlas(rt: Any, folder: Path, stats: Path, p50: List[float], tmp: Path) -> Dict[int, np.ndarray]:
    before = len(failures)
    lm = language_model(rt.hf_model)
    w = ao.Weights(lm, rt)
    source = ecv.CheckpointColumns(ecv.checkpoint_files(str(folder)))

    def adds(pushes: Sequence[ecv.Entry]) -> Dict[int, np.ndarray]:
        iv = ao.resolve(w, [ao.Push(e.layer, e.neuron, "add", e.dose) for e in pushes], [], [1],
                        p50=lambda l: p50[l], act_scale=lambda _l, _n: 1.0)
        return {l: t[0].numpy() for l, t in iv.adds.items()}

    worst = 0.0
    for e in (ecv.Entry(LAYER, NEURONS[1], -2.0), ecv.Entry(LAYER - 2, 3, 1.0)):
        got = export_one(source, stats, rt.model_id, tmp / f"atlas-{e.layer}-{e.neuron}.gguf", [e])
        ref = adds([e])
        check(set(got) == set(ref) == {e.layer}, f"one push: direction.{e.layer} alone")
        worst = max(worst, float(np.abs(got[e.layer] - ref[e.layer]).max() / np.abs(ref[e.layer]).max()))
    got_set = export_one(source, stats, rt.model_id, tmp / "set.gguf", SET)
    ref = adds(SET)
    check(set(got_set) == set(ref) == {LAYER, LAYER - 1}, f"the set: direction.{LAYER - 1} and direction.{LAYER}")
    for l in ref:
        worst = max(worst, float(np.abs(got_set[l] - ref[l]).max() / np.abs(ref[l]).max()))
    check(worst < 1e-6, f"the atlas's pushes within float32 rounding of the files' (worst {worst:.2e} of the largest element)")
    section(f"the atlas's push: one neuron and a set across two layers (worst {worst:.1e})", before)
    return got_set


def check_file(rt: Any, folder: Path, stats: Path, p50: List[float], tmp: Path, vec: Dict[Tuple[int, float], np.ndarray]) -> None:
    import gguf

    before = len(failures)
    n = NEURONS[0]
    path = tmp / f"n{n}-d+1.gguf"
    r = gguf.GGUFReader(str(path))
    t = {x.name: x for x in r.tensors}
    D = int(language_model(rt.hf_model).norm.weight.shape[0])
    check(list(t) == [f"direction.{LAYER}"], f"one tensor, direction.{LAYER} (got {list(t)})")
    x = t[f"direction.{LAYER}"]
    check(x.tensor_type == gguf.GGMLQuantizationType.F32 and len(x.shape) == 1 and int(x.shape[0]) == D,
          f"F32, one-dimensional, {D} long, as the loader asks")
    meta = gm.read_metadata(r, long_arrays=True)
    check(meta.get("general.architecture") == "controlvector" and meta.get("controlvector.model_hint") == "gemma4"
          and meta.get("controlvector.layer_count") == 1, "llama.cpp's keys, as its cvector-generator writes them")
    rec = {k[len(ecv.KEY):]: v for k, v in meta.items() if k.startswith(ecv.KEY)}
    check(rec.get("model") == rt.model_id and rec.get("layers") == [LAYER] and rec.get("pages") == [LAYER + 1]
          and rec.get("neurons") == [n] and rec.get("doses") == [1.0]
          and abs(rec.get("p50", [0])[0] - p50[LAYER]) < 1e-3 * p50[LAYER], "the record: model, layer, page, neuron, dose, p50")
    check(rec.get("stats_index_sha256") == cz.file_sha256(stats), "the record: the stats index's sha256")
    check("model.safetensors" in str(rec.get("weights")) and all(rec.get(k) for k in ("push", "start_mark", "scale", "caveat")),
          "the record: the weights read, the push, the start mark, the scale and the caveat")

    two = llama_load([(path, 2.0)])
    check(np.array_equal(two[LAYER], vec[(n, 1.0)] * np.float32(2)), "SCALE 2 is the file's vector times 2")
    check(np.array_equal(two[LAYER], vec[(n, 2.0)]), "a dose-1 file at SCALE 2 is the census's push at dose 2, bit for bit")
    both = llama_load([(path, 1.0), (tmp / f"n{NEURONS[1]}-d-1.gguf", 1.0)])
    check(np.allclose(both[LAYER], vec[(n, 1.0)] + vec[(NEURONS[1], -1.0)], rtol=0, atol=0),
          "two files at one layer: their sum")

    out = tmp / "cli.gguf"
    got = run_cli(["--checkpoint", folder, "--stats", stats, "--model", rt.model_id, "--page", LAYER + 1, "--neuron", n,
                   "-o", out])
    check(got.returncode == 0, f"the command line writes a file ({got.stderr.strip()[-300:]})")
    if got.returncode == 0:
        check(np.array_equal(ecv.read_control_vector(out)[0][LAYER], vec[(n, 1.0)]),
              "the command line's file (--page) holds the same vector")
    base = ["--checkpoint", folder, "--stats", stats, "--model", rt.model_id]
    other = tmp / "act_scale_other.npz"
    with np.load(stats, allow_pickle=False) as z:
        save_npz(other, resid_q=z["resid_q"], tokens=z["tokens"], meta=np.array(json.dumps({"model": "another model"})))
    refusals = [
        (["--layer", 0, "--neuron", 1, "-o", tmp / "r0.gguf"], "layer 0"),
        (["--layer", 6, "--neuron", 1, "-o", tmp / "r1.gguf"], "does not exist"),
        (["--layer", LAYER, "--neuron", 128, "-o", tmp / "r2.gguf"], "does not exist"),
        (["--layer", LAYER, "--neuron", 1, "--dose", "nan", "-o", tmp / "r3.gguf"], "finite"),
        (["--layer", LAYER, "--neuron", 1, "-o", out], "already there"),
        (["--set", f"{LAYER}:1,{LAYER - 1}:2"], "-o FILE"),
        (["--set", f"{LAYER}:1", "--dose", 2, "-o", tmp / "r5.gguf"], "not both"),
    ]
    for args, says in refusals:
        got = run_cli(base + args)
        check(got.returncode != 0 and says in got.stderr, f"refused: {' '.join(str(a) for a in args[:4])} ({says}): "
              f"{got.stderr.strip()[-200:]}")
    got = run_cli(["--checkpoint", folder, "--stats", other, "--model", rt.model_id, "--layer", LAYER, "--neuron", 1, "-o",
                   tmp / "r4.gguf"])
    check(got.returncode != 0 and "another model" in got.stderr, "refused: a stats index built for another model")
    check(not any((tmp / f"r{i}.gguf").exists() or (tmp / f"r{i}.gguf.tmp").exists() for i in range(6)),
          "nothing written where refused")
    got = run_cli(["--show", path])
    check(got.returncode == 0 and f"direction.{LAYER}" in got.stdout and "1.0000 x p50" in got.stdout, "--show reads it back")
    section("the file as llama.cpp's loader reads it, the command line and its refusals", before)


def check_gguf_source(rt: Any, stats: Path, tmp: Path, vec: Dict[Tuple[int, float], np.ndarray]) -> Path:
    before = len(failures)
    path = gm.write_gemma4_gguf(rt.hf_model, tmp / "stand-in-f32.gguf", tokenizer=rt.tokenizer, qtype="F32")
    source = ecv.GGUFColumns(path)
    for n, d in ((NEURONS[0], 1.0), (NEURONS[1], -2.0)):
        got = export_one(source, stats, rt.model_id, tmp / f"gguf-{n}.gguf", [ecv.Entry(LAYER, n, d)])
        check(np.array_equal(got[LAYER], vec[(n, d)]), f"--gguf: neuron {n} at dose {d:+g} is the census's push, bit for bit")
    rec = ecv.read_control_vector(tmp / f"gguf-{NEURONS[0]}.gguf")[1]
    check(str(rec.get(ecv.KEY + "weights", "")).startswith("gguf stand-in-f32.gguf"), "the record names the GGUF read")
    section("--gguf: the same weights as a float32 GGUF give the same vector", before)
    return path


# ------------------------------------------------------------------------------------------------- llama.cpp itself


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class LlamaServer:
    """A llama-server on the CPU over `model`, with `vectors` [(file, SCALE)] loaded, for the length of a with-block."""

    def __init__(self, exe: Path, model: Path, vectors: Sequence[Tuple[Path, float]], log: Path) -> None:
        self.exe, self.model, self.vectors, self.log = exe, model, list(vectors), log
        self.port = free_port()
        self.proc: Optional[subprocess.Popen] = None

    def __enter__(self) -> "LlamaServer":
        cmd = [str(self.exe), "-m", str(self.model), "--host", "127.0.0.1", "--port", str(self.port), "-c", "512",
               "-np", "1", "-ngl", "0", "-t", "2", "-ctk", "f32", "-ctv", "f32", "-fa", "off", "--no-warmup"]
        if self.vectors:
            cmd += ["--control-vector-scaled", ",".join(f"{p}:{s:g}" for p, s in self.vectors)]
        env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
        env["CUDA_VISIBLE_DEVICES"] = ""
        self._log = open(self.log, "ab")
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT, env=env)
        for _ in range(200):
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server stopped at start (see {self.log})")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        return self
            except OSError:
                pass
            time.sleep(0.1)
        raise RuntimeError(f"llama-server did not come up (see {self.log})")

    def __exit__(self, *_a: Any) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self._log.close()

    def first_logprobs(self, ids: Sequence[int], vocab: int, barred: Sequence[int]) -> np.ndarray:
        """[vocab] the log-probabilities of every word-piece at the first one written after `ids` (the prompt given as
        ids, so nothing is added to it). The server sends them only with a piece it can show, so the `barred` pieces
        (special marks, and bytes that are not whole characters, which the random stand-in writes often) are kept from
        being chosen by a logit bias; the log-probabilities it sends are the softmax of the raw logits, before any
        sampler (`get_token_probabilities`, tools/server/server-common.cpp), so the bias does not touch them."""
        body = json.dumps({"prompt": list(map(int, ids)), "n_predict": 1, "temperature": 0, "n_probs": vocab,
                           "cache_prompt": False, "logit_bias": [[int(i), False] for i in barred]}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/completion", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            msg = json.loads(r.read())
        probs = msg.get("completion_probabilities")
        if not probs:
            raise RuntimeError(f"llama-server sent no log-probabilities: {json.dumps(msg)[:300]}")
        out = np.full(vocab, np.nan)
        for t in probs[0]["top_logprobs"]:
            out[int(t["id"])] = float(t["logprob"])
        return out


def instrument_logprobs(rt: Any, ids: Sequence[int], adds: Dict[int, np.ndarray], spare_bos: bool) -> np.ndarray:
    """[vocab] the instrument's own log-probabilities at the first word-piece written (`atlas_ops.write_rows`), the
    pushes `adds` ({layer: [D]}) added at the stream leaving each layer."""
    lm = language_model(rt.hf_model)
    V = int(rt.lm_head_weight.shape[0])
    iv = ao.Interventions(rows=1, adds={l: torch.from_numpy(v.copy())[None, :] for l, v in adds.items()}) if adds else None
    step = next(iter(ao.write_rows(lm, rt.lm_head_weight, [list(ids)], iv, max_new=1, stop_ids=[],
                                   bos_id=rt.tokenizer.bos_token_id, spare_bos=spare_bos, softcap=ao.softcap_of(rt.hf_model),
                                   top_k=V)))
    out = np.full(V, np.nan)
    out[np.asarray(step.top_ids)] = np.asarray(step.top_lps, dtype=np.float64)
    return out


def check_llama(exe: Optional[Path], rt: Any, gguf_model: Path, tmp: Path, vec: Dict[Tuple[int, float], np.ndarray],
                set_dirs: Dict[int, np.ndarray]) -> Optional[Dict[str, float]]:
    before = len(failures)
    if exe is None:
        print("  --    llama.cpp itself: not run (no llama-server: --llama-server PATH, or $LLAMA_SERVER)")
        return None
    question = cz.design_questions("break_rel")["who"]
    ids = rt.tokenize_prompt(rt.compose_prompt(question, chat=True))
    check(ids[0] == rt.tokenizer.bos_token_id, "the census's prompt opens with the start mark")
    V = int(rt.lm_head_weight.shape[0])
    tok = rt.tokenizer
    barred = [i for i in range(V) if i in set(tok.all_special_ids) or not tok.decode([i]) or "�" in tok.decode([i])]
    log = tmp / "llama-server.log"
    n = NEURONS[0]
    one = tmp / f"n{n}-d+1.gguf"
    gap = lambda a, b: float(np.nanmax(np.abs(a - b)))  # noqa: E731
    with LlamaServer(exe, gguf_model, [], log) as srv:
        plain = srv.first_logprobs(ids, V, barred)
    with LlamaServer(exe, gguf_model, [(one, 2.0)], log) as srv:
        pushed = srv.first_logprobs(ids, V, barred)
    with LlamaServer(exe, gguf_model, [(tmp / "set.gguf", 1.0)], log) as srv:
        pushed_set = srv.first_logprobs(ids, V, barred)
    two = vec[(n, 2.0)]
    d = {
        "unpushed": gap(plain, instrument_logprobs(rt, ids, {}, True)),
        "same layer, start mark pushed": gap(pushed, instrument_logprobs(rt, ids, {LAYER: two}, False)),
        "same layer, start mark spared": gap(pushed, instrument_logprobs(rt, ids, {LAYER: two}, True)),
        "one layer below": gap(pushed, instrument_logprobs(rt, ids, {LAYER - 1: two}, False)),
        "one layer above": gap(pushed, instrument_logprobs(rt, ids, {LAYER + 1: two}, False)),
        "pushed against unpushed": gap(pushed, plain),
        "the set, start mark pushed": gap(pushed_set, instrument_logprobs(rt, ids, set_dirs, False)),
    }
    check(not np.isnan(plain).any() and not np.isnan(pushed).any(), f"llama-server gave all {V} log-probabilities")
    floor = max(d["unpushed"], d["same layer, start mark pushed"], d["the set, start mark pushed"])
    check(floor < 1e-3, f"llama.cpp and the instrument alike where they should be (worst {floor:.1e} in a log-probability)")
    apart = min(d["same layer, start mark spared"], d["one layer below"], d["one layer above"], d["pushed against unpushed"])
    check(apart > 20 * floor, f"and far apart where they should be (least {apart:.2e}, at least 20x the {floor:.1e} floor)")
    for k, v in d.items():
        print(f"          {k}: largest difference in a log-probability {v:.2e}")
    section("llama.cpp itself: direction.<l> lands after layer l, at SCALE x the dose, summed, the start mark pushed", before)
    return d


def find_llama_server(given: Optional[str]) -> Optional[Path]:
    for c in (given, os.environ.get("LLAMA_SERVER"), shutil.which("llama-server")):
        if c and Path(c).expanduser().is_file():
            return Path(c).expanduser()
    return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--llama-server", help="a llama-server binary for part 5 (else $LLAMA_SERVER, else the PATH's)")
    a = ap.parse_args(argv)
    torch.set_grad_enabled(False)
    exe = find_llama_server(a.llama_server)
    with tempfile.TemporaryDirectory(prefix="control-vector-") as d:
        tmp = Path(d)
        rt, folder, stats, p50 = packed_stand_in(tmp)
        vec = check_census(rt, folder, stats, p50, tmp)
        set_dirs = check_atlas(rt, folder, stats, p50, tmp)
        check_file(rt, folder, stats, p50, tmp, vec)
        gguf_model = check_gguf_source(rt, stats, tmp, vec)
        try:
            check_llama(exe, rt, gguf_model, tmp, vec, set_dirs)
        except (RuntimeError, OSError) as exc:
            check(False, f"llama.cpp itself: {exc}")
            log = tmp / "llama-server.log"
            if log.exists():
                print(log.read_text(errors="replace")[-2000:])
    print(f"\n{passed} checks passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

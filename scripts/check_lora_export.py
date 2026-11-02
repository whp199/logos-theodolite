#!/usr/bin/env python3
"""A CPU-only check of scripts/export_lora.py on the tiny random Gemma 4 of scripts/atlas_stand_in.py, in a temporary
folder, with scripts/check_control_vector.py's stand-in (its down projections packed as the W4A16 checkpoint packs
them and written as a checkpoint, here with its config.json) and its llama-server harness:
1. the HTTP API's PushIn: bodies read by the exporter (`pushes_of`) as `atlas.Atlas.parse_pushes` reads them, defaults
   and all;
2. the arithmetic: each adapter's W + B A, set against the instrument's own change on random inputs: a neuron's mul
   (x_n times f at the down projection's input), a head's scale (o_proj's inputs of head h times f, on a sliding layer
   and on the full one), a vault edit (the vault's own UPDATE, `LarqlVaultEngine.execute_lql`, its column laid on by
   `gguf_weights.install_overlay_forward`), and a neuron both edited and multiplied; the same B read from the checkpoint
   and from the same weights written as a float32 GGUF, bit for bit;
3. the file as llama.cpp's loader reads it: general.type adapter, general.architecture gemma4, adapter.type lora, no
   alpha; <tensor>.lora_a [n_in, r] and .lora_b [r, n_out], F32; theodolite's record; the command line writes the same
   tensors, reads `--push` as the short forms, and refuses add, act, what does not exist, a dose that is not a number,
   a whole block or every head at any factor but 0, a file already there, several pushes without -o, a silenced block
   without --model-out or --gguf, --vault beside --checkpoint, and a vault statement that is not an edit;
4. the silenced copies (`--model-out`): by an added `.scale` tensor (every other tensor and key as it was) and by a
   zeroed norm (every other byte as it was, the record beside it);
5. llama.cpp itself (`--llama-server`, else $LLAMA_SERVER, else `llama-server` on the PATH; without one, skipped and
   said so): the stand-in's float32 GGUF served on the CPU (CUDA_VISIBLE_DEVICES empty, -ngl 0, the cache in float32, no
   flash attention), the adapters loaded at start and scaled per request, and the log-probabilities of every word-piece
   at the first one written to a census question set against the instrument's own (`atlas_ops.write_rows`): each push
   alike the instrument's with the start mark pushed too, at SCALE 1, and at other scales the factor 1 + s (f - 1) (a
   mul-0 file at -3 is mul 4); far from it a layer off, another head, another scale and unpushed (20 times the worst
   alike); the silenced copies alike the instrument's block or heads at 0; the instrument with the start mark spared at
   least 10 times further from llama.cpp than with it pushed (the stand-in's pushes whose start mark matters most were
   chosen for that); and what the server does with what it cannot use: an adapter it cannot
   read stops it, a control vector it cannot read is logged and the model served unpushed, a request naming an
   adapter id it does not hold is answered unpushed, and a request naming no adapter gets every adapter at the scale
   `GET /lora-adapters` shows (1, even with --lora-init-without-apply).

Run: .venv/bin/python -I scripts/check_lora_export.py [--llama-server PATH]   (exits 0 when every check holds)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from types import SimpleNamespace
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
import check_control_vector as ccv  # noqa: E402
import export_lora as el  # noqa: E402
from backend import census as cz  # noqa: E402
from backend import gguf_model as gm  # noqa: E402
from backend.causal_check import language_model  # noqa: E402
from backend.gguf_weights import install_overlay_forward  # noqa: E402
from backend.larql_vault import LarqlVaultEngine  # noqa: E402

PY = sys.executable
EXPORT_PY = ROOT_DIR / "scripts" / "export_lora.py"
LAYER = 3  # a sliding layer: 4 heads of 16
FULL = 5  # the full layer: 4 heads of 32
NEURON = 26  # of the stand-in's neurons at LAYER, among those whose push the start mark moves most
HEAD = 1
BLOCK = 1  # the feed-forward block silenced in a copy
HEADS = 2  # every head silenced in a copy
UPDATE = f"UPDATE LAYER {LAYER} SLOT 7 SET object = 'she' STRENGTH 3"

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


def pack_o_proj(lm: Any, folder: Path) -> None:
    """The stand-in's o_proj packed as the W4A16 checkpoint packs it, beside check_control_vector.py's down projections
    (a second safetensors file in the checkpoint's folder), its own weight replaced by the unpacking."""
    from safetensors.torch import save_file

    from backend.engine_runtime import dequantize_w4a16_full
    from check_dequant import pack_w4a16

    ct: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for l, layer in enumerate(lm.layers):
            w = layer.self_attn.o_proj.weight
            packed, scale = pack_w4a16(w.detach())
            w.copy_(dequantize_w4a16_full(packed, scale).float())
            P = f"model.language_model.layers.{l}.self_attn.o_proj."
            ct[P + "weight_packed"], ct[P + "weight_scale"] = packed, scale
            ct[P + "weight_shape"] = torch.tensor(list(w.shape), dtype=torch.int64)
    save_file({k: v.contiguous() for k, v in ct.items()}, str(folder / "model-o_proj.safetensors"))


def run_cli(args: Sequence[Any]) -> subprocess.CompletedProcess:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", THEODOLITE_DEVICE="cpu")
    return subprocess.run([PY, "-I", str(EXPORT_PY)] + [str(a) for a in args], env=env, cwd=ROOT_DIR,
                          capture_output=True, text=True, timeout=300)


def export(source: el.Columns, path: Path, bodies: Sequence[Dict[str, Any]] = (),
           vault: Optional[Dict[Tuple[int, int], np.ndarray]] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Write an adapter as the command line writes it, returning its tensors."""
    ps, hs = el.pushes_of(bodies)
    plan = el.plan_of(ps, hs, vault or {}, source)
    tensors = el.lora_tensors(plan, source)
    el.write_lora(path, tensors, {"name": "check", **el.base_record(plan, source, {"pushes": list(bodies)}, ["check"]),
                                  "tensors": sorted(tensors), "scale": el.SCALE})
    return tensors


# ------------------------------------------------------------------------------------------- PushIn and arithmetic


def check_pushin(lm: Any) -> None:
    from backend.atlas import Atlas, PushIn

    before = len(failures)
    bodies = [{"layer": 3, "n": 5, "mode": "mul"}, {"layer": 3, "n": 5, "mode": "mul", "dose": 0},
              {"layer": 2, "head": 1}, {"layer": 2, "head": "2", "dose": 0.5}, {"layer": 4, "head": "all"},
              {"layer": 1, "all": True}, {"layer": 1, "n": 9, "mode": "act", "dose": 1.5}, {"layer": 1, "n": 9}]
    mine = el.pushes_of(bodies)
    theirs = Atlas.parse_pushes(SimpleNamespace(lm=lm), [PushIn(**b) for b in bodies])
    check(mine == theirs, f"PushIn bodies read as the HTTP API reads them ({mine} / {theirs})")
    check(mine[0][0].dose == 2.0 and mine[1][0].scale == 0.0 and mine[1][2].head is None and mine[0][2].neuron is None,
          "the API's defaults: mul 2, a head 0, every head and a whole block silenced")
    section("the HTTP API's PushIn, read as the API reads it", before)


def check_arithmetic(rt: Any, ct: el.Columns, gg: el.Columns, tmp: Path) -> Dict[Tuple[int, int], np.ndarray]:
    """W + B A against the instrument's change, on random inputs; returns the vault UPDATE's column."""
    before = len(failures)
    lm = language_model(rt.hf_model)
    g = torch.Generator().manual_seed(11)
    worst = 0.0

    def gap(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a - b).abs().max() / b.abs().max())

    def lora_out(W: torch.Tensor, ab: Tuple[np.ndarray, np.ndarray], x: torch.Tensor, s: float = 1.0) -> torch.Tensor:
        A, B = (torch.from_numpy(t.copy()) for t in ab)
        return x @ W.T + s * ((x @ A.T) @ B.T)

    W_down = lm.layers[LAYER].mlp.down_proj.weight.detach().float()
    x = torch.randn(7, W_down.shape[1], generator=g)
    for f in (0.0, 2.0, -1.5):
        t = export(ct, tmp / f"a-mul{f:g}.gguf", [{"layer": LAYER, "n": NEURON, "mode": "mul", "dose": f}])
        check(list(t) == [el.FFN_DOWN.format(LAYER)] and t[el.FFN_DOWN.format(LAYER)][0].shape[0] == 1,
              f"mul {f:g}: blk.{LAYER}.ffn_down alone, rank 1")
        xs = x.clone()
        xs[:, NEURON] *= f
        worst = max(worst, gap(lora_out(W_down, t[el.FFN_DOWN.format(LAYER)], x), xs @ W_down.T))
    zero = export(ct, tmp / "a-mul0-again.gguf",
                  [{"layer": LAYER, "n": NEURON, "mode": "mul", "dose": 0}])[el.FFN_DOWN.format(LAYER)]
    for s, f in ((-1.0, 2.0), (0.5, 0.5), (0.0, 1.0)):
        xs = x.clone()
        xs[:, NEURON] *= f
        worst = max(worst, gap(lora_out(W_down, zero, x, s), xs @ W_down.T))
    for l, h, f in ((LAYER, HEAD, 0.0), (FULL, 2, 0.0), (FULL, 0, 3.0), (LAYER, 2, 0.5)):
        W_o = lm.layers[l].self_attn.o_proj.weight.detach().float()
        hd = ao.attn_shape(lm, l).head_dim
        t = export(ct, tmp / f"a-head{l}-{h}.gguf", [{"layer": l, "head": h, "dose": f}])
        check(list(t) == [el.ATTN_OUT.format(l)] and t[el.ATTN_OUT.format(l)][0].shape[0] == hd,
              f"layer {l} head {h}: blk.{l}.attn_output alone, rank {hd}")
        xo = torch.randn(7, W_o.shape[1], generator=g)
        xs = xo.clone()
        xs[:, h * hd:(h + 1) * hd] *= f
        worst = max(worst, gap(lora_out(W_o, t[el.ATTN_OUT.format(l)], xo), xs @ W_o.T))

    vault = LarqlVaultEngine(rt)
    cols, said = el.vault_columns(vault, [UPDATE])
    check(list(cols) == [(LAYER, 7)] and "UPDATE" in said[0] and not vault.edits_in_force(),
          f"the vault's UPDATE read and rolled back ({said})")
    with el.vault_kept(vault):
        vault.execute_lql(UPDATE)
        edited = lm.layers[LAYER].mlp.down_proj(x)
    check(not vault.edits_in_force() and not lm.layers[LAYER].mlp.down_proj._memit_col_overlays, "the vault left as it was")
    t = export(ct, tmp / "a-vault.gguf", vault=cols)
    worst = max(worst, gap(lora_out(W_down, t[el.FFN_DOWN.format(LAYER)], x), edited))
    t = export(ct, tmp / "a-vault-mul.gguf", [{"layer": LAYER, "n": 7, "mode": "mul", "dose": 2}], vault=cols)
    check(t[el.FFN_DOWN.format(LAYER)][0].shape[0] == 1, "a neuron both edited and multiplied: one rank")
    xs = x.clone()
    xs[:, 7] *= 2
    with el.vault_kept(vault):
        vault.execute_lql(UPDATE)
        both = lm.layers[LAYER].mlp.down_proj(xs)
    worst = max(worst, gap(lora_out(W_down, t[el.FFN_DOWN.format(LAYER)], x), both))
    check(worst < 1e-5, f"W + B A is the instrument's change within float32 rounding (worst {worst:.1e} of the largest)")

    mix = [{"layer": LAYER, "n": NEURON, "mode": "mul", "dose": 0}, {"layer": FULL, "head": 2, "dose": 0.25}]
    a, b = export(ct, tmp / "a-ct.gguf", mix), export(gg, tmp / "a-gg.gguf", mix)
    check(set(a) == set(b) and all(np.array_equal(a[k][i], b[k][i]) for k in a for i in (0, 1)),
          "the checkpoint's columns and the float32 GGUF's give the same adapter, bit for bit")
    try:
        export(ct, tmp / "a-one.gguf", [{"layer": LAYER, "n": NEURON, "mode": "mul", "dose": 1}])
        check(False, "a factor of 1 is refused: nothing to write")
    except ValueError as exc:
        check("Nothing to write" in str(exc) and not (tmp / "a-one.gguf").exists(), f"a factor of 1 is refused ({exc})")
    section(f"the arithmetic: mul, scales, heads (sliding and full), the vault's edit, both sources (worst {worst:.1e})", before)
    return cols


# -------------------------------------------------------------------------------------------- the file and the CLI


def check_file(rt: Any, folder: Path, gguf_model: Path, tmp: Path) -> None:
    import gguf

    before = len(failures)
    path = tmp / "a-mul0.gguf"
    r = gguf.GGUFReader(str(path))
    meta = gm.read_metadata(r, long_arrays=True)
    check(meta.get("general.type") == "adapter" and meta.get("general.architecture") == "gemma4"
          and meta.get("adapter.type") == "lora" and "adapter.lora.alpha" not in meta,
          "llama.cpp's keys: general.type adapter, general.architecture gemma4, adapter.type lora, no alpha")
    t = {x.name: x for x in r.tensors}
    name = el.FFN_DOWN.format(LAYER)
    check(sorted(t) == [f"{name}.lora_a", f"{name}.lora_b"], f"one pair, {name}.lora_a and .lora_b ({sorted(t)})")
    base = {x.name: x for x in gguf.GGUFReader(str(gguf_model)).tensors}[name]
    A, B = t[f"{name}.lora_a"], t[f"{name}.lora_b"]
    check(all(x.tensor_type == gguf.GGMLQuantizationType.F32 for x in (A, B)), "A and B in F32")
    check(int(A.shape[0]) == int(base.shape[0]) and int(B.shape[1]) == int(base.shape[1])
          and int(A.shape[1]) == int(B.shape[0]) == 1,
          f"the loader's shapes: lora_a [n_in, r] {list(map(int, A.shape))}, lora_b [r, n_out] {list(map(int, B.shape))}, "
          f"the model's {list(map(int, base.shape))}")
    rec = {k[len(el.KEY):]: v for k, v in meta.items() if k.startswith(el.KEY)}
    check(rec.get("model") == el.MODEL_ID and "model.safetensors" in str(rec.get("weights"))
          and all(rec.get(k) for k in ("entries", "given", "push", "start_mark", "llama_cpp", "caveat", "made_by", "scale")),
          "theodolite's record: the model, the weights, the entries, the push, the start mark, llama.cpp, the scale")

    cli = tmp / "cli.gguf"
    got = run_cli(["--checkpoint", folder, "--page", LAYER + 1, "--neuron", NEURON, "--mul", 0, "-o", cli])
    check(got.returncode == 0, f"the command line writes an adapter ({got.stderr.strip()[-300:]})")
    if got.returncode == 0:
        mine = el.read_lora(cli)[0][name]
        ref = el.read_lora(path)[0][name]
        check(np.array_equal(mine[0], ref[0]) and np.array_equal(mine[1], ref[1]), "its tensors are the API's, bit for bit")
        check("GET /lora-adapters" in got.stdout and "failed to apply lora adapter" in got.stdout,
              "it says what to look for in llama-server's log")
    cli2 = tmp / "cli-push.gguf"
    got = run_cli(["--checkpoint", folder, "--push", json.dumps({"layer": LAYER, "n": NEURON, "mode": "mul", "dose": 0}),
                   "-o", cli2])
    check(got.returncode == 0 and cli2.exists()
          and np.array_equal(el.read_lora(cli2)[0][name][1], el.read_lora(path)[0][name][1]),
          "--push writes the same adapter as --neuron --mul")
    base_args = ["--checkpoint", folder]
    refusals = [
        (["--push", '{"layer": 3, "n": 5}', "-o", tmp / "r0.gguf"], "control vector"),
        (["--push", '{"layer": 3, "n": 5, "mode": "act", "dose": 1}', "-o", tmp / "r1.gguf"], "no hook"),
        (["--layer", 6, "--neuron", 1, "--mul", 0, "-o", tmp / "r2.gguf"], "does not exist"),
        (["--layer", LAYER, "--neuron", 128, "--mul", 0, "-o", tmp / "r3.gguf"], "does not exist"),
        (["--layer", LAYER, "--head", 4, "--scale", 0, "-o", tmp / "r4.gguf"], "attention heads"),
        (["--layer", LAYER, "--neuron", 1, "--mul", "nan", "-o", tmp / "r5.gguf"], "finite"),
        (["--push", '{"layer": 3, "all": true, "dose": 1}', "-o", tmp / "r6.gguf"], "only be silenced"),
        (["--layer", LAYER, "--head", "all", "--scale", 0.5, "-o", tmp / "r7.gguf"], "only be silenced"),
        (["--layer", LAYER, "--neuron", NEURON, "--mul", 0, "-o", cli], "already there"),
        (["--push", '[{"layer": 3, "n": 5, "mode": "mul"}, {"layer": 2, "head": 1}]'], "-o FILE"),
        (["--layer", LAYER, "--block"], "--model-out"),
        (["--layer", LAYER, "--block", "--model-out", tmp / "r8.gguf"], "--gguf"),
        (["--vault", "DELETE FROM LAYER 3 WHERE slot = 5"], "--checkpoint"),
    ]
    for args, says in refusals:
        got = run_cli(base_args + args)
        check(got.returncode != 0 and says in got.stderr, f"refused: {' '.join(str(a) for a in args[:4])} ({says}): "
              f"{got.stderr.strip()[-200:]}")
    check(not any((tmp / f"r{i}.gguf").exists() or (tmp / f"r{i}.gguf.tmp").exists() for i in range(9)),
          "nothing written where refused")
    try:
        el.vault_columns(LarqlVaultEngine(rt), ["DESCRIBE"])
        check(False, "a vault statement that is not an edit is refused")
    except ValueError as exc:
        check("an edit" in str(exc), f"a vault statement that is not an edit is refused ({exc})")
    got = run_cli(["--show", path])
    check(got.returncode == 0 and "rank 1" in got.stdout and "theodolite.entries" in got.stdout, "--show reads it back")

    # --vault through the command line's own code, its runtime the stand-in (make_runtime would load the 31B)
    import contextlib
    import io

    import backend.runtime_factory as rf

    made, env = rf.make_runtime, os.environ.get(rf.GGUF_ENV)
    rf.make_runtime = lambda **_kw: SimpleNamespace(load=lambda: rt)
    said = io.StringIO()
    try:
        with contextlib.redirect_stdout(said):
            code = el.main(["--gguf", str(gguf_model), "--vault", UPDATE, "-o", str(tmp / "cli-vault.gguf")])
    finally:
        rf.make_runtime = made
        if env is None:
            os.environ.pop(rf.GGUF_ENV, None)
        else:
            os.environ[rf.GGUF_ENV] = env
    if code == 0 and (tmp / "cli-vault.gguf").exists():
        mine, ref = el.read_lora(tmp / "cli-vault.gguf")[0][name], el.read_lora(tmp / "a-vault.gguf")[0][name]
        check(np.array_equal(mine[0], ref[0]) and np.array_equal(mine[1], ref[1]) and "UPDATE" in said.getvalue(),
              "--vault writes the vault's edit, as the API does, bit for bit")
    else:
        check(False, f"--vault writes an adapter ({said.getvalue()[-300:]})")
    check(not LarqlVaultEngine(rt).edits_in_force() and not any(
        getattr(layer.mlp.down_proj, "_memit_col_overlays", None) for layer in language_model(rt.hf_model).layers),
        "and leaves the runtime's vault as it was")
    # a statement that is no edit refused before any model is loaded; without --gguf the vault's runtime is the
    # checkpoint's (B's columns are), whatever THEODOLITE_GGUF the shell holds, and on the CPU
    seen: Dict[str, Any] = {}

    def no_load(**_kw: Any) -> Any:
        raise AssertionError("a model was loaded")

    def watched(**_kw: Any) -> Any:
        seen.update(gguf=os.environ.get(rf.GGUF_ENV), device=os.environ.get(rf.DEVICE_ENV))
        return SimpleNamespace(load=lambda: rt)

    files_of, dev = el.ecv.checkpoint_files, os.environ.get(rf.DEVICE_ENV)
    try:
        rf.make_runtime = no_load
        try:
            el.main(["--gguf", str(gguf_model), "--vault", "ROLLBACK", "-o", str(tmp / "cli-rollback.gguf")])
            check(False, "--vault ROLLBACK refused")
        except SystemExit as exc:
            check("--vault takes an edit" in str(exc) and not (tmp / "cli-rollback.gguf").exists(),
                  f"--vault with a statement that is no edit: refused before any model is loaded ({exc})")
        rf.make_runtime = watched
        os.environ[rf.GGUF_ENV], os.environ[rf.DEVICE_ENV] = str(gguf_model), "cuda:0"
        el.ecv.checkpoint_files = lambda _c=None: files_of(str(folder))
        with contextlib.redirect_stdout(io.StringIO()):
            code = el.main(["--vault", UPDATE, "-o", str(tmp / "cli-vault-ct.gguf")])
        check(code == 0 and seen == {"gguf": None, "device": "cpu"},
              f"--vault without --gguf, THEODOLITE_GGUF set in the shell: the vault's runtime is the checkpoint's, on the "
              f"CPU, as B's columns are ({seen})")
    finally:
        rf.make_runtime, el.ecv.checkpoint_files = made, files_of
        for k_, v_ in ((rf.GGUF_ENV, env), (rf.DEVICE_ENV, dev)):
            if v_ is None:
                os.environ.pop(k_, None)
            else:
                os.environ[k_] = v_
    section("the file as llama.cpp's loader reads it, the command line and its refusals", before)


def check_silenced(gguf_model: Path, tmp: Path) -> Dict[str, Path]:
    import gguf

    before = len(failures)
    out: Dict[str, Path] = {}
    src = gguf.GGUFReader(str(gguf_model))
    for by in ("scale", "norm"):
        for what, args in (("ffn", ["--block"]), ("attn", ["--head", "all", "--scale", 0])):
            l = BLOCK if what == "ffn" else HEADS
            p = tmp / f"silenced-{what}-{by}.gguf"
            got = run_cli(["--gguf", gguf_model, "--layer", l, *args, "--model-out", p, "--silence-by", by])
            check(got.returncode == 0 and p.exists(), f"--model-out writes the {what} copy by {by} ({got.stderr.strip()[-300:]})")
            if not p.exists():
                continue
            out[f"{what}-{by}"] = p
            r = gguf.GGUFReader(str(p))
            t = {x.name: x for x in r.tensors}
            scale, norm = (s.format(l) for s in el.SILENCE_TENSORS[what])
            if by == "scale":
                check(scale in t and t[scale].tensor_type == gguf.GGMLQuantizationType.F32
                      and list(map(int, t[scale].shape)) == [1]
                      and float(np.asarray(t[scale].data)[0]) == 0.0, f"{scale}: F32, one number, 0")
                same = all(np.array_equal(np.asarray(t[x.name].data), np.asarray(x.data)) for x in src.tensors)
                check(same and len(t) == len(src.tensors) + 1, "every other tensor as it was")
                m0, m1 = gm.read_metadata(src, long_arrays=True), gm.read_metadata(r, long_arrays=True)
                check(all(m1.get(k) == v for k, v in m0.items()) and m1.get(el.KEY + "silence_by"), "every key as it was, "
                      "and theodolite's record")
            else:
                check(scale not in t and not np.asarray(t[norm].data).any(), f"{norm} zeroed, no tensor added")
                a, b = gguf_model.read_bytes(), p.read_bytes()
                lo, n = int(t[norm].data_offset), int(t[norm].n_bytes)
                check(len(a) == len(b) and a[:lo] == b[:lo] and a[lo + n:] == b[lo + n:], "every other byte as it was")
                side = Path(str(p) + ".theodolite.json")
                check(side.exists() and json.loads(side.read_text()).get("silence_by"), "the record beside it")
    for key in ("ffn-scale", "ffn-norm"):  # a copy of a silenced copy would record only the second silencing
        if key in out:
            p = tmp / f"chain-{key}.gguf"
            got = run_cli(["--gguf", out[key], "--layer", HEADS, "--head", "all", "--scale", 0, "--model-out", p])
            check(got.returncode != 0 and "is itself a silenced copy" in got.stderr and not p.exists(),
                  f"a copy made from the {key} copy: refused, nothing written ({got.stderr.strip()[-160:]})")
    section("the silenced copies: an added .scale tensor, a zeroed norm; none made from another", before)
    return out


# ------------------------------------------------------------------------------------------------- llama.cpp itself


def instrument_logprobs(rt: Any, ids: Sequence[int], spare_bos: bool, muls: Optional[Dict[int, list]] = None,
                        heads: Optional[Dict[int, list]] = None) -> np.ndarray:
    """[vocab] the instrument's log-probabilities at the first word-piece written (`atlas_ops.write_rows`), with
    `muls` {layer: [(neuron or None, factor)]} and `heads` {layer: [(head or None, factor)]} as `resolve` makes them
    and the vault's edits in force on the model."""
    lm = language_model(rt.hf_model)
    V = int(rt.lm_head_weight.shape[0])
    one = lambda d: {l: [(n, torch.tensor([float(f)])) for n, f in v] for l, v in (d or {}).items()}  # noqa: E731
    iv = ao.Interventions(rows=1, muls=one(muls), heads=one(heads))
    step = next(iter(ao.write_rows(lm, rt.lm_head_weight, [list(ids)], None if iv.empty() else iv, max_new=1, stop_ids=[],
                                   bos_id=rt.tokenizer.bos_token_id, spare_bos=spare_bos, softcap=ao.softcap_of(rt.hf_model),
                                   top_k=V)))
    out = np.full(V, np.nan)
    out[np.asarray(step.top_ids)] = np.asarray(step.top_lps, dtype=np.float64)
    return out


def adapters_held(port: int) -> List[Dict[str, Any]]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/lora-adapters", timeout=10) as r:
        return json.loads(r.read())


def check_llama(exe: Optional[Path], rt: Any, gguf_model: Path, tmp: Path, silenced: Dict[str, Path]) -> None:
    before = len(failures)
    if exe is None:
        print("  --    llama.cpp itself: not run (no llama-server: --llama-server PATH, or $LLAMA_SERVER)")
        return
    lm = language_model(rt.hf_model)
    ids = rt.tokenize_prompt(rt.compose_prompt(cz.design_questions("break_rel")["who"], chat=True))
    V = int(rt.lm_head_weight.shape[0])
    tok = rt.tokenizer
    barred = [i for i in range(V) if i in set(tok.all_special_ids) or not tok.decode([i]) or "�" in tok.decode([i])]
    log = tmp / "llama-server.log"
    gap = lambda a, b: float(np.nanmax(np.abs(a - b)))  # noqa: E731
    # the adapters, in the order their ids are given
    files = [tmp / "a-mul0.gguf", tmp / "a-head3-1.gguf", tmp / "a-head5-2.gguf", tmp / "a-vault.gguf", tmp / "mix.gguf"]
    names = ["mul0", "head", "head-full", "vault", "mix"]
    vault = LarqlVaultEngine(rt)
    mix_pushes = [{"layer": LAYER - 1, "n": 40, "mode": "mul", "dose": 3}, {"layer": FULL - 1, "head": 3, "dose": 0}]
    ct = el.Columns(ccv.ecv.CheckpointColumns(ccv.ecv.checkpoint_files(str(tmp / "checkpoint"))))
    cols, _ = el.vault_columns(vault, [UPDATE])
    export(ct, files[-1], mix_pushes, vault=cols)
    lora = lambda **s: {"lora": [{"id": i, "scale": float(s.get(n, 0.0))} for i, n in enumerate(names)]}  # noqa: E731
    got: Dict[str, np.ndarray] = {}
    with ccv.LlamaServer(exe, gguf_model, [], log, ["--lora", ",".join(map(str, files)), "--lora-init-without-apply"]) as srv:
        held = adapters_held(srv.port)
        check([Path(h["path"]).name for h in held] == [f.name for f in files], f"GET /lora-adapters lists the {len(files)} "
              "adapters, in order")
        for key, body in (("plain", lora()), ("mul0", lora(mul0=1)), ("mul0 at -3", lora(mul0=-3)),
                          ("mul0 at 0.5", lora(mul0=0.5)), ("head", lora(head=1)), ("head at 0.5", lora(head=0.5)),
                          ("head-full", lora(**{"head-full": 1})),
                          ("vault", lora(vault=1)), ("mix", lora(mix=1)), ("unheld id", {"lora": [{"id": 99, "scale": 1}]}),
                          ("no lora field", None), ("all at 1", lora(**{n: 1 for n in names}))):
            got[key] = srv.first_logprobs(ids, V, barred, body)
    one = lambda l, n, f: {l: [(n, f)]}  # noqa: E731
    pushed = lambda **kw: instrument_logprobs(rt, ids, False, **kw)  # noqa: E731  the start mark pushed, as llama.cpp
    spared = lambda **kw: instrument_logprobs(rt, ids, True, **kw)  # noqa: E731  the instrument's way
    ref = {
        "plain": spared(),
        "mul0": pushed(muls=one(LAYER, NEURON, 0.0)),
        "mul0 at -3": pushed(muls=one(LAYER, NEURON, 4.0)),
        "mul0 at 0.5": pushed(muls=one(LAYER, NEURON, 0.5)),
        "head": pushed(heads=one(LAYER, HEAD, 0.0)),
        "head at 0.5": pushed(heads=one(LAYER, HEAD, 0.5)),
        "head-full": pushed(heads=one(FULL, 2, 0.0)),
    }
    with el.vault_kept(vault):
        vault.execute_lql(UPDATE)
        ref["vault"] = spared()
        ref["mix"] = pushed(muls=one(LAYER - 1, 40, 3.0), heads=one(FULL - 1, 3, 0.0))
    d = {k: gap(got[k], ref[k]) for k in ref}
    apart = {
        "mul0 at -3, a layer below": gap(got["mul0 at -3"], pushed(muls=one(LAYER - 1, NEURON, 4.0))),
        "mul0 at -3, a layer above": gap(got["mul0 at -3"], pushed(muls=one(LAYER + 1, NEURON, 4.0))),
        "mul0 at -3 against unpushed": gap(got["mul0 at -3"], got["plain"]),
        "mul0 at -3 against mul0 at 1": gap(got["mul0 at -3"], got["mul0"]),
        "head, another head": gap(got["head"], pushed(heads=one(LAYER, HEAD + 1, 0.0))),
        "head against unpushed": gap(got["head"], got["plain"]),
        "head at 0.5 against head at 1": gap(got["head at 0.5"], got["head"]),
        "head-full against unpushed": gap(got["head-full"], got["plain"]),
        "vault against unpushed": gap(got["vault"], got["plain"]),
    }
    marks = {  # the same push with the start mark spared, against llama.cpp: (that gap, the gap with it pushed)
        "mul0 at -3": (gap(got["mul0 at -3"], spared(muls=one(LAYER, NEURON, 4.0))), d["mul0 at -3"]),
        "head": (gap(got["head"], spared(heads=one(LAYER, HEAD, 0.0))), d["head"]),
    }
    check(all(not np.isnan(v).any() for v in got.values()), f"llama-server gave all {V} log-probabilities each time")

    # the silenced copies, each served alone
    for key, p in sorted(silenced.items()):
        with ccv.LlamaServer(exe, p, [], log) as srv:
            got[key] = srv.first_logprobs(ids, V, barred)
        kw = {"muls": {BLOCK: [(None, 0.0)]}} if key.startswith("ffn") else {"heads": {HEADS: [(None, 0.0)]}}
        d[key] = gap(got[key], pushed(**kw))
        apart[f"{key} against unpushed"] = gap(got[key], got["plain"])
        marks[key] = (gap(got[key], spared(**kw)), d[key])
    # an adapter on a head of the silenced attention's layer, served beside each copy: the zeroed norm silences its part
    # too; the added attn_output.scale, which llama.cpp applies before an adapter's part, does not
    beside = tmp / "a-silenced-layer-head.gguf"
    export(el.Columns(ccv.ecv.GGUFColumns(gguf_model)), beside, [{"layer": HEADS, "head": 0, "dose": 3}])
    leak: Dict[str, float] = {}
    for key in ("attn-norm", "attn-scale"):
        if key in silenced:
            with ccv.LlamaServer(exe, silenced[key], [], log, ["--lora", str(beside), "--lora-init-without-apply"]) as srv:
                leak[key] = gap(srv.first_logprobs(ids, V, barred, {"lora": [{"id": 0, "scale": 1.0}]}),
                                srv.first_logprobs(ids, V, barred, {"lora": [{"id": 0, "scale": 0.0}]}))
    floor = max(d.values())
    check(floor < 1e-3, f"llama.cpp and the instrument alike where they should be (worst {floor:.1e} in a log-probability)")
    check(leak.get("attn-norm", 1.0) < 1e-6 and leak.get("attn-scale", 0.0) > 20 * floor,
          f"an adapter on a head of layer {HEADS} beside its silenced attention: nothing through the by-norm copy "
          f"({leak.get('attn-norm', float('nan')):.1e}), something through the by-scale one "
          f"({leak.get('attn-scale', float('nan')):.1e}), as the docs warn")
    least = min(apart.values())
    check(least > 20 * floor, f"and far apart where they should be (least {least:.2e}, at least 20x the {floor:.1e} floor)")
    for k, (far, near) in marks.items():
        check(far > 10 * max(near, 1e-4), f"{k}: llama.cpp acts on the start mark (the instrument with it spared {far:.1e} "
              f"away, with it pushed {near:.1e}: at least 10x)")
    check(gap(got["unheld id"], got["plain"]) < 1e-6, "a request naming an adapter id the server does not hold: unpushed, "
          f"without a word ({gap(got['unheld id'], got['plain']):.1e})")
    d_all = gap(got["no lora field"], got["all at 1"])
    check(d_all < 1e-6 and all(h["scale"] == 1.0 for h in held) and gap(got["all at 1"], got["plain"]) > 20 * floor,
          f"a request naming no adapter: every adapter at the scale /lora-adapters shows, 1 ({d_all:.1e})")
    for k, v in d.items():
        print(f"          alike  {k}: largest difference in a log-probability {v:.2e}")
    for k, v in apart.items():
        print(f"          apart  {k}: {v:.2e}")
    for k, (far, _near) in marks.items():
        print(f"          spared {k}: {far:.2e}")

    bad = tmp / "bad-lora.gguf"
    import gguf

    w = gguf.GGUFWriter(str(bad), "gemma4")
    w.add_type("adapter")
    w.add_string("adapter.type", "lora")
    w.add_tensor(f"blk.{len(lm.layers) + 3}.ffn_down.weight.lora_a", np.zeros((1, 128), np.float32))
    w.add_tensor(f"blk.{len(lm.layers) + 3}.ffn_down.weight.lora_b", np.zeros((64, 1), np.float32))
    w.write_header_to_file(), w.write_kv_data_to_file(), w.write_tensors_to_file(), w.close()
    bad_log = tmp / "bad-lora.log"
    try:
        with ccv.LlamaServer(exe, gguf_model, [], bad_log, ["--lora", str(bad)]):
            stopped = False
    except RuntimeError:
        stopped = True
    text = bad_log.read_text(errors="replace")
    check(stopped and "failed to apply lora adapter" in text and "does not exist in base model" in text,
          "an adapter llama-server cannot read stops it at start, and its log says why")
    cv = tmp / "bad-cv.gguf"
    w = gguf.GGUFWriter(str(cv), "controlvector")
    w.add_tensor("direction.3", np.zeros(int(lm.norm.weight.shape[0]) + 1, np.float32))
    w.write_header_to_file(), w.write_kv_data_to_file(), w.write_tensors_to_file(), w.close()
    cv_log = tmp / "bad-cv.log"
    with ccv.LlamaServer(exe, gguf_model, [(cv, 1.0)], cv_log) as srv:
        served = srv.first_logprobs(ids, V, barred)
    text = cv_log.read_text(errors="replace")
    check("control vector n_embd does not match model" in text and gap(served, got["plain"]) < 1e-6,
          "a control vector llama-server cannot read is logged, and the model served unpushed")
    section("llama.cpp itself: each adapter at its scale, per request; the silenced copies; what the server does with "
            "what it cannot use", before)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--llama-server", help="a llama-server binary for part 5 (else $LLAMA_SERVER, else the PATH's)")
    a = ap.parse_args(argv)
    torch.set_grad_enabled(False)
    exe = ccv.find_llama_server(a.llama_server)
    with tempfile.TemporaryDirectory(prefix="lora-export-") as d:
        tmp = Path(d)
        rt, folder, _stats, _p50 = ccv.packed_stand_in(tmp)
        rt.hf_model.config.save_pretrained(str(folder))  # the head count, as the checkpoint's snapshot holds it
        lm = language_model(rt.hf_model)
        pack_o_proj(lm, folder)
        for layer in lm.layers:
            install_overlay_forward(layer.mlp.down_proj)  # the vault's column overlays, as the instrument's forward lays them
        gguf_model = gm.write_gemma4_gguf(rt.hf_model, tmp / "stand-in-f32.gguf", tokenizer=rt.tokenizer, qtype="F32")
        ct = el.Columns(ccv.ecv.CheckpointColumns(ccv.ecv.checkpoint_files(str(folder))))
        gg = el.Columns(ccv.ecv.GGUFColumns(gguf_model))
        check_pushin(lm)
        check_arithmetic(rt, ct, gg, tmp)
        check_file(rt, folder, gguf_model, tmp)
        silenced = check_silenced(gguf_model, tmp)
        try:
            check_llama(exe, rt, gguf_model, tmp, silenced)
        except (RuntimeError, OSError) as exc:
            check(False, f"llama.cpp itself: {exc}")
            log = tmp / "llama-server.log"
            if log.exists():
                print(log.read_text(errors="replace")[-2000:])
    print(f"\n{passed} checks passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

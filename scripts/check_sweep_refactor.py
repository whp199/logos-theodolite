#!/usr/bin/env python3
"""Regression check for the split of `TheodoliteRuntime.run_theodolite_sweep` into `capture_for_sweep` +
`prepare_sweep` + `analyse_position(0)`: an old caller (verify_optics.py, the E6 audit scripts, the experiments) must
get bit-identical results. It needs the GPU and the model: run it with the server on :8765 stopped.

Record mode, once per checkout, each in its own process:

    .venv/bin/python -I scripts/check_sweep_refactor.py --repo <checkout root> --out <file.pt>

imports `backend` from `--repo` (put first on sys.path, and checked), loads the runtime once, and runs six fixed
cases. Cases a-d call `run_theodolite_sweep` with only the arguments the old code knew:
  (a) the default bare two-hop prompt, 6 word-pieces, landmarks spider / eight / insect (the space-then-digit rule
      fires: the answer starts with a lone space);
  (b) chat + system "You are Gemma." + begun answer "Hi! I" + an fnw she/he nudge (strength 0.01, quiet dims masked,
      the start mark spared; built by server._build_steering_spec_from_req after ensure_quiet_masks), a Löwdin frame
      on she / he / it, centred on a control chat "Tell me about yourself." with no begun answer;
  (c) the pole-motif frame locked on case a's layer-35 strongest pole, nothing written;
  (d) the default prompt with threads scored by the direct path and poles ranked by write size.
Two more read every word-piece of an answer through `iter_theodolite_sweeps` (the rows past token 0, which the
four above never reach; the sweep's capture keeps most addresses at the read rows only):
  (e) chat, with a system prompt of about 130 word-pieces, 8 word-pieces written, each sighted, centred on the
      control chat reading this very answer (`centre_answer`), landmarks she / he / it;
  (f) case a's answer given back as word-piece ids (teacher-forced, nothing written), each sighted in the pole-motif
      frame of case c (built at token 0 and kept).
Cases e and f need `iter_theodolite_sweeps` with `centre_answer` and `answer_token_ids` (f788a36 or later): for an
older checkout pass --cases a,b,c,d (the script refuses e and f there before it loads anything).
A checkout whose runtime turns on the CUDA allocator's expandable segments when imported (backend/engine_runtime.py,
above `import torch`, unless PYTORCH_ALLOC_CONF or PYTORCH_CUDA_ALLOC_CONF is set) records under them; an older one
records under the default allocator, so comparing the two compares the allocators too. Each record names what the
allocator ran with (meta "allocator").
Every tensor and list of each bundle is saved under a flat key (trajectory, deltas, CKA, poles, write sizes, pole
words, threads, J++ and naked-eye readouts with the tracked ranks / logits / probs, the words written, the survey
labels, frame_info); a tensor of more than 2^20 elements (the pole logits) is kept as the SHA-256 of its bytes.
Timings and peak memory are left out.

Compare mode:

    .venv/bin/python -I scripts/check_sweep_refactor.py --compare old.pt new.pt

prints every key that differs (torch.equal for tensors, bit for bit; == for the rest) or is missing from the second
file, and exits 1 if any does. Keys only in the second file are the refactor's additive fields: listed, not failures.
To tell a real difference from GPU nondeterminism, record the old checkout twice and compare those two first.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import inspect
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_PROMPT = "The number of legs on the animal that spins webs is"
CASES = ("a", "b", "c", "d", "e", "f")
# case e's system prompt: this paragraph twice (132 word-pieces), about 160 with the question and the chat marks
# (159 by the instrument's tokenizer)
LONG_SYSTEM_PARAGRAPH = (
    "You are a careful assistant who answers questions about the natural world. When a question names an animal, "
    "say what it eats, where it lives and how many legs it has, and keep each answer to a few plain sentences. If "
    "you are not sure of a fact, say so instead of guessing, and never invent a source."
)
# above this many elements a tensor is kept as a digest of its bytes (the [15, 262144] pole logits)
DIGEST_ABOVE = 1 << 20
# fields that measure the run, not the reading
SKIP_FIELDS = {"elapsed_ms", "peak_vram_gb", "analyse_ms"}


def flatten(obj: Any, prefix: str, out: Dict[str, Any]) -> None:
    """Every tensor, number and string under `obj`, at a flat key (dataclass fields, dict keys, list items)."""
    import torch

    if isinstance(obj, torch.Tensor):
        t = obj.detach().cpu().contiguous()
        if t.numel() > DIGEST_ABOVE:
            raw = t.reshape(-1).view(torch.uint8).numpy().tobytes()
            out[prefix] = ("sha256", str(t.dtype), tuple(t.shape), hashlib.sha256(raw).hexdigest())
        else:
            out[prefix] = t.clone()
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            if f.name not in SKIP_FIELDS:
                flatten(getattr(obj, f.name), f"{prefix}.{f.name}", out)
    elif isinstance(obj, dict):
        out[f"{prefix}.keys"] = [repr(k) for k in obj]
        for k, v in obj.items():
            flatten(v, f"{prefix}[{k!r}]", out)
    elif isinstance(obj, (list, tuple)):
        out[f"{prefix}.len"] = len(obj)
        for i, v in enumerate(obj):
            flatten(v, f"{prefix}[{i}]", out)
    else:
        out[prefix] = obj


def _same(x: Any, y: Any) -> bool:
    import torch

    if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
        if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
            return False
        if x.dtype != y.dtype or x.shape != y.shape:
            return False
        if torch.equal(x, y):
            return True
        # NaN != NaN under torch.equal: compare the bits
        a, b = x.contiguous().reshape(-1), y.contiguous().reshape(-1)
        return bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))
    if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
        return True
    return type(x) is type(y) and x == y


def same_entry(key: str, x: Any, y: Any) -> bool:
    """`_same`, except that a dict's key list may gain keys (the refactor's additive fields, e.g. frame_info's
    centre_kind): every old key must still be there, and its value is compared under its own entry."""
    if key.endswith(".keys") and isinstance(x, list) and isinstance(y, list):
        return set(x) <= set(y)
    return _same(x, y)


def _describe(x: Any, y: Any) -> str:
    import torch

    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and x.shape == y.shape and x.is_floating_point():
        d = (x.float() - y.float()).abs()
        return f"max |diff| {float(d.max()):.3e} at {int(d.argmax())}, {int((d > 0).sum())} of {d.numel()} differ"
    sx, sy = repr(x), repr(y)
    return f"{sx[:120]} != {sy[:120]}"


def compare(a_path: str, b_path: str) -> int:
    import torch

    a = torch.load(a_path, map_location="cpu", weights_only=True)
    b = torch.load(b_path, map_location="cpu", weights_only=True)
    print(f"a: {a['meta']}\nb: {b['meta']}")
    n_diff = n_missing = n_same = 0
    for case in a["cases"]:
        fa = a["cases"][case]
        fb = b["cases"].get(case)
        if fb is None:
            print(f"case {case}: missing from {b_path}")
            n_missing += 1
            continue
        diff = [k for k in fa if k in fb and not same_entry(k, fa[k], fb[k])]
        missing = [k for k in fa if k not in fb]
        added = [k for k in fb if k not in fa]
        n_diff += len(diff)
        n_missing += len(missing)
        n_same += len(fa) - len(diff) - len(missing)
        print(f"case {case}: {len(fa) - len(diff) - len(missing)} keys equal, {len(diff)} differ, {len(missing)} missing, "
              f"{len(added)} added")
        for k in diff[:200]:
            print(f"  DIFFERS {k}: {_describe(fa[k], fb[k])}")
        for k in missing[:200]:
            print(f"  MISSING {k}")
        if added:
            heads = sorted({k.split("[")[0].split(".keys")[0] for k in added})
            print(f"  added (not a failure): {len(added)} keys under {', '.join(heads[:30])}")
    if n_diff or n_missing:
        print(f"\nsweep refactor: {n_diff} keys differ and {n_missing} are missing ({n_same} equal)")
        return 1
    print(f"\nsweep refactor: all {n_same} keys equal, bit for bit")
    return 0


async def run_cases(repo: Path, cases: List[str]) -> Dict[str, Dict[str, Any]]:
    import backend
    import backend.engine_runtime as er
    import backend.server as server

    where = Path(backend.__file__).resolve().parent.parent
    if where != repo:
        raise SystemExit(f"imported backend from {where}, not from --repo {repo}")
    if {"e", "f"} & set(cases):
        # refused before the model loads: an older checkout would run a-d on the GPU, then fail at e and save nothing
        every = getattr(er.TheodoliteRuntime, "iter_theodolite_sweeps", None)
        params = inspect.signature(every).parameters if every is not None else {}
        if "centre_answer" not in params or "answer_token_ids" not in params:
            raise SystemExit(
                "cases e and f need iter_theodolite_sweeps with centre_answer and answer_token_ids (f788a36 or "
                "later), which this checkout lacks: pass --cases a,b,c,d"
            )
    runtime = er.TheodoliteRuntime(load_jlens=False).load()
    landmarks = ("spider", "eight", "insect")
    tracked = tuple(er.DEFAULT_TRACKED_CONCEPTS)
    out: Dict[str, Dict[str, Any]] = {}
    bundle_a = None

    async def record(name: str, **kw: Any) -> Any:
        t0 = time.perf_counter()
        bundle = await runtime.run_theodolite_sweep(**kw)
        flat: Dict[str, Any] = {}
        flatten(bundle, "bundle", flat)
        out[name] = flat
        print(f"case {name}: {len(flat)} keys, wrote {bundle.generated_token_ids} in {time.perf_counter() - t0:.1f}s")
        return bundle

    if "a" in cases or "c" in cases or "f" in cases:
        bundle_a = await record(
            "a", prompt=DEFAULT_PROMPT, monument_concepts=landmarks, tracked_concepts=tracked, max_new_tokens=6
        )
    if "b" in cases:
        await runtime.ensure_quiet_masks()
        nudge = server.SteeringLaserRequest(
            enabled=True, target_concept="she", contrast_concept="he", direction="fnw", mask_quiet=True,
            spare_bos=True, strength=0.01,
        )
        spec = server._build_steering_spec_from_req(runtime, nudge)
        mask = server._steering_position_mask(nudge)
        prompt = runtime.compose_prompt(
            "Tell me about yourself.", chat=True, system="You are Gemma.", assistant_prefix="Hi! I"
        )
        control = runtime.compose_prompt("Tell me about yourself.", chat=True)
        await record(
            "b",
            prompt=prompt,
            monument_concepts=("she", "he", "it"),
            tracked_concepts=tracked + ("she", "he", "it"),
            steering_spec=spec,
            steering_position_mask=mask,
            frame_squaring="lowdin",
            centre_prompts=[control],
            max_new_tokens=6,
        )
    async def record_every(name: str, prompt: str, **kw: Any) -> None:
        t0 = time.perf_counter()
        flat: Dict[str, Any] = {}
        n = 0
        async for b in runtime.iter_theodolite_sweeps(prompt, **kw):
            # flattened as it comes and dropped, as the server sends each one: a bundle holds its pole logits on
            # the GPU (about 0.5 GB), so keeping all of them would run out of memory on a long answer
            flatten(b, f"bundles[{n}]", flat)
            n_prompt, written = len(b.prompt_token_ids), b.generated_token_ids
            del b
            n += 1
        flat["bundles.len"] = n
        out[name] = flat
        print(f"case {name}: {len(flat)} keys over {n} sightings of {n_prompt} + {written} in "
              f"{time.perf_counter() - t0:.1f}s")

    neuron = int(bundle_a.bipolar_poles[35].neuron_indices[0].item()) if bundle_a is not None else None
    if "c" in cases:
        assert bundle_a is not None and neuron is not None
        await record(
            "c",
            prompt=DEFAULT_PROMPT,
            monument_concepts=landmarks,
            tracked_concepts=tracked,
            survey_mode="pole_motif",
            locked_pole=(35, neuron, 1.0),
            max_new_tokens=0,
        )
    if "d" in cases:
        await record(
            "d",
            prompt=DEFAULT_PROMPT,
            monument_concepts=landmarks,
            tracked_concepts=tracked,
            thread_score="direct",
            pole_ranking="write",
            max_new_tokens=6,
        )
    if "e" in cases:
        question = "Tell me about the animal that spins webs."
        await record_every(
            "e",
            runtime.compose_prompt(question, chat=True, system=" ".join([LONG_SYSTEM_PARAGRAPH] * 2)),
            monument_concepts=("she", "he", "it"),
            tracked_concepts=tracked + ("she", "he", "it"),
            centre_prompts=[runtime.compose_prompt(question, chat=True)],
            centre_answer=True,
            max_new_tokens=8,
        )
    if "f" in cases:
        assert bundle_a is not None and neuron is not None
        await record_every(
            "f",
            DEFAULT_PROMPT,
            monument_concepts=landmarks,
            tracked_concepts=tracked,
            survey_mode="pole_motif",
            locked_pole=(35, neuron, 1.0),
            answer_token_ids=list(bundle_a.generated_token_ids),
        )
    if "a" not in cases:
        out.pop("a", None)
    return out


def allocator_settings() -> str:
    """The CUDA allocator's settings as torch reports them, "" when it reports none (an older torch: "")."""
    import torch

    try:
        return str(torch._C._accelerator_getAllocatorSettings() or "")
    except Exception:
        return ""


def record_mode(repo: Path, out_path: Path, cases: List[str]) -> int:
    sys.path.insert(0, str(repo))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch

    head: Optional[str]
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        head, dirty = None, False
    flat = asyncio.run(run_cases(repo, cases))
    meta = {
        "repo": str(repo),
        "git_head": head,
        "dirty": dirty,
        "torch": str(torch.__version__),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        # what the CUDA allocator ran with ("" for its defaults); the readings do not depend on it
        "allocator": allocator_settings(),
        "cases": list(flat),
    }
    torch.save({"meta": meta, "cases": flat}, str(out_path))
    print(f"saved {sum(len(v) for v in flat.values())} keys over cases {list(flat)} to {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__
    )
    ap.add_argument("--repo", type=Path, help="the checkout whose backend to record (its root directory)")
    ap.add_argument("--out", type=Path, help="where to save the recording (.pt)")
    ap.add_argument("--cases", default=",".join(CASES), help="which cases to record, e.g. a,c (default: all six)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare two recordings and exit")
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    if not args.repo or not args.out:
        ap.error("record mode needs --repo and --out (or use --compare A B)")
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    unknown = [c for c in cases if c not in CASES]
    if unknown:
        ap.error(f"unknown cases {unknown}; choose from {CASES}")
    return record_mode(args.repo.resolve(), args.out, cases)


if __name__ == "__main__":
    sys.exit(main())

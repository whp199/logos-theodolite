#!/usr/bin/env python3
"""What the pod scripts (scripts/pod/*.sh) ask of Python: the self-test a pod's setup ends with, the pilot's decode
steps and peak memory per unpacker and batch, the census_run.py arguments a home canary was taken with, the listing a
final pull is checked by, and the pilot's summary with its projections.

    pod_tool.py selftest [--stand-in] [--unpacker lut] [--out selftest.json]
    pod_tool.py steps --layer 45 --batches 32,96,128,256,512 --unpackers reference,lut,compiled --max-new 64 --out steps.json
    pod_tool.py canary-args home/L45-main_rel.canary.jsonl      # census_run.py's arguments for the same batch, one a line
    pod_tool.py listing --local ~/pods/h200-a --remote-list ~/pods/h200-a/remote.list [--json listing.json]
    pod_tool.py summary <pilot folder> [--approved-hours 12 --jobs whole_layer:2,depth_curve:1,steer_lens:1]

selftest loads the runtime (backend/runtime_factory.py: the W4A16 checkpoint on cuda:0 unless THEODOLITE_GGUF or
THEODOLITE_DEVICE say otherwise; --stand-in the CPU stand-in), installs the unpacker with its self-test against the
reference on every 4-bit linear (backend/w4a16_fast.py; refused in plain words on any difference), and prints one line:
the card, the versions, the unpacker's test and the runtime class (census.FLOOR_CLASS, the fields a run's bits depend on
beside its batch, with a short digest to compare pods by). steps writes census-shaped batches (one question, row 0
unpushed, the rest pushed by controls at +0.8 of the stream; no end mark, so every row runs to --max-new, as a pushed
answer that loops does) and times each decode step from the first row's pieces, with torch's peak memory above the
model on a GPU; a batch that does not fit is recorded as such and the larger ones skipped. listing sets a pulled copy
against the pod's own listing (`rsync --list-only`): every file the pod has, here at its size, and every census meta's
files (its run, a shard's canary, a joined run's shards), and which jobs have not ended (logs/<job>.started without its
.done, .failed or .timeout). summary gathers a pilot's parts (scripts/pod/pilot.sh) into pilot.json, with each job's
projected GPU-hours on the fastest batch that fit, by census_run.py's own formula (answers / batch x (max_new + 1) x
the step; full batches, every answer run to max_new), x 1.25 (the budget rule's margin), and "proceed" or "stop" when
an approval is given. Every projection is arithmetic on one measured step, not a measured job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

MARGIN = 1.25  # the budget rule's registered margin (R3_PREREG D30; the plan for rented cards, §7)
CONTROL_DOSE = 0.8
# a whole main_rel layer: every neuron and 256 controls, the design's 10 doses each, and the four baselines; the
# reference's depth curve: 384 neurons and 64 controls a layer (census_run.py's help)
CENSUS_CONTROLS = 256
DEPTH_NEURONS, DEPTH_CONTROLS = 384, 64


def load(stand_in: bool) -> Any:
    if stand_in:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["HF_HUB_OFFLINE"] = "1"
    import torch

    torch.set_grad_enabled(False)
    if stand_in:
        from atlas_stand_in import stand_in as build, use_stand_in_atlas

        use_stand_in_atlas()
        return build()[0]
    from backend.runtime_factory import make_runtime

    return make_runtime(load_jlens=False).load()


def runtime_class(rec: Dict[str, Any]) -> Dict[str, Any]:
    from backend.census import FLOOR_CLASS

    cls = {k: rec.get(k) for k in FLOOR_CLASS}
    cls["id"] = hashlib.sha256(json.dumps(cls, sort_keys=True).encode()).hexdigest()[:12]
    return cls


def write_json(path: str, data: Any) -> None:
    """`data` to `path` atomically (Python's own json alone: the listing runs on the owner's machine, venv or not)."""
    if path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, p)


def cmd_selftest(a: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    rt = load(a.stand_in)
    load_s = round(time.perf_counter() - t0, 1)
    import torch

    from backend import atlas_ops as ao
    from backend import census as cz
    from backend import w4a16_fast
    from backend.causal_check import language_model

    lm = language_model(rt.hf_model)
    device = ao.model_device(lm)
    try:
        unp = w4a16_fast.install(rt, a.unpacker, self_test=True, log=lambda s: None)
        st = unp.get("self_test") or {}
        tested = (f"{unp['mode']} on {unp['linears']} linears: " +
                  ("the reference unpacker itself" if unp["mode"] == "reference" else
                   "the same bits as the reference" + (" and on the piece-by-scale table" if st.get("table") else "")
                   if st.get("same_bits") else "not tested"))
        refused = None
    except w4a16_fast.Refused as exc:
        unp, tested, refused = {"mode": "reference"}, f"{a.unpacker} REFUSED: {exc}", str(exc)
    rec = cz.runtime_record(device)
    cls = runtime_class(rec)
    total = torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None
    import platform

    out = {"what": "a pod's self-test (scripts/pod/pod_tool.py selftest)", "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "model": getattr(rt, "model_id", None), "stand_in": bool(getattr(rt, "atlas_stand_in", False)), "load_s": load_s,
           "card": rec["device"], "card_gib": None if total is None else round(total / 2**30, 1), "python": platform.python_version(),
           "runtime": rec, "class": cls, "unpacker": unp, "unpacker_test": tested, "refused": refused,
           "n_layers": len(lm.layers), "d_ff": ao.ffn_width(lm, 0)}
    write_json(a.out, out)
    print(f"self-test: card {rec['device']}" + (f" ({out['card_gib']} GiB)" if total else "") + f" | python {out['python']}, torch "
          f"{rec['torch']} (CUDA {rec['cuda']}), transformers {rec['transformers']}, interp-engine {rec['interp_engine']}, "
          f"compressed-tensors {rec['compressed_tensors']} | unpacker {tested} | class {cls['id']} | {out['model']} loaded in {load_s} s",
          flush=True)
    return 1 if refused else 0


def stream_p50(rt: Any, layer: int) -> Any:
    from backend import atlas_ops as ao
    from backend.atlas_index import AtlasIndex
    from backend.causal_check import language_model

    lm = language_model(rt.hf_model)
    L, N = len(lm.layers), ao.ffn_width(lm, 0)
    stats = AtlasIndex(model=getattr(rt, "model_id", None), shape=(L, N)).get("act_scale")
    if stats is not None:
        return float(stats["resid_q"][layer, 1]), "the stats index"
    from backend.mcp_scientist import _study_ids

    return float(ao.stream_p50(ao.activation_extremes(lm, _study_ids(rt)))[layer]), "the study's 36 texts"


def cmd_steps(a: argparse.Namespace) -> int:
    rt = load(a.stand_in)
    import torch

    from backend import atlas_ops as ao
    from backend import census as cz
    from backend import w4a16_fast
    from backend.causal_check import language_model
    from census_run import free_lens

    free_lens(rt)
    lm = language_model(rt.hf_model)
    w = ao.Weights(lm, rt)
    L, N, D = w.n_layers, w.d_ff, w.d_model
    layer = a.layer if a.layer is not None else (3 if a.stand_in else 45)
    if not 0 <= layer < L:
        raise SystemExit(f"Layer {layer} does not exist: layers count from 0 to {L - 1}.")
    p50, p50_source = stream_p50(rt, layer)
    batches = sorted({int(b) for b in a.batches.split(",") if b.strip()})
    unpackers = [u.strip() for u in a.unpackers.split(",") if u.strip()]
    ids = rt.tokenize_prompt(rt.compose_prompt(cz.PROMPTS["who"], chat=True))
    ctrl = torch.from_numpy(cz.control_dirs(0, max(batches), w.norm(layer, "post_feedforward").cpu().numpy())).to(w.device)
    on_cuda = w.device.type == "cuda"
    rows: List[Dict[str, Any]] = []
    for u in unpackers:
        try:
            rec = w4a16_fast.install(rt, u, log=lambda s: print(s, flush=True))
        except w4a16_fast.Refused as exc:
            rows.append({"unpacker": u, "refused": str(exc)})
            print(f"{u}: refused ({exc})", flush=True)
            continue
        for B in batches:
            delta = torch.zeros(B, D, dtype=torch.float32, device=w.device)
            delta[1:] = ctrl[: B - 1] * (CONTROL_DOSE * p50)
            iv = ao.Interventions(rows=B, adds={layer: delta}) if B > 1 else None
            if on_cuda:
                torch.cuda.synchronize(w.device)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(w.device)
                base = torch.cuda.memory_allocated(w.device)
            t0 = time.perf_counter()
            stamps: List[float] = []
            try:
                for st in ao.write_rows(lm, rt.lm_head_weight, [ids] * B, iv, max_new=a.max_new, stop_ids=[],
                                        bos_id=rt.tokenizer.bos_token_id, spare_bos=True, softcap=ao.softcap_of(rt.hf_model), top_k=5):
                    if st.row == 0:
                        stamps.append(time.perf_counter())
            except torch.cuda.OutOfMemoryError as exc:
                del exc
                if on_cuda:
                    torch.cuda.empty_cache()
                rows.append({"unpacker": u, "batch": B, "fits": False, "max_new": a.max_new})
                print(f"{u} batch {B}: does not fit at --max-new {a.max_new}; larger batches skipped", flush=True)
                break
            diffs = [y - x for x, y in zip(stamps, stamps[1:])]
            r = {"unpacker": u, "batch": B, "fits": True, "max_new": a.max_new, "steps": len(stamps),
                 "prefill_s": round(stamps[0] - t0, 4) if stamps else None,
                 "step_s": round(statistics.median(diffs), 5) if diffs else None,
                 "step_s_mean": round(statistics.fmean(diffs), 5) if diffs else None,
                 "peak_gib": round((torch.cuda.max_memory_allocated(w.device) - base) / 2**30, 3) if on_cuda else None,
                 "install_s": rec.get("install_s")}
            r["rows_per_s"] = round(B / r["step_s"], 2) if r["step_s"] else None
            rows.append(r)
            print(f"{u} batch {B}: {r['step_s']} s a step (median of {len(diffs)}), prefill {r['prefill_s']} s"
                  + (f", {r['peak_gib']} GiB above the model at its peak" if on_cuda else ""), flush=True)
    w4a16_fast.install(rt, "reference", log=lambda s: None)
    fit = [r for r in rows if r.get("fits") and r.get("rows_per_s")]
    best = max(fit, key=lambda r: (r["rows_per_s"], -r["batch"])) if fit else None
    out = {"what": "decode steps of census-shaped batches, per unpacker and batch (scripts/pod/pod_tool.py steps)",
           "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": getattr(rt, "model_id", None),
           "card": cz.runtime_record(w.device)["device"], "layer": layer, "p50": p50, "p50_source": p50_source,
           "n_layers": L, "d_ff": N, "max_new": a.max_new, "memory": "torch's peak above the model" if on_cuda else "not measured off the GPU",
           "rows": rows, "best": best}
    write_json(a.out, out)
    if best:
        print(f"fastest that fit: {best['unpacker']} at batch {best['batch']}, {best['step_s']} s a step ({best['rows_per_s']} answers' "
              "pieces a second)", flush=True)
    return 0 if best else 1


def cmd_canary_args(a: argparse.Namespace) -> int:
    from backend import census as cz

    can = Path(a.canary)
    about = cz.canary_about_path_of(can)
    if not about.exists():
        raise SystemExit(f"{about.name} is not beside {can.name}: only a canary taken by census_run.py --canary-only says what "
                         "it was taken with.")
    m = json.loads(about.read_text(encoding="utf-8"))["meta"]
    if m.get("plan") != cz.PLAN:
        raise SystemExit(f"{can.name} was taken by the plan of a run begun before batch plans ({m.get('plan')}): take it again.")
    run = can.name[: -len(".canary.jsonl")] if can.name.endswith(".canary.jsonl") else can.stem
    args = ["--layer", str(m["layer"]), "--design", str(m["design_name"]), "--neurons", str(m["neuron_ids"]),
            "--random-controls", str(m["random_controls"]), "--seed", str(m["seed"]), "--max-new", str(m["max_new"]),
            "--top-k", str(m["top_k"]), "--batch", str(m["batch"]), "--dose-unit", str(m.get("dose_unit", "stream")), "--run", run]
    print("\n".join(args))
    return 0


LIST_LINE = re.compile(r"^(?P<perm>[-dlcbps][-rwxsStT]{9})\s+(?P<size>[\d,]+)\s+\S+\s+\S+\s+(?P<path>.+)$")


def read_listing(path: Path) -> Dict[str, int]:
    """{relative path: size} of the files in an `rsync -r --list-only` listing (directories and links left out)."""
    files: Dict[str, int] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LIST_LINE.match(line.strip())
        if m and m.group("perm")[0] == "-":
            files[m.group("path")] = int(m.group("size").replace(",", ""))
    return files


def cmd_listing(a: argparse.Namespace) -> int:
    local = Path(a.local)
    remote = read_listing(Path(a.remote_list))
    here = {str(p.relative_to(local)): p.stat().st_size for p in local.rglob("*") if p.is_file()}
    missing = sorted(p for p in remote if p not in here)
    short = sorted(p for p in remote if p in here and here[p] < remote[p])
    other = sorted(p for p in remote if p in here and here[p] > remote[p])
    named: List[str] = []
    metas = sorted(p for p in here if p.endswith(".meta.json"))
    for mp in metas:
        try:
            m = json.loads((local / mp).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            named.append(f"{mp}: not a JSON meta")
            continue
        stem = mp[: -len(".meta.json")]
        want = [stem + ".jsonl"]
        sh = m.get("shard") or {}
        if sh and int(sh.get("i", 0)) > 0:
            want.append(stem + ".canary.jsonl")
        folder = str(Path(mp).parent)
        want += [str(Path(folder) / s) if folder != "." else s for s in (m.get("shards") or [])]
        for f in want:
            if f not in here:
                named.append(f"{f} (named by {mp})")
    logs = {p for p in here if p.startswith("logs/")}
    running = sorted(p[len("logs/"): -len(".started")] for p in logs if p.endswith(".started")
                     and not any(p[: -len(".started")] + e in logs for e in (".done", ".failed", ".timeout")))
    ok = not (missing or short or other or named)
    says = (f"{len(remote)} files on the pod, all here at their size; {len(metas)} census metas, every file they name here"
            if ok else
            f"{len(missing)} missing, {len(short)} shorter here, {len(other)} longer here than on the pod, {len(named)} named by a meta "
            "and absent: pull again (sync.sh --final) before the pod or its volume goes")
    if running:
        says += f"; {len(running)} job{'s' if len(running) != 1 else ''} not ended ({', '.join(running)}): pull again when they have"
    rec = {"ok": ok, "remote_files": len(remote), "local_files": len(here), "missing": missing, "shorter_here": short,
           "longer_here": other, "metas": len(metas), "named_absent": named, "running": running, "says": says}
    write_json(a.json, rec)
    print(f"listing: {says}")
    for what, xs in (("missing", missing), ("shorter here", short), ("longer here", other), ("named by a meta, absent", named)):
        for x in xs[:20]:
            print(f"  {what}: {x}")
    return 0 if ok else 1


def projections(steps: Dict[str, Any], steer: Optional[Dict[str, Any]], margin: float = MARGIN) -> Dict[str, Any]:
    """GPU-hours of each job by census_run.py's formula on the fastest batch that fit (and of the steer lens, from one
    timed layer, by each layer's share of the later layers it runs through: an inference), each x `margin`."""
    best = steps.get("best")
    out: Dict[str, Any] = {"margin": margin, "basis": None}
    if best:
        N, L, M = int(steps["d_ff"]), int(steps["n_layers"]), int(steps["max_new"])
        step, B = float(best["step_s"]), int(best["batch"])

        def hours(answers: int) -> float:
            return answers / B * (M + 1) * step / 3600.0

        whole = (N + CENSUS_CONTROLS) * 10 + 4
        depth = ((min(DEPTH_NEURONS, N) + DEPTH_CONTROLS) * 10 + 4) * L
        out["basis"] = (f"{best['unpacker']} at batch {B}, {step} s a step at --max-new {M}: answers / batch x (max_new + 1) x step, "
                        "full batches and every answer run to max_new")
        out["whole_layer"] = {"answers": whole, "gpu_hours": round(hours(whole), 3), "with_margin": round(hours(whole) * margin, 3)}
        out["depth_curve"] = {"answers": depth, "gpu_hours": round(hours(depth), 3), "with_margin": round(hours(depth) * margin, 3)}
    if steer and steer.get("seconds") is not None and steps.get("n_layers"):
        L, lm_ = int(steps["n_layers"]), int(steer["layer"])
        total = steer["seconds"] * sum(L - l for l in range(L)) / (L - lm_) / 3600.0
        out["steer_lens"] = {"layer_timed": lm_, "layer_s": steer["seconds"], "gpu_hours": round(total, 3),
                             "with_margin": round(total * margin, 3),
                             "basis": "one layer timed, the others by their share of the later layers (L - l), inferred"}
    return out


def cmd_summary(a: argparse.Namespace) -> int:
    d = Path(a.dir)

    def part(name: str) -> Optional[Dict[str, Any]]:
        p = d / name
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        except ValueError:
            return {"unreadable": name}

    selftest, steps, floor = part("selftest.json"), part("steps.json"), part("atlas_writer_floor.json")
    calib, curves, steer, canary, zst = part("calib.json"), part("curves.json"), part("steer.json"), part("canary.json"), part("zstd.json")
    proj = projections(steps or {}, steer)
    out: Dict[str, Any] = {
        "what": "a pod's one-hour pilot (scripts/pod/pilot.sh): the card, the decode step and peak memory per unpacker and batch, "
                "the writer's floor, a calib_rel run, one steer-lens layer, a canary against the home card's, the zstd ratio",
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "folder": str(d),
        "card": (selftest or {}).get("card"), "class": (selftest or {}).get("class"), "selftest": selftest,
        "steps": (steps or {}).get("rows"), "chosen": (steps or {}).get("best"),
        "writer_floor": None if floor is None else {k: floor.get(k) for k in ("floor", "floor_at", "census_floor", "by_shape",
                                                                                "alone_twice_same", "max_new", "seconds")},
        "calib_rel": calib, "curves": None if curves is None else {"run": curves.get("run"), "window": curves.get("window")},
        "steer_lens": steer, "canary": canary, "zstd": zst, "projections": proj,
    }
    if a.approved_hours is not None:
        plan = {}
        for item in [x for x in a.jobs.split(",") if x.strip()]:
            name, _, count = item.partition(":")
            plan[name.strip()] = float(count or 1)
        missing = [n for n in plan if n not in proj]
        need = sum(proj[n]["with_margin"] * c for n, c in plan.items() if n in proj)
        out["budget"] = {"approved_gpu_hours": a.approved_hours, "jobs": plan, "projected_with_margin": round(need, 3),
                         "cannot_project": missing,
                         "decision": "stop" if missing or not plan or need > a.approved_hours else "proceed"}
    write_json(str(d / "pilot.json"), out)
    print(f"pilot: {out['card']}, class {(out['class'] or {}).get('id')}")
    if out["chosen"]:
        c = out["chosen"]
        print(f"  fastest that fit: {c['unpacker']} at batch {c['batch']}, {c['step_s']} s a step"
              + (f", {c['peak_gib']} GiB above the model" if c.get("peak_gib") is not None else ""))
    for name in ("whole_layer", "depth_curve", "steer_lens"):
        if name in proj:
            print(f"  {name}: {proj[name]['gpu_hours']} GPU-hours, {proj[name]['with_margin']} with the 1.25 margin")
    if floor:
        print(f"  writer's floor: {floor.get('floor')} (census batches: {floor.get('census_floor')})")
    if canary:
        print(f"  canary: {canary.get('says')}")
    if zst:
        print(f"  zstd: {zst.get('says')}")
    if "budget" in out:
        b = out["budget"]
        print(f"  budget: {b['projected_with_margin']} of {b['approved_gpu_hours']} approved GPU-hours -> {b['decision']}")
    print(f"-> {d / 'pilot.json'}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    sub = ap.add_subparsers(dest="what", required=True)
    s = sub.add_parser("selftest", help="load the runtime, test the unpacker, print the card, versions and class")
    s.add_argument("--stand-in", action="store_true")
    s.add_argument("--unpacker", default="lut", choices=["reference", "lut", "compiled", "resident"])
    s.add_argument("--out", default="")
    t = sub.add_parser("steps", help="time decode steps of census-shaped batches, per unpacker and batch")
    t.add_argument("--stand-in", action="store_true")
    t.add_argument("--layer", type=int, default=None, help="the layer pushed (default 45; 3 on the stand-in)")
    t.add_argument("--batches", default="32,96,128,256,512")
    t.add_argument("--unpackers", default="reference,lut,compiled")
    t.add_argument("--max-new", type=int, default=64)
    t.add_argument("--out", default="")
    c = sub.add_parser("canary-args", help="census_run.py's arguments for the batch a canary taken alone holds")
    c.add_argument("canary")
    li = sub.add_parser("listing", help="set a pulled copy against the pod's own listing")
    li.add_argument("--local", required=True)
    li.add_argument("--remote-list", required=True)
    li.add_argument("--json", default="")
    su = sub.add_parser("summary", help="gather a pilot's parts into pilot.json")
    su.add_argument("dir")
    su.add_argument("--approved-hours", type=float, default=None)
    su.add_argument("--jobs", default="", help='the jobs the approval is for, "whole_layer:2,depth_curve:1,steer_lens:1"')
    a = ap.parse_args(argv)
    return {"selftest": cmd_selftest, "steps": cmd_steps, "canary-args": cmd_canary_args, "listing": cmd_listing,
            "summary": cmd_summary}[a.what](a)


if __name__ == "__main__":
    sys.exit(main())

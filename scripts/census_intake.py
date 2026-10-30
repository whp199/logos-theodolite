#!/usr/bin/env python3
"""A volunteer's layer brought home and checked before it stands beside the home card's (VOLUNTEERS.md, "How a run is
checked"). A bundle is someone else's file: it is unpacked into a folder of its own, only the names a pack writes are
taken from it, and nothing in it is run.

    census_intake.py home-canaries                     # on the home card: each kit layer's canary, into volunteers/home_canaries/
    census_intake.py check ~/Downloads/L47-break_rel-3f9a1c2e.tar   # unpack, check, plan the spot chunks
    census_intake.py spot atlas_data/census/volunteer/L47-3f9a1c2e --run    # those chunks written again on the home card
    census_intake.py judge atlas_data/census/volunteer/L47-3f9a1c2e         # the spot chunks against the volunteer's
    census_intake.py accept atlas_data/census/volunteer/L47-3f9a1c2e        # into the census folder, ranked, on the board

check: the manifest (the kit, the layer, the claim, whose code its secret must give, each file's sha256); the answers
unpacked (zstd) and their sha256; the meta set against the home canary's of that layer (backend/census_volunteer.
meta_problems: every setting a run is continued by, the push's p50, the job list, the stats index, the unpacker's test),
the commit the kit's tag names and a clean checkout; every answer of the job list once and no other; the volunteer's
canary against their run's first batch (the same card writing the same batch) and against the home card's (the same
bits, or a card class of its own); the six unpushed answers against the home card's; and the spot plan, SPOT_CHUNKS of
the run's planned chunks drawn at random here, after the bundle came, so no volunteer knows them. A bundle that fails
any of these is not spot-checked.

spot: each spot chunk written again on the home card (census_run.py --shard i/C, the kit's settings): a minute or so a
chunk on the 5090, the model loaded for each; the server must be stopped for the card's memory. Without --run it prints
the commands.

judge: the spot answers against the volunteer's (census_volunteer.compare_rows, judge): bit for bit when its canary was
the home card's bits, else by ACCEPT_OTHER_CLASS; the verdict into verdict.json and onto the board.

accept: an accepted run copied into the census folder as L<layer>-<design>.jsonl with its meta (refused if one is there),
a provenance file beside it (who, the card, the bits, the bundle, the checks), ranked, and its layer done on the board.
Layers count from 0 (page = layer + 1).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend import census as cz  # noqa: E402
from backend import census_volunteer as cv  # noqa: E402

PY = sys.executable
SPOT_CHUNKS = 5
MAX_RUN_BYTES = 4_000_000_000
MAX_SMALL_BYTES = 64_000_000
PACKED = [r"manifest\.json", r"L\d{1,2}-[a-z_]+\.jsonl\.zst", r"L\d{1,2}-[a-z_]+\.meta\.json", r"L\d{1,2}-[a-z_]+\.canary\.jsonl",
          r"L\d{1,2}-[a-z_]+\.canary\.json", r"L\d{1,2}\.writer_floor\.json", r"L\d{1,2}\.canary_check\.json"]


def home_dir() -> Path:
    return Path(os.environ.get("THEODOLITE_KIT_HOME_CANARIES") or cv.HOME_CANARIES)


def work_root() -> Path:
    from backend.atlas_index import atlas_dir

    return atlas_dir() / "census" / "volunteer"


def kit_args(kit: Dict[str, Any]) -> List[str]:
    return ["--design", kit["KIT_DESIGN"], "--neurons", str(kit["KIT_NEURONS"]), "--random-controls", str(kit["KIT_CONTROLS"]),
            "--seed", str(kit["KIT_SEED"]), "--batch", str(kit["KIT_BATCH"]), "--max-new", str(kit["KIT_MAX_NEW"]),
            "--unpacker", kit["KIT_UNPACKER"]]


def run(cmd: List[str], **kw: Any) -> subprocess.CompletedProcess:
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kw)


# ------------------------------------------------------------------------------------------------------- home canaries

def cmd_home_canaries(a: argparse.Namespace, kit: Dict[str, Any]) -> int:
    """Each kit layer's canary on this card, and the six unpushed answers a volunteer's are set against."""
    out = home_dir()
    out.mkdir(parents=True, exist_ok=True)
    layers = a.layers or ",".join(str(l) for l in kit["layers"])
    stand = ["--stand-in"] if a.dry_run else []
    got = run([PY, "-I", "scripts/census_run.py", *stand, "--layers", layers, *kit_args(kit), "--canary-only",
               "--out-dir", str(out), "--run", "L{layer}-" + kit["KIT_DESIGN"]], cwd=ROOT_DIR)
    if got.returncode != 0:
        return got.returncode
    src = Path(a.baselines_from) if a.baselines_from else cz.census_dir() / f"L45-{kit['KIT_DESIGN']}.jsonl"
    base = {}
    for r in cz.read_rows(src):
        if r["kind"] == "baseline":
            base[r["prompt"]] = {"token_ids": r["token_ids"], "logprob": r["logprob"]}
        elif len(base) >= 6:
            break
    cz.write_json_atomic(out / "baselines.json", {"what": "the six unpushed answers of the kit's design on the home card (they "
                                                          "are the same at every layer)", "from": src.name, "baselines": base})
    prints = {}
    for p in sorted(out.glob(f"L*-{kit['KIT_DESIGN']}.canary.jsonl")):
        prints[p.name] = cv.rows_fingerprint(list(cz.read_rows(p)))
    cz.write_json_atomic(out / "fingerprints.json", prints)
    # a fresh canary of a layer the home card has taken whole must be that run's first batch
    for p in sorted(out.glob(f"L*-{kit['KIT_DESIGN']}.canary.jsonl")):
        whole = cz.census_dir() / p.name.replace(".canary.jsonl", ".jsonl")
        if whole.exists() and not a.dry_run:
            rows = list(cz.read_rows(p))
            import itertools

            first = list(itertools.islice(cz.read_rows(whole), len(rows)))
            print(f"{p.name}: {'the same as' if first == rows else 'NOT the same as'} {whole.name}'s first batch")
    print(f"-> {out}: {len(prints)} canaries, baselines from {src.name}")
    return 0


# --------------------------------------------------------------------------------------------------------------- check

def safe_unpack(tar_path: Path, dest: Path) -> Dict[str, Path]:
    """The bundle's files, by name, unpacked into `dest` (made empty): regular files with the names a pack writes, no
    folders or links, under the size caps; anything else refused."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with tarfile.open(tar_path, "r:") as t:
        members = t.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise SystemExit("The bundle holds a name twice.")
        for m in members:
            if not m.isreg() or "/" in m.name or not any(re.fullmatch(p, m.name) for p in PACKED):
                raise SystemExit(f"The bundle holds {m.name!r}, which a pack does not write: not unpacked.")
            cap = MAX_RUN_BYTES if m.name.endswith(".jsonl.zst") else MAX_SMALL_BYTES
            if m.size > cap:
                raise SystemExit(f"{m.name} is {m.size:,} bytes, more than a pack writes ({cap:,}).")
        t.extractall(dest, members=members, filter="data")
    return {n: dest / n for n in names}


def check_bundle(a: argparse.Namespace, kit: Dict[str, Any]) -> Dict[str, Any]:
    tar_path = Path(a.bundle).expanduser().resolve()
    m = re.fullmatch(r"L(\d{1,2})-[a-z_]+-([0-9a-f]{8})\.tar", tar_path.name)
    if not m:
        raise SystemExit(f"{tar_path.name} is not named as a pack names a bundle (L<layer>-<design>-<claim>.tar).")
    work = Path(a.work) if a.work else work_root() / f"L{int(m.group(1))}-{m.group(2)}"
    files = safe_unpack(tar_path, work / "bundle")
    fails: List[str] = []
    notes: List[str] = []
    man = json.loads(files["manifest.json"].read_text(encoding="utf-8")) if "manifest.json" in files else None
    if man is None:
        raise SystemExit("The bundle has no manifest.json.")
    layer, code = int(man.get("layer", -1)), str(man.get("claim", ""))
    out: Dict[str, Any] = {"bundle": tar_path.name, "bundle_sha256": cz.file_sha256(tar_path), "work": str(work), "layer": layer,
                           "claim": code, "name": str(man.get("name", ""))[:24], "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if (layer, code) != (int(m.group(1)), m.group(2)):
        fails.append("the manifest's layer and claim are not the bundle's name's")
    if int(man.get("kit", -1)) != kit["KIT"]:
        fails.append(f"packed by kit {man.get('kit')}, not kit {kit['KIT']}")
    if layer not in kit["layers"]:
        fails.append(f"layer {layer} is not a kit layer ({' '.join(map(str, kit['layers']))})")
    out["secret_ok"] = cv.claim_code(str(man.get("secret", ""))) == code
    if not out["secret_ok"]:
        fails.append("the manifest's secret does not give its claim's code: not the claim's bundle")
    for name, want in (man.get("files") or {}).items():
        p = files.get(name)
        if p is None:
            fails.append(f"{name} is named in the manifest but not in the bundle")
        elif p.stat().st_size != want.get("bytes") or cz.file_sha256(p) != want.get("sha256"):
            fails.append(f"{name} is not the file the manifest names (size or sha256)")
    board = load_board(a)
    if board is not None:
        from census_board import find

        c = find(board, code)
        if c is None or c["layer"] != layer:
            notes.append(f"claim {code} on layer {layer} is not on the board")
        else:
            out["board_state"] = c["state"]
            if c.get("bundle_sha256") and not out["bundle_sha256"].startswith(c["bundle_sha256"]):
                fails.append(f"the bundle's sha256 is not the one its DONE line gave ({c['bundle_sha256']})")
    design = kit["KIT_DESIGN"]
    stem = f"L{layer}-{design}"
    zst, meta_p = files.get(f"{stem}.jsonl.zst"), files.get(f"{stem}.meta.json")
    if fails or zst is None or meta_p is None:
        out.update({"fails": fails + ([] if zst and meta_p else [f"the bundle lacks {stem}.jsonl.zst or its meta"]), "notes": notes})
        return out
    runp = work / f"{stem}.jsonl"
    raw = (man.get("run") or {})
    if int(raw.get("bytes") or 0) > MAX_RUN_BYTES:
        fails.append(f"the manifest says the answers are {raw.get('bytes')} bytes, more than a layer's")
    else:
        got = run(["zstd", "-q", "-d", "-f", "-o", str(runp), str(zst)])
        if got.returncode != 0 or not runp.exists() or runp.stat().st_size != int(raw.get("bytes") or -1):
            fails.append("the answers do not unpack to the size the manifest names")
    home_rows, home_about = cv.home_canary(layer, design, home_dir())
    if not home_about.exists():
        fails.append(f"no home canary of layer {layer} ({home_about}): census_intake.py home-canaries first")
    if fails:
        out.update({"fails": fails, "notes": notes})
        return out
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    home_meta = json.loads(home_about.read_text(encoding="utf-8"))["meta"]
    fails += cv.meta_problems(meta, home_meta, kit)
    rt = meta.get("runtime") or {}
    out["runtime"] = rt
    if rt.get("git_clean") is not True:
        (notes if a.dry_run else fails).append("the run was begun from a checkout with changes (git_clean is not true)")
    if not a.dry_run:
        tag = subprocess.run(["git", "-C", str(ROOT_DIR), "rev-parse", f"{kit['KIT_TAG']}^{{commit}}"], capture_output=True, text=True)
        if tag.returncode != 0:
            fails.append(f"the kit's tag {kit['KIT_TAG']} is not in this checkout: the run's commit cannot be checked")
        elif rt.get("git") != tag.stdout.strip():
            fails.append(f"the run's commit {str(rt.get('git'))[:12]} is not the kit's ({kit['KIT_TAG']}, {tag.stdout.strip()[:12]})")
    if meta.get("mixed_runtime"):
        parts = meta["mixed_runtime"]
        moved = sorted({k for p in parts for k in cz.runtime_differ(parts[0], p)})
        (fails if set(moved) - {"git_clean"} else notes).append(f"the run was continued on other runtimes ({', '.join(moved)} moved)")
    jobs = cv.expected_jobs(meta)
    if cz.jobs_digest(jobs) != meta.get("jobs"):
        fails.append("the meta's job list is not the one its settings plan")
    print(f"reading the answers through ({runp.stat().st_size / 1e9:.2f} GB)", flush=True)
    scan = cv.scan_run(runp, jobs, layer)
    out["scan"] = {k: v for k, v in scan.items() if k != "baselines"}
    if scan["sha256"] != raw.get("sha256"):
        fails.append("the answers' sha256 is not the manifest's")
    if not scan["whole"]:
        fails.append(f"the run is not whole: {scan['missing']:,} missing, {scan['extra']} not of its job list, {scan['twice']} twice, "
                     f"{scan['other_layer']} of another layer, {scan['misshapen']} misshapen")
    # the canary: the same card's first batch, and the home card's bits or not
    canary = files.get(f"{stem}.canary.jsonl")
    bits = "own"
    if canary is None:
        fails.append("the bundle has no canary")
    else:
        theirs = list(cz.read_rows(canary))
        import itertools

        first = list(itertools.islice(cz.read_rows(runp), len(theirs)))
        out["own_canary_is_first_batch"] = first == theirs
        if first != theirs:
            fails.append("the run's first batch is not the canary taken on the same card before it")
        cmp = cz.compare_canaries(home_rows, canary)
        out["canary"] = {k: cmp.get(k) for k in ("same", "rows_differ", "earliest_departure", "settings_differ", "runtime_differ", "says")}
        bits = "home" if cmp["same"] else "own"
    out["bits"] = bits
    # the unpushed answers: the same at every layer
    hb = json.loads((home_dir() / "baselines.json").read_text(encoding="utf-8"))["baselines"] if (home_dir() / "baselines.json").exists() else {}
    shared = {}
    for q, b in hb.items():
        t = scan["baselines"].get(q)
        if t is None:
            continue
        shared[q] = [cv.shared_prefix(b["token_ids"], t["token_ids"]), len(b["token_ids"])]
        if bits == "home" and (t["token_ids"] != b["token_ids"] or t["logprob"] != b["logprob"]):
            fails.append(f"its canary is the home card's bits, but its unpushed {q} answer is not the home card's")
    out["baselines_shared"] = shared
    if not hb:
        notes.append("no home baselines (volunteers/home_canaries/baselines.json): the unpushed answers were not compared")
    floor_p = files.get(f"L{layer}.writer_floor.json")
    if floor_p:
        fl = json.loads(floor_p.read_text(encoding="utf-8"))
        out["writer_floor"] = {k: fl.get(k) for k in ("floor", "census_floor") if k in fl}
    else:
        notes.append("no writer's floor in the bundle")
    # the spot chunks, drawn here, now
    C = cz.plan_batches(jobs, int(meta["batch"]), meta["plan"])[-1][0] + 1
    seed = int.from_bytes(os.urandom(8), "big")
    chunks = sorted(random.Random(seed).sample(range(C), min(a.chunks, C)))
    out["spot"] = {"chunks": chunks, "of": C, "seed": seed, "drawn_at": out["checked_at"]}
    out.update({"fails": fails, "notes": notes})
    return out


def load_board(a: argparse.Namespace) -> Optional[Dict[str, Any]]:
    if not a.board or not Path(a.board).exists():
        return None
    sys.path.insert(0, str(ROOT_DIR / "scripts"))
    return json.loads(Path(a.board).read_text(encoding="utf-8"))


def set_board(a: argparse.Namespace, code: str, state: str, note: str) -> None:
    if not a.board or not Path(a.board).exists():
        return
    got = subprocess.run([PY, "-I", str(ROOT_DIR / "scripts" / "census_board.py"), "--board", a.board, "set", code, state,
                          "--note", note], capture_output=True, text=True, env={**os.environ})
    print(got.stdout.splitlines()[0] if got.stdout else got.stderr.strip()[:300])


def cmd_check(a: argparse.Namespace, kit: Dict[str, Any]) -> int:
    out = check_bundle(a, kit)
    work = Path(out["work"])
    cz.write_json_atomic(work / "check.json", out)
    for n in out["notes"]:
        print(f"note: {n}")
    if out["fails"]:
        for f in out["fails"]:
            print(f"FAIL: {f}")
        print(f"-> {work / 'check.json'}: not taken; the volunteer may be told the lines above")
        if out.get("secret_ok"):  # a bundle that does not hold its claim's secret changes nothing on the board
            set_board(a, out["claim"], "rejected", "; ".join(out["fails"])[:200])
        return 1
    whose = "the home card's bits" if out["bits"] == "home" else f"a card class of its own ({out['runtime'].get('device')})"
    print(f"layer {out['layer']} from {out['name']} ({out['claim']}): whole, the kit's settings, {whose}")
    print(f"spot chunks {', '.join(map(str, out['spot']['chunks']))} of {out['spot']['of']}: census_intake.py spot {work} --run")
    set_board(a, out["claim"], "checking", f"on paper whole; {out['bits']} bits; spot-check next")
    print(f"-> {work / 'check.json'}")
    return 0


# ---------------------------------------------------------------------------------------------------------- spot, judge

def cmd_spot(a: argparse.Namespace, kit: Dict[str, Any]) -> int:
    work = Path(a.work)
    chk = json.loads((work / "check.json").read_text(encoding="utf-8"))
    if chk["fails"]:
        raise SystemExit("This bundle failed its check: it is not spot-checked.")
    stand = ["--stand-in"] if a.dry_run else []
    cmds = [[PY, "-I", "scripts/census_run.py", *stand, "--layer", str(chk["layer"]), *kit_args(kit), "--shard", f"{c}/{chk['spot']['of']}",
             "--run", f"L{chk['layer']}-spot", "--out-dir", str(work / "spot"), "--no-rank"] for c in chk["spot"]["chunks"]]
    if not a.run:
        print("on the home card, the server stopped (its model holds the card's memory):")
        for c in cmds:
            print("  " + " ".join(c))
        print(f"or: census_intake.py spot {work} --run")
        return 0
    for c in cmds:
        if run(c, cwd=ROOT_DIR).returncode != 0:
            return 1
    print(f"-> {work / 'spot'}: next census_intake.py judge {work}")
    return 0


def cmd_judge(a: argparse.Namespace, kit: Dict[str, Any]) -> int:
    work = Path(a.work)
    chk = json.loads((work / "check.json").read_text(encoding="utf-8"))
    ours = []
    for c in chk["spot"]["chunks"]:
        p = work / "spot" / f"L{chk['layer']}-spot.s{c}of{chk['spot']['of']}.jsonl"
        if not p.exists():
            raise SystemExit(f"{p.name} is not there: census_intake.py spot {work} --run")
        ours += list(cz.read_rows(p))
    keys = {cz.row_key(r) for r in ours}
    theirs = {}
    for r in cz.read_rows(work / f"L{chk['layer']}-{kit['KIT_DESIGN']}.jsonl"):
        k = cz.row_key(r)
        if k in keys:
            theirs[k] = r
    cmp = cv.compare_rows(theirs, ours)
    ok, why = cv.judge(cmp, chk["bits"] == "home", chk.get("baselines_shared") if chk["bits"] != "home" else None)
    verdict = {"layer": chk["layer"], "claim": chk["claim"], "bits": chk["bits"], "card": (chk.get("runtime") or {}).get("device"),
               "spot": cmp, "rules": "bit for bit" if chk["bits"] == "home" else cv.ACCEPT_OTHER_CLASS,
               "accepted": ok, "why_not": why, "judged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    cz.write_json_atomic(work / "verdict.json", verdict)
    print(f"{cmp['present']} of {cmp['answers']} spot answers there, {cmp['identical']} identical, first piece {cmp['first_piece']}, "
          f"prefix median {cmp['prefix_median']}, |d log-prob| median {cmp['lp_median']} p99 {cmp['lp_p99']} over {cmp['lp_pieces']} pieces")
    if ok:
        note = "bit for bit on the spot chunks" if chk["bits"] == "home" else f"its own card class ({verdict['card']}), within the kit's rules"
        print(f"ACCEPTED: {note}. Next: census_intake.py accept {work}")
        set_board(a, chk["claim"], "checking", f"spot-checked: {note}; accept next")
    else:
        for w in why:
            print(f"REJECTED: {w}")
        set_board(a, chk["claim"], "rejected", "; ".join(why)[:200])
    print(f"-> {work / 'verdict.json'}")
    return 0 if ok else 1


def cmd_accept(a: argparse.Namespace, kit: Dict[str, Any]) -> int:
    work = Path(a.work)
    chk = json.loads((work / "check.json").read_text(encoding="utf-8"))
    ver = json.loads((work / "verdict.json").read_text(encoding="utf-8")) if (work / "verdict.json").exists() else {}
    if chk["fails"] or not ver.get("accepted"):
        raise SystemExit("Only a run that passed its check and its spot-check is accepted (check.json, verdict.json).")
    stem = f"L{chk['layer']}-{kit['KIT_DESIGN']}"
    dest = Path(a.into) if a.into else cz.census_dir()
    target = dest / f"{stem}.jsonl"
    if target.exists() or cz.meta_path_of(target).exists():
        raise SystemExit(f"{target} is there already: a layer's main run is not replaced (move it aside first, or --into elsewhere).")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(work / f"{stem}.jsonl", target)
    shutil.copy2(work / "bundle" / f"{stem}.meta.json", cz.meta_path_of(target))
    prov = {"what": f"layer {chk['layer']} taken by a volunteer with kit {kit['KIT']} (VOLUNTEERS.md), checked and spot-checked "
                    "at home (scripts/census_intake.py)", "name": chk["name"], "claim": chk["claim"], "card": ver.get("card"),
            "bits": chk["bits"], "bundle": chk["bundle"], "bundle_sha256": chk["bundle_sha256"], "check": chk, "verdict": ver,
            "accepted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    cz.write_json_atomic(dest / f"{stem}.provenance.json", prov)
    print(f"-> {target} (and its meta and provenance); ranking it", flush=True)
    units = cz.write_units(target)
    print(f"-> {units}")
    set_board(a, chk["claim"], "accepted", f"{chk['bits']} bits ({ver.get('card')}), spot-checked")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="the stand-in's kit (KIT_DRY_*): no stats index or tag checked")
    ap.add_argument("--board", default=str(ROOT_DIR / "volunteers" / "board.json"), help="the board to keep up to date ('' for none)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("home-canaries")
    h.add_argument("--layers", default="", help="default: the kit's")
    h.add_argument("--baselines-from", default="", help="a whole run of the home card (default: layer 45's)")
    c = sub.add_parser("check")
    c.add_argument("bundle")
    c.add_argument("--work", default="", help="default: <atlas>/census/volunteer/L<layer>-<claim>")
    c.add_argument("--chunks", type=int, default=SPOT_CHUNKS)
    s = sub.add_parser("spot")
    s.add_argument("work")
    s.add_argument("--run", action="store_true", help="write the chunks now, on this card")
    j = sub.add_parser("judge")
    j.add_argument("work")
    ac = sub.add_parser("accept")
    ac.add_argument("work")
    ac.add_argument("--into", default="", help="default: the census folder")
    a = ap.parse_args(argv)
    kit = cv.read_kit(dry=a.dry_run)
    return {"home-canaries": cmd_home_canaries, "check": cmd_check, "spot": cmd_spot, "judge": cmd_judge,
            "accept": cmd_accept}[a.cmd](a, kit)


if __name__ == "__main__":
    sys.exit(main())

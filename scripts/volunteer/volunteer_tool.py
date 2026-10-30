#!/usr/bin/env python3
"""What scripts/volunteer/census.sh runs in the venv (VOLUNTEERS.md): the canary set against the home card's, a run's
progress and the line to post, and the pack. It reads files only; the GPU is census_run.py's.

    volunteer_tool.py canary-check --layer 47 --canary <folder>/L47-break_rel.canary.jsonl --out <folder>/L47.canary_check.json
    volunteer_tool.py status --layer 47 --run <runs>/L47-break_rel.jsonl --logs <root>/logs --claim <root>/claims/L47.claim
    volunteer_tool.py pack --layer 47 --run <runs>/L47-break_rel.jsonl --canary-dir <folder> --claim <claim> --out <root>/outbox

canary-check: plan batch 0 of the layer's run as this card wrote it, against the same batch the home card wrote
(volunteers/home_canaries/, or $THEODOLITE_KIT_HOME_CANARIES): identical is the home card's bits ("bits": "home"), and
the run will be checked bit for bit; otherwise a card class of its own ("own"), checked by the kit's rules for one
(backend/census_volunteer.ACCEPT_OTHER_CLASS). A canary taken with other settings than the home card's is refused.

status: answers written of the run's total, the rate (over the samples this command has taken in the last hour, else
since the run began) and when it should end; whether its job is still running; and the PROGRESS line to post.

pack: the run checked as the kit's at its layer (its meta's settings against the home canary's, every answer of its
job list once, no other), its canary against its own first batch, then the answers compressed (zstd -19) and packed
with the meta, the canary, the writer's floor and a manifest (the claim's code and secret, each file's sha256) into one
uncompressed tar in --out: the file to send. It prints the DONE line to post with the link.
"""

from __future__ import annotations

import argparse
import calendar
import itertools
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend import census as cz  # noqa: E402
from backend import census_volunteer as cv  # noqa: E402


def home_dir() -> Path:
    return Path(os.environ.get("THEODOLITE_KIT_HOME_CANARIES") or cv.HOME_CANARIES)


def read_claim(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            k, eq, v = line.partition("=")
            if eq:
                out.setdefault(k, v)
    return out


def utc(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime(t))


def cmd_canary_check(a: argparse.Namespace, kit: Dict[str, Any]) -> int:
    canary = Path(a.canary)
    if not canary.exists():
        raise SystemExit(f"There is no canary at {canary}: census.sh canary {a.layer} writes it.")
    about = cz.canary_about_path_of(canary)
    theirs_meta = json.loads(about.read_text(encoding="utf-8"))["meta"] if about.exists() else {}
    rows = list(cz.read_rows(canary))
    home_rows, home_about = cv.home_canary(a.layer, kit["KIT_DESIGN"], home_dir())
    out: Dict[str, Any] = {"layer": a.layer, "kit": kit["KIT"], "fingerprint": cv.rows_fingerprint(rows),
                           "card": (theirs_meta.get("runtime") or {}).get("device"), "runtime": theirs_meta.get("runtime")}
    if not home_rows.exists():
        out.update({"bits": "unknown", "says": f"no home canary of layer {a.layer} in {home_dir()}: the owner sets yours against "
                                               "it when the run comes home"})
    else:
        cmp = cz.compare_canaries(home_rows, canary)
        out.update({"home_fingerprint": cv.rows_fingerprint(list(cz.read_rows(home_rows))), "same": cmp["same"],
                    "rows_differ": cmp["rows_differ"], "earliest_departure": cmp["earliest_departure"],
                    "settings_differ": cmp["settings_differ"], "runtime_differ": cmp["runtime_differ"], "says": cmp["says"]})
        if cmp["settings_differ"]:
            cz.write_json_atomic(Path(a.out), {**out, "bits": "refused"})
            raise SystemExit(f"This canary was taken with other settings than the home card's ({', '.join(cmp['settings_differ'])}): "
                             "the kit's run would not be the kit's. Is the checkout at the kit's tag, and the stats index the kit's?")
        out["bits"] = "home" if cmp["same"] else "own"
    cz.write_json_atomic(Path(a.out), out)
    if out["bits"] == "home":
        print(f"the same bits as the home card ({out['says']}): your run is the home card's instrument, and is checked bit for bit")
    elif out["bits"] == "own":
        print(f"not the home card's bits ({out['says']}): that is expected on another card or driver. Your run is recorded as a card "
              "class of its own and checked by the kit's rules for one (VOLUNTEERS.md, 'How a run is checked')")
    else:
        print(out["says"])
    print(f"-> {a.out}")
    return 0


def count_rows(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 23), b""):
            n += block.count(b"\n")
    return n


def writer_alive(layer: int) -> bool:
    """A census_run.py of this layer among this machine's processes (Linux /proc)."""
    for p in Path("/proc").glob("[0-9]*"):
        try:
            argv = (p / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if any(x.endswith(b"census_run.py") for x in argv) and b"--layer" in argv:
            i = argv.index(b"--layer")
            if i + 1 < len(argv) and argv[i + 1] == str(layer).encode():
                return True
    return False


def cmd_status(a: argparse.Namespace, kit: Dict[str, Any]) -> int:
    run = Path(a.run)
    meta_p = cz.meta_path_of(run)
    if not meta_p.exists():
        raise SystemExit(f"No run of layer {a.layer} here yet ({run}): census.sh run {a.layer} begins it.")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    total = int(meta["jobs"]["n"])
    now = time.time()
    rows = count_rows(run) if run.exists() else 0
    claim = read_claim(Path(a.claim))
    state_p = Path(a.claim).with_suffix(".status.json")
    samples = json.loads(state_p.read_text(encoding="utf-8")) if state_p.exists() else []
    samples = [s for s in samples if now - s[0] < 6 * 3600 and s[1] <= rows] + [[now, rows]]
    cz.write_json_atomic(state_p, samples[-60:])
    hour = [s for s in samples if now - s[0] <= 3600]
    rate = None
    if len(hour) >= 2 and hour[-1][0] - hour[0][0] >= 300 and hour[-1][1] > hour[0][1]:
        rate, how = (hour[-1][1] - hour[0][1]) / (hour[-1][0] - hour[0][0]), "over the last hour's looks"
    else:
        try:
            t0 = calendar.timegm(time.strptime(meta["started_at"], "%Y-%m-%dT%H:%M:%SZ"))
        except (KeyError, ValueError):
            t0 = None
        if t0 and rows and now > t0:
            rate, how = rows / (now - t0), "since the run began (a run stopped and continued reads slow here)"
    logs = Path(a.logs)
    job = run.stem
    ended = next((e for e in ("done", "failed", "timeout") if (logs / f"{job}.{e}").exists()), None)
    alive = writer_alive(a.layer)
    pct = 100.0 * rows / total if total else 0.0
    print(f"layer {a.layer}: {rows:,} of {total:,} answers ({pct:.1f}%)")
    if rows >= total:
        print(f"the run is whole: census.sh pack {a.layer}")
        line = cv.format_line("PROGRESS", a.layer, claim=claim.get("code"), done="100%")
    else:
        eta = None
        if rate:
            eta = now + (total - rows) / rate
            print(f"{rate:.2f} answers a second ({how}); about {(total - rows) / rate / 3600:.1f} hours left, done about {utc(eta)}")
        if alive:
            print("the census is running")
        elif ended == "failed":
            print(f"the run's job failed ({logs / (job + '.log')}): read its last lines; the same census.sh run continues it")
        else:
            print(f"no census is running: census.sh run {a.layer} continues it, answer for answer")
        line = cv.format_line("PROGRESS", a.layer, claim=claim.get("code"), done=f"{pct:.0f}%", eta=utc(eta) if eta else None)
    print(f"\n{line}\n")
    return 0


def cmd_pack(a: argparse.Namespace, kit: Dict[str, Any]) -> int:
    run = Path(a.run)
    layer = int(a.layer)
    claim = read_claim(Path(a.claim))
    if not claim.get("secret") or cv.claim_code(claim["secret"]) != claim.get("code"):
        raise SystemExit(f"{a.claim} holds no claim whose code its secret gives: was it edited? (census.sh reserve {layer})")
    if claim.get("released"):
        raise SystemExit(f"You released layer {layer} ({claim['released']}): reserve it again, if it is still free, before packing.")
    meta_p = cz.meta_path_of(run)
    if not (run.exists() and meta_p.exists()):
        raise SystemExit(f"No run of layer {layer} here ({run}): census.sh run {layer} writes it.")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    canary_dir = Path(a.canary_dir)
    canary = canary_dir / f"{run.stem}.canary.jsonl"
    home_rows, home_about = cv.home_canary(layer, kit["KIT_DESIGN"], home_dir())
    ref_about = home_about if home_about.exists() else cz.canary_about_path_of(canary)
    if not ref_about.exists():
        raise SystemExit(f"Neither the home card's canary of layer {layer} nor yours is here: census.sh canary {layer} first.")
    ref = json.loads(ref_about.read_text(encoding="utf-8"))["meta"]
    problems = cv.meta_problems(meta, ref, kit)
    if problems:
        raise SystemExit("This run is not the kit's run at its layer, and would not be taken:\n- " + "\n- ".join(problems))
    jobs = cv.expected_jobs(meta)
    if cz.jobs_digest(jobs) != meta["jobs"]:
        raise SystemExit("The run's job list is not the one its meta records: was the meta edited?")
    print(f"reading the run through ({run.stat().st_size / 1e9:.2f} GB)", flush=True)
    scan = cv.scan_run(run, jobs, layer)
    if not scan["whole"]:
        raise SystemExit(f"The run is not whole: {scan['missing']:,} answers missing, {scan['extra']} not of its job list, "
                         f"{scan['twice']} twice, {scan['other_layer']} of another layer, {scan['misshapen']} misshapen. The same "
                         f"census.sh run {layer} writes what is missing.")
    rt = meta.get("runtime") or {}
    notes: List[str] = []
    if rt.get("git_clean") is not True:
        notes.append("the checkout had changes when the run was begun (git_clean false): it may not be taken")
    if meta.get("mixed_runtime"):
        notes.append(f"the run was continued on {len(meta['mixed_runtime'])} runtimes (mixed_runtime): say why when you post it")
    own_batch: Optional[bool] = None
    if canary.exists():
        theirs = {cz.row_key(r): r for r in cz.read_rows(canary)}
        got = list(itertools.islice(cz.read_rows(run), len(theirs)))  # plan batch 0 is the first batch a run writes
        own_batch = len(got) == len(theirs) and all(theirs[cz.row_key(r)] == r for r in got)
        if not own_batch:
            notes.append("the run's first batch is not the canary taken before it (another card, driver or setting since?)")
    else:
        notes.append("no canary here: census.sh canary was not run, or its folder moved")
    code = claim["code"]
    stage = Path(a.out) / f"L{layer}-{code}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    zst = stage / f"{run.name}.zst"
    print(f"compressing the answers (zstd -19; some minutes on a few cores) into {zst.name}", flush=True)
    subprocess.run(["zstd", "-q", "-19", "-T0", "-o", str(zst), str(run)], check=True)
    subprocess.run(["zstd", "-q", "-t", str(zst)], check=True)
    shutil.copy2(meta_p, stage / meta_p.name)
    files = [zst, stage / meta_p.name]
    for p in (canary, cz.canary_about_path_of(canary), canary_dir / f"L{layer}.writer_floor.json",
              canary_dir / f"L{layer}.canary_check.json"):
        if p.exists():
            shutil.copy2(p, stage / p.name)
            files.append(stage / p.name)
        else:
            notes.append(f"{p.name} is not here, and is left out")
    check = json.loads((canary_dir / f"L{layer}.canary_check.json").read_text(encoding="utf-8")) \
        if (canary_dir / f"L{layer}.canary_check.json").exists() else {}
    manifest = {
        "what": f"layer {layer} of the census, taken with the volunteer kit {kit['KIT']} (VOLUNTEERS.md)",
        "kit": kit["KIT"], "layer": layer, "claim": code, "secret": claim["secret"], "name": claim.get("name") or "anon",
        "card": rt.get("device"), "bits": check.get("bits"), "packed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run": {"name": run.name, "answers": scan["answers"], "bytes": run.stat().st_size, "sha256": scan["sha256"]},
        "own_canary_is_first_batch": own_batch, "notes": notes,
        "files": {f.name: {"bytes": f.stat().st_size, "sha256": cz.file_sha256(f)} for f in files},
    }
    cz.write_json_atomic(stage / "manifest.json", manifest)
    tar_p = Path(a.out) / f"{run.stem}-{code}.tar"
    with tarfile.open(tar_p, "w", format=tarfile.PAX_FORMAT) as t:
        for f in [stage / "manifest.json"] + files:
            ti = t.gettarinfo(str(f), arcname=f.name)
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            with open(f, "rb") as fh:
                t.addfile(ti, fh)
    shutil.rmtree(stage)
    sha = cz.file_sha256(tar_p)
    mb = tar_p.stat().st_size / 1e6
    for n in notes:
        print(f"note: {n}")
    print(f"-> {tar_p} ({mb:.0f} MB, sha256 {sha})")
    print("Upload it where anyone can download it (a Hugging Face dataset, a torrent, a file host that keeps it a week or more),")
    print("then post this line with your link in place of PASTE_LINK_HERE:")
    print("\n" + cv.format_line("DONE", layer, claim=code, sha256=sha[:16], size=f"{mb:.0f}MB", link="PASTE_LINK_HERE") + "\n")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="the stand-in's kit settings (KIT_DRY_*)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("canary-check")
    c.add_argument("--layer", type=int, required=True)
    c.add_argument("--canary", required=True)
    c.add_argument("--out", required=True)
    s = sub.add_parser("status")
    s.add_argument("--layer", type=int, required=True)
    s.add_argument("--run", required=True)
    s.add_argument("--logs", required=True)
    s.add_argument("--claim", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--canary-dir", required=True)
    p.add_argument("--claim", required=True)
    p.add_argument("--out", required=True)
    for q in (c, s, p):
        q.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    kit = cv.read_kit(dry=a.dry_run)
    return {"canary-check": cmd_canary_check, "status": cmd_status, "pack": cmd_pack}[a.cmd](a, kit)


if __name__ == "__main__":
    sys.exit(main())

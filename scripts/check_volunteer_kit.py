#!/usr/bin/env python3
"""A CPU-only check of the census volunteer kit (VOLUNTEERS.md) end to end on the tiny random Gemma 4 of
scripts/atlas_stand_in.py, in a temporary folder: the volunteer's side (scripts/volunteer/census.sh --dry-run) and the
owner's (scripts/census_board.py, scripts/census_intake.py --dry-run), against a "home card" that is this machine:
1. the home card: a whole stand-in layer run, and the kit layers' canaries and unpushed answers (home-canaries);
2. the volunteer: check, reserve (a layer outside the kit refused), setup (setup.sh --dry-run, ~/.bashrc untouched),
   canary (the home card's bits), run (detached, waited for), status (the PROGRESS line), pack (the bundle and the DONE
   line), the bundle holding no path of this machine;
3. the board: RESERVE, a second RESERVE on a held layer refused, one claim one layer, START, PROGRESS, DONE, a RELEASE
   without the secret refused, a held claim lapsing on the clock, the renders (board.txt under 2,000 characters);
4. the intake: the bundle checked, spot-checked and judged bit for bit, accepted into a census folder, ranked, and its
   layer done on the board;
5. the refusals: a run tampered after its canary (a spot chunk's answers changed, packed again as a cheat would) is
   rejected by the spot-check; a made-up run of another card class fails the first-piece rule; a run of another card
   class (every log-probability moved by 0.0005, as another card's sums move them) passes the kit's rules for one; a
   bundle whose secret does not give its claim is refused and leaves the board as it was; a bundle holding a file a pack
   does not write is not unpacked.

Run: .venv/bin/python -I scripts/check_volunteer_kit.py   (exits 0 when every check holds; a few minutes)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend import census as cz  # noqa: E402
from backend import census_volunteer as cv  # noqa: E402

PY = sys.executable
CENSUS_SH = ROOT_DIR / "scripts" / "volunteer" / "census.sh"
BOARD_PY = ROOT_DIR / "scripts" / "census_board.py"
INTAKE_PY = ROOT_DIR / "scripts" / "census_intake.py"
LAYER = 3

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


def run(cmd: List[Any], env: Dict[str, str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], env=env, cwd=ROOT_DIR, capture_output=True, text=True, timeout=timeout)


def last_line(text: str, start: str) -> str:
    return next((ln for ln in reversed(text.splitlines()) if ln.startswith(start)), "")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="volunteer-kit-") as d:
        return run_checks(Path(d))


def run_checks(tmp: Path) -> int:
    home_atlas, vroot, canaries = tmp / "home-atlas", tmp / "volunteer", tmp / "home-canaries"
    fake_home = tmp / "user-home"
    fake_home.mkdir()
    (fake_home / ".bashrc").write_text("# the volunteer's own\n")
    env = {**os.environ, "THEODOLITE_ATLAS_DIR": str(home_atlas), "THEODOLITE_KIT_HOME_CANARIES": str(canaries),
           "HOME": str(fake_home), "CUDA_VISIBLE_DEVICES": "", "NO_COLOR": "1"}
    env.pop("THEODOLITE_POD_ROOT", None)
    kit = cv.read_kit(dry=True)
    args = ["--design", kit["KIT_DESIGN"], "--neurons", kit["KIT_NEURONS"], "--random-controls", str(kit["KIT_CONTROLS"]), "--seed",
            str(kit["KIT_SEED"]), "--batch", str(kit["KIT_BATCH"]), "--max-new", str(kit["KIT_MAX_NEW"]), "--unpacker", kit["KIT_UNPACKER"]]

    # 1. the home card
    before = len(failures)
    r = run([PY, "-I", "scripts/census_run.py", "--stand-in", "--layer", LAYER, *args, "--out-dir", tmp / "home-runs", "--no-rank"], env)
    home_run = tmp / "home-runs" / f"L{LAYER}-{kit['KIT_DESIGN']}.jsonl"
    check(r.returncode == 0 and home_run.exists(), f"the home card's stand-in run ({r.stderr[-300:]})")
    r = run([PY, "-I", INTAKE_PY, "--dry-run", "--board", "", "home-canaries", "--layers", ",".join(map(str, kit["layers"])),
             "--baselines-from", home_run], env)
    prints = json.loads((canaries / "fingerprints.json").read_text()) if (canaries / "fingerprints.json").exists() else {}
    base = json.loads((canaries / "baselines.json").read_text())["baselines"] if (canaries / "baselines.json").exists() else {}
    check(r.returncode == 0 and len(prints) == len(kit["layers"]) and len(base) == 6,
          f"home-canaries: a canary a kit layer and six unpushed answers ({r.stdout[-300:]}{r.stderr[-300:]})")
    section("the home card: a stand-in run, the kit layers' canaries, the unpushed answers", before)

    # 2. the volunteer
    before = len(failures)
    v = ["bash", CENSUS_SH, "--dry-run", "--root", vroot]
    r = run(v + ["check"], env)
    check(r.returncode == 0 and "can take a layer" in r.stdout, f"check ({r.stdout[-400:]})")
    r = run(v + ["reserve", "47"], env)
    check(r.returncode != 0 and "the kit takes layers" in r.stderr, "a layer outside the kit is refused")
    r = run(v + ["reserve", str(LAYER), "--name", "tester"], env)
    reserve = last_line(r.stdout, "CENSUS RESERVE")
    claim = {k: v_ for k, _, v_ in (ln.partition("=") for ln in (vroot / "claims" / f"L{LAYER}.claim").read_text().splitlines())} \
        if (vroot / "claims" / f"L{LAYER}.claim").exists() else {}
    code = claim.get("code", "")
    check(r.returncode == 0 and reserve and cv.claim_code(claim.get("secret", "")) == code and f"claim={code}" in reserve
          and oct((vroot / "claims" / f"L{LAYER}.claim").stat().st_mode & 0o777) == "0o600", f"reserve: the claim kept, its code posted ({r.stdout[-300:]})")
    r2 = run(v + ["reserve", str(LAYER)], env)
    check(last_line(r2.stdout, "CENSUS RESERVE") == reserve, "reserve again: the same claim, not a new one")
    r = run(v + ["setup"], env)
    check(r.returncode == 0 and "self-test" in r.stdout and (fake_home / ".bashrc").read_text() == "# the volunteer's own\n",
          f"setup: setup.sh's dry run, ~/.bashrc untouched ({r.stdout[-300:]}{r.stderr[-300:]})")
    r = run(v + ["run", str(LAYER)], env)
    check(r.returncode != 0 and "take the canary first" in r.stderr, "run before the canary is refused")
    r = run(v + ["canary", str(LAYER)], env)
    cc = vroot / "atlas_data" / "stand-in" / "census" / "canary" / f"L{LAYER}.canary_check.json"
    check(r.returncode == 0 and cc.exists() and json.loads(cc.read_text())["bits"] == "home", f"canary: the home card's bits ({r.stdout[-300:]}{r.stderr[-300:]})")
    r = run(v + ["run", str(LAYER), "--wait"], env)
    start = last_line(r.stdout, "CENSUS START")
    run_p = vroot / "atlas_data" / "stand-in" / "census" / "runs" / f"L{LAYER}-{kit['KIT_DESIGN']}.jsonl"
    check(r.returncode == 0 and "bits=home" in start and run_p.exists(), f"run: detached, waited for, the START line ({r.stdout[-300:]}{r.stderr[-300:]})")
    r = run(v + ["status", str(LAYER)], env)
    progress = last_line(r.stdout, "CENSUS PROGRESS")
    check(r.returncode == 0 and "done=100%" in progress and "the run is whole" in r.stdout, f"status: whole, the PROGRESS line ({r.stdout[-300:]})")
    r = run(v + ["pack", str(LAYER)], env)
    done = last_line(r.stdout, "CENSUS DONE")
    bundle = vroot / "outbox" / f"L{LAYER}-{kit['KIT_DESIGN']}-{code}.tar"
    check(r.returncode == 0 and bundle.exists() and f"claim={code}" in done, f"pack: the bundle and the DONE line ({r.stdout[-300:]}{r.stderr[-300:]})")
    if bundle.exists():
        with tarfile.open(bundle) as t:
            blob = b"".join(t.extractfile(m).read() for m in t.getmembers() if not m.name.endswith(".zst"))
        check(str(tmp).encode() not in blob and str(ROOT_DIR).encode() not in blob and str(Path.home()).encode() not in blob,
              "the bundle names no folder of this machine")
    section("the volunteer: check, reserve, setup, canary, run, status, pack", before)

    # 3. the board
    before = len(failures)
    board = tmp / "board.json"
    board.write_text(json.dumps({"kit": kit["KIT"], "repo": "whp199/logos-theodolite", "issue": None, "seen_comments": [], "claims": [],
                                 "layers": {str(l): {"state": "open"} for l in kit["layers"]}}))
    b = [PY, "-I", BOARD_PY, "--board", board]
    benv = {**env, "THEODOLITE_BOARD_NOW": "1793390400"}  # 2026-10-30T20:00Z
    r = run(b + ["add", ">>98765\n" + reserve], benv)
    check(r.returncode == 0 and f"L{LAYER} reserved by {code}" in r.stdout, f"a RESERVE in a quoted post ({r.stdout[:200]})")
    other = kit["layers"][0]
    r = run(b + ["add", f"CENSUS RESERVE L{LAYER} claim=0badc0de name=late"], benv)
    check(r.returncode == 1 and "refused" in r.stdout and "held by" in r.stdout, "a RESERVE on a held layer is refused")
    r = run(b + ["add", f"CENSUS RESERVE L{other} claim={code}"], benv)
    check(r.returncode == 1 and "one layer a claim" in r.stdout, "one claim, one layer")
    r = run(b + ["add", start, progress.replace("done=100%", "done=40% eta=2026-11-01T08:00Z")], benv)
    check(r.returncode == 0 and "running" in r.stdout and "held until 2026-11-02T08:00Z" in r.stdout, f"START and PROGRESS ({r.stdout[:300]})")
    r = run(b + ["add", f"CENSUS RELEASE L{LAYER} claim={code} secret=wrong"], benv)
    check(r.returncode == 1 and "without the claim's secret" in r.stdout, "a RELEASE without the secret is refused")
    r = run(b + ["add", f"CENSUS RESERVE L{other} claim=0badc0de name=lapses"], benv)
    r = run(b + ["expire"], {**env, "THEODOLITE_BOARD_NOW": str(1793390400 + 37 * 3600)})
    check("L%d 0badc0de lapsed" % other in r.stdout and f"L{LAYER} {code} lapsed" not in r.stdout,
          f"a claim past its hold lapses, a running one holds ({r.stdout[:300]})")
    r = run(b + ["add", done.replace("PASTE_LINK_HERE", "https://example.org/b.tar")], benv)
    check(r.returncode == 0 and "submitted" in r.stdout and "link given" in r.stdout, f"DONE with a link ({r.stdout[:200]})")
    r = run(b + ["render", "--out", tmp / "render", "--no-png"], benv)
    txt = (tmp / "render" / "board.txt").read_text() if (tmp / "render" / "board.txt").exists() else ""
    check(r.returncode == 0 and 0 < len(txt) <= 2000 and (tmp / "render" / "BOARD.md").exists(), "the renders, the post under 2,000 characters")
    section("the board: claims, refusals, holds, lapses, the renders", before)

    # 4. the intake
    before = len(failures)
    i = [PY, "-I", INTAKE_PY, "--dry-run", "--board", board]
    work = tmp / "work"
    r = run(i + ["check", bundle, "--work", work], env)
    chk = json.loads((work / "check.json").read_text()) if (work / "check.json").exists() else {}
    check(r.returncode == 0 and not chk.get("fails") and chk.get("bits") == "home", f"check: whole, the kit's, home bits ({r.stdout[-400:]})")
    r = run(i + ["spot", work, "--run"], env)
    check(r.returncode == 0, f"spot: the chunks written again ({r.stderr[-300:]})")
    r = run(i + ["judge", work], env)
    ver = json.loads((work / "verdict.json").read_text()) if (work / "verdict.json").exists() else {}
    check(r.returncode == 0 and ver.get("accepted") and ver["spot"]["identical"] == ver["spot"]["answers"] > 0, f"judge: bit for bit ({r.stdout[-300:]})")
    into = tmp / "census-runs"
    r = run(i + ["accept", work, "--into", into], env)
    stem = f"L{LAYER}-{kit['KIT_DESIGN']}"
    check(r.returncode == 0 and all((into / f"{stem}{s}").exists() for s in (".jsonl", ".meta.json", ".provenance.json", ".units.jsonl")),
          f"accept: the run, its meta, provenance and ranking ({r.stdout[-300:]}{r.stderr[-300:]})")
    bj = json.loads(board.read_text())
    check(bj["layers"][str(LAYER)]["state"] == "done", "the layer done on the board")
    r = run(i + ["accept", work, "--into", into], env)
    check(r.returncode != 0 and "is there already" in r.stderr, "a layer's run is not replaced")
    section("the intake: checked, spot-checked bit for bit, accepted, ranked, on the board", before)

    # 5. the refusals
    before = len(failures)
    rows = [json.loads(x) for x in run_p.read_text().splitlines()]
    canary = run_p.parent.parent / "canary" / f"{stem}.canary.jsonl"
    canary_rows = [json.loads(x) for x in canary.read_text().splitlines()]

    def repack(label: str, change_run: Any, change_canary: Any = None) -> Path:
        """The volunteer's run (and canary) changed as a cheat or another card would change them, packed again."""
        root = tmp / f"v-{label}"
        shutil.copytree(vroot, root, ignore=shutil.ignore_patterns("outbox", "logs"))
        envf = root / "env.sh"  # setup.sh wrote the folder's own path into it
        envf.write_text(envf.read_text().replace(str(vroot), str(root)))
        rp = root / run_p.relative_to(vroot)
        rp.write_text("".join(json.dumps(change_run(dict(r_), k)) + "\n" for k, r_ in enumerate(rows)))
        if change_canary:
            cp = root / canary.relative_to(vroot)
            cp.write_text("".join(json.dumps(change_canary(dict(r_), k)) + "\n" for k, r_ in enumerate(canary_rows)))
        rr = run(["bash", CENSUS_SH, "--dry-run", "--root", root, "pack", str(LAYER)], env)
        check(rr.returncode == 0, f"{label}: packed again ({rr.stderr[-300:]})")
        return root / "outbox" / bundle.name

    def through(label: str, tar: Path) -> Dict[str, Any]:
        w = tmp / f"work-{label}"
        nb = tmp / f"board-{label}.json"
        nb.write_text(json.dumps({"kit": kit["KIT"], "repo": "x/y", "issue": None, "seen_comments": [], "claims": [],
                                  "layers": {str(l): {"state": "open"} for l in kit["layers"]}}))
        run([PY, "-I", BOARD_PY, "--board", nb, "add", reserve], benv)
        ii = [PY, "-I", INTAKE_PY, "--dry-run", "--board", nb]
        c1 = run(ii + ["check", tar, "--work", w], env)
        out = {"check": c1, "board": nb}
        if c1.returncode == 0:
            run(ii + ["spot", w, "--run"], env)
            out["judge"] = run(ii + ["judge", w], env)
            out["verdict"] = json.loads((w / "verdict.json").read_text()) if (w / "verdict.json").exists() else {}
        out["claim"] = next((c for c in json.loads(nb.read_text())["claims"] if c["code"] == code), {})
        return out

    jobs = cv.expected_jobs(json.loads(cz.meta_path_of(run_p).read_text()))
    plan = cz.plan_batches(jobs, kit["KIT_BATCH"], "12 x batch")
    in_chunk = {cz.row_key(jobs[j]): c for c, _, js in plan for j in js}

    def tamper(r_: Dict[str, Any], k: int) -> Dict[str, Any]:  # every pushed answer's first piece moved: a cheat's made-up rows
        if r_["kind"] != "baseline" and r_["token_ids"]:
            r_["token_ids"] = [(r_["token_ids"][0] + 1) % 500] + r_["token_ids"][1:]
        return r_

    # a run whose canary is honest and whose later chunks are made up: caught where the spot draws a made-up chunk
    t = through("tampered", repack("tampered", lambda r_, k: tamper(r_, k) if in_chunk.get(cz.row_key(r_), 0) > 0 else r_))
    check(t["check"].returncode == 0 and t.get("verdict", {}).get("accepted") is False and t["claim"].get("state") == "rejected",
          f"a run made up after its canary is rejected by the spot-check ({t['check'].stdout[-300:]})")
    # made up throughout, its canary too (so not the home card's bits): fails the rules for another card class
    t = through("madeup", repack("madeup", tamper, tamper))
    check(t["check"].returncode == 0 and t.get("verdict", {}).get("accepted") is False
          and any("first word-piece" in w for w in t["verdict"].get("why_not", [])), f"a made-up run fails the first-piece rule ({t.get('verdict')})")

    def nudge(r_: Dict[str, Any], k: int) -> Dict[str, Any]:  # another card's sums: each log-probability moved a little
        r_["logprob"] = [round(x - 0.0005, 4) if x < -0.01 else x for x in r_["logprob"]]
        return r_

    t = through("othercard", repack("othercard", nudge, nudge))
    chk2 = json.loads((tmp / "work-othercard" / "check.json").read_text()) if (tmp / "work-othercard" / "check.json").exists() else {}
    check(chk2.get("bits") == "own" and t.get("verdict", {}).get("accepted") is True,
          f"another card class within the kit's rules is accepted ({chk2.get('bits')}, {t.get('verdict', {}).get('why_not')})")
    # a bundle whose secret does not give its claim
    bad = tmp / "badsecret"
    bad.mkdir()
    with tarfile.open(bundle) as tf:
        tf.extractall(bad, filter="data")
    man = json.loads((bad / "manifest.json").read_text())
    man["secret"] = "0" * 32
    (bad / "manifest.json").write_text(json.dumps(man))
    bad_tar = tmp / "bad" / bundle.name
    bad_tar.parent.mkdir()
    with tarfile.open(bad_tar, "w") as tf:
        for p in sorted(bad.iterdir()):
            tf.add(p, arcname=p.name)
    t = through("badsecret", bad_tar)
    check(t["check"].returncode != 0 and "secret does not give" in t["check"].stdout and t["claim"].get("state") == "reserved",
          f"a bundle without its claim's secret is refused, the board untouched ({t['check'].stdout[-200:]}, {t['claim'].get('state')})")
    # a file a pack does not write
    odd = tmp / "odd" / bundle.name
    odd.parent.mkdir()
    shutil.copy(bundle, odd)
    with tarfile.open(odd, "a") as tf:
        extra = tmp / "run.sh"
        extra.write_text("echo hi\n")
        tf.add(extra, arcname="run.sh")
    r = run([PY, "-I", INTAKE_PY, "--dry-run", "--board", "", "check", odd, "--work", tmp / "work-odd"], env)
    check(r.returncode != 0 and "which a pack does not write" in r.stderr and not (tmp / "work-odd" / "bundle" / "run.sh").exists(),
          "a bundle holding a file a pack does not write is not unpacked")
    section("the refusals: made up after its canary, made up throughout, another card class taken, a wrong secret, a stray file", before)

    print(f"\n{passed} checks passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

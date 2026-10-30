#!/usr/bin/env python3
"""A CPU-only check of what a rented card runs (scripts/pod/*.sh, scripts/export_constraints.sh, scripts/gpu_log.py), each
script's --dry-run end to end on the tiny random Gemma 4 of scripts/atlas_stand_in.py, in temporary volume roots:
1. every shell script parses (bash -n), and passes shellcheck where it is installed; each begins `set -euo pipefail`,
   answers --help and refuses an option it does not know; the token setup.sh is given is used in one place alone; a
   command run through `scrubbed` sees no variable naming a token, key or secret, but the one it keeps;
2. export_constraints.sh: the venv's exact versions into a dated file, its header naming the commit, the Python,
   torch's CUDA and the torch backend;
3. setup.sh --dry-run: env.sh puts HF_HOME, uv's caches, THEODOLITE_ATLAS_DIR and XDG_DATA_HOME on the volume with
   HF_HUB_OFFLINE=1; the download (a fake `hf` that refuses a request without a token) is tried without one first,
   then with the one given, which is in no file under the root and in nothing printed; without one the setup stops in
   plain words; the venv is checked against the constraints offline (and a pin it does not hold refused); the last line
   is the self-test's: the card, the versions, the unpacker's test and the runtime class; the self-test refused while
   another job holds the card;
4. census_shard.sh --dry-run: two shards of one stand-in layer, detached (the job a session's leader, SIGHUP ignored,
   as it records itself), each with its log and sentinels, shard 1 with its canary; census_run.py handed no key or
   token (a Python that records its environment); a job refused while another holds the card; merged by
   census_tools.py merge (the canary identical), ranked, and the curves read; a job stopped by its time limit
   (CENSUS_TIMEOUT, only whole answers kept) and finished by the same command; the stop hook run only when asked, with
   the pod's own environment;
5. sync.sh --dry-run (against rsync, or a stand-in for it where rsync is not installed, which reads its paths as rsync
   does: the last of several the destination, never two remote): a periodic pull (rows with --append-verify, the rest
   whole), then the final pull, checked by the pod's listing (every source's files in it) and what each meta names; the
   check (pod_tool.py listing) names a file missing, one cut short, one a meta names and no one has, and jobs not ended;
   with the stand-in rsync, the pod's own command lines too (over "SSH", the pod's layout);
6. pilot.sh --dry-run: refused while another job holds the card; every step on the stand-in into pilot.json: the
   self-test, the steps per unpacker and batch, the writer's floor (at 32, 64 and the pilot's own batch), a calib_rel
   run and its curves, one steer-lens layer, a canary set against a home canary taken alone (identical, and a changed
   one differing), the zstd ratio or why not, the projections, "proceed" and "stop"; every Python it runs handed no
   key or token;
7. gpu_log.py: approvals, starts and stops summed by hand (GPU-hours over pods and their cards, cost at each pod's
   rate), deviations named, a pod still running counted to now, an overspent approval exiting 1, and the refusals.

Run: .venv/bin/python -I scripts/check_pod_scripts.py   (exits 0 when every check holds)
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
POD = ROOT_DIR / "scripts" / "pod"
SCRIPTS = [POD / "setup.sh", POD / "census_shard.sh", POD / "sync.sh", POD / "pilot.sh", ROOT_DIR / "scripts" / "export_constraints.sh"]
TOKEN = "hf_check_pod_scripts_sentinel_7f3a"  # what a check's token looks like; it must surface nowhere

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


FAKE_HF = """import os, sys
# a stand-in for `hf download`: a request without a token is refused, as a gated repository's would be
tok = os.environ.get("HF_TOKEN", "")
print("fake hf", sys.argv[1:], "with a token" if tok else "without a token",
      "implicit token off" if os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN") else "")
sys.exit(0 if (tok == os.environ.get("FAKE_HF_WANTS", tok)) and (tok or os.environ.get("FAKE_HF_OPEN")) else 1)
"""

FAKE_RSYNC = '''#!{python}
"""A stand-in for rsync where it is not installed (scripts/check_pod_scripts.py): local copies only, with the options
sync.sh uses: -a, -r, --relative (a source's /./), --append-verify, --include/--exclude (first match, on a name),
--list-only and --no-human-readable; -e is taken and left unused, and a path's "user@host:" is read as this machine.
As rsync (main.c, start_client), the last of several paths is the destination, never listed or read, and a remote
source with a remote destination is refused; a lone source is listed. Each call's arguments are logged."""
import fnmatch, os, re, shutil, sys, time
args = sys.argv[1:]
if os.environ.get("FAKE_RSYNC_LOG"):
    with open(os.environ["FAKE_RSYNC_LOG"], "a") as f:
        f.write(" ".join(args) + "\\n")
rules, flags, paths, i = [], set(), [], 0
while i < len(args):
    a = args[i]
    if a == "-e":
        i += 2
        continue
    if a.startswith("--include="):
        rules.append((True, a.split("=", 1)[1]))
    elif a.startswith("--exclude="):
        rules.append((False, a.split("=", 1)[1]))
    elif a.startswith("--"):
        flags.add(a)
    elif a.startswith("-") and len(a) > 1:
        flags.update("-" + c for c in a[1:])
    else:
        paths.append(a)
    i += 1
def host(p):  # rsync's check_for_hostspec, near enough: a colon before any slash
    m = re.match(r"^([^/:]+):(.*)$", p)
    return (m.group(1), m.group(2)) if m else (None, p)

if not paths:
    print("rsync: no source given", file=sys.stderr)
    sys.exit(1)
srcs, dest = (paths, None) if len(paths) == 1 else (paths[:-1], paths[-1])
listing = "--list-only" in flags or dest is None
if dest is not None and host(srcs[0])[0] and host(dest)[0]:
    print("The source and destination cannot both be remote.", file=sys.stderr)
    print("rsync error: syntax or usage error (code 1)", file=sys.stderr)
    sys.exit(1)
srcs = [host(s)[1] for s in srcs]

def wanted(name, is_dir):
    for keep, pat in rules:
        if pat.endswith("/"):
            if is_dir and fnmatch.fnmatch(name, pat[:-1]):
                return keep
            continue
        if fnmatch.fnmatch(name, pat):
            return keep
    return True

rc, entries = 0, []
for s in srcs:
    base, rel = s.split("/./", 1) if "/./" in s else (os.path.dirname(s.rstrip("/")), os.path.basename(s.rstrip("/")))
    rel = rel.rstrip("/")
    top = os.path.join(base, rel)
    if not os.path.exists(top):
        print(f'rsync: link_stat "{{top}}" failed: No such file or directory (2)', file=sys.stderr)
        rc = 23
        continue
    parts = rel.split("/")
    entries += [("/".join(parts[:k]), os.path.join(base, *parts[:k]), True) for k in range(1, len(parts) + 1)]
    for dirpath, dirnames, filenames in os.walk(top):
        dirnames[:] = sorted(d for d in dirnames if wanted(d, True))
        r = os.path.relpath(dirpath, base)
        entries += [(os.path.join(r, d), os.path.join(dirpath, d), True) for d in dirnames]
        entries += [(os.path.join(r, f), os.path.join(dirpath, f), False) for f in sorted(filenames) if wanted(f, False)]
seen = set()
for rel, ab, is_dir in entries:
    if rel in seen:
        continue
    seen.add(rel)
    st = os.stat(ab)
    if listing:
        when = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(st.st_mtime))
        print(f"{{'d' if is_dir else '-'}}rw-r--r-- {{st.st_size:>14}} {{when}} {{rel}}")
        continue
    out = os.path.join(dest, rel)
    if is_dir:
        os.makedirs(out, exist_ok=True)
        continue
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if "--append-verify" in flags and os.path.exists(out):
        have = os.path.getsize(out)
        if have >= st.st_size:
            continue
        data, old = open(ab, "rb").read(), open(out, "rb").read()
        if data[:have] == old:
            with open(out, "ab") as f:
                f.write(data[have:])
        else:
            shutil.copyfile(ab, out)
    elif not (os.path.exists(out) and os.path.getsize(out) == st.st_size and int(os.path.getmtime(out)) == int(st.st_mtime)):
        shutil.copyfile(ab, out)
    os.utime(out, (st.st_atime, st.st_mtime))
sys.exit(rc)
'''


def run(cmd: List[str], env: Dict[str, str], timeout: int = 900, cwd: Any = None) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], env=env, cwd=str(cwd or ROOT_DIR), capture_output=True, text=True, timeout=timeout)


def wait_for(paths: List[Path], timeout: float = 600.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if any(p.exists() for p in paths):
            return True
        time.sleep(0.5)
    return False


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="pod-check-"))
    try:
        return run_checks(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_checks(tmp: Path) -> int:
    py = sys.executable
    fakebin = tmp / "bin"
    fakebin.mkdir()
    (tmp / "tmp").mkdir()
    env = {**os.environ, "POD_PYTHON": py, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "2"), "TMPDIR": str(tmp / "tmp"),
           "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"}
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "THEODOLITE_POD_ROOT", "THEODOLITE_ATLAS_DIR", "THEODOLITE_CENSUS_DIR"):
        env.pop(k, None)
    have_rsync = shutil.which("rsync") is not None
    if not have_rsync:
        (fakebin / "rsync").write_text(FAKE_RSYNC.format(python=py))
        (fakebin / "rsync").chmod(0o755)
        env["PATH"] = f"{fakebin}:{env['PATH']}"
    (tmp / "fake_hf.py").write_text(FAKE_HF)
    # the check's Python, where a test asks what a script hands its Python: each run's environment and arguments, then the venv's
    wrap = fakebin / "python-env"
    wrap.write_text("#!/usr/bin/env bash\n"
                    '{ env; echo "-- argv: $*"; } >>"${CHECK_ENV_DUMP:-/dev/null}"\n'
                    f"exec {py} \"$@\"\n")
    wrap.chmod(0o755)
    secret_line = re.compile(r"^(RUNPOD_API_KEY|HF_TOKEN|HUGGING_FACE_HUB_TOKEN)=", re.M)

    # ---------------------------------------------------------------------------------------- 1. the scripts as text
    before = len(failures)
    for s in SCRIPTS:
        r = run(["bash", "-n", s], env)
        text = s.read_text()
        check(r.returncode == 0, f"bash -n {s.name}: {r.stderr[-200:]}")
        check(text.startswith("#!/usr/bin/env bash\n") and "\nset -euo pipefail\n" in text and os.access(s, os.X_OK),
              f"{s.name}: a bash script, executable, under set -euo pipefail")
        h = run([s, "--help"], env)
        check(h.returncode == 0 and "Options:" in h.stdout and "--dry-run" in h.stdout, f"{s.name} --help ({h.stderr[-200:]})")
        u = run([s, "--no-such-option"], env)
        check(u.returncode != 0 and "unknown option" in u.stderr, f"{s.name} refuses an option it does not know ({u.stderr[-200:]})")
    check(run(["bash", "-n", POD / "common.sh"], env).returncode == 0, "bash -n common.sh")
    if shutil.which("shellcheck"):
        for s in SCRIPTS + [POD / "common.sh"]:
            r = run(["shellcheck", "-x", "-s", "bash", s], env)
            check(r.returncode == 0, f"shellcheck {s.name}: {r.stdout[-600:]}")
    else:
        print("  --    shellcheck is not installed here: skipped")
    for tool in (POD / "pod_tool.py", ROOT_DIR / "scripts" / "gpu_log.py", ROOT_DIR / "scripts" / "verify_atlas_writer.py"):
        r = run([py, "-I", tool, "--help"], env)
        check(r.returncode == 0 and "usage" in r.stdout, f"{tool.name} --help")
    setup_text = (POD / "setup.sh").read_text()
    uses = [ln.strip() for ln in setup_text.splitlines() if "_hf_token" in ln]
    check(len(uses) == 4 and all(any(p in u_ for p in ("_hf_token=${HF_TOKEN", '[ -n "$_hf_token" ]', '[ -z "$_hf_token" ]',
                                                         'export HF_TOKEN="$_hf_token"')) for u_ in uses),
          f"setup.sh: the token is read once, tested, and exported to the download's retry alone ({uses})")
    check(not re.search(r"(echo|printf|say|note)[^\n]*\$\{?(HF_TOKEN|_hf_token|HUGGING_FACE_HUB_TOKEN|RUNPOD_API_KEY)", "\n".join(
        s.read_text() for s in SCRIPTS + [POD / "common.sh"])), "no script echoes a token's value")
    r = run(["bash", "-c", f'. {POD / "common.sh"}; scrubbed --keep KEEP_TOKEN env'],
            {**env, "HF_TOKEN": TOKEN, "RUNPOD_API_KEY": "rk", "AWS_SECRET_ACCESS_KEY": "a", "MY_PASSWORD": "p", "KEEP_TOKEN": "k",
             "PLAIN_VAR": "x"})
    names = {ln.split("=", 1)[0] for ln in r.stdout.splitlines()}
    check(r.returncode == 0 and not names & {"HF_TOKEN", "RUNPOD_API_KEY", "AWS_SECRET_ACCESS_KEY", "MY_PASSWORD"}
          and {"KEEP_TOKEN", "PLAIN_VAR", "PATH"} <= names, "scrubbed: no token, key, secret or password, but the one kept")
    section("every script parses, answers --help, refuses unknown options, and keeps tokens out of what it runs", before)

    # ----------------------------------------------------------------------------------- 2. export_constraints.sh
    before = len(failures)
    cons = tmp / "home.txt"
    r = run([ROOT_DIR / "scripts" / "export_constraints.sh", "--python", py, "--out", cons], env)
    lines = cons.read_text().splitlines() if cons.exists() else []
    head = [ln for ln in lines if ln.startswith("#")]
    pins = {ln.split("==")[0]: ln.split("==")[1] for ln in lines if "==" in ln and not ln.startswith("#")}
    import importlib.metadata as md

    import torch

    want_tb = "cpu" if not torch.version.cuda else "cu" + torch.version.cuda.replace(".", "")
    check(r.returncode == 0 and pins.get("torch") == md.version("torch").split("+")[0] and pins.get("transformers") == md.version("transformers")
          and pins.get("interp-engine") == md.version("interp-engine") and pins.get("compressed-tensors") == md.version("compressed-tensors"),
          f"export_constraints.sh: the venv's exact versions ({r.stdout[-300:]}{r.stderr[-300:]})")
    check(any(ln.startswith("# written: ") and "at " in ln for ln in head) and any(ln.startswith("# python: ") for ln in head)
          and f"# torch-backend: {want_tb}" in head, f"its header: the date, the commit, the Python, the torch backend ({head})")
    held = ROOT_DIR / "scripts" / "pod" / "constraints"  # the checkout's own (the volunteer kit's pins), left as they are
    before_dry = sorted((p.name, p.stat().st_mtime_ns) for p in held.glob("*")) if held.exists() else None
    r = run([ROOT_DIR / "scripts" / "export_constraints.sh", "--python", py, "--dry-run"], env)
    m = re.search(r"-> (\S+home-\d{4}-\d\d-\d\d\.txt)", r.stdout)
    after_dry = sorted((p.name, p.stat().st_mtime_ns) for p in held.glob("*")) if held.exists() else None
    check(r.returncode == 0 and m is not None and Path(m.group(1)).exists() and str(tmp / "tmp") in m.group(1)
          and before_dry == after_dry, f"--dry-run writes a dated file in a temporary folder, the checkout untouched ({r.stdout[-200:]})")
    section("export_constraints.sh: the home venv's exact versions, dated, with the torch backend", before)

    # ---------------------------------------------------------------------------------------------- 3. setup.sh
    before = len(failures)
    r1 = tmp / "pod1"
    fake_hf = f"{py} {tmp / 'fake_hf.py'}"
    r = run([POD / "setup.sh", "--dry-run", "--root", r1, "--constraints", cons], {**env, "HF_TOKEN": TOKEN, "POD_HF_CMD": fake_hf})
    out = r.stdout + r.stderr
    envsh = (r1 / "env.sh").read_text() if (r1 / "env.sh").exists() else ""
    want = {"HF_HOME": r1 / "hf", "UV_CACHE_DIR": r1 / ".uv-cache", "UV_PYTHON_INSTALL_DIR": r1 / ".uv-python",
            "THEODOLITE_ATLAS_DIR": r1 / "atlas_data", "XDG_DATA_HOME": r1 / "share"}
    sourced = run(["bash", "-c", f". {r1 / 'env.sh'}; env"], env).stdout.splitlines()
    got = dict(ln.split("=", 1) for ln in sourced if "=" in ln)
    check(r.returncode == 0 and all(got.get(k) == str(v) for k, v in want.items()) and got.get("HF_HUB_OFFLINE") == "1"
          and f"{r1}/bin" in got.get("PATH", ""), f"setup.sh --dry-run: env.sh puts every cache and folder on the volume, offline ({out[-500:]})")
    dl = (r1 / "logs" / "download.log").read_text() if (r1 / "logs" / "download.log").exists() else ""
    check(dl.count("without a token\n") == 2 and dl.count("again with the one given (not shown)") == 2 and dl.count("implicit token off") == 2
          and "52f3f65bc7a02d555763bc923bd1d9094898219d" in dl and "e166deb5ad540932100ef803c8c9bcc2a5453694" in dl
          and "gemma-4-31B_jpp_lens.pt" in dl and (r1 / "logs" / "download.done").exists(),
          f"the downloads pinned to the README's snapshots, each tried without a token first, then with it ({dl[-400:]})")
    leaked = [str(p.relative_to(r1)) for p in r1.rglob("*") if p.is_file() and TOKEN.encode() in p.read_bytes()]
    check(not leaked and TOKEN not in out, f"the token is in no file under the root and in nothing printed ({leaked})")
    last = [ln for ln in r.stdout.splitlines() if ln.strip()][-1] if r.stdout.strip() else ""
    st = json.loads((r1 / "logs" / "selftest.json").read_text()) if (r1 / "logs" / "selftest.json").exists() else {}
    check(last.startswith("self-test: card cpu |") and f"torch {torch.__version__}" in last and "unpacker lut" in last
          and f"class {st.get('class', {}).get('id')}" in last and st.get("stand_in") is True and st.get("runtime", {}).get("git"),
          f"the last line is the self-test's: card, versions, unpacker, class ({last})")
    check("the venv holds every version" in out, "the venv checked against the constraints, offline")
    bad = tmp / "bad.txt"
    bad.write_text(cons.read_text().replace(f"torch=={pins.get('torch')}", "torch==0.0.1"))
    r = run([POD / "setup.sh", "--dry-run", "--root", tmp / "pod-bad", "--constraints", bad, "--no-download", "--no-selftest"], env)
    check(r.returncode != 0 and "does not hold the versions" in r.stderr, f"a pin the venv does not hold is refused ({r.stderr[-300:]})")
    r = run([POD / "setup.sh", "--dry-run", "--root", tmp / "pod-notoken", "--no-selftest"], {**env, "POD_HF_CMD": fake_hf})
    dl2 = (tmp / "pod-notoken" / "logs" / "download.log").read_text() if (tmp / "pod-notoken" / "logs" / "download.log").exists() else ""
    check(r.returncode != 0 and "the download failed" in r.stderr and "none was given" in dl2 and (tmp / "pod-notoken" / "logs" / "download.failed").exists(),
          f"refused without a token and none given: setup stops in plain words ({r.stderr[-200:]})")
    r = run([POD / "setup.sh", "--dry-run", "--root", tmp / "pod-open", "--no-selftest"],
            {**env, "HF_TOKEN": TOKEN, "POD_HF_CMD": fake_hf, "FAKE_HF_OPEN": "1", "FAKE_HF_WANTS": ""})
    dl3 = (tmp / "pod-open" / "logs" / "download.log").read_text() if (tmp / "pod-open" / "logs" / "download.log").exists() else ""
    check(r.returncode == 0 and "again with the one given" not in dl3 and dl3.count("without a token\n") == 2,
          f"an open repository: downloaded without the token, which is never used ({dl3[-300:]})")
    r_lock = tmp / "pod-locked"
    (r_lock / "logs").mkdir(parents=True)
    with open(r_lock / "logs" / "gpu.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r = run([POD / "setup.sh", "--dry-run", "--root", r_lock, "--no-download"], env)
        fcntl.flock(held, fcntl.LOCK_UN)
    check(r.returncode != 0 and "another GPU job holds this pod's card" in r.stderr and not (r_lock / "logs" / "selftest.json").exists(),
          f"setup.sh's self-test refused while another job holds the card ({r.stderr[-200:]})")
    section("setup.sh --dry-run: the env file on the volume, downloads without a token first, the token nowhere, the self-test line", before)

    # ------------------------------------------------------------------------------------------ 4. census_shard.sh
    before = len(failures)
    logs = r1 / "logs"
    runs = r1 / "atlas_data" / "stand-in" / "census" / "runs"
    small = ["--", "--neurons", "0-7", "--random-controls", "2", "--max-new", "8", "--batch", "4"]
    shard = [POD / "census_shard.sh", "--dry-run", "--root", r1, "--layer", "3", "--projected-hours", "0.1"]
    with open(logs / "gpu.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r = run(shard + ["--shard", "0/2"] + small, env)
        fcntl.flock(held, fcntl.LOCK_UN)
    check(r.returncode != 0 and "another GPU job holds this pod's card" in r.stderr, f"a job refused while another holds the card ({r.stderr[-200:]})")
    sdump = tmp / "shard_env.txt"
    r = run(shard + ["--shard", "0/2", "--stop-cmd", f"env > {r1 / 'stop_env.txt'}"] + small,
            {**env, "RUNPOD_API_KEY": "rk_check", "HF_TOKEN": "hf_check_shard", "POD_PYTHON": str(wrap), "CHECK_ENV_DUMP": str(sdump)})
    detached = r.returncode == 0 and "detached" in r.stdout
    done0 = wait_for([logs / "L3-main_rel.s0of2.done", logs / "L3-main_rel.s0of2.failed"])
    r = run(shard + ["--shard", "1/2", "--wait"] + small, env)
    check(detached and done0 and (logs / "L3-main_rel.s0of2.done").exists() and r.returncode == 0 and (logs / "L3-main_rel.s1of2.done").exists(),
          f"two shards detached and done ({r.stdout[-300:]}{r.stderr[-300:]})")
    log0 = (logs / "L3-main_rel.s0of2.log").read_text()
    job0 = (logs / "L3-main_rel.s0of2.job.sh").read_text()
    check(all(w in log0 for w in ("CENSUS_JOB", "CENSUS_START", "CENSUS_DONE rc=0", "time limit 450s (0.1 h projected x 1.25)", "--shard 0/2", "--stand-in"))
          and (logs / "L3-main_rel.s0of2.started").exists() and "rk_check" not in job0 and "hf_check_shard" not in job0,
          "each job's log, its start and end sentinels, its time limit, and its script with no secret in it")
    # "<time> pid P session S sigign HEX", as the job wrote it of itself
    st0 = (logs / "L3-main_rel.s0of2.started").read_text().split()
    st0 = dict(zip(st0[1::2], st0[2::2]))
    check(st0.get("pid") is not None and st0.get("session") == st0.get("pid") and st0.get("session") != str(os.getsid(0))
          and bool(int(st0.get("sigign") or "0", 16) & 1),
          f"the job detached: the leader of a session of its own (setsid), SIGHUP ignored (nohup) ({st0})")
    dumped = sdump.read_text() if sdump.exists() else ""
    check("census_run.py" in dumped and "--shard 0/2" in dumped and not secret_line.search(dumped) and "rk_check" not in dumped
          and "hf_check_shard" not in dumped, "census_run.py, as the job runs it, is handed no key or token (the stop hook alone has the pod's)")
    stop_env = (r1 / "stop_env.txt").read_text() if (r1 / "stop_env.txt").exists() else ""
    check("RUNPOD_API_KEY=rk_check" in stop_env and "stopping the pod" in log0
          and "stopping the pod" not in (logs / "L3-main_rel.s1of2.log").read_text(),
          "the stop hook runs at the end only when asked, with the pod's own environment (where the provider puts its key)")
    shards = [runs / "L3-main_rel.s0of2.jsonl", runs / "L3-main_rel.s1of2.jsonl"]
    check(all(p.exists() for p in shards) and (runs / "L3-main_rel.s1of2.canary.jsonl").exists()
          and not (runs / "L3-main_rel.s0of2.canary.jsonl").exists(), "the shards' files in the stand-in's census folder, shard 1 with its canary")
    tenv = {**env, "THEODOLITE_ATLAS_DIR": str(r1 / "atlas_data")}
    merged = runs / "L3-main_rel.jsonl"
    r = run([py, "-I", ROOT_DIR / "scripts" / "census_tools.py", "--stand-in", "merge"] + shards + [runs / "L3-main_rel.s1of2.canary.jsonl", "--out", merged], tenv)
    n_rows = sum(1 for _ in open(merged)) if merged.exists() else 0
    check(r.returncode == 0 and "canary: identical on 1 of 1 shard" in r.stdout and n_rows == (8 + 2) * 10 + 4,
          f"census_tools.py merge: the shards joined whole, the canary identical ({r.stdout[-300:]}{r.stderr[-300:]})")
    r = run([py, "-I", ROOT_DIR / "scripts" / "census_tools.py", "--stand-in", "rank", merged], tenv)
    check(r.returncode == 0 and (runs / "L3-main_rel.units.jsonl").exists() and "8 neurons, 2 random controls" in r.stdout, f"ranked ({r.stderr[-200:]})")
    r = run([py, "-I", ROOT_DIR / "scripts" / "census_tools.py", "--stand-in", "curves", merged, "--json", tmp / "curves.json"], tenv)
    cj = json.loads((tmp / "curves.json").read_text()) if (tmp / "curves.json").exists() else {}
    check(r.returncode == 0 and cj.get("layer") == 3 and len(cj.get("curves", [])) > 2, f"the curves read ({r.stderr[-200:]})")
    # a job stopped by its time limit, between steps, and finished by the same command
    tshard = [POD / "census_shard.sh", "--dry-run", "--root", r1, "--layer", "3", "--run", "L3-t", "--wait"]
    targs = ["--", "--neurons", "all", "--random-controls", "2", "--max-new", "32", "--batch", "8"]  # some 30 s here: past 12 s
    r = run(tshard + ["--projected-hours", "0.0025"] + targs, env)
    tlog = (logs / "L3-t.log").read_text()
    trun = runs / "L3-t.jsonl"
    whole = trun.read_text().endswith("\n") if trun.exists() and trun.stat().st_size else True
    rows_t = [json.loads(x) for x in trun.read_text().splitlines()] if trun.exists() else []
    check(r.returncode != 0 and (logs / "L3-t.timeout").exists() and "CENSUS_TIMEOUT rc=124" in tlog and "time limit 12s" in tlog and whole
          and len(rows_t) < (128 + 2) * 10 + 4, f"a job past its limit is stopped between steps, only whole answers kept ({len(rows_t)} rows; {r.stdout[-300:]})")
    r = run(tshard + ["--projected-hours", "1"] + targs, env)
    rows_t = [json.loads(x) for x in trun.read_text().splitlines()] if trun.exists() else []
    check(r.returncode == 0 and (logs / "L3-t.done").exists() and not (logs / "L3-t.timeout").exists()
          and len(rows_t) == (128 + 2) * 10 + 4 and len({(x["kind"], x["neuron"], x["prompt"], x["dose"]) for x in rows_t}) == len(rows_t),
          f"the same command finishes it, every answer once ({len(rows_t)} rows)")
    r = run([POD / "census_shard.sh", "--dry-run", "--root", r1, "--layer", "3"] + small, env)
    check(r.returncode != 0 and "--projected-hours" in r.stderr, "a job with no projection and no --no-timeout is refused")
    r = run([POD / "census_shard.sh", "--dry-run", "--root", r1, "--layer", "3", "--no-timeout", "--", "--shard", "0/2"], env)
    check(r.returncode != 0 and "give --shard to census_shard.sh itself" in r.stderr, "census_run.py's own --shard is refused after --")
    section("census_shard.sh --dry-run: two shards detached with logs and sentinels, merged and ranked; the lock, the limit, the stop hook", before)

    # ------------------------------------------------------------------------------------------------- 5. sync.sh
    before = len(failures)
    home = tmp / "home"
    rlog = tmp / "rsync.log"
    senv = {**env, "FAKE_RSYNC_LOG": str(rlog)}
    r = run([POD / "sync.sh", "--dry-run", "--root", r1, "--to", home, "--once"], senv)
    hruns = home / "atlas_data" / "stand-in" / "census" / "runs"
    check(r.returncode == 0 and (hruns / "L3-main_rel.jsonl").exists() and (hruns / "L3-main_rel.s1of2.canary.jsonl").exists()
          and (hruns / "L3-main_rel.meta.json").exists() and (home / "logs" / "L3-main_rel.s0of2.done").exists()
          and (home / "logs" / "L3-main_rel.s0of2.log").exists(), f"sync.sh --once: runs, metas, canaries and logs pulled ({r.stderr[-300:]})")
    if not have_rsync:
        calls = rlog.read_text().splitlines()
        check(len(calls) == 2 and "--append-verify" in calls[0] and "--include=*.jsonl" in calls[0] and "--exclude=*" in calls[0]
              and "--exclude=*.jsonl" in calls[1] and "--append-verify" not in calls[1] and all("/./" in c for c in calls),
              f"a round: the rows with --append-verify, then the rest whole ({calls})")
    # the pod goes on writing: a row appended to a run, a steer-lens layer built; the final pull takes both, checked
    with open(runs / "L3-t.jsonl", "a") as f:
        f.write(json.dumps({"kind": "baseline", "neuron": -1, "prompt": "extra", "dose": 0.0}) + "\n")
    (r1 / "atlas_data" / "stand-in" / "index" / "steer_lens.d" / "L3.check.txt").write_text("a steer-lens layer, as the check writes one\n")
    r = run([POD / "sync.sh", "--dry-run", "--root", r1, "--to", home, "--final"], senv)
    lj = json.loads((home / "listing.json").read_text()) if (home / "listing.json").exists() else {}
    check(r.returncode == 0 and lj.get("ok") is True and lj.get("metas", 0) >= 4 and lj.get("running") == []
          and (hruns / "L3-t.jsonl").read_bytes() == (runs / "L3-t.jsonl").read_bytes() and (home / "remote.list").exists(),
          f"sync.sh --final: everything pulled, checked against the pod's listing and every meta ({lj.get('says')}; {r.stderr[-300:]})")
    listed = (home / "remote.list").read_text().splitlines() if (home / "remote.list").exists() else []
    check(all(any(ln.endswith(" " + p) for ln in listed) for p in ("atlas_data/stand-in/census/runs/L3-main_rel.jsonl", "logs/L3-t.log",
                                                                   "atlas_data/stand-in/index/steer_lens.d/L3.check.txt")),
          f"the pod's listing names the files under each source, the last one named too ({len(listed)} lines)")
    if not have_rsync:
        last_call = rlog.read_text().splitlines()[-1]
        check("--list-only" in last_call and last_call.split()[-1] == f"{home.resolve()}/",
              f"the final check lists the pod's files with rsync --list-only, the copy given as the destination ({last_call[-200:]})")
    listing = [py, "-I", POD / "pod_tool.py", "listing", "--local", home, "--remote-list", home / "remote.list"]
    (hruns / "L3-main_rel.s1of2.canary.jsonl").unlink()
    with open(hruns / "L3-t.jsonl", "r+b") as f:
        f.truncate(100)
    r = run(listing, env)
    check(r.returncode != 0 and "missing: atlas_data/stand-in/census/runs/L3-main_rel.s1of2.canary.jsonl" in r.stdout
          and "shorter here: atlas_data/stand-in/census/runs/L3-t.jsonl" in r.stdout
          and "named by a meta, absent: atlas_data/stand-in/census/runs/L3-main_rel.s1of2.canary.jsonl" in r.stdout,
          f"the check names a canary missing, a run cut short, and what a meta names ({r.stdout[-500:]})")
    # on the pod: a meta naming a shard no one has, and a job begun and not ended
    meta_text = (runs / "L3-main_rel.meta.json").read_text()
    mm = json.loads(meta_text)
    (runs / "L3-main_rel.meta.json").write_text(json.dumps({**mm, "shards": mm["shards"] + ["L3-main_rel.s2of2.jsonl"]}))
    (logs / "L3-x.started").write_text("now\n")
    r = run([POD / "sync.sh", "--dry-run", "--root", r1, "--to", home, "--final"], senv)
    lj = json.loads((home / "listing.json").read_text())
    check(r.returncode != 0 and lj["ok"] is False and lj["missing"] == [] and lj["shorter_here"] == []
          and lj["named_absent"] == ["atlas_data/stand-in/census/runs/L3-main_rel.s2of2.jsonl (named by atlas_data/stand-in/census/runs/L3-main_rel.meta.json)"]
          and lj["running"] == ["L3-x"] and "NOT whole" in r.stderr,
          f"a final pull restores what was cut, and says what no one has and which jobs have not ended ({lj.get('says')})")
    (runs / "L3-main_rel.meta.json").write_text(meta_text)
    (logs / "L3-x.started").unlink()
    (logs / "L3-x.done").write_text("now\n")
    shutil.rmtree(home)
    r = run([POD / "sync.sh", "--dry-run", "--root", r1, "--to", home, "--until-done", "--every", "0.01"], senv)
    check(r.returncode == 0 and "every job on the pod has ended" in r.stderr and json.loads((home / "listing.json").read_text())["ok"] is True,
          f"--until-done: pulls until every job has ended, then the final pull ({r.stderr[-300:]})")
    if not have_rsync:
        # the pod's own command lines, over "SSH" in the pod's layout, the stand-in rsync reading check@pod: as this machine
        podlike = tmp / "podlike"
        (podlike / "atlas_data").mkdir(parents=True)
        for link, to in (("atlas_data/census", "atlas_data/stand-in/census"), ("atlas_data/index", "atlas_data/stand-in/index"),
                         ("logs", "logs"), ("pilot", "pilot")):
            (podlike / link).symlink_to(r1 / to)
        home2 = tmp / "home-ssh"
        n_calls = len(rlog.read_text().splitlines())
        r = run([POD / "sync.sh", "--from", "check@pod", "--port", "22022", "--remote-root", podlike, "--to", home2, "--final"], senv)
        calls = rlog.read_text().splitlines()[n_calls:]
        lj2 = json.loads((home2 / "listing.json").read_text()) if (home2 / "listing.json").exists() else {}
        check(r.returncode == 0 and lj2.get("ok") is True and lj2.get("metas", 0) >= 4 and len(calls) == 2
              and all("ssh -p 22022" in c and c.split()[-1] == f"{home2.resolve()}/" and f"check@pod:{podlike}/./atlas_data/index/steer_lens.d/" in c
                      for c in calls) and (home2 / "atlas_data" / "index" / "steer_lens.d" / "L3.check.txt").exists(),
              f"sync.sh --final from a pod: its sources, then the copy as the destination, never two remote paths ({lj2.get('says')}; {r.stderr[-300:]})")
    r = run([POD / "sync.sh", "--to", home, "--final"], senv)
    check(r.returncode != 0 and "--from" in r.stderr, "sync.sh without a pod to pull from is refused")
    section("sync.sh --dry-run: periodic pulls, the final pull checked by listing and by what each meta names", before)

    # ------------------------------------------------------------------------------------------------ 6. pilot.sh
    before = len(failures)
    hc = tmp / "home-canary"
    r = run([py, "-I", ROOT_DIR / "scripts" / "census_run.py", "--stand-in", "--design", "main_rel", "--layer", "3", "--neurons", "all", "--random-controls", "4",
             "--batch", "4", "--max-new", "8", "--canary-only", "--out-dir", hc], {**env, "THEODOLITE_ATLAS_DIR": str(tmp / "home-atlas")})
    home_can = hc / "L3-main_rel.canary.jsonl"
    check(r.returncode == 0 and home_can.exists() and (hc / "L3-main_rel.canary.json").exists(), f"a home canary taken alone ({r.stderr[-200:]})")
    r2 = tmp / "pod2"
    (r2 / "logs").mkdir(parents=True)
    with open(r2 / "logs" / "gpu.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r = run([POD / "pilot.sh", "--dry-run", "--root", r2], env)
        fcntl.flock(held, fcntl.LOCK_UN)
    check(r.returncode != 0 and "another GPU job holds this pod's card" in r.stderr and not any((r2 / "pilot").glob("*")),
          f"the pilot refused while another job holds the card, before it writes anything ({r.stderr[-200:]})")
    pdump = tmp / "pilot_env.txt"
    r = run([POD / "pilot.sh", "--dry-run", "--root", r2, "--home-canary", home_can, "--approved-hours", "100", "--jobs",
             "whole_layer:2,steer_lens:1"], {**env, "HF_TOKEN": TOKEN, "RUNPOD_API_KEY": "rk_check", "POD_PYTHON": str(wrap),
                                            "CHECK_ENV_DUMP": str(pdump)}, timeout=1800)
    pj_path = next(iter(sorted((r2 / "pilot").glob("*/pilot.json"))), None)
    pj = json.loads(pj_path.read_text()) if pj_path else {}
    pdir = pj_path.parent if pj_path else tmp
    check(r.returncode == 0 and pj.get("card") == "cpu" and (pj.get("class") or {}).get("id") and (pdir / "pilot.log").exists()
          and (pdir / "out" / "steps.txt").exists(), f"pilot.sh --dry-run: every step done, one log, one summary ({r.stdout[-600:]}{r.stderr[-300:]})")
    rows = pj.get("steps") or []
    check(len(rows) == 9 and {x["unpacker"] for x in rows} == {"reference", "lut", "compiled"} and {x["batch"] for x in rows} == {2, 4, 8}
          and all(x["fits"] and x["step_s"] and x["steps"] == 8 for x in rows) and pj.get("chosen") in rows,
          f"the decode step for each unpacker at each batch, the fastest that fit chosen ({len(rows)} rows)")
    pdumped = pdump.read_text() if pdump.exists() else ""
    leaked = [str(p.relative_to(r2)) for p in r2.rglob("*") if p.is_file() and TOKEN.encode() in p.read_bytes()]
    check(pdumped.count("-- argv: ") >= 10 and all(t in pdumped for t in ("pod_tool.py selftest", "pod_tool.py canary-args", "verify_atlas_writer.py",
                                                                        "census_run.py", "atlas_build.py steer", "census_tools.py"))
          and not secret_line.search(pdumped) and TOKEN not in pdumped and "rk_check" not in pdumped and not leaked and TOKEN not in r.stdout + r.stderr,
          f"every Python the pilot runs is handed no key or token, and the token is in nothing it writes ({pdumped.count('-- argv: ')} runs; {leaked})")
    wf = pj.get("writer_floor") or {}
    pb = (pj.get("chosen") or {}).get("batch") or 0
    check(wf.get("alone_twice_same") is True and wf.get("census_floor") == {"32": None, "64": None, **({str(pb): None} if pb >= 3 else {})}
          and (pdir / "atlas_writer_floor.json").exists(),
          f"the writer's floor at census batches of 32 and 64 and the pilot's own {pb} (none parts on the stand-in, in float32 on the CPU) ({wf})")
    cr = pj.get("calib_rel") or {}
    check(cr.get("answers") == (4 + 2) * 36 + 4 and cr.get("batch") == pj["chosen"]["batch"] and cr.get("wall_s") is not None
          and (pj.get("curves") or {}).get("window") is not None, f"a calib_rel run timed at the chosen batch, its dose window read ({cr})")
    sl = pj.get("steer_lens") or {}
    check(sl.get("layer") == 3 and sl.get("seconds") is not None and (pdir / "index" / "steer_lens.d" / "L3.npz").exists(),
          f"one steer-lens layer timed, in the pilot's own index ({sl})")
    cn = pj.get("canary") or {}
    check(cn.get("same") is True and cn.get("says") == "identical: 4 answers, every field" and cn.get("settings_differ") == [],
          f"the canary written with the home canary's settings, identical ({cn.get('says')})")
    ours = next(iter((pdir / "canary").glob("*.canary.jsonl")), None)
    bent = tmp / "bent" / "L3-main_rel.canary.jsonl"
    bent.parent.mkdir()
    brows = [json.loads(x) for x in home_can.read_text().splitlines()]
    brows[0]["token_ids"][3] += 1
    bent.write_text("".join(json.dumps(x) + "\n" for x in brows))
    shutil.copy(hc / "L3-main_rel.canary.json", bent.parent / "L3-main_rel.canary.json")
    r = run([py, "-I", ROOT_DIR / "scripts" / "census_tools.py", "canary", ours, bent], env)
    check(r.returncode == 0 and "differ in 1 of 4 answers" in r.stdout and "at piece 3" in r.stdout, f"a changed home canary differs, where it parts ({r.stdout[-300:]})")
    zs = pj.get("zstd") or {}
    check((zs.get("ratio") is None and "not installed" in zs.get("says", "")) if not shutil.which("zstd") else (zs.get("ratio") or 0) > 1,
          f"the zstd ratio, or why it is not measured ({zs})")
    pr = pj.get("projections") or {}
    best = pj.get("chosen") or {}
    want_h = ((128 + 256) * 10 + 4) / best.get("batch", 1) * (8 + 1) * best.get("step_s", 0) / 3600
    check(abs(pr.get("whole_layer", {}).get("gpu_hours", -1) - round(want_h, 3)) < 1e-9 and pr["whole_layer"]["with_margin"] == round(want_h * 1.25, 3)
          and "steer_lens" in pr and "depth_curve" in pr, f"the projections by census_run.py's formula, x 1.25 ({pr.get('whole_layer')})")
    check((pj.get("budget") or {}).get("decision") == "proceed", f"within the approval: proceed ({pj.get('budget')})")
    r = run([py, "-I", POD / "pod_tool.py", "summary", pdir, "--approved-hours", "0.0000001", "--jobs", "whole_layer:2"], env)
    check(r.returncode == 0 and json.loads((pdir / "pilot.json").read_text())["budget"]["decision"] == "stop" and "-> stop" in r.stdout,
          "over the approval: stop")
    r = run([py, "-I", POD / "pod_tool.py", "summary", pdir, "--approved-hours", "5", "--jobs", "every_layer:1"], env)
    check(json.loads((pdir / "pilot.json").read_text())["budget"]["decision"] == "stop", "a job it cannot project: stop")
    section("pilot.sh --dry-run: self-test, steps, floor, calib_rel, steer layer, canary, zstd, projections and the decision", before)

    # ---------------------------------------------------------------------------------------------- 7. gpu_log.py
    before = len(failures)
    glog = tmp / "gpu_runs.jsonl"
    g = [py, "-I", ROOT_DIR / "scripts" / "gpu_log.py", "--log", glog]
    r = run(g + ["start", "--pod", "p1", "--provider", "RunPod Secure", "--card", "H200", "--rate", "5.29", "--job", "x", "--approval", "A1"], env)
    check(r.returncode != 0 and "no approval A1" in r.stderr and not glog.exists(), "a pod naming no approval on record is refused")
    r = run(g + ["approve", "A1", "--gpu-hours", "12", "--usd", "64", "--card", "H200", "--provider", "RunPod Secure", "--rate", "5.29",
                 "--note", "the check's", "--at", "2026-10-28T09:00:00Z"], env)
    check(r.returncode == 0 and "records approval A1" in r.stdout, f"an approval written down ({r.stderr[-200:]})")
    for cmd in (["start", "--pod", "p1", "--provider", "RunPod Secure", "--card", "H200", "--rate", "5.29", "--job", "L45 s0/2",
                 "--approval", "A1", "--at", "2026-10-28T10:00:00Z"],
                ["start", "--pod", "p2", "--provider", "Vast.ai", "--card", "H200", "--gpus", "2", "--rate", "5.50", "--job", "L45 s1/2",
                 "--approval", "A1", "--at", "2026-10-28T10:00:00Z"],
                ["stop", "--pod", "p1", "--at", "2026-10-28T13:30:00Z"], ["stop", "--pod", "p2", "--at", "2026-10-28T11:00:00Z"]):
        r = run(g + cmd, env)
        check(r.returncode == 0, f"gpu_log.py {' '.join(cmd[:3])}: {r.stderr[-200:]}")
    r = run(g + ["sum", "A1", "--pods", "--json", tmp / "sums.json"], env)
    sums = json.loads((tmp / "sums.json").read_text())[0] if (tmp / "sums.json").exists() else {}
    # p1: 3.5 h x 1 card at $5.29; p2: 1 h x 2 cards at $5.50
    check(r.returncode == 0 and sums.get("gpu_hours") == 5.5 and sums.get("usd") == round(3.5 * 5.29 + 2 * 5.50, 2) and not sums.get("over")
          and "used 5.50 of the 12 GPU-hours ($29.52 of $64.00) the owner approved" in r.stdout,
          f"summed by hand: 5.5 GPU-hours, ${3.5 * 5.29 + 11:.2f} ({r.stdout[-400:]})")
    check(sums.get("deviations") == ["pod p2 ran on provider Vast.ai, not the approved RunPod Secure", "pod p2 at $5.50/h, above the approved $5.29/h"],
          f"the deviations named ({sums.get('deviations')})")
    r = run(g + ["stop", "--pod", "p1"], env)
    check(r.returncode != 0 and "not running" in r.stderr, "a pod not running cannot stop")
    r = run(g + ["start", "--pod", "p3", "--provider", "RunPod Secure", "--card", "H200", "--rate", "5.29", "--job", "long", "--approval", "A1",
                 "--at", "2026-10-28T00:00:00Z"], env)
    r_ = run(g + ["start", "--pod", "p3", "--provider", "RunPod Secure", "--card", "H200", "--rate", "5.29", "--job", "again", "--approval", "A1"], env)
    check(r.returncode == 0 and r_.returncode != 0 and "running since" in r_.stderr, "a pod already running cannot start again")
    r = run(g + ["stop", "--pod", "p3", "--at", "2026-10-27T00:00:00Z"], env)
    check(r.returncode != 0 and "cannot stop before" in r.stderr, "a pod cannot stop before it started")
    r = run(g + ["sum", "A1", "--json", tmp / "sums.json"], env)
    s2 = json.loads((tmp / "sums.json").read_text())[0]
    check(r.returncode == 1 and s2["running"] == ["p3"] and s2["over"] and "OVER the approval" in r.stdout and "1 pod still running" in r.stdout
          and s2["gpu_hours"] > 5.5 + 12, f"a pod still running counted to now, the approval overspent: exit 1 ({r.stdout[-300:]})")
    events = [json.loads(x) for x in glog.read_text().splitlines()]
    check([e["event"] for e in events] == ["approved", "started", "started", "stopped", "stopped", "started"]
          and all("at" in e and "host" in e for e in events) and events[0]["by"] == "the owner", "the log: one line an event, each with its time and machine")
    section("gpu_log.py: approvals, pods and their sums by hand, deviations, refusals, a pod running, an approval overspent", before)

    print(f"\n{passed} checks passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

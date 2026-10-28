#!/usr/bin/env python3
"""Submit experiment code to labd.  Usage:
    python labclient.py script.py [--label NAME] [--out result.json]
    echo 'result = 1+1' | python labclient.py - --label quick
Prints stdout from the daemon, then the JSON `result`. Exit code 1 on error.
With --out X, the daemon ALSO writes the full response (stdout/error/result) to X.full.json server-side,
so if this client is killed by a timeout while queued/running, the work is not lost: wait for X.full.json.
It sends the token labd wrote when it started (labd.py's docstring says where; --token-file to read another copy).
GET http://127.0.0.1:8765/health shows {busy, waiting, running}.
"""
import argparse, json, os, sys, urllib.error, urllib.request
from pathlib import Path


def token_path(port):
    """Where labd's launch on `port` keeps its token: as labd.token_path."""
    if os.environ.get("LABD_TOKEN_FILE"):
        return Path(os.environ["LABD_TOKEN_FILE"]).expanduser()
    base = os.environ.get("XDG_RUNTIME_DIR") or str(Path.home() / ".cache")
    return Path(base) / "theodolite" / f"labd-{port}.token"


p = argparse.ArgumentParser()
p.add_argument("script")
p.add_argument("--label", default="")
p.add_argument("--out", default=None)
p.add_argument("--port", default="8765")
p.add_argument("--token-file", default=None, help="labd's token file (default: where labd writes it for --port)")
p.add_argument("--timeout", type=float, default=3600)
a = p.parse_args()
tok_file = Path(a.token_file).expanduser() if a.token_file else token_path(a.port)
try:
    token = tok_file.read_text().strip()
except OSError:
    sys.exit(f"labclient: no labd token at {tok_file}. labd writes one each time it starts: is it running on port "
             f"{a.port}, started by you? (A copy of the file elsewhere: --token-file.)")
code = sys.stdin.read() if a.script == "-" else open(a.script).read()
req = urllib.request.Request(f"http://127.0.0.1:{a.port}/exec",
                             data=json.dumps({"code": code, "label": a.label,
                                                 "out": (os.path.abspath(a.out) + ".full.json") if a.out else ""}).encode(),
                             headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
try:
    r = json.loads(urllib.request.urlopen(req, timeout=a.timeout).read())
except urllib.error.HTTPError as e:
    if e.code not in (401, 403):
        raise
    detail = e.read().decode("utf-8", "replace")
    try:
        detail = json.loads(detail).get("detail", detail)
    except ValueError:
        pass
    sys.exit(f"labclient: labd refused the job ({e.code}): {detail}"
             + (f" ({tok_file} may be an earlier launch's.)" if e.code == 401 else ""))
if r["stdout"]:
    print(r["stdout"])
if r["error"]:
    print("ERROR:\n" + r["error"], file=sys.stderr)
print(f"[elapsed {r['elapsed_s']}s]", file=sys.stderr)
if a.out:
    json.dump(r["result"], open(a.out, "w"), indent=1)
    print(f"[result -> {a.out}]", file=sys.stderr)
elif r["result"] is not None:
    print(json.dumps(r["result"], indent=1)[:20000])
sys.exit(0 if r["ok"] else 1)

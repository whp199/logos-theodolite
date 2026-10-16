#!/usr/bin/env python3
"""Submit experiment code to labd.  Usage:
    python labclient.py script.py [--label NAME] [--out result.json]
    echo 'result = 1+1' | python labclient.py - --label quick
Prints stdout from the daemon, then the JSON `result`. Exit code 1 on error.
With --out X, the daemon ALSO writes the full response (stdout/error/result) to X.full.json server-side,
so if this client is killed by a timeout while queued/running, the work is not lost: wait for X.full.json.
GET http://127.0.0.1:8765/health shows {busy, waiting, running}.
"""
import argparse, json, os, sys, urllib.request

p = argparse.ArgumentParser()
p.add_argument("script")
p.add_argument("--label", default="")
p.add_argument("--out", default=None)
p.add_argument("--port", default="8765")
p.add_argument("--timeout", type=float, default=3600)
a = p.parse_args()
code = sys.stdin.read() if a.script == "-" else open(a.script).read()
req = urllib.request.Request(f"http://127.0.0.1:{a.port}/exec",
                             data=json.dumps({"code": code, "label": a.label,
                                                 "out": (os.path.abspath(a.out) + ".full.json") if a.out else ""}).encode(),
                             headers={"Content-Type": "application/json"})
r = json.loads(urllib.request.urlopen(req, timeout=a.timeout).read())
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

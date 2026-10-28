#!/usr/bin/env python3
"""The rented cards' log, after R3's experiments/results/R3_gpu_runs.jsonl: each approval, and every pod's start and stop,
appended to one JSON-lines file, and the GPU-hours and their cost summed against the approval they count against.

    .venv/bin/python -I scripts/gpu_log.py approve A1 --gpu-hours 12 --usd 64 --card "H200 141GB" \
        --provider "RunPod Secure" --rate 5.29 --note "the pilot, the steer lens, the depth curve at --max-new 64"
    .venv/bin/python -I scripts/gpu_log.py start --pod abc123 --provider "RunPod Secure" --card "H200 141GB" --rate 5.29 \
        --job "L45 main_rel, shard 0 of 4" --approval A1
    .venv/bin/python -I scripts/gpu_log.py stop --pod abc123
    .venv/bin/python -I scripts/gpu_log.py sum A1 --pods    # used 5.54 of the 12 GPU-hours ($29.31 of $64.00) ...
    .venv/bin/python -I scripts/gpu_log.py sum              # every approval

The budget rule (the plan for rented cards, after R3_PREREG D30 and A15): the owner approves ahead, in GPU-hours on a
named card and provider with a dollar ceiling at the rate read that day; a pilot on the same card class projects the
job, x 1.25, against it (scripts/pod/pilot.sh); every pod is logged; the write-up says "used X of the Y GPU-hours ($a
of $b) the owner approved". This script records and sums, and approves nothing of itself: `approve` writes down what
the owner decided, and `start` refuses a pod that names no approval on record. GPU-hours are summed over pods and their
cards (a pod of two cards for 3 h is 6), each from its start to its stop, and cost at the rate each pod was started at;
a pod not yet stopped is counted to now and said to be running. A pod on another card or provider than its approval
names, or at a higher rate, is said: a deviation, to be recorded as R3 recorded D30-D36. Times are UTC (--at takes
one read off the provider's console instead of now). `sum` exits 1 when an approval is overspent, in hours or dollars.

The log is experiments/results/gpu_runs.jsonl (--log). No key or token is read or written.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
LOG = ROOT_DIR / "experiments" / "results" / "gpu_runs.jsonl"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def when(s: str) -> dt.datetime:
    """A time as the log keeps it (UTC, "2026-10-28T12:00:00Z"); one without a zone is read as UTC."""
    t = dt.datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    return t.replace(tzinfo=dt.timezone.utc) if t.tzinfo is None else t.astimezone(dt.timezone.utc)


def read_log(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except ValueError as exc:
                    raise SystemExit(f"{path}:{i} is not a JSON line ({exc}): mend it by hand, the log is append-only.")
    return out


def append(path: Path, event: Dict[str, Any]) -> Dict[str, Any]:
    """One line appended to the log, with its time (given, else now, UTC) and the machine that wrote it."""
    event = {**event, "at": event.get("at") or now(), "host": platform.node()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event


def pods_of(events: List[Dict[str, Any]], at: Optional[dt.datetime] = None) -> List[Dict[str, Any]]:
    """Every pod interval in the log, start to stop (or to `at`, else now, when it has not stopped), in the order begun:
    the start's fields with `started`, `stopped` (None while running), `hours` (wall), `gpu_hours` and `usd`."""
    end_now = at or dt.datetime.now(dt.timezone.utc)
    open_: Dict[str, Dict[str, Any]] = {}
    out: List[Dict[str, Any]] = []
    for e in events:
        if e.get("event") == "started":
            rec = {**{k: v for k, v in e.items() if k not in ("event", "at", "host")}, "started": e["at"], "stopped": None}
            open_[str(e["pod"])] = rec
            out.append(rec)
        elif e.get("event") == "stopped" and str(e.get("pod")) in open_:
            open_.pop(str(e["pod"]))["stopped"] = e["at"]
    for rec in out:
        stop = when(rec["stopped"]) if rec["stopped"] else end_now
        hours = max(0.0, (stop - when(rec["started"])).total_seconds() / 3600.0)
        rec["hours"] = hours
        rec["gpu_hours"] = hours * int(rec.get("gpus") or 1)
        rec["usd"] = rec["gpu_hours"] * float(rec.get("rate_usd_h") or 0.0)
    return out


def approvals_of(events: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """{approval id: its approval}, the last recorded for an id standing (an amended approval is a new line)."""
    return {str(e["approval"]): e for e in events if e.get("event") == "approved"}


def summary(events: List[Dict[str, Any]], approval: str, at: Optional[dt.datetime] = None) -> Dict[str, Any]:
    """What was used against one approval: GPU-hours and dollars against those approved, the pods, which still run,
    and every deviation from what the approval names (another card or provider, a higher rate), with `says`, the
    line a write-up quotes."""
    ap = approvals_of(events).get(approval)
    if ap is None:
        raise SystemExit(f"There is no approval {approval} in the log.")
    pods = [p for p in pods_of(events, at) if str(p.get("approval")) == approval]
    used_h = sum(p["gpu_hours"] for p in pods)
    used_usd = sum(p["usd"] for p in pods)
    running = [p for p in pods if p["stopped"] is None]
    deviations = []
    for p in pods:
        for k, what in (("card", "card"), ("provider", "provider")):
            if ap.get(k) and p.get(k) and str(p[k]) != str(ap[k]):
                deviations.append(f"pod {p['pod']} ran on {what} {p[k]}, not the approved {ap[k]}")
        if ap.get("rate_usd_h") is not None and float(p.get("rate_usd_h") or 0.0) > float(ap["rate_usd_h"]) + 1e-9:
            deviations.append(f"pod {p['pod']} at ${float(p['rate_usd_h']):.2f}/h, above the approved ${float(ap['rate_usd_h']):.2f}/h")
    over = used_h > float(ap["gpu_hours"]) + 1e-9 or (ap.get("usd") is not None and used_usd > float(ap["usd"]) + 1e-9)
    says = (f"used {used_h:.2f} of the {float(ap['gpu_hours']):g} GPU-hours"
            + (f" (${used_usd:.2f} of ${float(ap['usd']):.2f})" if ap.get("usd") is not None else f" (${used_usd:.2f})")
            + f" the owner approved ({approval}: {ap.get('card')} on {ap.get('provider')}"
            + (f" at <= ${float(ap['rate_usd_h']):.2f}/h" if ap.get("rate_usd_h") is not None else "") + ")"
            + (f"; {len(running)} pod{'s' if len(running) != 1 else ''} still running, counted to now" if running else ""))
    return {"approval": approval, "approved": ap, "gpu_hours": round(used_h, 4), "usd": round(used_usd, 2), "pods": pods,
            "running": [p["pod"] for p in running], "deviations": deviations, "over": over, "says": says}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--log", default=str(LOG), help="the JSON-lines log (default experiments/results/gpu_runs.jsonl)")
    sub = ap.add_subparsers(dest="what", required=True)
    a_ = sub.add_parser("approve", help="write down an approval the owner gave")
    a_.add_argument("approval", help="its id, e.g. A1")
    a_.add_argument("--gpu-hours", type=float, required=True, help="GPU-hours approved, summed over pods")
    a_.add_argument("--usd", type=float, default=None, help="the dollar ceiling at the rate read that day")
    a_.add_argument("--card", required=True, help='the card class, e.g. "H200 141GB"')
    a_.add_argument("--provider", required=True, help='e.g. "RunPod Secure"')
    a_.add_argument("--rate", type=float, default=None, help="the rate read that day, $ per GPU-hour")
    a_.add_argument("--note", default="", help="what it is for, in the owner's words")
    a_.add_argument("--at", default="", help="when it was given (UTC; default now)")
    s_ = sub.add_parser("start", help="a pod started")
    s_.add_argument("--pod", required=True, help="the provider's pod id")
    s_.add_argument("--provider", required=True)
    s_.add_argument("--card", required=True)
    s_.add_argument("--gpus", type=int, default=1, help="cards in the pod (default 1)")
    s_.add_argument("--rate", type=float, required=True, help="$ per GPU-hour, as the provider bills it that day")
    s_.add_argument("--job", required=True, help="what it runs")
    s_.add_argument("--approval", required=True, help="the approval it counts against")
    s_.add_argument("--at", default="", help="when it started (UTC; default now)")
    t_ = sub.add_parser("stop", help="a pod stopped (or was removed)")
    t_.add_argument("--pod", required=True)
    t_.add_argument("--note", default="")
    t_.add_argument("--at", default="", help="when it stopped (UTC; default now)")
    u_ = sub.add_parser("sum", help="GPU-hours and cost against an approval (every approval when none is named)")
    u_.add_argument("approval", nargs="?", default="")
    u_.add_argument("--pods", action="store_true", help="list each pod too")
    u_.add_argument("--json", default="", help="also write the sums there")
    a = ap.parse_args(argv)
    log = Path(a.log)
    events = read_log(log)
    for t in ([a.at] if getattr(a, "at", "") else []):
        try:
            when(t)
        except ValueError:
            raise SystemExit(f"--at {t!r} is not a time (2026-10-28T12:00:00Z, UTC).")

    if a.what == "approve":
        if a.gpu_hours <= 0:
            raise SystemExit("An approval is of some GPU-hours, more than 0.")
        e = append(log, {"event": "approved", "approval": a.approval, "gpu_hours": a.gpu_hours, "usd": a.usd, "card": a.card,
                         "provider": a.provider, "rate_usd_h": a.rate, "note": a.note, "by": "the owner",
                         "at": when(a.at).strftime("%Y-%m-%dT%H:%M:%SZ") if a.at else None})
        was = "amends" if sum(1 for x in events if x.get("event") == "approved" and str(x.get("approval")) == a.approval) else "records"
        print(f"{log.name} {was} approval {a.approval}: {a.gpu_hours:g} GPU-hours of {a.card} on {a.provider}"
              + (f", at most ${a.usd:.2f}" if a.usd is not None else "") + f" ({e['at']})")
    elif a.what == "start":
        if a.approval not in approvals_of(events):
            raise SystemExit(f"There is no approval {a.approval} in {log}: write down the owner's approval first "
                             "(gpu_log.py approve), or name the one this pod counts against.")
        running = [p for p in pods_of(events) if str(p["pod"]) == a.pod and p["stopped"] is None]
        if running:
            raise SystemExit(f"Pod {a.pod} has been running since {running[0]['started']} by the log: stop it first (gpu_log.py stop).")
        if a.gpus < 1 or a.rate < 0:
            raise SystemExit("A pod has one card or more, at a rate of 0 or more.")
        e = append(log, {"event": "started", "pod": a.pod, "provider": a.provider, "card": a.card, "gpus": a.gpus,
                         "rate_usd_h": a.rate, "job": a.job, "approval": a.approval,
                         "at": when(a.at).strftime("%Y-%m-%dT%H:%M:%SZ") if a.at else None})
        s = summary(read_log(log), a.approval)
        print(f"pod {a.pod} started {e['at']} ({a.card} on {a.provider}, ${a.rate:.2f}/h, {a.job}); {a.approval}: {s['says']}")
        for d in s["deviations"]:
            print(f"deviation: {d}")
    elif a.what == "stop":
        running = [p for p in pods_of(events) if str(p["pod"]) == a.pod and p["stopped"] is None]
        if not running:
            raise SystemExit(f"Pod {a.pod} is not running by the log (no start since its last stop).")
        at = when(a.at).strftime("%Y-%m-%dT%H:%M:%SZ") if a.at else now()
        if when(at) < when(running[0]["started"]):
            raise SystemExit(f"Pod {a.pod} started {running[0]['started']}: it cannot stop before ({at}).")
        append(log, {"event": "stopped", "pod": a.pod, "note": a.note, "at": at})
        s = summary(read_log(log), str(running[0]["approval"]))
        mine = next(p for p in s["pods"] if p["pod"] == a.pod and p["started"] == running[0]["started"])
        print(f"pod {a.pod} stopped {at}: {mine['gpu_hours']:.2f} GPU-hours, ${mine['usd']:.2f}; {s['approval']}: {s['says']}")
    else:
        ids = [a.approval] if a.approval else list(approvals_of(events))
        if not ids:
            print(f"No approval in {log}.")
            return 0
        sums = [summary(events, i) for i in ids]
        for s in sums:
            print(f"{s['approval']}: {s['says']}" + ("  OVER the approval" if s["over"] else ""))
            for d in s["deviations"]:
                print(f"  deviation: {d}")
            if a.pods:
                for p in s["pods"]:
                    print(f"  pod {p['pod']}: {p['card']} on {p['provider']} x{p.get('gpus') or 1}, {p['started']} to "
                          f"{p['stopped'] or 'now (running)'}, {p['gpu_hours']:.2f} GPU-hours at ${float(p['rate_usd_h']):.2f}/h = "
                          f"${p['usd']:.2f}; {p['job']}")
        unassigned = [p for p in pods_of(events) if str(p.get("approval")) not in approvals_of(events)]
        for p in unassigned:
            print(f"pod {p['pod']} counts against approval {p.get('approval')}, which is not in the log")
        if a.json:
            out = Path(a.json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(sums, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"-> {out}")
        return 1 if any(s["over"] for s in sums) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())

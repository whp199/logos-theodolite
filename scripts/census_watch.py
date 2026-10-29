#!/usr/bin/env python3
"""Watch a census run in a terminal: a progress bar over its answers, how fast they come, when it should finish, and
whether a census_run.py is still writing it. Run it in a terminal of your own beside a long census (it reads files
only, never the GPU, and stops nothing):

    .venv/bin/python scripts/census_watch.py                    # the census folder's newest run
    .venv/bin/python scripts/census_watch.py L45-break_rel      # a run by name (or give its .jsonl path)
    .venv/bin/python scripts/census_watch.py L45-break_rel --once   # one reading, then exit

It counts the answers already written (the lines of the run's .jsonl; a half-written last line is not counted)
against the run's total (its meta's `jobs`; for a shard, the share of the run's planned chunks it was given), and
reads only what has been added since its last look, so a run of several gigabytes costs nothing to watch. The rate is
taken over the last --window minutes it has watched; until a whole window, since it began watching, and in its first
minute (answers land a batch at a time) since the run began, by its meta's start; the time left and the finish come
from that rate, and the line says which rate it is. A writer is looked for among this machine's processes (Linux: /proc), as a census_run.py
naming the run's layer; with none, and no answer for a while, it says the run has stopped and that the same command
continues it row for row. It exits when every answer is written, or on Ctrl-C. Layers count from 0, as the commands
do (page = layer + 1). Colour only on a terminal, and never with NO_COLOR set.
"""

from __future__ import annotations

import argparse
import calendar
import collections
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend import census as cz  # noqa: E402

STALLED_S = 180  # no new answer for this long: say so


def find_run(spec: str) -> Path:
    """The run's .jsonl: a path, a name in the census folder, or (none given) the folder's newest run."""
    if spec:
        p = Path(spec)
        if p.suffix != ".jsonl":
            p = cz.census_dir() / f"{spec}.jsonl" if not p.exists() else p
        if not p.exists():
            raise SystemExit(f"No census run {spec!r}: give its name in {cz.census_dir()} or its .jsonl path.")
        return p
    runs = [p for p in cz.census_dir().glob("*.jsonl")
            if not p.name.endswith((".units.jsonl", ".canary.jsonl")) and cz.meta_path_of(p).exists()]
    if not runs:
        raise SystemExit(f"No census run in {cz.census_dir()}: give its name or its .jsonl path.")
    return max(runs, key=lambda p: p.stat().st_mtime)


def run_total(meta: Dict[str, Any]) -> int:
    """How many answers the run (or the shard) writes in all."""
    n = int((meta.get("jobs") or {}).get("n") or 0)
    sh = meta.get("shard")
    if not sh:
        return n
    size = int(meta["batch"]) * cz.PLANS[meta.get("plan") or cz.PLAN]
    c0, c1 = (int(x) for x in sh["chunks"])
    return max(0, min(c1 * size, n) - c0 * size)


class Lines:
    """The complete lines of a growing file, counted from where the last look stopped."""

    def __init__(self, path: Path) -> None:
        self.path, self.offset, self.count = path, 0, 0

    def update(self) -> int:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return self.count
        if size < self.offset:  # rewritten from the start (a merge, a fresh run): count again
            self.offset, self.count = 0, 0
        step = 1 << 22
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            while True:
                chunk = f.read(step)
                if not chunk:
                    break
                last = chunk.rfind(b"\n")
                if last < 0:
                    if len(chunk) < step:
                        break  # a half-written last line: counted once it is whole
                    step *= 2  # one line longer than a read: read more at once
                    f.seek(self.offset)
                    continue
                self.count += chunk.count(b"\n", 0, last + 1)
                self.offset += last + 1
                f.seek(self.offset)
        return self.count


def writers(run: Path, layer: int) -> List[int]:
    """The pids of census_run.py processes writing this run (Linux; [] where /proc cannot be read): a process whose
    layers hold the run's and whose run name (its --run, else census_run.py's own L<layer>-<design>, -top<M> for a
    retest) is the run's."""
    out = []
    for d in Path("/proc").glob("[0-9]*"):
        try:
            argv = (d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        args = [x.decode(errors="replace") for x in argv if x]
        if not any(x.endswith("census_run.py") for x in args):
            continue
        opt = {args[i]: args[i + 1] for i in range(len(args) - 1) if args[i].startswith("--")}
        layers = _layers(opt.get("--layer") or opt.get("--layers") or "")
        if layer not in layers:
            continue
        name = (opt.get("--run") or "").replace("{layer}", str(layer)) or (
            f"L{layer}-{opt.get('--design') or cz.DEFAULT_DESIGN}" + (f"-top{opt['--top']}" if opt.get("--neurons-from") and opt.get("--top") else ""))
        if run.stem == name or run.stem.startswith(name + ".s"):  # the run, or one of its shards
            out.append(int(d.name))
    return sorted(out)


def _layers(spec: str) -> List[int]:
    out: List[int] = []
    for part in spec.split(","):
        lo, _, hi = part.strip().partition("-")
        if lo.isdigit():
            out += list(range(int(lo), int(hi) + 1)) if hi.isdigit() else [int(lo)]
    return out


def span(seconds: float) -> str:
    s = int(max(seconds, 0))
    h, m = divmod(s // 60, 60)
    return f"{h} h {m:02d} min" if h else (f"{m} min" if m else f"{s} s")


def colour(on: bool):
    def c(code: str, text: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if on else text
    return c


def draw(run: Path, meta: Dict[str, Any], done: int, total: int, rates: Tuple[Optional[float], Optional[float], str], window_min: float,
         idle_s: float, pids: List[int], width: int, c) -> List[str]:
    layer = int(meta.get("layer", -1))
    controls = int(meta.get("random_controls") or 0)
    per_unit = len(meta.get("design") or []) or 1
    units = (int((meta.get("jobs") or {}).get("n") or 0) - len(meta.get("prompt_ids") or {})) // per_unit
    head = [f"{run.stem} · layer {layer} (page {layer + 1}) · {meta.get('design_name')}"
            + (f" · shard {meta['shard']['i']}/{meta['shard']['n']}" if meta.get("shard") else ""),
            f"{max(units - controls, 0):,} neurons + {controls:,} controls · {meta.get('max_new')} pieces an answer · batch {meta.get('batch')}"]
    fit = lambda t: t[: max(width - 1, 20)]  # noqa: E731  every line within the terminal, so the redraw counts true
    frac = done / total if total else 0.0
    tail = f"] {frac:6.1%}  {done:,} / {total:,} answers"
    bar_w = max(10, min(60, width - 2 - len(tail)))
    full = int(frac * bar_w)
    part = ("", "▏", "▎", "▍", "▌", "▋", "▊", "▉")[int((frac * bar_w - full) * 8)] if full < bar_w else ""
    bar = c("33", "█" * full + part) + c("2", "░" * (bar_w - full - len(part)))
    lines = [c("1", fit(head[0])), c("2", fit(head[1])), f"[{bar}" + tail]
    recent, overall, said = rates
    rate = recent if recent else overall
    left = total - done
    if done >= total and total:
        lines.append(c("32", "every answer written"))
    elif rate:
        eta = time.time() + left / rate
        lines.append(fit(f"{rate:.1f} answers/s ({'last ' + format(window_min, 'g') + ' min' if recent else said})"
                         f" · {span(left / rate)} left · done about {time.strftime('%H:%M %a', time.localtime(eta))}"))
    else:
        lines.append("waiting for answers to measure the rate…")
    started = meta.get("started_at")
    since = ""
    if started:
        try:
            t0 = calendar.timegm(time.strptime(started, "%Y-%m-%dT%H:%M:%SZ"))
            since = f" · begun {time.strftime('%H:%M %a', time.localtime(t0))}, {span(time.time() - t0)} ago"
        except ValueError:
            pass
    alarm = False
    if pids:
        state = f"writer running (pid {', '.join(map(str, pids))})"
        if idle_s > STALLED_S and done < total:
            state, alarm = state + f" but no answer for {span(idle_s)}", True
    elif done < total:
        state = ("no census_run.py is writing it" + (f" and no answer for {span(idle_s)}" if idle_s > 30 else "")
                 + ": stopped? the same command continues it row for row")
        alarm = True
    else:
        state = "finished"
    lines.append(c("31", fit(state + since)) if alarm else fit(state + since))
    return lines


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("run", nargs="?", default="", help="the run's name or .jsonl path (default: the census folder's newest)")
    ap.add_argument("--every", type=float, default=5.0, help="seconds between looks (default 5)")
    ap.add_argument("--window", type=float, default=10.0, help="minutes the rate is taken over (default 10)")
    ap.add_argument("--once", action="store_true", help="one reading, then exit")
    a = ap.parse_args(argv)
    run = find_run(a.run)
    meta = json.loads(cz.meta_path_of(run).read_text(encoding="utf-8"))
    total = run_total(meta)
    lines = Lines(run)
    seen: Deque[Tuple[float, int]] = collections.deque()
    tty = sys.stdout.isatty()
    c = colour(tty and not os.environ.get("NO_COLOR"))
    first_t, first_n = time.time(), lines.update()
    try:
        began: Optional[float] = calendar.timegm(time.strptime(meta["started_at"], "%Y-%m-%dT%H:%M:%SZ"))
    except (KeyError, TypeError, ValueError):
        began = None
    last_change_t, last_n = first_t, first_n
    drawn = 0
    try:
        while True:
            now = time.time()
            done = lines.update()
            if done != last_n:
                last_change_t, last_n = now, done
            seen.append((now, done))
            while seen and now - seen[0][0] > a.window * 60:
                seen.popleft()
            # the rate over the last window; under a whole window, since watching (a minute at least: answers land a
            # batch at a time); under a minute, since the run began (its meta's start, which a continued run keeps)
            recent = (done - seen[0][1]) / (now - seen[0][0]) if now - seen[0][0] >= 60 and done > seen[0][1] else None
            overall, said = None, ""
            if recent is not None and now - first_t < a.window * 60:
                recent, overall, said = None, recent, "since watching"
            elif recent is None and began and done and now > began:
                overall, said = done / (now - began), "since it began"
            width = shutil.get_terminal_size((100, 20)).columns
            out = draw(run, meta, done, total, (recent, overall, said), a.window, now - last_change_t,
                       writers(run, int(meta.get("layer", -1))), width, c)
            if tty and drawn:
                sys.stdout.write(f"\x1b[{drawn}F\x1b[J")
            sys.stdout.write("\n".join(out) + "\n")
            sys.stdout.flush()
            drawn = len(out)
            if a.once or (total and done >= total):
                return 0
            time.sleep(max(a.every, 0.5))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())

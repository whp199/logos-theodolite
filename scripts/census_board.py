#!/usr/bin/env python3
"""The board of the census's volunteers (VOLUNTEERS.md): which layer each holds, how far each run is, and what has come
home, kept in volunteers/board.json and shown on a pinned GitHub issue, in volunteers/BOARD.md, as a thread post
(volunteers/board.txt, under 2,000 characters) and as an image (volunteers/board.png), with the guide as one image
(volunteers/post/guide.png, from guide.html).

    census_board.py                                   # the board as it stands
    census_board.py add 'CENSUS RESERVE L47 claim=3f9a1c2e card="NVIDIA GeForce RTX 5090 32 GB" name=anon kit=1'
    census_board.py add - --via thread                # every CENSUS line pasted on stdin (a thread's posts, say)
    census_board.py sync                              # the CENSUS lines among the board issue's new comments
    census_board.py expire                            # claims past their hold given back
    census_board.py set 3f9a1c2e accepted --note "checked bit for bit"   # census_intake.py sets these
    census_board.py layer 45 done --note "the home card, 30 October 2026"
    census_board.py render                            # BOARD.md, board.txt and board.png
    census_board.py publish                           # what would go to the issue; --yes to send it, --create to make it

A volunteer's run passes through: reserved (RESERVE), running (START, then PROGRESS), submitted (DONE, with a link),
checking, then accepted or rejected (census_intake.py); or released (RELEASE, with the claim's secret) or lapsed. One
claim holds a layer at a time. A claim holds for HOLD_RESERVED hours after its RESERVE, until it is started; a run for
HOLD_STARTED hours after START until its first PROGRESS with an end time, then until that end time and HOLD_GRACE hours
more, each PROGRESS moving it; never past HOLD_MAX_DAYS after the RESERVE. A submitted claim holds until it is judged.
A claim past its hold is lapsed when the board is shown, and its layer open again; its holder may claim it anew.

Anyone may post a line, so a line is taken as what it says only so far as it can be checked: a RESERVE on a layer held
is refused, a RELEASE needs the claim's secret, and a DONE's bundle is known to be the claim's only by the secret in its
manifest (census_intake.py). Nothing here is sent anywhere but by `publish --yes` and `sync`, which use the gh CLI.
Layers count from 0 (page = layer + 1).
"""

from __future__ import annotations

import argparse
import calendar
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend import census as cz  # noqa: E402
from backend import census_volunteer as cv  # noqa: E402

BOARD = ROOT_DIR / "volunteers" / "board.json"
HOLD_RESERVED = 36
HOLD_STARTED = 48
HOLD_GRACE = 24
HOLD_MAX_DAYS = 10
LIVE = ("reserved", "running", "submitted", "checking")
ENDED = ("accepted", "rejected", "released", "lapsed")
CHROME = ("google-chrome-stable", "google-chrome", "chromium", "chromium-browser")


def now() -> float:
    return float(os.environ.get("THEODOLITE_BOARD_NOW") or time.time())  # a check's clock


def iso(t: Optional[float]) -> Optional[str]:
    return None if t is None else time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime(t))


def parse_iso(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%MZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return float(calendar.timegm(time.strptime(s, fmt)))
        except ValueError:
            pass
    return None


def load(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    kit = cv.read_kit()
    return {"kit": kit["KIT"], "repo": "whp199/logos-theodolite", "issue": None, "seen_comments": [], "claims": [],
            "layers": {str(l): {"state": "open"} for l in kit["layers"]}}


def save(board: Dict[str, Any], path: Path) -> None:
    board["updated_at"] = iso(now())
    cz.write_json_atomic(path, board)


def holds_until(c: Dict[str, Any]) -> Optional[float]:
    """When a claim's hold ends (None: it holds until judged)."""
    if c["state"] in ("submitted", "checking"):
        return None
    t0 = parse_iso(c["at"]) or 0.0
    cap = t0 + HOLD_MAX_DAYS * 86400
    if c["state"] == "reserved":
        return min(t0 + HOLD_RESERVED * 3600, cap)
    eta = parse_iso(c.get("eta"))
    if eta is not None:
        return min(eta + HOLD_GRACE * 3600, cap)
    return min((parse_iso(c.get("started_at")) or t0) + HOLD_STARTED * 3600, cap)


def lapse(board: Dict[str, Any]) -> List[str]:
    out = []
    for c in board["claims"]:
        h = holds_until(c) if c["state"] in LIVE else None
        if h is not None and now() > h:
            c["state"] = "lapsed"
            c.setdefault("events", []).append({"at": iso(now()), "what": f"lapsed (held until {iso(h)})"})
            out.append(f"L{c['layer']} {c['code']} lapsed: held until {iso(h)}")
    return out


def live_claim(board: Dict[str, Any], layer: int) -> Optional[Dict[str, Any]]:
    return next((c for c in board["claims"] if c["layer"] == layer and c["state"] in LIVE), None)


def find(board: Dict[str, Any], code: str) -> Optional[Dict[str, Any]]:
    cs = [c for c in board["claims"] if c["code"] == code]
    return cs[-1] if cs else None


def clean(s: Any, n: int = 40) -> str:
    """A volunteer's words as the board may show them: printable, without markup, cut short."""
    s = re.sub(r"[^\w .,:/+%()@#-]", "", str(s or ""), flags=re.ASCII)
    return s[:n]


def apply(board: Dict[str, Any], line: Dict[str, Any], via: str, who: Optional[str] = None) -> str:
    """One CENSUS line applied to the board; what it did, in words (a refusal starts "refused")."""
    lapse(board)
    verb, layer, code = line["verb"], int(line["layer"]), clean(line.get("claim"), 8).lower()
    if not re.fullmatch(r"[0-9a-f]{8}", code):
        return f"refused: {verb} L{layer} names no claim code (claim=<8 hex>)"
    t = iso(now())
    ev = {"at": t, "via": via, "what": " ".join(f"{k}={v}" for k, v in line.items() if k not in ("verb", "layer", "words"))}
    c = find(board, code)
    if verb == "RESERVE":
        st = board["layers"].get(str(layer), {}).get("state")
        if st != "open":
            return f"refused: layer {layer} is not open ({st or 'not a kit layer'})"
        held = live_claim(board, layer)
        if held and held["code"] != code:
            return f"refused: layer {layer} is held by {held['code']} ({held['state']}) until {iso(holds_until(held)) or 'judged'}"
        if held:
            return f"L{layer} {code} holds it already"
        mine = [x for x in board["claims"] if x["code"] == code and x["state"] in LIVE]
        if mine:
            return f"refused: claim {code} holds layer {mine[0]['layer']} already: one layer a claim"
        board["claims"].append({"code": code, "layer": layer, "name": clean(line.get("name") or who or "anon", 24),
                                "card": clean(line.get("card"), 64), "kit": line.get("kit"), "via": via, "at": t,
                                "state": "reserved", "events": [{**ev, "verb": verb}]})
        return f"L{layer} reserved by {code} ({clean(line.get('name') or who or 'anon', 24)}, {clean(line.get('card'), 48)})"
    if c is None or c["layer"] != layer:
        return f"refused: no claim {code} on layer {layer} (a RESERVE first)"
    if verb == "RELEASE":
        if cv.claim_code(str(line.get("secret") or "")) != code:
            return f"refused: RELEASE L{layer} {code} without the claim's secret"
        if c["state"] not in LIVE:
            return f"L{layer} {code} is {c['state']} already"
        c["state"] = "released"
        c["events"].append({**ev, "what": "released (the secret shown)", "verb": verb})
        return f"L{layer} released by {code}: open again"
    if c["state"] not in LIVE:
        return f"refused: claim {code} is {c['state']}"
    c["events"].append({**ev, "verb": verb})
    if verb == "START":
        if c["state"] == "reserved":
            c["state"] = "running"
            c["started_at"] = t
        c["bits"] = clean(line.get("bits"), 8)
        if line.get("card"):
            c["card"] = clean(line["card"], 64)
        return f"L{layer} {code} running ({c['bits'] or '?'} bits) until at least {iso(holds_until(c))}"
    if verb == "PROGRESS":
        if c["state"] == "reserved":
            c["state"] = "running"
            c["started_at"] = t
        c["done"] = clean(line.get("done"), 5)
        if parse_iso(line.get("eta")):
            c["eta"] = line["eta"]
        return f"L{layer} {code} {c['done'] or '?'} done, held until {iso(holds_until(c))}"
    if verb == "DONE":
        c["state"] = "submitted"
        c["submitted_at"] = t
        c["bundle_sha256"] = clean(line.get("sha256"), 64)
        c["size"] = clean(line.get("size"), 10)
        link = str(line.get("link") or "")
        c["link"] = link if re.fullmatch(r"https?://[^\s<>\"']{4,300}", link) else None
        return f"L{layer} {code} submitted ({c['size'] or '?'}, link {'given' if c['link'] else 'missing: ask for it'})"
    return f"refused: {verb}"


def layer_rows(board: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Each layer's line on the board, its state from its live claim, else its own."""
    rows = []
    for key in sorted(board["layers"], key=int):
        L = int(key)
        info = board["layers"][key]
        c = live_claim(board, L)
        row = {"layer": L, "page": L + 1, "state": info["state"], "who": "", "card": "", "when": "", "note": info.get("note", "")}
        if info["state"] == "open" and c:
            row.update({"state": c["state"], "who": f"{c['name']} {c['code']}", "card": c.get("card", ""), "note": ""})
            if c["state"] == "running":
                row["when"] = f"{c.get('done') or '?'}" + (f", ends ~{c['eta']}" if c.get("eta") else "")
            elif c["state"] == "reserved":
                row["when"] = f"start by {iso(holds_until(c))}"
            else:
                row["when"] = c.get("submitted_at") or ""
        elif info["state"] == "done":
            row.update({"who": info.get("by", ""), "card": info.get("card", "")})
        rows.append(row)
    return rows


STATE_WORDS = {"open": "OPEN", "reserved": "reserved", "running": "running", "submitted": "sent home", "checking": "checking",
               "done": "done", "closed": "closed"}


def render_md(board: Dict[str, Any]) -> str:
    kit = cv.read_kit()
    lines = ["# The census: which layer is whose", "",
             f"Kit {kit['KIT']} ({kit['KIT_TAG']}). How to take a layer: [VOLUNTEERS.md]"
             f"(https://github.com/{board['repo']}/blob/main/VOLUNTEERS.md). Post `CENSUS ...` lines as comments here or in "
             "the thread; the board is updated by hand, so give it a little while. Layers count from 0 (page = layer + 1).", "",
             "| Layer | Page | State | Who | Card | When | Note |", "|---:|---:|---|---|---|---|---|"]
    for r in layer_rows(board):
        lines.append(f"| {r['layer']} | {r['page']} | {STATE_WORDS.get(r['state'], r['state'])} | {r['who']} | {r['card']} | "
                     f"{r['when']} | {r['note']} |")
    ended = [c for c in board["claims"] if c["state"] in ("accepted", "rejected")]
    if ended:
        lines += ["", "Judged:", ""] + [f"- L{c['layer']} {c['name']} {c['code']}: {c['state']}"
                                        + (f" ({clean(c.get('note'), 120)})" if c.get("note") else "") for c in ended]
    lines += ["", f"Holds: {HOLD_RESERVED} h after RESERVE to START; then the run's end time (from PROGRESS) plus {HOLD_GRACE} h; "
              f"at most {HOLD_MAX_DAYS} days in all. Updated {board.get('updated_at') or iso(now())}.", ""]
    return "\n".join(lines)


def render_txt(board: Dict[str, Any]) -> str:
    kit = cv.read_kit()
    rows = layer_rows(board)
    out = ["CENSUS BOARD (layers count from 0; page = layer + 1)"]
    for r in rows:
        s = STATE_WORDS.get(r["state"], r["state"])
        bits = [f"L{r['layer']}", s] + [x for x in (r["who"], r["card"], r["when"]) if x]
        out.append(" ".join(bits) if r["state"] != "done" else f"L{r['layer']} done ({r['who']})")
    n_open = sum(1 for r in rows if r["state"] == "open")
    out += ["", f"{n_open} open. To take one: {kit['KIT_BOARD_URL'].rstrip('/')}",
            f"guide: https://github.com/{board['repo']}/blob/main/VOLUNTEERS.md",
            "reply with the CENSUS line census.sh prints"]
    text = "\n".join(out)
    return text if len(text) <= 2000 else text[:1990] + "\n(cut)"


def render_html(board: Dict[str, Any]) -> str:
    rows = layer_rows(board)
    tr = []
    for r in rows:
        st = r["state"]
        tr.append(f"<tr class='{html.escape(st)}'><td class='n'>{r['layer']}</td><td class='n'>{r['page']}</td>"
                  f"<td class='st'>{html.escape(STATE_WORDS.get(st, st))}</td><td>{html.escape(r['who'])}</td>"
                  f"<td>{html.escape(r['card'])}</td><td>{html.escape(r['when'])}</td><td>{html.escape(r['note'])}</td></tr>")
    n_open = sum(1 for r in rows if r["state"] == "open")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono&display=block" rel="stylesheet">
<style>
:root {{ --vellum:#F6F0E3; --ink:#2B2117; --muted:#6B5D4D; --rubric:#8F2B1E; --gilt:#A07A2E; --gilt-text:#7A5C28; }}
html,body {{ margin:0; background:var(--vellum); color:var(--ink); }}
body {{ font-family:'EB Garamond', Georgia, serif; font-size:21px; padding:34px 40px 30px; width:920px; }}
h1 {{ font-weight:600; font-size:34px; margin:0 0 2px; letter-spacing:.01em; }}
.sub {{ color:var(--rubric); font-style:italic; margin:0 0 18px; }}
table {{ border-collapse:collapse; width:100%; }}
th {{ font-variant:small-caps; letter-spacing:.06em; font-weight:600; color:var(--gilt-text); text-align:left;
      border-bottom:1px solid var(--gilt); padding:4px 8px; }}
td {{ border-bottom:1px solid rgba(160,122,46,.35); padding:5px 8px; vertical-align:top; }}
td.n {{ font-variant-numeric:tabular-nums; text-align:right; width:44px; }}
td.st {{ font-variant:small-caps; letter-spacing:.05em; }}
tr.open td.st {{ color:var(--rubric); font-weight:600; }}
tr.done td, tr.closed td {{ color:var(--muted); }}
.foot {{ margin-top:16px; color:var(--muted); font-size:18px; }}
code {{ font-family:'IBM Plex Mono', monospace; font-size:16px; }}
</style></head><body>
<h1>The census: which layer is whose</h1>
<p class="sub">{n_open} layers open. Gemma-4-31B, every neuron of a layer pushed and read against 2,048 random directions.</p>
<table><tr><th>layer</th><th>page</th><th>state</th><th>who</th><th>card</th><th>when</th><th>note</th></tr>
{''.join(tr)}</table>
<p class="foot">To take one: the guide, <code>VOLUNTEERS.md</code> in github.com/{html.escape(board['repo'])}; post the
<code>CENSUS</code> line the script prints. Updated {html.escape(board.get('updated_at') or iso(now()) or '')}.</p>
</body></html>"""


def chrome() -> Optional[str]:
    from shutil import which

    return next((which(c) for c in CHROME if which(c)), None)


# appended to a page whose height is measured: once its fonts are in, the page's height on a body attribute
_MEASURE = ("<script>document.fonts.ready.then(() => document.body.setAttribute('data-h', "
            "Math.ceil(document.documentElement.scrollHeight)))</script>")


def html_to_png(html_text: str, out: Path, width: int, height: Optional[int] = None) -> bool:
    """The page as a PNG `width` wide and, unless `height` is given, as tall as the page is at that width with its
    fonts loaded (measured first in a window shorter than any page, so the page sets the height)."""
    exe = chrome()
    if not exe:
        print("no Chrome or Chromium here: the PNG is not made (BOARD.md and board.txt are)")
        return False
    with tempfile.TemporaryDirectory() as d:
        page = Path(d) / "page.html"
        base = [exe, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run", "--no-default-browser-check",
                f"--user-data-dir={d}/profile", "--virtual-time-budget=8000", "--force-device-scale-factor=1"]
        if height is None:
            page.write_text(html_text.replace("</body>", _MEASURE + "</body>"), encoding="utf-8")
            got = subprocess.run(base + [f"--window-size={width},300", "--dump-dom", page.as_uri()], capture_output=True,
                                 text=True, timeout=120)
            m = re.search(r'data-h="(\d+)"', got.stdout)
            if not m:
                print(f"Chrome did not measure the page: {got.stderr.strip()[-300:]}")
                return False
            height = int(m.group(1))
        page.write_text(html_text, encoding="utf-8")
        cmd = base + [f"--window-size={width},{height}", f"--screenshot={out}", page.as_uri()]
        got = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if got.returncode != 0 or not out.exists():
            print(f"Chrome did not make the PNG: {got.stderr.strip()[-300:]}")
            return False
    return True


def cmd_render(board: Dict[str, Any], a: argparse.Namespace) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "BOARD.md").write_text(render_md(board), encoding="utf-8")
    (out / "board.txt").write_text(render_txt(board) + "\n", encoding="utf-8")
    print(f"-> {out / 'BOARD.md'}, {out / 'board.txt'} ({len(render_txt(board))} characters)")
    if not a.no_png:
        if html_to_png(render_html(board), out / "board.png", 1000):
            print(f"-> {out / 'board.png'}")
        guide = out / "post" / "guide.html"  # the guide as one image, to post beside the board
        if guide.exists() and html_to_png(guide.read_text(encoding="utf-8"), out / "post" / "guide.png", 1000):
            print(f"-> {out / 'post' / 'guide.png'}")


def gh(*args: str) -> str:
    got = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=120)
    if got.returncode != 0:
        raise SystemExit(f"gh {' '.join(args[:3])}... failed: {got.stderr.strip()[:400]}")
    return got.stdout


def cmd_sync(board: Dict[str, Any], a: argparse.Namespace) -> List[str]:
    if not board.get("issue"):
        raise SystemExit("The board has no issue yet: census_board.py publish --create --yes makes one.")
    # one comment a line, as a JSON array [id, login, body] (gh prints a jq string raw)
    raw = gh("api", "--paginate", f"repos/{board['repo']}/issues/{board['issue']}/comments",
             "--jq", ".[] | [.id, .user.login, .body] | @json")
    seen = set(board.get("seen_comments") or [])
    said = []
    for ln in raw.splitlines():
        if not ln.strip():
            continue
        cid, login, body = json.loads(ln)
        if cid in seen:
            continue
        seen.add(cid)
        line = cv.parse_line(body or "")
        if line:
            said.append(f"{login}: " + apply(board, line, "github", who=login))
    board["seen_comments"] = sorted(seen)
    return said


def cmd_publish(board: Dict[str, Any], a: argparse.Namespace, path: Path) -> None:
    body = render_md(board)
    if a.create:
        if board.get("issue"):
            raise SystemExit(f"The board has an issue already (#{board['issue']}).")
        if not a.yes:
            print(f"would create an issue on {board['repo']} titled 'Census layer board', pin it, and put this in it:\n\n{body}")
            print("(again with --yes to do it)")
            return
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
            f.write(body)
        url = gh("issue", "create", "--repo", board["repo"], "--title", "Census layer board", "--body-file", f.name).strip()
        os.unlink(f.name)
        board["issue"] = int(url.rstrip("/").rsplit("/", 1)[-1])
        gh("issue", "pin", str(board["issue"]), "--repo", board["repo"])
        save(board, path)
        print(f"-> {url} (pinned); set KIT_BOARD_URL in volunteers/kit.env to it")
        return
    if not board.get("issue"):
        raise SystemExit("The board has no issue yet: census_board.py publish --create shows what it would make.")
    if not a.yes:
        print(f"would put this in {board['repo']}#{board['issue']}:\n\n{body}\n(again with --yes to send it)")
        return
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(body)
    gh("issue", "edit", str(board["issue"]), "--repo", board["repo"], "--body-file", f.name)
    os.unlink(f.name)
    print(f"-> {board['repo']}#{board['issue']} updated")


def show(board: Dict[str, Any]) -> None:
    for r in layer_rows(board):
        print(f"L{r['layer']:<3} page {r['page']:<3} {STATE_WORDS.get(r['state'], r['state']):<10} {r['who']:<24} {r['card']:<30} {r['when']}"
              + (f"  {r['note']}" if r["note"] else ""))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", default=str(BOARD), help="the board's file (default volunteers/board.json)")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("add", help="apply CENSUS lines (a line, or - for stdin)")
    p.add_argument("text", nargs="+")
    p.add_argument("--via", default="thread", help="where the line was posted (thread, github, mail...)")
    p.add_argument("--who", default=None, help="the poster's name, where the line gives none")
    sub.add_parser("sync", help="the CENSUS lines among the board issue's new comments")
    sub.add_parser("expire", help="lapse claims past their hold")
    s = sub.add_parser("set", help="a claim's state, as the intake judges it")
    s.add_argument("code")
    s.add_argument("state", choices=["checking", "accepted", "rejected", "submitted", "lapsed", "released"])
    s.add_argument("--note", default="")
    lp = sub.add_parser("layer", help="a layer's own state")
    lp.add_argument("layer", type=int)
    lp.add_argument("state", choices=["open", "done", "closed"])
    lp.add_argument("--by", default="")
    lp.add_argument("--card", default="")
    lp.add_argument("--note", default="")
    r = sub.add_parser("render", help="BOARD.md, board.txt and board.png")
    r.add_argument("--out", default=str(ROOT_DIR / "volunteers"))
    r.add_argument("--no-png", action="store_true")
    pub = sub.add_parser("publish", help="the board onto its GitHub issue (--yes to send)")
    pub.add_argument("--yes", action="store_true")
    pub.add_argument("--create", action="store_true", help="make and pin the issue")
    a = ap.parse_args(argv)
    path = Path(a.board)
    board = load(path)
    said = lapse(board)
    if a.cmd == "add":
        texts = [sys.stdin.read()] if a.text == ["-"] else a.text
        found = 0
        for t in texts:
            for raw in t.splitlines():
                line = cv.parse_line(raw)
                if line:
                    found += 1
                    said.append(apply(board, line, a.via, a.who))
        if not found:
            said.append("no CENSUS line in what was given")
    elif a.cmd == "sync":
        said += cmd_sync(board, a)
    elif a.cmd == "set":
        c = find(board, a.code)
        if c is None:
            raise SystemExit(f"No claim {a.code} on the board.")
        c["state"] = a.state
        if a.note:
            c["note"] = a.note
        c.setdefault("events", []).append({"at": iso(now()), "what": f"set {a.state}" + (f": {a.note}" if a.note else "")})
        if a.state == "accepted":
            board["layers"][str(c["layer"])] = {"state": "done", "by": f"{c['name']} {c['code']}", "card": c.get("card", ""),
                                                "note": a.note}
        said.append(f"L{c['layer']} {c['code']} {a.state}")
    elif a.cmd == "layer":
        board["layers"][str(a.layer)] = {"state": a.state, **{k: v for k, v in (("by", a.by), ("card", a.card), ("note", a.note)) if v}}
        said.append(f"L{a.layer} {a.state}")
    for s_ in said:
        print(s_)
    if a.cmd in ("add", "sync", "expire", "set", "layer") or (said and a.cmd is None):
        save(board, path)
    if a.cmd == "render":
        save(board, path)
        cmd_render(board, a)
    elif a.cmd == "publish":
        cmd_publish(board, a, path)
    elif a.cmd in (None, "add", "sync", "expire", "set", "layer"):
        show(board)
    return 1 if any(s_.startswith("refused") for s_ in said) else 0


if __name__ == "__main__":
    sys.exit(main())

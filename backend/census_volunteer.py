"""The census's volunteer kit (VOLUNTEERS.md): what a layer taken on someone else's card must be to stand beside the home
card's, and the one-line messages volunteers and the owner pass each other. Read on a volunteer's machine by
scripts/volunteer/volunteer_tool.py, and at home by scripts/census_board.py and scripts/census_intake.py.

- The kit: volunteers/kit.env, the settings every volunteer's run is taken with (scripts/volunteer/census.sh sources
  the same file). A volunteer's layer is the home card's layer 45 run's command at another layer.
- A claim: a secret the volunteer keeps on their machine, and its code, the first 8 hex of its sha256, which they post.
  Whoever shows the secret later (in a submitted bundle's manifest, or a RELEASE line) is the claim's holder; no
  account and no name are needed.
- The lines: `CENSUS <VERB> L<layer> key=value ...` (RESERVE, START, PROGRESS, DONE, RELEASE), printed by census.sh for
  the volunteer to post in the thread or on the board's issue, read back by census_board.py.
- A canary's fingerprint: the sha256 of its rows as canonical JSON. Plan batch 0 of a layer's run written on the home
  card (volunteers/home_canaries/) against the same batch written on the volunteer's: the same fingerprint is the same
  bits, and the run is of the home card's class; another is a card class of its own, checked by ACCEPT_OTHER_CLASS.
- The rules a submitted run is checked by (`compare_rows`, ACCEPT_OTHER_CLASS), fixed on 30 October 2026 before any
  volunteer's run was seen.

Layers count from 0 (page = layer + 1), as every script does.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
KIT_FILE = ROOT_DIR / "volunteers" / "kit.env"
KIT_FILE_ENV = "THEODOLITE_KIT_FILE"  # another kit file, for a check (scripts/check_volunteer_kit.py, a test at full size)
HOME_CANARIES = ROOT_DIR / "volunteers" / "home_canaries"
VERBS = ("RESERVE", "START", "PROGRESS", "DONE", "RELEASE")

# A run of another card class (its canary not the home card's bits) is taken when, on the chunks re-written at home
# (scripts/census_intake.py spot), every answer is there and
#   first_piece: at least this share of answers write the same first word-piece as the home card's (greedy answers
#                part at near-ties, which are rare at the first piece; a run at another layer, dose or push, or made
#                up, parts at once);
#   lp_median, lp_p99: on the word-pieces both wrote alike before the two part, where the home card is not sure of
#                its piece (log-probability below -0.01), the median and 99th percentile of |difference| stay under
#                these (another card's sums move a log-probability by about 1e-3; another model or push, far more);
#   baseline_shared: each unpushed answer shares at least this many word-pieces with the home card's (the whole
#                answer, where the home card's is shorter).
# A run of the home card's class must match every spot answer bit for bit.
ACCEPT_OTHER_CLASS = {"first_piece": 0.90, "lp_median": 0.01, "lp_p99": 0.10, "baseline_shared": 16}
LP_SURE = -0.01


def read_kit(path: Optional[Path] = None, dry: bool = False) -> Dict[str, Any]:
    """The kit's settings (KEY=VALUE lines, # comments), the stand-in's KIT_DRY_* in place of theirs when `dry`."""
    import os

    path = path or Path(os.environ.get(KIT_FILE_ENV) or KIT_FILE)
    kit: Dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        kit[k.strip()] = shlex.split(v)[0] if v.strip() else ""
    if dry:
        for k in [k for k in kit if k.startswith("KIT_DRY_")]:
            kit["KIT_" + k[len("KIT_DRY_"):]] = kit[k]
    out: Dict[str, Any] = dict(kit)
    out["layers"] = [int(x) for x in kit.get("KIT_LAYERS", "").split()]
    for k in ("KIT", "KIT_CONTROLS", "KIT_SEED", "KIT_BATCH", "KIT_MAX_NEW", "KIT_MIN_VRAM_MIB", "KIT_MIN_DISK_GB"):
        if k in kit:
            out[k] = int(kit[k])
    out["dry"] = dry
    return out


def claim_code(secret: str) -> str:
    """The code a claim is posted by: the first 8 hex of the secret's sha256."""
    return hashlib.sha256(secret.strip().encode("utf-8")).hexdigest()[:8]


_LINE = re.compile(r"CENSUS\s+(" + "|".join(VERBS) + r")\s+L(\d{1,2})\b(.*)")


def parse_line(text: str) -> Optional[Dict[str, Any]]:
    """The first CENSUS line in `text` (a post may quote it, greentext it or wrap it in other words), as {verb, layer,
    and its key=value fields}; None when there is none. A field without "=" is kept under `words`."""
    for raw in str(text).splitlines():
        m = _LINE.search(raw.replace("“", '"').replace("”", '"'))
        if not m:
            continue
        out: Dict[str, Any] = {"verb": m.group(1), "layer": int(m.group(2)), "words": []}
        try:
            parts = shlex.split(m.group(3))
        except ValueError:  # an unclosed quote: read it by spaces
            parts = m.group(3).split()
        for p in parts:
            k, eq, v = p.partition("=")
            if eq and re.fullmatch(r"[a-z_]+", k):
                out[k] = v
            else:
                out["words"].append(p)
        return out
    return None


def format_line(verb: str, layer: int, **fields: Any) -> str:
    """A CENSUS line, fields in the order given (a value with a space quoted), None left out."""
    if verb not in VERBS:
        raise ValueError(f"A CENSUS line is one of {', '.join(VERBS)}, not {verb!r}.")
    parts = [f"CENSUS {verb} L{int(layer)}"]
    for k, v in fields.items():
        if v is None or v == "":
            continue
        v = str(v)
        parts.append(f"{k}={shlex.quote(v) if re.search(r'[^A-Za-z0-9._:/+%-]', v) else v}")
    return " ".join(parts)


def canonical_rows(rows: Iterable[Dict[str, Any]]) -> bytes:
    return b"".join(json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n" for r in rows)


def rows_fingerprint(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """{answers, sha256} of rows as canonical JSON (sorted keys, no spaces), in their order."""
    return {"answers": len(rows), "sha256": hashlib.sha256(canonical_rows(rows)).hexdigest()}


def home_canary(layer: int, design: str, folder: Path = HOME_CANARIES) -> Tuple[Path, Path]:
    """The home card's canary of a layer's kit run (its rows, and what it was taken with)."""
    stem = f"L{int(layer)}-{design}"
    return folder / f"{stem}.canary.jsonl", folder / f"{stem}.canary.json"


def expected_jobs(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every answer a run of this meta must hold, as census_run.py plans them (its neurons, controls, design and
    questions); `census.jobs_digest` of it is the meta's own `jobs` for a run that kept its plan."""
    from backend import census as cz

    held = cz.parse_neurons(meta["neuron_ids"], int(meta["d_ff"])) if meta.get("neuron_ids") else []
    return cz.census_jobs(held, int(meta.get("random_controls") or 0), [tuple(x) for x in meta["design"]],
                          cz.meta_questions(meta))


def meta_problems(meta: Dict[str, Any], home: Dict[str, Any], kit: Dict[str, Any]) -> List[str]:
    """What keeps a run's meta from being the kit's run at its layer, set against the meta the home card's canary of
    that layer was taken with: the settings a run is continued by (census.settings_differ, the push's p50 among them),
    the job list, the stats index, the unpacker's test and the stand-in. Empty when it is the kit's run."""
    from backend import census as cz

    out = []
    differ = cz.settings_differ(home, meta)
    if differ:
        out.append("taken with other settings than the kit's: " + ", ".join(differ))
    for k in ("neuron_ids", "d_ff", "jobs", "stand_in"):
        if json.dumps(meta.get(k)) != json.dumps(home.get(k)):
            out.append(f"its {k} is not the kit's ({str(meta.get(k))[:60]} against {str(home.get(k))[:60]})")
    if meta.get("shard") or meta.get("stage") or meta.get("joined"):
        out.append("a shard, a retest or a joined run, not a whole layer's run")
    rt = meta.get("runtime") or {}
    if not kit.get("dry") and rt.get("act_scale_sha256") != kit.get("KIT_ACT_SCALE_SHA256"):
        out.append(f"its push's unit came from another stats index (sha256 {str(rt.get('act_scale_sha256'))[:12]})")
    test = (meta.get("unpacker") or {}).get("self_test") or {}
    if not kit.get("dry") and ((meta.get("unpacker") or {}).get("mode") != kit.get("KIT_UNPACKER") or test.get("same_bits") is not True):
        out.append("its unpacker is not the kit's, or did not pass its test against the reference on the card")
    return out


def scan_run(path: Path, jobs: Sequence[Dict[str, Any]], layer: int) -> Dict[str, Any]:
    """A run file read through once against its job list: answers, keys missing, extra or twice, rows of another layer,
    rows whose word-pieces and log-probabilities do not line up, the unpushed answers (by question), and the file's
    sha256. A row is not kept beyond its key, so a whole layer costs a pass over the file."""
    want = {}
    for j in jobs:
        want[(j["kind"], int(j["neuron"]), j["prompt"], float(j["dose"]))] = 0
    h = hashlib.sha256()
    n = extra = twice = other_layer = misshapen = 0
    baselines: Dict[str, Dict[str, Any]] = {}
    with open(path, "rb") as f:
        for line in f:
            h.update(line)
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            k = (str(r["kind"]), int(r["neuron"]), str(r["prompt"]), float(r["dose"]))
            if k not in want:
                extra += 1
            else:
                twice += want[k] > 0
                want[k] += 1
            other_layer += int(r.get("layer", -1)) != int(layer)
            ids, lp = r.get("token_ids"), r.get("logprob")
            if not (isinstance(ids, list) and isinstance(lp, list) and len(ids) == len(lp) and all(isinstance(t, int) for t in ids)):
                misshapen += 1
            if r["kind"] == "baseline":
                baselines[r["prompt"]] = {"token_ids": ids, "logprob": lp}
    missing = sum(1 for v in want.values() if v == 0)
    return {"answers": n, "planned": len(jobs), "missing": missing, "extra": extra, "twice": twice,
            "other_layer": other_layer, "misshapen": misshapen, "baselines": baselines, "sha256": h.hexdigest(),
            "whole": missing == 0 and extra == 0 and twice == 0 and other_layer == 0 and misshapen == 0}


def shared_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if int(x) != int(y):
            break
        n += 1
    return n


def _quantile(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


def compare_rows(theirs: Dict[Tuple[str, int, str, float], Dict[str, Any]], ours: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """The volunteer's answers (`theirs`, by census.row_key) against the same answers written at home (`ours`): how many
    are there, how many are the same bit for bit (every field), how many write the same first word-piece, the shared
    prefix of each pair, and on the pieces both wrote alike before they part, where the home card is not sure of its
    piece, |difference| of the log-probabilities."""
    from backend import census as cz

    n = present = same = first = 0
    prefixes: List[int] = []
    lp_diffs: List[float] = []
    for r in ours:
        n += 1
        t = theirs.get(cz.row_key(r))
        if t is None:
            continue
        present += 1
        same += t == r
        a, b = r.get("token_ids") or [], t.get("token_ids") or []
        k = shared_prefix(a, b)
        prefixes.append(k)
        first += bool(a and b and int(a[0]) == int(b[0]))
        la, lb = r.get("logprob") or [], t.get("logprob") or []
        lp_diffs.extend(abs(float(la[i]) - float(lb[i])) for i in range(min(k, len(la), len(lb))) if float(la[i]) < LP_SURE)
    return {"answers": n, "present": present, "identical": same, "first_piece": first / present if present else None,
            "prefix_median": _quantile([float(p) for p in prefixes], 0.5), "prefix_min": min(prefixes) if prefixes else None,
            "lp_pieces": len(lp_diffs), "lp_median": _quantile(lp_diffs, 0.5), "lp_p99": _quantile(lp_diffs, 0.99),
            "lp_max": max(lp_diffs) if lp_diffs else None}


def judge(cmp: Dict[str, Any], home_bits: bool, baselines_shared: Optional[Dict[str, Sequence[int]]] = None) -> Tuple[bool, List[str]]:
    """Whether a spot comparison (`compare_rows`) passes: bit for bit for the home card's class, ACCEPT_OTHER_CLASS for
    another, with each unpushed answer's [shared word-pieces, the home card's length] in `baselines_shared`. Returns
    (passes, the reasons it does not)."""
    why = []
    if cmp["present"] < cmp["answers"]:
        why.append(f"{cmp['answers'] - cmp['present']} of the {cmp['answers']} spot answers are not in the run")
    if home_bits:
        if cmp["identical"] < cmp["present"]:
            why.append(f"its canary is the home card's bits, but {cmp['present'] - cmp['identical']} of {cmp['present']} spot "
                       "answers differ from the home card's")
        return not why, why
    a = ACCEPT_OTHER_CLASS
    if cmp["first_piece"] is None or cmp["first_piece"] < a["first_piece"]:
        why.append(f"only {cmp['first_piece'] or 0:.3f} of its answers write the home card's first word-piece (at least "
                   f"{a['first_piece']})")
    if cmp["lp_median"] is not None and cmp["lp_median"] > a["lp_median"]:
        why.append(f"its log-probabilities differ from the home card's by a median {cmp['lp_median']:.4f} (at most {a['lp_median']})")
    if cmp["lp_p99"] is not None and cmp["lp_p99"] > a["lp_p99"]:
        why.append(f"its log-probabilities' 99th-percentile difference is {cmp['lp_p99']:.4f} (at most {a['lp_p99']})")
    for q, (k, n) in sorted((baselines_shared or {}).items()):
        if k < min(a["baseline_shared"], n):
            why.append(f"its unpushed {q} answer shares {k} word-pieces with the home card's (at least {min(a['baseline_shared'], n)})")
    return not why, why

#!/usr/bin/env python3
"""R2 03 (CPU): apply the pre-registered rule (experiments/results/R2_PREREG.md §6-§8) to R2_02_eval.json.

Primary metric: at each read position, layer and readout, hit = the filtered top-1 is the same word as the content
target (R2_common.word_key: NFKC, surrounding whitespace stripped, case folded). Positions whose content target is a
raw-text collapse word (R2_common.COLLAPSE_TARGETS: "own", "same") have no usable target, for every readout. Prompt
score = mean hit over the prompt's positions in the condition (and over the band's layers); pooled = mean over the
prompts of the four decision conditions (mid-reply = the "any" draw). 95% percentile intervals from 2000 prompt-level
bootstrap resamples (stratified by condition, seed 20261019, the same indices for every readout, layer and band).

A lens is served at a band iff all five rules hold (R2_PREREG.md §7):
  1. pooled, the 95% intervals of second - J++ and of second - naked eye lie above 0;
  2. in no decision condition is either per-condition interval entirely below 0 (no data counts as a failure);
  3. at every layer of the band and in every decision condition, the stuck-word share (by word, not counting
     positions where the stuck word is that position's content target) is at most 0.25;
  4. the exact one-sided sign test on prompt-level differences, against J++ and against the naked eye (the larger
     p of the two), Holm-adjusted across every lens-band test, is below 0.025;
  5. the pooled point estimates of second - J++ and of second - naked eye are each at least 0.05.
Miru at 58 is reference only, and where both lenses pass, R1's lens is served. Comparisons with 0 and the thresholds
use unrounded values. Four sensitivity analyses (S1-S4) rerun the rule on subsets and report any verdict that changes.

Writes experiments/results/R2_03_analysis.json and R2_03_report_section.md (a section for the R2 report).
Run:       CUDA_VISIBLE_DEVICES="" /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R2_03_analyze.py
               [--in X] [--out Y] [--report Z]
Self-test: CUDA_VISIBLE_DEVICES="" /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R2_03_analyze.py --selftest
           (a synthetic results file with known answers; checks every branch of the rule; writes nothing to results)
"""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import R2_common as C  # noqa: E402

ALL_CONDS: Tuple[str, ...] = C.CONDITIONS + C.SECONDARY_CONDITIONS
KINDS: Tuple[str, ...] = ("jpp", "naive", "r1", "miru")
BASELINES: Tuple[Tuple[str, str], ...] = (("jpp", "J++"), ("naive", "naked eye"))
# metric families: the bootstrap draws over the prompts that have a defined value, so each family has its own indices
FAMILY = {"f_word": 0, "f_id": 0, "f5_word": 0, "u_word_ct": 0, "u_next": 1, "u_fmt": 1}
PRIMARY = "f_word"
EPS = 1e-12


def memberships(p: Dict[str, Any]) -> List[str]:
    """The conditions a position belongs to: mid-reply is the "any" draw (a decision condition); the "content_next"
    draw is the reported-only mid-reply-content-next."""
    if p["cond"] != "mid-reply":
        return [p["cond"]]
    out = []
    d = p.get("draws") or []
    if "any" in d:
        out.append("mid-reply")
    if "content_next" in d:
        out.append("mid-reply-content-next")
    return out


def prev_str_of(p: Dict[str, Any]) -> Optional[str]:
    """The token just before the content target (R2_02 records it; else derived from the read token and window)."""
    if "ct_prev_str" in p:
        return p["ct_prev_str"]
    off = p.get("ct_offset")
    if off is None:
        return None
    seq = [p.get("read_tok")] + list(p.get("cont_strs") or [])
    return seq[off] if off < len(seq) else None


def r4(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(float(x), 4)


class Table:
    """Per-position hit values for every metric, readout and layer, and the prompts of each condition."""

    def __init__(self, data: Dict[str, Any], exclude_formatlike: bool = False):
        self.pos = data["positions"]
        N = len(self.pos)
        self.N = N
        self.cond_pos: Dict[str, List[int]] = {c: [] for c in ALL_CONDS}
        for j, p in enumerate(self.pos):
            for c in memberships(p):
                self.cond_pos[c].append(j)
        # prompts per condition, in order of first appearance, and their positions
        self.prompts: Dict[str, List[str]] = {}
        self.prompt_pos: Dict[str, Dict[str, List[int]]] = {}
        for c in ALL_CONDS:
            pp: Dict[str, List[int]] = {}
            for j in self.cond_pos[c]:
                pp.setdefault(self.pos[j]["pid"], []).append(j)
            self.prompts[c] = list(pp)
            self.prompt_pos[c] = pp
        self.layers: Dict[str, List[int]] = {}
        for k in KINDS:
            ls = set()
            for p in self.pos:
                ls |= {int(x) for x in p["r"].get(k, {})}
            self.layers[k] = sorted(ls)
        # the content target's word, and the pre-registered exclusions (the same for every readout)
        self.raw_ct_key = [C.word_key(p.get("ct_str")) if p.get("ct_id") is not None else None for p in self.pos]
        self.collapse = [k is not None and k in C.COLLAPSE_TARGETS for k in self.raw_ct_key]
        self.formatlike = []
        for p, k in zip(self.pos, self.raw_ct_key):
            if k is None:
                self.formatlike.append(False)
            elif "ct_formatlike" in p:
                self.formatlike.append(bool(p["ct_formatlike"]))
            else:
                self.formatlike.append(C.formatlike_target(p.get("ct_str"), prev_str_of(p)))
        self.exclude_formatlike = exclude_formatlike
        self.ct_key = [None if (k is None or col or (exclude_formatlike and fl)) else k
                       for k, col, fl in zip(self.raw_ct_key, self.collapse, self.formatlike)]
        self.vals: Dict[Tuple[str, str, int], np.ndarray] = {}
        self._ps: Dict[Tuple[Any, ...], np.ndarray] = {}

    def values(self, metric: str, kind: str, layer: int) -> np.ndarray:
        """[N] float: 1/0 per position, NaN where the metric is undefined (no usable content target) or not read."""
        key = (metric, kind, layer)
        if key in self.vals:
            return self.vals[key]
        v = np.full(self.N, np.nan)
        for j, p in enumerate(self.pos):
            r = p["r"].get(kind, {}).get(str(layer))
            if r is None:
                continue
            ck = self.ct_key[j]
            if metric == "f_word":
                v[j] = np.nan if ck is None else float(C.word_key(r["fs"][0]) == ck)
            elif metric == "f_id":
                v[j] = np.nan if ck is None else float(r["f"][0] == p["ct_id"])
            elif metric == "f5_word":
                v[j] = np.nan if ck is None else float(ck in {C.word_key(s) for s in r["fs"]})
            elif metric == "u_word_ct":
                v[j] = np.nan if ck is None else float(C.word_key(r["us"][0]) == ck)
            elif metric == "u_next":
                v[j] = float(r["u"][0] == p["next_id"])
            elif metric == "u_fmt":
                v[j] = float(not r["u0c"])
            else:
                raise ValueError(metric)
        self.vals[key] = v
        return v

    def band_values(self, metric: str, kind: str, layers: Sequence[int]) -> np.ndarray:
        """[N] position-level mean over the band's layers (NaN where undefined)."""
        import warnings

        vs = np.stack([self.values(metric, kind, l) for l in layers], 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN columns (no usable target) stay NaN
            return np.nanmean(vs, 0)

    def prompt_scores(self, metric: str, kind: str, layers: Sequence[int], cond: str) -> np.ndarray:
        """[n prompts in cond] mean over the prompt's positions and the given layers (NaN if none defined)."""
        key = (metric, kind, tuple(layers), cond)
        if key in self._ps:
            return self._ps[key]
        vs = [self.values(metric, kind, l) for l in layers]
        out = np.full(len(self.prompts[cond]), np.nan)
        for i, pid in enumerate(self.prompts[cond]):
            js = self.prompt_pos[cond][pid]
            x = np.concatenate([v[js] for v in vs])
            x = x[~np.isnan(x)]
            if len(x):
                out[i] = x.mean()
        self._ps[key] = out
        return out


class Boot:
    """Prompt-level bootstrap indices per (condition, metric family) over the prompts with a defined value. The
    primary family uses numpy.random.default_rng([BOOT_SEED, condition index]) as pre-registered; the others add the
    family index."""

    def __init__(self, table: Table, n_boot: int = C.N_BOOT, seed: int = C.BOOT_SEED):
        self.t, self.B, self.seed = table, n_boot, seed
        self.cache: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}

    def valid(self, cond: str, fam: int) -> np.ndarray:
        # validity depends only on the targets: use the naked eye at 59 (read at every position) as the reference
        metric = "f_word" if fam == 0 else "u_next"
        s = self.t.prompt_scores(metric, "naive", [59], cond)
        return np.flatnonzero(~np.isnan(s))

    def idx(self, cond: str, fam: int) -> Tuple[np.ndarray, np.ndarray]:
        key = (cond, fam)
        if key not in self.cache:
            ok = self.valid(cond, fam)
            ci = ALL_CONDS.index(cond)
            rng = np.random.default_rng([self.seed, ci] if fam == 0 else [self.seed, ci, fam])
            ix = rng.integers(0, len(ok), size=(self.B, len(ok))) if len(ok) else np.zeros((self.B, 0), dtype=int)
            self.cache[key] = (ok, ix)
        return self.cache[key]


def ci_of(boot: np.ndarray) -> Tuple[float, float]:
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def stat(n: int, mean: Optional[float], lo: Optional[float], hi: Optional[float], **extra) -> Dict[str, Any]:
    """Rounded values for display and the unrounded ones (…_raw) that every rule compares."""
    return dict({"n": int(n), "mean": r4(mean), "lo": r4(lo), "hi": r4(hi),
                 "mean_raw": mean, "lo_raw": lo, "hi_raw": hi}, **extra)


def summarize_arrays(get_x: Callable[[str, np.ndarray], Optional[np.ndarray]], boot: Boot, fam: int,
                     conds: Tuple[str, ...] = C.CONDITIONS) -> Dict[str, Any]:
    """Mean with 95% CIs per condition (every condition) and pooled over `conds`, from per-prompt values get_x(c, ok)
    aligned with the bootstrap's valid prompts."""
    per: Dict[str, Any] = {}
    sums, ns, tot = np.zeros(boot.B), 0, 0.0
    for c in ALL_CONDS:
        ok, ix = boot.idx(c, fam)
        if not len(ok):
            per[c] = stat(0, None, None, None)
            continue
        x = get_x(c, ok)
        if x is None or np.isnan(x).any():  # a readout missing at some layer: not comparable here
            per[c] = stat(len(ok), None, None, None, note="readout missing")
            continue
        bs = x[ix].sum(1)
        lo, hi = ci_of(bs / len(ok))
        per[c] = stat(len(ok), float(x.mean()), lo, hi)
        if c in conds:
            sums += bs
            ns += len(ok)
            tot += float(x.sum())
    pooled = stat(ns, None, None, None)
    if ns:
        lo, hi = ci_of(sums / ns)
        pooled = stat(ns, tot / ns, lo, hi)
    return {"pooled": pooled, "by_cond": per}


def summarize(table: Table, boot: Boot, metric: str, a: Tuple[str, List[int]],
              b: Optional[Tuple[str, List[int]]] = None, conds: Tuple[str, ...] = C.CONDITIONS) -> Dict[str, Any]:
    """Mean (or paired difference a - b) with 95% CIs, pooled over `conds` and per condition."""
    def get_x(c, ok):
        x = table.prompt_scores(metric, a[0], a[1], c)[ok]
        if b is not None:
            x = x - table.prompt_scores(metric, b[0], b[1], c)[ok]
        return x
    return summarize_arrays(get_x, boot, FAMILY[metric], conds)


def binom_sf_half(k: int, n: int) -> float:
    """P(X >= k) for X ~ Binomial(n, 1/2), exactly."""
    if n <= 0 or k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def sign_test(table: Table, boot: Boot, lens: str, base: str, layers: List[int],
              conds: Tuple[str, ...] = C.CONDITIONS) -> Dict[str, Any]:
    """Exact one-sided sign test on the prompt-level differences (lens - base) over the primary metric's valid prompts
    of `conds`: p = P(at least n_pos successes of n_pos + n_neg fair coin flips); ties are dropped."""
    ds = []
    for c in conds:
        ok, _ = boot.idx(c, 0)
        if not len(ok):
            continue
        ds.append(table.prompt_scores(PRIMARY, lens, layers, c)[ok] - table.prompt_scores(PRIMARY, base, layers, c)[ok])
    d = np.concatenate(ds) if ds else np.zeros(0)
    if np.isnan(d).any():
        return {"n_pos": None, "n_neg": None, "n_tie": None, "p": 1.0, "note": "readout missing"}
    n_pos, n_neg = int((d > 1e-9).sum()), int((d < -1e-9).sum())
    return {"n_pos": n_pos, "n_neg": n_neg, "n_tie": int(len(d) - n_pos - n_neg), "p": binom_sf_half(n_pos, n_pos + n_neg)}


def holm(pvals: Dict[Any, float]) -> Dict[Any, float]:
    """Holm step-down adjusted p-values (monotone), over the whole family given."""
    items = sorted(pvals.items(), key=lambda kv: (kv[1], str(kv[0])))
    m, run, adj = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        adj[k] = run
    return adj


def stuck(table: Table, kind: str, layer: int, conds: Tuple[str, ...]) -> Dict[str, Any]:
    """Stuck-word share over the read positions of `conds` (rule 3): the share of positions whose filtered top-1 is the
    layer's most common filtered top-1 WORD (word_key), not counting positions where that word is the position's own
    usable content target. Also, for the record: the same share counting every position (by word and by id)."""
    rows = []
    for j in sorted({j for c in conds for j in table.cond_pos[c]}):
        r = table.pos[j]["r"].get(kind, {}).get(str(layer))
        if r is not None:
            rows.append((j, r))
    if not rows:
        return {"share": None, "share_raw": None, "n": 0}
    n = len(rows)
    keys = [C.word_key(r["fs"][0]) for _, r in rows]
    rep: Dict[str, str] = {}
    for (_, r), k in zip(rows, keys):
        rep.setdefault(k, r["fs"][0])
    cnt = collections.Counter(k for (j, _), k in zip(rows, keys) if k != table.ct_key[j])
    w, c = cnt.most_common(1)[0] if cnt else (None, 0)
    w_all, c_all = collections.Counter(keys).most_common(1)[0]
    ids_ex = collections.Counter(r["f"][0] for (j, r), k in zip(rows, keys) if k != table.ct_key[j])
    c_idx = ids_ex.most_common(1)[0][1] if ids_ex else 0
    (i_all, c_id), = collections.Counter(r["f"][0] for _, r in rows).most_common(1)
    return {"share": r4(c / n), "share_raw": c / n, "n": n, "word": rep.get(w) if w is not None else None,
            "share_by_id": r4(c_idx / n),
            "share_incl_target": r4(c_all / n), "word_incl_target": rep[w_all],
            "share_by_id_incl_target": r4(c_id / n), "token_id_incl_target": int(i_all)}


def fmt_share(table: Table, kind: str, layer: int, conds: Tuple[str, ...] = C.CONDITIONS) -> Optional[float]:
    js = sorted({j for c in conds for j in table.cond_pos[c]})
    v = table.values("u_fmt", kind, layer)[js]
    v = v[~np.isnan(v)]
    return round(float(v.mean()), 4) if len(v) else None


def band_layers_for(table: Table, kind: str, layers: Tuple[int, ...]) -> bool:
    return all(l in table.layers.get(kind, []) for l in layers)


def beats(x: Dict[str, Any]) -> bool:
    """Rule 1 on one interval: its (unrounded) lower bound is above 0."""
    return x.get("lo_raw") is not None and x["lo_raw"] > 0


def significantly_worse(x: Dict[str, Any]) -> bool:
    """Rule 2 on one interval: its (unrounded) upper bound is below 0."""
    return x.get("hi_raw") is not None and x["hi_raw"] < 0


def evaluate_band(table: Table, boot: Boot, lens: str, band: str, layers: List[int],
                  conds: Tuple[str, ...] = C.CONDITIONS) -> Dict[str, Any]:
    """Rules 1, 2, 3 and 5, and the sign tests for rule 4 (whose Holm step is applied over all bands by finalize)."""
    sec = summarize(table, boot, PRIMARY, (lens, layers), conds=conds)
    base = {b: summarize(table, boot, PRIMARY, (b, layers), conds=conds) for b, _ in BASELINES}
    diff = {b: summarize(table, boot, PRIMARY, (lens, layers), (b, layers), conds=conds) for b, _ in BASELINES}
    signs = {b: sign_test(table, boot, lens, b, layers, conds) for b, _ in BASELINES}
    st = {str(l): {c: stuck(table, lens, l, (c,)) for c in conds} for l in layers}
    st_pooled = {str(l): stuck(table, lens, l, conds) for l in layers}
    reasons: List[str] = []
    rules: Dict[str, bool] = {}
    # 1. beats both, pooled
    rules["1"] = all(beats(diff[b]["pooled"]) for b, _ in BASELINES)
    for b, name in BASELINES:
        if not beats(diff[b]["pooled"]):
            reasons.append(f"[rule 1] pooled CI of second - {name} does not lie above 0")
    # 2. not significantly worse in any decision condition (no data counts as a failure)
    worse = []
    for c in conds:
        for b, name in BASELINES:
            x = diff[b]["by_cond"].get(c, {})
            if x.get("hi_raw") is None:
                worse.append(f"{c}: no data against {name}")
            elif significantly_worse(x):
                worse.append(f"{c}: significantly worse than {name}")
    rules["2"] = not worse
    reasons += [f"[rule 2] {w}" for w in worse]
    # 3. no stuck word, in every decision condition, at every layer of the band
    bad = []
    for l in layers:
        for c in conds:
            v = st[str(l)][c]
            if v["share_raw"] is None:
                bad.append(f"{l} {c} (no data)")
            elif v["share_raw"] > C.STUCK_MAX + EPS:
                bad.append(f"{l} {c} ({v['share_raw']:.2f}, {v['word']!r})")
    rules["3"] = not bad
    if bad:
        reasons.append("[rule 3] stuck-word share above 25% at " + ", ".join(bad))
    st_max = max((v["share_raw"] for d in st.values() for v in d.values() if v["share_raw"] is not None), default=None)
    st_where = None
    if st_max is not None:
        st_where = next((l, c, v["word"]) for l, d in st.items() for c, v in d.items() if v["share_raw"] == st_max)
    # 4. sign test (p of the conjunction = the larger of the two); Holm in finalize()
    p_band = max(signs[b]["p"] for b, _ in BASELINES)
    # 5. minimum pooled effect
    rules["5"] = all(diff[b]["pooled"]["mean_raw"] is not None and diff[b]["pooled"]["mean_raw"] >= C.MIN_EFFECT - EPS
                     for b, _ in BASELINES)
    for b, name in BASELINES:
        m = diff[b]["pooled"]["mean_raw"]
        if m is None or m < C.MIN_EFFECT - EPS:
            reasons.append(f"[rule 5] pooled second - {name} is {'n/a' if m is None else f'{m:+.3f}'}, "
                           f"below the minimum {C.MIN_EFFECT}")
    return {"lens": lens, "band": band, "layers": layers, "conds": list(conds), "second": sec,
            "jpp": base["jpp"], "naive": base["naive"], "diff_jpp": diff["jpp"], "diff_naive": diff["naive"],
            "sign_jpp": signs["jpp"], "sign_naive": signs["naive"], "p_sign": p_band,
            "stuck": st, "stuck_pooled": st_pooled, "stuck_max": r4(st_max), "stuck_max_where": st_where,
            "fmt_share": {str(l): fmt_share(table, lens, l, conds) for l in layers},
            "worse_in": worse, "rules": rules, "reasons": reasons}


def finalize(bands: Dict[str, Dict[str, Dict[str, Any]]], holm_family: bool = True) -> None:
    """Rule 4 (Holm across every lens-band test in `bands`) and the verdict of each band."""
    ps = {(lens, band): ev["p_sign"] for lens, d in bands.items() for band, ev in d.items()}
    adj = holm(ps) if holm_family else ps
    for (lens, band), p in adj.items():
        ev = bands[lens][band]
        ev["p_sign_holm"] = p
        ev["holm_m"] = len(ps) if holm_family else 1
        ev["rules"]["4"] = p < C.ALPHA
        if not ev["rules"]["4"]:
            sj, sn = ev["sign_jpp"], ev["sign_naive"]
            ev["reasons"].append(f"[rule 4] sign test p = {ev['p_sign']:.3g} (J++ {sj['n_pos']}+/{sj['n_neg']}-, "
                                 f"naked {sn['n_pos']}+/{sn['n_neg']}-), Holm-adjusted {p:.3g} over "
                                 f"{ev['holm_m']} tests, not below {C.ALPHA}")
        ev["reasons"].sort()
        ev["passes"] = all(ev["rules"][k] for k in ("1", "2", "3", "4", "5"))


def decide(bands: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Eligibility (Miru at 58 reference only) and precedence (R1 where both pass)."""
    by_band: Dict[str, Optional[str]] = {}
    served: Dict[str, List[int]] = {k: [] for k in C.SECOND_LENSES}
    for band, layers in C.BANDS:
        choice = None
        for lens in C.PRECEDENCE:
            ev = bands.get(lens, {}).get(band)
            if ev is None:
                continue
            ev["eligible"] = (lens, band) not in C.REFERENCE_ONLY
            ev["served"] = False
            ev.pop("note", None)
            if ev["passes"] and ev["eligible"] and choice is None:
                choice = lens
                ev["served"] = True
            elif ev["passes"] and ev["eligible"] and choice is not None:
                ev["note"] = f"passes, but {choice} takes precedence at this band"
            elif ev["passes"] and not ev["eligible"]:
                ev["note"] = "passes, but this band is reference only for this lens"
        by_band[band] = choice
        if choice:
            served[choice] += list(layers)
    return {"by_band": by_band, "served_layers": served}


def decision_suite(data: Dict[str, Any], conds: Tuple[str, ...] = C.CONDITIONS, exclude_formatlike: bool = False,
                   n_boot: int = C.N_BOOT, seed: int = C.BOOT_SEED):
    """The whole pre-registered decision on `data`: (table, boot, bands, decision)."""
    t = Table(data, exclude_formatlike=exclude_formatlike)
    boot = Boot(t, n_boot, seed)
    bands: Dict[str, Dict[str, Any]] = {}
    for lens in C.SECOND_LENSES:
        for band, layers in C.BANDS:
            if layers and band_layers_for(t, lens, layers) and band_layers_for(t, "jpp", layers):
                bands.setdefault(lens, {})[band] = evaluate_band(t, boot, lens, band, list(layers), conds)
    finalize(bands)
    return t, boot, bands, decide(bands)


def verdicts(bands: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {f"{lens} {band}": {"passes": ev["passes"], "served": ev.get("served", False),
                               "failed_rules": [k for k, v in ev["rules"].items() if not v]}
            for lens, d in bands.items() for band, ev in d.items()}


def sensitivity(data: Dict[str, Any], base_bands, name: str, why: str, drop_pids=frozenset(),
                conds: Tuple[str, ...] = C.CONDITIONS, exclude_formatlike: bool = False,
                n_boot: int = C.N_BOOT, seed: int = C.BOOT_SEED) -> Dict[str, Any]:
    """The rule rerun on a subset (pre-registered sensitivity analyses S1-S4); any verdict that changes is listed."""
    sub = dict(data, positions=[p for p in data["positions"] if p["pid"] not in drop_pids])
    t, boot, bands, dec = decision_suite(sub, conds, exclude_formatlike, n_boot, seed)
    v0, v1 = verdicts(base_bands), verdicts(bands)
    changes = []
    for k in sorted(set(v0) | set(v1)):
        a, b = v0.get(k), v1.get(k)
        if a is None or b is None or a["passes"] != b["passes"] or a["served"] != b["served"]:
            changes.append({"lens_band": k, "primary": a, "sensitivity": b})
    dropped = sorted({p["pid"] for p in data["positions"] if p["pid"] in drop_pids})
    return {"name": name, "why": why, "conds": list(conds), "exclude_formatlike": exclude_formatlike,
            "prompts_dropped": len(dropped), "pids_dropped": dropped,
            "prompts_valid_primary": {c: int(len(boot.idx(c, 0)[0])) for c in conds},
            "positions_formatlike_excluded": int(sum(1 for j in range(t.N) if exclude_formatlike and t.formatlike[j]
                                                     and any(j in t.cond_pos[c] for c in conds))),
            "decision": dec, "verdicts": v1, "changes": changes,
            "bands": {lens: {band: {k: ev[k] for k in ("passes", "rules", "reasons", "p_sign", "p_sign_holm")}
                             | {"diff_jpp": ev["diff_jpp"]["pooled"], "diff_naive": ev["diff_naive"]["pooled"]}
                             for band, ev in d.items()} for lens, d in bands.items()}}


def constant_guess(table: Table, boot: Boot, conds: Tuple[str, ...] = C.CONDITIONS) -> Dict[str, Any]:
    """A readout that always names one word: for each prompt, the most common usable content-target word among the
    OTHER prompts of its condition (leave-one-out; ties to the alphabetically first word), scored like the primary
    metric. Reported only."""
    modes: Dict[str, Any] = {}

    def get_x(c, ok):
        pids = table.prompts[c]
        keys = [[table.ct_key[j] for j in table.prompt_pos[c][pid] if table.ct_key[j] is not None] for pid in pids]
        total = collections.Counter(k for ks in keys for k in ks)
        if total:
            w, n = sorted(total.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            modes[c] = {"word": w, "count": n, "of": sum(total.values())}
        out = np.full(len(pids), np.nan)
        for i, ks in enumerate(keys):
            if not ks:
                continue
            loo = total.copy()
            loo.subtract(ks)
            loo = +loo
            if not loo:
                out[i] = 0.0
                continue
            guess = sorted(loo.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            out[i] = float(np.mean([k == guess for k in ks]))
        return out[ok]
    s = summarize_arrays(get_x, boot, 0, conds)
    s["most_common_target"] = modes
    return s


def function_word_split(table: Table, kind: str, layers: List[int], conds: Tuple[str, ...]) -> Dict[str, Any]:
    """Position-level primary metric (band mean) split by whether the content target is a function word."""
    v = table.band_values(PRIMARY, kind, layers)
    out: Dict[str, Any] = {}
    groups = {c: sorted(table.cond_pos[c]) for c in conds}
    groups["pooled"] = sorted({j for c in conds for j in table.cond_pos[c]})
    for g, js in groups.items():
        row = {}
        for name, want in (("function", True), ("other", False)):
            sel = [j for j in js if table.ct_key[j] is not None and (table.ct_key[j] in C.FUNCTION_WORDS) == want
                   and not np.isnan(v[j])]
            row[name] = {"n": len(sel), "mean": r4(float(np.mean(v[sel]))) if sel else None}
        out[g] = row
    return out


def analyze(data: Dict[str, Any], n_boot: int = C.N_BOOT, seed: int = C.BOOT_SEED) -> Dict[str, Any]:
    t, boot, bands, dec = decision_suite(data, C.CONDITIONS, False, n_boot, seed)
    out: Dict[str, Any] = {"schema": "R2_03/2", "bootstrap": {"n": n_boot, "seed": seed, "ci": "95% percentile"},
                           "constants": {"STUCK_MAX": C.STUCK_MAX, "MIN_EFFECT": C.MIN_EFFECT, "ALPHA": C.ALPHA,
                                         "COLLAPSE_TARGETS": list(C.COLLAPSE_TARGETS)}}
    # counts and context
    counts = {}
    for c in ALL_CONDS:
        js = t.cond_pos[c]
        counts[c] = {"prompts": len(t.prompts[c]), "positions": len(js),
                     "positions_with_content_target": sum(1 for j in js if t.raw_ct_key[j] is not None),
                     "positions_collapse_target_excluded": sum(1 for j in js if t.collapse[j]),
                     "positions_usable_target": sum(1 for j in js if t.ct_key[j] is not None),
                     "positions_formatlike_target": sum(1 for j in js if t.formatlike[j]),
                     "prompts_valid_primary": int(len(boot.idx(c, 0)[0]))}
    out["counts"] = counts
    ctx: Dict[str, Any] = {}
    for c in ALL_CONDS:
        js = t.cond_pos[c]
        if not js:
            continue
        agree = float(np.mean([bool(t.pos[j].get("model_agrees")) for j in js]))
        offs = [t.pos[j]["ct_offset"] for j in js if t.pos[j].get("ct_offset") is not None]
        ctx[c] = {
            "next_is_noncontent": round(float(np.mean([not t.pos[j]["next_content"] for j in js])), 4),
            "ct_offset_mean": round(float(np.mean(offs)), 3) if offs else None,
            "model_agrees_with_next": round(agree, 4),
            "agreement_below_0.95": agree < 0.95,
        }
    out["context"] = ctx
    if 59 in t.layers.get("naive", []):
        out["context"]["model_ceiling_filtered_59"] = summarize(t, boot, PRIMARY, ("naive", [59]))
    out["context"]["constant_guess"] = constant_guess(t, boot)
    # per layer
    per_layer: Dict[str, Dict[str, Any]] = {}
    for k in KINDS:
        for l in t.layers.get(k, []):
            row = {
                "f_word": summarize(t, boot, "f_word", (k, [l])),
                "f_id": summarize(t, boot, "f_id", (k, [l]))["pooled"],
                "f5_word": summarize(t, boot, "f5_word", (k, [l]))["pooled"],
                "u_next": summarize(t, boot, "u_next", (k, [l]))["pooled"],
                "u_word_ct": summarize(t, boot, "u_word_ct", (k, [l]))["pooled"],
                "fmt_share": fmt_share(t, k, l),
                "fmt_share_by_cond": {c: fmt_share(t, k, l, (c,)) for c in ALL_CONDS if t.cond_pos[c]},
                "stuck": stuck(t, k, l, C.CONDITIONS),
                "stuck_by_cond": {c: stuck(t, k, l, (c,)) for c in ALL_CONDS if t.cond_pos[c]},
            }
            if k in C.SECOND_LENSES and l in t.layers.get("jpp", []):
                row["diff_jpp"] = summarize(t, boot, "f_word", (k, [l]), ("jpp", [l]))["pooled"]
                row["diff_naive"] = summarize(t, boot, "f_word", (k, [l]), ("naive", [l]))["pooled"]
            per_layer.setdefault(k, {})[str(l)] = row
    out["per_layer"] = per_layer
    out["bands"] = bands
    out["decision"] = dec
    # secondary (not used for the decision)
    sec: Dict[str, Any] = {}
    for name, metric in (("unfiltered_next_token", "u_next"), ("unfiltered_vs_content_target", "u_word_ct"),
                         ("filtered_exact_id", "f_id"), ("filtered_top5_word", "f5_word")):
        sec[name] = {}
        for lens in C.SECOND_LENSES:
            for band, layers in C.BANDS:
                if band_layers_for(t, lens, layers) and band_layers_for(t, "jpp", layers):
                    L = list(layers)
                    sec[name].setdefault(lens, {})[band] = {
                        "second": summarize(t, boot, metric, (lens, L))["pooled"],
                        "jpp": summarize(t, boot, metric, ("jpp", L))["pooled"],
                        "naive": summarize(t, boot, metric, ("naive", L))["pooled"],
                        "diff_jpp": summarize(t, boot, metric, (lens, L), ("jpp", L))["pooled"],
                        "diff_naive": summarize(t, boot, metric, (lens, L), ("naive", L))["pooled"],
                    }
    # mid-reply-content-next: the same rule's numbers on the "content_next" draw alone (no Holm family)
    sec["mid_reply_content_next"] = {}
    if t.cond_pos["mid-reply-content-next"]:
        cn_bands: Dict[str, Dict[str, Any]] = {}
        for lens in C.SECOND_LENSES:
            for band, layers in C.BANDS:
                if band_layers_for(t, lens, layers) and band_layers_for(t, "jpp", layers):
                    cn_bands.setdefault(lens, {})[band] = evaluate_band(t, boot, lens, band, list(layers),
                                                                        conds=("mid-reply-content-next",))
        finalize(cn_bands, holm_family=False)
        for lens, d in cn_bands.items():
            for band, ev in d.items():
                sec["mid_reply_content_next"].setdefault(lens, {})[band] = {
                    k: ev[k] for k in ("second", "jpp", "naive", "diff_jpp", "diff_naive", "sign_jpp", "sign_naive",
                                       "stuck_max", "passes", "reasons")}
    # split by the next token's kind (positions of the four decision conditions; position-weighted, no CIs)
    split = {}
    js_all = sorted({j for c in C.CONDITIONS for j in t.cond_pos[c]})
    for name, want in (("next_is_noncontent", False), ("next_is_content", True)):
        js = [j for j in js_all if bool(t.pos[j]["next_content"]) == want]
        row: Dict[str, Any] = {"n_positions": len(js)}
        for k in KINDS:
            for l in t.layers.get(k, []):
                v = t.values("f_word", k, l)[js]
                v = v[~np.isnan(v)]
                row.setdefault(k, {})[str(l)] = round(float(v.mean()), 4) if len(v) else None
        split[name] = row
    sec["split_by_next_token"] = split
    # the primary metric split by function-word targets, per lens-band, for the second reading and both baselines
    fw: Dict[str, Any] = {}
    for lens, d in bands.items():
        for band, ev in d.items():
            fw.setdefault(lens, {})[band] = {k: function_word_split(t, k, ev["layers"], C.CONDITIONS)
                                             for k in (lens, "jpp", "naive")}
    sec["function_word_split"] = fw
    # Miru - R1 where both read
    mr = {}
    for band, layers in C.BANDS:
        if band_layers_for(t, "r1", layers) and band_layers_for(t, "miru", layers):
            mr[band] = summarize(t, boot, PRIMARY, ("miru", list(layers)), ("r1", list(layers)))["pooled"]
    sec["miru_minus_r1"] = mr
    out["secondary"] = sec
    # pre-registered sensitivity analyses (reported; the primary decision above stands)
    about = frozenset(p["pid"] for p in data.get("prompts", []) if p.get("about_model"))
    out["sensitivity"] = {
        "S1_without_about_model": sensitivity(
            data, bands, "S1", "without the prompts about the model itself (R1's fit set had many self-questions)",
            drop_pids=about, n_boot=n_boot, seed=seed),
        "S2_without_r1_overlap": sensitivity(
            data, bands, "S2", "without the prompts that paraphrase or share an answer topic with R1's fit prompts "
            "(R2_common.R1_OVERLAP)", drop_pids=frozenset(C.R1_OVERLAP), n_boot=n_boot, seed=seed),
        "S3_without_bare": sensitivity(
            data, bands, "S3", "without the bare condition (raw text puts this instruct model off-distribution)",
            conds=tuple(c for c in C.CONDITIONS if c != "bare"), n_boot=n_boot, seed=seed),
        "S4_without_formatlike_targets": sensitivity(
            data, bands, "S4", "without positions whose content target is a list numeral after a newline or the "
            "token after a code fence", exclude_formatlike=True, n_boot=n_boot, seed=seed),
    }
    return out


# ------------------------------------------------------------------------------------------------ report text
def fmt_ci(d: Dict[str, Any], signed: bool = True) -> str:
    if d.get("mean") is None:
        return "n/a"
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(d['mean'])} [{f.format(d['lo'])}, {f.format(d['hi'])}]"


def fmt_num(x: Optional[float], f: str = "{:.3f}") -> str:
    return "n/a" if x is None else f.format(x)


def runs_record() -> Dict[str, Any]:
    """The GPU runs recorded in R2_02_runs.jsonl (every one is reported)."""
    if not C.RUNS_LOG.is_file():
        return {"n": 0, "events": []}
    ev = []
    for line in C.RUNS_LOG.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        ev.append({k: d.get(k) for k in ("event", "at", "host", "out_sha256", "error", "forced_overwrite",
                                         "allow_uncommitted")})
    return {"n": sum(1 for e in ev if e["event"] == "started"), "events": ev}


def report_md(an: Dict[str, Any], meta: Dict[str, Any], integrity: Dict[str, Any]) -> str:
    L: List[str] = []
    selftest = bool(meta.get("selftest"))
    L.append("## R2 result: the filtered second reading, layer band by layer band")
    L.append("")
    if selftest:
        L.append("**SELF-TEST DATA (fake model): these numbers mean nothing.**")
        L.append("")
    dec = an["decision"]
    L.append("Applied exactly as pre-registered (R2_PREREG.md §7). A lens is served at a band only if all five rules "
             "hold: (1) pooled over the four conditions, its filtered top word names the model's next content word "
             "more often than both filtered J++ and the filtered naked eye (95% prompt-bootstrap intervals above 0); "
             "(2) it is significantly worse than neither in any single condition; (3) its stuck-word share is at most "
             "25% in every condition at every layer of the band; (4) the exact sign test over prompts, against both "
             "baselines, is below 0.025 after Holm's adjustment across the lens-band tests; (5) both pooled "
             "differences are at least 0.05. Layers count from 0 (the pages show layer + 1).")
    L.append("")
    L.append("| band (code) | R1's lens | Miru | served |")
    L.append("|---|---|---|---|")
    for band, _ in C.BANDS:
        cells = []
        for lens in C.SECOND_LENSES:
            ev = an["bands"].get(lens, {}).get(band)
            if ev is None:
                cells.append("not read")
            else:
                failed = [k for k in ("1", "2", "3", "4", "5") if not ev["rules"].get(k)]
                cells.append(ev["note"] if ev.get("note") else
                             ("passes" if ev["passes"] else "fails rule " + ", ".join(failed)))
        L.append(f"| {band} | {cells[0]} | {cells[1]} | {dec['by_band'].get(band) or 'none'} |")
    L.append("")
    sl = dec["served_layers"]
    L.append(f"**Served:** R1's lens at {sl.get('r1') or 'no layer'}; Miru at {sl.get('miru') or 'no layer'}.")
    L.append("")
    L.append("Pooled filtered top-1 (the share of a prompt's positions where the readout's top content word is the "
             "model's next content word, averaged over prompts), the paired differences, and the sign test's "
             "discordant prompts (+ the second reading better, - worse):")
    L.append("")
    L.append("| lens | band | second | J++ | naked | second - J++ | second - naked | +/- vs J++ | +/- vs naked | "
             "p (Holm) | worse in | max stuck (layer, condition, word) | verdict |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for lens in C.SECOND_LENSES:
        for band, _ in C.BANDS:
            ev = an["bands"].get(lens, {}).get(band)
            if ev is None:
                continue
            worse = "; ".join(w.split(":")[0] + (" (J++)" if "J++" in w else " (naked)") for w in ev["worse_in"]) or "-"
            verdict = "SERVE" if ev.get("served") else ("pass, not served" if ev["passes"] else "do not serve")
            sj, sn = ev["sign_jpp"], ev["sign_naive"]
            where = ev.get("stuck_max_where")
            stw = f"{ev['stuck_max']:.2f} ({where[0]}, {where[1]}, {where[2]!r})" if where else "n/a"
            L.append(f"| {lens} | {band} | {fmt_num(ev['second']['pooled']['mean'])} | "
                     f"{fmt_num(ev['jpp']['pooled']['mean'])} | {fmt_num(ev['naive']['pooled']['mean'])} | "
                     f"{fmt_ci(ev['diff_jpp']['pooled'])} | {fmt_ci(ev['diff_naive']['pooled'])} | "
                     f"{sj['n_pos']}/{sj['n_neg']} | {sn['n_pos']}/{sn['n_neg']} | {ev['p_sign_holm']:.2g} | "
                     f"{worse} | {stw} | {verdict} |")
    L.append("")
    fails = [(lens, band, ev) for lens in C.SECOND_LENSES for band, ev in an["bands"].get(lens, {}).items()
             if not ev["passes"]]
    if fails:
        L.append("Why each failing band fails:")
        for lens, band, ev in fails:
            L.append(f"- {lens} {band}: " + "; ".join(ev["reasons"]))
        L.append("")
    L.append("Sensitivity analyses (pre-registered; reported beside the decision above, which stands):")
    L.append("")
    L.append("| analysis | what is left out | served | verdicts that change |")
    L.append("|---|---|---|---|")
    for key, s in an.get("sensitivity", {}).items():
        left = (f"{s['prompts_dropped']} prompts" if s["prompts_dropped"] else
                (f"{s['positions_formatlike_excluded']} positions" if s["exclude_formatlike"] else
                 "the bare condition" if "bare" not in s["conds"] else "-"))
        sl2 = s["decision"]["served_layers"]
        ch = "; ".join(
            f"{c['lens_band']}: {'pass' if (c['primary'] or {}).get('passes') else 'fail'}"
            f"{'/served' if (c['primary'] or {}).get('served') else ''} -> "
            f"{'pass' if (c['sensitivity'] or {}).get('passes') else 'fail'}"
            f"{'/served' if (c['sensitivity'] or {}).get('served') else ''}" for c in s["changes"]) or "none"
        L.append(f"| {s['name']} ({s['why']}) | {left} | R1 {sl2.get('r1') or '-'}; Miru {sl2.get('miru') or '-'} "
                 f"| {ch} |")
    L.append("")
    L.append("For transparency (not used for the decision): the unfiltered top word as the model's next token "
             "(the earlier test's metric) and the share of positions whose unfiltered top word is formatting.")
    L.append("")
    L.append("| lens | band | unfiltered top-1 = next token: second / J++ / naked | formatting share of the second (unfiltered), by layer |")
    L.append("|---|---|---|---|")
    un = an["secondary"]["unfiltered_next_token"]
    for lens in C.SECOND_LENSES:
        for band, _ in C.BANDS:
            u = un.get(lens, {}).get(band)
            ev = an["bands"].get(lens, {}).get(band)
            if u is None or ev is None:
                continue
            fs = ", ".join(f"{l}: {v:.2f}" for l, v in ev["fmt_share"].items() if v is not None)
            L.append(f"| {lens} | {band} | {fmt_num(u['second']['mean'])} / {fmt_num(u['jpp']['mean'])} / "
                     f"{fmt_num(u['naive']['mean'])} | {fs} |")
    L.append("")
    ctx = an["context"]
    L.append("Context:")
    for c in ALL_CONDS:
        if c in ctx:
            x = ctx[c]
            n = an["counts"][c]
            L.append(f"- {c}: {n['prompts']} prompts, {n['positions']} positions "
                     f"({n['positions_with_content_target']} with a content target, of which "
                     f"{n['positions_collapse_target_excluded']} left out as raw-text collapse words; "
                     f"{n['positions_formatlike_target']} formatting-like targets); the model's next token is "
                     f"formatting at {x['next_is_noncontent']:.0%}; reading forward agrees with the target's next "
                     f"token at {x['model_agrees_with_next']:.1%}" + (" (BELOW 0.95)" if x["agreement_below_0.95"] else "") + ".")
    if "model_ceiling_filtered_59" in ctx:
        L.append(f"- The model's own filtered top word at 59 is its next content word at "
                 f"{fmt_num(ctx['model_ceiling_filtered_59']['pooled']['mean'])} (pooled): the ceiling a "
                 "same-step readout can reach.")
    cg = ctx.get("constant_guess")
    if cg:
        L.append("- A constant guess (each condition's most common target word, leave-one-out) scores "
                 f"{fmt_ci(cg['pooled'], signed=False)} pooled; by condition: " + "; ".join(
                     f"{c} {fmt_num(cg['by_cond'][c]['mean'])} ({cg['most_common_target'].get(c, {}).get('word')!r})"
                     for c in C.CONDITIONS if c in cg["by_cond"]) + ".")
    fw = an["secondary"].get("function_word_split", {})
    if fw:
        parts = []
        for lens, d in fw.items():
            for band, v in d.items():
                s, j = v[lens]["pooled"], v["jpp"]["pooled"]
                parts.append(f"{lens} {band}: function words {fmt_num(s['function']['mean'])} vs J++ "
                             f"{fmt_num(j['function']['mean'])} (n {s['function']['n']}), other "
                             f"{fmt_num(s['other']['mean'])} vs {fmt_num(j['other']['mean'])} (n {s['other']['n']})")
        L.append("- Split by function-word targets (position-level, pooled): " + "; ".join(parts) + ".")
    if an["secondary"].get("miru_minus_r1"):
        L.append("- Miru - R1 where both read (filtered, pooled): " + "; ".join(
            f"{b}: {fmt_ci(d)}" for b, d in an["secondary"]["miru_minus_r1"].items()) + ".")
    L.append("- Power: pooled intervals resolve differences of about 0.06-0.07; per-condition intervals (40-60 "
             "prompts) are about ±0.12-0.17 wide, so rule 2 catches only large harms (R2_PREREG.md §7).")
    L.append("")
    files = integrity.get("files", {})
    bad = [k for k, v in files.items() if not v.get("same")]
    if not files or all(v.get("run") is None for v in files.values()):
        L.append("Integrity: the run recorded no file hashes.")
    elif not bad:
        L.append("Integrity: the pre-registration, prompts, scripts and freshness record on disk are those the GPU "
                 "run recorded.")
    else:
        L.append(f"Integrity: CHANGED SINCE THE RUN: {', '.join(bad)} (a deviation; see R2_PREREG.md).")
    g = (meta.get("git") or {})
    if g:
        L.append(f"Git: the run was made at HEAD {str(g.get('head'))[:12]}"
                 + (f"; uncommitted then: {g['dirty']} (a deviation)" if g.get("dirty") else
                    " with every pre-registered file committed") + ".")
    runs = integrity.get("runs")
    if runs is not None and not selftest:
        L.append(f"Runs: {runs['n']} GPU run(s) recorded in R2_02_runs.jsonl"
                 + ("" if runs["n"] <= 1 else " (MORE THAN ONE: each is reported in the write-up)") + ": "
                 + "; ".join(f"{e['event']} {e['at']}" for e in runs["events"]) + ".")
    L.append(f"Bootstrap: {an['bootstrap']['n']} prompt-level resamples, seed {an['bootstrap']['seed']}. Per-condition "
             "intervals, per-layer tables, exact-id and top-5 variants, the mid-reply content-next draw, the split by "
             "the next token's kind and the sensitivity details are in R2_03_analysis.json.")
    L.append("")
    return "\n".join(L)


def integrity_of(meta: Dict[str, Any]) -> Dict[str, Any]:
    rec = {}
    for name, p in C.PREREG_FILES.items():
        run = (meta.get("files_sha256") or {}).get(name)
        cur = C.sha256_file(p) if p.is_file() else None
        rec[name] = {"run": run, "now": cur, "same": run is not None and run == cur}
    return {"files": rec, "runs": runs_record()}


# ------------------------------------------------------------------------------------------------ self-test
PREFIX = {"chat-no-answer": "cna", "chat-begun": "cb", "bare": "bare", "mid-reply": "mid"}
# tokens of the synthetic vocabulary with fixed meanings
T_KIND, T_KIND_CAP, T_OWN, T_ONE, T_THE, T_FMT = 1638, 1639, 1700, 1800, 2000, 5213
# r1 at 55: the base pattern plus 10 extra favourable prompts (a small, consistent win: rule 5 fails, 10/211 < 0.05)
R1_55_EXTRA = {("chat-no-answer", 5), ("chat-no-answer", 15), ("chat-no-answer", 25), ("chat-begun", 3),
               ("chat-begun", 13), ("bare", 3), ("bare", 13), ("bare", 23), ("mid-reply", 3), ("mid-reply", 13)}


def synthetic(seed: int = 7) -> Tuple[Dict[str, Any], Dict[Tuple[str, str], bool], Dict[str, Any]]:
    """A results file in R2_02's shape with known answers. Hit patterns are deterministic (prompt index modulo 10),
    so every verdict below is far from its threshold. Expected verdicts (lens, band) -> served, and other facts."""
    rng = np.random.default_rng(seed)
    n = {"chat-no-answer": 60, "chat-begun": 60, "bare": 60, "mid-reply": 40}

    def hit(kind: str, layer: int, cond: str, i: int) -> bool:
        r = i % 10
        if kind == "r1":
            if layer == 50:
                return r < 8                              # clear win -> served
            if layer == 55:
                return r < 3 or (cond, i) in R1_55_EXTRA  # small win: fails rule 5 only
            if layer == 58:
                return r < 9                              # better, but stuck in chat-begun (below) -> fails rule 3
        if kind == "miru":
            if 40 <= layer <= 49:
                return False if cond == "chat-begun" else r < 9   # pooled win, worse in chat-begun -> rule 2
            if layer in (50, 55, 56, 58):
                return r < 8          # 50 loses to R1 by precedence; 55 served (R1 fails there); 58 reference only
            if 51 <= layer <= 54:
                # wins by a whole band on 30% of prompts, loses one layer of four on another 30%: the mean is
                # clearly positive but the signs split evenly -> fails rule 4 only
                return 3 <= r <= 5 or (r < 3 and layer != 51)
            if layer == 57:
                return r < 6                              # 6-9 stuck on " kind" / "Kind" (below) -> rule 3
        if kind == "naive" and layer == 59:
            return True
        return r < 3

    def stuck_word(kind: str, layer: int, cond: str, i: int):
        r = i % 10
        if kind == "r1" and layer == 58 and cond == "chat-begun" and r >= 4:
            return T_KIND, " kind"                        # 60% of chat-begun only (12% pooled)
        if kind == "miru" and layer == 57 and r in (6, 7):
            return T_KIND, " kind"                        # 20% by id ...
        if kind == "miru" and layer == 57 and r in (8, 9):
            return T_KIND_CAP, "Kind"                     # ... and 20% by id: 40% by word
        return None

    positions = []
    prompts = []
    for cond, cnt in n.items():
        for i in range(cnt):
            pid = f"{PREFIX[cond]}-{i + 1:02d}"
            prompts.append({"pid": pid, "cond": cond, "about_model": i % 10 == 9})
            ks = [None] if cond != "mid-reply" else [3, 9, 15, 21]
            for q, k in enumerate(ks):
                draws = None
                if cond == "mid-reply":
                    draws = [["any"], ["any"], ["content_next", "any"], ["content_next"]][q]
                tid = int(rng.integers(2100, 4000))
                ct_str = f" w{tid}".strip().upper() if i % 3 == 0 else f" w{tid}"
                prev = " x"
                if cond == "chat-no-answer" and i % 10 < 5:
                    tid, ct_str = T_THE, "The"            # a common opener: correct "The" is not "stuck"
                if cond == "bare" and i % 10 == 8:
                    tid, ct_str = T_OWN, " own"           # a raw-text collapse word: left out
                if cond == "chat-begun" and i % 10 == 9:
                    tid, ct_str, prev = T_ONE, "1", "\n\n"  # a list numeral after a newline: formatting-like
                no_target = cond == "bare" and i % 20 == 19   # 3 bare prompts without a content target
                ct_id = None if no_target else tid
                word = {T_THE: "The", T_OWN: " own", T_ONE: "1"}.get(tid, f" w{tid}")
                p = {"i": len(positions), "pid": pid, "cond": cond, "draws": draws, "k": k, "pos": 10 + (k or 0),
                     "read_tok": prev, "next_id": tid if i % 2 else T_FMT, "next_str": word if i % 2 else " **",
                     "next_content": bool(i % 2), "cont_ids": [], "cont_strs": [],
                     "ct_id": ct_id, "ct_str": None if no_target else ct_str,
                     "ct_offset": None if no_target else 0, "ct_prev_str": None if no_target else prev,
                     "model_top1_id": tid if i % 2 else T_FMT, "model_agrees": True, "r": {}}
                for kind in KINDS:
                    for l in C.LENS_LAYERS[kind]:
                        sw = stuck_word(kind, l, cond, i)
                        if sw is not None:
                            ftop, fstr = sw
                        elif hit(kind, l, cond, i) and not no_target:
                            ftop, fstr = tid, word
                        else:
                            other = 6000 + (len(positions) * 131 + l * 7 + KINDS.index(kind) * 1009) % 3000
                            ftop, fstr = other, f" v{other}"
                        fmt = kind in ("r1", "miru") and i % 4 == 0      # unfiltered top is " **" at 25%
                        u0, us0 = (T_FMT, " **") if fmt else (ftop, fstr)
                        p["r"].setdefault(kind, {})[str(l)] = {
                            "u": [u0, 1, 2, 3, 4], "us": [us0, "a", "b", "c", "d"],
                            "f": [ftop, 6, 7, 8, 9], "fs": [fstr, "e", "f", "g", "h"],
                            "ur": 1 if u0 == p["next_id"] else 7, "fr": None if no_target else (1 if ftop == tid else 4),
                            "u0c": not fmt,
                        }
                positions.append(p)
    data = {"schema": "R2_02/1", "meta": {"selftest": True}, "prompts": prompts, "positions": positions}
    expected_served = {("r1", "50"): True, ("r1", "55"): False, ("r1", "58"): False,
                       ("miru", "40-49"): False, ("miru", "50"): False, ("miru", "51-54"): False,
                       ("miru", "55"): True, ("miru", "56-57"): False, ("miru", "58"): False}
    expected_pass = {("r1", "50"): True, ("r1", "55"): False, ("r1", "58"): False,
                     ("miru", "40-49"): False, ("miru", "50"): True, ("miru", "51-54"): False, ("miru", "55"): True,
                     ("miru", "56-57"): False, ("miru", "58"): True}
    expected_failed_rules = {("r1", "55"): ["5"], ("r1", "58"): ["3"], ("miru", "40-49"): ["2"],
                             ("miru", "51-54"): ["4"], ("miru", "56-57"): ["3"]}
    facts = {"expected_pass": expected_pass, "expected_failed_rules": expected_failed_rules}
    return data, expected_served, facts


def selftest() -> int:
    fails = []
    # units: R2_common's target search, formatting-like targets, the filter's new invisible characters
    # ids 0-9 (none of them a stop id but 1): 2, 3, 5 and 8 are content, 4 is special
    is_content = np.isin(np.arange(10), [2, 3, 5, 8])
    is_special = np.arange(10) == 4
    units = [
        ("first_content skips non-content", C.first_content([0, 6, 3], is_content, is_special) == (2, 3)),
        ("first_content stops at a special token", C.first_content([0, 4, 2], is_content, is_special) is None),
        ("first_content without is_special reads past it", C.first_content([0, 4, 2], is_content) == (2, 2)),
        ("first_content stops at a stop id", C.first_content([0, C.STOP_IDS[0], 2], np.arange(200) > 0,
                                                             np.zeros(200, bool)) is None
         and C.first_content([0, 2, C.STOP_IDS[1]], np.arange(200) > 0) == (1, 2)),
        ("list numeral after newline is formatting-like", C.formatlike_target("1", "\n\n")
         and C.formatlike_target(" 12", ":\n") and not C.formatlike_target("1", " ")),
        ("token after a code fence is formatting-like", C.formatlike_target("python", "```")
         and not C.formatlike_target(" Paris", "\n")),
        ("invisible tokens are NON-CONTENT", C.classify_text("\x00") == "punct_symbol"
         and C.classify_text(" ​") == "punct_symbol" and C.classify_text("✔️") == "punct_symbol"
         and C.classify_text("​the") == "content" and C.classify_text(" Paris") == "content"),
        ("rule comparisons use unrounded bounds", significantly_worse({"hi": -0.0, "hi_raw": -3e-5})
         and beats({"lo": 0.0, "lo_raw": 3e-5}) and not beats({"lo": 0.0001, "lo_raw": 0.0})),
        ("exact sign test", abs(binom_sf_half(4, 4) - 0.0625) < 1e-12 and abs(binom_sf_half(5, 5) - 0.03125) < 1e-12
         and binom_sf_half(0, 0) == 1.0),
        ("Holm adjustment", all(abs(holm({"a": 0.01, "b": 0.04, "c": 0.03})[k] - v) < 1e-12
                                for k, v in {"a": 0.03, "c": 0.06, "b": 0.06}.items())),
    ]
    for name, ok in units:
        if not ok:
            fails.append("unit check failed: " + name)

    data, expected, facts = synthetic()
    an = analyze(data)
    for (lens, band), want in expected.items():
        ev = an["bands"][lens][band]
        if ev["served"] != want:
            fails.append(f"{lens} {band}: served {ev['served']}, expected {want}; reasons {ev['reasons']}")
        if ev["passes"] != facts["expected_pass"][(lens, band)]:
            fails.append(f"{lens} {band}: passes {ev['passes']}, expected {facts['expected_pass'][(lens, band)]}; "
                         f"reasons {ev['reasons']}")
        got_failed = sorted(k for k, v in ev["rules"].items() if not v)
        if got_failed != facts["expected_failed_rules"].get((lens, band), []):
            fails.append(f"{lens} {band}: failed rules {got_failed}, expected "
                         f"{facts['expected_failed_rules'].get((lens, band), [])}; reasons {ev['reasons']}")
    r = an["bands"]
    n_valid = 60 + 60 + 51 + 40
    s1 = an["sensitivity"]["S1_without_about_model"]
    checks = [
        ("r1 55: small win passes rule 1 and the sign test (10+/0-)",
         r["r1"]["55"]["rules"]["1"] and r["r1"]["55"]["rules"]["4"] and r["r1"]["55"]["sign_jpp"]["n_pos"] == 10
         and r["r1"]["55"]["sign_jpp"]["n_neg"] == 0),
        ("r1 55: pooled difference is 10/211", abs(r["r1"]["55"]["diff_jpp"]["pooled"]["mean_raw"] - 10 / n_valid) < 1e-9),
        ("r1 58 stuck 0.60 in chat-begun", abs(r["r1"]["58"]["stuck"]["58"]["chat-begun"]["share_raw"] - 0.6) < 1e-9),
        ("r1 58 pooled stuck share is under 0.25 (the old pooled rule would pass)",
         r["r1"]["58"]["stuck_pooled"]["58"]["share_raw"] < 0.25),
        ("miru 40-49 is worse in chat-begun only", r["miru"]["40-49"]["worse_in"]
         and all(w.startswith("chat-begun") for w in r["miru"]["40-49"]["worse_in"])),
        ("miru 40-49 still beats both pooled", r["miru"]["40-49"]["diff_jpp"]["pooled"]["lo_raw"] > 0
         and r["miru"]["40-49"]["diff_naive"]["pooled"]["lo_raw"] > 0),
        ("miru 51-54: mean clearly positive but signs split", r["miru"]["51-54"]["diff_jpp"]["pooled"]["mean_raw"] > 0.15
         and r["miru"]["51-54"]["sign_jpp"]["n_neg"] > 50 and r["miru"]["51-54"]["p_sign"] > 0.1),
        ("miru 56-57 stuck at 57 by word (0.40) though at most 0.25 by id",
         all(abs(v["share_raw"] - 0.4) < 1e-9 and v["share_by_id"] <= 0.25
             for v in r["miru"]["56-57"]["stuck"]["57"].values())),
        ("a correct common opener is not stuck: miru 55 in chat-no-answer",
         r["miru"]["55"]["stuck"]["55"]["chat-no-answer"]["share_incl_target"] >= 0.5
         and r["miru"]["55"]["stuck"]["55"]["chat-no-answer"]["share_raw"] < 0.25),
        ("miru 50 loses to R1 by precedence", "precedence" in r["miru"]["50"].get("note", "")),
        ("miru 58 reference only", "reference only" in r["miru"]["58"].get("note", "")),
        ("served layers", an["decision"]["served_layers"] == {"r1": [50], "miru": [55]}),
        ("Holm family is the 9 lens-band tests", all(ev["holm_m"] == 9 for d in r.values() for ev in d.values())),
        ("r1 50 pooled = 176/211 (case and space ignored; collapse and untargeted positions left out)",
         abs(r["r1"]["50"]["second"]["pooled"]["mean_raw"] - 176 / n_valid) < 1e-9),
        ("bare valid prompts exclude 3 untargeted and 6 collapse-target prompts",
         an["counts"]["bare"]["prompts_valid_primary"] == 51
         and an["counts"]["bare"]["positions_collapse_target_excluded"] == 6),
        ("formatting-like targets counted from the preceding token", an["counts"]["chat-begun"]["positions_formatlike_target"] == 6),
        ("mid-reply uses only the 'any' draw (3 per prompt)", an["counts"]["mid-reply"]["positions"] == 120),
        ("mid-reply-content-next has 2 per prompt", an["counts"]["mid-reply-content-next"]["positions"] == 80),
        ("formatting share of second is 0.25", abs(an["per_layer"]["r1"]["50"]["fmt_share"] - 0.25) < 0.02),
        ("unfiltered J++ hits the next token only on content-next positions",
         an["secondary"]["unfiltered_next_token"]["r1"]["50"]["jpp"]["mean"] < 0.3),
        ("constant guess in chat-no-answer is 'the' at 0.5",
         abs(an["context"]["constant_guess"]["by_cond"]["chat-no-answer"]["mean_raw"] - 0.5) < 1e-9
         and an["context"]["constant_guess"]["most_common_target"]["chat-no-answer"]["word"] == "the"),
        ("function-word split counts the 30 'the' targets",
         an["secondary"]["function_word_split"]["r1"]["50"]["r1"]["chat-no-answer"]["function"]["n"] == 30),
        ("S1 drops the 22 prompts about the model", s1["prompts_dropped"] == 22),
        ("S1 flips r1 55 (10/192 >= 0.05) and so Miru 55 is no longer served",
         {c["lens_band"] for c in s1["changes"]} == {"r1 55", "miru 55"}
         and s1["decision"]["served_layers"] == {"r1": [50, 55], "miru": []}),
        ("S2 drops the R1-overlap prompts", an["sensitivity"]["S2_without_r1_overlap"]["prompts_dropped"] == len(C.R1_OVERLAP)),
        ("S2, S3, S4 change no verdict", not an["sensitivity"]["S2_without_r1_overlap"]["changes"]
         and not an["sensitivity"]["S3_without_bare"]["changes"]
         and not an["sensitivity"]["S4_without_formatlike_targets"]["changes"]),
        ("S4 leaves out the 6 formatting-like positions",
         an["sensitivity"]["S4_without_formatlike_targets"]["positions_formatlike_excluded"] == 6
         and an["sensitivity"]["S4_without_formatlike_targets"]["prompts_valid_primary"]["chat-begun"] == 54),
    ]
    for name, ok in checks:
        if not ok:
            fails.append("check failed: " + name)
    # determinism of the bootstrap
    an2 = analyze(data)
    if json.dumps(an2["bands"], sort_keys=True, default=str) != json.dumps(an["bands"], sort_keys=True, default=str):
        fails.append("bootstrap not deterministic")
    # a different seed moves the intervals a little but not the verdicts
    an3 = analyze(data, seed=C.BOOT_SEED + 1)
    if an3["decision"] != an["decision"]:
        fails.append("verdicts depend on the bootstrap seed in the synthetic case")
    # the report renders
    md = report_md(an, {"selftest": True}, {"files": {}})
    if "SELF-TEST" not in md or "| r1 | 50 |" not in md or "S1" not in md:
        fails.append("report text incomplete")
    for f in fails:
        print("FAIL:", f)
    print(f"self-test: {len(units)} unit checks, {len(expected)} verdicts and {len(checks)} facts checked; "
          f"{'ALL PASS' if not fails else str(len(fails)) + ' FAILURES'}")
    print(md)
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(C.EVAL_JSON))
    ap.add_argument("--out", default=str(C.ANALYSIS_JSON))
    ap.add_argument("--report", default=None,
                    help="default: R2_03_report_section.md beside the default --out, else the --out path with .md")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    out = Path(args.out)
    report = Path(args.report) if args.report else (
        C.REPORT_SECTION if out.resolve() == C.ANALYSIS_JSON.resolve() else out.with_suffix(".md"))
    raw = Path(args.inp).read_bytes()
    data = json.loads(raw)
    if data.get("schema") != "R2_02/1":
        raise SystemExit(f"unexpected schema {data.get('schema')}")
    meta = data.get("meta", {})
    if meta.get("selftest") and (out.resolve() == C.ANALYSIS_JSON.resolve()
                                 or report.resolve() == C.REPORT_SECTION.resolve()):
        raise SystemExit("this input is self-test data; write the analysis and report elsewhere (--out, --report)")
    an = analyze(data)
    integ = integrity_of(meta)
    an["input"] = {"path": str(args.inp), "sha256": hashlib.sha256(raw).hexdigest(), "selftest": bool(meta.get("selftest")),
                   "run_summary": meta.get("summary"), "run_files_sha256": meta.get("files_sha256"),
                   "run_git": meta.get("git")}
    an["integrity"] = integ
    an["written_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    out.write_text(json.dumps(an, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    md = report_md(an, meta, integ)
    report.write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {out} and {report}")


if __name__ == "__main__":
    main()

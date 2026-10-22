#!/usr/bin/env python3
"""R3 06 (CPU): the band evaluation. R2's decision rule (R2_PREREG.md §6-§8), unchanged, applied to R3's lens set,
bands and counts (R3_PREREG.md §7).

The rule is re-implemented here, parameterised by a Design (the decision conditions, the readouts, the candidate
lenses, the reference lenses, the bands, eligibility, precedence, the lowest layer that may be served, the seeds), so
that R2's design and R3's run through the same code. Only pure pieces are imported from R2: word_key, the collapse
targets and formatting-like test (R2_common), and the exact sign-test tail and Holm's adjustment (R2_03_analyze).

Primary metric: at each read position, layer and readout, hit = the filtered top-1 is the same word as the content
target (word_key: NFKC, surrounding whitespace stripped, case folded); positions whose target is a raw-text collapse
word ("own", "same") have no usable target. Prompt score = mean hit over the prompt's positions in the condition
(and the band's layers); pooled = mean over the prompts of the decision conditions. 95% percentile intervals from
2000 prompt-level bootstrap resamples, stratified by condition, the same indices for every readout, layer and band.

A candidate is served at a band iff all five rules hold:
  1. pooled, the 95% intervals of candidate - J++ and of candidate - naked eye lie above 0;
  2. in no decision condition is either per-condition interval entirely below 0 (no data counts as a failure);
  3. at every layer of the band and in every decision condition, the stuck-word share is at most 0.25;
  4. the exact one-sided sign test against both baselines (the larger p), Holm-adjusted across every candidate-band
     test (R3: 2 candidates x 7 bands = 14), is below 0.025;
  5. the pooled point estimates of candidate - J++ and of candidate - naked eye are each at least 0.05.
R3: the J++ recipe (jpp-it) takes precedence over the pooled R-lens (rl-it) where both pass; no band with a layer
below code 30 (page 31) is ever served; R1's lens and Miru are evaluated by the same rules for reference only (outside
the Holm family, never served). Sensitivity analyses S1-S5 rerun the rule on subsets or variants.

Modes:
  default        R3_05_eval.json -> R3_06_analysis.json and R3_06_report_section.md
  --reproduce-r2 R2_02_eval.json under R2's design; every verdict, rule, sign test, Holm p, interval and sensitivity
                 verdict must equal R2_03_analysis.json exactly; writes R3_06_r2_reproduction.json
  --selftest     synthetic files with known answers (each rule failing alone, precedence, the never-below-30 rule,
                 the reference lenses, the sensitivity analyses, sign test and Holm) and R2_03's own synthetic file,
                 whose analysis under R2's design must equal R2_03's; writes nothing to experiments/results
Run: CUDA_VISIBLE_DEVICES="" /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R3_06_analyze.py
       [--reproduce-r2 | --selftest | --in X --out Y --report Z [--replication R2_SET.json]]
"""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True

import argparse
import collections
import datetime
import hashlib
import json
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import R2_common as R2  # noqa: E402
import R3_common as C  # noqa: E402
from R2_03_analyze import binom_sf_half, holm  # noqa: E402  (pure: the exact tail and Holm's step-down)

EPS = 1e-12
PRIMARY = "f_word"
FAMILY = {"f_word": 0, "f_id": 0, "f5_word": 0, "u_word_ct": 0, "u_next": 1, "u_fmt": 1}


@dataclass(frozen=True)
class Design:
    name: str
    conditions: Tuple[str, ...]
    secondary_conditions: Tuple[str, ...]
    kinds: Tuple[str, ...]                                  # every readout read from the data
    candidates: Tuple[str, ...]                             # lenses the rule decides on (the Holm family)
    reference: Tuple[str, ...]                              # lenses evaluated alike, reported only
    baselines: Tuple[Tuple[str, str], ...]
    bands: Tuple[Tuple[str, Tuple[int, ...]], ...]
    reference_only: FrozenSet[Tuple[str, str]]              # (candidate, band) measured but never served
    precedence: Tuple[str, ...]
    min_serve_layer: int
    boot_seed: int
    n_boot: int
    stuck_max: float = R2.STUCK_MAX
    min_effect: float = R2.MIN_EFFECT
    alpha: float = R2.ALPHA
    validity_ref: Tuple[str, int] = ("naive", 59)
    overlap: Dict[str, str] = field(default_factory=dict)   # S2's prompts
    overlap_why: str = ""
    filtered_key: str = "fs"                                 # the filtered top-k texts read by the primary metric
    filtered_ids_key: str = "f"                              # ... and their ids (S5: "fjs" / "fj", J++'s filter)

    @property
    def all_conds(self) -> Tuple[str, ...]:
        return self.conditions + self.secondary_conditions


R2_DESIGN = Design(
    name="R2", conditions=R2.CONDITIONS, secondary_conditions=R2.SECONDARY_CONDITIONS,
    kinds=("jpp", "naive", "r1", "miru"), candidates=R2.SECOND_LENSES, reference=(),
    baselines=(("jpp", "J++"), ("naive", "naked eye")), bands=R2.BANDS,
    reference_only=frozenset(R2.REFERENCE_ONLY), precedence=R2.PRECEDENCE, min_serve_layer=40,
    boot_seed=R2.BOOT_SEED, n_boot=R2.N_BOOT, overlap=dict(R2.R1_OVERLAP),
    overlap_why="without the prompts that paraphrase or share an answer topic with R1's fit prompts "
                "(R2_common.R1_OVERLAP)")

R3_DESIGN = Design(
    name="R3", conditions=C.CONDITIONS, secondary_conditions=C.SECONDARY_CONDITIONS,
    kinds=("jpp", "naive") + C.CANDIDATES + C.REFERENCE_LENSES, candidates=C.CANDIDATES,
    reference=C.REFERENCE_LENSES, baselines=C.BASELINES, bands=C.BANDS, reference_only=frozenset(),
    precedence=C.PRECEDENCE, min_serve_layer=C.MIN_SERVE_LAYER, boot_seed=C.BOOT_SEED, n_boot=C.N_BOOT,
    overlap=dict(C.FIT_OVERLAP),
    overlap_why="without the prompts that paraphrase or share an answer topic with a fit source "
                "(R3_common.FIT_OVERLAP: WikiText records, J++ items, R1's fit set)")


def memberships(p: Dict[str, Any]) -> List[str]:
    """mid-reply is the "any" draw (a decision condition); the "content_next" draw is mid-reply-content-next."""
    if p["cond"] != "mid-reply":
        return [p["cond"]]
    d = p.get("draws") or []
    return (["mid-reply"] if "any" in d else []) + (["mid-reply-content-next"] if "content_next" in d else [])


def prev_str_of(p: Dict[str, Any]) -> Optional[str]:
    if "ct_prev_str" in p:
        return p["ct_prev_str"]
    off = p.get("ct_offset")
    if off is None:
        return None
    seq = [p.get("read_tok")] + list(p.get("cont_strs") or [])
    return seq[off] if off < len(seq) else None


def r4(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(float(x), 4)


# ------------------------------------------------------------------------------------------------ the table
class Table:
    """Per-position hit values for every metric, readout and layer, and the prompts of each condition."""

    def __init__(self, D: Design, data: Dict[str, Any], exclude_formatlike: bool = False):
        self.D = D
        self.pos = data["positions"]
        self.N = len(self.pos)
        A = D.all_conds
        self.cond_pos: Dict[str, List[int]] = {c: [] for c in A}
        for j, p in enumerate(self.pos):
            for c in memberships(p):
                if c in self.cond_pos:
                    self.cond_pos[c].append(j)
        self.prompts: Dict[str, List[str]] = {}
        self.prompt_pos: Dict[str, Dict[str, List[int]]] = {}
        for c in A:
            pp: Dict[str, List[int]] = {}
            for j in self.cond_pos[c]:
                pp.setdefault(self.pos[j]["pid"], []).append(j)
            self.prompts[c] = list(pp)
            self.prompt_pos[c] = pp
        self.layers: Dict[str, List[int]] = {}
        for k in D.kinds:
            ls = set()
            for p in self.pos:
                ls |= {int(x) for x in p["r"].get(k, {})}
            self.layers[k] = sorted(ls)
        self.raw_ct_key = [R2.word_key(p.get("ct_str")) if p.get("ct_id") is not None else None for p in self.pos]
        self.collapse = [k is not None and k in R2.COLLAPSE_TARGETS for k in self.raw_ct_key]
        self.formatlike = []
        for p, k in zip(self.pos, self.raw_ct_key):
            if k is None:
                self.formatlike.append(False)
            elif "ct_formatlike" in p:
                self.formatlike.append(bool(p["ct_formatlike"]))
            else:
                self.formatlike.append(R2.formatlike_target(p.get("ct_str"), prev_str_of(p)))
        self.exclude_formatlike = exclude_formatlike
        self.ct_key = [None if (k is None or col or (exclude_formatlike and fl)) else k
                       for k, col, fl in zip(self.raw_ct_key, self.collapse, self.formatlike)]
        self.vals: Dict[Tuple[str, str, int], np.ndarray] = {}
        self._ps: Dict[Tuple[Any, ...], np.ndarray] = {}

    def values(self, metric: str, kind: str, layer: int) -> np.ndarray:
        key = (metric, kind, layer)
        if key in self.vals:
            return self.vals[key]
        fk = self.D.filtered_key
        v = np.full(self.N, np.nan)
        for j, p in enumerate(self.pos):
            r = p["r"].get(kind, {}).get(str(layer))
            if r is None:
                continue
            ck = self.ct_key[j]
            if metric == "f_word":
                v[j] = np.nan if ck is None else float(R2.word_key(r[fk][0]) == ck)
            elif metric == "f_id":
                v[j] = np.nan if ck is None else float(r[self.D.filtered_ids_key][0] == p["ct_id"])
            elif metric == "f5_word":
                v[j] = np.nan if ck is None else float(ck in {R2.word_key(s) for s in r[fk]})
            elif metric == "u_word_ct":
                v[j] = np.nan if ck is None else float(R2.word_key(r["us"][0]) == ck)
            elif metric == "u_next":
                v[j] = float(r["u"][0] == p["next_id"])
            elif metric == "u_fmt":
                v[j] = float(not r["u0c"])
            else:
                raise ValueError(metric)
        self.vals[key] = v
        return v

    def band_values(self, metric: str, kind: str, layers: Sequence[int]) -> np.ndarray:
        vs = np.stack([self.values(metric, kind, l) for l in layers], 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(vs, 0)

    def prompt_scores(self, metric: str, kind: str, layers: Sequence[int], cond: str) -> np.ndarray:
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
    """Prompt-level bootstrap indices per (condition, metric family), over the prompts with a defined value:
    numpy default_rng([seed, condition index]) for the primary family, [seed, condition index, family] otherwise."""

    def __init__(self, t: Table, n_boot: int, seed: int):
        self.t, self.B, self.seed = t, n_boot, seed
        self.cache: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}

    def valid(self, cond: str, fam: int) -> np.ndarray:
        metric = "f_word" if fam == 0 else "u_next"
        k, l = self.t.D.validity_ref
        return np.flatnonzero(~np.isnan(self.t.prompt_scores(metric, k, [l], cond)))

    def idx(self, cond: str, fam: int) -> Tuple[np.ndarray, np.ndarray]:
        key = (cond, fam)
        if key not in self.cache:
            ok = self.valid(cond, fam)
            ci = self.t.D.all_conds.index(cond)
            rng = np.random.default_rng([self.seed, ci] if fam == 0 else [self.seed, ci, fam])
            ix = rng.integers(0, len(ok), size=(self.B, len(ok))) if len(ok) else np.zeros((self.B, 0), dtype=int)
            self.cache[key] = (ok, ix)
        return self.cache[key]


def ci_of(boot: np.ndarray) -> Tuple[float, float]:
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def stat(n: int, mean, lo, hi, **extra) -> Dict[str, Any]:
    return dict({"n": int(n), "mean": r4(mean), "lo": r4(lo), "hi": r4(hi),
                 "mean_raw": mean, "lo_raw": lo, "hi_raw": hi}, **extra)


def summarize_arrays(get_x: Callable[[str, np.ndarray], Optional[np.ndarray]], boot: Boot, fam: int,
                     conds: Tuple[str, ...]) -> Dict[str, Any]:
    per: Dict[str, Any] = {}
    sums, ns, tot = np.zeros(boot.B), 0, 0.0
    for c in boot.t.D.all_conds:
        ok, ix = boot.idx(c, fam)
        if not len(ok):
            per[c] = stat(0, None, None, None)
            continue
        x = get_x(c, ok)
        if x is None or np.isnan(x).any():
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


def summarize(t: Table, boot: Boot, metric: str, a, b=None, conds: Optional[Tuple[str, ...]] = None):
    conds = t.D.conditions if conds is None else conds

    def get_x(c, ok):
        x = t.prompt_scores(metric, a[0], a[1], c)[ok]
        if b is not None:
            x = x - t.prompt_scores(metric, b[0], b[1], c)[ok]
        return x
    return summarize_arrays(get_x, boot, FAMILY[metric], conds)


def sign_test(t: Table, boot: Boot, lens: str, base: str, layers: List[int], conds: Tuple[str, ...]) -> Dict[str, Any]:
    ds = []
    for c in conds:
        ok, _ = boot.idx(c, 0)
        if not len(ok):
            continue
        ds.append(t.prompt_scores(PRIMARY, lens, layers, c)[ok] - t.prompt_scores(PRIMARY, base, layers, c)[ok])
    d = np.concatenate(ds) if ds else np.zeros(0)
    if np.isnan(d).any():
        return {"n_pos": None, "n_neg": None, "n_tie": None, "p": 1.0, "note": "readout missing"}
    n_pos, n_neg = int((d > 1e-9).sum()), int((d < -1e-9).sum())
    return {"n_pos": n_pos, "n_neg": n_neg, "n_tie": int(len(d) - n_pos - n_neg), "p": binom_sf_half(n_pos, n_pos + n_neg)}


def stuck(t: Table, kind: str, layer: int, conds: Tuple[str, ...]) -> Dict[str, Any]:
    """Rule 3's share: positions whose filtered top-1 is the layer's most common filtered top-1 word, not counting
    positions where that word is the position's own usable target; also the share counting every position."""
    fk = t.D.filtered_key
    rows = []
    for j in sorted({j for c in conds for j in t.cond_pos[c]}):
        r = t.pos[j]["r"].get(kind, {}).get(str(layer))
        if r is not None:
            rows.append((j, r))
    if not rows:
        return {"share": None, "share_raw": None, "n": 0}
    n = len(rows)
    keys = [R2.word_key(r[fk][0]) for _, r in rows]
    rep: Dict[str, str] = {}
    for (_, r), k in zip(rows, keys):
        rep.setdefault(k, r[fk][0])
    cnt = collections.Counter(k for (j, _), k in zip(rows, keys) if k != t.ct_key[j])
    w, c = cnt.most_common(1)[0] if cnt else (None, 0)
    w_all, c_all = collections.Counter(keys).most_common(1)[0]
    fi = t.D.filtered_ids_key
    ids_ex = collections.Counter(r[fi][0] for (j, r), k in zip(rows, keys) if k != t.ct_key[j])
    c_idx = ids_ex.most_common(1)[0][1] if ids_ex else 0
    (i_all, c_id), = collections.Counter(r[fi][0] for _, r in rows).most_common(1)
    return {"share": r4(c / n), "share_raw": c / n, "n": n, "word": rep.get(w) if w is not None else None,
            "share_by_id": r4(c_idx / n),
            "share_incl_target": r4(c_all / n), "word_incl_target": rep[w_all],
            "share_by_id_incl_target": r4(c_id / n), "token_id_incl_target": int(i_all)}


def fmt_share(t: Table, kind: str, layer: int, conds: Optional[Tuple[str, ...]] = None) -> Optional[float]:
    conds = t.D.conditions if conds is None else conds
    js = sorted({j for c in conds for j in t.cond_pos[c]})
    v = t.values("u_fmt", kind, layer)[js]
    v = v[~np.isnan(v)]
    return round(float(v.mean()), 4) if len(v) else None


def band_layers_for(t: Table, kind: str, layers: Sequence[int]) -> bool:
    return all(l in t.layers.get(kind, []) for l in layers)


def beats(x: Dict[str, Any]) -> bool:
    return x.get("lo_raw") is not None and x["lo_raw"] > 0


def significantly_worse(x: Dict[str, Any]) -> bool:
    return x.get("hi_raw") is not None and x["hi_raw"] < 0


def evaluate_band(t: Table, boot: Boot, lens: str, band: str, layers: List[int], conds: Tuple[str, ...]):
    """Rules 1, 2, 3 and 5, and the sign tests for rule 4 (Holm is applied over the family by finalize)."""
    D = t.D
    sec = summarize(t, boot, PRIMARY, (lens, layers), conds=conds)
    base = {b: summarize(t, boot, PRIMARY, (b, layers), conds=conds) for b, _ in D.baselines}
    diff = {b: summarize(t, boot, PRIMARY, (lens, layers), (b, layers), conds=conds) for b, _ in D.baselines}
    signs = {b: sign_test(t, boot, lens, b, layers, conds) for b, _ in D.baselines}
    st = {str(l): {c: stuck(t, lens, l, (c,)) for c in conds} for l in layers}
    st_pooled = {str(l): stuck(t, lens, l, conds) for l in layers}
    reasons: List[str] = []
    rules: Dict[str, bool] = {}
    rules["1"] = all(beats(diff[b]["pooled"]) for b, _ in D.baselines)
    for b, name in D.baselines:
        if not beats(diff[b]["pooled"]):
            reasons.append(f"[rule 1] pooled CI of second - {name} does not lie above 0")
    worse = []
    for c in conds:
        for b, name in D.baselines:
            x = diff[b]["by_cond"].get(c, {})
            if x.get("hi_raw") is None:
                worse.append(f"{c}: no data against {name}")
            elif significantly_worse(x):
                worse.append(f"{c}: significantly worse than {name}")
    rules["2"] = not worse
    reasons += [f"[rule 2] {w}" for w in worse]
    bad = []
    for l in layers:
        for c in conds:
            v = st[str(l)][c]
            if v["share_raw"] is None:
                bad.append(f"{l} {c} (no data)")
            elif v["share_raw"] > D.stuck_max + EPS:
                bad.append(f"{l} {c} ({v['share_raw']:.2f}, {v['word']!r})")
    rules["3"] = not bad
    if bad:
        reasons.append("[rule 3] stuck-word share above 25% at " + ", ".join(bad))
    st_max = max((v["share_raw"] for d in st.values() for v in d.values() if v["share_raw"] is not None), default=None)
    st_where = None
    if st_max is not None:
        st_where = next((l, c, v["word"]) for l, d in st.items() for c, v in d.items() if v["share_raw"] == st_max)
    p_band = max(signs[b]["p"] for b, _ in D.baselines)
    rules["5"] = all(diff[b]["pooled"]["mean_raw"] is not None and diff[b]["pooled"]["mean_raw"] >= D.min_effect - EPS
                     for b, _ in D.baselines)
    for b, name in D.baselines:
        m = diff[b]["pooled"]["mean_raw"]
        if m is None or m < D.min_effect - EPS:
            reasons.append(f"[rule 5] pooled second - {name} is {'n/a' if m is None else f'{m:+.3f}'}, "
                           f"below the minimum {D.min_effect}")
    out = {"lens": lens, "band": band, "layers": layers, "conds": list(conds), "second": sec}
    for b, _ in D.baselines:
        out[b] = base[b]
    for b, _ in D.baselines:
        out[f"diff_{b}"] = diff[b]
    for b, _ in D.baselines:
        out[f"sign_{b}"] = signs[b]
    out.update({"p_sign": p_band, "stuck": st, "stuck_pooled": st_pooled, "stuck_max": r4(st_max),
                "stuck_max_where": st_where, "fmt_share": {str(l): fmt_share(t, lens, l, conds) for l in layers},
                "worse_in": worse, "rules": rules, "reasons": reasons})
    return out


def finalize(D: Design, bands: Dict[str, Dict[str, Dict[str, Any]]], holm_family: bool = True) -> None:
    """Rule 4 (Holm across every lens-band test in `bands`) and each band's verdict."""
    ps = {(lens, band): ev["p_sign"] for lens, d in bands.items() for band, ev in d.items()}
    adj = holm(ps) if holm_family else ps
    for (lens, band), p in adj.items():
        ev = bands[lens][band]
        ev["p_sign_holm"] = p
        ev["holm_m"] = len(ps) if holm_family else 1
        ev["rules"]["4"] = p < D.alpha
        if not ev["rules"]["4"]:
            b0, b1 = D.baselines[0][0], D.baselines[1][0]
            sj, sn = ev[f"sign_{b0}"], ev[f"sign_{b1}"]
            ev["reasons"].append(f"[rule 4] sign test p = {ev['p_sign']:.3g} (J++ {sj['n_pos']}+/{sj['n_neg']}-, "
                                 f"naked {sn['n_pos']}+/{sn['n_neg']}-), Holm-adjusted {p:.3g} over "
                                 f"{ev['holm_m']} tests, not below {D.alpha}")
        ev["reasons"].sort()
        ev["passes"] = all(ev["rules"][k] for k in ("1", "2", "3", "4", "5"))


def decide(D: Design, bands: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Eligibility (reference-only bands; never below min_serve_layer) and precedence."""
    by_band: Dict[str, Optional[str]] = {}
    served: Dict[str, List[int]] = {k: [] for k in D.candidates}
    for band, layers in D.bands:
        choice = None
        for lens in D.precedence:
            ev = bands.get(lens, {}).get(band)
            if ev is None:
                continue
            low = min(layers) < D.min_serve_layer
            ev["eligible"] = (lens, band) not in D.reference_only and not low
            ev["served"] = False
            ev.pop("note", None)
            if ev["passes"] and ev["eligible"] and choice is None:
                choice = lens
                ev["served"] = True
            elif ev["passes"] and ev["eligible"] and choice is not None:
                ev["note"] = f"passes, but {choice} takes precedence at this band"
            elif ev["passes"] and low:
                ev["note"] = f"passes, but no band below code {D.min_serve_layer} is ever served"
            elif ev["passes"] and not ev["eligible"]:
                ev["note"] = "passes, but this band is reference only for this lens"
        by_band[band] = choice
        if choice:
            served[choice] += list(layers)
    return {"by_band": by_band, "served_layers": served}


def decision_suite(D: Design, data: Dict[str, Any], conds: Optional[Tuple[str, ...]] = None,
                   exclude_formatlike: bool = False):
    """The whole decision on `data`: (table, boot, bands, decision, reference bands)."""
    conds = D.conditions if conds is None else conds
    t = Table(D, data, exclude_formatlike=exclude_formatlike)
    boot = Boot(t, D.n_boot, D.boot_seed)
    bands: Dict[str, Dict[str, Any]] = {}
    for lens in D.candidates:
        for band, layers in D.bands:
            if layers and band_layers_for(t, lens, layers) and band_layers_for(t, "jpp", layers):
                bands.setdefault(lens, {})[band] = evaluate_band(t, boot, lens, band, list(layers), conds)
    finalize(D, bands)
    ref: Dict[str, Dict[str, Any]] = {}
    for lens in D.reference:
        for band, layers in D.bands:
            if layers and band_layers_for(t, lens, layers) and band_layers_for(t, "jpp", layers):
                ref.setdefault(lens, {})[band] = evaluate_band(t, boot, lens, band, list(layers), conds)
    finalize(D, ref, holm_family=False)
    for d in ref.values():
        for ev in d.values():
            ev["eligible"], ev["served"] = False, False
            ev["note"] = "reference only: outside the Holm family, never served"
    return t, boot, bands, decide(D, bands), ref


def verdicts(bands) -> Dict[str, Dict[str, Any]]:
    return {f"{lens} {band}": {"passes": ev["passes"], "served": ev.get("served", False),
                               "failed_rules": [k for k, v in ev["rules"].items() if not v]}
            for lens, d in bands.items() for band, ev in d.items()}


def sensitivity(D: Design, data, base_bands, name: str, why: str, drop_pids=frozenset(),
                conds: Optional[Tuple[str, ...]] = None, exclude_formatlike: bool = False) -> Dict[str, Any]:
    conds = D.conditions if conds is None else conds
    sub = dict(data, positions=[p for p in data["positions"] if p["pid"] not in drop_pids])
    t, boot, bands, dec, _ = decision_suite(D, sub, conds, exclude_formatlike)
    v0, v1 = verdicts(base_bands), verdicts(bands)
    changes = []
    for k in sorted(set(v0) | set(v1)):
        a, b = v0.get(k), v1.get(k)
        if a is None or b is None or a["passes"] != b["passes"] or a["served"] != b["served"]:
            changes.append({"lens_band": k, "primary": a, "sensitivity": b})
    dropped = sorted({p["pid"] for p in data["positions"] if p["pid"] in drop_pids})
    b0, b1 = D.baselines[0][0], D.baselines[1][0]
    return {"name": name, "why": why, "conds": list(conds), "exclude_formatlike": exclude_formatlike,
            "prompts_dropped": len(dropped), "pids_dropped": dropped,
            "prompts_valid_primary": {c: int(len(boot.idx(c, 0)[0])) for c in conds},
            "positions_formatlike_excluded": int(sum(1 for j in range(t.N) if exclude_formatlike and t.formatlike[j]
                                                     and any(j in t.cond_pos[c] for c in conds))),
            "decision": dec, "verdicts": v1, "changes": changes,
            "bands": {lens: {band: {k: ev[k] for k in ("passes", "rules", "reasons", "p_sign", "p_sign_holm")}
                             | {f"diff_{b0}": ev[f"diff_{b0}"]["pooled"], f"diff_{b1}": ev[f"diff_{b1}"]["pooled"]}
                             for band, ev in d.items()} for lens, d in bands.items()}}


def constant_guess(t: Table, boot: Boot) -> Dict[str, Any]:
    modes: Dict[str, Any] = {}

    def get_x(c, ok):
        pids = t.prompts[c]
        keys = [[t.ct_key[j] for j in t.prompt_pos[c][pid] if t.ct_key[j] is not None] for pid in pids]
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
    s = summarize_arrays(get_x, boot, 0, t.D.conditions)
    s["most_common_target"] = modes
    return s


def function_word_split(t: Table, kind: str, layers: List[int], conds: Tuple[str, ...]) -> Dict[str, Any]:
    v = t.band_values(PRIMARY, kind, layers)
    out: Dict[str, Any] = {}
    groups = {c: sorted(t.cond_pos[c]) for c in conds}
    groups["pooled"] = sorted({j for c in conds for j in t.cond_pos[c]})
    for g, js in groups.items():
        row = {}
        for name, want in (("function", True), ("other", False)):
            sel = [j for j in js if t.ct_key[j] is not None and (t.ct_key[j] in R2.FUNCTION_WORDS) == want
                   and not np.isnan(v[j])]
            row[name] = {"n": len(sel), "mean": r4(float(np.mean(v[sel]))) if sel else None}
        out[g] = row
    return out


def has_jpp_filter(data: Dict[str, Any]) -> bool:
    """Whether the run recorded J++'s filtered readouts (fj ids, fjs texts) beside R2's f/fs (needed by S5)."""
    for p in data["positions"][:50]:
        for d in p["r"].values():
            for r in d.values():
                return "fjs" in r
    return False


def analyze(D: Design, data: Dict[str, Any]) -> Dict[str, Any]:
    t, boot, bands, dec, ref = decision_suite(D, data)
    A = D.all_conds
    b0, b1 = D.baselines[0][0], D.baselines[1][0]
    out: Dict[str, Any] = {"schema": f"{D.name}_06/1", "design": D.name,
                           "bootstrap": {"n": D.n_boot, "seed": D.boot_seed, "ci": "95% percentile"},
                           "constants": {"STUCK_MAX": D.stuck_max, "MIN_EFFECT": D.min_effect, "ALPHA": D.alpha,
                                         "COLLAPSE_TARGETS": list(R2.COLLAPSE_TARGETS),
                                         "MIN_SERVE_LAYER": D.min_serve_layer,
                                         "HOLM_M": sum(len(d) for d in bands.values())}}
    counts = {}
    for c in A:
        js = t.cond_pos[c]
        counts[c] = {"prompts": len(t.prompts[c]), "positions": len(js),
                     "positions_with_content_target": sum(1 for j in js if t.raw_ct_key[j] is not None),
                     "positions_collapse_target_excluded": sum(1 for j in js if t.collapse[j]),
                     "positions_usable_target": sum(1 for j in js if t.ct_key[j] is not None),
                     "positions_formatlike_target": sum(1 for j in js if t.formatlike[j]),
                     "prompts_valid_primary": int(len(boot.idx(c, 0)[0]))}
    out["counts"] = counts
    ctx: Dict[str, Any] = {}
    for c in A:
        js = t.cond_pos[c]
        if not js:
            continue
        agree = float(np.mean([bool(t.pos[j].get("model_agrees")) for j in js]))
        offs = [t.pos[j]["ct_offset"] for j in js if t.pos[j].get("ct_offset") is not None]
        ctx[c] = {"next_is_noncontent": round(float(np.mean([not t.pos[j]["next_content"] for j in js])), 4),
                  "ct_offset_mean": round(float(np.mean(offs)), 3) if offs else None,
                  "model_agrees_with_next": round(agree, 4), "agreement_below_0.95": agree < 0.95}
    out["context"] = ctx
    vk, vl = D.validity_ref
    if vl in t.layers.get(vk, []):
        out["context"]["model_ceiling_filtered_59"] = summarize(t, boot, PRIMARY, (vk, [vl]))
    out["context"]["constant_guess"] = constant_guess(t, boot)
    per_layer: Dict[str, Dict[str, Any]] = {}
    for k in D.kinds:
        for l in t.layers.get(k, []):
            row = {
                "f_word": summarize(t, boot, "f_word", (k, [l])),
                "f_id": summarize(t, boot, "f_id", (k, [l]))["pooled"],
                "f5_word": summarize(t, boot, "f5_word", (k, [l]))["pooled"],
                "u_next": summarize(t, boot, "u_next", (k, [l]))["pooled"],
                "u_word_ct": summarize(t, boot, "u_word_ct", (k, [l]))["pooled"],
                "fmt_share": fmt_share(t, k, l),
                "fmt_share_by_cond": {c: fmt_share(t, k, l, (c,)) for c in A if t.cond_pos[c]},
                "stuck": stuck(t, k, l, D.conditions),
                "stuck_by_cond": {c: stuck(t, k, l, (c,)) for c in A if t.cond_pos[c]},
            }
            if (k in D.candidates or k in D.reference) and l in t.layers.get(b0, []):
                row[f"diff_{b0}"] = summarize(t, boot, "f_word", (k, [l]), (b0, [l]))["pooled"]
                row[f"diff_{b1}"] = summarize(t, boot, "f_word", (k, [l]), (b1, [l]))["pooled"]
            per_layer.setdefault(k, {})[str(l)] = row
    out["per_layer"] = per_layer
    out["bands"] = bands
    out["decision"] = dec
    out["reference_bands"] = ref
    sec: Dict[str, Any] = {}
    for name, metric in (("unfiltered_next_token", "u_next"), ("unfiltered_vs_content_target", "u_word_ct"),
                         ("filtered_exact_id", "f_id"), ("filtered_top5_word", "f5_word")):
        sec[name] = {}
        for lens in D.candidates + D.reference:
            for band, layers in D.bands:
                if band_layers_for(t, lens, layers) and band_layers_for(t, b0, layers):
                    L = list(layers)
                    sec[name].setdefault(lens, {})[band] = {
                        "second": summarize(t, boot, metric, (lens, L))["pooled"],
                        b0: summarize(t, boot, metric, (b0, L))["pooled"],
                        b1: summarize(t, boot, metric, (b1, L))["pooled"],
                        f"diff_{b0}": summarize(t, boot, metric, (lens, L), (b0, L))["pooled"],
                        f"diff_{b1}": summarize(t, boot, metric, (lens, L), (b1, L))["pooled"],
                    }
    sec["mid_reply_content_next"] = {}
    if "mid-reply-content-next" in t.cond_pos and t.cond_pos["mid-reply-content-next"]:
        cn: Dict[str, Dict[str, Any]] = {}
        for lens in D.candidates:
            for band, layers in D.bands:
                if band_layers_for(t, lens, layers) and band_layers_for(t, b0, layers):
                    cn.setdefault(lens, {})[band] = evaluate_band(t, boot, lens, band, list(layers),
                                                                  conds=("mid-reply-content-next",))
        finalize(D, cn, holm_family=False)
        for lens, d in cn.items():
            for band, ev in d.items():
                sec["mid_reply_content_next"].setdefault(lens, {})[band] = {
                    k: ev[k] for k in ("second", b0, b1, f"diff_{b0}", f"diff_{b1}", f"sign_{b0}", f"sign_{b1}",
                                       "stuck_max", "passes", "reasons")}
    split = {}
    js_all = sorted({j for c in D.conditions for j in t.cond_pos[c]})
    for name, want in (("next_is_noncontent", False), ("next_is_content", True)):
        js = [j for j in js_all if bool(t.pos[j]["next_content"]) == want]
        row: Dict[str, Any] = {"n_positions": len(js)}
        for k in D.kinds:
            for l in t.layers.get(k, []):
                v = t.values("f_word", k, l)[js]
                v = v[~np.isnan(v)]
                row.setdefault(k, {})[str(l)] = round(float(v.mean()), 4) if len(v) else None
        split[name] = row
    sec["split_by_next_token"] = split
    fw: Dict[str, Any] = {}
    for lens, d in bands.items():
        for band, ev in d.items():
            fw.setdefault(lens, {})[band] = {k: function_word_split(t, k, ev["layers"], D.conditions)
                                             for k in (lens, b0, b1)}
    sec["function_word_split"] = fw
    out["secondary"] = sec
    about = frozenset(p["pid"] for p in data.get("prompts", []) if p.get("about_model"))
    sens = {
        "S1_without_about_model": sensitivity(
            D, data, bands, "S1", "without the prompts about the model itself (R1's fit set had many self-questions)",
            drop_pids=about),
        f"S2_without_{'r1' if D.name == 'R2' else 'fit'}_overlap": sensitivity(
            D, data, bands, "S2", D.overlap_why, drop_pids=frozenset(D.overlap)),
        "S3_without_bare": sensitivity(
            D, data, bands, "S3", "without the bare condition (raw text puts this instruct model off-distribution)",
            conds=tuple(c for c in D.conditions if c != "bare")),
        "S4_without_formatlike_targets": sensitivity(
            D, data, bands, "S4", "without positions whose content target is a list numeral after a newline or the "
            "token after a code fence", exclude_formatlike=True),
    }
    if D.name != "R2":
        if has_jpp_filter(data):
            sens["S5_jpp_filter"] = sensitivity(
                replace(D, filtered_key="fjs", filtered_ids_key="fj"), data, bands, "S5",
                "every readout filtered by J++'s no-letter-or-digit rule instead of R2's filter (D7)")
        else:
            sens["S5_jpp_filter"] = {"name": "S5", "why": "J++'s filter for every readout (D7)",
                                     "not_run": "the run recorded no J++-filtered readouts (fjs)", "changes": []}
    out["sensitivity"] = sens
    return out


# ------------------------------------------------------------------------------------------------ R2 reproduction
REPRO_BAND_KEYS = ("passes", "served", "eligible", "rules", "reasons", "p_sign", "p_sign_holm", "holm_m",
                   "stuck_max", "stuck_max_where", "worse_in", "layers", "conds", "sign_jpp", "sign_naive")
REPRO_STAT_KEYS = ("second", "jpp", "naive", "diff_jpp", "diff_naive")


def _norm(x):
    return json.loads(json.dumps(x, default=str))


def reproduce_r2(eval_path: Path, analysis_path: Path) -> Dict[str, Any]:
    """R2_02_eval.json through R3_06 under R2's design, against R2_03_analysis.json: every band's verdict, rules,
    reasons, sign tests, p and Holm p, interval bounds (unrounded), the decision, and every sensitivity verdict."""
    raw = eval_path.read_bytes()
    data = json.loads(raw)
    ref = json.loads(analysis_path.read_text(encoding="utf-8"))
    got = analyze(R2_DESIGN, data)
    mism: List[str] = []
    n_cmp = 0

    def cmp(path: str, a, b):
        nonlocal n_cmp
        n_cmp += 1
        if _norm(a) != _norm(b):
            mism.append(f"{path}: R3_06 {json.dumps(_norm(a), default=str)[:200]} != R2_03 "
                        f"{json.dumps(_norm(b), default=str)[:200]}")

    cmp("bands.keys", sorted((l, b) for l, d in got["bands"].items() for b in d),
        sorted((l, b) for l, d in ref["bands"].items() for b in d))
    rows = []
    for lens, d in ref["bands"].items():
        for band, rb in d.items():
            gb = got["bands"].get(lens, {}).get(band, {})
            for k in REPRO_BAND_KEYS:
                cmp(f"bands.{lens}.{band}.{k}", gb.get(k), rb.get(k))
            cmp(f"bands.{lens}.{band}.note", gb.get("note"), rb.get("note"))
            for k in REPRO_STAT_KEYS:
                cmp(f"bands.{lens}.{band}.{k}", gb.get(k), rb.get(k))
            cmp(f"bands.{lens}.{band}.stuck", gb.get("stuck"), rb.get("stuck"))
            rows.append({"lens": lens, "band": band, "passes": gb.get("passes"), "served": gb.get("served"),
                         "failed_rules": sorted(k for k, v in (gb.get("rules") or {}).items() if not v),
                         "p_sign": gb.get("p_sign"), "p_sign_holm": gb.get("p_sign_holm"),
                         "r2_p_sign_holm": rb.get("p_sign_holm"),
                         "diff_jpp_mean": (gb.get("diff_jpp") or {}).get("pooled", {}).get("mean_raw"),
                         "diff_naive_mean": (gb.get("diff_naive") or {}).get("pooled", {}).get("mean_raw")})
    cmp("decision", got["decision"], ref["decision"])
    cmp("counts", got["counts"], ref["counts"])
    cmp("context", got["context"], ref["context"])
    cmp("per_layer", got["per_layer"], ref["per_layer"])
    cmp("secondary", {k: v for k, v in got["secondary"].items() if k != "miru_minus_r1"},
        {k: v for k, v in ref["secondary"].items() if k != "miru_minus_r1"})
    for key, rs in ref["sensitivity"].items():
        gs = got["sensitivity"].get(key, {})
        for k in ("verdicts", "changes", "decision", "prompts_dropped", "pids_dropped", "prompts_valid_primary",
                  "positions_formatlike_excluded", "bands"):
            cmp(f"sensitivity.{key}.{k}", gs.get(k), rs.get(k))
    return {"schema": "R3_06_r2_reproduction/1", "written_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "input": {"path": str(eval_path.relative_to(C.REPO)), "sha256": hashlib.sha256(raw).hexdigest()},
            "reference": {"path": str(analysis_path.relative_to(C.REPO)), "sha256": C.sha256_file(analysis_path)},
            "design": "R2 (R2_common: lenses r1, miru; bands 40-49, 50, 51-54, 55, 56-57, 58; miru 58 reference "
                      "only; r1 over miru; bootstrap seed 20261019)",
            "compared": n_cmp, "mismatches": mism, "ok": not mism,
            "not_compared": "secondary.miru_minus_r1 (an R2-only table of Miru - R1, not part of the rule)",
            "bands": rows, "decision": got["decision"],
            "sensitivity_changes": {k: v.get("changes") for k, v in got["sensitivity"].items()}}


# ------------------------------------------------------------------------------------------------ report
def fmt_ci(d: Dict[str, Any], signed: bool = True) -> str:
    if d.get("mean") is None:
        return "n/a"
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(d['mean'])} [{f.format(d['lo'])}, {f.format(d['hi'])}]"


def fmt_num(x: Optional[float], f: str = "{:.3f}") -> str:
    return "n/a" if x is None else f.format(x)


def report_md(D: Design, an: Dict[str, Any], meta: Dict[str, Any]) -> str:
    L: List[str] = []
    L.append("## R3 result: J++'s recipe fitted on the instruct model, layer band by layer band")
    L.append("")
    if meta.get("selftest"):
        L.append("**SELF-TEST DATA (fake model): these numbers mean nothing.**")
        L.append("")
    dec = an["decision"]
    L.append("Applied exactly as pre-registered (R3_PREREG.md §7, R2's five rules unchanged). A lens is served at a band "
             "only if all five hold: (1) pooled, its filtered top word names the model's next content word more often "
             "than both filtered J++ and the filtered naked eye (95% prompt-bootstrap intervals above 0); (2) it is "
             "significantly worse than neither in any condition; (3) its stuck-word share is at most 25% in every "
             f"condition at every layer of the band; (4) the exact sign test against both baselines is below "
             f"{D.alpha} after Holm's adjustment over {an['constants']['HOLM_M']} lens-band tests; (5) both pooled "
             f"differences are at least {D.min_effect}. No band below code {D.min_serve_layer} is served. Layers count "
             "from 0 (the pages show layer + 1).")
    L.append("")
    L.append("| band (code) | " + " | ".join(D.candidates) + " | served |")
    L.append("|---|" + "---|" * len(D.candidates) + "---|")
    for band, _ in D.bands:
        cells = []
        for lens in D.candidates:
            ev = an["bands"].get(lens, {}).get(band)
            if ev is None:
                cells.append("not read")
            else:
                failed = [k for k in ("1", "2", "3", "4", "5") if not ev["rules"].get(k)]
                cells.append(ev["note"] if ev.get("note") else
                             ("passes" if ev["passes"] else "fails rule " + ", ".join(failed)))
        L.append(f"| {band} | " + " | ".join(cells) + f" | {dec['by_band'].get(band) or 'none'} |")
    L.append("")
    L.append("| lens | band | second | J++ | naked | second - J++ | second - naked | +/- vs J++ | +/- vs naked | "
             "p (Holm) | worse in | max stuck (layer, condition, word) | verdict |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for group in (an["bands"], an.get("reference_bands", {})):
        for lens, d in group.items():
            for band, _ in D.bands:
                ev = d.get(band)
                if ev is None:
                    continue
                worse = "; ".join(w.split(":")[0] + (" (J++)" if "J++" in w else " (naked)") for w in ev["worse_in"]) or "-"
                verdict = ("reference" if lens in D.reference else
                           "SERVE" if ev.get("served") else ("pass, not served" if ev["passes"] else "do not serve"))
                sj, sn = ev["sign_jpp"], ev["sign_naive"]
                where = ev.get("stuck_max_where")
                stw = f"{ev['stuck_max']:.2f} ({where[0]}, {where[1]}, {where[2]!r})" if where else "n/a"
                L.append(f"| {lens} | {band} | {fmt_num(ev['second']['pooled']['mean'])} | "
                         f"{fmt_num(ev['jpp']['pooled']['mean'])} | {fmt_num(ev['naive']['pooled']['mean'])} | "
                         f"{fmt_ci(ev['diff_jpp']['pooled'])} | {fmt_ci(ev['diff_naive']['pooled'])} | "
                         f"{sj['n_pos']}/{sj['n_neg']} | {sn['n_pos']}/{sn['n_neg']} | {ev['p_sign_holm']:.2g} | "
                         f"{worse} | {stw} | {verdict} |")
    L.append("")
    fails = [(lens, band, ev) for lens, d in an["bands"].items() for band, ev in d.items() if not ev["passes"]]
    if fails:
        L.append("Why each failing band fails:")
        for lens, band, ev in fails:
            L.append(f"- {lens} {band}: " + "; ".join(ev["reasons"]))
        L.append("")
    L.append("Sensitivity analyses (pre-registered; the decision above stands):")
    L.append("")
    L.append("| analysis | served | verdicts that change |")
    L.append("|---|---|---|")
    for key, s in an.get("sensitivity", {}).items():
        if "not_run" in s:
            L.append(f"| {s['name']} ({s['why']}) | not run: {s['not_run']} | - |")
            continue
        sl = s["decision"]["served_layers"]
        ch = "; ".join(f"{c['lens_band']}: {'pass' if (c['primary'] or {}).get('passes') else 'fail'} -> "
                       f"{'pass' if (c['sensitivity'] or {}).get('passes') else 'fail'}" for c in s["changes"]) or "none"
        L.append(f"| {s['name']} ({s['why']}) | " + "; ".join(f"{k} {v or '-'}" for k, v in sl.items()) + f" | {ch} |")
    L.append("")
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------ self-test
T_KIND, T_KIND_CAP, T_OWN, T_THE, T_FMT = 1638, 1639, 1700, 2000, 5213
PREFIX = {"chat-no-answer": "cna", "chat-begun": "cb", "bare": "bare", "mid-reply": "mid"}


def synthetic_r3(seed: int = 11) -> Tuple[Dict[str, Any], Dict[Tuple[str, str], Dict[str, Any]]]:
    """A results file in R3_05's shape (80 prompts per condition, layers 30-59) with known answers, deterministic
    in the prompt index (r = i % 10), every verdict far from its threshold.
      jpp-it 30-39: clear win -> served (the lowest band that may be served)
      jpp-it 40-49: win pooled but worse in chat-begun -> rule 2 only
      jpp-it 50:    win, rl-it also wins -> jpp-it served by precedence, rl-it 'passes, but jpp-it ...'
      jpp-it 51-54: a small consistent win -> rule 5 only
      jpp-it 55:    clear win, stuck in bare -> rule 3 only
      jpp-it 56-57: mean clearly positive, signs split -> rule 4 only
      jpp-it 58:    no better than J++ -> rules 1, 4, 5
      rl-it 30-39 and 51-58: as J++; rl-it 40-49: clear win -> served (jpp-it fails rule 2 there)
      r1, miru: clear wins at 58 -> reference only, never served."""
    n = dict(C.N_PROMPTS)
    small = {(c, i) for c in C.CONDITIONS for i in (5, 15, 25)}      # 12 of 320: 0.0375 < 0.05

    def hit(kind, l, cond, i):
        r = i % 10
        if kind == "jpp-it":
            if 30 <= l <= 39 or l == 50 or l == 55:
                return r < 8
            if 40 <= l <= 49:
                return False if cond == "chat-begun" else r < 9
            if 51 <= l <= 54:
                return r < 3 or (cond, i) in small
            if l in (56, 57):
                return 3 <= r <= 5 or (r < 3 and l != 56)
        if kind == "rl-it":
            if 40 <= l <= 50:
                return r < 8
        if kind in ("r1", "miru") and l == 58:
            return r < 8
        if kind == "naive" and l == 59:
            return True
        return r < 3

    def stuck_word(kind, l, cond, i):
        if kind == "jpp-it" and l == 55 and cond == "bare" and i % 10 >= 6:
            return T_KIND, " kind"                         # 40% of bare at 55
        return None

    positions, prompts = [], []
    rng = np.random.default_rng(seed)
    kinds = ("jpp", "naive") + C.CANDIDATES + C.REFERENCE_LENSES
    for cond, cnt in n.items():
        for i in range(cnt):
            pid = f"{PREFIX[cond]}-{i + 1:02d}"
            prompts.append({"pid": pid, "cond": cond, "about_model": i % 10 == 9})
            ks = [None] if cond != "mid-reply" else [3, 9, 15, 21]
            for q, k in enumerate(ks):
                draws = None if cond != "mid-reply" else [["any"], ["any"], ["content_next", "any"], ["content_next"]][q]
                tid = int(rng.integers(2100, 4000))
                word = f" w{tid}"
                if cond == "bare" and i % 20 == 18:
                    tid, word = T_OWN, " own"                     # collapse target: left out
                p = {"i": len(positions), "pid": pid, "cond": cond, "draws": draws, "k": k, "pos": 10,
                     "read_tok": " x", "next_id": tid, "next_str": word, "next_content": True, "cont_ids": [],
                     "cont_strs": [], "ct_id": tid, "ct_str": word, "ct_offset": 0, "ct_prev_str": " x",
                     "model_top1_id": tid, "model_agrees": True, "r": {}}
                for kind in kinds:
                    lays = range(30, 60) if kind == "naive" else range(30, 59)
                    for l in lays:
                        sw = stuck_word(kind, l, cond, i)
                        if sw is not None:
                            ftop, fstr = sw
                        elif hit(kind, l, cond, i):
                            ftop, fstr = tid, word
                        else:
                            other = 6000 + (len(positions) * 131 + l * 7 + kinds.index(kind) * 1009) % 3000
                            ftop, fstr = other, f" v{other}"
                        p["r"].setdefault(kind, {})[str(l)] = {"u": [ftop, 1, 2, 3, 4], "us": [fstr, "a", "b", "c", "d"],
                                                               "f": [ftop, 6, 7, 8, 9], "fs": [fstr, "e", "f", "g", "h"],
                                                               "u0c": True}
                positions.append(p)
    expect = {
        ("jpp-it", "30-39"): {"passes": True, "served": True, "failed": []},
        ("jpp-it", "40-49"): {"passes": False, "served": False, "failed": ["2"]},
        ("jpp-it", "50"): {"passes": True, "served": True, "failed": []},
        ("jpp-it", "51-54"): {"passes": False, "served": False, "failed": ["5"]},
        ("jpp-it", "55"): {"passes": False, "served": False, "failed": ["3"]},
        ("jpp-it", "56-57"): {"passes": False, "served": False, "failed": ["4"]},
        ("jpp-it", "58"): {"passes": False, "served": False, "failed": ["1", "4", "5"]},
        ("rl-it", "30-39"): {"passes": False, "served": False, "failed": ["1", "4", "5"]},
        ("rl-it", "40-49"): {"passes": True, "served": True, "failed": []},
        ("rl-it", "50"): {"passes": True, "served": False, "failed": []},
        ("rl-it", "51-54"): {"passes": False, "served": False, "failed": ["1", "4", "5"]},
        ("rl-it", "55"): {"passes": False, "served": False, "failed": ["1", "4", "5"]},
        ("rl-it", "56-57"): {"passes": False, "served": False, "failed": ["1", "4", "5"]},
        ("rl-it", "58"): {"passes": False, "served": False, "failed": ["1", "4", "5"]},
    }
    return {"schema": "R3_05/1", "meta": {"selftest": True}, "prompts": prompts, "positions": positions}, expect


def selftest(record: bool = False) -> int:
    import R2_03_analyze as R2A

    fails: List[str] = []
    done: List[Tuple[bool, str]] = []
    t_start = time.time()

    def check(ok: bool, what: str) -> None:
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        done.append((bool(ok), what))
        if not ok:
            fails.append(what)

    # units
    check(abs(binom_sf_half(4, 4) - 0.0625) < 1e-12 and abs(binom_sf_half(5, 5) - 0.03125) < 1e-12
          and binom_sf_half(0, 0) == 1.0, "exact sign test (R2_03's binom_sf_half)")
    check(all(abs(holm({"a": 0.01, "b": 0.04, "c": 0.03})[k] - v) < 1e-12 for k, v in
              {"a": 0.03, "c": 0.06, "b": 0.06}.items()), "Holm's step-down (R2_03's holm)")
    check(significantly_worse({"hi": -0.0, "hi_raw": -3e-5}) and beats({"lo": 0.0, "lo_raw": 3e-5})
          and not beats({"lo": 0.0001, "lo_raw": 0.0}), "rules compare unrounded bounds")
    check(R3_DESIGN.bands == C.BANDS and len(R3_DESIGN.candidates) * len(R3_DESIGN.bands) == C.HOLM_M == 14
          and R3_DESIGN.boot_seed == 20261022 and R3_DESIGN.min_serve_layer == 30,
          "R3's design: 7 bands from code 30, 2 candidates (a Holm family of 14), bootstrap seed 20261022")

    # R2_03's own synthetic file: R3_06 under R2's design gives exactly R2_03's analysis
    data2, _, _ = R2A.synthetic()
    a = R2A.analyze(data2)
    b = analyze(R2_DESIGN, data2)
    same = all(_norm(a[k]) == _norm(b[k]) for k in ("bands", "decision", "counts", "context", "per_layer"))
    same &= all(_norm(a["sensitivity"][k][f]) == _norm(b["sensitivity"][k][f]) for k in a["sensitivity"]
                for f in ("verdicts", "changes", "decision", "bands"))
    same &= _norm({k: v for k, v in a["secondary"].items() if k != "miru_minus_r1"}) == _norm(b["secondary"])
    check(same, "R2_03's synthetic file (every rule branch of R2): R3_06 under R2's design equals R2_03 exactly")

    # R3's synthetic file
    data, expect = synthetic_r3()
    an = analyze(R3_DESIGN, data)
    for (lens, band), want in expect.items():
        ev = an["bands"][lens][band]
        got = {"passes": ev["passes"], "served": ev["served"], "failed": sorted(k for k, v in ev["rules"].items() if not v)}
        check(got == want, f"{lens} {band}: {got} (expected {want})" + ("" if got == want else f"; {ev['reasons']}"))
    r = an["bands"]
    check(all(ev["holm_m"] == 14 for d in r.values() for ev in d.values()), "the Holm family is the 14 candidate-band tests")
    check("precedence" in r["rl-it"]["50"].get("note", ""), "rl-it 50 passes but jpp-it takes precedence")
    check(an["decision"]["served_layers"] == {"jpp-it": list(range(30, 40)) + [50], "rl-it": list(range(40, 50))},
          f"served layers: {an['decision']['served_layers']}")
    ref = an["reference_bands"]
    check(ref["r1"]["58"]["passes"] and ref["miru"]["58"]["passes"] and not ref["r1"]["58"]["served"]
          and ref["r1"]["58"]["holm_m"] == 1 and an["decision"]["by_band"]["58"] is None,
          "R1 and Miru pass at 58 but are reference only: outside the Holm family, never served")
    check(abs(r["jpp-it"]["55"]["stuck"]["55"]["bare"]["share_raw"] - 0.4) < 1e-9,
          "jpp-it 55: stuck share 0.40 in bare")
    s1 = an["sensitivity"]["S1_without_about_model"]
    check(s1["prompts_dropped"] == 32, "S1 drops the 32 prompts about the model")
    s3 = an["sensitivity"]["S3_without_bare"]
    check({c["lens_band"] for c in s3["changes"]} >= {"jpp-it 55"},
          "S3 (without bare) clears jpp-it 55's stuck word: the verdict change is listed")
    check(an["sensitivity"]["S5_jpp_filter"].get("not_run") is not None,
          "S5 is reported as not run when the run recorded no J++-filtered readouts")
    data5 = json.loads(json.dumps(data))
    for p in data5["positions"]:
        for kind, d in p["r"].items():
            for l, rr in d.items():
                # under J++'s filter, jpp-it's top word at 30-39 would be a formatting-free miss
                miss = kind == "jpp-it" and 30 <= int(l) <= 39
                rr["fjs"] = ([" zz"] + rr["fs"][1:]) if miss else list(rr["fs"])
                rr["fj"] = ([9999] + rr["f"][1:]) if miss else list(rr["f"])
    an5 = analyze(R3_DESIGN, data5)
    ch5 = {c["lens_band"] for c in an5["sensitivity"]["S5_jpp_filter"]["changes"]}
    check(ch5 == {"jpp-it 30-39"} and an5["decision"] == an["decision"],
          "S5 (J++'s filter for every readout) reruns the rule on the recorded fjs and lists the change it causes; "
          "the primary decision is unchanged")

    # never below code 30: a design whose lowest band reaches code 29
    low = replace(R3_DESIGN, bands=(("29-30", (29, 30)),) + C.BANDS[1:])
    for p in data["positions"]:
        for kind in ("jpp", "jpp-it", "rl-it", "naive"):
            p["r"][kind]["29"] = dict(p["r"][kind]["30"])
    an_low = analyze(low, data)
    ev = an_low["bands"]["jpp-it"]["29-30"]
    check(ev["passes"] and not ev["served"] and "below code 30" in ev.get("note", "")
          and an_low["decision"]["by_band"]["29-30"] is None,
          "a band reaching code 29 that passes is never served")

    an2 = analyze(R3_DESIGN, synthetic_r3()[0])
    check(_norm(an2["bands"]) == _norm(analyze(R3_DESIGN, synthetic_r3()[0])["bands"]), "the bootstrap is deterministic")
    md = report_md(R3_DESIGN, an, {"selftest": True})
    check("SELF-TEST" in md and "| jpp-it | 30-39 |" in md and "S5" in md, "the report renders")
    print(f"R3_06 self-test: {'ALL PASS' if not fails else str(len(fails)) + ' FAILURES'}")
    if record:
        C.record_selftest(Path(__file__).name, done, time.time() - t_start)
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="inp", default=str(C.EVAL_JSON))
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--replication", default=None, help="R3_05's run on R2's prompts (reported only)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--record", action="store_true", help="with --selftest: keep the result in R3_selftests.json")
    ap.add_argument("--reproduce-r2", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest(record=args.record))
    if args.reproduce_r2:
        out = Path(args.out) if args.out else C.R2_REPRODUCTION_JSON
        rec = reproduce_r2(R2.EVAL_JSON, R2.ANALYSIS_JSON)
        out.write_text(json.dumps(rec, indent=1, default=str) + "\n", encoding="utf-8")
        for m in rec["mismatches"][:40]:
            print("MISMATCH", m)
        for row in rec["bands"]:
            print(f"  {row['lens']:5} {row['band']:6} passes {row['passes']!s:5} served {row['served']!s:5} "
                  f"failed {row['failed_rules']} Holm p {row['p_sign_holm']:.6g} (R2_03 {row['r2_p_sign_holm']:.6g})")
        print(f"{'R2 REPRODUCED EXACTLY' if rec['ok'] else 'R2 NOT REPRODUCED'}: {rec['compared']} comparisons, "
              f"{len(rec['mismatches'])} mismatches; wrote {out}")
        sys.exit(0 if rec["ok"] else 1)
    out = Path(args.out) if args.out else C.ANALYSIS_JSON
    report = Path(args.report) if args.report else (
        C.REPORT_SECTION if out.resolve() == C.ANALYSIS_JSON.resolve() else out.with_suffix(".md"))
    raw = Path(args.inp).read_bytes()
    data = json.loads(raw)
    if data.get("schema") != "R3_05/1":
        raise SystemExit(f"unexpected schema {data.get('schema')}")
    meta = data.get("meta", {})
    results = C.RESULTS.resolve()
    if meta.get("selftest") and (out.resolve().parent == results or report.resolve().parent == results):
        raise SystemExit("this input is self-test data; write the analysis and report outside experiments/results")
    an = analyze(R3_DESIGN, data)
    if args.replication:
        rep = json.loads(Path(args.replication).read_bytes())
        an["replication_r2_prompts"] = {k: analyze(R3_DESIGN, rep)[k] for k in ("bands", "decision", "counts")}
    an["input"] = {"path": str(args.inp), "sha256": hashlib.sha256(raw).hexdigest(),
                   "selftest": bool(meta.get("selftest")), "run_files_sha256": meta.get("files_sha256"),
                   "run_git": meta.get("git")}
    an["written_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    out.write_text(json.dumps(an, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    md = report_md(R3_DESIGN, an, meta)
    report.write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {out} and {report}")


if __name__ == "__main__":
    main()

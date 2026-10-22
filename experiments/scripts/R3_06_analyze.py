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

Secondary readings (decide nothing; R3_PREREG.md §1.2, §8.1, §8.2, §8.4), written before any lens file (D31):
  H2  R3_05_pullbacks.pt against Q0's pullbacks (J++, JLens, Miru) and R1's IT covectors, q10 and loud, per band
      30-39 / 40-49 / 50-58, 95% intervals over the 98 token-pair directions; read for jpp-it and rl-it against
      Q0's two yardsticks (R1's IT covectors vs J++; J++ vs JLens); split-half ceilings from the halves
  H3  energy in the top 100 output directions in y = fnw * output (diag(fnw) J), split-half debiased for jpp-it and
      rl-it, and inside base J++'s top 100; every fitted layer, read at code 30/35/40/45/50/55/58 against Miru
  H5  jpp-it - rl-it, filtered top-1, at every band, from the rule's own resamples, 95% and Bonferroni over seven
      bands; J++'s recall@10 beside it

Modes:
  default        R3_05_eval.json -> R3_06_analysis.json and R3_06_report_section.md (with H2, H3, H5)
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
    # R3: every candidate-band test is in the family whatever the data hold; a band with a readout missing at any of
    # its layers fails rule 2 (and keeps p = 1), so Holm is never run over fewer tests than registered
    require_all_bands: bool = False
    labels: Dict[str, str] = field(default_factory=dict)    # how a reference lens is labelled in the report

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
    kinds=("jpp", "naive") + C.CANDIDATES + C.REFERENCE_LENSES + C.SENSITIVITY_LENSES, candidates=C.CANDIDATES,
    reference=C.REFERENCE_LENSES + C.SENSITIVITY_LENSES, baselines=C.BASELINES, bands=C.BANDS,
    reference_only=frozenset(), precedence=C.PRECEDENCE, min_serve_layer=C.MIN_SERVE_LAYER, boot_seed=C.BOOT_SEED,
    n_boot=C.N_BOOT, overlap=dict(C.FIT_OVERLAP),
    overlap_why="without the prompts that paraphrase or share an answer topic with a fit source "
                "(R3_common.FIT_OVERLAP: WikiText records, J++ items, R1's fit set)",
    require_all_bands=True,
    labels={"r1": "reference", "miru": "reference", "jpp-it-full-macro": "full-macro sensitivity"})


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
            complete = bool(layers) and band_layers_for(t, lens, layers) and all(
                band_layers_for(t, b, layers) for b, _ in D.baselines)
            if complete:
                bands.setdefault(lens, {})[band] = evaluate_band(t, boot, lens, band, list(layers), conds)
            elif D.require_all_bands and layers:
                ev = evaluate_band(t, boot, lens, band, list(layers), conds)
                missing = {k: [l for l in layers if l not in t.layers.get(k, [])] for k in (lens,) + tuple(
                    b for b, _ in D.baselines)}
                ev["missing_layers"] = {k: v for k, v in missing.items() if v}
                ev["rules"]["2"] = False
                ev["reasons"].append(f"[rule 2] a readout is missing: {ev['missing_layers']}")
                ev["p_sign"] = 1.0
                bands.setdefault(lens, {})[band] = ev
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


# ------------------------------------------------------------------------------------------------ the replication
# R3_PREREG.md §8.6: R3_05 reads R2's 220 prompts again, with R2's draw seed. The test of R3's reading path (D21):
REPL_KINDS = ("jpp", "naive", "r1", "miru")           # the readouts R2 read, at the layers R2 read (40-59)
REPL_MIN_POSITIONS = 0.95                             # share of R2's read positions found again, same context and target
REPL_MIN_TOP1 = 0.99                                  # per readout: share of (position, layer) with R2's filtered top-1


def _pos_key(p: Dict[str, Any]) -> Tuple[Any, ...]:
    return (p["pid"], p["cond"], p["pos"], p.get("k"))


def replication_readouts(rep: Dict[str, Any], r2: Dict[str, Any]) -> Dict[str, Any]:
    """R3's reading of R2's prompts against R2_02_eval.json, position by position: a position is found again when
    the prompt, condition, read position, draws, next token, content target and (mid-reply) the reply up to the
    read position are R2's; then per readout and layer of R2's, whether the filtered top-1 id (and the unfiltered
    one) is R2's. Passes when at least REPL_MIN_POSITIONS of R2's positions are found again and, for every readout,
    at least REPL_MIN_TOP1 of its (position, layer) pairs have R2's filtered top-1. Every mismatch is listed."""
    reply3 = {p["pid"]: p.get("reply_ids") for p in rep.get("prompts", [])}
    reply2 = {p["pid"]: p.get("reply_ids") for p in r2.get("prompts", [])}
    mine = {_pos_key(p): p for p in rep["positions"]}
    found, lost = [], []
    for q in r2["positions"]:
        p = mine.get(_pos_key(q))
        same = p is not None and all(p.get(k) == q.get(k) for k in ("draws", "next_id", "ct_id", "cont_ids"))
        if same and q["cond"] == "mid-reply":
            k = q["k"]
            same = (reply3.get(q["pid"]) or [])[:k + 2] == (reply2.get(q["pid"]) or [])[:k + 2]
        (found if same else lost).append((q, p))
    per: Dict[str, Any] = {}
    mism: List[Dict[str, Any]] = []
    for kind in REPL_KINDS:
        n = same_f = same_u = 0
        for q, p in found:
            for l, rq in q["r"].get(kind, {}).items():
                rp = p["r"].get(kind, {}).get(l)
                n += 1
                if rp is not None and rp["f"][0] == rq["f"][0]:
                    same_f += 1
                elif len(mism) < 500:
                    mism.append({"pid": q["pid"], "pos": q["pos"], "k": q.get("k"), "kind": kind, "layer": int(l),
                                 "r2_f0": rq["f"][0], "r3_f0": None if rp is None else rp["f"][0]})
                same_u += int(rp is not None and rp["u"][0] == rq["u"][0])
        per[kind] = {"pairs": n, "filtered_top1_same": same_f, "unfiltered_top1_same": same_u,
                     "filtered_share": r4(same_f / n) if n else None, "unfiltered_share": r4(same_u / n) if n else None,
                     "ok": n > 0 and same_f / n >= REPL_MIN_TOP1}
    n2 = len(r2["positions"])
    share = len(found) / n2 if n2 else 0.0
    return {"r2_positions": n2, "found_again": len(found), "found_share": r4(share),
            "lost": [{"pid": q["pid"], "pos": q["pos"], "k": q.get("k")} for q, _ in lost][:200],
            "readouts": per, "mismatches_listed": mism, "n_mismatches": sum(v["pairs"] - v["filtered_top1_same"]
                                                                            for v in per.values()),
            "tolerance": {"positions": REPL_MIN_POSITIONS, "filtered_top1": REPL_MIN_TOP1},
            "ok": share >= REPL_MIN_POSITIONS and all(v["ok"] for v in per.values())}


def replication_verdicts(an2: Dict[str, Any], ref: Dict[str, Any]) -> Dict[str, Any]:
    """R2's design applied to R3's reading of R2's prompts against R2_03_analysis.json: every lens-band's pass and
    failed rules, and the decision."""
    got, want = verdicts(an2["bands"]), verdicts(ref["bands"])
    diff = [{"lens_band": k, "r3_reading": got.get(k), "r2": want.get(k)} for k in sorted(set(got) | set(want))
            if (got.get(k) or {}).get("passes") != (want.get(k) or {}).get("passes")
            or sorted((got.get(k) or {}).get("failed_rules", [])) != sorted((want.get(k) or {}).get("failed_rules", []))]
    return {"differ": diff, "decision_same": _norm(an2["decision"]) == _norm(ref["decision"]),
            "ok": not diff and _norm(an2["decision"]) == _norm(ref["decision"])}


def replication(rep: Dict[str, Any], r2_eval: Dict[str, Any], r2_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """The registered replication (R3_PREREG.md §8.6): the readouts compared with R2's, R2's design on R3's reading
    (its verdicts against R2_03's), and R3's design on the same prompts (S2 with R2's own overlap list), reported
    whole (bands, reference bands, per layer, decision, counts). Decides nothing; on a failure the report says so
    first, and no R3 pass is a recommendation until an amendment explains the difference."""
    D3 = replace(R3_DESIGN, overlap=dict(R2.R1_OVERLAP),
                 overlap_why=R2_DESIGN.overlap_why + "; the replication set, so R2's own list")
    a3 = analyze(D3, rep)
    a2 = analyze(R2_DESIGN, rep)
    ro = replication_readouts(rep, r2_eval)
    rv = replication_verdicts(a2, r2_analysis)
    return {"ok": ro["ok"] and rv["ok"], "readouts_against_r2_02": ro, "r2_design_against_r2_03": rv,
            "r2_design": {k: a2[k] for k in ("bands", "decision", "counts")},
            "r3_design": {k: a3[k] for k in ("bands", "reference_bands", "per_layer", "decision", "counts",
                                              "sensitivity")},
            "on_failure": "reported first in the report; the rule is still applied to R3's prompts, but no R3 pass is "
                          "a recommendation to a serving track until an amendment explains the difference"}


def check_input(data: Dict[str, Any], counts: Dict[str, int], prompts_sha: str, what: str) -> List[str]:
    """What must hold of R3_05's output before the rule is applied: the registered prompt counts and the prompts'
    sha256 (and, for R3's prompts, the one a passing R3_01_check.json recorded)."""
    bad = []
    got = dict(collections.Counter(p["cond"] for p in data.get("prompts", [])))
    if got != dict(counts):
        bad.append(f"{what}: prompt counts {got}, registered {dict(counts)}")
    if (data.get("meta") or {}).get("prompts_sha256") != prompts_sha:
        bad.append(f"{what}: prompts sha256 {(data.get('meta') or {}).get('prompts_sha256')} is not {prompts_sha}")
    return bad


def check_family(D: Design, an: Dict[str, Any]) -> List[str]:
    """Every candidate-band test is present and Holm ran over exactly the registered family."""
    bad = []
    if an["constants"]["HOLM_M"] != len(D.candidates) * len(D.bands):
        bad.append(f"Holm family of {an['constants']['HOLM_M']} tests, registered {len(D.candidates) * len(D.bands)}")
    for lens in D.candidates:
        for band, _ in D.bands:
            ev = an["bands"].get(lens, {}).get(band)
            if ev is None:
                bad.append(f"{lens} {band}: no test")
            elif ev.get("holm_m") != len(D.candidates) * len(D.bands):
                bad.append(f"{lens} {band}: Holm over {ev.get('holm_m')} tests")
    return bad


# ------------------------------------------------------------------------------------------------ H2, H3, H5
# The secondary readings D31 required to be written before any lens file existed (R3_PREREG.md §1.2, §8.1, §8.2,
# §8.4). They decide nothing (§1.1: only H1 decides); each is reported whatever it shows.
PULLBACKS_PT = C.RESULTS / "R3_05_pullbacks.pt"       # R3_05's pullbacks through R3's lens files (gitignored)
RECALL_JSON = C.RESULTS / "R3_05_recall.json"         # R3_05's J++ held-out recall@10 (§8.3)
R1_SPECTRUM_JSON = C.RESULTS / "R1_15_spectrum.json"   # R1 §3.2's figures (R1_15_spectrum.py), for H3's code check

# H2 (§8.1): pullbacks d.J_l of R1's 128 directions; cosines as R1_04 (the masked cosine per direction and layer,
# averaged over a band's layers, then over the 98 token-pair directions), with 95% percentile intervals over them
H2_BANDS: Tuple[Tuple[str, int, int], ...] = (("L30-39", 30, 39), ("L40-49", 40, 49), ("L50-58", 50, 58))
H2_MASKS: Tuple[str, ...] = ("q10", "loud")
H2_LENSES: Tuple[str, ...] = ("jpp-it", "rl-it", "jpp-it-full-macro", "jpp-it-A", "jpp-it-B", "rl-it-A", "rl-it-B")
H2_REFS: Tuple[str, ...] = ("jpp", "jlens", "miru", "r1_it")
H2_SPLIT: Tuple[Tuple[str, str, str], ...] = (("jpp-it", "jpp-it-A", "jpp-it-B"), ("rl-it", "rl-it-A", "rl-it-B"))
H2_READ: Tuple[str, ...] = ("jpp-it", "rl-it")
H2_N_BOOT = 2000
# the two yardsticks of H2's reading, as Q0 measured them (R3_Q0_pullbacks.json, q10, the 98 token pairs)
H2_Q0_LOW = "r1 IT (4-bit, same-position VJPs) vs jpp"   # R1's instruct covectors against base J++ (0.27/0.29/0.29)
H2_Q0_HIGH = "fp32: jpp vs jlens"                         # J++ against JLens, two methods on one model (0.77/0.68/0.66)
H2_REGISTERED_2DP = {"low": {"L30-39": 0.27, "L40-49": 0.29, "L50-58": 0.29},     # the figures §1.2 prints
                     "high": {"L30-39": 0.77, "L40-49": 0.68, "L50-58": 0.66}}
H2_LABELS = {
    "recipe": "the recipe sets the lens",
    "checkpoint": {"jpp-it": "the checkpoint, with the recipe version and the items, dominates",
                   "rl-it": "the checkpoint, with the recipe version, dominates"},
    "both": "both",
}

# H3 (§8.2): the share of a lens's squared Frobenius norm in its top 100 output directions, in R1's coordinates
H3_K = 100
H3_JUDGED: Tuple[int, ...] = (30, 35, 40, 45, 50, 55, 58)        # R1 §3.2's seven layers; reported at all 29
H3_LENSES: Tuple[str, ...] = ("jpp-it", "rl-it", "jpp", "jlens", "miru")
H3_HALVES: Dict[str, Tuple[str, str]] = {"jpp-it": ("jpp-it-A", "jpp-it-B"), "rl-it": ("rl-it-A", "rl-it-B")}
H3_READ: Tuple[str, ...] = ("jpp-it", "rl-it")
H3_R1_TOL = 0.003                                              # R1 stored 3 decimals from a randomised SVD (q 320)

# H5 (§1.2, §8.4): jpp-it - rl-it, filtered top-1, at every band; Bonferroni over the seven bands
H5_A, H5_B = "jpp-it", "rl-it"
H5_PCT95 = (2.5, 97.5)
H5_PCT_BONF = (0.357, 99.643)                                  # §1.2, as printed: 99.29% over seven bands


def _tp_summary(v_K: np.ndarray, idx: np.ndarray) -> Dict[str, Any]:
    """The mean over the 98 token-pair directions of v [K] and its 95% percentile interval (resampled directions)."""
    t = v_K[list(C.TOKPAIR_DIRS)]
    lo, hi = np.percentile(t[idx].mean(1), [2.5, 97.5])
    return {"tokpair98": r4(float(t.mean())), "ci95": [r4(float(lo)), r4(float(hi))],
            "tokpair98_raw": float(t.mean()), "ci95_raw": [float(lo), float(hi)]}


def h2_band_summary(cos_KL: np.ndarray, layers: Sequence[int], idx: np.ndarray) -> Dict[str, Any]:
    """cos [K, len(layers)] -> per H2 band: the per-direction mean over the band's layers, then the token-pair mean
    and interval (R1_04's order)."""
    col = {l: i for i, l in enumerate(layers)}
    out = {}
    for name, a, b in H2_BANDS:
        if all(l in col for l in range(a, b + 1)):
            out[name] = _tp_summary(cos_KL[:, [col[l] for l in range(a, b + 1)]].mean(1), idx)
    return out


def h2_yardsticks(q0: Dict[str, Any]) -> Dict[str, Any]:
    """H2's two yardsticks from R3_Q0_pullbacks.json (q10, token-pair means at code 30-39 / 40-49 / 50-58)."""
    def row(key):
        b = q0["comparisons"][key]["q10"]["bands"]
        return {name: b[name]["tokpair98"] for name, _, _ in H2_BANDS}
    return {"low": row(H2_Q0_LOW), "high": row(H2_Q0_HIGH), "low_key": H2_Q0_LOW, "high_key": H2_Q0_HIGH,
            "source": "R3_Q0_pullbacks.json (4 decimals, as Q0 stored them)"}


def h2_reading_one(lens: str, s: Dict[str, Any], low: float, high: float) -> Dict[str, Any]:
    """§1.2, for one lens at one band, in q10: "the recipe sets the lens" is refuted if the interval's upper end is at
    or below `low` (R1's instruct covectors against J++); "the checkpoint ... dominates" is refuted if its lower end
    is at or above `high` (J++ against JLens); otherwise the reading is "both"."""
    lo, hi = s["ci95_raw"]
    rec_ref = hi <= low
    ck_ref = lo >= high
    refuted = [H2_LABELS["recipe"]] * rec_ref + [H2_LABELS["checkpoint"][lens]] * ck_ref
    reading = "both" if not refuted else ("; ".join(f"'{r}' refuted" for r in refuted))
    return {"reading": reading, "recipe_sets_the_lens_refuted": rec_ref, "checkpoint_dominates_refuted": ck_ref,
            "cos_vs_jpp": s, "low": low, "high": high}


def h2_tables(pb: Dict[str, Any], refs: Dict[str, Any], masks: Dict[str, Any], layers: Sequence[int],
              yard: Dict[str, Any], seed: Sequence[int] = (C.BOOT_SEED, 2), n_boot: int = H2_N_BOOT) -> Dict[str, Any]:
    """H2 (§1.2, §8.1). pb: {lens: [K, len(layers), D]} (R3_05_pullbacks.pt); refs: {ref: [K, 59, D]} over code
    layers 0-58 (Q0's J++, JLens and Miru pullbacks, R1's IT covectors); masks: {"q10", "loud": [>=59, D]}. Every
    lens against every reference, the split-half ceilings, and the reading for jpp-it and rl-it in q10 (loud
    beside it). One set of resampled directions serves every comparison."""
    import R3_Q0_pullbacks as Q0

    layers = list(layers)
    rng = np.random.default_rng(list(seed))
    tp = list(C.TOKPAIR_DIRS)
    idx = rng.integers(0, len(tp), size=(n_boot, len(tp)))
    M = {mk: masks[mk][layers].float() for mk in H2_MASKS}

    def cmp(a, b):
        return {mk: h2_band_summary(Q0.mcos(a.float(), b.float(), M[mk]).double().numpy(), layers, idx)
                for mk in H2_MASKS}

    out: Dict[str, Any] = {"layers": layers, "bands": [b for b, _, _ in H2_BANDS], "masks": list(H2_MASKS),
                           "boot": {"n": n_boot, "seed": list(seed), "over": "the 98 token-pair directions (30-127)",
                                    "paired": "the same resampled directions for every comparison"},
                           "vs": {}, "split_half": {}, "yardsticks": yard}
    for lens in H2_LENSES:
        if lens not in pb:
            out["vs"][lens] = {"missing": True}
            continue
        out["vs"][lens] = {ref: cmp(pb[lens], refs[ref][:, layers]) for ref in H2_REFS if ref in refs}
    for whole, a, b in H2_SPLIT:
        if a in pb and b in pb:
            out["split_half"][whole] = cmp(pb[a], pb[b])
    reading: Dict[str, Any] = {}
    for lens in H2_READ:
        v = out["vs"].get(lens, {})
        for name, _, _ in H2_BANDS:
            s = (v.get("jpp") or {}).get("q10", {}).get(name)
            if s is None:
                reading.setdefault(lens, {})[name] = {"reading": "not read: no pullbacks"}
                continue
            r = h2_reading_one(lens, s, yard["low"][name], yard["high"][name])
            r2 = h2_reading_one(lens, s, H2_REGISTERED_2DP["low"][name], H2_REGISTERED_2DP["high"][name])
            r["reading_with_the_2dp_figures"] = r2["reading"]
            r["loud_beside"] = v["jpp"]["loud"].get(name)
            reading.setdefault(lens, {})[name] = r
    out["reading"] = reading
    out["note"] = ("H2 is read for jpp-it and rl-it alike; rl-it carries the checkpoint and the recipe's code version "
                   "without the items and weights, so the two side by side show the weighting's share. No reading is "
                   "laid on the checkpoint alone (§1).")
    return out


def load_h2_inputs(pb_path: Path = PULLBACKS_PT, full_hash: bool = True) -> Tuple[Dict[str, Any], ...]:
    """R3_05's pullbacks, Q0's (J++, JLens, Miru), R1's IT covectors, the q10 masks and Q0's yardsticks."""
    import torch

    import R3_Q0_pullbacks as Q0

    for p, sha, size in ((C.R1_GBAR_PATH, C.R1_GBAR_SHA256, C.R1_GBAR_SIZE),
                         (C.QUIET_STATS_PATH, C.QUIET_STATS_SHA256, C.QUIET_STATS_SIZE)):
        rec = C.check_pinned_file(p, sha, size, full=full_hash)
        if not rec["size_ok"] or rec["sha256_ok"] is False:
            raise SystemExit(f"{p} is not the pinned file ({rec})")
    r3 = torch.load(pb_path, map_location="cpu", weights_only=False)
    q0 = torch.load(C.Q0_PT, map_location="cpu", weights_only=False)
    G = torch.load(C.R1_GBAR_PATH, map_location="cpu", mmap=True, weights_only=False)
    if list(r3["layers"]) != list(C.FIT_LAYERS) or list(q0["layers"]) != list(range(59)):
        raise SystemExit(f"pullback layers: R3_05 {r3['layers']}, Q0 {q0['layers']}")
    if not (list(r3["names"]) == list(q0["names"]) == list(G["names"])):
        raise SystemExit("the pullbacks are not of R1's 128 directions in one order")
    refs = {k: v.float() for k, v in q0["pb"].items()}
    refs["r1_it"] = G["mix"][:, :59].float()
    masks = Q0.quiet_masks()
    q0j = json.loads(C.Q0_JSON.read_text(encoding="utf-8"))
    files = {"pullbacks": {"path": str(pb_path), "sha256": C.sha256_file(pb_path)},
             "q0_pt": {"path": str(C.Q0_PT), "sha256": C.sha256_file(C.Q0_PT)},
             "q0_json": {"path": str(C.Q0_JSON), "sha256": C.sha256_file(C.Q0_JSON)}}
    yard = h2_yardsticks(q0j)
    yard["context_copied_from_R1_04"] = {k: v[3:] for k, v in q0j.get("yardsticks_copied_from_R1_04", {}).items()}
    return r3["pb"], refs, masks, yard, files


def _gram_top(M, k: int, vectors: bool):
    """M [D_out, D_in] -> (share of ||M||_F^2 in its top k left singular directions, U_k [D_out, k] or None), from
    the fp64 Gram matrix M M^T."""
    import torch

    X = M.double()
    G = X @ X.T
    tot = float(torch.diagonal(G).sum())
    if vectors:
        e, V = torch.linalg.eigh(G)
        return float(e[-k:].sum()) / tot, V[:, -k:]
    e = torch.linalg.eigvalsh(G)
    return float(e[-k:].sum()) / tot, None


def _debiased_share(U, A, B) -> float:
    """sum_i u_i^T (A B^T) u_i / <A, B>_F (§8.2), A and B the half-lenses in y, U [D_out, k]."""
    return float(((U.T @ A) * (U.T @ B)).sum()) / float((A * B).sum())


def _in_subspace(M, U) -> float:
    X = M.double()
    return float((U.T @ X).pow(2).sum()) / float(X.pow(2).sum())


def h3_layer(mats: Dict[str, Any], fnw, k: int = H3_K) -> Dict[str, Any]:
    """One layer of H3 (§8.2). mats: {name: J [D, D]}, J read h @ J.T (rows the output side). In R1's coordinates
    y = fnw * output the matrix is diag(fnw) J. Per lens: the share of its squared Frobenius norm in its own top k
    output directions (y, and raw beside it), and inside base J++'s top-k output subspace (y; raw beside); for jpp-it
    and rl-it also the split-half debiased share sum_i u_i^T (A~ B~^T) u_i / <A~, B~>_F, U the top k left singular
    vectors of the whole lens in y, A~ and B~ the half-lenses in y."""
    w = fnw.double()[:, None]
    Y = {n: w * m.double() for n, m in mats.items()}
    U_j = U_jr = None
    if "jpp" in mats:
        _, U_j = _gram_top(Y["jpp"], k, True)
        _, U_jr = _gram_top(mats["jpp"].double(), k, True)
    out: Dict[str, Any] = {}
    for n in H3_LENSES:
        if n not in mats:
            continue
        need_u = n in H3_HALVES and all(h in mats for h in H3_HALVES[n])
        sy, U = _gram_top(Y[n], k, need_u)
        sr, _ = _gram_top(mats[n].double(), k, False)
        rec = {"share_y": sy, "share_raw": sr}
        if U_j is not None:
            rec["in_jpp_top_y"] = _in_subspace(Y[n], U_j)
            rec["in_jpp_top_raw"] = _in_subspace(mats[n].double(), U_jr)
        if need_u:
            A, B = (Y[h] for h in H3_HALVES[n])
            rec["share_y_debiased"] = _debiased_share(U, A, B)
            rec["halves_inner_y"] = float((A * B).sum())
        out[n] = rec
    return out


def h3_verdict(plain: float, debiased: Optional[float], miru: float) -> Dict[str, Any]:
    """§1.2 at one layer: "supported" if the lens's plain share (y) is above Miru's; "refuted" if even its debiased
    share is at or below Miru's; otherwise "not separated". Both conditions holding at once (a debiased share below
    the plain one, Miru between them) is not ordered by the registration and is reported as such."""
    sup = plain > miru
    ref = debiased is not None and debiased <= miru
    if sup and ref:
        v = "supported and refuted (the debiased share is below the plain one; the registration orders neither)"
    elif sup:
        v = "supported"
    elif ref:
        v = "refuted"
    else:
        v = "not separated"
    return {"verdict": v, "supported_condition": sup, "refuted_condition": ref, "plain_y": plain,
            "debiased_y": debiased, "miru_plain_y": miru}


def h3_tables(read: Callable[[str, int], Any], fnw, layers: Sequence[int], k: int = H3_K,
              judged: Sequence[int] = H3_JUDGED) -> Dict[str, Any]:
    """H3 at every layer in `layers` (reported) and its reading at the judged ones. read(name, layer) -> J [D, D] or
    None, for the names of H3_LENSES and the four half-lenses."""
    per: Dict[str, Any] = {}
    names = H3_LENSES + tuple(h for hs in H3_HALVES.values() for h in hs)
    for l in layers:
        mats = {}
        for n in names:
            m = read(n, l)
            if m is not None:
                mats[n] = m.float()
        per[str(l)] = h3_layer(mats, fnw, k)
        del mats
    reading: Dict[str, Any] = {}
    for lens in H3_READ:
        for l in judged:
            d = per.get(str(l), {})
            if lens not in d or "miru" not in d:
                reading.setdefault(lens, {})[str(l)] = {"verdict": "not read: a lens is missing at this layer"}
                continue
            reading.setdefault(lens, {})[str(l)] = h3_verdict(d[lens]["share_y"], d[lens].get("share_y_debiased"),
                                                              d["miru"]["share_y"])
    return {"k": k, "layers": list(layers), "judged_layers": list(judged), "coordinates": "y = fnw * output: "
            "diag(fnw) J, J read h @ J.T (rows the output side), fnw the final norm's weight (V6_quiet_dims_stats.pt, "
            "as R1_15)", "per_layer": per, "reading": reading,
            "note": "Miru (a plain J-lens on the instruct model) also differs in code and prompts: a 'supported' says "
                    "that the recipe as R3 runs it concentrates more than Miru's lens does, not which of its steps does. "
                    "JLens and base J++ (the shipped file) are reported beside. Raw shares are reported, never read."}


def h3_readers(weights_json: Path = C.WEIGHTS_JSON, full_hash: bool = True):
    """read(name, layer) over R3_04's lens files (their sha256 as R3_04_weights.json records them), the pinned J++,
    JLens and Miru files; and fnw. Returns (read, fnw, files)."""
    import torch

    import R3_Q0_pullbacks as Q0

    lf = json.loads(weights_json.read_text(encoding="utf-8"))["lens_files"]
    files: Dict[str, Any] = {}
    J: Dict[str, Any] = {}
    for n in ("jpp-it", "rl-it") + tuple(h for hs in H3_HALVES.values() for h in hs):
        p = C.LENS_FILES[n]
        if Path(lf[n]["path"]).resolve() != p.resolve():
            raise SystemExit(f"{n}: R3_04_weights.json names {lf[n]['path']}, not {p}")
        sha = C.sha256_file(p) if full_hash else None
        if full_hash and sha != lf[n]["sha256"]:
            raise SystemExit(f"{n}: {p} is not the file R3_04 wrote (sha256 {sha}, recorded {lf[n]['sha256']})")
        files[n] = {"path": str(p), "sha256": lf[n]["sha256"], "sha256_checked": full_hash}
        J[n] = torch.load(p, map_location="cpu", mmap=True, weights_only=True)["J"]
    for n, p, sha, size in (("jpp", C.JPP_BASE_PATH, C.JPP_BASE_SHA256, C.JPP_BASE_SIZE),
                            ("jlens", C.JLENS_PATH, C.JLENS_SHA256, C.JLENS_SIZE),
                            ("miru", C.MIRU_PATH, C.MIRU_SHA256, C.MIRU_SIZE)):
        rec = C.check_pinned_file(p, sha, size, full=full_hash)
        if not rec["size_ok"] or rec["sha256_ok"] is False:
            raise SystemExit(f"{p} is not the pinned file ({rec})")
        files[n] = rec
    other = {n: Q0.lens_reader(n) for n in ("jpp", "jlens", "miru")}
    rec = C.check_pinned_file(C.QUIET_STATS_PATH, C.QUIET_STATS_SHA256, C.QUIET_STATS_SIZE, full=full_hash)
    if not rec["size_ok"] or rec["sha256_ok"] is False:
        raise SystemExit(f"{C.QUIET_STATS_PATH} is not the pinned file ({rec})")
    fnw = torch.load(C.QUIET_STATS_PATH, weights_only=False, map_location="cpu")["fnw"].float()

    def read(n: str, l: int):
        if n in J:
            return J[n][l]
        if n in other:
            return other[n](l)
        return None
    return read, fnw, files


def h5_tables(D: Design, data: Dict[str, Any], recall: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """H5 (§1.2, §8.4): jpp-it - rl-it, filtered top-1 (the primary metric), pooled over the decision conditions and
    per condition, at every band whether or not either lens beats the naked eye there; paired intervals from the
    rule's own 2000 prompt-level resamples (the same Table and Boot, so the same indices), at 95% and Bonferroni-
    adjusted over the seven bands (the 0.357th and 99.643th percentiles). Per band, on the pooled Bonferroni
    interval: "helps" if its lower end is above 0, "hurts" if its upper end is below 0, otherwise "not shown". H5 is
    refuted if no band reads "helps". J++'s own held-out recall@10 (R3_05_recall.json) is reported beside, for
    both lenses; it decides nothing."""
    t = Table(D, data)
    boot = Boot(t, D.n_boot, D.boot_seed)
    bands: Dict[str, Any] = {}
    for band, layers in D.bands:
        L = list(layers)
        have = band_layers_for(t, H5_A, L) and band_layers_for(t, H5_B, L)
        per: Dict[str, Any] = {}
        sums, ns, tot = np.zeros(boot.B), 0, 0.0
        missing = not have
        for c in D.all_conds:
            ok, ix = boot.idx(c, 0)
            if not len(ok):
                per[c] = {"n": 0, "mean": None}
                continue
            x = (t.prompt_scores(PRIMARY, H5_A, L, c) - t.prompt_scores(PRIMARY, H5_B, L, c))[ok] if have else None
            if x is None or np.isnan(x).any():
                per[c] = {"n": int(len(ok)), "mean": None, "note": "readout missing"}
                if c in D.conditions:
                    missing = True
                continue
            bs = x[ix].sum(1) / len(ok)
            per[c] = _h5_stat(len(ok), float(x.mean()), bs)
            if c in D.conditions:
                sums += x[ix].sum(1)
                ns += len(ok)
                tot += float(x.sum())
        if missing or not ns:
            pooled = {"n": ns, "mean": None}
            reading = "not shown (a readout is missing)"
        else:
            pooled = _h5_stat(ns, tot / ns, sums / ns)
            lo_b, hi_b = pooled["bonferroni_raw"]
            reading = "helps" if lo_b > 0 else ("hurts" if hi_b < 0 else "not shown")
        bands[band] = {"layers": L, "pooled": pooled, "by_cond": per, "reading": reading}
    out = {"lenses": [H5_A, H5_B], "metric": PRIMARY, "conds": list(D.conditions),
           "boot": {"n": D.n_boot, "seed": D.boot_seed, "same_resamples_as": "the decision rule (Table, Boot)"},
           "percentiles": {"ci95": list(H5_PCT95), "bonferroni": list(H5_PCT_BONF), "bands": len(D.bands)},
           "bands": bands, "helps_at": [b for b, v in bands.items() if v["reading"] == "helps"],
           "hurts_at": [b for b, v in bands.items() if v["reading"] == "hurts"]}
    out["refuted"] = not out["helps_at"]
    out["verdict"] = ("H5 refuted: no band reads 'helps'" if out["refuted"] else
                      "Jacobian filtering helps at " + ", ".join(out["helps_at"]))
    if recall is not None:
        out["recall_at_10"] = {m: {"evals": v.get("evals"), "n_items": v.get("n_items"),
                                   "lenses": {k: v["lenses"].get(k) for k in (H5_A, H5_B, "jpp-it-full-macro")
                                              if k in v.get("lenses", {})}}
                               for m, v in recall.get("macros", {}).items()}
        out["recall_note"] = ("J++'s own held-out recall@10 decides nothing: jpp-it's weights were fitted to maximise "
                              "pass@10 on fitting items of the same evals and rl-it was never tuned, so that "
                              "comparison favours jpp-it (§1.2)")
    else:
        out["recall_at_10"] = None
    return out


def _h5_stat(n: int, mean: float, bs: np.ndarray) -> Dict[str, Any]:
    lo, hi = np.percentile(bs, list(H5_PCT95))
    blo, bhi = np.percentile(bs, list(H5_PCT_BONF))
    return {"n": int(n), "mean": r4(mean), "ci95": [r4(lo), r4(hi)], "bonferroni": [r4(blo), r4(bhi)],
            "mean_raw": mean, "ci95_raw": [float(lo), float(hi)], "bonferroni_raw": [float(blo), float(bhi)]}


def secondary_readings(D: Design, data: Dict[str, Any], pb_path: Path = PULLBACKS_PT,
                       weights_json: Path = C.WEIGHTS_JSON, recall_json: Path = RECALL_JSON,
                       full_hash: bool = True) -> Dict[str, Any]:
    """H2, H3 and H5 on the registered inputs. An input that does not exist is said, and its reading not run."""
    out: Dict[str, Any] = {}
    rec = json.loads(recall_json.read_text(encoding="utf-8")) if recall_json.is_file() else None
    out["H5"] = h5_tables(D, data, rec)
    if rec is None:
        out["H5"]["recall_missing"] = f"{recall_json} does not exist"
    if pb_path.is_file() and C.Q0_PT.is_file():
        pb, refs, masks, yard, files = load_h2_inputs(pb_path, full_hash)
        out["H2"] = h2_tables(pb, refs, masks, C.FIT_LAYERS, yard)
        out["H2"]["files"] = files
    else:
        out["H2"] = {"not_run": "does not exist: " + ", ".join(str(q) for q in (pb_path, C.Q0_PT) if not q.is_file())}
    if weights_json.is_file() and all(C.LENS_FILES[n].is_file() for n in ("jpp-it", "rl-it", "jpp-it-A", "jpp-it-B",
                                                                          "rl-it-A", "rl-it-B")):
        read, fnw, files = h3_readers(weights_json, full_hash)
        out["H3"] = h3_tables(read, fnw, C.FIT_LAYERS)
        out["H3"]["files"] = files
    else:
        out["H3"] = {"not_run": "does not exist: " + ", ".join(str(q) for q in (weights_json,) + tuple(
            C.LENS_FILES[n] for n in ("jpp-it", "rl-it", "jpp-it-A", "jpp-it-B", "rl-it-A", "rl-it-B"))
            if not q.is_file())}
    for k in ("H2", "H3"):
        if "not_run" in out[k]:
            print(f"WARNING: {k} not run: {out[k]['not_run']}", flush=True)
    return out


def secondary_md(rd: Dict[str, Any]) -> List[str]:
    L: List[str] = ["Secondary readings (R3_PREREG.md §1.2, §8; they decide nothing):", ""]
    h2 = rd.get("H2", {})
    if "not_run" in h2:
        L += [f"- H2 not run: {h2['not_run']}", ""]
    else:
        L.append("H2, agreement with base J++ in q10 (98 token-pair directions, 95% interval; loud beside), against "
                 "R1's instruct covectors vs J++ (low) and J++ vs JLens (high), as Q0 measured them:")
        L.append("")
        L.append("| lens | band | vs J++ q10 | loud | low | high | split-half q10 | reading |")
        L.append("|---|---|---|---|---|---|---|---|")
        for lens, d in h2["reading"].items():
            for band, r in d.items():
                if "cos_vs_jpp" not in r:
                    L.append(f"| {lens} | {band} | n/a | | | | | {r['reading']} |")
                    continue
                s, ld = r["cos_vs_jpp"], r.get("loud_beside") or {}
                sh = (h2["split_half"].get(lens) or {}).get("q10", {}).get(band, {})
                L.append(f"| {lens} | {band} | {s['tokpair98']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}] | "
                         f"{fmt_num(ld.get('tokpair98'))} | {r['low']:.4f} | {r['high']:.4f} | "
                         f"{fmt_num(sh.get('tokpair98'))} | {r['reading']} |")
        L.append("")
    h3 = rd.get("H3", {})
    if "not_run" in h3:
        L += [f"- H3 not run: {h3['not_run']}", ""]
    else:
        L.append(f"H3, the share of a lens's energy in its own top {h3['k']} output directions in y = fnw * output "
                 "(jpp-it and rl-it also split-half debiased), against Miru's (plain J-lens, instruct):")
        L.append("")
        L.append("| layer | jpp-it (debiased) | rl-it (debiased) | Miru | JLens | base J++ | jpp-it | rl-it |")
        L.append("|---|---|---|---|---|---|---|---|")
        for l in h3["judged_layers"]:
            d = h3["per_layer"].get(str(l), {})

            def sh(n, key="share_y"):
                return fmt_num((d.get(n) or {}).get(key))
            L.append(f"| {l} | {sh('jpp-it')} ({sh('jpp-it', 'share_y_debiased')}) | {sh('rl-it')} "
                     f"({sh('rl-it', 'share_y_debiased')}) | {sh('miru')} | {sh('jlens')} | {sh('jpp')} | "
                     f"{h3['reading']['jpp-it'][str(l)]['verdict']} | {h3['reading']['rl-it'][str(l)]['verdict']} |")
        L.append("")
    h5 = rd.get("H5", {})
    if h5:
        L.append("H5, jpp-it - rl-it, filtered top-1, pooled (95% interval; Bonferroni over seven bands, 99.29%):")
        L.append("")
        L.append("| band | jpp-it - rl-it | 95% | Bonferroni | reading |")
        L.append("|---|---|---|---|---|")
        for band, v in h5["bands"].items():
            p = v["pooled"]
            if p.get("mean") is None:
                L.append(f"| {band} | n/a | | | {v['reading']} |")
            else:
                L.append(f"| {band} | {p['mean']:+.3f} | [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] | "
                         f"[{p['bonferroni'][0]:+.3f}, {p['bonferroni'][1]:+.3f}] | {v['reading']} |")
        L.append("")
        L.append(f"{h5['verdict']}.")
        L.append("")
    return L


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
    rp = an.get("replication_r2_prompts")
    if rp is not None and not rp["ok"]:
        L.append("**R3's reading path did not reproduce R2's readings on R2's prompts (§8.6): no pass below is a "
                 "recommendation to a serving track until an amendment explains the difference.**")
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
             "p (Holm for candidates; raw, outside Holm, for the rest) | worse in | max stuck (layer, condition, "
             "word) | verdict |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for group in (an["bands"], an.get("reference_bands", {})):
        for lens, d in group.items():
            for band, _ in D.bands:
                ev = d.get(band)
                if ev is None:
                    continue
                worse = "; ".join(w.split(":")[0] + (" (J++)" if "J++" in w else " (naked)") for w in ev["worse_in"]) or "-"
                ref_lens = lens in D.reference
                verdict = ((D.labels.get(lens, "reference") + (": passes" if ev["passes"] else ": fails"))
                           if ref_lens else
                           "SERVE" if ev.get("served") else ("pass, not served" if ev["passes"] else "do not serve"))
                p_cell = f"{ev['p_sign_holm']:.2g}" + (" (raw)" if ref_lens else "")
                sj, sn = ev["sign_jpp"], ev["sign_naive"]
                where = ev.get("stuck_max_where")
                stw = f"{ev['stuck_max']:.2f} ({where[0]}, {where[1]}, {where[2]!r})" if where else "n/a"
                L.append(f"| {lens} | {band} | {fmt_num(ev['second']['pooled']['mean'])} | "
                         f"{fmt_num(ev['jpp']['pooled']['mean'])} | {fmt_num(ev['naive']['pooled']['mean'])} | "
                         f"{fmt_ci(ev['diff_jpp']['pooled'])} | {fmt_ci(ev['diff_naive']['pooled'])} | "
                         f"{sj['n_pos']}/{sj['n_neg']} | {sn['n_pos']}/{sn['n_neg']} | {p_cell} | "
                         f"{worse} | {stw} | {verdict} |")
    L.append("")
    fails = [(lens, band, ev) for lens, d in an["bands"].items() for band, ev in d.items() if not ev["passes"]]
    if fails:
        L.append("Why each failing band fails:")
        for lens, band, ev in fails:
            L.append(f"- {lens} {band}: " + "; ".join(ev["reasons"]))
        L.append("")
    rp = an.get("replication_r2_prompts")
    if rp is not None:
        ro = rp["readouts_against_r2_02"]
        L.append(f"Replication on R2's prompts (R3_PREREG.md §8.6, decides nothing): "
                 f"{'passes' if rp['ok'] else 'FAILS'}. {ro['found_again']} of {ro['r2_positions']} of R2's read "
                 f"positions found again; filtered top-1 the same as R2's: " +
                 ", ".join(f"{k} {v['filtered_share']}" for k, v in ro["readouts"].items()) +
                 f" (tolerance {ro['tolerance']['filtered_top1']}); R2's design gives R2_03's verdicts: "
                 f"{'yes' if rp['r2_design_against_r2_03']['ok'] else 'no'}.")
        L.append("")
    if an.get("readings"):
        L += secondary_md(an["readings"])
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


def selftest_h5(check: Callable[[bool, str], None]) -> None:
    """H5 on R3's synthetic file, whose jpp-it - rl-it is known band by band (synthetic_r3's docstring)."""
    data, _ = synthetic_r3()
    h5 = h5_tables(R3_DESIGN, data, {"macros": {"primary": {"evals": ["typo"], "n_items": 3, "lenses": {
        "jpp-it": {"30": {"macro": 0.5}}, "rl-it": {"30": {"macro": 0.25}}, "jpp": {"30": {"macro": 0.1}}}}}})
    want = {"30-39": "helps", "40-49": "hurts", "50": "not shown", "51-54": "helps", "55": "helps", "56-57": "helps",
            "58": "not shown"}
    got = {b: v["reading"] for b, v in h5["bands"].items()}
    check(got == want and not h5["refuted"] and h5["helps_at"] == ["30-39", "51-54", "55", "56-57"]
          and h5["hurts_at"] == ["40-49"], f"H5 on the synthetic file: every band read, as built ({got})")
    p = h5["bands"]
    # 316 valid prompts (4 bare prompts have a collapse target, all with r = 8): the sums below count them by hand
    check(abs(p["30-39"]["pooled"]["mean_raw"] - 160 / 316) < 1e-12 and abs(p["56-57"]["pooled"]["mean_raw"] - 48 / 316)
          < 1e-12 and abs(p["51-54"]["pooled"]["mean_raw"] - 12 / 316) < 1e-12 and p["50"]["pooled"]["mean_raw"] == 0.0
          and abs(p["40-49"]["pooled"]["mean_raw"] + 44 / 316) < 1e-12
          and abs(p["40-49"]["by_cond"]["chat-begun"]["mean_raw"] + 0.8) < 1e-12
          and abs(p["40-49"]["by_cond"]["chat-no-answer"]["mean_raw"] - 0.1) < 1e-12
          and p["30-39"]["pooled"]["n"] == 316,
          "H5's point estimates are the built differences over the 316 valid prompts (30-39 160/316, 56-57 48/316, "
          "51-54 12/316, 50 0, 40-49 -44/316; 40-49 chat-begun -0.8, chat-no-answer +0.1)")
    # the same resamples as the rule: the 95% interval equals summarize()'s, the Bonferroni one is recomputed by hand
    t = Table(R3_DESIGN, data)
    boot = Boot(t, R3_DESIGN.n_boot, R3_DESIGN.boot_seed)
    same, wider = True, True
    for band, layers in R3_DESIGN.bands:
        L = list(layers)
        s = summarize(t, boot, PRIMARY, (H5_A, L), (H5_B, L))["pooled"]
        q = p[band]["pooled"]
        sums, ns = np.zeros(boot.B), 0
        for c in R3_DESIGN.conditions:
            ok, ix = boot.idx(c, 0)
            x = t.prompt_scores(PRIMARY, H5_A, L, c)[ok] - t.prompt_scores(PRIMARY, H5_B, L, c)[ok]
            sums += x[ix].sum(1)
            ns += len(ok)
        bl, bh = np.percentile(sums / ns, [0.357, 99.643])
        same &= (q["mean_raw"] == s["mean_raw"] and q["ci95_raw"] == [s["lo_raw"], s["hi_raw"]]
                 and q["bonferroni_raw"] == [float(bl), float(bh)])
        wider &= q["bonferroni_raw"][0] <= q["ci95_raw"][0] and q["bonferroni_raw"][1] >= q["ci95_raw"][1]
    check(same and wider, "H5's intervals come from the rule's own 2000 paired resamples: the 95% one equals the "
                          "rule's summarize(), the Bonferroni one is the 0.357th/99.643th percentiles of the same "
                          "sums and contains the 95% one")
    # a difference the 95% interval shows and the Bonferroni one does not: "not shown"
    d2 = json.loads(json.dumps(data))
    for q in d2["positions"]:
        if q["cond"] == "chat-no-answer" and q["pid"] in ("cna-04", "cna-05", "cna-07", "cna-08"):
            r = q["r"]["jpp-it"]["58"]
            r["f"][0], r["fs"][0] = q["ct_id"], q["ct_str"]
    b58 = h5_tables(R3_DESIGN, d2)["bands"]["58"]
    check(b58["pooled"]["ci95_raw"][0] > 0 and b58["pooled"]["bonferroni_raw"][0] <= 0
          and b58["reading"] == "not shown",
          f"H5 reads the Bonferroni interval, not the 95% one: 4 extra hits in one condition at 58 give a 95% lower "
          f"end {b58['pooled']['ci95_raw'][0]:.4f} > 0 but a Bonferroni one {b58['pooled']['bonferroni_raw'][0]:.4f} "
          f"-> 'not shown'")
    # rl-it equal to jpp-it everywhere: no band helps -> H5 refuted; every band still read
    d3 = json.loads(json.dumps(data))
    for q in d3["positions"]:
        q["r"]["rl-it"] = json.loads(json.dumps(q["r"]["jpp-it"]))
    h3_ = h5_tables(R3_DESIGN, d3)
    check(h3_["refuted"] and len(h3_["bands"]) == 7 and all(v["reading"] == "not shown" for v in h3_["bands"].values()),
          "H5 refuted when no band reads 'helps'; all seven bands are still read")
    d4 = json.loads(json.dumps(data))
    for q in d4["positions"]:
        q["r"]["rl-it"].pop("52", None)
    h4_ = h5_tables(R3_DESIGN, d4)
    check(h4_["bands"]["51-54"]["reading"] == "not shown (a readout is missing)" and len(h4_["bands"]) == 7
          and h4_["bands"]["30-39"]["reading"] == "helps",
          "a band with a readout missing stays in H5's table and reads 'not shown'")
    rc = h5["recall_at_10"]["primary"]["lenses"]
    check(set(rc) == {"jpp-it", "rl-it"} and rc["jpp-it"]["30"]["macro"] == 0.5 and "favours jpp-it" in h5["recall_note"],
          "J++'s recall@10 is reported beside H5 for both lenses, with the note that it decides nothing")
    md = "\n".join(secondary_md({"H5": h5, "H2": {"not_run": "x"}, "H3": {"not_run": "y"}}))
    check(f"| 40-49 | {-44 / 316:+.3f} |" in md and "hurts" in md and "H2 not run" in md,
          "H5's table renders in the report; a reading not run is said")


def _h2_synth(seed: int = 3):
    """Synthetic pullbacks with known q10 cosines against J++ (K 128, 59 layers, D 40, 4 quiet dims per layer)."""
    import torch

    g = torch.Generator().manual_seed(seed)
    K, D, NL = 128, 40, 59
    q10 = torch.zeros(60, D)
    for l in range(60):
        q10[l, torch.randperm(D, generator=g)[:4]] = 1.0
    masks = {"q10": q10, "loud": 1 - q10, "all": torch.ones(60, D)}
    refs = {k: torch.randn(K, NL, D, generator=g, dtype=torch.float64) for k in H2_REFS}
    layers = list(C.FIT_LAYERS)

    def offsets():
        d = torch.zeros(len(layers), dtype=torch.float64)
        for _, a, b in H2_BANDS:
            n = b - a + 1
            for i in range(n - n % 2):
                d[layers.index(a + i)] = 0.03 if i % 2 == 0 else -0.03
        return d

    def build(base: Dict[str, float]) -> Tuple[Any, Dict[str, float]]:
        u = (torch.rand(K, generator=g, dtype=torch.float64) - 0.5) * 0.04
        c = torch.empty(K, len(layers), dtype=torch.float64)
        for name, a, b in H2_BANDS:
            for l in range(a, b + 1):
                c[:, layers.index(l)] = base[name] + u
        c += offsets()[None]
        c[:30] = -0.9                                     # the 30 named directions: never in the token-pair mean
        J = refs["jpp"][:, layers]
        m = q10[layers][None].double()
        a = J * m
        a = a / a.norm(dim=-1, keepdim=True)
        r = torch.randn(K, len(layers), D, generator=g, dtype=torch.float64) * m
        r = r - (r * a).sum(-1, keepdim=True) * a
        r = r / r.norm(dim=-1, keepdim=True)
        x = c[..., None] * a + (1 - c[..., None] ** 2).sqrt() * r - J * (1 - m)     # loud: exactly -J++
        want = {name: base[name] + float(u[30:].mean()) for name, _, _ in H2_BANDS}
        return x.float(), want
    return refs, masks, build


def selftest_h2(check: Callable[[bool, str], None]) -> None:
    import tempfile

    import torch

    q0j = json.loads(C.Q0_JSON.read_text(encoding="utf-8"))
    yard = h2_yardsticks(q0j)
    check(all(round(yard[w][b], 2) == H2_REGISTERED_2DP[w][b] for w in ("low", "high") for b, _, _ in H2_BANDS)
          and abs(yard["high"]["L30-39"] - 0.7666) < 1e-9 and abs(yard["low"]["L50-58"] - 0.2868) < 1e-9,
          f"H2's yardsticks are Q0's rows R1 IT vs J++ ({list(yard['low'].values())}) and J++ vs JLens "
          f"({list(yard['high'].values())}), which round to §1.2's 0.27/0.29/0.29 and 0.77/0.68/0.66")
    lo_, hi_ = 0.2883, 0.7666
    edge = [h2_reading_one("jpp-it", {"ci95_raw": [0.1, lo_]}, lo_, hi_),
            h2_reading_one("jpp-it", {"ci95_raw": [hi_, 0.9]}, lo_, hi_),
            h2_reading_one("rl-it", {"ci95_raw": [0.1, lo_ + 1e-9]}, lo_, hi_),
            h2_reading_one("rl-it", {"ci95_raw": [hi_ - 1e-9, 0.9]}, lo_, hi_)]
    check([e["reading"] for e in edge] == ["'the recipe sets the lens' refuted",
                                           "'the checkpoint, with the recipe version and the items, dominates' refuted",
                                           "both", "both"],
          "H2's quantifiers: an upper end at the low yardstick refutes 'the recipe sets the lens', a lower end at the "
          "high one refutes 'the checkpoint ... dominates' (at or below / at or above); just inside reads 'both'")
    refs, masks, build = _h2_synth()
    X, want_x = build({"L30-39": 0.9, "L40-49": 0.1, "L50-58": 0.5})
    Y, want_y = build({"L30-39": 0.5, "L40-49": 0.5, "L50-58": 0.9})
    pb = {"jpp-it": X, "rl-it": Y, "jpp-it-A": X, "jpp-it-B": X.clone(),
          "rl-it-A": torch.randn_like(Y), "rl-it-B": torch.randn_like(Y)}
    syard = {"low": {"L30-39": 0.2679, "L40-49": 0.2883, "L50-58": 0.2868},
             "high": {"L30-39": 0.7666, "L40-49": 0.6846, "L50-58": 0.6574}}
    h2 = h2_tables(pb, {k: v.float() for k, v in refs.items()}, masks, C.FIT_LAYERS, syard)
    gx = {b: h2["vs"]["jpp-it"]["jpp"]["q10"][b]["tokpair98_raw"] for b in want_x}
    gy = {b: h2["vs"]["rl-it"]["jpp"]["q10"][b]["tokpair98_raw"] for b in want_y}
    check(all(abs(gx[b] - want_x[b]) < 2e-5 and abs(gy[b] - want_y[b]) < 2e-5 for b in want_x),
          f"H2's q10 cosines are the built ones: band means over layers, then the 98 token pairs only (jpp-it {gx}, "
          f"built {want_x})")
    check(all(abs(h2["vs"]["jpp-it"]["jpp"]["loud"][b]["tokpair98_raw"] + 1) < 1e-5 for b in want_x)
          and all(h2["vs"]["jpp-it"]["jpp"]["q10"][b]["ci95_raw"][0] < gx[b] < h2["vs"]["jpp-it"]["jpp"]["q10"][b]["ci95_raw"][1]
                  for b in want_x),
          "the loud cosine (built at -1) is apart from the q10 one; each interval holds its mean")
    rd = {lens: {b: r["reading"] for b, r in d.items()} for lens, d in h2["reading"].items()}
    ck = "'the checkpoint, with the recipe version and the items, dominates' refuted"
    check(rd == {"jpp-it": {"L30-39": ck, "L40-49": "'the recipe sets the lens' refuted", "L50-58": "both"},
                 "rl-it": {"L30-39": "both", "L40-49": "both",
                           "L50-58": "'the checkpoint, with the recipe version, dominates' refuted"}},
          f"H2's reading for jpp-it and rl-it from the built cosines: {rd}")
    sh = h2["split_half"]
    check(all(abs(sh["jpp-it"]["q10"][b]["tokpair98_raw"] - 1) < 1e-5 for b in want_x)
          and all(abs(sh["rl-it"]["q10"][b]["tokpair98_raw"]) < 0.1 for b in want_x)
          and set(h2["vs"]) == set(H2_LENSES) and h2["vs"]["jpp-it-full-macro"] == {"missing": True},
          "the split-half ceilings (A against B) are read; a lens without pullbacks is marked missing")
    # end to end on the real inputs: jpp-it's pullbacks := Q0's J++ ones, rl-it's := Q0's Miru ones, so that H2 must
    # give Q0's own rows (q10 point values, 4 decimals) through the real loader, masks and layer offsets
    q0 = torch.load(C.Q0_PT, map_location="cpu", weights_only=False)
    L = list(C.FIT_LAYERS)
    fake = {"names": q0["names"], "layers": L,
            "pb": {"jpp-it": q0["pb"]["jpp"][:, L].clone(), "rl-it": q0["pb"]["miru"][:, L].clone(),
                   "jpp-it-A": q0["pb"]["jpp"][:, L].clone(), "jpp-it-B": q0["pb"]["jlens"][:, L].clone()}}
    del q0
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td) / "pb.pt"
        torch.save(fake, tp)
        del fake
        rdg = secondary_readings(R3_DESIGN, synthetic_r3()[0], pb_path=tp, weights_json=Path(td) / "none.json",
                                 recall_json=Path(td) / "none.json", full_hash=False)
    e2 = rdg["H2"]
    qc = q0j["comparisons"]
    got = {b: (e2["vs"]["rl-it"]["jpp"]["q10"][b]["tokpair98"], e2["vs"]["jpp-it"]["jlens"]["q10"][b]["tokpair98"],
               e2["vs"]["jpp-it"]["r1_it"]["q10"][b]["tokpair98"], e2["split_half"]["jpp-it"]["q10"][b]["tokpair98"])
           for b, _, _ in H2_BANDS}
    want = {b: (qc["fp32: miru vs jpp"]["q10"]["bands"][b]["tokpair98"], qc[H2_Q0_HIGH]["q10"]["bands"][b]["tokpair98"],
                qc[H2_Q0_LOW]["q10"]["bands"][b]["tokpair98"], qc[H2_Q0_HIGH]["q10"]["bands"][b]["tokpair98"])
            for b, _, _ in H2_BANDS}
    ok = all(abs(x - y) <= 1e-4 for b in got for x, y in zip(got[b], want[b]))
    ok &= all(abs(e2["vs"]["jpp-it"]["jpp"]["q10"][b]["tokpair98_raw"] - 1) < 1e-5 for b, _, _ in H2_BANDS)
    ok &= all(e2["reading"]["jpp-it"][b]["checkpoint_dominates_refuted"] for b, _, _ in H2_BANDS)
    check(ok, f"H2 end to end through the real loader: pullbacks set to Q0's J++ / Miru / JLens give Q0's own q10 rows "
              f"(Miru vs J++, J++ vs JLens, R1 IT vs J++) to 1e-4 ({got})")
    check("not_run" in rdg["H3"] and rdg["H5"]["recall_at_10"] is None and "recall_missing" in rdg["H5"],
          "a missing lens file or recall file is said, and that reading is not run")


def _orth(n: int, g) -> Any:
    import torch

    q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    return q


def selftest_h3(check: Callable[[bool, str], None]) -> None:
    import torch

    import R3_Q0_pullbacks as Q0

    g = torch.Generator().manual_seed(5)
    D, k = 300, H3_K
    fnw = torch.exp((torch.rand(D, generator=g, dtype=torch.float64) - 0.5) * 3)
    Uo, Vi = _orth(D, g), _orth(D, g)
    s = torch.cat([3 + torch.rand(k, generator=g, dtype=torch.float64), 0.5 * torch.ones(D - k, dtype=torch.float64)])
    Ly = Uo @ torch.diag(s) @ Vi.T                                   # a lens in y with a known spectrum
    known = float((s[:k] ** 2).sum() / (s ** 2).sum())
    J = Ly / fnw[:, None]                                            # the file's matrix (h @ J.T): y = diag(fnw) J
    # base J++ in y: its top-k output subspace is Uj[:, :k]; its input side is unrelated
    Uj, Vj = _orth(D, g), _orth(D, g)
    sj = torch.cat([5 + torch.rand(k, generator=g, dtype=torch.float64), 0.2 * torch.ones(D - k, dtype=torch.float64)])
    Jpp = (Uj @ torch.diag(sj) @ Vj.T) / fnw[:, None]

    def with_share_in(f: float):
        P = Uj[:, :k] @ torch.randn(k, D, generator=g, dtype=torch.float64)
        Q = Uj[:, k:] @ torch.randn(D - k, D, generator=g, dtype=torch.float64)
        return (f ** 0.5 * P / P.norm() + (1 - f) ** 0.5 * Q / Q.norm()) / fnw[:, None]
    sig = 0.1
    NA = torch.randn(D, D, generator=g, dtype=torch.float64) * sig
    NB = torch.randn(D, D, generator=g, dtype=torch.float64) * sig
    A, B = J + NA / fnw[:, None], J + NB / fnw[:, None]
    mats = {"jpp-it": (A + B) / 2, "jpp-it-A": A, "jpp-it-B": B, "rl-it": J, "rl-it-A": J, "rl-it-B": J.clone(),
            "jpp": Jpp, "jlens": with_share_in(0.3), "miru": with_share_in(0.7)}
    r = h3_layer(mats, fnw.float())
    check(abs(r["rl-it"]["share_y"] - known) < 1e-9 and abs(r["jpp"]["share_y"] - float((sj[:k] ** 2).sum()
                                                                                       / (sj ** 2).sum())) < 1e-9,
          f"H3's share in y is the built one ({r['rl-it']['share_y']:.6f} = {known:.6f})")
    wrong_side, _ = _gram_top(J * fnw[None, :], k, False)
    check(abs(r["rl-it"]["share_raw"] - known) > 0.01 and abs(wrong_side - known) > 0.01,
          f"the coordinates matter: raw {r['rl-it']['share_raw']:.3f} and fnw on the input side {wrong_side:.3f} both "
          f"differ from the y share {known:.3f}, so a wrong side or no fnw would fail the check above")
    check(abs(r["jlens"]["in_jpp_top_y"] - 0.3) < 1e-9 and abs(r["miru"]["in_jpp_top_y"] - 0.7) < 1e-9
          and abs(r["jpp"]["in_jpp_top_y"] - r["jpp"]["share_y"]) < 1e-9,
          "the share inside base J++'s top-100 output subspace is the built one (0.3, 0.7; J++'s own = its share)")
    check(abs(r["rl-it"]["share_y_debiased"] - r["rl-it"]["share_y"]) < 1e-9,
          "with identical halves the debiased share equals the plain one")
    Y = {n: fnw[:, None] * m for n, m in mats.items()}
    _, U = _gram_top(Y["jpp-it"], k, True)
    brute = sum(float(U[:, i] @ (Y["jpp-it-A"] @ Y["jpp-it-B"].T) @ U[:, i]) for i in range(k)) / float(
        torch.trace(Y["jpp-it-A"] @ Y["jpp-it-B"].T))
    oracle = float((U.T @ Ly).pow(2).sum() / Ly.pow(2).sum())
    d = r["jpp-it"]
    check(abs(d["share_y_debiased"] - brute) < 1e-9,
          "the debiased share is sum_i u_i^T (A~ B~^T) u_i / <A~, B~>_F, checked term by term")
    # noisy halves. With U fixed apart from the noise (the signal's own top directions) the cross product is unbiased:
    # it recovers the signal's share. With U taken from the whole lens, as §8.2 registers, U leans towards the
    # directions where the halves' noise agrees (N_A N_B^T = (S S^T - Dm Dm^T) / 4 with S = N_A + N_B chosen by U),
    # so the debiased share is not pulled down by noise but can overshoot the signal's share, here above 1
    U_sig, _ = torch.linalg.qr(Uo[:, :k])
    fixed = _debiased_share(U_sig, Y["jpp-it-A"], Y["jpp-it-B"])
    check(abs(fixed - known) < 0.02 and d["share_y"] < known - 0.05 and d["share_y_debiased"] > d["share_y"]
          and d["share_y_debiased"] > oracle,
          f"noisy halves: with U the signal's own directions the cross product gives {fixed:.3f} (the signal's share "
          f"{known:.3f}); with U from the whole lens, as registered, the plain share {d['share_y']:.3f} is pulled down "
          f"and the debiased one {d['share_y_debiased']:.3f} overshoots the signal's share in those directions "
          f"({oracle:.3f}): the registered estimator is biased upward by U's selection (reported, not changed)")
    V = [h3_verdict(0.9, 0.95, 0.8), h3_verdict(0.7, 0.75, 0.8), h3_verdict(0.7, 0.8, 0.8), h3_verdict(0.7, 0.85, 0.8),
         h3_verdict(0.8, 0.9, 0.8), h3_verdict(0.85, 0.75, 0.8)]
    check([v["verdict"].split(" (")[0] for v in V] == ["supported", "refuted", "refuted", "not separated",
                                                       "not separated", "supported and refuted"],
          "H3's quantifiers: plain above Miru supports; debiased at or below Miru refutes; otherwise not separated; "
          "both at once is reported unordered")

    # end to end over layers: judged layers read, the others reported only
    def read(n, l):
        m = dict(mats)
        if l == 35:
            m["miru"] = Uo @ torch.diag(torch.cat([s[:k] * 10, s[k:]])) @ Vi.T / fnw[:, None]   # Miru more concentrated
        if l == 35:                                                  # jpp-it's halves nearly noise-free here
            gl = torch.Generator().manual_seed(8)
            a = J + 0.005 * torch.randn(D, D, generator=gl, dtype=torch.float64) / fnw[:, None]
            b = J + 0.005 * torch.randn(D, D, generator=gl, dtype=torch.float64) / fnw[:, None]
            m["jpp-it"], m["jpp-it-A"], m["jpp-it-B"] = (a + b) / 2, a, b
        if l == 30:                                                  # Miru unconcentrated; rl-it equal to it
            m["miru"] = torch.randn(D, D, generator=torch.Generator().manual_seed(7), dtype=torch.float64) / fnw[:, None]
            m["rl-it"], m["rl-it-A"], m["rl-it-B"] = m["miru"], m["miru"], m["miru"]
        return m.get(n)
    t3 = h3_tables(read, fnw.float(), (30, 31, 35), judged=(30, 35))
    rv = {lens: {l: v["verdict"] for l, v in d.items()} for lens, d in t3["reading"].items()}
    check(rv == {"jpp-it": {"30": "supported", "35": "refuted"}, "rl-it": {"30": "refuted", "35": "refuted"}}
          and set(t3["per_layer"]) == {"30", "31", "35"},
          f"H3 end to end: verdicts at the judged layers only, every layer reported ({rv})")
    check(H3_JUDGED == (30, 35, 40, 45, 50, 55, 58) and H3_K == 100 and set(H3_LENSES) == {"jpp-it", "rl-it", "jpp",
                                                                                          "jlens", "miru"},
          "H3's registered constants: R1 §3.2's seven layers, top 100, the five lenses of §8.2")
    # the coordinates on the real files: base J++ and JLens through this code give R1 §3.2's figures (R1_15)
    r1 = json.loads(R1_SPECTRUM_JSON.read_text(encoding="utf-8"))["full"]
    for n, p, sha, size in (("jpp", C.JPP_BASE_PATH, C.JPP_BASE_SHA256, C.JPP_BASE_SIZE),
                            ("jlens", C.JLENS_PATH, C.JLENS_SHA256, C.JLENS_SIZE)):
        if not C.check_pinned_file(p, sha, size, full=False)["size_ok"]:
            check(False, f"{p} is not the pinned file")
            return
    fnw_r = torch.load(C.QUIET_STATS_PATH, weights_only=False, map_location="cpu")["fnw"].float()
    rj, rl = Q0.lens_reader("jpp"), Q0.lens_reader("jlens")
    worst, rows = 0.0, {}
    for l in H3_JUDGED:
        x = h3_layer({"jpp": rj(l).float(), "jlens": rl(l).float()}, fnw_r)
        w = r1[f"L{l}"]
        got = (x["jpp"]["share_y"], x["jpp"]["share_raw"], x["jlens"]["share_y"])
        want = (w["jpp_self_topk_y"]["100"], w["jpp_self_topk_raw"]["100"], w["jlens_self_topk_y"]["100"])
        rows[l] = [round(v, 4) for v in got]
        worst = max(worst, max(abs(a - b) for a, b in zip(got, want)))
    check(worst <= H3_R1_TOL, f"H3's code on the real files reproduces R1 §3.2 (R1_15_spectrum.json): base J++ in y "
                              f"and raw, JLens in y, top 100, at the seven layers (max |diff| {worst:.4f} <= "
                              f"{H3_R1_TOL}; {rows})")


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

    # a candidate missing one layer of a band: the band stays in the family (rule 2 fails, p = 1); Holm over 14
    dm = synthetic_r3()[0]
    for p in dm["positions"]:
        p["r"]["rl-it"].pop("52", None)
    anm = analyze(R3_DESIGN, dm)
    evm = anm["bands"]["rl-it"]["51-54"]
    check(not check_family(R3_DESIGN, anm) and evm["holm_m"] == 14 and not evm["rules"]["2"] and evm["p_sign"] == 1.0
          and evm["missing_layers"] == {"rl-it": [52]} and not evm["passes"],
          "a readout missing at one layer of a band fails rule 2 and keeps the band in the Holm family (14 tests)")
    bad = check_family(R3_DESIGN, dict(anm, constants=dict(anm["constants"], HOLM_M=13),
                                       bands={"jpp-it": anm["bands"]["jpp-it"]}))
    check(len(bad) == 8, f"a family of fewer than 14 tests is refused before anything is written ({len(bad)} reasons)")
    check(check_input({"prompts": [{"cond": c} for c, n in C.N_PROMPTS.items() for _ in range(n)],
                       "meta": {"prompts_sha256": "x"}}, C.N_PROMPTS, "x", "t") == []
          and len(check_input({"prompts": [], "meta": {}}, C.N_PROMPTS, "x", "t")) == 2,
          "the input check: the registered prompt counts and the prompts' sha256")
    # the replication's readout comparison: R2's own reading passes against itself; a perturbed copy fails
    r2e = json.loads(R2.EVAL_JSON.read_bytes())
    ro = replication_readouts(r2e, r2e)
    pert = json.loads(json.dumps(r2e))
    nflip = 0
    for p in pert["positions"][::20]:
        for l, rr in p["r"]["miru"].items():
            rr["f"] = [rr["f"][1]] + rr["f"][1:]
            nflip += 1
    rp2 = replication_readouts(pert, r2e)
    check(ro["ok"] and ro["found_share"] == 1.0 and all(v["filtered_share"] == 1.0 for v in ro["readouts"].values())
          and not rp2["ok"] and not rp2["readouts"]["miru"]["ok"] and rp2["readouts"]["jpp"]["ok"]
          and rp2["n_mismatches"] == nflip,
          f"the replication's readout check: R2's own file against itself passes (every position found, every top-1 "
          f"the same); {nflip} changed Miru top-1s fail Miru's tolerance and are counted")
    r2a = json.loads(R2.ANALYSIS_JSON.read_text(encoding="utf-8"))
    rv = replication_verdicts(r2a, r2a)
    flip = json.loads(json.dumps(r2a))
    flip["bands"]["r1"]["58"]["rules"]["5"] = True
    flip["bands"]["r1"]["58"]["passes"] = True
    check(rv["ok"] and not replication_verdicts(flip, r2a)["ok"],
          "the replication's verdict check: R2_03's verdicts against themselves agree; one changed verdict is caught")
    md_f = report_md(R3_DESIGN, dict(an, replication_r2_prompts={"ok": False, "readouts_against_r2_02": rp2,
                                                                 "r2_design_against_r2_03": rv}), {"selftest": True})
    check("did not reproduce R2's readings" in md_f and "FAILS" in md_f,
          "a failed replication is said first in the report")

    an2 = analyze(R3_DESIGN, synthetic_r3()[0])
    check(_norm(an2["bands"]) == _norm(analyze(R3_DESIGN, synthetic_r3()[0])["bands"]), "the bootstrap is deterministic")
    md = report_md(R3_DESIGN, an, {"selftest": True})
    check("SELF-TEST" in md and "| jpp-it | 30-39 |" in md and "S5" in md and "(raw)" in md
          and "reference: passes" in md, "the report renders; reference rows give a raw p, outside Holm")
    selftest_h5(check)
    selftest_h2(check)
    selftest_h3(check)
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
    ap.add_argument("--pullbacks", default=str(PULLBACKS_PT), help="H2's input: R3_05's pullbacks")
    ap.add_argument("--recall", default=str(RECALL_JSON), help="H5's secondary input: R3_05's J++ recall@10")
    ap.add_argument("--weights", default=str(C.WEIGHTS_JSON), help="H3's input: R3_04_weights.json (its lens files)")
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
    if not meta.get("selftest"):
        chk = json.loads(C.CHECK_JSON.read_text(encoding="utf-8")) if C.CHECK_JSON.is_file() else {}
        bad = check_input(data, C.N_PROMPTS, C.sha256_file(C.PROMPTS_PATH), "R3_05_eval.json")
        if not chk.get("ok") or chk.get("prompts_sha256") != C.sha256_file(C.PROMPTS_PATH):
            bad.append("R3_01_check.json does not pass this R3_prompts.json")
        if bad:
            raise SystemExit("the input is not R3's registered reading: " + "; ".join(bad))
    an = analyze(R3_DESIGN, data)
    bad = check_family(R3_DESIGN, an)
    if bad:
        raise SystemExit("the Holm family is not the registered one: " + "; ".join(bad))
    if args.replication:
        rep = json.loads(Path(args.replication).read_bytes())
        if not (rep.get("meta") or {}).get("selftest"):
            bad = check_input(rep, R2.N_PROMPTS, R2.sha256_file(R2.PROMPTS_PATH), "the replication file")
            if bad:
                raise SystemExit("; ".join(bad))
        an["replication_r2_prompts"] = replication(rep, json.loads(R2.EVAL_JSON.read_bytes()),
                                                   json.loads(R2.ANALYSIS_JSON.read_text(encoding="utf-8")))
    an["readings"] = secondary_readings(R3_DESIGN, data, pb_path=Path(args.pullbacks), weights_json=Path(args.weights),
                                        recall_json=Path(args.recall))
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

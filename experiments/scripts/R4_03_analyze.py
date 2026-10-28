#!/usr/bin/env python3
"""R4 03 (CPU): R2's decision rule, applied by R3's code to R1's lens at code 55 and 58 (R4_PREREG.md §5).

R3_06_analyze.analyze is imported unchanged and given R4's Design: the readouts J++, the naked eye and R1; R1 the only
candidate; bands "55" (code 55) and "58" (code 58); R1 takes precedence (decide() serves only lenses named in
precedence); never below code 55; Holm over both bands always (require_all_bands); bootstrap seed 20261029; S2 drops
R4_common.FIT_OVERLAP. R1 passes at a band when all five of R2's rules hold:
  1. the 95% prompt-bootstrap intervals of the pooled R1 - J++ and R1 - naked eye lie above 0;
  2. in no decision condition is either per-condition interval entirely below 0 (no data counts as a failure);
  3. the stuck-word share is at most 0.25 in every decision condition;
  4. the exact one-sided sign test against both baselines (the larger p), Holm-adjusted over the 2 tests, is below
     0.025;
  5. the pooled point estimates of R1 - J++ and R1 - naked eye are each at least 0.05.
R2's filter counts function words ("the", "of", "is") as content. So R4 serves a band only if R1 also passes all five
rules again under S6, with every function-word target (R2_common.FUNCTION_WORDS) made no target, Holm over the 2 bands
within S6. This was the owner's choice of 28 October 2026 (R4_PREREG.md §2, §5): an["r4_decision"]. R3_06's own
an["decision"] is R2's rule alone, and is kept. A served band is a recommendation to a later serving track; R4 serves
nothing itself. Sensitivity analyses S1-S4 rerun the rule; S5 (J++'s filter) is not run, as the run records no
J++-filtered readout. Reported only: R1 - naked eye at 55 and 58 pooled over R2's, R3's and R4's prompts.

Refusals (R4_PREREG.md §9), before anything is written: an input whose schema is not R4_02's; self-test data written
into experiments/results; an input whose prompt counts or prompts sha256 are not the registered ones, or whose
prompts R4_01_check.json does not pass; an input that is not the output of R4_02's last finished run (R4_02_runs.jsonl
ends with 'finished' and the input's sha256); a replication record that is missing or failed; a Holm family other
than the registered 2 tests.

Modes:
  default         R4_02_eval.json -> R4_03_analysis.json and R4_03_report_section.md
  --reproduce-r3  R3_05_eval.json restricted to R4's readouts, under R4's design with R3's bootstrap seed (20261022):
                  R1's rules, verdicts, means, intervals, sign counts and raw sign-test p at 55 and 58 must equal
                  R3_06_analysis.json's reference_bands (R3 took R1's p outside a Holm family, so the Holm-adjusted p
                  is not compared, and R3 ran no sensitivity analysis for R1); writes R4_03_r3_reproduction.json
  --selftest      synthetic files with known answers (each rule failing alone at a band, Holm over two,
                  require_all_bands, S5 not run, the refusals); writes nothing to experiments/results
Run: CUDA_VISIBLE_DEVICES="" /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R4_03_analyze.py
       [--reproduce-r3 | --selftest [--record] | --in X --out Y --report Z]
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
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import R2_common as R2  # noqa: E402
import R3_common as C3  # noqa: E402
import R3_06_analyze as A  # noqa: E402  (R3's rule, unchanged)
import R4_common as C  # noqa: E402
from R2_03_analyze import holm  # noqa: E402

R4_DESIGN = A.Design(**C.DESIGN_FIELDS, overlap=dict(C.FIT_OVERLAP))
PAGE = {"55": 56, "58": 59}


# ------------------------------------------------------------------------------------------------ the input's checks
def check_run_finished(inp_sha: str, runs_log: Path = C.RUNS_LOG) -> List[str]:
    """§9: a run that stops part-way is rerun from the start and its partial data is not analysed. R4_02 logs
    'finished' with its output's sha256 only after writing it, so the last event must be that."""
    bad: List[str] = []
    events: List[Dict[str, Any]] = []
    if runs_log.is_file():
        for i, line in enumerate(runs_log.read_text(encoding="utf-8").splitlines()):
            if line.strip():
                try:
                    events.append(json.loads(line))
                except ValueError:
                    bad.append(f"{runs_log.name} line {i + 1} is not JSON")
    if not events:
        bad.append(f"{runs_log.name} records no run of R4_02")
    elif events[-1].get("event") != "finished":
        bad.append(f"the last run of R4_02 did not finish (its last event is {events[-1].get('event')!r}); §9: it is "
                   f"rerun from the start, its partial data not analysed")
    elif events[-1].get("out_sha256") != inp_sha:
        bad.append(f"the input (sha256 {inp_sha[:12]}) is not the output of R4_02's last finished run")
    return bad


def check_replication(path: Path = C.REPLICATION_JSON) -> List[str]:
    if not path.is_file():
        return [f"{path.name} does not exist: R4 is not registered without its replication record (§4.2)"]
    rec = json.loads(path.read_text(encoding="utf-8"))
    return [] if rec.get("ok") else [f"{path.name} records a failed replication (§4.2)"]


def check_prompts_checked(prompts_sha: str, check_json: Path = C.CHECK_JSON) -> List[str]:
    chk = json.loads(check_json.read_text(encoding="utf-8")) if check_json.is_file() else {}
    if not chk.get("ok") or chk.get("prompts_sha256") != prompts_sha:
        return ["R4_01_check.json does not pass this R4_prompts.json"]
    return []


def refuse_selftest_into_results(meta: Dict[str, Any], *paths: Path) -> None:
    if meta.get("selftest") and any(Path(p).resolve().parent == C.RESULTS.resolve() for p in paths):
        raise SystemExit("this input is self-test data; write the analysis and report outside experiments/results")


def check_code(meta: Dict[str, Any], replication_json: Path = C.REPLICATION_JSON) -> List[str]:
    """The rule is decided by the code that was registered and run: R2's and R3's rule files have their pinned sha256;
    R4_common.py (the design and FIT_OVERLAP) is the one the run used; the replication record is the one the run
    checked."""
    bad = [f"{r} differs from its pinned sha256" for r in C.pinned_differ(C.RULE_CODE)]
    run_files = meta.get("files_sha256") or {}
    if run_files.get("R4_common.py") != C.sha256_file(C.SCRIPTS / "R4_common.py"):
        bad.append("R4_common.py (the design and FIT_OVERLAP) is not the one R4_02's run used")
    if not replication_json.is_file() or meta.get("replication_sha256") != C.sha256_file(replication_json):
        bad.append("the replication record is not the one R4_02's run checked")
    return bad


FW_WHY = ("without the positions whose content target is a function word (R2_common.FUNCTION_WORDS), which R2's "
          "filter counts as content")


def without_function_word_targets(data: Dict[str, Any]) -> Dict[str, Any]:
    """The reading without the positions whose usable target is a function word: they leave every metric, the
    stuck-word share included (were they kept with no target, a lens right there, naming "the", would be counted as
    stuck on it). A prompt left with no position leaves the condition."""
    keep = [p for p in data["positions"]
            if not (p.get("ct_id") is not None and R2.word_key(p.get("ct_str")) in R2.FUNCTION_WORDS)]
    return dict(data, positions=keep)


S6_KEY = "S6_without_function_word_targets"
GATE_RULE = ("a band is served only if R1 passes R2's five rules there (Holm over the 2 bands) AND passes them again "
             "under S6, without function-word targets (Holm over the 2 bands within S6); the owner's choice of 28 "
             "October 2026 (R4_PREREG.md §2, §5)")


def r4_decision(an: Dict[str, Any]) -> Dict[str, Any]:
    """R4's decision: R2's rule (R3_06's decide) and the S6 gate, both required at a band."""
    s6 = an["sensitivity"][S6_KEY]["bands"].get("r1", {})
    by_band: Dict[str, Any] = {}
    served: List[int] = []
    for band, layers in R4_DESIGN.bands:
        r2 = bool(an["bands"]["r1"][band]["passes"])
        g = bool((s6.get(band) or {}).get("passes"))
        by_band[band] = {"page": PAGE[band], "r2_rule_passes": r2, "s6_passes": g, "served": r2 and g,
                         "s6_failed_rules": sorted(k for k, v in ((s6.get(band) or {}).get("rules") or {}).items()
                                                   if not v)}
        if r2 and g:
            served += list(layers)
    return {"rule": GATE_RULE, "by_band": by_band, "served_layers": {"r1": served}}


def analyze_r4(data: Dict[str, Any]) -> Dict[str, Any]:
    """R3_06's analysis under R4's design, S6 (the whole rule again, Holm included, without function-word targets),
    and R4's decision, which requires both (an["r4_decision"]; R3_06's own an["decision"] is R2's rule alone)."""
    an = A.analyze(R4_DESIGN, data)
    an["sensitivity"][S6_KEY] = A.sensitivity(R4_DESIGN, without_function_word_targets(data), an["bands"], "S6",
                                              FW_WHY)
    an["r4_decision"] = r4_decision(an)
    return an


# ------------------------------------------------------------------------------------------------ R3's R1 record
CMP_STATS = ("second", "jpp", "naive", "diff_jpp", "diff_naive")


def compare_with_r3(an: Dict[str, Any], ref: Dict[str, Any], bands: Sequence[str] = ("55", "58")) -> Dict[str, Any]:
    """R1 at each band of an R4-design analysis against R3_06_analysis.json's reference_bands.r1: rules 1, 2, 3 and 5;
    rule 4 on the raw p (R3 took R1's p outside a Holm family); the raw p, the sign counts, the stuck-word maximum and
    every interval (pooled and per condition, rounded as stored). Lists every difference."""
    mism: List[str] = []
    verdict_same = True
    rows = []
    for band in bands:
        g = an["bands"]["r1"][band]
        r = ref["reference_bands"]["r1"][band]
        for k in ("1", "2", "3", "5"):
            if g["rules"][k] != r["rules"][k]:
                mism.append(f"{band} rule {k}: R4 {g['rules'][k]}, R3 {r['rules'][k]}")
                verdict_same = False
        raw4 = g["p_sign"] < R4_DESIGN.alpha
        if raw4 != r["rules"]["4"]:
            mism.append(f"{band} rule 4 on the raw p: R4 {raw4}, R3 {r['rules']['4']}")
            verdict_same = False
        if g["p_sign"] != r["p_sign"]:
            mism.append(f"{band} raw sign-test p: R4 {g['p_sign']!r}, R3 {r['p_sign']!r}")
        for b in ("jpp", "naive"):
            for k in ("n_pos", "n_neg", "n_tie"):
                if g[f"sign_{b}"][k] != r[f"sign_{b}"][k]:
                    mism.append(f"{band} sign test against {b}: {k} R4 {g[f'sign_{b}'][k]}, R3 {r[f'sign_{b}'][k]}")
        if g["stuck_max"] != r["stuck_max"]:
            mism.append(f"{band} stuck_max: R4 {g['stuck_max']}, R3 {r['stuck_max']}")
        for s in CMP_STATS:
            for where, gs, rs_ in [("pooled", g[s]["pooled"], r[s]["pooled"])] + [
                    (c, g[s]["by_cond"].get(c, {}), r[s]["by_cond"].get(c, {})) for c in r[s]["by_cond"]]:
                for k in ("n", "mean", "lo", "hi"):
                    if gs.get(k) != rs_.get(k):
                        mism.append(f"{band} {s} {where} {k}: R4 {gs.get(k)}, R3 {rs_.get(k)}")
        r3_pass = all(r["rules"][k] for k in ("1", "2", "3", "4", "5"))
        rows.append({"band": band, "page": PAGE[band], "r3_passes": r3_pass,
                     "r4_design_passes_on_raw_p": all(g["rules"][k] for k in ("1", "2", "3", "5")) and raw4,
                     "diff_naive_mean": g["diff_naive"]["pooled"]["mean"], "r3_diff_naive_mean": r["diff_naive"]["pooled"]["mean"],
                     "diff_jpp_mean": g["diff_jpp"]["pooled"]["mean"], "r3_diff_jpp_mean": r["diff_jpp"]["pooled"]["mean"],
                     "p_sign": g["p_sign"], "r3_p_sign": r["p_sign"]})
    return {"verdict_same": verdict_same, "mismatches": mism, "exact": not mism, "rows": rows}


def restrict(data: Dict[str, Any]) -> Dict[str, Any]:
    """A reading's positions with only R4's readouts at R4's layers."""
    pos = []
    for p in data["positions"]:
        q = dict(p)
        q["r"] = {k: {str(l): p["r"][k][str(l)] for l in C.LENS_LAYERS[k] if str(l) in p["r"].get(k, {})}
                  for k in C.KINDS if k in p["r"]}
        pos.append(q)
    return dict(data, positions=pos)


def r3_design() -> A.Design:
    """R4's design with R3's bootstrap seed and R3's overlap list, for R3's own prompts."""
    return replace(R4_DESIGN, boot_seed=C.R3_BOOT_SEED, overlap=dict(C3.FIT_OVERLAP))


def reproduce_r3(eval_path: Path = C3.EVAL_JSON, analysis_path: Path = C3.ANALYSIS_JSON) -> Dict[str, Any]:
    raw = eval_path.read_bytes()
    data = restrict(json.loads(raw))
    ref = json.loads(analysis_path.read_text(encoding="utf-8"))
    _t, _b, bands, dec, _r = A.decision_suite(r3_design(), data)
    cmp = compare_with_r3({"bands": bands}, ref)
    return {"schema": "R4_03_r3_reproduction/1", "written_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "input": {"path": str(eval_path.relative_to(C.REPO)), "sha256": hashlib.sha256(raw).hexdigest()},
            "reference": {"path": str(analysis_path.relative_to(C.REPO)), "sha256": C.sha256_file(analysis_path)},
            "design": "R4's (R1 at bands 55 and 58, J++ and the naked eye) with R3's bootstrap seed 20261022",
            "not_compared": "the Holm-adjusted p (R3 took R1's p outside a Holm family, holm_m 1; R4 adjusts over 2) "
                            "and sensitivity verdicts (R3_06 runs them for candidates only)",
            "ok": cmp["exact"], **cmp}


# ------------------------------------------------------------------------------------------------ R2 + R3 + R4
def pooled_record(r4data: Dict[str, Any], r2_path: Path = R2.EVAL_JSON, r3_path: Path = C3.EVAL_JSON) -> Dict[str, Any]:
    """Reported only, never a decision (§8): R1 - J++ and R1 - naked eye at 55 and 58 over R2's, R3's and R4's
    prompts together (the prompt-level bootstrap stratified by condition over all three sets), and each set's own
    point estimates."""
    sets = {"R2": restrict(json.loads(r2_path.read_bytes())), "R3": restrict(json.loads(r3_path.read_bytes())),
            "R4": restrict(r4data)}
    merged = {"positions": [p for d in sets.values() for p in d["positions"]],
              "prompts": [p for d in sets.values() for p in d.get("prompts", [])]}
    D = replace(R4_DESIGN, overlap={})
    _t, boot, bands, _d, _r = A.decision_suite(D, merged)
    out: Dict[str, Any] = {"note": "reported only, never a decision", "bootstrap": {"seed": D.boot_seed, "n": D.n_boot},
                           "bands": {}}
    for band, layers in D.bands:
        ev = bands["r1"][band]
        row = {"page": PAGE[band], "n_prompts": ev["diff_naive"]["pooled"]["n"],
               "diff_naive": {k: ev["diff_naive"]["pooled"][k] for k in ("mean", "lo", "hi")},
               "diff_jpp": {k: ev["diff_jpp"]["pooled"][k] for k in ("mean", "lo", "hi")}, "by_set": {}}
        for name, d in sets.items():
            t = A.Table(D, d)
            b = A.Boot(t, 1, 0)
            row["by_set"][name] = {
                f"diff_{base}": A.r4(np.mean(np.concatenate(
                    [t.prompt_scores(A.PRIMARY, "r1", list(layers), c)[b.idx(c, 0)[0]]
                     - t.prompt_scores(A.PRIMARY, base, list(layers), c)[b.idx(c, 0)[0]] for c in D.conditions])))
                for base in ("jpp", "naive")}
        out["bands"][band] = row
    return out


# ------------------------------------------------------------------------------------------------ the report
def report_md(an: Dict[str, Any], meta: Dict[str, Any]) -> str:
    D = R4_DESIGN
    L: List[str] = ["## R4 result: R1's instruct lens at pages 56 and 59 (code 55 and 58)", ""]
    if meta.get("selftest"):
        L += ["**SELF-TEST DATA (fake model or synthetic file): these numbers mean nothing.**", ""]
    dev = [d for d in (("run with --force" if meta.get("forced") else None), (meta.get("git") or {}).get("deviation"))
           if d]
    runs = an.get("runs_finished")
    if dev or (runs and runs > 1):
        L += [f"**Deviations: {'; '.join(dev) or 'none recorded in the run'}. R4_02 finished {runs} run(s); this is "
              f"the last (§9).**", ""]
    L.append("Applied exactly as pre-registered (R4_PREREG.md §5: R2's five rules, unchanged, by R3's code). R1 passes "
             "R2's rule at a band only if all five hold: (1) pooled, its filtered top word names the model's next word "
             "(R2's filter: function words count) more often than both filtered J++ and the filtered naked eye (95% "
             "prompt-bootstrap intervals above 0); (2) it is significantly worse than neither in any condition; (3) "
             f"its stuck-word share is at most 25% in every condition; (4) the exact sign test against both baselines "
             f"is below {D.alpha} after Holm's adjustment over the {an['constants']['HOLM_M']} bands; (5) both pooled "
             f"differences are at least {D.min_effect}. **A band is served only if R1 also passes all five again under "
             "S6, without function-word targets** (the owner's choice, §2). A served band permits a later serving track "
             "to show R1's filtered top-1 at that page; R4 serves nothing itself.")
    L.append("")
    rd = an.get("r4_decision") or {}
    if rd:
        L.append("| page (code) | R2's rule | S6, without function-word targets | served |")
        L.append("|---|---|---|---|")
        for band, _ in D.bands:
            x = rd["by_band"][band]
            ev = an["bands"]["r1"][band]
            r2 = "passes" if x["r2_rule_passes"] else "fails rule " + ", ".join(
                k for k in ("1", "2", "3", "4", "5") if not ev["rules"].get(k))
            s6 = "passes" if x["s6_passes"] else "fails rule " + ", ".join(x["s6_failed_rules"])
            L.append(f"| {PAGE[band]} ({band}) | {r2} | {s6} | **{'SERVED' if x['served'] else 'not served'}** |")
        L.append("")
    L.append("R2's rule, in detail:")
    L.append("")
    L.append("| page (code) | R1 | J++ | naked eye | R1 - J++ | R1 - naked eye | +/- vs J++ | +/- vs naked | "
             "p (Holm over 2) | worse in | max stuck (condition, word) | R2's rule |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for band, _ in D.bands:
        ev = an["bands"]["r1"][band]
        worse = "; ".join(w.split(":")[0] + (" (J++)" if "J++" in w else " (naked)") for w in ev["worse_in"]) or "-"
        where = ev.get("stuck_max_where")
        stw = f"{ev['stuck_max']:.2f} ({where[1]}, {where[2]!r})" if where else "n/a"
        failed = [k for k in ("1", "2", "3", "4", "5") if not ev["rules"].get(k)]
        verdict = "PASSES" if ev["passes"] else "fails rule " + ", ".join(failed)
        sj, sn = ev["sign_jpp"], ev["sign_naive"]
        L.append(f"| {PAGE[band]} ({band}) | {A.fmt_num(ev['second']['pooled']['mean'])} | "
                 f"{A.fmt_num(ev['jpp']['pooled']['mean'])} | {A.fmt_num(ev['naive']['pooled']['mean'])} | "
                 f"{A.fmt_ci(ev['diff_jpp']['pooled'])} | {A.fmt_ci(ev['diff_naive']['pooled'])} | "
                 f"{sj['n_pos']}/{sj['n_neg']} | {sn['n_pos']}/{sn['n_neg']} | {ev['p_sign_holm']:.2g} | {worse} | "
                 f"{stw} | {verdict} |")
    L.append("")
    fails = [(band, ev) for band, ev in an["bands"]["r1"].items() if not ev["passes"]]
    if fails:
        L.append("Why each band that fails R2's rule fails:")
        for band, ev in fails:
            L.append(f"- page {PAGE[band]} (code {band}): " + "; ".join(ev["reasons"]))
        L.append("")
    s6b = ((an.get("sensitivity") or {}).get(S6_KEY) or {}).get("bands", {}).get("r1", {})
    s6_fails = [(band, ev) for band, ev in s6b.items() if not ev["passes"]]
    if s6_fails:
        L.append("Why each band that fails S6 fails (without function-word targets):")
        for band, ev in s6_fails:
            L.append(f"- page {PAGE[band]} (code {band}): R1 - J++ {A.fmt_ci(ev['diff_jpp'])}, R1 - naked eye "
                     f"{A.fmt_ci(ev['diff_naive'])}; " + "; ".join(ev["reasons"]))
        L.append("")
    L.append("Per condition (prompt-level 95% intervals; the decision conditions):")
    L.append("")
    L.append("| page | condition | prompts | R1 | R1 - J++ | R1 - naked eye | stuck share |")
    L.append("|---|---|---|---|---|---|---|")
    for band, _ in D.bands:
        ev = an["bands"]["r1"][band]
        for c in D.conditions:
            share = ev["stuck"][band][c]["share"]
            L.append(f"| {PAGE[band]} | {c} | {ev['second']['by_cond'][c]['n']} | "
                     f"{A.fmt_num(ev['second']['by_cond'][c]['mean'])} | {A.fmt_ci(ev['diff_jpp']['by_cond'][c])} | "
                     f"{A.fmt_ci(ev['diff_naive']['by_cond'][c])} | {A.fmt_num(share, '{:.2f}')} |")
    L.append("")
    L.append("Sensitivity analyses (pre-registered; the decision above stands):")
    L.append("")
    L.append("| analysis | passes | verdicts that change |")
    L.append("|---|---|---|")
    for s in an.get("sensitivity", {}).values():
        if "not_run" in s:
            L.append(f"| {s['name']} ({s['why']}) | not run: {s['not_run']} | - |")
            continue
        passes = "; ".join(f"page {PAGE[b]} {'yes' if v['passes'] else 'no'}"
                           for b, v in (s["bands"].get("r1") or {}).items())
        ch = "; ".join(f"{c['lens_band']}: {'pass' if (c['primary'] or {}).get('passes') else 'fail'} -> "
                       f"{'pass' if (c['sensitivity'] or {}).get('passes') else 'fail'}" for c in s["changes"]) or "none"
        L.append(f"| {s['name']} ({s['why']}) | {passes} | {ch} |")
    L.append("")
    sec = an.get("secondary", {})
    fws = (sec.get("function_word_split") or {}).get("r1") or {}
    if fws:
        L.append("Targets that are function words and targets that are not (R2's filter counts both as content; "
                 "share of positions named, reported only):")
        L.append("")
        L.append("| page | where | function-word targets: R1 / J++ / naked (n) | other targets: R1 / J++ / naked (n) |")
        L.append("|---|---|---|---|")
        for band, _ in D.bands:
            sp = fws.get(band) or {}
            for g in ("pooled",) + tuple(D.conditions):
                cells = []
                for part in ("function", "other"):
                    vals = [((sp.get(k) or {}).get(g) or {}).get(part) or {} for k in ("r1", "jpp", "naive")]
                    cells.append(" / ".join(A.fmt_num(v.get("mean")) for v in vals) + f" ({vals[0].get('n', 0)})")
                L.append(f"| {PAGE[band]} | {g} | {cells[0]} | {cells[1]} |")
        L.append("")
    if sec:
        L.append("Reported only (§8):")
        L.append("")
        L.append("| page | filtered exact id: R1 - J++ / R1 - naked | filtered top-5 word: R1 - J++ / R1 - naked | "
                 "unfiltered next word-piece: R1 / J++ / naked |")
        L.append("|---|---|---|---|")
        for band, _ in D.bands:
            fi = sec["filtered_exact_id"]["r1"][band]
            f5 = sec["filtered_top5_word"]["r1"][band]
            un = sec["unfiltered_next_token"]["r1"][band]
            L.append(f"| {PAGE[band]} | {A.fmt_ci(fi['diff_jpp'])} / {A.fmt_ci(fi['diff_naive'])} | "
                     f"{A.fmt_ci(f5['diff_jpp'])} / {A.fmt_ci(f5['diff_naive'])} | "
                     f"{A.fmt_num(un['second']['mean'])} / {A.fmt_num(un['jpp']['mean'])} / "
                     f"{A.fmt_num(un['naive']['mean'])} |")
        L.append("")
    ctx = an.get("context", {})
    if ctx:
        agree = ", ".join(f"{c} {v['model_agrees_with_next']}" for c, v in ctx.items()
                          if isinstance(v, dict) and "model_agrees_with_next" in v)
        L.append(f"The reading forward's own top-1 at 59 against the generated next token (R2 §5, report only): {agree}.")
        low = [c for c, v in ctx.items() if isinstance(v, dict) and v.get("agreement_below_0.95")]
        if low:
            L.append(f"Below 0.95 in {', '.join(low)}: reported, and the rule applied unchanged (§9).")
        L.append("")
    pr = an.get("pooled_r2_r3_r4")
    if pr:
        L.append("Pooled over R2's, R3's and R4's prompts (reported only, never a decision):")
        L.append("")
        L.append("| page | prompts | R1 - naked eye | R1 - J++ | R2 / R3 / R4 alone: R1 - naked eye |")
        L.append("|---|---|---|---|---|")
        for band, row in pr["bands"].items():
            alone = " / ".join(f"{row['by_set'][s]['diff_naive']:+.3f}" for s in ("R2", "R3", "R4"))
            L.append(f"| {row['page']} | {row['n_prompts']} | {A.fmt_ci(row['diff_naive'])} | "
                     f"{A.fmt_ci(row['diff_jpp'])} | {alone} |")
        L.append("")
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------ self-test
T_KIND = 1638


def synthetic_r4(spec: Dict[str, str], seed: int = 11) -> Dict[str, Any]:
    """A results file in R4_02's shape (160 prompts per condition; the naked eye at 59 always right, so every prompt
    with a usable target is valid) where R1's pattern at each band is one of:
      win     R1 right at 8 in 10 prompts, the baselines at 3 in 10 -> passes
      worse   as win, but R1 never right in chat-begun -> rule 2 only
      stuck   as win, but R1's top word is ' kind' at 40% of bare -> rule 3 only
      split   no different in the first three conditions; in mid-reply (4 positions a prompt) R1 misses one position
              the baselines get at 6 in 10 prompts and gets all four they miss at 4 in 10: pooled +0.0625, signs
              96 against / 64 for -> rule 4 only
      small   right where the baselines are, and also at 3% of prompts -> rule 5 only
      none    as the baselines -> rules 1, 4 and 5."""
    rng = np.random.default_rng(seed)
    positions, prompts = [], []
    small = {(c, i) for c in C.CONDITIONS for i in range(0, 160, 33)}            # 5 a condition, 20 of 640: 0.031
    for cond in C.CONDITIONS:
        for i in range(160):
            pid = f"r4-{C.PREFIX[cond]}-{i + 1:03d}"
            prompts.append({"pid": pid, "cond": cond, "about_model": i % 10 == 9})
            ks = [None] if cond != "mid-reply" else [3, 9, 15, 21]
            for q, k in enumerate(ks):
                draws = None if cond != "mid-reply" else [["any"], ["any"], ["content_next", "any"], ["any"]][q]
                tid = int(rng.integers(2100, 4000))
                word = f" w{tid}"
                p = {"i": len(positions), "pid": pid, "cond": cond, "draws": draws, "k": k, "pos": 10,
                     "read_tok": " x", "next_id": tid, "next_str": word, "next_content": True, "cont_ids": [],
                     "cont_strs": [], "ct_id": tid, "ct_str": word, "ct_offset": 0, "ct_prev_str": " x",
                     "model_top1_id": tid, "model_agrees": True, "r": {}}
                r = i % 10
                base_hit = r < 3
                if cond == "mid-reply":
                    base_hit = (q == 0) if r < 6 else False
                for kind in C.KINDS:
                    for l in C.LENS_LAYERS[kind]:
                        hit = base_hit
                        word_top = None
                        if kind == "naive" and l == 59:
                            hit = True
                        elif kind == "r1":
                            mode = spec[str(l)]
                            if mode in ("win", "worse", "stuck"):
                                hit = r < 8
                                if mode == "worse" and cond == "chat-begun":
                                    hit = False
                                if mode == "stuck" and cond == "bare" and r >= 6:
                                    word_top = (T_KIND, " kind")
                            elif mode == "split":
                                hit = base_hit if cond != "mid-reply" else (False if r < 6 else True)
                            elif mode == "small":
                                hit = base_hit or (cond, i) in small
                        if word_top is not None:
                            ftop, fstr = word_top
                        elif hit:
                            ftop, fstr = tid, word
                        else:
                            other = 6000 + (len(positions) * 131 + l * 7 + C.KINDS.index(kind) * 1009) % 3000
                            ftop, fstr = other, f" v{other}"
                        p["r"].setdefault(kind, {})[str(l)] = {"u": [ftop, 1, 2, 3, 4], "us": [fstr, "a", "b", "c", "d"],
                                                               "f": [ftop, 6, 7, 8, 9], "fs": [fstr, "e", "f", "g", "h"],
                                                               "u0c": True}
                positions.append(p)
    return {"schema": "R4_02/1", "meta": {"selftest": True}, "prompts": prompts, "positions": positions}


def selftest(record: bool = False) -> int:
    import tempfile

    fails: List[str] = []
    done: List[Tuple[bool, str]] = []
    t_start = time.time()

    def check(ok: bool, what: str) -> None:
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        done.append((bool(ok), what))
        if not ok:
            fails.append(what)

    D = R4_DESIGN
    check(D.bands == (("55", (55,)), ("58", (58,))) and D.candidates == ("r1",) and D.precedence == ("r1",)
          and D.boot_seed == 20261029 and D.require_all_bands and D.min_effect == 0.05 and D.alpha == 0.025
          and D.stuck_max == 0.25 and not D.reference,
          "R4's design: R1 alone at bands 55 and 58, precedence r1, Holm always over both, seed 20261029, R2's "
          "constants")
    cases = [({"55": "win", "58": "small"}, {"55": [], "58": ["5"]}),
             ({"55": "worse", "58": "stuck"}, {"55": ["2"], "58": ["3"]}),
             ({"55": "split", "58": "none"}, {"55": ["4"], "58": ["1", "4", "5"]})]
    an0 = None
    for spec, want in cases:
        an = analyze_r4(synthetic_r4(spec))
        an0 = an0 or an
        for band in ("55", "58"):
            ev = an["bands"]["r1"][band]
            got = sorted(k for k, v in ev["rules"].items() if not v)
            ok = got == want[band] and ev["passes"] == (not want[band]) and ev["served"] == (not want[band]) \
                and ev["holm_m"] == 2
            check(ok, f"R1 {band} as {spec[band]!r}: failed rules {got} (expected {want[band]})"
                      + ("" if ok else f"; {ev['reasons']}"))
    check(an0["decision"] == {"by_band": {"55": "r1", "58": None}, "served_layers": {"r1": [55]}}
          and an0["r4_decision"]["served_layers"] == {"r1": [55]}
          and an0["r4_decision"]["by_band"]["55"] == {"page": 56, "r2_rule_passes": True, "s6_passes": True,
                                                      "served": True, "s6_failed_rules": []},
          f"a pass at 55 alone, by R2's rule and by S6 (no target there is a function word), serves R1 at code 55 "
          f"only: {an0['r4_decision']['by_band']}")
    # Holm over the two bands: two raw p of 0.02 each fail rule 4 together (adjusted 0.04); 0.01 and 0.02 pass
    ev = {b: {"p_sign": p, "rules": {k: True for k in "1235"}, "reasons": [], "sign_jpp": {"n_pos": 1, "n_neg": 0},
              "sign_naive": {"n_pos": 1, "n_neg": 0}} for b, p in (("55", 0.02), ("58", 0.02))}
    A.finalize(D, {"r1": ev})
    ev2 = {b: {"p_sign": p, "rules": {k: True for k in "1235"}, "reasons": [], "sign_jpp": {"n_pos": 1, "n_neg": 0},
               "sign_naive": {"n_pos": 1, "n_neg": 0}} for b, p in (("55", 0.01), ("58", 0.02))}
    A.finalize(D, {"r1": ev2})
    check(not ev["55"]["rules"]["4"] and not ev["58"]["rules"]["4"] and ev["55"]["p_sign_holm"] == 0.04
          and ev2["55"]["rules"]["4"] and ev2["58"]["rules"]["4"] and ev2["55"]["holm_m"] == 2
          and holm({"a": 0.01, "b": 0.02}) == {"a": 0.02, "b": 0.02},
          "rule 4 is Holm over the two bands: raw p 0.02 and 0.02 both fail (adjusted 0.04), 0.01 and 0.02 both pass")
    # require_all_bands: R1 missing at code 58 fails rule 2 there and keeps both tests in the family
    dm = synthetic_r4({"55": "win", "58": "win"})
    for p in dm["positions"]:
        p["r"]["r1"].pop("58")
    anm = A.analyze(D, dm)
    evm = anm["bands"]["r1"]["58"]
    check(not A.check_family(D, anm) and evm["holm_m"] == 2 and not evm["rules"]["2"] and evm["p_sign"] == 1.0
          and evm["missing_layers"] == {"r1": [58]} and anm["bands"]["r1"]["55"]["passes"],
          "R1 missing at code 58 fails rule 2 there (p = 1) and the family stays the 2 registered tests")
    bad = A.check_family(D, dict(anm, constants=dict(anm["constants"], HOLM_M=1),
                                 bands={"r1": {"55": anm["bands"]["r1"]["55"]}}))
    check(len(bad) == 2, f"a family of one test is refused before anything is written ({bad})")
    s = an0["sensitivity"]
    check(s["S1_without_about_model"]["prompts_dropped"] == 64 and "S5_jpp_filter" in s
          and s["S5_jpp_filter"].get("not_run") and set(s) == {"S1_without_about_model", "S2_without_fit_overlap",
                                                               "S3_without_bare", "S4_without_formatlike_targets",
                                                               "S5_jpp_filter", "S6_without_function_word_targets"}
          and s["S6_without_function_word_targets"]["changes"] == [],
          "S1-S4 and S6 are run (S1 drops the 64 prompts about the model; S6 changes nothing where no target is a "
          "function word) and S5 is reported as not run")
    # S6: where R1's whole advantage is on function-word targets, the primary passes and S6 does not
    fw = synthetic_r4({"55": "none", "58": "none"})                 # R1 no better than the baselines ...
    for p in fw["positions"]:
        if (int(p["pid"][-3:]) - 1) % 10 in (3, 4, 5, 6, 7):     # ... except at targets that are " the"
            p["ct_str"] = p["next_str"] = " the"
            for r in p["r"]["r1"].values():
                r["f"], r["fs"] = [p["ct_id"]] + r["f"][1:], [" the"] + r["fs"][1:]
    an_fw = analyze_r4(fw)
    s6 = an_fw["sensitivity"]["S6_without_function_word_targets"]
    check(an_fw["bands"]["r1"]["55"]["passes"] and not s6["bands"]["r1"]["55"]["passes"]
          and {c["lens_band"] for c in s6["changes"]} == {"r1 55", "r1 58"}
          and an_fw["decision"] == {"by_band": {"55": "r1", "58": "r1"}, "served_layers": {"r1": [55, 58]}}
          and an_fw["r4_decision"]["served_layers"] == {"r1": []}
          and all(x["r2_rule_passes"] and not x["s6_passes"] and not x["served"]
                  for x in an_fw["r4_decision"]["by_band"].values()),
          "S6 makes function-word targets no target: where R1 wins only on them, both bands pass R2's rule and fail "
          "S6, so neither is served (R4's decision), though R2's rule alone would serve both")
    md_fw = report_md(an_fw, {"selftest": True, "forced": True})
    check("function-word targets: R1 / J++ / naked" in md_fw and "| 56 (55) | passes | fails rule" in md_fw
          and "**not served**" in md_fw and "Why each band that fails S6 fails" in md_fw
          and "Deviations: run with --force" in md_fw,
          "the report leads with R4's decision (R2's rule, S6, served), shows the function-word split, why S6 fails, "
          "and a forced run as a deviation at the top")
    meta_ok = {"files_sha256": {"R4_common.py": C.sha256_file(C.SCRIPTS / "R4_common.py")}}
    with tempfile.TemporaryDirectory() as tmp:
        rep = Path(tmp) / "rep.json"
        rep.write_text("{}")
        good = check_code(dict(meta_ok, replication_sha256=C.sha256_file(rep)), rep)
        other_common = check_code(dict(meta_ok, files_sha256={"R4_common.py": "x"}, replication_sha256=C.sha256_file(rep)), rep)
        other_rep = check_code(dict(meta_ok, replication_sha256="y"), rep)
    check(not good and other_common and other_rep and not C.pinned_differ(C.RULE_CODE),
          "the analysis refuses an R4_common.py or a replication record other than the run's, and the rule's code "
          "(R2_common, R2_03, R3_common, R3_06) has its pinned sha256")
    # the input checks
    counts_ok = A.check_input({"prompts": [{"cond": c} for c, n in C.N_PROMPTS.items() for _ in range(n)],
                               "meta": {"prompts_sha256": "x"}}, C.N_PROMPTS, "x", "t") == []
    check(counts_ok and len(A.check_input({"prompts": [{"cond": "bare"}] * 160, "meta": {}}, C.N_PROMPTS, "x", "t"))
          == 2, "the input check: R4's prompt counts (160 a condition) and the prompts' sha256")
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "runs.jsonl"
        r_none = check_run_finished("abc", log)
        log.write_text(json.dumps({"event": "started"}) + "\n" + json.dumps({"event": "failed", "error": "x"}) + "\n")
        r_failed = check_run_finished("abc", log)
        log.write_text(json.dumps({"event": "finished", "out_sha256": "abc"}) + "\n")
        r_ok = check_run_finished("abc", log)
        r_other = check_run_finished("def", log)
        rep = Path(tmp) / "rep.json"
        no_rep = check_replication(rep)
        rep.write_text(json.dumps({"ok": False}))
        bad_rep = check_replication(rep)
        rep.write_text(json.dumps({"ok": True}))
        good_rep = check_replication(rep)
        chk = Path(tmp) / "chk.json"
        chk.write_text(json.dumps({"ok": True, "prompts_sha256": "s"}))
        p_ok, p_bad = check_prompts_checked("s", chk), check_prompts_checked("t", chk)
    check(r_none and r_failed and not r_ok and r_other and no_rep and bad_rep and not good_rep and not p_ok and p_bad,
          "refused: no run logged, a last run that failed, an input other than the last finished run's output, a "
          "missing or failed replication record, prompts R4_01_check.json did not pass")
    try:
        refuse_selftest_into_results({"selftest": True}, C.ANALYSIS_JSON)
        guarded = False
    except SystemExit:
        guarded = True
    check(guarded, "self-test data is never written into experiments/results")
    # the comparison with R3's record: an analysis against itself agrees; one changed rule or interval is caught
    fake_ref = {"reference_bands": {"r1": an0["bands"]["r1"]}}
    same = compare_with_r3(an0, fake_ref)
    flip = json.loads(json.dumps(an0["bands"]["r1"], default=str))
    flip["55"]["rules"]["5"] = not flip["55"]["rules"]["5"]
    flip["58"]["diff_naive"]["pooled"]["lo"] = -1.0
    other = compare_with_r3(an0, {"reference_bands": {"r1": flip}})
    check(same["exact"] and same["verdict_same"] and not other["verdict_same"] and len(other["mismatches"]) == 2,
          "the comparison with R3's record of R1: equal to itself; a changed rule and a changed interval are both "
          "listed")
    md = report_md(an0, {"selftest": True})
    check("SELF-TEST" in md and "| 56 (55) |" in md and "PASSES" in md and "fails rule 5" in md and "S5" in md,
          "the report renders, with the self-test banner")
    an_b = A.analyze(D, synthetic_r4({"55": "win", "58": "small"}))
    check(A._norm(an_b["bands"]) == A._norm(an0["bands"]), "the bootstrap is deterministic")
    print(f"R4_03 self-test: {'ALL PASS' if not fails else str(len(fails)) + ' FAILURES'}")
    if record:
        C.record_selftest(Path(__file__).name, 'CUDA_VISIBLE_DEVICES="" .venv/bin/python -I '
                          'experiments/scripts/R4_03_analyze.py --selftest', done, time.time() - t_start)
    return 0 if not fails else 1


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="inp", default=str(C.EVAL_JSON))
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--record", action="store_true", help="with --selftest: keep the result in R4_selftests.json")
    ap.add_argument("--reproduce-r3", action="store_true")
    ap.add_argument("--allow-uncommitted", action="store_true", help="a recorded deviation")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest(record=args.record))
    if args.reproduce_r3:
        out = Path(args.out) if args.out else C.R3_REPRODUCTION_JSON
        rec = reproduce_r3()
        out.write_text(json.dumps(rec, indent=1, default=str) + "\n", encoding="utf-8")
        for m in rec["mismatches"][:40]:
            print("MISMATCH", m)
        for row in rec["rows"]:
            print(f"  r1 {row['band']} (page {row['page']}): R3 passes {row['r3_passes']}, here (raw p) "
                  f"{row['r4_design_passes_on_raw_p']}; R1 - naked {row['diff_naive_mean']} (R3 "
                  f"{row['r3_diff_naive_mean']}); R1 - J++ {row['diff_jpp_mean']} (R3 {row['r3_diff_jpp_mean']}); "
                  f"p {row['p_sign']:.3g} (R3 {row['r3_p_sign']:.3g})")
        print(f"{'R3 REPRODUCED EXACTLY' if rec['ok'] else 'R3 NOT REPRODUCED'}: {len(rec['mismatches'])} mismatches; "
              f"wrote {out}")
        sys.exit(0 if rec["ok"] else 1)
    out = Path(args.out) if args.out else C.ANALYSIS_JSON
    report = Path(args.report) if args.report else (
        C.REPORT_SECTION if out.resolve() == C.ANALYSIS_JSON.resolve() else out.with_suffix(".md"))
    raw = Path(args.inp).read_bytes()
    data = json.loads(raw)
    if data.get("schema") != "R4_02/1":
        raise SystemExit(f"unexpected schema {data.get('schema')}")
    meta = data.get("meta", {})
    refuse_selftest_into_results(meta, out, report)
    gst = None
    if not meta.get("selftest"):
        psha = C.sha256_file(C.PROMPTS_PATH)
        bad = A.check_input(data, C.N_PROMPTS, psha, "R4_02_eval.json")
        bad += check_prompts_checked(psha)
        bad += check_run_finished(hashlib.sha256(raw).hexdigest())
        bad += check_replication()
        bad += check_code(meta)
        if bad:
            raise SystemExit("the input is not R4's registered reading: " + "; ".join(bad))
        gst = C.git_state([C.SCRIPTS / "R4_03_analyze.py", C.SCRIPTS / "R4_common.py"])
        if gst["dirty"] or gst["head"] is None:
            msg = f"R4_03_analyze.py or R4_common.py is not committed at HEAD {gst['head']}: {gst['dirty']}"
            if not args.allow_uncommitted:
                raise SystemExit(msg + "; commit them, or pass --allow-uncommitted (a deviation)")
            gst["deviation"] = msg
    an = analyze_r4(data)
    bad = A.check_family(R4_DESIGN, an)
    if bad:
        raise SystemExit("the Holm family is not the registered one: " + "; ".join(bad))
    if C.RUNS_LOG.is_file():
        an["runs_finished"] = sum(json.loads(x).get("event") == "finished"
                                  for x in C.RUNS_LOG.read_text(encoding="utf-8").splitlines() if x.strip())
    an["analysis_code"] = {"git": gst, "R4_03_analyze.py": C.sha256_file(Path(__file__)),
                           "R4_common.py": C.sha256_file(C.SCRIPTS / "R4_common.py"),
                           "rule_code": {r: C.PINNED_UNCHANGED[r] for r in C.RULE_CODE}}
    an["pooled_r2_r3_r4"] = pooled_record(data)
    an["input"] = {"path": str(args.inp), "sha256": hashlib.sha256(raw).hexdigest(),
                   "selftest": bool(meta.get("selftest")), "run_files_sha256": meta.get("files_sha256"),
                   "run_git": meta.get("git")}
    an["written_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    out.write_text(json.dumps(an, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    md = report_md(an, meta)
    report.write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {out} and {report}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""GPU check of the causal spot-check on the real model (`causal_spot_check`, backend/causal_check.py): does the
instrument's test reproduce the study's own numbers, and do its premises hold on this runtime's W4A16 path?

Stop the :8765 server first (this loads the model itself and needs the whole card). It loads TheodoliteRuntime and the
vault, binds the scientist tools, and calls the tool directly (and once through the server app's WebSocket, in
process). Layers count from 0 here, as in the study's reports (page layer = code layer + 1). It checks:
A. N1A's gender poles (study mode, k = 60, activation; experiments/results/N1A_gender.json, tag kA): layer 50 neuron
   5313, face -: max|a| 6.531 (within 2%), c(' she') - c(' he') +13.81 (within 5%), mean KL 0.442 (within 10%), and at
   least 7 of its ten top rises among N1A's causal_top10; layer 50 n549, face -: -12.85 (within 5%); layer 40 n16757,
   face -: above 0 (N1A: +2.62).
B. N1V's linear-regime re-test (study mode, k = 10, the 24 N1 neurons, both faces;
   experiments/results/N1V_audit_04_smallk_metrics.json): at layer 50 the mean P@10 of the lens labels 0.546 and of the
   naked eye's 0.544 (within 0.05 each), the median split-half rho 0.754 (within 0.1), the mean rho(c+, c-) -0.917
   (within 0.05); at layer 20 both P@10 at most 0.02; the pooled median linearity at layer 20 0.965 at k = 10 and
   0.550 at k = 60 (within 0.03 and 0.06; experiments/results/N1V_audit_01_postnorm.json.full.json diag["20"]).
C. The premise, bit for bit on this path: a zero push is torch.equal to the base row and its KL exactly 0, in given
   mode (a short text, and 2048 word-pieces) and in study mode (180 right-padded rows); given mode's base row agrees
   with the model's own next-word choice; given mode over its frozen prefix cache agrees with a whole-text read of
   every row, on the short text and at 2048 pieces (past the 1024 window; the repeated prose, ending in the short
   question, where the model is not merely copying): the untouched rows within KL 0.05, c by Spearman at least 0.9,
   at least 7 of ten top rises shared, the push's KL within 10%; the two writes at k = 10 share at least 7 of
   n5313's ten top rises on a she/he text; the floor's KL is below the push's.
D. 2048 word-pieces in given mode through the server's WebSocket (in process): it completes, reports its forwards,
   the socket answers the next action, and torch's peak stays at or below the 2048 sweep's own (25.5 GiB, the comment
   above MAX_SWEEP_POSITIONS) plus 0.5 GiB.
E. The vault: after an UPDATE at layer 30, the study max|a| is recomputed (max_abs_act_cached false, vault_edits the
   edit's id); after ROLLBACK it is the clean model's again, reused from the cache, equal bit for bit.
Prints a summary and exits 0 when every check holds.

Run: .venv/bin/python -I scripts/verify_causal_check.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import asyncio
import json
import socket
import statistics
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# before torch, so the allocator's expandable segments apply (the note above `import torch` in engine_runtime)
from backend.engine_runtime import MAX_SWEEP_POSITIONS, TheodoliteRuntime  # noqa: E402

import torch  # noqa: E402

import backend.causal_check as cc  # noqa: E402
import backend.mcp_scientist as ms  # noqa: E402
from backend.larql_vault import LarqlVaultEngine  # noqa: E402

RESULTS = ROOT_DIR / "experiments" / "results"
SWEEP_2048_PEAK_GIB = 25.5  # torch's own peak of the worst sweep at 2048 (engine_runtime, above MAX_SWEEP_POSITIONS)
PEAK_MARGIN_GIB = 0.5
# given mode over the frozen prefix cache against a whole-text read of every row: the untouched rows may differ by batch
# shape alone (rows of different batch shapes differed by KL up to 0.014, backend/causal_check.py's header), so a few
# times that; a prefix read wrongly (a trimmed window, a wrong mask) moves the base row by far more
PREFIX_BASE_KL_MAX = 0.05
PREFIX_C_RHO_MIN = 0.9

failures: List[str] = []
passed = 0
summary: Dict[str, Any] = {}


def check(ok: bool, what: str) -> None:
    global passed
    if ok:
        passed += 1
        print(f"  ok    {what}")
    else:
        failures.append(what)
        print(f"  FAIL  {what}")


def near(x: float, want: float, rel: float) -> bool:
    return abs(x - want) <= rel * abs(want)


def tool(**kw: Any) -> Dict[str, Any]:
    return asyncio.run(ms.causal_spot_check(**kw))


def server_running(port: int = 8765) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def by_id(face: Dict[str, Any], tid: int) -> float:
    return next(w["dlogprob"] for w in face["tracked"] if w["token_id"] == tid)


def part_a(rt: TheodoliteRuntime) -> None:
    print("A. N1A's gender poles (study, k = 60)")
    gender = json.loads((RESULTS / "N1A_gender.json").read_text())["poles"]
    ref = {(p["layer"], p["neuron"], p["sign"]): p for p in gender if p["tag"] == "kA"}
    she, _ = rt.resolve_concept_token_id("she")
    he, _ = rt.resolve_concept_token_id("he")
    for (l, n), (rel_m, want_lean) in {(50, 5313): (0.02, "value"), (50, 549): (None, "value"), (40, 16757): (None, "sign")}.items():
        p = ref[(l, n, -1)]
        r = tool(layer=l, neuron_idx=n, polarity=-1, contexts="study", k=60.0, track_ids=[she, he])
        f = r["faces"]["-"]
        lean = by_id(f, she) - by_id(f, he)
        max_abs = abs(p["delta"]) / 60.0
        summary[f"A_L{l}_n{n}"] = {"max_abs": r["max_abs_act"], "she_minus_he": lean, "kl_mean": f["kl_mean"],
                                   "study": {"max_abs": max_abs, "she_minus_he": p["c_she_minus_he"], "kl": p["kl"]}}
        if rel_m is not None:
            check(near(r["max_abs_act"], max_abs, rel_m), f"L{l} n{n}: max|a| {r['max_abs_act']:.4f} vs {max_abs:.4f}")
        if want_lean == "value":
            check(near(lean, p["c_she_minus_he"], 0.05), f"L{l} n{n}: she-he {lean:+.3f} vs {p['c_she_minus_he']:+.3f}")
        else:
            check(lean > 0, f"L{l} n{n}: she-he {lean:+.3f} > 0 (N1A {p['c_she_minus_he']:+.3f})")
        if (l, n) == (50, 5313):
            check(near(f["kl_mean"], p["kl"], 0.10), f"L50 n5313: KL {f['kl_mean']:.4f} vs {p['kl']:.4f}")
            ours = [w["token"] for w in f["rises"][:10]]
            shared = len(set(ours) & set(p["causal_top10"]))
            check(shared >= 7, f"L50 n5313: {shared} of ten top rises shared with N1A ({ours})")


def part_b(rt: TheodoliteRuntime) -> None:
    print("B. N1V's linear-regime re-test (study, k = 10, 24 neurons)")
    neurons = json.loads((RESULTS / "N1_neurons.json").read_text())["neurons"]
    audit4 = json.loads((RESULTS / "N1V_audit_04_smallk_metrics.json").read_text())
    audit1 = json.loads((RESULTS / "N1V_audit_01_postnorm.json.full.json").read_text())["result"]["diag"]
    for l in (50, 20):
        p_lens, p_eye, split, anti, lin10 = [], [], [], [], []
        for n in neurons[str(l)]:
            r = tool(layer=l, neuron_idx=int(n), contexts="study", k=10.0, top_n=1)
            anti.append(r["antipodal"]["rho"])
            for face in ("+", "-"):
                f = r["faces"][face]
                p_lens.append(f["label_overlap"]["lens"]["p10"])
                p_eye.append(f["label_overlap"]["naked_eye"]["p10"])
                split.append(f["split_half_rho"])
                lin10 += f["linear"]["by_context"]["linearity"]
        ref = audit4[f"k10_L{l}"]
        got = {"p10_lens": statistics.mean(p_lens), "p10_eye": statistics.mean(p_eye),
               "split_half_median": statistics.median(split), "rho_mean": statistics.mean(anti),
               "linearity_median_k10": statistics.median(lin10)}
        summary[f"B_L{l}"] = {"ours": got, "study": ref}
        if l == 50:
            check(abs(got["p10_lens"] - ref["jpp|p10"]["real"]) <= 0.05, f"L50: P@10 lens {got['p10_lens']:.3f} vs {ref['jpp|p10']['real']}")
            check(abs(got["p10_eye"] - ref["naive|p10"]["real"]) <= 0.05, f"L50: P@10 eye {got['p10_eye']:.3f} vs {ref['naive|p10']['real']}")
            check(abs(got["split_half_median"] - ref["split_rho_median"]) <= 0.10,
                  f"L50: split-half {got['split_half_median']:.3f} vs {ref['split_rho_median']}")
            check(abs(got["rho_mean"] - ref["rho_c+_c-"]) <= 0.05, f"L50: rho(c+,c-) {got['rho_mean']:.3f} vs {ref['rho_c+_c-']}")
        else:
            check(got["p10_lens"] <= 0.02 and got["p10_eye"] <= 0.02,
                  f"L20: P@10 {got['p10_lens']:.3f} / {got['p10_eye']:.3f} at most 0.02")
            want10 = audit1["20"]["k10"]["sat"]["median"]
            check(abs(got["linearity_median_k10"] - want10) <= 0.03,
                  f"L20: linearity at k=10 {got['linearity_median_k10']:.3f} vs {want10:.3f}")
            lin60: List[float] = []
            for n in neurons["20"]:
                r = tool(layer=20, neuron_idx=int(n), contexts="study", k=60.0, top_n=1, floor=False)
                for face in ("+", "-"):
                    lin60 += r["faces"][face]["linear"]["by_context"]["linearity"]
            want60 = audit1["20"]["k60"]["sat"]["median"]
            got60 = statistics.median(lin60)
            summary["B_L20"]["ours"]["linearity_median_k60"] = got60
            check(abs(got60 - want60) <= 0.06, f"L20: linearity at k=60 {got60:.3f} vs {want60:.3f}")


def long_ids(rt: TheodoliteRuntime, n: int) -> List[int]:
    text = " ".join(t[1] for t in cc.STUDY_TEXTS if t[0] == "prose")
    ids = rt.tokenize_prompt(text)
    while len(ids) < n:
        ids += rt.tokenize_prompt(text, prepend_bos=False)
    return ids[:n]


def part_c(rt: TheodoliteRuntime) -> None:
    print("C. the premise on the W4A16 path")
    lm = cc.language_model(rt.hf_model)
    l, i = 50, 5313
    w_down = rt.slice_w_down(l, torch.tensor([i], device=rt.device))[0]
    norm = lm.layers[l].post_feedforward_layernorm
    wp, eps = norm.weight.detach(), float(norm.eps)
    short = rt.tokenize_prompt(rt.compose_prompt("Tell me about your sister.", chat=True, assistant_prefix="My sister is"))
    # the 2048 text is the repeated prose with the short question at its end: read on the repeats alone the model only
    # copies, near certain of the next piece, and a push there moves it by less (KL ~1e-5) than batch shape does
    bos = rt.tokenizer.bos_token_id
    tail = short[1:] if short[0] == bos else short
    long_q = long_ids(rt, MAX_SWEEP_POSITIONS - len(tail)) + tail
    cases = {"given, short": {"token_ids": short}, "given, 2048": {"token_ids": long_q},
             "study": {"texts_ids": ms._study_ids(rt)}}
    for label, kw in cases.items():
        for write in cc.CAUSAL_WRITES:
            r = cc.run_check(lm, l, i, [0.0, 10 * 6.53], write, w_down, wp, eps, rt.lm_head_weight, **kw)
            same = torch.equal(r.steered[0], r.base)
            klz = float(cc.kl(r.steered[0], r.base).abs().max())
            check(same and klz == 0.0, f"{label} ({write}): zero push == base bit for bit, KL {klz}")
            torch.cuda.empty_cache()
    # given mode reads through the frozen prefix cache (causal_check._frozen_prefix): against an ordinary read of the
    # whole text for every row, on the short text and past the 1024 window (2048 pieces), the untouched rows agree to
    # within batch-shape noise and the push moves the same words (one push, so two rows, as the 2048 sweep holds)
    summary["C_prefix"] = {}
    for label in ("given, short", "given, 2048"):
        ids = cases[label]["token_ids"]
        try:
            rk = cc.run_check(lm, l, i, [10 * 6.53], "activation", w_down, wp, eps, rt.lm_head_weight, token_ids=ids)
            torch.cuda.empty_cache()
            rw = cc.run_check(lm, l, i, [10 * 6.53], "activation", w_down, wp, eps, rt.lm_head_weight, token_ids=ids,
                              kv_cache=False)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            check(False, f"{label}: the whole-text read for the prefix comparison ran out of memory")
            continue
        base_kl = float(cc.kl(rk.base, rw.base).abs().max())
        ck, cw = (rk.steered[0] - rk.base)[0], (rw.steered[0] - rw.base)[0]
        rho = cc.spearman(ck, cw)
        shared10 = len(set(cc.top_ids(ck, 10)) & set(cc.top_ids(cw, 10)))
        kl_k, kl_w = float(cc.kl(rk.steered[0], rk.base)[0]), float(cc.kl(rw.steered[0], rw.base)[0])
        summary["C_prefix"][label] = {"base_kl": base_kl, "c_spearman": rho, "top10_shared": shared10,
                                      "push_kl_cached": kl_k, "push_kl_whole": kl_w}
        check(base_kl <= PREFIX_BASE_KL_MAX, f"{label}: prefix-cache base row vs whole text, KL {base_kl:.2e}")
        check(rho >= PREFIX_C_RHO_MIN and shared10 >= 7,
              f"{label}: prefix-cache c vs whole text, Spearman {rho:.4f}, {shared10} of ten top rises shared")
        check(abs(kl_k - kl_w) <= 0.1 * max(kl_w, 1e-9), f"{label}: the push's KL {kl_k:.4g} cached vs {kl_w:.4g} whole")
        del rk, rw
        torch.cuda.empty_cache()
    # the base row's top word-piece is the model's own choice (its bf16 logits, one forward of the whole text)
    r = tool(layer=l, neuron_idx=i, token_ids=short)
    with torch.no_grad():
        logits = rt.hf_model(input_ids=torch.tensor([short], device=rt.device), logits_to_keep=1).logits[0, -1]
    check(r["base_top"][0]["token_id"] == int(logits.argmax()), f"given base top {r['base_top'][0]} vs the model's own")
    ra = tool(layer=l, neuron_idx=i, token_ids=short, polarity=-1)
    rf = tool(layer=l, neuron_idx=i, token_ids=short, polarity=-1, write="frozen_norm")
    shared = len({w["token_id"] for w in ra["faces"]["-"]["rises"]} & {w["token_id"] for w in rf["faces"]["-"]["rises"]})
    check(shared >= 7, f"activation and frozen-norm writes share {shared} of ten top rises")
    check(ra["floor"]["-"]["kl_mean"] < ra["faces"]["-"]["kl_mean"], "the floor's KL is below the push's")
    summary["C"] = {"given_kl": ra["faces"]["-"]["kl_mean"], "floor_kl": ra["floor"]["-"]["kl_mean"],
                    "linearity": ra["faces"]["-"]["linear"]["linearity"], "writes_shared_top10": shared}


def part_d(rt: TheodoliteRuntime, vault: LarqlVaultEngine) -> None:
    print("D. 2048 word-pieces through the WebSocket")
    from fastapi.testclient import TestClient

    import backend.server as server

    server._RUNTIME, server._VAULT = rt, vault  # type: ignore[assignment]

    @asynccontextmanager
    async def no_lifespan(_app: Any):  # the model is loaded already
        yield

    server.app.router.lifespan_context = no_lifespan
    ids = long_ids(rt, MAX_SWEEP_POSITIONS)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    # a local Host, as the pages send it: backend/local_only.py refuses TestClient's "testserver"
    with TestClient(server.app).websocket_connect("ws://127.0.0.1:8765/ws/theodolite") as ws:
        ws.send_json({"action": "causal_check", "layer": 50, "neuron_idx": 5313, "token_ids": ids})
        msg = ws.receive_json()
        took = time.perf_counter() - t0
        ok = msg.get("type") == "causal_check"
        check(ok, f"2048 pieces: {msg.get('type')} {msg.get('message', '')} in {took:.1f} s")
        if ok:
            d = msg["data"]
            check(d["read_pos"] == MAX_SWEEP_POSITIONS - 1 and d["forwards"] >= 1,
                  f"2048 pieces: read at {d['read_pos']}, {d['forwards']} forward(s) {d['slots_per_forward']}")
        ws.send_json({"action": "inspect_pole", "layer": 50, "neuron_idx": 5313, "polarity": -1.0})
        check(ws.receive_json().get("type") == "pole_inspection", "the socket answers the next action")
    peak = torch.cuda.max_memory_allocated(rt.device) / 1024**3
    summary["D"] = {"seconds": round(took, 1), "peak_gib": round(peak, 2)}
    check(peak <= SWEEP_2048_PEAK_GIB + PEAK_MARGIN_GIB, f"2048 pieces: torch's peak {peak:.2f} GiB")


def part_e(rt: TheodoliteRuntime, vault: LarqlVaultEngine) -> None:
    print("E. the study max|a| under vault edits")
    clean = tool(layer=50, neuron_idx=5313, contexts="study", top_n=1)
    clean_table = getattr(rt, "_causal_study_maxabs")[()].clone()
    upd = vault.execute_lql("UPDATE LAYER 30 SLOT 100 SET object = 'cat' STRENGTH 4")
    pid = upd["patch"]["patch_id"]
    try:
        edited = tool(layer=50, neuron_idx=5313, contexts="study", top_n=1)
        check(edited["max_abs_act_cached"] is False and edited["vault_edits"] == [pid],
              f"after UPDATE: recomputed ({edited['max_abs_act_cached']}), edits {edited['vault_edits']}")
        summary["E"] = {"clean": clean["max_abs_act"], "edited": edited["max_abs_act"]}
    finally:
        vault.execute_lql("ROLLBACK")
    again = tool(layer=50, neuron_idx=5313, contexts="study", top_n=1)
    check(again["max_abs_act_cached"] is True and again["vault_edits"] == [] and again["max_abs_act"] == clean["max_abs_act"],
          "after ROLLBACK: the clean table reused")
    check(torch.equal(getattr(rt, "_causal_study_maxabs")[()], clean_table), "the clean table unchanged")


def main() -> int:
    if not torch.cuda.is_available():
        print("A CUDA GPU is needed.")
        return 2
    if server_running():
        print("Something answers on :8765: stop the Theodolite server first (this loads the model itself).")
        return 2
    t0 = time.perf_counter()
    rt = TheodoliteRuntime(load_jlens=False).load()
    vault = LarqlVaultEngine(rt)
    ms.bind_runtime(rt, vault)
    lm = cc.language_model(rt.hf_model)
    check(len(lm.layers) == 60 and ms._ffn_width(lm.layers[0].mlp.down_proj) == 21504,
          f"the model: {len(lm.layers)} layers of {ms._ffn_width(lm.layers[0].mlp.down_proj)} neurons")
    for part in (part_a, part_b, part_c):
        part(rt)
        torch.cuda.empty_cache()
    part_d(rt, vault)
    part_e(rt, vault)
    summary["seconds"] = round(time.perf_counter() - t0, 1)
    print(json.dumps(summary, indent=1, default=str))
    if failures:
        print(f"\nverify_causal_check: {len(failures)} of {passed + len(failures)} checks FAILED")
        return 1
    print(f"\nverify_causal_check: all {passed} checks hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Does an exported LoRA carry a push over to llama.cpp on Gemma-4-31B? Both engines on the CPU, on the same GGUF
(2 November 2026: gemma-4-31B-it-QAT-Q4_0.gguf, llama.cpp cd26896; README.md beside this file has the results).

    REAL_LORA_SET=<set> run.py instrument --gguf G --out DIR   # the instrument (GGUFRuntime, int4mm): first-piece
                                                               # log-probs, unpushed and each push, the start mark
                                                               # pushed (as llama.cpp) and spared
    REAL_LORA_SET=<set> run.py export     --gguf G --out DIR   # the adapters, from G's own columns (export_lora.py)
    REAL_LORA_SET=<set> run.py llama      --gguf G --out DIR --llama-server EXE   # llama-server -ngl 0, -t 10, the
                                                               # adapters per request (and one at SCALE 0.001)
    REAL_LORA_SET=<set> run.py read       --out DIR            # each push's change, llama.cpp's against the instrument's

The sets: none (census/: layer 45's census leads, 13641 at mul 0 and 5 and 21200 at mul 5; head 0 at 0, head 7 at 3), "strong" (strong/:
the same two at mul 30 and -20, heads 0-7 at 0, head 0 at 5) and "active" (active/: the neurons of layer 45 most active
on the six questions, |activation| x |W_down column| summed over their pieces but the start mark, which the
instrument step picks and writes to pushes.json: the first at mul 0 and 3, the first eight at mul 0 and -1). The
instrument step runs first (it writes the prompt ids llama.cpp is given).

The change compared: at the first word-piece written after each of break_rel's six census questions (chat, thinking
off), log-prob pushed minus unpushed, over the instrument's 200 likeliest unpushed pieces and 50 likeliest pushed.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

LAYER = 45  # counted from 0 (page 46): the census's layer 45, its two top-ranked neurons
PUSHES = {
    "n13641-mul0": {"muls": {LAYER: [(13641, 0.0)]}, "body": {"layer": LAYER, "n": 13641, "mode": "mul", "dose": 0}},
    "n13641-mul5": {"muls": {LAYER: [(13641, 5.0)]}, "body": {"layer": LAYER, "n": 13641, "mode": "mul", "dose": 5}},
    "n21200-mul5": {"muls": {LAYER: [(21200, 5.0)]}, "body": {"layer": LAYER, "n": 21200, "mode": "mul", "dose": 5}},
    "h0-x0": {"heads": {LAYER: [(0, 0.0)]}, "body": {"layer": LAYER, "head": 0, "dose": 0}},
    "h7-x3": {"heads": {LAYER: [(7, 3.0)]}, "body": {"layer": LAYER, "head": 7, "dose": 3}},
}
if os.environ.get("REAL_LORA_SET") == "strong":  # pushes that move the first piece well past llama.cpp's own floor
    PUSHES = {
        "n13641-mul30": {"muls": {LAYER: [(13641, 30.0)]}, "body": {"layer": LAYER, "n": 13641, "mode": "mul", "dose": 30}},
        "n21200-mul-20": {"muls": {LAYER: [(21200, -20.0)]}, "body": {"layer": LAYER, "n": 21200, "mode": "mul", "dose": -20}},
        "h0to7-x0": {"heads": {LAYER: [(h, 0.0) for h in range(8)]}, "body": [{"layer": LAYER, "head": h, "dose": 0} for h in range(8)]},
        "h0-x5": {"heads": {LAYER: [(0, 5.0)]}, "body": {"layer": LAYER, "head": 0, "dose": 5}},
    }
N_PROBS = 2048


def active_pushes(out):
    """REAL_LORA_SET=active: muls of the neurons of LAYER most active on the six questions (pushes.json, which the
    instrument step writes)."""
    top = json.loads((Path(out) / "pushes.json").read_text())["top"]
    one, eight = top[:1], top[:8]
    mk = lambda ns, f: {"muls": {LAYER: [(n, float(f)) for n in ns]},  # noqa: E731
                        "body": [{"layer": LAYER, "n": n, "mode": "mul", "dose": f} for n in ns]}
    return {f"n{one[0]}-mul0": mk(one, 0), f"n{one[0]}-mul3": mk(one, 3), "top8-mul0": mk(eight, 0), "top8-mul-1": mk(eight, -1)}


def export(a):
    import export_lora as el
    global PUSHES
    if os.environ.get("REAL_LORA_SET") == "active":
        PUSHES = active_pushes(a.out)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, p in PUSHES.items():
        t0 = time.time()
        code = el.main(["--gguf", a.gguf, "--push", json.dumps(p["body"]), "-o", str(out / f"{name}.gguf"), "--force"])
        print(f"{name}: exit {code}, {time.time() - t0:.1f} s")


def instrument(a):
    from backend import census as cz
    from backend import int4_cpu as ic
    from backend import runtime_factory as rf
    import check_lora_export as cle

    os.environ.update({rf.GGUF_ENV: a.gguf, rf.DEVICE_ENV: "cpu", rf.LINEARS_ENV: "int4mm", rf.LENS_ENV: "none"})
    t0 = time.time()
    rt = rf.make_runtime(load_jlens=False).load()
    print(f"loaded in {time.time() - t0:.0f} s: {rt.model_id}; {rt.weights_card['layers_as']}", flush=True)
    torch.set_grad_enabled(False)
    qs = cz.design_questions("break_rel")
    global PUSHES
    if os.environ.get("REAL_LORA_SET") == "active":  # the neurons of LAYER most active on these questions, |a| x |W[:, n]|
        from backend.causal_check import language_model

        down = language_model(rt.hf_model).layers[LAYER].mlp.down_proj
        norms = rt.write_col_norms(LAYER).float()
        score = torch.zeros_like(norms)
        h = down.register_forward_pre_hook(lambda _m, args: score.add_((args[0][0, 1:].float().abs() * norms).sum(0)))
        for q in qs.values():
            rt.hf_model(torch.tensor([rt.tokenize_prompt(rt.compose_prompt(q, chat=True))]))
        h.remove()
        top = [int(i) for i in score.topk(16).indices]
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "pushes.json").write_text(json.dumps({"top": top, "score": [float(score[i]) for i in top]}))
        PUSHES = active_pushes(a.out)
        print("most active:", top[:8], flush=True)
    res = {"model_id": rt.model_id, "layers_as": rt.weights_card["layers_as"], "questions": {}}
    lps = {}
    for qk, q in qs.items():
        ids = rt.tokenize_prompt(rt.compose_prompt(q, chat=True))
        res["questions"][qk] = {"text": q, "ids": list(map(int, ids))}
        t0 = time.time()
        lps[f"{qk}|plain"] = cle.instrument_logprobs(rt, ids, True)
        for name, p in PUSHES.items():
            kw = {k: p[k] for k in ("muls", "heads") if k in p}
            lps[f"{qk}|{name}|pushed"] = cle.instrument_logprobs(rt, ids, False, **kw)
            lps[f"{qk}|{name}|spared"] = cle.instrument_logprobs(rt, ids, True, **kw)
        print(f"{qk}: {len(ids)} pieces, {1 + 2 * len(PUSHES)} forwards in {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(Path(a.out) / "instrument.npz", **lps)
    (Path(a.out) / "instrument.json").write_text(json.dumps(res, indent=1))


def llama(a):
    out = Path(a.out)
    global PUSHES
    if os.environ.get("REAL_LORA_SET") == "active":
        PUSHES = active_pushes(a.out)
    meta = json.loads((out / "instrument.json").read_text())
    files = [out / f"{n}.gguf" for n in PUSHES]
    port = 39000 + os.getpid() % 900
    cmd = [a.llama_server, "-m", a.gguf, "--host", "127.0.0.1", "--port", str(port), "-c", "1024", "-np", "1", "-ngl", "0",
           "-t", str(a.threads), "--no-warmup", "--lora", ",".join(map(str, files)), "--lora-init-without-apply"]
    log = open(out / "llama-server.log", "ab")
    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        for _ in range(1200):
            if proc.poll() is not None:
                raise SystemExit("llama-server stopped at start: see llama-server.log")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        break
            except OSError:
                pass
            time.sleep(0.5)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/lora-adapters", timeout=10) as r:
            held = json.loads(r.read())
        print("adapters:", [(h["id"], Path(h["path"]).name, h["scale"]) for h in held], flush=True)
        assert [Path(h["path"]).name for h in held] == [f.name for f in files]
        V = 262144
        lps = {}

        def ask(ids, scales):
            body = json.dumps({"prompt": ids, "n_predict": 1, "temperature": 0, "n_probs": N_PROBS, "cache_prompt": False,
                               "lora": [{"id": i, "scale": float(s)} for i, s in enumerate(scales)]}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r:
                msg = json.loads(r.read())
            v = np.full(V, np.nan)
            for t in msg["completion_probabilities"][0]["top_logprobs"]:
                v[int(t["id"])] = float(t["logprob"])
            return v

        for qk, q in meta["questions"].items():
            t0 = time.time()
            lps[f"{qk}|plain"] = ask(q["ids"], [0] * len(files))
            lps[f"{qk}|floor"] = ask(q["ids"], [1e-3] + [0] * (len(files) - 1))  # any adapter on, at a vanishing scale
            for i, name in enumerate(PUSHES):
                lps[f"{qk}|{name}"] = ask(q["ids"], [1 if j == i else 0 for j in range(len(files))])
            print(f"{qk}: {1 + len(files)} requests in {time.time() - t0:.0f} s", flush=True)
        np.savez_compressed(out / "llama.npz", **lps)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=10) as r:
            props = json.loads(r.read())
        (out / "llama.json").write_text(json.dumps({"cmd": cmd, "build_info": props.get("build_info")}, indent=1))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


def read(a):
    out = Path(a.out)
    global PUSHES
    if os.environ.get("REAL_LORA_SET") == "active":
        PUSHES = active_pushes(a.out)
    I = dict(np.load(out / "instrument.npz"))
    L = dict(np.load(out / "llama.npz"))
    meta = json.loads((out / "instrument.json").read_text())
    report = {"model_id": meta["model_id"], "pushes": {}}
    floor = []
    for qk in meta["questions"]:
        ip, lp = I[f"{qk}|plain"], L[f"{qk}|plain"]
        top = np.argsort(-np.nan_to_num(ip, nan=-1e9))[:50]
        ok = ~np.isnan(lp[top])
        floor.append(float(np.max(np.abs(ip[top][ok] - lp[top][ok]))))
    report["unpushed: largest |log-prob difference| between the engines over the instrument's 50 likeliest"] = max(floor)
    if f"{qk}|floor" in L:
        fl = []
        for qk in meta["questions"]:
            S = np.argsort(-np.nan_to_num(I[f"{qk}|plain"], nan=-1e9))[:200]
            fl.append(float(np.nanmax(np.abs((L[f"{qk}|floor"] - L[f"{qk}|plain"])[S]))))
        report["llama.cpp's floor: the first adapter at scale 0.001 against none, largest |change| over the 200 likeliest"] = round(max(fl), 3)
    for name in PUSHES:
        rows = {"pushed": [], "spared": []}
        top1 = []
        for qk in meta["questions"]:
            ip = I[f"{qk}|plain"]
            dl = L[f"{qk}|{name}"] - L[f"{qk}|plain"]
            ipp = I[f"{qk}|{name}|pushed"]
            S = np.union1d(np.argsort(-np.nan_to_num(ip, nan=-1e9))[:200], np.argsort(-np.nan_to_num(ipp, nan=-1e9))[:50])
            for mark in ("pushed", "spared"):
                di = I[f"{qk}|{name}|{mark}"] - ip
                ok = ~np.isnan(dl[S]) & ~np.isnan(di[S])
                rows[mark] += list(zip(di[S][ok], dl[S][ok]))
            top1.append(int(np.nanargmax(L[f"{qk}|{name}"])) == int(np.nanargmax(ipp)))
        rep = {"same first piece as the instrument (start mark pushed)": f"{sum(top1)} of {len(top1)}"}
        for mark, pr in rows.items():
            di, dl = np.array(pr).T
            moved = np.abs(di) > 0.5
            rep[f"start mark {mark}"] = {
                "n": int(len(di)), "r": round(float(np.corrcoef(di, dl)[0, 1]), 4),
                "largest |instrument change|": round(float(np.abs(di).max()), 3),
                "moved > 0.5": int(moved.sum()),
                "median llama/instrument where moved": round(float(np.median(dl[moved] / di[moved])), 3) if moved.any() else None,
                "largest |llama - instrument change|": round(float(np.abs(dl - di).max()), 3)}
        report["pushes"][name] = rep
    (out / "summary.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["export", "instrument", "llama", "read"])
    ap.add_argument("--gguf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--llama-server", default="/home/eric/bin/ai/llama.cpp/build/bin/llama-server")
    ap.add_argument("--threads", type=int, default=10)
    a = ap.parse_args()
    {"export": export, "instrument": instrument, "llama": llama, "read": read}[a.step](a)

"""llama.cpp's own floor: the same request twice, and each adapter at a vanishing scale (0.001), on the 31B (CPU)."""
import json, os, subprocess, sys, time, urllib.request
from pathlib import Path
import numpy as np
out = Path(sys.argv[1]); gguf = sys.argv[2]; exe = sys.argv[3] if len(sys.argv) > 3 else "llama-server"
names = ["n13641-mul0", "n13641-mul5", "n21200-mul5", "h0-x0", "h7-x3"]
meta = json.loads((out / "instrument.json").read_text())
I = dict(np.load(out / "instrument.npz"))
port = 39950
cmd = [exe, "-m", gguf, "--host", "127.0.0.1", "--port", str(port), "-c", "1024", "-np", "1", "-ngl", "0", "-t", "10",
       "--no-warmup", "--lora", ",".join(str(out / f"{n}.gguf") for n in names), "--lora-init-without-apply"]
env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
proc = subprocess.Popen(cmd, stdout=open(out / "noise-server.log", "wb"), stderr=subprocess.STDOUT, env=env)
try:
    for _ in range(600):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
                if json.loads(r.read()).get("status") == "ok": break
        except OSError: pass
        time.sleep(0.5)
    def ask(ids, scales, n_probs=2048):
        body = json.dumps({"prompt": ids, "n_predict": 1, "temperature": 0, "n_probs": n_probs, "cache_prompt": False,
                           "lora": [{"id": i, "scale": float(s)} for i, s in enumerate(scales)]}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r: msg = json.loads(r.read())
        v = np.full(262144, np.nan)
        for t in msg["completion_probabilities"][0]["top_logprobs"]: v[int(t["id"])] = float(t["logprob"])
        return v
    rows = {}
    for qk, q in meta["questions"].items():
        ip = I[f"{qk}|plain"]; S = np.argsort(-np.nan_to_num(ip, nan=-1e9))[:200]
        a = ask(q["ids"], [0]*5); b = ask(q["ids"], [0]*5)
        d = lambda x, y: float(np.nanmax(np.abs((x - y)[S])))
        rows.setdefault("plain twice", []).append(d(a, b))
        for i, n in enumerate(names):
            for s in (1e-3, 1e-1):
                rows.setdefault(f"{n} at scale {s:g}", []).append(d(ask(q["ids"], [s if j == i else 0 for j in range(5)]), a))
    for k, v in rows.items(): print(f"{k:28s} largest |change| over the 200 likeliest, per question: " + " ".join(f"{x:.3f}" for x in v))
finally:
    proc.terminate(); proc.wait(timeout=20)

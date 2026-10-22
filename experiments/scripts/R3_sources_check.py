#!/usr/bin/env python3
"""R3 sources check (CPU only, offline; loads no model, uses no GPU): the cache R3_fetch.py filled, against the
committed manifest experiments/scripts/R3_sources.json.

1. jpp_lens: the checkout's HEAD is the pinned commit, it has no local changes, and every tracked file has the
   manifest's sha256 and size (no file missing, none added).
2. J++'s six eval item files: sha256, size and item count.
3. WikiText: the cached rows file's sha256 and size; the rows numbered from 0; the 1,000 records re-selected by J++'s
   rule from the rows, with the manifest's row indices, lengths and per-record sha256; both sequence hashes.
4. The environment, in a subprocess of its own interpreter with the GPU hidden and the hub offline:
   - versions and locations as recorded (torch and the rest from the shared .venv, jpp_lens from the checkout);
   - jpp_lens's modules import: jlens, the router (PCA, k-means), the expert, Jacobian and RelP fitters, the condense
     fitter and its rank objective, the LRP rules, the lenses;
   - they run on the CPU: fit_pca against numpy's eigh, k-means on three separate blobs, a router fit at E = 8 and
     PCA 64, the R-lens rules on jpp_lens's own tiny Gemma 4 (forward unchanged, the patch report covering every
     block norm and MLP and no attention norm), CondenseConfig's defaults as R3 registers them (D15);
   - jlens.examples.load_wikitext_prompts, J++'s own loader, run on the cached rows (the dataset stream replaced by
     them) returns exactly the 1,000 records, so the selection is J++'s;
   - scripts/jpp_cli.py and lens_evals fail to import for the one recorded reason (pandas is not installed); any
     other error fails the check, and once pandas is installed they must import and every stage's --help must run.
5. With --tests, jpp_lens's own test suite on the CPU: no test may fail, and a module may go uncollected only
   for want of pandas (listed).

Exits 1 on any failure.
Run: CUDA_VISIBLE_DEVICES="" /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R3_sources_check.py
       [--tests]
"""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "experiments" / "scripts" / "R3_sources.json"
T0 = time.time()
FAILURES: List[str] = []


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def check(ok: bool, what: str) -> bool:
    log("ok  " if ok else "FAIL", what)
    if not ok:
        FAILURES.append(what)
    return ok


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sequence_sha256(hashes: List[str]) -> str:
    return hashlib.sha256("\n".join(hashes).encode("ascii")).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


# ------------------------------------------------------------------------------------------------ 1-2. jpp_lens, items
def check_jpp(m: Dict[str, Any]) -> None:
    jm = m["jpp_lens"]
    d = REPO / jm["cache_dir"]
    if not check((d / ".git").exists(), f"jpp_lens checkout at {jm['cache_dir']}"):
        return
    check(git(d, "rev-parse", "HEAD").strip() == jm["commit"], f"jpp_lens HEAD is {jm['commit'][:12]}")
    check(git(d, "status", "--porcelain").strip() == "", "jpp_lens checkout has no local changes")
    tracked = sorted(p for p in git(d, "ls-files", "-z").split("\0") if p)
    check(tracked == sorted(jm["files"]), f"jpp_lens tracks the manifest's {len(jm['files'])} files")
    bad = [rel for rel, v in jm["files"].items()
           if not (d / rel).exists() or (d / rel).stat().st_size != v["size"] or sha256_file(d / rel) != v["sha256"]]
    check(not bad, f"every jpp_lens file matches its sha256 and size{' (not: ' + ', '.join(bad) + ')' if bad else ''}")
    im = m["jpp_eval_items"]
    for slug, v in im["files"].items():
        p = d / v["path"]
        ok = p.exists() and sha256_file(p) == v["sha256"] and p.stat().st_size == v["size"]
        n = len(json.loads(p.read_text(encoding="utf-8"))["items"]) if ok else None
        check(ok and n == v["n_items"], f"eval items {slug}: sha256 {v['sha256'][:12]}, {v['n_items']} items")


# ------------------------------------------------------------------------------------------------ 3. WikiText
def wikitext_records(wm: Dict[str, Any]) -> List[str]:
    rows_path = REPO / wm["cache_file"]
    if not check(rows_path.exists(), f"WikiText rows at {wm['cache_file']}"):
        return []
    check(sha256_file(rows_path) == wm["cache_file_sha256"] and rows_path.stat().st_size == wm["cache_file_size"],
          f"WikiText rows file sha256 {wm['cache_file_sha256'][:12]}")
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    check([r["row"] for r in rows] == list(range(wm["raw_rows_cached"])),
          f"{wm['raw_rows_cached']} raw rows, numbered from 0")
    check(sequence_sha256([text_sha256(r["text"]) for r in rows]) == wm["raw_rows_sequence_sha256"],
          "raw rows sequence sha256")
    min_chars = int(re.search(r">= (\d+)", wm["selection"]).group(1))
    kept = [r for r in rows if len(r["text"].strip()) >= min_chars]
    records = [[r["row"], len(r["text"]), text_sha256(r["text"])] for r in kept]
    check(records == wm["records"], f"the {len(wm['records'])} records: row indices, lengths and sha256")
    check(sequence_sha256([r[2] for r in records]) == wm["records_sequence_sha256"], "records sequence sha256")
    check(bool(records) and records[-1][0] == len(rows) - 1, "the rows end at the last kept record")
    check(wm["resolved_revision"] == wm["revision"], f"WikiText revision {wm['revision'][:12]} as resolved")
    return [r["text"] for r in rows]


# ------------------------------------------------------------------------------------------------ 4. environment probe
def probe(rows_path: str, jpp_dir: str) -> Dict[str, Any]:
    """Runs inside the env's own interpreter (--probe). Prints one JSON object of findings."""
    import importlib
    import importlib.metadata as md
    import types

    out: Dict[str, Any] = {"python": sys.version.split()[0]}
    names = ["jpp-lens", "torch", "transformers", "numpy", "huggingface_hub", "safetensors", "accelerate", "pytest"]
    out["versions"] = {n: md.version(n) for n in names}
    out["locations"] = {n: str(md.distribution(n).locate_file("")) for n in names}
    import numpy as np
    import torch as t

    out["cuda_available"] = t.cuda.is_available()
    modules = ["jlens", "jlens.examples", "jlens.hf", "workspace_lens", "workspace_lens.config",
               "workspace_lens.routing.cluster", "workspace_lens.routing.router",
               "workspace_lens.fitting.expert_fitting", "workspace_lens.fitting.jacobian_fitting",
               "workspace_lens.fitting.relp_fitting", "workspace_lens.fitting.utils",
               "workspace_lens.fitting.condense_experts", "workspace_lens.fitting.condense_experts.rank_objective",
               "workspace_lens.lrp", "workspace_lens.lrp.lrp", "workspace_lens.lenses.jacobian_lens",
               "workspace_lens.tests.tiny_gemma_olmo"]
    out["imports"] = {}
    for name in modules:
        try:
            mod = importlib.import_module(name)
            out["imports"][name] = str(Path(mod.__file__).resolve())
        except Exception as e:  # noqa: BLE001 - every failure is reported
            out["imports"][name] = f"ERROR {type(e).__name__}: {e}"
    out["src"] = str(Path(importlib.import_module("workspace_lens").__file__).resolve())

    from workspace_lens.routing.cluster import fit_pca, kmeans
    from workspace_lens.routing.router import ActivationRouterCollection

    g = t.Generator().manual_seed(0)
    pts = t.randn(400, 48, generator=g) @ t.randn(48, 48, generator=g)
    mean, comps = fit_pca(pts, 8)
    c = np.cov(pts.double().numpy(), rowvar=False)
    ref = np.linalg.eigh(c)[1][:, ::-1][:, :8]
    out["fit_pca_min_abs_cos"] = float(np.abs((comps.double().numpy() * ref).sum(0)).min())
    blobs = t.cat([t.randn(50, 4, generator=g) * 0.05 + t.tensor([k * 5.0, 0, 0, 0]) for k in range(3)])
    _, assign = kmeans(blobs, 3, seed=0)
    out["kmeans_blobs_recovered"] = all(len(set(assign[k * 50:(k + 1) * 50].tolist())) == 1 for k in range(3)) \
        and len(set(assign.tolist())) == 3
    router = ActivationRouterCollection(num_clusters=8, projection_dim=64, seed=0)
    router.fit({30: t.randn(600, 96, generator=g)})
    lr = router.layer_routers_L_dict[30]
    out["router_shapes"] = [list(lr.pca_components_ND.shape), list(lr.centroids_ED.shape)]

    from workspace_lens.fitting.condense_experts import CondenseConfig

    cc = CondenseConfig()
    out["condense_defaults"] = {k: getattr(cc, k) for k in
                                ("pass_k", "top_n", "steps", "learning_rate", "temperature", "exact_every")}

    from workspace_lens.lrp.lrp import apply_lrp_rules_for_mode
    from workspace_lens.tests.tiny_gemma_olmo import build_tiny_gemma4

    model = build_tiny_gemma4().float().eval()
    ids = t.randint(3, model.config.vocab_size, (1, 12), generator=g)
    with t.no_grad():
        stock = model(ids).logits
    with apply_lrp_rules_for_mode(model, "rlens") as report:
        with t.no_grad():
            patched = model(ids).logits
        names = report.patched_names()
    with t.no_grad():
        restored = model(ids).logits
    n_layers = model.config.num_hidden_layers
    out["lrp"] = {
        "max_abs_forward_change": float((patched - stock).abs().max()),
        "restored_bit_exact": bool(t.equal(restored, stock)),
        "n_layers": n_layers, "ln_rule": len(names["ln_rule"]), "mlp_rule": len(names["mlp_rule"]),
        "attention_norms_patched": [n for n in names["ln_rule"] if "self_attn" in n],
        "final_norm_patched": any(n.endswith("model.norm") for n in names["ln_rule"]),
        "classes": sorted({type(model.get_submodule(n)).__name__ for n in names["ln_rule"] + names["mlp_rule"]}),
    }

    rows = [json.loads(line)["text"] for line in Path(rows_path).read_text(encoding="utf-8").splitlines()]
    seen: Dict[str, Any] = {}

    def load_dataset(path, name=None, split=None, streaming=False, **kw):
        seen.update(path=path, name=name, split=split, streaming=streaming, extra=sorted(kw))
        return ({"text": text} for text in rows)

    sys.modules["datasets"] = types.SimpleNamespace(load_dataset=load_dataset)
    from jlens.examples import load_wikitext_prompts

    prompts = load_wikitext_prompts(1000)
    out["jpp_loader"] = {"call": seen, "n": len(prompts), "sha256": [text_sha256(p) for p in prompts]}
    del sys.modules["datasets"]

    out["cli"] = {}
    os.chdir(jpp_dir)                       # jpp_lens's data paths are relative to its checkout
    for name in ("lens_evals.readout_evals.readout_eval_items", "scripts/jpp_cli.py"):
        try:
            if name.endswith(".py"):
                import runpy

                runpy.run_path(name, run_name="jpp_cli_probe")
            else:
                importlib.import_module(name)
            out["cli"][name] = "imported"
        except ModuleNotFoundError as e:
            out["cli"][name] = f"ModuleNotFoundError: {e.name}"
        except Exception as e:  # noqa: BLE001
            out["cli"][name] = f"ERROR {type(e).__name__}: {e}"
    if out["cli"]["scripts/jpp_cli.py"] == "imported":
        stages = ["fit-router", "fit-shard", "merge-experts", "pooled-lens", "fit-weights", "evaluate"]
        out["cli_help"] = {s: subprocess.run([sys.executable, "-I", "-B", "scripts/jpp_cli.py", s, "--help"],
                                             capture_output=True).returncode for s in stages}
    return out


def check_env(m: Dict[str, Any], rows_ok: bool) -> None:
    em = m["environment"]
    env = REPO / em["dir"]
    py = env / "bin" / "python"
    if not check(py.exists(), f"env interpreter at {em['dir']}/bin/python"):
        return
    pth = env / "lib" / "python3.12" / "site-packages" / em["shared_pth"]
    check(pth.exists() and pth.read_text(encoding="utf-8").strip() == em["shared_site_packages"],
          f"{em['shared_pth']} points at the shared site-packages")
    if not rows_ok:
        return
    # -B here and below: no byte-code is written into the shared .venv the env reads from
    p = subprocess.run([str(py), "-I", "-B", str(Path(__file__).resolve()), "--probe",
                        str(REPO / m["wikitext"]["cache_file"]), str(REPO / m["jpp_lens"]["cache_dir"])],
                       capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"})
    if not check(p.returncode == 0, "the env probe ran"):
        print(p.stdout[-4000:], p.stderr[-4000:])
        return
    r = json.loads(p.stdout.strip().splitlines()[-1])
    check(not r["cuda_available"], "the GPU is hidden from the env")
    check(r["python"] == em["python"], f"env Python {em['python']}")
    check(r["versions"]["jpp-lens"] == em["jpp_lens"]["version"], f"jpp-lens {em['jpp_lens']['version']}")
    for n, v in em["reused_read_only"].items():
        check(r["versions"][n] == v and r["locations"][n].startswith(em["shared_site_packages"]),
              f"{n} {v} from the shared .venv")
    jpp_src = str((REPO / m["jpp_lens"]["cache_dir"] / "src").resolve())
    check(r["src"].startswith(jpp_src), "workspace_lens imported from the pinned checkout")
    errs = {k: v for k, v in r["imports"].items() if v.startswith("ERROR") or not v.startswith(jpp_src)}
    check(not errs, f"{len(r['imports'])} jpp_lens modules import from the checkout{' ' + str(errs) if errs else ''}")
    check(r["fit_pca_min_abs_cos"] > 1 - 1e-5,
          f"fit_pca equals numpy's eigh (min |cos| {r['fit_pca_min_abs_cos']:.8f})")
    check(r["kmeans_blobs_recovered"], "k-means recovers three separate blobs")
    check(r["router_shapes"] == [[96, 64], [8, 64]], f"router at E = 8, PCA 64: shapes {r['router_shapes']}")
    want = {"pass_k": 10, "top_n": 3, "steps": 200, "learning_rate": 0.05, "temperature": 0.1, "exact_every": 5}
    check(r["condense_defaults"] == want, f"CondenseConfig defaults as registered (D15): {r['condense_defaults']}")
    lr = r["lrp"]
    check(lr["max_abs_forward_change"] < 1e-5 and lr["restored_bit_exact"],
          f"R-lens rules leave the tiny Gemma 4's forward unchanged (max |diff| {lr['max_abs_forward_change']:.2e}) "
          "and restore it exactly")
    check(lr["mlp_rule"] == lr["n_layers"] and lr["ln_rule"] >= 4 * lr["n_layers"]
          and not lr["attention_norms_patched"],
          f"R-lens patch report: {lr['ln_rule']} norms, {lr['mlp_rule']} MLPs over {lr['n_layers']} layers, "
          f"no attention norm; final norm patched: {lr['final_norm_patched']}; classes {lr['classes']}")
    want_ids = [rec[2] for rec in m["wikitext"]["records"]]
    jl = r["jpp_loader"]
    check(jl["call"]["path"] == m["wikitext"]["dataset_id"] and jl["call"]["name"] == m["wikitext"]["config"]
          and jl["call"]["split"] == "train" and jl["call"]["streaming"],
          f"J++'s loader asks for {jl['call']['path']} {jl['call']['name']} {jl['call']['split']} (streamed)")
    check(jl["sha256"] == want_ids, f"J++'s own loader on the cached rows returns the {len(want_ids)} records")
    for name, status in r["cli"].items():
        if status == "imported":
            check(True, f"{name} imports")
        else:
            check(status == "ModuleNotFoundError: pandas" and "pandas" in em["not_installed"],
                  f"{name} does not import, for the recorded reason only ({status})")
    for stage, code in r.get("cli_help", {}).items():
        check(code == 0, f"jpp_cli.py {stage} --help")


# ------------------------------------------------------------------------------------------------ 5. jpp_lens's tests
def run_tests(m: Dict[str, Any]) -> None:
    """The whole suite, collection errors allowed but each one must be the missing pandas; no test may fail."""
    import tempfile
    import xml.etree.ElementTree as ET

    d = REPO / m["jpp_lens"]["cache_dir"]
    py = REPO / m["environment"]["dir"] / "bin" / "python"
    with tempfile.TemporaryDirectory() as tmp:
        xml_path = Path(tmp) / "junit.xml"
        p = subprocess.run([str(py), "-I", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
                            "--continue-on-collection-errors", f"--junitxml={xml_path}", "src"],
                           cwd=d, capture_output=True, text=True,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"})
        cases = list(ET.parse(xml_path).getroot().iter("testcase")) if xml_path.exists() else []
    passed, failed, uncollected, other = [], [], [], []
    for c in cases:
        err, fail, skip = c.find("error"), c.find("failure"), c.find("skipped")
        name = f"{c.get('classname')}::{c.get('name')}" if c.get("classname") else c.get("name", "")
        if fail is not None:
            failed.append(name)
        elif err is not None:
            pandas_only = (err.get("message") == "collection failure" and not c.get("classname")
                           and (err.text or "").rstrip().endswith("ModuleNotFoundError: No module named 'pandas'"))
            (uncollected if pandas_only else other).append(name)
        elif skip is None:
            passed.append(name)
    log("pytest:", p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-500:])
    if failed or other or not passed:
        print("\n".join(p.stdout.splitlines()[-80:]))
    check(bool(passed) and not failed and not other,
          f"jpp_lens's tests on the CPU: {len(passed)} passed, {len(failed)} failed, {len(other)} other errors")
    check("pandas" in m["environment"]["not_installed"] or not uncollected,
          f"{len(uncollected)} test modules not collected, each for want of pandas only: "
          + ", ".join(n.rsplit(".", 1)[-1] for n in uncollected))
    check(git(d, "status", "--porcelain").strip() == "", "the tests left the jpp_lens checkout unchanged")


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--probe":
        print(json.dumps(probe(sys.argv[2], sys.argv[3])))
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests", action="store_true", help="also run jpp_lens's own test suite on the CPU")
    args = ap.parse_args()
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    check_jpp(m)
    rows = wikitext_records(m["wikitext"])
    check_env(m, bool(rows))
    if args.tests:
        run_tests(m)
    if FAILURES:
        log(f"{len(FAILURES)} check(s) failed")
        return 1
    log("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

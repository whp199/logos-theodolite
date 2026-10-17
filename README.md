# Logos Theodolite

A geometry and circuit instrument for **Gemma-4-31B-it**, and a study it was used for: **is Gemma 4 "female-coded"?**

- **The instrument** (`backend/`, `frontend/`): a FastAPI server with a React UI. It reads Gemma-4-31B-it
  (QAT W4A16) through the J++ and Jacobian lenses. It shows layer trajectories, an attention/FFN split, GeGLU neuron "poles",
  reader "filaments", a 3D survey frame, steering, and a LARQL-style neuron vault. Tensor contracts are in
  [`ARCHITECTURE.md`](ARCHITECTURE.md).
- **The study** (`experiments/`): 7 experiments, 8 adversarial verifiers, a neuron-label validation study and a lens refit
  on the instruct model. That is 314 GPU jobs through a shared lab daemon. **Start with
  [`experiments/REPORT.md`](experiments/REPORT.md).**

Built on the Jacobian lens and J-space of Gurnee et al. (2026), the J++ lens of Ayonrinde & Lindsey (2026) as hosted by
Neuronpedia, Goodfire's neuron-"pole" framing (Cammarata 2026), interp-engine, and LARQL's naming (chrishayuk). The full,
web-checked reference list is in [`REFERENCES.md`](REFERENCES.md).

## Headline findings

- **Gemma-4-31B is not female-coded by default.** With no system prompt it calls itself *it* (0/320 *she* across
  single- and multi-turn samples). Under the J++ and Jacobian lenses, its self-token gender readout equals a nameless AI's
  to within about 1–3% of the woman–man persona span.
- **The "female" readings trace to other causes:**
  - the bare name **"Gemma" written as an antecedent**; "Gemma 4" stays *it*;
  - a forced-choice format that defaults to *woman*, plus an "assistant" stereotype;
  - late-layer lens artifacts;
  - a raw persona axis contaminated by an AI/service-role offset.
- **J-space at pronoun tokens encodes a role default, not resolved reference.** Quoted speech dissociates the two, while
  the model's actual predictions resolve the referent.
- **A "quiet carry channel" in Gemma 4.** About 180 early-loud residual dimensions are read with up to 1700× RMSNorm
  gain. Every Jacobian or gradient covector concentrates there, which is why lens-derived steering vectors break the model.
- **Neuron-pole labels are valid only at layers 40–57**, where J++ ≈ the plain logit lens. Input-side labels are
  near-useless.
- **The shipped J++ lens was fitted on the base model.** An instruct-fitted lens fixes late-layer fidelity
  ([`R1_IT_LENS_REPORT.md`](experiments/results/R1_IT_LENS_REPORT.md)).

## Status and known issues — read before using the tool

This is research code.

**Already applied in `backend/`:** two verified fixes from the V7 audit.
- [`V7_module1_fix.patch`](experiments/results/V7_module1_fix.patch): the attention/FFN split now accounts for Gemma 4's
  per-layer `layer_scalar`.
- [`V7_bos_fix.patch`](experiments/results/V7_bos_fix.patch): no more double BOS for chat-template prompts.

The E6/V7 tool-audit numbers were measured on the pre-patch backend. The repo history keeps the original instrument as its
first commit and the patches as a separate commit.

**Fixed as request options (the server's own defaults are unchanged, so old requests mean what they meant):**
- **Steering (bug 1).** `SteeringLaserRequest` takes `direction="fnw"` (fnw ⊙ ΔW_U, no inverse, no clamp), `mask_quiet`
  (zero the 181 persistent quiet dims and each pushed layer's quietest tenth), `spare_bos` (a position mask over the
  start mark) and `strength` (push `strength × ‖h‖` at every position, `NormScaledAddSpec`). The frontend's nudge page
  sends all four by default at strength 0.01. Without them the server still defaults to scale 650 at layers
  [29, 35, 41, 47, 53] with the clamped inverse direction, which destroys the output (bug #1 in
  [`V7_tool_bugs_NOTES.md`](experiments/results/V7_tool_bugs_NOTES.md)). Masking the 181 dims alone is not enough: a
  fnw she/he nudge masked that way still collapsed at 0.005 × ‖h‖ in a check on the running model; with the quietest
  tenth masked too it read fluently at 0.005–0.01 and broke at 0.02.
- **Survey frame (bug 3).** `frame_squaring="lowdin"` (order-free), `frame_fnw` (final-norm weighted rows),
  `frame_per_layer` (pulled back by each layer's J++) and `centre_prompts` (subtract a matched control's state). Every
  sighting now reports the fraction of the state the frame catches (`proj_frac`) and the words' cosines and condition
  numbers (`frame.geometry`).
- **Threads (bug 4).** Every filament carries the direct-path estimates `direct_gate` / `direct_act` beside the
  coupling C; `thread_score="direct"` (sweeps) and `score="direct"` (`trace_downstream_readers`) select and rank by it.
- **Chat template (bug 5).** `chat`, `system` and `assistant_prefix` on sweeps, traces and hypothesis tests: the
  tokenizer's chat template (thinking off), an optional system turn, then the begun answer, with one BOS. On by default
  in the frontend.
- **Poles (bug 8).** `pole_ranking="write"` ranks by |a| · ‖w_down ⊙ w_post‖; every layer reports each pole's write size
  and the share of the FFN write the listed poles explain.

See [`experiments/REPORT.md` §3.4](experiments/REPORT.md).

**LARQL:** `backend/larql_vault.py` only calls the `larql`/`vindex` binaries for `--version`. All vault queries run in
Python. Its ".patch MEMIT" edit is a single-neuron column nudge, not MEMIT.

## Repository layout

| Path | Contents |
|---|---|
| `backend/` | `theodolite_math.py` (the four modules, CKA, unembed), `engine_runtime.py` (interp-engine runtime, lens loader, W4A16 dequantiser), `server.py` (FastAPI + WebSocket), `mcp_scientist.py` (FastMCP tools), `larql_vault.py` |
| `frontend/` | Vite + React UI: nine pages (instrument, layer leaf, neuron folio, landmarks, nudge, vault, lexicon, lineage, welcome) with "rubrics" and "apparatus" reading toggles; shows the study's recorded sighting (E6) when no server answers |
| `scripts/` | `verify_optics.py` (CLI checks), `verify_e2e_ui_and_mcp.py` (server, MCP and headless-Chrome UI checks) |
| `experiments/REPORT.md` | Main write-up |
| `experiments/lab/` | `labd.py` (GPU daemon holding the model and lenses), `labclient.py` (submit jobs), `labkit.py` (lens readouts, gender scores, corrected decomposition, interventions, exact same-position VJPs through the 4-bit model) |
| `experiments/scripts/` | 376 scripts; the filename prefix is the experiment ID (see below) |
| `experiments/results/` | Per-experiment `*_REPORT.md` / `*_NOTES.md`, JSON results, the two patches, the frozen job log `labd_exec_log_2026-10-16.jsonl` |

**Experiment IDs and their write-ups**
- **Main experiments:**
  - E1 self-token readout → [report](experiments/results/E1_self_readout_REPORT.md)
  - E2 behaviour → [report](experiments/results/E2_behavior_REPORT.md)
  - E3 lens validation → [report](experiments/results/E3_lens_validation_REPORT.md)
  - E4 causal → [report](experiments/results/E4_causal_REPORT.md)
  - E5 name → [report](experiments/results/E5_name_REPORT.md)
  - E6 tool audit → [report](experiments/results/E6_tool_audit_REPORT.md)
  - E7 binding → [report](experiments/results/E7_binding_REPORT.md)
- **Adversarial verifiers:**
  - V1 name behaviour → [notes](experiments/results/V1_name_behavior_NOTES.md)
  - V2 forced choice → [notes](experiments/results/V2_forced_choice_NOTES.md)
  - V3 persona axis → [notes](experiments/results/V3_persona_axis_traits_NOTES.md)
  - V4 latent identity → [notes](experiments/results/V4_latent_identity_NOTES.md)
  - V5 binding dissociation → [notes](experiments/results/V5_binding_dissociation_NOTES.md)
  - V6 quiet dims → [notes](experiments/results/V6_quiet_dims_NOTES.md)
  - V7 tool bugs → [notes](experiments/results/V7_tool_bugs_NOTES.md)
  - V8 statistics audit → [notes](experiments/results/V8_stat_audit_NOTES.md)
- **Neuron-label study:** N1A (output, causal), N1B (activation labels) and N1C (integration) →
  [N1_REPORT](experiments/results/N1_REPORT.md); N1V verifiers → [code audit](experiments/results/N1V_code_audit.md) and
  [robustness](experiments/results/N1V_robustness.md).
- **Lens refit on the instruct model:** R1 → [report](experiments/results/R1_IT_LENS_REPORT.md).
- **Other scripts:**
  - S1/S2: quick checks on the neuron/lens synthesis;
  - `pilot_self_gender.py`: the first pilot;
  - `patch_verify.py`: an end-to-end check of the applied patches.

## Requirements

- One NVIDIA GPU with **≥ 32 GB**. The study ran on an RTX 5090 (32,607 MiB). The lab daemon uses about 21 GB with both
  lenses loaded.
- Linux, Python 3.12, [uv](https://github.com/astral-sh/uv), and Node 20+ for the UI.
- Disk:
  - about 22 GB for the model and 6.4 GB for the two lenses;
  - up to about 27 GB more if you regenerate every cached activation.
- Versions used: torch 2.14.1+cu130, transformers 5.20.0, interp-engine 1.14.0, compressed-tensors 0.19.0, fastapi 0.143.2,
  fastmcp 4.1.0.

**Run only one GPU process at a time.** The lab daemon, the instrument server and a standalone `mcp_scientist` each load
the 31B model. **Never expose the lab daemon:** it executes arbitrary submitted Python and binds to 127.0.0.1 only.

## Install and download

```bash
uv venv --python 3.12 .venv
uv pip install -r pyproject.toml
# the code runs with HF_HUB_OFFLINE=1, so download first:
.venv/bin/hf download google/gemma-4-31B-it-qat-w4a16-ct
.venv/bin/hf download neuronpedia/jacobian-lens \
    gemma-4-31b/jpp/Salesforce-wikitext/gemma-4-31B_jpp_lens.pt \
    gemma-4-31b/jlens/Salesforce-wikitext/gemma-4-31B_jacobian_lens.pt
```

The study used model snapshot `52f3f65bc7a02d555763bc923bd1d9094898219d` and lens snapshot
`e166deb5ad540932100ef803c8c9bcc2a5453694`.

## Run the instrument

```bash
(cd frontend && npm ci && npm run build)      # the server serves frontend/dist at /
.venv/bin/uvicorn backend.server:app --host 127.0.0.1 --port 8765
# open http://127.0.0.1:8765/ ; dev UI: (cd frontend && npm run dev), which proxies to :8765
```

## Run the lab and reproduce experiments

Run everything from the **repository root**: scripts use repo-relative paths.

```bash
.venv/bin/python experiments/lab/labd.py                     # ~15 s load, port 8765 (LABD_PORT to change)
python3 experiments/lab/labclient.py experiments/scripts/pilot_self_gender.py --out experiments/results/pilot_self_gender.json
```

- **Daemon jobs vs local scripts:** most scripts are daemon jobs. They use the pre-bound `rt`, `lab` and `STORE` and
  top-level `await`, and are submitted with `labclient.py`. Scripts named `*_analyze*`, `*_tables*` or `*_local_*` run
  directly with `.venv/bin/python`.
- **Results file conventions:**
  - `X.json` is the job's `result`.
  - `X.json.full.json` is the daemon's full response (`ok`, `stdout`, `error`, `result`).
  - `*.partial.json` files in E6/V7 hold the final data.
  - A `null` `.json`, or `ok: false`, marks a failed attempt; the rerun is named in the report.
- **Artifacts not in this repo:** about 27 GB of `*.pt`, `*.npz`, `*_cache/` and `*_acts/` files (activation caches,
  gradient chunks, fitted IT-lens tensors). Reports still mention them; regenerate them by re-running the job each report
  names.
- **Third-party text removed:** the N1B activation corpus (local `/usr/share/doc` and package sources) is not distributed,
  and verbatim contexts from it were removed from `N1B_labels.json`. Results that depend on it will differ slightly on
  another machine.

## Provenance and AI assistance

The experiments, analyses and reports in `experiments/` were designed, run and written by Claude Code agents (Anthropic),
directed by the repository owner. That covers parallel experiment agents, independent adversarial verifiers, and a separate
lens-refit session. Every headline claim in `experiments/REPORT.md` was re-tested by an independent verifier, and the
claims that failed are listed in §4 of that report.

## License, attribution and references

- Code is licensed under the **Apache License 2.0** ([`LICENSE`](LICENSE)). Third-party attributions and exceptions are in
  [`NOTICE`](NOTICE), and the full reference list is in [`REFERENCES.md`](REFERENCES.md).
- This repository does **not** redistribute model weights, lens tensors or third-party corpora.
- Gemma is a trademark of Google LLC. This project is independent research and is not affiliated with or endorsed by
  Google.

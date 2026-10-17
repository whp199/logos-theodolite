<p align="center">
  <img src="docs/readme/frontispiece.png" width="100%"
       alt="The opening page of Logos Theodolite: an engraved initial L beside the title, and sixty stacked rings with the path of one word winding up through them, labelled with what the model was thinking at a few of its layers.">
</p>

<p align="center">
  <em>The instrument, and the study it was first used for: is Gemma 4 “female-coded”?</em>
</p>

<p align="center">
  <sub>Gemma-4-31B-it &nbsp;◆&nbsp; 60 layers &nbsp;◆&nbsp; 5,376 dimensions in each &nbsp;◆&nbsp; a vocabulary of 262,144 word-pieces</sub>
</p>

<p align="center">
  <a href="#i--what-it-is">what it is</a> ·
  <a href="#ii--the-instrument-leaf-by-leaf">the instrument</a> ·
  <a href="#iii--what-the-study-found">what the study found</a> ·
  <a href="#iv--how-far-to-trust-it">how far to trust it</a> ·
  <a href="#v--setting-it-up">setting it up</a> ·
  <a href="#vi--the-shelves">the shelves</a> ·
  <a href="#vii--the-makers-marks">the makers’ marks</a>
</p>

<br>

## i · what it is

Between a question and its answer, a language model carries each word up through all of its layers, and little of that
country has been mapped. **Logos Theodolite** lets you follow a single word as it rises through the sixty layers of
**Gemma-4-31B-it**, and read, at every layer, what the model was thinking: sometimes in words you never wrote, in languages
you never used.

The repository holds two things.

- **The instrument** (`backend/`, `frontend/`). A FastAPI server and a React interface that read Gemma-4-31B-it (QAT W4A16)
  through the J++ and Jacobian lenses: the path of a word through the layers, what each layer gathers from the words before
  and recalls from memory, the neurons that fire, the threads between them, a survey frame of three landmark words, a
  nudge on the model's stream of thought, and a vault of neurons you can search and edit. The tensor contracts are in
  [`ARCHITECTURE.md`](ARCHITECTURE.md).
- **The study** (`experiments/`). Is Gemma 4 "female-coded"? Seven experiments, eight adversarial verifiers, a
  neuron-label validation study and a lens refit on the instruct model: 314 GPU jobs through a shared lab daemon.
  **Start with [`experiments/REPORT.md`](experiments/REPORT.md).**

<p align="center">
  <img src="docs/readme/two-readings.png" width="100%"
       alt="Two ways to read the instrument: rubrics, red directions for those beginning, switched on when you arrive; and the apparatus, the formulas, raw values, sources and every control, opened beneath a double gilt rule for scholars.">
</p>

Every page can be read two ways. **Rubrics**, the red italic lines, are directions for a first reading, as scribes and early
printers set their headings and directions apart in red. **The apparatus**, as in a critical edition, opens beneath each part
of a page under a double gilt rule: the technical name, the formula, the raw values, the sources and every control the
instrument has. Both are switches at the top of every page; nothing is hidden from either reader.

<br>

## ii · the instrument, leaf by leaf

<p align="center">
  <img src="docs/readme/instrument.png" width="100%"
       alt="The instrument page: sixty engraved rings with the path of the word I winding up from layer 1 to layer 60, labelled computer, HUAWEI, bot and am at several layers; beside it, the leaf of layer 36 with its top words through the lens and a bar splitting the layer's change into 37 percent gathered and 63 percent recalled.">
</p>

**The instrument.** One ring for each layer, the first at the bottom. The dark thread is the path of the word "I", in the
reply "Hi! I", as it rises; beside it stands what the model was thinking at a few layers, read through the J++ lens. The
gilded rings, every sixth, are the layers that look back over the whole conversation. On the right is the leaf of the chosen
layer: its likeliest words, and how much of its change was **gathered** from the words before (attention, lapis) or
**recalled** from memory (the feed-forward block, gilt).

<table>
  <tr>
    <td width="50%" valign="top">
      <img src="docs/readme/layer-leaf.png" alt="A layer's leaf: the numeral 36 at the hub of an engraved dial, with its bearing of 14 degrees and elevation of 52 degrees against the landmark words she, he and it, and a paragraph saying the model has not yet settled on a word.">
      <p><b>A layer’s leaf.</b> Where the thought points at one layer, as a bearing round the horizon and an elevation toward the
      third landmark, and what the model would say if it stopped there.</p>
    </td>
    <td width="50%" valign="top">
      <img src="docs/readme/neuron-folio.png" alt="A neuron's folio for neuron 15491 at layer 54, titled the I am neuron, with the particulars of the sighting it fired in.">
      <p><b>A neuron’s folio.</b> One of the 21,504 neurons in a layer: its two faces, what each writes through the lens and
      through the naked eye, and the later neurons that read it.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <img src="docs/readme/landmarks-frame.png" alt="The survey frame drawn as an engraved armillary: the landmark words she, he and it as axes, the thought at layer 36 marked with its bearing and elevation, and beside it the angles between each pair of words.">
      <p><b>Landmarks.</b> Three words fix a frame, as a surveyor sights known landmarks. Each layer’s thought is read as a
      bearing and an elevation against them, with the angles between the words beside it.</p>
    </td>
    <td width="50%" valign="top">
      <img src="docs/readme/nudge-dial.png" alt="The strength dial of the nudge, from times 0.001 to times 10 of the stream's own strength, clear up to times 0.01, gilded where the study's fluent nudge sat, and hatched red where it collapses, with the old defaults of 280 and 650 marked in the red.">
      <p><b>The nudge.</b> A small push on the stream of thought while the model reads, measured as a share of the stream’s own
      strength, with the old defaults marked where the study found they break the model.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <img src="docs/readme/vault.png" alt="The vault, where the model keeps its 1,290,240 neurons, with an engraved protractor whose pointer rests on layer 51 seeking she.">
      <p><b>The vault.</b> The catalogue of the model’s 1,290,240 neurons: look one up by the words it writes, change one, and
      read back what was asked before.</p>
    </td>
    <td width="50%" valign="top">
      <p><b>Also in the instrument</b></p>
      <p><b>Lexicon</b>: every word of the instrument in plain terms, with its formula, and eleven cautions on where it can
      mislead you. <b>Lineage</b>: the research it is made from, pillar by pillar, each with what the study found about it.</p>
      <p><sub>Every view above is a live sighting from the running server: “Tell me about yourself.”, its answer begun
      “Hi! I”, sent as a chat. With no server running, the pages show the study’s recorded sighting (E6) and say so.</sub></p>
    </td>
  </tr>
</table>

**The instrument’s words, and what they name**

| In the instrument | In the literature |
|---|---|
| a **sighting** | one 60-layer sweep of a prompt, read at its last word-piece |
| **layers 1–60** | decoder layers, counted from 1 (the code counts 0–59) |
| **the lens** · **the naked eye** | the J++ lens readout `unembed(h·Jₗᵀ)` · the logit lens `unembed(h)` |
| **gathering** · **remembering** | the attention write · the feed-forward (GeGLU) write, with Gemma 4’s layer scalar |
| a neuron’s **two faces** | the positive and negative poles of a GeGLU neuron (Cammarata 2026) |
| **threads** | reader filaments: a later neuron whose gate reads an earlier neuron’s write |
| **landmarks** | the survey frame: three directions, usually words, squared into axes |
| **the nudge** | additive activation steering on the residual stream |
| **the vault** | a LARQL/VINDEX-style catalogue of neurons, queried in Python |

<br>

## iii · what the study found

- **Gemma-4-31B is not female-coded by default.** With no system prompt it calls itself *it* (0/320 *she* across
  single- and multi-turn samples). Under the J++ and Jacobian lenses, its self-token gender readout equals a nameless AI’s
  to within about 1–3% of the woman–man persona span.
- **The "female" readings trace to other causes:**
  - the bare name **"Gemma" written as an antecedent**; "Gemma 4" stays *it*;
  - a forced-choice format that defaults to *woman*, plus an "assistant" stereotype;
  - late-layer lens artifacts;
  - a raw persona axis contaminated by an AI/service-role offset.
- **J-space at pronoun tokens encodes a role default, not resolved reference.** Quoted speech dissociates the two, while
  the model’s actual predictions resolve the referent.
- **A "quiet carry channel" in Gemma 4.** About 180 early-loud residual dimensions are read with up to 1700× RMSNorm
  gain. Every Jacobian or gradient covector concentrates there, which is why lens-derived steering vectors break the model.
- **Neuron-pole labels are valid only at layers 40–57**, where J++ ≈ the plain logit lens. Input-side labels are
  near-useless.
- **The shipped J++ lens was fitted on the base model.** An instruct-fitted lens fixes late-layer fidelity
  ([`R1_IT_LENS_REPORT.md`](experiments/results/R1_IT_LENS_REPORT.md)).

Every headline claim was re-tested by an independent verifier; the claims that did not survive are listed in §4 of the
report.

<br>

## iv · how far to trust it

*This is research code. Read a single sighting as a lead, never as a finding: the study reads lens outputs against a
matched control (REPORT §3.5), and the instrument can keep one for you.*

The study audited its own instrument ([`V7_tool_bugs_NOTES.md`](experiments/results/V7_tool_bugs_NOTES.md),
REPORT §3.4). Where each fault stands now:

| | The fault the audit found | Where it stands |
|---|---|---|
| **1** | The nudge’s defaults, scale 650 at five layers (2.4–7.4× the stream’s strength), with a clamped inverse direction, collapse the model | **Mended in the instrument.** `direction="fnw"`, `mask_quiet`, `spare_bos` and `strength` (a share of ‖h‖); the nudge page sends all four, at 0.01. The server’s own defaults are kept for old callers |
| **2** | The attention / feed-forward split ignored Gemma 4’s `layer_scalar` | **Applied**: [`V7_module1_fix.patch`](experiments/results/V7_module1_fix.patch) |
| **3** | The survey frame depends on the order of the words, and uses raw unembedding rows | **An option**: `frame_squaring="lowdin"` (order-free), `frame_fnw`, `frame_per_layer`, `centre_prompts`. The fraction of the state the frame catches is always sent |
| **4** | The thread score C is a weak guide: ρ ≈ 0.41 with measured effects, against 0.87 for a direct-path estimate | **An option**: `thread_score="direct"`; the direct-path estimate is sent beside C on every thread |
| **5** | No chat template: the instruct model ran as a raw language model | **Mended**: `chat`, `system`, `assistant_prefix`; on by default in the interface |
| **6** | A doubled start mark for chat prompts | **Applied**: [`V7_bos_fix.patch`](experiments/results/V7_bos_fix.patch) |
| **7** | One input label for a neuron’s two faces | Open |
| **8** | Neurons ranked by \|a\|, not by the size of their write; the top fifteen explain 10–41% of the feed-forward write | **An option**: `pole_ranking="write"`; every layer reports each neuron’s write size and the share they explain |
| **9** | The J++ lens was fitted on the base model and is applied to the instruct model | Open in the instrument; the refit is in [`R1_IT_LENS_REPORT.md`](experiments/results/R1_IT_LENS_REPORT.md) |

- The E6/V7 audit numbers were measured on the backend before these fixes. The history keeps the original instrument as
  its first commit.
- **The quiet mask.** Masking only the 181 persistently quiet dimensions is not enough: in a check made while building the
  fix (not part of the study), a she/he nudge masked that way still broke the answer at 0.005 × ‖h‖. `mask_quiet` also masks
  each pushed layer’s quietest tenth of dimensions; with both, the nudge read fluently at 0.005–0.01 and broke at 0.02.
- **LARQL.** `backend/larql_vault.py` calls the `larql`/`vindex` binaries only for `--version`; every vault query runs in
  Python, and its ".patch MEMIT" edit is a single-neuron column nudge, not MEMIT.

<br>

## v · setting it up

**What it needs**

- One NVIDIA GPU with **≥ 32 GB**. The study ran on an RTX 5090 (32,607 MiB); the instrument peaks near 23 GB.
- Linux, Python 3.12, [uv](https://github.com/astral-sh/uv), and Node 20+ for the interface.
- About 22 GB of disk for the model and 6.4 GB for the two lenses; up to about 27 GB more to regenerate every cached
  activation.
- Versions used: torch 2.14.1+cu130, transformers 5.20.0, interp-engine 1.14.0, compressed-tensors 0.19.0,
  fastapi 0.143.2, fastmcp 4.1.0.

> [!IMPORTANT]
> Run one GPU process at a time: the lab daemon, the instrument server and a standalone `mcp_scientist` each load the 31B model.

> [!WARNING]
> Never expose the lab daemon. It executes arbitrary submitted Python, and binds to 127.0.0.1 only.

**Install and download**

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

**Take a first sighting**

```bash
(cd frontend && npm ci && npm run build)      # the server serves frontend/dist at /
.venv/bin/uvicorn backend.server:app --host 127.0.0.1 --port 8765
# open http://127.0.0.1:8765/   ·   live-reloading interface: (cd frontend && npm run dev), proxied to :8765
```

The same sighting, asked of the server directly over its WebSocket, `/ws/theodolite` (every field below the first three
is optional, and each default reproduces the instrument as first built):

```jsonc
{"action": "sweep", "payload": {
  "prompt": "Tell me about yourself.", "chat": true, "assistant_prefix": "Hi! I",
  "system": null,                                   // an optional system turn
  "monuments": ["she", "he", "it"],                 // the three landmarks
  "frame_squaring": "lowdin", "frame_fnw": true,    // an order-free frame on the weighted rows
  "pole_ranking": "write", "thread_score": "direct",
  "steering": {"enabled": false, "direction": "fnw", "mask_quiet": true, "spare_bos": true, "strength": 0.01}
}}
```

It answers with one binary frame (the path, the split, kinship between layers, fifteen neurons a layer, the threads) and
one JSON frame of words and readings; [`ARCHITECTURE.md`](ARCHITECTURE.md) gives both. The circuit tools
(`inspect_pole`, `trace_downstream_readers`, `test_motif_hypothesis`, `query_larql_vault`) are served over HTTP and as
FastMCP tools from `backend/mcp_scientist.py`. To check a build end to end, server, tools and pages together:
`.venv/bin/python scripts/verify_e2e_ui_and_mcp.py`.

**Reproduce the study**

Run everything from the repository root: the scripts use repo-relative paths.

```bash
.venv/bin/python experiments/lab/labd.py                     # ~15 s load, port 8765 (LABD_PORT to change)
python3 experiments/lab/labclient.py experiments/scripts/pilot_self_gender.py --out experiments/results/pilot_self_gender.json
```

- **Daemon jobs and local scripts.** Most scripts are daemon jobs: they use the pre-bound `rt`, `lab` and `STORE` and
  top-level `await`, and are submitted with `labclient.py`. Scripts named `*_analyze*`, `*_tables*` or `*_local_*` run
  directly with `.venv/bin/python`.
- **Results files.** `X.json` is the job’s `result`; `X.json.full.json` is the daemon’s full response (`ok`, `stdout`,
  `error`, `result`); the `*.partial.json` files in E6/V7 hold the final data; a `null` `.json`, or `ok: false`, marks a
  failed attempt, and the report names the rerun.
- **Not in this repository.** About 27 GB of `*.pt`, `*.npz`, `*_cache/` and `*_acts/` files (activation caches, gradient
  chunks, fitted IT-lens tensors). The reports still mention them; re-run the job each report names to regenerate them.
- **Third-party text removed.** The N1B activation corpus (local `/usr/share/doc` and package sources) is not distributed,
  and verbatim contexts from it were removed from `N1B_labels.json`; results that depend on it will differ slightly on
  another machine.

<br>

## vi · the shelves

| Path | What is there |
|---|---|
| `backend/` | `theodolite_math.py` (the four modules, the frames, the direct-path score, CKA, the unembed), `engine_runtime.py` (the interp-engine runtime, lens loader, W4A16 dequantiser, chat template), `server.py` (FastAPI and the WebSocket), `mcp_scientist.py` (FastMCP tools), `larql_vault.py` |
| `frontend/` | Vite and React: nine pages (welcome, instrument, layer leaf, neuron folio, landmarks, nudge, vault, lexicon, lineage), the rubrics and apparatus switches, and the study’s recorded sighting for when no server answers |
| `scripts/` | `verify_optics.py` (command-line checks), `verify_e2e_ui_and_mcp.py` (server, tools and headless-Chrome page checks) |
| `experiments/REPORT.md` | The main write-up |
| `experiments/lab/` | `labd.py` (the GPU daemon holding the model and lenses), `labclient.py` (submit jobs), `labkit.py` (lens readouts, gender scores, the corrected decomposition, interventions, exact same-position VJPs through the 4-bit model) |
| `experiments/scripts/` | 376 scripts; the filename prefix is the experiment’s ID |
| `experiments/results/` | Per-experiment `*_REPORT.md` / `*_NOTES.md`, JSON results, the two patches, the frozen job log `labd_exec_log_2026-10-16.jsonl` |

<details>
<summary><b>The experiments and their write-ups</b></summary>

<br>

- **Main experiments:** E1 self-token readout ([report](experiments/results/E1_self_readout_REPORT.md)) · E2 behaviour
  ([report](experiments/results/E2_behavior_REPORT.md)) · E3 lens validation
  ([report](experiments/results/E3_lens_validation_REPORT.md)) · E4 causal ([report](experiments/results/E4_causal_REPORT.md)) ·
  E5 name ([report](experiments/results/E5_name_REPORT.md)) · E6 tool audit
  ([report](experiments/results/E6_tool_audit_REPORT.md)) · E7 binding ([report](experiments/results/E7_binding_REPORT.md))
- **Adversarial verifiers:** V1 name behaviour ([notes](experiments/results/V1_name_behavior_NOTES.md)) · V2 forced choice
  ([notes](experiments/results/V2_forced_choice_NOTES.md)) · V3 persona axis
  ([notes](experiments/results/V3_persona_axis_traits_NOTES.md)) · V4 latent identity
  ([notes](experiments/results/V4_latent_identity_NOTES.md)) · V5 binding dissociation
  ([notes](experiments/results/V5_binding_dissociation_NOTES.md)) · V6 quiet dims
  ([notes](experiments/results/V6_quiet_dims_NOTES.md)) · V7 tool bugs ([notes](experiments/results/V7_tool_bugs_NOTES.md)) ·
  V8 statistics audit ([notes](experiments/results/V8_stat_audit_NOTES.md))
- **Neuron-label study:** N1A (output, causal), N1B (activation labels) and N1C (integration) in the
  [N1 report](experiments/results/N1_REPORT.md); the N1V verifiers’ [code audit](experiments/results/N1V_code_audit.md) and
  [robustness check](experiments/results/N1V_robustness.md).
- **Lens refit on the instruct model:** R1 ([report](experiments/results/R1_IT_LENS_REPORT.md)).
- **Other scripts:** S1/S2, quick checks on the neuron and lens synthesis; `pilot_self_gender.py`, the first pilot;
  `patch_verify.py`, an end-to-end check of the applied patches.

</details>

<br>

## vii · the makers’ marks

<p align="center">
  <img src="docs/readme/cartouche.png" width="420"
       alt="An engraved maker's cartouche: Logos Theodolite, made after Gurnee et al., Ayonrinde and Lindsey, Cammarata, and Hayuk, with the logit lens, linear CKA and activation steering, for Gemma-4-31B-it, MMXXVI.">
</p>

This instrument is its maker’s synthesis of four recent pieces of research, joined on one mount:

- the **Jacobian lens and J-space** of Gurnee et al. (Anthropic, 2026) [2];
- the **J++ lens** of Ayonrinde & Lindsey (2026) [4], as hosted by Neuronpedia [6];
- **neurons as two-poled units**, from Cammarata’s *Neural Networks Are Already Clean* (Goodfire, 2026) [9];
- the **LARQL / VINDEX** naming of a neuron vault, after Hayuk (chrishayuk) [10];

over older foundations: the logit lens [8], the residual-stream view of Elhage et al. [14], linear CKA [13], activation
steering [21, 22], Löwdin’s symmetric orthogonalisation [33], and interp-engine [36], which runs the model. Bracketed numbers
refer to [`REFERENCES.md`](REFERENCES.md), the full, web-checked list; the interface tells the same story on its Lineage page.

**Provenance and AI assistance.** The experiments, analyses and reports in `experiments/` were designed, run and written by
Claude Code agents (Anthropic), directed by the repository owner: parallel experiment agents, independent adversarial
verifiers, and a separate lens-refit session. Every headline claim in `experiments/REPORT.md` was re-tested by an independent
verifier, and the claims that failed are listed in §4 of that report.

**License.** Code is licensed under the **Apache License 2.0** ([`LICENSE`](LICENSE)); third-party attributions and
exceptions are in [`NOTICE`](NOTICE). This repository does **not** redistribute model weights, lens tensors or third-party
corpora. Gemma is a trademark of Google LLC; this project is independent research, not affiliated with or endorsed by Google.

<br>

<p align="center"><sub>◆</sub></p>

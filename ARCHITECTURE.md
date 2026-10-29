# Logos Theodolite — Architecture & Tensor Contracts

**Target Model:** `google/gemma-4-31b-it` (`google/gemma-4-31B-it-qat-w4a16-ct` / 4-bit W4A16 within a single-GPU 32GB RTX 5090 VRAM envelope)  
**Coordinate Lenses:** `neuronpedia/jacobian-lens` (`gemma-4-31B_jpp_lens.pt` & `gemma-4-31B_jacobian_lens.pt`)  
**Execution Engine:** `decoderesearch/interp-engine` (LARQL / VINDEX3 naming is borrowed from `chrishayuk/larql`; the vault queries are implemented in Python in `backend/larql_vault.py`, and the `larql`/`vindex` binaries are optional — only their `--version` is reported)  
**References:** Gemma 4 [1], Jacobian lens [2,3], J++ [4], Neuronpedia lens files [6], logit lens [8], neuron poles [9], linear CKA [13], neuron-explorer [47]; see `REFERENCES.md`  
**Study & lab:** see `README.md` and `experiments/REPORT.md` (the `experiments/` tree is not shown in the layout below)

---

## 1. Directory Structure

```text
theodolite2/
├── ARCHITECTURE.md                  # System architecture, tensor contracts & wire protocol
├── pyproject.toml                   # Python dependencies & project config
├── backend/
│   ├── __init__.py
│   ├── theodolite_math.py           # Four core vectorized mathematical modules + contrasts + CKA + unembed
│   ├── engine_runtime.py            # interp-engine + J++ / JLens loader + 4-bit weight dequantizer
│   ├── w4a16_fast.py                # faster unpackers of those weights, the same bits (lut, compiled, resident; §2)
│   ├── gguf_weights.py              # GGUF block unpackers (CPU/GPU), GGUFLinear, dense/dense_cols/dense_rows (§5)
│   ├── gguf_model.py                # a Gemma 4 from a GGUF: config, tensors, tokenizer, provenance; a writer (§5)
│   ├── gguf_runtime.py              # GGUFRuntime: the pinned runtime on a GGUF, its packing accessors replaced (§5)
│   ├── runtime_factory.py           # make_runtime(): the runtime from THEODOLITE_GGUF / THEODOLITE_DEVICE (§5)
│   ├── larql_vault.py               # LARQL / VINDEX3 FFN vault indexer, LQL parser, .patch edits & their voice
│   ├── label_gate.py                # the label gate: label_check, LABEL_GATE_CARD (which neuron labels pages hold back)
│   ├── causal_check.py              # the causal spot-check: one neuron turned up at one word-piece (N1A's protocol)
│   ├── server.py                    # FastAPI + Binary Float32 WebSocket server + REST endpoints
│   ├── sampler_request.py           # how an answer is written, as a request asks: the sweep's and the atlas's sampler
│   ├── local_only.py                # who may ask: every route and the socket answer only this machine's pages (§4)
│   ├── mcp_scientist.py             # FastMCP server for autonomous circuit motif discovery
│   ├── atlas.py                     # the atlas's routes, /api/atlas/* (§4.7)
│   ├── atlas_ops.py                 # live neuron operations: pushes, three voices, reads, odds, heads, wiring, steer lens
│   ├── atlas_index.py               # the atlas's indexes on disk, and the queries over them (cone, search, links)
│   ├── atlas_build.py               # building those indexes on the GPU (lens, stats, wiring, steer lens, layout, passages)
│   ├── census.py                    # the census: every neuron of a layer pushed on fixed questions, ranked vs controls
│   ├── field_notes.py               # names and notes kept beside neurons and heads, shared by every page and chat; the folio's field book
│   ├── scientist_chat.py            # the atlas's scientist: Claude Code, Codex or an OpenAI-compatible endpoint (§4.8)
│   ├── scientist_tools.py           # the scientist's 30 tools, in the pages' count of layers (stdlib only)
│   ├── scientist_mcp.py             # those tools as a stdio MCP server, for Claude Code and Codex
│   ├── scientist_prompt.md          # the scientist's instructions (with scientist_methodology.md)
│   └── scientist_methodology.md
├── scripts/
│   ├── atlas_build.py               # builds the atlas's indexes (GPU): lens, stats, wiring, layout, steer, passages
│   ├── atlas_stand_in.py            # a tiny random Gemma 4 for the atlas's checks, and the server on it (--serve)
│   ├── census_run.py                # takes a layer's census (GPU), resumable, in planned batches or shards
│   ├── census_tools.py              # rank, merge, depth, spread, the dose window and canaries over census runs (CPU)
│   ├── pod/                         # a rented card's jobs, each with --dry-run on the stand-in (README §v):
│   │   ├── setup.sh                 #   everything on the volume, the pinned downloads, the venv from constraints, a self-test
│   │   ├── pilot.sh                 #   the one-hour pilot: steps per unpacker and batch, the writer's floor, projections
│   │   ├── census_shard.sh          #   one shard detached, logged, under a time limit, the pod stopped only when asked
│   │   ├── sync.sh                  #   pulls to this machine every N minutes, and a final pull checked by listing
│   │   ├── pod_tool.py              #   their Python: the self-test, the steps, the listing check, the pilot's summary
│   │   └── common.sh                #   the volume root, the env file, the card's lock, a child's environment free of secrets
│   ├── export_constraints.sh        # the home venv's exact versions in a dated constraints file, for a pod (CPU)
│   ├── gpu_log.py                   # rented cards' approvals, starts and stops, summed against the approval (CPU)
│   ├── check_pod_scripts.py         # every pod script's --dry-run end to end on the stand-in, gpu_log's sums (CPU only)
│   ├── check_atlas_ops.py           # the atlas's live operations against brute force on the stand-in (CPU only)
│   ├── check_atlas_index.py         # every index built on the stand-in against the same quantity read live (CPU only)
│   ├── check_atlas_api.py           # every atlas route against the operation it serves, the GPU lock, refusals (CPU only)
│   ├── check_census.py              # the census's statistics, dose window, units, resume, shards, merge and spread (CPU only)
│   ├── check_scientist.py           # the scientist's tools, MCP server and three backends against fakes (CPU only)
│   ├── check_local_only.py          # who may ask: every route, the socket and the lab daemon, refused or let in (CPU only)
│   ├── check_hypothesis_test.py     # test_motif_hypothesis's test against chance, its route and the folio's words (CPU only)
│   ├── check_causal_check.py        # the causal spot-check on a tiny CPU Gemma 4 (CPU only, run first by verify_e2e)
│   ├── check_contrast_readings.py   # one word against another: scores, labkit parity, backgrounds, μ's pin (CPU only)
│   ├── check_think_template.py      # the cached chat template's thinking switch, as measured (CPU only)
│   ├── check_vault_voice.py         # an edit's voice: order, failures, reply keys, over HTTP, the WebSocket and MCP (CPU only)
│   ├── check_dequant.py             # the faster unpackers against the runtime's, bit for bit, on a 4-bit stand-in (CPU only)
│   ├── check_gguf.py                # the GGUF path against gguf-py and the stand-in, bit for bit (CPU only)
│   ├── compare_gguf_ct.py           # does a GGUF hold the W4A16 checkpoint's weights? tensor by tensor (CPU)
│   ├── export_E3_background_mu.py   # writes experiments/results/E3_background_mu.npy (+ .json) from E3's caches (CPU)
│   ├── verify_atlas_writer.py       # the writer's floor: unpushed answers in each batch shape against one written alone (GPU)
│   ├── verify_causal_check.py       # the causal spot-check against N1A's and N1V's figures on the real model (GPU)
│   ├── verify_e2e_ui_and_mcp.py     # server, MCP tools and headless-Chrome pages end to end (GPU)
│   └── verify_optics.py             # Standalone deterministic CLI verification suite
└── frontend/
    ├── package.json
    ├── vite.config.ts
    ├── index.html
    ├── scripts/make_fixture.py        # builds src/fixtures/e6-demo-sighting.json from the E6 demo sweep
    └── src/
        ├── main.tsx
        ├── App.tsx                    # hash routes -> pages (each page its own chunk)
        ├── index.css                  # the visual system: vellum, ink, rubric red, gilt, lapis; EB Garamond
        ├── reading.ts                 # "rubrics" / "apparatus" reading toggles, remembered per viewer
        ├── router.ts                  # #/instrument, #/layer/36, #/neuron/54/15491/pos, #/atlas/54/15491, ...
        ├── lib/words.ts               # plain-word helpers (word-pieces, glosses, certainty)
        ├── lib/labels.ts              # the label gate on the pages (labelGate, trustLine, held-back wording)
        ├── components/HeldBack.tsx    # a held-back label's plate, reason and rubric; its words in the apparatus
        ├── fixtures/recordedLabelChecks.ts # the recorded labels' checks (written by N1G_01 from its results)
        ├── fixtures/e6-demo-sighting.json # the recorded sighting shown when no server answers
        ├── store/
        │   ├── useTheodoliteStore.ts  # Zustand state, WebSocket + HTTP wiring, nudge defaults
        │   └── sighting.ts            # one normalised "sighting" from a live frame pair or the fixture
        ├── components/kit.tsx         # page frame, nav, reading toggles, <Rubric>, <Apparatus>, <Missing>, <NotYet>
        └── screens/                   # Incipit, Instrument, LayerLeaf, NeuronFolio, Atlas, Landmarks, Nudge, Vault,
                                       # Lexicon, Lineage (+ a folder of parts for each: landmarks/ContrastSection.tsx
                                       # is section vii, one word against another; vault/EditPanel.tsx the edit
                                       # and its voice; atlas/map.ts the atlas's three.js map, atlas/state.ts its
                                       # store, atlas/scientist.ts its chat)
```

---

## 2. Model & Hardware Constants (`google/gemma-4-31B-it`)

| Parameter | Value | Notes |
|---|---|---|
| `d_model` | `5376` | Residual stream hidden dimension |
| `L` (`n_layers`) | `60` | Decoder layers indexed `0..59` (1-based `L1..L60`) |
| `d_ff` (`intermediate_size`) | `21504` | GeGLU FFN width (`43,008` bipolar poles per layer) |
| `vocab_size` | `262144` | Tied embedding / LM head vocabulary size |
| `final_logit_softcapping` | `30.0` | `logits = 30.0 * tanh(logits / 30.0)` |
| `rms_norm_eps` | `1e-6` | Gemma 4 RMSNorm with `(1 + w)` unit-offset scaling |
| `global_layers_1based` | `[6, 12, 18, 24, 30, 36, 42, 48, 54, 60]` | Every 6th layer is Full Global Attention (`head_dim=512`, p-RoPE) |
| `global_layers_0based` | `[5, 11, 17, 23, 29, 35, 41, 47, 53, 59]` | 0-indexed equivalents in `interp-engine` and `J[l]` |
| `sliding_window` | `1024` | Local attention window on the remaining 50 layers (`head_dim=256`) |
| `VRAM Envelope` | `< 30.0 GB` | Single RTX 5090 (32,607 MiB total) |

**The 4-bit weights.** The 410 quantised linears stay packed (int32 words of eight 4-bit pieces, a bf16 scale per 32
weights: compressed-tensors' pack-quantized layout) and are unpacked to bf16 on every forward by
`engine_runtime.dequantize_w4a16_full`, whose file R4 pins by sha256. `backend/w4a16_fast.py` installs a faster unpacker
from outside it, chosen by `--unpacker` (`scripts/census_run.py`, `scripts/atlas_build.py`) or `$THEODOLITE_UNPACKER`
(the server): `lut`, a 256-entry table over the packed bytes and then the same multiply by the scale; `compiled`, the
same arithmetic through `torch.compile`, the unpacker alone, the model left eager with its hooks; `resident`, each weight
unpacked once and kept beside the packed one, overlays applied to a clone as the pinned forward applies them. Each gives
the reference's bits: the pieces minus 8 are exact in bf16, so on every route the one rounding is the multiply by the
scale, an exact float32 product rounded once to nearest even. `scripts/check_dequant.py` shows it on the CPU for every
(piece, bf16 scale) pair (`compiled`: every number; only the NaNs' own bits differ, which no finite checkpoint holds)
and on a 4-bit stand-in's readings. At install `compiled` (and `lut` on CUDA only) is compared with the reference on
every linear, by bits and strides, and on that table; `resident` by each kept weight against a fresh unpacking and each
linear's forward against the pinned one on a fixed two-row probe, by bits; any difference is refused in plain words. A
reading records which ran (`unpacker` in a census run's or an index's meta, and in `/api/health`).

---

## 3. Core Tensor & Precision Contracts

### 3.1 `interp-engine` Hook Addresses
For each layer $l \in \{0, \dots, 59\}$:
- `Address("resid_pre", l)`: Residual stream entering layer $l$, shape `[batch, pos, 5376]` (or `[pos, 5376]` unbatched).
- `Address("resid_mid", l)`: Residual stream after Attention + post-attention sandwich RMSNorm, before MLP, shape `[batch, pos, 5376]`.
- `Address("mlp_act", l)`: Post-GeGLU intermediate activations inside the FFN ($\text{GeLU}_{\tanh}(x W_{\text{gate}}) \odot (x W_{\text{up}})$), shape `[batch, pos, 21504]`.
- `Address("resid_post", l)`: Residual stream exiting layer $l$ after post-FFN sandwich RMSNorm + residual add, shape `[batch, pos, 5376]`.

### 3.2 Pre-Fitted Coordinate Lenses (`neuronpedia/jacobian-lens`)
- **Files**:
  - `gemma-4-31b/jpp/Salesforce-wikitext/gemma-4-31B_jpp_lens.pt` (J++ Lens, primary workspace lens)
  - `gemma-4-31b/jlens/Salesforce-wikitext/gemma-4-31B_jacobian_lens.pt` (Jacobian Lens, comparison lens)
- **Precision Contract**:
  - Stored tensors `J[l]` for $l \in \{0, \dots, 58\}$ are `float16` of shape `[5376, 5376]`.
  - Immediately on load: assert `torch.isfinite(J[l]).all()`, cast to `torch.bfloat16` for matmuls (and `torch.float32` for norms/Gram-Schmidt reductions).
  - For the final layer $l = 59$ (`target_layer = 59`), define $J[59] = I_{5376 \times 5376}$.
- **Readout Contract**:
  $$\text{logits}(h, l) = \text{unembed}\!\left(h \, J[l]^\top\right)$$
  where $\text{unembed}(u) = 30.0 \cdot \tanh\!\left(\frac{\text{LMHead}(\text{FinalRMSNorm}(u))}{30.0}\right)$.

### 3.3 Four Mathematical Modules (`backend/theodolite_math.py`)

1. **Attention Router vs. FFN Vault Causal Vector Split**:
   $$\Delta_{\text{attn}, J}^{(l)} = \left(\text{resid\_mid}[l] - \text{resid\_pre}[l]\right) J_{\text{pp}}[l]^\top \in \mathbb{R}^{5376}$$
   $$\Delta_{\text{ffn}, J}^{(l)} = \left(\text{resid\_post}[l] - \text{resid\_mid}[l]\right) J_{\text{pp}}[l]^\top \in \mathbb{R}^{5376}$$

2. **Bipolar GeGLU Neuron Poles & J-VINDEX Fusion**:
   - Top-$k$ active poles per layer ($k = 15$):
     $$\text{vals}, \text{idx} = \text{topk}\!\left(|\text{mlp\_act}[l]|, k=15, \text{dim}=-1\right), \quad \sigma = \text{sign}\!\left(\text{mlp\_act}[l][\text{idx}]\right) \in \{-1, +1\}$$
   - Ranked by write size instead (REPORT §3.4 bug 8, from E3; `pole_ranking="write"`), the top 15 by
     $$\text{score}_i = |\text{mlp\_act}[l][i]| \cdot \big\|W_{\text{down}}[l][:, i] \odot w_{\text{post}}\big\|$$
     with $w_{\text{post}}$ the post-FFN norm weight; each listed pole's `write_size` is $s \cdot \text{score}_i / \text{rms}(y)$,
     and `poles_ffn_explained` is the share of the feed-forward write the fifteen explain, sent under either ranking.
   - Down-projection vector $w_{\text{down}}^{(l, i)} = W_{\text{down}}[l][:, i] \in \mathbb{R}^{5376}$ (including post-FFN norm weight scaling when projecting into residual space).
   - **J-VINDEX Readout**:
     $$\text{pole\_logits}(l, i, \sigma) = \text{unembed}\!\left((\sigma \cdot w_{\text{down}}^{(l, i)}) \, J_{\text{pp}}[l]^\top\right)$$
   - **Naive Logit Lens Baseline**:
     $$\text{naive\_pole\_logits}(l, i, \sigma) = \text{unembed}\!\left(\sigma \cdot w_{\text{down}}^{(l, i)}\right)$$

3. **Live Downstream Reader Circuit Filaments (Spillover Filter)**:
   - Between active pole $(l, i, \sigma_i)$ at layer $l$ and firing poles $(l', j, \sigma_j)$ at downstream layers $l' \in \{l+1, \dots, \min(l+6, 59)\}$:
     $$C_{(l, i) \to (l', j)} = \left\langle \sigma_i \cdot w_{\text{down}}^{(l, i)},\, w_{\text{gate}}^{(l', j)} \right\rangle \cdot \left|\text{mlp\_act}[l', j]\right|$$
     where $w_{\text{gate}}^{(l', j)} = W_{\text{gate}}[l'][j, :] \in \mathbb{R}^{5376}$ (incorporating pre-FFN RMSNorm weights).
   - **Direct-path estimate** (V7 bug 4; `direct_path_source_write` / `direct_path_reader_effects`), from the clean forward's
     $y_l$ (`mlp_out`) and $\text{mid}_{l'}$ (`resid_mid`):
     $$D_{(l,i)\to(l',j)} = \Big\langle J_{\text{pre}}(\text{mid}_{l'})\, \big(\textstyle\prod_{l<k<l'} s_k\big)\, s_l\, J_{\text{post}}(y_l)\, a_i w_{\text{down}}^{(l,i)},\; W_{\text{gate}}[l'][j,:] \Big\rangle$$
     with exact RMSNorm Jacobians; `direct_act` adds $W_{\text{up}}$ and the clean pre-activations. Selected with
     `thread_score="direct"`.

4. **Projective Theodolite Triangulation (Survey Frame, Gram–Schmidt or Löwdin)**:
   - Orthonormal basis $(e_1, e_2, e_3) \in \mathbb{R}^{3 \times 5376}$ in J++-aligned final residual space (or layer-pulled-back space):
     - **Monument Mode**: Given 3 vocabulary tokens $(t_1, t_2, t_3)$, take their unembedding directions $u_m = W_U[t_m]$ (or layer-pulled-back $v_m^{(l)} = W_U[t_m] J_{\text{pp}}[l]$) and orthonormalize in `fp32` via modified Gram-Schmidt to obtain $(e_1, e_2, e_3)$.
     - **Pole-Locked Motif Mode**: Given a selected pole $(l, i, \sigma_i)$, set raw vectors $v_1 = w_{\text{gate}}^{(l, i)}$, $v_2 = (\sigma_i w_{\text{down}}^{(l, i)}) J_{\text{pp}}[l]^\top$, $v_3 = w_{\text{gate}}^{(l+1, j^*)}$ (dominant downstream reader), and orthonormalize via Gram-Schmidt (or Löwdin, if `frame_squaring` asks for it).
   - Project each layer's J++-aligned state $\tilde{h}_l = h_l J_{\text{pp}}[l]^\top$:
     $$x_l = \langle \tilde{h}_l, e_1 \rangle, \quad y_l = \langle \tilde{h}_l, e_2 \rangle, \quad z_l = \langle \tilde{h}_l, e_3 \rangle$$
     $$\theta_l = \text{atan2}(y_l, x_l), \quad \phi_l = \text{asin}\!\left(\frac{z_l}{\sqrt{x_l^2 + y_l^2 + z_l^2 + \epsilon}}\right), \quad \|h_l\|_2 = \sqrt{\sum_{d=1}^{5376} h_{l,d}^2}$$
   - Options (V7 bug 3): `frame_squaring="lowdin"` squares symmetrically, $E = (VV^\top)^{-1/2}V$ on unit rows (order-free);
     `frame_fnw` uses $\text{fnw} \odot W_U[t_m]$; `frame_per_layer` reads the raw $h_l$ in the frame of $W_U[t_m] J_{\text{pp}}[l]$;
     `centre_prompts` subtracts a matched control's state before projecting. `proj_frac` $= \|(x_l, y_l, z_l)\| / \|\tilde{h}_l\|$.
   - Whose defaults are whose: the server's defaults (Gram–Schmidt on raw rows; poles by $|a|$; threads by $C$) reproduce
     the instrument as first built, for old callers. The interface sends `frame_squaring="lowdin"` on raw rows
     (`frame_fnw=false`, since V7 did not measure the weighted rows), `pole_ranking="write"` and `thread_score="direct"`
     unless the reader chooses otherwise; in pole-locked motif mode it sends `frame_squaring="gram_schmidt"`, since that
     frame's axes have fixed roles.

Beside the four, one word against another (V7 bug 3's contrastive axes, read one at a time rather than as a frame;
E3 (c); requested as a sweep's `contrasts`, 4.1–4.2):
   - `contrast_of_state(state, toward_ids, away_ids, fnw, W_U, softcap=30)`, for pairs $(a_i, b_i)$, $i = 1 \dots n$:
     $$s(u) = \frac{1}{n}\sum_i \Big[\text{cap}\big\langle \text{n}(u), W_U[a_i] \big\rangle - \text{cap}\big\langle \text{n}(u), W_U[b_i] \big\rangle\Big], \quad \text{n}(u) = \frac{u}{\text{rms}(u)} \odot \text{fnw}, \quad \text{cap}(z) = 30\tanh(z/30)$$
     in fp32 on the $2n$ pair rows only, with the unembed's own norm (eps 1e-6, or 1e-12 where mean $u^2$ < 1e-3, as
     `gemma4_rms_norm`); through the lens $u = h_l J_{\text{pp}}[l]^\top$, by the naked eye $u = h_l$. For one pair it
     is the log-odds of $a$ against $b$ under that distribution (nats); for the six pronoun pairs, labkit's
     `gender_scores(...)["pron"]`.
   - `contrast_direction`: $d = \text{fnw} \odot \frac{1}{n}\sum_i (W_U[a_i] - W_U[b_i])$ (labkit's `gender_direction`);
     `contrast_mean_part(μJ, hJ, …)` $= \langle \mu_l J_l^\top, d \rangle / \text{rms}(h_l J_l^\top)$, softcap-free: the
     average state's linear share of the score (E3_lens_validation_09_background.py).
   - `CONTRAST_BACKGROUNDS = ("controls", "study_mean")`. Contrasts of three landmark words lie in the plane of their
     differences, inside the landmark frame's span; read one at a time, in log-odds, they need no squaring.

---

## 4. Binary WebSocket + JSON Metadata Wire Protocol (`backend/server.py`)

Each sweep over WebSocket `/ws/theodolite` transmits:
1. **Binary Frame (`ArrayBuffer`, `Float32Array`)** with deterministic header + packed float32 buffers:
   - `magic` (`0x5448454F` = `"THEO"`), `version` (`1`), `n_layers` (`60`), `n_poles_per_layer` (`15`), `n_filaments` (`M`), `flags`
   - `traj_xyz`: `[60, 3]` (`x_l, y_l, z_l` in 3D Survey Frame)
   - `traj_angles_norm`: `[60, 3]` (`theta_l, phi_l, ||h_l||_2`)
   - `delta_attn_xyz`: `[60, 3]` (Attention router vector projected into 3D Survey Frame + norm)
   - `delta_ffn_xyz`: `[60, 3]` (FFN vault vector projected into 3D Survey Frame + norm)
   - `cka_matrix`: `[60, 60]` (Linear CKA similarity matrix across all 60 layers)
   - `poles_data`: `[60, 15, 6]` (`neuron_idx, polarity, activation, dir_x, dir_y, dir_z`)
   - `filaments_data`: `[M, 7]` (`src_layer, src_rank, dst_layer, dst_rank, coupling_score, src_polarity, dst_polarity`)
2. **JSON Frame** carrying human-readable token labels:
   - Top-5 J++ Ghost Token Plumes per layer (`token_str`, `prob`, `logit`)
   - Top J-VINDEX vs. Naive Logit Lens token labels for active Bipolar Poles
   - Generated continuation text, the thinking vs. silent comparison (`think_protocol` names how thinking was switched
     on, 4.2), and active LARQL `.patch` / `SteeringSpec` status.
   - Added without changing the binary layout: `prompt` (the exact text read), `prompt_token_ids`, `chat`, `system`,
     `user_prompt`, `assistant_prefix`, `pole_ranking`, `thread_score`, `frame` (squaring, per-layer, weighting, centring,
     word geometry), `steering_applied`; per layer `proj_frac`, `pole_write_sizes`, `poles_ffn_explained`; per filament
     `direct_gate`, `direct_act`, `score`, `score_kind` (binary column 4 carries `score`, which is C unless
     `thread_score="direct"`).

**Who may ask.** Every route of the server, its pages and `/ws/theodolite` answer only the instrument's own pages,
served from this machine, and scripts run on it (`backend/local_only.py`: `LocalOnly`, one ASGI middleware around the
whole app, outside CORS, applying the atlas's `local_pages_only`, §4.7). The peer must be loopback; the Host must be
127.0.0.1 / localhost / [::1] on any port (or absent), and so must the Origin when there is one; a request marked
`Sec-Fetch-Site: cross-site` or `same-site` is refused unless it is a GET navigation (a link). A refused request gets
403 in plain words; a refused socket is closed with 1008 before it is accepted, which uvicorn sends as an HTTP 403 to
the handshake. CORS (`allow_origin_regex`) lets only those origins read a reply, and governs no WebSocket: the guard is
what covers the socket, the older routes (`/api/larql`, `/api/scientist/*`), the docs and the static pages. Vite's dev
proxy passes (`/api` with the server's Host, `/ws` with its own, 127.0.0.1:5173; both with the page's own Origin).
On Linux the guard also asks which account made the connection, and refuses another's (403, or 1008 on the socket;
`other_account`; root's connections pass, since root can act as any account already and a container's port-forwarder
connects as root): it finds the client's end of the connection in /proc/net/tcp or tcp6 (local end the client's address
and port, remote end the server's) and reads its owner from a line some process still holds (inode not 0), afresh for
every request and socket, keeping nothing (under 1 ms over some 30 to 110 sockets here, about 2 ms over 1,000 and 7 ms
over 4,500). A request whose sender has closed or reset its end (a line left that no process holds, printed as root's,
or no line, or no client named at all) is refused when the tables list the server's own listening port, as no live
caller on this machine is missing from them: before this, a uid-65534 process that sent a request and reset its socket
at once got every such request run, and one that closed it got through while the server was busy. A connection it
cannot place (not Linux, tables that do not list the server's port, a Unix socket) passes, and the log says so once; a
request carrying `X-Forwarded-For` is refused, since uvicorn takes the client's address and port from that header on a
loopback connection; an in-process client that names itself (TestClient, the openai backend's calls) has no socket and
passes. Vite's dev proxy runs as the person, so its connections pass, and so does what another account sends through it
while `npm run dev` runs (tried: a uid-65534 client through it was answered); the check guards the server's port only.
`THEODOLITE_PEER_CHECK=off` turns this off, and each of its refusals says so. The MCP servers
(`backend/scientist_mcp.py`, and `backend/mcp_scientist.py` run alone, on FastMCP's default stdio transport) listen on
no port. `scripts/check_local_only.py` checks every route and the socket.

### 4.1 A sighting of every word-piece (request fields, all optional)

A sweep writes an answer of n word-pieces w_0 … w_{n-1} after the prompt's P (ids 0..P-1), then reads prompt +
w_0..w_{n-2} back in one forward, inside the same steering block as the writing. Token index k (0..n-1) is read at
`read_pos = P - 1 + k`: the stream that has just read `read_token` (the prompt's last piece for k = 0, else w_{k-1})
and whose layer 60 chose `chosen` = w_k. The last piece is chosen but never read; with nothing written there is one
position, P-1, and `chosen` is null. In the runtime this is `capture_for_sweep` (write and read back once) +
`prepare_sweep` (tracked words, the frame, the controls) + `analyse_position(k)`; `run_theodolite_sweep` is the three
at k = 0 and gives an old caller the same numbers bit for bit (scripts/check_sweep_refactor.py checks it on the GPU).

| field | default | meaning |
|---|---|---|
| `request_id` | null | client's id (≤ 64 chars), echoed on every frame; with it the reply is a stream (4.2) |
| `sampler` | greedy | `{temperature 0–2 (0 = argmax), top_k 0–262144 (0 = off), top_p (0, 1] (1 = off), presence_penalty −2–2, seed 0–2^53−1 or null}`; every knob is passed explicitly (interp_engine reads a missing one as the checkpoint's T 1, top-k 64, top-p 0.95); the penalty is subtracted from pieces already written, not the prompt's; a sampled request without a seed gets one drawn and echoed |
| `max_new_tokens` | 6 | 0–256 |
| `stop_at_end_of_turn` | true | stop after the checkpoint's end marks (generation_config.eos_token_id: 1 `<eos>`, 106 `<turn|>`, 50 `<|tool_response>`); false: interp_engine's rule, `<eos>` only |
| `sight_every_token` | false | one pair per written piece, the first `max_sighted` (1–64, default 64) |
| `sight_tokens` | null | these token indices, in this order (≤ 64; repeats dropped; indices outside 0..max(n,1)−1 dropped and echoed in `sight_dropped`) |
| `answer_token_ids` | null | a given answer (≤ 256 ids): nothing is written; the ids are read back teacher-forced (a retake), their odds taken from the re-read's layer 60 |
| `rank_lookahead` | true | the old space-then-digit rule: where the chosen piece is blank, a tracked word's J++ rank is taken from the next position when better there (`tracked_from_next` says where); the new pages send false |
| `centre_token_ids` | [] | matched controls as exact ids (≤ 16 lists, each non-empty), beside `centre_prompts`' texts |
| `centre_answer` | false | centre token k ≥ 1 on the controls reading this very answer (control ids + w_0..w_{n-2}, row len(control)−1+k); without it a position past token 0 is not centred at all |
| `think_bare` | "prefix" | with `compare_think` and bare text: "prefix" sets `<|think|>\n` before the text, as first built (not Gemma 4's switch; E6 §3, item 11); "none" leaves the thinking comparison out (`think_left_out` "bare_text"). A chat always uses the template's own switch (`compose_prompt(thinking=True)`: `enable_thinking`, `<|think|>` at the head of a system turn the template opens if there is none, and the model's turn begun with no empty thought; scripts/check_think_template.py checks the render). The interface sends "none" |
| `contrasts` | [] | one word against another (≤ 4): `{toward: [words], away: [words], label}`, 1–12 words a side, as many of each (they pair by position, as the study's pronoun pairs); each word read as the landmarks read theirs (" word" first, then the word) but only as one word-piece. Read at every layer of every sighting (below); none: the runtime is called exactly as before and `contrasts` is [] |
| `contrast_backgrounds` | [] | what each contrast is read against (≤ 2, once each): "controls" (the matched controls sent, read where the centring reads them) \| "study_mean" (E3's average state, experiments/results/E3_background_mu.npy, sha256 pinned in `STUDY_MEAN_SHA256`) |

Which indices are sighted: `sight_tokens` if given, else 0..min(n, max_sighted)−1 with `sight_every_token`, else
[0]; never empty ([0] when nothing asked for exists). `prompt`, `system` and `assistant_prefix` are at most 32000
characters, and P + max(max_new_tokens, len(answer_token_ids)) must stay within `MAX_SWEEP_POSITIONS` = 2048 (the
recapture keeps resid_post at every row and the other four points at the read rows only, moved to the CPU as each
hook fires, so what grows with P is the writing's full-length logits and kinship's float32 states; the runtime sets
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True unless PYTORCH_ALLOC_CONF, PYTORCH_CUDA_ALLOC_CONF or
PYTORCH_HIP_ALLOC_CONF is already set, which is then kept whole; in the worst sweep the segments lower the card's peak
by 2.2 GiB at 1024, 1.9 at 1536 and at least 1.36 at 2048, where the default allocator filled the card;
`GET /api/health` sends the allocator's settings as `allocator`, and the server warns at load when the segments are
off; the measurements are in the constant's comment; the thinking prompt is held to the same).
Every sighting's metadata sends `sliding_window` (the model's, from `text_config.sliding_window`: 1024) beside
`read_pos`, and `GET /api/health` sends it too: from read position 1024 on, the 50 sliding-window layers no longer
look back to the start of the text. Neither the lens's fit (positions 16-127 of 128-piece passages) nor the study's
checks (short texts: at most 141 word-pieces in its fidelity check) reached that far, and the lens card's caution
says so.
A refused id, control or length is answered in plain words, and so is a `pole_ranking`, `thread_score`,
`frame_squaring` or `think_bare` the analysis does not know, before anything is written; so are a contrast background
it does not know, "controls" with no matched control sent, "study_mean" when the average state is missing or not the
file pinned, and a contrast whose words do not pair or are blank. A word that is no single word-piece does not refuse
the sweep: that contrast comes back with its `error` and `pieces`.

### 4.2 Messages, server → client

Without `request_id`: exactly one (binary, JSON) pair, at the first index of the sight order (`sight_tokens[0]` when
given, else token 0), as before (with the fields below added to the JSON), and nothing else. With one, in this order:
1. `{"type":"theodolite_progress","request_id","phase":"writing","done":i,"of":max_new_tokens}` after each written
   piece.
2. `{"type":"theodolite_written","request_id","answer","generation","sight_order":[k…],"n_to_sight":m,
   "prompt_token_ids","prompt_len"}` once, after the read-back and before the frame is built (so a failure from there
   on cannot lose the answer), then `{"type":"theodolite_progress",… "phase":"sighting","done":0,"of":m}`.
3. One pair per sighted index, in sight order: the binary frame, then its `theodolite_metadata` JSON, back to back.
   Layout and `WIRE_VERSION` 1 unchanged; flags bit 0 (thinking path) on the first pair only, bit 1 steered, bit 2 (4)
   on every pair of a request that sights more than one index.
4. `{"type":"theodolite_sweep_end","request_id","n_sent","n_to_sight","cancelled","error"}`, always last.

Added to every pair's JSON (the one-pair reply too; `generated_text` is the whole answer, `prompt*` the prompt only):
`request_id, token_index, sight_index, n_to_sight, last, read_pos, sliding_window, read_token, read_token_id,
prompt_len, chosen (STEP | null), answer, generation, recapture_check, kinship_n_states (= read_pos + 1),
rank_lookahead, sight_dropped, timing {generate_ms, capture_ms, analyse_ms, total_ms}, think_seed, think_read_token,
think_stop_reason, think_skipped, think_protocol, think_left_out` (the last six, and `think_generated_text`, set on the
first pair only: `think_stop_reason` is how the thinking answer's writing ended, "end_mark" | "length" | "stopped",
null without one; `think_skipped` is true when a stop asked before the comparison began left it unwritten, else false;
`think_protocol` is how thinking was switched on, "template" (a chat) | "bare_prefix" (the imitation), null when none
was written; `think_left_out` is "bare_text" when `compare_think` was asked with bare text and `think_bare` "none",
else null);
`frame.centre_kind` ("none" | "control_prompt_end" | "control_reading_same_answer") and, for a pole-motif frame,
`frame.reader {layer, neuron, from_token_index}` (built at the first position analysed and kept for the whole answer);
per layer `jpp_top_ids` (5), `naive_top_id`, `tracked_from_next {word: bool}`.
- STEP = `{index, token_id, token, model_prob, model_logprob, model_rank, n_tied, chosen_logit, top:[{token_id, token,
  prob}]×5, sampler_prob, sampler_support, end_mark, source}`: the model's own odds at that step (float32 softmax of
  the writing's raw logits; a given piece: of the re-read's layer 60) and the sampler's chance for it, replayed on the
  same logits op for op (`sampler_chances`; null when greedy; scripts/check_sampler_mirror.py).
- ANSWER = `{token_ids, token_strs (each id decoded alone), text, n_written, stop_reason ("end_mark" | "length" |
  "stopped" | "given" | "none"), stop_token_ids, source ("written" | "given"), steps: [STEP]}`.
- GENERATION = `{greedy, settings {temperature, top_k, top_p, presence_penalty} as resolved and applied (top_k / top_p
  null when off or greedy), seed, seed_drawn, max_new_tokens, sight_every_token, max_sighted, capped,
  stop_at_end_of_turn}`.
- `recapture_check` = `{top1_id, top1_token, top1_is_chosen, chosen_logit_recapture, chosen_logit_diff}` at layer 60
  (J = I): the reading forward against the writing one, which agree to bf16 rounding. For a given answer
  (`chosen.source` "given") `chosen_logit_recapture` and `chosen_logit_diff` are null: its odds are the re-read's own
  layer 60, so nothing was written to compare with (a diff would be 0 by construction); `top1_is_chosen` still
  compares the re-read's top piece with the given one.

Neuron labels and the label gate (`backend/label_gate.py`; REPORT §3.3, conditions 2 and 3; erratum E47). Each added;
old fields unchanged. Layers in these fields count from 0.
- `poles_labels[l][r]`, for ranks r = 0–4 of layer l: `{rank, neuron_idx, polarity ("+" | "-"), abs_act, write_size
  (null where the layer's feed-forward write was not captured), j_vindex_tokens, naive_tokens}` as before, the token lists three a side
  (`{token, logit}`), filled at the global layers and at 54–59 (every layer with `decode_all_pole_tokens`), empty
  elsewhere; added `label_check`, null where no words were read.
- `label_check` = `{shared_top10, held_back, reasons, band, sampled_layer, in_measured_band, lens_validated, edited}`:
  `shared_top10` (0–10, null when a reading is missing) counts the word-piece ids both readings share in their first
  ten, scored as N1 scored them (the server's bf16 logits shortlist 64 ids, scored again in fp32 and softcapped, ties to
  the lower id); `reasons` ⊆ {"below_layer_40", "shares_nothing", "writes_nothing"}, `held_back` = any reason.
  "below_layer_40" holds below layer 40; "shares_nothing" holds where `shared_top10` is 0, below layer 59, and only
  under the base-fitted J++ lens (`lens.id` "jpp-base"), the one the gate was validated for (edited neurons included);
  "writes_nothing" holds where the neuron's write vector is exactly zero (a vault DELETE in force), and then
  `shared_top10` is null, since every word-piece ties. `band` is "below_sampled" (0–9),
  "chance_where_sampled" (10–30), "not_sampled" (31–39), "band" (40–57), "above_band" (58) or "one_reading" (59, where
  J++ is the identity). `edited` is true when a vault edit is in force on the neuron (`edits_in_force()`, the latest
  record per neuron), and then `lens_validated` is false. The words themselves are still sent: `held_back` marks what
  the pages leave out of their main reading.
- `label_gate` (`LABEL_GATE_CARD`: `rule` "n1-gate-1", `layers_count_from`, `min_layer`, `measure`, `cut`,
  `held_back_if_shared_top10_at_most`, `validated` (the count's AUC, as a graded score), `cut_record` (the cut's own
  record on N1's sample: held back, kept, how many of each sound, its AUC as a yes/no sign; from
  experiments/results/N1C_gate_cut.json), `sources`, `caution`) rides just before `lens` in every pair's JSON and in
  `/api/health`.
- The scientist tools and the vault, over MCP, HTTP and the WebSocket alike: `inspect_pole` and DESCRIBE POLE add
  `label_check` to each face (`north_pole`, `south_pole`; `active_pole` is the same face) and `label_gate` before
  `lens`, so their keys match; each NEAREST row adds `label_check` (its five-word lists unchanged), and the reply
  `label_gate` before `lens` (UPDATE, INSERT, DELETE and ROLLBACK reply as before but for an edit's voice; DESCRIBE
  adds `edit_method`, and every patch record adds `voice`, `edit_kind` and `replaces_patch`; 4.6);
  `trace_downstream_readers` (`trace_readers`) adds to each reader returned `dst_naive_tokens` (four, as through the
  lens) and `dst_label_check`, and `label_gate` to the reply; `test_motif_hypothesis` adds `pole_label_check` (its
  pole's face) and `label_gate`.
- `test_motif_hypothesis` (`POST /api/scientist/test_hypothesis`) adds `selectivity` after the means: on the last
  word-piece's unrounded readings, `n_pos`, `n_neg`, the means, sample standard deviations, `cohens_d` over the pooled
  sample variance, `welch_t` and `welch_df`, a two-sided `p` (`p_method` "exact" over every split of the sentences up
  to 10,000 of them, else "permutation", 10,000 shuffles under `seed` 0) and `p_text` (that p as `words` writes it),
  `min_n` (4) and `alpha` (0.05), its own `verdict` (SELECTIVE_NORTH_POLE (+), SELECTIVE_SOUTH_POLE (-),
  DIFFERS_NOT_AS_GUESSED, NOT_SHOWN, TOO_FEW_TO_JUDGE below four a side, CANNOT_TEST when a reading is not a finite
  number), `face` and `words`. `verdict` and `cohens_d` stay as first built (population variances, a threshold at any
  n), for old callers; the folio shows `selectivity`. An empty side, or a layer or neuron that does not exist, is
  refused in plain words (400), where it was read as 0.0 or moved to the nearest layer.

Contrasts (one word against another; V7 bug 3's contrastive axes, E3 (c)'s background; backend/theodolite_math.py
`contrast_of_state`, engine_runtime `_contrast_readings`; scripts/check_contrast_readings.py). Every pair's JSON adds
`contrasts`, after `frame`: [] without any asked for, else one entry per contrast, in order, `{label, toward, away
(each word as read: {word, id, piece}, or id and piece null and `pieces` {form tried: [word-pieces]}), toward_ids,
away_ids, error}`. A contrast that cannot be read has its `error` in plain words, `pieces` by word, and nothing more.
One that can adds, per layer (0-based, 60 values, 4 decimals):
- `lens`, `naked`: the mean over the pairs of capped logit(toward_i) − capped logit(away_i) of h_l J_l^T (a bf16
  product, as labkit's lens) and of h_l, scored in fp32 on the pair rows only with the unembed's own norm and softcap (nats of log-odds; for the six pronoun pairs,
  labkit's "pron"). At index 59 the lens is the identity, so the two agree.
- `backgrounds`, one key per background asked for:
  - "controls" = `{kind, n_controls, control_lens, control_naked, delta_lens, delta_naked, why}`: the controls' mean
    score and the sighting's score less it, a difference of two log-odds (E3 recommendation 1), read where the
    centring reads the controls: `kind` "control_prompt_end" at token 0 (their own last piece, from the centre's own
    captures), "control_reading_same_answer" past it with `centre_answer` (from the same captures as that centre),
    else "none" with the series null and `why`.
  - "study_mean" = `{kind, centred_lens, centred_naked, mean_part_lens, mean_part_naked, file, sha256}`: the score of
    h − μ (through the lens (h − μ) J_l^T), as E3 centred it, which is not a log-odds the model gives; and the mean
    part <μJ, d>/rms(hJ) (by the naked eye <μ, d>/rms(h)), d = fnw ⊙ mean(W_U[toward] − W_U[away]), softcap-free. μ
    was read by E3 through HF decoder-layer hooks (labkit.forward_resid), over every word-piece from the fifth on of
    its prose passages and chats. scripts/verify_e2e_ui_and_mcp.py (item 12) checks it is in the coordinates of the
    server's resid_post: on one of E3's neutral passages the two captures agree at every layer, and E3's neutral-prose
    figures at L40–49 come back through `contrast_of_state`; each GPU run writes what it measured to
    experiments/results/E3_background_mu_check.json, by layer.
`/api/health` adds `contrasts` = `{backgrounds, max_contrasts, max_words, study_mean_problem}` before `label_gate`
(`study_mean_problem` null when the average state can be read, else why, in plain words). The thinking comparison is
read without contrasts.
On the pages (`contrastFields` in the store): the request always carries both keys, empty unless the landmarks page's
section vii is on with words typed on both sides of each of its rows, and they pair; then it carries a contrast for
each row, in order (several words a side, by commas), at most `/api/health`'s `max_contrasts` (four): the first row is
the settings' `toward` and `away`, as when the section sent one, the rest `more`. More rows than that, or a row that
does not pair, sends none, and the page says why, naming the row ("the second pair: …"). The backgrounds are
checkboxes, either or both, sent in `CONTRAST_BACKGROUNDS`' order, dropping "controls" when no matched control goes
with the request and "study_mean" when `/api/health` gives a `study_mean_problem`, so a sighting is never refused for
them; a box ticked that can no longer be sent can still be unticked. Ticking the matched control also centres the
frame, since the control goes with the request as `centre_token_ids`. The chart draws one contrast at a time, picked
by a pill for each the sighting read (the first it could read until one is picked; one it could not read shows its
error), and each background it was read against as a line of its own (against the control solid, centred on the
average state dashed), with the bands as before. Both keys are reading keys: changing them reads the answer shown again
from its ids (`answer_token_ids`), nothing written again.
`Sighting.contrasts` is null when the server sent none (the recorded sighting, an older server) and [] when none was
asked for.

The causal spot-check (§4.5): the WebSocket action `causal_check` is answered `{"type":"causal_check","data":REPLY}`,
REPLY as `POST /api/scientist/causal_check` and the MCP tool `causal_spot_check` return it. `/api/health` adds
`causal_check` = `{version: 1, writes: ["activation", "frozen_norm"], contexts: ["given", "study"], scale_refs:
["study", "this_text"], k_default: 10, k_max: 200, k_floor: 0.01, n_study_texts: 36}` before `label_gate`: an older
server has no such card, and ignores the action without a word, so a page offers the test only when the card is there.

Errors: a request with a `request_id` that is refused or fails gets `{"type":"error","action":"sweep","request_id",
"recoverable":true,"message","n_sent","token_index"}` and then its `theodolite_sweep_end`; the socket stays open
(the request_id of a refused payload is read from the raw message). Without one, the old refusal
`{"type":"error","message","action","recoverable","layer","neuron_idx","contexts"}`; `contexts` (added) is the
request's own `contexts` as sent, so a page can fail the right one of a neuron's two tests (null for the other
actions). A refused or failed `causal_check` (a bad option, a CausalCheckRequest that does not parse, the GPU out of
memory) comes back this way, in plain words, and the socket stays open.

### 4.3 Cancel, client → server

`{"action":"cancel_sweep","request_id":R}`. A receiver task reads the socket into a queue and handles a cancel at
once: a running request stops at its next check (after each written piece: the writing stops, `stop_reason`
"stopped", and only the first index of the sight order is sighted; between pairs: after the current pair), ending
`cancelled: true`; a queued one is dropped and its end (`n_sent` 0) sent at once. Other actions keep their order.
`_GPU_LOCK` is held for the whole stream (writing, read-back, every position, the thinking comparison), so no vault
edit lands between two positions of one answer. The thinking comparison is written with the same sampler and seed
(length max(min(max_new_tokens, 64), 10)), read at its token 0 in the main sweep's frame and centre, and rides on the
first pair. It can be stopped too: a stop asked before it begins (while the answer is written, or while the first
position is read) skips it (`think_skipped` true, no thinking path); one asked while it is written ends it after the
current piece (`think_stop_reason` "stopped"); either way the request ends `cancelled: true`, with one sighting too.

### 4.4 Exact ids for `trace_readers`, and saved prompts

- `trace_readers` (WebSocket, `POST /api/scientist/trace_readers`, MCP `trace_downstream_readers`) takes
  `token_ids` (1 to `MAX_SWEEP_POSITIONS` = 2048 ids): read exactly as given, at the last (a sighting's ids up to its
  read position), echoing `token_ids_given`, `read_pos` and `prompt` (the ids decoded); every reply says
  `steered: false` (a trace reads the model without a nudge).
- Saved questions and system prompts (`backend/saved_prompts.py`, declared above the static mount): `GET
  /api/saved_prompts` → `{questions, systems, file, warning}`; `POST` `{kind: "question" | "system", name, question?,
  answer_begins?, chat?, text?, replace_id?}` → the lists (409 for a name already used in that list, trimmed and in any
  case); `DELETE /api/saved_prompts/{kind}/{id}` → the lists (404 when absent). One file outside every checkout
  (`$THEODOLITE_SAVED_PROMPTS`, else `${XDG_DATA_HOME:-~/.local/share}/theodolite/saved_prompts.json`), written
  atomically under its own lock (never the GPU's); an unreadable file is moved aside to `<file>.corrupt-<UTC stamp>` and
  reported once (a file that is not UTF-8, is nested too deeply, or holds a lone surrogate counts as unreadable).
  Requests from a page on another host get 403 (from the server's guard, §4, and from these routes' own rule): the Host
  must be 127.0.0.1 / localhost / [::1] on any port (a DNS-rebinding page sends its own name as the Host, and a
  same-origin GET sends no Origin; vite's proxy sends the server's own Host), and so must the Origin when there is one;
  no Origin (a script) is allowed. A name or text holding half of a character (a lone surrogate) is refused with 422.

### 4.5 The causal spot-check (`backend/causal_check.py`)

WebSocket `{"action":"causal_check", …}`, `POST /api/scientist/causal_check` (400 for a plain-word refusal, 422 when
the body does not parse) and the MCP tool `causal_spot_check` take the same fields and give the same reply, under the
GPU lock. A neuron's label is tested by the study's N1A protocol (experiments/scripts/N1A_02_causal_core.py:3-8,
79-120) at erratum E47's k = 10. It is a new request, so its defaults are the mended ones.

Request (CausalCheckRequest: types only; every bound is checked by the tool, in plain words):

| field | default | meaning |
| --- | --- | --- |
| `layer`, `neuron_idx` | required | whole numbers, counting from 0 (layer 0–59, neuron 0–21503) |
| `polarity` | null | the face put first in the reply's `polarity`; null: the face it fired with at the read position (given), else + |
| `contexts` | "given" | "given": one text, read at its last word-piece; "study": the study's 36 texts, each at its last |
| `token_ids` | null | given: 2–2048 exact ids (a sighting's ids up to its read position), pushed at the last |
| `prompt`, `chat`, `system`, `assistant_prefix` | null, false, null, "" | given, without `token_ids`: composed as `trace_readers` composes them |
| `k` | 10 | the push is k · max\|a\|; 0 < k ≤ 200, and k > 0.01 with the floor on |
| `write` | "activation" | "activation" \| "frozen_norm" |
| `scale_ref` | "study" | max\|a\| over the study's 36 texts (cached per set of vault edits in force) \| "this_text" (given only) |
| `floor` | true | also push at k = 0.01, N1A's bf16 floor |
| `top_n` | 10 | 1–50 word-pieces that rose and fell most, per face |
| `track_ids` | null | at most 16 ids followed on both faces |

Computation:
- max\|a\| = max \|mlp_act[l, ·, i]\| over the study's 36 texts (N1A_common.py:19-60: 12 chats through the chat
  template, thinking off, with their begun answers; 12 passages of prose and 12 of code as bare text), every word-piece
  after the start mark, from one forward with a pre-hook on every down_proj; or over the given text. δ± = ±k · max\|a\|,
  and ±0.01 · max\|a\| for the floor.
- Rows are slot-major: base, +k, −k, +0.01, −0.01; the base sits in the same forward as the pushes (rows of one
  forward are bit-identical; different batch shapes were not, KL up to 0.014).
- write "activation": mlp_act[l, p, i] += δ (a pre-hook on down_proj, at the read position p only). write
  "frozen_norm": the post-feed-forward norm's output at p gains δ · (w_post ⊙ W_down[:, i]) / rms(y_p), with y_p that
  norm's input (the layer's feed-forward output) and rms over its eps, in fp32 (N1V_02_causal.py:73-83). W_down[:, i]
  is the column as the model now uses it (`slice_w_down`, vault edits included); w_post the norm's own weight.
- Given mode: the text before p runs once, batch 1, with a cache; then only p runs, for every row, over that shared
  prefix (a push at p changes nothing before it). Study mode: one right-padded forward of every row.
- Read-out: fp32 softcapped logits at p (fp32 GEMM on the final-normed state), c = log_softmax(pushed) −
  log_softmax(base) over the whole vocabulary; in study mode c is the mean over the texts.
- On running out of memory, the pushed slots per forward halve, down to one, each forward with its own base row
  (`forwards`, `slots_per_forward`, `base_rows_identical`); a forward of the base and one push that still fails is
  refused in plain words.
- The check reads the model without any nudge and with any vault edits in force.

Reply keys, in order: `layer, neuron_idx, polarity, contexts, n_contexts, prompt (the text read; null in study mode),
chat, token_ids_given, read_pos | read_pos_by_context, read_token_id, read_token, study_texts (the texts' source),
steered (false), write, k, k_floor (null with the floor off), scale_ref, max_abs_act, max_abs_act_over,
max_abs_act_cached, act_here | act_here_by_context (the neuron's own activation at p, unpushed), faces, antipodal,
floor, base_top, forwards, slots_per_forward, base_rows_identical, vault_edits (patch ids in force), timing_ms
{scale_ms, check_ms, total_ms}, label_gate, lens`.
- `faces["+"]`, `faces["-"]`: `{delta, kl_mean, kl_median, kl_by_context, kl_over_floor (this KL over the floor's of
  the same sign; null when the floor's is not above 0), split_half_rho (even against odd texts; null in given mode),
  rises, falls, tracked ([WORD]), floor_top100_overlap (how many of the hundred ids this push raised most the floor's
  raised too), linear, label_overlap}`. WORD = `{token_id, token (decoded alone), dlogprob (c), base_logprob,
  steered_logprob}` (means over the texts in study mode).
- `linear` (closed form from the base y at p, N1V_audit_01_postnorm.py:90-104): `write_over_ffn` (‖δ W_down[:, i]‖ /
  ‖y‖), `linearity` (‖pn(y + δw) − pn(y)‖ / ‖w_post ⊙ δw / rms(y)‖: 1 is a straight push), `ablation_cos` (cos of the
  change with −pn(y)), `linearity_q10`, each the median over the texts (the one text in given mode), and `by_context`
  lists. For write "frozen_norm", `linearity` and `linearity_q10` are null, with `linearity_reason`: that write cannot
  squash the norm, so `write_over_ffn` and `antipodal.rho` are its size guards (and `ablation_cos` is that of the
  straight write itself).
- `label_overlap` = `{lens, naked_eye: {p10, label_ids, tokens, in_top100}, label_k: 10, causal_k: 100,
  label_check}`: the face's label ids are its first ten through the lens and to the naked eye, the same ids
  `inspect_pole`'s `label_check` counts (`label_gate.label_top_ids`), and `p10` the share of them among the hundred
  ids c raised most, by signed rise (N1_REPORT.md:84-86); null when the neuron writes nothing (a DELETE in force).
  Only this comparison goes through the lens; the effect itself does not.
- `antipodal` = `{rho (Spearman of c+ and c−), top10_in_neg_top100 (of the ten ids the + push raised most, how many
  are among the hundred the − push lowered most)}`.
- `floor` = `{"+", "-": {delta, kl_mean, kl_median, kl_by_context, split_half_rho}}` | null; `base_top` the base
  row's five likeliest word-pieces `{token_id, token, logprob}` (given mode; null in study mode).

scripts/check_causal_check.py checks all of this on a tiny CPU Gemma 4; scripts/verify_causal_check.py reproduces the
study's N1A and N1V numbers on the real model; scripts/verify_e2e_ui_and_mcp.py checks the three transports against
one another and the folio's button against the MCP tool. On the pages the test sits in a neuron's folio, "ii · what it
says": "test it at this word-piece" sends the shown sighting's exact ids up to its read position, with the word chosen
there in `track_ids`; "test it on the study's 36 texts" sends `contexts: "study"`. Both send `floor: true`,
`top_n: 10`, and the apparatus's k and write; the apparatus's scale_ref goes with a test at one word-piece only, the
36 texts always being sized on themselves (`useTheodoliteStore.causalCheck`). The study's figures quoted beside a
result are in frontend/src/screens/NeuronFolio/causal.ts.

### 4.6 An edit's voice (`larql`, `voice`)

WebSocket `{"action":"larql","query":…,"voice":{…}}` and `POST /api/larql` `{"query":…,"voice":{…}}` take an optional
`voice` (VoiceRequest): `prompt` (1–32000 characters, required), `chat` (true), `system` (null), `assistant_prefix`
(""), `max_new_tokens` (1–64, default 12, the length V7 judged steering by). It is heard only by an edit (DELETE,
UPDATE, INSERT); every other statement ignores it, and without one every reply is exactly as before. With one,
`LarqlVaultEngine._heard` composes the question as a sweep's (`compose_prompt`), notes the edits in force (the latest
per neuron, `edits_in_force`), has the model write its answer the likeliest way (`write_voice`: greedy, no nudge, no
re-read, stopping at the end of its turn), makes the edit, and writes the answer again, all inside the one
`_GPU_LOCK` the route holds. The route runs the statement in a worker thread (`asyncio.to_thread`), so the server
keeps answering its health route and other sockets while the two answers are written (one default heard edit took
12.6 s on the GPU); the lock still keeps every sweep and other GPU route out. If the answer before cannot be written, nothing is edited and the statement is refused
in plain words (HTTP 400; over the WebSocket `larql_result` ok false, as is a voice the request model refuses, which
over HTTP is 422). If the answer after cannot be written, the edit stays in force with `after` null and `error` set.
VoiceRequest is backend/larql_vault.py's (the server imports it). The MCP tool `query_larql_vault(lql_query, voice)`
takes the same voice (its schema, as a client sees it, VoiceRequest's), refused in plain words (`voice_request`; one
that is not an object, by FastMCP's check of the tool's types), and replies the same way. Run on its own (`python -m
backend.mcp_scientist`), the MCP server's tools take turns: each runs whole under one lock of backend/mcp_scientist.py's
(`_turn`), and the vault's statement (as `inspect_pole`'s read) runs in a worker thread under it (anyio's, so a call
its client cancels keeps the lock until the thread ends, its edit made), so no other tool's call lands between an
edit's two answers. When the server binds its runtime (`bind_runtime`), that lock is a no-op: the routes hold
`_GPU_LOCK` around every tool they call.

Reply: today's keys plus `voice`, after `patch` and before `lens` (UPDATE, INSERT: `status, patch,
active_patches_count, voice, lens`; DELETE without `lens`). `voice` = `{prompt, chat, system, assistant_prefix,
max_new_tokens, edits_in_force_before (patch ids), before, after, first_step, parted_at, elapsed_ms, error}`, with
before and after = `{token_ids, token_strs, text, stop_reason ("end_mark" | "length"), n_prompt, steps [{index, token,
token_id, model_prob, model_rank}], first_top [{token_id, token, prob}]×5, watch}` (watch: the edit's target word's
`{token_id, token, prob, rank}` at the first word-piece; null for DELETE), `first_step` = `{kl_before_after (nats,
KL(before ‖ after) of the first word-piece's float32 distributions), top_before, top_after, watch_before,
watch_after}` (null when after failed), and `parted_at` the first word-piece index where the two answers differ (null
when they are the same, or after failed). Every patch record (the reply's `patch`, DESCRIBE's `active_patches`) gains
`voice` (null when not heard), `edit_kind` ("one_neuron_column") and `replaces_patch` (the id of the edit in force on
the same neuron that this one replaced, else null); DESCRIBE gains `edit_method`: one down_proj column per edit, rank
one, no covariance solve, named after MEMIT but not MEMIT. The voice is one question, written by the model itself: it
shows what the edit did to that answer only. scripts/check_vault_voice.py checks all of this on a stand-in runtime,
over HTTP, the WebSocket and MCP (`fastmcp.Client`, in memory), the tools' turns among it.

On the pages: the vault's edit tab hears every edit unless the reader turns hearing off (`executeLarql(query, voice)`,
`voicePayload`: the instrument page's question, begun answer, chat setting and system prompt unless typed there, at
`VOICE_LENGTH` = 12 word-pieces), and so does the console for an UPDATE, INSERT or DELETE typed there, with the same
question (`consoleVoice`, on the kind the page's parser gives the statement; hearing on by default, and a blank
question said, not an edit made unheard); ROLLBACK and every other statement are sent without one. The store
keeps each voice heard in `editVoices`, keyed `${patch_id}:${timestamp}` (patch ids start again with the server): a
DESCRIBE keeps only those of records the vault still holds, ROLLBACK forgets them all, and a sighting's cut-down
`larql_status` is merged with the full records already held, so the ledger keeps each edit's time, norms and voice.
scripts/check_interface_defaults.py checks what the bundled store sends and keeps.

### 4.7 The atlas (`backend/atlas.py`, `/api/atlas/*`)

The page `#/atlas` (`frontend/src/screens/Atlas.tsx`) shows all 60 × 21,504 feed-forward neurons at once and reads them
through these routes; the design follows lyramakesmusic/neuron-explorer, built for Qwen3.5-4B (REFERENCES.md [47]; see
`NOTICE`). The rules every route keeps:

- **Who may ask.** Every route answers only pages served from this machine (`local_pages_only`: Host, Origin and
  Sec-Fetch-Site checked; 403 otherwise), since these routes run the model and, through the scientist (§4.8), a shell.
  The server's guard (§4) applies the same rule to every route it has, before these routes' own.
- **The GPU.** A route that runs the model takes the server's `_GPU_LOCK` and works in a worker thread
  (`on_gpu`), so `/api/health` and the sockets keep answering; a push chat holds the lock while it writes and stops at
  its next word-piece when its page goes away (the wait for the writer is shielded from the request's cancellation).
- **Counting.** Layers count from 0 in every field (page layer = layer + 1).
- **Labels.** Every word list read through the J++ lens or by the naked eye goes with its `check` (`label_check`,
  §4.5's gate); a pooled summary pools only the labels the gate lets through. The steer lens is a measured effect on
  the model's own odds and is not gated.
- **Refusals** are plain words (400); a route that needs an index that is not built says so (`ready: false`), or
  falls back to a live reading and names it (`source: "live"`).

**Indexes** (`backend/atlas_index.py`; built by `scripts/atlas_build.py`, `backend/atlas_build.py`), in
`$THEODOLITE_ATLAS_DIR/index`, else `atlas_data/index` (git-ignored). Each file keeps its provenance in `meta` (model, lens
id, built_at, what it was read over), and the server reads a file only for the model and shape it was built on (a
stand-in's never; `GET /arch` names a refused one).

| file | contents | built |
|---|---|---|
| `lens.npz` | `lens_pos`, `lens_neg`, `eye_pos`, `eye_neg` [L, N, 12] int32: each face's first twelve word-pieces through the lens and by the naked eye (the first ten are its label's, ranked as `label_top_ids`); `silenced` [L, N]; `built` [L], the layers read so far (`--layers` adds to the file; an unread layer shows no words) | `all`, minutes |
| `act_scale.npz` | `pos`, `neg` and the GeGLU sign regimes `pp`, `nn`, `pn`, `np` [L, N] float32: each neuron's extremes over a corpus, start marks left out; `resid_q` [L, 3]: the stream's length leaving each layer (10th, 50th, 90th percentile); `mlp_rms_q`, `mid_rms_q` [L, 3]: the same of the feed-forward block's output rms and of the stream's rms entering it, the norms' usual scales the wiring is weighed by | `stats` (in `all`, before wiring), over `--text`, `--jsonl` or `--hf` corpora, else the study's 36 texts |
| `wiring.npz` | `{down,up}_{layer, n, z, lw, sign, gate}` [L, N, 24]: each neuron's 24 strongest direct-path links each way, by strength across layers (below); `{down,up}_{g, u}` float16, each link's two weight products, which the sign in use is read from (248 MB more on the 31B; a build before 28 October 2026 lacks them and serves the sign from rest only) | `all`, minutes |
| `steer_lens.npz` | `up`, `down` [L, N, 12] word-piece ids, `up_kl`, `down_kl` [L, N]: the steer lens (below) | `steer`, hours, by ranges of layers, resumable |
| `layout_writes.npz` | `cell` [L, N] int16: each neuron's cell on its layer's square grid, neurons writing alike side by side (unit write directions, PCA to 50, UMAP where installed, else the first two principal directions, snapped to the grid) | `all` |
| `passages.d/` | one `.npy` per array, memory-mapped, and `meta.json` (written last): `windows` [W, seq] int32, the windows' word-piece ids (−1 past a short one's end; the corpus's own text, never committed); `unigram` [V] int32, each word-piece's count over the word-pieces read; per face `{pos,neg}_{win, at}` [L, N, 16] and `{pos,neg}_{act, g, u}` float16: its 16 strongest windows, one event a window (each window's peak past the start mark, then the 16 largest, ties to the lower window), with the activation and the gate and up pre-activations there; `{pos,neg}_smp_*` [L, N, S], S events drawn at random above half the face's extreme in `act_scale.npz`, at most one a window. About 0.7 GB on the 31B with S = 6 | `passages`, over the corpus it is given (give it the one `stats` read: `rel` and the samples' threshold are against the stats index's extremes, and the route says when the two corpora differ); parts of 1,024 windows, resumable, merged exactly; not in `all` |
| `wiring_check.npz` | the links measured (below): per layer `L{l}_*`, the writers and their pushes, each one's linked readers and the null's, each pair's g, u, signs, scale `f` and strength, the readers' operating points from the passages (`op`), every read [W, R, 4, 3, 2], and each reader's own gate and up values on the unpushed row, `g0`, `u0` [W, R, 4, P] (every word-piece past the start mark, the last at [..., −1]: where it sits); `layers`; the meta, all the server reads of it, holds each layer's summary and cost (`GET /arch` → `links.measured`); a check whose meta has no `format` (measured before g₀ and u₀ were recorded, as the 31B's of the night of 28–29 October 2026 was) lacks `f`, `g0` and `u0` | `wiring_check`, after wiring (and passages, for the sign in use the atlas serves): one `wiring_check.d/L{l}.npz` a layer, resumable (a run with other settings, against another wiring, passages or stats index, or against a folder holding a layer measured before `g0` and `u0` were recorded or a merged file of that format it would replace, refused before it measures anything; `--fresh` sets the folder and the merged file aside, renamed, and starts again), merged as it goes |
| `head_profiles.json`, `vocab.json` | where each head looks over eight reference texts; every word-piece decoded alone | by the server, on first need |

**The push's unit.** A push of dose d adds d · p50<sub>ℓ</sub> along unit(w<sub>post</sub> ⊙ W<sub>down</sub>[:, n]) to the
stream leaving layer ℓ (`resid_post`), at every position but the start mark; p50<sub>ℓ</sub> is the stream's median length
leaving the layer, from `act_scale.npz` when built, else read over the study's 36 texts once per set of vault edits in
force (`scale_source`). A push item is `{layer, n, mode, dose}` (`mode`: `"add"` as above, `"mul"` the activation × d, `"act"`
d × the neuron's usual extreme added to its activation), or `{layer, all: true, mode: "mul", dose: 0}` to silence a whole
feed-forward block; a head is scaled by `{layer, head, dose}` (`head: "all"` silences every head of the layer). A whole
block or every head can only be silenced: the norm after them undoes any other uniform scale (`check_pushes` refuses
it). At most 32 pushes and 64 head scales go into one reading. The three voices sign each push +1, 0, −1 (for `"mul"`:
× d, × 1, × −d). The most a single neuron can write to the stream leaving its layer is s<sub>ℓ</sub> · √D ·
‖w<sub>post</sub> ⊙ unit(W<sub>down</sub>[:, n])‖, however hard it fires (the norm caps it; `Weights.write_ceiling`);
`/neuron/lens` gives it, and as a share of p50 (`ceiling.rel`), and the steer lens, the push tab and the census tab say
how many times that most a dose is. On the 31B the steer lens's 0.8 is 1 to 10 times it at most layers.

**The links.** A writer A at layer i writes v<sub>A</sub> = w<sub>post,i</sub> ⊙ W<sub>down,i</sub>[:, A]; a reader B at layer
j > i reads g = W<sub>gate,j</sub>[B] · (w<sub>pre,j</sub> ⊙ v<sub>A</sub>) and u = W<sub>up,j</sub>[B] · (w<sub>pre,j</sub> ⊙
v<sub>A</sub>); P = g · u. Gemma 4 multiplies the whole stream by `layer_scalar` (0.03 to 0.99 on the 31B) at the end of
every layer, and its norms divide by the rms of what they read, so a link's strength as it arrives is |P| f[i, j]², f[i, j]
= s<sub>i</sub>⋯s<sub>j−1</sub> / (rms of block i's output · rms of the stream entering block j), each rms the median over
the stats corpus (else the study's 36 texts; `atlas_ops.link_scale`). The 24 links each way are chosen and ordered by that
strength across every layer (`w`: over the neuron's strongest; the index keeps log10 as `lw`); `z` is |P| z-scored within
the pair of layers, how unusual the link is there. Signs are read from rest (`atlas_ops.LINK_MODEL`): `gate` = sign(g), the
writer's face that opens the reader's gate (gelu keeps it near shut the other way), `sign` = sign(P), the reader's face it
then pushes toward; a writer is shown on its `gate` face, a reader on its `sign` face, and the active paths of a run keep a
link only when its writer fired on its `gate` face. The sign holds for a reader whose gate is open, or a push big
enough to open it, and the gate-face rule only for a push that big: a reader whose gate is open is moved by its writer
firing either way, the other way about as hard, and a small push at rest moves it the same way either way (the change
is even). At a reader sitting shut (gelu(g₀) < 0), the up path carries about −sign(P), and below gelu's minimum (g₀ ≈
−0.7525, `atlas_ops.GELU_MIN`) the gate path turns too; on the 31B about nine readers in ten sat shut at the census
questions' last word-piece (the chat template's closing `<channel|>`, where the answer begins, not a word of the
question), the one place measured (below). A neuron that writes nothing has no readers. Weight-only: leads to test, not
circuits. The cone follows each neuron's `first` strongest links, then each node's top `k` (2) with z ≥ `zmin`
(15), out to the last layer and back to the first; the web summary and `/cone` follow a link only when the face its writer
was reached on opens the next gate.

From rest is the reading at g₀ = u₀ = 0, where a reader's change is second order in the writer's. Where the reader sits,
its gate and up values (g₀, u₀), the change is first order (`atlas_ops.LINK_MODEL_USE`): a writer's change δa moves the
reader by f[i, j] (α g + β u) δa, with α = gelu′(g₀) u₀ and β = gelu(g₀), so it is odd in the writer, which moves the
reader whichever face it fires on, and the other way when it fires the other way. The operating point read on each face
is the median α and β over the reader's K strongest stored events of that face (`passages.d/`'s g and u:
`operating_points`, `AtlasIndex.op_point`): where the reader fires hardest on that face, its gate open (at 94.5% of
every face's sixteen strongest stored events on the 31B), not where it sits at an ordinary word-piece: at the one place
measured, the census questions' last word-piece, most sat shut (β < 0), and there the sign it gives agreed with the
measured change at about 0.42 of links in the median layer, below chance (below). No resting operating point is built
(ROADMAP §vi), so `use_sign` is the sign while the reader fires hard on a face, and the pages say so. `model=use` on
`/neuron/wiring`, `/web/summary`, `/cone` and `/run/paths` reads links there: each link carries `use_sign` per face of
its reader, gate · sign(α g + β u), the face the writer fired on its `gate` face pushes the reader toward (so it reads
against `sign`), null for a face with no stored event, and every one null with `use.why` without the passages index or
for a wiring index without g and u. The walks reach a reader on a face its writer's firing pushes further (the harder of
two; a link that only pushes it back toward rest is not followed) and a writer upstream on the face whose firing pushes
its reader the way the chain needs; the active paths keep every link whose ends both fired, signed at the reader's own
face. A vault-edited neuron's links are read live, the operating points still from the index (`use.stale` after an edit
at an earlier layer). `check_atlas_ops.py` §11 measures it at stored events of the stand-in: the writer's write, held
through both norms at their values there and pushed ±ε along the direct path alone, moves the reader by exactly
f (α g + β u) ε to first order; with the norms free (the writer's own activation pushed) the change had the sign of
α g + β u at 113 of 120 events, and the median-based sign was each event's own at 76% of (link, event) pairs (both
printed, on random weights, not asserted). `atlas_build.py` counts how often the two readings disagree over the whole
index once both are built.

Both readings are weight-only; `atlas_build.py wiring_check` measures them, prefill only (after neuron-explorer's
steered prefill captures: a push at every prompt word-piece is rebuilt by one prefill). Per layer, `--writers`
(random:512) are pushed by ±d along their write directions at every word-piece of the census's four questions but the
start mark, d = `--dose` (0.25) times each one's write ceiling, a share of what it can write firing alone; its indexed
readers, and as many readers at the same layers it is not linked to (the null, `null_readers`), are read at the last
word-piece and over the prompt (`pushed_reads`: prefills of up to `--batch` rows, 63 at its 64, each with its own
unpushed row, on which each reader's own gate and up values are kept at every word-piece past the start mark, g₀ and u₀,
by hooks on its gate_proj and up_proj, which fire on every unpacker of the 4-bit weights). Reported per layer, for the
linked readers and the null (`wiring_check_summary`, in the log and `GET /arch` → `links.measured`), over the changes
that are not 0 (`zero` counts those): the share whose change under +d has the sign from rest, and under the push on the
writer's `gate` face; the same against the sign in use where the reader sits, at α₀ = gelu′(g₀) u₀ and β₀ = gelu(g₀)
from its own g₀ and u₀ (over the prompt, each coefficient's mean over the word-pieces, the push taken to reach the
reader equally at each) in float64 (`use_plus`, `use_gate`), and against the sign in use the atlas serves, at the
passages' operating point on the face it sat on (`use_plus_peak`, `use_gate_peak`); at the last word-piece the sign from
rest and the sign where it sits by the reader's regime there, gate open (g₀ > 0), shut above gelu's minimum and below
it, with each regime's share; Spearman's ρ of the change's size against |P| f², the second-order term's coefficient, and
against the first-order size where the reader sits, f |α₀ g + β₀ u| d (`rho_site`); how often +d and −d move a reader
the same way (from rest the change is even in the writer, in use odd); how much more the linked readers move than the
null; and a cost line (forwards, rows, seconds, minutes inferred on the 5090). A share of signs is set against 0.5: the
null's sign and gate are its own weights', which its change follows by the same law, so the null tests size
(`moved_over_null`, ρ). A layer measured before g₀ and u₀ were recorded is summed up as it was (a change of 0 a miss,
`use_plus` and `use_gate` at the passages' operating point) and says it needs measuring again (`site_score` when summed
up now; its stored summary has none, so `links.measured` carries it); the two formats never share a folder, and a run
that would mix them, or replace a merged file of the older format, is refused before it measures anything. On the
stand-in's random weights, where every other path counts as much as the direct one, the sign from rest held at about
half the links, as for the null, the sign in use where the reader sits at about 0.87 / 0.79 (linked / null), and the
linked readers moved about twice as much as the null's.

On Gemma-4-31B (the night of 28–29 October 2026: all 59 layers a later one reads, 512 writers a layer, d a quarter of
each writer's ceiling, about 31 minutes on the 5090 with `compiled`; per-layer medians, linked / null, at the last
word-piece) the sign from rest held on the gate face at 0.336 / 0.341 and the sign in use as built, at the passages'
operating point, at 0.418 / 0.438: below chance, links and null alike (changes of exactly 0 left out, as the summary now
counts them; the file's stored summary, counting a 0 as a miss, gives 0.320 / 0.324 and 0.417 / 0.427). No sign
convention is wrong (the index's signs are sign(g u) and sign(g) of the weights at every pair, every layer scalar is
positive, and the push moves linked readers' g and u with the index's signs at 99–100% of pairs, measured on the CPU in
the same way for 24 writers on one question); the readers sit shut. Their g₀ and u₀ at that word-piece, read afterwards
on the CPU (29 October 2026) by a script of their own
(`experiments/results/atlas_wiring_check/scripts/z1_31b_g0u0_cpu.py`; `THEODOLITE_DEVICE=cpu`, the packed checkpoint on
the reference unpacker, one prefill a question with hooks on every layer's gate_proj and up_proj, 31 to 37 s a prefill
on a 12-core desktop; its unpushed activations match the GPU run's at r = 0.9999, the median over layers), put about
nine in ten with the gate shut and 70% / 63% (pooled) below gelu's minimum, where a quarter of the ceiling moves them by
first order and the up path carries about −sign(P); the sign from rest holds where the gate is open (0.82 / 0.69) and
falls to 0.47 / 0.29 shut above the minimum and 0.19 / 0.26 below it. Scored offline from the stored responses at that
measured operating point, the same responses agree with the sign in use at 0.835 / 0.854 (0.842 / 0.853 with α₀ and β₀
in float64, as the summary computes them: in float32 a reader far below the minimum has gelu and gelu′ of exactly 0),
over the prompt at 0.812 / 0.880, and the change's size follows f |α₀ g + β₀ u| d with ρ 0.89 / 0.87, against
0.31 / 0.31 for |P| f². The links are real by size: the linked readers moved 5.5 times as much as the null's. Over the
whole index the sign in use, read where each reader fires hardest, differs from the sign from rest on 28% of links on
each face of the reader (28.2% on +, 28.3% on −; on both only 1.9%, so on about 55% of links on one face or the other).
These were scored offline, outside the check (the scripts, their outputs and a summary naming each figure in
`experiments/results/atlas_wiring_check/`; the inputs over 5 MB in `atlas_data/offline/wiring_check_31b/`, but for the
passages index, read in place at `atlas_data/index/passages.d`, built at 23:46 CDT on 28 October, and the checkpoint the
CPU reads load); it now records g₀ and u₀ itself, and the run that does (`wiring_check --fresh`, the night's layers set
aside) is what it reports from here on.

**The steer lens.** ±0.8 · p50<sub>ℓ</sub> pushed as above at every position but the start mark of two chat turns ("Who are
you?", "Write whatever you like.") and two plain openings, one forward with an unpushed row each; per sign, the mean change
of the next word-piece's log-probability (fp32, softcapped at 30), its twelve largest rises, and KL(pushed ‖ base).

| route | what it answers |
|---|---|
| `GET /arch` | the model's shape (layers, their attention kind, heads, window, `values_from_keys`), which index files are built or refused, which layers have a census, the push's modes and units and its sampler (fields, the shared draw, samples, what is not offered), the limits, `links` (both readings of a sign, whether links can be read in use here and why not, and `measured`, the wiring check per layer), `label_gate` and `lens` cards |
| `GET /neuron/lens?layer&n` | both faces through the lens and by the naked eye with their checks, read live (vault edits included); the steer lens (index, or as read live; for a neuron edited in the vault only as read live since the edit, `steer_stale` saying why the index's is not shown); `act_range` {act_max, act_min, over_tokens, source, regimes}; `ceiling` {write, p50, rel, source}, the most it can write by itself |
| `POST /neuron/steer_lens?layer&n` | the steer lens of one neuron, read live (GPU) |
| `GET /neuron/wiring?layer&n&k=24&model=rest` | its strongest links each way, `down` (readers) and `up` (writers), each with `face`, that face's words and check, its census words and name; from the index, else live (`source`); `model=use` adds each link's `use_sign` {+, −} and `use` {ready, why, operating_points, stale, note} (above) |
| `GET /neuron/heads?layer&n` | the heads it meets by role (q, k, v: later heads reading its write; o: heads whose write it reads), z per head; `z_by_layer` |
| `GET /neuron?layer&n`, `GET /cone?layer&n&model` | all of the above for the scientist; its cone as a list of neurons with hop, z, face and words, from rest or in use (`model`) |
| `GET /neuron/passages?layer&n&face&k&before&after` | one face's stored passages from `passages.d/` (no GPU): its k strongest (16) then those sampled, each with `act`, `rel` (over the face's extreme in the stats index; `rel_to` names its corpus, or says it is another), `regime` (pp, nn, pn, np from `g` and `u`), `window`, `at`, its display units `[text, is_peak]` over `span` (before 24 and after 6 word-pieces, at most 128 each, never across the window's start or end, an end mark, a start mark inside the window (another document's) or a turn mark: `cut` says why each side stops), `next`, and `ids` (the window up to the span's end, for `POST /reads`); `reads` and `follows`, the word-pieces at the peaks and just after, counted over the k with their lift over `unigram`, marked `words_note` "from K passages of a W-word-piece corpus; not validated as a label"; `ready: false` with the build that makes it; `stale` after a vault edit on the neuron or at an earlier layer |
| `GET /hover?layer&n` | the tooltip, from the indexes alone (no GPU) |
| `GET /search?q&limit` | neurons whose face holds `q` among its first twelve word-pieces through the lens, by the naked eye or in the steer lens (`src`), with each face's check: matches the label gate holds back go to `held` (`held_back` counts them), for the apparatus, never the main list or the map; `/run/view`'s word filter likewise matches only labels the gate shows, and the steer lens |
| `GET /web?layer&n&first&k&zmin&cap&hops` | the cone for the map: int16 rows of six (src layer, src n, dst layer, dst n, round(10 z) · sign, hop; hop > 0 downstream); `X-Web-Total` |
| `GET /web/summary?layer&n&…&src&model` | the cone's words, followed from rest (or in use, `model=use`: not ready, with `missing` and `why`, without the passages index) and pooled: `down["+"]` / `down["-"]`, what the neurons it drives write when it fires on its + / − face; `up["+"]` / `up["-"]`, what the neurons that push it toward its + / − face write; weighted by the reaching link's strength (0.8 a hop past the first) times the log of their lift over the layer's base rate; from the steer lens, else from labels the gate lets through (`gated` counts the faces left out); not ready, with `missing`, without the wiring index or without any words to pool |
| `GET /layout` | `cell` [L, N] int16 as base64 |
| `POST /push` | the three voices (server-sent events): `{prompt, history: [{user, replies: [3]}], layer, n, dose (0.8), mode ("add"), neurons, system, max_new (160, ≤ 512), sampler}`; `sampler` as §4.1's (greedy unless a temperature is set; a seed drawn and echoed when none is given; a presence penalty only with a temperature; no min_p or repetition_penalty, `backend/sampler_request.py` says why), drawn once a step for all three voices: one exponential race over the vocabulary, its noise from a CPU generator seeded `seed`, so the voices part only where the push moves the odds (a row's draw follows the sampler's chances but is not interp_engine's, so a seed does not repeat a sweep's); events `{row, t}` per piece of text, then `{row: null, done: true, sampling}` (`"greedy"`, or `{temperature, top_k, top_p, presence_penalty, seed, seed_drawn, samples, seeds, draw: "shared across the three voices"}`) or `{row: null, error}` |
| `POST /steer` | the three voices as whole texts, with `sampling`; `samples` (1–8, sampled only): 3k rows in one batch, sample s drawn with seed + s, `up`, `baseline` and `down` lists when samples > 1; several samples only while their 3k rows hold at most 6,144 word-pieces in all (prompt and answer), about 4.9 GB of context on the 31B (inferred from the census pilot, not measured) |
| `POST /reads` | one neuron's activation at every display unit of a text (`{layer, n, text, chat, neurons, heads}`, or `ids` in place of `text`, read exactly as given, a stored passage's among them; at most 2,048 positions), with `act_range` |
| `POST /logprobs` | the next word-piece's log-probabilities (fp32, softcapped), with pushes or head scales: both rows, their KL and the pieces that gained and lost most |
| `POST /compare` | up to 300 neurons over up to 24 texts, one forward a text: per neuron and text its largest and smallest activation with their units, and its peak score against its usual range |
| `POST /fires` | the neurons that fire hardest on a text against their usual range |
| `POST /run`, `GET /run/view?pos&sign&lo&hi&min&q&limit`, `GET /run/heads?pos`, `GET /run/paths?pos&top&min&model`, `GET /head/row?layer&head&pos` | a text run once (at most 256 positions) and every activation kept: the neurons passing the filters at a word-piece or over all of them (score = activation over the neuron's usual extreme on that side), each head's write there, the links among the strongest (int16 rows as `/web`, from rest or in use: `X-Model`, `X-Use-Refused`), one head's attention by unit |
| `GET /head?layer&head`, `GET /head/profile?layer&head`, `POST /head/pattern`, `POST /attention`, `POST /attn_drive` | a head's q / k / v / o neurons from the weights with their words pooled; where it looks over eight texts; its attention on a text; which heads, blocks and earlier neurons drive a neuron at one unit (`text`, or `ids` read exactly as given, a sighting's own, at most 512): the activation linearised where it reads the stream, the norm's rms held, split exactly over every head (and the units it read), every earlier block and the embedding, and each block's part over its neurons, a<sub>A</sub> (J′<sub>k</sub> · W<sub>down,k</sub>[:, A]) with J′<sub>k</sub> the s-carried w<sub>post,k</sub> ⊙ J over the block output's rms, one mat-vec a layer (`neurons_sum`, equal to the block's part up to the rounding of its output); `neurons`, the twelve strongest across every block, each on the face it fired on with that face's words, check and name |
| `GET /census/units?layer`, `GET /census/unit?layer&n&kind`, `GET /census/card?layer`, `GET /census/rollout?layer&i`, `GET /census/spread?layer` | the layer's census ranking; one unit's answers, the unpushed answers, its statistics and the run's card; the card alone (where the doses and the push unit come from, the unit a dose counts in, how loudly the next layer reads controls against neurons, how many neurons beat every control and how many would by chance, the card it was written on and whether its parts mixed runtime classes, a partial merge, a retest's stage, the writer's floor where measured); one answer; the layout of the run's answers |
| `GET /notes`, `POST /notes` | names and notes (`backend/field_notes.py`): `{kind: "neuron" \| "head", layer, index, name?, note?, by}`; an omitted field is kept, both empty forgets the entry. A neuron folio's name is this name, by `reader` |
| `GET /fieldbook?layer&n`, `POST /fieldbook` | a neuron folio's field book, per face, in `field_book.json` beside the notes (`$THEODOLITE_FIELD_BOOK`; a file of its own, which a server from before never reads or writes): `{jobs: "one" \| "two" \| "unknown" \| null, pattern (≤ 4,000 characters), checked: {other_pole, long_contexts, steered, readers_traced}, by, at}`; given a neuron, both faces (`faces`, null where nothing is said) and its entry in the notes (`notes`); given neither, the whole book. POST `{layer, n, face, jobs?, pattern?, checked?, by}`: a field left out is kept (`checked` may give some ticks only), a face with nothing said is forgotten |
| `GET /proposals`, `POST /proposals`, `POST /proposals/keep`, `POST /proposals/discard` | the names and notes the scientist proposes (its `set_name`, `write_note`), waiting for the person in `field_proposals.json` beside the notes (`$THEODOLITE_FIELD_PROPOSALS`): `{kind, layer, index, name?, note?, chat}`, checked as a note is (an empty string proposes clearing), the reply naming the entry the notes keep now. Nothing reads a proposal as a name or note (the routes above, the map, search, later chats). Keep `{kind, layer, index, part?}` writes it to the notes as the scientist's (`name_by` / `note_by` "scientist") and removes it; discard removes it; `part` is "name" or "note", else both; 404 when none waits |

**The census** (`backend/census.py`; `scripts/census_run.py` on the GPU, `scripts/census_tools.py` on the CPU). For one
layer, every neuron (or a range) and some random directions are pushed as above on the design's questions and doses
(`main_rel`: "Who are you?", a snake game in Python and the colour orange at −0.7, +0.5 and +0.8, and "Write whatever you
like." at +3.0, plus one unpushed answer each), greedy, batched by question. A dose counts in one of two units
(`dose_unit`, in the meta and the card): `stream`, a share of the stream's median length, the same push for every unit,
or `ceiling`, a multiple of each neuron's write ceiling (a control pushed by the ceiling of the neuron it is paired
with), so that one dose is the same share of each neuron's most at every layer. Every pushed answer records its unit's
ceiling and its push's real share of the stream (`rel_push`, which the ranking's 1.6 filters on). Two calibration
ladders come first: `calib_rel` (±0.1 to 4 of the stream) and `calib_ceil` (±0.5 to 8 × the ceiling). A control is
unit(w_post ⊙ r), r isotropic from a seed: the reference's random down column carried through the post-feed-forward
norm every neuron's write passes, so the next layer reads controls about as loudly as neurons (on the 31B's own
weights, 1.02× at page 46 and 1.18× at page 58, where a plain isotropic direction is read 103× and 236× as loudly;
`check_census.py` measures it when the checkpoint is cached). The run's meta records that ratio (`read_gain`) and
where the doses come from (`doses_from`:
neuron-explorer's, chosen on Qwen3.5-4B, not yet calibrated here); runs taken with isotropic controls (meta `version`
below 2) are not read, and `/arch` says why (`census_refused`). Answers go to `$THEODOLITE_CENSUS_DIR`, else `<atlas
folder>/census/runs`, `L<layer>-<design>.jsonl` with its `.meta.json`, in batches planned from the whole job list and
the batch size (`plan_batches`: 10 × batch a chunk, each chunk grouped by question), each written whole or not at all.
So a stopped run, or one killed mid-write, resumes to the rows an unbroken run writes: on the GPU a row's numbers depend
on its batch's size. A run with other settings, the batch among them once it has a plan, is refused before anything in
it is cut; else a half-written last line, and a last batch written in part, are cut off first. The meta records the
runtime class (the card, the library versions, the commit, the stats index's sha256); a change is warned and recorded
(`mixed_runtime`), never refused. `--shard i/n`
takes whole chunks of the plan, and every shard but 0 writes the plan's first batch again as a canary. A shard is never
read as a layer's census: `census_tools.py merge` joins shards only whole, every shard and every answer (else refused,
unless `--partial`, which the meta records), with each canary compared with shard 0's answers as whole JSON values.
`census_run.py --canary-only` writes that first batch alone, beside the run's path with the meta its run would begin
with (`<run>.canary.json`), and `census_tools.py canary` sets two such against each other, so a rented card's batch is
set against the home card's before a run. Every batch is written by one function (`write_census_batch`), which
`scripts/verify_atlas_writer.py` also writes its census-shaped batches with; the card carries the writer's floor that
script measured for the run's model (`writer_floor`, `census.writer_floor`: with no push, after how many word-pieces an answer
written in batches like the run's parts from the same answer written alone, the floor of the run's runtime class first;
none where no floor is on disk). A
second stage (`census_run.py --neurons-from <screen>.units.jsonl --top M`, a fresh seed) retests a screen's strongest
neurons against fresh controls; its p and q are taken over those M alone (`stage`), and the layer's census stays the
screen. Ranking (`compute_units`, after
the explorer's `sweep/rank.py`, reading a run one line at a time and keeping only each answer's word ids): a word (three
letters or more) counts once per answer, over pushes of at most 1.6 of the stream, leaving out every word of the
unpushed answers; each unit's words scored against the average unit, z = (count − mean) / √(mean + 0.5); `mass` the
summed positive z of its top five words, the sort key; `null_pct` the share of controls below it; `p` = (1 + controls at
least as strong) / (C + 1), and `q` the Benjamini–Hochberg false-discovery q over the run's neurons (about N / (C + 1)
neurons beat every control by chance alone, 84 of a whole layer with 256 controls, and every page that ranks says so);
per dose, the push's length over the stream's, distinct 4-grams, the share cut off, the mean log-probability. A
neuron's census words describe it beside links, search hits and the map's tooltip only when it beat every control.
`census_tools.py` ranks (`.units.jsonl`, with the word-shift score), merges shards, writes the depth report (each entry
names its dose unit; in the stream's, one nominal dose is a different push at each layer) and lays a run's answers out
(`.spread.json`: words by tf-idf, or `st:<model>` through sentence-transformers; UMAP where installed, else PCA). It
also reads a calibration run's dose window (`curves`, after the explorer's `sweep/peek.py`): per dose, neurons against
controls, the share of answers moved from the unpushed one, where they first part, the share cut off, distinct 4-grams,
the mean log-probability, and the push over each neuron's ceiling (from the answers' own record, else the checkpoint's
weights), then a hedged verdict per sign on where neurons start to move more answers than controls and where text
breaks. The server keeps the last two layers' answers it opened.

`/api/health` carries an `atlas` card (routes, the index folder, which index files exist, the census layers, the
notes' file, the field book's and the proposals') just before `label_gate` and `lens`. scripts/check_atlas_ops.py, check_atlas_index.py, check_atlas_api.py
and check_census.py check all of this on the CPU stand-in (`scripts/atlas_stand_in.py`).

### 4.8 The scientist (`backend/scientist_chat.py`, `/api/atlas/scientist/*`)

A language model beside the atlas, with the atlas's tools. A chat keeps the backend it began with:

- `claude`: Claude Code (`claude -p`, stream-json both ways), one process per chat, kept open so a message sent while
  it works reaches it between tool calls; its session file is resumed turn by turn. Its tools are the atlas's (the
  stdio MCP server `backend/scientist_mcp.py`) and, by its reach (`$THEODOLITE_SCIENTIST_REACH`): "folder", the
  default, Read, Write and Edit scoped to its work folder (`Read(//<work>/**)` and so on), no Bash, no other folder;
  "home", set by the person, Read, Glob, Grep, Write, Edit and Bash pre-approved with the home folder added: **it runs
  commands and edits files there without asking**. At either reach no settings file, hook, rules file or CLAUDE.md is
  loaded (`--setting-sources ""`): under `project`, as first built, a CLAUDE.md in its work folder, which it may write,
  or in any folder above it (the home folder's own `.claude/CLAUDE.md` included) reached every later chat as
  instructions, and so did a `.claude/rules/` file in the work folder, and its `.claude/settings.json` ran its hooks as
  shell commands, a shell at the folder reach too (Claude Code 2.1.312; that CLI asks before writing in `.claude/`
  itself, so refused the scientist's own write there, but a Codex chat's sandbox may write in the same folder).
  `$THEODOLITE_CLAUDE_MODEL` picks the model.
- `codex`: Codex (`codex app-server`, JSON-RPC) with the same MCP tools and its own shell, in a Codex home of its own,
  `approvalPolicy: "never"`; `sandbox: "workspace-write"` by default (writes only in its work folder, no network; it
  can read elsewhere), `"danger-full-access"` under "home" (the same caution).
- Neither process gets the endpoints' keys (`child_env`), and `/backends` says what each backend can do here
  (`reach`); under "home" the page asks once, per browser, before the first Claude Code or Codex chat.
- Nor does either get the Hugging Face or cloud credentials in the server's environment (`SECRET_ENV`: `HF_TOKEN`,
  `HUGGING_FACE_HUB_TOKEN`, `HF_API_TOKEN`, `RUNPOD_API_KEY`, `VAST_API_KEY`, `LAMBDA_API_KEY`, `MODAL_TOKEN_ID`,
  `MODAL_TOKEN_SECRET`, `GOOGLE_APPLICATION_CREDENTIALS`, `NEURONARY_TOKEN`, and every `AWS_*` and `AZURE_*`). Each
  keeps its own login: Claude Code's `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `CLAUDE_CODE_OAUTH_TOKEN` or
  `~/.claude`, and the cloud's variables when `CLAUDE_CODE_USE_BEDROCK`, `_VERTEX` or `_FOUNDRY` is on (1, true, yes or
  on, as Claude Code reads it; to Claude Code only); Codex's `OPENAI_API_KEY`, `CODEX_*` or its `auth.json`. A
  credential in a file (`~/.cache/huggingface/token`, say) is bounded only by the reach.
- `openai`: any OpenAI-compatible Chat Completions endpoint that takes tool calls (LM Studio, Ollama, OpenRouter, vLLM,
  llama.cpp's server). The server runs the loop: it streams the reply, runs each tool call through the atlas's routes in
  this process (`InProcessApi`, an ASGI client), sends the results back (each cut to 16,000 characters) and goes on until
  the model answers without a call, at most 40 rounds a turn. Such a model has the atlas's tools only, no shell and no
  files; the person's messages and the tools' replies are sent to that endpoint.

Endpoints are profiles: `lmstudio` (http://127.0.0.1:1234/v1, `$THEODOLITE_LMSTUDIO_URL`), `ollama`
(http://127.0.0.1:11434/v1, `$THEODOLITE_OLLAMA_URL`), `openrouter` (https://openrouter.ai/api/v1, its key in
`$OPENROUTER_API_KEY`), `custom` (`$THEODOLITE_OPENAI_BASE_URL`, `$THEODOLITE_OPENAI_API_KEY`,
`$THEODOLITE_OPENAI_MODEL`), and those of `<scientist folder>/endpoints.json` (`{name: {label, base_url, key_env, model,
stream, headers, params}}`). A profile names the environment variable its key is in, never the key (a file entry holding
one is refused); a page picks a profile by name and never sees a key.

The scientist folder is `$THEODOLITE_SCIENTIST_DIR`, else `<atlas folder>/scientist`: `work/` (the chat's working folder),
`uploads/`, `logs/` (the codex and openai chats' events and messages; a line cut off mid-write, by a server stopped while
writing it, is left out wherever a log or a Claude Code session is read, and cut off before the next event is written),
`chats.json`, and the system prompt, built from
`backend/scientist_prompt.md`, `backend/scientist_methodology.md` and the person's own `prompt.local.md` there (for
`openai`, without the shell's section).

A Claude Code or Codex chat reads those files, `scientist_mcp.py` and `scientist_tools.py` when its process starts, and
the server watches them by modification time. At the person's next message to a live, idle chat that started before
they last changed, a Claude Code chat at the folder reach is ended and started again, resuming its session, with a
`note` event: it has no shell, so only its MCP server lives in its process group. Any other chat (Codex at either
reach, Claude Code at "home") is not started again unasked, since ending its process group would also end any commands
it left running in the background in that group: it is marked stale (a `stale` event, which the first `state` event of
a stream also carries, until its process ends), and the page offers `POST /restart`. A chat mid-turn is never ended, as
far as its events tell: Claude Code answers from its first streamed event, thinking included, and one whose last turn
took a message mid-turn may yet run that message as a turn of its own, so until its next turn begins it is marked stale
instead. An `openai` chat's prompt is built afresh every round; its tools change only when the server starts again.

The routes answer only this machine's pages and scripts (§4), and on Linux only the account the server runs as (a
request whose sender has gone before the guard could look is refused too), so another account on the machine can no
longer send a chat a message that it takes as the person's own turn, nor an attachment (except through `npm run dev`'s
proxy while it runs, §4, or where the guard cannot tell whose a connection is, which its log says).

| route | |
|---|---|
| `GET /backends` | whether `claude` and `codex` are installed; every endpoint's card `{name, label, base_url, this_machine, key: "set" \| "missing: …" \| "none needed", model, stream}` |
| `GET /models?profile` | the endpoint's `GET <base>/models`: `{id, name, context, tools}` (`tools` null when it does not say) |
| `POST /send` | `{chat (a UUID), message, agent, profile, model, layer, n \| head, attachments}`: a message into the chat (its first fixes the backend, and for `openai` the profile and model); `layer` and `n` (or `head`) say what the page has open, which the scientist is told in the pages' count |
| `GET /events?chat` | NDJSON: `{t: "state", busy, running, agent, stale}` and the turn so far, then `sent`, `text`, `thinking`, `tool`, `tool_result`, `ui`, `error`, `done`, `note` (started again on changed instructions or tools), `stale` (`{stale}`), `ping`; it never ends of itself, so the server is started with a graceful-shutdown limit (`--timeout-graceful-shutdown 3`, README §v), or a stop would wait on every page left open |
| `POST /interrupt` | `{chat}`: stops the reply; the chat (and Claude Code's or Codex's background commands) goes on |
| `POST /ui` | from the tools: a screen action (`select`, `head`, `tab`, `camera`, `highlight`, `annotate`, `text`, `search`, `rename`), passed to the chat's page as a `ui` event, layers from 0 |
| `GET /chats`, `GET /history?chat` | past chats, newest first; one chat as events |
| `POST /rewind` | `{chat, text}`: cut the chat back to before its last message (a Claude Code session is backed up first), for "edit" and "ask again" → `{ok, cut}`; a Codex turn is reverted whole (`thread/revert`), so a message sent into it while it ran goes with it: its log is cut back to where that turn began (`turn_at` in `chats.json`), and `messages` says how many of the person's messages went, which the page takes off its thread |
| `POST /restart` | `{chat}`: end its process group, with any commands it left running in that group (409 while it answers), so the next message resumes it on its files as they now stand → `{ended, note}` |
| `POST /upload` | `{name, type, data (base64)}` (at most 25 MB) → `{id}`, for `attachments` |
| `GET /prompt?agent` | the system prompt as built |

The tools (`backend/scientist_tools.py`, 30, standard library only) take and give layers in the pages' count (L54 N15491)
and call the atlas's routes: reading the atlas (`overview`, `neuron`, `passages`, `rollouts`, `cone`, `web_summary`,
`search_lens`, `top_units`), names and notes (`set_name` and `write_note` propose, for the person to keep or discard on the
page; `names` lists those kept), live readings (`steer`, `activations`, `compare`,
`logprobs`, `top_firing`, `steer_lens`, `causal_check`, `attention`, `attn_drive`, `head`, `neuron_heads`), and the
person's screen (`select_neuron`, `select_head`, `open_tab`, `camera`, `highlight_neurons`, `annotate_neurons`,
`show_text`, `show_search`). scripts/check_scientist.py checks the tools, the MCP server and all three backends against
fakes, over real connections. scripts/verify_scientist_cli.py runs both reaches on the real `claude` (where it is on
PATH) against a stand-in Messages API, with no model, no network and no login: what each allows and refuses, the
atlas's tools through the real MCP server, and that no planted CLAUDE.md or rules file reaches a request and no planted
hook runs (it records the CLI's version; 2.1.312 here).

---

## 5. CPU and GGUF

The instrument was built for the W4A16 checkpoint on one GPU. Since 28 October 2026 it can also read a GGUF of Gemma 4
(llama.cpp's format), and run on the CPU; the first two layers of that work, checked on the CPU only
(`scripts/check_gguf.py` and `scripts/check_gguf_cpu.py`, on the stand-in of `scripts/atlas_stand_in.py` written to tiny
GGUF files). Nothing here has run on Gemma-4-31B from a GGUF, and the checkpoint has run on the CPU only for the wiring
check's offline reads (§4.7: eight prefills of the packed checkpoint, 31 to 51 s each on a 12-core desktop, 29 October
2026).

**Choosing the runtime** (`backend/runtime_factory.py`, `make_runtime()`; the server, `scripts/census_run.py`,
`scripts/atlas_build.py`, the scientist's MCP server run alone, the lab daemon `experiments/lab/labd.py` and the GPU
checks `scripts/verify_optics.py` and `scripts/verify_causal_check.py` call it; with nothing set each builds the pinned
runtime as before. The study's registered R2–R4 scripts, `N1G_01` (the frontend's fixture of the checkpoint's labels)
and `scripts/check_sweep_refactor.py` (two checkouts, one older than the factory) still build it themselves):

| variable | values | |
|---|---|---|
| `THEODOLITE_GGUF` | a `.gguf` file | `GGUFRuntime` on it; unset: the pinned `TheodoliteRuntime` on the W4A16 checkpoint, as before |
| `THEODOLITE_DEVICE` | `cpu`, `cuda:N` | where the model runs (default `cuda:0`); `cpu` hides the GPUs from the process (`CUDA_VISIBLE_DEVICES=""`) |
| `THEODOLITE_GGUF_LINEARS` | `packed`, `resident`, `dense`, `int4mm` | how a GGUF's quantised layers are held (default `packed`); `int4mm`, the CPU only, multiplies Q4_0 layers by torch's int4 kernel (below) |
| `THEODOLITE_CHECKPOINT_LINEARS` | `packed`, `int4mm` | the same for the W4A16 checkpoint on the CPU (default `packed`) |
| `THEODOLITE_GGUF_CONFIG` | a `config.json` | build from this config: it stands in for what the file's metadata leaves out, and is refused where it disagrees with what the file states |
| `THEODOLITE_DTYPE` | `bfloat16`, `float32` | a GGUF model's dtype (default `bfloat16`, as the instrument runs) |
| `THEODOLITE_GGUF_LENS` | `jpp`, `none` | load the J++ lens beside a GGUF model (default), or leave it out (3.4 GB) |
| `THEODOLITE_CPU_MEMORY` | `off` | leave out the CPU's memory estimates (below) |

`GET /api/health` now also sends `device` and `weights`, the weights card (`runtime_factory.weights_card`: a GGUF's
provenance; for the checkpoint its repo id, the snapshot of it the Hugging Face cache holds and its weight files); the
lab daemon's `/health` and every `/exec` reply send the model id and the card too.

**What works now**

- `backend/gguf_weights.py`: torch unpackers for F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K and
  Q6_K, on the CPU or a GPU, each equal to gguf-py's own dequantiser (gguf 0.19) bit for bit: exhaustively for F16,
  BF16, Q8_0 (every code under every fp16 scale) and Q4_0, over every fp16 scale field of every other type, and on random
  blocks. `GGUFLinear` holds a tensor's blocks; `dense`, `dense_cols` and `dense_rows` read any linear layer's own
  weight (plain, W4A16 through the pinned unpackers, or GGUF) with the pinned unpackers' shapes. `atlas_ops.dense_weight`
  and the vault's column read go through them.
- `backend/gguf_model.py`: a `Gemma4ForCausalLM` built from the file without allocating random weights, its config from
  the metadata llama.cpp's converter writes (or a matching `config.json`, which also stands in for keys a file leaves
  out), its tensors by gguf-py's name map, and a provenance record. `GGUFRuntime` (`backend/gguf_runtime.py`) subclasses
  the pinned runtime: its `load()` does what the pinned one does once its model is built, and the seven accessors that
  read `weight_packed` read through the accessors above; everything else is the pinned code. On the stand-in its
  accessors, a push in three voices, a neuron read, a sighting's row capture and naked-eye readout, a vault DELETE on a
  packed layer and a census equal the stand-in runtime's, bit for bit in float32.
- The model id of a GGUF reading is `gguf:<file>#<fingerprint>`, and any reading taken off the GPU or in float32 carries
  where and in what after it (`runtime_factory.reading_class`: `gguf:<file>#<fingerprint>@cpu/bfloat16`, and the
  checkpoint on the CPU `google/gemma-4-31B-it-qat-w4a16-ct@cpu/bfloat16`; bfloat16 on a GPU adds nothing). Every index,
  census run and lens card records that id, a census run is continued only on the same id, and the server reads an
  index only for it: so a file's readings are not read as another file's (as far as the fingerprint tells them apart)
  or the checkpoint's, and readings of two numeric classes are not mixed in one run. How a GGUF's layers are held
  (packed, resident, dense) is not in the id: the three give the same forward bit for bit (in float32 and bf16 on the
  CPU stand-in), and the `weights` card says which; int4mm, which does not, is (`+int4mm`, below).
- `scripts/compare_gguf_ct.py` says whether a GGUF holds the checkpoint's weights, tensor by tensor and value for value;
  for Q4_0 against W4A16 (both group-32 symmetric 4-bit) also the codes and the scales, a Q4_0 block read either way
  round (llama.cpp's quantiser writes d = max / -8, so a block may hold -scale and the codes 16 - q). Its self-test runs
  on the CPU.
- Every index's meta names the weights it was built from (`weights`, the weights card), beside the model id; the
  server, `census_run.py`'s stats lookup and `atlas_build.py` read an index only when `atlas_index.weights_differ`
  finds no difference (a GGUF by its fingerprint and header; the checkpoint by its repo and the weight files its
  snapshot links to, config.json and the .safetensors, by the blobs the hub's cache names for their content, so another
  commit of the same files, a model card edited, is the same weights; an index built before metas named their weights,
  or a cache that does not link its files, by the model id alone), and refuse it in plain words otherwise ("built from
  other weights (…): rebuild it"); `census_run.py` and `atlas_build.py` say so when the stats index is there but
  refused, rather than that there is none. A steer lens layer, the passages' parts and a partial lens build are not
  resumed on other weights. A census run's meta records the model id only, so a run is not kept to one set of weights
  behind one id.
- The lab's gradient passes (`labkit.Lab.grad_linears`, `same_pos_vjp`, what the lens fits rest on) run through a GGUF
  twin of its W4A16 autograd function (`_GGUFLinearFn`: the weight unpacked by `gguf_weights`, overlays laid on, and
  unpacked again in backward, so no weight is kept), so lens fits and other lab experiments can run from a GGUF or on
  the CPU; on the stand-in their gradients equal the dense stand-in's bit for bit, in float32 and bf16. `Lab.model_id`
  and `Lab.weights` name what an experiment read.
- The CPU's memory (`backend/cpu_memory.py`): before a sweep, a census batch, the vault's voice, a causal check's
  forward and a steer lens batch, an estimate of what it holds at once (from the model's config: keys and values per
  row-piece, the widest layer's activations, attention scores, logits, the largest packed layer's unpacking) against
  MemAvailable, and a refusal in plain words, before anything is read, when it cannot hold it (a causal check first
  splits its pushed rows, as it does on the card when memory runs out); a CPU allocator's failure is answered in the
  same words. Worked out for the 31B the estimates lean high by about half against the card's two measurements (a
  census batch, a 2048-piece sweep): arithmetic, not measured on a CPU.

**Packed or dense.** `packed` keeps each quantised layer's blocks (on the CPU read in place through the file's memory
map) and unpacks the whole weight at every forward, as the W4A16 forward unpacks the checkpoint; `resident` unpacks once
at load and drops the blocks; `dense` builds plain `nn.Linear` layers, as transformers' own GGUF loader would. A tensor
stored as F32, F16 or BF16 is a plain layer in every mode. All three hear the vault's overlays as `_w4a16_forward` lays
them (column overlays, then row overlays, on a clone), and the forward is the same numbers in every mode. The embedding
(which is also the tied lm_head), the norms and the layer scalars are unpacked to the model's dtype.
`int4mm`, the int4 kernel below, is none of the three.

**The norm convention.** `Gemma4RMSNorm` multiplies by its stored weight, which already holds what Gemma 2 and 3 write as
1 + w; a Gemma 4 GGUF stores that same number. Read in llama.cpp's converter (`conversion/gemma.py`, master of
28 October 2026: `Gemma4Model.norm_shift` returns 0, where its Gemma 3 adds 1) and in transformers (its GGUF reader
subtracts 1 for gemma2 and gemma3 and has no rule for gemma4). The check writes the stand-in with its norms as they are,
reads them back unchanged, gets the stand-in's own forward, and shows a file holding w + 1 would not;
`compare_gguf_ct.py` names a file whose norms sit exactly 1 above the checkpoint's.

**Provenance** (`gguf_model.provenance`, the `weights` card): the file's name and size, sha256 of its header (every key
and every tensor's name, type, shape and offset), of the first MiB of its tensor data (the header alone is some MiB on
the 31B, its 262,144-piece vocabulary among the keys: inferred, no 31B GGUF read here) and of its last MiB, a
fingerprint of the four, the count of each tensor type, the arch and the metadata that matter, and how the model was
built (config source, linear layers, dtype, device) and where its tokenizer came from; not the file's path, which
stays in the log. The whole file is not hashed: the fingerprint moves with any key, tensor table or byte in the first or
last MiB of the data, not with a change in between.

**The tokenizer.** From `google/gemma-4-31B-it-qat-w4a16-ct` or `google/gemma-4-31b-it` when cached here and its pieces
are the file's; else from the file's own metadata through transformers' GGUF tokenizer converter. transformers'
`AutoTokenizer.from_pretrained(..., gguf_file=...)` cannot be used for a llama.cpp Gemma 4: it first builds a config,
and `Gemma4TextConfig` refuses the per-layer `head_count_kv` list llama.cpp writes. Whether the GGUF-built tokenizer
splits text exactly as the repo's `tokenizer.json` is not checked (no Gemma 4 tokenizer is cached on the machine this was
written on); on the stand-in it does, and transformers' converter adds GPT-2's `<|endoftext|>` past the model's rows.

**transformers' own GGUF loader** (5.21) does not build a llama.cpp Gemma 4 correctly: besides the config above, its name
map looks for `layer_output_scale`, where llama.cpp writes `layer_output_scale.weight`, so the layer scalars are left out.
That is why `gguf_model.py` builds the model itself.

**The CPU path** (`THEODOLITE_DEVICE=cpu`; measured on 4 vCPUs of a 2.1 GHz Xeon with AMX and AVX512-BF16, the rest
inferred):

| | memory | a forward |
|---|---|---|
| packed 4-bit (a Q4_0 GGUF, or the W4A16 checkpoint) | about 20 GB (the 4-bit weights and the bf16 embedding) | every weight unpacked: 7.8 ns a weight for Q4_0, 11.4 for Q4_K, 10.0 for the pinned W4A16 unpacker (measured on a 5,376 × 21,504 layer), so about 4 to 6 minutes a forward on that machine for 29.3 billion weights |
| resident bf16 | about 62 GB | a plain product: bf16 at 12 GB/s of weights for one row, 306 GFLOP/s at 32 rows there, so about 5 s a written word-piece for one row and 6–7 s for 32 |
| int4mm (a Q4_0 GGUF, or the checkpoint with `THEODOLITE_CHECKPOINT_LINEARS=int4mm`) | about 21 GB (the kernel's 0.625 bytes a weight and the bf16 embedding; the checkpoint keeps its 15 GB of `weight_packed` beside them, a GGUF its file's pages) | torch's int4 kernel, no unpacking (below): about 2 s a written word-piece at one row, 21 s a step of 64 rows (one layer timed, the rest inferred) |
| the J++ lens | 3.4 GB more (bf16, in RAM) | — |

**The int4 kernel** (`backend/int4_cpu.py`, `THEODOLITE_GGUF_LINEARS=int4mm` or `THEODOLITE_CHECKPOINT_LINEARS=int4mm`).
torch 2.14's `_weight_int4pack_mm_for_cpu` multiplies 4-bit codes with a scale and zero per 32 in place, never writing
the bf16 weight. Measured on this 4-vCPU container (28 October 2026) on one random layer of 5,376 × 21,504 each way
round, in bf16, the best of 3 or 5 in each of three runs (`scripts/compare_int4mm.py --time` twice, a first script
once), with other work running beside them, so the figures are a spread, not a value:

| one 5,376 × 21,504 layer (115.6 M weights) | 1 row | 8 rows | 64 rows |
|---|---|---|---|
| Q4_0, unpacked then multiplied (the reference) | 0.66–1.49 s | 0.30–0.88 s | 0.69–1.84 s |
| W4A16, the pinned unpacker then multiplied | 0.52–0.67 s | 0.51–0.78 s | 0.44–0.86 s |
| W4A16, `w4a16_fast`'s lut unpacker (one run) | 0.26–0.38 s | 0.29–0.66 s | 0.24–0.50 s |
| resident bf16 | 7–17 ms | 4.5–33 ms | 17–24 ms |
| int4mm (Q4_0 or W4A16) | 1.7–8 ms | 6.6–40 ms | 56–131 ms |

So, within a run, int4mm was 70 to 150 times as fast as the reference at one row (several hundred times in the first,
quieter run), 20 to 55 times at eight and 4 to 14 times at 64; as fast as resident bf16 at one or eight rows, and 3 to 5
times slower at 64. torch's other int4 route, `_dyn_quant_matmul_4bit` (fast through Arm's KleidiAI, here a fallback),
took 0.19 s at one row and 10 s at 64, and quantises the activations to int8; the int4 kernel in float32 was slower than
the reference at 8 rows and more. The kernel's weight is (q − 8) × scale + zero: a W4A16 scale is held exactly, a Q4_0
d (fp16) is rounded to bf16, and the sums are the kernel's own, so its readings are not the reference's bits: their
model id ends `+int4mm` (no census run, index or lens card mixes them with the reference's) and the weights card says
`int4mm`. Its accessors read the file's blocks (or `weight_packed`) as before. A vault edit keeps the kernel's
product and adds the edit as its difference from the file's numbers (`int4_cpu.with_overlay_terms`: the edited
columns' new values less the file's, times their activations, in float32), so an edit that changes no weight changes no
reading, and what the voice, a sweep or a causal check credits to an edit is the edit's alone, not a change of numeric
path as well (before this the edited layer ran the reference, and on the stand-in an edit setting a column to its own
value changed 13 of 30 greedy answers). The lab's gradient passes run the reference; a layer the kernel cannot take (out
features not in sixteens, a GGUF type other than Q4_0) stays packed, and a file of which it takes none is refused. A
row's product through the kernel is the same alone or in a batch.

How far it departs is `int4_cpu.compare_rows`, after neuron-explorer's `sweep/check_equivalence.py` judged on what a
census reads: `scripts/compare_int4mm.py` takes a canary census on the reference at two batches (the floor) and through
the kernel (by default the design's canary: 16 neurons and 8 controls), and holds it to a rule fixed before any run
on the 31B: its readings are their own class if they depart further than the floor in identical answers, shared
prefix, largest |Δ log-prob| or top-5 overlap. Those are only the per-answer half of the equivalence design it follows:
the per-run half (the rankings of a whole depth pool compared: Spearman's ρ of mass, the top 100's overlap, the neurons
above every control, the false discoveries) is not taken, so the script can only refuse; within the floor on all four,
the kernel's readings still do not stand for the reference's. On the stand-in's Q4_0 GGUF in bf16 (4 neurons, 2 controls, 64
answers) the floor is nought on the 4-vCPU container it was first run on (the reference is the same at batch 8 and 3);
on a Ryzen 9 7900X, whose AVX-512 bf16 products round by the batch, every answer is the same but a log-prob moves by
up to 0.0039 (nought again with `ONEDNN_MAX_CPU_ISA=AVX2`), so a CPU's floor is to be measured on it, as a card's is.
Through the kernel 30 of 64 answers are identical (29 on the 7900X's AVX-512 path), the shared prefix 65%, the largest
|Δ log-prob| over it 0.024, the top-5 overlap 0.95 (at the script's defaults, 244 answers: 119 identical, 67%, 0.030,
0.96): its own class by the rule (the stand-in's small, near-tied logits split greedy answers easily; the 31B is
untested).

A machine with more cores and memory bandwidth should be some times faster (inferred, not measured). Either way the CPU
suits reading weights (the wiring, the lens build, `compare_gguf_ct.py`), single forwards (a sighting's capture) and
pilots, not a census of a layer (217,604 answers: a step of 64 rows takes minutes on the reference path and about 21 s
through int4mm, 253 layer-sized products, so some 50 days a layer at `--max-new 64` even so; inferred, not run).
bf16 on the CPU runs through oneDNN, fast with AMX or
AVX512-BF16 and emulated without; its sums are not the GPU's, so a reading taken on the CPU is its own determinism class,
and its model id says so (`@cpu/bfloat16`).

**CUDA assumptions, audited.** In the unpinned code: GET /api/health asked the card's memory whenever CUDA was available
(now only for a runtime on a GPU); the server, the atlas, the vault, the causal check and the scientist's tools catch
`torch.cuda.OutOfMemoryError`, which the CPU never raises (an allocation the CPU cannot make is a `RuntimeError` or the
kernel's out-of-memory killer: five readings that hold large buffers are now estimated first and refused in plain
words, `backend/cpu_memory.py`); `_GPU_LOCK` is named for the GPU but serialises the
model wherever it runs; `scripts/verify_*.py` are GPU checks by design. In the pinned `engine_runtime.py`, left as it is:
`load()` (`:833`) and every sighting (`:2393`) call `torch.cuda.max_memory_allocated(self.device)` whenever CUDA is
available, which for a CPU device torch 2.14.1 answers with 0 while CUDA is not initialised in the process, and refuses
once it is (`memory_stats_as_nested_dict`; read from torch's code, not seen: there is no GPU here), so the GPUs are
hidden as a precaution against a library initialising CUDA (a process that already has is warned); the default device
is `cuda:0` (`:741`); the J++ lens always goes to the runtime's device
(`:879`); the norm loop counts 60 layers (`:818`); the accessors read `weight_packed` (`:990-1077`, `:1252`); the sweep's
shapes are the 31B's (`N_LAYERS`, `D_MODEL`, `D_FF` from the pinned `theodolite_math.py`). Changing any of these in the
pinned file would make `R4_common.pinned_differ()` name it, and R4 reruns would then go from R4's registered commit, as
R3's go from `it-jpp-refit`.

**Not supported yet:** the IQ types, TQ1_0, TQ2_0, MXFP4, NVFP4 and Q1_0 (refused in plain words); Gemma 4's
mixture-of-experts (26B-A4B), per-layer embeddings (E2B, E4B) and shared key-value layers (refused); a bit-identical
fast path on the CPU (every packed forward of the reference unpacks the whole weight; int4mm is fast but its own class);
int4mm for Q4_1, Q4_K and the other K types (they stay packed); split GGUF files (`-00001-of-0000N`); the study's
registered scripts, which build the pinned runtime on the checkpoint and the GPU they were registered on. interp-engine
takes the model as built (its own dtype argument is not used for a model handed to it); it ran the stand-in on this
CPU-only machine without complaint, which cannot show what it asks of a GPU that is there but not used.

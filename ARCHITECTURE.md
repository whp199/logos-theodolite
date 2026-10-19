# Logos Theodolite — Architecture & Tensor Contracts

**Target Model:** `google/gemma-4-31b-it` (`google/gemma-4-31B-it-qat-w4a16-ct` / 4-bit W4A16 within a single-GPU 32GB RTX 5090 VRAM envelope)  
**Coordinate Lenses:** `neuronpedia/jacobian-lens` (`gemma-4-31B_jpp_lens.pt` & `gemma-4-31B_jacobian_lens.pt`)  
**Execution Engine:** `decoderesearch/interp-engine` (LARQL / VINDEX3 naming is borrowed from `chrishayuk/larql`; the vault queries are implemented in Python in `backend/larql_vault.py`, and the `larql`/`vindex` binaries are optional — only their `--version` is reported)  
**References:** Gemma 4 [1], Jacobian lens [2,3], J++ [4], Neuronpedia lens files [6], logit lens [8], neuron poles [9], linear CKA [13]; see `REFERENCES.md`  
**Study & lab:** see `README.md` and `experiments/REPORT.md` (the `experiments/` tree is not shown in the layout below)

---

## 1. Directory Structure

```text
theodolite2/
├── ARCHITECTURE.md                  # System architecture, tensor contracts & wire protocol
├── pyproject.toml                   # Python dependencies & project config
├── backend/
│   ├── __init__.py
│   ├── theodolite_math.py           # Four core vectorized mathematical modules + CKA + unembed
│   ├── engine_runtime.py            # interp-engine + J++ / JLens loader + 4-bit weight dequantizer
│   ├── larql_vault.py               # LARQL / VINDEX3 FFN vault indexer, LQL parser & .patch surgery
│   ├── server.py                    # FastAPI + Binary Float32 WebSocket server + REST endpoints
│   └── mcp_scientist.py             # FastMCP server for autonomous circuit motif discovery
├── scripts/
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
        ├── router.ts                  # #/instrument, #/layer/36, #/neuron/54/15491/pos, #/landmarks, ...
        ├── lib/words.ts               # plain-word helpers (word-pieces, glosses, certainty, layer trust)
        ├── fixtures/e6-demo-sighting.json # the recorded sighting shown when no server answers
        ├── store/
        │   ├── useTheodoliteStore.ts  # Zustand state, WebSocket + HTTP wiring, nudge defaults
        │   └── sighting.ts            # one normalised "sighting" from a live frame pair or the fixture
        ├── components/kit.tsx         # page frame, nav, reading toggles, <Rubric>, <Apparatus>, <Missing>, <NotYet>
        └── screens/                   # Incipit, Instrument, LayerLeaf, NeuronFolio, Landmarks, Nudge, Vault,
                                       # Lexicon, Lineage (+ a folder of parts for each)
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

4. **Projective Theodolite Triangulation (Gram-Schmidt Survey Frame)**:
   - Orthonormal basis $(e_1, e_2, e_3) \in \mathbb{R}^{3 \times 5376}$ in J++-aligned final residual space (or layer-pulled-back space):
     - **Monument Mode**: Given 3 vocabulary tokens $(t_1, t_2, t_3)$, take their unembedding directions $u_m = W_U[t_m]$ (or layer-pulled-back $v_m^{(l)} = W_U[t_m] J_{\text{pp}}[l]$) and orthonormalize in `fp32` via modified Gram-Schmidt to obtain $(e_1, e_2, e_3)$.
     - **Pole-Locked Motif Mode**: Given a selected pole $(l, i, \sigma_i)$, set raw vectors $v_1 = w_{\text{gate}}^{(l, i)}$, $v_2 = (\sigma_i w_{\text{down}}^{(l, i)}) J_{\text{pp}}[l]^\top$, $v_3 = w_{\text{gate}}^{(l+1, j^*)}$ (dominant downstream reader), and orthonormalize via Gram-Schmidt.
   - Project each layer's J++-aligned state $\tilde{h}_l = h_l J_{\text{pp}}[l]^\top$:
     $$x_l = \langle \tilde{h}_l, e_1 \rangle, \quad y_l = \langle \tilde{h}_l, e_2 \rangle, \quad z_l = \langle \tilde{h}_l, e_3 \rangle$$
     $$\theta_l = \text{atan2}(y_l, x_l), \quad \phi_l = \text{asin}\!\left(\frac{z_l}{\sqrt{x_l^2 + y_l^2 + z_l^2 + \epsilon}}\right), \quad \|h_l\|_2 = \sqrt{\sum_{d=1}^{5376} h_{l,d}^2}$$
   - Options (V7 bug 3): `frame_squaring="lowdin"` squares symmetrically, $E = (VV^\top)^{-1/2}V$ on unit rows (order-free);
     `frame_fnw` uses $\text{fnw} \odot W_U[t_m]$; `frame_per_layer` reads the raw $h_l$ in the frame of $W_U[t_m] J_{\text{pp}}[l]$;
     `centre_prompts` subtracts a matched control's state before projecting. `proj_frac` $= \|(x_l, y_l, z_l)\| / \|\tilde{h}_l\|$.

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
   - Generated continuation text, `<|think|>` vs. silent comparison metrics, and active LARQL `.patch` / `SteeringSpec` status.
   - Added without changing the binary layout: `prompt` (the exact text read), `prompt_token_ids`, `chat`, `system`,
     `user_prompt`, `assistant_prefix`, `pole_ranking`, `thread_score`, `frame` (squaring, per-layer, weighting, centring,
     word geometry), `steering_applied`; per layer `proj_frac`, `pole_write_sizes`, `poles_ffn_explained`; per filament
     `direct_gate`, `direct_act`, `score`, `score_kind` (binary column 4 carries `score`, which is C unless
     `thread_score="direct"`).

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

Which indices are sighted: `sight_tokens` if given, else 0..min(n, max_sighted)−1 with `sight_every_token`, else
[0]; never empty ([0] when nothing asked for exists). `prompt`, `system` and `assistant_prefix` are at most 32000
characters, and P + max(max_new_tokens, len(answer_token_ids)) must stay within `MAX_SWEEP_POSITIONS` = 512 (a
capture holds ~5.2 MB per position on the GPU before it moves to the CPU; the thinking prompt is held to the same).
A refused id, control or length is answered in plain words, and so is a `pole_ranking`, `thread_score` or
`frame_squaring` the analysis does not know, before anything is written.

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
`request_id, token_index, sight_index, n_to_sight, last, read_pos, read_token, read_token_id, prompt_len, chosen
(STEP | null), answer, generation, recapture_check, kinship_n_states (= read_pos + 1), rank_lookahead, sight_dropped,
timing {generate_ms, capture_ms, analyse_ms, total_ms}, think_seed, think_read_token, think_stop_reason,
think_skipped` (the last four, and `think_generated_text`, set on the first pair only: `think_stop_reason` is how the
thinking answer's writing ended, "end_mark" | "length" | "stopped", null without one; `think_skipped` is true when a
stop asked before the comparison began left it unwritten, else false); `frame.centre_kind` ("none" |
"control_prompt_end" | "control_reading_same_answer") and, for a pole-motif frame, `frame.reader {layer, neuron, from_token_index}` (built
at the first position analysed and kept for the whole answer); per layer `jpp_top_ids` (5), `naive_top_id`,
`tracked_from_next {word: bool}`.
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

Errors: a request with a `request_id` that is refused or fails gets `{"type":"error","action":"sweep","request_id",
"recoverable":true,"message","n_sent","token_index"}` and then its `theodolite_sweep_end`; the socket stays open
(the request_id of a refused payload is read from the raw message). Without one, the old refusal
`{"type":"error","message","action","recoverable","layer","neuron_idx"}`.

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
  `token_ids` (1–512 ids): read exactly as given, at the last (a sighting's ids up to its read position), echoing
  `token_ids_given`, `read_pos` and `prompt` (the ids decoded); every reply says `steered: false` (a trace reads the
  model without a nudge).
- Saved questions and system prompts (`backend/saved_prompts.py`, declared above the static mount): `GET
  /api/saved_prompts` → `{questions, systems, file, warning}`; `POST` `{kind: "question" | "system", name, question?,
  answer_begins?, chat?, text?, replace_id?}` → the lists (409 for a name already used in that list, trimmed and in any
  case); `DELETE /api/saved_prompts/{kind}/{id}` → the lists (404 when absent). One file outside every checkout
  (`$THEODOLITE_SAVED_PROMPTS`, else `${XDG_DATA_HOME:-~/.local/share}/theodolite/saved_prompts.json`), written
  atomically under its own lock (never the GPU's); an unreadable file is moved aside to `<file>.corrupt-<UTC stamp>` and
  reported once (a file that is not UTF-8, is nested too deeply, or holds a lone surrogate counts as unreadable).
  Requests from a page on another host get 403 (CORS is "*"): the Host must be 127.0.0.1 / localhost / [::1] on any
  port (a DNS-rebinding page sends its own name as the Host, and a same-origin GET sends no Origin; vite's proxy sends
  the server's own Host), and so must the Origin when there is one; no Origin (a script) is allowed. A name or text
  holding half of a character (a lone surrogate) is refused with 422.

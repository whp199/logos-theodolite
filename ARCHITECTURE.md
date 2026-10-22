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
│   ├── label_gate.py                # the label gate: label_check, LABEL_GATE_CARD (which neuron labels pages hold back)
│   ├── causal_check.py              # the causal spot-check: one neuron turned up at one word-piece (N1A's protocol)
│   ├── server.py                    # FastAPI + Binary Float32 WebSocket server + REST endpoints
│   └── mcp_scientist.py             # FastMCP server for autonomous circuit motif discovery
├── scripts/
│   ├── check_causal_check.py        # the causal spot-check on a tiny CPU Gemma 4 (CPU only, run first by verify_e2e)
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
        ├── router.ts                  # #/instrument, #/layer/36, #/neuron/54/15491/pos, #/landmarks, ...
        ├── lib/words.ts               # plain-word helpers (word-pieces, glosses, certainty)
        ├── lib/labels.ts              # the label gate on the pages (labelGate, trustLine, held-back wording)
        ├── components/HeldBack.tsx    # a held-back label's plate, reason and rubric; its words in the apparatus
        ├── fixtures/recordedLabelChecks.ts # the recorded labels' checks (written by N1G_01 from its results)
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
`frame_squaring` or `think_bare` the analysis does not know, before anything is written.

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
  `label_gate` before `lens` (UPDATE, INSERT, DESCRIBE, DELETE and ROLLBACK reply as before);
  `trace_downstream_readers` (`trace_readers`) adds to each reader returned `dst_naive_tokens` (four, as through the
  lens) and `dst_label_check`, and `label_gate` to the reply; `test_motif_hypothesis` adds `pole_label_check` (its
  pole's face) and `label_gate`.

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
  Requests from a page on another host get 403 (CORS is "*"): the Host must be 127.0.0.1 / localhost / [::1] on any
  port (a DNS-rebinding page sends its own name as the Host, and a same-origin GET sends no Origin; vite's proxy sends
  the server's own Host), and so must the Origin when there is one; no Origin (a script) is allowed. A name or text
  holding half of a character (a lone surrogate) is refused with 422.

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
`_GPU_LOCK` the route holds. If the answer before cannot be written, nothing is edited and the statement is refused
in plain words (HTTP 400; over the WebSocket `larql_result` ok false, as is a voice the request model refuses, which
over HTTP is 422). If the answer after cannot be written, the edit stays in force with `after` null and `error` set.
The MCP tool `query_larql_vault` takes no voice, since it does not hold the GPU lock.

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
shows what the edit did to that answer only. scripts/check_vault_voice.py checks all of this on a stand-in runtime.

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
    └── src/
        ├── main.tsx
        ├── App.tsx
        ├── store/
        │   └── useTheodoliteStore.ts # Zustand state & Binary WebSocket frame decoder
        └── components/
            ├── AstrolabeScene.tsx    # React Three Fiber 60-layer 3D astrolabe canvas
            ├── LayerRings.tsx        # 60 horizontal glass rings (L6..L60 Global Attention in gold/cyan)
            ├── BallisticTrajectory.tsx # Token trajectory, delta_attn_J (cyan) & delta_ffn_J (gold), plumes
            ├── DipoleFilaments.tsx   # Oriented Bipolar GeGLU needles (+ North / - South) & reader filaments
            └── ReticleHUD.tsx        # Theodolite HUD: Azimuth θ, Elevation φ, ||h_l||_2, 60x60 CKA, Laser Steer, LARQL console
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

4. **Projective Theodolite Triangulation (Gram-Schmidt Survey Frame)**:
   - Orthonormal basis $(e_1, e_2, e_3) \in \mathbb{R}^{3 \times 5376}$ in J++-aligned final residual space (or layer-pulled-back space):
     - **Monument Mode**: Given 3 vocabulary tokens $(t_1, t_2, t_3)$, take their unembedding directions $u_m = W_U[t_m]$ (or layer-pulled-back $v_m^{(l)} = W_U[t_m] J_{\text{pp}}[l]$) and orthonormalize in `fp32` via modified Gram-Schmidt to obtain $(e_1, e_2, e_3)$.
     - **Pole-Locked Motif Mode**: Given a selected pole $(l, i, \sigma_i)$, set raw vectors $v_1 = w_{\text{gate}}^{(l, i)}$, $v_2 = (\sigma_i w_{\text{down}}^{(l, i)}) J_{\text{pp}}[l]^\top$, $v_3 = w_{\text{gate}}^{(l+1, j^*)}$ (dominant downstream reader), and orthonormalize via Gram-Schmidt.
   - Project each layer's J++-aligned state $\tilde{h}_l = h_l J_{\text{pp}}[l]^\top$:
     $$x_l = \langle \tilde{h}_l, e_1 \rangle, \quad y_l = \langle \tilde{h}_l, e_2 \rangle, \quad z_l = \langle \tilde{h}_l, e_3 \rangle$$
     $$\theta_l = \text{atan2}(y_l, x_l), \quad \phi_l = \text{asin}\!\left(\frac{z_l}{\sqrt{x_l^2 + y_l^2 + z_l^2 + \epsilon}}\right), \quad \|h_l\|_2 = \sqrt{\sum_{d=1}^{5376} h_{l,d}^2}$$

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

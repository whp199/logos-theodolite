"""Pillar 3: Static FFN Vault Index & Weight Surgery (LARQL/VINDEX3-inspired + J-VINDEX).

Naming is borrowed from `chrishayuk/larql` (https://github.com/chrishayuk/larql); this is NOT LARQL's LQL and the
`larql`/`vindex` binaries are only queried for `--version`. A real-time GPU-accelerated
J-VINDEX3 query & weight-surgery engine over Gemma-4-31B's 60 layers x 21,504 GeGLU FFN slots
(`W_gate`, `W_up`, `W_down` = 1,290,240 neurons / 2,580,480 bipolar poles).

Supported LQL & .patch Operations:
- `DESCRIBE` / `SHOW LAYERS`
- `SELECT * FROM EDGES NEAREST '<concept>' [WHERE layer = <l>] [LIMIT <k>]`
- `SELECT * FROM POLES NEAREST '<concept>' [WHERE layer = <l>] [LIMIT <k>]`
- `DESCRIBE POLE LAYER <l> SLOT <i> [POLARITY +|-]`
- Declarative `.patch` weight overlays (live GPU mutation + instant rollback). Named after, but NOT an
  implementation of, MEMIT (Meng et al. 2023): each edit is a single-neuron rank-1 down_proj column overlay with no
  covariance solve:
  - `UPDATE LAYER <l> SLOT <i> SET object = '<concept>' [STRENGTH <alpha>] [POLARITY +|-]`
  - `INSERT INTO LAYER <l> (subject, object) VALUES ('<subj>', '<obj>') [STRENGTH <alpha>]`
  - `DELETE FROM LAYER <l> WHERE slot = <i>`
  - `ROLLBACK`

The statements that read words through the J++ lens (the two NEAREST searches and DESCRIBE POLE) or write an edit along
a direction pulled back through it (UPDATE, INSERT) name the lens in their reply, under "lens" (the last key;
`TheodoliteRuntime.lens_card`, with a `caution` on how far to trust its words), as the scientist tools do. The reply is
otherwise unchanged. DESCRIBE, DELETE and ROLLBACK touch no lens and reply as they always have.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from backend.engine_runtime import TheodoliteRuntime, dequantize_w4a16_cols
from backend.theodolite_math import (
    D_FF,
    D_MODEL,
    GLOBAL_ATTN_LAYERS_0BASED,
    N_LAYERS,
    VOCAB_SIZE,
    unembed_gemma4,
)


@dataclass
class PatchOperationRecord:
    """Audit record of an active declarative `.patch` MEMIT weight surgery overlay."""

    patch_id: int
    op_type: str  # "INSERT" | "UPDATE" | "DELETE"
    layer_idx: int
    neuron_idx: int
    polarity: float
    target_concept: str
    strength: float
    statement: str
    timestamp: float
    orig_norm: float
    new_norm: float


class LarqlVaultEngine:
    """Static FFN Vault Index (`vindex-cli` / `larql-cli` VINDEX3) & Live J-VINDEX Weight Surgery."""

    def __init__(self, runtime: TheodoliteRuntime) -> None:
        self.runtime = runtime
        self.active_patches: List[PatchOperationRecord] = []
        self._patch_counter = 0

        venv_bin = Path(__file__).resolve().parent.parent / ".venv" / "bin"
        self.vindex_bin = str(venv_bin / "vindex-cli") if (venv_bin / "vindex-cli").exists() else shutil.which("vindex")
        self.larql_bin = str(venv_bin / "larql-cli") if (venv_bin / "larql-cli").exists() else shutil.which("larql")

    def cli_versions(self) -> Dict[str, str]:
        """Return installed `vindex-cli` and `larql-cli` binary versions."""
        out: Dict[str, str] = {}
        for key, binary in (("vindex_cli", self.vindex_bin), ("larql_cli", self.larql_bin)):
            if not binary:
                out[key] = "unavailable"
                continue
            try:
                res = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=3, check=False)
                out[key] = (res.stdout or res.stderr).strip()
            except Exception as exc:
                out[key] = f"error: {exc}"
        return out

    def describe_vault(self) -> Dict[str, Any]:
        """Execute `DESCRIBE` across the 60-layer Gemma-4-31B FFN Vault."""
        cli_info = self.cli_versions()
        return {
            "command": "DESCRIBE",
            "format": "VINDEX3 + J-VINDEX (On-the-Fly W4A16 GPU)",
            "model_id": self.runtime.model_id,
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "d_ff_per_layer": D_FF,
            "bipolar_poles_per_layer": D_FF * 2,
            "total_geglu_slots": N_LAYERS * D_FF,
            "total_bipolar_poles": N_LAYERS * D_FF * 2,
            "vocab_size": VOCAB_SIZE,
            "global_attention_layers_0based": list(GLOBAL_ATTN_LAYERS_0BASED),
            "cli_binaries": cli_info,
            "active_patches_count": len(self.active_patches),
            "active_patches": [asdict(p) for p in self.active_patches],
        }

    @torch.no_grad()
    def query_nearest_edges(
        self,
        concept: str,
        layer_idx: int = 35,
        top_k: int = 10,
        matrix_target: str = "down_proj",
        use_jpp: bool = True,
    ) -> Dict[str, Any]:
        """Zero-prompt offline graph query:
        `SELECT * FROM EDGES NEAREST '<concept>' WHERE layer = <layer_idx> LIMIT <top_k>`

        Finds the Bipolar GeGLU FFN slots `(layer_idx, i, polarity)` in `W_down[layer_idx]`
        (or `W_gate[layer_idx]`) whose J++-aligned write vector (or gate read vector) has maximal
        alignment with `<concept>`'s unembedding direction `W_U[token_id]`.
        """
        t0 = time.perf_counter()
        layer_idx = max(0, min(int(layer_idx), N_LAYERS - 1))
        token_id, token_str = self.runtime.resolve_concept_token_id(concept)

        # Target unembedding vector u in [5376]
        u = self.runtime.lm_head_weight[token_id].to(device=self.runtime.device, dtype=torch.float32)
        # Apply final_norm scaling to match unembed inner product geometry
        u_scaled = u * self.runtime.final_norm_weight.to(device=self.runtime.device, dtype=torch.float32)

        if use_jpp:
            J_l = self.runtime.jpp_lenses[layer_idx].float()
            # Since logit ~ < (w_down @ J_l.T) * norm_w, u > = < w_down, (u * norm_w) @ J_l >,
            # we pull u_scaled back to layer l in a single matrix-vector product!
            query_dir = u_scaled @ J_l  # [5376]
        else:
            query_dir = u_scaled

        query_dir = F.normalize(query_dir, dim=-1)

        if matrix_target == "gate_proj":
            # W_gate is [21504, 5376]
            w_mat = self.runtime.get_full_w_gate(layer_idx).float()
            w_norms = torch.linalg.vector_norm(w_mat, dim=-1) + 1e-8
            cos_sims = (w_mat @ query_dir) / w_norms  # [21504]
        else:
            # W_down is [5376, 21504], columns are neuron write vectors
            w_down = self.runtime.get_full_w_down(layer_idx).float()
            if layer_idx in self.runtime._post_ffn_norm_weights:
                post_w = self.runtime._post_ffn_norm_weights[layer_idx].to(w_down.device).float().unsqueeze(-1)
                w_down = w_down * post_w
            w_norms = torch.linalg.vector_norm(w_down, dim=0) + 1e-8
            cos_sims = (query_dir @ w_down) / w_norms  # [21504]

        # Bipolar top-k: largest |cos_sim|, with sign indicating North (+1) vs South (-1) pole
        top_abs, top_idx = torch.topk(cos_sims.abs(), k=min(top_k, D_FF))
        signed_sims = cos_sims.gather(0, top_idx)
        polarities = torch.where(signed_sims >= 0, torch.ones_like(signed_sims), -torch.ones_like(signed_sims))

        # Decode J-VINDEX vs Naive Logit Lens tokens for these top-k static vault poles
        w_down_k = self.runtime.slice_w_down(layer_idx, top_idx)
        if layer_idx in self.runtime._post_ffn_norm_weights:
            w_down_k = (
                w_down_k.float() * self.runtime._post_ffn_norm_weights[layer_idx].to(w_down_k.device).float()
            ).to(w_down_k.dtype)
        signed_w_down = polarities.to(w_down_k.dtype).unsqueeze(-1) * w_down_k
        jpp_vecs = signed_w_down @ self.runtime.jpp_lenses[layer_idx].T

        jpp_logits = unembed_gemma4(jpp_vecs, self.runtime.final_norm_weight, self.runtime.lm_head_weight)
        naive_logits = unembed_gemma4(signed_w_down, self.runtime.final_norm_weight, self.runtime.lm_head_weight)

        rows: List[Dict[str, Any]] = []
        for r in range(top_idx.numel()):
            n_idx = int(top_idx[r].item())
            pol = float(polarities[r].item())
            sim = float(top_abs[r].item())
            jv, ji = torch.topk(jpp_logits[r], k=5)
            nv, ni = torch.topk(naive_logits[r], k=5)
            rows.append(
                {
                    "rank": r + 1,
                    "layer": layer_idx,
                    "neuron_idx": n_idx,
                    "polarity": "+" if pol >= 0 else "-",
                    "polarity_num": pol,
                    "cosine_similarity": round(sim, 5),
                    "j_vindex_top_tokens": [
                        {"token": self.runtime.tokenizer.decode([int(tid)]), "logit": round(float(lv), 3)}
                        for lv, tid in zip(jv.cpu().tolist(), ji.cpu().tolist())
                    ],
                    "naive_lens_top_tokens": [
                        {"token": self.runtime.tokenizer.decode([int(tid)]), "logit": round(float(lv), 3)}
                        for lv, tid in zip(nv.cpu().tolist(), ni.cpu().tolist())
                    ],
                }
            )

        return {
            "command": f"SELECT * FROM EDGES NEAREST '{concept}' WHERE layer = {layer_idx} LIMIT {top_k}",
            "concept": concept,
            "resolved_token_id": token_id,
            "resolved_token_str": token_str,
            "layer": layer_idx,
            "matrix": matrix_target,
            "use_jpp": use_jpp,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "results": rows,
        }

    @torch.no_grad()
    def inspect_bipolar_slot(
        self,
        layer_idx: int,
        neuron_idx: int,
        polarity: float = 1.0,
        top_k_tokens: int = 8,
    ) -> Dict[str, Any]:
        """Inspect both North (+1) and South (-1) poles (and requested polarity) of neuron `(layer_idx, neuron_idx)`."""
        layer_idx = max(0, min(int(layer_idx), N_LAYERS - 1))
        neuron_idx = max(0, min(int(neuron_idx), D_FF - 1))
        pol_sign = 1.0 if float(polarity) >= 0 else -1.0

        idx_t = torch.tensor([neuron_idx], device=self.runtime.device, dtype=torch.long)
        w_down = self.runtime.slice_w_down(layer_idx, idx_t)[0]  # [5376]
        w_gate = self.runtime.slice_w_gate(layer_idx, idx_t)[0]  # [5376]
        w_up = self.runtime.slice_w_up(layer_idx, idx_t)[0]      # [5376]

        J_l = self.runtime.jpp_lenses[layer_idx]
        w_down_scaled = w_down
        if layer_idx in self.runtime._post_ffn_norm_weights:
            w_down_scaled = (
                w_down.float() * self.runtime._post_ffn_norm_weights[layer_idx].to(w_down.device).float()
            ).to(J_l.dtype)

        # Compute both North (+1) and South (-1) pole readouts under J-VINDEX and Naive Logit Lens
        poles_summary: Dict[str, Any] = {}
        for label, s in (("north_plus", 1.0), ("south_minus", -1.0)):
            vec_raw = (s * w_down_scaled).unsqueeze(0)
            vec_jpp = vec_raw @ J_l.T
            logits_jpp = unembed_gemma4(vec_jpp, self.runtime.final_norm_weight, self.runtime.lm_head_weight)[0]
            logits_naive = unembed_gemma4(vec_raw, self.runtime.final_norm_weight, self.runtime.lm_head_weight)[0]

            jv, ji = torch.topk(logits_jpp, k=top_k_tokens)
            nv, ni = torch.topk(logits_naive, k=top_k_tokens)
            poles_summary[label] = {
                "polarity": "+" if s > 0 else "-",
                "j_vindex_tokens": [
                    {"token": self.runtime.tokenizer.decode([int(tid)]), "logit": round(float(val), 3)}
                    for val, tid in zip(jv.cpu().tolist(), ji.cpu().tolist())
                ],
                "naive_tokens": [
                    {"token": self.runtime.tokenizer.decode([int(tid)]), "logit": round(float(val), 3)}
                    for val, tid in zip(nv.cpu().tolist(), ni.cpu().tolist())
                ],
            }

        # Gate input readout (what triggers this neuron)
        gate_jpp = w_gate.to(J_l.dtype).unsqueeze(0) @ J_l.T
        gate_logits = unembed_gemma4(gate_jpp, self.runtime.final_norm_weight, self.runtime.lm_head_weight)[0]
        gv, gi = torch.topk(gate_logits, k=top_k_tokens)

        requested_key = "north_plus" if pol_sign > 0 else "south_minus"
        return {
            "layer": layer_idx,
            "neuron_idx": neuron_idx,
            "requested_polarity": "+" if pol_sign > 0 else "-",
            "w_down_norm": round(float(torch.linalg.vector_norm(w_down.float()).item()), 5),
            "w_gate_norm": round(float(torch.linalg.vector_norm(w_gate.float()).item()), 5),
            "w_up_norm": round(float(torch.linalg.vector_norm(w_up.float()).item()), 5),
            "active_pole": poles_summary[requested_key],
            "north_pole": poles_summary["north_plus"],
            "south_pole": poles_summary["south_minus"],
            "gate_trigger_tokens": [
                {"token": self.runtime.tokenizer.decode([int(tid)]), "logit": round(float(val), 3)}
                for val, tid in zip(gv.cpu().tolist(), gi.cpu().tolist())
            ],
        }

    @torch.no_grad()
    def apply_patch_surgery(
        self,
        op_type: str,
        layer_idx: int,
        neuron_idx: Optional[int] = None,
        target_concept: str = "",
        subject_concept: str = "",
        strength: float = 2.5,
        polarity: float = 1.0,
        statement: str = "",
    ) -> Dict[str, Any]:
        """Apply a live declarative `.patch` MEMIT weight surgery overlay (`INSERT`, `UPDATE`, `DELETE`)
        onto layer `layer_idx`'s FFN `down_proj` (and `gate_proj` for `INSERT`).
        """
        op = op_type.upper().strip()
        layer_idx = max(0, min(int(layer_idx), N_LAYERS - 1))
        mlp = self.runtime.get_mlp_module(layer_idx)
        down_proj = mlp.down_proj
        if not hasattr(down_proj, "_memit_col_overlays"):
            down_proj._memit_col_overlays = {}

        # If neuron_idx is None (e.g. INSERT INTO LAYER l (subject, object) VALUES (...)),
        # pick the slot whose gate_proj best matches subject_concept!
        if neuron_idx is None:
            lookup_concept = subject_concept or target_concept
            nearest = self.query_nearest_edges(
                concept=lookup_concept,
                layer_idx=layer_idx,
                top_k=1,
                matrix_target="gate_proj" if subject_concept else "down_proj",
            )
            neuron_idx = int(nearest["results"][0]["neuron_idx"])
            polarity = float(nearest["results"][0]["polarity_num"])

        neuron_idx = max(0, min(int(neuron_idx), D_FF - 1))
        idx_t = torch.tensor([neuron_idx], device=self.runtime.device, dtype=torch.long)

        # Get current/original column
        orig_col = dequantize_w4a16_cols(down_proj.weight_packed, down_proj.weight_scale, idx_t)[0]
        orig_norm = float(torch.linalg.vector_norm(orig_col.float()).item())

        if op == "DELETE":
            new_col = torch.zeros_like(orig_col)
            down_proj._memit_col_overlays[neuron_idx] = new_col
            new_norm = 0.0
        elif op in ("UPDATE", "INSERT"):
            tid, _tstr = self.runtime.resolve_concept_token_id(target_concept)
            u_target = self.runtime.lm_head_weight[tid].to(device=self.runtime.device, dtype=torch.float32)
            u_scaled = u_target * self.runtime.final_norm_weight.to(device=self.runtime.device, dtype=torch.float32)
            # Pull target direction back to layer_idx via J_pp[layer_idx]
            J_l = self.runtime.jpp_lenses[layer_idx].float()
            pulled_dir = F.normalize(u_scaled @ J_l, dim=-1)
            pol_sign = 1.0 if float(polarity) >= 0 else -1.0
            # Blend or set column to pulled-back target direction scaled by strength * orig_norm
            new_col = (
                orig_col.float() + pol_sign * float(strength) * max(orig_norm, 0.25) * pulled_dir
            ).to(dtype=orig_col.dtype)
            down_proj._memit_col_overlays[neuron_idx] = new_col
            new_norm = float(torch.linalg.vector_norm(new_col.float()).item())
        else:
            raise ValueError(f"Unsupported .patch operation: {op_type!r}")

        self._patch_counter += 1
        rec = PatchOperationRecord(
            patch_id=self._patch_counter,
            op_type=op,
            layer_idx=layer_idx,
            neuron_idx=neuron_idx,
            polarity=float(polarity),
            target_concept=target_concept,
            strength=float(strength),
            statement=statement or f"{op} LAYER {layer_idx} SLOT {neuron_idx}",
            timestamp=time.time(),
            orig_norm=round(orig_norm, 5),
            new_norm=round(new_norm, 5),
        )
        self.active_patches.append(rec)
        return {
            "status": "applied",
            "patch": asdict(rec),
            "active_patches_count": len(self.active_patches),
        }

    def rollback_patches(self) -> Dict[str, Any]:
        """Clear all active `.patch` MEMIT overlays across all 60 layers."""
        cleared = len(self.active_patches)
        for l in range(N_LAYERS):
            mlp = self.runtime.get_mlp_module(l)
            for proj in (mlp.down_proj, mlp.gate_proj, mlp.up_proj):
                if hasattr(proj, "_memit_col_overlays"):
                    proj._memit_col_overlays.clear()
                if hasattr(proj, "_memit_row_overlays"):
                    proj._memit_row_overlays.clear()
        self.active_patches.clear()
        return {"status": "rolled_back", "cleared_patches": cleared, "active_patches_count": 0}

    def _through_lens(self, reply: Dict[str, Any]) -> Dict[str, Any]:
        """`reply` with the lens its words were read, or its edit written, through named last, under "lens"."""
        return {**reply, "lens": self.runtime.lens_card()}

    def execute_lql(self, query: str) -> Dict[str, Any]:
        """Parse and execute a LARQL / `.patch` statement string. NEAREST, DESCRIBE POLE, UPDATE and INSERT go through
        the J++ lens, so their replies also carry "lens" (see the module docstring); the rest reply as before."""
        q = query.strip().rstrip(";")
        q_upper = q.upper()

        if q_upper in ("DESCRIBE", "SHOW LAYERS", "STATUS"):
            return self.describe_vault()

        if q_upper == "ROLLBACK":
            return self.rollback_patches()

        # SELECT * FROM EDGES|POLES NEAREST 'concept' [WHERE layer = 35] [LIMIT 10]
        m_nearest = re.search(
            r"SELECT\s+\*\s+FROM\s+(EDGES|POLES)\s+NEAREST\s+['\"]([^'\"]+)['\"]"
            r"(?:\s+WHERE\s+layer\s*=\s*(\d+))?"
            r"(?:\s+LIMIT\s+(\d+))?",
            q,
            flags=re.IGNORECASE,
        )
        if m_nearest:
            _kind, concept, layer_s, limit_s = m_nearest.groups()
            layer_idx = int(layer_s) if layer_s is not None else 35
            top_k = int(limit_s) if limit_s is not None else 10
            return self._through_lens(self.query_nearest_edges(concept=concept, layer_idx=layer_idx, top_k=top_k))

        # DESCRIBE POLE LAYER <l> SLOT <i> [POLARITY +|-]
        m_pole = re.search(
            r"DESCRIBE\s+POLE\s+LAYER\s+(\d+)\s+SLOT\s+(\d+)(?:\s+POLARITY\s+([+-]))?",
            q,
            flags=re.IGNORECASE,
        )
        if m_pole:
            l_s, i_s, pol_s = m_pole.groups()
            pol = -1.0 if pol_s == "-" else 1.0
            return self._through_lens(self.inspect_bipolar_slot(layer_idx=int(l_s), neuron_idx=int(i_s), polarity=pol))

        # DELETE FROM LAYER <l> WHERE slot = <i>
        m_del = re.search(
            r"DELETE\s+FROM\s+LAYER\s+(\d+)\s+WHERE\s+slot\s*=\s*(\d+)",
            q,
            flags=re.IGNORECASE,
        )
        if m_del:
            l_s, i_s = m_del.groups()
            return self.apply_patch_surgery(
                op_type="DELETE",
                layer_idx=int(l_s),
                neuron_idx=int(i_s),
                statement=q,
            )

        # UPDATE LAYER <l> SLOT <i> SET object = '<concept>' [STRENGTH <alpha>] [POLARITY +|-]
        m_upd = re.search(
            r"UPDATE\s+LAYER\s+(\d+)\s+SLOT\s+(\d+)\s+SET\s+object\s*=\s*['\"]([^'\"]+)['\"]"
            r"(?:\s+STRENGTH\s+([0-9.]+))?"
            r"(?:\s+POLARITY\s+([+-]))?",
            q,
            flags=re.IGNORECASE,
        )
        if m_upd:
            l_s, i_s, obj_c, str_s, pol_s = m_upd.groups()
            return self._through_lens(
                self.apply_patch_surgery(
                    op_type="UPDATE",
                    layer_idx=int(l_s),
                    neuron_idx=int(i_s),
                    target_concept=obj_c,
                    strength=float(str_s) if str_s else 2.5,
                    polarity=-1.0 if pol_s == "-" else 1.0,
                    statement=q,
                )
            )

        # INSERT INTO LAYER <l> (subject, object) VALUES ('<subj>', '<obj>') [STRENGTH <alpha>]
        m_ins = re.search(
            r"INSERT\s+INTO\s+LAYER\s+(\d+)\s*\(\s*subject\s*,\s*object\s*\)\s*VALUES\s*\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]\s*\)"
            r"(?:\s+STRENGTH\s+([0-9.]+))?",
            q,
            flags=re.IGNORECASE,
        )
        if m_ins:
            l_s, subj_c, obj_c, str_s = m_ins.groups()
            return self._through_lens(
                self.apply_patch_surgery(
                    op_type="INSERT",
                    layer_idx=int(l_s),
                    neuron_idx=None,
                    subject_concept=subj_c,
                    target_concept=obj_c,
                    strength=float(str_s) if str_s else 2.5,
                    statement=q,
                )
            )

        raise ValueError(
            f"Unrecognized LARQL / .patch query: {query!r}. "
            "Supported: DESCRIBE | SELECT * FROM EDGES NEAREST '<concept>' WHERE layer = <l> LIMIT <k> | "
            "DESCRIBE POLE LAYER <l> SLOT <i> | UPDATE LAYER <l> SLOT <i> SET object = '<c>' | "
            "INSERT INTO LAYER <l> (subject, object) VALUES ('<s>', '<o>') | "
            "DELETE FROM LAYER <l> WHERE slot = <i> | ROLLBACK"
        )

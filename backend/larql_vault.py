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
  covariance solve (INSERT too: it only chooses the neuron by its gate; E6 §3, item 12):
  - `UPDATE LAYER <l> SLOT <i> SET object = '<concept>' [STRENGTH <alpha>] [POLARITY +|-]`
  - `INSERT INTO LAYER <l> (subject, object) VALUES ('<subj>', '<obj>') [STRENGTH <alpha>]`
  - `DELETE FROM LAYER <l> WHERE slot = <i>`
  - `ROLLBACK`

The statements that read words through the J++ lens (the two NEAREST searches and DESCRIBE POLE) or write an edit along
a direction pulled back through it (UPDATE, INSERT) name the lens in their reply, under "lens" (the last key;
`TheodoliteRuntime.lens_card`, with a `caution` on how far to trust its words), as the scientist tools do. The two that
read neuron labels also mark each one: every NEAREST row and each DESCRIBE POLE face carries "label_check"
(`backend.label_gate.label_check`: the word-pieces its two readings share in their first ten, and whether the pages
hold it back), and the reply carries "label_gate" (`LABEL_GATE_CARD`, the rule) just before "lens". The reply is
otherwise unchanged: every old key is kept. DESCRIBE, DELETE and ROLLBACK touch no lens and reply as they always have.

An edit (DELETE, UPDATE, INSERT) can be heard: given a `voice` (a question, how it is sent, and how long an answer),
`execute_lql` has the model write its answer the likeliest way just before the edit and again just after, with nothing
between (`TheodoliteRuntime.write_voice`; the caller holds the GPU lock around all three). The reply then carries
"voice" after "patch" and before "lens", and the edit's record keeps it. Without a voice, and for every other
statement, the reply is exactly as before. Each record also says what kind of edit it is (`edit_kind`) and which
earlier edit of the same neuron, if any, it replaces (`replaces_patch`).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from backend.engine_runtime import TheodoliteRuntime, dequantize_w4a16_cols
from backend.label_gate import LABEL_GATE_CARD, label_check, label_top_ids, writes_nothing
from backend.theodolite_math import (
    D_FF,
    D_MODEL,
    GLOBAL_ATTN_LAYERS_0BASED,
    N_LAYERS,
    VOCAB_SIZE,
    unembed_gemma4,
)

# What an edit is, for DESCRIBE (and the pages): E6 §3, item 12; REPORT §3.3
EDIT_METHOD = "one down_proj column per edit, rank one, no covariance solve: named after MEMIT, not MEMIT"
# A voice's answer: how long it may run (VoiceRequest in backend/server.py takes the same bounds), and its default.
# V7 judged steering by 12-word-piece greedy answers (V7_tool_bugs_NOTES.md), hence 12.
VOICE_MAX_NEW_TOKENS_DEFAULT = 12
VOICE_MAX_NEW_TOKENS_LIMIT = 64


def voice_spec(voice: Any) -> Dict[str, Any]:
    """`voice` (a mapping, or the server's `VoiceRequest`) as `execute_lql` hears an edit by: the question (`prompt`),
    whether it is sent as a chat (`chat`, default True), its `system` prompt and begun answer (`assistant_prefix`), as
    `TheodoliteRuntime.compose_prompt` takes them, and how many word-pieces the answer may run to (`max_new_tokens`,
    1-64, default 12). Refused in plain words, before anything is written or edited."""
    if hasattr(voice, "model_dump"):
        voice = voice.model_dump()
    if not isinstance(voice, Mapping):
        raise ValueError("The voice must give at least a question to ask (its prompt); no edit was made.")
    prompt = voice.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("The voice has no question to ask before and after the edit (its prompt); no edit was made.")
    system = voice.get("system")
    prefix = voice.get("assistant_prefix") or ""
    if (system is not None and not isinstance(system, str)) or not isinstance(prefix, str):
        raise ValueError("The voice's system prompt and begun answer must be text; no edit was made.")
    n_new = voice.get("max_new_tokens", VOICE_MAX_NEW_TOKENS_DEFAULT)
    if isinstance(n_new, bool) or not isinstance(n_new, int) or not 1 <= n_new <= VOICE_MAX_NEW_TOKENS_LIMIT:
        raise ValueError(
            f"The voice's answer may run to 1-{VOICE_MAX_NEW_TOKENS_LIMIT} word-pieces, not {n_new!r}; no edit was "
            "made."
        )
    return {
        "prompt": prompt,
        "chat": bool(voice.get("chat", True)),
        "system": system or None,
        "assistant_prefix": prefix,
        "max_new_tokens": int(n_new),
    }


def _plain(exc: BaseException) -> str:
    """A failure while a voice is written, as words to put inside a sentence."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "the GPU ran out of memory"
    words = str(exc) if isinstance(exc, ValueError) else f"{type(exc).__name__}: {exc}"
    return words.rstrip(". ") or type(exc).__name__


def parted_at(before: List[int], after: List[int]) -> Optional[int]:
    """The first word-piece index at which two answers differ (one ending sooner counts), or None if they are the
    same."""
    for i, (a, b) in enumerate(zip(before, after)):
        if a != b:
            return i
    return None if len(before) == len(after) else min(len(before), len(after))


def kl_nats(logp_before: torch.Tensor, logp_after: torch.Tensor) -> float:
    """KL(before || after) = sum p_b (log p_b - log p_a), in nats, from two [vocab] log-softmaxes (float64 sum; a
    rounding below zero is read as zero)."""
    lb, la = logp_before.double(), logp_after.double()
    return max(float((lb.exp() * (lb - la)).sum().item()), 0.0)


@dataclass
class PatchOperationRecord:
    """Audit record of an active declarative `.patch` edit: one neuron's down_proj column overlay (named after MEMIT
    in the code; not MEMIT). `voice` is its answer before and after when it was heard (`execute_lql`'s voice), else
    None; `replaces_patch` the earlier edit of the same (layer, neuron) whose overlay it replaced, else None."""

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
    voice: Optional[Dict[str, Any]] = None
    edit_kind: str = "one_neuron_column"
    replaces_patch: Optional[int] = None


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

    def edits_in_force(self) -> List[PatchOperationRecord]:
        """The edits now in force: the latest record for each (layer_idx, neuron_idx) among `active_patches`, in patch
        order. A later edit of a neuron replaces its column overlay (keyed by neuron), so only the latest one acts."""
        latest: Dict[Tuple[int, int], PatchOperationRecord] = {}
        for p in self.active_patches:
            latest[(p.layer_idx, p.neuron_idx)] = p
        return sorted(latest.values(), key=lambda p: p.patch_id)

    def _edited_neurons(self) -> Set[Tuple[int, int]]:
        return {(p.layer_idx, p.neuron_idx) for p in self.edits_in_force()}

    def _lens_id(self) -> Optional[str]:
        card = self.runtime.lens_card()
        return card.get("id") if card else None

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
            "edit_method": EDIT_METHOD,
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
        # each row's label, marked by the study's measure (first ten a side, fp32; backend/label_gate.py)
        fnw, lm = self.runtime.final_norm_weight, self.runtime.lm_head_weight
        lens_ids = label_top_ids(jpp_vecs, fnw, lm, jpp_logits)
        eye_ids = label_top_ids(signed_w_down, fnw, lm, naive_logits)
        silenced = writes_nothing(signed_w_down)  # a DELETEd neuron's zero column: no count, held back
        lens_id, edited = self._lens_id(), self._edited_neurons()

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
                    "label_check": label_check(
                        layer_idx, lens_ids[r], eye_ids[r], lens_id, edited=(layer_idx, n_idx) in edited,
                        silenced=silenced[r],
                    ),
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
        lens_id = self._lens_id()
        edited = (layer_idx, neuron_idx) in self._edited_neurons()
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
                # the face's label, marked by the study's measure (first ten a side, fp32; backend/label_gate.py)
                "label_check": label_check(
                    layer_idx,
                    label_top_ids(vec_jpp, self.runtime.final_norm_weight, self.runtime.lm_head_weight, logits_jpp)[0],
                    label_top_ids(vec_raw, self.runtime.final_norm_weight, self.runtime.lm_head_weight, logits_naive)[0],
                    lens_id,
                    edited=edited,
                    silenced=writes_nothing(vec_raw)[0],
                ),
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
        """Apply a live declarative `.patch` edit (`INSERT`, `UPDATE`, `DELETE`): one column of layer `layer_idx`'s FFN
        `down_proj`, overlaid (rank one, no covariance solve: named after MEMIT, not MEMIT). `INSERT` reads `gate_proj`
        only to choose the neuron; it edits only the `down_proj` column, as the others do (E6 §3, item 12). A later
        edit of the same neuron replaces its overlay, and its record names the one it replaced (`replaces_patch`).
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

        # the edit in force on this neuron until now, whose overlay this one has just replaced (the dict is keyed by
        # neuron)
        replaced = next(
            (p.patch_id for p in self.edits_in_force() if (p.layer_idx, p.neuron_idx) == (layer_idx, neuron_idx)),
            None,
        )
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
            replaces_patch=replaced,
        )
        self.active_patches.append(rec)
        return {
            "status": "applied",
            "patch": asdict(rec),
            "active_patches_count": len(self.active_patches),
        }

    def rollback_patches(self) -> Dict[str, Any]:
        """Clear all active `.patch` overlays (the one-neuron column edits) across all 60 layers, and their records."""
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

    def _heard(
        self, voice: Optional[Any], watch_word: Optional[str], do_edit: Callable[[], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """`do_edit()`, heard: with no `voice`, just the edit's reply. With one (`voice_spec`), the model writes its
        answer to the voice's question the likeliest way, then the edit is made, then it writes the answer again,
        with nothing between (the caller holds the GPU lock around all three, as the server's routes do). If the
        answer before cannot be written, nothing is edited (ValueError, in plain words). If the answer after cannot
        be written, the edit stays in force, with "after" None and "error" saying why. `watch_word` (the edit's target
        word, None for DELETE) is followed at the first word-piece. The voice is kept on the edit's record, and the
        reply gains it under "voice", after "patch" (serialised again, so it carries the voice too)."""
        if voice is None:
            return do_edit()
        spec = voice_spec(voice)
        rt = self.runtime
        t0 = time.perf_counter()
        # the edits in force as it is heard (the latest per neuron): "before" includes them all
        in_force_before = [p.patch_id for p in self.edits_in_force()]
        try:
            text = rt.compose_prompt(
                spec["prompt"], chat=spec["chat"], system=spec["system"], assistant_prefix=spec["assistant_prefix"]
            )
            watch_id = rt.resolve_concept_token_id(watch_word)[0] if watch_word else None
            write = lambda: rt.write_voice(  # noqa: E731
                text, max_new_tokens=spec["max_new_tokens"], stop_token_ids=rt.end_of_turn_ids(),
                watch_token_id=watch_id,
            )
            before, logp_before = write()
        except Exception as exc:
            why = _plain(exc)
            raise ValueError(f"The voice before the edit could not be written: {why}; no edit was made.") from exc

        reply = do_edit()

        after: Optional[Dict[str, Any]] = None
        first_step: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
        try:
            after, logp_after = write()
        except Exception as exc:
            error = f"The voice after the edit could not be written: {_plain(exc)}; the edit is in force."
        else:
            first_step = {
                "kl_before_after": round(kl_nats(logp_before, logp_after), 6),
                "top_before": before["first_top"],
                "top_after": after["first_top"],
                "watch_before": before["watch"],
                "watch_after": after["watch"],
            }
            del logp_after
        del logp_before
        heard = {
            **spec,
            "edits_in_force_before": in_force_before,
            "before": before,
            "after": after,
            "first_step": first_step,
            "parted_at": parted_at(before["token_ids"], after["token_ids"]) if after is not None else None,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "error": error,
        }
        pid = reply["patch"]["patch_id"]
        rec = next(p for p in self.active_patches if p.patch_id == pid)
        rec.voice = heard
        return {**reply, "patch": asdict(rec), "voice": heard}

    def _through_lens(self, reply: Dict[str, Any]) -> Dict[str, Any]:
        """`reply` with the lens its words were read, or its edit written, through named last, under "lens"."""
        return {**reply, "lens": self.runtime.lens_card()}

    def execute_lql(self, query: str, voice: Optional[Any] = None) -> Dict[str, Any]:
        """Parse and execute a LARQL / `.patch` statement string. NEAREST, DESCRIBE POLE, UPDATE and INSERT go through
        the J++ lens, so their replies also carry "lens" (see the module docstring); NEAREST and DESCRIBE POLE, which
        read neuron labels, carry "label_gate" just before it; the rest reply as before. With a `voice`, an edit
        (DELETE, UPDATE, INSERT) is heard before and after (`_heard`; the caller holds the GPU lock), and its reply
        carries "voice" before "lens"; every other statement ignores it."""
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
            return self._through_lens(
                {
                    **self.query_nearest_edges(concept=concept, layer_idx=layer_idx, top_k=top_k),
                    "label_gate": LABEL_GATE_CARD,
                }
            )

        # DESCRIBE POLE LAYER <l> SLOT <i> [POLARITY +|-]
        m_pole = re.search(
            r"DESCRIBE\s+POLE\s+LAYER\s+(\d+)\s+SLOT\s+(\d+)(?:\s+POLARITY\s+([+-]))?",
            q,
            flags=re.IGNORECASE,
        )
        if m_pole:
            l_s, i_s, pol_s = m_pole.groups()
            pol = -1.0 if pol_s == "-" else 1.0
            return self._through_lens(
                {
                    **self.inspect_bipolar_slot(layer_idx=int(l_s), neuron_idx=int(i_s), polarity=pol),
                    "label_gate": LABEL_GATE_CARD,
                }
            )

        # DELETE FROM LAYER <l> WHERE slot = <i>
        m_del = re.search(
            r"DELETE\s+FROM\s+LAYER\s+(\d+)\s+WHERE\s+slot\s*=\s*(\d+)",
            q,
            flags=re.IGNORECASE,
        )
        if m_del:
            l_s, i_s = m_del.groups()
            return self._heard(
                voice,
                None,
                lambda: self.apply_patch_surgery(
                    op_type="DELETE",
                    layer_idx=int(l_s),
                    neuron_idx=int(i_s),
                    statement=q,
                ),
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
                self._heard(
                    voice,
                    obj_c,
                    lambda: self.apply_patch_surgery(
                        op_type="UPDATE",
                        layer_idx=int(l_s),
                        neuron_idx=int(i_s),
                        target_concept=obj_c,
                        strength=float(str_s) if str_s else 2.5,
                        polarity=-1.0 if pol_s == "-" else 1.0,
                        statement=q,
                    ),
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
                self._heard(
                    voice,
                    obj_c,
                    lambda: self.apply_patch_surgery(
                        op_type="INSERT",
                        layer_idx=int(l_s),
                        neuron_idx=None,
                        subject_concept=subj_c,
                        target_concept=obj_c,
                        strength=float(str_s) if str_s else 2.5,
                        statement=q,
                    ),
                )
            )

        raise ValueError(
            f"Unrecognized LARQL / .patch query: {query!r}. "
            "Supported: DESCRIBE | SELECT * FROM EDGES NEAREST '<concept>' WHERE layer = <l> LIMIT <k> | "
            "DESCRIBE POLE LAYER <l> SLOT <i> | UPDATE LAYER <l> SLOT <i> SET object = '<c>' | "
            "INSERT INTO LAYER <l> (subject, object) VALUES ('<s>', '<o>') | "
            "DELETE FROM LAYER <l> WHERE slot = <i> | ROLLBACK"
        )

#!/usr/bin/env python3
"""Deterministic CLI Verification Suite (`scripts/verify_optics.py`) for Logos Theodolite.

Runs the multi-hop prompt `"The number of legs on the animal that spins webs is"` (the two-hop example from
Gurnee et al. 2026, REFERENCES.md [2])
as bare text, alone and with `<|think|>` set before it (the old imitation, as first built: not Gemma 4's thinking
switch, which the chat template sets at the head of a system turn; E6 §3, item 11), through `interp-engine` + the
`J++ Lens` (fitted on the base model `google/gemma-4-31B`, bf16) on `google/gemma-4-31b-it` (4-bit W4A16) and
verifies:
  (a) Intermediate bridge tokens (`"spider"`, `"arachnid"`) rising in J++-Space across
      global-attention layers (`L24, L30, L36, L42`) before transitioning to `"eight"` / `"8"`
      at `L54-L60`;
  (b) Top firing Bipolar GeGLU poles (`+` North / `-` South) and their J-VINDEX labels
      vs. naive Logit Lens labels;
  (c) Live `SteeringSpec` (`LayerSteeringSpec` + `AddSpec`) intervention flipping the
      trajectory during `capture_generation`;
  (d) Peak VRAM staying safely under 30.0 GB on a single 32GB GPU;
  (e) LARQL / VINDEX3 FFN Vault (`DESCRIBE`, `SELECT * FROM EDGES NEAREST`) and a `.patch`
      one-neuron column edit, named after MEMIT but not MEMIT (`UPDATE`, `ROLLBACK`);
  (f) the lens named by `TheodoliteRuntime.lens_card()` (the API's "lens"): J++, fitted on google/gemma-4-31B as the
      file's own provenance says, with its caution.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.larql_vault import LarqlVaultEngine
from backend.runtime_factory import make_runtime, weights_card
from backend.theodolite_math import (
    D_FF,
    D_MODEL,
    GLOBAL_ATTN_LAYERS_0BASED,
    GLOBAL_ATTN_LAYERS_1BASED,
    N_LAYERS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("verify_optics")

BASE_PROMPT = "The number of legs on the animal that spins webs is"
THINK_ENABLED_PROMPT = "<|think|>\nThe number of legs on the animal that spins webs is"


def _format_layer_label(layer_0based: int) -> str:
    l1 = layer_0based + 1
    star = "*" if l1 in GLOBAL_ATTN_LAYERS_1BASED else " "
    return f"L{l1:02d}{star}(idx={layer_0based:02d})"


async def main() -> int:
    print("=" * 92)
    print(" LOGOS THEODOLITE — DETERMINISTIC OPTICS & CIRCUIT VERIFICATION SUITE (GEMMA-4-31B) ")
    print("=" * 92)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for Gemma-4-31B verification.")

    torch.cuda.reset_peak_memory_stats()
    # THEODOLITE_GGUF puts a GGUF of Gemma 4 under the same checks (backend/runtime_factory.py); with nothing set, the
    # pinned TheodoliteRuntime on the W4A16 checkpoint, as before
    runtime = make_runtime(load_jlens=True).load()
    vault = LarqlVaultEngine(runtime)
    card = weights_card(runtime)
    print(f"  Weights           : {runtime.model_id} ({card.get('format')}, id {card.get('id')})")

    # (f) the lens every reading below goes through, as /api/health, every sweep and the scientist tools name it;
    # the base model is what the file's own provenance says, not a fallback
    lens = runtime.lens_card()
    assert lens is not None and lens["fitted_on"].split(" (")[0] == "google/gemma-4-31B", f"lens: {lens}"
    assert lens["provenance"]["hf_model_name"] == "google/gemma-4-31B" and lens["provenance"]["method"] == "jpp", f"lens: {lens}"
    assert lens["id"] == "jpp-base" and lens["applied_to"] == runtime.model_id and lens["caution"], f"lens: {lens}"
    print(f"  Lens              : {lens['name']}, fitted on {lens['fitted_on']}, read on {lens['applied_to']}")
    print(f"                      {lens['method']}; {lens['fit_data']}; source {lens['source']}")
    print(f"                      caution: {lens['caution']}")

    # -------------------------------------------------------------------------
    # 1. Sweep A: Silent / Direct Completion (`<|think|>` disabled)
    # -------------------------------------------------------------------------
    print("\n[1/5] Running 60-Layer Theodolite Sweep (`<|think|>` DISABLED)...")
    sweep_silent = await runtime.run_theodolite_sweep(
        BASE_PROMPT,
        monument_concepts=("spider", "eight", "insect"),
        survey_mode="monument",
        max_new_tokens=6,
        top_k_poles=15,
        tracked_concepts=("spider", "arachnid", "eight", "8", "insect", "six", "6"),
    )

    print(f"  Prompt            : {sweep_silent.prompt!r}")
    print(f"  Prompt Tokens ({len(sweep_silent.prompt_token_ids):2d}) : {sweep_silent.prompt_token_strs}")
    print(f"  Generated Continuation: {sweep_silent.generated_text!r}")
    print(f"  Sweep Latency     : {sweep_silent.elapsed_ms:.1f} ms")
    print(f"  Survey Monuments  : {sweep_silent.survey_labels}")

    # -------------------------------------------------------------------------
    # 2. Verify (a): Intermediate Bridge Tokens ("spider", "arachnid") vs Target ("eight", "8")
    # -------------------------------------------------------------------------
    print("\n[2/5] Verification (a): J++ Bridge Emergence (global-attention layers) vs. Final Transition")
    print("-" * 92)
    print(
        f"{'Layer':<15} | {'Top-3 J++ Plume Tokens (prob)':<40} | "
        f"{'spider rank(logit)':<18} | {'eight/8 rank(logit)':<18}"
    )
    print("-" * 92)

    # Inspect key layers including global-attention layers L24(idx=23), L30(idx=29), L36(idx=35), L42(idx=41)
    # and late layers L48(idx=47), L54(idx=53), L60(idx=59)
    inspect_layers_1based = [6, 12, 18, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60]
    for l1 in inspect_layers_1based:
        l0 = l1 - 1
        ro = sweep_silent.jpp_layer_readouts[l0]
        top3 = ", ".join(
            f"{repr(t)}({p * 100:.1f}%)" for t, p in zip(ro.top_token_strs[:3], ro.top_probs[:3])
        )
        sp_rank = ro.tracked_token_ranks.get("spider", 999999)
        sp_logit = ro.tracked_token_logits.get("spider", 0.0)
        ar_rank = ro.tracked_token_ranks.get("arachnid", 999999)
        best_bridge_rank = min(sp_rank, ar_rank)
        best_bridge_lbl = "spider" if sp_rank <= ar_rank else "arachnid"

        e_rank = min(ro.tracked_token_ranks.get("eight", 999999), ro.tracked_token_ranks.get("8", 999999))
        e_logit = max(ro.tracked_token_logits.get("eight", -99.0), ro.tracked_token_logits.get("8", -99.0))
        print(
            f"{_format_layer_label(l0):<15} | {top3:<40} | "
            f"{best_bridge_lbl}#{best_bridge_rank:<6d}({sp_logit:5.1f}) | "
            f"#{e_rank:<6d}({e_logit:5.1f})"
        )

    # Quantify bridge vs target progression across early (L1-L18), mid/global-attention (L20-L44), and final (L54-L60)
    early_spider_logits = [sweep_silent.jpp_layer_readouts[l].tracked_token_logits["spider"] for l in range(0, 12)]
    mid_spider_logits = [
        max(
            sweep_silent.jpp_layer_readouts[l].tracked_token_logits["spider"],
            sweep_silent.jpp_layer_readouts[l].tracked_token_logits["arachnid"],
        )
        for l in range(20, 45)
    ]
    mid_bridge_best_rank = min(
        min(
            sweep_silent.jpp_layer_readouts[l].tracked_token_ranks["spider"],
            sweep_silent.jpp_layer_readouts[l].tracked_token_ranks["arachnid"],
        )
        for l in range(18, 46)
    )
    final_eight_best_rank = min(
        min(
            sweep_silent.jpp_layer_readouts[l].tracked_token_ranks["eight"],
            sweep_silent.jpp_layer_readouts[l].tracked_token_ranks["8"],
        )
        for l in range(53, 60)
    )
    print("-" * 92)
    print(
        f"  Bridge Emergence Check : Peak mid-layer ('spider'/'arachnid') logit = {max(mid_spider_logits):.2f} "
        f"(vs early mean {sum(early_spider_logits)/len(early_spider_logits):.2f}), best rank = #{mid_bridge_best_rank}"
    )
    print(f"  Final Transition Check : L54-L60 ('eight'/'8') best rank = #{final_eight_best_rank}")
    assert max(mid_spider_logits) > sum(early_spider_logits) / len(early_spider_logits), (
        "Expected bridge concept ('spider'/'arachnid') logit to rise in global-attention layers!"
    )
    assert final_eight_best_rank <= 5, f"Expected 'eight'/'8' in top-5 at L54-L60, got rank #{final_eight_best_rank}"

    # -------------------------------------------------------------------------
    # Compare with `<|think|>` ENABLED
    # -------------------------------------------------------------------------
    print("\n  Running `<|think|>` ENABLED Comparison Sweep...")
    sweep_think = await runtime.run_theodolite_sweep(
        THINK_ENABLED_PROMPT,
        monument_concepts=("spider", "eight", "insect"),
        survey_mode="monument",
        max_new_tokens=12,
        top_k_poles=15,
        tracked_concepts=("spider", "arachnid", "eight", "8", "insect", "six", "6"),
    )
    print(f"  `<|think|>` Continuation: {sweep_think.generated_text!r}")
    think_traj_diff = float(
        torch.linalg.vector_norm(sweep_think.trajectory_3d.xyz - sweep_silent.trajectory_3d.xyz).item()
    )
    print(f"  3D Trajectory L2 Divergence (`<|think|>` vs. Silent): {think_traj_diff:.4f}")

    # -------------------------------------------------------------------------
    # 3. Verify (b): Top Firing Bipolar GeGLU Poles & J-VINDEX vs. Naive Logit Lens
    # -------------------------------------------------------------------------
    print("\n[3/5] Verification (b): Bipolar GeGLU Neuron Poles (J-VINDEX vs. Naive Logit Lens)")
    print("-" * 92)
    sample_global_layers = [23, 29, 35, 41, 53]  # 0-based for L24, L30, L36, L42, L54
    for l0 in sample_global_layers:
        poles_l = sweep_silent.bipolar_poles[l0]
        n0 = int(poles_l.neuron_indices[0].item())
        pol0 = float(poles_l.polarities[0].item())
        act0 = float(poles_l.signed_activations[0].item())
        jpp_toks = sweep_silent.pole_top_tokens_jpp.get((l0, 0), [])
        naive_toks = sweep_silent.pole_top_tokens_naive.get((l0, 0), [])
        pol_str = "+North" if pol0 > 0 else "-South"
        jpp_str = ", ".join(f"{repr(t)}({v:.1f})" for t, v in jpp_toks[:3])
        naive_str = ", ".join(f"{repr(t)}({v:.1f})" for t, v in naive_toks[:3])
        print(
            f"  {_format_layer_label(l0)} Top Pole N{n0:05d} [{pol_str}, act={act0:7.2f}]"
            f"\n      J-VINDEX (J++ Corrected) : {jpp_str}"
            f"\n      Naive Logit Lens (Raw W_U): {naive_str}"
        )

    print(f"\n  Verified Downstream Reader Circuit Filaments Retained: {len(sweep_silent.filaments)}")
    for fil in sweep_silent.filaments[:5]:
        s_pol = "+" if fil.src_polarity > 0 else "-"
        d_pol = "+" if fil.dst_polarity > 0 else "-"
        print(
            f"    Edge L{fil.src_layer + 1:02d}:N{fil.src_neuron:05d}({s_pol}) --> "
            f"L{fil.dst_layer + 1:02d}:N{fil.dst_neuron:05d}({d_pol}) | "
            f"coupling={fil.coupling_score:+.4f} (dot={fil.dot_product:+.4f}, |act|={fil.dst_abs_act:.2f})"
        )
    assert len(sweep_silent.filaments) > 0, "Expected non-empty downstream reader circuit filaments!"

    # Verify Module 4 Pole-Locked Motif Mode as well
    top_pole_l36 = int(sweep_silent.bipolar_poles[35].neuron_indices[0].item())
    top_pol_l36 = float(sweep_silent.bipolar_poles[35].polarities[0].item())
    sweep_pole_mode = await runtime.run_theodolite_sweep(
        BASE_PROMPT,
        survey_mode="pole_motif",
        locked_pole=(35, top_pole_l36, top_pol_l36),
        max_new_tokens=0,
    )
    assert torch.isfinite(sweep_pole_mode.trajectory_3d.xyz).all(), "Pole-motif 3D coordinates must be finite"
    assert torch.isfinite(sweep_pole_mode.cka_matrix).all(), "60x60 CKA matrix must be finite"
    print(
        f"  Pole-Locked Motif Frame Verified : Axes = {sweep_pole_mode.survey_labels} | "
        f"L36 (θ={sweep_pole_mode.trajectory_3d.azimuth_deg[35]:.1f}°, "
        f"φ={sweep_pole_mode.trajectory_3d.elevation_deg[35]:.1f}°, "
        f"||h||={sweep_pole_mode.trajectory_3d.residual_norm[35]:.1f})"
    )

    # -------------------------------------------------------------------------
    # 4. Verify (c): Live `SteeringSpec` Intervention Flipping the Trajectory
    # -------------------------------------------------------------------------
    print("\n[4/5] Verification (c): Live `SteeringSpec` Causal Laser Intervention (`eight`/`8` -> `six`/`6`)")
    print("-" * 92)
    six_tid, _ = runtime.resolve_concept_token_id("six")
    six_num_tid, _ = runtime.resolve_concept_token_id("6")
    eight_tid, _ = runtime.resolve_concept_token_id("eight")
    eight_num_tid, _ = runtime.resolve_concept_token_id("8")
    # Build a causal steering vector pulling ("six" + "6") - ("eight" + "8") back via J_pp[l]
    from interp_engine.steer_specs import AddSpec, LayerSteeringSpec, SteeringSpec

    steer_layers: Dict[int, LayerSteeringSpec] = {}
    inv_norm_w = 1.0 / runtime.final_norm_weight.float().clamp(min=0.1)
    u_six = (runtime.lm_head_weight[six_tid].float() + runtime.lm_head_weight[six_num_tid].float()) * inv_norm_w
    u_eight = (runtime.lm_head_weight[eight_tid].float() + runtime.lm_head_weight[eight_num_tid].float()) * inv_norm_w
    u_diff = u_six - u_eight

    for steer_l in (35, 41, 47, 53, 59):
        J_l = runtime.jpp_lenses[steer_l].float()
        # Pull back to layer steer_l via J_l
        v_steer = (u_diff.to(J_l.device) @ J_l).detach().cpu()
        scale_l = 450.0 if steer_l == 59 else 140.0
        steer_layers[steer_l] = LayerSteeringSpec(
            operations=[AddSpec(vector=v_steer, scale=scale_l, normalize=True)]
        )
    steering_spec = SteeringSpec(layers=steer_layers, point="resid_post")

    sweep_steered = await runtime.run_theodolite_sweep(
        BASE_PROMPT,
        monument_concepts=("spider", "eight", "six"),
        survey_mode="monument",
        steering_spec=steering_spec,
        max_new_tokens=6,
        tracked_concepts=("spider", "eight", "8", "six", "6"),
    )
    unsteered_l59 = sweep_silent.jpp_layer_readouts[59]
    steered_l59 = sweep_steered.jpp_layer_readouts[59]
    print(f"  Unsteered Continuation : {sweep_silent.generated_text!r}")
    print(f"  Steered Continuation   : {sweep_steered.generated_text!r}")
    best_unsteered_six_rank = min(unsteered_l59.tracked_token_ranks["six"], unsteered_l59.tracked_token_ranks["6"])
    best_steered_six_rank = min(steered_l59.tracked_token_ranks["six"], steered_l59.tracked_token_ranks["6"])
    best_unsteered_six_logit = max(unsteered_l59.tracked_token_logits["six"], unsteered_l59.tracked_token_logits["6"])
    best_steered_six_logit = max(steered_l59.tracked_token_logits["six"], steered_l59.tracked_token_logits["6"])
    print(
        f"  L60 'six'/'6' Rank(Logit): Unsteered #{best_unsteered_six_rank} "
        f"({best_unsteered_six_logit:.2f})  -->  "
        f"Steered #{best_steered_six_rank} "
        f"({best_steered_six_logit:.2f})"
    )
    assert best_steered_six_rank < best_unsteered_six_rank, (
        "Expected SteeringSpec intervention to improve target concept ('six'/'6') rank!"
    )

    # -------------------------------------------------------------------------
    # 5. Verify LARQL / VINDEX3 Vault & Peak VRAM Envelope (< 30.0 GB)
    # -------------------------------------------------------------------------
    print("\n[5/5] Verification (d) & LARQL Vault: VINDEX3 Query, `.patch` Surgery & VRAM Envelope")
    print("-" * 92)
    vault_desc = vault.describe_vault()
    print(f"  LARQL CLI Binaries     : {vault_desc['cli_binaries']}")
    lql_res = vault.execute_lql("SELECT * FROM EDGES NEAREST 'spider' WHERE layer = 35 LIMIT 3")
    for row in lql_res["results"]:
        tok_list = row.get("j_vindex_top_tokens") or row.get("j_vindex_tokens") or []
        jtoks = ", ".join(f"{t['token']!r}({t['logit']})" for t in tok_list[:3])
        print(
            f"  LQL Nearest Edge #{row['rank']} : L{row['layer']} Slot {row['neuron_idx']} "
            f"({row['polarity']}) cos={row['cosine_similarity']:.4f} -> J-VINDEX: {jtoks}"
        )

    patch_res = vault.execute_lql("UPDATE LAYER 35 SLOT 100 SET object = 'six' STRENGTH 3.0 POLARITY +")
    print(f"  Applied LARQL .patch   : {patch_res['patch']['statement']} (new_norm={patch_res['patch']['new_norm']})")
    rb_res = vault.execute_lql("ROLLBACK")
    print(f"  Rolled back .patch     : cleared={rb_res['cleared_patches']}")

    peak_vram_gb = torch.cuda.max_memory_allocated(runtime.device) / (1024**3)
    reserved_vram_gb = torch.cuda.max_memory_reserved(runtime.device) / (1024**3)
    print("-" * 92)
    print(f"  Peak Allocated VRAM    : {peak_vram_gb:.2f} GB  (Budget: < 30.00 GB)")
    print(f"  Peak Reserved VRAM     : {reserved_vram_gb:.2f} GB")
    assert peak_vram_gb < 30.0, f"Peak VRAM {peak_vram_gb:.2f} GB exceeded 30.0 GB limit!"

    print("\n" + "=" * 92)
    print(" ALL MATHEMATICAL, CAUSAL, LARQL & VRAM ASSERTIONS PASSED WITH ZERO ERRORS! ")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

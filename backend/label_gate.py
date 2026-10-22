"""The label gate (`backend/label_gate.py`): which neuron labels the instrument's pages hold back from their main reading.

The study's synthesis asks for output labels "only at L≥40, gated by J++/naive agreement" (REPORT §3.3, lines 353 and
380, with erratum E47). N1 measured both conditions on its fixed sample of random neurons (N1_REPORT.md:22-29, 124-127;
the fresh sample in N1V_robustness.md:111-115):
- condition 2, the layer: output labels held up at code layers 40-57 and were at chance at 10-30, where they were
  sampled (10, 20, 30, 40, 45, 50, 55 and 57; nothing at or above 58);
- condition 3, agreement: how many word-piece ids the J++ label and the naked eye's (naive logit lens) label share in
  their first ten, over the full vocabulary, from fp32 softcapped logits (experiments/scripts/N1C_02_matrix.py:248-255),
  ranked a sound label above an unsound one with AUC 0.84 within layers 40-57, for base-fitted J++ labels only (E47).
  The study set no threshold: holding back a label whose two readings share none of ten (`CUT`) is the instrument's
  choice.

`label_check` marks one label; `LABEL_GATE_CARD` states the rule, and travels with the replies that carry label checks.
Nothing is hidden from a caller: the words are still sent, and `held_back` only marks what the pages leave out of their
main reading. Layers here count from 0 (page layer = code layer + 1). Pure and CPU-importable: the server, the vault,
the scientist tools and scripts/check_label_gate.py all import it, and the check runs `top_ids_from_shortlist` itself.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from backend.theodolite_math import FINAL_LOGIT_SOFTCAP, gemma4_rms_norm

LABEL_GATE_RULE = "n1-gate-1"
LABEL_MIN_LAYER = 40  # counting from 0: page layer 41
BAND = (40, 57)  # where the study measured the gate (both ends sampled)
SAMPLED = (10, 20, 30, 40, 45, 50, 55, 57)  # the layers N1 sampled (N1_REPORT.md:7)
ONE_READING_LAYER = 59  # J++[59] = I: the lens and the naked eye are one reading there
TOPK = 10
SHORTLIST = 64
CUT = 0  # held back when shared_top10 <= CUT (decision 3: none of ten); the instrument's choice, not the study's
VALIDATED_LENS_ID = "jpp-base"  # the only lens the gate was validated for (E47)

HOLD_REASONS = ("below_layer_40", "shares_nothing")
BANDS = ("below_sampled", "chance_where_sampled", "not_sampled", "band", "above_band", "one_reading")

LABEL_GATE_CARD: Dict[str, Any] = {
    "rule": LABEL_GATE_RULE,
    "layers_count_from": 0,
    "min_layer": LABEL_MIN_LAYER,
    "measure": "word-piece ids in both readings' first ten, fp32, softcapped",
    "cut": "none of ten: the instrument's choice; the study set no threshold",
    "held_back_if_shared_top10_at_most": CUT,
    "validated": "AUC 0.84 within layers 40-57 (counting from 0), for base-fitted J++ labels on random neurons",
    "sources": ["N1_REPORT.md:124-127", "N1V_robustness.md:111-115", "REPORT.md erratum E47"],
    "caution": (
        "The words are still sent: held_back marks the labels the instrument's pages leave out of their main reading, "
        "and they keep them in the apparatus. A label is held back below layer 40 (reason below_layer_40), and, under "
        "the base-fitted J++ lens (jpp-base) below layer 59, where its two readings share none of their first ten "
        "word-pieces (reason shares_nothing). Under any other lens the agreement gate is unvalidated (erratum E47): "
        "only the layer holds a label back, and shared_top10 is a count, not a verdict (lens_validated false). A "
        "neuron edited in the vault is read through its edit (edited true, lens_validated false). Below layer 40 "
        "shared_top10 depends on how ties at the softcap break, so it may differ from the study's count there; the "
        "hold-back there rests on the layer alone. At layer 59 the lens is the identity, so the two readings are one "
        "and always share ten."
    ),
}


def _softcap(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x / FINAL_LOGIT_SOFTCAP) * FINAL_LOGIT_SOFTCAP


@torch.no_grad()
def top_ids_from_shortlist(
    bf16_logits: torch.Tensor,
    exact_logits: Callable[[torch.Tensor], torch.Tensor],
    k: int = TOPK,
    shortlist: int = SHORTLIST,
) -> List[List[int]]:
    """The first `k` word-piece ids of each row, as the study ranked them: shortlist the `shortlist` best of the
    server's own logits (`bf16_logits`, [R, V], from a bf16 GEMM, whose rounding ties scores near the softcap), score
    the shortlist exactly with `exact_logits(ids [R, S]) -> [R, S]` (fp32, softcapped), and rank those. The shortlist
    is put in id order first and the ranking is stable, so equal exact scores break toward the lower id whatever order
    the bf16 scores gave them."""
    if bf16_logits.dim() == 1:
        bf16_logits = bf16_logits.unsqueeze(0)
    s = min(int(shortlist), bf16_logits.shape[-1])
    _, cand = torch.topk(bf16_logits.float(), k=s, dim=-1)
    cand, _ = cand.sort(dim=-1)
    exact = exact_logits(cand).float()
    order = exact.sort(dim=-1, descending=True, stable=True).indices[:, : min(int(k), s)]
    return cand.gather(-1, order).cpu().tolist()


@torch.no_grad()
def label_top_ids(
    vecs: torch.Tensor,
    final_norm_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    bf16_logits: torch.Tensor,
    k: int = TOPK,
    chunk: int = 16,
) -> List[List[int]]:
    """The first `k` word-piece ids of each vector's label, by the study's measure: the normed vector in bf16 (as the
    tool norms it), an fp32 GEMM against the unembedding, softcapped (experiments/scripts/N1V_lib.py:30-43,
    N1A_common.py:88-104). `bf16_logits` are the logits the server already computed for `vecs` ([R, V], or [V] for one
    vector); only their top `SHORTLIST` are scored again, `chunk` rows at a time."""
    if vecs.dim() == 1:
        vecs = vecs.unsqueeze(0)
    if bf16_logits.dim() == 1:
        bf16_logits = bf16_logits.unsqueeze(0)
    out: List[List[int]] = []
    for s in range(0, vecs.shape[0], chunk):
        v = vecs[s : s + chunk]
        hn = gemma4_rms_norm(v, final_norm_weight).float()  # [r, D]

        def exact(ids: torch.Tensor, hn: torch.Tensor = hn) -> torch.Tensor:
            w = lm_head_weight[ids.to(lm_head_weight.device)].float()  # [r, S, D]
            return _softcap(torch.einsum("rd,rsd->rs", hn.to(w.device), w))

        out += top_ids_from_shortlist(bf16_logits[s : s + chunk], exact, k=k)
    return out


def band_of(layer_idx: int) -> str:
    """Where a layer (counting from 0) stands against the study's sample: `BANDS`."""
    l = int(layer_idx)
    if l < SAMPLED[0]:
        return "below_sampled"
    if l <= 30:
        return "chance_where_sampled"
    if l < BAND[0]:
        return "not_sampled"
    if l <= BAND[1]:
        return "band"
    if l < ONE_READING_LAYER:
        return "above_band"
    return "one_reading"


def label_check(
    layer_idx: int,
    lens_ids: Optional[Sequence[int]],
    eye_ids: Optional[Sequence[int]],
    lens_id: Optional[str],
    edited: bool = False,
) -> Dict[str, Any]:
    """One label's check, for the face whose lens (J++) label's first ids are `lens_ids` and whose naked-eye label's
    are `eye_ids`, at code layer `layer_idx`, read through the lens `lens_id` (the lens card's id):
    - shared_top10: word-piece ids in both first tens (None when either list is missing);
    - held_back, reasons: `HOLD_REASONS` that apply: "below_layer_40" (condition 2), "shares_nothing" (condition 3:
      shared_top10 <= CUT, below layer 59, under jpp-base only);
    - band (`band_of`), sampled_layer (one of `SAMPLED`), in_measured_band (40-57);
    - lens_validated: the gate was validated for this lens (jpp-base) and the neuron is not edited in the vault;
    - edited: the vault has an edit in force on this neuron (not a reason to hold back: it unvalidates the gate).
    """
    l = int(layer_idx)
    shared: Optional[int] = None
    if lens_ids and eye_ids:
        shared = len(set(int(i) for i in list(lens_ids)[:TOPK]) & set(int(i) for i in list(eye_ids)[:TOPK]))
    base = lens_id == VALIDATED_LENS_ID
    reasons: List[str] = []
    if l < LABEL_MIN_LAYER:
        reasons.append("below_layer_40")
    if shared is not None and shared <= CUT and l < ONE_READING_LAYER and base:
        reasons.append("shares_nothing")
    return {
        "shared_top10": shared,
        "held_back": bool(reasons),
        "reasons": reasons,
        "band": band_of(l),
        "sampled_layer": l in SAMPLED,
        "in_measured_band": BAND[0] <= l <= BAND[1],
        "lens_validated": bool(base and not edited),
        "edited": bool(edited),
    }

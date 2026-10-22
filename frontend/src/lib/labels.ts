/**
 * The label gate on the pages: which neuron labels a page leaves out of its main reading, and why. One source for every
 * page, so that the instrument, a layer's leaf, a neuron's folio, the nudge, the threads and the vault hold back the
 * same labels and say the same thing about them.
 *
 * The study's synthesis asks for output labels "only at L≥40, gated by J++/naive agreement" (REPORT §3.3, erratum E47).
 * The server marks each label it sends with `label_check` (backend/label_gate.py) and states the rule in `label_gate`:
 * - condition 2, the layer: below page layer 41 (code 40) a label is held back. At the layers N1 sampled there (pages
 *   11, 21 and 31) labels were no better than chance; pages 1-10 and 32-40 were not sampled (N1_REPORT.md:7, 22-29);
 * - condition 3, agreement: how many word-piece ids the lens's label and the naked eye's share in their first ten
 *   (`shared_top10`, fp32, softcapped, as N1 counted). Within pages 41-58 it ranked a sound label above an unsound one
 *   with AUC 0.84, for base-fitted J++ labels on random neurons (N1_REPORT.md:124-127). A label sharing none of ten is
 *   held back: that cut is the instrument's (the study set none). It applies under the base-fitted lens only (E47), and
 *   never at page 60, where the lens is the identity and the two readings are one.
 * Nothing is hidden from a scholar: the words stay in the apparatus, with the reason (components/HeldBack.tsx).
 *
 * Layers here are code indices, counted from 0 (`LayerIdx`; page layer = index + 1), as the server's fields count them.
 * Condition 3 is never computed from the three or five words a page shows: without the server's count it is
 * "unchecked", and only condition 2 holds a label back.
 */
import { FIXTURE_LABEL_CHECKS, RECORDED_LABEL_CHECKS_RUN, VAULT_LABEL_CHECKS } from "../fixtures/recordedLabelChecks";

/** A layer as a code index, 0-59 (page layer = index + 1). Branded, so a page layer number (1-60) cannot be passed by
 * mistake: wrap a reading's `idx` with `layerIdx`, a page number with `layerIdxOfPage`. */
export type LayerIdx = number & { readonly __layerIdx: "counted from 0" };
export const layerIdx = (idx: number): LayerIdx => idx as LayerIdx;
export const layerIdxOfPage = (layerNo: number): LayerIdx => (layerNo - 1) as LayerIdx;

/** The server's constants (backend/label_gate.py), as code indices. */
export const LABEL_MIN_IDX = 40; // page layer 41
export const BAND_IDX: readonly [number, number] = [40, 57]; // pages 41-58, where the gate was measured
export const SAMPLED_IDX: readonly number[] = [10, 20, 30, 40, 45, 50, 55, 57]; // N1's layers (pages 11 … 58)
export const ONE_READING_IDX = 59; // page 60: J++[59] = I

/** Why a label is held back, as the server names it (code layers: "below_layer_40" is below page layer 41). */
export type HoldReason = "below_layer_40" | "shares_nothing";
const HOLD_REASONS: readonly HoldReason[] = ["below_layer_40", "shares_nothing"];

/** Where a layer stands against N1's sample, as the server names it. */
export type LabelBand = "below_sampled" | "chance_where_sampled" | "not_sampled" | "band" | "above_band" | "one_reading";
const BANDS: readonly LabelBand[] = ["below_sampled", "chance_where_sampled", "not_sampled", "band", "above_band", "one_reading"];

/** `label_check` as the server sends it (and as experiments/scripts/N1G_01_recorded_label_checks.py records it). */
export interface RawLabelCheck {
  shared_top10: number | null;
  held_back: boolean;
  reasons: string[];
  band: string;
  sampled_layer: boolean;
  in_measured_band: boolean;
  lens_validated: boolean;
  edited: boolean;
}

/** One label's check, as the pages read it. */
export interface LabelCheck {
  /** word-piece ids in both readings' first ten, 0-10; null when either reading was missing */
  sharedTop10: number | null;
  heldBack: boolean;
  reasons: HoldReason[];
  band: LabelBand;
  /** the layer is one N1 sampled */
  sampledLayer: boolean;
  /** pages 41-58, where the gate was measured */
  inMeasuredBand: boolean;
  /** read through the base-fitted J++ lens (the only one the gate was validated for), and not edited in the vault */
  lensValidated: boolean;
  /** the vault has an edit in force on this neuron: its words echo the edit */
  edited: boolean;
}

/** The server's `label_check` (snake_case) for the pages, or null when it is missing or not one. */
export function parseLabelCheck(x: unknown): LabelCheck | null {
  if (!x || typeof x !== "object") return null;
  const c = x as Record<string, unknown>;
  if (typeof c.held_back !== "boolean") return null;
  const shared = typeof c.shared_top10 === "number" && Number.isInteger(c.shared_top10) && c.shared_top10 >= 0 && c.shared_top10 <= 10 ? c.shared_top10 : null;
  const reasons = Array.isArray(c.reasons) ? c.reasons.filter((r): r is HoldReason => HOLD_REASONS.includes(r as HoldReason)) : [];
  return {
    sharedTop10: shared,
    heldBack: c.held_back,
    reasons,
    band: BANDS.includes(c.band as LabelBand) ? (c.band as LabelBand) : "below_sampled",
    sampledLayer: c.sampled_layer === true,
    inMeasuredBand: c.in_measured_band === true,
    lensValidated: c.lens_validated === true,
    edited: c.edited === true,
  };
}

/** Where a layer stands against N1's sample (the server's band_of). */
export function bandOf(idx: LayerIdx): LabelBand {
  if (idx < SAMPLED_IDX[0]) return "below_sampled";
  if (idx <= 30) return "chance_where_sampled";
  if (idx < BAND_IDX[0]) return "not_sampled";
  if (idx <= BAND_IDX[1]) return "band";
  if (idx < ONE_READING_IDX) return "above_band";
  return "one_reading";
}

/**
 * What a page does with one label:
 * - heldBack, reasons: the server's verdict when it sent a check; condition 2 is applied here as well, so a label below
 *   page 41 is held back whatever the check says;
 * - condition3: "applied" (counted ten a side, through the base-fitted lens, not edited), "unvalidated" (counted, but
 *   under another lens or for an edited neuron: a count, not a verdict; erratum E47), "one-reading" (page 60, where the
 *   two readings are one), "unchecked" (no ten-a-side count: a recording not yet checked, an older server).
 */
export interface Gate {
  idx: LayerIdx;
  heldBack: boolean;
  reasons: HoldReason[];
  band: LabelBand;
  sharedTop10: number | null;
  condition3: "applied" | "unvalidated" | "one-reading" | "unchecked";
  edited: boolean;
  check: LabelCheck | null;
}

/** The gate for a label at layer `idx`, from its check (null: none sent or recorded). */
export function labelGate(idx: LayerIdx, check: LabelCheck | null): Gate {
  const reasons: HoldReason[] = check ? [...check.reasons] : [];
  if (idx < LABEL_MIN_IDX && !reasons.includes("below_layer_40")) reasons.unshift("below_layer_40");
  const shared = check?.sharedTop10 ?? null;
  const condition3: Gate["condition3"] =
    shared == null ? "unchecked" : idx >= ONE_READING_IDX ? "one-reading" : check?.lensValidated ? "applied" : "unvalidated";
  return {
    idx,
    heldBack: reasons.length > 0 || (check?.heldBack ?? false),
    reasons,
    band: bandOf(idx),
    sharedTop10: shared,
    condition3,
    edited: check?.edited ?? false,
    check,
  };
}

/* ---------------------------------------------------------------------------------------------- recorded checks */

/** Whether the recorded labels (the demonstration sighting's and the vault's recorded searches) have been checked ten
 * a side: experiments/results/N1G_recorded_label_checks.json, through fixtures/recordedLabelChecks.ts. */
export const RECORDED_CHECKED = RECORDED_LABEL_CHECKS_RUN !== null;
export const RECORDED_CHECKS_RUN = RECORDED_LABEL_CHECKS_RUN;

/** The recorded check for the demonstration sighting's label at layer `idx`, rank `rank`; null until checked. */
export const fixtureCheck = (idx: number, rank: number): LabelCheck | null => parseLabelCheck(FIXTURE_LABEL_CHECKS[`${idx}:${rank}`]);

/** The recorded check for a row of a recorded vault search ("she@20": word @ code layer); null until checked. */
export const vaultCheck = (search: string, neuron: number, face: number): LabelCheck | null =>
  parseLabelCheck(VAULT_LABEL_CHECKS[search]?.[`${neuron}${face >= 0 ? "+" : "-"}`]);

/* ------------------------------------------------------------------------------------------------------ wording */

/** The figures the wording quotes, with their sources (scripts/check_label_gate.py checks them against the files). */
export const GATE_FIGURES = {
  // N1C_matrix.json q.static_confidence_predictor.jv_nv_top10_overlap.auc_late_target_p10_ge_0.3 = 0.8432
  auc: "0.84",
  aucIn100: 84,
  // N1C_gate_cut.json band (a re-analysis of N1's data, experiments/scripts/N1C_05_gate_cut.py): within pages 41-58,
  // of 240 faces, 31 shared none of ten and 7 of those were sound; 160 of the 209 others were sound (P@10 >= 0.3)
  shareNothing: 31,
  shareNothingSound: 7,
  shareSome: 209,
  shareSomeSound: 160,
} as const;

/** The short tag a held-back label carries: "held back · below layer 41". */
export function heldTag(g: Gate): string {
  if (g.reasons.includes("below_layer_40")) return "held back · below layer 41";
  if (g.reasons.includes("shares_nothing")) return "held back · the two readings share none of ten";
  return "held back";
}

/** The italic line in place of the words (empty when the label is shown). */
export function heldLine(g: Gate): string {
  if (!g.heldBack) return "";
  if (g.reasons.includes("below_layer_40")) return "Its words are held back at this layer.";
  if (g.reasons.includes("shares_nothing")) return "Its words are held back: the lens and the naked eye share none of their first ten.";
  return "Its words are held back here.";
}

/** The rubric beside a held-back label: why, in plain words (empty when the label is shown). */
export function heldRubric(g: Gate): string {
  if (!g.heldBack) return "";
  if (g.reasons.includes("below_layer_40")) {
    return (
      "Below layer 41 a neuron’s words are not set out here. At the layers the study tried there, 11, 21 and 31, they " +
      "were no better than chance at saying what the neuron does; layers 1 to 10 and 32 to 40 were not tried. They are " +
      "kept in the apparatus, with the reason."
    );
  }
  if (g.reasons.includes("shares_nothing")) {
    const F = GATE_FIGURES;
    const beyond =
      g.idx > BAND_IDX[1]
        ? " Layer 59 lies above every layer the study tested labels at, its highest being 58, so here the rule is carried one layer beyond what was measured."
        : "";
    return (
      "Within layers 41 to 58, the more words the lens and the naked eye shared, the likelier a sound label: in the " +
      `study’s sample of random neurons, the overlap ranked a sound label above an unsound one ${F.aucIn100} times in 100 ` +
      `(AUC ${F.auc}, for labels read through the base-fitted J++ lens). Counted again from the study’s data, ` +
      `${F.shareNothingSound} of the ${F.shareNothing} labels whose readings shared none of their first ten were sound, ` +
      `against ${F.shareSomeSound} of the ${F.shareSome} that shared at least one. A label whose two readings share none ` +
      `is held back here; the cut is the instrument’s, since the study set none. It is kept in the apparatus.${beyond}`
    );
  }
  return "This label is held back from the main reading. It is kept in the apparatus.";
}

/** The ten-a-side count in words, for the apparatus; null when there is no count (the page shows a placeholder). */
export function sharedLine(g: Gate): string | null {
  const k = g.sharedTop10;
  if (k == null) return null;
  const counted = `${k} of 10 shared with the naked eye, counted by word-piece, as the study counted`;
  if (g.condition3 === "one-reading") return `${counted}: at layer 60 the two readings are one, so they share all ten`;
  if (g.condition3 === "unvalidated") {
    return g.edited
      ? `${counted}; this neuron is edited in the vault, its words echo the edit, and the gate was never measured on such neurons`
      : `${counted}; under this lens the count is not a verdict, since the gate was validated for the base-fitted J++ lens only (erratum E47)`;
  }
  return counted;
}

/**
 * How far a neuron's label can be trusted at a layer, by its code index (N1 sampled code layers 10, 20, 30, 40, 45, 50,
 * 55 and 57: labels held up at pages 41-58 and were at chance at 11-31; 1-10, 32-40 and 59-60 were not sampled; erratum
 * E9). Replaces lib/words.ts labelTrust, which said the two eyes "see much the same thing" at the last layers.
 */
export function trustLine(idx: LayerIdx): string {
  if (idx >= ONE_READING_IDX) {
    return "At layer 60 no lens stands between: the lens and the naked eye are one reading here, so they always agree, and their agreement says nothing. The study did not test neuron labels at this layer.";
  }
  if (idx > BAND_IDX[1]) return "Layer 59 lies above every layer the study tested labels at, its highest being 58: these words are untested here.";
  if (idx >= LABEL_MIN_IDX) return "Layers 41 to 58 are where neuron labels held up under causal tests, in the study’s sample of random neurons.";
  if (idx >= 31) {
    return "A rumour, not a finding: the study did not test labels at layers 32 to 40, and found them no better than chance below.";
  }
  if (idx >= 10) return "A rumour, not a finding: at layers 11 to 31 the study found neuron labels no better than chance.";
  return "A rumour, not a finding: the study did not test labels below layer 11.";
}

/**
 * The word-pieces two shown lists share, spelt exactly (" Am" and "Am" are different word-pieces): for the ◆ marks
 * only, and to be worded "among the words shown", never as the gate's agreement, which counts ten a side by id.
 * `k` counts the distinct pieces the lens shows that the naked eye also shows; `of` is how many the lens shows.
 */
export function sharedShown(lens: readonly string[], eye: readonly string[]): { k: number; of: number; shared: string[] } {
  const e = new Set(eye);
  const shared: string[] = [];
  for (const t of lens) if (e.has(t) && !shared.includes(t)) shared.push(t);
  return { k: shared.length, of: lens.length, shared };
}

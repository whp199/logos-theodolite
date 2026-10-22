/**
 * The study's figures the folio's causal test quotes beside its own result, and the small helpers that word them. Each
 * figure is traced to its file in the comment above it; scripts/check_causal_check.py (item 15) checks every one against
 * the files. Layers here are code indices, counted from 0, as the study's reports count them (page layer = index + 1).
 *
 * The test is the study's N1A protocol (experiments/scripts/N1A_02_causal_core.py) at erratum E47's gentle k = 10, run
 * by the server's causal_spot_check (backend/causal_check.py; ARCHITECTURE §4.5).
 */
import { SAMPLED_IDX } from "../../lib/labels";

/**
 * How straight an activation write stayed at each layer N1 sampled: the share of a straight write's size it kept,
 * ‖pn(y + δw) − pn(y)‖ / ‖w_post ⊙ δw / rms(y)‖, over 24 neurons × 36 texts × 2 faces, at the last word-piece.
 * experiments/results/N1V_audit_01_postnorm.json.full.json result.diag[L].k10.sat.{median, q10} and
 * .k60.sat.median, rounded to four places.
 */
export const STUDY_LINEARITY: Readonly<Record<number, { k10Median: number; k10Q10: number; k60Median: number }>> = {
  10: { k10Median: 0.9994, k10Q10: 0.9682, k60Median: 0.9811 },
  20: { k10Median: 0.9653, k10Q10: 0.7012, k60Median: 0.5498 },
  30: { k10Median: 0.9675, k10Q10: 0.6865, k60Median: 0.5599 },
  40: { k10Median: 0.9858, k10Q10: 0.9282, k60Median: 0.6994 },
  45: { k10Median: 0.9659, k10Q10: 0.7489, k60Median: 0.5229 },
  50: { k10Median: 0.9763, k10Q10: 0.8854, k60Median: 0.6013 },
  55: { k10Median: 0.9925, k10Q10: 0.8975, k60Median: 0.8162 },
  57: { k10Median: 0.9924, k10Q10: 0.8998, k60Median: 0.8039 },
};

/**
 * Below this an activation write is called too strong to read straight: the lowest k = 10 tenth percentile the study
 * saw at any layer (STUDY_LINEARITY[30].k10Q10, 0.68649, page layer 31), to three places. Not the per-layer median (0.965
 * at its lowest), which half of all gentle tests fall below by construction.
 */
export const LINEARITY_TAIL = 0.686;
export const LINEARITY_TAIL_IDX = 30;

/**
 * At or below this ratio of a test's KL to its floor's (the same test at k = 0.01), the model moved no more than rounding
 * noise. N1V_code_audit.md F2 judged layer index 10's KL median of 0.0022 "at the noise floor" against a bf16 floor of
 * 0.0008-0.0018: 0.0022 / 0.0008 = 2.75 at most. (Layers 20-30, at 50-140 times the floor, were not noise.)
 */
export const NOISE_RATIO_MAX = 2.75;

/**
 * The study's gentle tests (k = 10) on its 36 texts, by layer: the median split-half ρ of each neuron's c (even texts
 * against odd), ρ(c+, c−) and P@10 of the lens's (J-VINDEX) and the naked eye's labels.
 * experiments/results/N1V_audit_04_smallk_metrics.json k10_L{L}: split_rho_median, "rho_c+_c-", "jpp|p10".real,
 * "naive|p10".real (N1V_code_audit.md, the table "Re-test in the linear regime").
 */
export const STUDY_K10: Readonly<Record<number, { splitHalf: number; antipodal: number; p10Lens: number; p10Eye: number }>> = {
  20: { splitHalf: 0.0252, antipodal: 0.2137, p10Lens: 0.0, p10Eye: 0.0042 },
  30: { splitHalf: 0.004, antipodal: -0.0657, p10Lens: 0.0, p10Eye: 0.0021 },
  40: { splitHalf: 0.3898, antipodal: -0.7699, p10Lens: 0.3854, p10Eye: 0.3646 },
  45: { splitHalf: 0.5525, antipodal: -0.8379, p10Lens: 0.3146, p10Eye: 0.2792 },
  50: { splitHalf: 0.7538, antipodal: -0.9168, p10Lens: 0.5458, p10Eye: 0.5437 },
  55: { splitHalf: 0.9146, antipodal: -0.9604, p10Lens: 0.6896, p10Eye: 0.6979 },
};

/** The layers (code) of STUDY_K10 in order, and the lens's P@10 range at those from 40 up (pages 41-56). */
export const STUDY_K10_IDX = Object.keys(STUDY_K10)
  .map(Number)
  .sort((a, b) => a - b);
const lateP10 = STUDY_K10_IDX.filter((l) => l >= 40).map((l) => STUDY_K10[l].p10Lens);
export const STUDY_K10_LATE_P10: readonly [number, number] = [Math.min(...lateP10), Math.max(...lateP10)];

/** The sampled layer nearest to `idx` (the lower one on a tie), for the per-layer comparisons. */
export function nearestSampled(idx: number): number {
  let best = SAMPLED_IDX[0];
  for (const l of SAMPLED_IDX) if (Math.abs(l - idx) < Math.abs(best - idx)) best = l;
  return best;
}

/** A probability from a log-probability, in plain words: whole percent from 10%, one decimal from 0.1%, then the first
 * figure that is not 0 down to one in a million, below that "under one in a million". */
export function probWord(logprob: number | null): string {
  if (logprob == null || !Number.isFinite(logprob)) return "?";
  const p = Math.exp(logprob);
  const v = p * 100;
  if (v >= 10) return `${Math.round(v)}%`;
  if (v >= 0.1) return `${Math.round(v * 10) / 10}%`;
  if (p >= 1e-6) return `${Number(v.toPrecision(1))}%`;
  return "under one in a million";
}

/** How many times as likely a word became, from its change of log-probability. */
export function timesWord(dlogprob: number): string {
  const x = Math.exp(dlogprob);
  if (x >= 100) return String(Math.round(x));
  if (x >= 10) return x.toFixed(1);
  const two = x.toFixed(2);
  return two === "1.00" ? x.toFixed(4) : two;
}

/** A number for the apparatus: three significant figures, a true minus sign. */
export function sig(v: number | null | undefined, n = 3): string {
  if (v == null || !Number.isFinite(v)) return "—";
  const s = Math.abs(v) >= 1e5 || (v !== 0 && Math.abs(v) < 1e-4) ? v.toExponential(n - 1) : String(Number(v.toPrecision(n)));
  return s.replace(/-/g, "−");
}

/** Two lists of ids hold the same ids (in any order). */
export function sameIds(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false;
  const x = [...a].sort((p, q) => p - q);
  const y = [...b].sort((p, q) => p - q);
  return x.every((v, i) => v === y[i]);
}

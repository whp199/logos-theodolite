/**
 * Which lens a reading went through, in plain words. The server names its lens (`lens` in GET /api/health, in every
 * sweep and in the scientist replies); the recorded sighting keeps no such record, but the study took it through the
 * same file. Every reading on these pages so far is the J++ lens, fitted on the base model google/gemma-4-31B and
 * applied to the instruct 4-bit checkpoint (REPORT §3.4, bug 9; R1).
 */
import { LENS_MODEL_NOT_SENT, LENS_NAME_NOT_SENT, useTheodoliteStore, type LensInfo, type Sighting } from "../store/useTheodoliteStore";

/** The lens the instrument loads, as its file's provenance names it (backend/engine_runtime.py, JPP_LENS_FALLBACK). */
export const BASE_LENS: LensInfo = {
  id: "jpp-base",
  name: "J++",
  fittedOn: "google/gemma-4-31B (base, bf16)",
  appliedTo: "google/gemma-4-31B-it-qat-w4a16-ct",
  method: "J++, R-lens LRP backward pass",
  fitData: "64 WikiText passages of 128 word-pieces",
  source: "koayon/jpp-lenses@6c96867a91c43c14cec17cf0f6d157b933c5fee5:gemma-4-31b/lens.pt",
  file: "neuronpedia/jacobian-lens:gemma-4-31b/jpp/Salesforce-wikitext/gemma-4-31B_jpp_lens.pt",
  // no server sent this record, so there is no server caution to repeat
  caution: null,
  identified: true,
};

/** Where the lens named on a page comes from: the sighting's own record, the server's, the recorded sighting (taken
 * through the same file, which it does not record), or no record at all (a server that is not answering, or an older
 * one whose replies carry no `lens`). */
export type LensFrom = "sighting" | "server" | "recorded" | "unnamed";

/** The shipped lens, the one every band and check on these pages was measured with: J++ fitted on the base model. */
export const isShippedLens = (lens: LensInfo): boolean => lens.identified && lens.id === "jpp-base";

/** Fitted on the base model, as the lens's own record says. False for a card that leaves out its name or its model. */
export const isBaseFitted = (lens: LensInfo | null): boolean =>
  !lens || (lens.identified && (lens.id === "jpp-base" || /\(base\b/.test(lens.fittedOn)));

/** "google/gemma-4-31B" from "google/gemma-4-31B (base, bf16)" */
export const fittedModel = (lens: LensInfo): string => lens.fittedOn.replace(/\s*\(.*$/, "");

/** The phrase for a lens the server sent without its name or without its model, saying only what the card says. */
function unidentifiedPhrase(lens: LensInfo): string {
  if (lens.name !== LENS_NAME_NOT_SENT) return `a ${lens.name} lens whose file does not name the model it was fitted on`;
  if (lens.fittedOn !== LENS_MODEL_NOT_SENT) return `an unnamed lens, fitted on ${lens.fittedOn}`;
  return "a lens the server does not identify";
}

/** "the J++ lens, fitted on the base model" (with `named`, the model's name follows) */
export function lensPhrase(lens: LensInfo, named = false): string {
  if (!lens.identified) return unidentifiedPhrase(lens);
  return isBaseFitted(lens)
    ? `the ${lens.name} lens, fitted on the base model${named ? ` ${fittedModel(lens)}` : ""}`
    : `the ${lens.name} lens, fitted on ${lens.fittedOn}`;
}

/** The short label for status lines: "J++ lens, fitted on the base model"; "lens not identified" for an incomplete card. */
export const lensLabel = (lens: LensInfo): string =>
  lens.identified ? lensPhrase(lens).replace(/^the /, "") : "lens not identified";

/** "base" or "instruct", as the lens's own record says what it was fitted on; null when it does not say. */
export function lensKind(lens: LensInfo): "base" | "instruct" | null {
  if (!lens.identified) return null;
  if (isBaseFitted(lens)) return "base";
  return /-it\b|instruct/i.test(lens.fittedOn) ? "instruct" : null;
}

/** The narrowest label, for the head of the page on a phone: "J++ lens (base model)". */
export function lensShort(lens: LensInfo): string {
  if (!lens.identified) return "lens not identified";
  const kind = lensKind(lens);
  return kind ? `${lens.name} lens (${kind === "base" ? "base model" : "chat model"})` : `${lens.name} lens`;
}

/**
 * What the study found a reading through the shipped J++ lens (fitted on the base model) can bear at a layer counted
 * 1-60 (UI layer = code layer + 1). Erratum E49 sets the table; the bands are measured with that lens only.
 * - "untested", layers 1-20: nothing was validated there, not even comparisons. J++ condition means track behaviour at
 *   r 0.31 at code L10-19 against 0.88 at L20-29 (E2 §6); the average Jacobian cannot be estimated below code L20 (E3).
 * - "order", layers 21-30: only the order of conditions, checked against behaviour (r 0.88-0.99 in E1, E2 and E5;
 *   erratum E49); no average-Jacobian lens tested was stable below code L30 (R1: directional split-half at most 0.63).
 * - "compare", layers 31-50: condition contrasts against a matched control (erratum E49: code L30-49); single readings
 *   stay poor (erratum E39).
 * - "late", layers 51-59: as "compare", but J++ pronoun readouts lean toward "she" there: 6-7 of 9 AI conditions at or
 *   above the woman persona at code L50-58, against 0 of 9 under JLens and 0-1 under instruct-fitted readouts (erratum
 *   E2; R1_IT_LENS_REPORT). Erratum E49: flag them unless an instruct-fitted readout agrees; the instrument loads none.
 * - "own", layer 60: the lens there is the identity, so the readout is the model's own next-word odds.
 */
export type LensBand = "untested" | "order" | "compare" | "late" | "own";
export function lensBand(layerNo: number): LensBand {
  if (layerNo >= 60) return "own";
  if (layerNo >= 51) return "late";
  if (layerNo >= 31) return "compare";
  if (layerNo >= 21) return "order";
  return "untested";
}

/** The lens a sighting's readings went through (else the server's, else the instrument's own), and whence that is known. */
export function useReadingLens(sighting?: Sighting | null): { lens: LensInfo; from: LensFrom } {
  const served = useTheodoliteStore((s) => s.health?.lens ?? null);
  if (sighting?.lens) return { lens: sighting.lens, from: "sighting" };
  if (sighting?.source === "fixture") return { lens: BASE_LENS, from: "recorded" };
  if (served) return { lens: served, from: "server" };
  return { lens: BASE_LENS, from: "unnamed" };
}

/** The lens the instrument reads through now: the server's own record (complete or not), else the current sighting's,
 * else the one the instrument loads (a server that is not answering, or an older one that sends no `lens`). */
export function useInstrumentLens(): { lens: LensInfo; from: LensFrom } {
  const served = useTheodoliteStore((s) => s.health?.lens ?? null);
  const sighting = useTheodoliteStore((s) => s.sighting);
  if (served) return { lens: served, from: "server" };
  if (sighting?.lens) return { lens: sighting.lens, from: "sighting" };
  if (sighting?.source === "fixture") return { lens: BASE_LENS, from: "recorded" };
  return { lens: BASE_LENS, from: "unnamed" };
}

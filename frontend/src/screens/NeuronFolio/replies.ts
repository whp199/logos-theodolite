/**
 * The circuit scientist's replies, read defensively. Shapes from backend/larql_vault.py
 * (inspect_bipolar_slot) and backend/mcp_scientist.py (trace_downstream_readers, test_motif_hypothesis,
 * causal_spot_check).
 * Nothing here invents a value: a field the reply lacks becomes null. Each label's `label_check` (backend/label_gate.py)
 * is read with lib/labels.ts parseLabelCheck; an older server sends none, and the pages then hold labels back by the
 * layer alone.
 */
import { parseLabelCheck, type LabelCheck } from "../../lib/labels";

export interface Tok {
  token: string;
  logit: number | null;
}

type Json = Record<string, unknown>;
const obj = (v: unknown): Json | null => (v && typeof v === "object" && !Array.isArray(v) ? (v as Json) : null);
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const str = (v: unknown): string | null => (typeof v === "string" ? v : null);
const faceOf = (v: unknown): 1 | -1 => (v === "-" || (typeof v === "number" && v < 0) ? -1 : 1);

export function toks(v: unknown): Tok[] | null {
  if (!Array.isArray(v)) return null;
  return v.map((t) => {
    const o = obj(t) ?? {};
    return { token: String(o.token ?? ""), logit: num(o.logit) };
  });
}

export interface FaceReading {
  lens: Tok[] | null;
  eye: Tok[] | null;
  /** the server's check of this face's label (ten a side); null = not sent */
  check: LabelCheck | null;
}

/** inspect_pole: both faces come back in one reply. */
export interface Inspection {
  wDownNorm: number | null;
  wGateNorm: number | null;
  wUpNorm: number | null;
  pos: FaceReading;
  neg: FaceReading;
  /** the withheld input label (gate_trigger_tokens) */
  gate: Tok[] | null;
  raw: unknown;
}

export function parseInspection(d: unknown): Inspection | null {
  const o = obj(d);
  if (!o) return null;
  const face = (p: unknown): FaceReading => {
    const f = obj(p) ?? {};
    return { lens: toks(f.j_vindex_tokens), eye: toks(f.naive_tokens), check: parseLabelCheck(f.label_check) };
  };
  return {
    wDownNorm: num(o.w_down_norm),
    wGateNorm: num(o.w_gate_norm),
    wUpNorm: num(o.w_up_norm),
    pos: face(o.north_pole),
    neg: face(o.south_pole),
    gate: toks(o.gate_trigger_tokens),
    raw: d,
  };
}

export interface Reader {
  idx: number;
  neuron: number;
  face: 1 | -1;
  act: number | null;
  dot: number | null;
  /** the coupling C */
  c: number;
  /** direct-path estimates of what the source adds to the reader's gate pre-activation and activation */
  directGate: number | null;
  directAct: number | null;
  /** the score the readers were ranked by (C, or the direct-path gate estimate) */
  score: number;
  words: Tok[] | null;
  /** the reader's words to the naked eye (dst_naive_tokens); null = not sent (an older server) */
  eye: Tok[] | null;
  /** the server's check of the reader's label (dst_label_check); null = not sent */
  check: LabelCheck | null;
}

export interface Readers {
  prompt: string | null;
  srcAct: number | null;
  srcFace: 1 | -1 | null;
  readers: Reader[];
  scoreKind: "coupling" | "direct";
  hop: number | null;
  topK: number | null;
  /** the server read the exact word-piece ids sent (token_ids_given); false: it read the prompt's text; null: not said (an
   * older server, which reads the text) */
  idsGiven: boolean | null;
  /** the position it read at, counting from 0 (read_pos); null = not sent */
  readPos: number | null;
  /** whether the trace read a nudged model (steered: always false, the trace takes no nudge); null = not sent */
  steered: boolean | null;
  raw: unknown;
}

export function parseReaders(d: unknown): Readers | null {
  const o = obj(d);
  if (!o) return null;
  const src = obj(o.source_pole);
  const list = Array.isArray(o.top_downstream_readers) ? o.top_downstream_readers : [];
  const readers: Reader[] = [];
  for (const r of list) {
    const x = obj(r);
    if (!x) continue;
    const idx = num(x.dst_layer);
    const neuron = num(x.dst_neuron_idx);
    const c = num(x.coupling_score);
    if (idx === null || neuron === null || c === null) continue;
    readers.push({
      idx,
      neuron,
      face: faceOf(x.dst_polarity),
      act: num(x.dst_signed_activation),
      dot: num(x.dot_product),
      c,
      directGate: num(x.direct_gate),
      directAct: num(x.direct_act),
      score: num(x.score) ?? c,
      words: toks(x.dst_j_vindex_tokens),
      eye: toks(x.dst_naive_tokens),
      check: parseLabelCheck(x.dst_label_check),
    });
  }
  return {
    prompt: str(o.prompt),
    srcAct: src ? num(src.signed_activation_on_prompt) : null,
    srcFace: src && src.polarity !== undefined ? faceOf(src.polarity) : null,
    readers,
    scoreKind: o.score_kind === "direct" ? "direct" : "coupling",
    hop: num(o.max_layer_hop),
    topK: num(o.top_k),
    idsGiven: typeof o.token_ids_given === "boolean" ? o.token_ids_given : null,
    readPos: num(o.read_pos),
    steered: typeof o.steered === "boolean" ? o.steered : null,
    raw: d,
  };
}

export interface HypRow {
  prompt: string;
  last: number | null;
  peak: number | null;
  peakToken: string | null;
  face: 1 | -1;
}

/** test_motif_hypothesis's test against chance (`selectivity`, backend/mcp_scientist.py motif_statistics): on the last
 * word-piece's readings, Welch's t and a p from splitting the sentences between the sides afresh. */
export interface Selectivity {
  /** SELECTIVE_NORTH_POLE (+), SELECTIVE_SOUTH_POLE (-), DIFFERS_NOT_AS_GUESSED, NOT_SHOWN or TOO_FEW_TO_JUDGE */
  verdict: string;
  /** the server's own words for it */
  words: string | null;
  nPos: number | null;
  nNeg: number | null;
  meanPos: number | null;
  meanNeg: number | null;
  /** Cohen's d over the pooled sample variance; null below two sentences a side, or when neither side varies */
  d: number | null;
  /** Welch's t and its degrees of freedom; null below two sentences a side, or for a gap with no spread at all */
  t: number | null;
  df: number | null;
  /** two-sided; null below two sentences a side */
  p: number | null;
  /** "exact" (every split) or "permutation" (`splits` shuffles, seeded `seed`) */
  pMethod: string | null;
  splits: number | null;
  seed: number | null;
  /** sentences a side below which it does not judge, and the p it judges by */
  minN: number | null;
  alpha: number | null;
}

export interface Hypothesis {
  /** as first built (a threshold on `d` at any n, nothing set against chance), kept for old callers: read `test` */
  verdict: string;
  /** as first built: Cohen's d over population variances */
  d: number | null;
  meanPos: number | null;
  meanNeg: number | null;
  /** the test against chance; null from an older server, which sends none */
  test: Selectivity | null;
  pos: HypRow[];
  neg: HypRow[];
  signature: Tok[] | null;
  /** the server's check of the pole's label (pole_label_check); null = not sent */
  signatureCheck: LabelCheck | null;
  raw: unknown;
}

function parseSelectivity(v: unknown): Selectivity | null {
  const s = obj(v);
  if (!s || typeof s.verdict !== "string") return null;
  return {
    verdict: s.verdict,
    words: str(s.words),
    nPos: num(s.n_pos),
    nNeg: num(s.n_neg),
    meanPos: num(s.mean_pos),
    meanNeg: num(s.mean_neg),
    d: num(s.cohens_d),
    t: num(s.welch_t),
    df: num(s.welch_df),
    p: num(s.p),
    pMethod: str(s.p_method),
    splits: num(s.splits),
    seed: num(s.seed),
    minN: num(s.min_n),
    alpha: num(s.alpha),
  };
}

export function parseHypothesis(d: unknown): Hypothesis | null {
  const o = obj(d);
  if (!o) return null;
  const rows = (v: unknown): HypRow[] =>
    (Array.isArray(v) ? v : []).map((r) => {
      const x = obj(r) ?? {};
      return {
        prompt: String(x.prompt ?? ""),
        last: num(x.last_token_signed_act),
        peak: num(x.peak_signed_act),
        peakToken: str(x.peak_token),
        face: faceOf(x.polarity),
      };
    });
  return {
    verdict: String(o.verdict ?? ""),
    d: num(o.cohens_d),
    meanPos: num(o.mean_pos_signed_act),
    meanNeg: num(o.mean_neg_signed_act),
    test: parseSelectivity(o.selectivity),
    pos: rows(o.pos_prompt_results),
    neg: rows(o.neg_prompt_results),
    signature: toks(o.pole_j_vindex_signature),
    signatureCheck: parseLabelCheck(o.pole_label_check),
    raw: d,
  };
}

/** The verdict as first built, in plain words: a threshold on d, at any number of sentences, with nothing set against
 * chance (one sentence a side can pass it), so it is never read as a yes. What an older server, which sends no test,
 * is shown by. */
export function verdictWords(v: string): string {
  const old = "an older server’s threshold, with nothing set against chance";
  if (v.startsWith("CONFIRMED_NORTH_POLE")) return `d at 0.8 or more on its positive face (${old})`;
  if (v.startsWith("CONFIRMED_SOUTH_POLE")) return `d at −0.8 or less on its negative face (${old})`;
  if (v.startsWith("WEAK_SELECTIVITY")) return `|d| at 0.5 or more (${old})`;
  if (v.startsWith("REJECTED_OR_INCONCLUSIVE")) return `|d| under 0.5 (${old})`;
  return v || "";
}

/** A p as the server writes it: two significant figures, or more where two would round it up to the 0.05 it is judged
 * by (0.0496 is not 0.05). */
export function pText(p: number, alpha = 0.05): string {
  let s = "";
  for (const k of [2, 3, 4, 6]) {
    s = String(Number(p.toPrecision(k)));
    if (!(p < alpha && Number(s) >= alpha)) break;
  }
  return s;
}

/** How many sentences each side had: "4 a side", or "6 and 4" (where it should fire, then where quiet). */
export const nText = (t: Selectivity): string =>
  t.nPos !== null && t.nPos === t.nNeg ? `${t.nPos} a side` : `${t.nPos ?? "?"} and ${t.nNeg ?? "?"}`;

/** The test against chance in plain words, hedged as the server's own `words` are: what it found, its p and its n. */
export function selectivityWords(t: Selectivity): string {
  const p = t.p === null ? null : pText(t.p, t.alpha ?? 0.05);
  const n = nText(t);
  if (t.verdict === "TOO_FEW_TO_JUDGE") return `too few to judge: n = ${n}, and the test needs at least ${t.minN ?? 4} a side`;
  if (t.verdict === "NOT_SHOWN") return p === null ? `no difference shown: n = ${n}` : `no difference shown: p = ${p}, n = ${n}`;
  if (t.verdict.startsWith("SELECTIVE_NORTH_POLE")) return `selective on its positive face, p = ${p}, n = ${n}`;
  if (t.verdict.startsWith("SELECTIVE_SOUTH_POLE")) return `selective on its negative face, p = ${p}, n = ${n}`;
  if (t.verdict === "DIFFERS_NOT_AS_GUESSED") {
    const quiet = t.meanPos !== null && t.meanNeg !== null && t.meanPos > t.meanNeg ? "negative" : "positive";
    return `the sides differ, p = ${p}, n = ${n}, but not as guessed: the quiet sentences fire its ${quiet} face more`;
  }
  return t.words ?? t.verdict;
}

/** What the folio says the test found: the test against chance, or an older server's threshold, hedged. */
export const hypothesisWords = (h: Hypothesis): string => (h.test ? selectivityWords(h.test) : verdictWords(h.verdict));

/** One word-piece of a causal check: its change of log-probability c and its log-probability before and after (means
 * over the texts in study mode). */
export interface CausalWord {
  tokenId: number;
  token: string;
  dlogprob: number;
  baseLogprob: number | null;
  steeredLogprob: number | null;
}

/** The label a face was compared with: its first ten ids through the lens or to the naked eye, and how many of them the
 * push raised among its hundred most (p10, null for a silenced neuron, which has no label to test). */
export interface CausalLabel {
  p10: number | null;
  ids: number[];
  tokens: string[];
  inTop100: boolean[];
}

export interface CausalLinear {
  writeOverFfn: number | null;
  /** null for the frozen-norm write (reason in `reason`) */
  linearity: number | null;
  ablationCos: number | null;
  linearityQ10: number | null;
  reason: string | null;
}

export interface CausalFace {
  delta: number | null;
  klMean: number | null;
  klMedian: number | null;
  klByContext: number[];
  /** this face's KL over the floor's of the same sign; null with the floor off or a floor KL not above 0 */
  klOverFloor: number | null;
  splitHalfRho: number | null;
  rises: CausalWord[];
  falls: CausalWord[];
  tracked: CausalWord[];
  floorTop100Overlap: number | null;
  linear: CausalLinear;
  lens: CausalLabel | null;
  eye: CausalLabel | null;
  labelK: number | null;
  causalK: number | null;
  check: LabelCheck | null;
}

export interface CausalFloor {
  delta: number | null;
  klMean: number | null;
  klMedian: number | null;
  splitHalfRho: number | null;
}

/** causal_spot_check (backend/mcp_scientist.py; ARCHITECTURE §4.5). */
export interface CausalCheck {
  layer: number | null;
  neuron: number | null;
  /** the face the reply puts first */
  polarity: 1 | -1;
  contexts: "given" | "study";
  nContexts: number | null;
  prompt: string | null;
  tokenIdsGiven: boolean | null;
  readPos: number | null;
  readToken: string | null;
  readTokenId: number | null;
  studyTexts: string | null;
  steered: boolean | null;
  write: "activation" | "frozen_norm";
  k: number | null;
  kFloor: number | null;
  scaleRef: "study" | "this_text";
  maxAbsAct: number | null;
  maxAbsActOver: string | null;
  maxAbsActCached: boolean | null;
  actHere: number | null;
  actHereByContext: number[] | null;
  pos: CausalFace | null;
  neg: CausalFace | null;
  antipodalRho: number | null;
  top10InNegTop100: number | null;
  floor: { pos: CausalFloor | null; neg: CausalFloor | null } | null;
  baseTop: { tokenId: number; token: string; logprob: number | null }[] | null;
  forwards: number | null;
  /** how many pushed rows each forward held, in order ([4] for one forward; more, split, after running out of memory);
   * null when not sent */
  slotsPerForward: number[] | null;
  baseRowsIdentical: boolean | null;
  vaultEdits: number[];
  /** the reply named the vault edits it was read under (vault_edits was a list) */
  vaultEditsSent: boolean;
  timing: { scaleMs: number | null; checkMs: number | null; totalMs: number | null } | null;
  raw: unknown;
}

const nums = (v: unknown): number[] => (Array.isArray(v) ? v.filter((x): x is number => typeof x === "number" && Number.isFinite(x)) : []);

function causalWords(v: unknown): CausalWord[] {
  const out: CausalWord[] = [];
  for (const w of Array.isArray(v) ? v : []) {
    const o = obj(w);
    const id = o ? num(o.token_id) : null;
    const c = o ? num(o.dlogprob) : null;
    if (!o || id === null || c === null) continue;
    out.push({ tokenId: id, token: String(o.token ?? ""), dlogprob: c, baseLogprob: num(o.base_logprob), steeredLogprob: num(o.steered_logprob) });
  }
  return out;
}

function causalLabel(v: unknown): CausalLabel | null {
  const o = obj(v);
  if (!o) return null;
  return {
    p10: num(o.p10),
    ids: nums(o.label_ids),
    tokens: Array.isArray(o.tokens) ? o.tokens.map((t) => String(t ?? "")) : [],
    inTop100: Array.isArray(o.in_top100) ? o.in_top100.map((x) => x === true) : [],
  };
}

function causalFace(v: unknown): CausalFace | null {
  const o = obj(v);
  if (!o) return null;
  const lin = obj(o.linear) ?? {};
  const lab = obj(o.label_overlap) ?? {};
  return {
    delta: num(o.delta),
    klMean: num(o.kl_mean),
    klMedian: num(o.kl_median),
    klByContext: nums(o.kl_by_context),
    klOverFloor: num(o.kl_over_floor),
    splitHalfRho: num(o.split_half_rho),
    rises: causalWords(o.rises),
    falls: causalWords(o.falls),
    tracked: causalWords(o.tracked),
    floorTop100Overlap: num(o.floor_top100_overlap),
    linear: {
      writeOverFfn: num(lin.write_over_ffn),
      linearity: num(lin.linearity),
      ablationCos: num(lin.ablation_cos),
      linearityQ10: num(lin.linearity_q10),
      reason: str(lin.linearity_reason),
    },
    lens: causalLabel(lab.lens),
    eye: causalLabel(lab.naked_eye),
    labelK: num(lab.label_k),
    causalK: num(lab.causal_k),
    check: parseLabelCheck(lab.label_check),
  };
}

function causalFloor(v: unknown): CausalFloor | null {
  const o = obj(v);
  return o ? { delta: num(o.delta), klMean: num(o.kl_mean), klMedian: num(o.kl_median), splitHalfRho: num(o.split_half_rho) } : null;
}

export function parseCausalCheck(d: unknown): CausalCheck | null {
  const o = obj(d);
  if (!o) return null;
  const faces = obj(o.faces) ?? {};
  const anti = obj(o.antipodal) ?? {};
  const fl = obj(o.floor);
  const t = obj(o.timing_ms);
  return {
    layer: num(o.layer),
    neuron: num(o.neuron_idx),
    polarity: faceOf(o.polarity),
    contexts: o.contexts === "study" ? "study" : "given",
    nContexts: num(o.n_contexts),
    prompt: str(o.prompt),
    tokenIdsGiven: typeof o.token_ids_given === "boolean" ? o.token_ids_given : null,
    readPos: num(o.read_pos),
    readToken: str(o.read_token),
    readTokenId: num(o.read_token_id),
    studyTexts: str(o.study_texts),
    steered: typeof o.steered === "boolean" ? o.steered : null,
    write: o.write === "frozen_norm" ? "frozen_norm" : "activation",
    k: num(o.k),
    kFloor: num(o.k_floor),
    scaleRef: o.scale_ref === "this_text" ? "this_text" : "study",
    maxAbsAct: num(o.max_abs_act),
    maxAbsActOver: str(o.max_abs_act_over),
    maxAbsActCached: typeof o.max_abs_act_cached === "boolean" ? o.max_abs_act_cached : null,
    actHere: num(o.act_here),
    actHereByContext: Array.isArray(o.act_here_by_context) ? nums(o.act_here_by_context) : null,
    pos: causalFace(faces["+"]),
    neg: causalFace(faces["-"]),
    antipodalRho: num(anti.rho),
    top10InNegTop100: num(anti.top10_in_neg_top100),
    floor: fl ? { pos: causalFloor(fl["+"]), neg: causalFloor(fl["-"]) } : null,
    baseTop: Array.isArray(o.base_top)
      ? o.base_top.flatMap((b) => {
          const x = obj(b);
          const id = x ? num(x.token_id) : null;
          return x && id !== null ? [{ tokenId: id, token: String(x.token ?? ""), logprob: num(x.logprob) }] : [];
        })
      : null,
    forwards: num(o.forwards),
    slotsPerForward: Array.isArray(o.slots_per_forward) ? nums(o.slots_per_forward) : null,
    baseRowsIdentical: typeof o.base_rows_identical === "boolean" ? o.base_rows_identical : null,
    vaultEdits: nums(o.vault_edits),
    vaultEditsSent: Array.isArray(o.vault_edits),
    timing: t ? { scaleMs: num(t.scale_ms), checkMs: num(t.check_ms), totalMs: num(t.total_ms) } : null,
    raw: d,
  };
}

/**
 * The circuit scientist's replies, read defensively. Shapes from backend/larql_vault.py
 * (inspect_bipolar_slot) and backend/mcp_scientist.py (trace_downstream_readers, test_motif_hypothesis).
 * Nothing here invents a value: a field the reply lacks becomes null.
 */

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
    return { lens: toks(f.j_vindex_tokens), eye: toks(f.naive_tokens) };
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
}

export interface Readers {
  prompt: string | null;
  srcAct: number | null;
  srcFace: 1 | -1 | null;
  readers: Reader[];
  scoreKind: "coupling" | "direct";
  hop: number | null;
  topK: number | null;
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

export interface Hypothesis {
  verdict: string;
  d: number | null;
  meanPos: number | null;
  meanNeg: number | null;
  pos: HypRow[];
  neg: HypRow[];
  signature: Tok[] | null;
  raw: unknown;
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
    pos: rows(o.pos_prompt_results),
    neg: rows(o.neg_prompt_results),
    signature: toks(o.pole_j_vindex_signature),
    raw: d,
  };
}

/** The verdict in plain words. */
export function verdictWords(v: string): string {
  if (v.startsWith("CONFIRMED_NORTH_POLE")) return "yes, clearly, on its positive face";
  if (v.startsWith("CONFIRMED_SOUTH_POLE")) return "yes, clearly, on its negative face";
  if (v.startsWith("WEAK_SELECTIVITY")) return "only weakly";
  if (v.startsWith("REJECTED_OR_INCONCLUSIVE")) return "no, not clearly";
  return v || "";
}

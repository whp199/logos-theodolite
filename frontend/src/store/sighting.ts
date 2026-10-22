/**
 * A "sighting" is one position of a sweep followed up through all sixty layers. A sweep writes an answer of n
 * word-pieces w_0 … w_{n-1} after the prompt's P word-pieces; the sighting of word-piece k is read at position
 * read_pos = P - 1 + k, where the stream has just READ the prompt's last piece (k = 0) or w_{k-1}, and where its
 * layer 60 CHOSE w_k. Token index 0 is the sighting the instrument always took, read at the prompt's last
 * word-piece. It is built either from a live sweep (binary frame + JSON metadata from /ws/theodolite, one pair per
 * word-piece sighted) or from the recorded demonstration sweep of the study (E6), which is what the approved design
 * canvas shows.
 *
 * Every field that a source does not record is `null`. Screens must show such values as bracketed
 * placeholders (see <Missing/>), never as an invented number.
 */

import { pieceWord } from "../lib/words";
import { fixtureCheck, parseLabelCheck, type LabelCheck } from "../lib/labels";

export const N_LAYERS = 60;
export const D_FF = 21504;
export const D_MODEL = 5376;
export const VOCAB = 262144;
/** Global-attention layers as repo indices (every sixth layer: 6, 12, ... 60 counted from 1). */
export const GLOBAL_IDX = new Set([5, 11, 17, 23, 29, 35, 41, 47, 53, 59]);
export const isGlobalIdx = (idx: number) => GLOBAL_IDX.has(idx);
/** Layers are counted 1-60 on every page; the repo indexes them 0-59. */
export const layerNoOf = (idx: number) => idx + 1;
export const idxOf = (layerNo: number) => layerNo - 1;

export interface TokenReading {
  token: string;
  prob: number | null;
  logit: number | null;
}

export interface PoleReading {
  rank: number; // 0 = first in the sighting's own ranking (poleRanking): the loudest by |a|, or the biggest writer
  neuron: number;
  act: number; // signed activation a = mlp_act; the face is sign(a)
  face: 1 | -1;
  /** top word-pieces of the face's write vector through the lens; null = not recorded */
  lens: TokenReading[] | null;
  /** the same through the naked eye (logit lens); null = not recorded */
  eye: TokenReading[] | null;
  dir: [number, number, number] | null;
  /** s · |a| · ‖w_down ⊙ w_post‖ / rms(y), the length of what it adds to the stream; null = not sent */
  writeSize: number | null;
  /** the label gate's check of these words (the server's label_check: the word-pieces both readings share in their
   * first ten, and whether the pages hold the label back; lib/labels.ts); null = not sent (an older server, a pole
   * whose words were not read) or, for the recorded sighting, a face its recorded checks do not cover (they cover every
   * labelled face: experiments/scripts/N1G_01_recorded_label_checks.py) */
  check: LabelCheck | null;
}

export interface LayerReading {
  idx: number;
  no: number;
  isGlobal: boolean;
  az: number | null; // bearing, degrees
  el: number | null; // elevation, degrees
  xyz: [number, number, number] | null;
  streamStrength: number | null; // ‖h‖
  projFrac: number | null; // fraction of ‖h̃‖ the frame catches
  attnNorm: number | null;
  ffnNorm: number | null;
  /** the gathering (attention) and remembering (feed-forward) changes projected on the survey frame */
  attnXyz: [number, number, number] | null;
  ffnXyz: [number, number, number] | null;
  /** share of the layer's change that is remembering (feed-forward), V7-corrected */
  ffnShare: number | null;
  /** the same as the instrument read it before the V7 fix (recorded fixture only) */
  ffnShareBeforeFix: number | null;
  layerScalar: number | null;
  rescaleNorm: number | null;
  /** what it was thinking, through the lens; null = not recorded at this layer */
  lensTop: TokenReading[] | null;
  /** the token ids of lensTop, rank for rank (the server's jpp_top_ids); null = not sent (an older server) */
  lensTopIds: number[] | null;
  /** the naked eye's top word-piece; null = not recorded */
  eyeTop: string | null;
  /** its token id (naive_top_id); null = not sent */
  eyeTopId: number | null;
  /** tracked word-piece ranks of 262,144, with their softcapped logits where recorded */
  ranks: Record<string, { lens: number | null; eye: number | null; lensLogit?: number | null; eyeLogit?: number | null }>;
  /** for each tracked word, whether its rank was taken from the next position (the legacy space-then-digit rule,
   * tracked_from_next); null = not sent */
  rankFromNext: Record<string, boolean> | null;
  poles: PoleReading[];
  /** share of the layer's feed-forward write the listed poles explain, ⟨Σ c_i, f⟩ / ⟨f, f⟩ (E3); null = not sent */
  ffnExplained: number | null;
}

export type ThreadScore = "coupling" | "direct";
export type PoleRanking = "activation" | "write";

export interface ThreadReading {
  src: { idx: number; neuron: number; face: 1 | -1 };
  dst: { idx: number; neuron: number; face: 1 | -1 };
  /** the coupling C = ⟨σ w_down ⊙ w_post, w_gate ⊙ w_pre⟩ · |a|; null when the thread was scored by the
   * direct-path estimate and the server sent only that score for it */
  c: number | null;
  dot: number | null;
  /** direct-path estimates of what the source adds to the reader's gate pre-activation and activation; null = not sent */
  directGate: number | null;
  directAct: number | null;
  /** the score the thread was chosen by: C, or the direct-path gate estimate */
  score: number;
  scoreKind: ThreadScore;
}

/** The survey frame a live sighting was taken in (the server's `frame`). */
export interface FrameInfo {
  squaring: "gram_schmidt" | "lowdin";
  perLayer: boolean;
  fnwWeighted: boolean;
  centred: boolean;
  nCentre: number;
  /** "h_J": h·Jᵀ read in a frame from the last layer; "h_raw": the raw h read in each layer's own frame */
  state: "h_J" | "h_raw";
  tokens: string[] | null;
  /** a neuron frame (pole_motif) only: its third landmark, the locked pole's dominant reader (layer as a repo index,
   * 0-59), and the word-piece whose position it was found at (k; the frame is built once, at the first position the
   * request analyses, and kept for the rest); null for a frame from words, or not sent */
  reader: { layer: number; neuron: number; fromTokenIndex: number } | null;
  /** angles and squareness of the three words, raw and final-norm weighted (V7 frame_stats definitions) */
  geometry: {
    raw: { cond: number; cos: Record<string, number> };
    fnw: { cond: number; cos: Record<string, number> };
    selfCos: number[];
  } | null;
}

/** What `name` holds when the server's card leaves out the lens's name (see LensInfo.identified); pages that print
 * "the {name} lens" then read "the unnamed lens". */
export const LENS_NAME_NOT_SENT = "unnamed";
/** What `fittedOn` holds when the server's card leaves out the model the lens was fitted on. */
export const LENS_MODEL_NOT_SENT = "a model its file does not name";

/**
 * The lens a reading was taken through, as the server names it (`lens` in GET /api/health, in every sweep and in the
 * scientist replies): the J++ lens, fitted on the base model google/gemma-4-31B and applied to the instruct 4-bit one.
 */
export interface LensInfo {
  /** "jpp-base" for the J++ lens fitted on the base model; "" = not sent */
  id: string;
  /** the method's name, e.g. "J++"; LENS_NAME_NOT_SENT when the card leaves it out */
  name: string;
  /** what it was fitted on, e.g. "google/gemma-4-31B (base, bf16)"; LENS_MODEL_NOT_SENT when the card leaves it out */
  fittedOn: string;
  /** the served checkpoint it reads; null = not sent */
  appliedTo: string | null;
  method: string;
  fitData: string;
  /** how long, in word-pieces, the passages it was fitted on were: its file's own `source_config.max_seq_len`, which the
   * server repeats as the card's `provenance.max_seq_len` (backend/engine_runtime.py, lens_card); 128 for the shipped
   * J++ file. null = not sent (an older server, or a file that does not record it): the pages then say nothing about
   * reading past it */
  fitLen: number | null;
  /** where the fitted file comes from (its provenance); null = not sent */
  source: string | null;
  /** the file the server loaded; null = not sent */
  file: string | null;
  /** the server's own word on how far to trust this lens (its card's `caution`); null = not sent */
  caution: string | null;
  /** false when the server sent a card without the lens's name or without the model it was fitted on: the pages then
   * say the lens is not identified, and never take it for the shipped base-fitted J++ */
  identified: boolean;
}

/** What the server pushed, when a sighting was nudged (the server's `steering_applied`). */
export interface SteeringApplied {
  mode: "words" | "neuron";
  direction: string;
  maskQuiet: boolean;
  spareBos: boolean;
  strength: number | null;
  scale: number | null;
  layers: number[];
}

/**
 * The word-piece a position chose (the server's STEP): w_k, written there by the model or, for a re-reading of an
 * answer already written, given to it. Its odds are the model's own, from the raw last-layer scores.
 */
export interface ChosenPiece {
  id: number;
  token: string;
  /** its share of the model's own odds, and its rank there (1 = the likeliest); null = not sent */
  modelProb: number | null;
  modelRank: number | null;
  /** how many other word-pieces had exactly its score */
  nTied: number | null;
  /** its raw last-layer score (chosen_logit); null = not sent */
  logit: number | null;
  /** its chance under the sampler that drew it (after presence penalty, temperature, top-k, top-p); null when greedy */
  samplerProb: number | null;
  /** how many word-pieces the sampler could have drawn; null when greedy */
  samplerSupport: number | null;
  /** one of the checkpoint's stop marks (generation_config.eos_token_id: 106 <turn|>, 1 <eos>, 50 <|tool_response>):
   * writing stopped here. Only <turn|> ends its turn; endMarkWords says which mark it was */
  endMark: boolean;
  source: "written" | "given" | "recorded";
  /** the five likeliest in the model's own odds at that moment; null = not sent */
  top: { id: number; token: string; prob: number }[] | null;
}

/** How an answer was written, as the server resolved and applied it (GENERATION and ANSWER in the wire protocol). */
export interface GenerationEcho {
  /** the top-scoring word-piece each time (argmax), as the study wrote the answers it checked the lens on; with a
   * repeat penalty (presencePenalty not 0) the argmax is taken after it, so not always the model's own likeliest */
  greedy: boolean;
  temperature: number | null;
  /** null when greedy (unused) or off */
  topK: number | null;
  topP: number | null;
  presencePenalty: number | null;
  seed: number | null;
  /** the server drew the seed (none was sent) */
  seedDrawn: boolean;
  maxNewTokens: number | null;
  sightEvery: boolean;
  maxSighted: number | null;
  /** more word-pieces were written than were sighted */
  capped: boolean;
  stopAtEndOfTurn: boolean | null;
  source: "written" | "given" | "recorded";
  stopReason: "end_mark" | "length" | "stopped" | "given" | "none" | null;
  nWritten: number | null;
}

/** One word-piece of the answer, as the row of word-pieces shows it. */
export interface AnswerPiece {
  /** its token index k: the sighting of k is read where the model chose it */
  index: number;
  id: number;
  /** the piece decoded alone */
  str: string;
  endMark: boolean;
  step: ChosenPiece | null;
}

export type CentreKind = "none" | "control_prompt_end" | "control_reading_same_answer";
/** How a sighting with thinking turned on was switched on (the server's think_protocol). */
export type ThinkProtocol = "template" | "bare_prefix";

export interface Sighting {
  /** a number unique to this sighting in this page's life, so results asked of it can be told from older ones */
  id: number;
  source: "live" | "fixture";
  /** one line saying where this sighting came from */
  provenance: string;
  /** the request it answers, as the server echoed it; null for the fixture and an older server */
  requestId: string | null;
  /** which word-piece of the answer it is the sighting of (k): 0 for the fixture and an older server */
  tokenIndex: number;
  question: string;
  answerBegins: string;
  /** the exact text sent to the server; null when it was not recorded (the fixture's chat-wrapped prompt) */
  textSent: string | null;
  chatWrapped: boolean;
  /** the prompt's word-pieces (the prompt only, not the answer) */
  promptTokens: string[] | null;
  /** word-pieces read up to and including the one read here (kinship_n_states = read_pos + 1); an older server: the
   * prompt's length */
  nTokens: number | null;
  /** the position read, counting from 0 (read_pos = P - 1 + k) */
  readPos: number | null;
  /** how far back the model's local-attention layers look, in word-pieces, as the server read it from the model's own
   * config (`sliding_window`, sent beside read_pos; text_config.sliding_window = 1024 for this checkpoint): from
   * read_pos = slidingWindow on, those layers no longer reach position 0. null for the recorded sighting and an older
   * server, which do not send it */
  slidingWindow: number | null;
  /** the word-piece the stream had just read there: the prompt's last for k = 0, else w_{k-1} */
  readAt: string | null;
  readAtId: number | null;
  /** the prompt's length in word-pieces (P) */
  promptLen: number | null;
  firstIds: number[] | null;
  /** the exact ids up to and including read_pos: the prompt's, then the answer's first k */
  contextIds: number[] | null;
  /** the prompt's ids */
  promptIds: number[] | null;
  /** the word-piece chosen here (w_k); null when nothing was written, or not sent */
  chosen: ChosenPiece | null;
  /** the whole answer, as ids and as each id decoded alone; null = not sent */
  answerIds: number[] | null;
  answerStrs: string[] | null;
  /** each answer piece's stats, by token index; null = not sent */
  answerSteps: (ChosenPiece | null)[] | null;
  generation: GenerationEcho | null;
  /** layer 60 of the re-reading at this position: does its top-1 equal the piece chosen? (writing and re-reading are
   * different 4-bit computations); null = not sent. chosenLogitDiff is the re-read's score for the chosen piece less
   * the writing's own: null for a given answer (chosen.source "given"), whose odds are the re-reading's own, so there
   * is no writing to compare with (an older server sent 0 there; read it only when chosen.source is "written") */
  recaptureCheck: { top1Id: number; top1Token: string; top1IsChosen: boolean; chosenLogitDiff: number | null } | null;
  /** whether the legacy space-then-digit rank rule was allowed; null = not sent */
  rankLookahead: boolean | null;
  /** what the frame was centred on at this position; null = not sent */
  centreKind: CentreKind | null;
  timing: { generateMs: number | null; captureMs: number | null; analyseMs: number | null; totalMs: number | null } | null;
  continuation: string;
  thinkingContinuation: string | null;
  thinkingPath: [number, number, number][] | null;
  /** the seed the thinking answer was drawn with, and the word-piece it was read at (first pair sent only) */
  thinkSeed: number | null;
  thinkReadAt: string | null;
  /** why the thinking answer stopped (the server's think_stop_reason, first pair only): "end_mark", "length", or
   * "stopped" when "stop here" cut it short; null when there was no thinking comparison, or an older server */
  thinkStopReason: string | null;
  /** the thinking comparison was asked for but not written, because "stop here" was asked before it began (the
   * server's think_skipped, first pair only); false otherwise, and for an older server */
  thinkSkipped: boolean;
  /** how the thinking answer was switched on (the server's think_protocol, first pair only): "template" (a chat: the
   * template's own switch, <|think|> at the head of a system turn, the model's turn begun with no empty thought) or
   * "bare_prefix" (<|think|> set before bare text: the old imitation, not Gemma 4's switch); null when none was
   * written; undefined when the server did not say (an older server, which always used one of the two) */
  thinkProtocol: ThinkProtocol | null | undefined;
  /** "bare_text" when thinking was asked for and left out, since bare text has no turn to think in (the server's
   * think_left_out, first pair only, with think_bare "none"); null otherwise, and for an older server */
  thinkLeftOut: "bare_text" | null;
  landmarks: [string, string, string];
  surveyMode: string;
  elapsedMs: number | null;
  peakVramGb: number | null;
  isNudged: boolean;
  /** the system turn the question was sent with (chat only); null = none */
  system: string | null;
  poleRanking: PoleRanking;
  threadScore: ThreadScore;
  frame: FrameInfo | null;
  /** the lens the server says it read through; null for the recorded sighting, which keeps no such record (it was
   * taken through the same J++ lens, fitted on the base model), and for an older server whose reply has no `lens` */
  lens: LensInfo | null;
  steeringApplied: SteeringApplied | null;
  /** the nudge settings this page sent with the request (live only), to tell this nudge from an earlier one */
  steeringSent: Record<string, unknown> | null;
  layers: LayerReading[];
  /** 60 x 60 linear CKA, row-major; null = not recorded */
  kinship: number[] | null;
  kinStats: { meanOffDiag: number; adjacent: number; l11vs51: number } | null;
  /** every thread the server sent, strongest score first (the recorded sighting keeps only its top 12) */
  threads: ThreadReading[];
  threadCount: number | null;
  activeEdits: ActiveEdit[];
  /** the payload that produced it (live only) */
  raw: unknown;
}

export interface ActiveEdit {
  patch_id: number;
  op_type: string;
  layer_idx: number;
  neuron_idx: number;
  target_concept: string;
  statement: string;
  polarity?: number;
  strength?: number;
  timestamp?: number;
  orig_norm?: number;
  new_norm?: number;
}

export function kinshipStats(k: number[] | null) {
  if (!k || k.length !== N_LAYERS * N_LAYERS) return null;
  let off = 0;
  let n = 0;
  let adj = 0;
  for (let i = 0; i < N_LAYERS; i++) {
    for (let j = 0; j < N_LAYERS; j++) {
      if (i === j) continue;
      off += k[i * N_LAYERS + j];
      n++;
    }
    if (i < N_LAYERS - 1) adj += k[i * N_LAYERS + i + 1];
  }
  return { meanOffDiag: off / n, adjacent: adj / (N_LAYERS - 1), l11vs51: k[10 * N_LAYERS + 50] };
}

/* ------------------------------------------------------------------ live */

export interface LiveBinary {
  nLayers: number;
  nPoles: number;
  nFilaments: number;
  flags: number;
  trajXyz: Float32Array;
  trajAnglesNorm: Float32Array;
  deltaAttnXyz: Float32Array;
  deltaFfnXyz: Float32Array;
  ckaMatrix: Float32Array;
  polesData: Float32Array;
  filamentsData: Float32Array;
  thinkTrajXyz: Float32Array | null;
}

export function parseBinaryFrame(buf: ArrayBuffer): LiveBinary | null {
  const dv = new DataView(buf);
  if (buf.byteLength < 24 || dv.getUint32(0, true) !== 0x5448454f) return null;
  const nLayers = dv.getUint32(8, true);
  const nPoles = dv.getUint32(12, true);
  const nFilaments = dv.getUint32(16, true);
  const flags = dv.getUint32(20, true);
  const floats = new Float32Array(buf, 24);
  let offset = 0;
  const take = (count: number) => {
    const slice = floats.slice(offset, offset + count);
    offset += count;
    return slice;
  };
  return {
    nLayers,
    nPoles,
    nFilaments,
    flags,
    trajXyz: take(nLayers * 3),
    trajAnglesNorm: take(nLayers * 3),
    deltaAttnXyz: take(nLayers * 4),
    deltaFfnXyz: take(nLayers * 4),
    ckaMatrix: take(nLayers * nLayers),
    polesData: take(nLayers * nPoles * 6),
    filamentsData: take(nFilaments * 7),
    thinkTrajXyz: (flags & 1) !== 0 ? take(nLayers * 3) : null,
  };
}

type Json = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any

let nextSightingId = 1;

const numOrNull = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
/** A length or count the server sent (a lens's fit length, the attention window): a whole number above 0, else null. */
const positiveIntOrNull = (v: unknown): number | null => (typeof v === "number" && Number.isInteger(v) && v > 0 ? v : null);
/** A list of whole numbers (token ids), or null when the server did not send one. */
const idsOrNull = (v: unknown): number[] | null =>
  Array.isArray(v) && v.every((x) => typeof x === "number" && Number.isInteger(x)) ? (v as number[]).slice() : null;
/** A list of word-pieces, or null when none were read: the server sends [] for poles it did not decode, and an
 * empty list must not read as "read, and no word in common". */
const toks = (list: unknown): TokenReading[] | null =>
  Array.isArray(list) && list.length > 0
    ? list.map((t: Json) => ({ token: String(t.token ?? ""), prob: numOrNull(t.prob), logit: numOrNull(t.logit) }))
    : null;

/**
 * Every thread of a live sighting. The JSON keeps only the first 60 the server found, in source-layer order (not
 * the strongest), so the threads are rebuilt from the binary rows (src layer, src rank, dst layer, dst rank, score,
 * src face, dst face), with each rank turned into its neuron through that layer's poles. The JSON's extra fields
 * (C, the dot product, the direct-path estimates) are joined on where it has the same thread.
 */
function liveThreads(meta: Json, bin: LiveBinary | null, kind: ThreadScore): ThreadReading[] {
  const metaRows: Json[] = Array.isArray(meta.filaments_meta) ? meta.filaments_meta : [];
  const fromMeta = (f: Json): ThreadReading => ({
    src: { idx: f.src_layer, neuron: f.src_neuron, face: f.src_polarity === "-" ? -1 : 1 },
    dst: { idx: f.dst_layer, neuron: f.dst_neuron, face: f.dst_polarity === "-" ? -1 : 1 },
    c: numOrNull(f.coupling_score),
    dot: numOrNull(f.dot_product),
    directGate: numOrNull(f.direct_gate),
    directAct: numOrNull(f.direct_act),
    score: numOrNull(f.score) ?? Number(f.coupling_score),
    scoreKind: f.score_kind === "direct" ? "direct" : "coupling",
  });
  let out: ThreadReading[];
  if (bin && bin.nFilaments > 0 && bin.nPoles > 0) {
    const key = (a: number, an: number, b: number, bn: number) => `${a}:${an}>${b}:${bn}`;
    const known = new Map(metaRows.map((f) => [key(f.src_layer, f.src_neuron, f.dst_layer, f.dst_neuron), f]));
    const neuronAt = (layer: number, rank: number) =>
      rank >= 0 && rank < bin.nPoles ? Math.round(bin.polesData[(layer * bin.nPoles + rank) * 6]) : null;
    out = [];
    for (let i = 0; i < bin.nFilaments; i++) {
      const row = bin.filamentsData.subarray(i * 7, i * 7 + 7);
      const [sl, sr, dl, dr, score, sp, dp] = Array.from(row);
      const sn = neuronAt(Math.round(sl), Math.round(sr));
      const dn = neuronAt(Math.round(dl), Math.round(dr));
      if (sn === null || dn === null) continue;
      const m = known.get(key(Math.round(sl), sn, Math.round(dl), dn));
      out.push({
        src: { idx: Math.round(sl), neuron: sn, face: sp >= 0 ? 1 : -1 },
        dst: { idx: Math.round(dl), neuron: dn, face: dp >= 0 ? 1 : -1 },
        c: kind === "coupling" ? score : m ? numOrNull(m.coupling_score) : null,
        dot: m ? numOrNull(m.dot_product) : null,
        directGate: kind === "direct" ? score : m ? numOrNull(m.direct_gate) : null,
        directAct: m ? numOrNull(m.direct_act) : null,
        score,
        scoreKind: kind,
      });
    }
  } else {
    out = metaRows.map(fromMeta);
  }
  return out.sort((a, b) => Math.abs(b.score) - Math.abs(a.score));
}

export function sightingFromLive(
  meta: Json,
  bin: LiveBinary | null,
  req: { question: string; answerBegins: string; textSent: string; steering?: Record<string, unknown> | null },
): Sighting {
  const plumes: Json[] = Array.isArray(meta.layer_plumes) ? meta.layer_plumes : [];
  const labels: Record<string, Json[]> = meta.poles_labels ?? {};
  const nPoles = bin?.nPoles ?? 0;
  const layers: LayerReading[] = [];
  for (let l = 0; l < N_LAYERS; l++) {
    const p: Json = plumes[l] ?? {};
    const poles: PoleReading[] = [];
    if (bin) {
      for (let r = 0; r < nPoles; r++) {
        const base = (l * nPoles + r) * 6;
        const pol = bin.polesData[base + 1] >= 0 ? 1 : -1;
        const lab = (labels[String(l)] ?? []).find((x) => x.rank === r);
        poles.push({
          rank: r,
          neuron: Math.round(bin.polesData[base]),
          act: pol * bin.polesData[base + 2],
          face: pol as 1 | -1,
          lens: lab ? toks(lab.j_vindex_tokens) : null,
          eye: lab ? toks(lab.naive_tokens) : null,
          dir: [bin.polesData[base + 3], bin.polesData[base + 4], bin.polesData[base + 5]],
          writeSize: Array.isArray(p.pole_write_sizes) ? numOrNull(p.pole_write_sizes[r]) : null,
          check: lab ? parseLabelCheck(lab.label_check) : null,
        });
      }
    } else {
      for (const lab of labels[String(l)] ?? []) {
        const pol = lab.polarity === "-" ? -1 : 1;
        poles.push({
          rank: lab.rank,
          neuron: lab.neuron_idx,
          act: pol * (numOrNull(lab.abs_act) ?? 0),
          face: pol as 1 | -1,
          lens: toks(lab.j_vindex_tokens),
          eye: toks(lab.naive_tokens),
          dir: null,
          writeSize: numOrNull(lab.write_size),
          check: parseLabelCheck(lab.label_check),
        });
      }
    }
    const tracked: Record<string, number> = p.tracked_ranks ?? {};
    const trackedLogits: Record<string, number> = p.tracked_logits ?? {};
    const ranks: LayerReading["ranks"] = {};
    for (const [k, v] of Object.entries(tracked)) ranks[k] = { lens: numOrNull(v), eye: null, lensLogit: numOrNull(trackedLogits[k]), eyeLogit: null };
    const attn = numOrNull(p.attn_norm) ?? (bin ? bin.deltaAttnXyz[l * 4 + 3] : null);
    const ffn = numOrNull(p.ffn_norm) ?? (bin ? bin.deltaFfnXyz[l * 4 + 3] : null);
    layers.push({
      idx: l,
      no: l + 1,
      isGlobal: typeof p.is_global_attention === "boolean" ? p.is_global_attention : isGlobalIdx(l),
      az: numOrNull(p.azimuth_deg) ?? (bin ? bin.trajAnglesNorm[l * 3] : null),
      el: numOrNull(p.elevation_deg) ?? (bin ? bin.trajAnglesNorm[l * 3 + 1] : null),
      xyz: bin ? [bin.trajXyz[l * 3], bin.trajXyz[l * 3 + 1], bin.trajXyz[l * 3 + 2]] : null,
      streamStrength: numOrNull(p.residual_norm) ?? (bin ? bin.trajAnglesNorm[l * 3 + 2] : null),
      projFrac: numOrNull(p.proj_frac),
      attnNorm: attn,
      ffnNorm: ffn,
      attnXyz: bin ? [bin.deltaAttnXyz[l * 4], bin.deltaAttnXyz[l * 4 + 1], bin.deltaAttnXyz[l * 4 + 2]] : null,
      ffnXyz: bin ? [bin.deltaFfnXyz[l * 4], bin.deltaFfnXyz[l * 4 + 1], bin.deltaFfnXyz[l * 4 + 2]] : null,
      ffnShare: numOrNull(p.ffn_dominance) ?? (attn !== null && ffn !== null && attn + ffn > 0 ? ffn / (attn + ffn) : null),
      ffnShareBeforeFix: null,
      layerScalar: numOrNull(p.layer_scalar),
      rescaleNorm: numOrNull(p.rescale_norm),
      lensTop: toks(p.jpp_top_tokens),
      lensTopIds: idsOrNull(p.jpp_top_ids),
      eyeTop: typeof p.naive_top_token === "string" ? p.naive_top_token : null,
      eyeTopId: numOrNull(p.naive_top_id),
      ranks,
      rankFromNext:
        p.tracked_from_next && typeof p.tracked_from_next === "object"
          ? Object.fromEntries(Object.entries(p.tracked_from_next as Json).map(([k, v]) => [k, v === true]))
          : null,
      poles,
      ffnExplained: numOrNull(p.poles_ffn_explained),
    });
  }
  const threadScore: ThreadScore = meta.thread_score === "direct" ? "direct" : "coupling";
  const threads = liveThreads(meta, bin, threadScore);
  const kinship = bin && bin.ckaMatrix.length === N_LAYERS * N_LAYERS ? Array.from(bin.ckaMatrix) : null;
  const tokensList: string[] | null = Array.isArray(meta.prompt_token_strs) ? meta.prompt_token_strs : null;
  const think = bin?.thinkTrajXyz;
  const promptIds = idsOrNull(meta.prompt_token_ids);
  const answer = answerOf(meta.answer);
  const k = numOrNull(meta.token_index) ?? 0;
  // where it was read: the server's own read_pos; an older server reads the prompt's last word-piece
  const sentPos = numOrNull(meta.read_pos);
  const readPos = sentPos ?? (tokensList ? tokensList.length - 1 : null);
  const readAt =
    sentPos !== null
      ? typeof meta.read_token === "string"
        ? meta.read_token
        : null
      : tokensList && tokensList.length
        ? tokensList[tokensList.length - 1]
        : null;
  const contextIds =
    promptIds && answer && k <= answer.ids.length ? [...promptIds, ...answer.ids.slice(0, k)] : promptIds && k === 0 ? promptIds : null;
  const rc: Json | null = meta.recapture_check && typeof meta.recapture_check === "object" ? meta.recapture_check : null;
  const tm: Json | null = meta.timing && typeof meta.timing === "object" ? meta.timing : null;
  const centreKind = meta.frame?.centre_kind;
  return {
    id: nextSightingId++,
    source: "live",
    provenance: "a live sighting from the running server",
    requestId: typeof meta.request_id === "string" ? meta.request_id : null,
    tokenIndex: k,
    question: req.question,
    answerBegins: req.answerBegins,
    textSent: typeof meta.prompt === "string" ? meta.prompt : req.textSent,
    chatWrapped: meta.chat === true,
    promptTokens: tokensList,
    nTokens: numOrNull(meta.kinship_n_states) ?? (sentPos !== null ? sentPos + 1 : tokensList ? tokensList.length : null),
    readPos,
    slidingWindow: positiveIntOrNull(meta.sliding_window),
    readAt,
    readAtId: numOrNull(meta.read_token_id),
    promptLen: numOrNull(meta.prompt_len) ?? (promptIds ? promptIds.length : null),
    firstIds: promptIds ? promptIds.slice(0, 4) : null,
    contextIds,
    promptIds,
    chosen: stepOf(meta.chosen),
    answerIds: answer ? answer.ids : null,
    answerStrs: answer ? answer.strs : null,
    answerSteps: answer ? answer.steps : null,
    generation: generationOf(meta.generation, meta.answer),
    recaptureCheck:
      rc && numOrNull(rc.top1_id) !== null
        ? {
            top1Id: Number(rc.top1_id),
            top1Token: String(rc.top1_token ?? ""),
            top1IsChosen: rc.top1_is_chosen === true,
            chosenLogitDiff: numOrNull(rc.chosen_logit_diff),
          }
        : null,
    rankLookahead: typeof meta.rank_lookahead === "boolean" ? meta.rank_lookahead : null,
    centreKind:
      centreKind === "none" || centreKind === "control_prompt_end" || centreKind === "control_reading_same_answer" ? centreKind : null,
    timing: tm
      ? {
          generateMs: numOrNull(tm.generate_ms),
          captureMs: numOrNull(tm.capture_ms),
          analyseMs: numOrNull(tm.analyse_ms),
          totalMs: numOrNull(tm.total_ms),
        }
      : null,
    continuation: String(meta.generated_text ?? ""),
    thinkingContinuation: typeof meta.think_generated_text === "string" ? meta.think_generated_text : null,
    thinkingPath: think
      ? Array.from({ length: N_LAYERS }, (_, l) => [think[l * 3], think[l * 3 + 1], think[l * 3 + 2]] as [number, number, number])
      : null,
    thinkSeed: numOrNull(meta.think_seed),
    thinkReadAt: typeof meta.think_read_token === "string" ? meta.think_read_token : null,
    thinkStopReason: typeof meta.think_stop_reason === "string" && meta.think_stop_reason ? meta.think_stop_reason : null,
    thinkSkipped: meta.think_skipped === true,
    thinkProtocol: !("think_protocol" in meta)
      ? undefined
      : meta.think_protocol === "template" || meta.think_protocol === "bare_prefix"
        ? meta.think_protocol
        : null,
    thinkLeftOut: meta.think_left_out === "bare_text" ? "bare_text" : null,
    landmarks: (Array.isArray(meta.survey_labels) && meta.survey_labels.length === 3
      ? meta.survey_labels
      : ["", "", ""]) as [string, string, string],
    surveyMode: String(meta.survey_mode ?? "monument"),
    elapsedMs: numOrNull(meta.elapsed_ms),
    peakVramGb: numOrNull(meta.peak_vram_gb),
    isNudged: Boolean(meta.is_steered) || (bin ? (bin.flags & 2) !== 0 : false),
    system: typeof meta.system === "string" && meta.system ? meta.system : null,
    poleRanking: meta.pole_ranking === "write" ? "write" : "activation",
    threadScore,
    frame: frameOf(meta.frame),
    lens: lensOf(meta.lens),
    steeringApplied: steeringOf(meta.steering_applied),
    steeringSent: req.steering ?? null,
    layers,
    kinship,
    kinStats: kinshipStats(kinship),
    threads,
    threadCount: numOrNull(meta.n_filaments) ?? (bin ? bin.nFilaments : null),
    activeEdits: Array.isArray(meta.larql_status?.active_patches) ? meta.larql_status.active_patches : [],
    raw: meta,
  };
}

/** A neuron frame's reader {layer, neuron, from_token_index}, or null when the frame has none (or it is malformed). */
function readerOf(r: unknown): FrameInfo["reader"] {
  if (!r || typeof r !== "object" || Array.isArray(r)) return null;
  const o = r as Json;
  const layer = numOrNull(o.layer);
  const neuron = numOrNull(o.neuron);
  const at = numOrNull(o.from_token_index);
  return layer !== null && neuron !== null && at !== null ? { layer, neuron, fromTokenIndex: at } : null;
}

function frameOf(f: Json | null | undefined): FrameInfo | null {
  if (!f || typeof f !== "object") return null;
  const g: Json | null = f.geometry ?? null;
  return {
    squaring: f.squaring === "lowdin" ? "lowdin" : "gram_schmidt",
    perLayer: f.per_layer === true,
    fnwWeighted: f.fnw_weighted === true,
    centred: f.centred === true,
    nCentre: numOrNull(f.n_centre_prompts) ?? 0,
    state: f.state === "h_raw" ? "h_raw" : "h_J",
    tokens: Array.isArray(f.tokens) ? f.tokens.map(String) : null,
    reader: readerOf(f.reader),
    geometry:
      g && g.raw && g.fnw
        ? {
            raw: { cond: Number(g.raw.cond), cos: g.raw.cos ?? {} },
            fnw: { cond: Number(g.fnw.cond), cos: g.fnw.cos ?? {} },
            selfCos: Array.isArray(g.self_cos) ? g.self_cos.map(Number) : [],
          }
        : null,
  };
}

/**
 * The server's `lens` card, or null when it sent none (an older server, or one whose lens is not loaded yet; the pages
 * then fall back to the lens the instrument loads). A card that leaves out the lens's name or the model it was fitted
 * on is kept, as sent, with `identified` false: it describes some other lens file, which must not be taken for the
 * shipped one (backend/engine_runtime.py, lens_card).
 */
export function lensOf(l: unknown): LensInfo | null {
  if (!l || typeof l !== "object" || Array.isArray(l)) return null;
  const o = l as Json;
  const str = (v: unknown) => (typeof v === "string" && v.trim() ? v : null);
  // the server sends "unnamed" when the file records no method (engine_runtime.UNNAMED_LENS): not a name
  const sentName = str(o.name);
  const name = sentName === LENS_NAME_NOT_SENT ? null : sentName;
  const fittedOn = str(o.fitted_on);
  // the file's own fit length, as the card repeats it (provenance.max_seq_len; None when the file does not record it)
  const prov = o.provenance && typeof o.provenance === "object" && !Array.isArray(o.provenance) ? (o.provenance as Json) : null;
  return {
    id: str(o.id) ?? "",
    name: name ?? LENS_NAME_NOT_SENT,
    fittedOn: fittedOn ?? LENS_MODEL_NOT_SENT,
    appliedTo: str(o.applied_to),
    method: str(o.method) ?? "",
    fitData: str(o.fit_data) ?? "",
    fitLen: prov ? positiveIntOrNull(prov.max_seq_len) : null,
    source: str(o.source),
    file: str(o.file),
    caution: str(o.caution),
    identified: name !== null && fittedOn !== null,
  };
}

function steeringOf(a: Json | null | undefined): SteeringApplied | null {
  if (!a || typeof a !== "object") return null;
  return {
    mode: a.mode === "neuron" ? "neuron" : "words",
    direction: String(a.direction ?? ""),
    maskQuiet: a.mask_quiet === true,
    spareBos: a.spare_bos === true,
    strength: numOrNull(a.strength),
    scale: numOrNull(a.scale),
    layers: Array.isArray(a.layers) ? a.layers.map(Number) : [],
  };
}

/** A STEP from the server (the piece chosen at one position, with its odds), or null when none was sent. */
function stepOf(j: unknown): ChosenPiece | null {
  if (!j || typeof j !== "object" || Array.isArray(j)) return null;
  const o = j as Json;
  const id = numOrNull(o.token_id);
  if (id === null) return null;
  return {
    id,
    token: typeof o.token === "string" ? o.token : "",
    modelProb: numOrNull(o.model_prob),
    modelRank: numOrNull(o.model_rank),
    nTied: numOrNull(o.n_tied),
    logit: numOrNull(o.chosen_logit),
    samplerProb: numOrNull(o.sampler_prob),
    samplerSupport: numOrNull(o.sampler_support),
    endMark: o.end_mark === true,
    source: o.source === "given" ? "given" : "written",
    top: Array.isArray(o.top)
      ? o.top
          .filter((t: Json) => t && numOrNull(t.token_id) !== null)
          .map((t: Json) => ({ id: Number(t.token_id), token: String(t.token ?? ""), prob: numOrNull(t.prob) ?? 0 }))
      : null,
  };
}

/** The server's ANSWER: its ids, each id decoded alone, and each piece's stats; null when it sent none. */
function answerOf(a: unknown): { ids: number[]; strs: string[] | null; steps: (ChosenPiece | null)[] | null } | null {
  if (!a || typeof a !== "object" || Array.isArray(a)) return null;
  const o = a as Json;
  const ids = idsOrNull(o.token_ids);
  if (!ids) return null;
  const strs = Array.isArray(o.token_strs) && o.token_strs.length === ids.length ? o.token_strs.map((t: unknown) => String(t ?? "")) : null;
  const steps = Array.isArray(o.steps) ? ids.map((_, i) => stepOf(o.steps[i])) : null;
  return { ids, strs, steps };
}

const STOP_REASONS = ["end_mark", "length", "stopped", "given", "none"] as const;

/**
 * How the answer was written, from the server's GENERATION and ANSWER echoes (sent with theodolite_written and with
 * every pair); null when the server sent no GENERATION (an older server, which always wrote greedily).
 */
export function generationOf(gen: unknown, answer?: unknown): GenerationEcho | null {
  if (!gen || typeof gen !== "object" || Array.isArray(gen)) return null;
  const g = gen as Json;
  const a: Json = answer && typeof answer === "object" && !Array.isArray(answer) ? (answer as Json) : {};
  const st: Json = g.settings && typeof g.settings === "object" ? g.settings : {};
  const reason = STOP_REASONS.find((r) => r === a.stop_reason) ?? null;
  return {
    greedy: g.greedy === true,
    temperature: numOrNull(st.temperature),
    topK: numOrNull(st.top_k),
    topP: numOrNull(st.top_p),
    presencePenalty: numOrNull(st.presence_penalty),
    seed: numOrNull(g.seed),
    seedDrawn: g.seed_drawn === true,
    maxNewTokens: numOrNull(g.max_new_tokens),
    sightEvery: g.sight_every_token === true,
    maxSighted: numOrNull(g.max_sighted),
    capped: g.capped === true,
    stopAtEndOfTurn: typeof g.stop_at_end_of_turn === "boolean" ? g.stop_at_end_of_turn : null,
    source: a.source === "given" ? "given" : "written",
    stopReason: reason,
    nWritten: numOrNull(a.n_written),
  };
}

function piecesFrom(ids: number[], strs: string[], steps: (ChosenPiece | null)[] | null, endedOnMark: boolean): AnswerPiece[] {
  return ids.map((id, i) => {
    const step = steps?.[i] ?? null;
    // without steps, the stop reason still says whether the last piece was an end mark
    const endMark = step ? step.endMark : endedOnMark && i === ids.length - 1;
    return { index: i, id, str: strs[i], endMark, step };
  });
}

/** The pieces of a server ANSWER (theodolite_written's `answer`), or null when it sent none. */
export function piecesFromAnswer(answer: unknown): AnswerPiece[] | null {
  const a = answerOf(answer);
  if (!a || !a.strs) return null;
  return piecesFrom(a.ids, a.strs, a.steps, (answer as Json).stop_reason === "end_mark");
}

/* --------------------------------------------------------------- fixture */

export interface FixtureFile {
  source: string;
  question: string;
  answerBegins: string;
  textSent: string | null;
  readAt: string;
  readPos: number;
  nTokens: number;
  firstIds: number[];
  continuation: string;
  landmarks: [string, string, string];
  az: number[];
  el: number[];
  xyz: [number, number, number][];
  projFrac: number[];
  ffnShare: number[];
  ffnShareBeforeFix: number[];
  layerScalar: number[];
  lensTop: Record<string, [string, number][]>;
  ranks: { lens: Record<string, number[]>; eye: Record<string, number[]> };
  rankLogits?: { lens: Record<string, number[]>; eye: Record<string, number[]> };
  poleNeurons: Record<string, number[]>;
  poleActs: Record<string, number[]>;
  poleLabels: { lens: Record<string, [string, number][]>; eye: Record<string, [string, number][]> };
  threads: { src: [number, number, number]; dst: [number, number, number]; c: number; dot: number }[];
  threadCount: number;
  kinship: number[];
  kinStats: { meanOffDiag: number; adjacent: number; l11vs51: number };
  elapsedMs: number;
}

/** How the study's recorded sighting was written: greedily (HF greedy decoding), 6 word-pieces; nothing else recorded. */
export const RECORDED_GENERATION: GenerationEcho = {
  greedy: true,
  temperature: null,
  topK: null,
  topP: null,
  presencePenalty: null,
  seed: null,
  seedDrawn: false,
  maxNewTokens: 6,
  sightEvery: false,
  maxSighted: null,
  capped: false,
  stopAtEndOfTurn: null,
  source: "recorded",
  stopReason: null,
  nWritten: null,
};

export function sightingFromFixture(f: FixtureFile): Sighting {
  const layers: LayerReading[] = [];
  for (let l = 0; l < N_LAYERS; l++) {
    const neurons = f.poleNeurons[String(l)] ?? [];
    const acts = f.poleActs[String(l)] ?? [];
    const poles: PoleReading[] = neurons.map((n, r) => {
      const lens = f.poleLabels.lens[`${l}:${r}`];
      const eye = f.poleLabels.eye[`${l}:${r}`];
      return {
        rank: r,
        neuron: n,
        act: acts[r] ?? 0,
        face: (acts[r] ?? 0) >= 0 ? 1 : -1,
        lens: lens ? lens.map(([token, logit]) => ({ token, prob: null, logit })) : null,
        eye: eye ? eye.map(([token, logit]) => ({ token, prob: null, logit })) : null,
        dir: null,
        writeSize: null,
        // checked ten a side from the weights, once (N1G_01_recorded_label_checks.py), or null until then
        check: lens ? fixtureCheck(l, r) : null,
      };
    });
    const ranks: LayerReading["ranks"] = {};
    for (const w of Object.keys(f.ranks.lens)) {
      ranks[w] = {
        lens: f.ranks.lens[w]?.[l] ?? null,
        eye: f.ranks.eye[w]?.[l] ?? null,
        lensLogit: f.rankLogits?.lens[w]?.[l] ?? null,
        eyeLogit: f.rankLogits?.eye[w]?.[l] ?? null,
      };
    }
    const top = f.lensTop[String(l)];
    layers.push({
      idx: l,
      no: l + 1,
      isGlobal: isGlobalIdx(l),
      az: f.az[l],
      el: f.el[l],
      xyz: f.xyz[l],
      streamStrength: null,
      projFrac: f.projFrac[l],
      attnNorm: null,
      ffnNorm: null,
      attnXyz: null,
      ffnXyz: null,
      ffnShare: f.ffnShare[l],
      ffnShareBeforeFix: f.ffnShareBeforeFix[l],
      layerScalar: f.layerScalar[l] ?? null,
      rescaleNorm: null,
      lensTop: top ? top.map(([token, prob]) => ({ token, prob, logit: null })) : null,
      lensTopIds: null,
      eyeTop: null,
      eyeTopId: null,
      ranks,
      rankFromNext: null,
      poles,
      ffnExplained: null,
    });
  }
  return {
    id: nextSightingId++,
    source: "fixture",
    provenance: f.source,
    requestId: null,
    tokenIndex: 0,
    question: f.question,
    answerBegins: f.answerBegins,
    textSent: f.textSent,
    chatWrapped: true,
    promptTokens: null,
    nTokens: f.nTokens,
    readPos: f.readPos,
    // the recording keeps no attention window (and, read at position 21, is far inside it)
    slidingWindow: null,
    readAt: f.readAt,
    readAtId: null,
    // the recorded sighting was read at the prompt's last word-piece, so its prompt is the word-pieces it read
    promptLen: f.nTokens,
    firstIds: f.firstIds,
    contextIds: null,
    promptIds: null,
    // the recording keeps the answer's text only, not its first piece's id: no choice is invented for it
    chosen: null,
    answerIds: null,
    answerStrs: null,
    answerSteps: null,
    generation: RECORDED_GENERATION,
    recaptureCheck: null,
    rankLookahead: null,
    centreKind: null,
    timing: null,
    continuation: f.continuation,
    thinkingContinuation: null,
    thinkingPath: null,
    thinkSeed: null,
    thinkReadAt: null,
    thinkStopReason: null,
    thinkSkipped: false,
    // the recording has no sighting with thinking turned on
    thinkProtocol: null,
    thinkLeftOut: null,
    landmarks: f.landmarks,
    surveyMode: "monument",
    elapsedMs: f.elapsedMs,
    peakVramGb: null,
    isNudged: false,
    system: null,
    poleRanking: "activation",
    threadScore: "coupling",
    frame: null,
    lens: null,
    steeringApplied: null,
    steeringSent: null,
    layers,
    kinship: f.kinship,
    kinStats: f.kinStats,
    threads: f.threads.map((t) => ({
      src: { idx: t.src[0], neuron: t.src[1], face: t.src[2] >= 0 ? 1 : -1 },
      dst: { idx: t.dst[0], neuron: t.dst[1], face: t.dst[2] >= 0 ? 1 : -1 },
      c: t.c,
      dot: t.dot,
      directGate: null,
      directAct: null,
      score: t.c,
      scoreKind: "coupling" as const,
    })),
    threadCount: f.threadCount,
    activeEdits: [],
    raw: null,
  };
}

/* ------------------------------------------------- where it was read, what it chose */

/** 1st, 2nd, 3rd, 4th, … 11th, 12th, 13th, 21st … */
export function ordinal(n: number): string {
  const m100 = Math.abs(n) % 100;
  const m10 = Math.abs(n) % 10;
  const suffix = m100 >= 11 && m100 <= 13 ? "th" : m10 === 1 ? "st" : m10 === 2 ? "nd" : m10 === 3 ? "rd" : "th";
  return `${n}${suffix}`;
}

export type ReadRole = "question_end" | "template_end" | "begun_end" | "written" | "unknown";

/**
 * Where a sighting was read, in plain words, for every page to use instead of its own phrasing. For k = 0 the three
 * phrasings the pages always used (the end of your question, the end of the chat template, the end of the begun
 * answer); for k ≥ 1 the piece read is w_{k-1}, the k-th the model wrote itself: “, at “am”, the 3rd word-piece of its
 * own answer”.
 */
export function readWhere(s: Sighting | null | undefined): { piece: string | null; role: ReadRole; k: number; phrase: string } {
  if (!s) return { piece: null, role: "unknown", k: 0, phrase: "" };
  const k = s.tokenIndex;
  const piece = s.readAt;
  if (k >= 1) {
    return {
      piece,
      role: "written",
      k,
      phrase: `, at ${piece != null ? pieceWord(piece) : "a word-piece"}, the ${ordinal(k)} word-piece of its own answer`,
    };
  }
  const begun = s.answerBegins.trim();
  if (begun) return { piece, role: "begun_end", k, phrase: `, in its reply “${begun}”` };
  if (s.chatWrapped) return { piece, role: "template_end", k, phrase: ", at the end of the chat template" };
  return { piece, role: "question_end", k, phrase: ", at the end of your question" };
}

/**
 * Whether a sighting reads inside the model's own answer, in the class of positions where E3 measured the lens in
 * answers (E3_lens_validation_fidlib.py e3_chat_positions: every piece after the template's closing "<channel|>", to
 * the end of the answer): every word-piece it wrote (k ≥ 1, the piece read is w_{k-1}), and at k = 0 the last piece of
 * an answer begun for it inside a chat turn. Not the template's own end (a chat with no begun answer), nor the end of a
 * bare-text prompt, begun answer or not: no E3 class holds those. The pages gate the in-answer lens caution by it.
 */
export function readsInsideAnswer(s: Sighting | null | undefined): boolean {
  if (!s) return false;
  if (s.tokenIndex >= 1) return true;
  return s.tokenIndex === 0 && s.chatWrapped && readWhere(s).role === "begun_end";
}

/** The word-piece this sighting's position chose (w_k), or null (nothing written, the recorded sighting, an older server). */
export const chosenOf = (s: Sighting | null | undefined): ChosenPiece | null => s?.chosen ?? null;

/** The checkpoint's mark that ends the model's turn (106), and its mark that ends the text (1). */
const TURN_END = { id: 106, token: "<turn|>" } as const;
const TEXT_END = { id: 1, token: "<eos>" } as const;
export const isTurnEnd = (id: number | null | undefined, token?: string | null) => id === TURN_END.id || token === TURN_END.token;
export const isTextEnd = (id: number | null | undefined, token?: string | null) => id === TEXT_END.id || token === TEXT_END.token;

/**
 * A stop mark (a piece with endMark), in plain words, by its id or its text. Writing stops at any of the checkpoint's
 * stop marks (generation_config.eos_token_id: 1, 106, 50; backend engine_runtime end_of_turn_ids), but only <turn|>
 * ends its turn: <eos> ends the text, and any other mark (<|tool_response>, which the old nudge defaults made this model
 * write twelve times over: the Nudge page's example) is a control mark written where the answer should go on, so the
 * words name it rather than call it a clean ending.
 */
export function endMarkWords(id: number | null | undefined, token?: string | null): string {
  if (isTurnEnd(id, token)) return "it ends its turn";
  if (isTextEnd(id, token)) return "the end of the text (<eos>)";
  const named = token || (id != null ? `#${id}` : null);
  return named ? `a control mark, ${named}, where writing stops` : "a control mark, where writing stops";
}

/** “where it chose “a”” / “where it ended its turn” (<turn|> only) / “where it wrote … and stopped” for another stop
 * mark / null when no choice is known. */
export function chosenPhrase(s: Sighting | null | undefined): string | null {
  const c = chosenOf(s);
  if (!c) return null;
  if (!c.endMark) return `where it chose ${pieceWord(c.token)}`;
  if (isTurnEnd(c.id, c.token)) return "where it ended its turn";
  if (isTextEnd(c.id, c.token)) return "where it wrote the end of the text (<eos>) and stopped";
  return `where it wrote a control mark, ${c.token || `#${c.id}`}, and writing stopped`;
}

/**
 * Whether row `rank` of a layer's lens readout is the word-piece the position chose. Matched by token id; by the
 * string only when the server sent no ids for the readout. Never by the continuation's first word, which is the
 * wrong word for k ≥ 1 and can be a different piece with the same spelling.
 */
export function isChosenRow(s: Sighting | null | undefined, L: LayerReading, rank: number): boolean {
  const c = chosenOf(s);
  if (!c) return false;
  if (L.lensTopIds) return L.lensTopIds[rank] === c.id;
  const t = L.lensTop?.[rank];
  return t != null && t.token === c.token;
}

/** The pieces of the answer this sighting belongs to, with their stats; null when the server sent no answer. */
export function piecesOf(s: Sighting | null | undefined): AnswerPiece[] | null {
  if (!s || !s.answerIds || !s.answerStrs) return null;
  return piecesFrom(s.answerIds, s.answerStrs, s.answerSteps, s.generation?.stopReason === "end_mark");
}

/** A repeat penalty in plain words: a positive one holds repeats back, a negative one favours them; "" for none. */
function repeatWords(p: number | null | undefined): string {
  if (p == null || !Number.isFinite(p) || p === 0) return "";
  const by = String(Math.round(Math.abs(p) * 1000) / 1000);
  return p > 0 ? `holding back repeats by ${by}` : `favouring repeats by ${by}`;
}

/**
 * How an answer was written, in plain words, from the server's echo (never from the form's current settings). A
 * repeat penalty comes before the greedy argmax (interp_engine steer.py:654-659), so a penalised greedy answer is not
 * called its likeliest: the words match OneDrawLine's (lib/cautions.tsx, oneDrawParts).
 */
export function generationWords(g: GenerationEcho | null | undefined): string | null {
  if (!g) return null;
  if (g.source === "recorded") return "its likeliest word-piece each time (greedy), as the study recorded it";
  const repeats = repeatWords(g.presencePenalty);
  const held = repeats ? `, ${repeats}` : "";
  if (g.greedy) {
    return repeats
      ? `the likeliest word-piece each time after ${repeats} (greedy), so not always its own likeliest`
      : "its likeliest word-piece each time (greedy)";
  }
  const how = [
    `temperature ${g.temperature ?? "[not sent]"}`,
    g.topK != null ? `from its likeliest ${g.topK}` : "no top-k limit",
    g.topP != null ? `within ${g.topP} of its odds` : "no top-p limit",
  ];
  const seed = g.seed != null ? `, seed ${g.seed}${g.seedDrawn ? ", drawn by the server" : ""}` : "";
  return `drawn by chance (${how.join(", ")}${held}${seed})`;
}

/* ------------------------------------------------------------- loudness */

// The poles of a layer come in the order the server ranked them: by |a| (loudness), or by write size when
// `pole_ranking` is "write". The listed fifteen are the top fifteen of the layer's 21,504 by that measure.

/**
 * The first listed neuron in the sighting's own ranking: the loudest by |a|, or the biggest writer. Since the list is
 * the top fifteen by that measure, its rank among all 21,504 in the layer is exact. The server reads words for the
 * first five listed only (engine_runtime.py), so the lead always has them when they were read.
 */
export function leadOf(L: LayerReading): PoleReading | null {
  return L.poles[0] ?? null;
}

/**
 * The loudest by |a| among the fifteen listed only. When the sighting ranks by write size the listed fifteen are not
 * the fifteen loudest, so a page must then say "of the fifteen listed", never "in its layer"; and this neuron may lie
 * past the first five, whose words alone are read. A page that names the layer's leading neuron uses leadOf.
 */
export function loudestOf(L: LayerReading): PoleReading | null {
  return L.poles.length ? L.poles.reduce((a, b) => (Math.abs(b.act) > Math.abs(a.act) ? b : a)) : null;
}

/** 0 = loudest among the layer's listed poles, by |a|; null when the neuron is not listed */
export function loudnessRank(L: LayerReading, neuron: number): number | null {
  const p = L.poles.find((x) => x.neuron === neuron);
  if (!p) return null;
  return L.poles.filter((x) => Math.abs(x.act) > Math.abs(p.act)).length;
}

/** the largest |a| among the layer's listed poles (for strength bars), or null */
export function maxAct(L: LayerReading): number | null {
  return L.poles.length ? Math.max(...L.poles.map((p) => Math.abs(p.act))) : null;
}

/* -------------------------------------------------------- the ranking's words */

const RANK_ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
  "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth"];
/** the first five listed rows of a layer, in the words of a ranking by |a| (the read rows' labels) */
export const RANKW = ["the loudest", "second", "third", "fourth", "fifth"];
/** the same when the server ranked the poles by write size, |a| · ‖w_down ⊙ w_post‖: the list's order is the writers' */
export const RANKW_WRITE = ["the biggest writer", "second", "third", "fourth", "fifth"];

/**
 * One vocabulary for a sighting's ranking of its fifteen neurons, by |a| ("activation", as first built) or by write
 * size ("write"). `nth(rank)` takes a rank from 0 in that ranking ("loudest", "third loudest"; "biggest writer", "third
 * biggest writer"); `measure` is the score as a formula; `rows` labels the first five listed.
 */
export function rankingWords(r: PoleRanking) {
  const write = r === "write";
  const lead = write ? "biggest writer" : "loudest";
  return {
    fifteen: write ? "the fifteen that wrote most" : "the fifteen that fired hardest",
    lead: `the ${lead}`,
    nth: (rank: number) => (rank === 0 ? lead : `${RANK_ORDINALS[rank] ?? ordinal(rank + 1)} ${lead}`),
    did: write ? "wrote the most into the stream" : "fired hardest",
    measure: write ? "|a| · ‖w_down ⊙ w_post‖" : "|a|",
    head: write ? "the neuron that wrote most" : "its loudest neuron",
    notAmong: write ? "not among the fifteen that wrote most" : "not among the fifteen that fired hardest",
    rows: write ? RANKW_WRITE : RANKW,
  };
}

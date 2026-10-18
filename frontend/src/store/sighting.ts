/**
 * A "sighting" is one sweep of the instrument: the model's last word-piece followed up through all
 * sixty layers. It is built either from a live sweep (binary frame + JSON metadata from
 * /ws/theodolite) or from the recorded demonstration sweep of the study (E6), which is what the
 * approved design canvas shows.
 *
 * Every field that a source does not record is `null`. Screens must show such values as bracketed
 * placeholders (see <Missing/>), never as an invented number.
 */

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
  rank: number; // 0 = loudest
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
  /** the naked eye's top word-piece; null = not recorded */
  eyeTop: string | null;
  /** tracked word-piece ranks of 262,144, with their softcapped logits where recorded */
  ranks: Record<string, { lens: number | null; eye: number | null; lensLogit?: number | null; eyeLogit?: number | null }>;
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

export interface Sighting {
  /** a number unique to this sighting in this page's life, so results asked of it can be told from older ones */
  id: number;
  source: "live" | "fixture";
  /** one line saying where this sighting came from */
  provenance: string;
  question: string;
  answerBegins: string;
  /** the exact text sent to the server; null when it was not recorded (the fixture's chat-wrapped prompt) */
  textSent: string | null;
  chatWrapped: boolean;
  promptTokens: string[] | null;
  nTokens: number | null;
  readPos: number | null;
  readAt: string | null;
  firstIds: number[] | null;
  continuation: string;
  thinkingContinuation: string | null;
  thinkingPath: [number, number, number][] | null;
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
      eyeTop: typeof p.naive_top_token === "string" ? p.naive_top_token : null,
      ranks,
      poles,
      ffnExplained: numOrNull(p.poles_ffn_explained),
    });
  }
  const threadScore: ThreadScore = meta.thread_score === "direct" ? "direct" : "coupling";
  const threads = liveThreads(meta, bin, threadScore);
  const kinship = bin && bin.ckaMatrix.length === N_LAYERS * N_LAYERS ? Array.from(bin.ckaMatrix) : null;
  const tokensList: string[] | null = Array.isArray(meta.prompt_token_strs) ? meta.prompt_token_strs : null;
  const think = bin?.thinkTrajXyz;
  return {
    id: nextSightingId++,
    source: "live",
    provenance: "a live sighting from the running server",
    question: req.question,
    answerBegins: req.answerBegins,
    textSent: typeof meta.prompt === "string" ? meta.prompt : req.textSent,
    chatWrapped: meta.chat === true,
    promptTokens: tokensList,
    nTokens: tokensList ? tokensList.length : null,
    readPos: tokensList ? tokensList.length - 1 : null,
    readAt: tokensList && tokensList.length ? tokensList[tokensList.length - 1] : null,
    firstIds: Array.isArray(meta.prompt_token_ids) ? (meta.prompt_token_ids as number[]).slice(0, 4) : null,
    continuation: String(meta.generated_text ?? ""),
    thinkingContinuation: typeof meta.think_generated_text === "string" ? meta.think_generated_text : null,
    thinkingPath: think
      ? Array.from({ length: N_LAYERS }, (_, l) => [think[l * 3], think[l * 3 + 1], think[l * 3 + 2]] as [number, number, number])
      : null,
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
  return {
    id: str(o.id) ?? "",
    name: name ?? LENS_NAME_NOT_SENT,
    fittedOn: fittedOn ?? LENS_MODEL_NOT_SENT,
    appliedTo: str(o.applied_to),
    method: str(o.method) ?? "",
    fitData: str(o.fit_data) ?? "",
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
      eyeTop: null,
      ranks,
      poles,
      ffnExplained: null,
    });
  }
  return {
    id: nextSightingId++,
    source: "fixture",
    provenance: f.source,
    question: f.question,
    answerBegins: f.answerBegins,
    textSent: f.textSent,
    chatWrapped: true,
    promptTokens: null,
    nTokens: f.nTokens,
    readPos: f.readPos,
    readAt: f.readAt,
    firstIds: f.firstIds,
    continuation: f.continuation,
    thinkingContinuation: null,
    thinkingPath: null,
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

/* ------------------------------------------------------------- loudness */

/**
 * The poles of a layer come in the order the server ranked them: by |a| (loudness), or by write size when
 * `pole_ranking` is "write". Pages that speak of "the loudest" use these, which always go by |a|.
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

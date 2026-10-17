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
}

export interface ThreadReading {
  src: { idx: number; neuron: number; face: 1 | -1 };
  dst: { idx: number; neuron: number; face: 1 | -1 };
  c: number;
  dot: number | null;
}

export interface Sighting {
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
  layers: LayerReading[];
  /** 60 x 60 linear CKA, row-major; null = not recorded */
  kinship: number[] | null;
  kinStats: { meanOffDiag: number; adjacent: number; l11vs51: number } | null;
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

const numOrNull = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const toks = (list: unknown): TokenReading[] | null =>
  Array.isArray(list)
    ? list.map((t: Json) => ({ token: String(t.token ?? ""), prob: numOrNull(t.prob), logit: numOrNull(t.logit) }))
    : null;

export function sightingFromLive(
  meta: Json,
  bin: LiveBinary | null,
  req: { question: string; answerBegins: string; textSent: string },
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
      projFrac: null,
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
    });
  }
  const threads: ThreadReading[] = (Array.isArray(meta.filaments_meta) ? meta.filaments_meta : []).map((f: Json) => ({
    src: { idx: f.src_layer, neuron: f.src_neuron, face: f.src_polarity === "-" ? -1 : 1 },
    dst: { idx: f.dst_layer, neuron: f.dst_neuron, face: f.dst_polarity === "-" ? -1 : 1 },
    c: Number(f.coupling_score),
    dot: numOrNull(f.dot_product),
  }));
  const kinship = bin && bin.ckaMatrix.length === N_LAYERS * N_LAYERS ? Array.from(bin.ckaMatrix) : null;
  const tokensList: string[] | null = Array.isArray(meta.prompt_token_strs) ? meta.prompt_token_strs : null;
  const think = bin?.thinkTrajXyz;
  return {
    source: "live",
    provenance: "a live sighting from the running server",
    question: req.question,
    answerBegins: req.answerBegins,
    textSent: typeof meta.prompt === "string" ? meta.prompt : req.textSent,
    chatWrapped: false,
    promptTokens: tokensList,
    nTokens: tokensList ? tokensList.length : null,
    readPos: tokensList ? tokensList.length - 1 : null,
    readAt: tokensList && tokensList.length ? tokensList[tokensList.length - 1] : null,
    firstIds: null,
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
    layers,
    kinship,
    kinStats: kinshipStats(kinship),
    threads,
    threadCount: numOrNull(meta.n_filaments) ?? (bin ? bin.nFilaments : null),
    activeEdits: Array.isArray(meta.larql_status?.active_patches) ? meta.larql_status.active_patches : [],
    raw: meta,
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
    });
  }
  return {
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
    layers,
    kinship: f.kinship,
    kinStats: f.kinStats,
    threads: f.threads.map((t) => ({
      src: { idx: t.src[0], neuron: t.src[1], face: t.src[2] >= 0 ? 1 : -1 },
      dst: { idx: t.dst[0], neuron: t.dst[1], face: t.dst[2] >= 0 ? 1 : -1 },
      c: t.c,
      dot: t.dot,
    })),
    threadCount: f.threadCount,
    activeEdits: [],
    raw: null,
  };
}

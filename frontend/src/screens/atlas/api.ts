/**
 * The atlas's routes as the page reads them (backend/atlas.py, backend/scientist_chat.py; ARCHITECTURE.md §4.7-4.8).
 * Layers count from 0 in every field here, as the server sends them; the page shows layer + 1 (`pageLayer`).
 */

export class AtlasError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function detailOf(r: Response): Promise<string> {
  const text = await r.text();
  try {
    const d = JSON.parse(text)?.detail;
    if (typeof d === "string") return d;
    if (d) return JSON.stringify(d);
  } catch {
    // not JSON: say what came back
  }
  return text.slice(0, 200) || `HTTP ${r.status}`;
}

const qs = (q?: Record<string, unknown>) => {
  if (!q) return "";
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(q)) if (v !== undefined && v !== null && v !== "") p.set(k, String(v));
  const s = p.toString();
  return s ? `?${s}` : "";
};

export async function get<T>(path: string, q?: Record<string, unknown>, signal?: AbortSignal): Promise<T> {
  let r: Response;
  try {
    r = await fetch(`${path}${qs(q)}`, { signal });
  } catch (e) {
    if ((e as Error)?.name === "AbortError") throw e;
    throw new AtlasError("The server is not answering.", 0);
  }
  if (!r.ok) throw new AtlasError(await detailOf(r), r.status);
  return (await r.json()) as T;
}

export async function post<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  let r: Response;
  try {
    r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}), signal });
  } catch (e) {
    if ((e as Error)?.name === "AbortError") throw e;
    throw new AtlasError("The server is not answering.", 0);
  }
  if (!r.ok) throw new AtlasError(await detailOf(r), r.status);
  return (await r.json()) as T;
}

/** A binary reply (the web's int16 rows) with its headers. */
export async function getBinary(path: string, q?: Record<string, unknown>): Promise<{ data: ArrayBuffer; headers: Headers }> {
  const r = await fetch(`${path}${qs(q)}`);
  if (!r.ok) throw new AtlasError(await detailOf(r), r.status);
  return { data: await r.arrayBuffer(), headers: r.headers };
}

/** Lines of a streamed reply as they come: server-sent events ("data: {...}") or NDJSON. */
export async function* streamLines(r: Response, sse: boolean): AsyncGenerator<unknown> {
  const reader = r.body!.getReader();
  const dec = new TextDecoder();
  let buf = "";
  const sep = sse ? "\n\n" : "\n";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i: number;
    while ((i = buf.indexOf(sep)) >= 0) {
      const chunk = buf.slice(0, i);
      buf = buf.slice(i + sep.length);
      if (sse) {
        for (const line of chunk.split("\n")) if (line.startsWith("data: ")) yield JSON.parse(line.slice(6));
      } else if (chunk.trim()) yield JSON.parse(chunk);
    }
  }
}

export const pageLayer = (l: number) => l + 1;
export const cite = (l: number, n: number) => `L${l + 1} N${n}`;
export const citeHead = (l: number, h: number) => `L${l + 1} h${h}`;

/* ------------------------------------------------------------------------------------------------------ replies */

export interface LayerShape {
  kind: "sliding" | "full";
  n_heads: number;
  n_kv: number;
  head_dim: number;
  values_from_keys: boolean;
  window: number | null;
}
export interface IndexStatus {
  built: boolean;
  file?: string;
  refused?: string;
  built_at?: string;
  corpus?: string;
  method?: string;
  [k: string]: unknown;
}
export interface ScaleSource {
  kind: "index" | "study_texts";
  corpus: string;
  tokens: number;
}
export interface Arch {
  model: string;
  stand_in: boolean;
  n_layers: number;
  d_model: number;
  d_ff: number;
  vocab_size: number;
  layers: LayerShape[];
  layer_scalars: number[];
  indexes: Record<string, IndexStatus | string>;
  census: Record<string, string>;
  /** layers whose only census is not read, and why (a run taken before the controls were shaped as neurons' writes) */
  census_refused?: Record<string, string>;
  push: { modes: string[]; units: Record<string, string>; scale_source: ScaleSource | null };
  limits: { text: number; run: number; heads: number; push_new: number };
  label_gate: Record<string, unknown>;
  lens: Record<string, unknown> | null;
}

/** One face's words from the indexes: through the lens and by the naked eye (with the gate's check), the steer lens. */
export interface FaceWords {
  lens: string[] | null;
  eye: string[] | null;
  check: unknown;
  steer: string[] | null;
  stale?: string;
}
/** A weight link, read from rest (backend/atlas_ops.py LINK_MODEL): `gate` is the writer's face that opens the reader's
 * gate, `sign` the reader's face it then pushes toward; `w` its strength as it reaches the reader (Gemma's layer
 * scalars and the norms counted) over the neuron's strongest; `z` how unusual it is within its pair of layers. */
export interface Link {
  layer: number;
  n: number;
  z: number;
  w?: number;
  sign: number;
  gate?: number;
  face?: "+" | "-";
  words?: FaceWords;
  census?: string[];
  name?: string;
}
export interface Wiring {
  layer: number;
  n: number;
  source: "index" | "live";
  down: Link[];
  up: Link[];
  note: string;
}
export interface Hover {
  layer: number;
  n: number;
  faces: Record<"+" | "-", FaceWords>;
  links: { down: Link[]; up: Link[] } | null;
  name: string;
  ready: boolean;
}
export interface ActRange {
  act_max: number;
  act_min: number;
  over_tokens: number;
  source: string;
  regimes?: Record<"pp" | "nn" | "pn" | "np", number>;
}
export interface SteerLens {
  source: "index" | "live";
  up: string[];
  down: string[];
  kl_up: number;
  kl_down: number;
  dose: number;
}
export interface LensFace {
  lens: string[];
  eye: string[];
  check: unknown;
}
export interface NeuronLens {
  layer: number;
  n: number;
  faces: Record<"+" | "-", LensFace>;
  gate_trigger: string[];
  steer: SteerLens | null;
  /** set when the steer-lens index was read before the vault edit in force on this neuron (it is not shown) */
  steer_stale?: string;
  act_range: ActRange | null;
  edited: boolean;
  /** the most the neuron can ever write to the stream leaving its layer; `rel` is that over the push's unit, so an add
   * push of dose d is d / rel times it */
  ceiling?: { write: number; p50: number | null; rel: number | null; source: string | null } | null;
  lens: Record<string, unknown> | null;
}
export interface HeadRole {
  layer: number;
  head: number;
  heads: number[];
  z: number;
}
export interface NeuronHeads {
  layer: number;
  n: number;
  roles: Record<"q" | "k" | "v" | "o", HeadRole[]>;
  z_by_layer: Record<string, (number | null)[]>;
}
export interface PooledWord {
  tok: string;
  n: number;
  w: number;
}
/** The web's words, the cone followed from rest: `down["+"]`, what the neurons it drives write when it fires on its +
 * face; `up["+"]`, what the neurons that push it toward its + face write (and so for "-"). */
export interface WebSummary {
  src: "steer" | "lens" | null;
  ready: boolean;
  missing?: "wiring" | "lens" | "steer_lens";
  /** set when the neuron is edited in the vault: the index's links predate the edit */
  stale?: string;
  links?: number;
  down?: Record<"+" | "-", { neurons: number; words: PooledWord[] }>;
  up?: Record<"+" | "-", { neurons: number; words: PooledWord[] }>;
  gated?: { down: number; up: number };
}
export interface SearchRow {
  layer: number;
  n: number;
  sign: "+" | "-";
  rank: number;
  matched: number;
  src: "lens" | "eye" | "steer";
  words: FaceWords;
  census?: string[] | null;
  name: string;
}
export interface SearchReply {
  ready: boolean;
  /** matches through the steer lens, or through labels the gate shows */
  results: SearchRow[];
  /** matches through labels the gate holds back: for the apparatus, never the map */
  held?: SearchRow[];
  held_back?: number;
  tokens: number;
  total: number;
}
export interface CensusUnitRow {
  kind: "neuron" | "random";
  n: number;
  z: number;
  null_pct: number | null;
  q?: number | null;
  words: string[];
  name: string;
}
/** What every census reply says of its run (backend/atlas.py census_card). */
export interface CensusCard {
  run: string;
  design: string | null;
  doses_from: string | null;
  p50_source: string | null;
  read_gain: { neurons: number; controls: number; ratio: number | null } | null;
  null: { neurons: number; controls: number; above_all: number; above_all_by_chance: number | null; q05: number };
}
export interface Rollout {
  prompt: string;
  dose: number;
  rel_push: number;
  text: string;
  finish: "stop" | "length";
  n_gen: number;
}
export interface CensusStats {
  words: [string, number, number][];
  z: number;
  mass: number;
  null_pct: number | null;
  p?: number | null;
  q?: number | null;
  per_prompt: Record<string, string[]>;
  dose: Record<string, { rel_push: number; distinct4: number; capped: number; logprob: number }>;
  shift?: number | null;
  shift_up?: [string, number][];
  shift_down?: [string, number][];
}
export interface CensusUnit {
  layer: number;
  n: number;
  kind: string;
  name: string;
  run?: string;
  prompts?: Record<string, string>;
  design?: string;
  card?: CensusCard;
  rollouts?: Rollout[];
  baselines?: Rollout[];
  stats?: CensusStats | null;
  rank?: number | null;
  n_ranked?: number;
}
export interface Spread {
  x?: number[];
  y?: number[];
  row?: number[];
  kind?: string[];
  neuron?: number[];
  prompt?: string[];
  dose?: number[];
  embedder?: string;
  layout?: string;
}
export interface RunReply {
  ready: boolean;
  tokens: string[];
  pos: number[];
  scale_source: ScaleSource;
  n_pieces: number;
}
export interface RunRow {
  layer: number;
  n: number;
  face: "+" | "-";
  score: number;
  act: number;
  token: string;
  pos: number;
  words: FaceWords;
}
export interface RunView {
  results: RunRow[];
  total: number;
  map?: string;
  shape?: [number, number];
}
export interface HeadLinkNeuron {
  layer: number;
  n: number;
  z: number;
  words?: FaceWords;
}
export interface HeadInfo {
  layer: number;
  head: number;
  kind: "sliding" | "full";
  kv_head: number;
  values_from_keys: boolean;
  q: HeadLinkNeuron[];
  k: HeadLinkNeuron[];
  v: HeadLinkNeuron[];
  o: HeadLinkNeuron[];
  q_pooled: { src: string | null; tokens: PooledWord[]; gated: number };
  k_pooled: { src: string | null; tokens: PooledWord[]; gated: number };
  v_pooled: { src: string | null; tokens: PooledWord[]; gated: number };
  o_pooled: { src: string | null; tokens: PooledWord[]; gated: number };
}
export interface HeadProfile {
  dist: number[];
  dist_names: string[];
  src: [string, number][];
  pairs: { w: number; text: string; q: number; k: number; qt: string; kt: string }[];
}
export interface Note {
  name: string;
  note: string;
  /** who wrote it last; `name_by` and `note_by` who wrote its name and its note */
  by: "reader" | "scientist";
  name_by?: "reader" | "scientist";
  note_by?: "reader" | "scientist";
  at: string;
}
export interface Notes {
  version: number;
  neurons: Record<string, Note>;
  heads: Record<string, Note>;
  file: string;
  warning: string | null;
}
export interface ReadsReply {
  units: { text: string; act: number; pieces: number[]; pos: number }[];
  act_range: ActRange | null;
}
/** One stored passage (GET /neuron/passages): the event's activation, its share of the face's extreme, GeGLU's regime
 * there and its gate and up values; its window and position; the display units of `span` [s, e) of the window, the
 * peak marked; why each side stops (`cut`: "start", the window's own; "document", an end mark or a start mark inside
 * the window; "turn", "end", or null where the count ran out); and
 * `ids`, the window up to the span's end, which POST /reads takes to read it again. */
export interface Passage {
  kind: "top" | "sample";
  act: number;
  rel: number | null;
  regime: "pp" | "nn" | "pn" | "np";
  g: number;
  u: number;
  window: number;
  at: number;
  units: [string, boolean][];
  span: [number, number];
  next: string | null;
  cut: [string | null, string | null];
  ids: number[];
}
export interface PassageWord {
  tok: string;
  id: number;
  n: number;
  lift: number | null;
}
/** One face's stored passages; `reads` and `follows` are counted over the top `k` with their lift over the corpus's
 * base rate: leads (`words_note`), not labels the gate has checked. Not `ready` without the passages index. */
export interface Passages {
  layer: number;
  n: number;
  face: "+" | "-";
  ready: boolean;
  note?: string;
  corpus?: string;
  tokens?: number;
  windows?: number;
  built_at?: string;
  k?: number;
  K?: number;
  sampled?: number;
  sample_threshold?: string | null;
  extreme?: number | null;
  rel_to?: string | null;
  passages?: Passage[];
  reads?: PassageWord[];
  follows?: PassageWord[];
  words_note?: string;
  stale?: string;
}
/** A passage's activation at its peak as POST /reads gives it now, from the reads' units (each unit's pieces end at `pos`). */
export function actAt(r: ReadsReply, at: number): number | null {
  for (const u of r.units) {
    const first = u.pos - u.pieces.length + 1;
    if (at >= first && at <= u.pos) return u.pieces[at - first];
  }
  return null;
}

/* -------------------------------------------------------------------------------------------------- the scientist */

export interface EndpointCard {
  name: string;
  label: string;
  base_url: string;
  this_machine: boolean;
  key: string;
  model: string | null;
  stream: boolean;
}
export interface Backends {
  claude: { available: boolean; model: string | null };
  codex: { available: boolean };
  openai: EndpointCard[];
  warning: string | null;
  /** what each backend can do beyond the atlas's tools: "folder" (the default) or "home" (a shell over the home folder) */
  reach?: { mode: "folder" | "home"; env: string; claude: string; codex: string; openai: string };
}
export interface ChatEvent {
  t: "state" | "sent" | "text" | "thinking" | "tool" | "tool_result" | "ui" | "error" | "done" | "ping" | "user";
  [k: string]: unknown;
}
export interface PastChat {
  chat: string;
  agent: "claude" | "codex" | "openai";
  t: number;
  title: string | null;
  turns: number;
  profile?: string;
  model?: string;
}

export const noteKey = (layer: number, n: number) => `${layer}:${n}`;

/** A base64 run of little-endian int16 or float16 values. */
export function b64ToU16(b64: string): Uint16Array {
  const bin = atob(b64);
  const out = new Uint16Array(bin.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
  return out;
}

export function halfToFloat(h: number): number {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  return e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
}

import { create } from "zustand";
import {
  type ActiveEdit,
  type AnswerPiece,
  type FixtureFile,
  type GenerationEcho,
  type LensInfo,
  type LiveBinary,
  type PoleRanking,
  type Sighting,
  type ThreadScore,
  VOCAB,
  generationOf,
  lensOf,
  parseBinaryFrame,
  piecesFromAnswer,
  piecesOf,
  sightingFromFixture,
  sightingFromLive,
} from "./sighting";

export * from "./sighting";

/**
 * The instrument's state, wired to the backend:
 *   WebSocket /ws/theodolite  actions sweep | cancel_sweep | larql | inspect_pole | trace_readers
 *   HTTP      /api/health, /api/larql, /api/scientist/{inspect_pole,trace_readers,test_hypothesis}, /api/saved_prompts
 * When no server answers, the recorded demonstration sighting (E6) is shown instead, and every page
 * says so.
 *
 * One sweep request writes an answer and streams a (binary, JSON) pair for each word-piece it sights, all tagged with
 * the request_id the page sent: theodolite_progress while it writes, theodolite_written once the answer is written,
 * the pairs in sight order, then theodolite_sweep_end. The pairs of one request gather into an AnswerSet; the page
 * shows one of them, the word-piece the reader chose. An older server sends one pair per request and no request_id;
 * the requests are then answered in the order they were sent, as before.
 */

/* ------------------------------------------------------------ how it writes */

/** How the model writes its answer before it is sighted (the sweep's max_new_tokens, sampler and sight_every_token). */
export interface Writing {
  /** "greedy": its likeliest word-piece every time, as the study wrote; "sampled": drawn by chance */
  mode: "greedy" | "sampled";
  temperature: number;
  /** from its likeliest K word-pieces only; 0 = no limit */
  topK: number;
  /** within this share of its odds only; 1 = no limit */
  topP: number;
  /** taken off the score of every word-piece it has already written (not the prompt's); greedy too */
  presencePenalty: number;
  /** null: the server draws one and sends it back */
  seed: number | null;
  /** word-pieces it may write, 1-256; it stops sooner when it ends its turn */
  length: number;
  /** a sighting of each word-piece it writes (at most SIGHT_CAP), not only of the first */
  sightEach: boolean;
}

/** The checkpoint's own recommendation (generation_config.json: do_sample true, temperature 1.0, top_k 64, top_p 0.95).
 * A sampler field sent blank would take these values unseen, so the page always sends every one. */
export const MODEL_SAMPLING = { temperature: 1, topK: 64, topP: 0.95 } as const;
/** The study's writing: its likeliest word-piece each time, six of them; each word-piece sighted. */
export const STUDY_WRITING: Writing = {
  mode: "greedy",
  temperature: 1,
  topK: 64,
  topP: 0.95,
  presencePenalty: 0,
  seed: null,
  length: 6,
  sightEach: true,
};
/** The most word-pieces one request sights (the server's max_sighted bound). */
export const SIGHT_CAP = 64;
/** The most word-pieces one request writes (the server's max_new_tokens bound). */
export const MAX_WRITE = 256;
/** Timed on this server, in seconds: nothing written 1.4 (one forward pass, one position analysed); 6 written 5.9;
 * 24 written 19.2. So about 0.75 s a word-piece written, and about 0.7 s for each further position sighted. */
export const PACE = { base: 1.4, write: 0.75, sight: 0.7 } as const;

/** The largest seed the page sends, the server's own bound (2^53 - 1): a browser's numbers hold it exactly. */
const MAX_SEED = Number.MAX_SAFE_INTEGER;

/**
 * About how long a sighting takes with these settings, at most (it stops sooner when the model ends its turn):
 * the first sighting comes once the answer is written (and the thinking answer, which the first pair carries); the
 * others follow one position apart.
 */
export function paceOf(w: Writing, think: boolean): { sighted: number; first: number; all: number; thinking: number } {
  const sighted = w.sightEach ? Math.max(1, Math.min(w.length, SIGHT_CAP)) : 1;
  // the thinking answer is written as long as the main one, at least 10 and at most 64 word-pieces, and read at its start
  const thinking = think ? PACE.base + PACE.write * Math.max(Math.min(w.length, 64), 10) : 0;
  const first = PACE.base + PACE.write * w.length + thinking;
  return { sighted, first, all: first + PACE.sight * (sighted - 1), thinking };
}

/** A duration in plain words: “4 s”, “35 s”, “2½ minutes”. */
export function about(sec: number): string {
  if (sec < 10) return `${Math.max(1, Math.round(sec))} s`;
  if (sec < 90) return `${Math.round(sec / 5) * 5} s`;
  const m = Math.round((sec / 60) * 2) / 2;
  return `${String(Math.floor(m))}${m % 1 ? "½" : ""} minutes`;
}

/** The settings kept in range: what a field holds while typed never reaches the server out of its bounds. */
function fitWriting(w: Writing): Writing {
  const fit = (v: number, lo: number, hi: number, dflt: number) => (Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : dflt);
  return {
    mode: w.mode === "sampled" ? "sampled" : "greedy",
    temperature: fit(w.temperature, 0, 2, MODEL_SAMPLING.temperature),
    topK: Math.round(fit(w.topK, 0, VOCAB, 0)),
    topP: fit(w.topP, 0.01, 1, 1),
    presencePenalty: fit(w.presencePenalty, -2, 2, 0),
    seed: w.seed == null || !Number.isFinite(w.seed) ? null : Math.round(fit(w.seed, 0, MAX_SEED, 0)),
    length: Math.round(fit(w.length, 1, MAX_WRITE, STUDY_WRITING.length)),
    sightEach: w.sightEach === true,
  };
}

/* ------------------------------------------------------------ answer sets */

/** Where a request stands: sent, writing its answer, sighting its word-pieces, or finished. */
export interface SweepProgress {
  phase: "sending" | "writing" | "sighting" | "done";
  done: number;
  of: number;
}

/** The sightings of one request's answer, gathered as its pairs arrive. */
export interface AnswerSet {
  requestId: string;
  question: string;
  answerBegins: string;
  chat: boolean;
  /** the system turn sent ("" for none, or with the chat template off) */
  system: string;
  /** exactly what was sent (for retakes and export) */
  payload: Record<string, unknown>;
  /** the answer's word-pieces, from theodolite_written (or the first pair); null when the server sent none */
  pieces: AnswerPiece[] | null;
  /** the token indices it sights, in the order the server sights them */
  order: number[];
  nToSight: number;
  /** by token index, filled as pairs arrive */
  sightings: Record<number, Sighting>;
  /** the token index the reader chose */
  wanted: number;
  progress: SweepProgress;
  done: boolean;
  cancelled: boolean;
  error: string | null;
  nudged: boolean;
  /** how the server resolved and applied the writing of this request (source "given" for a re-reading) */
  generation: GenerationEcho | null;
  /** for a re-reading of an answer already written (an option that changes only the reading), the set it re-reads */
  retakeOf: string | null;
  /** how these word-pieces were written: this set's own generation, or, for a re-reading, that of the set it re-reads */
  wrote: GenerationEcho | null;
  /** the sighting with thinking turned on, which the server sends with the first pair only: read where its answer begins */
  thinking: { continuation: string; seed: number | null; readAt: string | null; path: [number, number, number][] | null } | null;
  /** "stop here" was asked, and the server has not stopped yet */
  stopAsked: boolean;
  /** the server tagged this request's frames (it can stream, and stop between word-pieces); false for an older server */
  streamed: boolean;
}

/** What `error` holds on a set whose connection closed before every word-piece came back. */
export const CONNECTION_CLOSED = "The connection closed before every word-piece was sighted.";

/* ------------------------------------------------------------ saved prompts */

export interface SavedQuestion {
  id: string;
  name: string;
  question: string;
  answerBegins: string;
  chat: boolean;
  savedAt: string;
}
export interface SavedSystem {
  id: string;
  name: string;
  text: string;
  savedAt: string;
}
export type SavedKind = "question" | "system";
export interface SavedLists {
  questions: SavedQuestion[];
  systems: SavedSystem[];
  /** "idle": never asked; "reading": asked; "ok": the last read answered; "failed": it did not (the lists keep the last read) */
  status: "idle" | "reading" | "ok" | "failed";
  error: string | null;
  /** the file the server keeps them in */
  file: string | null;
  /** what the server said once about its file (a corrupt one moved aside) */
  warning: string | null;
}

export interface SteeringConfig {
  enabled: boolean;
  target_concept: string;
  contrast_concept: string;
  /** repo indices 0-59 (layer number - 1) */
  layers: number[];
  /** absolute length the server adds (AddSpec normalize=True) when `strength` is null */
  scale: number;
  pole_layer: number | null;
  pole_neuron: number | null;
  pole_polarity: number;
  /** "fnw": u = fnw ⊙ ΔW_U, no inverse, no clamp (the V7 fix); "legacy": the original 1/fnw.clamp(0.1) recipe */
  direction: "fnw" | "legacy";
  /** zero the quiet dims of the vector added: the 181 persistent ones and each layer's quietest tenth */
  mask_quiet: boolean;
  /** leave the start mark (position 0) unsteered */
  spare_bos: boolean;
  /** push this share of ‖h‖ at every position and layer, set by the server; null = the absolute `scale` */
  strength: number | null;
}

/**
 * Safe nudge defaults (the nudge page's own default, ×0.01 of the stream). With the server's share-of-‖h‖
 * scaling (`strength`), 0.01 is sent as is. `SAFE_NUDGE_SCALE` is the same nudge as one absolute length, for
 * when that scaling is turned off: 0.01 × ‖h‖ at the weakest stream among the default layers, layer 54, whose
 * median non-BOS ‖h‖ is 88.3 (V7_tool_bugs_02_steer_p0.json). The tool audit puts a nudge at or below one
 * hundredth of ‖h‖ (V7 notes, bug 1). The old screen default was 280, i.e. 1.0-3.2 × ‖h‖, which collapses the model.
 */
export const SAFE_NUDGE_SCALE = 0.883;
export const SAFE_NUDGE_STRENGTH = 0.01;
export const DEFAULT_NUDGE_LAYERS = [29, 35, 41, 47, 53];

/** How the survey frame is built (SweepRequest frame_* fields; the defaults are the frame as first built). */
export interface FrameOptions {
  squaring: "gram_schmidt" | "lowdin";
  perLayer: boolean;
  fnw: boolean;
  /** centre on the kept matched control (sent as centre_prompts: [its exact text]) */
  centre: boolean;
}
export const FRAME_AS_BUILT: FrameOptions = { squaring: "gram_schmidt", perLayer: false, fnw: false, centre: false };

export interface Health {
  status: string;
  model_id: string;
  d_model: number;
  n_layers: number;
  d_ff: number;
  vocab_size: number;
  global_layers_1based: number[];
  vram_allocated_gb: number;
  vram_peak_gb: number;
  larql_cli: Record<string, string>;
  /** the lens the server reads through (its `lens`, parsed); null when it does not name one */
  lens: LensInfo | null;
}

export interface LarqlEntry {
  id: number;
  query: string;
  ok: boolean;
  data: Record<string, unknown> | null;
  error: string | null;
  at: number;
  via: "ws" | "http";
  /** why it failed: the vault answered and refused the statement, or the request never got an answer */
  failure: "refused" | "transport" | null;
}

export type Conn = "idle" | "connecting" | "open" | "closed";

type Keyed<T> = Record<string, T>;
export interface Pending<T> {
  pending: boolean;
  data: T | null;
  error: string | null;
  /** the sighting the result was asked for (traces depend on the question), when it depends on one */
  sightingId?: number | null;
  /** what was sent, so a page can say how the answer was read and how to ask it again by hand */
  sent?: Record<string, unknown>;
}

export const poleKey = (idx: number, neuron: number) => `${idx}:${neuron}`;

export interface TheodoliteState {
  conn: Conn;
  health: Health | null;
  /** the sighting shown: always answer.sightings[answer.wanted] once it has arrived (or the fixture, or a legacy reply) */
  sighting: Sighting | null;
  /** the token-0 sighting of the last set taken without a nudge, kept so the nudge page can compare */
  plainSighting: Sighting | null;
  /** a matched control kept by hand, to read the current sighting against (REPORT §3.5): the token-0 sighting of its set */
  control: Sighting | null;
  /** true until every request sent has ended (theodolite_sweep_end, or the socket closed) */
  isSweeping: boolean;
  sweepError: string | null;
  /** the set whose sightings are shown */
  answer: AnswerSet | null;
  /** the set still being taken (the oldest request in flight), its progress in `incoming.progress`; it becomes `answer`
   * when its first wanted sighting arrives, and stays here until it ends */
  incoming: AnswerSet | null;

  // the question
  question: string;
  answerBegins: string;
  /** wrap the question in the chat template (thinking off), with the begun answer after the model-turn header */
  chat: boolean;
  /** an optional system turn before the question (chat only) */
  system: string;
  compareThink: boolean;
  /** how the next sighting writes its answer */
  writing: Writing;
  /** the length the next sighting writes (writing.length), kept for the pages that read it; never a sighting's own */
  maxNewTokens: number;
  landmarks: [string, string, string];
  surveyMode: "monument" | "pole_motif";
  lockedPole: [number, number, number] | null;
  steering: SteeringConfig;
  frame: FrameOptions;
  poleRanking: PoleRanking;
  threadScore: ThreadScore;

  selectedLayer: number; // repo index

  // vault
  larqlQuery: string;
  larqlLog: LarqlEntry[];
  larqlPending: boolean;
  activeEdits: ActiveEdit[];

  // the circuit scientist
  inspections: Keyed<Pending<Record<string, any>>>; // eslint-disable-line @typescript-eslint/no-explicit-any
  readers: Keyed<Pending<Record<string, any>>>; // eslint-disable-line @typescript-eslint/no-explicit-any
  hypotheses: Keyed<Pending<Record<string, any>>>; // eslint-disable-line @typescript-eslint/no-explicit-any

  // saved questions and system prompts, kept by the server only when asked
  saved: SavedLists;

  // actions
  setWriting: (w: Partial<Writing>) => void;
  /** choose the word-piece shown: answer.sightings[k] as soon as it has arrived */
  pickToken: (k: number) => void;
  /** ask the server to stop the request in flight between word-pieces (what arrived is kept) */
  cancelSweep: () => void;
  /** GET /api/saved_prompts; on socket open and on first use */
  fetchSaved: () => Promise<void>;
  /** keep what the boxes hold now under `name` (replacing `replaceId` when given); false when the server did not keep it */
  keepSaved: (kind: SavedKind, name: string, replaceId?: string | null) => Promise<boolean>;
  forgetSaved: (kind: SavedKind, id: string) => Promise<boolean>;
  /** undo a forget: keep the entry again, with its own fields */
  restoreSaved: (kind: SavedKind, entry: SavedQuestion | SavedSystem) => Promise<boolean>;
  /** fill the boxes from an entry (a question also restores its chat setting); never sends anything */
  useSaved: (kind: SavedKind, entry: SavedQuestion | SavedSystem) => void;
  setQuestion: (q: string) => void;
  setAnswerBegins: (a: string) => void;
  setChat: (v: boolean) => void;
  setSystem: (v: string) => void;
  setFrame: (f: Partial<FrameOptions>) => void;
  setPoleRanking: (r: PoleRanking) => void;
  setThreadScore: (t: ThreadScore) => void;
  setCompareThink: (v: boolean) => void;
  setLandmarks: (m: [string, string, string]) => void;
  setSurveyMode: (mode: "monument" | "pole_motif", lockedPole?: [number, number, number] | null) => void;
  setSteering: (s: Partial<SteeringConfig>) => void;
  setSelectedLayer: (idx: number) => void;
  setLarqlQuery: (q: string) => void;
  setControl: (s: Sighting | null) => void;

  connect: () => void;
  loadFixture: () => Promise<void>;
  fetchHealth: () => Promise<void>;
  /** take a sighting; the options override the page's settings for this one request only */
  takeSighting: (opts?: TakeOptions) => boolean;
  executeLarql: (query?: string) => void;
  inspectPole: (idx: number, neuron: number, face: number) => void;
  traceReaders: (idx: number, neuron: number, face: number, opts?: TraceOptions) => void;
  /** run test_motif_hypothesis; by default each sentence goes in as bare text and is read at its own last
   * word-piece (with `chat`, the server reads the end of the template instead) */
  testHypothesis: (idx: number, neuron: number, pos: string[], neg: string[], opts?: { chat?: boolean }) => Promise<void>;
}

export interface TakeOptions {
  question?: string;
  answerBegins?: string;
  steering?: Partial<SteeringConfig>;
  chat?: boolean;
  system?: string;
  /** how this one answer is written (the rest from the page's settings) */
  writing?: Partial<Writing>;
  sightEach?: boolean;
  /** an answer already written, by its ids: nothing is written, these are read (answer_token_ids) */
  answerIds?: number[];
  /** the token indices to sight, in this order (sight_tokens) */
  sightOrder?: number[];
  /** keep showing the current sighting until this token index of the new answer arrives, and show that one */
  keepWanted?: number;
  /** centre each later word-piece on the control reading this very answer (default, when centring on a control) */
  centreAnswer?: boolean;
}

export interface TraceOptions {
  /** max_layer_hop: how many later layers to look through (1-6) */
  hop?: number;
  /** top_k: how many readers to list */
  topK?: number;
  /** score the readers by C or by the direct-path estimate */
  score?: ThreadScore;
}

/** The bare text sent as the prompt when the chat template is off: the question and the begun answer, one
 * line each (the server's own composition, `compose_prompt(chat=False)`, is the same). */
export function composeText(question: string, answerBegins: string) {
  const q = question.trim();
  const a = answerBegins.trim();
  return a ? `${q}\n${a}` : q;
}

/** The fields of a sweep request that say what text the model reads. With the chat template on, the server
 * wraps the question (and the optional system turn) and returns the exact text it read as `prompt`. */
export function promptFields(question: string, answerBegins: string, chat: boolean, system: string) {
  return chat
    ? { prompt: question.trim(), chat: true, system: system.trim() || null, assistant_prefix: answerBegins.trim() }
    : { prompt: composeText(question, answerBegins), chat: false };
}

/** A short label for the text a request will send, before the server has echoed it. */
export function describeText(question: string, answerBegins: string, chat: boolean) {
  return chat ? `[chat] ${question.trim()}${answerBegins.trim() ? ` → ${answerBegins.trim()}` : ""}` : composeText(question, answerBegins);
}

let ws: WebSocket | null = null;
let pendingBinary: LiveBinary | null = null;
/** LQL statements sent over the socket and not yet answered, oldest first; silent ones only resync the ledger */
const wsQueries: { query: string; silent: boolean }[] = [];

/** A sweep request sent and not yet ended. */
interface PendingReq {
  id: string;
  question: string;
  answerBegins: string;
  textSent: string;
  nudged: boolean;
  chat: boolean;
  /** the sighting taken by itself on (re)connect */
  auto: boolean;
  steering: Record<string, unknown>;
  /** its set has been shown (made `answer`) */
  promoted: boolean;
  /** a re-reading: the current sighting stays shown until this token index arrives */
  keepWanted: number | null;
  /** "sight this one too": the set the one new sighting joins, and its token index */
  mergeInto: { setId: string; k: number } | null;
}
/** sweep requests in flight, by request_id */
const pending = new Map<string, PendingReq>();
/** their ids, oldest first: the server answers one socket's sweeps in order (an older server's replies are matched so) */
const sentOrder: string[] = [];
/** the set of each request in flight, by request_id (the state's `answer` and `incoming` are copies of these) */
const liveSets = new Map<string, AnswerSet>();
let requestCount = 0;
const newRequestId = () => `r${Date.now().toString(36)}-${++requestCount}`;

let reconnectTimer: number | null = null;
let reconnectDelay = 4000;
/** set when the sighting taken automatically on (re)connect failed: do not resend it on every reconnect */
let autoRetakeBlocked = false;
let larqlId = 0;
let fixtureCache: Sighting | null = null;

const wsOpen = () => ws !== null && ws.readyState === WebSocket.OPEN;

/** An HTTP failure: `refused` when the server answered with its own reason (4xx with a detail), else transport. */
class RequestError extends Error {
  constructor(
    message: string,
    readonly refused: boolean,
  ) {
    super(message);
  }
}

async function requestJson(method: "GET" | "POST" | "DELETE", path: string, body?: unknown) {
  let r: Response;
  try {
    r = await fetch(
      path,
      body === undefined
        ? { method }
        : { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) },
    );
  } catch (e) {
    throw new RequestError(String((e as Error)?.message ?? e), false);
  }
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const detail = typeof data?.detail === "string" ? data.detail : null;
    throw new RequestError(detail ?? `HTTP ${r.status}`, detail !== null && r.status >= 400 && r.status < 500);
  }
  return data;
}

const postJson = (path: string, body: unknown) => requestJson("POST", path, body);

/** The server's saved lists (GET, POST and DELETE /api/saved_prompts all answer with the whole of them). */
function savedListsOf(d: unknown): Pick<SavedLists, "questions" | "systems" | "file" | "warning"> {
  const o = (d && typeof d === "object" ? d : {}) as Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
  const rows = (v: unknown) => (Array.isArray(v) ? v.filter((e) => e && typeof e === "object" && typeof e.id === "string") : []);
  return {
    questions: rows(o.questions).map((e) => ({
      id: e.id,
      name: String(e.name ?? ""),
      question: String(e.question ?? ""),
      answerBegins: String(e.answer_begins ?? ""),
      chat: e.chat === true,
      savedAt: String(e.saved_at ?? ""),
    })),
    systems: rows(o.systems).map((e) => ({ id: e.id, name: String(e.name ?? ""), text: String(e.text ?? ""), savedAt: String(e.saved_at ?? "") })),
    file: typeof o.file === "string" ? o.file : null,
    warning: typeof o.warning === "string" ? o.warning : null,
  };
}

/** The body that keeps an entry: the exact text, as written (only the name is trimmed, by the server). */
function savedBody(kind: SavedKind, name: string, from: { question: string; answerBegins: string; chat: boolean } | { text: string }) {
  return "text" in from
    ? { kind, name, text: from.text }
    : { kind, name, question: from.question, answer_begins: from.answerBegins, chat: from.chat };
}

/** A write to the saved lists (POST or DELETE): the server answers with the whole lists, which replace ours. */
async function writeSaved(call: () => Promise<unknown>): Promise<boolean> {
  try {
    const lists = savedListsOf(await call());
    useTheodoliteStore.setState((s) => ({
      saved: { ...s.saved, ...lists, warning: lists.warning ?? s.saved.warning, status: "ok", error: null },
    }));
    return true;
  } catch (e) {
    useTheodoliteStore.setState((s) => ({ saved: { ...s.saved, error: String((e as Error)?.message ?? e) } }));
    return false;
  }
}

const CLOSED_EARLY = "The connection closed before the instrument answered.";

export const useTheodoliteStore = create<TheodoliteState>((set, get) => ({
  conn: "idle",
  health: null,
  sighting: null,
  plainSighting: null,
  control: null,
  isSweeping: false,
  sweepError: null,
  answer: null,
  incoming: null,

  question: "Tell me about yourself.",
  answerBegins: "Hi! I",
  chat: true,
  system: "",
  compareThink: false,
  writing: STUDY_WRITING,
  maxNewTokens: STUDY_WRITING.length,
  landmarks: ["she", "he", "it"],
  surveyMode: "monument",
  lockedPole: null,
  steering: {
    enabled: false,
    target_concept: "six",
    contrast_concept: "eight",
    layers: DEFAULT_NUDGE_LAYERS,
    scale: SAFE_NUDGE_SCALE,
    pole_layer: null,
    pole_neuron: null,
    pole_polarity: 1.0,
    direction: "fnw",
    mask_quiet: true,
    spare_bos: true,
    strength: SAFE_NUDGE_STRENGTH,
  },
  frame: FRAME_AS_BUILT,
  poleRanking: "activation",
  // the direct-path estimate (ρ ≈ 0.87 with measured effects) rather than C (≈ 0.41), whose largest scores sit at
  // layer 1 where the real effects are tiny (REPORT §3.4, bug 4); the server's own default stays C for old callers
  threadScore: "direct",

  selectedLayer: 35, // layer 36

  larqlQuery: "SELECT * FROM POLES NEAREST 'she' WHERE layer = 50 LIMIT 10",
  larqlLog: [],
  larqlPending: false,
  activeEdits: [],

  inspections: {},
  readers: {},
  hypotheses: {},

  saved: { questions: [], systems: [], status: "idle", error: null, file: null, warning: null },

  setWriting: (partial) =>
    set((s) => {
      const writing = fitWriting({ ...s.writing, ...partial });
      return { writing, maxNewTokens: writing.length };
    }),

  pickToken: (k) =>
    set((s) => {
      if (!s.answer || !Number.isInteger(k) || k < 0) return {};
      const a = { ...s.answer, wanted: k };
      if (liveSets.has(a.requestId)) liveSets.set(a.requestId, a);
      return {
        answer: a,
        incoming: s.incoming?.requestId === a.requestId ? a : s.incoming,
        ...(a.sightings[k] ? { sighting: a.sightings[k] } : {}),
      };
    }),

  cancelSweep: () => {
    const id = sentOrder[0];
    if (id === undefined || !wsOpen()) return;
    ws!.send(JSON.stringify({ action: "cancel_sweep", request_id: id }));
    const a = liveSets.get(id);
    if (a) publish({ ...a, stopAsked: true });
  },

  fetchSaved: async () => {
    set((s) => ({ saved: { ...s.saved, status: "reading" } }));
    try {
      const lists = savedListsOf(await requestJson("GET", "/api/saved_prompts"));
      set((s) => ({ saved: { ...s.saved, ...lists, status: "ok", error: null } }));
    } catch (e) {
      // the lists last read stay usable: choosing still works, saving and forgetting wait for the server
      set((s) => ({ saved: { ...s.saved, status: "failed", error: String((e as Error)?.message ?? e) } }));
    }
  },

  keepSaved: async (kind, name, replaceId) => {
    const st = get();
    const body = {
      ...savedBody(kind, name, kind === "question" ? { question: st.question, answerBegins: st.answerBegins, chat: st.chat } : { text: st.system }),
      ...(replaceId ? { replace_id: replaceId } : {}),
    };
    return writeSaved(() => requestJson("POST", "/api/saved_prompts", body));
  },

  forgetSaved: async (kind, id) =>
    writeSaved(() => requestJson("DELETE", `/api/saved_prompts/${encodeURIComponent(kind)}/${encodeURIComponent(id)}`)),

  restoreSaved: async (kind, entry) => {
    const from =
      "text" in entry ? { text: entry.text } : { question: entry.question, answerBegins: entry.answerBegins, chat: entry.chat };
    return writeSaved(() => requestJson("POST", "/api/saved_prompts", savedBody(kind, entry.name, from)));
  },

  useSaved: (kind, entry) => {
    if (kind === "question" && "question" in entry) set({ question: entry.question, answerBegins: entry.answerBegins, chat: entry.chat });
    else if (kind === "system" && "text" in entry) set({ system: entry.text });
  },

  setQuestion: (question) => set({ question }),
  setAnswerBegins: (answerBegins) => set({ answerBegins }),
  setChat: (chat) => set({ chat }),
  setSystem: (system) => set({ system }),
  setFrame: (partial) => set((s) => ({ frame: { ...s.frame, ...partial } })),
  setPoleRanking: (poleRanking) => set({ poleRanking }),
  setThreadScore: (threadScore) => set({ threadScore }),
  setCompareThink: (compareThink) => set({ compareThink }),
  setLandmarks: (landmarks) => set({ landmarks }),
  setSurveyMode: (surveyMode, lockedPole = null) => set({ surveyMode, lockedPole }),
  setSteering: (partial) => set((s) => ({ steering: { ...s.steering, ...partial } })),
  setSelectedLayer: (idx) => set({ selectedLayer: Math.max(0, Math.min(59, Math.round(idx))) }),
  setLarqlQuery: (larqlQuery) => set({ larqlQuery }),
  // keep the condition, not the word: a word-piece of an answer is kept as that answer's token-0 sighting, read where the
  // question ends, which is what a control is compared at (later word-pieces are centred on it reading the same answer)
  setControl: (control) =>
    set((s) => {
      if (!control) return { control: null };
      const holder = [s.answer, s.incoming].find((a) => a && a.sightings[control.tokenIndex] === control);
      return { control: holder?.sightings[0] ?? control };
    }),

  connect: () => {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
    if (reconnectTimer !== null) {
      window.clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    let sock: WebSocket;
    try {
      sock = new WebSocket(`${proto}//${window.location.host}/ws/theodolite`);
    } catch {
      set({ conn: "closed" });
      return;
    }
    sock.binaryType = "arraybuffer";
    ws = sock;
    set({ conn: "connecting" });

    sock.onopen = () => {
      // the backoff is reset only when the server has answered something (see the replies below)
      set({ conn: "open", sweepError: null });
      void get().fetchHealth();
      void get().fetchSaved();
      // the vault lives in the server's memory: after a restart its ledger of edits is empty, so ask it again
      wsQueries.push({ query: "DESCRIBE", silent: true });
      sock.send(JSON.stringify({ action: "larql", query: "DESCRIBE" }));
      // a fresh live sighting of the current question replaces the recorded one, unless that very sighting
      // failed on the last connection (it would only fail again, every few seconds); it is written as the study
      // wrote, greedily and six word-pieces long, whatever the page is set to, with "sight each" as set
      if (get().sighting?.source !== "live" && !autoRetakeBlocked) {
        takeSightingImpl({ writing: { ...STUDY_WRITING, sightEach: get().writing.sightEach } }, true);
      }
    };

    sock.onclose = () => {
      if (ws === sock) ws = null;
      const lost = wsQueries.filter((q) => !q.silent);
      wsQueries.length = 0;
      pendingBinary = null;
      const wasSweeping = pending.size > 0;
      // the sets in flight keep what arrived, and say the connection closed before the rest
      for (const id of [...sentOrder]) finalise(id, { error: CONNECTION_CLOSED });
      const fail = <T,>(m: Record<string, Pending<T>>) =>
        Object.fromEntries(Object.entries(m).map(([k, v]) => [k, v.pending ? { ...v, pending: false, error: CLOSED_EARLY } : v]));
      set((s) => ({
        conn: "closed",
        isSweeping: false,
        larqlPending: false,
        sweepError: wasSweeping ? "The server closed the connection before the sighting came back." : s.sweepError,
        inspections: fail(s.inspections),
        readers: fail(s.readers),
      }));
      // statements in flight got no answer: say so in the vault's log rather than letting them vanish
      for (const q of lost) {
        recordLarql({ id: ++larqlId, query: q.query, ok: false, data: null, error: CLOSED_EARLY, at: Date.now(), via: "ws", failure: "transport" });
      }
      if (!get().sighting) void get().loadFixture();
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = null;
        get().connect();
      }, reconnectDelay);
      reconnectDelay = Math.min(30000, reconnectDelay * 1.6);
    };

    sock.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        const bin = parseBinaryFrame(event.data);
        if (!bin) {
          console.error("Invalid THEO binary frame");
          return;
        }
        pendingBinary = bin;
        return;
      }
      if (typeof event.data !== "string") return;
      let msg: Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
      try {
        msg = JSON.parse(event.data);
      } catch (err) {
        console.error("Failed to parse JSON WebSocket message:", err);
        return;
      }
      switch (msg.type) {
        case "theodolite_metadata": {
          const bin = pendingBinary;
          pendingBinary = null;
          onPair(msg, bin);
          break;
        }
        case "theodolite_progress": {
          reconnectDelay = 4000;
          onProgress(msg);
          break;
        }
        case "theodolite_written": {
          reconnectDelay = 4000;
          onWritten(msg);
          break;
        }
        case "theodolite_sweep_end": {
          reconnectDelay = 4000;
          const id = typeof msg.request_id === "string" ? msg.request_id : null;
          if (id === null || !pending.has(id)) {
            console.warn("A sweep ended for a request this page is not waiting on:", msg.request_id);
            break;
          }
          finalise(id, { cancelled: msg.cancelled === true, error: typeof msg.error === "string" && msg.error ? msg.error : null });
          break;
        }
        case "larql_result": {
          reconnectDelay = 4000;
          // the reply does not echo its statement; the socket answers in the order statements were sent
          const sent = wsQueries.shift() ?? { query: "", silent: false };
          const entry: LarqlEntry = {
            id: ++larqlId,
            query: sent.query,
            ok: Boolean(msg.ok),
            data: msg.ok ? msg.data : null,
            error: msg.ok ? null : String(msg.error ?? "error"),
            at: Date.now(),
            via: "ws",
            failure: msg.ok ? null : "refused",
          };
          recordLarql(entry, sent.silent);
          break;
        }
        case "pole_inspection": {
          reconnectDelay = 4000;
          const d = msg.data ?? {};
          const key = poleKey(d.layer, d.neuron_idx);
          set((s) => ({
            inspections: { ...s.inspections, [key]: { ...s.inspections[key], pending: false, data: d, error: null } },
          }));
          break;
        }
        case "downstream_readers": {
          reconnectDelay = 4000;
          const src = msg.data?.source_pole ?? {};
          const key = poleKey(src.layer, src.neuron_idx);
          set((s) => ({
            readers: { ...s.readers, [key]: { ...s.readers[key], pending: false, data: msg.data, error: null } },
          }));
          break;
        }
        case "error": {
          const message = String(msg.message ?? "The server reported an error.");
          if (msg.recoverable === true) {
            // a malformed request failed alone; the socket stays open and everything else is still answered in order
            if (msg.action === "sweep") {
              pendingBinary = null;
              const rid = typeof msg.request_id === "string" ? msg.request_id : null;
              if (rid !== null) {
                // that set's error, the others untouched; its sweep_end follows and ends it, keeping what arrived
                const a = liveSets.get(rid);
                if (a) publish({ ...a, error: message }, { sweepError: message });
                else {
                  console.warn("A sweep error for a request this page is not waiting on:", rid);
                  set({ sweepError: message });
                }
              } else {
                // an older server: the oldest request failed alone, and nothing more comes for it
                const id = sentOrder[0];
                if (id !== undefined) finalise(id, { error: message });
                set({ sweepError: message });
              }
            } else if (msg.action === "inspect_pole" || msg.action === "trace_readers") {
              const key = poleKey(Number(msg.layer), Number(msg.neuron_idx));
              const which = msg.action === "inspect_pole" ? "inspections" : "readers";
              set((s) => ({ [which]: { ...s[which], [key]: { ...s[which][key], pending: false, error: message } } }) as Partial<TheodoliteState>);
            } else set({ sweepError: message });
            break;
          }
          // otherwise the server's handler loop ends on the error, so nothing pending will be answered
          if ([...pending.values()].some((r) => r.auto)) autoRetakeBlocked = true;
          pendingBinary = null;
          for (const id of [...sentOrder]) finalise(id, { error: message });
          // statements still waiting are logged as unanswered when the socket closes, just after this
          const fail = <T,>(m: Record<string, Pending<T>>) =>
            Object.fromEntries(Object.entries(m).map(([k, v]) => [k, v.pending ? { ...v, pending: false, error: message } : v]));
          set((s) => ({
            isSweeping: false,
            larqlPending: false,
            sweepError: message,
            inspections: fail(s.inspections),
            readers: fail(s.readers),
          }));
          break;
        }
      }
    };
  },

  loadFixture: async () => {
    if (fixtureCache) {
      if (!get().sighting) set({ sighting: fixtureCache, plainSighting: fixtureCache });
      return;
    }
    const mod = await import("../fixtures/e6-demo-sighting.json");
    fixtureCache = sightingFromFixture(mod.default as unknown as FixtureFile);
    if (!get().sighting) set({ sighting: fixtureCache, plainSighting: get().plainSighting ?? fixtureCache });
  },

  fetchHealth: async () => {
    try {
      const r = await fetch("/api/health");
      if (r.ok) {
        const h = await r.json();
        set({ health: { ...h, lens: lensOf(h?.lens) } as Health });
      }
    } catch {
      // health is decorative; the connection state already says what matters
    }
  },

  takeSighting: (opts) => takeSightingImpl(opts ?? {}, false),

  executeLarql: (custom) => {
    const query = (custom ?? get().larqlQuery).trim();
    if (!query) return;
    set({ larqlPending: true });
    if (wsOpen()) {
      wsQueries.push({ query, silent: false });
      ws!.send(JSON.stringify({ action: "larql", query }));
      return;
    }
    postJson("/api/larql", { query })
      .then((data) =>
        recordLarql({ id: ++larqlId, query, ok: true, data, error: null, at: Date.now(), via: "http", failure: null }),
      )
      .catch((e) =>
        recordLarql({
          id: ++larqlId,
          query,
          ok: false,
          data: null,
          error: String(e?.message ?? e),
          at: Date.now(),
          via: "http",
          failure: e instanceof RequestError && e.refused ? "refused" : "transport",
        }),
      );
  },

  inspectPole: (idx, neuron, face) => {
    const key = poleKey(idx, neuron);
    set((s) => ({ inspections: { ...s.inspections, [key]: { pending: true, data: s.inspections[key]?.data ?? null, error: null } } }));
    if (wsOpen()) {
      ws!.send(JSON.stringify({ action: "inspect_pole", layer: idx, neuron_idx: neuron, polarity: face }));
      return;
    }
    postJson("/api/scientist/inspect_pole", { layer: idx, neuron_idx: neuron, polarity: face })
      .then((data) => set((s) => ({ inspections: { ...s.inspections, [key]: { pending: false, data, error: null } } })))
      .catch((e) =>
        set((s) => ({
          inspections: { ...s.inspections, [key]: { pending: false, data: null, error: String(e?.message ?? e) } },
        })),
      );
  },

  traceReaders: (idx, neuron, face, opts) => {
    const key = poleKey(idx, neuron);
    const st = get();
    // the sighting's own text is exact (the server echoes what the model read, chat template included); its exact ids up
    // to the position read go too, so a word-piece of the answer is traced where it was read, never at the prompt's end
    // after re-tokenizing decoded text (an older server ignores token_ids and reads the text)
    const text = st.sighting?.textSent
      ? { prompt: st.sighting.textSent, chat: false }
      : promptFields(st.question, st.answerBegins, st.chat, st.system);
    const ids = st.sighting?.source === "live" ? st.sighting.contextIds : null;
    const body = {
      layer: idx,
      neuron_idx: neuron,
      polarity: face,
      ...text,
      ...(ids && ids.length ? { token_ids: ids } : {}),
      max_layer_hop: Math.max(1, Math.min(6, Math.round(opts?.hop ?? 6))),
      top_k: Math.max(1, Math.min(150, Math.round(opts?.topK ?? 10))),
      score: opts?.score ?? st.threadScore,
    };
    // a trace depends on the question: remember which sighting it was asked for, so an older one is not shown as this one's
    const sightingId = st.sighting?.id ?? null;
    set((s) => ({
      readers: { ...s.readers, [key]: { pending: true, data: null, error: null, sightingId, sent: body } },
    }));
    if (wsOpen()) {
      ws!.send(JSON.stringify({ action: "trace_readers", ...body }));
      return;
    }
    postJson("/api/scientist/trace_readers", body)
      .then((data) => set((s) => ({ readers: { ...s.readers, [key]: { ...s.readers[key], pending: false, data, error: null } } })))
      .catch((e) =>
        set((s) => ({ readers: { ...s.readers, [key]: { ...s.readers[key], pending: false, data: null, error: String(e?.message ?? e) } } })),
      );
  },

  testHypothesis: async (idx, neuron, pos, neg, opts) => {
    const key = poleKey(idx, neuron);
    const st = get();
    // Bare text by default: the server reads each sentence's own last word-piece. As a chat it would read the
    // end of the model-turn header, the same template tokens for every sentence.
    const chat = opts?.chat ?? false;
    const body = {
      layer: idx,
      neuron_idx: neuron,
      pos_prompts: pos,
      neg_prompts: neg,
      chat,
      system: chat ? st.system.trim() || null : null,
    };
    set((s) => ({ hypotheses: { ...s.hypotheses, [key]: { pending: true, data: s.hypotheses[key]?.data ?? null, error: null, sent: body } } }));
    try {
      const data = await postJson("/api/scientist/test_hypothesis", body);
      set((s) => ({ hypotheses: { ...s.hypotheses, [key]: { pending: false, data, error: null, sent: body } } }));
    } catch (e) {
      set((s) => ({
        hypotheses: { ...s.hypotheses, [key]: { pending: false, data: null, error: String((e as Error)?.message ?? e), sent: body } },
      }));
    }
  },
}));

/** The sampler fields of a sweep, every one sent: a field left blank would be filled in from the checkpoint's own
 * recommendation (temperature 1, top-k 64, top-p 0.95) without saying so. Greedy is temperature 0 (argmax); top-k 0 and
 * top-p 1 are "no limit"; the seed null asks the server to draw one, which it sends back. */
export function samplerPayload(w: Writing) {
  return w.mode === "greedy"
    ? { temperature: 0, top_k: 0, top_p: 1, presence_penalty: w.presencePenalty, seed: null }
    : { temperature: w.temperature, top_k: w.topK, top_p: w.topP, presence_penalty: w.presencePenalty, seed: w.seed };
}

/** The payload of a "sweep" request for the current settings (also shown in the pages' request panels). */
export function sweepPayload(
  s: Pick<
    TheodoliteState,
    | "chat"
    | "system"
    | "landmarks"
    | "surveyMode"
    | "lockedPole"
    | "compareThink"
    | "frame"
    | "poleRanking"
    | "threadScore"
    | "control"
  > & { writing?: Writing; maxNewTokens?: number },
  question: string,
  answerBegins: string,
  steering: SteeringConfig,
  overrides: { chat?: boolean; system?: string; answerIds?: number[]; sightOrder?: number[]; centreAnswer?: boolean } = {},
) {
  // a page that passes no `writing` (an older request panel) gets the study's writing at its length
  const writing = s.writing ?? { ...STUDY_WRITING, length: s.maxNewTokens ?? STUDY_WRITING.length };
  // never centre a question on itself: the state less itself is nothing, and the reading would be noise
  const chatSent = overrides.chat ?? s.chat;
  const systemSent = chatSent ? (overrides.system ?? s.system).trim() : "";
  const controlIsThis =
    !!s.control &&
    s.control.question.trim() === question.trim() &&
    s.control.answerBegins.trim() === answerBegins.trim() &&
    s.control.chatWrapped === chatSent &&
    (s.control.system ?? "").trim() === systemSent;
  const control = s.frame.centre && s.control && !controlIsThis ? s.control : null;
  // the control's exact prompt ids where it has them (never its text tokenized again); its text only from a sighting
  // that keeps no ids
  const centreIds = control?.promptIds && control.promptIds.length ? control.promptIds : null;
  return {
    ...promptFields(question, answerBegins, chatSent, overrides.system ?? s.system),
    monuments: s.landmarks,
    survey_mode: s.surveyMode,
    locked_pole: s.lockedPole,
    compare_think: s.compareThink,
    max_new_tokens: writing.length,
    sampler: samplerPayload(writing),
    sight_every_token: writing.sightEach,
    max_sighted: SIGHT_CAP,
    stop_at_end_of_turn: true,
    // the legacy space-then-digit rule took a rank from the next position; with every word-piece sighted it is not used
    rank_lookahead: false,
    steering,
    frame_squaring: s.frame.squaring,
    frame_per_layer: s.frame.perLayer,
    frame_fnw: s.frame.fnw,
    centre_prompts: control && !centreIds && control.textSent ? [control.textSent] : [],
    ...(centreIds ? { centre_token_ids: [centreIds] } : {}),
    // later word-pieces are centred on the control reading this very answer, never on the control's prompt end
    centre_answer: control ? (overrides.centreAnswer ?? true) : false,
    pole_ranking: s.poleRanking,
    thread_score: s.threadScore,
    ...(overrides.answerIds ? { answer_token_ids: overrides.answerIds } : {}),
    ...(overrides.sightOrder ? { sight_tokens: overrides.sightOrder } : {}),
  };
}

/** The fields of a sweep that change only how its answer is read, not what is written or read. */
const READING_KEYS = [
  "monuments",
  "survey_mode",
  "locked_pole",
  "frame_squaring",
  "frame_per_layer",
  "frame_fnw",
  "centre_prompts",
  "centre_token_ids",
  "centre_answer",
  "pole_ranking",
  "thread_score",
] as const;

const omit = (o: Record<string, unknown>, keys: readonly string[]) => Object.fromEntries(Object.entries(o).filter(([k]) => !keys.includes(k)));

function recordLarql(e: LarqlEntry, silent = false) {
  const st = useTheodoliteStore.getState();
  // keep the ledger of active edits in step with whatever the vault reports
  const d = e.data as Record<string, any> | null; // eslint-disable-line @typescript-eslint/no-explicit-any
  let activeEdits = st.activeEdits;
  if (d && Array.isArray(d.active_patches)) activeEdits = latestPerNeuron(d.active_patches as ActiveEdit[]);
  else if (d && d.status === "applied" && d.patch) {
    // a new edit of a neuron replaces the vault's overlay for that neuron: list only the latest one
    const p = d.patch as ActiveEdit;
    activeEdits = [...st.activeEdits.filter((x) => !(x.layer_idx === p.layer_idx && x.neuron_idx === p.neuron_idx)), p];
  } else if (d && d.status === "rolled_back") activeEdits = [];
  useTheodoliteStore.setState({
    larqlLog: silent ? st.larqlLog : [e, ...st.larqlLog].slice(0, 50),
    larqlPending: e.via === "ws" ? wsQueries.some((q) => !q.silent) : false,
    activeEdits,
  });
}

/** The vault keeps one overlay per neuron (a new edit replaces the old) but a record per statement: keep the last. */
function latestPerNeuron(edits: ActiveEdit[]): ActiveEdit[] {
  const last = new Map<string, ActiveEdit>();
  for (const e of edits) last.set(`${e.layer_idx}:${e.neuron_idx}`, e);
  return [...last.values()].sort((a, b) => a.patch_id - b.patch_id);
}

/* ------------------------------------------------------------ the stream of one request */

/** A set's sightings that have arrived, by count. */
const arrivedCount = (a: AnswerSet) => Object.keys(a.sightings).length;

/** Put a set's new state in the store: as `answer` when it is the set shown, as `incoming` when it is the oldest in flight. */
function publish(a: AnswerSet | null, extra: Partial<TheodoliteState> = {}) {
  if (a && pending.has(a.requestId)) liveSets.set(a.requestId, a);
  useTheodoliteStore.setState((s) => ({
    answer: a && s.answer?.requestId === a.requestId ? a : s.answer,
    incoming: sentOrder.length ? (liveSets.get(sentOrder[0]) ?? null) : null,
    isSweeping: pending.size > 0,
    ...extra,
  }));
}

/** The request a frame answers: by its request_id; an older server's untagged frame answers the oldest request. */
function requestOf(msg: Record<string, unknown>): { id: string; p: PendingReq; legacy: boolean } | null {
  const legacy = typeof msg.request_id !== "string";
  const id = legacy ? sentOrder[0] : (msg.request_id as string);
  const p = id !== undefined ? pending.get(id) : undefined;
  if (id === undefined || !p) {
    console.warn(`A ${String(msg.type)} frame for a request this page is not waiting on:`, msg.request_id ?? "(untagged)");
    return null;
  }
  return { id, p, legacy };
}

const numOr = (v: unknown, d: number) => (typeof v === "number" && Number.isFinite(v) ? v : d);
const intList = (v: unknown): number[] | null =>
  Array.isArray(v) && v.every((x) => typeof x === "number" && Number.isInteger(x)) ? (v as number[]).slice() : null;

function thinkingOf(s: Sighting): AnswerSet["thinking"] {
  return s.thinkingContinuation != null
    ? { continuation: s.thinkingContinuation, seed: s.thinkSeed, readAt: s.thinkReadAt, path: s.thinkingPath }
    : null;
}

/** theodolite_progress: where the request in flight stands (only a server that streams sends it). */
function onProgress(msg: Record<string, unknown>) {
  const r = requestOf(msg);
  if (!r) return;
  const a = liveSets.get(r.id);
  if (!a) return;
  const writing = msg.phase === "writing";
  const done = numOr(msg.done, 0);
  publish({
    ...a,
    streamed: true,
    progress: { phase: writing ? "writing" : "sighting", done: writing ? done : Math.max(done, arrivedCount(a)), of: numOr(msg.of, a.progress.of) },
  });
}

/** theodolite_written: the answer is written; its word-pieces, how it was written, and the order they will be sighted in. */
function onWritten(msg: Record<string, unknown>) {
  const r = requestOf(msg);
  if (!r) return;
  const a0 = liveSets.get(r.id);
  if (!a0) return;
  const order = intList(msg.sight_order) ?? a0.order;
  const nToSight = numOr(msg.n_to_sight, order.length);
  const generation = generationOf(msg.generation, msg.answer) ?? a0.generation;
  if (r.p.mergeInto) {
    publish({ ...a0, order, nToSight, generation, streamed: true, progress: { phase: "sighting", done: arrivedCount(a0), of: nToSight } });
    return;
  }
  const a: AnswerSet = {
    ...a0,
    pieces: piecesFromAnswer(msg.answer) ?? a0.pieces,
    generation,
    // a re-reading keeps the record of how its word-pieces were first written
    wrote: a0.retakeOf ? a0.wrote : generation,
    order,
    nToSight,
    streamed: true,
    progress: { phase: "sighting", done: arrivedCount(a0), of: nToSight },
  };
  // the word-piece to show first must be one it sights
  if (order.length && !order.includes(a.wanted)) a.wanted = order[0];
  if (r.p.keepWanted !== null && order.length && !order.includes(r.p.keepWanted)) r.p.keepWanted = order[0];
  publish(a);
}

/** A (binary, JSON) pair: one word-piece's sighting, put in its set; the set is shown once the word-piece wanted arrives. */
function onPair(msg: Record<string, any>, bin: LiveBinary | null) { // eslint-disable-line @typescript-eslint/no-explicit-any
  const r = requestOf(msg);
  if (!r) return;
  const { id, p, legacy } = r;
  const sighting = sightingFromLive(msg, bin, p);
  reconnectDelay = 4000;
  autoRetakeBlocked = false;
  const st = useTheodoliteStore.getState();
  const k = sighting.tokenIndex;
  const extra: Partial<TheodoliteState> = { sweepError: null, activeEdits: latestPerNeuron(sighting.activeEdits) };
  const a0 = liveSets.get(id);
  if (!a0) return;
  const sightings = { ...a0.sightings, [k]: sighting };
  const count = Object.keys(sightings).length;
  const nToSight = Math.max(a0.nToSight, numOr(msg.n_to_sight, 0), count);
  let a: AnswerSet = {
    ...a0,
    sightings,
    order: a0.order.includes(k) ? a0.order : [...a0.order, k],
    nToSight,
    pieces: a0.pieces ?? piecesOf(sighting),
    generation: a0.generation ?? sighting.generation,
    thinking: a0.thinking ?? thinkingOf(sighting),
    streamed: !legacy,
    progress: { phase: "sighting", done: count, of: nToSight },
  };
  if (!a.retakeOf && !a.wrote) a.wrote = a.generation;

  if (p.mergeInto) {
    // "sight this one too": the new word-piece joins the set shown, read with that set's own request
    const t = st.answer;
    if (k === p.mergeInto.k && t && t.requestId === p.mergeInto.setId && !t.sightings[k]) {
      const joined: AnswerSet = {
        ...t,
        sightings: { ...t.sightings, [k]: sighting },
        order: t.order.includes(k) ? t.order : [...t.order, k],
        nToSight: t.order.includes(k) ? t.nToSight : t.nToSight + 1,
      };
      extra.answer = joined;
      if (joined.wanted === k) extra.sighting = sighting;
    }
  } else {
    if (k === 0 && !sighting.isNudged) extra.plainSighting = sighting;
    if (!p.promoted) {
      // a fresh answer is shown at once (on the word-piece wanted, else this one); a re-reading only once the word-piece
      // the reader is on arrives, so the reading shown never jumps to another word-piece
      let show: number | null;
      if (p.keepWanted === null) show = a.sightings[a.wanted] ? a.wanted : k;
      else {
        const onOld = a.retakeOf !== null && st.answer?.requestId === a.retakeOf ? st.answer.wanted : p.keepWanted;
        const want = a.order.includes(onOld) ? onOld : p.keepWanted;
        show = k === want ? want : null;
      }
      if (show !== null) {
        p.promoted = true;
        a = { ...a, wanted: show };
        extra.answer = a;
        extra.sighting = a.sightings[show];
      }
    } else if (k === a.wanted && st.answer?.requestId === a.requestId) {
      extra.sighting = sighting;
    }
  }
  publish(a, extra);
  // an older server sends one pair per request and nothing after it
  if (legacy) finalise(id, {});
}

/**
 * End a request: theodolite_sweep_end, an older server's single pair or its error, or the socket closing. A set
 * keeps what arrived; one never shown is shown now if anything arrived.
 */
function finalise(id: string, how: { cancelled?: boolean; error?: string | null }) {
  const p = pending.get(id);
  if (!p) return;
  pending.delete(id);
  const at = sentOrder.indexOf(id);
  if (at >= 0) sentOrder.splice(at, 1);
  const a0 = liveSets.get(id);
  liveSets.delete(id);
  if (!a0) {
    publish(null);
    return;
  }
  const count = arrivedCount(a0);
  // the sighting taken by itself on (re)connect failed: do not resend it on every reconnect
  if (p.auto && count === 0 && how.error && how.error !== CONNECTION_CLOSED) autoRetakeBlocked = true;
  const a: AnswerSet = {
    ...a0,
    done: true,
    cancelled: how.cancelled === true,
    stopAsked: false,
    error: a0.error ?? how.error ?? null,
    progress: { phase: "done", done: count, of: Math.max(a0.nToSight, count) },
  };
  if (p.mergeInto) {
    publish(null, a.error ? { sweepError: a.error } : {});
    return;
  }
  if (!p.promoted && count > 0) {
    const w = a.sightings[a.wanted] ? a.wanted : (a.order.find((i) => a.sightings[i]) ?? Number(Object.keys(a.sightings)[0]));
    const shown = { ...a, wanted: w };
    p.promoted = true;
    publish(shown, { answer: shown, sighting: shown.sightings[w] });
    return;
  }
  publish(a);
}

/** Send a sweep with its own request_id, and start its set. */
function dispatch(
  payload: Record<string, unknown>,
  req: Omit<PendingReq, "id" | "promoted">,
  start: Pick<AnswerSet, "question" | "answerBegins" | "chat" | "system" | "nudged" | "wanted" | "retakeOf" | "wrote" | "pieces" | "order" | "nToSight">,
  of: number,
): boolean {
  if (!wsOpen()) return false;
  const id = newRequestId();
  const sent = { ...payload, request_id: id };
  const a: AnswerSet = {
    requestId: id,
    ...start,
    payload: sent,
    sightings: {},
    progress: { phase: "sending", done: 0, of },
    done: false,
    cancelled: false,
    error: null,
    generation: null,
    thinking: null,
    stopAsked: false,
    streamed: false,
  };
  if (!req.auto) autoRetakeBlocked = false;
  pending.set(id, { ...req, id, promoted: false });
  sentOrder.push(id);
  liveSets.set(id, a);
  publish(a, { sweepError: null });
  ws!.send(JSON.stringify({ action: "sweep", payload: sent }));
  return true;
}

const NOT_ANSWERING =
  "The server is not answering, so no new sighting can be taken. The pages show the study's recorded sighting until it returns.";

function takeSightingImpl(opts: TakeOptions, auto: boolean): boolean {
  const { getState: get, setState: set } = useTheodoliteStore;
  const s = get();
  const question = opts.question ?? s.question;
  const answerBegins = opts.answerBegins ?? s.answerBegins;
  const chat = opts.chat ?? s.chat;
  const system = opts.system ?? s.system;
  const steering = { ...s.steering, ...(opts.steering ?? {}) };
  if (opts.question !== undefined || opts.answerBegins !== undefined) set({ question, answerBegins });
  if (!wsOpen()) {
    set({ sweepError: NOT_ANSWERING });
    if (!s.sighting) void s.loadFixture();
    return false;
  }
  const writing = fitWriting({
    ...s.writing,
    ...(opts.writing ?? {}),
    ...(opts.sightEach !== undefined ? { sightEach: opts.sightEach } : {}),
  });
  const given = opts.answerIds && opts.answerIds.length ? opts.answerIds : undefined;
  const payload = sweepPayload({ ...s, writing }, question, answerBegins, steering, {
    chat,
    system,
    answerIds: given,
    sightOrder: opts.sightOrder,
    centreAnswer: opts.centreAnswer,
  });
  const order = opts.sightOrder ?? [];
  return dispatch(
    payload,
    {
      question,
      answerBegins,
      textSent: describeText(question, answerBegins, chat),
      nudged: steering.enabled,
      chat,
      auto,
      steering: { ...steering },
      keepWanted: opts.keepWanted ?? null,
      mergeInto: null,
    },
    {
      question,
      answerBegins,
      chat,
      system: chat ? system.trim() : "",
      nudged: steering.enabled,
      wanted: opts.keepWanted ?? order[0] ?? 0,
      retakeOf: null,
      wrote: null,
      pieces: null,
      order,
      nToSight: order.length,
    },
    given ? Math.max(order.length, 1) : writing.length,
  );
}

/** The set the shown sighting belongs to, or null (the recorded sighting, or none). */
export function shownSetOf(st: Pick<TheodoliteState, "answer" | "sighting">): AnswerSet | null {
  const a = st.answer;
  const s = st.sighting;
  return a && s && a.sightings[s.tokenIndex] === s ? a : null;
}

/** The set the shown sighting belongs to, for a page (null for the recorded sighting). */
export const useShownSet = () => useTheodoliteStore((st) => shownSetOf(st));

/** A re-reading's request: the shown set's own (its text, chat, system, nudge and sampler), its answer given by ids so
 * nothing is written again, and the seed it was written with kept on the record. */
function rereadPayload(a: AnswerSet, answerIds: number[], sightTokens: number[], reading: Record<string, unknown> | null) {
  const base = omit(a.payload, reading ? [...READING_KEYS, "request_id", "answer_token_ids", "sight_tokens"] : ["request_id"]);
  const seed = (a.wrote ?? a.generation)?.seed ?? null;
  const sampler = { ...((a.payload.sampler as Record<string, unknown> | undefined) ?? {}), ...(seed !== null ? { seed } : {}) };
  return { ...base, ...(reading ?? {}), sampler, answer_token_ids: answerIds, sight_tokens: sightTokens };
}

/**
 * After an option that changes only the reading (pole ranking, thread score, frame, landmarks): read the shown answer
 * again from its exact ids, with nothing written again (exact even when it was drawn by chance), sighting the same
 * word-pieces, the one shown first; the sighting shown stays until that word-piece arrives. Without such an answer, a
 * fresh sighting with the current settings. Only when the server is answering and idle.
 */
export function retakeIfLive() {
  const st = useTheodoliteStore.getState();
  if (st.conn !== "open" || st.isSweeping) return;
  const a = shownSetOf(st);
  const s = st.sighting;
  if (!a || !s || s.source !== "live" || !a.streamed || !s.answerIds || s.answerIds.length === 0) {
    st.takeSighting();
    return;
  }
  // the reading options as set now; everything else as the shown set sent it
  const reading = Object.fromEntries(
    Object.entries(sweepPayload(st, a.question, a.answerBegins, st.steering, { chat: a.chat, system: a.system })).filter(([k]) =>
      (READING_KEYS as readonly string[]).includes(k),
    ),
  );
  const others = a.order.filter((i) => i !== a.wanted && a.sightings[i]);
  const sightTokens = [a.wanted, ...others].slice(0, SIGHT_CAP);
  dispatch(
    rereadPayload(a, s.answerIds, sightTokens, reading),
    {
      question: a.question,
      answerBegins: a.answerBegins,
      textSent: s.textSent ?? describeText(a.question, a.answerBegins, a.chat),
      nudged: a.nudged,
      chat: a.chat,
      auto: false,
      steering: (a.payload.steering as Record<string, unknown> | null) ?? {},
      keepWanted: a.wanted,
      mergeInto: null,
    },
    {
      question: a.question,
      answerBegins: a.answerBegins,
      chat: a.chat,
      system: a.system,
      nudged: a.nudged,
      wanted: a.wanted,
      retakeOf: a.requestId,
      wrote: a.wrote ?? a.generation,
      pieces: a.pieces,
      order: sightTokens,
      nToSight: sightTokens.length,
    },
    sightTokens.length,
  );
}

/**
 * Sight one more word-piece of the shown answer, one written but not sighted (past the cap, or with "sight each" off):
 * the shown set's own request read again from its exact ids, sighting only word-piece k, which joins the set. A frame
 * built from the word-pieces it reads (landmarks from a neuron) is built where the set's first sighting was, as before.
 */
export function sightPieceToo(k: number): boolean {
  const st = useTheodoliteStore.getState();
  const a = shownSetOf(st);
  const s = st.sighting;
  if (st.conn !== "open" || st.isSweeping || !a || !a.streamed || !s?.answerIds || k < 0 || k >= s.answerIds.length || a.sightings[k]) {
    return false;
  }
  const first = a.payload.survey_mode === "pole_motif" && a.order.length && a.order[0] !== k ? [a.order[0]] : [];
  const sightTokens = [...first, k];
  st.pickToken(k);
  return dispatch(
    rereadPayload(a, s.answerIds, sightTokens, null),
    {
      question: a.question,
      answerBegins: a.answerBegins,
      textSent: s.textSent ?? describeText(a.question, a.answerBegins, a.chat),
      nudged: a.nudged,
      chat: a.chat,
      auto: false,
      steering: (a.payload.steering as Record<string, unknown> | null) ?? {},
      keepWanted: null,
      mergeInto: { setId: a.requestId, k },
    },
    {
      question: a.question,
      answerBegins: a.answerBegins,
      chat: a.chat,
      system: a.system,
      nudged: a.nudged,
      wanted: k,
      retakeOf: a.requestId,
      wrote: a.wrote ?? a.generation,
      pieces: a.pieces,
      order: sightTokens,
      nToSight: sightTokens.length,
    },
    sightTokens.length,
  );
}

function download(name: string, data: unknown) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

const stamp = () => new Date().toISOString().replace(/[:.]/g, "-");

/** Save the current sighting as a .json file in the viewer's downloads: which word-piece it is and how its answer was
 * written are in the sighting (token_index, read_pos, chosen, generation), and in the file's name. */
export function exportSighting(s: Sighting) {
  const { raw, ...rest } = s;
  download(`sighting-${s.source}${s.source === "live" ? `-k${s.tokenIndex}` : ""}-${stamp()}.json`, {
    sighting: rest,
    word_piece: { token_index: s.tokenIndex, read_pos: s.readPos, read_at: s.readAt, chosen: s.chosen },
    generation: s.generation,
    server_metadata: raw ?? null,
  });
}

/** Save a whole answer's sightings as one .json file: the request as sent, how it was written, its word-pieces, every
 * sighting once, and the server's own metadata once (the first pair sent, which also carries the thinking comparison;
 * the others differ only in what each sighting already holds). */
export function exportAnswer(a: AnswerSet) {
  const ks = Object.keys(a.sightings)
    .map(Number)
    .sort((x, y) => x - y);
  const first = a.order.find((i) => a.sightings[i]) ?? ks[0];
  download(`answer-${a.requestId}-${stamp()}.json`, {
    request_id: a.requestId,
    request: a.payload,
    question: a.question,
    answer_begins: a.answerBegins,
    chat: a.chat,
    system: a.system,
    nudged: a.nudged,
    generation: a.generation,
    written_with: a.wrote,
    retake_of: a.retakeOf,
    pieces: a.pieces,
    sight_order: a.order,
    n_to_sight: a.nToSight,
    sighted: ks.length,
    done: a.done,
    cancelled: a.cancelled,
    error: a.error,
    thinking: a.thinking,
    sightings: Object.fromEntries(
      ks.map((k) => {
        const { raw: _raw, ...rest } = a.sightings[k]; // eslint-disable-line @typescript-eslint/no-unused-vars
        return [k, rest];
      }),
    ),
    server_metadata_token_index: first ?? null,
    server_metadata: first !== undefined ? (a.sightings[first].raw ?? null) : null,
  });
}

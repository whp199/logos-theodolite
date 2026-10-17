import { create } from "zustand";
import {
  type ActiveEdit,
  type FixtureFile,
  type LiveBinary,
  type PoleRanking,
  type Sighting,
  type ThreadScore,
  parseBinaryFrame,
  sightingFromFixture,
  sightingFromLive,
} from "./sighting";

export * from "./sighting";

/**
 * The instrument's state, wired to the backend:
 *   WebSocket /ws/theodolite  actions sweep | larql | inspect_pole | trace_readers
 *   HTTP      /api/health, /api/larql, /api/scientist/{inspect_pole,trace_readers,test_hypothesis}
 * When no server answers, the recorded demonstration sighting (E6) is shown instead, and every page
 * says so.
 */

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
}

export interface LarqlEntry {
  id: number;
  query: string;
  ok: boolean;
  data: Record<string, unknown> | null;
  error: string | null;
  at: number;
  via: "ws" | "http";
}

export type Conn = "idle" | "connecting" | "open" | "closed";

type Keyed<T> = Record<string, T>;
export interface Pending<T> {
  pending: boolean;
  data: T | null;
  error: string | null;
}

export const poleKey = (idx: number, neuron: number) => `${idx}:${neuron}`;

export interface TheodoliteState {
  conn: Conn;
  health: Health | null;
  sighting: Sighting | null;
  /** the last sighting taken without a nudge, kept so the nudge page can compare */
  plainSighting: Sighting | null;
  /** a matched control kept by hand, to read the current sighting against (REPORT §3.5) */
  control: Sighting | null;
  isSweeping: boolean;
  sweepError: string | null;

  // the question
  question: string;
  answerBegins: string;
  /** wrap the question in the chat template (thinking off), with the begun answer after the model-turn header */
  chat: boolean;
  /** an optional system turn before the question (chat only) */
  system: string;
  compareThink: boolean;
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

  // actions
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
  takeSighting: (opts?: { question?: string; answerBegins?: string; steering?: Partial<SteeringConfig> }) => boolean;
  executeLarql: (query?: string) => void;
  inspectPole: (idx: number, neuron: number, face: number) => void;
  traceReaders: (idx: number, neuron: number, face: number, opts?: TraceOptions) => void;
  testHypothesis: (idx: number, neuron: number, pos: string[], neg: string[]) => Promise<void>;
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
/** LQL statements sent over the socket and not yet answered, oldest first */
const wsQueries: string[] = [];
/** requests sent and not yet answered, oldest first: the server answers one socket's sweeps in order */
const pendingReqs: { question: string; answerBegins: string; textSent: string; nudged: boolean; chat: boolean }[] = [];
let reconnectTimer: number | null = null;
let reconnectDelay = 4000;
let larqlId = 0;
let fixtureCache: Sighting | null = null;

const wsOpen = () => ws !== null && ws.readyState === WebSocket.OPEN;

async function postJson(path: string, body: unknown) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof data?.detail === "string" ? data.detail : `HTTP ${r.status}`);
  return data;
}

export const useTheodoliteStore = create<TheodoliteState>((set, get) => ({
  conn: "idle",
  health: null,
  sighting: null,
  plainSighting: null,
  control: null,
  isSweeping: false,
  sweepError: null,

  question: "Tell me about yourself.",
  answerBegins: "Hi! I",
  chat: true,
  system: "",
  compareThink: false,
  maxNewTokens: 6,
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
  threadScore: "coupling",

  selectedLayer: 35, // layer 36

  larqlQuery: "SELECT * FROM POLES NEAREST 'she' WHERE layer = 50 LIMIT 10",
  larqlLog: [],
  larqlPending: false,
  activeEdits: [],

  inspections: {},
  readers: {},
  hypotheses: {},

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
  setControl: (control) => set({ control }),

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
      reconnectDelay = 4000;
      set({ conn: "open", sweepError: null });
      void get().fetchHealth();
      // a fresh live sighting of the current question replaces the recorded one
      if (get().sighting?.source !== "live") get().takeSighting();
    };

    sock.onclose = () => {
      if (ws === sock) ws = null;
      pendingReqs.length = 0;
      wsQueries.length = 0;
      pendingBinary = null;
      const wasSweeping = get().isSweeping;
      set({
        conn: "closed",
        isSweeping: false,
        larqlPending: false,
        sweepError: wasSweeping ? "The server closed the connection before the sighting came back." : get().sweepError,
      });
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
          const req = pendingReqs.shift() ?? {
            question: get().question,
            answerBegins: get().answerBegins,
            textSent: describeText(get().question, get().answerBegins, get().chat),
            nudged: false,
            chat: get().chat,
          };
          const sighting = sightingFromLive(msg, pendingBinary, req);
          pendingBinary = null;
          set((s) => ({
            sighting,
            plainSighting: sighting.isNudged ? s.plainSighting : sighting,
            isSweeping: pendingReqs.length > 0,
            sweepError: null,
            activeEdits: sighting.activeEdits,
          }));
          break;
        }
        case "larql_result": {
          const entry: LarqlEntry = {
            id: ++larqlId,
            // the reply does not echo its statement; the socket answers in the order statements were sent
            query: wsQueries.shift() ?? "",
            ok: Boolean(msg.ok),
            data: msg.ok ? msg.data : null,
            error: msg.ok ? null : String(msg.error ?? "error"),
            at: Date.now(),
            via: "ws",
          };
          recordLarql(entry);
          break;
        }
        case "pole_inspection": {
          const d = msg.data ?? {};
          set((s) => ({
            inspections: { ...s.inspections, [poleKey(d.layer, d.neuron_idx)]: { pending: false, data: d, error: null } },
          }));
          break;
        }
        case "downstream_readers": {
          const src = msg.data?.source_pole ?? {};
          set((s) => ({
            readers: { ...s.readers, [poleKey(src.layer, src.neuron_idx)]: { pending: false, data: msg.data, error: null } },
          }));
          break;
        }
        case "error": {
          // the server's handler loop ends on an error, so nothing pending will be answered
          pendingReqs.length = 0;
          wsQueries.length = 0;
          pendingBinary = null;
          const message = String(msg.message ?? "The server reported an error.");
          const fail = <T,>(m: Record<string, Pending<T>>) =>
            Object.fromEntries(Object.entries(m).map(([k, v]) => [k, v.pending ? { pending: false, data: v.data, error: message } : v]));
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
      if (r.ok) set({ health: (await r.json()) as Health });
    } catch {
      // health is decorative; the connection state already says what matters
    }
  },

  takeSighting: (opts) => {
    const s = get();
    const question = opts?.question ?? s.question;
    const answerBegins = opts?.answerBegins ?? s.answerBegins;
    const steering = { ...s.steering, ...(opts?.steering ?? {}) };
    if (opts?.question !== undefined || opts?.answerBegins !== undefined) set({ question, answerBegins });
    if (!wsOpen()) {
      set({
        sweepError:
          "The server is not answering, so no new sighting can be taken. The pages show the study's recorded sighting until it returns.",
      });
      if (!s.sighting) void get().loadFixture();
      return false;
    }
    const textSent = describeText(question, answerBegins, s.chat);
    pendingReqs.push({ question, answerBegins, textSent, nudged: steering.enabled, chat: s.chat });
    set({ isSweeping: true, sweepError: null });
    ws!.send(JSON.stringify({ action: "sweep", payload: sweepPayload(s, question, answerBegins, steering) }));
    return true;
  },

  executeLarql: (custom) => {
    const query = (custom ?? get().larqlQuery).trim();
    if (!query) return;
    set({ larqlPending: true });
    if (wsOpen()) {
      wsQueries.push(query);
      ws!.send(JSON.stringify({ action: "larql", query }));
      return;
    }
    postJson("/api/larql", { query })
      .then((data) =>
        recordLarql({ id: ++larqlId, query, ok: true, data, error: null, at: Date.now(), via: "http" }),
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
    // the sighting's own text is exact (the server echoes what the model read, chat template included)
    const text = st.sighting?.textSent
      ? { prompt: st.sighting.textSent, chat: false }
      : promptFields(st.question, st.answerBegins, st.chat, st.system);
    const body = {
      layer: idx,
      neuron_idx: neuron,
      polarity: face,
      ...text,
      max_layer_hop: Math.max(1, Math.min(6, Math.round(opts?.hop ?? 6))),
      top_k: Math.max(1, Math.min(150, Math.round(opts?.topK ?? 10))),
      score: opts?.score ?? st.threadScore,
    };
    set((s) => ({ readers: { ...s.readers, [key]: { pending: true, data: s.readers[key]?.data ?? null, error: null } } }));
    if (wsOpen()) {
      ws!.send(JSON.stringify({ action: "trace_readers", ...body }));
      return;
    }
    postJson("/api/scientist/trace_readers", body)
      .then((data) => set((s) => ({ readers: { ...s.readers, [key]: { pending: false, data, error: null } } })))
      .catch((e) =>
        set((s) => ({ readers: { ...s.readers, [key]: { pending: false, data: null, error: String(e?.message ?? e) } } })),
      );
  },

  testHypothesis: async (idx, neuron, pos, neg) => {
    const key = poleKey(idx, neuron);
    set((s) => ({ hypotheses: { ...s.hypotheses, [key]: { pending: true, data: s.hypotheses[key]?.data ?? null, error: null } } }));
    try {
      const st = get();
      const data = await postJson("/api/scientist/test_hypothesis", {
        layer: idx,
        neuron_idx: neuron,
        pos_prompts: pos,
        neg_prompts: neg,
        // each sentence becomes the user's turn of a chat when the instrument sends chats
        chat: st.chat,
        system: st.chat ? st.system.trim() || null : null,
      });
      set((s) => ({ hypotheses: { ...s.hypotheses, [key]: { pending: false, data, error: null } } }));
    } catch (e) {
      set((s) => ({
        hypotheses: { ...s.hypotheses, [key]: { pending: false, data: null, error: String((e as Error)?.message ?? e) } },
      }));
    }
  },
}));

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
    | "maxNewTokens"
    | "frame"
    | "poleRanking"
    | "threadScore"
    | "control"
  >,
  question: string,
  answerBegins: string,
  steering: SteeringConfig,
) {
  const centre = s.frame.centre && s.control?.textSent ? [s.control.textSent] : [];
  return {
    ...promptFields(question, answerBegins, s.chat, s.system),
    monuments: s.landmarks,
    survey_mode: s.surveyMode,
    locked_pole: s.lockedPole,
    compare_think: s.compareThink,
    max_new_tokens: s.maxNewTokens,
    steering,
    frame_squaring: s.frame.squaring,
    frame_per_layer: s.frame.perLayer,
    frame_fnw: s.frame.fnw,
    centre_prompts: centre,
    pole_ranking: s.poleRanking,
    thread_score: s.threadScore,
  };
}

function recordLarql(e: LarqlEntry) {
  const st = useTheodoliteStore.getState();
  // keep the ledger of active edits in step with whatever the vault reports
  const d = e.data as Record<string, any> | null; // eslint-disable-line @typescript-eslint/no-explicit-any
  let activeEdits = st.activeEdits;
  if (d && Array.isArray(d.active_patches)) activeEdits = d.active_patches as ActiveEdit[];
  else if (d && d.status === "applied" && d.patch) activeEdits = [...st.activeEdits, d.patch as ActiveEdit];
  else if (d && d.status === "rolled_back") activeEdits = [];
  useTheodoliteStore.setState({
    larqlLog: [e, ...st.larqlLog].slice(0, 50),
    larqlPending: e.via === "ws" ? wsQueries.length > 0 : false,
    activeEdits,
  });
}

/** Take a fresh sighting with the current settings, when the server is answering and idle. */
export function retakeIfLive() {
  const st = useTheodoliteStore.getState();
  if (st.conn === "open" && !st.isSweeping) st.takeSighting();
}

/** Save the current sighting as a .json file in the viewer's downloads. */
export function exportSighting(s: Sighting) {
  const { raw, ...rest } = s;
  const blob = new Blob([JSON.stringify({ sighting: rest, server_metadata: raw ?? null }, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `sighting-${s.source}-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

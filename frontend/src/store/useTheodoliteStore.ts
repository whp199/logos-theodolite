import { create } from "zustand";
import {
  type ActiveEdit,
  type FixtureFile,
  type LiveBinary,
  type Sighting,
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
  /** absolute length the server adds (AddSpec normalize=True), not a share of ‖h‖ */
  scale: number;
  pole_layer: number | null;
  pole_neuron: number | null;
  pole_polarity: number;
}

/**
 * Safe nudge default (the nudge page's own default, ×0.01 of the stream). The server reads `scale` as one
 * absolute length (normalize=True), so a share of
 * the stream strength must be converted on this side: 0.01 × ‖h‖ at the weakest stream among the
 * default layers, layer 54, whose median non-BOS ‖h‖ is 88.3 (V7_tool_bugs_02_steer_p0.json). The tool
 * audit puts a nudge at or below one hundredth of ‖h‖ (V7 notes, bug 1). The old screen default was
 * 280, i.e. 1.0-3.2 × ‖h‖, which collapses the model.
 */
export const SAFE_NUDGE_SCALE = 0.883;
export const DEFAULT_NUDGE_LAYERS = [29, 35, 41, 47, 53];

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
  compareThink: boolean;
  maxNewTokens: number;
  landmarks: [string, string, string];
  surveyMode: "monument" | "pole_motif";
  lockedPole: [number, number, number] | null;
  steering: SteeringConfig;

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
  traceReaders: (idx: number, neuron: number, face: number) => void;
  testHypothesis: (idx: number, neuron: number, pos: string[], neg: string[]) => Promise<void>;
}

/** The text sent as the prompt. The server applies no chat template yet, so the question and the
 * begun answer go in as bare text, one line each. */
export function composeText(question: string, answerBegins: string) {
  const q = question.trim();
  const a = answerBegins.trim();
  return a ? `${q}\n${a}` : q;
}

let ws: WebSocket | null = null;
let pendingBinary: LiveBinary | null = null;
/** LQL statements sent over the socket and not yet answered, oldest first */
const wsQueries: string[] = [];
/** requests sent and not yet answered, oldest first: the server answers one socket's sweeps in order */
const pendingReqs: { question: string; answerBegins: string; textSent: string; nudged: boolean }[] = [];
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
  },

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
            textSent: composeText(get().question, get().answerBegins),
            nudged: false,
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
    const textSent = composeText(question, answerBegins);
    pendingReqs.push({ question, answerBegins, textSent, nudged: steering.enabled });
    set({ isSweeping: true, sweepError: null });
    ws!.send(
      JSON.stringify({
        action: "sweep",
        payload: {
          prompt: textSent,
          monuments: s.landmarks,
          survey_mode: s.surveyMode,
          locked_pole: s.lockedPole,
          compare_think: s.compareThink,
          max_new_tokens: s.maxNewTokens,
          steering,
        },
      }),
    );
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

  traceReaders: (idx, neuron, face) => {
    const key = poleKey(idx, neuron);
    const prompt = get().sighting?.textSent ?? composeText(get().question, get().answerBegins);
    set((s) => ({ readers: { ...s.readers, [key]: { pending: true, data: s.readers[key]?.data ?? null, error: null } } }));
    if (wsOpen()) {
      ws!.send(JSON.stringify({ action: "trace_readers", layer: idx, neuron_idx: neuron, polarity: face, prompt }));
      return;
    }
    postJson("/api/scientist/trace_readers", { layer: idx, neuron_idx: neuron, polarity: face, prompt })
      .then((data) => set((s) => ({ readers: { ...s.readers, [key]: { pending: false, data, error: null } } })))
      .catch((e) =>
        set((s) => ({ readers: { ...s.readers, [key]: { pending: false, data: null, error: String(e?.message ?? e) } } })),
      );
  },

  testHypothesis: async (idx, neuron, pos, neg) => {
    const key = poleKey(idx, neuron);
    set((s) => ({ hypotheses: { ...s.hypotheses, [key]: { pending: true, data: s.hypotheses[key]?.data ?? null, error: null } } }));
    try {
      const data = await postJson("/api/scientist/test_hypothesis", {
        layer: idx,
        neuron_idx: neuron,
        pos_prompts: pos,
        neg_prompts: neg,
      });
      set((s) => ({ hypotheses: { ...s.hypotheses, [key]: { pending: false, data, error: null } } }));
    } catch (e) {
      set((s) => ({
        hypotheses: { ...s.hypotheses, [key]: { pending: false, data: null, error: String((e as Error)?.message ?? e) } },
      }));
    }
  },
}));

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

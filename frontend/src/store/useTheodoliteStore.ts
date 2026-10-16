import { create } from "zustand";

export interface TokenReadout {
  token: string;
  prob?: number;
  logit: number;
}

export interface LayerPlumeMeta {
  layer: number;
  layer_1based: number;
  is_global_attention: boolean;
  jpp_top_tokens: TokenReadout[];
  naive_top_token: string;
  tracked_ranks: Record<string, number>;
  tracked_logits: Record<string, number>;
  attn_norm: number;
  ffn_norm: number;
  ffn_dominance: number;
  azimuth_deg: number;
  elevation_deg: number;
  residual_norm: number;
}

export interface PoleLabelMeta {
  rank: number;
  neuron_idx: number;
  polarity: string;
  abs_act: number;
  j_vindex_tokens: TokenReadout[];
  naive_tokens: TokenReadout[];
}

export interface FilamentMeta {
  src_layer: number;
  src_rank: number;
  src_neuron: number;
  src_polarity: string;
  dst_layer: number;
  dst_rank: number;
  dst_neuron: number;
  dst_polarity: string;
  coupling_score: number;
  dot_product: number;
}

export interface SteeringConfig {
  enabled: boolean;
  target_concept: string;
  contrast_concept: string;
  layers: number[];
  scale: number;
  pole_layer: number | null;
  pole_neuron: number | null;
  pole_polarity: number;
}

export interface TheodoliteState {
  wsConnected: boolean;
  isSweeping: boolean;
  prompt: string;
  monuments: [string, string, string];
  surveyMode: "monument" | "pole_motif";
  lockedPole: [number, number, number] | null;
  compareThink: boolean;
  steering: SteeringConfig;

  selectedLayer: number;
  selectedPoleRank: number;
  showAttnSplit: boolean;
  showFfnSplit: boolean;
  showDipoles: boolean;
  showFilaments: boolean;
  showPlumes: boolean;

  // Binary Float32 buffers
  nLayers: number;
  nPolesPerLayer: number;
  nFilaments: number;
  trajXyz: Float32Array;
  trajAnglesNorm: Float32Array;
  deltaAttnXyz: Float32Array;
  deltaFfnXyz: Float32Array;
  ckaMatrix: Float32Array;
  polesData: Float32Array;
  filamentsData: Float32Array;
  thinkTrajXyz: Float32Array | null;

  // JSON metadata
  generatedText: string;
  thinkGeneratedText: string | null;
  surveyLabels: [string, string, string];
  elapsedMs: number;
  peakVramGb: number;
  isSteered: boolean;
  layerPlumes: LayerPlumeMeta[];
  polesLabels: Record<string, PoleLabelMeta[]>;
  filamentsMeta: FilamentMeta[];
  larqlQuery: string;
  larqlOutput: Record<string, unknown> | null;
  poleInspection: Record<string, unknown> | null;
  downstreamReaders: Record<string, unknown> | null;
  activePatchesCount: number;
  fps: number;

  // Actions
  setPrompt: (p: string) => void;
  setMonuments: (m: [string, string, string]) => void;
  setSurveyMode: (mode: "monument" | "pole_motif", lockedPole?: [number, number, number] | null) => void;
  setCompareThink: (v: boolean) => void;
  setSteering: (s: Partial<SteeringConfig>) => void;
  setSelectedLayer: (l: number) => void;
  setSelectedPoleRank: (r: number) => void;
  toggleLayerVisibility: (key: "showAttnSplit" | "showFfnSplit" | "showDipoles" | "showFilaments" | "showPlumes") => void;
  setLarqlQuery: (q: string) => void;
  setFps: (fps: number) => void;

  connectWebSocket: () => void;
  triggerSweep: () => void;
  executeLarql: (query?: string) => void;
  inspectPole: (layer: number, neuronIdx: number, polarity: number) => void;
  traceReaders: (layer: number, neuronIdx: number, polarity: number) => void;
}

const GLOBAL_LAYERS_0 = new Set([5, 11, 17, 23, 29, 35, 41, 47, 53, 59]);

function createInitialBuffers() {
  const nLayers = 60;
  const nPoles = 15;
  const trajXyz = new Float32Array(nLayers * 3);
  const trajAnglesNorm = new Float32Array(nLayers * 3);
  const deltaAttnXyz = new Float32Array(nLayers * 4);
  const deltaFfnXyz = new Float32Array(nLayers * 4);
  const ckaMatrix = new Float32Array(nLayers * nLayers);
  const polesData = new Float32Array(nLayers * nPoles * 6);
  const filamentsData = new Float32Array(24 * 7);
  const layerPlumes: LayerPlumeMeta[] = [];

  for (let l = 0; l < nLayers; l++) {
    const t = l / 59;
    // Bridge ("spider") rises in mid layers (L24-L51), then transitions toward "eight" (L51-L60)
    const xSpider = Math.sin(t * Math.PI) * 32.0 + (l >= 23 && l <= 53 ? 18.0 : 2.0);
    const yEight = Math.pow(t, 2.4) * 44.0;
    const zInsect = Math.sin(t * Math.PI * 2.0) * 8.0;

    trajXyz[l * 3 + 0] = xSpider;
    trajXyz[l * 3 + 1] = yEight;
    trajXyz[l * 3 + 2] = zInsect;

    const az = (Math.atan2(yEight, xSpider) * 180) / Math.PI;
    const projNorm = Math.hypot(xSpider, yEight, zInsect) + 1e-6;
    const el = (Math.asin(zInsect / projNorm) * 180) / Math.PI;
    const rNorm = 45.0 + l * 6.8;

    trajAnglesNorm[l * 3 + 0] = az;
    trajAnglesNorm[l * 3 + 1] = el;
    trajAnglesNorm[l * 3 + 2] = rNorm;

    const isGlobal = GLOBAL_LAYERS_0.has(l);
    const attnMag = isGlobal ? 14.5 : 5.2;
    const ffnMag = l >= 20 ? 16.8 : 7.4;

    deltaAttnXyz[l * 4 + 0] = Math.cos(l * 0.3) * (isGlobal ? 4.2 : 1.5);
    deltaAttnXyz[l * 4 + 1] = Math.sin(l * 0.3) * (isGlobal ? 3.8 : 1.2);
    deltaAttnXyz[l * 4 + 2] = isGlobal ? 1.8 : 0.5;
    deltaAttnXyz[l * 4 + 3] = attnMag;

    deltaFfnXyz[l * 4 + 0] = l >= 22 && l <= 48 ? 4.8 : 1.2;
    deltaFfnXyz[l * 4 + 1] = l >= 42 ? 5.6 : 1.0;
    deltaFfnXyz[l * 4 + 2] = -0.8;
    deltaFfnXyz[l * 4 + 3] = ffnMag;

    for (let l2 = 0; l2 < nLayers; l2++) {
      const dist = Math.abs(l - l2) / 60;
      ckaMatrix[l * nLayers + l2] = Math.max(0, Math.exp(-dist * 2.8));
    }

    for (let p = 0; p < nPoles; p++) {
      const base = (l * nPoles + p) * 6;
      const angle = (p / nPoles) * Math.PI * 2 + l * 0.17;
      const pol = p % 2 === 0 ? 1.0 : -1.0;
      polesData[base + 0] = (l * 353 + p * 997) % 21504;
      polesData[base + 1] = pol;
      polesData[base + 2] = Math.max(1.2, 18.0 - p * 0.9);
      polesData[base + 3] = Math.cos(angle);
      polesData[base + 4] = Math.sin(angle);
      polesData[base + 5] = pol * 0.35;
    }

    const topTok =
      l < 18
        ? " animal"
        : l < 44
          ? " spider"
          : l < 55
            ? " Spider"
            : " 8";

    layerPlumes.push({
      layer: l,
      layer_1based: l + 1,
      is_global_attention: isGlobal,
      jpp_top_tokens: [
        { token: topTok, prob: l >= 42 ? 0.42 : 0.16, logit: l >= 42 ? 28.7 : 18.1 },
        { token: l >= 50 ? " eight" : " arachnid", prob: 0.11, logit: 22.4 },
      ],
      naive_top_token: " animal",
      tracked_ranks: {
        spider: l >= 41 && l <= 56 ? 1 : l >= 26 ? 34 : 4606,
        eight: l >= 58 ? 1 : l >= 50 ? 4 : 645,
      },
      tracked_logits: {
        spider: l >= 41 ? 28.7 : 15.3,
        eight: l >= 50 ? 28.6 : 12.3,
      },
      attn_norm: attnMag,
      ffn_norm: ffnMag,
      ffn_dominance: ffnMag / (attnMag + ffnMag),
      azimuth_deg: Number(az.toFixed(1)),
      elevation_deg: Number(el.toFixed(1)),
      residual_norm: Number(rNorm.toFixed(1)),
    });
  }

  for (let f = 0; f < 24; f++) {
    const srcL = Math.min(52, f * 2 + 4);
    const dstL = Math.min(59, srcL + 3 + (f % 3));
    filamentsData[f * 7 + 0] = srcL;
    filamentsData[f * 7 + 1] = f % 5;
    filamentsData[f * 7 + 2] = dstL;
    filamentsData[f * 7 + 3] = (f + 1) % 5;
    filamentsData[f * 7 + 4] = (f % 2 === 0 ? 1 : -1) * (18.5 + f * 1.4);
    filamentsData[f * 7 + 5] = f % 2 === 0 ? 1 : -1;
    filamentsData[f * 7 + 6] = f % 3 === 0 ? -1 : 1;
  }

  return {
    trajXyz,
    trajAnglesNorm,
    deltaAttnXyz,
    deltaFfnXyz,
    ckaMatrix,
    polesData,
    filamentsData,
    layerPlumes,
  };
}

const initial = createInitialBuffers();
let wsInstance: WebSocket | null = null;

export const useTheodoliteStore = create<TheodoliteState>((set, get) => ({
  wsConnected: false,
  isSweeping: false,
  // two-hop example prompt from Gurnee et al. 2026 (see REFERENCES.md [2])
  prompt: "The number of legs on the animal that spins webs is",
  monuments: ["spider", "eight", "insect"],
  surveyMode: "monument",
  lockedPole: null,
  compareThink: false,
  steering: {
    enabled: false,
    target_concept: "six",
    contrast_concept: "eight",
    layers: [35, 41, 47, 53, 59],
    scale: 280.0,
    pole_layer: null,
    pole_neuron: null,
    pole_polarity: 1.0,
  },

  selectedLayer: 44, // L45 where ' spider' hits #1
  selectedPoleRank: 0,
  showAttnSplit: true,
  showFfnSplit: true,
  showDipoles: true,
  showFilaments: true,
  showPlumes: true,

  nLayers: 60,
  nPolesPerLayer: 15,
  nFilaments: 24,
  trajXyz: initial.trajXyz,
  trajAnglesNorm: initial.trajAnglesNorm,
  deltaAttnXyz: initial.deltaAttnXyz,
  deltaFfnXyz: initial.deltaFfnXyz,
  ckaMatrix: initial.ckaMatrix,
  polesData: initial.polesData,
  filamentsData: initial.filamentsData,
  thinkTrajXyz: null,

  generatedText: " 8.\nThe number",
  thinkGeneratedText: null,
  surveyLabels: ["spider", "eight", "insect"],
  elapsedMs: 6797.9,
  peakVramGb: 23.02,
  isSteered: false,
  layerPlumes: initial.layerPlumes,
  polesLabels: {},
  filamentsMeta: [],
  larqlQuery: "SELECT * FROM EDGES NEAREST 'spider' WHERE layer = 35 LIMIT 5",
  larqlOutput: null,
  poleInspection: null,
  downstreamReaders: null,
  activePatchesCount: 0,
  fps: 60,

  setPrompt: (prompt) => set({ prompt }),
  setMonuments: (monuments) => set({ monuments }),
  setSurveyMode: (surveyMode, lockedPole = null) => set({ surveyMode, lockedPole }),
  setCompareThink: (compareThink) => set({ compareThink }),
  setSteering: (partial) => set((state) => ({ steering: { ...state.steering, ...partial } })),
  setSelectedLayer: (selectedLayer) => set({ selectedLayer }),
  setSelectedPoleRank: (selectedPoleRank) => set({ selectedPoleRank }),
  toggleLayerVisibility: (key) => set((state) => ({ [key]: !state[key] } as Partial<TheodoliteState>)),
  setLarqlQuery: (larqlQuery) => set({ larqlQuery }),
  setFps: (fps) => set({ fps }),

  connectWebSocket: () => {
    if (wsInstance && (wsInstance.readyState === WebSocket.OPEN || wsInstance.readyState === WebSocket.CONNECTING)) {
      return;
    }
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${proto}//${window.location.host}/ws/theodolite`;
    const ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";
    wsInstance = ws;

    ws.onopen = () => {
      set({ wsConnected: true });
      get().triggerSweep();
    };

    ws.onclose = () => {
      set({ wsConnected: false, isSweeping: false });
      wsInstance = null;
    };

    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        const buf = event.data;
        const dv = new DataView(buf);
        const magic = dv.getUint32(0, true);
        if (magic !== 0x5448454f) {
          console.error("Invalid THEO binary frame magic:", magic.toString(16));
          return;
        }
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

        const trajXyz = take(nLayers * 3);
        const trajAnglesNorm = take(nLayers * 3);
        const deltaAttnXyz = take(nLayers * 4);
        const deltaFfnXyz = take(nLayers * 4);
        const ckaMatrix = take(nLayers * nLayers);
        const polesData = take(nLayers * nPoles * 6);
        const filamentsData = take(nFilaments * 7);
        const thinkTrajXyz = (flags & 1) !== 0 ? take(nLayers * 3) : null;

        set({
          nLayers,
          nPolesPerLayer: nPoles,
          nFilaments,
          trajXyz,
          trajAnglesNorm,
          deltaAttnXyz,
          deltaFfnXyz,
          ckaMatrix,
          polesData,
          filamentsData,
          thinkTrajXyz,
          isSteered: (flags & 2) !== 0,
        });
      } else if (typeof event.data === "string") {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === "theodolite_metadata") {
            set({
              isSweeping: false,
              generatedText: msg.generated_text ?? "",
              thinkGeneratedText: msg.think_generated_text ?? null,
              surveyLabels: msg.survey_labels ?? ["X", "Y", "Z"],
              elapsedMs: msg.elapsed_ms ?? 0,
              peakVramGb: msg.peak_vram_gb ?? 0,
              isSteered: Boolean(msg.is_steered),
              layerPlumes: msg.layer_plumes ?? [],
              polesLabels: msg.poles_labels ?? {},
              filamentsMeta: msg.filaments_meta ?? [],
              activePatchesCount: msg.larql_status?.active_patches_count ?? 0,
            });
          } else if (msg.type === "larql_result") {
            set({
              larqlOutput: msg.ok ? msg.data : { error: msg.error },
              activePatchesCount: msg.data?.active_patches_count ?? get().activePatchesCount,
            });
          } else if (msg.type === "pole_inspection") {
            set({ poleInspection: msg.data });
          } else if (msg.type === "downstream_readers") {
            set({ downstreamReaders: msg.data });
          }
        } catch (err) {
          console.error("Failed to parse JSON WebSocket message:", err);
        }
      }
    };
  },

  triggerSweep: () => {
    const state = get();
    if (!wsInstance || wsInstance.readyState !== WebSocket.OPEN) {
      return;
    }
    set({ isSweeping: true });
    wsInstance.send(
      JSON.stringify({
        action: "sweep",
        payload: {
          prompt: state.prompt,
          monuments: state.monuments,
          survey_mode: state.surveyMode,
          locked_pole: state.lockedPole,
          compare_think: state.compareThink,
          max_new_tokens: 6,
          steering: state.steering,
        },
      })
    );
  },

  executeLarql: (customQuery) => {
    const q = customQuery ?? get().larqlQuery;
    if (wsInstance && wsInstance.readyState === WebSocket.OPEN) {
      wsInstance.send(JSON.stringify({ action: "larql", query: q }));
    } else {
      fetch("/api/larql", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q }),
      })
        .then((r) => r.json())
        .then((data) => set({ larqlOutput: data }))
        .catch((e) => set({ larqlOutput: { error: String(e) } }));
    }
  },

  inspectPole: (layer, neuronIdx, polarity) => {
    if (wsInstance && wsInstance.readyState === WebSocket.OPEN) {
      wsInstance.send(
        JSON.stringify({
          action: "inspect_pole",
          layer,
          neuron_idx: neuronIdx,
          polarity,
        })
      );
    }
  },

  traceReaders: (layer, neuronIdx, polarity) => {
    const prompt = get().prompt;
    if (wsInstance && wsInstance.readyState === WebSocket.OPEN) {
      wsInstance.send(
        JSON.stringify({
          action: "trace_readers",
          layer,
          neuron_idx: neuronIdx,
          polarity,
          prompt,
        })
      );
    }
  },
}));

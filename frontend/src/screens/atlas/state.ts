/**
 * The atlas page's state and what it does (after the explorer's app.js): which layer and neuron or head is open, what
 * the server said of it, the lens search and the run of a text, and the map (one AtlasMap, driven from here). Layers
 * count from 0 in the state, as the server's fields do; the page shows layer + 1.
 */
import { create } from "zustand";
import {
  AtlasError,
  b64ToU16,
  get,
  getBinary,
  halfToFloat,
  post,
  type Arch,
  type CensusUnit,
  type CensusUnitRow,
  type HeadInfo,
  type HeadProfile,
  type Hover,
  type NeuronHeads,
  type NeuronLens,
  type Notes,
  type Proposals,
  type RunReply,
  type RunView,
  type ScaleSource,
  type SearchReply,
  type Spread,
  type WebSummary,
  type Wiring,
} from "./api";
import type { AtlasMap, Mark, MapNote } from "./map";

export type Focus = { kind: "neuron" | "head"; layer: number; n: number };
export type RightTab = "census" | "push" | "details" | "passages" | "scientist";
export type LeftTab = "web" | "run";

export interface RunState {
  text: string | null;
  /** where the scores' usual extremes come from (the stats index, or the study's 36 texts: provisional) */
  scaleSource: ScaleSource | null;
  tokens: string[];
  pos: number[];
  pin: number;
  hover: number | null;
  sign: "both" | "+" | "-";
  lo: number;
  hi: number;
  min: number;
  q: string;
  view: RunView | null;
  busy: boolean;
  error: string | null;
}

export interface AtlasState {
  arch: Arch | null;
  archError: string | null;
  layer: number;
  view: "map" | "spread";
  layoutKind: "index" | "writes";
  layoutNote: string | null;
  webOn: boolean;
  webFirst: number;
  webTotal: number | null;
  focus: Focus | null;
  layerMode: number | null;
  units: Record<number, CensusUnitRow[]>;
  notes: Notes | null;
  /** the scientist's proposed names and notes, waiting for the person */
  proposals: Proposals | null;
  unit: CensusUnit | null;
  lens: NeuronLens | null;
  wiring: Wiring | null;
  wiringError: string | null;
  nheads: NeuronHeads | null;
  headInfo: HeadInfo | null;
  headProfile: HeadProfile | null;
  webSum: (WebSummary & { layer: number; n: number }) | null;
  leftTab: LeftTab;
  leftHidden: boolean;
  qMode: "lens" | "fires";
  query: string;
  search: SearchReply | null;
  searchError: string | null;
  run: RunState;
  rightTab: RightTab;
  rightOpen: boolean;
  spread: (Spread & { layer: number }) | null;
  colorBy: "neuron" | "prompt" | "dose" | "kind";
  scientistMarks: boolean;
}

const RUN0: RunState = { text: null, scaleSource: null, tokens: [], pos: [], pin: -1, hover: null, sign: "both", lo: 1, hi: 60, min: 1, q: "", view: null, busy: false, error: null };

export const useAtlas = create<AtlasState>(() => ({
  arch: null,
  archError: null,
  layer: 0,
  view: "map",
  layoutKind: "writes",
  layoutNote: null,
  webOn: true,
  webFirst: 5,
  webTotal: null,
  focus: null,
  layerMode: null,
  units: {},
  notes: null,
  proposals: null,
  unit: null,
  lens: null,
  wiring: null,
  wiringError: null,
  nheads: null,
  headInfo: null,
  headProfile: null,
  webSum: null,
  leftTab: "web",
  leftHidden: false,
  qMode: "lens",
  query: "",
  search: null,
  searchError: null,
  run: RUN0,
  rightTab: "scientist",
  rightOpen: false,
  spread: null,
  colorBy: "neuron",
  scientistMarks: false,
}));

const set = useAtlas.setState;
const st = useAtlas.getState;

let map: AtlasMap | null = null;
export const setMap = (m: AtlasMap | null) => {
  map = m;
};
export const theMap = () => map;

export const isFocus = (layer: number, n: number, kind: Focus["kind"]) => {
  const f = st().focus;
  return !!f && f.layer === layer && f.n === n && f.kind === kind;
};

const message = (e: unknown) => (e instanceof AtlasError ? e.message : String((e as Error)?.message ?? e));

/* ------------------------------------------------------------------------------------------------------- start */

const LAYER_KEY = "theodolite.atlas.layer";
const FOCUS_KEY = "theodolite.atlas.focus";

export async function loadArch(): Promise<Arch | null> {
  try {
    const arch = await get<Arch>("/api/atlas/arch");
    let layer = Math.min(arch.n_layers - 1, 53);
    try {
      const v = localStorage.getItem(LAYER_KEY);
      if (v !== null && +v >= 0 && +v < arch.n_layers) layer = +v;
    } catch {
      // not remembered
    }
    set({ arch, archError: null, layer, run: { ...st().run, hi: arch.n_layers } });
    void loadNotes();
    void loadProposals();
    return arch;
  } catch (e) {
    set({ archError: message(e) });
    return null;
  }
}

export function restoreFocus(deep?: { layer: number; n: number } | null) {
  if (deep) return void openNeuron(deep.layer, deep.n, { fly: true });
  try {
    const f = JSON.parse(localStorage.getItem(FOCUS_KEY) || "null") as Focus | null;
    if (f && f.kind === "head") void openHead(f.layer, f.n);
    else if (f) void openNeuron(f.layer, f.n, { fly: false });
  } catch {
    // nothing kept
  }
}

function saveFocus() {
  try {
    localStorage.setItem(FOCUS_KEY, JSON.stringify(st().focus));
  } catch {
    // not remembered
  }
}

export async function loadNotes() {
  try {
    const notes = await get<Notes>("/api/atlas/notes");
    set({ notes });
    const keys = Object.entries(notes.neurons)
      .filter(([, e]) => e.name)
      .map(([k]) => {
        const [l, n] = k.split(":").map(Number);
        return { layer: l, n };
      });
    map?.setNamed(keys);
  } catch {
    // the names stay as they were
  }
}

/* -------------------------------------------------------------------------------------------- layer and layout */

export async function setLayer(l: number) {
  set({ layer: l });
  try {
    localStorage.setItem(LAYER_KEY, String(l));
  } catch {
    // not remembered
  }
  const arch = st().arch;
  if (!st().units[l] && arch?.census[String(l)]) {
    try {
      const us = await get<CensusUnitRow[]>("/api/atlas/census/units", { layer: l });
      set({ units: { ...st().units, [l]: us } });
    } catch {
      set({ units: { ...st().units, [l]: [] } });
    }
  }
  const us = st().units[l];
  if (st().layer === l && us?.length) map?.setUnits(l, us);
  if (st().view === "spread") void loadSpread();
}

export async function setLayout(kind: "index" | "writes") {
  set({ layoutKind: kind });
  if (kind === "index") {
    map?.setLayout(null);
    return;
  }
  try {
    const r = await get<{ available: boolean; cell?: string; shape?: number[]; method?: string }>("/api/atlas/layout");
    if (st().layoutKind !== kind) return;
    if (!r.available || !r.cell) {
      set({ layoutKind: "index", layoutNote: "The layout index is not built: neurons lie in index order (scripts/atlas_build.py layout)." });
      map?.setLayout(null);
      return;
    }
    const u = b64ToU16(r.cell);
    map?.setLayout(new Int16Array(u.buffer));
    set({ layoutNote: null });
  } catch (e) {
    set({ layoutKind: "index", layoutNote: message(e) });
  }
}

export async function loadSpread() {
  const l = st().layer;
  if (!st().arch?.census[String(l)]) return set({ spread: null });
  try {
    const d = await get<Spread>("/api/atlas/census/spread", { layer: l });
    if (st().layer === l) set({ spread: d.x ? { ...d, layer: l } : null });
  } catch {
    set({ spread: null });
  }
}

/* ------------------------------------------------------------------------------------------------- the web */

let webKey: string | null = null;
export async function refreshWeb() {
  const { focus: f, webOn, webFirst } = st();
  if (!webOn || !f || f.kind !== "neuron") {
    webKey = null;
    set({ webSum: null, webTotal: null });
    map?.setWeb(null);
    return;
  }
  const key = `${f.layer}/${f.n}/${webFirst}`;
  if (key === webKey) return;
  webKey = key;
  try {
    const r = await getBinary("/api/atlas/web", { layer: f.layer, n: f.n, first: webFirst, k: 2 });
    if (webKey !== key) return;
    map?.setWeb(new Int16Array(r.data));
    set({ webTotal: Number(r.headers.get("X-Web-Total") ?? 0) });
    const sum = await get<WebSummary>("/api/atlas/web/summary", { layer: f.layer, n: f.n, first: webFirst, k: 2, top: 24 });
    if (webKey === key) set({ webSum: { ...sum, layer: f.layer, n: f.n } });
  } catch {
    if (webKey === key) set({ webSum: null });
  }
}

/* --------------------------------------------------------------------------------------------- opening things */

let liveCtl: AbortController | null = null;

export async function openNeuron(layer: number, n: number, o: { fly?: boolean; fromScientist?: boolean } = {}) {
  if (!o.fromScientist) clearScientistMarks();
  liveCtl?.abort();
  const ctl = (liveCtl = new AbortController());
  set({
    focus: { kind: "neuron", layer, n },
    unit: null,
    lens: null,
    wiring: null,
    wiringError: null,
    nheads: null,
    headInfo: null,
    headProfile: null,
    rightOpen: true,
    // the scientist stays up when it is showing; otherwise the neuron's census answers, else its details
    rightTab: st().rightTab === "scientist" && st().rightOpen ? "scientist" : st().rightTab === "scientist" ? "census" : st().rightTab,
  });
  saveFocus();
  map?.selectHead(null);
  map?.setHeadLevels(null);
  if (map && map.layerMode !== null && map.layerMode !== layer) map.enterLayer(layer);
  if (layer !== st().layer) await setLayer(layer);
  if (!isFocus(layer, n, "neuron")) return;
  map?.select(layer, n, o.fly !== false);
  void refreshWeb();
  void (async () => {
    try {
      const nh = await get<NeuronHeads>("/api/atlas/neuron/heads", { layer, n }, ctl.signal);
      if (!isFocus(layer, n, "neuron")) return;
      set({ nheads: nh });
      map?.setNeuronHeads(roleHeads(nh).slice(0, 8));
      lightHeads();
    } catch {
      // the heads stay unread
    }
  })();
  void (async () => {
    try {
      // with the sign in use beside the one from rest, where the passages index lets it be read (the details tab shows both)
      const w = await get<Wiring>("/api/atlas/neuron/wiring", { layer, n, k: 24, model: "use" }, ctl.signal);
      if (!isFocus(layer, n, "neuron")) return;
      set({ wiring: w });
      map?.setConnections(w.down, w.up);
    } catch (e) {
      if (isFocus(layer, n, "neuron") && (e as Error)?.name !== "AbortError") set({ wiringError: message(e) });
    }
  })();
  void (async () => {
    try {
      const lens = await get<NeuronLens>("/api/atlas/neuron/lens", { layer, n }, ctl.signal);
      if (isFocus(layer, n, "neuron")) set({ lens });
    } catch {
      // shown as unread
    }
  })();
  let u: CensusUnit = { layer, n, kind: "neuron", name: "" };
  try {
    u = await get<CensusUnit>("/api/atlas/census/unit", { layer, n }, ctl.signal);
  } catch (e) {
    if ((e as Error)?.name === "AbortError") return;
  }
  if (!isFocus(layer, n, "neuron")) return;
  set({ unit: u });
  if (st().rightTab === "census" && !(u.rollouts && u.rollouts.length)) set({ rightTab: "details" });
}

/** Where a neuron is opened from a list or a link: the camera centres on it as well. */
export function goToNeuron(layer: number, n: number) {
  void openNeuron(layer, n, { fly: false });
  map?.focusNeuron(layer, n);
}

export async function openHead(layer: number, head: number, o: { fromScientist?: boolean } = {}) {
  if (!o.fromScientist) clearScientistMarks();
  liveCtl?.abort();
  set({
    focus: { kind: "head", layer, n: head },
    unit: null,
    lens: null,
    wiring: null,
    nheads: null,
    headInfo: null,
    headProfile: null,
    leftHidden: false,
    leftTab: "web",
    // a head opens in the left panel: the right one steps aside unless the scientist is showing
    rightOpen: st().rightOpen && st().rightTab === "scientist",
  });
  saveFocus();
  map?.select(null);
  map?.selectHead(layer, head);
  void refreshWeb();
  lightHeads();
  try {
    const r = await get<HeadInfo>("/api/atlas/head", { layer, head });
    if (!isFocus(layer, head, "head")) return;
    set({ headInfo: r });
    map?.setHeadLinks(layer, head, headRoleLinks(r));
  } catch {
    // unread
  }
  try {
    const p = await get<HeadProfile>("/api/atlas/head/profile", { layer, head });
    if (isFocus(layer, head, "head")) set({ headProfile: p });
  } catch {
    // unread
  }
}

export function closeFocus() {
  liveCtl?.abort();
  set({ focus: null, unit: null, lens: null, wiring: null, nheads: null, headInfo: null, headProfile: null, webSum: null });
  if (st().rightTab !== "scientist") set({ rightOpen: false });
  map?.select(null);
  map?.selectHead(null);
  void refreshWeb();
  lightHeads();
  saveFocus();
}

export const roleHeads = (r: NeuronHeads) => {
  const seen = new Set<string>();
  return (["q", "k", "v", "o"] as const)
    .flatMap((role) => (r.roles?.[role] ?? []).map((x) => ({ ...x, role })))
    .filter((x) => x.z >= 3)
    .sort((a, b) => b.z - a.z)
    .filter((x) => !seen.has(`${x.layer}/${x.head}`) && !!seen.add(`${x.layer}/${x.head}`));
};

export const headRoleLinks = (h: HeadInfo) =>
  (["q", "k", "v", "o"] as const).flatMap((role) => (h[role] ?? []).slice(0, 5).map((c) => ({ ...c, role, side: role === "o" ? ("down" as const) : ("up" as const) })));

/* ------------------------------------------------------------------------------------- the search and the run */

let qGen = 0;
let searchTimer: number | undefined;

export function setQuery(q: string, mode?: "lens" | "fires") {
  const m = mode ?? st().qMode;
  set({ query: q, qMode: m, leftTab: m === "fires" ? "run" : "web", leftHidden: false });
  window.clearTimeout(searchTimer);
  searchTimer = window.setTimeout(() => void runSearch(), m === "fires" ? 700 : 250);
}

export async function runSearch() {
  const q = st().query.trim();
  const g = ++qGen;
  if (q.length < 2) {
    set({ search: null, searchError: null, run: { ...st().run, text: null, tokens: [], pos: [], pin: -1, view: null } });
    syncMarks();
    return;
  }
  if (st().qMode === "fires") return runText(q, g);
  try {
    const r = await get<SearchReply>("/api/atlas/search", { q, limit: 400 });
    if (g !== qGen) return;
    set({ search: r, searchError: null });
    syncMarks();
  } catch (e) {
    if (g === qGen) set({ searchError: message(e) });
  }
}

export async function runText(text: string, g = ++qGen) {
  const r0 = st().run;
  if (text !== r0.text) {
    set({ run: { ...r0, busy: true, error: null } });
    try {
      const r = await post<RunReply>("/api/atlas/run", { text });
      if (g !== qGen) return;
      set({ run: { ...st().run, text, scaleSource: r.scale_source ?? null, tokens: r.tokens, pos: r.pos, pin: -1, hover: null, busy: false, error: null } });
    } catch (e) {
      if (g === qGen) set({ run: { ...st().run, busy: false, error: message(e) } });
      return;
    }
  }
  await runView(g);
}

let viewGen = 0;
let pathsGen = 0;

export async function runView(g = qGen) {
  const r = st().run;
  if (!r.text) return;
  const mine = ++viewGen;
  const pos = r.hover ?? r.pin;
  try {
    const v = await get<RunView>("/api/atlas/run/view", {
      pos,
      sign: r.sign,
      lo: Math.max(0, r.lo - 1),
      hi: Math.max(0, r.hi - 1),
      min: r.min,
      q: r.q,
      limit: 300,
    });
    if (mine !== viewGen || g !== qGen) return;
    set({ run: { ...st().run, view: v } });
    syncMarks();
  } catch (e) {
    set({ run: { ...st().run, error: message(e) } });
  }
  void showTokenPaths(pos);
  lightHeads();
}

async function showTokenPaths(pos: number) {
  const mine = ++pathsGen;
  if (pos < 0 || st().leftTab !== "run") {
    webKey = null;
    return void refreshWeb();
  }
  try {
    const r = await getBinary("/api/atlas/run/paths", { pos, min: st().run.min || 1 });
    if (mine !== pathsGen) return;
    webKey = `tok/${pos}`;
    map?.setWeb(new Int16Array(r.data));
  } catch {
    // no paths
  }
}

let headsGen = 0;
export async function lightHeads() {
  const g = ++headsGen;
  const { leftTab, run, focus: f, nheads } = st();
  const pos = run.hover ?? run.pin;
  if (leftTab === "run" && run.text && pos >= 0) {
    try {
      const r = await get<{ layers: number[][] | null }>("/api/atlas/run/heads", { pos });
      if (g !== headsGen) return;
      if (r.layers) return map?.setHeadLevels(r.layers.map((row) => row.map((v) => Math.max(0, (v - 0.15) / 0.6))));
    } catch {
      // fall through
    }
  }
  if (g !== headsGen) return;
  if (f?.kind === "neuron" && nheads && nheads.layer === f.layer && nheads.n === f.n) {
    const lv: number[][] = [];
    for (const [l, zs] of Object.entries(nheads.z_by_layer)) lv[+l] = zs.map((z) => Math.max(0, Math.min(1, ((z ?? 0) - 2.5) / 5)));
    map?.setHeadLevels(lv);
  } else map?.setHeadLevels(null);
}

/** The map shows what the open tab shows: lens hits on the web tab, what fired on the run tab. */
export function syncMarks() {
  if (st().scientistMarks) return;
  const { leftTab, search, run } = st();
  if (leftTab === "run" && run.view?.map && run.view.shape) {
    const u = b64ToU16(run.view.map);
    const N = run.view.shape[1];
    const marks: Mark[] = [];
    for (let i = 0; i < u.length; i++) {
      if (!u[i]) continue;
      const v = halfToFloat(u[i]);
      if (v) marks.push({ layer: Math.floor(i / N), n: i % N, sign: v < 0 ? "-" : "+", w: Math.min(1, 0.3 + Math.abs(v) / 4) });
    }
    map?.setMarks(marks.length > 40000 ? marks.sort((a, b) => (b.w ?? 0) - (a.w ?? 0)).slice(0, 40000) : marks);
    return;
  }
  map?.setMarks(leftTab === "web" && search?.results ? search.results.map((r) => ({ layer: r.layer, n: r.n, sign: r.sign, w: 1 })) : []);
}

export function setLeftTab(t: LeftTab) {
  set({ leftTab: t });
  syncMarks();
  if (t === "run") {
    const r = st().run;
    void showTokenPaths(r.hover ?? r.pin);
  } else if (String(webKey).startsWith("tok/")) {
    webKey = null;
    void refreshWeb();
  }
  lightHeads();
}

/* ---------------------------------------------------------------------------------------------- names, notes */

export async function saveNote(layer: number, n: number, body: { name?: string; note?: string }, by: "reader" | "scientist" = "reader") {
  const r = await post<{ entry: unknown; warning: string | null }>("/api/atlas/notes", { kind: "neuron", layer, index: n, by, ...body });
  await loadNotes();
  return r;
}

export async function loadProposals() {
  try {
    set({ proposals: await get<Proposals>("/api/atlas/proposals") });
  } catch {
    // they stay as they were
  }
}

/** The person's answer to the scientist's proposal on a neuron or head: kept, it is written to the notes as the
 * scientist's and becomes the name or note everywhere; discarded, it goes. `part` answers for the name or the note
 * alone. */
export async function decideProposal(kind: "neuron" | "head", layer: number, n: number, keep: boolean, part?: "name" | "note") {
  await post(`/api/atlas/proposals/${keep ? "keep" : "discard"}`, { kind, layer, index: n, ...(part ? { part } : {}) });
  await Promise.all([loadNotes(), loadProposals()]);
}

/* ---------------------------------------------------------------------------------------- the scientist's hand */

export function clearScientistMarks() {
  map?.setNotes([]);
  if (st().scientistMarks) {
    set({ scientistMarks: false });
    syncMarks();
  }
}

/** What the scientist does to the screen (backend/scientist_tools.py's screen tools; layers counted from 0). */
export async function scientistUi(e: Record<string, unknown>) {
  const num = (k: string) => Number(e[k]);
  switch (e.action) {
    case "select":
      await openNeuron(num("layer"), num("n"), { fly: false, fromScientist: true });
      map?.flyToNeuron(num("layer"), num("n"), 0.3);
      break;
    case "head":
      await openHead(num("layer"), num("head"), { fromScientist: true });
      break;
    case "highlight":
      set({ scientistMarks: true });
      map?.setMarks(((e.neurons as { layer: number; n: number; face?: string }[]) ?? []).map((x) => ({ layer: x.layer, n: x.n, sign: x.face === "-" ? "-" : "+", w: 1 })));
      break;
    case "annotate":
      map?.setNotes((e.items as MapNote[]) ?? []);
      break;
    case "rename":
      await loadNotes();
      break;
    case "propose":
      await loadProposals();
      break;
    case "camera":
      if (map?.layerMode !== null) map?.exitLayer();
      if (e.mode === "home") map?.goHome();
      else if (e.mode === "layer") map?.viewLayer(num("layer"));
      else if (e.mode === "neuron") map?.flyToNeuron(num("layer"), num("n"), Number(e.closeness) || 0.22);
      break;
    case "tab":
      if (st().focus && ["census", "push", "details", "passages"].includes(String(e.tab))) set({ rightTab: e.tab as RightTab });
      break;
    case "search":
      setQuery(String(e.query ?? ""), "lens");
      break;
    case "text": {
      set({ qMode: "fires", query: String(e.text ?? ""), leftTab: "run", leftHidden: false });
      const g = ++qGen;
      await runText(String(e.text ?? ""), g);
      const r = st().run;
      const token = String(e.token ?? "");
      const i = token ? r.tokens.findIndex((t) => t.includes(token)) : -1;
      set({ run: { ...st().run, pin: i >= 0 ? r.pos[i] : -1 } });
      await runView(g);
      break;
    }
  }
}

/** What hovering and the details have read (the tooltip's words, heads), the oldest dropped past a few thousand. */
export const hoverCache = new Map<string, Hover | NeuronHeads | HeadInfo>();
export function remember(key: string, v: Hover | NeuronHeads | HeadInfo) {
  hoverCache.set(key, v);
  if (hoverCache.size > 4000) for (const k of [...hoverCache.keys()].slice(0, 1000)) hoverCache.delete(k);
}

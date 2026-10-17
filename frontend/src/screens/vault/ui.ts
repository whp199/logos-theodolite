import { create } from "zustand";
import { useTheodoliteStore } from "../../store/useTheodoliteStore";
import { nearestStatement } from "./lql";
import { num } from "./live";

/**
 * What the reader has typed on the vault page (kept while they visit a neuron's folio and come back).
 * The vault itself lives in the store: executeLarql / larqlLog / activeEdits / larqlQuery.
 * Opening values are the approved board's.
 */
export type VaultTab = "find" | "edit" | "history" | "lql";

interface Asked {
  word: string;
  layer: string;
  k: string;
}

interface VaultUi {
  tab: VaultTab;
  // find
  fWord: string;
  fLayer: string;
  fK: string;
  asked: Asked;
  /** when the live search for `asked` was sent (null: not sent) */
  askedAt: number | null;
  naked: boolean;
  gate: number;
  faRaw: boolean;
  // edit
  eKind: "update" | "delete";
  eN: string;
  eLayer: string;
  eWord: string;
  eStrength: string;
  ePol: "+" | "-";
  editSent: { lql: string; at: number } | null;
  undoAt: number | null;
  // write LQL
  conRun: string | null;
  conRunAt: number | null;
  set: (p: Partial<Omit<VaultUi, "set">>) => void;
}

export const useVaultUi = create<VaultUi>((set) => ({
  tab: "find",
  fWord: "she",
  fLayer: "51",
  fK: "5",
  asked: { word: "she", layer: "51", k: "5" },
  askedAt: null,
  naked: false,
  gate: 0,
  faRaw: false,
  eKind: "update",
  eN: "100",
  eLayer: "36",
  eWord: "six",
  eStrength: "3.0",
  ePol: "+",
  editSent: null,
  undoAt: null,
  conRun: null,
  conRunAt: null,
  set: (p) => set(p),
}));

/** Whether a find can be written as a statement the vault reads. */
export function findIsSendable(word: string, layer: string, k: string) {
  const w = word.trim();
  const L = num(layer);
  const K = num(k);
  return w !== "" && !/['"]/.test(w) && L !== null && L >= 1 && L <= 60 && K !== null && K >= 1;
}

/** Ask the vault: record the question and send it to the running instrument (the store answers or records the refusal). */
export function runFind(word: string, layer: string, k: string) {
  const ui = useVaultUi.getState();
  const send = findIsSendable(word, layer, k);
  ui.set({ tab: "find", fWord: word, fLayer: layer, fK: k, asked: { word, layer, k }, askedAt: send ? Date.now() : null });
  if (send) useTheodoliteStore.getState().executeLarql(nearestStatement(word.trim(), num(layer), num(k)));
}

/** Write a statement into the console, for the reader to run. */
export function toConsole(text: string) {
  useTheodoliteStore.getState().setLarqlQuery(text);
  useVaultUi.getState().set({ tab: "lql", conRun: null, conRunAt: null });
}

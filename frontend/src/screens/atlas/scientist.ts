/**
 * The scientist's chat as the page holds it (after the explorer's Claude tab; backend/scientist_chat.py): one chat at a
 * time, its thread kept in this browser, its events read from one stream (GET /api/atlas/scientist/events). Messages
 * go in any time: mid-reply they reach the scientist between tool calls. Stop ends the reply only. A chat runs on the
 * backend it started with (Claude Code, Codex, or an OpenAI-compatible endpoint and model); the pick applies to the next
 * new chat. A chat whose instructions or tools changed after it started says so (`note` when it was started again on
 * them, `stale` when it runs the old ones until it is restarted: restartChat).
 */
import { create } from "zustand";
import { AtlasError, get, post, streamLines, type Backends, type ChatEvent, type PastChat } from "./api";
import { clearScientistMarks, saveNote, scientistUi, useAtlas } from "./state";

export interface Att {
  name: string;
  id: string | null;
}
export type Part =
  | { t: "md"; text: string }
  | { t: "thinking"; text: string }
  | { t: "tool"; id: string | null; name: string; input: unknown; output?: string; error?: boolean }
  | { t: "error"; text: string }
  | { t: "note"; text: string }
  | { t: "stopped" }
  | { t: "usename"; name: string; layer: number; n: number };
export type Item = { kind: "user"; msg: string; shown: string; atts: Att[] } | { kind: "assistant"; parts: Part[] };
export type Agent = "claude" | "codex" | "openai";
export interface Pick {
  agent: Agent;
  profile: string | null;
  model: string | null;
}
export interface ModelRow {
  id: string;
  name?: string | null;
  tools: boolean | null;
}

interface SciState {
  backends: Backends | null;
  backendsError: string | null;
  pick: Pick;
  chat: string | null;
  chatPick: Pick | null;
  items: Item[];
  box: number | null;
  busy: boolean;
  /** the chat's process runs instructions or tools older than the files' (the server's `stale`) */
  stale: boolean;
  pending: Att[];
  past: PastChat[] | "loading" | null;
  pastError: string | null;
  /** a reply ended while the scientist's tab was not showing */
  unseen: boolean;
  models: Record<string, ModelRow[] | string>;
  input: string;
}

const THREAD_KEY = "theodolite.atlas.scientist";
const PICK_KEY = "theodolite.atlas.scientist.pick";

function loadJson<T>(key: string): T | null {
  try {
    return JSON.parse(localStorage.getItem(key) || "null") as T | null;
  } catch {
    return null;
  }
}

const kept = loadJson<{ chat: string | null; chatPick: Pick | null; items: Item[] }>(THREAD_KEY);
const keptPick = loadJson<Pick>(PICK_KEY);

export const useScientist = create<SciState>(() => ({
  backends: null,
  backendsError: null,
  pick: keptPick && ["claude", "codex", "openai"].includes(keptPick.agent) ? keptPick : { agent: "claude", profile: null, model: null },
  chat: kept?.chat ?? null,
  chatPick: kept?.chatPick ?? null,
  // a reply that was being written when the page closed is replayed by the server into a fresh box
  items: (kept?.items ?? []).filter((x, i, a) => !(x.kind === "assistant" && !x.parts.length && i === a.length - 1)),
  box: null,
  busy: false,
  stale: false,
  pending: [],
  past: null,
  pastError: null,
  unseen: false,
  models: {},
  input: "",
}));

const set = useScientist.setState;
const st = useScientist.getState;
const errText = (e: unknown) => (e instanceof AtlasError ? e.message : String((e as Error)?.message ?? e));

/** The thread as kept in this browser: tool outputs cut, the last 300 entries. */
function save() {
  const { chat, chatPick, items } = st();
  const cut = items.slice(-300).map((x) =>
    x.kind === "assistant"
      ? { ...x, parts: x.parts.map((p) => (p.t === "tool" && p.output && p.output.length > 4000 ? { ...p, output: p.output.slice(0, 4000) + " …" } : p)) }
      : x,
  );
  try {
    localStorage.setItem(THREAD_KEY, JSON.stringify({ chat, chatPick, items: cut }));
  } catch {
    // not kept
  }
}

export function setPick(p: Pick) {
  set({ pick: p });
  try {
    localStorage.setItem(PICK_KEY, JSON.stringify(p));
  } catch {
    // not kept
  }
  if (p.agent === "openai" && p.profile && !st().models[p.profile]) void loadModels(p.profile);
}

export async function loadBackends() {
  try {
    const b = await get<Backends>("/api/atlas/scientist/backends");
    set({ backends: b, backendsError: null });
    const p = st().pick;
    if (p.agent === "openai" && p.profile && b.openai.some((x) => x.name === p.profile)) void loadModels(p.profile);
    else if (p.agent === "openai") setPick({ agent: b.claude.available ? "claude" : "codex", profile: null, model: null });
  } catch (e) {
    set({ backendsError: errText(e) });
  }
}

export async function loadModels(profile: string) {
  try {
    const r = await get<{ models: ModelRow[]; default: string | null }>("/api/atlas/scientist/models", { profile });
    set({ models: { ...st().models, [profile]: r.models } });
    const p = st().pick;
    if (p.profile === profile && !p.model) setPick({ ...p, model: r.default ?? r.models[0]?.id ?? null });
  } catch (e) {
    set({ models: { ...st().models, [profile]: errText(e) } });
  }
}

/* ------------------------------------------------------------------------------------------------- the thread */

function setItems(items: Item[]) {
  set({ items });
}

/** The reply being written: the box under the latest message, made when the first of it arrives. */
function ensureBox(): number {
  const s = st();
  if (s.box !== null && s.items[s.box]?.kind === "assistant") return s.box;
  const last = s.items.length - 1;
  if (last >= 0 && s.items[last].kind === "assistant" && !(s.items[last] as { parts: Part[] }).parts.length) {
    set({ box: last });
    return last;
  }
  setItems([...s.items, { kind: "assistant", parts: [] }]);
  set({ box: s.items.length });
  return s.items.length;
}

function editParts(b: number, fn: (parts: Part[]) => Part[]) {
  const items = st().items.slice();
  const it = items[b];
  if (!it || it.kind !== "assistant") return;
  items[b] = { kind: "assistant", parts: fn(it.parts) };
  setItems(items);
}

function appendText(b: number, t: "md" | "thinking", text: string) {
  editParts(b, (parts) => {
    const last = parts[parts.length - 1];
    if (last && last.t === t) return [...parts.slice(0, -1), { t, text: last.text + text }];
    return [...parts, { t, text }];
  });
}

const pushPart = (b: number, p: Part) => editParts(b, (parts) => [...parts, p]);

let evCtl: AbortController | null = null;
let evChat: string | null = null;
let sending = 0;
let reply = "";
const waiters: ((r: { reply: string; box: number | null }) => void)[] = [];

export async function openEvents() {
  const chat = st().chat;
  if (!chat || (evCtl && evChat === chat)) return;
  evCtl?.abort();
  const ctl = (evCtl = new AbortController());
  evChat = chat;
  while (!ctl.signal.aborted) {
    try {
      const r = await fetch(`/api/atlas/scientist/events?chat=${encodeURIComponent(chat)}`, { signal: ctl.signal });
      if (r.ok && r.body) for await (const ev of streamLines(r, false)) if (evCtl === ctl) onEvent(ev as ChatEvent);
    } catch {
      if (ctl.signal.aborted) return;
    }
    // the server restarted, or the line dropped: its state event settles a turn that ended meanwhile
    await new Promise((res) => window.setTimeout(res, 1500));
  }
}

export function closeEvents() {
  evCtl?.abort();
  evCtl = null;
  evChat = null;
}

function onEvent(e: ChatEvent) {
  if (e.t === "state") {
    set({ stale: e.stale === true });
    if (e.busy) {
      // (re)connected mid-turn: the turn so far comes again, so what was drawn of it is drawn again
      const b = st().box;
      if (b !== null) editParts(b, () => []);
      reply = "";
      set({ busy: true });
      ensureBox();
    } else if (st().busy && !sending) finishTurn();
    return;
  }
  if (e.t === "ping" || e.t === "sent" || e.t === "user") return;
  if (e.t === "stale") return void set({ stale: e.stale === true });
  if (e.t === "ui") return void scientistUi(e).catch(() => undefined);
  if (e.t === "done") return finishTurn();
  if (!st().busy) set({ busy: true }); // a turn the scientist began itself (a background command finished)
  const b = ensureBox();
  if (e.t === "text") {
    appendText(b, "md", String(e.text ?? ""));
    reply += String(e.text ?? "");
  } else if (e.t === "thinking") appendText(b, "thinking", String(e.text ?? ""));
  else if (e.t === "tool") pushPart(b, { t: "tool", id: (e.id as string) ?? null, name: String(e.name ?? ""), input: e.input ?? {} });
  else if (e.t === "tool_result")
    editParts(b, (parts) => parts.map((p) => (p.t === "tool" && p.id === e.id ? { ...p, output: String(e.output ?? ""), error: !!e.error } : p)));
  else if (e.t === "error") pushPart(b, { t: "error", text: String(e.error ?? "") });
  else if (e.t === "note") pushPart(b, { t: "note", text: String(e.text ?? "") });
}

function finishTurn() {
  const r = { reply, box: st().box };
  reply = "";
  const a = useAtlas.getState();
  set({ busy: false, box: null, unseen: st().unseen || !(a.rightOpen && a.rightTab === "scientist") });
  save();
  for (const w of waiters.splice(0)) w(r);
}

function newId(): string {
  if (typeof crypto.randomUUID === "function") return crypto.randomUUID();
  const b = crypto.getRandomValues(new Uint8Array(16));
  b[6] = (b[6] & 0x0f) | 0x40;
  b[8] = (b[8] & 0x3f) | 0x80;
  const h = [...b].map((x) => x.toString(16).padStart(2, "0")).join("");
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
}

/** A message into the chat (begun with the pick when there is none); resolves when its reply ends. */
export async function agentTurn(message: string, shown: string, atts: Att[] = []): Promise<{ reply: string; box: number | null }> {
  if (st().past !== null) set({ past: null });
  clearScientistMarks();
  if (!st().chat) set({ chat: newId(), chatPick: st().pick });
  const f = useAtlas.getState().focus;
  // the reply, or the rest of the one under way, goes under this message
  setItems([...st().items, { kind: "user", msg: message, shown, atts }]);
  set({ box: null, busy: true });
  void openEvents();
  save();
  const out = new Promise<{ reply: string; box: number | null }>((res) => waiters.push(res));
  sending++;
  try {
    const p = st().chatPick ?? st().pick;
    await post("/api/atlas/scientist/send", {
      chat: st().chat,
      agent: p.agent,
      ...(p.agent === "openai" ? { profile: p.profile, model: p.model } : {}),
      message,
      attachments: atts.map((a) => a.id).filter(Boolean),
      ...(f?.kind === "neuron" ? { layer: f.layer, n: f.n } : f?.kind === "head" ? { layer: f.layer, head: f.n } : {}),
    });
  } catch (e) {
    pushPart(ensureBox(), { t: "error", text: errText(e) });
    finishTurn();
  } finally {
    sending--;
  }
  return out;
}

export async function stop() {
  if (!st().busy) return;
  pushPart(ensureBox(), { t: "stopped" });
  let running = false;
  const chat = st().chat;
  if (chat) {
    try {
      running = !!(await post<{ running: boolean }>("/api/atlas/scientist/interrupt", { chat })).running;
    } catch {
      running = false;
    }
  }
  if (!running && st().busy) finishTurn(); // nothing runs on the server: settle here
}

export function send(): boolean {
  const { input, pending } = st();
  const msg = input.trim();
  if (pending.some((a) => !a.id) || (!msg && !pending.length)) return false;
  set({ input: "", pending: [] });
  void agentTurn(msg || "(see the attached files)", msg || "(attached files)", pending);
  return true;
}

export function newChat() {
  closeEvents();
  if (st().busy) finishTurn();
  set({ chat: null, chatPick: null, items: [], box: null, past: null, stale: false });
  clearScientistMarks();
  save();
}

/** Ends the chat's process, and any commands it left running, so that the next message starts it again on its
 * instructions and tools as they now stand (POST /restart, refused while it answers); the server's note goes in the
 * thread. */
export async function restartChat() {
  const chat = st().chat;
  if (!chat || st().busy) return;
  try {
    const r = await post<{ ended: boolean; note?: string }>("/api/atlas/scientist/restart", { chat });
    set({ stale: false });
    if (r.ended && r.note) setItems([...st().items, { kind: "assistant", parts: [{ t: "note", text: r.note }] }]);
  } catch (e) {
    setItems([...st().items, { kind: "assistant", parts: [{ t: "error", text: errText(e) }] }]);
  }
  save();
}

/* ------------------------------------------------------------------------------------------------ past chats */

export async function togglePast() {
  if (st().past !== null) return set({ past: null });
  set({ past: "loading", pastError: null });
  try {
    const list = await get<PastChat[]>("/api/atlas/scientist/chats");
    if (st().past === "loading") set({ past: list });
  } catch (e) {
    set({ past: null, pastError: errText(e) });
  }
}

export async function openPast(c: PastChat) {
  let r: { events: ChatEvent[]; agent: Agent; profile?: string; model?: string };
  try {
    r = await get("/api/atlas/scientist/history", { chat: c.chat });
  } catch (e) {
    return set({ pastError: errText(e) });
  }
  closeEvents();
  if (st().busy) finishTurn();
  const items: Item[] = [];
  let box: { kind: "assistant"; parts: Part[] } | null = null;
  for (const e of r.events ?? []) {
    if (e.t === "user") {
      items.push({ kind: "user", msg: String(e.text ?? ""), shown: String(e.text ?? ""), atts: [] });
      box = null;
      continue;
    }
    if (!box) items.push((box = { kind: "assistant", parts: [] }));
    const last = box.parts[box.parts.length - 1];
    if (e.t === "text") {
      if (last && last.t === "md") last.text += "\n\n" + String(e.text ?? "");
      else box.parts.push({ t: "md", text: String(e.text ?? "") });
    } else if (e.t === "tool") box.parts.push({ t: "tool", id: (e.id as string) ?? null, name: String(e.name ?? ""), input: e.input ?? {} });
    else if (e.t === "tool_result") {
      const p = box.parts.find((x) => x.t === "tool" && x.id === e.id);
      if (p && p.t === "tool") Object.assign(p, { output: String(e.output ?? ""), error: !!e.error });
    } else if (e.t === "error") box.parts.push({ t: "error", text: String(e.error ?? "") });
  }
  const chatPick: Pick = { agent: r.agent ?? c.agent, profile: r.profile ?? null, model: r.model ?? null };
  set({ chat: c.chat, chatPick, items, box: null, past: null, stale: false }); // its stream's state says whether it is stale
  clearScientistMarks();
  save();
  void openEvents();
}

/* --------------------------------------------------------------------------------------------- edit and reroll */

export async function rewindLast(act: "edit" | "reroll") {
  const items = st().items;
  let u = -1;
  for (let i = items.length - 1; i >= 0; i--)
    if (items[i].kind === "user") {
      u = i;
      break;
    }
  if (u < 0) return;
  const it = items[u] as Extract<Item, { kind: "user" }>;
  if (st().busy) await stop();
  const chat = st().chat;
  if (chat) {
    try {
      await post("/api/atlas/scientist/rewind", { chat, text: it.msg });
    } catch (e) {
      // still answering, or the chat is gone: say so and keep the thread
      return pushPart(ensureBox(), { t: "error", text: errText(e) });
    }
  }
  setItems(items.slice(0, u));
  save();
  if (act === "reroll") void agentTurn(it.msg, it.shown, it.atts);
  else set({ input: it.shown, pending: it.atts });
}

/* ------------------------------------------------------------------------------------------------ attachments */

export async function attachFiles(files: File[]) {
  for (const file of files) {
    const a: Att = { name: file.name || "pasted", id: null };
    set({ pending: [...st().pending, a] });
    try {
      const data = await new Promise<string>((ok, bad) => {
        const r = new FileReader();
        r.onload = () => ok(String(r.result).split(",")[1] || "");
        r.onerror = () => bad(r.error);
        r.readAsDataURL(file);
      });
      const res = await post<{ id: string; name: string }>("/api/atlas/scientist/upload", { name: a.name, type: file.type, data });
      set({ pending: st().pending.map((x) => (x === a ? { ...a, id: res.id } : x)) });
    } catch (e) {
      set({ pending: st().pending.filter((x) => x !== a) });
      setItems([...st().items, { kind: "assistant", parts: [{ t: "error", text: `${a.name}: ${errText(e)}` }] }]);
    }
  }
}

export function removePending(i: number) {
  set({ pending: st().pending.filter((_, j) => j !== i) });
}

/* -------------------------------------------------------------------------------------------- the quick asks */

export async function explainFocus() {
  const f = useAtlas.getState().focus;
  if (!f || st().busy) return;
  const at = `L${f.layer + 1} ${f.kind === "head" ? "h" : "N"}${f.n}`;
  if (f.kind === "head") return void agentTurn(`Explain attention head ${at}: write the head report.`, `explain ${at}`);
  const turn = agentTurn(`Explain ${at}: write the neuron report.`, `explain ${at}`);
  const chat = st().chat;
  const { reply: text, box } = await turn;
  if (st().chat !== chat) return; // another chat was opened meanwhile
  const name = text
    .match(/\*\*Name\*\*:?\s*(.+)/)?.[1]
    .replace(/[*`]/g, "")
    .trim();
  if (name && box !== null) pushPart(box, { t: "usename", name, layer: f.layer, n: f.n });
  save();
}

export function askWeb() {
  const f = useAtlas.getState().focus;
  if (!f || f.kind !== "neuron" || st().busy) return;
  const at = `L${f.layer + 1} N${f.n}`;
  void agentTurn(
    `Look at the cone of ${at} (what it reads from and writes into, a few hops out) and tell me briefly what that circuit seems to do. Highlight the neurons you mention.`,
    `what does the web of ${at} do?`,
  );
}

/** "use name": the name the report gave, kept in the field notes as the scientist's (its guess, which the reader takes
 * up; typing over it makes it the reader's own). */
export async function takeName(p: Extract<Part, { t: "usename" }>, b: number) {
  await saveNote(p.layer, p.n, { name: p.name }, "scientist");
  editParts(b, (parts) => parts.filter((x) => x !== p));
  save();
}

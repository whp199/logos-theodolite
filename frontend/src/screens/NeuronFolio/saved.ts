import { useCallback, useEffect, useRef, useState } from "react";

/**
 * A neuron's folio state: the name you give it and your field notes, one record per face.
 * Kept on the server whenever it answers for the neuron (backend/field_notes.py): the name as the atlas's name for it
 * (field_notes.json, written by "reader", the same name the atlas's map and lists show), the notes per face in the field
 * book beside it (field_book.json, GET and POST /api/atlas/fieldbook), shared by every page and chat this server
 * answers. With no server (the recorded sighting on a page served alone), an older one without a field book, or a
 * neuron the server's model does not have, they are kept in this browser's localStorage under
 * `theodolite.folio.v1.${layerNo}:${neuron}`, as before; every read and write is wrapped so the page works when storage
 * is blocked (the notes then last for this visit). Which of the two keeps them is said beside them (`FolioWhere`).
 * `folioMigration` turns this browser's keys into the requests that carry them to the server, which the folio offers
 * once for each key.
 */

export type Jobs = "one" | "two" | "unknown";

export interface FaceNotes {
  jobs: Jobs | null;
  pattern: string;
  other: boolean;
  long: boolean;
  steered: boolean;
  readers: boolean;
}

export interface FolioSaved {
  nick: string | null;
  notes: { pos: FaceNotes; neg: FaceNotes };
}

export const blankNotes = (): FaceNotes => ({
  jobs: null,
  pattern: "",
  other: false,
  long: false,
  steered: false,
  readers: false,
});

const blank = (): FolioSaved => ({ nick: null, notes: { pos: blankNotes(), neg: blankNotes() } });

export const FOLIO_PREFIX = "theodolite.folio.v1.";
export const folioKey = (layerNo: number, neuron: number) => `${FOLIO_PREFIX}${layerNo}:${neuron}`;
/** the keys this browser has carried to the server, or was asked about once and kept */
export const CARRIED_KEY = "theodolite.folio.carried.v1";
/** the most the atlas keeps of a name (backend/field_notes.py MAX_NAME) */
export const MAX_NAME = 120;

function cleanNotes(v: unknown): FaceNotes {
  const o = (v && typeof v === "object" ? v : {}) as Partial<FaceNotes>;
  const jobs = o.jobs === "one" || o.jobs === "two" || o.jobs === "unknown" ? o.jobs : null;
  return {
    jobs,
    pattern: typeof o.pattern === "string" ? o.pattern : "",
    other: o.other === true,
    long: o.long === true,
    steered: o.steered === true,
    readers: o.readers === true,
  };
}

/** A stored value as the folio reads it (anything unreadable is a blank folio). */
function parseSaved(raw: string | null): FolioSaved {
  if (!raw) return blank();
  try {
    const v = JSON.parse(raw) as Partial<FolioSaved> & { notes?: { pos?: unknown; neg?: unknown } };
    return {
      nick: typeof v.nick === "string" && v.nick.trim() ? v.nick : null,
      notes: { pos: cleanNotes(v.notes?.pos), neg: cleanNotes(v.notes?.neg) },
    };
  } catch {
    return blank();
  }
}

function load(key: string): FolioSaved {
  try {
    return parseSaved(window.localStorage.getItem(key));
  } catch {
    return blank();
  }
}

function save(key: string, v: FolioSaved) {
  try {
    window.localStorage.setItem(key, JSON.stringify(v));
  } catch {
    // storage blocked or full: the notes simply last for this visit
  }
}

export const isBlank = (f: FaceNotes) => f.jobs === null && !f.pattern && !f.other && !f.long && !f.steered && !f.readers;

/* ------------------------------------------------------------------------------------------ the server's field book */

/** One face as the field book keeps it (backend/field_notes.py). */
export interface BookFace {
  jobs: Jobs | null;
  pattern: string;
  checked: { other_pole: boolean; long_contexts: boolean; steered: boolean; readers_traced: boolean };
  by: "reader" | "scientist";
  at: string;
}
export interface BookNote {
  name: string;
  note: string;
  by: "reader" | "scientist";
  name_by?: "reader" | "scientist";
  note_by?: "reader" | "scientist";
  at: string;
}
/** GET /api/atlas/fieldbook?layer&n (layers from 0). */
export interface BookReply {
  layer: number;
  n: number;
  faces: Record<"+" | "-", BookFace | null>;
  notes: BookNote | null;
  file: string;
  notes_file: string;
  warning: string | null;
  notes_warning?: string | null;
}

const fromBook = (f: BookFace | null | undefined): FaceNotes =>
  f
    ? {
        jobs: f.jobs,
        pattern: f.pattern,
        other: f.checked.other_pole,
        long: f.checked.long_contexts,
        steered: f.checked.steered,
        readers: f.checked.readers_traced,
      }
    : blankNotes();

/** The request that keeps one face in the field book, written by the reader. */
export const faceBody = (layer: number, n: number, face: "+" | "-", f: FaceNotes) => ({
  layer,
  n,
  face,
  jobs: f.jobs,
  pattern: f.pattern,
  checked: { other_pole: f.other, long_contexts: f.long, steered: f.steered, readers_traced: f.readers },
  by: "reader",
});

/** Where the folio's name and notes are kept for this neuron, said beside them. */
export type FolioWhere =
  | { kind: "reading" }
  | {
      kind: "server";
      file: string;
      notesFile: string;
      /** who wrote the name the atlas keeps */
      nameBy: "reader" | "scientist" | null;
      /** the atlas's own note on the neuron, and who wrote it */
      note: { text: string; by: "reader" | "scientist"; at: string } | null;
      /** the last write that did not come back, or a moved-aside file's warning */
      error: string | null;
    }
  | { kind: "browser"; why: string; detail?: string };

async function readBook(layer: number, n: number): Promise<{ ok: true; reply: BookReply } | { ok: false; why: string; detail?: string }> {
  let r: Response;
  try {
    r = await fetch(`/api/atlas/fieldbook?layer=${layer}&n=${n}`);
  } catch {
    return { ok: false, why: "the server is not answering" };
  }
  if (r.status === 404) return { ok: false, why: "this server keeps no field book (one from before it)" };
  if (!r.ok) {
    let detail = `HTTP ${r.status}`;
    try {
      const d = (await r.json())?.detail;
      if (typeof d === "string") detail = d;
    } catch {
      // not JSON
    }
    return { ok: false, why: r.status === 400 ? "the server’s model has no such neuron" : "the server did not answer for it", detail };
  }
  try {
    return { ok: true, reply: (await r.json()) as BookReply };
  } catch {
    return { ok: false, why: "the server’s reply could not be read" };
  }
}

async function postJson(path: string, body: unknown): Promise<unknown> {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) {
    let detail = `HTTP ${r.status}`;
    try {
      const d = (await r.json())?.detail;
      if (typeof d === "string") detail = d;
    } catch {
      // not JSON
    }
    throw new Error(detail);
  }
  return r.json();
}

/**
 * The folio's name and notes for one neuron (`layerNo` counted 1-60), and where they are kept. Through the server when
 * it answers for the neuron: the name goes to POST /api/atlas/notes when kept, each face to POST /api/atlas/fieldbook a
 * moment after its last change (one write at a time per face, in order). Else in this browser, as before. `reload`
 * reads the server again (after notes are carried to it).
 */
export function useFolioSaved(layerNo: number, neuron: number) {
  const key = folioKey(layerNo, neuron);
  const layer = layerNo - 1;
  const [saved, setSaved] = useState<FolioSaved>(() => load(key));
  const [where, setWhere] = useState<FolioWhere>({ kind: "reading" });
  const [gen, setGen] = useState(0);
  const whereRef = useRef<FolioWhere>(where);
  whereRef.current = where;
  const keyRef = useRef(key);
  const timers = useRef<Partial<Record<"+" | "-", number>>>({});
  const chains = useRef<Record<"+" | "-", Promise<unknown>>>({ "+": Promise.resolve(), "-": Promise.resolve() });
  const latest = useRef<FolioSaved>(saved);
  latest.current = saved;

  useEffect(() => {
    let live = true;
    if (keyRef.current !== key) {
      keyRef.current = key;
      setSaved(load(key));
    }
    setWhere({ kind: "reading" });
    void readBook(layer, neuron).then((got) => {
      if (!live) return;
      if (!got.ok) {
        setSaved(load(key));
        setWhere({ kind: "browser", why: got.why, detail: got.detail });
        return;
      }
      const b = got.reply;
      const nameBy = b.notes?.name ? (b.notes.name_by ?? b.notes.by) : null;
      setSaved({ nick: b.notes?.name ? b.notes.name : null, notes: { pos: fromBook(b.faces["+"]), neg: fromBook(b.faces["-"]) } });
      setWhere({
        kind: "server",
        file: b.file,
        notesFile: b.notes_file,
        nameBy,
        note: b.notes?.note ? { text: b.notes.note, by: b.notes.note_by ?? b.notes.by, at: b.notes.at } : null,
        error: b.warning ?? b.notes_warning ?? null,
      });
    });
    return () => {
      live = false;
    };
  }, [key, gen]); // eslint-disable-line react-hooks/exhaustive-deps

  const fail = (e: unknown) =>
    setWhere((w) => (w.kind === "server" ? { ...w, error: `not kept on the server: ${(e as Error).message}` } : w));

  const writeFace = useCallback(
    (face: "+" | "-") => {
      const f = face === "+" ? latest.current.notes.pos : latest.current.notes.neg;
      const body = faceBody(layer, neuron, face, f);
      chains.current[face] = chains.current[face]
        .then(() => postJson("/api/atlas/fieldbook", body))
        .then(() => setWhere((w) => (w.kind === "server" && w.error?.startsWith("not kept") ? { ...w, error: null } : w)))
        .catch(fail);
    },
    [layer, neuron], // eslint-disable-line react-hooks/exhaustive-deps
  );

  // a face changed a moment ago is written before the folio closes
  useEffect(
    () => () => {
      for (const face of ["+", "-"] as const) {
        if (timers.current[face] !== undefined) {
          window.clearTimeout(timers.current[face]);
          timers.current[face] = undefined;
          writeFace(face);
        }
      }
    },
    [writeFace],
  );

  const update = useCallback(
    (fn: (prev: FolioSaved) => FolioSaved) => {
      const prev = latest.current;
      const next = fn(prev);
      latest.current = next;
      setSaved(next);
      const w = whereRef.current;
      if (w.kind !== "server") {
        save(key, next);
        return;
      }
      if ((next.nick ?? "") !== (prev.nick ?? "")) {
        void postJson("/api/atlas/notes", { kind: "neuron", layer, index: neuron, name: (next.nick ?? "").trim(), by: "reader" })
          .then(() => setWhere((x) => (x.kind === "server" ? { ...x, nameBy: next.nick ? "reader" : null } : x)))
          .catch(fail);
      }
      for (const [face, side] of [
        ["+", "pos"],
        ["-", "neg"],
      ] as const) {
        if (JSON.stringify(next.notes[side]) === JSON.stringify(prev.notes[side])) continue;
        window.clearTimeout(timers.current[face]);
        timers.current[face] = window.setTimeout(() => {
          timers.current[face] = undefined;
          writeFace(face);
        }, 600);
      }
    },
    [key, layer, neuron, writeFace], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const reload = useCallback(() => setGen((g) => g + 1), []);
  return [saved, update, where, reload] as const;
}

/* ---------------------------------------------------------------------------------- carrying this browser's notes */

export interface MigrationPost {
  path: "/api/atlas/notes" | "/api/atlas/fieldbook";
  body: Record<string, unknown>;
}
export interface Migration {
  /** the requests that carry them, the names first (layers counted from 0, as the server counts them) */
  posts: MigrationPost[];
  /** the keys with something carried by `posts` */
  keys: string[];
  /** what is left in this browser, and why */
  left: { key: string; what: string; why: string }[];
}
/** What the server already keeps, by "layer:n" (layers from 0): a name, and the faces said. */
export interface OnServer {
  names?: Record<string, { name?: string } | undefined>;
  faces?: Record<string, Partial<Record<"+" | "-", unknown>> | undefined>;
}

const KEY_RE = /^theodolite\.folio\.v1\.(\d+):(\d+)$/;

/**
 * This browser's folio keys (`[key, stored value]`, as localStorage holds them) as the requests that keep them on the
 * server: each name (trimmed) to POST /api/atlas/notes by the reader, each face with anything said to POST
 * /api/atlas/fieldbook, at layer = layerNo − 1. Nothing the server already keeps is written over: a name or a face it
 * holds stays, and this browser's is left here, as is a name longer than the atlas keeps or holding a control
 * character. A pure function, checked by scripts/check_interface_defaults.py.
 */
export function folioMigration(entries: readonly (readonly [string, string | null])[], server: OnServer = {}): Migration {
  const posts: MigrationPost[] = [];
  const faces: MigrationPost[] = [];
  const keys = new Set<string>();
  const left: Migration["left"] = [];
  for (const [key, raw] of entries) {
    const m = KEY_RE.exec(key);
    if (!m) continue;
    const layerNo = parseInt(m[1], 10);
    const n = parseInt(m[2], 10);
    if (layerNo < 1) {
      left.push({ key, what: "all of it", why: "its layer is not one of the pages’ (they count from 1)" });
      continue;
    }
    const layer = layerNo - 1;
    const at = `${layer}:${n}`;
    const v = parseSaved(raw);
    const nick = (v.nick ?? "").trim();
    if (nick) {
      // eslint-disable-next-line no-control-regex
      if (/[\u0000-\u001f\u007f-\u009f]/.test(nick)) left.push({ key, what: "its name", why: "it holds a control character, which a name cannot" });
      else if (nick.length > MAX_NAME) left.push({ key, what: "its name", why: `it is longer than the ${MAX_NAME} characters the atlas keeps` });
      else if (server.names?.[at]?.name) left.push({ key, what: "its name", why: "the server already names it" });
      else {
        posts.push({ path: "/api/atlas/notes", body: { kind: "neuron", layer, index: n, name: nick, by: "reader" } });
        keys.add(key);
      }
    }
    for (const [face, side] of [
      ["+", "pos"],
      ["-", "neg"],
    ] as const) {
      const f = v.notes[side];
      if (isBlank(f)) continue;
      if (server.faces?.[at]?.[face]) {
        left.push({ key, what: `its ${face === "+" ? "positive" : "negative"} face’s notes`, why: "the server already keeps notes on that face" });
        continue;
      }
      faces.push({ path: "/api/atlas/fieldbook", body: faceBody(layer, n, face, f) });
      keys.add(key);
    }
  }
  return { posts: [...posts, ...faces], keys: [...keys], left };
}

/** This browser's folio keys and their values; empty when storage is blocked. */
export function folioEntries(): [string, string | null][] {
  const out: [string, string | null][] = [];
  try {
    const ls = window.localStorage;
    for (let i = 0; i < ls.length; i++) {
      const k = ls.key(i);
      if (k && k.startsWith(FOLIO_PREFIX)) out.push([k, ls.getItem(k)]);
    }
  } catch {
    // storage blocked: nothing to carry
  }
  return out;
}

/** The keys carried, or asked about once and kept here. */
export function carriedKeys(): Set<string> {
  try {
    const v = JSON.parse(window.localStorage.getItem(CARRIED_KEY) || "[]");
    return new Set(Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : []);
  } catch {
    return new Set();
  }
}

export function markCarried(keys: Iterable<string>) {
  try {
    const all = carriedKeys();
    for (const k of keys) all.add(k);
    window.localStorage.setItem(CARRIED_KEY, JSON.stringify([...all]));
  } catch {
    // not remembered: the folio offers again next time
  }
}

/** Send the migration's requests one at a time; the keys whose every request was kept, and what failed. */
export async function carryToServer(mig: Migration): Promise<{ kept: string[]; failed: { what: string; why: string }[] }> {
  const failedAt = new Set<string>();
  const failed: { what: string; why: string }[] = [];
  for (const p of mig.posts) {
    const b = p.body as { layer: number; index?: number; n?: number; face?: string };
    const at = `${b.layer}:${b.index ?? b.n}`;
    try {
      await postJson(p.path, p.body);
    } catch (e) {
      failedAt.add(at);
      failed.push({ what: `L${b.layer + 1} N${b.index ?? b.n}${b.face ? ` ${b.face}` : " name"}`, why: (e as Error).message });
    }
  }
  const kept = mig.keys.filter((k) => {
    const m = KEY_RE.exec(k);
    return m ? !failedAt.has(`${parseInt(m[1], 10) - 1}:${m[2]}`) : false;
  });
  return { kept, failed };
}

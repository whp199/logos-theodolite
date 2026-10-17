/**
 * Reading the vault's live replies out of the store's log (useTheodoliteStore().larqlLog, newest first).
 * Shapes: backend/larql_vault.py (query_nearest_edges, apply_patch_surgery, rollback_patches, describe_vault).
 */
import type { ActiveEdit, LarqlEntry } from "../../store/useTheodoliteStore";
import { clampL, clampN, parseLql, type Parsed } from "./lql";

export interface TokenLogit {
  token: string;
  logit: number;
}
export interface NearestRow {
  rank: number;
  layer: number;
  neuron_idx: number;
  polarity: string;
  polarity_num: number;
  cosine_similarity: number;
  j_vindex_top_tokens: TokenLogit[];
  naive_lens_top_tokens: TokenLogit[];
}
export interface NearestReply {
  command: string;
  concept: string;
  resolved_token_id: number;
  resolved_token_str: string;
  layer: number;
  matrix: string;
  use_jpp: boolean;
  elapsed_ms: number;
  results: NearestRow[];
}

/** canvas num(): parseInt of the trimmed text, or null */
export const num = (s: string): number | null => {
  const n = parseInt(String(s).trim(), 10);
  return isNaN(n) ? null : n;
};

export function asNearest(data: Record<string, unknown> | null | undefined): NearestReply | null {
  if (!data || !Array.isArray((data as { results?: unknown }).results)) return null;
  const d = data as unknown as NearestReply;
  if (typeof d.concept !== "string" || typeof d.layer !== "number") return null;
  return d;
}

/** The LIMIT the server ran, from the command it echoes. */
export function limitOf(d: NearestReply): number {
  const m = /LIMIT\s+(\d+)\s*$/i.exec(String(d.command ?? ""));
  return m ? parseInt(m[1], 10) : d.results.length;
}

/**
 * The newest log entry that answers the nearness search (word, repo index, k). A reply is matched by what it
 * carries (concept, layer and limit), not by the query it is filed under; a refusal by the statement sent.
 */
export function findNearestEntry(log: LarqlEntry[], word: string, idx: number, k: number): LarqlEntry | null {
  for (const e of log) {
    if (e.ok) {
      const d = asNearest(e.data);
      if (d && d.concept === word && d.layer === idx && limitOf(d) === k) return e;
    } else {
      const p = parseLql(e.query);
      if (
        p.kind === "select" &&
        p.concept === word &&
        (p.layerS === null ? 35 : clampL(p.layerS)) === idx &&
        (p.limitS === null ? 10 : parseInt(p.limitS, 10)) === k
      )
        return e;
    }
  }
  return null;
}

/** Whether a reply has the shape the statement's kind should get (guards against a reply filed under the wrong query). */
export function replyFits(kind: Parsed["kind"], e: LarqlEntry): boolean {
  if (!e.ok) return true;
  const d = (e.data ?? {}) as Record<string, unknown>;
  switch (kind) {
    case "describe":
      return d.command === "DESCRIBE";
    case "rollback":
      return d.status === "rolled_back";
    case "select":
      return asNearest(e.data) !== null;
    case "pole":
      return d.north_pole !== undefined;
    case "delete":
    case "update":
    case "insert":
      return d.status === "applied";
    case "error":
      // the server refuses what does not parse; an answer filed under it belongs to another statement
      return false;
  }
}

/** The newest reply, filed under exactly this statement and no older than `since`, whose shape fits it. */
export function replyTo(log: LarqlEntry[], statement: string, since: number | null): LarqlEntry | null {
  const q = statement.trim();
  const kind = parseLql(q).kind;
  for (const e of log) {
    if (since !== null && e.at < since) break;
    if (e.query === q && replyFits(kind, e)) return e;
  }
  return null;
}

export const clock = (at: number) =>
  new Date(at).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" });

export const prettyJson = (v: unknown) => JSON.stringify(v, null, 2);

/** One line on what a logged reply said, in plain words. */
export function entrySummary(e: LarqlEntry): string {
  if (!e.ok) return "refused or unanswered: " + (e.error ?? "no reason given");
  const d = (e.data ?? {}) as Record<string, unknown>;
  const near = asNearest(e.data);
  if (near) {
    const n = near.results.length;
    return n + (n === 1 ? " neuron" : " neurons") + " nearest “" + near.concept + "” in layer " + (near.layer + 1);
  }
  if (d.command === "DESCRIBE") {
    const c = Number(d.active_patches_count ?? 0);
    return "the vault described itself · " + (c === 0 ? "no edits in force" : c + (c === 1 ? " edit in force" : " edits in force"));
  }
  if (d.status === "applied") {
    const p = (d.patch ?? {}) as Partial<ActiveEdit>;
    return "edit made · patch " + (p.patch_id ?? "?") + " · neuron " + (p.neuron_idx ?? "?") + " of layer " + (Number(p.layer_idx ?? NaN) + 1);
  }
  if (d.status === "rolled_back") {
    const c = Number(d.cleared_patches ?? 0);
    return "every edit undone · " + c + " cleared";
  }
  if (d.north_pole !== undefined) return "both faces of neuron " + String(d.neuron_idx) + " of layer " + (Number(d.layer) + 1);
  return "answered";
}

/** An edit in force, in the board's words (built from the statement the vault kept, else from its record). */
export function editDescription(e: ActiveEdit): string {
  const p = parseLql(e.statement ?? "");
  const lN = e.layer_idx + 1;
  const face = (pol: number | undefined, fallback: "+" | "-") =>
    (pol !== undefined ? pol >= 0 : fallback !== "-") ? "positive" : "negative";
  const op = String(e.op_type).toUpperCase();
  if (op === "DELETE") return "neuron " + e.neuron_idx + " of layer " + lN + " silenced";
  if (op === "UPDATE") {
    const strength = e.strength ?? (p.kind === "update" ? (p.strength === null ? 2.5 : Number(p.strength)) : undefined);
    return (
      "neuron " +
      e.neuron_idx +
      " of layer " +
      lN +
      " says “" +
      e.target_concept +
      "” more strongly, on its " +
      face(e.polarity, p.kind === "update" ? p.pol : "+") +
      " face" +
      (strength !== undefined ? ", at strength " + strength : "")
    );
  }
  if (op === "INSERT") {
    const subj = p.kind === "insert" ? p.subj : null;
    const strength = e.strength ?? (p.kind === "insert" ? (p.strength === null ? 2.5 : Number(p.strength)) : undefined);
    return (
      "neuron " +
      e.neuron_idx +
      " of layer " +
      lN +
      (subj ? ", chosen by the vault as nearest “" + subj + "”," : ", chosen by the vault,") +
      " says “" +
      e.target_concept +
      "” more strongly" +
      (strength !== undefined ? ", at strength " + strength : "")
    );
  }
  return op + " at neuron " + clampN(String(e.neuron_idx)) + " of layer " + lN;
}

/* ------------------------------------------------------------- word-pieces */

/** Lens and naked-eye words as the board shows them: trimmed, de-duplicated, emoji and byte fragments omitted. */
export function cleanWords(tokens: string[]): string[] {
  const out: string[] = [];
  for (const raw of tokens) {
    const t = raw.trim();
    if (!t) continue;
    if (/[\uD800-\uDFFF�]/.test(t) || /\p{Extended_Pictographic}/u.test(t)) continue;
    if (!out.includes(t)) out.push(t);
  }
  return out;
}

/** How many of the five word-pieces the lens and the naked eye share (exact word-pieces, as the study counted). */
export function agreement(lens: string[], eye: string[]): number {
  const a = new Set(lens.slice(0, 5));
  let n = 0;
  for (const t of new Set(eye.slice(0, 5))) if (a.has(t)) n++;
  return n;
}

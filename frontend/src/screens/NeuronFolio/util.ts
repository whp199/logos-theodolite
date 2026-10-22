import { bare, gloss, minus } from "../../lib/words";
import { sharedShown } from "../../lib/labels";
import { TOKEN_GLOSS } from "./recorded";
import type { Tok } from "./replies";

/** "a", "a and b", "a, b and c" */
export function andList(xs: string[]): string {
  if (xs.length <= 1) return xs.join("");
  if (xs.length === 2) return `${xs[0]} and ${xs[1]}`;
  return `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`;
}

export const cap = (s: string) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);

/** The canvas's sentence counter: "no sentences yet where it should fire", "one sentence …", "3 sentences …". */
export const count = (n: number, side: string) =>
  `${n === 0 ? "no sentences yet" : n === 1 ? "one sentence" : `${n} sentences`} ${side}`;

/** A signed value at full recorded precision, with a true minus sign. */
export const signedRaw = (v: number) => (v >= 0 ? "+" : "−") + String(Math.abs(v));

/** A gloss for a word-piece: the canvas's hand-written translation, else the generic one. */
export const glossFor = (token: string) => TOKEN_GLOSS[token] ?? gloss(token);

/** A word-piece shortened for the dial (leading space dropped). */
export function shortToken(t: string): string {
  const b = bare(t);
  return b.length > 12 ? `${b.slice(0, 11)}…` : b;
}

/** Raw scores as the apparatus prints them: “ Am” 27.0627 · … (a leading space marks a word start). */
export function scoresText(toks: Tok[], digits?: number): string {
  return toks
    .map((t) => `“${t.token}”${t.logit === null ? "" : ` ${minus(digits !== undefined ? t.logit.toFixed(digits) : String(t.logit))}`}`)
    .join(" · ");
}

export interface Agreement {
  /** how many of the lens's first n word-pieces also stand in the naked eye's first n, spelt exactly */
  k: number;
  verdict: "agree" | "partly agree" | "disagree";
  /** the shared word-pieces, raw (as the lens gave them) */
  shared: string[];
  sharedRaw: string[];
}

/**
 * The word-pieces a face's two readings share among the first n shown (lib/labels.ts sharedShown: exact word-pieces,
 * so " Am" and "Am" are two), for the ◆ marks and the seal's hint, worded "among the words shown". It is not the label
 * gate's measure, which counts word-piece ids ten a side (the server's label_check.shared_top10).
 */
export function agreement(lens: Tok[], eye: Tok[], n = 3): Agreement {
  const L = lens.slice(0, n).map((t) => t.token);
  const E = eye.slice(0, n).map((t) => t.token);
  const { k, shared } = sharedShown(L, E);
  const same = k === new Set(L).size && k === new Set(E).size;
  return { k, verdict: k === 0 ? "disagree" : same ? "agree" : "partly agree", shared, sharedRaw: shared };
}

/**
 * Plain words for the layer leaf, built from a sighting's readings. With the recorded demonstration
 * sighting at layer 36 these reproduce the canvas's copy; at other layers they say the same things in
 * the same voice from the data.
 */
import { comma, gloss } from "../../lib/words";
import { PICTOGRAPHS } from "./recorded";

const ONES = [
  "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
  "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
];
const TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

/** 36 → "thirty-six" (0-99; digits beyond). */
export function cardinal(n: number): string {
  if (!Number.isInteger(n) || n < 0 || n > 99) return comma(n);
  if (n < 20) return ONES[n];
  const t = Math.floor(n / 10);
  const o = n % 10;
  return o ? `${TENS[t]}-${ONES[o]}` : TENS[t];
}

export const capital = (s: string) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);

/** 36 → "XXXVI". */
export function roman(n: number): string {
  const map: [number, string][] = [
    [1000, "M"], [900, "CM"], [500, "D"], [400, "CD"], [100, "C"], [90, "XC"],
    [50, "L"], [40, "XL"], [10, "X"], [9, "IX"], [5, "V"], [4, "IV"], [1, "I"],
  ];
  let out = "";
  let v = Math.round(n);
  for (const [k, s] of map) {
    while (v >= k) {
      out += s;
      v -= k;
    }
  }
  return out;
}

/** 9612 → "9,612th" (the canvas's ord()). */
export function ordinal(n: number): string {
  const h = n % 100;
  const t = n % 10;
  const suf = h >= 11 && h <= 13 ? "th" : t === 1 ? "st" : t === 2 ? "nd" : t === 3 ? "rd" : "th";
  return comma(n) + suf;
}

export const ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"];

/** ["a","b","c"] → "a, b and c". */
export function listAnd(items: string[]): string {
  if (items.length <= 1) return items.join("");
  return `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
}

/** A probability as the canvas's leaf prints it: one decimal, always. */
export const pct1 = (p: number) => `${(Math.round(p * 1000) / 10).toFixed(1)}%`;

/** How far up the sixty layers: 36 → "Three-fifths of the way up". */
export function wayUp(layerNo: number): string {
  if (layerNo <= 1) return "At the first layer";
  if (layerNo >= 60) return "At the last layer";
  const f = layerNo / 60;
  if (f < 0.07) return "Near the bottom of the stack";
  if (f > 0.95) return "Near the top of the stack";
  const marks: [number, string][] = [
    [1 / 10, "a tenth"], [1 / 5, "a fifth"], [1 / 4, "a quarter"], [1 / 3, "a third"], [2 / 5, "two-fifths"],
    [1 / 2, "half"], [3 / 5, "three-fifths"], [2 / 3, "two-thirds"], [3 / 4, "three-quarters"],
    [4 / 5, "four-fifths"], [9 / 10, "nine-tenths"],
  ];
  let best = marks[0];
  for (const m of marks) if (Math.abs(m[0] - f) < Math.abs(best[0] - f)) best = m;
  const exact = Math.abs(best[0] - f) < 0.005;
  if (best[1] === "half") return exact ? "Halfway up" : "About halfway up";
  return exact ? `${capital(best[1])} of the way up` : `About ${best[1]} of the way up`;
}

/* --------------------------------------------------------------- pictographs */

export const isPicto = (token: string) => {
  const t = token.trim();
  return t !== "" && /^[\p{Extended_Pictographic}‍️]+$/u.test(t);
};

const codepoints = (token: string) =>
  Array.from(token.trim())
    .map((ch) => ch.codePointAt(0) ?? 0)
    .filter((cp) => cp !== 0x200d && cp !== 0xfe0f);

/** "U+1F916" */
export const uPlus = (token: string) =>
  codepoints(token)
    .map((cp) => `U+${cp.toString(16).toUpperCase()}`)
    .join(" ");

/** "a pictograph of a robot’s face", or "a pictograph (U+…)" when the canvas never named it. */
export function pictoWords(token: string): string {
  const cps = codepoints(token);
  const name = cps.length === 1 ? PICTOGRAPHS[cps[0]] : undefined;
  return name ? `a pictograph of ${name}` : `a pictograph (${uPlus(token)})`;
}

/** "U+1F916, a robot’s face" for the raw values line. */
export function pictoRaw(token: string): string {
  const cps = codepoints(token);
  const name = cps.length === 1 ? PICTOGRAPHS[cps[0]] : undefined;
  return name ? `${uPlus(token)}, ${name}` : `${uPlus(token)}, a pictograph`;
}

/* ------------------------------------------------------------------ glosses */

/** A line of glosses for three word-pieces, from lib/words gloss(). "" when they read plainly. */
export function glossLine(tokens: string[]): string {
  const parts = tokens.map((t, i) => [i, gloss(t)] as const).filter(([, g]) => g !== "");
  if (parts.length === 0) return "";
  const distinct = new Set(parts.map(([, g]) => g));
  // three different glosses, none of them a remark about joining: list them, as the canvas does
  if (parts.length === tokens.length && distinct.size === parts.length && parts.every(([, g]) => !g.startsWith("joined"))) {
    return parts.map(([, g]) => g).join(", ");
  }
  const groups = new Map<string, number[]>();
  for (const [i, g] of parts) groups.set(g, [...(groups.get(g) ?? []), i]);
  return [...groups.entries()]
    .map(([g, ids]) => {
      const who =
        ids.length === tokens.length && ids.length > 1
          ? ids.length === 2
            ? "both"
            : `all ${cardinal(ids.length)}`
          : `the ${listAnd(ids.map((i) => ORDW[i] ?? `${i + 1}th`))}`;
      const verb = ids.length > 1 ? "are" : "is";
      return `${who} ${verb} ${/(script|characters)$/.test(g) ? "in " : ""}${g}`;
    })
    .join("; ");
}

/* ----------------------------------------------------------- bearing words */

const q = (w: string) => `“${w}”`;

/** The bearing in words: 13.9° → turned from “she” toward “he”. */
export function bearingWords(az: number, a: string, b: string, quote = true): string {
  const A = quote ? q(a) : `the landmark word ${a}`;
  const B = quote ? q(b) : b;
  const x = ((az + 540) % 360) - 180; // (-180, 180]
  const ax = Math.abs(x);
  if (ax <= 2) return `pointing at ${A}`;
  if (ax >= 178) return `turned away from ${A}`;
  if (x > 0) {
    if (x < 88) return `turned from ${A} toward ${B}`;
    if (x <= 92) return `pointing at ${quote ? B : `the landmark word ${b}`}`;
    return `turned past ${B}, away from ${quote ? A : a}`;
  }
  if (x > -88) return `turned from ${A}, away from ${B}`;
  if (x >= -92) return `turned away from ${quote ? B : `the landmark word ${b}`}`;
  return `turned away from both ${quote ? A : a} and ${B}`;
}

/** The elevation in words: 49° → risen toward “it”. */
export function elevationWords(el: number, c: string, quote = true): string {
  const C = quote ? q(c) : `the landmark word ${c}`;
  if (Math.abs(el) <= 2) return `level, neither toward ${C} nor away`;
  return el > 0 ? `risen toward ${C}` : `sunk away from ${C}`;
}

/** 0-360 for the dial's figure. */
export const bearingFigure = (az: number) => ((Math.round(az) % 360) + 360) % 360;

/** 2.53e-4 → { mant: "2.5", exp: -4 } */
export function sci(p: number): { mant: string; exp: number } | null {
  if (!(p > 0) || !Number.isFinite(p)) return null;
  const exp = Math.floor(Math.log10(p));
  return { mant: (p / 10 ** exp).toFixed(1), exp };
}

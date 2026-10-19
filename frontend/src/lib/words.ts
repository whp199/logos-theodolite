/**
 * Plain words for the pages. The vocabulary is the canvas's: a sighting, layers counted 1-60,
 * word-pieces, the lens and the naked eye, gathering (attention) and remembering (feed-forward),
 * a neuron's two faces, threads, landmarks, bearing and elevation, stream strength, kinship,
 * a nudge, the vault, an edit.
 */

/** Hand-written glosses from the approved canvas for word-pieces that turn up in the study. */
const GLOSSARY: Record<string, string> = {
  Iman: "a given name",
  Tôi: "Vietnamese: “I”",
  HUAWEI: "a company",
  "！」": "full-width “!” closing a quotation",
  面试: "Chinese: “interview”",
  俩: "Chinese: “the two of us”",
  OpenAI: "a company",
  是一个: "Chinese: “is a”",
  是一個: "Chinese (traditional): “is a”",
  是一: "Chinese: “is one”",
  是我: "Chinese: “is me”",
  是一种: "Chinese: “is a kind of”",
  adalah: "Indonesian: “is”",
  是一款: "Chinese: “is a (model of)”",
  机器人: "Chinese: “robot”",
  JE: "French “I”, in capitals",
  myself: "English “myself”",
  मुझ: "Hindi: “me”",
  的我: "Chinese: “of me”",
};

const SCRIPTS: [RegExp, string][] = [
  [/\p{Script=Han}/u, "Chinese or Japanese characters"],
  [/[\p{Script=Hiragana}\p{Script=Katakana}]/u, "Japanese script"],
  [/\p{Script=Hangul}/u, "Korean script"],
  [/\p{Script=Devanagari}/u, "Devanagari script"],
  [/\p{Script=Bengali}/u, "Bengali script"],
  [/\p{Script=Tamil}/u, "Tamil script"],
  [/\p{Script=Arabic}/u, "Arabic script"],
  [/\p{Script=Hebrew}/u, "Hebrew script"],
  [/\p{Script=Cyrillic}/u, "Cyrillic script"],
  [/\p{Script=Thai}/u, "Thai script"],
  [/\p{Script=Greek}/u, "Greek script"],
  [/\p{Extended_Pictographic}/u, "an emoji"],
];

/** A word-piece as shown on the page: its leading space dropped, quoted. */
export function wp(token: string | null | undefined): string {
  if (token == null) return "";
  const t = token.replace(/^\s+/, "");
  if (t === "") return token.length ? "“␣”" : "“”";
  return `“${t.replace(/\n/g, "↵")}”`;
}

/** A word-piece as wp() shows it, except that a piece of white space alone says which: “␣” a space, “¶” a new line. */
export function pieceWord(token: string | null | undefined): string {
  if (token == null) return "";
  if (token !== "" && token.trim() === "") {
    const lines = (token.match(/\n/g) ?? []).length;
    return lines ? `“${"¶".repeat(lines)}”` : "“␣”";
  }
  return wp(token);
}

/** The first words of a text, for a name: whole words, at most `max` letters, with "…" when cut. */
export function firstWords(text: string, max = 40): string {
  const t = text.trim().replace(/\s+/g, " ");
  if (t.length <= max) return t;
  const cut = t.slice(0, max);
  const sp = cut.lastIndexOf(" ");
  return `${sp > 12 ? cut.slice(0, sp) : cut}…`;
}

/** The bare word-piece without quotes (leading space dropped). */
export function bare(token: string | null | undefined): string {
  if (token == null) return "";
  const t = token.replace(/^\s+/, "");
  return t === "" ? "␣" : t.replace(/\n/g, "↵");
}

/** A short gloss for a word-piece, or "" when it reads plainly. */
export function gloss(token: string | null | undefined): string {
  if (!token) return "";
  const t = token.trim();
  if (GLOSSARY[t]) return GLOSSARY[t];
  for (const [re, name] of SCRIPTS) if (re.test(t)) return name;
  if (/^\p{L}/u.test(token) && !/^\s/.test(token)) return "joined to the word before";
  return "";
}

/** A probability as the canvas prints it: whole percent from 10%, one decimal below. */
export function pct(p: number | null | undefined): string {
  if (p == null || !Number.isFinite(p)) return "";
  const v = p * 100;
  return (v >= 10 ? String(Math.round(v)) : String(Math.round(v * 10) / 10)) + "%";
}

/**
 * How a readout leans, from the share of its top word-piece. Through a lens that share describes the lens's readout, not
 * the model's confidence (and the shares are uncalibrated: erratum E30). At the last layer (`own`) no lens stands
 * between, and the share is the model's own odds for its next word.
 */
export function certainty(top: number | null | undefined, own = false): string {
  if (top == null) return "";
  if (own) {
    if (top >= 0.6) return "all but certain";
    if (top >= 0.2) return "leaning one way";
    if (top >= 0.1) return "undecided";
    return "spread thin";
  }
  if (top >= 0.6) return "the readout leans hard one way";
  if (top >= 0.2) return "the readout leans one way";
  if (top >= 0.1) return "the readout is divided";
  return "a murmur of many words";
}

export const comma = (n: number) => String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ",");

/** A signed number with a true minus sign. */
export function signed(v: number, digits = 2): string {
  return (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(digits);
}

/** Minus signs for display. */
export const minus = (s: string) => s.replace(/-/g, "−");

export function fixed(v: number | null | undefined, digits = 1): string {
  if (v == null || !Number.isFinite(v)) return "";
  return minus(v.toFixed(digits));
}

export const faceWord = (face: number) => (face >= 0 ? "positive" : "negative");

export function kindOf(isGlobal: boolean) {
  return isGlobal
    ? { kind: "✦ global attention", gloss: "looks back over the whole conversation" }
    : { kind: "local attention", gloss: "looks back over the last 1,024 word-pieces" };
}

/** How far a neuron's label can be trusted at a layer, by its code index (N1 sampled code layers 10, 20, 30, 40, 45, 50,
 * 55 and 57: labels held up at UI 41-58 and were at chance at 11-31; 1-10 and 32-40 were not sampled; erratum E9). */
export function labelTrust(idx: number): string {
  if (idx >= 58) return "At the last layers the J++ lens and the naked eye see much the same thing.";
  if (idx >= 40) return "Layers 41 to 58 are where neuron labels held up under causal tests.";
  if (idx >= 31) {
    return "A rumour, not a finding: the study did not test labels at layers 32 to 40, and found them no better than chance below.";
  }
  if (idx >= 10) return "A rumour, not a finding: at layers 11 to 31 the study found neuron labels no better than chance.";
  return "A rumour, not a finding: the study did not test labels below layer 11.";
}

const ORD = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"];
export const numberWord = (n: number) => (n >= 0 && n < ORD.length ? ORD[n] : String(n));

export function ms(v: number | null | undefined): string {
  if (v == null) return "";
  return v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${Math.round(v)} ms`;
}

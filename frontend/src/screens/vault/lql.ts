/**
 * The vault's query language, read on this side exactly as the server reads it: a mirror of
 * LarqlVaultEngine.execute_lql in backend/larql_vault.py (same order, same patterns), ported from the approved
 * board's script. It only explains a statement ("how the vault reads it"); the server always runs the statement
 * itself and its reply is the one shown.
 *
 * Layers in LQL are repo indices (layer number - 1). Every conversion happens here, at the boundary.
 */

/*
 * Python's `re` on str patterns differs from JavaScript's: `\d` is any Unicode decimal digit (and int() reads
 * it), `\s` is any Unicode whitespace by str.isspace() (which counts U+001C-001F and U+0085 but not U+FEFF), and
 * re.I folds case the Unicode way ("ſ" matches "s"). The patterns below are written once with \s and \d and
 * rebuilt with those classes and the u flag, so this side reads a statement as the server does.
 */
const PY_WS = "[\\t\\n\\x0B\\x0C\\r\\x1C-\\x1F \\x85\\xA0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000]";
const PY_WS_RX = new RegExp(PY_WS, "u");
const py = (rx: RegExp) => new RegExp(rx.source.replace(/\\s/g, PY_WS).replace(/\\d/g, "\\p{Nd}"), "iu");

export const RX_SEL = py(
  /SELECT\s+\*\s+FROM\s+(EDGES|POLES)\s+NEAREST\s+['"]([^'"]+)['"](?:\s+WHERE\s+layer\s*=\s*(\d+))?(?:\s+LIMIT\s+(\d+))?/,
);
const RX_POLE = py(/DESCRIBE\s+POLE\s+LAYER\s+(\d+)\s+SLOT\s+(\d+)(?:\s+POLARITY\s+([+-]))?/);
const RX_DEL = py(/DELETE\s+FROM\s+LAYER\s+(\d+)\s+WHERE\s+slot\s*=\s*(\d+)/);
const RX_UPD = py(
  /UPDATE\s+LAYER\s+(\d+)\s+SLOT\s+(\d+)\s+SET\s+object\s*=\s*['"]([^'"]+)['"](?:\s+STRENGTH\s+([0-9.]+))?(?:\s+POLARITY\s+([+-]))?/,
);
const RX_INS = py(
  /INSERT\s+INTO\s+LAYER\s+(\d+)\s*\(\s*subject\s*,\s*object\s*\)\s*VALUES\s*\(\s*['"]([^'"]+)['"]\s*,\s*['"]([^'"]+)['"]\s*\)(?:\s+STRENGTH\s+([0-9.]+))?/,
);

/** Python's str.strip(): its own whitespace, not JavaScript's (which includes U+FEFF). */
export function pyStrip(s: string): string {
  const chars = Array.from(s);
  let a = 0;
  let b = chars.length;
  while (a < b && PY_WS_RX.test(chars[a])) a++;
  while (b > a && PY_WS_RX.test(chars[b - 1])) b--;
  return chars.slice(a, b).join("");
}

/** Python's int() on a run of Unicode decimal digits: each digit's value is its place in its run of ten. */
export function pyInt(s: string): number {
  const isNd = (cp: number) => /\p{Nd}/u.test(String.fromCodePoint(cp));
  let out = "";
  for (const ch of s) {
    const cp = ch.codePointAt(0)!;
    if (cp >= 48 && cp <= 57) {
      out += ch;
      continue;
    }
    let start = cp;
    while (isNd(start - 1)) start--;
    out += String((cp - start) % 10);
  }
  return parseInt(out, 10);
}

const SUPPORTED =
  "Supported: DESCRIBE | SELECT * FROM EDGES NEAREST '<concept>' WHERE layer = <l> LIMIT <k> | DESCRIBE POLE LAYER <l> SLOT <i> | UPDATE LAYER <l> SLOT <i> SET object = '<c>' | INSERT INTO LAYER <l> (subject, object) VALUES ('<s>', '<o>') | DELETE FROM LAYER <l> WHERE slot = <i> | ROLLBACK";

export const clampL = (s: string) => Math.max(0, Math.min(pyInt(s), 59));
export const clampN = (s: string) => Math.max(0, Math.min(pyInt(s), 21503));
export const isFloat = (s: string) => /^(\d+\.?\d*|\.\d+)$/.test(s);

/** Python's repr() of a string, as the server's error messages print it. */
export function pyRepr(s: string): string {
  const dq = s.indexOf("'") >= 0 && s.indexOf('"') < 0;
  let b = s.replace(/\\/g, "\\\\").replace(/\n/g, "\\n").replace(/\r/g, "\\r").replace(/\t/g, "\\t");
  if (!dq) b = b.replace(/'/g, "\\'");
  return dq ? '"' + b + '"' : "'" + b + "'";
}

/** Characters outside the basic plane (emoji among the word-pieces) and U+FFFD, shown as JSON escapes. */
export const esc = (s: string) =>
  s.replace(/[\uD800-\uDFFF�]/g, (c) => "\\u" + ("000" + c.charCodeAt(0).toString(16)).slice(-4));

export type Parsed =
  | { kind: "describe"; stmt: string; alias: string }
  | { kind: "rollback"; stmt: string }
  | { kind: "select"; stmt: string; from: string; concept: string; layerS: string | null; limitS: string | null }
  | { kind: "pole"; stmt: string; layerS: string; slotS: string; pol: "+" | "-"; polGiven: boolean }
  | { kind: "delete"; stmt: string; layerS: string; slotS: string }
  | {
      kind: "update";
      stmt: string;
      layerS: string;
      slotS: string;
      word: string;
      strength: string | null;
      pol: "+" | "-";
      polGiven: boolean;
    }
  | { kind: "insert"; stmt: string; layerS: string; subj: string; obj: string; strength: string | null }
  | { kind: "error"; stmt: string; empty: boolean; num?: boolean; err: string };

export function parseLql(query: string): Parsed {
  const q = pyStrip(query).replace(/;+$/, "");
  const qu = q.toUpperCase();
  if (qu === "DESCRIBE" || qu === "SHOW LAYERS" || qu === "STATUS")
    return { kind: "describe", stmt: q, alias: qu === "DESCRIBE" ? "" : qu };
  if (qu === "ROLLBACK") return { kind: "rollback", stmt: q };
  let m = RX_SEL.exec(q);
  if (m)
    return {
      kind: "select",
      stmt: q,
      from: m[1].toUpperCase(),
      concept: m[2],
      layerS: m[3] === undefined ? null : m[3],
      limitS: m[4] === undefined ? null : m[4],
    };
  m = RX_POLE.exec(q);
  if (m) return { kind: "pole", stmt: q, layerS: m[1], slotS: m[2], pol: m[3] === "-" ? "-" : "+", polGiven: m[3] !== undefined };
  m = RX_DEL.exec(q);
  if (m) return { kind: "delete", stmt: q, layerS: m[1], slotS: m[2] };
  m = RX_UPD.exec(q);
  if (m) {
    if (m[4] !== undefined && !isFloat(m[4]))
      return { kind: "error", stmt: q, empty: false, num: true, err: "could not convert string to float: " + pyRepr(m[4]) };
    return {
      kind: "update",
      stmt: q,
      layerS: m[1],
      slotS: m[2],
      word: m[3],
      strength: m[4] === undefined ? null : m[4],
      pol: m[5] === "-" ? "-" : "+",
      polGiven: m[5] !== undefined,
    };
  }
  m = RX_INS.exec(q);
  if (m) {
    if (m[4] !== undefined && !isFloat(m[4]))
      return { kind: "error", stmt: q, empty: false, num: true, err: "could not convert string to float: " + pyRepr(m[4]) };
    return { kind: "insert", stmt: q, layerS: m[1], subj: m[2], obj: m[3], strength: m[4] === undefined ? null : m[4] };
  }
  return { kind: "error", stmt: q, empty: q === "", err: "Unrecognized LARQL / .patch query: " + pyRepr(query) + ". " + SUPPORTED };
}

export const mutates = (p: Parsed) => p.kind === "delete" || p.kind === "update" || p.kind === "insert" || p.kind === "rollback";

/** The layer a statement names, counted 1-60 (a SELECT without WHERE reads index 35, layer 36). */
export function statementLayerNo(p: Parsed): number | null {
  if ("layerS" in p && p.layerS !== null && p.layerS !== undefined) return clampL(p.layerS) + 1;
  return p.kind === "select" ? 36 : null;
}

/* ------------------------------------------------------------- statements */

/** Find: the nearness search, with the layer converted to the vault's count from nought. */
export function nearestStatement(word: string, layerNo: number | null, k: number | null): string {
  return (
    "SELECT * FROM POLES NEAREST '" +
    (word || "…") +
    "' WHERE layer = " +
    (layerNo !== null ? layerNo - 1 : "…") +
    " LIMIT " +
    (k !== null ? k : "…")
  );
}

/* ---------------------------------------------------------- the grammar */

export const FORMS: [string, string, string][] = [
  ["DESCRIBE | SHOW LAYERS | STATUS", "The vault describes itself: its size, its layers that look back over everything, and the edits in force.", "DESCRIBE"],
  ["ROLLBACK", "Undo every edit, in all sixty layers.", "ROLLBACK"],
  [
    "SELECT * FROM EDGES|POLES NEAREST '<word>' [WHERE layer = <l>] [LIMIT <k>]",
    "Find, as on the first tab; EDGES and POLES run the same search. Left out, the layer is 35 (layer 36) and the limit 10.",
    "SELECT * FROM POLES NEAREST 'she' WHERE layer = 50 LIMIT 10",
  ],
  [
    "DESCRIBE POLE LAYER <l> SLOT <i> [POLARITY +|-]",
    "Both faces of one neuron, read through the lens and by the naked eye.",
    "DESCRIBE POLE LAYER 50 SLOT 5313 POLARITY -",
  ],
  ["DELETE FROM LAYER <l> WHERE slot = <i>", "Silence one neuron, as on the second tab.", "DELETE FROM LAYER 50 WHERE slot = 5313"],
  [
    "UPDATE LAYER <l> SLOT <i> SET object = '<word>' [STRENGTH <alpha>] [POLARITY +|-]",
    "Make one neuron say a word more strongly. Left out, the strength is 2.5 and the face positive.",
    "UPDATE LAYER 35 SLOT 100 SET object = 'six' STRENGTH 3.0 POLARITY +",
  ],
  [
    "INSERT INTO LAYER <l> (subject, object) VALUES ('<subject>', '<object>') [STRENGTH <alpha>]",
    "The vault chooses the neuron itself, the one whose input side lies nearest the subject, and makes it say the object more strongly.",
    "INSERT INTO LAYER 35 (subject, object) VALUES ('spider', 'eight')",
  ],
];

/** The original instrument's three buttons (at its opening layer, index 44). */
export const QUICK: [string, string][] = [
  ["nearest ‘spider’", "SELECT * FROM EDGES NEAREST 'spider' WHERE layer = 44 LIMIT 4"],
  [".patch slot", "UPDATE LAYER 44 SLOT 765 SET object = 'six' STRENGTH 3.0 POLARITY +"],
  ["rollback", "ROLLBACK"],
];

export const INSERT_EXAMPLE = "INSERT INTO LAYER 35 (subject, object) VALUES ('spider', 'eight')";
/** The board's opening statement in the console. */
export const CONSOLE_DEFAULT = "SELECT * FROM POLES NEAREST 'she' WHERE layer = 50 LIMIT 10";

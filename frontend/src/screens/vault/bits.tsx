import React from "react";
import { RAW_LISTS } from "./recorded";
import { esc } from "./lql";

/** Small pieces shared by the vault's tabs (the board's inline styles, gathered once). */

export const ROMAN = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv"];
export const roman = (i: number) => ROMAN[i] ?? String(i + 1);

/** A word-piece in quotes, unless it is itself a quotation mark (board: show()). */
const BARE = /["“”„«»]/;
export const show = (t: string) => (BARE.test(t) ? t : "“" + t + "”");

export const fmtCos = (c: number) => Number(c).toPrecision(2);
export const f1 = (n: number) => Math.round(n * 10) / 10;
export const signedFixed = (x: number, k: number) => (x < 0 ? "−" : "+") + Math.abs(x).toFixed(k);

/** The face mark: a gilt lozenge, filled for the positive face, open for the negative. */
export const FaceMark: React.FC<{ face: number; style?: React.CSSProperties }> = ({ face, style }) => (
  <svg width="13" height="13" viewBox="0 0 12 12" aria-hidden="true" style={style}>
    <path d="M6 0.8 L11.2 6 L6 11.2 L0.8 6 Z" fill={face >= 0 ? "#A07A2E" : "#F6F0E3"} stroke="#A07A2E" strokeWidth="1.2" />
  </svg>
);

export const MONO = "'IBM Plex Mono', ui-monospace, Menlo, monospace";

export const smallCaps = (size = 15, spacing = "0.16em"): React.CSSProperties => ({
  fontVariantCaps: "all-small-caps",
  letterSpacing: spacing,
  fontSize: size,
  color: "var(--muted)",
});

/** The big inline inputs of the board's sentences ("Find the neurons … in layer …"). */
export function sentenceInput(size: number, width: string, opts: { italic?: boolean; numeric?: boolean } = {}): React.CSSProperties {
  return {
    fontFamily: "inherit",
    fontSize: size,
    fontStyle: opts.italic ? "italic" : undefined,
    color: "var(--ink)",
    background: "transparent",
    border: 0,
    borderBottom: "1px solid var(--control)",
    borderRadius: 0,
    padding: opts.italic ? "0 4px 2px" : "0 2px 2px",
    width,
    boxSizing: "border-box",
    textAlign: "center",
    fontVariantNumeric: opts.numeric ? "lining-nums tabular-nums" : undefined,
  };
}

/** A chip of the board (word and layer choices, the agreement gate). */
export function chipStyle(on: boolean, extra: React.CSSProperties = {}): React.CSSProperties {
  return {
    fontFamily: "inherit",
    color: "var(--ink)",
    background: on ? "var(--pressed)" : "transparent",
    border: `1px solid ${on ? "var(--gilt)" : "var(--control)"}`,
    borderRadius: 2,
    minHeight: 44,
    minWidth: 44,
    cursor: "pointer",
    ...extra,
  };
}

export const codeLine: React.CSSProperties = {
  fontFamily: MONO,
  fontSize: 16,
  lineHeight: 1.5,
  color: "var(--ink)",
  overflowWrap: "anywhere",
};

/** "in the instrument's own words": the statement as the vault will read it. */
export const OwnWords: React.FC<{ lql: string; note: React.ReactNode; maxWidth: number }> = ({ lql, note, maxWidth }) => (
  <div style={{ display: "flex", flexDirection: "column", gap: 6, paddingTop: 14, borderTop: "1px solid var(--hair-soft)", maxWidth }}>
    <span style={smallCaps()}>in the instrument’s own words</span>
    <code style={codeLine}>{lql}</code>
    <span style={{ fontSize: 15, fontStyle: "italic", color: "var(--muted)" }}>{note}</span>
  </div>
);

/** The board's empty answer: a bracketed line between two rules. */
export const EmptyAnswer: React.FC<{ head: string; children?: React.ReactNode; pad?: string }> = ({ head, children, pad = "34px 0 36px" }) => (
  <div
    style={{
      display: "flex",
      flexDirection: "column",
      gap: 10,
      padding: pad,
      borderTop: "1px solid var(--hair)",
      borderBottom: "1px solid var(--hair)",
      alignItems: "center",
      textAlign: "center",
    }}
  >
    <span style={{ fontSize: 15, letterSpacing: "0.12em", color: "var(--gilt-text)" }}>{head}</span>
    {children ? <span style={{ fontSize: 16, color: "var(--muted)", maxWidth: 620 }}>{children}</span> : null}
  </div>
);

/** A recorded nearness reply as the study's script kept it: lists["<word>@<index>"], its first k rows (board: rawJson()). */
export const rawJson = (key: string, k: number) =>
  "[\n  " +
  RAW_LISTS[key]
    .slice(0, k)
    .map(
      (r) =>
        '{"n": ' +
        r[0] +
        ', "pol": ' +
        (r[1] < 0 ? "-1.0" : "1.0") +
        ', "cos": ' +
        JSON.stringify(r[2]) +
        ', "jv": [' +
        r[3].map((t) => JSON.stringify(t)).join(", ") +
        '], "nv": [' +
        r[4].map((t) => JSON.stringify(t)).join(", ") +
        "]}",
    )
    .map(esc)
    .join(",\n  ") +
  "\n]";

import React from "react";
import { IfApparatus, Missing, Rubric } from "./kit";
import { heldLine, heldRubric, heldTag, sharedLine, type Gate } from "../lib/labels";
import { wp } from "../lib/words";

/**
 * A neuron's words where the label gate holds them back (lib/labels.ts): in the main reading a small-caps plate,
 * "[held back]", in place of the word line, an italic line saying why, and a rubric for newcomers; the glosses of the
 * words are held back with them. With the apparatus open the words come back, muted between gilt brackets, with what
 * the naked eye read, their glosses, the reason as a tag, and the ten-a-side count the gate went by.
 *
 * It replaces only the words and their glosses: the page keeps the neuron's number, its face, its strength and its
 * links (open the neuron, nudge with it, survey from it) around it. It draws nothing when the gate shows the label.
 * `compact` is for a cell of a table: the plate carries its tag, and the line and the rubric are left to the page.
 * `unchecked` names what stands in for a missing count, e.g. "not checked ten a side: this sighting carries three
 * words a side" (a bracketed placeholder; never a count made from the words shown).
 */
export const HeldBackWords: React.FC<{
  gate: Gate;
  /** the lens's word-pieces, raw (with their leading spaces) */
  words: readonly string[] | null;
  /** the naked eye's word-pieces, raw */
  eye?: readonly string[] | null;
  /** the glosses a page would set under the words, held back with them */
  glosses?: React.ReactNode;
  unchecked?: string;
  compact?: boolean;
  /** the size of the page's own word line, for the plate */
  size?: number;
  style?: React.CSSProperties;
}> = ({ gate, words, eye, glosses, unchecked = "not checked ten a side", compact = false, size = 22, style }) => {
  if (!gate.heldBack) return null;
  const tag = heldTag(gate);
  const count = sharedLine(gate);
  const bracketed = (list: readonly string[] | null | undefined, label: string) =>
    list && list.length ? (
      <span>
        <span className="sc" style={{ fontSize: 14, letterSpacing: "0.12em", color: "var(--muted)" }}>{label} </span>
        <span style={{ color: "var(--gilt)" }}>[</span>
        <span style={{ color: "var(--muted)" }}>{list.map((t) => wp(t)).join("  ")}</span>
        <span style={{ color: "var(--gilt)" }}>]</span>
      </span>
    ) : null;

  if (compact) {
    return (
      <IfApparatus
        else={
          <span className="placeholder" style={style}>
            [{tag}]
          </span>
        }
      >
        <span style={{ display: "inline-flex", flexDirection: "column", gap: 2, ...style }}>
          {bracketed(words, "lens")}
          {bracketed(eye, "naked eye")}
          <span className="placeholder">
            [{tag}]{count ? ` · ${count}` : ""}
          </span>
        </span>
      </IfApparatus>
    );
  }

  return (
    <div className="stack" style={{ gap: 4, ...style }} aria-label={`${tag}: ${heldLine(gate)}`}>
      <IfApparatus
        else={
          <span className="sc" style={{ fontSize: Math.round(size * 0.8), letterSpacing: "0.16em", color: "var(--muted)" }}>
            [held back]
          </span>
        }
      >
        <div className="stack" style={{ gap: 2, fontSize: Math.round(size * 0.8), lineHeight: 1.35 }}>
          {bracketed(words, "lens")}
          {bracketed(eye, "naked eye")}
        </div>
        {glosses ? <div style={{ fontSize: 15, fontStyle: "italic", color: "var(--muted)" }}>{glosses}</div> : null}
        <span className="kicker">{tag}</span>
        <span style={{ fontSize: 15, color: "var(--muted)" }}>{count ?? <Missing what={unchecked} />}</span>
      </IfApparatus>
      <div style={{ fontSize: 15, fontStyle: "italic", color: "var(--muted)" }}>{heldLine(gate)}</div>
      <Rubric style={{ margin: "2px 0 0", fontSize: 15 }}>{heldRubric(gate)}</Rubric>
    </div>
  );
};

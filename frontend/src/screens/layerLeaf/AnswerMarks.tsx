import React from "react";
import { piecesOf, shownSetOf, useTheodoliteStore, type AnswerPiece, type Sighting } from "../../store/useTheodoliteStore";

/**
 * A sighting's answer as running text, for the strips at the head of the leaf and the folio. The word-piece the stream had
 * just read is underlined in gilt, as these pages always marked the word read; the word-piece chosen there is marked with a
 * dotted red rule. The sighting of word-piece k reads w_{k-1} (for k = 0, the prompt's own last word-piece, which comes
 * before the answer) and chooses w_k; the last piece written is chosen but never read.
 */

/** The gilt rule under the word-piece read. */
export const READ_MARK: React.CSSProperties = { borderBottom: "2px solid #A07A2E" };
/** The red rule under the word-piece chosen. */
export const CHOSEN_MARK: React.CSSProperties = { borderBottom: "2px dotted #8F2B1E" };

const SR_ONLY: React.CSSProperties = {
  position: "absolute",
  width: 1,
  height: 1,
  overflow: "hidden",
  clip: "rect(0 0 0 0)",
  whiteSpace: "nowrap",
};

/** How a written word-piece is drawn in running text: its leading space outside the mark, a new line as ¶, a bare space as
 * ␣, its end of turn in words (as the instrument's row of word-pieces draws them). */
function pieceFace(p: AnswerPiece): { lead: string; body: string; plain: boolean } {
  if (p.endMark) return { lead: " ", body: "it ends its turn", plain: false };
  const lines = (p.str.match(/\n/g) ?? []).length;
  const text = p.str.replace(/^\s+/, "").replace(/\n/g, "¶");
  if (!text) return { lead: lines ? " " : "", body: lines ? "¶".repeat(lines) : "␣", plain: false };
  return { lead: /^\s/.test(p.str) ? " " : "", body: (lines ? "¶".repeat(lines) + " " : "") + text, plain: true };
}

/** The word-pieces of the answer a sighting belongs to: its set's (when it is the one shown), else its own echo. */
export function useAnswerPieces(s: Sighting | null): AnswerPiece[] | null {
  const set = useTheodoliteStore((st) => {
    const a = shownSetOf(st);
    return a && s && a.sightings[s.tokenIndex] === s ? a : null;
  });
  return set?.pieces ?? piecesOf(s);
}

export const AnswerMarks: React.FC<{ s: Sighting; pieces: AnswerPiece[] }> = ({ s, pieces }) => {
  const k = s.tokenIndex;
  return (
    <>
      {pieces.map((p, i) => {
        const f = pieceFace(p);
        const read = i === k - 1;
        const chose = i === k;
        const mark: React.CSSProperties = {
          ...(read ? READ_MARK : chose ? CHOSEN_MARK : {}),
          ...(f.plain ? {} : { fontStyle: "normal", fontSize: "0.7em", fontVariantCaps: "all-small-caps", letterSpacing: "0.12em" }),
        };
        const role = read ? "the word-piece read here" : chose ? "the word-piece chosen here" : "";
        return (
          <React.Fragment key={i}>
            {f.lead}
            <span style={mark} title={role ? `${JSON.stringify(p.str)}, ${role}` : JSON.stringify(p.str)}>
              {f.body}
            </span>
            {role ? <span style={SR_ONLY}> ({role})</span> : null}
          </React.Fragment>
        );
      })}
    </>
  );
};

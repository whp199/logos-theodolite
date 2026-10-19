import React from "react";
import {
  chosenOf,
  ordinal,
  readWhere,
  useShownSet,
  useTheodoliteStore,
  type AnswerSet,
  type Sighting,
} from "../store/useTheodoliteStore";
import { pct, pieceWord, wp } from "../lib/words";

/**
 * The matched control, as an apparatus row's content: which sighting is kept, a button to keep or clear
 * it, and (given a layer) what the lens read the control as there, worded as on the instrument. The study reads
 * lens outputs against a matched control (a nameless or background condition), never alone (REPORT §3.5).
 *
 * A control is kept as a condition, not as one word of an answer: the store keeps the token-0 sighting of the answer it
 * came from, read where the model chose its first word-piece. So the per-layer comparison is made only when the sighting
 * shown is a first word-piece too; a later word-piece of an answer is never compared with it.
 */

/** Whether the control is the answer shown: the same set (or a re-reading of it), not merely the same object. */
function isShownAnswer(control: Sighting | null, sighting: Sighting | null, shown: AnswerSet | null): boolean {
  if (!control || !sighting) return false;
  if (control === sighting) return true;
  if (shown && shown.sightings[control.tokenIndex] === control) return true;
  // a re-reading of the same answer (an option that changes only the reading) is still the same question and answer
  const a = control.requestId;
  return a !== null && (a === sighting.requestId || (shown !== null && shown.retakeOf === a));
}

/** Where the kept control was read: “read where it chose its first word-piece, “am” (it read “I”, in its reply “Hi! I”)”,
 * readWhere's phrase after the piece read. */
function controlRead(c: Sighting): React.ReactNode {
  const where = readWhere(c);
  const chose = chosenOf(c);
  const read = c.readAt != null ? <span title={JSON.stringify(c.readAt)}>{pieceWord(c.readAt)}</span> : null;
  if (where.k >= 1) {
    // only when the set's first word-piece was never sighted: the control then stands at a later word-piece
    return (
      <>
        read at {read ?? "a word-piece"}, the {ordinal(where.k)} word-piece of its own answer
        {chose ? <>, {chose.endMark ? "where it ended its turn" : <>where it chose {pieceWord(chose.token)}</>}</> : null}
      </>
    );
  }
  return (
    <>
      read where it chose its first word-piece
      {chose ? <>, {chose.endMark ? "the end of its turn" : pieceWord(chose.token)}</> : null}
      {read ? <> (it read {read}{where.phrase})</> : where.phrase ? <> ({where.phrase.replace(/^, /, "")})</> : null}
    </>
  );
}

export const ControlSighting: React.FC<{ layerIdx?: number }> = ({ layerIdx }) => {
  const sighting = useTheodoliteStore((s) => s.sighting);
  const control = useTheodoliteStore((s) => s.control);
  const setControl = useTheodoliteStore((s) => s.setControl);
  const shown = useShownSet();
  const isThis = isShownAnswer(control, sighting, shown);
  const k = sighting?.tokenIndex ?? 0;
  // the comparison layer by layer: only when the sighting shown is read where its first word-piece is chosen, as the
  // control is, and the control is another question
  const matched = !!control && !isThis && k === 0 && control.tokenIndex === 0;
  const at = control && layerIdx !== undefined && matched ? control.layers[layerIdx] : null;
  return (
    <>
      {control ? (
        <span>
          kept as the control: “{control.question}”
          {control.answerBegins ? <span className="italic"> / “{control.answerBegins}”</span> : null}
          {control.system ? <span className="muted"> (with a system prompt)</span> : null} — {controlRead(control)} (
          {control.source === "live" ? "live" : "recorded"}
          {isThis ? "; the answer shown now" : ""})
        </span>
      ) : (
        <span className="placeholder">[NONE SET]</span>
      )}
      {control && !isThis && sighting && k >= 1 ? (
        <span>
          It is matched to the first word-piece only, where each chose its first word-piece; this is word-piece {k + 1} of its
          answer, so the two are not compared here.
          {sighting.centreKind === "control_reading_same_answer"
            ? " This word-piece was centred on the control reading this very answer, up to the same word-piece."
            : sighting.centreKind === "none" && (sighting.frame?.nCentre ?? 0) > 0
              ? " Nothing was subtracted at this word-piece: a control read where its question ends is never subtracted from a later one."
              : ""}
        </span>
      ) : null}
      {at ? (
        <span>
          At layer {at.no} the lens read the control as{" "}
          {at.lensTop?.[0] ? (
            <>
              {wp(at.lensTop[0].token)} ({pct(at.lensTop[0].prob)})
            </>
          ) : (
            <span className="placeholder">[NOT RECORDED AT THIS LAYER]</span>
          )}
          {at.ffnShare != null ? <>, remembering {Math.round(at.ffnShare * 100)}%</> : null}.
        </span>
      ) : null}
      <button type="button" className="btn-link" disabled={!sighting} onClick={() => setControl(isThis ? null : sighting)}>
        {isThis
          ? "clear the control"
          : control
            ? "keep the question shown now as the control instead"
            : "keep the question shown now as the control"}
      </button>
    </>
  );
};

import {
  chosenPhrase,
  endMarkWords,
  readWhere,
  type AnswerPiece,
  type AnswerSet,
  type GenerationEcho,
  type Sighting,
} from "../../store/useTheodoliteStore";
import { pieceWord } from "../../lib/words";

/** The words endMarkWords gives the mark that ends the model's turn (<turn|>, id 106). */
const TURN_ENDS = endMarkWords(106, "<turn|>");

/**
 * Writing stopped at a stop mark, as a clause in the past: “it ended its turn” only for <turn|>; for any other of the
 * checkpoint's stop marks, “it wrote” and endMarkWords' name for it (“it wrote the end of the text (<eos>)”, “it wrote a
 * control mark, <|tool_response>, where writing stops”), so a mark a strong nudge forced is never called a clean ending.
 * Without the piece (an older server), only that it wrote one of its stop marks.
 */
export function endedWords(id: number | null | undefined, token?: string | null): string {
  if (id == null && !token) return "it wrote one of its stop marks";
  const words = endMarkWords(id, token);
  return words === TURN_ENDS ? "it ended its turn" : `it wrote ${words}`;
}

/** Whether a stop mark is the one that ends the model's turn (<turn|>), by endMarkWords. */
export const endsTurn = (id: number | null | undefined, token?: string | null): boolean => endMarkWords(id, token) === TURN_ENDS;

/**
 * Why the writing stopped, in plain words (the server's stop_reason); at a stop mark, which one, from the answer's last
 * piece (endedWords). Null when the reason was not sent.
 */
export function stopWords(r: GenerationEcho["stopReason"] | undefined, pieces: AnswerPiece[] | null | undefined): string | null {
  switch (r) {
    case "end_mark": {
      const last = pieces && pieces.length ? pieces[pieces.length - 1] : null;
      return last && last.endMark ? endedWords(last.id, last.str) : endedWords(null);
    }
    case "length":
      return "the length ran out";
    case "stopped":
      return "stopped by you";
    case "given":
      return "given, not written";
    case "none":
      return "it was asked to write none";
    default:
      return null;
  }
}

/**
 * The moment a sighting shows, in the pages' words: where the stream was read (readWhere's phrasing) and what its layer
 * 60 chose there. “read at “ I”, in its reply “Hi! I”, where it chose “ am”” for the first word-piece; “read at “am”,
 * the 3rd word-piece of its own answer, where it chose “a”” for a later one. No choice is named for the recorded
 * sighting or an older server, which keep no record of it.
 */
export function momentWords(s: Sighting): string {
  const where = readWhere(s);
  const at =
    where.k >= 1
      ? `read ${where.phrase.replace(/^,\s*/, "")}`
      : `read at ${s.readAt != null ? pieceWord(s.readAt) : "its last word-piece"}${where.phrase}`;
  const chose = chosenPhrase(s);
  return chose ? `${at}, ${chose}` : at;
}

/** How many word-pieces the answer of a sighting has (its set's, else its own echo); null when none was sent. */
export function answerLength(s: Sighting, set: AnswerSet | null): number | null {
  return set?.pieces?.length ?? s.answerIds?.length ?? null;
}

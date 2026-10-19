import { chosenPhrase, readWhere, type AnswerSet, type Sighting } from "../../store/useTheodoliteStore";
import { pieceWord } from "../../lib/words";

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

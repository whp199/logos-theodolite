/**
 * The lines the pages place beside the word-pieces the model wrote: what the study measured where a sighting reads,
 * how the answer was written, and when two sightings stop being a matched pair. Red lines for newcomers go through
 * <Rubric>; lines that state how this answer was written or compared show whatever the rubrics are set to. `strings`
 * carries the same texts as plain strings, for titles and aria labels.
 *
 * Every sentence is traced in the comment above it. REPORT means experiments/REPORT.md (line numbers as checked on
 * 2026-10-18); its layers count from 0, the app's from 1, so app layer = report layer + 1 (REPORT.md:35-36). Other
 * files are under experiments/results/ or experiments/scripts/ unless a path is given.
 *
 * Where the study read, from its scripts (REPORT.md names no position class itself, only "assistant-turn positions",
 * REPORT.md:292):
 * - E3 measured the lens on 24 encyclopedic paragraphs and on 24 chats whose greedy default answers (110 word-pieces)
 *   were read back (E3_lens_validation_REPORT.md:69-70; E3_lens_validation_02_genchats.py:6). Its answer class runs
 *   from the piece after the template's closing mark "<channel|>" to the end of the answer
 *   (E3_lens_validation_fidlib.py:9-11). So it holds every piece a sighting at k >= 1 reads (w_0 … w_{n-2}), and not
 *   the mark that a chat sighting with no begun answer reads at k = 0.
 * - R1's held-out tests use E3's classes (R1_06_evaldir.py:34, R1_10_evalfull.py:46). Only R1's fit corpus leaves out
 *   the first answer piece and every "<…>" piece (R1_02_prep.py:27).
 * - The checks against behaviour also read inside answer text: self tokens (E1_self_readout_common.py:104), the last
 *   piece of a begun answer at the pronoun slot (E2_behavior_b_teacherforced.py:55, 75), and the last piece of a
 *   passage begun in the model's turn (E5_name_common.py:59-64; E5_name_A_thirdperson.py:29). The template's own
 *   pieces were only patched, with no lens (E7_binding_REPORT.md:184).
 */
import React from "react";
import { Rubric } from "../components/kit";
import { href } from "../router";
import type { GenerationEcho } from "../store/sighting";
import { useTheodoliteStore } from "../store/useTheodoliteStore";
import { isShippedLens, useReadingLens } from "./lens";

/** A setting as the server echoed it: whole numbers bare, others to at most three places. */
const num = (x: number): string => (Number.isInteger(x) ? String(x) : String(Math.round(x * 1000) / 1000));

/** A signed setting, with a true minus sign. */
const signed = (x: number): string => (x < 0 ? `−${num(-x)}` : num(x));

/* ------------------------------------------------------- inside its own answer */

/**
 * The line for a sighting at k >= 1, in three parts: the text before the link, the link's words, the text after.
 * - The lens's lead over the naked eye at answer pieces: REPORT.md:292 (§3.1, "Assistant-turn positions": J++'s
 *   advantage over the logit lens shrinks from +9.1 nats on prose to +3.1 at L30-39, and turns negative at L40-49).
 *   By band, E3_lens_validation_REPORT.md:17-24, 109-112: +3.1 [2.3, 3.9] at L30-39, −0.95 [−1.4, −0.5] at L40-49,
 *   −0.3 at L50-58, against +9.1, +1.5 and +0.8 on prose. 3.1 / 9.1 = 0.34 gives "a third", at app layers 31-40.
 *   "Gone from layer 41" is a band statement. Per layer, R1 has J++ top-1 on the model's own replies at 0.13, 0.22,
 *   0.25, 0.39 and 0.69 at L40, 45, 50, 55 and 58, against 0.17, 0.27, 0.32, 0.36 and 0.70 for the naked eye
 *   (R1_IT_LENS_REPORT.md:463): ahead only at L55, by 0.03. Erratum E31 (REPORT.md:758-765) confirms this for the
 *   base-fitted J++ and adds that an instruct-fitted lens restores some lead at L40-58. That is why the line names the
 *   shipped lens when another one is loaded.
 * - Why the line speaks of the lead and not of "the lens is weakest here": in absolute terms the lens matches the
 *   model's next word more often inside answers than on prose (top-1 0.155 against 0.033 at L40-49, 0.352 against
 *   0.111 at L50-58), because the model's own next word is easy to predict there (final entropy 0.21 against 2.38
 *   nats; E3_lens_validation_REPORT.md:90-96).
 * - "Compare … across matched sightings": REPORT.md:309-311 (§3.1 bottom line: a single-condition readout is
 *   uninterpretable without matched controls), REPORT.md:376-377 (§3.5, recommendation 2) and erratum E14
 *   (REPORT.md:594-596): at answer pieces the J++ pronoun reading tracks the model's own final score at only
 *   r 0.08-0.29 position by position. Condition-level orderings were validated; single positions were not.
 * - The raw wording (no chat template): the study measured the model's own writing only in chat answers, E3's answer
 *   class above and R1's held-out positions, which come from E3's chats (R1_IT_LENS_REPORT.md:387-388).
 */
function answerPositionParts(raw: boolean, shipped: boolean): [string, string, string] {
  const lens = shipped ? "the lens" : "the shipped J++ lens";
  const lead = raw
    ? `Here the model is reading text it wrote itself, without the chat template. The study measured ${lens} inside ` +
      "the model’s own writing in chat answers, where its lead over the naked eye was a third of its lead on prose " +
      "at layers 31 to 40, and gone from layer 41 (E3): compare this word-piece across "
    : `Here the model is reading its own answer. In its answers the study found ${lens}’s lead over the naked eye a ` +
      "third of its lead on prose at layers 31 to 40, and gone from layer 41 (E3): compare this word-piece across ";
  return [lead, "matched sightings", " rather than read it alone."];
}

/**
 * For a sighting of word-piece k >= 1 (it reads w_{k-1}, a piece the model wrote): what the study found the lens can
 * bear inside the model's own answers. Nothing for k = 0. The words and their sources are at answerPositionParts.
 * The lens is the shown sighting's (else the server's); the wording follows the shown sighting's chat setting.
 */
export const AnswerPositionCaution: React.FC<{ k: number }> = ({ k }) => {
  const sighting = useTheodoliteStore((s) => s.sighting);
  const { lens } = useReadingLens(sighting);
  if (!(k >= 1)) return null;
  const [lead, link, tail] = answerPositionParts(sighting ? !sighting.chatWrapped : false, isShippedLens(lens));
  return (
    <Rubric>
      {lead}
      <a href={href.lexicon("matched-control")}>{link}</a>
      {tail}
    </Rubric>
  );
};

/* ------------------------------------------------------ two pieces of one answer */

/**
 * Beside the clickable answer: two word-pieces of one answer are not a matched pair.
 * - "Their sightings read different word-pieces after different amounts of text": sighting k reads position P - 1 + k,
 *   after the prompt and w_0 … w_{k-1} (SPEC §0).
 * - "The lens leans differently at different kinds of word": the average gradient a lens stands for depends on the
 *   kind of position (R1_IT_LENS_REPORT.md:204-211: self vs assistant pieces cos 0.62 / 0.70 / 0.82 at L30-39 / 40-49 /
 *   50-58; assistant vs user 0.40 / 0.44 / 0.51). The lens's lead over the naked eye differs by kind of position
 *   (E3_lens_validation_REPORT.md:109-112). Its background leaves a role-specific residue (self tokens ≈ −2.8 on the
 *   noun axis at L40-49) that matched-control contrasts cancel (erratum E34, REPORT.md:798-799). At a pronoun the
 *   reading shows the role's default, while prediction positions show the resolved referent (erratum E26,
 *   REPORT.md:709-713).
 * - "Only layer 60 … compares cleanly": there no lens stands between; the final-layer unembed reproduces the real
 *   logits (REPORT.md:86; erratum E12, REPORT.md:571-576).
 * - "Compare the same word-piece across matched sightings": the study trusted contrasts between conditions at one kind
 *   of position (REPORT.md:309-311, 376-377; erratum E39, REPORT.md:845-849). It never measured change from one
 *   position to the next.
 */
const NOT_A_PAIR: [string, string, string] = [
  "Two word-pieces of one answer are not a matched pair: their sightings read different word-pieces after different " +
    "amounts of text, and the lens leans differently at different kinds of word. Only layer 60, the model’s own odds, " +
    "compares cleanly from one word-piece to the next; otherwise compare the same word-piece across ",
  "matched sightings",
  ".",
];

export const NotAMatchedPair: React.FC = () => (
  <Rubric>
    {NOT_A_PAIR[0]}
    <a href={href.lexicon("matched-control")}>{NOT_A_PAIR[1]}</a>
    {NOT_A_PAIR[2]}
  </Rubric>
);

/* ------------------------------------------------------------ how it wrote */

/**
 * The advice after a draw.
 * - "One draw says little about what it usually says": with the study's sampled settings (temperature 1.0, top-p
 *   0.95, top-k 64; 32 draws per prompt: V1_name_behavior_NOTES.md:10), its third-person self-descriptions with no
 *   system prompt named it ChatGPT in 37 of 128 draws, 25 of 32 for one prompt, and Gemma in none (REPORT.md:129-130).
 * - "The sampler changes which words were written, not how any word was read": every reading comes from one re-read of
 *   the written ids (prompt + answer[:-1]) that takes no sampler setting (SPEC §2). The sampler only picks the next id
 *   from the logits (interp_engine steer.py:643-673, _sample_next).
 */
const DRAW_ADVICE =
  "One draw says little about what it usually says. The sampler changes which words were written, not how any word " +
  "was read.";

/** "temperature 1, top-k 64, top-p 0.95, seed 1234", as the server echoed them (top-k/top-p null or 0/1 = off). */
function drawSettings(g: GenerationEcho, withSeed: boolean): string {
  const parts: string[] = [];
  if (g.temperature != null) parts.push(`temperature ${num(g.temperature)}`);
  parts.push(g.topK != null && g.topK > 0 ? `top-k ${g.topK}` : "top-k off");
  parts.push(g.topP != null && g.topP < 1 ? `top-p ${num(g.topP)}` : "top-p off");
  if (withSeed && g.seed != null) parts.push(`seed ${g.seed}`);
  return parts.join(", ");
}

/**
 * How the answer was written, from the server's echo only (never the form): the plain fact, and for a draw the advice.
 * - Greedy is the argmax of the step's scores (steer.py:658-659). The study read the lens on greedy answers: E3's 110
 *   word-pieces at temperature 0 (E3_lens_validation_02_genchats.py:6; E3_lens_validation_REPORT.md:69) and R1's 120
 *   (R1_IT_LENS_REPORT.md:144). Its behaviour counts were sampled (V1, above), hence "the answers the study checked the
 *   lens on", not "the study".
 * - With a penalty, greedy takes the argmax after the penalty (steer.py:654-659), so the line stops saying "its
 *   likeliest"; PenaltyLine says why. The study used no penalty (no study script sets one).
 * - A re-read of given pieces (source "given": a retake, or "sight this one too") echoes the request's sampler, which
 *   the store copies from the set's own request (SPEC §3.2, retakeIfLive), but not the seed that drew them: a seed
 *   echoed for a re-read was never used, so it is not printed.
 * - The recorded sighting was written greedily, 6 word-pieces (store/sighting.ts, RECORDED_GENERATION).
 */
function oneDrawParts(g: GenerationEcho | null): { fact: string; advice: string | null } | null {
  if (!g) return null;
  if (g.source === "recorded") return { fact: "its likeliest word-piece each time (greedy), as the study recorded it.", advice: null };
  const penalised = g.presencePenalty != null && g.presencePenalty !== 0;
  const given = g.source === "given";
  if (g.greedy) {
    if (given) {
      return {
        fact: `its likeliest word-pieces${penalised ? " after the repeat penalty" : ""}, written before and read again.`,
        advice: null,
      };
    }
    return {
      fact: penalised
        ? "the likeliest word-piece each time after the repeat penalty (greedy)."
        : "its likeliest word-piece each time (greedy), as in the answers the study checked the lens on.",
      advice: null,
    };
  }
  return {
    fact: given
      ? `drawn at random before (${drawSettings(g, false)}) and read again here: one of many answers it could give.`
      : `drawn at random (${drawSettings(g, true)}): one of many answers it could give.`,
    advice: DRAW_ADVICE,
  };
}

/**
 * Under the answer: how it was written, from `generation` (the server's echo). Drawn at random, it names the settings
 * and seed and says one draw is one of many answers, with the advice as a red line. Greedy, it says so. Nothing when
 * the echo is missing. Words and sources at oneDrawParts and DRAW_ADVICE.
 */
export const OneDrawLine: React.FC<{ generation: GenerationEcho | null }> = ({ generation }) => {
  const line = oneDrawParts(generation);
  if (!line) return null;
  return (
    <span>
      {line.fact}
      {line.advice ? (
        <>
          {" "}
          <Rubric as="span" style={{ fontSize: "inherit" }}>
            {line.advice}
          </Rubric>
        </>
      ) : null}
    </span>
  );
};

/**
 * The repeat penalty, when the echo's presence penalty is not 0.
 * - interp_engine sampling.py:146-159 (apply_presence_penalty): the penalty is subtracted from the score (logit) of
 *   every id already generated, once each however often it came; "the prompt's tokens are not penalized". The loop's
 *   `generated` list starts empty, after the prompt, so the begun answer does not count either (steer.py:860-877).
 * - It comes before the temperature and before the greedy shortcut (steer.py:654-659), so in either mode the piece
 *   chosen need not be the model's own likeliest. A negative value raises those scores instead (SPEC §1.1 allows −2 to 2).
 */
function penaltyText(p: number): string | null {
  if (!Number.isFinite(p) || p === 0) return null;
  return (
    `With a repeat penalty of ${signed(p)}, every word-piece it had already written had its score ` +
    `${p > 0 ? "lowered" : "raised"} by ${num(Math.abs(p))} before each choice (your question and its begun answer do ` +
    "not count), so the one it chose need not be its likeliest."
  );
}

export const PenaltyLine: React.FC<{ penalty: number }> = ({ penalty }) => {
  const text = penaltyText(penalty);
  return text ? <span>{text}</span> : null;
};

/* ------------------------------------------------------- the template's end */

/**
 * For a chat sighting read at the end of the chat template, with no begun answer (the closing mark "<channel|>",
 * k = 0). Rendered because the study's record confirms the lens was never checked there. REPORT.md says it only by
 * omission: its fidelity row is "assistant-turn positions" (REPORT.md:292). The scripts behind it are explicit (see the
 * file's head). E3's answer class starts one piece after "<channel|>", and its user class is the question's content up
 * to "<turn|>" (E3_lens_validation_fidlib.py:9-18). R1 evaluated on the same classes. E1, E2 and E5 read inside answer
 * text, E2 and E5 at the end of a begun answer, which is why the line says that is where the study read.
 * - "On prose, on questions and inside answers": E3_lens_validation_REPORT.md:70 (classes: prose, assistant content,
 *   assistant self tokens, user content); R1_IT_LENS_REPORT.md:387-388 (2,010 prose, 2,452 assistant, 86 self and
 *   260 user positions).
 */
const TEMPLATE_END =
  "Here the model reads the end of the chat template, where its answer is about to begin. The study never checked the " +
  "lens at this mark, only on prose, on questions and inside answers (E3, R1); begin its answer to read where the study did.";

export const TemplateEndCaution: React.FC = () => <Rubric>{TEMPLATE_END}</Rubric>;

/* ------------------------------------------------------- a nudge's two answers */

/**
 * Where a plain answer and a nudged one part. `j` is the token index (counted from 0, as `k` and a sighting's
 * tokenIndex are) of the first word-piece at which the two answers differ. The line counts from 1, as the strip's
 * "word-piece 3 of 12" does (w_j is word-piece j + 1).
 * - Sighting k reads the prompt and w_0 … w_{k-1} (SPEC §0). With w_0 … w_{j-1} the same in both, both sightings of
 *   index j read the same text and differ only by the nudge, the matched design the study trusted (REPORT.md:376-377).
 *   There they chose differently. From index j + 1 on they read different texts.
 */
function nudgePartText(j: number): string | null {
  if (!Number.isInteger(j) || j < 0) return null;
  return (
    `The two answers part at word-piece ${j + 1}: both read the same text there, and chose differently. From word-piece ` +
    `${j + 2} on they read different texts and are no longer a matched pair.`
  );
}

export const NudgePartLine: React.FC<{ j: number }> = ({ j }) => {
  const text = nudgePartText(j);
  return text ? (
    <p role="note" style={{ margin: 0 }}>
      {text}
    </p>
  ) : null;
};

/* ------------------------------------------------------------- as strings */

const joined = (parts: [string, string, string]) => parts.join("");

/**
 * The same texts as plain strings, for titles and aria labels. The lines that depend on a value are functions; each
 * returns null where its component renders nothing. answerPosition and answerPositionRaw are the shipped lens's
 * wording (chat and without the chat template); oneDraw gives the fact and the advice together.
 */
export const strings = {
  answerPosition: joined(answerPositionParts(false, true)),
  answerPositionRaw: joined(answerPositionParts(true, true)),
  notAMatchedPair: joined(NOT_A_PAIR),
  oneDraw: (generation: GenerationEcho | null): string | null => {
    const line = oneDrawParts(generation);
    return line ? (line.advice ? `${line.fact} ${line.advice}` : line.fact) : null;
  },
  drawAdvice: DRAW_ADVICE,
  penalty: penaltyText,
  templateEnd: TEMPLATE_END,
  nudgePart: nudgePartText,
} as const;

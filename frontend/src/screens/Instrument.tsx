import React, { useEffect, useMemo, useState } from "react";
import { Apparatus, IfApparatus, LensCaution, LensRecord, Missing, Page, RecordedNotice, Rubric, SweepError } from "../components/kit";
import { useApparatus } from "../reading";
import { href, type Route } from "../router";
import {
  type AnswerSet,
  type LayerReading,
  type ReadingOptions,
  type Sighting,
  READING_AS_BUILT,
  READING_MENDED,
  SIGHT_CAP,
  VOCAB,
  chosenOf,
  exportAnswer,
  exportSighting,
  generationWords,
  isChosenRow,
  leadOf,
  ordinal,
  piecesOf,
  rankingWords,
  readWhere,
  readsInsideAnswer,
  retakeIfLive,
  thinkingOf,
  useShownSet,
  useTheodoliteStore,
} from "../store/useTheodoliteStore";
import { certainty, comma, gloss, kindOf, minus, ms, pct, pieceWord, signed, wp } from "../lib/words";
import { gateTrustLine, heldRubric, heldTag, labelCheckJson, labelGate, layerIdx, overlapLine, sharedLine, sharedShown } from "../lib/labels";
import { HeldBackWords } from "../components/HeldBack";
import { InstrumentFigure, type Shown } from "./instrument/InstrumentFigure";
import { Ascent } from "./instrument/Ascent";
import { Kinship } from "./instrument/Kinship";
import { CANVAS_STATIONS, RECORDED_POLE_NOTES } from "./instrument/recorded";
import { answerLength, endedWords, endsTurn, momentWords, stopWords } from "./instrument/moment";
import { describeFrame } from "../lib/frame";
import { lensLabel, useReadingLens } from "../lib/lens";
import { ControlSighting } from "../components/ControlSighting";
import { QuestionBox } from "../components/QuestionBoxes";
import { SavedShelf } from "../components/SavedShelf";
import { HowItWrites } from "../components/HowItWrites";
import { WordsItWrote } from "../components/WordsItWrote";
import { TemplateEndCaution } from "../lib/cautions";

const sci = (v: number) => minus(Math.abs(v) >= 0.01 ? v.toFixed(3) : v.toExponential(1));

/**
 * The word-piece read where the question ends (word-piece 0's), whichever word-piece of the answer is shown: a later
 * one reads a piece the model wrote, which the question's hint must never present as the reader's own last word. The
 * prompt's last piece is the same for every sighting of one answer (prompt_token_strs is the prompt only).
 */
function promptEndPiece(s: Sighting, set: AnswerSet | null): string | null {
  if (s.tokenIndex === 0) return s.readAt;
  const first = set?.sightings[0];
  if (first && first.tokenIndex === 0 && first.readAt != null) return first.readAt;
  const p = s.promptTokens;
  return p && p.length ? p[p.length - 1] : null;
}

/**
 * Whether the answer's block under the question (WordsItWrote) already shows the template-end caution for the sighting
 * shown: it does for a word-piece 0 read at the end of the chat template, when the answer's word-pieces are drawn.
 * The question's hint then leaves it to that block, so the page does not say it twice.
 */
function answerSaysTemplateEnd(s: Sighting | null, set: AnswerSet | null): boolean {
  if (!s || !set) return false;
  return (set.pieces ?? piecesOf(s)) != null && s.tokenIndex === 0 && readWhere(s).role === "template_end";
}

/** The kept control is this sighting's own question (the same set, or the same text as the store compares it): the
 * store never centres on it, and a row comparing the two would compare a reading with itself. */
function sameQuestion(c: Sighting, s: Sighting): boolean {
  if (c === s || (c.requestId != null && c.requestId === s.requestId)) return true;
  return (
    c.question.trim() === s.question.trim() &&
    c.answerBegins.trim() === s.answerBegins.trim() &&
    c.chatWrapped === s.chatWrapped &&
    (c.system ?? "").trim() === (s.system ?? "").trim()
  );
}

function stationsOf(s: Sighting): number[] {
  const recorded = s.layers.filter((L) => L.lensTop && L.lensTop.length).map((L) => L.idx);
  // a live sighting reads every layer; keep the canvas's nine stations so the words stay legible
  return recorded.length > 12 ? CANVAS_STATIONS.filter((i) => recorded.includes(i)) : recorded;
}

export const Instrument: React.FC<{ route: Route }> = ({ route }) => {
  const sighting = useTheodoliteStore((s) => s.sighting);
  const sel = useTheodoliteStore((s) => s.selectedLayer);
  const setSel = useTheodoliteStore((s) => s.setSelectedLayer);
  const apparatus = useApparatus();
  const [shown, setShown] = useState<Shown>(() => ({
    path: true,
    words: true,
    split: true,
    neurons: apparatus,
    threads: apparatus,
    marks: true,
    rings: true,
  }));
  const stations = useMemo(() => (sighting ? stationsOf(sighting) : []), [sighting]);
  const { lens } = useReadingLens(sighting);

  return (
    <Page route={route} folio="folio i">
      <Question />
      {sighting ? (
        <>
          <WentOn sighting={sighting} />
          <div style={{ display: "flex", flexWrap: "wrap", gap: "40px 56px", paddingTop: 34 }}>
            <section aria-labelledby="inst-head" style={{ flex: "999 1 640px", minWidth: 0, display: "flex", flexDirection: "column", gap: 14 }}>
              <div className="row-between">
                <h2 id="inst-head" className="h-sec">
                  ii · the instrument
                </h2>
                <span style={{ fontSize: 15, color: "#6B5D4D" }}>drag to turn · scroll to draw near</span>
              </div>
              <InstrumentFigure sighting={sighting} sel={sel} stations={stations} shown={shown} onPick={setSel} />
              <fieldset style={{ border: 0, margin: 0, padding: 0, display: "flex", flexWrap: "wrap", alignItems: "center", gap: "2px 22px" }}>
                <legend
                  style={{ float: "left", padding: 0, marginRight: 6, fontVariantCaps: "all-small-caps", letterSpacing: "0.16em", fontSize: 15, color: "#6B5D4D" }}
                >
                  show
                </legend>
                {(
                  [
                    ["path", "the word’s path"],
                    ["words", "words through the lens"],
                    ["split", "gathering & remembering"],
                    ["neurons", "neurons that fired"],
                    ["threads", "threads"],
                    ["marks", "landmarks"],
                    ["rings", "layer rings"],
                  ] as [keyof Shown, string][]
                ).map(([k, label]) => (
                  <label key={k} className="check" style={{ gap: 8 }}>
                    <input type="checkbox" checked={shown[k]} onChange={() => setShown((s) => ({ ...s, [k]: !s[k] }))} />
                    {label}
                  </label>
                ))}
              </fieldset>
              <Rubric style={{ margin: 0, maxWidth: 700 }}>
                Each ring is one layer: the first at the bottom, the sixtieth at the top. The dark thread is the word’s path as the{" "}
                {lens.name} lens reads it, and the words beside it are that lens’s readings. Below layer 21 the study could validate none
                of them, so those words are drawn faint; at layers 21 to 30 trust only how sightings rank against one another, not the words
                themselves. On the gilded rings, every sixth, the model looks back over the whole conversation, not only the recent words.
              </Rubric>
              <WhatIsDrawn sighting={sighting} />
            </section>
            <Leaf sighting={sighting} sel={sel} stations={stations} onPick={setSel} />
          </div>
          <Ascent sighting={sighting} sel={sel} stations={stations} onPick={setSel} />
          <IfApparatus>
            <Kinship sighting={sighting} sel={sel} />
          </IfApparatus>
        </>
      ) : (
        <p className="status-line" style={{ padding: "40px 0" }}>
          Waiting for the first sighting…
        </p>
      )}
    </Page>
  );
};

/* ------------------------------------------------------------ i · the question */

const Question: React.FC = () => {
  const question = useTheodoliteStore((s) => s.question);
  const answerBegins = useTheodoliteStore((s) => s.answerBegins);
  const setQuestion = useTheodoliteStore((s) => s.setQuestion);
  const setAnswerBegins = useTheodoliteStore((s) => s.setAnswerBegins);
  const compareThink = useTheodoliteStore((s) => s.compareThink);
  const setCompareThink = useTheodoliteStore((s) => s.setCompareThink);
  const thinkBare = useTheodoliteStore((s) => s.thinkBare);
  const chat = useTheodoliteStore((s) => s.chat);
  const setChat = useTheodoliteStore((s) => s.setChat);
  const system = useTheodoliteStore((s) => s.system);
  const setSystem = useTheodoliteStore((s) => s.setSystem);
  const takeSighting = useTheodoliteStore((s) => s.takeSighting);
  const isSweeping = useTheodoliteStore((s) => s.isSweeping);
  const sighting = useTheodoliteStore((s) => s.sighting);
  const sightEach = useTheodoliteStore((s) => s.writing.sightEach);
  const shownSet = useShownSet();
  const lastWord = (answerBegins.trim() || question.trim()).split(/\s+/).pop() ?? "";
  // With the chat template on and no begun answer, the text ends with the template's model-turn header, so the
  // sweep reads that closing piece, not the question's last word. Once a matching sighting is in, say exactly what:
  // the piece read where the question ends, whichever word-piece of its answer is shown below.
  const readsTemplate = chat && !answerBegins.trim();
  const matches =
    sighting?.source === "live" &&
    sighting.question.trim() === question.trim() &&
    sighting.answerBegins.trim() === answerBegins.trim() &&
    sighting.chatWrapped === chat;
  const readNow = matches && sighting ? promptEndPiece(sighting, shownSet) : null;
  // the setting for the next sighting, which this hint describes: then each word-piece it writes is read too
  const then = sightEach ? ", then each word-piece it writes" : "";

  return (
    <>
      <section aria-labelledby="q-head" style={{ display: "flex", flexWrap: "wrap", gap: "28px 56px", padding: "34px 0 30px", alignItems: "flex-start" }}>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            // a stray Enter while a sighting is being taken would queue a whole second answer behind it
            if (!useTheodoliteStore.getState().isSweeping) takeSighting();
          }}
          style={{ flex: "999 1 620px", minWidth: 0, display: "flex", flexDirection: "column", gap: 16 }}
        >
          <h2 id="q-head" className="h-sec">
            i · the question
          </h2>
          <div className="q-grid">
            <label htmlFor="ask" className="label-sc q-label q-label-ask">
              you ask
            </label>
            <QuestionBox id="ask" kind="ask" value={question} onChange={setQuestion} />
            <label htmlFor="begin" className="label-sc q-label q-label-begin">
              its answer begins
            </label>
            <div className="q-begin-row">
              <QuestionBox id="begin" kind="begin" value={answerBegins} onChange={setAnswerBegins} aria-describedby="begin-hint" />
              <span id="begin-hint" className="q-hint">
                {readsTemplate ? (
                  <>
                    optional — with the chat template and no begun answer, the instrument reads the end of the template
                    {readNow != null ? (
                      <>
                        ,{" "}
                        <span className="read" title={JSON.stringify(readNow)}>
                          {wp(readNow)}
                        </span>
                      </>
                    ) : null}
                    , not your last word{then}; begin the answer to read a word of it
                  </>
                ) : (
                  <>
                    optional — the instrument reads the last word,{" "}
                    <span className="read" title={readNow != null ? JSON.stringify(readNow) : undefined}>
                      {readNow != null ? wp(readNow) : `“${lastWord}”`}
                    </span>
                    {then}
                  </>
                )}
              </span>
            </div>
            <div className="q-tools-cell">
              <SavedShelf kind="question" />
            </div>
          </div>
          {readsTemplate && !answerSaysTemplateEnd(sighting, shownSet) ? <TemplateEndCaution /> : null}
          <Rubric style={{ margin: "2px 0 0", maxWidth: 640 }}>
            Write anything you like, then press Enter or take a sighting; Shift+Enter begins a new line, and the corner of a box draws it
            taller. The instrument follows the model’s last word up through all sixty of its layers
            {sightEach ? ", and then each word-piece it writes." : ", and shows at each one what the lens reads there."}
          </Rubric>
          <button type="submit" hidden aria-hidden="true" tabIndex={-1} />
        </form>
        <div style={{ flex: "1 1 300px", display: "flex", flexDirection: "column", gap: 6, alignItems: "flex-start" }}>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 12, alignItems: "center", marginBottom: 8 }}>
            <button type="button" className="btn-primary" onClick={() => takeSighting()} disabled={isSweeping}>
              {isSweeping ? "taking a sighting…" : "take a sighting"}
            </button>
            <a href={href.incipit()} style={{ fontSize: 17, padding: "12px 4px" }}>
              try an example
            </a>
          </div>
          <label className="check">
            <input type="checkbox" checked={chat} onChange={(e) => setChat(e.target.checked)} />
            <span>
              send it as a chat <span style={{ fontStyle: "italic" }}>(chat template)</span>
            </span>
          </label>
          {chat ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 4, paddingLeft: 28, width: "100%", boxSizing: "border-box" }}>
              <label htmlFor="system" className="label-sc" style={{ fontSize: 14 }}>
                system prompt <span style={{ fontVariantCaps: "normal", letterSpacing: 0, fontStyle: "italic" }}>(optional)</span>
              </label>
              <QuestionBox id="system" kind="system" value={system} onChange={setSystem} placeholder="none: the chat begins with your question" />
              <SavedShelf kind="system" />
            </div>
          ) : (
            <span style={{ fontSize: 15, color: "#6B5D4D", paddingLeft: 28 }}>
              Off: the words go in as bare text. In the study two of six bare prompts came back as one word repeated.
            </span>
          )}
          {/* bare text has no turn to think in: offered only for a chat, unless the old imitation is chosen in the
              question's apparatus (think_bare "prefix") */}
          <label className="check" style={{ color: "#6B5D4D" }}>
            <input
              type="checkbox"
              checked={compareThink && (chat || thinkBare === "prefix")}
              disabled={!chat && thinkBare !== "prefix"}
              aria-describedby={!chat && thinkBare !== "prefix" ? "think-why" : undefined}
              onChange={(e) => setCompareThink(e.target.checked)}
            />
            also sight it with thinking turned on
          </label>
          {!chat && thinkBare !== "prefix" ? (
            <>
              {/* the reason a disabled control is disabled is always shown, rubrics or none (kit: PendingCheck) */}
              <span id="think-why" style={{ fontSize: 15, color: "#6B5D4D", paddingLeft: 28, fontStyle: "italic" }}>
                Only for a chat: bare text has no turn to think in.
                {sighting ? " The old imitation can be chosen in the question’s apparatus." : null}
              </span>
              <Rubric style={{ margin: "0 0 0 28px", fontSize: 15, maxWidth: 560 }}>
                Only for a chat. Bare text has no turn for the model to think in: Gemma 4 turns thinking on with a mark at the head of
                the system prompt, and begins its answer without the empty thought it writes otherwise. Send it as a chat to sight it
                with thinking.
              </Rubric>
            </>
          ) : !chat && compareThink ? (
            <span style={{ fontSize: 15, color: "#6B5D4D", paddingLeft: 28, fontStyle: "italic" }}>
              With bare text this sends the old imitation chosen in the question’s apparatus, <span className="mono">&lt;|think|&gt;</span>{" "}
              before the text: not Gemma 4’s own switch.
            </span>
          ) : null}
          {!chat && !sighting ? (
            // before the first sighting the question's apparatus is not drawn, so the old imitation is offered here
            <IfApparatus>
              <span style={{ paddingLeft: 28 }}>
                <ThinkBareChoice />
              </span>
            </IfApparatus>
          ) : null}
          {compareThink && chat && answerBegins.trim() ? (
            <span style={{ fontSize: 15, color: "#6B5D4D", paddingLeft: 28, fontStyle: "italic" }}>
              With a begun answer the reply starts straight after the model’s header, before any thought, so this second sighting
              has thinking allowed but none written.
            </span>
          ) : null}
          <SweepError />
          <RecordedNotice />
        </div>
      </section>
      <HowItWrites />
      {sighting ? <QuestionApparatus sighting={sighting} /> : null}
    </>
  );
};

/**
 * The re-read check in words: whether layer 60 of the one re-reading puts the word-piece chosen first. Writing goes
 * step by step and the re-reading is one pass over the whole text, two different 4-bit computations, so in a near-tie
 * the two can part. Drawn by chance, or after a repeat penalty, the piece chosen need not be its likeliest at all.
 */
const RecaptureWords: React.FC<{ s: Sighting }> = ({ s }) => {
  const rc = s.recaptureCheck;
  const c = s.chosen;
  if (!rc || !c) {
    if (s.source === "fixture") return <Missing what="not recorded in the study’s sighting" />;
    if (s.generation?.nWritten === 0) return <>nothing was written, so there is no choice to check.</>;
    return <Missing what="not sent" />;
  }
  const diff =
    rc.chosenLogitDiff != null && c.source === "written"
      ? `; read again, its score differs by ${signed(rc.chosenLogitDiff, 3)} from when it was written`
      : "";
  if (rc.top1IsChosen) {
    return (
      <>
        the re-read put the word it chose, {pieceWord(c.token)}, first at layer 60{diff}.
      </>
    );
  }
  const g = s.generation;
  const drawn = g != null && !g.greedy;
  if (c.source === "given") {
    return (
      <>
        the re-read puts {pieceWord(rc.top1Token)} first at layer 60; the word-piece given here, read again, is {pieceWord(c.token)}
        {drawn ? ", drawn by chance when it was written" : ""}.
      </>
    );
  }
  if (drawn || !!g?.presencePenalty) {
    return (
      <>
        the re-read put {pieceWord(rc.top1Token)} first at layer 60; it chose {pieceWord(c.token)}, {drawn ? "drawn by chance" : "after the repeat penalty"}
        , so the two need not agree{diff}.
      </>
    );
  }
  return (
    <span className="red">
      a near-tie: the re-read put {pieceWord(rc.top1Token)} first at layer 60, not the word it chose, {pieceWord(c.token)}
      {diff}. Writing step by step and reading again in one pass are not computed exactly alike in 4-bit.
    </span>
  );
};

/** How the sighting with thinking turned on was switched on, as the server says it (think_protocol). */
const ThinkProtocolWords: React.FC<{ protocol: "template" | "bare_prefix" | null | undefined }> = ({ protocol }) =>
  protocol === "template" ? (
    <span className="muted">
      It was switched on by the chat template’s own switch (enable_thinking: <span className="mono">&lt;|think|&gt;</span> at the head of
      the system turn, opened for it if there is none; the answer begun with no empty thought).
    </span>
  ) : protocol === "bare_prefix" ? (
    <span className="red">
      It was switched on by the old imitation: <span className="mono">&lt;|think|&gt;</span> set before bare text, not Gemma 4’s switch
      (E6 §3, item 11), so a difference it shows may not be a difference in thinking.
    </span>
  ) : (
    <span className="muted">How it was switched on was not said: an older server, which sent a chat by the template’s switch and bare text by the old imitation.</span>
  );

/** The expert's choice of the old imitation for bare text (think_bare "prefix"), kept in the apparatus with its reason. */
const ThinkBareChoice: React.FC = () => {
  const thinkBare = useTheodoliteStore((s) => s.thinkBare);
  const setThinkBare = useTheodoliteStore((s) => s.setThinkBare);
  return (
    <span style={{ display: "block", marginTop: 6 }}>
      <label className="check">
        <input type="checkbox" checked={thinkBare === "prefix"} onChange={(e) => setThinkBare(e.target.checked ? "prefix" : "none")} />
        <span>
          send the old imitation with bare text (<span className="mono">&lt;|think|&gt;</span> before the text, as first built)
        </span>
      </label>
      <span className="muted" style={{ display: "block", paddingLeft: 28 }}>
        It is not Gemma 4’s switch, so a difference it shows may not be a difference in thinking. Left off, a bare question is sighted
        without thinking, and the request says so (think_bare “none”); the server’s own default, for older callers, is still the
        imitation (“prefix”).
      </span>
    </span>
  );
};

const QuestionApparatus: React.FC<{ sighting: Sighting }> = ({ sighting }) => {
  const set = useShownSet();
  const s = sighting;
  const k = s.tokenIndex;
  const g = s.generation;
  // how its word-pieces were written: a re-reading of an answer already written keeps the record of its first writing
  const wrote = set?.wrote ?? g;
  const given = g?.source === "given";
  const n = answerLength(s, set);
  const max = wrote?.maxNewTokens ?? g?.maxNewTokens ?? null;
  const sentMax = typeof set?.payload.max_new_tokens === "number" ? set.payload.max_new_tokens : null;
  const stop = stopWords(wrote?.stopReason, set?.pieces ?? piecesOf(s));
  const sighted = set ? Object.keys(set.sightings).length : null;
  const capped = (wrote ?? g)?.capped === true;
  // the thinking comparison comes with the first pair only, so the set keeps it for its later word-pieces
  const thinking = set?.thinking ?? thinkingOf(s);
  // asked for, but not written: "stop here" came before the thinking comparison began (the set keeps the flag, since
  // its thinking record stays empty)
  const thinkSkipped = set?.thinkSkipped ?? s.thinkSkipped;
  // asked for, and left out by the server: bare text has no turn to think in (think_bare "none")
  const thinkLeftOut = set?.thinkLeftOut ?? s.thinkLeftOut;
  const answeredFirst = s.chatWrapped && s.answerBegins.trim() !== "";
  return (
    <Apparatus title="the question" style={{ padding: "12px 0 22px" }}>
      <dl className="dl">
        <dt>word-pieces</dt>
        <dd className="num">
          {s.nTokens != null ? `${s.nTokens} read up to here` : <Missing />}
          {s.readAt != null ? (
            <>
              ; reading <span title={JSON.stringify(s.readAt)}>{pieceWord(s.readAt)}</span>
              {s.readPos != null ? ` at position ${s.readPos} (counting from 0)` : ""}
              {k >= 1 ? `, the ${ordinal(k)} word-piece of its own answer` : ""}.
            </>
          ) : (
            "."
          )}
          {k >= 1 && s.promptLen != null ? ` The prompt is ${s.promptLen} of them.` : null}{" "}
          {s.firstIds ? (
            <>
              Start marks <span className="mono">[{s.firstIds.join(", ")}]</span>
              {s.firstIds[0] === 2 && s.firstIds[1] === 2 ? ": the doubled start mark of the instrument before the single-BOS fix." : "."}
            </>
          ) : s.promptTokens ? (
            <>
              First pieces <span className="mono-wrap">{JSON.stringify(s.promptTokens.slice(0, 4))}</span>.
            </>
          ) : null}
        </dd>
        <dt>text sent</dt>
        <dd>
          {s.textSent != null ? (
            <span className="mono-wrap" style={{ whiteSpace: "pre-wrap" }}>
              {JSON.stringify(s.textSent)}
            </span>
          ) : (
            "wrapped as a chat by the study’s lab script (the tokenizer’s chat template, thinking off), then the begun answer; the exact string was not recorded."
          )}
          {k >= 1 ? (
            <>
              ; then its own answer up to here:{" "}
              {s.answerStrs ? (
                <span className="mono-wrap" style={{ whiteSpace: "pre-wrap" }}>
                  {JSON.stringify(s.answerStrs.slice(0, k).join(""))}
                </span>
              ) : (
                <Missing what="not sent" />
              )}
            </>
          ) : null}
        </dd>
        <dt>continuation</dt>
        <dd className="num">
          {generationWords(wrote) ?? (
            <>
              <Missing what="not sent by this server" /> (an older server, which always wrote its likeliest word-piece)
            </>
          )}
          {n != null ? `; ${n} word-piece${n === 1 ? "" : "s"}${max != null ? ` of at most ${max}` : ""}` : max != null ? `; at most ${max} word-pieces` : ""}
          {stop ? ` (${stop})` : ""}
          {given ? "; this sighting read them again from their ids: given, not written" : ""}
          {capped && n != null
            ? `; sighted the first ${Math.min(n, (wrote ?? g)?.maxSighted ?? SIGHT_CAP)} of ${n}`
            : sighted != null && n != null
              ? `; ${sighted} of them sighted`
              : ""}{" "}
          {max != null ? (
            <span className="mono">max_new_tokens={max}</span>
          ) : sentMax != null ? (
            <span className="mono">max_new_tokens={sentMax} as sent</span>
          ) : null}
          {s.timing == null && s.elapsedMs != null ? `; sweep ${comma(s.elapsedMs)} ms` : null}
          {s.peakVramGb != null ? `; peak memory ${s.peakVramGb.toFixed(1)} GB` : null}
        </dd>
        <dt>re-read check</dt>
        <dd>
          <RecaptureWords s={s} />
        </dd>
        <dt>chat template</dt>
        <dd>
          {s.source === "fixture"
            ? "This sighting was wrapped as a chat by the study’s lab script (the tokenizer’s chat template, thinking off), then the begun answer."
            : s.chatWrapped
              ? (
                  <>
                    on: the tokenizer’s chat template, thinking off (an empty thought channel), the question as the user’s turn
                    {s.system ? (
                      <>
                        {" "}after a system turn, <span className="italic">“{s.system}”</span>
                      </>
                    ) : (
                      ", with no system turn"
                    )}
                    , then the begun answer after the model’s turn header, as the study’s lab script builds it (labkit.chat). One start mark.
                  </>
                )
              : "off: the words go in as bare text, and two of six plain prompts in the study came back as one word repeated (REPORT §3.4, bug 5)."}
        </dd>
        <dt>thinking</dt>
        <dd>
          {thinking ? (
            <>
              with thinking turned on, {answeredFirst ? "it went on" : "its thought began"}{" "}
              <span className="italic">“{thinking.continuation}”</span>
              {thinking.stopReason === "stopped" ? (
                <>
                  {" "}
                  and was cut short there by “stop here”, so the thought is unfinished
                </>
              ) : null}
              ; that sighting is read only where it begins
              {thinking.readAt != null ? (
                <>
                  , at <span title={JSON.stringify(thinking.readAt)}>{pieceWord(thinking.readAt)}</span>
                </>
              ) : null}
              , never word-piece by word-piece
              {wrote && !wrote.greedy && thinking.seed != null ? <>, and was drawn with the same settings and seed ({thinking.seed})</> : null}.
              {answeredFirst ? (
                <span className="muted">
                  {" "}
                  The begun answer came straight after the model’s header, so the model answered before any thought.
                </span>
              ) : null}{" "}
              <ThinkProtocolWords protocol={thinking.protocol} />
            </>
          ) : thinkSkipped ? (
            <span className="muted">
              asked for, but not written: “stop here” was asked before the sighting with thinking turned on began, so it was left out
            </span>
          ) : thinkLeftOut === "bare_text" ? (
            <span className="muted">
              left out: bare text has no turn to think in, so the server wrote no sighting with thinking turned on (think_left_out
              “bare_text”)
            </span>
          ) : (
            <span className="muted">not sighted with thinking turned on</span>
          )}
          <ThinkBareChoice />
        </dd>
        <dt>matched control</dt>
        <dd style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: "4px 14px" }}>
          <ControlSighting />
          <span style={{ color: "#6B5D4D" }}>
            The study reads lens outputs against a matched control (a nameless or background condition), never alone (REPORT §3.5). Take the
            control first, keep it, then take the sighting to compare; each layer’s apparatus shows both.
          </span>
        </dd>
      </dl>
    </Apparatus>
  );
};

/**
 * It went on: the answer as word-pieces to choose from (WordsItWrote places the line naming the moment shown, how the
 * answer was written, and the red lines on what two word-pieces of one answer can and cannot be compared for), then
 * the times. A pair's total_ms runs from the request to that pair, so the whole sweep is the last one's, once all are in.
 */
const WentOn: React.FC<{ sighting: Sighting }> = ({ sighting }) => {
  const { lens } = useReadingLens(sighting);
  const set = useShownSet();
  const t = sighting.timing;
  const totals = set
    ? Object.values(set.sightings)
        .map((x) => x.timing?.totalMs ?? null)
        .filter((v): v is number => v != null)
    : [];
  const whole = set?.done && totals.length ? Math.max(...totals) : null;
  return (
    <div className="went-on" style={{ paddingBottom: 18, borderBottom: "1px solid var(--hair-soft)" }}>
      <span className="label-sc">it went on</span>
      <WordsItWrote />
      {sighting.isNudged ? (
        <a href={href.nudge()} className="sc" style={{ fontSize: 15, letterSpacing: "0.14em" }}>
          nudged
        </a>
      ) : null}
      <span className="went-facts">
        {t?.analyseMs != null ? (
          <>
            60 layers surveyed · this word-piece analysed in {ms(t.analyseMs)}
            {whole != null ? ` · the whole sweep ${ms(whole)}` : t.totalMs != null ? ` · ${ms(t.totalMs)} from the request to this one` : ""}
          </>
        ) : (
          <>60 layers surveyed{sighting.elapsedMs != null ? ` in ${ms(sighting.elapsedMs)}` : ""}</>
        )}{" "}
        · {lensLabel(lens)}
        {sighting.source === "fixture" ? " · the study’s recorded sighting, taken through the same lens" : ""}
      </span>
    </div>
  );
};

/* ------------------------------------------------------- apparatus: what is drawn */

/** the threads drawn on the rings: the strongest, as the canvas drew them */
export const THREADS_DRAWN = 12;

/** The bands of layers (counted 1-60) the live check of the lens against this answer is counted in; layer 60 is left
 * out, where no lens stands between and its first word-piece is the model's own likeliest. */
const ANSWER_BANDS: [number, number][] = [
  [1, 20],
  [21, 30],
  [31, 40],
  [41, 50],
  [51, 59],
];

/**
 * The study's own figure for the same check inside the model's answers: E3's top-1 agreement of the J++ lens and of the
 * naked eye with the final layer, at the positions of the model's own greedy chat answers ("assistant" rows,
 * experiments/results/E3_lens_validation_REPORT.md:92-101). Report layers L30-39, L40-49, L50-58 are app layers 31-40,
 * 41-50, 51-59 (REPORT.md:35-36). E3 gives no top-1 by band below report layer 30.
 */
const E3_IN_ANSWERS: Record<string, { lens: number; eye: number }> = {
  "31–40": { lens: 0.027, eye: 0.049 },
  "41–50": { lens: 0.155, eye: 0.253 },
  "51–59": { lens: 0.352, eye: 0.402 },
};

/**
 * The lens against this answer: for each band of layers, how many readings (one word-piece sighted, at one layer) had
 * the lens's first word-piece equal to the word-piece chosen there, and the same for the naked eye. Matched by token id
 * only (jpp_top_ids, naive_top_id); a reading whose ids were not sent is not counted. Only readings inside the answer
 * are counted (readsInsideAnswer), the class of positions E3's row is measured at: the reading where the chat template
 * or a bare-text prompt ends has no place in it, and is left out and said to be.
 */
function lensAgainstAnswer(set: AnswerSet) {
  const withChoice = Object.values(set.sightings).filter((x) => x.chosen != null);
  const sighted = withChoice.filter((x) => readsInsideAnswer(x));
  const rows = ANSWER_BANDS.map(([a, b]) => {
    let lens = 0;
    let lensOf = 0;
    let eye = 0;
    let eyeOf = 0;
    for (const x of sighted) {
      const id = x.chosen!.id;
      for (let no = a; no <= b; no++) {
        const L = x.layers[no - 1];
        const top = L?.lensTopIds?.[0];
        if (top != null) {
          lensOf++;
          if (top === id) lens++;
        }
        if (L?.eyeTopId != null) {
          eyeOf++;
          if (L.eyeTopId === id) eye++;
        }
      }
    }
    return { band: `${a}–${b}`, lens, lensOf, eye, eyeOf };
  });
  return { n: sighted.length, left: withChoice.length - sighted.length, rows };
}

/** Where the reading left out of the count was taken: the end of the text the model was given, before its answer. */
const leftWhere = (set: AnswerSet): string =>
  set.chat ? "where the chat template ends" : set.answerBegins.trim() ? "where the begun answer, sent as bare text, ends" : "where your question ends";

const LensAgainstAnswer: React.FC<{ set: AnswerSet }> = ({ set }) => {
  const { n, left, rows } = lensAgainstAnswer(set);
  if (!n && left) {
    return (
      <span>
        Only the reading {leftWhere(set)} is sighted, and the study’s row has no such position: it counts readings inside the model’s
        own answers. Sight a word-piece the model wrote to compare.
      </span>
    );
  }
  if (!n || rows.every((r) => r.lensOf === 0)) {
    return <Missing what="the word-pieces of a live answer, with their token ids" />;
  }
  const wrote = set.wrote ?? set.generation;
  const drawn = wrote?.greedy === false;
  const penalised = (wrote?.presencePenalty ?? 0) !== 0;
  const cell: React.CSSProperties = { textAlign: "right", padding: "4px 0 4px 14px", whiteSpace: "nowrap" };
  const head: React.CSSProperties = { ...cell, fontWeight: 400, color: "#6B5D4D", fontVariantCaps: "all-small-caps", letterSpacing: "0.12em" };
  return (
    <>
      <div style={{ overflowX: "auto" }}>
        <table style={{ borderCollapse: "collapse", fontSize: 15, fontVariantNumeric: "lining-nums tabular-nums", minWidth: 420 }}>
          <thead>
            <tr style={{ borderBottom: "1px solid #D9CFBD" }}>
              <th scope="col" style={{ ...head, textAlign: "left", paddingLeft: 0 }}>
                layers
              </th>
              <th scope="col" style={head}>
                the lens
              </th>
              <th scope="col" style={head}>
                the naked eye
              </th>
              <th scope="col" style={head}>
                the study, in answers (E3)
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const study = E3_IN_ANSWERS[r.band];
              return (
                <tr key={r.band} style={{ borderBottom: "1px solid #ECE4D5" }}>
                  <th scope="row" style={{ ...cell, textAlign: "left", paddingLeft: 0, fontWeight: 400 }}>
                    {r.band}
                  </th>
                  <td style={cell}>{r.lensOf ? `${r.lens} of ${r.lensOf}` : "—"}</td>
                  <td style={cell}>{r.eyeOf ? `${r.eye} of ${r.eyeOf}` : <Missing what="not sent" />}</td>
                  <td style={cell}>{study ? `${pct(study.lens)} · ${pct(study.eye)}` : "not measured by band"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <span>
        Each count is of readings, one word-piece sighted at one layer: how often the first word-piece through the lens, and through
        the naked eye, was the word-piece the model chose there, matched by id. {n} word-piece{n === 1 ? "" : "s"} of this answer
        only, sighted so far: a live check of the kind the study made (top-1 agreement; E3, and R1 on the model’s own replies), but
        on one answer, so it says little alone.
        {left
          ? ` The reading ${leftWhere(set)} is not counted: the study’s row has no such position.`
          : ""}{" "}
        The study’s figures are the share for the lens · the naked eye inside the model’s own greedy chat answers, against its final
        layer’s likeliest word-piece, which there is the word-piece it wrote
        {drawn
          ? "; drawn by chance, the word-piece it chose need not be its likeliest, so these counts are not like for like"
          : penalised
            ? "; after a repeat penalty, the word-piece it chose need not be its likeliest, so these counts are not like for like"
            : ""}
        .{set.chat ? "" : " The study’s figures are from chat answers; this answer was written without the chat template."}
      </span>
    </>
  );
};

const WhatIsDrawn: React.FC<{ sighting: Sighting }> = ({ sighting }) => {
  const pf = sighting.layers.map((L) => L.projFrac).filter((v): v is number => v != null);
  const drawn = sighting.threads.slice(0, THREADS_DRAWN);
  const srcLayers = drawn.map((t) => t.src.idx + 1);
  const F = describeFrame(sighting);
  const poleRanking = useTheodoliteStore((s) => s.poleRanking);
  const setPoleRanking = useTheodoliteStore((s) => s.setPoleRanking);
  const threadScore = useTheodoliteStore((s) => s.threadScore);
  const setThreadScore = useTheodoliteStore((s) => s.setThreadScore);
  const setReadingPreset = useTheodoliteStore((s) => s.setReadingPreset);
  const frameNow = useTheodoliteStore((s) => s.frame);
  const neuronSquaring = useTheodoliteStore((s) => s.neuronSquaring);
  // which preset the reading options now match, centring aside (setReadingPreset keeps it)
  const onPreset = (r: ReadingOptions) =>
    frameNow.squaring === r.frame.squaring &&
    frameNow.perLayer === r.frame.perLayer &&
    frameNow.fnw === r.frame.fnw &&
    neuronSquaring === r.neuronSquaring &&
    poleRanking === r.poleRanking &&
    threadScore === r.threadScore;
  const readWith = (p: "as_built" | "mended") => {
    setReadingPreset(p);
    retakeIfLive();
  };
  // the options are read again from the shown answer's ids, which waits until the server is free: while it is busy, a
  // change would set the page without reading anything again, so they wait too
  const isSweeping = useTheodoliteStore((s) => s.isSweeping);
  const set = useShownSet();
  const expl = sighting.layers.map((L) => L.ffnExplained).filter((v): v is number => v != null);
  const direct = sighting.threadScore === "direct";
  // the layers where a tracked rank was taken from the next position (the server's tracked_from_next flags)
  const fromNext = sighting.layers.filter((L) => L.rankFromNext && Object.values(L.rankFromNext).some(Boolean)).map((L) => L.no);
  const waitNote = isSweeping ? "Wait for the sighting being taken to finish to change this." : undefined;
  return (
    <Apparatus title="what is drawn" label="Apparatus for the instrument" sources={false}>
      <dl className="dl">
        <dt>survey frame</dt>
        <dd>
          {F.frameFormula}; {F.notes.join("; ")}.{" "}
          {!F.poleMotif && !F.lowdin ? "A Löwdin frame, the interface’s default, squares the words all at once. " : null}
          <a href={href.landmarks()}>landmarks</a>
        </dd>
        <dt>path</dt>
        <dd>
          {F.state} projected on the frame; drawn by direction (θ, φ) only, its radius schematic.{" "}
          {pf.length ? (
            <>
              The frame catches {(Math.min(...pf) * 100).toFixed(1)}–{(Math.max(...pf) * 100).toFixed(1)}% of {F.denom} ({F.denomWords}) in
              this sighting{F.studyComparable ? "; the study found 2.4–3.9% by stretch of layers (V7)" : ""}.
            </>
          ) : (
            <>
              The share of {F.denom} the frame catches: <Missing what="not sent by the server" />
            </>
          )}
        </dd>
        <dt>gathering · remembering</dt>
        <dd>
          the ring round the chosen point: lapis Δ<sub>attn</sub> = s·(mid − pre)·J<sup>⊤</sup>, gilt Δ<sub>ffn</sub> = (post − s·mid)·J
          <sup>⊤</sup>, as shares of their sum (V7-corrected, with the layer scalar s).
        </dd>
        <dt>neurons</dt>
        <dd>
          the fifteen GeGLU poles listed at the chosen layer,{" "}
          {sighting.poleRanking === "write" ? (
            <>
              the top fifteen by write size |a| · ‖w<sub>down</sub> ⊙ w<sub>post</sub>‖
            </>
          ) : (
            "the top fifteen by |a| = |mlp_act|"
          )}
          ; filled the positive pole, hollow the negative (after Cammarata 2026); size by |a|, placement on the ring schematic.
          {expl.length ? (
            <>
              {" "}Together they explain {(Math.min(...expl) * 100).toFixed(0)}–{(Math.max(...expl) * 100).toFixed(0)}% of a layer’s feed-forward
              write in this sighting (⟨Σ c<sub>i</sub>, f⟩ / ⟨f, f⟩, E3).
            </>
          ) : null}{" "}
          Ranked either way, the fifteen explained 10–41% of the feed-forward write at the layers E3 sampled from 21 to 59
          (21, 31, 41, 51, 59), and 70–74% at layers 3 and 11; E3 calls this indicative (12 prompts, last word-piece).
          <label className="check" style={{ marginTop: 4 }} title={waitNote}>
            <input
              type="checkbox"
              checked={poleRanking === "write"}
              disabled={isSweeping}
              onChange={(e) => {
                setPoleRanking(e.target.checked ? "write" : "activation");
                retakeIfLive();
              }}
            />
            <span>
              rank them by write size, |a| · ‖w<sub>down</sub> ⊙ w<sub>post</sub>‖ (untick for |a|, as first built)
            </span>
          </label>
          {poleRanking !== sighting.poleRanking ? (
            <span className="muted italic">
              The sighting shown ranks them {sighting.poleRanking === "write" ? "by write size" : "by |a|"}; the next one taken will rank them as
              set here.
            </span>
          ) : null}
        </dd>
        <dt>threads</dt>
        <dd>
          the {drawn.length} highest-scoring{sighting.threadCount != null ? ` of ${comma(sighting.threadCount)}` : ""}:{" "}
          {direct ? (
            <>
              D = what the source adds to the reader’s gate through the stream alone, from its real activation, the layer scalars and both
              exact RMSNorm Jacobians; gilt where D &gt; 0, dashed where D &lt; 0
            </>
          ) : (
            <>
              C = ⟨σ<sub>i</sub> w<sub>down</sub>, w<sub>gate</sub>⟩ · |a|, gilt where C &gt; 0, dashed where C &lt; 0
            </>
          )}
          {srcLayers.length ? (
            <>
              ; they start in layers {Math.min(...srcLayers)}–{Math.max(...srcLayers)}
              {!direct && Math.max(...srcLayers) <= 2 ? ", where the study found the real effects tiny" : ""}
            </>
          ) : null}
          .{" "}
          {direct
            ? "A direct-path linear estimate, not a verified circuit: ρ ≈ 0.87 with the measured effect, where C reaches 0.41 (REPORT §3.4, bug 4)."
            : "A weak guide (ρ ≈ 0.41; a direct-path estimate reaches 0.87)."}{" "}
          Threads join the fifteen listed at each layer, so they follow this ranking; V7 measured the direct-path score’s 0.87 on
          threads among the fifteen ranked by |a|.
          <label className="check" style={{ marginTop: 4 }} title={waitNote}>
            <input
              type="checkbox"
              checked={threadScore === "direct"}
              disabled={isSweeping}
              onChange={(e) => {
                setThreadScore(e.target.checked ? "direct" : "coupling");
                retakeIfLive();
              }}
            />
            score threads by the direct-path estimate
          </label>
          {threadScore !== sighting.threadScore ? (
            <span className="muted italic">
              The sighting shown scores them {direct ? "by the direct-path estimate" : "by C"}; the next one taken will score them as set here.
            </span>
          ) : null}
        </dd>
        <dt>ranks</dt>
        <dd>
          each tracked word’s rank of {comma(VOCAB)} at every layer, read at this position (each layer’s apparatus lists them).{" "}
          {fromNext.length ? (
            <>
              At {fromNext.length === 60 ? "every layer" : `layer${fromNext.length === 1 ? "" : "s"} ${layerList(fromNext)}`} some of the
              lens’s ranks were taken from the next position, by the instrument’s old space-then-digit rule (when the word-piece chosen
              was a bare space, the better of this position’s rank and the next’s): it favours the lens over the naked eye (E6, item
              9). Each layer’s apparatus marks which.
            </>
          ) : sighting.rankLookahead === false ? (
            "Each is read at this position only: the instrument’s old space-then-digit rule, which took a lens rank from the next position when the word-piece chosen was a bare space, is off."
          ) : sighting.rankLookahead === true && sighting.layers.some((L) => L.rankFromNext != null) ? (
            "The instrument’s old space-then-digit rule, which takes a lens rank from the next position when the word-piece chosen is a bare space, was allowed and did not take any here."
          ) : (
            "This sighting does not say whether a rank was taken from the next position, as the instrument’s old space-then-digit rule did when the word-piece chosen was a bare space (E6, item 9)."
          )}
        </dd>
        <dt>the lens against this answer</dt>
        <dd>{set ? <LensAgainstAnswer set={set} /> : <Missing what="needs a live answer, its word-pieces sighted" />}</dd>
        <dt>all the reading options</dt>
        <dd style={{ display: "flex", flexDirection: "column", gap: 6, alignItems: "flex-start" }}>
          <span style={{ display: "flex", flexWrap: "wrap", gap: "8px 12px" }}>
            <button
              type="button"
              className="btn-ghost"
              aria-pressed={onPreset(READING_AS_BUILT)}
              disabled={isSweeping}
              title={waitNote}
              onClick={() => readWith("as_built")}
            >
              read with the reading options as first built (Gram–Schmidt on raw rows, |a|, C)
            </button>
            <button
              type="button"
              className="btn-ghost"
              aria-pressed={onPreset(READING_MENDED)}
              disabled={isSweeping}
              title={waitNote}
              onClick={() => readWith("mended")}
            >
              read with the mends (the default)
            </button>
          </span>
          <span className="muted italic">
            The study’s readings were taken with these options as first built. E6’s sightings and the recorded one were also
            taken before the V7 fixes, their chat prompts with a doubled start mark, so these options alone do not reproduce
            them; V7’s own figures were taken with one start mark, as the instrument now sends.
          </span>
        </dd>
      </dl>
    </Apparatus>
  );
};

/** “3–5, 9, 12–14”: layer numbers, runs written as ranges. */
function layerList(nos: number[]): string {
  const out: string[] = [];
  for (let i = 0; i < nos.length; ) {
    let j = i;
    while (j + 1 < nos.length && nos[j + 1] === nos[j] + 1) j++;
    out.push(j > i ? `${nos[i]}–${nos[j]}` : String(nos[i]));
    i = j + 1;
  }
  return out.join(", ");
}

/* ---------------------------------------------------------------- iii · the leaf */

function poleNote(s: Sighting, L: LayerReading) {
  // the first listed in the sighting's own ranking (the loudest by |a|, or the biggest writer): its words are read, and
  // its rank among the layer's 21,504 is exact
  const p = leadOf(L);
  if (!p) return null;
  // the label gate decides first (lib/labels.ts): a held-back label's words, and any gloss of them, leave the main
  // reading for the apparatus; the canvas's recorded notes never override it
  const gate = labelGate(layerIdx(L.idx), p.check);
  const recorded = s.source === "fixture" ? RECORDED_POLE_NOTES[`${L.idx}:${p.neuron}`] : undefined;
  const lens = p.lens?.slice(0, 3) ?? null;
  const eye = p.eye?.slice(0, 3) ?? null;
  let auto = "";
  if (lens && eye) {
    // the ◆ count of the words shown, spelt exactly; the gate's own count is ten a side, by id
    const shown = sharedShown(
      lens.map((t) => t.token),
      eye.map((t) => t.token),
    );
    const rec = overlapLine(gate);
    const ten =
      gate.sharedTop10 != null
        ? `; counted ten a side, by word-piece, they share ${gate.sharedTop10} of their first ten${rec ? ` (${rec})` : ""}`
        : "";
    auto = shown.k
      ? `Among the words shown, the lens and the naked eye both read ${shown.shared.map((t) => `“${t.trim() || t}”`).join(" and ")}${ten}.`
      : `Among the words shown, the lens and the naked eye share no word-piece${ten}.`;
  }
  return {
    p,
    gate,
    words: lens,
    eye,
    gloss: recorded ? recorded.gloss(gate) : auto,
    canvasTrust: recorded?.canvasTrust ?? null,
    // what the study found at this layer, or, with an edit in force on the neuron, that its words echo the edit
    trust: gateTrustLine(gate),
  };
}

const Leaf: React.FC<{ sighting: Sighting; sel: number; stations: number[]; onPick: (idx: number) => void }> = ({
  sighting,
  sel,
  stations,
  onPick,
}) => {
  const L = sighting.layers[sel];
  const recordedSet = sighting.layers.filter((x) => x.lensTop).map((x) => x.idx);
  // step between the layers whose reading was recorded (every layer in a live sighting)
  const steps = recordedSet.length ? recordedSet : stations;
  const prev = [...steps].reverse().find((i) => i < sel);
  const next = steps.find((i) => i > sel);
  const k = kindOf(L.isGlobal);
  const top = L.lensTop?.[0]?.prob ?? null;
  const gather = L.ffnShare != null ? Math.round((1 - L.ffnShare) * 100) : null;
  const note = poleNote(sighting, L);
  const { lens } = useReadingLens(sighting);
  const set = useShownSet();
  const n = answerLength(sighting, set);
  // at layer 60 no lens stands between: name the word-piece chosen there (by id) apart from its likeliest
  const own = sel === 59;
  const chosen = chosenOf(sighting);
  const rows = L.lensTop?.slice(0, 5) ?? null;
  const chosenListed = own && chosen != null && rows != null && rows.some((_, i) => isChosenRow(sighting, L, i));
  // a stop mark by which one it is: only <turn|> ends its turn (endedWords)
  const chose = chosen?.endMark
    ? endsTurn(chosen.id, chosen.token)
      ? "it ended its turn here"
      : endedWords(chosen.id, chosen.token)
    : "the word it chose";
  const drawnAt = chosen?.samplerProb != null ? ` · ${pct(chosen.samplerProb)} under the sampler` : "";
  const roleOf = (i: number, token: string): string => {
    if (!own) return gloss(token);
    if (chosen != null && isChosenRow(sighting, L, i)) return i === 0 ? `its likeliest, and ${chose}${drawnAt}` : `${chose}${drawnAt}`;
    return i === 0 ? "its likeliest" : gloss(token);
  };
  const g = sighting.generation;
  const sampled = g != null && g.source !== "recorded" && !g.greedy;
  return (
    <aside aria-labelledby="leaf-head" style={{ flex: "1 1 400px", minWidth: 0, display: "flex", flexDirection: "column", gap: 22 }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12 }}>
          <h2 id="leaf-head" className="h-sec">
            iii · the leaf
          </h2>
          <div style={{ display: "flex", gap: 4 }}>
            <button type="button" className="btn-ghost plain" onClick={() => prev !== undefined && onPick(prev)} disabled={prev === undefined} aria-label="Earlier layer">
              ↓ earlier
            </button>
            <button type="button" className="btn-ghost plain" onClick={() => next !== undefined && onPick(next)} disabled={next === undefined} aria-label="Later layer">
              later ↑
            </button>
          </div>
        </div>
        <p className="leaf-at" aria-live="polite">
          {momentWords(sighting)}
          {n != null && n > 1 ? ` · the sighting of word-piece ${sighting.tokenIndex + 1} of ${n}` : ""}
        </p>
      </div>

      <div style={{ display: "flex", alignItems: "flex-end", gap: 20, borderBottom: "1px solid #D9CFBD", paddingBottom: 16 }}>
        <div className="lining" style={{ fontSize: 112, lineHeight: 0.8, fontWeight: 500, color: "#8A6A2C", letterSpacing: "-0.02em" }}>
          {L.no}
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 2, paddingBottom: 4 }}>
          <span style={{ fontVariantCaps: "all-small-caps", letterSpacing: "0.2em", fontSize: 15, color: "#6B5D4D" }}>layer {L.no} of sixty</span>
          <span style={{ fontSize: 17 }}>{k.kind}</span>
          <span style={{ fontSize: 15, fontStyle: "italic", color: "#6B5D4D" }}>{k.gloss}</span>
        </div>
      </div>

      <div className="stack">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12 }}>
          <h3 className="h-sub">{own ? "its own odds" : "through the lens"}</h3>
          <span style={{ fontSize: 15, fontStyle: "italic", color: "#6B5D4D" }}>{certainty(top, own)}</span>
        </div>
        {rows ? (
          rows.map((t, i) => (
            <div
              key={i}
              style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) 64px", gap: "2px 14px", alignItems: "baseline", padding: "6px 0 8px", borderBottom: "1px solid #ECE4D5" }}
            >
              <div style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: "2px 12px", minWidth: 0 }}>
                <span style={{ fontSize: 25, lineHeight: 1.2 }} title={JSON.stringify(t.token)}>
                  {wp(t.token)}
                </span>
                <span style={{ fontSize: 15, fontStyle: "italic", color: "#6B5D4D" }}>{roleOf(i, t.token)}</span>
              </div>
              <span className="num" style={{ textAlign: "right", fontSize: 16 }}>
                {t.prob != null ? pct(t.prob) : <Missing what="—" />}
              </span>
              <div className="bar-track" style={{ gridColumn: "1 / -1" }}>
                <div className="bar-fill" style={{ width: `${Math.max(0.6, (t.prob ?? 0) * 100)}%` }} />
              </div>
            </div>
          ))
        ) : (
          <p style={{ margin: 0, fontSize: 16 }}>
            <Missing what="not recorded at this layer in the study’s sighting" />{" "}
            <span className="muted">The recording kept the lens’s words at the marked layers only; choose one of them, or take a live sighting.</span>
          </p>
        )}
        {own && chosen != null && rows != null && !chosenListed ? (
          // the piece chosen lies outside the five likeliest: its rank and share are the writing step's own (or, for a
          // word-piece given and read again, the re-reading's), not this readout's
          <div
            style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) 64px", gap: "2px 14px", alignItems: "baseline", padding: "6px 0 8px", borderBottom: "1px solid #ECE4D5" }}
          >
            <div style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: "2px 12px", minWidth: 0 }}>
              <span style={{ fontSize: 25, lineHeight: 1.2 }} title={JSON.stringify(chosen.token)}>
                {pieceWord(chosen.token)}
              </span>
              <span style={{ fontSize: 15, fontStyle: "italic", color: "#6B5D4D" }}>
                {chose}, outside the five above
                {chosen.modelRank != null ? `: rank ${comma(chosen.modelRank)} of ${comma(VOCAB)}` : ""}
                {chosen.source === "given" ? " as read again" : " as it wrote it"}
                {drawnAt}
              </span>
            </div>
            <span className="num" style={{ textAlign: "right", fontSize: 16 }}>
              {chosen.modelProb != null ? pct(chosen.modelProb) : <Missing what="—" />}
            </span>
            <div className="bar-track" style={{ gridColumn: "1 / -1" }}>
              <div className="bar-fill" style={{ width: `${Math.max(0.6, (chosen.modelProb ?? 0) * 100)}%` }} />
            </div>
          </div>
        ) : null}
        <Rubric style={{ margin: "4px 0 0", fontSize: 16 }}>
          {own ? (
            <>
              At the last layer no lens stands between: these are the model’s own odds for its next word
              {sampled ? ", before the sampler" : g?.presencePenalty ? ", before the repeat penalty" : ""}.{" "}
            </>
          ) : (
            <>
              These are the words read from this layer’s stream through the {lens.name} lens: its guess at what the model would say
              if it stopped here.{" "}
            </>
          )}
          <a href={href.lexicon("lens")}>What is a lens?</a>
        </Rubric>
        <LensCaution sighting={sighting} no={L.no} style={{ margin: 0, fontSize: 16 }} />
      </div>

      <div className="stack">
        <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-between", alignItems: "baseline", gap: "2px 12px" }}>
          <h3 className="h-sub">gathering or remembering</h3>
          <span style={{ fontSize: 15, fontStyle: "italic", color: "#6B5D4D" }}>as the {lens.name} lens weighs the two writes</span>
        </div>
        {gather != null ? (
          <>
            <div
              role="img"
              aria-label={`${gather} percent gathered from earlier words, ${100 - gather} percent recalled from memory, as the lens weighs them`}
              style={{ display: "flex", height: 14, borderRadius: 1, overflow: "hidden" }}
            >
              <div style={{ width: `${gather}%`, background: "#2E4A7D" }} />
              <div style={{ width: `${100 - gather}%`, background: "#A07A2E" }} />
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 16, fontSize: 16 }}>
              <span>
                <span className="num" style={{ color: "#2E4A7D", fontWeight: 600 }}>
                  {gather}%
                </span>{" "}
                gathered from the words before <span style={{ color: "#6B5D4D", fontStyle: "italic" }}>(attention)</span>
              </span>
              <span style={{ textAlign: "right" }}>
                <span className="num" style={{ color: "#7A5C28", fontWeight: 600 }}>
                  {100 - gather}%
                </span>{" "}
                recalled from memory <span style={{ color: "#6B5D4D", fontStyle: "italic" }}>(feed-forward)</span>
              </span>
            </div>
          </>
        ) : (
          <Missing />
        )}
      </div>

      <div className="stack" style={{ gap: 8 }}>
        <h3 className="h-sub">{rankingWords(sighting.poleRanking).head}</h3>
        {note && note.words ? (
          <>
            <div style={{ display: "flex", alignItems: "baseline", flexWrap: "wrap", gap: "4px 14px" }}>
              <a href={href.neuron(L.no, note.p.neuron, note.p.face)} className="num" style={{ fontSize: 17 }}>
                neuron {note.p.neuron}
              </a>
              <span style={{ fontSize: 15, color: "#6B5D4D" }}>
                {note.p.face > 0 ? "positive" : "negative"} face · strength {Math.abs(note.p.act).toFixed(1)}
              </span>
            </div>
            {note.gate.heldBack ? (
              // the plate, the reason and the rubric in place of the words, the gloss and the trust line; the apparatus
              // keeps them (and the leaf's apparatus, below, the canvas's own note)
              <HeldBackWords
                gate={note.gate}
                words={note.words.map((t) => t.token)}
                eye={note.eye?.map((t) => t.token) ?? null}
                glosses={note.gloss || null}
                unchecked={sighting.source === "fixture" ? "not yet checked ten a side: this sighting carries three words a side" : "not yet in the server"}
              />
            ) : (
              <>
                <div style={{ fontSize: 22, lineHeight: 1.35 }}>{note.words.map((t) => wp(t.token)).join("  ")}</div>
                {note.gloss ? <div style={{ fontSize: 15, fontStyle: "italic", color: "#6B5D4D" }}>{note.gloss}</div> : null}
                <Rubric style={{ margin: "2px 0 0", fontSize: 15 }}>{note.trust}</Rubric>
              </>
            )}
          </>
        ) : note ? (
          <p style={{ margin: 0, fontSize: 16, color: "#6B5D4D" }}>
            Fifteen neurons {rankingWords(sighting.poleRanking).did} here; {rankingWords(sighting.poleRanking).lead} is{" "}
            <a href={href.neuron(L.no, note.p.neuron, note.p.face)}>neuron {note.p.neuron}</a>. Open the layer’s leaf to read them.
          </p>
        ) : (
          <Missing />
        )}
      </div>

      <div
        style={{ display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: "10px 20px", paddingTop: 14, borderTop: "1px solid #D9CFBD" }}
      >
        <span className="num" style={{ fontSize: 15, color: "#6B5D4D" }}>
          {L.az != null && L.el != null ? `bearing ${minus(String(Math.round(L.az)))}° · elevation ${minus(String(Math.round(L.el)))}°` : <Missing />}
        </span>
        <a href={href.layer(L.no)} style={{ fontSize: 17, padding: "10px 0" }}>
          open this layer’s leaf →
        </a>
      </div>

      <LayerApparatus sighting={sighting} L={L} />
    </aside>
  );
};

const LayerApparatus: React.FC<{ sighting: Sighting; L: LayerReading }> = ({ sighting, L }) => {
  const [raw, setRaw] = useState(false);
  const control = useTheodoliteStore((s) => s.control);
  useEffect(() => setRaw(false), [L.idx]);
  // the leaf's neuron, with its words and the reason, when the label gate holds them back from the leaf
  const lead = poleNote(sighting, L);
  const words = [...sighting.landmarks.filter((w) => L.ranks[w]), ...Object.keys(L.ranks).filter((w) => !sighting.landmarks.includes(w))].slice(0, 8);
  const notRecorded = [
    L.streamStrength == null ? "‖h‖" : null,
    L.rescaleNorm == null ? "the rescale ‖(s − 1)·pre‖" : null,
    L.eyeTop == null ? "the naked eye’s top word-pieces" : null,
    L.projFrac == null ? "the share of the state the frame catches" : null,
  ].filter(Boolean) as string[];
  const set = useShownSet();
  const many = set != null && Object.keys(set.sightings).length > 1;
  const rawObj = {
    layer: L.no,
    index: L.idx,
    token_index: sighting.tokenIndex,
    read_pos: sighting.readPos,
    read_token: sighting.readAt,
    read_token_id: sighting.readAtId,
    chosen: sighting.chosen,
    generation: sighting.generation,
    recapture_check: sighting.recaptureCheck,
    jpp_top5: L.lensTop?.map((t) => [t.token, t.prob]) ?? null,
    jpp_top_ids: L.lensTopIds,
    naive_top: L.eyeTop,
    naive_top_id: L.eyeTopId,
    ffn_dominance: { corrected: L.ffnShare, tool_before_fix: L.ffnShareBeforeFix },
    attn_norm: L.attnNorm,
    ffn_norm: L.ffnNorm,
    layer_scalar: L.layerScalar,
    rescale_norm: L.rescaleNorm,
    residual_norm: L.streamStrength,
    azimuth_deg: L.az,
    elevation_deg: L.el,
    xyz: L.xyz,
    [describeFrame(sighting).exportKey]: L.projFrac,
    tracked_rank: L.ranks,
    rank_from_next: L.rankFromNext,
    poles: L.poles.map((p) => ({
      neuron: p.neuron,
      act: p.act,
      lens: p.lens?.map((t) => t.token) ?? null,
      eye: p.eye?.map((t) => t.token) ?? null,
      // the server's check (null: not sent, or a recorded label not yet checked ten a side), and the page's verdict
      label_check: labelCheckJson(p.check),
      held_back: p.lens || p.eye ? labelGate(layerIdx(L.idx), p.check).heldBack : null,
    })),
  };
  // the kept control is a condition read where its answer begins: it is compared at word-piece 0 only, and only when it
  // is another question (the store never centres a question on itself)
  const first = sighting.tokenIndex === 0;
  const ctl = control && first && !sameQuestion(control, sighting) ? control.layers[L.idx] : null;
  const ck = sighting.centreKind;
  const centring = sighting.frame?.centred === true || ck === "control_reading_same_answer" || ck === "control_prompt_end";
  const laterControl = !first && (control != null || centring);
  return (
    <Apparatus
      title={
        <>
          layer {L.no} <span style={{ letterSpacing: "0.06em" }}>(index {L.idx})</span>
        </>
      }
      label="Apparatus for this layer"
    >
      <dl className="dl tight num">
        <dt>readout</dt>
        <dd>
          logits(h, ℓ) = unembed(h · J<sub>ℓ</sub>
          <sup>⊤</sup>), unembed(u) = 30 · tanh(W<sub>U</sub> · RMSNorm(u) / 30); J = the J++ lens (Ayonrinde &amp; Lindsey 2026), fitted on the
          base model. The Jacobian lens (Gurnee et al. 2026) is on disk; the server does not load it.
        </dd>
        <LensRecord sighting={sighting} />
        <dt>split</dt>
        <dd>
          {L.ffnShare != null ? (
            <>
              remembering {Math.round(L.ffnShare * 100)}% after the V7 fix
              {L.layerScalar != null ? <>, with this layer’s scalar s = {L.layerScalar.toFixed(3)}</> : null}
              {L.ffnShareBeforeFix != null ? <>; the instrument read {Math.round(L.ffnShareBeforeFix * 100)}% before it</> : null}.
              {L.attnNorm != null && L.ffnNorm != null ? (
                <>
                  {" "}
                  ‖Δ<sub>attn</sub>‖ = {L.attnNorm.toFixed(2)}, ‖Δ<sub>ffn</sub>‖ = {L.ffnNorm.toFixed(2)}.
                </>
              ) : null}
            </>
          ) : (
            <Missing />
          )}
        </dd>
        <dt>frame</dt>
        <dd>
          {L.xyz ? <>(x, y, z) = ({L.xyz.map(sci).join(", ")}); </> : null}
          {L.az != null && L.el != null ? (
            <>
              θ = atan2(y, x) = {minus(L.az.toFixed(1))}°, φ = asin(z / ‖p‖) = {minus(L.el.toFixed(1))}°
            </>
          ) : (
            <Missing />
          )}
          {L.projFrac != null ? (
            <>
              ; the frame catches {(L.projFrac * 100).toFixed(1)}% of {describeFrame(sighting).denom}.
            </>
          ) : (
            "."
          )}
        </dd>
        <dt>stream strength</dt>
        <dd>{L.streamStrength != null ? <>‖h‖ = {L.streamStrength.toFixed(1)}</> : <Missing />}</dd>
        <dt>ranks</dt>
        <dd>
          of 262,144, lens / naked eye:{" "}
          {words.length
            ? words.map((w, i) => (
                <span key={w}>
                  {i ? " · " : ""}“{w}” {L.ranks[w].lens != null ? comma(L.ranks[w].lens!) : "—"} /{" "}
                  {L.ranks[w].eye != null ? comma(L.ranks[w].eye!) : <Missing what="not sent" />}
                  {L.rankFromNext?.[w] ? <span className="muted"> (the lens’s rank taken from the next position)</span> : null}
                </span>
              ))
            : <Missing />}
        </dd>
        <dt>poles</dt>
        <dd>
          signed a of the {L.poles.length} listed, ranked by{" "}
          {sighting.poleRanking === "write" ? `write size, ${rankingWords("write").measure}` : "|a|"}:{" "}
          {L.poles.map((p) => signed(p.act)).join("  ")}. Labels: unembed((σ · w<sub>down</sub>) · J
          <sup>⊤</sup>), against the naked eye’s unembed(σ · w<sub>down</sub>). Output labels held up at layers 41–58, were at chance at
          11–31, and were not tested at 1–10 or 32–40 (N1; erratum E9). Labels below layer 41, and labels whose two readings share none
          of their first ten word-pieces, are held back from the leaf (REPORT §3.3, conditions 2 and 3; the cut at none of ten is the
          instrument’s); the server marks each label it sends (label_check), and the words stay here.
        </dd>
        {lead && (lead.p.lens || lead.p.eye) && lead.gate.heldBack ? (
          <>
            <dt>held back</dt>
            <dd>
              neuron {lead.p.neuron}, {lead.p.face > 0 ? "positive" : "negative"} face: {heldTag(lead.gate)}. Through the lens{" "}
              {lead.words ? lead.words.map((t) => `“${t.token}”`).join(" · ") : <Missing what="not read" />}; to the naked eye{" "}
              {lead.eye ? lead.eye.map((t) => `“${t.token}”`).join(" · ") : <Missing what="not read" />}.{" "}
              {sharedLine(lead.gate) ?? (
                <Missing
                  what={sighting.source === "fixture" ? "not yet checked ten a side: this sighting carries three words a side" : "not yet in the server"}
                />
              )}
              {lead.gloss ? <> Its gloss: {lead.gloss}</> : null}
              {lead.canvasTrust ? <> The canvas’s note: {lead.canvasTrust}</> : null} {heldRubric(lead.gate)}
            </dd>
          </>
        ) : null}
        {ctl ? (
          <>
            <dt>control</dt>
            <dd>
              at this layer the lens read the control as {ctl.lensTop?.[0] ? <>{wp(ctl.lensTop[0].token)} ({pct(ctl.lensTop[0].prob)})</> : <Missing what="not recorded" />}
              {ctl.ffnShare != null ? <>; remembering {Math.round(ctl.ffnShare * 100)}%</> : null}.
            </dd>
          </>
        ) : null}
        {laterControl ? (
          <>
            <dt>control</dt>
            <dd>
              {control ? "The kept control is read where it chose its first word-piece; it is matched to word-piece 0 only, so it is not compared here. " : null}
              {centring ? (
                ck === "control_reading_same_answer" ? (
                  "This word-piece is centred on the control reading this very answer: the control’s state at the same word-piece, after its own question."
                ) : ck === "none" ? (
                  "This word-piece is not centred: a control read where its question ends is never subtracted from a later word-piece."
                ) : ck === "control_prompt_end" ? (
                  "This word-piece is centred on the control read where its question ends."
                ) : (
                  <>
                    What it was centred on: <Missing what="not sent" />
                  </>
                )
              ) : null}
            </dd>
          </>
        ) : null}
        {notRecorded.length ? (
          <>
            <dt>not recorded</dt>
            <dd style={{ color: "#6B5D4D" }}>
              for this sighting, {notRecorded.join(", ")}: <Missing />
            </dd>
          </>
        ) : null}
      </dl>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "8px 12px", paddingTop: 4 }}>
        <button type="button" className="btn-ghost" onClick={() => setRaw((r) => !r)} aria-expanded={raw}>
          {raw ? "hide the raw readout" : "show the raw readout"}
        </button>
        <button type="button" className="btn-ghost" onClick={() => exportSighting(sighting)}>
          {many ? "export this word-piece’s sighting (.json)" : "export the sighting (.json)"}
        </button>
        {many && set ? (
          <button type="button" className="btn-ghost" onClick={() => exportAnswer(set)}>
            export every word-piece’s sighting (.json)
          </button>
        ) : null}
      </div>
      {raw ? <pre className="raw">{JSON.stringify(rawObj, null, 2)}</pre> : null}
      <p className="muted" style={{ margin: 0 }}>
        {sighting.source === "fixture" ? (
          <>
            Recorded in <span className="mono-wrap">experiments/results/E6_tool_audit_02_sweep_demo.json</span> (key prepend_bos=True); layer
            scalars from E6_tool_audit_01_basics.json.
          </>
        ) : (
          "From the running server, as it sent this sighting."
        )}
      </p>
    </Apparatus>
  );
};

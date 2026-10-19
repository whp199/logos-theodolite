import React from "react";
import { Apparatus, Missing } from "./kit";
import { NotAMatchedPair, OneDrawLine, PenaltyLine, TemplateEndCaution } from "../lib/cautions";
import { ms, pct, pieceWord } from "../lib/words";
import {
  CONNECTION_CLOSED,
  PACE,
  SIGHT_CAP,
  chosenPhrase,
  endMarkWords,
  exportAnswer,
  piecesOf,
  readWhere,
  sightPieceToo,
  useTheodoliteStore,
  type AnswerPiece,
  type AnswerSet,
  type Sighting,
} from "../store/useTheodoliteStore";

/**
 * The words it wrote: the answer as word-pieces, each one the sighting taken where the model chose it. Choosing a
 * piece shows its sighting (the stream there had just read the piece before it). The begun answer comes first, in
 * muted ink, not to be chosen. The selected piece is underlined in gilt, the others with a dotted hair; pieces still
 * being sighted are faint, pieces written but not sighted are plain text. Under it, a line naming the moment shown,
 * how the answer was written, and while a request is in flight how far it has got, with "stop here". The recorded
 * sighting and an older server's reply keep the one italic line they always had.
 */

/** The words endMarkWords gives the mark that ends the model's turn (<turn|>), and only that one. */
const TURN_ENDS = endMarkWords(106, "<turn|>");

/** Whether a stop mark is the one that ends its turn (<turn|>), not the end of the text or a control mark. */
const endsTurn = (p: { id: number; str: string }) => endMarkWords(p.id, p.str) === TURN_ENDS;

/** How a written word-piece is drawn: its leading space outside the button, a new line as ¶, a bare space as ␣, a stop
 * mark as words that say which mark it is (endMarkWords: only <turn|> "ends its turn"). */
function pieceFace(p: AnswerPiece): { lead: string; body: string; kind: "word" | "mark" | "stop" } {
  if (p.endMark) return { lead: " ", body: endMarkWords(p.id, p.str), kind: "stop" };
  const lines = (p.str.match(/\n/g) ?? []).length;
  const text = p.str.replace(/^\s+/, "").replace(/\n/g, "¶");
  if (!text) return { lead: lines ? " " : "", body: lines ? "¶".repeat(lines) : "␣", kind: "mark" };
  return { lead: /^\s/.test(p.str) ? " " : "", body: (lines ? "¶".repeat(lines) + " " : "") + text, kind: "word" };
}

/** What a piece is called aloud. */
const spoken = (p: AnswerPiece) => (p.endMark ? (endsTurn(p) ? "the end of its turn" : endMarkWords(p.id, p.str)) : pieceWord(p.str));

/** A piece is narrow (a comma, a one-letter word, ¶ or ␣): its touch target is widened past its edges (index.css). */
const narrow = (body: string) => Array.from(body).length <= 2;

/** Why the server stopped writing, in plain words. `mark` is the answer's last piece: for "end_mark", the mark it wrote,
 * of which only <turn|> ends its turn (endMarkWords); any other is named, since it may be a broken answer. */
export function stopWords(r: string | null | undefined, mark?: { id: number; str: string; endMark: boolean } | null): string | null {
  switch (r) {
    case "end_mark":
      if (!mark?.endMark) return "it wrote one of its stop marks";
      if (endsTurn(mark)) return "it ended its turn";
      return `it wrote ${endMarkWords(mark.id, mark.str)}`;
    case "length":
      return "the length ran out";
    case "stopped":
      return "it was stopped";
    case "given":
      return "the word-pieces were given, not written";
    case "none":
      return "it was asked to write none";
    default:
      return null;
  }
}

/** The line naming the moment shown: “word-piece 3 of 12 · reading “am”, where it chose “a” (97% in its own odds)”. */
const Moment: React.FC<{ s: Sighting; n: number | null }> = ({ s, n }) => {
  const k = s.tokenIndex;
  const where = readWhere(s);
  const chose = chosenPhrase(s);
  const c = s.chosen;
  return (
    <>
      {n != null && n > 0 ? <>word-piece {k + 1} of {n} · </> : null}
      reading {s.readAt != null ? <span title={JSON.stringify(s.readAt)}>{pieceWord(s.readAt)}</span> : <Missing what="not sent" />}
      {k === 0 ? where.phrase : null}
      {chose ? (
        <>
          , {chose}
          {c?.modelProb != null ? <> ({pct(c.modelProb)} in its own odds{c.source === "given" ? ", as read again" : ""})</> : null}
        </>
      ) : n === 0 ? (
        ", where it wrote nothing"
      ) : null}
    </>
  );
};

/** Where a request in flight has got, and "stop here" when the server can stop it between word-pieces. */
const InFlight: React.FC<{ shown: AnswerSet | null }> = ({ shown }) => {
  const incoming = useTheodoliteStore((s) => s.incoming);
  const cancelSweep = useTheodoliteStore((s) => s.cancelSweep);
  if (!incoming) return null;
  const same = shown != null && incoming.requestId === shown.requestId;
  const { phase, done, of } = incoming.progress;
  // "sight this one too": one more word-piece of the answer shown, read with that answer's own options (and, in a frame
  // built from a neuron, read where its frame was built as well, so that it is read in the same frame)
  const oneMore = incoming.oneMoreOf != null;
  const what = oneMore
    ? phase === "sighting" && of > 1
      ? `reading ${Math.min(done + 1, of)} of ${of}${done === 0 ? ", the first where its frame was built" : ", the one asked for"}…`
      : "reading it…"
    : phase === "writing"
      ? `writing ${done} of ${of}…`
      : phase === "sighting"
        ? `sighting ${Math.min(done + 1, Math.max(of, 1))} of ${Math.max(of, 1)}…`
        : incoming.retakeOf
          ? "reading the same word-pieces again…"
          : "writing…";
  const lead = oneMore
    ? `Sighting word-piece ${incoming.wanted + 1} too, read as the rest of this answer was read: `
    : same
      ? ""
      : incoming.retakeOf
        ? "Reading it again with the options as set now: "
        : "A new answer: ";
  const canStop = incoming.streamed && (phase === "writing" || phase === "sighting");
  return (
    <p className="went-progress" role="status" aria-live="polite">
      {lead}
      {what}
      {canStop ? (
        incoming.stopAsked ? (
          <span className="italic"> stopping after this word-piece…</span>
        ) : (
          <>
            {" "}
            <button type="button" className="btn-link" onClick={cancelSweep}>
              stop here
            </button>
          </>
        )
      ) : null}
    </p>
  );
};

/** How many of the answer were sighted, when not all: the cap, "sight each" off, or a stream cut short. */
function sightedNote(a: AnswerSet, n: number): string | null {
  const arrived = Object.keys(a.sightings).length;
  if (a.done && arrived < a.nToSight) {
    const why = a.cancelled ? "it was stopped" : a.error === CONNECTION_CLOSED ? "the connection closed" : a.error ? "the server failed" : "the stream ended";
    return `Sighted ${arrived} of ${a.nToSight} before ${why}; the rest are plain text.`;
  }
  // the plan is known once the answer is written: say what will not be sighted while it is still sighting, too
  if (!a.order.length || a.order.length >= n) return null;
  if (a.generation?.capped && a.order.length === Math.min(n, a.generation.maxSighted ?? SIGHT_CAP)) {
    return `Only the first ${a.order.length} of the ${n} it wrote are sighted: at most ${a.generation.maxSighted ?? SIGHT_CAP} at a time.`;
  }
  if (a.order.length === 1 && a.order[0] === 0 && !a.retakeOf) return "Only the moment it chose its first word-piece is sighted; the rest are plain text.";
  return `${a.order.length} of the ${n} it wrote are sighted; the rest are plain text.`;
}

export const WordsItWrote: React.FC<{
  /** the sighting whose answer is drawn (default: the one shown) */
  sighting?: Sighting | null;
  /** only the word-pieces and the line naming the moment (for a strip on another page) */
  compact?: boolean;
}> = ({ sighting: given, compact = false }) => {
  const shownSighting = useTheodoliteStore((s) => s.sighting);
  const answer = useTheodoliteStore((s) => s.answer);
  const pick = useTheodoliteStore((s) => s.pickToken);
  const conn = useTheodoliteStore((s) => s.conn);
  const isSweeping = useTheodoliteStore((s) => s.isSweeping);
  const incoming = useTheodoliteStore((s) => s.incoming);
  const refs = React.useRef<(HTMLButtonElement | null)[]>([]);
  const [more, setMore] = React.useState<string>("");
  const s = given ?? shownSighting;
  if (!s) return null;

  // the set this sighting belongs to, when it is the set shown: then its pieces can be chosen
  const set = answer && answer.sightings[s.tokenIndex] === s ? answer : null;
  const pieces = set?.pieces ?? piecesOf(s);
  const begun = s.answerBegins.trim();

  // the recorded sighting, or an older server: the words as one line, as before
  if (!pieces) {
    return (
      <div className="went-block">
        <span className="went-words">“… {s.continuation.trim()}”</span>
        {compact ? null : <InFlight shown={null} />}
      </div>
    );
  }

  const n = pieces.length;
  const k = s.tokenIndex;
  const ready = (i: number) => !!set?.sightings[i];
  const planned = (i: number) => !!set && set.order.includes(i) && (!set.done || ready(i));
  // a word-piece chosen and not yet sighted is still to come while this set is sighting, or while "sight this one too"
  // is reading it; once neither is in flight (stopped, failed, the connection closed) the piece shown is the one chosen
  const coming = (i: number) =>
    !!set &&
    (!set.done || (incoming != null && !incoming.done && incoming.oneMoreOf === set.requestId && incoming.order.includes(i)));
  const wanted = set && (ready(set.wanted) || coming(set.wanted)) ? set.wanted : k;
  const tabStop = set && planned(wanted) ? wanted : k;
  const sightedIdx = set
    ? Object.keys(set.sightings)
        .map(Number)
        .sort((x, y) => x - y)
    : [];
  const before = [...sightedIdx].reverse().find((i) => i < wanted);
  const after = sightedIdx.find((i) => i > wanted);

  const onKey = (e: React.KeyboardEvent) => {
    if (!set) return;
    let to: number | undefined;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") to = after;
    else if (e.key === "ArrowLeft" || e.key === "ArrowUp") to = before;
    else if (e.key === "Home") to = sightedIdx[0];
    else if (e.key === "End") to = sightedIdx[sightedIdx.length - 1];
    else return;
    e.preventDefault();
    if (to !== undefined) {
      pick(to);
      refs.current[to]?.focus();
    }
  };

  const words = (
    <div
      className="went-words"
      role={set ? "radiogroup" : undefined}
      aria-label={set ? "The word-pieces it wrote: choose one to see the sighting taken where it chose it" : "The word-pieces it wrote"}
      onKeyDown={set ? onKey : undefined}
    >
      <span aria-hidden="true">“…{begun ? <span className="begun"> {begun}</span> : null}</span>
      {pieces.map((p, i) => {
        const f = pieceFace(p);
        const cls = f.kind === "word" ? "went-piece" : `went-piece ${f.kind}`;
        if (!set) {
          // another answer's sighting, drawn as text with its own word-piece marked
          return (
            <React.Fragment key={i}>
              {f.lead}
              <span className={`went-plain${f.kind === "word" ? "" : ` ${f.kind}`}${i === k ? " went-here" : ""}`} title={JSON.stringify(p.str)}>
                {f.body}
              </span>
            </React.Fragment>
          );
        }
        if (!planned(i)) {
          return (
            <React.Fragment key={i}>
              {f.lead}
              <span className={`went-plain${f.kind === "word" ? "" : ` ${f.kind}`}`} title={JSON.stringify(p.str)}>
                {f.body}
              </span>
            </React.Fragment>
          );
        }
        const on = i === wanted;
        const waiting = !ready(i);
        return (
          <React.Fragment key={i}>
            {f.lead}
            <button
              ref={(el) => {
                refs.current[i] = el;
              }}
              type="button"
              role="radio"
              aria-checked={on}
              tabIndex={i === tabStop ? 0 : -1}
              className={`${cls}${narrow(f.body) ? " narrow" : ""}${waiting ? " waiting" : ""}`}
              title={JSON.stringify(p.str)}
              aria-label={`word-piece ${i + 1} of ${n}, ${spoken(p)}${waiting ? ", still being sighted: shown when it arrives" : ""}`}
              onClick={() => pick(i)}
            >
              {f.body}
            </button>
          </React.Fragment>
        );
      })}
      <span aria-hidden="true">”</span>
    </div>
  );

  if (compact || !set) {
    return (
      <div className="went-block">
        {words}
        <p className="went-moment">
          <Moment s={s} n={n} />
        </p>
      </div>
    );
  }

  const g = s.generation ?? set.generation;
  // how these word-pieces were first written (a re-reading's own echo says only "given")
  const w0 = set.wrote ?? g;
  const likeliest = (x: typeof g) => x != null && x.greedy && !x.presencePenalty;
  const offTop = s.recaptureCheck != null && !s.recaptureCheck.top1IsChosen;
  // written greedily, the piece chosen was the top one when written: read again, layer 60 can put another first only by
  // a near-tie that the two computations break differently. For a piece given (read again from the answer's ids) its
  // odds are the re-reading's own, so the note says when it was written instead.
  const drift = offTop && s.chosen?.source === "written" && likeliest(g);
  const driftGiven = offTop && s.chosen?.source === "given" && w0?.source === "written" && likeliest(w0);
  const note = sightedNote(set, n);
  const unsighted = pieces.filter((p) => !set.sightings[p.index] && !planned(p.index));
  const canMore = conn === "open" && !isSweeping && set.streamed && !!s.answerIds;
  const stop = stopWords(set.wrote?.stopReason ?? g?.stopReason, pieces[n - 1]);
  // "sight this one too" reads one position, or two in a frame built from a neuron (where the set's frame was built,
  // then the piece): PACE, as timed on this server
  const morePositions = set.payload.survey_mode === "pole_motif" ? 2 : 1;
  const moreSecs = Math.round((PACE.base + PACE.sight * (morePositions - 1)) * 2) / 2;
  const moreTime = `${Math.floor(moreSecs) || ""}${moreSecs % 1 ? "½" : ""} s`;
  const anyWaiting = set.order.some((i) => !ready(i) && planned(i));
  const sentPrompts = set.payload.centre_prompts;
  const hadControl = Array.isArray(set.payload.centre_token_ids) || (Array.isArray(sentPrompts) && sentPrompts.length > 0);

  return (
    <div className="went-block">
      {words}
      <InFlight shown={set} />
      <p className="went-moment" aria-live="polite">
        <Moment s={s} n={n} />
        {wanted !== k ? <> · word-piece {wanted + 1} is chosen, and shown when its sighting arrives</> : null}.
      </p>
      {set.order.length > 1 ? (
        <div className="went-choose">
          <p className="went-note">
            {`Each underlined word-piece has a sighting of its own, taken where the model chose it: choose one${
              sightedIdx.length > 1 ? ", or step with the buttons," : ""
            } to see the sixty layers there.${anyWaiting ? " Faint ones are still being sighted." : ""}`}
          </p>
          {sightedIdx.length > 1 ? (
            <span className="went-step">
              <button type="button" className="btn-ghost plain" disabled={before === undefined} onClick={() => before !== undefined && pick(before)}>
                ← earlier word-piece
              </button>
              <button type="button" className="btn-ghost plain" disabled={after === undefined} onClick={() => after !== undefined && pick(after)}>
                later word-piece →
              </button>
            </span>
          ) : null}
        </div>
      ) : null}
      {drift && s.recaptureCheck && s.chosen ? (
        <p className="went-note">
          Read again, layer 60 puts {pieceWord(s.recaptureCheck.top1Token)} first, not {pieceWord(s.chosen.token)}: writing and reading
          again are not computed exactly alike in 4-bit.
        </p>
      ) : null}
      {driftGiven && s.recaptureCheck && s.chosen ? (
        <p className="went-note">
          Read again, layer 60 puts {pieceWord(s.recaptureCheck.top1Token)} first, not {pieceWord(s.chosen.token)}; when it was written it
          chose {pieceWord(s.chosen.token)}, its likeliest then: a near-tie, which writing and reading again do not compute exactly
          alike in 4-bit. The odds shown here are this re-reading’s own.
        </p>
      ) : null}
      {g ? (
        <p className="went-how">
          <span className="label-sc">how it wrote</span> <OneDrawLine generation={g} />
          {g.presencePenalty ? (
            <>
              {" "}
              <PenaltyLine penalty={g.presencePenalty} />
            </>
          ) : null}
        </p>
      ) : null}
      {note ? <p className="went-note">{note}</p> : null}
      {k === 0 && readWhere(s).role === "template_end" ? <TemplateEndCaution /> : null}
      {sightedIdx.length > 1 ? <NotAMatchedPair /> : null}
      <Apparatus title="the word-pieces it wrote" sources={false}>
        <dl className="dl tight">
          <dt>this one</dt>
          <dd className="num">
            token index {k} (counting from 0); read at position {s.readPos ?? <Missing what="not sent" />}
            {s.readAtId != null ? <>, the word-piece id {s.readAtId}</> : null}
            {s.chosen ? (
              <>
                ; it chose id {s.chosen.id} {pieceWord(s.chosen.token)}
                {s.chosen.modelRank != null ? `, rank ${s.chosen.modelRank}` : ""}
                {s.chosen.nTied ? ` (tied with ${s.chosen.nTied})` : ""}
                {s.chosen.modelProb != null ? `, ${pct(s.chosen.modelProb)} of its own odds` : ""}
                {s.chosen.logit != null ? `, score ${s.chosen.logit.toFixed(2)}` : ""}
                {s.chosen.samplerProb != null
                  ? `; ${pct(s.chosen.samplerProb)} under the sampler${s.chosen.samplerSupport != null ? `, one of ${s.chosen.samplerSupport} it could draw` : ""}`
                  : ""}
                {s.chosen.source === "given" ? "; given, read again (its odds from the re-reading’s layer 60)" : ""}
              </>
            ) : null}
            . The sighting of word-piece k reads position P − 1 + k, where it had just read the piece before it; the last piece it
            wrote is chosen but never read.
          </dd>
          <dt>the answer</dt>
          <dd className="num">
            {n} word-piece{n === 1 ? "" : "s"}
            {g?.maxNewTokens != null ? ` of at most ${g.maxNewTokens}` : ""}
            {stop ? ` (${stop})` : ""}; ids <span className="mono-wrap">[{(s.answerIds ?? pieces.map((p) => p.id)).join(", ")}]</span>
          </dd>
          <dt>sighted</dt>
          <dd className="num">
            {sightedIdx.length} of {set.nToSight} planned
            {set.order.length ? (
              <>
                , in the order <span className="mono-wrap">[{set.order.join(", ")}]</span>
              </>
            ) : null}
            {set.retakeOf ? "; read again from the answer’s ids, with nothing written again" : ""}
            {set.error ? <>; the server said: {set.error}</> : null}
          </dd>
          <dt>read again</dt>
          <dd className="num">
            {s.recaptureCheck ? (
              <>
                layer 60 of the re-reading puts id {s.recaptureCheck.top1Id} {pieceWord(s.recaptureCheck.top1Token)} first
                {s.recaptureCheck.top1IsChosen ? ", the piece chosen" : ", not the piece chosen"}
                {s.recaptureCheck.chosenLogitDiff != null && s.chosen?.source === "written"
                  ? `; the chosen piece’s score differs by ${s.recaptureCheck.chosenLogitDiff.toFixed(3)} from when it was written`
                  : s.chosen?.source === "given"
                    ? "; the piece was given and its odds are this re-reading’s own, so there is no writing to compare its score with"
                    : ""}
              </>
            ) : (
              <Missing what="not sent" />
            )}
          </dd>
          <dt>timing</dt>
          <dd className="num">
            {s.timing ? (
              <>
                writing {ms(s.timing.generateMs) || "–"}, reading again {ms(s.timing.captureMs) || "–"}, this position {ms(s.timing.analyseMs) || "–"}
                {s.timing.totalMs != null ? `, in all ${ms(s.timing.totalMs)}` : ""}
              </>
            ) : (
              <Missing what="not sent" />
            )}
          </dd>
          <dt>centred on</dt>
          <dd>
            {s.centreKind === "control_reading_same_answer"
              ? "the control, reading this very answer up to the same word-piece"
              : s.centreKind === "control_prompt_end"
                ? "the control, read where its question ends"
                : s.centreKind === "none"
                  ? k > 0 && hadControl
                    ? "nothing at this word-piece: a control read where its question ends is never subtracted from a later word-piece"
                    : "nothing"
                  : <Missing what="not sent" />}
          </dd>
          {unsighted.length ? (
            <>
              <dt>sight one more</dt>
              <dd style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: "4px 12px" }}>
                <label className="visually-hidden" htmlFor="sight-more">
                  a word-piece written but not sighted
                </label>
                <select id="sight-more" value={more} onChange={(e) => setMore(e.target.value)} className="input-line" style={{ width: "auto", fontSize: 15 }}>
                  <option value="">choose a word-piece…</option>
                  {unsighted.map((p) => (
                    <option key={p.index} value={String(p.index)}>
                      word-piece {p.index + 1} · {spoken(p)}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  className="btn-link"
                  disabled={!canMore || more === ""}
                  onClick={() => {
                    if (sightPieceToo(Number(more))) setMore("");
                  }}
                >
                  sight this one too
                </button>
                <span className="muted">
                  read again from the answer’s exact ids with the options it was read with, not the options as set now; nothing
                  written again; about {moreTime}
                  {morePositions > 1 ? ", for it reads the word-piece its frame was built at as well, so the frame stays the same" : ""}
                </span>
              </dd>
            </>
          ) : null}
          <dt>export</dt>
          <dd>
            <button type="button" className="btn-link" onClick={() => exportAnswer(set)}>
              export all {sightedIdx.length} sighting{sightedIdx.length === 1 ? "" : "s"} of this answer (.json)
            </button>
          </dd>
        </dl>
      </Apparatus>
    </div>
  );
};

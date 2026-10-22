import React, { useMemo, useState } from "react";
import { Apparatus, Lozenge, Missing, Page, ReadingThrough, Rubric } from "../components/kit";
import { useReading } from "../reading";
import { href, navigate, type Route } from "../router";
import {
  FNW_DEFAULT,
  type Sighting,
  type SweepProgress,
  endMarkWords,
  generationWords,
  ordinal,
  piecesOf,
  readWhere,
  useShownSet,
  useTheodoliteStore,
} from "../store/useTheodoliteStore";
import { comma, ms, numberWord, pieceWord, wp } from "../lib/words";
import { describeFrame } from "../lib/frame";
import { isBaseFitted, useInstrumentLens } from "../lib/lens";
import {
  FIRST_QUESTIONS,
  HOW_PUT,
  MEMORY_ENVELOPE_GB,
  RECORDED_CALL,
  RECORDED_MAX_NEW_TOKENS,
} from "./incipit/data";
import { type Frontispiece, FrontispieceFigure, frontispieceOf } from "./incipit/Frontispiece";
import { stopWords } from "./instrument/moment";
import {
  Diamond10,
  LeafFigure,
  MakersFrame,
  RuleWithLozenge,
  SectionDivider,
  StepAsk,
  StepRead,
  StepSight,
  TitleInitial,
} from "./incipit/ornaments";

/*
 * Incipit: the welcome page and the first sighting (route #/), built from the approved canvas
 * Incipit.dc.html. The frontispiece is drawn from the current sighting (the study's recorded E6
 * sweep until a live one arrives); the example questions are the study's recorded runs.
 */

const MUTED = "#6B5D4D";
const INK = "#2B2117";
const HAIR_SOFT = "#E3DACA";
const HAIR_FAINT = "#ECE4D5";
const HAIR = "#D9CFBD";

const sc = (letterSpacing: string, fontSize?: number, color = MUTED): React.CSSProperties => ({
  fontVariantCaps: "all-small-caps",
  letterSpacing,
  color,
  ...(fontSize ? { fontSize } : null),
});

/** "11, 21, 36 and 60" */
function andList(items: (string | number)[]) {
  const s = items.map(String);
  return s.length <= 1 ? s.join("") : s.slice(0, -1).join(", ") + " and " + s[s.length - 1];
}

const pct1 = (p: number) => `${(p * 100).toFixed(1)}%`;

/** Where a sighting was read, after “read at”: “I”, in its reply “Hi! I” (the first word-piece, read where your words
 * end); “am”, the 3rd word-piece of its own answer (a later one). */
function readAtWords(s: Sighting): string | null {
  if (s.readAt == null) return null;
  const w = readWhere(s);
  return w.role === "written" ? `${pieceWord(s.readAt)}, the ${ordinal(w.k)} word-piece of its own answer` : `${pieceWord(s.readAt)}${w.phrase}`;
}

/** Where a request in flight has got: “writing 3 of 6…”, “sighting 2 of 6…”. */
function progressWords(p: SweepProgress | null | undefined): string | null {
  if (!p) return null;
  if (p.phase === "writing") return `writing ${p.done} of ${p.of}…`;
  if (p.phase === "sighting") return `sighting ${Math.min(p.done + 1, Math.max(p.of, 1))} of ${Math.max(p.of, 1)}…`;
  return null;
}

/** How long a sighting took, and which time that is: for a live one, from the request to this word-piece’s sighting
 * (writing the answer, reading it again, this position’s sixty layers); for the recorded one, its sweep. */
function tookWords(s: Sighting | null): { secs: string; what: string } | null {
  if (!s) return null;
  const total = s.timing?.totalMs ?? null;
  if (s.source === "live" && total != null) {
    return { secs: `${(total / 1000).toFixed(1)} seconds`, what: "from asking to this word-piece’s sighting, its answer written and read again first" };
  }
  if (s.elapsedMs != null) return { secs: `${(s.elapsedMs / 1000).toFixed(1)} seconds`, what: "its whole sweep" };
  return null;
}

/* ------------------------------------------------------------ the title and frontispiece */

const TitleColumn: React.FC = () => {
  const rubrics = useReading((s) => s.rubrics);
  const toggleRubrics = useReading((s) => s.toggleRubrics);

  const toBegin = (e: React.MouseEvent<HTMLAnchorElement>) => {
    const el = document.getElementById("begin");
    if (!el) return;
    e.preventDefault();
    let reduce = false;
    try {
      reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    } catch {
      // no media queries: jump
    }
    el.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
    document.getElementById("begin-head")?.focus({ preventScroll: true });
  };

  return (
    <div style={{ flex: "999 1 600px", minWidth: 0, display: "flex", flexDirection: "column", gap: 24 }}>
      <p style={{ margin: 0, fontWeight: 500, ...sc("0.24em", 14, "#8F2B1E") }}>incipit · here begins a survey of the interior</p>

      <h1 id="title-head" style={{ margin: 0, position: "relative", fontWeight: 500 }}>
        <span className="visually-hidden">Logos Theodolite</span>
        <span aria-hidden="true" style={{ display: "flex", flexWrap: "nowrap", alignItems: "center", gap: "14px clamp(12px, 2vw, 26px)", minWidth: 0 }}>
          <TitleInitial />
          <span style={{ display: "flex", flexDirection: "column", minWidth: 0, fontSize: "clamp(36px, 6vw, 84px)", lineHeight: 1.04, letterSpacing: "0.005em", color: INK }}>
            <span>ogos</span>
            <span>Theodolite</span>
          </span>
        </span>
      </h1>

      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <RuleWithLozenge />
        <p style={{ margin: 0, ...sc("0.22em", 18) }}>an instrument for surveying the inside of a language model</p>
      </div>

      <p style={{ margin: 0, fontSize: 22, lineHeight: 1.55, maxWidth: 640 }}>
        Between a question and its answer, this model carries each word up through sixty layers, and little of that country
        has been mapped. This instrument lets you follow a single word as it rises and, at every layer, read what a lens makes
        of it: sometimes words you never wrote, in languages you never used.
      </p>

      <p className="num" style={{ margin: 0, display: "flex", flexWrap: "wrap", alignItems: "center", gap: "4px 16px", fontSize: 16, color: MUTED }}>
        <span>60 layers</span>
        <span aria-hidden="true" style={{ color: "#A07A2E", fontSize: 11 }}>
          ◆
        </span>
        <span>5,376 dimensions in each</span>
        <span aria-hidden="true" style={{ color: "#A07A2E", fontSize: 11 }}>
          ◆
        </span>
        <span>a vocabulary of 262,144 word-pieces</span>
      </p>

      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "12px 18px" }}>
        <a
          href="#begin"
          onClick={toBegin}
          style={{
            display: "inline-flex",
            alignItems: "center",
            minHeight: 48,
            padding: "12px 22px",
            fontSize: 18,
            ...sc("0.16em", undefined, INK),
            textDecoration: "none",
            background: "transparent",
            border: "1px solid #C9BEAC",
            borderRadius: 2,
          }}
        >
          choose a first question ↓
        </a>
        <a href={href.instrument()} style={{ fontSize: 17, padding: "12px 4px" }}>
          or write your own
        </a>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 6, maxWidth: 640, paddingTop: 14, borderTop: `1px solid ${HAIR_SOFT}` }}>
        <Rubric style={{ maxWidth: 640 }}>
          Lines in red italic, like this one, are rubrics: directions for the reader, marked in red the way scribes and early
          printers set apart their headings and directions. Follow them on your first sightings, and turn them off once you
          know your way.
        </Rubric>
        {rubrics ? null : (
          <p style={{ margin: 0, fontSize: 16, color: MUTED }}>Rubrics are hidden. Tick the box to bring back the red directions.</p>
        )}
        <label className="check" style={{ color: MUTED }}>
          <input type="checkbox" checked={rubrics} onChange={toggleRubrics} />
          show rubrics
        </label>
      </div>
    </div>
  );
};

/* ------------------------------------------------------------ apparatus · the frontispiece sighting */

const FrontApparatus: React.FC<{ s: Sighting | null; geo: Frontispiece }> = ({ s, geo }) => {
  const [callOpen, setCallOpen] = useState(false);
  // the request this sighting answers, exactly as the page sent it (null for the recorded sighting)
  const set = useShownSet();
  const apparatus = useReading((st) => st.apparatus);
  const toggleApparatus = useReading((st) => st.toggleApparatus);
  const recorded = s?.source === "fixture";
  const stations = geo.stations;

  const projFracs = s ? s.layers.map((l) => l.projFrac).filter((v): v is number => v != null) : [];
  const strengths = s ? stations.map((st) => s.layers[st.idx].streamStrength) : [];
  const eyes = s ? stations.map((st) => s.layers[st.idx].eyeTop) : [];
  const haveStrength = !!s && s.layers.some((l) => l.streamStrength != null);
  const haveEyes = stations.length > 0 && eyes.every((e) => e != null);
  const notRecorded: string[] = [];
  if (s && !haveStrength) notRecorded.push("‖h‖ at each layer");
  if (s && stations.length > 0 && !haveEyes) notRecorded.push(`the naked eye’s words at the ${numberWord(stations.length)} stations`);

  // the call as sent: the request this sighting's answer was taken with; without one (an older page state), rebuilt from
  // the sighting's own echo, never from the form's settings
  const g = s?.generation ?? null;
  const call = recorded || !s
    ? RECORDED_CALL
    : set
      ? JSON.stringify({ action: "sweep", payload: set.payload }, null, 1)
      : '{"action": "sweep",\n' +
        (s.chatWrapped
          ? ' "payload": {"prompt": ' + JSON.stringify(s.question) +
            ', "chat": true' +
            (s.system ? ', "system": ' + JSON.stringify(s.system) : "") +
            ', "assistant_prefix": ' + JSON.stringify(s.answerBegins)
          : ' "payload": {"prompt": ' + JSON.stringify(s.textSent ?? "")) +
        ', "monuments": ' + JSON.stringify(s.landmarks) +
        ', "survey_mode": ' + JSON.stringify(s.surveyMode) +
        (g?.maxNewTokens != null ? ', "max_new_tokens": ' + g.maxNewTokens : "") +
        "}\n}";
  const k = s?.tokenIndex ?? 0;
  const nAnswer = s?.answerIds?.length ?? g?.nWritten ?? null;
  // why it stopped writing, naming the stop mark it wrote (only <turn|> ends its turn)
  const stopped = g ? stopWords(g.stopReason, (s && set?.sightings[s.tokenIndex] === s ? set.pieces : null) ?? piecesOf(s)) : null;

  const lm = s?.landmarks ?? ["she", "he", "it"];

  return (
    <Apparatus title="the frontispiece sighting" label="Apparatus for the frontispiece" style={{ padding: "12px 0 34px" }}>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "16px 48px", alignItems: "flex-start" }}>
        <dl className="dl" style={{ flex: "999 1 560px", minWidth: 0 }}>
          <dt>the question</dt>
          <dd>
            {!s ? (
              <Missing what="no sighting yet" />
            ) : recorded ? (
              <>
                Wrapped as a chat by the study’s lab script, not by the server: the user asks “{s.question}” and the reply is
                begun “{s.answerBegins}”. The same sighting stands on folio i, <a href={href.instrument()}>the instrument</a> (E6).
              </>
            ) : (
              <>
                {s.chatWrapped
                  ? "Wrapped as a chat by the server (the tokenizer’s chat template, thinking off), then the begun answer; the exact text read: "
                  : "Sent by the server as bare text, with no chat template (REPORT §3.4, bug 5): "}
                <span className="mono-wrap">{JSON.stringify(s.textSent ?? "")}</span>. The same sighting stands on folio i,{" "}
                <a href={href.instrument()}>the instrument</a>.
              </>
            )}
          </dd>

          <dt>word-pieces</dt>
          <dd className="num">
            {s?.nTokens != null ? s.nTokens : <Missing />} read up to here; read at{" "}
            {s && readAtWords(s) != null ? readAtWords(s) : <Missing />} (position {s?.readPos != null ? s.readPos : <Missing />},
            counting from 0)
            {s && k > 0 && nAnswer != null ? (
              <>
                ; this is the sighting of word-piece {k + 1} of the {nAnswer} it wrote
                {s.chosen ? <>, {s.chosen.endMark ? endMarkWords(s.chosen.id, s.chosen.token) : pieceWord(s.chosen.token)}</> : null}
              </>
            ) : null}
            .
            {recorded && s?.firstIds ? (
              <>
                {" "}Start marks <span className="mono">[{s.firstIds.join(", ")}]</span>: the start mark doubled, as the instrument
                read chat text before the single-BOS fix (REPORT §3.4, bug 6; now applied). Sighted again with{" "}
                <span className="mono">prepend_bos=False</span>: 21 word-pieces, <span className="mono">[2, 105, 2364]</span>, the
                same continuation.
              </>
            ) : null}
            {!recorded && s?.firstIds ? (
              <>
                {" "}Start marks <span className="mono">[{s.firstIds.join(", ")}]</span>.
              </>
            ) : null}
            {!recorded && s?.promptTokens ? (
              <>
                {" "}The word-pieces of the question: <span className="mono-wrap">{s.promptTokens.map((t) => JSON.stringify(t)).join(", ")}</span>
                {k > 0 && s.answerStrs ? (
                  <>
                    , then of its own answer up to here:{" "}
                    <span className="mono-wrap">{s.answerStrs.slice(0, k).map((t) => JSON.stringify(t)).join(", ")}</span>
                  </>
                ) : null}
                .
              </>
            ) : null}
          </dd>

          <dt>continuation</dt>
          <dd className="num">
            {recorded ? (
              <>
                greedy, <span className="mono">max_new_tokens={RECORDED_MAX_NEW_TOKENS}</span>
              </>
            ) : g ? (
              <>
                {generationWords(g)}
                {g.source === "given" ? ", read again from its word-pieces" : ""};{" "}
                {g.nWritten ?? nAnswer ?? <Missing what="not sent" />} written of at most{" "}
                <span className="mono">max_new_tokens={g.maxNewTokens ?? "[not sent]"}</span>
                {stopped ? ` (${stopped})` : ""}
                {g.capped ? `; sighted the first ${set?.order.length ?? g.maxSighted ?? "[not sent]"} of them` : ""}
              </>
            ) : (
              <>
                as an older server writes, its likeliest word-piece each time; how many: <Missing what="not sent by this server" />
              </>
            )}
            : {s ? <>“{s.continuation}”</> : <Missing />}.{" "}
            {recorded ? (
              <>
                The sweep took {s?.elapsedMs != null ? `${comma(s.elapsedMs)} ms` : <Missing />} (6.25 s by the wall clock).
              </>
            ) : s?.timing ? (
              <>
                Writing took {ms(s.timing.generateMs) || "[not sent]"}, reading it again {ms(s.timing.captureMs) || "[not sent]"}, this
                word-piece’s sixty layers {ms(s.timing.analyseMs) || "[not sent]"}; it came back {ms(s.timing.totalMs) || "[not sent]"} after
                the request.
              </>
            ) : (
              <>The sweep took {s?.elapsedMs != null ? `${comma(s.elapsedMs)} ms` : <Missing />}.</>
            )}
          </dd>

          <dt>readout</dt>
          <dd className="num">
            logits(h, ℓ) = unembed(h · J<sub>ℓ</sub>
            <sup>⊤</sup>), unembed(u) = 30 · tanh(W<sub>U</sub> · RMSNorm(u) / 30); J = the J++ lens (Ayonrinde &amp; Lindsey
            2026) [4], fitted on the base model and used here on the instruct model (bug 9).{" "}
            {stations.length === 0 ? (
              <>
                No station words: the lens was not recorded at any layer of this sighting. <Missing />
              </>
            ) : (
              <>
                {stations.length === 1 ? "The one station word is its first choice" : `The ${numberWord(stations.length)} station words are its first choices`}{" "}
                at {stations.length === 1 ? "layer" : "layers"} {andList(stations.map((st) => st.no))} (index{" "}
                {stations.length === 1 ? stations[0].idx : `${stations[0].idx} to ${stations[stations.length - 1].idx}`}):{" "}
                {stations.map((st, i) => (
                  <React.Fragment key={st.idx}>
                    {i > 0 ? ", " : null}
                    {wp(st.token)} {st.prob != null ? pct1(st.prob) : <Missing what="not recorded" />}
                  </React.Fragment>
                ))}
                .
              </>
            )}
          </dd>

          <dt>drawing</dt>
          <dd className="num">
            bearing θ = atan2(y, x) and elevation φ = asin(z / ‖p‖) on{" "}
            {s && !lm.every(Boolean) && s.surveyMode === "monument" ? (
              <>the frame of the three landmark words’ rows of W<sub>U</sub> (their names were not reported)</>
            ) : s ? (
              <>{describeFrame(s).frameFormula}</>
            ) : (
              <Missing what="no sighting yet" />
            )}
            ; the radius is schematic. The frame catches{" "}
            {projFracs.length ? (
              `${(Math.min(...projFracs) * 100).toFixed(1)}–${(Math.max(...projFracs) * 100).toFixed(1)}%`
            ) : (
              <Missing what="not recorded in this sighting" />
            )}{" "}
            of {s ? describeFrame(s).denom : "the state"} across the sixty layers
            {s && describeFrame(s).lowdin
              ? `; a Löwdin frame squares ${describeFrame(s).poleMotif ? "its three axes" : "the words"} all at once, so their order does not change the frame, though it still says which axis is which.`
              : s && describeFrame(s).poleMotif
                ? "; a neuron’s frame is squared in the fixed order of its roles (Gram–Schmidt), so axis 1 is exactly the neuron’s gate."
                : ", and with a Gram–Schmidt frame the bearing turns with the order of the three words (58–99°, bug 3)."}{" "}
            <a href={href.landmarks()}>landmarks</a>
          </dd>

          <dt>caveat</dt>
          <dd className="num">
            On prose, the lens’s first choice matched the model’s own at most about 5% of the time through layer 50, and 28% at
            layer 59 (E3 per layer; REPORT §3.1’s “3%” understates the first: erratum E30). Below layer 31 no lens of this averaged
            kind proved stable (R1), and below layer 21 not even comparisons through it were validated (E2; E3). The middle
            stations are glimpses, and this lone sighting has no matched control beside it (§3.5).
          </dd>

          {haveStrength && stations.length > 0 ? (
            <>
              <dt>stream strength</dt>
              <dd className="num">
                ‖h‖ at the stations:{" "}
                {stations.map((st, i) => (
                  <React.Fragment key={st.idx}>
                    {i > 0 ? ", " : null}
                    layer {st.no} {strengths[i] != null ? comma(strengths[i]!) : <Missing what="not recorded" />}
                  </React.Fragment>
                ))}
                .
              </dd>
            </>
          ) : null}

          {haveEyes ? (
            <>
              <dt>the naked eye</dt>
              <dd className="num">
                its first choices at the stations:{" "}
                {stations.map((st, i) => (
                  <React.Fragment key={st.idx}>
                    {i > 0 ? ", " : null}
                    {wp(eyes[i])} at layer {st.no}
                  </React.Fragment>
                ))}
                .
              </dd>
            </>
          ) : null}

          {notRecorded.length ? (
            <>
              <dt>not recorded</dt>
              <dd style={{ color: MUTED }}>
                {notRecorded.join(", and ")}: <Missing />
              </dd>
            </>
          ) : null}
        </dl>

        <div style={{ flex: "1 1 340px", minWidth: 0, display: "flex", flexDirection: "column", gap: 8 }}>
          <span style={sc("0.14em")}>{recorded || !s ? "the call, as recorded" : "the call, as sent"}</span>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "8px 12px" }}>
            <button
              className="btn-ghost"
              type="button"
              onClick={() => setCallOpen((v) => !v)}
              aria-expanded={callOpen}
              aria-controls="incipit-call"
            >
              {callOpen ? "hide the call" : "show the call"}
            </button>
          </div>
          {callOpen ? (
            <div id="incipit-call" style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <pre className="raw" style={{ maxHeight: 320 }}>
                {call}
              </pre>
              <span style={{ color: MUTED, overflowWrap: "anywhere" }}>
                {recorded || !s
                  ? "excerpted from experiments/scripts/E6_tool_audit_02_sweep_demo.py; the result is E6_tool_audit_02_sweep_demo.json, key “prepend_bos=True”."
                  : set
                    ? `exactly as the page sent it over /ws/theodolite, with its own request id${s.chatWrapped ? "; the server wraps the question in the chat template" : ", as bare text"}. How it was written, as the server resolved it, is under “continuation”.`
                    : s.chatWrapped
                      ? "rebuilt from this sighting’s own record (its question, begun answer, landmarks, survey mode and length); the page sends it over /ws/theodolite with the nudge, frame and writing settings, and the server wraps it in the chat template."
                      : "rebuilt from this sighting’s own record (its prompt, landmarks, survey mode and length); the page sends it over /ws/theodolite with the nudge, frame and writing settings, as bare text."}
              </span>
            </div>
          ) : null}
          <a
            href={href.instrument()}
            onClick={() => {
              if (!apparatus) toggleApparatus();
            }}
            style={{ fontSize: 16, padding: "10px 0" }}
          >
            open it in the instrument, apparatus open →
          </a>
        </div>
      </div>
    </Apparatus>
  );
};

/* ------------------------------------------------------------ how a sighting works */

const WORDS: { term: React.ReactNode; def: React.ReactNode }[] = [
  { term: "sighting", def: "One reading of all sixty layers, taken at a single word." },
  { term: "layer", def: "One of the sixty stages a word passes through, counted from 1 at the bottom to 60 at the top." },
  { term: "word-piece", def: "What the model reads and writes: a whole word, part of one, or a mark such as “!”." },
  {
    term: "the lens",
    def: (
      <>
        Turns a layer’s inner state into the words the model might say there <span style={{ fontStyle: "italic" }}>(the J++ lens)</span>.
        The plainer reading, without that correction, is the naked eye <span style={{ fontStyle: "italic" }}>(the logit lens)</span>.
        The lens was fitted on this model as it was before its training to chat{" "}
        <span style={{ fontStyle: "italic" }}>(the base model)</span>: a fair guide when you compare two sightings, a rough one for
        any single word, and unproven below layer 21.
      </>
    ),
  },
  {
    term: (
      <>
        <span style={{ color: "#2E4A7D" }}>gathering</span> · <span style={{ color: "#7A5C28" }}>remembering</span>
      </>
    ),
    def: (
      <>
        Each layer gathers from the words before <span style={{ fontStyle: "italic" }}>(attention)</span> and remembers what it
        knows <span style={{ fontStyle: "italic" }}>(feed-forward)</span>, in a measure the instrument weighs through the lens.
      </>
    ),
  },
];

const Step: React.FC<{ icon: React.ReactNode; numeral: string; title: string; children: React.ReactNode }> = ({
  icon,
  numeral,
  title,
  children,
}) => (
  <li style={{ flex: "1 1 300px", minWidth: 0, display: "flex", flexDirection: "column", gap: 12 }}>
    {icon}
    <div style={{ display: "flex", alignItems: "baseline", gap: 14 }}>
      <span aria-hidden="true" style={{ fontSize: 44, lineHeight: 1, fontStyle: "italic", color: "#8A6A2C" }}>
        {numeral}
      </span>
      <h3 className="h-sub">{title}</h3>
    </div>
    <p style={{ margin: 0, maxWidth: 400 }}>{children}</p>
  </li>
);

const HowItWorks: React.FC<{ s: Sighting | null }> = ({ s }) => (
  <section aria-labelledby="how-head" style={{ display: "flex", flexDirection: "column", gap: 28, padding: "40px 0 44px" }}>
    <h2 className="h-sec" id="how-head">
      how a sighting works
    </h2>
    <ol style={{ listStyle: "none", margin: 0, padding: 0, display: "flex", flexWrap: "wrap", gap: "36px 56px" }}>
      <Step icon={<StepAsk />} numeral="i" title="ask a question">
        Write a question in your own words and, if you like, the first words of its answer. The instrument follows the last word
        you give it, then each word-piece the model writes.
      </Step>
      <Step icon={<StepSight />} numeral="ii" title="take a sighting">
        The model writes its answer, reads your words and its own once more, and the instrument takes a reading at each of its
        sixty layers where it chose each word-piece it wrote, the first of them where it read your last word (unless you ask for
        the first alone). The sighting on this page took{" "}
        <span className="num">{tookWords(s)?.secs ?? <Missing />}</span>
        {tookWords(s) ? <>, {tookWords(s)!.what}</> : null}.
      </Step>
      <Step icon={<StepRead />} numeral="iii" title="read the ascent">
        Climb the layers one at a time. At each, read what the lens makes of the word, and whether, as the lens weighs it, the
        layer was <span style={{ color: "#2E4A7D" }}>gathering</span> from the words before or{" "}
        <span style={{ color: "#7A5C28" }}>remembering</span> what it knows.
      </Step>
    </ol>

    <div style={{ display: "flex", flexDirection: "column", gap: 14, paddingTop: 12 }}>
      <div className="row-between">
        <h3 className="h-sub">words you will meet</h3>
        <a href={href.lexicon()} style={{ fontSize: 17, padding: "10px 0" }}>
          the whole lexicon →
        </a>
      </div>
      <dl style={{ margin: 0, display: "flex", flexWrap: "wrap", gap: "18px 40px" }}>
        {WORDS.map((w, i) => (
          <div key={i} style={{ flex: "1 1 220px", minWidth: 0, paddingTop: 12, borderTop: `1px solid ${HAIR_SOFT}` }}>
            <dt style={sc("0.16em", 16, INK)}>{w.term}</dt>
            <dd style={{ margin: "4px 0 0", fontSize: 17, color: MUTED }}>{w.def}</dd>
          </div>
        ))}
      </dl>
    </div>
  </section>
);

/* ------------------------------------------------------------ two ways to read */

const TwoWays: React.FC = () => {
  const { rubrics, apparatus, toggleRubrics, toggleApparatus } = useReading();
  return (
    <section aria-labelledby="ways-head" style={{ display: "flex", flexDirection: "column", gap: 24, padding: "34px 0 44px", borderTop: `1px solid ${HAIR}` }}>
      <h2 className="h-sec" id="ways-head">
        two ways to read this instrument
      </h2>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "32px 56px", alignItems: "flex-start" }}>
        <LeafFigure rubrics={rubrics} apparatus={apparatus} />

        <div style={{ flex: "1 1 320px", minWidth: 0, display: "flex", flexDirection: "column", gap: 12 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
            <button type="button" className="reading-toggle rubrics" style={{ fontSize: 14 }} onClick={toggleRubrics} aria-pressed={rubrics}>
              <Lozenge fill={rubrics ? "#8F2B1E" : "none"} stroke="#8F2B1E" />
              rubrics
            </button>
            <span style={{ fontSize: 15, fontStyle: "italic", color: MUTED }}>
              {rubrics ? "on: the red directions are showing" : "off: the red directions are hidden"}
            </span>
          </div>
          <h3 style={{ margin: 0, display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: "2px 12px", fontWeight: 500 }}>
            <span style={{ fontSize: 26, fontStyle: "italic", color: "#8F2B1E" }}>Rubrics</span>
            <span style={sc("0.2em", 14)}>for those beginning</span>
          </h3>
          <p style={{ margin: 0, maxWidth: 460 }}>
            The directions in red italic, like the one near the top of this page. Scribes and early printers set their headings
            and directions apart in red ink, and the name came with the ink: <span style={{ fontStyle: "italic" }}>rubrica</span> is
            Latin for red ochre. Here they tell you what you are looking at and what to try next.
          </p>
          <p style={{ margin: 0, maxWidth: 460, color: MUTED }}>
            They are on when you arrive. Switch them off at any time, here or at the top of every page; nothing else changes.
          </p>
        </div>

        <div style={{ flex: "1 1 320px", minWidth: 0, display: "flex", flexDirection: "column", gap: 12 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
            <button
              type="button"
              className="reading-toggle apparatus"
              style={{ fontSize: 14 }}
              onClick={toggleApparatus}
              aria-pressed={apparatus}
            >
              <Lozenge fill={apparatus ? "#A07A2E" : "none"} stroke="#A07A2E" />
              apparatus
            </button>
            <span style={{ fontSize: 15, fontStyle: "italic", color: MUTED }}>
              {apparatus
                ? "open: on this page it lies beneath the frontispiece, the questions, the cautions and the makers’ marks"
                : "closed"}
            </span>
          </div>
          <h3 style={{ margin: 0, display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: "2px 12px", fontWeight: 500 }}>
            <span style={{ fontSize: 26, fontStyle: "italic", color: "#7A5C28" }}>The apparatus</span>
            <span style={sc("0.2em", 14)}>for scholars</span>
          </h3>
          <p style={{ margin: 0, maxWidth: 460 }}>
            In a critical edition, the apparatus is the small type at the foot of the page that gives the sources and the variant
            readings. Here it opens beneath each part of a page, under a double gilt rule: the technical name, the formula, the raw
            values, the sources and every control the instrument has.
          </p>
          <p style={{ margin: 0, maxWidth: 460, color: MUTED }}>
            It is closed when you arrive, and one click away. Nothing in the instrument is hidden from you.
          </p>
          <a
            href={href.instrument()}
            onClick={() => {
              if (!apparatus) toggleApparatus();
            }}
            style={{ fontSize: 17, padding: "10px 0", alignSelf: "flex-start" }}
          >
            see the instrument with its apparatus open →
          </a>
        </div>
      </div>
    </section>
  );
};

/* ------------------------------------------------------------ begin with a question */

const lineStyle: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  alignItems: "baseline",
  gap: "2px 10px",
  fontSize: 17,
  color: MUTED,
};

const BeginWithAQuestion: React.FC = () => {
  const setQuestion = useTheodoliteStore((s) => s.setQuestion);
  const setAnswerBegins = useTheodoliteStore((s) => s.setAnswerBegins);
  const setChat = useTheodoliteStore((s) => s.setChat);
  const takeSighting = useTheodoliteStore((s) => s.takeSighting);
  const sightEach = useTheodoliteStore((s) => s.writing.sightEach);
  const [sel, setSel] = useState(() => {
    const st = useTheodoliteStore.getState();
    const i = FIRST_QUESTIONS.findIndex((x) => x.send.question === st.question && x.send.answerBegins === st.answerBegins);
    return i >= 0 ? i : 0;
  });
  const cur = FIRST_QUESTIONS[sel];

  const takeIt = () => {
    // send each question as the study put it (bare text, or a chat with the reply begun), and leave the instrument
    // set the same way, so a sighting taken from it next reads the same word
    const { question, answerBegins, chat } = cur.send;
    setQuestion(question);
    setAnswerBegins(answerBegins);
    setChat(chat);
    // a first sighting only reads the model: no nudge, whatever the nudge page holds; and it is written as the study wrote
    // this question (its likeliest word-piece each time, as many as it recorded, no system prompt, no repeat penalty),
    // whatever the instrument is set to, so the live answer can be set beside the recorded one
    takeSighting({
      question,
      answerBegins,
      chat,
      system: "",
      steering: { enabled: false },
      writing: { mode: "greedy", presencePenalty: 0, seed: null, length: cur.recordedLength },
    });
    // if the server is not answering, the store has set sweepError and the instrument shows it
    navigate(href.instrument());
  };

  return (
    <section id="begin" aria-labelledby="begin-head" style={{ display: "flex", flexDirection: "column", gap: 18, padding: "34px 0 44px", borderTop: `1px solid ${HAIR}` }}>
      <h2 className="h-sec" id="begin-head" tabIndex={-1}>
        begin with a question
      </h2>
      <Rubric style={{ maxWidth: 700 }}>
        Choose a question below for your first sighting. Each one was really put to the model, and beneath it is what the model
        went on to say.
      </Rubric>

      <div style={{ display: "flex", flexWrap: "wrap", gap: "36px 56px", alignItems: "flex-start", paddingTop: 6 }}>
        <div role="group" aria-label="First questions" style={{ flex: "999 1 620px", minWidth: 0, display: "flex", flexDirection: "column", borderTop: `1px solid ${HAIR_SOFT}` }}>
          {FIRST_QUESTIONS.map((q, i) => {
            const on = i === sel;
            return (
              <button
                key={i}
                type="button"
                onClick={() => setSel(i)}
                aria-pressed={on}
                style={{
                  width: "100%",
                  display: "grid",
                  gridTemplateColumns: "22px minmax(0, 1fr)",
                  gap: "0 16px",
                  alignItems: "start",
                  textAlign: "left",
                  color: INK,
                  background: on ? "#EDE5D3" : "transparent",
                  border: 0,
                  borderBottom: `1px solid ${HAIR_SOFT}`,
                  borderRadius: 0,
                  padding: "18px 18px 20px 14px",
                  minHeight: 44,
                  cursor: "pointer",
                }}
              >
                <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true" style={{ display: "block", marginTop: 30 }}>
                  <path d="M7 1 L13 7 L7 13 L1 7 Z" fill={on ? "#8F2B1E" : "none"} stroke={on ? "#8F2B1E" : "#A07A2E"} strokeWidth="1.2" />
                </svg>
                <span style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 0 }}>
                  <span style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: "2px 14px", ...sc("0.16em", 15) }}>
                    <span>{q.kind}</span>
                    <span style={{ fontStyle: "italic", letterSpacing: "0.08em" }}>{q.form}</span>
                  </span>
                  <span style={{ fontSize: 24, lineHeight: 1.3, color: INK }}>{q.ask}</span>
                  {q.begins ? (
                    <span style={lineStyle}>
                      <span style={sc("0.16em")}>its answer begins</span>
                      <span style={{ fontStyle: "italic", color: INK }}>{q.begins}</span>
                    </span>
                  ) : null}
                  <span style={{ fontSize: 17, lineHeight: 1.5, color: INK, maxWidth: 680 }}>{q.note}</span>
                  <span style={lineStyle}>
                    <span style={sc("0.16em")}>{q.wentLabel}</span>
                    <span style={{ fontStyle: "italic" }}>{q.went}</span>
                  </span>
                  {q.plain ? (
                    <span style={lineStyle}>
                      <span style={sc("0.16em")}>as plain text</span>
                      <span style={{ fontStyle: "italic" }}>{q.plain}</span>
                    </span>
                  ) : null}
                </span>
              </button>
            );
          })}
        </div>

        <aside
          aria-labelledby="chosen-head"
          style={{ flex: "1 1 340px", minWidth: 0, display: "flex", flexDirection: "column", gap: 16, position: "sticky", top: 24, alignSelf: "flex-start" }}
        >
          <h3 className="h-sub" id="chosen-head">
            your first sighting
          </h3>
          <div aria-live="polite" style={{ display: "flex", flexDirection: "column", gap: 8, paddingBottom: 16, borderBottom: `1px solid ${HAIR}` }}>
            <span style={sc("0.16em", 15)}>
              {cur.kind} · {cur.form}
            </span>
            <p style={{ margin: 0, fontSize: 30, lineHeight: 1.25 }}>{cur.ask}</p>
            {cur.begins ? (
              <p style={{ margin: 0, fontSize: 17, color: MUTED }}>
                its answer begins <span style={{ fontStyle: "italic", color: INK }}>{cur.begins}</span>
              </p>
            ) : null}
          </div>
          <p style={{ margin: 0, fontSize: 17, color: MUTED }}>
            The instrument will follow the last word, <span style={{ color: INK, borderBottom: "2px solid #A07A2E" }}>{cur.follows}</span>,
            up through all sixty layers{sightEach ? ", then each word-piece the model writes" : ""}. The model went on{" "}
            <span style={{ fontStyle: "italic", color: INK }}>{cur.went}</span>
          </p>
          <p style={{ margin: 0, fontSize: 15, color: MUTED }}>
            Taken as the study took it: its likeliest word-piece each time, up to {cur.recordedLength}, with no system prompt and no
            nudge, whatever the instrument is set to.
          </p>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 12, alignItems: "center" }}>
            <button type="button" className="btn-primary" onClick={takeIt}>
              take this sighting
            </button>
            <a
              href={href.instrument()}
              style={{
                display: "inline-flex",
                alignItems: "center",
                minHeight: 48,
                padding: "12px 20px",
                fontSize: 17,
                ...sc("0.16em", undefined, INK),
                background: "transparent",
                textDecoration: "none",
                border: "1px solid #C9BEAC",
                borderRadius: 2,
              }}
            >
              write my own question
            </a>
          </div>
          {cur.caution ? <p style={{ margin: 0, fontStyle: "italic", fontSize: 16, color: "#8F2B1E" }}>{cur.caution}</p> : null}
          <Rubric style={{ fontSize: 16 }}>
            Taking a sighting only reads the model; it changes nothing inside it. You can come back and choose another at any time.
          </Rubric>
        </aside>
      </div>

      <Apparatus title="how each question was put" label="Apparatus for the example questions" style={{ marginTop: 10, gap: 12 }}>
        <dl className="dl num">
          <dt>bare text</dt>
          <dd>
            the words alone, after one start mark (id 2): <span className="mono">lab.ids(p)</span>, then greedy decoding by HF{" "}
            <span className="mono">generate</span>, <span className="mono">max_new_tokens=12</span>. This is how the server sent
            every question before its chat option (REPORT §3.4, bug 5), and how the instrument sends rows 2–5 now, as the study did.
          </dd>
          <dt>as a chat</dt>
          <dd>
            <span className="mono">lab.chat("Continue this text.", assistant=p)</span>: the Gemma-4 chat template with thinking
            off, the question set as the opening of the model’s own reply. Done for the first four bare prompts of the run only.
          </dd>
          <dt>the first</dt>
          <dd>
            “Tell me about yourself.” comes from another run (E6): a chat, with the reply begun “Hi! I”, continued by the sweep
            itself for 6 word-pieces.
          </dd>
          <dt>checks</dt>
          <dd>
            The instrument’s own generator gave HF’s words on every bare prompt; on the capital of Japan it also prints the marks{" "}
            <span className="mono">{"<channel|>"}</span> and <span className="mono">{"<turn|>"}</span> that HF’s decoder hides. A
            nudged call left nothing behind: 0 hooks before it and 0 after, all six continuations unchanged.
          </dd>
          <dt>taken here</dt>
          <dd>
            “take this sighting” sends the question as the study put it and writes the answer as the study wrote it, whatever the
            instrument is set to for its own sightings: its likeliest word-piece each time (<span className="mono">temperature 0</span>,
            no repeat penalty), {FIRST_QUESTIONS[0].recordedLength} word-pieces for the first question as E6 wrote it and 12 for the
            others as V7 did (<span className="mono">max_new_tokens</span>), no system prompt and no nudge. The instrument’s own writing
            and system prompt are left as they were; “sight each word-piece it writes” follows the instrument’s setting. So the live
            answer can be set beside the recorded one (the checks below found the instrument’s generator giving HF’s words on every bare
            prompt). The request as sent is in the frontispiece’s apparatus above, once the sighting is back.
          </dd>
          <dt>the fix</dt>
          <dd>
            a chat-template option in the server (<span className="mono">chat</span>, <span className="mono">system</span>,{" "}
            <span className="mono">assistant_prefix</span>; REPORT §3.4, bug 5), now in the backend and on by default in{" "}
            <a href={href.instrument()}>the instrument</a>; the single-BOS fix (bug 6) gives a chat-wrapped prompt one start mark. Each
            question above is sent as the study put it: rows 2–5 as bare text, the first and the last as chats.
          </dd>
        </dl>
        <div style={{ overflowX: "auto", paddingBottom: 6 }}>
          <div role="table" aria-label="How each question was put, and what the model went on to say" style={{ minWidth: 940, display: "flex", flexDirection: "column", borderTop: `1px solid ${HAIR_SOFT}` }}>
            <div role="row" style={{ display: "grid", gridTemplateColumns: "minmax(0, 1.15fr) minmax(0, 0.75fr) minmax(0, 1.25fr) minmax(0, 1.25fr)", gap: "4px 18px", padding: "8px 8px", borderBottom: `1px solid ${HAIR}`, ...sc("0.14em") }}>
              <span role="columnheader">the question</span>
              <span role="columnheader">sent as</span>
              <span role="columnheader">as bare text, it went on</span>
              <span role="columnheader">as a chat, it went on</span>
            </div>
            {HOW_PUT.map((r, i) => (
              <div
                key={i}
                role="row"
                style={{
                  display: "grid",
                  gridTemplateColumns: "minmax(0, 1.15fr) minmax(0, 0.75fr) minmax(0, 1.25fr) minmax(0, 1.25fr)",
                  gap: "4px 18px",
                  alignItems: "baseline",
                  padding: "10px 8px",
                  borderBottom: `1px solid ${HAIR_FAINT}`,
                  background: i === sel ? "#EDE5D3" : "transparent",
                }}
              >
                <span role="cell" style={{ minWidth: 0 }}>
                  {r.ask}
                </span>
                <span role="cell" style={{ minWidth: 0, color: MUTED }}>
                  {r.how}
                </span>
                <span role="cell" style={{ minWidth: 0 }}>
                  {r.bare ? (
                    <span style={{ fontFamily: "var(--mono)", fontSize: 13, overflowWrap: "anywhere" }}>{r.bare}</span>
                  ) : (
                    <span className="placeholder">[NOT ASKED AS BARE TEXT]</span>
                  )}
                </span>
                <span role="cell" style={{ minWidth: 0 }}>
                  {r.chat ? (
                    <span style={{ fontFamily: "var(--mono)", fontSize: 13, overflowWrap: "anywhere" }}>{r.chat}</span>
                  ) : (
                    <span className="placeholder">[NOT ASKED AS A CHAT]</span>
                  )}
                </span>
              </div>
            ))}
          </div>
        </div>
        <p style={{ margin: 0, color: MUTED, overflowWrap: "anywhere" }}>
          Continuations as recorded, escapes kept; the reader’s view above cuts them at the first sentence. From
          experiments/results/V7_tool_bugs_08_diag.json and its script, experiments/scripts/V7_tool_bugs_08_diag.py (with
          experiments/lab/labkit.py); the first row from E6_tool_audit_02_sweep_demo.json.
        </p>
      </Apparatus>
    </section>
  );
};

/* ------------------------------------------------------------ before you set out */

const NowRow: React.FC<{ term: string; children: React.ReactNode }> = ({ term, children }) => (
  <div style={{ display: "grid", gridTemplateColumns: "112px minmax(0, 1fr)", gap: "4px 18px", alignItems: "baseline", padding: "12px 0", borderBottom: `1px solid ${HAIR_FAINT}` }}>
    <dt style={sc("0.16em", 15)}>{term}</dt>
    {children}
  </div>
);

const CAUTIONS: React.ReactNode[] = [
  <>
    Words read in the middle layers are glimpses. Up to layer 50, on ordinary prose, the lens, fitted on the base model, matched the
    model’s own final choice at most about five times in a hundred, and from about layer 34 to 55 less often than the naked eye.
  </>,
  <>
    Labels on single neurons, read through the lens or the naked eye, held up at the layers tested from 41 to 58, and not at 11 to
    31. Below layer 41 the pages hold a neuron’s words back, and so too where the lens and the naked eye share none of their first
    ten; the apparatus keeps them. At layers 59 and 60, never tested, treat them as rumours.
  </>,
  <>
    A bearing, the compass direction the instrument gives a reading <span style={{ fontStyle: "italic", color: MUTED }}>(azimuth)</span>,
    is a sketch: no bearing was checked against what the model does. As first built, the instrument squared the three landmark
    words in order, and in the study reordering them typically swung a bearing by 58 to 99 degrees. It now squares them all at
    once, so their order no longer changes the frame; it still says which word the bearing is measured from.
  </>,
  <>
    The nudge <span style={{ fontStyle: "italic", color: MUTED }}>(steering)</span> breaks the model’s speech if it pushes too hard. The
    server’s own default is that hard; the nudge page sends the study’s mended recipe at one hundredth of the stream, and even that
    window is narrow.
  </>,
  <>
    Sent as plain text, without the chat wrapping the model was trained on{" "}
    <span style={{ fontStyle: "italic", color: MUTED }}>(chat template)</span>, two of six prompts in the study came back as one word
    repeated. The instrument wraps your words as a chat unless you turn it off.
  </>,
];

const BeforeYouSetOut: React.FC = () => {
  const conn = useTheodoliteStore((s) => s.conn);
  const health = useTheodoliteStore((s) => s.health);
  const isSweeping = useTheodoliteStore((s) => s.isSweeping);
  const progress = useTheodoliteStore((s) => s.incoming?.progress);

  let stateWord: string;
  let stateFill: string;
  if (conn === "open") {
    stateWord = isSweeping ? (progressWords(progress) ?? "taking a sighting…") : "ready";
    stateFill = "#A07A2E";
  } else if (conn === "connecting" || conn === "idle") {
    stateWord = "reaching the server…";
    stateFill = "#C9BEAC";
  } else {
    stateWord = "not answering";
    stateFill = "#8F2B1E";
  }
  const vram = conn === "open" && health ? health.vram_allocated_gb : null;
  const { lens } = useInstrumentLens();

  return (
    <section aria-labelledby="set-out-head" style={{ display: "flex", flexDirection: "column", gap: 22, padding: "34px 0 8px", borderTop: `1px solid ${HAIR}` }}>
      <h2 className="h-sec" id="set-out-head">
        before you set out
      </h2>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "36px 72px", alignItems: "flex-start" }}>
        <div style={{ flex: "1 1 380px", minWidth: 0, display: "flex", flexDirection: "column", gap: 6 }}>
          <h3 className="h-sub">the instrument now</h3>
          <dl style={{ margin: 0, display: "flex", flexDirection: "column" }}>
            <NowRow term="the model">
              <dd style={{ margin: 0, fontSize: 18 }}>
                Gemma-4-31B-it <span style={{ fontSize: 16, color: MUTED }}>· read in 4-bit</span>
              </dd>
            </NowRow>
            <NowRow term="its state">
              <dd role="status" style={{ margin: 0, fontSize: 18, display: "flex", alignItems: "center", gap: 8 }}>
                <Diamond10 fill={stateFill} />
                {stateWord}
              </dd>
            </NowRow>
            <NowRow term="memory">
              <dd style={{ margin: 0, display: "flex", flexDirection: "column", gap: 8 }}>
                {vram != null ? (
                  <>
                    <span className="num" style={{ fontSize: 18 }}>
                      {vram.toFixed(1)} of {MEMORY_ENVELOPE_GB} GB
                    </span>
                    <span aria-hidden="true" className="bar-track" style={{ display: "block", maxWidth: 320 }}>
                      <span className="bar-fill" style={{ width: `${Math.min(100, (vram / MEMORY_ENVELOPE_GB) * 100).toFixed(1)}%` }} />
                    </span>
                  </>
                ) : (
                  <span style={{ fontSize: 18 }}>
                    <Missing what="read from the server when it runs" />
                  </span>
                )}
              </dd>
            </NowRow>
            <NowRow term="the lens">
              <dd style={{ margin: 0, fontSize: 18 }}>
                {lens.name}{" "}
                <span style={{ fontSize: 16, fontStyle: "italic", color: MUTED }}>
                  {isBaseFitted(lens)
                    ? "fitted on the base model, here used on its chat-trained sibling (the instruct model)"
                    : `fitted on ${lens.fittedOn}`}
                </span>
              </dd>
            </NowRow>
            <NowRow term="its depth">
              <dd className="num" style={{ margin: 0, fontSize: 18 }}>
                60 layers · 5,376 dimensions
              </dd>
            </NowRow>
          </dl>
        </div>

        <div style={{ flex: "999 1 560px", minWidth: 0, display: "flex", flexDirection: "column", gap: 12 }}>
          <h3 className="h-sub">where it can mislead</h3>
          <Rubric style={{ maxWidth: 700 }}>
            This is a research instrument. Its readings are evidence, not verdicts: set one sighting beside another before you trust
            what either seems to say.
          </Rubric>
          <ul style={{ listStyle: "none", margin: 0, padding: 0, display: "flex", flexDirection: "column", borderTop: `1px solid ${HAIR_FAINT}` }}>
            {CAUTIONS.map((c, i) => (
              <li
                key={i}
                style={{ display: "grid", gridTemplateColumns: "14px minmax(0, 1fr)", gap: "0 14px", alignItems: "baseline", padding: "12px 0", borderBottom: `1px solid ${HAIR_FAINT}`, fontSize: 18 }}
              >
                <span aria-hidden="true" style={{ color: "#A07A2E", fontSize: 11 }}>
                  ◆
                </span>
                <span>{c}</span>
              </li>
            ))}
          </ul>
          <div style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: "4px 16px" }}>
            <a href={href.lexicon()} style={{ fontSize: 17, padding: "10px 0" }}>
              read the study
            </a>
            <span style={{ fontSize: 15, color: MUTED }}>its findings, and every fault it found in this instrument</span>
          </div>
        </div>
      </div>

      <Apparatus title="the instrument and its faults" label="Apparatus for the instrument and its faults">
        <dl className="dl num">
          <dt>the model</dt>
          <dd>
            <span className="mono-wrap">google/gemma-4-31B-it-qat-w4a16-ct</span> (Gemma 4) [1], its 4-bit W4A16 weights unpacked on
            the GPU as needed (backend/engine_runtime.py); 60 layers, d<sub>model</sub> 5,376, d<sub>ff</sub> 21,504, a vocabulary of
            262,144; global attention at layers 6, 12, 18 … 60 (backend/theodolite_math.py), the rest looking back over 1,024
            word-pieces (ARCHITECTURE.md).
          </dd>
          <dt>its state</dt>
          <dd>
            the server answers <span className="mono">GET /api/health</span> with model_id, vram_allocated_gb, vram_peak_gb and
            larql_cli; the memory envelope is under 30.0 GB on one RTX 5090 (ARCHITECTURE.md). Live figures:{" "}
            {conn === "open" && health ? (
              <>
                model_id <span className="mono-wrap">{health.model_id}</span>, vram_allocated_gb {health.vram_allocated_gb.toFixed(2)},
                vram_peak_gb {health.vram_peak_gb.toFixed(2)}, larql_cli{" "}
                <span className="mono-wrap">{JSON.stringify(health.larql_cli)}</span>
              </>
            ) : (
              <Missing what="read from the server when it runs" />
            )}
          </dd>
          <dt>the lens</dt>
          <dd>
            J++ (Ayonrinde &amp; Lindsey 2026) [4], fitted on the base model in bf16 and used on the instruct model at 4-bit (REPORT
            §3.4, bug 9); only its matrix is applied, not its filter of non-semantic word-pieces. The Jacobian lens [2] is on disk;
            the server starts with <span className="mono">load_jlens=False</span>. R1 refitted a lens on the instruct checkpoint:
            along 128 fixed directions, stable at layers 31–59, where it agrees with the model’s own average gradients at 0.85–0.95
            against J++’s 0.15–0.33; and as a full matrix at 11 layers, usable only at 41, 46, 51, 56 and 59 (top-1 on the model’s
            own reply at layer 51: 0.25 → 0.45). Neither is loaded; every lens reading here is through the base-fitted J++ (R1; §3.1).
          </dd>
          <dt>glimpses</dt>
          <dd>
            top-1 at most about 5% through layer 50 on prose (5.1–5.2% at layers 46–47; §3.1’s “3%” matches only the average
            over layers 41–50 and understates it: erratum E30), 28% at layer 59. J++ about equal to the naked eye at layers 46–59
            in KL; in top-1 below it from about layer 34 to 55 and about equal above (E3). A lens fitted on the instruct model
            beats the naked eye at layers 46–59 (R1). With the base-fitted J++, compare conditions at layers 31–50, against a
            matched control, and confirm with an instruct-fitted reading where one exists; at layers 21–30 only the study’s
            orderings checked against behaviour held, since no averaged lens is stable below layer 31 (R1); below layer 21
            nothing was validated (erratum E49; §3.5).
          </dd>
          <dt>neuron labels</dt>
          <dd>
            output labels against causal effect, P@10 0.35–0.67 for J++ at the layers sampled from 41 to 58, the naked eye’s about
            the same (null ≤ 0.003), at chance at layers 11–31; 32–40 not sampled; labels through an instruct-fitted lens untested
            (§3.2; N1). Within layers 41–58, agreement between the lens and the naked eye predicts validity (AUC 0.84, for
            base-fitted J++ labels on random neurons). Fix: stored activation contexts for each pole; output labels only from
            layer 41, gated by that agreement (§3.5; bug 7). Now done for the layer and the agreement: the server marks every label
            it sends (<span className="mono">label_check</span>, the word-pieces both readings share in their first ten, and{" "}
            <span className="mono">label_gate</span>, the rule), and the pages hold a label back below layer 41, and where the two
            share none of ten (the cut is the instrument’s), keeping it in the apparatus. The causal spot-check is done in part: a
            neuron’s folio puts one label at a time to the test when asked (<span className="mono">causal_spot_check</span>), and
            nothing tests every label. Stored activation contexts are not built.{" "}
            <a href={href.vault()}>the vault</a>
          </dd>
          <dt>bearing</dt>
          <dd>
            as first built, Gram–Schmidt on the raw W<sub>U</sub> rows, without the final-norm weight: 58–99° across orders, and
            the frame catches 2.4–3.9% of ‖h<sub>J</sub>‖ (bug 3). Fix: a Löwdin frame (Löwdin 1950) [33] on fnw · W<sub>U</sub>,
            with the captured fraction shown. The interface now sends a Löwdin frame by default,{" "}
            {FNW_DEFAULT ? (
              <>
                on fnw · W<sub>U</sub>, which V7 did not measure
              </>
            ) : (
              <>
                on the raw rows, where V7 measured it (it agreed with the sign of the lens’s she−he score in 79–89% of readings by
                stretch of layers, against 72–88% squared in order); fnw · W<sub>U</sub> is a choice V7 did not measure
              </>
            )}
            , with the fraction caught sent at every layer; the server’s own default stays as first built, for old callers.{" "}
            <a href={href.landmarks()}>landmarks</a>
          </dd>
          <dt>nudge</dt>
          <dd>
            default scale 650 at layers 30, 36, 42, 48 and 54 (index 29 … 53), with <span className="mono">normalize=True</span>:
            2.4–7.4× the median ‖h‖, and no position mask (bug 1). Fix: scale relative to ‖h‖ (about 0.02), direction fnw · ΔW
            <sub>U</sub>, quiet dimensions masked, a position mask (§3.4; §3.5); now in the server, and what the nudge page sends.{" "}
            <a href={href.nudge()}>nudge</a>
          </dd>
          <dt>chat</dt>
          <dd>
            no chat template anywhere in the server; 2 of 6 bare prompts came back as one word repeated (bug 5). Fix: a
            chat-template option, on by default; now in the server, with an optional system turn, and on by default in the instrument.
          </dd>
          <dt>since the study</dt>
          <dd>
            the two verified patches are in the backend: V7_module1_fix.patch (the gathering and remembering split, with the layer
            scalar s; bug 2) and V7_bos_fix.patch (one start mark; bug 6). Since then the server also takes the fixes for bugs 1, 3,
            4, 5 and 8 as options (the mended nudge, the Löwdin and weighted frames with the fraction caught, the direct-path thread
            score, the chat template, poles ranked by write size); the instrument sends the nudge, chat, frame, thread-score and
            write-ranking fixes by default (the frame squared all at once,{" "}
            {FNW_DEFAULT ? "on the weighted rows" : "on the raw rows; the weighted rows stay a choice"}), while the server’s own
            frame, ranking and thread defaults stay as first built, for old callers. Still open: bugs 7 and 9.
          </dd>
        </dl>
      </Apparatus>
    </section>
  );
};

/* ------------------------------------------------------------ the makers' marks */

const MAKERS: { numeral: string; what: string; name: string; who: string }[] = [
  { numeral: "i", what: "reading what each layer holds as the words it could say", name: "the Jacobian lens and J-space", who: "Gurnee et al., Anthropic, 2026" },
  { numeral: "ii", what: "a clearer, more faithful lens", name: "the J++ lens", who: "Ayonrinde & Lindsey, 2026" },
  {
    numeral: "iii",
    what: "every neuron has two faces, a positive and a negative",
    name: "neurons as two-poled units",
    who: "Cammarata, “Neural Networks Are Already Clean”, Goodfire, 2026",
  },
  { numeral: "iv", what: "a catalogue of neurons you can search and question", name: "the LARQL / VINDEX vault", who: "Hayuk" },
];

const MakersMarks: React.FC = () => (
  <section aria-labelledby="makers-head" style={{ display: "flex", flexDirection: "column", gap: 22, marginTop: 40, padding: "34px 0 8px", borderTop: `1px solid ${HAIR}` }}>
    <h2 className="h-sec" id="makers-head">
      the makers’ marks
    </h2>

    <div style={{ position: "relative", width: "100%", maxWidth: 1040, margin: "0 auto", padding: "56px clamp(26px, 5vw, 68px) 50px" }}>
      <MakersFrame />
      <div style={{ position: "relative", display: "flex", flexDirection: "column", gap: 20 }}>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8, textAlign: "center" }}>
          <p style={{ margin: 0, ...sc("0.32em", 17, "#7A5C28") }}>on whose shoulders</p>
          <p style={{ margin: 0, maxWidth: 640, fontSize: 18, fontStyle: "italic", color: MUTED }}>
            Like the maker’s name engraved on a brass instrument, these are the names behind this one: four lines of recent research
            that it brings together.
          </p>
        </div>
        <ol style={{ listStyle: "none", margin: 0, padding: 0, display: "flex", flexDirection: "column", borderTop: `1px solid ${HAIR_SOFT}` }}>
          {MAKERS.map((m) => (
            <li
              key={m.numeral}
              style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: "4px 20px", padding: "14px 0 15px", borderBottom: `1px solid ${HAIR_SOFT}` }}
            >
              <span aria-hidden="true" style={{ flex: "0 0 36px", fontSize: 30, lineHeight: 1, fontStyle: "italic", color: "#8A6A2C" }}>
                {m.numeral}
              </span>
              <span style={{ flex: "1 1 180px", minWidth: 0, fontSize: 20, lineHeight: 1.4 }}>{m.what}</span>
              <span style={{ flex: "1.4 1 340px", minWidth: 0, fontSize: 16, lineHeight: 1.5, color: MUTED }}>
                <span style={sc("0.12em", undefined, INK)}>{m.name}</span> · {m.who}
              </span>
            </li>
          ))}
        </ol>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2, textAlign: "center" }}>
          <p style={{ margin: 0, maxWidth: 640, fontSize: 16, color: MUTED }}>
            How they are joined here is this project’s own work; where it falls short, the fault is the instrument’s, not theirs.
          </p>
          <a href={href.lineage()} style={{ fontSize: 17, padding: "10px 0" }}>
            the full lineage →
          </a>
        </div>
      </div>
    </div>

    <Apparatus title="the makers’ marks" label="Apparatus for the makers’ marks">
      <dl className="dl num">
        <dt>i · Jacobian lens</dt>
        <dd>
          Gurnee et al. 2026, Transformer Circuits Thread [2]; code, the jlens package [3]; fitted lens files from Neuronpedia [6].
          Readout unembed(h · J<sub>ℓ</sub>
          <sup>⊤</sup>); the default two-hop question (the spider and its eight legs) comes from the same paper.
        </dd>
        <dt>ii · J++ lens</dt>
        <dd>
          Ayonrinde &amp; Lindsey 2026 [4], on the R-lens LRP rules (Blank, Bhatia &amp; Nanda 2026) [5]: it down-weights poorly read
          clusters, takes Jacobians by an LRP backward pass, and drops non-semantic word-pieces from the readout. Fitted on the base
          model over 64 × 128-token WikiText-103 sequences [7]. The study: fitted on the base model and used on the instruct model,
          a good comparative instrument (from about layer 21 it orders conditions as the model’s behaviour does) and a poor
          absolute one (REPORT §3.1).
        </dd>
        <dt>iii · poles</dt>
        <dd>
          Cammarata 2026 [9], on the GeGLU block (Shazeer 2020) [15]: each neuron’s positive and negative directions are its two
          poles. The study: Gemma-4’s neurons are truly two-poled, through the sign of the up-projection, and the two poles fire on
          unrelated inputs; output labels, read through J++ or the naked eye, held up at the layers sampled from 41 to 58 and not
          at 11–31 (REPORT §3.2). The instrument now holds a label back below layer 41, and where its two readings share none of
          their first ten word-pieces (REPORT §3.3).
        </dd>
        <dt>iv · the vault</dt>
        <dd>
          Hayuk (chrishayuk), LARQL, the reference implementation of VINDEX3 [10]. The names come from LARQL; backend/larql_vault.py
          calls it only for <span className="mono">--version</span>, and its <span className="mono">EDGES/POLES NEAREST</span>,{" "}
          <span className="mono">.patch</span> and <span className="mono">ROLLBACK</span> are this project’s own. An edit (.patch) is
          a single-neuron column nudge, not MEMIT [11] (REPORT §3.3).
        </dd>
        <dt>also</dt>
        <dd>
          the logit lens (nostalgebraist 2020) [8], the naked eye; linear CKA (Kornblith et al. 2019) [13], the kinship of layers;
          the transformer-circuits framework (Elhage et al. 2021) [14], gathering and remembering as additive writes; activation
          steering [21] [22], the nudge.
        </dd>
        <dt>the verdict</dt>
        <dd>
          “Neat as a question, oversold as built” (REPORT §3.3). The synthesis earns its keep if neuron labels are grounded in stored
          activations, kept to layer 41 and above (index 40 and above), gated by agreement between the lens and the naked eye, and
          causally spot-checked.
        </dd>
      </dl>
    </Apparatus>
  </section>
);

/* ------------------------------------------------------------ the page */

export const Incipit: React.FC<{ route: Route }> = ({ route }) => {
  const sighting = useTheodoliteStore((s) => s.sighting);
  const geo = useMemo(() => frontispieceOf(sighting), [sighting]);

  // name the word-piece shown: a sweep can sight every word-piece it writes, and the frontispiece draws one of them
  const nShown = sighting?.answerIds?.length ?? null;
  const frontNote =
    sighting?.source === "live"
      ? `the frontispiece is the live sighting shown${
          nShown ? `, word-piece ${sighting.tokenIndex + 1} of ${nShown}` : ""
        }${sighting.readAt != null ? `, reading ${pieceWord(sighting.readAt)}` : ""}`
      : sighting
        ? "the frontispiece is the study’s demonstration sighting (E6), taken before the two V7 fixes"
        : "no sighting yet";

  return (
    <Page
      route={route}
      folio="incipit"
      footer={
        <>
          <ReadingThrough /> · 60 layers · 5,376 dimensions · {frontNote} ·{" "}
          <a href={href.lineage()}>the lineage of these ideas</a>
        </>
      }
    >
      <section aria-labelledby="title-head" style={{ display: "flex", flexWrap: "wrap", gap: "48px 64px", padding: "52px 0 44px", alignItems: "flex-start" }}>
        <TitleColumn />
        <FrontispieceFigure sighting={sighting} geo={geo} />
      </section>

      <FrontApparatus s={sighting} geo={geo} />

      <SectionDivider />
      <HowItWorks s={sighting} />
      <TwoWays />
      <BeginWithAQuestion />
      <BeforeYouSetOut />
      <MakersMarks />
    </Page>
  );
};

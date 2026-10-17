import React, { useMemo, useState } from "react";
import { Apparatus, Lozenge, Missing, Page, Rubric } from "../components/kit";
import { useReading } from "../reading";
import { href, navigate, type Route } from "../router";
import { type Sighting, useTheodoliteStore } from "../store/useTheodoliteStore";
import { comma, numberWord, wp } from "../lib/words";
import {
  FIRST_QUESTIONS,
  HOW_PUT,
  MEMORY_ENVELOPE_GB,
  RECORDED_CALL,
  RECORDED_MAX_NEW_TOKENS,
} from "./incipit/data";
import { type Frontispiece, FrontispieceFigure, frontispieceOf } from "./incipit/Frontispiece";
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
        has been mapped. This instrument lets you follow a single word as it rises, and read, at every layer, what the model
        was thinking: sometimes in words you never wrote, in languages you never used.
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
  const maxNewTokens = useTheodoliteStore((st) => st.maxNewTokens);
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

  const call = recorded || !s
    ? RECORDED_CALL
    : '{"action": "sweep",\n' +
      ' "payload": {"prompt": ' + JSON.stringify(s.textSent ?? "") +
      ', "monuments": ' + JSON.stringify(s.landmarks) +
      ', "survey_mode": ' + JSON.stringify(s.surveyMode) +
      ', "max_new_tokens": ' + maxNewTokens + "}\n}";

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
                Sent by the server as bare text, with no chat template (REPORT §3.4, bug 5):{" "}
                <span className="mono-wrap">{JSON.stringify(s.textSent ?? "")}</span>. The same sighting stands on folio i,{" "}
                <a href={href.instrument()}>the instrument</a>.
              </>
            )}
          </dd>

          <dt>word-pieces</dt>
          <dd className="num">
            {s?.nTokens != null ? s.nTokens : <Missing />} in all; read at the last,{" "}
            {s?.readAt != null ? <>“{s.readAt}”</> : <Missing />} (position {s?.readPos != null ? s.readPos : <Missing />},
            counting from 0).
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
                {" "}The word-pieces: <span className="mono-wrap">{s.promptTokens.map((t) => JSON.stringify(t)).join(", ")}</span>.
              </>
            ) : null}
          </dd>

          <dt>continuation</dt>
          <dd className="num">
            greedy, <span className="mono">max_new_tokens={recorded ? RECORDED_MAX_NEW_TOKENS : maxNewTokens}</span>:{" "}
            {s ? <>“{s.continuation}”</> : <Missing />}. The sweep took{" "}
            {s?.elapsedMs != null ? `${comma(s.elapsedMs)} ms` : <Missing />}
            {recorded ? " (6.25 s by the wall clock)" : null}.
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
            {s && s.surveyMode !== "monument" ? (
              <>the Gram–Schmidt frame of a locked neuron’s motif (its gate, its corrected write and the gate of its main reader)</>
            ) : !lm.every(Boolean) ? (
              <>the Gram–Schmidt frame of the three landmark words’ rows of W<sub>U</sub> (their names were not reported)</>
            ) : (
              <>
                the Gram–Schmidt frame of W<sub>U</sub>[{lm[0]}], W<sub>U</sub>[{lm[1]}], W<sub>U</sub>[{lm[2]}]
              </>
            )}
            ; the radius is schematic. The frame catches{" "}
            {projFracs.length ? (
              `${(Math.min(...projFracs) * 100).toFixed(1)}–${(Math.max(...projFracs) * 100).toFixed(1)}%`
            ) : (
              <Missing what="not recorded in this sighting" />
            )}{" "}
            of ‖h̃‖ across the sixty layers, and the bearing turns with the order of the three words (58–99°, bug 3).{" "}
            <a href={href.landmarks()}>landmarks</a>
          </dd>

          <dt>caveat</dt>
          <dd className="num">
            On prose, the lens’s first choice matched the model’s own at most 3% of the time through layer 50, and 28% at layer 59
            (REPORT §3.1). The middle stations are glimpses, and this lone sighting has no matched control beside it (§3.5).
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
                  : "rebuilt from this sighting’s own record (its prompt, landmarks and survey mode); the page sends it over /ws/theodolite with the nudge settings, and the server adds no chat template."}
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
        knows <span style={{ fontStyle: "italic" }}>(feed-forward)</span>, in different measure.
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
        you give it.
      </Step>
      <Step icon={<StepSight />} numeral="ii" title="take a sighting">
        The model reads your words once, and the instrument takes a reading at each of its sixty layers. The sighting on this
        page took{" "}
        <span className="num">{s?.elapsedMs != null ? `${(s.elapsedMs / 1000).toFixed(1)} seconds` : <Missing />}</span>.
      </Step>
      <Step icon={<StepRead />} numeral="iii" title="read the ascent">
        Climb the layers one at a time. At each, read what the model was thinking, and whether it was{" "}
        <span style={{ color: "#2E4A7D" }}>gathering</span> from the words before or <span style={{ color: "#7A5C28" }}>remembering</span>{" "}
        what it knows.
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
  const takeSighting = useTheodoliteStore((s) => s.takeSighting);
  const [sel, setSel] = useState(() => {
    const q = useTheodoliteStore.getState().question;
    const i = FIRST_QUESTIONS.findIndex((x) => x.question === q);
    return i >= 0 ? i : 0;
  });
  const cur = FIRST_QUESTIONS[sel];

  const takeIt = () => {
    setQuestion(cur.question);
    setAnswerBegins(cur.answerBegins);
    // a first sighting only reads the model: no nudge, whatever the nudge page holds
    takeSighting({ question: cur.question, answerBegins: cur.answerBegins, steering: { enabled: false } });
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
            up through all sixty layers. The model went on <span style={{ fontStyle: "italic", color: INK }}>{cur.went}</span>
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
            <span className="mono">generate</span>, <span className="mono">max_new_tokens=12</span>. This is how the server sends
            every question today (REPORT §3.4, bug 5).
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
          <dt>the fix</dt>
          <dd>
            a chat-template option in the server, on by default (REPORT §3.4, bug 5). The single-BOS fix (bug 6) is already in the
            backend, so a chat-wrapped prompt now gets one start mark. Until then, “send it as a chat” on{" "}
            <a href={href.instrument()}>the instrument</a> asks for it.
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
    Words read in the middle layers are glimpses. Up to layer 50, on ordinary prose, the lens’s first guess matched the model’s own
    final choice at most three times in a hundred.
  </>,
  <>Labels on single neurons held up only from layer 41 to layer 58. Below that, treat them as rumours.</>,
  <>
    A bearing, the compass direction the instrument gives a reading <span style={{ fontStyle: "italic", color: MUTED }}>(azimuth)</span>,
    depends on the order of the three landmark words. In the study, reordering them typically swung it by 58 to 99 degrees.
  </>,
  <>
    At its default strength, the nudge <span style={{ fontStyle: "italic", color: MUTED }}>(steering)</span> breaks the model’s speech.
    The fault is known and not yet fixed.
  </>,
  <>
    Today the server sends your words as plain text, without the chat wrapping the model was trained on{" "}
    <span style={{ fontStyle: "italic", color: MUTED }}>(chat template)</span>. In the study, two of six plain prompts came back as one
    word repeated.
  </>,
];

const BeforeYouSetOut: React.FC = () => {
  const conn = useTheodoliteStore((s) => s.conn);
  const health = useTheodoliteStore((s) => s.health);
  const isSweeping = useTheodoliteStore((s) => s.isSweeping);

  let stateWord: string;
  let stateFill: string;
  if (conn === "open") {
    stateWord = isSweeping ? "taking a sighting…" : "ready";
    stateFill = "#A07A2E";
  } else if (conn === "connecting" || conn === "idle") {
    stateWord = "reaching the server…";
    stateFill = "#C9BEAC";
  } else {
    stateWord = "not answering";
    stateFill = "#8F2B1E";
  }
  const vram = conn === "open" && health ? health.vram_allocated_gb : null;

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
                J++{" "}
                <span style={{ fontSize: 16, fontStyle: "italic", color: MUTED }}>
                  fitted on the base model, here used on its chat-trained sibling (the instruct model)
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
            the server starts with <span className="mono">load_jlens=False</span>. Proposed: refit on the instruct checkpoint; a lens
            refitted there agrees with the model’s own average gradients at 0.85–0.95, where J++ reaches 0.15–0.33 (R1; §3.1, §3.5).
          </dd>
          <dt>glimpses</dt>
          <dd>
            top-1 at most 3% through layer 50 on prose, 28% at layer 59; J++ about equal to the naked eye at layers 46–59 (§3.1).
            Prefer layers 21–50 for comparisons, against a matched control (§3.5).
          </dd>
          <dt>neuron labels</dt>
          <dd>
            output labels against causal effect, P@10 0.35–0.69 at layers 41–58 (null ≤ 0.003), at chance at layers 11–31 (§3.2);
            within layers 41–58, agreement between the lens and the naked eye predicts validity (AUC 0.84). Fix: stored activation
            contexts for each pole; output labels only from layer 41, gated by that agreement (§3.5; bug 7).{" "}
            <a href={href.vault()}>the vault</a>
          </dd>
          <dt>bearing</dt>
          <dd>
            Gram–Schmidt on the raw W<sub>U</sub> rows, without the final-norm weight: 58–99° across orders, and the frame catches
            2.4–3.9% of ‖h<sub>J</sub>‖ (bug 3). Fix: a Löwdin frame (Löwdin 1950) [33] on fnw · W<sub>U</sub>, with the captured
            fraction shown. <a href={href.landmarks()}>landmarks</a>
          </dd>
          <dt>nudge</dt>
          <dd>
            default scale 650 at layers 30, 36, 42, 48 and 54 (index 29 … 53), with <span className="mono">normalize=True</span>:
            2.4–7.4× the median ‖h‖, and no position mask (bug 1). Fix: scale relative to ‖h‖ (about 0.02), direction fnw · ΔW
            <sub>U</sub>, quiet dimensions masked, a position mask (§3.4; §3.5). <a href={href.nudge()}>nudge</a>
          </dd>
          <dt>chat</dt>
          <dd>
            no chat template anywhere in the server; 2 of 6 bare prompts came back as one word repeated (bug 5). Fix: a
            chat-template option, on by default.
          </dd>
          <dt>since the study</dt>
          <dd>
            the two verified patches are in the backend: V7_module1_fix.patch (the gathering and remembering split, with the layer
            scalar s; bug 2) and V7_bos_fix.patch (one start mark; bug 6). Still open: bugs 1, 3, 4, 5, 7, 8 and 9.
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
          model over 64 × 128-token WikiText-103 sequences [7]. The study: a good comparative instrument and a poor absolute one
          (REPORT §3.1).
        </dd>
        <dt>iii · poles</dt>
        <dd>
          Cammarata 2026 [9], on the GeGLU block (Shazeer 2020) [15]: each neuron’s positive and negative directions are its two
          poles. The study: Gemma-4’s neurons are truly two-poled, through the sign of the up-projection, and the two poles fire on
          unrelated inputs; output labels held up at layers 41–58 only (REPORT §3.2).
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

  const frontNote =
    sighting?.source === "live"
      ? "the frontispiece is the latest live sighting"
      : sighting
        ? "the frontispiece is the study’s demonstration sighting (E6), taken before the two V7 fixes"
        : "no sighting yet";

  return (
    <Page
      route={route}
      folio="incipit"
      footer={
        <>
          Reading google/gemma-4-31b-it (4-bit) through the J++ lens · 60 layers · 5,376 dimensions · {frontNote} ·{" "}
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

import React, { useEffect, useMemo, useState } from "react";
import { Apparatus, IfRubrics, LensCaution, LensRecord, Missing, Page, Rubric, Wp } from "../components/kit";
import { ControlSighting } from "../components/ControlSighting";
import { href, navigate, type Route } from "../router";
import {
  DEFAULT_NUDGE_LAYERS,
  N_LAYERS,
  VOCAB,
  chosenOf,
  chosenPhrase,
  endMarkWords,
  exportAnswer,
  exportSighting,
  isChosenRow,
  leadOf,
  maxAct,
  rankingWords,
  readWhere,
  readsInsideAnswer,
  retakeIfLive,
  useShownSet,
  useTheodoliteStore,
  type LayerReading,
  type PoleReading,
  type Sighting,
  type TokenReading,
} from "../store/useTheodoliteStore";
import { certainty, comma, firstWords, gloss, kindOf, minus, ms, pct, pieceWord, signed, wp } from "../lib/words";
import { describeFrame } from "../lib/frame";
import { lensLabel, useReadingLens } from "../lib/lens";
import { NotAMatchedPair, OneDrawLine, PenaltyLine, TemplateEndCaution } from "../lib/cautions";
import { endedWords } from "./instrument/moment";
import { KINSHIP_GROWS } from "./instrument/Kinship";
import { Dial } from "./layerLeaf/Dial";
import { AnswerMarks, CHOSEN_MARK, READ_MARK, useAnswerPieces } from "./layerLeaf/AnswerMarks";
import {
  L36_COS_ATTN,
  L36_COS_FFN,
  L36_IDX,
  L36_POLES,
  L36_PROJ_NORM,
  L36_SUBTITLE,
  L36_TRACKED_LOGITS,
  N1_LAYERS,
} from "./layerLeaf/recorded";
import {
  ORDW,
  bearingFigure,
  bearingWords,
  capital,
  cardinal,
  elevationWords,
  glossLine,
  isPicto,
  listAnd,
  ordinal,
  pct1,
  pictoRaw,
  pictoWords,
  roman,
  sci,
  uPlus,
  wayUp,
} from "./layerLeaf/text";

/**
 * A layer's leaf (canvas: Layer.dc.html, "A layer's leaf — Layer 36"), for any of the sixty layers of the
 * current sighting. Every reading comes from `sighting.layers[idx]`; what the sighting does not carry is a
 * bracketed placeholder. The canvas's hand-written copy about layer 36 appears for layer 36 of the
 * recorded demonstration sighting; elsewhere the same sentences are built from the data.
 */

/* ------------------------------------------------------------------ setup */

interface Ctx {
  s: Sighting;
  L: LayerReading;
  idx: number;
  no: number;
  live: boolean;
  /** layer 36 of the recorded demonstration sighting: the canvas's hand-written copy applies */
  hand: boolean;
  /** placeholder wording for a value this sighting does not carry */
  miss: string;
}

const MUTED = "#6B5D4D";
const HAIR = "#D9CFBD";
const HAIR_SOFT = "#E3DACA";
const HAIR_FAINT = "#ECE4D5";
const SMALL_CAPS: React.CSSProperties = { fontVariantCaps: "all-small-caps" };
const NUM: React.CSSProperties = { fontVariantNumeric: "lining-nums tabular-nums" };
const LINK44: React.CSSProperties = { display: "inline-flex", alignItems: "center", minHeight: 44 };
const SR_ONLY: React.CSSProperties = {
  position: "absolute",
  width: 1,
  height: 1,
  overflow: "hidden",
  clip: "rect(0 0 0 0)",
  whiteSpace: "nowrap",
};
const PRE: React.CSSProperties = {
  margin: 0,
  padding: "8px 0",
  borderTop: `1px solid ${HAIR_SOFT}`,
  borderBottom: `1px solid ${HAIR_SOFT}`,
  fontFamily: "var(--mono)",
  fontSize: 12.5,
  lineHeight: 1.6,
  color: "#2B2117",
  whiteSpace: "pre-wrap",
  overflowWrap: "anywhere",
};

const M: React.FC<{ c: Ctx }> = ({ c }) => <Missing what={c.miss} />;
const NotRead: React.FC = () => <Missing what="not read in this sighting" />;
const Mono: React.FC<{ children: React.ReactNode }> = ({ children }) => <span className="mono-wrap">{children}</span>;

const two = (v: number) => minus(v.toFixed(2));
const fmt3 = (v: number) => minus(v.toPrecision(3));
const top1 = (L: LayerReading): TokenReading | null => (L.lensTop && L.lensTop.length ? L.lensTop[0] : null);
/** The word-piece read, quoted; a new line or a bare space read later in the answer says which (“¶”, “␣”). */
const lastWord = (s: Sighting) => (s.readAt ? pieceWord(s.readAt) : "the last word-piece");
/**
 * What it went on to say from here, by the piece this position chose (matched by id, never by the continuation's first
 * word, which is the wrong word for every later word-piece): “It chose “a”.” The recorded sighting keeps no chosen id, so
 * its recorded words follow instead (it was read where it chose its first word-piece, greedily).
 */
const saidHere = (s: Sighting): string | null => {
  const c = chosenOf(s);
  if (c) return c.endMark ? `Here ${endedWords(c.id, c.token)}.` : `It chose ${pieceWord(c.token)}.`;
  const went = s.continuation.trim();
  return s.tokenIndex === 0 && went ? `It went on “… ${firstWords(went, 40)}”.` : null;
};
/** Where the sighting was read, for a line of facts: “read at “I”, in its reply “Hi! I”” or “read at “am”, the 2nd
 * word-piece of its own answer” (readWhere's phrase, with the piece named once). */
const readFact = (s: Sighting): string => {
  const w = readWhere(s);
  const at = s.readAt != null ? pieceWord(s.readAt) : "the last word-piece";
  return w.k >= 1 ? `read at ${at}, the ${ordinal(w.k)} word-piece of its own answer` : `read at ${at}${w.phrase}`;
};
/**
 * How long it took, named for what was timed: a streamed sighting says how long this word-piece took to analyse and how far
 * into the request it came (writing, reading again and the word-pieces before it included); an older one, its whole sweep.
 */
const surveyedIn = (s: Sighting): string => {
  const t = s.timing;
  if (t?.analyseMs != null) {
    return `${N_LAYERS} layers analysed in ${ms(t.analyseMs)}${t.totalMs != null ? `, ${ms(t.totalMs)} after it was asked` : ""}`;
  }
  return s.elapsedMs != null ? `${N_LAYERS} layers surveyed in ${ms(s.elapsedMs)}` : `${N_LAYERS} layers surveyed`;
};
/** Drawn by chance (or with a repeat penalty): the server's echo says so; greedy with no penalty, as the study wrote the
 * answers it checked the lens on, needs no line here. */
const drawn = (s: Sighting) => {
  const g = s.generation;
  return g != null && g.source !== "recorded" && (!g.greedy || (g.presencePenalty ?? 0) !== 0) ? g : null;
};
const recallOf = (share: number | null) => (share == null ? null : 100 - Math.round((1 - share) * 100));

/** "“computer”" or "a pictograph of a robot’s face" */
const wordOrPicto = (token: string) => (isPicto(token) ? pictoWords(token) : wp(token));

/* ----------------------------------------------------------------- screen */

export const LayerLeaf: React.FC<{ route: Route }> = ({ route }) => {
  const layerNo = route.name === "layer" ? Math.max(1, Math.min(N_LAYERS, route.layerNo)) : 36;
  const idx = layerNo - 1;
  const sighting = useTheodoliteStore((s) => s.sighting);
  const setSelectedLayer = useTheodoliteStore((s) => s.setSelectedLayer);

  // the instrument follows the leaf being read
  useEffect(() => {
    setSelectedLayer(idx);
  }, [idx, setSelectedLayer]);

  const L = sighting?.layers[idx] ?? null;
  if (!sighting || !L) {
    return (
      <Page route={route} folio="folio ii">
        <Breadcrumb no={layerNo} />
        <p className="status-line" role="status" style={{ padding: "40px 0" }}>
          <Missing what="the leaf appears when a sighting has been taken" />
        </p>
      </Page>
    );
  }
  const c: Ctx = {
    s: sighting,
    L,
    idx,
    no: layerNo,
    live: sighting.source === "live",
    hand: sighting.source === "fixture" && idx === L36_IDX,
    miss: sighting.source === "live" ? "not exposed by the server" : "measured in a live sighting",
  };

  return (
    <Page route={route} folio="folio ii">
      <Breadcrumb no={layerNo} />
      <AskedStrip s={sighting} />
      <Leaf c={c} />
      <BearingApparatus c={c} />
      <div style={{ display: "flex", flexWrap: "wrap", gap: "48px 72px", paddingTop: 40, borderTop: `1px solid ${HAIR}` }}>
        <Thinking c={c} />
        <Split c={c} />
      </div>
      <Ranks c={c} />
      <Poles c={c} />
      <Kinship c={c} />
    </Page>
  );
};

/* ------------------------------------------------------------- the strip */

const Breadcrumb: React.FC<{ no: number }> = ({ no }) => (
  <nav
    aria-label="Breadcrumb"
    style={{ display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: "0 28px", padding: "4px 0" }}
  >
    <div style={{ display: "flex", alignItems: "center", gap: 12, ...SMALL_CAPS, letterSpacing: "0.18em", fontSize: 16 }}>
      <a href={href.instrument()} style={LINK44}>
        instrument
      </a>
      <span aria-hidden="true" style={{ color: "#A07A2E" }}>
        ›
      </span>
      <span aria-current="location" style={{ color: "#2B2117" }}>
        layer {no}
      </span>
    </div>
    <div style={{ display: "flex", flexWrap: "wrap", gap: "0 26px", fontSize: 16, ...NUM }}>
      {no > 1 ? (
        <a href={href.layer(no - 1)} style={LINK44}>
          ↓ earlier · layer {no - 1}
        </a>
      ) : (
        <span style={{ ...LINK44, color: MUTED }}>the first layer</span>
      )}
      {no < N_LAYERS ? (
        <a href={href.layer(no + 1)} style={LINK44}>
          later · layer {no + 1} ↑
        </a>
      ) : (
        <span style={{ ...LINK44, color: MUTED }}>the last layer</span>
      )}
    </div>
  </nav>
);

/** `text` with its last `tail` letters underlined in gilt, when it ends with them: the word read, at the first word-piece. */
const withReadTail = (text: string, tail: string): React.ReactNode =>
  tail !== "" && text.endsWith(tail) ? (
    <>
      {text.slice(0, text.length - tail.length)}
      <span style={READ_MARK}>{tail}</span>
    </>
  ) : (
    text
  );

/**
 * What was asked and what it wrote, with the word-piece read here underlined in gilt and the one chosen here marked in red;
 * the system prompt when there was one; and, when more than one word-piece of the answer was sighted, a step to the
 * word-piece before or after, so the leaf follows one layer along the answer.
 */
const AskedStrip: React.FC<{ s: Sighting }> = ({ s }) => {
  const label: React.CSSProperties = { ...SMALL_CAPS, letterSpacing: "0.16em", fontSize: 16, color: MUTED };
  const k = s.tokenIndex;
  // at the first word-piece the piece read is the prompt's own last: in the begun answer, or the question's end when it was
  // sent bare (as a chat with no begun answer, the template's own closing mark is read, which is not on the page)
  const read = k === 0 && s.readAt ? s.readAt.replace(/^\s+/, "") : "";
  const begun = s.answerBegins.trim();
  const bareQuestion = !begun && !s.chatWrapped;
  const readShown = read !== "" && (begun ? begun.endsWith(read) : bareQuestion && s.question.trim().endsWith(read));
  const pieces = useAnswerPieces(s);
  const set = useShownSet();
  const pickToken = useTheodoliteStore((st) => st.pickToken);
  const went = s.continuation.trim();
  const { lens } = useReadingLens(s);
  const chose = chosenPhrase(s);
  const facts = [
    readFact(s),
    ...(chose ? [chose] : []),
    surveyedIn(s),
    lensLabel(lens),
  ];
  if (s.source === "fixture") facts.push("the study’s recorded sighting, taken through the same lens");
  if (s.isNudged) facts.push("a nudged sighting");
  // stepping along the answer goes through the word-pieces sighted so far, in order
  const sighted = set
    ? Object.keys(set.sightings)
        .map(Number)
        .sort((x, y) => x - y)
    : [];
  const before = [...sighted].reverse().find((i) => i < k);
  const after = sighted.find((i) => i > k);
  const g = drawn(s);
  return (
    <div style={{ padding: "14px 0 16px", borderTop: `1px solid ${HAIR_SOFT}`, borderBottom: `1px solid ${HAIR_SOFT}` }}>
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: "6px 22px" }}>
        <span style={label}>you asked</span>
        <span style={{ fontSize: 21 }}>“{bareQuestion ? withReadTail(s.question.trim(), read) : s.question}”</span>
        {s.system ? (
          <>
            <span style={label}>its system prompt</span>
            <span style={{ fontSize: 17, color: MUTED }} title={s.system}>
              “{firstWords(s.system, 140)}”
            </span>
          </>
        ) : null}
        {begun ? (
          <>
            <span style={label}>its answer began</span>
            <span style={{ fontSize: 21, fontStyle: "italic" }}>“{withReadTail(begun, read)}”</span>
          </>
        ) : null}
        <span style={label}>it went on</span>
        <span style={{ fontSize: 21, fontStyle: "italic" }}>
          {pieces ? (
            pieces.length ? (
              <>
                “…
                <AnswerMarks s={s} pieces={pieces} />”
              </>
            ) : (
              <span style={{ fontSize: 17, fontStyle: "normal", color: MUTED }}>nothing: it was asked to write no word-piece</span>
            )
          ) : went ? (
            `“… ${went}”`
          ) : (
            <Missing what="no continuation recorded" />
          )}
        </span>
        <span style={{ marginLeft: "auto", fontSize: 15, color: MUTED, ...NUM }}>{facts.join(" · ")}</span>
      </div>
      {pieces && pieces.length ? (
        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "4px 18px", paddingTop: 8, fontSize: 15, color: MUTED }}>
          <span>
            {k >= 1 || readShown ? (
              <>
                <span style={{ ...READ_MARK, color: "#2B2117" }}>gilt</span>: the word-piece it had just read
                {k === 0 ? ", the last of what it was given" : ""}
              </>
            ) : begun ? (
              `it had just read ${lastWord(s)}, the last word-piece of its begun answer`
            ) : s.chatWrapped ? (
              "it had just read the end of the chat template, which the page does not show"
            ) : (
              `it had just read ${lastWord(s)}, the last word-piece of your question`
            )}{" "}
            · <span style={{ ...CHOSEN_MARK, color: "#2B2117" }}>red</span>: the one it chose there · word-piece {k + 1} of {pieces.length}
          </span>
          {sighted.length > 1 ? (
            <span style={{ display: "flex", gap: 6, marginLeft: "auto" }}>
              <button
                type="button"
                className="btn-ghost plain"
                disabled={before === undefined}
                onClick={() => before !== undefined && pickToken(before)}
              >
                ← earlier word-piece
              </button>
              <button type="button" className="btn-ghost plain" disabled={after === undefined} onClick={() => after !== undefined && pickToken(after)}>
                later word-piece →
              </button>
            </span>
          ) : null}
        </div>
      ) : null}
      {pieces && pieces.length && sighted.length > 1 ? <NotAMatchedPair /> : null}
      {g ? (
        <p style={{ margin: "8px 0 0", fontSize: 15, color: MUTED, maxWidth: 980 }}>
          {!g.greedy ? <OneDrawLine generation={g} /> : null}
          {(g.presencePenalty ?? 0) !== 0 ? (
            <>
              {" "}
              <PenaltyLine penalty={g.presencePenalty as number} />
            </>
          ) : null}
        </p>
      ) : null}
    </div>
  );
};

/* ---------------------------------------------------------- the leaf head */

/** The head sentence's reading. Through a lens the readout, not the model, is the subject: its shares describe the lens's
 * guess, uncalibrated (erratum E30). At the last layer the lens is the identity, and the odds are the model's own. */
function thinkingClause(c: Ctx, lensName: string): string {
  const t = top1(c.L);
  if (!t) return c.live ? "the server sent no reading through the lens" : "the lens’s reading was not kept in this sighting";
  const w = wordOrPicto(t.token);
  const p = t.prob;
  if (c.no >= N_LAYERS) {
    // the likeliest and the piece it chose are two roles: by chance, a repeat penalty or a near-tie they part
    const chosen = chosenOf(c.s);
    const other =
      chosen && !isChosenRow(c.s, c.L, 0)
        ? chosen.endMark
          ? `, yet here ${endedWords(chosen.id, chosen.token)}`
          : `, yet it chose ${wordOrPicto(chosen.token)}`
        : "";
    if (p == null) return `the model’s likeliest next word is ${w}${other}`;
    if (p < 0.1) return `the model’s own odds are spread thin: its likeliest word, ${w}, holds less than a tenth of them${other}`;
    if (p < 0.2) return `the model’s own odds are divided: its likeliest word, ${w}, holds ${pct1(p)}${other}`;
    if (p < 0.6) return `the model’s own odds lean toward ${w}, which holds ${pct1(p)} of them${other}`;
    return `the model is all but certain of ${w}, which holds ${pct1(p)} of its odds${other}`;
  }
  const through = `through the ${lensName} lens`;
  if (p == null) return `${through} the strongest word is ${w}`;
  if (p < 0.1) return `${through} no word stands out: the strongest, ${w}, holds less than a tenth of the readout`;
  if (p < 0.2) return `${through} the readout is divided: its strongest word, ${w}, holds ${pct1(p)}`;
  if (p < 0.6) return `${through} the readout leans toward ${w}, which holds ${pct1(p)} of it`;
  return `${through}, ${w} holds ${pct1(p)} of the readout`;
}

function subtitle(c: Ctx): string {
  if (c.hand) return L36_SUBTITLE;
  const t = top1(c.L);
  const at = lastWord(c.s);
  if (!t) return c.live ? `where the server sent no lens reading for the word ${at}` : `where this sighting kept no lens reading for the word ${at}`;
  const w = wordOrPicto(t.token);
  const p = t.prob;
  const verb =
    p == null ? `turns to ${w}` : p < 0.1 ? `murmurs of ${w}, among many voices` : p < 0.2 ? `wavers, with ${w} in front` : p < 0.6 ? `leans toward ${w}` : `all but says ${w}`;
  // at layer 60 the lens is the identity: these are the model's own odds
  return c.no === 60 ? `where, in the model’s own odds, the word ${at} ${verb}` : `where, through the lens, the word ${at} ${verb}`;
}

const Leaf: React.FC<{ c: Ctx }> = ({ c }) => {
  const { L, s, no } = c;
  const { lens } = useReadingLens(s);
  const [a, b, lc] = s.landmarks;
  const kind = L.isGlobal
    ? { kind: "global attention", gloss: "looks back over the whole conversation, not only the last 1,024 word-pieces" }
    : { kind: kindOf(false).kind, gloss: kindOf(false).gloss };
  const recall = recallOf(L.ffnShare);
  // the first listed in the sighting's own ranking, whose words the server reads (ranks 0-4 only)
  const loud = leadOf(L);
  const rw = rankingWords(s.poleRanking);
  // the caught share is of the lens-read state, or of the raw stream when each layer has its own frame (lib/frame.tsx)
  const F = describeFrame(s);
  const caught = F.raw || F.centred ? F.denomWords : "of that reading";
  const aria =
    `The numeral ${no} at the hub of an engraved dial. ` +
    (L.az != null
      ? `On the level circle a red index marks a bearing of ${bearingFigure(L.az)} degrees, ${bearingWords(L.az, a, b, false)}. `
      : "No bearing was recorded at this layer. ") +
    (L.el != null
      ? `On the upright quadrant a red needle marks an elevation of ${minus(String(Math.round(L.el)))} degrees ${
          L.el >= 0 ? "toward" : "away from"
        } the landmark word ${lc}.`
      : "No elevation was recorded at this layer.");
  const capLabel: React.CSSProperties = { ...SMALL_CAPS, letterSpacing: "0.18em", fontSize: 15, color: MUTED };
  const capFig: React.CSSProperties = { fontSize: 32, lineHeight: 1, color: "#8A6A2C", ...NUM };
  const capNote: React.CSSProperties = { fontSize: 16, fontStyle: "italic", color: MUTED };

  return (
    <section
      aria-labelledby="leaf-title"
      style={{ display: "flex", flexWrap: "wrap", gap: "32px 56px", padding: "40px 0 48px", alignItems: "center" }}
    >
      <figure style={{ flex: "999 1 600px", minWidth: 0, margin: 0, display: "flex", flexDirection: "column", gap: 14 }}>
        <Dial layerNo={no} az={L.az} el={L.el} landmarks={s.landmarks} ariaLabel={aria} />
        <figcaption style={{ display: "flex", flexDirection: "column", gap: 10, alignItems: "center" }}>
          <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", alignItems: "baseline", gap: "6px 44px" }}>
            <span style={{ display: "flex", alignItems: "baseline", gap: 10, flexWrap: "wrap" }}>
              <span style={capLabel}>bearing</span>
              {L.az != null ? (
                <>
                  <span style={capFig}>{bearingFigure(L.az)}°</span>
                  <span style={capNote}>{bearingWords(L.az, a, b)}</span>
                </>
              ) : (
                <M c={c} />
              )}
            </span>
            <span style={{ display: "flex", alignItems: "baseline", gap: 10, flexWrap: "wrap" }}>
              <span style={capLabel}>elevation</span>
              {L.el != null ? (
                <>
                  <span style={capFig}>{minus(String(Math.round(L.el)))}°</span>
                  <span style={capNote}>{elevationWords(L.el, lc)}</span>
                </>
              ) : (
                <M c={c} />
              )}
            </span>
          </div>
          <Rubric style={{ fontSize: 16, maxWidth: 660, textAlign: "center" }}>
            {F.raw ? (
              <>
                Bearing and elevation place this layer’s stream among three landmark words, “{a}”, “{b}” and “{lc}”, each carried back
                to this layer through the {lens.name} lens.
              </>
            ) : (
              <>
                Bearing and elevation place this layer’s stream, as the {lens.name} lens reads it, among three landmark words,{" "}
                “{a}”, “{b}” and “{lc}”.
              </>
            )}{" "}
            Read them loosely: here the three landmarks catch only{" "}
            {L.projFrac != null ? `${(Math.round(L.projFrac * 1000) / 10).toFixed(1)}%` : <M c={c} />} {caught}
            {F.lowdin ? ", " : "; in the study, merely reordering them swung the bearing by 58–99° (REPORT §3.4); "}and no bearing was
            checked against what the model does, at any layer.
          </Rubric>
        </figcaption>
      </figure>

      <div style={{ flex: "1 1 400px", minWidth: 0, display: "flex", flexDirection: "column", gap: 18 }}>
        <p style={{ margin: 0, fontSize: 14, fontWeight: 500, ...SMALL_CAPS, letterSpacing: "0.24em", color: "#8F2B1E" }}>
          the leaf of one layer
        </p>
        <h1 id="leaf-title" style={{ margin: 0, fontSize: 50, lineHeight: 1.05, fontWeight: 400, letterSpacing: "-0.005em" }}>
          Layer {cardinal(no)}
        </h1>
        <p style={{ margin: 0, fontSize: 27, lineHeight: 1.3, fontStyle: "italic" }}>{subtitle(c)}</p>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 2,
            padding: "14px 0",
            borderTop: `1px solid ${HAIR}`,
            borderBottom: `1px solid ${HAIR}`,
          }}
        >
          <span style={{ ...SMALL_CAPS, letterSpacing: "0.2em", fontSize: 15, color: MUTED, fontVariantNumeric: "lining-nums" }}>
            layer {no} of sixty · {roman(no).toLowerCase()}
          </span>
          <span style={{ fontSize: 20 }}>
            {L.isGlobal ? (
              <>
                <span aria-hidden="true" style={{ color: "#A07A2E" }}>
                  ✦
                </span>{" "}
              </>
            ) : null}
            {kind.kind}
          </span>
          <span style={{ fontSize: 16, fontStyle: "italic", color: MUTED }}>{kind.gloss}</span>
        </div>
        <p style={{ margin: 0 }}>
          {wayUp(no)}, {thinkingClause(c, lens.name)}.{" "}
          {recall != null ? (
            <>
              Of what changed here, as the {lens.name} lens weighs the two writes, {recall}% came from memory and {100 - recall}% was
              gathered from the words before.
            </>
          ) : (
            <>
              How much of what changed here came from memory, and how much was gathered from the words before: <M c={c} />.
            </>
          )}{" "}
          {loud ? (
            <>
              {capital(cardinal(L.poles.length))} neurons {rw.did}; {rw.lead}, neuron {loud.neuron}, fired on its{" "}
              {loud.face > 0 ? "positive" : "negative"} face.
            </>
          ) : (
            <>
              The neurons that fired here: <M c={c} />.
            </>
          )}
        </p>
        {/* Where the lens's words are weakest (below layer 31, and its late lean at 51-59: erratum E49), the red line stands
            in the head beside the dial, not only in section i. It speaks of the words; the dial's own rubric covers bearings. */}
        {no <= 30 || (no >= 51 && no <= 59) ? <LensCaution sighting={s} no={no} style={{ margin: 0, maxWidth: "none" }} /> : null}
        <Rubric style={{ maxWidth: "none" }}>
          This is the leaf for one layer. Read it from top to bottom: what the lens reads here, how the layer’s writing splits between
          gathering and remembering, how far some telling words were from being said, which neurons fired, and which other layers think
          alike. The links above step to the neighbouring layers.
        </Rubric>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "12px 18px", alignItems: "center" }}>
          {loud ? (
            <a
              href={href.neuron(no, loud.neuron, loud.face)}
              style={{
                display: "inline-flex",
                alignItems: "center",
                fontSize: 19,
                ...SMALL_CAPS,
                letterSpacing: "0.16em",
                background: "#8F2B1E",
                color: "#F6F0E3",
                textDecoration: "none",
                borderRadius: 2,
                padding: "0 26px",
                minHeight: 48,
              }}
            >
              {s.poleRanking === "write" ? "read the biggest writer" : "read the loudest neuron"}
            </a>
          ) : null}
          <a
            href={href.instrument()}
            style={{
              display: "inline-flex",
              alignItems: "center",
              fontSize: 17,
              color: "#2B2117",
              textDecoration: "none",
              border: "1px solid #C9BEAC",
              borderRadius: 2,
              padding: "0 18px",
              minHeight: 48,
            }}
          >
            back to the instrument
          </a>
        </div>
      </div>
    </section>
  );
};

/* ------------------------------------------------- apparatus: the bearing */

function buildRaw(c: Ctx): string {
  const { s, L, idx } = c;
  const fixture = !c.live;
  const lensLabels: Record<string, [string, number | null][]> = {};
  const eyeLabels: Record<string, [string, number | null][]> = {};
  L.poles.forEach((p) => {
    if (p.lens) lensLabels[`${idx}:${p.rank}`] = p.lens.map((t) => [t.token, t.logit]);
    if (p.eye) eyeLabels[`${idx}:${p.rank}`] = p.eye.map((t) => [t.token, t.logit]);
  });
  const lensRank: Record<string, number | null> = {};
  const eyeRank: Record<string, number | null> = {};
  for (const [w, r] of Object.entries(L.ranks)) {
    lensRank[w] = r.lens;
    eyeRank[w] = r.eye;
  }
  const row = s.kinship && s.kinship.length === N_LAYERS * N_LAYERS ? s.kinship.slice(idx * N_LAYERS, (idx + 1) * N_LAYERS) : null;
  const notRecorded: string[] = [];
  if (L.streamStrength == null) notRecorded.push("residual_norm");
  if (L.attnNorm == null) notRecorded.push("attn_norm");
  if (L.ffnNorm == null) notRecorded.push("ffn_norm");
  if (L.rescaleNorm == null) notRecorded.push("rescale_norm");
  if (!L.lensTop || L.lensTop.some((t) => t.logit == null)) notRecorded.push("jpp_top_logits");
  if (L.eyeTop == null) notRecorded.push("naive_top_tokens");

  const raw: Record<string, unknown> = fixture
    ? {
        source: `experiments/results/E6_tool_audit_02_sweep_demo.json (prepend_bos=True), index ${idx}; layer_scalar: E6_tool_audit_01_basics.json; is_global_attention: backend/theodolite_math.py`,
        layer: L.no,
        index: idx,
        is_global_attention: L.isGlobal,
        token_index: s.tokenIndex,
        read_at: s.readAt,
        target_pos: s.readPos,
        n_tokens: s.nTokens,
        first_ids: s.firstIds,
        generated: s.continuation,
        generation: s.generation,
        jpp_top5: L.lensTop ? L.lensTop.map((t) => [t.token, t.prob]) : null,
        m1_corrected_ffn_dominance: L.ffnShare,
        m1_tool_ffn_dominance: L.ffnShareBeforeFix,
        ...(c.hand ? { m1_cos_tool_vs_true_ffn: L36_COS_FFN, m1_cos_tool_vs_true_attn: L36_COS_ATTN } : {}),
        layer_scalar: L.layerScalar,
        survey_labels: s.landmarks,
        azimuth_deg: L.az,
        elevation_deg: L.el,
        xyz: L.xyz,
        ...(c.hand ? { proj_norm: L36_PROJ_NORM } : {}),
        [describeFrame(c.s).exportKey]: L.projFrac,
        jpp_tracked_rank: lensRank,
        naive_tracked_rank: eyeRank,
        jpp_tracked_logit: Object.fromEntries(Object.entries(L.ranks).map(([w, v]) => [w, v.lensLogit ?? null])),
        naive_tracked_logit: Object.fromEntries(Object.entries(L.ranks).map(([w, v]) => [w, v.eyeLogit ?? null])),
        pole_neurons: L.poles.map((p) => p.neuron),
        pole_acts: L.poles.map((p) => p.act),
        pole_labels_jpp: lensLabels,
        pole_labels_naive: eyeLabels,
        cka_row: row,
        not_recorded: notRecorded,
      }
    : {
        source: "a live sighting from the running server (/ws/theodolite: layer_plumes, the binary frame and poles_labels)",
        layer: L.no,
        index: idx,
        is_global_attention: L.isGlobal,
        request_id: s.requestId,
        token_index: s.tokenIndex,
        read_pos: s.readPos,
        read_token: s.readAt,
        read_token_id: s.readAtId,
        prompt_len: s.promptLen,
        kinship_n_states: s.nTokens,
        chosen: s.chosen,
        generation: s.generation,
        recapture_check: s.recaptureCheck,
        rank_lookahead: s.rankLookahead,
        centre_kind: s.centreKind,
        timing: s.timing,
        generated: s.continuation,
        is_steered: s.isNudged,
        jpp_top_tokens: L.lensTop ? L.lensTop.map((t) => [t.token, t.prob, t.logit]) : null,
        jpp_top_ids: L.lensTopIds,
        naive_top_token: L.eyeTop,
        naive_top_id: L.eyeTopId,
        rank_from_next: L.rankFromNext,
        ffn_dominance: L.ffnShare,
        attn_norm: L.attnNorm,
        ffn_norm: L.ffnNorm,
        rescale_norm: L.rescaleNorm,
        layer_scalar: L.layerScalar,
        residual_norm: L.streamStrength,
        survey_labels: s.landmarks,
        azimuth_deg: L.az,
        elevation_deg: L.el,
        xyz: L.xyz,
        tracked_ranks: lensRank,
        pole_neurons: L.poles.map((p) => p.neuron),
        pole_acts: L.poles.map((p) => p.act),
        pole_labels_jpp: lensLabels,
        pole_labels_naive: eyeLabels,
        cka_row: row,
        not_recorded: notRecorded,
      };
  return JSON.stringify(raw, null, 2)
    .replace(/\[([^[\]{}]*)\]/g, (_m, inner: string) => `[${inner.replace(/\n\s*/g, " ").trim()}]`)
    .replace(/[\p{Extended_Pictographic}]/gu, (ch) =>
      Array.from({ length: ch.length }, (_, i) => `\\u${ch.charCodeAt(i).toString(16).padStart(4, "0")}`).join(""),
    );
}

const BearingApparatus: React.FC<{ c: Ctx }> = ({ c }) => {
  const F = describeFrame(c.s);
  const { L, s, idx, no } = c;
  const [rawOpen, setRawOpen] = useState(false);
  const json = rawOpen ? buildRaw(c) : "";
  const pn = L.xyz ? Math.hypot(L.xyz[0], L.xyz[1], L.xyz[2]) : null;
  const set = useShownSet();
  const nSighted = set ? Object.keys(set.sightings).length : 0;
  return (
    <Apparatus
      title={
        <>
          layer {no} <span style={{ letterSpacing: "0.06em" }}>(index {idx})</span> · the bearing
        </>
      }
      label="Apparatus for this layer and its bearing"
      style={{ padding: "12px 0 34px" }}
    >
      <dl className="dl num">
        <dt>counting</dt>
        <dd>
          Layer {no} is index {idx} wherever the repository, the server, the MCP tools and LQL count from 0. The report’s “L{idx}” is this
          layer.
        </dd>
        <dt>attention</dt>
        <dd>
          {L.isGlobal ? (
            <>
              global: every sixth layer (6, 12, … 60) attends over the whole conversation, the other fifty over the last 1,024 word-pieces
              (Gemma Team 2026) [1].
            </>
          ) : (
            <>
              local: this layer, like fifty of the sixty, attends over the last 1,024 word-pieces; every sixth layer (6, 12, … 60) attends
              over the whole conversation (Gemma Team 2026) [1].
            </>
          )}
        </dd>
        <dt>survey frame</dt>
        <dd>
          {F.frameFormula}; p = (x, y, z) = ⟨{F.raw ? "h" : "h̃"}
          {F.centred ? " − c" : ""}, e<sub>k</sub>⟩, where {F.state}
          {F.raw ? " is the raw stream, read in this layer’s own frame" : " is the stream seen through the lens"}; {F.notes.join("; ")}.
        </dd>
        <dt>frame values</dt>
        <dd>
          (x, y, z) = {L.xyz ? `(${L.xyz.map(fmt3).join(", ")})` : <M c={c} />}; ‖p‖ = {pn != null ? fmt3(pn) : <M c={c} />}; θ = atan2(y, x)
          = {L.az != null ? `${minus(L.az.toFixed(1))}°` : <M c={c} />}, φ = asin(z / ‖p‖) ={" "}
          {L.el != null ? `${minus(L.el.toFixed(1))}°` : <M c={c} />}. The frame catches{" "}
          {L.projFrac != null ? `${(L.projFrac * 100).toFixed(2)}%` : <M c={c} />} of {F.denom} (<Mono>{F.exportKey}</Mono>)
          {F.studyComparable ? "; the study found 2.4–3.9% by stretch of layers (V7)" : ""}.
        </dd>
        <dt>caveat</dt>
        <dd>
          {F.lowdin
            ? "A Löwdin frame [33] squares the landmarks all at once: their order no longer changes the three axes, but it still says which sets the bearing’s 0°, which its 90° and which points up, so reordering them relabels the bearing and elevation (swap the first two and θ reads 90° − θ). "
            : "A Gram–Schmidt frame depends on the order of its landmarks: reordering them swings the bearing by 58–99° (REPORT §3.4, bug 3); a Löwdin frame [33], the interface’s default, keeps the same three axes in any order and only relabels them. "}
          The captured fraction is always shown.{" "}
          <a href={href.landmarks()}>change the landmarks, or lock the frame to a neuron</a>
        </dd>
        <dt>stream strength</dt>
        <dd>
          ‖h‖ for this sighting: {L.streamStrength != null ? L.streamStrength.toFixed(1) : <M c={c} />}.
          {idx === L36_IDX ? <> On the study’s reference prompts the median here is 253 to 269 (E6, V7).</> : null}
        </dd>
      </dl>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "8px 12px", paddingTop: 4 }}>
        <button type="button" className="btn-ghost" aria-expanded={rawOpen} onClick={() => setRawOpen((v) => !v)}>
          {rawOpen ? "hide the raw readout" : "show the raw readout"}
        </button>
        <button type="button" className="btn-ghost" onClick={() => exportSighting(s)}>
          {s.source === "live" && nSighted > 1 ? "export this word-piece’s sighting (.json)" : "export the sighting (.json)"}
        </button>
        {set && nSighted > 1 ? (
          <button type="button" className="btn-ghost" onClick={() => exportAnswer(set)}>
            export every word-piece’s sighting, all {nSighted} (.json)
          </button>
        ) : null}
      </div>
      {rawOpen ? <pre className="raw">{json}</pre> : null}
    </Apparatus>
  );
};

/* ------------------------------------------------- i · through the lens */

const Thinking: React.FC<{ c: Ctx }> = ({ c }) => {
  const { L, s, no, live } = c;
  const top = L.lensTop;
  const at60 = no >= N_LAYERS;
  // each row keeps its rank in the readout, so the piece it chose is found by id (isChosenRow), pictographs or not
  const ranked = top ? top.map((t, rank) => ({ t, rank })) : [];
  const shown = ranked.filter((r) => !isPicto(r.t.token));
  const pictos = ranked.filter((r) => isPicto(r.t.token));
  const chosen = chosenOf(s);
  const chosenListed = !!chosen && ranked.some((r) => isChosenRow(s, L, r.rank));
  // at layer 60 two roles: its likeliest, and the word-piece it chose (they part by chance, a repeat penalty or a near-tie)
  const roleOf = (rank: number): string | null => {
    if (!at60 || !chosen) return null;
    if (isChosenRow(s, L, rank)) return rank === 0 ? "its likeliest, and the word it chose" : "the word it chose";
    return rank === 0 ? "its likeliest" : null;
  };
  const said = saidHere(s);
  const g = s.generation;
  const sampled = !!g && g.source !== "recorded" && !g.greedy;
  const penalised = !!g && g.source !== "recorded" && (g.presencePenalty ?? 0) !== 0;
  const kept = s.layers.filter((l) => l.lensTop && l.lensTop.length).map((l) => l.no);
  const nearest = (() => {
    if (top || kept.length === 0) return [] as number[];
    const below = kept.filter((n) => n < no).pop();
    const above = kept.find((n) => n > no);
    return [below, above].filter((n): n is number => n !== undefined);
  })();
  const logitsKnown = !!top && top.length > 0 && top.every((t) => t.logit != null);
  const { lens } = useReadingLens(s);
  const roleStyle = (role: string): React.CSSProperties => ({
    ...SMALL_CAPS,
    letterSpacing: "0.14em",
    fontSize: 15,
    color: role === "its likeliest" ? MUTED : "#8F2B1E",
  });

  return (
    <section aria-labelledby="think-head" style={{ flex: "999 1 560px", minWidth: 0, display: "flex", flexDirection: "column", gap: 12 }}>
      <div className="row-between" style={{ gap: "6px 24px" }}>
        <h2 id="think-head" className="h-sec">
          i · {at60 ? "its own odds" : "through the lens"}
        </h2>
        {top && top[0]?.prob != null ? (
          <span style={{ fontSize: 16, fontStyle: "italic", color: MUTED }}>{certainty(top[0].prob, at60)}</span>
        ) : null}
      </div>
      {top ? (
        <div style={{ display: "flex", flexDirection: "column" }}>
          {shown.map(({ t, rank }) => {
            const role = roleOf(rank);
            return (
              <div
                key={`${t.token}-${rank}`}
                style={{
                  display: "grid",
                  gridTemplateColumns: "minmax(0, 1fr) 72px",
                  gap: "2px 14px",
                  alignItems: "baseline",
                  padding: "8px 0 10px",
                  borderBottom: `1px solid ${HAIR_FAINT}`,
                }}
              >
                <div style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: "2px 14px", minWidth: 0 }}>
                  <Wp token={t.token} style={{ fontSize: 28, lineHeight: 1.2, ...(role && role !== "its likeliest" ? CHOSEN_MARK : {}) }} />
                  {role ? <span style={roleStyle(role)}>{role}</span> : null}
                  <span style={{ fontSize: 15, fontStyle: "italic", color: MUTED }}>{gloss(t.token)}</span>
                </div>
                <span style={{ textAlign: "right", fontSize: 17, ...NUM }}>{t.prob != null ? pct1(t.prob) : <M c={c} />}</span>
                <div aria-hidden="true" className="bar-track" style={{ gridColumn: "1 / -1" }}>
                  <div className="bar-fill" style={{ width: `${t.prob != null ? Math.round(t.prob * 1000) / 10 : 0}%` }} />
                </div>
              </div>
            );
          })}
          {at60 && chosen && !chosenListed ? (
            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                alignItems: "baseline",
                gap: "2px 14px",
                padding: "8px 0 10px",
                borderBottom: `1px solid ${HAIR_FAINT}`,
              }}
            >
              {chosen.endMark ? (
                <span style={{ fontSize: 20, ...CHOSEN_MARK }} title={JSON.stringify(chosen.token)}>
                  {endMarkWords(chosen.id, chosen.token)}
                </span>
              ) : (
                <Wp token={chosen.token} style={{ fontSize: 28, lineHeight: 1.2, ...CHOSEN_MARK }} />
              )}
              <span style={roleStyle("the word it chose")}>the word it chose</span>
              <span style={{ fontSize: 16, color: MUTED, ...NUM }}>
                not among the five shown here:{" "}
                {chosen.modelRank != null ? `its ${ordinal(chosen.modelRank)} likeliest` : "its place in its odds was not sent"}
                {chosen.modelProb != null ? `, ${pct(chosen.modelProb)} of its own odds` : ""}
                {chosen.source === "given" ? ", in the re-reading" : ", as it wrote"}
                {sampled ? "; the sampler drew it" : ""}.
              </span>
            </div>
          ) : null}
        </div>
      ) : (
        <div className="stack" style={{ gap: 6, padding: "8px 0 10px", borderBottom: `1px solid ${HAIR_FAINT}` }}>
          <span>
            {live ? (
              <M c={c} />
            ) : (
              <Missing what={`recorded only at layers ${listAnd(kept.map(String))}`} />
            )}
          </span>
          {nearest.length ? (
            <span style={{ fontSize: 16, color: MUTED }}>
              the nearest readings:{" "}
              {nearest.map((n, i) => (
                <React.Fragment key={n}>
                  {i ? " · " : null}
                  <a href={href.layer(n)} style={{ ...LINK44, display: "inline-flex" }}>
                    layer {n}
                  </a>
                </React.Fragment>
              ))}
            </span>
          ) : null}
        </div>
      )}
      {pictos.map(({ t, rank }) => (
        <p key={`picto-${rank}`} style={{ margin: 0, fontSize: 15, color: MUTED }}>
          {capital(`a ${ORDW[rank] ?? `${rank + 1}th`}`)} word-piece, {pictoWords(t.token)}, held {t.prob != null ? pct1(t.prob) : <M c={c} />};
          it is described here rather than shown{roleOf(rank) ? `. It is ${roleOf(rank)}` : ""}.
        </p>
      ))}
      <Rubric style={{ margin: "2px 0 0" }}>
        {at60 ? (
          <>
            At the last layer no lens stands between: these are the model’s own odds for its next word-piece
            {sampled && penalised
              ? ", before the repeat penalty and the sampler"
              : sampled
                ? ", before the sampler drew one by chance"
                : penalised
                  ? ", before the repeat penalty"
                  : ""}
            .{said ? ` ${said}` : ""}{" "}
          </>
        ) : readsInsideAnswer(s) ? (
          // inside its own answer (a begun one in a chat turn too) the study measured the lens apart from prose: the red line
          // below (LensCaution) gives that
          <>
            These are the words read from this layer’s stream through the {lens.name} lens: its guess at what the model would say if it
            stopped here. Take them as glimpses, not a forecast.{said ? ` ${said}` : ""}{" "}
          </>
        ) : (
          <>
            These are the words read from this layer’s stream through the {lens.name} lens: its guess at what the model would say if it
            stopped here. Take them as glimpses, not a forecast: on ordinary prose, the lens’s first choice{" "}
            {no <= 50 ? "at this depth matched" : "matched"} the model’s real next word at most about 5% of the time
            {no <= 50 ? "" : " up to layer 50"} (E3).{said ? ` ${said}` : ""}{" "}
          </>
        )}
        <a href={href.lexicon("lens")}>What is a lens?</a>
      </Rubric>
      <LensCaution sighting={s} no={no} style={{ margin: 0 }} />
      {/* the end of the chat template, with no begun answer: the study never read the lens there */}
      {!at60 && readWhere(s).role === "template_end" ? <TemplateEndCaution /> : null}
      <Apparatus title="the readout" label="Apparatus for the readout">
        <dl className="dl num">
          <dt>readout</dt>
          <dd>
            logits(h, ℓ) = unembed(h · J<sub>ℓ</sub>
            <sup>⊤</sup>), unembed(u) = 30 · tanh(W<sub>U</sub> · RMSNorm(u) / 30). The Jacobian lens and J-space are Gurnee et al. 2026
            [2]; J here is the J++ lens (Ayonrinde &amp; Lindsey 2026) [4]. The naked eye is the logit lens, unembed(h) (nostalgebraist 2020)
            [8].
          </dd>
          <dt>the lens</dt>
          <dd>
            J++ takes its Jacobians with the R-lens LRP rules (Blank, Bhatia &amp; Nanda 2026) [5]. It was fitted on the base model, on 64
            WikiText-103 passages [7], and is applied here to the instruct 4-bit checkpoint (REPORT §0; §3.4, bug 9). The Jacobian lens is
            on disk; the server does not load it.
          </dd>
          <LensRecord sighting={s} />
          <dt>raw values</dt>
          <dd>
            {top && top.length ? (
              <>
                probabilities after the softcap:{" "}
                {(() => {
                  const parts = top.map((t) => {
                    const v = t.prob != null ? ` ${t.prob.toFixed(3)}` : "";
                    return isPicto(t.token) ? `${pictoRaw(t.token)},${v}` : `${wp(t.token)}${v}`;
                  });
                  return parts.length > 2
                    ? `${parts.slice(0, -1).join(", ")}, and ${parts[parts.length - 1]}`
                    : parts.join(" and ");
                })()}
                .{" "}
              </>
            ) : (
              <>
                probabilities after the softcap: <M c={c} />.{" "}
              </>
            )}
            {!logitsKnown && L.eyeTop == null ? (
              <>
                Their logits, and the naked eye’s top word-pieces: <M c={c} />.
              </>
            ) : (
              <>
                Their logits:{" "}
                {logitsKnown && top ? top.map((t) => `${isPicto(t.token) ? uPlus(t.token) : wp(t.token)} ${two(t.logit as number)}`).join(", ") : <M c={c} />}
                . The naked eye’s top word-piece: {L.eyeTop != null ? <Wp token={L.eyeTop} /> : <M c={c} />}.
              </>
            )}
          </dd>
          <dt>fidelity</dt>
          <dd>
            On prose, the lens’s first choice was the real next word at most about 5% of the time up to layer 50 (5.1–5.2% at layers 46–47 in
            E3’s per-layer data; the report’s “3%” understates it), and 28% at layer 59. On assistant turns at layers 31–40 it leads the
            naked eye by 3.1 nats of KL, against 9.1 on prose. It orders conditions well from about layer 21, but compresses the scale 8–14
            times (REPORT §3.1). A lens refitted on the instruct model matched the model’s own replies at layer 51 45% of the time, against
            25% for J++ (R1).
          </dd>
          <dt>caveat</dt>
          <dd>
            The published J++ readout also drops punctuation and whitespace tokens [4]; this instrument applies the lens matrix alone. Below
            layer 31 no averaged-Jacobian lens the study fitted was stable, and none improved next-word agreement below layer 41 (R1). Read a
            single sighting only beside a matched control (§3.5).
          </dd>
        </dl>
      </Apparatus>
    </section>
  );
};

/* ------------------------------------------------ ii · gathering or remembering */

const Split: React.FC<{ c: Ctx }> = ({ c }) => {
  const { L, s, idx, no, live } = c;
  const recall = recallOf(L.ffnShare);
  const gather = recall == null ? null : 100 - recall;
  const neighbours = [idx - 1, idx + 1]
    .filter((j) => j >= 0 && j < N_LAYERS)
    .map((j) => ({ no: j + 1, r: recallOf(s.layers[j]?.ffnShare ?? null) }))
    .filter((n): n is { no: number; r: number } => n.r != null);
  const neighbourLine = (() => {
    if (!neighbours.length) return null;
    if (neighbours.length === 1) return `Its neighbour, layer ${neighbours[0].no}, recalled ${neighbours[0].r}%.`;
    const alike = recall != null && neighbours.every((n) => Math.abs(n.r - recall) <= 5);
    return `Its neighbours ${alike ? "are much alike" : "differ"}: layer ${neighbours[0].no} recalled ${neighbours[0].r}%, layer ${
      neighbours[1].no
    } ${neighbours[1].r}%.`;
  })();
  const { lens } = useReadingLens(s);
  // the shares are of the two writes mapped through the lens (theodolite_math: Δ·Jᵀ), so the lens is named with them
  const weighed = `as the ${lens.name} lens weighs the two writes`;
  const leaned =
    recall == null
      ? ""
      : L.isGlobal
        ? recall > 50
          ? ` This layer may look back over the whole conversation, yet, ${weighed}, it leaned on memory.`
          : recall < 50
            ? ` This layer may look back over the whole conversation, and, ${weighed}, it leaned on what it gathered.`
            : ` This layer may look back over the whole conversation; ${weighed}, it drew on both alike.`
        : recall > 50
          ? ` ${capital(weighed)}, this layer leaned on memory.`
          : recall < 50
            ? ` ${capital(weighed)}, this layer leaned on what it gathered from the words before.`
            : ` ${capital(weighed)}, this layer drew on both alike.`;
  const sl = L.layerScalar;
  const after = s.layers.slice(idx + 1).map((l) => l.layerScalar);
  const leak = idx < N_LAYERS - 1 && after.every((v) => v != null) ? after.reduce<number>((p, v) => p * (v as number), 1) : null;
  const leakSci = leak != null ? sci(leak) : null;
  const big: React.CSSProperties = { fontSize: 44, lineHeight: 1, ...NUM };

  return (
    <section aria-labelledby="split-head" style={{ flex: "1 1 420px", minWidth: 0, display: "flex", flexDirection: "column", gap: 14 }}>
      <div className="row-between" style={{ gap: "6px 24px" }}>
        <h2 id="split-head" className="h-sec">
          ii · gathering or remembering
        </h2>
        <span style={{ fontSize: 15, color: MUTED }}>weighed through the {lens.name} lens</span>
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 16, paddingTop: 6 }}>
        <span style={{ ...big, color: "#2E4A7D" }}>{gather != null ? `${gather}%` : <M c={c} />}</span>
        <span style={{ ...big, color: "#8A6A2C" }}>{recall != null ? `${recall}%` : <M c={c} />}</span>
      </div>
      <div
        role="img"
        aria-label={
          gather != null
            ? `${gather} percent gathered from earlier words, ${100 - gather} percent recalled from memory`
            : "The split between gathering and remembering was not recorded at this layer."
        }
        style={{ display: "flex", height: 14, borderRadius: 1, overflow: "hidden", background: gather == null ? HAIR_FAINT : undefined }}
      >
        {gather != null ? (
          <>
            <div style={{ width: `${gather}%`, background: "#2E4A7D" }} />
            <div style={{ width: `${100 - gather}%`, background: "#A07A2E" }} />
          </>
        ) : null}
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 16, fontSize: 16 }}>
        <span>
          gathered from the words before <span style={{ color: MUTED, fontStyle: "italic" }}>(attention)</span>
        </span>
        <span style={{ textAlign: "right" }}>
          recalled from memory <span style={{ color: MUTED, fontStyle: "italic" }}>(feed-forward)</span>
        </span>
      </div>
      {neighbourLine ? (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 6,
            paddingTop: 12,
            borderTop: `1px solid ${HAIR_FAINT}`,
            fontSize: 16,
            color: MUTED,
          }}
        >
          <span>{neighbourLine}</span>
        </div>
      ) : null}
      <Rubric style={{ maxWidth: "none" }}>
        Every layer adds two things to the stream: what it gathers from the words before, and what it recalls from its own memory.
        {leaned}
      </Rubric>
      <Apparatus title="the split" label="Apparatus for gathering and remembering">
        <dl className="dl tight num">
          <dt>split</dt>
          <dd>
            Δ<sub>attn</sub> = s·(mid − pre)·J<sup>⊤</sup>, Δ<sub>ffn</sub> = (post − s·mid)·J<sup>⊤</sup>, and the rescale (s − 1)·pre·J
            <sup>⊤</sup> kept apart; the three add exactly to (post − pre)·J<sup>⊤</sup>. Remembering = ‖Δ<sub>ffn</sub>‖ / (‖Δ
            <sub>attn</sub>‖ + ‖Δ<sub>ffn</sub>‖). Attention and feed-forward as additive writes to the stream: Elhage et al. 2021 [14].
          </dd>
          <dt>this layer</dt>
          <dd>
            layer scalar s = {sl != null ? sl.toFixed(3) : <M c={c} />}
            {live ? "" : " (E6)"}. Remembering {L.ffnShare != null ? `${(L.ffnShare * 100).toFixed(1)}%` : <M c={c} />}
            {live ? (
              <>
                , as the server reports it with the V7 fix (<Mono>ffn_dominance</Mono>; REPORT §3.4, bug 2).
              </>
            ) : (
              <>
                {" "}after the V7 fix <Mono>m1_corrected_ffn_dominance</Mono>
                {L.ffnShareBeforeFix != null ? (
                  <>
                    ; {(L.ffnShareBeforeFix * 100).toFixed(1)}% as the instrument read it before <Mono>m1_tool_ffn_dominance</Mono>
                  </>
                ) : null}{" "}
                (REPORT §3.4, bug 2).
              </>
            )}
          </dd>
          <dt>before the fix</dt>
          <dd>
            {c.hand ? (
              <>
                The old remembering vector, post − mid, folded the rescale in. Here it stood almost at right angles to the true feed-forward
                write: cos {L36_COS_FFN.toFixed(2)} in this sighting (E6), −0.13 [−0.19, −0.07] over twelve study prompts (E3). The gathering
                vector pointed true (cos {L36_COS_ATTN}) but was 1/s ≈ {sl != null ? (1 / sl).toFixed(2) : "1.48"} times too long.
              </>
            ) : (
              <>
                The old remembering vector, post − mid, folded the rescale in{idx === L36_IDX ? <>: at this layer it stood almost at right angles to the true feed-forward write, cos −0.13 [−0.19, −0.07] over twelve study prompts (E3)</> : null}. Across the
                model the old vector met the true feed-forward write at a median cos of 0.49, below zero in 6–14 layers, and the remembering
                share flipped in 12 of 60 layers (REPORT §3.4, bug 2). The old gathering vector was 1/s ≈{" "}
                {sl != null ? (1 / sl).toFixed(2) : <M c={c} />} times too long here.
              </>
            )}
          </dd>
          {idx < N_LAYERS - 1 ? (
            <>
              <dt>a leaky stream</dt>
              <dd>
                the product of the layer scalars of layers {no + 1}
                {no + 1 < N_LAYERS ? `–${N_LAYERS}` : ""} is{" "}
                {leakSci ? (
                  leak != null && leak >= 0.01 ? (
                    leak.toFixed(2)
                  ) : (
                    <>
                      {leakSci.mant} × 10<sup>{minus(String(leakSci.exp))}</sup>
                    </>
                  )
                ) : (
                  <M c={c} />
                )}
                {leak != null && leak < 0.1 ? ": what this layer writes is mostly written over, not stored" : ""} (
                {live ? "layer scalars from the server" : "E6"}; REPORT §0, §2.3).
              </dd>
            </>
          ) : null}
          <dt>{L.attnNorm != null || L.ffnNorm != null || L.rescaleNorm != null ? "the norms" : "not recorded"}</dt>
          <dd>
            ‖Δ<sub>attn</sub>‖{L.attnNorm != null ? ` = ${L.attnNorm.toFixed(2)}` : ""}, ‖Δ<sub>ffn</sub>‖
            {L.ffnNorm != null ? ` = ${L.ffnNorm.toFixed(2)}` : ""} and ‖(s − 1)·pre‖{L.rescaleNorm != null ? ` = ${L.rescaleNorm.toFixed(2)}` : ""}{" "}
            for this sighting
            {L.attnNorm == null || L.ffnNorm == null || L.rescaleNorm == null ? (
              <>
                : <M c={c} />
              </>
            ) : null}
            . The patched server returns them with the layer scalar (REPORT §3.4, bug 2; <Mono>V7_module1_fix.patch</Mono>).
          </dd>
        </dl>
      </Apparatus>
    </section>
  );
};

/* --------------------------------------------- iii · how far the landmark words were */

const LV = Math.log10(VOCAB);
const lx = (n: number) => Math.round((Math.log10(Math.max(1, n)) / LV) * 1000) / 10;
const TICKS = [1, 10, 100, 1000, 10000, 100000].map((n, i) => ({ x: lx(n), label: ordinal(n), tx: i === 0 ? "0" : "-50%" }));
const RANK_GRID = "160px minmax(0, 1fr) 128px 128px";

const Ranks: React.FC<{ c: Ctx }> = ({ c }) => {
  const { L, s } = c;
  const [showLens, setShowLens] = useState(true);
  const [showEye, setShowEye] = useState(true);
  const words = Object.keys(L.ranks);
  const lm = s.landmarks.filter((w) => w);
  const ordered = [...words.filter((w) => lm.includes(w)), ...words.filter((w) => !lm.includes(w))];
  const rows = ordered.map((w) => ({ w, ...L.ranks[w], landmark: lm.includes(w) }));
  // a lead landmark only: never present another tracked word as if it were one of the landmarks
  const lead = rows.find((r) => r.landmark) ?? null;
  const untracked = lm.filter((w) => !words.includes(w));
  const others = rows.filter((r) => !r.landmark).map((r) => `“${r.w}”`);
  const both = rows.filter((r): r is typeof r & { lens: number; eye: number } => r.lens != null && r.eye != null);

  const closing = (() => {
    if (!both.length) return null;
    const lr = both.map((r) => ({ ...r, d: Math.log10(r.lens / r.eye) }));
    const abs = lr.map((r) => Math.abs(r.d)).sort((x, y) => x - y);
    const mid = abs.length % 2 ? abs[(abs.length - 1) / 2] : (abs[abs.length / 2 - 1] + abs[abs.length / 2]) / 2;
    const first = mid > 1 ? "Here the two eyes disagree widely." : mid > 0.5 ? "Here the two eyes disagree." : "Here the two eyes roughly agree.";
    const far = lr.reduce((m, r) => (Math.abs(r.d) > Math.abs(m.d) ? r : m), lr[0]);
    const second =
      far.d > 0
        ? `The naked eye ranks “${far.w}” ${ordinal(far.eye)}; the lens, ${ordinal(far.lens)}.`
        : `The lens ranks “${far.w}” ${ordinal(far.lens)}; the naked eye, ${ordinal(far.eye)}.`;
    const nearer = lr.filter((r) => r.lens < r.eye).map((r) => `“${r.w}”`);
    const third =
      nearer.length === 0
        ? "For none of them does the lens see nearer than the naked eye."
        : nearer.length === lr.length
          ? "For every one of them the lens sees nearer than the naked eye."
          : `Only for ${listAnd(nearer)} does the lens see nearer than the naked eye.`;
    return `${first} ${second} ${third}`;
  })();

  return (
    <section
      aria-labelledby="far-head"
      style={{ marginTop: 56, paddingTop: 36, borderTop: `1px solid ${HAIR}`, display: "flex", flexDirection: "column", gap: 16 }}
    >
      <div className="row-between" style={{ gap: "6px 24px" }}>
        <h2 id="far-head" className="h-sec">
          iii · how far the landmark words were
        </h2>
        <span style={{ fontSize: 15, color: MUTED }}>place among {comma(VOCAB)} word-pieces · drawn on a log scale</span>
      </div>
      <p style={{ margin: 0, fontSize: 24, lineHeight: 1.4, maxWidth: 980 }}>
        {lead ? (
          <>
            Through the lens, “{lead.w}” was {lead.lens != null ? `the ${ordinal(lead.lens)}` : <M c={c} />} most likely word-piece out of{" "}
            {comma(VOCAB)}; to the naked eye, {lead.eye != null ? `the ${ordinal(lead.eye)}` : <M c={c} />}.
          </>
        ) : (
          <>
            This sighting did not track the landmarks
            {words.length ? <>; it tracked {listAnd(others)}, which are listed below</> : <>: <M c={c} /></>}.
          </>
        )}
      </p>
      {lead && untracked.length ? (
        <p className="status-line" style={{ margin: 0 }}>
          Not tracked in this sighting: {listAnd(untracked.map((w) => `“${w}”`))}.
        </p>
      ) : null}
      {others.length && lead ? (
        <p className="status-line" style={{ margin: 0 }}>
          {s.source === "fixture" ? "Also tracked in the study’s sighting" : "Also tracked by the server whatever the landmarks"}:{" "}
          {listAnd(others)}.
        </p>
      ) : null}
      <Rubric style={{ maxWidth: 900 }}>
        Each line is one word. The further right its mark, the lower the lens ranks that word (at layer 60, the further the model was
        from saying it); each faint gridline is ten times
        further than the last, and the right edge is the {ordinal(VOCAB)}, the least likely of all. One sighting cannot show a lean toward any
        of these words: the study found such readings meaningful only beside a matched comparison (REPORT §3.1).
      </Rubric>
      <fieldset style={{ border: 0, margin: 0, padding: 0, display: "flex", flexWrap: "wrap", alignItems: "center", gap: "2px 26px" }}>
        <legend style={{ float: "left", padding: 0, marginRight: 6, ...SMALL_CAPS, letterSpacing: "0.16em", fontSize: 15, color: MUTED }}>
          show
        </legend>
        <label className="check">
          <input type="checkbox" checked={showLens} onChange={() => setShowLens((v) => !v)} style={{ width: 17, height: 17 }} />
          <span aria-hidden="true" style={{ width: 11, height: 11, borderRadius: "50%", background: "#2B2117", display: "inline-block" }} />
          through the lens
        </label>
        <label className="check">
          <input type="checkbox" checked={showEye} onChange={() => setShowEye((v) => !v)} style={{ width: 17, height: 17 }} />
          <span
            aria-hidden="true"
            style={{ width: 11, height: 11, borderRadius: "50%", border: "1.5px solid #2B2117", boxSizing: "border-box", display: "inline-block" }}
          />
          to the naked eye
        </label>
      </fieldset>
      {rows.length ? (
        <div style={{ overflowX: "auto", paddingBottom: 6 }}>
          <div style={{ minWidth: 860 }}>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: RANK_GRID,
                gap: "0 24px",
                alignItems: "end",
                paddingBottom: 8,
                borderBottom: `1px solid ${HAIR}`,
                ...SMALL_CAPS,
                letterSpacing: "0.14em",
                fontSize: 15,
                color: MUTED,
              }}
            >
              <span>word</span>
              <span style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                <span>nearer to being said</span>
                <span>further</span>
              </span>
              <span style={{ textAlign: "right" }}>through the lens</span>
              <span style={{ textAlign: "right" }}>to the naked eye</span>
            </div>
            <div aria-hidden="true" style={{ display: "grid", gridTemplateColumns: RANK_GRID, gap: "0 24px" }}>
              <span />
              <div style={{ position: "relative", height: 28, fontSize: 13, color: "#7A5C28", ...NUM }}>
                {TICKS.map((k) => (
                  <span key={k.label} style={{ position: "absolute", left: `${k.x}%`, top: 6, transform: `translateX(${k.tx})`, whiteSpace: "nowrap" }}>
                    {k.label}
                  </span>
                ))}
              </div>
              <span />
              <span />
            </div>
            <ul style={{ listStyle: "none", margin: 0, padding: 0 }}>
              {rows.map((r) => {
                const a = r.lens != null ? lx(r.lens) : null;
                const b = r.eye != null ? lx(r.eye) : null;
                return (
                  <li
                    key={r.w}
                    style={{
                      display: "grid",
                      gridTemplateColumns: RANK_GRID,
                      gap: "0 24px",
                      alignItems: "center",
                      minHeight: 54,
                      borderBottom: `1px solid ${HAIR_FAINT}`,
                    }}
                  >
                    <span style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
                      <span style={{ fontSize: 23 }}>“{r.w}”</span>
                      <span style={{ fontSize: 13, ...SMALL_CAPS, letterSpacing: "0.16em", color: "#7A5C28" }}>{r.landmark ? "landmark" : ""}</span>
                    </span>
                    <div aria-hidden="true" style={{ position: "relative", height: 32 }}>
                      {TICKS.map((k) => (
                        <div key={k.label} style={{ position: "absolute", left: `${k.x}%`, top: 0, bottom: 0, width: 1, background: HAIR_FAINT }} />
                      ))}
                      <div style={{ position: "absolute", left: 0, right: 0, top: 16, height: 1, background: HAIR_SOFT }} />
                      {showLens && showEye && a != null && b != null ? (
                        <div
                          style={{
                            position: "absolute",
                            left: `${Math.min(a, b)}%`,
                            width: `${Math.round(Math.abs(a - b) * 10) / 10}%`,
                            top: 15,
                            height: 2,
                            background: "#A07A2E",
                          }}
                        />
                      ) : null}
                      {showEye && b != null ? (
                        <div
                          style={{
                            position: "absolute",
                            left: `${b}%`,
                            top: 10,
                            width: 13,
                            height: 13,
                            marginLeft: -6.5,
                            borderRadius: "50%",
                            border: "1.5px solid #2B2117",
                            background: "#F6F0E3",
                            boxSizing: "border-box",
                          }}
                        />
                      ) : null}
                      {showLens && a != null ? (
                        <div
                          style={{
                            position: "absolute",
                            left: `${a}%`,
                            top: 10,
                            width: 13,
                            height: 13,
                            marginLeft: -6.5,
                            borderRadius: "50%",
                            background: "#2B2117",
                          }}
                        />
                      ) : null}
                    </div>
                    <span style={{ textAlign: "right", fontSize: 17, ...NUM }}>
                      <span style={SR_ONLY}>through the lens: </span>
                      {r.lens != null ? ordinal(r.lens) : <M c={c} />}
                    </span>
                    <span style={{ textAlign: "right", fontSize: 17, color: MUTED, ...NUM }}>
                      <span style={SR_ONLY}>to the naked eye: </span>
                      {r.eye != null ? ordinal(r.eye) : <M c={c} />}
                    </span>
                  </li>
                );
              })}
            </ul>
          </div>
        </div>
      ) : null}
      {closing ? <p style={{ margin: 0, fontSize: 16, color: MUTED, maxWidth: 900 }}>{closing}</p> : null}
      <Apparatus title="tracked words" label="Apparatus for the landmark words" style={{ gap: 10 }}>
        <dl className="dl num">
          <dt>rank and logit</dt>
          <dd>
            the place of word t among all {comma(VOCAB)}, counted from 1, in unembed(h · J<sub>ℓ</sub>
            <sup>⊤</sup>) (the lens) and in unembed(h) (the naked eye); the logit is the softcapped score, never above 30.
            {(() => {
              const others = ordered.filter((w) => !lm.includes(w)).map((w) => `“${w}”`);
              const lmIn = ordered.filter((w) => lm.includes(w));
              return (
                <>
                  {" "}
                  The {cardinal(lmIn.length || lm.length)} landmarks are the survey labels
                  {others.length ? `; ${listAnd(others)} ${others.length === 1 ? "is" : "are"} tracked beside them` : ""}.
                </>
              );
            })()}
          </dd>
        </dl>
        <div style={{ overflowX: "auto", paddingBottom: 4 }}>
          <div style={{ minWidth: 600, maxWidth: 860 }}>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "120px repeat(4, minmax(0, 1fr))",
                gap: "0 18px",
                paddingBottom: 6,
                borderBottom: `1px solid ${HAIR}`,
                ...SMALL_CAPS,
                letterSpacing: "0.12em",
                fontSize: 14,
                color: MUTED,
              }}
            >
              <span>word</span>
              <span style={{ textAlign: "right" }}>lens rank</span>
              <span style={{ textAlign: "right" }}>lens logit</span>
              <span style={{ textAlign: "right" }}>naked-eye rank</span>
              <span style={{ textAlign: "right" }}>naked-eye logit</span>
            </div>
            {rows.map((r) => {
              const rk = c.L.ranks[r.w];
              const hand = c.hand ? L36_TRACKED_LOGITS[r.w] : undefined;
              const lensLg = rk?.lensLogit ?? hand?.[0] ?? null;
              const eyeLg = rk?.eyeLogit ?? hand?.[1] ?? null;
              const short = <Missing what={c.live ? "not exposed" : "measured live"} />;
              return (
                <div
                  key={r.w}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "120px repeat(4, minmax(0, 1fr))",
                    gap: "0 18px",
                    padding: "5px 0",
                    borderBottom: `1px solid ${HAIR_FAINT}`,
                    ...NUM,
                  }}
                >
                  <span>“{r.w}”</span>
                  <span style={{ textAlign: "right" }}>{r.lens != null ? comma(r.lens) : short}</span>
                  <span style={{ textAlign: "right" }}>{lensLg != null ? two(lensLg) : short}</span>
                  <span style={{ textAlign: "right", color: MUTED }}>{r.eye != null ? comma(r.eye) : short}</span>
                  <span style={{ textAlign: "right", color: MUTED }}>{eyeLg != null ? two(eyeLg) : short}</span>
                </div>
              );
            })}
          </div>
        </div>
        <dl className="dl num">
          <dt>caveat</dt>
          <dd>
            The lens squeezes differences between conditions 8–14 times and adds layer-dependent offsets that come from the stream’s mean.
            Centre first, and read against a matched control, never alone (REPORT §3.1, §3.5).
          </dd>
          <dt>matched control</dt>
          <dd style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: "4px 14px" }}>
            <ControlSighting layerIdx={c.idx} />
            {s.tokenIndex === 0 ? (
              <span style={{ color: MUTED }}>
                a nameless or background condition, read where it chose its first word-piece, as this sighting is.
              </span>
            ) : null}
          </dd>
        </dl>
      </Apparatus>
    </section>
  );
};

/* ---------------------------------------------------- iv · the neurons that fired */

const Loz: React.FC<{ pos: boolean; size?: number }> = ({ pos, size = 13 }) => (
  <svg width={size} height={size} viewBox="0 0 13 13" aria-hidden="true">
    <path d="M6.5 0.9 L12.1 6.5 L6.5 12.1 L0.9 6.5 Z" fill={pos ? "#A07A2E" : "none"} stroke="#A07A2E" strokeWidth="1.2" />
  </svg>
);

interface PoleView {
  p: PoleReading;
  shared: Set<string>;
  agreeOn: string[];
  lensGloss: string;
  eyeGloss: string;
  note: string;
}

function poleView(c: Ctx, p: PoleReading): PoleView {
  const lens = p.lens?.map((t) => t.token) ?? null;
  const eye = p.eye?.map((t) => t.token) ?? null;
  const shared = new Set(lens && eye ? lens.filter((t) => eye.includes(t)) : []);
  const agreeOn = [...shared].map((t) => wp(t));
  const hand = c.hand ? L36_POLES[p.neuron] : undefined;
  if (hand) return { p, shared, agreeOn, ...hand };
  let note = "";
  if (shared.size && (c.idx < 40 || c.idx > 57)) {
    note = `Agreement was tested as a sign of trust only at layers 41 to 58; ${c.idx < 40 ? "down" : "up"} here it is a hint, not a warrant.`;
  }
  return { p, shared, agreeOn, lensGloss: lens ? glossLine(lens) : "", eyeGloss: eye ? glossLine(eye) : "", note };
}

const Poles: React.FC<{ c: Ctx }> = ({ c }) => {
  const { L, idx, no } = c;
  const [more, setMore] = useState(false);
  const poleRanking = useTheodoliteStore((s) => s.poleRanking);
  const setPoleRanking = useTheodoliteStore((s) => s.setPoleRanking);
  const setSteering = useTheodoliteStore((s) => s.setSteering);
  const setSurveyMode = useTheodoliteStore((s) => s.setSurveyMode);
  const threads = c.s.threads;
  const poles = L.poles;
  const read = poles.slice(0, 5).map((p) => poleView(c, p));
  const rest = poles.slice(5);
  const neg = poles.filter((p) => p.face < 0).length;
  const pos = poles.length - neg;
  const labelled = read.filter((v) => v.p.lens && v.p.eye);
  const agreeing = labelled.filter((v) => v.shared.size > 0).length;
  // bars against the loudest listed pole, whatever order the server ranked them in
  const maxS = maxAct(L) ?? 1;
  const shownWords = labelled.length ? Math.max(...labelled.map((v) => v.p.lens?.length ?? 0)) : 0;
  // the first listed in the sighting's own ranking: the server reads words for ranks 0-4 only, so the loudest by |a| among
  // the fifteen may have none when they are ranked by write size
  const loud = leadOf(L);
  const byWrite = c.s.poleRanking === "write";
  const rw = rankingWords(c.s.poleRanking);
  const leadName = byWrite ? "the biggest writer" : "the loudest neuron";
  const loudScores = loud?.lens && loud.lens.every((t) => t.logit != null) ? loud.lens.map((t) => (t.logit as number).toFixed(2)) : null;
  const touching = threads.filter((t) => t.src.idx === idx || t.dst.idx === idx);
  // a trace by hand reads the exact ids up to the position read, as the instrument now sends it (store, traceReaders)
  const traceIds = c.live && c.s.contextIds && c.s.contextIds.length ? c.s.contextIds : null;
  const traceText = traceIds ? `"token_ids": [${traceIds.join(", ")}]` : `"prompt": "[THIS SIGHTING]"`;
  const srcLayers = threads.map((t) => t.src.idx + 1);

  const aside = poles.length
    ? [
        byWrite ? `${cardinal(poles.length)} wrote the most` : `${cardinal(poles.length)} fired hardest`,
        neg >= pos
          ? `${cardinal(neg)} on the negative face, ${cardinal(pos)} on the positive`
          : `${cardinal(pos)} on the positive face, ${cardinal(neg)} on the negative`,
        labelled.length
          ? `the lens and the naked eye agree on ${agreeing ? cardinal(agreeing) : "none"} of ${cardinal(labelled.length)}`
          : "no words read at this layer",
      ].join(" · ")
    : null;

  const nudgeWith = (p: PoleReading) => {
    setSteering({ pole_layer: idx, pole_neuron: p.neuron, pole_polarity: p.face });
    navigate(href.nudge());
  };
  const surveyFrom = (p: PoleReading) => {
    setSurveyMode("pole_motif", [idx, p.neuron, p.face]);
    navigate(href.landmarks());
  };

  return (
    <section
      aria-labelledby="poles-head"
      style={{ marginTop: 56, paddingTop: 36, borderTop: `1px solid ${HAIR}`, display: "flex", flexDirection: "column", gap: 16 }}
    >
      <div className="row-between" style={{ gap: "6px 24px" }}>
        <h2 id="poles-head" className="h-sec">
          iv · the neurons that fired
        </h2>
        {aside ? <span style={{ fontSize: 15, color: MUTED }}>{aside}</span> : null}
      </div>
      <IfRubrics>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "10px 48px" }}>
          <p className="rubric" style={{ flex: "1 1 380px", maxWidth: "none" }}>
            {labelled.length
              ? `The first ${cardinal(read.length)} listed are read below.`
              : `The first ${cardinal(read.length)} listed are shown below; this sighting read no words for them at this layer.`}{" "}
            Each neuron has two faces: a filled lozenge marks its positive face, a hollow one its negative face, and each face pushes the
            stream the opposite way to the other.
          </p>
          <p className="rubric" style={{ flex: "1 1 380px", maxWidth: "none" }}>
            Read these words as rumours, not findings. Labels held up at the layers the study sampled from 41 to 58, and were at chance at
            11 to 31; layers 32 to 40 were not sampled (REPORT §3.2; N1). One free sign of trust is when the lens and the naked eye share a
            word: within layers 41 to 58, that agreement ranked a trustworthy label above an untrustworthy one 84 times in 100{" "}
            <span style={{ fontStyle: "normal" }}>(AUC 0.84)</span>. The study compared ten words a side
            {shownWords && shownWords < 10 ? `; here only ${cardinal(shownWords)} are shown` : ""}.
          </p>
        </div>
      </IfRubrics>
      {poles.length ? (
        <ol style={{ listStyle: "none", margin: 0, padding: 0, borderTop: `1px solid ${HAIR_SOFT}` }}>
          {read.map((v, i) => {
            const p = v.p;
            const posFace = p.face > 0;
            const agree = v.shared.size > 0;
            const both = !!(p.lens && p.eye);
            const words = (list: TokenReading[] | null, glossText: string, title: string) => (
              <div style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 0 }}>
                <span style={{ ...SMALL_CAPS, letterSpacing: "0.16em", fontSize: 15, color: MUTED }}>{title}</span>
                {list ? (
                  <>
                    <div style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: "2px 16px", fontSize: 25, lineHeight: 1.35 }}>
                      {list.map((t, k) => (
                        <Wp
                          key={`${t.token}-${k}`}
                          token={t.token}
                          style={{ borderBottom: v.shared.has(t.token) ? "2px solid #A07A2E" : "2px solid transparent" }}
                        />
                      ))}
                    </div>
                    {glossText ? <span style={{ fontSize: 15, fontStyle: "italic", color: MUTED }}>{glossText}</span> : null}
                  </>
                ) : (
                  <span>
                    <NotRead />
                  </span>
                )}
              </div>
            );
            return (
              <li
                key={p.neuron}
                style={{
                  display: "flex",
                  flexWrap: "wrap",
                  gap: "16px 44px",
                  padding: "24px 0 22px",
                  borderBottom: `1px solid ${HAIR_SOFT}`,
                  alignItems: "flex-start",
                }}
              >
                <div style={{ flex: "1 1 200px", maxWidth: 260, display: "flex", flexDirection: "column", gap: 6 }}>
                  <span style={{ ...SMALL_CAPS, letterSpacing: "0.18em", fontSize: 15, color: MUTED }}>
                    {rw.rows[i]}
                  </span>
                  <span style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
                    <span style={{ ...SMALL_CAPS, letterSpacing: "0.16em", fontSize: 16, color: MUTED }}>neuron</span>
                    <span style={{ fontSize: 32, lineHeight: 1.05, ...NUM }}>{p.neuron}</span>
                  </span>
                  <span style={{ display: "flex", alignItems: "center", gap: 9, fontSize: 16 }}>
                    <Loz pos={posFace} />
                    {posFace ? "positive face" : "negative face"}
                  </span>
                  <div style={{ display: "flex", flexDirection: "column", gap: 5, paddingTop: 2 }}>
                    <span style={{ fontSize: 16, color: MUTED }}>
                      strength <span style={{ color: "#2B2117", ...NUM }}>{Math.abs(p.act).toFixed(1)}</span>
                    </span>
                    <div aria-hidden="true" className="bar-track" style={{ maxWidth: 200 }}>
                      <div className="bar-fill" style={{ width: `${Math.round((Math.abs(p.act) / maxS) * 1000) / 10}%` }} />
                    </div>
                  </div>
                </div>
                <div
                  style={{
                    flex: "999 1 480px",
                    minWidth: 0,
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
                    gap: "16px 40px",
                  }}
                >
                  {words(p.lens, v.lensGloss, "through the lens")}
                  {words(p.eye, v.eyeGloss, "to the naked eye")}
                </div>
                <div style={{ flex: "1 1 230px", minWidth: 0, display: "flex", flexDirection: "column", gap: 3 }}>
                  {both ? (
                    <>
                      <div style={{ display: "flex", alignItems: "center", gap: 10, minHeight: 30 }}>
                        {agree ? (
                          <svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true">
                            <circle cx="12" cy="12" r="10.4" fill="none" stroke="#A07A2E" strokeWidth="1.2" />
                            <circle cx="12" cy="12" r="7.4" fill="none" stroke="#A07A2E" strokeWidth="0.7" />
                            <path d="M12 7 L15.2 12 L12 17 L8.8 12 Z" fill="#A07A2E" />
                          </svg>
                        ) : (
                          <svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true">
                            <circle cx="12" cy="12" r="10.4" fill="none" stroke={MUTED} strokeWidth="1" strokeDasharray="2.2 2.8" />
                          </svg>
                        )}
                        <span style={{ fontSize: 18, color: agree ? "#2B2117" : MUTED }}>{agree ? "they agree" : "they disagree"}</span>
                      </div>
                      <span style={{ fontSize: 15, fontStyle: "italic", color: MUTED }}>
                        {agree ? `both read ${listAnd(v.agreeOn)}` : "no word in common"}
                      </span>
                    </>
                  ) : null}
                  {v.note ? <span style={{ fontSize: 15, fontStyle: "italic", color: MUTED }}>{v.note}</span> : null}
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "0 20px", marginTop: 4, fontSize: 17 }}>
                    <a href={href.neuron(no, p.neuron, p.face)} aria-label={`Open neuron ${p.neuron}`} style={LINK44}>
                      open neuron
                    </a>
                    <a
                      href={href.nudge()}
                      aria-label={`Nudge with it: neuron ${p.neuron}`}
                      style={LINK44}
                      onClick={(e) => {
                        e.preventDefault();
                        nudgeWith(p);
                      }}
                    >
                      nudge with it
                    </a>
                    <a
                      href={href.landmarks()}
                      aria-label={`Survey from it: neuron ${p.neuron}`}
                      style={LINK44}
                      onClick={(e) => {
                        e.preventDefault();
                        surveyFrom(p);
                      }}
                    >
                      survey from it
                    </a>
                  </div>
                </div>
              </li>
            );
          })}
        </ol>
      ) : (
        <p style={{ margin: 0 }}>
          The neurons that fired at this layer: <M c={c} />.
        </p>
      )}

      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: "10px 28px", paddingTop: 6 }}>
        {rest.length ? (
          <button
            type="button"
            className="btn-ghost plain"
            onClick={() => setMore((m) => !m)}
            aria-expanded={more}
            style={{ fontSize: 17, padding: "0 20px", minHeight: 48 }}
          >
            {more
              ? rest.length === 1
                ? "hide the other one"
                : `hide the ${cardinal(rest.length)} others`
              : `${cardinal(rest.length)} more neuron${rest.length === 1 ? "" : "s"}`}
          </button>
        ) : (
          <span />
        )}
        <span style={{ fontSize: 16, color: MUTED }}>
          threads, the later neurons that listen to these:{" "}
          <span style={{ ...SMALL_CAPS, letterSpacing: "0.12em", color: "#2B2117" }}>[READERS APPEAR AFTER TRACING]</span>
        </span>
      </div>
      {more && rest.length ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <ul
            style={{
              listStyle: "none",
              margin: 0,
              padding: 0,
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))",
              gap: "0 40px",
            }}
          >
            {rest.map((p) => {
              const face = p.face > 0 ? "positive" : "negative";
              const st = Math.abs(p.act).toFixed(1);
              return (
                <li
                  key={p.neuron}
                  style={{ display: "flex", alignItems: "center", gap: 10, minHeight: 50, borderBottom: `1px solid ${HAIR_FAINT}`, fontSize: 17 }}
                >
                  <Loz pos={p.face > 0} size={12} />
                  <a
                    href={href.neuron(no, p.neuron, p.face)}
                    aria-label={`Open neuron ${p.neuron}, ${face} face, strength ${st}`}
                    style={{ ...LINK44, ...NUM }}
                  >
                    neuron {p.neuron}
                  </a>
                  <span style={{ fontSize: 15, color: MUTED }}>{face}</span>
                  <span style={{ marginLeft: "auto", ...NUM }}>{st}</span>
                </li>
              );
            })}
          </ul>
          <p style={{ margin: 0, fontSize: 15, color: MUTED }}>
            The instrument reads words for the first five listed only. Open a neuron to read its words.
          </p>
        </div>
      ) : null}
      <Rubric style={{ margin: "4px 0 0", maxWidth: 980 }}>
        Strength is how hard a neuron fired; its write is how much it changed the stream. This sighting lists {rw.fifteen}. Ranked either
        way, the fifteen carry only a part of what a layer’s memory writes (where the study looked, 10–41% at layers 21 to 59 and about
        70% at layers 3 and 11), so much of the story lies elsewhere. Before you nudge: the server’s own default
        nudge is 2.4 to 7.4 times the size of the stream itself and wrecks the model’s output. The nudge page sends a hundredth of the
        stream instead, and even that can break a plain answer. Start very small.
      </Rubric>
      <Apparatus title="the poles" label="Apparatus for the neurons that fired" style={{ gap: 10 }}>
        <dl className="dl num" style={{ maxWidth: 1100 }}>
          <dt>poles</dt>
          <dd>
            the {poles.length || 15} GeGLU neurons (Shazeer 2020) [15] listed at this word-piece,{" "}
            {byWrite ? (
              <>
                the top fifteen by write size |a| · ‖w<sub>down</sub> ⊙ w<sub>post</sub>‖ (strength |a| = |mlp_act|)
              </>
            ) : (
              "the top fifteen by strength |a| = |mlp_act|"
            )}
            ; σ = sign(a) names the pole that fired. A neuron’s two faces are its two poles, positive and negative, after Cammarata 2026, “Neural Networks Are Already
            Clean” [9].
          </dd>
          <dt>labels</dt>
          <dd>
            through the lens (J-VINDEX): unembed((σ · w<sub>down</sub>) · J<sup>⊤</sup>); to the naked eye: unembed(σ · w<sub>down</sub>).
            Here w<sub>down</sub> is the neuron’s down-projection column, scaled by the post-feed-forward norm weight. The name VINDEX is
            borrowed from LARQL (chrishayuk) [10].
          </dd>
          <dt>signed a</dt>
          <dd>{poles.length ? poles.map((p) => `${p.neuron} ${signed(p.act)}`).join("  ·  ") : <M c={c} />}</dd>
          <dt>label scores</dt>
          <dd>
            {loudScores ? (
              <>
                {leadName}’s lens words score {listAnd(loudScores)}
                {loudScores.every((v) => parseFloat(v) >= 25) ? ", close to the cap of 30" : ", under the cap of 30"}. They are scores for
                one neuron’s direction scaled to unit size, not the model’s probabilities (E3; REPORT §3.4, bug 8).
              </>
            ) : (
              <>
                {leadName}’s lens words and their scores: <NotRead />. They are scores for one neuron’s direction scaled to unit
                size, not the model’s probabilities (E3; REPORT §3.4, bug 8).
              </>
            )}
          </dd>
          <dt>what held up</dt>
          <dd>
            At layers 11–31 the two eyes’ labels are equal, and at chance; at the layers sampled there (11, 21 and 31) single-neuron effects
            had a split-half reliability of 0.00–0.07, and layers 32–40 were not sampled (erratum E43). At layers 41–58 J++ labels predicted
            causal effects (P@10 0.35–0.67, against a null of at most 0.003); J++ gained 0.04–0.12 over the naked eye at layers 41–46, was
            equal or slightly behind at 51–58 (erratum E41), and the Jacobian lens matched or beat it. There, agreement of the two eyes’ top
            ten scored AUC 0.84 (REPORT §3.2).{" "}
            {N1_LAYERS.includes(no)
              ? `This layer was sampled: N1 took layers ${listAnd(N1_LAYERS.map(String))}.`
              : `This layer was not sampled: N1 took layers ${listAnd(N1_LAYERS.map(String))}.`}
          </dd>
          <dt>two poles</dt>
          <dd>
            Gemma-4’s poles are truly two-sided, through the sign of the up-projection: the gate is open in 72–100% of strong events, and the
            double-negative regime [9] holds under 1% of strong-event mass (13–33% of the top 1% in sparse layers). The two poles fire on
            unrelated inputs, so one input label per neuron cannot describe both (bug 7). The fix: activation contexts stored per pole, output
            labels only from layer 41, gated by the two eyes’ agreement, and spot-checked causally (§3.3, §3.5).
          </dd>
          <dt>strength or write</dt>
          <dd style={{ display: "flex", flexDirection: "column", gap: 8, alignItems: "flex-start" }}>
            <span>
              |a| is not the size of the write. A neuron adds s · a · (w<sub>down</sub> ⊙ w<sub>post</sub>) / rms(y) to the stream. In E3
              (12 prompts, last word-piece, indicative) the fifteen explained 10–41% of the feed-forward write at layers 21, 31, 41, 51 and
              59 and 70–74% at layers 3 and 11, the seven it sampled, ranked by |a| or by write size, within half a percentage point
              (E3_lens_validation_d_analysis.json): by |a|, 14% at
              layer 31 and 22% at layer 41; by write size, 14% and 21% (bug 8). The interface ranks by write size unless you choose
              otherwise; the server’s default and the recorded sighting rank by |a|.
            </span>
            <button
              type="button"
              className="btn-ghost"
              aria-pressed={poleRanking === "write"}
              onClick={() => {
                setPoleRanking(poleRanking === "write" ? "activation" : "write");
                retakeIfLive();
              }}
              aria-describedby="leaf-write-rank-note"
              style={{ display: "flex", alignItems: "center", gap: 7 }}
            >
              <svg width="9" height="11" viewBox="0 0 9 11" aria-hidden="true">
                <path d="M4.5 0.8 L8.2 5.5 L4.5 10.2 L0.8 5.5 Z" fill={poleRanking === "write" ? "#A07A2E" : "none"} stroke="#A07A2E" strokeWidth="1" />
              </svg>
              rank by write size
            </button>
            <span id="leaf-write-rank-note">
              Ranking by write size orders the neurons by s · |a| · ‖w<sub>down</sub> ⊙ w<sub>post</sub>‖ / rms(y); within one layer s and rms(y)
              are shared, so the server ranks by |a| · ‖w<sub>down</sub> ⊙ w<sub>post</sub>‖. In the study the two rankings shared 67–96% of
              their top fifteen (Jaccard, E3). This sighting is ranked {c.s.poleRanking === "write" ? "by write size" : "by |a|"}
              {L.ffnExplained != null ? (
                <>
                  , and its fifteen here explain {(L.ffnExplained * 100).toFixed(1)}% of this layer’s feed-forward write (⟨Σ c<sub>i</sub>, f⟩ /
                  ⟨f, f⟩, f = post − s · mid)
                </>
              ) : (
                <>
                  ; the share of the write they explain: <M c={c} />
                </>
              )}
              .
              {L.poles.some((p) => p.writeSize != null) ? (
                <>
                  {" "}Write sizes, {c.s.poleRanking === "write" ? "biggest first" : "loudest first"}:{" "}
                  {L.poles
                    .slice(0, 5)
                    .map((p) => `${p.neuron} ${p.writeSize != null ? p.writeSize.toFixed(2) : "?"}`)
                    .join(" · ")}
                  .
                </>
              ) : null}
            </span>
          </dd>
        </dl>
        <dl className="dl num" style={{ maxWidth: 1100 }}>
          <dt>before you nudge</dt>
          <dd>
            {DEFAULT_NUDGE_LAYERS.includes(idx) ? (
              <>
                The server’s default nudge strikes this layer, one of five (layers {DEFAULT_NUDGE_LAYERS.map((j) => j + 1).join(", ")};{" "}
                <Mono>layers=[{DEFAULT_NUDGE_LAYERS.join(", ")}]</Mono>), at scale 650:{" "}
                {idx === L36_IDX ? "2.4 times the median stream strength here" : "2.4 to 7.4 times the median stream strength at those layers"}{" "}
                (V7; REPORT §3.4, bug 1). That default is kept for old callers; the nudge page sends the mended recipe at 0.01 of ‖h‖.
              </>
            ) : (
              <>
                The server’s default nudge strikes five layers, not this one (layers {DEFAULT_NUDGE_LAYERS.map((j) => j + 1).join(", ")};{" "}
                <Mono>layers=[{DEFAULT_NUDGE_LAYERS.join(", ")}]</Mono>), at scale 650: 2.4 to 7.4 times the median stream strength there (V7;
                REPORT §3.4, bug 1). That default is kept for old callers; the nudge page sends the mended recipe at 0.01 of ‖h‖.
              </>
            )}{" "}
            {idx === L36_IDX
              ? "From this layer up to layer 59, every lens covector puts 99–100% of its energy in the quietest tenth of dimensions, most of it in the quiet carry channel (REPORT §2.2; erratum E7)."
              : idx > L36_IDX && idx <= 58
                ? "Here, as everywhere from layer 36 up to layer 59, every lens covector puts 99–100% of its energy in the quietest tenth of dimensions, most of it in the quiet carry channel (REPORT §2.2; erratum E7)."
                : "From layer 36 up to layer 59, every lens covector puts 99–100% of its energy in the quietest tenth of dimensions, most of it in the quiet carry channel (REPORT §2.2; erratum E7)."}{" "}
            Push with the final-norm direction, the quiet dimensions masked and the start mark spared, at 0.005–0.01 of ‖h‖: masked that
            way the study’s nudge read fluently there and broke at 0.02 (§3.4, §3.5; V7); activation steering after Turner et al. and
            Panickssery et al. [21][22]. <a href={href.nudge()}>the nudge</a>
          </dd>
          <dt>threads</dt>
          <dd>
            {c.s.threadScore === "direct" ? (
              <>
                chosen by D, the direct-path estimate of what the source adds to the reader’s gate (real activation, layer scalars, exact RMSNorm
                Jacobians): ρ ≈ 0.87 with the real effect, where C = ⟨σ<sub>i</sub> w<sub>down</sub>, w<sub>gate</sub>⟩ · |a| [14] reaches 0.41
                (bug 4).{" "}
              </>
            ) : (
              <>
                C = ⟨σ<sub>i</sub> w<sub>down</sub>, w<sub>gate</sub>⟩ · |a| [14], a weak guide: ρ ≈ 0.41 with the real effect, where a
                direct-path estimate reaches 0.87 (bug 4).{" "}
              </>
            )}
            {c.s.poleRanking === "write"
              ? "These threads join the fifteen biggest writers listed; V7 measured both scores only on threads among the fifteen ranked by |a|. "
              : null}
            {threads.length === 0 ? (
              <>
                This sighting’s threads: <M c={c} />.
              </>
            ) : touching.length === 0 ? (
              <>
                None of this sighting’s {comma(threads.length)} threads touches layer {no}; they start in{" "}
                {Math.min(...srcLayers) === Math.max(...srcLayers)
                  ? `layer ${Math.min(...srcLayers)}`
                  : `layers ${Math.min(...srcLayers)}–${Math.max(...srcLayers)}`}
                .
              </>
            ) : (
              <>
                {capital(cardinal(touching.length))} of this sighting’s {comma(threads.length)} threads touch{touching.length === 1 ? "es" : ""}{" "}
                layer {no}
                {touching.length > 6 ? "; the six strongest" : ""}:{" "}
                {touching
                  .slice(0, 6)
                  .map(
                    (t) =>
                      `neuron ${t.src.neuron} (layer ${t.src.idx + 1}, ${t.src.face > 0 ? "positive" : "negative"}) → neuron ${t.dst.neuron} (layer ${
                        t.dst.idx + 1
                      }, ${t.dst.face > 0 ? "positive" : "negative"}), ${t.c != null ? `C = ${minus(t.c.toFixed(1))}` : "C not sent"}${
                        t.directGate != null ? `, D = ${minus(t.directGate.toFixed(3))}` : ""
                      }`,
                  )
                  .join("; ")}
                .
              </>
            )}
          </dd>
          {loud ? (
            <>
              <dt>by hand</dt>
              <dd style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 0 }}>
                <span style={{ color: MUTED }}>
                  {leadName}, three ways; these count layers from 0
                  {traceIds ? "; the trace reads this sighting’s exact word-piece ids, up to the one read here, without any nudge" : ""}:
                </span>
                <pre style={PRE}>
                  {`DESCRIBE POLE LAYER ${idx} SLOT ${loud.neuron} POLARITY ${loud.face > 0 ? "+" : "-"}
inspect_pole(layer=${idx}, neuron_idx=${loud.neuron}, polarity=${loud.face > 0 ? "1.0" : "-1.0"})
POST /api/scientist/trace_readers
{"layer": ${idx}, "neuron_idx": ${loud.neuron}, "polarity": ${loud.face > 0 ? "1.0" : "-1.0"}, "max_layer_hop": 6, "top_k": 10, ${traceText}}`}
                </pre>
              </dd>
            </>
          ) : null}
        </dl>
      </Apparatus>
    </section>
  );
};

/* ------------------------------------------------------ v · kinship with other layers */

const KA = [0xec, 0xe4, 0xd5];
const KB = [0x7a, 0x5c, 0x28];
/** The canvas's gilt shading: 0.25 → vellum hairline, 1 → deep gilt. */
const kinShade = (v: number) => {
  const t = Math.max(0, Math.min(1, (v - 0.25) / 0.75));
  return `#${KA.map((ch, i) => Math.round(ch + (KB[i] - ch) * t).toString(16).padStart(2, "0")).join("")}`;
};

interface KinLabel {
  key: string;
  k: number;
  text: string;
  color: string;
  border: string;
  prefer: "left" | "right";
}

/** Places the three labels above the strip without letting them run into each other. */
function placeLabels(labels: KinLabel[]) {
  const ROW = 28;
  const placed: { key: string; x: number; dir: "left" | "right"; row: number; lo: number; hi: number; l: KinLabel }[] = [];
  for (const l of labels) {
    const x = ((l.k + 0.5) / N_LAYERS) * 100;
    const w = ((l.text.length * 7.4 + 12) / 900) * 100; // worst case: the strip at its minimum width of 900px
    let dir = l.prefer;
    if (dir === "right" && x + w > 100) dir = "left";
    if (dir === "left" && x - w < 0) dir = "right";
    const lo = dir === "right" ? x : x - w;
    const hi = dir === "right" ? x + w : x;
    let row = 0;
    while (placed.some((p) => p.row === row && lo < p.hi + 1 && hi > p.lo - 1)) row++;
    placed.push({ key: l.key, x, dir, row, lo, hi, l });
  }
  const rows = placed.reduce((m, p) => Math.max(m, p.row + 1), 1);
  return { placed, extra: (rows - 1) * ROW, ROW };
}

const Kinship: React.FC<{ c: Ctx }> = ({ c }) => {
  const { s, idx, no } = c;
  const kinship = s.kinship;
  const row = useMemo(
    () => (kinship && kinship.length === N_LAYERS * N_LAYERS ? kinship.slice(idx * N_LAYERS, (idx + 1) * N_LAYERS) : null),
    [kinship, idx],
  );
  // CKA runs over every position read up to this one (kinship_n_states = read_pos + 1), the answer's own pieces included
  const words =
    s.nTokens != null ? `the ${comma(s.nTokens)} word-pieces of this conversation read up to here` : "the word-pieces of this conversation read up to here";
  const beforeBos = !c.live;

  const facts = useMemo(() => {
    if (!row) return null;
    const others = row.map((v, k) => ({ k, v })).filter((x) => x.k !== idx);
    const byV = [...others].sort((x, y) => y.v - x.v);
    const most = byV[0];
    const second = byV[1];
    const least = byV[byV.length - 1];
    const least2 = byV[byV.length - 2];
    let lo = idx;
    let hi = idx;
    while (lo - 1 >= 0 && row[lo - 1] > 0.85) lo--;
    while (hi + 1 < N_LAYERS && row[hi + 1] > 0.85) hi++;
    return { most, second, least, least2, lo, hi };
  }, [row, idx]);

  const labels = useMemo(() => {
    if (!facts) return null;
    const { most, least } = facts;
    return placeLabels([
      { key: "self", k: idx, text: `itself · layer ${no}`, color: "#8F2B1E", border: "1.5px solid #8F2B1E", prefer: most.k > idx ? "left" : "right" },
      {
        key: "most",
        k: most.k,
        text: `most alike · layer ${most.k + 1} · ${most.v.toFixed(2)}`,
        color: "#2B2117",
        border: "1px solid #A07A2E",
        prefer: most.k > idx ? "right" : "left",
      },
      {
        key: "least",
        k: least.k,
        text: `least alike · layer ${least.k + 1} · ${least.v.toFixed(2)}`,
        color: MUTED,
        border: "1px solid #A07A2E",
        prefer: least.k < N_LAYERS / 2 ? "right" : "left",
      },
    ]);
  }, [facts, idx, no]);

  const nth = (k: number) => (k === N_LAYERS - 1 ? "the last, " : k === 0 ? "the first, " : "");

  return (
    <section
      aria-labelledby="kin-head"
      style={{ marginTop: 56, paddingTop: 36, borderTop: `1px solid ${HAIR}`, display: "flex", flexDirection: "column", gap: 14 }}
    >
      <div className="row-between" style={{ gap: "6px 24px" }}>
        <h2 id="kin-head" className="h-sec">
          v · kinship with other layers
        </h2>
        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "6px 12px", fontSize: 15, color: MUTED }}>
          <span>less alike</span>
          {[0.3, 0.5, 0.7, 0.9, 1].map((v) => (
            <span key={v} style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span aria-hidden="true" style={{ width: 22, height: 13, background: kinShade(v), display: "inline-block" }} />
              <span style={NUM}>{v.toFixed(1)}</span>
            </span>
          ))}
          <span>more alike</span>
        </div>
      </div>
      <p style={{ margin: 0, fontSize: 22, lineHeight: 1.4, maxWidth: 980 }}>
        {facts ? (
          <>
            {facts.hi > facts.lo
              ? `Layer ${no} keeps close kin from layer ${facts.lo + 1} to layer ${facts.hi + 1}, all above 0.85.`
              : `No layer beside it comes above 0.85.`}{" "}
            Most alike is layer {facts.most.k + 1}, at {facts.most.v.toFixed(2)}; least alike is layer {facts.least.k + 1}, at{" "}
            {facts.least.v.toFixed(2)}, and then {nth(facts.least2.k)}layer {facts.least2.k + 1}, at {facts.least2.v.toFixed(2)}.
          </>
        ) : (
          <>
            This layer’s kinship with the others: <M c={c} />.
          </>
        )}
      </p>
      <Rubric style={{ maxWidth: 920 }}>
        Each cell is one layer, shaded by how closely it arranges {words} the way layer {no} does{" "}
        <span style={{ fontStyle: "normal" }}>(linear CKA, where 1 is the same arrangement)</span>. Darker gilt is closer kin.
        {s.tokenIndex >= 1 || s.generation?.sightEvery ? ` ${KINSHIP_GROWS}` : null}
        {beforeBos ? (
          <>
            {" "}
            This sighting was taken before the fix that stopped the instrument adding a second start mark{" "}
            <span style={{ fontStyle: "normal" }}>(the single-BOS fix)</span>, which lowers kinship by about 0.06 on average.
          </>
        ) : null}
      </Rubric>
      {row && labels ? (
        <div style={{ overflowX: "auto", paddingBottom: 6 }}>
          <div style={{ minWidth: 900, position: "relative", paddingTop: 40 + labels.extra }}>
            <div
              aria-hidden="true"
              style={{ position: "absolute", left: 0, right: 0, top: 0, height: 36 + labels.extra, fontSize: 15, whiteSpace: "nowrap", ...NUM }}
            >
              {labels.placed.map((p) => (
                <span
                  key={p.key}
                  style={{
                    position: "absolute",
                    left: `${Math.round(p.x * 100) / 100}%`,
                    bottom: p.row * labels.ROW,
                    color: p.l.color,
                    ...(p.dir === "left"
                      ? { transform: "translateX(-100%)", padding: "0 8px 4px 0", borderRight: p.l.border }
                      : { padding: "0 0 4px 8px", borderLeft: p.l.border }),
                  }}
                >
                  {p.l.text}
                </span>
              ))}
            </div>
            <div aria-hidden="true" style={{ display: "grid", gridTemplateColumns: "repeat(60, minmax(0, 1fr))", gap: 2, height: 56 }}>
              {row.map((v, k) => (
                <div
                  key={k}
                  title={`layer ${k + 1}: ${v.toFixed(2)}`}
                  style={{ background: kinShade(v), outline: k === idx ? "2px solid #8F2B1E" : "0 solid transparent", outlineOffset: 1 }}
                />
              ))}
            </div>
            <div
              aria-hidden="true"
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(60, minmax(0, 1fr))",
                gap: 2,
                marginTop: 6,
                fontSize: 12,
                color: "#7A5C28",
                textAlign: "center",
                ...NUM,
              }}
            >
              {row.map((_, k) => (
                <div key={k} style={{ display: "flex", flexDirection: "column", alignItems: "center", lineHeight: 1.2, whiteSpace: "nowrap" }}>
                  <span>{k % 6 === 5 ? "✦" : " "}</span>
                  <span style={{ color: k === idx ? "#8F2B1E" : MUTED }}>{k === 0 || k % 6 === 5 || k === idx ? String(k + 1) : " "}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : null}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 22px", fontSize: 15, color: MUTED }}>
        <span>read left to right, layer 1 to layer 60</span>
        <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span aria-hidden="true" style={{ color: "#A07A2E" }}>
            ✦
          </span>
          looks back over everything
        </span>
      </div>
      <Apparatus title="kinship" label="Apparatus for kinship with other layers">
        <dl className="dl num" style={{ maxWidth: 1100 }}>
          <dt>measure</dt>
          <dd>
            linear CKA (Kornblith et al. 2019) [13]: CKA(K, L) = ⟨K̃, L̃⟩ / (‖K̃‖ · ‖L̃‖), where K = X X<sup>⊤</sup> is the Gram matrix of one
            layer’s word-piece states <Mono>resid_post</Mono>
            {s.nTokens != null ? (
              <>
                {" "}
                over the {comma(s.nTokens)} word-pieces read up to here (positions 0 to {s.readPos ?? comma(s.nTokens - 1)}
                {s.tokenIndex >= 1
                  ? `, the prompt and the first ${s.tokenIndex} word-piece${s.tokenIndex === 1 ? "" : "s"} of its answer`
                  : ""}
                )
              </>
            ) : null}
            , centred over the word-pieces, and K̃ = HKH.
          </dd>
          <dt>this row</dt>
          <dd>
            {facts ? (
              <>
                most alike, layer {facts.most.k + 1} at {facts.most.v.toFixed(3)} and layer {facts.second.k + 1} at {facts.second.v.toFixed(3)};
                least alike, layer {facts.least.k + 1} at {facts.least.v.toFixed(3)}, then layer {facts.least2.k + 1} at{" "}
                {facts.least2.v.toFixed(3)}.{" "}
                {facts.hi > facts.lo ? `Above 0.85: layers ${facts.lo + 1}–${facts.hi + 1}. ` : "No neighbour above 0.85. "}
                All sixty values are in the raw readout above.
              </>
            ) : (
              <M c={c} />
            )}
          </dd>
          <dt>the whole map</dt>
          <dd>
            {s.kinStats ? (
              <>
                mean off the diagonal {s.kinStats.meanOffDiag.toFixed(2)}; neighbouring layers {s.kinStats.adjacent.toFixed(2)}; layer 11
                against layer 51, {s.kinStats.l11vs51.toFixed(2)}.
              </>
            ) : (
              <M c={c} />
            )}{" "}
            <a href={href.instrument()}>the full sixty by sixty map</a>
          </dd>
          <dt>caveat</dt>
          <dd>
            start marks {s.firstIds ? <Mono>[{s.firstIds.join(", ")}]</Mono> : <M c={c} />}
            {beforeBos
              ? ": taken before the single-BOS fix, which lowers kinship by about 0.06 on average"
              : "; the single-BOS fix lowers kinship by about 0.06 on average against sightings taken before it"}{" "}
            (REPORT §3.4, bug 6; <Mono>V7_bos_fix.patch</Mono>). Briggs [26] reports blocks of kin layers in this model, with a seam near layer
            26.
          </dd>
        </dl>
      </Apparatus>
    </section>
  );
};

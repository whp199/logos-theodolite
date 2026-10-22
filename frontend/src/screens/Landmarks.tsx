import React, { useMemo, useState } from "react";
import { Apparatus, LensRecord, Missing, Page, Rubric, SweepError } from "../components/kit";
import { AnswerPositionCaution, SightingFarCaution, TemplateEndCaution } from "../lib/cautions";
import type { FrameInfo } from "../store/sighting";
import { ControlSighting } from "../components/ControlSighting";
import { describeFrame, shownFrameOptions, type ShownFrame } from "../lib/frame";
import { isBaseFitted, lensLabel, lensPhrase, useReadingLens } from "../lib/lens";
import { href, type Route } from "../router";
import {
  FNW_DEFAULT,
  FRAME_AS_BUILT,
  FRAME_MENDED,
  type FrameOptions,
  type LayerReading,
  type Sighting,
  SIGHT_CAP,
  VOCAB,
  chosenOf,
  chosenPhrase,
  isChosenRow,
  ordinal,
  readWhere,
  readsInsideAnswer,
  retakeIfLive,
  sweepPayload,
  useShownSet,
  useTheodoliteStore,
} from "../store/useTheodoliteStore";
import { comma, faceWord, kindOf, minus, ms, pct, pieceWord, wp } from "../lib/words";
import { endedWords } from "./instrument/moment";
import {
  ORDERS,
  ORDER_STRETCHES,
  ORDER_TABLE,
  PRESETS,
  SINGLE_MARK_AZ,
  SINGLE_MARK_EL,
  type Triple,
  type RecordedFrame,
  cosOf,
  recordedFrame,
} from "./landmarks/recorded";
import {
  PLATE,
  RAD,
  allDots,
  chartGeometry,
  cxL,
  f1,
  ghostGeometry,
  lowdinGhost,
  loz,
  readingGeometry,
  ring,
  sparkPath,
  sx,
  syRank,
  yB,
  yE,
} from "./landmarks/geometry";

/* ------------------------------------------------------------- helpers */

const q = (w: string) => "“" + w + "”";
const norm = (w: string | null | undefined) => String(w ?? "").trim().toLowerCase();
const sgn = (n: number) => (n < 0 ? "−" + Math.abs(n) : String(n));
const d1 = (n: number) => (n < 0 ? "−" : "") + Math.abs(n).toFixed(1);
const SUP: Record<string, string> = {
  "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
};
/** Three significant figures; small values as a × 10ⁿ, as the canvas prints them. */
const sci = (v: number) => {
  if (v === 0) return "0";
  const sign = v < 0 ? "−" : "";
  const a = Math.abs(v);
  if (a >= 0.01) return sign + a.toPrecision(3);
  const e = Math.floor(Math.log10(a));
  return sign + (a / Math.pow(10, e)).toFixed(2) + " × 10" + String(e).split("").map((ch) => SUP[ch] ?? ch).join("");
};
const pct1 = (frac: number) => (frac * 100).toFixed(1) + "%";
const clampIdx = (i: number) => Math.max(0, Math.min(59, Math.round(i)));
/** Placeholder for a value this sighting does not carry. */
const NOT_HERE = "not recorded in this sighting";

const ORDINALS = ["first", "second", "third"];
const NEURON_NAMES: Triple = ["what wakes it", "what it writes", "who listens next"];
/** What each stone does in the frame: squared in order (Gram–Schmidt), each word against those before it; or all at
 * once (Löwdin), where no word lies exactly on its axis and the order names which axis is which. */
const ROLE_IN_ORDER = [
  { numeral: "I", ordinal: "first landmark", sets: "sets bearing 0°", how: "kept exactly as it lies" },
  { numeral: "II", ordinal: "second landmark", sets: "sets bearing 90°", how: "squared against the first" },
  { numeral: "III", ordinal: "third landmark", sets: "points straight up", how: "squared against both" },
];
const ROLE_AT_ONCE = [
  { numeral: "I", ordinal: "first landmark", sets: "its axis sets bearing 0°", how: "squared with the other two, all at once" },
  { numeral: "II", ordinal: "second landmark", sets: "its axis sets bearing 90°", how: "squared with the other two, all at once" },
  { numeral: "III", ordinal: "third landmark", sets: "its axis points straight up", how: "squared with the other two, all at once" },
];
const roles = (sq: FrameOptions["squaring"]) => (sq === "lowdin" ? ROLE_AT_ONCE : ROLE_IN_ORDER);
/** the frame options the store sends, centring aside: whether they are those of a preset */
const sameFrame = (a: FrameOptions, b: FrameOptions) => a.squaring === b.squaring && a.perLayer === b.perLayer && a.fnw === b.fnw;
/** "the raw rows" or "the weighted rows", as the frame shown was squared from */
const rowsWords = (f: ShownFrame) => (f.fnw ? "the weighted rows" : "the raw rows");
const PAIRS: [number, number, string][] = [
  [0, 1, "first and second"],
  [0, 2, "first and third"],
  [1, 2, "second and third"],
];

/** One line saying how a sighting's frame was built (the server's `frame`). */
function frameWords(f: FrameInfo | null): string {
  if (!f) return "in order, from raw rows, at the final layer (the frame as first built)";
  return [
    f.squaring === "lowdin" ? "all at once" : "in order",
    f.fnwWeighted ? "from the weighted rows" : "from raw rows",
    f.perLayer ? "from each layer’s own lens" : "at the final layer",
    ...(f.centred ? [`centred on ${f.nCentre === 1 ? "a matched control" : `${f.nCentre} matched controls`}`] : []),
  ].join(", ");
}

/**
 * The angles and squareness the server computed for the shown sighting's words (V7 frame_stats definitions),
 * in the shape of the study's recorded frames. Pairs are keyed by the sighting's own order.
 */
function liveFrame(sf: { norm: string[] } | null, f: FrameInfo | null): RecordedFrame | null {
  if (!sf || !f?.geometry || sf.norm.length !== 3 || sf.norm.some((w) => !w)) return null;
  const g = f.geometry;
  const [a, b, c] = sf.norm;
  const pair = (k: string) => [g.raw.cos[k], g.fnw.cos[k]] as [number, number];
  return {
    words: [a, b, c],
    cond: g.raw.cond,
    condFnw: g.fnw.cond,
    self: { [a]: g.selfCos[0], [b]: g.selfCos[1], [c]: g.selfCos[2] },
    cos: { [a + "|" + b]: pair("01"), [a + "|" + c]: pair("02"), [b + "|" + c]: pair("12") },
  };
}

interface Pole {
  idx: number;
  neuron: number;
  face: 1 | -1;
}
const poleLabel = (p: Pole) => `layer ${p.idx + 1} · neuron ${p.neuron}, ${faceWord(p.face)} face`;
const sameFace = (n: number): 1 | -1 => (n >= 0 ? 1 : -1);

/** What frame a sighting was taken in: three words, or a neuron's three directions (pole-locked). */
interface SightFrame {
  kind: "words" | "neuron";
  words: string[];
  norm: string[];
  /** display names: quoted words, or the neuron's three directions */
  names: Triple;
  pole: (Pole & { reader: { idx: number; neuron: number } | null }) | null;
}

function sightFrame(s: Sighting | null): SightFrame | null {
  if (!s) return null;
  const [a, b, c] = s.landmarks.map((x) => String(x ?? "").trim());
  // engine_runtime.py labels a pole-locked frame "Gate L{l}:N{n}", "Write L{l}:N{n}(±)@J++", "Reader L{l}:N{n}"
  const g = /^Gate L(\d+):N(\d+)$/.exec(a);
  if (g) {
    const r = /^Reader L(\d+):N(\d+)$/.exec(c);
    return {
      kind: "neuron",
      words: [],
      norm: [],
      names: NEURON_NAMES,
      pole: {
        idx: Number(g[1]),
        neuron: Number(g[2]),
        face: /\(-\)/.test(b) ? -1 : 1,
        reader: r ? { idx: Number(r[1]), neuron: Number(r[2]) } : null,
      },
    };
  }
  const words = [a, b, c];
  return {
    kind: "words",
    words,
    norm: words.map(norm),
    names: words.map((w, i) => (w ? q(w) : `[${ORDINALS[i]} landmark]`)) as Triple,
    pole: null,
  };
}

/** Where a sighting was read, for a line of facts: “read at “I”, in its reply “Hi! I”” at its first word-piece, “read at
 * “am”, the 2nd word-piece of its own answer” after (readWhere's phrase, with the piece named once). */
function readFact(s: Sighting): string {
  const w = readWhere(s);
  const at = s.readAt != null ? pieceWord(s.readAt) : "the last word-piece";
  return w.k >= 1 ? `read at ${at}, the ${ordinal(w.k)} word-piece of its own answer` : `read at ${at}${w.phrase}`;
}

/** What the server centred a sighting on, from its echo (frame.centre_kind); null when it did not say (an older server). */
function centreWords(s: Sighting): string | null {
  switch (s.centreKind) {
    case "control_prompt_end":
      return "centred on the kept control, read where it chose its first word-piece";
    case "control_reading_same_answer":
      return "centred on the control’s state at the same word-piece, reading your sighting’s own answer";
    case "none":
      // controls were sent (n_centre_prompts), yet none was subtracted here: a later word-piece without centre_answer
      return (s.frame?.nCentre ?? 0) > 0 && s.tokenIndex >= 1
        ? "not centred, since a control read where its question ends is never subtracted from a later word-piece"
        : "not centred";
    default:
      return null;
  }
}

function rankOf(L: LayerReading, w: string) {
  return L.ranks[w] ?? L.ranks[w.toLowerCase()] ?? L.ranks[" " + w] ?? null;
}

/* ------------------------------------------------------------- styles */

const smallCaps: React.CSSProperties = {
  fontVariantCaps: "all-small-caps",
  letterSpacing: "0.16em",
  fontSize: "16px",
  color: "#6B5D4D",
};
const modeBtn = (on: boolean): React.CSSProperties => ({
  fontFamily: "inherit",
  fontSize: "17px",
  color: "#2B2117",
  background: on ? "#EDE5D3" : "transparent",
  border: `1px solid ${on ? "#2B2117" : "#C9BEAC"}`,
  borderRadius: "2px",
  padding: "8px 18px",
  minHeight: "44px",
  cursor: "pointer",
});
const moveBtn = (off: boolean): React.CSSProperties => ({
  fontFamily: "inherit",
  fontSize: "15px",
  color: off ? "#6B5D4D" : "#2B2117",
  background: "transparent",
  border: `1px solid ${off ? "#E3DACA" : "#C9BEAC"}`,
  borderRadius: "2px",
  minWidth: "44px",
  minHeight: "44px",
  padding: "0 14px",
  cursor: off ? "default" : "pointer",
});
const stepBtn: React.CSSProperties = {
  fontFamily: "inherit",
  fontSize: "15px",
  color: "#2B2117",
  background: "transparent",
  border: "1px solid #C9BEAC",
  borderRadius: "2px",
  minWidth: "44px",
  minHeight: "44px",
  padding: "0 12px",
  cursor: "pointer",
};
const optLabel = (enabled: boolean): React.CSSProperties => ({
  display: "flex",
  alignItems: "center",
  gap: "10px",
  minHeight: "44px",
  cursor: enabled ? "pointer" : "not-allowed",
  color: enabled ? "#2B2117" : "#6B5D4D",
});
const optInput: React.CSSProperties = {
  width: "17px",
  height: "17px",
  accentColor: "#8F2B1E",
  margin: "0",
  flex: "none",
};
const optLegend: React.CSSProperties = {
  padding: "6px 0 0",
  fontVariantCaps: "all-small-caps",
  letterSpacing: "0.14em",
  color: "#6B5D4D",
};
const optFieldset: React.CSSProperties = {
  border: "0",
  margin: "0",
  padding: "0",
  flex: "1 1 300px",
  minWidth: "0",
  display: "flex",
  flexDirection: "column",
};
const aside: React.CSSProperties = { fontStyle: "italic", color: "#6B5D4D" };
const numTile: React.CSSProperties = {
  fontSize: "40px",
  lineHeight: 1.1,
  fontWeight: 500,
  color: "#8A6A2C",
  fontVariantNumeric: "lining-nums tabular-nums",
};
const th: React.CSSProperties = { textAlign: "right", fontWeight: 400, padding: "4px 0 6px 12px" };
const td: React.CSSProperties = { textAlign: "right", padding: "6px 0 6px 12px" };

/* ------------------------------------------------------------- pieces */

const Stone: React.FC<{ numeral: string; hatched?: boolean }> = ({ numeral, hatched }) => (
  <svg width="56" height="100" viewBox="0 0 56 100" aria-hidden="true" style={{ flex: "none" }}>
    <path d="M28 2 L31.5 7.5 L28 13 L24.5 7.5 Z" fill="#A07A2E" />
    <path d="M22 22 L28 15 L34 22 Z" fill="none" stroke="#2B2117" strokeWidth="1" />
    <path d="M22 22 L19 82 M34 22 L37 82" fill="none" stroke="#2B2117" strokeWidth="1" />
    {hatched ? (
      <path
        d="M30.5 26 L31.4 44 M32.6 26 L33.5 44 M31.5 66 L32.1 78 M33.6 66 L34.2 78"
        fill="none"
        stroke="#2B2117"
        strokeOpacity="0.3"
        strokeWidth="0.6"
      />
    ) : null}
    <path d="M14 82 H42 V88 H14 Z M10 88 H46 V95 H10 Z" fill="none" stroke="#2B2117" strokeWidth="1" />
    <path d="M4 98 H52" fill="none" stroke="#A07A2E" strokeWidth="1" />
    <text
      x="28"
      y="59"
      textAnchor="middle"
      fontFamily="EB Garamond, Garamond, serif"
      fontSize="13"
      fill="#7A5C28"
      letterSpacing={hatched ? "0.5" : undefined}
    >
      {numeral}
    </text>
  </svg>
);

const NeuronStone: React.FC<{ numeral: string; ordinal: string; name: string; value: React.ReactNode; how: string }> = ({
  numeral,
  ordinal,
  name,
  value,
  how,
}) => (
  <div style={{ flex: "1 1 280px", minWidth: "0", display: "flex", gap: "18px", alignItems: "flex-start" }}>
    <Stone numeral={numeral} />
    <div style={{ flex: "1 1 auto", minWidth: "0", display: "flex", flexDirection: "column", gap: "4px" }}>
      <span style={smallCaps}>{ordinal}</span>
      <span style={{ fontSize: "30px", fontStyle: "italic", lineHeight: 1.2, paddingBottom: "6px", borderBottom: "1px solid #C9BEAC" }}>
        {name}
      </span>
      <span style={{ fontSize: "16px", color: "#6B5D4D", fontVariantNumeric: "lining-nums" }}>{value}</span>
      <span style={{ fontSize: "15px", fontStyle: "italic", color: "#6B5D4D" }}>{how}</span>
    </div>
  </div>
);

/* ------------------------------------------------------------- screen */

/**
 * Whether the old space-then-digit rule moved any rank of this sighting: the server's flags (tracked_from_next, per layer and
 * word) and whether the rule was allowed at all (rank_lookahead; the instrument sends false). The recorded sighting and an
 * older server keep no such record, so for them the page says it is not known rather than guess from the layer-60 top word.
 */
const SpaceRule: React.FC<{ s: Sighting | null; L: LayerReading | null; li: number }> = ({ s, L, li }) => {
  if (!s) return <Missing what={NOT_HERE} />;
  const moved = s.layers
    .map((x) => ({ no: x.no, words: x.rankFromNext ? Object.keys(x.rankFromNext).filter((w) => x.rankFromNext![w]) : [] }))
    .filter((x) => x.words.length);
  const anyFlags = s.layers.some((x) => x.rankFromNext !== null);
  if (moved.length) {
    const here = L?.rankFromNext ? Object.keys(L.rankFromNext).filter((w) => L.rankFromNext![w]) : [];
    return (
      <>
        Here it applied: at {moved.length === 1 ? "one layer" : `${moved.length} layers`} some ranks were taken from the next position
        {here.length ? <>, at layer {li + 1} those of {here.map(q).join(", ")}</> : <>, none of them at layer {li + 1}</>}. Read the lens’s
        ranks with care.
      </>
    );
  }
  if (s.rankLookahead === false) return <>This sighting was taken with the rule off (rank_lookahead false): every rank is read at its own position.</>;
  if (anyFlags) return <>Here it did not apply: no rank was taken from the next position.</>;
  return (
    <>
      {s.source === "fixture" ? "The recorded sighting" : "This server"} does not record whether it applied:{" "}
      <Missing what={NOT_HERE} />
    </>
  );
};

/**
 * The red line beside a bearing, in place of the per-layer LensCaution. The instrument's bands of layers (lib/lens.ts,
 * lensBand) were measured on the lens's word readings (E1, E2, E5; R1; erratum E49). No experiment set a bearing against
 * what the model does, at any layer; the study's only tests of bearings found them swinging 58-99° with the landmarks'
 * order and catching 2.4-3.9% of the reading (V7; E6). So this line does not follow the chosen layer, and it stands at
 * layer 60 too, where a bearing is still a projection on three words.
 */
const BearingCaution: React.FC<{ sighting: Sighting | null; chart?: boolean; style?: React.CSSProperties }> = ({
  sighting,
  chart = false,
  style,
}) => {
  const { lens } = useReadingLens(sighting);
  return (
    <Rubric style={style}>
      {isBaseFitted(lens) ? "This lens was fitted on the base model, not on the chat model you are reading. " : null}
      Elsewhere a note in red beside the lens’s words says how far to trust them at each layer. Those bands of layers come from
      the words, not from bearings,{" "}
      {chart
        ? "and none of them applies to these charts, at the sixtieth layer or any other."
        : "and do not carry over here: no bearing was checked against what the model does, at this layer or any other."}{" "}
      <a href={href.lexicon("frame")}>Read the caution.</a>
    </Rubric>
  );
};

export const Landmarks: React.FC<{ route: Route }> = ({ route }) => {
  const sighting = useTheodoliteStore((s) => s.sighting);
  const { lens } = useReadingLens(sighting);
  const frameDesc = sighting ? describeFrame(sighting) : null;
  const landmarks = useTheodoliteStore((s) => s.landmarks);
  const setLandmarks = useTheodoliteStore((s) => s.setLandmarks);
  const surveyMode = useTheodoliteStore((s) => s.surveyMode);
  const lockedPole = useTheodoliteStore((s) => s.lockedPole);
  const setSurveyMode = useTheodoliteStore((s) => s.setSurveyMode);
  const takeSighting = useTheodoliteStore((s) => s.takeSighting);
  const isSweeping = useTheodoliteStore((s) => s.isSweeping);
  const question = useTheodoliteStore((s) => s.question);
  const answerBegins = useTheodoliteStore((s) => s.answerBegins);
  const compareThink = useTheodoliteStore((s) => s.compareThink);
  const writing = useTheodoliteStore((s) => s.writing);
  const steering = useTheodoliteStore((s) => s.steering);
  const chat = useTheodoliteStore((s) => s.chat);
  const system = useTheodoliteStore((s) => s.system);
  const frameOpts = useTheodoliteStore((s) => s.frame);
  const setFrame = useTheodoliteStore((s) => s.setFrame);
  const neuronSquaring = useTheodoliteStore((s) => s.neuronSquaring);
  const setNeuronSquaring = useTheodoliteStore((s) => s.setNeuronSquaring);
  const poleRanking = useTheodoliteStore((s) => s.poleRanking);
  const threadScore = useTheodoliteStore((s) => s.threadScore);
  const control = useTheodoliteStore((s) => s.control);
  const shownSet = useShownSet();
  const k = sighting?.tokenIndex ?? 0;
  const chosen = chosenOf(sighting);
  // whether a control can be centred on is the store's own rule (sweepPayload): the request it would send with centring
  // on carries the control's exact ids (or its text), unless the control is this very question, whose state less itself
  // leaves nothing to read
  const reqState = {
    chat,
    system,
    landmarks: landmarks.map((w) => w.trim()) as Triple,
    surveyMode,
    lockedPole,
    compareThink,
    writing,
    frame: frameOpts,
    neuronSquaring,
    poleRanking,
    threadScore,
    control,
  };
  const probe = sweepPayload({ ...reqState, frame: { ...frameOpts, centre: true } }, question, answerBegins, steering);
  const canCentre = (probe.centre_token_ids?.length ?? 0) > 0 || probe.centre_prompts.length > 0;
  const controlIsThis = !!control && !canCentre && (!!control.promptIds?.length || !!control.textSent);
  const selectedLayer = useTheodoliteStore((s) => s.selectedLayer);
  const setSelectedLayer = useTheodoliteStore((s) => s.setSelectedLayer);
  const [rawOpen, setRawOpen] = useState(false);

  /* ---- the landmarks as set on this page */
  const isWords = surveyMode === "monument";
  const words = landmarks;
  const nw = words.map(norm);
  // the frame's angles and squareness: the server's, when the shown sighting used these words; else the study's record
  const sfEarly = useMemo(() => sightFrame(sighting), [sighting]);
  const live = isWords && sfEarly?.kind === "words" && sfEarly.norm.slice().sort().join("|") === nw.slice().sort().join("|")
    ? liveFrame(sfEarly, sighting?.frame ?? null)
    : null;
  const frame = isWords ? (live ?? recordedFrame(nw)) : null;
  const frameFromServer = live !== null;
  const known = frame !== null;
  const lp: Pole | null = lockedPole ? { idx: lockedPole[0], neuron: lockedPole[1], face: sameFace(lockedPole[2]) } : null;
  const anyEmpty = isWords && words.some((w) => !w.trim());

  /* ---- the frame of the sighting shown */
  const sf = useMemo(() => sightFrame(sighting), [sighting]);
  const SN: Triple = sf?.names ?? ["[first landmark]", "[second landmark]", "[third landmark]"];
  const matches =
    sf !== null &&
    (isWords
      ? sf.kind === "words" && nw.join("|") === sf.norm.join("|")
      : sf.kind === "neuron" && lp !== null && sf.pole !== null &&
        sf.pole.idx === lp.idx && sf.pole.neuron === lp.neuron && sf.pole.face === lp.face);

  // the frame every frame-dependent word, label and drawing below describes: the shown sighting's own when it was taken
  // with these landmarks (the recorded one's is the frame as first built), else the frame the next sighting is taken in
  const shown = shownFrameOptions(sighting, matches, frameOpts, { surveyMode, neuronSquaring, lockedPole });
  // with no neuron chosen in neuron mode, a sighting falls back to the frame from words (and `shown` says so); the page
  // itself then describes the neuron frame a chosen neuron will be squared in
  const atOnce = (isWords || lockedPole ? shown.squaring : neuronSquaring) === "lowdin";
  // the shown sighting's own frame, whatever the page is set to: the charts of section v draw that sighting
  const ownAtOnce = !!sighting && shownFrameOptions(sighting, true, frameOpts).squaring === "lowdin";
  // the frame the next sighting from words is squared in, and the one from a neuron, as the radios set them
  const nextSquaring = isWords ? frameOpts.squaring : neuronSquaring;
  // the word frame the apparatus's formula describes, and how a neuron frame is squared (with no neuron chosen, `shown`
  // is the frame from words the server falls back to, so the neuron frame is the one the radios set)
  const wordFrame: FrameOptions = isWords ? shown : frameOpts;
  const neuronFrameSq = isWords || !lockedPole ? neuronSquaring : shown.squaring;
  const wordRows = (m: number) =>
    wordFrame.fnw ? (
      <>
        fnw ⊙ W<sub>U</sub>[t<sub>{m}</sub>]
      </>
    ) : (
      <>
        W<sub>U</sub>[t<sub>{m}</sub>]
      </>
    );
  // the two frame presets; each keeps the reader's centring, and a neuron frame keeps Gram–Schmidt in both
  const setFramePreset = (f: FrameOptions) => {
    setFrame({ ...f, centre: frameOpts.centre });
    setNeuronSquaring("gram_schmidt");
  };
  const onAsBuilt = sameFrame(frameOpts, FRAME_AS_BUILT) && neuronSquaring === "gram_schmidt";
  const onMended = sameFrame(frameOpts, FRAME_MENDED) && neuronSquaring === "gram_schmidt";

  const li = clampIdx(selectedLayer);
  const layers = useMemo(() => sighting?.layers ?? [], [sighting]);
  const L: LayerReading | null = layers[li] ?? null;
  const az = L?.az ?? null;
  const el = L?.el ?? null;
  const showReading = matches && az !== null && el !== null;

  /* ---- i · the stones */
  const swap = (i: number, j: number) => {
    const w = words.slice() as Triple;
    const t = w[i];
    w[i] = w[j];
    w[j] = t;
    setLandmarks(w);
  };
  const setWord = (i: number, v: string) => {
    const w = words.slice() as Triple;
    w[i] = v;
    setLandmarks(w);
  };
  const keepPole = lockedPole;
  const pickWords = () => setSurveyMode("monument", keepPole);
  const pickNeuron = () => setSurveyMode("pole_motif", keepPole);
  // Landmarks change only how the answer is read (SPEC §3.2): when the shown sighting is a live one whose answer was
  // written for the question the boxes still hold (its text, begun answer, chat and system prompt, nudged or not), it is
  // read again from its own word-pieces with the new frame (retakeIfLive), keeping the word-piece in view; otherwise a new
  // answer is written.
  const rereads = (() => {
    const a = shownSet;
    const s0 = sighting;
    if (!a || !s0 || s0.source !== "live" || !a.streamed || !s0.answerIds?.length) return null;
    const asked =
      a.question === question &&
      a.answerBegins === answerBegins &&
      a.chat === chat &&
      a.system === (chat ? system.trim() : "") &&
      a.nudged === steering.enabled;
    if (!asked) return null;
    // as retakeIfLive takes them: the word-piece wanted when it has arrived, else the one shown
    const on = a.sightings[a.wanted] ? a.wanted : s0.tokenIndex;
    const n = Math.min(Object.keys(a.sightings).length, SIGHT_CAP);
    return { on, n };
  })();
  const survey = () => {
    if (isWords) setLandmarks(words.map((w) => w.trim()) as Triple);
    if (rereads && useTheodoliteStore.getState().conn === "open") retakeIfLive();
    else takeSighting();
  };
  const ab = answerBegins.trim();
  const lastWord = ab.split(/\s+/).pop() ?? "";
  // where the next sighting reads, in readWhere's three phrasings for a first word-piece; then, with "sight each" on, every
  // word-piece it writes
  const readsFirst = ab
    ? /^[\p{L}\p{N}]+$/u.test(lastWord)
      ? `it will read the word “${lastWord}” in the reply “${ab}”`
      : `it will read the last word-piece of the reply “${ab}”`
    : chat
      ? `it will read the end of the chat template, after “${question.trim()}”`
      : `it will read the last word-piece of “${question.trim()}”`;
  // a sighting of word-piece k is taken where the model chose it, reading the piece before: the first is the reading above,
  // where it chooses its first word-piece, so an answer of n word-pieces gives n sightings, not n + 1
  const readsAt = rereads
    ? `it will read the answer shown again from its own word-pieces, writing nothing anew: the ${
        rereads.n === 1 ? "word-piece" : `${rereads.n} word-pieces`
      } sighted so far, and word-piece ${rereads.on + 1} stays in view`
    : !writing.sightEach
      ? readsFirst
      : writing.length === 1
        ? `${readsFirst}, where it chooses the one word-piece it writes`
        : `${readsFirst}, where it chooses its first word-piece; then one more where it chooses each later one, up to ${Math.min(
            writing.length,
            SIGHT_CAP,
          )} sightings in all${writing.length > SIGHT_CAP ? ` (it may write ${writing.length}; the first ${SIGHT_CAP} are sighted)` : ""}`;

  const nm = (i: number) => (words[i].trim() ? q(words[i].trim()) : `[${ORDINALS[i]} landmark]`);
  const lab: Triple = isWords ? [nm(0), nm(1), nm(2)] : NEURON_NAMES;
  const named = `${lab[0]}, ${lab[1]} and ${lab[2]}`;

  const sightDesc = !sf
    ? ""
    : sf.kind === "neuron" && sf.pole
      ? "taken in the frame of " + poleLabel(sf.pole)
      : isWords
        ? "taken with " + SN.join(", ") + " in that order"
        : "taken with the words " + SN.join(", ");
  const notice = !sf
    ? `No sighting yet. Take one to measure against ${named}.`
    : isWords
      ? `These readings belong to the sighting shown, ${sightDesc}. Take a new sighting to measure against ${named}.`
      : lp
        ? `These readings belong to the sighting shown, ${sightDesc}. Take a sighting in the frame of ${poleLabel(lp)} to measure against it.`
        : `These readings belong to the sighting shown, ${sightDesc}. Choose a neuron, then take a sighting in its frame.`;

  // "choose a neuron": the locked neuron's folio, else the first listed neuron at the chosen layer
  const loud = L?.poles?.[0] ?? null;
  const neuronHref = lp
    ? href.neuron(lp.idx + 1, lp.neuron, lp.face)
    : loud
      ? href.neuron(li + 1, loud.neuron, loud.face)
      : href.instrument();
  const reader =
    matches && sf?.pole?.reader ? `layer ${sf.pole.reader.idx + 1} · neuron ${sf.pole.reader.neuron}` : null;

  /* ---- ii · the engraved frame */
  // where the words truly point, from the cosines of the rows the frame was squared from (raw, or final-norm weighted);
  // a frame taken afresh at each layer has another Gram matrix at each, so the final layer's cosines do not place it
  const cw = shown.fnw ? 1 : 0;
  const cos3 = frame && !shown.perLayer
    ? ([cosOf(frame, nw[0], nw[1])[cw], cosOf(frame, nw[0], nw[2])[cw], cosOf(frame, nw[1], nw[2])[cw]] as const)
    : null;
  const ghost = cos3 && !atOnce ? ghostGeometry(cos3[0], cos3[1], cos3[2]) : null;
  const lghost = cos3 && atOnce ? lowdinGhost(cos3[0], cos3[1], cos3[2]) : null;
  const dots = useMemo(() => allDots(layers.map((x) => ({ az: x.az, el: x.el }))), [layers]);
  const rd = showReading ? readingGeometry(az as number, el as number) : null;
  const azR = az !== null ? Math.round(az) : null;
  const elR = el !== null ? Math.round(el) : null;
  let lean = "";
  if (rd) {
    let mi = 0;
    rd.comps.forEach((v, i) => {
      if (Math.abs(v) > Math.abs(rd.comps[mi])) mi = i;
    });
    lean = (rd.comps[mi] >= 0 ? "toward " : "away from ") + SN[mi];
  }
  const diagAria =
    (atOnce
      ? `The survey frame, squared all at once: the axis nearest ${lab[0]} sets bearing 0°, the axis nearest ${lab[1]} 90°, and the axis nearest ${lab[2]} points straight up.`
      : `The survey frame: ${lab[0]} sets bearing 0°, ${lab[1]} bearing 90°, and ${lab[2]} points straight up.`) +
    (lghost
      ? ` Dashed rays show where the three words truly point, each a little off its own axis; the first two stand ${lghost.angle}° apart.`
      : ghost
        ? ` Dashed rays show where the second and third words truly point before squaring; the first two stand ${ghost.angle}° apart.`
        : "") +
    (showReading && azR !== null && elR !== null
      ? ` At layer ${li + 1} the stream, as the ${lens.name} lens reads it, points at bearing ${sgn(azR)}°, elevation ${sgn(elR)}°.`
      : " No reading yet for these landmarks.");
  const kind = kindOf(L?.isGlobal ?? li % 6 === 5);
  const pfs = layers.map((x) => x.projFrac).filter((v): v is number => v !== null);
  const pfRange = pfs.length ? `${(Math.min(...pfs) * 100).toFixed(1)}–${(Math.max(...pfs) * 100).toFixed(1)}%` : null;
  const pfRangeWords = pfs.length
    ? `between ${(Math.min(...pfs) * 100).toFixed(1)} and ${(Math.max(...pfs) * 100).toFixed(1)} percent`
    : null;
  // a share of the lens-read state, or of the raw stream when each layer has its own frame (lib/frame.tsx)
  const caughtWords = frameDesc && (frameDesc.raw || frameDesc.centred) ? frameDesc.denomWords : "of that reading";

  /* ---- iii · how far apart they stand */
  const rayD = (t: number) => "M10,70 L" + f1(10 + 58 * Math.cos(t)) + "," + f1(70 - 58 * Math.sin(t));
  const pairs = PAIRS.map(([a, b, which]) => {
    const names = `${lab[a]} and ${lab[b]}`;
    if (!frame) return { names, which, known: false as const };
    const c = cosOf(frame, nw[a], nw[b]);
    const t = Math.acos(c[0]);
    const tw = Math.acos(c[1]);
    return {
      names,
      which,
      known: true as const,
      arc: "M32,70 A22,22 0 0,0 " + f1(10 + 22 * Math.cos(t)) + "," + f1(70 - 22 * Math.sin(t)),
      rawRay: rayD(t),
      wRay: rayD(tw),
      angle: Math.round(t / RAD) + "°",
      cosText: "cosine " + c[0].toFixed(2),
      weighed: Math.round(tw / RAD) + "° as the model weighs them (cosine " + c[1].toFixed(2) + ")",
    };
  });

  /* ---- v · bearing through the layers */
  const chart = useMemo(
    () => chartGeometry(layers.map((x) => x.az), layers.map((x) => x.el)),
    [layers],
  );

  /* ---- vi · how close each came to being said */
  const rows = useMemo(() => {
    if (!sf || sf.kind !== "words") return [];
    return sf.words.map((w) => {
      const lens = layers.map((x) => rankOf(x, w)?.lens ?? null);
      const eye = layers.map((x) => rankOf(x, w)?.eye ?? null);
      let best: number | null = null;
      let bi = -1;
      lens.forEach((v, i) => {
        if (v !== null && (best === null || v < best)) {
          best = v;
          bi = i;
        }
      });
      let eyeBest: number | null = null;
      let ebi = -1;
      eye.forEach((v, i) => {
        if (v !== null && (eyeBest === null || v < eyeBest)) {
          eyeBest = v;
          ebi = i;
        }
      });
      return { w, lens, eye, best: best as number | null, bi, eyeBest: eyeBest as number | null, ebi };
    });
  }, [sf, layers]);
  const anyRanks = rows.some((r) => r.best !== null);
  const eyeKnown = rows.length > 0 && rows.every((r) => r.eyeBest !== null);
  const lastIdx = layers.length - 1;
  const L60: LayerReading | null = layers[lastIdx] ?? null;
  const top60 = L60?.lensTop?.[0] ?? null;
  const lastRanked = rows
    .map((r) => ({ w: r.w, last: r.lens[lastIdx] ?? null }))
    .filter((r): r is { w: string; last: number } => r.last !== null);
  const nearest = lastRanked.length ? lastRanked.reduce((a, b) => (b.last < a.last ? b : a)) : null;
  const furthest = lastRanked.length ? lastRanked.reduce((a, b) => (b.last > a.last ? b : a)) : null;
  const studySet = sf?.kind === "words" && sf.norm.slice().sort().join("|") === "he|it|she";

  /* ---- apparatus · the request this page sends: the store's own payload, how it writes included */
  const payload = sweepPayload(reqState, question, answerBegins, steering);
  const request = JSON.stringify({ action: "sweep", payload }, null, 2);
  const requestNote = isWords
    ? "The four frame options above go as frame_squaring, frame_per_layer, frame_fnw and centre_token_ids (the kept control’s exact prompt ids), with centre_answer, which centres each later word-piece on the control reading this very answer. How it writes (max_new_tokens, sampler, sight_every_token) is set under the question on the instrument page; a request_id is added when it is sent."
    : lp
      ? `locked_pole takes [layer index, neuron, polarity]: layer ${lp.idx + 1} is index ${lp.idx}. frame_squaring carries how the neuron’s frame is squared.`
      : `locked_pole takes [layer index, neuron, polarity]. Without it the server quietly falls back to the word frame, here ${words.map((w) => q(w.trim())).join(" · ")}; its own default is “spider” · “eight” · “insect”.`;

  /* ---- apparatus · layer N */
  const xyz = L?.xyz ?? null;
  const pn = xyz ? Math.hypot(xyz[0], xyz[1], xyz[2]) : null;
  const isFixture = sighting?.source === "fixture";
  const rawRec = L
    ? {
        layer: li + 1,
        index: li,
        survey_labels: sighting?.landmarks ?? null,
        token_index: sighting?.tokenIndex ?? null,
        read_at: sighting?.readAt ?? null,
        read_pos: sighting?.readPos ?? null,
        chosen: chosen ? { token_id: chosen.id, token: chosen.token } : null,
        centre_kind: sighting?.centreKind ?? null,
        xyz,
        proj_norm: pn !== null ? Number(pn.toPrecision(3)) : null,
        [frameDesc?.exportKey ?? "proj_frac_of_hJ"]: L.projFrac,
        azimuth_deg: az,
        elevation_deg: el,
        ...(isFixture ? { single_bos: { azimuth_deg: SINGLE_MARK_AZ[li], elevation_deg: SINGLE_MARK_EL[li] } } : {}),
        jpp_tracked_rank: Object.fromEntries(rows.map((r) => [r.w, r.lens[li]])),
        naive_tracked_rank: Object.fromEntries(rows.map((r) => [r.w, r.eye[li]])),
        rank_from_next: L.rankFromNext,
        residual_norm: L.streamStrength ?? "[MEASURED IN A LIVE SIGHTING]",
      }
    : null;

  /* ---- apparatus · angles and squareness */
  const sqCosines = frame
    ? PAIRS.map(([a, b]) => {
        const c = cosOf(frame, nw[a], nw[b]);
        return `${q(nw[a])} · ${q(nw[b])} ${c[0].toFixed(4)} raw, ${c[1].toFixed(4)} weighted`;
      }).join("; ")
    : "";

  const rankNode = (v: number | null | undefined) =>
    v === null || v === undefined ? <Missing what={NOT_HERE} /> : comma(v);

  return (
    <Page route={route} folio="folio iv">
      {/* ------------------------------------------------ i · three landmarks */}
      <section
        aria-labelledby="lm-head"
        style={{ display: "flex", flexDirection: "column", gap: "24px", padding: "34px 0 32px" }}
      >
        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: "12px 32px" }}>
          <h2 className="h-sec" id="lm-head">
            i · three landmarks
          </h2>
          <fieldset style={{ position: "relative", border: "0", margin: "0", padding: "0", display: "flex", flexWrap: "wrap", gap: "8px" }}>
            <legend className="visually-hidden">Where the landmarks come from</legend>
            <button type="button" onClick={pickWords} aria-pressed={isWords} style={modeBtn(isWords)}>
              landmarks from words
            </button>
            <button type="button" onClick={pickNeuron} aria-pressed={!isWords} style={modeBtn(!isWords)}>
              landmarks from a neuron
            </button>
          </fieldset>
        </div>
        <p style={{ display: "flow-root", margin: "0", fontSize: "28px", lineHeight: 1.4, maxWidth: "1000px", color: "#2B2117" }}>
          <span
            aria-hidden="true"
            style={{ float: "left", fontSize: "98px", lineHeight: 0.8, fontWeight: 500, color: "#8A6A2C", margin: "7px 14px 0 0" }}
          >
            A
          </span>
          <span className="visually-hidden">A</span> surveyor fixes a position by sighting known landmarks. Here the
          landmarks are three directions in the model’s inner space, most often named by words, and at each of its sixty
          layers the instrument measures where the stream points against them, as the {lens.name} lens reads it: its bearing
          round the horizon, and its elevation toward the third.
        </p>

        {isWords ? (
          <div style={{ display: "flex", flexDirection: "column", gap: "26px" }}>
            <Rubric style={{ maxWidth: "760px" }}>
              {atOnce
                ? "Write one word on each stone, or begin from a set below, then take a sighting. Choose three words of a kind. The three are squared all at once, each axis as near its own word as the others allow, so no word takes another’s share. The three axes are the same whatever their order, but the order still says which word sets the bearing’s 0°, which its 90° and which points up: reorder the stones and the bearing and elevation are relabelled, though the stream points where it did."
                : "Write one word on each stone, or begin from a set below, then take a sighting. Choose three words of a kind, and mind their order: the first stands exactly where it lies, and the other two are squared against it."}
              {shown.own && shown.squaring !== nextSquaring
                ? nextSquaring === "lowdin"
                  ? " That is how the sighting shown was squared; the next will be squared all at once, as set in the apparatus below."
                  : " That is how the sighting shown was squared; the next will be squared in order, as set in the apparatus below."
                : ""}
            </Rubric>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "30px 56px" }}>
              {roles(shown.squaring).map((r, i) => {
                const shown = words[i].trim() || "this landmark";
                return (
                  <div key={i} style={{ flex: "1 1 280px", minWidth: "0", display: "flex", gap: "18px", alignItems: "flex-start" }}>
                    <Stone numeral={r.numeral} hatched />
                    <div style={{ flex: "1 1 auto", minWidth: "0", display: "flex", flexDirection: "column", gap: "4px" }}>
                      <label htmlFor={`landmark-${i + 1}`} style={smallCaps}>
                        {r.ordinal}
                      </label>
                      <input
                        id={`landmark-${i + 1}`}
                        type="text"
                        value={words[i]}
                        onChange={(e) => setWord(i, e.target.value)}
                        autoComplete="off"
                        spellCheck={false}
                        style={{
                          fontFamily: "inherit",
                          fontStyle: "italic",
                          fontSize: "34px",
                          lineHeight: 1.2,
                          color: "#2B2117",
                          background: "transparent",
                          border: "0",
                          borderBottom: "1px solid #C9BEAC",
                          padding: "2px 0 6px",
                          width: "100%",
                          boxSizing: "border-box",
                        }}
                      />
                      <span style={{ fontSize: "17px", color: "#2B2117", fontVariantNumeric: "lining-nums", marginTop: "4px" }}>
                        {r.sets}
                      </span>
                      <span style={{ fontSize: "15px", fontStyle: "italic", color: "#6B5D4D" }}>{r.how}</span>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", marginTop: "6px" }}>
                        <button
                          type="button"
                          onClick={i === 0 ? undefined : () => swap(i, i - 1)}
                          disabled={i === 0}
                          aria-label={`Move ${q(shown)} earlier`}
                          style={moveBtn(i === 0)}
                        >
                          ‹ earlier
                        </button>
                        <button
                          type="button"
                          onClick={i === 2 ? undefined : () => swap(i, i + 1)}
                          disabled={i === 2}
                          aria-label={`Move ${q(shown)} later`}
                          style={moveBtn(i === 2)}
                        >
                          later ›
                        </button>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
            <fieldset style={{ border: "0", margin: "0", padding: "0", display: "flex", flexWrap: "wrap", alignItems: "center", gap: "10px 14px" }}>
              <legend style={{ float: "left", padding: "0", marginRight: "10px", ...smallCaps }}>or begin from a set</legend>
              {PRESETS.map((p) => {
                const on = nw.join("|") === p.words.join("|");
                return (
                  <button
                    key={p.words.join("|")}
                    type="button"
                    onClick={() => {
                      setLandmarks(p.words.slice() as Triple);
                      setSurveyMode("monument", keepPole);
                    }}
                    aria-pressed={on}
                    style={{
                      fontFamily: "inherit",
                      color: "#2B2117",
                      background: on ? "#EDE5D3" : "transparent",
                      border: `1px solid ${on ? "#2B2117" : "#C9BEAC"}`,
                      borderRadius: "2px",
                      padding: "7px 18px 9px",
                      minHeight: "48px",
                      cursor: "pointer",
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "flex-start",
                      textAlign: "left",
                    }}
                  >
                    <span style={{ fontSize: "20px", fontStyle: "italic", lineHeight: 1.25 }}>{p.words.join(" · ")}</span>
                    <span style={{ fontSize: "14px", color: "#6B5D4D", lineHeight: 1.3 }}>{p.note}</span>
                  </button>
                );
              })}
            </fieldset>
            <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "10px 22px" }}>
              <button
                className="btn-primary"
                type="button"
                onClick={survey}
                disabled={isSweeping || anyEmpty}
                title={anyEmpty ? "Write a word on each of the three stones first" : undefined}
              >
                {isSweeping ? "taking a sighting…" : "take a sighting with these landmarks"}
              </button>
              <span style={{ fontSize: "16px", color: "#6B5D4D" }}>{readsAt}</span>
              <a href={href.instrument()} style={{ fontSize: "17px", padding: "12px 4px" }}>
                change the question
              </a>
            </div>
            <SweepError />
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: "26px" }}>
            <Rubric style={{ maxWidth: "760px" }}>
              Open a neuron from any layer of the instrument, and its own three directions become the landmarks: what
              wakes it, what it writes, and the later neuron that listens to it most. Choose from layers 41 to 58 if you
              can; outside them, the study found neuron labels unreliable.
            </Rubric>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "30px 56px" }}>
              <NeuronStone
                numeral="I"
                ordinal="first landmark"
                name="what wakes it"
                value={lp ? poleLabel(lp) : "[NO NEURON CHOSEN]"}
                how="its gate, the direction that makes it fire"
              />
              <NeuronStone
                numeral="II"
                ordinal="second landmark"
                name="what it writes"
                value={lp ? poleLabel(lp) : "[NO NEURON CHOSEN]"}
                how="what it adds to the stream, read through the J++ lens"
              />
              <NeuronStone
                numeral="III"
                ordinal="third landmark"
                name="who listens next"
                value={lp ? reader ?? "[FOUND WHEN YOU TAKE A SIGHTING]" : "[NO NEURON CHOSEN]"}
                how="along its strongest thread, the later neuron that reads it"
              />
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "10px 22px" }}>
              {lp ? (
                <>
                  <button className="btn-primary" type="button" onClick={survey} disabled={isSweeping}>
                    {isSweeping ? "taking a sighting…" : "take a sighting with these landmarks"}
                  </button>
                  <a href={neuronHref} style={{ fontSize: "17px", padding: "12px 4px" }}>
                    open this neuron
                  </a>
                  <a href={href.instrument()} style={{ fontSize: "17px", padding: "12px 4px" }}>
                    choose another neuron
                  </a>
                </>
              ) : (
                <a
                  href={neuronHref}
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    fontSize: "19px",
                    fontVariantCaps: "all-small-caps",
                    letterSpacing: "0.16em",
                    background: "#8F2B1E",
                    color: "#F6F0E3",
                    textDecoration: "none",
                    borderRadius: "2px",
                    padding: "12px 26px",
                    minHeight: "48px",
                    boxSizing: "border-box",
                  }}
                >
                  choose a neuron
                </a>
              )}
              <span style={{ fontSize: "16px", color: "#6B5D4D" }}>
                {atOnce
                  ? "the neuron’s frame is squared all at once, as chosen below: what wakes it names the first axis"
                  : "the neuron’s frame is squared in order: what wakes it comes first, kept exactly as it lies"}
              </span>
            </div>
            {lp ? <SweepError /> : null}
          </div>
        )}
      </section>

      {/* ------------------------------------------- apparatus · how the frame is built */}
      <Apparatus title="how the frame is built" label="Apparatus for how the frame is built" style={{ padding: "12px 0 26px", gap: "10px" }}>
        <dl className="dl">
          <dt>from words</dt>
          <dd>
            {wordFrame.squaring === "lowdin" ? (
              <>
                e<sub>1</sub>, e<sub>2</sub>, e<sub>3</sub> = Löwdin({wordRows(1)}, {wordRows(2)}, {wordRows(3)}) = (V V<sup>⊤</sup>)
                <sup>−½</sup> V on the unit rows V, in fp32 [33]: what the interface sends unless you choose otherwise, on{" "}
                {FNW_DEFAULT ? "the weighted rows" : "the raw rows"} by default. As first built, and in the server’s own default,
                modified Gram–Schmidt on the raw rows W<sub>U</sub>[t].
              </>
            ) : (
              <>
                e<sub>1</sub>, e<sub>2</sub>, e<sub>3</sub> = Gram–Schmidt({wordRows(1)}, {wordRows(2)}, {wordRows(3)}), modified
                Gram–Schmidt in fp32: the frame as first built and the server’s own default, on the raw rows (the “monument
                mode” of ARCHITECTURE.md §3.3, module 4). The interface squares all at once (Löwdin [33]) unless you choose
                otherwise.
              </>
            )}{" "}
            Each word is read as one word-piece, the spaced form when that is a single piece:{" "}
            {known ? nw.map((w) => "“ " + w + "”").join(" · ") : <Missing what="resolved in a live sighting" />}.
          </dd>
          <dt>from a neuron</dt>
          <dd>
            v<sub>1</sub> = w<sub>gate</sub>
            <sup>(ℓ, i)</sup>, v<sub>2</sub> = (σ · w<sub>down</sub>
            <sup>(ℓ, i)</sup>) · J<sub>ℓ</sub>
            <sup>⊤</sup>, v<sub>3</sub> = w<sub>gate</sub>
            <sup>(ℓ′, j*)</sup>,{" "}
            {neuronFrameSq === "lowdin" ? (
              <>
                squared all at once (Löwdin [33]), as you chose, so axis 1 is no longer exactly w<sub>gate</sub>
              </>
            ) : (
              <>
                squared in that order (Gram–Schmidt), so axis 1 is exactly w<sub>gate</sub>
              </>
            )}
            : the “pole-locked motif mode”. A neuron frame keeps Gram–Schmidt by default: its three axes have fixed roles, so
            their order is not a choice and the order fault does not arise. σ = ±1 names which of the neuron’s two poles is
            locked (after Cammarata 2026 [9]); j* is the dominant downstream reader, in the code the pole’s first listed
            thread, up to six layers on, or failing that the first listed neuron of the next layer (its loudest, or, ranked by
            write size, its biggest writer; engine_runtime.py). No pole-locked sighting is recorded in the study: <Missing />
          </dd>
          <dt>the reading</dt>
          <dd>
            h̃<sub>ℓ</sub> = h<sub>ℓ</sub> · J<sub>ℓ</sub>
            <sup>⊤</sup>; (x, y, z) = (⟨h̃<sub>ℓ</sub>, e<sub>1</sub>⟩, ⟨h̃<sub>ℓ</sub>, e<sub>2</sub>⟩, ⟨h̃
            <sub>ℓ</sub>, e<sub>3</sub>⟩); bearing θ = atan2(y, x), elevation φ = asin(z / ‖p‖), with ‖p‖ = √(x² + y² +
            z²). Stream strength ‖h<sub>ℓ</sub>‖ is the length of the raw state, before the lens.
          </dd>
          <dt>the lens</dt>
          <dd>
            J++ lens (Ayonrinde &amp; Lindsey 2026) [4], a refinement of the Jacobian lens (Gurnee et al. 2026) [2]. It was
            fitted on the base model and is applied here to the instruct model (REPORT §3.4, bug 9).
          </dd>
          <LensRecord sighting={sighting} />
          <dt>where it is taken</dt>
          <dd>
            As built, and by default, the word frame is taken at the last layer, and every layer’s h̃<sub>ℓ</sub> is read in it. A frame
            squared afresh at each layer, v<sub>m</sub>
            <sup>(ℓ)</sup> = W<sub>U</sub>[t<sub>m</sub>] · J<sub>ℓ</sub>, reads the raw h<sub>ℓ</sub> instead, since ⟨h<sub>ℓ</sub>, u J
            <sub>ℓ</sub>⟩ = ⟨h<sub>ℓ</sub> J<sub>ℓ</sub>
            <sup>⊤</sup>, u⟩ (<span className="mono">build_monument_frame(pullback_per_layer=True)</span>,{" "}
            <span className="mono">frame_per_layer</span> in the request); the fraction caught is then a fraction of ‖h<sub>ℓ</sub>‖.
            {shown.perLayer
              ? " Each layer has its own frame, so the final layer’s cosines do not place the words, and the dashed rays where the words truly point are not drawn."
              : ""}
          </dd>
          <dt>weights and centre</dt>
          <dd>
            Weighted rows: fnw ⊙ W<sub>U</sub>[t], the directions the model reads just before speaking (logit<sub>t</sub> = ⟨h / rms(h), fnw ⊙ W
            <sub>U</sub>[t]⟩); the rms is one number per layer, so it moves no angle and no fraction. Centring subtracts a matched control’s state
            from every layer’s state before it is read (REPORT §3.5): at the first word-piece, the control’s state where it chose its own
            first word-piece (the last of its prompt, sent as its exact ids); at each later word-piece, the control’s state at the same
            word-piece, reading your sighting’s own answer (its prompt, then this answer’s word-pieces up to there,{" "}
            <span className="mono">centre_answer</span>), so the two differ only in what was asked. The control is read without a nudge.
            {sighting?.source === "live" ? (
              <> The sighting shown: {centreWords(sighting) ?? <Missing what="not sent" />}.</>
            ) : null}
          </dd>
          <dt>known fault</dt>
          <dd>
            Bug 3, high (REPORT §3.4): as first built, the frame changed with the order of the words, and was built from the
            raw rows of W<sub>U</sub>, not the rows the model weighs, fnw · W<sub>U</sub>. The interface now sends the mend for
            the order unless you choose otherwise: squared all at once (Löwdin 1950 [33]),{" "}
            {FNW_DEFAULT
              ? "on the weighted rows, which V7 did not measure"
              : "on the raw rows, where V7 measured it; the weighted rows are an option below, which V7 did not measure"}
            . The server’s own default stays the frame as first built, for old callers. Every sighting shows the fraction of
            the state its frame catches. Section iv below has the evidence.
          </dd>
        </dl>
        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "8px 12px", paddingTop: "8px", borderTop: "1px solid #E3DACA" }}>
          <button type="button" className="btn-ghost" aria-pressed={onAsBuilt} onClick={() => setFramePreset(FRAME_AS_BUILT)}>
            the frame options as first built
          </button>
          <button type="button" className="btn-ghost" aria-pressed={onMended} onClick={() => setFramePreset(FRAME_MENDED)}>
            the default frame
          </button>
          <span style={{ flex: "1 1 360px", minWidth: "0", fontSize: "15px", color: "#6B5D4D" }}>
            To set a bearing beside V7’s frame figures, take it in the frame as first built: V7 took them with one start
            mark, as the instrument now sends. Beside E6’s sightings or the recorded one, that is needed but not enough:
            they were taken before the V7 fixes, their chat prompts with a doubled start mark (fault 6), which by itself
            moved bearings by a median of 0.4–1.9° in V7.
          </span>
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 40px" }}>
          <fieldset style={optFieldset}>
            <legend style={optLegend}>how the frame is squared{isWords ? "" : " (a neuron’s frame)"}</legend>
            <label style={optLabel(true)}>
              <input
                type="radio"
                name="frame-squaring"
                checked={nextSquaring === "gram_schmidt"}
                onChange={() => (isWords ? setFrame({ squaring: "gram_schmidt" }) : setNeuronSquaring("gram_schmidt"))}
                style={optInput}
              />
              <span>
                in order <span style={aside}>(Gram–Schmidt, as built; the default for a neuron frame)</span>
              </span>
            </label>
            <label style={optLabel(true)}>
              <input
                type="radio"
                name="frame-squaring"
                checked={nextSquaring === "lowdin"}
                onChange={() => (isWords ? setFrame({ squaring: "lowdin" }) : setNeuronSquaring("lowdin"))}
                style={optInput}
              />
              <span>
                all at once <span style={aside}>(Löwdin [33], the default for words)</span>
              </span>
            </label>
          </fieldset>
          <fieldset style={optFieldset}>
            <legend style={optLegend}>where the frame is taken</legend>
            <label style={optLabel(true)}>
              <input type="radio" name="frame-place" checked={!frameOpts.perLayer || !isWords} onChange={() => setFrame({ perLayer: false })} style={optInput} />
              <span>
                from the final layer <span style={aside}>(as built, and the default)</span>
              </span>
            </label>
            <label style={optLabel(isWords)}>
              <input
                type="radio"
                name="frame-place"
                checked={frameOpts.perLayer && isWords}
                disabled={!isWords}
                onChange={() => setFrame({ perLayer: true })}
                aria-describedby="frame-note"
                style={optInput}
              />
              <span>
                from the layer’s own lens{" "}
                <span style={aside}>
                  (pulled back by J<sub>ℓ</sub>)
                </span>
              </span>
            </label>
          </fieldset>
          <fieldset style={optFieldset}>
            <legend style={optLegend}>weights and centre</legend>
            <label style={optLabel(isWords)}>
              <input
                type="checkbox"
                checked={frameOpts.fnw && isWords}
                disabled={!isWords}
                onChange={(e) => setFrame({ fnw: e.target.checked })}
                aria-describedby="frame-note"
                style={optInput}
              />
              <span>
                weigh the words as the model does{" "}
                <span style={aside}>
                  (final-norm weighted rows, fnw·W<sub>U</sub>)
                </span>
              </span>
            </label>
            <label style={optLabel(canCentre)}>
              <input
                type="checkbox"
                checked={frameOpts.centre && canCentre}
                disabled={!canCentre}
                onChange={(e) => setFrame({ centre: e.target.checked })}
                aria-describedby="frame-note"
                style={optInput}
              />
              <span>
                centre on a matched control <span style={aside}>(subtract its mean; REPORT §3.5)</span>
              </span>
            </label>
          </fieldset>
        </div>
        <p id="frame-note" style={{ margin: 0, maxWidth: "900px", fontSize: "15px", color: "#6B5D4D" }}>
          {isWords ? "" : "The final-norm weights and the per-layer lens apply to a frame of words, not to a neuron’s frame. "}
          {isWords && frameOpts.fnw ? "V7 did not measure a frame on the weighted rows. " : ""}
          {canCentre
            ? "Centring reads against the matched control kept on the instrument’s first page: the first word-piece against the control where it chose its own first, each later one against the control reading the same answer up to there. "
            : controlIsThis
              ? "The control kept is this very question; centring on it would subtract its state from itself and leave nothing to read. Keep a different sighting as the control to centre. "
              : "To centre, take and keep a matched control first (the question’s apparatus on the instrument page). "}
          The options go with the next sighting.{" "}
          {sighting?.source === "live"
            ? `The sighting shown was squared ${frameWords(sighting.frame)}${
                (sighting.frame?.nCentre ?? 0) > 0 && centreWords(sighting) ? ` (at this word-piece: ${centreWords(sighting)})` : ""
              }.`
            : "The recorded sighting was squared in order, from raw rows, at the final layer."}
        </p>
        <div style={{ display: "flex", flexDirection: "column", gap: "4px", paddingTop: "4px" }}>
          <span style={{ fontVariantCaps: "all-small-caps", letterSpacing: "0.14em", color: "#6B5D4D" }}>
            the request this page sends
          </span>
          <pre
            style={{
              margin: "0",
              padding: "8px 0",
              borderTop: "1px solid #E3DACA",
              borderBottom: "1px solid #E3DACA",
              fontFamily: "'IBM Plex Mono', ui-monospace, monospace",
              fontSize: "12.5px",
              lineHeight: 1.5,
              color: "#2B2117",
              whiteSpace: "pre-wrap",
              overflowX: "auto",
            }}
          >
            {request}
          </pre>
          <span style={{ color: "#6B5D4D", maxWidth: "900px" }}>
            Sent as JSON over the WebSocket <span className="mono">/ws/theodolite</span> (backend/server.py,
            SweepRequest). {requestNote}
          </span>
        </div>
      </Apparatus>

      {/* ------------------------------------------------ the sighting shown */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          alignItems: "baseline",
          gap: "8px 22px",
          padding: "16px 0 18px",
          borderTop: "1px solid #E3DACA",
          borderBottom: "1px solid #E3DACA",
        }}
      >
        <span style={smallCaps}>sighting shown</span>
        <span style={{ fontSize: "26px", fontStyle: "italic" }}>
          {!sf ? <Missing what="no sighting yet" /> : sf.kind === "neuron" && sf.pole ? poleLabel(sf.pole) : SN.join(" · ")}
        </span>
        {sighting ? (
          <span style={{ fontSize: "16px", color: "#6B5D4D" }}>
            {sighting.readAt !== null ? readFact(sighting) : <>read at <Missing what={NOT_HERE} /></>}
            {chosenPhrase(sighting) ? `, ${chosenPhrase(sighting)}` : ""}
            {shownSet?.pieces && shownSet.pieces.length > 1 ? ` · word-piece ${k + 1} of ${shownSet.pieces.length}` : ""} · you asked “
            {sighting.question}”
          </span>
        ) : null}
        {sighting ? (
          <span style={{ marginLeft: "auto", fontSize: "15px", color: "#6B5D4D", fontVariantNumeric: "lining-nums tabular-nums" }}>
            {sighting.timing?.analyseMs != null ? (
              <>
                {layers.length} layers analysed in {ms(sighting.timing.analyseMs)}
                {sighting.timing.totalMs != null ? `, ${ms(sighting.timing.totalMs)} after it was asked` : ""}
              </>
            ) : (
              <>
                {layers.length} layers surveyed in {sighting.elapsedMs !== null ? ms(sighting.elapsedMs) : <Missing what={NOT_HERE} />}
              </>
            )}{" "}
            ·{" "}
            {lensLabel(lens)}
            {sighting.source === "fixture" ? " · the study’s recorded sighting, taken through the same lens" : ""}
            {sighting.isNudged ? " · nudged" : ""}
          </span>
        ) : null}
      </div>
      {!matches ? (
        <p role="status" style={{ margin: "14px 0 0", fontStyle: "italic", fontSize: "17px", color: "#8F2B1E", maxWidth: "900px" }}>
          {notice}
        </p>
      ) : null}

      <div style={{ display: "flex", flexWrap: "wrap", gap: "44px 64px", paddingTop: "34px" }}>
        {/* ------------------------------------------- ii · the frame they make */}
        <section aria-labelledby="frame-head" style={{ flex: "999 1 640px", minWidth: "0", display: "flex", flexDirection: "column", gap: "14px" }}>
          <div className="row-between">
            <h2 className="h-sec" id="frame-head">
              ii · the frame they make
            </h2>
            <span style={{ fontSize: "15px", color: "#6B5D4D", fontVariantNumeric: "lining-nums tabular-nums" }}>
              layer {li + 1} of sixty
            </span>
          </div>
          <figure style={{ margin: "0" }}>
            <svg
              viewBox="0 0 700 540"
              role="img"
              aria-label={diagAria}
              style={{ width: "100%", maxWidth: "780px", height: "auto", display: "block", margin: "0 auto" }}
            >
              <path d={PLATE.merA} fill="none" stroke="#A07A2E" strokeOpacity="0.45" strokeWidth="0.8" />
              <path d={PLATE.merB} fill="none" stroke="#A07A2E" strokeOpacity="0.45" strokeWidth="0.8" />
              <path d={PLATE.par} fill="none" stroke="#A07A2E" strokeOpacity="0.45" strokeWidth="0.7" strokeDasharray="1 3" />
              <path d={PLATE.back} fill="none" stroke="#2B2117" strokeOpacity="0.25" strokeWidth="0.8" strokeDasharray="2 4" />
              <ellipse cx="320" cy="300" rx="240" ry="78" fill="none" stroke="#A07A2E" strokeWidth="1.2" />
              <ellipse cx="320" cy="300" rx="211" ry="68.6" fill="none" stroke="#A07A2E" strokeOpacity="0.55" strokeWidth="0.8" />
              <path d={PLATE.ticks} fill="none" stroke="#A07A2E" strokeWidth="0.9" />
              <g fontFamily="EB Garamond, Garamond, serif" fontSize="13" fill="#6B5D4D" style={{ fontVariantNumeric: "lining-nums tabular-nums" }}>
                <text x="100" y="258" textAnchor="end">
                  180°
                </text>
                <text x="442" y="225">
                  −90°
                </text>
              </g>
              {ghost ? (
                <g>
                  <path d={ghost.rays} fill="none" stroke="#7A5C28" strokeWidth="1.1" strokeDasharray="4 4" />
                  <path d={ghost.arc} fill="none" stroke="#7A5C28" strokeWidth="0.9" />
                  <path d={ghost.tips} fill="#F6F0E3" stroke="#7A5C28" strokeWidth="1" />
                  <g fontFamily="EB Garamond, Garamond, serif" fontSize="14" fontStyle="italic" fill="#7A5C28">
                    <text x={ghost.bx} y={ghost.by}>
                      true {q(words[1].trim())}
                    </text>
                    <text x={ghost.cx} y={ghost.cy}>
                      true {q(words[2].trim())}
                    </text>
                    <text x={ghost.ax} y={ghost.ay} fontStyle="normal" textAnchor="middle" style={{ fontVariantNumeric: "lining-nums" }}>
                      {ghost.angle}°
                    </text>
                  </g>
                </g>
              ) : lghost ? (
                <g>
                  <path d={lghost.rays} fill="none" stroke="#7A5C28" strokeWidth="1.1" strokeDasharray="4 4" />
                  <path d={lghost.arc} fill="none" stroke="#7A5C28" strokeWidth="0.9" />
                  <path d={lghost.tips} fill="#F6F0E3" stroke="#7A5C28" strokeWidth="1" />
                  <g fontFamily="EB Garamond, Garamond, serif" fontSize="14" fontStyle="italic" fill="#7A5C28">
                    {lghost.labels.map(([x, y], m) => (
                      <text key={m} x={x} y={y}>
                        true {q(words[m].trim())}
                      </text>
                    ))}
                    <text x={lghost.ax} y={lghost.ay} fontStyle="normal" textAnchor="middle" style={{ fontVariantNumeric: "lining-nums" }}>
                      {lghost.angle}°
                    </text>
                  </g>
                </g>
              ) : null}
              <path d={PLATE.axes} fill="none" stroke="#2B2117" strokeWidth="1.4" />
              <path d={PLATE.tips} fill="#A07A2E" />
              <circle cx="320" cy="300" r="3" fill="#2B2117" />
              <g fontFamily="EB Garamond, Garamond, serif">
                <text x="590" y="344" fontSize="21" fontStyle="italic" fill="#2B2117">
                  {lab[0]}
                </text>
                <text x="590" y="364" fontSize="13" fill="#6B5D4D" style={{ fontVariantNumeric: "lining-nums" }}>
                  first · 0°
                </text>
                <text x="162" y="394" textAnchor="end" fontSize="21" fontStyle="italic" fill="#2B2117">
                  {lab[1]}
                </text>
                <text x="162" y="413" textAnchor="end" fontSize="13" fill="#6B5D4D" style={{ fontVariantNumeric: "lining-nums" }}>
                  second · 90°
                </text>
                <text x="338" y="58" fontSize="21" fontStyle="italic" fill="#2B2117">
                  {lab[2]}
                </text>
                <text x="338" y="77" fontSize="13" fill="#6B5D4D">
                  third · straight up
                </text>
              </g>
              {rd && azR !== null && elR !== null ? (
                <g>
                  <path d={dots} fill="#2B2117" fillOpacity="0.3" />
                  <path d={rd.ray} fill="none" stroke="#2B2117" strokeWidth="1.3" />
                  <path d={rd.drop} fill="none" stroke="#2B2117" strokeOpacity="0.55" strokeWidth="0.9" strokeDasharray="2 3" />
                  <path d={rd.bArc} fill="none" stroke="#A07A2E" strokeWidth="2" />
                  <path d={rd.eArc} fill="none" stroke="#2E4A7D" strokeWidth="2" />
                  <path d={rd.index} fill="none" stroke="#8F2B1E" strokeWidth="2" />
                  <g fontFamily="EB Garamond, Garamond, serif" fontSize="14" style={{ fontVariantNumeric: "lining-nums" }}>
                    <text x={rd.bx} y={rd.by} textAnchor="middle" fill="#7A5C28">
                      bearing {sgn(azR)}°
                    </text>
                    <text x={rd.ex} y={rd.ey} fill="#2E4A7D">
                      elevation {sgn(elR)}°
                    </text>
                  </g>
                  <circle cx={rd.x} cy={rd.y} r="13" fill="none" stroke="#8F2B1E" strokeOpacity="0.5" strokeWidth="1" />
                  <circle cx={rd.x} cy={rd.y} r="6.5" fill="#A07A2E" stroke="#8F2B1E" strokeWidth="1.6" />
                  <text
                    x={rd.lx}
                    y={rd.ly}
                    fontFamily="EB Garamond, Garamond, serif"
                    fontSize="15"
                    fill="#8F2B1E"
                    letterSpacing="1.5"
                    style={{ fontVariantCaps: "all-small-caps", fontVariantNumeric: "lining-nums" }}
                  >
                    layer {li + 1}
                  </text>
                </g>
              ) : null}
            </svg>
            <figcaption style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: "6px 22px", fontSize: "15px", color: "#6B5D4D", marginTop: "4px" }}>
              <span>solid: the frame the instrument draws</span>
              <span aria-hidden="true" style={{ color: "#A07A2E" }}>
                ◆
              </span>
              <span>
                {ghost || lghost
                  ? `dashed: where the words truly point${lghost ? ", each a little off its axis" : ""}`
                  : frame && shown.perLayer
                    ? "dashed rays not drawn: each layer has its own frame (apparatus above)"
                    : "dashed: where the words truly point"}
              </span>
              <span aria-hidden="true" style={{ color: "#A07A2E" }}>
                ◆
              </span>
              <span>red: the stream at the chosen layer, as the lens reads it, and its bearing on the rim</span>
            </figcaption>
          </figure>
          <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "6px 18px" }}>
            <label htmlFor="layer-pick" style={smallCaps}>
              choose a layer
            </label>
            <input
              id="layer-pick"
              type="range"
              min="1"
              max="60"
              step="1"
              value={li + 1}
              onChange={(e) => setSelectedLayer(Number(e.target.value) - 1)}
              style={{ flex: "1 1 240px", minWidth: "0", minHeight: "44px", margin: "0", accentColor: "#8F2B1E", cursor: "pointer" }}
            />
            <div style={{ display: "flex", gap: "4px" }}>
              <button type="button" onClick={() => setSelectedLayer(li - 1)} aria-label="Earlier layer" style={stepBtn}>
                ↓ earlier
              </button>
              <button type="button" onClick={() => setSelectedLayer(li + 1)} aria-label="Later layer" style={stepBtn}>
                later ↑
              </button>
            </div>
          </div>
          {showReading && azR !== null && elR !== null ? (
            <div style={{ display: "flex", flexWrap: "wrap", alignItems: "flex-end", gap: "16px 30px", borderTop: "1px solid #D9CFBD", paddingTop: "18px" }}>
              <div style={{ fontSize: "76px", lineHeight: 0.8, fontWeight: 500, color: "#8A6A2C", fontVariantNumeric: "lining-nums", letterSpacing: "-0.02em" }}>
                {li + 1}
              </div>
              <div style={{ flex: "1 1 280px", minWidth: "0", display: "flex", flexDirection: "column", gap: "2px" }}>
                <span style={{ fontVariantCaps: "all-small-caps", letterSpacing: "0.2em", fontSize: "15px", color: "#6B5D4D" }}>
                  layer {li + 1} of sixty · {kind.kind}
                </span>
                <span style={{ fontSize: "20px", fontVariantNumeric: "lining-nums tabular-nums" }}>
                  bearing {sgn(azR)}° · elevation {sgn(elR)}°
                </span>
                <span style={{ fontSize: "15px", fontStyle: "italic", color: "#6B5D4D" }}>{kind.gloss}</span>
                <span style={{ fontSize: "16px", fontStyle: "italic", color: "#6B5D4D" }}>in this frame the reading leans most {lean}</span>
              </div>
              <div style={{ flex: "1 1 220px", minWidth: "0", display: "flex", flexDirection: "column", gap: "8px", paddingBottom: "4px" }}>
                <span style={{ fontSize: "16px", color: "#6B5D4D" }}>
                  the landmarks catch{" "}
                  <span style={{ color: "#2B2117", fontVariantNumeric: "lining-nums tabular-nums" }}>
                    {L?.projFrac != null ? pct1(L.projFrac) : <Missing what={NOT_HERE} />}
                  </span>{" "}
                  {frameDesc ? frameDesc.denomWords : "of the reading"} here
                </span>
                <div aria-hidden="true" className="bar-track">
                  {L?.projFrac != null ? <div className="bar-fill" style={{ width: `${Math.min(100, L.projFrac * 100)}%` }} /> : null}
                </div>
              </div>
            </div>
          ) : (
            <p style={{ margin: "0", paddingTop: "18px", borderTop: "1px solid #D9CFBD", fontSize: "17px", color: "#6B5D4D" }}>
              [THE READING APPEARS AFTER A SIGHTING WITH THESE LANDMARKS]
            </p>
          )}
          <Rubric style={{ maxWidth: "720px" }}>
            The red point is the stream at the chosen layer, as the {lens.name} lens reads it; the faint dots are the other
            fifty-nine. It shows only a direction.{" "}
            {pfRangeWords
              ? `In this sighting the three landmarks catch ${pfRangeWords} ${caughtWords} at any layer, and the rest lies in directions they do not name.`
              : "In the study three landmarks caught only 2.4–3.9 percent of that reading, by stretch of layers, and the rest lies in directions they do not name."}{" "}
            Read a bearing as a hint, never a verdict.
          </Rubric>
          <BearingCaution sighting={sighting} style={{ maxWidth: "720px" }} />

          <Apparatus
            title={
              <>
                layer {li + 1} <span style={{ letterSpacing: "0.06em" }}>(index {li})</span>
              </>
            }
            label="Apparatus for the reading at this layer"
          >
            <dl className="dl num">
              <dt>coordinates</dt>
              <dd>
                (x, y, z) = {xyz ? "(" + xyz.map(sci).join(", ") + ")" : <Missing what={NOT_HERE} />}, in the frame of{" "}
                {SN.join(" · ")}; ‖p‖ = {pn !== null ? sci(pn) : <Missing what={NOT_HERE} />}
              </dd>
              <dt>angles</dt>
              <dd>
                θ = atan2(y, x) = {az !== null ? d1(az) + "°" : <Missing what={NOT_HERE} />}; φ = asin(z / ‖p‖) ={" "}
                {el !== null ? d1(el) + "°" : <Missing what={NOT_HERE} />}
              </dd>
              <dt>caught</dt>
              <dd>
                ‖p‖ / {frameDesc?.denom ?? (
                  <>
                    ‖h̃<sub>ℓ</sub>‖
                  </>
                )}{" "}
                = {L?.projFrac != null ? (L.projFrac * 100).toFixed(1) + "%" : <Missing what={NOT_HERE} />}{" "}
                <span className="mono">({frameDesc?.exportKey ?? "proj_frac_of_hJ"})</span>, {frameDesc?.denomWords ?? "of the lens-read state"};{" "}
                {pfRange ?? <Missing what={NOT_HERE} />} across this sighting
                {frameDesc === null || frameDesc.studyComparable
                  ? ", 2.4–3.9% by stretch of layers over V7’s 18 prompts."
                  : ". The study’s 2.4–3.9% was measured on the uncentred h·Jᵀ frame of raw rows, so it does not compare with this one."}
              </dd>
              {L?.streamStrength != null ? (
                <>
                  <dt>stream strength</dt>
                  <dd>
                    ‖h<sub>ℓ</sub>‖ = {L.streamStrength.toFixed(1)} <span className="mono">(residual_norm)</span>
                  </dd>
                </>
              ) : null}
              <dt>one start mark</dt>
              {isFixture ? (
                <dd>
                  θ = {d1(SINGLE_MARK_AZ[li])}°, φ = {d1(SINGLE_MARK_EL[li])}° when the same question begins with a single
                  start mark, ids [2, 105, 2364], as after the V7 BOS fix (REPORT §3.4, bug 6). The readings on this page
                  used the doubled mark, [{(sighting?.firstIds ?? [2, 2, 105]).join(", ")}].
                </dd>
              ) : (
                <dd>
                  Recorded only for the study’s demonstration sighting (E6, key “prepend_bos=False”, ids [2, 105, 2364]):
                  θ = {d1(SINGLE_MARK_AZ[li])}°, φ = {d1(SINGLE_MARK_EL[li])}° at this layer. Not a reading of this sighting
                  (REPORT §3.4, bug 6).
                </dd>
              )}
              {L?.attnXyz && L?.ffnXyz ? (
                <>
                  <dt>gathering · remembering</dt>
                  <dd className="num">
                    the two changes at this layer in the frame <span className="mono">(delta_attn_xyz, delta_ffn_xyz)</span>: Δ
                    <sub>attn</sub> = ({L.attnXyz.map((v) => minus(v.toExponential(2))).join(", ")}), Δ<sub>ffn</sub> = (
                    {L.ffnXyz.map((v) => minus(v.toExponential(2))).join(", ")})
                  </dd>
                </>
              ) : null}
              {L?.streamStrength != null && L?.attnXyz ? null : (
                <>
                  <dt>not recorded</dt>
                  <dd style={{ margin: "0", color: "#6B5D4D" }}>
                    {L?.streamStrength != null ? null : (
                      <>
                        stream strength ‖h<sub>ℓ</sub>‖{L?.attnXyz ? "" : ", and "}
                      </>
                    )}
                    {L?.attnXyz ? null : (
                      <>
                        the gathering and remembering arrows drawn in this frame{" "}
                        <span className="mono">(delta_attn_xyz, delta_ffn_xyz)</span>
                      </>
                    )}
                    : <Missing />
                  </dd>
                </>
              )}
              <dt>source</dt>
              {isFixture ? (
                <dd>
                  experiments/results/E6_tool_audit_02_sweep_demo.json, key “prepend_bos=True” (the single mark: key
                  “prepend_bos=False”). The question as the study set it:{" "}
                  <span className="mono">
                    lab.chat({JSON.stringify(sighting?.question ?? "")}, assistant={JSON.stringify(sighting?.answerBegins ?? "")})
                  </span>
                  , read at “{sighting?.readAt ?? ""}”, position {sighting?.readPos ?? "?"} of {sighting?.nTokens ?? "?"},
                  counting from 0.
                </dd>
              ) : sighting ? (
                <dd>
                  a live sighting from the running server (backend/server.py: the binary trajectory and{" "}
                  <span className="mono">layer_plumes[{li}]</span>). The text sent:{" "}
                  <span className="mono-wrap">{JSON.stringify(sighting.textSent ?? "")}</span>
                  {k >= 1 ? (
                    <>
                      , then its own answer up to here:{" "}
                      {sighting.answerStrs ? (
                        <span className="mono-wrap">{JSON.stringify(sighting.answerStrs.slice(0, k).join(""))}</span>
                      ) : (
                        <Missing what="not sent" />
                      )}
                    </>
                  ) : null}
                  ; read at {sighting.readAt !== null ? JSON.stringify(sighting.readAt) : <Missing what={NOT_HERE} />}
                  {sighting.readPos !== null ? `, position ${sighting.readPos} counting from 0` : ""}
                  {sighting.nTokens !== null ? `, ${comma(sighting.nTokens)} word-pieces read up to here` : ""}
                  {k >= 1 ? `, the ${ordinal(k)} word-piece of its own answer` : ""}
                  {chosen ? `; it chose ${JSON.stringify(chosen.token)} (id ${chosen.id}) there` : ""}
                  .
                </dd>
              ) : (
                <dd>
                  <Missing />
                </dd>
              )}
            </dl>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "8px 12px", paddingTop: "4px" }}>
              <button className="btn-ghost" type="button" onClick={() => setRawOpen(!rawOpen)} aria-expanded={rawOpen} disabled={!rawRec}>
                {rawOpen ? "hide the raw readout" : "show the raw readout"}
              </button>
            </div>
            {rawOpen && rawRec ? <pre className="raw">{JSON.stringify(rawRec, null, 2)}</pre> : null}
          </Apparatus>
        </section>

        {/* ------------------------------------------- iii · how far apart they stand */}
        <aside aria-labelledby="apart-head" style={{ flex: "1 1 400px", minWidth: "0", display: "flex", flexDirection: "column", gap: "20px" }}>
          <h2 className="h-sec" id="apart-head">
            iii · how far apart they stand
          </h2>
          <p style={{ margin: "0", fontSize: "18px", lineHeight: 1.5 }}>
            A theodolite wants its landmarks well apart, and 90° would be a perfect right angle. The smaller the angle
            between two words, the more they share,{" "}
            {atOnce
              ? "and the further each word stands from its own axis once the three are squared."
              : "and the more the first takes from the second."}
          </p>
          <div style={{ display: "flex", flexDirection: "column" }}>
            {pairs.map((p) => (
              <div
                key={p.which}
                style={{
                  display: "grid",
                  gridTemplateColumns: "76px minmax(0, 1fr)",
                  gap: "4px 18px",
                  alignItems: "center",
                  padding: "12px 0 14px",
                  borderBottom: "1px solid #ECE4D5",
                }}
              >
                <svg width="76" height="78" viewBox="0 0 76 78" aria-hidden="true">
                  <path d="M68 70 A58 58 0 0 0 10 12" fill="none" stroke="#C9BEAC" strokeWidth="0.8" />
                  <path d="M10 70 L10 12" fill="none" stroke="#C9BEAC" strokeWidth="0.8" strokeDasharray="2 3" />
                  <path d="M10 70 L68 70" fill="none" stroke="#2B2117" strokeWidth="1.2" />
                  {p.known ? (
                    <g>
                      <path d={p.arc} fill="none" stroke="#A07A2E" strokeWidth="1" />
                      <path d={p.wRay} fill="none" stroke="#2B2117" strokeWidth="1" strokeDasharray="1.5 2.5" />
                      <path d={p.rawRay} fill="none" stroke="#A07A2E" strokeWidth="1.6" />
                    </g>
                  ) : null}
                  <circle cx="10" cy="70" r="2.4" fill="#2B2117" />
                </svg>
                <div style={{ minWidth: "0", display: "flex", flexDirection: "column" }}>
                  <div style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: "0 12px" }}>
                    <span style={{ fontSize: "20px", fontStyle: "italic" }}>{p.names}</span>
                    <span style={{ fontSize: "14px", fontVariantCaps: "all-small-caps", letterSpacing: "0.14em", color: "#6B5D4D" }}>
                      {p.which}
                    </span>
                  </div>
                  {p.known ? (
                    <>
                      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: "0 14px" }}>
                        <span style={{ fontSize: "36px", lineHeight: 1.15, fontWeight: 500, color: "#8A6A2C", fontVariantNumeric: "lining-nums tabular-nums" }}>
                          {p.angle}
                        </span>
                        <span style={{ fontSize: "15px", color: "#6B5D4D", fontVariantNumeric: "lining-nums tabular-nums" }}>
                          {p.cosText}
                        </span>
                      </div>
                      <span style={{ fontSize: "15px", fontStyle: "italic", color: "#6B5D4D", fontVariantNumeric: "lining-nums" }}>
                        {p.weighed}
                      </span>
                    </>
                  ) : (
                    <span style={{ fontSize: "15px", color: "#6B5D4D", paddingTop: "6px" }}>[TAKE A SIGHTING WITH THESE WORDS]</span>
                  )}
                </div>
              </div>
            ))}
          </div>
          <p style={{ margin: "0", fontSize: "15px", color: "#6B5D4D" }}>
            Gilt ray: the angle between the raw unembedding rows. Dotted ray: the angle once the model’s last scaling is
            applied, as it weighs words just before speaking <span style={{ fontStyle: "italic" }}>(final-norm weighted)</span>.{" "}
            {shown.own ? `The frame shown was squared from ${rowsWords(shown)}.` : `The next sighting’s frame will be squared from ${rowsWords(shown)}.`}
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: "6px", paddingTop: "14px", borderTop: "1px solid #D9CFBD" }}>
            <div style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", justifyContent: "space-between", gap: "8px 16px" }}>
              <h3 className="h-sub">how square the frame stands</h3>
              {frame ? (
                <span style={{ display: "flex", alignItems: "baseline", gap: "10px" }}>
                  <span style={{ fontSize: "26px", color: "#8A6A2C", fontWeight: 500, fontVariantNumeric: "lining-nums tabular-nums" }}>
                    {(shown.fnw ? frame.condFnw : frame.cond).toFixed(2)}
                  </span>
                  <span style={{ fontSize: "15px", fontStyle: "italic", color: "#6B5D4D" }}>on {rowsWords(shown)}</span>
                </span>
              ) : (
                <span style={{ fontSize: "15px", color: "#6B5D4D" }}>[TAKE A SIGHTING WITH THESE WORDS]</span>
              )}
            </div>
            <p style={{ margin: "0", fontSize: "15px", color: "#6B5D4D" }}>
              A condition number: 1.00 would be perfectly square. The study measured “she” · “he” · “it” at 1.75 on raw rows
              and 2.65 on weighted rows, and the server’s default, “spider” · “eight” · “insect”, at 1.20 and 1.43 (V7). The
              real trouble lay in the order, below.
            </p>
          </div>
          <Apparatus title="angles and squareness" label="Apparatus for the angles between the landmarks" sources={false}>
            <dl className="dl tight num">
              <dt>angle</dt>
              <dd>
                ∠(a, b) = arccos(⟨a, b⟩ / ‖a‖ ‖b‖), on the raw rows W<sub>U</sub>[t] (gilt ray) and on the weighted rows
                fnw · W<sub>U</sub>[t] (dotted ray).
              </dd>
              <dt>squareness</dt>
              <dd>
                κ = σ<sub>max</sub> / σ<sub>min</sub>, the singular values of the three rows, each scaled to length 1.
              </dd>
            </dl>
            {frame ? (
              <dl className="dl tight num">
                <dt>cosines</dt>
                <dd>{sqCosines}</dd>
                <dt>κ here</dt>
                <dd>
                  {frame.cond.toFixed(2)} on the raw rows, {frame.condFnw.toFixed(2)} on the weighted rows
                </dd>
                <dt>raw and weighed</dt>
                <dd>
                  cos(W<sub>U</sub>[t], fnw · W<sub>U</sub>[t]): {nw.map((w) => `${q(w)} ${frame.self[w].toFixed(2)}`).join(", ")}.
                  The final-norm weight runs from −2.4 to 512 (E6 report, Module 4 (a)), so a word’s raw row and the row the
                  model weighs point well apart.
                </dd>
                <dt>source</dt>
                <dd>
                  {frameFromServer ? (
                    <>
                      the server, for the sighting shown (<span className="mono">frame.geometry</span>, the definitions of V7’s frame_stats); word-pieces{" "}
                      {(sighting?.frame?.tokens ?? nw.map((w) => " " + w)).map((w) => "“" + w + "”").join(" · ")}.
                    </>
                  ) : (
                    <>
                      experiments/results/V7_tool_bugs_05_m1patch_frames.json, “frames”; word-pieces{" "}
                      {nw.map((w) => "“ " + w + "”").join(" · ")}.
                    </>
                  )}
                </dd>
              </dl>
            ) : (
              <p style={{ margin: "0", color: "#6B5D4D" }}>
                The study measured three sets only (“she” · “he” · “it”, “woman” · “man” · “person”, “spider” · “eight” ·
                “insect”). For other words the server measures them with every sighting; take one with these words: <Missing />
              </p>
            )}
          </Apparatus>
        </aside>
      </div>

      {/* ------------------------------------------------ iv · why the order matters */}
      <section aria-labelledby="order-head" style={{ marginTop: "52px", display: "flex", flexDirection: "column", gap: "16px" }}>
        <h2 className="h-sec" id="order-head">
          iv · why the order matters
        </h2>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "24px 64px", alignItems: "flex-start" }}>
          <div style={{ flex: "999 1 520px", minWidth: "0", display: "flex", flexDirection: "column", gap: "14px" }}>
            <p style={{ margin: "0", fontSize: "21px", lineHeight: 1.5, maxWidth: "760px" }}>
              As first built, the instrument squared the frame one word at a time <span style={aside}>(Gram–Schmidt)</span>.
              The first word keeps its whole direction, with everything it shares with the others; the second keeps only
              what differs from the first; the third only what differs from both. Name the same three words in another
              order, and you stand in another frame. It now squares all three at once unless you choose otherwise.
            </p>
            <p style={{ margin: "0", fontSize: "15px", color: "#6B5D4D", maxWidth: "760px" }}>
              The elevation depends only on which word comes last. Measured in the study: all six orders of “she” · “he” ·
              “it” over 18 prompts (V7), and the prompts where Gemma speaks of itself (E6).
            </p>
            <Rubric style={{ maxWidth: "760px" }}>
              These figures are for a frame squared in order. The instrument now squares the frame all at once, which gives the
              same three axes for every order; the order still says which axis sets 0°, which 90° and which points up. Squared
              either way, compare a sighting only with sightings whose landmarks were named in the same order: squared in order
              (the recorded one, or one you choose above), another order is another frame; squared all at once, it is the same
              frame with its axes relabelled.
            </Rubric>
          </div>
          <div
            style={{
              flex: "1 1 560px",
              minWidth: "0",
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
              gap: "18px 28px",
              paddingTop: "14px",
              borderTop: "1px solid #D9CFBD",
            }}
          >
            {/* recorded: REPORT §3.4 (58–99°), V7 NOTES (75–98%), E6 report (15° · 51°) */}
            <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
              <span style={numTile}>58–99°</span>
              <span style={{ fontSize: "15px", lineHeight: 1.45 }}>
                how far the bearing swung across the six orders <span style={aside}>(median, by stretch of layers)</span>
              </span>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
              <span style={numTile}>75–98%</span>
              <span style={{ fontSize: "15px", lineHeight: 1.45 }}>of readings moved by more than 30° when only the order changed</span>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
              <span style={numTile}>15° · 51°</span>
              <span style={{ fontSize: "15px", lineHeight: 1.45 }}>
                the bearing at layers 41–50 with “she” first, and with “he” first, where Gemma speaks of itself
              </span>
            </div>
          </div>
        </div>
        <Apparatus title="bug 3, the order of the words" label="Apparatus for the order of the landmarks" style={{ gap: "12px" }}>
          <dl className="dl num">
            <dt>the fault</dt>
            <dd>
              High (REPORT §3.4; theodolite_math.py:387–438 and engine_runtime.py:773, as the REPORT gives them; in today’s
              code, <span className="mono">gram_schmidt_orthonormalize_3d</span> and{" "}
              <span className="mono">build_monument_frame</span>). Gram–Schmidt keeps e<sub>1</sub> = v<sub>1</sub> / ‖v
              <sub>1</sub>‖ whole, so the first word takes the share the three hold in common. The frame is not
              ill-conditioned (κ = 1.75, against 1.20 for the server’s default set); the cause is the order of squaring (V7 NOTES).
            </dd>
            <dt>elevation</dt>
            <dd>
              e<sub>3</sub> is what remains of v<sub>3</sub> once the plane of the other two is removed, and that plane is
              the same in either order: so φ depends only on the last word.
            </dd>
            <dt>the fix</dt>
            <dd>
              Löwdin: E = (V V<sup>⊤</sup>)<sup>−½</sup> V, one frame for every order (V7 checked this on the raw rows, to
              1.2 × 10<sup>−7</sup>), taken on the weighted rows fnw · W<sub>U</sub> and applied to the normed h̃, with the
              fraction caught always shown: all four are now options of the request (frame_squaring “lowdin”, frame_fnw,
              frame_per_layer, centre_prompts), and the server sends the fraction caught at every layer (proj_frac). The
              interface sends frame_squaring “lowdin” by default,{" "}
              {FNW_DEFAULT ? "with frame_fnw" : "on the raw rows (frame_fnw false)"}; the server’s own default stays Gram–Schmidt
              on raw rows. V7 also suggests
              contrast axes such as fnw · (W<sub>U</sub>[she] − W<sub>U</sub>[he]), less a background mean; those are not built.{" "}
              <span className="placeholder">[NOT YET IN THE SERVER]</span>
            </dd>
            <dt>agreement</dt>
            <dd>
              How often the sign of “she” minus “he” in the frame matches the sign of the J++ she−he logit, by stretch of
              layers: 76–88% with “she” first, 72–87% with “he” first, 79–89% squared all at once (the summary file, 18
              prompts; the V7 NOTES print 74–86%, 79–85% and 75–89%). Even the order-free frame agrees only about four
              times in five, which the V7 NOTES put down to its raw rows. V7 did not measure a frame on weighted rows.
            </dd>
            <dt>E6</dt>
            <dd>
              at layers 41–50, prompts where Gemma speaks of itself: θ = 15° in the order she · he · it, 51° in the order he
              · she · it (E6 report).
            </dd>
          </dl>
          <div style={{ overflowX: "auto", paddingBottom: "4px" }}>
            <table style={{ minWidth: "760px", width: "100%", borderCollapse: "collapse", fontSize: "15px", fontVariantNumeric: "lining-nums tabular-nums" }}>
              <caption
                style={{
                  captionSide: "top",
                  textAlign: "left",
                  paddingBottom: "6px",
                  fontVariantCaps: "all-small-caps",
                  letterSpacing: "0.14em",
                  color: "#6B5D4D",
                }}
              >
                all six orders of “she” · “he” · “it”, 18 prompts, medians by stretch of layers
              </caption>
              <thead>
                <tr style={{ borderBottom: "1px solid #D9CFBD", color: "#6B5D4D", fontVariantCaps: "all-small-caps", letterSpacing: "0.12em" }}>
                  <th scope="col" style={{ textAlign: "left", fontWeight: 400, padding: "4px 12px 6px 0" }}>
                    layers
                  </th>
                  {ORDER_STRETCHES.map((s) => (
                    <th key={s} scope="col" style={th}>
                      {s}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {ORDER_TABLE.map((row) => (
                  <tr key={row.label} style={{ borderBottom: "1px solid #ECE4D5" }}>
                    <th scope="row" style={{ textAlign: "left", fontWeight: 400, padding: "6px 12px 6px 0" }}>
                      {row.label}
                    </th>
                    {row.cells.map((c, i) => (
                      <td key={i} style={td}>
                        {c}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p style={{ margin: "0", color: "#6B5D4D", maxWidth: "900px" }}>
            A swing is the range across the six orders. Source: experiments/results/V7_tool_bugs_04_bos_monument_summary.json,
            “monument_order”, which pools 12 chat and 6 raw prompts (report layers L0–9 … L59 = layers 1–10 … 60). The
            figures in the section above are the ones the study quotes: 58–99° from REPORT §3.4, and 75–98% from the V7
            NOTES, which give 59–99° for the chat prompts alone.
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
            <span style={{ fontVariantCaps: "all-small-caps", letterSpacing: "0.14em", color: "#6B5D4D" }}>
              one prompt, six orders, layer 46
            </span>
            <span style={{ color: "#6B5D4D", maxWidth: "900px" }}>
              the last words of one V7 prompt, as recorded:{" "}
              <span style={{ fontStyle: "italic", color: "#2B2117" }}>“Gemma is a helpful AI assistant. In conversations,”</span>.
              {atOnce
                ? `These rows are frames squared in order; ${
                    shown.own
                      ? "the sighting shown is squared all at once, so reordering its landmarks keeps its three axes and only relabels them (swap the first two and its bearing reads 90° − θ)"
                      : "the frame set above squares all at once, so reordering the landmarks keeps its three axes and only relabels them (swap the first two and a bearing reads 90° − θ)"
                  }. To find your own row, take it in the frame as first built.`
                : "Reorder the stones above with “she”, “he”, “it” to find your own row."}
            </span>
            <div style={{ overflowX: "auto", paddingBottom: "4px" }}>
              <div style={{ minWidth: "460px", maxWidth: "640px" }} role="table" aria-label="One prompt, six orders, layer 46">
                <div
                  role="row"
                  style={{
                    display: "grid",
                    gridTemplateColumns: "minmax(150px, 1fr) 90px 100px 120px",
                    gap: "0 14px",
                    padding: "4px 0 6px",
                    borderBottom: "1px solid #D9CFBD",
                    fontVariantCaps: "all-small-caps",
                    letterSpacing: "0.12em",
                    color: "#6B5D4D",
                  }}
                >
                  <span role="columnheader">order</span>
                  <span role="columnheader" style={{ textAlign: "right" }}>
                    bearing θ
                  </span>
                  <span role="columnheader" style={{ textAlign: "right" }}>
                    elevation φ
                  </span>
                  <span role="columnheader">
                    <span className="visually-hidden">yours</span>
                  </span>
                </div>
                {ORDERS.map(([order, oaz, oel]) => {
                  const mine = isWords && nw.join("|") === order.join("|");
                  return (
                    <div
                      key={order.join("|")}
                      role="row"
                      style={{
                        display: "grid",
                        gridTemplateColumns: "minmax(150px, 1fr) 90px 100px 120px",
                        gap: "0 14px",
                        alignItems: "baseline",
                        padding: "6px 0",
                        borderBottom: "1px solid #ECE4D5",
                        background: mine ? "#EDE5D3" : "transparent",
                        fontVariantNumeric: "lining-nums tabular-nums",
                      }}
                    >
                      <span role="cell" style={{ fontStyle: "italic" }}>
                        {order.join(" · ")}
                      </span>
                      <span role="cell" style={{ textAlign: "right" }}>
                        {d1(oaz)}°
                      </span>
                      <span role="cell" style={{ textAlign: "right" }}>
                        {d1(oel)}°
                      </span>
                      <span role="cell" style={{ fontSize: "14px", color: "#8F2B1E" }}>
                        {mine ? (atOnce ? "◆ your order, if squared in order" : "◆ your order") : ""}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
            <span style={{ color: "#6B5D4D", maxWidth: "900px" }}>
              Rows with the same last word share one elevation. Source: the same file, “example_L45” (report layer L45 =
              layer 46).
            </span>
          </div>
        </Apparatus>
      </section>

      {/* ------------------------------------------------ v · bearing through the layers */}
      <section aria-labelledby="bearing-head" style={{ marginTop: "48px", display: "flex", flexDirection: "column", gap: "12px" }}>
        <div className="row-between">
          <h2 className="h-sec" id="bearing-head">
            v · bearing through the layers
          </h2>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 22px", fontSize: "15px", color: "#6B5D4D" }}>
            <span style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <svg width="22" height="10" viewBox="0 0 22 10" aria-hidden="true">
                <path d="M1 5 H21" stroke="#A07A2E" strokeWidth="1.6" />
                <circle cx="11" cy="5" r="2.4" fill="#A07A2E" />
              </svg>
              bearing, round the horizon <span style={{ fontStyle: "italic" }}>(azimuth)</span>
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <svg width="22" height="10" viewBox="0 0 22 10" aria-hidden="true">
                <path d="M1 5 H21" stroke="#2E4A7D" strokeWidth="1.6" />
                <path d="M11 1.8 L13.4 5 L11 8.2 L8.6 5 Z" fill="#2E4A7D" />
              </svg>
              elevation, toward the third
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span aria-hidden="true" style={{ color: "#A07A2E" }}>
                ✦
              </span>
              looks back over everything
            </span>
          </div>
        </div>
        <Rubric style={{ maxWidth: "860px" }}>
          These charts follow the stream as the {lens.name} lens reads it; read them left to right, from the first layer to the
          sixtieth. A bearing of 0° points along the first landmark’s axis and 90° along the second’s; an elevation of 90° points
          straight along the third’s{ownAtOnce ? " (in a frame squared all at once each word stands a little off its axis)" : ""}. Move
          the layer above, and the red line follows. No bearing was checked against what the model does, at any layer: in the
          study, bearings swung 58–99° when the landmarks were reordered
          {ownAtOnce ? " (in a frame squared in order; this one is squared all at once)" : ""}, and the landmarks caught only a
          few percent of the reading.
        </Rubric>
        <BearingCaution sighting={sighting} chart style={{ maxWidth: "860px" }} />
        <div style={{ overflowX: "auto", paddingBottom: "6px" }}>
          <div style={{ minWidth: "940px" }}>
            <svg
              viewBox="0 0 1280 456"
              role="img"
              aria-label={`Two engraved line charts across the sixty layers of the sighting shown. Above, the bearing of the stream as the ${lens.name} lens reads it, from minus 180 to 180 degrees; below, its elevation, from minus 90 to 90 degrees. The chosen layer is marked in red.`}
              style={{ width: "100%", height: "auto", display: "block" }}
            >
              <g fontFamily="EB Garamond, Garamond, serif">
                <text x="22" y="150" transform="rotate(-90 22 150)" textAnchor="middle" fontSize="12" letterSpacing="2.5" fill="#7A5C28">
                  BEARING
                </text>
                <text x="22" y="350" transform="rotate(-90 22 350)" textAnchor="middle" fontSize="12" letterSpacing="2.5" fill="#2E4A7D">
                  ELEVATION
                </text>
              </g>
              <path d="M80 50 H1140 M80 100 H1140 M80 200 H1140 M80 250 H1140 M80 300 H1140 M80 400 H1140" fill="none" stroke="#E3DACA" strokeWidth="1" />
              <path d="M80 150 H1140 M80 350 H1140" fill="none" stroke="#C9BEAC" strokeWidth="1" />
              <path d={chart.globalD} fill="none" stroke="#A07A2E" strokeOpacity="0.32" strokeWidth="1" />
              <g fontFamily="EB Garamond, Garamond, serif" fontSize="13" fill="#6B5D4D" textAnchor="end" style={{ fontVariantNumeric: "lining-nums tabular-nums" }}>
                <text x="70" y="54">180°</text>
                <text x="70" y="104">90°</text>
                <text x="70" y="154">0°</text>
                <text x="70" y="204">−90°</text>
                <text x="70" y="254">−180°</text>
                <text x="70" y="304">90°</text>
                <text x="70" y="354">0°</text>
                <text x="70" y="404">−90°</text>
              </g>
              <g fontFamily="EB Garamond, Garamond, serif" fontSize="14" fontStyle="italic" fill="#6B5D4D">
                <text x="1152" y="54">away from {SN[0]}</text>
                <text x="1152" y="104">toward {SN[1]}</text>
                <text x="1152" y="154" fill="#2B2117">
                  toward {SN[0]}
                </text>
                <text x="1152" y="204">away from {SN[1]}</text>
                <text x="1152" y="254">away from {SN[0]}</text>
                <text x="1152" y="304" fill="#2B2117">
                  toward {SN[2]}
                </text>
                <text x="1152" y="354">level</text>
                <text x="1152" y="404">away from {SN[2]}</text>
              </g>
              <path d={chart.bPath} fill="none" stroke="#A07A2E" strokeWidth="1.6" strokeLinejoin="round" strokeLinecap="round" />
              <path d={chart.bDots} fill="#A07A2E" />
              <path d={chart.ePath} fill="none" stroke="#2E4A7D" strokeWidth="1.6" strokeLinejoin="round" strokeLinecap="round" />
              <path d={chart.eDots} fill="#2E4A7D" />
              <line x1={cxL(li)} y1="36" x2={cxL(li)} y2="410" stroke="#8F2B1E" strokeOpacity="0.55" strokeWidth="1" />
              {az !== null ? <circle cx={cxL(li)} cy={yB(az)} r="5.5" fill="#F6F0E3" stroke="#8F2B1E" strokeWidth="1.6" /> : null}
              {el !== null ? <circle cx={cxL(li)} cy={yE(el)} r="5.5" fill="#F6F0E3" stroke="#8F2B1E" strokeWidth="1.6" /> : null}
              <text
                x={cxL(li)}
                y="26"
                textAnchor="middle"
                fontFamily="EB Garamond, Garamond, serif"
                fontSize="15"
                fill="#8F2B1E"
                letterSpacing="1.5"
                style={{ fontVariantCaps: "all-small-caps", fontVariantNumeric: "lining-nums" }}
              >
                layer {li + 1}
              </text>
              <g fontFamily="EB Garamond, Garamond, serif" fontSize="12" fill="#7A5C28" textAnchor="middle">
                {[5, 11, 17, 23, 29, 35, 41, 47, 53, 59].map((l) => (
                  <text key={l} x={cxL(l)} y="426">
                    ✦
                  </text>
                ))}
              </g>
              <g fontFamily="EB Garamond, Garamond, serif" fontSize="13" fill="#6B5D4D" textAnchor="middle" style={{ fontVariantNumeric: "lining-nums tabular-nums" }}>
                <text x="80" y="446">
                  1
                </text>
                {[5, 11, 17, 23, 29, 35, 41, 47, 53, 59].map((l) => (
                  <text key={l} x={cxL(l)} y="446">
                    {l + 1}
                  </text>
                ))}
                <text x="1152" y="446" textAnchor="start" fontSize="12" letterSpacing="2">
                  LAYER
                </text>
              </g>
            </svg>
          </div>
        </div>
        <p style={{ margin: "0", fontSize: "15px", color: "#6B5D4D", maxWidth: "900px" }}>
          Where the gilt line breaks, the bearing has crossed ±180°: both edges are one direction, straight away from{" "}
          {SN[0]}. Source:{" "}
          {isFixture
            ? "the study’s demonstration sighting (E6), taken before the two V7 fixes."
            : sighting
              ? sighting.provenance + "."
              : <Missing what="no sighting yet" />}
        </p>
      </section>

      {/* ------------------------------------------------ vi · how close each came to being said */}
      <section aria-labelledby="said-head" style={{ marginTop: "48px", display: "flex", flexDirection: "column", gap: "14px" }}>
        <h2 className="h-sec" id="said-head">
          vi · how near each word came, through the lens
        </h2>
        <Rubric style={{ maxWidth: "860px" }}>
          A rank is a word’s place among 262,144 word-pieces as the J++ lens reads the stream at each layer: rank 1 is the
          lens’s top word. Only at layer 60 is it the model’s own queue. Higher on each little chart means the lens ranks
          that word nearer the top.
        </Rubric>
        {sf?.kind === "words" ? (
          <div style={{ overflowX: "auto", paddingBottom: "6px" }}>
            <div style={{ minWidth: "900px" }}>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "120px minmax(300px, 1fr) 140px 100px 130px 120px",
                  gap: "0 18px",
                  alignItems: "end",
                  paddingBottom: "10px",
                  borderBottom: "1px solid #D9CFBD",
                  fontSize: "14px",
                  fontVariantCaps: "all-small-caps",
                  letterSpacing: "0.14em",
                  color: "#6B5D4D",
                }}
              >
                <span>landmark</span>
                <span>its place, layer by layer</span>
                <span style={{ textAlign: "right" }}>nearest it came</span>
                <span style={{ textAlign: "right" }}>at layer</span>
                <span style={{ textAlign: "right" }}>at layer {li + 1}</span>
                <span style={{ textAlign: "right" }}>at layer 60</span>
              </div>
              {rows.map((r) => {
                const last = r.lens[lastIdx] ?? null;
                const aria =
                  r.best !== null
                    ? `${q(r.w)} through the sixty layers: nearest rank ${comma(r.best)} at layer ${r.bi + 1}, rank ${last !== null ? comma(last) : "not recorded"} at layer 60.`
                    : `${q(r.w)}: no rank recorded in this sighting.`;
                return (
                  <div
                    key={r.w}
                    style={{
                      display: "grid",
                      gridTemplateColumns: "120px minmax(300px, 1fr) 140px 100px 130px 120px",
                      gap: "0 18px",
                      alignItems: "center",
                      padding: "10px 0",
                      borderBottom: "1px solid #ECE4D5",
                      fontVariantNumeric: "lining-nums tabular-nums",
                    }}
                  >
                    <span style={{ fontSize: "28px", fontStyle: "italic", lineHeight: 1.2 }}>{q(r.w)}</span>
                    <svg viewBox="0 0 460 60" role="img" aria-label={aria} style={{ width: "100%", maxWidth: "460px", height: "60px", display: "block" }}>
                      <path d="M48 4 H452" fill="none" stroke="#D9CFBD" strokeWidth="1" />
                      <path d="M48 21.7 H452 M48 39.4 H452 M48 52 H452" fill="none" stroke="#ECE4D5" strokeWidth="1" />
                      <g fontFamily="EB Garamond, Garamond, serif" fontSize="11" fill="#6B5D4D" textAnchor="end">
                        <text x="42" y="8">1</text>
                        <text x="42" y="25.5">100</text>
                        <text x="42" y="43">10,000</text>
                      </g>
                      <line x1={sx(li)} y1="1" x2={sx(li)} y2="57" stroke="#8F2B1E" strokeOpacity="0.5" strokeWidth="1" />
                      <path d={sparkPath(r.lens, VOCAB)} fill="none" stroke="#2B2117" strokeWidth="1.2" strokeLinejoin="round" />
                      {last !== null ? <path d={ring([sx(lastIdx), syRank(last, VOCAB)], 3)} fill="#F6F0E3" stroke="#2B2117" strokeWidth="1" /> : null}
                      {r.best !== null ? <path d={loz([sx(r.bi), syRank(r.best, VOCAB)], 5)} fill="#A07A2E" /> : null}
                    </svg>
                    <span style={{ textAlign: "right", fontSize: "24px", color: "#2B2117" }}>
                      {rankNode(r.best)}
                    </span>
                    <span style={{ textAlign: "right", fontSize: "17px", color: "#6B5D4D" }}>{r.best !== null ? r.bi + 1 : ""}</span>
                    <span style={{ textAlign: "right", fontSize: "17px" }}>
                      {rankNode(r.lens[li])}
                    </span>
                    <span style={{ textAlign: "right", fontSize: "17px" }}>
                      {rankNode(last)}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        ) : (
          <p style={{ margin: 0, fontSize: "17px", color: "#6B5D4D" }}>
            {sf ? (
              <>
                The landmarks of the sighting shown are a neuron’s directions, not words, so they have no rank:{" "}
                <Missing what={NOT_HERE} />
              </>
            ) : (
              <Missing what="no sighting yet" />
            )}
          </p>
        )}
        {sf?.kind === "words" ? (
          <>
            <p style={{ margin: "0", fontSize: "15px", color: "#6B5D4D" }}>
              Gilt lozenge: the nearest each word came. Hollow ring: where it stood at the last layer. Read through {lensPhrase(lens)},
              at the word{" "}
              {sighting?.readAt != null ? wp(sighting.readAt) : <Missing what={NOT_HERE} />}
              {k >= 1 ? `, the ${ordinal(k)} word-piece of its own answer,` : ""} in the sighting shown.
            </p>
            <p style={{ margin: "0", fontSize: "15px", color: "#6B5D4D", maxWidth: "900px" }}>
              To the naked eye <span style={{ fontStyle: "italic" }}>(the logit lens)</span>{" "}
              {eyeKnown ? (
                <>
                  {rows.every((r) => r.best === null || (r.eyeBest as number) < r.best)
                    ? "the three climb higher"
                    : "the three come nearest"}
                  {rows.every((r) => r.ebi >= 10 && r.ebi <= 49) ? " in the middle layers" : ""}:{" "}
                  {rows.map((r, i) => (
                    <React.Fragment key={r.w}>
                      {i === 0 ? "" : ", "}
                      {q(r.w)} {i === 0 ? "to rank " : "to "}
                      {comma(r.eyeBest as number)} at layer {r.ebi + 1}
                    </React.Fragment>
                  ))}
                  .
                  {rows.every((r) => r.eye[lastIdx] !== null && r.eye[lastIdx] === r.lens[lastIdx])
                    ? " At layer 60 the two views agree exactly."
                    : ""}
                </>
              ) : (
                <>
                  : <Missing what={NOT_HERE} />
                </>
              )}
            </p>
            {isFixture && nearest && furthest && top60 ? (
              <p style={{ margin: "4px 0 0", fontSize: "20px", lineHeight: 1.55, maxWidth: "900px" }}>
                None of the three came near, and none was expected to: after “{sighting?.answerBegins}” the next word is a
                verb, and at the last layer the model was all but certain of it, {wp(top60.token)} (
                {top60.prob !== null ? pct1(top60.prob) : <Missing what={NOT_HERE} />}). Of the landmarks, {q(nearest.w)}{" "}
                ended nearest, at rank {comma(nearest.last)}, and {q(furthest.w)} furthest, at {comma(furthest.last)}. One
                sighting proves little alone, but it sits well with the study’s headline: with no system prompt,
                Gemma-4-31B calls itself “it”, and said “she” in none of 320 samples.
              </p>
            ) : sighting ? (
              <p style={{ margin: "4px 0 0", fontSize: "20px", lineHeight: 1.55, maxWidth: "900px" }}>
                At the last layer the model’s likeliest next word-piece was{" "}
                {top60 ? (
                  <>
                    {wp(top60.token)} ({top60.prob !== null ? pct1(top60.prob) : <Missing what={NOT_HERE} />})
                  </>
                ) : (
                  <Missing what={NOT_HERE} />
                )}
                {chosen && top60 && L60
                  ? isChosenRow(sighting, L60, 0)
                    ? ", the one it chose"
                    : chosen.endMark
                      ? `; ${endedWords(chosen.id, chosen.token)} instead`
                      : `; it chose ${wp(chosen.token)}${chosen.modelProb != null ? ` (${pct(chosen.modelProb)} of its own odds as it wrote)` : ""}`
                  : ""}
                .{" "}
                {nearest && furthest ? (
                  <>
                    Of the landmarks, {q(nearest.w)} ended nearest, at rank {comma(nearest.last)}, and {q(furthest.w)}{" "}
                    furthest, at {comma(furthest.last)}.{" "}
                  </>
                ) : !anyRanks ? (
                  <>
                    The server tracks the ranks of its own fixed list of words, and these landmarks were not among them:{" "}
                    <Missing what={NOT_HERE} />.{" "}
                  </>
                ) : null}
                One sighting proves little alone
                {studySet
                  ? ". The study’s headline: with no system prompt, Gemma-4-31B calls itself “it”, and said “she” in none of 320 samples."
                  : "."}
              </p>
            ) : null}
          </>
        ) : null}
        <Rubric style={{ maxWidth: "860px" }}>
          Compare the landmarks with one another, not with certainty. The study found that at layers 51 to 59 the J++ lens
          leans toward “she” for most of the AIs it tried (six or seven of nine), far more than the other lenses it tried: mostly a quirk of the J++ recipe,
          for the Jacobian lens, also fitted on the base model, removes it almost as well as a lens fitted on the instruct
          model (R1). Do not read it as how the model sees itself. Compare a landmark across versions of a question at the same
          word-piece, not across the word-pieces of one answer.
        </Rubric>
        {readsInsideAnswer(sighting) ? <AnswerPositionCaution sighting={sighting} /> : null}
        {sighting && readWhere(sighting).role === "template_end" ? <TemplateEndCaution /> : null}
        <SightingFarCaution sighting={sighting ?? null} />
        <Apparatus title="ranks" label="Apparatus for the ranks">
          <dl className="dl num">
            <dt>rank</dt>
            <dd>
              r(t) = 1 + #{"{"}v : logit<sub>v</sub> &gt; logit<sub>t</sub>
              {"}"} among 262,144 word-pieces (engine_runtime.py). The lens reads unembed(h̃<sub>ℓ</sub>); the naked eye
              reads unembed(h<sub>ℓ</sub>), the logit lens (nostalgebraist 2020) [8].
            </dd>
            <dt>at layer {li + 1}</dt>
            <dd>
              lens / naked eye:{" "}
              {rows.length ? (
                rows.map((r, i) => (
                  <React.Fragment key={r.w}>
                    {i === 0 ? "" : " · "}
                    {q(r.w)} {rankNode(r.lens[li])} / {rankNode(r.eye[li])}
                  </React.Fragment>
                ))
              ) : (
                <Missing what={NOT_HERE} />
              )}
            </dd>
            <dt>caveat</dt>
            <dd>
              When the word-piece chosen here is a bare space, the old rule took each of the lens’s ranks from the better of
              this position and the next, and left the naked eye’s alone, which flatters the lens (E6 report, item 9). Whether it
              applied is read from the server’s own record of each rank, never from the likeliest word at layer 60.{" "}
              <SpaceRule s={sighting} L={L} li={li} />
            </dd>
            <dt>read with care</dt>
            <dd>
              Against a matched control, centred, never alone; and flag pronoun readings at layers 51–59 as unreliable
              (REPORT §3.5, recommendations 2 and 3).
            </dd>
            <dt>matched control</dt>
            <dd style={{ margin: "0", display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: "4px 14px" }}>
              <ControlSighting layerIdx={li} />
            </dd>
            <dt>source</dt>
            {isFixture ? (
              <dd>
                E6_tool_audit_02_sweep_demo.json, key “prepend_bos=True”: <span className="mono">jpp_tracked_rank</span>,{" "}
                <span className="mono">naive_tracked_rank</span>.
                {(() => {
                  const extra = Object.keys(layers[0]?.ranks ?? {}).filter((k) => !rows.some((r) => r.w === k));
                  if (!extra.length) return null;
                  const qs = extra.map(q);
                  return ` The sighting also tracked ${qs.length > 1 ? qs.slice(0, -1).join(", ") + " and " + qs[qs.length - 1] : qs[0]}.`;
                })()}
              </dd>
            ) : (
              <dd>
                the live sighting’s <span className="mono">layer_plumes[ℓ].tracked_ranks</span> (backend/server.py), read
                through the lens only; the server sends no naked-eye ranks.
                {(() => {
                  const keys = Object.keys(layers[0]?.ranks ?? {});
                  return keys.length ? ` It tracked ${keys.map(q).join(", ")}.` : "";
                })()}
              </dd>
            )}
          </dl>
        </Apparatus>
      </section>
    </Page>
  );
};

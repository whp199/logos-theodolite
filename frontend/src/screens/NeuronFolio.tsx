import React, { useEffect, useMemo, useRef, useState } from "react";
import { Apparatus, IfApparatus, IfRubrics, LensRecord, Missing, NotYet, Page, Rubric } from "../components/kit";
import { href, type Route } from "../router";
import { useApparatus } from "../reading";
import {
  type AnswerSet,
  type CausalCheckCard,
  type CausalContexts,
  type Conn,
  type LayerReading,
  type Pending,
  type PoleReading,
  type Sighting,
  CAUSAL_DEFAULTS,
  causalKey,
  chosenOf,
  chosenPhrase,
  editIdsOf,
  isGlobalIdx,
  isTextEnd,
  isTurnEnd,
  loudnessRank,
  maxAct,
  ordinal as nth,
  poleKey,
  rankingWords,
  readWhere,
  retakeIfLive,
  useShownSet,
  useTheodoliteStore,
} from "../store/useTheodoliteStore";
import { andList, agreement, cap, count, glossFor, scoresText, shortToken, signedRaw, type Agreement } from "./NeuronFolio/util";
import { splitAtPeak, usePassages } from "./NeuronFolio/passages";
import { get as atlasGet, post as atlasPost, type Drive, type FaceWords, type Wiring } from "./atlas/api";
import { comma, faceWord, ms, numberWord, pct, pieceWord, signed, wp } from "../lib/words";
import { HeldBackWords } from "../components/HeldBack";
import {
  type Gate,
  bandOf,
  fixtureCheck,
  heldLine,
  heldRubric,
  heldTag,
  labelCheckJson,
  labelGate,
  layerIdx,
  parseLabelCheck,
  overlapLine,
  sharedLine,
  trustLine,
} from "../lib/labels";
import { AnswerMarks, useAnswerPieces } from "./layerLeaf/AnswerMarks";
import { lensLabel, useReadingLens } from "../lib/lens";
import { OneDrawLine, PenaltyLine } from "../lib/cautions";
import {
  ACTS,
  CMP,
  FFN,
  FFN_TOOL,
  IDS,
  JPP,
  LOUDEST,
  NAIVE,
  NB_NOTES,
  RAW_READOUT,
  RECORDED,
  ordinal,
} from "./NeuronFolio/recorded";
import {
  type CausalCheck,
  type CausalFace,
  type CausalLabel,
  type CausalWord,
  type FaceReading,
  type Hypothesis,
  type Inspection,
  type Readers,
  type Tok,
  hypothesisWords,
  nText,
  parseCausalCheck,
  parseHypothesis,
  parseInspection,
  parseReaders,
  pShown,
} from "./NeuronFolio/replies";
import {
  type FaceNotes,
  type FolioSaved,
  type FolioWhere,
  type Jobs,
  type Migration,
  MAX_NAME,
  carriedKeys,
  carryToServer,
  folioEntries,
  folioMigration,
  markCarried,
  useFolioSaved,
} from "./NeuronFolio/saved";
import {
  LINEARITY_TAIL,
  LINEARITY_TAIL_IDX,
  NOISE_RATIO_MAX,
  STUDY_K10,
  STUDY_K10_IDX,
  STUDY_K10_LATE_P10,
  STUDY_LINEARITY,
  nearestSampled,
  probWord,
  sameIds,
  sig,
  timesWord,
} from "./NeuronFolio/causal";

/**
 * A neuron's folio (canvas: Neuron.dc.html, "folio iii"). Works for any neuron in the route:
 *   per-sighting values come from sighting.layers[idx];
 *   both faces come from inspect_pole (called on arrival), the threads from trace_readers (on request),
 *   what builds it from the atlas's attn_drive at the sighting's own word-piece ids, its heads, blocks and earlier
 *   neurons, and from the atlas's wiring, its writers by the weights alone (on request),
 *   the test of a label from causal_spot_check (on request: at the sighting's word-piece, or on the study's 36 texts),
 *   the guess from test_motif_hypothesis;
 *   the canvas's recorded neuron (layer 54, neuron 15491, positive face) shows the study's recorded extras
 *   when the recorded sighting is loaded;
 *   the name and the field notes are kept on the server whenever it answers for the neuron (NeuronFolio/saved.ts): the
 *   name as the atlas's name for it, the notes per face in the field book beside the atlas's notes, with the atlas's own
 *   note shown in vii; with no server they stay in this browser's localStorage, and vii says which keeps them and
 *   offers, once, to carry this browser's to the server.
 */

type NeuronRoute = Extract<Route, { name: "neuron" }>;
type Upd = (fn: (prev: FolioSaved) => FolioSaved) => void;

const INK = "#2B2117";
const MUTED = "#6B5D4D";
const GILT = "#A07A2E";
const GILT_TEXT = "#7A5C28";
const HAIR = "#D9CFBD";
const HAIR_SOFT = "#E3DACA";
const HAIR_FAINT = "#ECE4D5";
const RED = "#8F2B1E";
const LAPIS = "#2E4A7D";

const SECTION: React.CSSProperties = {
  padding: "40px 0 46px",
  borderTop: `1px solid ${HAIR}`,
  display: "flex",
  flexDirection: "column",
  gap: "18px",
};
const ASIDE: React.CSSProperties = { fontSize: "15px", color: MUTED };
const PH: React.CSSProperties = { fontSize: "13.5px", letterSpacing: "0.08em", color: MUTED };
const PRE: React.CSSProperties = {
  margin: 0,
  padding: "8px 0",
  borderTop: `1px solid ${HAIR_FAINT}`,
  borderBottom: `1px solid ${HAIR_FAINT}`,
  fontFamily: "var(--mono)",
  fontSize: "13px",
  lineHeight: 1.55,
  color: INK,
  whiteSpace: "pre-wrap",
  overflowWrap: "anywhere",
};
const SC_LABEL: React.CSSProperties = {
  fontVariantCaps: "all-small-caps",
  letterSpacing: "0.16em",
  fontSize: "16px",
  color: INK,
};
const LINK_BOX: React.CSSProperties = {
  alignSelf: "flex-start",
  display: "inline-flex",
  alignItems: "center",
  minHeight: "48px",
  padding: "0 22px",
  border: "1px solid #C9BEAC",
  borderRadius: "2px",
  color: INK,
  textDecoration: "none",
  fontSize: "18px",
  fontVariantCaps: "all-small-caps",
  letterSpacing: "0.16em",
};
const TEXTAREA: React.CSSProperties = {
  fontFamily: "inherit",
  fontSize: "18px",
  lineHeight: 1.5,
  color: INK,
  background: "transparent",
  border: "1px solid #C9BEAC",
  borderRadius: "2px",
  padding: "10px 12px",
  width: "100%",
  boxSizing: "border-box",
  resize: "vertical",
};
const NUM: React.CSSProperties = { fontVariantNumeric: "lining-nums tabular-nums" };
const SERIF = "EB Garamond, Garamond, serif";

/* ------------------------------------------------------------------ the model */

/**
 * What follows “the word X” where a sighting was read: readWhere's phrase (“, in its reply “Hi! I””, “, at the end of the
 * chat template”, “, at the end of your question”); for a later word-piece of the answer, whose piece is already named, only
 * its place: “, the 2nd word-piece of its own answer”.
 */
function afterTheWord(s: Sighting | null): string {
  if (!s) return "";
  const w = readWhere(s);
  return w.k >= 1 ? `, the ${nth(w.k)} word-piece of its own answer` : w.phrase;
}

interface Labels {
  lens: Tok[] | null;
  eye: Tok[] | null;
  from: "inspection" | "recorded" | "sighting" | null;
}

interface Model {
  idx: number;
  layerNo: number;
  neuron: number;
  /** 0 = loudest among the layer's fifteen listed poles by |a|; when they are ranked by write size this is not a rank
   * in the layer, only among the listed (pole.rank is the exact rank in the sighting's own ranking) */
  loudRank: number | null;
  /** the poles were ranked by write size, |a| · ‖w_down ⊙ w_post‖, not by |a| */
  byWrite: boolean;
  /** the words of the sighting's ranking (sighting.ts rankingWords) */
  rw: ReturnType<typeof rankingWords>;
  conn: Conn;
  sighting: Sighting | null;
  layer: LayerReading | null;
  poles: PoleReading[];
  pole: PoleReading | null;
  loudest: PoleReading | null;
  fired: boolean;
  /** the layer's fifteen were listed (by the sighting's ranking), and this neuron is not among them */
  notAmong: boolean;
  routeFace: 1 | -1;
  focusFace: 1 | -1;
  otherFace: 1 | -1;
  isFixture: boolean;
  /** the canvas's neuron, with the recorded sighting loaded */
  recorded: boolean;
  /** the canvas's layer (54), with the recorded sighting loaded */
  recordedLayer: boolean;
  /** signed activation a in this sighting; null when not recorded */
  act: number | null;
  loudAct: number | null;
  barPct: number | null;
  insp: Pending<unknown> | undefined;
  inspection: Inspection | null;
  focus: Labels;
  other: FaceReading | null;
  agr: Agreement | null;
  /** the label gate for the face read first (lib/labels.ts): from the inspection's check of that face, else the
   * sighting's (or, for the recorded neuron, the recorded check); with none, the layer alone decides */
  focusGate: Gate;
  /** the same for the other face, from the inspection's check of it */
  otherGate: Gate;
  defaultNick: string | null;
  readAt: string;
  inReply: string;
}

const tokOf = (s: [string, number][]): Tok[] => s.map(([token, logit]) => ({ token, logit }));
const tokFromReading = (r: { token: string; logit: number | null }[] | null): Tok[] | null =>
  r ? r.map((t) => ({ token: t.token, logit: t.logit })) : null;
const f1 = (n: number) => Math.round(n * 10) / 10;
/** "loudest", "fifth loudest" */
const loudPhrase = (rank: number) => (rank === 0 ? "loudest" : `${ordinal(rank)} loudest`);

function useModel(route: NeuronRoute): Model {
  const idx = route.layerNo - 1;
  const { neuron } = route;
  const sighting = useTheodoliteStore((s) => s.sighting);
  const conn = useTheodoliteStore((s) => s.conn);
  const insp = useTheodoliteStore((s) => s.inspections[poleKey(idx, neuron)]) as Pending<unknown> | undefined;
  return useMemo(() => {
    const layer = sighting?.layers[idx] ?? null;
    const poles = layer?.poles ?? [];
    const pole = poles.find((p) => p.neuron === neuron) ?? null;
    const loudest = poles.length ? poles.reduce((a, b) => (Math.abs(b.act) > Math.abs(a.act) ? b : a)) : null;
    const isFixture = sighting?.source === "fixture";
    const isCanvasNeuron = idx === RECORDED.idx && neuron === RECORDED.neuron;
    const recorded = isFixture && isCanvasNeuron;
    const recordedLayer = isFixture && idx === RECORDED.idx;
    const focusFace: 1 | -1 = pole ? pole.face : route.face;
    const otherFace: 1 | -1 = focusFace > 0 ? -1 : 1;
    const act = recorded ? RECORDED.act : pole ? pole.act : null;
    const loudAct = recordedLayer ? LOUDEST : loudest ? Math.abs(loudest.act) : null;
    const barPct = act !== null && loudAct ? f1((Math.abs(act) / loudAct) * 100) : null;
    const inspection = insp?.data ? parseInspection(insp.data) : null;
    const pick = (r: Inspection | null, f: 1 | -1) => (r ? (f > 0 ? r.pos : r.neg) : null);
    const inspFocus = pick(inspection, focusFace);
    let focus: Labels = { lens: null, eye: null, from: null };
    if (inspFocus && (inspFocus.lens || inspFocus.eye)) focus = { ...inspFocus, from: "inspection" };
    else if (recorded) focus = { lens: tokOf(JPP), eye: tokOf(NAIVE), from: "recorded" };
    else if (pole && (pole.lens || pole.eye))
      focus = { lens: tokFromReading(pole.lens), eye: tokFromReading(pole.eye), from: "sighting" };
    const other = pick(inspection, otherFace);
    const agr = focus.lens && focus.eye ? agreement(focus.lens, focus.eye) : null;
    // the sighting's check is of the face that fired, which is the face read first whenever the pole was listed
    const poleCheck = pole && pole.face === focusFace ? pole.check : null;
    const focusCheck =
      focus.from === "inspection"
        ? (inspFocus?.check ?? poleCheck)
        : focus.from === "recorded"
          ? (poleCheck ?? fixtureCheck(RECORDED.idx, RECORDED.rank))
          : poleCheck;
    const focusGate = labelGate(layerIdx(idx), focusCheck);
    const otherGate = labelGate(layerIdx(idx), other?.check ?? null);
    const readAt = sighting?.readAt != null ? pieceWord(sighting.readAt) : "";
    const inReply = afterTheWord(sighting);
    return {
      idx,
      layerNo: route.layerNo,
      neuron,
      loudRank: layer ? loudnessRank(layer, neuron) : null,
      byWrite: sighting?.poleRanking === "write",
      rw: rankingWords(sighting?.poleRanking ?? "activation"),
      conn,
      sighting,
      layer,
      poles,
      pole,
      loudest,
      fired: pole !== null,
      notAmong: pole === null && poles.length > 0,
      routeFace: route.face,
      focusFace,
      otherFace,
      isFixture,
      recorded,
      recordedLayer,
      act,
      loudAct,
      barPct,
      insp,
      inspection,
      focus,
      other,
      agr,
      focusGate,
      otherGate,
      defaultNick: isCanvasNeuron ? RECORDED.nick : null,
      readAt,
      inReply,
    };
  }, [sighting, conn, insp, idx, neuron, route.face, route.layerNo]);
}

/** What a placeholder says while a live answer is awaited. */
function awaitWhat(m: Model, pending: boolean | undefined): string {
  if (pending) return "BEING READ…";
  if (m.conn !== "open") return "THE INSTRUMENT ANSWERS WHEN THE SERVER IS RUNNING";
  return "NOT READ IN THIS SIGHTING";
}

/** The server's WebSocket errors land in sweepError; show them while a reply is awaited. */
const ServerErrorWhilePending: React.FC<{ pending: boolean | undefined }> = ({ pending }) => {
  const err = useTheodoliteStore((s) => s.sweepError);
  const conn = useTheodoliteStore((s) => s.conn);
  // only an error that arrived after the request went out belongs to it
  const atStart = useRef<string | null | undefined>(undefined);
  if (!pending) atStart.current = undefined;
  else if (atStart.current === undefined) atStart.current = err;
  if (!pending || !err || conn !== "open" || err === atStart.current) return null;
  return (
    <p className="error-line" role="alert" style={{ margin: 0 }}>
      {err}
    </p>
  );
};

const ReplyError: React.FC<{ m: Model; error: string | null | undefined; what: string }> = ({ m, error, what }) => {
  if (!error) return null;
  if (m.conn !== "open")
    return (
      <p className="status-line" role="status" style={{ margin: 0 }}>
        <span className="placeholder">[THE INSTRUMENT ANSWERS WHEN THE SERVER IS RUNNING]</span>
      </p>
    );
  return (
    <p className="error-line" role="alert" style={{ margin: 0 }}>
      The {what} did not come back: {error}
    </p>
  );
};

const RawToggle: React.FC<{ label: string; data: unknown }> = ({ label, data }) => {
  const [open, setOpen] = useState(false);
  return (
    <div className="stack" style={{ gap: 6 }}>
      <div>
        <button className="btn-ghost" type="button" aria-expanded={open} onClick={() => setOpen(!open)}>
          {open ? `hide ${label}` : `show ${label}`}
        </button>
      </div>
      {open ? <pre className="raw">{JSON.stringify(data, null, 2)}</pre> : null}
    </div>
  );
};

const FaceMark: React.FC<{ filled: boolean; size?: number }> = ({ filled, size = 12 }) => (
  <svg width={size} height={size} viewBox="0 0 12 12" aria-hidden="true" style={{ flex: "0 0 auto" }}>
    <path d="M6 0.8 L11.2 6 L6 11.2 L0.8 6 Z" fill={filled ? GILT : "none"} stroke={GILT} strokeWidth="1.2" />
  </svg>
);

function downloadJson(name: string, data: unknown) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/* -------------------------------------------------------------------- page */

export const NeuronFolio: React.FC<{ route: Route }> = ({ route }) => {
  if (route.name !== "neuron")
    return (
      <Page route={route} folio="folio iii">
        <Missing what="no neuron chosen" />
      </Page>
    );
  return <Folio key={`${route.layerNo}:${route.neuron}:${route.face}`} route={route} />;
};

const Folio: React.FC<{ route: NeuronRoute }> = ({ route }) => {
  const m = useModel(route);
  const [saved, update, where, reload] = useFolioSaved(route.layerNo, route.neuron);

  // read both faces on arrival, and again when the server comes back without an answer in hand
  const isOpen = m.conn === "open";
  const asked = useRef("");
  useEffect(() => {
    const k = `${m.idx}:${m.neuron}`;
    const st = useTheodoliteStore.getState();
    const cur = st.inspections[poleKey(m.idx, m.neuron)];
    if (asked.current !== k) {
      asked.current = k;
      st.inspectPole(m.idx, m.neuron, m.focusFace);
      return;
    }
    if (isOpen && !cur?.data && !cur?.pending) st.inspectPole(m.idx, m.neuron, m.focusFace);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [m.idx, m.neuron, isOpen]);

  return (
    <Page route={route} folio="folio iii">
      <Breadcrumb m={m} />
      <Hero m={m} saved={saved} update={update} where={where} />
      <NeuronApparatus m={m} saved={saved} where={where} />
      <Faces m={m} />
      <Says m={m} />
      <Listen m={m} />
      <Neighbours m={m} />
      <TestGuess m={m} />
      <DoMore m={m} />
      <Notes m={m} saved={saved} update={update} where={where} reload={reload} />
    </Page>
  );
};

/* -------------------------------------------------------------- the head */

const Breadcrumb: React.FC<{ m: Model }> = ({ m }) => (
  <nav
    aria-label="Breadcrumb"
    style={{
      display: "flex",
      flexWrap: "wrap",
      alignItems: "center",
      gap: "0 12px",
      padding: "4px 0",
      fontVariantCaps: "all-small-caps",
      letterSpacing: "0.18em",
      fontSize: "16px",
      ...NUM,
    }}
  >
    <a href={href.instrument()} style={{ display: "inline-flex", alignItems: "center", minHeight: "44px" }}>
      instrument
    </a>
    <span aria-hidden="true" style={{ color: GILT }}>
      ›
    </span>
    <a href={href.layer(m.layerNo)} style={{ display: "inline-flex", alignItems: "center", minHeight: "44px" }}>
      layer {m.layerNo}
    </a>
    <span aria-hidden="true" style={{ color: GILT }}>
      ›
    </span>
    <span aria-current="location" style={{ color: INK }}>
      neuron {m.neuron}
    </span>
  </nav>
);

const Particular: React.FC<{
  dt: string;
  first?: boolean;
  last?: boolean;
  ddStyle?: React.CSSProperties;
  children: React.ReactNode;
}> = ({ dt, first, last, ddStyle, children }) => (
  <div
    style={{
      display: "grid",
      gridTemplateColumns: "118px minmax(0, 1fr)",
      gap: "2px 18px",
      alignItems: "baseline",
      padding: "11px 0",
      borderTop: `1px solid ${first ? HAIR : HAIR_FAINT}`,
      borderBottom: last ? `1px solid ${HAIR}` : undefined,
    }}
  >
    <dt style={{ fontVariantCaps: "all-small-caps", letterSpacing: "0.16em", fontSize: "15px", color: MUTED }}>{dt}</dt>
    <dd style={{ margin: 0, fontSize: "18px", ...ddStyle }}>{children}</dd>
  </div>
);

const Hero: React.FC<{ m: Model; saved: FolioSaved; update: Upd; where: FolioWhere }> = ({ m, saved, update, where }) => {
  const [renaming, setRenaming] = useState(false);
  const [draft, setDraft] = useState("");
  const kept = saved.nick ?? m.defaultNick;
  const shown = renaming ? draft : kept ?? `neuron ${m.neuron}`;
  const startRename = () => {
    setDraft(kept ?? "");
    setRenaming(true);
  };
  const doneRename = () => {
    const t = draft.trim();
    update((p) => ({ ...p, nick: t && t !== m.defaultNick ? t : null }));
    setRenaming(false);
  };
  const s = m.sighting;
  const { lens } = useReadingLens(s);
  const pieces = useAnswerPieces(s);
  const words = m.focus.lens?.slice(0, 3).map((t) => t.token) ?? null;
  const btn: React.CSSProperties = { fontSize: "16px", padding: "0 18px" };

  let firedLine: React.ReactNode;
  if (!s) firedLine = <Missing />;
  else if (m.fired)
    firedLine = (
      <>
        As the model read the word {m.readAt}
        {m.inReply}, this neuron fired, the {m.rw.nth(m.pole!.rank)} in its layer.
      </>
    );
  else if (m.poles.length)
    firedLine = (
      <>
        As the model read the word {m.readAt}
        {m.inReply}, this neuron was {m.rw.notAmong} in its layer, so its strength there was not recorded.
      </>
    );
  else
    firedLine = (
      <>
        As the model read the word {m.readAt}
        {m.inReply}, the instrument listed no neurons at this layer: <Missing />
      </>
    );

  return (
    <section
      aria-labelledby="folio-title"
      style={{ display: "flex", flexWrap: "wrap", gap: "36px 72px", padding: "22px 0 48px", alignItems: "flex-start" }}
    >
      <div style={{ flex: "999 1 620px", minWidth: 0, display: "flex", flexDirection: "column", gap: "16px" }}>
        <p
          style={{
            margin: 0,
            fontSize: "16px",
            fontVariantCaps: "all-small-caps",
            letterSpacing: "0.22em",
            color: MUTED,
            ...NUM,
          }}
        >
          a neuron’s folio · neuron {m.neuron} · layer {m.layerNo} of sixty
        </p>
        <h1
          id="folio-title"
          style={{
            margin: 0,
            fontWeight: 400,
            fontStyle: "italic",
            fontSize: "clamp(46px, 6.4vw, 84px)",
            lineHeight: 1.04,
            letterSpacing: "-0.01em",
            color: INK,
            overflowWrap: "anywhere",
          }}
        >
          {shown || `neuron ${m.neuron}`}
        </h1>
        <svg width="260" height="14" viewBox="0 0 260 14" aria-hidden="true" style={{ display: "block", maxWidth: "100%" }}>
          <line x1="0" y1="7" x2="116" y2="7" stroke={GILT} strokeWidth="0.8" />
          <path d="M130 1 L136 7 L130 13 L124 7 Z" fill={GILT} />
          <line x1="144" y1="7" x2="260" y2="7" stroke={GILT} strokeWidth="0.8" />
        </svg>
        {!renaming ? (
          <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "6px 18px" }}>
            <span style={{ fontSize: "16px", fontStyle: "italic", color: MUTED }}>
              {kept && where.kind === "server" && where.nameBy === "scientist" && saved.nick
                ? "a name the scientist gave it in the atlas, which you kept: a guess, not a finding"
                : kept
                  ? "a name you gave it, not one the model knows"
                  : "not yet named; any name you give it is yours, not one the model knows"}
              {where.kind === "server" ? "; the atlas shows it too" : where.kind === "browser" ? "; kept in this browser" : ""}
            </span>
            <button
              type="button"
              className="btn-ghost"
              onClick={startRename}
              aria-label={kept ? "Rename this neuron" : undefined}
              style={btn}
            >
              {kept ? "rename" : "name it"}
            </button>
          </div>
        ) : (
          <div style={{ display: "flex", flexWrap: "wrap", alignItems: "flex-end", gap: "10px 18px" }}>
            <div style={{ display: "flex", flexDirection: "column", gap: "2px", flex: "1 1 280px", minWidth: 0, maxWidth: "460px" }}>
              <label htmlFor="nick" style={{ fontVariantCaps: "all-small-caps", letterSpacing: "0.16em", fontSize: "15px", color: MUTED }}>
                its name in your notes
              </label>
              <input
                id="nick"
                type="text"
                value={draft}
                maxLength={MAX_NAME}
                autoFocus
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") doneRename();
                }}
                style={{
                  fontFamily: "inherit",
                  fontStyle: "italic",
                  fontSize: "26px",
                  color: INK,
                  background: "transparent",
                  border: 0,
                  borderBottom: "1px solid #C9BEAC",
                  padding: "2px 0 6px",
                  width: "100%",
                  boxSizing: "border-box",
                }}
              />
            </div>
            <button type="button" className="btn-ghost" onClick={doneRename} style={btn}>
              keep this name
            </button>
          </div>
        )}
        <p style={{ margin: "6px 0 0", maxWidth: "660px" }}>
          One of 21,504 neurons in layer {m.layerNo}. {firedLine}{" "}
          {(m.focus.lens || m.focus.eye) && m.focusGate.heldBack ? (
            // the label gate holds the words back (§ii says why, and its apparatus keeps them)
            <>{heldLine(m.focusGate)}</>
          ) : m.recorded ? (
            <>
              Through the {lens.name} lens its positive face reads as English “Am”, Polish “jestem” and Spanish “Somos”; to the naked eye,
              as Polish “jestem”, Czech “jsem” and French “êtes”. Both readings are forms of one verb, <i>to be</i>, in several tongues.
            </>
          ) : words ? (
            <>
              Through the lens, its {faceWord(m.focusFace)} face writes {andList(words.map(wp))}.
            </>
          ) : null}
        </p>
        <p style={{ margin: 0, maxWidth: "660px", fontStyle: "italic", color: MUTED }}>
          {m.recorded && !m.focusGate.heldBack
            ? "One small cell of an unfamiliar model. At layers 41 to 58, a third to two-thirds of such a reading’s top ten words were among those the neuron pushed when the study made it louder: a guide to what it pushes toward, not to what it does in running text."
            : "One small cell of an unfamiliar model."}
        </p>
        {m.pole && m.pole.face !== m.routeFace ? (
          <p className="status-line" role="status" style={{ margin: 0, maxWidth: "660px" }}>
            You opened its {faceWord(m.routeFace)} face; at this word it fired on its {faceWord(m.pole.face)} face, so this folio reads that
            face first.
          </p>
        ) : null}
        <Rubric style={{ margin: "2px 0 0", maxWidth: "640px" }}>
          This folio gathers everything the instrument read about one neuron. Read it from the top; the red lines tell you what to do next, and
          how far to trust what you see.
        </Rubric>
      </div>
      <aside
        aria-labelledby="part-head"
        style={{ flex: "1 1 380px", minWidth: 0, display: "flex", flexDirection: "column", gap: "8px", paddingTop: "6px" }}
      >
        <h2
          id="part-head"
          style={{ margin: "0 0 4px", fontSize: "14px", fontWeight: 500, fontVariantCaps: "all-small-caps", letterSpacing: "0.2em", color: INK }}
        >
          the particulars
        </h2>
        <dl style={{ margin: 0, display: "flex", flexDirection: "column" }}>
          <Particular dt="you asked" first>
            {s ? `“${s.question}”` : <Missing />}
          </Particular>
          <Particular dt={m.fired ? "fired at" : "read at"}>
            {s && s.readAt != null ? (
              <>
                the word <span style={{ borderBottom: `2px solid ${GILT}` }}>{m.readAt}</span>
                {m.inReply}
                {chosenPhrase(s) ? `, ${chosenPhrase(s)}` : ""}
              </>
            ) : (
              <Missing />
            )}
          </Particular>
          <Particular dt="it went on" ddStyle={{ fontStyle: "italic" }}>
            {s && pieces && pieces.length ? (
              <>
                “…
                <AnswerMarks s={s} pieces={pieces} />”
                <span style={{ fontStyle: "normal", fontSize: "15px", color: MUTED }}>
                  {" "}
                  · {s.tokenIndex >= 1 ? "gilt: the word-piece read · " : ""}red: the one it chose there
                </span>
              </>
            ) : s && s.continuation.trim() ? (
              `“… ${s.continuation.trim()}”`
            ) : (
              <Missing />
            )}
            {s?.generation && s.generation.source !== "recorded" && !s.generation.greedy ? (
              // drawn by chance, the answer is one draw of many: said beside it, as on every page that shows one
              <span style={{ fontStyle: "normal", fontSize: "15px", color: MUTED }}>
                {" "}
                · its answer was <OneDrawLine generation={s.generation} />
              </span>
            ) : null}
            {(s?.generation?.presencePenalty ?? 0) !== 0 ? (
              <span style={{ fontStyle: "normal", fontSize: "15px", color: MUTED }}>
                {" "}
                <PenaltyLine penalty={s!.generation!.presencePenalty as number} />
              </span>
            ) : null}
            {s?.isNudged ? (
              <span style={{ fontStyle: "normal", color: "var(--rubric)" }}>
                {" "}
                · nudged: this sighting was taken with a <a href={href.nudge()}>nudge</a> armed, so what this folio reads of it is the nudged
                model, not its plain behaviour; only its readers are traced without the nudge, on this very text
              </span>
            ) : null}
          </Particular>
          <Particular dt={m.byWrite ? "write size" : "loudness"}>
            {m.pole && m.byWrite ? (
              <>
                the {m.rw.nth(m.pole.rank)} of the <span style={NUM}>21,504</span> in its layer
                {m.loudRank != null ? (
                  <span style={{ color: MUTED }}> ({loudPhrase(m.loudRank)} by |a| of the fifteen listed)</span>
                ) : null}
              </>
            ) : m.pole ? (
              <>
                {m.loudRank === 0 ? "the loudest" : loudPhrase(m.loudRank ?? m.pole.rank)} of the <span style={NUM}>21,504</span> in its
                layer
              </>
            ) : m.poles.length ? (
              <>
                {m.rw.notAmong} of the <span style={NUM}>21,504</span> in its layer
              </>
            ) : (
              <Missing />
            )}
          </Particular>
          <Particular dt="its layer" last ddStyle={{ display: "flex", flexDirection: "column", gap: "2px" }}>
            <span style={{ fontSize: "18px" }}>
              <span style={NUM}>{m.layerNo}</span> of sixty{" "}
              {isGlobalIdx(m.idx) ? (
                <>
                  <span aria-hidden="true" style={{ color: GILT }}>
                    ✦
                  </span>{" "}
                  global attention
                </>
              ) : (
                "local attention"
              )}
            </span>
            <span style={{ fontSize: "15px", fontStyle: "italic", color: MUTED }}>
              {isGlobalIdx(m.idx) ? "looks back over the whole conversation" : "looks back over the last 1,024 word-pieces"}
            </span>
            <span style={{ fontSize: "15px", color: MUTED }}>
              {m.layer?.ffnShare != null ? (
                <>
                  <span style={{ color: GILT_TEXT, fontWeight: 600, ...NUM }}>{pct(m.layer.ffnShare)}</span> recalled from memory{" "}
                  <span style={{ fontStyle: "italic" }}>(feed-forward)</span>,{" "}
                  <span style={{ color: LAPIS, fontWeight: 600, ...NUM }}>{pct(1 - m.layer.ffnShare)}</span> gathered{" "}
                  <span style={{ fontStyle: "italic" }}>(attention)</span>
                </>
              ) : (
                <>
                  the split between remembering and gathering: <Missing />
                </>
              )}
            </span>
          </Particular>
        </dl>
        <a href={href.layer(m.layerNo)} style={{ alignSelf: "flex-start", fontSize: "17px", padding: "10px 0" }}>
          open layer {m.layerNo}’s leaf →
        </a>
      </aside>
    </section>
  );
};

/** The raw readout: recorded or sighting values only. */
function rawReadout(m: Model) {
  if (m.recorded) return RAW_READOUT;
  const s = m.sighting;
  return {
    source: s ? s.provenance : null,
    nudged: s ? s.isNudged : null,
    layer: m.layerNo,
    index: m.idx,
    neuron: m.neuron,
    token_index: s?.tokenIndex ?? null,
    read_at: s?.readAt ?? null,
    position: s?.readPos ?? null,
    n_tokens: s?.nTokens ?? null,
    chosen: s?.chosen ? { token_id: s.chosen.id, token: s.chosen.token, model_prob: s.chosen.modelProb, model_rank: s.chosen.modelRank } : null,
    generation: s?.generation ?? null,
    context_ids: s?.contextIds ?? null,
    signed_activation: m.pole ? m.pole.act : null,
    polarity: m.pole ? (m.pole.face > 0 ? "+" : "-") : null,
    rank_by_abs: m.loudRank != null ? m.loudRank + 1 : null,
    // the same rank, named for what it is: by |a| among the fifteen listed, whichever ranking listed them
    rank_by_abs_among_listed: m.loudRank != null ? m.loudRank + 1 : null,
    rank_in_list: m.pole ? m.pole.rank + 1 : null,
    pole_ranking: m.byWrite ? "write" : "activation",
    pole_label_jpp: m.pole?.lens ? m.pole.lens.map((t) => [t.token, t.logit]) : null,
    pole_label_naive: m.pole?.eye ? m.pole.eye.map((t) => [t.token, t.logit]) : null,
    layer_poles_top15: m.poles.map((p) => [p.neuron, p.act]),
    ffn_dominance: { corrected: m.layer?.ffnShare ?? null, tool_before_fix: m.layer?.ffnShareBeforeFix ?? null },
  };
}

const NeuronApparatus: React.FC<{ m: Model; saved: FolioSaved; where: FolioWhere }> = ({ m, saved, where }) => {
  const [rawOpen, setRawOpen] = useState(false);
  const s = m.sighting;
  const chosen = chosenOf(s);
  const L = m.layer;
  const raw = rawReadout(m);
  const exportFolio = () => {
    const st = useTheodoliteStore.getState();
    const key = poleKey(m.idx, m.neuron);
    downloadJson(`folio-layer${m.layerNo}-neuron${m.neuron}.json`, {
      readout: raw,
      inspect_pole: st.inspections[key]?.data ?? null,
      // only a trace taken for the sighting shown now belongs with this readout
      trace_readers: st.readers[key]?.sightingId === st.sighting?.id ? (st.readers[key]?.data ?? null) : null,
      test_hypothesis: st.hypotheses[key]?.data ?? null,
      your_notes: saved,
      your_notes_kept:
        where.kind === "server"
          ? { on: "the server", field_book: where.file, name_in: where.notesFile }
          : where.kind === "browser"
            ? { on: "this browser", why: where.why }
            : null,
    });
  };
  const notRecorded: string[] = ["the gate g and up-projection u apart"];
  if (L?.streamStrength == null) notRecorded.push(`‖h‖ at layer ${m.layerNo}`);
  if (L?.layerScalar == null) notRecorded.push("the layer scalar s");

  return (
    <Apparatus
      title={
        <>
          neuron {m.neuron}, layer {m.layerNo} <span style={{ letterSpacing: "0.06em" }}>(index {m.idx})</span>
        </>
      }
      label="Apparatus for this neuron"
      style={{ padding: "12px 0 30px" }}
    >
      <dl className="dl num">
        <dt>address</dt>
        <dd>
          layer index {m.idx}, slot {m.neuron} of the layer’s 21,504 GeGLU feed-forward neurons (GLU variants, Shazeer 2020) [15]; the model
          holds 60 × 21,504 = 1,290,240 neurons, so 2,580,480 poles.
        </dd>
        <dt>sighting</dt>
        <dd>
          {!s ? (
            <Missing />
          ) : m.isFixture ? (
            <>
              the study’s E6 demonstration sweep, <span className="mono">E6_tool_audit_02_sweep_demo.json</span>, key{" "}
              <span className="mono">prepend_bos=True</span>: {s.nTokens} word-pieces, read at the last, “{s.readAt}” (position {s.readPos},
              counting from 0). Start marks <span className="mono">[{(s.firstIds ?? []).join(", ")}]</span>: the doubled start mark of the
              instrument before the single-BOS fix.
            </>
          ) : (
            <>
              a live sighting from the running server, read at position {s.readPos ?? <Missing what="not sent" />} (counting from 0),{" "}
              <span className="mono-wrap">{JSON.stringify(s.readAt ?? "")}</span>
              {s.tokenIndex >= 1
                ? `, the ${nth(s.tokenIndex)} word-piece of its own answer`
                : s.promptLen != null && s.readPos === s.promptLen - 1
                  ? ", the last of the prompt"
                  : ""}
              ; {s.nTokens ?? <Missing what="not sent" />} word-pieces read up to there
              {chosen ? `; it chose ${JSON.stringify(chosen.token)} (id ${chosen.id}) there` : ""}. Text sent:{" "}
              <span className="mono-wrap">{JSON.stringify(s.textSent ?? "")}</span>
              {s.tokenIndex >= 1 ? (
                <>
                  , then its own answer up to there:{" "}
                  {s.answerStrs ? (
                    <span className="mono-wrap">{JSON.stringify(s.answerStrs.slice(0, s.tokenIndex).join(""))}</span>
                  ) : (
                    <Missing what="not sent" />
                  )}
                </>
              ) : null}
              . The exact ids read, up to the one read:{" "}
              {s.contextIds ? <span className="mono-wrap">[{s.contextIds.join(", ")}]</span> : <Missing what="not sent" />}
            </>
          )}
        </dd>
        <dt>strength</dt>
        <dd>
          {m.act !== null && m.pole ? (
            <>
              a = <span className="mono">mlp_act</span>[{m.idx}, {s?.readPos ?? "t"}, {m.neuron}] = {signedRaw(m.act)}, so |a| ={" "}
              {String(Math.abs(m.act))}, {ordinal(m.pole.rank)} of {m.rw.fifteen};{" "}
              {m.loudest && m.loudest.neuron !== m.neuron ? (
                <>
                  the loudest{m.byWrite ? " of the fifteen listed" : ""}, neuron {m.loudest.neuron}, has a ={" "}
                  {signedRaw(m.recordedLayer ? ACTS[0] : m.loudest.act)} (the bar stands at {m.barPct}%).
                </>
              ) : (
                <>it is the loudest{m.byWrite ? " of the fifteen listed" : ""} (the bar stands full).</>
              )}{" "}
              In the raw readout, <span className="mono">rank_by_abs</span> (and{" "}
              <span className="mono">rank_by_abs_among_listed</span>) count by |a| among the fifteen listed;{" "}
              <span className="mono">rank_in_list</span> follows <span className="mono">pole_ranking</span>.
            </>
          ) : (
            <>
              {m.rw.notAmong} at this word-piece, so a was not recorded: <Missing />
            </>
          )}
        </dd>
        <dt>layer split</dt>
        <dd>
          {L?.ffnShare != null ? (
            <>
              remembering {(m.recordedLayer ? FFN : L.ffnShare).toFixed(4)}
              {m.isFixture ? " after the V7 fix" : ""} (shown as {pct(L.ffnShare)})
              {L.ffnShareBeforeFix != null ? (
                <>; the instrument read {(m.recordedLayer ? FFN_TOOL : L.ffnShareBeforeFix).toFixed(4)} before it.</>
              ) : (
                "."
              )}
              {L.attnNorm != null && L.ffnNorm != null ? (
                <>
                  {" "}
                  From ‖Δ attention‖ = {L.attnNorm.toFixed(2)} and ‖Δ feed-forward‖ = {L.ffnNorm.toFixed(2)}.
                </>
              ) : null}
            </>
          ) : (
            <Missing />
          )}
        </dd>
        {L?.layerScalar != null ? (
          <>
            <dt>layer scalar</dt>
            <dd>
              s = {L.layerScalar}
              {m.isFixture ? (
                <>
                  , a model constant (<span className="mono">E6_tool_audit_01_basics.json</span>)
                </>
              ) : null}
            </dd>
          </>
        ) : null}
        {L?.streamStrength != null ? (
          <>
            <dt>stream strength</dt>
            <dd>‖h‖ = {L.streamStrength.toFixed(1)} at layer {m.layerNo}</dd>
          </>
        ) : null}
        <dt>not recorded</dt>
        <dd style={{ margin: 0, color: MUTED }}>
          {andList(notRecorded)}: <Missing />
        </dd>
      </dl>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "8px 12px", paddingTop: "4px" }}>
        <button className="btn-ghost" type="button" onClick={() => setRawOpen(!rawOpen)} aria-expanded={rawOpen}>
          {rawOpen ? "hide the raw readout" : "show the raw readout"}
        </button>
        <button className="btn-ghost" type="button" onClick={exportFolio}>
          export the folio (.json)
        </button>
      </div>
      {rawOpen ? (
        <pre className="raw" style={{ maxHeight: "320px" }}>
          {JSON.stringify(raw, null, 2)}
        </pre>
      ) : null}
    </Apparatus>
  );
};

/* ------------------------------------------------------- i · its two faces */

const Faces: React.FC<{ m: Model }> = ({ m }) => {
  const inspectPole = useTheodoliteStore((s) => s.inspectPole);
  const limb = useMemo(() => {
    let d = "";
    for (let deg = 0; deg < 360; deg += 5) {
      const a = (deg * Math.PI) / 180;
      const ri = deg % 30 === 0 ? 132 : 142;
      d += `M${f1(380 + 150 * Math.cos(a))},${f1(200 + 150 * Math.sin(a))} L${f1(380 + ri * Math.cos(a))},${f1(200 + ri * Math.sin(a))} `;
    }
    return d;
  }, []);

  const focusName = faceWord(m.focusFace);
  const otherName = faceWord(m.otherFace);
  // a face whose label the gate holds back shows "[held back]" at its needle end; the words come back, muted, with the
  // apparatus open (lib/labels.ts)
  const apparatus = useApparatus();
  const leftHeld = !!m.focus.lens && m.focusGate.heldBack;
  const rightHeld = !!m.other?.lens && m.otherGate.heldBack;
  const leftWords = m.focus.lens && (!leftHeld || apparatus) ? m.focus.lens.slice(0, 3).map((t) => t.token) : null;
  const rightWords = m.other?.lens && (!rightHeld || apparatus) ? m.other.lens.slice(0, 3).map((t) => t.token) : null;
  const longest = (ws: string[] | null) => (ws ? Math.max(...ws.map((w) => shortToken(w).length)) : 0);
  const leftSize = longest(leftWords) > 8 ? 20 : 24;
  const rightSize = longest(rightWords) > 8 ? 20 : 24;
  const strength = m.act !== null ? Math.abs(m.act).toFixed(2) : null;
  const loud = m.loudAct !== null ? m.loudAct.toFixed(2) : null;
  const pending = m.insp?.pending;
  const quoted = (ws: string[]) => andList(ws.map(wp));

  const aria =
    `A two-ended needle pivoting on an engraved dial. ` +
    (m.fired
      ? `The filled end, the ${focusName} face, fired at the word ${m.readAt} with strength ${strength}`
      : m.notAmong
        ? `The left end, the ${focusName} face, was ${m.rw.notAmong} at this word`
        : `The left end, the ${focusName} face; its strength at this word was not recorded`) +
    (leftHeld && !apparatus
      ? `; its words are held back here. `
      : leftWords
        ? ` and writes ${quoted(leftWords)}${leftHeld ? ", words held back from the main reading" : ""}. `
        : `; its words were not read. `) +
    `The hollow end, the ${otherName} face, ${m.fired ? "was silent" : "is the other way to fire"}` +
    (rightHeld && !apparatus
      ? ` and its words are held back here.`
      : rightWords
        ? ` and writes ${quoted(rightWords)}${rightHeld ? ", words held back from the main reading" : ""}.`
        : ` and its words were not read in this sighting.`);

  return (
    <section aria-labelledby="faces-head" style={SECTION}>
      <div className="row-between">
        <h2 className="h-sec" id="faces-head">
          i · its two faces
        </h2>
        <span style={ASIDE}>one neuron, two ways to fire</span>
      </div>
      <p style={{ margin: 0, maxWidth: "760px" }}>
        Every neuron in this model can fire two ways, positive or negative, and each way pushes the model toward different words. These are its
        two faces (Cammarata calls them poles), and this folio reads each face on its own.
      </p>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "32px 60px", alignItems: "center" }}>
        <figure style={{ flex: "999 1 600px", minWidth: 0, margin: 0 }}>
          <svg
            viewBox="0 0 760 400"
            role="img"
            aria-label={aria}
            style={{ width: "100%", maxWidth: "820px", height: "auto", display: "block", margin: "0 auto" }}
          >
            <circle cx="380" cy="200" r="150" fill="none" stroke={GILT} strokeWidth="1.2" />
            <circle cx="380" cy="200" r="142" fill="none" stroke={GILT} strokeOpacity="0.55" strokeWidth="0.8" />
            <path d={limb} fill="none" stroke={GILT} strokeWidth="0.9" />
            <circle cx="380" cy="200" r="104" fill="none" stroke={INK} strokeOpacity="0.12" strokeWidth="0.8" />
            <circle cx="380" cy="200" r="58" fill="none" stroke={INK} strokeOpacity="0.12" strokeWidth="0.8" />
            <line x1="380" y1="44" x2="380" y2="356" stroke={INK} strokeOpacity="0.3" strokeWidth="1" />
            <path d="M380 22 L387 34 L380 46 L373 34 Z" fill={GILT} />
            {m.fired ? (
              <path d="M200,200 L380,189 L380,211 Z" fill={GILT} stroke={GILT_TEXT} strokeWidth="0.8" strokeLinejoin="round" />
            ) : (
              <path d="M200,200 L380,189 L380,211 Z" fill="#F6F0E3" stroke={INK} strokeWidth="1.2" strokeLinejoin="round" />
            )}
            <path d="M560,200 L380,189 L380,211 Z" fill="#F6F0E3" stroke={INK} strokeWidth="1.2" strokeLinejoin="round" />
            <line x1="392" y1="200" x2="548" y2="200" stroke={INK} strokeOpacity="0.3" strokeWidth="0.6" />
            {!m.fired ? <line x1="212" y1="200" x2="368" y2="200" stroke={INK} strokeOpacity="0.3" strokeWidth="0.6" /> : null}
            <circle cx="380" cy="200" r="10" fill="#F6F0E3" stroke={INK} strokeWidth="1.2" />
            <circle cx="380" cy="200" r="3.4" fill={RED} />
            <g fontFamily={SERIF} textAnchor="middle">
              <text x="100" y="106" fontSize="12" letterSpacing="2" fill={MUTED}>
                {focusName.toUpperCase()} FACE
              </text>
              <text x="668" y="170" fontSize="12" letterSpacing="2" fill={MUTED}>
                {otherName.toUpperCase()} FACE
              </text>
              <text x="308" y="240" fontSize="15" fontStyle="italic" fill={m.fired ? INK : MUTED} style={NUM}>
                {m.fired ? `fired here · ${strength}` : "not recorded here"}
              </text>
              {m.fired ? (
                <text x="450" y="240" fontSize="15" fontStyle="italic" fill={MUTED}>
                  silent at this word
                </text>
              ) : null}
              <text x="380" y="388" fontSize="12" letterSpacing="2.4" fill={MUTED} style={NUM}>
                NEURON {m.neuron} · LAYER {m.layerNo}
              </text>
            </g>
            {leftWords ? (
              <>
                <path
                  d="M200,200 L150,140 M200,200 L144,200 M200,200 L150,260"
                  fill="none"
                  stroke={GILT_TEXT}
                  strokeOpacity="0.55"
                  strokeWidth="0.8"
                  strokeDasharray="2 3"
                />
                <g fontFamily={SERIF} fontStyle="italic" fontSize={leftSize} fill={leftHeld ? MUTED : INK} textAnchor="end">
                  {leftWords.map((w, i) => (
                    <text key={i} x={i === 1 ? 136 : 142} y={[146, 208, 270][i]}>
                      “{shortToken(w)}”
                    </text>
                  ))}
                  <text x="136" y="306" fontSize="13" fill={MUTED}>
                    {leftHeld ? "through the lens · held back" : "through the lens"}
                  </text>
                </g>
              </>
            ) : (
              <>
                <path d="M200,200 L156,200" fill="none" stroke={MUTED} strokeOpacity="0.6" strokeWidth="0.8" strokeDasharray="2 3" />
                <g fontFamily={SERIF} fontSize="12" letterSpacing="1.2" fill={MUTED} textAnchor="end">
                  {leftHeld ? (
                    <text x="148" y="205">
                      [HELD BACK]
                    </text>
                  ) : (
                    <>
                      <text x="148" y="196">
                        [NOT READ IN
                      </text>
                      <text x="148" y="214">
                        THIS SIGHTING]
                      </text>
                    </>
                  )}
                </g>
              </>
            )}
            {rightWords ? (
              <>
                <path
                  d="M560,200 L610,140 M560,200 L616,200 M560,200 L610,260"
                  fill="none"
                  stroke={MUTED}
                  strokeOpacity="0.6"
                  strokeWidth="0.8"
                  strokeDasharray="2 3"
                />
                <g fontFamily={SERIF} fontStyle="italic" fontSize={rightSize} fill={MUTED} textAnchor="start">
                  {rightWords.map((w, i) => (
                    <text key={i} x={i === 1 ? 624 : 618} y={[146, 208, 270][i]}>
                      “{shortToken(w)}”
                    </text>
                  ))}
                  <text x="624" y="306" fontSize="13">
                    {rightHeld ? "through the lens · held back" : "through the lens"}
                  </text>
                </g>
              </>
            ) : (
              <>
                <path d="M560,200 L604,200" fill="none" stroke={MUTED} strokeOpacity="0.6" strokeWidth="0.8" strokeDasharray="2 3" />
                <g fontFamily={SERIF} fontSize="12" letterSpacing="1.2" fill={MUTED}>
                  {rightHeld ? (
                    <text x="612" y="205">
                      [HELD BACK]
                    </text>
                  ) : (
                    <>
                      <text x="612" y="196">
                        [NOT READ IN
                      </text>
                      <text x="612" y="214">
                        THIS SIGHTING]
                      </text>
                    </>
                  )}
                </g>
              </>
            )}
          </svg>
          <figcaption
            style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: "6px 22px", fontSize: "15px", color: MUTED, marginTop: "6px" }}
          >
            {m.fired ? (
              <>
                <span>the filled end fired at the word {m.readAt}</span>
                <span aria-hidden="true" style={{ color: GILT }}>
                  ◆
                </span>
                <span>the hollow end was silent</span>
              </>
            ) : (
              <>
                <span>the left end is the face you opened</span>
                <span aria-hidden="true" style={{ color: GILT }}>
                  ◆
                </span>
                <span>
                  {m.notAmong ? `neither end was among ${m.rw.fifteen} at this word` : "its strength at this word was not recorded"}
                </span>
              </>
            )}
          </figcaption>
        </figure>
        <div style={{ flex: "1 1 340px", minWidth: 0, display: "flex", flexDirection: "column", gap: "22px" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: "10px", paddingBottom: "20px", borderBottom: `1px solid ${HAIR_SOFT}` }}>
            <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
              <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
                {m.fired ? (
                  <path d="M8 1 L15 8 L8 15 L1 8 Z" fill={GILT} />
                ) : (
                  <path d="M8 1.5 L14.5 8 L8 14.5 L1.5 8 Z" fill="none" stroke={GILT} strokeWidth="1.3" />
                )}
              </svg>
              <h3 className="h-sub">
                {focusName} face · the {m.fired ? "filled" : "left"} end
              </h3>
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", alignItems: "flex-end", gap: "6px 16px" }}>
              {strength !== null ? (
                <span style={{ fontSize: "60px", lineHeight: 0.9, fontWeight: 500, color: "#8A6A2C", letterSpacing: "-0.01em", ...NUM }}>
                  {strength}
                </span>
              ) : (
                <Missing />
              )}
              <span style={{ fontSize: "16px", color: MUTED, paddingBottom: "4px" }}>
                strength <span style={{ fontStyle: "italic" }}>(|activation|)</span>
              </span>
            </div>
            {m.barPct !== null && loud !== null ? (
              <>
                <div
                  role="img"
                  aria-label={`Strength ${strength}, against ${loud} for the loudest neuron ${m.byWrite ? "of the fifteen listed" : "in this layer"}`}
                  style={{ height: "3px", background: HAIR_FAINT, position: "relative" }}
                >
                  <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${m.barPct}%`, background: GILT }} />
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", gap: "12px", fontSize: "14px", color: MUTED, ...NUM }}>
                  <span>0</span>
                  <span>
                    the loudest {m.byWrite ? "of the fifteen listed" : "in this layer"}, {loud}
                  </span>
                </div>
              </>
            ) : null}
            <p style={{ margin: 0, fontSize: "17px" }}>
              {m.fired ? (
                <>Fired here, at the word {m.readAt}. </>
              ) : m.notAmong ? (
                <>
                  {cap(m.rw.notAmong)} at the word {m.readAt}, so its strength here was not recorded.{" "}
                </>
              ) : (
                <>Its strength at this word was not recorded. </>
              )}
              {leftHeld ? (
                <>{heldLine(m.focusGate)}</>
              ) : leftWords ? (
                <>It writes toward {quoted(leftWords)}.</>
              ) : (
                <>
                  What it writes: <Missing what={awaitWhat(m, pending)} />
                </>
              )}
            </p>
            {leftHeld ? (
              // the line above says why; the apparatus brings the words back (and §ii's rubric explains)
              <IfApparatus>
                <HeldBackWords
                  gate={m.focusGate}
                  words={m.focus.lens?.slice(0, 3).map((t) => t.token) ?? null}
                  eye={m.focus.eye?.slice(0, 3).map((t) => t.token) ?? null}
                  compact
                  style={{ fontSize: 16 }}
                />
              </IfApparatus>
            ) : null}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
              <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
                <path d="M8 1.5 L14.5 8 L8 14.5 L1.5 8 Z" fill="none" stroke={GILT} strokeWidth="1.3" />
              </svg>
              <h3 className="h-sub">{otherName} face · the hollow end</h3>
            </div>
            <p style={{ margin: 0, fontSize: "17px" }}>
              {m.fired ? "Silent at this word. " : ""}Fired, it would push the other way, away from the words above.
            </p>
            {m.other?.lens && rightHeld ? (
              <HeldBackWords
                gate={m.otherGate}
                words={m.other.lens.slice(0, 3).map((t) => t.token)}
                eye={m.other.eye?.slice(0, 3).map((t) => t.token) ?? null}
                size={20}
              />
            ) : m.other?.lens ? (
              <>
                <div
                  style={{
                    display: "flex",
                    flexWrap: "wrap",
                    gap: "4px 18px",
                    fontSize: "20px",
                    padding: "6px 0",
                    borderTop: `1px solid ${HAIR_FAINT}`,
                    borderBottom: `1px solid ${HAIR_FAINT}`,
                  }}
                >
                  {m.other.lens.slice(0, 3).map((t, i) => (
                    <span key={i} title={JSON.stringify(t.token)}>
                      {wp(t.token)}
                    </span>
                  ))}
                </div>
                <p style={{ margin: 0, fontSize: "15px", fontStyle: "italic", color: MUTED }}>
                  its three words, read through the lens
                  {m.other.eye ? <>; to the naked eye, {m.other.eye.slice(0, 3).map((t) => wp(t.token)).join(" ")}</> : null}
                </p>
              </>
            ) : (
              <>
                <p style={{ margin: 0, fontSize: "15px", fontVariantCaps: "all-small-caps", letterSpacing: "0.14em", color: MUTED }}>
                  {pending ? "[BEING READ…]" : "[NOT READ IN THIS SIGHTING]"}
                </p>
                <div>
                  <button
                    type="button"
                    className="btn-ghost"
                    disabled={pending}
                    onClick={() => inspectPole(m.idx, m.neuron, m.focusFace)}
                    style={{ fontSize: "16px", padding: "0 20px" }}
                  >
                    {pending ? "reading both faces…" : "read both faces"}
                  </button>
                </div>
                <ReplyError m={m} error={m.insp?.error} what="reading of both faces" />
                <ServerErrorWhilePending pending={pending} />
              </>
            )}
          </div>
        </div>
      </div>
      <IfRubrics>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "10px 48px" }}>
          <p className="rubric" style={{ flex: "1 1 340px", minWidth: 0, maxWidth: "620px" }}>
            In the study the two faces of one neuron woke for unrelated reasons, so the name you give it belongs to the face that fired, not to the
            whole neuron.
          </p>
          <p className="rubric" style={{ flex: "1 1 340px", minWidth: 0, maxWidth: "620px" }}>
            What wakes a neuron is harder to read than what it writes: the study found the instrument’s input-side labels nearly useless, so this
            folio shows only what it writes. To learn what wakes it, put a guess to the test, below.
          </p>
        </div>
      </IfRubrics>
      <PolesApparatus m={m} />
    </section>
  );
};

const PolesApparatus: React.FC<{ m: Model }> = ({ m }) => {
  const pol = m.focusFace > 0 ? "1.0" : "-1.0";
  const sign = m.focusFace > 0 ? "+" : "-";
  const code =
    "# MCP (backend/mcp_scientist.py)\n" +
    `inspect_pole(layer=${m.idx}, neuron_idx=${m.neuron}, polarity=${pol})\n\n` +
    "# HTTP\nPOST /api/scientist/inspect_pole\n" +
    `{"layer": ${m.idx}, "neuron_idx": ${m.neuron}, "polarity": ${pol}}\n\n` +
    "# the vault (this project’s LQL-like dialect)\n" +
    `DESCRIBE POLE LAYER ${m.idx} SLOT ${m.neuron} POLARITY ${sign}`;
  const ins = m.inspection;
  // the atlas's passages index, when it is built: each pole's strongest passages and the word-pieces it reads and that follow
  const pas = usePassages(m.idx, m.neuron);
  const stored = pas && pas["+"].ready ? pas : null;
  const wordsOf = (list: { tok: string; n: number; lift: number | null }[] | undefined) =>
    list && list.length ? list.slice(0, 6).map((t) => `${wp(t.tok)} ${t.n}${t.lift != null ? ` (×${t.lift})` : ""}`).join(", ") : "none";
  const ctxCol = (title: string, face: "+" | "-") => {
    const r = stored ? stored[face] : null;
    const tops = (r?.passages ?? []).filter((p) => p.kind === "top").slice(0, 5);
    const cells: [string, React.ReactNode][] = r
      ? [
          [
            "top passages",
            tops.length ? (
              <ol style={{ margin: 0, paddingLeft: "1.2em", display: "flex", flexDirection: "column", gap: "4px" }}>
                {tops.map((p, i) => {
                  const [pre, peak, post] = splitAtPeak(p.units);
                  return (
                    <li key={i} style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>
                      {p.cut[0] ? "" : "…"}
                      {pre}
                      <b style={{ color: GILT_TEXT }}>{peak}</b>
                      {post}
                      {p.cut[1] ? "" : "…"} <span style={{ color: MUTED }}>{p.act.toFixed(2)}</span>
                    </li>
                  );
                })}
              </ol>
            ) : (
              "it never fired on this face over the corpus"
            ),
          ],
          ["word-pieces it reads", wordsOf(r.reads)],
          ["word-pieces that follow", wordsOf(r.follows)],
        ]
      : ["top passages", "word-pieces it reads", "word-pieces that follow"].map((k) => [k, <span style={PH}>[CONTEXTS NOT YET COLLECTED]</span>]);
    return (
      <div style={{ flex: "1 1 300px", minWidth: 0, display: "flex", flexDirection: "column", gap: "4px" }}>
        <span style={{ fontVariantCaps: "all-small-caps", letterSpacing: "0.16em", color: INK }}>{title}</span>
        <dl style={{ margin: 0, display: "grid", gridTemplateColumns: "minmax(0, 170px) minmax(0, 1fr)", gap: "4px 14px" }}>
          {cells.map(([k, v]) => (
            <React.Fragment key={k}>
              <dt style={{ color: MUTED }}>{k}</dt>
              <dd style={{ margin: 0 }}>{v}</dd>
            </React.Fragment>
          ))}
        </dl>
      </div>
    );
  };
  return (
    <Apparatus title="the two poles" label="Apparatus for the two poles" style={{ gap: "10px" }}>
      <dl className="dl num">
        <dt>poles</dt>
        <dd>
          σ = +1 and σ = −1 of one GeGLU neuron [15], treated as the model’s native units after Cammarata, “Neural Networks Are Already Clean”
          (Goodfire 2026) [9]. The original instrument calls them “+North” and “−South”.
        </dd>
        <dt>activation</dt>
        <dd>
          a = GELU(g) · u, with g = ⟨w<sub>gate</sub>, x̃⟩, u = ⟨w<sub>up</sub>, x̃⟩ and x̃ the pre-feed-forward RMSNorm of the stream; the pole
          is σ = sign(a).{" "}
          {m.act !== null ? (
            <>
              Here a = {signedRaw(m.act)}, so σ = {m.act >= 0 ? "+1" : "−1"}.
            </>
          ) : (
            <>Here a was not recorded in this sighting.</>
          )}
        </dd>
        <dt>bipolarity</dt>
        <dd>
          real in Gemma-4, through the sign of the up-projection: at strong events the gate is open (g &gt; 0) in 72–100% of cases, and sign(a) =
          sign(u). The double-negative regime (g &lt; 0 and u &lt; 0; the term is from [9]) is under 1% of strong-event mass, but 13–33% of top-1%
          mass in sparse layers (REPORT §3.2).
        </dd>
        <dt>two jobs</dt>
        <dd>
          the poles of one neuron fire on unrelated inputs: their top input word-pieces overlap at the null rate (P@10 0.015 against 0.016), and
          their responses are anti-correlated (Spearman −0.36 to −0.54, layers 21–58). At layers 41–58 their outputs are mirror images (N1, Q4).
        </dd>
        <dt>input label</dt>
        <dd>
          withheld here. The instrument’s <span className="mono">gate_trigger_tokens</span> = unembed(w<sub>gate</sub> · J<sub>ℓ</sub>
          <sup>⊤</sup>) passes a read covector through the lens as if it were a state, and gives one label for two poles; it carries about 6–8% of
          the reproducible signal (REPORT §3.4, bug 7). <span className="mono">inspect_pole</span> still returns it.
        </dd>
        <dt>the fix</dt>
        <dd>
          stored activation contexts per pole: the top passages, and the word-pieces each pole reads and is followed by, kept for each pole
          separately from one streaming pass over a corpus (REPORT §3.3, §3.5 item 5). The atlas keeps them in its passages index (
          <span className="mono">scripts/atlas_build.py passages</span>: each pole’s sixteen strongest windows, one event a window, with the gate
          and up values there);{" "}
          {stored
            ? `they are below, read over ${stored["+"].tokens?.toLocaleString()} word-pieces of ${stored["+"].corpus}; the words each pole reads and that follow are counted ${stored["+"].words_note}.`
            : "it is not built on this server, so they are not collected below."}
          {stored?.["+"].stale ? <span style={{ color: RED }}> {stored["+"].stale}.</span> : null}
        </dd>
      </dl>
      <div
        role="group"
        aria-label={stored ? "Stored contexts per pole, from the atlas's passages index" : "Stored contexts per pole, not yet collected"}
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "10px 40px",
          padding: "10px 0",
          borderTop: `1px solid ${HAIR_SOFT}`,
          borderBottom: `1px solid ${HAIR_SOFT}`,
        }}
      >
        {ctxCol("positive pole · σ = +1", "+")}
        {ctxCol("negative pole · σ = −1", "-")}
      </div>
      <dl className="dl">
        <dt>by hand</dt>
        <dd style={{ margin: 0, display: "flex", flexDirection: "column", gap: "6px", minWidth: 0 }}>
          <span>
            “read both faces” makes this call (this folio makes it on arrival). It returns both poles’ top eight word-pieces through the lens and
            to the naked eye, the norms of w<sub>down</sub>, w<sub>gate</sub> and w<sub>up</sub>, and the withheld input label.
          </span>
          <pre style={PRE}>{code}</pre>
        </dd>
        <dt>the reply</dt>
        <dd style={{ margin: 0, display: "flex", flexDirection: "column", gap: "6px", minWidth: 0 }}>
          {ins ? (
            <>
              <span className="num">
                ‖w<sub>down</sub>‖ = {ins.wDownNorm ?? "?"}, ‖w<sub>gate</sub>‖ = {ins.wGateNorm ?? "?"}, ‖w<sub>up</sub>‖ = {ins.wUpNorm ?? "?"}
              </span>
              <span>
                positive pole, lens: {ins.pos.lens ? scoresText(ins.pos.lens) : "—"}; naked eye: {ins.pos.eye ? scoresText(ins.pos.eye) : "—"}
              </span>
              <span>
                negative pole, lens: {ins.neg.lens ? scoresText(ins.neg.lens) : "—"}; naked eye: {ins.neg.eye ? scoresText(ins.neg.eye) : "—"}
              </span>
              <span style={{ color: MUTED }}>
                the withheld input label, for the record only: {ins.gate ? scoresText(ins.gate) : "—"}
              </span>
              <RawToggle label="the raw reply" data={ins.raw} />
            </>
          ) : m.insp?.pending ? (
            <span className="status-line">asking the server…</span>
          ) : m.insp?.error ? (
            <span className="error-line">the server did not answer: {m.insp.error}</span>
          ) : (
            <Missing what="appears after reading both faces" />
          )}
        </dd>
      </dl>
    </Apparatus>
  );
};

/* --------------------------------------------------------- ii · what it says */

const WordRow: React.FC<{ t: Tok; shared: boolean }> = ({ t, shared }) => {
  const g = glossFor(t.token);
  const note = [g, shared ? "on both sides" : ""].filter(Boolean).join(" · ");
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "22px minmax(0, 1fr) 72px",
        gap: "0 12px",
        alignItems: "baseline",
        padding: "12px 0",
        borderBottom: `1px solid ${HAIR_FAINT}`,
      }}
    >
      {shared ? (
        <span aria-hidden="true" style={{ color: GILT, fontSize: "15px" }}>
          ◆
        </span>
      ) : (
        <span />
      )}
      <div style={{ display: "flex", flexDirection: "column", minWidth: 0 }}>
        <span style={{ fontSize: "34px", lineHeight: 1.15, overflowWrap: "anywhere" }} title={JSON.stringify(t.token)}>
          {wp(t.token)}
        </span>
        {note ? <span style={{ fontSize: "15px", fontStyle: "italic", color: MUTED }}>{note}</span> : null}
      </div>
      <span style={{ textAlign: "right", fontSize: "16px", ...NUM }}>{t.logit !== null ? t.logit.toFixed(2) : ""}</span>
    </div>
  );
};

const WordColumn: React.FC<{ title: string; sub: string; toks: Tok[] | null; sharedWith: Tok[] | null; missing: string }> = ({
  title,
  sub,
  toks,
  sharedWith,
  missing,
}) => {
  // ◆ marks a word-piece the other column also shows, spelt exactly (lib/labels.ts sharedShown), among the three shown
  const other = new Set((sharedWith ?? []).slice(0, 3).map((t) => t.token));
  return (
    <div style={{ breakInside: "avoid", display: "flex", flexDirection: "column", minWidth: 0, marginBottom: "28px" }}>
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: "4px 12px", paddingBottom: "8px" }}>
        <h3 className="h-sub">{title}</h3>
        <span style={{ fontSize: "15px", fontStyle: "italic", color: MUTED }}>{sub}</span>
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "22px minmax(0, 1fr) 72px",
          gap: "0 12px",
          fontSize: "13px",
          fontVariantCaps: "all-small-caps",
          letterSpacing: "0.16em",
          color: MUTED,
          paddingBottom: "4px",
          borderBottom: `1px solid ${HAIR}`,
        }}
      >
        <span />
        <span>word-piece</span>
        <span style={{ textAlign: "right" }}>score</span>
      </div>
      {toks ? (
        toks.slice(0, 3).map((t, i) => <WordRow key={i} t={t} shared={other.has(t.token)} />)
      ) : (
        <div style={{ padding: "14px 0", borderBottom: `1px solid ${HAIR_FAINT}` }}>
          <Missing what={missing} />
        </div>
      )}
    </div>
  );
};

/** A labelled neighbour at this layer, other than this neuron, whose two readings agree most among the words shown; a
 * neighbour the label gate holds back is never picked. */
function comparisonPick(m: Model) {
  let best: { pole: PoleReading; agr: Agreement; gate: Gate } | null = null;
  for (const p of m.poles) {
    if (p.neuron === m.neuron || !p.lens || !p.eye) continue;
    const gate = labelGate(layerIdx(m.idx), p.check);
    if (gate.heldBack) continue;
    const a = agreement(tokFromReading(p.lens)!, tokFromReading(p.eye)!);
    if (a.k === 0) continue;
    if (!best || a.k > best.agr.k) best = { pole: p, agr: a, gate };
  }
  return best;
}

/** How far labels held at a layer. N1 sampled layers 11, 21, 31, 41, 46, 51, 56 and 58 (counting from 1): labels held up at 41–58,
 * were at chance at 11–31, and were not tested at 32–40 or below 11 (N1_REPORT; REPORT errata E9, E43). Above 58, lib/labels.ts
 * trustLine: layer 59 was not tested, and at layer 60 the two readings are one. */
function trustText(n: number): string {
  const rumour = "Read both columns as a rumour, not a finding.";
  if (n >= 41 && n <= 58) {
    let s =
      `Layer ${n} lies inside the band where the study’s labels held up when single neurons were steered: between a third and two-thirds of a ` +
      `label’s top ten words were among the hundred words the neuron most pushed. At the layers it sampled below 41 (11, 21 and 31) the study ` +
      `found no stable single-neuron effect to label, and labels did no better than chance; layers 32 to 40 were not sampled.`;
    if (n >= 51) s += ` From layer 51 on, the lens did no better than the naked eye, so at layer ${n} neither column outranks the other.`;
    else if (n <= 46) s += ` At layers 41 to 46 the lens did a little better than the naked eye.`;
    return s;
  }
  if (n >= 32 && n <= 40)
    return (
      `Layer ${n} lies in a stretch the study did not sample, layers 32 to 40. Its labels held up at layers 41 to 58 and did no better than ` +
      `chance at 11 to 31; here nobody has tested them. ${rumour}`
    );
  if (n >= 11 && n <= 31)
    return (
      `Layer ${n} lies in the stretch where the study’s labels did no better than chance: at the layers it sampled there (11, 21 and 31) it ` +
      `found no stable single-neuron effect to label. Of the layers sampled, labels held up only at 41 to 58; 32 to 40 were not sampled. ${rumour}`
    );
  if (n < 11)
    return (
      `Layer ${n} lies below every layer the study tested; the lowest was layer 11, where labels did no better than chance. Of the layers ` +
      `sampled, labels held up only at 41 to 58. ${rumour}`
    );
  return `${trustLine(layerIdx(n - 1))} Labels held up at layers 41 to 58.`;
}

/** The canvas's comparison neuron in the recorded sighting: layer index 53, rank 2 (NeuronFolio/recorded.ts CMP). */
const CMP_IDX = 53;
const CMP_RANK = 2;

/** The seal's verdict from the server's ten-a-side count (the gate's own measure). */
const verdictOfTen = (k: number): Agreement["verdict"] => (k === 0 ? "disagree" : k === 10 ? "agree" : "partly agree");

const Says: React.FC<{ m: Model }> = ({ m }) => {
  const rx = (l: number) => f1(24 + ((l - 1) * 852) / 59);
  const ruler = useMemo(() => {
    let local = "";
    let global = "";
    for (let l = 1; l <= 60; l++) {
      if (l % 6 === 0) global += `M${rx(l)},60 L${rx(l)},49 `;
      else local += `M${rx(l)},60 L${rx(l)},54 `;
    }
    return { local, global };
  }, []);
  const missing = awaitWhat(m, m.insp?.pending);
  const agr = m.agr;
  const cmp = m.recorded ? null : comparisonPick(m);
  // the canvas's comparison neuron, 18422 (the recorded sighting's 53:2), is held back like any other if the gate says so
  const cmpRecordedGate = labelGate(layerIdx(CMP_IDX), fixtureCheck(CMP_IDX, CMP_RANK));
  const x = rx(m.layerNo);
  const band =
    m.layerNo > 58
      ? "above the band"
      : m.layerNo >= 41
        ? "inside the band"
        : m.layerNo >= 32
          ? "in the stretch the study did not sample"
          : m.layerNo >= 11
            ? "in the stretch where labels did no better than chance"
            : "below every layer the study tested";
  // the gate decides first: a held-back label's seal reads "held back"; with the server's count the seal goes by the first
  // ten, as the study counted, else by the three shown, as a hint
  const g = m.focusGate;
  const held = !!(m.focus.lens || m.focus.eye) && g.heldBack;
  const ten = g.sharedTop10;
  const verdict = ten != null ? verdictOfTen(ten) : (agr?.verdict ?? null);
  const sealLines = held
    ? ["held", "back"]
    : g.condition3 === "one-reading"
      ? ["one", "reading"]
      : !verdict
        ? ["not yet", "read"]
        : verdict === "partly agree"
          ? ["partly", "agree"]
          : [verdict];
  const sealSub =
    ten != null
      ? `${ten} IN 10`
      : held
        ? g.reasons.includes("writes_nothing")
          ? "SILENCED"
          : g.reasons.includes("below_layer_40")
            ? "BELOW 41"
            : ""
        : agr
          ? `${agr.k} IN 3`
          : "[UNREAD]";
  const unchecked = m.focus.from === "recorded" ? "not yet checked ten a side: this sighting carries three words a side" : "not yet in the server";
  const { lens } = useReadingLens(m.sighting);
  // what the next causal test asks for: the apparatus sets it, both buttons in the main reading send it
  const [causalAsk, setCausalAsk] = useState<CausalAsk>(CAUSAL_ASK);

  return (
    <section aria-labelledby="says-head" style={SECTION}>
      <div className="row-between">
        <h2 className="h-sec" id="says-head">
          ii · what it says
        </h2>
        <span style={ASIDE}>the {faceWord(m.focusFace)} face, read two ways</span>
      </div>
      <Rubric style={{ maxWidth: "760px" }}>
        Read across, as in a book with two languages on facing pages. The first column shows this face seen through the lens; the second, the
        same face seen with the naked eye. A word-piece that stands in both, among the words shown, is marked ◆; it must be spelt exactly
        alike, so a piece that follows a space and one that does not count as different, though they look the same here.
      </Rubric>
      {held ? (
        // the label gate holds this face's words back: the plate, the reason and its rubric; the apparatus keeps the words
        <HeldBackWords
          gate={g}
          words={m.focus.lens?.slice(0, 3).map((t) => t.token) ?? null}
          eye={m.focus.eye?.slice(0, 3).map((t) => t.token) ?? null}
          unchecked={unchecked}
          style={{ maxWidth: "760px", paddingTop: "6px", marginBottom: "12px" }}
        />
      ) : (
        <div style={{ columns: "2 340px", columnGap: "88px", columnRule: `1px solid ${HAIR}`, paddingTop: "6px" }}>
          <WordColumn title="through the lens" sub={`(${lensLabel(lens)})`} toks={m.focus.lens} sharedWith={m.focus.eye} missing={missing} />
          <WordColumn title="to the naked eye" sub="(logit lens)" toks={m.focus.eye} sharedWith={m.focus.lens} missing={missing} />
        </div>
      )}
      <Rubric style={{ maxWidth: "760px" }}>
        These are the words each reading says this face pushes the model toward when it fires; how far to trust them depends on the layer, as
        set out below. They are not a forecast: in the study, the words that follow when a
        neuron fires on its own were mostly not its label. Scores sit near 30, the ceiling where the model squashes them, and a label looks as
        confident as noise does, so read the order, not the size.
      </Rubric>
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "28px 44px",
          alignItems: "center",
          paddingTop: "22px",
          marginTop: "4px",
          borderTop: `1px solid ${HAIR_SOFT}`,
        }}
      >
        <svg
          width="140"
          height="140"
          viewBox="0 0 140 140"
          role="img"
          aria-label={
            held
              ? `A seal reading: held back. ${heldLine(g)}${ten != null ? ` ${cap(numberWord(ten))} word-piece${ten === 1 ? "" : "s"} in ten ${ten === 1 ? "is" : "are"} shared.` : ""}`
              : ten != null
                ? `A seal reading: ${g.condition3 === "one-reading" ? "one reading" : verdict}. ${cap(numberWord(ten))} word-piece${ten === 1 ? "" : "s"} in ten ${ten === 1 ? "is" : "are"} shared.`
                : agr
                  ? `A seal reading: ${agr.verdict}. ${cap(numberWord(agr.k))} word-piece${agr.k === 1 ? "" : "s"} in three ${agr.k === 1 ? "is" : "are"} shared.`
                  : "A seal reading: not yet read. The two readings have not been compared."
          }
          style={{ flex: "0 0 auto" }}
        >
          <circle cx="70" cy="70" r="65" fill="none" stroke={GILT} strokeWidth="1.2" />
          <circle cx="70" cy="70" r="60" fill="none" stroke={GILT} strokeWidth="5" strokeDasharray="0.9 3.3" />
          <circle cx="70" cy="70" r="55" fill="none" stroke={GILT} strokeWidth="0.8" />
          <circle cx="70" cy="70" r="46" fill="none" stroke={GILT} strokeOpacity="0.55" strokeWidth="0.8" />
          <path d="M70 30 L74 36 L70 42 L66 36 Z" fill={GILT} />
          <g fontFamily={SERIF} textAnchor="middle">
            {sealLines.length === 2 ? (
              <>
                <text x="70" y="67" fontSize="21" fontStyle="italic" fill={INK}>
                  {sealLines[0]}
                </text>
                <text x="70" y="88" fontSize="21" fontStyle="italic" fill={INK}>
                  {sealLines[1]}
                </text>
              </>
            ) : (
              <text x="70" y="80" fontSize="21" fontStyle="italic" fill={INK}>
                {sealLines[0]}
              </text>
            )}
            <text x="70" y="106" fontSize="12" letterSpacing="1.4" fill={MUTED} style={NUM}>
              {sealSub}
            </text>
          </g>
        </svg>
        <div style={{ flex: "999 1 460px", minWidth: 0, display: "flex", flexDirection: "column", gap: "10px" }}>
          <h3 className="h-sub">
            {held
              ? "the lens and the naked eye, held back"
              : g.condition3 === "one-reading"
                ? "the lens and the naked eye, one reading"
                : verdict
                  ? `the lens and the naked eye ${verdict}`
                  : "the lens and the naked eye, not yet compared"}
          </h3>
          <p style={{ margin: 0, fontSize: "18px", maxWidth: "700px" }}>
            {held ? (
              // the plate above already says why; here only what was counted
              ten != null ? (
                `Counted ten a side, by word-piece, they share ${numberWord(ten)}.`
              ) : g.reasons.includes("writes_nothing") ? (
                "Silenced in the vault, the neuron writes nothing: its two readings tie on every word-piece, so there is nothing to compare."
              ) : (
                "Below layer 41 the two readings are not compared on this page; the apparatus keeps them both."
              )
            ) : ten != null ? (
              <>
                {cap(numberWord(ten))} of their first ten word-pieces stand on both sides, counted by word-piece, as the study counted
                {g.condition3 === "one-reading" ? ": at layer 60 the two readings are one" : ""}.
                {overlapLine(g) ? ` ${cap(overlapLine(g))}.` : ""}
                {m.focus.from === "recorded" ? <> Among the three shown, “jestem”; and every word in both columns is a form of one verb, <i>to be</i>.</> : null}
                {g.condition3 === "unvalidated" ? ` ${sharedLine(g)}.` : ""}
              </>
            ) : m.focus.from === "recorded" ? (
              <>
                One word-piece in three, “jestem”, stands on both sides; and every word in both columns is a form of one verb, <i>to be</i>.
              </>
            ) : agr ? (
              agr.k === 0 ? (
                "No word-piece stands on both sides."
              ) : (
                `${cap(numberWord(agr.k))} word-piece${agr.k === 1 ? "" : "s"} in three, ${andList(agr.shared.map(wp))}, stand${agr.k === 1 ? "s" : ""} on both sides.`
              )
            ) : (
              <Missing what={missing} />
            )}
          </p>
          <p style={{ margin: 0, fontSize: "16px", color: MUTED, maxWidth: "700px" }}>
            Why it matters: in the study, how many word-pieces the two readings shared was a free sign that a neuron’s label is real. Take one
            sound label and one unsound, from its sample of random neurons, and the count of their first ten shared ranked the sound one higher
            84 times in 100{" "}
            <span style={{ fontStyle: "italic" }}>(AUC 0.84, layers 41–58, for labels read through the base-fitted J++ lens)</span>; sharing just
            one was a weak sign, sound less often than the average label.{" "}
            {ten != null
              ? "The instrument counted the first ten a side, as the study did, and shows three."
              : "The study compared ten words a side; the instrument shows three, so treat this seal as a hint, not a verdict."}{" "}
            A label whose two readings share none of their first ten is held back from this page, and so is any label below layer 41; the
            cut at none is the instrument’s, since the study set none.
          </p>
        </div>
        {m.recorded && !cmpRecordedGate.heldBack ? (
          <ComparisonBlock
            neuron={CMP.neuron}
            lens="“Am” “am” “Am”"
            eye="“am” “Am” “am”"
            text={
              "Third loudest in this layer, fired on its negative face. Both readings give only forms of “am”. Of the three word-pieces " +
              "shown, two stand on both sides, “Am” and “am”, each after a space (the lens’s third “Am” and the naked eye’s third “am” " +
              "have none before them, so they are different word-pieces, though they look the same)" +
              (cmpRecordedGate.sharedTop10 != null ? `; counted ten a side, they share ${numberWord(cmpRecordedGate.sharedTop10)}.` : ".")
            }
          />
        ) : cmp ? (
          <ComparisonBlock
            neuron={cmp.pole.neuron}
            lens={(cmp.pole.lens ?? []).slice(0, 3).map((t) => wp(t.token)).join(" ")}
            eye={(cmp.pole.eye ?? []).slice(0, 3).map((t) => wp(t.token)).join(" ")}
            text={
              `${cap(m.rw.nth(cmp.pole.rank))} in this layer, fired on its ${faceWord(cmp.pole.face)} face. ` +
              (cmp.agr.verdict === "agree"
                ? "Both readings show the same words"
                : `${cap(numberWord(cmp.agr.k))} word-piece${cmp.agr.k === 1 ? "" : "s"} in three stand${cmp.agr.k === 1 ? "s" : ""} on both sides`) +
              (cmp.gate.sharedTop10 != null ? `, and ${numberWord(cmp.gate.sharedTop10)} of their first ten.` : ", among the words shown.")
            }
          />
        ) : null}
      </div>
      <CausalCheckBlock m={m} ask={causalAsk} />
      <div
        style={{ display: "flex", flexDirection: "column", gap: "10px", paddingTop: "22px", marginTop: "4px", borderTop: `1px solid ${HAIR_SOFT}` }}
      >
        <h3 className="h-sub">where neuron labels can be trusted</h3>
        <p style={{ margin: 0, fontSize: "18px", maxWidth: "900px" }}>{trustText(m.layerNo)}</p>
        <div style={{ overflowX: "auto", paddingBottom: "4px" }}>
          <div style={{ minWidth: "640px" }}>
            <svg
              viewBox="0 0 900 96"
              role="img"
              aria-label={`A ruler of the sixty layers. Layers 41 to 58 are marked as the band where labels held up; at layers 11 to 31 they did no better than chance, and layers 32 to 40 were not sampled. This neuron sits at layer ${m.layerNo}, ${band}.`}
              style={{ width: "100%", height: "auto", display: "block" }}
            >
              <g fontFamily={SERIF} fontSize="14" textAnchor="middle">
                <text x={rx(21)} y="28" fontStyle="italic" fill={MUTED} style={NUM}>
                  11–31: no better than chance
                </text>
                <text x={rx(36)} y="28" fontStyle="italic" fill={MUTED} style={NUM}>
                  32–40: not sampled
                </text>
                <text x="724.4" y="28" fill={GILT_TEXT} style={NUM}>
                  labels held up · layers 41–58
                </text>
              </g>
              <path
                d={`M${rx(11)},36 L${rx(11)},44 M${rx(31)},36 L${rx(31)},44`}
                fill="none"
                stroke={INK}
                strokeOpacity="0.45"
                strokeWidth="1"
              />
              <line x1={rx(11)} y1="40" x2={rx(31)} y2="40" stroke={INK} strokeOpacity="0.45" strokeWidth="1" />
              <line x1={rx(32)} y1="40" x2={rx(40)} y2="40" stroke={INK} strokeOpacity="0.35" strokeWidth="1" strokeDasharray="1 4" />
              <path d="M601.6,34 L601.6,46 M847.1,34 L847.1,46" fill="none" stroke={GILT} strokeWidth="1" />
              <line x1="601.6" y1="40" x2="847.1" y2="40" stroke={GILT} strokeWidth="3" />
              <line x1="24" y1="60" x2="876" y2="60" stroke={INK} strokeOpacity="0.35" strokeWidth="1" />
              <path d={ruler.local} fill="none" stroke={INK} strokeOpacity="0.35" strokeWidth="0.8" />
              <path d={ruler.global} fill="none" stroke={GILT} strokeWidth="1" />
              <path d={`M${x},51 L${f1(x + 9)},60 L${x},69 L${f1(x - 9)},60 Z`} fill={RED} />
              <g fontFamily={SERIF} fontSize="13" textAnchor="middle" fill={MUTED} style={NUM}>
                {[1, 11, 31, 41, 58, 60]
                  .filter((l) => Math.abs(l - m.layerNo) > 2)
                  .map((l) => (
                    <text key={l} x={rx(l)} y="86">
                      {l}
                    </text>
                  ))}
                <text x={x} y="88" fill={RED} fontSize="14">
                  {m.layerNo}
                </text>
              </g>
            </svg>
          </div>
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 22px", fontSize: "15px", color: MUTED }}>
          <span style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
              <path d="M5 0 L10 5 L5 10 L0 5 Z" fill={RED} />
            </svg>
            this neuron, layer {m.layerNo}
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <span aria-hidden="true" style={{ width: "1px", height: "12px", background: GILT, display: "inline-block" }} />
            gilded ticks: every sixth layer looks back over everything
          </span>
        </div>
      </div>
      <SaysApparatus m={m} cmp={cmp} ask={causalAsk} setAsk={(a) => setCausalAsk((x) => ({ ...x, ...a }))} />
    </section>
  );
};

const ComparisonBlock: React.FC<{ neuron: number; lens: string; eye: string; text: string }> = ({ neuron, lens, eye, text }) => (
  <div
    style={{
      flex: "1 1 300px",
      minWidth: 0,
      display: "flex",
      flexDirection: "column",
      gap: "8px",
      paddingTop: "14px",
      borderTop: `1px solid ${HAIR_FAINT}`,
    }}
  >
    <h3 className="h-sub">
      for comparison · neuron <span style={NUM}>{neuron}</span>
    </h3>
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "110px minmax(0, 1fr)",
        gap: "2px 14px",
        alignItems: "baseline",
        fontSize: "15px",
        color: MUTED,
      }}
    >
      <span>through the lens</span>
      <span style={{ fontSize: "20px", color: INK, overflowWrap: "anywhere" }}>{lens}</span>
      <span>naked eye</span>
      <span style={{ fontSize: "20px", color: INK, overflowWrap: "anywhere" }}>{eye}</span>
    </div>
    <p style={{ margin: 0, fontSize: "16px" }}>{text}</p>
  </div>
);

const SaysApparatus: React.FC<{
  m: Model;
  cmp: { pole: PoleReading; agr: Agreement; gate: Gate } | null;
  ask: CausalAsk;
  setAsk: (a: Partial<CausalAsk>) => void;
}> = ({ m, cmp, ask, setAsk }) => {
  const exact = m.focus.from === "recorded";
  const lensRaw = m.focus.lens ? scoresText(m.focus.lens, exact ? 4 : undefined) : null;
  const eyeRaw = m.focus.eye ? scoresText(m.focus.eye, exact ? 4 : undefined) : null;
  const from =
    m.focus.from === "inspection"
      ? "from the server’s inspect_pole reply, eight a side"
      : m.focus.from === "recorded"
        ? "recorded in the study’s sighting, three a side"
        : m.focus.from === "sighting"
          ? "from this sighting’s stored labels"
          : null;
  const agr = m.agr;
  const all = m.focus.lens && m.focus.eye && m.focus.from === "inspection" ? agreement(m.focus.lens, m.focus.eye, 8) : null;
  return (
    <Apparatus title="the output label (J-VINDEX)" label="Apparatus for the output label">
      <dl className="dl num">
        <dt>formula</dt>
        <dd>
          logits<sub>σ</sub> = 30 · tanh(W<sub>U</sub> · RMSNorm<sub>final</sub>((σ · w<sub>down</sub> ⊙ w<sub>post</sub>) · J<sub>ℓ</sub>
          <sup>⊤</sup>) / 30), with w<sub>post</sub> the post-feed-forward norm weight; the naked eye drops J<sub>ℓ</sub>
          <sup>⊤</sup>.
        </dd>
        <dt>lens</dt>
        <dd>
          J = the J++ lens (Ayonrinde &amp; Lindsey 2026) [4], an LRP backward pass after the R-lens rules [5] on the Jacobian lens (Gurnee et
          al. 2026) [2]. It was fitted on the base bf16 model and is applied to the instruct W4A16 one (REPORT §3.4, bug 9). Whether its
          disagreement with the instruct model comes from the chat training or the 4-bit rounding could not be told apart, and J++’s
          published recipe, cluster weighting and word-piece filter were not tested (R1). R1 refitted a lens on the instruct model: along 128
          fixed directions, stable at layers 31–59, and as a full matrix usable only at layers 41, 46, 51, 56 and 59. The server loads
          neither, and neuron labels through it were not tested. The published J++ readout also drops non-semantic word-pieces; the
          instrument applies only the matrix (REPORT §3.1). The naked eye is the logit lens (nostalgebraist 2020) [8].
        </dd>
        <LensRecord sighting={m.sighting} reply={m.focus.from === "inspection" ? m.insp?.data : undefined} />
        <dt>raw scores</dt>
        <dd>
          {lensRaw || eyeRaw ? (
            <>
              lens {lensRaw ?? "—"}; naked eye {eyeRaw ?? "—"}. A leading space marks a word-piece that begins a word.
              {from ? <> ({from}.)</> : null}
            </>
          ) : (
            <Missing what={awaitWhat(m, m.insp?.pending)} />
          )}
        </dd>
        {m.recorded ? (
          <>
            <dt>neuron 18422</dt>
            <dd>
              a = {signedRaw(CMP.act)}; lens {scoresText(tokOf(CMP.lens), 4)}; naked eye {scoresText(tokOf(CMP.eye), 4)}.
            </dd>
          </>
        ) : cmp ? (
          <>
            <dt>neuron {cmp.pole.neuron}</dt>
            <dd>
              a = {signedRaw(cmp.pole.act)}; lens {scoresText(tokFromReading(cmp.pole.lens)!)}; naked eye{" "}
              {scoresText(tokFromReading(cmp.pole.eye)!)}.
            </dd>
          </>
        ) : null}
        <dt>agreement</dt>
        <dd>
          {agr ? (
            <>
              among the words shown, {agr.k} of 3 shared, spelt exactly{agr.k ? <>: {agr.sharedRaw.map((t) => `“${t}”`).join(", ")}</> : null}.
              {all ? (
                <>
                  {" "}
                  Among all eight the server returned, {all.k} of 8 shared.
                </>
              ) : null}{" "}
              The study’s signal is the top-10 overlap: AUC 0.84 for telling a sound label (causal P@10 ≥ 0.3) from an unsound one, within layers
              41–58, for base-fitted J++ labels on random neurons. {m.focus.from === "inspection" ? "The columns show the first three." : "This sighting stored three a side."}{" "}
              The server’s count of it: {sharedLine(m.focusGate) ?? <Missing what={m.focus.from === "recorded" ? "not yet checked ten a side" : "not yet in the server"} />}.
            </>
          ) : (
            <Missing what={awaitWhat(m, m.insp?.pending)} />
          )}
        </dd>
        <dt>held back</dt>
        <dd>
          {(m.focus.lens || m.focus.eye) && m.focusGate.heldBack ? (
            <>
              the {faceWord(m.focusFace)} face’s label, from the main reading: {heldTag(m.focusGate)}
              {m.focusGate.reasons.includes("below_layer_40") ? " (condition 2)" : " (condition 3)"} (REPORT §3.3; N1). Through the lens{" "}
              {m.focus.lens ? m.focus.lens.slice(0, 3).map((t) => `“${t.token}”`).join(" · ") : "—"}; to the naked eye{" "}
              {m.focus.eye ? m.focus.eye.slice(0, 3).map((t) => `“${t.token}”`).join(" · ") : "—"}.{" "}
            </>
          ) : (
            <>the {faceWord(m.focusFace)} face’s label is shown. </>
          )}
          {m.other && (m.other.lens || m.other.eye) && m.otherGate.heldBack ? (
            <>
              The {faceWord(m.otherFace)} face’s label is held back too ({heldTag(m.otherGate)}): through the lens{" "}
              {m.other.lens ? m.other.lens.slice(0, 3).map((t) => `“${t.token}”`).join(" · ") : "—"}; to the naked eye{" "}
              {m.other.eye ? m.other.eye.slice(0, 3).map((t) => `“${t.token}”`).join(" · ") : "—"}.{" "}
            </>
          ) : null}
          The rule: a label is held back below layer 41, and where its two readings share none of their first ten word-pieces, counted by
          word-piece as the study counted them (fp32; “ Am” and “Am” are different word-pieces); the cut at none of ten is the
          instrument’s. The gate was measured for base-fitted J++ labels within layers 41–58 only (erratum E47).
          {m.focusGate.heldBack ? <> {heldRubric(m.focusGate)}</> : null}
        </dd>
        <dt>validity</dt>
        <dd>
          P@10 in the causal top 100 for J++ labels is 0.35–0.67 at layers 41–58 (null ≤ 0.003; over the whole vocabulary the
          rank-correlation lift is 45–77% of the split-half reliability ceiling) and about 0 at layers 11–31; layers 32–40 were not sampled
          (N1). J++ beats the naked eye by 0.04 to 0.12 only at layers 41–46; at 51–58 it is equal or slightly worse (−0.04 [−0.07, −0.01]
          in N1’s sample, about 0 in a fresh one) (REPORT §3.2; errata E40, E41).
        </dd>
        <dt>caveat</dt>
        <dd>
          every score sits under the softcap of 30, and single neurons decode as peaked as noise, so confidence is not evidence (REPORT §3.4, bug
          8; §4). Prior-z-scored labels did worse (−0.07 P@10). An output label says what the pole pushes when amplified, not what it is used for
          in running text (N1, Q2).
        </dd>
        <CausalApparatus m={m} ask={ask} setAsk={setAsk} />
      </dl>
    </Apparatus>
  );
};

/* ----------------------------------------------- ii · what it says: the test */

/** What the next test asks for, set in the apparatus (CAUSAL_DEFAULTS until the reader changes it). */
interface CausalAsk {
  k: number;
  write: "activation" | "frozen_norm";
  scaleRef: "study" | "this_text";
}

const CAUSAL_ASK: CausalAsk = { ...CAUSAL_DEFAULTS };
const askIsDefault = (a: CausalAsk) => a.k === CAUSAL_DEFAULTS.k && a.write === CAUSAL_DEFAULTS.write && a.scaleRef === CAUSAL_DEFAULTS.scaleRef;

/** The size the next test may take: more than the floor's k (0.01) and at most the server's k_max (200). */
function askFits(a: CausalAsk, card: CausalCheckCard | null): boolean {
  const lo = card?.k_floor ?? 0.01;
  const hi = card?.k_max ?? 200;
  return Number.isFinite(a.k) && a.k > lo && a.k <= hi;
}

/** "a thousand times", "6,000 times": how much smaller the floor's test was than this one. */
function smallerWord(k: number | null, kFloor: number | null): string {
  if (k == null || kFloor == null || !(kFloor > 0)) return "much";
  const r = Math.round(k / kFloor);
  return r === 1000 ? "a thousand times" : `${comma(r)} times`;
}

/**
 * Why a test kept for this neuron is set aside: a test at one word-piece answers for the exact word-pieces it read (the
 * shown sighting's ids up to its read position) and the vault's edits in force; a test on the study's texts, for the
 * edits only. Null when it still answers for what is shown.
 */
function causalStaleWhy(stored: Pending<unknown>, contexts: CausalContexts, s: Sighting | null, editsNow: number[]): string | null {
  const again = "Test again to read it for this one.";
  const sent = stored.sent ?? {};
  const reply = stored.data ? parseCausalCheck(stored.data) : null;
  if (contexts === "given") {
    const ids = Array.isArray(sent.token_ids) ? (sent.token_ids as unknown[]) : null;
    const here = s?.source === "live" ? s.contextIds : null;
    if (!ids || !here || ids.length !== here.length || ids.some((v, i) => v !== here[i])) {
      const piece = reply?.readToken != null ? ` (${pieceWord(reply.readToken)})` : "";
      return here
        ? `This test was taken at another word-piece${piece}, so it is set aside here. ${again}`
        : `This test was taken at a word-piece${piece} of a live sighting, not of the one shown, so it is set aside here.`;
    }
  }
  // the edits the test was read under, against those in force now: once the reply is in, its own vault_edits (which may
  // hold an edit this page never heard of, the scientist's over MCP; the store then asks the vault for its ledger again,
  // so the page's record of the edits in force catches up); while it is pending, the edits as this page knew them
  const editsThen = reply?.vaultEditsSent ? reply.vaultEdits : (stored.edits ?? null);
  if (editsThen && !sameIds(editsThen, editsNow)) {
    return "This test read the model under other vault edits than those in force now, so it is set aside. Test again to read the model as it is now.";
  }
  return null;
}

/** A count in words, "none" for 0. */
const countWord = (n: number) => (n === 0 ? "none" : numberWord(n));

/** A ratio of two KLs: whole numbers with thousands marked from 1,000, else two figures. */
const ratioWord = (x: number) => (x >= 1000 ? comma(x) : sig(x, 2));

/**
 * A face's words with their odds before and after (the one text), or alone (means over the study's texts). Where the two
 * odds read the same in words (both under one in a million, say: the ranking by change of log-probability favours rare
 * word-pieces that grow by a large factor), the measured factor is given instead, so a word that rose never reads as
 * unmoved.
 */
function moversText(ws: CausalWord[], given: boolean, n = 3): string {
  return andList(
    ws.slice(0, n).map((w) => {
      if (!given) return wp(w.token);
      const before = probWord(w.baseLogprob);
      const after = probWord(w.steeredLogprob);
      return before === after
        ? `${wp(w.token)} (${timesWord(w.dlogprob)} times as likely, still ${before})`
        : `${wp(w.token)} (${before} → ${after})`;
    }),
  );
}

/** The word-piece a test at one word-piece followed: the one the model wrote there, or the one the reader gave. */
type FollowedPiece = { id: number; token: string; source: "written" | "given" | "recorded"; endMark: boolean };

/** "The word the model chose here, ‘a’," / "The word-piece given here, ‘a’," / "The mark that ends its turn, …,". */
function followedLead(c: FollowedPiece): string {
  if (c.source === "given") return `The word-piece given here, ${wp(c.token)},`;
  if (c.endMark) {
    if (isTurnEnd(c.id, c.token)) return `The mark that ends its turn, ${wp(c.token)}, which the model wrote here,`;
    if (isTextEnd(c.id, c.token)) return `The mark that ends the text, ${wp(c.token)}, which the model wrote here,`;
    return `The control mark the model wrote here, ${wp(c.token)}, where writing stops,`;
  }
  return `The word the model chose here, ${wp(c.token)},`;
}

/** What a test found, in plain words, for the main reading. */
const CausalResult: React.FC<{ m: Model; r: CausalCheck; chosen: FollowedPiece | null }> = ({ m, r, chosen }) => {
  const given = r.contexts === "given";
  const opened = r.polarity;
  const faces = (opened > 0 ? [[1, r.pos], [-1, r.neg]] : [[-1, r.neg], [1, r.pos]]) as [1 | -1, CausalFace | null][];
  const first = faces[0][1];
  const gate = labelGate(layerIdx(m.idx), first?.check ?? null);
  const silenced = gate.reasons.includes("writes_nothing") || (first?.lens != null && first.lens.p10 === null);
  const inTop = (l: CausalLabel | null) => (l ? l.inTop100.filter(Boolean).length : null);
  const nLens = inTop(first?.lens ?? null);
  const nEye = inTop(first?.eye ?? null);
  const labelK = first?.labelK ?? 10;
  const causalK = first?.causalK ?? 100;
  const hundred = causalK === 100 ? "hundred" : String(causalK);
  const tracked = chosen && first ? (first.tracked.find((w) => w.tokenId === chosen.id) ?? null) : null;
  const tooStrong =
    r.write === "activation" && !silenced
      ? faces.filter(([, f]) => f && f.linear.linearity != null && f.linear.linearity < LINEARITY_TAIL)
      : [];
  const noisy = silenced ? [] : faces.filter(([, f]) => f && f.klOverFloor != null && f.klOverFloor <= NOISE_RATIO_MAX);
  const band = bandOf(layerIdx(m.idx));
  const oneReading = band === "one_reading";
  const rho = first?.splitHalfRho ?? null;
  // the study's bar and its k = 10 figures were measured with the activation write, sized on the study's own 36 texts:
  // only a test made so is "this size" or "six times" smaller than they were
  const studySized = r.scaleRef === "study" && r.write === "activation";
  const sizeHow = [r.scaleRef === "this_text" ? "sized on this text" : "", r.write === "frozen_norm" ? "written after the norm" : ""]
    .filter(Boolean)
    .join(", ");
  // the bar was measured for base-fitted J++ labels of unedited neurons: under another lens, or with an edit on this
  // neuron in force, the count is not set beside it (as A2's overlapLine and gateTrustLine hold the study's record back)
  const lensValidated = first?.check ? first.check.lensValidated : true;
  const asDefault = r.k === CAUSAL_DEFAULTS.k && r.write === CAUSAL_DEFAULTS.write && r.scaleRef === CAUSAL_DEFAULTS.scaleRef;
  const late = STUDY_K10_LATE_P10.map((v) => v.toFixed(2)).join("–");
  const bandLine =
    band === "below_sampled"
      ? "This layer is below every layer the study tested."
      : band === "chance_where_sampled"
        ? `At the layers the study sampled in this stretch (11, 21 and 31) no effect held from one text to the next${m.layerNo === 11 ? ", and the effects sat at rounding noise" : ""}.`
        : band === "not_sampled"
          ? "The study did not sample this stretch, layers 32 to 40."
          : band === "above_band"
            ? "This layer is above every layer the study tested; its highest was 58."
            : oneReading
              ? "At layer 60 the lens and the naked eye are one reading, so they always agree, and their agreement says nothing; the study did not test neuron labels at this layer."
              : "";
  const scopeLine = given
    ? "What moves here speaks for this text only."
    : rho != null
      ? `The two halves of the texts agree at ρ = ${sig(rho, 2)} (the apparatus sets out the split).`
      : "";
  const barLine = !lensValidated
    ? `The study’s bar for this count (three of ten words among the hundred, over its 36 texts) was measured for base-fitted J++ labels of unedited neurons; this label was read ${gate.edited ? "with an edit in force on this very neuron" : "through a lens other than the base-fitted J++ one"}, so the count is not compared with it.`
    : `In the study, over 36 texts, a label with three of its ten words among the hundred counted as sound; that bar was set on tests ${
        studySized && r.k === 10
          ? "six times this size"
          : studySized && r.k === 60
            ? "of this size"
            : `at k = 60, sized on the study’s texts with the activation write (this one was k = ${sig(r.k)}${sizeHow ? `, ${sizeHow}` : ""})`
      }, and ${studySized && r.k === 10 ? "at this size" : "at k = 10"} the share of a lens label’s ten words among the hundred ran ${late} at layers 41 to 56 (N1V code audit).`;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
      <h4 className="h-sub">
        {given ? `at this word-piece${r.readToken != null ? `, ${pieceWord(r.readToken)}` : ""}` : `on the study’s ${r.nContexts ?? 36} texts`}
      </h4>
      {!asDefault ? (
        <p className="status-line" style={{ margin: 0 }}>
          Taken at {sig(r.k)} times the neuron’s strongest firing on {r.scaleRef === "this_text" ? "this text" : "the study’s 36 texts"}
          {r.write === "frozen_norm" ? ", written after the layer’s norm (the frozen-norm write)" : ""}, as the apparatus set it.
        </p>
      ) : null}
      {silenced ? (
        <p style={{ margin: 0, fontSize: "18px", maxWidth: "820px" }}>
          Silenced in the vault, the neuron writes nothing, so turning it up moved nothing it could write: the words the
          apparatus lists as rising and falling are only ties or rounding noise, and the neuron has no label to test.
        </p>
      ) : (
        faces.map(([face, f]) =>
          f ? (
            <p key={face} style={{ margin: 0, fontSize: "18px", maxWidth: "820px" }}>
              Turned up on its {faceWord(face)} face, the odds of {moversText(f.rises, given)} rose most
              {given ? "" : " on average over the texts"}, and of {moversText(f.falls, given)} fell most.{" "}
              {f.klOverFloor != null
                ? f.klOverFloor > NOISE_RATIO_MAX
                  ? `Overall the model moved beyond rounding noise: ${ratioWord(f.klOverFloor)} times as much as under the same test ${smallerWord(r.k, r.kFloor)} smaller.`
                  : `Overall the model moved no more than rounding noise: ${ratioWord(f.klOverFloor)} times as much as under the same test ${smallerWord(r.k, r.kFloor)} smaller.`
                : r.kFloor != null
                  ? `The same test ${smallerWord(r.k, r.kFloor)} smaller moved nothing measurable, so the level of rounding noise could not be read.`
                  : ""}
            </p>
          ) : null,
        )
      )}
      {!silenced ? (
        <p style={{ margin: 0, fontSize: "18px", maxWidth: "820px" }}>
          {nLens != null && nEye != null
            ? oneReading
              ? `${cap(countWord(nLens))} of the label’s ${numberWord(labelK)} words ${nLens === 1 ? "is" : "are"} among the ${hundred} words whose odds this test raised most, on its ${faceWord(opened)} face; at this layer the lens and the naked eye give one label.`
              : `${cap(countWord(nLens))} of the lens’s ${numberWord(labelK)} words and ${countWord(nEye)} of the naked eye’s are among the ${hundred} words whose odds this test raised most, on its ${faceWord(opened)} face.`
            : null}
          {gate.heldBack ? ` The label itself is held back from this page (${heldTag(gate).replace(/^held back · /, "")}); the apparatus keeps its words.` : null}
          {tracked && chosen ? ` ${followedLead(chosen)} became ${timesWord(tracked.dlogprob)} times as likely.` : null}
        </p>
      ) : null}
      {tooStrong.length ? (
        <p className="error-line" style={{ margin: 0 }}>
          This test was too strong to read straight: on its {andList(tooStrong.map(([face]) => faceWord(face)))} face
          {tooStrong.length > 1 ? "s" : ""} it squashed the rest of this layer’s output, keeping only{" "}
          {andList(tooStrong.map(([, f]) => sig(f!.linear.linearity, 2)))} of the size a straight write would have had, below{" "}
          {LINEARITY_TAIL}, the lowest tenth the study saw for its gentle tests (k = 10, layer {LINEARITY_TAIL_IDX + 1}). Turn it up less (the
          apparatus sets the size).
        </p>
      ) : null}
      {noisy.length ? (
        <p className="error-line" style={{ margin: 0 }}>
          On its {andList(noisy.map(([face]) => faceWord(face)))} face{noisy.length > 1 ? "s" : ""} this test moved the model hardly more than the
          same test {smallerWord(r.k, r.kFloor)} smaller did, which is rounding noise, so the words that moved there say nothing about the label.
        </p>
      ) : null}
      {given && m.sighting?.isNudged ? (
        <p className="status-line" style={{ margin: 0 }}>
          Read without the nudge this sighting was taken with.
        </p>
      ) : null}
      {r.vaultEdits.length ? (
        <p className="status-line" style={{ margin: 0 }}>
          Taken with {numberWord(r.vaultEdits.length)} vault edit{r.vaultEdits.length === 1 ? "" : "s"} in force, so this is the edited model
          {gate.edited ? ", with an edit on this very neuron, whose words echo it" : ""}.
        </p>
      ) : null}
      {bandLine || scopeLine ? (
        <Rubric style={{ maxWidth: "820px" }}>
          {bandLine}
          {bandLine && scopeLine ? " " : ""}
          {scopeLine}
        </Rubric>
      ) : null}
      {!silenced ? (
        <Rubric style={{ maxWidth: "820px" }}>
          {barLine}
          {given ? " One word-piece is a smaller test than 36 texts; only the test on the study’s 36 texts compares with its figures." : ""}
        </Rubric>
      ) : null}
    </div>
  );
};

const CausalCheckBlock: React.FC<{ m: Model; ask: CausalAsk }> = ({ m, ask }) => {
  const health = useTheodoliteStore((s) => s.health);
  const fetchHealth = useTheodoliteStore((s) => s.fetchHealth);
  const causalCheck = useTheodoliteStore((s) => s.causalCheck);
  const givenEntry = useTheodoliteStore((s) => s.causalChecks[causalKey(m.idx, m.neuron, "given")]) as Pending<unknown> | undefined;
  const studyEntry = useTheodoliteStore((s) => s.causalChecks[causalKey(m.idx, m.neuron, "study")]) as Pending<unknown> | undefined;
  const activeEdits = useTheodoliteStore((s) => s.activeEdits);
  const editsNow = useMemo(() => editIdsOf(activeEdits), [activeEdits]);
  const card = health?.causal_check ?? null;
  const open = m.conn === "open";
  // the folio needs the server's card to know it answers the test: ask for it when the server answers and none is held
  const askedHealth = useRef(false);
  useEffect(() => {
    if (open && health === null && !askedHealth.current) {
      askedHealth.current = true;
      void fetchHealth();
    }
  }, [open, health, fetchHealth]);
  const s = m.sighting;
  const live = s?.source === "live" && !!s.contextIds?.length;
  const fits = askIsDefault(ask) || askFits(ask, card);
  const canStudy = open && !!card && fits;
  const canGiven = canStudy && live;
  const send = (contexts: CausalContexts) =>
    causalCheck(m.idx, m.neuron, m.focusFace, { contexts, k: ask.k, write: ask.write, scaleRef: ask.scaleRef });
  const readAt = m.readAt ? `${m.readAt}${m.inReply}` : "";
  const noteId = `causal-note-${m.idx}-${m.neuron}`;
  const rows = (["given", "study"] as const).map((contexts) => {
    const entry = contexts === "given" ? givenEntry : studyEntry;
    const why = entry && (entry.data || entry.sent) ? causalStaleWhy(entry, contexts, s, editsNow) : null;
    const r = entry?.data && !why ? parseCausalCheck(entry.data) : null;
    const chosenId = Array.isArray(entry?.sent?.track_ids) ? (entry!.sent!.track_ids as number[])[0] : undefined;
    const chosen: FollowedPiece | null =
      contexts === "given" && s?.chosen && s.chosen.id === chosenId
        ? { id: s.chosen.id, token: s.chosen.token, source: s.chosen.source, endMark: s.chosen.endMark }
        : null;
    return { contexts, entry, why, r, chosen };
  });

  return (
    <div
      style={{ display: "flex", flexDirection: "column", gap: "12px", paddingTop: "22px", marginTop: "4px", borderTop: `1px solid ${HAIR_SOFT}` }}
    >
      <h3 className="h-sub">put the label to the test</h3>
      <p style={{ margin: 0, fontSize: "18px", maxWidth: "820px" }}>
        A label is a guess read off the neuron’s wiring. To test it, the instrument turns this neuron up at the word-piece{" "}
        {(live && readAt) || "a live sighting reads"}, once on each face, keeps an untouched copy of the same reading beside it, and lists the words whose
        odds rose and fell most. It can do the same at the last word-piece of each of the study’s 36 texts. The test reads the model’s own odds,
        not the lens.
      </p>
      <Rubric style={{ maxWidth: "820px" }}>
        {askIsDefault(ask)
          ? "The test turns the neuron up by ten times the strongest it fired on the study’s 36 texts, the gentle size the study’s second look advised."
          : `The apparatus has set the next test to ${sig(ask.k)} times the strongest the neuron fired on ${ask.scaleRef === "this_text" ? "the text read (a test on the study’s texts is still sized on them)" : "the study’s 36 texts"}${ask.write === "frozen_norm" ? ", written after the layer’s norm" : ""}; the study’s second look advised ten times, on its 36 texts.`}{" "}
        The test counts how many of the label’s ten words are among the hundred whose odds rose most; the study counted a label as sound at
        three of ten, over its 36 texts, and also asks that it sit at layer 41 or above and that the lens and the naked eye agree.
      </Rubric>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "12px 20px", alignItems: "center" }}>
        <button
          className="btn-primary"
          type="button"
          disabled={!canGiven || !!givenEntry?.pending}
          aria-describedby={!canGiven ? noteId : undefined}
          onClick={() => send("given")}
        >
          {givenEntry?.pending ? "testing…" : "test it at this word-piece"}
        </button>
        <button
          className="btn-ghost"
          type="button"
          disabled={!canStudy || !!studyEntry?.pending}
          aria-describedby={!canStudy ? noteId : undefined}
          onClick={() => send("study")}
        >
          {studyEntry?.pending ? "testing on the 36 texts…" : "test it on the study’s 36 texts"}
        </button>
        <span id={noteId}>
          {!open ? (
            <Missing what="the instrument answers when the server is running" />
          ) : health === null ? (
            <Missing what="cannot tell yet whether the server offers it" />
          ) : !card ? null : !fits ? (
            <span className="error-line">
              The size set in the apparatus is out of range: k must be more than {card.k_floor} and at most {card.k_max}.
            </span>
          ) : !live ? (
            <span style={{ fontSize: "16px", color: MUTED }}>
              <Missing what="needs a live sighting" /> for the first; the second needs none
            </span>
          ) : null}
        </span>
      </div>
      {open && health !== null && !card ? (
        <NotYet style={{ margin: 0 }}> This server does not offer the test; a newer one does (its health names causal_check).</NotYet>
      ) : null}
      {/* a test takes seconds (the study's texts, longer): say when it is under way and when it has come back */}
      <p className="visually-hidden" role="status">
        {rows
          .flatMap(({ contexts, entry, why, r }) => {
            const which = contexts === "given" ? "The test at this word-piece" : "The test on the study’s texts";
            if (entry?.pending) return [`${which}: testing…`];
            if (why) return [];
            if (entry?.error) return [`${which} did not come back.`];
            return r ? [`${which} came back; its result follows.`] : [];
          })
          .join(" ")}
      </p>
      {rows.map(({ contexts, entry, why, r, chosen }) => (
        <React.Fragment key={contexts}>
          {why ? (
            <p className="status-line" style={{ margin: 0 }}>
              {contexts === "study" ? "On the study’s texts: " : ""}
              {why}
            </p>
          ) : null}
          {!why ? <ReplyError m={m} error={entry?.error} what={contexts === "given" ? "test at this word-piece" : "test on the study’s texts"} /> : null}
          {r ? <CausalResult m={m} r={r} chosen={chosen} /> : null}
        </React.Fragment>
      ))}
      <p style={{ margin: 0, fontSize: "15px", color: MUTED, maxWidth: "820px" }}>
        A test shows what the neuron does when turned up, not what it does when it fires on its own (the study’s N1, Q2).
      </p>
    </div>
  );
};

/** A face's words in the apparatus: each with c, and its log-probability before → after. */
const causalWordsRaw = (ws: CausalWord[]) =>
  ws.map((w) => `“${w.token}” ${sig(w.dlogprob)} (${sig(w.baseLogprob)} → ${sig(w.steeredLogprob)})`).join(" · ");

/** A label's ten in the apparatus, those among the hundred the push raised most marked ◆. */
const causalLabelRaw = (l: CausalLabel | null) =>
  l ? l.tokens.map((t, i) => `${l.inTop100[i] ? "◆" : ""}“${t}”`).join(" · ") : "—";

/** The full detail of one test, for the apparatus. */
const CausalDetail: React.FC<{ m: Model; r: CausalCheck; why: string | null; sent: Record<string, unknown> | undefined }> = ({ m, r, why, sent }) => {
  const given = r.contexts === "given";
  const near = nearestSampled(m.idx);
  const lin = STUDY_LINEARITY[near];
  const faces = [
    ["+", r.pos, r.floor?.pos ?? null],
    ["−", r.neg, r.floor?.neg ?? null],
  ] as const;
  const ratio = r.actHere != null && r.maxAbsAct ? Math.abs(r.actHere) / r.maxAbsAct : null;
  const ctxRatios = r.actHereByContext && r.maxAbsAct ? r.actHereByContext.map((a) => Math.abs(a) / r.maxAbsAct!) : null;
  const medRatio = ctxRatios && ctxRatios.length ? [...ctxRatios].sort((a, b) => a - b)[Math.floor((ctxRatios.length - 1) / 2)] : null;
  return (
    <dd style={{ margin: 0, display: "flex", flexDirection: "column", gap: "6px", minWidth: 0 }}>
      {why ? <span style={{ color: RED, fontStyle: "italic" }}>Set aside in the main reading: {why}</span> : null}
      <span>
        <b>size</b>: k = {sig(r.k)}, {r.write === "frozen_norm" ? "the frozen-norm write" : "the activation write"}; max|a| = {sig(r.maxAbsAct, 4)},
        over {r.maxAbsActOver ?? "?"}
        {r.maxAbsActCached ? " (kept from an earlier test under the same vault edits)" : ""}
        {given ? (
          <>
            ; a here = {r.actHere != null ? signedRaw(r.actHere) : "?"}
            {ratio != null ? `, which is ${sig(ratio, 2)} of max|a|; the push is ${ratio > 0 ? comma((r.k ?? 0) / ratio) : "∞"} times its own firing here` : ""}
          </>
        ) : (
          <>
            ; a at each text’s last word-piece, as a share of max|a|: median {sig(medRatio, 2)} over {r.nContexts ?? "?"} texts
          </>
        )}
        . δ = {sig(r.pos?.delta ?? null, 4)} and {sig(r.neg?.delta ?? null, 4)}
        {r.kFloor != null ? `; the floor’s, at k = ${r.kFloor}, ${sig(r.floor?.pos?.delta ?? null, 3)} and ${sig(r.floor?.neg?.delta ?? null, 3)}` : "; no floor"}.
      </span>
      <span>
        <b>straight line</b>:{" "}
        {faces.map(([f, x]) =>
          x ? (
            <React.Fragment key={f}>
              {f} face: kept {sig(x.linear.linearity, 3)}
              {x.linear.linearityQ10 != null && !given ? ` (lowest tenth over the texts ${sig(x.linear.linearityQ10, 3)})` : ""} of the
              straight-line size; write against the layer’s feed-forward output {sig(x.linear.writeOverFfn, 2)}; cosine with removing that
              output {sig(x.linear.ablationCos, 2)}.{" "}
            </React.Fragment>
          ) : null,
        )}
        {r.pos?.linear.reason ? <>{r.pos.linear.reason}. </> : null}
        For comparison, at layer {near + 1}, the nearest the study sampled, its k = 10 tests kept {lin.k10Median.toFixed(3)} at the median and{" "}
        {lin.k10Q10.toFixed(3)} at the lowest tenth, and its k = 60 tests {lin.k60Median.toFixed(3)} at the median
        (N1V_audit_01_postnorm.json.full.json). The main reading calls a test too strong below {LINEARITY_TAIL}, the lowest tenth at any
        layer (layer {LINEARITY_TAIL_IDX + 1}).
      </span>
      <span>
        <b>effect</b>:{" "}
        {faces.map(([f, x, fl]) =>
          x ? (
            <React.Fragment key={f}>
              {f} face KL mean {sig(x.klMean)}, median {sig(x.klMedian)}
              {fl ? `; the floor’s ${sig(fl.klMean)}` : ""}
              {x.klOverFloor != null ? `, ${sig(x.klOverFloor, 3)} times the floor` : ""}
              {x.floorTop100Overlap != null ? `; ${x.floorTop100Overlap} of the hundred it raised most are among the floor’s hundred` : ""}
              {!given && x.splitHalfRho != null ? `; split-half ρ ${sig(x.splitHalfRho, 2)}` : ""}.{" "}
            </React.Fragment>
          ) : null,
        )}
        ρ(c+, c−) = {sig(r.antipodalRho, 2)}; {r.top10InNegTop100 ?? "?"} of the ten the + push raised most are among the hundred the − push
        lowered most. The main reading calls an effect rounding noise at or below {NOISE_RATIO_MAX} times the floor (N1V code audit, F2: layer
        11’s 0.0022 against a floor of 0.0008–0.0018 was judged noise). At k = 10 the study found split-half{" "}
        {STUDY_K10_IDX.map((l) => sig(STUDY_K10[l].splitHalf, 2)).join(", ")} and ρ(c+, c−){" "}
        {STUDY_K10_IDX.map((l) => (STUDY_K10[l].antipodal >= 0 ? "+" : "") + sig(STUDY_K10[l].antipodal, 2)).join(", ")}, at layers{" "}
        {STUDY_K10_IDX.map((l) => l + 1).join(", ")} (N1V_audit_04_smallk_metrics.json).
      </span>
      <span>
        <b>words</b>:{" "}
        {faces.map(([f, x]) =>
          x ? (
            <React.Fragment key={f}>
              {f} face rose: {causalWordsRaw(x.rises)}; fell: {causalWordsRaw(x.falls)}
              {x.tracked.length ? `; followed: ${causalWordsRaw(x.tracked)}` : ""}.{" "}
            </React.Fragment>
          ) : null,
        )}
        c is the change of log-probability{given ? "" : ", the mean over the texts, as are the log-probabilities"}.
        {r.baseTop ? ` The untouched row’s likeliest: ${r.baseTop.map((b) => `“${b.token}” ${probWord(b.logprob)}`).join(" · ")}.` : ""}
      </span>
      <span>
        <b>label</b>:{" "}
        {faces.map(([f, x]) =>
          x ? (
            <React.Fragment key={f}>
              {f} face P@10 lens {x.lens?.p10 != null ? x.lens.p10.toFixed(1) : "—"}, naked eye {x.eye?.p10 != null ? x.eye.p10.toFixed(1) : "—"} (lens{" "}
              {causalLabelRaw(x.lens)}; naked eye {causalLabelRaw(x.eye)}; ◆ among the hundred raised most)
              {x.check?.sharedTop10 != null ? `; the two share ${x.check.sharedTop10} of ten` : ""}
              {x.check?.heldBack ? `; ${heldTag(labelGate(layerIdx(m.idx), x.check))}` : ""}.{" "}
            </React.Fragment>
          ) : null,
        )}
        P@10 is the share of the label’s first ten ids among the hundred the push raised most, by signed rise, over the whole vocabulary
        (N1_REPORT). At k = 10 the study’s lens labels scored {STUDY_K10_IDX.map((l) => STUDY_K10[l].p10Lens.toFixed(3)).join(", ")} at
        those layers, and the naked eye’s {STUDY_K10_IDX.map((l) => STUDY_K10[l].p10Eye.toFixed(3)).join(", ")}. Only this comparison goes
        through the lens; the effect does not. The study’s figures are for the base-fitted J++ lens (REPORT §3.4, bug 9), and its agreement
        gate was validated for those labels only (erratum E47).
      </span>
      <span>
        <b>run</b>: {r.forwards ?? "?"} forward{r.forwards === 1 ? "" : "s"}
        {r.slotsPerForward && r.slotsPerForward.length
          ? r.slotsPerForward.length === 1
            ? `, ${r.slotsPerForward[0]} pushed row${r.slotsPerForward[0] === 1 ? "" : "s"} in it`
            : `, pushed rows a forward: ${r.slotsPerForward.join(", ")} (split over separate forwards, which the server does only after running out of memory; each forward kept its own untouched row)`
          : ""}
        ; untouched rows
        identical across forwards: {r.baseRowsIdentical == null ? "?" : r.baseRowsIdentical ? "yes" : "no"}; vault edits in force:{" "}
        {r.vaultEdits.length ? r.vaultEdits.map((p) => `#${p}`).join(", ") : "none"}; read without any nudge
        {r.steered === false ? " (steered: false)" : ""}
        {given ? `; at position ${r.readPos ?? "?"}${r.readToken != null ? `, “${r.readToken}”` : ""}${r.tokenIdsGiven ? ", the last of the ids sent" : ""}` : r.studyTexts ? `; the texts: ${r.studyTexts}` : ""}
        {r.timing?.totalMs != null ? `; ${ms(r.timing.totalMs)}` : ""}.
      </span>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 14px" }}>
        <RawToggle label="the raw reply" data={r.raw} />
        <div>
          <button
            className="btn-ghost"
            type="button"
            onClick={() =>
              downloadJson(`causal-check-layer${m.layerNo}-neuron${m.neuron}-${r.contexts}.json`, { request: sent ?? null, reply: r.raw })
            }
          >
            download the test
          </button>
        </div>
      </div>
    </dd>
  );
};

/** The test's rows in the apparatus of ii: the formula, each kept test in full, the controls, and the request by hand. */
const CausalApparatus: React.FC<{ m: Model; ask: CausalAsk; setAsk: (a: Partial<CausalAsk>) => void }> = ({ m, ask, setAsk }) => {
  const givenEntry = useTheodoliteStore((s) => s.causalChecks[causalKey(m.idx, m.neuron, "given")]) as Pending<unknown> | undefined;
  const studyEntry = useTheodoliteStore((s) => s.causalChecks[causalKey(m.idx, m.neuron, "study")]) as Pending<unknown> | undefined;
  const activeEdits = useTheodoliteStore((s) => s.activeEdits);
  const card = useTheodoliteStore((s) => s.health?.causal_check ?? null);
  const editsNow = useMemo(() => editIdsOf(activeEdits), [activeEdits]);
  const [kText, setKText] = useState(String(ask.k));
  const s = m.sighting;
  const ids = s?.source === "live" && s.contextIds?.length ? s.contextIds : null;
  const pol = m.focusFace > 0 ? "1.0" : "-1.0";
  const track = s?.source === "live" && s.chosen ? `[${s.chosen.id}]` : "[]";
  const idList = ids ? `[${ids.join(", ")}]` : "<A_LIVE_SIGHTING'S_IDS>";
  const opts = `k=${ask.k}, write="${ask.write}"`;
  const code =
    "# MCP (backend/mcp_scientist.py)\n" +
    `causal_spot_check(layer=${m.idx}, neuron_idx=${m.neuron}, polarity=${pol}, contexts="given",\n` +
    `    token_ids=${idList}, track_ids=${track},\n    ${opts}, scale_ref="${ask.scaleRef}", floor=True, top_n=10)\n` +
    `causal_spot_check(layer=${m.idx}, neuron_idx=${m.neuron}, polarity=${pol}, contexts="study", ${opts}, scale_ref="study")\n\n` +
    "# HTTP\nPOST /api/scientist/causal_check\n" +
    `{"layer": ${m.idx}, "neuron_idx": ${m.neuron}, "polarity": ${pol}, "contexts": "study", "k": ${ask.k}, "write": "${ask.write}",\n` +
    ` "scale_ref": "study", "floor": true, "top_n": 10}\n\n` +
    "# WebSocket /ws/theodolite\n" +
    `{"action": "causal_check", "layer": ${m.idx}, "neuron_idx": ${m.neuron}, "polarity": ${pol}, "contexts": "given",\n` +
    ` "token_ids": ${idList}, "track_ids": ${track}, "k": ${ask.k}, "write": "${ask.write}", "scale_ref": "${ask.scaleRef}", "floor": true, "top_n": 10}`;
  const kept = (["given", "study"] as const).map((contexts) => {
    const entry = contexts === "given" ? givenEntry : studyEntry;
    const r = entry?.data ? parseCausalCheck(entry.data) : null;
    return { contexts, entry, r, why: r && entry ? causalStaleWhy(entry, contexts, s, editsNow) : null };
  });
  return (
    <>
      <dt>causal check</dt>
      <dd>
        the activation write, a<sub>i</sub> ← a<sub>i</sub> + s · k · max|a<sub>i</sub>| at the read position p only (the study’s N1A,
        experiments/scripts/N1A_02_causal_core.py); or the frozen-norm write, where the post-feed-forward norm’s output at p gains s · k ·
        max|a<sub>i</sub>| · (w<sub>post</sub> ⊙ w<sub>down,i</sub>) / rms(y<sub>p</sub>), the same first-order change, which cannot squash
        the norm (N1V, experiments/scripts/N1V_02_causal.py). Then c<sub>s</sub> = log_softmax(z<sub>pushed</sub>) −
        log_softmax(z<sub>untouched</sub>), on fp32 softcapped logits at p, over the whole vocabulary; on the study’s texts c<sub>s</sub> is
        the mean over the 36 (12 chats, 12 passages of prose, 12 of code). The untouched row sits in the same forward as the pushes, because
        rows of one forward are bit-identical and different batch shapes were not (KL up to 0.014). The floor is the same test at k = 0.01,
        the study’s measure of bf16 rounding. w<sub>down,i</sub> is the column as the model now uses it, vault edits included; nothing is
        nudged. Proposed by the study (N1, recommendation 5; REPORT §3.3, “causally spot-checked”); not in the original instrument.
      </dd>
      {kept.map(({ contexts, entry, r, why }) =>
        r ? (
          <React.Fragment key={contexts}>
            <dt>{contexts === "given" ? "at this word-piece" : "on the 36 texts"}</dt>
            <CausalDetail m={m} r={r} why={why} sent={entry?.sent} />
          </React.Fragment>
        ) : null,
      )}
      <dt>test size</dt>
      <dd style={{ margin: 0, display: "flex", flexWrap: "wrap", alignItems: "center", gap: "6px 24px" }}>
        <label style={{ display: "flex", alignItems: "center", gap: "8px", minHeight: "44px" }}>
          times its strongest firing{" "}
          <input
            type="number"
            min={card?.k_floor ?? 0.01}
            max={card?.k_max ?? 200}
            step="any"
            value={kText}
            onChange={(e) => {
              setKText(e.target.value);
              const v = parseFloat(e.target.value);
              setAsk({ k: Number.isFinite(v) ? v : NaN });
            }}
            style={{ ...NUM_INPUT, width: "72px", color: "inherit" }}
          />{" "}
          <span className="mono" style={{ color: MUTED }}>
            k
          </span>
        </label>
        <div role="group" aria-label="How the test writes" style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
          <button type="button" className="btn-ghost plain" aria-pressed={ask.write === "activation"} onClick={() => setAsk({ write: "activation" })}>
            on the activation, as the study did
          </button>
          <button type="button" className="btn-ghost plain" aria-pressed={ask.write === "frozen_norm"} onClick={() => setAsk({ write: "frozen_norm" })}>
            after the norm (frozen norm)
          </button>
        </div>
        <div role="group" aria-label="What sizes the test" style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
          <button type="button" className="btn-ghost plain" aria-pressed={ask.scaleRef === "study"} onClick={() => setAsk({ scaleRef: "study" })}>
            max|a| on the study’s texts
          </button>
          <button type="button" className="btn-ghost plain" aria-pressed={ask.scaleRef === "this_text"} onClick={() => setAsk({ scaleRef: "this_text" })}>
            max|a| on this text
          </button>
        </div>
        <span style={{ flex: "1 1 100%", color: MUTED, fontStyle: "italic" }}>
          Sent with the next test: k more than {card?.k_floor ?? 0.01} and at most {card?.k_max ?? 200} (the study tested at 3, 10, 60 and 180;
          E47 advises 10, where its tests stayed close to straight). The frozen-norm write is no remedy for a test too strong: at 60 times it
          still lost the two faces’ opposition at layers 21 to 31 (N1V robustness, V2b). Sizing by this text applies to the test at a word-piece; the test on the
          study’s texts is always sized on them.
        </span>
      </dd>
      <dt>by hand</dt>
      <dd style={{ margin: 0, display: "flex", flexDirection: "column", gap: "6px", minWidth: 0 }}>
        <span>
          {ids
            ? `The test at this word-piece sends this sighting’s exact word-pieces, ${ids.length} ids up to the one read here, and pushes the last of them; no template is applied and nothing is tokenized again. ${
                s?.chosen?.source === "given"
                  ? "It follows the word-piece given here."
                  : s?.chosen?.endMark
                    ? "It follows the mark the model wrote here, where writing stopped."
                    : s?.chosen
                      ? "It follows the word the model chose here."
                      : "It follows no word-piece: none was chosen here."
              }`
            : "The test at a word-piece needs a live sighting’s exact ids; the recorded sighting keeps none."}{" "}
          The answer comes back as {"{"}"type": "causal_check", "data": …{"}"}; a refusal, in plain words, as a recoverable error naming the
          contexts (ARCHITECTURE §4.5).
        </span>
        <pre style={PRE}>{code}</pre>
      </dd>
    </>
  );
};

/* ------------------------------------------------------ iii · who listens */

/**
 * Why a trace kept for this neuron is set aside for the sighting shown: it was read at another word-piece of the same
 * question (named, from the ids it sent, or from the sighting it was asked for), on another answer to that question, with
 * other options on this very text, or for another question. A trace depends only on the text read up to its position and on
 * the vault, never on the frame or the ranking, but a re-reading is a new sighting, so the page asks for it again.
 */
function staleWhy(stored: Pending<unknown>, s: Sighting, set: AnswerSet | null): string {
  const again = "Trace again to read them for this one.";
  const sent = stored.sent ?? {};
  // the sighting it was asked for, when it is another word-piece of the answer shown
  const traced = set ? (Object.values(set.sightings).find((x) => x.id === stored.sightingId) ?? null) : null;
  if (traced && traced.tokenIndex !== s.tokenIndex) {
    return (
      `Its readers were traced at another word-piece (${traced.readAt != null ? pieceWord(traced.readAt) : "word-piece " + (traced.tokenIndex + 1)}) ` +
      `of this answer, so they are set aside here. ${again}`
    );
  }
  const prompt = typeof sent.prompt === "string" ? sent.prompt : null;
  if (prompt === null || s.textSent === null || prompt !== s.textSent) {
    return `Its readers were traced for an earlier sighting; the question has changed since, so they are set aside. ${again}`;
  }
  // the same question: the word-piece read was the last of the ids sent, or, with none sent, the prompt's own last
  const ids = Array.isArray(sent.token_ids) ? (sent.token_ids as unknown[]) : null;
  const P = s.promptLen ?? s.promptIds?.length ?? null;
  const context = s.promptIds && s.answerIds ? [...s.promptIds, ...s.answerIds] : s.promptIds;
  const pos = ids ? ids.length - 1 : P !== null ? P - 1 : null;
  const onThisAnswer = !ids || (!!context && ids.length <= context.length && ids.every((v, i) => v === context[i]));
  if (!onThisAnswer || pos === null || P === null) {
    return `Its readers were traced on another answer to this same question, so they are set aside. ${again}`;
  }
  if (pos === s.readPos) {
    return `Its readers were traced on this very text, up to the same word-piece, for a sighting taken with other options. ${again}`;
  }
  const piece = pos < P ? (s.promptTokens?.[pos] ?? null) : (s.answerStrs?.[pos - P] ?? null);
  return `Its readers were traced at another word-piece${piece != null ? ` (${pieceWord(piece)})` : ""} of this question, so they are set aside. ${again}`;
}

const Listen: React.FC<{ m: Model }> = ({ m }) => {
  const traceReaders = useTheodoliteStore((s) => s.traceReaders);
  const storeScore = useTheodoliteStore((s) => s.threadScore);
  const stored = useTheodoliteStore((s) => s.readers[poleKey(m.idx, m.neuron)]) as Pending<unknown> | undefined;
  const shownSet = useShownSet();
  // a trace answers for the text it was traced on, up to the word-piece read: one asked of another sighting is set aside, not
  // shown as this one's (stepping back to the word-piece it was traced at shows it again)
  const stale = !!stored && stored.sightingId != null && m.sighting != null && stored.sightingId !== m.sighting.id;
  const s = m.sighting;
  const exact = s?.source === "live" && !!s.contextIds?.length;
  const entry = stale ? undefined : stored;
  const reply: Readers | null = entry?.data ? parseReaders(entry.data) : null;
  const pending = entry?.pending;
  const hopLimit = Math.max(0, Math.min(6, 60 - m.layerNo));
  // what the next trace asks for (max_layer_hop, top_k, score); the drawing follows what the reply says it used
  const [hopAsk, setHopAsk] = useState(hopLimit);
  const [kAsk, setKAsk] = useState(10);
  const [score, setScore] = useState<"coupling" | "direct">(storeScore);
  const hopN = Math.max(0, Math.min(hopLimit, reply?.hop ?? hopAsk));
  const kN = reply?.topK ?? kAsk;
  const kMax = 25 * hopN;
  const kind = reply?.scoreKind ?? score;
  const sym = kind === "direct" ? "D" : "C";
  const first = m.layerNo + 1;
  const last = m.layerNo + hopN;
  const hopWord = hopN === 1 ? `layer, ${first}` : `${numberWord(hopN)} layers, ${first} to ${last}`;
  const cols = Array.from({ length: hopN }, (_, i) => ({ idx: m.idx + 1 + i, no: first + i, x: 190 + i * 100 }));
  const mid = cols.length ? (cols[0].x + cols[cols.length - 1].x) / 2 : 440;
  const readers = reply?.readers ?? [];
  const maxC = readers.reduce((a, r) => Math.max(a, Math.abs(r.score)), 0) || 1;

  // place readers in their layer's column, strongest at the middle
  const placed = useMemo(() => {
    const out: { r: (typeof readers)[number]; x: number; y: number }[] = [];
    for (const col of cols) {
      const here = readers.filter((r) => r.idx === col.idx);
      const n = here.length;
      const step = n > 1 ? Math.min(22, 104 / (n - 1)) : 0;
      here.forEach((r, k) => out.push({ r, x: col.x, y: f1(93 + (k - (n - 1) / 2) * step) }));
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reply, m.idx, hopN]);

  const traced = !!entry && (!!entry.pending || !!entry.data);
  const showThreads = !!reply && readers.length > 0;
  const boxText = pending
    ? "[TRACING…]"
    : reply
      ? readers.length
        ? null
        : "[NO READERS CAME BACK]"
      : entry?.error
        ? m.conn !== "open"
          ? "[THE INSTRUMENT ANSWERS WHEN THE SERVER IS RUNNING]"
          : "[THE TRACE DID NOT COME BACK]"
        : "[READERS APPEAR AFTER TRACING]";
  const aria = showThreads
    ? `Layers ${first} to ${last}, with ${readers.length} threads drawn from this neuron to the later neurons that read it, the strongest drawn thickest.`
    : traced
      ? `Layers ${first} to ${last}, where the traced readers of this neuron will be drawn as threads.`
      : `${cap(numberWord(hopN))} empty layers, ${first} to ${last}, waiting to be traced for neurons that listen to this one.`;

  return (
    <section aria-labelledby="listen-head" style={SECTION}>
      <div className="row-between">
        <h2 className="h-sec" id="listen-head">
          iii · who listens to it
        </h2>
        <span style={ASIDE}>
          threads <span style={{ fontStyle: "italic" }}>(reader filaments)</span>
        </span>
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "28px 60px", alignItems: "flex-start" }}>
        <div style={{ flex: "999 1 600px", minWidth: 0, display: "flex", flexDirection: "column", gap: "18px" }}>
          {hopN > 0 ? (
            <p style={{ margin: 0, maxWidth: "680px" }}>
              Later neurons that read what this one writes are its threads. Tracing looks through the next {hopWord}, and lists up to{" "}
              {kN <= 20 ? numberWord(kN) : String(kN)} readers, the strongest thread first
              {kind === "direct" ? ", scored by the direct-path estimate" : ""}.
            </p>
          ) : (
            <p style={{ margin: 0, maxWidth: "680px" }}>
              Later neurons that read what this one writes are its threads. Layer 60 is the last, so no later neuron can read this one.
            </p>
          )}
          <div style={{ display: "flex", flexWrap: "wrap", gap: "12px 20px", alignItems: "center" }}>
            <button
              className="btn-primary"
              type="button"
              disabled={hopLimit === 0 || pending}
              onClick={() => traceReaders(m.idx, m.neuron, m.focusFace, { hop: hopAsk, topK: kAsk, score })}
            >
              {pending ? "tracing…" : "trace its readers"}
            </button>
            <span style={{ fontSize: "16px", color: MUTED }}>
              {exact
                ? `reads this very text, up to the same word-piece${m.readAt ? `, ${m.readAt}${m.inReply}` : ""}`
                : `uses the same question and the same word${m.readAt ? `, ${m.readAt}` : ""}`}
              {s?.isNudged ? "; the trace reads the model without the nudge, on this very text" : ""}
            </span>
          </div>
          {stale && stored && s ? (
            <p className="status-line" style={{ margin: 0 }}>
              {staleWhy(stored, s, shownSet)}
            </p>
          ) : null}
          {reply && s && s.tokenIndex >= 1 && reply.idsGiven !== true ? (
            <p className="error-line" role="alert" style={{ margin: 0 }}>
              This server read the text sent, not the exact word-pieces: its trace was read at the end of the question
              {reply.readPos != null ? ` (position ${reply.readPos})` : ""}, not at this word-piece (position {s.readPos ?? "?"}).
            </p>
          ) : null}
          <ReplyError m={m} error={entry?.error} what="trace" />
          <ServerErrorWhilePending pending={pending} />
          {hopN > 0 ? (
            <div style={{ overflowX: "auto" }}>
              <div style={{ minWidth: "560px" }}>
                <svg viewBox="0 0 760 190" role="img" aria-label={aria} style={{ width: "100%", height: "auto", display: "block" }}>
                  {showThreads
                    ? placed.map(({ r, x, y }, i) => (
                        <path
                          key={`t${i}`}
                          d={`M74,95 C${f1(74 + (x - 74) / 2)},95 ${f1(74 + (x - 74) / 2)},${y} ${x - 8},${y}`}
                          fill="none"
                          stroke={r.score >= 0 ? GILT_TEXT : INK}
                          strokeOpacity={r.score >= 0 ? 0.85 : 0.6}
                          strokeWidth={f1(0.8 + 2.4 * (Math.abs(r.score) / maxC))}
                          strokeDasharray={r.score >= 0 ? undefined : "3 3"}
                        />
                      ))
                    : null}
                  <path
                    d="M60,81 L74,95 L60,109 L46,95 Z"
                    fill={m.fired ? GILT : "#F6F0E3"}
                    stroke={GILT_TEXT}
                    strokeWidth={m.fired ? 0.8 : 1.2}
                  />
                  {cols.map((c) => (
                    <line
                      key={c.idx}
                      x1={c.x}
                      y1="34"
                      x2={c.x}
                      y2="152"
                      stroke={isGlobalIdx(c.idx) ? GILT : INK}
                      strokeOpacity={isGlobalIdx(c.idx) ? 0.8 : 0.2}
                      strokeWidth={isGlobalIdx(c.idx) ? 1.1 : 1}
                    />
                  ))}
                  {showThreads
                    ? placed.map(({ r, x, y }, i) => (
                        <g key={`r${i}`}>
                          <path
                            d={`M${x},${y - 7} L${x + 7},${y} L${x},${y + 7} L${x - 7},${y} Z`}
                            fill={r.face > 0 ? GILT : "#F6F0E3"}
                            stroke={GILT_TEXT}
                            strokeWidth="1"
                          />
                          <text x={x + 10} y={y + 4} fontFamily={SERIF} fontSize="11" fill={MUTED} style={NUM}>
                            {r.neuron}
                          </text>
                        </g>
                      ))
                    : null}
                  {boxText ? (
                    <g>
                      <rect x={mid - 158} y="80" width="316" height="28" fill="#F6F0E3" />
                      <text
                        x={mid}
                        y="99"
                        fontFamily={SERIF}
                        textAnchor="middle"
                        fontSize={boxText.length > 34 ? 10.5 : 13}
                        letterSpacing="1.6"
                        fill={MUTED}
                      >
                        {boxText}
                      </text>
                    </g>
                  ) : null}
                  <g fontFamily={SERIF} textAnchor="middle" style={NUM}>
                    <text x="60" y="138" fontSize="15" fill={INK}>
                      {m.layerNo}
                    </text>
                    <text x="60" y="158" fontSize="13" fontStyle="italic" fill={MUTED}>
                      this neuron
                    </text>
                    {cols.map((c) => (
                      <React.Fragment key={c.idx}>
                        <text x={c.x} y="176" fontSize="13" fill={isGlobalIdx(c.idx) ? GILT_TEXT : MUTED}>
                          {c.no}
                        </text>
                        {isGlobalIdx(c.idx) ? (
                          <text x={c.x} y="24" fontSize="14" fill={GILT}>
                            ✦
                          </text>
                        ) : null}
                      </React.Fragment>
                    ))}
                    <text x={mid} y="24" fontSize="12" letterSpacing="2.4" fill={MUTED}>
                      LAYER
                    </text>
                  </g>
                </svg>
              </div>
            </div>
          ) : null}
          {showThreads ? (
            <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 22px", fontSize: "15px", color: MUTED }}>
              <span>solid gilt: a thread that opens the reader ({sym} &gt; 0)</span>
              <span>dashed ink: a thread that closes it ({sym} &lt; 0)</span>
              <span>thicker: a larger |{sym}|</span>
            </div>
          ) : null}
          {traced ? (
            <div style={{ display: "flex", flexDirection: "column" }}>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "minmax(0, 1.2fr) minmax(0, 0.8fr) minmax(0, 1fr) minmax(0, 1fr)",
                  gap: "0 16px",
                  fontSize: "13px",
                  fontVariantCaps: "all-small-caps",
                  letterSpacing: "0.16em",
                  color: MUTED,
                  paddingBottom: "4px",
                  borderBottom: `1px solid ${HAIR}`,
                }}
              >
                <span>reader</span>
                <span>layer</span>
                <span>face</span>
                <span style={{ textAlign: "right" }}>thread score{kind === "direct" ? ", D" : ", C"}</span>
              </div>
              {readers.length ? (
                readers.map((r, i) => (
                  <div
                    key={i}
                    style={{
                      display: "grid",
                      gridTemplateColumns: "minmax(0, 1.2fr) minmax(0, 0.8fr) minmax(0, 1fr) minmax(0, 1fr)",
                      gap: "0 16px",
                      alignItems: "baseline",
                      padding: "4px 0",
                      borderBottom: `1px solid ${HAIR_FAINT}`,
                      fontSize: "17px",
                      ...NUM,
                    }}
                  >
                    <span style={{ display: "flex", flexDirection: "column", minWidth: 0 }}>
                      <a href={href.neuron(r.idx + 1, r.neuron, r.face)} style={{ display: "inline-flex", alignItems: "center", minHeight: "44px" }}>
                        neuron {r.neuron}
                      </a>
                      {r.words?.length ? (
                        // a reader's words pass the same gate: by its own dst_label_check, else by its layer alone
                        labelGate(layerIdx(r.idx), r.check).heldBack ? (
                          <span style={{ fontSize: "14px", fontStyle: "italic", color: MUTED, overflowWrap: "anywhere", paddingBottom: "6px" }}>
                            <HeldBackWords
                              gate={labelGate(layerIdx(r.idx), r.check)}
                              words={r.words.map((t) => t.token)}
                              eye={r.eye?.map((t) => t.token) ?? null}
                              compact
                            />
                          </span>
                        ) : (
                          <span style={{ fontSize: "14px", fontStyle: "italic", color: MUTED, overflowWrap: "anywhere", paddingBottom: "6px" }}>
                            writes {r.words.map((t) => wp(t.token)).join(" ")}
                          </span>
                        )
                      ) : null}
                    </span>
                    <span>{r.idx + 1}</span>
                    <span style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "16px" }}>
                      <FaceMark filled={r.face > 0} />
                      {faceWord(r.face)}
                    </span>
                    <span style={{ textAlign: "right" }} title={`C ${signed(r.c, 3)}${r.directGate != null ? ` · D ${signed(r.directGate, 3)}` : ""}${r.directAct != null ? ` · change of the reader’s activation ${signed(r.directAct, 3)}` : ""}`}>
                      {signed(r.score, kind === "direct" ? 3 : 2)}
                    </span>
                  </div>
                ))
              ) : (
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "minmax(0, 1.2fr) minmax(0, 0.8fr) minmax(0, 1fr) minmax(0, 1fr)",
                    gap: "0 16px",
                    alignItems: "baseline",
                    padding: "12px 0",
                    borderBottom: `1px solid ${HAIR_FAINT}`,
                    fontSize: "14px",
                    fontVariantCaps: "all-small-caps",
                    letterSpacing: "0.12em",
                    color: MUTED,
                  }}
                >
                  <span>[READER NEURON]</span>
                  <span>
                    [{first}–{last}]
                  </span>
                  <span>[FACE]</span>
                  <span style={{ textAlign: "right" }}>[SCORE]</span>
                </div>
              )}
              <p style={{ margin: "10px 0 0", fontSize: "15px", color: MUTED }}>
                Up to {kN <= 20 ? numberWord(kN) : String(kN)} rows, the strongest thread first. Open any reader to read its own folio.
              </p>
            </div>
          ) : null}
        </div>
        <div style={{ flex: "1 1 340px", minWidth: 0, display: "flex", flexDirection: "column", gap: "14px" }}>
          <Rubric>
            {kind === "direct"
              ? "Scored by the direct-path estimate, which in the study matched a reader’s measured effect with a correlation of 0.87, against about 0.4 for the coupling C. Still treat each thread as a lead to test with a nudge, not a wire you have found."
              : "A thread’s score is a weak guide. In the study it matched a reader’s measured effect with a correlation of only about 0.4, where a simple direct-path estimate reached 0.87. Treat each thread as a lead to test with a nudge, not a wire you have found."}
          </Rubric>
          <p style={{ margin: 0, fontSize: "16px", color: MUTED }}>
            The score is a linear estimate, not a measured circuit.{" "}
            {kind === "direct"
              ? "The coupling C, as the instrument first scored threads, can be chosen under “score by” in the apparatus."
              : "The direct-path estimate can be chosen under “score by” in the apparatus."}
          </p>
        </div>
      </div>
      <ThreadsApparatus
        m={m}
        hopN={hopN}
        kN={kN}
        kMax={kMax}
        reply={reply}
        ask={{ hop: hopAsk, k: kAsk, score, hopLimit }}
        onAsk={(a) => {
          if (a.hop !== undefined) setHopAsk(Math.max(1, Math.min(hopLimit, a.hop)));
          if (a.k !== undefined) setKAsk(Math.max(1, Math.min(150, a.k)));
          if (a.score !== undefined) setScore(a.score);
        }}
      />
      <Builds m={m} />
    </section>
  );
};

/* ------------------------------------------------------ iii · what builds it */

/** "+12%", "−3.4%": a part's share of the drive, signed. */
const sharePct = (x: number | null | undefined) => (x == null ? "" : (x >= 0 ? "+" : "−") + pct(Math.abs(x)));

/** An earlier neuron's words on one face, under the label gate (its own layer's rule and its index check). */
const WriterWords: React.FC<{ layer: number; w: FaceWords | undefined }> = ({ layer, w }) => {
  if (!w || !w.lens || !w.lens.length) return null;
  const gate = labelGate(layerIdx(layer), parseLabelCheck(w.check));
  return (
    <span style={{ fontSize: "14px", fontStyle: "italic", color: MUTED, overflowWrap: "anywhere", paddingBottom: "6px" }}>
      {gate.heldBack ? <HeldBackWords gate={gate} words={w.lens} eye={w.eye} compact /> : <>writes {w.lens.slice(0, 5).map(wp).join(" ")}</>}
    </span>
  );
};

interface BuildsRead {
  sightingId: number | null;
  drive: Drive | null;
  driveWhy: string | null;
  wiring: Wiring | null;
  wiringWhy: string | null;
}

/**
 * What builds the neuron, beside who listens to it (Lineage practice v): in use, at the sighting's own word-piece, the
 * drive (POST /api/atlas/attn_drive with the sighting's ids), which splits its activation over the heads, the earlier
 * feed-forward blocks and their neurons, and the embedding; by the weights alone, its strongest writers (GET
 * /api/atlas/neuron/wiring, `up`), each on the face that opens its gate. Read when asked: the drive runs the model.
 */
const Builds: React.FC<{ m: Model }> = ({ m }) => {
  const s = m.sighting;
  const ids = s?.source === "live" && s.contextIds?.length ? s.contextIds : null;
  const [got, setGot] = useState<BuildsRead | null>(null);
  const [busy, setBusy] = useState(false);
  const mine = got && (got.sightingId === (s?.id ?? null) || !got.drive) ? got : null;
  const read = async () => {
    setBusy(true);
    const noIds = !s
      ? "no sighting is loaded"
      : s.source !== "live"
        ? "the recorded sighting keeps no word-piece ids to read again; take a live sighting"
        : "this server did not send the sighting’s word-piece ids";
    const [d, w] = await Promise.allSettled([
      ids ? atlasPost<Drive>("/api/atlas/attn_drive", { layer: m.idx, n: m.neuron, ids, at: -1, top: 8 }) : Promise.reject(new Error(noIds)),
      atlasGet<Wiring>("/api/atlas/neuron/wiring", { layer: m.idx, n: m.neuron, k: 24 }),
    ]);
    setGot({
      sightingId: s?.id ?? null,
      drive: d.status === "fulfilled" ? d.value : null,
      driveWhy: d.status === "rejected" ? String((d.reason as Error)?.message ?? d.reason) : null,
      wiring: w.status === "fulfilled" ? w.value : null,
      wiringWhy: w.status === "rejected" ? String((w.reason as Error)?.message ?? w.reason) : null,
    });
    setBusy(false);
  };
  const d = mine?.drive ?? null;
  const writers = (mine?.wiring?.up ?? []).slice(0, 8);
  const row: React.CSSProperties = {
    display: "grid",
    gridTemplateColumns: "minmax(0, 1.5fr) minmax(0, 0.5fr) minmax(0, 0.8fr) minmax(0, 0.6fr)",
    gap: "0 16px",
    alignItems: "baseline",
    padding: "4px 0",
    borderBottom: `1px solid ${HAIR_FAINT}`,
    fontSize: "17px",
    ...NUM,
  };
  const head: React.CSSProperties = {
    ...row,
    fontSize: "13px",
    fontVariantCaps: "all-small-caps",
    letterSpacing: "0.16em",
    color: MUTED,
    borderBottom: `1px solid ${HAIR}`,
    padding: "0 0 4px",
  };
  const neuronRow = (l: number, n: number, face: "+" | "-", amount: string, w: FaceWords | undefined, name: string | undefined, key: string) => (
    <div key={key} style={row}>
      <span style={{ display: "flex", flexDirection: "column", minWidth: 0 }}>
        <a href={href.neuron(l + 1, n, face === "+" ? 1 : -1)} style={{ display: "inline-flex", alignItems: "center", minHeight: "44px" }}>
          neuron {n}
          {name ? <span style={{ fontStyle: "italic", color: MUTED }}>&nbsp;· {name}</span> : null}
        </a>
        <WriterWords layer={l} w={w} />
      </span>
      <span>{l + 1}</span>
      <span style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "16px" }}>
        <FaceMark filled={face === "+"} />
        {faceWord(face === "+" ? 1 : -1)}
      </span>
      <span style={{ textAlign: "right" }}>{amount}</span>
    </div>
  );
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px", paddingTop: "18px", borderTop: `1px solid ${HAIR_FAINT}` }}>
      <div className="row-between">
        <h3 className="h-sub" id="builds-head">
          what builds it
        </h3>
        <span style={ASIDE}>looking upstream</span>
      </div>
      <p style={{ margin: 0, maxWidth: "760px" }}>
        Earlier parts of the model write what this neuron reads. In use, at this very word-piece, its activation splits over every attention head,
        every earlier feed-forward block, and each block over its neurons. By the weights alone, its strongest writers are leads to test, whatever
        the text.
      </p>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "12px 20px", alignItems: "center" }}>
        <button className="btn-ghost" type="button" disabled={busy} onClick={() => void read()}>
          {busy ? "reading…" : mine ? "read it again" : "read what builds it"}
        </button>
        <span style={{ fontSize: "16px", color: MUTED }}>
          {ids
            ? `reads this very text, its ${numberWord(ids.length)} word-pieces, at the one read${m.readAt ? `, ${m.readAt}${m.inReply}` : ""}`
            : "the weights alone: no word-piece ids of a live sighting to read the drive on"}
        </span>
      </div>
      {got && !mine ? (
        <p className="status-line" style={{ margin: 0 }}>
          What builds it was read for another sighting, so it is set aside. Read it again for this one.
        </p>
      ) : null}
      {mine ? (
        <div style={{ display: "flex", flexWrap: "wrap", gap: "24px 56px", alignItems: "flex-start" }}>
          <div style={{ flex: "999 1 560px", minWidth: 0, display: "flex", flexDirection: "column", gap: "12px" }}>
            <h4 style={{ margin: 0, ...SC_LABEL }}>in use at this word-piece</h4>
            {d ? (
              <>
                <p style={{ margin: 0 }}>
                  At “{d.token}” it reads a = {signed(d.act, 3)}; linearised where it reads the stream, {signed(d.linearised_total, 3)}, of which
                  the attention heads give {sharePct(d.linearised_total ? d.attention_sum / d.linearised_total : null)}, the earlier feed-forward
                  blocks {sharePct(d.linearised_total ? d.mlp_sum / d.linearised_total : null)} and the word-piece’s own embedding{" "}
                  {sharePct(d.linearised_total ? d.embedding / d.linearised_total : null)}.
                </p>
                <p style={{ margin: 0, fontSize: "16px", color: MUTED }}>
                  heads:{" "}
                  {d.heads
                    .slice(0, 5)
                    .map((h) => `layer ${h.layer + 1} head ${h.head} ${sharePct(h.share)}${h.from?.length ? `, reading “${h.from[0][0]}”` : ""}`)
                    .join(" · ")}
                </p>
                <p style={{ margin: 0, fontSize: "16px", color: MUTED }}>
                  blocks:{" "}
                  {d.mlps
                    .slice(0, 4)
                    .map((b) => `layer ${b.layer + 1} ${sharePct(d.linearised_total ? b.contribution / d.linearised_total : null)}`)
                    .join(" · ")}
                </p>
                {d.neurons?.length ? (
                  <div style={{ display: "flex", flexDirection: "column" }}>
                    <div style={head}>
                      <span>earlier neuron</span>
                      <span>layer</span>
                      <span>fired on</span>
                      <span style={{ textAlign: "right" }}>share</span>
                    </div>
                    {d.neurons.map((x) => neuronRow(x.layer, x.n, x.face, sharePct(x.share), x.words, x.name, `d${x.layer}/${x.n}`))}
                  </div>
                ) : null}
              </>
            ) : (
              <p className="status-line" style={{ margin: 0 }}>
                The drive was not read: {mine.driveWhy}
              </p>
            )}
          </div>
          <div style={{ flex: "1 1 380px", minWidth: 0, display: "flex", flexDirection: "column", gap: "12px" }}>
            <h4 style={{ margin: 0, ...SC_LABEL }}>by the weights alone, as leads</h4>
            {writers.length ? (
              <div style={{ display: "flex", flexDirection: "column" }}>
                <div style={head}>
                  <span>writer</span>
                  <span>layer</span>
                  <span>opens it on</span>
                  <span style={{ textAlign: "right" }}>strength</span>
                </div>
                {writers.map((c) =>
                  neuronRow(c.layer, c.n, c.face ?? (c.gate != null && c.gate < 0 ? "-" : "+"), c.w != null ? pct(c.w) : "", c.words, c.name, `w${c.layer}/${c.n}`),
                )}
              </div>
            ) : (
              <p className="status-line" style={{ margin: 0 }}>
                {mine.wiringWhy ? `Its writers were not read: ${mine.wiringWhy}` : m.idx === 0 ? "Nothing writes into the first layer’s neurons." : "No writers came back."}
              </p>
            )}
            <span style={{ fontSize: "15px", color: MUTED }}>
              each writer on the face that opens this neuron’s gate, read from rest; the strength as it arrives, over the strongest
            </span>
          </div>
        </div>
      ) : null}
      <Rubric style={{ maxWidth: "760px" }}>
        Both are leads, not causes. The drive is exact arithmetic at one word-piece, but a neuron’s share counts only its own write straight into
        this layer, the norms held at their scale there; what it does through the layers between is in those layers’ parts. The weights alone know
        nothing of the text. Test a lead by silencing it and reading this neuron again.
      </Rubric>
      <Apparatus title="what builds it" label="Apparatus for what builds it">
        <dl className="dl num">
          <dt>the drive</dt>
          <dd>
            a = gelu(g) · u, g = W<sub>gate</sub>[n] · x̂, u = W<sub>up</sub>[n] · x̂, x̂ = w<sub>pre</sub> ⊙ mid / rms(mid), linearised in mid,
            the stream entering the neuron’s feed-forward block, with the rms held: J = (gelu′(g) u W<sub>gate</sub>[n] + gelu(g) W<sub>up</sub>[n])
            ⊙ w<sub>pre</sub> / rms(mid). mid is a sum of parts, each carried by the layer scalars it passed, so J · mid splits exactly over every
            head (and the word-pieces it read), every earlier block and the embedding. Block k’s part is J · (s-carried w<sub>post,k</sub> ⊙ y
            <sub>k</sub> / rms(y<sub>k</sub>)), y<sub>k</sub> = Σ<sub>A</sub> a<sub>A</sub> W<sub>down,k</sub>[:, A], so neuron A’s share is a
            <sub>A</sub> (J′<sub>k</sub> · W<sub>down,k</sub>[:, A]), J′<sub>k</sub> = s-carried (w<sub>post,k</sub> ⊙ J) / rms(y<sub>k</sub>):
            one mat-vec a layer (<span className="mono">backend/atlas_ops.py attention_drive</span>, <span className="mono">block_split</span>). The
            twelve strongest across every block are listed; each block’s shares add up to its part, to the rounding of the block’s own output.
          </dd>
          <dt>read on</dt>
          <dd>
            <span className="mono">POST /api/atlas/attn_drive</span> with the sighting’s own word-piece ids (
            <span className="mono">ids</span>, at most 512) and the last of them read (<span className="mono">at: -1</span>), so a word-piece of
            the answer is read where it was read
            {mine?.drive ? `; read at position ${mine.drive.read_pos}${s?.readPos != null ? ` (the sighting’s ${s.readPos})` : ""}` : ""}.
          </dd>
          <dt>by the weights</dt>
          <dd>
            <span className="mono">GET /api/atlas/neuron/wiring</span> <span className="mono">up</span>: a writer A at layer i writes w<sub>post,i</sub>{" "}
            ⊙ W<sub>down,i</sub>[:, A], read through this neuron’s gate and up rows; P = g · u, ranked by |P| f² across layers (the layer scalars
            and the norms at their usual scale between the two), shown on the face that opens the gate (sign g){" "}
            {mine?.wiring ? `; ${mine.wiring.source === "index" ? "from the wiring index" : "read from the weights just now"}` : ""}. Leads, not
            circuits.
          </dd>
          <dt>labels</dt>
          <dd>each neuron’s words pass the label gate, by its own layer and its index check, as every label on these pages does.</dd>
        </dl>
      </Apparatus>
    </div>
  );
};

const NUM_INPUT: React.CSSProperties = {
  fontFamily: "inherit",
  fontSize: "16px",
  color: MUTED,
  background: "transparent",
  border: 0,
  borderBottom: "1px solid #C9BEAC",
  width: "56px",
  minHeight: "44px",
  boxSizing: "border-box",
  padding: "4px 2px",
  ...NUM,
};

interface TraceAsk {
  hop: number;
  k: number;
  score: "coupling" | "direct";
  hopLimit: number;
}

const ThreadsApparatus: React.FC<{
  m: Model;
  hopN: number;
  kN: number;
  kMax: number;
  reply: Readers | null;
  ask: TraceAsk;
  onAsk: (a: Partial<Omit<TraceAsk, "hopLimit">>) => void;
}> = ({ m, hopN, kN, kMax, reply, ask, onAsk }) => {
  const last = m.layerNo + hopN;
  const range = hopN === 1 ? `${m.layerNo + 1}` : `${m.layerNo + 1}–${last}`;
  const idxRange = hopN === 1 ? `${m.idx + 1}` : `${m.idx + 1}–${last - 1}`;
  const pol = m.focusFace > 0 ? "1.0" : "-1.0";
  // a live sighting is traced at its exact ids up to the word-piece read, as the instrument sends it (store, traceReaders)
  const ids = m.sighting?.source === "live" && m.sighting.contextIds?.length ? m.sighting.contextIds : null;
  const idList = ids ? `[${ids.join(", ")}]` : null;
  const code =
    "# MCP (backend/mcp_scientist.py)\n" +
    `trace_downstream_readers(layer=${m.idx}, neuron_idx=${m.neuron},\n` +
    (idList ? `    token_ids=${idList}, polarity=${pol},\n` : `    prompt="<SIGHTING_PROMPT>", polarity=${pol},\n`) +
    `    max_layer_hop=${ask.hop}, top_k=${ask.k}, score="${ask.score}")\n\n` +
    "# HTTP\nPOST /api/scientist/trace_readers\n" +
    `{"layer": ${m.idx}, "neuron_idx": ${m.neuron}, ${idList ? `"token_ids": ${idList}` : `"prompt": "<SIGHTING_PROMPT>"`},\n` +
    ` "polarity": ${pol}, "max_layer_hop": ${ask.hop}, "top_k": ${ask.k}, "score": "${ask.score}"}`;
  const direct = (reply?.scoreKind ?? ask.score) === "direct";
  return (
    <Apparatus title="threads (reader filaments)" label="Apparatus for the threads">
      <dl className="dl num">
        <dt>formula</dt>
        <dd>
          C<sub>i→j</sub> = ⟨σ<sub>i</sub> · w<sub>down,i</sub> ⊙ w<sub>post</sub>, w<sub>gate,j</sub> ⊙ w<sub>pre</sub>⟩ · |a<sub>j</sub>|, over
          the 25 strongest poles by |a| at each of layers {hopN ? range : "(none)"} (index {hopN ? idxRange : "none"}), whatever the ranking
          chosen for the sighting, sorted by |{direct ? "D" : "C"}|; the
          first {kN} of {kMax} are listed. A weight-based coupling in the residual-stream view of Elhage et al. 2021 [14].
        </dd>
        <dt>direct path</dt>
        <dd>
          D<sub>i→j</sub> = J<sub>pre</sub>(mid<sub>ℓ′</sub>) · (Π<sub>ℓ&lt;k&lt;ℓ′</sub> s<sub>k</sub>) · s<sub>ℓ</sub> · J<sub>post</sub>(y
          <sub>ℓ</sub>) (a<sub>i</sub> w<sub>down,i</sub>) · w<sub>gate,j</sub>: what the source adds to the reader’s gate through the stream
          alone, from its real activation a<sub>i</sub>, the layer scalars and the exact Jacobians of both RMSNorms at this word-piece. Each reader
          also carries the change of its activation, gelu(g)u − gelu(g − D)(u − D<sub>up</sub>). The study’s formula
          (experiments/scripts/V7_tool_bugs_03_filaments.py), with the sign of what the source adds rather than of its ablation.
        </dd>
        <dt>caveat</dt>
        <dd>
          it ignores the source’s own activation, the layer scalars, both RMSNorms and W<sub>up</sub>: ρ(|C|, measured effect) = +0.41, where a
          direct-path estimate reaches 0.87 and the true gradient 0.93; precision@6 is 0.24 against 0.54 (REPORT §3.4, bug 4). A reader’s pole
          is set by its W<sub>up</sub>; ⟨σ<sub>i</sub> w<sub>down</sub>, w<sub>gate</sub>⟩ only opens or closes it (N1).
        </dd>
        <dt>score by</dt>
        <dd style={{ margin: 0, display: "flex", flexDirection: "column", gap: "6px" }}>
          <div role="group" aria-label="Thread score" style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
            <button type="button" className="btn-ghost plain" aria-pressed={ask.score === "coupling"} onClick={() => onAsk({ score: "coupling" })}>
              coupling C, as first built
            </button>
            <button
              type="button"
              className="btn-ghost plain"
              aria-pressed={ask.score === "direct"}
              onClick={() => onAsk({ score: "direct" })}
              aria-describedby="score-d-note"
            >
              direct-path estimate, the fix
            </button>
          </div>
          <span style={{ color: MUTED }}>
            The coupling C, as the instrument first scored it: a weak guide, ρ ≈ 0.41 with the measured effect.
          </span>
          <span id="score-d-note" style={{ color: MUTED }}>
            The study’s direct-path estimate D, using the real activation, layer scalars and exact RMSNorm Jacobians (ρ ≈ 0.87; formula in{" "}
            <span className="mono">experiments/scripts/V7_tool_bugs_03_filaments.py</span>). Chosen here, it applies to the next trace; the reply
            {reply ? ` above was scored by ${reply.scoreKind === "direct" ? "D" : "C"}` : " says which it used"}.
          </span>
        </dd>
        <dt>controls</dt>
        <dd style={{ margin: 0, display: "flex", flexWrap: "wrap", alignItems: "center", gap: "6px 24px" }}>
          <label style={{ display: "flex", alignItems: "center", gap: "8px", minHeight: "44px" }}>
            layers ahead{" "}
            <input
              type="number"
              min={1}
              max={Math.max(1, ask.hopLimit)}
              value={ask.hop}
              disabled={ask.hopLimit === 0}
              onChange={(e) => {
                const v = parseInt(e.target.value, 10);
                if (Number.isFinite(v)) onAsk({ hop: v });
              }}
              style={{ ...NUM_INPUT, color: "inherit" }}
            />{" "}
            <span className="mono" style={{ color: MUTED }}>
              max_layer_hop
            </span>
          </label>
          <label style={{ display: "flex", alignItems: "center", gap: "8px", minHeight: "44px" }}>
            readers listed{" "}
            <input
              type="number"
              min={1}
              max={150}
              value={ask.k}
              onChange={(e) => {
                const v = parseInt(e.target.value, 10);
                if (Number.isFinite(v)) onAsk({ k: v });
              }}
              style={{ ...NUM_INPUT, width: "64px", color: "inherit" }}
            />{" "}
            <span className="mono" style={{ color: MUTED }}>
              top_k
            </span>
          </label>
          <span style={{ color: MUTED }}>
            pole: from the sign of a{m.act !== null ? `, here ${m.act >= 0 ? "+1" : "−1"}` : `; not recorded here, so the face you opened, ${m.focusFace > 0 ? "+1" : "−1"}`}{" "}
            <span className="mono">polarity</span>
          </span>
          <span style={{ flex: "1 1 100%", color: MUTED, fontStyle: "italic" }}>
            Sent with the next trace: from one to {ask.hopLimit === 6 ? "six" : numberWord(Math.max(ask.hopLimit, 0))} layers ahead (the server
            stops at layer 60), and up to 25 readers a layer.
          </span>
        </dd>
        <dt>by hand</dt>
        <dd style={{ margin: 0, display: "flex", flexDirection: "column", gap: "6px", minWidth: 0 }}>
          <span>
            Each reader comes back with its layer, neuron, pole, signed activation, the raw dot product, C, the direct-path estimates D and the
            change of its activation, and its top four word-pieces through the lens.{" "}
            {m.isFixture ? (
              <>
                The sighting’s prompt is “Tell me about yourself.” wrapped by the lab’s chat template, then “Hi! I”; its literal text is not stored
                in the repository. Without it, the instrument sends the question and the begun answer as the page is set (through the chat template
                when “send it as a chat” is ticked, which is the lab script’s wrapping; as bare text otherwise), so a live trace
                of the recorded sighting reads the same text only with the chat template on.
              </>
            ) : ids ? (
              <>
                It reads this sighting’s exact word-pieces, {ids.length} ids up to the one read here,{" "}
                {m.sighting?.readAt != null ? JSON.stringify(m.sighting.readAt) : ""} at position {ids.length - 1}
                {m.sighting && m.sighting.tokenIndex >= 1 ? `, the ${nth(m.sighting.tokenIndex)} word-piece of its own answer` : ""}: no
                template is applied and nothing is tokenized again, and it reads at the last of them. The text sent beside them, which an
                older server reads instead (at its last word-piece):{" "}
                <span className="mono-wrap">{JSON.stringify(m.sighting?.textSent ?? "")}</span>.
                {m.sighting?.isNudged ? " The trace reads the model without the nudge, on this very text." : ""}
              </>
            ) : (
              <>
                The prompt sent is this sighting’s text: <span className="mono-wrap">{JSON.stringify(m.sighting?.textSent ?? "")}</span>
                {m.sighting?.isNudged ? ". The trace reads the model without the nudge, on this very text." : ""}
              </>
            )}
          </span>
          <pre style={PRE}>{code}</pre>
        </dd>
        {reply ? (
          <>
            <dt>the reply</dt>
            <dd style={{ margin: 0, display: "flex", flexDirection: "column", gap: "6px", minWidth: 0 }}>
              <span>
                {reply.idsGiven === true
                  ? `read at position ${reply.readPos ?? "?"}, the last of the ids given, without a nudge; `
                  : reply.idsGiven === false
                    ? `read at the last word-piece of the text sent${reply.readPos != null ? `, position ${reply.readPos}` : ""}, without a nudge; `
                    : "the server did not say where it read (an older server reads the text sent, at its last word-piece); "}
                source a on the prompt = {reply.srcAct !== null ? signedRaw(reply.srcAct) : "?"}, pole{" "}
                {reply.srcFace !== null ? (reply.srcFace > 0 ? "+" : "−") : "?"}.{" "}
                {reply.readers
                  .map(
                    (r) =>
                      `layer ${r.idx + 1}, neuron ${r.neuron} ${r.face > 0 ? "+" : "−"}: a ${r.act === null ? "?" : signedRaw(r.act)}, dot ${r.dot === null ? "?" : signedRaw(r.dot)}, C ${signedRaw(r.c)}${r.directGate !== null ? `, D ${signedRaw(r.directGate)}` : ""}${r.directAct !== null ? `, Δa ${signedRaw(r.directAct)}` : ""}`,
                  )
                  .join(" · ")}
              </span>
              <RawToggle label="the raw reply" data={reply.raw} />
            </dd>
          </>
        ) : null}
      </dl>
    </Apparatus>
  );
};

/* ----------------------------------------------------- iv · its neighbours */

interface NbRow {
  p: PoleReading;
  me: boolean;
  lens: React.ReactNode;
  lensNote: string;
  eye: React.ReactNode;
  eyeNote: string;
  verdict: string;
  vNote: string;
  w: number;
  /** the label gate, when it holds this row's words back (their glosses then stand only in the apparatus) */
  held: Gate | null;
}

const NB_GRID = "84px 80px 112px 150px minmax(0, 1fr) minmax(0, 1fr) 190px";

function glossNote(toks: Tok[]): string {
  const gs = Array.from(new Set(toks.slice(0, 3).map((t) => glossFor(t.token)).filter(Boolean)));
  return gs.join("; ");
}

const Neighbours: React.FC<{ m: Model }> = ({ m }) => {
  const poleRanking = useTheodoliteStore((s) => s.poleRanking);
  const setPoleRanking = useTheodoliteStore((s) => s.setPoleRanking);
  const poles = m.poles;
  // strength bars against the loudest by |a| of the fifteen listed, whichever ranking listed them
  const loud = m.loudAct ?? (m.layer ? maxAct(m.layer) : null) ?? 1;
  const shown = poles.filter((p) => p.rank < 5 || p.neuron === m.neuron);
  const rest = poles.filter((p) => !shown.includes(p));
  const labelled = poles.filter((p) => p.lens || p.eye).length;
  const rows: NbRow[] = shown.map((p) => {
    const me = p.neuron === m.neuron;
    const notes = m.recordedLayer ? NB_NOTES[p.neuron] : undefined;
    const w = f1((Math.abs(m.recordedLayer ? (ACTS[IDS.indexOf(p.neuron)] ?? p.act) : p.act) / loud) * 100);
    const lt = tokFromReading(p.lens);
    const et = tokFromReading(p.eye);
    // the label gate first: a held-back row keeps its neuron, face and strength, and its words go to the apparatus
    const gate = labelGate(layerIdx(m.idx), p.check);
    if ((lt || et) && gate.heldBack) {
      return {
        p,
        me,
        lens: <HeldBackWords gate={gate} words={lt?.slice(0, 3).map((t) => t.token) ?? null} eye={et?.slice(0, 3).map((t) => t.token) ?? null} compact />,
        lensNote: notes?.lensNote ?? (lt ? glossNote(lt) : ""),
        eye: "",
        eyeNote: notes?.eyeNote ?? (et ? glossNote(et) : ""),
        verdict: "held back",
        vNote: heldTag(gate).replace(/^held back · /, ""),
        w,
        held: gate,
      };
    }
    const ten = gate.sharedTop10 != null ? `; ${gate.sharedTop10} of their first ten` : "";
    if (notes) {
      // the canvas read its verdicts from the three shown; with the server's count, the verdict goes by the first ten
      const verdict = gate.sharedTop10 != null ? verdictOfTen(gate.sharedTop10) : notes.verdict;
      return { p, me, ...notes, verdict, vNote: notes.vNote + (ten ? `, among those shown${ten}` : ""), w, held: null };
    }
    const a = lt && et ? agreement(lt, et) : null;
    return {
      p,
      me,
      lens: lt ? lt.slice(0, 3).map((t) => wp(t.token)).join(" ") : <Missing />,
      lensNote: lt ? glossNote(lt) : "",
      eye: et ? et.slice(0, 3).map((t) => wp(t.token)).join(" ") : <Missing />,
      eyeNote: et ? glossNote(et) : "",
      verdict: gate.sharedTop10 != null ? verdictOfTen(gate.sharedTop10) : a ? a.verdict : "",
      vNote: a ? (a.k === 0 ? `no word-piece shared among those shown${ten}` : `${andList(a.shared.map(wp))} on both sides${ten}`) : "",
      w,
      held: null,
    };
  });
  const actText = (p: PoleReading) => {
    const k = IDS.indexOf(p.neuron);
    return signedRaw(m.recordedLayer && k >= 0 ? ACTS[k] : p.act);
  };

  return (
    <section aria-labelledby="kin-head" style={{ ...SECTION, gap: "16px" }}>
      <div className="row-between">
        <h2 className="h-sec" id="kin-head">
          iv · its neighbours in layer {m.layerNo}
        </h2>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 22px", fontSize: "15px", color: MUTED }}>
          <span style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <FaceMark filled />
            positive face
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <FaceMark filled={false} />
            negative face
          </span>
        </div>
      </div>
      <Rubric style={{ maxWidth: "760px" }}>
        {m.byWrite
          ? "The five that wrote most into the stream at this word, each read both ways. Doubt the ones whose two readings share nothing."
          : "The five loudest neurons at this word, each read both ways. Doubt the ones whose two readings share nothing."}{" "}
        {m.idx < 40
          ? "Below layer 41 their words are held back; the apparatus keeps them."
          : "A row whose two readings share none of their first ten is held back; the apparatus keeps its words."}
      </Rubric>
      {rows.length ? (
        <div style={{ overflowX: "auto", paddingBottom: "6px" }}>
          <div style={{ minWidth: "1060px", display: "flex", flexDirection: "column" }}>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: NB_GRID,
                gap: "0 18px",
                fontSize: "13px",
                fontVariantCaps: "all-small-caps",
                letterSpacing: "0.16em",
                color: MUTED,
                padding: "0 10px 6px",
                borderBottom: `1px solid ${HAIR}`,
              }}
            >
              <span>{m.byWrite ? "by write size" : "loudness"}</span>
              <span>neuron</span>
              <span>face</span>
              <span>strength</span>
              <span>through the lens</span>
              <span>to the naked eye</span>
              <span>the two readings</span>
            </div>
            {rows.map((n) => (
              <div
                key={n.p.neuron}
                style={{
                  display: "grid",
                  gridTemplateColumns: NB_GRID,
                  gap: "0 18px",
                  alignItems: "baseline",
                  padding: "12px 10px",
                  borderBottom: `1px solid ${HAIR_FAINT}`,
                  background: n.me ? "#EDE5D3" : "transparent",
                }}
              >
                <span style={{ fontSize: "16px", color: MUTED }}>{ordinal(n.p.rank)}</span>
                <span style={{ display: "flex", flexDirection: "column", fontSize: "17px", ...NUM }}>
                  {n.me ? (
                    <span>{n.p.neuron}</span>
                  ) : (
                    <a href={href.neuron(m.layerNo, n.p.neuron, n.p.face)} aria-label={`Open the folio of neuron ${n.p.neuron}`}>
                      {n.p.neuron}
                    </a>
                  )}
                  <span style={{ fontSize: "12px", fontVariantCaps: "all-small-caps", letterSpacing: "0.16em", color: RED }}>
                    {n.me ? "this folio" : ""}
                  </span>
                </span>
                <span style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "16px" }}>
                  <FaceMark filled={n.p.face > 0} />
                  {faceWord(n.p.face)}
                </span>
                <span style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  <span style={{ fontSize: "16px", ...NUM }}>{Math.abs(n.p.act).toFixed(2)}</span>
                  <span style={{ display: "block", height: "3px", background: HAIR_FAINT, position: "relative" }}>
                    <span style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${Math.min(100, n.w)}%`, background: GILT }} />
                  </span>
                </span>
                <span style={{ display: "flex", flexDirection: "column", minWidth: 0 }}>
                  <span style={{ fontSize: n.held ? "16px" : "20px", lineHeight: 1.3, overflowWrap: "anywhere" }}>{n.lens}</span>
                  {n.held ? (
                    <IfApparatus>
                      <span style={{ fontSize: "14px", fontStyle: "italic", color: MUTED }}>{[n.lensNote, n.eyeNote].filter(Boolean).join("; ")}</span>
                    </IfApparatus>
                  ) : (
                    <span style={{ fontSize: "14px", fontStyle: "italic", color: MUTED }}>{n.lensNote}</span>
                  )}
                </span>
                <span style={{ display: "flex", flexDirection: "column", minWidth: 0 }}>
                  <span style={{ fontSize: "20px", lineHeight: 1.3, overflowWrap: "anywhere" }}>{n.eye}</span>
                  {n.held ? null : <span style={{ fontSize: "14px", fontStyle: "italic", color: MUTED }}>{n.eyeNote}</span>}
                </span>
                <span style={{ display: "flex", flexDirection: "column" }}>
                  <span style={{ fontSize: "17px", fontStyle: "italic" }}>{n.verdict}</span>
                  <span style={{ fontSize: "14px", color: MUTED }}>{n.vNote}</span>
                </span>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <p style={{ margin: 0 }}>
          The neurons listed at this layer: <Missing />
        </p>
      )}
      {rest.length ? (
        <p style={{ margin: 0, fontSize: "16px", color: MUTED, maxWidth: "900px" }}>
          {cap(numberWord(rest.length))} more fired here,{" "}
          {labelled > 0
            ? `but the instrument read words for only the first ${numberWord(labelled)} ${m.byWrite ? "by write size" : "by loudness"}.`
            : "and the instrument read words for none of them at this layer."}{" "}
          The {labelled > 0 ? "unread" : "other"} {numberWord(rest.length)}: neurons{" "}
          <span style={NUM}>
            {rest.map((p, i) => (
              <React.Fragment key={p.neuron}>
                {i ? ", " : ""}
                <a href={href.neuron(m.layerNo, p.neuron, p.face)} style={{ color: "inherit" }}>
                  {p.neuron}
                </a>
              </React.Fragment>
            ))}
          </span>
          .
        </p>
      ) : null}
      <Rubric style={{ maxWidth: "760px" }}>
        Even all fifteen together carry only part of what the layer recalls from memory: between a tenth and two-fifths of it, in the study.{" "}
        {m.byWrite
          ? "Write size ranks a neuron by how much it changed the stream here; it does not say how much it changed the model’s mind."
          : "Loudness ranks a neuron among its neighbours; it does not say how much the neuron changed the model’s mind."}
      </Rubric>
      <Apparatus title={`the fifteen listed at layer ${m.layerNo}`} label="Apparatus for the neighbours">
        <dl className="dl num">
          <dt>ranking</dt>
          <dd>
            {m.byWrite ? (
              <>
                by |a| · ‖w<sub>down</sub> ⊙ w<sub>post</sub>‖ (write size, the interface’s default)
              </>
            ) : (
              <>
                by |a| = |<span className="mono">mlp_act</span>| (as first built, and the server’s default)
              </>
            )}{" "}
            at the word-piece {m.sighting?.readAt != null ? `“${m.sighting.readAt}”` : <Missing />}; σ = sign(a).
          </dd>
          <dt>signed a</dt>
          <dd>{poles.length ? poles.map((p) => `${p.neuron} ${actText(p)}`).join(" · ") : <Missing />}</dd>
          <dt>labels stored</dt>
          <dd>
            {m.isFixture ? (
              labelled > 0 ? (
                <>
                  for the first five listed only (<span className="mono">pole_labels_jpp</span> and <span className="mono">pole_labels_naive</span>,
                  keys {m.idx}:0 to {m.idx}:4); the other ten: <Missing />
                </>
              ) : (
                <>
                  none at this layer: the recording keeps labels only at every sixth layer (index 5, 11, … 59), for the first five listed. Here:{" "}
                  <Missing />
                </>
              )
            ) : (
              <>
                for {numberWord(labelled)} of the {numberWord(poles.length)} (the server’s <span className="mono">poles_labels</span>)
                {labelled < poles.length ? (
                  <>
                    ; the others: <Missing />
                  </>
                ) : null}
              </>
            )}
          </dd>
          <dt>caveat</dt>
          <dd>
            a ranking by |a| ignores how large each neuron’s write is; ranked either way, the fifteen explained only 10–41% of the
            feed-forward write at the layers E3 sampled from 21 to 59 (21, 31, 41, 51 and 59), and 70–74% at layers 3 and 11; E3
            calls this indicative (12 prompts, last word-piece; REPORT §3.4, bug 8).
          </dd>
          <dt>the fix</dt>
          <dd style={{ margin: 0, display: "flex", flexDirection: "column", gap: 4 }}>
            <label className="check">
              <input
                type="checkbox"
                checked={poleRanking === "write"}
                onChange={(e) => {
                  setPoleRanking(e.target.checked ? "write" : "activation");
                  retakeIfLive();
                }}
              />
              rank by write size, and show the fraction of the feed-forward write the listed poles explain
            </label>
            <span style={{ color: MUTED, paddingLeft: 28 }}>
              {m.sighting?.source === "live" ? (
                <>
                  This sighting ranks them {m.sighting.poleRanking === "write" ? "by write size, |a| · ‖w_down ⊙ w_post‖" : "by |a|"}; the fraction
                  they explain at layer {m.layerNo}:{" "}
                  {m.layer?.ffnExplained != null ? `${(m.layer.ffnExplained * 100).toFixed(1)}%` : <Missing />}{" "}
                  (⟨Σ c<sub>i</sub>, f⟩ / ⟨f, f⟩, E3).
                </>
              ) : (
                <>
                  the fraction itself: <Missing />
                </>
              )}
            </span>
          </dd>
        </dl>
      </Apparatus>
    </section>
  );
};

/* ------------------------------------------------- v · put a guess to the test */

const lines = (t: string) =>
  t
    .split("\n")
    .map((x) => x.trim())
    .filter((x) => x.length > 0);

const AnswerRow: React.FC<{ title: string; sub?: string; first?: boolean; last?: boolean; children: React.ReactNode }> = ({
  title,
  sub,
  first,
  last,
  children,
}) => (
  <div
    style={{
      display: "grid",
      gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1fr)",
      gap: "2px 16px",
      alignItems: "baseline",
      padding: "10px 0",
      borderTop: `1px solid ${first ? HAIR : HAIR_FAINT}`,
      borderBottom: last ? `1px solid ${HAIR}` : undefined,
    }}
  >
    <dt style={{ display: "flex", flexDirection: "column" }}>
      <span style={{ fontSize: "17px" }}>{title}</span>
      {sub ? <span style={{ fontSize: "14px", fontStyle: "italic", color: MUTED }}>{sub}</span> : null}
    </dt>
    <dd style={{ margin: 0, fontSize: "17px" }}>{children}</dd>
  </div>
);

const mean = (xs: (number | null)[]) => {
  const v = xs.filter((x): x is number => x !== null);
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
};

const TestGuess: React.FC<{ m: Model }> = ({ m }) => {
  const testHypothesis = useTheodoliteStore((s) => s.testHypothesis);
  const entry = useTheodoliteStore((s) => s.hypotheses[poleKey(m.idx, m.neuron)]) as Pending<unknown> | undefined;
  const res: Hypothesis | null = entry?.data ? parseHypothesis(entry.data) : null;
  const [pos, setPos] = useState("");
  const [neg, setNeg] = useState("");
  const [ran, setRan] = useState(false);
  const P = lines(pos);
  const N = lines(neg);
  const run = () => {
    setRan(true);
    if (P.length && N.length) void testHypothesis(m.idx, m.neuron, P, N);
  };
  let status: React.ReactNode = "";
  if (entry?.pending)
    status = `Sent to the instrument: ${count(P.length, "where it should fire")}, ${count(N.length, "where it should stay quiet")}. Its answer will fill the table marked “its answer”.`;
  else if (ran && (P.length === 0 || N.length === 0)) status = "Write at least one sentence on each side first.";
  else if (res) status = `Answered, on ${count(res.pos.length, "where it should fire")} and ${count(res.neg.length, "where it should stay quiet")}.`;
  const ph = <span style={PH}>[APPEARS AFTER THE TEST]</span>;
  const peakPos = res ? mean(res.pos.map((r) => r.peak)) : null;
  const peakNeg = res ? mean(res.neg.map((r) => r.peak)) : null;
  const workFace = res && res.meanPos !== null ? (res.meanPos >= 0 ? 1 : -1) : null;
  // the test's d, from sample variances; an older server's, from population variances
  const dShown = res ? (res.test ? res.test.d : res.d) : null;
  // how its p was counted: over every split, or from seeded shuffles
  const sel = res?.test;
  const pSel = sel ? pShown(sel) : null;
  const howP =
    sel?.pMethod === "exact" ? `, over all ${sel.splits ?? "?"} splits` : sel?.pMethod ? `, from ${sel.splits ?? "?"} shuffles (seed ${sel.seed ?? "?"})` : "";
  // what was (or will be) sent: the folio sends bare text, so each sentence is read at its own last word-piece
  const sentChat = entry?.sent?.chat === true;
  const sentSystem = typeof entry?.sent?.system === "string" ? (entry.sent.system as string) : null;
  const hypCode =
    "# MCP (backend/mcp_scientist.py)\n" +
    `test_motif_hypothesis(layer=${m.idx}, neuron_idx=${m.neuron},\n` +
    `    pos_prompts=${JSON.stringify(P)},\n` +
    `    neg_prompts=${JSON.stringify(N)},\n` +
    `    chat=${sentChat ? "True" : "False"}, system=${sentSystem === null ? "None" : JSON.stringify(sentSystem)})\n\n` +
    "# HTTP\nPOST /api/scientist/test_hypothesis\n" +
    JSON.stringify({ layer: m.idx, neuron_idx: m.neuron, pos_prompts: P, neg_prompts: N, chat: sentChat, system: sentSystem }, null, 2);
  const rowsAll = res ? [...res.pos.map((r) => ({ r, side: "+" })), ...res.neg.map((r) => ({ r, side: "−" }))] : [];
  const fmt = (v: number | null) => (v === null ? "—" : signedRaw(v));
  const sides = (a: number | null, b: number | null) => (
    <span style={NUM}>
      firing {a === null ? "—" : signed(a, 2)} · quiet {b === null ? "—" : signed(b, 2)}
    </span>
  );

  return (
    <section aria-labelledby="test-head" style={SECTION}>
      <div className="row-between">
        <h2 className="h-sec" id="test-head">
          v · put a guess to the test
        </h2>
        <span style={ASIDE}>what wakes it? your guess, its answer</span>
      </div>
      <p style={{ margin: 0, maxWidth: "760px" }}>
        Have a guess about what wakes this neuron? Write sentences where you expect it to fire, and sentences where you expect it to stay quiet.
        The instrument reads the neuron on every one, then tells you whether it told the two sides apart, and which of its faces did the work.
      </p>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "28px 56px", alignItems: "flex-start" }}>
        <div style={{ flex: "999 1 560px", minWidth: 0, display: "flex", flexDirection: "column", gap: "16px" }}>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "18px 28px" }}>
            {(
              [
                ["hyp-pos", "where it should fire", pos, setPos, P.length],
                ["hyp-neg", "where it should stay quiet", neg, setNeg, N.length],
              ] as const
            ).map(([id, label, val, set, n]) => (
              <div key={id} style={{ flex: "1 1 260px", minWidth: 0, display: "flex", flexDirection: "column", gap: "6px" }}>
                <div style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", justifyContent: "space-between", gap: "2px 12px" }}>
                  <label htmlFor={id} style={SC_LABEL}>
                    {label}
                  </label>
                  <span style={{ fontSize: "15px", fontStyle: "italic", color: MUTED }}>one sentence per line</span>
                </div>
                <textarea
                  id={id}
                  rows={6}
                  value={val}
                  onChange={(e) => set(e.target.value)}
                  aria-describedby={`${id}-count`}
                  style={{ ...TEXTAREA, minHeight: "168px" }}
                />
                <span id={`${id}-count`} style={{ fontSize: "15px", color: MUTED }}>
                  {count(n, label)}
                </span>
              </div>
            ))}
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "12px 20px", alignItems: "center" }}>
            <button className="btn-primary" type="button" onClick={run} disabled={entry?.pending}>
              {entry?.pending ? "testing…" : "put it to the test"}
            </button>
            <span style={{ fontSize: "16px", color: MUTED }}>
              reads neuron {m.neuron} in layer {m.layerNo} on every sentence
            </span>
          </div>
          <p role="status" style={{ margin: 0, fontSize: "16px", color: INK }}>
            {status}
          </p>
          <ReplyError m={m} error={entry?.error} what="test" />
        </div>
        <div aria-live="polite" style={{ flex: "1 1 340px", minWidth: 0, display: "flex", flexDirection: "column", gap: "8px" }}>
          <h3 className="h-sub">its answer</h3>
          <dl style={{ margin: 0, display: "flex", flexDirection: "column" }}>
            <AnswerRow title="did it tell them apart?" first>
              {res ? <i>{hypothesisWords(res)}</i> : ph}
            </AnswerRow>
            <AnswerRow
              title="how far apart"
              sub={res && !res.test ? "(Cohen’s d as an older server gives it, from population variances)" : "(Cohen’s d, from each side’s sample variance)"}
            >
              {res ? dShown !== null ? <span style={NUM}>d = {signed(dShown, 2)}</span> : "—" : ph}
            </AnswerRow>
            <AnswerRow
              title="could chance do it?"
              sub="(p: how often the sentences, split between the sides afresh, give a Welch’s t as large; under 0.05 counts)"
            >
              {res?.test ? (
                <span style={NUM}>
                  {pSel === null ? "no p" : `p = ${pSel}`}
                  {res.test.t !== null ? ` · t = ${signed(res.test.t, 2)}` : ""} · n = {nText(res.test)}
                </span>
              ) : res ? (
                "not asked: an older server sets nothing against chance"
              ) : (
                ph
              )}
            </AnswerRow>
            <AnswerRow
              title="strength at the last word"
              sub={sentChat ? "(signed, on the chat template’s last word-piece, the same for every sentence)" : "(signed, on each sentence’s last word-piece)"}
            >
              {res ? sides(res.meanPos, res.meanNeg) : ph}
            </AnswerRow>
            <AnswerRow title="strength at its loudest word" sub="(signed, the strongest word-piece in each sentence)">
              {res ? sides(peakPos, peakNeg) : ph}
            </AnswerRow>
            <AnswerRow title="which face did the work" sub="(the pole your guess belongs to)" last>
              {res && workFace !== null ? (
                <>
                  the {faceWord(workFace)} face
                  {res.signature?.length ? (
                    // the pole's words pass the label gate: by its pole_label_check, else by its layer alone
                    labelGate(layerIdx(m.idx), res.signatureCheck).heldBack ? (
                      <span style={{ color: MUTED }}>
                        {" "}
                        <HeldBackWords
                          gate={labelGate(layerIdx(m.idx), res.signatureCheck)}
                          words={res.signature.map((t) => t.token)}
                          compact
                        />
                      </span>
                    ) : (
                      <span style={{ color: MUTED }}>, which writes {res.signature.map((t) => wp(t.token)).join(" ")}</span>
                    )
                  ) : null}
                </>
              ) : (
                ph
              )}
            </AnswerRow>
          </dl>
          {res ? <p style={{ margin: 0, fontSize: "15px", color: MUTED }}>Strengths are means over each side’s sentences.</p> : null}
        </div>
      </div>
      <IfRubrics>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "10px 48px" }}>
          <p className="rubric" style={{ flex: "1 1 340px", minWidth: 0, maxWidth: "620px" }}>
            Make the quiet sentences close cousins of the firing ones: the same length and subject, missing only the thing you think wakes the
            neuron. If the two sides differ in every way, a gap between them proves little.
          </p>
          <p className="rubric" style={{ flex: "1 1 340px", minWidth: 0, maxWidth: "620px" }}>
            This test asks what wakes the neuron; the columns above show what it writes. They are different questions, and in the study the
            answers rarely matched: in 4 of 30 late-layer faces.
          </p>
        </div>
      </IfRubrics>
      <Apparatus title="the motif hypothesis test" label="Apparatus for the hypothesis test">
        <dl className="dl num">
          <dt>tool</dt>
          <dd>
            <span className="mono">test_motif_hypothesis</span> in <span className="mono">backend/mcp_scientist.py</span>, also served at{" "}
            <span className="mono">POST /api/scientist/test_hypothesis</span>. The original interface had no control for it; this folio restores
            it.
          </dd>
          <dt>per sentence</dt>
          <dd>
            a<sub>t</sub> = <span className="mono">mlp_act</span>[{m.idx}, t, {m.neuron}] over every word-piece t; it reports a<sub>last</sub>, and
            the peak a<sub>t*</sub> with t* = argmax<sub>t</sub> |a<sub>t</sub>|, with that word-piece.
          </dd>
          <dt>how sent</dt>
          <dd>
            {sentChat ? (
              <>
                as chats (<span className="mono">chat=True</span>): each sentence was the user’s turn, so a<sub>last</sub> is read on the
                model-turn header that ends the template, the same word-pieces for every sentence.
              </>
            ) : (
              <>
                as bare text (<span className="mono">chat=False</span>), one start mark and then the sentence, so a<sub>last</sub> is read on the
                sentence’s own last word-piece. As chats the last word-piece would be the template’s, the same for every sentence, and the
                test would not tell the sentences apart.
              </>
            )}
          </dd>
          <dt>effect size</dt>
          <dd>
            d = (ā<sub>+</sub> − ā<sub>−</sub>) / s<sub>p</sub>, Cohen’s d on the last-word-piece values only, with s<sub>p</sub>
            <sup>2</sup> = ((n<sub>+</sub> − 1) s<sub>+</sub>
            <sup>2</sup> + (n<sub>−</sub> − 1) s<sub>−</sub>
            <sup>2</sup>) / (n<sub>+</sub> + n<sub>−</sub> − 2) from sample variances; ā<sub>+</sub> is the mean over the n<sub>+</sub> firing
            sentences, ā<sub>−</sub> over the n<sub>−</sub> quiet ones. As first built the tool took population variances, d = (ā<sub>+</sub> −
            ā<sub>−</sub>) / √(½ (s<sub>+</sub>
            <sup>2</sup> + s<sub>−</sub>
            <sup>2</sup>)), and it still sends that d as <span className="mono">cohens_d</span>, for old callers.
          </dd>
          <dt>against chance</dt>
          <dd>
            Welch’s t = (ā<sub>+</sub> − ā<sub>−</sub>) / √(s<sub>+</sub>
            <sup>2</sup>/n<sub>+</sub> + s<sub>−</sub>
            <sup>2</sup>/n<sub>−</sub>). Its p is two-sided: of the ways to split all the sentences into n<sub>+</sub> and n<sub>−</sub>, the share
            whose |t| is as large as the one observed, counted over every split when there are at most 10,000, else over 10,000 random splits
            (seed 0) as (1 + as large) / (1 + 10,000), never 0.
            {res?.test ? (
              <>
                {" "}
                Returned here: {pSel === null ? "no p" : `p = ${pSel}`}
                {howP}
                , t = {res.test.t === null ? "—" : signedRaw(res.test.t)} on {res.test.df === null ? "—" : res.test.df.toFixed(1)} degrees of freedom, n ={" "}
                {nText(res.test)}.
              </>
            ) : res ? (
              " An older server sent none."
            ) : null}
          </dd>
          <dt>verdict</dt>
          <dd>
            a reading that is not a finite number: <span className="mono">CANNOT_TEST</span>; under 4 sentences a side:{" "}
            <span className="mono">TOO_FEW_TO_JUDGE</span>, whatever the numbers (at three a side no split reaches p &lt; 0.05); p ≥ 0.05: <span className="mono">NOT_SHOWN</span>; else <span className="mono">SELECTIVE_NORTH_POLE (+)</span> when ā<sub>+</sub>{" "}
            &gt; ā<sub>−</sub> and ā<sub>+</sub> &gt; 0, <span className="mono">SELECTIVE_SOUTH_POLE (-)</span> when ā<sub>+</sub> &lt; ā
            <sub>−</sub> and ā<sub>+</sub> &lt; 0, and <span className="mono">DIFFERS_NOT_AS_GUESSED</span> otherwise (the quiet sentences
            fire a face more). The verdict as first built is still sent, for old callers, and is not the answer above: d ≥ 0.8 and ā<sub>+</sub>{" "}
            &gt; 0: <span className="mono">CONFIRMED_NORTH_POLE (+)</span>; d ≤ −0.8 and ā<sub>+</sub> &lt; 0:{" "}
            <span className="mono">CONFIRMED_SOUTH_POLE (-)</span>; |d| ≥ 0.5: <span className="mono">WEAK_SELECTIVITY</span>; otherwise{" "}
            <span className="mono">REJECTED_OR_INCONCLUSIVE</span>, at any number of sentences (one a side gives d near 10<sup>4</sup>). It then
            attaches the top five lens word-pieces of the pole that matches the sign of ā<sub>+</sub>.
            {res ? (
              <>
                {" "}
                Returned here: {res.test ? <span className="mono">{res.test.verdict}</span> : null}
                {res.test ? "; as first built " : ""}
                <span className="mono">{res.verdict}</span>, d = {res.d === null ? "?" : signedRaw(res.d)}, ā<sub>+</sub> ={" "}
                {res.meanPos === null ? "?" : signedRaw(res.meanPos)}, ā<sub>−</sub> = {res.meanNeg === null ? "?" : signedRaw(res.meanNeg)}.
              </>
            ) : null}
          </dd>
          <dt>caveat</dt>
          <dd>
            the peak is reported but not tested, nor any word-piece but the last. A p is one neuron’s: testing several, correct their p’s together
            (the study used Benjamini–Hochberg, E6 §2.3). “Selective” marks a gap that chance seldom makes between these sentences, not a cause.
            The sentences go in as the instrument is set: each as the user’s turn of a chat when “send it as a chat” is ticked (the reading is then
            at the end of the model’s turn header), as bare text otherwise{res?.raw && (res.raw as { chat?: boolean }).chat !== undefined ? `; this test was sent ${(res.raw as { chat?: boolean }).chat ? "as chats" : "as bare text"}` : ""}. The
            study reads such tests against a matched control and checks them causally (REPORT §3.3, §3.5).
          </dd>
          <dt>results by sentence</dt>
          <dd style={{ margin: 0, minWidth: 0 }}>
            <div style={{ overflowX: "auto" }}>
              <div style={{ minWidth: "620px" }}>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "minmax(0, 2fr) 70px 90px 90px minmax(0, 1fr) 60px",
                    gap: "0 14px",
                    fontSize: "13px",
                    fontVariantCaps: "all-small-caps",
                    letterSpacing: "0.14em",
                    color: MUTED,
                    paddingBottom: "4px",
                    borderBottom: `1px solid ${HAIR}`,
                  }}
                >
                  <span>sentence</span>
                  <span>side</span>
                  <span style={{ textAlign: "right" }}>a at last</span>
                  <span style={{ textAlign: "right" }}>peak a</span>
                  <span>at word-piece</span>
                  <span>pole</span>
                </div>
                {rowsAll.length ? (
                  rowsAll.map(({ r, side }, i) => (
                    <div
                      key={i}
                      style={{
                        display: "grid",
                        gridTemplateColumns: "minmax(0, 2fr) 70px 90px 90px minmax(0, 1fr) 60px",
                        gap: "0 14px",
                        padding: "8px 0",
                        borderBottom: `1px solid ${HAIR_FAINT}`,
                        fontSize: "15px",
                      }}
                    >
                      <span style={{ overflowWrap: "anywhere" }}>{r.prompt}</span>
                      <span>{side}</span>
                      <span style={{ textAlign: "right" }}>{fmt(r.last)}</span>
                      <span style={{ textAlign: "right" }}>{fmt(r.peak)}</span>
                      <span title={r.peakToken !== null ? JSON.stringify(r.peakToken) : undefined}>{r.peakToken !== null ? wp(r.peakToken) : "—"}</span>
                      <span>{r.face > 0 ? "+" : "−"}</span>
                    </div>
                  ))
                ) : (
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "minmax(0, 2fr) 70px 90px 90px minmax(0, 1fr) 60px",
                      gap: "0 14px",
                      padding: "8px 0",
                      borderBottom: `1px solid ${HAIR_FAINT}`,
                      fontSize: "13px",
                      letterSpacing: "0.08em",
                      color: MUTED,
                    }}
                  >
                    <span>[APPEARS AFTER THE TEST]</span>
                    <span>[±]</span>
                    <span style={{ textAlign: "right" }}>[a]</span>
                    <span style={{ textAlign: "right" }}>[a]</span>
                    <span>[WORD-PIECE]</span>
                    <span>[±]</span>
                  </div>
                )}
              </div>
            </div>
            {res ? <RawToggle label="the raw reply" data={res.raw} /> : null}
          </dd>
          <dt>by hand</dt>
          <dd style={{ margin: 0, display: "flex", flexDirection: "column", gap: "6px", minWidth: 0 }}>
            <span>Written from the sentences above, as you type.</span>
            <pre style={PRE}>{hypCode}</pre>
          </dd>
        </dl>
      </Apparatus>
    </section>
  );
};

/* -------------------------------------------- vi · what else you can do */

const DoMore: React.FC<{ m: Model }> = ({ m }) => {
  const setSurveyMode = useTheodoliteStore((s) => s.setSurveyMode);
  const setSteering = useTheodoliteStore((s) => s.setSteering);
  const setLarqlQuery = useTheodoliteStore((s) => s.setLarqlQuery);
  const steering = useTheodoliteStore((s) => s.steering);
  // how the interface squares a neuron-locked frame (frame_squaring in pole_motif): Gram–Schmidt unless chosen
  const neuronSquaring = useTheodoliteStore((s) => s.neuronSquaring);
  const gs = neuronSquaring === "gram_schmidt";
  const pol = m.focusFace > 0 ? "1.0" : "-1.0";
  const sign = m.focusFace > 0 ? "+" : "-";
  // a held-back label's words are not offered as the vault's concept either: the statement then names "<word>"
  const heldWords = !!(m.focus.lens || m.focus.eye) && m.focusGate.heldBack;
  const firstWord = heldWords ? null : m.recorded ? "“am”" : m.focus.lens?.length ? wp(m.focus.lens[0].token) : null;
  const conceptTok = heldWords ? "" : m.recorded ? "jestem" : (m.agr?.shared[0] ?? m.focus.lens?.[0]?.token ?? "").trim().replace(/['\\]/g, "");
  const concept = conceptTok || "<word>";
  const vaultQuery = `SELECT * FROM POLES NEAREST '${concept}' WHERE layer = ${m.idx} LIMIT 10`;
  const surveyCode =
    "# over the /ws/theodolite socket\n" +
    '{"action": "sweep",\n' +
    ' "payload": {"prompt": "<SIGHTING_PROMPT>",\n' +
    '             "survey_mode": "pole_motif",\n' +
    `             "frame_squaring": "${neuronSquaring}",\n` +
    `             "locked_pole": [${m.idx}, ${m.neuron}, ${pol}]}}`;
  const steerCode =
    "# over the /ws/theodolite socket (as the nudge page sends it)\n" +
    '{"action": "sweep",\n' +
    ' "payload": {"prompt": "<SIGHTING_PROMPT>",\n' +
    `             "steering": {"enabled": true, "pole_layer": ${m.idx},\n` +
    `                          "pole_neuron": ${m.neuron}, "pole_polarity": ${pol},\n` +
    `                          "scale": ${steering.scale}}}}`;
  const vaultCode =
    `${vaultQuery}\n` +
    `DESCRIBE POLE LAYER ${m.idx} SLOT ${m.neuron} POLARITY ${sign}\n` +
    `UPDATE LAYER ${m.idx} SLOT ${m.neuron} SET object = '<concept>' STRENGTH 2.5 POLARITY ${sign}\n` +
    "ROLLBACK\n\n" +
    `# MCP\nquery_larql_vault(lql_query="DESCRIBE POLE LAYER ${m.idx} SLOT ${m.neuron} POLARITY ${sign}")`;
  const card: React.CSSProperties = {
    flex: "1 1 320px",
    minWidth: 0,
    display: "flex",
    flexDirection: "column",
    gap: "12px",
    paddingTop: "16px",
    borderTop: `1px solid ${HAIR_SOFT}`,
  };

  return (
    <section aria-labelledby="do-head" style={{ ...SECTION, gap: "22px" }}>
      <h2 className="h-sec" id="do-head">
        vi · what else you can do with it
      </h2>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "36px 48px" }}>
        <div style={card}>
          <h3 className="h-sub">survey from it</h3>
          <p style={{ margin: 0, fontSize: "18px" }}>
            Make this neuron the frame of a new survey: what wakes it, what it writes, and a later neuron that may read it become the three
            landmarks. <span style={{ fontSize: "16px", fontStyle: "italic", color: MUTED }}>(pole-locked frame)</span>
          </p>
          <a href={href.landmarks()} style={LINK_BOX} onClick={() => setSurveyMode("pole_motif", [m.idx, m.neuron, m.focusFace])}>
            survey from this neuron →
          </a>
          <Rubric>
            Treat its angles as rough. A frame built from a neuron has three axes with fixed roles (what wakes it, what it writes, who listens
            next), so the order fault of word landmarks, which swung bearings by 58 to 99 degrees in the study, does not arise; but no frame
            built from a neuron was tested.
          </Rubric>
        </div>
        <div style={card}>
          <h3 className="h-sub">nudge with it</h3>
          <p style={{ margin: 0, fontSize: "18px" }}>
            Turn this neuron up and see whether the model reaches for {firstWord ?? "its words"} more readily, or turn it down and see what it
            says instead.
          </p>
          <a
            href={href.nudge()}
            style={LINK_BOX}
            onClick={() => setSteering({ pole_layer: m.idx, pole_neuron: m.neuron, pole_polarity: m.focusFace })}
          >
            nudge with this neuron →
          </a>
          <Rubric>
            {m.idx === RECORDED.idx
              ? "Begin gently. At this layer the server’s own default nudge is more than seven times the size of the stream itself, and in the study nudges of that size broke the model’s replies whatever their direction. The nudge page sends a hundredth of the stream instead."
              : "Begin gently. In the study, nudges of the server’s own default size broke the model’s replies whatever their direction. The nudge page sends a hundredth of the stream instead."}
          </Rubric>
        </div>
        <div style={card}>
          <h3 className="h-sub">find its kin in the vault</h3>
          <p style={{ margin: 0, fontSize: "18px" }}>Ask the vault which other neurons in layer {m.layerNo} lean toward the same words.</p>
          <code
            style={{
              display: "block",
              fontFamily: "var(--mono)",
              fontSize: "14px",
              lineHeight: 1.6,
              color: INK,
              padding: "8px 0",
              borderTop: `1px solid ${HAIR_FAINT}`,
              borderBottom: `1px solid ${HAIR_FAINT}`,
              overflowWrap: "anywhere",
            }}
          >
            {vaultQuery}
          </code>
          <span style={{ fontSize: "15px", fontStyle: "italic", color: MUTED }}>
            The vault counts layers from 0, so layer {m.layerNo} is written {m.idx}.
            {conceptTok ? "" : " Put a word in place of <word> once its faces are read."}
          </span>
          <a href={href.vault()} style={LINK_BOX} onClick={() => (conceptTok ? setLarqlQuery(vaultQuery) : undefined)}>
            open the vault →
          </a>
          <Rubric>A nearest-neighbour list is a projection, not an experiment. Kin found this way still need a nudge to prove them.</Rubric>
        </div>
      </div>
      <Apparatus title="survey, nudge and vault" label="Apparatus for survey, nudge and vault">
        <dl className="dl num">
          <dt>survey frame</dt>
          <dd style={{ margin: 0, display: "flex", flexDirection: "column", gap: "6px", minWidth: 0 }}>
            <span>
              pole-locked: e<sub>1</sub>, e<sub>2</sub>, e<sub>3</sub> = {gs ? "Gram–Schmidt" : "Löwdin"}(w<sub>gate</sub>, (σ · w
              <sub>down</sub>) · J<sub>ℓ</sub>
              <sup>⊤</sup>, w<sub>gate,j*</sub>) in fp32: what wakes it, what it writes, and j*, the reader at the end of the sweep’s first thread
              from this neuron, or failing one, the first listed neuron in layer {Math.min(60, m.layerNo + 1)} (
              <span className="mono">backend/engine_runtime.py</span>).{" "}
              {gs ? (
                <>
                  Gram–Schmidt keeps axis 1 exactly w<sub>gate</sub>. A neuron frame keeps it by default: its three axes have fixed
                  roles, so their order is not a choice and the order fault of the word landmarks (REPORT §3.4, bug 3) does not
                  arise. A symmetric (Löwdin) orthonormalisation [33] is an option on the landmarks page.
                </>
              ) : (
                <>
                  Squared all at once (Löwdin [33]), as chosen on the landmarks page, so axis 1 is no longer exactly w
                  <sub>gate</sub>; a neuron frame is squared in order (Gram–Schmidt) by default, since its axes have fixed roles.
                </>
              )}{" "}
              This frame was not tested. The link above sets this lock for the next sighting.{" "}
              <a href={href.landmarks()}>landmarks</a>
            </span>
            <pre style={PRE}>{surveyCode}</pre>
          </dd>
          <dt>nudge</dt>
          <dd style={{ margin: 0, display: "flex", flexDirection: "column", gap: "6px", minWidth: 0 }}>
            <span>
              pushes along v = σ · w<sub>down</sub> ⊙ w<sub>post</sub> in the stream after layer {m.layerNo} (activation steering, Turner et al.
              2023) [21]. The server’s own default, kept for old callers, adds a fixed 650 · v / ‖v‖ at every word-piece: 2.4–7.4 times the
              median non-BOS ‖h‖ across the five default layers{m.idx === RECORDED.idx ? ", and 7.4 times at this one (median ‖h‖ = 88 at index 53)" : ""};
              a random direction at that size also broke the model (V7). The nudge page sends a share of ‖h‖ instead, the start mark spared (REPORT
              §3.4, bug 1; §3.5 items 1 and 4): now{" "}
              {steering.strength != null ? `${steering.strength} of ‖h‖` : `a fixed length of ${steering.scale}`}. ‖h‖ in this sighting:{" "}
              {m.layer?.streamStrength != null ? m.layer.streamStrength.toFixed(1) : <Missing />}. The link above sets only which pole to nudge
              along; the size stays the nudge page’s own. <a href={href.nudge()}>nudge</a>
            </span>
            <pre style={PRE}>{steerCode}</pre>
          </dd>
          <dt>vault</dt>
          <dd style={{ margin: 0, display: "flex", flexDirection: "column", gap: "6px", minWidth: 0 }}>
            <span>
              <span className="mono">POLES NEAREST</span> ranks every w<sub>down,i</sub> ⊙ w<sub>post</sub> in the layer by |cos| with (W
              <sub>U</sub>[t] ⊙ γ) · J<sub>ℓ</sub>, γ the final-norm weight, and the sign picks the pole: a projected contribution, not a
              counterfactual, as the LARQL README warns [10]. An edit (.patch) is a single-neuron column edit along a base-model J++ pullback, not
              MEMIT [11] (REPORT §3.3). The vault borrows LARQL’s names but never calls it beyond <span className="mono">--version</span>.{" "}
              <a href={href.vault()}>vault</a>
            </span>
            <pre style={PRE}>{vaultCode}</pre>
          </dd>
          <dt>the prompt</dt>
          <dd>
            <span className="mono">&lt;SIGHTING_PROMPT&gt;</span> stands for this sighting’s prompt
            {m.isFixture || !m.sighting ? (
              <>
                : “Tell me about yourself.” wrapped by the lab’s chat template, then “Hi! I”. Its literal text is not stored in the repository:{" "}
                <span className="placeholder">[RECORDED IN A LIVE SIGHTING]</span>
              </>
            ) : (
              <>
                , sent as <span className="mono-wrap">{JSON.stringify(m.sighting.textSent ?? "")}</span>
                {m.sighting.tokenIndex >= 1 && m.sighting.answerIds ? (
                  <>
                    . A sweep writes its answer afresh after it; to read this very word-piece of this answer instead, add{" "}
                    <span className="mono-wrap">
                      "answer_token_ids": [{m.sighting.answerIds.join(", ")}], "sight_tokens": [{m.sighting.tokenIndex}]
                    </span>
                  </>
                ) : null}
              </>
            )}
          </dd>
        </dl>
      </Apparatus>
    </section>
  );
};

/* ------------------------------------------------------ vii · field notes */

const JOBS: Record<Jobs, string> = { one: "one job", two: "two related jobs", unknown: "not yet understood" };
/** "one neuron", "three names", "24 faces" */
const many = (k: number, noun: string) => `${numberWord(k)} ${noun}${k === 1 ? "" : "s"}`;

/**
 * Once the server keeps the field book: this browser's notes from before, on neurons the server has nothing on, offered
 * once (each key is asked about once: carried, or kept here when the reader says so).
 */
const CarryOffer: React.FC<{ where: FolioWhere; reload: () => void }> = ({ where, reload }) => {
  const [mig, setMig] = useState<Migration | null>(null);
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState<string | null>(null);
  const server = where.kind === "server";
  useEffect(() => {
    if (!server) return;
    let live = true;
    const entries = folioEntries().filter(([k]) => !carriedKeys().has(k));
    if (!entries.length) return;
    void (async () => {
      try {
        const [book, notes] = await Promise.all([
          fetch("/api/atlas/fieldbook").then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status))))),
          fetch("/api/atlas/notes").then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status))))),
        ]);
        const got = folioMigration(entries, { names: notes.neurons ?? {}, faces: book.neurons ?? {} });
        if (live && got.posts.length) setMig(got);
      } catch {
        // the server's book or notes not read: nothing is offered
      }
    })();
    return () => {
      live = false;
    };
  }, [server]);
  if (done) return <p className="status-line" style={{ margin: 0, maxWidth: "760px" }}>{done}</p>;
  if (!mig) return null;
  const names = mig.posts.filter((x) => x.path === "/api/atlas/notes").length;
  const faces = mig.posts.length - names;
  const keep = async () => {
    setBusy(true);
    const r = await carryToServer(mig);
    markCarried(r.kept);
    setBusy(false);
    setMig(null);
    const leftHere = [...mig.left.map((x) => `${x.what} (${x.why})`), ...r.failed.map((x) => `${x.what} (${x.why})`)];
    setDone(
      `Carried to the server: the notes on ${many(r.kept.length, "neuron")}.` +
        (leftHere.length ? ` Left in this browser: ${leftHere.slice(0, 4).join("; ")}${leftHere.length > 4 ? "; and more" : ""}.` : ""),
    );
    reload();
  };
  return (
    <div
      role="region"
      aria-label="Notes kept in this browser"
      style={{ display: "flex", flexDirection: "column", gap: "8px", maxWidth: "760px", padding: "10px 0", borderTop: `1px solid ${HAIR_FAINT}`, borderBottom: `1px solid ${HAIR_FAINT}` }}
    >
      <p style={{ margin: 0 }}>
        This browser keeps notes from before the server kept them, on {many(mig.keys.length, "neuron")} ({many(names, "name")},{" "}
        {many(faces, "face")}) the server has nothing on. Keep them on the server, where the atlas and every page it serves read them?
      </p>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "8px 14px" }}>
        <button type="button" className="btn-ghost" disabled={busy} onClick={() => void keep()}>
          {busy ? "carrying…" : "keep them on the server"}
        </button>
        <button
          type="button"
          className="btn-ghost"
          disabled={busy}
          onClick={() => {
            markCarried(mig.keys);
            setMig(null);
          }}
        >
          leave them in this browser
        </button>
      </div>
      <span style={{ fontSize: "15px", color: MUTED }}>
        Nothing the server keeps is written over: a name or a face it already has stays as it is, and this browser’s copy stays here.
      </span>
    </div>
  );
};

const Notes: React.FC<{ m: Model; saved: FolioSaved; update: Upd; where: FolioWhere; reload: () => void }> = ({
  m,
  saved,
  update,
  where,
  reload,
}) => {
  const pas = usePassages(m.idx, m.neuron);
  const stored = pas && pas["+"].ready ? pas : null;
  const [face, setFace] = useState<"pos" | "neg">(m.focusFace > 0 ? "pos" : "neg");
  const nf = saved.notes[face];
  const setNote = (patch: Partial<FaceNotes>) =>
    update((p) => ({ ...p, notes: { ...p.notes, [face]: { ...p.notes[face], ...patch } } }));
  const record = {
    layer: m.layerNo,
    index: m.idx,
    neuron: m.neuron,
    kept: where.kind === "server" ? "on the server (field_book.json)" : where.kind === "browser" ? "in this browser" : null,
    pole: face === "pos" ? "+" : "-",
    jobs: nf.jobs === null ? null : JOBS[nf.jobs],
    pattern: nf.pattern,
    checked: { other_pole: nf.other, long_contexts: nf.long, steered: nf.steered, readers_traced: nf.readers },
    // the face's label as read, whole, with the server's check and whether the pages hold it back (lib/labels.ts)
    label: (() => {
      const isFocus = (face === "pos" ? 1 : -1) === m.focusFace;
      const r = isFocus ? m.focus : m.other;
      const g = isFocus ? m.focusGate : m.otherGate;
      if (!r || (!r.lens && !r.eye)) return null;
      return {
        lens: r.lens?.map((t) => t.token) ?? null,
        eye: r.eye?.map((t) => t.token) ?? null,
        label_check: labelCheckJson(g.check),
        held_back: g.heldBack,
        held_back_why: g.reasons,
      };
    })(),
  };
  const top = m.focus.lens?.slice(0, 3).map((t) => t.token) ?? null;
  const faceName = face === "pos" ? "the positive face" : "the negative face";
  const check = (k: "other" | "long" | "steered" | "readers", label: string, hint: string, lastRow?: boolean) => (
    <label
      style={{
        display: "flex",
        alignItems: "center",
        gap: "10px",
        minHeight: "44px",
        fontSize: "18px",
        cursor: "pointer",
        borderBottom: lastRow ? undefined : `1px solid ${HAIR_FAINT}`,
      }}
    >
      <input
        type="checkbox"
        checked={nf[k]}
        onChange={() => setNote({ [k]: !nf[k] } as Partial<FaceNotes>)}
        style={{ width: "18px", height: "18px", accentColor: RED, margin: 0, flex: "0 0 auto" }}
      />
      <span>
        {label} <span style={{ fontSize: "15px", fontStyle: "italic", color: MUTED }}>· {hint}</span>
      </span>
    </label>
  );
  const faceBtn = (f: "pos" | "neg", label: string) => (
    <button
      type="button"
      className="btn-ghost"
      onClick={() => setFace(f)}
      aria-pressed={face === f}
      style={{ fontSize: "16px", letterSpacing: "0.14em", display: "flex", alignItems: "center", gap: "8px" }}
    >
      <FaceMark filled={f === "pos"} />
      {label}
    </button>
  );

  return (
    <section aria-labelledby="notes-head" style={{ ...SECTION, padding: "40px 0 10px" }}>
      <div className="row-between">
        <h2 className="h-sec" id="notes-head">
          vii · field notes
        </h2>
        <span style={ASIDE}>a surveyor’s field book, one face at a time</span>
      </div>
      <p style={{ margin: 0, maxWidth: "760px" }}>
        Before you trust a name, write down what you saw. These notes follow the spirit of Nick Cammarata’s “Neural Networks Are Already Clean”
        (Goodfire 2026) <a href={href.lineage()}>[9]</a>: take one face at a time, say how many jobs it seems to do, and say so plainly when you
        cannot yet tell.
      </p>
      <p style={{ margin: 0, maxWidth: "760px", padding: "10px 0", borderTop: `1px solid ${HAIR_FAINT}`, borderBottom: `1px solid ${HAIR_FAINT}` }}>
        <span style={{ fontVariantCaps: "all-small-caps", letterSpacing: "0.16em", fontSize: "16px", color: MUTED }}>so far ·</span>{" "}
        {top && m.focusGate.heldBack ? (
          <>
            What its {faceWord(m.focusFace)} face writes is held back here
            {m.focusGate.reasons.includes("writes_nothing")
              ? ": an edit in the vault has silenced it, so it writes nothing"
              : m.focusGate.reasons.includes("below_layer_40")
                ? ", as is every label below layer 41"
                : ", as is every label whose two readings share none of their first ten"}
            .{" "}
          </>
        ) : (
          <>
            {m.focus.from === "recorded" ? (
              <>Through the lens and to the naked eye, its positive face reads as forms of “to be”</>
            ) : top ? (
              <>
                Through the lens, its {faceWord(m.focusFace)} face writes {andList(top.map(wp))}
              </>
            ) : null}
            {top ? (m.agr ? <>, and the two readings {m.focusGate.sharedTop10 != null ? verdictOfTen(m.focusGate.sharedTop10) : m.agr.verdict}. </> : <>. </>) : <>What its faces write has not been read yet. </>}
          </>
        )}
        {stored ? (
          <>
            What wakes either face is <i>not yet understood</i>: its passages are stored (each face’s strongest from{" "}
            {stored["+"].tokens?.toLocaleString()} word-pieces, in the apparatus of i), with the word-pieces each face reads and that follow, which
            no test has checked yet
          </>
        ) : (
          <>
            What wakes either face is <i>not yet understood</i>: no passages have been gathered for this neuron
          </>
        )}
        {m.other?.lens ? "." : `, and its ${faceWord(m.otherFace)} face has not been read.`}
      </p>
      <CarryOffer where={where} reload={reload} />
      <p style={{ margin: 0, fontSize: "16px", color: MUTED, maxWidth: "760px" }}>
        {where.kind === "server" ? (
          <>
            Kept on the server, in its field book: the atlas shows them in this neuron’s details, beside its name, for every page this server
            answers. <a href={href.atlas(m.layerNo, m.neuron)}>open it in the atlas</a>
          </>
        ) : where.kind === "browser" ? (
          <>Kept in this browser only, for you: {where.why}.</>
        ) : (
          <>Reading the server’s field book…</>
        )}
      </p>
      {where.kind === "server" && where.error ? (
        <p className="error-line" role="alert" style={{ margin: 0 }}>
          {where.error}
        </p>
      ) : null}
      {where.kind === "server" && where.note ? (
        <p style={{ margin: 0, maxWidth: "760px" }}>
          <span style={{ fontVariantCaps: "all-small-caps", letterSpacing: "0.16em", fontSize: "16px", color: MUTED }}>
            the atlas’s note on it · {where.note.by === "scientist" ? "the scientist’s, kept by you" : "yours"} ·
          </span>{" "}
          <span style={{ whiteSpace: "pre-wrap" }}>{where.note.text}</span>
        </p>
      ) : null}
      <div role="group" aria-label="Which face these notes are about" style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "6px 10px" }}>
        <span style={{ fontVariantCaps: "all-small-caps", letterSpacing: "0.16em", fontSize: "16px", color: MUTED, marginRight: "6px" }}>notes on</span>
        {faceBtn("pos", "the positive face")}
        {faceBtn("neg", "the negative face")}
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "24px 56px", alignItems: "flex-start" }}>
        <div style={{ flex: "999 1 560px", minWidth: 0, display: "flex", flexDirection: "column", gap: "14px" }}>
          <fieldset style={{ border: 0, margin: 0, padding: 0, display: "flex", flexWrap: "wrap", alignItems: "center", gap: "0 26px" }}>
            <legend style={{ padding: 0, marginBottom: "2px", ...SC_LABEL }}>how many jobs does {faceName} do?</legend>
            {(["one", "two", "unknown"] as Jobs[]).map((j) => (
              <label key={j} style={{ display: "flex", alignItems: "center", gap: "10px", minHeight: "44px", fontSize: "18px", cursor: "pointer" }}>
                <input
                  type="radio"
                  name={`jobs-${face}`}
                  checked={nf.jobs === j}
                  onChange={() => setNote({ jobs: j })}
                  style={{ width: "18px", height: "18px", accentColor: RED, margin: 0 }}
                />
                {JOBS[j]}
              </label>
            ))}
          </fieldset>
          {nf.jobs === "unknown" ? (
            <p style={{ margin: 0, fontSize: "17px", maxWidth: "640px" }}>
              Then say so, here and wherever you share it. “Not yet understood” is an honest entry, and a better one than a wrong name.
            </p>
          ) : null}
          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            <label htmlFor="pattern" style={SC_LABEL}>
              describe the pattern without a list of nouns
            </label>
            <textarea
              id="pattern"
              rows={4}
              value={nf.pattern}
              onChange={(e) => setNote({ pattern: e.target.value })}
              style={{ ...TEXTAREA, minHeight: "120px" }}
            />
          </div>
        </div>
        <fieldset style={{ flex: "1 1 340px", minWidth: 0, border: 0, margin: 0, padding: 0, display: "flex", flexDirection: "column", gap: 0 }}>
          <legend style={{ padding: 0, marginBottom: "2px", ...SC_LABEL }}>before you trust it</legend>
          {check("other", "checked the other face", "read both faces, in i")}
          {check("long", "read it in long contexts", "whole passages, not single words")}
          {check("steered", "steered it", "the test in ii, or a gentle nudge in vi")}
          {check("readers", "traced its readers", "who listens to it, in iii", true)}
        </fieldset>
      </div>
      <p style={{ margin: 0, fontSize: "16px", color: MUTED, maxWidth: "900px" }}>
        How tidy are neurons, usually? Goodfire’s preliminary audit of another model, Qwen3-32B, put about 90% of faces at one main job, about 7% at
        two related jobs, and 2–4% not yet understood <a href={href.lineage()}>[9]</a>. This study’s single-rater pass over 48 Gemma-4 faces found
        46% with one clear job. The two used methods too different to compare directly <span style={{ fontStyle: "italic" }}>(study report, §3.2)</span>.
      </p>
      <Rubric style={{ maxWidth: "760px" }}>
        A field book is for what you saw, not what you hoped. Describe what the passages share in a sentence; a list of words can hide whether you
        have understood it. Tick a box only when you have done the thing.
      </Rubric>
      <Apparatus title="field notes" label="Apparatus for the field notes">
        <dl className="dl num">
          <dt>practice</dt>
          <dd>
            after Cammarata, “Neural Networks Are Already Clean” (Goodfire 2026) [9], whose preliminary audit sorts poles into one main job, two
            related jobs, or not understood; this page asks the same of each face, one at a time.
          </dd>
          <dt>Goodfire</dt>
          <dd>preliminary, Qwen3-32B: about 90% of poles one main job, about 7% two related jobs, about 2–4% not understood [9].</dd>
          <dt>this study</dt>
          <dd>
            N1 cleanliness audit: 48 random poles, 6 at each of 8 layers, one rater not blind to layer or strength, a strict rule (one job = at least
            7 of the top 10 contexts share one describable feature). 46% [33, 60] one job, 12% [6, 25] two related jobs, 42% [29, 56] unclear;
            strong poles 62% [43, 79] one job, weak poles 29% [15, 49] (REPORT §3.2; N1_REPORT).
          </dd>
          <dt>not comparable</dt>
          <dd>
            a 147.5k-word-piece corpus of mostly technical READMEs and code, where many poles may never meet their trigger; Goodfire’s mix of
            strong and weak poles is not stated. “The methods differ too much for a direct comparison” (REPORT §3.2).
          </dd>
          <dt>kept where</dt>
          <dd>
            {where.kind === "server" ? (
              <>
                on the server: the name as the atlas’s name for neuron {m.neuron} at index {m.idx} (
                <span className="mono-wrap">{where.notesFile}</span>, by “reader”), these notes per face in its field book beside it (
                <span className="mono-wrap">{where.file}</span>; <span className="mono">GET/POST /api/atlas/fieldbook</span>), a file of its own
                that a server from before never reads or writes.
              </>
            ) : (
              <>
                in this browser only, for you alone (its local storage, key <span className="mono">theodolite.folio.v1.{m.layerNo}:{m.neuron}</span>
                ), with the name you gave it{where.kind === "browser" ? `: ${where.why}` : ""}
                {where.kind === "browser" && where.detail ? (
                  <>
                    {" "}
                    (<span className="mono-wrap">{where.detail}</span>)
                  </>
                ) : null}
                . A server that answers for this neuron keeps them in its field book instead.
              </>
            )}{" "}
            The original instrument kept no notes. Proposed: stored with each pole’s contexts in the vault{" "}
            <span className="placeholder">[NOT YET BUILT]</span>
          </dd>
          <dt>the record</dt>
          <dd style={{ margin: 0, minWidth: 0 }}>
            <pre style={PRE}>{JSON.stringify(record, null, 2)}</pre>
          </dd>
        </dl>
      </Apparatus>
    </section>
  );
};

import React, { useEffect, useMemo, useRef, useState } from "react";
import { Apparatus, IfRubrics, Missing, NotYet, Page, PendingCheck, Rubric } from "../components/kit";
import { href, type Route } from "../router";
import {
  type Conn,
  type LayerReading,
  type Pending,
  type PoleReading,
  type Sighting,
  isGlobalIdx,
  poleKey,
  useTheodoliteStore,
} from "../store/useTheodoliteStore";
import { andList, agreement, cap, count, glossFor, scoresText, shortToken, signedRaw, type Agreement } from "./NeuronFolio/util";
import { faceWord, numberWord, pct, signed, wp } from "../lib/words";
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
  type FaceReading,
  type Hypothesis,
  type Inspection,
  type Readers,
  type Tok,
  parseHypothesis,
  parseInspection,
  parseReaders,
  verdictWords,
} from "./NeuronFolio/replies";
import { type FaceNotes, type FolioSaved, type Jobs, useFolioSaved } from "./NeuronFolio/saved";

/**
 * A neuron's folio (canvas: Neuron.dc.html, "folio iii"). Works for any neuron in the route:
 *   per-sighting values come from sighting.layers[idx];
 *   both faces come from inspect_pole (called on arrival), the threads from trace_readers (on request),
 *   the guess from test_motif_hypothesis;
 *   the canvas's recorded neuron (layer 54, neuron 15491, positive face) shows the study's recorded extras
 *   when the recorded sighting is loaded;
 *   the name and the field notes are kept per viewer in localStorage.
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

interface Labels {
  lens: Tok[] | null;
  eye: Tok[] | null;
  from: "inspection" | "recorded" | "sighting" | null;
}

interface Model {
  idx: number;
  layerNo: number;
  neuron: number;
  conn: Conn;
  sighting: Sighting | null;
  layer: LayerReading | null;
  poles: PoleReading[];
  pole: PoleReading | null;
  loudest: PoleReading | null;
  fired: boolean;
  /** the layer's loudest were recorded, and this neuron is not among them */
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
    const readAt = sighting?.readAt != null ? wp(sighting.readAt) : "";
    const inReply = sighting?.answerBegins?.trim() ? `, in its reply “${sighting.answerBegins.trim()}”` : ", at the end of your question";
    return {
      idx,
      layerNo: route.layerNo,
      neuron,
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
  const [saved, update] = useFolioSaved(route.layerNo, route.neuron);

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
      <Hero m={m} saved={saved} update={update} />
      <NeuronApparatus m={m} saved={saved} />
      <Faces m={m} />
      <Says m={m} />
      <Listen m={m} />
      <Neighbours m={m} />
      <TestGuess m={m} />
      <DoMore m={m} />
      <Notes m={m} saved={saved} update={update} />
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

const Hero: React.FC<{ m: Model; saved: FolioSaved; update: Upd }> = ({ m, saved, update }) => {
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
  const words = m.focus.lens?.slice(0, 3).map((t) => t.token) ?? null;
  const btn: React.CSSProperties = { fontSize: "16px", padding: "0 18px" };

  let firedLine: React.ReactNode;
  if (!s) firedLine = <Missing />;
  else if (m.fired)
    firedLine = (
      <>
        As the model read the word {m.readAt}
        {m.inReply}, this neuron fired, the {loudPhrase(m.pole!.rank)} in its layer.
      </>
    );
  else if (m.poles.length)
    firedLine = (
      <>
        As the model read the word {m.readAt}
        {m.inReply}, this neuron was not among the fifteen loudest in its layer, so its strength there was not recorded.
      </>
    );
  else
    firedLine = (
      <>
        As the model read the word {m.readAt}
        {m.inReply}, the instrument recorded no loudest neurons at this layer: <Missing />
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
              {kept ? "a name you gave it, not one the model knows" : "not yet named; any name you give it is yours, not one the model knows"}
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
          {m.recorded ? (
            <>
              The words it writes are all one verb, <i>to be</i>, in several tongues: English “Am”, Polish “jestem”, Czech “jsem”, Spanish
              “Somos”, French “êtes”.
            </>
          ) : words ? (
            <>
              Through the lens, its {faceWord(m.focusFace)} face writes {andList(words.map(wp))}.
            </>
          ) : null}
        </p>
        <p style={{ margin: 0, maxWidth: "660px", fontStyle: "italic", color: MUTED }}>
          {m.recorded ? "One small cell of an unfamiliar mind, holding a single idea across languages." : "One small cell of an unfamiliar mind."}
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
              </>
            ) : (
              <Missing />
            )}
          </Particular>
          <Particular dt="it went on" ddStyle={{ fontStyle: "italic" }}>
            {s && s.continuation.trim() ? `“… ${s.continuation.trim()}”` : <Missing />}
          </Particular>
          <Particular dt="loudness">
            {m.pole ? (
              <>
                {m.pole.rank === 0 ? "the loudest" : loudPhrase(m.pole.rank)} of the <span style={NUM}>21,504</span> in its layer
              </>
            ) : m.poles.length ? (
              <>
                not among the fifteen loudest of the <span style={NUM}>21,504</span> in its layer
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
    layer: m.layerNo,
    index: m.idx,
    neuron: m.neuron,
    read_at: s?.readAt ?? null,
    position: s?.readPos ?? null,
    n_tokens: s?.nTokens ?? null,
    signed_activation: m.pole ? m.pole.act : null,
    polarity: m.pole ? (m.pole.face > 0 ? "+" : "-") : null,
    rank_by_abs: m.pole ? m.pole.rank + 1 : null,
    pole_label_jpp: m.pole?.lens ? m.pole.lens.map((t) => [t.token, t.logit]) : null,
    pole_label_naive: m.pole?.eye ? m.pole.eye.map((t) => [t.token, t.logit]) : null,
    layer_poles_top15: m.poles.map((p) => [p.neuron, p.act]),
    ffn_dominance: { corrected: m.layer?.ffnShare ?? null, tool_before_fix: m.layer?.ffnShareBeforeFix ?? null },
  };
}

const NeuronApparatus: React.FC<{ m: Model; saved: FolioSaved }> = ({ m, saved }) => {
  const [rawOpen, setRawOpen] = useState(false);
  const s = m.sighting;
  const L = m.layer;
  const raw = rawReadout(m);
  const exportFolio = () => {
    const st = useTheodoliteStore.getState();
    const key = poleKey(m.idx, m.neuron);
    downloadJson(`folio-layer${m.layerNo}-neuron${m.neuron}.json`, {
      readout: raw,
      inspect_pole: st.inspections[key]?.data ?? null,
      trace_readers: st.readers[key]?.data ?? null,
      test_hypothesis: st.hypotheses[key]?.data ?? null,
      your_notes: saved,
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
              a live sighting from the running server: {s.nTokens ?? <Missing />} word-pieces, read at the last, “{s.readAt ?? ""}” (position{" "}
              {s.readPos ?? "?"}, counting from 0). Text sent: <span className="mono-wrap">{JSON.stringify(s.textSent ?? "")}</span>
            </>
          )}
        </dd>
        <dt>strength</dt>
        <dd>
          {m.act !== null && m.pole ? (
            <>
              a = <span className="mono">mlp_act</span>[{m.idx}, {s?.readPos ?? "t"}, {m.neuron}] = {signedRaw(m.act)}, so |a| ={" "}
              {String(Math.abs(m.act))}, {ordinal(m.pole.rank)} of the layer’s fifteen strongest;{" "}
              {m.loudest && m.loudest.neuron !== m.neuron ? (
                <>
                  the loudest, neuron {m.loudest.neuron}, has a = {signedRaw(m.recordedLayer ? ACTS[0] : m.loudest.act)} (the bar stands at{" "}
                  {m.barPct}%).
                </>
              ) : (
                <>it is the loudest (the bar stands full).</>
              )}
            </>
          ) : (
            <>
              not among the layer’s fifteen strongest at this word-piece, so a was not recorded: <Missing />
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
  const leftWords = m.focus.lens?.slice(0, 3).map((t) => t.token) ?? null;
  const rightWords = m.other?.lens?.slice(0, 3).map((t) => t.token) ?? null;
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
        ? `The left end, the ${focusName} face, was not among the fifteen loudest at this word`
        : `The left end, the ${focusName} face; its strength at this word was not recorded`) +
    (leftWords ? ` and writes ${quoted(leftWords)}. ` : `; its words were not read. `) +
    `The hollow end, the ${otherName} face, ${m.fired ? "was silent" : "is the other way to fire"}` +
    (rightWords ? ` and writes ${quoted(rightWords)}.` : ` and its words were not read in this sighting.`);

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
                <g fontFamily={SERIF} fontStyle="italic" fontSize={leftSize} fill={INK} textAnchor="end">
                  {leftWords.map((w, i) => (
                    <text key={i} x={i === 1 ? 136 : 142} y={[146, 208, 270][i]}>
                      “{shortToken(w)}”
                    </text>
                  ))}
                  <text x="136" y="306" fontSize="13" fill={MUTED}>
                    through the lens
                  </text>
                </g>
              </>
            ) : (
              <>
                <path d="M200,200 L156,200" fill="none" stroke={MUTED} strokeOpacity="0.6" strokeWidth="0.8" strokeDasharray="2 3" />
                <g fontFamily={SERIF} fontSize="12" letterSpacing="1.2" fill={MUTED} textAnchor="end">
                  <text x="148" y="196">
                    [NOT READ IN
                  </text>
                  <text x="148" y="214">
                    THIS SIGHTING]
                  </text>
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
                    through the lens
                  </text>
                </g>
              </>
            ) : (
              <>
                <path d="M560,200 L604,200" fill="none" stroke={MUTED} strokeOpacity="0.6" strokeWidth="0.8" strokeDasharray="2 3" />
                <g fontFamily={SERIF} fontSize="12" letterSpacing="1.2" fill={MUTED}>
                  <text x="612" y="196">
                    [NOT READ IN
                  </text>
                  <text x="612" y="214">
                    THIS SIGHTING]
                  </text>
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
                  {m.notAmong ? "neither end was among the fifteen loudest at this word" : "its strength at this word was not recorded"}
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
                  aria-label={`Strength ${strength}, against ${loud} for the loudest neuron in this layer`}
                  style={{ height: "3px", background: HAIR_FAINT, position: "relative" }}
                >
                  <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${m.barPct}%`, background: GILT }} />
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", gap: "12px", fontSize: "14px", color: MUTED, ...NUM }}>
                  <span>0</span>
                  <span>the loudest in this layer, {loud}</span>
                </div>
              </>
            ) : null}
            <p style={{ margin: 0, fontSize: "17px" }}>
              {m.fired ? (
                <>Fired here, at the word {m.readAt}. </>
              ) : m.notAmong ? (
                <>Not among the fifteen loudest at the word {m.readAt}, so its strength here was not recorded. </>
              ) : (
                <>Its strength at this word was not recorded. </>
              )}
              {leftWords ? <>It writes toward {quoted(leftWords)}.</> : <>What it writes: <Missing what={awaitWhat(m, pending)} /></>}
            </p>
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
            {m.other?.lens ? (
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
  const ctxCol = (title: string) => (
    <div style={{ flex: "1 1 300px", minWidth: 0, display: "flex", flexDirection: "column", gap: "4px" }}>
      <span style={{ fontVariantCaps: "all-small-caps", letterSpacing: "0.16em", color: INK }}>{title}</span>
      <dl style={{ margin: 0, display: "grid", gridTemplateColumns: "minmax(0, 170px) minmax(0, 1fr)", gap: "4px 14px" }}>
        {["top passages", "word-pieces it reads", "word-pieces that follow"].map((k) => (
          <React.Fragment key={k}>
            <dt style={{ color: MUTED }}>{k}</dt>
            <dd style={{ margin: 0, ...PH }}>[CONTEXTS NOT YET COLLECTED]</dd>
          </React.Fragment>
        ))}
      </dl>
    </div>
  );
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
          separately from one streaming pass over a corpus (REPORT §3.3, §3.5 item 5).
        </dd>
      </dl>
      <div
        role="group"
        aria-label="Stored contexts per pole, not yet collected"
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "10px 40px",
          padding: "10px 0",
          borderTop: `1px solid ${HAIR_SOFT}`,
          borderBottom: `1px solid ${HAIR_SOFT}`,
        }}
      >
        {ctxCol("positive pole · σ = +1")}
        {ctxCol("negative pole · σ = −1")}
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
  const other = new Set((sharedWith ?? []).slice(0, 3).map((t) => t.token.trim().toLowerCase()));
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
        toks.slice(0, 3).map((t, i) => <WordRow key={i} t={t} shared={other.has(t.token.trim().toLowerCase())} />)
      ) : (
        <div style={{ padding: "14px 0", borderBottom: `1px solid ${HAIR_FAINT}` }}>
          <Missing what={missing} />
        </div>
      )}
    </div>
  );
};

/** A labelled neighbour at this layer, other than this neuron, whose two readings agree most. */
function comparisonPick(m: Model) {
  let best: { pole: PoleReading; agr: Agreement } | null = null;
  for (const p of m.poles) {
    if (p.neuron === m.neuron || !p.lens || !p.eye) continue;
    const a = agreement(tokFromReading(p.lens)!, tokFromReading(p.eye)!);
    if (a.k === 0) continue;
    if (!best || a.k > best.agr.k) best = { pole: p, agr: a };
  }
  return best;
}

function trustText(n: number): string {
  if (n >= 41 && n <= 58) {
    let s =
      `Layer ${n} lies inside the band where the study’s labels held up when single neurons were steered: between a third and two-thirds of a ` +
      `label’s top ten words were among the hundred words the neuron most pushed. Below layer 41 the study found no stable single-neuron effect ` +
      `to label; at layers 11 to 31, labels did no better than chance.`;
    if (n >= 51) s += ` From layer 51 on, the lens did no better than the naked eye, so at layer ${n} neither column outranks the other.`;
    else if (n <= 46) s += ` At layers 41 to 46 the lens did a little better than the naked eye.`;
    return s;
  }
  if (n < 41)
    return (
      `Layer ${n} lies below the band where the study’s labels held up when single neurons were steered, layers 41 to 58. Below layer 41 the ` +
      `study found no stable single-neuron effect to label; at layers 11 to 31, labels did no better than chance. Read both columns as a rumour, ` +
      `not a finding.`
    );
  return `Layer ${n} lies above the band the study tested, layers 41 to 58. At the last layers the lens and the naked eye see much the same thing.`;
}

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
  const x = rx(m.layerNo);
  const band = m.layerNo < 41 ? "below the band" : m.layerNo > 58 ? "above the band" : "inside the band";
  const sealLines = !agr ? ["not yet", "read"] : agr.verdict === "partly agree" ? ["partly", "agree"] : [agr.verdict];

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
        same face seen with the naked eye. A word that stands in both is marked ◆.
      </Rubric>
      <div style={{ columns: "2 340px", columnGap: "88px", columnRule: `1px solid ${HAIR}`, paddingTop: "6px" }}>
        <WordColumn title="through the lens" sub="(J++ lens)" toks={m.focus.lens} sharedWith={m.focus.eye} missing={missing} />
        <WordColumn title="to the naked eye" sub="(logit lens)" toks={m.focus.eye} sharedWith={m.focus.lens} missing={missing} />
      </div>
      <Rubric style={{ maxWidth: "760px" }}>
        These are the words this face pushes the model toward when it fires. They are not a forecast: in the study, the words that follow when a
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
            agr
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
              {agr ? `${agr.k} IN 3` : "[UNREAD]"}
            </text>
          </g>
        </svg>
        <div style={{ flex: "999 1 460px", minWidth: 0, display: "flex", flexDirection: "column", gap: "10px" }}>
          <h3 className="h-sub">{agr ? `the lens and the naked eye ${agr.verdict}` : "the lens and the naked eye, not yet compared"}</h3>
          <p style={{ margin: 0, fontSize: "18px", maxWidth: "700px" }}>
            {m.focus.from === "recorded" ? (
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
            Why it matters: in the study, agreement between the two readings was a free sign that a neuron’s label is real. Take one sound label
            and one unsound, and agreement ranked the sound one higher 84 times in 100 <span style={{ fontStyle: "italic" }}>(AUC 0.84, layers 41–58)</span>.
            The study compared ten words a side; the instrument shows three, so treat this seal as a hint, not a verdict.
          </p>
        </div>
        {m.recorded ? (
          <ComparisonBlock
            neuron={CMP.neuron}
            lens="“Am” “am” “Am”"
            eye="“am” “Am” “am”"
            text="Third loudest in this layer, fired on its negative face. Both readings give only “am”: they agree, and such a label earns more trust."
          />
        ) : cmp ? (
          <ComparisonBlock
            neuron={cmp.pole.neuron}
            lens={(cmp.pole.lens ?? []).slice(0, 3).map((t) => wp(t.token)).join(" ")}
            eye={(cmp.pole.eye ?? []).slice(0, 3).map((t) => wp(t.token)).join(" ")}
            text={
              `${cap(loudPhrase(cmp.pole.rank))} in this layer, fired on its ${faceWord(cmp.pole.face)} face. ` +
              (cmp.agr.verdict === "agree"
                ? "Both readings give the same words: they agree, and such a label earns more trust."
                : `${cap(numberWord(cmp.agr.k))} word-piece${cmp.agr.k === 1 ? "" : "s"} in three stand${cmp.agr.k === 1 ? "s" : ""} on both sides.`)
            }
          />
        ) : null}
      </div>
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
              aria-label={`A ruler of the sixty layers. Layers 41 to 58 are marked as the band where labels held up; below layer 41 the study found no stable effect to label. This neuron sits at layer ${m.layerNo}, ${band}.`}
              style={{ width: "100%", height: "auto", display: "block" }}
            >
              <g fontFamily={SERIF} fontSize="14" textAnchor="middle">
                <text x="312" y="28" fontStyle="italic" fill={MUTED}>
                  below layer 41: no stable effect to label
                </text>
                <text x="724.4" y="28" fill={GILT_TEXT} style={NUM}>
                  labels held up · layers 41–58
                </text>
              </g>
              <path d="M601.6,34 L601.6,46 M847.1,34 L847.1,46" fill="none" stroke={GILT} strokeWidth="1" />
              <line x1="601.6" y1="40" x2="847.1" y2="40" stroke={GILT} strokeWidth="3" />
              <line x1="24" y1="60" x2="876" y2="60" stroke={INK} strokeOpacity="0.35" strokeWidth="1" />
              <path d={ruler.local} fill="none" stroke={INK} strokeOpacity="0.35" strokeWidth="0.8" />
              <path d={ruler.global} fill="none" stroke={GILT} strokeWidth="1" />
              <path d={`M${x},51 L${f1(x + 9)},60 L${x},69 L${f1(x - 9)},60 Z`} fill={RED} />
              <g fontFamily={SERIF} fontSize="13" textAnchor="middle" fill={MUTED} style={NUM}>
                {[1, 41, 58, 60]
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
      <SaysApparatus m={m} cmp={cmp} />
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

const SaysApparatus: React.FC<{ m: Model; cmp: { pole: PoleReading; agr: Agreement } | null }> = ({ m, cmp }) => {
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
          al. 2026) [2]. It was fitted on the base bf16 model and is applied to the instruct W4A16 one (REPORT §3.4, bug 9); the proposed fix is
          to refit it on the instruct checkpoint (§3.5 item 6). The published J++ readout also drops non-semantic word-pieces; the instrument
          applies only the matrix (REPORT §3.1). The naked eye is the logit lens (nostalgebraist 2020) [8].
        </dd>
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
              {agr.k} of 3 shared{agr.k ? <>, {agr.sharedRaw.map((t) => `“${t}”`).join(", ")}</> : null}.
              {all ? (
                <>
                  {" "}
                  Among all eight the server returned, {all.k} of 8 shared.
                </>
              ) : null}{" "}
              The study’s signal is the top-10 overlap: AUC 0.84 for telling a sound label (causal P@10 ≥ 0.3) from an unsound one, within layers
              41–58. {m.focus.from === "inspection" ? "The columns show the first three." : "This sighting stored three a side."}
            </>
          ) : (
            <Missing what={awaitWhat(m, m.insp?.pending)} />
          )}
        </dd>
        <dt>validity</dt>
        <dd>
          P@10 in the causal top 100 is 0.35–0.69 at layers 41–58 (45–80% of the reliability ceiling; null ≤ 0.003) and about 0 at layers 11–31.
          J++ beats the naked eye by 0.04 to 0.12 only at layers 41–46, and they are equal at 51–58 (REPORT §3.2).
        </dd>
        <dt>caveat</dt>
        <dd>
          every score sits under the softcap of 30, and single neurons decode as peaked as noise, so confidence is not evidence (REPORT §3.4, bug
          8; §4). Prior-z-scored labels did worse (−0.07 P@10). An output label says what the pole pushes when amplified, not what it is used for
          in running text (N1, Q2).
        </dd>
        <dt>causal check</dt>
        <dd style={{ margin: 0, display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: "4px 14px" }}>
          <button className="btn-ghost" type="button" disabled aria-describedby="causal-note">
            run a causal spot-check
          </button>
          <span className="placeholder">[APPEARS AFTER THE CHECK]</span>
          <span id="causal-note" style={{ flex: "1 1 100%", color: MUTED }}>
            Adds ±k · max|a| to this neuron at this word-piece, with the baseline in the same batch, and lists the word-pieces whose
            log-probability moves most. Proposed by the study (N1, recommendation 5; REPORT §3.3, “causally spot-checked”); not in the original
            instrument.
          </span>
          <NotYet style={{ margin: 0, flex: "1 1 100%" }} />
        </dd>
      </dl>
    </Apparatus>
  );
};

/* ------------------------------------------------------ iii · who listens */

const Listen: React.FC<{ m: Model }> = ({ m }) => {
  const traceReaders = useTheodoliteStore((s) => s.traceReaders);
  const entry = useTheodoliteStore((s) => s.readers[poleKey(m.idx, m.neuron)]) as Pending<unknown> | undefined;
  const reply: Readers | null = entry?.data ? parseReaders(entry.data) : null;
  const pending = entry?.pending;
  const hopN = Math.max(0, Math.min(6, 60 - m.layerNo));
  const kN = 10;
  const kMax = 25 * hopN;
  const first = m.layerNo + 1;
  const last = m.layerNo + hopN;
  const hopWord = hopN === 1 ? `layer, ${first}` : `${numberWord(hopN)} layers, ${first} to ${last}`;
  const cols = Array.from({ length: hopN }, (_, i) => ({ idx: m.idx + 1 + i, no: first + i, x: 190 + i * 100 }));
  const mid = cols.length ? (cols[0].x + cols[cols.length - 1].x) / 2 : 440;
  const readers = reply?.readers ?? [];
  const maxC = readers.reduce((a, r) => Math.max(a, Math.abs(r.c)), 0) || 1;

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
              {numberWord(kN)} readers, the strongest thread first.
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
              disabled={hopN === 0 || pending}
              onClick={() => traceReaders(m.idx, m.neuron, m.focusFace)}
            >
              {pending ? "tracing…" : "trace its readers"}
            </button>
            <span style={{ fontSize: "16px", color: MUTED }}>
              uses the same question and the same word{m.readAt ? `, ${m.readAt}` : ""}
            </span>
          </div>
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
                          stroke={r.c >= 0 ? GILT_TEXT : INK}
                          strokeOpacity={r.c >= 0 ? 0.85 : 0.6}
                          strokeWidth={f1(0.8 + 2.4 * (Math.abs(r.c) / maxC))}
                          strokeDasharray={r.c >= 0 ? undefined : "3 3"}
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
              <span>solid gilt: a thread that opens the reader (C &gt; 0)</span>
              <span>dashed ink: a thread that closes it (C &lt; 0)</span>
              <span>thicker: a larger |C|</span>
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
                <span style={{ textAlign: "right" }}>thread score</span>
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
                        <span style={{ fontSize: "14px", fontStyle: "italic", color: MUTED, overflowWrap: "anywhere", paddingBottom: "6px" }}>
                          writes {r.words.map((t) => wp(t.token)).join(" ")}
                        </span>
                      ) : null}
                    </span>
                    <span>{r.idx + 1}</span>
                    <span style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "16px" }}>
                      <FaceMark filled={r.face > 0} />
                      {faceWord(r.face)}
                    </span>
                    <span style={{ textAlign: "right" }}>{signed(r.c, 2)}</span>
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
                Up to {numberWord(kN)} rows, the strongest thread first. Open any reader to read its own folio.
              </p>
            </div>
          ) : null}
        </div>
        <div style={{ flex: "1 1 340px", minWidth: 0, display: "flex", flexDirection: "column", gap: "14px" }}>
          <Rubric>
            A thread’s score is a weak guide. In the study it matched a reader’s measured effect with a correlation of only about 0.4, where a
            simple direct-path estimate reached 0.87. Treat each thread as a lead to test with a nudge, not a wire you have found.
          </Rubric>
          <p style={{ margin: 0, fontSize: "16px", color: MUTED }}>
            The score is a linear estimate from the model’s weights, not a measured circuit. It is a known fault of the instrument, still unfixed.
          </p>
        </div>
      </div>
      <ThreadsApparatus m={m} hopN={hopN} kN={kN} kMax={kMax} reply={reply} />
    </section>
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

const ThreadsApparatus: React.FC<{ m: Model; hopN: number; kN: number; kMax: number; reply: Readers | null }> = ({
  m,
  hopN,
  kN,
  kMax,
  reply,
}) => {
  const last = m.layerNo + hopN;
  const range = hopN === 1 ? `${m.layerNo + 1}` : `${m.layerNo + 1}–${last}`;
  const idxRange = hopN === 1 ? `${m.idx + 1}` : `${m.idx + 1}–${last - 1}`;
  const pol = m.focusFace > 0 ? "1.0" : "-1.0";
  const code =
    "# MCP (backend/mcp_scientist.py)\n" +
    `trace_downstream_readers(layer=${m.idx}, neuron_idx=${m.neuron},\n` +
    `    prompt="<SIGHTING_PROMPT>", polarity=${pol},\n` +
    `    max_layer_hop=6, top_k=${kN})\n\n` +
    "# HTTP\nPOST /api/scientist/trace_readers\n" +
    `{"layer": ${m.idx}, "neuron_idx": ${m.neuron}, "prompt": "<SIGHTING_PROMPT>",\n` +
    ` "polarity": ${pol}, "max_layer_hop": 6, "top_k": ${kN}}`;
  return (
    <Apparatus title="threads (reader filaments)" label="Apparatus for the threads">
      <dl className="dl num">
        <dt>formula</dt>
        <dd>
          C<sub>i→j</sub> = ⟨σ<sub>i</sub> · w<sub>down,i</sub> ⊙ w<sub>post</sub>, w<sub>gate,j</sub> ⊙ w<sub>pre</sub>⟩ · |a<sub>j</sub>|, over
          the 25 strongest poles at each of layers {hopN ? range : "(none)"} (index {hopN ? idxRange : "none"}), sorted by |C|; the first {kN} of{" "}
          {kMax} are listed. A weight-based coupling in the residual-stream view of Elhage et al. 2021 [14].
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
            <button type="button" className="btn-ghost plain" aria-pressed={true}>
              coupling C, as the instrument scores it
            </button>
            <button type="button" className="btn-ghost plain" aria-pressed={false} disabled aria-describedby="score-d-note">
              direct-path estimate, the fix
            </button>
          </div>
          <span style={{ color: MUTED }}>
            The coupling C the instrument computes today: a weak guide, ρ ≈ 0.41 with the measured effect.
          </span>
          <NotYet style={{ margin: 0 }}>
            <span id="score-d-note">
              The study’s direct-path estimate, using the real activation, layer scalars and exact RMSNorm Jacobians (ρ ≈ 0.87; formula in{" "}
              <span className="mono">experiments/scripts/V7_tool_bugs_03_filaments.py</span>).
            </span>
          </NotYet>
        </dd>
        <dt>controls</dt>
        <dd style={{ margin: 0, display: "flex", flexWrap: "wrap", alignItems: "center", gap: "6px 24px" }}>
          <label style={{ display: "flex", alignItems: "center", gap: "8px", minHeight: "44px", color: MUTED }}>
            layers ahead <input type="number" value={6} disabled readOnly style={NUM_INPUT} />{" "}
            <span className="mono" style={{ color: MUTED }}>
              max_layer_hop
            </span>
          </label>
          <label style={{ display: "flex", alignItems: "center", gap: "8px", minHeight: "44px", color: MUTED }}>
            readers listed <input type="number" value={kN} disabled readOnly style={{ ...NUM_INPUT, width: "64px" }} />{" "}
            <span className="mono" style={{ color: MUTED }}>
              top_k
            </span>
          </label>
          <span style={{ color: MUTED }}>
            pole: from the sign of a{m.act !== null ? `, here ${m.act >= 0 ? "+1" : "−1"}` : `; not recorded here, so the face you opened, ${m.focusFace > 0 ? "+1" : "−1"}`}{" "}
            <span className="mono">polarity</span>
          </span>
          <span style={{ flex: "1 1 100%", color: MUTED, fontStyle: "italic" }}>
            Fixed at the defaults this page sends, 6 and {kN}: the server takes other values over HTTP, but the instrument cannot pass them on yet.
            The server stops at layer 60, so {hopN === 6 ? "all six layers lie ahead here" : `only ${numberWord(hopN)} lie ahead here`}.
          </span>
        </dd>
        <dt>by hand</dt>
        <dd style={{ margin: 0, display: "flex", flexDirection: "column", gap: "6px", minWidth: 0 }}>
          <span>
            Each reader comes back with its layer, neuron, pole, signed activation, the raw dot product, C, and its top four word-pieces through
            the lens.{" "}
            {m.isFixture ? (
              <>
                The sighting’s prompt is “Tell me about yourself.” wrapped by the lab’s chat template, then “Hi! I”; its literal text is not stored
                in the repository. Without it, the instrument sends the question and the begun answer as bare text, one line each, so a live trace
                of the recorded sighting reads a slightly different prompt.
              </>
            ) : (
              <>
                The prompt sent is this sighting’s text: <span className="mono-wrap">{JSON.stringify(m.sighting?.textSent ?? "")}</span>
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
                source a on the prompt = {reply.srcAct !== null ? signedRaw(reply.srcAct) : "?"}, pole{" "}
                {reply.srcFace !== null ? (reply.srcFace > 0 ? "+" : "−") : "?"}.{" "}
                {reply.readers
                  .map(
                    (r) =>
                      `layer ${r.idx + 1}, neuron ${r.neuron} ${r.face > 0 ? "+" : "−"}: a ${r.act === null ? "?" : signedRaw(r.act)}, dot ${r.dot === null ? "?" : signedRaw(r.dot)}, C ${signedRaw(r.c)}`,
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
}

const NB_GRID = "84px 80px 112px 150px minmax(0, 1fr) minmax(0, 1fr) 190px";

function glossNote(toks: Tok[]): string {
  const gs = Array.from(new Set(toks.slice(0, 3).map((t) => glossFor(t.token)).filter(Boolean)));
  return gs.join("; ");
}

const Neighbours: React.FC<{ m: Model }> = ({ m }) => {
  const poles = m.poles;
  const loud = m.loudAct ?? (poles[0] ? Math.abs(poles[0].act) : 1);
  const shown = poles.filter((p) => p.rank < 5 || p.neuron === m.neuron);
  const rest = poles.filter((p) => !shown.includes(p));
  const labelled = poles.filter((p) => p.lens || p.eye).length;
  const rows: NbRow[] = shown.map((p) => {
    const me = p.neuron === m.neuron;
    const notes = m.recordedLayer ? NB_NOTES[p.neuron] : undefined;
    const w = f1((Math.abs(m.recordedLayer ? (ACTS[IDS.indexOf(p.neuron)] ?? p.act) : p.act) / loud) * 100);
    if (notes) return { p, me, ...notes, w };
    const lt = tokFromReading(p.lens);
    const et = tokFromReading(p.eye);
    const a = lt && et ? agreement(lt, et) : null;
    return {
      p,
      me,
      lens: lt ? lt.slice(0, 3).map((t) => wp(t.token)).join(" ") : <Missing />,
      lensNote: lt ? glossNote(lt) : "",
      eye: et ? et.slice(0, 3).map((t) => wp(t.token)).join(" ") : <Missing />,
      eyeNote: et ? glossNote(et) : "",
      verdict: a ? a.verdict : "",
      vNote: a ? (a.k === 0 ? "no word-piece shared" : `${andList(a.shared.map(wp))} on both sides`) : "",
      w,
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
        The five loudest neurons at this word, each read both ways. Doubt the ones whose two readings share nothing.
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
              <span>loudness</span>
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
                  <span style={{ fontSize: "20px", lineHeight: 1.3, overflowWrap: "anywhere" }}>{n.lens}</span>
                  <span style={{ fontSize: "14px", fontStyle: "italic", color: MUTED }}>{n.lensNote}</span>
                </span>
                <span style={{ display: "flex", flexDirection: "column", minWidth: 0 }}>
                  <span style={{ fontSize: "20px", lineHeight: 1.3, overflowWrap: "anywhere" }}>{n.eye}</span>
                  <span style={{ fontSize: "14px", fontStyle: "italic", color: MUTED }}>{n.eyeNote}</span>
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
          The loudest neurons at this layer: <Missing />
        </p>
      )}
      {rest.length ? (
        <p style={{ margin: 0, fontSize: "16px", color: MUTED, maxWidth: "900px" }}>
          {cap(numberWord(rest.length))} more fired here,{" "}
          {labelled > 0 ? `but the instrument read words for only the loudest ${numberWord(labelled)}.` : "and the instrument read words for none of them at this layer."}{" "}
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
        Even all fifteen together carry only part of what the layer recalls from memory: between a tenth and two-fifths of it, in the study.
        Loudness ranks a neuron among its neighbours; it does not say how much the neuron changed the model’s mind.
      </Rubric>
      <Apparatus title={`the fifteen strongest at layer ${m.layerNo}`} label="Apparatus for the neighbours">
        <dl className="dl num">
          <dt>ranking</dt>
          <dd>
            by |a| = |<span className="mono">mlp_act</span>| at the word-piece {m.sighting?.readAt != null ? `“${m.sighting.readAt}”` : <Missing />}, as
            the instrument does; σ = sign(a).
          </dd>
          <dt>signed a</dt>
          <dd>{poles.length ? poles.map((p) => `${p.neuron} ${actText(p)}`).join(" · ") : <Missing />}</dd>
          <dt>labels stored</dt>
          <dd>
            {m.isFixture ? (
              labelled > 0 ? (
                <>
                  for the five strongest only (<span className="mono">pole_labels_jpp</span> and <span className="mono">pole_labels_naive</span>,
                  keys {m.idx}:0 to {m.idx}:4); the other ten: <Missing />
                </>
              ) : (
                <>
                  none at this layer: the recording keeps labels only at every sixth layer (index 5, 11, … 59), for the five strongest. Here:{" "}
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
            a ranking by |a| ignores how large each neuron’s write is, and the top fifteen explain only 10–41% of the feed-forward write at layers
            21–59 (REPORT §3.4, bug 8).
          </dd>
          <dt>the fix</dt>
          <dd style={{ margin: 0 }}>
            <PendingCheck
              label="rank by write size, and show the fraction of the feed-forward write the listed poles explain"
              note={
                <>
                  the fraction itself: <Missing />
                </>
              }
            />
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
  const hypCode =
    "# MCP (backend/mcp_scientist.py)\n" +
    `test_motif_hypothesis(layer=${m.idx}, neuron_idx=${m.neuron},\n` +
    `    pos_prompts=${JSON.stringify(P)},\n` +
    `    neg_prompts=${JSON.stringify(N)})\n\n` +
    "# HTTP\nPOST /api/scientist/test_hypothesis\n" +
    JSON.stringify({ layer: m.idx, neuron_idx: m.neuron, pos_prompts: P, neg_prompts: N }, null, 2);
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
              {res ? <i>{verdictWords(res.verdict)}</i> : ph}
            </AnswerRow>
            <AnswerRow title="how far apart" sub="(Cohen’s d; from 0.8 up, the instrument calls the gap clear)">
              {res && res.d !== null ? <span style={NUM}>d = {signed(res.d, 2)}</span> : ph}
            </AnswerRow>
            <AnswerRow title="strength at the last word" sub="(signed, on each sentence’s last word-piece)">
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
                    <span style={{ color: MUTED }}>, which writes {res.signature.map((t) => wp(t.token)).join(" ")}</span>
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
          <dt>effect size</dt>
          <dd>
            d = (ā<sub>+</sub> − ā<sub>−</sub>) / √(½ (s<sub>+</sub>
            <sup>2</sup> + s<sub>−</sub>
            <sup>2</sup>)), Cohen’s d on the last-word-piece values only, with population variances s<sup>2</sup>; ā<sub>+</sub> is the mean
            over the firing sentences, ā<sub>−</sub> over the quiet ones.
          </dd>
          <dt>verdict</dt>
          <dd>
            d ≥ 0.8 and ā<sub>+</sub> &gt; 0: <span className="mono">CONFIRMED_NORTH_POLE (+)</span>; d ≤ −0.8 and ā<sub>+</sub> &lt; 0:{" "}
            <span className="mono">CONFIRMED_SOUTH_POLE (-)</span>; |d| ≥ 0.5: <span className="mono">WEAK_SELECTIVITY</span>; otherwise{" "}
            <span className="mono">REJECTED_OR_INCONCLUSIVE</span>. It then attaches the top five lens word-pieces of the pole that matches the
            sign of ā<sub>+</sub>.
            {res ? (
              <>
                {" "}
                Returned here: <span className="mono">{res.verdict}</span>, d = {res.d === null ? "?" : signedRaw(res.d)}, ā<sub>+</sub> ={" "}
                {res.meanPos === null ? "?" : signedRaw(res.meanPos)}, ā<sub>−</sub> = {res.meanNeg === null ? "?" : signedRaw(res.meanNeg)}.
              </>
            ) : null}
          </dd>
          <dt>caveat</dt>
          <dd>
            the peak is reported but not tested; there is no null, interval or permutation, and “confirmed” marks a threshold crossed, not a cause.
            The sentences go in as bare text, with no chat template (REPORT §3.4, bug 5). The study reads such tests against a matched control and
            checks them causally (REPORT §3.3, §3.5).
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
  const pol = m.focusFace > 0 ? "1.0" : "-1.0";
  const sign = m.focusFace > 0 ? "+" : "-";
  const firstWord = m.recorded ? "“am”" : m.focus.lens?.length ? wp(m.focus.lens[0].token) : null;
  const conceptTok = m.recorded ? "jestem" : (m.agr?.shared[0] ?? m.focus.lens?.[0]?.token ?? "").trim().replace(/['\\]/g, "");
  const concept = conceptTok || "<word>";
  const vaultQuery = `SELECT * FROM POLES NEAREST '${concept}' WHERE layer = ${m.idx} LIMIT 10`;
  const surveyCode =
    "# over the /ws/theodolite socket\n" +
    '{"action": "sweep",\n' +
    ' "payload": {"prompt": "<SIGHTING_PROMPT>",\n' +
    '             "survey_mode": "pole_motif",\n' +
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
            Treat its angles as rough. With three words as landmarks, the study found bearings swung by 58 to 99 degrees when their order changed;
            a frame built from a neuron is made the same way and was not tested.
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
              ? "Begin gently. At this layer the server’s default nudge is more than seven times the size of the stream itself, and in the study nudges of the default size broke the model’s replies whatever their direction. This is a known fault, still unfixed."
              : "Begin gently. In the study, nudges of the server’s default size broke the model’s replies whatever their direction. This is a known fault, still unfixed."}
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
              pole-locked: e<sub>1</sub>, e<sub>2</sub>, e<sub>3</sub> = Gram–Schmidt(w<sub>gate</sub>, (σ · w<sub>down</sub>) · J<sub>ℓ</sub>
              <sup>⊤</sup>, w<sub>gate,j*</sub>) in fp32: what wakes it, what it writes, and j*, the reader at the end of the sweep’s first thread
              from this neuron, or failing one, the loudest neuron in layer {Math.min(60, m.layerNo + 1)} (<span className="mono">backend/server.py</span>
              ). Gram–Schmidt is order-dependent, as with the word landmarks (REPORT §3.4, bug 3); a symmetric (Löwdin) orthonormalization [33] is
              the proposed fix. This frame was not tested. The link above sets this lock for the next sighting. <a href={href.landmarks()}>landmarks</a>
            </span>
            <pre style={PRE}>{surveyCode}</pre>
          </dd>
          <dt>nudge</dt>
          <dd style={{ margin: 0, display: "flex", flexDirection: "column", gap: "6px", minWidth: 0 }}>
            <span>
              adds 650 · v / ‖v‖, with v = σ · w<sub>down</sub> ⊙ w<sub>post</sub>, to the stream after layer {m.layerNo} at every word-piece
              (activation steering, Turner et al. 2023) [21]. The default 650 is 2.4–7.4 times the median non-BOS ‖h‖ across the five default
              layers{m.idx === RECORDED.idx ? ", and 7.4 times at this one (median ‖h‖ = 88 at index 53)" : ""}; a random direction at that size
              also broke the model (V7). The fix scales it relative to ‖h‖ (about 0.02 of it) and masks positions (REPORT §3.4, bug 1; §3.5 items 1
              and 4). ‖h‖ in this sighting: {m.layer?.streamStrength != null ? m.layer.streamStrength.toFixed(1) : <Missing />}. The link above
              sets only which pole to nudge along; the size stays the nudge page’s own, now {steering.scale}. <a href={href.nudge()}>nudge</a>
            </span>
            <pre style={PRE}>{steerCode}</pre>
          </dd>
          <dt>vault</dt>
          <dd style={{ margin: 0, display: "flex", flexDirection: "column", gap: "6px", minWidth: 0 }}>
            <span>
              <span className="mono">POLES NEAREST</span> ranks every w<sub>down,i</sub> ⊙ w<sub>post</sub> in the layer by |cos| with (W
              <sub>U</sub>[t] ⊙ γ) · J<sub>ℓ</sub>, γ the final-norm weight, and the sign picks the pole: a projected contribution, not a
              counterfactual, as the LARQL README warns [10]. An edit (.patch) is a single-neuron column nudge along a base-model J++ pullback, not
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

const Notes: React.FC<{ m: Model; saved: FolioSaved; update: Upd }> = ({ m, saved, update }) => {
  const [face, setFace] = useState<"pos" | "neg">(m.focusFace > 0 ? "pos" : "neg");
  const nf = saved.notes[face];
  const setNote = (patch: Partial<FaceNotes>) =>
    update((p) => ({ ...p, notes: { ...p.notes, [face]: { ...p.notes[face], ...patch } } }));
  const record = {
    layer: m.layerNo,
    index: m.idx,
    neuron: m.neuron,
    pole: face === "pos" ? "+" : "-",
    jobs: nf.jobs === null ? null : JOBS[nf.jobs],
    pattern: nf.pattern,
    checked: { other_pole: nf.other, long_contexts: nf.long, steered: nf.steered, readers_traced: nf.readers },
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
        {m.focus.from === "recorded" ? (
          <>Its positive face writes forms of “to be”</>
        ) : top ? (
          <>
            Its {faceWord(m.focusFace)} face writes {andList(top.map(wp))}
          </>
        ) : null}
        {top ? (m.agr ? <>, and the two readings {m.agr.verdict}. </> : <>. </>) : <>What its faces write has not been read yet. </>}
        What wakes either face is <i>not yet understood</i>: no passages have been gathered for this neuron
        {m.other?.lens ? "." : `, and its ${faceWord(m.otherFace)} face has not been read.`}
      </p>
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
          {check("steered", "steered it", "a gentle nudge, in vi")}
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
            in this browser only, for you alone (its local storage, key <span className="mono">theodolite.folio.v1.{m.layerNo}:{m.neuron}</span>),
            with the name you gave it; the original instrument kept no notes. Proposed: stored with each pole’s contexts in the vault{" "}
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

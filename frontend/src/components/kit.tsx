import React from "react";
import { useReading } from "../reading";
import { href, type Route } from "../router";
import {
  LENS_MODEL_NOT_SENT,
  LENS_NAME_NOT_SENT,
  lensOf,
  readsInsideAnswer,
  useTheodoliteStore,
  type LensInfo,
  type Sighting,
  type SweepProgress,
} from "../store/useTheodoliteStore";
import { wp } from "../lib/words";
import {
  isBaseFitted,
  isShippedLens,
  lensBand,
  lensLabel,
  lensPhrase,
  lensShort,
  useInstrumentLens,
  useReadingLens,
  type LensBand,
} from "../lib/lens";
// cautions.tsx draws its lines with this file's <Rubric>; both are used only at render time, so the cycle is harmless
import { AnswerPositionCaution, SightingFarCaution } from "../lib/cautions";

/* ------------------------------------------------------------------ marks */

export const BrandMark: React.FC<{ size?: number }> = ({ size = 30 }) => (
  <svg width={size} height={size} viewBox="0 0 30 30" aria-hidden="true">
    <circle cx="15" cy="15" r="13" fill="none" stroke="#A07A2E" strokeWidth="1.2" />
    <ellipse cx="15" cy="15" rx="13" ry="4.5" fill="none" stroke="#A07A2E" strokeWidth="1" />
    <line x1="15" y1="2" x2="15" y2="28" stroke="#2B2117" strokeWidth="1" />
    <circle cx="15" cy="15" r="2.4" fill="#8F2B1E" />
  </svg>
);

export const Lozenge: React.FC<{ fill?: string; stroke?: string; w?: number; h?: number }> = ({
  fill = "#A07A2E",
  stroke = "none",
  w = 9,
  h = 11,
}) => (
  <svg width={w} height={h} viewBox="0 0 9 11" aria-hidden="true">
    <path d="M4.5 0.8 L8.2 5.5 L4.5 10.2 L0.8 5.5 Z" fill={fill} stroke={stroke} strokeWidth="1" />
  </svg>
);

const Diamond: React.FC<{ fill: string }> = ({ fill }) => (
  <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
    <path d="M5 0 L10 5 L5 10 L0 5 Z" fill={fill} />
  </svg>
);

/* ------------------------------------------------------------ the frame */

const NAV: { key: string; label: string; to: string; match: Route["name"][] }[] = [
  { key: "instrument", label: "instrument", to: href.instrument(), match: ["instrument", "layer", "neuron"] },
  { key: "atlas", label: "atlas", to: href.atlas(), match: ["atlas"] },
  { key: "landmarks", label: "landmarks", to: href.landmarks(), match: ["landmarks"] },
  { key: "nudge", label: "nudge", to: href.nudge(), match: ["nudge"] },
  { key: "vault", label: "vault", to: href.vault(), match: ["vault"] },
  { key: "lexicon", label: "lexicon", to: href.lexicon(), match: ["lexicon"] },
  { key: "lineage", label: "lineage", to: href.lineage(), match: ["lineage"] },
];

export const ReadingToggles: React.FC = () => {
  const { rubrics, apparatus, toggleRubrics, toggleApparatus } = useReading();
  return (
    <div role="group" aria-label="How to read these pages" className="reading-group">
      <button
        type="button"
        className="reading-toggle for-rubrics"
        aria-pressed={rubrics}
        onClick={toggleRubrics}
        title="Red directions for newcomers"
      >
        <Lozenge fill={rubrics ? "#8F2B1E" : "none"} stroke="#8F2B1E" />
        rubrics
      </button>
      <button
        type="button"
        className="reading-toggle for-apparatus"
        aria-pressed={apparatus}
        onClick={toggleApparatus}
        title="Formulas, raw values, sources and every control, for scholars"
      >
        <Lozenge fill={apparatus ? "#A07A2E" : "none"} stroke="#A07A2E" />
        apparatus
      </button>
    </div>
  );
};

/** True while the viewport is at most `px` wide (a phone); false where media queries cannot be read. */
function useNarrow(px = 560): boolean {
  const query = `(max-width: ${px}px)`;
  const subscribe = React.useCallback(
    (on: () => void) => {
      try {
        const mq = window.matchMedia(query);
        mq.addEventListener("change", on);
        return () => mq.removeEventListener("change", on);
      } catch {
        return () => {};
      }
    },
    [query],
  );
  const read = () => {
    try {
      return window.matchMedia(query).matches;
    } catch {
      return false;
    }
  };
  return React.useSyncExternalStore(subscribe, read, () => false);
}

/**
 * Where the request in flight stands, for the head of every page: "writing 3 of up to 6…" while it writes (it may end its
 * turn sooner), "sighting 2 of 6…" while it sights, "taking a sighting…" before the server says (and for an older server,
 * which sends no progress).
 */
function inFlightWords(p: SweepProgress | null): string {
  if (p?.phase === "writing" && p.of > 0) return `writing ${p.done} of up to ${p.of}…`;
  if (p?.phase === "sighting" && p.of > 0) return `sighting ${Math.min(p.done + 1, p.of)} of ${p.of}…`;
  return "taking a sighting…";
}

export const ServerStatus: React.FC = () => {
  const conn = useTheodoliteStore((s) => s.conn);
  const health = useTheodoliteStore((s) => s.health);
  const sighting = useTheodoliteStore((s) => s.sighting);
  const isSweeping = useTheodoliteStore((s) => s.isSweeping);
  const progress = useTheodoliteStore((s) => s.incoming?.progress ?? null);
  const { lens } = useInstrumentLens();
  const narrow = useNarrow();
  const model = "Gemma-4-31B-it";
  // the head of every page names the lens every reading goes through, from the server's own record where it sent one,
  // and says "lens not identified" when that record leaves out its name or its model
  const through = narrow ? lensShort(lens) : lensLabel(lens);
  let state: string;
  let fill = "#A07A2E";
  if (conn === "open") state = isSweeping ? inFlightWords(progress) : "ready";
  else if (conn === "connecting" || conn === "idle") {
    state = "reaching the server…";
    fill = "#C9BEAC";
  } else {
    state = sighting?.source === "fixture" ? "server not answering · recorded sighting" : "server not answering";
    fill = "#8F2B1E";
  }
  return (
    <>
      <span style={{ display: "flex", alignItems: "center", gap: 8 }} role="status" aria-live="polite">
        <Diamond fill={fill} />
        {/* one run of text, so the lens's name is read with the rest (a separate titled span was read apart from it, leaving
            "Gemma-4-31B-it · · writing…"), and an empty part never doubles a separator */}
        <span title={`Every reading goes through ${lensPhrase(lens, true)}`}>{[model, through, state].filter((x) => x.trim() !== "").join(" · ")}</span>
      </span>
      {conn === "open" && health ? <span>memory {health.vram_allocated_gb.toFixed(1)} GB in use</span> : null}
    </>
  );
};

export const SiteHeader: React.FC<{ route: Route }> = ({ route }) => (
  <>
    <header className="site-header">
      <a href={href.incipit()} className="brand" aria-label="Logos Theodolite, the welcome page">
        <BrandMark />
        <span>Logos Theodolite</span>
      </a>
      <nav aria-label="Main" className="site-nav">
        {NAV.map((n) => (
          <a key={n.key} href={n.to} aria-current={n.match.includes(route.name) ? "page" : undefined}>
            {n.label}
          </a>
        ))}
      </nav>
      <div className="site-status">
        <ServerStatus />
        <ReadingToggles />
      </div>
    </header>
    <div className="rule" />
  </>
);

export const SiteFooter: React.FC<{ folio: string; children?: React.ReactNode }> = ({ folio, children }) => {
  const sighting = useTheodoliteStore((s) => s.sighting);
  return (
    <footer className="site-footer">
      <span>
        {children ?? (
          <>
            <ReadingThrough /> · 60 layers · 5,376 dimensions ·{" "}
            {sighting?.source === "live"
              ? "a live sighting"
              : sighting
                ? `${sighting.provenance}, read through the same lens`
                : "no sighting yet"}{" "}
            ·{" "}
            <a href={href.lineage()}>the lineage of these ideas</a>
          </>
        )}
      </span>
      <span className="folio">{folio}</span>
    </footer>
  );
};

/* ------------------------------------------------- reader / scholar pieces */

/** A red italic direction for newcomers; hidden when rubrics are off. */
export const Rubric: React.FC<{ children: React.ReactNode; style?: React.CSSProperties; as?: "p" | "div" | "span" }> = ({
  children,
  style,
  as = "p",
}) => {
  const on = useReading((s) => s.rubrics);
  if (!on) return null;
  const Tag = as;
  return (
    <Tag className="rubric" style={style}>
      {children}
    </Tag>
  );
};

/** Shown only when the rubrics are on (for non-paragraph guidance such as legends). */
export const IfRubrics: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const on = useReading((s) => s.rubrics);
  return on ? <>{children}</> : null;
};

/** Shown only when the apparatus is open. */
export const IfApparatus: React.FC<{ children: React.ReactNode; else?: React.ReactNode }> = ({ children, else: alt }) => {
  const on = useReading((s) => s.apparatus);
  return on ? <>{children}</> : <>{alt ?? null}</>;
};

/** A block for scholars, ruled off by a 3px double gilt line; hidden unless the apparatus is open. */
export const Apparatus: React.FC<{
  title: React.ReactNode;
  label?: string;
  sources?: string | false;
  children: React.ReactNode;
  style?: React.CSSProperties;
  as?: "div" | "section";
}> = ({ title, label, sources, children, style, as = "div" }) => {
  const on = useReading((s) => s.apparatus);
  if (!on) return null;
  const Tag = as;
  return (
    <Tag role="note" aria-label={label ?? (typeof title === "string" ? `Apparatus for ${title}` : "Apparatus")} className="apparatus" style={style}>
      <div className="apparatus-head">
        <span className="kicker">apparatus · {title}</span>
        {sources !== false ? <a href={sources ?? href.lineage("sources")}>sources</a> : null}
      </div>
      {children}
    </Tag>
  );
};

/** A bracketed placeholder for a value that was not measured. Never invent a number. */
export const Missing: React.FC<{ what?: string }> = ({ what = "MEASURED IN A LIVE SIGHTING" }) => (
  <span className="placeholder">[{what.toUpperCase()}]</span>
);

/** The canvas's note for controls that wait on a server change. */
export const NotYet: React.FC<{ children?: React.ReactNode; style?: React.CSSProperties }> = ({ children, style }) => (
  <p className="not-yet" role="note" style={style}>
    <span className="placeholder">[NOT YET IN THE SERVER]</span>
    {children}
  </p>
);

/** A control that waits on a backend change: shown, disabled, with the note. */
export const PendingCheck: React.FC<{ label: React.ReactNode; note: React.ReactNode; checked?: boolean }> = ({
  label,
  note,
  checked = false,
}) => (
  <div className="stack" style={{ gap: 0 }}>
    <label className="check disabled">
      <input type="checkbox" disabled checked={checked} readOnly aria-describedby={undefined} />
      {label}
    </label>
    <NotYet style={{ margin: "0 0 6px 28px" }}>{note}</NotYet>
  </div>
);

export const SectionHead: React.FC<{ id?: string; children: React.ReactNode; aside?: React.ReactNode }> = ({ id, children, aside }) => (
  <div className="row-between">
    <h2 id={id} className="h-sec">
      {children}
    </h2>
    {aside ? <span style={{ fontSize: 15, color: "var(--muted)" }}>{aside}</span> : null}
  </div>
);

/** A word-piece, quoted, with its raw form on hover. */
export const Wp: React.FC<{ token: string | null | undefined; style?: React.CSSProperties; className?: string }> = ({
  token,
  style,
  className,
}) => (
  <span title={token != null ? JSON.stringify(token) : undefined} style={style} className={className}>
    {wp(token)}
  </span>
);

/* ------------------------------------------------------------- the lens */

/** "Reading google/gemma-4-31b-it (4-bit) through the J++ lens, fitted on the base model google/gemma-4-31B" */
export const ReadingThrough: React.FC = () => {
  const sighting = useTheodoliteStore((s) => s.sighting);
  const { lens } = useReadingLens(sighting);
  return <>Reading google/gemma-4-31b-it (4-bit) through {lensPhrase(lens, true)}</>;
};

/** The bands a red line can speak for: one per layer (lib/lens.ts, lensBand; none at layer 60), or `all` at once. */
export type CautionBand = Exclude<LensBand, "own"> | "all";

/** Erratum E2: at layers 51-59, 6-7 of the 9 AI conditions read at or above the woman persona through J++ (R1). */
const sheLean = (where: "here" | "there") =>
  `the J++ lens leans toward “she” for most of the AIs the study tried, so a pronoun read ${where} does not tell you how ` +
  "the model sees itself";

/** What a reading through the shipped lens can bear, in plain words, by the study's bands of layers (erratum E49). */
const SHIPPED_BANDS: Record<CautionBand, string> = {
  untested:
    "Below layer 21 the study’s two checks of even how sightings rank against each other disagreed, and no lens it tried " +
    "was reliable here: treat what you read as unconfirmed.",
  order:
    "At layers 21 to 30 trust only how several versions of a question rank against one another, and only where what the model " +
    "goes on to say bears it out; not the words themselves.",
  compare: "Compare two sightings, one of them a matched control, rather than read a single word as the model’s own.",
  late:
    "Compare two sightings, one of them a matched control, rather than read a single word as the model’s own. At layers 51 to 59 " +
    `${sheLean("here")}.`,
  all:
    "Compare two sightings, one of them a matched control, rather than read a single word as the model’s own. Below layer 21 " +
    "even comparisons are unconfirmed; at 21 to 30 trust only how several versions of a question rank against one another; at 51 to 59 " +
    `${sheLean("there")}.`,
};

/** For any other lens: the same bands, reported as what the study measured with the shipped lens, not with this one. */
const OTHER_LEAD =
  "What these pages say a layer can bear was measured with the shipped J++ lens, fitted on the base model, not with this one:";
const OTHER_BANDS: Record<CautionBand, string> = {
  untested: "below layer 21 even how two sightings rank against each other was not reliably confirmed with it.",
  order:
    "at layers 21 to 30 only how several versions of a question rank against one another held up with it, not the words " +
    "themselves.",
  compare: "from layer 31 the study compared two sightings, one of them a matched control, and did not trust single words.",
  late: "the study compared sightings at layers 31 to 50; at 51 to 59 that lens leaned toward “she” for most of the AIs the study tried.",
  all:
    "below layer 21 not even comparisons were reliably confirmed with it; at 21 to 30 only how several versions of a question rank; from 31 the " +
    "study compared sightings against a matched control rather than trust single words; at 51 to 59 that lens leaned toward " +
    "“she” for most of the AIs the study tried.",
};

/** What the line says of a lens that is not the shipped one when its card carries no caution of its own. */
const OTHER_NO_CAUTION =
  "This lens has not been checked against this model’s behaviour; compare matched sightings and do not read its words as the " +
  "model’s own.";

const BASE_SENTENCE = "This lens was fitted on the base model, not on the chat model you are reading.";

/** The band's words for `lens`: the study's advice for the shipped lens, and for any other what it measured with that one. */
function bandWords(lens: LensInfo, band: CautionBand): string {
  return isShippedLens(lens) ? SHIPPED_BANDS[band] : `${OTHER_LEAD} ${OTHER_BANDS[band]}`;
}

/** The same words in two parts, for a page that lists the bands one under another: the lead a lens other than the shipped
 * one takes once (null for the shipped lens), and the band's own words. */
export function bandParts(lens: LensInfo, band: CautionBand): { lead: string | null; words: string } {
  if (isShippedLens(lens)) return { lead: null, words: SHIPPED_BANDS[band] };
  const w = OTHER_BANDS[band];
  return { lead: OTHER_LEAD, words: w.charAt(0).toUpperCase() + w.slice(1) };
}

/**
 * The red line beside every readout of words, for the layer `no` (counted 1-60): what the study found a reading through
 * the shipped J++ lens, fitted on the base model, can bear there. For any other lens (or a card that does not identify
 * its lens) it gives the server's own caution and reports the bands as measured with the shipped lens, not with this
 * one. Every lens gets the line, since even one fitted on the instruct model was poor on single words and unstable below
 * layer 31 (R1; erratum E39). Nothing at layer 60, where the lens is the identity and the words are the model's own
 * odds; without `no`, all the bands at once. It ends by pointing to how a control is taken, since keeping one is done in
 * the apparatus. For a sighting read inside the model's answer (readsInsideAnswer: a word-piece it wrote, k ≥ 1, or at
 * k = 0 the end of an answer begun for it inside a chat turn), the line that says what the study measured there follows
 * it (lib/cautions.tsx, AnswerPositionCaution). For every sighting read at or past the end of the passages its lens was
 * fitted on (k = 0 included), a last line says so, and from the model's attention window on, that most of its layers no
 * longer reach the start of the text (lib/cautions.tsx, SightingFarCaution); none when the fit length is not known. The
 * sighting is the one given, else the one shown.
 */
export const LensCaution: React.FC<{ sighting?: Sighting | null; no?: number; style?: React.CSSProperties }> = ({
  sighting,
  no,
  style,
}) => {
  const shown = useTheodoliteStore((s) => s.sighting);
  const { lens } = useReadingLens(sighting);
  const band = no != null ? lensBand(no) : "all";
  if (band === "own") return null;
  const shipped = isShippedLens(lens);
  const read = sighting !== undefined ? sighting : shown;
  return (
    <>
      <Rubric style={style}>
        {shipped ? (
          `${BASE_SENTENCE} `
        ) : (
          <>
            {isBaseFitted(lens) && !lens.caution ? `${BASE_SENTENCE} ` : null}
            {lens.caution ? `The server says of this lens: ${lens.caution} ` : `${OTHER_NO_CAUTION} `}
          </>
        )}
        {bandWords(lens, band)} <a href={href.lexicon(band === "late" ? "late" : "single")}>Read the caution.</a>
        {band !== "untested" ? (
          <>
            {" "}
            <a href={href.lexicon("matched-control")}>How to take a matched control.</a>
          </>
        ) : null}
      </Rubric>
      {readsInsideAnswer(read) ? <AnswerPositionCaution sighting={read} /> : null}
      <SightingFarCaution sighting={read} />
    </>
  );
};

const LENS_FROM = {
  sighting: "As the server reported it with this sighting.",
  server: "As the server reports it (GET /api/health).",
  recorded: "The recorded sighting keeps no record of its lens; the study took it through this same file (E6).",
  unnamed:
    "No lens record has come from the server (it is not answering, or it is an older one that sends none); this is the lens " +
    "the instrument loads.",
} as const;

/** What an incomplete card leaves out, for the record's opening words. */
function leftOut(lens: LensInfo): string {
  const noName = lens.name === LENS_NAME_NOT_SENT;
  const noModel = lens.fittedOn === LENS_MODEL_NOT_SENT;
  if (noName && noModel) return "the lens’s name and the model it was fitted on";
  return noName ? "the lens’s name" : "the model it was fitted on";
}

/**
 * The apparatus line naming the lens, from the server's own record where it sent one: a <dt>/<dd> pair for a <dl>.
 * `reply` is a scientist reply (inspect_pole, trace_readers, test_hypothesis) whose own `lens` comes first. A card that
 * does not identify its lens is printed as sent, with what it leaves out marked. With the rubrics off, the red line
 * beside each readout is hidden, so the bands follow here once, as a second pair.
 */
export const LensRecord: React.FC<{ sighting?: Sighting | null; reply?: unknown; term?: string }> = ({
  sighting,
  reply,
  term = "lens record",
}) => {
  const read = useReadingLens(sighting);
  const rubrics = useReading((s) => s.rubrics);
  const replied = reply && typeof reply === "object" ? lensOf((reply as { lens?: unknown }).lens) : null;
  const lens = replied ?? read.lens;
  return (
    <>
      <dt>{term}</dt>
      <dd>
        {lens.identified ? null : <>Lens not identified: the server’s card leaves out {leftOut(lens)}. As sent: </>}
        {lens.name === LENS_NAME_NOT_SENT ? <Missing what="no name sent" /> : lens.name}, fitted on{" "}
        {lens.fittedOn === LENS_MODEL_NOT_SENT ? <Missing what="not sent" /> : <span className="mono-wrap">{lens.fittedOn}</span>}
        {lens.appliedTo ? (
          <>
            {" "}
            and read here on <span className="mono-wrap">{lens.appliedTo}</span>
          </>
        ) : null}
        ; method: {lens.method || <Missing what="not sent" />}; fit data: {lens.fitData || <Missing what="not sent" />}; source{" "}
        {lens.source ? <span className="mono-wrap">{lens.source}</span> : <Missing what="not sent" />}
        {lens.file ? (
          <>
            , loaded from <span className="mono-wrap">{lens.file}</span>
          </>
        ) : null}
        . {replied ? "As the server reported it with this reply." : LENS_FROM[read.from]}
        {isShippedLens(lens)
          ? " Base model in bf16, read on the instruct model in 4-bit: two changes at once, which the study could not tell apart (R1)."
          : null}
        {lens.caution ? <> The server’s caution, as sent: “{lens.caution}”</> : null}
      </dd>
      {rubrics ? null : (
        <>
          <dt>what to trust, by layer</dt>
          <dd>
            {bandWords(lens, "all")} The words at layer 60 are the model’s own odds.{" "}
            <a href={href.lexicon("single")}>Read the caution</a>, and{" "}
            <a href={href.lexicon("matched-control")}>how to take a matched control</a>.
          </dd>
        </>
      )}
    </>
  );
};

/** A notice shown wherever a live server is needed but the page is reading the recorded sighting. */
export const RecordedNotice: React.FC<{ children?: React.ReactNode }> = ({ children }) => {
  const conn = useTheodoliteStore((s) => s.conn);
  if (conn === "open") return null;
  return (
    <p className="status-line" role="status">
      {children ?? "The server is not answering. What is shown comes from the study’s recorded sighting; live readings return when it does."}
    </p>
  );
};

export const SweepError: React.FC = () => {
  const err = useTheodoliteStore((s) => s.sweepError);
  return err ? (
    <p className="error-line" role="alert">
      {err}
    </p>
  ) : null;
};

/** Page wrapper: header, content, footer. */
export const Page: React.FC<{ route: Route; folio: string; children: React.ReactNode; footer?: React.ReactNode }> = ({
  route,
  folio,
  children,
  footer,
}) => (
  <div className="page">
    <SiteHeader route={route} />
    <main id="main">{children}</main>
    <SiteFooter folio={folio}>{footer}</SiteFooter>
  </div>
);

import React from "react";
import { useReading } from "../reading";
import { href, type Route } from "../router";
import { useTheodoliteStore } from "../store/useTheodoliteStore";
import { wp } from "../lib/words";

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

export const ServerStatus: React.FC = () => {
  const conn = useTheodoliteStore((s) => s.conn);
  const health = useTheodoliteStore((s) => s.health);
  const sighting = useTheodoliteStore((s) => s.sighting);
  const isSweeping = useTheodoliteStore((s) => s.isSweeping);
  const model = "Gemma-4-31B-it";
  let line: string;
  let fill = "#A07A2E";
  if (conn === "open") line = isSweeping ? `${model} · taking a sighting…` : `${model} · ready`;
  else if (conn === "connecting" || conn === "idle") {
    line = `${model} · reaching the server…`;
    fill = "#C9BEAC";
  } else {
    line = sighting?.source === "fixture" ? "server not answering · recorded sighting" : "server not answering";
    fill = "#8F2B1E";
  }
  return (
    <>
      <span style={{ display: "flex", alignItems: "center", gap: 8 }} role="status" aria-live="polite">
        <Diamond fill={fill} />
        {line}
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
            Reading google/gemma-4-31b-it (4-bit) through the J++ lens · 60 layers · 5,376 dimensions ·{" "}
            {sighting?.source === "live" ? "a live sighting" : sighting ? sighting.provenance : "no sighting yet"} ·{" "}
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

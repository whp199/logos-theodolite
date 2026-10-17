import React from "react";
import { Apparatus } from "../../components/kit";
import { href } from "../../router";

/*
 * The Lexicon's building blocks: an entry (headword, specialist's name, plain account, marginal figure),
 * its apparatus band, the drop-capped lede of each section, and links between entries.
 * Styles are the canvas's (Lexicon.dc.html); colours come from the kit's tokens.
 */

/**
 * A link to an entry or a caution on this page. It is a route (#/lexicon/<id>), so every page links the
 * same way and the app scrolls the entry into view; clicking the link already in the address bar fires
 * no hashchange, so scroll by hand then.
 */
export const LexLink: React.FC<{ to: string; children: React.ReactNode; style?: React.CSSProperties }> = ({
  to,
  children,
  style,
}) => {
  const target = href.lexicon(to);
  return (
    <a
      href={target}
      style={style}
      onClick={() => {
        if (window.location.hash === target) document.getElementById(to)?.scrollIntoView({ block: "start" });
      }}
    >
      {children}
    </a>
  );
};

/** A section of the lexicon: "ii · inside the model" and its successors. */
export const LexSection: React.FC<{ id: string; title: string; pad?: number; children: React.ReactNode }> = ({
  id,
  title,
  pad = 64,
  children,
}) => (
  <section aria-labelledby={id} style={{ display: "flex", flexDirection: "column", gap: 22, paddingTop: pad }}>
    <h2 className="h-sec" id={id}>
      {title}
    </h2>
    {children}
  </section>
);

/** A section's opening paragraph with its gilt drop capital. */
export const Lede: React.FC<{ cap: string; children: React.ReactNode }> = ({ cap, children }) => (
  <p style={{ margin: 0, fontSize: 22, lineHeight: 1.5, maxWidth: 860 }}>
    <span
      style={{
        float: "left",
        fontSize: 96,
        lineHeight: 0.8,
        fontWeight: 500,
        padding: "8px 14px 0 0",
        color: "var(--gilt-numeral)",
      }}
    >
      {cap}
    </span>
    {children}
  </p>
);

/** Two columns of entries, one on a phone. */
export const EntryGrid: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div
    style={{
      display: "grid",
      gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 520px), 1fr))",
      columnGap: 64,
      marginTop: 8,
    }}
  >
    {children}
  </div>
);

const ENTRY: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: "14px 28px",
  alignItems: "flex-start",
  padding: "24px 0 30px",
  borderTop: "1px solid var(--hair-soft)",
};
const ENTRY_BODY: React.CSSProperties = {
  flex: "999 1 300px",
  minWidth: 0,
  display: "flex",
  flexDirection: "column",
  gap: 10,
};
const TECH: React.CSSProperties = {
  margin: "2px 0 0",
  fontSize: 16,
  fontStyle: "italic",
  fontVariantCaps: "all-small-caps",
  letterSpacing: "0.14em",
  color: "var(--muted)",
};

/**
 * One entry. `id` is its stable anchor (#/lexicon/<id>); `aliases` are further anchors that land on the
 * same entry, for links written with another slug (e.g. "naked-eye" for the lens entry).
 */
export const Entry: React.FC<{
  id: string;
  aliases?: string[];
  head: React.ReactNode;
  tech: React.ReactNode;
  figure?: React.ReactNode;
  children: React.ReactNode;
}> = ({ id, aliases, head, tech, figure, children }) => (
  <article id={id} aria-labelledby={`${id}-h`} style={ENTRY}>
    <div style={ENTRY_BODY}>
      <div>
        {aliases?.map((a) => <span key={a} id={a} aria-hidden="true" />)}
        <h3 className="h-entry" id={`${id}-h`}>
          {head}
        </h3>
        <p style={TECH}>{tech}</p>
      </div>
      {children}
    </div>
    {figure}
  </article>
);

/** A symbol inside the small-capital technical line, kept out of small capitals. */
export const Sym: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <span style={{ fontVariantCaps: "normal", letterSpacing: 0 }}>{children}</span>
);

/** The entry's plain account. */
export const Body: React.FC<{ children: React.ReactNode }> = ({ children }) => <p style={{ margin: 0 }}>{children}</p>;

/** A quieter second paragraph: where to find it, or what the study found. */
export const Note: React.FC<{ children: React.ReactNode; style?: React.CSSProperties }> = ({ children, style }) => (
  <p style={{ margin: 0, fontSize: 16, color: "var(--muted)", ...style }}>{children}</p>
);

/** "open the vault →": a link onward from an entry. */
export const Onward: React.FC<{ to: string; children: React.ReactNode }> = ({ to, children }) => (
  <a href={to} style={{ alignSelf: "flex-start", fontSize: 17, padding: "10px 0" }}>
    {children}
  </a>
);

/** A query or command, set in IBM Plex Mono between hairlines. */
export const Query: React.FC<{ children: string }> = ({ children }) => (
  <pre
    style={{
      margin: "2px 0 0",
      padding: "10px 0",
      borderTop: "1px solid var(--hair-soft)",
      borderBottom: "1px solid var(--hair-soft)",
      fontFamily: "var(--mono)",
      fontSize: 15,
      lineHeight: 1.5,
      color: "var(--ink)",
      whiteSpace: "pre-wrap",
      overflowWrap: "anywhere",
    }}
  >
    {children}
  </pre>
);

/** The entry's band for scholars: formula, raw values, sources. Hidden unless the apparatus is open. */
export const EntryApparatus: React.FC<{ title: string; label?: string; children: React.ReactNode }> = ({
  title,
  label,
  children,
}) => (
  <Apparatus
    title={title}
    label={label ?? `Apparatus for ${title}`}
    sources={false}
    style={{ marginTop: 4, gap: 2, fontVariantNumeric: "lining-nums tabular-nums" }}
  >
    <p style={{ margin: "2px 0 0" }}>{children}</p>
  </Apparatus>
);

/** Inline code inside the apparatus. */
export const M: React.FC<{ children: React.ReactNode }> = ({ children }) => <span className="mono">{children}</span>;

import React, { useEffect, useState } from "react";
import { IfApparatus } from "../../components/kit";
import { useReading } from "../../reading";
import { href } from "../../router";
import { useTheodoliteStore } from "../../store/useTheodoliteStore";

/* ------------------------------------------------------------ destinations */

/** The other folios this page points to (the canvas's *.dc.html links). */
export type Dest =
  | "instrument"
  | "instrument-apparatus"
  | "layer"
  | "neuron"
  | "landmarks"
  | "nudge"
  | "vault"
  | "lexicon"
  | "lexicon-lens"
  | "atlas";

/** The neuron the canvas's neuron folio shows: neuron 15491 at layer 54 (index 53), one of the recorded poles. */
const FOLIO_NEURON = { layerNo: 54, neuron: 15491 } as const;

function useDestHref(to: Dest): string {
  // a layer's leaf opens on the layer last chosen on the instrument (layer 36 until one is chosen)
  const selectedLayer = useTheodoliteStore((s) => s.selectedLayer);
  switch (to) {
    case "instrument":
    case "instrument-apparatus":
      return href.instrument();
    case "layer":
      return href.layer(selectedLayer + 1);
    case "neuron":
      return href.neuron(FOLIO_NEURON.layerNo, FOLIO_NEURON.neuron);
    case "landmarks":
      return href.landmarks();
    case "nudge":
      return href.nudge();
    case "vault":
      return href.vault();
    case "lexicon":
      return href.lexicon();
    case "lexicon-lens":
      return href.lexicon("lens");
    case "atlas":
      return href.atlas();
  }
}

/** A plain link to another folio. "instrument-apparatus" also opens the apparatus, as its label promises. */
export const To: React.FC<{ dest: Dest; children: React.ReactNode; style?: React.CSSProperties }> = ({
  dest,
  children,
  style,
}) => {
  const to = useDestHref(dest);
  const apparatus = useReading((s) => s.apparatus);
  const toggleApparatus = useReading((s) => s.toggleApparatus);
  const onClick = dest === "instrument-apparatus" && !apparatus ? () => toggleApparatus() : undefined;
  return (
    <a href={to} style={style} onClick={onClick}>
      {children}
    </a>
  );
};

const GO_STYLE: React.CSSProperties = { display: "inline-flex", alignItems: "center", minHeight: 44 };

/** The canvas's "… →" links: 44px tall targets. */
export const Go: React.FC<{ dest: Dest; children: React.ReactNode; size?: number }> = ({ dest, children, size = 17 }) => (
  <To dest={dest} style={{ ...GO_STYLE, fontSize: size }}>
    {children}
  </To>
);

/** A link to a source elsewhere on the web. */
export const Ext: React.FC<{ url: string; children: React.ReactNode; style?: React.CSSProperties; go?: number }> = ({
  url,
  children,
  style,
  go,
}) => (
  <a href={url} target="_blank" rel="noreferrer" style={go ? { ...GO_STYLE, fontSize: go, ...style } : style}>
    {children}
  </a>
);

/** A row of links under a paragraph. */
export const LinkRow: React.FC<{ children: React.ReactNode; gap?: number }> = ({ children, gap = 26 }) => (
  <div style={{ display: "flex", flexWrap: "wrap", gap: `0 ${gap}px` }}>{children}</div>
);

/* ------------------------------------------------------ links within this page */

/**
 * A link to an anchor on this page (#/lineage/<id>). The app scrolls to the anchor when the route changes;
 * when the hash is already that route, the click scrolls by itself.
 */
export const InPageLink: React.FC<{ id: string; children: React.ReactNode; style?: React.CSSProperties }> = ({
  id,
  children,
  style,
}) => {
  const to = href.lineage(id);
  return (
    <a
      href={to}
      style={style}
      onClick={(e) => {
        if (window.location.hash === to) {
          e.preventDefault();
          document.getElementById(id)?.scrollIntoView({ block: "start" });
        }
      }}
    >
      {children}
    </a>
  );
};

/** Anchors that live inside the apparatus: the sources section and each reference. */
export const isApparatusAnchor = (anchor: string | undefined): anchor is string =>
  !!anchor && /^(sources|src-head|ref-\d+)$/.test(anchor);

/**
 * Shown only when a link into the sources arrives while the apparatus is closed: the board keeps the sources
 * inside the apparatus, so the page says where the source is and offers to open it, rather than opening it unasked.
 */
export const ApparatusLinkNotice: React.FC<{ anchor: string | undefined }> = ({ anchor }) => {
  const apparatus = useReading((s) => s.apparatus);
  const toggleApparatus = useReading((s) => s.toggleApparatus);
  const [pending, setPending] = useState(false);

  useEffect(() => {
    if (pending && apparatus && anchor) {
      document.getElementById(anchor)?.scrollIntoView({ block: "start" });
      setPending(false);
    }
  }, [pending, apparatus, anchor]);

  if (apparatus || !isApparatusAnchor(anchor)) return null;
  const n = /^ref-(\d+)$/.exec(anchor)?.[1];
  return (
    <p className="status-line" role="status" style={{ margin: "18px 0 0" }}>
      {n
        ? `The link you followed points to source [${n}], which is listed with the sources in the apparatus. The apparatus is closed. `
        : "The link you followed points to the sources, which are listed in the apparatus. The apparatus is closed. "}
      <button
        type="button"
        className="btn-link"
        style={{ fontStyle: "normal" }}
        onClick={() => {
          setPending(true);
          toggleApparatus();
        }}
      >
        open the apparatus and go there
      </button>
    </p>
  );
};

/* ------------------------------------------------------------- apparatus */

/**
 * The page's apparatus blocks: ruled off by the 3px double gilt line and hidden unless the apparatus is open
 * (via the kit's IfApparatus). The "sources" link goes to the sources at the foot of this page.
 */
export const LinApparatus: React.FC<{
  title: string;
  label: string;
  style?: React.CSSProperties;
  children: React.ReactNode;
}> = ({ title, label, style, children }) => (
  <IfApparatus>
    <div role="note" aria-label={label} className="apparatus" style={style}>
      <div className="apparatus-head">
        <span className="kicker">apparatus · {title}</span>
        <InPageLink id="sources">sources</InPageLink>
      </div>
      <dl className="dl">{children}</dl>
    </div>
  </IfApparatus>
);

/* --------------------------------------------------------------- marks */

/** The canvas's bracketed note for a practice or fix the instrument does not have yet. */
export const NotInInstrument: React.FC = () => (
  <span className="placeholder" style={{ fontSize: 14 }}>
    [NOT YET IN THE INSTRUMENT]
  </span>
);

/** A file name, query or code identifier, in IBM Plex Mono. */
export const M: React.FC<{ children: React.ReactNode }> = ({ children }) => <span className="mono">{children}</span>;

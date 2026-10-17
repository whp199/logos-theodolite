import React from "react";

/* ---------------------------------------------------------------- cartouche */

/** The engraved maker's cartouche beside the title (drawn as on the canvas). */
export const Cartouche: React.FC = () => (
  <figure
    style={{
      flex: "1 1 380px",
      minWidth: 0,
      margin: 0,
      display: "flex",
      flexDirection: "column",
      gap: 12,
      alignItems: "center",
    }}
  >
    <svg
      viewBox="0 0 560 420"
      role="img"
      aria-label="An engraved maker’s cartouche: Logos Theodolite, made after Gurnee et al., Ayonrinde and Lindsey, Cammarata, and Hayuk; with the logit lens, linear CKA and activation steering; for Gemma-4-31B-it, 2026."
      style={{ width: "100%", maxWidth: 520, height: "auto", display: "block" }}
    >
      <path
        d="M30 8 H530 A22 22 0 0 0 552 30 V390 A22 22 0 0 0 530 412 H30 A22 22 0 0 0 8 390 V30 A22 22 0 0 0 30 8 Z"
        fill="#F6F0E3"
        stroke="#A07A2E"
        strokeWidth="1.4"
      />
      <path
        d="M37.7 17 H522.3 A31 31 0 0 0 543 37.7 V382.3 A31 31 0 0 0 522.3 403 H37.7 A31 31 0 0 0 17 382.3 V37.7 A31 31 0 0 0 37.7 17 Z"
        fill="none"
        stroke="#A07A2E"
        strokeWidth="0.7"
      />
      <path
        d="M8 8 m-3.5 0 a3.5 3.5 0 1 0 7 0 a3.5 3.5 0 1 0 -7 0 M552 8 m-3.5 0 a3.5 3.5 0 1 0 7 0 a3.5 3.5 0 1 0 -7 0 M8 412 m-3.5 0 a3.5 3.5 0 1 0 7 0 a3.5 3.5 0 1 0 -7 0 M552 412 m-3.5 0 a3.5 3.5 0 1 0 7 0 a3.5 3.5 0 1 0 -7 0"
        fill="none"
        stroke="#A07A2E"
        strokeWidth="1"
      />
      <path
        d="M280 1 L285 8 L280 15 L275 8 Z M280 405 L285 412 L280 419 L275 412 Z M8 203 L13 210 L8 217 L3 210 Z M552 203 L557 210 L552 217 L547 210 Z"
        fill="#A07A2E"
      />
      {/* the theodolite on its tripod */}
      <line x1="280" y1="108" x2="252" y2="148" stroke="#2B2117" strokeWidth="1.1" />
      <line x1="280" y1="108" x2="308" y2="148" stroke="#2B2117" strokeWidth="1.1" />
      <line x1="280" y1="108" x2="272" y2="150" stroke="#2B2117" strokeOpacity="0.45" strokeWidth="1" />
      <line x1="280" y1="104" x2="280" y2="132" stroke="#2B2117" strokeOpacity="0.6" strokeWidth="0.6" />
      <path d="M280 132 L283.5 138 L280 145 L276.5 138 Z" fill="#8F2B1E" />
      <line x1="268" y1="94" x2="268" y2="66" stroke="#2B2117" strokeWidth="1.1" />
      <line x1="292" y1="94" x2="292" y2="61" stroke="#2B2117" strokeWidth="1.1" />
      <ellipse cx="280" cy="98" rx="34" ry="8" fill="#F6F0E3" stroke="#A07A2E" strokeWidth="1.2" />
      <ellipse cx="280" cy="98" rx="25" ry="5.6" fill="none" stroke="#A07A2E" strokeOpacity="0.6" strokeWidth="0.7" />
      <path d="M247.5 72.6 L314.1 59.2 L312.5 51.4 L245.9 64.8 Z" fill="#F6F0E3" stroke="#2B2117" strokeWidth="1.1" />
      <path d="M314.5 61.2 L312.1 49.4" stroke="#2B2117" strokeWidth="2" />
      <path d="M245.9 64.8 L241.8 65.6 L243.1 71.8 L247.5 72.6" fill="none" stroke="#2B2117" strokeWidth="1" />
      <circle cx="280" cy="62" r="3.2" fill="#A07A2E" />
      <g fontFamily="EB Garamond, Garamond, serif" textAnchor="middle">
        <text x="280" y="190" fontSize="30" letterSpacing="5" fontWeight="500" fill="#2B2117">
          LOGOS THEODOLITE
        </text>
        <line x1="196" y1="208" x2="272" y2="208" stroke="#A07A2E" strokeWidth="0.9" />
        <path d="M280 203 L284 208 L280 213 L276 208 Z" fill="#A07A2E" />
        <line x1="288" y1="208" x2="364" y2="208" stroke="#A07A2E" strokeWidth="0.9" />
        <text x="280" y="240" fontSize="19" fontStyle="italic" fill="#6B5D4D">
          made after
        </text>
        <text x="280" y="272" fontSize="23" fill="#2B2117">
          Gurnee et al., Ayonrinde &amp; Lindsey,
        </text>
        <text x="280" y="302" fontSize="23" fill="#2B2117">
          Cammarata, and Hayuk
        </text>
        <text x="280" y="336" fontSize="15" fontStyle="italic" fill="#6B5D4D">
          with the logit lens, linear CKA and activation steering
        </text>
        <line x1="240" y1="356" x2="320" y2="356" stroke="#C9A85F" strokeWidth="0.9" />
        <text
          x="280"
          y="382"
          fontSize="15"
          letterSpacing="2.5"
          fill="#7A5C28"
          style={{ fontVariantCaps: "all-small-caps", fontVariantNumeric: "lining-nums" }}
        >
          for Gemma-4-31B-it · MMXXVI
        </text>
      </g>
    </svg>
    <figcaption
      style={{ fontSize: 15, fontStyle: "italic", color: "var(--muted)", textAlign: "center", maxWidth: 440 }}
    >
      Instrument-makers engraved their names in a framed panel like this one, on the brass itself or on their printed
      trade cards.
    </figcaption>
  </figure>
);

/* ------------------------------------------------------------------ stemma */

type BoxKind = "foundation" | "pillar" | "earlier" | "instrument" | "study";

const BORDER: Record<BoxKind, string> = {
  foundation: "1px solid var(--control)",
  pillar: "1px solid var(--gilt)",
  earlier: "1px dashed var(--control)",
  instrument: "3px double var(--gilt)",
  study: "1px solid var(--control)",
};

/** One node of the family tree, placed on the canvas's 1280 × 880 grid. */
const Node: React.FC<{
  kind: BoxKind;
  left: number;
  top: number;
  width?: number;
  height: number;
  children: React.ReactNode;
}> = ({ kind, left, top, width = 172, height, children }) => (
  <div
    style={{
      position: "absolute",
      left,
      top,
      width,
      height,
      padding: 10,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      textAlign: "center",
      background: "var(--vellum)",
      border: BORDER[kind],
      lineHeight: 1.3,
    }}
  >
    <span>{children}</span>
  </div>
);

const Kicker: React.FC<{ children: React.ReactNode; muted?: boolean }> = ({ children, muted }) => (
  <>
    <span
      style={{
        fontSize: 13,
        fontVariantCaps: "all-small-caps",
        letterSpacing: "0.18em",
        color: muted ? "var(--muted)" : "var(--gilt-text)",
      }}
    >
      {children}
    </span>
    <br />
  </>
);

const Name: React.FC<{ children: React.ReactNode }> = ({ children }) => <span style={{ fontSize: 17 }}>{children}</span>;

const Cite: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <>
    <br />
    <span style={{ fontSize: 14, color: "var(--muted)" }}>{children}</span>
  </>
);

/**
 * The connectors of the canvas's stemma, in the figure's own 1280 × 880 coordinates
 * (each was a small absolutely placed SVG on the canvas; here they share one).
 */
const ARROWS: { d: string; dashed?: boolean }[] = [
  { d: "M90 128 L90 348" }, // the logit lens → pillar i
  { d: "M310 128 L310 204" }, // relevance propagation → the R-lens
  { d: "M310 304 L310 348" }, // the R-lens → pillar ii
  { d: "M530 128 L530 348" }, // GLU variants → pillar iii
  { d: "M176 424 L220 424" }, // pillar i → pillar ii
  { d: "M664 424 L620 424", dashed: true }, // the earlier way, set beside pillar iii
  { d: "M640 544 L640 612" }, // all of the above → the instrument
  { d: "M640 712 L640 764" }, // the instrument → the study
];
const LINES: string[] = [
  "M750 128 L750 168", // the circuits framework
  "M1190 128 L1190 168", // activation steering
  "M750 168 L1190 168",
  "M970 128 L970 544", // linear CKA, and the two above, lent directly
  "M90 496 L90 544", // pillar i
  "M310 496 L310 544", // pillar ii
  "M530 496 L530 544", // pillar iii
  "M1190 496 L1190 544", // pillar iv
  "M90 544 L1190 544",
];

const LEGEND: { kind: BoxKind; label: string }[] = [
  { kind: "foundation", label: "an older foundation" },
  { kind: "pillar", label: "one of the four pillars" },
  { kind: "earlier", label: "an earlier method, set beside a pillar for comparison" },
  { kind: "instrument", label: "this instrument" },
];

/** i · the family tree: older foundations, the four pillars, the instrument and the study made with it. */
export const Stemma: React.FC = () => (
  <figure style={{ margin: "6px 0 0", display: "flex", flexDirection: "column", gap: 14 }}>
    <div
      role="region"
      aria-label="The family tree; on a narrow screen it scrolls sideways"
      tabIndex={0}
      style={{ overflowX: "auto", paddingBottom: 6 }}
    >
      <div style={{ position: "relative", width: 1280, height: 880, margin: "0 auto" }}>
        <svg
          width="1280"
          height="880"
          viewBox="0 0 1280 880"
          aria-hidden="true"
          style={{ position: "absolute", left: 0, top: 0, overflow: "visible" }}
          fill="none"
          stroke="#6B5D4D"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <defs>
            <marker id="lin-stemma-arrow" orient="auto" markerWidth="5" markerHeight="5" refX="2" refY="2" overflow="visible">
              <path d="M0 0 L4 2 L0 4 Z" fill="#6B5D4D" stroke="none" />
            </marker>
          </defs>
          {LINES.map((d) => (
            <path key={d} d={d} />
          ))}
          {ARROWS.map((a) => (
            <path key={a.d} d={a.d} markerEnd="url(#lin-stemma-arrow)" strokeDasharray={a.dashed ? "4 4" : undefined} />
          ))}
        </svg>

        {/* older foundations */}
        <Node kind="foundation" left={4} top={8} height={120}>
          <Name>the logit lens</Name>
          <Cite>nostalgebraist 2020 · [8]</Cite>
        </Node>
        <Node kind="foundation" left={224} top={8} height={120}>
          <Name>relevance propagation (LRP)</Name>
          <Cite>Bach et al. 2015 · [17]</Cite>
        </Node>
        <Node kind="foundation" left={444} top={8} height={120}>
          <Name>GLU variants</Name>
          <Cite>Shazeer 2020 · [15]</Cite>
        </Node>
        <Node kind="foundation" left={664} top={8} height={120}>
          <Name>the transformer circuits framework</Name>
          <Cite>Elhage et al. 2021 · [14]</Cite>
        </Node>
        <Node kind="foundation" left={884} top={8} height={120}>
          <Name>linear CKA</Name>
          <Cite>Kornblith et al. 2019 · [13]</Cite>
        </Node>
        <Node kind="foundation" left={1104} top={8} height={120}>
          <Name>activation steering</Name>
          <Cite>Turner et al. 2023; Panickssery et al. 2024 · [21] [22]</Cite>
        </Node>
        <Node kind="foundation" left={224} top={208} height={96}>
          <Name>the R-lens</Name>
          <Cite>Blank, Bhatia &amp; Nanda 2026 · [5]</Cite>
        </Node>

        {/* the four pillars, and the earlier way beside pillar iii */}
        <Node kind="pillar" left={4} top={352} height={144}>
          <Kicker>pillar i</Kicker>
          <Name>the Jacobian lens and J-space</Name>
          <Cite>Gurnee et al. 2026 · [2] [3] [6]</Cite>
        </Node>
        <Node kind="pillar" left={224} top={352} height={144}>
          <Kicker>pillar ii</Kicker>
          <Name>the J++ lens</Name>
          <Cite>Ayonrinde &amp; Lindsey 2026 · [4]</Cite>
        </Node>
        <Node kind="pillar" left={444} top={352} height={144}>
          <Kicker>pillar iii</Kicker>
          <Name>neuron poles</Name>
          <Cite>
            Cammarata 2026, <i>Neural Networks Are Already Clean</i> · [9]
          </Cite>
        </Node>
        <Node kind="earlier" left={664} top={360} height={128}>
          <Kicker muted>the earlier way</Kicker>
          <Name>labels from what wakes a neuron</Name>
          <Cite>Bills et al. 2023 · [25]</Cite>
        </Node>
        <Node kind="pillar" left={1104} top={352} height={144}>
          <Kicker>pillar iv</Kicker>
          <Name>LARQL and VINDEX</Name>
          <Cite>Hayuk · [10]</Cite>
        </Node>

        {/* the instrument and the study */}
        <Node kind="instrument" left={440} top={616} width={400} height={96}>
          <span style={{ fontSize: 24, fontVariantCaps: "all-small-caps", letterSpacing: "0.16em", color: "var(--ink)" }}>
            Logos Theodolite
          </span>
          <Cite>the instrument these pages describe, joined from all of the above</Cite>
        </Node>
        <Node kind="study" left={440} top={768} width={400} height={96}>
          <span style={{ fontSize: 19 }}>the study</span>
          <br />
          <span style={{ fontSize: 17, fontStyle: "italic" }}>Gemma-4: is it female-coded?</span>
          <br />
          <span style={{ fontFamily: "var(--mono)", fontSize: 12.5, color: "var(--muted)" }}>experiments/REPORT.md</span>
        </Node>

        <div
          style={{
            position: "absolute",
            left: 982,
            top: 252,
            width: 104,
            fontSize: 13,
            lineHeight: "16px",
            fontStyle: "italic",
            color: "var(--muted)",
          }}
        >
          lent directly
        </div>
      </div>
    </div>
    <figcaption style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 26px", fontSize: 15, color: "var(--muted)" }}>
        {LEGEND.map((l) => (
          <span key={l.kind} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span aria-hidden="true" style={{ width: 22, height: 14, border: BORDER[l.kind], display: "inline-block" }} />
            {l.label}
          </span>
        ))}
      </div>
      <p style={{ margin: 0, fontSize: 16, color: "var(--muted)", maxWidth: 900 }}>
        In words: the logit lens leads to the Jacobian lens; the Jacobian lens and the R-lens, which comes from relevance
        propagation, lead to J++; GLU leads to neuron poles, with Bills and colleagues’ activation labels set beside them.
        The circuits framework, linear CKA and activation steering pass straight to the instrument, as do the four
        pillars. The instrument made the study.
      </p>
    </figcaption>
  </figure>
);

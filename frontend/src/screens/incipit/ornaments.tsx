import React from "react";

/* Static drawings of the Incipit, as on the approved canvas. All are decorative (aria-hidden) except
 * the miniature leaf, which is a figure with its own label. */

const SERIF = "EB Garamond, Garamond, serif";

/* the ticks round the initial "L": every 10°, longer every 30° */
const INITIAL_TICKS = (() => {
  const f = (n: number) => Math.round(n * 10) / 10;
  let d = "";
  for (let deg = 0; deg < 360; deg += 10) {
    const a = (deg * Math.PI) / 180;
    const r1 = deg % 30 === 0 ? 55 : 63;
    d += "M" + f(100 + r1 * Math.cos(a)) + "," + f(100 + r1 * Math.sin(a)) + " L" + f(100 + 70 * Math.cos(a)) + "," + f(100 + 70 * Math.sin(a)) + " ";
  }
  return d;
})();

/** The engraved initial "L" of the title, set in a theodolite's limb. */
export const TitleInitial: React.FC = () => (
  <svg viewBox="0 0 200 200" aria-hidden="true" style={{ width: "clamp(96px, 14vw, 200px)", height: "auto", display: "block", flex: "none" }}>
    <rect x="0.5" y="0.5" width="199" height="199" fill="none" stroke="#8F2B1E" strokeWidth="1" />
    <rect x="7" y="7" width="186" height="186" fill="none" stroke="#A07A2E" strokeWidth="1.4" />
    <rect x="11" y="11" width="178" height="178" fill="none" stroke="#A07A2E" strokeWidth="0.6" />
    <path d="M7 2 L12 7 L7 12 L2 7 Z M193 2 L198 7 L193 12 L188 7 Z M7 188 L12 193 L7 198 L2 193 Z M193 188 L198 193 L193 198 L188 193 Z" fill="#A07A2E" />
    <circle cx="100" cy="100" r="70" fill="none" stroke="#A07A2E" strokeWidth="0.9" strokeOpacity="0.7" />
    <circle cx="100" cy="100" r="63" fill="none" stroke="#A07A2E" strokeWidth="0.5" strokeOpacity="0.6" />
    <path d={INITIAL_TICKS} fill="none" stroke="#A07A2E" strokeWidth="0.7" strokeOpacity="0.75" />
    <ellipse cx="100" cy="100" rx="70" ry="20" fill="none" stroke="#A07A2E" strokeWidth="0.6" strokeOpacity="0.55" />
    <line x1="100" y1="22" x2="100" y2="178" stroke="#2B2117" strokeWidth="0.6" strokeOpacity="0.3" />
    <line x1="22" y1="100" x2="178" y2="100" stroke="#2B2117" strokeWidth="0.6" strokeOpacity="0.3" />
    <text x="92" y="161" textAnchor="middle" fontFamily={SERIF} fontSize="164" fontWeight="500" fill="#8A6A2C" stroke="#F6F0E3" strokeWidth="5" paintOrder="stroke">
      L
    </text>
    <circle cx="104" cy="100" r="6" fill="none" stroke="#8F2B1E" strokeWidth="0.9" />
    <circle cx="104" cy="100" r="2" fill="#8F2B1E" />
  </svg>
);

export const RuleWithLozenge: React.FC = () => (
  <svg width="140" height="12" viewBox="0 0 140 12" aria-hidden="true" style={{ display: "block" }}>
    <line x1="0" y1="6" x2="58" y2="6" stroke="#A07A2E" strokeWidth="1" />
    <path d="M70 0 L76 6 L70 12 L64 6 Z" fill="#A07A2E" />
    <line x1="82" y1="6" x2="140" y2="6" stroke="#A07A2E" strokeWidth="1" />
  </svg>
);

export const SectionDivider: React.FC = () => (
  <div aria-hidden="true" style={{ display: "flex", alignItems: "center", gap: 18 }}>
    <div style={{ flex: "1 1 0", height: 1, background: "#D9CFBD" }} />
    <svg width="34" height="12" viewBox="0 0 34 12">
      <path d="M5 3 L8 6 L5 9 L2 6 Z M17 0 L23 6 L17 12 L11 6 Z M29 3 L32 6 L29 9 L26 6 Z" fill="#A07A2E" />
    </svg>
    <div style={{ flex: "1 1 0", height: 1, background: "#D9CFBD" }} />
  </div>
);

export const Diamond10: React.FC<{ fill: string }> = ({ fill }) => (
  <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
    <path d="M5 0 L10 5 L5 10 L0 5 Z" fill={fill} />
  </svg>
);

/* ---- how a sighting works: the three small drawings ---- */

export const StepAsk: React.FC = () => (
  <svg width="160" height="96" viewBox="0 0 160 96" aria-hidden="true" style={{ display: "block" }}>
    <line x1="8" y1="82" x2="152" y2="82" stroke="#C9BEAC" strokeWidth="1" />
    <path d="M12 78 c4 -10 9 -10 11 -2 c2 7 6 7 9 0 c3 -8 7 -8 9 -1 c2 6 6 6 9 0 c3 -7 7 -7 9 0 c2 5 5 5 8 0" fill="none" stroke="#2B2117" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M76 70 L80 74 L76 78 L72 74 Z" fill="#A07A2E" />
    <line x1="86" y1="80" x2="146" y2="12" stroke="#2B2117" strokeWidth="1" />
    <path d="M97 67.5 C 96 50, 116 30, 146 12 C 140 36, 120 58, 97 67.5 Z" fill="none" stroke="#A07A2E" strokeWidth="1" />
    <path d="M104 59.6 L99 54.5 M113 49.4 L107 44 M122 39.2 L116 33.5 M131 29 L126 24 M104 59.6 L110 64 M113 49.4 L119.5 54 M122 39.2 L128.5 43.5 M131 29 L137 32.5" fill="none" stroke="#A07A2E" strokeWidth="0.7" strokeOpacity="0.7" />
  </svg>
);

export const StepSight: React.FC = () => (
  <svg width="160" height="96" viewBox="0 0 160 96" aria-hidden="true" style={{ display: "block" }}>
    <ellipse cx="80" cy="84" rx="60" ry="9" fill="none" stroke="#A07A2E" strokeWidth="1" />
    <path d="M140 84 L131 84 M132 88.5 L124.2 87.8 M110 91.8 L105.5 90.6 M80 93 L80 91.6 M50 91.8 L54.5 90.6 M28 88.5 L35.8 87.8 M20 84 L29 84 M28 79.5 L35.8 80.2 M50 76.2 L54.5 77.4 M110 76.2 L105.5 77.4 M132 79.5 L124.2 80.2" fill="none" stroke="#A07A2E" strokeWidth="0.8" />
    <g fill="none" stroke="#2B2117" strokeOpacity="0.4" strokeWidth="0.8">
      <ellipse cx="80" cy="76" rx="44" ry="7" />
      <ellipse cx="80" cy="66" rx="44" ry="7" />
      <ellipse cx="80" cy="46" rx="44" ry="7" />
      <ellipse cx="80" cy="36" rx="44" ry="7" />
    </g>
    <g fill="none" stroke="#A07A2E" strokeWidth="1">
      <ellipse cx="80" cy="56" rx="44" ry="7" />
      <ellipse cx="80" cy="26" rx="44" ry="7" />
    </g>
    <line x1="80" y1="90" x2="80" y2="12" stroke="#2B2117" strokeOpacity="0.45" strokeWidth="1" />
    <path d="M80 2 L84 8 L80 14 L76 8 Z" fill="#A07A2E" />
  </svg>
);

export const StepRead: React.FC = () => (
  <svg width="160" height="96" viewBox="0 0 160 96" aria-hidden="true" style={{ display: "block" }}>
    <g fill="none" stroke="#2B2117" strokeOpacity="0.13" strokeWidth="0.8">
      <ellipse cx="80" cy="80" rx="44" ry="7" />
      <ellipse cx="80" cy="62" rx="44" ry="7" />
      <ellipse cx="80" cy="44" rx="44" ry="7" />
      <ellipse cx="80" cy="26" rx="44" ry="7" />
    </g>
    <line x1="16" y1="88" x2="16" y2="10" stroke="#6B5D4D" strokeWidth="0.8" />
    <path d="M12 88 L20 88 M13 75 L19 75 M13 62 L19 62 M13 49 L19 49 M13 36 L19 36 M13 23 L19 23 M12 10 L20 10" fill="none" stroke="#6B5D4D" strokeWidth="0.8" />
    <path d="M70 86 C 44 78, 108 68, 84 56 S 52 38, 86 28 S 98 16, 92 12" fill="none" stroke="#2B2117" strokeWidth="1.3" strokeLinecap="round" />
    <circle cx="70" cy="86" r="2.4" fill="#2B2117" />
    <circle cx="84" cy="56" r="2.6" fill="#F6F0E3" stroke="#2B2117" strokeWidth="1" />
    <circle cx="86" cy="28" r="2.6" fill="#F6F0E3" stroke="#2B2117" strokeWidth="1" />
    <circle cx="92" cy="12" r="3.4" fill="#A07A2E" stroke="#2B2117" strokeWidth="1" />
    <line x1="97" y1="12" x2="112" y2="12" stroke="#7A5C28" strokeWidth="0.7" strokeDasharray="2 2" />
    <text x="115" y="16" fontFamily={SERIF} fontStyle="italic" fontSize="14" fill="#2B2117">
      “am”
    </text>
  </svg>
);

/* ---- two ways to read: a printed leaf in miniature ---- */

export const LeafFigure: React.FC<{ rubrics: boolean; apparatus: boolean }> = ({ rubrics, apparatus }) => {
  const rOp = rubrics ? 1 : 0.18;
  const aOp = apparatus ? 1 : 0.22;
  return (
    <figure style={{ flex: "0 1 300px", minWidth: 0, margin: 0, display: "flex", flexDirection: "column", gap: 8 }}>
      <svg
        viewBox="0 0 300 250"
        role="img"
        aria-label="A printed leaf in miniature. Its heading and one direction, set in red, are the rubrics. Beneath a double gilt rule, in small type, lies the apparatus."
        style={{ width: "100%", maxWidth: 300, height: "auto", display: "block" }}
      >
        <rect x="8.5" y="8.5" width="176" height="232" fill="none" stroke="#C9BEAC" strokeWidth="1" />
        <rect x="28" y="25" width="64" height="3.2" fill="#8F2B1E" opacity={rOp} />
        <g fill="#2B2117" fillOpacity="0.5">
          <rect x="28" y="42" width="136" height="2" />
          <rect x="28" y="50" width="136" height="2" />
          <rect x="28" y="58" width="136" height="2" />
          <rect x="28" y="66" width="136" height="2" />
          <rect x="28" y="74" width="92" height="2" />
          <rect x="28" y="102" width="136" height="2" />
          <rect x="28" y="110" width="136" height="2" />
          <rect x="28" y="118" width="136" height="2" />
          <rect x="28" y="126" width="136" height="2" />
          <rect x="28" y="134" width="70" height="2" />
        </g>
        <rect x="28" y="88" width="122" height="2" fill="#8F2B1E" opacity={rOp} />
        <line x1="28" y1="157.5" x2="164" y2="157.5" stroke="#C9A85F" strokeWidth="1" />
        <line x1="28" y1="160.5" x2="164" y2="160.5" stroke="#C9A85F" strokeWidth="1" />
        <g fill="#7A5C28" opacity={aOp}>
          <rect x="28" y="172" width="40" height="1.6" />
          <rect x="78" y="172" width="86" height="1.6" />
          <rect x="28" y="180" width="34" height="1.6" />
          <rect x="78" y="180" width="86" height="1.6" />
          <rect x="78" y="188" width="60" height="1.6" />
          <rect x="28" y="196" width="44" height="1.6" />
          <rect x="78" y="196" width="86" height="1.6" />
          <rect x="28" y="204" width="30" height="1.6" />
          <rect x="78" y="204" width="86" height="1.6" />
          <rect x="78" y="212" width="74" height="1.6" />
          <rect x="28" y="220" width="38" height="1.6" />
          <rect x="78" y="220" width="52" height="1.6" />
        </g>
        <path d="M196 54 L98 28 M196 60 L154 88" fill="none" stroke="#8F2B1E" strokeOpacity="0.55" strokeWidth="0.7" strokeDasharray="2 3" />
        <path d="M196 192 L170 192" fill="none" stroke="#7A5C28" strokeOpacity="0.65" strokeWidth="0.7" strokeDasharray="2 3" />
        <text x="200" y="62" fontFamily={SERIF} fontStyle="italic" fontSize="16" fill="#8F2B1E">
          rubrics
        </text>
        <text x="200" y="78" fontFamily={SERIF} fontStyle="italic" fontSize="12.5" fill="#6B5D4D">
          set in red
        </text>
        <text x="200" y="196" fontFamily={SERIF} fontStyle="italic" fontSize="16" fill="#7A5C28">
          apparatus
        </text>
        <text x="200" y="212" fontFamily={SERIF} fontStyle="italic" fontSize="12.5" fill="#6B5D4D">
          beneath the
        </text>
        <text x="200" y="226" fontFamily={SERIF} fontStyle="italic" fontSize="12.5" fill="#6B5D4D">
          double rule
        </text>
      </svg>
      <figcaption style={{ fontSize: 14, fontVariantCaps: "all-small-caps", letterSpacing: "0.2em", color: "#6B5D4D" }}>
        a leaf, with its rubrics and its apparatus
      </figcaption>
    </figure>
  );
};

/* ---- the makers' marks: the engraved frame ---- */

const Corner: React.FC<{ style: React.CSSProperties }> = ({ style }) => (
  <svg width="46" height="46" viewBox="0 0 46 46" aria-hidden="true" style={{ position: "absolute", display: "block", ...style }}>
    <rect x="0" y="0" width="46" height="46" fill="#F6F0E3" />
    <path d="M6.5 46 L6.5 26.5 A20 20 0 0 0 26.5 6.5 L46 6.5" fill="none" stroke="#A07A2E" strokeWidth="1" />
    <path d="M11.5 46 L11.5 31 A25 25 0 0 0 31 11.5 L46 11.5" fill="none" stroke="#C9A85F" strokeWidth="1" />
    <path d="M9 4.5 L13.5 9 L9 13.5 L4.5 9 Z" fill="#C9A85F" />
    <circle cx="27.5" cy="27.5" r="1.6" fill="#A07A2E" />
  </svg>
);

export const MakersFrame: React.FC = () => (
  <>
    <div aria-hidden="true" style={{ position: "absolute", top: 6, right: 6, bottom: 6, left: 6, border: "1px solid #A07A2E" }} />
    <div aria-hidden="true" style={{ position: "absolute", top: 11, right: 11, bottom: 11, left: 11, border: "1px solid #C9A85F" }} />
    <Corner style={{ top: 0, left: 0 }} />
    <Corner style={{ top: 0, right: 0, transform: "scaleX(-1)" }} />
    <Corner style={{ bottom: 0, left: 0, transform: "scaleY(-1)" }} />
    <Corner style={{ bottom: 0, right: 0, transform: "scale(-1, -1)" }} />
    <svg width="120" height="18" viewBox="0 0 120 18" aria-hidden="true" style={{ position: "absolute", top: 0, left: "50%", marginLeft: -60, display: "block" }}>
      <rect x="0" y="0" width="120" height="18" fill="#F6F0E3" />
      <path d="M0 6.5 H30 M30 6.5 A2.5 2.5 0 0 1 30 11.5 M90 6.5 H120 M90 6.5 A2.5 2.5 0 0 0 90 11.5" fill="none" stroke="#A07A2E" strokeWidth="1" />
      <path d="M0 11.5 H30 M90 11.5 H120" fill="none" stroke="#C9A85F" strokeWidth="1" />
      <path d="M60 1 L68 9 L60 17 L52 9 Z" fill="#A07A2E" />
      <path d="M43 6.5 L45.5 9 L43 11.5 L40.5 9 Z M77 6.5 L79.5 9 L77 11.5 L74.5 9 Z" fill="#C9A85F" />
    </svg>
    <svg
      width="120"
      height="18"
      viewBox="0 0 120 18"
      aria-hidden="true"
      style={{ position: "absolute", bottom: 0, left: "50%", marginLeft: -60, display: "block", transform: "scaleY(-1)" }}
    >
      <rect x="0" y="0" width="120" height="18" fill="#F6F0E3" />
      <path d="M0 6.5 H30 M30 6.5 A2.5 2.5 0 0 1 30 11.5 M90 6.5 H120 M90 6.5 A2.5 2.5 0 0 0 90 11.5" fill="none" stroke="#A07A2E" strokeWidth="1" />
      <path d="M0 11.5 H30 M90 11.5 H120" fill="none" stroke="#C9A85F" strokeWidth="1" />
      <path d="M60 4 L65 9 L60 14 L55 9 Z" fill="#A07A2E" />
    </svg>
  </>
);

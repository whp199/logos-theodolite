import React, { useMemo } from "react";
import { roman } from "./text";

/**
 * The engraved dial of the leaf (Layer.dc.html): the layer's numeral at the hub of a level circle that
 * carries the bearing (centred 250,250), beside an upright quadrant that carries the elevation (centred
 * 600,470). The canvas's quadrant runs 0-90°; a thought that sinks below the level (negative elevation)
 * gets the mirrored lower quadrant, so its needle is never drawn off the plate.
 */

const f1 = (n: number) => Math.round(n * 10) / 10;
type Pt = [number, number];
const P = (cx: number, cy: number, r: number, d: number): Pt => {
  const a = (d * Math.PI) / 180;
  return [f1(cx + r * Math.cos(a)), f1(cy + r * Math.sin(a))];
};
const pt = (p: Pt) => `${p[0]},${p[1]}`;
const seg = (cx: number, cy: number, r1: number, r2: number, d: number) => `M${pt(P(cx, cy, r1, d))} L${pt(P(cx, cy, r2, d))} `;
const loz = (p: Pt, s: number) =>
  `M${p[0]},${f1(p[1] - s)} L${f1(p[0] + s)},${p[1]} L${p[0]},${f1(p[1] + s)} L${f1(p[0] - s)},${p[1]} Z`;

/** The fixed engraving: limb ticks of the level circle, the upper and the lower quadrant. */
const ENGRAVING = (() => {
  let limbFine = "";
  let limbMid = "";
  let limbMajor = "";
  for (let d = 0; d < 360; d++) {
    if (d % 30 === 0) limbMajor += seg(250, 250, 215, 195, d);
    else if (d % 5 === 0) limbMid += seg(250, 250, 215, d % 10 === 0 ? 202 : 206, d);
    else limbFine += seg(250, 250, 215, 210, d);
  }
  const quad = (sign: 1 | -1) => {
    let fine = "";
    let mid = "";
    let major = "";
    for (let e = 0; e <= 90; e++) {
      const d = -sign * e;
      if (e % 30 === 0) major += seg(600, 470, 190, 170, d);
      else if (e % 5 === 0) mid += seg(600, 470, 190, e % 10 === 0 ? 177 : 181, d);
      else fine += seg(600, 470, 190, 185, d);
    }
    return { fine, mid, major };
  };
  return { limbFine, limbMid, limbMajor, up: quad(1), down: quad(-1) };
})();

const LIMB_LABELS: [number, number, string][] = [
  [430, 254.5, "0"], [405.9, 344.5, "30"], [340, 410.4, "60"], [250, 434.5, "90"], [160, 410.4, "120"],
  [94.1, 344.5, "150"], [70, 254.5, "180"], [94.1, 164.5, "210"], [160, 98.6, "240"], [250, 74.5, "270"],
  [340, 98.6, "300"], [405.9, 164.5, "330"],
];

const SERIF = "EB Garamond, Garamond, serif";

export const Dial: React.FC<{
  layerNo: number;
  az: number | null;
  el: number | null;
  landmarks: [string, string, string];
  ariaLabel: string;
}> = ({ layerNo, az, el, landmarks, ariaLabel }) => {
  const sunk = el !== null && el < 0;
  const marks = useMemo(() => {
    const out: { azArc?: string; azIdx?: string; azLoz?: string; elArc?: string; elNeedle?: string; elLoz?: string } = {};
    if (az !== null) {
      const x = ((az + 540) % 360) - 180;
      if (Math.abs(x) >= 0.05) out.azArc = `M${pt(P(250, 250, 162, 0))} A162,162 0 0,${x >= 0 ? 1 : 0} ${pt(P(250, 250, 162, x))}`;
      out.azIdx = seg(250, 250, 162, 232, x);
      out.azLoz = loz(P(250, 250, 243, x), 6);
    }
    if (el !== null) {
      if (Math.abs(el) >= 0.05) out.elArc = `M${pt(P(600, 470, 96, 0))} A96,96 0 0,${el >= 0 ? 0 : 1} ${pt(P(600, 470, 96, -el))}`;
      out.elNeedle = seg(600, 470, 6, 196, -el);
      out.elLoz = loz(P(600, 470, 207, -el), 6);
    }
    return out;
  }, [az, el]);
  const [a, b, c] = landmarks.map((w) => w || "?");

  return (
    <svg
      viewBox={`0 0 830 ${sunk ? 700 : 520}`}
      role="img"
      aria-label={ariaLabel}
      style={{ width: "100%", maxWidth: 830, height: "auto", display: "block", margin: "0 auto" }}
    >
      <circle cx="250" cy="250" r="224" fill="none" stroke="#A07A2E" strokeWidth="0.8" strokeOpacity="0.6" />
      <circle cx="250" cy="250" r="215" fill="none" stroke="#A07A2E" strokeWidth="1.3" />
      <circle cx="250" cy="250" r="190" fill="none" stroke="#A07A2E" strokeWidth="0.7" strokeOpacity="0.55" />
      <circle cx="250" cy="250" r="150" fill="none" stroke="#2B2117" strokeWidth="0.6" strokeOpacity="0.14" />
      <circle cx="250" cy="250" r="118" fill="none" stroke="#2B2117" strokeWidth="0.6" strokeOpacity="0.2" />
      <path d={ENGRAVING.limbFine} fill="none" stroke="#A07A2E" strokeWidth="0.5" strokeOpacity="0.75" />
      <path d={ENGRAVING.limbMid} fill="none" stroke="#A07A2E" strokeWidth="0.85" />
      <path d={ENGRAVING.limbMajor} fill="none" stroke="#7A5C28" strokeWidth="1.3" />
      <path
        d="M250 132 L250 100 M250 368 L250 400 M132 250 L100 250 M368 250 L400 250"
        fill="none"
        stroke="#2B2117"
        strokeWidth="0.8"
        strokeOpacity="0.35"
      />
      <g fontFamily={SERIF} fontSize="12.5" fill="#7A5C28" textAnchor="middle" style={{ fontVariantNumeric: "lining-nums tabular-nums" }}>
        {LIMB_LABELS.map(([x, y, t]) => (
          <text key={t} x={x} y={y}>
            {t}
          </text>
        ))}
      </g>
      <text x="250" y="20" textAnchor="middle" fontFamily={SERIF} fontSize="18" fill="#A07A2E">
        ✦
      </text>
      <text x="250" y="170" textAnchor="middle" fontFamily={SERIF} fontSize="13" letterSpacing="4" fill="#6B5D4D">
        LAYER
      </text>
      <text
        x="250"
        y="312"
        textAnchor="middle"
        fontFamily={SERIF}
        fontSize="190"
        fontWeight="500"
        fill="#8A6A2C"
        style={{ fontVariantNumeric: "lining-nums" }}
      >
        {layerNo}
      </text>
      <text x="250" y="352" textAnchor="middle" fontFamily={SERIF} fontSize="14" letterSpacing="5" fill="#7A5C28">
        {roman(layerNo)}
      </text>
      {marks.azArc ? <path d={marks.azArc} fill="none" stroke="#8F2B1E" strokeWidth="1.2" /> : null}
      {marks.azIdx ? <path d={marks.azIdx} fill="none" stroke="#8F2B1E" strokeWidth="1.6" strokeLinecap="round" /> : null}
      {marks.azLoz ? <path d={marks.azLoz} fill="#8F2B1E" /> : null}
      <g fontFamily={SERIF} fontStyle="italic" fontSize="19" fill="#2E4A7D">
        <text x="484" y="256">
          {a}{" "}
          <tspan fontStyle="normal" fontSize="12.5" fill="#6B5D4D">
            0°
          </tspan>
        </text>
        <text x="250" y="508" textAnchor="middle">
          {b}{" "}
          <tspan fontStyle="normal" fontSize="12.5" fill="#6B5D4D">
            90°
          </tspan>
        </text>
        <text x="600" y="262" textAnchor="middle">
          {c}{" "}
          <tspan fontStyle="normal" fontSize="12.5" fill="#6B5D4D">
            up
          </tspan>
        </text>
        {sunk ? (
          <text x="600" y="694" textAnchor="middle">
            <tspan fontStyle="normal" fontSize="12.5" fill="#6B5D4D">
              away from{" "}
            </tspan>
            {c}
          </text>
        ) : null}
      </g>
      <path d="M790 470 A190 190 0 0 0 600 280" fill="none" stroke="#A07A2E" strokeWidth="1.3" />
      <path d="M796 470 A196 196 0 0 0 600 274" fill="none" stroke="#A07A2E" strokeWidth="0.7" strokeOpacity="0.6" />
      <path d="M600 470 L800 470" fill="none" stroke="#2B2117" strokeWidth="0.9" strokeOpacity="0.4" />
      <path d="M600 470 L600 282" fill="none" stroke="#2B2117" strokeWidth="0.7" strokeOpacity="0.2" />
      <path d={ENGRAVING.up.fine} fill="none" stroke="#A07A2E" strokeWidth="0.5" strokeOpacity="0.75" />
      <path d={ENGRAVING.up.mid} fill="none" stroke="#A07A2E" strokeWidth="0.85" />
      <path d={ENGRAVING.up.major} fill="none" stroke="#7A5C28" strokeWidth="1.3" />
      {sunk ? (
        <>
          <path d="M790 470 A190 190 0 0 1 600 660" fill="none" stroke="#A07A2E" strokeWidth="1.3" />
          <path d="M796 470 A196 196 0 0 1 600 666" fill="none" stroke="#A07A2E" strokeWidth="0.7" strokeOpacity="0.6" />
          <path d="M600 470 L600 658" fill="none" stroke="#2B2117" strokeWidth="0.7" strokeOpacity="0.2" />
          <path d={ENGRAVING.down.fine} fill="none" stroke="#A07A2E" strokeWidth="0.5" strokeOpacity="0.75" />
          <path d={ENGRAVING.down.mid} fill="none" stroke="#A07A2E" strokeWidth="0.85" />
          <path d={ENGRAVING.down.major} fill="none" stroke="#7A5C28" strokeWidth="1.3" />
        </>
      ) : null}
      <g fontFamily={SERIF} fontSize="12.5" fill="#7A5C28" textAnchor="middle" style={{ fontVariantNumeric: "lining-nums tabular-nums" }}>
        <text x="736.8" y="395.5">
          30
        </text>
        <text x="679" y="337.7">
          60
        </text>
        {sunk ? (
          <>
            <text x="736.8" y="553.5">
              −30
            </text>
            <text x="679" y="611.3">
              −60
            </text>
          </>
        ) : null}
        <text x="808" y="474.5" textAnchor="start" fill="#6B5D4D">
          0°
        </text>
      </g>
      {marks.elArc ? <path d={marks.elArc} fill="none" stroke="#8F2B1E" strokeWidth="1.2" /> : null}
      {marks.elNeedle ? <path d={marks.elNeedle} fill="none" stroke="#8F2B1E" strokeWidth="1.6" strokeLinecap="round" /> : null}
      {marks.elLoz ? <path d={marks.elLoz} fill="#8F2B1E" /> : null}
      <circle cx="600" cy="470" r="4.5" fill="#F6F0E3" stroke="#A07A2E" strokeWidth="1.2" />
      <circle cx="600" cy="470" r="1.8" fill="#8F2B1E" />
    </svg>
  );
};

import React, { useMemo } from "react";
import { f1 } from "./bits";

/**
 * The board's figure: a brass limb engraved with the sixty layers, and a pointer (the alidade) resting on the
 * layer the reader is working in. Geometry ported from the board's renderVals().
 */
const CX = 300;
const CY = 318;
const ang = (i: number) => ((180 - (i * 180) / 59) * Math.PI) / 180;
const at = (i: number, r: number) => f1(CX + r * Math.cos(ang(i))) + "," + f1(CY - r * Math.sin(ang(i)));
const seg = (i: number, r1: number, r2: number) => "M" + at(i, r1) + " L" + at(i, r2) + " ";
const semi = (r: number) => "M" + (CX - r) + "," + CY + " A" + r + "," + r + " 0 0,1 " + (CX + r) + "," + CY + " ";

function limbGeometry() {
  let ticksLocal = "";
  let ticksGlobal = "";
  let radials = "";
  let fine = "";
  for (let i = 0; i < 60; i++) {
    if (i % 6 === 5) {
      ticksGlobal += seg(i, 221, 249);
      radials += seg(i, 40, 205);
    } else ticksLocal += seg(i, 229, 240);
  }
  for (let dg = 0; dg <= 180; dg += 5) {
    const a = (dg * Math.PI) / 180;
    const r2 = dg % 30 === 0 ? 247 : 244;
    fine +=
      "M" + f1(CX + 240 * Math.cos(a)) + "," + f1(CY - 240 * Math.sin(a)) + " L" + f1(CX + r2 * Math.cos(a)) + "," + f1(CY - r2 * Math.sin(a)) + " ";
  }
  const band =
    "M" + at(40, 213) + " A213,213 0 0,1 " + at(57, 213) + " M" + at(40, 208) + " A208,208 0 0,1 " + at(57, 208) + " " + seg(40, 205, 216) + seg(57, 205, 216);
  return { rete: semi(170) + semi(104), radials, fine, limb: semi(240) + semi(229), ticksLocal, ticksGlobal, band };
}

const NUMERALS: [number, number, string][] = [
  [45.3, 253, "6"],
  [80, 177, "12"],
  [137, 115, "18"],
  [210.4, 74, "24"],
  [293, 59, "30"],
  [376.2, 70, "36"],
  [451.7, 107, "42"],
  [511.9, 165, "48"],
  [550.6, 240, "54"],
];

export const Alidade: React.FC<{ layerNo: number; what: string }> = ({ layerNo, what }) => {
  const geo = useMemo(limbGeometry, []);
  const ali = useMemo(() => {
    const li = layerNo - 1;
    const aa = ang(li);
    const dx = Math.cos(aa);
    const dy = -Math.sin(aa);
    const px = Math.sin(aa);
    const py = Math.cos(aa);
    const P = (r: number, s: number) => f1(CX + r * dx + s * px) + "," + f1(CY + r * dy + s * py);
    return {
      tick: seg(li, 219, 251),
      arm: "M" + P(-22, 0) + " L" + P(250, 0),
      vanes: "M" + P(118, -8) + " L" + P(118, 8) + " M" + P(196, -8) + " L" + P(196, 8),
      lozenge: "M" + P(268, 0) + " L" + P(259, 5) + " L" + P(250, 0) + " L" + P(259, -5) + " Z",
    };
  }, [layerNo]);
  const star = layerNo % 6 === 0 ? " ✦" : "";

  return (
    <figure style={{ flex: "1 1 500px", minWidth: 0, margin: 0, display: "flex", flexDirection: "column", alignItems: "center", gap: 10 }}>
      <svg
        viewBox="0 0 600 372"
        role="img"
        aria-label={`A brass limb engraved with the sixty layers of the model; the pointer rests on layer ${layerNo}.`}
        style={{ width: "100%", maxWidth: 580, height: "auto", display: "block" }}
      >
        <path d={geo.rete} fill="none" stroke="#D9CFBD" strokeWidth="0.9" />
        <path d={geo.radials} fill="none" stroke="#E3DACA" strokeWidth="0.9" />
        <path d={geo.fine} fill="none" stroke="#C9A85F" strokeWidth="0.7" />
        <path d={geo.limb} fill="none" stroke="#A07A2E" strokeWidth="1.2" />
        <path d={geo.ticksLocal} fill="none" stroke="#2B2117" strokeOpacity="0.4" strokeWidth="0.8" />
        <path d={geo.ticksGlobal} fill="none" stroke="#A07A2E" strokeWidth="1.4" />
        <path d={geo.band} fill="none" stroke="#A07A2E" strokeWidth="1" />
        <line x1="52" y1="318" x2="548" y2="318" stroke="#2B2117" strokeWidth="1" />
        <path d="M52 312 L58 318 L52 324 L46 318 Z M548 312 L554 318 L548 324 L542 318 Z" fill="#A07A2E" />
        <g
          fontFamily="EB Garamond, Garamond, serif"
          fontSize="14"
          fill="#7A5C28"
          textAnchor="middle"
          style={{ fontVariantNumeric: "lining-nums tabular-nums" }}
        >
          <text x="36" y="323" fill="#6B5D4D">
            1
          </text>
          {NUMERALS.map(([x, y, t]) => (
            <text key={t} x={x} y={y}>
              {t}
            </text>
          ))}
          <text x="564" y="323">
            60
          </text>
          <text x="300" y="352" fontSize="12" letterSpacing="3" fill="#6B5D4D">
            LAYER
          </text>
        </g>
        <path d={ali.tick} fill="none" stroke="#8F2B1E" strokeWidth="2" />
        <path d={ali.arm} fill="none" stroke="#8F2B1E" strokeWidth="1.6" strokeLinecap="round" />
        <path d={ali.vanes} fill="none" stroke="#8F2B1E" strokeWidth="1.2" />
        <path d={ali.lozenge} fill="#A07A2E" stroke="#8F2B1E" strokeWidth="1.2" />
        <circle cx="300" cy="318" r="10" fill="#F6F0E3" stroke="#A07A2E" strokeWidth="1.2" />
        <circle cx="300" cy="318" r="3.2" fill="#8F2B1E" />
      </svg>
      <figcaption
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 4,
          textAlign: "center",
          fontSize: 15,
          color: "var(--muted)",
          maxWidth: 560,
        }}
      >
        <span style={{ fontSize: 19, color: "var(--ink)" }}>
          the pointer rests on <span className="num">layer {layerNo}</span>
          {star} · <span style={{ fontStyle: "italic" }}>{what}</span>
        </span>
        <span>
          gilt ticks: every sixth layer, which looks back over the whole conversation ✦ · double arc: layers 41–58, where neuron words
          held up under test
        </span>
      </figcaption>
    </figure>
  );
};

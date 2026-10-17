/**
 * The strength dial and the recorded figures of the Nudge page, ported from the approved canvas
 * (Nudge.dc.html, renderVals). The geometry is the canvas's, number for number.
 *
 * Sources (all real, from the repo):
 * - Default concepts six/eight: backend/server.py:102-103.
 * - Old default layers 0-based [29,35,41,47,53] = layers 30,36,42,48,54, scale 650: backend/server.py:104-105.
 * - Old screen: layers [35,41,47,53,59] = layers 36..60, scale 280; slider 50..800 step 10 (ReticleHUD.tsx:627-634).
 * - "Steer Pole" (ReticleHUD.tsx:536-551) sets pole_layer (0-based), pole_neuron, pole_polarity +/-1;
 *   server.py:156-169 pushes pol * (W_down[:, i] * post_ffn_norm[l]) at that one layer, AddSpec(normalize=True).
 * - Default neuron: 15491 at repo idx 53 = layer 54, act +4.34375 (positive face), 5th loudest;
 *   E6_tool_audit_02_sweep_demo.json "prepend_bos=True" pole_neurons["53"][4].
 * - Stream strength ||h|| (median non-BOS over six questions) and 650/||h||:
 *   experiments/results/V7_tool_bugs_02_steer_p0.json -> resid_norms[29|35|41|47|53].median_nonBOS / ratio_650_over_median.
 * - Zone edges: V7_tool_bugs_NOTES.md bug 1 (fix: strength <= 0.01 of ||h||; corrected nudge fluent at 0.8-2.3% of ||h||;
 *   at 6.5 = 2.4-7.4% of ||h||, 4 of 6 degenerate; at 20, 6 of 6). D_FF = 21504: backend/theodolite_math.py:26.
 */

/** Median non-BOS ‖h‖ per layer (layer numbers 1-60), rounded as the canvas prints it in the table. */
export const H: Record<number, number> = { 30: 237, 36: 269, 42: 175, 48: 142, 54: 88 };
/** The same medians to one decimal: the reference for the length sent. */
export const HP: Record<number, number> = { 30: 237.1, 36: 268.7, 42: 174.9, 48: 142.4, 54: 88.3 };
/** 650 ÷ median ‖h‖, as the canvas prints it. */
export const RATIO_650: Record<number, string> = { 30: "2.7", 36: "2.4", 42: "3.7", 48: "4.6", 54: "7.4" };

/** The apparatus table: V7_tool_bugs_02_steer_p0.json resid_norms (repo index 29, 35, 41, 47, 53). */
export const STREAM_TABLE: { layer: number; median: string; p10: string; p90: string; bos: string; ratio: string }[] = [
  { layer: 30, median: "237.1", p10: "176.9", p90: "267.9", bos: "129.0", ratio: "2.74" },
  { layer: 36, median: "268.7", p10: "223.4", p90: "312.1", bos: "213.3", ratio: "2.42" },
  { layer: 42, median: "174.9", p10: "149.6", p90: "222.8", bos: "177.4", ratio: "3.72" },
  { layer: 48, median: "142.4", p10: "120.2", p90: "181.0", bos: "168.4", ratio: "4.56" },
  { layer: 54, median: "88.3", p10: "71.4", p90: "111.8", bos: "126.3", ratio: "7.36" },
];

/** The plain answer to the spider question, greedy: V7_tool_bugs_08_diag.json gens[...].hf_greedy
 * (and V7_tool_bugs_02_steer_p0 gens[0]). */
export const RECORDED_PLAIN_SPIDER = " 8.\nThe number of legs on the animal that";

export const GENTLE_MAX = 0.01;
export const EDGE = 0.024;
/** The ten global-attention layers, every sixth, counted 1-60. */
export const GLOBAL = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60];

export type Zone = "gentle" | "strong" | "collapse";

/** s = 10^(v/100 − 3), v from 0 to 400. */
export const strengthOf = (v: number) => Math.pow(10, -3 + v / 100);
export const zoneOf = (s: number): Zone => (s <= GENTLE_MAX * 1.0001 ? "gentle" : s <= EDGE ? "strong" : "collapse");
export const clampV = (x: number) => Math.max(0, Math.min(400, x));
/** The multiplier as the canvas prints it. */
export const fmt = (x: number) => (x < 0.1 ? x.toFixed(3) : x < 1 ? x.toFixed(2) : x.toFixed(1));
/** The absolute length as the canvas prints it (and as it is sent). */
export const scaleFmt = (x: number) => (x < 1 ? x.toFixed(3) : x < 100 ? x.toFixed(2) : x.toFixed(1));

/* --------------------------------------------------------------- geometry */

const CX = 500;
const CY = 560;
const A0 = 152;
const SPAN = 124;
const f = (n: number) => Math.round(n * 10) / 10;
export const tOf = (x: number) => (Math.log(x) / Math.LN10 + 3) / 4;
const phi = (t: number) => A0 - SPAN * t;
type Pt = [number, number];
const P = (R: number, t: number): Pt => {
  const a = (phi(t) * Math.PI) / 180;
  return [f(CX + R * Math.cos(a)), f(CY - R * Math.sin(a))];
};
const pt = (p: Pt) => p[0] + "," + p[1];
const arc = (R: number, ta: number, tb: number) => "M" + pt(P(R, ta)) + " A" + R + "," + R + " 0 0,1 " + pt(P(R, tb)) + " ";
const sector = (ri: number, ro: number, ta: number, tb: number) =>
  "M" + pt(P(ro, ta)) + " A" + ro + "," + ro + " 0 0,1 " + pt(P(ro, tb)) + " L" + pt(P(ri, tb)) + " A" + ri + "," + ri + " 0 0,0 " + pt(P(ri, ta)) + " Z";
const radial = (r1: number, r2: number, t: number) => "M" + pt(P(r1, t)) + " L" + pt(P(r2, t)) + " ";

export interface Lab {
  x: number;
  y: number;
  tr: string;
}
const lab = (R: number, t: number): Lab => {
  const p = P(R, t);
  const rot = f(90 - phi(t));
  return { x: p[0], y: p[1], tr: "rotate(" + rot + " " + p[0] + " " + p[1] + ")" };
};
const dec = (t: number) => {
  const p = P(498 + 26 * Math.abs(Math.cos((phi(t) * Math.PI) / 180)), t);
  return { x: p[0], y: f(p[1] + 6) };
};
const bracket = (R: number, ta: number, tb: number) => arc(R, ta, tb) + radial(R - 6, R + 6, ta) + radial(R - 6, R + 6, tb);

function buildDial() {
  const tG = tOf(GENTLE_MAX);
  const tE = tOf(EDGE);
  let minor = "";
  let major = "";
  for (let d = -3; d <= 0; d++) {
    for (let m = 1; m <= 9; m++) {
      const t = tOf(m * Math.pow(10, d));
      if (m === 1) major += radial(460, 481, t);
      else minor += radial(460, m === 5 ? 473 : 467, t);
    }
  }
  major += radial(460, 481, 1);
  let fine = "";
  for (let k = 0; k <= 62; k++) fine += radial(420, 413, k / 62);
  let innerTicks = "";
  for (let k = 0; k <= 24; k++) innerTicks += radial(270, k % 6 === 0 ? 280 : 275, k / 24);
  let hatch = "";
  for (let t = tE + 0.009; t <= 1.0001; t += 0.0065) hatch += "M" + pt(P(425, t)) + " L" + pt(P(455, t - 0.009)) + " ";
  return {
    innerRing: arc(270, 0, 1),
    innerTicks,
    strongBand: sector(425, 455, tG, tE),
    collapseBand: sector(425, 455, tE, 1),
    hatch,
    bandLines: arc(425, 0, tE) + arc(455, 0, tE),
    limbOuter: arc(460, 0, 1),
    limbInner: arc(420, 0, 1),
    fine,
    minor,
    major,
    dividers: radial(420, 460, tG) + radial(420, 460, tE),
    dA: dec(0),
    dB: dec(0.25),
    dC: dec(0.5),
    dD: dec(0.75),
    dE: dec(1),
    zG: lab(393, tG / 2),
    zS: lab(393, (tG + tE) / 2),
    zC: lab(393, (tE + 1) / 2),
    brFluent: bracket(374, tOf(0.008), tOf(0.023)),
    brOld: bracket(374, tOf(2.4), tOf(7.4)) + bracket(333, tOf(1.04), tOf(3.17)),
    lF: lab(349, (tOf(0.008) + tOf(0.023)) / 2),
    lO: lab(349, (tOf(2.4) + tOf(7.4)) / 2),
    lS: lab(304, (tOf(1.04) + tOf(3.17)) / 2),
  };
}

/** Everything on the dial that does not move with the needle. */
export const DIAL = buildDial();

/** The needle, its lozenge and its tip, for a dial value v (0-400). */
export function needleAt(v: number) {
  const tv = v / 400;
  const aN = (phi(tv) * Math.PI) / 180;
  const ux = Math.cos(aN);
  const uy = -Math.sin(aN);
  const c0 = P(290, tv);
  const lz = (k1: number, k2: number) => f(c0[0] + ux * k1 - uy * k2) + "," + f(c0[1] + uy * k1 + ux * k2);
  const loz = "M" + lz(13, 0) + " L" + lz(0, 6) + " L" + lz(-13, 0) + " L" + lz(0, -6) + " Z";
  const tip = P(487, tv);
  return { needle: radial(290, 484, tv), loz, tip: { x: tip[0], y: tip[1] } };
}

/** The hatching of the collapse part of the strip under the slider. */
export const STRIP_HATCH = (() => {
  let s = "";
  for (let x = 349; x + 7 <= 1000; x += 9) s += "M" + x + ",14 L" + (x + 7) + ",0 ";
  return s;
})();

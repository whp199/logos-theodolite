/**
 * SVG geometry for the Landmarks page, ported from the canvas's renderVals():
 * the engraved survey frame (an oblique view, the plate turned 28 degrees), the bearing and elevation
 * charts across the sixty layers, and the little rank charts.
 */

export const RAD = Math.PI / 180;
export const f1 = (n: number) => Math.round(n * 10) / 10;

type Pt = [number, number];
const pt = (p: Pt) => p[0] + "," + p[1];

export const loz = (p: Pt, r: number) =>
  "M" + p[0] + "," + f1(p[1] - r) + " L" + f1(p[0] + r * 0.7) + "," + p[1] + " L" + p[0] + "," + f1(p[1] + r) +
  " L" + f1(p[0] - r * 0.7) + "," + p[1] + " Z ";
export const ring = (p: Pt, r: number) =>
  "M" + f1(p[0] - r) + "," + p[1] + " a" + r + "," + r + " 0 1,0 " + 2 * r + ",0 a" + r + "," + r + " 0 1,0 " + -2 * r + ",0 ";

/* ------------------------------------------------------------ the frame */

const CX = 320;
const CY = 300;
const R = 240;
const RY = 78;
const H = 210;
const CP = Math.cos(28 * RAD);
const SP = Math.sin(28 * RAD);
/** Project a point of the unit frame onto the plate. */
export const P = (x: number, y: number, z: number): Pt => [
  f1(CX + R * (x * CP - y * SP)),
  f1(CY + RY * (x * SP + y * CP) - H * z),
];
const O: Pt = [CX, CY];
const poly = (fn: (t: number) => Pt, a0: number, a1: number, n: number) => {
  let s = "";
  for (let k = 0; k <= n; k++) s += (k ? " L" : "M") + pt(fn(a0 + ((a1 - a0) * k) / n));
  return s;
};

/** The fixed engraving: meridians, the 45° parallel, back axes, rim ticks, axes and their lozenges. */
export const PLATE = (() => {
  let ticks = "";
  for (let d = 0; d < 360; d += 10) {
    const a = d * RAD;
    const k = d % 30 === 0 ? 0.9 : 0.95;
    ticks += "M" + pt(P(Math.cos(a), Math.sin(a), 0)) + " L" + pt(P(k * Math.cos(a), k * Math.sin(a), 0)) + " ";
  }
  return {
    ticks,
    merA: poly((t) => P(Math.cos(t * RAD), 0, Math.sin(t * RAD)), 0, 360, 90) + " Z",
    merB: poly((t) => P(0, Math.cos(t * RAD), Math.sin(t * RAD)), 0, 360, 90) + " Z",
    par: poly((t) => P(0.7071 * Math.cos(t * RAD), 0.7071 * Math.sin(t * RAD), 0.7071), 0, 360, 72) + " Z",
    back: "M" + pt(O) + " L" + pt(P(-1, 0, 0)) + " M" + pt(O) + " L" + pt(P(0, -1, 0)) + " M" + pt(O) + " L" + pt(P(0, 0, -1)),
    axes: "M" + pt(O) + " L" + pt(P(1.1, 0, 0)) + " M" + pt(O) + " L" + pt(P(0, 1.2, 0)) + " M" + pt(O) + " L" + pt(P(0, 0, 1.06)),
    tips: loz(P(1.16, 0, 0), 7) + loz(P(0, 1.28, 0), 7) + loz(P(0, 0, 1.12), 7),
  };
})();

/**
 * Dashed rays for a frame squared in order (Gram–Schmidt): where the second and third words truly point before
 * squaring, from the cosines c12, c13, c23 of the rows the frame was squared from (raw, or final-norm weighted). The
 * first word lies along axis I.
 */
export function ghostGeometry(c12: number, c13: number, c23: number) {
  const s2 = Math.sqrt(1 - c12 * c12);
  const y3 = (c23 - c12 * c13) / s2;
  const z3 = Math.sqrt(Math.max(0, 1 - c13 * c13 - y3 * y3));
  const u2 = P(c12, s2, 0);
  const u3 = P(c13, y3, z3);
  const th = Math.acos(c12);
  const mid = P(0.7 * Math.cos(th * 0.75), 0.7 * Math.sin(th * 0.75), 0);
  return {
    rays: "M" + pt(O) + " L" + pt(u2) + " M" + pt(O) + " L" + pt(u3),
    arc: poly((t) => P(0.62 * Math.cos(t), 0.62 * Math.sin(t), 0), 0, th, 24),
    tips: loz(u2, 4.5) + loz(u3, 4.5),
    bx: f1(u2[0] + 8),
    by: f1(u2[1] + 20),
    cx: f1(u3[0] + 10),
    cy: f1(u3[1] - 6),
    ax: mid[0],
    ay: f1(mid[1] + 5),
    angle: Math.round(th / RAD),
  };
}

/** The eigenvalues and eigenvectors (as columns of V) of a symmetric 3×3 matrix, by cyclic Jacobi rotations. */
function jacobi3(A: number[][]): { values: number[]; V: number[][] } {
  const a = A.map((r) => r.slice());
  const V = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  for (let sweep = 0; sweep < 50; sweep++) {
    const off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
    if (off < 1e-24) break;
    for (const [p, q] of [
      [0, 1],
      [0, 2],
      [1, 2],
    ]) {
      if (Math.abs(a[p][q]) < 1e-300) continue;
      // the rotation that zeroes a[p][q]: tan 2θ = 2 a_pq / (a_qq − a_pp)
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1);
      const sn = t * c;
      for (let k = 0; k < 3; k++) {
        const akp = a[k][p];
        const akq = a[k][q];
        a[k][p] = c * akp - sn * akq;
        a[k][q] = sn * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = a[p][k];
        const aqk = a[q][k];
        a[p][k] = c * apk - sn * aqk;
        a[q][k] = sn * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = V[k][p];
        const vkq = V[k][q];
        V[k][p] = c * vkp - sn * vkq;
        V[k][q] = sn * vkp + c * vkq;
      }
    }
  }
  return { values: [a[0][0], a[1][1], a[2][2]], V };
}

/**
 * Dashed rays for a frame squared all at once (Löwdin): where each of the three words truly points. The frame is
 * E = (V Vᵀ)^−½ V on the unit rows V, so word m stands at E v_m, column m of G^½, where G is the Gram of the unit rows
 * (G^½ = Q diag(√λ) Qᵀ). Each column is of unit length and lies a little off its own axis; from the cosines c12, c13,
 * c23 of the rows the frame was squared from (raw, or final-norm weighted).
 */
export function lowdinGhost(c12: number, c13: number, c23: number) {
  const G = [
    [1, c12, c13],
    [c12, 1, c23],
    [c13, c23, 1],
  ];
  const { values, V } = jacobi3(G);
  const root = values.map((v) => Math.sqrt(Math.max(0, v)));
  const half = (i: number, j: number) => V[i][0] * root[0] * V[j][0] + V[i][1] * root[1] * V[j][1] + V[i][2] * root[2] * V[j][2];
  const cols = [0, 1, 2].map((m) => [half(0, m), half(1, m), half(2, m)] as [number, number, number]);
  const u = cols.map((c) => P(c[0], c[1], c[2]));
  // the arc between the first two words, at radius 0.62, along the great circle joining them
  const th = Math.acos(Math.max(-1, Math.min(1, c12)));
  const along = (r: number, t: number): Pt => {
    const s0 = Math.sin(th);
    const wa = s0 > 1e-9 ? Math.sin((1 - t) * th) / s0 : 1 - t;
    const wb = s0 > 1e-9 ? Math.sin(t * th) / s0 : t;
    const [a, b] = [cols[0], cols[1]];
    return P(r * (wa * a[0] + wb * b[0]), r * (wa * a[1] + wb * b[1]), r * (wa * a[2] + wb * b[2]));
  };
  const mid = along(0.7, 0.5);
  return {
    rays: u.map((p) => "M" + pt(O) + " L" + pt(p)).join(" "),
    arc: poly((t) => along(0.62, t), 0, 1, 24),
    tips: u.map((p) => loz(p, 4.5)).join(""),
    /** label positions for the three words' rays */
    labels: u.map((p, m) => [f1(p[0] + (m === 2 ? 10 : 8)), f1(p[1] + (m === 2 ? -6 : 20))] as Pt),
    ax: mid[0],
    ay: f1(mid[1] + 5),
    angle: Math.round(th / RAD),
    /** each word's bearing and elevation in the frame, and how far it stands from its own axis, in degrees */
    words: cols.map((c, m) => ({
      az: Math.atan2(c[1], c[0]) / RAD,
      el: Math.asin(Math.max(-1, Math.min(1, c[2]))) / RAD,
      off: Math.acos(Math.max(-1, Math.min(1, c[m]))) / RAD,
    })),
  };
}

const RHO = 0.85;

/** The thought at one layer: the red point, its ray, its drop to the plate, and the two arcs. */
export function readingGeometry(az: number, el: number) {
  const ca = Math.cos(az * RAD);
  const sa = Math.sin(az * RAD);
  const ce = Math.cos(el * RAD);
  const se = Math.sin(el * RAD);
  const rp = P(RHO * ce * ca, RHO * ce * sa, RHO * se);
  const foot = P(RHO * ce * ca, RHO * ce * sa, 0);
  const bMid = P(0.2 * Math.cos((az / 2) * RAD), 0.2 * Math.sin((az / 2) * RAD), 0);
  const eMid = P(
    0.56 * Math.cos((el / 2) * RAD) * ca,
    0.56 * Math.cos((el / 2) * RAD) * sa,
    0.56 * Math.sin((el / 2) * RAD),
  );
  return {
    x: rp[0],
    y: rp[1],
    lx: f1(rp[0] + 18),
    ly: f1(rp[1] - 14),
    ray: "M" + pt(O) + " L" + pt(rp),
    drop: "M" + pt(rp) + " L" + pt(foot) + " M" + pt(O) + " L" + pt(foot),
    bArc: poly((t) => P(0.34 * Math.cos(t * RAD), 0.34 * Math.sin(t * RAD), 0), 0, az, 30),
    eArc: poly((t) => P(0.5 * Math.cos(t * RAD) * ca, 0.5 * Math.cos(t * RAD) * sa, 0.5 * Math.sin(t * RAD)), 0, el, 24),
    index: "M" + pt(P(0.9 * ca, 0.9 * sa, 0)) + " L" + pt(P(1.06 * ca, 1.06 * sa, 0)),
    bx: bMid[0],
    by: f1(bMid[1] + 18),
    ex: f1(eMid[0] + 12),
    ey: eMid[1],
    /** components of the unit direction, for "leans most toward" */
    comps: [ce * ca, ce * sa, se] as [number, number, number],
  };
}

/** Faint rings for every layer that has a bearing and an elevation. */
export function allDots(angles: { az: number | null; el: number | null }[]) {
  let s = "";
  for (const { az, el } of angles) {
    if (az == null || el == null) continue;
    const a = az * RAD;
    const e = el * RAD;
    s += ring(P(RHO * Math.cos(e) * Math.cos(a), RHO * Math.cos(e) * Math.sin(a), RHO * Math.sin(e)), 1.8);
  }
  return s;
}

/* ------------------------------------------- bearing through the layers */

const X0 = 80;
const XW = 1060;
export const cxL = (l: number) => f1(X0 + (l * XW) / 59);
export const yB = (a: number) => f1(150 - (a * 100) / 180);
export const yE = (e: number) => f1(350 - (e * 50) / 90);

/** The two engraved lines; the bearing breaks where it crosses ±180°, and at any layer not recorded. */
export function chartGeometry(az: (number | null)[], el: (number | null)[]) {
  let bPath = "";
  let prev: number | null = null;
  for (let l = 0; l < az.length; l++) {
    const a2 = az[l];
    if (a2 == null) {
      prev = null;
      continue;
    }
    if (prev == null) {
      bPath += "M" + cxL(l) + "," + yB(a2);
    } else if (Math.abs(a2 - prev) > 180) {
      const d = ((a2 - prev + 540) % 360) - 180;
      const edge = d > 0 ? 180 : -180;
      const xm = f1(X0 + ((l - 1 + (edge - prev) / d) * XW) / 59);
      bPath += " L" + xm + "," + yB(edge) + " M" + xm + "," + yB(-edge) + " L" + cxL(l) + "," + yB(a2);
    } else {
      bPath += " L" + cxL(l) + "," + yB(a2);
    }
    prev = a2;
  }
  let ePath = "";
  let open = false;
  el.forEach((e, l) => {
    if (e == null) {
      open = false;
      return;
    }
    ePath += (open ? " L" : "M") + cxL(l) + "," + yE(e);
    open = true;
  });
  const bDots = az.map((a, l) => (a == null ? "" : ring([cxL(l), yB(a)], 2.2))).join("");
  const eDots = el.map((e, l) => (e == null ? "" : loz([cxL(l), yE(e)], 3))).join("");
  let globalD = "";
  for (let l = 5; l < 60; l += 6) globalD += "M" + cxL(l) + ",40 V410 ";
  return { bPath, ePath, bDots, eDots, globalD };
}

/* ---------------------------------------------- how close to being said */

export const sx = (l: number) => f1(48 + (l * 404) / 59);
export const syRank = (r: number, vocab: number) => f1(4 + (Math.log10(r) / Math.log10(vocab)) * 48);

export function sparkPath(ranks: (number | null)[], vocab: number) {
  let s = "";
  let open = false;
  ranks.forEach((r, i) => {
    if (r == null || r < 1) {
      open = false;
      return;
    }
    s += (open ? " L" : "M") + sx(i) + "," + syRank(r, vocab);
    open = true;
  });
  return s;
}

/**
 * The instrument's drawing: sixty stacked rings, one per layer, with the path of the model's thought
 * winding upward through them. Geometry follows the approved canvas (Main.dc.html): the path is drawn by
 * direction only (bearing θ, elevation φ); its radius is schematic. A turn (yaw) adds to every bearing,
 * a tilt changes how open the rings are seen.
 */
import type { LayerReading, ThreadReading } from "../../store/useTheodoliteStore";

export const CX = 320;
export const R = 200;
export const BASE_TILT = 52 / 200; // the canvas's ring ratio RY / R
export const LIMB_Y = 606;
export const ringY = (idx: number) => 570 - 7 * idx;
const rad = Math.PI / 180;
const f = (n: number) => Math.round(n * 10) / 10;

export interface View {
  yaw: number; // degrees added to every bearing
  tilt: number; // ring ratio RY / R
}

export function ellipse(cx: number, cy: number, rx: number, ry: number) {
  return `M${f(cx - rx)},${f(cy)} a${f(rx)},${f(ry)} 0 1,0 ${f(2 * rx)},0 a${f(rx)},${f(ry)} 0 1,0 ${f(-2 * rx)},0 `;
}
export function dot(p: [number, number], r: number) {
  return `M${f(p[0] - r)},${p[1]} a${r},${r} 0 1,0 ${f(2 * r)},0 a${r},${r} 0 1,0 ${f(-2 * r)},0 `;
}
export function lozenge(p: [number, number], s: number) {
  return `M${p[0]},${f(p[1] - s)} L${f(p[0] + s * 0.7)},${p[1]} L${p[0]},${f(p[1] + s)} L${f(p[0] - s * 0.7)},${p[1]} Z `;
}

/** Where each layer's reading sits on the drawing; null when bearing/elevation were not recorded. */
export function pathPoints(layers: LayerReading[], v: View): ([number, number] | null)[] {
  const lift = (14 * Math.sqrt(Math.max(0, 1 - v.tilt * v.tilt))) / Math.sqrt(1 - BASE_TILT * BASE_TILT);
  return layers.map((L) => {
    if (L.az == null || L.el == null) return null;
    const az = (L.az + v.yaw) * rad;
    const el = L.el * rad;
    const r = R * (0.3 + 0.7 * Math.cos(el));
    return [f(CX + r * Math.cos(az)), f(ringY(L.idx) + r * Math.sin(az) * v.tilt - Math.sin(el) * lift)];
  });
}

export function onRing(idx: number, deg: number, v: View): [number, number] {
  const a = (deg + v.yaw) * rad;
  return [f(CX + R * Math.cos(a)), f(ringY(idx) + R * v.tilt * Math.sin(a))];
}

export function rings(v: View) {
  let local = "";
  let global = "";
  for (let l = 0; l < 60; l++) {
    const d = ellipse(CX, ringY(l), R, R * v.tilt);
    if (l % 6 === 5) global += d;
    else local += d;
  }
  return { local, global };
}

export function limb(v: View) {
  let ticks = "";
  // the limb is drawn a touch flatter than the rings (64 / 246 against 52 / 200), as on the canvas
  const k = (64 / 246) * (v.tilt / BASE_TILT);
  for (let d = 0; d < 360; d += 10) {
    const a = (d + v.yaw) * rad;
    const ri = d % 30 === 0 ? 0.86 : 0.93;
    ticks += `M${f(CX + 246 * Math.cos(a))},${f(LIMB_Y + 246 * k * Math.sin(a))} L${f(CX + 246 * ri * Math.cos(a))},${f(LIMB_Y + 246 * k * ri * Math.sin(a))} `;
  }
  return {
    ticks,
    outer: { rx: 246, ry: f(246 * k) },
    inner: { rx: 226, ry: f(226 * k) },
  };
}

/** A landmark's label on the limb: bearing 0° for the first word, 90° for the second (as the canvas). */
export function limbLabel(deg: number, v: View) {
  const a = (deg + v.yaw) * rad;
  const k = (64 / 246) * (v.tilt / BASE_TILT);
  const c = Math.cos(a);
  return {
    x: f(CX + 260 * c),
    y: f(LIMB_Y + (268 * k + 14) * Math.sin(a) + 5),
    anchor: (c > 0.3 ? "start" : c < -0.3 ? "end" : "middle") as "start" | "end" | "middle",
  };
}

export function threadPaths(threads: ThreadReading[], v: View) {
  let pos = "";
  let neg = "";
  let ends = "";
  for (const t of threads) {
    // real layer pairs and signs; the angle on each ring is schematic (golden angle by neuron number)
    const a = onRing(t.src.idx, (t.src.neuron * 137.508) % 360, v);
    const b = onRing(t.dst.idx, (t.dst.neuron * 137.508) % 360, v);
    const mx = (a[0] + b[0]) / 2;
    const my = (a[1] + b[1]) / 2;
    const c = `M${a[0]},${a[1]} Q${f(CX + (mx - CX) * 1.35)},${f(my - 16)} ${b[0]},${b[1]} `;
    if (t.c > 0) pos += c;
    else neg += c;
    ends += dot(a, 2) + dot(b, 2);
  }
  return { pos, neg, ends };
}

export function neuronMarks(L: LayerReading, v: View) {
  let pos = "";
  let neg = "";
  const maxA = Math.max(1e-9, ...L.poles.map((p) => Math.abs(p.act)));
  L.poles.forEach((p, k) => {
    const at = onRing(L.idx, k * 24 - 78, v);
    const sz = f(3 + (7 * Math.abs(p.act)) / maxA);
    if (p.act > 0) pos += lozenge(at, sz);
    else neg += lozenge(at, sz);
  });
  return { pos, neg };
}

export function splitArcs(p: [number, number], ffnShare: number) {
  const arcPt = (deg: number) => `${f(p[0] + 22 * Math.cos(deg * rad))},${f(p[1] + 22 * Math.sin(deg * rad))}`;
  const gather = 1 - ffnShare;
  const gEnd = -90 + 360 * gather;
  return {
    gather: `M${arcPt(-90)} A22,22 0 ${gather > 0.5 ? 1 : 0},1 ${arcPt(gEnd)}`,
    recall: `M${arcPt(gEnd)} A22,22 0 ${ffnShare > 0.5 ? 1 : 0},1 ${arcPt(269.9)}`,
  };
}

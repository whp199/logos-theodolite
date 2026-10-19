import React, { useEffect, useMemo, useRef, useState } from "react";
import { chosenOf, isChosenRow, type Sighting } from "../../store/useTheodoliteStore";
import { bare } from "../../lib/words";
import { lensBand } from "../../lib/lens";
import { endedWords } from "./moment";
import {
  BASE_TILT,
  CX,
  LIMB_Y,
  type View,
  dot,
  limb,
  limbLabel,
  neuronMarks,
  pathPoints,
  ringY,
  rings,
  splitArcs,
  threadPaths,
} from "./geometry";

export interface Shown {
  path: boolean;
  words: boolean;
  split: boolean;
  neurons: boolean;
  threads: boolean;
  marks: boolean;
  rings: boolean;
}

const FULL = 700;
const MIN_W = FULL / 3.5;
const AXIS_IDX = [59, 53, 47, 41, 35, 29, 23, 17, 11, 5];

export const InstrumentFigure: React.FC<{
  sighting: Sighting;
  sel: number;
  stations: number[];
  shown: Shown;
  onPick: (idx: number) => void;
}> = ({ sighting, sel, stations, shown, onPick }) => {
  const [view, setView] = useState<View>({ yaw: 0, tilt: BASE_TILT });
  const [box, setBox] = useState({ x: 0, y: 0, w: FULL });
  const svgRef = useRef<SVGSVGElement | null>(null);
  const drag = useRef<{ x: number; y: number; yaw: number; tilt: number; moved: boolean; id: number } | null>(null);

  const layers = sighting.layers;
  const L = layers[sel];
  const pts = useMemo(() => pathPoints(layers, view), [layers, view]);
  const geo = useMemo(() => {
    const r = rings(view);
    const lb = limb(view);
    const stationSet = new Set(stations);
    let pathD = "";
    let started = false;
    let dots = "";
    let stationDots = "";
    pts.forEach((p, l) => {
      if (!p) {
        started = false;
        return;
      }
      pathD += `${started ? " L" : "M"}${p[0]},${p[1]}`;
      started = true;
      if (stationSet.has(l)) stationDots += dot(p, 3.4);
      else dots += dot(p, 1.8);
    });
    const leaders = stations
      .filter((s) => pts[s])
      .map((s) => `M${pts[s]![0]},${pts[s]![1]} L542,${ringY(s)}`)
      .join(" ");
    return { r, lb, pathD, dots, stationDots, leaders, th: threadPaths(sighting.threads.slice(0, 12), view) };
  }, [pts, stations, view, sighting.threads]);
  const marks = useMemo(() => (L ? neuronMarks(L, view) : { pos: "", neg: "" }), [L, view]);
  const sp = pts[sel];
  const split = sp && L?.ffnShare != null ? splitArcs(sp, L.ffnShare) : null;

  // wheel: draw near (zoom about the pointer); at full view, scrolling outward lets the page scroll
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const onWheel = (e: WheelEvent) => {
      const ctm = svg.getScreenCTM();
      if (!ctm) return;
      setBox((b) => {
        const factor = Math.exp(-e.deltaY * 0.0015);
        const w = Math.max(MIN_W, Math.min(FULL, b.w / factor));
        if (w === b.w) return b;
        const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(ctm.inverse());
        let x = pt.x - (pt.x - b.x) * (w / b.w);
        let y = pt.y - (pt.y - b.y) * (w / b.w);
        x = Math.max(0, Math.min(FULL - w, x));
        y = Math.max(0, Math.min(FULL - w, y));
        return { x, y, w };
      });
      const atFull = box.w >= FULL && e.deltaY > 0;
      if (!atFull) e.preventDefault();
    };
    svg.addEventListener("wheel", onWheel, { passive: false });
    return () => svg.removeEventListener("wheel", onWheel);
  }, [box.w]);

  const toSvg = (clientX: number, clientY: number) => {
    const svg = svgRef.current;
    const ctm = svg?.getScreenCTM();
    if (!svg || !ctm) return null;
    return new DOMPoint(clientX, clientY).matrixTransform(ctm.inverse());
  };

  const pickNearest = (clientX: number, clientY: number) => {
    const p = toSvg(clientX, clientY);
    if (!p) return;
    let best = -1;
    let bestD = 18 * (box.w / FULL) + 6;
    pts.forEach((q, l) => {
      if (!q) return;
      const d = Math.hypot(q[0] - p.x, q[1] - p.y);
      if (d < bestD) {
        bestD = d;
        best = l;
      }
    });
    if (best < 0) {
      // otherwise the ring nearest the click, if the click falls inside the drum
      const l = Math.round((570 - p.y) / 7);
      if (l >= 0 && l < 60 && Math.abs(p.x - CX) < 210) best = l;
    }
    if (best >= 0) onPick(best);
  };

  const onPointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    if (e.button !== 0) return;
    drag.current = { x: e.clientX, y: e.clientY, yaw: view.yaw, tilt: view.tilt, moved: false, id: e.pointerId };
  };
  const onPointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const d = drag.current;
    if (!d || d.id !== e.pointerId) return;
    const dx = e.clientX - d.x;
    const dy = e.clientY - d.y;
    if (!d.moved && Math.hypot(dx, dy) < 4) return;
    if (!d.moved) {
      d.moved = true;
      (e.currentTarget as SVGSVGElement).setPointerCapture(e.pointerId);
    }
    setView({
      yaw: d.yaw - dx * 0.45,
      tilt: Math.max(0.06, Math.min(0.62, d.tilt + dy * 0.0016)),
    });
  };
  const onPointerUp = (e: React.PointerEvent<SVGSVGElement>) => {
    const d = drag.current;
    drag.current = null;
    if (d && !d.moved && d.id === e.pointerId) pickNearest(e.clientX, e.clientY);
  };
  const onKeyDown = (e: React.KeyboardEvent<SVGSVGElement>) => {
    const k = e.key;
    if (k === "ArrowLeft") setView((v) => ({ ...v, yaw: v.yaw + 10 }));
    else if (k === "ArrowRight") setView((v) => ({ ...v, yaw: v.yaw - 10 }));
    else if (k === "ArrowUp") setView((v) => ({ ...v, tilt: Math.min(0.62, v.tilt + 0.04) }));
    else if (k === "ArrowDown") setView((v) => ({ ...v, tilt: Math.max(0.06, v.tilt - 0.04) }));
    else if (k === "+" || k === "=") setBox((b) => zoomCentre(b, 1.25));
    else if (k === "-" || k === "_") setBox((b) => zoomCentre(b, 1 / 1.25));
    else if (k === "0") reset();
    else return;
    e.preventDefault();
  };
  const reset = () => {
    setView({ yaw: 0, tilt: BASE_TILT });
    setBox({ x: 0, y: 0, w: FULL });
  };
  const turned = view.yaw !== 0 || view.tilt !== BASE_TILT || box.w !== FULL;

  const [first, second, third] = sighting.landmarks;
  const lab0 = limbLabel(0, view);
  const lab90 = limbLabel(90, view);
  const wordColor = (s: number) => (s === sel ? "#8F2B1E" : "#2B2117");
  const topOf = (idx: number) => layers[idx]?.lensTop?.[0]?.token ?? null;

  return (
    <figure style={{ margin: 0, position: "relative" }}>
      <svg
        ref={svgRef}
        viewBox={`${box.x} ${box.y} ${box.w} ${box.w}`}
        role="img"
        tabIndex={0}
        aria-label={`Sixty stacked rings, one per layer, with the word’s path as the lens reads it winding upward through them; the lens’s words below layer 21, where the study could validate none, are drawn faint. Layer ${sel + 1} is ringed in red. Drag or use the arrow keys to turn and tilt; scroll or press plus and minus to draw near; press 0 to reset.`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={() => (drag.current = null)}
        onKeyDown={onKeyDown}
        style={{
          width: "100%",
          maxWidth: 760,
          height: "auto",
          display: "block",
          margin: "0 auto",
          cursor: "grab",
          touchAction: "pan-y",
          userSelect: "none",
        }}
      >
        <g display={shown.rings ? "inline" : "none"}>
          <path d={geo.r.local} fill="none" stroke="#2B2117" strokeOpacity="0.11" strokeWidth="0.8" />
          <path d={geo.r.global} fill="none" stroke="#A07A2E" strokeOpacity="0.75" strokeWidth="1.1" />
          <g fontFamily="EB Garamond, Garamond, serif" fontSize="13" fill="#7A5C28" textAnchor="end" style={{ fontVariantNumeric: "lining-nums tabular-nums" }}>
            <text x="104" y="140" fontSize="12" letterSpacing="2" fill="#6B5D4D">
              LAYER
            </text>
            {AXIS_IDX.map((i) => (
              <text key={i} x="108" y={ringY(i) + 4}>
                {i + 1}
              </text>
            ))}
            <text x="108" y="574" fill="#6B5D4D">
              1
            </text>
          </g>
        </g>
        <ellipse cx={CX} cy={LIMB_Y} rx={geo.lb.outer.rx} ry={geo.lb.outer.ry} fill="none" stroke="#A07A2E" strokeWidth="1.2" />
        <ellipse cx={CX} cy={LIMB_Y} rx={geo.lb.inner.rx} ry={geo.lb.inner.ry} fill="none" stroke="#A07A2E" strokeOpacity="0.55" strokeWidth="0.8" />
        <path d={geo.lb.ticks} stroke="#A07A2E" strokeWidth="0.9" fill="none" />
        <line x1={CX} y1="642" x2={CX} y2="92" stroke="#2B2117" strokeOpacity="0.35" strokeWidth="1" />
        <path d="M320 74 L327 86 L320 98 L313 86 Z" fill="#A07A2E" />
        <g display={shown.marks ? "inline" : "none"} fontFamily="EB Garamond, Garamond, serif" fontStyle="italic" fontSize="17" fill="#2E4A7D">
          <text x={lab0.x} y={lab0.y} textAnchor={lab0.anchor}>
            {first} <tspan fontStyle="normal" fontSize="12" fill="#6B5D4D">0°</tspan>
          </text>
          <text x={lab90.x} y={lab90.y} textAnchor={lab90.anchor}>
            {second} <tspan fontStyle="normal" fontSize="12" fill="#6B5D4D">90°</tspan>
          </text>
          <text x="333" y="80">
            {third} <tspan fontStyle="normal" fontSize="12" fill="#6B5D4D">↑ up</tspan>
          </text>
        </g>
        <g display={shown.threads ? "inline" : "none"}>
          <path d={geo.th.pos} fill="none" stroke="#A07A2E" strokeWidth="1.3" />
          <path d={geo.th.neg} fill="none" stroke="#2B2117" strokeOpacity="0.7" strokeWidth="1.1" strokeDasharray="3 3" />
          <path d={geo.th.ends} fill="#F6F0E3" stroke="#2B2117" strokeWidth="0.9" />
        </g>
        <ellipse cx={CX} cy={ringY(sel)} rx="200" ry={200 * view.tilt} fill="#A07A2E" fillOpacity="0.07" stroke="#8F2B1E" strokeWidth="1.5" />
        <g display={shown.neurons ? "inline" : "none"}>
          <path d={marks.pos} fill="#A07A2E" stroke="#7A5C28" strokeWidth="0.8" />
          <path d={marks.neg} fill="#F6F0E3" stroke="#A07A2E" strokeWidth="1.2" />
        </g>
        <g display={shown.path ? "inline" : "none"}>
          <path d={geo.pathD} fill="none" stroke="#2B2117" strokeWidth="1.6" strokeLinejoin="round" strokeLinecap="round" />
          <path d={geo.dots} fill="#2B2117" />
          <path d={geo.stationDots} fill="#F6F0E3" stroke="#2B2117" strokeWidth="1.2" />
        </g>
        <g display={shown.words ? "inline" : "none"}>
          <path d={geo.leaders} fill="none" stroke="#7A5C28" strokeOpacity="0.55" strokeWidth="0.7" strokeDasharray="2 3" />
          <g fontFamily="EB Garamond, Garamond, serif" fontStyle="italic" fontSize="17">
            {stations.map((s) => {
              const w = topOf(s);
              if (w == null) return null;
              // below layer 21 the study validated nothing the lens reads (lib/lens.ts, lensBand): drawn faint
              const faint = lensBand(s + 1) === "untested";
              return (
                <text key={s} x="548" y={ringY(s) + 5} fill={faint && s !== sel ? "#6B5D4D" : wordColor(s)} fillOpacity={faint ? 0.6 : 1}>
                  {faint ? <title>{`Layer ${s + 1}: below layer 21 the study could validate nothing the lens reads`}</title> : null}“{bare(w)}”
                </text>
              );
            })}
          </g>
        </g>
        {split ? (
          <g display={shown.split ? "inline" : "none"}>
            <path d={split.gather} fill="none" stroke="#2E4A7D" strokeWidth="4" />
            <path d={split.recall} fill="none" stroke="#A07A2E" strokeWidth="4" />
          </g>
        ) : null}
        {sp ? (
          <>
            <circle cx={sp[0]} cy={sp[1]} r="14" fill="none" stroke="#8F2B1E" strokeOpacity="0.5" strokeWidth="1" />
            <circle cx={sp[0]} cy={sp[1]} r="6.5" fill="#A07A2E" stroke="#8F2B1E" strokeWidth="1.6" />
          </>
        ) : null}
      </svg>
      <figcaption style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: "6px 22px", fontSize: 15, color: "#6B5D4D", marginTop: 4 }}>
        <span>bottom: the word-piece {sighting.readAt != null ? `“${bare(sighting.readAt)}”` : "read"} arrives</span>
        <span aria-hidden="true" style={{ color: "#A07A2E" }}>
          ◆
        </span>
        <span>top: {topWords(sighting)}</span>
        {turned ? (
          <button type="button" className="btn-link" style={{ fontSize: 15, minHeight: 0, padding: 0 }} onClick={reset}>
            reset the view
          </button>
        ) : null}
      </figcaption>
    </figure>
  );
};

/**
 * The top of the figure: the word-piece layer 60 chose here, matched by id, and its likeliest when that is another
 * (drawn by chance, after a repeat penalty, or a near-tie when read again). Without a recorded choice (the study's
 * sighting, an older server) only its likeliest is named: never the answer's first word, which is the wrong word for
 * any later word-piece.
 */
function topWords(s: Sighting): string {
  const L = s.layers[59];
  const top = L?.lensTop?.[0]?.token ?? null;
  const c = chosenOf(s);
  if (!c) return top != null ? `its likeliest “${bare(top)}”` : "its own odds";
  // a stop mark by which one it is: only <turn|> ends its turn (endedWords)
  const chose = c.endMark ? endedWords(c.id, c.token) : `it chose “${bare(c.token)}”`;
  return L && top != null && !isChosenRow(s, L, 0) ? `${chose}; its likeliest was “${bare(top)}”` : chose;
}

function zoomCentre(b: { x: number; y: number; w: number }, factor: number) {
  const w = Math.max(MIN_W, Math.min(FULL, b.w / factor));
  const cx = b.x + b.w / 2;
  const cy = b.y + b.w / 2;
  return {
    x: Math.max(0, Math.min(FULL - w, cx - w / 2)),
    y: Math.max(0, Math.min(FULL - w, cy - w / 2)),
    w,
  };
}

import React from "react";
import { Rubric } from "../../components/kit";
import { bare, gloss, numberWord, pieceWord, wp } from "../../lib/words";
import { lensPhrase, useReadingLens } from "../../lib/lens";
import { AnswerPositionCaution, OneDrawLine, SightingFarCaution, TemplateEndCaution } from "../../lib/cautions";
import {
  type Sighting,
  N_LAYERS,
  chosenOf,
  chosenPhrase,
  endMarkWords,
  isChosenRow,
  isGlobalIdx,
  ordinal,
  readWhere,
  readsInsideAnswer,
} from "../../store/useTheodoliteStore";

/*
 * The frontispiece: a sighting drawn as sixty stacked rings, one per layer, with the path of the
 * word it was read at winding up through them, and beside the path what the lens read at a few
 * stations. The geometry is the canvas's (Incipit.dc.html renderVals()): bearing and elevation per
 * layer, rounded to whole degrees as the canvas rounds them, on a schematic radius.
 */

const CX = 200;
const R = 140;
const RY = 30;
const f = (n: number) => Math.round(n * 10) / 10;
const ringY = (l: number) => 640 - 8.6 * l;

/** The canvas's stations (repo indices; layers 11, 21, 36, 41, 51 and 60). */
const PREFERRED_STATIONS = [10, 20, 35, 40, 50, 59];
const LAST = N_LAYERS - 1;

export interface Station {
  idx: number;
  no: number;
  token: string;
  prob: number | null;
  gloss: string;
  /** the lens's first choice at the top is the word-piece the model chose there (by id), or, for the recorded sighting
   * and an older server's first word-piece, written greedily, the first word it went on with */
  saysIt: boolean;
  labelY: number;
}

export interface Frontispiece {
  pathD: string;
  dotsD: string;
  stationDotsD: string;
  leadersD: string;
  start: [number, number] | null;
  end: [number, number] | null;
  stations: Station[];
  ariaLabel: string;
}

function ell(y: number) {
  return "M" + (CX - R) + "," + f(y) + " a" + R + "," + RY + " 0 1,0 " + 2 * R + ",0 a" + R + "," + RY + " 0 1,0 " + -2 * R + ",0 ";
}

function dot(p: [number, number], r: number) {
  return "M" + f(p[0] - r) + "," + p[1] + " a" + r + "," + r + " 0 1,0 " + f(2 * r) + ",0 a" + r + "," + r + " 0 1,0 " + f(-2 * r) + ",0 ";
}

/* the fixed parts of the drawing: the rings (gilded at the global-attention layers) and the limb */
const RINGS = (() => {
  let local = "";
  let global = "";
  for (let l = 0; l < N_LAYERS; l++) {
    if (isGlobalIdx(l)) global += ell(ringY(l));
    else local += ell(ringY(l));
  }
  let limb = "";
  for (let d = 0; d < 360; d += 10) {
    const a = (d * Math.PI) / 180;
    const ri = d % 30 === 0 ? 0.88 : 0.94;
    limb +=
      "M" + f(CX + 170 * Math.cos(a)) + "," + f(668 + 38 * Math.sin(a)) +
      " L" + f(CX + 170 * ri * Math.cos(a)) + "," + f(668 + 38 * ri * Math.sin(a)) + " ";
  }
  return { local, global, limb };
})();

/**
 * Which layers get a station word: the canvas's six where the lens was recorded (a live sighting
 * records it at every layer; the recorded one at nine). Failing that, whatever layers the lens was
 * recorded at, at least five layers apart so the words do not collide, six at most.
 */
function pickStations(s: Sighting): number[] {
  const has = (i: number) => (s.layers[i]?.lensTop?.length ?? 0) > 0;
  const preferred = PREFERRED_STATIONS.filter(has);
  if (preferred.length >= 4) return preferred;
  const recorded = s.layers.filter((l) => has(l.idx)).map((l) => l.idx).sort((a, b) => b - a);
  const out: number[] = [];
  for (const i of recorded) {
    if (out.length >= 6) break;
    if (out.every((j) => Math.abs(j - i) >= 5)) out.push(i);
  }
  return out.sort((a, b) => a - b);
}

/** Whether a word-piece is the first word-piece of the continuation (a whole piece, not a prefix). Used only where no
 * choice by id is known: the recorded sighting, and an older server's first word-piece, both written greedily. */
function isFirstOf(token: string, continuation: string) {
  if (!token.trim() || !continuation.startsWith(token)) return false;
  const next = continuation.charAt(token.length);
  const lastOfToken = token.charAt(token.length - 1);
  return !(next && /[\p{L}\p{N}]/u.test(next) && /[\p{L}\p{N}]/u.test(lastOfToken));
}

function stationPhrase(st: Station) {
  const m = /^(.+?): (“.+”)$/.exec(st.gloss);
  if (m && !st.saysIt) return `the ${m[1]} ${wp(st.token)}, meaning ${m[2]}, at layer ${st.no}`;
  return `${wp(st.token)} at layer ${st.no}${st.saysIt || st.gloss === "its likeliest" ? `, ${st.gloss}` : ""}`;
}

/** The top station's gloss: the word-piece it chose there, matched by id; for the recorded sighting (and an older server's
 * first word-piece), written greedily, the first word it went on with; else, when it chose another, "its likeliest". */
function topGloss(s: Sighting, top: { token: string }): { saysIt: boolean; gloss: string | null } {
  const c = chosenOf(s);
  const L = s.layers[LAST];
  // a stop mark by which one it is: only <turn|> ends its turn (endMarkWords)
  if (c) return isChosenRow(s, L, 0) ? { saysIt: true, gloss: c.endMark ? endMarkWords(c.id, c.token) : "the word it chose" } : { saysIt: false, gloss: "its likeliest" };
  const greedyFirst = s.tokenIndex === 0 && (s.source === "fixture" || s.generation == null || s.generation.greedy);
  if (greedyFirst && isFirstOf(top.token, s.continuation)) return { saysIt: true, gloss: "the word it said next" };
  return { saysIt: false, gloss: null };
}

function listPhrase(items: string[]) {
  if (items.length <= 1) return items.join("");
  return items.slice(0, -1).join(", ") + ", and " + items[items.length - 1];
}

export function frontispieceOf(s: Sighting | null): Frontispiece {
  if (!s) {
    return {
      pathD: "",
      dotsD: "",
      stationDotsD: "",
      leadersD: "",
      start: null,
      end: null,
      stations: [],
      ariaLabel:
        "Frontispiece: sixty stacked rings, one per layer, under the words terra incognita. No sighting has been read yet, so no path is drawn.",
    };
  }
  const pts: ([number, number] | null)[] = s.layers.map((layer, l) => {
    if (layer.az == null || layer.el == null) return null;
    const az = (Math.round(layer.az) * Math.PI) / 180;
    const el = (Math.round(layer.el) * Math.PI) / 180;
    const r = R * (0.3 + 0.7 * Math.cos(el));
    return [f(CX + r * Math.cos(az)), f(ringY(l) + r * Math.sin(az) * (RY / R) - Math.sin(el) * 10)];
  });
  const firstIdx = pts.findIndex((p) => p !== null);
  let lastIdx = -1;
  for (let l = pts.length - 1; l >= 0; l--) if (pts[l]) { lastIdx = l; break; }

  // the path, broken wherever a layer's bearing was not recorded
  let pathD = "";
  let pen = false;
  for (const p of pts) {
    if (!p) { pen = false; continue; }
    pathD += (pathD ? (pen ? " L" : " M") : "M") + p[0] + "," + p[1];
    pen = true;
  }

  const stationIdx = pickStations(s);
  const stations: Station[] = stationIdx.map((idx) => {
    const top = s.layers[idx].lensTop![0];
    const atTop = idx === LAST ? topGloss(s, top) : null;
    const saysIt = atTop?.saysIt ?? false;
    return {
      idx,
      no: idx + 1,
      token: top.token,
      prob: top.prob,
      gloss: atTop?.gloss ?? gloss(top.token),
      saysIt,
      labelY: Math.round(ringY(idx) + 5),
    };
  });

  const marked = new Set<number>([...stationIdx, firstIdx, lastIdx]);
  const dotsD = pts
    .map((p, l) => (p && !marked.has(l) ? dot(p, 1.6) : ""))
    .join("");
  const stationDotsD = stationIdx
    .filter((l) => l !== lastIdx && l !== firstIdx && pts[l])
    .map((l) => dot(pts[l]!, 3.2))
    .join("");
  const leadersD = stationIdx
    .filter((l) => pts[l])
    .map((l) => "M" + pts[l]![0] + "," + pts[l]![1] + " L352," + f(ringY(l)))
    .join(" ");

  const word = s.readAt != null ? pieceWord(s.readAt) : s.tokenIndex > 0 ? "its word-piece" : "the last word";
  const ariaLabel =
    "Frontispiece: a real sighting drawn as sixty stacked rings, one per layer, under the words terra incognita. " +
    (pathD
      ? `A dark path winds upward through them: the word ${word} rising from layer ${firstIdx + 1} to layer ${lastIdx + 1}. `
      : "No bearings were recorded, so no path is drawn. ") +
    (stations.length
      ? `Beside the path, the words the lens reads there: ${listPhrase(stations.map(stationPhrase))}.`
      : "The lens was not recorded at any layer, so no words stand beside it.");

  return {
    pathD,
    dotsD,
    stationDotsD,
    leadersD,
    start: firstIdx >= 0 ? pts[firstIdx] : null,
    end: lastIdx >= 0 ? pts[lastIdx] : null,
    stations,
    ariaLabel,
  };
}

/** A station's gloss as the lines it is drawn on: the top station's, which can name a stop mark at length (“a control
 * mark, <|tool_response>, where writing stops”), is broken after its commas and before a bracket, so it stays inside the
 * figure; the others are one line. */
function glossLines(st: Station): string[] {
  return st.idx === LAST ? st.gloss.replace(/, /g, ",\n").replace(/ \(/g, "\n(").split("\n") : [st.gloss];
}

/** A station word as drawn: quoted, cut short if it would run off the leaf. */
function drawnWord(token: string) {
  const b = bare(token);
  return b.length > 12 ? wp(b.slice(0, 11) + "…") : wp(token);
}

const SERIF = "EB Garamond, Garamond, serif";

export const FrontispieceFigure: React.FC<{ sighting: Sighting | null; geo: Frontispiece }> = ({ sighting, geo }) => {
  const s = sighting;
  const { lens } = useReadingLens(sighting);
  // a comma closes the phrase only where it says what the lens was fitted on ("the X lens, fitted on ...,"); the
  // shipped lens and a card that does not identify its lens take none before the verb
  const phrase = lensPhrase(lens);
  const said = s ? s.continuation.trim() : "";
  // what this position chose, by id (“where it chose “am””); nothing for the recorded sighting, which keeps no ids
  const chose = chosenPhrase(s);
  return (
    <figure style={{ flex: "1 1 460px", minWidth: 0, margin: 0, display: "flex", flexDirection: "column", gap: 10 }}>
      <svg
        viewBox="0 0 490 760"
        role="img"
        aria-label={geo.ariaLabel}
        style={{ width: "100%", maxWidth: 490, height: "auto", display: "block", margin: "0 auto" }}
      >
        <line x1="64" y1="41" x2="112" y2="41" stroke="#C9A85F" strokeWidth="0.8" />
        <line x1="288" y1="41" x2="336" y2="41" stroke="#C9A85F" strokeWidth="0.8" />
        <path d="M58 37 L62 41 L58 45 L54 41 Z M342 37 L346 41 L342 45 L338 41 Z" fill="#C9A85F" />
        <text x="200" y="46" textAnchor="middle" fontFamily={SERIF} fontStyle="italic" fontSize="16" letterSpacing="4" fill="#7A5C28">
          terra incognita
        </text>
        <path d={RINGS.local} fill="none" stroke="#2B2117" strokeOpacity="0.11" strokeWidth="0.8" />
        <path d={RINGS.global} fill="none" stroke="#A07A2E" strokeOpacity="0.75" strokeWidth="1" />
        <ellipse cx="200" cy="668" rx="170" ry="38" fill="none" stroke="#A07A2E" strokeWidth="1.2" />
        <ellipse cx="200" cy="668" rx="158" ry="35.3" fill="none" stroke="#A07A2E" strokeOpacity="0.55" strokeWidth="0.8" />
        <path d={RINGS.limb} fill="none" stroke="#A07A2E" strokeWidth="0.9" />
        <line x1="200" y1="704" x2="200" y2="96" stroke="#2B2117" strokeOpacity="0.35" strokeWidth="1" />
        <path d="M200 76 L206 87 L200 98 L194 87 Z" fill="#A07A2E" />
        <g fontFamily={SERIF} fontSize="13" fill="#7A5C28" textAnchor="end" style={{ fontVariantNumeric: "lining-nums tabular-nums" }}>
          <text x="46" y="116" fontSize="11" letterSpacing="2" fill="#6B5D4D">
            LAYER
          </text>
          <text x="46" y="137">60</text>
          <text x="46" y="240">48</text>
          <text x="46" y="343">36</text>
          <text x="46" y="446">24</text>
          <text x="46" y="549">12</text>
          <text x="46" y="644" fill="#6B5D4D">
            1
          </text>
        </g>
        {geo.leadersD ? (
          <path d={geo.leadersD} fill="none" stroke="#7A5C28" strokeOpacity="0.55" strokeWidth="0.7" strokeDasharray="2 3" />
        ) : null}
        {geo.pathD ? (
          <path d={geo.pathD} fill="none" stroke="#2B2117" strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />
        ) : null}
        {geo.dotsD ? <path d={geo.dotsD} fill="#2B2117" /> : null}
        {geo.stationDotsD ? <path d={geo.stationDotsD} fill="#F6F0E3" stroke="#2B2117" strokeWidth="1.1" /> : null}
        {geo.start ? <circle cx={geo.start[0]} cy={geo.start[1]} r="3.2" fill="#2B2117" /> : null}
        {geo.end ? (
          <>
            <circle cx={geo.end[0]} cy={geo.end[1]} r="11" fill="none" stroke="#8F2B1E" strokeOpacity="0.5" strokeWidth="1" />
            <circle cx={geo.end[0]} cy={geo.end[1]} r="5" fill="#A07A2E" stroke="#8F2B1E" strokeWidth="1.4" />
          </>
        ) : null}
        <g fontFamily={SERIF} fontSize="13" fill="#7A5C28" textAnchor="end" style={{ fontVariantNumeric: "lining-nums tabular-nums" }}>
          {geo.stations.map((st) => (
            <text key={st.idx} x="380" y={st.labelY}>
              {st.no}
            </text>
          ))}
        </g>
        <g fontFamily={SERIF} fontStyle="italic" fontSize="18" fill="#2B2117">
          {geo.stations.map((st) => (
            <text key={st.idx} x="390" y={st.labelY}>
              {drawnWord(st.token)}
            </text>
          ))}
        </g>
        <g fontFamily={SERIF} fontStyle="italic" fontSize="13" fill="#6B5D4D">
          {geo.stations
            .filter((st) => st.gloss)
            .map((st) => (
              <text key={st.idx} x="390" y={st.labelY + 17}>
                {glossLines(st).map((line, i) => (
                  <tspan key={i} x="390" dy={i ? 15 : 0}>
                    {line}
                  </tspan>
                ))}
              </text>
            ))}
        </g>
        {s ? null : (
          <text x="200" y="404" textAnchor="middle" fontFamily={SERIF} fontSize="13.5" letterSpacing="1.2" fill="#6B5D4D">
            [NO SIGHTING YET]
          </text>
        )}
        <text x="200" y="744" textAnchor="middle" fontFamily={SERIF} fontStyle="italic" fontSize="16" fill="#6B5D4D">
          {s?.readAt != null ? `the word ${pieceWord(s.readAt)} arrives` : s && s.tokenIndex > 0 ? "its word-piece arrives" : "the last word arrives"}
        </text>
      </svg>
      <figcaption style={{ display: "flex", flexDirection: "column", gap: 8, width: "100%", maxWidth: 490, margin: "0 auto" }}>
        <span style={{ fontSize: 14, fontVariantCaps: "all-small-caps", letterSpacing: "0.2em", color: "#6B5D4D" }}>
          frontispiece · a real sighting
        </span>
        {s ? (
          <span style={{ fontSize: 17, lineHeight: 1.5 }}>
            Asked “{s.question}”
            {s.answerBegins ? <>, with its answer begun for it as “{s.answerBegins}”</> : null}, the model went on{" "}
            {said ? <>“{said}”</> : <span className="placeholder">[NO CONTINUATION RECORDED]</span>}.{" "}
            {readWhere(s).role === "written" ? (
              <>
                Here {s.readAt != null ? pieceWord(s.readAt) : "a word-piece"}, the {ordinal(s.tokenIndex)} word-piece of its own answer,
                climbs all sixty layers
              </>
            ) : (
              <>Here that last {s.readAt != null ? pieceWord(s.readAt) : "word"} climbs all sixty layers</>
            )}
            {chose ? <>, to {chose}</> : null}
            {geo.stations.length ? (
              <>, and beside its path stand the words the lens reads at {numberWord(geo.stations.length)} of them</>
            ) : null}
            . The gilded rings are every sixth layer, where it looks back over the whole conversation.
            {s.isNudged ? " This sighting was taken with a nudge in place." : null}
            {s.generation && !s.generation.greedy ? (
              <>
                {" "}Its answer was <OneDrawLine generation={s.generation} />
              </>
            ) : null}
          </span>
        ) : (
          <span style={{ fontSize: 17, lineHeight: 1.5 }}>
            <span className="placeholder">[NO SIGHTING YET]</span> The path appears once a sighting has been read.
          </span>
        )}
        <Rubric as="span" style={{ fontSize: 16 }}>
          Read the middle words as glimpses, not the model’s own speech: {phrase}
          {phrase.includes(", fitted on ") ? "," : ""} is far less faithful there than
          near the top, and below layer 21 nothing it reads was validated.
        </Rubric>
        {s && readsInsideAnswer(s) ? <AnswerPositionCaution sighting={s} /> : null}
        {/* the end of the chat template, with no begun answer: the study never read the lens there */}
        {s && readWhere(s).role === "template_end" ? <TemplateEndCaution /> : null}
        {/* read at or past the end of the passages the lens was fitted on (any k), and past the attention window */}
        <SightingFarCaution sighting={s ?? null} />
      </figcaption>
    </figure>
  );
};

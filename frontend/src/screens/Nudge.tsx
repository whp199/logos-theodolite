import React, { useEffect, useState } from "react";
import { Apparatus, IfRubrics, Missing, NotYet, Page, Rubric, SweepError } from "../components/kit";
import { href, type Route } from "../router";
import { composeText, useTheodoliteStore, type Sighting, type SteeringConfig } from "../store/useTheodoliteStore";
import { faceWord, wp } from "../lib/words";
import {
  DIAL,
  EDGE,
  GENTLE_MAX,
  GLOBAL,
  H,
  HP,
  RATIO_650,
  RECORDED_PLAIN_SPIDER,
  STREAM_TABLE,
  STRIP_HATCH,
  clampV,
  fmt,
  needleAt,
  scaleFmt,
  strengthOf,
  zoneOf,
  type Zone,
} from "./nudge/dial";

/* ------------------------------------------------------------------ style */

const C = {
  ink: "#2B2117",
  muted: "#6B5D4D",
  rubric: "#8F2B1E",
  gilt: "#A07A2E",
  giltText: "#7A5C28",
  numeral: "#8A6A2C",
  control: "#C9BEAC",
  pressed: "#EDE5D3",
  hair: "#D9CFBD",
  hairSoft: "#E3DACA",
  hairFaint: "#ECE4D5",
  vellum: "#F6F0E3",
};
const SERIF = "EB Garamond, Garamond, serif";
const MONO = "'IBM Plex Mono', ui-monospace, monospace";
const LOZ = "M5 1 L9 6 L5 11 L1 6 Z";

const labelSc: React.CSSProperties = { fontVariantCaps: "all-small-caps", letterSpacing: "0.16em", fontSize: 16, color: C.muted };
const bigInput = (extra: React.CSSProperties = {}): React.CSSProperties => ({
  fontFamily: "inherit",
  fontSize: 30,
  lineHeight: 1.2,
  color: C.ink,
  background: "transparent",
  border: 0,
  borderBottom: `1px solid ${C.control}`,
  padding: "2px 0 6px",
  width: "100%",
  boxSizing: "border-box",
  ...extra,
});
const smallInput = (extra: React.CSSProperties = {}): React.CSSProperties => ({
  fontFamily: "inherit",
  fontSize: 17,
  color: C.ink,
  background: "transparent",
  border: 0,
  borderBottom: `1px solid ${C.control}`,
  padding: "6px 0",
  minHeight: 44,
  boxSizing: "border-box",
  fontVariantNumeric: "lining-nums tabular-nums",
  ...extra,
});
const ON = { color: C.ink, bg: C.pressed, border: C.rubric };
const OFF = { color: C.muted, bg: "transparent", border: C.control };
const toggleBtn = (on: boolean): React.CSSProperties => {
  const t = on ? ON : OFF;
  return {
    fontFamily: "inherit",
    fontSize: 17,
    fontVariantCaps: "all-small-caps",
    letterSpacing: "0.14em",
    color: t.color,
    background: t.bg,
    border: `1px solid ${t.border}`,
    borderRadius: 2,
    minHeight: 44,
    padding: "0 16px",
    cursor: "pointer",
  };
};
const ghostBtn: React.CSSProperties = {
  fontFamily: "inherit",
  fontSize: 17,
  fontVariantCaps: "all-small-caps",
  letterSpacing: "0.16em",
  color: C.ink,
  background: "transparent",
  border: `1px solid ${C.control}`,
  borderRadius: 2,
  padding: "8px 18px",
  minHeight: 44,
  cursor: "pointer",
};
const linkBtn: React.CSSProperties = {
  fontFamily: "inherit",
  fontSize: 17,
  color: C.rubric,
  background: "transparent",
  border: 0,
  padding: "10px 4px",
  minHeight: 44,
  cursor: "pointer",
  textDecoration: "underline",
  textDecorationThickness: "1px",
  textUnderlineOffset: "3px",
};
const outlineBtn = (color: string, disabled = false): React.CSSProperties => ({
  fontFamily: "inherit",
  fontSize: 19,
  fontVariantCaps: "all-small-caps",
  letterSpacing: "0.16em",
  background: "transparent",
  color,
  border: `1px solid ${C.control}`,
  borderRadius: 2,
  padding: "11px 24px",
  minHeight: 48,
  cursor: disabled ? "not-allowed" : "pointer",
  opacity: disabled ? 0.45 : 1,
});
const pushGrid: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "54px minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1fr)",
  gap: 8,
};
const streamGrid: React.CSSProperties = { display: "grid", gridTemplateColumns: "64px repeat(5, minmax(0, 1fr))", gap: 8 };
const placeholderBox: React.CSSProperties = {
  padding: "26px 0",
  borderTop: `1px dashed ${C.control}`,
  borderBottom: `1px dashed ${C.control}`,
  textAlign: "center",
  fontSize: 15,
  letterSpacing: "0.12em",
  color: C.muted,
};

const Loz: React.FC<{ fill: string; stroke: string }> = ({ fill, stroke }) => (
  <svg width="10" height="12" viewBox="0 0 10 12" aria-hidden="true">
    <path d={LOZ} fill={fill} stroke={stroke} strokeWidth="1.2" />
  </svg>
);

/* ------------------------------------------------------------------ words */

/** The question the canvas begins with (Gurnee et al. 2026's two-hop example; the server's default prompt). */
const SPIDER = "The number of legs on the animal that spins webs is";
const NUM = ["no layers", "one layer", "two layers", "three layers", "four layers", "five layers", "six layers", "seven layers", "eight layers", "nine layers", "ten layers"];
const ORDINAL = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth"];

const ZONES: Record<Zone, { name: string; color: string; textColor: string; textStyle: "normal" | "italic"; text: string }> = {
  gentle: {
    name: "gentle",
    color: C.ink,
    textColor: C.ink,
    textStyle: "normal",
    text: "Below one hundredth of the stream: where the tool audit says a nudge should sit.",
  },
  strong: {
    name: "strong",
    color: C.giltText,
    textColor: C.ink,
    textStyle: "normal",
    text: "Between one and about two and a half hundredths. In the study a corrected nudge still read fluently here, inside a chat. The edge is close.",
  },
  collapse: {
    name: "collapse",
    color: C.rubric,
    textColor: C.rubric,
    textStyle: "italic",
    text: "Past the edge. In the study, pushes of ×0.024 to ×0.074 broke 4 of 6 answers; at three times that, all six broke.",
  },
};

/** The reference stream strength for the length sent: the neuron's own layer, or the weakest recorded
 * median ‖h‖ among the chosen layers. Null when any chosen layer has no recorded median. */
function refStream(isNeuron: boolean, poleLayerNo: number, chosen: number[]): { h: number; layer: number } | null {
  if (isNeuron) return poleLayerNo >= 1 && poleLayerNo <= 60 && HP[poleLayerNo] ? { h: HP[poleLayerNo], layer: poleLayerNo } : null;
  if (!chosen.length || !chosen.every((n) => !!HP[n])) return null;
  let best: { h: number; layer: number } | null = null;
  for (const n of chosen) if (best === null || HP[n] < best.h) best = { h: HP[n], layer: n };
  return best;
}

/** The page's starting state, read from the nudge kept in the store (so the page and the store agree). */
function initialState() {
  const st = useTheodoliteStore.getState();
  const sc = st.steering;
  const nums = Array.from(new Set(sc.layers.map((i) => i + 1).filter((n) => Number.isInteger(n) && n >= 1 && n <= 60))).sort(
    (a, b) => a - b,
  );
  const layers: Record<number, boolean> = {};
  for (const n of GLOBAL) layers[n] = nums.includes(n);
  const extra = nums.filter((n) => !GLOBAL.includes(n));
  const neuron = sc.pole_layer !== null && sc.pole_neuron !== null;
  const poleLayerNo = neuron ? (sc.pole_layer as number) + 1 : 54;
  const ref = refStream(neuron, poleLayerNo, nums);
  let v = 100;
  if (ref && sc.scale > 0) {
    const raw = clampV(100 * (Math.log(sc.scale / ref.h) / Math.LN10 + 3));
    const snapped = Math.round(raw / 5) * 5;
    v = Math.abs(raw - snapped) < 0.5 ? snapped : raw;
  }
  const s = st.sighting;
  const question =
    s && s.source === "live" && s.isNudged && !s.answerBegins.trim() && s.question.trim() ? s.question : SPIDER;
  return {
    question,
    v,
    toward: sc.target_concept,
    away: sc.contrast_concept,
    layers,
    extraText: extra.join(", "),
    mode: (neuron ? "neuron" : "words") as "words" | "neuron",
    poleLayer: String(poleLayerNo),
    poleNeuron: neuron ? String(sc.pole_neuron) : "15491",
    poleFace: (neuron && sc.pole_polarity < 0 ? "negative" : "positive") as "positive" | "negative",
  };
}

/** An answer as the canvas sets it: the first line large, the rest smaller, ¶ where a new line began. */
const Answer: React.FC<{ text: string }> = ({ text }) => {
  if (!text.trim()) return <Missing what="the server returned no words" />;
  const nl = text.indexOf("\n");
  const head = (nl >= 0 ? text.slice(0, nl) : text).trim();
  const restLines = nl >= 0 ? text.slice(nl + 1).split("\n") : [];
  const rest = (
    <>
      {restLines.map((line, i) => (
        <React.Fragment key={i}>
          <span style={{ color: C.giltText, fontStyle: "normal" }}>¶</span> {line}
          {i < restLines.length - 1 ? " " : ""}
        </React.Fragment>
      ))}
    </>
  );
  if (head.length > 0 && head.length <= 24) {
    return (
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: "4px 14px" }}>
        <span style={{ fontSize: 44, lineHeight: 1.1, fontStyle: "italic" }}>
          “{head}
          {restLines.length ? "" : " …”"}
        </span>
        {restLines.length ? (
          <span style={{ fontSize: 22, fontStyle: "italic", color: C.muted }}>
            {rest} …”
          </span>
        ) : null}
      </div>
    );
  }
  return (
    <p style={{ margin: 0, fontSize: 22, fontStyle: "italic", lineHeight: 1.4, overflowWrap: "anywhere" }}>
      “{head}
      {restLines.length ? " " : ""}
      {rest} …”
    </p>
  );
};

/* ------------------------------------------------------------------ page */

export const Nudge: React.FC<{ route: Route }> = ({ route }) => {
  const sighting = useTheodoliteStore((s) => s.sighting);
  const plainSighting = useTheodoliteStore((s) => s.plainSighting);
  const armed = useTheodoliteStore((s) => s.steering.enabled);
  const isSweeping = useTheodoliteStore((s) => s.isSweeping);
  const setSteering = useTheodoliteStore((s) => s.setSteering);
  const takeSighting = useTheodoliteStore((s) => s.takeSighting);

  const [init] = useState(initialState);
  const [question, setQuestion] = useState(init.question);
  const [v, setV] = useState(init.v);
  const [toward, setToward] = useState(init.toward);
  const [away, setAway] = useState(init.away);
  const [layers, setLayers] = useState<Record<number, boolean>>(init.layers);
  const [mode, setMode] = useState<"words" | "neuron">(init.mode);
  const [poleLayer, setPoleLayer] = useState(init.poleLayer);
  const [poleNeuron, setPoleNeuron] = useState(init.poleNeuron);
  const [poleFace, setPoleFace] = useState<"positive" | "negative">(init.poleFace);
  const [view, setView] = useState<"page" | "old" | "server">("page");
  const [extraLayers, setExtraLayers] = useState(init.extraText);
  // typed text is kept as typed, so a half-written number is never snapped back while it is being written
  const [exactText, setExactText] = useState<string | null>(null);
  const [lengthText, setLengthText] = useState<string | null>(null);
  // when a plain sighting of this question is taken first, for comparison, the question waiting for its nudged one
  const [plainFirst, setPlainFirst] = useState<string | null>(null);

  /* ---- strength */
  const s = strengthOf(v);
  const zone = zoneOf(s);
  const z = ZONES[zone];
  const isCollapse = zone === "collapse";
  const mult = "×" + fmt(s);
  const pctS = s * 100;
  const words =
    s < 1
      ? (pctS < 10 ? pctS.toFixed(1) : String(Math.round(pctS))) + "% of the stream’s own strength"
      : fmt(s) + " times the stream’s own strength";
  const needle = needleAt(v);
  const dialAria =
    "Strength dial, from ×0.001 to ×10 of the stream’s own strength in tenfold steps. Clear up to ×0.01 (gentle), gilded to ×0.024 (strong), hatched red beyond (collapse). The study’s fluent nudge is marked near ×0.01; the old default of 650 sits at ×2.4 to ×7.4 and the old screen default of 280 at ×1.0 to ×3.2. The needle is at " +
    mult +
    ", " +
    z.name +
    ".";
  const setVTo = (x: number) => {
    setV(clampV(x));
    setExactText(null);
    setLengthText(null);
  };

  /* ---- push with: two words, or one neuron */
  const isNeuron = mode === "neuron";
  const plStr = poleLayer.trim();
  const pnStr = poleNeuron.trim();
  const pl = /^\d+$/.test(plStr) ? parseInt(plStr, 10) : NaN;
  const pn = /^\d+$/.test(pnStr) ? parseInt(pnStr, 10) : NaN;
  const okL = pl >= 1 && pl <= 60;
  const okN = pn >= 0 && pn <= 21503;
  const okPole = okL && okN;
  const facePos = poleFace !== "negative";
  const faceW = facePos ? "positive" : "negative";
  // what this sighting read of the named neuron, if it was among the loudest at its layer
  const poleHere = okPole && sighting ? (sighting.layers[pl - 1]?.poles.find((p) => p.neuron === pn) ?? null) : null;
  const known = poleHere !== null;
  const unknown = okPole && !known;

  /* ---- at which layers (the server takes any list of indices 0-59; server.py:182-183 clamps each) */
  const exTokens = extraLayers.split(/[\s,]+/).filter((x) => x.length > 0);
  const exNums = exTokens
    .filter((x) => /^\d+$/.test(x))
    .map((x) => parseInt(x, 10))
    .filter((n) => n >= 1 && n <= 60);
  const exBad = exTokens.length !== exNums.length;
  const exGilded = exNums.some((n) => GLOBAL.includes(n));
  const extra = exNums.filter((n, i) => exNums.indexOf(n) === i && !GLOBAL.includes(n));
  const gilded = GLOBAL.filter((n) => !!layers[n]);
  const chosen = gilded.concat(extra).sort((a, b) => a - b);
  const chosenText = isNeuron
    ? "the neuron’s own layer"
    : (chosen.length === 10 && !extra.length ? "all ten layers" : chosen.length <= 10 ? NUM[chosen.length] : chosen.length + " layers") +
      " chosen";

  const pushFmt = (x: number) => (x < 10 ? x.toFixed(1) : String(Math.round(x)));
  const rowFor = (n: number) => {
    const h = H[n];
    if (h) return { layer: n, h: String(h), push: pushFmt(s * h), old: "×" + RATIO_650[n], recorded: true };
    return { layer: n, h: "[AT SIGHTING]", push: "[AT SIGHTING]", old: "[AT SIGHTING]", recorded: false };
  };
  const rows = isNeuron ? (okL ? [rowFor(pl)] : []) : chosen.map(rowFor);
  const blocked = isNeuron ? !okPole : chosen.length === 0;

  /* ---- the scale the server receives: s × the reference ‖h‖ */
  const ref = refStream(isNeuron, pl, chosen);
  const scaleNum = ref === null ? null : s * ref.h;
  const scaleUnknown = !blocked && scaleNum === null;
  const cantArm = blocked || scaleUnknown;

  // a live sighting's own stream strength at the chosen layers, shown beside the recorded medians
  const liveH =
    sighting && sighting.source === "live"
      ? (isNeuron ? (okL ? [pl] : []) : chosen)
          .map((n) => ({ n, h: sighting.layers[n - 1]?.streamStrength ?? null }))
          .filter((x): x is { n: number; h: number } => x.h !== null)
      : [];

  /* ---- the nudge as set */
  const tw = toward.trim();
  const aw = away.trim();
  const wordsLine =
    (tw ? "toward “" + tw + "”" : "toward [NO WORD YET]") +
    (aw ? ", away from “" + aw + "”" : "") +
    " · " +
    (chosen.length ? (chosen.length === 1 ? "layer " : "layers ") + chosen.join(", ") : "no layers chosen");
  const neuronLine = okPole ? "neuron " + pn + " at layer " + pl + ", its " + faceW + " face" : "no neuron chosen yet";
  const setLine = (isNeuron ? neuronLine : wordsLine) + " · " + mult + " of the stream";
  const nudgedNote = isNeuron
    ? okPole
      ? "with neuron " + pn + ", " + faceW + " face"
      : "no neuron yet"
    : tw
      ? "toward “" + tw + "”"
      : "no word yet";

  /* ---- the request: SteeringLaserRequest (backend/server.py:100-108), sent as payload.steering of a "sweep" */
  let scaleRule: string;
  if (!isNeuron && !chosen.length) scaleRule = "No layer is chosen, so there is nothing to scale.";
  else if (ref === null || scaleNum === null)
    scaleRule =
      "The server reads scale as one absolute length for every layer it pushes. A layer here has no recorded ‖h‖, so the length waits for a sighting: [MEASURED IN A LIVE SIGHTING]";
  else
    scaleRule =
      "The server reads scale as one absolute length for every layer it pushes, so the page sends s × ‖h‖ = " +
      s.toPrecision(3) +
      " × " +
      ref.h.toFixed(1) +
      " = " +
      scaleFmt(scaleNum) +
      ", the median ‖h‖ at layer " +
      ref.layer +
      (isNeuron ? ", the neuron’s own." : ": the weakest stream among those chosen, so no layer is pushed past the share you set.");
  const q = (x: unknown) => JSON.stringify(x);
  const bodyFor = (o: { enabled: boolean; t: string; c: string; layers: number[]; scale: string; pl: string; pn: string; pp: string }) =>
    '{\n  "enabled": ' +
    o.enabled +
    ',\n  "target_concept": ' +
    q(o.t) +
    ',\n  "contrast_concept": ' +
    q(o.c) +
    ',\n  "layers": [' +
    o.layers.join(", ") +
    '],\n  "scale": ' +
    o.scale +
    ',\n  "pole_layer": ' +
    o.pl +
    ',\n  "pole_neuron": ' +
    o.pn +
    ',\n  "pole_polarity": ' +
    o.pp +
    "\n}";
  const poleL = isNeuron && okPole ? String(pl - 1) : "null";
  const poleN = isNeuron && okPole ? String(pn) : "null";
  const poleP = isNeuron && !facePos ? "-1.0" : "1.0";
  const pageScale =
    scaleNum === null ? q(!isNeuron && !chosen.length ? "[CHOOSE A LAYER]" : "[MEASURED IN A LIVE SIGHTING]") : scaleFmt(scaleNum);
  const BODIES = {
    page: bodyFor({ enabled: armed, t: toward, c: away, layers: chosen.map((n) => n - 1), scale: pageScale, pl: poleL, pn: poleN, pp: poleP }),
    old: bodyFor({ enabled: true, t: toward, c: away, layers: [35, 41, 47, 53, 59], scale: "280.0", pl: poleL, pn: poleN, pp: poleP }),
    server: bodyFor({ enabled: false, t: "six", c: "eight", layers: [29, 35, 41, 47, 53], scale: "650.0", pl: "null", pn: "null", pp: "1.0" }),
  };
  const NOTES = {
    page:
      "What this page would send now. Layers go as repo indices, one below the layer number" +
      (isNeuron ? "; pole_layer " + (okPole ? pl - 1 : "—") + " is layer " + (okPole ? pl : "—") + "." : "."),
    old: isNeuron
      ? "What the old screen sent for the same neuron: its “Steer Pole” button set enabled to true and the three pole fields, and left the rest as it was: the words, layers 36 to 60, and its slider’s default, 280."
      : "What the old screen sent for the same words, at its defaults: its fire button set enabled to true, it had no control for the layers, and 280 was its slider’s default.",
    server: "SteeringLaserRequest() with nothing filled in (server.py:100–108); enabled is false, so nothing is pushed.",
  };
  // The audit's fixes are asked for apart: none is a field of SteeringLaserRequest yet, so none is ever sent.
  const asks = [
    ...(isNeuron ? [] : ["the corrected direction, with no inverse and no clamp"]),
    "the quiet dimensions masked",
    "the start mark left alone",
    isNeuron ? "the strength as a share of ‖h‖" : "the strength as a share of ‖h‖ at each layer",
  ];
  const ignored = isNeuron
    ? "layers, target_concept and contrast_concept: with pole_layer and pole_neuron set, the server pushes along the neuron alone, at its one layer."
    : "pole_layer, pole_neuron and pole_polarity: left null, so the server runs the two-word recipe at every listed layer.";

  /* ---- what is sent: exactly the page's request, with enabled set by arming */
  const cfg: Omit<SteeringConfig, "enabled"> | null =
    cantArm || scaleNum === null
      ? null
      : {
          target_concept: toward,
          contrast_concept: away,
          layers: chosen.map((n) => n - 1),
          scale: Number(scaleFmt(scaleNum)),
          pole_layer: isNeuron ? pl - 1 : null,
          pole_neuron: isNeuron ? pn : null,
          pole_polarity: isNeuron && !facePos ? -1 : 1,
        };
  const cfgKey = cfg ? JSON.stringify(cfg) : null;

  // Keep the store's nudge in step with the page whenever the page's nudge is complete; whether it is
  // armed is changed only by arming and releasing.
  useEffect(() => {
    if (cfgKey === null) return;
    const next = JSON.parse(cfgKey) as Omit<SteeringConfig, "enabled">;
    const cur = useTheodoliteStore.getState().steering;
    const same =
      cur.target_concept === next.target_concept &&
      cur.contrast_concept === next.contrast_concept &&
      cur.scale === next.scale &&
      cur.pole_layer === next.pole_layer &&
      cur.pole_neuron === next.pole_neuron &&
      cur.pole_polarity === next.pole_polarity &&
      cur.layers.length === next.layers.length &&
      cur.layers.every((x, i) => x === next.layers[i]);
    if (!same) setSteering(next);
  }, [cfgKey, setSteering]);

  const prompt = composeText(question, "");
  const samePrompt = (x: Sighting | null): x is Sighting =>
    !!x && x.source === "live" && composeText(x.question, x.answerBegins) === prompt;

  /** Arm (or keep armed) and take a sighting of this question with the nudge. If there is no live plain
   * sighting of the same question to compare with, one is taken first, without the nudge. */
  const sight = () => {
    if (!cfg) return;
    setSteering({ ...cfg, enabled: true });
    const st = useTheodoliteStore.getState();
    if (st.isSweeping) return;
    const p = st.plainSighting;
    const havePlain = samePrompt(p) && !p.isNudged;
    if (st.conn === "open" && !havePlain) {
      if (takeSighting({ question, answerBegins: "", steering: { ...cfg, enabled: false } })) setPlainFirst(question);
      return;
    }
    takeSighting({ question, answerBegins: "", steering: { ...cfg, enabled: true } });
  };
  const release = () => {
    setPlainFirst(null);
    setSteering({ enabled: false });
  };

  // once the plain sighting is back, take the nudged one
  useEffect(() => {
    if (plainFirst === null || isSweeping) return;
    setPlainFirst(null);
    const st = useTheodoliteStore.getState();
    if (st.sweepError || st.conn !== "open" || !st.steering.enabled) return;
    takeSighting({ question: plainFirst, answerBegins: "", steering: { enabled: true } });
  }, [plainFirst, isSweeping, takeSighting]);

  /* ---- before and after */
  const nudgedNow = sighting && sighting.isNudged && samePrompt(sighting) ? sighting : null;
  const plainNow = samePrompt(plainSighting) && !plainSighting.isNudged ? plainSighting : null;
  const recordedPlain = !plainNow && question.trim() === SPIDER;

  /* ---- the strength apparatus */
  const sExact = exactText === null ? String(Number(s.toPrecision(3))) : exactText;
  const lenValue = lengthText !== null ? lengthText : scaleNum === null ? "" : scaleFmt(scaleNum);
  const onExact = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value;
    const x = parseFloat(raw);
    setExactText(raw);
    if (x > 0) {
      setLengthText(null);
      setV(clampV(100 * (Math.log(x) / Math.LN10 + 3)));
    }
  };
  const onLength = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value;
    const x = parseFloat(raw);
    setLengthText(raw);
    if (ref !== null && x > 0) {
      setExactText(null);
      setV(clampV(100 * (Math.log(x / ref.h) / Math.LN10 + 3)));
    }
  };

  const steps = {
    two: isNeuron ? "Name a neuron, its layer, and the face to push with." : "Name a word to nudge toward, and one to nudge away from.",
    three: isNeuron ? "The push is made at the neuron’s own layer." : "Choose the layers where the push is made.",
  };
  const blockedText = isNeuron
    ? "Choose a layer from 1 to 60 and a neuron from 0 to 21,503 to arm the nudge."
    : "Choose at least one layer to arm the nudge.";
  const blockedAside = isNeuron ? "Choose the neuron’s layer above to see its push." : "Choose a layer above to see its push.";
  const unknownScaleText = isNeuron
    ? "The neuron’s layer has no recorded stream strength, so the length to send waits for it: "
    : "A chosen layer has no recorded stream strength, so the length to send waits for it: ";

  const lensWords = poleHere?.lens ? poleHere.lens.slice(0, 3).map((t) => wp(t.token)).join(", ") : null;

  return (
    <Page route={route} folio="folio v">
      {/* i · the question */}
      <section
        aria-labelledby="q-head"
        style={{ display: "flex", flexWrap: "wrap", gap: "28px 56px", padding: "34px 0 32px", alignItems: "flex-start" }}
      >
        <div style={{ flex: "999 1 620px", minWidth: 0, display: "flex", flexDirection: "column", gap: 16 }}>
          <h2 className="h-sec" id="q-head">
            i · the question
          </h2>
          <div style={{ display: "grid", gridTemplateColumns: "150px minmax(0, 1fr)", gap: "10px 20px", alignItems: "baseline" }}>
            <label htmlFor="ask" style={labelSc}>
              you ask
            </label>
            <input
              id="ask"
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              style={bigInput({ lineHeight: 1.25, padding: "4px 0 8px", outlineOffset: 6 })}
            />
          </div>
          <Rubric style={{ margin: "2px 0 0 170px", maxWidth: 640 }}>
            A nudge is a small push on the model’s stream of thought while it reads. Begin with a question whose answer you know: here the
            model counts a spider’s legs, and says eight. We will try to tip it toward six.
          </Rubric>
        </div>
        <div style={{ flex: "1 1 340px", minWidth: 0, display: "flex", flexDirection: "column", gap: 10 }}>
          <IfRubrics>
            <h3 style={{ margin: 0, fontSize: 14, fontWeight: 500, fontVariantCaps: "all-small-caps", letterSpacing: "0.2em", color: C.rubric }}>
              the order of the nudge
            </h3>
            <ol style={{ listStyle: "none", margin: 0, padding: 0, display: "flex", flexDirection: "column", gap: 6 }}>
              {[
                ["ii", steps.two],
                ["iii", steps.three],
                ["iv", "Choose how hard, keeping the needle in the clear part of the dial."],
                ["v", "Arm it and take a sighting. Release it when you are done."],
              ].map(([n, text]) => (
                <li key={n} style={{ display: "grid", gridTemplateColumns: "30px minmax(0, 1fr)", gap: 8, alignItems: "baseline" }}>
                  <span style={{ fontSize: 15, fontVariantCaps: "all-small-caps", letterSpacing: "0.12em", color: C.giltText }}>{n}</span>
                  <span style={{ fontStyle: "italic", fontSize: 17, color: C.rubric }}>{text}</span>
                </li>
              ))}
            </ol>
          </IfRubrics>
        </div>
      </section>

      <div style={{ display: "flex", flexWrap: "wrap", gap: "36px 56px", padding: "30px 0 34px", borderTop: `1px solid ${C.hairSoft}` }}>
        {/* ii · which way */}
        <section aria-labelledby="way-head" style={{ flex: "1 1 400px", minWidth: 0, display: "flex", flexDirection: "column", gap: 16 }}>
          <h2 className="h-sec" id="way-head">
            ii · which way
          </h2>
          <div role="group" aria-label="Push with" style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "6px 10px" }}>
            <span aria-hidden="true" style={{ ...labelSc, marginRight: 8 }}>
              push with
            </span>
            <button
              type="button"
              onClick={() => {
                setMode("words");
                setLengthText(null);
              }}
              aria-pressed={!isNeuron}
              style={toggleBtn(!isNeuron)}
            >
              two words
            </button>
            <span aria-hidden="true" style={{ color: C.gilt }}>
              ·
            </span>
            <button
              type="button"
              onClick={() => {
                setMode("neuron");
                setLengthText(null);
              }}
              aria-pressed={isNeuron}
              style={toggleBtn(isNeuron)}
            >
              one neuron
            </button>
          </div>

          {!isNeuron ? (
            <>
              <div style={{ display: "grid", gridTemplateColumns: "150px minmax(0, 1fr)", gap: "14px 20px", alignItems: "baseline" }}>
                <label htmlFor="toward" style={labelSc}>
                  nudge toward
                </label>
                <input
                  id="toward"
                  type="text"
                  value={toward}
                  onChange={(e) => setToward(e.target.value)}
                  style={bigInput({ fontStyle: "italic", maxWidth: 260 })}
                />
                <label htmlFor="away" style={labelSc}>
                  and away from
                </label>
                <input
                  id="away"
                  type="text"
                  value={away}
                  onChange={(e) => setAway(e.target.value)}
                  style={bigInput({ fontStyle: "italic", maxWidth: 260 })}
                />
              </div>
              <Rubric style={{ maxWidth: 520 }}>
                Use one short word in each. The instrument reads each as a single word-piece, and a longer word is quietly cut to its first
                piece.
              </Rubric>
            </>
          ) : (
            <>
              <div style={{ display: "grid", gridTemplateColumns: "150px minmax(0, 1fr)", gap: "14px 20px", alignItems: "baseline" }}>
                <label htmlFor="pole-layer" style={labelSc}>
                  at layer
                </label>
                <input
                  id="pole-layer"
                  type="number"
                  min="1"
                  max="60"
                  step="1"
                  value={poleLayer}
                  onChange={(e) => {
                    setPoleLayer(e.target.value);
                    setLengthText(null);
                  }}
                  style={bigInput({ maxWidth: 160, fontVariantNumeric: "lining-nums tabular-nums" })}
                />
                <label htmlFor="pole-neuron" style={labelSc}>
                  neuron
                </label>
                <input
                  id="pole-neuron"
                  type="number"
                  min="0"
                  max="21503"
                  step="1"
                  value={poleNeuron}
                  onChange={(e) => setPoleNeuron(e.target.value)}
                  style={bigInput({ maxWidth: 200, fontVariantNumeric: "lining-nums tabular-nums" })}
                />
                <span id="face-label" style={labelSc}>
                  push with its
                </span>
                <div role="group" aria-labelledby="face-label" style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                  <button type="button" onClick={() => setPoleFace("positive")} aria-pressed={facePos} style={toggleBtn(facePos)}>
                    positive face
                  </button>
                  <button type="button" onClick={() => setPoleFace("negative")} aria-pressed={!facePos} style={toggleBtn(!facePos)}>
                    negative face
                  </button>
                </div>
              </div>
              {!okPole ? <p style={{ margin: 0, fontSize: 16, color: C.rubric }}>A layer runs from 1 to 60, and a neuron from 0 to 21,503.</p> : null}
              {known && poleHere ? (
                <>
                  <p style={{ margin: 0, fontSize: 17, maxWidth: 520 }}>
                    Neuron {pn} fired
                    {sighting?.readAt != null ? <> as the model read {wp(sighting.readAt)}</> : null}
                    {sighting?.answerBegins.trim() ? <> in its reply, “{sighting.answerBegins.trim()}”</> : null}: the{" "}
                    {poleHere.rank === 0 ? "" : (ORDINAL[poleHere.rank] ?? `number ${poleHere.rank + 1}`) + " "}loudest in layer {pl}.{" "}
                    {lensWords ? (
                      <>
                        Its {faceWord(poleHere.face)} face writes{" "}
                        {pn === 15491 && pl === 54 ? (
                          <>
                            forms of <i>to be</i>:{" "}
                          </>
                        ) : null}
                        {lensWords}.
                      </>
                    ) : (
                      <>
                        What its {faceWord(poleHere.face)} face writes:{" "}
                        <Missing what={sighting?.source === "live" ? "not labelled in this sighting" : "measured in a live sighting"} />
                      </>
                    )}
                  </p>
                  {(facePos ? 1 : -1) !== poleHere.face ? (
                    <p style={{ margin: 0, fontSize: 16, color: C.muted, maxWidth: 520 }}>
                      Pushing with its {faceW} face sends the same push the other way.
                    </p>
                  ) : null}
                  <a href={href.neuron(pl, pn, poleHere.face)} style={{ fontSize: 17, padding: "10px 0", alignSelf: "flex-start" }}>
                    read neuron {pn}’s folio →
                  </a>
                </>
              ) : null}
              {unknown ? (
                <p style={{ margin: 0, fontSize: 17, maxWidth: 520 }}>
                  The instrument has not read this neuron yet. Take a sighting, then open <a href={href.layer(pl)}>its layer’s leaf</a> to see
                  what it writes. Or begin with one already read, <a href={href.neuron(54, 15491, 1)}>neuron 15491</a>.
                </p>
              ) : null}
              <Rubric style={{ maxWidth: 520 }}>
                A neuron has two faces: one pushes the stream one way, the other the opposite way. Pushing with a face adds that push at the
                neuron’s own layer. Begin with one already read: neuron 15491, from the sighting of “Tell me about yourself.”
              </Rubric>
            </>
          )}
        </section>

        {/* iii · at which layers */}
        <section aria-labelledby="layers-head" style={{ flex: "999 1 560px", minWidth: 0, display: "flex", flexDirection: "column", gap: 14 }}>
          <div className="row-between">
            <h2 className="h-sec" id="layers-head">
              iii · at which layers
            </h2>
            <span style={{ fontSize: 15, color: C.muted }}>{chosenText}</span>
          </div>
          {!isNeuron ? (
            <>
              <div role="group" aria-label="Layers to nudge" style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                {GLOBAL.map((n) => {
                  const on = !!layers[n];
                  return (
                    <button
                      key={n}
                      type="button"
                      onClick={() => {
                        setLayers((cur) => ({ ...cur, [n]: !cur[n] }));
                        setLengthText(null);
                      }}
                      aria-pressed={on}
                      aria-label={"Layer " + n}
                      style={{
                        fontFamily: "inherit",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        gap: 8,
                        minWidth: 66,
                        minHeight: 52,
                        padding: "6px 12px",
                        border: `1px solid ${on ? C.rubric : C.control}`,
                        borderRadius: 2,
                        background: on ? C.pressed : "transparent",
                        color: C.ink,
                        cursor: "pointer",
                        fontSize: 20,
                        fontVariantNumeric: "lining-nums tabular-nums",
                      }}
                    >
                      <Loz fill={on ? C.rubric : "none"} stroke={on ? C.rubric : C.gilt} />
                      <span>{n}</span>
                    </button>
                  );
                })}
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "4px 22px", fontSize: 15, color: C.muted }}>
                <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <Loz fill={C.rubric} stroke={C.rubric} />
                  chosen
                </span>
                <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <Loz fill="none" stroke={C.gilt} />
                  not chosen
                </span>
                <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span aria-hidden="true" style={{ color: C.gilt }}>
                    ✦
                  </span>
                  all ten look back over the whole conversation
                </span>
              </div>
              <Rubric style={{ maxWidth: 700 }}>
                The nudge is offered at the ten gilded layers, every sixth, where the model looks back over the whole conversation. The five
                already chosen are the old server’s default, and the layers where the study tested it. The old screen sent layers 36, 42, 48,
                54 and 60 instead.
              </Rubric>
              <Apparatus title="any layer" label="Apparatus for the layers" sources={false}>
                <dl className="dl" style={{ gap: "7px 18px" }}>
                  <dt>the server takes</dt>
                  <dd className="num">
                    any list of layer indices, each clamped to 0–59 (<span className="mono">server.py:182–183</span>). The gilded ten are offered
                    above because the study tested the nudge at five of them; its own persona steering held at layers 26–40 (REPORT §1.3).
                  </dd>
                  <dt>
                    <label htmlFor="extra-layers">add other layers</label>
                  </dt>
                  <dd style={{ margin: 0, display: "flex", flexDirection: "column", gap: 4 }}>
                    <input
                      id="extra-layers"
                      type="text"
                      inputMode="numeric"
                      value={extraLayers}
                      onChange={(e) => {
                        setExtraLayers(e.target.value);
                        setLengthText(null);
                      }}
                      placeholder="e.g. 26, 33, 40"
                      style={smallInput({ width: "100%", maxWidth: 280 })}
                    />
                    <span style={{ color: C.muted }}>
                      {extra.length
                        ? "now added: layer" +
                          (extra.length === 1 ? " " : "s ") +
                          extra.join(", ") +
                          " (index " +
                          extra.map((n) => n - 1).join(", ") +
                          ")."
                        : "none added."}{" "}
                      Layer numbers from 1 to 60, separated by commas; the gilded ten stay with their buttons. Stream strength there:{" "}
                      <Missing />
                    </span>
                  </dd>
                </dl>
                {exBad ? <p style={{ margin: 0, color: C.rubric }}>Only whole numbers from 1 to 60 are read; anything else is set aside.</p> : null}
                {exGilded ? <p style={{ margin: 0, color: C.muted }}>A gilded layer typed here is left to its button above.</p> : null}
              </Apparatus>
            </>
          ) : (
            <>
              <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "6px 16px", minHeight: 52 }}>
                <Loz fill={C.rubric} stroke={C.rubric} />
                <span style={{ fontSize: 20, fontVariantNumeric: "lining-nums tabular-nums" }}>layer {okL ? String(pl) : "not yet chosen"}</span>
                <span style={{ fontSize: 15, color: C.muted }}>the neuron’s own; no other layer is pushed</span>
              </div>
              <Rubric style={{ maxWidth: 700 }}>
                A neuron lives in a single layer, so in this mode the push is made there alone. To push at several layers at once, push with two
                words.
              </Rubric>
            </>
          )}
        </section>
      </div>

      {/* apparatus · the direction of the push */}
      <Apparatus title="the direction of the push" label="Apparatus for the direction" style={{ padding: "12px 0 26px" }}>
        <dl className="dl" style={{ gap: "7px 18px" }}>
          <dt>two words, as built</dt>
          <dd>
            u = (W<sub>U</sub>[t<sub>+</sub>] − W<sub>U</sub>[t<sub>−</sub>]) ⊙ 1 / max(fnw, 0.1), then v<sub>ℓ</sub> = u · J<sub>ℓ</sub> through
            the J++ lens (Ayonrinde &amp; Lindsey 2026) [4], at each chosen layer: the lens-vector steering of Gurnee et al. 2026 [2]. Each
            word becomes one word-piece, the space-led piece when there is one. <span className="mono">server.py:171–189</span>
          </dd>
          <dt>one neuron, as built</dt>
          <dd>
            v = σ · (W<sub>down</sub>[:, i] ⊙ γ<sub>post-ffn, ℓ</sub>), σ = +1 for the positive pole and −1 for the negative (poles after
            Cammarata 2026 [9]; the GeGLU block [15]), at the neuron’s own layer only. The column carries any vault edit in force. The old
            screen’s “Steer Pole” button sent it, and called the poles +North and −South. <span className="mono">server.py:156–169</span>
          </dd>
          <dt>how it is added</dt>
          <dd>
            h<sub>ℓ</sub> ← h<sub>ℓ</sub> + scale · v<sub>ℓ</sub> / ‖v<sub>ℓ</sub>‖ on the stream after layer ℓ (resid_post), by{" "}
            <span className="mono">AddSpec(vector=v, scale=scale, normalize=True)</span>: the scale is an absolute length, not a share of ‖h‖. By
            default 650 at indices <span className="mono">[29, 35, 41, 47, 53]</span>, layers 30–54 (<span className="mono">server.py:104–105</span>
            ). No position mask, so the start mark and every word-piece the model writes are pushed too. Additive steering after ActAdd (Turner
            et al. 2023) [21].
          </dd>
          <dt>the audit’s fix</dt>
          <dd>
            u = fnw ⊙ ΔW<sub>U</sub>, with no inverse and no clamp; the 181 quiet dimensions projected out (deep-quiet at every layer from 43 to
            58, the report’s L42–57); the start mark masked; strength s · ‖h<sub>ℓ</sub>‖, s ≈ 0.01 (V7 notes) to 0.02 (REPORT §3.4). REPORT
            §3.4, bug 1; §3.5, item 4.
          </dd>
          <dt>units</dt>
          <dd>
            A lens or gradient pullback is a covector, a way of reading the stream; adding it as though it were a displacement is a units error
            in Gemma 4 (REPORT §2.2, after the causal inner product of Park et al. 2024 [24]). Such covectors put 99–100% of their energy in the
            quiet dimensions at layers 36–59. A neuron’s write column is a vector the model itself adds, not a pullback; whether its push
            reaches the quiet dimensions was not measured: <Missing what="not measured in the study" />
          </dd>
          <dt>measured</dt>
          <dd className="num">
            cos(as built, the fix) = 0.12 in the final space; after the pullback 0.14, 0.10, 0.03, 0.04 and 0.09 at layers 30, 36, 42, 48 and
            54. Three inverse weights clamped to 10, against a median of 0.156; 79.1% of the as-built energy in those three dimensions.{" "}
            <span className="mono">V7_tool_bugs_02_steer_p0.json</span>
          </dd>
          <dt>trusting a neuron</dt>
          <dd>
            Output labels held up only at layers 41–58 (REPORT §3.2), and best where the lens and the naked eye agree (AUC 0.84). Neuron 15491
            at layer 54: the lens reads “Am”, “jestem”, “Somos”, the naked eye “jestem”, “jsem”, “êtes”; one word-piece shared. The study’s
            own single-neuron test pushed the activation, not the stream: s · 60 · max|a| added to mlp_act at the last word-piece, over 36
            contexts (N1A).
          </dd>
          <dt>ask also for</dt>
          <dd style={{ margin: 0, display: "flex", flexDirection: "column", gap: 0 }}>
            <label className="check disabled">
              <input type="checkbox" checked disabled readOnly />
              <span>
                the corrected direction, fnw ⊙ ΔW<sub>U</sub> <span style={{ color: C.muted }}>(two words only)</span>
              </span>
            </label>
            <label className="check disabled">
              <input type="checkbox" checked disabled readOnly />
              mask the 181 quiet dimensions
            </label>
            <label className="check disabled">
              <input type="checkbox" checked disabled readOnly />
              leave the start mark alone (a position mask)
            </label>
            <label className="check disabled">
              <input type="checkbox" checked disabled readOnly />
              {isNeuron ? "the strength as a share of ‖h‖, set by the server" : "the strength as a share of ‖h‖ at each layer, set by the server"}
            </label>
            <NotYet style={{ margin: 0 }}>
              Not yet fields of SteeringLaserRequest: the server ignores them until the fix lands, so the request below lists them apart.
            </NotYet>
          </dd>
        </dl>
      </Apparatus>

      {/* iv · how hard */}
      <section
        aria-labelledby="hard-head"
        style={{ borderTop: `1px solid ${C.hairSoft}`, paddingTop: 30, display: "flex", flexDirection: "column", gap: 16 }}
      >
        <div className="row-between">
          <h2 className="h-sec" id="hard-head">
            iv · how hard
          </h2>
          <span style={{ fontSize: 15, color: C.muted }}>
            a share of the stream’s own strength <span style={{ fontStyle: "italic" }}>(‖h‖)</span>, on a scale of tenfold steps
          </span>
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "36px 56px", alignItems: "flex-start" }}>
          <div style={{ flex: "999 1 620px", minWidth: 0, display: "flex", flexDirection: "column", gap: 14 }}>
            <figure style={{ margin: 0 }}>
              <svg
                viewBox="0 40 1000 400"
                role="img"
                aria-label={dialAria}
                style={{ width: "100%", maxWidth: 900, height: "auto", display: "block", margin: "0 auto" }}
              >
                <path d={DIAL.innerRing} fill="none" stroke="#C9A85F" strokeWidth="0.9" />
                <path d={DIAL.innerTicks} fill="none" stroke="#C9A85F" strokeWidth="0.8" />
                <path d={DIAL.strongBand} fill={C.gilt} fillOpacity="0.3" stroke="none" />
                <path d={DIAL.hatch} fill="none" stroke={C.rubric} strokeWidth="1" />
                <path d={DIAL.collapseBand} fill="none" stroke={C.rubric} strokeWidth="0.8" />
                <path d={DIAL.bandLines} fill="none" stroke={C.gilt} strokeOpacity="0.55" strokeWidth="0.6" />
                <path d={DIAL.limbOuter} fill="none" stroke={C.gilt} strokeWidth="1.4" />
                <path d={DIAL.limbInner} fill="none" stroke={C.gilt} strokeWidth="1" />
                <path d={DIAL.fine} fill="none" stroke={C.gilt} strokeOpacity="0.6" strokeWidth="0.7" />
                <path d={DIAL.minor} fill="none" stroke={C.gilt} strokeWidth="0.9" />
                <path d={DIAL.major} fill="none" stroke={C.ink} strokeWidth="1.3" />
                <path d={DIAL.dividers} fill="none" stroke={C.ink} strokeWidth="1.2" />
                <g fontFamily={SERIF} fontSize="19" fill={C.giltText} textAnchor="middle" style={{ fontVariantNumeric: "lining-nums tabular-nums" }}>
                  <text x={DIAL.dA.x} y={DIAL.dA.y}>
                    ×0.001
                  </text>
                  <text x={DIAL.dB.x} y={DIAL.dB.y}>
                    ×0.01
                  </text>
                  <text x={DIAL.dC.x} y={DIAL.dC.y}>
                    ×0.1
                  </text>
                  <text x={DIAL.dD.x} y={DIAL.dD.y}>
                    ×1
                  </text>
                  <text x={DIAL.dE.x} y={DIAL.dE.y}>
                    ×10
                  </text>
                </g>
                <g fontFamily={SERIF} fontSize="18" letterSpacing="3" textAnchor="middle" style={{ fontVariantCaps: "all-small-caps" }}>
                  <text x={DIAL.zG.x} y={DIAL.zG.y} transform={DIAL.zG.tr} fill={C.muted}>
                    gentle
                  </text>
                  <text x={DIAL.zS.x} y={DIAL.zS.y} transform={DIAL.zS.tr} fill={C.giltText}>
                    strong
                  </text>
                  <text x={DIAL.zC.x} y={DIAL.zC.y} transform={DIAL.zC.tr} fill={C.rubric}>
                    collapse
                  </text>
                </g>
                <path d={DIAL.brFluent} fill="none" stroke={C.gilt} strokeWidth="1.6" />
                <path d={DIAL.brOld} fill="none" stroke={C.rubric} strokeWidth="1.6" />
                <g fontFamily={SERIF} fontSize="17" fontStyle="italic" textAnchor="middle">
                  <text x={DIAL.lF.x} y={DIAL.lF.y} transform={DIAL.lF.tr} fill={C.giltText}>
                    fluent in the study
                  </text>
                  <text x={DIAL.lO.x} y={DIAL.lO.y} transform={DIAL.lO.tr} fill={C.rubric}>
                    the old default, 650
                  </text>
                  <text x={DIAL.lS.x} y={DIAL.lS.y} transform={DIAL.lS.tr} fill={C.rubric}>
                    the old screen default, 280
                  </text>
                </g>
                <text
                  x="500"
                  y="366"
                  textAnchor="middle"
                  fontFamily={SERIF}
                  fontSize="17"
                  letterSpacing="3"
                  fill={C.giltText}
                  style={{ fontVariantCaps: "all-small-caps" }}
                >
                  the stream’s own strength
                </text>
                <text x="500" y="394" textAnchor="middle" fontFamily={SERIF} fontSize="16" fontStyle="italic" fill={C.muted}>
                  at ×1 the push is as strong as the stream
                </text>
                <path d="M500 410 L505 418 L500 426 L495 418 Z" fill={C.gilt} />
                <path d={needle.needle} fill="none" stroke={C.ink} strokeWidth="1.8" strokeLinecap="round" />
                <path d={needle.loz} fill={C.rubric} />
                <circle cx={needle.tip.x} cy={needle.tip.y} r="4" fill={C.vellum} stroke={C.rubric} strokeWidth="1.6" />
              </svg>
            </figure>
            <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-between", alignItems: "baseline", gap: "4px 16px" }}>
                <label htmlFor="strength" style={labelSc}>
                  strength
                </label>
                <span style={{ fontSize: 15, color: C.muted }}>slide to turn the needle, or use the arrow keys</span>
              </div>
              <input
                id="strength"
                type="range"
                min="0"
                max="400"
                step="5"
                value={v}
                onChange={(e) => setVTo(Number(e.target.value))}
                aria-valuetext={mult + ", " + words + ", " + z.name}
                style={{ width: "100%", height: 44, margin: 0, accentColor: C.rubric, cursor: "pointer" }}
              />
              <div style={{ padding: "0 8px" }}>
                <svg viewBox="0 0 1000 14" preserveAspectRatio="none" aria-hidden="true" style={{ display: "block", width: "100%", height: 14 }}>
                  <rect x="0.5" y="0.5" width="249.5" height="13" fill="none" stroke={C.control} strokeWidth="1" vectorEffect="non-scaling-stroke" />
                  <rect
                    x="250"
                    y="0.5"
                    width="95"
                    height="13"
                    fill={C.gilt}
                    fillOpacity="0.3"
                    stroke={C.gilt}
                    strokeWidth="1"
                    vectorEffect="non-scaling-stroke"
                  />
                  <rect x="345" y="0.5" width="654.5" height="13" fill="none" stroke={C.rubric} strokeWidth="1" vectorEffect="non-scaling-stroke" />
                  <path d={STRIP_HATCH} fill="none" stroke={C.rubric} strokeWidth="1" vectorEffect="non-scaling-stroke" />
                </svg>
                <div
                  aria-hidden="true"
                  style={{ position: "relative", height: 28, fontSize: 14, fontVariantCaps: "all-small-caps", letterSpacing: "0.16em" }}
                >
                  <span style={{ position: "absolute", left: "12.5%", top: 4, transform: "translateX(-50%)", color: C.muted, whiteSpace: "nowrap" }}>
                    gentle
                  </span>
                  <span
                    style={{ position: "absolute", left: "29.75%", top: 4, transform: "translateX(-50%)", color: C.giltText, whiteSpace: "nowrap" }}
                  >
                    strong
                  </span>
                  <span style={{ position: "absolute", left: "67.25%", top: 4, transform: "translateX(-50%)", color: C.rubric, whiteSpace: "nowrap" }}>
                    collapse
                  </span>
                </div>
              </div>
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "8px 12px" }}>
              <button type="button" onClick={() => setVTo(v - 10)} style={ghostBtn}>
                gentler
              </button>
              <button type="button" onClick={() => setVTo(v + 10)} style={ghostBtn}>
                stronger
              </button>
              <span aria-hidden="true" style={{ color: C.gilt, padding: "0 6px" }}>
                ◆
              </span>
              <button type="button" onClick={() => setVTo(100)} style={linkBtn}>
                set the suggested strength, ×0.01
              </button>
              <button type="button" onClick={() => setVTo(340)} style={linkBtn}>
                show where the old default sat
              </button>
            </div>
            <div
              aria-live="polite"
              style={{
                display: "flex",
                flexWrap: "wrap",
                alignItems: "flex-end",
                gap: "10px 28px",
                padding: "16px 0 18px",
                borderTop: `1px solid ${C.hairSoft}`,
                borderBottom: `1px solid ${C.hairSoft}`,
              }}
            >
              <div
                style={{
                  fontSize: 64,
                  lineHeight: 0.9,
                  fontWeight: 500,
                  color: C.numeral,
                  fontVariantNumeric: "lining-nums",
                  letterSpacing: "-0.01em",
                  whiteSpace: "nowrap",
                }}
              >
                {mult}
              </div>
              <div style={{ flex: "1 1 320px", minWidth: 0, display: "flex", flexDirection: "column", gap: 4 }}>
                <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "4px 12px" }}>
                  {zone === "gentle" ? (
                    <svg width="26" height="12" viewBox="0 0 26 12" aria-hidden="true">
                      <rect x="0.5" y="0.5" width="25" height="11" fill="none" stroke={C.muted} strokeWidth="1" />
                    </svg>
                  ) : null}
                  {zone === "strong" ? (
                    <svg width="26" height="12" viewBox="0 0 26 12" aria-hidden="true">
                      <rect x="0.5" y="0.5" width="25" height="11" fill={C.gilt} fillOpacity="0.3" stroke={C.gilt} strokeWidth="1" />
                    </svg>
                  ) : null}
                  {isCollapse ? (
                    <svg width="26" height="12" viewBox="0 0 26 12" aria-hidden="true">
                      <rect x="0.5" y="0.5" width="25" height="11" fill="none" stroke={C.rubric} strokeWidth="1" />
                      <path d="M2 12 L8 0 M8 12 L14 0 M14 12 L20 0 M20 12 L26 0" stroke={C.rubric} strokeWidth="1" />
                    </svg>
                  ) : null}
                  <span style={{ fontSize: 21, fontVariantCaps: "all-small-caps", letterSpacing: "0.2em", color: z.color }}>{z.name}</span>
                  <span style={{ fontSize: 16, color: C.muted }}>{words}</span>
                </div>
                <p style={{ margin: 0, fontSize: 17, fontStyle: z.textStyle, color: z.textColor, maxWidth: 640 }}>{z.text}</p>
              </div>
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "4px 26px", fontSize: 15, color: C.muted, fontVariantNumeric: "lining-nums tabular-nums" }}>
              <span>gentle · up to ×{GENTLE_MAX}</span>
              <span>
                strong · ×{GENTLE_MAX} to ×{EDGE}
              </span>
              <span style={{ color: C.rubric }}>collapse · beyond ×{EDGE}</span>
            </div>
            <Rubric>
              Strength is measured against the stream itself, so one setting means the same thing at every layer. Keep the needle in the clear
              part of the dial. The edges are rough: they come from one small test of six questions, with the model always taking its likeliest
              next word. <a href={href.lexicon("strength")}>What is the stream’s strength?</a>
            </Rubric>
            {isNeuron ? (
              <Rubric>
                The dial’s clear and gilded parts were found with two words pushing at five layers. The study did not measure a single neuron’s
                push, so read the dial here as a rough guide.
              </Rubric>
            ) : null}
          </div>

          <aside aria-labelledby="push-head" style={{ flex: "1 1 340px", minWidth: 0, display: "flex", flexDirection: "column", gap: 12, paddingTop: 8 }}>
            <h3 className="h-sub" id="push-head">
              what the push amounts to
            </h3>
            <div style={{ overflowX: "auto" }}>
              <div style={{ minWidth: 330 }}>
                <div
                  style={{
                    ...pushGrid,
                    padding: "0 0 6px",
                    borderBottom: `1px solid ${C.hair}`,
                    fontSize: 13,
                    fontVariantCaps: "all-small-caps",
                    letterSpacing: "0.14em",
                    color: C.muted,
                  }}
                >
                  <span>layer</span>
                  <span style={{ textAlign: "right" }}>stream strength</span>
                  <span style={{ textAlign: "right" }}>this nudge adds</span>
                  <span style={{ textAlign: "right" }}>the old 650 was</span>
                </div>
                {rows.map((r) => {
                  const size = r.recorded ? 17 : 13;
                  const hColor = r.recorded ? C.ink : C.muted;
                  const oldColor = r.recorded ? C.rubric : C.muted;
                  return (
                    <div
                      key={r.layer}
                      style={{
                        ...pushGrid,
                        alignItems: "baseline",
                        padding: "9px 0",
                        borderBottom: `1px solid ${C.hairFaint}`,
                        fontSize: 17,
                        fontVariantNumeric: "lining-nums tabular-nums",
                      }}
                    >
                      <span style={{ color: C.giltText }}>{r.layer}</span>
                      <span style={{ textAlign: "right", fontSize: size, color: hColor }}>{r.h}</span>
                      <span style={{ textAlign: "right", fontSize: size, color: hColor }}>{r.push}</span>
                      <span style={{ textAlign: "right", fontSize: size, color: oldColor }}>{r.old}</span>
                    </div>
                  );
                })}
              </div>
            </div>
            {blocked ? <p style={{ margin: 0, fontSize: 16, color: C.muted }}>{blockedAside}</p> : null}
            {liveH.length ? (
              <p className="num" style={{ margin: 0, fontSize: 15, color: C.muted }}>
                This sighting’s own stream strength, at the word-piece it read:{" "}
                {liveH.map((x) => "layer " + x.n + ", " + x.h.toFixed(1)).join(" · ")}.
              </p>
            ) : null}
            <p style={{ margin: 0, fontSize: 15, color: C.muted }}>
              Stream strength <span style={{ fontStyle: "italic" }}>(‖h‖)</span> is the stream’s median size at that layer over six test
              questions, measured in the tool audit. Layers the audit did not measure are filled in by a sighting.
            </p>
            <Rubric>
              The old instrument added a fixed 650 at every chosen layer, whatever the stream’s strength. At layer 54, where the stream is only
              88 strong, that was more than seven times the stream.
            </Rubric>
          </aside>
        </div>

        {/* apparatus · the strength */}
        <Apparatus title="the strength" label="Apparatus for the strength" style={{ marginTop: 8 }}>
          <dl className="dl" style={{ gap: "7px 18px" }}>
            <dt>the dial</dt>
            <dd className="num">
              s = 10<sup>v/100 − 3</sup>, with v from 0 to 400 in steps of 5: from ×0.001 to ×10 of ‖h<sub>ℓ</sub>‖, a tenfold step every 100.
              Now v = {String(Math.round(v))}, s = {s.toPrecision(3)}.
            </dd>
            <dt>
              <label htmlFor="exact">set it exactly</label>
            </dt>
            <dd style={{ margin: 0, display: "flex", flexWrap: "wrap", alignItems: "center", gap: "4px 14px" }}>
              <input id="exact" type="number" min="0.001" max="10" step="0.001" value={sExact} onChange={onExact} style={smallInput({ width: 130 })} />
              <span style={{ color: C.muted }}>a share of ‖h‖, from 0.001 to 10</span>
            </dd>
            <dt>
              <label htmlFor="length">or the length sent</label>
            </dt>
            <dd style={{ margin: 0, display: "flex", flexWrap: "wrap", alignItems: "center", gap: "4px 14px" }}>
              <input
                id="length"
                type="number"
                min="0"
                step="0.1"
                value={lenValue}
                onChange={onLength}
                disabled={ref === null}
                style={smallInput({ width: 130 })}
              />
              {ref !== null ? (
                <span style={{ flex: "1 1 260px", minWidth: 0, color: C.muted, fontVariantNumeric: "lining-nums tabular-nums" }}>
                  the absolute scale the server receives, as the old screen’s slider set it (50 to 800): s = length ÷ {ref.h.toFixed(1)}, the
                  median ‖h‖ at layer {ref.layer}, so at most {scaleFmt(10 * ref.h)}. 650 sends the old default length.
                </span>
              ) : (
                <span style={{ flex: "1 1 260px", minWidth: 0, color: C.muted }}>
                  waits for a chosen layer whose ‖h‖ is recorded: <Missing />
                </span>
              )}
            </dd>
            <dt>stream strength</dt>
            <dd style={{ margin: 0, display: "flex", flexDirection: "column", gap: 6 }}>
              <div style={{ overflowX: "auto" }}>
                <div style={{ minWidth: 560, fontVariantNumeric: "lining-nums tabular-nums" }}>
                  <div
                    style={{
                      ...streamGrid,
                      padding: "0 0 4px",
                      borderBottom: `1px solid ${C.hair}`,
                      fontSize: 13,
                      fontVariantCaps: "all-small-caps",
                      letterSpacing: "0.12em",
                      color: C.muted,
                    }}
                  >
                    <span>layer</span>
                    <span style={{ textAlign: "right" }}>median ‖h‖</span>
                    <span style={{ textAlign: "right" }}>10th pct</span>
                    <span style={{ textAlign: "right" }}>90th pct</span>
                    <span style={{ textAlign: "right" }}>start mark</span>
                    <span style={{ textAlign: "right" }}>650 ÷ median</span>
                  </div>
                  {STREAM_TABLE.map((r) => (
                    <div key={r.layer} style={{ ...streamGrid, padding: "4px 0", borderBottom: `1px solid ${C.hairFaint}` }}>
                      <span style={{ color: C.giltText }}>{r.layer}</span>
                      <span style={{ textAlign: "right" }}>{r.median}</span>
                      <span style={{ textAlign: "right" }}>{r.p10}</span>
                      <span style={{ textAlign: "right" }}>{r.p90}</span>
                      <span style={{ textAlign: "right" }}>{r.bos}</span>
                      <span style={{ textAlign: "right" }}>{r.ratio}</span>
                    </div>
                  ))}
                </div>
              </div>
              <span style={{ color: C.muted }}>
                Over the word-pieces after the start mark of six test questions, three plain and three chats (repo index 29, 35, 41, 47, 53).{" "}
                <span className="mono">V7_tool_bugs_02_steer_p0.json</span>. Every other layer: <Missing />
              </span>
            </dd>
            <dt>as built</dt>
            <dd className="num">
              one absolute scale for every chosen layer: 650 in the server (<span className="mono">server.py:105</span>), 280 on the old screen,
              whose slider ran from 50 to 800 in steps of 10. The old screen pushed a neuron with the same slider: 280 at layer 54 is ×3.17 of the
              stream.
            </dd>
            <dt>the edges</dt>
            <dd className="num">
              The quiet-masked fnw direction at layers 30–54, 12 greedy word-pieces, six prompts. At scale 2, 0.8–2.3% of ‖h‖: 2 of 6 degenerate
              (the two plain prompts, bug 5), first-word KL 1.5 nats, target log-odds +12.0, a fluent “she” in 3 of 3 chats. At 6.5: 4 of 6
              degenerate, KL 12.6. At 20: 6 of 6, KL 16.3. The as-built direction at 2: 6 of 6, KL 15.6, she − he −6.1. V7 notes, part (c).
            </dd>
            <dt>caveats</dt>
            <dd>
              Six prompts and greedy decoding only. The edges hold for two words at five layers; for one neuron at one layer:{" "}
              <Missing what="not measured in the study" />. The study’s own persona direction, a difference of means after CAA (Panickssery et al.
              2024) [22], gave 16 of 16 “She is…” (or “He is…”) at ±2% of ‖h‖ in layers 26–40 (REPORT §1.3).
            </dd>
          </dl>
        </Apparatus>
      </section>

      {/* v · arm, then sight */}
      <section
        aria-labelledby="arm-head"
        style={{ marginTop: 40, borderTop: `1px solid ${C.hairSoft}`, paddingTop: 30, display: "flex", flexDirection: "column", gap: 22 }}
      >
        <h2 className="h-sec" id="arm-head">
          v · arm, then sight
        </h2>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "24px 56px", alignItems: "flex-start" }}>
          <div style={{ flex: "999 1 560px", minWidth: 0, display: "flex", flexDirection: "column", gap: 10 }}>
            {armed ? (
              <>
                <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                  <svg width="22" height="30" viewBox="0 0 22 30" aria-hidden="true">
                    <path d="M11 1 L21 15 L11 29 L1 15 Z" fill={C.rubric} />
                  </svg>
                  <span style={{ fontSize: 34, lineHeight: 1, fontVariantCaps: "all-small-caps", letterSpacing: "0.2em", color: C.rubric }}>armed</span>
                </div>
                <p style={{ margin: 0, fontSize: 18 }}>Every sighting is nudged until you release it.</p>
              </>
            ) : (
              <>
                <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                  <svg width="22" height="30" viewBox="0 0 22 30" aria-hidden="true">
                    <path d="M11 1 L21 15 L11 29 L1 15 Z" fill="none" stroke={C.muted} strokeWidth="1.4" />
                  </svg>
                  <span style={{ fontSize: 34, lineHeight: 1, fontVariantCaps: "all-small-caps", letterSpacing: "0.2em", color: C.muted }}>
                    disarmed
                  </span>
                </div>
                <p style={{ margin: 0, fontSize: 18 }}>Nothing is pushed. The model answers as itself.</p>
              </>
            )}
            <div style={{ display: "flex", flexWrap: "wrap", gap: "4px 20px", alignItems: "baseline", paddingTop: 6 }}>
              <span style={{ flex: "0 0 150px", ...labelSc }}>the nudge as set</span>
              <span style={{ flex: "1 1 260px", minWidth: 0, fontSize: 19, fontVariantNumeric: "lining-nums tabular-nums" }}>{setLine}</span>
            </div>
          </div>
          <div style={{ flex: "1 1 360px", minWidth: 0, display: "flex", flexDirection: "column", gap: 12, alignItems: "flex-start" }}>
            {isCollapse ? (
              <p style={{ margin: 0, fontStyle: "italic", fontSize: 17, color: C.rubric }}>
                The needle is past the edge. Armed at this strength, the nudge will not steer the answer; it will break it, as the example below
                shows. Turn it back into the clear part of the dial first.
              </p>
            ) : null}
            <div style={{ display: "flex", flexWrap: "wrap", gap: 12, alignItems: "center" }}>
              {!armed && !isCollapse ? (
                <button type="button" className="btn-primary" onClick={sight} disabled={cantArm}>
                  arm the nudge
                </button>
              ) : null}
              {!armed && isCollapse ? (
                <button type="button" onClick={sight} disabled={cantArm} style={outlineBtn(C.rubric, cantArm)}>
                  arm anyway, knowing it breaks
                </button>
              ) : null}
              {armed ? (
                <>
                  <button className="btn-primary" type="button" onClick={sight} disabled={cantArm || isSweeping}>
                    take a sighting
                  </button>
                  <button type="button" onClick={release} style={outlineBtn(C.ink)}>
                    release the nudge
                  </button>
                </>
              ) : null}
            </div>
            {blocked ? <p style={{ margin: 0, fontSize: 16, color: C.muted }}>{blockedText}</p> : null}
            {scaleUnknown ? (
              <p style={{ margin: 0, fontSize: 16, color: C.muted }}>
                {unknownScaleText}
                <Missing />
              </p>
            ) : null}
            {isSweeping ? (
              <p className="status-line" role="status" style={{ margin: 0 }}>
                {plainFirst !== null
                  ? "Taking a sighting without the nudge first, to compare; the nudged one follows."
                  : "Taking a sighting…"}
              </p>
            ) : null}
            <SweepError />
          </div>
        </div>

        {/* apparatus · what is sent */}
        <Apparatus title="what is sent" label="Apparatus for what is sent">
          <dl className="dl" style={{ gap: "7px 18px" }}>
            <dt>route</dt>
            <dd>
              the WebSocket <span className="mono">/ws/theodolite</span>, action <span className="mono">"sweep"</span>, inside{" "}
              <span className="mono">payload.steering</span>: a SteeringLaserRequest within the SweepRequest, as{" "}
              <span className="mono-wrap">
                {"{"}"action": "sweep", "payload": {"{"}"prompt": {q(question)}, "steering": {"{"}…{"}"} {"}"} {"}"}
              </span>{" "}
              (<span className="mono">server.py:574–577</span>). None of the four MCP tools (inspect_pole, trace_downstream_readers,
              test_motif_hypothesis, query_larql_vault) takes a nudge.
            </dd>
            <dt>scale as sent</dt>
            <dd className="num">{scaleRule}</dd>
            <dt>ignored</dt>
            <dd>{ignored}</dd>
            <dt>asked for apart</dt>
            <dd>{asks.join("; ") + ". None is a field of the request yet."}</dd>
            <dt>leaves no trace</dt>
            <dd>
              In the audit the model carried no hooks before or after a nudged run, and the plain spider question came back word for word as
              before (V7, bug 5).
            </dd>
          </dl>
          <div role="group" aria-label="Which request to show" style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "8px 10px", paddingTop: 4 }}>
            <span style={{ fontVariantCaps: "all-small-caps", letterSpacing: "0.14em", color: C.muted, marginRight: 6 }}>the request</span>
            <button type="button" className="btn-ghost" onClick={() => setView("page")} aria-pressed={view === "page"}>
              from this page
            </button>
            <button type="button" className="btn-ghost" onClick={() => setView("old")} aria-pressed={view === "old"}>
              from the old screen
            </button>
            <button type="button" className="btn-ghost" onClick={() => setView("server")} aria-pressed={view === "server"}>
              the server’s defaults
            </button>
          </div>
          <span style={{ color: C.muted }}>{NOTES[view]}</span>
          <pre className="raw" style={{ maxHeight: "none" }}>
            {BODIES[view]}
          </pre>
        </Apparatus>

        <Rubric style={{ maxWidth: 860 }}>
          A nudge lasts only while it is armed, and it changes nothing in the model itself: in the study, after a nudged sighting, all six test
          questions were answered exactly as before. Release it when you are done.
        </Rubric>

        <div style={{ display: "flex", flexWrap: "wrap", gap: "32px 56px", paddingTop: 6 }}>
          {/* as it was */}
          <div
            style={{
              flex: "1 1 440px",
              minWidth: 0,
              display: "flex",
              flexDirection: "column",
              gap: 10,
              borderTop: `1px solid ${C.hair}`,
              paddingTop: 16,
            }}
          >
            <div className="apparatus-head">
              <h3 className="h-sub">as it was</h3>
              <span style={{ fontSize: 15, fontStyle: "italic", color: C.muted }}>no nudge</span>
            </div>
            {plainNow ? (
              <>
                <Answer text={plainNow.continuation} />
                <p style={{ margin: 0, fontSize: 15, fontStyle: "italic", color: C.muted, maxWidth: 520 }}>
                  From a live sighting, without the nudge.
                  {plainNow.continuation.includes("\n") ? " ¶ marks where it began a new line." : ""}
                </p>
              </>
            ) : recordedPlain ? (
              <>
                <Answer text={RECORDED_PLAIN_SPIDER} />
                <p style={{ margin: 0, fontSize: 15, fontStyle: "italic", color: C.muted, maxWidth: 520 }}>
                  It answers 8, then starts the question over: with no chat template, the model treats your words as text to continue. ¶ marks
                  where it began a new line.
                </p>
                <p style={{ margin: 0, fontSize: 14, color: C.muted }}>
                  From the tool audit (V7), with the model always taking its likeliest word.
                </p>
              </>
            ) : (
              <div style={placeholderBox}>[TAKE A SIGHTING TO SEE THE PLAIN ANSWER]</div>
            )}
          </div>

          {/* as nudged */}
          <div
            style={{
              flex: "1 1 440px",
              minWidth: 0,
              display: "flex",
              flexDirection: "column",
              gap: 10,
              borderTop: `1px solid ${C.hair}`,
              paddingTop: 16,
            }}
          >
            <div className="apparatus-head">
              <h3 className="h-sub">as nudged</h3>
              <span style={{ fontSize: 15, fontStyle: "italic", color: C.muted }}>{nudgedNote}</span>
            </div>
            {nudgedNow ? (
              <>
                <Answer text={nudgedNow.continuation} />
                <p style={{ margin: 0, fontSize: 15, fontStyle: "italic", color: C.muted, maxWidth: 520 }}>
                  From a live sighting, with the nudge armed.
                  {nudgedNow.continuation.includes("\n") ? " ¶ marks where it began a new line." : ""}
                </p>
              </>
            ) : !isCollapse ? (
              <div style={placeholderBox}>[TAKE A SIGHTING TO SEE THE NUDGED ANSWER]</div>
            ) : null}
            {!isCollapse && !armed ? (
              <p style={{ margin: 0, fontSize: 16, color: C.muted }}>Nothing is armed, so a sighting now would answer as it was.</p>
            ) : null}
            {!isCollapse ? (
              <p style={{ margin: 0, fontStyle: "italic", fontSize: 16, color: C.rubric, maxWidth: 560 }}>
                Be warned: on a plain question like this one, even a careful nudge may break the answer. In the study, a corrected nudge of
                strength 2, about a hundredth of the stream (toward “she”), read fluently inside three chats, yet turned this very question’s
                answer into “own same same same …”. The instrument sends no chat template yet.
              </p>
            ) : (
              <>
                <span style={{ fontSize: 14, fontVariantCaps: "all-small-caps", letterSpacing: "0.2em", color: C.rubric }}>
                  an example, not your nudge
                </span>
                <p style={{ margin: 0, fontSize: 16, color: C.muted }}>
                  What the old defaults did to this very question: toward “six”, away from “eight”, a push of 650 at layers 30, 36, 42, 48 and
                  54.
                </p>
                <div style={{ fontFamily: MONO, fontSize: 16, lineHeight: 1.6, color: C.ink, wordBreak: "break-all" }}>
                  “&lt;|tool_response&gt;&lt;|tool_response&gt;&lt;|tool_response&gt; …”
                </div>
                <p style={{ margin: 0, fontSize: 15, fontStyle: "italic", color: C.muted }}>
                  one of the model’s control marks, not a word at all, twelve times over
                </p>
                <p style={{ margin: "6px 0 0", fontSize: 16, color: C.muted }}>A random direction at the same strength gave:</p>
                <div style={{ fontFamily: MONO, fontSize: 16, lineHeight: 1.6, color: C.ink, wordBreak: "break-all" }}>
                  “InstrumentedTestInstrumentedTestInstrumentedTest …”
                </div>
                <p style={{ margin: 0, fontSize: 15, fontStyle: "italic", color: C.muted }}>
                  a word from software test code, six times over: strength alone is enough to break it
                </p>
                <p style={{ margin: "4px 0 0", fontSize: 14, color: C.muted }}>
                  From the tool audit (V7, bug 1, and its follow-up check), with the model always taking its likeliest word.
                </p>
              </>
            )}
          </div>
        </div>
      </section>

      <div aria-hidden="true" style={{ display: "flex", alignItems: "center", gap: 18, margin: "52px 0 0" }}>
        <div style={{ flex: "1 1 0", height: 1, background: C.hair }} />
        <svg width="14" height="18" viewBox="0 0 14 18">
          <path d="M7 1 L13 9 L7 17 L1 9 Z" fill={C.gilt} />
        </svg>
        <div style={{ flex: "1 1 0", height: 1, background: C.hair }} />
      </div>

      {/* vi · why the old nudge broke */}
      <section aria-labelledby="why-head" style={{ paddingTop: 30, display: "flex", flexDirection: "column", gap: 20 }}>
        <div className="row-between">
          <h2 className="h-sec" id="why-head">
            vi · why the old nudge broke
          </h2>
          <a href={href.lexicon("nudge-defaults")} style={{ fontSize: 16, padding: "10px 0" }}>
            read more in the lexicon →
          </a>
        </div>
        <Rubric style={{ maxWidth: 860 }}>
          The instrument shipped with a nudge that breaks the model rather than steering it. Here is why, in three parts, so you can recognise it
          if you meet it.
        </Rubric>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "40px 56px" }}>
          <article style={{ flex: "1 1 340px", minWidth: 0, display: "flex", flexDirection: "column", gap: 12 }}>
            <Headline figure="7.4×" note="the stream’s strength, at layer 54" />
            <h3 className="h-sub">it pushed too hard</h3>
            <p style={{ margin: 0, fontSize: 17 }}>
              It added a push of 650 at each of five layers. The stream there is only 88 to 269 strong, so the push was 2.4 to 7.4 times the
              stream itself. All six test questions came out as junk. The screen’s own default of 280 was past the edge too, and so was the old
              slider’s gentlest setting, 50.
            </p>
            <div
              role="img"
              aria-label="Stream strength against the old push of 650. Layer 30: 237, the push 2.7 times the stream. Layer 36: 269, 2.4 times. Layer 42: 175, 3.7 times. Layer 48: 142, 4.6 times. Layer 54: 88, 7.4 times."
              style={{ display: "flex", flexDirection: "column", gap: 9, paddingTop: 4 }}
            >
              {[
                [30, "36.5%", "×2.7"],
                [36, "41.4%", "×2.4"],
                [42, "26.9%", "×3.7"],
                [48, "21.8%", "×4.6"],
                [54, "13.5%", "×7.4"],
              ].map(([n, w, x]) => (
                <div
                  key={n}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "64px minmax(0, 1fr) 44px",
                    gap: "4px 12px",
                    alignItems: "center",
                    fontSize: 15,
                    fontVariantNumeric: "lining-nums tabular-nums",
                  }}
                >
                  <span style={{ color: C.muted }}>layer {n}</span>
                  <div style={{ height: 3, background: C.hairFaint, position: "relative" }}>
                    <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: w, background: C.gilt }} />
                    <div style={{ position: "absolute", right: 0, top: -4, width: 2, height: 11, background: C.rubric }} />
                  </div>
                  <span style={{ textAlign: "right" }}>{x}</span>
                </div>
              ))}
              <span style={{ fontSize: 14, fontStyle: "italic", color: C.muted }}>
                gilt: the stream’s own strength; the red mark at the end: the old push of 650
              </span>
            </div>
          </article>
          <article style={{ flex: "1 1 340px", minWidth: 0, display: "flex", flexDirection: "column", gap: 12 }}>
            <Headline figure="79%" note="of its aim, in three dimensions" />
            <h3 className="h-sub">it pointed the wrong way</h3>
            <p style={{ margin: 0, fontSize: 17 }}>
              To aim the push, the old code divided by the model’s final norm weights. Three of the 5,376 weights are negative, and a safety clamp
              turned them into the largest multipliers: 10, against a typical 0.16. So 79% of the aim’s energy sat in those three dimensions,
              before it was carried back to each layer. Even at a strength of 2, about a hundredth of the stream, this direction broke all six
              test questions, and moved “she” against “he” the wrong way.
            </p>
            <div
              role="img"
              aria-label="79 percent of the old direction’s energy lies in 3 dimensions; 21 percent in the other 5,373."
              style={{ display: "flex", flexDirection: "column", gap: 6, paddingTop: 4 }}
            >
              <div style={{ height: 3, background: C.hairFaint, position: "relative" }}>
                <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: "79%", background: C.gilt }} />
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", gap: 12, fontSize: 15, fontVariantNumeric: "lining-nums tabular-nums" }}>
                <span>
                  <span style={{ color: C.giltText, fontWeight: 600 }}>79%</span> in the 3 clamped dimensions
                </span>
                <span style={{ textAlign: "right", color: C.muted }}>21% in the other 5,373</span>
              </div>
            </div>
          </article>
          <article style={{ flex: "1 1 340px", minWidth: 0, display: "flex", flexDirection: "column", gap: 12 }}>
            <Headline figure="1,700×" note="the read gain on quiet dimensions, at most" />
            <h3 className="h-sub">it fell into the quiet carry channel</h3>
            <p style={{ margin: 0, fontSize: 17 }}>
              About 180 dimensions are among the loudest in the early layers. Later the model stops writing to them, yet reads them with gains of
              up to 1,700 times, carrying early content forward. Every direction read off a lens or a gradient gathers there, which is why
              lens-derived nudges break Gemma 4.
            </p>
            <div
              role="img"
              aria-label="How far the next word is thrown, in nats: a 30 percent push in the loud dimensions, about 0.1; a 1 percent push in the quiet dimensions, 10 to 11."
              style={{ display: "flex", flexDirection: "column", gap: 9, paddingTop: 4 }}
            >
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "minmax(0, 1fr) 54px",
                  gap: "4px 12px",
                  alignItems: "baseline",
                  fontSize: 15,
                  fontVariantNumeric: "lining-nums tabular-nums",
                }}
              >
                <span style={{ color: C.muted }}>a 30% push in the loud dimensions</span>
                <span style={{ textAlign: "right" }}>0.1</span>
                <div style={{ gridColumn: "1 / -1", height: 3, background: C.hairFaint, position: "relative" }}>
                  <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: "0.9%", background: C.gilt }} />
                </div>
              </div>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "minmax(0, 1fr) 54px",
                  gap: "4px 12px",
                  alignItems: "baseline",
                  fontSize: 15,
                  fontVariantNumeric: "lining-nums tabular-nums",
                }}
              >
                <span style={{ color: C.muted }}>a 1% push in the quiet dimensions</span>
                <span style={{ textAlign: "right" }}>10–11</span>
                <div style={{ gridColumn: "1 / -1", height: 3, background: C.hairFaint, position: "relative" }}>
                  <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: "90.9%", background: C.gilt }} />
                  <div style={{ position: "absolute", left: "90.9%", top: 1, height: 1, width: "9.1%", background: C.gilt }} />
                </div>
              </div>
              <span style={{ fontSize: 14, fontStyle: "italic", color: C.muted }}>
                how far the next word is thrown, in nats, a measure of surprise; the thin end spans 10 to 11
              </span>
            </div>
          </article>
        </div>

        <div style={{ display: "flex", flexWrap: "wrap", gap: "20px 56px", paddingTop: 18, borderTop: `1px solid ${C.hairSoft}` }}>
          <div style={{ flex: "999 1 560px", minWidth: 0, display: "flex", flexDirection: "column", gap: 10 }}>
            <h3 className="h-sub">how this page nudges instead</h3>
            <p style={{ margin: 0, fontSize: 18, maxWidth: 760 }}>
              Strength is a share of the stream, not a fixed number. The direction is taken from the norm weights themselves, with no inverse and
              no clamp; the quiet channel is masked; and the start mark <span style={{ fontStyle: "italic", color: C.muted }}>(BOS)</span> is left
              alone. These are the tool audit’s fixes. The window that works is narrow, which is why the dial shows it.
            </p>
            <p style={{ margin: 0, fontSize: 17, fontStyle: "italic", color: C.rubric, maxWidth: 760 }}>
              Not yet in the server. Until the fix lands, the backend still builds the old direction at a fixed strength, whatever this page
              shows.
            </p>
          </div>
          <div style={{ flex: "1 1 360px", minWidth: 0, display: "flex", flexDirection: "column", gap: 8 }}>
            <h3 className="h-sub">done right, it works</h3>
            <p style={{ margin: 0, fontSize: 17 }}>
              With the quiet channel masked, a push of 2 at the same five layers, about a hundredth of the stream, turned a chat answer into
            </p>
            <p style={{ margin: 0, fontSize: 24, fontStyle: "italic", lineHeight: 1.3 }}>“she is a thoughtful and creative collaborator …”</p>
            <p style={{ margin: 0, fontSize: 16, color: C.muted }}>
              At 6.5, two to seven hundredths of the stream, the same direction broke four answers in six.
            </p>
          </div>
        </div>

        <Apparatus title="the quiet carry channel" label="Apparatus for the audit">
          <dl className="dl num" style={{ gap: "7px 18px" }}>
            <dt>read gain</dt>
            <dd>
              the largest weights fall on the persistent quiet dimensions: input_layernorm 1,248, pre_feedforward_layernorm 860, the final norm
              512; there the read gain runs 260–1,700 times the median, at layers 41–59 (V6 notes; REPORT §2.2). RMSNorm (Zhang &amp; Sennrich
              2019) [16].
            </dd>
            <dt>the quiet set</dt>
            <dd>
              181 dimensions deep-quiet at every layer from 43 to 58 (std below 0.05 × the layer median); among the loudest at layers 6–21 (97th
              percentile of std); written at about 0.002× after layer 26; their values at layer 46 still correlate r = 0.71 with their own at
              layers 21 and 26 (loud dimensions: 0.03). Neither attention-sink [20] nor massive-activation [19] dimensions.
            </dd>
            <dt>sensitivity</dt>
            <dd>
              a single-layer push, KL in nats over two seeds: at layer 41, 10.4 for 1% of ‖h‖ in the quietest tenth, 2.9 for 0.3% in the
              persistent set, 0.11 for 30% in loud dimensions; at layer 51, 10.9, 8.6 and 0.065. KL follows the gain-weighted size ‖w
              <sub>in, ℓ+1</sub> ⊙ δ‖ / rms(h) (R² 0.82–0.96), not the raw norm (0.02–0.58).
            </dd>
            <dt>the lens</dt>
            <dd>
              masking the quiet dimensions raises J++ ↔ true-gradient agreement at layers 41–59 from cos 0.26 / 0.33 to 0.84 / 0.86 (ceiling
              0.97–0.99): the lens’s late disagreement with the chat model lives in this subspace (REPORT §2.2).
            </dd>
            <dt>related</dt>
            <dd>Briggs 2026 [26] notes, independently, that the skewed final gain makes plain Euclidean influence misleading.</dd>
          </dl>
        </Apparatus>
      </section>
    </Page>
  );
};

/** A large gilt figure with its note, heading each of the three parts of section vi. */
const Headline: React.FC<{ figure: string; note: string }> = ({ figure, note }) => (
  <div style={{ display: "flex", alignItems: "flex-end", gap: 14, borderBottom: `1px solid ${C.hair}`, paddingBottom: 12 }}>
    <span style={{ fontSize: 60, lineHeight: 0.85, fontWeight: 500, color: C.numeral, fontVariantNumeric: "lining-nums" }}>{figure}</span>
    <span style={{ fontSize: 15, color: C.muted, paddingBottom: 2 }}>{note}</span>
  </div>
);

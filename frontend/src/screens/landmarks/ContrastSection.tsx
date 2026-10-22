import React from "react";
import { Apparatus, Missing, Rubric } from "../../components/kit";
import { isBaseFitted, lensPhrase, useReadingLens } from "../../lib/lens";
import { signed } from "../../lib/words";
import {
  type ContrastBackground,
  type ContrastReading,
  type ContrastWord,
  type Sighting,
  CONTRAST_MAX_WORDS,
  contrastProblem,
  contrastWords,
  useTheodoliteStore,
} from "../../store/useTheodoliteStore";
import { cxL, f1 } from "./geometry";

/*
 * Landmarks section vii: one word against another at every layer (the sweep's `contrasts`, read by
 * TheodoliteRuntime._contrast_readings). The chart draws what the sighting shown was read with, never the boxes as
 * set; the boxes go with the next sighting, and only when the section is on with words typed both sides.
 */

const q = (w: string) => "“" + w + "”";
/** A resolved word-piece with its leading space shown, as the server read it. */
const pieceShown = (p: string) => p.replace(/^ /, "␣").replace(/\n/g, "↵");
const sideWords = (ws: ContrastWord[]) => ws.map((w) => q(w.word.trim())).join(" · ");
const nats = (v: number | null | undefined) => (v === null || v === undefined ? null : signed(v, 2));

/** The layer bands the study's errata draw for a contrast read through an average-Jacobian lens, as page layers 1-60
 * (report layers L0-9, L10-29, L30-49, L50-58; L59, page 60, is the model itself): about the lens line only. */
const BANDS: { from: number; to: number; red: boolean; tag: string; words: string }[] = [
  { from: 1, to: 10, red: false, tag: "1–10", words: "Not tested by the study." },
  { from: 11, to: 30, red: true, tag: "11–30 · E50", words: "No averaged lens tested was reliable here (E50)." },
  {
    from: 31,
    to: 50,
    red: false,
    tag: "31–50 · E49",
    words: "To be confirmed with an instruct-fitted reading, which the instrument does not load (E49).",
  },
  {
    from: 51,
    to: 59,
    red: true,
    tag: "51–59 · E49, E50",
    words: "Pronoun readings here count only where an instruct-fitted reading agrees, and none is loaded (E49, E50).",
  },
];

/** A round bound for the chart's scale, in nats: 1, 2, 5, 10, 20, … */
function niceBound(v: number): number {
  if (!(v > 0)) return 1;
  const e = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 5, 10]) if (v <= m * e) return m * e;
  return 10 * e;
}

const linePath = (vals: number[] | null, y: (v: number) => number) =>
  vals ? vals.map((v, l) => (l === 0 ? "M" : " L") + cxL(l) + "," + f1(y(v))).join("") : "";

/** What a background reads, for the number row and the chart's red line. */
function backgroundSeries(c: ContrastReading, b: ContrastBackground | null) {
  if (b === "controls" && c.controls) return { lens: c.controls.deltaLens, naked: c.controls.deltaNaked };
  if (b === "study_mean" && c.studyMean) return { lens: c.studyMean.centredLens, naked: c.studyMean.centredNaked };
  return null;
}

/** Where the controls were read, from the server's own kind. */
function controlWhere(kind: string): string {
  switch (kind) {
    case "control_prompt_end":
      return "read where the control chose its own first word-piece";
    case "control_reading_same_answer":
      return "read with the control reading this very answer, up to the same word-piece";
    default:
      return "not read";
  }
}

export const ContrastSection: React.FC<{
  sighting: Sighting | null;
  li: number;
  /** the first two landmarks, the boxes' placeholders */
  placeholders: [string, string];
  /** a matched control is kept that the next sighting can be centred on */
  canCentre: boolean;
  /** what the next sighting will send, from the store's own sweepPayload */
  sent: { contrasts: { toward: string[]; away: string[] }[]; contrast_backgrounds: string[] };
  /** read the answer shown again (or take a new sighting), as the page's own button does */
  onRead: () => void;
  readLabel: string;
}> = ({ sighting, li, placeholders, canCentre, sent, onRead, readLabel }) => {
  const contrast = useTheodoliteStore((s) => s.contrast);
  const setContrast = useTheodoliteStore((s) => s.setContrast);
  const setFrame = useTheodoliteStore((s) => s.setFrame);
  const isSweeping = useTheodoliteStore((s) => s.isSweeping);
  const health = useTheodoliteStore((s) => s.health);
  const { lens } = useReadingLens(sighting);

  // the server's word on the study's average state: unknown before its health is read; an older server reads no contrasts
  const card = health?.contrasts ?? null;
  const smProblem = card ? card.study_mean_problem : null;
  const smOk = !!card && smProblem === null;
  const bg: ContrastBackground | null = contrast.backgrounds[0] ?? null;
  const problem = contrastProblem(contrast);
  const sentBgs = sent.contrast_backgrounds;
  const turnOn = (on: boolean) => {
    // the placeholders become words typed only when the reader turns the section on, so nothing unseen is ever sent
    if (on && !contrast.toward.trim() && !contrast.away.trim()) setContrast({ on, toward: placeholders[0], away: placeholders[1] });
    else setContrast({ on });
  };
  const choose = (b: ContrastBackground | null) => {
    setContrast({ backgrounds: b ? [b] : [] });
    // the matched control goes with the request as the frame's centring, so reading against it centres the frame too
    if (b === "controls") setFrame({ centre: true });
  };

  /* ---- what the sighting shown was read with */
  const list = sighting?.contrasts ?? null;
  const c = list?.find((x) => x.error === null && x.lens !== null) ?? null;
  const failed = list?.filter((x) => x.error !== null) ?? [];
  const shownBg: ContrastBackground | null = c ? (c.controls ? "controls" : c.studyMean ? "study_mean" : null) : null;
  const bgSeries = c ? backgroundSeries(c, shownBg) : null;
  const aName = c ? sideWords(c.toward) : "";
  const bName = c ? sideWords(c.away) : "";
  const pairs = c ? c.toward.length : 0;
  // the words the rubric names: the sighting's own when it weighed some, else those the boxes will send
  const typed = contrastWords(contrast.toward)[0] ?? placeholders[0];
  const typedAway = contrastWords(contrast.away)[0] ?? placeholders[1];
  const first = c ? (pairs === 1 ? aName : "the first words") : typed ? q(typed) : "the first word";
  const second = c ? (pairs === 1 ? bName : "the second") : typedAway ? q(typedAway) : "the second";

  /* ---- the chart: 60 layers in nats, symmetric about nought */
  const plotted = c ? [c.lens, c.naked, bgSeries?.lens ?? null].filter((v): v is number[] => v !== null) : [];
  const bound = niceBound(Math.max(0, ...plotted.flat().map((v) => Math.abs(v))));
  const y = (v: number) => 200 - (Math.max(-bound, Math.min(bound, v)) * 150) / bound;
  const half = 1060 / 59 / 2;
  const at = (vals: number[] | null | undefined, l: number) => (vals ? vals[l] : null);
  const bgWords =
    shownBg === "controls"
      ? "against the matched control"
      : shownBg === "study_mean"
        ? "centred on the study’s average state"
        : null;

  const row = (l: number) =>
    c ? (
      <p style={{ margin: 0, fontSize: "17px", fontVariantNumeric: "lining-nums tabular-nums", maxWidth: "900px" }}>
        <span style={{ fontVariantCaps: "all-small-caps", letterSpacing: "0.12em", color: "#6B5D4D" }}>at layer {l + 1}</span>{" "}
        {nats(at(c.lens, l)) ?? <Missing what="not recorded in this sighting" />} through the lens ·{" "}
        {nats(at(c.naked, l)) ?? <Missing what="not recorded in this sighting" />} to the naked eye
        {shownBg === "controls" ? (
          <>
            {" "}
            · {nats(at(bgSeries?.lens, l)) ?? <Missing what="no control read here" />} against the control
          </>
        ) : null}
        {shownBg === "study_mean" ? (
          <>
            {" "}
            · {nats(at(bgSeries?.lens, l)) ?? <Missing what="not recorded in this sighting" />} centred on the study’s average state
          </>
        ) : null}
        <span style={{ color: "#6B5D4D" }}>
          {" "}
          ({shownBg === "study_mean" ? "the first two in nats of log-odds; the third is the score of the state less the average, as E3 centred it: not a log-odds the model gives" : "nats of log-odds"})
        </span>
      </p>
    ) : null;

  const radio = (b: ContrastBackground | null, label: React.ReactNode, enabled: boolean, note?: React.ReactNode) => (
    <label
      style={{
        display: "flex",
        alignItems: "flex-start",
        gap: "10px",
        minHeight: "44px",
        paddingTop: "10px",
        cursor: enabled ? "pointer" : "not-allowed",
        color: enabled ? "#2B2117" : "#6B5D4D",
      }}
    >
      <input
        type="radio"
        name="contrast-background"
        checked={bg === b}
        disabled={!enabled}
        onChange={() => choose(b)}
        style={{ width: "17px", height: "17px", accentColor: "#8F2B1E", margin: "3px 0 0", flex: "none" }}
      />
      <span>
        {label}
        {note ? <span style={{ display: "block", fontSize: "14px", fontStyle: "italic", color: "#6B5D4D" }}>{note}</span> : null}
      </span>
    </label>
  );

  const box = (id: string, value: string, ph: string, set: (v: string) => void) => (
    <input
      id={id}
      type="text"
      value={value}
      placeholder={ph}
      onChange={(e) => set(e.target.value)}
      autoComplete="off"
      spellCheck={false}
      disabled={!contrast.on}
      style={{
        fontFamily: "inherit",
        fontStyle: "italic",
        fontSize: "28px",
        color: "#2B2117",
        background: "transparent",
        border: "0",
        borderBottom: "1px solid #C9BEAC",
        padding: "0 4px 2px",
        width: "7em",
        maxWidth: "100%",
        boxSizing: "border-box",
        textAlign: "center",
      }}
    />
  );

  return (
    <section aria-labelledby="contrast-head" style={{ marginTop: "48px", display: "flex", flexDirection: "column", gap: "14px" }}>
      <h2 className="h-sec" id="contrast-head">
        vii · one word against another
      </h2>
      <Rubric style={{ maxWidth: "860px" }}>
        Here two words are weighed against each other at every layer, with no frame to square. Above the line the lens leans
        toward {first}; below it, toward {second}. At layer 60 the lens and the naked eye are the model itself: the line there
        is how much likelier it makes {first} than {second} as its next word-piece, in nats (for one pair, +1 is about 2.7
        times as likely).
      </Rubric>

      {/* ---- what the next sighting sends */}
      <label style={{ display: "flex", alignItems: "center", gap: "10px", minHeight: "44px", fontSize: "18px", cursor: "pointer" }}>
        <input
          type="checkbox"
          checked={contrast.on}
          onChange={(e) => turnOn(e.target.checked)}
          style={{ width: "17px", height: "17px", accentColor: "#8F2B1E", margin: 0 }}
        />
        weigh one word against another in the next sighting
      </label>
      <div style={{ fontSize: "28px", lineHeight: 1.8, maxWidth: "900px", color: contrast.on ? "#2B2117" : "#6B5D4D" }}>
        <label htmlFor="contrast-toward">Weigh</label>{" "}
        {box("contrast-toward", contrast.toward, placeholders[0], (v) => setContrast({ toward: v }))}{" "}
        <label htmlFor="contrast-away">against</label>{" "}
        {box("contrast-away", contrast.away, placeholders[1], (v) => setContrast({ away: v }))}.
      </div>
      <fieldset style={{ border: "0", margin: "0", padding: "0", display: "flex", flexDirection: "column", maxWidth: "900px" }}>
        <legend style={{ padding: "6px 0 0", fontVariantCaps: "all-small-caps", letterSpacing: "0.14em", color: "#6B5D4D" }}>
          read it against
        </legend>
        {radio(null, "nothing (as it reads)", contrast.on)}
        {radio(
          "controls",
          "the matched control kept (this centres the frame above too)",
          contrast.on && canCentre,
          canCentre ? undefined : "Keep a matched control first, in the question’s apparatus on the instrument page.",
        )}
        {radio(
          "study_mean",
          "the study’s average state (5,000 word-pieces of prose and chat, E3)",
          contrast.on && smOk,
          !health
            ? "Waiting for the server to say whether it holds the average state."
            : !card
              ? "This server reads no contrasts: it is older than this page."
              : smProblem
                ? `The server cannot read it: ${smProblem}`
                : undefined,
        )}
      </fieldset>
      {contrast.on && problem ? (
        <p role="status" style={{ margin: 0, fontStyle: "italic", fontSize: "17px", color: "#8F2B1E", maxWidth: "900px" }}>
          Nothing is weighed yet: {problem}
        </p>
      ) : null}
      {contrast.on && !problem && bg && !sentBgs.includes(bg) ? (
        <p role="status" style={{ margin: 0, fontStyle: "italic", fontSize: "17px", color: "#8F2B1E", maxWidth: "900px" }}>
          {bg === "controls"
            ? "The next sighting will not be read against the control: none goes with it. Keep a matched control, and centre on it in the frame’s options above."
            : "The next sighting will not be read against the study’s average state: the server cannot read it."}
        </p>
      ) : null}
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "10px 22px" }}>
        <button className="btn-primary" type="button" onClick={onRead} disabled={isSweeping || !contrast.on || problem !== null}>
          {isSweeping ? "taking a sighting…" : readLabel}
        </button>
        <span style={{ fontSize: "16px", color: "#6B5D4D" }}>
          {sent.contrasts.length
            ? `It will weigh ${sent.contrasts[0].toward.map(q).join(" · ")} against ${sent.contrasts[0].away.map(q).join(" · ")}${
                sentBgs.length
                  ? sentBgs.includes("controls")
                    ? ", and against the matched control"
                    : ", and centred on the study’s average state"
                  : ""
              }.`
            : "The next sighting weighs nothing."}
        </span>
      </div>

      {/* ---- what the sighting shown was read with */}
      {!sighting ? (
        <p style={{ margin: 0, fontSize: "17px", color: "#6B5D4D" }}>
          <Missing what="no sighting yet" />
        </p>
      ) : list === null ? (
        <p style={{ margin: 0, fontSize: "17px", color: "#6B5D4D", maxWidth: "900px" }}>
          {sighting.source === "fixture" ? (
            "The recorded sighting weighs no word against another."
          ) : (
            <>
              The server that took this sighting sent no contrasts: it is older than this page.{" "}
              <span className="placeholder">[NOT YET IN THE SERVER]</span>
            </>
          )}
        </p>
      ) : !c ? (
        <p style={{ margin: 0, fontSize: "17px", color: failed.length ? "#8F2B1E" : "#6B5D4D", maxWidth: "900px", fontStyle: failed.length ? "italic" : undefined }}>
          {failed.length
            ? `The sighting shown asked for a contrast it could not read. ${failed[0].error}`
            : "The sighting shown was taken without weighing one word against another. Turn it on above, and read the answer again."}
        </p>
      ) : (
        <>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 22px", fontSize: "15px", color: "#6B5D4D" }}>
            <span style={{ fontSize: "22px", fontStyle: "italic", color: "#2B2117" }}>
              {aName} against {bName}
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <svg width="22" height="10" viewBox="0 0 22 10" aria-hidden="true">
                <path d="M1 5 H21" stroke="#2B2117" strokeWidth="1.6" />
              </svg>
              through the lens
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <svg width="22" height="10" viewBox="0 0 22 10" aria-hidden="true">
                <path d="M1 5 H21" stroke="#6B5D4D" strokeWidth="1.4" strokeDasharray="4 3" />
              </svg>
              to the naked eye
            </span>
            {bgWords ? (
              <span style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <svg width="22" height="10" viewBox="0 0 22 10" aria-hidden="true">
                  <path d="M1 5 H21" stroke="#8F2B1E" strokeWidth="1.6" />
                </svg>
                through the lens, {bgWords}
              </span>
            ) : null}
          </div>
          <div style={{ overflowX: "auto", paddingBottom: "6px" }}>
            <div style={{ minWidth: "940px" }}>
              <svg
                viewBox="0 0 1280 420"
                role="img"
                aria-label={`A line chart across the sixty layers of the sighting shown: ${aName} weighed against ${bName}, in nats, from minus ${bound} to ${bound}; the lens as a solid line, the naked eye dashed${
                  bgWords ? `, and in red the lens ${bgWords}` : ""
                }. The chosen layer, ${li + 1}, is marked in red.`}
                style={{ width: "100%", height: "auto", display: "block" }}
              >
                {BANDS.map((b) => (
                  <rect
                    key={b.tag}
                    x={f1(cxL(b.from - 1) - half)}
                    y="44"
                    width={f1(cxL(b.to - 1) - cxL(b.from - 1) + 2 * half)}
                    height="312"
                    fill={b.red ? "#8F2B1E" : "#C9BEAC"}
                    fillOpacity={b.red ? 0.06 : 0.16}
                  />
                ))}
                <g fontFamily="EB Garamond, Garamond, serif" fontSize="12" fontStyle="italic" textAnchor="middle">
                  {BANDS.map((b) => (
                    <text key={b.tag} x={f1((cxL(b.from - 1) + cxL(b.to - 1)) / 2)} y="38" fill={b.red ? "#8F2B1E" : "#6B5D4D"}>
                      {b.tag}
                    </text>
                  ))}
                  <text x={cxL(59)} y="38" fill="#2B2117">
                    60
                  </text>
                </g>
                <path d="M80 50 H1140 M80 125 H1140 M80 275 H1140 M80 350 H1140" fill="none" stroke="#E3DACA" strokeWidth="1" />
                <path d="M80 200 H1140" fill="none" stroke="#2B2117" strokeOpacity="0.55" strokeWidth="1" />
                <g fontFamily="EB Garamond, Garamond, serif" fontSize="13" fill="#6B5D4D" textAnchor="end" style={{ fontVariantNumeric: "lining-nums tabular-nums" }}>
                  <text x="70" y="54">+{bound}</text>
                  <text x="70" y="129">+{bound / 2}</text>
                  <text x="70" y="204">0</text>
                  <text x="70" y="279">−{bound / 2}</text>
                  <text x="70" y="354">−{bound}</text>
                </g>
                <g fontFamily="EB Garamond, Garamond, serif" fontSize="14" fontStyle="italic" fill="#6B5D4D">
                  <text x="1152" y="54">toward {c.toward.length === 1 ? q(c.toward[0].word.trim()) : "the first"}</text>
                  <text x="1152" y="204" fill="#2B2117">
                    even
                  </text>
                  <text x="1152" y="354">toward {c.away.length === 1 ? q(c.away[0].word.trim()) : "the second"}</text>
                </g>
                <path d={linePath(c.naked, y)} fill="none" stroke="#6B5D4D" strokeWidth="1.4" strokeDasharray="5 4" strokeLinejoin="round" />
                {bgSeries?.lens ? (
                  <path d={linePath(bgSeries.lens, y)} fill="none" stroke="#8F2B1E" strokeWidth="1.6" strokeLinejoin="round" />
                ) : null}
                <path d={linePath(c.lens, y)} fill="none" stroke="#2B2117" strokeWidth="1.8" strokeLinejoin="round" strokeLinecap="round" />
                <line x1={cxL(li)} y1="44" x2={cxL(li)} y2="356" stroke="#8F2B1E" strokeOpacity="0.55" strokeWidth="1" />
                {c.lens ? <circle cx={cxL(li)} cy={f1(y(c.lens[li]))} r="5.5" fill="#F6F0E3" stroke="#8F2B1E" strokeWidth="1.6" /> : null}
                <g fontFamily="EB Garamond, Garamond, serif" fontSize="13" fill="#6B5D4D" textAnchor="middle" style={{ fontVariantNumeric: "lining-nums tabular-nums" }}>
                  <text x="80" y="384">
                    1
                  </text>
                  {[5, 11, 17, 23, 29, 35, 41, 47, 53, 59].map((l) => (
                    <text key={l} x={cxL(l)} y="384">
                      {l + 1}
                    </text>
                  ))}
                  <text x="1152" y="384" textAnchor="start" fontSize="12" letterSpacing="2">
                    LAYER
                  </text>
                </g>
                <text
                  x="22"
                  y="200"
                  transform="rotate(-90 22 200)"
                  textAnchor="middle"
                  fontFamily="EB Garamond, Garamond, serif"
                  fontSize="12"
                  letterSpacing="2.5"
                  fill="#6B5D4D"
                >
                  NATS
                </text>
              </svg>
            </div>
          </div>
          {row(li)}
          {li !== 59 ? row(59) : null}
          {shownBg === "controls" && c.controls && !c.controls.deltaLens ? (
            <p style={{ margin: 0, fontSize: "16px", fontStyle: "italic", color: "#8F2B1E", maxWidth: "900px" }}>
              Not read against the control at this word-piece: {c.controls.why ?? "the server gave no reason."}
            </p>
          ) : null}
          <div style={{ display: "flex", flexDirection: "column", gap: "4px", maxWidth: "900px", fontSize: "15px", color: "#6B5D4D" }}>
            <span style={{ fontVariantCaps: "all-small-caps", letterSpacing: "0.14em" }}>the bands, about the lens line</span>
            {BANDS.map((b) => (
              <span key={b.tag} style={{ display: "flex", gap: "10px", alignItems: "baseline" }}>
                <span
                  aria-hidden="true"
                  style={{ flex: "none", width: "14px", height: "10px", background: b.red ? "rgba(143,43,30,0.18)" : "rgba(201,190,172,0.5)" }}
                />
                <span style={{ color: b.red ? "#8F2B1E" : undefined }}>
                  layers {b.from}–{b.to}: {b.words}
                </span>
              </span>
            ))}
            <span>
              Read through {lensPhrase(lens)}.
              {isBaseFitted(lens) ? "" : " The study drew these bands for the base-fitted J++ lens, not for this one."} The naked eye
              needs no lens, and none of these bands is about it.
            </span>
          </div>
        </>
      )}

      <Rubric style={{ maxWidth: "860px" }}>
        Alone, these numbers carry a lean that comes from the average state as this lens reads it, partly from how the lens was
        fitted, not from your question: on the six pronoun pairs it reads +2.3 toward the female words at layers 41–50 (E3; not
        measured for one pair or for other words). Read them against the matched control, or failing that against the study’s
        average state.
      </Rubric>
      {bg === "study_mean" || shownBg === "study_mean" ? (
        <Rubric style={{ maxWidth: "860px" }}>
          On neutral prose the average state takes out most of it (−0.7 left, E3); where the model speaks of itself about −2.8 is
          left on the noun pairs (E34); at the template’s own word-pieces it was never tested. The study read its average state
          with its own code; that it lies in the same coordinates as the states this server reads is not yet checked.
        </Rubric>
      ) : null}

      <Apparatus title="one word against another" label="Apparatus for one word against another" style={{ gap: "12px" }}>
        <dl className="dl num">
          <dt>the score</dt>
          <dd>
            s<sub>ℓ</sub> = mean<sub>i</sub> [cap⟨n(h̃<sub>ℓ</sub>), W<sub>U</sub>[a<sub>i</sub>]⟩ − cap⟨n(h̃<sub>ℓ</sub>), W
            <sub>U</sub>[b<sub>i</sub>]⟩], with n(x) = x / rms(x) ⊙ fnw, cap(z) = 30 tanh(z / 30) and h̃<sub>ℓ</sub> = h
            <sub>ℓ</sub> J<sub>ℓ</sub>
            <sup>⊤</sup> (J is the identity at layer 60). The naked eye reads h<sub>ℓ</sub> for h̃<sub>ℓ</sub>. Read in fp32 on
            the pair rows only (theodolite_math.contrast_of_state), so for one pair it is the log-odds of a against b, in nats.
            One pair is the study’s she–he score; the six pronoun pairs are its pronoun score (labkit gender_scores).
          </dd>
          <dt>the words</dt>
          <dd>
            Each word is read as the landmarks read theirs, “␣word” first, then the word, but only as one word-piece: a word
            that is no single piece is not read, and the sighting goes on. Several words a side, separated by commas, are read in
            pairs, the first against the first, at most {CONTRAST_MAX_WORDS} a side.
            {c ? (
              <>
                {" "}
                Here: {c.toward.map((w, i) => (
                  <React.Fragment key={"t" + i}>
                    {i ? ", " : ""}
                    {q(w.word.trim())} as {q(pieceShown(w.piece ?? ""))} ({w.id ?? "—"})
                  </React.Fragment>
                ))}{" "}
                against{" "}
                {c.away.map((w, i) => (
                  <React.Fragment key={"a" + i}>
                    {i ? ", " : ""}
                    {q(w.word.trim())} as {q(pieceShown(w.piece ?? ""))} ({w.id ?? "—"})
                  </React.Fragment>
                ))}
                {pairs > 1 ? `, ${pairs} pairs` : ""}.
              </>
            ) : null}
            {failed.map((f, i) => (
              <React.Fragment key={"e" + i}>
                {" "}
                Not read: {f.error}
                {[...f.toward, ...f.away]
                  .filter((w) => w.pieces)
                  .map((w) => ` ${q(w.word)} splits into ${Object.entries(w.pieces ?? {}).map(([form, p]) => `${q(pieceShown(form))} → ${p.map((x) => q(pieceShown(x))).join(" + ")}`).join("; ")}.`)
                  .join("")}
              </React.Fragment>
            ))}
          </dd>
          <dt>against a control</dt>
          <dd>
            Δ<sub>ℓ</sub> = s<sub>ℓ</sub>(h) − mean over the controls of s<sub>ℓ</sub>(c), a difference of two log-odds, which
            cancels any background the two share (E3 recommendation 1). The controls are the matched control kept, sent as the
            frame’s centring: at the first word-piece read where it chose its own first, past it reading this very answer (with
            centre_answer), else not at all.
            {c?.controls ? (
              <>
                {" "}
                Here: {c.controls.nControls ?? "—"} control{c.controls.nControls === 1 ? "" : "s"}, {controlWhere(c.controls.kind)}
                {c.controls.controlLens ? (
                  <>
                    ; the control’s own score at layer {li + 1}, {nats(c.controls.controlLens[li])} through the lens and{" "}
                    {nats(c.controls.controlNaked?.[li])} to the naked eye; Δ to the naked eye {nats(c.controls.deltaNaked?.[li])}
                  </>
                ) : null}
                {c.controls.why ? `; ${c.controls.why}` : ""}.
              </>
            ) : null}
          </dd>
          <dt>against the average</dt>
          <dd>
            the same score of (h<sub>ℓ</sub> − μ<sub>ℓ</sub>) J<sub>ℓ</sub>
            <sup>⊤</sup>, as the study centred it (E3), with μ the mean state over 5,000 word-pieces of E3’s prose and chats
            (2,010 and 2,990, from the fifth on of each, template pieces included), and the average state’s own part ⟨μ
            <sub>ℓ</sub> J<sub>ℓ</sub>
            <sup>⊤</sup>, d⟩ / rms(h̃<sub>ℓ</sub>), d = fnw ⊙ mean(W<sub>U</sub>[a] − W<sub>U</sub>[b]): the average state’s
            linear share of the raw score, before the cap. Neither is a log-odds the model gives. μ is
            experiments/results/E3_background_mu.npy, pinned by its sha256; it was read through the study’s HF decoder-layer
            hooks, and its agreement with this server’s resid_post is <span className="placeholder">[NOT YET MEASURED]</span>.
            {c?.studyMean ? (
              <>
                {" "}
                Here, at layer {li + 1}: centred {nats(c.studyMean.centredLens?.[li])} through the lens and{" "}
                {nats(c.studyMean.centredNaked?.[li])} to the naked eye; the average state’s part{" "}
                {nats(c.studyMean.meanPartLens?.[li])} through the lens and {nats(c.studyMean.meanPartNaked?.[li])} to the naked
                eye (sha256 {c.studyMean.sha256 ? c.studyMean.sha256.slice(0, 12) + "…" : "not sent"}).
              </>
            ) : null}
          </dd>
          <dt>as recorded</dt>
          <dd>
            through the J++ lens, the average state’s own pronoun score at layers 41–50 is +2.3, and neutral prose centred on it
            reads −0.7 there (E3 (c)); global centring leaves about −2.8 on the noun pairs where the model speaks of itself (E34).
          </dd>
          <dt>why no frame</dt>
          <dd>
            Contrasts of these three words lie in the plane of their differences, inside the landmark frame’s span; read one at a
            time, in log-odds, they need no squaring.
          </dd>
          <dt>request</dt>
          <dd>
            <span className="mono">contrasts</span> ([{"{"}toward, away{"}"}], none unless this section is on with words typed)
            and <span className="mono">contrast_backgrounds</span> (“controls”, “study_mean”); the reply’s{" "}
            <span className="mono">contrasts</span>, 60 values a series, counted from 0 (report layer L = page layer L + 1).
            Sent now: <span className="mono">{JSON.stringify({ contrasts: sent.contrasts, contrast_backgrounds: sentBgs })}</span>
          </dd>
        </dl>
      </Apparatus>
    </section>
  );
};

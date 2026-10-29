import React from "react";
import { Apparatus, Missing, Rubric, bandParts } from "../../components/kit";
import { type LensBand, isShippedLens, lensBand, lensPhrase, useReadingLens } from "../../lib/lens";
import { signed } from "../../lib/words";
import {
  type ContrastBackground,
  type ContrastPair,
  type ContrastReading,
  type ContrastWord,
  type Sighting,
  CONTRAST_BACKGROUND_ORDER,
  CONTRAST_MAX_PAIRS,
  CONTRAST_MAX_WORDS,
  contrastPairProblems,
  contrastPairs,
  contrastProblem,
  contrastWords,
  pairName,
  useTheodoliteStore,
  withContrastPairs,
} from "../../store/useTheodoliteStore";
import { cxL, f1 } from "./geometry";

/*
 * Landmarks section vii: one word against another at every layer (the sweep's `contrasts`, read by
 * TheodoliteRuntime._contrast_readings), up to four contrasts a sighting, against either background or both. The
 * chart draws what the sighting shown was read with, one contrast at a time (a pill for each it read), never the boxes
 * as set; the boxes go with the next sighting, and only when the section is on with words typed on both sides of each.
 */

const q = (w: string) => "“" + w + "”";
/** A resolved word-piece with its leading space shown, as the server read it. */
const pieceShown = (p: string) => p.replace(/^ /, "␣").replace(/\n/g, "↵");
const sideWords = (ws: ContrastWord[]) => ws.map((w) => q(w.word.trim())).join(" · ");
const nats = (v: number | null | undefined) => (v === null || v === undefined ? null : signed(v, 2));

/** The layers of one of the lens's bands (lib/lens.ts lensBand), counted 1-60: the bands every other page draws. */
function bandLayers(band: LensBand): [number, number] {
  const ns = Array.from({ length: 59 }, (_, i) => i + 1).filter((n) => lensBand(n) === band);
  return [ns[0], ns[ns.length - 1]];
}

/** The layer bands for the lens line, as page layers (layer 60 is the model itself): the same bands, and the same words,
 * as the red line beside every reading of words (components/kit.tsx), since a contrast is one; then what the study's
 * errata add for a contrast read through an average-Jacobian lens. The lowest layers are the least trusted of all. */
const BANDS: { band: Exclude<LensBand, "own">; from: number; to: number; red: boolean; study: string }[] = (
  [
    {
      band: "untested",
      red: true,
      study:
        "There the lens’s pull-back of the pronoun direction barely agreed with the model’s own: a cosine of 0.07 at layers 1–10 and 0.03 at " +
        "11–20, with split-half reliabilities of 0.06 and 0.02, the weakest of all (E3 (b)); erratum E50 keeps a caveat on " +
        "layers 11–30 for every averaged lens.",
    },
    {
      band: "order",
      red: true,
      study:
        "Erratum E49: here only orderings checked against behaviour hold, and token-level readings such as this one are " +
        "unvalidated; erratum E50 keeps its caveat on layers 11–30.",
    },
    {
      band: "compare",
      red: false,
      study: "Erratum E49: confirm a contrast here with an instruct-fitted reading, which the instrument does not load.",
    },
    {
      band: "late",
      red: true,
      study:
        "Errata E49 and E50: a pronoun reading here counts only where an instruct-fitted reading agrees, and none is loaded.",
    },
  ] as const
).map((b) => {
  const [from, to] = bandLayers(b.band);
  return { ...b, from, to };
});

/** One fill for a band, on the chart and in its legend alike. */
const bandFill = (red: boolean) => (red ? "rgba(143,43,30,0.08)" : "rgba(201,190,172,0.22)");
const bandEdge = (red: boolean) => (red ? "rgba(143,43,30,0.45)" : "rgba(107,93,77,0.35)");

/** A round bound for the chart's scale, in nats: 1, 2, 5, 10, 20, … */
function niceBound(v: number): number {
  if (!(v > 0)) return 1;
  const e = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 5, 10]) if (v <= m * e) return m * e;
  return 10 * e;
}

const linePath = (vals: number[] | null, y: (v: number) => number) =>
  vals ? vals.map((v, l) => (l === 0 ? "M" : " L") + cxL(l) + "," + f1(y(v))).join("") : "";

/** What a background reads, for the number row and the chart's red lines. */
function backgroundSeries(c: ContrastReading, b: ContrastBackground | null) {
  if (b === "controls" && c.controls) return { lens: c.controls.deltaLens, naked: c.controls.deltaNaked };
  if (b === "study_mean" && c.studyMean) return { lens: c.studyMean.centredLens, naked: c.studyMean.centredNaked };
  return null;
}

/** Each background's red line, solid against the control and dashed centred on the average, and its words. */
const BACKGROUND_LINE: Record<ContrastBackground, { dash: string | undefined; words: string; row: string }> = {
  controls: { dash: undefined, words: "against the matched control", row: "against the control" },
  study_mean: { dash: "9 4", words: "centred on the study’s average state", row: "centred on the study’s average state" },
};

/** A contrast as the sighting read it, for its pill and its heading. */
const contrastName = (c: ContrastReading) => sideWords(c.toward) + " against " + sideWords(c.away);
const readable = (c: ContrastReading) => c.error === null && c.lens !== null;
const countWord = (n: number) => ["no", "one", "two", "three", "four", "five", "six"][n] ?? String(n);

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
  const bandLead = bandParts(lens, "all").lead;

  // the server's word on the study's average state: unknown before its health is read; an older server reads no contrasts
  const card = health?.contrasts ?? null;
  const smProblem = card ? card.study_mean_problem : null;
  const smOk = !!card && smProblem === null;
  // how many contrasts one sighting may weigh: the server's own bound, four until its health says
  const maxPairs = card?.max_contrasts ?? CONTRAST_MAX_PAIRS;
  const maxWord = countWord(maxPairs);
  const pairs = contrastPairs(contrast);
  const pairProblems = contrastPairProblems(contrast);
  const problem = contrastProblem(contrast, maxPairs);
  const sentBgs = sent.contrast_backgrounds;
  const turnOn = (on: boolean) => {
    // the placeholders become words typed only when the reader turns the section on, so nothing unseen is ever sent
    if (on && !contrast.toward.trim() && !contrast.away.trim()) setContrast({ on, toward: placeholders[0], away: placeholders[1] });
    else setContrast({ on });
  };
  const setPairs = (ps: ContrastPair[]) => setContrast(withContrastPairs(ps));
  const setPair = (i: number, p: Partial<ContrastPair>) => setPairs(pairs.map((x, j) => (j === i ? { ...x, ...p } : x)));
  const tick = (b: ContrastBackground, on: boolean) => {
    setContrast({ backgrounds: on ? [...new Set([...contrast.backgrounds, b])] : contrast.backgrounds.filter((x) => x !== b) });
    // the matched control goes with the request as the frame's centring, so reading against it centres the frame too
    if (b === "controls" && on) setFrame({ centre: true });
  };

  /* ---- what the sighting shown was read with: the contrast picked by its pill, else the first it could read */
  const [picked, setPicked] = React.useState<number | null>(null);
  const list = sighting?.contrasts ?? null;
  const firstRead = list ? list.findIndex(readable) : -1;
  const shownIdx = list && picked !== null && picked < list.length ? picked : firstRead;
  const shown = list && shownIdx >= 0 ? list[shownIdx] : null;
  const c = shown && readable(shown) ? shown : null;
  const failed = list?.filter((x) => x.error !== null) ?? [];
  // the contrast that could not be read, whose words are shown in its place: the one picked, else the first
  const notRead = shown && !c ? shown : (failed[0] ?? null);
  const shownBgs = c ? CONTRAST_BACKGROUND_ORDER.filter((b) => (b === "controls" ? c.controls : c.studyMean) !== null) : [];
  const bgLines = c ? shownBgs.map((b) => ({ b, ...BACKGROUND_LINE[b], s: backgroundSeries(c, b) })) : [];
  const aName = c ? sideWords(c.toward) : "";
  const bName = c ? sideWords(c.away) : "";
  const nPairs = c ? c.toward.length : 0;
  // the words the rubric names: the sighting's own when it weighed some, else those the boxes will send
  const typed = contrastWords(contrast.toward)[0] ?? placeholders[0];
  const typedAway = contrastWords(contrast.away)[0] ?? placeholders[1];
  const first = c ? (nPairs === 1 ? aName : "the first words") : typed ? q(typed) : "the first word";
  const second = c ? (nPairs === 1 ? bName : "the second") : typedAway ? q(typedAway) : "the second";

  /* ---- the chart: 60 layers in nats, symmetric about nought */
  const plotted = c ? [c.lens, c.naked, ...bgLines.map((x) => x.s?.lens ?? null)].filter((v): v is number[] => v !== null) : [];
  const bound = niceBound(Math.max(0, ...plotted.flat().map((v) => Math.abs(v))));
  const y = (v: number) => 200 - (Math.max(-bound, Math.min(bound, v)) * 150) / bound;
  const half = 1060 / 59 / 2;
  const at = (vals: number[] | null | undefined, l: number) => (vals ? vals[l] : null);

  const row = (l: number) =>
    c ? (
      <p style={{ margin: 0, fontSize: "17px", fontVariantNumeric: "lining-nums tabular-nums", maxWidth: "900px" }}>
        <span style={{ fontVariantCaps: "all-small-caps", letterSpacing: "0.12em", color: "#6B5D4D" }}>at layer {l + 1}</span>{" "}
        {nats(at(c.lens, l)) ?? <Missing what="not recorded in this sighting" />} through the lens ·{" "}
        {nats(at(c.naked, l)) ?? <Missing what="not recorded in this sighting" />} to the naked eye
        {bgLines.map((x) => (
          <React.Fragment key={x.b}>
            {" "}
            ·{" "}
            {nats(at(x.s?.lens, l)) ?? <Missing what={x.b === "controls" ? "no control read here" : "not recorded in this sighting"} />}{" "}
            {x.row}
          </React.Fragment>
        ))}
        <span style={{ color: "#6B5D4D" }}>
          {" "}
          (
          {shownBgs.includes("study_mean")
            ? "nats of log-odds, but the one centred on the study’s average state is the score of the state less the average, as E3 centred it: not a log-odds the model gives"
            : "nats of log-odds"}
          )
        </span>
      </p>
    ) : null;

  const checkbox = (b: ContrastBackground, label: React.ReactNode, enabled: boolean, note?: React.ReactNode) => {
    const ticked = contrast.backgrounds.includes(b);
    // one ticked that can no longer be read against can still be unticked, as the radio's "nothing" could always be chosen
    const usable = enabled || (contrast.on && ticked);
    return (
      <label
        style={{
          display: "flex",
          alignItems: "flex-start",
          gap: "10px",
          minHeight: "44px",
          paddingTop: "10px",
          cursor: usable ? "pointer" : "not-allowed",
          color: enabled ? "#2B2117" : "#6B5D4D",
        }}
      >
        <input
          type="checkbox"
          checked={ticked}
          disabled={!usable}
          onChange={(e) => tick(b, e.target.checked)}
          style={{ width: "17px", height: "17px", accentColor: "#8F2B1E", margin: "3px 0 0", flex: "none" }}
        />
        <span>
          {label}
          {note ? <span style={{ display: "block", fontSize: "14px", fontStyle: "italic", color: "#6B5D4D" }}>{note}</span> : null}
        </span>
      </label>
    );
  };

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
        times as likely). One sighting can weigh up to {maxWord} such contrasts, each read on
        its own; the chart draws one at a time.
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
      <div style={{ display: "flex", flexDirection: "column", gap: "2px", maxWidth: "900px" }}>
        {pairs.map((p, i) => {
          // the first pair keeps the ids it always had; its boxes show the first two landmarks as placeholders
          const tid = i === 0 ? "contrast-toward" : `contrast-toward-${i + 1}`;
          const aid = i === 0 ? "contrast-away" : `contrast-away-${i + 1}`;
          return (
            <div key={i} style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
              <div
                style={{
                  display: "flex",
                  flexWrap: "wrap",
                  alignItems: "baseline",
                  gap: "0 12px",
                  fontSize: "28px",
                  lineHeight: 1.8,
                  color: contrast.on ? "#2B2117" : "#6B5D4D",
                }}
              >
                <span>
                  <label htmlFor={tid}>Weigh</label>{" "}
                  {box(tid, p.toward, i === 0 ? placeholders[0] : "", (v) => setPair(i, { toward: v }))}{" "}
                  <label htmlFor={aid}>against</label>{" "}
                  {box(aid, p.away, i === 0 ? placeholders[1] : "", (v) => setPair(i, { away: v }))}.
                </span>
                {pairs.length > 1 ? (
                  <button
                    type="button"
                    className="btn-ghost"
                    disabled={!contrast.on}
                    onClick={() => setPairs(pairs.filter((_, j) => j !== i))}
                    aria-label={`Remove ${pairName(i)}`}
                  >
                    remove
                  </button>
                ) : null}
              </div>
              {contrast.on && pairs.length > 1 && pairProblems[i] ? (
                <p style={{ margin: 0, fontStyle: "italic", fontSize: "16px", color: "#8F2B1E" }}>{pairProblems[i]}</p>
              ) : null}
            </div>
          );
        })}
        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "8px 16px", paddingTop: "4px" }}>
          <button
            type="button"
            className="btn-ghost"
            disabled={!contrast.on || pairs.length >= maxPairs}
            onClick={() => setPairs([...pairs, { toward: "", away: "" }])}
          >
            add a pair
          </button>
          <span style={{ fontSize: "15px", color: "#6B5D4D" }}>
            {pairs.length >= maxPairs
              ? `No more: the server weighs at most ${maxWord} contrasts in one sighting.`
              : `Each pair is read as its own contrast, up to ${maxWord} in one sighting.`}
          </span>
        </div>
      </div>
      <fieldset style={{ border: "0", margin: "0", padding: "0", display: "flex", flexDirection: "column", maxWidth: "900px" }}>
        <legend style={{ padding: "6px 0 0", fontVariantCaps: "all-small-caps", letterSpacing: "0.14em", color: "#6B5D4D" }}>
          read it also against (either, both, or neither: each is a line of its own)
        </legend>
        {checkbox(
          "controls",
          "the matched control kept (this centres the frame above too)",
          contrast.on && canCentre,
          canCentre ? undefined : "Keep a matched control first, in the question’s apparatus on the instrument page.",
        )}
        {checkbox(
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
      {contrast.on && !problem
        ? contrast.backgrounds
            .filter((b) => !sentBgs.includes(b))
            .map((b) => (
              <p key={b} role="status" style={{ margin: 0, fontStyle: "italic", fontSize: "17px", color: "#8F2B1E", maxWidth: "900px" }}>
                {b === "controls"
                  ? "The next sighting will not be read against the control: none goes with it. Keep a matched control, and centre on it in the frame’s options above."
                  : "The next sighting will not be read against the study’s average state: the server cannot read it."}
              </p>
            ))
        : null}
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "10px 22px" }}>
        <button className="btn-primary" type="button" onClick={onRead} disabled={isSweeping || !contrast.on || problem !== null}>
          {isSweeping ? "taking a sighting…" : readLabel}
        </button>
        <span style={{ fontSize: "16px", color: "#6B5D4D" }}>
          {sent.contrasts.length
            ? `It will weigh ${sent.contrasts.map((x) => `${x.toward.map(q).join(" · ")} against ${x.away.map(q).join(" · ")}`).join("; ")}${
                sentBgs.length
                  ? (sent.contrasts.length > 1 ? ", each also " : ", and ") +
                    sentBgs.map((b) => BACKGROUND_LINE[b as ContrastBackground]?.words ?? b).join(" and ")
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
              The server that took this sighting sent no contrasts: it is older than this page. Take a new sighting from
              this server to weigh one word against another.
            </>
          )}
        </p>
      ) : list.length === 0 ? (
        <p style={{ margin: 0, fontSize: "17px", color: "#6B5D4D", maxWidth: "900px" }}>
          The sighting shown was taken without weighing one word against another. Turn it on above, and read the answer again.
        </p>
      ) : (
        <>
          {list.length > 1 ? (
            <div role="group" aria-label="The contrast the chart draws" style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "8px 10px" }}>
              <span style={{ fontVariantCaps: "all-small-caps", letterSpacing: "0.14em", color: "#6B5D4D" }}>draw</span>
              {list.map((x, i) => (
                <button
                  key={i}
                  type="button"
                  className="btn-ghost plain"
                  aria-pressed={i === shownIdx}
                  onClick={() => setPicked(i)}
                  style={{ fontSize: "16px", fontStyle: "italic" }}
                >
                  {contrastName(x)}
                  {readable(x) ? "" : " (not read)"}
                </button>
              ))}
              <span style={{ fontSize: "15px", color: "#6B5D4D" }}>
                the {countWord(list.length)} contrasts this sighting was asked for, in the order sent
              </span>
            </div>
          ) : null}
          {!c ? (
            <p style={{ margin: 0, fontSize: "17px", color: "#8F2B1E", maxWidth: "900px", fontStyle: "italic" }}>
              {notRead
                ? `The sighting shown asked for a contrast it could not read${list.length > 1 ? `, ${contrastName(notRead)}` : ""}. ${notRead.error ?? "The server sent no reading for it."}`
                : "The sighting shown sent no reading for this contrast."}
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
                {bgLines.map((x) => (
                  <span key={x.b} style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                    <svg width="22" height="10" viewBox="0 0 22 10" aria-hidden="true">
                      <path d="M1 5 H21" stroke="#8F2B1E" strokeWidth="1.6" strokeDasharray={x.dash} />
                    </svg>
                    through the lens, {x.words}
                  </span>
                ))}
              </div>
              <div style={{ overflowX: "auto", paddingBottom: "6px" }}>
                <div style={{ minWidth: "940px" }}>
                  <svg
                    viewBox="0 0 1280 420"
                    role="img"
                    aria-label={`A line chart across the sixty layers of the sighting shown: ${aName} weighed against ${bName}, in nats, from minus ${bound} to ${bound}; the lens as a solid line, the naked eye dashed${
                      bgLines.length
                        ? `, and in red the lens ${bgLines.map((x) => x.words + (x.dash ? " (dashed)" : bgLines.length > 1 ? " (solid)" : "")).join(" and ")}`
                        : ""
                    }. The chosen layer, ${li + 1}, is marked in red.`}
                    style={{ width: "100%", height: "auto", display: "block" }}
                  >
                    {BANDS.map((b) => (
                      <rect
                        key={b.band}
                        x={f1(cxL(b.from - 1) - half)}
                        y="44"
                        width={f1(cxL(b.to - 1) - cxL(b.from - 1) + 2 * half)}
                        height="312"
                        fill={bandFill(b.red)}
                      />
                    ))}
                    <g fontFamily="EB Garamond, Garamond, serif" fontSize="12" fontStyle="italic" textAnchor="middle">
                      {BANDS.map((b) => (
                        <text key={b.band} x={f1((cxL(b.from - 1) + cxL(b.to - 1)) / 2)} y="38" fill={b.red ? "#8F2B1E" : "#6B5D4D"}>
                          {b.from}–{b.to}
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
                    {bgLines.map((x) =>
                      x.s?.lens ? (
                        <path
                          key={x.b}
                          d={linePath(x.s.lens, y)}
                          fill="none"
                          stroke="#8F2B1E"
                          strokeWidth="1.6"
                          strokeDasharray={x.dash}
                          strokeLinejoin="round"
                        />
                      ) : null,
                    )}
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
              {c.controls && !c.controls.deltaLens ? (
                <p style={{ margin: 0, fontSize: "16px", fontStyle: "italic", color: "#8F2B1E", maxWidth: "900px" }}>
                  Not read against the control at this word-piece: {c.controls.why ?? "the server gave no reason."}
                </p>
              ) : null}
              <div style={{ display: "flex", flexDirection: "column", gap: "4px", maxWidth: "900px", fontSize: "15px", color: "#6B5D4D" }}>
                <span style={{ fontVariantCaps: "all-small-caps", letterSpacing: "0.14em" }}>the bands, about the lens line</span>
                {bandLead ? <span>{bandLead}</span> : null}
                {BANDS.map((b) => (
                  <span key={b.band} style={{ display: "flex", gap: "10px", alignItems: "baseline" }}>
                    <span
                      aria-hidden="true"
                      style={{ flex: "none", width: "14px", height: "10px", background: bandFill(b.red), border: `1px solid ${bandEdge(b.red)}` }}
                    />
                    <span style={{ color: b.red ? "#8F2B1E" : undefined }}>
                      layers {b.from}–{b.to}: {bandParts(lens, b.band).words} {b.study}
                    </span>
                  </span>
                ))}
                <span>
                  Read through {lensPhrase(lens)}.
                  {isShippedLens(lens) ? "" : " The study measured these bands with the shipped J++ lens, fitted on the base model, not with this one."}{" "}
                  The bands are about the lens line; the naked eye has a background of its own, given below.
                </span>
              </div>
            </>
          )}
        </>
      )}

      <Rubric style={{ maxWidth: "860px" }}>
        Alone, both lines carry a lean that comes from the model’s average state, not from your question, and the lens and the
        naked eye read it differently. Through the J++ lens, on E3’s neutral prose at layers 41–50, the average state’s share
        of the six pronoun pairs’ score was +1.4 toward the female words, and the score itself +0.8; how large it is depends
        partly on how the lens was fitted (E3 (c); erratum E34). On E3’s noun pairs, woman–man and the
        like, the same share was −3.1, toward the male words, and the score −2.1; for one pair, or for other words, it was
        not measured. Read them against the matched control, or failing that against the study’s average state.
      </Rubric>
      <Rubric style={{ maxWidth: "860px" }}>
        The naked eye needs no lens, but it has a background of its own, leaning the other way: E3 read its average state by
        the naked eye at −4.3 on the six pronoun pairs, toward the male words, at layers 1–20, and −0.9 at layers 41–50. On
        neutral prose the naked eye read −3.8, −6.5 and −3.6 at layers 1–10, 11–20 and 21–30, and centring on the average state
        still left −4.8 and −5.7 at layers 11–30 (E3 (c), its naked-eye figures). The naked eye’s grey dashed line is no
        clean baseline either.
      </Rubric>
      {contrast.backgrounds.includes("study_mean") || shownBgs.includes("study_mean") ? (
        <Rubric style={{ maxWidth: "860px" }}>
          On E3’s neutral prose, centring on the average state took the lens’s score at layers 41–50 from +0.8 to −0.7, taking out
          the average state’s +1.4 share (E3 (c)); where the model speaks of itself, the same centring left about −2.8 on the noun
          pairs (erratum E34); at the template’s own word-pieces it was never tested. The study read its average state with its
          own code; on one of E3’s neutral passages this server’s states agree with that code’s at every layer, and E3’s
          neutral-prose figures come back through this page’s own scoring (the end-to-end check).
        </Rubric>
      ) : null}

      <Apparatus title="one word against another" label="Apparatus for one word against another" style={{ gap: "12px" }}>
        <dl className="dl num">
          <dt>the score</dt>
          <dd>
            s<sub>ℓ</sub> = mean<sub>i</sub> [cap⟨n(h̃<sub>ℓ</sub>), W<sub>U</sub>[a<sub>i</sub>]⟩ − cap⟨n(h̃<sub>ℓ</sub>), W
            <sub>U</sub>[b<sub>i</sub>]⟩], with n(x) = x / rms(x) ⊙ fnw, cap(z) = 30 tanh(z / 30) and h̃<sub>ℓ</sub> = h
            <sub>ℓ</sub> J<sub>ℓ</sub>
            <sup>⊤</sup> (J is the identity at layer 60), a product taken in bf16, as the study’s labkit takes it. The naked eye
            reads h<sub>ℓ</sub> for h̃<sub>ℓ</sub>. The score is then taken in fp32 on the pair rows only
            (theodolite_math.contrast_of_state), so for one pair it is the log-odds of a against b, in nats.
            One pair is the study’s she–he score; the six pronoun pairs are its pronoun score (labkit gender_scores).
          </dd>
          <dt>the words</dt>
          <dd>
            Each word is read as the landmarks read theirs, “␣word” first, then the word, but only as one word-piece: a word
            that is no single piece is not read, and the sighting goes on. Several words a side, separated by commas, are read in
            pairs, the first against the first, at most {CONTRAST_MAX_WORDS} a side. Each row above is a contrast of its own, up
            to {maxWord} a sighting, read and drawn apart.
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
                {nPairs > 1 ? `, ${nPairs} pairs of words` : ""}.
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
            cancels any background the two share (E3 (c), recommendation 1). The controls are the matched control kept, sent as the
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
            hooks. On the first of E3’s neutral passages this server’s resid_post agrees with that capture at every layer, within
            the check’s bound of 1e-2 (relative), and on E3’s twelve neutral passages the raw score, the score centred on μ and
            μ’s part at layers 41–50 come back through the server’s contrast_of_state within 0.1 of E3’s
            (scripts/verify_e2e_ui_and_mcp.py, item 12, run on the GPU; each run’s figures are in
            experiments/results/E3_background_mu_check.json). Only E3’s neutral prose at layers 41–50 was checked so.
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
            Through the J++ lens, on E3’s neutral prose at layers 41–50, the average state’s linear share of the six pronoun pairs’
            score is +1.4, the score itself +0.8, and the score centred on the average state −0.7; on its noun pairs (labkit
            NOUN_PAIRS: woman–man, girl–boy and the like) the same share is −3.1 and the score −2.1 (E3 (c)); global centring leaves
            about −2.8 on the noun pairs where the model speaks of itself (erratum E34). The average state read on its own
            scores +2.3 there in E3’s record (μ alone, normalised by its own size: not the lean any reading carries). This
            server, like the study’s own code today, reads +2.35 for it; E3’s per-layer lens figures for μ differ from today’s
            by more at some layers, while by the naked eye they agree to within 0.03 at every layer, and why E3’s run read
            otherwise is not established, so only band means are quoted here (by layer: experiments/results/E3_background_mu_check.json). By the naked eye
            the average state scores −4.3 at layers 1–20 and −0.9 at 41–50; on neutral prose the naked eye’s raw score is −3.8,
            −6.5 and −3.6 at layers 1–10, 11–20 and 21–30, its average-state share −3.4, −4.8 and −1.3, and centred −4.8 and
            −5.7 at 11–20 and 21–30 (E3 (c), E3_lens_validation_c_background.json).
          </dd>
          <dt>why no frame</dt>
          <dd>
            Contrasts of these three words lie in the plane of their differences, inside the landmark frame’s span; read one at a
            time, in log-odds, they need no squaring.
          </dd>
          <dt>request</dt>
          <dd>
            <span className="mono">contrasts</span> ([{"{"}toward, away{"}"}, …], one for each row, in order, at most the server’s{" "}
            <span className="mono">max_contrasts</span> ({maxPairs}); none unless this section is on with words typed on both sides of
            every row) and <span className="mono">contrast_backgrounds</span> (“controls”, “study_mean”: those ticked, in that order,
            “controls” only with a matched control sent, “study_mean” only when the server can read its file); the reply’s{" "}
            <span className="mono">contrasts</span>, one for each sent, in order, 60 values a series, counted from 0 (report layer L =
            page layer L + 1), with a series under <span className="mono">backgrounds</span> for each background.
            Sent now: <span className="mono">{JSON.stringify({ contrasts: sent.contrasts, contrast_backgrounds: sentBgs })}</span>
          </dd>
        </dl>
      </Apparatus>
    </section>
  );
};

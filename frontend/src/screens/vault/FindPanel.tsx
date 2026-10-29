import React from "react";
import { Apparatus, IfApparatus, Missing, Rubric, IfRubrics } from "../../components/kit";
import { HeldBackWords } from "../../components/HeldBack";
import {
  type Gate,
  GATE_FIGURES,
  OVERLAP_RECORD,
  RECORDED_CHECKED,
  heldRubric,
  labelGate,
  layerIdx,
  parseLabelCheck,
  recordedVaultHeld,
  vaultCheck,
} from "../../lib/labels";
import { useApparatus } from "../../reading";
import { isShippedLens, useInstrumentLens } from "../../lib/lens";
import { href } from "../../router";
import { useTheodoliteStore } from "../../store/useTheodoliteStore";
import { CHANCE, CONTRAST, OVERLAP, STUDY_LAYERS, STUDY_WORDS, VAULT_ANSWERS } from "./recorded";
import { GLOSSES } from "./glossary";
import { esc, nearestStatement } from "./lql";
import { agreement, asNearest, cleanWords, clock, findNearestEntry, num, prettyJson } from "./live";
import { EmptyAnswer, FaceMark, OwnWords, chipStyle, f1, fmtCos, rawJson, roman, sentenceInput, show, signedFixed, smallCaps } from "./bits";
import { runFind, toConsole, useVaultUi } from "./ui";

/** The study's test, per neuron (repo index:neuron) → [selective at BH q < 0.05 (1) or not (0), Cohen d]. */
const TESTED = new Map<string, [number, number | null]>();
for (const [key, rows] of Object.entries(VAULT_ANSWERS)) {
  const idx = key.split("@")[1];
  for (const r of rows) if (r[6] !== null) TESTED.set(`${idx}:${r[0]}`, [r[6], r[7]]);
}

interface ShownRow {
  neuron: number;
  face: number;
  cos: number;
  words: string[];
  naked: string[];
  /** word-pieces shared among the five shown, spelt exactly (the chip filter's count) */
  agree: number;
  tested: number | null;
  d: number | null;
  /** the label gate (lib/labels.ts): from the server's label_check for a live row, the recorded check for a recorded
   * one (checked ten a side once, from the weights: N1G_recorded_label_checks.json), else the layer alone */
  gate: Gate;
}

const sgn2 = (c: number) => (c < 0 ? "−" : "+") + Math.abs(c).toPrecision(2);
const GRID = "40px 130px 140px 150px minmax(0, 1fr) 100px 150px";

export const FindPanel: React.FC = () => {
  const ui = useVaultUi();
  const larqlLog = useTheodoliteStore((s) => s.larqlLog);
  const larqlPending = useTheodoliteStore((s) => s.larqlPending);
  const ap = useApparatus();
  // the lens the vault reads through: the server's current one
  const { lens: vaultLens } = useInstrumentLens();

  // ---------- the question being written
  const fWord = ui.fWord.trim();
  const fL = num(ui.fLayer);
  const fK = num(ui.fK);
  const fLOk = fL !== null && fL >= 1 && fL <= 60;
  const fKOk = fK !== null && fK >= 1;
  const lql = nearestStatement(fWord, fLOk ? fL : null, fKOk ? fK : null);
  const zeroNote = !fLOk
    ? "Choose a layer from 1 to 60."
    : /['"]/.test(fWord)
      ? "Words go in single or double quotes and may not contain one."
      : `The vault counts its layers from nought, so layer ${fL} is written ${fL - 1}.`;

  // ---------- the question asked
  const A = ui.asked;
  const aWordExact = A.word.trim();
  const aWord = aWordExact.toLowerCase();
  const aL = num(A.layer);
  const aKraw = num(A.k);
  const aLOk = aL !== null && aL >= 1 && aL <= 60;
  const stale = fWord !== aWordExact || ui.fLayer !== A.layer || ui.fK !== A.k;

  // the running instrument's answer (the newest in the log for exactly this question)
  const entry =
    aLOk && aKraw !== null && aKraw >= 1 && aWordExact ? findNearestEntry(larqlLog, aWordExact, aL - 1, aKraw) : null;
  const live = entry && entry.ok ? asNearest(entry.data) : null;
  const liveError = entry && !entry.ok ? (entry.error ?? "no reason given") : null;
  const pending = larqlPending && ui.askedAt !== null && !(entry && entry.at >= ui.askedAt);

  // the study's recorded answer (lower case, as the board looks it up)
  const recKey = aLOk && aWord ? `${aWord}@${aL - 1}` : null;
  const rec = recKey && Object.prototype.hasOwnProperty.call(VAULT_ANSWERS, recKey) ? VAULT_ANSWERS[recKey] : null;
  const aK = aKraw !== null && aKraw >= 1 ? Math.min(10, aKraw) : 10;

  const source: "live" | "recorded" | null = live && live.results.length ? "live" : rec ? "recorded" : null;
  const layerNo = live ? live.layer + 1 : (aL ?? 0);

  let all: ShownRow[] = [];
  if (source === "live" && live) {
    all = live.results.map((r) => {
      const lens = r.j_vindex_top_tokens.map((t) => t.token);
      const eye = r.naive_lens_top_tokens.map((t) => t.token);
      const t = TESTED.get(`${live.layer}:${r.neuron_idx}`);
      return {
        neuron: r.neuron_idx,
        face: r.polarity_num >= 0 ? 1 : -1,
        cos: r.cosine_similarity,
        words: cleanWords(lens),
        naked: cleanWords(eye),
        agree: agreement(lens, eye),
        tested: t ? t[0] : null,
        d: t ? t[1] : null,
        // live.layer counts from 0, as the server's fields do
        gate: labelGate(layerIdx(live.layer), parseLabelCheck(r.label_check)),
      };
    });
  } else if (source === "recorded" && rec) {
    all = rec.slice(0, aK).map((r) => ({
      neuron: r[0],
      face: r[1] > 0 ? 1 : -1,
      cos: r[2],
      words: r[3],
      naked: r[4],
      agree: r[5],
      tested: r[6],
      d: r[7],
      // aL counts from 1 (the page's layer); the recorded key and the gate count from 0
      gate: labelGate(layerIdx(aL! - 1), recKey ? vaultCheck(recKey, r[0], r[1]) : null),
    }));
  }
  const heldRows = all.filter((r) => r.gate.heldBack);
  const allHeld = all.length > 0 && heldRows.length === all.length;
  const anyTen = all.some((r) => r.gate.sharedTop10 != null);
  const top = source === "live" && live ? live.results[0].cosine_similarity : rec ? rec[0][2] : 0;
  const scaleMax = Math.max(0.0175, top * 1.08);
  const chanceL = f1((CHANCE / scaleMax) * 100);
  const gateN = ap ? ui.gate : 0;
  const rows = all.filter((r) => r.agree >= gateN);
  const setAside = all.length - rows.length;
  const shownWord = source === "live" ? aWordExact : aWord;

  let meta: string;
  if (source) {
    meta =
      all.length +
      (all.length === 1 ? " neuron" : " neurons") +
      " nearest “" +
      shownWord +
      "” in layer " +
      layerNo +
      (source === "live" && entry
        ? " · answered by the running instrument at " + clock(entry.at)
        : " · from the study’s search of 16 October 2026") +
      (heldRows.length ? " · " + (allHeld ? "the words of every row" : "the words of " + heldRows.length) + " held back" : "") +
      (setAside > 0 ? " · " + setAside + " set aside by the chips, on the five shown" : "");
  } else {
    meta = aWord ? "nothing recorded for “" + aWord + "” in layer " + (aL === null ? "…" : aL) : "no search yet";
  }
  const rel = top < CHANCE * 0.93 ? "below" : top > CHANCE * 1.07 ? "above" : "about level with";
  const chanceNote =
    "At layer " +
    layerNo +
    (rel === "below" ? " even" : "") +
    " the nearest neuron scores " +
    fmtCos(top) +
    ", " +
    rel +
    " the 0.014 that two unrelated directions typically share: the upright hairline on each bar." +
    (rel === "above" ? " Even so, a clear score is a lead, not a proof." : " The ranking still turns up some real neurons.");
  const kNote = source === "recorded" && aKraw !== null && aKraw > 10 ? "The study kept only the nearest ten." : "";

  // ---------- the study's chips
  const curL = () => {
    const n = num(useVaultUi.getState().fLayer);
    return n !== null && (STUDY_LAYERS as readonly number[]).includes(n) ? n : 51;
  };
  const curW = () => {
    const w = useVaultUi.getState().fWord.trim().toLowerCase();
    return (STUDY_WORDS as readonly string[]).includes(w) ? w : "she";
  };
  const curK = () => {
    const n = num(useVaultUi.getState().fK);
    return n !== null && n >= 1 ? n : 5;
  };

  // ---------- apparatus
  const aIdx = aL === null ? -1 : aL - 1;
  const ov = OVERLAP[aIdx];
  let thisSearch: string;
  if (source === "live" && live) {
    const last = live.results[live.results.length - 1];
    thisSearch =
      "This search, at layer " +
      layerNo +
      ", as the running instrument answered it: nearest " +
      fmtCos(top) +
      " (neuron " +
      live.results[0].neuron_idx +
      "), " +
      (top / CHANCE).toFixed(2) +
      " × chance; the last of " +
      live.results.length +
      ", " +
      fmtCos(last.cosine_similarity) +
      ". The word was sought as the word-piece " +
      JSON.stringify(live.resolved_token_str) +
      " (id " +
      live.resolved_token_id +
      "), in " +
      live.elapsed_ms +
      " ms.";
  } else if (rec) {
    thisSearch =
      "This search, at layer " +
      aL +
      ": nearest " +
      fmtCos(rec[0][2]) +
      " (neuron " +
      rec[0][0] +
      "), " +
      (rec[0][2] / CHANCE).toFixed(2) +
      " × chance; the tenth " +
      fmtCos(rec[rec.length - 1][2]) +
      ".";
  } else thisSearch = "This search: [NOT RECORDED; MEASURED IN A LIVE SIGHTING].";
  const layerNote = ov
    ? "At layer " +
      aL +
      " the “she” and “he” tens share " +
      ov[0] +
      " neurons, " +
      ov[1] +
      " of them with the same face; " +
      ov[2] +
      " of the “she” ten are also in the ten nearest the she − he contrast (overlap)."
    : "[RECORDED ONLY AT LAYERS 21, 31, 41, 51, 56 AND 59]";
  const contrast = CONTRAST[aIdx]
    ? "At layer " + aL + ": " + CONTRAST[aIdx].map((c) => c[0] + " (" + sgn2(c[1]) + ")").join(", ") + " (contrast_lists)."
    : "Recorded only at layers 21, 31, 41, 51, 56 and 59.";
  const isLiveRaw = source === "live" && entry !== null;
  const rawLabel = ui.faRaw
    ? isLiveRaw
      ? "hide the instrument’s reply"
      : "hide the recorded reply"
    : isLiveRaw
      ? "show the instrument’s reply"
      : "show the recorded reply";
  const rawJsonText = isLiveRaw && entry ? esc(prettyJson(entry.data)) : rec && recKey ? rawJson(recKey, aK) : "[NO RECORDED REPLY FOR THIS SEARCH; MEASURED IN A LIVE SIGHTING]";
  const rawCaption =
    isLiveRaw && entry
      ? "As the running instrument returned it at " +
        clock(entry.at) +
        ", for “" +
        entry.query +
        "”: every field of query_nearest_edges, among them the word-piece it resolved, each word-piece’s logit and the time taken."
      : rec
        ? "As recorded, lists[\"" +
          recKey +
          "\"] in E6_tool_audit_04_vault.partial.json: the first " +
          aK +
          " of ten, kept by the study’s script as neuron (n), face (pol), nearness (cos) and five word-pieces through the lens (jv) and to the naked eye (nv)."
        : "The study recorded no reply for this search.";

  return (
    <section role="tabpanel" id="panel-find" aria-labelledby="tab-find" style={{ paddingTop: 34, display: "flex", flexDirection: "column", gap: 40 }}>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "36px 64px", alignItems: "flex-start" }}>
        <div style={{ flex: "999 1 640px", minWidth: 0, display: "flex", flexDirection: "column", gap: 20 }}>
          <h2 className="h-sec">ii · find a neuron by its words</h2>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              runFind(ui.fWord, ui.fLayer, ui.fK);
            }}
            style={{ display: "contents" }}
          >
            <div style={{ fontSize: 30, lineHeight: 1.75, maxWidth: 800 }}>
              <label htmlFor="f-word">Find the neurons whose words lie nearest to</label>{" "}
              <input
                id="f-word"
                type="text"
                value={ui.fWord}
                onChange={(e) => ui.set({ fWord: e.target.value })}
                autoComplete="off"
                spellCheck={false}
                style={sentenceInput(30, "6.4em", { italic: true })}
              />{" "}
              <label htmlFor="f-layer">in layer</label>{" "}
              <input
                id="f-layer"
                type="number"
                min={1}
                max={60}
                value={ui.fLayer}
                onChange={(e) => ui.set({ fLayer: e.target.value })}
                style={sentenceInput(30, "2.8em", { numeric: true })}
              />
              ,{" "}
              <label htmlFor="f-k">and show me</label>{" "}
              <input
                id="f-k"
                type="number"
                min={1}
                max={10}
                value={ui.fK}
                onChange={(e) => ui.set({ fK: e.target.value })}
                style={sentenceInput(30, "2.4em", { numeric: true })}
              />
              .{" "}
            </div>
            <OwnWords lql={lql} note={zeroNote} maxWidth={800} />
            <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "12px 20px" }}>
              <button className="btn-primary" type="submit">
                search the vault
              </button>
              {stale ? (
                <span style={{ fontSize: 16, fontStyle: "italic", color: "var(--muted)" }}>
                  You have changed the question. Search again to see its answer.
                </span>
              ) : null}
            </div>
          </form>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <span style={{ fontSize: 16, color: "var(--muted)" }}>or begin from the study’s own searches</span>
            <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "8px 10px" }}>
              <span style={{ ...smallCaps(), marginRight: 4 }}>word</span>
              {STUDY_WORDS.map((w) => {
                const on = source !== null && aWord === w;
                return (
                  <button
                    key={w}
                    type="button"
                    aria-pressed={on}
                    onClick={() => runFind(w, String(curL()), String(curK()))}
                    style={chipStyle(on, { fontSize: 18, fontStyle: "italic", padding: "6px 14px" })}
                  >
                    “{w}”
                  </button>
                );
              })}
              <span style={{ ...smallCaps(), margin: "0 4px 0 14px" }}>layer</span>
              {STUDY_LAYERS.map((L) => {
                const on = source !== null && aL === L;
                return (
                  <button
                    key={L}
                    type="button"
                    aria-pressed={on}
                    aria-label={`Layer ${L}`}
                    onClick={() => runFind(curW(), String(L), String(curK()))}
                    style={chipStyle(on, { fontSize: 17, minWidth: 48, padding: "6px 10px", fontVariantNumeric: "lining-nums tabular-nums" })}
                  >
                    {L}
                  </button>
                );
              })}
            </div>
          </div>
          <Rubric style={{ maxWidth: 720 }}>
            Write any word and choose a layer from 1 to 60. The vault turns your word into a direction in the model’s space and lists
            the neurons, in that layer, whose writing points most nearly along it.
          </Rubric>
        </div>
        <ReadingKey />
      </div>

      <section aria-labelledby="answer-head" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <div className="row-between" style={{ paddingBottom: 4 }}>
          <h3 className="h-sub" id="answer-head">
            the vault answers
          </h3>
          <span className="num" style={{ fontSize: 15, color: "var(--muted)" }}>
            {meta}
          </span>
        </div>
        {pending ? (
          <p className="status-line" role="status">
            Asking the running instrument…
          </p>
        ) : null}
        {liveError && entry?.failure === "refused" ? (
          <p className="error-line" role="alert">
            The vault refused the search: {liveError}
          </p>
        ) : null}
        {liveError && entry?.failure !== "refused" ? (
          <p className="status-line" role="status">
            The search got no answer from the running instrument ({liveError}).
            {rec ? " What is shown is the study’s recorded answer." : ""}
          </p>
        ) : null}
        {source ? (
          <>
            <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: "4px 24px" }}>
              <label className="check" style={{ color: "var(--muted)" }}>
                <input type="checkbox" checked={ui.naked} onChange={() => ui.set({ naked: !ui.naked })} />
                also show what the naked eye reads
              </label>
              {kNote ? <span style={{ fontSize: 15, fontStyle: "italic", color: "var(--muted)" }}>{kNote}</span> : null}
            </div>
            <div style={{ overflowX: "auto", paddingBottom: 6 }}>
              <div style={{ minWidth: 1120 }} role="table" aria-label={meta}>
                <div
                  role="row"
                  style={{
                    display: "grid",
                    gridTemplateColumns: GRID,
                    gap: "0 20px",
                    alignItems: "end",
                    paddingBottom: 10,
                    borderBottom: "1px solid var(--hair)",
                    ...smallCaps(15),
                    lineHeight: 1.3,
                  }}
                >
                  <span role="columnheader">rank</span>
                  <span role="columnheader">neuron</span>
                  <span role="columnheader">face</span>
                  <span role="columnheader">nearness</span>
                  <span role="columnheader">its words, through the J++ lens</span>
                  <span role="columnheader">shared with the naked eye</span>
                  <span role="columnheader">when tested</span>
                </div>
                {rows.map((r, i) => (
                  <div
                    key={`${r.neuron}:${r.face}`}
                    role="row"
                    style={{
                      display: "grid",
                      gridTemplateColumns: GRID,
                      gap: "0 20px",
                      alignItems: "start",
                      padding: "16px 0 18px",
                      borderBottom: "1px solid var(--hair-faint)",
                    }}
                  >
                    <span role="cell" style={{ fontSize: 19, fontVariantCaps: "all-small-caps", letterSpacing: "0.06em", color: "var(--gilt-text)", lineHeight: 1.3 }}>
                      {roman(i)}
                    </span>
                    <div role="cell" style={{ display: "flex", flexDirection: "column", lineHeight: 1.3 }}>
                      <a
                        href={href.neuron(layerNo, r.neuron, r.face >= 0 ? 1 : -1)}
                        className="num"
                        style={{ fontSize: 19, alignSelf: "flex-start", padding: "10px 0", margin: "-10px 0" }}
                      >
                        neuron {r.neuron}
                      </a>
                      <span className="num" style={{ fontSize: 15, color: "var(--muted)" }}>
                        layer {layerNo}
                      </span>
                    </div>
                    <div role="cell" style={{ display: "flex", alignItems: "center", gap: 9, lineHeight: 1.3, paddingTop: 2 }}>
                      <FaceMark face={r.face} />
                      <span style={{ fontSize: 17 }}>{r.face > 0 ? "positive face" : "negative face"}</span>
                    </div>
                    <div role="cell" style={{ display: "flex", flexDirection: "column", gap: 9, lineHeight: 1.3, paddingTop: 2 }}>
                      <span className="num" style={{ fontSize: 17 }}>
                        {fmtCos(r.cos)}
                      </span>
                      <div aria-hidden="true" style={{ height: 3, background: "var(--hair-faint)", position: "relative" }}>
                        <div
                          style={{
                            position: "absolute",
                            left: 0,
                            top: 0,
                            bottom: 0,
                            width: `${f1(Math.max(0.8, (r.cos / scaleMax) * 100))}%`,
                            background: "var(--gilt)",
                          }}
                        />
                        <div style={{ position: "absolute", left: `${chanceL}%`, top: -4, width: 1, height: 11, background: "var(--muted)" }} />
                      </div>
                    </div>
                    <div role="cell" style={{ display: "flex", flexDirection: "column", gap: 8, minWidth: 0 }}>
                      {r.gate.heldBack ? (
                        // the label gate holds the words back (and their glosses with them); the apparatus brings them back
                        <>
                          <HeldBackWords gate={r.gate} words={r.words} eye={r.naked} compact style={{ fontSize: 17 }} />
                          {r.words.some((w) => GLOSSES[w]) ? (
                            <IfApparatus>
                              <span style={{ fontSize: 13, fontStyle: "italic", color: "var(--muted)", lineHeight: 1.3 }}>
                                {r.words
                                  .filter((w) => GLOSSES[w])
                                  .map((w) => `${show(w)}: ${GLOSSES[w]}`)
                                  .join("; ")}
                              </span>
                            </IfApparatus>
                          ) : null}
                        </>
                      ) : (
                        <>
                          <div style={{ display: "flex", flexWrap: "wrap", alignItems: "flex-start", gap: "6px 20px" }}>
                            {r.words.map((w, j) => (
                              <span key={j} style={{ display: "inline-flex", flexDirection: "column", gap: 1 }}>
                                <span style={{ fontSize: 22, lineHeight: 1.25 }}>{show(w)}</span>
                                <span style={{ fontSize: 13, fontStyle: "italic", color: "var(--muted)", lineHeight: 1.3 }}>{GLOSSES[w] || ""}</span>
                              </span>
                            ))}
                          </div>
                          {ui.naked ? (
                            <span style={{ fontSize: 15, color: "var(--muted)" }}>
                              <span style={{ fontVariantCaps: "all-small-caps", letterSpacing: "0.14em", marginRight: 10 }}>naked eye</span>
                              {r.naked.map(show).join("  ")}
                            </span>
                          ) : null}
                        </>
                      )}
                    </div>
                    <span role="cell" className="num" style={{ fontSize: 17, lineHeight: 1.3, paddingTop: 2 }}>
                      {/* the gate's own count, ten a side by id, where the server sent it; else the five shown, spelt exactly */}
                      {r.gate.reasons.includes("writes_nothing")
                        ? "writes nothing"
                        : r.gate.sharedTop10 != null
                          ? `${r.gate.sharedTop10} of 10`
                          : `${r.agree} of 5 shown`}
                      {r.gate.condition3 === "unvalidated" ? (
                        <span style={{ display: "block", fontSize: 14, fontStyle: "italic", color: "var(--muted)" }}>
                          {r.gate.edited ? "edited in the vault" : "under another lens"}
                        </span>
                      ) : null}
                    </span>
                    <div role="cell" style={{ display: "flex", flexDirection: "column", lineHeight: 1.3, paddingTop: 2 }}>
                      <span
                        style={{
                          fontSize: 16,
                          color: r.tested === 1 ? "var(--ink)" : "var(--muted)",
                          fontStyle: r.tested === null ? "italic" : "normal",
                        }}
                      >
                        {r.tested === 1 ? "told Mary from John" : r.tested === 0 ? "could not tell" : "not tested"}
                      </span>
                      <span className="num" style={{ fontSize: 14, color: "var(--muted)" }}>
                        {r.d === null ? "" : "difference " + signedFixed(r.d, 1)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </>
        ) : (
          <EmptyAnswer head="[THE VAULT’S ANSWER APPEARS HERE AFTER THE SEARCH]">
            This page carries only the twenty-four searches made in the study: “she”, “he”, “her” and “woman” at layers 21, 31, 41,
            51, 56 and 59. At layers 21 and 31 their neurons’ words are held back, as every neuron’s are below layer 41
            {RECORDED_CHECKED
              ? `; at the higher layers, so are the rows whose two readings share none of their first ten (${recordedVaultHeld(40).held} of the ${recordedVaultHeld(40).rows})`
              : ""}
            . The neurons, their nearness and the study’s test still stand. Any other search is answered by the running instrument.
          </EmptyAnswer>
        )}
        <IfRubrics>
          {source ? (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 320px), 1fr))",
                gap: "14px 48px",
                paddingTop: 12,
              }}
            >
              <Rubric style={{ maxWidth: "none" }}>
                Treat each row as a lead to test, not a finding. As LARQL’s own README warns, “a projected contribution is not a
                counterfactual”: nearness shows what a neuron would write, not what it does.
              </Rubric>
              <Rubric style={{ maxWidth: "none" }}>{chanceNote}</Rubric>
              <Rubric style={{ maxWidth: "none" }}>
                Some rows are spelling neighbours: words that share letters with yours, not meaning. In the study’s searches, “usher”,
                “shed”, “heap” and “helix” came up this way.
              </Rubric>
              <Rubric style={{ maxWidth: "none" }}>
                Neuron words are held back below layer 41, and wherever the lens and the naked eye share none of their first ten; the
                cut at none is the instrument’s, since the study set none. Inside layers 41 to 58, in the study’s sample of random
                neurons, the more words shared of the first ten, the likelier a sound label (the count, used as a score: AUC{" "}
                {GATE_FIGURES.auc}, for labels read through the {isShippedLens(vaultLens) ? "" : "base-fitted "}J++ lens); sharing just one
                was a weak sign, sound{" "}
                {OVERLAP_RECORD[1][1]} times in {OVERLAP_RECORD[1][0]} against {GATE_FIGURES.bandSound} in {GATE_FIGURES.band} for the band.
                Of the 102 neurons the study tested, 35 told Mary from John.
              </Rubric>
              {allHeld && heldRows[0] ? (
                <Rubric style={{ maxWidth: "none" }}>
                  {`Every row’s words are held back here. ${heldRubric(heldRows[0].gate)} `}
                  “Told Mary from John” is a test of what wakes a neuron, how it fires on sentences about women and about men, not of
                  its words, so it still stands beside a held-back row.
                </Rubric>
              ) : null}
            </div>
          ) : null}
        </IfRubrics>
        <Apparatus title="find, the nearness search" label="Apparatus for find" style={{ marginTop: 8 }}>
          <dl className="dl">
            <dt>technical name</dt>
            <dd>
              <span className="mono-wrap">SELECT * FROM EDGES NEAREST</span>, run by{" "}
              <span className="mono-wrap">LarqlVaultEngine.query_nearest_edges</span> on the down-projection, through the J++ lens
              (Ayonrinde &amp; Lindsey 2026) [4]. POLES in place of EDGES runs the very same search: the parser ignores which is
              written (E6 §3, item 6).
            </dd>
            <dt>nearness</dt>
            <dd>
              q = (W<sub>U</sub>[t] ⊙ g) · J<sub>ℓ</sub>, normalised to length 1; for neuron i, w<sub>i</sub> = W<sub>down</sub>[:, i] ⊙
              g<sub>post,ℓ</sub>, and nearness<sub>i</sub> = |⟨q, w<sub>i</sub>⟩| / (‖w<sub>i</sub>‖ + 10<sup>−8</sup>), its face the sign
              of ⟨q, w<sub>i</sub>⟩. Here t is your word’s word-piece (the word with a leading space if that is one piece, else the bare
              word if it is, else the bare word’s first piece), g the final-norm weight, g<sub>post,ℓ</sub> the post-feed-forward norm
              weight and J<sub>ℓ</sub> the lens of layer ℓ. The k largest are listed.
            </dd>
            <dt>its words</dt>
            <dd>
              through the lens, the top five of unembed((±w<sub>i</sub>) · J<sub>ℓ</sub>
              <sup>⊤</sup>); to the naked eye (the logit lens [8]), the top five of unembed(±w<sub>i</sub>); where unembed(u) = 30 ·
              tanh(W<sub>U</sub> · RMSNorm(u) / 30).
            </dd>
            <dt>chance</dt>
            <dd className="num">
              two unrelated directions in 5,376 dimensions share about 1/√5376 ≈ 0.0136 (E6 §3, item 6). {thisSearch}
            </dd>
            <dt>at this layer</dt>
            <dd className="num">{layerNote}</dd>
            <dt>the contrast</dt>
            <dd className="num">
              The study’s proposed ranking (E6 §3, item 6): |cos| with ((W<sub>U</sub>[she] − W<sub>U</sub>[he]) ⊙ g) · J<sub>ℓ</sub> in
              place of a single word. {contrast}
            </dd>
            <dt>caveat</dt>
            <dd>
              A projected contribution, not a counterfactual (REPORT §3.3). At layers 31–59 nearness ran 0.0004–0.02, at or below
              chance: from layer 36 the pullback puts 99–100% of its energy in the quietest tenth of the model’s dimensions (E6 §4b).
              Output labels held up at layers 41–58 and were at chance at 11–31; layers 32–40 were not sampled (REPORT §3.2; N1). Within
              41–58, the overlap of the J++ lens’s and the naked eye’s top ten, used as a graded score, predicts a real effect (AUC 0.84,
              for base-fitted J++ labels on random neurons); the instrument’s yes/no cut at none of ten scores AUC {GATE_FIGURES.cutAuc} on
              the same sample (experiments/results/N1C_gate_cut.json).
            </dd>
            <dt>held back</dt>
            <dd>
              The board holds a row’s words, and their glosses, back from its main reading below layer 41 (condition 2) and where the
              two readings share none of their first ten word-pieces (condition 3; REPORT §3.3), counted by id in fp32 as N1 counted
              (label_check.shared_top10). The cut at none of ten is the instrument’s; it applies under the base-fitted J++ lens only
              (erratum E47), and to a neuron edited in the vault as well, though the gate was never measured on edited neurons. A
              neuron silenced in the vault (DELETE) writes nothing, has no count, and is held back (label_check reason
              writes_nothing). With the apparatus open the words come back, muted.{" "}
              {source === "recorded" ? (
                RECORDED_CHECKED ? (
                  "The study’s recorded searches were checked ten a side once, from the weights (experiments/results/N1G_recorded_label_checks.json)."
                ) : (
                  <>
                    The study’s recorded searches kept five word-pieces a side, too few for the measure; their ten-a-side count:{" "}
                    <Missing what="not yet checked ten a side" />. Until then only the layer holds a recorded row back.
                  </>
                )
              ) : source === "live" && !anyTen ? (
                <>
                  This reply carries no ten-a-side count: <Missing what="not yet in the server" />. Only the layer holds a row back.
                </>
              ) : null}
            </dd>
            <dt>not reachable</dt>
            <dd>
              <span className="mono-wrap">query_nearest_edges</span> also takes <span className="mono-wrap">matrix_target</span> (the
              input side, gate_proj, which INSERT uses) and <span className="mono-wrap">use_jpp</span> (the bare word direction);
              neither the query language nor the server passes them <span className="placeholder">[NOT EXPOSED BY THE SERVER]</span>.
            </dd>
          </dl>
          <div role="group" aria-label="Filter on the five word-pieces shown" style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "6px 8px", paddingTop: 6 }}>
            <span style={{ fontVariantCaps: "all-small-caps", letterSpacing: "0.14em", color: "var(--muted)", marginRight: 6 }}>
              a filter on the five shown: keep rows where lens and naked eye share at least
            </span>
            {[0, 2, 3, 4, 5].map((g) => (
              <button
                key={g}
                type="button"
                onClick={() => ui.set({ gate: g })}
                aria-pressed={ui.gate === g}
                aria-label={g === 0 ? "0: keep every row" : `${g}: keep rows that share at least ${g} of five word-pieces`}
                style={chipStyle(ui.gate === g, { fontSize: 15, padding: "0 10px", fontVariantNumeric: "lining-nums tabular-nums" })}
              >
                {g}
              </button>
            ))}
            <span style={{ color: "var(--muted)" }}>
              of the five word-pieces shown, spelt exactly (REPORT §3.5, item 5). It applies while the apparatus is open; it is not the
              label gate, which counts ten a side.
            </span>
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "8px 12px", paddingTop: 4 }}>
            <button className="btn-ghost" type="button" onClick={() => toConsole(lql)}>
              write this search by hand
            </button>
            <button className="btn-ghost" type="button" onClick={() => ui.set({ faRaw: !ui.faRaw })} aria-expanded={ui.faRaw}>
              {rawLabel}
            </button>
          </div>
          {ui.faRaw ? (
            <>
              <span style={{ color: "var(--muted)" }}>{rawCaption}</span>
              <pre className="raw" style={{ maxHeight: 360 }}>
                {rawJsonText}
              </pre>
            </>
          ) : null}
        </Apparatus>
      </section>
    </section>
  );
};

/** The board's key: how to read a row. */
const ReadingKey: React.FC = () => (
  <aside aria-labelledby="key-head" style={{ flex: "1 1 340px", minWidth: 0, display: "flex", flexDirection: "column", gap: 16, paddingTop: 2 }}>
    <h3 className="h-sub" id="key-head">
      how to read a row
    </h3>
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "22px minmax(0, 1fr)",
        gap: "2px 12px",
        alignItems: "baseline",
        paddingBottom: 14,
        borderBottom: "1px solid var(--hair-faint)",
      }}
    >
      <FaceMark face={1} />
      <div>
        <span style={{ fontSize: 18 }}>positive face</span>
        <span style={{ display: "block", fontSize: 15, color: "var(--muted)" }}>Firing the usual way, the neuron pushes toward these words.</span>
      </div>
      <FaceMark face={-1} style={{ marginTop: 10 }} />
      <div style={{ marginTop: 10 }}>
        <span style={{ fontSize: 18 }}>negative face</span>
        <span style={{ display: "block", fontSize: 15, color: "var(--muted)" }}>
          Firing the other way, it pushes toward them instead. <a href={href.lexicon("neuron")}>What is a face?</a>
        </span>
      </div>
    </div>
    <div style={{ display: "flex", flexDirection: "column", gap: 6, paddingBottom: 14, borderBottom: "1px solid var(--hair-faint)" }}>
      <span style={{ fontSize: 18 }}>nearness</span>
      <div aria-hidden="true" style={{ height: 3, background: "var(--hair-faint)", position: "relative", margin: "6px 0 4px", maxWidth: 220 }}>
        <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: "22%", background: "var(--gilt)" }} />
        <div style={{ position: "absolute", left: "78%", top: -4, width: 1, height: 11, background: "var(--muted)" }} />
      </div>
      <span style={{ fontSize: 15, color: "var(--muted)" }}>
        How nearly the neuron’s writing points along your word, from 0 to 1. The upright hairline marks about 0.014: what two unrelated
        directions in this space typically share.
      </span>
    </div>
    <div style={{ display: "flex", flexDirection: "column", gap: 2, paddingBottom: 14, borderBottom: "1px solid var(--hair-faint)" }}>
      <span style={{ fontSize: 18 }}>shared with the naked eye</span>
      <span style={{ fontSize: 15, color: "var(--muted)" }}>
        How many word-pieces the plain reading, without the lens, shares with the lens: of their first ten, as the study counted (by the
        running instrument, or, for the study’s recorded searches, by a check made once from the weights), else of the five shown. More
        agreement, more trust, at layers 41 to 58, where the study tested it, though one of ten was a weak sign; a row that shares none
        of ten has its words held back.
      </span>
    </div>
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span style={{ fontSize: 18 }}>when tested</span>
      <span style={{ fontSize: 15, color: "var(--muted)" }}>
        Whether it fired differently, beyond chance, for twelve sentences about women (Mary, Emma, Sarah …) than for twelve about men
        (John, James, David …). The small number beneath is the size of that difference <span style={{ fontStyle: "italic" }}>(Cohen’s d)</span>
        : 0.8 or more, either way, is large. Only neurons found by the “she” and “he” searches were tested; some turn up in the other
        searches too.
      </span>
    </div>
  </aside>
);

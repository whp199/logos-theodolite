import React from "react";
import { Apparatus, Rubric } from "../../components/kit";
import { useTheodoliteStore, type LarqlEntry } from "../../store/useTheodoliteStore";
import { OVERLAP, STUDY_LAYERS, STUDY_WORDS, TRIALS, TRIALS_B, VAULT_ANSWERS } from "./recorded";
import { clock, entrySummary, prettyJson } from "./live";
import { esc } from "./lql";
import { MONO, show, signedFixed, smallCaps } from "./bits";
import { runFind, toConsole } from "./ui";
import { HeldBackWords } from "../../components/HeldBack";
import { heldRubric, labelGate, layerIdxOfPage, vaultCheck } from "../../lib/labels";

const codeStyle: React.CSSProperties = { fontFamily: MONO, fontSize: 15, color: "var(--ink)", overflowWrap: "anywhere" };
const smallMono: React.CSSProperties = { fontFamily: MONO, fontSize: 13, overflowWrap: "anywhere" };
const numeral: React.CSSProperties = {
  fontSize: 30,
  lineHeight: 1,
  fontWeight: 500,
  color: "var(--gilt-numeral)",
  fontVariantCaps: "all-small-caps",
};
const entryGrid: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "56px minmax(0, 1fr)",
  gap: "4px 18px",
  paddingTop: 18,
  borderTop: "1px solid var(--hair)",
};

export const HistoryPanel: React.FC = () => {
  const larqlLog = useTheodoliteStore((s) => s.larqlLog);

  return (
    <section role="tabpanel" id="panel-history" aria-labelledby="tab-history" style={{ paddingTop: 34, display: "flex", flexDirection: "column", gap: 34 }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 12, maxWidth: 780 }}>
        <h2 className="h-sec">ii · the ledger of past questions</h2>
        <p style={{ margin: 0, fontSize: 21, lineHeight: 1.5 }}>
          One session from the study, 16 October 2026: every question it put to the vault, in order, through the instrument’s own code. The
          session took 102 seconds.
        </p>
        <Rubric>Choose any search to ask it again; its answer opens under find.</Rubric>
      </div>

      <div style={entryGrid}>
        <span style={numeral}>i</span>
        <div style={{ display: "flex", flexDirection: "column", gap: 6, minWidth: 0 }}>
          <h3 className="h-sub">the vault describes itself</h3>
          <code style={codeStyle}>DESCRIBE</code>
          <span className="num" style={{ fontSize: 17 }}>
            60 layers · 21,504 neurons in each · 1,290,240 in all · 2,580,480 faces · 262,144 word-pieces · vindex 0.8.0 · larql 0.2.0 · no
            edits in force
          </span>
        </div>
      </div>

      <div style={entryGrid}>
        <span style={numeral}>ii</span>
        <div style={{ display: "flex", flexDirection: "column", gap: 10, minWidth: 0 }}>
          <h3 className="h-sub">twenty-four searches</h3>
          <code style={codeStyle}>SELECT * FROM POLES NEAREST 'she' WHERE layer = 20 LIMIT 10</code>
          <span style={{ fontSize: 16, color: "var(--muted)" }}>
            and so on, for four words at six layers. The vault counts from nought, so{" "}
            <span style={{ fontFamily: MONO, fontSize: 14, overflowWrap: "anywhere" }}>layer = 20</span> is layer 21 here. POLES and EDGES run
            the same search in this vault. Each cell shows the nearest neuron and its first word through the lens; below layer 41 the word is
            held back, and kept in the apparatus.
          </span>
          <Rubric>{heldRubric(labelGate(layerIdxOfPage(STUDY_LAYERS[0]), null))}</Rubric>
          <div style={{ overflowX: "auto", paddingBottom: 6 }}>
            <div style={{ minWidth: 760 }}>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "110px repeat(4, minmax(140px, 1fr))",
                  gap: "0 18px",
                  paddingBottom: 8,
                  borderBottom: "1px solid var(--hair)",
                  fontSize: 20,
                  fontStyle: "italic",
                }}
              >
                <span></span>
                {STUDY_WORDS.map((w) => (
                  <span key={w}>“{w}”</span>
                ))}
              </div>
              {STUDY_LAYERS.map((L) => (
                <div
                  key={L}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "110px repeat(4, minmax(140px, 1fr))",
                    gap: "0 18px",
                    alignItems: "center",
                    borderBottom: "1px solid var(--hair-faint)",
                  }}
                >
                  <span className="num" style={smallCaps(15)}>
                    layer {L}
                  </span>
                  {STUDY_WORDS.map((w) => {
                    const top = VAULT_ANSWERS[`${w}@${L - 1}`][0];
                    // the label gate (lib/labels.ts): by its recorded check, or by the layer alone until it is checked
                    const gate = labelGate(layerIdxOfPage(L), vaultCheck(`${w}@${L - 1}`, top[0], top[1]));
                    return (
                      <button
                        key={w}
                        type="button"
                        onClick={() => runFind(w, String(L), "10")}
                        aria-label={`Ask again: “${w}” in layer ${L}; nearest neuron ${top[0]}`}
                        style={{
                          fontFamily: "inherit",
                          background: "transparent",
                          border: 0,
                          padding: "10px 0",
                          minHeight: 60,
                          cursor: "pointer",
                          display: "flex",
                          flexWrap: "wrap",
                          alignItems: "baseline",
                          gap: "2px 10px",
                          textAlign: "left",
                          color: "var(--ink)",
                        }}
                      >
                        <span className="num" style={{ fontSize: 17 }}>
                          {top[0]}
                        </span>
                        {gate.heldBack ? (
                          <HeldBackWords gate={gate} words={top[3].slice(0, 1)} eye={top[4].slice(0, 1)} compact style={{ fontSize: 15 }} />
                        ) : (
                          <span style={{ fontSize: 19, fontStyle: "italic" }}>{show(top[3][0])}</span>
                        )}
                        <span
                          style={{
                            fontSize: 15,
                            color: "var(--rubric)",
                            textDecoration: "underline",
                            textDecorationThickness: 1,
                            textUnderlineOffset: 3,
                          }}
                        >
                          ask again
                        </span>
                      </button>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div style={entryGrid}>
        <span style={numeral}>iii</span>
        <div style={{ display: "flex", flexDirection: "column", gap: 10, minWidth: 0 }}>
          <h3 className="h-sub">twelve trials: silence ten, measure, undo</h3>
          <div style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: "4px 12px", fontSize: 16, color: "var(--muted)" }}>
            <code style={codeStyle}>DELETE FROM LAYER 50 WHERE slot = 5313</code>
            <span style={{ fontStyle: "italic" }}>once for each of the ten, then</span>
            <code style={codeStyle}>ROLLBACK</code>
          </div>
          <span style={{ fontSize: 16, color: "var(--muted)", maxWidth: 780 }}>
            Each trial silenced ten neurons with the vault’s own edit, measured how strongly the model then preferred “she”-words over
            “he”-words, and undid the edit before the next. Before any edit, sentences about women stood at +7.26 and sentences about men at
            −5.64; the table gives how far each trial moved them.
          </span>
          <div style={{ overflowX: "auto", paddingBottom: 6 }}>
            <div style={{ minWidth: 820 }}>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "90px minmax(0, 1fr) 170px 170px",
                  gap: "0 20px",
                  alignItems: "end",
                  padding: "0 10px 8px",
                  borderBottom: "1px solid var(--hair)",
                  ...smallCaps(15),
                  lineHeight: 1.3,
                }}
              >
                <span>layer</span>
                <span>which ten fell silent</span>
                <span style={{ textAlign: "right" }}>shift, sentences about women</span>
                <span style={{ textAlign: "right" }}>shift, sentences about men</span>
              </div>
              {TRIALS.map((t, i) => (
                <div
                  key={i}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "90px minmax(0, 1fr) 170px 170px",
                    gap: "0 20px",
                    alignItems: "baseline",
                    padding: "10px 10px 11px",
                    borderBottom: "1px solid var(--hair-faint)",
                    background: t[1] === 50 && t[0] !== "random" ? "var(--pressed)" : "transparent",
                  }}
                >
                  <span className="num" style={{ fontSize: 17 }}>
                    {t[1] + 1}
                  </span>
                  <div style={{ display: "flex", flexDirection: "column", minWidth: 0 }}>
                    <span style={{ fontSize: 17 }}>{t[0] === "random" ? "ten at random" : `the ten nearest “${t[0]}”`}</span>
                    <span className="num" style={{ fontSize: 14, color: "var(--muted)" }}>
                      {t[0] === "random"
                        ? "their numbers were not kept in the results file"
                        : VAULT_ANSWERS[`${t[0]}@${t[1]}`].map((r) => r[0]).join(", ")}
                    </span>
                  </div>
                  <span className="num" style={{ fontSize: 17, textAlign: "right" }}>
                    {signedFixed(t[2], 2)}
                  </span>
                  <span className="num" style={{ fontSize: 17, textAlign: "right" }}>
                    {signedFixed(t[3], 2)}
                  </span>
                </div>
              ))}
            </div>
          </div>
          <Rubric style={{ maxWidth: 780 }}>
            The largest shift came at layer 51: silencing its ten nearest “she” neurons moved sentences about women by −1.80, against no more
            than 0.04 for ten random neurons. The “he” list did the same, because both lists share neuron 5313, which has a face for each.
          </Rubric>
        </div>
      </div>

      <div style={entryGrid}>
        <span style={numeral}>iv</span>
        <div style={{ display: "flex", flexDirection: "column", gap: 6, minWidth: 0 }}>
          <h3 className="h-sub">every edit undone</h3>
          <code style={codeStyle}>ROLLBACK</code>
          <span style={{ fontSize: 17 }}>
            Afterwards the model’s readings matched the untouched model exactly: the largest difference was <span className="num">0.0</span>.
          </span>
        </div>
      </div>

      <p style={{ margin: 0, fontSize: 15, color: "var(--muted)" }}>
        From <span style={smallMono}>experiments/results/E6_tool_audit_04_vault.partial.json</span>, written by{" "}
        <span style={smallMono}>experiments/scripts/E6_tool_audit_04_vault.py</span>.
      </p>

      <SessionLog entries={larqlLog} />

      <Apparatus title="the ledger" label="Apparatus for the ledger">
        <dl className="dl">
          <dt>statements</dt>
          <dd>
            <span className="mono-wrap">SELECT * FROM POLES NEAREST '&lt;word&gt;' WHERE layer = &lt;l&gt; LIMIT 10</span> for the four words at
            index 20, 30, 40, 50, 55 and 58; then, per trial at index 40, 50, 55 and 58,{" "}
            <span className="mono-wrap">DELETE FROM LAYER &lt;l&gt; WHERE slot = &lt;i&gt;</span> ten times and{" "}
            <span className="mono-wrap">ROLLBACK</span>. The random tens were drawn with seed 7.
          </dd>
          <dt>she and he share</dt>
          <dd className="num">
            {STUDY_LAYERS.map((L) => "layer " + L + ": " + OVERLAP[L - 1][0] + " (" + OVERLAP[L - 1][1] + " with the same face)").join(" · ") +
              ". The “he” ten at layer 41 includes neuron 16757, which causally pushes toward “she” (REPORT §3.3)."}
          </dd>
          <dt>the study’s test</dt>
          <dd className="num">
            a paired sign-flip permutation test (2,000 draws) on the last word-piece of twelve sentences about women against twelve about men;
            35 of 102 neurons selective at BH q &lt; 0.05, 30 with |d| ≥ 0.8 (E6 §2.3).
          </dd>
          <dt>the tool’s own test</dt>
          <dd className="num">
            <span className="mono-wrap">test_motif_hypothesis</span>, the MCP scientist’s, on neuron 5313 of layer 51:{" "}
            <span className="mono-wrap">CONFIRMED_SOUTH_POLE (-)</span> with d = −22.38 on three sentences against three, and −5.48 on twelve
            against twelve; on neuron 15381 of layer 59, −1.91 and −1.25. That verdict, still sent for old callers, uses the population variance
            and no significance test, and says CONFIRMED at |d| ≥ 0.8 for any number of sentences (E6 §3, item 7); its test against chance
            (Welch’s t, a permutation p, at least four sentences a side) came after.
          </dd>
          <dt>other readings</dt>
          <dd style={{ margin: 0, display: "flex", flexDirection: "column", gap: 4, minWidth: 0 }}>
            <span>
              the same trials, read on Gemma describing itself (baseline +1.54) and on a description of a human friend named Gemma (baseline
              +8.41):
            </span>
            <div style={{ overflowX: "auto" }}>
              <div className="num" style={{ minWidth: 520 }}>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "60px minmax(0, 1fr) 120px 120px",
                    gap: "0 16px",
                    padding: "4px 0",
                    borderBottom: "1px solid var(--hair)",
                    fontVariantCaps: "all-small-caps",
                    letterSpacing: "0.14em",
                    color: "var(--muted)",
                  }}
                >
                  <span>layer</span>
                  <span>which ten</span>
                  <span style={{ textAlign: "right" }}>Gemma, itself</span>
                  <span style={{ textAlign: "right" }}>human Gemma</span>
                </div>
                {TRIALS.map((t, i) => (
                  <div
                    key={i}
                    style={{
                      display: "grid",
                      gridTemplateColumns: "60px minmax(0, 1fr) 120px 120px",
                      gap: "0 16px",
                      padding: "3px 0",
                      borderBottom: "1px solid var(--hair-faint)",
                    }}
                  >
                    <span>{t[1] + 1}</span>
                    <span>{t[0] === "random" ? "ten at random" : `the ten nearest “${t[0]}”`}</span>
                    <span style={{ textAlign: "right" }}>{signedFixed(TRIALS_B[i][0], 2)}</span>
                    <span style={{ textAlign: "right" }}>{signedFixed(TRIALS_B[i][1], 2)}</span>
                  </div>
                ))}
              </div>
            </div>
          </dd>
          <dt>kept in the file</dt>
          <dd>
            <span style={smallMono}>
              describe, lists, overlap, contrast_lists, behavior, neuron_tests, tool_test_motif, rollback_check_max_abs_diff, causal_delete
            </span>
            ; the session ran 101.93 s. Every vault reply on this page comes from these keys; any other is{" "}
            <span className="placeholder">[MEASURED IN A LIVE SIGHTING]</span>.
          </dd>
        </dl>
      </Apparatus>
    </section>
  );
};

/** This session's own questions to the running vault (the store's log, newest first). */
const SessionLog: React.FC<{ entries: LarqlEntry[] }> = ({ entries }) => (
  <section aria-labelledby="session-head" style={{ display: "flex", flexDirection: "column", gap: 10, paddingTop: 18, borderTop: "1px solid var(--hair)" }}>
    <h3 className="h-sub" id="session-head">
      asked in this session
    </h3>
    {entries.length === 0 ? (
      <p style={{ margin: 0, fontSize: 17, color: "var(--muted)" }}>
        Nothing yet. Every statement this page sends to the running instrument is listed here, newest first, with its reply.
      </p>
    ) : (
      <ol style={{ listStyle: "none", margin: 0, padding: 0, display: "flex", flexDirection: "column" }}>
        {entries.map((e) => (
          <li
            key={e.id}
            style={{
              display: "grid",
              gridTemplateColumns: "minmax(70px, 90px) minmax(0, 1fr)",
              gap: "2px 18px",
              padding: "10px 0 8px",
              borderBottom: "1px solid var(--hair-faint)",
            }}
          >
            <span className="num" style={{ fontSize: 15, color: "var(--muted)" }}>
              {clock(e.at)}
            </span>
            <div style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 0 }}>
              <code style={codeStyle}>{e.query}</code>
              <span style={{ fontSize: 16, color: e.ok ? "var(--ink)" : "var(--rubric)" }}>{entrySummary(e)}</span>
              <details>
                <summary className="btn-link" style={{ width: "fit-content" }}>
                  its reply
                </summary>
                <pre className="raw">{esc(e.ok ? prettyJson(e.data) : prettyJson({ detail: e.error }))}</pre>
                <button type="button" className="btn-link" onClick={() => toConsole(e.query)}>
                  write it again, in the console
                </button>
              </details>
            </div>
          </li>
        ))}
      </ol>
    )}
  </section>
);

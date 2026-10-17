import React, { useEffect, useMemo, useState } from "react";
import { Apparatus, IfApparatus, Missing, Page, RecordedNotice, Rubric, SweepError } from "../components/kit";
import { useApparatus } from "../reading";
import { href, type Route } from "../router";
import {
  type LayerReading,
  type Sighting,
  exportSighting,
  retakeIfLive,
  useTheodoliteStore,
} from "../store/useTheodoliteStore";
import { certainty, comma, gloss, kindOf, labelTrust, minus, ms, pct, signed, wp } from "../lib/words";
import { InstrumentFigure, type Shown } from "./instrument/InstrumentFigure";
import { Ascent } from "./instrument/Ascent";
import { Kinship } from "./instrument/Kinship";
import { CANVAS_STATIONS, RECORDED_POLE_NOTES } from "./instrument/recorded";
import { ControlSighting } from "../components/ControlSighting";

const sci = (v: number) => minus(Math.abs(v) >= 0.01 ? v.toFixed(3) : v.toExponential(1));

function stationsOf(s: Sighting): number[] {
  const recorded = s.layers.filter((L) => L.lensTop && L.lensTop.length).map((L) => L.idx);
  // a live sighting reads every layer; keep the canvas's nine stations so the words stay legible
  return recorded.length > 12 ? CANVAS_STATIONS.filter((i) => recorded.includes(i)) : recorded;
}

export const Instrument: React.FC<{ route: Route }> = ({ route }) => {
  const sighting = useTheodoliteStore((s) => s.sighting);
  const sel = useTheodoliteStore((s) => s.selectedLayer);
  const setSel = useTheodoliteStore((s) => s.setSelectedLayer);
  const apparatus = useApparatus();
  const [shown, setShown] = useState<Shown>(() => ({
    path: true,
    words: true,
    split: true,
    neurons: apparatus,
    threads: apparatus,
    marks: true,
    rings: true,
  }));
  const stations = useMemo(() => (sighting ? stationsOf(sighting) : []), [sighting]);

  return (
    <Page route={route} folio="folio i">
      <Question />
      {sighting ? (
        <>
          <WentOn sighting={sighting} />
          <div style={{ display: "flex", flexWrap: "wrap", gap: "40px 56px", paddingTop: 34 }}>
            <section aria-labelledby="inst-head" style={{ flex: "999 1 640px", minWidth: 0, display: "flex", flexDirection: "column", gap: 14 }}>
              <div className="row-between">
                <h2 id="inst-head" className="h-sec">
                  ii · the instrument
                </h2>
                <span style={{ fontSize: 15, color: "#6B5D4D" }}>drag to turn · scroll to draw near</span>
              </div>
              <InstrumentFigure sighting={sighting} sel={sel} stations={stations} shown={shown} onPick={setSel} />
              <fieldset style={{ border: 0, margin: 0, padding: 0, display: "flex", flexWrap: "wrap", alignItems: "center", gap: "2px 22px" }}>
                <legend
                  style={{ float: "left", padding: 0, marginRight: 6, fontVariantCaps: "all-small-caps", letterSpacing: "0.16em", fontSize: 15, color: "#6B5D4D" }}
                >
                  show
                </legend>
                {(
                  [
                    ["path", "path of thought"],
                    ["words", "words it was thinking"],
                    ["split", "gathering & remembering"],
                    ["neurons", "neurons that fired"],
                    ["threads", "threads"],
                    ["marks", "landmarks"],
                    ["rings", "layer rings"],
                  ] as [keyof Shown, string][]
                ).map(([k, label]) => (
                  <label key={k} className="check" style={{ gap: 8 }}>
                    <input type="checkbox" checked={shown[k]} onChange={() => setShown((s) => ({ ...s, [k]: !s[k] }))} />
                    {label}
                  </label>
                ))}
              </fieldset>
              <Rubric style={{ margin: 0, maxWidth: 700 }}>
                Each ring is one layer: the first at the bottom, the sixtieth at the top. The dark thread is the path of the model’s thought. On
                the gilded rings, every sixth, the model looks back over the whole conversation, not only the recent words.
              </Rubric>
              <WhatIsDrawn sighting={sighting} />
            </section>
            <Leaf sighting={sighting} sel={sel} stations={stations} onPick={setSel} />
          </div>
          <Ascent sighting={sighting} sel={sel} stations={stations} onPick={setSel} />
          <IfApparatus>
            <Kinship sighting={sighting} sel={sel} />
          </IfApparatus>
        </>
      ) : (
        <p className="status-line" style={{ padding: "40px 0" }}>
          Waiting for the first sighting…
        </p>
      )}
    </Page>
  );
};

/* ------------------------------------------------------------ i · the question */

const Question: React.FC = () => {
  const question = useTheodoliteStore((s) => s.question);
  const answerBegins = useTheodoliteStore((s) => s.answerBegins);
  const setQuestion = useTheodoliteStore((s) => s.setQuestion);
  const setAnswerBegins = useTheodoliteStore((s) => s.setAnswerBegins);
  const compareThink = useTheodoliteStore((s) => s.compareThink);
  const setCompareThink = useTheodoliteStore((s) => s.setCompareThink);
  const chat = useTheodoliteStore((s) => s.chat);
  const setChat = useTheodoliteStore((s) => s.setChat);
  const system = useTheodoliteStore((s) => s.system);
  const setSystem = useTheodoliteStore((s) => s.setSystem);
  const takeSighting = useTheodoliteStore((s) => s.takeSighting);
  const isSweeping = useTheodoliteStore((s) => s.isSweeping);
  const sighting = useTheodoliteStore((s) => s.sighting);
  const lastWord = (answerBegins.trim() || question.trim()).split(/\s+/).pop() ?? "";

  return (
    <>
      <section aria-labelledby="q-head" style={{ display: "flex", flexWrap: "wrap", gap: "28px 56px", padding: "34px 0 30px", alignItems: "flex-end" }}>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            takeSighting();
          }}
          style={{ flex: "999 1 620px", minWidth: 0, display: "flex", flexDirection: "column", gap: 16 }}
        >
          <h2 id="q-head" className="h-sec">
            i · the question
          </h2>
          <div className="q-grid">
            <label htmlFor="ask" className="label-sc">
              you ask
            </label>
            <input
              id="ask"
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              className="q-ask"
              style={{ lineHeight: 1.25, background: "transparent", border: 0, borderBottom: "1px solid #C9BEAC", padding: "4px 0 8px", width: "100%", outlineOffset: 6, borderRadius: 0 }}
            />
            <label htmlFor="begin" className="label-sc">
              its answer begins
            </label>
            <div style={{ display: "flex", alignItems: "baseline", gap: 16, flexWrap: "wrap" }}>
              <input
                id="begin"
                type="text"
                value={answerBegins}
                onChange={(e) => setAnswerBegins(e.target.value)}
                style={{ fontStyle: "italic", fontSize: 26, background: "transparent", border: 0, borderBottom: "1px solid #C9BEAC", padding: "2px 0 6px", width: 180, borderRadius: 0 }}
              />
              <span style={{ fontSize: 16, color: "#6B5D4D" }}>
                optional — the instrument reads the last word,{" "}
                <span style={{ color: "#2B2117", borderBottom: "2px solid #A07A2E" }}>“{lastWord}”</span>
              </span>
            </div>
          </div>
          <Rubric style={{ margin: "2px 0 0", maxWidth: 640 }}>
            Write anything you like, then take a sighting. The instrument follows the model’s last word up through all sixty of its layers,
            and shows what it was thinking at each one.
          </Rubric>
          <button type="submit" hidden aria-hidden="true" tabIndex={-1} />
        </form>
        <div style={{ flex: "1 1 300px", display: "flex", flexDirection: "column", gap: 6, alignItems: "flex-start" }}>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 12, alignItems: "center", marginBottom: 8 }}>
            <button type="button" className="btn-primary" onClick={() => takeSighting()} disabled={isSweeping}>
              {isSweeping ? "taking a sighting…" : "take a sighting"}
            </button>
            <a href={href.incipit()} style={{ fontSize: 17, padding: "12px 4px" }}>
              try an example
            </a>
          </div>
          <label className="check">
            <input type="checkbox" checked={chat} onChange={(e) => setChat(e.target.checked)} />
            <span>
              send it as a chat <span style={{ fontStyle: "italic" }}>(chat template)</span>
            </span>
          </label>
          {chat ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 4, paddingLeft: 28, width: "100%", boxSizing: "border-box" }}>
              <label htmlFor="system" className="label-sc" style={{ fontSize: 14 }}>
                system prompt <span style={{ fontVariantCaps: "normal", letterSpacing: 0, fontStyle: "italic" }}>(optional)</span>
              </label>
              <textarea
                id="system"
                value={system}
                onChange={(e) => setSystem(e.target.value)}
                rows={2}
                placeholder="none: the chat begins with your question"
                style={{
                  width: "100%",
                  boxSizing: "border-box",
                  fontFamily: "inherit",
                  fontSize: 16,
                  lineHeight: 1.35,
                  background: "transparent",
                  border: 0,
                  borderBottom: "1px solid #C9BEAC",
                  padding: "2px 0 6px",
                  resize: "vertical",
                  borderRadius: 0,
                }}
              />
            </div>
          ) : (
            <span style={{ fontSize: 15, color: "#6B5D4D", paddingLeft: 28 }}>
              Off: the words go in as bare text. In the study two of six bare prompts came back as one word repeated.
            </span>
          )}
          <label className="check" style={{ color: "#6B5D4D" }}>
            <input type="checkbox" checked={compareThink} onChange={(e) => setCompareThink(e.target.checked)} />
            also sight it with thinking turned on
          </label>
          <SweepError />
          <RecordedNotice />
        </div>
      </section>
      {sighting ? <QuestionApparatus sighting={sighting} /> : null}
    </>
  );
};

const QuestionApparatus: React.FC<{ sighting: Sighting }> = ({ sighting }) => {
  const maxNewTokens = useTheodoliteStore((s) => s.maxNewTokens);
  const s = sighting;
  return (
    <Apparatus title="the question" style={{ padding: "12px 0 22px" }}>
      <dl className="dl">
        <dt>word-pieces</dt>
        <dd className="num">
          {s.nTokens != null ? `${s.nTokens} in all; ` : <Missing />}
          {s.readAt != null ? (
            <>
              read at the last, {wp(s.readAt)}
              {s.readPos != null ? ` (position ${s.readPos}, counting from 0).` : "."}
            </>
          ) : null}{" "}
          {s.firstIds ? (
            <>
              Start marks <span className="mono">[{s.firstIds.join(", ")}]</span>
              {s.firstIds[0] === 2 && s.firstIds[1] === 2 ? ": the doubled start mark of the instrument before the single-BOS fix." : "."}
            </>
          ) : s.promptTokens ? (
            <>
              First pieces <span className="mono-wrap">{JSON.stringify(s.promptTokens.slice(0, 4))}</span>.
            </>
          ) : null}
        </dd>
        <dt>text sent</dt>
        <dd>
          {s.textSent != null ? (
            <span className="mono-wrap" style={{ whiteSpace: "pre-wrap" }}>
              {JSON.stringify(s.textSent)}
            </span>
          ) : (
            "wrapped as a chat by the study’s lab script (the tokenizer’s chat template, thinking off), then the begun answer; the exact string was not recorded."
          )}
        </dd>
        <dt>continuation</dt>
        <dd className="num">
          greedy, {s.source === "fixture" ? 6 : maxNewTokens} word-pieces{" "}
          <span className="mono">max_new_tokens={s.source === "fixture" ? 6 : maxNewTokens}</span>
          {s.elapsedMs != null ? `; sweep ${comma(s.elapsedMs)} ms` : null}
          {s.peakVramGb != null ? `; peak memory ${s.peakVramGb.toFixed(1)} GB` : null}
        </dd>
        <dt>chat template</dt>
        <dd>
          {s.source === "fixture"
            ? "This sighting was wrapped as a chat by the study’s lab script (the tokenizer’s chat template, thinking off), then the begun answer."
            : s.chatWrapped
              ? (
                  <>
                    on: the tokenizer’s chat template, thinking off (an empty thought channel), the question as the user’s turn
                    {s.system ? (
                      <>
                        {" "}after a system turn, <span className="italic">“{s.system}”</span>
                      </>
                    ) : (
                      ", with no system turn"
                    )}
                    , then the begun answer after the model’s turn header, as the study’s lab script builds it (labkit.chat). One start mark.
                  </>
                )
              : "off: the words go in as bare text, and two of six plain prompts in the study came back as one word repeated (REPORT §3.4, bug 5)."}
        </dd>
        <dt>thinking</dt>
        <dd>
          {s.thinkingContinuation != null ? (
            <>
              with thinking turned on it went on <span className="italic">“{s.thinkingContinuation}”</span>
            </>
          ) : (
            <span className="muted">not sighted with thinking turned on</span>
          )}
        </dd>
        <dt>matched control</dt>
        <dd style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: "4px 14px" }}>
          <ControlSighting />
          <span style={{ color: "#6B5D4D" }}>
            The study reads lens outputs against a matched control (a nameless or background condition), never alone (REPORT §3.5). Take the
            control first, keep it, then take the sighting to compare; each layer’s apparatus shows both.
          </span>
        </dd>
      </dl>
    </Apparatus>
  );
};

const WentOn: React.FC<{ sighting: Sighting }> = ({ sighting }) => (
  <div
    style={{
      display: "flex",
      flexWrap: "wrap",
      alignItems: "baseline",
      gap: "8px 22px",
      padding: "16px 0 18px",
      borderTop: "1px solid #E3DACA",
      borderBottom: "1px solid #E3DACA",
    }}
  >
    <span className="label-sc">it went on</span>
    <span style={{ fontSize: 26, fontStyle: "italic" }}>“… {sighting.continuation.trim()}”</span>
    {sighting.isNudged ? (
      <a href={href.nudge()} className="sc" style={{ fontSize: 15, letterSpacing: "0.14em" }}>
        nudged
      </a>
    ) : null}
    <span className="num" style={{ marginLeft: "auto", fontSize: 15, color: "#6B5D4D" }}>
      60 layers surveyed{sighting.elapsedMs != null ? ` in ${ms(sighting.elapsedMs)}` : ""} · J++ lens
      {sighting.source === "fixture" ? " · the study’s recorded sighting" : ""}
    </span>
  </div>
);

/* ------------------------------------------------------- apparatus: what is drawn */

const WhatIsDrawn: React.FC<{ sighting: Sighting }> = ({ sighting }) => {
  const pf = sighting.layers.map((L) => L.projFrac).filter((v): v is number => v != null);
  const srcLayers = sighting.threads.map((t) => t.src.idx + 1);
  const [first, second, third] = sighting.landmarks;
  const poleRanking = useTheodoliteStore((s) => s.poleRanking);
  const setPoleRanking = useTheodoliteStore((s) => s.setPoleRanking);
  const threadScore = useTheodoliteStore((s) => s.threadScore);
  const setThreadScore = useTheodoliteStore((s) => s.setThreadScore);
  const fr = sighting.frame;
  const lowdin = fr?.squaring === "lowdin";
  const rows = fr?.fnwWeighted ? "fnw ⊙ W" : "W";
  const expl = sighting.layers.map((L) => L.ffnExplained).filter((v): v is number => v != null);
  const direct = sighting.threadScore === "direct";
  return (
    <Apparatus title="what is drawn" label="Apparatus for the instrument" sources={false}>
      <dl className="dl">
        <dt>survey frame</dt>
        <dd>
          e<sub>1</sub>, e<sub>2</sub>, e<sub>3</sub> = {lowdin ? "Löwdin" : "Gram–Schmidt"}({rows}
          <sub>U</sub>[{first}], {rows}
          <sub>U</sub>[{second}], {rows}
          <sub>U</sub>[{third}]) in fp32
          {fr?.perLayer ? <>, pulled back by each layer’s own lens J<sub>ℓ</sub></> : null}
          {fr?.centred ? ", read against a matched control’s state" : null}.{" "}
          {lowdin
            ? "Squared all at once, so the same frame for every order of the words."
            : "Order-dependent: a Löwdin frame on the final-norm-weighted rows is the proposed fix."}{" "}
          <a href={href.landmarks()}>landmarks</a>
        </dd>
        <dt>path</dt>
        <dd>
          h̃<sub>ℓ</sub> = h<sub>ℓ</sub> · J<sub>ℓ</sub>
          <sup>⊤</sup> projected on the frame; drawn by direction (θ, φ) only, its radius schematic.{" "}
          {pf.length ? (
            <>
              The frame catches {(Math.min(...pf) * 100).toFixed(1)}–{(Math.max(...pf) * 100).toFixed(1)}% of ‖h̃‖ in this sighting.
            </>
          ) : (
            <>
              The share of ‖h̃‖ the frame catches: <Missing what="not sent by the server" />
            </>
          )}
        </dd>
        <dt>gathering · remembering</dt>
        <dd>
          the ring round the chosen point: lapis Δ<sub>attn</sub> = s·(mid − pre)·J<sup>⊤</sup>, gilt Δ<sub>ffn</sub> = (post − s·mid)·J
          <sup>⊤</sup>, as shares of their sum (V7-corrected, with the layer scalar s).
        </dd>
        <dt>neurons</dt>
        <dd>
          the 15 strongest GeGLU poles at the chosen layer,{" "}
          {sighting.poleRanking === "write" ? (
            <>
              by write size |a| · ‖w<sub>down</sub> ⊙ w<sub>post</sub>‖
            </>
          ) : (
            "by |a| = |mlp_act|"
          )}
          ; filled the positive pole, hollow the negative (after Cammarata 2026); size by |a|, placement on the ring schematic.
          {expl.length ? (
            <>
              {" "}Together they explain {(Math.min(...expl) * 100).toFixed(0)}–{(Math.max(...expl) * 100).toFixed(0)}% of a layer’s feed-forward
              write in this sighting (⟨Σ c<sub>i</sub>, f⟩ / ⟨f, f⟩, E3).
            </>
          ) : null}
          <label className="check" style={{ marginTop: 4 }}>
            <input
              type="checkbox"
              checked={poleRanking === "write"}
              onChange={(e) => {
                setPoleRanking(e.target.checked ? "write" : "activation");
                retakeIfLive();
              }}
            />
            <span>
              rank them by write size, |a| · ‖w<sub>down</sub> ⊙ w<sub>post</sub>‖, instead
            </span>
          </label>
        </dd>
        <dt>threads</dt>
        <dd>
          the {sighting.threads.length} highest-scoring{sighting.threadCount != null ? ` of ${comma(sighting.threadCount)}` : ""}:{" "}
          {direct ? (
            <>
              D = what the source adds to the reader’s gate through the stream alone, from its real activation, the layer scalars and both
              exact RMSNorm Jacobians; gilt where D &gt; 0, dashed where D &lt; 0
            </>
          ) : (
            <>
              C = ⟨σ<sub>i</sub> w<sub>down</sub>, w<sub>gate</sub>⟩ · |a|, gilt where C &gt; 0, dashed where C &lt; 0
            </>
          )}
          {srcLayers.length ? (
            <>
              ; they start in layers {Math.min(...srcLayers)}–{Math.max(...srcLayers)}
              {!direct && Math.max(...srcLayers) <= 2 ? ", where the study found the real effects tiny" : ""}
            </>
          ) : null}
          .{" "}
          {direct
            ? "A direct-path linear estimate, not a verified circuit: ρ ≈ 0.87 with the measured effect, where C reaches 0.41 (REPORT §3.4, bug 4)."
            : "A weak guide (ρ ≈ 0.41; a direct-path estimate reaches 0.87)."}
          <label className="check" style={{ marginTop: 4 }}>
            <input
              type="checkbox"
              checked={threadScore === "direct"}
              onChange={(e) => {
                setThreadScore(e.target.checked ? "direct" : "coupling");
                retakeIfLive();
              }}
            />
            score threads by the direct-path estimate
          </label>
        </dd>
      </dl>
    </Apparatus>
  );
};

/* ---------------------------------------------------------------- iii · the leaf */

function poleNote(s: Sighting, L: LayerReading) {
  const p = L.poles[0];
  if (!p) return null;
  const recorded = s.source === "fixture" ? RECORDED_POLE_NOTES[`${L.idx}:${p.neuron}`] : undefined;
  const lens = p.lens?.slice(0, 3) ?? null;
  const eye = p.eye ?? null;
  let auto = "";
  if (lens && eye) {
    const shared = lens.map((t) => t.token.trim()).find((t) => eye.slice(0, 5).some((e) => e.token.trim() === t));
    if (shared) auto = `The lens and the naked eye agree on “${shared}”: a good sign.`;
    else auto = "The lens and the naked eye share no word-piece here.";
  }
  return {
    p,
    words: lens,
    gloss: recorded?.gloss ?? auto,
    trust: recorded?.trust ?? labelTrust(L.idx),
  };
}

const Leaf: React.FC<{ sighting: Sighting; sel: number; stations: number[]; onPick: (idx: number) => void }> = ({
  sighting,
  sel,
  stations,
  onPick,
}) => {
  const L = sighting.layers[sel];
  const recordedSet = sighting.layers.filter((x) => x.lensTop).map((x) => x.idx);
  // step between the layers whose reading was recorded (every layer in a live sighting)
  const steps = recordedSet.length ? recordedSet : stations;
  const prev = [...steps].reverse().find((i) => i < sel);
  const next = steps.find((i) => i > sel);
  const k = kindOf(L.isGlobal);
  const top = L.lensTop?.[0]?.prob ?? null;
  const gather = L.ffnShare != null ? Math.round((1 - L.ffnShare) * 100) : null;
  const note = poleNote(sighting, L);
  return (
    <aside aria-labelledby="leaf-head" style={{ flex: "1 1 400px", minWidth: 0, display: "flex", flexDirection: "column", gap: 22 }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12 }}>
        <h2 id="leaf-head" className="h-sec">
          iii · the leaf
        </h2>
        <div style={{ display: "flex", gap: 4 }}>
          <button type="button" className="btn-ghost plain" onClick={() => prev !== undefined && onPick(prev)} disabled={prev === undefined} aria-label="Earlier layer">
            ↓ earlier
          </button>
          <button type="button" className="btn-ghost plain" onClick={() => next !== undefined && onPick(next)} disabled={next === undefined} aria-label="Later layer">
            later ↑
          </button>
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "flex-end", gap: 20, borderBottom: "1px solid #D9CFBD", paddingBottom: 16 }}>
        <div className="lining" style={{ fontSize: 112, lineHeight: 0.8, fontWeight: 500, color: "#8A6A2C", letterSpacing: "-0.02em" }}>
          {L.no}
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 2, paddingBottom: 4 }}>
          <span style={{ fontVariantCaps: "all-small-caps", letterSpacing: "0.2em", fontSize: 15, color: "#6B5D4D" }}>layer {L.no} of sixty</span>
          <span style={{ fontSize: 17 }}>{k.kind}</span>
          <span style={{ fontSize: 15, fontStyle: "italic", color: "#6B5D4D" }}>{k.gloss}</span>
        </div>
      </div>

      <div className="stack">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12 }}>
          <h3 className="h-sub">what it was thinking</h3>
          <span style={{ fontSize: 15, fontStyle: "italic", color: "#6B5D4D" }}>{certainty(top)}</span>
        </div>
        {L.lensTop ? (
          L.lensTop.slice(0, 5).map((t, i) => (
            <div
              key={i}
              style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) 64px", gap: "2px 14px", alignItems: "baseline", padding: "6px 0 8px", borderBottom: "1px solid #ECE4D5" }}
            >
              <div style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: "2px 12px", minWidth: 0 }}>
                <span style={{ fontSize: 25, lineHeight: 1.2 }} title={JSON.stringify(t.token)}>
                  {wp(t.token)}
                </span>
                <span style={{ fontSize: 15, fontStyle: "italic", color: "#6B5D4D" }}>{i === 0 && sel === 59 ? "the word it says next" : gloss(t.token)}</span>
              </div>
              <span className="num" style={{ textAlign: "right", fontSize: 16 }}>
                {t.prob != null ? pct(t.prob) : <Missing what="—" />}
              </span>
              <div className="bar-track" style={{ gridColumn: "1 / -1" }}>
                <div className="bar-fill" style={{ width: `${Math.max(0.6, (t.prob ?? 0) * 100)}%` }} />
              </div>
            </div>
          ))
        ) : (
          <p style={{ margin: 0, fontSize: 16 }}>
            <Missing what="not recorded at this layer in the study’s sighting" />{" "}
            <span className="muted">The recording kept the lens’s words at the marked layers only; choose one of them, or take a live sighting.</span>
          </p>
        )}
        <Rubric style={{ margin: "4px 0 0", fontSize: 16 }}>
          These are the words the model would say if it stopped at this layer, read through the J++ lens.{" "}
          <a href={href.lexicon("lens")}>What is a lens?</a>
        </Rubric>
      </div>

      <div className="stack">
        <h3 className="h-sub">gathering or remembering</h3>
        {gather != null ? (
          <>
            <div
              role="img"
              aria-label={`${gather} percent gathered from earlier words, ${100 - gather} percent recalled from memory`}
              style={{ display: "flex", height: 14, borderRadius: 1, overflow: "hidden" }}
            >
              <div style={{ width: `${gather}%`, background: "#2E4A7D" }} />
              <div style={{ width: `${100 - gather}%`, background: "#A07A2E" }} />
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 16, fontSize: 16 }}>
              <span>
                <span className="num" style={{ color: "#2E4A7D", fontWeight: 600 }}>
                  {gather}%
                </span>{" "}
                gathered from the words before <span style={{ color: "#6B5D4D", fontStyle: "italic" }}>(attention)</span>
              </span>
              <span style={{ textAlign: "right" }}>
                <span className="num" style={{ color: "#7A5C28", fontWeight: 600 }}>
                  {100 - gather}%
                </span>{" "}
                recalled from memory <span style={{ color: "#6B5D4D", fontStyle: "italic" }}>(feed-forward)</span>
              </span>
            </div>
          </>
        ) : (
          <Missing />
        )}
      </div>

      <div className="stack" style={{ gap: 8 }}>
        <h3 className="h-sub">its loudest neuron</h3>
        {note && note.words ? (
          <>
            <div style={{ display: "flex", alignItems: "baseline", flexWrap: "wrap", gap: "4px 14px" }}>
              <a href={href.neuron(L.no, note.p.neuron, note.p.face)} className="num" style={{ fontSize: 17 }}>
                neuron {note.p.neuron}
              </a>
              <span style={{ fontSize: 15, color: "#6B5D4D" }}>
                {note.p.face > 0 ? "positive" : "negative"} face · strength {Math.abs(note.p.act).toFixed(1)}
              </span>
            </div>
            <div style={{ fontSize: 22, lineHeight: 1.35 }}>{note.words.map((t) => wp(t.token)).join("  ")}</div>
            {note.gloss ? <div style={{ fontSize: 15, fontStyle: "italic", color: "#6B5D4D" }}>{note.gloss}</div> : null}
            <Rubric style={{ margin: "2px 0 0", fontSize: 15 }}>{note.trust}</Rubric>
          </>
        ) : note ? (
          <p style={{ margin: 0, fontSize: 16, color: "#6B5D4D" }}>
            Fifteen neurons fired here; the loudest is{" "}
            <a href={href.neuron(L.no, note.p.neuron, note.p.face)}>neuron {note.p.neuron}</a>. Open the layer’s leaf to read them.
          </p>
        ) : (
          <Missing />
        )}
      </div>

      <div
        style={{ display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: "10px 20px", paddingTop: 14, borderTop: "1px solid #D9CFBD" }}
      >
        <span className="num" style={{ fontSize: 15, color: "#6B5D4D" }}>
          {L.az != null && L.el != null ? `bearing ${minus(String(Math.round(L.az)))}° · elevation ${minus(String(Math.round(L.el)))}°` : <Missing />}
        </span>
        <a href={href.layer(L.no)} style={{ fontSize: 17, padding: "10px 0" }}>
          open this layer’s leaf →
        </a>
      </div>

      <LayerApparatus sighting={sighting} L={L} />
    </aside>
  );
};

const LayerApparatus: React.FC<{ sighting: Sighting; L: LayerReading }> = ({ sighting, L }) => {
  const [raw, setRaw] = useState(false);
  const control = useTheodoliteStore((s) => s.control);
  useEffect(() => setRaw(false), [L.idx]);
  const words = [...sighting.landmarks.filter((w) => L.ranks[w]), ...Object.keys(L.ranks).filter((w) => !sighting.landmarks.includes(w))].slice(0, 8);
  const notRecorded = [
    L.streamStrength == null ? "‖h‖" : null,
    L.rescaleNorm == null ? "the rescale ‖(s − 1)·pre‖" : null,
    L.eyeTop == null ? "the naked eye’s top word-pieces" : null,
    L.projFrac == null ? "the share of ‖h̃‖ the frame catches" : null,
  ].filter(Boolean) as string[];
  const rawObj = {
    layer: L.no,
    index: L.idx,
    read_at: sighting.readAt,
    jpp_top5: L.lensTop?.map((t) => [t.token, t.prob]) ?? null,
    naive_top: L.eyeTop,
    ffn_dominance: { corrected: L.ffnShare, tool_before_fix: L.ffnShareBeforeFix },
    attn_norm: L.attnNorm,
    ffn_norm: L.ffnNorm,
    layer_scalar: L.layerScalar,
    rescale_norm: L.rescaleNorm,
    residual_norm: L.streamStrength,
    azimuth_deg: L.az,
    elevation_deg: L.el,
    xyz: L.xyz,
    proj_frac_of_hJ: L.projFrac,
    tracked_rank: L.ranks,
    poles: L.poles.map((p) => ({ neuron: p.neuron, act: p.act, lens: p.lens?.map((t) => t.token) ?? null, eye: p.eye?.map((t) => t.token) ?? null })),
  };
  const ctl = control && control !== sighting ? control.layers[L.idx] : null;
  return (
    <Apparatus
      title={
        <>
          layer {L.no} <span style={{ letterSpacing: "0.06em" }}>(index {L.idx})</span>
        </>
      }
      label="Apparatus for this layer"
    >
      <dl className="dl tight num">
        <dt>readout</dt>
        <dd>
          logits(h, ℓ) = unembed(h · J<sub>ℓ</sub>
          <sup>⊤</sup>), unembed(u) = 30 · tanh(W<sub>U</sub> · RMSNorm(u) / 30); J = the J++ lens (Ayonrinde &amp; Lindsey 2026), fitted on the
          base model. The Jacobian lens (Gurnee et al. 2026) is on disk; the server does not load it.
        </dd>
        <dt>split</dt>
        <dd>
          {L.ffnShare != null ? (
            <>
              remembering {Math.round(L.ffnShare * 100)}% after the V7 fix
              {L.layerScalar != null ? <>, with this layer’s scalar s = {L.layerScalar.toFixed(3)}</> : null}
              {L.ffnShareBeforeFix != null ? <>; the instrument read {Math.round(L.ffnShareBeforeFix * 100)}% before it</> : null}.
              {L.attnNorm != null && L.ffnNorm != null ? (
                <>
                  {" "}
                  ‖Δ<sub>attn</sub>‖ = {L.attnNorm.toFixed(2)}, ‖Δ<sub>ffn</sub>‖ = {L.ffnNorm.toFixed(2)}.
                </>
              ) : null}
            </>
          ) : (
            <Missing />
          )}
        </dd>
        <dt>frame</dt>
        <dd>
          {L.xyz ? <>(x, y, z) = ({L.xyz.map(sci).join(", ")}); </> : null}
          {L.az != null && L.el != null ? (
            <>
              θ = atan2(y, x) = {minus(L.az.toFixed(1))}°, φ = asin(z / ‖p‖) = {minus(L.el.toFixed(1))}°
            </>
          ) : (
            <Missing />
          )}
          {L.projFrac != null ? <>; the frame catches {(L.projFrac * 100).toFixed(1)}% of ‖h̃‖.</> : "."}
        </dd>
        <dt>stream strength</dt>
        <dd>{L.streamStrength != null ? <>‖h‖ = {L.streamStrength.toFixed(1)}</> : <Missing />}</dd>
        <dt>ranks</dt>
        <dd>
          of 262,144, lens / naked eye:{" "}
          {words.length
            ? words.map((w, i) => (
                <span key={w}>
                  {i ? " · " : ""}“{w}” {L.ranks[w].lens != null ? comma(L.ranks[w].lens!) : "—"} /{" "}
                  {L.ranks[w].eye != null ? comma(L.ranks[w].eye!) : <Missing what="not sent" />}
                </span>
              ))
            : <Missing />}
        </dd>
        <dt>poles</dt>
        <dd>
          signed a of the {L.poles.length} strongest: {L.poles.map((p) => signed(p.act)).join("  ")}. Labels: unembed((σ · w<sub>down</sub>) · J
          <sup>⊤</sup>), against the naked eye’s unembed(σ · w<sub>down</sub>). Output labels held up at layers 41–58 only.
        </dd>
        {ctl ? (
          <>
            <dt>control</dt>
            <dd>
              at this layer the control was thinking {ctl.lensTop?.[0] ? <>{wp(ctl.lensTop[0].token)} ({pct(ctl.lensTop[0].prob)})</> : <Missing what="not recorded" />}
              {ctl.ffnShare != null ? <>; remembering {Math.round(ctl.ffnShare * 100)}%</> : null}.
            </dd>
          </>
        ) : null}
        {notRecorded.length ? (
          <>
            <dt>not recorded</dt>
            <dd style={{ color: "#6B5D4D" }}>
              for this sighting, {notRecorded.join(", ")}: <Missing />
            </dd>
          </>
        ) : null}
      </dl>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "8px 12px", paddingTop: 4 }}>
        <button type="button" className="btn-ghost" onClick={() => setRaw((r) => !r)} aria-expanded={raw}>
          {raw ? "hide the raw readout" : "show the raw readout"}
        </button>
        <button type="button" className="btn-ghost" onClick={() => exportSighting(sighting)}>
          export the sighting (.json)
        </button>
      </div>
      {raw ? <pre className="raw">{JSON.stringify(rawObj, null, 2)}</pre> : null}
      <p className="muted" style={{ margin: 0 }}>
        {sighting.source === "fixture" ? (
          <>
            Recorded in <span className="mono-wrap">experiments/results/E6_tool_audit_02_sweep_demo.json</span> (key prepend_bos=True); layer
            scalars from E6_tool_audit_01_basics.json.
          </>
        ) : (
          "From the running server, as it sent this sighting."
        )}
      </p>
    </Apparatus>
  );
};

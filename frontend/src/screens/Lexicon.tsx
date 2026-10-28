import React from "react";
import { Apparatus, Missing, Page, ReadingThrough, Rubric } from "../components/kit";
import { href, type Route } from "../router";
import {
  FNW_DEFAULT,
  MAX_WRITE,
  MODEL_SAMPLING,
  SIGHT_CAP,
  STUDY_WRITING,
  chosenPhrase,
  ordinal,
  readWhere,
  sweepPayload,
  useTheodoliteStore,
} from "../store/useTheodoliteStore";
import { NotAMatchedPair, strings } from "../lib/cautions";
import { pieceWord, wp } from "../lib/words";
import { fixtureCheck, labelGate, layerIdx, sharedLine, sharedShown } from "../lib/labels";
import { HeldBackWords } from "../components/HeldBack";
import { JPP as FOLIO_JPP, NAIVE as FOLIO_NAIVE, RECORDED as FOLIO_EXAMPLE } from "./NeuronFolio/recorded";
import { STUDY_K10, STUDY_K10_LATE_P10 } from "./NeuronFolio/causal";
import { ALIASES, INDEX } from "./lexicon/entries";
import { REFERENCE_COUNT } from "./lineage/references";
import { Cautions } from "./lexicon/Cautions";
import {
  Astrolabe,
  FigApparatus,
  FigBearing,
  FigDoubleNegative,
  FigEdit,
  FigGathering,
  FigGlobal,
  FigJSpace,
  FigKinship,
  FigLandmarks,
  FigLayer,
  FigLens,
  FigMatchedControl,
  FigNeuron,
  FigNudge,
  FigRemembering,
  FigSighting,
  FigStream,
  FigThreads,
  FigVault,
} from "./lexicon/figures";
import {
  Body,
  Entry,
  EntryApparatus,
  EntryGrid,
  Lede,
  LexLink,
  LexSection,
  M,
  Note,
  Onward,
  Query,
  Sym,
} from "./lexicon/parts";

/*
 * Lexicon — plain words for strange things (folio vii).
 * Built from the approved canvas Lexicon.dc.html. The page is the instrument's glossary: almost all of it
 * is static, sourced copy (README.md, ARCHITECTURE.md, REFERENCES.md, experiments/REPORT.md, the V6/V7/V8
 * notes, E6_tool_audit_REPORT.md, N1_REPORT.md, R1_IT_LENS_REPORT.md and the backend code as audited).
 * Every entry has a stable id (see ./lexicon/entries.ts) so other pages can link href.lexicon('<id>').
 * Only one line reads the store: the sighting entry says which sighting is on the instrument page.
 */

/* ------------------------------------------------------------ recorded examples */

/**
 * "through the lens" (once "what it was thinking"): the lens's top word-pieces after "Hi! I" in the study's
 * demonstration sighting (E6_tool_audit_02_sweep_demo.json, prepend_bos=True; lensTop at index 50 = layer 51 and
 * 59 = layer 60).
 * Shares as the canvas prints them; `bar` is the probability in per cent.
 */
const AT_51: { word: string; gloss?: string; share: string; bar: number }[] = [
  { word: "“bot”", share: "26%", bar: 25.7 },
  { word: "“Bot”", gloss: "joined to the word before", share: "18%", bar: 18.2 },
  { word: "“AI”", share: "9.1%", bar: 9.1 },
];
const AT_60: { word: string; gloss?: string; share: string; bar: number }[] = [
  { word: "“am”", gloss: "the word it said next in the recorded, greedy sighting", share: "99.8%", bar: 99.8 },
];

/**
 * "neuron and its two faces": the recorded neuron the folio opens on, 15491 at layer 54 (index 53, rank 4 of the
 * demonstration sighting), through the label gate (lib/labels.ts) by its recorded check (N1G_01's, from the weights).
 * Inside layers 41-58 the gate shows it; were the check to share none, the entry would hold it back like any page.
 */
const EXAMPLE_GATE = labelGate(layerIdx(FOLIO_EXAMPLE.idx), fixtureCheck(FOLIO_EXAMPLE.idx, FOLIO_EXAMPLE.rank));
const EXAMPLE_SHOWN = sharedShown(
  FOLIO_JPP.map(([t]) => t),
  FOLIO_NAIVE.map(([t]) => t),
);

const ReadoutRow: React.FC<{ word: string; gloss?: string; share: string; bar: number }> = ({ word, gloss, share, bar }) => (
  <div
    style={{
      display: "grid",
      gridTemplateColumns: "minmax(0, 1fr) 56px",
      gap: "2px 14px",
      alignItems: "baseline",
      padding: "4px 0 6px",
      borderBottom: "1px solid var(--hair-faint)",
    }}
  >
    {gloss ? (
      <span style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: "2px 10px", minWidth: 0 }}>
        <span style={{ fontSize: 21, lineHeight: 1.2 }}>{word}</span>
        <span style={{ fontSize: 14, fontStyle: "italic", color: "var(--muted)" }}>{gloss}</span>
      </span>
    ) : (
      <span style={{ fontSize: 21, lineHeight: 1.2 }}>{word}</span>
    )}
    <span className="num" style={{ textAlign: "right", fontSize: 15 }}>
      {share}
    </span>
    <div className="bar-track" style={{ gridColumn: "1 / -1" }}>
      <div className="bar-fill" style={{ width: `${bar}%` }} />
    </div>
  </div>
);

const READOUT_LABEL: React.CSSProperties = {
  fontSize: 14,
  fontVariantCaps: "all-small-caps",
  letterSpacing: "0.16em",
  color: "var(--muted)",
};

/* ------------------------------------------------------------ the one live line */

/**
 * "The sighting on the instrument page follows “I”, in its reply “Hi! I”; its sweep took 6.2 seconds." With the
 * recorded sighting loaded this reads as the canvas did; after a live sighting it names that one: which word-piece of
 * its answer, where it was read and what it chose there, and how long it took from the request to that word-piece.
 */
const SightingOnInstrument: React.FC = () => {
  const s = useTheodoliteStore((st) => st.sighting);
  if (!s) return <Note>No sighting is on the instrument page yet.</Note>;
  const w = readWhere(s);
  const n = s.answerIds?.length ?? null;
  const chose = chosenPhrase(s);
  const total = s.source === "live" ? (s.timing?.totalMs ?? null) : null;
  const secs = (v: number) => `${(v / 1000).toFixed(1)} seconds`;
  return (
    <Note>
      The sighting on the instrument page{n ? <> is word-piece {s.tokenIndex + 1} of the {n} its answer holds; it</> : null} follows{" "}
      {s.readAt != null ? pieceWord(s.readAt) : <Missing />}
      {w.role === "written" ? `, the ${ordinal(w.k)} word-piece of its own answer` : w.phrase}
      {chose ? `, ${chose}` : ""};{" "}
      {total != null ? (
        <>it came back {secs(total)} after it was asked for, the answer written and read again first.</>
      ) : s.elapsedMs != null && Number.isFinite(s.elapsedMs) ? (
        <>its sweep took {secs(s.elapsedMs)}.</>
      ) : (
        <>
          its time: <Missing what="not recorded" />.
        </>
      )}
    </Note>
  );
};

/** What the instrument sends by default, from the store's own sweepPayload and its first state (the study's writing),
 * never typed by hand; each request also carries a request id of its own. */
const DEFAULT_REQUEST = (() => {
  const st = useTheodoliteStore.getInitialState();
  return JSON.stringify({
    action: "sweep",
    payload: sweepPayload({ ...st, writing: STUDY_WRITING }, st.question, st.answerBegins, st.steering),
  });
})();

/* ------------------------------------------------------------ the page */

const STATS: [string, string][] = [
  ["60", "layers, one above another"],
  ["5,376", "numbers in the stream"],
  ["21,504", "neurons in every layer"],
  ["262,144", "word-pieces it knows"],
];

const INDEX_LINK: React.CSSProperties = {
  display: "flex",
  alignItems: "baseline",
  gap: 8,
  minHeight: 44,
  paddingTop: 8,
  boxSizing: "border-box",
  color: "var(--ink)",
  textDecoration: "none",
};

const FOOTER = (
  <>
    <ReadingThrough /> · 60 layers · 5,376 dimensions ·{" "}
    <a href={href.lineage()}>the lineage of these ideas</a>
  </>
);

export const Lexicon: React.FC<{ route: Route }> = ({ route }) => (
  <Page route={route} folio="folio vii" footer={FOOTER}>
    {/* ---------------------------------------------------------------- the title leaf */}
    <section
      aria-labelledby="lex-title"
      style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "36px 72px", padding: "52px 0 48px" }}
    >
      <div style={{ flex: "999 1 560px", minWidth: 0, display: "flex", flexDirection: "column", gap: 20 }}>
        <p className="h-sec">the lexicon</p>
        <h1
          id="lex-title"
          style={{
            margin: 0,
            fontSize: "clamp(42px, 5.2vw, 68px)",
            lineHeight: 1.04,
            fontWeight: 400,
            letterSpacing: "-0.005em",
            color: "var(--ink)",
            maxWidth: 720,
          }}
        >
          Plain words for strange things
        </h1>
        <p style={{ margin: 0, fontSize: 23, lineHeight: 1.5, maxWidth: 660 }}>
          Inside Gemma 4 lies a country with no names on its map. This instrument surveys it, and gives its strange
          features plain names. Each entry sets the name you will meet on the instrument beside the specialist’s, then
          says in a sentence or two what the thing is.
        </p>
        <Rubric style={{ maxWidth: 640 }}>
          Keep this leaf open beside the instrument. Lines in red, like this one, tell you what to do; switch them off
          with the rubrics button above once you know the way. The apparatus button beside it adds each entry’s formula
          and sources.
        </Rubric>
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: "18px 44px",
            marginTop: 10,
            paddingTop: 20,
            borderTop: "1px solid var(--hair)",
          }}
        >
          {STATS.map(([n, what]) => (
            <div key={what} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span className="num" style={{ fontSize: 42, lineHeight: 1, color: "var(--gilt-numeral)" }}>
                {n}
              </span>
              <span style={{ fontSize: 15, color: "var(--muted)" }}>{what}</span>
            </div>
          ))}
        </div>
      </div>
      <figure
        style={{
          flex: "1 1 380px",
          minWidth: 0,
          margin: 0,
          display: "flex",
          flexDirection: "column",
          gap: 12,
          alignItems: "center",
        }}
      >
        <Astrolabe />
        <figcaption
          style={{ fontSize: 15, fontStyle: "italic", color: "var(--muted)", textAlign: "center", maxWidth: 420 }}
        >
          The rete of an astrolabe carries pointers for the bright stars. Here its ring carries the words of this
          lexicon.
        </figcaption>
      </figure>
    </section>

    <Apparatus
      title="notation used beneath every entry"
      label="Apparatus for the notation of this lexicon"
      style={{ padding: "12px 0 28px" }}
    >
      <dl className="dl num">
        <dt>counting</dt>
        <dd>
          layer ℓ = index + 1. Code, report and the vault’s query language count from 0, so the report’s L40–57 is
          layers 41–58 here.
        </dd>
        <dt>the stream</dt>
        <dd>
          h<sub>ℓ</sub> ∈ ℝ<sup>5376</sup>, the residual stream leaving layer ℓ at the read position; h<sub>pre</sub>, h
          <sub>mid</sub>, h<sub>post</sub> the hook points <M>resid_pre</M>, <M>resid_mid</M>, <M>resid_post</M>.
        </dd>
        <dt>the lens</dt>
        <dd>
          J<sub>ℓ</sub> ∈ ℝ<sup>5376 × 5376</sup>, the J++ matrix (fitted on the base model) unless another is named, fp16
          on disk and cast to bf16 on load, for layers 1–59; J<sub>60</sub> = I.
        </dd>
        <dt>the readout</dt>
        <dd>
          unembed(u) = 30 · tanh(W<sub>U</sub> · RMSNorm(u) / 30); W<sub>U</sub> the 262,144 unembedding rows (tied to
          the embedding); fnw the final norm’s weight.
        </dd>
        <dt>the neuron</dt>
        <dd>
          a its activation, σ = sign(a) its pole; w<sub>gate</sub>, w<sub>up</sub> its read rows, w<sub>down</sub> its
          write column; s the layer scalar of Gemma 4.
        </dd>
        <dt>references</dt>
        <dd>
          [n] as numbered in <M>REFERENCES.md</M> ({REFERENCE_COUNT} entries); E1–E7, V1–V8, N1 and R1 are the study’s experiments,
          verifiers, neuron-label study and lens refit, written up in <M>experiments/REPORT.md</M>.
        </dd>
      </dl>
    </Apparatus>

    {/* ---------------------------------------------------------------- i · the index */}
    <section
      aria-labelledby="index-head"
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 14,
        padding: "28px 0 8px",
        borderTop: "1px solid var(--hair)",
      }}
    >
      <div className="row-between">
        <h2 className="h-sec" id="index-head">
          i · the index
        </h2>
        <span style={{ fontSize: 15, color: "var(--muted)" }}>
          thirty-seven words and four cross-references, in alphabetical order
        </span>
      </div>
      <Rubric style={{ maxWidth: 760 }}>
        Choose a word to go to it. Each entry gives the name you will meet on the instrument first, then the
        specialist’s name in small capitals.
      </Rubric>
      <ul
        style={{ listStyle: "none", margin: "4px 0 0", padding: 0, columns: "4 250px", columnGap: 48, fontSize: 18 }}
      >
        {INDEX.map((item) => (
          <li key={item.label} style={{ breakInside: "avoid" }}>
            <LexLink to={item.id} style={INDEX_LINK}>
              <span>
                {item.see ? (
                  <>
                    {item.label} <span style={{ fontStyle: "italic", color: "var(--muted)" }}>see</span> {item.see}
                  </>
                ) : (
                  item.label
                )}
              </span>
              <span
                aria-hidden="true"
                style={{ flex: "1 1 auto", minWidth: 12, borderBottom: "1px dotted var(--control)" }}
              />
              <span
                style={{
                  fontSize: 15,
                  color: "var(--gilt-text)",
                  fontVariantCaps: "all-small-caps",
                  letterSpacing: "0.12em",
                }}
              >
                {item.section}
              </span>
            </LexLink>
          </li>
        ))}
      </ul>
    </section>

    {/* ---------------------------------------------------------------- ii · inside the model */}
    <LexSection id="model-head" title="ii · inside the model" pad={60}>
      <Lede cap="S">
        ixty layers stand between the word that arrives and the word the model says next. At every one, the model
        gathers from the words before, remembers what it learned in training, and adds both to a single running
        stream. These are the names for what lives in there.
      </Lede>
      <Rubric style={{ maxWidth: 760 }}>New here? Read “layer” and “word-piece” first; the rest follow from them.</Rubric>
      <EntryGrid>
        <Entry id="layer" head="layer" tech="decoder layer · layers 1 to 60" figure={<FigLayer />}>
          <Body>
            One of the sixty stages each word-piece passes through on its way to becoming the next word. Every layer
            reads the stream, gathers, remembers, and writes its change back. The instrument draws them as stacked
            rings: layer 1 at the bottom, layer 60 at the top.
          </Body>
          <Rubric>
            This instrument counts layers from 1. The study’s report counts from 0, so its “L35” is layer 36 here, and
            its “L40–57” is layers 41–58.
          </Rubric>
          <EntryApparatus title="layer">
            h<sub>mid</sub> = h<sub>pre</sub> + Attn(h<sub>pre</sub>), h<sub>post</sub> = s · (h<sub>mid</sub> + FFN(h
            <sub>mid</sub>)), the sandwich norms folded into Attn and FFN; the layer scalar s is 0.03 at layer 60,
            0.07–0.10 at layers 1–2 and 0.45–0.99 elsewhere (REPORT §0). Gemma 4 [1]; the stream as a sum of writes
            after Elhage et al. 2021 [14].
          </EntryApparatus>
        </Entry>

        <Entry
          id="word-piece"
          head="word-piece"
          tech="token · 262,144 in the vocabulary"
          figure={
            <div
              style={{
                flex: "0 0 auto",
                minWidth: 120,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 2,
                paddingTop: 4,
              }}
            >
              <span lang="zh" style={{ fontSize: 36, lineHeight: 1.15 }}>
                “是一个”
              </span>
              <span style={{ fontSize: 15, fontStyle: "italic", color: "var(--muted)" }}>Chinese: “is a”</span>
              <span
                style={{ fontSize: 13, fontVariantCaps: "all-small-caps", letterSpacing: "0.16em", color: "var(--muted)" }}
              >
                one word-piece
              </span>
            </div>
          }
        >
          <Body>
            The unit the model reads and writes. A word-piece may be a whole word, part of one, or several words fused
            together; the same word with and without a space before it counts as two different pieces.
          </Body>
          <Note>
            On the instrument, word-pieces stand in curly quotes. Pieces in other languages carry an italic
            translation, given only where we are sure of it.
          </Note>
          <EntryApparatus title="word-piece">
            a token id in 0 … 262,143 of the Gemma 4 vocabulary [1]. A chat prompt now opens <M>[2, 105, …]</M>, one
            start mark; before the V7 fix it opened <M>[2, 2, 105, …]</M> (REPORT §3.4, bug 6).
          </EntryApparatus>
        </Entry>

        <Entry
          id="stream"
          aliases={ALIASES.stream}
          head="stream strength"
          tech={
            <>
              residual stream · residual norm <Sym>‖h‖</Sym>
            </>
          }
          figure={<FigStream />}
        >
          <Body>
            The stream is the model’s running draft: a list of 5,376 numbers for each word-piece, which every layer
            reads and adds to. Its strength is the size of that draft at one layer, a single number.
          </Body>
          <Note>
            Gemma 4 also scales the whole stream down at every layer, so it forgets: something written at layer 30
            fades below a tenth within about five layers. That shrinking is about as large as what a layer remembers,
            so since the study’s fix (V7) the instrument measures it on its own rather than counting it as remembering.
          </Note>
          <EntryApparatus title="stream strength">
            ‖h<sub>ℓ</sub>‖<sub>2</sub> = (Σ<sub>d</sub> h<sub>ℓ,d</sub>
            <sup>2</sup>)<sup>½</sup> over the 5,376 numbers. The figure’s values are medians over the study’s
            positions, start marks excluded: 237, 269, 175, 142, 88 at layers 30, 36, 42, 48, 54 (V7, bug 1). The
            rescale (s − 1) · h<sub>pre</sub> is reported apart as <M>rescale_norm</M>. Residual stream after [14];
            RMSNorm [16].
          </EntryApparatus>
        </Entry>

        <Entry
          id="gathering"
          head="gathering"
          tech={
            <>
              attention · <Sym>Δ</Sym> attn
            </>
          }
          figure={<FigGathering />}
        >
          <Body>
            The part of each layer that looks back at the earlier word-pieces and draws in what it needs from them: the
            subject of the sentence, the question that was asked. On the instrument, gathering is always drawn in lapis
            blue.
          </Body>
          <Note>
            Find it on <a href={href.instrument()}>the instrument</a>, in the leaf’s “gathering or remembering”.
          </Note>
          <EntryApparatus title="gathering">
            Δ<sub>attn</sub> = s · (h<sub>mid</sub> − h<sub>pre</sub>) · J<sub>ℓ</sub>
            <sup>⊤</sup>, the attention write as it lands in the stream (V7 fix; REPORT §3.4, bug 2). Before the fix
            the tool reported it a factor of 1/s too large. Attention as an additive write after [14].
          </EntryApparatus>
        </Entry>

        <Entry
          id="remembering"
          head="remembering"
          tech={
            <>
              feed-forward network · FFN · <Sym>Δ</Sym> ffn
            </>
          }
          figure={<FigRemembering />}
        >
          <Body>
            The other part of each layer. It consults what the model learned in training, held in the layer’s 21,504
            neurons, and writes what it recalls into the stream. On the instrument, remembering is always drawn in
            gilt.
          </Body>
          <Note>
            Since the study’s fix (V7) the instrument separates the two writes correctly; before it, the larger share
            was misjudged at 12 of the 60 layers. Each share is weighed after both writes pass through the J++ lens,
            fitted on the base model; how a lens fitted on this model would weigh them was not measured.
          </Note>
          <EntryApparatus title="remembering">
            Δ<sub>ffn</sub> = (h<sub>post</sub> − s · h<sub>mid</sub>) · J<sub>ℓ</sub>
            <sup>⊤</sup>; remembering’s share = ‖Δ<sub>ffn</sub>‖ / (‖Δ<sub>attn</sub>‖ + ‖Δ<sub>ffn</sub>‖). The old h
            <sub>post</sub> − h<sub>mid</sub> met the true write at a median cosine of 0.49; the patch reaches ≥ 0.9985
            at all 60 layers (<M>V7_module1_fix.patch</M>; REPORT §3.4, bug 2). GeGLU block [15].
          </EntryApparatus>
        </Entry>

        <Entry
          id="global"
          head="global and local attention"
          tech={
            <>
              full and sliding-window attention ·{" "}
              <span style={{ fontVariantCaps: "normal", fontStyle: "normal", color: "var(--gilt-text)" }}>✦</span>
            </>
          }
          figure={<FigGlobal />}
        >
          <Body>
            Fifty of the sixty layers gather only from the last 1,024 word-pieces: local attention. Every sixth layer,
            marked <span style={{ color: "var(--gilt-text)" }}>✦</span> (layers 6, 12, 18 and so on to 60), looks back
            over the whole conversation: global attention. On the instrument these ten are the gilded rings.
          </Body>
          <EntryApparatus title="global and local attention">
            sliding window of 1,024 on 50 layers (head dimension 256); full attention at layers 6, 12, … 60, indices 5,
            11, … 59 (head dimension 512, p-RoPE). From the Gemma 4 model card and config [1]; ARCHITECTURE §2.
          </EntryApparatus>
        </Entry>

        <Entry
          id="neuron"
          aliases={ALIASES.neuron}
          head="neuron and its two faces"
          tech="GeGLU neuron · its two poles, after Cammarata [9] · 43,008 poles per layer"
          figure={<FigNeuron />}
        >
          <Body>
            Each layer remembers with 21,504 neurons. A neuron can fire forwards or in reverse, and each direction
            carries its own meaning, so we speak of its two faces: the positive face, drawn as a filled lozenge, and the
            negative face, a hollow one. The two faces answer to unrelated things, yet at layers 41–58 they push the
            next word in mirror-image directions.
          </Body>
          {/* the example is the recorded page-54 neuron the folio opens on (NeuronFolio/recorded.ts), a label inside
              the band where labels held up; it goes through the label gate like any other, and the layer-36 example
              the entry once gave, a label the pages hold back, now stands in its apparatus */}
          {EXAMPLE_GATE.heldBack ? (
            <div>
              <Note>
                At layer {FOLIO_EXAMPLE.layerNo}, neuron {FOLIO_EXAMPLE.neuron} showed its positive face.
              </Note>
              <HeldBackWords
                gate={EXAMPLE_GATE}
                words={FOLIO_JPP.map(([t]) => t)}
                eye={FOLIO_NAIVE.map(([t]) => t)}
                size={17}
                style={{ marginTop: 4 }}
              />
            </div>
          ) : (
            <Note>
              At layer {FOLIO_EXAMPLE.layerNo}, neuron {FOLIO_EXAMPLE.neuron} showed its positive face; through the lens
              its words are {FOLIO_JPP.map(([t]) => wp(t)).join(", ")}:{" "}
              <span style={{ fontStyle: "italic" }}>English “am”, Polish “I am”, Spanish or Portuguese “we are”</span>,
              forms of “to be”. To the naked eye they are {FOLIO_NAIVE.map(([t]) => wp(t)).join(", ")}; among the words
              shown the two readings share {EXAMPLE_SHOWN.shared.map((t) => wp(t)).join(", ") || "none"}. Layer{" "}
              {FOLIO_EXAMPLE.layerNo} lies inside layers 41–58, where the study found such labels held up, in its sample
              of random neurons.
            </Note>
          )}
          <Note>
            Specialists call the two faces poles, after Cammarata’s essay{" "}
            <span style={{ fontStyle: "italic" }}>Neural Networks Are Already Clean</span> (Goodfire, 2026). Its place
            among the instrument’s ideas is told in <a href={href.lineage()}>the lineage</a>.
          </Note>
          <Rubric>
            Below layer 41, a neuron’s words are only a rumour, so the pages hold them back and keep them in the
            apparatus; so too where the lens and the naked eye share none of their first ten.{" "}
            <LexLink to="labels">Read the first caution.</LexLink>
          </Rubric>
          <EntryApparatus title="neuron and its two poles" label="Apparatus for neuron and its two faces">
            a = GELU<sub>tanh</sub>(x · w<sub>gate</sub>) · (x · w<sub>up</sub>), x the normed stream; pole σ =
            sign(a), bipolar in practice through the sign of x · w<sub>up</sub>; the pole writes σ · w<sub>down</sub>,
            labelled unembed((σ · w<sub>down</sub>) · J<sub>ℓ</sub>
            <sup>⊤</sup>) against the naked eye’s unembed(σ · w<sub>down</sub>). Poles after Cammarata 2026 [9]; GeGLU
            [15]. Output labels held up at layers 41–58 and were at chance at 11–31; layers 32–40 were not sampled
            (N1). One input label cannot describe both poles (REPORT §3.2). The label gate (REPORT §3.3, conditions 2
            and 3): the pages hold a neuron’s words back from their main reading below layer 41, and where the lens’s
            first ten word-pieces and the naked eye’s share none (<M>label_check.shared_top10</M>, counted by id in
            fp32, as N1 counted; within 41–58 that count ranked a sound label above an unsound one with AUC 0.84, for
            base-fitted J++ labels on random neurons; the cut at none of ten is the instrument’s, the study set none).
            The example above, 53:{FOLIO_EXAMPLE.neuron}+ in the study’s counting, ten a side:{" "}
            {sharedLine(EXAMPLE_GATE) ?? <Missing what="not yet checked ten a side" />}. A label the pages hold back:
            at layer 36 (index 35), neuron 21189’s negative face reads through the lens “myself”, “
            <span lang="hi">मुझ</span>”, “<span lang="zh">的我</span>” (English “myself”, Hindi “me”, Chinese “of
            me”); the instrument’s leaf keeps those words in its apparatus, with the reason.
          </EntryApparatus>
        </Entry>

        <Entry
          id="double-negative"
          head="double-negative regime"
          tech="gated feed-forward · a term from Cammarata [9]"
          figure={<FigDoubleNegative />}
        >
          <Body>
            Every neuron has a gate, and it usually fires hard only when the gate is open. A shut gate still lets a
            small backward trickle through. If the neuron’s other input is negative too, the two negatives multiply into
            a positive, and the neuron shows its positive face for an inverted reason.
          </Body>
          <Note>
            The study found this rare: under one per cent of strong firing. It matters most among the very strongest
            firings of neurons that seldom fire, which is one more reason a neuron’s label can mislead.
          </Note>
          <EntryApparatus title="double-negative regime">
            g = x · w<sub>gate</sub> &lt; 0 and u = x · w<sub>up</sub> &lt; 0, so a = GELU<sub>tanh</sub>(g) · u &gt;
            0, since GELU of a negative number is a small negative number. The gate is open (g &gt; 0) in 72–100% of
            strong events; the regime holds &lt; 1% of strong-event mass, but closed-gate leak events make 13–33% of
            the top-1% mass of sparse neurons at layers 11, 21, 56 and 58 (REPORT §3.2; N1). Term from Cammarata 2026
            [9].
          </EntryApparatus>
        </Entry>

        <Entry
          id="strength"
          head="strength"
          tech={
            <>
              absolute activation <Sym>|a|</Sym>
            </>
          }
        >
          <Body>
            How hard a neuron fired on this word-piece, whichever face it showed. At each layer the instrument lists
            fifteen neurons: by default the fifteen that <LexLink to="write">wrote most</LexLink> into the stream, or, if
            you choose, the fifteen that fired hardest.
          </Body>
          {/* a recorded example: the demonstration sighting's loudest neuron at layer 36 (act −7.156); that sighting ranks
              its neurons by |a|, as first built */}
          <p
            className="num"
            style={{
              margin: 0,
              fontSize: 16,
              color: "var(--ink)",
              borderTop: "1px solid var(--hair-faint)",
              borderBottom: "1px solid var(--hair-faint)",
              padding: "6px 0",
            }}
          >
            neuron 21189 · layer 36 · negative face · strength 7.2
          </p>
          <Note>
            However they are ranked, those fifteen carried only 10–41% of what a layer remembers where the study looked
            from layer 21 to 59, and about 70% at layers 3 and 11: a sample of the layer’s work, not the whole of it.
          </Note>
          <EntryApparatus title="strength">
            |a| = |mlp_act<sub>ℓ</sub>[i]| at the read position; σ = sign(a). The interface keeps the top 15 by write size
            |a| · ‖w<sub>down</sub> ⊙ w<sub>post</sub>‖ (<M>"pole_ranking": "write"</M>) unless you choose otherwise; the
            server’s own default, the recorded sighting and the instrument as first built keep the top 15 by |a| (
            <M>"activation"</M>). Each layer reports the fraction of its FFN write the fifteen explain. In E3 (12 prompts,
            last word-piece; indicative only), at the seven layers it sampled (3, 11, 21, 31, 41, 51 and 59), the two lists
            had a Jaccard index of 0.67–0.96 by layer, and the fifteen explained 10–41% of the FFN write at layers 21–59 and
            70–74% at layers 3 and 11, ranked either way, within half a percentage point (REPORT §3.4, bug 8). Magnitudes and shown shares are not calibrated (eps floor; softcap).
            Poles after Cammarata 2026 [9]; GeGLU [15].
          </EntryApparatus>
        </Entry>

        <Entry
          id="write"
          aliases={["write-size"]}
          head="write size"
          tech={
            <>
              <Sym>
                s · |a| · ‖w<sub>down</sub> ⊙ w<sub>post</sub>‖ / rms(y)
              </Sym>
            </>
          }
        >
          <Body>
            How much a neuron changed the stream at this word-piece: how hard it fired, times the length of what it
            writes. A neuron can fire hard and write little. The instrument lists each layer’s neurons by it unless you
            choose otherwise.
          </Body>
          <Note>
            The biggest writers explained no more of what a layer remembers than the loudest did: 10–41% either way where
            the study looked from layer 21 to 59, and about 70% at layers 3 and 11.
          </Note>
          <EntryApparatus title="write size">
            write_size = s · |a<sub>i</sub>| · ‖w<sub>down,i</sub> ⊙ w<sub>post</sub>‖ / rms(y) (<M>pole_write_sizes</M>;{" "}
            <M>poles_labels[].write_size</M>). Within a layer s and rms(y) are shared, so the server ranks by |a| · ‖w
            <sub>down</sub> ⊙ w<sub>post</sub>‖ (<M>"pole_ranking": "write"</M>, what the interface sends; the server’s own
            default ranks by |a|, for old callers); an edit in the vault changes its column, and its write size with it. The
            share explained is ⟨Σ c<sub>i</sub>, f⟩ / ⟨f, f⟩, f = post − s · mid (<M>poles_ffn_explained</M>). In E3 (12
            prompts, last word-piece; indicative only), the fifteen biggest writers explained 10–41% of the FFN write at
            layers 21, 31, 41, 51 and 59 and 70–74% at layers 3 and 11, the seven it sampled, within half a percentage point
            of the fifteen loudest (REPORT §3.4, bug 8).
          </EntryApparatus>
        </Entry>

        <Entry id="threads" head="threads" tech="reader filaments · direct-path score" figure={<FigThreads />}>
          <Body>
            A thread joins a neuron to a later one, up to six layers on, that seems to listen to it: what the first
            writes, the second reads. Trace a neuron’s threads to see where its influence may travel next.
          </Body>
          <Rubric>
            Treat a thread as a hint of where to look. Its score is an estimate from the weights and this sighting, not a
            measured effect.{" "}
            <LexLink to="thread-score">Read the sixth caution.</LexLink>
          </Rubric>
          <EntryApparatus title="threads">
            C<sub>(ℓ,i)→(ℓ′,j)</sub> = ⟨σ<sub>i</sub> w<sub>down</sub>
            <sup>(ℓ,i)</sup>, w<sub>gate</sub>
            <sup>(ℓ′,j)</sup>⟩ · |a<sub>ℓ′,j</sub>|, for ℓ &lt; ℓ′ ≤ ℓ + 6, norm weights folded in; a virtual-weight
            score after [14]. ρ(|C|, measured effect) = +0.41; the direct-path estimate reaches +0.87, the true gradient
            +0.93 (V7; REPORT §3.4, bug 4). The instrument chooses and ranks threads by the direct-path estimate (
            <M>thread_score="direct"</M>); C remains the server’s default and a choice on the instrument. Threads join the
            fifteen neurons listed at each layer, so they follow the sighting’s ranking: by default, between the biggest
            writers. V7 measured the direct-path score’s agreement with silencing (0.87) on threads among the neurons ranked
            by |a|; among the biggest writers it was not measured.
          </EntryApparatus>
        </Entry>
      </EntryGrid>
    </LexSection>

    {/* ---------------------------------------------------------------- iii · the instrument */}
    <LexSection id="inst-head" title="iii · the instrument">
      <Lede cap="T">
        he theodolite is a surveyor’s telescope set on graduated circles, made to measure the bearing and elevation of
        distant landmarks. This one surveys a stream of 5,376 numbers instead of a hillside. These are its parts, and
        the names of what it shows you.
      </Lede>
      <EntryGrid>
        <Entry
          id="lens"
          aliases={ALIASES.lens}
          head="the lens and the naked eye"
          tech="J++ lens · Jacobian lens · logit lens"
          figure={<FigLens />}
        >
          <Body>
            To read what a layer holds, the instrument turns the stream into word-pieces. The naked eye reads it as if
            it were already the last layer (the logit lens). The lens first translates each layer’s stream into the
            last layer’s terms with a fitted matrix, then reads it (the J++ lens; its sibling, the Jacobian lens, served
            the study for comparison but is not loaded by the instrument). From about layer 46 upward the J++ lens,
            fitted on the base model, reads much like the naked eye and is no more faithful to what the model says; a
            lens fitted on the instruct model does better there (R1).
          </Body>
          <Note>
            The shipped J++ lens was fitted on the base model, from 64 short passages of Wikipedia text, not on the
            instruct model you are reading. A lens refitted on the instruct model raised agreement with the model’s
            real next word, in its own replies at layer 51, from 25% to 45% (the study’s R1 refit). That refit is usable
            as a full lens at only five layers (41, 46, 51, 56 and 59), and the instrument does not load it: every reading
            here goes through the base-fitted J++. The head of every page names it, and a red line beside each layer’s
            words says what the reading can bear there.
          </Note>
          <EntryApparatus title="the lens and the naked eye">
            naked eye: unembed(h<sub>ℓ</sub>), the logit lens [8]. The lens: unembed(h<sub>ℓ</sub> · J<sub>ℓ</sub>
            <sup>⊤</sup>) with J the J++ lens (Ayonrinde &amp; Lindsey 2026) [4], whose backward pass follows the
            R-lens rules [5]; the study’s comparison lens is the Jacobian lens (Gurnee et al. 2026) [2, 3], on disk but
            not loaded by the server (<M>load_jlens=False</M>, <M>backend/server.py:76</M>). Both were fitted on the
            base model, <M>google/gemma-4-31B</M> in bf16, over WikiText-103 [7] (J++ on 64 passages of 128 word-pieces,
            the Jacobian lens on 861) and hosted by Neuronpedia [6]; the instrument reads the instruct model in 4-bit (QAT
            W4A16), so chat training and quantisation cannot be told apart in any gap (R1 §6). The server names its lens
            in <M>GET /api/health</M> and every sighting (<M>"lens"</M>). R1’s refit: along 128 fixed directions, stable
            from layer 31; as a full matrix at 11 layers, usable at 41, 46, 51, 56 and 59 (assistant-reply top-1 at
            layer 51: 0.45 against J++’s 0.25). The published J++ readout also drops non-semantic word-pieces; this
            instrument applies the matrix only (REPORT §3.1).
          </EntryApparatus>
        </Entry>

        <Entry
          id="j-space"
          head="J-space"
          tech="Jacobian space · the workspace the lens reads · Gurnee et al. [2]"
          figure={<FigJSpace />}
        >
          <Body>
            The common ground into which the lens carries every layer: the last layer’s own terms, where a stream can be
            read directly as words. When the instrument shows a word through the lens at a layer, it means only that the
            layer’s stream, carried into J-space by the lens, reads as that word. The carrying was learned on the base
            model, so this is the lens’s reading, not the model’s own report.
          </Body>
          <Note>
            The idea comes from Gurnee and colleagues at Anthropic (2026), in a paper titled{" "}
            <span style={{ fontStyle: "italic" }}>
              Verbalizable Representations Form a Global Workspace in Language Models
            </span>
            . At a word like “I”, in the middle layers (21–40), the study found the gender carried there is a default
            guess, usually the speaker’s, rather than the person the model finally works out is meant. The lens, the
            naked eye, a trained probe and a direction read without any lens agree.
          </Note>
          <EntryApparatus title="J-space">
            h̃<sub>ℓ</sub> = h<sub>ℓ</sub> · J<sub>ℓ</sub>
            <sup>⊤</sup> ∈ ℝ<sup>5376</sup>, J<sub>ℓ</sub> an averaged Jacobian of the final state with respect to
            layer ℓ’s (the J-lens estimator sums cotangents over targets t′ ≥ t and averages over sources), read out by
            unembed. Gurnee et al. 2026 [2], code [3], the J++ refinement [4], pre-fitted lenses [6]. Mid-layer pronoun
            binding is a role default, not resolved reference (REPORT §2.1, E7, V5); the lenses are comparative, not
            absolute, instruments (REPORT §3.1). J++’s pronoun direction stands nearly at right angles to the instruct
            model’s own average gradient at layers 21–40 (cos 0.07–0.15; E3, R1 §2.1), so a base-fitted lens is not shown
            to read this model’s own J-space there.
          </EntryApparatus>
        </Entry>

        <Entry
          id="thinking"
          aliases={ALIASES.thinking}
          head="through the lens"
          tech="J++ readout · the top five · once called plumes, then “what it was thinking”"
        >
          <Body>
            The few word-pieces the J++ lens, fitted on the base model, reads in this layer’s stream, each with its
            share: its guess at what the model would say next if it stopped here. Low in the model they are a murmur of
            many words; at the top, where no lens is needed, they are the model’s own odds for its next word. Below the
            top they are the lens’s reading of the stream, not a report from the model. The word it then writes is its
            likeliest only when it writes greedily with no repeat penalty:{" "}
            <LexLink to="chosen">the word it chose and its likeliest</LexLink>.
          </Body>
          <div style={{ display: "flex", flexDirection: "column", gap: 2, maxWidth: 380, marginTop: 4 }}>
            <span style={READOUT_LABEL}>after “Hi! I”, at layer 51</span>
            {AT_51.map((r) => (
              <ReadoutRow key={r.word} {...r} />
            ))}
            <span style={{ ...READOUT_LABEL, marginTop: 10 }}>and at layer 60</span>
            {AT_60.map((r) => (
              <ReadoutRow key={r.word} {...r} />
            ))}
          </div>
          <Rubric>
            A single reading is not a finding: compare it with a{" "}
            <LexLink to="matched-control">matched sighting</LexLink> before you believe it.{" "}
            <LexLink to="single">Read the second caution.</LexLink>
          </Rubric>
          <EntryApparatus title="through the lens">
            the top five of softmax(unembed(h<sub>ℓ</sub> · J<sub>ℓ</sub>
            <sup>⊤</sup>)) at the read position, from the demonstration sighting (E6, “Hi! I”). Shares are not
            calibrated probabilities (softcap 30; REPORT §3.4, bug 8). On prose the lens’s top word matched the model’s
            real next word at most about 5% of the time through layer 50 (5.1–5.2% at layers 46–47 in E3’s per-layer
            data; REPORT §3.1’s “3%” understates it) and 28% at layer 59. J++ [4].
          </EntryApparatus>
        </Entry>

        <Entry
          id="chosen"
          head="the word it chose and its likeliest"
          tech="the token written · the argmax · layer 60, the model’s own odds"
        >
          <Body>
            At layer 60 the instrument shows the model’s own odds for what comes next; its likeliest is the first of them.
            The word it chose is the word-piece it actually wrote there. Written greedily with no repeat penalty, the two
            are one (bar a rare near-tie between the writing and the instrument’s second reading, which the page notes),
            and the page marks it. With a repeat penalty, greedy takes the likeliest after the penalty, so the two can
            differ. Drawn by chance, it can choose any word-piece the sampler lets it draw. Whenever the two differ the
            page names both, and when the word it chose is not among the five likeliest it adds it below them, with its
            rank and its share of the odds.
          </Body>
          <Note>
            Earlier the page called layer 60’s first word “the word it says next”. That holds only for the first word-piece
            of an answer written greedily, as in the recorded sighting; the word it chose is matched by its id, never by
            its spelling or by the answer’s first word.
          </Note>
          <EntryApparatus title="the word it chose and its likeliest">
            chosen, sent with every sighting as <M>chosen</M>: its id, its share of the model’s own odds softmax(z)[t] and
            its rank 1 + #{"{"}z &gt; z<sub>t</sub>{"}"} from the raw last-layer scores z at the writing, ties counted, and its
            chance under the sampler (after the penalty, the temperature, top-k and top-p), null when greedy. The readout
            at layer 60 comes from the instrument’s second reading of the whole text, a separate 4-bit computation: it
            can, rarely, put another word-piece first where two were all but tied, and the page says so (
            <M>recapture_check</M>: the re-read’s first id and its score for the chosen piece, against the writing’s).
            Rows are matched by token id (<M>jpp_top_ids</M>).
          </EntryApparatus>
        </Entry>

        <Entry
          id="landmarks"
          head="landmarks"
          tech="monuments · survey frame · Löwdin"
          figure={<FigLandmarks />}
        >
          <Body>
            Three word-pieces you choose, such as “she”, “he” and “it”, that fix the directions of the survey. The
            instrument squares them into three axes all at once, each as near its own word as the others allow: the first
            word’s axis sets 0°, the second’s 90°, and the third’s points straight up. Every layer’s stream is then
            measured against them.
          </Body>
          <Note>
            Choose them on <a href={href.landmarks()}>the landmarks page</a>.
          </Note>
          <Rubric>
            Keep the same three, in the same order, whenever you compare sightings, and compare only sightings squared the
            same way: a frame squared in order, as first built and in the recorded sighting, turns with the order of the
            words.{" "}
            <LexLink to="frame">Read the fifth caution.</LexLink>
          </Rubric>
          <EntryApparatus title="landmarks">
            e<sub>1</sub>, e<sub>2</sub>, e<sub>3</sub> = (V V<sup>⊤</sup>)<sup>−½</sup> V on the unit rows{" "}
            {FNW_DEFAULT ? (
              <>
                fnw ⊙ W<sub>U</sub>[t<sub>m</sub>]
              </>
            ) : (
              <>
                W<sub>U</sub>[t<sub>m</sub>]
              </>
            )}
            , in fp32 [33] (<M>"frame_squaring": "lowdin"</M>, <M>"frame_fnw": {FNW_DEFAULT ? "true" : "false"}</M>: what
            the interface sends unless you choose otherwise); as first built, and in the server’s own default, modified
            Gram–Schmidt on raw W<sub>U</sub> without the final-norm weight, which turns with the order of the words. The
            server’s default landmarks are “spider”, “eight”, “insect”, chosen for its default two-hop prompt, which comes
            from [2]. A second mode locks the frame to one neuron: w<sub>gate</sub>, (σ · w<sub>down</sub>) · J
            <sub>ℓ</sub>
            <sup>⊤</sup> and the w<sub>gate</sub> of the reader on its first thread, up to six layers on, else of the
            next layer’s first listed neuron (its loudest, or, ranked by write size, its biggest writer) (
            <M>"survey_mode": "pole_motif"</M>). A neuron frame keeps Gram–Schmidt by default: its three axes have fixed
            roles, so their order is not a choice and the order fault does not arise; Löwdin is a choice there. The
            squaring applies to either mode; the weighted rows fnw ⊙ W<sub>U</sub> (<M>"frame_fnw": true</M>
            {FNW_DEFAULT ? ", the default" : ", a choice V7 did not measure"}) to words only (REPORT §3.4, bug 3).
          </EntryApparatus>
        </Entry>

        <Entry
          id="bearing"
          aliases={ALIASES.bearing}
          head="bearing and elevation"
          tech={
            <>
              azimuth <Sym>θ</Sym> · elevation <Sym>φ</Sym>
            </>
          }
          figure={<FigBearing />}
        >
          <Body>
            Where the stream points, measured against the landmarks the way a surveyor measures a hilltop. Bearing is
            the direction around the plane of the first two landmarks; elevation is how far it rises toward the third,
            from −90° to +90°.
          </Body>
          <Note>
            The three landmarks catch only a sliver of the stream, 2.4–3.9% of it in the study’s frame of raw rows, so a
            bearing is a rough sketch of direction. Each sighting shows its own share.
          </Note>
          <EntryApparatus title="bearing and elevation">
            (x, y, z) = (⟨h̃<sub>ℓ</sub>, e<sub>1</sub>⟩, ⟨h̃<sub>ℓ</sub>, e<sub>2</sub>⟩, ⟨h̃<sub>ℓ</sub>, e
            <sub>3</sub>⟩); θ = atan2(y, x), φ = asin(z / ‖(x, y, z)‖). In the frame as first built the frame catches
            2.4–3.9% of ‖h̃‖, and reordering the landmarks swings θ by a median of 58–99° (V7; REPORT §3.4, bug 3). The
            interface now squares the frame all at once (Löwdin’s symmetric orthonormalisation [33]) on{" "}
            {FNW_DEFAULT ? (
              <>
                the weighted rows fnw ⊙ W<sub>U</sub>
              </>
            ) : (
              "the raw rows"
            )} unless you choose otherwise; there reordering
            only relabels the axes (swap the first two and θ reads 90° − θ, with φ unchanged).{" "}
            {FNW_DEFAULT
              ? "Its caught share was not measured in the study; every sighting reports its own."
              : "On the same raw rows it spans the same three directions as the frame squared in order, so it catches the same share; on the weighted rows the share was not measured in the study. Every sighting reports its own."}
          </EntryApparatus>
        </Entry>

        <Entry id="kinship" head="kinship of layers" tech="linear CKA · the 60 × 60 map" figure={<FigKinship />}>
          <Body>
            How alike two layers arrange the same word-pieces, from 0, unrelated, to 1, identical. Every layer is
            perfectly akin to itself, so the map’s diagonal is always full; look for blocks of close kin, stretches
            where the model’s arrangement holds steady.
          </Body>
          <Note>
            For a question of a single word-piece the instrument falls back to a simpler likeness than true CKA.
          </Note>
          <EntryApparatus title="kinship of layers">
            CKA(X, Y) = ⟨H K<sub>X</sub> H, H K<sub>Y</sub> H⟩<sub>F</sub> / (‖H K<sub>X</sub> H‖<sub>F</sub> ‖H K
            <sub>Y</sub> H‖<sub>F</sub>), K = X X<sup>⊤</sup> over the word-pieces read up to the sighting’s position at <M>resid_post</M> (the prompt’s,
            then, for a later word-piece, its answer’s up to it), H the
            centring matrix; with one word-piece, the squared cosine of feature-centred states. Linear CKA (Kornblith et
            al. 2019) [13]. The doubled start mark raised mean kinship from 0.608 to 0.663 (V7; REPORT §3.4, bug 6).
          </EntryApparatus>
        </Entry>

        <Entry id="how-it-writes" head="how it writes" tech="decoding · greedy, or sampled · max_new_tokens">
          <Body>
            Before the instrument sights an answer, the model writes it, one word-piece at a time, up to as many as you
            allow (1 to {MAX_WRITE}); it stops sooner when it writes one of its stop marks: the mark that ends its turn,
            or another, such as the end of the text, which the page names, since a strong nudge can force one where the
            answer should go on. By default it writes its likeliest word-piece every time, as in the answers the study
            checked the lens on (greedy). Drawn by chance, each word-piece is picked at random, weighted by its
            odds, as chat apps do, and four settings shape the draw, in the order the sampler applies them:
          </Body>
          <dl className="dl" style={{ margin: 0, fontSize: 17 }}>
            <dt>hold back repeats</dt>
            <dd>
              takes a fixed amount off the score of every word-piece it has already written, once each, so it repeats
              itself less; your question and its begun answer do not count. It applies to the likeliest word too{" "}
              <i>(presence penalty)</i>.
            </dd>
            <dt>temperature</dt>
            <dd>
              divides every score: below 1 the likeliest gain, above 1 the odds even out; 0 is the same as its likeliest
              word.
            </dd>
            <dt>from its likeliest</dt>
            <dd>
              only the K likeliest can be drawn; 0 sets no limit <i>(top-k)</i>.
            </dd>
            <dt>within</dt>
            <dd>
              only the fewest likeliest whose odds add up to P can be drawn; 1 sets no limit <i>(top-p, or nucleus)</i>.
            </dd>
          </dl>
          <Note>
            Then one word-piece is drawn. The model’s makers suggest temperature {MODEL_SAMPLING.temperature}, top-k{" "}
            {MODEL_SAMPLING.topK} and top-p {MODEL_SAMPLING.topP}. The study’s counts of what the model says were drawn that
            way, many times over; the answers of its own on which the study checked the lens (E3, R1) were written greedily.
          </Note>
          <Rubric>{strings.drawAdvice}</Rubric>
          <Onward to={href.instrument()}>set it on the instrument, under the question →</Onward>
          <EntryApparatus title="how it writes">
            for each word-piece, from the raw last-layer scores z: z<sub>t</sub> ← z<sub>t</sub> − p for each t already
            written; greedy, argmax z; else z / T, then every score below the k-th highest set to −∞, softmax, then the
            fewest likeliest whose odds reach top-p kept and renormalised, and one drawn by <M>torch.multinomial</M> after{" "}
            <M>torch.manual_seed(seed)</M> (interp-engine <M>steer._sample_next</M> [36]). Top-k after Fan, Lewis &amp;
            Dauphin 2018 [42]; top-p after Holtzman et al. 2020 [43]; the penalty on written word-pieces only and before the
            temperature, as vLLM orders it [44]. The page sends every field, <M>"sampler": {"{"}temperature, top_k, top_p,
            presence_penalty, seed{"}"}</M>; the server takes one left out as off (greedy, no limit, no penalty) and passes each
            on explicitly, since interp-engine reads a missing one as the checkpoint’s own setting. The writing stops at
            the checkpoint’s end marks (ids 1, 106 <M>{"<turn|>"}</M> and 50), and that mark is the answer’s last word-piece.
          </EntryApparatus>
        </Entry>

        <Entry id="seed" head="the seed" tech="random seed · torch.manual_seed">
          <Body>
            The number the chance in a drawn answer starts from. Leave it blank and the server draws one and sends it back
            with the sighting; keep it to draw with it again. The same seed, settings and question draw the same
            word-pieces again only if the model’s scores come out exactly as before, which a GPU does not promise.
          </Body>
          <Note>
            So to read one answer again (after a change of landmarks, say), the instrument does not draw it again: it
            reads the very word-pieces it wrote, by their ids, with nothing written. The nudge page draws its nudged answer
            with the plain answer’s own seed, so the two share their chance up to where they part.
          </Note>
          <EntryApparatus title="the seed">
            <M>seed</M> 0 to 2<sup>53</sup> − 1, or null: the server then draws one below 2<sup>31</sup> and echoes it with{" "}
            <M>seed_drawn: true</M>. A greedy answer uses none (null). The sighting with thinking turned on is drawn with
            the same settings, and with the same seed when the server drew it (<M>think_seed</M>).
          </EntryApparatus>
        </Entry>

        <Entry
          id="each-word-piece"
          head="a sighting of each word-piece"
          tech="one sighting per token written · read_pos = P − 1 + k · one frame pair each"
        >
          <Body>
            With “sight each word-piece it writes” ticked, one sweep gives a sighting of each word-piece the model wrote,
            the first {SIGHT_CAP} at most, each taken where the model chose it: there the stream had just read the
            word-piece before it. Choose a word-piece of the answer on the instrument to see its sighting. The last one it
            wrote is chosen, but never read.
          </Body>
          <NotAMatchedPair />
          <Note>
            Inside its own answers the study found the lens’s lead over the naked eye a third of its lead on prose at
            layers 31 to 40, and gone from layer 41 (E3).
          </Note>
          <EntryApparatus title="a sighting of each word-piece">
            <M>sight_every_token</M> (or <M>sight_tokens</M>, a list of k), at most <M>max_sighted</M> = {SIGHT_CAP}. The
            answer w<sub>0</sub> … w<sub>n−1</sub> is written once and read again once, the prompt’s P word-pieces and w
            <sub>0</sub> … w<sub>n−2</sub>, under the same nudge; the sighting of k reads position P − 1 + k, where layer 60
            chose w<sub>k</sub>. Kinship runs over positions 0 to P − 1 + k. With a kept control and centring, word-piece k
            ≥ 1 is centred on the control reading this same answer (<M>centre_answer</M>), never on the control’s own last
            word-piece. The space-then-digit rule, which took a rank from the next position, is off (
            <M>rank_lookahead: false</M>). The server holds its GPU lock from the writing to the last pair, so no vault
            edit lands between two word-pieces of one answer; each further word-piece sighted takes about 0.7 s.
          </EntryApparatus>
        </Entry>

        <Entry
          id="rubric"
          head="rubric"
          tech="from the Latin rubrica, red ochre"
          figure={
            <p
              style={{
                flex: "0 1 200px",
                margin: 0,
                paddingTop: 6,
                fontStyle: "italic",
                fontSize: 19,
                lineHeight: 1.4,
                color: "var(--rubric)",
              }}
            >
              Take a sighting, then choose any layer to open it.
            </p>
          }
        >
          <Body>
            A line in red italic that tells you what to do next. Scribes and early printers set their headings and
            directions apart in red ink, so that a reader could find the way through a long book at a glance; the word
            comes from the Latin <span style={{ fontStyle: "italic" }}>rubrica</span>, the red ochre they used. Rubrics
            run through every page of the instrument; once you know the way, switch them off with the rubrics button at
            the top of the page.
          </Body>
          <EntryApparatus title="rubric">
            rubrica, red ochre: the headings, initials and directions that manuscripts and early printed books set in
            red. On these pages every italic line in rubric red is guidance, never a reading; the rubrics setting (on by
            default) shows or hides them all, on every page.
          </EntryApparatus>
        </Entry>

        <Entry
          id="apparatus"
          head="apparatus"
          tech="from the critical edition’s apparatus criticus"
          figure={<FigApparatus />}
        >
          <Body>
            The scholar’s notes. In a critical edition, the apparatus is the band of small print at the foot of the page
            that records the variant readings, the sources and the editor’s reasoning. Here it is a band beneath each
            section, ruled off by a double gilt line, that gives the formulas, the raw numbers, the sources and every
            control the instrument has.
          </Body>
          <Note>
            Switch it on with the apparatus button at the top of any page. Switch it off, and the page reads as plainly
            as before; nothing on it is lost.
          </Note>
          <EntryApparatus title="apparatus">
            each block gives, where they exist: the technical name, the formula, the raw values from the repository,
            the reference numbers of REFERENCES.md, the study’s caveat with its section of <M>experiments/REPORT.md</M>,
            and the controls for experts. A value the repository does not record is shown as <Missing />, never
            estimated. The sources are gathered in <a href={href.lineage()}>the lineage</a>.
          </EntryApparatus>
        </Entry>

        <Entry id="atlas" head="the atlas" tech="neuron explorer · every feed-forward neuron at once">
          <Body>
            Every neuron of every layer on one map: sixty sheets of 21,504 dots, each layer’s attention heads drawn as
            lines just before its sheet. Open a neuron to read its words, follow its <LexLink to="web">web</LexLink>, read
            its <LexLink to="census">census</LexLink> answers and <LexLink to="push">push</LexLink> it; run a text to see
            what fires on each word-piece; or ask <LexLink to="scientist">the scientist</LexLink>.
          </Body>
          <Note>
            Built after lyramakesmusic’s neuron-explorer, made for Qwen3.5-4B in the wake of Cammarata’s neuron poles.
            Its labels go through the same gate as every other page’s: below layer 41 a neuron’s words through the lens
            are held back.
          </Note>
          <Onward to={href.atlas()}>open the atlas →</Onward>
          <EntryApparatus title="the atlas">
            One draw call for all 1,290,240 neurons (three.js points); picking meets each sheet as a plane, never the dots.
            Its indexes are built once on the GPU by <M>scripts/atlas_build.py</M> (lens words, activation ranges,
            wiring, layout; the <LexLink to="steer-lens">steer lens</LexLink> on its own) into <M>atlas_data/index</M>;
            the routes are <M>/api/atlas/*</M> (ARCHITECTURE §4.7). Layers count from 0 in the API, from 1 on the page.
          </EntryApparatus>
        </Entry>

        <Entry id="census" head="the census" tech="steering sweep · ranked against random directions">
          <Body>
            Every neuron of a layer pushed in turn, along its own write direction, on four fixed questions at a few
            doses, and the words its pushes bring out counted. The same is done along random directions, as controls,
            shaped the way the layer shapes every neuron’s write so the next layer hears them as loudly. A neuron whose
            pushes bring out words that the random pushes do not ranks high; one indistinguishable from the controls
            ranks low.
          </Body>
          <Note>
            A rank is a lead, not a finding: with 256 controls about one neuron in 257 lands above all of them by chance
            alone, some 84 of a layer’s 21,504. The doses are neuron-explorer’s, chosen on Qwen3.5-4B and not yet
            calibrated on this model, so a low rank can mean only that the dose was wrong for the layer. A census runs a
            layer at a time on the GPU and is kept; the map colours a layer’s neurons by it, from pale to dark gilt above
            the strongest control.
          </Note>
          <EntryApparatus title="the census">
            Questions: who are you, a snake game in Python, the colour orange (at −0.7, +0.5 and +0.8 of the stream’s
            median length leaving the layer) and “write whatever you like” at +3.0; greedy answers. A word (three
            letters or more) counts once per answer; z = (count − mean) / √(mean + 0.5) against the average unit, over
            pushes of at most 1.6 of the stream, every word of the unpushed answers left out; mass = the summed positive z
            of its top five words, the sort key; p = (1 + controls with at least its mass) / (C + 1), and q its
            Benjamini–Hochberg false-discovery q over the layer. A control is unit(w<sub>post</sub> ⊙ r), r isotropic from
            a seed, pushed exactly as a neuron is: the reference rotates a down column to a random direction, and on Gemma
            that column passes w<sub>post</sub>, near zero on the ~180 dimensions the next layer reads loudest (a plain
            isotropic direction reaches it about a hundred times louder). <M>scripts/census_run.py</M>,{" "}
            <M>backend/census.py</M>; the reference’s <M> sweep/rank.py</M>.
          </EntryApparatus>
        </Entry>

        <Entry id="web" head="the web" tech="wiring · connection cone · weight-only links">
          <Body>
            The open neuron’s strongest links through the weights, followed hop by hop: back through the earlier neurons
            whose writes it reads, forward through the later ones that read its write. Each link is read from rest: a
            writer moves a reader when it fires on the face that opens the reader’s gate, and pushes the reader toward
            one face, gilt its + face, blue its − face. The words of the neurons in its web are pooled to say what drives
            it up and down, and what it drives when it fires up and when it fires down.
          </Body>
          <Note>
            A link through the weights says what could pass, not what does on a given text, where the reader’s other
            inputs decide where it sits. Gemma scales the whole stream at the end of every layer, so a writer far back
            reaches a late reader faint: links are ranked by how strongly they arrive. Past the first hop every path is
            a guess, and pooled words go beyond what the label gate was tested on: test it with a push or a read.
          </Note>
          <EntryApparatus title="the web">
            A writer A at layer i writes v<sub>A</sub> = w<sub>post,i</sub> ⊙ W<sub>down,i</sub>[:, A]; a reader B at
            layer j &gt; i reads it as g = W<sub>gate,j</sub>[B] · (w<sub>pre,j</sub> ⊙ v<sub>A</sub>) and u = W
            <sub>up,j</sub>[B] · (w<sub>pre,j</sub> ⊙ v<sub>A</sub>); P = g·u. Its strength as it arrives is |P| f², f =
            s<sub>i</sub>⋯s<sub>j−1</sub> / (rms of block i’s output · rms of the stream entering block j), the layer
            scalars and the two norms at their usual scale; z is |P| z-scored within the layer pair. From rest the writer
            opens the reader’s gate on face sign(g) and pushes it toward face sign(P); gelu keeps the gate near shut the
            other way. The web takes the neuron’s strongest links, then each node’s top two with z ≥ 15, following a link
            only when the face it was reached on opens the next gate (<M>wiring.npz</M>). Direct-path only, as the
            reference’s <M>connections</M>, whose signs and ranking this mends for Gemma.
          </EntryApparatus>
        </Entry>

        <Entry id="steer-lens" head="the steer lens" tech="steering lens · the next word-piece’s change of log-probability">
          <Body>
            What pushing a neuron does to the next word, measured rather than read: push the stream along what the
            neuron writes, at every word-piece of four short texts, and see which word-pieces become likelier next, each
            way. It reads the model’s own output, not a lens fitted on another model, so the label gate does not hold it
            back.
          </Body>
          <Note>
            The push is not small. At most layers of this model it is longer than anything the neuron can write by
            itself, however hard it fires (the details say by how much for each neuron), so the steer lens says what a
            push that large along the neuron’s direction does, not what the neuron does when it fires. Where it and the
            lens disagree, neither wins: test with a smaller push and with what the neuron reads.
          </Note>
          <EntryApparatus title="the steer lens">
            ±0.8 · p50<sub>ℓ</sub> along unit(w<sub>post</sub> ⊙ W<sub>down</sub>[:, n]) at <M>resid_post</M> of its layer,
            at every position but the start mark, of two chat turns (“Who are you?”, “Write whatever you like.”) and two
            plain openings; the mean Δ log p of the next word-piece (fp32, softcapped at 30), its twelve largest rises
            each way, and KL(pushed ‖ base). <M>steer_lens.npz</M>, or read live for one neuron.
          </EntryApparatus>
        </Entry>

        <Entry id="push" head="a push" tech="steering · three voices · one neuron’s write direction">
          <Body>
            Ask the model something and it answers three times at once: with the open neuron pushed up, left alone, and
            pushed down, each answer keeping its own earlier replies. “add” pushes along the neuron’s own write direction
            by a share of the stream’s usual length; “×” multiplies what the neuron writes.
          </Body>
          <Note>
            A push lasts for that answer only. Unlike a <LexLink to="nudge">nudge</LexLink>, it moves along what one neuron
            writes. On Qwen3.5-4B the explorer’s authors found answers bent but readable at 0.5 to 0.8 of the stream and
            coming apart by 3; on Gemma-4-31B that is not yet measured.
          </Note>
          <EntryApparatus title="a push">
            “add”: h ← h + σ · d · p50<sub>ℓ</sub> · unit(w<sub>post</sub> ⊙ W<sub>down</sub>[:, n]) at <M>resid_post</M>{" "}
            of layer ℓ, σ = +1, 0, −1 for the three voices, start mark spared; “×”: a<sub>n</sub> ← d · a<sub>n</sub>, a
            <sub>n</sub>, −d · a<sub>n</sub>. One batch of three, greedy unless “sampled” is chosen: then the instrument’s
            own sampler at the temperature and seed shown, one draw a step shared by the three voices, so they part only
            where the push moves the odds; each turn says which. p50 is the stream’s median length leaving the
            layer, over the stats index’s text, or over the study’s 36 texts (provisional) until that index is built. The
            most a neuron can write by itself is s<sub>ℓ</sub> · √D · ‖w<sub>post</sub> ⊙ unit(W<sub>down</sub>[:, n])‖
            (the norm caps it), and the push tab says what a dose is against it. <M>POST /api/atlas/push</M> (events).
          </EntryApparatus>
        </Entry>

        <Entry id="names" head="names and notes" tech="the atlas’s field book · field_notes.json, field_book.json">
          <Body>
            A name and a note kept beside a neuron on the server, by you or by the scientist, for everyone who uses the
            atlas there. A named neuron shows in red on the map, on every layer. Name a neuron in the atlas’s left panel
            or on its folio, which is the same name; leave a note in its details. A folio’s field notes, face by face, are
            kept beside them, and the atlas’s details show them.
          </Body>
          <Note>
            A name is a guess written down. The scientist marks its own with how sure it is: nothing when its firing, its
            pushes and its words agree, (?) to (???) as they part, and only proposes it: the atlas shows its proposal with
            keep and discard, and nothing reads it, the map and later chats included, until you keep it. With no server
            to keep them, a folio keeps its name and field notes in this browser, for you alone, and says so.
          </Note>
          <EntryApparatus title="names and notes">
            One JSON file outside every checkout, written whole each time: <M>$THEODOLITE_FIELD_NOTES</M>, else{" "}
            <M>~/.local/share/theodolite/field_notes.json</M>; each entry, a neuron’s or a head’s, keeps who wrote it and
            when. <M>GET/POST /api/atlas/notes</M>. The folios’ field book beside it, per face (jobs, pattern, four ticks,
            who and when): <M>$THEODOLITE_FIELD_BOOK</M>, else <M>field_book.json</M> in the same folder, a file a server
            from before never reads or writes. <M>GET/POST /api/atlas/fieldbook</M>. The scientist’s proposals wait in{" "}
            <M>field_proposals.json</M> beside them (<M>$THEODOLITE_FIELD_PROPOSALS</M>), each with the chat that made it;
            kept, one is written to the notes as the scientist’s. <M>GET/POST /api/atlas/proposals</M>, and{" "}
            <M>/keep</M> and <M>/discard</M>.
          </EntryApparatus>
        </Entry>

        <Entry id="scientist" head="the scientist" tech="an agent with the atlas’s tools">
          <Body>
            A language model in the atlas’s side panel, with the atlas’s own tools: ask it about a neuron and it reads the
            neuron’s words, web and census, runs pushes and texts, names what it finds and shows you on the map. Press
            “explain” for a short report on the open neuron or head.
          </Body>
          <Note>
            It runs on Claude Code or Codex where they are installed beside the server, or on any service that speaks
            OpenAI’s chat completions: LM Studio or Ollama on this machine, OpenRouter, or one you name. Under Claude Code
            it reads and writes files only in its own work folder and has no shell, unless the server is started to give
            it a shell over your home folder, which it then uses without asking; the page says which, and asks first.
            What it says is a lead to check: it cites the neurons it read, so open them, and a name it gives is marked as
            its own guess.
          </Note>
          <EntryApparatus title="the scientist">
            30 tools (<M>backend/scientist_tools.py</M>), served to Claude Code and Codex over MCP (
            <M>backend/scientist_mcp.py</M>) and to an OpenAI-compatible endpoint as functions, the tool loop then run by
            the server; its instructions are <M>backend/scientist_prompt.md</M> and <M>scientist_methodology.md</M>.
            Endpoints come from the environment or <M>endpoints.json</M>, their keys only from environment variables.
            The tools count layers as the pages do. <M>/api/atlas/scientist/*</M> (ARCHITECTURE §4.8).
          </EntryApparatus>
        </Entry>
      </EntryGrid>
    </LexSection>

    {/* ---------------------------------------------------------------- iv · acts of the reader */}
    <LexSection id="acts-head" title="iv · acts of the reader">
      <Lede cap="S">
        ix things you can do here. The first three only look. The last three change the model: a test only for the one
        reading it takes, a nudge only while it is on, an edit until you undo it.
      </Lede>
      <EntryGrid>
        <Entry id="sighting" head="a sighting" tech="sweep · all sixty layers" figure={<FigSighting />}>
          <Body>
            Write a question, and if you like the start of an answer; then take a sighting. A sighting is the stream at
            one word-piece, read up through all sixty layers: at each one, the words the lens reads there, how the
            layer’s change divides between gathering and remembering, and which neurons fired. One sweep gives one for
            each word-piece it writes (as the instrument asks by default), taken where it chose that piece: the first where it read your last word-piece, each
            later one where it read the piece before; the last piece it writes is chosen but never read.
          </Body>
          <SightingOnInstrument />
          <Onward to={href.instrument()}>take a sighting on the instrument →</Onward>
          <EntryApparatus title="a sighting">
            sent over the WebSocket <M>/ws/theodolite</M> as <span className="mono-wrap">{DEFAULT_REQUEST}</span> (what the
            instrument sends by default, generated from its own request builder, with a <M>request_id</M> of its own; old
            callers that send only a prompt get the server’s bare-text defaults). The reply streams{" "}
            <M>theodolite_progress</M> while it writes, <M>theodolite_written</M> with the answer once written, then one
            frame pair for each word-piece sighted, a binary THEO frame (path, split, kinship, 15 poles a layer, threads) and
            a JSON frame of word-pieces (ARCHITECTURE §4), then <M>theodolite_sweep_end</M>; a request with no{" "}
            <M>request_id</M> gets one pair, as before. The demonstration sighting is E6’s, 6,237 ms, taken before the two V7
            fixes. Served by FastAPI [40] over interp-engine’s hook points [36].
          </EntryApparatus>
        </Entry>

        <Entry
          id="matched-control"
          head="a matched control"
          tech="matched control · background correction"
          figure={<FigMatchedControl />}
        >
          <Body>
            A second sighting, taken to read the first against. It differs in only the one thing you are asking about:
            the same question with no name in it, say, or a plain background word in place of yours. Read the
            difference between the two; that difference, not either reading on its own, is what the study trusted.
          </Body>
          <Note>
            A surveyor checks a bearing against a known mark before writing it in the field book. A sighting needs its
            control for the same reason.
          </Note>
          <Rubric>
            To keep one, switch on the apparatus at the top of the page. On the instrument, take the control first and
            keep it with the button beneath your question; then change the one thing and take your sighting. Each
            layer’s apparatus then shows the control’s reading beside yours.
          </Rubric>
          <EntryApparatus title="a matched control">
            Δ = (r<sub>sighting</sub> − r<sub>control</sub>) / (r<sub>woman</sub> − r<sub>man</sub>): a readout’s
            shift, mean-centred, in units of the woman–man persona span (or another concept’s span). For example, the
            default against a nameless AI at layers 31–40: +0.6 [+0.2, +0.9]% of the span through the base-fitted J++
            lens (E1), and +3 [+2, +4]% through a lens refitted on the instruct model (R1): the sign holds, the size
            depends on the lens. A single-condition readout is uninterpretable without matched controls and background
            correction (REPORT §3.1; §3.5, recommendation 2).
          </EntryApparatus>
        </Entry>

        <Entry id="vault" head="the vault" tech="LARQL · VINDEX3 · LQL queries" figure={<FigVault />}>
          <Body>
            A catalogue of every neuron in the model, sixty layers of 21,504, that you can question in a small query
            language: which neurons lean toward a word, or what one neuron’s two faces say.
          </Body>
          <Query>DESCRIBE POLE LAYER 50 SLOT 5313</Query>
          <Note>
            The vault’s language counts layers from 0, so this asks after neuron 5313 at layer 51: a two-faced neuron
            for “she” and “he” that the study confirmed is real.
          </Note>
          <Note>
            Its answers are suggestions, not tests: of 102 gender neurons it proposed, 35 proved selective. The queries
            run in Python; LARQL lends only its name.
          </Note>
          <Onward to={href.vault()}>open the vault →</Onward>
          <EntryApparatus title="the vault">
            <M>SELECT * FROM POLES NEAREST 'she' WHERE layer = 50 LIMIT 10</M> ranks the 21,504 neurons of the vault’s
            layer 50, layer 51 here, by |cos(w<sub>down,i</sub>, (fnw ⊙ W<sub>U</sub>[t]) · J<sub>ℓ</sub>)|, the write
            column scaled by the post-feed-forward norm weight, the sign naming the pole; <M>EDGES NEAREST</M> runs the
            same query; <M>DESCRIBE POLE</M> decodes both poles by lens and naked eye. Also as the MCP tools{" "}
            <M>query_larql_vault</M>, <M>inspect_pole</M>, <M>trace_downstream_readers</M>,{" "}
            <M>test_motif_hypothesis</M>, and <M>causal_spot_check</M> for <LexLink to="test">a test of a label</LexLink>. Named after LARQL / VINDEX3 (chrishayuk) [10]; a projected contribution, not a
            counterfactual (REPORT §3.3). Proposed: activation contexts stored per pole (REPORT §3.4, bug 7).
          </EntryApparatus>
        </Entry>

        <Entry
          id="test"
          aliases={ALIASES.test}
          head="a test of a label"
          tech="causal spot-check · the study’s N1A, gently, at k = 10"
        >
          <Body>
            Put a neuron’s label to the test. The instrument turns the neuron up at one word-piece, once on each face,
            keeps an untouched copy of the same reading beside it, and lists the words whose odds rose and fell most.
            Then it counts how many of the label’s ten words are among the hundred whose odds rose most. The test reads
            the model’s own odds, not the lens, and nothing of it stays: the model answers as before. It is not the
            folio’s guess put to the test, which reads how strongly the neuron fires on sentences you write and turns
            nothing up.
          </Body>
          <Note>
            The study tested 384 faces on the same 36 texts, turned up six times as hard as the folio’s gentle test: at
            layers 41–58 a label’s words were among those that rose far more often than chance, and at layers 21 and 31
            no more often; at layer 11 the effect was too small to measure (N1V code audit, F2). It counted a label as
            sound when three of its ten words were among the hundred. A later look at the gentle size, on six layers,
            found the same split. A neuron’s folio can test at the word-piece of your sighting, or on the study’s 36
            texts; only the second compares with the study’s figures.
          </Note>
          <Rubric>
            Read a test’s count as what that one test showed, not as a verdict on the label: one word-piece speaks for
            its own text only, so test on the study’s 36 texts before you lean on a label. A test shows what a neuron
            does when turned up, not what it does when it fires on its own.
          </Rubric>
          <Onward to={href.neuron(FOLIO_EXAMPLE.layerNo, FOLIO_EXAMPLE.neuron)}>put a label to the test on a neuron’s folio →</Onward>
          <EntryApparatus title="a test of a label" label="Apparatus for a test of a label">
            a<sub>i</sub> ← a<sub>i</sub> ± k · max|a<sub>i</sub>| at the read position p only (a pre-hook on down_proj),
            k = 10 by default (erratum E47), max|a| over the study’s 36 texts (12 chats, 12 passages of prose, 12 of
            code; every word-piece after the start mark); or the frozen-norm write, the same first-order change added
            after the post-feed-forward norm, which cannot squash it (N1V). c = log_softmax(z<sub>pushed</sub>) −
            log_softmax(z<sub>untouched</sub>) on fp32 softcapped logits at p, the untouched row in the same forward;
            the same test at k = 0.01 measures bf16 rounding. P@10: the share of the label’s first ten ids (the lens’s
            and the naked eye’s, as <M>label_check</M> counts them) among the hundred ids c raised most. On its 36 texts
            at k = 10 the study’s lens labels scored {STUDY_K10_LATE_P10.map((v) => v.toFixed(2)).join("–")} at layers
            41–56, and{" "}
            {STUDY_K10[20].p10Lens === STUDY_K10[30].p10Lens
              ? STUDY_K10[20].p10Lens.toFixed(2)
              : `${STUDY_K10[20].p10Lens.toFixed(2)} and ${STUDY_K10[30].p10Lens.toFixed(2)}`}{" "}
            at layers 21 and 31 (N1V code audit). Its bar of three in ten (P@10 ≥ 0.3, what N1’s agreement measure predicted with AUC 0.84)
            was set on N1A’s tests of 384 faces at k = 60, which saturate the post-feed-forward norm at layers 21–51
            (E47). The comparison with the label goes through the base-fitted J++ lens; the effect does
            not. Sent as the WebSocket action <M>causal_check</M>, <M>POST /api/scientist/causal_check</M> and the MCP
            tool <M>causal_spot_check</M> (ARCHITECTURE §4.5; <M>backend/causal_check.py</M>).
          </EntryApparatus>
        </Entry>

        <Entry id="nudge" head="nudge" tech="steering · once called the laser" figure={<FigNudge />}>
          <Body>
            Push the stream at chosen layers toward one word and away from another, then watch what the model says
            instead. A nudge leaves nothing behind: turn it off and the model answers as before.
          </Body>
          <Rubric>
            Keep a nudge small, about a hundredth of the stream’s strength. The old defaults break the model:{" "}
            <LexLink to="nudge-defaults">read the seventh caution</LexLink> before you nudge.
          </Rubric>
          <Onward to={href.nudge()}>open the nudge →</Onward>
          <EntryApparatus title="nudge">
            What the instrument sends: h<sub>ℓ</sub> ← h<sub>ℓ</sub> + s · ‖h<sub>ℓ</sub>‖ · v / ‖v‖ at <M>resid_post</M> of
            each chosen layer and position, start mark spared, with v = (fnw ⊙ ΔW<sub>U</sub>) · J<sub>ℓ</sub> (no inverse,
            no clamp) and its quiet dimensions masked, or one pole’s write σ · w<sub>down</sub>; s = 0.01 by default (
            <M>direction="fnw", mask_quiet, spare_bos, strength</M>). Masked that way the study’s she/he nudge read
            fluently at 0.005–0.01 of ‖h‖ and broke at 0.02. The server’s own defaults, kept for old callers: v = ((W
            <sub>U</sub>[t<sub>+</sub>] − W<sub>U</sub>[t<sub>−</sub>]) / max(fnw, 0.1)) · J<sub>ℓ</sub>, “six” against
            “eight”, a fixed k = 650 at layers 30, 36, 42, 48, 54, every position, which collapses the model (REPORT §3.4,
            bug 1; §3.5, recommendation 4; the old interface sent k = 280). Activation addition [21, 22].
          </EntryApparatus>
        </Entry>

        <Entry id="edit" head="an edit" tech=".patch · a one-neuron column edit, not MEMIT" figure={<FigEdit />}>
          <Body>
            Change what one neuron writes, in the model’s own weights, and see what follows. An edit stays until you
            undo it; one command undoes them all:
          </Body>
          <Query>ROLLBACK</Query>
          <Note>
            The study tested the vault’s top ten “she” neurons at layer 51 this way, deleting them and rolling back: the
            pull toward “she” in passages about women fell by 1.80 log-odds, while ten random neurons moved it by 0.04
            at most.
          </Note>
          <Note>
            Despite its old label, an edit is not MEMIT: it rewrites one neuron’s output column and solves for nothing
            else.
          </Note>
          <Note>
            The vault can hear an edit: it asks the model one question just before the edit and again just after, and
            shows both answers. That shows what the edit did to that one question only; it may change answers you never
            asked for.
          </Note>
          <EntryApparatus title="an edit">
            <M>UPDATE LAYER l SLOT i SET object = 't' [STRENGTH α] [POLARITY ±]</M>: w<sub>down,i</sub> ← w
            <sub>down,i</sub> + σ · α · max(‖w<sub>down,i</sub>‖, 0.25) · (fnw ⊙ W<sub>U</sub>[t]) · J<sub>ℓ</sub> /
            ‖(fnw ⊙ W<sub>U</sub>[t]) · J<sub>ℓ</sub>‖, α = 2.5 by default; <M>INSERT</M> picks the slot whose gate
            best matches the subject; <M>DELETE FROM LAYER l WHERE slot = i</M> sets the column to zero;{" "}
            <M>ROLLBACK</M> clears every overlay. Not MEMIT [11]: no covariance solve (REPORT §3.3); rank-one editing
            as background [12]. Heard (<M>voice</M>): the vault writes the answer the likeliest way, with no nudge, at
            most 12 word-pieces by default (1–64), before the edit and again after it under the one GPU lock, and gives
            both with the KL between the two at the first word-piece. The vault page asks for it, 12 word-pieces long;
            the console and the MCP tool make an edit unheard. The study tested only <M>DELETE</M> causally
            (E6 §2.3).
          </EntryApparatus>
        </Entry>
      </EntryGrid>
    </LexSection>

    {/* ---------------------------------------------------------------- v · where the instrument can mislead you */}
    <Cautions />

    {/* ---------------------------------------------------------------- vi · about the study */}
    <section aria-labelledby="study-head" style={{ display: "flex", flexWrap: "wrap", gap: "32px 72px", paddingTop: 64 }}>
      <div style={{ flex: "999 1 560px", minWidth: 0, display: "flex", flexDirection: "column", gap: 16 }}>
        <h2 className="h-sec" id="study-head">
          vi · about the study
        </h2>
        <p style={{ margin: 0, fontSize: 21, lineHeight: 1.55, maxWidth: 760 }}>
          This instrument was built alongside a study that asked a single question: is Gemma 4 “female-coded”? By
          default, no. Along the way the study turned the instrument on itself: seven experiments, eight adversarial
          verifiers that re-tested every headline with new controls, a study of neuron labels against their real
          effects, and a lens fitted afresh on the instruct model, 314 GPU jobs in all. Every caution on this page comes
          from that work.
        </p>
        <p style={{ margin: 0, maxWidth: 760 }}>
          Where the verifiers read through a lens, it was one of the two the experiments used, J++ and the Jacobian
          lens: published lenses, both fitted on the base model outside the study. One
          of them (V6) measured the J++ lens against the instruct model’s own gradients and put the gap down to a small
          set of quiet numbers, which the refit later found too narrow. The refit (R1) re-read only the self-gender
          readings and the lens’s faithfulness, with a lens fitted on the instruct model.
        </p>
        <p style={{ margin: 0, maxWidth: 760 }}>
          To read more, open <code style={{ fontFamily: "var(--mono)", fontSize: 16 }}>experiments/REPORT.md</code> in
          the repository, and read the README’s “Status and known issues” before you trust a reading.
        </p>
        <p style={{ margin: 0, maxWidth: 760 }}>
          The ideas this instrument joins together, from the Jacobian lens and J-space to the neuron’s two poles and the
          vault’s names, are set out with their makers in <a href={href.lineage()}>the lineage</a>.
        </p>
        <Apparatus
          title="about the study"
          style={{ maxWidth: 760, gap: 0, fontVariantNumeric: "lining-nums tabular-nums" }}
        >
          <p style={{ margin: "2px 0 0" }}>
            E1–E7 experiments, V1–V8 adversarial verifiers, the N1 neuron-label study with two verifiers, and the R1 lens
            refit: 314 GPU jobs through one lab daemon, on Gemma-4-31B-it (QAT W4A16) [1] and one RTX 5090. Its lens
            readings went mostly through lenses fitted on the base model <M>google/gemma-4-31B</M> in bf16; E3 and V6
            also took the instruct model’s own average gradients, and R1 fitted a lens on it. The study’s
            verdict on the synthesis of [2], [4], [9] and [10]: neat as a question, oversold as built (REPORT §3.3).
            Statistics use bootstrap intervals [30] and Wilson intervals [29]; the full list of {REFERENCE_COUNT} references is{" "}
            <M>REFERENCES.md</M>.
          </p>
        </Apparatus>
        <p style={{ margin: 0, fontSize: 16, color: "var(--muted)", maxWidth: 760 }}>
          The experiments and reports were designed, run and written by Claude Code agents, directed by the repository
          owner. Claims that did not survive verification are listed in §4 of the report.
        </p>
      </div>
      <div style={{ flex: "1 1 320px", minWidth: 0, display: "flex", flexDirection: "column", gap: 22 }}>
        <div
          style={{
            display: "flex",
            alignItems: "flex-end",
            gap: 18,
            paddingBottom: 16,
            borderBottom: "1px solid var(--hair)",
          }}
        >
          <span
            style={{
              fontSize: 96,
              lineHeight: 0.8,
              fontWeight: 500,
              color: "var(--gilt-numeral)",
              fontVariantNumeric: "lining-nums",
            }}
          >
            0
          </span>
          <span style={{ display: "flex", flexDirection: "column", gap: 2, paddingBottom: 2 }}>
            <span
              style={{
                fontSize: 15,
                fontVariantCaps: "all-small-caps",
                letterSpacing: "0.2em",
                color: "var(--muted)",
                fontVariantNumeric: "lining-nums",
              }}
            >
              of 320 samples
            </span>
            <span style={{ fontSize: 17 }}>in which Gemma 4, given no system prompt, called itself “she”</span>
          </span>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <h3 className="h-sub">a note on counting</h3>
          <p style={{ margin: 0, fontSize: 17 }}>
            The report numbers layers from 0; this instrument numbers them from 1. Every layer named on this page has
            been converted: the report’s “L40–57” is layers 41–58 here.
          </p>
        </div>
      </div>
    </section>
  </Page>
);

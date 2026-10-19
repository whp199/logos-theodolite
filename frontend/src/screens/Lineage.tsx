import React from "react";
import { IfApparatus, Page, ReadingThrough, Rubric, SectionHead } from "../components/kit";
import type { Route } from "../router";
import { Cartouche, Stemma } from "./lineage/figures";
import { Pillars } from "./lineage/Pillars";
import {
  ApparatusLinkNotice,
  Ext,
  Go,
  LinApparatus,
  LinkRow,
  M,
  NotInInstrument,
  To,
  type Dest,
} from "./lineage/bits";
import { REFERENCE_COUNT, REFS_D_TO_F, refId, refsA, refsB, refsC, refsG, type Reference } from "./lineage/references";

/**
 * Lineage — the synthesis and its makers (folio viii).
 *
 * A static folio: where each part of the instrument comes from, what the study found about it, and the
 * REFERENCES.md entries A–C and G. Nothing here reads the current sighting; every figure is a recorded finding of the
 * study (experiments/REPORT.md and the reports it names), as on the approved canvas.
 *
 * Anchors for href.lineage('<id>'):
 *   sections   stemma · pillars · jacobian-lens · jpp-lens · poles · larql · foundations · logit-lens · circuits ·
 *              glu · lrp · cka · steering · keep · practices · conventions · sources (apparatus)
 *   references ref-1 … ref-28 and ref-42 … ref-44 (apparatus); ref-29 … ref-41 land on the line that says they are in
 *   REFERENCES.md
 *   (the canvas's heading ids lin-title, stemma-head, pillars-head, p1-head … p4-head, found-head, keep-head,
 *   ask-head, conv-head and src-head also work.)
 */

const SECTION: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 14,
  padding: "34px 0 10px",
  borderTop: "1px solid var(--hair)",
  scrollMarginTop: 16,
};

/* ------------------------------------------------------------------ title */

const STATS: [number, string][] = [
  [4, "pillars it is chiefly made from"],
  [6, "older foundations beneath them"],
  [REFERENCE_COUNT, "sources in its reference list"],
  [1, "study made with it"],
];

const Title: React.FC = () => (
  <section
    aria-labelledby="lin-title"
    style={{ display: "flex", flexWrap: "wrap", gap: "36px 72px", padding: "40px 0 44px", alignItems: "center" }}
  >
    <div style={{ flex: "999 1 560px", minWidth: 0, display: "flex", flexDirection: "column", gap: 20 }}>
      <p
        style={{
          margin: 0,
          fontSize: 14,
          fontWeight: 500,
          fontVariantCaps: "all-small-caps",
          letterSpacing: "0.24em",
          color: "var(--rubric)",
        }}
      >
        the lineage
      </p>
      <h1
        id="lin-title"
        style={{
          margin: 0,
          fontSize: "clamp(42px, 5.2vw, 68px)",
          lineHeight: 1.04,
          fontWeight: 400,
          letterSpacing: "-0.005em",
          color: "var(--ink)",
          maxWidth: 760,
        }}
      >
        The instruments this one is made from
      </h1>
      <p style={{ margin: 0, fontSize: 23, lineHeight: 1.5, maxWidth: 680 }}>
        Logos Theodolite is a synthesis. Its maker took four recent pieces of research, a lens, a sharper grinding of
        that lens, a way of reading neurons, and a catalogue of neurons you can question, and joined them on one mount,
        over older foundations reaching back to 2015.
      </p>
      <p style={{ margin: 0, fontSize: 20, lineHeight: 1.55, maxWidth: 680 }}>
        Knowing where a reading comes from tells you what it was built to show. Knowing what the study found about each
        part tells you how far to trust it.
      </p>
      <Rubric style={{ maxWidth: 640 }}>
        Read the family tree first, then the four pillars. Each pillar ends with its source, so you can read the
        original for yourself.
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
        {STATS.map(([n, label]) => (
          <div key={label} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span className="num" style={{ fontSize: 42, lineHeight: 1, color: "var(--gilt-numeral)" }}>
              {n}
            </span>
            <span style={{ fontSize: 15, color: "var(--muted)" }}>{label}</span>
          </div>
        ))}
      </div>
    </div>
    <Cartouche />
  </section>
);

/** apparatus · this page */
const PageApparatus: React.FC = () => (
  <LinApparatus title="this page" label="Apparatus for this page" style={{ paddingBottom: 26 }}>
    <dt>numbering</dt>
    <dd>
      Bracketed numbers are those of <M>REFERENCES.md</M> ({REFERENCE_COUNT} entries, links checked 2026-10-16, those of G on
      2026-10-18); the full list of A–C and G is at the foot of this page.
    </dd>
    <dt>layers</dt>
    <dd className="num">
      Counted 1 to 60 on these pages. The report, the code and the vault count from 0: layer = index + 1, so the
      report’s L40–57 are layers 41–58 here.
    </dd>
    <dt>findings from</dt>
    <dd>
      <M>experiments/REPORT.md</M> §0, §2.1, §3.1–3.5; <M>E6_tool_audit_REPORT.md</M> §2.3; <M>N1_REPORT.md</M>;{" "}
      <M>R1_IT_LENS_REPORT.md</M>; the code in <M>backend/</M>.
    </dd>
    <dt>provenance</dt>
    <dd>
      The study’s experiments, analyses and reports were run and written by Claude Code agents directed by the
      repository’s owner. Every headline was re-tested by an independent verifier with new controls, but through the
      same base-fitted lenses; one verifier (V6) measured J++ against the instruct model’s own gradients and traced
      the gap to the quiet dimensions, which R1 later found was only part of it. R1 re-read only the self-gender
      readings and the lens-fidelity measures through an instruct-fitted lens. The claims that failed are listed in
      REPORT §4 (README).
    </dd>
    <dt>Cammarata</dt>
    <dd>The essay [9] is paraphrased throughout, never quoted; read it in full at its source.</dd>
  </LinApparatus>
);

/* ------------------------------------------------------------- foundations */

const Foundation: React.FC<{
  id: string;
  title: string;
  makers: React.ReactNode;
  what: React.ReactNode;
  found: React.ReactNode;
  links: React.ReactNode;
}> = ({ id, title, makers, what, found, links }) => (
  <div
    id={id}
    style={{
      flex: "1 1 340px",
      minWidth: 0,
      display: "flex",
      flexDirection: "column",
      gap: 8,
      padding: "18px 0 20px",
      borderTop: "1px solid var(--hair-soft)",
      scrollMarginTop: 16,
    }}
  >
    <h3 style={{ margin: 0, fontSize: 24, fontWeight: 400, lineHeight: 1.2 }}>{title}</h3>
    <span style={{ fontSize: 15, color: "var(--muted)" }}>{makers}</span>
    <p style={{ margin: 0, fontSize: 17 }}>{what}</p>
    <p style={{ margin: 0, fontSize: 16, color: "var(--muted)" }}>{found}</p>
    <LinkRow gap={22}>{links}</LinkRow>
  </div>
);

const Foundations: React.FC = () => (
  <section id="foundations" aria-labelledby="found-head" style={SECTION}>
    <SectionHead id="found-head" aside="six older ideas, 2015 to 2026">
      iii · the foundations beneath them
    </SectionHead>
    <Rubric style={{ maxWidth: 820 }}>
      These are smaller parts of the instrument, each borrowed whole. Follow a link to see one at work.
    </Rubric>
    <div style={{ display: "flex", flexWrap: "wrap", gap: "8px 44px", marginTop: 6 }}>
      <Foundation
        id="logit-lens"
        title="The logit lens"
        makers="nostalgebraist · LessWrong · 2020 · [8]"
        what="Read any layer as if it were the last, with no lens at all. The instrument calls this the naked eye and keeps it beside the lens wherever a neuron is read."
        found="The study: from layer 46 up the J++ lens, fitted on the base model, reads much like it and is no more faithful; a lens fitted on the instruct model does better there (R1). Within layers 41–58, where the two agree on a neuron, that agreement was a good free sign of a sound label, tested for J++ labels only (§3.1, §3.2)."
        links={
          <>
            <Go dest="neuron" size={16}>
              a neuron’s folio →
            </Go>
            <Ext url="https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens" go={16}>
              the source
            </Ext>
          </>
        }
      />
      <Foundation
        id="circuits"
        title="The transformer circuits framework"
        makers="Elhage, Nanda, Olsson and others · Anthropic · 2021 · [14]"
        what="Each layer adds what it has to say to one shared stream, and attention and the feed-forward block each write their own share. That is why the instrument can split every layer into gathering and remembering, and draw threads between neurons from their weights alone."
        found="The study: the split first ignored Gemma-4’s per-layer scaling, now fixed (V7); the threads are a weak guide, ρ ≈ 0.41 against 0.87 for a direct-path estimate (§3.4)."
        links={
          <>
            <Go dest="instrument" size={16}>
              the instrument →
            </Go>
            <Ext url="https://transformer-circuits.pub/2021/framework/index.html" go={16}>
              the source
            </Ext>
          </>
        }
      />
      <Foundation
        id="glu"
        title="GLU variants"
        makers="Shazeer · 2020 · [15]"
        what="Gemma’s feed-forward block multiplies a gate by a second reading of the stream. Because that product can come out positive or negative, every neuron has two faces."
        found="The study: in Gemma-4 the two faces are real, and their sign comes through the second reading, the up-projection (§3.2)."
        links={
          <>
            <Go dest="layer" size={16}>
              a layer’s leaf →
            </Go>
            <Ext url="https://arxiv.org/abs/2002.05202" go={16}>
              the source
            </Ext>
          </>
        }
      />
      <Foundation
        id="lrp"
        title="Relevance propagation, and the R-lens"
        makers="Bach et al. 2015 · [17]; Blank, Bhatia & Nanda 2026 · [5]"
        what="Relevance propagation shares out the credit for an answer among the parts that produced it. The R-lens adapted its rules to make the Jacobian lens truer in early layers, and J++ uses those rules."
        found="The study did not test these rules themselves. R1 tested a frozen-attention linearisation and cross-position terms, and neither explained J++’s disagreement with the instruct model; the likeliest remaining cause is the weights (the chat training and the 4-bit rounding, which could not be told apart), but J++’s own rules, cluster weighting and word-piece filter were never tested (R1 §2.4, §6)."
        links={
          <>
            <Go dest="lexicon-lens" size={16}>
              the lens, in the lexicon →
            </Go>
            <Ext url="https://doi.org/10.1371/journal.pone.0130140" go={16}>
              LRP
            </Ext>
            <Ext url="https://www.alignmentforum.org/posts/nv8oedrnLXKRzNEL9" go={16}>
              the R-lens
            </Ext>
          </>
        }
      />
      <Foundation
        id="cka"
        title="Linear CKA"
        makers="Kornblith, Norouzi, Lee & Hinton · 2019 · [13]"
        what="A measure, from 0 to 1, of how alike two layers arrange the same words. The instrument shows it as the kinship of layers, in the apparatus beneath the instrument."
        found="The study: neighbouring layers are close kin (0.96 in the demonstration sighting); the single start-mark fix lowers kinship by about 0.06 on average (E6, V7)."
        links={
          <>
            <Go dest="instrument-apparatus" size={16}>
              the instrument, apparatus open →
            </Go>
            <Ext url="https://arxiv.org/abs/1905.00414" go={16}>
              the source
            </Ext>
          </>
        }
      />
      <Foundation
        id="steering"
        title="Activation steering"
        makers="Turner et al. 2023 · [21]; Panickssery et al. 2024 · [22]"
        what="Add a chosen direction to the stream and watch what the model says next. The instrument calls this a nudge."
        found="The study: the code’s default nudge, at layers 30, 36, 42, 48 and 54, is 2.4 to 7.4 times the stream’s own strength and turns every answer to junk; it advises about 0.02 of the stream’s strength (§3.4, bug 1). The server keeps that default for old callers; the instrument now sends the mended recipe (the final-norm direction, the quiet dimensions masked, the start mark spared) at 0.01 of the stream, and a nudge masked that way read fluently at 0.005–0.01 and broke at 0.02."
        links={
          <>
            <Go dest="nudge" size={16}>
              nudge →
            </Go>
            <Ext url="https://arxiv.org/abs/2308.10248" go={16}>
              [21]
            </Ext>
            <Ext url="https://arxiv.org/abs/2312.06681" go={16}>
              [22]
            </Ext>
          </>
        }
      />
    </div>
  </section>
);

/* ------------------------------------------------- iv · earns its keep */

type Honoured = "not yet" | "in part" | "yes";

/** A filled lozenge: the instrument does this; half-filled: in part; hollow: not yet. */
const HonourMark: React.FC<{ state: Honoured }> = ({ state }) => (
  <svg width="12" height="14" viewBox="0 0 12 14" aria-hidden="true">
    {state === "yes" ? <path d="M6 1 L11 7 L6 13 L1 7 Z" fill="#A07A2E" /> : null}
    {state === "in part" ? <path d="M6 1 L6 13 L1 7 Z" fill="#A07A2E" /> : null}
    <path d="M6 1 L11 7 L6 13 L1 7 Z" fill="none" stroke="#A07A2E" strokeWidth="1.2" />
  </svg>
);

const KeepRow: React.FC<{ n: number; condition: React.ReactNode; state: Honoured; children: React.ReactNode }> = ({
  n,
  condition,
  state,
  children,
}) => (
  <div
    style={{
      display: "flex",
      flexWrap: "wrap",
      gap: "8px 36px",
      padding: "18px 0 20px",
      borderBottom: "1px solid var(--hair-soft)",
    }}
  >
    <div style={{ flex: "1 1 340px", minWidth: 0, display: "flex", gap: 14, alignItems: "baseline" }}>
      <span className="lining" style={{ fontSize: 30, lineHeight: 1, color: "var(--gilt-numeral)", minWidth: 22 }}>
        {n}
      </span>
      <span style={{ fontSize: 20 }}>{condition}</span>
    </div>
    <div
      style={{
        flex: "0 0 150px",
        display: "flex",
        alignItems: "center",
        gap: 8,
        fontSize: 15,
        fontVariantCaps: "all-small-caps",
        letterSpacing: "0.14em",
        color: "var(--muted)",
      }}
    >
      <HonourMark state={state} />
      {state}
    </div>
    <p style={{ flex: "999 1 420px", minWidth: 0, margin: 0, fontSize: 17 }}>{children}</p>
  </div>
);

const Keep: React.FC = () => (
  <section id="keep" aria-labelledby="keep-head" style={SECTION}>
    <SectionHead id="keep-head" aside="REPORT §3.3">
      iv · when the synthesis earns its keep
    </SectionHead>
    <p style={{ margin: 0, fontSize: 20, maxWidth: 860 }}>
      The study’s verdict on joining the pillars was “neat as a question, oversold as built.” It named four conditions
      under which the joined instrument earns its keep. Here is each, and where this instrument honours it, or should.
    </p>
    <Rubric style={{ maxWidth: 820 }}>
      A filled lozenge means the instrument already does this; a half-filled one, that it does so in part; a hollow
      one, that it does not yet.
    </Rubric>
    <div style={{ display: "flex", flexDirection: "column", marginTop: 6, borderTop: "1px solid var(--hair)" }}>
      <KeepRow
        n={1}
        state="not yet"
        condition="Labels grounded in what wakes a neuron: the passages that make each face fire, stored with it."
      >
        A neuron’s folio shows only what a neuron writes, because the instrument’s input-side labels carried just 6 to
        8% of the signal. Stored passages for each face, the study’s fix (§3.4, bug 7): <NotInInstrument />. Meanwhile,
        on a <To dest="neuron">neuron’s folio</To> you can put your own guess about what wakes it to the test.
      </KeepRow>
      <KeepRow n={2} state="in part" condition="Output labels trusted only from layer 41 upward.">
        The leaf on <To dest="instrument">the instrument</To> and <To dest="layer">a layer’s leaf</To> call labels below
        layer 41 rumours, not findings, and a <To dest="neuron">neuron’s folio</To> marks layers 41 to 58 as the band
        where labels held up. The labels are still shown below it.
      </KeepRow>
      <KeepRow n={3} state="in part" condition="A label counts only when the lens and the naked eye agree.">
        <To dest="layer">A layer’s leaf</To> and a <To dest="neuron">neuron’s folio</To> set the two readings side by
        side, say how far they agree, and tell you to doubt those that share nothing. In the study, within layers 41
        to 58, this agreement predicted a sound label (AUC 0.84, for J++ labels). Labels that disagree are still shown,
        not yet held back.
      </KeepRow>
      <KeepRow
        n={4}
        state="in part"
        condition="Each label spot-checked by cause and effect: turn the neuron up or down, and see."
      >
        By hand: from a neuron’s folio you can <To dest="nudge">nudge</To> it and watch what the model says. Nothing yet
        checks a label for you; the study did so for 384 faces (N1).
      </KeepRow>
    </div>
  </section>
);

/* ------------------------------------------- v · what Cammarata asks */

const Practice: React.FC<{
  numeral: string;
  asks: React.ReactNode;
  here: React.ReactNode;
  links: [Dest, string][];
}> = ({ numeral, asks, here, links }) => (
  <div
    style={{
      display: "flex",
      flexWrap: "wrap",
      gap: "8px 44px",
      padding: "18px 0 20px",
      borderBottom: "1px solid var(--hair-soft)",
    }}
  >
    <div style={{ flex: "1 1 380px", minWidth: 0, display: "flex", gap: 14, alignItems: "baseline" }}>
      <span
        style={{
          fontSize: 15,
          fontVariantCaps: "all-small-caps",
          letterSpacing: "0.14em",
          color: "var(--gilt-text)",
          minWidth: 28,
        }}
      >
        {numeral}
      </span>
      <span style={{ fontSize: 20 }}>{asks}</span>
    </div>
    <div
      style={{
        flex: "999 1 420px",
        minWidth: 0,
        display: "flex",
        flexWrap: "wrap",
        alignItems: "baseline",
        gap: "0 18px",
      }}
    >
      <span style={{ fontSize: 17 }}>{here}</span>
      {links.map(([dest, label]) => (
        <Go key={label} dest={dest}>
          {label}
        </Go>
      ))}
    </div>
  </div>
);

const Practices: React.FC = () => (
  <section id="practices" aria-labelledby="ask-head" style={{ ...SECTION, marginTop: 24 }}>
    <SectionHead id="ask-head" aside="and where this instrument helps">
      v · what Cammarata asks of a neuron reader
    </SectionHead>
    <p style={{ margin: 0, fontSize: 20, maxWidth: 860 }}>
      Cammarata’s essay gives practical advice to anyone who reads neurons one at a time. Here are seven of its
      practices, in our own words, each beside the place in this instrument where you can follow it.
    </p>
    <Rubric style={{ maxWidth: 820 }}>Where a practice has no screen yet, the line says so plainly.</Rubric>
    <div style={{ display: "flex", flexDirection: "column", marginTop: 6, borderTop: "1px solid var(--hair)" }}>
      <Practice
        numeral="i"
        asks="Keep a neuron whole. Do not split it into a tidier list of parts; what is left over is evidence."
        here="One neuron, one folio, both faces together."
        links={[["neuron", "a neuron’s folio →"]]}
      />
      <Practice
        numeral="ii"
        asks="Check the negative face. It shows which distinction the neuron is drawing."
        here="Read both faces on a neuron’s folio; on the rings, the hollow lozenges are negative faces."
        links={[
          ["neuron", "its two faces →"],
          ["instrument", "the instrument →"],
        ]}
      />
      <Practice
        numeral="iii"
        asks="Read long passages, not just the top word-pieces; the strongest word may not be the meaningful one."
        here={
          <>
            In part. Write whole sentences into a neuron’s guess test, and tick “read it in long contexts” in its field
            notes only once you have. Stored passages for each face, which the study recommends (§3.5):{" "}
            <NotInInstrument />
          </>
        }
        links={[["neuron", "its field notes →"]]}
      />
      <Practice
        numeral="iv"
        asks="Steer it, and make narrow predictions that could prove you wrong."
        here="Nudge one neuron and see what the model writes. On a neuron’s folio, put a guess to the test: sentences that should wake it against sentences that should not."
        links={[
          ["nudge", "nudge →"],
          ["neuron", "put a guess to the test →"],
        ]}
      />
      <Practice
        numeral="v"
        asks="Look upstream at what builds a neuron, and downstream at what reads it."
        here={
          <>
            Downstream: the threads on the rings, and who listens to it on a neuron’s folio. Upstream:{" "}
            <NotInInstrument />
          </>
        }
        links={[
          ["instrument", "threads →"],
          ["neuron", "who listens →"],
        ]}
      />
      <Practice
        numeral="vi"
        asks="Say plainly when you do not understand a neuron."
        here={
          <>
            A neuron’s field notes let you mark either face “not yet understood”, and its folio tells you to doubt a
            label when its two readings share nothing. The notes last for this visit only; keeping them with the vault:{" "}
            <NotInInstrument />
          </>
        }
        links={[["neuron", "field notes →"]]}
      />
      <Practice
        numeral="vii"
        asks="Suspect your tools before the network: when what you cannot explain keeps gathering in one place, look first at the instrument."
        here="The apparatus on every page states each reading’s known faults, the nudge explains why its old default broke, and the fourth section of this page lists what is still missing."
        links={[["nudge", "why the old nudge broke →"]]}
      />
    </div>
    <LinApparatus title="the practices, in the code" label="Apparatus for Cammarata’s practices" style={{ marginTop: 10 }}>
      <dt>both faces</dt>
      <dd>
        <M>inspect_pole</M> (MCP) and <M>POST /api/scientist/inspect_pole</M>;{" "}
        <M>{"DESCRIBE POLE LAYER <l> SLOT <i> POLARITY -"}</M> in the vault.
      </dd>
      <dt>hypothesis test</dt>
      <dd>
        <M>test_motif_hypothesis(layer, neuron_idx, pos_prompts, neg_prompts)</M>: records the signed <M>mlp_act</M> at
        the last word-piece and its peak over the sequence, but computes Cohen’s d on the last word-piece only, with
        population variance and no significance test, and says “CONFIRMED” at |d| ≥ 0.8 for any n (n = 3 gave d = −22;
        E6 §3, item 7: add a permutation p-value, a minimum n and multiple-comparison control). Also{" "}
        <M>POST /api/scientist/test_hypothesis</M>; on screen, a neuron’s folio, v. It tests when a neuron fires, not
        what it causes.
      </dd>
      <dt>downstream</dt>
      <dd>
        <M>trace_downstream_readers(layer, neuron_idx, prompt, max_layer_hop=6, top_k=10)</M>: C = ⟨σ<sub>i</sub> w
        <sub>down</sub>, w<sub>gate</sub>⟩ · |a| over the next six layers. A weak guide (ρ ≈ 0.41); the direct-path
        estimate in V7’s notes reaches 0.87 (§3.4 bug 4). No upstream counterpart exists in <M>backend/</M>.
      </dd>
      <dt>steer</dt>
      <dd>
        <M>SteeringLaserRequest</M> takes <M>pole_layer</M>, <M>pole_neuron</M>, <M>pole_polarity</M>; defaults{" "}
        <M>layers=[29, 35, 41, 47, 53]</M>, <M>scale=650.0</M> (server.py:104–105), the collapse regime of bug 1, still
        in the code.
      </dd>
      <dt>the essay</dt>
      <dd>
        practices paraphrased from [9]. The essay’s own list is longer. It also asks for a neuron to be read among its
        neighbours, which a neuron’s folio does in part (iv, the loudest neurons beside it), and watched as it forms
        across training checkpoints, which this instrument cannot do: [NOT YET IN THE INSTRUMENT].
      </dd>
      <dt>field notes</dt>
      <dd>
        a neuron’s folio, vii: per face, one job / two related jobs / not yet understood, after the essay’s preliminary
        Qwen3-32B audit (about 90% / 7% / 2–4% [9]), with ticks for the other pole, long contexts, steering and
        readers; kept for the visit only. Stored per-pole contexts: [CONTEXTS NOT YET COLLECTED].
      </dd>
    </LinApparatus>
  </section>
);

/* ---------------------------------------- vi · the lineage of these pages */

const CONVENTIONS: [string, React.ReactNode][] = [
  [
    "rubrics",
    <>
      The red headings and directions that scribes and early printers set apart in red ink, from the Latin{" "}
      <i>rubrica</i>, red ochre. Here, the red lines tell a newcomer what to do; switch them off once you know the way.
    </>,
  ],
  [
    "apparatus",
    "The notes at the foot of a critical edition, where the editor records variant readings and sources. Here, the formulas, raw values and sources, for scholars.",
  ],
  [
    "stemma",
    "The family tree philologists draw to show which copies of a text descend from which. Here, which ideas this instrument descends from.",
  ],
  ["cartouche", "The ornamental frame in which map-makers and instrument-makers engraved a title or a maker’s name."],
  ["folio", "A leaf of a book, numbered on one side. Each page of this instrument is a folio; this one is the eighth."],
  [
    "theodolite",
    "The surveyor’s instrument for measuring bearings and elevations across a landscape. This one takes them inside a language model.",
  ],
];

const Conventions: React.FC = () => (
  <section id="conventions" aria-labelledby="conv-head" style={{ ...SECTION, marginTop: 24 }}>
    <SectionHead id="conv-head" aside="the old conventions they borrow">
      vi · the lineage of these pages
    </SectionHead>
    <dl
      style={{
        margin: "6px 0 0",
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 360px), 1fr))",
        gap: "4px 44px",
      }}
    >
      {CONVENTIONS.map(([term, def]) => (
        <div key={term} style={{ padding: "14px 0 16px", borderTop: "1px solid var(--hair-soft)" }}>
          <dt style={{ fontSize: 22 }}>{term}</dt>
          <dd style={{ margin: "4px 0 0", fontSize: 17 }}>{def}</dd>
        </div>
      ))}
    </dl>
    <Rubric style={{ margin: "4px 0 0", maxWidth: 820 }}>
      To see every source, numbered as in the repository, switch on the apparatus at the top of the page.
    </Rubric>
  </section>
);

/* ------------------------------------------------------ vii · the sources */

const RefEntry: React.FC<{ r: Reference; current: boolean }> = ({ r, current }) => (
  <div
    role="listitem"
    id={refId(r.n)}
    aria-current={current ? "location" : undefined}
    style={{
      display: "grid",
      gridTemplateColumns: "44px minmax(0, 1fr)",
      gap: "0 12px",
      padding: "10px 0 4px",
      borderBottom: "1px solid var(--hair-faint)",
      scrollMarginTop: 16,
    }}
  >
    <span
      className="num"
      style={{ textAlign: "right", color: current ? "var(--rubric)" : "var(--gilt-text)" }}
    >
      [{r.n}]
    </span>
    <div style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 0 }}>
      <span>
        {r.who} <i>{r.title}</i> {r.venue}
      </span>
      <span style={{ color: "var(--muted)" }}>{r.use}</span>
      <span style={{ display: "flex", flexWrap: "wrap", gap: "0 20px" }}>
        <Ext url={r.url} go={15} style={{ overflowWrap: "anywhere" }}>
          {r.host}
        </Ext>
        {r.more.map((m) => (
          <Ext key={m.url} url={m.url} go={15}>
            {m.label}
          </Ext>
        ))}
      </span>
    </div>
  </div>
);

const RefGroup: React.FC<{ head: string; refs: Reference[]; anchor: string | undefined }> = ({ head, refs, anchor }) => (
  <>
    <h3
      style={{
        margin: "12px 0 0",
        fontSize: 14,
        fontWeight: 500,
        fontVariantCaps: "all-small-caps",
        letterSpacing: "0.2em",
        color: "var(--ink)",
      }}
    >
      {head}
    </h3>
    <div role="list" style={{ display: "flex", flexDirection: "column", borderTop: "1px solid var(--hair-soft)" }}>
      {refs.map((r) => (
        <RefEntry key={r.n} r={r} current={anchor === refId(r.n)} />
      ))}
    </div>
  </>
);

const Sources: React.FC<{ anchor: string | undefined }> = ({ anchor }) => (
  <IfApparatus>
    <section
      id="sources"
      aria-labelledby="src-head"
      className="apparatus"
      style={{ marginTop: 44, paddingTop: 14, gap: 12, scrollMarginTop: 16 }}
    >
      <span className="kicker">apparatus</span>
      <h2 className="h-sec" id="src-head">
        vii · the sources
      </h2>
      <p style={{ margin: 0, fontSize: 16, maxWidth: 900 }}>
        Numbered as in <M>REFERENCES.md</M>, sections A to C and G. Sections D to F (statistics, data and software,
        [29]–[41]) are listed there.
        {/* anchors for links to [29]–[41], which are listed only in REFERENCES.md: they land on this line */}
        {REFS_D_TO_F.map((n) => (
          <span key={n} id={refId(n)} style={{ scrollMarginTop: 16 }} />
        ))}
      </p>
      <RefGroup head="a · model, lenses and lens methods" refs={refsA} anchor={anchor} />
      <RefGroup head="b · interpretability methods and concepts" refs={refsB} anchor={anchor} />
      <RefGroup head="c · related work, acknowledged but not relied on" refs={refsC} anchor={anchor} />
      <RefGroup head="g · sampling: how the instrument writes, when it draws by chance" refs={refsG} anchor={anchor} />
    </section>
  </IfApparatus>
);

/* ------------------------------------------------------------------- page */

const Footer: React.FC = () => (
  <>
    Sources numbered as in REFERENCES.md · findings from experiments/REPORT.md §3.1–3.5 and the reports it names ·{" "}
    <ReadingThrough /> · layers counted 1 to 60 here; the report counts from 0 ·{" "}
    <To dest="lexicon">plain words for strange things</To>
  </>
);

export const Lineage: React.FC<{ route: Route }> = ({ route }) => {
  const anchor = route.name === "lineage" ? route.anchor : undefined;
  return (
    <Page route={route} folio="folio viii" footer={<Footer />}>
      <ApparatusLinkNotice anchor={anchor} />
      <Title />
      <PageApparatus />

      <section id="stemma" aria-labelledby="stemma-head" style={{ ...SECTION, padding: "30px 0 10px" }}>
        <SectionHead id="stemma-head" aside="read from the top down">
          i · the family tree
        </SectionHead>
        <Rubric style={{ maxWidth: 820 }}>
          A stemma is the family tree philologists draw to show which copy of a text descends from which. Read this one
          from the top: the older ideas the instrument rests on, the four it is chiefly built from, then the instrument
          and the study made with it.
        </Rubric>
        <Stemma />
      </section>

      <Pillars />
      <Foundations />
      <Keep />
      <Practices />
      <Conventions />
      <Sources anchor={anchor} />
    </Page>
  );
};

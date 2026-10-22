import React from "react";
import { Apparatus, Missing, Rubric } from "../../components/kit";
import { FNW_DEFAULT } from "../../store/useTheodoliteStore";
import { GATE_FIGURES, OVERLAP_RECORD, RECORDED_CHECKED } from "../../lib/labels";
import { Lede, LexLink, LexSection, M } from "./parts";

/*
 * v · where the instrument can mislead you. Eleven cautions, each a finding of the study told in plain
 * words, with what it means for the reader; the apparatus gives the measured values, where the code was
 * audited, and the proposed fix. All of it is recorded text from README.md, experiments/REPORT.md
 * (its caution on the lenses, §2.2, §3.1–3.5 and the errata), the V6/V7/V8 notes, the E6 tool-audit report,
 * N1 and R1, as the canvas sets it out, corrected where the errata qualify it. The second caution ("single")
 * is the one the red line beside each layer's words links to: it covers the lens's fit on the base model.
 * The line numbers under "where (as audited)" are the audit's, not today's.
 */

type Tone = "plain" | "unfixed" | "mended";

const STATUS: React.CSSProperties = {
  fontSize: 15,
  fontVariantCaps: "all-small-caps",
  letterSpacing: "0.18em",
};

const Status: React.FC<{ tone: Tone; children: React.ReactNode }> = ({ tone, children }) =>
  tone === "mended" ? (
    <span style={{ ...STATUS, display: "inline-flex", alignItems: "center", gap: 8, color: "var(--muted)" }}>
      <svg width="9" height="9" viewBox="0 0 10 10" aria-hidden="true">
        <path d="M5 0 L10 5 L5 10 L0 5 Z" fill="#A07A2E" />
      </svg>
      {children}
    </span>
  ) : (
    <span style={{ ...STATUS, color: tone === "unfixed" ? "var(--rubric)" : "var(--muted)" }}>{children}</span>
  );

const Caution: React.FC<{
  id: string;
  n: number;
  title: React.ReactNode;
  status: string;
  tone?: Tone;
  found: React.ReactNode;
  means: React.ReactNode;
  from: React.ReactNode;
  notes: [string, React.ReactNode][];
  last?: boolean;
}> = ({ id, n, title, status, tone = "plain", found, means, from, notes, last }) => (
  <li
    id={id}
    style={{
      display: "grid",
      gridTemplateColumns: "52px minmax(0, 1fr)",
      gap: "0 20px",
      padding: "26px 0 30px",
      borderTop: "1px solid var(--hair-soft)",
      borderBottom: last ? "1px solid var(--hair-soft)" : undefined,
    }}
  >
    <span
      aria-hidden="true"
      style={{ fontSize: 34, lineHeight: 1, color: "var(--gilt-numeral)", fontVariantNumeric: "lining-nums", paddingTop: 2 }}
    >
      {n}
    </span>
    <div style={{ display: "flex", flexDirection: "column", gap: 12, minWidth: 0 }}>
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: "4px 18px" }}>
        <h3 style={{ margin: 0, fontSize: 24, fontWeight: 500, lineHeight: 1.25 }}>
          <span className="visually-hidden">Caution {n}: </span>
          {title}
        </h3>
        <Status tone={tone}>{status}</Status>
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "14px 48px" }}>
        <div style={{ flex: "3 1 420px", minWidth: 0 }}>
          <h4 className="h-sub">what the study found</h4>
          <p style={{ margin: "6px 0 0", fontSize: 18 }}>{found}</p>
        </div>
        <div style={{ flex: "2 1 320px", minWidth: 0 }}>
          <h4 className="h-sub">what it means for you</h4>
          <p style={{ margin: "6px 0 0", fontSize: 18, fontStyle: "italic" }}>{means}</p>
        </div>
      </div>
      <p style={{ margin: 0, fontSize: 14, color: "var(--muted)" }}>{from}</p>
      <Apparatus title={`caution ${n}`} sources={false} style={{ gap: 6, fontVariantNumeric: "lining-nums tabular-nums" }}>
        <dl className="dl tight">
          {notes.map(([dt, dd]) => (
            <React.Fragment key={dt}>
              <dt>{dt}</dt>
              <dd>{dd}</dd>
            </React.Fragment>
          ))}
        </dl>
      </Apparatus>
    </div>
  </li>
);

export const Cautions: React.FC = () => (
  <LexSection id="mislead-head" title="v · where the instrument can mislead you">
    <Lede cap="E">
      very instrument has its errors, and a careful surveyor writes them in the margin. This one was tested hard by
      the study that came with it. Here is what that study found, in plain words, and what each finding means for what
      you read.
    </Lede>
    <Rubric style={{ maxWidth: 760 }}>Read this section before you trust a reading, and again before you report one.</Rubric>
    <ol style={{ listStyle: "none", margin: "8px 0 0", padding: 0 }}>
      <Caution
        id="labels"
        n={1}
        title="Neuron labels held up at layers 41–58, not at 11–31"
        status="a limit of the method"
        found={
          <>
            Between layers 41 and 58, a neuron’s words, read through the J++ lens (fitted on the base model) or the naked
            eye alike, predicted what it really does far above chance. At layers 11–31 they were no better than chance:
            their tidy look comes from the J++ lens itself, for random mixtures of neurons read out as clusters just as
            neat, and even pure noise yields neat-looking ones, such as “Elon”, “Tesla”, “aerospace”, “SpaceX”. Layers
            32–40 were not tested. The words said to make a neuron fire were nearly useless at every layer.
          </>
        }
        means={
          <>
            Outside layers 41–58, read a neuron’s words as a rumour. Inside them, trust a label more the more words the
            lens and the naked eye share; in the study, a large share was a good sign of a true label, but sharing just
            one was a weak sign. The pages now hold back a neuron’s words below layer 41, and where the lens and the
            naked eye share none of their first ten; the apparatus keeps them, with the reason. At layers 59 and 60 the
            words were never tested: layer 59 still holds back a label whose two readings share none of ten, and at
            layer 60, where the two readings are one, the words are shown.
          </>
        }
        from="From the README’s headline findings, the report’s §3.2 and the N1 neuron-label study."
        notes={[
          [
            "measured",
            <>
              384 poles (192 random neurons × 2 signs, at layers 11, 21, 31, 41, 46, 51, 56, 58) against causal steering
              and contexts from a 147k-token corpus. P@10 in the causal top 100: ≈ 0 at layers 11–31; 0.35–0.67 at layers
              41–58 for J++ labels, about the same for the naked eye’s (null ≤ 0.003; N1 T1). Within layers 41–58,
              the overlap of the lens’s and the naked eye’s top ten, used as a graded score, predicts a valid label, AUC
              0.84, for base-fitted J++ labels on random neurons (REPORT §3.2). Counted again from N1’s data
              (<M>N1C_gate_cut.json</M>, a re-analysis), {GATE_FIGURES.shareNothingSound} of the{" "}
              {GATE_FIGURES.shareNothing} faces in that band whose two readings shared none of ten were sound, against{" "}
              {GATE_FIGURES.shareSomeSound} of the {GATE_FIGURES.shareSome} that shared at least one (P@10 ≥ 0.3): as a
              yes/no sign the cut scores AUC {GATE_FIGURES.cutAuc}; faces sharing exactly one were sound{" "}
              {OVERLAP_RECORD[1][1]} times in {OVERLAP_RECORD[1][0]}, against {GATE_FIGURES.bandSound} in{" "}
              {GATE_FIGURES.band} for the band. Labels
              through a lens fitted on the instruct model were not tested. Tightness of a label’s top ten at layers 21 /
              31: neurons 0.090 / 0.087, random 512-neuron mixtures 0.080 / 0.090, isotropic noise 0.054 / 0.040 (N1 T5;
              erratum E44).
            </>,
          ],
          [
            "proposed fix",
            <>
              labels from activation contexts stored per pole; output labels only at layer 41 and above, gated by lens
              and naked-eye agreement, and causally spot-checked (REPORT §3.3; §3.5, recommendation 5). Poles [9];
              activation labelling [25].
            </>,
          ],
          [
            "fix, in part done",
            <>
              the layer and the agreement (conditions 2 and 3): every neuron label the server sends carries{" "}
              <M>label_check</M> (<M>shared_top10</M>, the word-piece ids both readings share in their first ten, fp32,
              softcapped, as N1 counted; <M>held_back</M>; <M>reasons</M>), and the replies carry the rule as{" "}
              <M>label_gate</M> (<M>backend/label_gate.py</M>). The pages hold a label back below layer 41 (index 40),
              and, through the base-fitted J++ lens below layer 60, where its readings share none of ten; the cut at
              none is the instrument’s, the study set none, and under any other lens the count is shown but not used
              (erratum E47). The words are still sent, and kept in the apparatus.{" "}
              {RECORDED_CHECKED ? (
                <>
                  Where a page has no ten-a-side count (an older server), only the layer holds a label back; the
                  recorded sighting and the vault’s recorded searches were checked ten a side once, from the weights (
                  <M>N1G_recorded_label_checks.json</M>).
                </>
              ) : (
                <>
                  Where a page has no ten-a-side count (the recorded sighting’s three words a side, an older server),
                  only the layer holds a label back: <Missing what="the recorded labels not yet checked ten a side" />.
                </>
              )}{" "}
              And condition 4 in part: a neuron’s folio puts one label at a time to the test when asked (
              <LexLink to="test">a test of a label</LexLink>, the MCP tool <M>causal_spot_check</M>), and says only how many
              of the label’s ten words are among the hundred whose odds rose most. Not done: activation contexts stored
              per pole, and a causal spot-check of every label.
            </>,
          ],
        ]}
      />
      <Caution
        id="single"
        n={2}
        title="The lens was fitted on the base model, and a single reading is not a finding"
        status="a limit of the lens"
        found={
          <>
            The lens the instrument reads through, J++, was fitted on the base model, google/gemma-4-31B in 16-bit:
            Gemma 4 as it was before it was trained to converse. You are reading the chat model, its weights stored in 4
            bits, and the study could not tell which of the two differences matters more. Read on its own, the lens is
            poor: up to layer 50 its top word matched the model’s real next word on ordinary prose at most about 5% of
            the time, and only 28% of the time at layer 59. A lens the study refitted on the chat model did better near
            the top, yet on prose it still matched at most 17% of the time up to layer 56, so the fault is not the base
            model’s alone. Read by comparison, J++ is good: given several versions of a question, it ranked them in the
            order of the model’s eventual answers from about layer 21, where the naked eye could not yet, though on a
            scale squeezed eight- to fourteen-fold. Even pure noise reads out as confident nonsense,
            such as “{"</h2>"}”.
          </>
        }
        means={
          <>
            Read the lens’s words as its view of this model’s stream, not as a report from the model. Trust the
            difference between two matched sightings, such as the same question with one word changed, more than the
            words of any one sighting, and most at layers 31–50. At layers 21–30 trust only the order of the conditions, and only
            where you can check it against what the model goes on to say; below layer 21 even that order did not hold up
            reliably. Confidence is not evidence.
          </>
        }
        from="From the report’s caution on the lenses, §3.1, §3.4 (bug 9) and errata E32 and E49, and the R1 lens report."
        notes={[
          [
            "the lens",
            <>
              J++ fitted on <M>google/gemma-4-31B</M> in bf16 over 64 WikiText-103 passages of 128 word-pieces, with the
              R-lens rules (the lens file’s provenance; REPORT §0), and applied to Gemma-4-31B-it in QAT W4A16, so chat
              training and quantisation cannot be told apart (R1 §6). Its pronoun direction meets the instruct model’s
              own average gradient at cos 0.07–0.15 at layers 21–40 and 0.26 / 0.33 at 41–50 / 51–59 (E3; R1 §2.1). The
              comparison lens, the Jacobian lens, was also fitted on the base model (861 passages) and is not loaded.
            </>,
          ],
          [
            "measured",
            <>
              prose KL 16–18 nats under J++; top-1 at most about 5% through layer 50 (5.1–5.2% at layers 46–47, E3 per
              layer; REPORT §3.1’s “≤ 3%” understates it, erratum E30) and 28% at layer 59; J++ ≈ the logit lens in KL at
              layers 46–59, and below it in top-1 from about layer 34 to 55 (E3). A lens refitted on the instruct model
              reaches only 9–17% on prose at layers 41–56 (R1 §4.2). J++ orders conditions as behaviour does at layers
              21–30 (r 0.88–0.99 across E1, E2 and E5) and compresses the scale 8–14×; at layers 11–20 the ordering held
              in E1 (r 0.92) and failed in E2 (r 0.31). No averaged lens the study tried was stable below layer 31 (R1
              split-half ≤ 0.63). Isotropic noise decodes to “{"</h2>"}” and “{"</em>"}” (REPORT §3.1).
            </>,
          ],
          [
            "proposed fix",
            <>
              show every readout against a matched control, mean-centred, with the concept’s span as the unit; never a
              raw single-condition readout as a finding (REPORT §3.5, recommendation 2). See{" "}
              <LexLink to="matched-control">a matched control</LexLink>. With the base-fitted J++, use layers 31–50 for
              condition contrasts and confirm them with an instruct-fitted readout (erratum E49); label every readout
              with the lens it came from (erratum E48).
            </>,
          ],
          [
            "fix, applied",
            <>
              the server names its lens in <M>GET /api/health</M> and every sweep reply (<M>"lens"</M>: id, fitted on,
              method, applied to, and a caution); the head and the foot of every page name it, and a red line beside
              each layer’s words says what the reading can bear there. R1’s refit is not loaded, so every reading still
              goes through the base-fitted J++.
            </>,
          ],
        ]}
      />
      <Caution
        id="late"
        n={3}
        title="Late layers read “she” too readily"
        status="a limit of the lens"
        found={
          <>
            At layers 51–59, six or seven of the nine AI personas the study tried read at or above a woman persona on the J++ lens’s
            pronoun scale. Under the Jacobian lens, which was also fitted on the base model, none did, and under a lens
            refitted on the instruct model at most one; so R1 traced this mostly to the J++ recipe itself, not to fitting
            on the base model. A smaller lean toward the woman persona remains under most readings, as much for a
            nameless AI as for the model left to its default. Across layers 31–59 of twelve prompts in which Gemma wrote
            about itself, the instrument showed “she” as its top word in 50 places and “he” in 2, yet the model itself
            chose “she” less than once in a hundred. R1 refitted the lens on the instruct model, along fixed directions
            at layers 31–59 and in full at only a few layers; the instrument still reads through the base-fitted J++, and
            names it at the head and foot of every page.
          </>
        }
        means={
          <>
            Do not read a late-layer pronoun as the model’s description of itself. Compare conditions at layers 31–50,
            against a matched control (below layer 31, read the second caution first), and treat J++ pronoun readings at
            layers 51–59 as unreliable.
          </>
        }
        from="From the README’s headline findings, the report’s summary, §3.1, §3.5 and errata E2 and E49, the E4 and E6 reports and the R1 lens report."
        notes={[
          [
            "measured",
            <>
              AI personas (of 9) at or above the woman persona at layers 51–59: J++ 6–7 (mean 1.04), the Jacobian lens
              0, R1’s directional refit 1 (mean 0.79), its full matrix 0 (0.67). The default sits at 0.96 under J++ and
              0.54–0.80 under the others, within 0.07 of the nameless AI under all; R1 puts the remaining lean down to
              the AI-versus-human offset, untested under its own readout (R1 §5). J++’s noun scale puts the default at
              0.33, R1’s at 0.68. Lens-free, self tokens there still carry up to 2.1× the woman–man gap of other
              assistant tokens (E4). J++ against the instruct model’s true average gradient: cos 0.15, 0.26, 0.33 at
              layers 31–40, 41–50, 51–59; R1’s directional refit reaches 0.85–0.95. E6: “she” the top word in 50 layer ×
              prompt cells and “he” in 2 (12 prompts, layers 31–59), while the real P(she) was 0.009.
            </>,
          ],
          [
            "proposed fix",
            <>
              with the base-fitted J++, use layers 31–50 for condition contrasts and confirm them with an instruct-fitted
              readout; at 21–30 trust only orderings checked against behaviour; flag pronoun readouts at layers 51–59
              unless an instruct-fitted readout agrees (REPORT §3.5, recommendation 3, as corrected by erratum E49; R1’s
              recommendations). R1 has refitted the lens on the instruct checkpoint: along 128 fixed directions at every
              layer, stable at 31–59, and as a full matrix at 11 layers, usable only at 41, 46, 51, 56 and 59. The
              instrument loads neither, and names the base-fitted J++ at the head and foot of every page.
            </>,
          ],
        ]}
      />
      <Caution
        id="quiet"
        n={4}
        title="The quiet carry channel"
        status="a property of Gemma 4"
        found={
          <>
            About 180 of the stream’s 5,376 numbers are among the loudest at layers 6–21, then fall almost silent, yet
            the model reads them back with up to 1,700 times the gain it gives the others. They ferry early content to
            late layers. At layers 36–59, every lens and every gradient puts nearly all its weight in the quietest tenth
            of the stream, most of it on these few, which is why nudges built straight from the lens break the model: a
            push of 1% there disturbed the next word about a hundred times more than a push of 30% elsewhere (10–11 nats
            against 0.1).
          </>
        }
        means={
          <>
            The stream’s size is a poor guide to what the model notices. Tiny changes in these numbers matter
            enormously, and a nudge must be built to avoid them.
          </>
        }
        from="From the README’s headline findings, the report’s §2.2 and the V6 notes."
        notes={[
          [
            "measured",
            <>
              181 dimensions quiet at every layer 43–58; norm weights on them up to 1,248 (input norm), 860
              (pre-feed-forward norm) and 512 (final norm), a read gain 260–1,700× the median dimension’s at layers 41–59
              (V6). Every covector puts 99–100% of its energy in the quietest tenth of dimensions, which holds the 181,
              at layers 36–59; the 181 alone hold 56–98% at layers 31 and 41 (V6). A 1% push into that tenth gives
              KL ≈ 10–11 nats; 30% in loud dimensions gives ≈ 0.1. For the pronoun direction, masking the quiet
              dimensions raises J++ ↔ gradient agreement at layers 41–59 from 0.26 / 0.33 to 0.84 / 0.86 (REPORT
              §2.2); for random word-pair directions it reaches only 0.62 / 0.73 against a ceiling of 0.99, so the
              lens’s error is not confined to them (R1 §2.2).
            </>,
          ],
          [
            "proposed fix",
            <>
              gain-normalise covectors or mask the quiet dimensions before steering, and scale relative to ‖h‖ (REPORT
              §2.2; §3.5, recommendation 4). Distinct from attention sinks [20] and massive activations [19]; RMSNorm
              [16]; covector and vector [24].
            </>,
          ],
        ]}
      />
      <Caution
        id="frame"
        n={5}
        title="The landmark frame depends on their order"
        status={
          FNW_DEFAULT
            ? "the order mended in the instrument; the weighting applied, not yet measured"
            : "the order mended in the instrument"
        }
        tone="mended"
        found={
          <>
            As first built, the instrument squares its frame one landmark at a time, so the first word absorbs whatever the three share.
            Reordering the same three words swung the bearing by a median of 58–99°, depending on the layer, and the
            frame catches only 2.4–3.9% of the stream.
          </>
        }
        means={
          <>
            The instrument now squares the frame all at once,{" "}
            {FNW_DEFAULT ? "on the rows the model weighs" : "on the same raw rows as before"}, so the order of the
            landmarks no longer changes the frame: its three axes are the same in any order. The order still says which
            axis sets the bearing’s 0°, which its 90° and which points up, so reordering relabels a bearing (swap the first
            two and it reads 90° − θ). Squared either way, compare a sighting only with sightings named with the same three
            words in the same order: squared in order, the recorded one or one you choose, another order is another frame;
            squared all at once, it is the same frame relabelled. To set a bearing beside V7’s frame figures, take it in
            the frame as first built; V7 took them with one start mark, as the instrument now sends. E6’s sightings and the
            recorded one were taken before the V7 fixes, their chat prompts with a doubled start mark, which by itself
            moved bearings by a median of 0.4–1.9° in V7, so beside them the frame as first built is needed but not
            enough. Either way, read a bearing as a sketch of direction, not a measurement.
          </>
        }
        from="From the README’s “Status and known issues”, the report’s §3.4 (bugs 3 and 6) and the V7 notes."
        notes={[
          [
            "where (as audited)",
            <>
              <M>backend/theodolite_math.py:387-438</M>, <M>backend/engine_runtime.py:773</M>
            </>,
          ],
          [
            "measured",
            <>
              median azimuth swing 58–99° across the six orders of (she, he, it), above 30° in 75–98% of cells; the frame
              captures 2.4–3.9% of ‖h̃‖ (V7). Squared all at once on the raw rows, the frame agreed with the sign of the
              lens’s she−he score in 79–89% of readings by stretch of layers, against 72–88% squared in order (
              <M>V7_tool_bugs_04_bos_monument_summary.json</M>, 18 prompts; the V7 notes print 74–86%, 79–85% and 75–89%);
              a frame on the weighted rows was not measured. A doubled start mark alone moved bearings by a median of
              0.4–1.9° (max 98°; V7, bug 6).
            </>,
          ],
          [
            "proposed fix",
            <>
              symmetric (Löwdin) orthonormalisation (W W<sup>⊤</sup>)<sup>−½</sup> W on fnw ⊙ W<sub>U</sub>, order-free
              to 1e-7 [33]; display the captured fraction; prefer contrastive axes such as fnw ⊙ (W<sub>she</sub> − W
              <sub>he</sub>) (REPORT §3.4, bug 3).
            </>,
          ],
          [
            "fix, applied",
            <>
              <M>frame_squaring="lowdin"</M>
              {FNW_DEFAULT ? (
                <>
                  {" "}
                  and <M>frame_fnw</M>, which the instrument sends by default (the weighted rows not yet measured)
                </>
              ) : (
                <>
                  , which the instrument sends by default, on the raw rows (<M>frame_fnw</M>, the weighted rows, is a choice
                  V7 did not measure)
                </>
              )}
              ; the server’s own default stays Gram–Schmidt on raw rows, so old callers get what they got before. A
              neuron-locked frame keeps Gram–Schmidt by default, its axes having fixed roles. <M>frame_per_layer</M> and{" "}
              <M>centre_prompts</M> are off unless you choose them. The fraction caught is sent at every layer (
              <M>proj_frac</M>), with the frame’s cosines and squareness for any three words. Contrastive axes are not built.
            </>,
          ],
        ]}
      />
      <Caution
        id="thread-score"
        n={6}
        title="Thread scores are weak"
        status="mended in the instrument"
        tone="mended"
        found={
          <>
            A thread’s score agreed only weakly with the effects the study measured by silencing neurons (
            <span style={{ fontVariantNumeric: "lining-nums" }}>ρ ≈ 0.4</span>); a sounder direct-path estimate reached
            0.87. The twenty highest scores all began at layer 1, where the real effects were tiny, a median change of
            1.2%.
          </>
        }
        means={
          <>
            Read a thread as a suggestion of where to look, not as a proven circuit. The instrument and a neuron’s folio
            now score threads by the direct-path estimate; if you switch them back to C, discount the strongest-looking
            threads from layer 1. Threads join the fifteen neurons listed at each layer, so by default they run between the
            biggest writers; V7 measured the direct-path score’s 0.87 only on threads among the fifteen ranked by |a|, and
            among the biggest writers it was not measured.
          </>
        }
        from="From the README’s “Status and known issues”, the report’s §3.4 (bug 4) and the V7 notes."
        notes={[
          [
            "where (as audited)",
            <>
              <M>backend/theodolite_math.py:333-335</M>, <M>backend/engine_runtime.py:729-736</M>
            </>,
          ],
          [
            "measured",
            <>
              900 poles zero-ablated over 8 prompts: ρ(|C|, effect) = +0.41 [0.38, 0.44]; the direct-path estimate
              +0.87, the true gradient +0.93; precision@6 of 0.24 against 0.54. E6’s −0.44 was a Simpson’s paradox [31]
              (V7, V8).
            </>,
          ],
          [
            "proposed fix",
            <>
              replace C with the direct-path estimate (<M>experiments/scripts/V7_tool_bugs_03_filaments.py</M>) and call
              it a direct-path linear estimate, not a verified live circuit (REPORT §3.4, bug 4).
            </>,
          ],
          [
            "fix, applied",
            <>
              <M>thread_score="direct"</M> (sweeps) and <M>score="direct"</M> (trace_downstream_readers) choose and rank
              threads by the direct-path estimate, and the instrument sends it by default; the server’s own default stays
              C, so old callers get what they got before. Each thread’s binary row carries the one score it
              was chosen by; the JSON’s first 60 threads (in source-layer order) carry C and the direct-path estimates side
              by side, and the pages rank all of them by score.
            </>,
          ],
        ]}
      />
      <Caution
        id="nudge-defaults"
        n={7}
        title="The old nudge defaults collapse the model"
        status="mended in the instrument"
        tone="mended"
        found={
          <>
            The server’s default nudge pushes at layers 30, 36, 42, 48 and 54 with a force of 650, which is 2.4 to 7.4
            times the stream’s own strength, in a direction that puts 79% of its weight on just 3 of the 5,376 numbers.
            In every prompt tested the answer dissolved into the same junk, and that direction collapsed the model even
            at a force of 2. The interface’s own default, 280, is in the same collapse.
          </>
        }
        means={
          <>
            If an answer turns into repeated word-pieces, the nudge has broken the model; it has revealed nothing. Never
            use the old defaults, and keep a nudge near a hundredth of the stream’s strength. The instrument now sends
            the mended recipe by default; the window that reads fluently is still narrow.
          </>
        }
        from="From the README’s “Status and known issues”, the report’s §3.4 (bug 1) and the V7 notes."
        notes={[
          [
            "where (as audited)",
            <>
              <M>backend/server.py:104-105, 166, 171-176, 186</M>; the interface’s 280 at{" "}
              <M>frontend/src/store/useTheodoliteStore.ts:265</M>
            </>,
          ],
          [
            "measured",
            <>
              median ‖h‖ 237 / 269 / 175 / 142 / 88 at layers 30 / 36 / 42 / 48 / 54, so 650 / ‖h‖ = 2.7 / 2.4 / 3.7 /
              4.6 / 7.4; 79% of the direction’s energy in the 3 dimensions with negative final-norm weight; at a force of
              2 (0.8–2.3% of ‖h‖) 6 of 6 prompts collapse, KL 15.6 nats (V7).
            </>,
          ],
          [
            "proposed fix",
            <>
              v = fnw ⊙ (W<sub>U</sub>[t<sub>+</sub>] − W<sub>U</sub>[t<sub>−</sub>]), with no inverse and no clamp;
              strength ≤ 0.01 of ‖h‖; spare the start mark; project out the quiet subspace. A quiet-masked direction at
              norm 2 steered fluently and collapsed at 6.5 (V7; REPORT §3.4, bug 1).
            </>,
          ],
          [
            "fix, applied",
            <>
              fields of the nudge request, all set by the nudge page: <M>direction="fnw"</M>, <M>mask_quiet</M> (the 181
              persistent quiet dimensions and each pushed layer’s quietest tenth), <M>spare_bos</M>, <M>strength</M> as a
              share of ‖h‖. The server’s own defaults stay as they were, for old callers. In a check when the fix was
              built (not part of the study), the she/he nudge read fluently at 0.005–0.01 of ‖h‖ and broke at 0.02; with
              the 181 alone masked it broke at 0.005.
            </>,
          ],
        ]}
      />
      <Caution
        id="template"
        n={8}
        title="No conversation template was applied"
        status="mended"
        tone="mended"
        found={
          <>
            As first built, what you type was fed to the model as bare text, not wrapped as a turn in a conversation,
            though this model was trained to converse. On bare prompts the study saw the answer degenerate in 2 of 6
            cases (“own own own …”).
          </>
        }
        means={
          <>
            An odd or repetitive answer may come from this rather than from the model itself. A reading of a bare
            sentence is not a reading inside a conversation.
          </>
        }
        from="From the README’s “Status and known issues”, the report’s §3.4 (bug 5) and the V7 notes."
        notes={[
          [
            "where (as audited)",
            <>
              <M>backend/server.py</M>: the prompt is sent raw; no <M>apply_chat_template</M> anywhere in{" "}
              <M>backend/</M>
            </>,
          ],
          [
            "measured",
            <>greedy continuations degenerate on 2 of 6 raw prompts; the same requests through the chat template are fluent (V7).</>,
          ],
          [
            "proposed fix",
            <>a chat-template option, on by default for instruct checkpoints, paired with the start-mark fix (REPORT §3.4, bug 5).</>,
          ],
          [
            "fix, applied",
            <>
              <M>chat</M>, <M>system</M> and <M>assistant_prefix</M> on sweeps, traces and guesses: the tokenizer’s chat
              template with thinking off, an optional system turn, then the begun answer, with one start mark; on by
              default in the instrument (“send it as a chat”).
            </>,
          ],
        ]}
      />
      <Caution
        id="think"
        n={9}
        title="“Thinking turned on” was an imitation"
        status="mended; bare text left out"
        tone="mended"
        found={
          <>
            As first built, to sight with thinking turned on, the instrument set the marker{" "}
            <code style={{ fontFamily: "var(--mono)", fontSize: 15 }}>{"<|think|>"}</code> in front of your bare text.
            That is not how Gemma 4’s thinking is switched on: its chat template sets the marker at the head of a system
            turn, and begins the model’s answer without the empty thought it writes otherwise. Bare text has no turn to
            think in.
          </>
        }
        means={
          <>
            Sent as a chat, the sighting with thinking turned on now uses the model’s own switch. Sent as bare text, it is
            no longer offered. A thinking sighting taken the old way (marked so in its question’s apparatus) is a rough
            imitation: a difference it shows may not be a difference in real thinking.
          </>
        }
        from="From the E6 tool-audit report, item 11."
        notes={[
          [
            "where (as audited)",
            <>
              <M>backend/server.py:464</M> then; <M>backend/server.py:897</M> now, which still prepends{" "}
              <M>{"<|think|>\\n"}</M> to bare text for callers that do not ask otherwise (E6 §3, item 11)
            </>,
          ],
          ["proposed fix", <Missing what="none recorded in the study" />],
          [
            "fix, applied",
            <>
              Sent as a chat, the thinking sighting uses the template’s own switch (<M>enable_thinking=True</M>:{" "}
              <M>{"<|think|>"}</M> at the head of a system turn, which the template opens when there is none, and the
              model’s turn begun with no empty thought). Sent as bare text, the instrument no longer offers it, since bare
              text has no turn to think in; the old imitation stays in the question’s apparatus, marked. The server keeps
              the imitation for callers that do not ask otherwise (<M>think_bare "prefix"</M>), and every reply names
              which was used (<M>think_protocol</M>).
            </>,
          ],
        ]}
      />
      <Caution
        id="split"
        n={10}
        title="Gathering and remembering were misdivided"
        status="mended"
        tone="mended"
        found={
          <>
            Before the V7 fix the instrument ignored Gemma 4’s shrinking of the stream at each layer, and judged the
            larger share wrongly at 12 of the 60 layers. The fix is now applied: the two writes are separated
            correctly. Their shares, though, are weighed after both pass through the J++ lens, fitted on the base model,
            and how a lens fitted on this model would weigh them was not measured.
          </>
        }
        means={
          <>
            Trust the division of each layer’s change into its two writes; read the shares as the lens’s weighing of
            them, not as a measured fact about the model. The study’s own tool-audit numbers were measured before the
            fix.
          </>
        }
        from="From the README’s “Status and known issues” and the V7 notes (bug 2)."
        notes={[
          [
            "where (as audited)",
            <>
              <M>backend/theodolite_math.py:155-156</M>; call site <M>backend/engine_runtime.py:660</M>
            </>,
          ],
          [
            "measured",
            <>
              the old feed-forward term met the true write at a median cosine of 0.49, below 0 at 6 layers; the larger
              share flipped at 12 of 60 layers; the omitted rescale was a median 1.23× the feed-forward write (V7).
            </>,
          ],
          [
            "fix, applied",
            <>
              <M>V7_module1_fix.patch</M>: Δ<sub>attn</sub> = s · (mid − pre), Δ<sub>ffn</sub> = post − s · mid, Δ
              <sub>rescale</sub> = (s − 1) · pre; cosine ≥ 0.9985 at all 60 layers (REPORT §3.4, bug 2). The patch
              checks the decomposition, not the weighting: the shares are ‖Δ<sub>ffn</sub> · J<sup>⊤</sup>‖ / (‖Δ
              <sub>attn</sub> · J<sup>⊤</sup>‖ + ‖Δ<sub>ffn</sub> · J<sup>⊤</sup>‖) through the base-fitted J++ (
              <M>backend/theodolite_math.py:186-197</M>), untested against an instruct-fitted lens.
            </>,
          ],
        ]}
      />
      <Caution
        id="bos"
        n={11}
        title="A doubled start mark"
        status="mended"
        tone="mended"
        last
        found={
          <>
            Conversation prompts once began with two start-of-text marks instead of one. The effect on answers was
            small, but it made layers look more alike: their kinship rose from 0.61 to 0.66 on average. The fix is now
            applied.
          </>
        }
        means={
          <>
            Nothing to do now; but sightings taken before the fix, and the study’s audit numbers, carry this small bias.
          </>
        }
        from="From the README’s “Status and known issues”, the report’s §3.4 (bug 6) and the E6 tool-audit report."
        notes={[
          [
            "where (as audited)",
            <>
              <M>backend/engine_runtime.py:568, 583</M>; <M>backend/mcp_scientist.py:94, 199</M>
            </>,
          ],
          [
            "measured",
            <>
              ids <M>[2, 2, 105, …]</M> in 12 of 12 chat prompts; next-token KL 0.010 [0.003, 0.019] nats; mean kinship
              0.608 → 0.663; thread-set Jaccard 0.78. The naive fix, <M>prepend_bos=False</M>, left raw text with no
              start mark at all (KL 5.2 nats) (V7).
            </>,
          ],
          [
            "fix, applied",
            <>
              <M>V7_bos_fix.patch</M>: skip the prepend only when the prompt already begins with <M>{"<bos>"}</M>{" "}
              (REPORT §3.4, bug 6).
            </>,
          ],
        ]}
      />
    </ol>
  </LexSection>
);

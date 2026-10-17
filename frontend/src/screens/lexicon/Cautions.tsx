import React from "react";
import { Apparatus, Missing, Rubric } from "../../components/kit";
import { Lede, LexLink, LexSection, M } from "./parts";

/*
 * v · where the instrument can mislead you. Eleven cautions, each a finding of the study told in plain
 * words, with what it means for the reader; the apparatus gives the measured values, where the code was
 * audited, and the proposed fix. All of it is recorded text from README.md, experiments/REPORT.md
 * (§2.2, §3.1–3.5), the V6/V7/V8 notes, the E6 tool-audit report, N1 and R1, as the canvas sets it out.
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
        title="Neuron labels hold only at layers 41–58"
        status="a limit of the method"
        found={
          <>
            Between layers 41 and 58, a neuron’s words predicted what it really does far above chance. At layers 11–31
            they were no better than chance: their tidy look comes from the lens itself, for random mixtures of neurons,
            and even pure noise, read out as clusters just as neat, such as “Elon”, “Tesla”, “aerospace”, “SpaceX”. The
            words said to make a neuron fire were nearly useless at every layer.
          </>
        }
        means={
          <>
            Outside layers 41–58, read a neuron’s words as a rumour. Inside them, trust a label more when the lens and
            the naked eye agree on it; in the study, that agreement was a good sign of a true label.
          </>
        }
        from="From the README’s headline findings, the report’s §3.2 and the N1 neuron-label study."
        notes={[
          [
            "measured",
            <>
              384 poles (192 random neurons × 2 signs, 8 layers) against causal steering and contexts from a 147k-token
              corpus. P@10 in the causal top 100: ≈ 0 at layers 11–31; 0.35–0.69 at layers 41–58 (null ≤ 0.003).
              Agreement of the lens’s and the naked eye’s top ten predicts a valid label, AUC 0.84 (REPORT §3.2).
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
        ]}
      />
      <Caution
        id="single"
        n={2}
        title="A single reading is not a finding"
        status="a limit of the lens"
        found={
          <>
            Read on its own, the lens is poor: up to layer 50 its top word matched the model’s real next word at most 3%
            of the time, and only 28% of the time at layer 59. Read by comparison, it is good: given several versions of
            a question, it ranked them in the order of the model’s eventual answers from about layer 21, fifteen to
            twenty layers before the naked eye could. Even pure noise reads out as confident nonsense, such as “
            {"</h2>"}”.
          </>
        }
        means={
          <>
            Trust the difference between two matched sightings, such as the same question with one word changed, more
            than the words of any one sighting. Confidence is not evidence.
          </>
        }
        from="From the report’s §3.1."
        notes={[
          [
            "measured",
            <>
              prose KL 16–18 nats under J++; top-1 ≤ 3% through layer 50 and 28% at layer 59; J++ ≈ the logit lens at
              layers 46–59. J++ orders conditions correctly from about layer 21 (r ≈ 0.9), but compresses the scale
              8–14×. Isotropic noise decodes to “{"</h2>"}” and “{"</em>"}” (REPORT §3.1).
            </>,
          ],
          [
            "proposed fix",
            <>
              show every readout against a matched control, mean-centred, with the concept’s span as the unit; never a
              raw single-condition readout as a finding (REPORT §3.5, recommendation 2). See{" "}
              <LexLink to="matched-control">a matched control</LexLink>.
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
            At layers 51–59, nearly every AI persona the study tried read at or above a woman persona on the lens’s
            pronoun scale, while the lens’s noun scale disagreed and nothing was specific to the model itself. Across
            layers 31–59 of twelve prompts in which Gemma wrote about itself, the instrument showed “she” as its top word
            in 50 places and “he” in 2, yet the model itself chose “she” less than once in a hundred. The shipped lens
            was fitted on the base model; a lens refitted on the instruct model repairs much of the late-layer fidelity
            (R1), but it is not yet the one the instrument uses.
          </>
        }
        means={
          <>
            Do not read a late-layer pronoun as the model’s description of itself. For comparisons, prefer layers 21–50,
            and treat pronoun readings at layers 51–59 as unreliable.
          </>
        }
        from="From the README’s headline findings, the report’s summary, §3.1 and §3.5, the E6 tool-audit report and the R1 lens report."
        notes={[
          [
            "measured",
            <>
              J++ against the instruct model’s true average gradient: cos 0.15, 0.26, 0.33 at layers 31–40, 41–50,
              51–59; a lens refitted on the instruct model reaches 0.85–0.95 (R1). E6: “she” the top word in 50 layer ×
              prompt cells and “he” in 2 (12 prompts, layers 31–59), while the real P(she) was 0.009.
            </>,
          ],
          [
            "proposed fix",
            <>
              prefer layers 21–50 for lens comparisons and flag pronoun readouts at layers 51–59; consider refitting J++
              on the instruct checkpoint (REPORT §3.5, recommendations 3 and 6).
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
            late layers. At layers 36–59, every lens and every gradient puts nearly all its weight there, which is why
            nudges built straight from the lens break the model: a push of 1% there disturbed the next word about a
            hundred times more than a push of 30% elsewhere (10–11 nats against 0.1).
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
              (V6). Every covector puts 99–100% of its energy there at layers 36–59. A 1% push there gives KL ≈ 10–11
              nats; 30% in loud dimensions gives ≈ 0.1. Masking them raises J++ ↔ gradient agreement at layers 41–59
              from 0.26 / 0.33 to 0.84 / 0.86 (REPORT §2.2).
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
        status="mended, as an option"
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
            Compare bearings only between sightings that used the same landmarks in the same order, or square the frame
            all at once on the landmarks page; either way, read them as a sketch of direction, not a measurement.
          </>
        }
        from="From the README’s “Status and known issues”, the report’s §3.4 (bug 3) and the V7 notes."
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
              captures 2.4–3.9% of ‖h̃‖ (V7).
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
              options of the request, off by default so the study’s readings stay comparable: <M>frame_squaring="lowdin"</M>,{" "}
              <M>frame_fnw</M>, <M>frame_per_layer</M>, <M>centre_prompts</M>; the fraction caught is sent at every layer (
              <M>proj_frac</M>), and the frame’s cosines and squareness for any three words. Contrastive axes are not built.
            </>,
          ],
        ]}
      />
      <Caution
        id="thread-score"
        n={6}
        title="Thread scores are weak"
        status="mended, as an option"
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
            Read a thread as a suggestion of where to look, not as a proven circuit, and discount the strongest-looking
            threads from layer 1; or score the threads by the direct-path estimate, on the instrument and on a neuron’s
            folio.
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
              every thread now carries the direct-path estimates beside C; <M>thread_score="direct"</M> (sweeps) and{" "}
              <M>score="direct"</M> (trace_downstream_readers) choose and rank by it; C stays the default.
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
        title="“Thinking turned on” is an imitation"
        status="mended for chats"
        tone="mended"
        found={
          <>
            To sight with thinking turned on, the instrument sets the marker{" "}
            <code style={{ fontFamily: "var(--mono)", fontSize: 15 }}>{"<|think|>"}</code> in front of your bare text.
            That is not how Gemma 4’s thinking is actually switched on.
          </>
        }
        means={
          <>
            Treat the thinking sighting as a rough imitation; a difference it shows may not be a difference in real
            thinking.
          </>
        }
        from="From the E6 tool-audit report, item 11."
        notes={[
          [
            "where (as audited)",
            <>
              <M>backend/server.py:464</M> prepends <M>{"<|think|>\\n"}</M> to the raw prompt (E6 §3, item 11)
            </>,
          ],
          ["proposed fix", <Missing what="none recorded in the study" />],
          [
            "fix, applied",
            <>
              sent as a chat, the thinking sighting uses the template’s own switch (<M>enable_thinking=True</M>); sent as
              bare text, it is still the imitation.
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
            larger share wrongly at 12 of the 60 layers. The fix is now applied.
          </>
        }
        means={<>Today’s split holds; but the study’s own tool-audit numbers were measured before the fix.</>}
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
              <sub>rescale</sub> = (s − 1) · pre; cosine ≥ 0.9985 at all 60 layers (REPORT §3.4, bug 2).
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

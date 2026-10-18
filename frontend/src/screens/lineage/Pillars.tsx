import React from "react";
import { Rubric, SectionHead } from "../../components/kit";
import { Ext, Go, LinApparatus, LinkRow, M, type Dest } from "./bits";

/**
 * ii · the four pillars. Every figure here is a finding of the study, quoted from the canvas, which took them from
 * experiments/REPORT.md (§0, §2.1, §3.1–3.5), R1_IT_LENS_REPORT, N1_REPORT and E6_tool_audit_REPORT §2.3.
 * Layers are counted 1–60 in the prose and by repo index (0–59) in the apparatus, as on the canvas.
 */

const P18: React.CSSProperties = { margin: 0, fontSize: 18 };

const Block: React.FC<{ head: string; children: React.ReactNode; ruled?: boolean }> = ({ head, children, ruled }) => (
  <div
    style={{
      display: "flex",
      flexDirection: "column",
      gap: 6,
      ...(ruled ? { paddingTop: 12, borderTop: "1px solid var(--hair-faint)" } : null),
    }}
  >
    <h4 className="h-sub">{head}</h4>
    {children}
  </div>
);

const Where: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <span style={{ fontSize: 15, color: "var(--muted)" }}>{children}</span>
);

const Pillar: React.FC<{
  id: string;
  headId: string;
  numeral: string;
  title: string;
  makers: React.ReactNode;
  about: React.ReactNode;
  rubric: React.ReactNode;
  lends: React.ReactNode;
  links: [Dest, string][];
  found: React.ReactNode;
  foundWhere: string;
  source: React.ReactNode;
  sourceMore: React.ReactNode;
  apparatusTitle: string;
  apparatusLabel: string;
  apparatus: React.ReactNode;
}> = (p) => (
  <article
    id={p.id}
    aria-labelledby={p.headId}
    style={{
      display: "flex",
      flexWrap: "wrap",
      gap: "22px 56px",
      padding: "30px 0 36px",
      borderTop: "1px solid var(--hair)",
      scrollMarginTop: 16,
    }}
  >
    <div style={{ flex: "1 1 360px", minWidth: 0, display: "flex", flexDirection: "column", gap: 12 }}>
      <span style={{ fontSize: 14, fontVariantCaps: "all-small-caps", letterSpacing: "0.2em", color: "var(--gilt-text)" }}>
        {p.numeral}
      </span>
      <h3 id={p.headId} style={{ margin: 0, fontSize: 38, fontWeight: 400, lineHeight: 1.1 }}>
        {p.title}
      </h3>
      <span style={{ fontSize: 16, color: "var(--muted)" }}>{p.makers}</span>
      <p style={{ margin: "4px 0 0", fontSize: 20, lineHeight: 1.55 }}>{p.about}</p>
      <Rubric style={{ maxWidth: "none" }}>{p.rubric}</Rubric>
    </div>
    <div style={{ flex: "999 1 560px", minWidth: 0, display: "flex", flexDirection: "column", gap: 20, paddingTop: 6 }}>
      <Block head="what it lends the instrument">
        <p style={P18}>{p.lends}</p>
        <LinkRow>
          {p.links.map(([dest, label]) => (
            <Go key={label} dest={dest}>
              {label}
            </Go>
          ))}
        </LinkRow>
      </Block>
      <Block head="what the study found in Gemma-4">
        <p style={P18}>{p.found}</p>
        <Where>{p.foundWhere}</Where>
      </Block>
      <Block head="the source" ruled>
        <p style={{ margin: 0, fontSize: 17 }}>{p.source}</p>
        <p style={{ margin: 0, fontSize: 16, color: "var(--muted)" }}>{p.sourceMore}</p>
      </Block>
    </div>
    <LinApparatus title={p.apparatusTitle} label={p.apparatusLabel} style={{ flex: "1 1 100%", minWidth: 0 }}>
      {p.apparatus}
    </LinApparatus>
  </article>
);

const WRAP: React.CSSProperties = { overflowWrap: "anywhere" };

/** The vault's statements, as the canvas prints them (backend/larql_vault.py, execute_lql). */
const LQL_STATEMENTS = `DESCRIBE | SHOW LAYERS | STATUS
SELECT * FROM EDGES NEAREST '<concept>' [WHERE layer = <l>] [LIMIT <k>]
SELECT * FROM POLES NEAREST '<concept>' [WHERE layer = <l>] [LIMIT <k>]
DESCRIBE POLE LAYER <l> SLOT <i> [POLARITY +|-]
UPDATE LAYER <l> SLOT <i> SET object = '<concept>' [STRENGTH <alpha>] [POLARITY +|-]
INSERT INTO LAYER <l> (subject, object) VALUES ('<s>', '<o>') [STRENGTH <alpha>]
DELETE FROM LAYER <l> WHERE slot = <i>
ROLLBACK`;

export const Pillars: React.FC = () => (
  <section
    id="pillars"
    aria-labelledby="pillars-head"
    style={{ display: "flex", flexDirection: "column", gap: 0, paddingTop: 40, scrollMarginTop: 16 }}
  >
    <div style={{ paddingBottom: 14 }}>
      <SectionHead id="pillars-head" aside="what each is, what it lends, what the study found">
        ii · the four pillars
      </SectionHead>
    </div>

    {/* ------------------------------------------------------------ pillar i */}
    <Pillar
      id="jacobian-lens"
      headId="p1-head"
      numeral="pillar i"
      title="The Jacobian lens and J-space"
      makers="Gurnee, Sofroniew, Pearce and thirteen others · Anthropic · 2026"
      about="Halfway up the model there are no words yet: each layer holds 5,376 numbers. The Jacobian lens asks how a small change at one layer would change the model’s last layer, averaged over many texts, and uses that answer to translate any layer into the last layer’s terms, where it can be read as words. Its authors call the space it reads J-space, and argue that what can be read there forms a workspace shared across the model."
      rubric="Every list of words headed “through the lens” is read with this idea."
      lends="The idea of reading every layer as words, and the formula beneath each layer’s list of words. The landmarks are taken in J-space: each layer is carried through the lens before its bearing is measured. The server’s default sighting, about the legs of the animal that spins webs, is the paper’s own example. The Jacobian lens itself sits on disk for comparison; the instrument reads through the J++ lens and does not load it."
      links={[
        ["instrument", "the instrument →"],
        ["layer", "a layer’s leaf →"],
        ["landmarks", "landmarks →"],
      ]}
      found={
        <>
          Good for comparing, poor for telling. The two Jacobian lenses here, both fitted on the base model, put
          conditions in the right order, but a single reading on its own (“J-space says <i>she</i>”) means little unless
          a comparison reading stands beside it. At the word “I” in the middle layers, the gender read there, through the
          lenses, the naked eye, a probe and a lens-free direction alike, is that of the role being played, not of the
          person meant: in quoted speech the two come apart, while the model’s own next-word guesses name the right
          person. As a map of what neurons write, these lenses add little over the naked eye, and only at layers 41 to 46.
        </>
      }
      foundWhere="REPORT §2.1, §3.1, §3.3"
      source={
        <>
          [2] Gurnee, W., Sofroniew, N., Pearce, A., Piotrowski, M., Kauvar, I., Chen, R., Soligo, A., Bogdan, P., Ong,
          E., Wang, R., Thompson, T. B., Abrahams, D., Kantamneni, S., Ameisen, E., Batson, J., Lindsey, J.{" "}
          <i>Verbalizable Representations Form a Global Workspace in Language Models.</i> Transformer Circuits Thread
          (Anthropic), 6 July 2026.{" "}
          <Ext url="https://transformer-circuits.pub/2026/workspace/index.html" style={WRAP}>
            transformer-circuits.pub/2026/workspace
          </Ext>
        </>
      }
      sourceMore={
        <>
          With its companion code, the <i>jacobian-lens</i> package [3],{" "}
          <Ext url="https://github.com/anthropics/jacobian-lens">github.com/anthropics/jacobian-lens</Ext>; and the
          pre-fitted lens files hosted by Neuronpedia [6],{" "}
          <Ext url="https://huggingface.co/neuronpedia/jacobian-lens" style={WRAP}>
            huggingface.co/neuronpedia/jacobian-lens
          </Ext>
          .
        </>
      }
      apparatusTitle="the Jacobian lens and J-space"
      apparatusLabel="Apparatus for pillar i, the Jacobian lens"
      apparatus={
        <>
          <dt>technical name</dt>
          <dd>
            Jacobian lens (J-lens; “JLens” in the code) and J-space, after Gurnee et al. 2026 [2], fitted with{" "}
            <i>jlens</i> [3].
          </dd>
          <dt>readout</dt>
          <dd>
            logits(h, ℓ) = unembed(h · J<sub>ℓ</sub>
            <sup>⊤</sup>), unembed(u) = 30 · tanh(W<sub>U</sub> · RMSNorm(u) / 30); at the last layer (index 59) J = I.
          </dd>
          <dt>estimator</dt>
          <dd>
            J<sub>ℓ</sub> sums cotangents over all target positions t′ ≥ t and averages over source positions t [2].
          </dd>
          <dt>the file</dt>
          <dd className="num">
            <span className="mono-wrap">gemma-4-31b/jlens/Salesforce-wikitext/gemma-4-31B_jacobian_lens.pt</span>:
            fitted by Neuronpedia on the base model <M>google/gemma-4-31B</M>, 861 WikiText-103 prompts of up to 128
            word-pieces [6] [7]; fp16, 5,376 × 5,376 per layer; snapshot <M>e166deb5…</M>
          </dd>
          <dt>in the server</dt>
          <dd>
            <M>TheodoliteRuntime(load_jlens=False)</M> (backend/server.py:76): not loaded; the lab daemon loads both
            lenses.
          </dd>
          <dt>values</dt>
          <dd className="num">
            For neuron labels, J++ beats the naked eye only at index 40–45 (layers 41–46), by +0.04 to +0.12 P@10, and is
            equal or slightly worse at 50–57; JLens ≥ J++ (§3.2; erratum E41). Pronoun-token J-space follows the role
            default, not the referent (E7, refuted as reference by V5; §2.1).
          </dd>
          <dt>caveat</dt>
          <dd>
            Fitted on the base bf16 model, applied to the instruct W4A16 checkpoint (§0; §3.4 bug 9). Read single
            conditions only against a matched control, mean-centred (§3.5, item 2).
          </dd>
          <dt>related</dt>
          <dd>
            Lin and Chanin, <i>Welcome to the J-Space</i>, Neuronpedia blog, 10 July 2026 [6]; Briggs 2026 [26];
            Panchal 2026 [27]; the tuned lens [28], not evaluated here.
          </dd>
        </>
      }
    />

    {/* ----------------------------------------------------------- pillar ii */}
    <Pillar
      id="jpp-lens"
      headId="p2-head"
      numeral="pillar ii"
      title="The J++ lens"
      makers="Ayonrinde & Lindsey · Anthropic · 2026 · after the Jacobian lens and the R-lens"
      about="J++ is a sharper grinding of the same lens. It trusts the measurements less where the lens is known to read poorly, traces them back with relevance rules borrowed from the R-lens, and leaves out word-pieces that mean nothing, such as stray punctuation. The instrument uses the lens itself but not that last step."
      rubric="When a page says “read through the lens”, this is the lens, as fitted on the base model."
      lends="It is the instrument’s main lens. The words that label the word’s path, every layer’s list “through the lens”, the bearings of the landmarks and what each neuron writes are all read through it, with the naked eye kept beside it for comparison. It is the fit made on the base model; the study’s refit on the instruct model is not loaded."
      links={[
        ["instrument", "the instrument →"],
        ["layer", "a layer’s leaf →"],
        ["neuron", "a neuron’s folio →"],
      ]}
      found="The shipped lens was fitted on the base model and is used here on the instruct model. It ranks one condition against another well from about layer 21, where the naked eye cannot yet, but it is poor at saying what the model will actually say: on ordinary prose its first guess matched the model’s at most about 5% of the time up to layer 50, and 28% at layer 59. From layer 46 up it reads much like the naked eye, and its first guess is right no more often. A lens refitted on the instruct model (R1) agrees far better with how the model itself responds in the late layers, and lifts first-guess agreement on the assistant’s words at layer 51 from 25% to 45%; the instrument does not load it. Below layer 31 no lens of this averaged kind proved stable, and none improved next-word agreement below layer 41."
      foundWhere="REPORT §0, §3.1; R1_IT_LENS_REPORT"
      source={
        <>
          [4] Ayonrinde, K. (Anthropic Fellows), Lindsey, J. (Anthropic).{" "}
          <i>J++ Lens: Jacobian Filtering Enables More Faithful Workspace Lenses.</i> LessWrong, 8 October 2026.{" "}
          <Ext url="https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr/" style={WRAP}>
            lesswrong.com/posts/nc9dHfcB22JdMzGbr
          </Ext>
        </>
      }
      sourceMore={
        <>
          Code: <Ext url="https://github.com/safety-research/jpp_lens">github.com/safety-research/jpp_lens</Ext>;
          lenses: <Ext url="https://huggingface.co/koayon/jpp-lenses">huggingface.co/koayon/jpp-lenses</Ext>. Built on
          the R-lens [5]: Blank, C., Bhatia, A., Nanda, N. <i>R-lens: Making J-lens More Faithful on Early Layers.</i>{" "}
          Alignment Forum, 5 August 2026,{" "}
          <Ext url="https://www.alignmentforum.org/posts/nv8oedrnLXKRzNEL9" style={WRAP}>
            alignmentforum.org/posts/nv8oedrnLXKRzNEL9
          </Ext>
          .
        </>
      }
      apparatusTitle="the J++ lens"
      apparatusLabel="Apparatus for pillar ii, the J++ lens"
      apparatus={
        <>
          <dt>technical name</dt>
          <dd>
            J++ (<M>jpp</M>), Jacobian filtering for workspace lenses [4].
          </dd>
          <dt>three changes</dt>
          <dd>
            down-weights Jacobians from activation clusters with poor readouts; computes Jacobians with an LRP backward
            pass, <M>lrp_mode="rlens"</M>: LN-rule on RMSNorm, identity and half rules on gated MLPs, plain gradients
            through attention [5] [17] [18]; removes non-semantic tokens from the readout, a step this instrument does
            not apply.
          </dd>
          <dt>the file</dt>
          <dd className="num">
            <span className="mono-wrap">gemma-4-31b/jpp/Salesforce-wikitext/gemma-4-31B_jpp_lens.pt</span>: provenance{" "}
            <M>hf_model_name: google/gemma-4-31B</M>, 64 × 128-token WikiText-103 sequences [7]; converted to fp16 by
            Neuronpedia [6]; J<sub>ℓ</sub> for index 0–58.
          </dd>
          <dt>fidelity</dt>
          <dd className="num">
            prose KL 16–18 nats; top-1 at most about 5% through index 49 (0.051–0.052 at 45–46, E3 per layer; §3.1’s
            “≤ 3%” understates it, erratum E30), 28% at 58; ≈ logit lens in KL at 45–58, below it in top-1 at about 33–54. Advantage over the
            logit lens on assistant turns: +9.1 nats (prose) falls to +3.1 at index 30–39 and turns negative at 40–49.
            Conditions ranked from ~index 20 (r ≈ 0.9), scale compressed 8–14×.
          </dd>
          <dt>geometry</dt>
          <dd className="num">
            cos with the true IT average gradient (gender direction) 0.15 / 0.26 / 0.33 at index 30–39 / 40–49 / 50–58;
            for the pronoun direction only, 0.84–0.86 at 40–58 with the quietest tenth of dimensions masked; the same masking
            gives 0.62 / 0.73 at 40–49 / 50–58 for 98 random token pairs (ceiling 0.99; R1 §2.2). Top-100 of 5,376
            directions hold 68–93% of the energy, a prior of the J++ fit: the IT average Jacobian puts 10–17% there (R1
            §3.2). Isotropic noise decodes to{" "}
            <M>{"</h2>"}</M>, <M>{"</em>"}</M>.
          </dd>
          <dt>R1 refit</dt>
          <dd className="num">
            exact same-position VJPs on the instruct model, in two forms. Directional (128 fixed directions, all
            indices): split-half 0.96–0.99 at 30–58, ≤ 0.63 below; agreement with held-out IT average gradients
            0.85–0.95. Full matrix at 11 indices (10, 15 … 55, 58), noise-dominated below 40: top-1 on assistant tokens
            at index 50: 0.25 → 0.45. Neither improves next-token fidelity below index 40. Likeliest remaining cause of
            J++’s disagreement: the weights (base bf16 vs instruct W4A16-QAT; fine-tuning and quantisation cannot be
            separated), after ruling out the fitting data and the method variants tested; J++’s published R-lens rules,
            cluster weighting and word-piece filter were not tested (R1 §2.4, §6). The instrument loads neither form.
          </dd>
          <dt>recommended</dt>
          <dd>
            with the base-fitted J++, use layers 31–50 (index 30–49) for condition contrasts and confirm them with an
            instruct-fitted readout; at 21–30 trust only orderings checked against behaviour (r 0.88–0.99 in E1, E2 and
            E5), and below 21 even orderings did not hold reliably (index 10–19: r 0.31 in E2, 0.92 in E1); flag
            pronoun readouts at layers 51–59 unless an instruct-fitted readout agrees (§3.5, item 3, as corrected by
            erratum E49). R1 has since refitted the lens: directional at layers 31–59, a full matrix usable at only five
            layers (41, 46, 51, 56 and 59); not yet loaded, so the instrument names the base-fitted J++ at the head and
            foot of every page.
          </dd>
        </>
      }
    />

    {/* ---------------------------------------------------------- pillar iii */}
    <Pillar
      id="poles"
      headId="p3-head"
      numeral="pillar iii"
      title="Neurons with two faces"
      makers="Cammarata · Goodfire · 2026 · beside it, Bills and colleagues · OpenAI · 2023"
      about={
        <>
          Each of a layer’s 21,504 neurons can fire two ways, strongly positive or strongly negative, and each way is a
          face of its own; the specialist calls the two faces its poles. In <i>Neural Networks Are Already Clean</i>,
          Cammarata argues that a network’s own neurons, read face by face, are already tidy building blocks, and that
          most of their apparent muddle is ordinary generalisation repeated at many scales. The earlier way, from Bills
          and colleagues, names a neuron by the passages that make it fire.
        </>
      }
      rubric="On the instrument, a filled lozenge is a neuron’s positive face and a hollow one its negative face."
      lends="Its unit of reading. Every neuron is shown with its two faces: as lozenges on the rings, in a layer’s list of the neurons that fired, and on a neuron’s own folio, which sets its two faces side by side and whose field notes take one face at a time. The vault finds neurons face by face."
      links={[
        ["neuron", "a neuron’s folio →"],
        ["layer", "a layer’s leaf →"],
        ["instrument", "the instrument →"],
      ]}
      found="The two faces are real: which way a neuron fires is set by the second of its two readings, the up-projection, and the “double-negative” case the essay describes carries under 1% of strong firings. But the two faces wake on unrelated inputs, while what they write is a mirror image at layers 41 to 58, so one label cannot describe both. Labels of what a neuron writes, read through the base-fitted J++ lens (the naked eye did as well), predicted its real effect at the layers tested from 41 to 58; at layers 11 to 31 they were no better than chance, and their tidiness was the J++ lens’s own habit, since random mixtures decode just as neatly. In one reader’s audit of 48 faces, 46% had one clear job, against the essay’s preliminary figure of about 90% for Qwen3-32B; the methods differ too much for a direct comparison."
      foundWhere="REPORT §3.2, §3.3; N1_REPORT"
      source={
        <>
          [9] Cammarata, N. <i>Neural Networks Are Already Clean.</i> Goodfire, 15 October 2026.{" "}
          <Ext url="https://already-clean.goodfire.com/">already-clean.goodfire.com</Ext> (index:{" "}
          <Ext url="https://www.goodfire.com/research/already-clean" style={WRAP}>
            goodfire.com/research/already-clean
          </Ext>
          )
        </>
      }
      sourceMore={
        <>
          The earlier way [25]: Bills, S., Cammarata, N., Mossing, D., et al.{" "}
          <i>Language models can explain neurons in language models.</i> OpenAI, 9 May 2023.{" "}
          <Ext url="https://openai.com/index/language-models-can-explain-neurons-in-language-models/" style={WRAP}>
            openai.com/index/language-models-can-explain-neurons-in-language-models
          </Ext>
        </>
      }
      apparatusTitle="neuron poles"
      apparatusLabel="Apparatus for pillar iii, neuron poles"
      apparatus={
        <>
          <dt>technical name</dt>
          <dd>bipolar GeGLU poles, after Cammarata 2026 [9], in the GeGLU block of [15]; output labels “J-VINDEX”.</dd>
          <dt>formula</dt>
          <dd>
            a = GeLU<sub>tanh</sub>(x W<sub>gate</sub>) ⊙ (x W<sub>up</sub>) at <M>mlp_act</M>; the 15 strongest by |a|,
            σ = sign(a); label = unembed((σ · w<sub>down</sub>) · J<sup>⊤</sup>), against the naked eye’s unembed(σ · w
            <sub>down</sub>). 21,504 neurons, 43,008 poles per layer.
          </dd>
          <dt>N1 design</dt>
          <dd className="num">
            384 poles (192 random neurons × 2 signs, 8 layers); ground truth from single-neuron causal steering and
            activation contexts over a 147,534-position corpus (not distributed).
          </dd>
          <dt>values</dt>
          <dd className="num">
            P@10 in the causal top-100: ≈0 at index 10–30, 0.35–0.67 for J-VINDEX at 40–57, naive and JLens about the
            same (null ≤ 0.003; N1 T1); 31–39 not sampled; labels through an IT-fitted lens untested; J++ over naive +0.04 to
            +0.12 at 40–45, equal or slightly worse at 50–57 (erratum E41); agreement of the J-VINDEX and naive top-10
            predicts validity, AUC 0.84 within 40–57. Gate open in 72–100% of strong events; double-negative &lt; 1% of
            strong-event mass (13–33% of top-1% mass in sparse layers). Cleanliness, single rater, 48 poles: 46% one clear job (62% strong, 29% weak).
          </dd>
          <dt>caveat</dt>
          <dd className="num">
            input-side <M>gate_trigger_tokens</M> carry ~6–8% of the reproducible signal, and give one label for two
            poles (§3.4 bug 7); pole magnitudes are not meaningful and the top 15 explain only 10–41% of the FFN write
            at index 20–58 (bug 8).
          </dd>
          <dt>recommended</dt>
          <dd>
            activation contexts stored per pole; output labels only at layers 41 and up (index ≥ 40), gated by
            J++/naive agreement (§3.5, item 5). That gate was validated for base-fitted J++ labels only and must be
            re-validated for any new lens; spot-checks should steer in the linear regime (k ≈ 10), since 60× saturates
            the post-FFN norm at index 20–50 (erratum E47).
          </dd>
          <dt>reports</dt>
          <dd>
            <M>N1_REPORT.md</M>, <M>N1V_code_audit.md</M>, <M>N1V_robustness.md</M>; Bills et al. [25] is the
            activation-labelling approach behind N1B.
          </dd>
        </>
      }
    />

    {/* ----------------------------------------------------------- pillar iv */}
    <Pillar
      id="larql"
      headId="p4-head"
      numeral="pillar iv"
      title="LARQL and VINDEX"
      makers="C. Hayuk (chrishayuk) · Apache-2.0"
      about="LARQL treats a model’s weights as a catalogue you can question, in a small query language over an index format called VINDEX. You ask which neurons lean toward a word and a list comes back, and you can lay a correction over the weights and lift it off again."
      rubric="The vault borrows these names. What runs underneath is the instrument’s own."
      lends="The vault’s names and the shape of its questions: find neurons by their words, describe one, and change one neuron as an edit saved as a .patch, with a way back."
      links={[["vault", "the vault →"]]}
      found={
        <>
          The vault does not run LARQL: it asks the LARQL programs only for their version number and answers every
          question in its own Python. Its edit, named after MEMIT, is a nudge to a single neuron’s column, not MEMIT. A
          list of nearest neurons is a projection, not an experiment, as LARQL’s own README warns. Even so, the “she”
          neurons it found were partly real: 35 of 102 were selective; deleting the top ten at layer 51 clearly
          weakened the model’s pull toward <i>she</i> after a woman’s name, where ten random neurons did nothing; neuron
          5313 at layer 51 alone swings <i>she</i> against <i>he</i> strongly; neuron 16757 at layer 41, listed as a
          “he” neuron, in fact pushes toward <i>she</i>; and the picks at layer 31 had only tiny effects on gender, at most a sixtieth
          of neuron 5313’s, measured with a push too strong to be reliable at that depth.
        </>
      }
      foundWhere="REPORT §3.3 and erratum E46; E6_tool_audit_REPORT §2.3; N1A"
      source={
        <>
          [10] Hayuk, C. (chrishayuk). <i>LARQL: reference implementation of VINDEX3.</i> Apache-2.0.{" "}
          <Ext url="https://github.com/chrishayuk/larql">github.com/chrishayuk/larql</Ext>
        </>
      }
      sourceMore={
        <>
          Its LQL guide:{" "}
          <Ext url="https://github.com/chrishayuk/larql/blob/main/docs/lql-guide.md" style={WRAP}>
            github.com/chrishayuk/larql/blob/main/docs/lql-guide.md
          </Ext>
          . Named only, never used: MEMIT [11], <Ext url="https://arxiv.org/abs/2210.07229">arxiv.org/abs/2210.07229</Ext>
          .
        </>
      }
      apparatusTitle="LARQL and VINDEX"
      apparatusLabel="Apparatus for pillar iv, LARQL and VINDEX"
      apparatus={
        <>
          <dt>technical name</dt>
          <dd>
            a LARQL-style vault (“J-VINDEX3”), <M>backend/larql_vault.py</M>; naming after [10]; the dialect is this
            project’s own, not LQL.
          </dd>
          <dt>statements</dt>
          <dd>
            <code
              style={{
                display: "block",
                fontFamily: "var(--mono)",
                fontSize: 13,
                lineHeight: 1.65,
                whiteSpace: "pre-wrap",
                overflowWrap: "anywhere",
              }}
            >
              {LQL_STATEMENTS}
            </code>
            <span style={{ color: "var(--muted)" }}>Layers are counted from 0 in these statements.</span>
          </dd>
          <dt>reached by</dt>
          <dd>
            <M>POST /api/larql</M>; the MCP tool <M>query_larql_vault(lql_query)</M> in <M>backend/mcp_scientist.py</M>.
          </dd>
          <dt>LARQL called</dt>
          <dd>
            <M>larql-cli --version</M> and <M>vindex-cli --version</M> only (<M>cli_versions</M>, larql_vault.py:75–87).
          </dd>
          <dt>the edit</dt>
          <dd>
            a single-neuron rank-1 <M>down_proj</M> column overlay along a base-model J++ pullback, with no covariance
            solve: not MEMIT [11] (E6 §3, item 12; REPORT §3.3).
          </dd>
          <dt>values</dt>
          <dd className="num">
            102 neurons (top-10 <M>NEAREST 'she'</M> ∪ <M>'he'</M> at index 20, 30, 40, 50, 55, 58), 12 matched female-
            vs male-name contexts, paired sign-flip permutation test: 35/102 selective at BH q &lt; 0.05 (30 with |d| ≥
            0.8). DELETE of the top-10 “she” neurons at index 50, rolled back: female contexts −1.80 log-odds (baseline
            7.26), male +0.90, human-Gemma −2.28, Gemma-self −0.33; ten random neurons |Δ| ≤ 0.04. Index 50 n5313 moves
            she/he by +13.8 nats (it is in the “he” list too); index 40 n16757, listed under “he”, causally pushes{" "}
            <i>she</i> (E6 §2.3; REPORT §3.3). The index-30 picks have only tiny effects, she−he ≤ 0.22 nats: the “she” pick n12173
            moves it +0.16 and +0.22 (92nd and 100th percentile against random neurons), with “ feminine” and “girl” in its causal
            top 10, measured with 60× steering, outside the linear regime at that depth (N1A; erratum E46).
          </dd>
          <dt>caveat</dt>
          <dd>
            <M>EDGES NEAREST</M> is a projected contribution, which LARQL’s README warns is not a counterfactual (REPORT
            §3.3). <M>EDGES</M> and <M>POLES</M> run the same query, and the reported cosines are 0.0004–0.02 at index
            30–58, at or below a random vector’s ≈ 0.014 (E6 §3, item 6). Prove a neighbour with a nudge.
          </dd>
        </>
      }
    />
  </section>
);

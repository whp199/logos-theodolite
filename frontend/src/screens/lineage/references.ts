/**
 * The reference list of the Lineage folio, sections A to C of the repository's REFERENCES.md.
 *
 * Transcribed from the approved canvas (Lineage.dc.html, renderVals()), whose entries were in turn transcribed
 * from REFERENCES.md: numbers, authors, titles, venues and links are those of that file (links checked there on
 * 2026-10-16). The "use" notes are short paraphrases of that file's own notes, in the pages' plain words.
 * Sections D to F ([29]–[41]: statistics, data and software) are not listed on the page; they are in REFERENCES.md.
 */

export interface Reference {
  /** the bracketed number used in REFERENCES.md and throughout the reports */
  n: number;
  who: string;
  title: string;
  venue: string;
  url: string;
  /** the link text: the URL without its scheme or trailing slash */
  host: string;
  /** how this repository uses the source */
  use: string;
  more: { label: string; url: string }[];
}

/** The anchor id of a reference on this page, for `href.lineage(refId(n))`. */
export const refId = (n: number) => `ref-${n}`;

const R = (
  n: number,
  who: string,
  title: string,
  venue: string,
  url: string,
  use: string,
  more: [string, string][] = [],
): Reference => ({
  n,
  who,
  title,
  venue,
  url,
  host: url.replace("https://", "").replace(/\/$/, ""),
  use,
  more: more.map(([label, u]) => ({ label, url: u })),
});

/** A · model, lenses and lens methods */
export const refsA: Reference[] = [
  R(
    1,
    "Gemma Team (Google DeepMind).",
    "Gemma 4 Technical Report.",
    "arXiv:2607.02770, 2026.",
    "https://arxiv.org/abs/2607.02770",
    "The model under study: 60 layers, local attention over the last 1,024 word-pieces, every sixth layer global. Gemma is a trademark of Google LLC.",
    [
      ["the checkpoint", "https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct"],
      ["licence", "https://ai.google.dev/gemma/docs/gemma_4_license"],
    ],
  ),
  R(
    2,
    "Gurnee, W., Sofroniew, N., Pearce, A., Piotrowski, M., Kauvar, I., Chen, R., Soligo, A., Bogdan, P., Ong, E., Wang, R., Thompson, T. B., Abrahams, D., Kantamneni, S., Ameisen, E., Batson, J., Lindsey, J.",
    "Verbalizable Representations Form a Global Workspace in Language Models.",
    "Transformer Circuits Thread (Anthropic), 6 July 2026.",
    "https://transformer-circuits.pub/2026/workspace/index.html",
    "The Jacobian lens and J-space; the readout unembed(h · Jᵀ), lens-vector steering and the default two-hop prompt about the animal that spins webs.",
  ),
  R(
    3,
    "Anthropic.",
    "jacobian-lens (Python package jlens): companion code for [2].",
    "Apache-2.0.",
    "https://github.com/anthropics/jacobian-lens",
    "Used to fit the Neuronpedia Jacobian-lens file; the reference recipe R1 compares against.",
  ),
  R(
    4,
    "Ayonrinde, K. (Anthropic Fellows), Lindsey, J. (Anthropic).",
    "J++ Lens: Jacobian Filtering Enables More Faithful Workspace Lenses.",
    "LessWrong, 8 October 2026.",
    "https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr/",
    "The instrument’s main lens. The gemma-4-31b J++ lens was fitted on the base model and is applied here to the instruct model; only the lens matrix is applied, not the readout token filter.",
    [
      ["code", "https://github.com/safety-research/jpp_lens"],
      ["lenses", "https://huggingface.co/koayon/jpp-lenses"],
    ],
  ),
  R(
    5,
    "Blank, C., Bhatia, A., Nanda, N.",
    "R-lens: Making J-lens More Faithful on Early Layers.",
    "Alignment Forum / LessWrong, 5 August 2026.",
    "https://www.alignmentforum.org/posts/nv8oedrnLXKRzNEL9",
    "The LRP rules J++ uses: LN-rule on RMSNorm, identity and half rules on gated MLPs, plain gradients through attention (as described in R1 §2.4, which did not test them).",
  ),
  R(
    6,
    "Neuronpedia.",
    "jacobian-lens: pre-fitted Jacobian and J++ lenses.",
    "Hugging Face; tagged apache-2.0.",
    "https://huggingface.co/neuronpedia/jacobian-lens",
    "Both lens files (snapshot e166deb5…), downloaded at run time and not redistributed. Related: Lin, J., Chanin, D., Welcome to the J-Space, Neuronpedia blog, 10 July 2026.",
    [["the blog post", "https://www.neuronpedia.org/blog/jacobian-lens"]],
  ),
  R(
    7,
    "Merity, S., Xiong, C., Bradbury, J., Socher, R.",
    "Pointer Sentinel Mixture Models.",
    "ICLR 2017. arXiv:1609.07843.",
    "https://arxiv.org/abs/1609.07843",
    "Introduces WikiText-103, the corpus both lenses were fitted on.",
  ),
];

/** B · interpretability methods and concepts */
export const refsB: Reference[] = [
  R(
    8,
    "nostalgebraist.",
    "interpreting GPT: the logit lens.",
    "LessWrong, 31 August 2020.",
    "https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens",
    "The naked eye: the baseline unembed(h) throughout E1–E7, N1 and R1, and the naive pole labels.",
  ),
  R(
    9,
    "Cammarata, N.",
    "Neural Networks Are Already Clean.",
    "Goodfire, 15 October 2026.",
    "https://already-clean.goodfire.com/",
    "Neurons read as positive and negative poles; the double-negative regime of gated MLPs; the preliminary Qwen3-32B audit N1 is compared against.",
    [["index", "https://www.goodfire.com/research/already-clean"]],
  ),
  R(
    10,
    "Hayuk, C. (chrishayuk).",
    "LARQL: reference implementation of VINDEX3.",
    "Apache-2.0.",
    "https://github.com/chrishayuk/larql",
    "Inspiration for the names of the vault and its query dialect; larql_vault.py calls LARQL only for --version.",
    [["LQL guide", "https://github.com/chrishayuk/larql/blob/main/docs/lql-guide.md"]],
  ),
  R(
    11,
    "Meng, K., Sen Sharma, A., Andonian, A., Belinkov, Y., Bau, D.",
    "Mass-Editing Memory in a Transformer (MEMIT).",
    "ICLR 2023. arXiv:2210.07229.",
    "https://arxiv.org/abs/2210.07229",
    "Named only: the vault’s edit is a single-neuron down_proj column overwrite with no covariance solve, not MEMIT.",
  ),
  R(
    12,
    "Meng, K., Bau, D., Andonian, A., Belinkov, Y.",
    "Locating and Editing Factual Associations in GPT (ROME; causal tracing).",
    "NeurIPS 2022. arXiv:2202.05262.",
    "https://arxiv.org/abs/2202.05262",
    "Background for rank-one weight edits and activation patching (E7, V4, V5).",
  ),
  R(
    13,
    "Kornblith, S., Norouzi, M., Lee, H., Hinton, G.",
    "Similarity of Neural Network Representations Revisited.",
    "ICML 2019, PMLR 97:3519–3529. arXiv:1905.00414.",
    "https://arxiv.org/abs/1905.00414",
    "Linear CKA: the 60 × 60 kinship of layers and the CKA numbers in E6 and V7.",
  ),
  R(
    14,
    "Elhage, N., Nanda, N., Olsson, C., et al.",
    "A Mathematical Framework for Transformer Circuits.",
    "Transformer Circuits Thread (Anthropic), 22 December 2021.",
    "https://transformer-circuits.pub/2021/framework/index.html",
    "The residual-stream view behind gathering and remembering, and the weight-based coupling behind threads and V7’s direct-path estimate.",
  ),
  R(
    15,
    "Shazeer, N.",
    "GLU Variants Improve Transformer.",
    "arXiv:2002.05202, 2020.",
    "https://arxiv.org/abs/2002.05202",
    "The GeGLU feed-forward block whose neurons the instrument calls poles.",
  ),
  R(
    16,
    "Zhang, B., Sennrich, R.",
    "Root Mean Square Layer Normalization.",
    "NeurIPS 2019. arXiv:1910.07467.",
    "https://arxiv.org/abs/1910.07467",
    "RMSNorm, on which the unembed contract and the read-gain analysis of the quiet carry channel depend.",
  ),
  R(
    17,
    "Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.-R., Samek, W.",
    "On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation.",
    "PLoS ONE 10(7): e0130140, 2015.",
    "https://doi.org/10.1371/journal.pone.0130140",
    "The origin of LRP, beneath the R-lens and the J++ backward pass. R1 §2.4 tested a frozen-attention linearisation, not these rules.",
  ),
  R(
    18,
    "Achtibat, R., et al.",
    "AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for Transformers.",
    "ICML 2024. arXiv:2402.05602.",
    "https://arxiv.org/abs/2402.05602",
    "Transformer LRP rules (identity and uniform or half rules for gated and multiplicative units) used in [5].",
  ),
  R(
    19,
    "Sun, M., Chen, X., Kolter, J. Z., Liu, Z.",
    "Massive Activations in Large Language Models.",
    "COLM 2024. arXiv:2402.17762.",
    "https://arxiv.org/abs/2402.17762",
    "The quiet carry channel is shown to be distinct from massive-activation dimensions (V6, REPORT §2.2).",
  ),
  R(
    20,
    "Xiao, G., Tian, Y., Chen, B., Han, S., Lewis, M.",
    "Efficient Streaming Language Models with Attention Sinks.",
    "ICLR 2024. arXiv:2309.17453.",
    "https://arxiv.org/abs/2309.17453",
    "The quiet carry channel is shown to be distinct from attention-sink and start-mark effects (V6, REPORT §2.2).",
  ),
  R(
    21,
    "Turner, A. M., Thiergart, L., Leech, G., Udell, D., Vazquez, J. J., Mini, U., MacDiarmid, M.",
    "Steering Language Models With Activation Engineering (ActAdd; first titled Activation Addition).",
    "arXiv:2308.10248, 2023.",
    "https://arxiv.org/abs/2308.10248",
    "Additive activation steering: the nudge (the code’s “laser”, AddSpec) and V7’s steering diagnostics.",
  ),
  R(
    22,
    "Panickssery (Rimsky), N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., Turner, A. M.",
    "Steering Llama 2 via Contrastive Activation Addition.",
    "ACL 2024. arXiv:2312.06681.",
    "https://arxiv.org/abs/2312.06681",
    "Difference-of-means steering directions (E4’s persona-gender direction).",
  ),
  R(
    23,
    "Heimersheim, S., Nanda, N.",
    "How to use and interpret activation patching.",
    "arXiv:2404.15255, 2024.",
    "https://arxiv.org/abs/2404.15255",
    "Method for the activation and interchange patching in E7, V4 and V5.",
  ),
  R(
    24,
    "Park, K., Choe, Y. J., Veitch, V.",
    "The Linear Representation Hypothesis and the Geometry of Large Language Models.",
    "ICML 2024. arXiv:2311.03658.",
    "https://arxiv.org/abs/2311.03658",
    "The covector-versus-vector distinction behind REPORT §2.2’s “units error” rule for lens- or gradient-derived steering.",
  ),
  R(
    25,
    "Bills, S., Cammarata, N., Mossing, D., et al.",
    "Language models can explain neurons in language models.",
    "OpenAI, 9 May 2023.",
    "https://openai.com/index/language-models-can-explain-neurons-in-language-models/",
    "Activation-based neuron labelling: the approach behind N1B’s activation labels and the comparison point in [9].",
  ),
];

/** C · related work, acknowledged but not relied on */
export const refsC: Reference[] = [
  R(
    26,
    "Briggs, C.",
    "An Exploration of the J-lens.",
    "LessWrong, 7 September 2026.",
    "https://www.lesswrong.com/posts/iQDXuY5Mb5JzjA8xT/an-exploration-of-the-j-lens",
    "An independent J-lens analysis of Gemma-4-31B: blocks of kindred layers by linear CKA, and a strongly skewed final gain that makes Euclidean influence on the final state misleading.",
  ),
  R(
    27,
    "Panchal, A.",
    "The J-lens offset is the model’s token frequency: z-scoring helps.",
    "LessWrong, 17 September 2026.",
    "https://www.lesswrong.com/posts/Qe3jdphWCg9kny3e5/the-j-lens-offset-is-the-model-s-token-frequency-z-scoring",
    "Per-token z-scoring improves J-lens readouts; in N1, z-scoring against an isotropic prior lowered label P@10.",
  ),
  R(
    28,
    "Belrose, N., Furman, Z., Smith, L., Halawi, D., Ostrovsky, I., McKinney, L., Biderman, S., Steinhardt, J.",
    "Eliciting Latent Predictions from Transformers with the Tuned Lens.",
    "arXiv:2303.08112, 2023.",
    "https://arxiv.org/abs/2303.08112",
    "The standard learned-lens baseline; not evaluated in this study.",
  ),
];

/** Numbers of the REFERENCES.md entries in sections D to F, named on the page but listed only in that file. */
export const REFS_D_TO_F = Array.from({ length: 13 }, (_, i) => 29 + i);

/** Total entries in REFERENCES.md (A to F). */
export const REFERENCE_COUNT = 41;

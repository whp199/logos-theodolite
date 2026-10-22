# References

This file lists the external sources, prior ideas and software that Logos Theodolite and the `experiments/` study rely on. Bracketed numbers such as [2] are the markers used in the reports and docs. Each entry has a short note on how this repository uses the source. Dates are publication or posting dates. All links were checked on 2026-10-16, those of section G on 2026-10-18.

## A. Model, lenses and lens methods

1. **Gemma Team (Google DeepMind). *Gemma 4 Technical Report.* arXiv:2607.02770, 2026.** https://arxiv.org/abs/2607.02770
   - Checkpoints: `google/gemma-4-31B-it` and `google/gemma-4-31B-it-qat-w4a16-ct` (snapshot `52f3f65bc7a02d555763bc923bd1d9094898219d`). https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct
   - License (Apache 2.0): https://ai.google.dev/gemma/docs/gemma_4_license
   - *Use:* the model under study. Architecture constants (60 layers, 1024-token sliding window, every 6th layer global) come from the model card and config. Gemma is a trademark of Google LLC.

2. **Gurnee, W., Sofroniew, N., Pearce, A., Piotrowski, M., Kauvar, I., Chen, R., Soligo, A., Bogdan, P., Ong, E., Wang, R., Thompson, T. B., Abrahams, D., Kantamneni, S., Ameisen, E., Batson, J., Lindsey, J. *Verbalizable Representations Form a Global Workspace in Language Models.* Transformer Circuits Thread (Anthropic), 6 July 2026.** https://transformer-circuits.pub/2026/workspace/index.html
   - *Use:* source of the Jacobian lens (J-lens, "JLens" here) and the J-space concept. The tool's readout `unembed(h·J[l]ᵀ)`, the lens-vector steering idea and the default two-hop prompt ("The number of legs on the animal that spins webs is", with its spider→ant / 8→6 swap) all come from this paper. The J-lens estimator sums cotangents over all target positions t′ ≥ t and averages over source positions.

3. **Anthropic. *jacobian-lens* (Python package `jlens`): companion code for [2]. Apache-2.0.** https://github.com/anthropics/jacobian-lens
   - *Use:* the library used to fit the Neuronpedia JLens files [6]. It is the reference for the J-lens fitting recipe that R1 compares against.

4. **Ayonrinde, K. (Anthropic Fellows), Lindsey, J. (Anthropic). *J++ Lens: Jacobian Filtering Enables More Faithful Workspace Lenses.* LessWrong, 8 October 2026.** https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr/
   - Code (Apache-2.0): https://github.com/safety-research/jpp_lens
   - Lenses: https://huggingface.co/koayon/jpp-lenses
   - *Use:* the tool's primary lens ("J++", `jpp`). J++ makes three changes to the J-lens:
     - it down-weights Jacobians from activation clusters with poor readouts;
     - it computes Jacobians with an LRP backward pass (`lrp_mode="rlens"`, following [5]);
     - it removes non-semantic tokens from the readout.
   - The gemma-4-31b J++ lens was fitted on the **base** model, on 64 × 128-token WikiText-103 sequences [7]. This instrument applies only the lens matrix, not the readout token filter.

5. **Blank, C., Bhatia, A., Nanda, N. *R-lens: Making J-lens More Faithful on Early Layers.* Alignment Forum / LessWrong, 5 August 2026.** https://www.alignmentforum.org/posts/nv8oedrnLXKRzNEL9
   - *Use:* source of the LRP rules J++ uses: LN-rule on RMSNorm, identity and half rules on gated MLPs, plain gradients through attention. R1 §2.4 discusses these rules.

6. **Neuronpedia. *jacobian-lens*: pre-fitted Jacobian and J++ lenses (Hugging Face; tagged apache-2.0).** https://huggingface.co/neuronpedia/jacobian-lens
   - Files used: `gemma-4-31b/jpp/Salesforce-wikitext/gemma-4-31B_jpp_lens.pt` (J++ by Ayonrinde & Lindsey, converted to fp16 by Neuronpedia) and `gemma-4-31b/jlens/Salesforce-wikitext/gemma-4-31B_jacobian_lens.pt` (fitted by Neuronpedia with [3]; base `google/gemma-4-31B`; 861 WikiText-103 prompts of up to 128 tokens). Snapshot `e166deb5ad540932100ef803c8c9bcc2a5453694`.
   - Related announcement: Lin, J., Chanin, D. *Welcome to the J-Space.* Neuronpedia blog, 10 July 2026. https://www.neuronpedia.org/blog/jacobian-lens
   - *Use:* both lens tensors are downloaded at run time and are not redistributed here.

7. **Merity, S., Xiong, C., Bradbury, J., Socher, R. *Pointer Sentinel Mixture Models.* ICLR 2017. arXiv:1609.07843.** https://arxiv.org/abs/1609.07843
   - *Use:* introduces WikiText-103 (`Salesforce/wikitext`, `wikitext-103-raw-v1`), the corpus both lenses were fitted on.

## B. Interpretability methods and concepts

8. **nostalgebraist. *interpreting GPT: the logit lens.* LessWrong, 31 August 2020.** https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
   - *Use:* the "naive lens" baseline (`unembed(h)`) throughout E1–E7, N1 and R1, and the "Naive Logit Lens" pole labels.

9. **Cammarata, N. *Neural Networks Are Already Clean.* Goodfire, 15 October 2026.** https://already-clean.goodfire.com/ (index: https://www.goodfire.com/research/already-clean)
   - *Use:*
     - treating each neuron's positive and negative directions as "poles", the native units behind the tool's "Bipolar GeGLU poles" and its "Already Clean" label;
     - the "double-negative regime" of gated MLPs;
     - the preliminary Qwen3-32B audit (about 90% of neuron poles with one main job, ~7% with two related jobs, ~2–4% not understood), which N1's cleanliness audit is compared against.

10. **Hayuk, C. (chrishayuk). *LARQL*: reference implementation of VINDEX3. Apache-2.0.** https://github.com/chrishayuk/larql (LQL guide: https://github.com/chrishayuk/larql/blob/main/docs/lql-guide.md)
    - *Use:* inspiration for the names of the tool's "vault" and query dialect (VINDEX, LQL-like SELECT/DESCRIBE, patch overlays). `backend/larql_vault.py` does not call LARQL beyond `--version`. Its `EDGES/POLES NEAREST`, `.patch` and `ROLLBACK` syntax is this project's own, not LQL. REPORT §3.3 quotes the LARQL README's warning that a projected contribution is not a counterfactual.

11. **Meng, K., Sen Sharma, A., Andonian, A., Belinkov, Y., Bau, D. *Mass-Editing Memory in a Transformer* (MEMIT). ICLR 2023. arXiv:2210.07229.** https://arxiv.org/abs/2210.07229
    - *Use:* named only. The tool's ".patch MEMIT" is a single-neuron `down_proj` column add (DELETE zeroes the column) with no covariance solve. It is **not** MEMIT (E6 §3 item 12; REPORT §3.3).

12. **Meng, K., Bau, D., Andonian, A., Belinkov, Y. *Locating and Editing Factual Associations in GPT* (ROME; causal tracing). NeurIPS 2022. arXiv:2202.05262.** https://arxiv.org/abs/2202.05262
    - *Use:* background for rank-one weight edits and causal tracing / activation patching (E7, V4, V5).

13. **Kornblith, S., Norouzi, M., Lee, H., Hinton, G. *Similarity of Neural Network Representations Revisited.* ICML 2019, PMLR 97:3519–3529. arXiv:1905.00414.** https://arxiv.org/abs/1905.00414
    - *Use:* linear CKA, the tool's 60×60 layer-similarity matrix (`compute_linear_cka_matrix`) and the CKA numbers in E6/V7.

14. **Elhage, N., Nanda, N., Olsson, C., et al. *A Mathematical Framework for Transformer Circuits.* Transformer Circuits Thread (Anthropic), 22 December 2021.** https://transformer-circuits.pub/2021/framework/index.html
    - *Use:* the residual-stream view behind Module 1 (attention and FFN writes as additive residual updates) and the weight-based "virtual weight" coupling behind Module 3 filaments and V7's direct-path estimate.

15. **Shazeer, N. *GLU Variants Improve Transformer.* arXiv:2002.05202, 2020.** https://arxiv.org/abs/2002.05202
    - *Use:* the GeGLU feed-forward block whose neurons the tool calls "poles".

16. **Zhang, B., Sennrich, R. *Root Mean Square Layer Normalization.* NeurIPS 2019. arXiv:1910.07467.** https://arxiv.org/abs/1910.07467
    - *Use:* RMSNorm. The read-gain analysis of the quiet carry channel (V6, REPORT §2.2) and the unembed contract depend on Gemma's RMSNorm weights.

17. **Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.-R., Samek, W. *On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation.* PLoS ONE 10(7): e0130140, 2015.** https://doi.org/10.1371/journal.pone.0130140
    - *Use:* origin of LRP, the basis of the R-Lens and J++ backward pass [4, 5] tested in R1 §2.4.

18. **Achtibat, R., et al. *AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for Transformers.* ICML 2024. arXiv:2402.05602.** https://arxiv.org/abs/2402.05602
    - *Use:* transformer LRP rules (identity and uniform/half rules for gated and multiplicative units) used in [5].

19. **Sun, M., Chen, X., Kolter, J. Z., Liu, Z. *Massive Activations in Large Language Models.* COLM 2024. arXiv:2402.17762.** https://arxiv.org/abs/2402.17762
    - *Use:* V6 and REPORT §2.2 show the quiet carry channel is distinct from massive-activation dimensions.

20. **Xiao, G., Tian, Y., Chen, B., Han, S., Lewis, M. *Efficient Streaming Language Models with Attention Sinks.* ICLR 2024. arXiv:2309.17453.** https://arxiv.org/abs/2309.17453
    - *Use:* V6 and REPORT §2.2 show the quiet carry channel is distinct from attention-sink and BOS effects.

21. **Turner, A. M., Thiergart, L., Leech, G., Udell, D., Vazquez, J. J., Mini, U., MacDiarmid, M. *Steering Language Models With Activation Engineering* (ActAdd; first titled *Activation Addition*). arXiv:2308.10248, 2023.** https://arxiv.org/abs/2308.10248
    - *Use:* additive activation steering (E4 steering, the tool's "laser" `AddSpec`, V7 steering diagnostics).

22. **Panickssery (Rimsky), N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., Turner, A. M. *Steering Llama 2 via Contrastive Activation Addition.* ACL 2024. arXiv:2312.06681.** https://arxiv.org/abs/2312.06681
    - *Use:* difference-of-means steering directions (E4 persona-gender direction).

23. **Heimersheim, S., Nanda, N. *How to use and interpret activation patching.* arXiv:2404.15255, 2024.** https://arxiv.org/abs/2404.15255
    - *Use:* methodology for the activation and interchange patching in E7, V4 and V5.

24. **Park, K., Choe, Y. J., Veitch, V. *The Linear Representation Hypothesis and the Geometry of Large Language Models.* ICML 2024. arXiv:2311.03658.** https://arxiv.org/abs/2311.03658
    - *Use:* the covector-versus-vector (causal inner product) distinction behind REPORT §2.2's "units error" rule for lens- or gradient-derived steering.

25. **Bills, S., Cammarata, N., Mossing, D., et al. *Language models can explain neurons in language models.* OpenAI, 9 May 2023.** https://openai.com/index/language-models-can-explain-neurons-in-language-models/
    - *Use:* activation-based (dataset-example) neuron labelling, the approach behind N1B's activation labels and the comparison point in [9].

## C. Related work (acknowledged; not relied on)

26. **Briggs, C. *An Exploration of the J-lens.* LessWrong, 7 September 2026.** https://www.lesswrong.com/posts/iQDXuY5Mb5JzjA8xT/an-exploration-of-the-j-lens
    - Independent J-lens analysis of Gemma-4-31B. It finds linear-CKA layer blocks with a seam near L25, and notes that a strongly skewed final gain makes Euclidean influence on the final state misleading. That is related to REPORT §2.2 (read-gain-weighted sensitivity, quiet carry channel).

27. **Panchal, A. *The J-lens offset is the model's token frequency: z-scoring helps.* LessWrong, 17 September 2026.** https://www.lesswrong.com/posts/Qe3jdphWCg9kny3e5/the-j-lens-offset-is-the-model-s-token-frequency-z-scoring
    - Reports that per-token z-scoring improves J-lens readouts. Compare N1, where z-scoring against an isotropic prior lowers label P@10.

28. **Belrose, N., Furman, Z., Smith, L., Halawi, D., Ostrovsky, I., McKinney, L., Biderman, S., Steinhardt, J. *Eliciting Latent Predictions from Transformers with the Tuned Lens.* arXiv:2303.08112, 2023.** https://arxiv.org/abs/2303.08112
    - The standard learned-lens baseline. It was not evaluated in this study.

## D. Statistics and numerical methods

29. **Wilson, E. B. Probable Inference, the Law of Succession, and Statistical Inference. *JASA* 22(158): 209–212, 1927.** https://doi.org/10.1080/01621459.1927.10502953
    - *Use:* Wilson 95% CIs for pronoun rates (E2, V1, N1C).
30. **Efron, B. Bootstrap Methods: Another Look at the Jackknife. *Annals of Statistics* 7(1): 1–26, 1979.** https://doi.org/10.1214/aos/1176344552
    - *Use:* bootstrap CIs throughout.
31. **Simpson, E. H. The Interpretation of Interaction in Contingency Tables. *JRSS B* 13(2): 238–241, 1951.** https://www.jstor.org/stable/2984065
    - *Use:* the Simpson's-paradox reversal of E6's filament correlation (V7, V8; REPORT §3.4 and §4).
32. **Lin, J. Divergence Measures Based on the Shannon Entropy. *IEEE Trans. Inf. Theory* 37(1): 145–151, 1991.** https://doi.org/10.1109/18.61115
    - *Use:* Jensen–Shannon divergence (E5).
33. **Löwdin, P.-O. On the Non-Orthogonality Problem Connected with the Use of Atomic Wave Functions in the Theory of Molecules and Crystals. *J. Chem. Phys.* 18(3): 365–375, 1950.** https://doi.org/10.1063/1.1747632
    - *Use:* symmetric (Löwdin) orthogonalisation, proposed as an order-independent survey frame (V7; REPORT §3.4 bug 3).

## E. Data sources

34. **Wikipedia contributors. English Wikipedia (lead sections of the articles named in `experiments/scripts/E3_lens_validation_data.py`).** https://en.wikipedia.org — CC BY-SA 4.0
    - *Use:* E3's 24 PROSE stimuli are close paraphrases or adaptations and are shared under CC BY-SA 4.0 (see NOTICE).
35. **U.S. Bureau of Labor Statistics. *Current Population Survey*, Table 11: Employed persons by detailed occupation, sex, race, and Hispanic or Latino ethnicity.** https://www.bls.gov/cps/cpsaat11.htm
    - *Use:* reference for V2's approximate occupational gender code G ("BLS-style % women"). The values in `V2_forced_choice_common.py` are hand-coded approximations.

The N1B activation corpus (local `/usr/share/doc` and package sources) is not distributed. Verbatim contexts from it were removed (README; N1_REPORT publication note).

## F. Software

36. **decoderesearch. *interp-engine*. Apache-2.0.** https://github.com/decoderesearch/interp-engine — hook addresses, `EagerModel` and the steering specs used by the instrument.
37. **PrefectHQ. *FastMCP*. Apache-2.0.** https://github.com/PrefectHQ/fastmcp — MCP server (`backend/mcp_scientist.py`).
38. **PyTorch** (https://pytorch.org) and **Hugging Face Transformers, huggingface_hub and safetensors** (https://github.com/huggingface) — model execution and downloads.
39. **compressed-tensors** (https://github.com/neuralmagic/compressed-tensors) — loading the W4A16 QAT checkpoint.
40. **FastAPI** (https://fastapi.tiangolo.com) and **Uvicorn** — server.
41. **React, React Three Fiber / drei, three.js, Zustand, Vite** — frontend (see `frontend/package.json`).

## G. Sampling: how the instrument writes

Where the study read the lens on answers the model wrote itself, they were written greedily, its likeliest word-piece each time (E3, R1). The instrument can also draw its answers by chance, as chat apps do. These are the methods it then applies, in the order it applies them: the presence penalty [44], the temperature, top-k [42], top-p [43], then one draw.

42. **Fan, A., Lewis, M., Dauphin, Y. *Hierarchical Neural Story Generation.* ACL 2018, pp. 889–898. arXiv:1805.04833.** https://aclanthology.org/P18-1082/
    - arXiv: https://arxiv.org/abs/1805.04833
    - *Use:* top-k sampling, drawing only from the k likeliest word-pieces ("from its likeliest K" on the instrument; `top_k`, 0 = no limit).

43. **Holtzman, A., Buys, J., Du, L., Forbes, M., Choi, Y. *The Curious Case of Neural Text Degeneration.* ICLR 2020. arXiv:1904.09751.** https://arxiv.org/abs/1904.09751
    - *Use:* nucleus (top-p) sampling, drawing only from the fewest likeliest word-pieces whose odds add up to p ("within P of its odds"; `top_p`, 1 = no limit).

44. **vLLM project. *SamplingParams* (`vllm/sampling_params.py`). Apache-2.0.** https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
    - *Use:* the presence penalty as vLLM defines it: a fixed amount taken off the score of every token that has appeared in the generated text so far, the prompt not counted ("hold back repeats"; `presence_penalty`). interp-engine [36] applies it the same way, on the raw scores before the temperature and on the greedy path too, which its code gives as vLLM's order (`interp_engine.sampling.apply_presence_penalty`, `interp_engine.steer._sample_next`).

## R2 result: the filtered second reading, layer band by layer band

Applied exactly as pre-registered (R2_PREREG.md §7). A lens is served at a band only if all five rules hold: (1) pooled over the four conditions, its filtered top word names the model's next content word more often than both filtered J++ and the filtered naked eye (95% prompt-bootstrap intervals above 0); (2) it is significantly worse than neither in any single condition; (3) its stuck-word share is at most 25% in every condition at every layer of the band; (4) the exact sign test over prompts, against both baselines, is below 0.025 after Holm's adjustment across the lens-band tests; (5) both pooled differences are at least 0.05. Layers count from 0 (the pages show layer + 1).

| band (code) | R1's lens | Miru | served |
|---|---|---|---|
| 40-49 | not read | fails rule 1, 2, 3, 4, 5 | none |
| 50 | fails rule 1, 4, 5 | fails rule 1, 3, 4, 5 | none |
| 51-54 | not read | fails rule 3, 4 | none |
| 55 | fails rule 1, 5 | fails rule 1 | none |
| 56-57 | not read | fails rule 1, 4, 5 | none |
| 58 | fails rule 5 | fails rule 1, 2, 3, 4, 5 | none |

**Served:** R1's lens at no layer; Miru at no layer.

Pooled filtered top-1 (the share of a prompt's positions where the readout's top content word is the model's next content word, averaged over prompts), the paired differences, and the sign test's discordant prompts (+ the second reading better, - worse):

| lens | band | second | J++ | naked | second - J++ | second - naked | +/- vs J++ | +/- vs naked | p (Holm) | worse in | max stuck (layer, condition, word) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r1 | 50 | 0.372 | 0.267 | 0.358 | +0.105 [+0.044, +0.167] | +0.014 [-0.040, +0.066] | 52/15 | 33/19 | 0.21 | - | 0.08 (50, chat-no-answer, 'Depending') | do not serve |
| r1 | 55 | 0.493 | 0.409 | 0.450 | +0.085 [+0.036, +0.135] | +0.044 [-0.004, +0.092] | 39/7 | 36/12 | 0.0029 | - | 0.07 (55, chat-no-answer, ' depending') | do not serve |
| r1 | 58 | 0.681 | 0.606 | 0.634 | +0.075 [+0.030, +0.124] | +0.046 [+0.013, +0.084] | 41/8 | 31/8 | 0.0013 | - | 0.03 (58, chat-no-answer, 'I') | do not serve |
| miru | 40-49 | 0.221 | 0.182 | 0.262 | +0.039 [-0.009, +0.086] | -0.041 [-0.074, -0.005] | 53/63 | 37/73 | 1 | chat-begun (naked); bare (naked); mid-reply (naked) | 0.28 (44, bare, ' kind') | do not serve |
| miru | 50 | 0.341 | 0.267 | 0.358 | +0.074 [+0.005, +0.146] | -0.018 [-0.073, +0.040] | 51/28 | 28/28 | 1 | - | 0.27 (50, chat-begun, 'textbf') | do not serve |
| miru | 51-54 | 0.456 | 0.372 | 0.394 | +0.084 [+0.034, +0.137] | +0.063 [+0.017, +0.111] | 66/40 | 53/39 | 0.44 | - | 0.32 (51, chat-begun, 'textbf') | do not serve |
| miru | 55 | 0.502 | 0.409 | 0.450 | +0.093 [+0.038, +0.150] | +0.052 [-0.003, +0.109] | 46/17 | 43/20 | 0.018 | - | 0.17 (55, bare, ' kind') | do not serve |
| miru | 56-57 | 0.529 | 0.505 | 0.535 | +0.024 [-0.032, +0.087] | -0.006 [-0.060, +0.054] | 53/45 | 50/44 | 1 | - | 0.15 (57, bare, ' kind') | do not serve |
| miru | 58 | 0.029 | 0.606 | 0.634 | -0.577 [-0.637, -0.517] | -0.606 [-0.662, -0.542] | 0/132 | 0/136 | 1 | chat-no-answer (J++); chat-no-answer (naked); chat-begun (J++); chat-begun (naked); bare (J++); bare (naked); mid-reply (J++); mid-reply (naked) | 1.00 (58, chat-begun, ' same') | do not serve |

Why each failing band fails:
- r1 50: [rule 1] pooled CI of second - naked eye does not lie above 0; [rule 4] sign test p = 0.0352 (J++ 52+/15-, naked 33+/19-), Holm-adjusted 0.211 over 9 tests, not below 0.025; [rule 5] pooled second - naked eye is +0.014, below the minimum 0.05
- r1 55: [rule 1] pooled CI of second - naked eye does not lie above 0; [rule 5] pooled second - naked eye is +0.044, below the minimum 0.05
- r1 58: [rule 5] pooled second - naked eye is +0.046, below the minimum 0.05
- miru 40-49: [rule 1] pooled CI of second - J++ does not lie above 0; [rule 1] pooled CI of second - naked eye does not lie above 0; [rule 2] bare: significantly worse than naked eye; [rule 2] chat-begun: significantly worse than naked eye; [rule 2] mid-reply: significantly worse than naked eye; [rule 3] stuck-word share above 25% at 42 bare (0.27, ' kind'), 43 bare (0.27, ' kind'), 44 bare (0.28, ' kind'), 47 bare (0.28, ' kind'), 48 bare (0.27, ' kind'), 49 chat-begun (0.28, 'textbf'); [rule 4] sign test p = 1 (J++ 53+/63-, naked 37+/73-), Holm-adjusted 1 over 9 tests, not below 0.025; [rule 5] pooled second - J++ is +0.039, below the minimum 0.05; [rule 5] pooled second - naked eye is -0.041, below the minimum 0.05
- miru 50: [rule 1] pooled CI of second - naked eye does not lie above 0; [rule 3] stuck-word share above 25% at 50 chat-begun (0.27, 'textbf'); [rule 4] sign test p = 0.553 (J++ 51+/28-, naked 28+/28-), Holm-adjusted 1 over 9 tests, not below 0.025; [rule 5] pooled second - naked eye is -0.018, below the minimum 0.05
- miru 51-54: [rule 3] stuck-word share above 25% at 51 chat-begun (0.32, 'textbf'); [rule 4] sign test p = 0.0875 (J++ 66+/40-, naked 53+/39-), Holm-adjusted 0.437 over 9 tests, not below 0.025
- miru 55: [rule 1] pooled CI of second - naked eye does not lie above 0
- miru 56-57: [rule 1] pooled CI of second - J++ does not lie above 0; [rule 1] pooled CI of second - naked eye does not lie above 0; [rule 4] sign test p = 0.303 (J++ 53+/45-, naked 50+/44-), Holm-adjusted 1 over 9 tests, not below 0.025; [rule 5] pooled second - J++ is +0.024, below the minimum 0.05; [rule 5] pooled second - naked eye is -0.006, below the minimum 0.05
- miru 58: [rule 1] pooled CI of second - J++ does not lie above 0; [rule 1] pooled CI of second - naked eye does not lie above 0; [rule 2] bare: significantly worse than J++; [rule 2] bare: significantly worse than naked eye; [rule 2] chat-begun: significantly worse than J++; [rule 2] chat-begun: significantly worse than naked eye; [rule 2] chat-no-answer: significantly worse than J++; [rule 2] chat-no-answer: significantly worse than naked eye; [rule 2] mid-reply: significantly worse than J++; [rule 2] mid-reply: significantly worse than naked eye; [rule 3] stuck-word share above 25% at 58 chat-no-answer (0.95, ' same'), 58 chat-begun (1.00, ' same'), 58 bare (0.93, ' same'), 58 mid-reply (0.85, ' same'); [rule 4] sign test p = 1 (J++ 0+/132-, naked 0+/136-), Holm-adjusted 1 over 9 tests, not below 0.025; [rule 5] pooled second - J++ is -0.577, below the minimum 0.05; [rule 5] pooled second - naked eye is -0.606, below the minimum 0.05

Sensitivity analyses (pre-registered; reported beside the decision above, which stands):

| analysis | what is left out | served | verdicts that change |
|---|---|---|---|
| S1 (without the prompts about the model itself (R1's fit set had many self-questions)) | 27 prompts | R1 [55]; Miru - | r1 55: fail -> pass/served |
| S2 (without the prompts that paraphrase or share an answer topic with R1's fit prompts (R2_common.R1_OVERLAP)) | 22 prompts | R1 [55]; Miru - | r1 55: fail -> pass/served |
| S3 (without the bare condition (raw text puts this instruct model off-distribution)) | the bare condition | R1 -; Miru - | none |
| S4 (without positions whose content target is a list numeral after a newline or the token after a code fence) | 0 positions | R1 -; Miru - | none |

For transparency (not used for the decision): the unfiltered top word as the model's next token (the earlier test's metric) and the share of positions whose unfiltered top word is formatting.

| lens | band | unfiltered top-1 = next token: second / J++ / naked | formatting share of the second (unfiltered), by layer |
|---|---|---|---|
| r1 | 50 | 0.316 / 0.137 / 0.160 | 50: 0.18 |
| r1 | 55 | 0.364 / 0.227 / 0.215 | 55: 0.16 |
| r1 | 58 | 0.732 / 0.520 / 0.514 | 58: 0.23 |
| miru | 40-49 | 0.185 / 0.086 / 0.128 | 40: 0.30, 41: 0.33, 42: 0.32, 43: 0.32, 44: 0.34, 45: 0.35, 46: 0.33, 47: 0.28, 48: 0.27, 49: 0.33 |
| miru | 50 | 0.255 / 0.137 / 0.160 | 50: 0.32 |
| miru | 51-54 | 0.320 / 0.131 / 0.186 | 51: 0.31, 52: 0.25, 53: 0.24, 54: 0.22 |
| miru | 55 | 0.360 / 0.227 / 0.215 | 55: 0.21 |
| miru | 56-57 | 0.380 / 0.274 / 0.256 | 56: 0.19, 57: 0.19 |
| miru | 58 | 0.173 / 0.520 / 0.514 | 58: 0.12 |

Context:
- chat-no-answer: 60 prompts, 60 positions (60 with a content target, of which 0 left out as raw-text collapse words; 0 formatting-like targets); the model's next token is formatting at 8%; reading forward agrees with the target's next token at 100.0%.
- chat-begun: 60 prompts, 60 positions (60 with a content target, of which 0 left out as raw-text collapse words; 0 formatting-like targets); the model's next token is formatting at 53%; reading forward agrees with the target's next token at 100.0%.
- bare: 60 prompts, 60 positions (55 with a content target, of which 16 left out as raw-text collapse words; 0 formatting-like targets); the model's next token is formatting at 20%; reading forward agrees with the target's next token at 100.0%.
- mid-reply: 40 prompts, 240 positions (239 with a content target, of which 0 left out as raw-text collapse words; 0 formatting-like targets); the model's next token is formatting at 19%; reading forward agrees with the target's next token at 99.6%.
- mid-reply-content-next: 40 prompts, 120 positions (120 with a content target, of which 0 left out as raw-text collapse words; 0 formatting-like targets); the model's next token is formatting at 0%; reading forward agrees with the target's next token at 100.0%.
- The model's own filtered top word at 59 is its next content word at 0.772 (pooled): the ceiling a same-step readout can reach.
- A constant guess (each condition's most common target word, leave-one-out) scores 0.127 [0.087, 0.167] pooled; by condition: chat-no-answer 0.350 ('the'); chat-begun 0.000 ('1'); bare 0.103 ('a'); mid-reply 0.004 ('a').
- Split by function-word targets (position-level, pooled): r1 50: function words 0.455 vs J++ 0.117 (n 154), other 0.385 vs 0.373 (n 244); r1 55: function words 0.435 vs J++ 0.247 (n 154), other 0.603 vs 0.574 (n 244); r1 58: function words 0.747 vs J++ 0.610 (n 154), other 0.705 vs 0.631 (n 244); miru 40-49: function words 0.419 vs J++ 0.140 (n 154), other 0.112 vs 0.243 (n 244); miru 50: function words 0.500 vs J++ 0.117 (n 154), other 0.287 vs 0.373 (n 244); miru 51-54: function words 0.521 vs J++ 0.177 (n 154), other 0.448 vs 0.510 (n 244); miru 55: function words 0.513 vs J++ 0.247 (n 154), other 0.553 vs 0.574 (n 244); miru 56-57: function words 0.562 vs J++ 0.321 (n 154), other 0.574 vs 0.658 (n 244); miru 58: function words 0.039 vs J++ 0.610 (n 154), other 0.053 vs 0.631 (n 244).
- Miru - R1 where both read (filtered, pooled): 50: -0.031 [-0.082, +0.018]; 55: +0.008 [-0.037, +0.055]; 58: -0.652 [-0.707, -0.590].
- Power: pooled intervals resolve differences of about 0.06-0.07; per-condition intervals (40-60 prompts) are about ±0.12-0.17 wide, so rule 2 catches only large harms (R2_PREREG.md §7).

Integrity: the pre-registration, prompts, scripts and freshness record on disk are those the GPU run recorded.
Git: the run was made at HEAD cb3d6e2570f3 with every pre-registered file committed.
Runs: 1 GPU run(s) recorded in R2_02_runs.jsonl: started 2026-10-17T23:50:50; finished 2026-10-17T23:56:54.
Bootstrap: 2000 prompt-level resamples, seed 20261019. Per-condition intervals, per-layer tables, exact-id and top-5 variants, the mid-reply content-next draw, the split by the next token's kind and the sensitivity details are in R2_03_analysis.json.

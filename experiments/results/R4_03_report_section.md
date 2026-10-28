## R4 result: R1's instruct lens at pages 56 and 59 (code 55 and 58)

Applied exactly as pre-registered (R4_PREREG.md §5: R2's five rules, unchanged, by R3's code). R1 passes R2's rule at a band only if all five hold: (1) pooled, its filtered top word names the model's next word (R2's filter: function words count) more often than both filtered J++ and the filtered naked eye (95% prompt-bootstrap intervals above 0); (2) it is significantly worse than neither in any condition; (3) its stuck-word share is at most 25% in every condition; (4) the exact sign test against both baselines is below 0.025 after Holm's adjustment over the 2 bands; (5) both pooled differences are at least 0.05. **A band is served only if R1 also passes all five again under S6, without function-word targets** (the owner's choice, §2). A served band permits a later serving track to show R1's filtered top-1 at that page; R4 serves nothing itself.

| page (code) | R2's rule | S6, without function-word targets | served |
|---|---|---|---|
| 56 (55) | passes | fails rule 1, 2, 4, 5 | **not served** |
| 59 (58) | fails rule 5 | fails rule 1, 4, 5 | **not served** |

R2's rule, in detail:

| page (code) | R1 | J++ | naked eye | R1 - J++ | R1 - naked eye | +/- vs J++ | +/- vs naked | p (Holm over 2) | worse in | max stuck (condition, word) | R2's rule |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 56 (55) | 0.570 | 0.464 | 0.482 | +0.106 [+0.077, +0.135] | +0.088 [+0.059, +0.117] | 146/21 | 149/30 | 3.6e-20 | - | 0.03 (chat-no-answer, 'Depending') | PASSES |
| 59 (58) | 0.717 | 0.653 | 0.679 | +0.064 [+0.039, +0.092] | +0.038 [+0.015, +0.059] | 118/32 | 104/28 | 8.9e-12 | - | 0.02 (chat-no-answer, 'While') | fails rule 5 |

Why each band that fails R2's rule fails:
- page 59 (code 58): [rule 5] pooled second - naked eye is +0.038, below the minimum 0.05

Why each band that fails S6 fails (without function-word targets):
- page 56 (code 55): R1 - J++ +0.025 [-0.003, +0.054], R1 - naked eye +0.015 [-0.016, +0.043]; [rule 1] pooled CI of second - J++ does not lie above 0; [rule 1] pooled CI of second - naked eye does not lie above 0; [rule 2] chat-no-answer: significantly worse than J++; [rule 4] sign test p = 0.0268 (J++ 48+/23-, naked 48+/30-), Holm-adjusted 0.0535 over 2 tests, not below 0.025; [rule 5] pooled second - J++ is +0.025, below the minimum 0.05; [rule 5] pooled second - naked eye is +0.014, below the minimum 0.05
- page 59 (code 58): R1 - J++ +0.073 [+0.045, +0.101], R1 - naked eye +0.021 [-0.004, +0.046]; [rule 1] pooled CI of second - naked eye does not lie above 0; [rule 4] sign test p = 0.126 (J++ 59+/14-, naked 36+/26-), Holm-adjusted 0.126 over 2 tests, not below 0.025; [rule 5] pooled second - naked eye is +0.021, below the minimum 0.05

Per condition (prompt-level 95% intervals; the decision conditions):

| page | condition | prompts | R1 | R1 - J++ | R1 - naked eye | stuck share |
|---|---|---|---|---|---|---|
| 56 | chat-no-answer | 160 | 0.606 | +0.175 [+0.106, +0.244] | +0.119 [+0.050, +0.188] | 0.03 |
| 56 | chat-begun | 159 | 0.516 | +0.038 [-0.019, +0.094] | +0.031 [-0.031, +0.094] | 0.01 |
| 56 | bare | 70 | 0.514 | +0.014 [-0.043, +0.086] | +0.000 [-0.057, +0.057] | 0.01 |
| 56 | mid-reply | 160 | 0.612 | +0.145 [+0.120, +0.170] | +0.152 [+0.127, +0.178] | 0.01 |
| 59 | chat-no-answer | 160 | 0.812 | +0.013 [-0.044, +0.075] | -0.019 [-0.062, +0.025] | 0.02 |
| 59 | chat-begun | 159 | 0.585 | +0.075 [+0.019, +0.138] | +0.038 [-0.013, +0.088] | 0.02 |
| 59 | bare | 70 | 0.643 | +0.071 [+0.000, +0.143] | +0.014 [-0.043, +0.071] | 0.01 |
| 59 | mid-reply | 160 | 0.785 | +0.102 [+0.077, +0.129] | +0.105 [+0.079, +0.130] | 0.01 |

Sensitivity analyses (pre-registered; the decision above stands):

| analysis | passes | verdicts that change |
|---|---|---|
| S1 (without the prompts about the model itself (R1's fit set had many self-questions)) | page 56 yes; page 59 no | none |
| S2 (without the prompts that paraphrase, or share a subject or an answer with, R1's fit questions and topics (R4_common.FIT_OVERLAP)) | page 56 yes; page 59 no | none |
| S3 (without the bare condition (raw text puts this instruct model off-distribution)) | page 56 yes; page 59 no | none |
| S4 (without positions whose content target is a list numeral after a newline or the token after a code fence) | page 56 yes; page 59 no | none |
| S5 (J++'s filter for every readout (D7)) | not run: the run recorded no J++-filtered readouts (fjs) | - |
| S6 (without the positions whose content target is a function word (R2_common.FUNCTION_WORDS), which R2's filter counts as content) | page 56 no; page 59 no | r1 55: pass -> fail |

Targets that are function words and targets that are not (R2's filter counts both as content; share of positions named, reported only):

| page | where | function-word targets: R1 / J++ / naked (n) | other targets: R1 / J++ / naked (n) |
|---|---|---|---|
| 56 | pooled | 0.566 / 0.335 / 0.324 (615) | 0.620 / 0.577 / 0.593 (724) |
| 56 | chat-no-answer | 0.628 / 0.345 / 0.434 (113) | 0.553 / 0.638 / 0.617 (47) |
| 56 | chat-begun | 0.286 / 0.191 / 0.143 (21) | 0.551 / 0.522 / 0.536 (138) |
| 56 | bare | 0.000 / 0.000 / 0.000 (7) | 0.571 / 0.556 / 0.571 (63) |
| 56 | mid-reply | 0.572 / 0.344 / 0.310 (474) | 0.653 / 0.590 / 0.609 (476) |
| 59 | pooled | 0.785 / 0.678 / 0.632 (615) | 0.733 / 0.664 / 0.721 (724) |
| 59 | chat-no-answer | 0.850 / 0.867 / 0.876 (113) | 0.723 / 0.638 / 0.723 (47) |
| 59 | chat-begun | 0.571 / 0.619 / 0.524 (21) | 0.587 / 0.493 / 0.551 (138) |
| 59 | bare | 0.143 / 0.143 / 0.429 (7) | 0.698 / 0.619 / 0.651 (63) |
| 59 | mid-reply | 0.789 / 0.643 / 0.582 (474) | 0.781 / 0.723 / 0.779 (476) |

Reported only (§8):

| page | filtered exact id: R1 - J++ / R1 - naked | filtered top-5 word: R1 - J++ / R1 - naked | unfiltered next word-piece: R1 / J++ / naked |
|---|---|---|---|
| 56 | +0.033 [+0.011, +0.056] / +0.064 [+0.045, +0.084] | +0.128 [+0.101, +0.154] / +0.097 [+0.072, +0.124] | 0.380 / 0.267 / 0.244 |
| 59 | +0.233 [+0.203, +0.263] / +0.145 [+0.115, +0.176] | +0.047 [+0.025, +0.070] / +0.029 [+0.012, +0.047] | 0.692 / 0.483 / 0.528 |

The reading forward's own top-1 at 59 against the generated next token (R2 §5, report only): chat-no-answer 1.0, chat-begun 1.0, bare 0.9875, mid-reply 0.9917, mid-reply-content-next 0.9979.

Pooled over R2's, R3's and R4's prompts (reported only, never a decision):

| page | prompts | R1 - naked eye | R1 - J++ | R2 / R3 / R4 alone: R1 - naked eye |
|---|---|---|---|---|
| 56 | 1031 | +0.075 [+0.053, +0.096] | +0.102 [+0.081, +0.123] | +0.044 / +0.071 / +0.088 |
| 59 | 1031 | +0.045 [+0.029, +0.062] | +0.072 [+0.052, +0.092] | +0.046 / +0.058 / +0.038 |

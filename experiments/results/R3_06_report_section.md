## R3 result: J++'s recipe fitted on the instruct model, layer band by layer band

**In short.** Neither instruct refit is served at any band. jpp-it is J++'s full recipe and rl-it its pooled R-lens.
Each fails at least two of R2's five rules at each of the seven bands (code 30-39, 40-49, 50, 51-54, 55, 56-57, 58).
**H1 is refuted at every band.** Refitting J++'s recipe on the served W4A16 instruct checkpoint did not give a second
reading that beats base J++ and the naked eye by R2's standard. So R3 does not show that the base fit is what holds the
second reading back. At code 56-57 and 58 both refits read worse than base J++ and the naked eye. Pooled, both
intervals lie below 0. In single conditions, each refit is significantly worse than both baselines in two or three of
the four. The secondary readings:

- **H2 "both"** in every band, for both lenses.
- **H3 refuted** at all seven judged layers, for both lenses.
- **H4** supported for jpp-it at code 40, 50 and 55 and refuted at 45 and 58; supported for rl-it at 40, 45, 50 and 55
  and refuted at 58.
- **H5 refuted.** No band reads "helps", and code 56-57 reads "hurts".
- **H6 "smaller"** in every band, in q10 and in loud coordinates. This reading carries the fallback's caution: Q1 ran on
  J++'s 64 prompts, not Miru's 425.

The replication on R2's prompts passes all three of its checks. The five sensitivity analyses change no verdict.
Outside the decision, R1's reference lens passes all five rules at code 55 and 58 on R3's prompts. It is not a
candidate, its p is raw, and on R2's prompts it still fails (see "Reference lenses" below).

**Layer numbering.** Every table in this section uses code layers 0-59, as the scripts and R1-R2 do. The pages show
code + 1, so code 30-39 is pages 31-40 and code 58 is page 59. H2's bands are labelled `L30-39`, `L40-49` and `L50-58`,
and H6's are `L30-39`, `L40-49`, `L50-57` and `L58`. These labels are code layers too.

**Provenance.** The text between the two rules below was written by `R3_06_analyze.py --replication
R3_05_eval_r2set.json` (§10.1 step 4). It was run on the CPU at HEAD `7bcd908` with the GPU hidden, and it is
reproduced unchanged except for its heading line, which is the one above. Its input is `R3_05_eval.json` (sha256 `d2fe1a33…`), the output of R3_05's last finished run,
and its output is `R3_06_analysis.json` (sha256 `a843c309…`). Before applying the rule, the script checked its input.
All of these checks passed:

- the registered prompt counts (80 per condition);
- the prompts' sha256 is the one a passing `R3_01_check.json` records;
- the last event of `R3_05_runs.jsonl` is "finished", with the input's sha256;
- every input of H2, H3 and H5 is present, with the lens files' sha256 as `R3_04_weights.json` records them;
- the Holm family: `HOLM_M` is 14, and all 14 candidate-band tests are present;
- the replication file has R2's counts and `R2_prompts.json`'s sha256.

The text after the second rule was written by hand from `R3_06_analysis.json`, `R3_05_h4.json`, `R3_05_recall.json`
and `R3_Q1_jlens.json`. `R3_Q1_jlens.py --analyze-only` was rerun on the CPU (§10.1 step 4) and left
`R3_Q1_jlens.json` byte for byte unchanged.

---

Applied exactly as pre-registered (R3_PREREG.md §7, R2's five rules unchanged). A lens is served at a band only if all five hold: (1) pooled, its filtered top word names the model's next content word more often than both filtered J++ and the filtered naked eye (95% prompt-bootstrap intervals above 0); (2) it is significantly worse than neither in any condition; (3) its stuck-word share is at most 25% in every condition at every layer of the band; (4) the exact sign test against both baselines is below 0.025 after Holm's adjustment over 14 lens-band tests; (5) both pooled differences are at least 0.05. No band below code 30 is served. Layers count from 0 (the pages show layer + 1).

| band (code) | jpp-it | rl-it | served |
|---|---|---|---|
| 30-39 | fails rule 1, 2, 3, 4, 5 | fails rule 1, 2, 3, 4, 5 | none |
| 40-49 | fails rule 1, 2, 3, 4, 5 | fails rule 1, 2, 3, 4, 5 | none |
| 50 | fails rule 1, 3, 4, 5 | fails rule 1, 2, 3, 4, 5 | none |
| 51-54 | fails rule 1, 2, 3, 4, 5 | fails rule 1, 3, 4, 5 | none |
| 55 | fails rule 1, 4, 5 | fails rule 1, 5 | none |
| 56-57 | fails rule 1, 2, 3, 4, 5 | fails rule 1, 2, 3, 4, 5 | none |
| 58 | fails rule 1, 2, 3, 4, 5 | fails rule 1, 2, 3, 4, 5 | none |

| lens | band | second | J++ | naked | second - J++ | second - naked | +/- vs J++ | +/- vs naked | p (Holm for candidates; raw, outside Holm, for the rest) | worse in | max stuck (layer, condition, word) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jpp-it | 30-39 | 0.007 | 0.020 | 0.033 | -0.013 [-0.021, -0.005] | -0.025 [-0.033, -0.018] | 19/47 | 13/81 | 1 | chat-no-answer (J++); chat-no-answer (naked); mid-reply (J++); mid-reply (naked) | 0.46 (30, chat-no-answer, ' ability') | do not serve |
| jpp-it | 40-49 | 0.178 | 0.176 | 0.253 | +0.002 [-0.019, +0.024] | -0.075 [-0.098, -0.053] | 75/82 | 46/110 | 1 | chat-no-answer (naked); chat-begun (naked); mid-reply (naked) | 0.53 (47, chat-no-answer, ' same') | do not serve |
| jpp-it | 50 | 0.302 | 0.238 | 0.346 | +0.064 [+0.015, +0.113] | -0.044 [-0.087, -0.004] | 67/26 | 36/46 | 1 | - | 0.33 (50, chat-no-answer, ' own') | do not serve |
| jpp-it | 51-54 | 0.382 | 0.353 | 0.390 | +0.029 [-0.003, +0.060] | -0.008 [-0.039, +0.024] | 102/57 | 75/68 | 1 | chat-no-answer (naked) | 0.61 (54, chat-no-answer, ' same') | do not serve |
| jpp-it | 55 | 0.470 | 0.425 | 0.460 | +0.045 [+0.004, +0.085] | +0.009 [-0.032, +0.048] | 54/22 | 47/25 | 0.083 | - | 0.20 (55, chat-no-answer, ' own') | do not serve |
| jpp-it | 56-57 | 0.408 | 0.515 | 0.531 | -0.107 [-0.149, -0.064] | -0.122 [-0.168, -0.079] | 41/100 | 43/106 | 1 | chat-no-answer (J++); chat-no-answer (naked); chat-begun (J++); chat-begun (naked); mid-reply (J++); mid-reply (naked) | 0.46 (57, chat-no-answer, ' same') | do not serve |
| jpp-it | 58 | 0.429 | 0.650 | 0.676 | -0.221 [-0.268, -0.176] | -0.247 [-0.291, -0.199] | 14/109 | 10/118 | 1 | chat-no-answer (J++); chat-no-answer (naked); chat-begun (J++); chat-begun (naked); mid-reply (J++); mid-reply (naked) | 0.56 (58, chat-no-answer, ' own') | do not serve |
| rl-it | 30-39 | 0.011 | 0.020 | 0.033 | -0.009 [-0.017, -0.001] | -0.022 [-0.030, -0.014] | 26/43 | 18/75 | 1 | chat-no-answer (J++); chat-no-answer (naked); mid-reply (naked) | 0.81 (37, chat-no-answer, ' own') | do not serve |
| rl-it | 40-49 | 0.163 | 0.176 | 0.253 | -0.014 [-0.040, +0.013] | -0.091 [-0.115, -0.067] | 67/99 | 31/123 | 1 | chat-no-answer (naked); chat-begun (naked); mid-reply (J++); mid-reply (naked) | 0.68 (48, chat-no-answer, ' same') | do not serve |
| rl-it | 50 | 0.272 | 0.238 | 0.346 | +0.034 [-0.019, +0.083] | -0.074 [-0.120, -0.028] | 58/41 | 31/62 | 1 | chat-no-answer (naked); mid-reply (naked) | 0.39 (50, chat-no-answer, ' same') | do not serve |
| rl-it | 51-54 | 0.392 | 0.353 | 0.390 | +0.040 [+0.003, +0.075] | +0.003 [-0.032, +0.037] | 94/60 | 71/63 | 1 | - | 0.41 (52, chat-no-answer, ' same') | do not serve |
| rl-it | 55 | 0.491 | 0.425 | 0.460 | +0.066 [+0.029, +0.102] | +0.030 [-0.009, +0.072] | 56/14 | 51/22 | 0.0064 | - | 0.21 (55, chat-no-answer, ' own') | do not serve |
| rl-it | 56-57 | 0.463 | 0.515 | 0.531 | -0.052 [-0.090, -0.014] | -0.068 [-0.110, -0.030] | 47/86 | 45/87 | 1 | chat-no-answer (J++); chat-no-answer (naked); mid-reply (J++); mid-reply (naked) | 0.42 (56, chat-no-answer, ' own') | do not serve |
| rl-it | 58 | 0.439 | 0.650 | 0.676 | -0.212 [-0.260, -0.165] | -0.237 [-0.282, -0.190] | 16/108 | 11/113 | 1 | chat-no-answer (J++); chat-no-answer (naked); chat-begun (J++); chat-begun (naked); mid-reply (J++); mid-reply (naked) | 0.39 (58, chat-begun, ' same') | do not serve |
| r1 | 50 | 0.392 | 0.238 | 0.346 | +0.154 [+0.103, +0.201] | +0.046 [+0.003, +0.087] | 91/16 | 55/27 | 0.0013 (raw) | - | 0.07 (50, chat-no-answer, ' depending') | reference: fails |
| r1 | 55 | 0.531 | 0.425 | 0.460 | +0.107 [+0.070, +0.144] | +0.071 [+0.035, +0.109] | 72/8 | 64/11 | 1.6e-10 (raw) | - | 0.04 (55, chat-no-answer, ' while') | reference: passes |
| r1 | 58 | 0.734 | 0.650 | 0.676 | +0.083 [+0.038, +0.123] | +0.058 [+0.024, +0.091] | 67/17 | 60/14 | 3.1e-08 (raw) | - | 0.04 (58, chat-no-answer, 'While') | reference: passes |
| miru | 30-39 | 0.014 | 0.020 | 0.033 | -0.006 [-0.015, +0.001] | -0.019 [-0.027, -0.012] | 26/49 | 12/74 | 1 (raw) | chat-begun (naked); mid-reply (J++); mid-reply (naked) | 0.93 (37, chat-no-answer, 'Essay') | reference: fails |
| miru | 40-49 | 0.214 | 0.176 | 0.253 | +0.038 [+0.002, +0.073] | -0.039 [-0.067, -0.012] | 84/92 | 51/103 | 1 (raw) | chat-begun (naked); bare (J++); bare (naked); mid-reply (naked) | 0.42 (42, bare, ' kind') | reference: fails |
| miru | 50 | 0.336 | 0.238 | 0.346 | +0.098 [+0.042, +0.149] | -0.011 [-0.055, +0.034] | 89/36 | 50/45 | 0.34 (raw) | bare (J++); bare (naked) | 0.23 (50, bare, ' kind') | reference: fails |
| miru | 51-54 | 0.431 | 0.353 | 0.390 | +0.078 [+0.033, +0.124] | +0.042 [-0.003, +0.084] | 115/64 | 91/66 | 0.028 (raw) | - | 0.20 (51, chat-begun, 'textbf') | reference: fails |
| miru | 55 | 0.481 | 0.425 | 0.460 | +0.057 [+0.006, +0.101] | +0.021 [-0.027, +0.066] | 65/31 | 58/32 | 0.004 (raw) | - | 0.17 (55, bare, ' kind') | reference: fails |
| miru | 56-57 | 0.542 | 0.515 | 0.531 | +0.027 [-0.016, +0.072] | +0.012 [-0.034, +0.058] | 86/61 | 85/68 | 0.098 (raw) | - | 0.21 (57, bare, ' kind') | reference: fails |
| miru | 58 | 0.024 | 0.650 | 0.676 | -0.626 [-0.673, -0.581] | -0.652 [-0.699, -0.603] | 1/209 | 1/217 | 1 (raw) | chat-no-answer (J++); chat-no-answer (naked); chat-begun (J++); chat-begun (naked); bare (J++); bare (naked); mid-reply (J++); mid-reply (naked) | 1.00 (58, chat-begun, ' same') | reference: fails |
| jpp-it-full-macro | 30-39 | 0.007 | 0.020 | 0.033 | -0.013 [-0.021, -0.005] | -0.025 [-0.033, -0.018] | 19/47 | 13/81 | 1 (raw) | chat-no-answer (J++); chat-no-answer (naked); mid-reply (J++); mid-reply (naked) | 0.46 (30, chat-no-answer, ' ability') | full-macro sensitivity: fails |
| jpp-it-full-macro | 40-49 | 0.178 | 0.176 | 0.253 | +0.002 [-0.019, +0.024] | -0.075 [-0.098, -0.053] | 75/82 | 46/110 | 1 (raw) | chat-no-answer (naked); chat-begun (naked); mid-reply (naked) | 0.53 (47, chat-no-answer, ' same') | full-macro sensitivity: fails |
| jpp-it-full-macro | 50 | 0.302 | 0.238 | 0.346 | +0.064 [+0.015, +0.113] | -0.044 [-0.087, -0.004] | 67/26 | 36/46 | 0.89 (raw) | - | 0.33 (50, chat-no-answer, ' own') | full-macro sensitivity: fails |
| jpp-it-full-macro | 51-54 | 0.382 | 0.353 | 0.390 | +0.029 [-0.003, +0.060] | -0.008 [-0.039, +0.024] | 102/57 | 75/68 | 0.31 (raw) | chat-no-answer (naked) | 0.61 (54, chat-no-answer, ' same') | full-macro sensitivity: fails |
| jpp-it-full-macro | 55 | 0.470 | 0.425 | 0.460 | +0.045 [+0.004, +0.085] | +0.009 [-0.032, +0.048] | 54/22 | 47/25 | 0.0064 (raw) | - | 0.20 (55, chat-no-answer, ' own') | full-macro sensitivity: fails |
| jpp-it-full-macro | 56-57 | 0.408 | 0.515 | 0.531 | -0.107 [-0.149, -0.064] | -0.122 [-0.168, -0.079] | 41/100 | 43/106 | 1 (raw) | chat-no-answer (J++); chat-no-answer (naked); chat-begun (J++); chat-begun (naked); mid-reply (J++); mid-reply (naked) | 0.46 (57, chat-no-answer, ' same') | full-macro sensitivity: fails |
| jpp-it-full-macro | 58 | 0.429 | 0.650 | 0.676 | -0.221 [-0.268, -0.176] | -0.247 [-0.291, -0.199] | 14/109 | 10/118 | 1 (raw) | chat-no-answer (J++); chat-no-answer (naked); chat-begun (J++); chat-begun (naked); mid-reply (J++); mid-reply (naked) | 0.56 (58, chat-no-answer, ' own') | full-macro sensitivity: fails |

Why each failing band fails:
- jpp-it 30-39: [rule 1] pooled CI of second - J++ does not lie above 0; [rule 1] pooled CI of second - naked eye does not lie above 0; [rule 2] chat-no-answer: significantly worse than J++; [rule 2] chat-no-answer: significantly worse than naked eye; [rule 2] mid-reply: significantly worse than J++; [rule 2] mid-reply: significantly worse than naked eye; [rule 3] stuck-word share above 25% at 30 chat-no-answer (0.46, ' ability'), 31 chat-no-answer (0.29, ' ద్వారా'), 32 chat-no-answer (0.36, '₀'), 33 chat-no-answer (0.34, ' बॉबी'), 37 chat-no-answer (0.30, ' tienes'), 38 chat-no-answer (0.44, '回答'); [rule 4] sign test p = 1 (J++ 19+/47-, naked 13+/81-), Holm-adjusted 1 over 14 tests, not below 0.025; [rule 5] pooled second - J++ is -0.013, below the minimum 0.05; [rule 5] pooled second - naked eye is -0.025, below the minimum 0.05
- jpp-it 40-49: [rule 1] pooled CI of second - J++ does not lie above 0; [rule 1] pooled CI of second - naked eye does not lie above 0; [rule 2] chat-begun: significantly worse than naked eye; [rule 2] chat-no-answer: significantly worse than naked eye; [rule 2] mid-reply: significantly worse than naked eye; [rule 3] stuck-word share above 25% at 40 chat-no-answer (0.45, ' own'), 44 chat-no-answer (0.39, ' own'), 44 mid-reply (0.30, ' same'), 45 chat-no-answer (0.39, ' own'), 45 mid-reply (0.30, ' same'), 46 chat-no-answer (0.30, ' own'), 47 chat-no-answer (0.53, ' same'), 48 chat-no-answer (0.30, ' same'), 48 mid-reply (0.29, ' same'), 49 chat-no-answer (0.30, ' same'); [rule 4] sign test p = 1 (J++ 75+/82-, naked 46+/110-), Holm-adjusted 1 over 14 tests, not below 0.025; [rule 5] pooled second - J++ is +0.002, below the minimum 0.05; [rule 5] pooled second - naked eye is -0.075, below the minimum 0.05
- jpp-it 50: [rule 1] pooled CI of second - naked eye does not lie above 0; [rule 3] stuck-word share above 25% at 50 chat-no-answer (0.33, ' own'); [rule 4] sign test p = 0.888 (J++ 67+/26-, naked 36+/46-), Holm-adjusted 1 over 14 tests, not below 0.025; [rule 5] pooled second - naked eye is -0.044, below the minimum 0.05
- jpp-it 51-54: [rule 1] pooled CI of second - J++ does not lie above 0; [rule 1] pooled CI of second - naked eye does not lie above 0; [rule 2] chat-no-answer: significantly worse than naked eye; [rule 3] stuck-word share above 25% at 51 chat-no-answer (0.28, ' same'), 53 chat-no-answer (0.35, ' same'), 54 chat-no-answer (0.61, ' same'); [rule 4] sign test p = 0.308 (J++ 102+/57-, naked 75+/68-), Holm-adjusted 1 over 14 tests, not below 0.025; [rule 5] pooled second - J++ is +0.029, below the minimum 0.05; [rule 5] pooled second - naked eye is -0.008, below the minimum 0.05
- jpp-it 55: [rule 1] pooled CI of second - naked eye does not lie above 0; [rule 4] sign test p = 0.00639 (J++ 54+/22-, naked 47+/25-), Holm-adjusted 0.083 over 14 tests, not below 0.025; [rule 5] pooled second - J++ is +0.045, below the minimum 0.05; [rule 5] pooled second - naked eye is +0.009, below the minimum 0.05
- jpp-it 56-57: [rule 1] pooled CI of second - J++ does not lie above 0; [rule 1] pooled CI of second - naked eye does not lie above 0; [rule 2] chat-begun: significantly worse than J++; [rule 2] chat-begun: significantly worse than naked eye; [rule 2] chat-no-answer: significantly worse than J++; [rule 2] chat-no-answer: significantly worse than naked eye; [rule 2] mid-reply: significantly worse than J++; [rule 2] mid-reply: significantly worse than naked eye; [rule 3] stuck-word share above 25% at 57 chat-no-answer (0.46, ' same'), 57 chat-begun (0.29, ' same'), 57 mid-reply (0.30, ' same'); [rule 4] sign test p = 1 (J++ 41+/100-, naked 43+/106-), Holm-adjusted 1 over 14 tests, not below 0.025; [rule 5] pooled second - J++ is -0.107, below the minimum 0.05; [rule 5] pooled second - naked eye is -0.122, below the minimum 0.05
- jpp-it 58: [rule 1] pooled CI of second - J++ does not lie above 0; [rule 1] pooled CI of second - naked eye does not lie above 0; [rule 2] chat-begun: significantly worse than J++; [rule 2] chat-begun: significantly worse than naked eye; [rule 2] chat-no-answer: significantly worse than J++; [rule 2] chat-no-answer: significantly worse than naked eye; [rule 2] mid-reply: significantly worse than J++; [rule 2] mid-reply: significantly worse than naked eye; [rule 3] stuck-word share above 25% at 58 chat-no-answer (0.56, ' own'), 58 chat-begun (0.34, ' own'); [rule 4] sign test p = 1 (J++ 14+/109-, naked 10+/118-), Holm-adjusted 1 over 14 tests, not below 0.025; [rule 5] pooled second - J++ is -0.221, below the minimum 0.05; [rule 5] pooled second - naked eye is -0.247, below the minimum 0.05
- rl-it 30-39: [rule 1] pooled CI of second - J++ does not lie above 0; [rule 1] pooled CI of second - naked eye does not lie above 0; [rule 2] chat-no-answer: significantly worse than J++; [rule 2] chat-no-answer: significantly worse than naked eye; [rule 2] mid-reply: significantly worse than naked eye; [rule 3] stuck-word share above 25% at 30 chat-no-answer (0.69, ' own'), 31 chat-no-answer (0.66, ' own'), 32 chat-no-answer (0.53, ' own'), 33 chat-no-answer (0.59, ' own'), 34 chat-no-answer (0.66, ' own'), 35 chat-no-answer (0.64, ' own'), 36 chat-no-answer (0.80, ' own'), 37 chat-no-answer (0.81, ' own'), 37 mid-reply (0.25, ' same'), 38 chat-no-answer (0.74, ' own'), 39 chat-no-answer (0.78, ' own'), 39 mid-reply (0.33, ' same'); [rule 4] sign test p = 1 (J++ 26+/43-, naked 18+/75-), Holm-adjusted 1 over 14 tests, not below 0.025; [rule 5] pooled second - J++ is -0.009, below the minimum 0.05; [rule 5] pooled second - naked eye is -0.022, below the minimum 0.05
- rl-it 40-49: [rule 1] pooled CI of second - J++ does not lie above 0; [rule 1] pooled CI of second - naked eye does not lie above 0; [rule 2] chat-begun: significantly worse than naked eye; [rule 2] chat-no-answer: significantly worse than naked eye; [rule 2] mid-reply: significantly worse than J++; [rule 2] mid-reply: significantly worse than naked eye; [rule 3] stuck-word share above 25% at 40 chat-no-answer (0.50, ' own'), 41 chat-no-answer (0.50, ' same'), 41 chat-begun (0.28, ' same'), 41 mid-reply (0.49, ' same'), 42 chat-no-answer (0.45, ' own'), 42 mid-reply (0.44, ' same'), 43 chat-no-answer (0.47, ' own'), 43 mid-reply (0.42, ' same'), 44 chat-no-answer (0.38, ' own'), 44 mid-reply (0.37, ' same'), 45 chat-no-answer (0.33, ' own'), 45 mid-reply (0.35, ' same'), 46 chat-no-answer (0.35, ' own'), 47 chat-no-answer (0.44, ' same'), 47 chat-begun (0.29, ' same'), 47 mid-reply (0.41, ' same'), 48 chat-no-answer (0.68, ' same'), 48 chat-begun (0.49, ' same'), 48 mid-reply (0.56, ' same'), 49 chat-no-answer (0.53, ' same'), 49 chat-begun (0.31, ' same'), 49 mid-reply (0.46, ' same'); [rule 4] sign test p = 1 (J++ 67+/99-, naked 31+/123-), Holm-adjusted 1 over 14 tests, not below 0.025; [rule 5] pooled second - J++ is -0.014, below the minimum 0.05; [rule 5] pooled second - naked eye is -0.091, below the minimum 0.05
- rl-it 50: [rule 1] pooled CI of second - J++ does not lie above 0; [rule 1] pooled CI of second - naked eye does not lie above 0; [rule 2] chat-no-answer: significantly worse than naked eye; [rule 2] mid-reply: significantly worse than naked eye; [rule 3] stuck-word share above 25% at 50 chat-no-answer (0.39, ' same'), 50 mid-reply (0.34, ' same'); [rule 4] sign test p = 1 (J++ 58+/41-, naked 31+/62-), Holm-adjusted 1 over 14 tests, not below 0.025; [rule 5] pooled second - J++ is +0.034, below the minimum 0.05; [rule 5] pooled second - naked eye is -0.074, below the minimum 0.05
- rl-it 51-54: [rule 1] pooled CI of second - naked eye does not lie above 0; [rule 3] stuck-word share above 25% at 51 chat-no-answer (0.35, ' same'), 51 mid-reply (0.25, ' same'), 52 chat-no-answer (0.41, ' same'), 52 mid-reply (0.26, ' same'), 53 chat-no-answer (0.26, ' same'); [rule 4] sign test p = 0.273 (J++ 94+/60-, naked 71+/63-), Holm-adjusted 1 over 14 tests, not below 0.025; [rule 5] pooled second - J++ is +0.039, below the minimum 0.05; [rule 5] pooled second - naked eye is +0.003, below the minimum 0.05
- rl-it 55: [rule 1] pooled CI of second - naked eye does not lie above 0; [rule 5] pooled second - naked eye is +0.031, below the minimum 0.05
- rl-it 56-57: [rule 1] pooled CI of second - J++ does not lie above 0; [rule 1] pooled CI of second - naked eye does not lie above 0; [rule 2] chat-no-answer: significantly worse than J++; [rule 2] chat-no-answer: significantly worse than naked eye; [rule 2] mid-reply: significantly worse than J++; [rule 2] mid-reply: significantly worse than naked eye; [rule 3] stuck-word share above 25% at 56 chat-no-answer (0.42, ' own'), 57 chat-no-answer (0.33, ' own'), 57 chat-begun (0.39, ' same'), 57 mid-reply (0.33, ' same'); [rule 4] sign test p = 1 (J++ 47+/86-, naked 45+/87-), Holm-adjusted 1 over 14 tests, not below 0.025; [rule 5] pooled second - J++ is -0.052, below the minimum 0.05; [rule 5] pooled second - naked eye is -0.068, below the minimum 0.05
- rl-it 58: [rule 1] pooled CI of second - J++ does not lie above 0; [rule 1] pooled CI of second - naked eye does not lie above 0; [rule 2] chat-begun: significantly worse than J++; [rule 2] chat-begun: significantly worse than naked eye; [rule 2] chat-no-answer: significantly worse than J++; [rule 2] chat-no-answer: significantly worse than naked eye; [rule 2] mid-reply: significantly worse than J++; [rule 2] mid-reply: significantly worse than naked eye; [rule 3] stuck-word share above 25% at 58 chat-no-answer (0.36, ' own'), 58 chat-begun (0.39, ' same'), 58 mid-reply (0.33, ' same'); [rule 4] sign test p = 1 (J++ 16+/108-, naked 11+/113-), Holm-adjusted 1 over 14 tests, not below 0.025; [rule 5] pooled second - J++ is -0.212, below the minimum 0.05; [rule 5] pooled second - naked eye is -0.237, below the minimum 0.05

Replication on R2's prompts (R3_PREREG.md §8.6, decides nothing): passes. 519 of 519 of R2's read positions found again; filtered top-1 the same as R2's: jpp 1.0, naive 1.0, r1 1.0, miru 1.0 (tolerance 0.99); R2's design gives R2_03's verdicts: yes.

Secondary readings (R3_PREREG.md §1.2, §8; they decide nothing):

H2, agreement with base J++ in q10 (98 token-pair directions, 95% interval; loud beside), against R1's instruct covectors vs J++ (low) and J++ vs JLens (high), as Q0 measured them:

| lens | band | vs J++ q10 | loud | low | high | split-half q10 | reading |
|---|---|---|---|---|---|---|---|
| jpp-it | L30-39 | 0.336 [0.321, 0.350] | 0.385 | 0.2679 | 0.7666 | 0.988 | both |
| jpp-it | L40-49 | 0.293 [0.279, 0.307] | 0.613 | 0.2883 | 0.6846 | 0.987 | both |
| jpp-it | L50-58 | 0.293 [0.281, 0.305] | 0.626 | 0.2868 | 0.6574 | 0.994 | both |
| rl-it | L30-39 | 0.425 [0.411, 0.440] | 0.379 | 0.2679 | 0.7666 | 0.998 | both |
| rl-it | L40-49 | 0.325 [0.310, 0.341] | 0.585 | 0.2883 | 0.6846 | 0.998 | both |
| rl-it | L50-58 | 0.299 [0.286, 0.312] | 0.642 | 0.2868 | 0.6574 | 0.999 | both |

H3, the share of a lens's energy in its own top 100 output directions in y = fnw * output (jpp-it and rl-it also split-half debiased), against Miru's (plain J-lens, instruct):

| layer | jpp-it (debiased) | rl-it (debiased) | Miru | JLens | base J++ | jpp-it | rl-it |
|---|---|---|---|---|---|---|---|
| 30 | 0.899 (0.904) | 0.920 (0.921) | 0.962 | 0.880 | 0.775 | refuted | refuted |
| 35 | 0.906 (0.913) | 0.919 (0.920) | 0.964 | 0.887 | 0.760 | refuted | refuted |
| 40 | 0.905 (0.881) | 0.909 (0.910) | 0.942 | 0.861 | 0.813 | refuted | refuted |
| 45 | 0.923 (0.925) | 0.920 (0.920) | 0.948 | 0.914 | 0.889 | refuted | refuted |
| 50 | 0.907 (0.913) | 0.916 (0.917) | 0.937 | 0.906 | 0.902 | refuted | refuted |
| 55 | 0.911 (0.912) | 0.911 (0.912) | 0.930 | 0.897 | 0.909 | refuted | refuted |
| 58 | 0.944 (0.946) | 0.946 (0.946) | 0.960 | 0.960 | 0.967 | refuted | refuted |

H5, jpp-it - rl-it, filtered top-1, pooled (95% interval; Bonferroni over seven bands, 99.29%):

| band | jpp-it - rl-it | 95% | Bonferroni | reading |
|---|---|---|---|---|
| 30-39 | -0.003 | [-0.007, -0.001] | [-0.008, +0.001] | not shown |
| 40-49 | +0.015 | [+0.001, +0.030] | [-0.005, +0.035] | not shown |
| 50 | +0.030 | [+0.002, +0.057] | [-0.010, +0.068] | not shown |
| 51-54 | -0.010 | [-0.029, +0.008] | [-0.037, +0.015] | not shown |
| 55 | -0.021 | [-0.043, +0.000] | [-0.053, +0.006] | not shown |
| 56-57 | -0.055 | [-0.086, -0.025] | [-0.099, -0.017] | hurts |
| 58 | -0.009 | [-0.042, +0.022] | [-0.057, +0.034] | not shown |

H5 refuted: no band reads 'helps'.

Sensitivity analyses (pre-registered; the decision above stands):

| analysis | served | verdicts that change |
|---|---|---|
| S1 (without the prompts about the model itself (R1's fit set had many self-questions)) | jpp-it -; rl-it - | none |
| S2 (without the prompts that paraphrase or share an answer topic with a fit source (R3_common.FIT_OVERLAP: WikiText records, J++ items, R1's fit set)) | jpp-it -; rl-it - | none |
| S3 (without the bare condition (raw text puts this instruct model off-distribution)) | jpp-it -; rl-it - | none |
| S4 (without positions whose content target is a list numeral after a newline or the token after a code fence) | jpp-it -; rl-it - | none |
| S5 (every readout filtered by J++'s no-letter-or-digit rule instead of R2's filter (D7)) | jpp-it -; rl-it - | none |

---

### Context of the reading (code layers)

- **Counts.** 320 prompts and 1,200 read positions:
  - chat-no-answer, chat-begun and bare: 80 prompts and 80 positions each;
  - mid-reply: 80 prompts and 480 positions, 475 of them with a content target;
  - mid-reply-content-next: 240 positions, reported only.

  In the bare condition, 73 positions have a content target. 30 of those are raw-text collapse words (`own`, `same`) and
  are left out, so only 43 bare prompts are valid. That makes **283 valid prompts**, against the 292 §7.3 expected, so
  the pooled intervals are a little wider than planned. No formatting-like target occurred, so S4 left nothing out.
- **Agreement with the next token.** At code 59, the reading forward's top-1 agrees with the next token at 1.000
  (chat-no-answer), 1.000 (chat-begun), 0.988 (bare) and 0.994 (mid-reply). No condition is below 0.95 (§9).
- **The ceiling.** The model's own filtered top word at code 59 is its next content word at 0.824 [0.788, 0.858],
  pooled. That is the most a same-step readout can reach.
- **A constant guess.** Each condition's most common target word, leave-one-out, scores 0.140 [0.104, 0.175] pooled.
  - Both refits score below that at code 30-39: jpp-it 0.007 and rl-it 0.011. So do base J++ (0.020) and the naked eye
    (0.033). No readout names the next content word there.
  - Both refits are below the naked eye at code 40-49 (0.178 and 0.163 against 0.253) and at 50.
- **The unfiltered metric** (top-1 = the next token, reported for transparency, not used): here both refits score above base
  J++ and the naked eye at code 40-57. For jpp-it at 40-49 the figures are 0.193 against 0.077 and 0.125. At code 58
  they are below both. Their gain, where there is one, is in the next word-piece, formatting included, not in the next
  content word, which is what the rule scores.
- **Stuck words.** The refits' stuck words are mostly the collapse words ` own` and ` same`, in chat-no-answer and
  mid-reply. For example, rl-it reads ` own` at 53-81% of chat-no-answer positions at every layer of 30-39. Rule 3
  fails for both refits at six of the seven bands, every band but 55.

### H1 (decides): refuted at every band

Under §1.2, H1 is refuted at a band when no candidate passes there. No candidate passes at any band, so H1 is refuted
at all seven.

- **The closest case is rl-it at code 55.** It fails rules 1 and 5 only. It is +0.066 [+0.029, +0.102] over J++ but
  only +0.030 [-0.009, +0.072] over the naked eye. Its sign test passes Holm (0.0064).
- **jpp-it at 55** is +0.045 over J++ and +0.009 over the naked eye. Holm gives 0.083.
- **At code 50,** jpp-it is +0.064 [+0.015, +0.113] over J++ but −0.044 [−0.087, −0.004] under the naked eye.

§1.2's own limit applies here. This does not show that no instruct lens could pass. It shows that J++'s recipe, as the
code at `d7efc0de` runs it on 64 WikiText prompts with the deviations of §2.2, does not pass. The same holds for its
pooled R-lens.

**Precedence (§7.2)** never came into play, because no band had a pass.

### Reference lenses, outside the Holm family (decide nothing, never candidates)

- **R1's lens (R1_18) passes all five rules on R3's 320 prompts at code 55 and 58,** with raw p:
  - at 55: +0.107 [+0.070, +0.144] over J++ and +0.071 [+0.035, +0.109] over the naked eye, raw p 1.6e-10;
  - at 58: +0.083 [+0.038, +0.123] and +0.058 [+0.024, +0.091], raw p 3.1e-08.

  At code 50 it fails rule 5 only (+0.046 over the naked eye).
  - **R2's prompts.** On R2's 220 prompts, R2's design still gives R2_03's verdicts: rule 5 fails at code 58 (+0.046),
    rules 1 and 5 at 55. R3's design on those prompts gives the same failed rules.
  - **What this is and is not.** R1 is not one of R3's candidates (§7.2). Its p values sit outside the 14-test family,
    and both R2 and R3 withhold it. This result is a reference reading on fresh prompts, not a recommendation. Serving
    R1 would need its own registered test.
- **Miru** passes at no band. At code 58 it collapses, as in R2: 0.024, with ` same` stuck at 1.00 in chat-begun.
- **jpp-it-full-macro** passes at no band. Poetry kept 9 scoreable items and 1 fitting item, so it failed D16's floor.
  The full macro is therefore the primary one, and `R3_04` gave this lens jpp-it's weights. Every number for it equals
  jpp-it's, except its p, which is raw and outside Holm.

### The replication on R2's prompts (§8.6): passes

- **(a) Positions.** All 519 of R2's 519 read positions were found again: their prompt, condition, position, draws,
  next token, content target and reply.
- **(b) Readouts.** The filtered top-1 equals R2's at every (position, layer) pair of code 40-59, with no mismatch:
  - J++: 9,861 of 9,861 pairs;
  - the naked eye: 10,380 of 10,380;
  - R1: 1,557 of 1,557;
  - Miru: 9,861 of 9,861.

  The unfiltered top-1 also matches at every pair.
- **(c) Verdicts.** R2's design on the replication file gives R2_03's verdicts and decision exactly.

R3's design on R2's prompts serves nothing either. Both refits fail rules 1, 4 and 5 at code 55 and fail every rule
at their other bands. None of S1-S5 changes a verdict. Its seeds and Holm family are R3's (§8.6), so its intervals
are not R2's.

So R3's own reading path (D21) gives R2's readings. Nothing in §8.6 holds back a pass, and there was none.

### H2: agreement with base J++ — "both", for jpp-it and rl-it, in every band

The table gives q10 cosines over the 98 token-pair directions, with 95% intervals. The bands are in code layers.

| lens | vs | code 30-39 | 40-49 | 50-58 |
|---|---|---|---|---|
| jpp-it | base J++ | 0.336 [0.321, 0.350] | 0.293 [0.279, 0.307] | 0.293 [0.281, 0.305] |
| rl-it | base J++ | 0.425 [0.411, 0.440] | 0.325 [0.310, 0.341] | 0.299 [0.286, 0.312] |
| low yardstick | R1's IT covectors vs J++ (Q0) | 0.268 | 0.288 | 0.287 |
| high yardstick | J++ vs JLens (Q0) | 0.767 | 0.685 | 0.657 |
| jpp-it | Miru | 0.534 | 0.585 | 0.683 |
| rl-it | Miru | 0.699 | 0.737 | 0.755 |
| jpp-it | JLens | 0.269 | 0.319 | 0.356 |
| rl-it | JLens | 0.382 | 0.377 | 0.383 |
| jpp-it | R1's IT covectors | 0.342 | 0.443 | 0.642 |
| rl-it | R1's IT covectors | 0.577 | 0.651 | 0.718 |
| split half | jpp-it A vs B | 0.988 | 0.987 | 0.994 |
| split half | rl-it A vs B | 0.998 | 0.998 | 0.999 |

**The reading.** In every band, each interval's upper end is above the low yardstick and its lower end is below the
high one. So neither claim is refuted, and the reading is **"both"** (§1.2). §1.2's two-decimal figures (0.27 / 0.29 /
0.29 and 0.77 / 0.68 / 0.66) give the same reading (D31 b).

**What this means.** Both refits agree with base J++ much as R1's instruct covectors did. They sit just above the low
yardstick at code 40-58: jpp-it's lower ends are 0.279 and 0.281, against 0.288 and 0.287. And they sit far below the
high one. The split halves are near 1, so the low agreement is not fit noise.

**What H2 cannot tell apart.** What differs from base J++ is the checkpoint together with the code version (D13).
For jpp-it, the items and weights (D8, D11) differ too. H2 lays nothing on the checkpoint alone.

**The weighting's share.** jpp-it is further from base J++ than rl-it at 30-39 and 40-49, and further from Miru and
from R1's covectors in every band. So the weighting moves the lens away from every other lens, not towards base J++.

### H3: concentration — refuted at all seven judged layers, for jpp-it and rl-it

The table gives the share of a lens's energy in its own top 100 output directions, in y = diag(fnw) J, at code layers.

| layer | jpp-it plain (debiased) | rl-it plain (debiased) | Miru | JLens | base J++ | jpp-it in J++'s top 100 | rl-it in J++'s top 100 |
|---|---|---|---|---|---|---|---|
| 30 | 0.899 (0.904) | 0.920 (0.921) | 0.962 | 0.880 | 0.775 | 0.249 | 0.244 |
| 35 | 0.906 (0.913) | 0.919 (0.920) | 0.964 | 0.887 | 0.760 | 0.175 | 0.210 |
| 40 | 0.905 (0.881) | 0.909 (0.910) | 0.942 | 0.861 | 0.813 | 0.184 | 0.219 |
| 45 | 0.923 (0.925) | 0.920 (0.920) | 0.948 | 0.914 | 0.889 | 0.147 | 0.208 |
| 50 | 0.907 (0.913) | 0.916 (0.917) | 0.937 | 0.906 | 0.902 | 0.163 | 0.203 |
| 55 | 0.911 (0.912) | 0.911 (0.912) | 0.930 | 0.897 | 0.909 | 0.194 | 0.198 |
| 58 | 0.944 (0.946) | 0.946 (0.946) | 0.960 | 0.960 | 0.967 | 0.126 | 0.134 |

**The reading.** At each judged layer, both lenses' debiased shares are at or below Miru's plain share, so §1.2 reads
**"refuted"**. Their plain shares are below Miru's too, so "supported" never applies, and D31 (d) does not arise.

**What H3 shows.** The recipe, as R3 runs it, does not concentrate a lens's energy beyond what Miru's plain instruct
J-lens shows. All the instruct J-lenses, these two and Miru, hold 90-96% of their energy in their top 100 directions.
That is at least as much as JLens (0.86-0.96), and more than base J++ at code 30-40 (0.76-0.81).

**Base J++'s subspace.** Only 13-25% of the refits' energy lies inside base J++'s top-100 subspace, against Miru's
15-21% and JLens's 40-61%. In this sense, too, they do not point where base J++ points.

**The debiasing.** jpp-it's debiased share falls below its plain share at code 38 (0.882 against 0.927, see
`R3_06_analysis.json`) and at code 40. The cross product of its halves is noisier than rl-it's. That changes no
reading, because the plain share is below Miru's as well.

### H4: fidelity on R1_10's 4,808 held-out positions

R1_10's top-1 is reproduced exactly for J++, JLens and the naked eye at every layer (maximum difference 0.0000,
tolerance 0.01), so `ok: true`. No H4 number carries D23's caution.

The table gives top-1 agreement with the model's own final readout, over all positions, at code layers. The last two
rows are the difference from base J++, with 95% sequence-bootstrap intervals (48 sequences, 2,000 resamples).

| readout | 40 | 45 | 50 | 55 | 58 |
|---|---|---|---|---|---|
| base J++ | 0.079 | 0.139 | 0.159 | 0.260 | 0.491 |
| jpp-it | 0.094 | 0.141 | 0.248 | 0.351 | 0.483 |
| rl-it | 0.099 | 0.167 | 0.226 | 0.356 | 0.468 |
| naked eye | 0.115 | 0.181 | 0.207 | 0.246 | 0.500 |
| JLens | 0.061 | 0.102 | 0.194 | 0.258 | 0.484 |
| Miru | 0.130 | 0.209 | 0.279 | 0.368 | 0.140 |
| R1 | | | 0.297 | 0.370 | 0.619 |
| jpp-it − J++ | +0.015 [+0.008, +0.023] | +0.001 [−0.007, +0.010] | +0.090 [+0.074, +0.106] | +0.092 [+0.075, +0.107] | −0.008 [−0.040, +0.026] |
| rl-it − J++ | +0.020 [+0.012, +0.029] | +0.028 [+0.019, +0.037] | +0.068 [+0.053, +0.082] | +0.097 [+0.081, +0.112] | −0.022 [−0.057, +0.014] |

**The reading (§1.2, coded in `R3_05_eval.h4_tables`).**

- **jpp-it:** supported at code 40, 50 and 55; refuted at 45 and 58.
- **rl-it:** supported at 40, 45, 50 and 55; refuted at 58.

**Beside R1's assistant-turn figures** (2,452 positions, §8.3), at code 40 / 45 / 50 / 55 / 58:

- base J++ here: 0.126 / 0.220 / 0.245 / 0.385 / 0.693, against R1's 0.13 / 0.22 / 0.25 / 0.39 / 0.69;
- jpp-it: 0.151 / 0.225 / 0.371 / 0.526 / 0.589;
- rl-it: 0.162 / 0.248 / 0.326 / 0.528 / 0.573;
- R1's instruct full matrix: 0.21 / 0.37 / 0.45 / 0.55 / 0.85 in R1, and 0.447 / 0.551 / 0.850 here at 50 / 55 / 58.

**Where the refits stand.** They are closer to the model's own readout than base J++ at code 50-55 and fall back at
58. At 58 they are 0.10-0.12 below J++ on assistant turns. The band test shows the same at 58.

**KL.** Calibrated KL is reported in `R3_05_h4.json` and decides nothing.

### H5: Jacobian filtering on instruct — refuted

The script's table above gives jpp-it − rl-it on the filtered top-1. No band reads "helps" on the pooled Bonferroni
interval.

- **"Hurts" at code 56-57:** −0.055, Bonferroni [−0.099, −0.017].
- **"Not shown" at the other six bands.** At code 40-49 and 50 the 95% interval lies above 0, and at 30-39 below 0,
  but in none of them does the Bonferroni interval.

So on Gemma 4's instruct model, R3 does not reproduce J++'s Qwen finding (55.2 against 43.5 pooled).

**J++'s own held-out recall@10** (primary macro, 225 held-out items, 355 pairs, J++'s filter) decides nothing. It
favours jpp-it, whose weights were fitted on the same evals' fitting items. At code 30 / 38 / 45 / 52:

| lens | 30 | 38 | 45 | 52 |
|---|---|---|---|---|
| jpp-it | 0.200 | 0.236 | 0.358 | 0.393 |
| rl-it | 0.148 | 0.130 | 0.292 | 0.381 |
| base J++ | 0.224 | 0.109 | 0.384 | 0.406 |
| naked eye | 0.012 | 0.059 | 0.370 | 0.362 |
| Miru | 0.000 | 0.003 | 0.036 | 0.352 |

On J++'s own items, too, jpp-it is below base J++ at three of the four layers. Base J++'s figures may include its
own fitting items (§8.3).

### H6: two bounded differences — "smaller" in every band

The table gives q10 cosines over the 98 token-pair directions, with 95% intervals, at code layers. Every Q1 number
here comes from J++'s 64 prompts, not Miru's 425 (the D22 fallback), so the prompts and their number differ too.

| code band | Q1 vs Miru (b) | Miru vs JLens (a, Q0) | Q1 split half (s) | reading |
|---|---|---|---|---|
| 30-39 | 0.789 [0.776, 0.801] | 0.368 [0.348, 0.390] | 0.995 | smaller |
| 40-49 | 0.796 [0.786, 0.806] | 0.390 [0.371, 0.409] | 0.997 | smaller |
| 50-57 | 0.808 [0.797, 0.818] | 0.409 [0.390, 0.428] | 0.998 | smaller |

**Loud coordinates** read "smaller" in all three bands too. b is 0.734 / 0.835 / 0.862, against a at 0.258 / 0.560 /
0.641.

**The triangle's other sides,** in q10 at the three bands:

- Q1 vs JLens: 0.325 / 0.347 / 0.370;
- Q1 vs base J++: 0.303 / 0.267 / 0.274.

**Code 58 is reported apart and not read.** Q1 vs Miru is 0.824 [0.812, 0.835] there, and the split half 0.999.

**The reading.** The difference Q1 against Miru measures is smaller than the one Miru against JLens measures. The
first is quantisation, with the QAT checkpoint's further training, R3's code against Miru Tracer's and, under the
fallback, other and fewer prompts. The second is fine-tuning, with Miru Tracer's code against
anthropics/jacobian-lens and the two fits' prompts. **H6 does not separate fine-tuning from quantisation (§1.1).**

**Beside H2.** A plain J-lens on the 4-bit model agrees with Miru at about 0.8. Base J++ agrees with every instruct
lens at about 0.3, Q1 included. So the instruct lenses form one group and the base lenses another.

### Sensitivity analyses (beside the decision, which stands)

S1 drops the 32 prompts about the model and S2 the 83 FIT_OVERLAP prompts. S3 drops the bare condition, and S4 drops
formatting-like targets (there were none). S5 uses J++'s exclusion list for every readout. Each reran the whole rule,
and none serves a candidate at any band or changes any lens-band verdict (the script's table above). The same holds
on R2's prompts under R3's design.

### Deviations

D1-D32 are departures fixed before data (§2.2). Three changes were made after R3 GPU data existed. §0 asks for the
result under the code as registered beside each one:

- **D34, the rules' patch on offloaded modules (A13).** The first 12B dry run (10:25, HEAD `ffdf10d`) failed under the
  registered code. The offloaded norms read meta weights, so the shards and the rules-on parity stopped. Its output is
  at `e1733fe`.
  - *Under the code as registered:* the dry run fails, the pilot refuses to start without it, and R3 measures nothing.
  - On the 31B the change is inert, because its modules carry no accelerate hook.
- **D33, the identity rule's arithmetic in fp32 (A14).** The first pilot (10:33-10:35, HEAD `e1733fe`, output at
  `8996fb5`) failed gate (a). Under the rules the top-1 changed at 90 of 2,048 positions, where 2 are allowed. Parity
  passed (min cos 0.999943 plain, 0.999983 rules on). The pilot wrote "stop".
  - *Under the code as registered:* the pilot says "stop", and no router, fit, weight, evaluation or Q1 job runs.
  - Context, not a gate: jpp_lens's own patcher changes the same 90 positions (`R3_00_jpp_identity.json`). Under D33
    none change.
- **D35, the model's name in the weight fit's identity check.** The first `R3_04_weights.py --run` (17:58-18:06,
  HEAD `f6266a9`, 450 s, failed in `R3_gpu_runs.jsonl`) did the following before it failed:
  - graded the items;
  - saved the residuals of 288 scoreable items;
  - passed both of D16's stops.

  It then failed at the first layer's weight fit, on jpp_lens's `check_model_matches_config`, and wrote no lens file.
  - *Under the code as registered:* no jpp-it, rl-it or half-lens file exists. R3_05, H1 and every secondary reading
    except H6 cannot run.
  - The fix (`bbc2f4d`) accepts only that one pairing of names. The rerun (632.5 s) passed D16's floor and both stops.
  - The floor kept 4 evals and 84 fitting pairs. Two graded evals stayed (multihop, multilingual) and poetry dropped.
- **No deviation in this step.** R3_06 and Q1's analysis ran as registered, with no code change and no
  `--allow-uncommitted`.
- **The run order.** The order of the GPU jobs was §10.1 step 3's: the weights, R3_05, then Q1. Neither of the last
  two reads the other's output.

### The GPU budget

R3 used **5.54 of the 8 approved GPU-hours**, counting every finished or failed run since the first dry run:

- **`R3_gpu_runs.jsonl`:** 18,963 s. This includes the 450 s of the failed weight run.
  - the dry runs and pilots: 642 s;
  - the router: 2,688 s;
  - the shards: 14,184 s;
  - the weights: 1,083 s;
  - Q1: 367 s.
- **`R3_05_runs.jsonl`:** 984 s for R3_05.

**The card at the end.** No R3 process and no Theodolite server is running. Firefox is not running, and 29,764 MiB of
the card is free; only the desktop compositor holds memory.

### Limits

- **The shared fit set.** Every refit is fitted on J++'s 64 WikiText prose prompts of 128 word-pieces. It is read on
  chat and mid-reply positions of an instruct model. The weights rest on 84 fitting pairs, of which multihop gives 5
  items. A refit on chat text, or on more prompts, was not tested (§3.1 excludes it), so H1's refutation is a
  statement about this recipe on this corpus.
- **Code 58.** Why both refits fall below base J++ at code 56-58 is not known from R3's data. The candidates are the
  estimator's sum over all later targets (D12), the code version (D13), the corpus or the checkpoint, and R3 cannot
  separate them. The refits' stuck collapse words (` own`, ` same`) resemble Miru's collapse at 58, but R3 does not
  test that link.
- **Power.** The 283 valid prompts are fewer than the 292 planned. With them, a true gain near 0.05 would most
  likely fail rule 4 (§7.3). That matters little here: the only bands where a candidate's pooled gain over the naked
  eye was above 0 are 51-54 and 55, and none of those gains was larger than +0.030.
- **R1's reference pass** is outside the registered family. It rests on a different prompt set from R2's, where the
  same lens failed rule 5 by 0.004. It is a finding to register, not to act on.
- **H2 and H3.** Each compares lenses that differ in more than one thing: checkpoint, code version, items, prompts.
  No reading names a cause (§1.1). H3's debiasing was found biased upward (D31), and it behaved noisily for jpp-it.
- **H4** uses R1_10's 48 sequences only. Its chats are R1's, and the prose is not WikiText.
- **H6** uses the fallback prompts. Miru's own split-half ceiling is unknown. Q1 uses directional covectors for R1's
  128 directions, not a full matrix (D22).
- **Base J++** came from older code whose router, weights and corpus revision are not recorded (D13, D17). So "base
  J++ against the refit" changes more than the checkpoint, everywhere in this section.

**Files.** `R3_06_analysis.json` holds per-condition intervals, per-layer tables, the exact-id and top-5 variants, the
content-next draw, the function-word split, the sensitivity details and the full replication. The rest of the
evidence is in `R3_05_h4.json` (H4), `R3_05_recall.json` (recall), `R3_Q1_jlens.json` (H6) and `R3_04_weights.json`
(the floor and the weights). Bootstrap: 2,000 prompt-level resamples within conditions, seed `[20261022, condition
index]`.

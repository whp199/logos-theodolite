# Roadmap

*What is still open in Logos Theodolite, gathered from the places that already name it: the README’s fault table, the
study’s report and its errata, the pages’ own “not yet” marks, and the records of R1’s instruct refit, R2’s test of
what to serve and R3’s test of J++’s recipe. Nothing here has a date or an owner. Layers are counted 1–60, as on the
pages; the study’s reports count from 0, so their L50 is layer 51 here.*

<sub>Gathered on 21 October 2026, at `3cbc4b5`; gathered again on 22 October 2026, at `75ed78a`, after the first round of
mends (the interface’s defaults, the label gate, the causal test and the smaller pieces) and R3’s registration; section
i brought up to date at `c15b2d3`, with R3’s result.</sub>

<br>

## i · the lens, the largest fault left

The J++ lens was fitted on the base model (`google/gemma-4-31B`, bf16) and is read here on the instruct model (4-bit
QAT), where it agrees with the model’s own average gradients for the pronoun direction at only cos 0.15–0.33 (layers
31–59; [README §iv](README.md#iv--how-far-to-trust-it), fault 9). Everything that reads through the lens inherits it: the
words, the gathered/recalled split, the landmark bearings, the neuron labels, the nudge and the vault. The pages now say
so, at the head and foot of every page and in a red line beside each layer’s words, and `/api/health` and every sweep
carry a `lens` object; the reading itself is unchanged.

- **An instruct lens exists, and is not loaded.** R1 refitted a lens on this checkpoint
  ([`R1_IT_LENS_REPORT.md`](experiments/results/R1_IT_LENS_REPORT.md)). Its directional form is stable from layer 31; its
  full matrix exists at 11 layers and is usable at five of them (41, 46, 51, 56 and 59). Its files are gitignored, so they
  are not in the repository ([REPORT](experiments/REPORT.md), erratum E50).
- **The second reading is built, and not merged.** The local branch `it-second-reading` (17–18 October, not pushed) reads
  a second lens beside J++: R1’s at layers 51, 56 and 59, and the Miru community J-lens for gemma-4-31B-it at 41–50,
  52–55 and 57–58. Each lens file is checksum-pinned and loaded only at layers that `scripts/check_second_lenses.py` has
  passed, and the layer and instrument leaves mark the reading checked or provisional. Its own commit says it is not for
  merge as is.
- **R2 served nothing.** The pre-registered test of which layers to serve
  ([`R2_PREREG.md`](experiments/results/R2_PREREG.md) and
  [`R2_03_report_section.md`](experiments/results/R2_03_report_section.md), brought here with R3’s registration; the
  serving code and `scripts/check_second_lenses.py` stay on `it-second-reading`) passed no lens at any layer, so under
  its rules the second reading shows nothing anywhere. The nearest miss is R1’s lens at layer 59, which failed only the
  minimum-size rule (+0.046 over the naked eye, where 0.05 was required). Two of the pre-registered sensitivity checks,
  one without the prompts about the model itself and one without the prompts that share a topic with R1’s fit prompts,
  would serve R1’s lens at layer 56; the registered decision stands.
- **R3 served nothing either.** J++’s published recipe, as the code at jpp_lens `d7efc0de` runs it, was fitted on
  the 4-bit instruct checkpoint and judged by R2’s rules on 320 new prompts
  ([`R3_PREREG.md`](experiments/results/R3_PREREG.md),
  [`R3_06_report_section.md`](experiments/results/R3_06_report_section.md)). Neither refit, the full recipe or its
  pooled R-lens, passes at any of the seven bands from layer 31 to 59, so R3 recommends nothing to serve. The nearest
  miss is the pooled R-lens at layer 56: +0.066 over base J++, but only +0.031 [−0.009, +0.072] over the naked eye,
  short of the 0.05 minimum. At layers 57–59 both refits read worse than base J++ and the naked eye. That refutes H1
  for this recipe on J++’s 64 WikiText prompts: it does not show that no instruct lens could pass, and a refit on chat
  text or on more prompts is untested. Its secondary readings decide nothing:
  - J++’s weighted experts were not shown to beat the pooled R-lens at any band, and read worse at layers 57–58 (H5);
  - the recipe concentrated a lens no more than Miru’s plain instruct J-lens does (H3);
  - a plain J-lens fitted here on the 4-bit QAT checkpoint agrees with Miru’s bf16 instruct lens more closely than
    Miru agrees with the base JLens (H6, “smaller” in every band). Both comparisons also change the code and the
    prompts, and the first carries the QAT checkpoint’s further training too; Q1 ran on J++’s 64 prompts because
    Miru’s 425 could not be rebuilt. So H6 lays nothing on fine-tuning or quantisation alone (E50 still stands).

  To pass its own gate, R3 had to compute J++’s identity rule in fp32: in bf16 the rule as registered changed the
  model’s top word-piece at 90 of 2,048 positions (D33), and jpp_lens’s own patcher changes the same 90 (the context
  check, D36). Every change made after the first GPU output is filed as a deviation (D30, D31,
  D33–D36). The run used 5.54 of the 8 GPU-hours the user approved. Whether an instruct lens gives better neuron
  labels is still untested (E45).
- **A lead: R1’s lens on R3’s prompts.** R1’s lens, read for reference outside the decision and its Holm family, passed
  all five rules at layers 56 and 59 on R3’s 320 prompts: +0.071 [+0.035, +0.109] and +0.058 [+0.024, +0.091] over the
  naked eye. It was not a candidate, so this is not a recommendation. On R2’s 220 prompts the same lens still fails: at
  layer 59 by the minimum-size rule (+0.046), at layer 56 by rules 1 and 5. Serving it would need a test of its own,
  registered before it sees fresh prompts, under the rules below.
- **Rules for any lens served later**, from the errata:
  - confirm condition contrasts at layers 31–50 with an instruct reading (E49);
  - show pronoun readings at layers 51–59 only where J++ and the instruct reading agree (E49, E50);
  - keep the caution at layers 11–30 for every averaged lens: none tested was reliable there (E50);
  - re-validate the label gate of ii for the new lens: its AUC of 0.84 was measured for base-fitted J++ labels only
    (E47).

R3 serves nothing itself: a pass would only have been a recommendation to later serving work on a branch from `main`
(R3_PREREG §7.2). With no lens passed by R2 or R3, the serving work on `it-second-reading` stays there as R2’s record.
R3’s jobs refuse a `backend/` that differs from their registered runtime, `d5405ee` (R3_PREREG §0), so any rerun goes
from the branch `it-jpp-refit`, not from a `main` that carries this round’s mends. R3’s line cites to the README (D26
and A9, `README.md:407–408`) are to `d5405ee`’s README; on this tree that passage is the licence paragraph of README
§vii.

<br>

## ii · neuron labels

The study judged joining the pillars “neat as a question, oversold as built”, and named four conditions under which the
joined instrument earns its keep (REPORT §3.3, erratum E47). The Lineage page (iv) marks where each stands:

| | The condition | Where it stands |
|---|---|---|
| **1** | Labels grounded in what wakes a neuron: the passages that make each face fire, stored with it | **Not yet.** One input label still serves both faces (fault 7), and input-side labels carried only 6–8% of the signal. No passages are stored for either face |
| **2** | Output labels trusted only from layer 41 up | **Yes.** No page sets a label below layer 41 in its main reading; the apparatus keeps it, with the reason, and the server marks it (`label_check`, with the rule in `label_gate`). The vault's console shows the server's replies whole and says which labels the pages hold back |
| **3** | A label counts only when the lens and the naked eye agree | **Yes.** A label whose two readings share none of their first ten word-pieces is held back; the others show their count. The cut at none of ten is the instrument's (the study set none; in N1's data 7 of the 31 such labels within layers 41–58 were sound, against 160 of the 209 others, so as a yes/no sign the cut scores AUC 0.64, against the count's 0.84 as a graded score; labels sharing just one of ten were sound 16 times in 40, below the band's 167 in 240, `N1C_gate_cut.json`), and it applies only through the base-fitted J++ lens, within layers 41–59, the gate having been validated for base-fitted J++ labels on random neurons within 41–58 (E47). The demonstration sighting's labels and the vault's 24 recorded searches carry too few words for the count, so `experiments/scripts/N1G_01_recorded_label_checks.py` checked each ten a side on the GPU (`N1G_recorded_label_checks.json`): 46 of the sighting's 75 labelled faces and 98 of the vault's 240 rows are held back |
| **4** | Each label spot-checked by cause and effect | **In part.** A neuron’s folio puts its label to the test (“test it at this word-piece”, and “test it on the study’s 36 texts” beside it), as do the WebSocket action `causal_check`, `POST /api/scientist/causal_check` and the MCP tool `causal_spot_check` (`backend/causal_check.py`, ARCHITECTURE §4.5): N1A’s ±k · max\|a\| on the neuron at the read position, with the untouched row in the same forward, at k = 10 in the linear regime by default and a frozen-norm write as a choice, since 60× steering saturates the post-feed-forward norm at layers 21–51 (E47). Each test reports how straight its push stayed, how far it rose above the same test at k = 0.01, and its P@10 for the lens’s and the naked eye’s labels; only the test on the 36 texts compares with the study’s figures. `scripts/verify_causal_check.py` checks it against N1A’s and N1V’s numbers on the real model. It is still run one label at a time, when asked: nothing tests every label |

Cammarata’s practices for reading a neuron ([9] in [`REFERENCES.md`](REFERENCES.md)), as the Lineage page sets them
against the instrument, leave four more:

- read each face in long passages, not only its top word-pieces: you can write whole sentences into a neuron’s guess
  test, but no passages are stored (the same gap as 1);
- look upstream at what builds a neuron (downstream, the threads and “who listens” are there);
- keep the field notes with the vault: they last for the visit only;
- watch a neuron form across training checkpoints.

<br>

## iii · smaller pieces, now built

- **Landmarks: one word against another. Built.** A sweep can weigh one word against another at every layer, through
  the lens and by the naked eye, against the matched controls sent or against E3’s average state (`contrasts`,
  `contrast_backgrounds`; ARCHITECTURE §4.1–4.2): the contrast axes V7 suggests
  ([`V7_tool_bugs_NOTES.md`](experiments/results/V7_tool_bugs_NOTES.md)), read in log-odds beside the frame rather
  than as an axis of it. The landmarks page reads one in its section vii. Still open: through the lens the reading
  inherits the base-fitted J++ (i), so the page marks its layers with the bands every page draws for the lens’s words
  (1–20 the least trusted, where E3 (b) found the lens barely agreeing with the model’s own pronoun direction; 21–30
  orderings only; 31–50 to be confirmed with an instruct-fitted reading; 51–59 counting only where one agrees: errata
  E49 and E50), and the naked eye with its own background, which runs the other way (E3 (c)); the study measured the
  average state’s lean on its six pronoun pairs and on its noun pairs (woman–man and the like), where it leans toward
  the male words and is larger (E3 (c): at layers 41–50 a share of +1.4 on the pronouns and −3.1 on the nouns), not on
  one pair or on other words; E3’s own lens figures for the average state reproduce today only as a band mean at
  layers 41–50 (+2.35 against its +2.31, within 0.04): layer by layer they miss by 0.20 and 0.10 at layers 41 and 42,
  and by more than 0.04 at 33 of the 60 layers (median 0.18, at most 1.85 at layer 3), by the study’s own code as by
  the server, with the cause not established (by layer:
  [`E3_background_mu_check.json`](experiments/results/E3_background_mu_check.json)); and the page sends one contrast
  and one background at a time, where the server takes four and two. That E3’s average state is in the coordinates of
  the server’s `resid_post` is checked by the end-to-end check (item 12) on one of E3’s neutral passages.
- **The vault: an edit’s voice. Built.** An edit can be heard before and after on a question of the reader’s choosing,
  written the likeliest way under the one GPU lock (`voice`; ARCHITECTURE §4.6), and the vault page asks for it by
  default. The pages now call the “.patch MEMIT” edit an edit, a one-neuron column edit and not MEMIT; the code’s
  names and the statements are kept for old callers and the study’s scripts. Every vault query runs in Python; the
  `larql`/`vindex` binaries are called only for `--version` (README §iv). Still open: the voice is not offered through
  MCP, whose vault tool does not hold the GPU lock, nor for an edit typed in the console; it is one question, so UPDATE
  and INSERT are still tested by cause and effect only as far as that one answer goes (the study tested only DELETE,
  E6 §2.3). The edit tab’s own example edit is not heard on the page’s own question: its answer does not change, so a
  first reader hears nothing; the end-to-end check finds an edit that is heard, at layer 60, instead.
- **Thinking on bare text. Built.** Sent as a chat, the sighting with thinking turned on uses the template’s own switch;
  sent as bare text, the interface leaves it out, since bare text has no turn to think in (`think_bare="none"`), and
  the first pair names how thinking was switched on (`think_protocol`; null on later pairs). Still open: the server’s own default stays the
  old imitation, `<|think|>` set before the text, for old callers, and the question’s apparatus keeps it as a marked
  choice; the study recorded no fix for the bare case (the Lexicon’s cautions; E6 audit, item 11). Even in a chat the
  thinking sighting is read at the end of its own prompt, where the silent one is read at the close of its empty
  thought, unless an answer is begun (`think_read_token` names the word-piece it was read at).

<br>

## iv · the server’s defaults, and the interface’s

The server keeps its old defaults for old callers. The interface sends the mended settings for all four; parts of fault
3’s mend, the weighted rows, the per-layer frame and centring on a control, stay choices:

| | The fault | The server’s default | What the interface sends unless you change it | The mend |
|---|---|---|---|---|
| **1** | The nudge collapses the model | five layers at scale 650 | `direction="fnw"`, `mask_quiet`, `spare_bos`, `strength` 0.01 × ‖h‖ | in use on the nudge page |
| **3** | The survey frame depends on the order of the words | Gram–Schmidt on raw rows | `frame_squaring="lowdin"` on raw rows (a neuron-locked frame keeps Gram–Schmidt) | also `frame_fnw`, `frame_per_layer`, `centre_prompts` (off unless chosen) |
| **4** | The thread score C is a weak guide | C | the direct-path estimate | `thread_score="direct"` |
| **8** | Neurons ranked by \|a\|, not by the size of their write | by \|a\| | `pole_ranking="write"` | `pole_ranking="write"` |

Still open here:
- **The weighted rows.** V7 put the order-free frame’s remaining disagreement with the lens (about one reading in five) down
  to the raw rows, but did not measure a frame on the rows the model weighs, fnw ⊙ W<sub>U</sub>; the interface leaves
  `frame_fnw` off until an instrument check measures it.
- **Threads among the biggest writers.** V7 measured the direct-path score’s 0.87 on threads among neurons ranked by
  \|a\|; the threads now join the biggest writers, where it was not measured.
- In E3 (12 prompts, last word-piece; indicative only), at the seven layers it sampled, the fifteen biggest writers
  explained the same share of the feed-forward write as the fifteen loudest, within half a percentage point (10–41% at
  layers 21, 31, 41, 51 and 59; 70–74% at layers 3 and 11), so
  fault 8’s mend changes which neurons are named, not how much of the write they cover.

<br>

## v · loose ends from the first round

- **Where the label gate cuts.** In N1’s own sample within layers 41–58 (`N1C_gate_cut.json`, `cuts`), holding back
  labels that share one of ten or fewer would have held back 71 instead of 31 and kept 144 sound labels of 169 (85%),
  against 160 of 209 (77%) at the cut chosen, none of ten. A cut chosen on that sample would be judged on the data it
  was chosen from.
- **The README’s pictures.** The views in README §ii were taken before this round, with the reading options as first
  built and before the label gate; their captions say so. They want retaking from a live sighting.
- **References.** Section H of `REFERENCES.md` ([45] and [46], R3’s sources) was written from local files; its links
  were not opened.

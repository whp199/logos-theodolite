# Roadmap

*What is still open in Logos Theodolite, gathered from the places that already name it: the README’s fault table, the
study’s report and its errata, the pages’ own “not yet” marks, and the one branch that is not merged. Nothing here has a
date or an owner. Layers are counted 1–60, as on the pages; the study’s reports count from 0, so their L50 is layer 51
here.*

<sub>Gathered on 21 October 2026, at `3cbc4b5`.</sub>

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
- **R2 served nothing.** The pre-registered test of which layers to serve (`R2_PREREG.md` and `R2_03_report_section.md`
  in `experiments/results/` on that branch) passed no lens at any layer, so under its rules the second reading shows
  nothing anywhere. The nearest miss is R1’s lens at layer 59, which failed only the minimum-size rule (+0.046 over the
  naked eye, where 0.05 was required). Two of the pre-registered sensitivity checks, one without the prompts about the
  model itself and one without the prompts that share a topic with R1’s fit prompts, would serve R1’s lens at layer 56;
  the registered decision stands.
- **Not yet tried.** R1’s lens is a plain same-position Jacobian, not the J++ recipe: J++’s published R-Lens rules,
  cluster weighting and token filter were never tested on the instruct checkpoint, and fine-tuning and quantisation cannot
  be told apart (E50). Whether an instruct lens gives better neuron labels is untested (E45).
- **Rules for any lens served later**, from the errata:
  - confirm condition contrasts at layers 31–50 with an instruct reading (E49);
  - show pronoun readings at layers 51–59 only where J++ and the instruct reading agree (E49, E50);
  - keep the caution at layers 11–30 for every averaged lens: none tested was reliable there (E50);
  - re-validate the label gate of ii for the new lens: its AUC of 0.84 was measured for base-fitted J++ labels only
    (E47).

Nothing on record says what comes next for the branch.

<br>

## ii · neuron labels

The study judged joining the pillars “neat as a question, oversold as built”, and named four conditions under which the
joined instrument earns its keep (REPORT §3.3, erratum E47). The Lineage page (iv) marks where each stands:

| | The condition | Where it stands |
|---|---|---|
| **1** | Labels grounded in what wakes a neuron: the passages that make each face fire, stored with it | **Not yet.** One input label still serves both faces (fault 7), and input-side labels carried only 6–8% of the signal. No passages are stored for either face |
| **2** | Output labels trusted only from layer 41 up | **Yes.** No page sets a label below layer 41 in its main reading; the apparatus keeps it, with the reason, and the server marks it (`label_check`, with the rule in `label_gate`). The vault's console shows the server's replies whole and says which labels the pages hold back |
| **3** | A label counts only when the lens and the naked eye agree | **In part**, until the recorded labels are checked. A label whose two readings share none of their first ten word-pieces is held back; the others show their count. The cut at none of ten is the instrument's (the study set none; in N1's data 7 of the 31 such labels within layers 41–58 were sound, against 160 of the 209 others, `N1C_gate_cut.json`), and it applies only through the base-fitted J++ lens, within layers 41–59, the gate having been validated for base-fitted J++ labels on random neurons within 41–58 (E47). The demonstration sighting's labels and the vault's 24 recorded searches carry too few words for the count, so only the layer holds them back until `experiments/scripts/N1G_01_recorded_label_checks.py` is run on the GPU; then this row is **Yes** |
| **4** | Each label spot-checked by cause and effect | **In part.** By hand only, through the nudge. A neuron’s folio draws “run a causal spot-check” disabled, waiting on the server: add ±k · max\|a\| to the neuron at its word-piece, with the baseline in the same batch, and list the word-pieces whose log-probability moves most. It should steer in the linear regime (k ≈ 10, or a frozen-norm write), since 60× steering saturates the post-feed-forward norm at layers 21–51 (E47) |

Cammarata’s practices for reading a neuron ([9] in [`REFERENCES.md`](REFERENCES.md)), as the Lineage page sets them
against the instrument, leave four more:

- read each face in long passages, not only its top word-pieces: you can write whole sentences into a neuron’s guess
  test, but no passages are stored (the same gap as 1);
- look upstream at what builds a neuron (downstream, the threads and “who listens” are there);
- keep the field notes with the vault: they last for the visit only;
- watch a neuron form across training checkpoints.

<br>

## iii · smaller pieces still missing

- **Landmarks.** Contrast axes such as fnw · (W<sub>U</sub>[she] − W<sub>U</sub>[he]), less a background mean, which V7
  suggests ([`V7_tool_bugs_NOTES.md`](experiments/results/V7_tool_bugs_NOTES.md)): not in the server.
- **The vault.** An edit’s voice before and after: not in the server’s reply. The “.patch MEMIT” edit is a single-neuron
  column nudge, not MEMIT, and every vault query runs in Python; the `larql`/`vindex` binaries are called only for
  `--version` (README §iv).
- **Thinking on bare text.** A bare-text sighting with thinking turned on still sets `<|think|>` before the text, which is
  not how Gemma 4 switches thinking on; sent as a chat, it uses the template’s own switch. The study recorded no fix for
  the bare case (the Lexicon’s cautions; E6 audit, item 11).

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

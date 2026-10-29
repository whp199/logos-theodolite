# Roadmap

*What is still open in Logos Theodolite, gathered from the places that already name it: the README’s fault table, the
study’s report and its errata, the pages’ own “not yet” marks, and the records of R1’s instruct refit, R2’s test of
what to serve, R3’s test of J++’s recipe and R4’s test of R1’s lens. Nothing here has a date or an owner. Layers are
counted 1–60, as on the pages; the study’s reports count from 0, so their L50 is layer 51 here.*

<sub>Gathered on 21 October 2026, at `3cbc4b5`; gathered again on 22 October 2026, at `75ed78a`, after the first round
of mends (the interface’s defaults, the label gate, the causal test and the smaller pieces) and R3’s registration;
section i brought up to date at `c15b2d3`, with R3’s result, and on 28 October 2026, at `9093dac`, with R4’s; sections
ii–vi gathered again on 28 October 2026, at `554f058`, after a session that guarded every route, readied the census for
its first calibration and for rented cards, built the passages, one field book, the wiring’s sign in use and faster
4-bit unpacking, and began the CPU and GGUF (section vii, new); sections i, ii, vi and vii brought up to date on the
night of 28–29 October 2026 (`f88dba5`, `88e832d`, `4e38b6d` and that night’s review), once the calibration, the
passages and the wiring’s check had been taken on the model; what each hard job costs is in `COMPUTE.md`.</sub>

<br>

## i · the lens, the largest fault left

The J++ lens was fitted on the base model (`google/gemma-4-31B`, bf16) and is read here on the instruct model (4-bit
QAT), where it agrees with the model’s own average gradients for the pronoun direction at only cos 0.15–0.33 (layers
31–59; [README §iv](README.md#iv--how-far-to-trust-it), fault 9). Everything that reads through the lens inherits it: the
words, the gathered/recalled split, the landmark bearings, the neuron labels, the nudge and the vault. The head and foot
of every page name the lens plainly, and a red line beside each layer’s words says what the reading can bear there; the
opening page says once, among its cautions, that the lens was fitted on the base model. That was the owner’s choice of
28 October 2026, after no refit passed R2, R3 or R4. `/api/health`, every sweep and the replies of the scientist’s tools
still carry the full `lens` object, its caution included; the reading itself is unchanged.

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
- **R4 served nothing, and closed R1’s lead.** R1’s lens, read for reference in R3, had passed all five rules at layers
  56 and 59 on R3’s prompts: +0.071 and +0.058 over the naked eye.

  R4 tested that lead on its own
  ([`R4_PREREG.md`](experiments/results/R4_PREREG.md),
  [`R4_03_report_section.md`](experiments/results/R4_03_report_section.md)). It was registered before any data, after
  its code had re-read R3’s prompts and matched R3 exactly. It had 640 fresh prompts and R1 as the only candidate. It
  used R2’s rules, and the owner added a gate: R1 must also pass without the targets that are function words.
  - **What R2’s rule counts.** R2’s filter counts "the", "of" and "is" as content words. On R2’s and R3’s prompts, R1’s
    lead lay almost wholly in those targets.
  - **Layer 56.** R1 passed R2’s rule (+0.088 over the naked eye, +0.106 over J++) but failed the gate. Without
    function-word targets it was +0.015 over the naked eye and +0.025 over J++, and worse than J++ on short questions.
    It named 57% of function-word targets against the naked eye’s 32%, and 62% of the rest against 59%.
  - **Layer 59.** R1 failed R2’s rule itself, at +0.038 over the naked eye. On three prompt sets its margin there has
    been +0.046, +0.058 and +0.038.

  So R1’s lens predicts the model’s next function word better, and its next substantive word no better.
- **Rules for any lens served later**, from the errata:
  - confirm condition contrasts at layers 31–50 with an instruct reading (E49);
  - show pronoun readings at layers 51–59 only where J++ and the instruct reading agree (E49, E50);
  - keep the caution at layers 11–30 for every averaged lens: none tested was reliable there (E50);
  - re-validate the label gate of ii for the new lens: its AUC of 0.84 was measured for base-fitted J++ labels only
    (E47);
  - judge the lens without function-word targets as well (R4’s S6 gate): R2’s filter counts them as content, and a lens
    can win on them alone.

R3 and R4 serve nothing themselves: a pass would only have been a recommendation to later serving work on a branch from
`main` (R3_PREREG §7.2, R4_PREREG §2). With no lens passed by R2, R3 or R4, the serving work on `it-second-reading`
stays there as R2’s record. Its worktree also holds the only other copy of R1’s pinned lens file (gitignored); R4 copied
it to `~/.local/share/theodolite/lenses/` and reads it from there.
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
| **1** | Labels grounded in what wakes a neuron: the passages that make each face fire, stored with it | **Built, and gathered on the model; not yet validated as labels.** `scripts/atlas_build.py passages` keeps each face’s sixteen strongest windows of a corpus, one event a window (ties to the lower window), with the gate and up values there and each word-piece’s count over the corpus (the passages index, ARCHITECTURE §4.7), checked by brute force on the CPU stand-in (`check_atlas_index.py`). The atlas’s passages tab, a neuron’s folio and the scientist’s `passages` show them, with the word-pieces each face reads and that follow and their lift over the corpus. They were gathered on Gemma-4-31B on 28 October 2026, over the 2,000,054 word-pieces the stats index read (§vi; the build’s log in `experiments/results/atlas_wiring_check/gpu_logs/`); those words are counts over sixteen passages that no test has checked, so they are not yet validated as labels, and fault 7 (one input label for both faces) stays open until a test has checked them |
| **2** | Output labels trusted only from layer 41 up | **Yes.** No page sets a label below layer 41 in its main reading; the apparatus keeps it, with the reason, and the server marks it (`label_check`, with the rule in `label_gate`). The vault's console shows the server's replies whole and says which labels the pages hold back |
| **3** | A label counts only when the lens and the naked eye agree | **Yes.** A label whose two readings share none of their first ten word-pieces is held back; the others show their count. The cut at none of ten is the instrument's (the study set none; in N1's data 7 of the 31 such labels within layers 41–58 were sound, against 160 of the 209 others, so as a yes/no sign the cut scores AUC 0.64, against the count's 0.84 as a graded score; labels sharing just one of ten were sound 16 times in 40, below the band's 167 in 240, `N1C_gate_cut.json`), and it applies only through the base-fitted J++ lens, within layers 41–59, the gate having been validated for base-fitted J++ labels on random neurons within 41–58 (E47). The demonstration sighting's labels and the vault's 24 recorded searches carry too few words for the count, so `experiments/scripts/N1G_01_recorded_label_checks.py` checked each ten a side on the GPU (`N1G_recorded_label_checks.json`): 46 of the sighting's 75 labelled faces and 98 of the vault's 240 rows are held back |
| **4** | Each label spot-checked by cause and effect | **In part.** A neuron’s folio puts its label to the test (“test it at this word-piece”, and “test it on the study’s 36 texts” beside it), as do the WebSocket action `causal_check`, `POST /api/scientist/causal_check` and the MCP tool `causal_spot_check` (`backend/causal_check.py`, ARCHITECTURE §4.5): N1A’s ±k · max\|a\| on the neuron at the read position, with the untouched row in the same forward, at k = 10 in the linear regime by default and a frozen-norm write as a choice, since 60× steering saturates the post-feed-forward norm at layers 21–51 (E47). Each test reports how straight its push stayed, how far it rose above the same test at k = 0.01, and its P@10 for the lens’s and the naked eye’s labels; only the test on the 36 texts compares with the study’s figures. `scripts/verify_causal_check.py` checks it against N1A’s and N1V’s numbers on the real model. It is still run one label at a time, when asked: nothing tests every label. Testing every label at pages 41–59 in N1A’s layout would be about 2 × 10²⁰ multiply-adds, some 3.7 days of one H100 (inferred, COMPUTE.md); sharing each text’s prefix would cut that about 45 times but changes N1A’s layout, and no resumable driver is written yet |

Cammarata’s practices for reading a neuron ([9] in [`REFERENCES.md`](REFERENCES.md)), as the Lineage page sets them
against the instrument, leave three more:

- read each face in long passages, not only its top word-pieces: the passages are built and gathered on the 31B
  (condition 1), not yet validated as labels;
- keep the field notes with the vault: the server keeps them now, the name as the atlas’s and each face’s notes in a
  field book beside it (§iii), but not with the vault’s records (`DESCRIBE POLE` does not return them);
- watch a neuron form across training checkpoints: no intermediate checkpoints of Gemma 4 31B are known; the base model
  against the instruct one is the nearest substitute, and needs the base model and the GPU.

Looking upstream at what builds a neuron is in the folio now (§iii).

<br>

## iii · smaller pieces, now built

- **Landmarks: one word against another. Built.** A sweep can weigh one word against another at every layer, through
  the lens and by the naked eye, against the matched controls sent or against E3’s average state (`contrasts`,
  `contrast_backgrounds`; ARCHITECTURE §4.1–4.2): the contrast axes V7 suggests
  ([`V7_tool_bugs_NOTES.md`](experiments/results/V7_tool_bugs_NOTES.md)), read in log-odds beside the frame rather
  than as an axis of it. The landmarks page reads them in its section vii: up to four contrasts, one a row, against
  either background or both, drawn one at a time (a pill for each contrast the sighting read, each background its own
  line). Still open: through the lens the reading inherits the base-fitted J++ (i), so the page marks its layers with
  the bands every page draws for the lens’s words (1–20 the least trusted, where E3 (b) found the lens barely agreeing
  with the model’s own pronoun direction; 21–30 orderings only; 31–50 to be confirmed with an instruct-fitted reading;
  51–59 counting only where one agrees: errata E49 and E50), and the naked eye with its own background, which runs the
  other way (E3 (c)); the study measured the average state’s lean on its six pronoun pairs and on its noun pairs
  (woman–man and the like), where it leans toward the male words and is larger (E3 (c): at layers 41–50 a share of
  +1.4 on the pronouns and −3.1 on the nouns), not on one pair or on other words; E3’s own lens figures for the
  average state reproduce today only as a band mean at layers 41–50 (+2.35 against its +2.31, within 0.04): layer by
  layer they miss by 0.20 and 0.10 at layers 41 and 42, and by more than 0.04 at 33 of the 60 layers (median 0.18, at
  most 1.85 at layer 3), by the study’s own code as by the server, with the cause not established (by layer:
  [`E3_background_mu_check.json`](experiments/results/E3_background_mu_check.json)); four contrasts and two
  backgrounds were checked on the CPU only (`check_interface_defaults.py`), and the chart seen only with readings
  made up for it, since the stand-in cannot take a sighting (a sweep reads sixty layers and it has six). That E3’s
  average state is in the coordinates of the server’s `resid_post` is checked by the end-to-end check (item 12) on one
  of E3’s neutral passages.
- **The vault: an edit’s voice. Built.** An edit can be heard before and after on a question of the reader’s choosing,
  written the likeliest way under the one GPU lock (`voice`; ARCHITECTURE §4.6), and the vault page asks for it by
  default. It is offered through MCP too (`query_larql_vault(lql_query, voice)`, the same `VoiceRequest`; run on its
  own, the MCP server’s tools take turns under one lock of their own, so nothing lands between an edit’s two answers),
  and for an edit typed in the console, heard by default with the edit tab’s question (`check_vault_voice.py`, on the
  CPU stand-in; the end-to-end check compares the tool’s heard edit with the route’s, not yet run on the GPU). The
  pages now call the “.patch MEMIT” edit an edit, a one-neuron column edit and not MEMIT; the code’s names and the
  statements are kept for old callers and the study’s scripts. Every vault query runs in Python; the `larql`/`vindex`
  binaries are called only for `--version` (README §iv). Still open: it is one question, so UPDATE and INSERT are still
  tested by cause and effect only as far as that one answer goes (the study tested only DELETE, E6 §2.3). The edit
  tab’s own example edit is not heard on the page’s own question: its answer does not change, so a first reader hears
  nothing; the end-to-end check finds an edit that is heard, at layer 60, instead.
- **Thinking on bare text. Built.** Sent as a chat, the sighting with thinking turned on uses the template’s own switch;
  sent as bare text, the interface leaves it out, since bare text has no turn to think in (`think_bare="none"`), and
  the first pair names how thinking was switched on (`think_protocol`; null on later pairs). Still open: the server’s
  own default stays the old imitation, `<|think|>` set before the text, for old callers, and the question’s apparatus
  keeps it as a marked choice; the study recorded no fix for the bare case (the Lexicon’s cautions; E6 audit, item 11).
  Even in a chat the thinking sighting is read at the end of its own prompt, where the silent one is read at the close
  of its empty thought, unless an answer is begun (`think_read_token` names the word-piece it was read at).
- **Who may ask, on every route. Built** (28 October 2026). Until then only the atlas, its scientist chat and the saved
  prompts refused a page from another site: CORS allowed every origin, and `/api/larql`, the four `/api/scientist/*`
  tools and `/ws/theodolite` checked neither Host nor Origin, so any page the reader visited (or one rebinding its name
  to 127.0.0.1, or any page while a pod’s server was tunnelled to 127.0.0.1:8765) could run sweeps and vault edits on
  the live model and read the replies. Every route, the pages and the socket now answer only this machine’s pages and
  scripts run on it (`backend/local_only.py`, ARCHITECTURE §4), CORS names only loopback origins, and on Linux a
  connection from another account on this machine is refused too (its owner read from `/proc/net/tcp`), except
  through `npm run dev`’s proxy while it runs; `scripts/check_local_only.py` checks every route and the socket. The
  lab daemon refuses web pages and any Host but this machine’s, and its `/exec` wants a token it writes at each start,
  readable by its user only (`labclient.py` sends it); its port, 8765, is still the server’s. The scientist’s CLIs are
  no longer given the Hugging Face or cloud credentials in the server’s environment, only their own logins, and a stop
  no longer waits on a chat’s open event stream (`--timeout-graceful-shutdown 3`, README §v).
- **The guess test’s statistics. Built.** A neuron’s folio, v (`test_motif_hypothesis`, `POST
  /api/scientist/test_hypothesis`) said “yes, clearly” whenever Cohen’s d over population variances reached 0.8, at any
  number of sentences, with nothing set against chance: one sentence a side gave d near 10⁴ (E6 §3, item 7). Its reply
  now adds `selectivity` on the last word-piece’s unrounded readings: sample variances, Welch’s t and the n each side; a
  two-sided p from splitting the sentences between the sides afresh (exact over every split up to 10,000 of them, else
  10,000 shuffles under seed 0); hedged words (“selective on its + face, p = 0.029, n = 4 a side”), and no judgement
  below four sentences a side. The folio shows that; the verdict as first built is still sent, for old callers
  (`scripts/check_hypothesis_test.py`). Still open: each test is one neuron’s, so p’s from several are not corrected
  together (the study used Benjamini–Hochberg, E6 §2.3); it is unpaired, where E6’s own test paired matched sentences
  with sign flips; the peak is reported, not tested.
- **One field book. Built.** A neuron’s folio reads and writes through the server whenever it answers for the neuron:
  its name is the atlas’s name (`field_notes.json`, by the reader), and its notes, face by face (how many jobs, the
  pattern in words, the four ticks), go to `field_book.json` beside it (`/api/atlas/fieldbook`; ARCHITECTURE §4.7), a
  file of its own because a server from before moves aside any notes file but version 1 and keeps only the fields it
  knows. The atlas’s details show the book, with a link to the folio. With no server, or from the moment it does not
  keep a change, the folio keeps them in the browser and says so, and it offers once to carry a browser’s notes to
  the server, writing over nothing the server keeps. Lineage practice vi.
- **What builds a neuron, in the folio. Built.** Practice v, upstream: at the sighting’s own word-pieces, the heads,
  the earlier feed-forward blocks and the earlier neurons whose writes make up its activation there (`POST
  /api/atlas/attn_drive` with `ids`; each block’s part split over its neurons, one mat-vec a layer, checked on the
  stand-in to add up to the block’s part to float error and against a direct-path finite difference), and by the
  weights alone its strongest writers, each on the face that opens its gate. Both are leads: a neuron’s share is its
  direct write with the norms held, not what it does through the layers between. Not yet run on the 31B.
- **The scientist’s names as proposals. Built.** `set_name` and `write_note` propose (`field_proposals.json`,
  `/api/atlas/proposals`): the atlas page shows each with keep and discard, nothing reads one before it is kept (the
  map, lists, search, tooltips and later chats read only the names and notes kept), and a kept one is written as the
  scientist’s.

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
  were not opened, and could not be from the machine of 28 October either (its network refused arxiv.org and
  huggingface.co).

<br>

## vi · the atlas

The atlas (`#/atlas`; README §ii, ARCHITECTURE §4.7–4.8) was built on 27 October 2026 without a GPU: every piece of it
was checked on a tiny random Gemma 4 on the CPU (`scripts/atlas_stand_in.py` and the atlas’s checks) and the page driven
in a headless browser against it. The same day its indexes were built on Gemma-4-31B on one RTX 5090 (the lens in 144 s;
the stats over 2,000,054 word-pieces, a document from each corpus in turn, in 58 minutes; the wiring in about 4 minutes;
the layout in 9) and the page driven end to end on them, a Codex scientist with it; the index files are kept outside the
repository. On 28 October the census, the wiring, the passages and the scientist gained the pieces below, each checked
on the CPU stand-in; each bullet says which have since run on the model (the calibration ladders and the writer’s floor
on 28 October; the passages, the wiring rebuilt and its measured check on the night of 28–29 October; one steer-lens
layer and the determinism run on 29 October). In this section layers count from 0, as the commands do
(page = layer + 1), unless pages are named. [`COMPUTE.md`](COMPUTE.md) says what each job costs and where to run it.

- **Take its censuses, and its steer lens, on the model.** Neither exists beyond the 8 + 8 `calib_rel` pilot of 27
  October (580 answers, about 16 minutes at `--max-new 64`), the two calibration ladders of 28 October (64 + 64 units at
  each of six layers, below), one steer-lens layer timed into a scratch folder, the 16 + 8 determinism run of 29 October
  (868 answers) and the `break_rel` trial taken and read that day (both below). The pilot measured the cost: every
  decode step of the 31B takes about 0.71 s whatever the batch (1 to 64 rows), so a whole `main_rel` layer is about 3.6
  days at `--batch 32 --max-new 64` and about two weeks at the default 256. The step is the 4-bit unpacking, not the
  model (COMPUTE §i). The steer lens was estimated at 15 to 20 hours for all 60 layers on a 5090. One layer timed on 29
  October 2026 (layer 45, `compiled`, into a scratch folder, not served;
  [`atlas_steer_timing/`](experiments/results/atlas_steer_timing/)) took 287 s; a layer runs only the layers after it
  (the forward starts at l + 1), 14 for layer 45, so its cost grows with their number, and all 60 would be about 10
  hours (inferred: 287 s / 14 × 1,770 layer-forwards; a shallow layer timed would check it). Its dose, the census’s 0.8
  of the stream, is still to be chosen against the calibration before it is built. On the home card, headless, time a
  pilot first; on a rented one, `scripts/pod/pilot.sh` is that pilot (it times the step and the memory for each unpacker
  at several batches, takes a `calib_rel` run, times one steer-lens layer, sets a canary batch against the home card’s
  and projects each job × 1.25 against the owner’s approval, which `scripts/gpu_log.py` keeps with every pod against
  it); a layer’s census then runs in shards (`scripts/pod/census_shard.sh`, every shard at the batch the pilot measured
  on that card class) and `sync.sh --final` pulls and checks them before a pod goes. None of these has yet run on a
  rented card.
- **Faster unpacking, the same bits. Built, and timed.** Every forward unpacks the 4-bit weights afresh (five eager
  operations, three over int32 temporaries), which by the arithmetic is most of the 0.71 s step. `backend/w4a16_fast.py`
  installs, without touching the R4-pinned runtime, a lookup-table unpacker (`--unpacker lut`), the same arithmetic
  compiled alone (`compiled`, the model left eager with its hooks) or the weights kept unpacked (`resident`, about 60 GB
  beside the 4-bit ones). Each gives the reference’s bits: `scripts/check_dequant.py` shows it on the CPU for every
  (piece, bf16 scale) pair and for a 4-bit stand-in’s logits, vault edits, pushed answers and census rows, and `install`
  tests every linear again on the card, refusing in plain words on any difference. Each census and index meta, and
  `/api/health`, records which unpacker ran. On the home 5090 (28 October 2026, beside a desktop) `lut` and `compiled`
  passed that test on all 410 linears and took 0.28 s and 0.103 s a step at a batch of 32, against the reference’s
  0.71 s; `compiled` held 0.126 s at 96 rows (COMPUTE §ii). By that step a whole `main_rel` layer is about 5 hours at
  `--batch 96 --max-new 64`, not 3.6 days (arithmetic, not a run); the calibration below, taken with `compiled` at a
  batch of 64, took 65 and 49 minutes for its six layers.
- **`main_rel`’s doses come from Qwen3.5-4B; `break_rel`’s from this model’s ladders.** `main_rel` (−0.7, +0.5, +0.8
  and +3 of the stream’s median length) is
  neuron-explorer’s design, where its log found answers readable near 0.5–0.8 and wrecked by 3. Take `calib_rel` (±0.1
  to 4 of the stream) and `calib_ceil` (±0.5 to 8 × each neuron’s write ceiling) on a few layers of Gemma-4-31B first;
  one command takes several layers (`census_run.py --layers 20,30,40,45,50,57 --design calib_ceil`, layers from 0).
  Read each run with `census_tools.py curves`, which lays out its dose window: per dose, neurons against controls, the
  share of answers moved from the unpushed one and where they first part, the share cut off, distinct 4-grams, and the
  push as a multiple of each neuron’s ceiling, with a hedged verdict per sign. The 8 + 8 pilot already taken can be read
  the same way (`curves <layer> --weights`). Change the design if the window lies elsewhere.

  **Both ladders taken** (28 October 2026, layers 20, 30, 40, 45, 50 and 57, pages 21–58: 64 random neurons and 64
  controls a layer, `--batch 64 --max-new 64 --unpacker compiled`, 65 and 49 minutes on the home 5090; each layer’s
  dose window and a summary in `experiments/results/atlas_calibration/`, the runs in the atlas folder). One question
  set, one greedy answer each, so these are leads:
  - **Where text breaks** (distinct 4-grams more than 0.1 below the unpushed answers’): at ±0.5 of the stream at layer
    20, ±1 at layer 30 and ±1 to 2 from layer 40 to 57; in each neuron’s ceiling, at about 4 to 5 × at layers 20 and
    30, 2 to 4 × at 40–50 and 1.2 to 2 × at 57. Neither unit puts it at one dose across depth. `main_rel`’s +3 breaks
    text at every layer, and its +0.5, +0.8 and −0.7 at layer 20.
  - **Where effects begin lies below both ladders.** The smallest push of each (0.1 of the stream; 0.5 × the ceiling,
    0.05–0.41 of the stream) already changes 38–95% of the 64-word-piece answers somewhere, and a matched control
    changes as many: the neurons moved more answers than the controls at one dose of some 130 tested (layer 30, −0.5 ×
    the ceiling, 88% against 81%, uncorrected), about what chance gives. Whether an answer changes at all is too coarse
    a reading at this length, and `curves`’ verdict reads only that.
  - **The ranking tells neurons from controls clearly at layers 40 and 57; where the lead lives is not settled.** The
    census’s own ranking (the words each push brings) caps the push it reads at 1.6 of the stream (`compute_units`’
    `max_rel`), so as first read here it left out `calib_rel`’s ±2 and ±4 and every `calib_ceil` dose past 1.6 of the
    stream (at layer 57 all but ±0.5, ±1 and a little of ±2): those first figures (0.62–0.80 at 40–50, alike at 20, 30
    and 57) were over the doses under the cap, not every dose. Re-ranked on every answer (29 October 2026, CPU;
    `experiments/results/atlas_calibration/`, `subset_rank.py` with its `subset_ranking.json` and `subset_table.json`),
    neurons lead their controls clearly at layers 40 and 57 in both ladders (AUC of mass, the chance that a neuron’s
    mass beats a control’s, 0.62–0.70, p ≤ 0.01; in the ceiling at layer 40, 13 of 64 beat every control where about one
    would by chance, least q 0.08) and at 45 in the ceiling (0.63, p 0.006). At 45 in `calib_rel` and at 50 in both, the
    lead is no stronger than at 30 (0.55–0.58, p 0.06–0.15, against 0.58 and 0.59, p 0.06 and 0.05, at 30), and at 20
    there is none (0.52 and 0.50). Under the break alone (the doses short of where text breaks, as read above) it holds
    at 40 and 45 in `calib_rel` (0.62 and 0.60, p 0.01 and 0.03, and no less once the new words an answer keeps are
    regressed out: 0.64 and 0.62), weakly in the ceiling at 40–50 (about 0.58, p 0.05–0.07), and not at 20, 30 or 57,
    nor at 50 in `calib_rel`. At ±2 of the stream it is largest (Stouffer z 6.4 over the six layers), but the file’s
    covariate check credits most of that to the neurons’ text keeping more new words under the push than the controls’
    text does: with that regressed out the AUC is 0.52–0.57, at no layer p < 0.05.

  A design to replace `main_rel` was drafted on these leads: `break_rel` (±1 and ±2 of the stream on six questions read
  as two halves, every answer counted, each word against the controls alone), built on the branch `census-break-rel`
  and merged on 29 October once its trial was read; `main_rel` stays the default (the owner’s choice). Its trial was taken early on 29 October, its reader and
  gates fixed and hashed before it began (`SHA256SUMS` sha256
  `8d8d81de91dd03529db83f091c7ddeb26bbb4dd8037cac66f6dd91884c7c3b2d`), and read that morning with the gates as they
  were (the owner’s word). At layer 45 (1,022 fresh neurons against 1,024 controls) it **passes every gate** at 48
  word-pieces, and at 32 every gate the reader computes there (the check that the units file ranks as read is made at
  48 only). At 48: mass AUC 0.714, 0.724 with six measures of how the text survives fitted out, words picked on one half
  of the questions and scored on the other 0.688 and 0.674, the three new questions alone 0.669 (all AUCs), against
  bars of 0.60, 0.58, 0.57 and 0.56. The probe at layer 57 (507 against 512) passes at 48 and fails at 32 (its residual
  0.5608 against a bar of 0.56, then 0.5061). By the rules written before it, that licenses whole layers at 40–50 at 32
  word-pieces and at 51–57 at 48. The shorter reading at 45 lay within 0.02 on the two measures the rule checks, but it
  names 191 neurons where 48 names 262, and its residual falls 0.0215. A review the same morning
  ([`census_break_rel_trial/`](experiments/results/census_break_rel_trial/), `summary.json` and `review/`) used six agents: three
  readers, each checked by a second that re-derived its headline figures, the gates with code that shares nothing with
  the pre-registered reader, the by-dose and per-neuron figures with code built on its parts. It found the reading
  faithful; four small departures, none able to move a gate, are recorded there, one of them that the 32-piece reading
  is the 48-piece answers cut short. Then it asked what the pass measures:
  - **At layer 45 the lead is the ±1 answers’.** On them alone the AUC is 0.706 and clears every bar. On the ±2 answers
    alone it is 0.571 and clears none: at ±2 the text of neurons and controls alike collapses into loops on the same
    few words (own, thought, same; distinct-4 0.21 for both), so that half of the answers adds 0.008, though alone it
    still holds a weak lead (5.6 SE above chance). Among units whose twelve ±1 answers all stay fluent (distinct-4 at
    least 0.9; 705 neurons, 942 controls) the ±1 lead is 0.641, about two-thirds of the pooled lead above chance. A
    neuron picked on one half is found again on the other three times in four (120 of 161, against 7 of the 51 strongest
    controls), and its words nine times in ten (147 of 161, against 10 of 51). The two halves’ ranks agree at 0.46
    (controls 0.14), and about 7 of the 262 neurons named would be named by chance (95% upper bound about 14). The 893
    neurons whose text survival the controls do cover clear every gate on their own (0.692, 0.691, 0.657 and 0.643,
    0.651), the check layer 57 fails. The order lies in the layer’s tail: below the controls’ 95th percentile the halves
    agree barely more than the controls’ do (0.17 against 0.12), though those 767 neurons still lead the controls as a
    group (0.61 on the other half). The calibration’s “none named” was forced by its size (64 + 64 units cannot reach
    q ≤ 0.05 with fewer than 20 neurons above every control; it had 15).
  - **Part of that lead is text breaking early, and the words need filtering.** The named neurons’ ±1 answers loop
    (distinct-4 below 0.5) in 9.6% of cases (fresh neurons 4.1%, controls 0.4%): 135 of their 303 looping answers on
    their own words, 127 on the collapse words and 41 on neither. 194 of the 262 list “same”, “own” or “thought” among
    their words. The gates do not rest on these: with the 789 collapse words removed from every answer, every gate still
    passes. But a page should not show them as a neuron’s words.
  - **At layer 57 the pass is fragile and is not words in fluent text.** ±1 alone gives 0.52, and only the two doses
    together clear the probe’s first bar. The named take 82% of their weight from broken answers (distinct-4 below 0.9;
    the controls 49%, nearly all at ±2, and all 507 neurons 60%); many loop on pieces such as “sem”, “pemb” and “ngày”,
    though about half their words are ordinary ones (vendor, volt, deck). Its residual falls below its bar in about half
    of the resamples. And 30% of its neurons write text whose survival no control matches, while the 354 the controls do
    cover fail every probe gate (0.551, 0.488, 0.522 and 0.537). This design cannot tell layer 57’s lead from how the
    text survives.
  - **Two sanity checks say less than their names.** The read gain is computed from the weights alone before any answer
    (at 57 the controls are read 1.18 times as loudly), and the relabelled-control split cannot name anything, so it
    shows only that the reading favours neither label.

  A whole layer is taken at batch 64 (the step-0 rule: chair parts at 20 pieces at batch 96), at the trial’s measured
  12.4 answers a second while writing: about 12.1 hours at 48 pieces (12.7 with 2,048 controls), about 8.0–8.4 at 32
  (estimated, not timed). The owner chose, on 29 October, to count `break_rel`’s doses calibrated at layers 40–50 alone
  (pages 41–51): `DOSES_CALIBRATED` gives each design its span of layers and the card tests the run’s own, so 51–57
  and every layer below 40 stay uncalibrated. These still wait on the owner: the piece count, whether ±2 stays
  (a design without it is a new design, and the ±1 figures above were read after the fact), the controls, the default
  design, and what a page says a lead means (the pre-registration left open whether to say “unlike a random direction
  pushed as far” rather than “has a theme”). Every page and tool that shows a census says its doses are uncalibrated
  (in red), or not calibrated at that layer for `break_rel` outside 40–50, save a `break_rel` census at 40–50.
- **The push’s size.** On the 31B the norm after each block caps what a single neuron can write, and the census’s 0.8 of
  the stream is 1 to 10 times that cap at most layers (review of 27 October 2026). A census can now be taken in each
  neuron’s own ceiling (`--dose-unit ceiling`): a neuron is pushed by a multiple of the most it can write by itself, and
  a control by the ceiling of a neuron it is paired with, so one dose is the same share of each neuron’s most at every
  layer and the depth curve compares like with like. Every answer records its ceiling and its push’s real share of the
  stream. Which unit, and which doses, a whole layer is taken at waits on the calibration, taken (above) but read so far
  only as leads; `break_rel`’s doses, ±1 and ±2 of the stream, are counted calibrated at layers 40–50 alone (above),
  though whether ±2 stays and which design is the default still wait on the owner. The steer lens’s and the census’s 0.8
  is to be chosen against the ceiling the same way before either is built on the model.
- **The census’s null.** Its controls are drawn as unit(w_post ⊙ r), matched to within 1.2× (`check_census.py`); with C
  controls about N / (C + 1) neurons beat every control by chance alone (84 of a whole layer with 256), so each neuron
  carries a p and the layer a false-discovery q, and with 256 controls no single neuron reaches q ≤ 0.05 unless some
  1,700 of the layer beat every control. A second stage now exists: `census_run.py --neurons-from <screen>.units.jsonl
  --top M --seed <another>` retests a screen’s M strongest neurons against fresh controls, its p and q taken over those
  M alone (`stage` in the meta and on the card; the layer’s census stays the screen). Its cost is mostly the controls:
  about (200 + 4,096) × 10 answers, some 17 hours on the 5090 at `--batch 32 --max-new 64` (worked out from the
  pilot’s step). How many controls a whole layer gets, or whether it is read as leads to retest, is the owner’s choice.
- **The census on larger and rented cards.** A run’s batches are planned from its job list and the batch size, written
  whole, so a stopped run, a torn write and a run in shards each write the rows an unbroken run does (checked row for
  row on the stand-in). The batch and the plan are resume keys and fix a run’s determinism class: measure what a card
  holds once, then keep that `--batch`. Each meta records the runtime it was written on (the card, the versions, the
  commit, the stats index’s sha256) and warns when that changes; the unpacker is recorded beside it. `--shard i/n`
  writes whole chunks, each shard but the first with a canary batch, and the merge refuses a missing shard or answer
  (unless `--partial`) and compares the canaries; a shard is never served as a layer’s census. The census frees the J++
  lens after loading and reads each step back in one transfer. `scripts/verify_atlas_writer.py` measures the writer’s
  floor (how far an unpushed answer agrees with itself written in the batch shapes the census writes in), and the census
  card quotes it: a pushed answer that parts from the unpushed one no sooner than that floor may have parted by the
  batch’s shape alone. On the home 5090 (28 October 2026, 2 minutes with `compiled`;
  `experiments/results/atlas_writer_floor_rtx5090.json`) an unpushed answer written in a census batch of 64 first parted
  from itself written alone after 22 word-pieces (one question of four), in a batch of 32 after 54; alone, as `/steer`’s
  middle row and by the sweep’s writer it held all 64. Within one shape the unpushed row and a dose-0 row part alike, so
  a run, whose baseline is written in its own batches, compares like with like. Without the lens the 5090 holds a batch
  of 96 at `--max-new 64` beside a desktop (7.2 GiB above the model, COMPUTE §ii). On one card, across processes, a run
  is bit for bit itself (29 October 2026, the 5090, `compiled`, a `calib_rel` run of 16 neurons and 8 controls at layer
  45, 868 answers; [`census_determinism/`](experiments/results/census_determinism/)): the same run taken in two shards
  and merged equals the unbroken run in every field of every row, and two canaries taken in separate processes are
  identical. Still to measure: how far a run differs across cards.
- **The wiring.** Links are ranked by |P| f², with Gemma’s layer scalars and the norms’ usual scales (the review of 27
  October 2026), and every consumer reads their signs one way, from rest through GeGLU’s gate. That model is a
  simplification, and a second reading is now built beside it: to first order about a reader’s operating point (g₀, u₀),
  a writer moves it by f (α g + β u), α = gelu′(g₀) u₀, β = gelu(g₀), with the operating point served on each face taken
  as the median over its strongest stored passages of that face: where it fires hardest, not where it sits (below). It
  is odd in the writer, where the sign from rest is even. `model=use` reads it on `/neuron/wiring`, `/web/summary`,
  `/cone` and `/run/paths` (the default stays from rest), the details tab marks a link where the two disagree, and
  `atlas_build.py wiring_check` measures both: readers under ±d pushes of 512 writers a layer, d a quarter of each
  writer’s own ceiling, against as many readers they are not linked to.

  **Measured on the 31B** (the night of 28–29 October 2026: the passages, the wiring rebuilt with each link’s g and u,
  then all 59 layers of the check in about 31 minutes on the 5090 with `compiled`; per-layer medians, linked / null, at
  the last word-piece of the census’s four questions, the chat template’s closing `<channel|>`, where the answer begins,
  not a word of the question; ARCHITECTURE §4.7; every figure here is kept with the script that gives it in
  [`atlas_wiring_check/`](experiments/results/atlas_wiring_check/)). As built, the sign from rest held on the gate face
  at 0.336 / 0.341 and the sign in use at 0.418 / 0.438, below chance for links and null alike (changes of exactly 0
  left out, as the summary now counts them; the file’s stored summary, counting a 0 as a miss, gives 0.320 / 0.324 and
  0.417 / 0.427). No sign convention is wrong: at that word-piece about nine readers in ten sat with the gate shut,
  70% / 63% (pooled) below gelu’s minimum, where the up path carries about −sign(P) (the sign from rest holds where the
  gate is open, 0.82 / 0.69); and the sign in use is read where each reader fires hardest, its strongest stored
  passages, where the gate is open. Scored offline from the stored responses at the readers’ measured operating point
  (their g₀ and u₀ read afterwards on the CPU from the packed checkpoint, one prefill a question, by
  `scripts/z1_31b_g0u0_cpu.py` in the folder above), the same responses agree with the sign in use at 0.835 / 0.854
  (0.842 / 0.853 in float64, as the summary now computes it), and the change’s size follows f |α₀ g + β₀ u| d with
  ρ 0.89 / 0.87, against 0.31 for |P| f². The links are real by size: linked readers move 5.5 times as much as the
  null’s. Over the whole index the sign in use differs from the sign from rest on 28% of links on each face of the
  reader (28.2% on +, 28.3% on −; on both only 1.9%, so on about 55% of links on one face or the other; the sign in use
  there is read where the reader fires hardest). The two readings now state their conditions (`LINK_MODEL`: the sign
  from rest holds for an open gate, or a push big enough to open it; `LINK_MODEL_USE` and the details tab: read where
  the reader fires hardest, not where it sits), and the check records each reader’s own g₀ and u₀, scores the sign in
  use there, splits both signs by the reader’s regime (open, shut above gelu’s minimum, below it), judges sign shares
  against 0.5 (the null tests size) and ranks the size against the first-order term too.

  **Rerun with g₀ and u₀ recorded** (29 October 2026, 04:18 CDT; `wiring_check --fresh`, the night’s first check set
  aside; 59 layers in about 32 minutes; [`rerun_at_site.json`](experiments/results/atlas_wiring_check/rerun_at_site.json)):
  the check itself now gives what the offline scoring gave. Medians over layers, linked / null: the sign in use scored
  at each reader’s site holds at 0.841 / 0.853 (under the plain +d push 0.866 / 0.852); read where it fires hardest,
  as served, 0.418 / 0.438; the sign from rest 0.336 / 0.341; size against f |α₀ g + β₀ u| d at ρ 0.89 / 0.87, against
  0.31 for |P| f²; linked readers moved 5.5 times the null’s. By regime, for linked readers: the gate open at 7% of
  pairs (the sign from rest 0.82, at the site 0.86), shut above gelu’s minimum at 14% (0.47, 0.74), below it at 80%
  (0.19, 0.88); the null’s shares 12%, 21% and 68%. (The 70% / 63% above are pooled over every pair, these the median
  over layers.)

  Still open: (1) The sign in use the atlas serves is read where each reader fires hardest on a face, not where it sits
  at an ordinary word-piece. A resting operating point (the median g and u over a sample of ordinary word-pieces,
  beside the passages index) or one per GeGLU regime would read it there; whether to build one is the owner’s choice,
  and until then every page says where it is read. (2) The dose: at a quarter of the ceiling the pushes stay near
  first order; at the whole ceiling gates opened in 11–61% of linked pairs and the sign from rest rose (layer 35: 0.44
  to 0.77), while the first-order sign in use at the unpushed point fell to 0.42–0.53: each reading holds in its own
  regime of dose, measured for 24 writers on one question.
- **Stored passages for each face** (Cammarata’s practice iii, the study’s §3.5): **built on Gemma-4-31B** (28 October
  2026, 20 minutes on the 5090 with `compiled`, over the 2,000,054 word-pieces in 3,914 windows the stats index read;
  its log in `experiments/results/atlas_wiring_check/gpu_logs/`). One pass over the corpus the stats index was read over
  (`atlas_build.py passages`, the same windows) keeps each face’s sixteen strongest windows and six drawn at random
  above half its extreme, with the gate and up values at each event (the operating point the wiring’s sign in use
  reads). Still open: whether the words over sixteen passages make sound labels (condition 1 of §ii; no test has checked
  them). The stats pass now keeps its norms in float32 buffers (about 1.4 GB at two million word-pieces, where Python
  floats took about 11.5 GB, inferred), so a pass over ten million fits in memory, its quantiles unchanged.
- **Single neurons at pages 21–31.** The study's 60-fold push overdrove them (errata E43, E46). The calibration (above)
  moved their answers readily at layers 20 and 30, from 0.05 of the stream at 20 and 0.1 at 30, but no more than matched
  controls moved theirs; on every answer the census’s ranking told them from the controls not at all at layer 20 and
  weakly at 30 (AUC of mass 0.58 and 0.59, as weak as at 50).
- **The scientist’s reach.** Since the review of 27 October 2026 it reads, writes and edits files only in its work
  folder under Claude Code, with no shell, and runs Codex’s shell in its workspace-write sandbox;
  `THEODOLITE_SCIENTIST_REACH=home` gives neuron-explorer’s shell over the home folder, and the page asks first. On 28
  October the folder rules were tried on the real CLI, Claude Code 2.1.312, against a stand-in Messages API, not a live
  model (`scripts/verify_scientist_cli.py`): in the work folder Read, Write and Edit run; beside it, a path climbing out
  of it, the prompt files, the home folder and `/proc/self/environ` are refused; Bash and Glob are not there; the
  atlas’s tools run through the real MCP server. That run found that `--setting-sources project` let a CLAUDE.md (or a
  settings file’s hook) in the work folder, which the scientist may write, or in any folder above it, the home folder’s
  own `.claude/CLAUDE.md` included, reach every later chat; a chat now loads none (`--setting-sources ""`). A chat
  takes up changed instructions and tools: a Claude Code chat at the folder reach is started again at the next message,
  resuming its session; any other is marked stale and the page offers a restart, since ending its process group also
  ends any commands it left running. The names and notes a scientist writes are proposals, kept or discarded by the
  person (§iii). Still open: Codex in its sandbox can still read files outside its folder; another account is not told
  apart on macOS or elsewhere, nor through vite’s proxy (a per-launch key would cover both); a scientist with a shell
  over the home folder could keep its own proposal through the loopback port, which the guard cannot tell from the
  page; the rules hold for the CLI version tried, so a new one should be tried with the same script; and no live model
  has yet been watched to see whether what it reads would lead it to try what the CLI refuses.
- **The push, greedy or sampled** (neuron-explorer read again at 6a6f0e9, 28 October 2026). `/push`, `/steer`, the push
  tab and the scientist’s `steer` write greedily by default, as the census reads a push. On request they write with the
  instrument’s own sampler (temperature, top-k, top-p, a presence penalty only with a temperature, a seed drawn and
  echoed when none is given), each step one draw shared by the three voices (an exponential race seeded seed + s), so
  the voices part only where the push moves the odds; `/steer` writes up to eight samples in one batch, and every reply
  says how it was written. Checked on the stand-in in float32 only; still to see on the 31B whether the voices stay
  alike at dose 0 in bf16, and what eight samples cost. The census stays greedy.
- **Not carried over:** the explorer’s HF-against-vLLM steering check as such (the atlas runs on the instrument’s one
  runtime; what it measured is measured here as the writer’s floor, above); its steered prefill captures (their idea,
  a push at every prompt word-piece rebuilt by one prefill, gives the wiring its measured check); its steer’s min-p and
  repetition penalty (`backend/sampler_request.py` says why: the penalty counts the prompt, the end-of-turn mark with
  it, and can smooth away the loops a push is read by); its neuronary.org tools (recipes for Qwen3.5-4B’s neurons,
  published as the person by a model, which the scientist’s reach rules would not allow) and its Chinese interface.

<br>

## vii · the CPU and GGUF

The owner’s aim (28 October 2026): every part of the instrument on the CPU and from a GGUF, not only the W4A16
checkpoint on one GPU. Two layers were built that day without a GPU and checked on the CPU stand-in only
(`scripts/check_gguf.py`, `scripts/check_gguf_cpu.py`, ARCHITECTURE §5). The first: torch unpackers for thirteen GGUF
types, equal to gguf-py’s bit for bit; a packed GGUF layer that hears the vault; a Gemma 4 built from a GGUF with its
provenance; `GGUFRuntime` beside the pinned runtime; `make_runtime()` choosing between them from the environment; and
`scripts/compare_gguf_ct.py`. The second: every entry point loads through the factory (the server, the census, the
builds, the scientist’s MCP server run alone, the lab daemon); the lab’s gradient passes have a GGUF twin of their W4A16
autograd function; every index names the weights it was built from, the server refuses one built from others, and a
build does not write over another model’s indexes unless asked; the CPU’s memory is estimated before a sweep, a census
batch, the vault’s voice, a causal check and a steer-lens batch and refused in plain words; and `int4mm`, torch’s int4
CPU kernel, holds a Q4_0 GGUF’s or the checkpoint’s linears on the CPU (1.7 to 8 ms for a 5,376 × 21,504 layer at one
row, against 0.5 to 1.5 s for the reference, measured on one 4-vCPU container), its readings kept apart by `+int4mm` in
their model id: on the stand-in 30 of its 64 canary answers matched the reference word for word, against 64 of 64
between two batches of the reference. The study’s registered scripts stay on the checkpoint and the card they were
registered on. Nothing has yet run on Gemma-4-31B from a GGUF; on the CPU only the wiring check’s offline reads have
(29 October 2026: eight prefills of the packed checkpoint on the reference unpacker, 31–51 s each on the home machine’s
12-core CPU; §vi, ARCHITECTURE §4.7). What is left, in order:

- **The real file, against the checkpoint.** Run `compare_gguf_ct.py` on a GGUF of the 31B: if Google’s QAT Q4_0 holds
  the checkpoint’s codes and scales exactly, the two are one model, and the owner can decide whether an index or census
  taken on one stands for the other (today the model ids and the indexes’ weights keep them apart). Then load it, take
  the same tiny census both ways on the card, and run `scripts/compare_int4mm.py` on the CPU, whose rule, fixed before
  any run, says whether int4mm’s readings may stand for the reference’s. Blocked by the file (which repository Google
  publishes it in is unverified, and huggingface.co was not reachable from the machine it was built on) and, for the
  tokenizer, by a cached Gemma 4 tokenizer to compare the GGUF-built one with.
- **The instrument’s sweeps.** A sighting on `GGUFRuntime` is the pinned code with the GGUF accessors underneath, and has
  never run: it reads 60 layers and the J++ lens, so the stand-in cannot hold it. The pinned runtime’s two CUDA memory
  calls are sidestepped by hiding the GPUs on the CPU; mending them in the file itself waits until R4’s record no longer
  needs `engine_runtime.py` pinned (a rerun of R4 would then go from its registered commit). Blocked by the file.
- **A fast path on the CPU with the reference’s bits.** The reference unpacks every weight at every forward (3 to 16 ns
  a weight on the 4-vCPU container, minutes a forward there; a prefill of the 31B took 31 to 37 s on the home machine’s
  12 cores); a lookup-table or compiled unpacker for Q4_0, as `w4a16_fast` has for W4A16, would keep the bits but leave
  the product dense; int4mm for Q4_1 and the K types (Q4_K_M files are common) is a reader and a check each. Blocked by
  the choice, and a machine to time it on.
- **The atlas’s builds and the census on the CPU.** Both take a GGUF and the CPU. The lens build and the wiring are
  products of the weights (hours on a CPU, inferred); the stats and passages passes are a forward per window (days
  even through int4mm); the steer lens stays the GPU’s; a whole census layer is weeks to months (about 21 s a step of 64
  rows through int4mm, inferred), so the CPU serves pilots. The census meta’s runtime record should also carry the
  weights card. Blocked by compute.
- **The CPU’s memory.** The estimates are arithmetic set against the card’s two measurements and lean high by about
  half: a pilot on a CPU should measure a sweep’s and a census batch’s peak and recalibrate them.
- **Beyond the 31B.** Refused today: Gemma 4’s mixture of experts (26B-A4B), its per-layer embeddings (E2B, E4B), its
  shared key-value layers, the IQ types, TQ1_0, TQ2_0, MXFP4, NVFP4, Q1_0 and split GGUF files. Each is a reader and a
  check, but the sweeps’ shapes are the 31B’s, fixed in the pinned `theodolite_math.py`, so the instrument’s own pages
  wait on that pin too.

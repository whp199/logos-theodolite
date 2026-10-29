# Where it runs

*What each hard job costs, on the owner’s RTX 5090 and on rented cards, how to make a card count without changing a
reading, and the long way to the CPU and GGUF. Written on 28 October 2026, and brought up to date on 29 October. Only
the 5090’s figures and the CPU timings of §vii are measured (in the pilot of 27 October, `cbbb5fa` and `8898010`, and in
the runs of 28 and 29 October, §i and §ii); every other figure is worked out from them and the cards’ published
bandwidth and arithmetic, and is to be timed in a pilot before anyone relies on it. Prices were read on 28 October
through web search (the providers’ own pages could not be opened from the machine this was written on) and move often:
read the live page before asking for a budget. Layers count from 0 in the commands, as every script does
(page = layer + 1).*

<br>

## i · the hard jobs

| Job | What it is | On the home 5090 today |
|---|---|---|
| **A layer’s census** | every neuron of a layer pushed on fixed questions, beside matched controls (`census_run.py`; ARCHITECTURE §4.7) | **measured step**: 0.71 s a decode step whatever the batch (1 to 64 rows); a whole `main_rel` layer (21,504 neurons + 256 controls, 217,604 answers) is about 3.6–3.9 days at `--batch 32 --max-new 64` beside a desktop, about two weeks at `--max-new 256`. **With `--unpacker compiled`** (measured step 0.103 s at batch 32, 0.126 s at 96; §ii): about 12.7 hours at `--batch 32`, about 5.2 at `--batch 96`. **Measured `break_rel`** (29 October 2026, its trial at layer 45, `compiled` at `--batch 64 --max-new 48`, 12.4 answers a second while writing): a whole layer (21,504 neurons + 1,024 controls, 24 answers a unit, 540,678 answers) about 12.1 hours, 12.7 with 2,048 controls; about 8.0–8.4 at `--max-new 32` (estimated, not timed). It takes batch 64, not 96: at 96 its chair question's unpushed answer parts from itself after 20 word-pieces |
| The depth curve | 384 neurons + 64 controls a layer (4,484 answers), every layer | about 2 hours a layer, about 120 hours for 60, at `--batch 32 --max-new 64`; with `compiled` about 17 minutes a layer, some 17 hours for 60 |
| Calibrating the doses | `calib_rel` and `calib_ceil` on a few layers, read by `census_tools.py curves` | 64 + 64 units a layer about 1.9 hours (by the measured step), about 16 minutes with `compiled`; the 8 + 8 pilot took about 16 minutes. **Measured** (28 October 2026): both ladders on six layers, 64 + 64 units, `compiled` at `--batch 64 --max-new 64`, 65 minutes (`calib_rel`, about 7 answers a second) and 49 (`calib_ceil`) |
| The steer lens | every neuron of a layer pushed both ways on four contexts, read at every later layer (`atlas_build.py steer`) | by the arithmetic about 15–20 hours for all 60 layers; **measured** (29 October 2026, layer 45, `compiled`; `experiments/results/atlas_steer_timing/`): 287 s for that layer, which runs the 14 layers after it; and layer 5, timed the same day, 1,181 s for the 54 after it. A layer’s cost grows with the layers after it (the forward starts at l + 1): layer 5 took 7% more than layer 45’s rate gives (21.9 s a layer-forward against 20.5). So all 60 come to about 10.1–10.8 hours (inferred, by either layer’s rate over 1,770 layer-forwards; `pod_tool.py`’s projection from layer 45 alone gives 9.7) |
| A corpus pass | the stats index, or the stored passages of every face (`atlas_build.py stats`, `passages`) | **measured**: 58 minutes for 2,000,054 word-pieces (stats, `reference`); the passages over the same windows 20 minutes with `compiled` (28 October 2026; its log in `experiments/results/atlas_wiring_check/gpu_logs/`) |
| The wiring | weight products of every pair of layers (`atlas_build.py wiring`) | **measured**: about 4 minutes (again on 28 October, keeping each link's g and u); its measured check (`wiring_check`, 512 writers a layer, the night of 28–29 October) 29–35 s a layer with `compiled`, 59 layers in about 31 minutes, before it kept each reader's g₀ and u₀ (the logs and the check's own cost line in `experiments/results/atlas_wiring_check/`) |
| The writer’s noise floor | how far an unpushed answer agrees with itself in other batch shapes (`scripts/verify_atlas_writer.py`) | about 18 minutes (by the measured step); **measured**: 2 minutes with `compiled` (28 October 2026) |
| A label survey | the causal check of ROADMAP §ii condition 4 for every label at pages 41–59 (408,576 neurons in N1A’s 36-text layout) | about 2 × 10²⁰ multiply-adds: some 17 days (inferred) |

Why the census is slow is now known (`research` of 28 October, read from the code): every forward unpacks all 29.3
billion 4-bit weights to bf16 with five separate torch operations over int32 temporaries, about 32 bytes of memory
traffic a weight, where the products themselves read 2. The products are under a tenth of the step until the batch passes
about a hundred rows, which is why the step costs the same from 1 to 64. The same arithmetic reproduces the stats pass
(59 minutes against 58 measured) and the calibration pilot (16.9 minutes against about 16).

<br>

## ii · making the card count, with every reading the same

Built on 28 October 2026 and checked on the CPU stand-in; the unpackers timed on the home 5090 that evening, the rest
not yet:

- **Faster unpacking, the same bits** (`backend/w4a16_fast.py`, `--unpacker`, `THEODOLITE_UNPACKER`; ARCHITECTURE §2).
  `lut` reads the packed bytes through a 256-entry table and multiplies by the scale as the runtime does; `compiled` is
  the runtime’s own arithmetic compiled alone, the model left eager with its hooks; `resident` keeps every weight
  unpacked (about 60 GB beside the 4-bit weights). Each gives the runtime’s bits: `scripts/check_dequant.py` shows it
  for every (piece, bf16 scale) pair and for a 4-bit stand-in’s logits, vault edits, pushed answers and census rows,
  and `install` tests every linear again on the card, refusing in plain words on any difference. None of it touches
  the R4-pinned `engine_runtime.py`. By the arithmetic `lut` shortens the step two to three times on a 5090 and
  `compiled` about five times. **Timed** on the home 5090 beside a desktop (28 October 2026, `scripts/pod/pod_tool.py
  steps`, layer 45, census-shaped batches at `--max-new 64`, the lens freed;
  `experiments/results/atlas_unpacker_steps_rtx5090.json`), each passing its test on all 410 linears first:

  | rows | `reference` | `lut` | `compiled` | torch's peak above the model |
  |---|---|---|---|---|
  | 1 | 0.706 s | 0.275 s | 0.091 s | 0.3–1.0 GiB |
  | 32 | 0.713 s | 0.282 s | 0.103 s | 2.4–3.1 GiB |
  | 64 | 0.721 s | 0.287 s | 0.109 s | 4.8–5.3 GiB |
  | 96 | 0.743 s | 0.303 s | 0.126 s | 7.2–7.5 GiB |

  `lut` is about 2.5 times shorter and `compiled` about seven (its graphs compile in about 4 s at load). With
  `compiled` the step grows with the batch, the products now counting beside the unpacking; 128 rows and more were
  not tried.
- **The lens freed and the steps read whole.** The census frees the J++ lens after loading (about 3.2 GiB more for
  rows, worked out) and reads each step’s pieces back in one transfer, bit for bit as before.
- **The batch fixes the bits, the unpacker does not.** On the GPU a row’s numbers depend on the size of its batch, so
  the batch and the batch plan are resume keys: measure what a card holds once, then keep that `--batch` for every
  continuation and shard of a run. Each run records the runtime it was written on (the card, the versions, the commit,
  the stats index’s sha256) and warns when that changes.
- **Shards.** `--shard i/n` splits a run on whole planned batches, every shard but the first writing a canary batch;
  `census_tools.py merge` knows shards by their metas, refuses a missing shard or answer and compares the canaries, so
  a merged run is known to be the run an unbroken one would have written, or says where it is not.

Not for a census of record without more: a fused 4-bit matrix product (Marlin and its kin), the steering fork of vLLM
that neuron-explorer’s sweep used, a static cache with CUDA graphs, compiling the whole model, and splitting a model
across cards. None gives the same bits, and the fork cannot yet spare the start mark as the census’s push does. Each
could serve a run marked as from another engine, after an equivalence check registered before it is read (the shared
prefix of greedy answers, the largest change of a chosen piece’s log-probability, the overlap of the five likeliest,
and the ranking’s leads, against the runtime’s own disagreement with itself across batch sizes).

<br>

## iii · which card

**Option zero: the home 5090, headless.** With the desktop off, the lens freed and the batch raised to what the card
then holds (about 96–128 rows at `--max-new 64`, to be measured), a whole layer at `--max-new 64` is about a day on the
reference unpacker and, with `compiled` as timed beside a desktop (§ii), about 5 hours at a batch of 96, less if the
card holds more. It costs nothing, and it is the study’s own card class, so the instrument’s recorded numbers carry
over. Calibration and the first whole layers belong here.

**Rented, for breadth** (the steer lens, a depth curve over all 60 layers, several whole layers, a large corpus pass, a
label survey):

| | Default | Why | On demand, 28 October 2026 |
|---|---|---|---|
| an interactive session | RunPod Secure Cloud, one **RTX 5090**, a network volume in the same datacentre, the page through an SSH tunnel | the study’s card class (GB202, sm_120): its GPU checks and recorded numbers carry over | $1.19/h Secure (Community $0.69, with no network volume); about $5 for four hours |
| batch jobs | RunPod Secure **H200 141 GB**, one pod a shard on one volume; Vast.ai or Hyperbolic H200 as a second source | 4.8 TB/s, and room for a batch of about 500 rows at `--max-new 256` (to be measured: 512 is at the edge) | $5.29/h Secure; about $4.00 elsewhere |
| if `resident` is wanted on one card | RTX PRO 6000 96 GB | the cheapest card that holds the unpacked weights with room for a batch | $2.49/h Secure (Vast.ai median $1.54) |

RunPod’s Secure rates rose twice in the six weeks before (the H200 from $4.59 to $5.29 on 12 October; fifteen of 21
cards up 11–20% on 15 October). The hyperscalers ask two to three times as much on demand, often in eight-card shapes
only; their spot prices are near the neoclouds’, and a census resumes after a pre-emption.

**A whole `main_rel` layer**, hours on one card (the two research reports’ estimates, which differ by up to half; the
5090’s first row is measured, every other is arithmetic):

| Card | `--max-new 64`: today’s unpacker | the same, `compiled` | `--max-new 256`: today’s | the same, `compiled` |
|---|---|---|---|---|
| RTX 5090, beside a desktop, batch 32 | 87–94 (measured step) | 12.7 (measured step) | 340–375 | — |
| RTX 5090, headless, batch ~120 | 24 | 5 | 280–300 | 39 |
| H100 80 GB SXM | 3.4–3.8 | 1.1 | 24–35 | 8 |
| H200 141 GB | 2.5–2.8 | 0.9 | 10–15 | 4.6 |
| B200 180 GB | 1.5–1.7 | 0.5 | 6–7 | 2.4 |

At $5.29 an H200-hour that is about $13–15 a layer at `--max-new 64` today, about $5 with `compiled`; all 60 layers at
`--max-new 256` are hundreds of GPU-hours either way, not a job to start before the depth curve says which layers are
worth it.

**The other jobs on an H200** (arithmetic): the depth curve over 60 layers about 3 hours at `--max-new 64` today (about
12.5 at 256), about an hour with `compiled`; the steer lens about 3–4.5 hours for all 60 layers (bound by arithmetic:
the unpacker barely matters, and four pods by depth, layers 0–7, 8–16, 17–29 and 30–59, balance it); a corpus pass of 2
million word-pieces about 20 minutes, 20 million about 3.5 hours; the label survey about 3.7 days of one H100, some 5
hours of an eight-card B200 node (sharing each text’s prefix would cut it about 45 times, but changes N1A’s layout, so
the causal check would have to pass `verify_causal_check.py` against N1A and N1V again).

<br>

## iv · the order

1. **Calibrate the doses**, which every later spend waits on (ROADMAP §vi): read the 8 + 8 pilot already taken
   (`census_tools.py curves <layer> --weights`), then `calib_rel` and `calib_ceil` on four to six layers (`census_run.py
   --layers 20,30,40,45,50,57 --design calib_ceil --neurons random:32 --random-controls 16`), on the home card.
   **Taken** (28 October 2026: both ladders at six layers, 64 + 64 units; `experiments/results/atlas_calibration/`).
   `break_rel`’s doses, ±1 and ±2 of the stream, were chosen from them and are counted calibrated at layers 40–50 alone
   (pages 41–51), the owner’s choice of 29 October 2026 on its trial (`census.DOSES_CALIBRATED`;
   `experiments/results/census_break_rel_trial/`); the doses at every other layer, and `main_rel`’s, are not yet
   calibrated (ROADMAP §vi).
2. **A pilot on the card that will do the work**, one hour: the decode step and peak memory for each unpacker at several
   batches, the unpacker self-tests, the batch the card holds, the writer’s noise floor, one steer-lens layer timed, a
   canary batch against one written on the home card, the size of a run file once compressed (`scripts/pod/pilot.sh`).
   Its numbers then replace this page’s arithmetic and the scripts’ help, as `cbbb5fa` did for the 5090.
3. **The steer lens**, all 60 layers (shards by depth), once its dose is chosen: about 10.1–10.8 hours on the 5090 (§i).
4. **A passages pass** of 2 to 10 million word-pieces, the wiring rebuilt with the links’ gate and up values, and the
   wiring’s measured check. **Done** for 2 million word-pieces (the night of 28–29 October 2026, §i), and the check run
   again with `--fresh` keeping each reader’s g₀ and u₀ (29 October, about 32 minutes; ROADMAP §vi).
5. **The depth curve**, at the dose and `--max-new` the calibration chose.
6. **Whole layers**, chosen from the depth curve, each on one card class at one `--batch`.
7. **A label survey**, if wanted, once a resumable driver exists (a small CPU job, not yet written).

<br>

## v · on a rented card

`scripts/pod/` holds what a pod runs, each with `--help` and a `--dry-run` that runs its whole flow on the CPU stand-in
in a temporary root (`scripts/check_pod_scripts.py` runs every one so): `setup.sh` (the volume, the venv from a
constraints file, the pinned downloads, a self-test naming the card, the versions, the unpacker’s test and the runtime
class), `pilot.sh` (the hour of §iv’s second step, ending in “proceed” or “stop” against an approval),
`census_shard.sh` (one shard of a layer’s census, detached, under a timeout at 1.25 times its projection, holding the
card’s lock) and `sync.sh` (the runs, metas, canaries and logs pulled home every few minutes, and a last pull checked
against what each meta names). At home, `scripts/export_constraints.sh` writes the venv’s exact versions for
`setup.sh`, `census_run.py --canary-only` writes the canary batch a pod compares its own with (`census_tools.py
canary`), and `scripts/gpu_log.py` records each approval and every pod’s start and stop and sums them. None has yet run
on a rented card. What they encode:

- **One volume.** The model (about 22 GB, at the README’s snapshot), the lenses (6.4 GB), the venv, the atlas’s
  indexes (`THEODOLITE_ATLAS_DIR`), the census runs and the field notes and saved prompts (`XDG_DATA_HOME`) all live on
  one network volume, so a stopped pod restarts ready and nothing written in a session is lost with the pod. Every
  shard of a run reads the same stats index (the merge refuses p50s that differ), so copy the home one up.
- **The model’s access.** Whether the checkpoint needs a Hugging Face token is unsettled (the repository calls it gated;
  one tracker says Gemma 4’s repositories are not). Try without one; if refused, a fine-grained read-only token scoped
  to that repository, given to the download alone, never written to the volume and never in the server’s environment.
  The scientist’s CLIs are started without Hugging Face and cloud credentials in any case (`child_env`).
- **The page through a tunnel, never a port.** The server binds 127.0.0.1 and answers only pages and scripts of the
  machine it runs on (`backend/local_only.py`): `ssh -N -L 8765:127.0.0.1:8765 …` from the owner’s machine works
  unchanged, while a provider’s HTTP proxy is refused, as is any peer beyond loopback should the server be started
  on `--host 0.0.0.0`. Start pods from a template without
  Jupyter, with only port 22 open. The lab daemon’s default port is also 8765: never tunnel both.
- **The scientist on a pod.** None on a batch pod. On an interactive one, a key kept for it with a low spend limit,
  revoked after, or an endpoint; never `THEODOLITE_SCIENTIST_REACH=home` on a machine whose environment may hold cloud
  credentials.
- **Cautions.** The pinned torch (2.14.1+cu130) needs a CUDA 13 driver: filter for it. A new card class is a new
  instrument until checked: run `verify_causal_check.py` and the pilot on it once; Hopper and Blackwell kernels differ
  from the 5090’s, so a greedy answer at a near tie may fall the other way. Pull every file a run’s meta names before
  a pod is deleted.

<br>

## vi · the budget

After R3’s practice (D30, A15: “used 5.54 of the 8 GPU-hours the user approved”):

1. The owner approves ahead, in GPU-hours on a named card and provider, with its dollar ceiling at the rate read that
   day (“12 H200-hours on RunPod Secure at ≤ $5.29/h: ≤ $64”), plus storage (“a 100 GB volume for a month: about $7”),
   written down beside the job before it runs. Hours are summed over pods: four shards of three hours are twelve.
2. A pilot on the same card class first, at the intended `--batch` and `--max-new`; its projection times 1.25 must fit
   the approval, else stop and ask.
3. Every pod logged (`scripts/gpu_log.py approve`, `start` and `stop`: provider, card, rate, job, the approval it counts
   against; `start` refuses a pod that names no approval on record); `gpu_log.py sum` gives the line the write-up
   ends with, “used X of the Y GPU-hours ($a of $b) the owner approved”, and names a pod on another card or at a
   higher rate as a deviation.
4. Hard stops: prepaid credit no larger than the approval, each batch job under `timeout` at 1.25 times its projection,
   interactive pods stopped at the end of the session.
5. A change of card or provider after the pilot is a deviation, recorded as R3 recorded D30–D36.

<br>

## vii · the CPU and GGUF

The owner’s aim (28 October 2026) is every part of the instrument on the CPU and from a GGUF, not only the W4A16
checkpoint on one GPU. The first layer of it is built and checked on the CPU stand-in (ARCHITECTURE §5,
`scripts/check_gguf.py`): torch unpackers for thirteen GGUF types equal to gguf-py’s bit for bit, a packed GGUF layer
the vault can edit, a Gemma 4 built from a GGUF with its provenance, `GGUFRuntime` beside the pinned runtime, one
choice of runtime from the environment (`THEODOLITE_GGUF`, `THEODOLITE_DEVICE`), and `scripts/compare_gguf_ct.py`,
which asks whether a GGUF holds the checkpoint’s own weights. A second layer followed the same day: the standalone MCP
server and the lab daemon choose their runtime the same way, the lab’s lens fits have a GGUF twin of their 4-bit
autograd function (its gradients the dense stand-in’s bit for bit), every index names the weights it was built from and
the server refuses one built from others, and a reading the CPU’s memory cannot hold is refused in plain words. A
reading off the GPU, or from a GGUF, carries that in its model id, so it never mixes with the card’s in one file.

What a CPU costs, measured on a 4-vCPU container: unpacking every weight on every forward (7.8 ns a weight for Q4_0,
10.0 for the runtime’s W4A16 unpacker) makes a forward of 29.3 billion weights some minutes there; resident bf16 (about
62 GB of RAM) writes a word-piece in seconds. The 31B itself has run on the CPU only for the wiring check’s offline
reads (29 October 2026, on the home machine’s AMD Ryzen 9 7900X, 12 cores and 24 threads, the reads using 22, the packed
checkpoint on the reference unpacker): a prefill of 17 to 20 word-pieces took 31 to 37 s, and one of 25 rows of 17
word-pieces 43 to 51 s (`experiments/results/atlas_wiring_check/`). torch’s own int4 kernel for the CPU
(`THEODOLITE_GGUF_LINEARS=int4mm`, or `THEODOLITE_CHECKPOINT_LINEARS=int4mm` for the checkpoint) takes a 5,376 × 21,504
layer at one row in 1.7 to 8 ms against 0.5 to 1.5 s for the reference, about 2 s a written word-piece of the 31B by the
arithmetic, but its sums are the kernel’s, not the reference’s: its readings carry `+int4mm` in their model id, and on
the stand-in 30 of 64 canary answers matched the reference’s word for word, where two batches of the reference matched
64 of 64. So the CPU suits weight reads, single forwards, pilots and looking through int4mm now; a census or a corpus
pass of record on it waits on a fast path with the reference’s bits (`scripts/compare_int4mm.py` holds the rule, fixed
before any run on the 31B, by which int4mm’s readings could stand for the reference’s). What is left, phase by phase, is
in ROADMAP §vii.

<br>

## viii · sources

Read on 28 October 2026 through web search; the dates in brackets are the snapshots the pages give.

- RunPod: <https://www.runpod.io/pricing-page> as indexed (July 2026); the Secure-cloud rises of 20 September and 15
  October 2026, <https://www.usagepricing.com/blueprint/activity/runpod-2026-10-15-secure-cloud-second-hike>, and of
  5–12 October, <https://pricingsaas.com/companies/runpod/diffs/2026W42>; network volumes, SSH over exposed TCP, secrets
  and `runpodctl` in <https://docs.runpod.io/storage/network-volumes>, <https://docs.runpod.io/pods/configuration/use-ssh>,
  <https://docs.runpod.io/pods/templates/secrets> and <https://docs.runpod.io/runpodctl/reference/runpodctl_create_pod>.
- Vast.ai: the medians of the Singularity Research tracker, <https://www.jinacapital.com/gpu-prices> [25 October 2026];
  its CLI, <https://docs.vast.ai/cli/get-started>.
- Lambda, <https://usagepricing.com/blueprint/lambda-labs> [June 2026]; Modal,
  <https://hostfleet.pages.dev/serverless-gpu-pricing-matrix-2026/> [April 2026]; Hyperbolic,
  <https://usagepricing.com/blueprint/activity/hyperbolic-2026-08-04-price-change> [August 2026]; Together,
  <https://pricingsaas.com/companies/together/diffs/2026W26> [June 2026]; CoreWeave, <https://coreweave.com/pricing>.
- Google Cloud's G4 (one RTX PRO 6000), <https://gcloud-compute.com/g4-standard-48.html>; AWS p5.4xlarge,
  <https://calculator.holori.com/aws/ec2/p5.4xlarge> and its spot history <https://compute.doit.com/spot/us-east-1/p5.4xlarge>;
  Azure, <https://sparecores.com/server/azure/Standard_NC40ads_H100_v5>.
- Gemma 4's licence and whether its repositories are gated: <https://vorplabs.com/models/open-weights-tracker> [reviewed
  4 August 2026]; unsettled, since huggingface.co could not be reached to look.
- The cards' bandwidths and arithmetic are NVIDIA's published figures as recalled, used only for the estimates above.

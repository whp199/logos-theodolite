# Take a layer of the census

Logos Theodolite is mapping what each MLP neuron of Gemma-4-31B writes. The census takes one layer at a time. Each of
the layer's 21,504 neurons is pushed along its own write direction, at four strengths (±1 and ±2 of the stream), on six
plain questions. Its 24 greedy answers are then read against those of 2,048 random directions pushed the same way.
Layer 45 is done: 13.2 hours on one RTX 5090, and it passed every check set for it before it was run
(`experiments/results/census_L45_whole/`).

Ten layers are left: **40, 41, 42, 43, 44, 46, 47, 48, 49 and 50**. These are the layers where the push strengths have
been calibrated. If you have a big enough card and a free night, you can take one. Layers count from 0, as every script
counts them (the app's page = layer + 1).

**The board** shows which layer is whose: the pinned *Census layer board* issue, <https://github.com/whp199/logos-theodolite/issues/1>.

## What you get

When the census is done, the atlas and its data are published, and the whole instrument runs on the same card you took
your layer on:

- **Chat with Gemma with any neuron turned up or down.** The atlas's push tab writes three answers side by side, over
  several turns: the neuron pushed up, left alone, and pushed down. You find neurons by the words they write, or by
  their rank in the census.
- **Watch a reply take shape** through all 60 layers, read through a lens at each one. The lens was fitted on the base
  model, and the pages say where it can be trusted.
- **Point your own local model at the map.** The scientist works with any OpenAI-compatible endpoint (LM Studio,
  Ollama, llama.cpp's server). It runs pushes and texts, names what it finds, and points at it on the map.
- **Your name on the first public map of a 31B's neurons.** We found none published, only sparse autoencoders
  (`REFERENCES.md` §J). Your layer is credited to the name you gave, and you name the strongest neurons of your layer
  in the published atlas.
- **Open data**: every answer of every layer, 565,254 a layer, published with the project.
- **Ready now: a neuron in your own frontend.** Any neuron exports as a llama.cpp control vector, for `llama-server`
  and SillyTavern or anything else on it. Keep the scale near ±1 (*Use a neuron in your own setup*, below).

A census rank says what a neuron's push does to six fixed questions at four strengths. What it does in your own
chats, you try for yourself.

## What you need

- **One NVIDIA card with at least 32 GB**, for example an RTX 5090, RTX PRO 6000, A6000, 6000 Ada, L40S, A100, H100 or
  H200. The 4-bit model is about 22 GB and a batch of answers about 5 GB more, and it is not split across cards, so a
  24 GB card (3090, 4090) cannot take part.
- **A driver for CUDA 13**: `nvidia-smi` shows `CUDA Version: 13.0` or later (driver 580 or newer).
- **Linux**, about **60 GB** free disk, 32 GB of memory, and `git`, `curl`, `zstd`, `tar` and `flock` (util-linux),
  and a C compiler (`gcc`), which Triton may need for its kernels.
- **Time**: about 13 hours for a layer on a 5090, plus about an hour of setup, mostly downloading about 35 GB. Other
  cards will differ (an Ampere card perhaps two to three times slower; that is a guess, not a measurement), and the
  status line tells you yours. The run can be stopped and continued at any point.
- **A quiet machine**: the card at full load, and nothing else heavy on it or on the CPU. Our own run was once stopped
  by NVIDIA's watchdog in the middle of a big kernel build.

No account, token or payment is needed anywhere.

## The steps

```bash
mkdir -p ~/theodolite-census && cd ~/theodolite-census
git clone https://github.com/whp199/logos-theodolite.git && cd logos-theodolite
git checkout census-kit-1
scripts/volunteer/census.sh check                      # can this machine take a layer?
```

1. **Pick an open layer on the board and claim it.**

   ```bash
   scripts/volunteer/census.sh reserve 47 --name yourname
   ```

   This prints one line, `CENSUS RESERVE L47 claim=… card=… name=…`. Post it in the thread, or as a comment on the
   board issue. The claim's secret stays in `~/theodolite-census/claims/`, and the bundle you send carries it: that is
   how it is known to come from your claim. The board is kept by hand, so give it a while, and pick another layer if
   someone was there first.

2. **Set up** (about an hour, mostly downloads):

   ```bash
   scripts/volunteer/census.sh setup
   ```

   This builds a Python venv with the exact versions our card used, then downloads:
   - the checkpoint, `google/gemma-4-31B-it-qat-w4a16-ct` at a pinned revision;
   - Neuronpedia's lens files;
   - a 31 MB stats index from this repository's release. The push strength is read from it, so every layer is pushed
     alike, and its sha256 is checked.

   The downloads, the venv and the runs all stay under `~/theodolite-census` (Triton keeps its compiled kernels in
   `~/.triton`), and your `~/.bashrc` is left alone.

3. **The canary** (a few minutes):

   ```bash
   scripts/volunteer/census.sh canary 47
   ```

   It writes the run's first batch on your card and sets it against the same batch written on ours. It also measures
   how far an unpushed answer agrees with itself across batch shapes on your card. Either the bits are the same as
   ours, or your card is a *card class of its own*. Both are fine: they are checked differently (below).

4. **Run the layer:**

   ```bash
   scripts/volunteer/census.sh run 47
   ```

   It runs detached, so you can close the terminal. Post the `CENSUS START` line it prints. If it stops (a reboot, a
   crash, Ctrl-C), run the same command again: it continues answer by answer, and the answers come out exactly as an
   unbroken run's would.

5. **Check on it** now and then:

   ```bash
   scripts/volunteer/census.sh status 47
   ```

   It prints the answers written, the rate, when it should finish, and a `CENSUS PROGRESS` line. Post that line every
   so often: it keeps your claim.

6. **Pack it** when it is whole (10–20 minutes):

   ```bash
   scripts/volunteer/census.sh pack 47
   ```

   It checks the run is whole and taken with the kit's settings, then packs about 500 MB into
   `~/theodolite-census/outbox/L47-break_rel-<claim>.tar`. Upload that somewhere anyone can download it for a week
   or more: a Hugging Face dataset, a torrent, a file host. Post the `CENSUS DONE` line with your link in it.

If you can't finish, run `scripts/volunteer/census.sh release 47` and post the line it prints. That frees the layer
for someone else.

## The rules

- **One layer at a time per person**, and claim it before you start the run.
- **Don't change anything**: not the code, the settings or the batch. A run of anything else can't stand beside the
  others. The script refuses to run from a checkout that isn't at the kit's tag with nothing changed.
- **One card from start to finish**, and one GPU job on it at a time. If a run has to move to another card, say so
  when you post it (the run records it anyway).
- **How long a claim holds**:
  - 36 hours from `RESERVE` to `START`;
  - then until the end time in your latest `PROGRESS` line, plus 24 hours;
  - never more than 10 days in all.

  A lapsed claim opens the layer again.

## How a run is checked

These rules were fixed on 30 October 2026, before any volunteer's run had been seen
(`backend/census_volunteer.py`, `scripts/census_intake.py`).

1. **The bundle**: each file's sha256 against the manifest, and the claim's secret against the code you posted.
2. **The run**:
   - every setting, the job list, the stats index and the push strength must match ours at that layer;
   - the commit must be the kit's tag, from a clean checkout;
   - every one of the layer's 565,254 answers must be there exactly once.
3. **The canary**:
   - against your own run's first batch (the same card writing the same batch);
   - against ours (the same bits, or a class of its own);
   - your six unpushed answers against ours.
4. **The spot-check**: when the bundle arrives, 5 of the run's 737 chunks are drawn at random, so nobody can know them
   in advance. Each is written again on our 5090, about a minute a chunk. Then:
   - **The same bits as ours**: every spot answer must match bit for bit.
   - **A card class of its own**: the run must meet all of these:
     - at least 90% of answers start with the same word-piece as ours;
     - on the word-pieces both cards wrote alike, where ours was not sure of its choice, the log-probabilities differ
       by a median of at most 0.01, and by at most 0.1 at the 99th percentile;
     - each unpushed answer shares at least 16 word-pieces with ours.

   Another card's sums move a log-probability by about a thousandth, while a run at another dose, at another layer or
   made up moves it far more.

A layer from another card class is recorded as such beside the run (`L<layer>-break_rel.provenance.json`). Its
neurons and its controls were pushed on the same card, so the layer's own reading stands. Wherever layers are compared
with each other, the card is named.

## What you send, and what runs on your machine

- **The bundle** holds:
  - the answers;
  - the run's record: your card's name, the torch, CUDA and library versions, and the commit;
  - the canary and the writer's floor;
  - a manifest with your claim's code, its secret and the name you chose.

  It holds no logs, folder names, host names or addresses. The logs stay on your machine.
- **Downloads**:
  - from GitHub: the code and the stats index;
  - from Hugging Face: Google's checkpoint and Neuronpedia's lens files, each at a pinned revision;
  - from PyPI and PyTorch's index: the Python packages, through `uv`, which setup installs from astral.sh if it is
    missing.
- **What runs**: everything runs as you, inside `~/theodolite-census` (and Triton's cache in `~/.triton`), and the census runs with every environment
  variable naming a token, key or secret removed. Please read `scripts/volunteer/census.sh` and
  `scripts/pod/setup.sh` before you run them.
- **Publication**: what you send is published with the project (Apache-2.0) and credited to the name you gave, or to
  "anon".

## If something goes wrong

- **`check` says no**: the line says why.
- **The canary refuses "other settings"**: the checkout isn't at `census-kit-1`, or the stats index isn't the kit's.
- **The run stopped**: run the same `census.sh run` again. The log is `~/theodolite-census/logs/L<layer>-break_rel.log`.
- **Out of memory**: close everything else on the card. The batch is fixed at 64, so a card that can't hold it
  can't take part.
- **Anything else**: ask in the thread or on the board issue.

## Use a neuron in your own setup

Any neuron the census ranks, or that you find in the atlas, can be loaded into llama.cpp as a control vector. Then
`llama-server`, and SillyTavern or any other frontend on it, writes with that neuron's push.

```bash
.venv/bin/python -I scripts/export_control_vector.py --layer 45 --neuron 13641     # writes theodolite-L45-n13641.gguf
llama-server -m gemma-4-31B-it-QAT-Q4_0.gguf --control-vector-scaled theodolite-L45-n13641.gguf:1
```

- **The layer** counts from 0. The atlas's page, the first number in its address (`#/atlas/46/13641`), is layer + 1,
  and `--page 46` takes it as it is.
- **The scale is the dose.** At 1 the push is as long as the stream's typical length at that layer: the census's +1.
  `:-1` pushes the other way. **Stay near ±1**: at ±2 llama.cpp's text collapses, because it also pushes the first
  token, which the census left alone. Other values work, but the census measured only ±1 and ±2. For several
  neurons: `--set 45:13641,44:801:-0.5 -o mix.gguf`.
- **What it reads**: two tensors of the layer and the 31 MB stats index, on the CPU, in seconds. Nothing is downloaded.
  - With the kit set up, run `. ~/theodolite-census/env.sh` first: it finds the checkpoint and the stats index you
    already have. The script is not in the kit's tag, and `census.sh` runs only from a clean checkout at it, so use
    a second checkout: `git fetch origin && git worktree add --detach ~/theodolite-census/export origin/main`, and
    run the kit's Python (`~/theodolite-census/logos-theodolite/.venv/bin/python`) from there.
  - Without the kit: add `--gguf` with your own GGUF of Gemma-4-31B-it and `--stats` with the kit's stats index
    (`https://github.com/whp199/logos-theodolite/releases/download/census-kit-1/act_scale.npz`, sha256
    `23a6dcefaf4eae99222c334a8294f773e37e4f20cbf36b63344a73fd5c2911fa`). It needs Python 3.12 with numpy, torch (the
    CPU build is enough) and gguf. The direction is then your file's own, a little off the census's: cosine 0.997 for
    the QAT Q4_0 file, 0.977 at worst for the Q4_K files we tried.
- **`--show FILE`** prints what a file was made from: the model, the weights, each neuron, dose and p50, the stats
  index's sha256 and the commit.

**What a census label can and can't tell you.** The census read each neuron from six questions, at ±1 and ±2, one
neuron at a time, from greedy answers of 48 word-pieces with thinking off, on the 4-bit W4A16 checkpoint. A neuron's
words are those its pushed answers used more often than 2,048 random pushes did: a lead, not a proof. Three things
differ when you load the file:
- the push is along the neuron's write direction, and about 1.8 to 2.5 times longer than anything the neuron can
  write by itself, so it is not "the neuron turned up";
- llama.cpp also pushes the first token (`<bos>`), which the census left alone;
- your GGUF is not the W4A16 checkpoint.

We checked it on the 31B for layer 45's two top neurons, against a bar fixed before the run (30 October 2026,
`ARCHITECTURE.md` §6). At ±1 their words came out in llama.cpp in all six answers, as in the census: 13641 at −1 wrote
"I am a own-company-wide staff meeting all-hands…", and 21200 at +1 "…young man, sir". At ±2 llama.cpp's answers
collapsed into "//" or blank space and lost the words, as the instrument's own do when it pushes the first token too.
Two neurons are not a layer. Sums of neurons, other scales, sampling, long chats and thinking were never measured.

## For the owner

- `scripts/census_board.py` keeps the board:
  - `add` takes `CENSUS` lines pasted from the thread;
  - `sync` reads them from the board issue's comments;
  - `render` writes `volunteers/BOARD.md`, `board.txt`, `board.png` and the guide image `volunteers/post/guide.png`
    (the thread post is `volunteers/post/thread.txt`);
  - `publish --yes` updates the issue.
- `scripts/census_intake.py` brings a bundle home: `check`, then `spot --run` (with the server stopped), then `judge`,
  then `accept`. `home-canaries` takes the canaries a kit's runs are set against.
- `scripts/check_volunteer_kit.py` runs all of it on the CPU stand-in.
- The kit's settings are in `volunteers/kit.env`.

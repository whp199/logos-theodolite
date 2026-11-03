# Gemma-4-31B from three Q4 GGUFs on the GPU, the reference unpacker against `compiled`

2 November 2026, the home RTX 5090 (32 GB, of which the desktop held about 5), torch 2.14.1+cu130, triton 3.8.0,
`run.py gpu`: the whole model on cuda:0, every quantised layer packed, in bf16. Three files:
`gemma-4-31B-it-QAT-Q4_0.gguf` (lmstudio-community: 410 Q4_0 linears; fingerprint `2ac81237f76238b4`),
`gemma-4-31B-it-Q4_K_M.gguf` (lmstudio-community: 355 Q4_K and 55 Q6_K; `9d70e76bba193a99`) and
`gemma-4-31B-it-UD-Q4_K_XL.gguf` (Unsloth, the copy in `ai-models/`: 280 Q4_K, 70 Q6_K, 36 Q5_K and 24 IQ4_XS;
`adfe6f42c3758a08`). Each was measured on the reference unpacker, then `compiled` (`backend/gguf_fast.py`) was
installed in place and the same readings taken again: three forwards of "The capital of France is" (the first warming),
a chat prefill of 23 word-pieces, "Name three rivers in Europe, one line each." written greedily, and 8 word-pieces
written at a batch of 32 copies of that prompt. The logits were compared by their bits and the pieces written piece
for piece; the memory is torch's peak, in GiB (2^30 bytes).

`qat_q4_0.json`, with the J++ lens: loaded in 22.3 s, 21.2 GiB on the card. The reference wrote "Danube / Rhine /
Volga" and its end mark, 9 word-pieces, at 0.542 s a step after its 0.553 s prefill, and 0.553 s a step at a batch of
32. `compiled` installed in 2.8 s (8 graphs; each of the 410 layers' whole weight, and every one of the 65,536 patterns
of the Q4_0 scale through each graph, the reference's bits) and wrote the same at 0.121 s a step after a 0.139 s
prefill, 0.131 s at 32. The optics check's first sighting (`scripts/verify_optics.py`'s prompt, 60 layers through the
lens, 6 written: " 8.\n\nThe number") took 5.22 s on the reference and 1.92 s compiled; taken twice on the reference
and once compiled, every number of it but its clock and memory gauge was the same. Peak 22.71 GiB on the reference,
23.05 compiled.

`q4_k_m.json` and `ud_q4_k_xl.json`, no lens: loaded in 21.5 and 20.5 s, 18.93 and 19.18 GiB on the card. The
reference wrote the same answer at 0.819 and 0.758 s a step, 0.736 and 0.737 s at 32; `compiled` installed in 8.7 and
10.4 s (10 and 19 graphs; Q4_K's and Q5_K's two fp16 fields 131,072 patterns, Q6_K's and IQ4_XS's one 65,536), and
wrote it at 0.126 and 0.121 s a step, 0.136 and 0.132 s at 32. Peak 20.57 and 20.89 GiB on the reference, 20.33 and
20.57 compiled.

In all three, through `compiled`, the logits of the forward and of the prefill were the reference's bits and the pieces
written alone and at 32 the reference's pieces, and the model id did not change. No call the graphs serve was sent
back to the reference (`compiled_fell_back` 0); the row and column slicers a sighting reads (`slice_w_*`,
`write_col_norms`) stay on the reference by design, so the sighting's agreement tests `compiled` through the forward
only. The reference's step moves between runs: the day's first runs (`first_run.py`, the reference alone, on the code
before `compiled` existed: `first_run_qat_q4_0.json`, `first_run_q4_k_m.json`, `first_run_ud_q4_k_xl.json`) wrote a word-piece in 0.512, 0.689 and 0.695 s,
peak 19.34, 20.31 and 20.46 GiB, no lens, so a file's step moved by 6 to 19% between two runs (the Q4_0's first run
without the lens; the cause not established). Set beside the reference of the same run, `compiled` is about 4.5 times
shorter on the Q4_0 and 6 to 6.5 on the K files at one row, 4.2 and 5.4 to 5.6 at a batch of 32: a band, not a value.

`first_run_q4_k_m_resident4.json` (`first_run.py --resident-gb 4`, the budget `THEODOLITE_GGUF_RESIDENT_GB=4` would
set): the Q4_K_M with 23 Q4_K and 8 Q6_K layers resident (3.99 GB), on the reference: 0.643 s a step, peak 22.84 GiB.

`qat_q4_0_cpu.json`, `run.py cpu`: the Q4_0 on the 12-core CPU (an AMD Ryzen 9 7900X, 61 GB, 12 threads), no lens, one
forward each way. The forward took 37.2 s on the reference and 11.9 s compiled, its logits the same bits; the install
took 64.3 s (Inductor's C++ compiled, then the tests), and through its graphs the table gave 16,466,963 NaNs whose
bits differ from the reference's NaNs (each a NaN where the reference has one; from infinite or NaN fp16 scales, which
no finite file holds), counted and said in the log. Against the GPU's forward of the same file (both packed, both the
reference) the same five likeliest pieces in the same order, the largest |Δ log-prob| among them 0.25; the day's first
pair of runs (`first_run_qat_q4_0.json`, `first_run_qat_q4_0_cpu.json`) gave a KL of 0.0015, worked out afterwards
from their saved log-probabilities (not kept), and 0.25 again over the top 50. The two devices' sums differ, which is why their model ids do
(`@cpu/bfloat16`).

`compare_gguf_ct_qat_q4_0.json`: `scripts/compare_gguf_ct.py` on the QAT Q4_0 against the cached W4A16 checkpoint. The
421 F32 tensors (361 norms and the 60 layer scalars) equal the checkpoint's; all 410 Q4_0 linears differ: 11–15% of
values equal, the codes equal in 68–71% (median 70%), in no linear the scales all equal (the record keeps no share of
equal scales), no block sign-flipped, the largest |difference| 0.006–0.025 (median 0.02). Every GGUF scale is exact
in bf16 and every checkpoint scale in fp16, so either format could hold the other's scales, but this file holds others;
its token_embd is Q6_K against the checkpoint's bf16, 15% of values equal. So this copy is a quantisation of its own,
not the checkpoint, and readings from it do not stand for the checkpoint's (their model ids already keep them apart).
Whether Google publishes a Q4_0 GGUF that is the checkpoint is still open.

`verify_optics_qat_q4_0.log`: `THEODOLITE_GGUF=<the Q4_0> THEODOLITE_DEVICE=cuda:0 scripts/verify_optics.py`, on the
reference unpacker (it installs none): every check passed, so the 60-layer sweep, the lens's mid-layer bridge, the
neuron poles, the steer and the vault run on this file and their answers have the shape the checks expect; peak
24.15 GiB allocated (the script prints GB for 2^30 bytes; its budget 30).

Not measured: a census, an atlas build or the server's pages on a GGUF with `compiled` on the GPU (only these readings
and the sighting); a sighting from the K files; the Q4_0's sighting against the checkpoint's on the same prompt (how far
a quantisation of its own moves the readings); another card; the CPU beyond the one forward each way of the Q4_0.
Splitting a model between the GPU and the CPU (llama.cpp's `-ngl`) is not built: the Q4_0 fits the card with the lens
and the optics check's short sweep (peak 24.15 GiB); the K files, run without the lens and about 1 to 1.2 GiB larger on
the card after loading, should fit too (inferred, not measured); a long sweep was not tried.

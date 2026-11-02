# The LoRA export on the 31B: a neuron's mul and a head's scale through llama.cpp

2 November 2026, the 12-core home machine, CPU only for both engines: `gemma-4-31B-it-QAT-Q4_0.gguf` (lmstudio-community),
llama.cpp cd26896 (`llama-server -ngl 0 -t 10`, adapters loaded at start and scaled per request), the instrument as
`GGUFRuntime` on the same file through int4mm (`gguf:gemma-4-31B-it-QAT-Q4_0.gguf#2ac81237f76238b4@cpu/bfloat16+int4mm`).
Every adapter was written by `scripts/export_lora.py --gguf` from that file's own columns. All pushes are at layer 45
(page 46). What is compared (`run.py`): at the first word-piece written after each of break_rel's six census questions,
each engine's change in log-probability, pushed minus unpushed, over 1,200 pieces (the instrument's 200 likeliest
unpushed and 50 likeliest pushed, per question). The bar is the control vector's (`../control_vector_L45`): r at least
0.8, and the median of llama.cpp's change over the instrument's between 0.67 and 1.5 where the instrument's moved by
more than 0.5.

**llama.cpp's own floor** (`floor.py`). The same request twice gives the same log-probabilities to the last digit. But
any adapter switched on, at any scale (0.001 as at 0.1), moves the 200 likeliest pieces by 0.25 to 0.65 in a
log-probability, per question: on the CPU llama.cpp quantises the activations to 8 bits before each product, so
any change, however small, shifts their rounding in every layer after it. A push that moves the first piece by less
than that cannot be told from it here. The unpushed engines differ by up to 0.70 over the 50 likeliest.

| set | push | r (start mark pushed / spared) | llama.cpp ÷ instrument, median where moved | moved > 0.5 | largest instrument change |
|---|---|---|---|---|---|
| active | neuron 19903 × 0 | 0.879 / 0.829 | 0.91 / 0.90 | 123 | 2.07 |
| active | neuron 19903 × 3 | 0.956 / 0.956 | 0.99 / 0.96 | 203 | 4.22 |
| active | the eight most active × 0 (rank 8) | 0.898 / 0.878 | 0.92 / 0.94 | 189 | 2.63 |
| active | the eight most active × −1 | 0.979 / 0.971 | 0.94 / 1.01 | 483 | 6.18 |
| strong | heads 0–7 × 0 (rank 2,048) | 0.941 / 0.939 | 0.96 / 0.96 | 144 | 3.86 |
| strong | head 0 × 5 | 0.940 / 0.937 | 1.15 / 1.09 | 131 | 3.81 |
| census | head 0 × 0 | 0.621 / 0.522 | 0.94 / 1.08 | 32 | 1.06 |
| census | head 7 × 3 | 0.04 / 0.11 | — | 1 | 0.56 |
| census, strong | neuron 13641 × 0, 5, 30; neuron 21200 × 5, −20 | −0.15 to 0.23 | — | 0 or 1 | 0.31 to 0.50 |

So a neuron's mul and a head's scale carry over through the LoRA where the push moves the first piece well past
llama.cpp's floor: every push that moves it by 2 or more passes the bar, with the start mark pushed (as llama.cpp
pushes it) or spared. Head 0 × 0, which moves it by 1.06 at the most, just past the floor, does not (r 0.62, though
its median ratio is 0.94).
Layer 45's two census leads do not move the first piece at all here: the instrument's own change stays under 0.5 at
× 5 and at × 30 alike, in steps of 1/16 (the bfloat16 logits' rounding), so they are hardly active on these prompts,
and there is nothing to carry over. Their census effect is in the answers, which this first-piece comparison does not
read. The most active neurons were picked by the instrument from these same prompts (`active/pushes.json`). Every
answer's first piece was the same in both engines, pushed and not (6 of 6 for every push).

Not measured: a GPU backend, the text written under a push (the census's labels were read from add pushes, not
these), a vault edit, a silenced copy of the 31B, and a quantised file other than this Q4_0. The head adapters'
files (14 MB and 107 MB) are not kept here; `run.py export` writes them again.

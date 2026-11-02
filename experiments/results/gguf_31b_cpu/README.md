# Gemma-4-31B from a UD-Q4_K_XL GGUF on the CPU, through the merged GGUF code

2 November 2026, the 12-core home machine (an AMD Ryzen 9 7900X, 61 GB), `run.py`, one run each, the J++ lens not
loaded, the file in the page cache, no other 31B beside it (a game and a browser were running).
`gemma-4-31B-it-UD-Q4_K_XL.gguf` (Unsloth, in `ai-models/`: 280 Q4_K, 70 Q6_K, 37 Q5_K and 24 IQ4_XS linears, fingerprint
`adfe6f42c3758a08`).

`ud_int4mm.json`, `THEODOLITE_GGUF_LINEARS=int4mm` with `THEODOLITE_GGUF_RESIDENT_GB=16`: the 280 Q4_K through the
kernel; 24 IQ4_XS, 69 Q6_K and 33 Q5_K resident (16.0 GB of the 16), one Q6_K and three Q5_K packed. Loaded in 68 s,
40.5 GiB held after loading. " Paris" first after "The capital of France is" (log-probability −0.018); a chat prefill of
23 word-pieces 2.6 s; "Name three rivers in Europe, one line each." answered "Danube / Rhine / Volga" and its end mark,
1.0 s a word-piece with the prefill. The kernel turned off in place (the model id without `+int4mm`): the same five
likeliest pieces in the same order, the KL from the reference at the last position under 10^-5, the largest |Δ
log-prob| among them 0.125.

`ud_packed.json`, every quantised layer packed: loaded in 15 s (5.0 GiB held, the rest read through the memory map;
21.9 GiB at the most during the forward); one forward of the six pieces of "The capital of France is" took 38.8 s on
12 threads, " Paris" first at −0.018. The 116 and 185 s measured earlier the same day were taken beside other
sessions' work, one of them reading the file from disk.

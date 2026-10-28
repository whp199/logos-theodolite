#!/usr/bin/env python3
"""Build the atlas's indexes (backend/atlas_build.py) on the GPU, into the atlas's index folder.

    .venv/bin/python scripts/atlas_build.py lens                  # every neuron's words, both faces, both readings
    .venv/bin/python scripts/atlas_build.py stats --text corpus.txt --tokens 200000
    .venv/bin/python scripts/atlas_build.py stats --hf HuggingFaceFW/fineweb-edu:sample-10BT \
        --hf HuggingFaceH4/ultrachat_200k::train_sft --tokens 2000000   # the reference's mix, a document of each in turn
    .venv/bin/python scripts/atlas_build.py wiring                # every neuron's 24 strongest weight links each way
    .venv/bin/python scripts/atlas_build.py layout                # each layer's sheet, neighbours writing alike
    .venv/bin/python scripts/atlas_build.py steer --layers 40-59  # the steer lens, a range of layers at a time
    .venv/bin/python scripts/atlas_build.py passages --hf HuggingFaceFW/fineweb-edu:sample-10BT \
        --hf HuggingFaceH4/ultrachat_200k::train_sft --tokens 2000000 --k 16 --sample 6   # each face's passages
    .venv/bin/python scripts/atlas_build.py all                   # lens, stats, wiring, layout (not the steer lens, nor passages)

The folder is `$THEODOLITE_ATLAS_DIR/index`, else `atlas_data/index` in the checkout (git ignores it), or `--out`. Each
file carries its provenance (model, lens, date, what it was read over); the server reads them as they appear, and the
atlas page says which are built (GET /api/atlas/arch). Run one GPU process at a time: this loads the model, as the
server and the lab daemon do.

What each costs on Gemma-4-31B (60 layers x 21,504 neurons, d_model 5,376, vocabulary 262,144), measured in a pilot on
one RTX 5090 (27 October 2026; the model and the J++ lens take 21.2 GiB of the card, and load in about 16 s):
- lens: about 2.6 s a layer, so under 3 minutes for all 60; 1.5 GiB above the model at its peak. About 250 MB. With
  --layers it adds those layers to the file there (built for this model and lens), and an unread layer shows no words.
- stats: one forward per window of the corpus (--seq 512), about 1.75 ms a word-piece: some 29 minutes a million
  word-pieces; 3.4 GiB above the model. It also keeps the feed-forward norms' usual scales, which the wiring needs
  (`all` builds stats before wiring).
- wiring: for every pair of layers i < j (1,770 pairs), two [21,504 x 5,376] x [5,376 x 21,504] products and a top 24
  each way, weighed across layers by Gemma's layer scalars and the norms' scales (from the stats index, else read over
  the study's 36 texts first): 4 minutes in all, 4.3 GiB above the model. About 250 MB compressed.
- layout: a principal-component projection and (with umap-learn installed) a UMAP of 21,504 points per layer, on the
  CPU; about a minute a layer with UMAP.
- steer: for every neuron, both signs and four contexts, a forward of every layer after its own: about 2 x 4 x 21,504
  rows of ~25 pieces through 30 layers on average, per layer: on the order of 10^17 FLOP a layer, so hours for all 60.
  Each layer is written as its own file under steer_lens.d/ and a stopped run resumes; steer_lens.npz is merged when
  all 60 are there.
- passages: the same forward per window as stats over the same corpus (give it the same --hf, --tokens and --seq), each
  neuron's 16 strongest windows a face kept on the card with the gate and up values at each (about 0.8 GiB above the
  model with --sample 6), so it should cost about what stats costs, some 29 minutes a million word-pieces (measured
  for stats; passages not yet run on the model, inferred). --sample S also keeps S events a face drawn at random above
  half its extreme, which the stats index holds (without it no samples are drawn, and the meta says so). A part per
  --chunk windows (1,024: about 0.8 GB each on the 31B) is written to passages.d/parts/ and a stopped run resumes;
  the parts are merged into passages.d/ (about 0.7 GB) and removed (--keep-parts keeps them). The windows it keeps are
  the corpus's own text: the folder stays in the atlas folder, which git ignores; never commit it.
`--stand-in` runs any of these on the CPU stand-in (scripts/atlas_stand_in.py) instead, for a dry run, into the stand-in's
own folder (`<atlas folder>/stand-in/index`, never among the model's; `atlas_stand_in.py --serve` reads it).
THEODOLITE_GGUF=<file.gguf> builds them on a GGUF of Gemma 4, and THEODOLITE_DEVICE=cpu on the CPU
(backend/runtime_factory.py, ARCHITECTURE §5); each file records the model id, a GGUF's naming its file and fingerprint,
and one off the GPU or in float32 its device and dtype; the server reads an index only for the model id it was built on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, List

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def parse_layers(spec: str, L: int) -> List[int]:
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b) + 1) if b else [int(a)])
    bad = [l for l in out if not 0 <= l < L]
    if bad:
        raise SystemExit(f"Layers count from 0 to {L - 1}; {bad} do not exist.")
    return out


def corpus_texts(a: argparse.Namespace, rt: Any) -> tuple:
    """(texts, name): the texts the extremes are read over, each a document."""
    texts: List[str] = []
    names: List[str] = []
    for f in a.text or []:
        raw = Path(f).read_text(encoding="utf-8")
        texts += [d for d in raw.split("\n\n") if d.strip()]
        names.append(Path(f).name)
    for f in a.jsonl or []:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    texts.append(str(json.loads(line)["text"]))
        names.append(Path(f).name)
    if a.hf:
        from datasets import load_dataset  # optional: only for --hf

        bos = rt.tokenizer.bos_token or ""

        def docs(spec: str):
            """name[:config[:split]] streamed and shuffled (split "train" unless named): chats through the chat
            template, thinking off, without their start mark (each window has its own); anything else its text."""
            name, cfg, split = (spec.split(":") + ["", ""])[:3]
            for d in load_dataset(name, cfg or None, split=split or "train", streaming=True).shuffle(seed=0):
                if "messages" in d:
                    chat = rt.tokenizer.apply_chat_template(d["messages"], tokenize=False, enable_thinking=False)
                    yield chat[len(bos):] if bos and chat.startswith(bos) else chat
                else:
                    yield str(d["text"])

        # one document from each source in turn, as the reference mixes web text and chats (sweep/neuron_stats.py)
        streams = [docs(s) for s in a.hf]
        n_chars = 0
        while streams and n_chars <= a.tokens * 6:
            for s in list(streams):
                try:
                    texts.append(next(s))
                except StopIteration:
                    streams.remove(s)
                    continue
                n_chars += len(texts[-1])
        names += a.hf
    if not texts:
        from backend.causal_check import STUDY_TEXTS

        for t in STUDY_TEXTS:
            texts.append(rt.compose_prompt(t[1], chat=True, assistant_prefix=t[2]) if t[0] == "chat" else t[1])
        names.append("the study's 36 texts (backend/causal_check.STUDY_TEXTS)")
    return texts, " + ".join(names)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("what", choices=["lens", "stats", "wiring", "steer", "layout", "passages", "all"])
    ap.add_argument("--out", help="the index folder (default: the atlas's)")
    ap.add_argument("--stand-in", action="store_true", help="run on the CPU stand-in, not the model")
    ap.add_argument("--layers", default="", help='for lens and steer: "40-59" or "3,7,11"')
    ap.add_argument("--text", action="append", help="stats, passages: a plain-text corpus file (documents split on blank lines)")
    ap.add_argument("--jsonl", action="append", help="stats, passages: a JSONL corpus file with a text field per line")
    ap.add_argument("--hf", action="append", help="stats, passages: a dataset name[:config[:split]], streamed (needs the datasets "
                    "package); several are read a document from each in turn")
    ap.add_argument("--tokens", type=int, default=200_000, help="stats, passages: word-pieces to read")
    ap.add_argument("--seq", type=int, default=512, help="stats, passages: window length")
    ap.add_argument("--chunk", type=int, default=0, help="lens: columns per product; wiring: columns per product; passages: "
                    "windows per part (1024)")
    ap.add_argument("--k", type=int, default=16, help="passages: the strongest windows kept a face, one event each")
    ap.add_argument("--sample", type=int, default=6, help="passages: events a face drawn at random above half its extreme "
                    "(needs the stats index; 0 for none)")
    ap.add_argument("--seed", type=int, default=0, help="passages: the seed of the random draw")
    ap.add_argument("--keep-parts", action="store_true", help="passages: keep the parts after merging them")
    ap.add_argument("--batch", type=int, default=64, help="steer: neurons per forward")
    ap.add_argument("--layout-method", default="auto", choices=["auto", "umap", "pca"])
    ap.add_argument("--unpacker", default="reference", choices=["reference", "lut", "compiled", "resident"],
                    help="how the 4-bit weights are unpacked (backend/w4a16_fast.py): each gives the reference's bits "
                    "(tested at load on a GPU); lut, compiled and resident are faster, resident needing ~60 GB more")
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # each step's progress as it comes, into a log file too

    if a.stand_in:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["HF_HUB_OFFLINE"] = "1"
    import torch

    from backend import atlas_build as ab
    from backend import atlas_ops as ao
    from backend.atlas_index import AtlasIndex, index_dir
    from backend.causal_check import language_model

    torch.set_grad_enabled(False)
    if a.stand_in:
        from atlas_stand_in import stand_in, use_stand_in_atlas

        use_stand_in_atlas()
        rt, _vault = stand_in()
    else:
        from backend.runtime_factory import make_runtime  # THEODOLITE_GGUF / THEODOLITE_DEVICE, else the W4A16 checkpoint

        rt = make_runtime(load_jlens=False).load()
    from backend import w4a16_fast

    try:
        w4a16_fast.install(rt, a.unpacker, log=print)
    except w4a16_fast.Refused as exc:
        raise SystemExit(str(exc))
    out = Path(a.out) if a.out else index_dir()
    out.mkdir(parents=True, exist_ok=True)
    lm = language_model(rt.hf_model)
    L = len(lm.layers)
    layers = parse_layers(a.layers, L) if a.layers else None
    steps = ["lens", "stats", "wiring", "layout"] if a.what == "all" else [a.what]
    def scales() -> dict:
        """The push's unit (p50) and the feed-forward norms' scales, from the stats index when built, else read over the
        study's 36 texts."""
        stats = AtlasIndex(out).get("act_scale")
        if stats is not None:
            return {"p50": [float(x) for x in stats["resid_q"][:, 1]], "mlp_rms": [float(x) for x in stats["mlp_rms_q"][:, 1]],
                    "mid_rms": [float(x) for x in stats["mid_rms_q"][:, 1]], "source": f"the stats index ({stats['meta'].get('corpus')})"}
        from backend.causal_check import STUDY_TEXTS

        ids = [rt.tokenize_prompt(rt.compose_prompt(t[1], chat=True, assistant_prefix=t[2]) if t[0] == "chat" else t[1])
               for t in STUDY_TEXTS]
        ex = ao.activation_extremes(lm, ids)
        return {"p50": ao.stream_p50(ex), **ao.norm_scales(ex), "source": "the study's 36 texts (no stats index yet)"}

    for step in steps:
        if step == "lens":
            print("->", ab.build_lens(rt, out, chunk=a.chunk or 512, layers=layers))
        elif step == "stats":
            texts, name = corpus_texts(a, rt)
            wins = ab.windows(rt, texts, a.seq, a.tokens)
            print("->", ab.build_stats(rt, out, wins, name))
        elif step == "wiring":
            sc = scales()
            print(f"wiring: links weighed across layers with the norms' scales from {sc['source']}")
            f = ao.link_scale(ao.Weights(lm, rt).scalars, sc["mlp_rms"], sc["mid_rms"])
            print("->", ab.build_wiring(rt, out, f, scale_source=sc["source"], chunk=a.chunk or 4096))
        elif step == "layout":
            print("->", ab.build_layout(rt, out, method=a.layout_method))
        elif step == "passages":
            texts, name = corpus_texts(a, rt)
            wins = ab.windows(rt, texts, a.seq, a.tokens)
            stats = AtlasIndex(out).get("act_scale")
            if a.sample and stats is None:
                print("passages: no stats index here, so no events are drawn at random (their threshold is half of each "
                      "face's extreme): build stats first to draw them")
            n_read = sum(max(0, len(wn) - 1) for wn in wins)
            if stats is not None and (stats["meta"].get("corpus"), int(stats["tokens"])) != (name, n_read):
                what = "each passage's share of its face's extreme" + (" and the samples' threshold are" if a.sample else " is")
                print(f"passages: the stats index was read over {stats['meta'].get('corpus')} ({int(stats['tokens']):,} word-pieces), "
                      f"these windows over {name} ({n_read:,}): {what} taken against the stats index; give both the same corpus")
            ext = (stats["pos"], stats["neg"]) if stats is not None and a.sample else None
            src = f"the stats index ({stats['meta'].get('corpus')}, {int(stats['tokens'])} word-pieces)" if ext is not None else None
            print("->", ab.build_passages(rt, out, wins, name, k=a.k, sample=a.sample, extremes=ext, extremes_source=src,
                                          seed=a.seed, chunk=a.chunk or 1024, keep_parts=a.keep_parts))
        elif step == "steer":
            sc = scales()
            print(f"steer: the push's unit is the stream's median length from {sc['source']}")
            print("->", ab.build_steer_lens(rt, out, sc["p50"], layers=layers, batch=a.batch))
    return 0


if __name__ == "__main__":
    sys.exit(main())

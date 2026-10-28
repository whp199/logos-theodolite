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
    .venv/bin/python scripts/atlas_build.py wiring_check --layers 40,50 --writers random:512 --dose 0.25   # links measured
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
  the study's 36 texts first): 4 minutes in all, 4.3 GiB above the model. About 250 MB compressed. It also keeps each
  link's g and u (float16: 248 MB more uncompressed, which compressed to about three quarters on the stand-in, so some
  180 MB more in the file; about 0.6 GiB more on the card and the same time, inferred), from which the sign in use is
  read at each reader's operating point in the passages index (ARCHITECTURE §4.7): an older build serves the sign from
  rest only, so rebuild it once the passages are built. With both there, a wiring or passages step ends by counting
  how often the sign in use differs from the sign from rest.
- layout: a principal-component projection and (with umap-learn installed) a UMAP of 21,504 points per layer, on the
  CPU; about a minute a layer with UMAP.
- steer: for every neuron, both signs and four contexts, a forward of every layer after its own: about 2 x 4 x 21,504
  rows of ~25 pieces through 30 layers on average, per layer: on the order of 10^17 FLOP a layer, so hours for all 60.
  Each layer is written as its own file under steer_lens.d/ and a stopped run resumes; steer_lens.npz is merged when
  all 60 are there.
- passages: the same forward per window as stats over the same corpus (give it the same --hf, --tokens and --seq), each
  neuron's 16 strongest windows a face kept on the card with the gate and up values at each (about 0.8 GiB more than
  stats needs with --sample 6, as both keep each layer's gate and up outputs to a window's end), so it should cost
  about what stats costs, some 29 minutes a million word-pieces (measured for stats; passages not yet run on the
  model, inferred). --sample S also keeps S events a face drawn at random above half its extreme, which the stats
  index holds (without it no samples are drawn, and the meta says so). A part per --chunk windows (1,024: about
  0.8 GB each on the 31B) is written to passages.d/parts/ and a stopped run resumes; the parts are merged into
  passages.d/ (about 0.7 GB) and removed (--keep-parts keeps them). The windows it keeps are the corpus's own text:
  the folder stays in the atlas folder, which git ignores; never commit it.
- wiring_check: the wiring measured, prefill only. Per layer (--layers, else all), --writers (random:512, drawn from
  --seed and the layer) are each pushed by +d and -d along their write direction at every word-piece of the census's
  four questions but the start mark, d = --dose (0.25) times the writer's own write ceiling, the most it can write by
  itself: a quarter of what it could write firing alone, the same share for every writer, never a push the neuron
  could not make (the steer lens's 0.8 of the stream is 1 to 10 times the ceiling at most layers), and meant to stay
  near the first order the sign in use is read at (how near, untested on the 31B: the share of readers +d and -d move
  the same way says). Its indexed readers and as many it is not linked to (the null)
  are read at the last word-piece and over the prompt, in prefills of --batch rows (64), each with its own unpushed
  row. Needs the wiring index (the passages index too, for the sign in use). Per layer it logs, and keeps in
  wiring_check.npz for /arch, how often the sign from rest holds (under +d, and on the writer's gate face), the sign
  in use, Spearman's rho of the change against |P| f^2, the same for the null, and how often +d and -d move a reader
  the same way. Cost (inferred, prefill only): at 512 writers and four questions of some 20 word-pieces, 68 prefills of
  up to 63 rows a layer, about 1.3 minutes a layer on the 5090 from the census's 0.71 s step and the stats pass's rate
  (the log gives each layer's own figure and the seconds it took), some 1.3 hours for the 59 layers a later one
  reads. A layer is written as it is done, and a stopped run resumes.
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
    ap.add_argument("what", choices=["lens", "stats", "wiring", "steer", "layout", "passages", "wiring_check", "all"])
    ap.add_argument("--out", help="the index folder (default: the atlas's)")
    ap.add_argument("--stand-in", action="store_true", help="run on the CPU stand-in, not the model")
    ap.add_argument("--layers", default="", help='for lens, steer and wiring_check: "40-59" or "3,7,11"')
    ap.add_argument("--writers", default="random:512", help='wiring_check: the writers a layer, "random:K", "0-99" or "1,5,9"')
    ap.add_argument("--dose", type=float, default=0.25, help="wiring_check: the push, a share of each writer's own write ceiling")
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
    ap.add_argument("--seed", type=int, default=0, help="passages, wiring_check: the seed of the random draw")
    ap.add_argument("--keep-parts", action="store_true", help="passages: keep the parts after merging them")
    ap.add_argument("--batch", type=int, default=64, help="steer: neurons per forward; wiring_check: rows per prefill")
    ap.add_argument("--layout-method", default="auto", choices=["auto", "umap", "pca"])
    ap.add_argument("--replace", action="store_true", help="replace an index built for another model or other weights "
                    "(by default refused: build into another folder, --out or $THEODOLITE_ATLAS_DIR, to keep both)")
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
    from backend.runtime_factory import weights_card

    # the stats index read for the push's unit and the norms' scales only when built on this model and its weights
    here = AtlasIndex(out, model=getattr(rt, "model_id", None), shape=(L, ao.ffn_width(lm, 0)), weights=weights_card(rt))
    steps = ["lens", "stats", "wiring", "layout"] if a.what == "all" else [a.what]
    # one folder holds one model's indexes: a build for another model id (the checkpoint on the CPU, a GGUF) or other
    # weights would write over the files the server reads for the first, so it is refused unless asked for
    for step in steps:
        name = {"stats": "act_scale", "steer": "steer_lens"}.get(step, step)
        if here.get(name) is None and (here.refused(name) or "").startswith(("built for ", "built from other weights")):
            if not a.replace:
                raise SystemExit(f"{step}: {here.path(name)} was {here.refused(name).split(': rebuild it')[0]}, and this build "
                                 "would replace it. Build into another folder (--out, or $THEODOLITE_ATLAS_DIR) to keep both, "
                                 "or give --replace.")
            print(f"{step}: replacing {here.path(name)}, {here.refused(name).split(': rebuild it')[0]} (--replace)", flush=True)
    def scales() -> dict:
        """The push's unit (p50) and the feed-forward norms' scales, from the stats index when built, else read over the
        study's 36 texts."""
        stats = here.get("act_scale")
        if stats is not None:
            return {"p50": [float(x) for x in stats["resid_q"][:, 1]], "mlp_rms": [float(x) for x in stats["mlp_rms_q"][:, 1]],
                    "mid_rms": [float(x) for x in stats["mid_rms_q"][:, 1]], "source": f"the stats index ({stats['meta'].get('corpus')})"}
        from backend.causal_check import STUDY_TEXTS

        ids = [rt.tokenize_prompt(rt.compose_prompt(t[1], chat=True, assistant_prefix=t[2]) if t[0] == "chat" else t[1])
               for t in STUDY_TEXTS]
        ex = ao.activation_extremes(lm, ids)
        refused = here.refused("act_scale")  # there, but built for another model id, shape or weights
        if refused:
            print(f"the stats index is not read: {refused}", flush=True)
        return {"p50": ao.stream_p50(ex), **ao.norm_scales(ex),
                "source": f"the study's 36 texts ({'the stats index refused: ' + refused if refused else 'no stats index yet'})"}

    def disagreement() -> None:
        """How often the sign in use differs from the sign from rest over the whole index (AtlasIndex.sign_disagreement),
        once the wiring index with its links' g and u and the passages index are both there."""
        d = AtlasIndex(out).sign_disagreement()
        if d is None:
            return
        f = d["faces"]
        say = lambda s: f"{f[s]['differ']:,} of {f[s]['read']:,} read on the reader's {s} face ({100 * f[s]['differ'] / max(1, f[s]['read']):.1f}%)"  # noqa: E731
        print(f"links in use: at each reader's operating point the sign differs from the sign from rest on {say('+')} and "
              f"{say('-')}; {d['differ_on_both']:,} of {d['links']:,} downstream links differ on both")

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
            disagreement()
        elif step == "layout":
            print("->", ab.build_layout(rt, out, method=a.layout_method))
        elif step == "passages":
            texts, name = corpus_texts(a, rt)
            wins = ab.windows(rt, texts, a.seq, a.tokens)
            stats = here.get("act_scale")
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
            disagreement()
        elif step == "steer":
            sc = scales()
            print(f"steer: the push's unit is the stream's median length from {sc['source']}")
            print("->", ab.build_steer_lens(rt, out, sc["p50"], layers=layers, batch=a.batch))
        elif step == "wiring_check":
            sc = scales()
            print(f"wiring_check: each link's strength |P| f^2 weighed with the norms' scales from {sc['source']}")
            f = ao.link_scale(ao.Weights(lm, rt).scalars, sc["mlp_rms"], sc["mid_rms"])
            try:
                print("->", ab.build_wiring_check(rt, out, f, layers=layers, writers=a.writers, dose=a.dose, seed=a.seed,
                                                  batch=a.batch, scale_source=sc["source"]))
            except ValueError as exc:
                raise SystemExit(str(exc))
            disagreement()
    return 0


def run() -> int:
    """`main()`, with the CPU's memory refused in plain words (backend/cpu_memory.py: a steer lens batch the estimate
    says MemAvailable cannot hold, or an allocation the CPU's allocator refused) as a one-line exit, not a traceback."""
    from backend.cpu_memory import CPUMemoryRefused, is_cpu_oom, plain

    try:
        return main()
    except CPUMemoryRefused as exc:
        raise SystemExit(str(exc))
    except RuntimeError as exc:
        if not is_cpu_oom(exc):
            raise
        raise SystemExit(f"{plain(exc, 'the build')} A build written a part or a layer at a time resumes where it "
                         "stopped: lower --batch or --chunk, and run it again.")


if __name__ == "__main__":
    sys.exit(run())

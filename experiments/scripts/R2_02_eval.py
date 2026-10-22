#!/usr/bin/env python3
"""R2 02 (GPU): read the fresh prompts through J++, the naked eye, R1's lens and the Miru lens, filtered and not.

Implements experiments/results/R2_PREREG.md §4-§5 (the analysis, §6-§7, is R2_03_analyze.py). Standalone: it loads
TheodoliteRuntime itself (as experiments/lab/labd.py does) and needs neither the server nor the lab daemon. Stop the
server first: the model needs the GPU's memory.

  1. Checks, before the model is loaded: the pre-registered files (R2_common.PREREG_FILES: R2_PREREG.md,
     R2_prompts.json, R2_common.py, R2_01_check.py, R2_02_eval.py, R2_03_analyze.py, R2_01_check.json) are committed
     and unchanged in git (else --allow-uncommitted, a deviation); R2_02_eval.json and R2_02_states.pt do not exist yet
     (else --force); the prompt file validates and its sha256 is the one R2_01_check.json recorded; the token filter
     passes its spot checks and has the class counts R2_01_check.json recorded; R1's lens file and the Miru file have their pinned size, then sha256 (Miru: 3.4 GB,
     read only through safetensors.safe_open, header checked as the server checks it). Every run that gets past these
     checks is appended to experiments/results/R2_02_runs.jsonl (started / finished / failed), for the write-up.
  2. Greedy continuations by full forwards (no KV cache): 8 tokens after each chat-no-answer / chat-begun / bare
     prompt (the targets), 48-token replies to the mid-reply questions. The model's own word at a step is the argmax
     of the final norm + W_U + softcap of resid_post[59] (unembed_gemma4), as the instrument names it.
  3. Read positions: each prompt's last token; for mid-reply, the seeded draws of R2_common.draw_mid_positions
     ("any": 6 positions, the decision condition; "content_next": 3 positions, reported only). The content target is
     the first CONTENT token of the model's own next 8, the search stopping at a stop id or any special token.
  4. One batched forward per group of prompts captures resid_post at code layers 40-59 at the read positions.
  5. Readouts, through the instrument's own functions (jpp_readout_logits / unembed_gemma4, bf16): J++ 40-58, naked
     eye 40-59, R1 50/55/58, Miru 40-58 (one layer on the GPU at a time). For each: top-5 ids and strings unfiltered
     and filtered (CONTENT ids only), the rank of the next token (unfiltered) and of the content target (filtered).
  6. Writes experiments/results/R2_02_eval.json (raw, per position and layer) and R2_02_states.pt (the captured
     states, bf16, gitignored) so the readouts can be recomputed on the CPU.

Run (GPU, server stopped), from the checkout these scripts are in (the venv is the main checkout's):
  /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R2_02_eval.py [--r1 PATH] [--miru PATH]
      [--batch 8] [--force] [--allow-uncommitted]
Self-test on the CPU with a tiny random stand-in for the model (real tokenizer, filter and prompts; no model):
  CUDA_VISIBLE_DEVICES="" /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R2_02_eval.py --selftest --limit 3 --out X.json
Every check before the model, on the CPU (lens hashes included; uncommitted files are only a warning here):
  CUDA_VISIBLE_DEVICES="" /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R2_02_eval.py --check-only
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import datetime
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import R2_common as C  # noqa: E402

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------------------------------------ backends
class GpuBackend:
    """The real model: TheodoliteRuntime + labkit.Lab, the runtime's J++ matrices, R1_18 and the Miru file."""

    @staticmethod
    def check_files(r1_path: Path, miru_path: Path):
        """The lens files, before any model is loaded (CPU only): size, then sha256 (pins equal to the server's);
        R1 through the server's own loader (sha256 of the very bytes loaded, provenance, layers 50/55/58); the Miru
        header as the server checks it (format, model, revision, estimator, fp16 [5376, 5376] at layers 40-58).
        Returns (lens_info, R1 matrices on the CPU)."""
        sys.path.insert(0, str(C.REPO))
        from backend import engine_runtime as ER
        from backend.theodolite_math import validate_and_cast_lens
        from safetensors import safe_open

        assert C.R1_SHA256 == ER.IT_LENS_SHA256 and C.R1_SIZE == ER.IT_LENS_SIZE, "R1 pins differ from the server's"
        assert C.MIRU_SHA256 == ER.MIRU_SHA256 and C.MIRU_SIZE == ER.MIRU_SIZE, "Miru pins differ from the server's"
        lens_info: Dict[str, Any] = {}
        for name, path, size, sha in (("r1", r1_path, C.R1_SIZE, C.R1_SHA256),
                                      ("miru", miru_path, C.MIRU_SIZE, C.MIRU_SHA256)):
            if not path.is_file():
                raise SystemExit(f"{name}: no file at {path}")
            n = path.stat().st_size
            if n != size:
                raise SystemExit(f"{name}: {path} is {n:,} bytes, not the pinned {size:,}")
            t = time.time()
            got = C.sha256_file(path)
            if got != sha:
                raise SystemExit(f"{name}: {path} has sha256 {got}, not the pinned {sha}")
            log(f"{name}: {path} {n:,} bytes, sha256 {got[:16]}... matches the pin ({time.time() - t:.1f}s)")
            lens_info[name] = {"path": str(path), "size": n, "sha256": got}
        payload = ER._load_it_lens_payload(r1_path)
        r1_cpu = {}
        for l in C.LENS_LAYERS["r1"]:
            r1_cpu[int(l)] = validate_and_cast_lens(payload["J"][l], layer_idx=int(l), target_dtype=payload["J"][l].dtype)
        del payload
        with safe_open(str(miru_path), framework="pt", device="cpu") as f:
            lens_info["miru"]["header"] = ER._check_miru_header(f, list(C.LENS_LAYERS["miru"]), miru_path.name)
        log(f"r1: layers {sorted(r1_cpu)} {next(iter(r1_cpu.values())).dtype}; miru: header ok at layers "
            f"{C.LENS_LAYERS['miru'][0]}-{C.LENS_LAYERS['miru'][-1]} ({lens_info['miru']['header'].get('model')}, "
            f"{lens_info['miru']['header'].get('estimator')})")
        return lens_info, r1_cpu

    def __init__(self, r1_path: Path, miru_path: Path):
        sys.path.insert(0, str(C.REPO))
        sys.path.insert(0, str(C.REPO / "experiments" / "lab"))
        import torch
        from backend import engine_runtime as ER
        from backend.theodolite_math import jpp_readout_logits, unembed_gemma4, validate_and_cast_lens
        from safetensors import safe_open

        self.torch = torch
        self._jpp_readout_logits = jpp_readout_logits
        self._unembed = unembed_gemma4
        self._cast = validate_and_cast_lens
        self.lens_info, r1_cpu = self.check_files(r1_path, miru_path)
        self.miru_path = miru_path
        self._safe_open = safe_open
        # the model, as labd.py loads it (no JLens: not used here; no server second lenses: loaded here instead)
        log("loading TheodoliteRuntime (W4A16 model + J++)...")
        t = time.time()
        self.rt = ER.TheodoliteRuntime(load_jlens=False, load_second_lenses=False).load()
        import labkit

        self.lab = labkit.Lab(self.rt)
        self.tok = self.rt.tokenizer
        self.dev = self.rt.device
        self.fnw = self.rt.final_norm_weight
        self.wu = self.rt.lm_head_weight
        log(f"runtime ready in {time.time() - t:.1f}s; VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GB; "
            f"W_U {tuple(self.wu.shape)} {self.wu.dtype}")
        self.r1 = {l: self._cast(J, layer_idx=l, target_dtype=torch.bfloat16, device=self.dev) for l, J in r1_cpu.items()}
        del r1_cpu
        self.lens_info["jpp"] = self.rt.lens_card()
        self.vocab_rows = int(self.wu.shape[0])

    def check_text(self, item: Dict[str, Any], text: str, ids: List[int]) -> None:
        """R2_common's chat/ids must be labkit's exactly."""
        if item["cond"] == "bare":
            ref = item["text"]
        else:
            ref = self.lab.chat(item["user"], assistant=item.get("begun", ""))
        assert ref == text, (item["pid"], ref, text)
        assert self.lab.ids(text) == ids, item["pid"]

    def states(self, batch_ids: List[List[int]], positions: List[List[int]], layers: Sequence[int]):
        """[L, P_b, D] float32 CPU per item: resid_post at `layers` and the given positions (lab.forward_resid)."""
        P = max(len(p) for p in positions)
        padded = [list(p) + [p[-1]] * (P - len(p)) for p in positions]
        out = self.lab.forward_resid(batch_ids, list(layers), padded)  # [B, L, P, D]
        return [out[b, :, : len(positions[b])] for b in range(len(batch_ids))]

    def final_logits(self, h):
        """The model's own readout of resid_post[59]: final RMSNorm, W_U, softcap (the instrument's naked eye at 59)."""
        return self._unembed(h.to(self.dev, self.torch.bfloat16), self.fnw, self.wu).float()

    def lens_matrix(self, kind: str, l: int):
        if kind == "naive":
            return None
        if kind == "jpp":
            return self.rt.jpp_lenses[l]
        if kind == "r1":
            return self.r1[l]
        if kind == "miru":
            with self._safe_open(str(self.miru_path), framework="pt", device="cpu") as f:
                raw = f.get_tensor(f"J.{l}")
            return self._cast(raw, layer_idx=l, target_dtype=self.torch.bfloat16, device=self.dev)
        raise ValueError(kind)

    def readout_logits(self, kind: str, l: int, h, J):
        h = h.to(self.dev, self.torch.bfloat16)
        if kind == "naive":
            return self._unembed(h, self.fnw, self.wu).float()
        return self._jpp_readout_logits(h, J, self.fnw, self.wu)[0].float()

    def release(self, kind: str, J) -> None:
        if kind == "miru":
            del J
            self.torch.cuda.empty_cache()

    def info(self) -> Dict[str, Any]:
        import torch
        import transformers

        return {"backend": "gpu", "device": torch.cuda.get_device_name(self.dev), "torch": torch.__version__,
                "transformers": transformers.__version__, "model": self.rt.model_id, "lenses": self.lens_info}


class FakeBackend:
    """A tiny deterministic stand-in for the model (D = 32, random embeddings and lenses) for the CPU self-test of the
    pipeline: generation, read positions, readouts, filtering, ranks and the output file. It knows nothing true."""

    D = 32

    @classmethod
    def build(cls, tok, seed: int = 0):
        import torch

        sys.path.insert(0, str(C.REPO))
        from backend.theodolite_math import jpp_readout_logits, unembed_gemma4

        self = cls.__new__(cls)
        self.torch = torch
        self.tok = tok
        self.dev = torch.device("cpu")
        self._jpp_readout_logits = jpp_readout_logits
        self._unembed = unembed_gemma4
        g = torch.Generator().manual_seed(seed)
        V, D = len(tok), cls.D
        self.E = torch.randn(V, D, generator=g)
        self.wu = torch.randn(V, D, generator=g)
        self.fnw = torch.ones(D)
        self.J: Dict[str, Dict[int, Any]] = {}
        for kind in ("jpp", "r1", "miru"):
            self.J[kind] = {l: torch.eye(D) + 0.4 * torch.randn(D, D, generator=g) for l in C.LENS_LAYERS[kind]}
        # one stuck layer, to exercise the stuck-word share: every state maps onto one fixed direction
        u = torch.randn(D, generator=g)
        self.J["miru"][58] = torch.outer(u, torch.ones(D)) * 0.1 + 1e-3 * torch.eye(D)
        self.vocab_rows = V
        self.lens_info = {"fake": True}
        return self

    def check_text(self, item, text, ids):
        pass

    def states(self, batch_ids, positions, layers):
        torch = self.torch
        out = []
        for ids, pos in zip(batch_ids, positions):
            e = self.E[torch.tensor(ids)]
            csum = torch.cumsum(e, 0) / torch.arange(1, len(ids) + 1).unsqueeze(1)
            rows = []
            for l in layers:
                rows.append(torch.stack([e[p] * (1 + l / 60) + 0.5 * csum[p] for p in pos], 0))
            out.append(torch.stack(rows, 0))
        return out

    def final_logits(self, h):
        return self._unembed(h.float(), self.fnw, self.wu).float()

    def lens_matrix(self, kind, l):
        return None if kind == "naive" else self.J[kind][l]

    def readout_logits(self, kind, l, h, J):
        if kind == "naive":
            return self._unembed(h.float(), self.fnw, self.wu).float()
        return self._jpp_readout_logits(h.float(), J, self.fnw, self.wu)[0].float()

    def release(self, kind, J):
        pass

    def info(self):
        return {"backend": "fake (CPU self-test; no model)", "torch": self.torch.__version__}


# ------------------------------------------------------------------------------------------------ the pipeline
def greedy(backend, seqs: List[List[int]], n_new: int, batch: int, label: str) -> List[List[int]]:
    """Greedy continuation of each sequence by full forwards (no KV cache). Each step reads resid_post[59] at each
    live sequence's last token, takes the argmax of the model's own readout, and appends it; a sequence stops after a
    stop id (which is kept in its list)."""
    out: List[List[int]] = [[] for _ in seqs]
    live = list(range(len(seqs)))
    for step in range(n_new):
        if not live:
            break
        nxt_live = []
        for b0 in range(0, len(live), batch):
            idx = live[b0:b0 + batch]
            ids = [seqs[i] + out[i] for i in idx]
            st = backend.states(ids, [[len(x) - 1] for x in ids], [59])
            h = backend.torch.stack([s[0, 0] for s in st], 0)
            tops = backend.final_logits(h).argmax(-1).tolist()
            for i, t in zip(idx, tops):
                out[i].append(int(t))
                if int(t) not in C.STOP_IDS:
                    nxt_live.append(i)
        live = nxt_live
        if step == 0 or (step + 1) % 8 == 0 or not live:
            log(f"  {label}: step {step + 1}/{n_new}, {len(live)} still going")
    return out


def build_positions(tok, items, conts, replies, is_content, is_special) -> List[Dict[str, Any]]:
    """The read positions and their targets (R2_PREREG.md §4.3, §5). "ct_prev" is the token just before the content
    target (the read token itself when the target is the next token)."""
    pos_recs: List[Dict[str, Any]] = []
    mid_index = 0
    for it in items:
        ids = it["ids"]
        if it["cond"] != "mid-reply":
            cont = conts[it["pid"]]
            window = [int(t) for t in cont[: C.CONT_WINDOW]]
            ct = C.first_content(window, is_content, is_special)
            pos_recs.append({
                "pid": it["pid"], "cond": it["cond"], "draws": None, "k": None, "pos": len(ids) - 1,
                "seq": ids, "next_id": int(cont[0]),
                "cont_ids": window,
                "ct": ct, "ct_prev": ([int(ids[-1])] + window)[ct[0]] if ct else None,
            })
            continue
        full = replies[it["pid"]]
        reply = []
        for t in full:
            if int(t) in C.STOP_IDS:
                break
            reply.append(int(t))
        it["reply_ids"] = reply
        it["reply_stopped"] = len(reply) < len(full)
        draws = C.draw_mid_positions(reply, is_content, mid_index)
        it["draws"] = {str(k): v for k, v in draws.items()}
        mid_index += 1
        seq = ids + reply
        for k, names in draws.items():
            window = reply[k + 1: k + 1 + C.CONT_WINDOW]
            ct = C.first_content(window, is_content, is_special)
            pos_recs.append({
                "pid": it["pid"], "cond": "mid-reply", "draws": names, "k": k, "pos": len(ids) + k,
                "seq": seq, "next_id": int(reply[k + 1]), "cont_ids": window,
                "ct": ct, "ct_prev": ([int(reply[k])] + window)[ct[0]] if ct else None,
            })
    return pos_recs


def run(backend, tok, data, mask, is_special, batch: int, limit: Optional[int], out_json: Path,
        out_states: Optional[Path], meta_extra: Dict[str, Any]) -> Dict[str, Any]:
    torch = backend.torch
    is_content = mask.numpy()
    if len(is_content) != backend.vocab_rows:
        raise SystemExit(f"mask has {len(is_content)} ids, W_U has {backend.vocab_rows} rows")
    items = C.prompt_items(data)
    if limit:
        items = [it for c in C.CONDITIONS for it in [x for x in items if x["cond"] == c][:limit]]
    checked = set()
    for it in items:
        it["text_read"] = C.prompt_text(tok, it)
        it["ids"] = C.text_ids(tok, it["text_read"])
        if it["cond"] not in checked:
            backend.check_text(it, it["text_read"], it["ids"])
            checked.add(it["cond"])
    log(f"{len(items)} prompts: " + ", ".join(f"{c} {sum(it['cond'] == c for it in items)}" for c in C.CONDITIONS))

    # targets: 8-token continuations; mid-reply: 48-token replies
    t = time.time()
    first = [it for it in items if it["cond"] != "mid-reply"]
    mids = [it for it in items if it["cond"] == "mid-reply"]
    cont = greedy(backend, [it["ids"] for it in first], C.CONT_WINDOW, batch, "continuations")
    conts = {it["pid"]: c for it, c in zip(first, cont)}
    rep = greedy(backend, [it["ids"] for it in mids], C.MID_NEW_TOKENS, batch, "mid-reply replies")
    replies = {it["pid"]: r for it, r in zip(mids, rep)}
    t_gen = time.time() - t
    log(f"generation done in {t_gen:.1f}s")

    pos_recs = build_positions(tok, items, conts, replies, is_content, is_special)
    N = len(pos_recs)
    log(f"{N} read positions: " + ", ".join(
        f"{c} {sum(p['cond'] == c for p in pos_recs)}" for c in C.CONDITIONS))

    # one forward per group of prompts: resid_post at layers 40-59 at each prompt's read positions
    t = time.time()
    L = len(C.READ_LAYERS)
    by_pid: Dict[str, List[int]] = {}
    for j, p in enumerate(pos_recs):
        by_pid.setdefault(p["pid"], []).append(j)
    pids = list(by_pid)
    H = None
    for b0 in range(0, len(pids), batch):
        grp = pids[b0:b0 + batch]
        seqs = [pos_recs[by_pid[pid][0]]["seq"] for pid in grp]
        poss = [[pos_recs[j]["pos"] for j in by_pid[pid]] for pid in grp]
        st = backend.states(seqs, poss, C.READ_LAYERS)
        if H is None:
            H = torch.empty(N, L, st[0].shape[-1], dtype=torch.bfloat16)
        for pid, s in zip(grp, st):
            for q, j in enumerate(by_pid[pid]):
                H[j] = s[:, q].to(torch.bfloat16)
        log(f"  reading forward {min(b0 + batch, len(pids))}/{len(pids)} prompts")
    t_read = time.time() - t

    # readouts: per layer, per lens (Miru one layer at a time), in chunks of positions
    t = time.time()
    dev = backend.dev
    mask_dev = mask.to(dev)
    next_ids = torch.tensor([p["next_id"] for p in pos_recs], device=dev)
    ct_ids = torch.tensor([p["ct"][1] if p["ct"] else -1 for p in pos_recs], device=dev)
    strs: Dict[int, str] = {}

    def s_of(i: int) -> str:
        if i not in strs:
            strs[i] = C.token_text(tok, i)
        return strs[i]

    R: List[Dict[str, Dict[str, Any]]] = [dict() for _ in range(N)]
    CH = 64
    for li, l in enumerate(C.READ_LAYERS):
        kinds = [k for k in ("jpp", "naive", "r1", "miru") if l in C.LENS_LAYERS[k]]
        for kind in kinds:
            J = backend.lens_matrix(kind, l)
            for s0 in range(0, N, CH):
                h = H[s0:s0 + CH, li]
                lg = backend.readout_logits(kind, l, h, J)
                u = torch.topk(lg, C.TOPK, dim=-1).indices
                lgf = lg.masked_fill(~mask_dev, float("-inf"))
                f = torch.topk(lgf, C.TOPK, dim=-1).indices
                nx = next_ids[s0:s0 + CH]
                ur = (lg > lg.gather(1, nx[:, None])).sum(1) + 1
                ct = ct_ids[s0:s0 + CH]
                ctc = ct.clamp(min=0)
                fr = (lgf > lgf.gather(1, ctc[:, None])).sum(1) + 1
                u_l, f_l, ur_l, fr_l = u.tolist(), f.tolist(), ur.tolist(), fr.tolist()
                for q in range(len(u_l)):
                    j = s0 + q
                    R[j].setdefault(kind, {})[str(l)] = {
                        "u": u_l[q], "us": [s_of(i) for i in u_l[q]],
                        "f": f_l[q], "fs": [s_of(i) for i in f_l[q]],
                        "ur": int(ur_l[q]),
                        "fr": int(fr_l[q]) if int(ct[q]) >= 0 else None,
                        "u0c": bool(is_content[u_l[q][0]]),
                    }
                del lg, lgf
            backend.release(kind, J)
        log(f"  readouts at layer {l} ({'/'.join(kinds)})")
    t_lens = time.time() - t

    # records
    positions = []
    agree = {}
    for j, p in enumerate(pos_recs):
        m = R[j]["naive"]["59"]
        ct = p["ct"]
        rec = {
            "i": j, "pid": p["pid"], "cond": p["cond"], "draws": p["draws"], "k": p["k"], "pos": p["pos"],
            "read_tok": s_of(p["seq"][p["pos"]]),
            "next_id": p["next_id"], "next_str": s_of(p["next_id"]), "next_content": bool(is_content[p["next_id"]]),
            "cont_ids": p["cont_ids"], "cont_strs": [s_of(i) for i in p["cont_ids"]],
            "ct_id": ct[1] if ct else None, "ct_str": s_of(ct[1]) if ct else None, "ct_offset": ct[0] if ct else None,
            "ct_prev_str": s_of(p["ct_prev"]) if ct else None,
            "ct_collapse": bool(ct) and C.word_key(s_of(ct[1])) in C.COLLAPSE_TARGETS,
            "ct_formatlike": bool(ct) and C.formatlike_target(s_of(ct[1]), s_of(p["ct_prev"])),
            "model_top1_id": m["u"][0], "model_top1_str": m["us"][0],
            "model_agrees": m["u"][0] == p["next_id"],
            "r": R[j],
        }
        positions.append(rec)
        agree.setdefault(p["cond"], []).append(rec["model_agrees"])
    prompts_out = []
    for it in items:
        d = {k: it[k] for k in ("pid", "cond", "user", "begun", "text", "kind", "format_likely", "about_model", "topic")
             if k in it}
        d["n_ids"] = len(it["ids"])
        d["read_tok"] = s_of(it["ids"][-1])
        if it["cond"] == "mid-reply":
            d["reply_ids"] = it["reply_ids"]
            d["reply_text"] = tok.decode(it["reply_ids"], skip_special_tokens=False)
            d["reply_stopped"] = it["reply_stopped"]
            d["draws"] = it["draws"]
        else:
            d["cont_ids"] = conts[it["pid"]]
            d["cont_text"] = tok.decode(conts[it["pid"]], skip_special_tokens=False)
        prompts_out.append(d)
    summary = {
        "n_positions": {c: sum(p["cond"] == c for p in positions) for c in C.CONDITIONS},
        "n_mid_any": sum(1 for p in positions if p["draws"] and "any" in p["draws"]),
        "n_mid_content_next": sum(1 for p in positions if p["draws"] and "content_next" in p["draws"]),
        "n_with_content_target": {c: sum(p["cond"] == c and p["ct_id"] is not None for p in positions)
                                  for c in C.CONDITIONS},
        "n_collapse_target": {c: sum(p["cond"] == c and p["ct_collapse"] for p in positions) for c in C.CONDITIONS},
        "n_formatlike_target": {c: sum(p["cond"] == c and p["ct_formatlike"] for p in positions) for c in C.CONDITIONS},
        "model_agrees_with_next": {c: round(sum(v) / len(v), 4) for c, v in agree.items()},
        "next_is_noncontent": {c: round(sum((not p["next_content"]) for p in positions if p["cond"] == c)
                                        / max(1, sum(p["cond"] == c for p in positions)), 4) for c in C.CONDITIONS},
    }
    log("summary:", json.dumps(summary))
    out = {
        "schema": "R2_02/1",
        "meta": dict(meta_extra, **{
            "written_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "host": platform.node(),
            "argv": sys.argv,
            "backend": backend.info(),
            "constants": {"READ_LAYERS": list(C.READ_LAYERS), "LENS_LAYERS": {k: list(v) for k, v in C.LENS_LAYERS.items()},
                          "BANDS": [[n, list(ls)] for n, ls in C.BANDS], "CONT_WINDOW": C.CONT_WINDOW,
                          "MID_NEW_TOKENS": C.MID_NEW_TOKENS, "MID_DRAW_SIZES": C.MID_DRAW_SIZES,
                          "DRAW_SEED": C.DRAW_SEED, "STOP_IDS": list(C.STOP_IDS), "TOPK": C.TOPK,
                          "COLLAPSE_TARGETS": list(C.COLLAPSE_TARGETS), "batch": batch, "limit": limit},
            "timing_s": {"generation": round(t_gen, 1), "reading": round(t_read, 1), "readouts": round(t_lens, 1),
                         "total": round(time.time() - T0, 1)},
            "summary": summary,
        }),
        "prompts": prompts_out,
        "positions": positions,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_json.with_suffix(out_json.suffix + ".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":"), default=str), encoding="utf-8")
    os.replace(tmp, out_json)
    log(f"wrote {out_json} ({out_json.stat().st_size / 1e6:.1f} MB)")
    if out_states is not None:
        torch.save({"H": H, "layers": list(C.READ_LAYERS),
                    "positions": [[p["pid"], p["pos"], p["k"]] for p in positions]}, out_states)
        log(f"wrote {out_states} ({out_states.stat().st_size / 1e6:.1f} MB)")
    return out


def git_state(paths: Sequence[Path]) -> Dict[str, Any]:
    """The checkout's HEAD and any of `paths` that are untracked or differ from HEAD (git status --porcelain)."""
    import subprocess

    rel = [str(Path(p).resolve().relative_to(C.REPO)) for p in paths]
    try:
        head = subprocess.run(["git", "-C", str(C.REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        st = subprocess.run(["git", "-C", str(C.REPO), "status", "--porcelain", "--untracked-files=all", "--"] + rel,
                            capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        return {"head": None, "dirty": rel, "error": str(e)}
    dirty = [line[3:] for line in st.splitlines() if line.strip()]
    return {"head": head, "dirty": dirty}


def log_run(event: Dict[str, Any]) -> None:
    """Append one line to experiments/results/R2_02_runs.jsonl: every real run is reported in the write-up."""
    event = dict(event, at=datetime.datetime.now().isoformat(timespec="seconds"), host=platform.node())
    with open(C.RUNS_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r1", default=str(C.R1_LENS_DEFAULT), help="R1_18_it_lens.pt")
    ap.add_argument("--miru", default=str(C.MIRU_DEFAULT), help="the Miru safetensors file (pinned revision)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=str(C.EVAL_JSON))
    ap.add_argument("--states", default=str(C.EVAL_STATES), help="where to save the captured states ('' = do not)")
    ap.add_argument("--selftest", action="store_true", help="CPU pipeline test with a tiny fake model (no model)")
    ap.add_argument("--limit", type=int, default=0, help="first N prompts per condition (self-test)")
    ap.add_argument("--check-only", action="store_true",
                    help="run every check made before the model is loaded (CPU; lens hashes included), then stop")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing output (the earlier run stays in R2_02_runs.jsonl and must be reported)")
    ap.add_argument("--allow-uncommitted", action="store_true",
                    help="run although the pre-registered files are not committed (a deviation; recorded)")
    args = ap.parse_args()

    if args.check_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.selftest:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if Path(args.out).resolve() == C.EVAL_JSON.resolve():
            raise SystemExit("--selftest must write elsewhere (--out)")
    elif args.limit:
        raise SystemExit("--limit is for --selftest only: the pre-registered run reads every prompt")

    files = {name: C.sha256_file(p) for name, p in C.PREREG_FILES.items() if p.is_file()}
    real = not args.selftest and not args.check_only
    # the checks of R2_PREREG.md §9, before the model is loaded
    gst = git_state(list(C.PREREG_FILES.values()))
    if gst["dirty"] or gst["head"] is None:
        msg = (f"pre-registered files not committed at HEAD {gst['head']}: {gst['dirty']}"
               + (f" ({gst['error']})" if gst.get("error") else ""))
        if real and not args.allow_uncommitted:
            raise SystemExit(msg + "; commit them first (R2_PREREG.md §9), or pass --allow-uncommitted (a deviation)")
        log(("WARNING: " if not args.selftest else "") + msg)
    else:
        log(f"pre-registered files committed and unchanged at HEAD {gst['head'][:12]}")
    if not args.selftest:
        existing = [str(x) for x in (Path(args.out), Path(args.states) if args.states else None)
                    if x is not None and x.exists()]
        if existing:
            if real and not args.force:
                raise SystemExit(f"{existing} already exist: an earlier run's output is not overwritten without "
                                 "--force (every run is reported; see R2_02_runs.jsonl)")
            log(f"{'WARNING: ' if args.check_only else ''}existing output {existing}"
                + (" will be overwritten (--force)" if real else ""))
    tok = C.load_tokenizer()
    data = C.load_prompts()
    val = C.validate_prompts(data, tok)
    if not val["ok"]:
        raise SystemExit(f"prompt file invalid: {val['problems']}")
    if not args.selftest:
        if not C.CHECK_JSON.is_file():
            raise SystemExit(f"run R2_01_check.py first (no {C.CHECK_JSON})")
        chk = json.loads(C.CHECK_JSON.read_text(encoding="utf-8"))
        if not chk.get("ok") or chk.get("prompts_sha256") != files["R2_prompts.json"]:
            raise SystemExit("R2_01_check.json does not pass this exact prompt file; rerun R2_01_check.py")
        if "R2_PREREG.md" not in files:
            raise SystemExit("no pre-registration on disk")
    fr = C.filter_report(tok)
    if not fr["ok"]:
        raise SystemExit("the token filter fails its spot checks")
    cls, _ = C.classify_vocab(tok)
    import torch

    mask = torch.from_numpy(cls == 0)
    is_special = C.special_mask(cls)
    if not args.selftest:
        chk_counts = (chk.get("filter") or {}).get("counts")
        if chk_counts != fr["counts"]:
            raise SystemExit(f"the filter's counts {fr['counts']} differ from R2_01_check.json's {chk_counts}; "
                             "rerun R2_01_check.py")
    log(f"filter: {fr['n_content']:,} CONTENT / {fr['n_noncontent']:,} NON-CONTENT of {fr['vocab_size']:,} "
        f"({int(is_special.sum())} special)")
    import hashlib

    meta_extra = {
        "files_sha256": files,
        "git": gst,
        "allow_uncommitted": bool(args.allow_uncommitted),
        "forced_overwrite": bool(args.force),
        "filter": {k: fr[k] for k in ("vocab_size", "n_content", "n_noncontent", "counts", "ok")},
        "mask_sha256": hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
        "selftest": bool(args.selftest),
    }
    if args.check_only:
        GpuBackend.check_files(Path(args.r1), Path(args.miru))
        log("check-only: every check before the model passes (prompts, R2_01 hash, filter, lens files); "
            "no model loaded")
        return
    if real:
        log_run({"event": "started", "argv": sys.argv, "files_sha256": files, "git": gst,
                 "forced_overwrite": bool(args.force), "allow_uncommitted": bool(args.allow_uncommitted)})
    try:
        if args.selftest:
            torch.set_num_threads(min(8, os.cpu_count() or 1))
            backend = FakeBackend.build(tok)
        else:
            backend = GpuBackend(Path(args.r1), Path(args.miru))
            if len(backend.tok) != len(mask):
                raise SystemExit(f"the runtime's tokenizer has {len(backend.tok)} ids, the mask {len(mask)}")
            tok = backend.tok
        states = Path(args.states) if args.states else None
        if args.selftest and states is not None and states.resolve() == C.EVAL_STATES.resolve():
            states = Path(args.out).with_suffix(".states.pt")
        with torch.no_grad():
            out = run(backend, tok, data, mask, is_special, args.batch, args.limit or None, Path(args.out), states,
                      meta_extra)
    except BaseException as e:
        if real:
            log_run({"event": "failed", "error": f"{type(e).__name__}: {e}"})
        raise
    if real:
        log_run({"event": "finished", "out": str(args.out), "out_sha256": C.sha256_file(Path(args.out)),
                 "summary": out["meta"]["summary"]})
    log("done")


if __name__ == "__main__":
    main()

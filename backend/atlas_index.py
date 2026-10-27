"""The atlas's indexes (`backend/atlas_index.py`): what is read once for every neuron and kept on disk, and the
queries over it.

The neuron explorer this page follows (lyramakesmusic/neuron-explorer) keeps, for every MLP neuron of its model, its
logit-lens words (`lens_index.npz`), its steer lens (`steer_index.npz`), its activation range (`act_scale.npz`), its
strongest weight links each way (`connections.npz`) and a seriated sheet layout (`layout_writes.npz`). The atlas keeps
the same for Gemma-4-31B's 60 x 21,504 neurons, built on the GPU by scripts/atlas_build.py, with one difference that is
this instrument's own: every neuron's words are kept through both readings the instrument uses, the J++ lens and the
naked eye (the logit lens), for both faces, ranked by the study's measure (`label_gate.label_top_ids`: fp32,
softcapped, ties to the lower id), so the label gate can be applied to every label the atlas shows or pools
(`label_check`: held back below layer 41, page count, and, under the base-fitted J++ lens, where the two readings share
none of their first ten word-pieces). The words are still sent; the pages leave held-back ones out of their main
reading and keep them in the apparatus, as everywhere else in the instrument.

Files, in the index folder (`index_dir()`: `$THEODOLITE_ATLAS_DIR/index`, else `atlas_data/index` in the checkout, which
git ignores), each with its provenance in a `meta` field (model, lens id, built_at, what it was read over):
- lens.npz: lens_pos, lens_neg, eye_pos, eye_neg [L, N, K] int32 (K = 12; the first ten are each label's first ten),
  silenced [L, N] bool (a zero write column), K, lens_id.
- act_scale.npz: pos, neg, pp, nn, pn, np [L, N] float32 (`atlas_ops.Extremes`), tokens, corpus, resid_q [L, 3] (the
  stream's length leaving each layer: 10th, 50th, 90th percentile).
- wiring.npz: {down,up}_{layer [L, N, K] int8, n int16, z float16, sign int8}, K (`atlas_ops.neuron_wiring` for every
  neuron at once; z below -1e4 marks an empty slot).
- steer_lens.npz: up, down [L, N, 12] int32, up_kl, down_kl [L, N] float16, dose, contexts (`atlas_ops.steer_lens`).
- layout_writes.npz: cell [L, N] int16, side (each neuron's place on its layer's sheet, neighbours writing alike).
- head_profiles.json: where each head looks over a few reference texts (built by the server on first need).
- vocab.json: every word-piece decoded alone (written by the server on first need).
Layers count from 0 in every array and field.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.label_gate import TOPK, label_check

ATLAS_DIR_ENV = "THEODOLITE_ATLAS_DIR"
LENS_K = 12
WIRING_K = 24
EMPTY_Z = -1e4  # an empty wiring slot (fewer links than K)
FILES = {
    "lens": "lens.npz",
    "act_scale": "act_scale.npz",
    "wiring": "wiring.npz",
    "steer_lens": "steer_lens.npz",
    "layout": "layout_writes.npz",
}
# the array of each file whose first two axes are [layers, neurons]
SHAPE_KEY = {"lens": "lens_pos", "act_scale": "pos", "wiring": "down_z", "steer_lens": "up", "layout": "cell"}


def atlas_dir() -> Path:
    env = os.environ.get(ATLAS_DIR_ENV)
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parent.parent / "atlas_data"


def index_dir() -> Path:
    return atlas_dir() / "index"


def save_npz(path: Path, compressed: bool = False, **arrays: Any) -> None:
    """Write an index file atomically (a temporary file beside it, then a rename), so a reader never sees half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    (np.savez_compressed if compressed else np.savez)(tmp, **arrays)
    os.replace(tmp, path)


def meta_of(z: Any) -> Dict[str, Any]:
    try:
        return json.loads(str(z["meta"]))
    except Exception:
        return {}


class AtlasIndex:
    """The index files of one folder, each loaded on first need and loaded again when it changes on disk (by mtime).

    Given the model read here (`model`, its runtime's model_id; `shape`, its (layers, neurons)), a file built for
    another model or of another shape is not used: it reads as not built, and `status()` says why (`refused`). A
    stand-in's indexes (scripts/atlas_stand_in.py) are never read for the model, nor the model's for a stand-in."""

    def __init__(self, root: Optional[Path] = None, model: Optional[str] = None, shape: Optional[Tuple[int, int]] = None) -> None:
        self.root = Path(root) if root is not None else index_dir()
        self.model = model
        self.shape = tuple(int(x) for x in shape) if shape is not None else None
        self._lock = threading.Lock()
        self._cache: Dict[str, Tuple[float, Optional[Dict[str, Any]], Optional[str]]] = {}
        self._base: Dict[str, Tuple[Tuple[float, int], np.ndarray]] = {}

    def path(self, name: str) -> Path:
        return self.root / FILES[name]

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        p = self.path(name)
        try:
            mt = p.stat().st_mtime
        except FileNotFoundError:
            return None
        with self._lock:
            hit = self._cache.get(name)
            if hit and hit[0] == mt:
                return hit[1]
            with np.load(p, allow_pickle=False) as z:
                data = {k: z[k] for k in z.files}
            data["meta"] = meta_of(data)
            why = self._refusal(name, data)
            self._cache[name] = (mt, None if why else data, why)
            return None if why else data

    def _refusal(self, name: str, data: Dict[str, Any]) -> Optional[str]:
        """Why a file does not belong to the model read here, in plain words; None when it does."""
        meta = data["meta"]
        if self.model is not None and meta.get("model") is not None and meta.get("model") != self.model:
            return f"built for {meta.get('model')}, not {self.model}: rebuild it (scripts/atlas_build.py)"
        arr = data.get(SHAPE_KEY.get(name, ""))
        if self.shape is not None and arr is not None and tuple(arr.shape[:2]) != self.shape:
            return (f"shaped for {arr.shape[0]} layers x {arr.shape[1]} neurons, not {self.shape[0]} x {self.shape[1]}: "
                    "rebuild it (scripts/atlas_build.py)")
        return None

    def refused(self, name: str) -> Optional[str]:
        hit = self._cache.get(name)
        return hit[2] if hit else None

    def status(self) -> Dict[str, Any]:
        """Which indexes exist here, and their provenance."""
        out: Dict[str, Any] = {"folder": str(self.root)}
        for name in FILES:
            p = self.path(name)
            if not p.exists():
                out[name] = {"built": False}
                continue
            d = self.get(name)
            if d is None:
                out[name] = {"built": False, "file": str(p), "refused": self.refused(name)}
                continue
            out[name] = {"built": True, "file": str(p), "bytes": p.stat().st_size, **d.get("meta", {})}
        return out

    # ------------------------------------------------------------------------------------------------------- labels

    def face_ids(self, layer: int, n: int, sign: str, reading: str, k: int = LENS_K) -> Optional[List[int]]:
        """A face's first k word-piece ids through the lens ("lens") or the naked eye ("eye"); None when not built."""
        d = self.get("lens")
        if d is None:
            return None
        key = f"{reading}_{'pos' if sign == '+' else 'neg'}"
        return [int(t) for t in d[key][int(layer), int(n), : int(k)]]

    def face_check(self, layer: int, n: int, sign: str, edited: bool = False) -> Optional[Dict[str, Any]]:
        """The label gate's check of one face, from the index (the same count `label_check` makes live: the ids were
        ranked by `label_top_ids`). None when the lens index is not built."""
        d = self.get("lens")
        if d is None:
            return None
        lens_ids = self.face_ids(layer, n, sign, "lens", TOPK)
        eye_ids = self.face_ids(layer, n, sign, "eye", TOPK)
        silenced = bool(d["silenced"][int(layer), int(n)])
        return label_check(int(layer), lens_ids, eye_ids, d["meta"].get("lens_id"), edited=edited, silenced=silenced)

    def label_passes(self, layer: int, n: int, sign: str) -> bool:
        c = self.face_check(layer, n, sign)
        return bool(c) and not c["held_back"]

    # ------------------------------------------------------------------------------------------------------- search

    def search(self, match: np.ndarray, limit: int = 300) -> Tuple[List[Dict[str, Any]], int]:
        """Neurons one of whose faces carries one of the word-piece ids `match` among its first LENS_K, through the lens,
        the naked eye or (when built) the steer lens: one row per neuron and face, its best match kept, best first (the
        reference's /api/search ranking: the match's rank, then how many matched). Returns (rows, total)."""
        sources: List[Tuple[str, str, np.ndarray]] = []
        lens = self.get("lens")
        if lens is not None:
            sources += [("+", "lens", lens["lens_pos"]), ("-", "lens", lens["lens_neg"]),
                        ("+", "eye", lens["eye_pos"]), ("-", "eye", lens["eye_neg"])]
        steer = self.get("steer_lens")
        if steer is not None:
            sources += [("+", "steer", steer["up"]), ("-", "steer", steer["down"])]
        if not sources or not len(match):
            return [], 0
        found = []
        for sign, src, arr in sources:
            hit = np.isin(arr, match)  # [L, N, K]
            count = hit.sum(-1)
            best = np.where(hit, np.arange(arr.shape[-1]), 99).min(-1)
            for l, n in zip(*np.nonzero(count)):
                found.append((int(best[l, n]), -int(count[l, n]), int(l), int(n), sign, src))
        seen, rows = set(), []
        for row in sorted(found):
            if (row[2], row[3], row[4]) not in seen:
                seen.add((row[2], row[3], row[4]))
                rows.append(row)
        out = [{"layer": l, "n": n, "sign": sg, "rank": rank, "matched": -cnt, "src": src} for rank, cnt, l, n, sg, src in rows[: int(limit)]]
        return out, len(rows)

    # ------------------------------------------------------------------------------------------------------- wiring

    def links(self, layer: int, n: int, k: int = WIRING_K) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """One neuron's k strongest weight links each way, from the wiring index; None when it is not built."""
        w = self.get("wiring")
        if w is None or int(k) > int(w["K"]):
            return None
        out: Dict[str, List[Dict[str, Any]]] = {}
        for side in ("down", "up"):
            z = w[f"{side}_z"][int(layer), int(n)]
            out[side] = [
                {"layer": int(w[f"{side}_layer"][int(layer), int(n), i]), "n": int(w[f"{side}_n"][int(layer), int(n), i]),
                 "z": round(float(z[i]), 2), "sign": int(w[f"{side}_sign"][int(layer), int(n), i])}
                for i in range(int(k)) if float(z[i]) > EMPTY_Z
            ]
        return out

    def cone(self, layer: int, n: int, k: int = 2, zmin: float = 15.0, cap: int = 20000, first: Optional[int] = None,
             hops: Optional[int] = None) -> List[Tuple[int, int, int, int, float, int]]:
        """A neuron's cone from the wiring index (the reference's `cone_links`): what it writes into, then what those
        write into, out to the last layer; and what writes into it, back to layer 0, each side moving one way only.
        The first hop takes the neuron's top `first` links (all K when None); later hops each node's top k with z at
        least zmin, up to `hops` hops; each side keeps at most cap / 2 links. Rows: (src layer, src neuron, dst layer,
        dst neuron, z * sign, hop), hop positive downstream and negative upstream. Weight-only links chained: every
        path past the first hop is a lead to check, not a circuit found."""
        ci = self.get("wiring")
        if ci is None:
            return []
        K = int(ci["K"])
        out: List[Tuple[int, int, int, int, float, int]] = []
        for side, step in (("down", 1), ("up", -1)):
            zs, ls, ns, sg = ci[f"{side}_z"], ci[f"{side}_layer"], ci[f"{side}_n"], ci[f"{side}_sign"]
            front, seen, hop, start = [(int(layer), int(n))], {(int(layer), int(n))}, 1, len(out)
            while front and len(out) - start < cap // 2 and (not hops or hop <= hops):
                nxt = []
                for a, b in front:
                    for i in range(min(int(first or K), K) if hop == 1 else min(int(k), K)):
                        z = float(zs[a, b, i])
                        if z <= EMPTY_Z or (hop > 1 and z < zmin):
                            break
                        t = (int(ls[a, b, i]), int(ns[a, b, i]))
                        src, dst = ((a, b), t) if side == "down" else (t, (a, b))
                        out.append((*src, *dst, z * int(sg[a, b, i]), hop * step))
                        if t not in seen:
                            seen.add(t)
                            nxt.append(t)
                front, hop = nxt, hop + 1
            del out[start + cap // 2 :]
        return out

    def _base_rates(self, src: str, ids_of: Any, n_layers: int, vocab: int) -> np.ndarray:
        """[L, V] float32: each word-piece's share among the first eight of every face of a layer's neurons (the lift's
        denominator), for the source pooled; remade when its index changes."""
        p = self.path("steer_lens" if src == "steer" else "lens")
        key = (p.stat().st_mtime, vocab)
        hit = self._base.get(src)
        if hit and hit[0] == key:
            return hit[1]
        pos, neg = ids_of
        N = pos.shape[1]
        df = np.stack([np.bincount(np.concatenate([pos[l, :, :8].ravel(), neg[l, :, :8].ravel()]), minlength=vocab)[:vocab]
                       for l in range(n_layers)])
        base = ((df + 1) / (2 * N)).astype(np.float32)
        self._base[src] = (key, base)
        return base

    def web_summary(self, layer: int, n: int, vocab: Sequence[str], first: int = 5, k: int = 2, top: int = 24,
                    hops: Optional[int] = None, src: str = "auto") -> Dict[str, Any]:
        """What the neuron's web says, in word-pieces (the reference's /api/web/summary): each neuron in the cone is read
        on the face the chain drives (the signs multiplied along the path), the first eight word-pieces of that face are
        pooled, merged across case and a leading space, weighted by link z over the largest z, times 0.8 per hop past
        the first, times the log of the lift (the share of the web's neurons carrying the word-piece over its share among
        every neuron of the same layer), so what every neuron of a layer carries sinks and what the web carries no more
        often than chance drops out. Downstream gives `toward` (the driven face) and `away` (the other); upstream
        `fed_by` (the face that excites it) and `held_by` (the face that holds it back).

        The words pooled are the steer lens's when it is built (`src` "steer": a measured effect on the next word-piece,
        which the label gate does not cover), else the J++ lens's, and then only faces the label gate lets through:
        `gated` counts the faces left out for it (a web that runs through early layers can leave out most of them)."""
        if src == "auto":
            src = "steer" if self.get("steer_lens") is not None else "lens"
        if src == "steer":
            st = self.get("steer_lens")
            if st is None:
                return {"src": None, "ready": False}
            pos, neg = st["up"], st["down"]
        else:
            lens = self.get("lens")
            if lens is None:
                return {"src": None, "ready": False}
            pos, neg = lens["lens_pos"], lens["lens_neg"]
        L = pos.shape[0]
        base = self._base_rates(src, (pos, neg), L, len(vocab))
        links = self.cone(layer, n, k, 15.0, 20000, first, hops)
        zmax = max([abs(z) for *_, z, _ in links] + [1.0])
        sign = {(int(layer), int(n)): 1}
        reach: Dict[Tuple[int, int], Tuple[int, float]] = {}
        for sl, sn, dl, dn, z, hop in links:
            near, far = ((sl, sn), (dl, dn)) if hop > 0 else ((dl, dn), (sl, sn))
            if far in sign or near not in sign:
                continue
            sign[far] = sign[near] * (1 if z > 0 else -1)
            reach[far] = (hop, abs(z))
        gated = {"down": 0, "up": 0}

        def pool(nodes: List[Tuple[Tuple[int, int], Tuple[int, float]]], flip: int, side: str) -> List[Dict[str, Any]]:
            acc: Dict[str, Dict[str, Any]] = {}
            used = 0
            for (l, m), (hop, z) in nodes:
                face = "+" if sign[(l, m)] * flip > 0 else "-"
                if src == "lens" and not self.label_passes(l, m, face):
                    gated[side] += 1
                    continue
                used += 1
                w = z / zmax * 0.8 ** (abs(hop) - 1)
                for t in (pos if face == "+" else neg)[l, m, :8]:
                    s = vocab[int(t)] if int(t) < len(vocab) else ""
                    key = s.strip().lower()
                    if not key or "�" in key:
                        continue
                    a = acc.setdefault(key, {"n": 0, "w": 0.0, "base": 0.0, "forms": {}})
                    a["n"] += 1
                    a["w"] += w
                    a["base"] += float(base[l, int(t)])
                    a["forms"][s.strip()] = a["forms"].get(s.strip(), 0) + 1
            for a in acc.values():
                lift = (a["n"] / max(1, used)) / (a["base"] / a["n"])
                a["score"] = a["w"] * float(np.log(lift)) if lift > 1 else 0.0
            ranked = sorted((a for a in acc.values() if a["score"] > 0), key=lambda a: -a["score"])[: int(top)]
            return [{"tok": max(a["forms"], key=a["forms"].get), "n": a["n"], "w": round(float(a["score"]), 2)} for a in ranked]

        down = [(key, val) for key, val in reach.items() if val[0] > 0]
        up = [(key, val) for key, val in reach.items() if val[0] < 0]
        res = {
            "src": src,
            "ready": True,
            "links": len(links),
            "down": {"neurons": len(down), "toward": pool(down, 1, "down"), "away": pool(down, -1, "down")},
            "up": {"neurons": len(up), "fed_by": pool(up, 1, "up"), "held_by": pool(up, -1, "up")},
        }
        res["gated"] = {"down": gated["down"], "up": gated["up"]}
        return res

    def active_paths(self, score: np.ndarray, top: int = 300, min_score: float = 1.0) -> Tuple[np.ndarray, int]:
        """The circuit active at one word-piece (the reference's /api/infer/paths): among the `top` neurons with the
        largest |score| there (at least `min_score`), every wiring link whose two ends both fired, its z signed by the
        push the writer's actual firing gives the reader (link sign x the sign of the writer's score). Rows int16
        (src layer, src neuron, dst layer, dst neuron, round(10 z) * push, 1); and how many neurons were active."""
        ci = self.get("wiring")
        if ci is None:
            return np.zeros((0, 6), np.int16), 0
        flat = np.abs(score).ravel()
        k = min(int(top), max(flat.size - 1, 1))
        cand = np.argpartition(-flat, k)[:k] if flat.size > k else np.arange(flat.size)
        cand = cand[flat[cand] >= float(min_score)]
        N = score.shape[1]
        active = {(int(i // N), int(i % N)) for i in cand}
        K = int(ci["K"])
        rows = []
        for l, n in active:
            for i in range(K):
                z = float(ci["down_z"][l, n, i])
                if z <= EMPTY_Z:
                    break
                t = (int(ci["down_layer"][l, n, i]), int(ci["down_n"][l, n, i]))
                if t in active:
                    push = int(ci["down_sign"][l, n, i]) * (1 if score[l, n] > 0 else -1)
                    rows.append((l, n, *t, int(round(z * 10)) * push, 1))
        return np.array(rows, np.int16).reshape(-1, 6), len(active)

    def layout(self) -> Optional[Dict[str, Any]]:
        d = self.get("layout")
        if d is None:
            return None
        return {"cell": d["cell"].astype(np.int16), "side": int(d["side"]), "meta": d["meta"]}


def scale_scores(acts: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Each activation as a multiple of the neuron's usual extreme on its side (the reference's `score_text`): the
    positive side over its largest positive activation, the negative side over its most negative, so 1 (or -1) is as
    strong as it usually gets; a side the neuron barely reaches is scaled by at least a tenth of its other side. acts
    [L, T, N]; pos, neg [L, N]. The start mark (position 0, the attention sink) scores 0."""
    p = np.maximum(pos, np.maximum(0.1 * -neg, 1e-3))[:, None, :]
    q = np.minimum(neg, np.minimum(-0.1 * pos, -1e-3))[:, None, :]
    a = acts.astype(np.float32)
    score = np.where(a >= 0, a / p, -(a / q)).astype(np.float32)
    if score.shape[1] > 1:
        score[:, 0, :] = 0
    return score


def match_ids(vocab: Sequence[str], query: str) -> np.ndarray:
    """The word-piece ids whose text contains the query, ignoring case (the reference's lens search)."""
    q = query.strip().lower()
    if len(q) < 2:
        return np.zeros(0, dtype=np.int32)
    return np.array([i for i, t in enumerate(vocab) if q in t.lower()], dtype=np.int32)

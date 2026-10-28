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
  silenced [L, N] bool (a zero write column), built [L] bool (the layers read so far: a build of some layers adds to
  the file, and an unread layer has no words), K, lens_id.
- act_scale.npz: pos, neg, pp, nn, pn, np [L, N] float32 (`atlas_ops.Extremes`), tokens, corpus, resid_q [L, 3] (the
  stream's length leaving each layer: 10th, 50th, 90th percentile), mlp_rms_q, mid_rms_q [L, 3] (the same of the two
  feed-forward norms' scales, which weigh the wiring across layers).
- wiring.npz: {down,up}_{layer [L, N, K] int8, n int16, z float16, lw float16, sign int8, gate int8}, K
  (`atlas_ops.neuron_wiring` for every neuron at once, ordered by strength across layers; lw is log10 of a link's
  strength over the neuron's strongest; z below -1e4 marks an empty slot).
- steer_lens.npz: up, down [L, N, 12] int32, up_kl, down_kl [L, N] float16, dose, contexts (`atlas_ops.steer_lens`).
- layout_writes.npz: cell [L, N] int16, side (each neuron's place on its layer's sheet, neighbours writing alike).
- passages.d/: one .npy per array, memory-mapped (on the 31B about 0.5 GB at K = 16, 0.7 GB with six sampled events a
  face), and meta.json, written last (`atlas_build.build_passages`): windows [W, seq] int32 (the windows' word-piece ids, -1 past a short one's end:
  the corpus's own text, never committed), unigram [V] int32 (each word-piece's count over the word-pieces read, the
  base rate), and per face ("pos", "neg") the K strongest events, at most one a window, ties to the lower window:
  {face}_win [L, N, K] int32 (-1 an empty slot), {face}_at int16 (the position in its window), {face}_act, {face}_g,
  {face}_u float16 (the activation, and the gate and up pre-activations there; float16 reaches 65,504); with a sampled
  build also
  {face}_smp_{win, at, act, g, u} [L, N, S], drawn at random above half the face's extreme.
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
PASSAGES_K = 16
PASSAGES_DIR = "passages.d"
FILES = {
    "lens": "lens.npz",
    "act_scale": "act_scale.npz",
    "wiring": "wiring.npz",
    "steer_lens": "steer_lens.npz",
    "layout": "layout_writes.npz",
    "passages": f"{PASSAGES_DIR}/meta.json",  # a folder of .npy arrays, memory-mapped; its meta.json is written last
}
FOLDERS = {"passages"}
# the array of each file whose first two axes are [layers, neurons]
SHAPE_KEY = {"lens": "lens_pos", "act_scale": "pos", "wiring": "down_z", "steer_lens": "up", "layout": "cell", "passages": "pos_act"}
# arrays an index file must hold to be read: older builds lack them and are refused, with what to do
NEEDS = {
    "lens": ("built", "built before it marked which layers it has read"),
    "act_scale": ("mid_rms_q", "built before it kept the feed-forward norms' scales, which weigh the wiring"),
    "wiring": ("down_lw", "built before links were weighed across layers (Gemma's layer scalars and norms) and kept their gate"),
}
# the arrays a folder index must hold beside its meta.json (with a sampled build, its samples' too): a folder copied or
# removed in part is refused, not read in part
PASSAGE_ARRAYS = tuple(f"{f}_{x}" for f in ("pos", "neg") for x in ("win", "at", "act", "g", "u"))
FOLDER_NEEDS = {"passages": ("windows", "unigram") + PASSAGE_ARRAYS}


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


def _weights_name(card: Dict[str, Any]) -> str:
    if card.get("format") == "gguf":
        return f"the GGUF {card.get('file')} (fingerprint {card.get('id')})"
    if card.get("format") == "stand-in":
        return "the stand-in"
    snap, files = card.get("snapshot"), card.get("files_sha256")
    said = ([f"weight files {str(files)[:16]}"] if files else []) + ([f"snapshot {str(snap)[:12]}"] if snap else [])
    return f"{card.get('repo') or card.get('format')}" + (f" ({', '.join(said)})" if said else "")


def weights_differ(built: Optional[Dict[str, Any]], here: Optional[Dict[str, Any]]) -> Optional[str]:
    """How the weights an index was built from (`built`: its meta's "weights", `runtime_factory.weights_card` of the
    runtime that built it) differ from the runtime's here (`here`), in plain words; None when they are the same as far
    as the two cards tell, or when either is unknown (an index built before its meta named its weights, or a
    checkpoint whose weight files the cache did not name), so that the model id alone is compared, as before. A GGUF
    is compared by its fingerprint and the sha256 of its header (`gguf_model.provenance`), the checkpoint by its repo
    id and its weight files (`files_sha256`: `runtime_factory.checkpoint_files_id`), not by the snapshot that holds
    them, which another commit of the same files (a model card edited) changes; how a GGUF's layers are held (packed,
    resident, dense) is not compared: the three give the same forward bit for bit (scripts/check_gguf.py), and a way
    of computing that does not is in the model id."""
    if not built or not here:
        return None
    a, b = built.get("format"), here.get("format")
    if a != b:
        return f"{_weights_name(built)}, not {_weights_name(here)}"
    if a == "gguf":
        same = (built.get("id"), built.get("sha256_header")) == (here.get("id"), here.get("sha256_header"))
        return None if same else f"{_weights_name(built)}, not {_weights_name(here)}"
    if a == "stand-in":
        return None
    if built.get("repo") != here.get("repo") or (built.get("files_sha256") and here.get("files_sha256")
                                                 and built.get("files_sha256") != here.get("files_sha256")):
        return f"{_weights_name(built)}, not {_weights_name(here)}"
    return None


def meta_of(z: Any) -> Dict[str, Any]:
    try:
        return json.loads(str(z["meta"]))
    except Exception:
        return {}


def load_folder(meta_path: Path) -> Dict[str, Any]:
    """A folder index: every .npy beside `meta_path` memory-mapped (read-only, nothing loaded until read), and the meta."""
    data: Dict[str, Any] = {f.stem: np.load(f, mmap_mode="r", allow_pickle=False) for f in sorted(meta_path.parent.glob("*.npy"))}
    try:
        data["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data["meta"] = {}
    return data


class AtlasIndex:
    """The index files of one folder, each loaded on first need and loaded again when it changes on disk (by mtime).

    Given the model read here (`model`, its runtime's model_id; `shape`, its (layers, neurons); `weights`, its weights
    card, `runtime_factory.weights_card`), a file built for another model, of another shape or from other weights
    (`weights_differ`: another GGUF behind the same model id, other weight files of the checkpoint) is not used: it reads
    as not built, and `status()` says why (`refused`). A stand-in's indexes (scripts/atlas_stand_in.py) are never read
    for the model, nor the model's for a stand-in."""

    def __init__(self, root: Optional[Path] = None, model: Optional[str] = None, shape: Optional[Tuple[int, int]] = None,
                 weights: Optional[Dict[str, Any]] = None) -> None:
        self.root = Path(root) if root is not None else index_dir()
        self.model = model
        self.shape = tuple(int(x) for x in shape) if shape is not None else None
        self.weights = weights
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
            if name in FOLDERS:
                data = load_folder(p)
            else:
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
        other = weights_differ(meta.get("weights"), self.weights)
        if other:
            return f"built from other weights ({other}): rebuild it (scripts/atlas_build.py)"
        arr = data.get(SHAPE_KEY.get(name, ""))
        if self.shape is not None and arr is not None and tuple(arr.shape[:2]) != self.shape:
            return (f"shaped for {arr.shape[0]} layers x {arr.shape[1]} neurons, not {self.shape[0]} x {self.shape[1]}: "
                    "rebuild it (scripts/atlas_build.py)")
        need = NEEDS.get(name)
        if need and need[0] not in data:
            return f"{need[1]}: rebuild it (scripts/atlas_build.py)"
        want = list(FOLDER_NEEDS.get(name, ()))
        if name == "passages" and int(meta.get("sample") or 0):
            want += [x.replace("_", "_smp_", 1) for x in PASSAGE_ARRAYS]
        missing = [x for x in want if x not in data]
        if missing:
            return (f"its folder lacks {', '.join(missing[:3])}{' and more' if len(missing) > 3 else ''} (copied or removed in "
                    f"part): rebuild it (scripts/atlas_build.py {name})")
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
            size = sum(f.stat().st_size for f in p.parent.iterdir() if f.is_file()) if name in FOLDERS else p.stat().st_size
            out[name] = {"built": True, "file": str(p.parent if name in FOLDERS else p), "bytes": size, **d.get("meta", {})}
        return out

    # ------------------------------------------------------------------------------------------------------- labels

    def lens_layer(self, layer: int) -> Optional[Dict[str, Any]]:
        """The lens index when it is built and has read `layer`; None otherwise."""
        d = self.get("lens")
        if d is None or not bool(d["built"][int(layer)]):
            return None
        return d

    def face_ids(self, layer: int, n: int, sign: str, reading: str, k: int = LENS_K) -> Optional[List[int]]:
        """A face's first k word-piece ids through the lens ("lens") or the naked eye ("eye"); None when not built, or
        when the build has not read this layer yet."""
        d = self.lens_layer(layer)
        if d is None:
            return None
        key = f"{reading}_{'pos' if sign == '+' else 'neg'}"
        return [int(t) for t in d[key][int(layer), int(n), : int(k)]]

    def face_check(self, layer: int, n: int, sign: str, edited: bool = False) -> Optional[Dict[str, Any]]:
        """The label gate's check of one face, from the index (the same count `label_check` makes live: the ids were
        ranked by `label_top_ids`). None when the lens index is not built or has not read this layer."""
        d = self.lens_layer(layer)
        if d is None:
            return None
        lens_ids = self.face_ids(layer, n, sign, "lens", TOPK)
        eye_ids = self.face_ids(layer, n, sign, "eye", TOPK)
        silenced = bool(d["silenced"][int(layer), int(n)])
        return label_check(int(layer), lens_ids, eye_ids, d["meta"].get("lens_id"), edited=edited, silenced=silenced)

    def label_passes(self, layer: int, n: int, sign: str) -> bool:
        c = self.face_check(layer, n, sign)
        return bool(c) and not c["held_back"]

    # ----------------------------------------------------------------------------------------------------- passages

    def passages(self, layer: int, n: int, sign: str, k: int = PASSAGES_K) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """One face's stored events from the passages index: `top`, its k strongest (at most one a window, strongest
        first), and `sample`, those drawn at random above half its extreme (when the build drew them); each {window, at,
        act, g, u}. None when the index is not built."""
        d = self.get("passages")
        if d is None:
            return None
        f = "pos" if sign == "+" else "neg"

        def slots(group: str, kk: int) -> List[Dict[str, Any]]:
            if f"{group}_win" not in d:
                return []
            win = d[f"{group}_win"][int(layer), int(n)]
            cols = [i for i in range(min(int(kk), win.shape[0])) if int(win[i]) >= 0]
            return [{"window": int(win[i]), "at": int(d[f"{group}_at"][int(layer), int(n), i]),
                     "act": float(d[f"{group}_act"][int(layer), int(n), i]), "g": float(d[f"{group}_g"][int(layer), int(n), i]),
                     "u": float(d[f"{group}_u"][int(layer), int(n), i])} for i in cols]

        return {"top": slots(f, k), "sample": slots(f"{f}_smp", 1 << 30)}

    def window_ids(self, w: int) -> Optional[List[int]]:
        """A stored window's word-piece ids (from its start mark), or None when the passages index is not built."""
        d = self.get("passages")
        if d is None:
            return None
        row = np.asarray(d["windows"][int(w)])
        return [int(t) for t in row[row >= 0]]

    # ------------------------------------------------------------------------------------------------------- search

    def search(self, match: np.ndarray, limit: int = 300) -> Tuple[List[Dict[str, Any]], int]:
        """Neurons one of whose faces carries one of the word-piece ids `match` among its first LENS_K, through the lens,
        the naked eye or (when built) the steer lens: one row per neuron and face, its best match kept, best first (the
        reference's /api/search ranking: the match's rank, then how many matched). Returns (rows, total)."""
        sources: List[Tuple[str, str, np.ndarray]] = []
        lens = self.get("lens")
        unread = None
        if lens is not None:
            unread = ~lens["built"].astype(bool)  # a layer the build has not read has no words to match
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
            if src != "steer" and unread is not None and unread.any():
                hit[unread] = False
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
        """One neuron's k strongest weight links each way, from the wiring index (each {layer, n, z, w, sign, gate}, as
        `atlas_ops.neuron_wiring` gives them); None when it is not built."""
        w = self.get("wiring")
        if w is None or int(k) > int(w["K"]):
            return None
        return {side: [self._slot(w, side, int(layer), int(n), i) for i in range(int(k)) if float(w[f"{side}_z"][int(layer), int(n), i]) > EMPTY_Z]
                for side in ("down", "up")}

    @staticmethod
    def _slot(w: Dict[str, Any], side: str, l: int, n: int, i: int) -> Dict[str, Any]:
        return {"layer": int(w[f"{side}_layer"][l, n, i]), "n": int(w[f"{side}_n"][l, n, i]), "z": round(float(w[f"{side}_z"][l, n, i]), 2),
                "w": float(f"{10.0 ** float(w[f'{side}_lw'][l, n, i]):.4g}"), "sign": int(w[f"{side}_sign"][l, n, i]),
                "gate": int(w[f"{side}_gate"][l, n, i])}

    def _hop_links(self, ci: Dict[str, Any], side: str, a: int, b: int, hop: int, first: Optional[int], k: int, zmin: float) -> List[Dict[str, Any]]:
        """The links a node passes on at one hop of a cone, strongest first: on the first hop its top `first` (all K when
        None); past it, its top k among those with z at least zmin."""
        K = int(ci["K"])
        want = min(int(first or K), K) if hop == 1 else min(int(k), K)
        out = []
        for i in range(K):
            z = float(ci[f"{side}_z"][a, b, i])
            if z <= EMPTY_Z or len(out) >= want:
                break
            if hop > 1 and z < zmin:
                continue
            out.append(self._slot(ci, side, a, b, i))
        return out

    def cone(self, layer: int, n: int, k: int = 2, zmin: float = 15.0, cap: int = 20000, first: Optional[int] = None,
             hops: Optional[int] = None) -> List[Tuple[int, int, int, int, float, int]]:
        """A neuron's cone from the wiring index (the reference's `cone_links`): what it writes into, then what those
        write into, out to the last layer; and what writes into it, back to layer 0, each side moving one way only.
        The first hop takes the neuron's top `first` links by strength (all K when None); later hops each node's top k
        with z at least zmin, up to `hops` hops; each side keeps at most cap / 2 links. Rows: (src layer, src neuron,
        dst layer, dst neuron, z * sign, hop), hop positive downstream and negative upstream. Weight-only links chained:
        every path past the first hop is a lead to check, not a circuit found."""
        ci = self.get("wiring")
        if ci is None:
            return []
        out: List[Tuple[int, int, int, int, float, int]] = []
        for side, step in (("down", 1), ("up", -1)):
            front, seen, hop, start = [(int(layer), int(n))], {(int(layer), int(n))}, 1, len(out)
            while front and len(out) - start < cap // 2 and (not hops or hop <= hops):
                nxt = []
                for a, b in front:
                    for c in self._hop_links(ci, side, a, b, hop, first, k, zmin):
                        t = (c["layer"], c["n"])
                        src, dst = ((a, b), t) if side == "down" else (t, (a, b))
                        out.append((*src, *dst, c["z"] * c["sign"], hop * step))
                        if t not in seen:
                            seen.add(t)
                            nxt.append(t)
                front, hop = nxt, hop + 1
            del out[start + cap // 2 :]
        return out

    def walk(self, layer: int, n: int, side: str, face: Optional[str], first: Optional[int] = 5, k: int = 2, zmin: float = 15.0,
             hops: Optional[int] = None, cap: int = 10000) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """The cone followed by the link model (atlas_ops.LINK_MODEL): from rest a writer moves its reader only when it
        fires on the face that opens the reader's gate (`gate`), and pushes it toward one face (`sign`).
        - "down", from the neuron firing on `face`: a link is followed when the face its writer was reached on opens the
          reader's gate; the reader is reached on the face the link pushes it toward.
        - "up" (`face` None): each writer is reached on the face that opens its reader's gate; past the first hop a
          writer counts only when it pushes its reader toward the face that reader must fire on. `drives` is the face of
          the open neuron the chain pushes it toward.
        {neuron: {face, hop, w, drives?}}, each reached once (the first way found), w the strength of the link that
        reached it over the strongest of its node's links."""
        ci = self.get("wiring")
        if ci is None:
            return {}
        sgn = lambda s: "+" if s > 0 else "-"  # noqa: E731
        reach: Dict[Tuple[int, int], Dict[str, Any]] = {}
        front: List[Tuple[Tuple[int, int], Optional[str], Optional[str]]] = [((int(layer), int(n)), face, None)]
        seen, hop = {(int(layer), int(n))}, 1
        while front and len(reach) < cap and (not hops or hop <= hops):
            nxt = []
            for (a, b), f, drives in front:
                for c in self._hop_links(ci, side, a, b, hop, first, k, zmin):
                    t = (c["layer"], c["n"])
                    if side == "down":
                        if sgn(c["gate"]) != f:
                            continue  # fired on this face, the writer leaves the reader's gate near shut
                        tf, td = sgn(c["sign"]), None
                    else:
                        if f is not None and sgn(c["sign"]) != f:
                            continue  # this writer pushes its reader toward the other face
                        tf, td = sgn(c["gate"]), drives or sgn(c["sign"])
                    if t in seen:
                        continue
                    seen.add(t)
                    reach[t] = {"face": tf, "hop": hop if side == "down" else -hop, "w": c["w"], "z": c["z"], **({"drives": td} if td else {})}
                    nxt.append((t, tf, td))
            front, hop = nxt, hop + 1
        return reach

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
        """What the neuron's web says, in word-pieces (after the reference's /api/web/summary), the cone followed by the
        link model (`walk`; atlas_ops.LINK_MODEL):
        - down `plus` / `minus`: what the neurons it drives write when it fires on its + face / its - face, each read on
          the face the chain pushes it toward;
        - up `plus` / `minus`: what the neurons that push it toward its + face / its - face write, each read on the face
          that opens its reader's gate.
        Each face's first eight word-pieces are pooled, merged across case and a leading space, weighted by the strength
        of the link that reached it (over the strongest of its node's links), times 0.8 per hop past the first, times
        the log of the lift (the share of the web's neurons carrying the word-piece over its share among every neuron of
        the same layer), so what every neuron of a layer carries sinks and what the web carries no more often than
        chance drops out.

        The words pooled are the steer lens's when it is built (`src` "steer": a measured effect on the next word-piece,
        which the label gate does not cover), else the J++ lens's, and then only faces the label gate lets through:
        `gated` counts the faces left out for it (a web that runs through early layers can leave out most of them). The
        gate was validated for single neurons' labels, not for words pooled over many: a pool is a lead. Not ready
        (`missing`) without the wiring index, or without both the steer-lens and the lens index."""
        if self.get("wiring") is None:
            return {"src": None, "ready": False, "missing": "wiring"}
        if src == "auto":
            src = "steer" if self.get("steer_lens") is not None else "lens"
        if src == "steer":
            st = self.get("steer_lens")
            if st is None:
                return {"src": None, "ready": False, "missing": "steer_lens"}
            pos, neg = st["up"], st["down"]
        else:
            lens = self.get("lens")
            if lens is None:
                return {"src": None, "ready": False, "missing": "lens"}
            pos, neg = lens["lens_pos"], lens["lens_neg"]
        L = pos.shape[0]
        base = self._base_rates(src, (pos, neg), L, len(vocab))
        walks = {("down", "+"): self.walk(layer, n, "down", "+", first, k, 15.0, hops),
                 ("down", "-"): self.walk(layer, n, "down", "-", first, k, 15.0, hops)}
        up = self.walk(layer, n, "up", None, first, k, 15.0, hops)
        walks[("up", "+")] = {t: r for t, r in up.items() if r["drives"] == "+"}
        walks[("up", "-")] = {t: r for t, r in up.items() if r["drives"] == "-"}
        gated = {"down": 0, "up": 0}

        def pool(nodes: Dict[Tuple[int, int], Dict[str, Any]], side: str) -> List[Dict[str, Any]]:
            acc: Dict[str, Dict[str, Any]] = {}
            used = 0
            for (l, m), r in nodes.items():
                face, hop = r["face"], r["hop"]
                if src == "lens" and not self.label_passes(l, m, face):
                    gated[side] += 1
                    continue
                used += 1
                w = r["w"] * 0.8 ** (abs(hop) - 1)
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

        res = {
            "src": src,
            "ready": True,
            "links": sum(len(v) for v in walks.values()),
            "down": {sg: {"neurons": len(walks[("down", sg)]), "words": pool(walks[("down", sg)], "down")} for sg in ("+", "-")},
            "up": {sg: {"neurons": len(walks[("up", sg)]), "words": pool(walks[("up", sg)], "up")} for sg in ("+", "-")},
        }
        res["gated"] = {"down": gated["down"], "up": gated["up"]}
        return res

    def active_paths(self, score: np.ndarray, top: int = 300, min_score: float = 1.0) -> Tuple[np.ndarray, int]:
        """The circuit active at one word-piece (the reference's /api/infer/paths): among the `top` neurons with the
        largest |score| there (at least `min_score`), every wiring link whose two ends both fired and whose writer fired
        on the face that opens its reader's gate (atlas_ops.LINK_MODEL; fired the other way it barely moves the reader),
        its z signed by the face the link pushes the reader toward. Rows int16 (src layer, src neuron, dst layer, dst
        neuron, round(10 z) * sign, 1); and how many neurons were active."""
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
                if t in active and int(ci["down_gate"][l, n, i]) == (1 if score[l, n] > 0 else -1):
                    rows.append((l, n, *t, int(round(z * 10)) * int(ci["down_sign"][l, n, i]), 1))
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

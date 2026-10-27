"""The scientist's tools (`backend/scientist_tools.py`): what the atlas's scientist chat can read and do, the same for
each of its backends.

After the explorer's agent tools (lyramakesmusic/neuron-explorer, `explorer/agent_mcp.py`), on the atlas's routes
(backend/atlas.py). Claude Code and Codex reach them through a stdio MCP server (backend/scientist_mcp.py); a model
behind an OpenAI-compatible endpoint (LM Studio, Ollama, OpenRouter, vLLM, llama.cpp's server) gets them as functions,
and the server runs its tool loop (backend/scientist_chat.py). Standard library only, so the MCP server starts fast.

Two of the instrument's rules shape every reply:
- Layers count as the pages count them, from 1 (page layer = the API's layer + 1), in every argument and every field,
  and neurons are cited `L54 N15491`, heads `L54 h3`: the scientist writes for a person reading the pages. The
  atlas's HTTP API counts from 0 (`from_page` turns arguments into its count, `to_page` turns replies back).
- A neuron's words read through the J++ lens, or by the naked eye, come with the label gate's verdict (`label`:
  "shown", or "held back: <reasons>"); the gate holds back every label below page layer 41 and, under the base-fitted
  lens, one whose two readings share none of their first ten word-pieces (backend/label_gate.py). The words are still
  given, for the scientist to weigh; a held-back label is not evidence of what a neuron does. The steer lens is a
  measured effect, which the gate does not cover.

`call(name, args, api, chat)` runs one tool: `api` reaches the atlas (`await api.get(path, **query)`,
`await api.post(path, body)`, each raising `ToolError` with the server's plain-word refusal), `chat` is the chat the
tool's screen actions go to. `TOOLS` lists them as MCP describes tools ({name, description, inputSchema});
`openai_tools()` as an OpenAI-compatible endpoint takes them.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

PAGE_NOTE = "Layers count from 1, as the pages count them (60 on Gemma-4-31B; `overview` gives this model's)."


class ToolError(Exception):
    """A tool's refusal or failure, in plain words, for the model to read."""


# ----------------------------------------------------------------------------------------------- counting layers


def from_page(args: Any) -> Any:
    """Arguments in the pages' count (layers from 1) turned into the API's (from 0): every "layer" (also inside
    `neurons`, `heads`, `items`) and both ends of `layers`. A layer below 1 is refused."""
    if isinstance(args, dict):
        out: Dict[str, Any] = {}
        for k, v in args.items():
            if k == "layer" and isinstance(v, int) and not isinstance(v, bool):
                if v < 1:
                    raise ToolError(f"Layer {v} does not exist: layers count from 1 here, as on the pages.")
                out[k] = v - 1
            elif k == "layers" and isinstance(v, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in v):
                if any(x < 1 for x in v):
                    raise ToolError("Layers count from 1 here, as on the pages.")
                out[k] = [x - 1 for x in v]
            else:
                out[k] = from_page(v)
        return out
    if isinstance(args, list):
        return [from_page(v) for v in args]
    return args


def to_page(x: Any) -> Any:
    """A reply in the API's count turned into the pages': every "layer" field, every key of a "z_by_layer" table, and
    every entry of "census_layers"."""
    if isinstance(x, dict):
        out: Dict[str, Any] = {}
        for k, v in x.items():
            if k == "layer" and isinstance(v, int) and not isinstance(v, bool):
                out[k] = v + 1
            elif k == "z_by_layer" and isinstance(v, dict):
                out[k] = {str(int(kk) + 1): vv for kk, vv in v.items()}
            elif k == "census_layers" and isinstance(v, list):
                out[k] = [int(i) + 1 for i in v]
            else:
                out[k] = to_page(v)
        return out
    if isinstance(x, list):
        return [to_page(v) for v in x]
    return x


def cite(layer: int, n: int) -> str:
    """A neuron as the pages cite it, from the API's layer: `L54 N15491`."""
    return f"L{int(layer) + 1} N{int(n)}"


def cite_head(layer: int, head: int) -> str:
    return f"L{int(layer) + 1} h{int(head)}"


def label(check: Optional[Dict[str, Any]]) -> Optional[str]:
    """The label gate's verdict on a face, in words."""
    if not check:
        return None
    if check.get("held_back"):
        return "held back: " + ", ".join(check.get("reasons") or [])
    return "shown"


def face_words(w: Optional[Dict[str, Any]], k: int = 6) -> Dict[str, Any]:
    """One face's words from an atlas reply ({lens, eye, check, steer}), with the gate's verdict."""
    w = w or {}
    out: Dict[str, Any] = {}
    if w.get("steer"):
        out["steer"] = list(w["steer"])[:k]
    if w.get("lens") is not None:
        out["lens"] = list(w["lens"])[:k]
        out["eye"] = list(w.get("eye") or [])[:k]
        out["label"] = label(w.get("check"))
    if w.get("stale"):
        out["stale"] = w["stale"]
    return out


# --------------------------------------------------------------------------------------------------------- schemas

LN = {"layer": {"type": "integer", "description": "page layer, from 1"}, "n": {"type": "integer", "description": "neuron, from 0"}}
HEAD = {"layer": {"type": "integer", "description": "page layer, from 1"}, "head": {"type": "integer", "description": "head, from 0 (`overview` gives each kind of layer's count)"}}
PUSHES = {"type": "array", "description": "neurons pushed (or layers' whole blocks scaled) during the same pass",
          "items": {"type": "object", "properties": {
              **LN, "all": {"type": "boolean", "description": "the layer's whole feed-forward block (mode mul)"},
              "dose": {"type": "number"}, "mode": {"type": "string", "enum": ["add", "mul", "act"]}}, "required": ["layer"]}}
HEADS = {"type": "array", "description": "attention heads scaled during the same pass (dose 0 silences one; head 'all' every head of the layer)",
         "items": {"type": "object", "properties": {"layer": {"type": "integer"}, "head": {"type": ["integer", "string"]},
                                                     "dose": {"type": "number", "default": 0}}, "required": ["layer", "head"]}}
AT = {"type": "integer", "description": "a display unit by its index in the returned tokens (negative from the end); overrides token"}

TOOLS: List[Dict[str, Any]] = [
    {"name": "overview", "description": "The model's shape (layers, neurons, attention kinds and head counts), which of the atlas's indexes are "
     "built, which layers have a census, the push's units, the label gate's rule and the lens's caution, and how many neurons "
     "and heads are named. Read it first. " + PAGE_NOTE,
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "neuron", "description": "Everything kept for one neuron: its name and note; both faces' words through the J++ lens and by "
     "the naked eye, each with the label gate's verdict; its steer lens (the next word-piece's measured change when pushed "
     "±0.8, with KL); its activation range; its census rank if any; its 24 strongest direct-path weight links each way "
     "(writes_into, reads_from), each linked neuron with the face the link drives, that face's words, census words and name.",
     "inputSchema": {"type": "object", "properties": LN, "required": ["layer", "n"]}},
    {"name": "rollouts", "description": "The census of a neuron: what the model wrote when it was pushed, per question and dose, beside the "
     "unpushed answers, with its distinctive words and their mass against random-direction controls. Empty when the layer has no census.",
     "inputSchema": {"type": "object", "properties": {**LN, "max_chars": {"type": "integer", "default": 500}}, "required": ["layer", "n"]}},
    {"name": "cone", "description": "The neuron's cone: its weight links chained forward (what reads it, hop > 0) and backward (what feeds "
     "it, hop < 0), the first hop all of them, later hops each node's top k. Each neuron reached with its hop, z, the face the "
     "chain of signs drives and its words. Multi-hop paths are weight-only guesses: leads to check, not circuits. Needs the wiring index.",
     "inputSchema": {"type": "object", "properties": {**LN, "k": {"type": "integer", "default": 2}, "limit": {"type": "integer", "default": 120}},
                     "required": ["layer", "n"]}},
    {"name": "web_summary", "description": "The words of the neuron's web, the summary the person sees beside the map: the steer lens (or, "
     "without it, the gate-passing lens labels) pooled over its cone, each neuron on the face the chain drives, ranked by lift "
     "over its layer's base rate. toward / away: what its readers push the model toward and away from when it fires; fed_by / "
     "held_by: what its writers say when they excite / inhibit it. Needs the wiring index.",
     "inputSchema": {"type": "object", "properties": {**LN, "first": {"type": "integer", "default": 5}, "k": {"type": "integer", "default": 2}},
                     "required": ["layer", "n"]}},
    {"name": "search_lens", "description": "Neurons one of whose faces carries a word-piece holding the query (ignoring case) among its first "
     "twelve, through the lens, by the naked eye or in its steer lens; best matches first, each with its words and the gate's verdict.",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 40}}, "required": ["query"]}},
    {"name": "top_units", "description": "A layer's neurons ranked by their census words' mass against random-direction controls, with their "
     "words and names. Empty when the layer has no census.",
     "inputSchema": {"type": "object", "properties": {"layer": LN["layer"], "limit": {"type": "integer", "default": 30}}, "required": ["layer"]}},
    {"name": "set_name", "description": "Name a neuron in the atlas (shown in its panel, in lists and on the map), or clear its name with an "
     "empty string. Only when the person asks, or after they agree to a name you suggested.",
     "inputSchema": {"type": "object", "properties": {**LN, "name": {"type": "string"}}, "required": ["layer", "n", "name"]}},
    {"name": "write_note", "description": "Keep a short note on a neuron (your evidence in a few lines, beside its name), or clear it with an "
     "empty string. Only when the person asks, or agrees.",
     "inputSchema": {"type": "object", "properties": {**LN, "note": {"type": "string"}}, "required": ["layer", "n", "note"]}},
    {"name": "names", "description": "Every neuron and head named so far, as {\"L54 N15491\": name} and {\"L54 h3\": name}.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "steer", "description": "Live: the model answers a chat prompt three ways in one batch, pushed up, as it is, and pushed down "
     "(greedy). One neuron (layer, n) or several (neurons: [{layer, n, dose?, mode?}], all at once, each the other way in the down "
     "row). mode add: dose is a share of the stream's median length leaving the layer, along the neuron's write direction (0.8 is "
     "the steer lens's push; 2 to 3 is strong); mul: a factor on its activation (0 silences it); act: a multiple of its largest "
     "|activation|, added to it.",
     "inputSchema": {"type": "object", "properties": {**LN, "prompt": {"type": "string"}, "neurons": PUSHES,
                                                       "dose": {"type": "number", "default": 0.8},
                                                       "mode": {"type": "string", "enum": ["add", "mul", "act"], "default": "add"},
                                                       "max_new": {"type": "integer", "default": 150}}, "required": ["prompt"]}},
    {"name": "activations", "description": "Live: the neuron's activation at every display unit of a text (chat: true makes it a user turn), "
     "beside its range from `neuron`. `neurons` and `heads` are pushed or scaled in the same pass, to test what drives it: compare "
     "with the same call without them. Only what comes before its own feed-forward block can move it.",
     "inputSchema": {"type": "object", "properties": {**LN, "text": {"type": "string"}, "chat": {"type": "boolean", "default": False},
                                                       "neurons": PUSHES, "heads": HEADS}, "required": ["layer", "n", "text"]}},
    {"name": "compare", "description": "Live: many neurons across several texts, one forward per text. Per neuron and text: its strongest "
     "positive and negative activation with the unit, and its peak score (activation over its usual extreme on that side; 1 = as "
     "strong as it usually gets). For screening candidates against targets and controls. Up to 300 neurons and 24 texts.",
     "inputSchema": {"type": "object", "properties": {
         "neurons": {"type": "array", "items": {"type": "object", "properties": LN, "required": ["layer", "n"]}},
         "texts": {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "object", "properties": {
             "label": {"type": "string"}, "text": {"type": "string"}}, "required": ["text"]}]}},
         "chat": {"type": "boolean", "default": False}}, "required": ["neurons", "texts"]}},
    {"name": "logprobs", "description": "Live: the next word-piece's log-probabilities after a text (chat: true, the default, makes it a user "
     "turn and reads the reply's first piece; `prefix` begins the reply). The top k and each target string's first piece. With "
     "`neurons` or `heads` it reads with and without them in one pass: both, the KL from base to intervened, and the pieces that "
     "gained and lost the most probability. The direct read of what an intervention does to the output.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}, "chat": {"type": "boolean", "default": True},
                                                       "prefix": {"type": "string"}, "k": {"type": "integer", "default": 15},
                                                       "targets": {"type": "array", "items": {"type": "string"}},
                                                       "neurons": PUSHES, "heads": HEADS}, "required": ["text"]}},
    {"name": "top_firing", "description": "Live: the neurons that fire hardest on a text against their usual range: each with its face, "
     "score (activation over its usual extreme there; 1 = as strong as it usually gets), raw activation, the unit where it fires "
     "and its words. Over the whole text, or at one unit (`token`: the first holding it; `at`). `layers` [lo, hi] keeps a range. "
     "Use it to find neurons for a concept: write text about it and read what fires on its key word. Leaves the person's screen alone.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}, "token": {"type": "string"}, "at": AT,
                                                       "limit": {"type": "integer", "default": 30},
                                                       "layers": {"type": "array", "items": {"type": "integer"}},
                                                       "chat": {"type": "boolean", "default": False}}, "required": ["text"]}},
    {"name": "steer_lens", "description": "Live: read one neuron's steer lens now (when the steer-lens index is not built, or the neuron is "
     "edited): ±0.8 of the stream's median length along its write direction at every word-piece of four contexts, the next "
     "piece's mean change of log-probability (its 12 biggest rises each way) and KL.",
     "inputSchema": {"type": "object", "properties": LN, "required": ["layer", "n"]}},
    {"name": "causal_check", "description": "Live: the instrument's own causal test of a face's label (the study's N1A protocol): the "
     "neuron's activation written k times its scale on the face's side at the last word-piece of a prompt, and how the model's own "
     "odds of the label's words move against other words'. Slower than logprobs; the record the instrument keeps.",
     "inputSchema": {"type": "object", "properties": {**LN, "face": {"type": "string", "enum": ["+", "-"], "default": "+"},
                                                       "prompt": {"type": "string"}, "chat": {"type": "boolean", "default": False},
                                                       "k": {"type": "number", "default": 10}}, "required": ["layer", "n", "prompt"]}},
    {"name": "attention", "description": "Live: one head's attention on a text. At the reading unit (the first holding `token`; `at`; else the "
     "last) the weight on every unit up to it, and the strongest single reads anywhere in the text. Softmax weights; sliding "
     "layers see only a window back; first_unit_weight is the mean weight on the first unit (the start mark, the attention sink).",
     "inputSchema": {"type": "object", "properties": {**HEAD, "text": {"type": "string"}, "token": {"type": "string"}, "at": AT,
                                                       "chat": {"type": "boolean", "default": False}}, "required": ["layer", "head", "text"]}},
    {"name": "attn_drive", "description": "Live: which attention heads (and feed-forward blocks) drive a neuron at one unit of a text (the "
     "first holding `token`; `at`; else where it fires hardest), and which earlier units each top head read that from. The "
     "activation gelu(gate) up is linearised in the stream entering its block; the parts (heads, blocks, the embedding) add up "
     "to linearised_total; `share` is a head's fraction of it; `from` splits a head's part over the units it read.",
     "inputSchema": {"type": "object", "properties": {**LN, "text": {"type": "string"}, "token": {"type": "string"}, "at": AT,
                                                       "chat": {"type": "boolean", "default": False}, "top": {"type": "integer", "default": 8}},
                     "required": ["layer", "n", "text"]}},
    {"name": "head", "description": "One attention head from its weights, by role: q / k / v are the earlier neurons whose writes land in "
     "its query (what it looks for from the reading unit), key (what makes an earlier unit findable) and value (what it carries); o "
     "the neurons at its layer or later reading its write. Each role's top neurons with their words, and their words pooled. "
     "`profile`: where it looks over eight reference texts (the mean share on self / previous / 2-5 back / further / the first "
     "unit; the units it reads most against chance; its strongest single reads).",
     "inputSchema": {"type": "object", "properties": HEAD, "required": ["layer", "head"]}},
    {"name": "neuron_heads", "description": "The attention heads a neuron meets, from the weights, by role: q / k / v are later heads whose "
     "query, key or value reads its write; o are heads at its layer or before whose write it reads. z per head over every head of "
     "the role; heads sharing one key-value space are listed together.",
     "inputSchema": {"type": "object", "properties": LN, "required": ["layer", "n"]}},
    {"name": "select_neuron", "description": "Open a neuron on the person's screen: its panel, and its web on the map.",
     "inputSchema": {"type": "object", "properties": LN, "required": ["layer", "n"]}},
    {"name": "select_head", "description": "Open an attention head on the person's screen: its q / k / v / o panel, and its links on the map.",
     "inputSchema": {"type": "object", "properties": HEAD, "required": ["layer", "head"]}},
    {"name": "open_tab", "description": "Switch the open neuron's panel to a tab: census (its census answers), push (the person's own "
     "three-voice push chat) or details (words, wiring, heads).",
     "inputSchema": {"type": "object", "properties": {"tab": {"type": "string", "enum": ["census", "push", "details"]}}, "required": ["tab"]}},
    {"name": "camera", "description": "Move the person's 3D view: 'home' (the whole model), 'layer' (one layer's sheet face-on; give layer), "
     "'neuron' (close to one neuron; give layer and n; closeness 0.1 very close, 0.5 wide).",
     "inputSchema": {"type": "object", "properties": {"mode": {"type": "string", "enum": ["home", "layer", "neuron"]}, **LN,
                                                       "closeness": {"type": "number", "default": 0.22}}, "required": ["mode"]}},
    {"name": "highlight_neurons", "description": "Mark a set of neurons on the map (replacing the marks before; an empty list clears them). "
     "face '-' draws a neuron in the negative face's colour.",
     "inputSchema": {"type": "object", "properties": {"neurons": {"type": "array", "items": {"type": "object", "properties": {
         **LN, "face": {"type": "string", "enum": ["+", "-"]}}, "required": ["layer", "n"]}}}, "required": ["neurons"]}},
    {"name": "annotate_neurons", "description": "Pin short labels to neurons on the map (they follow the camera), replacing the ones "
     "before; an empty list clears them. A few words each.",
     "inputSchema": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {
         **LN, "text": {"type": "string"}}, "required": ["layer", "n", "text"]}}}, "required": ["items"]}},
    {"name": "show_text", "description": "Put a text in the atlas's run box and run it, so the person sees its units; with `token`, pin the "
     "first unit holding it, showing the neurons firing hardest there and the links among them on the map.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}, "token": {"type": "string"}}, "required": ["text"]}},
    {"name": "show_search", "description": "Run a lens search in the atlas's left panel, so the person sees the matching neurons listed.",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
]
UI_TOOLS = {"select_neuron": "select", "select_head": "head", "open_tab": "tab", "camera": "camera", "highlight_neurons": "highlight",
            "annotate_neurons": "annotate", "show_text": "text", "show_search": "search"}
NAMES = [t["name"] for t in TOOLS]


def openai_tools() -> List[Dict[str, Any]]:
    """The tools as an OpenAI-compatible endpoint's `tools` (functions with JSON-schema parameters)."""
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["inputSchema"]}}
            for t in TOOLS]


# ------------------------------------------------------------------------------------------------------- the calls


def _need(a: Dict[str, Any], *keys: str) -> None:
    missing = [k for k in keys if a.get(k) is None]
    if missing:
        raise ToolError(f"This tool needs {', '.join(missing)}.")


def _slim_link(c: Dict[str, Any]) -> Dict[str, Any]:
    out = {"neuron": cite(c["layer"], c["n"]), "z": c["z"], "sign": c["sign"], "face": c.get("face"), **face_words(c.get("words"), 5)}
    if c.get("census"):
        out["census"] = c["census"]
    if c.get("name"):
        out["name"] = c["name"]
    return out


async def call(name: str, args: Optional[Dict[str, Any]], api: Any, chat: str = "") -> Any:
    """Run one tool: its arguments in the pages' count, its reply in the pages' count (see the module's docstring)."""
    if name not in NAMES:
        raise ToolError(f"There is no tool {name!r}.")
    a = from_page(dict(args or {}))
    if name == "overview":
        r = await api.get("/api/atlas/arch")
        notes = await api.get("/api/atlas/notes")
        layers = r["layers"]
        full = [i + 1 for i, s in enumerate(layers) if s["kind"] == "full"]
        sliding = [s for s in layers if s["kind"] == "sliding"]
        lens = r.get("lens") or {}
        return {
            "model": r["model"], "stand_in": r["stand_in"], "layers": r["n_layers"], "neurons_per_layer": r["d_ff"], "d_model": r["d_model"],
            "counting": "layers from 1 as on the pages (the HTTP API counts from 0); neurons and heads from 0",
            "attention": {"full_layers": full, "sliding_window": sliding[0]["window"] if sliding else None,
                          "heads": {"sliding": sliding[0]["n_heads"] if sliding else None,
                                    "full": layers[full[0] - 1]["n_heads"] if full else None},
                          "full_layers_read_values_from_keys": bool(full) and bool(layers[full[0] - 1]["values_from_keys"])},
            "indexes": {k: v.get("built", False) for k, v in r["indexes"].items() if k != "folder"},
            "census_layers": [int(k) + 1 for k in r["census"]],
            "push": r["push"],
            "label_gate": {"in_page_layers": f"a lens label is held back below page layer {int(r['label_gate']['min_layer']) + 1}; "
                                             "the card below counts layers from 0", "card": r["label_gate"]},
            "lens": {"id": lens.get("id"), "name": lens.get("name"), "fitted_on": lens.get("fitted_on"), "applied_to": lens.get("applied_to"),
                     "caution": lens.get("caution")},
            "named": {"neurons": len(notes.get("neurons", {})), "heads": len(notes.get("heads", {}))},
        }
    if name == "neuron":
        _need(a, "layer", "n")
        r = await api.get("/api/atlas/neuron", layer=a["layer"], n=a["n"])
        notes = await api.get("/api/atlas/notes")
        entry = notes.get("neurons", {}).get(f"{a['layer']}:{a['n']}") or {}
        faces = {}
        for s in ("+", "-"):
            f = r["faces"][s]
            faces[s] = {"lens": f["lens"], "eye": f["eye"], "label": label(f["check"]), "shared_top10": (f["check"] or {}).get("shared_top10")}
        return {
            "neuron": cite(a["layer"], a["n"]), "name": r.get("name") or None, "note": entry.get("note") or None,
            "edited_in_vault": r.get("edited"), "faces": faces, "gate_trigger": r.get("gate_trigger"),
            "steer": r.get("steer"), "activation": r.get("act_range"),
            "census": r.get("census"), "wiring": r.get("wiring_source"),
            "writes_into": [_slim_link(c) for c in r["writes_into"]], "reads_from": [_slim_link(c) for c in r["reads_from"]],
            "lens": (r.get("lens") or {}).get("name"),
        }
    if name == "rollouts":
        _need(a, "layer", "n")
        u = await api.get("/api/atlas/census/unit", layer=a["layer"], n=a["n"])
        cut = max(40, int(a.get("max_chars", 500)))
        for key in ("rollouts", "baselines"):
            for row in u.get(key, []):
                row["text"] = row["text"][:cut]
        return to_page(u)
    if name == "cone":
        _need(a, "layer", "n")
        r = await api.get("/api/atlas/cone", layer=a["layer"], n=a["n"], k=a.get("k", 2), limit=a.get("limit", 120))
        if not r.get("ready"):
            return r
        nodes = [{"neuron": cite(x["layer"], x["n"]), "hop": x["hop"], "z": x["z"], "face": x["face"], **face_words(x.get("words"), 5),
                  **({"census": x["census"]} if x.get("census") else {}), **({"name": x["name"]} if x.get("name") else {})} for x in r["nodes"]]
        return {"links": r["links"], "neurons": r["neurons"], "per_hop": r["per_hop"], "note": r["note"], "nodes": nodes}
    if name == "web_summary":
        _need(a, "layer", "n")
        r = await api.get("/api/atlas/web/summary", layer=a["layer"], n=a["n"], first=a.get("first", 5), k=a.get("k", 2), top=20)
        if not r.get("ready"):
            return {"ready": False, "note": "the wiring index is not built: the web needs every neuron's links"}
        slim = lambda lst: [f"{t['tok']} ({t['n']})" for t in lst]  # noqa: E731
        return {"pooled": r["src"], "links": r["links"], "left_out_by_the_gate": r.get("gated"),
                "downstream_neurons": r["down"]["neurons"], "toward": slim(r["down"]["toward"]), "away": slim(r["down"]["away"]),
                "upstream_neurons": r["up"]["neurons"], "fed_by": slim(r["up"]["fed_by"]), "held_by": slim(r["up"]["held_by"])}
    if name == "search_lens":
        _need(a, "query")
        r = await api.get("/api/atlas/search", q=a["query"], limit=a.get("limit", 40))
        if not r.get("ready"):
            return {"ready": False, "note": "neither the lens index nor the steer-lens index is built"}
        return {"word_pieces_matched": r["tokens"], "total": r["total"],
                "results": [{"neuron": cite(x["layer"], x["n"]), "face": x["sign"], "matched_in": x["src"], "rank": x["rank"],
                             **face_words(x["words"], 6), **({"census": x["census"]} if x.get("census") else {}),
                             **({"name": x["name"]} if x.get("name") else {})} for x in r["results"]]}
    if name == "top_units":
        _need(a, "layer")
        us = [u for u in await api.get("/api/atlas/census/units", layer=a["layer"]) if u["kind"] == "neuron"]
        us.sort(key=lambda u: -u["z"])
        return [{"neuron": cite(a["layer"], u["n"]), "mass": u["z"], "null_pct": u["null_pct"], "words": u["words"],
                 **({"name": u["name"]} if u.get("name") else {})} for u in us[: int(a.get("limit", 30))]]
    if name in ("set_name", "write_note"):
        _need(a, "layer", "n")
        body: Dict[str, Any] = {"kind": "neuron", "layer": a["layer"], "index": a["n"], "by": "scientist"}
        if name == "set_name":
            _need(a, "name")
            body["name"] = str(a["name"])
        else:
            _need(a, "note")
            body["note"] = str(a["note"])
        r = await api.post("/api/atlas/notes", body)
        entry = r.get("entry") or {}
        if name == "set_name":
            await api.post("/api/atlas/scientist/ui", {"chat": chat, "action": "rename", "layer": a["layer"], "n": a["n"],
                                                       "name": entry.get("name", "")})
        return {"ok": True, "neuron": cite(a["layer"], a["n"]), "name": entry.get("name") or None, "note": entry.get("note") or None}
    if name == "names":
        notes = await api.get("/api/atlas/notes")
        out_n = {cite(*map(int, k.split(":"))): e["name"] for k, e in notes.get("neurons", {}).items() if e.get("name")}
        out_h = {cite_head(*map(int, k.split(":"))): e["name"] for k, e in notes.get("heads", {}).items() if e.get("name")}
        return {"neurons": out_n, "heads": out_h}
    if name == "steer":
        _need(a, "prompt")
        if not a.get("neurons"):
            _need(a, "layer", "n")
        body = {k: a[k] for k in ("layer", "n", "neurons", "prompt", "dose", "mode", "max_new") if a.get(k) is not None}
        r = await api.post("/api/atlas/steer", body)
        return {"prompt": r["prompt"], "pushed": [_cite_push(p) for p in r["pushed"]], "up": r["up"], "baseline": r["baseline"], "down": r["down"]}
    if name == "activations":
        _need(a, "layer", "n", "text")
        body = {k: a[k] for k in ("layer", "n", "text", "chat", "neurons", "heads") if a.get(k) is not None}
        r = await api.post("/api/atlas/reads", body)
        units = [[u["text"], round(u["act"], 3)] for u in r["units"]]
        return {"tokens": units, "strongest": sorted(units, key=lambda u: -abs(u[1]))[:10], "act_range": r.get("act_range"),
                "pushed": [_cite_push(p) for p in r.get("pushed", [])]}
    if name == "compare":
        _need(a, "neurons", "texts")
        r = await api.post("/api/atlas/compare", {k: a[k] for k in ("neurons", "texts", "chat") if a.get(k) is not None})
        return {"texts": r["texts"], "scale_source": (r.get("scale_source") or {}).get("corpus"),
                "neurons": [{"neuron": cite(x["layer"], x["n"]), "act_range": x["act_range"], "by_text": x["by_text"]} for x in r["neurons"]]}
    if name == "logprobs":
        _need(a, "text")
        r = await api.post("/api/atlas/logprobs", {k: a[k] for k in ("text", "chat", "prefix", "k", "targets", "neurons", "heads") if a.get(k) is not None})
        if "pushed" in r:
            r["pushed"] = [_cite_push(p) for p in r["pushed"]]
        return r
    if name == "top_firing":
        _need(a, "text")
        r = await api.post("/api/atlas/fires", {k: a[k] for k in ("text", "token", "at", "limit", "layers", "chat") if a.get(k) is not None})
        return {"tokens": r["tokens"], "at": r["at"], "note": r["note"],
                "neurons": [{"neuron": cite(x["layer"], x["n"]), "face": x["face"], "score": x["score"], "act": x["act"], "token": x["token"],
                             **face_words(x.get("words"), 6)} for x in r["results"]]}
    if name == "steer_lens":
        _need(a, "layer", "n")
        r = await api.post(f"/api/atlas/neuron/steer_lens?layer={int(a['layer'])}&n={int(a['n'])}", {})
        return {"neuron": cite(a["layer"], a["n"]), **{k: r[k] for k in ("up", "down", "kl_up", "kl_down", "dose")}}
    if name == "causal_check":
        _need(a, "layer", "n", "prompt")
        body = {"layer": a["layer"], "neuron_idx": a["n"], "polarity": -1.0 if a.get("face") == "-" else 1.0, "contexts": "given",
                "prompt": a["prompt"], "chat": bool(a.get("chat", False)), "k": float(a.get("k", 10))}
        return to_page(await api.post("/api/scientist/causal_check", body))
    if name == "attention":
        _need(a, "layer", "head", "text")
        r = await api.post("/api/atlas/attention", {k: a[k] for k in ("layer", "head", "text", "token", "at", "chat") if a.get(k) is not None})
        return to_page({"head": cite_head(a["layer"], a["head"]), **{k: v for k, v in r.items() if k not in ("layer", "head")}})
    if name == "attn_drive":
        _need(a, "layer", "n", "text")
        r = await api.post("/api/atlas/attn_drive", {k: a[k] for k in ("layer", "n", "text", "token", "at", "chat", "top") if a.get(k) is not None})
        heads = [{"head": cite_head(h["layer"], h["head"]), **{k: v for k, v in h.items() if k not in ("layer", "head")}} for h in r["heads"]]
        mlps = [{"block": f"L{m['layer'] + 1} feed-forward", "contribution": m["contribution"]} for m in r["mlps"]]
        rest = {k: v for k, v in r.items() if k not in ("layer", "n", "heads", "mlps")}
        return {"neuron": cite(a["layer"], a["n"]), **rest, "heads": heads, "blocks": mlps}
    if name == "head":
        _need(a, "layer", "head")
        r = await api.get("/api/atlas/head", layer=a["layer"], head=a["head"])

        def role(key: str) -> Dict[str, Any]:
            pooled = r.get(f"{key}_pooled") or {}
            return {"neurons": [{"neuron": cite(c["layer"], c["n"]), "z": c["z"], **face_words(c.get("words"), 5)} for c in r[key][:12]],
                    "words": [f"{t['tok']} ({t['n']})" for t in pooled.get("tokens", [])[:16]], "pooled": pooled.get("src")}

        out = {"head": cite_head(a["layer"], a["head"]), "kind": r["kind"], "kv_head": r["kv_head"],
               "values_from_keys": r["values_from_keys"], "q": role("q"), "k": role("k"), "v": role("v"), "o": role("o")}
        try:
            p = await api.get("/api/atlas/head/profile", layer=a["layer"], head=a["head"])
            if "dist" in p:
                out["profile"] = {"dist": dict(zip(p["dist_names"], p["dist"])), "src": p["src"],
                                  "pairs": [{"query": x["qt"], "source": x["kt"], "weight": x["w"], "text": x["text"][:160]} for x in p["pairs"]]}
        except ToolError:
            pass
        return out
    if name == "neuron_heads":
        _need(a, "layer", "n")
        r = await api.get("/api/atlas/neuron/heads", layer=a["layer"], n=a["n"])
        return {role: [{"head": cite_head(x["layer"], x["heads"][0]) + (f"-{x['heads'][-1]}" if len(x["heads"]) > 1 else ""), "z": x["z"]}
                       for x in rows] for role, rows in r["roles"].items()}
    # the person's screen
    action = UI_TOOLS[name]
    if name in ("select_neuron",):
        _need(a, "layer", "n")
    if name == "select_head":
        _need(a, "layer", "head")
    if name == "camera":
        _need(a, "mode")
        if a["mode"] in ("layer", "neuron"):
            _need(a, "layer")
        if a["mode"] == "neuron":
            _need(a, "n")
    if name == "highlight_neurons":
        _need(a, "neurons")
    if name == "annotate_neurons":
        _need(a, "items")
    if name in ("show_text",):
        _need(a, "text")
    if name == "show_search":
        _need(a, "query")
    if name == "open_tab":
        _need(a, "tab")
    r = await api.post("/api/atlas/scientist/ui", {"chat": chat, "action": action, **a})
    return {"ok": bool(r.get("ok")), "note": r.get("note") or r.get("error")}


def _cite_push(p: Dict[str, Any]) -> Dict[str, Any]:
    if "head" in p:
        return {"head": f"L{p['layer'] + 1} " + ("all heads" if p["head"] == "all" else f"h{p['head']}"), "scale": p.get("scale")}
    return {"neuron": f"L{p['layer'] + 1} " + ("feed-forward (all)" if p.get("n") is None else f"N{p['n']}"), "mode": p.get("mode"), "dose": p.get("dose")}


def as_text(result: Any, cut: Optional[int] = None) -> str:
    """A tool's reply as the text the model reads (JSON), cut to `cut` characters with a note saying so."""
    text = json.dumps(result, ensure_ascii=False)
    if cut is not None and len(text) > cut:
        text = text[:cut] + f" … (cut: {len(text) - cut} more characters; ask for less, with limit or max_chars)"
    return text

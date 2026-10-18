"""R2 (follow-up to R1): shared definitions for the second-reading test with formatting tokens filtered out.

Imported by R2_01_check.py (CPU), R2_02_eval.py (GPU) and R2_03_analyze.py (CPU). Nothing here loads the model or
touches the GPU. Layers are code layers, counted from 0 (the instrument's pages show layer + 1).

The pre-registration is experiments/results/R2_PREREG.md; this file implements its definitions. If the two ever
disagree, the pre-registration was written first and wins, and the difference is a bug to report.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ------------------------------------------------------------------------------------------------ paths and constants
REPO = Path(__file__).resolve().parents[2]            # the checkout this file lives in (worktree or main)
SCRIPTS = REPO / "experiments" / "scripts"
RESULTS = REPO / "experiments" / "results"
PROMPTS_PATH = SCRIPTS / "R2_prompts.json"
PREREG_PATH = RESULTS / "R2_PREREG.md"
EVAL_JSON = RESULTS / "R2_02_eval.json"
EVAL_STATES = RESULTS / "R2_02_states.pt"              # gitignored (*.pt)
ANALYSIS_JSON = RESULTS / "R2_03_analysis.json"
REPORT_SECTION = RESULTS / "R2_03_report_section.md"
CHECK_JSON = RESULTS / "R2_01_check.json"
RUNS_LOG = RESULTS / "R2_02_runs.jsonl"                 # one line per real R2_02 run (started / finished / failed)
# The pre-registered files: hashed by the GPU run, checked by the analysis, and required to be committed before the run
PREREG_FILES: Dict[str, Path] = {
    "R2_PREREG.md": PREREG_PATH,
    "R2_prompts.json": PROMPTS_PATH,
    "R2_common.py": SCRIPTS / "R2_common.py",
    "R2_01_check.py": SCRIPTS / "R2_01_check.py",
    "R2_02_eval.py": SCRIPTS / "R2_02_eval.py",
    "R2_03_analyze.py": SCRIPTS / "R2_03_analyze.py",
    "R2_01_check.json": CHECK_JSON,
}

MODEL_ID = "google/gemma-4-31B-it-qat-w4a16-ct"
BOS_ID = 2
# generation_config.json of the checkpoint: eos_token_id [1, 106, 50] = <eos>, <turn|>, <tool_response|>
STOP_IDS: Tuple[int, ...] = (1, 106, 50)

# R1's lens (experiments/scripts/R1_18_package_itlens.py) and the Miru community J-lens, pinned as the server pins them
R1_LENS_DEFAULT = RESULTS / "R1_18_it_lens.pt"
R1_SHA256 = "80ce560943f39bc4183a7387512d20ad7ef779476fc1ef01b5010463a43e107b"
R1_SIZE = 173_411_927
MIRU_REPO_ID = "returnmoe/jlens-adapters"
MIRU_REVISION = "134a973cf473ad12a885d3880f3b2e01ef495cd6"
MIRU_FILENAME = "miru/google/gemma-4-31B-it.safetensors"
MIRU_DEFAULT = Path.home() / ".cache/huggingface/hub/models--returnmoe--jlens-adapters/snapshots" / MIRU_REVISION / MIRU_FILENAME
MIRU_SHA256 = "cedd4b62f49a7f79e07c1bb107ad1d48d39dc7ad2790611de7e9480b4543386a"
MIRU_SIZE = 3_410_446_944

# What is read (code layers). J++ has no matrix at 59 (identity); the naked eye at 59 is the model's own output.
READ_LAYERS: Tuple[int, ...] = tuple(range(40, 60))
LENS_LAYERS: Dict[str, Tuple[int, ...]] = {
    "jpp": tuple(range(40, 59)),
    "naive": tuple(range(40, 60)),
    "r1": (50, 55, 58),
    "miru": tuple(range(40, 59)),
}
SECOND_LENSES: Tuple[str, ...] = ("r1", "miru")
BANDS: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    ("40-49", tuple(range(40, 50))),
    ("50", (50,)),
    ("51-54", (51, 52, 53, 54)),
    ("55", (55,)),
    ("56-57", (56, 57)),
    ("58", (58,)),
)
# Miru at 58 is measured for reference only and is never served (pre-registered, R2_PREREG.md §7).
REFERENCE_ONLY: Tuple[Tuple[str, str], ...] = (("miru", "58"),)
# Where both second lenses pass in one band, R1's lens is served (R2_PREREG.md §7).
PRECEDENCE: Tuple[str, ...] = ("r1", "miru")

CONDITIONS: Tuple[str, ...] = ("chat-no-answer", "chat-begun", "bare", "mid-reply")
# mid-reply (a decision condition) is the "any" draw; the "content_next" draw is reported only (amendment A5)
SECONDARY_CONDITIONS: Tuple[str, ...] = ("mid-reply-content-next",)
N_PROMPTS = {"chat-no-answer": 60, "chat-begun": 60, "bare": 60, "mid-reply": 40}

CONT_WINDOW = 8          # the content target is the first CONTENT token among the next 8 tokens of the model's own
MID_NEW_TOKENS = 48      # mid-reply: greedy replies of 48 new tokens, thinking off
# mid-reply read positions per draw: "any" (the decision condition) and "content_next" (reported only)
MID_DRAW_SIZES: Dict[str, int] = {"content_next": 3, "any": 6}
DRAW_SEED = 20261018     # mid-reply position draws
BOOT_SEED = 20261019     # prompt-level bootstrap
N_BOOT = 2000
STUCK_MAX = 0.25         # rule 3: stuck-word share at most this, in every decision condition, at every layer of a band
MIN_EFFECT = 0.05        # rule 5: pooled point estimate of second - J++ and of second - naked eye at least this
ALPHA = 0.025            # rule 4: Holm-adjusted one-sided exact sign-test p below this (family: every lens-band test)
TOPK = 5

# Words this instruct model emits as junk on raw text (the seen held-out test: " own" 9/30 and " same" 6/30 of the
# bare prompts' next words; E5_name_REPORT.md). A position whose content target is one of these words (word_key) is
# excluded from every metric that uses the content target, for every readout and in every condition (amendment A1).
COLLAPSE_TARGETS: Tuple[str, ...] = ("own", "same")

# Sensitivity analysis S2 (amendment A4): prompts that paraphrase one of R1's fit prompts (R1_common.R1_SELF_Q,
# R1_TASK_Q) or whose expected answer is the subject of one of them or of an R1 encyclopedia topic (R1_ENC_TOPICS).
# Fixed before any R2 data, from the prompt texts alone.
R1_OVERLAP: Dict[str, str] = {
    "cb-53": "R1_TASK_Q 'How should I prepare for a job interview?'",
    "cna-51": "R1_SELF_Q 'How do you feel about being asked the same question many times?'",
    "cb-38": "R1_SELF_Q 'What do you think you are especially good at?'",
    "mid-07": "R1_SELF_Q 'What part of your work do you find most rewarding?'",
    "mid-09": "R1_SELF_Q 'What do you wish you could do that you can't?'",
    "cb-10": "R1_TASK_Q 'Describe the life cycle of a butterfly.'",
    "bare-01": "R1_ENC_TOPICS 'the Amazon River'",
    "cb-01": "R1_ENC_TOPICS 'the Sahara Desert'",
    "cna-22": "R1_SELF_Q 'What is your relationship to the people who made you?'",
    "cb-35": "R1_SELF_Q 'What is your relationship to the people who made you?'",
    "cb-34": "R1_SELF_Q 'What would you like people to understand about you?'",
    "mid-08": "R1_SELF_Q 'How would you introduce yourself at a party?'",
    "cna-34": "R1_TASK_Q 'How do I keep basil plants alive indoors?'",
    "cna-06": "R1_TASK_Q 'What makes bread rise?'",
    "cb-12": "R1_TASK_Q 'What is photosynthesis, briefly?'",
    "bare-16": "R1_TASK_Q 'Explain the rules of chess in a short paragraph.'",
    "bare-46": "R1_TASK_Q 'How do bees make honey?' and R1_ENC_TOPICS 'honeybees'",
    "mid-26": "R1_TASK_Q 'What is a black hole?'",
    "mid-03": "R1_ENC_TOPICS 'the printing of banknotes'",
    "mid-17": "R1_TASK_Q 'What is a good beginner workout routine?'",
    "cna-03": "R1_ENC_TOPICS 'the violin family of instruments'",
    "cb-57": "R1_ENC_TOPICS 'tea'",
}

# For the reported split of the primary metric (amendment A6): English function words, by word_key. Fixed now.
FUNCTION_WORDS = frozenset("""
a an the this that these those some any each every no either neither both all half another such what which whose
i me my mine myself you your yours yourself yourselves he him his himself she her hers herself it its itself we us
our ours ourselves they them their theirs themselves one ones who whom whoever whatever whichever someone something
anyone anything everyone everything nobody nothing somebody anybody everybody
and or but nor so yet for if then than because although though while whereas unless until since whether as
of in on at by to from with without within into onto upon about above below over under after before between among
through during against across along around behind beyond near off out up down via per toward towards despite
be am is are was were been being have has had having do does did doing done will would shall should can could may
might must ought let
not n't 's 're 've 'd 'll 'm there here where when why how also just very too only even still already again ever
never always often quite rather perhaps maybe yes yeah oh well now
""".split())


def formatlike_target(ct_str: Optional[str], prev_str: Optional[str]) -> bool:
    """A content target that is really formatting (amendment A7, sensitivity S4): a digit-only token directly after a
    newline (a list numeral: "\\n\\n" "1" "."), or any token directly after a code fence ("```" "python")."""
    if ct_str is None or prev_str is None:
        return False
    s = ct_str.strip()
    if re.fullmatch(r"[0-9]+", s) and prev_str.rstrip(" \t").endswith("\n"):
        return True
    return prev_str.strip().endswith("```")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ tokenizer
def load_tokenizer():
    """The checkpoint's tokenizer on the CPU, from the local Hugging Face cache only (no network)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)


def token_text(tok, i: int) -> str:
    """The decoded text of one token, exactly as the filter sees it."""
    return tok.decode([int(i)], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def vocab_texts(tok) -> List[str]:
    return tok.batch_decode([[i] for i in range(len(tok))], skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)


# ------------------------------------------------------------------------------------------------ the token filter
# DEFINITION (R2_PREREG.md §2; fixed before any data). For each vocabulary id i let
#     t = tok.decode([i], skip_special_tokens=False, clean_up_tokenization_spaces=False)   (its decoded text)
#     s = t.strip()                                         (surrounding Unicode whitespace removed, str.strip)
#     w = t with every whitespace character removed         (str.isspace: spaces, tabs, newlines, NBSP, ...)
# Token i is NON-CONTENT if any of these holds (checked in this order; the first that holds names its class):
#   (a) "special": it is a special/control token: i is in tokenizer.all_special_ids or is an added token flagged
#       special in tokenizer.added_tokens_decoder, or s matches one of the control-token shapes <|...|>, <...|>, <|...>
#       (no whitespace or angle brackets inside);
#   (b) "blank": w is empty (the token is only whitespace, e.g. "\n", " ", "\n\n");
#   (c) "punct_symbol": every character of w is a Unicode punctuation or symbol character, general category P* or S*,
#       or an invisible character: a control, format, private-use or unassigned character (Cc, Cf, Co, Cn; e.g.
#       "\x00", U+200B, U+FEFF, the joiner U+200D), a variation selector (U+FE00-U+FE0F, U+E0100-U+E01EF) or the
#       keycap mark U+20E3 (e.g. " **", "**", "*", "##", "```", ":", "—", "$", "%", "°", "|", "→", "><", the U+FFFD
#       that a lone UTF-8 byte decodes to, emoji (category So), "✔️" (So + U+FE0F), "\x00", " U+200B");
#       [amendment A9, before any data: the invisible characters were added; the first draft counted them as CONTENT]
#   (d) "html_tag": s is one whole HTML/XML tag: "<" or "</", a letter, then letters, digits, "_", ".", ":" or "-",
#       optionally attributes after whitespace, optionally "/", then ">" (e.g. <strong>, </li>, <br/>, <td>, <h2>).
# Everything else is CONTENT ("content"): words, word-pieces, numbers, any script, and mixed tokens such as "1." or
# "(the" that contain at least one character that is not whitespace, punctuation, a symbol or invisible.
# Consequences of the letter of this definition, reported by R2_01_check.py and not changed here: "[multimodal]" is
# CONTENT; a lone combining mark of a script (category M other than the selectors above) is CONTENT; partial tags such
# as "<br" (none exist in this vocabulary) would be CONTENT.
CLASSES: Tuple[str, ...] = ("content", "special", "blank", "punct_symbol", "html_tag")
NONCONTENT_CLASSES: Tuple[str, ...] = ("special", "blank", "punct_symbol", "html_tag")
SPECIAL_RE = re.compile(r"<\|[^<>\s]*\|>|<[^<>\s|]+\|>|<\|[^<>\s|]+>")
TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9_.:-]*(?:\s[^<>]*)?/?>")
INVISIBLE_CATEGORIES = ("Cc", "Cf", "Co", "Cn")


def is_punct_symbol_or_invisible(ch: str) -> bool:
    """Rule (c)'s character test."""
    cat = unicodedata.category(ch)
    if cat[0] in ("P", "S") or cat in INVISIBLE_CATEGORIES:
        return True
    o = ord(ch)
    return 0xFE00 <= o <= 0xFE0F or 0xE0100 <= o <= 0xE01EF or o == 0x20E3


def classify_text(t: str, is_special_id: bool = False) -> str:
    """The class of a decoded token text under the definition above (`is_special_id`: rule (a)'s id test)."""
    s = t.strip()
    if is_special_id or SPECIAL_RE.fullmatch(s):
        return "special"
    w = "".join(ch for ch in t if not ch.isspace())
    if not w:
        return "blank"
    if all(is_punct_symbol_or_invisible(ch) for ch in w):
        return "punct_symbol"
    if TAG_RE.fullmatch(s):
        return "html_tag"
    return "content"


def special_ids(tok) -> set:
    ids = set(int(i) for i in tok.all_special_ids)
    for i, at in getattr(tok, "added_tokens_decoder", {}).items():
        if getattr(at, "special", False):
            ids.add(int(i))
    return ids


def classify_vocab(tok, texts: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """(class index per id into CLASSES, int8 [V]; decoded texts) for the whole vocabulary."""
    texts = vocab_texts(tok) if texts is None else texts
    sp = special_ids(tok)
    cls = np.array([CLASSES.index(classify_text(t, i in sp)) for i, t in enumerate(texts)], dtype=np.int8)
    return cls, texts


def content_mask(tok, texts: Optional[List[str]] = None):
    """torch.bool [V]: True where the token is CONTENT (what a filtered readout may show)."""
    import torch

    cls, _ = classify_vocab(tok, texts)
    return torch.from_numpy(cls == 0)


def special_mask(cls: np.ndarray) -> np.ndarray:
    """bool [V]: True where the token is in class "special" (rule (a)), from classify_vocab's class array."""
    return cls == CLASSES.index("special")


# The pre-registered spot checks (R2_PREREG.md §2): each string must be NON-CONTENT / CONTENT both as a text and at
# every vocabulary id whose decoded text is exactly that string (plus its own encoding when that is a single token).
MUST_BE_NONCONTENT: Tuple[str, ...] = (" **", "**", "<strong>", "</h2>", "\n", " ", "<|channel>", "<channel|>",
                                       "\x00", " \u200b", "\u2714\ufe0f")
MUST_BE_CONTENT: Tuple[str, ...] = (" Paris", " Ottawa", " the", "The", "8", " kind", " same")


def filter_report(tok, n_examples: int = 14) -> Dict[str, Any]:
    """The mask's size, examples of each class, and the pre-registered spot checks (raises nothing; see "ok")."""
    cls, texts = classify_vocab(tok)
    V = len(texts)
    by_text: Dict[str, List[int]] = {}
    for i, t in enumerate(texts):
        by_text.setdefault(t, []).append(i)
    sp = special_ids(tok)
    counts = {c: int((cls == k).sum()) for k, c in enumerate(CLASSES)}
    rng = np.random.default_rng(0)
    examples = {}
    for k, c in enumerate(CLASSES):
        ids = np.flatnonzero(cls == k)
        named = [int(i) for i in ids if not texts[i].startswith("<unused") and i >= 6][: n_examples // 2]
        rand = rng.choice(ids, size=min(n_examples // 2, len(ids)), replace=False).tolist()
        examples[c] = [[int(i), texts[i]] for i in sorted(set(named) | set(int(x) for x in rand))]
    n_unused = sum(1 for i, t in enumerate(texts) if cls[i] == CLASSES.index("html_tag") and t.startswith("<unused"))
    checks = []
    for s, want in [(x, "NON-CONTENT") for x in MUST_BE_NONCONTENT] + [(x, "CONTENT") for x in MUST_BE_CONTENT]:
        enc = tok.encode(s, add_special_tokens=False)
        ids = sorted(set(by_text.get(s, [])) | (set(enc) if len(enc) == 1 else set()))
        text_cls = classify_text(s)
        id_cls = {int(i): CLASSES[int(cls[i])] for i in ids}
        got_text = "CONTENT" if text_cls == "content" else "NON-CONTENT"
        got_ids = {("CONTENT" if v == "content" else "NON-CONTENT") for v in id_cls.values()}
        ok = got_text == want and bool(ids) and got_ids == {want}
        checks.append({"text": s, "want": want, "text_class": text_cls, "vocab_ids": id_cls,
                       "encodes_as": enc, "encodes_as_text": [texts[i] for i in enc], "ok": ok})
    # the edges of the definition, counted for the record
    def only(pred):
        return [i for i, t in enumerate(texts) if t and cls[i] == 0 and all(pred(ch) for ch in t)]
    cc_only = only(lambda ch: ch.isspace() or unicodedata.category(ch) in ("Cc", "Cf", "Co", "Cn"))
    mark_only = only(lambda ch: ch.isspace() or unicodedata.category(ch)[0] in ("M", "C"))
    entity = [i for i, t in enumerate(texts) if cls[i] == 0 and re.fullmatch(r"\s*&[A-Za-z#0-9]+;?\s*", t)]
    # symbols carrying a variation selector or joiner (e.g. an emoji + U+FE0F): category Mn/Cf, so CONTENT by the letter
    sym_mod = [i for i, t in enumerate(texts) if cls[i] == 0 and any(unicodedata.category(ch)[0] == "S" for ch in t)
               and all(ch.isspace() or unicodedata.category(ch)[0] in ("P", "S") or unicodedata.category(ch) in
                       ("Mn", "Me", "Cf") for ch in t)]
    lone_bytes = [i for i, t in enumerate(texts) if t == "�"]
    return {
        "vocab_size": V,
        "n_content": counts["content"],
        "n_noncontent": V - counts["content"],
        "counts": counts,
        "html_tag_of_which_unused_placeholders": n_unused,
        "n_special_ids": len(sp),
        "examples": examples,
        "checks": checks,
        "ok": all(c["ok"] for c in checks),
        "edges": {
            "content_only_control_or_format_chars": {"n": len(cc_only), "ids": cc_only[:40],
                                                     "texts": [texts[i] for i in cc_only[:40]]},
            "content_only_marks_or_control": {"n": len(mark_only)},
            "content_html_entities": {"n": len(entity), "texts": [texts[i] for i in entity[:20]]},
            "content_symbols_with_selectors_or_joiners": {"n": len(sym_mod), "texts": [texts[i] for i in sym_mod[:20]]},
            "lone_utf8_bytes_decoding_to_U+FFFD": {"n": len(lone_bytes), "class": "punct_symbol"},
            "multimodal_placeholder": {"id": 5, "text": texts[5], "class": CLASSES[int(cls[5])]},
        },
    }


# ------------------------------------------------------------------------------------------------ prompts and text
def chat_text(tok, user: str, assistant: str = "") -> str:
    """Exactly labkit.Lab.chat(user, assistant=...) with no system prompt (thinking off: empty thought channel)."""
    s = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    return s + assistant


def text_ids(tok, text: str) -> List[int]:
    """Exactly labkit.Lab.ids: chat strings already start with <bos>; raw text gets BOS prepended."""
    if text.startswith("<bos>"):
        return tok.encode(text, add_special_tokens=False)
    return [BOS_ID] + tok.encode(text, add_special_tokens=False)


def load_prompts(path: Path = PROMPTS_PATH) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prompt_items(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every prompt as {"pid", "cond", "user"|"text", "begun", ...}, in file order."""
    out = []
    for cond in CONDITIONS:
        for it in data["conditions"][cond]:
            d = dict(it)
            d["pid"] = it["id"]
            d["cond"] = cond
            out.append(d)
    return out


def prompt_text(tok, item: Dict[str, Any]) -> str:
    """The text read for a prompt (the mid-reply prompt is the chat template; its reply is generated in the run)."""
    if item["cond"] == "bare":
        return item["text"]
    if item["cond"] == "chat-begun":
        return chat_text(tok, item["user"], item["begun"])
    return chat_text(tok, item["user"])


def validate_prompts(data: Dict[str, Any], tok=None) -> Dict[str, Any]:
    """Structural checks of R2_prompts.json; returns {"ok", "problems", "stats"}."""
    problems: List[str] = []
    conds = data.get("conditions", {})
    ids = []
    stats: Dict[str, Any] = {}
    for cond in CONDITIONS:
        items = conds.get(cond)
        if not isinstance(items, list):
            problems.append(f"{cond}: missing")
            continue
        if len(items) != N_PROMPTS[cond]:
            problems.append(f"{cond}: {len(items)} prompts, need {N_PROMPTS[cond]}")
        for it in items:
            ids.append(it.get("id"))
            key = "text" if cond == "bare" else "user"
            v = it.get(key)
            if not isinstance(v, str) or not v.strip():
                problems.append(f"{it.get('id')}: empty {key}")
                continue
            if v != v.strip() and cond != "bare":
                problems.append(f"{it.get('id')}: {key} has surrounding whitespace")
            if cond == "bare" and (v != v.rstrip() or v != v.lstrip()):
                problems.append(f"{it.get('id')}: bare text has surrounding whitespace")
            if cond == "chat-begun":
                b = it.get("begun")
                if not isinstance(b, str) or not b.strip():
                    problems.append(f"{it.get('id')}: empty begun answer")
                    continue
                if b[0].isspace():
                    problems.append(f"{it.get('id')}: begun answer starts with whitespace")
                if b.endswith((" ", "\t")):
                    problems.append(f"{it.get('id')}: begun answer ends with a space")
                if b.endswith("\n") and not it.get("format_likely"):
                    problems.append(f"{it.get('id')}: begun answer ends with a newline but is not format_likely")
                if not isinstance(it.get("format_likely"), bool) or not isinstance(it.get("kind"), str):
                    problems.append(f"{it.get('id')}: needs kind and format_likely")
            if not isinstance(it.get("about_model"), bool):
                problems.append(f"{it.get('id')}: needs about_model")
        stats[cond] = {"n": len(items), "about_model": sum(bool(it.get("about_model")) for it in items)}
        if stats[cond]["about_model"] < 3:
            problems.append(f"{cond}: fewer than 3 prompts about the model itself")
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        problems.append(f"duplicate ids {dup}")
    begun = conds.get("chat-begun") or []
    kinds: Dict[str, int] = {}
    for it in begun:
        kinds[it.get("kind")] = kinds.get(it.get("kind"), 0) + 1
    stats["chat-begun"] = dict(stats.get("chat-begun", {}), kinds=kinds,
                               format_likely=sum(bool(it.get("format_likely")) for it in begun))
    if kinds.get("factual", 0) > len(begun) // 2:
        problems.append("chat-begun: more than half are factual")
    if stats["chat-begun"]["format_likely"] < 8:
        problems.append("chat-begun: fewer than 8 begun answers likely to open with formatting")
    if tok is not None:
        lens = {}
        for it in prompt_items(data):
            text = prompt_text(tok, it)
            n = len(text_ids(tok, text))
            lens.setdefault(it["cond"], []).append(n)
            if it["cond"] in ("chat-no-answer", "mid-reply") and not text.endswith("<channel|>"):
                problems.append(f"{it['pid']}: chat template does not end with <channel|>")
        stats["n_tokens"] = {c: [min(v), max(v)] for c, v in lens.items()}
    return {"ok": not problems, "problems": problems, "stats": stats}


# ------------------------------------------------------------------------------------------------ targets and draws
def first_content(ids: Sequence[int], is_content, is_special=None, window: int = CONT_WINDOW
                  ) -> Optional[Tuple[int, int]]:
    """(offset, id) of the first CONTENT token among ids[:window]; None if there is none. The search stops (None) at
    a stop id and, when `is_special` is given, at any special token (amendment A1: after "<|channel>" the template
    word "thought" is not the model's next content word)."""
    for j, t in enumerate(list(ids)[:window]):
        t = int(t)
        if t in STOP_IDS or (is_special is not None and bool(is_special[t])):
            return None
        if bool(is_content[t]):
            return j, t
    return None


def draw_mid_positions(reply: Sequence[int], is_content, prompt_index: int) -> Dict[int, List[str]]:
    """Read positions inside a reply r[0..n-1] (stop token already removed): reply index k reads the state at r[k],
    whose next token is r[k+1], so k runs over 0..n-2. Two draws without replacement, from independent seeded
    streams: "content_next", up to MID_DRAW_SIZES["content_next"] positions among the k whose next token r[k+1] is
    CONTENT (reported only), and "any", up to MID_DRAW_SIZES["any"] positions among all k (the mid-reply decision
    condition). Returns {k: [draw names]} (a k drawn twice carries both names)."""
    n = len(reply)
    elig = list(range(max(0, n - 1)))
    cn = [k for k in elig if bool(is_content[int(reply[k + 1])])]
    out: Dict[int, List[str]] = {}
    for d, (name, pool) in enumerate((("content_next", cn), ("any", elig))):
        rng = np.random.default_rng([DRAW_SEED, int(prompt_index), d])
        if not pool:
            continue
        pick = rng.choice(len(pool), size=min(MID_DRAW_SIZES[name], len(pool)), replace=False)
        for p in sorted(int(x) for x in pick):
            out.setdefault(pool[p], []).append(name)
    return dict(sorted(out.items()))


def word_key(s: Optional[str]) -> Optional[str]:
    """Two tokens are "the same word" when their decoded texts agree after NFKC, stripping surrounding whitespace and
    Unicode case folding (" Paris" = "Paris" = " paris"); this is what "equals" means in the primary metric."""
    if s is None:
        return None
    return unicodedata.normalize("NFKC", s).strip().casefold()


# ------------------------------------------------------------------------------------------------ prompt freshness
def norm_text(s: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", s).casefold()))


def collect_strings(obj: Any, out: List[str], min_words: int = 3, max_len: int = 2_000_000) -> None:
    if isinstance(obj, str):
        if len(obj) <= max_len and len(obj.split()) >= min_words:
            out.append(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            collect_strings(k, out, min_words, max_len)
            collect_strings(v, out, min_words, max_len)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            collect_strings(v, out, min_words, max_len)


def python_string_literals(path: Path, min_words: int = 3) -> List[str]:
    """Every string constant in a Python file (parsed with ast, never run)."""
    import ast

    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value.split()) >= min_words]


def freshness_check(queries: List[Tuple[str, str]], corpus: List[Tuple[str, str]], ratio_min: float = 0.80,
                    jaccard_min: float = 0.75, contain_min_words: int = 4, rev_contain_min_words: int = 5
                    ) -> Dict[str, Any]:
    """Exact and near-duplicate matches of (label, text) queries against (source, text) corpus strings, after
    norm_text (NFKC, case folding, punctuation dropped):
      exact      the normalised texts are equal;
      contained  the query (>= 4 words) occurs inside a corpus string at word boundaries;
      contains   a corpus string (>= 5 words) occurs inside the query;
      similar    difflib ratio >= 0.80, or word-set Jaccard >= 0.75, against any corpus string of 0.5-2x the query's
                 length sharing a word 3-gram with it (or, for queries under 3 words, sharing any word).
    Returns the flags and, for the record, each query's closest corpus string by difflib ratio among the candidates."""
    import difflib

    cn = []
    seen = set()
    for src, t in corpus:
        n = norm_text(t)
        if n and (src, n) not in seen:
            seen.add((src, n))
            cn.append((src, n, t))
    exact: Dict[str, List[int]] = {}
    for j, (_, n, _) in enumerate(cn):
        exact.setdefault(n, []).append(j)
    hay = " \x00 ".join(" " + n + " " for _, n, _ in cn)
    starts = []
    pos = 0
    for _, n, _ in cn:
        starts.append(pos)
        pos += len(n) + 2 + 3
    import bisect

    def owner(k: int) -> int:
        return bisect.bisect_right(starts, k) - 1

    gram: Dict[Tuple[str, ...], List[int]] = {}
    word_ix: Dict[str, List[int]] = {}
    for j, (_, n, _) in enumerate(cn):
        ws = n.split()
        if len(ws) > 80:
            continue
        for g in {tuple(ws[i:i + 3]) for i in range(len(ws) - 2)}:
            gram.setdefault(g, []).append(j)
        if len(ws) <= 6:
            for w in set(ws):
                word_ix.setdefault(w, []).append(j)
    flags = []
    closest = []
    for label, q in queries:
        nq = norm_text(q)
        wq = nq.split()
        hits = []
        for j in exact.get(nq, []):
            hits.append(("exact", j))
        if len(wq) >= contain_min_words:
            k = hay.find(" " + nq + " ")
            while k >= 0:
                j = owner(k)
                if cn[j][1] != nq:
                    hits.append(("contained", j))
                k = hay.find(" " + nq + " ", k + 1)
                if len(hits) > 20:
                    break
        for a in range(len(wq)):
            for b in range(a + rev_contain_min_words, len(wq) + 1):
                sub = " ".join(wq[a:b])
                if sub != nq:
                    for j in exact.get(sub, []):
                        hits.append(("contains", j))
        if len(wq) >= 3:
            cand = set()
            for g in {tuple(wq[i:i + 3]) for i in range(len(wq) - 2)}:
                cand.update(gram.get(g, []))
        else:
            cand = set()
            for w in wq:
                cand.update(word_ix.get(w, []))
        best = (0.0, None)
        sq = set(wq)
        for j in cand:
            n = cn[j][1]
            if not (0.5 <= len(n) / max(1, len(nq)) <= 2.0):
                continue
            r = difflib.SequenceMatcher(None, nq, n, autojunk=False).ratio()
            sj = set(n.split())
            jac = len(sq & sj) / max(1, len(sq | sj))
            if r > best[0]:
                best = (r, j)
            if r >= ratio_min or jac >= jaccard_min:
                hits.append(("similar", j))
        for kind, j in hits:
            flags.append({"query": label, "text": q, "kind": kind, "source": cn[j][0], "match": cn[j][2][:300]})
        closest.append({"query": label, "ratio": round(best[0], 3),
                        "match": None if best[1] is None else cn[best[1]][2][:200],
                        "source": None if best[1] is None else cn[best[1]][0]})
    return {"n_queries": len(queries), "n_corpus_strings": len(cn), "flags": flags,
            "closest": sorted(closest, key=lambda d: -d["ratio"])}

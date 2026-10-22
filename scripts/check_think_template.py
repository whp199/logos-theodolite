#!/usr/bin/env python3
"""A CPU-only check of how the instrument switches Gemma 4's thinking on (E6 §3, item 11; the Lexicon's caution 9):
no model, no GPU, no running server; only the tokenizer and chat template in the local Hugging Face cache.

A thinking sighting of a chat is sent through `TheodoliteRuntime.compose_prompt(..., thinking=True)`, which renders
the checkpoint's own chat template with `enable_thinking=True` (server.py, think_protocol "template"). Bare text has
no turn to think in: the server either leaves the comparison out (think_bare "none") or, as first built and for old
callers, sets "<|think|>\\n" before the text (think_bare "prefix", think_protocol "bare_prefix"), which is not the
template's switch. This checks, on the cached template, that:
1. thinking on: the template opens a system turn even with no system prompt, writes "<|think|>\\n" at its very head,
   and begins the model's turn with nothing after "<|turn>model\\n" (no empty thought);
2. thinking off: no system turn without a system prompt, no "<|think|>", and the model's turn begun with the empty
   thought "<|channel>thought\\n<channel|>";
3. with a system prompt, "<|think|>\\n" comes before its words in the same turn; a begun answer follows
   "<|turn>model\\n" directly;
4. "<|think|>" is one special word-piece (id 98), as are the turn and channel marks, so the switch reaches the model
   as one mark and not as text;
5. the old imitation is not the template's switch: its "<|think|>" follows <bos> with no system turn around it.
A template update that changed any of this would change what "thinking turned on" means on every page, so the
expected renders are written out here in full.

Run: .venv/bin/python -I scripts/check_think_template.py   (exits 0 when every check holds)
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

from pathlib import Path
from typing import Any, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.engine_runtime import DEFAULT_W4A16_MODEL_ID, TheodoliteRuntime

failures: List[str] = []
passed = 0


def check(cond: bool, what: str) -> None:
    global passed
    if cond:
        passed += 1
    else:
        failures.append(what)
        print(f"  FAILED: {what}")


# The renders as measured on the cached template (google/gemma-4-31B-it-qat-w4a16-ct, chat_template.jinja), for the
# question "Q?"
THINK_ON = "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nQ?<turn|>\n<|turn>model\n"
THINK_OFF = "<bos><|turn>user\nQ?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
THINK_ON_SYSTEM = "<bos><|turn>system\n<|think|>\nBe brief.<turn|>\n<|turn>user\nQ?<turn|>\n<|turn>model\n"
THINK_OFF_SYSTEM = (
    "<bos><|turn>system\nBe brief.<turn|>\n<|turn>user\nQ?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
)
# the special word-pieces the switch and the turns are made of
SPECIAL = {"<bos>": 2, "<|think|>": 98, "<|channel>": 100, "<channel|>": 101, "<|turn>": 105, "<turn|>": 106}


def load_runtime() -> Optional[TheodoliteRuntime]:
    """A runtime with only the cached tokenizer: `compose_prompt` needs nothing else."""
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(DEFAULT_W4A16_MODEL_ID, local_files_only=True)
    except Exception as exc:
        failures.append(f"the instrument's tokenizer is not in the local Hugging Face cache ({exc})")
        print(f"  FAILED: {failures[-1]}")
        return None
    rt = TheodoliteRuntime(device="cpu", load_jlens=False)
    rt.tokenizer = tok
    return rt


def ids(tok: Any, text: str) -> List[int]:
    return list(tok.encode(text, add_special_tokens=False))


def main() -> int:
    rt = load_runtime()
    if rt is not None:
        tok = rt.tokenizer
        print("1-3. the template's thinking switch")
        on = rt.compose_prompt("Q?", chat=True, thinking=True)
        off = rt.compose_prompt("Q?", chat=True)
        check(on == THINK_ON, f"thinking on renders {on!r}, not {THINK_ON!r}")
        check(off == THINK_OFF, f"thinking off renders {off!r}, not {THINK_OFF!r}")
        check(on.startswith("<bos><|turn>system\n<|think|>\n"), "thinking on: <|think|> at the head of a system turn")
        check(on.endswith("<|turn>model\n") and "<|channel>" not in on, "thinking on: no empty thought")
        check("<|think|>" not in off and "<|turn>system" not in off, "thinking off: no switch and no system turn")
        check(off.endswith("<|turn>model\n<|channel>thought\n<channel|>"), "thinking off: the empty thought")
        on_s = rt.compose_prompt("Q?", chat=True, system="Be brief.", thinking=True)
        off_s = rt.compose_prompt("Q?", chat=True, system="Be brief.")
        check(on_s == THINK_ON_SYSTEM, f"thinking on with a system prompt renders {on_s!r}")
        check(off_s == THINK_OFF_SYSTEM, f"thinking off with a system prompt renders {off_s!r}")
        check(on_s.count("<|turn>system") == 1, "the switch shares the system prompt's turn, not one of its own")
        begun = rt.compose_prompt("Q?", chat=True, assistant_prefix="The", thinking=True)
        check(begun == THINK_ON + "The", f"a begun answer follows the model's turn directly: {begun!r}")

        print("4. the switch is one special word-piece")
        for mark, want in SPECIAL.items():
            got = ids(tok, mark)
            check(got == [want], f"{mark} encodes to {got}, not [{want}]")
            added = tok.added_tokens_decoder.get(want)
            check(added is not None and added.content == mark and added.special, f"{mark} ({want}) is not special")
        on_ids = ids(tok, THINK_ON)
        check(on_ids[:5] == [2, 105, ids(tok, "system")[0], ids(tok, "\n")[0], 98] and on_ids.count(98) == 1,
              f"thinking on: <bos>, <|turn>, system, a new line, then the switch, once: {on_ids[:8]}")
        check(98 not in ids(tok, THINK_OFF), "thinking off: no switch among the word-pieces")

        print("5. the old imitation is not the switch")
        bare = rt.compose_prompt("Q?")
        imitation = f"<|think|>\n{bare}"  # what server._stream_sweep sends for bare text with think_bare "prefix"
        im_ids = [2] + ids(tok, imitation)  # tokenize_prompt prepends the one <bos> bare text lacks
        check(98 not in ids(tok, bare), "bare text carries no switch of its own")
        check(im_ids[:3] == [2, 98, ids(tok, "\n")[0]] and 105 not in im_ids,
              f"the imitation: the switch straight after <bos>, with no turn around it: {im_ids[:6]}")
        check(im_ids != on_ids, "the imitation and the template's switch are not the same word-pieces")
    if failures:
        print(f"\nthinking template: {len(failures)} of {passed + len(failures)} checks FAILED")
        return 1
    print(f"\nthinking template: all {passed} checks hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())

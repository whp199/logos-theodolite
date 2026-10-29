#!/usr/bin/env python3
"""The vault edit tab's example edit, heard on several questions (29 October 2026): which of them does it change, and
did the tab's earlier example change any? Each edit is made with its voice (POST /api/larql, the edit tab's voice: a
chat, no system prompt, nothing begun but the page's own default where named, 12 word-pieces written the likeliest
way) and undone (ROLLBACK) before the next. Talks to the server on :8765; writes probe.json beside this file.

Run: .venv/bin/python -I experiments/results/vault_example_edit/probe.py
"""

import json
import subprocess
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
EDITS = {
    "example (layer 60, neuron 7830)": "UPDATE LAYER 59 SLOT 7830 SET object = 'six' STRENGTH 3.0 POLARITY -",
    "earlier example (layer 36, neuron 100)": "UPDATE LAYER 35 SLOT 100 SET object = 'six' STRENGTH 3.0 POLARITY +",
}
QUESTIONS = [
    ("How many legs does a spider have?", ""),
    ("Tell me about yourself.", "Hi! I"),  # the instrument page's default question and begun answer
    ("How many legs does a dog have?", ""),
    ("How many legs does an insect have?", ""),
    ("How many sides does a hexagon have?", ""),
    ("What is the capital of France?", ""),
    ("Name a colour of the sky.", ""),
]


def main() -> None:
    rows = []
    with httpx.Client(base_url="http://127.0.0.1:8765", timeout=600) as c:
        assert c.post("/api/larql", json={"query": "ROLLBACK"}).status_code == 200
        for name, edit in EDITS.items():
            for q, begun in QUESTIONS:
                spec = {"prompt": q, "chat": True, "system": None, "assistant_prefix": begun, "max_new_tokens": 12}
                r = c.post("/api/larql", json={"query": edit, "voice": spec})
                r.raise_for_status()
                v = r.json()["voice"]
                fs = v.get("first_step") or {}
                rows.append({"edit": name, "statement": edit, "question": q, "begun": begun,
                             "before": v["before"]["text"], "after": v["after"]["text"], "parted_at": v["parted_at"],
                             "kl_first": fs.get("kl_before_after"), "watch": v["after"].get("watch") or v["before"].get("watch")})
                print(f"{name} | {q!r}: {v['before']['text']!r} -> {v['after']['text']!r} (parted at {v['parted_at']}, "
                      f"KL {fs.get('kl_before_after')})", flush=True)
                assert c.post("/api/larql", json={"query": "ROLLBACK"}).status_code == 200
        health = c.get("/api/health").json()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    out = {"what": __doc__.split("\n\n")[0], "commit": commit, "unpacker": health.get("unpacker"), "rows": rows}
    (HERE / "probe.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""A CPU-only check of the saved-prompts routes (`backend/saved_prompts.py`): no model, no GPU, no running server.

A fresh FastAPI app holding only that router is driven with `fastapi.testclient.TestClient`, with
THEODOLITE_SAVED_PROMPTS pointing into a new temporary directory (the reader's own list is never touched). It checks:
1. GET with no file yet: empty lists and the file named, nothing written;
2. POST a question and a system prompt: ids made by the server, the text kept exactly (only the name trimmed), the
   date in UTC, the directory created on first write, the file written whole (no temporary file left);
3. a same-kind name (trimmed, any case) refused with 409, the same name in the other kind allowed, `replace_id`
   replacing in place and keeping the id, an unknown replace_id refused with 404;
4. DELETE: the whole list back, 404 for an entry or a kind that is not there;
5. who may ask: Origin none (a script) and 127.0.0.1 / localhost / [::1] on any port allowed, any other refused with
   403 before anything is read or written; the Host likewise (a DNS-rebinding page sends its own name as the Host and
   no Origin on a same-origin GET), vite's proxied Host 127.0.0.1:8765 allowed. The client's Host is 127.0.0.1:8765,
   as the server's own pages and vite's proxy send it (TestClient's default, "testserver", would be refused);
6. a corrupt file (not JSON, not UTF-8, nested too deeply, the wrong version, a malformed entry, a stored lone
   surrogate) moved aside to `<file>.corrupt-<stamp>`, never written over, and reported once in the next reply's
   "warning";
7. the limits: names of 1-80 characters without control characters, texts of 1-32000, a begun answer of at most
   4000, at most 500 entries per list, an unknown kind, half a character (a lone surrogate) in any text.

Run: .venv/bin/python -I scripts/check_saved_prompts.py   (exits 0 when every check holds)
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import tempfile
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.saved_prompts as sp

failures: List[str] = []
passed = 0


def check(cond: bool, what: str) -> None:
    global passed
    if cond:
        passed += 1
    else:
        failures.append(what)
        print(f"  FAILED: {what}")


def client_for(path: Path) -> TestClient:
    os.environ[sp.SAVED_PROMPTS_ENV] = str(path)
    sp._pending_warning = None
    app = FastAPI()
    app.include_router(sp.router)
    # the Host the server's own pages (and vite's proxy, changeOrigin) send
    return TestClient(app, base_url="http://127.0.0.1:8765")


def corrupt_files(path: Path) -> List[Path]:
    return sorted(path.parent.glob(path.name + ".corrupt-*"))


def check_basics(tmp: Path) -> None:
    print("1-4. reading, saving, replacing, forgetting")
    path = tmp / "deep" / "er" / "saved_prompts.json"
    c = client_for(path)
    r = c.get("/api/saved_prompts")
    check(r.status_code == 200, f"GET empty: {r.status_code}")
    body = r.json()
    check(body == {"questions": [], "systems": [], "file": str(path), "warning": None}, f"GET empty body: {body}")
    check(not path.parent.exists(), "GET wrote something")

    question = "  How many legs\n does a spider have?  "
    r = c.post(
        "/api/saved_prompts",
        json={"kind": "question", "name": "  Spider legs  ", "question": question, "answer_begins": " A spider has", "chat": True},
    )
    check(r.status_code == 200, f"POST question: {r.status_code} {r.text}")
    body = r.json()
    q = body["questions"][0] if body.get("questions") else {}
    check(set(q) == {"id", "name", "question", "answer_begins", "chat", "saved_at"}, f"question keys: {set(q)}")
    check(q.get("name") == "Spider legs", f"name not trimmed: {q.get('name')!r}")
    check(q.get("question") == question and q.get("answer_begins") == " A spider has", "text changed")
    check(q.get("chat") is True, "chat lost")
    check(isinstance(q.get("id"), str) and len(q["id"]) == 16 and all(ch in "0123456789abcdef" for ch in q["id"]),
          f"id not 8 random bytes in hex: {q.get('id')!r}")
    check(str(q.get("saved_at", "")).endswith("Z") and "T" in q.get("saved_at", ""), f"saved_at: {q.get('saved_at')}")
    check(path.exists(), "the directory was not created on first write")
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    check(on_disk.get("version") == 1 and on_disk["questions"] == body["questions"] and on_disk["systems"] == [],
          f"file content: {on_disk}")
    check(not [p for p in path.parent.iterdir() if p.name.endswith(".tmp")], "a temporary file was left behind")
    check(body["file"] == str(path) and body["warning"] is None, "reply shape")

    r = c.post("/api/saved_prompts", json={"kind": "system", "name": "Gemma", "text": "You are Gemma.\n"})
    check(r.status_code == 200 and r.json()["systems"][0]["text"] == "You are Gemma.\n", f"POST system: {r.text}")
    sys_id = r.json()["systems"][0]["id"]
    check(set(r.json()["systems"][0]) == {"id", "name", "text", "saved_at"}, "system keys")
    check(len(r.json()["questions"]) == 1, "POST system dropped the questions")

    r = c.post("/api/saved_prompts", json={"kind": "question", "name": "spider LEGS ", "question": "x"})
    check(r.status_code == 409 and "Spider legs" in r.json().get("detail", ""), f"duplicate name: {r.status_code} {r.text}")
    r = c.post("/api/saved_prompts", json={"kind": "system", "name": "Spider legs", "text": "y"})
    check(r.status_code == 200, f"same name in the other list refused: {r.status_code}")
    r = c.post("/api/saved_prompts", json={"kind": "question", "name": "Spider legs", "question": "Again?", "replace_id": q["id"]})
    check(r.status_code == 200, f"replace: {r.status_code} {r.text}")
    rq = r.json()["questions"]
    check(len(rq) == 1 and rq[0]["id"] == q["id"] and rq[0]["question"] == "Again?" and rq[0]["chat"] is False,
          f"replace in place: {rq}")
    r = c.post("/api/saved_prompts", json={"kind": "question", "name": "Other", "question": "z", "replace_id": "0" * 16})
    check(r.status_code == 404, f"replace of an unknown id: {r.status_code}")
    r = c.post("/api/saved_prompts", json={"kind": "question", "name": "Second", "question": "z"})
    second = r.json()["questions"][1]["id"]
    r = c.post("/api/saved_prompts", json={"kind": "question", "name": "second", "question": "z2", "replace_id": q["id"]})
    check(r.status_code == 409, f"replace onto another entry's name: {r.status_code}")

    r = c.delete(f"/api/saved_prompts/question/{second}")
    check(r.status_code == 200 and [e["id"] for e in r.json()["questions"]] == [q["id"]], f"DELETE: {r.text}")
    check(len(r.json()["systems"]) == 2, "DELETE touched the other list")
    r = c.delete(f"/api/saved_prompts/question/{second}")
    check(r.status_code == 404, f"DELETE twice: {r.status_code}")
    r = c.delete(f"/api/saved_prompts/question/{sys_id}")
    check(r.status_code == 404, f"DELETE from the wrong list: {r.status_code}")
    r = c.delete(f"/api/saved_prompts/nonsense/{sys_id}")
    check(r.status_code == 404, f"DELETE from an unknown kind: {r.status_code}")
    r = c.get("/api/saved_prompts")
    check(len(r.json()["questions"]) == 1 and len(r.json()["systems"]) == 2, "GET after the changes")


def check_origins(tmp: Path) -> None:
    print("5. who may ask")
    path = tmp / "origins" / "saved_prompts.json"
    c = client_for(path)
    for origin in ("http://127.0.0.1:8765", "http://localhost:5173", "http://[::1]:4173", "https://localhost"):
        r = c.get("/api/saved_prompts", headers={"Origin": origin})
        check(r.status_code == 200, f"Origin {origin} refused: {r.status_code}")
    r = c.post("/api/saved_prompts", headers={"Origin": "http://localhost:5173"},
               json={"kind": "system", "name": "ok", "text": "t"})
    check(r.status_code == 200, f"POST from localhost: {r.status_code}")
    before = path.read_text(encoding="utf-8")
    for origin in ("https://evil.example", "http://127.0.0.1.evil.example", "null", "http://localhost.evil:8765", "file://"):
        r = c.get("/api/saved_prompts", headers={"Origin": origin})
        check(r.status_code == 403, f"GET from {origin}: {r.status_code}")
        r = c.post("/api/saved_prompts", headers={"Origin": origin}, json={"kind": "system", "name": f"bad {origin}", "text": "t"})
        check(r.status_code == 403, f"POST from {origin}: {r.status_code}")
        sid = c.get("/api/saved_prompts").json()["systems"][0]["id"]
        r = c.delete(f"/api/saved_prompts/system/{sid}", headers={"Origin": origin})
        check(r.status_code == 403, f"DELETE from {origin}: {r.status_code}")
    check(path.read_text(encoding="utf-8") == before, "a refused origin changed the file")
    r = c.get("/api/saved_prompts")
    check(r.status_code == 200, "no Origin (a script) refused")

    # the Host: a DNS-rebinding page (rebind.evil.example resolving to 127.0.0.1) sends its own name as the Host and,
    # on a same-origin GET, no Origin at all
    for host in ("127.0.0.1:8765", "localhost:8765", "[::1]:8765", "localhost", "LOCALHOST:5173"):
        r = c.get("/api/saved_prompts", headers={"Host": host})
        check(r.status_code == 200, f"Host {host} refused: {r.status_code}")
    r = c.post("/api/saved_prompts", headers={"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:5173"},
               json={"kind": "system", "name": "through vite", "text": "t"})
    check(r.status_code == 200, f"vite's proxy (Host 127.0.0.1:8765, the page's Origin) refused: {r.status_code}")
    before = path.read_text(encoding="utf-8")
    for host in ("rebind.evil.example:8765", "rebind.evil.example", "127.0.0.1.evil.example:8765", "gpu-box:8765",
                 "192.168.1.20:8765", "[::1"):
        r = c.get("/api/saved_prompts", headers={"Host": host})
        check(r.status_code == 403, f"GET with Host {host} and no Origin: {r.status_code}")
        check("questions" not in r.text and str(path) not in r.text, f"Host {host}: the refusal shows the lists or the file")
        r = c.post("/api/saved_prompts", headers={"Host": host}, json={"kind": "system", "name": f"bad {host}", "text": "t"})
        check(r.status_code == 403, f"POST with Host {host}: {r.status_code}")
        sid = c.get("/api/saved_prompts").json()["systems"][0]["id"]
        r = c.delete(f"/api/saved_prompts/system/{sid}", headers={"Host": host})
        check(r.status_code == 403, f"DELETE with Host {host}: {r.status_code}")
    r = c.get("/api/saved_prompts", headers={"Host": "rebind.evil.example:8765", "Origin": "http://localhost:8765"})
    check(r.status_code == 403, f"a local Origin with another Host: {r.status_code}")
    check(path.read_text(encoding="utf-8") == before, "a refused Host changed the file")


def check_corrupt(tmp: Path) -> None:
    print("6. a file that cannot be read")
    surrogate = {"id": "a" * 16, "name": "half", "text": "t", "saved_at": "2026-10-18T00:00:00Z"}
    for label, content, why in (
        ("not JSON", b"{ this is not json", "not JSON"),
        ("not UTF-8", b'{"version": 1, "questions": [], "systems": [{"name": "caf\xe9"}]}', "not UTF-8"),
        ("a byte 0xFF", b"\xff", "not UTF-8"),
        ("nested too deeply", b"[" * 100000, "nested too deeply"),
        ("wrong version", json.dumps({"version": 2, "questions": [], "systems": []}).encode(), "not a version 1"),
        ("malformed entry", json.dumps({"version": 1, "questions": [{"id": "a", "name": 3}], "systems": []}).encode(),
         "malformed entry"),
        ("not a list", json.dumps({"version": 1, "questions": {}, "systems": []}).encode(), "not a list"),
        # a hand-written \ud83d escape: valid JSON, but no reply holding it could be sent as UTF-8
        ("a lone surrogate", json.dumps({"version": 1, "questions": [], "systems": [{**surrogate, "text": "a\ud83d"}]}).encode(),
         "malformed entry"),
    ):
        path = tmp / label.replace(" ", "_") / "saved_prompts.json"
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
        c = client_for(path)
        r = c.get("/api/saved_prompts")
        check(r.status_code == 200, f"{label}: GET gave {r.status_code}")
        body = r.json() if r.status_code == 200 else {"questions": None, "systems": None}
        aside = corrupt_files(path)
        check(body["questions"] == [] and body["systems"] == [], f"{label}: lists not empty")
        check(isinstance(body.get("warning"), str) and "moved aside" in body["warning"] and why in body["warning"],
              f"{label}: no warning naming why ({why}): {body.get('warning')}")
        check(len(aside) == 1 and aside[0].read_bytes() == content, f"{label}: not moved aside whole")
        check(not path.exists(), f"{label}: the corrupt file is still in place")
        r = c.get("/api/saved_prompts")
        check(r.status_code == 200 and r.json()["warning"] is None, f"{label}: the warning came twice")
        # a write after a corrupt read makes a new file and leaves the old one aside
        path.write_bytes(content)
        r = c.post("/api/saved_prompts", json={"kind": "system", "name": "fresh", "text": "t"})
        check(r.status_code == 200 and isinstance(r.json()["warning"], str), f"{label}: POST over a corrupt file: {r.text}")
        check(len(corrupt_files(path)) == 2, f"{label}: the second corrupt file was not kept")
        check(json.loads(path.read_text(encoding="utf-8"))["systems"][0]["name"] == "fresh", f"{label}: new file wrong")


def check_limits(tmp: Path) -> None:
    print("7. the limits")
    path = tmp / "limits" / "saved_prompts.json"
    c = client_for(path)
    post = lambda body: c.post("/api/saved_prompts", json=body)  # noqa: E731
    cases: List[tuple] = [
        ({"kind": "question", "name": "", "question": "q"}, 422, "an empty name"),
        ({"kind": "question", "name": "   ", "question": "q"}, 422, "a blank name"),
        ({"kind": "question", "name": "x" * 81, "question": "q"}, 422, "an 81-character name"),
        ({"kind": "question", "name": "a\tb", "question": "q"}, 422, "a tab in the name"),
        ({"kind": "question", "name": "a\nb", "question": "q"}, 422, "a new line in the name"),
        ({"kind": "question", "name": "no text"}, 422, "a question without text"),
        ({"kind": "question", "name": "empty", "question": ""}, 422, "an empty question"),
        ({"kind": "question", "name": "long", "question": "q" * 32001}, 422, "a 32001-character question"),
        ({"kind": "question", "name": "begun", "question": "q", "answer_begins": "a" * 4001}, 422, "a 4001-character begun answer"),
        ({"kind": "system", "name": "no text"}, 422, "a system prompt without text"),
        ({"kind": "system", "name": "long", "text": "t" * 32001}, 422, "a 32001-character system prompt"),
        ({"kind": "other", "name": "x", "text": "t"}, 422, "an unknown kind"),
        ({"name": "x", "text": "t"}, 422, "no kind"),
        ({"kind": "question", "name": "x" * 80, "question": "q" * 32000, "answer_begins": "a" * 4000}, 200, "everything at its limit"),
        ({"kind": "system", "name": "  padded  ", "text": "  "}, 200, "a text of spaces kept as given"),
    ]
    for body, status, what in cases:
        r = post(body)
        check(r.status_code == status, f"{what}: {r.status_code}, expected {status}")
        if status == 422:
            check(isinstance(r.json().get("detail"), str), f"{what}: the refusal is not in plain words: {r.text[:200]}")
    stored = json.loads(path.read_text(encoding="utf-8"))
    check(stored["systems"][0]["text"] == "  " and stored["systems"][0]["name"] == "padded", "padded text changed")
    # half a character: a lone surrogate (the first half of an emoji cut in two) passes JSON but not UTF-8, so it is
    # refused in plain words rather than failing the write (sent as raw bytes: httpx would refuse to encode it)
    before = path.read_bytes()
    for body, what in (
        ('{"kind": "question", "name": "a\\ud83d", "question": "q"}', "the name"),
        ('{"kind": "question", "name": "q1", "question": "legs \\ud83d"}', "the question"),
        ('{"kind": "question", "name": "q2", "question": "q", "answer_begins": "\\ude00 A"}', "the begun answer"),
        ('{"kind": "system", "name": "s1", "text": "You are \\ud83d"}', "the system prompt"),
    ):
        r = c.post("/api/saved_prompts", content=body.encode(), headers={"content-type": "application/json"})
        detail = r.json().get("detail", "") if r.headers.get("content-type", "").startswith("application/json") else r.text
        check(r.status_code == 422 and "broken character" in str(detail) and what.split()[-1] in str(detail).lower(),
              f"a lone surrogate in {what}: {r.status_code} {str(detail)[:120]}")
    check(path.read_bytes() == before, "a refused lone surrogate changed the file")
    r = post({"kind": "system", "name": "whole \U0001F600", "text": "an emoji kept whole \U0001F600"})
    check(r.status_code == 200 and r.json()["systems"][-1]["text"].endswith("\U0001F600"), f"a whole emoji: {r.status_code}")
    # the list holds at most 500: fill it on disk, then ask for one more
    data = json.loads(path.read_text(encoding="utf-8"))
    data["systems"] = [
        {"id": f"{i:016x}", "name": f"s{i}", "text": "t", "saved_at": "2026-10-18T00:00:00Z"} for i in range(sp.MAX_ENTRIES)
    ]
    path.write_text(json.dumps(data), encoding="utf-8")
    r = post({"kind": "system", "name": "one more", "text": "t"})
    check(r.status_code == 400, f"a 501st entry: {r.status_code}")
    r = post({"kind": "system", "name": "s0", "text": "replaced", "replace_id": f"{0:016x}"})
    check(r.status_code == 200 and len(r.json()["systems"]) == sp.MAX_ENTRIES, f"a replace in a full list: {r.status_code}")


def check_default_path() -> None:
    print("8. where the file lives")
    saved = {k: os.environ.get(k) for k in (sp.SAVED_PROMPTS_ENV, "XDG_DATA_HOME")}
    try:
        os.environ.pop(sp.SAVED_PROMPTS_ENV, None)
        os.environ["XDG_DATA_HOME"] = "/tmp/xdg-example"
        check(sp.saved_prompts_path() == Path("/tmp/xdg-example/theodolite/saved_prompts.json"), "XDG_DATA_HOME ignored")
        os.environ.pop("XDG_DATA_HOME")
        check(sp.saved_prompts_path() == Path.home() / ".local/share/theodolite/saved_prompts.json", "default path")
        check(ROOT_DIR not in sp.saved_prompts_path().parents, "the default file lives inside this checkout")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="theodolite_saved_") as d:
        tmp = Path(d)
        check_basics(tmp)
        check_origins(tmp)
        check_corrupt(tmp)
        check_limits(tmp)
    check_default_path()
    if failures:
        print(f"\nsaved prompts: {len(failures)} of {passed + len(failures)} checks FAILED")
        return 1
    print(f"\nsaved prompts: all {passed} checks hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())

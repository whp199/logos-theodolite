#!/usr/bin/env python3
"""A CPU-only check of the atlas's scientist chat (backend/scientist_chat.py), its tools (backend/scientist_tools.py)
and its MCP server (backend/scientist_mcp.py), on the instrument's server with the tiny random Gemma 4 of
scripts/atlas_stand_in.py bound in place of the model, served by uvicorn on a free port (an event stream never ends,
so the chat is driven over real connections):
1. the tools' schemas, and the pages' count: layers from 1 in every argument and reply, refused below 1;
2. each tool against the atlas's routes in this process, its reply the routes' own in the pages' count (a neuron, its
   links, reads, steering, odds, compare, fires, heads, the drive, the census, names and notes kept without
   overwriting each other, the cone and the web once the indexes are built), and each screen action reaching the
   chat's page in the API's count;
3. the MCP server, started as Claude Code starts it: initialize, its tools, a call, refusals in plain words;
4. Claude Code, played by a fake `claude` that starts the MCP server from the config it is given and calls it: the
   command line (stream-json, the prompt file, the MCP config naming this server and the chat, the tools allowed, a
   new session then a resumed one), the events of a turn (text, the tool, the screen action, its result, done with
   its cost), what the person had open told in the pages' count, attachments, a message sent mid-turn, an interrupt,
   past chats and their history from the session file, and a rewind; the environment both CLIs start with: no
   endpoint's key, no Hugging Face or cloud credential, each CLI's own login (and Claude Code's cloud login on Bedrock);
5. Codex, played by a fake `codex app-server`: its home, the thread and its instructions, a turn's events, an
   approval it asks for declined, an interrupt, history from the log, a rewind (thread/revert);
6. an OpenAI-compatible endpoint, played by a fake server: its models; the tool loop (the request: the model, the
   tools, the key from its environment variable, the system prompt without the shell section; the tool calls
   streamed in pieces and run against the atlas in this process; the result sent back; the answer); arguments that
   are not JSON; a message sent mid-turn answered in the same turn; an interrupt leaving a valid conversation; a
   profile without streaming; an endpoint's error and a missing key in plain words; a profile file holding a key
   refused; history, past chats, rewind;
7. who may ask: the chat's routes refuse a page from another machine.

Run: .venv/bin/python -I scripts/check_scientist.py   (exits 0 when every check holds)
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"

import asyncio
import json
import shutil
import socket
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

TMP = Path(tempfile.mkdtemp(prefix="scientist-check-"))
os.environ["THEODOLITE_ATLAS_DIR"] = str(TMP / "atlas")
os.environ["THEODOLITE_FIELD_NOTES"] = str(TMP / "notes.json")
os.environ["THEODOLITE_CENSUS_DIR"] = str(TMP / "census")
os.environ["THEODOLITE_SCIENTIST_DIR"] = str(TMP / "scientist")
os.environ["HOME"] = str(TMP / "home")
for k in ("THEODOLITE_OPENAI_BASE_URL", "THEODOLITE_OPENAI_API_KEY", "THEODOLITE_OPENAI_MODEL", "OPENROUTER_API_KEY", "THEODOLITE_SELF_URL",
          "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY"):
    os.environ.pop(k, None)
(TMP / "home").mkdir()
# credentials no scientist is given, and the CLIs' own logins, which they are (test values, set before section 4)
SECRETS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_API_TOKEN", "RUNPOD_API_KEY", "VAST_API_KEY", "LAMBDA_API_KEY", "MODAL_TOKEN_ID",
           "MODAL_TOKEN_SECRET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "GOOGLE_APPLICATION_CREDENTIALS",
           "AZURE_CLIENT_SECRET", "NEURONARY_TOKEN")
LOGINS = ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "OPENAI_API_KEY", "CODEX_API_KEY")

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import httpx
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

import backend.atlas_build as ab
import backend.scientist_chat as sc
import backend.scientist_tools as st
from atlas_stand_in import CORPUS, bind_server, stand_in
from backend import census as cz

failures: List[str] = []
passed = 0
PY = sys.executable


def check(ok: bool, what: str) -> None:
    global passed
    if ok:
        passed += 1
    else:
        failures.append(what)
        print(f"  FAIL  {what}")


def section(title: str, before: int) -> None:
    if len(failures) == before:
        print(f"  ok    {title}")


def quiet(_s: str) -> None:
    pass


def free_port() -> int:
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        return sk.getsockname()[1]


class Served:
    """An ASGI app on uvicorn in a thread, on a free port."""

    def __init__(self, app: Any) -> None:
        import uvicorn

        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning", lifespan="off"))
        self.th = threading.Thread(target=self.srv.run, daemon=True)
        self.th.start()
        for _ in range(400):
            if self.srv.started:
                break
            time.sleep(0.025)

    def stop(self) -> None:
        self.srv.should_exit = True
        self.th.join(timeout=10)


class Events:
    """A chat's event stream read in a thread, every event kept."""

    def __init__(self, base: str, chat: str) -> None:
        self.events: List[Dict[str, Any]] = []
        self.cv = threading.Condition()
        self.stop = False
        self.th = threading.Thread(target=self.run, args=(base, chat), daemon=True)
        self.th.start()
        self.wait(lambda e: e["t"] == "state", 20)

    def run(self, base: str, chat: str) -> None:
        try:
            with httpx.Client(base_url=base, timeout=None) as hc:
                with hc.stream("GET", "/api/atlas/scientist/events", params={"chat": chat}) as r:
                    for line in r.iter_lines():
                        if self.stop:
                            return
                        if line.strip():
                            with self.cv:
                                self.events.append(json.loads(line))
                                self.cv.notify_all()
        except Exception:
            pass

    def wait(self, pred: Callable[[Dict[str, Any]], bool], timeout: float = 60, after: int = 0) -> Optional[Dict[str, Any]]:
        end = time.time() + timeout
        with self.cv:
            while True:
                for e in self.events[after:]:
                    if pred(e):
                        return e
                left = end - time.time()
                if left <= 0:
                    return None
                self.cv.wait(left)

    def since(self, i: int) -> List[Dict[str, Any]]:
        with self.cv:
            return [e for e in self.events[i:] if e["t"] != "ping"]

    def mark(self) -> int:
        with self.cv:
            return len(self.events)


def write_exec(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


FAKE_CLAUDE = r'''#!{PY} -I
import json, os, queue, subprocess, sys, threading, time
args = sys.argv[1:]
def opt(name):
    return args[args.index(name) + 1]
def log(kind, **kw):
    with open(os.environ["FAKE_LOG"], "a") as f:
        f.write(json.dumps({"kind": kind, **kw}) + "\n")
sid = opt("--session-id") if "--session-id" in args else opt("--resume")
log("start", argv=args, cwd=os.getcwd(), sid=sid, resumed="--resume" in args, prompt=open(opt("--system-prompt-file")).read(),
    env=sorted(k for k in json.loads(os.environ.get("FAKE_WATCH", "[]")) if k in os.environ))
cfg = json.loads(opt("--mcp-config"))["mcpServers"]["atlas"]
mcp = subprocess.Popen([cfg["command"], *cfg["args"]], env={**os.environ, **cfg["env"]}, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
rid = [0]
def rpc(method, params=None):
    rid[0] += 1
    mcp.stdin.write(json.dumps({"jsonrpc": "2.0", "id": rid[0], "method": method, "params": params or {}}) + "\n")
    mcp.stdin.flush()
    return json.loads(mcp.stdout.readline())
rpc("initialize", {"protocolVersion": "2025-06-18"})
log("tools", names=[t["name"] for t in rpc("tools/list")["result"]["tools"]])
sdir = os.path.expanduser("~/.claude/projects/-fake-work")
os.makedirs(sdir, exist_ok=True)
def keep(obj):
    with open(os.path.join(sdir, sid + ".jsonl"), "a") as f:
        f.write(json.dumps(obj) + "\n")
def out(obj):
    print(json.dumps(obj), flush=True)
def delta(text):
    out({"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}}})
inbox = queue.Queue()
def read():
    for line in sys.stdin:
        inbox.put(line)
    inbox.put(None)
threading.Thread(target=read, daemon=True).start()
while True:
    line = inbox.get()
    if line is None:
        break
    m = json.loads(line)
    if m.get("type") != "user":
        log("idle_" + m.get("type", "?"))
        continue
    content = m["message"]["content"]
    text = content[0]["text"]
    log("user", text=text, blocks=[b["type"] for b in content], kinds=[b.get("source", {}).get("media_type") for b in content])
    keep({"type": "user", "message": {"role": "user", "content": content}})
    if "slow" in text:
        stopped = False
        for i in range(200):
            delta(f"{i} ")
            time.sleep(0.03)
            try:
                nxt = inbox.get_nowait()
            except queue.Empty:
                continue
            if nxt is None:
                break
            mm = json.loads(nxt)
            if mm.get("type") == "control_request" and mm["request"]["subtype"] == "interrupt":
                log("interrupt")
                stopped = True
                break
            if mm.get("type") == "user":
                log("midturn", text=mm["message"]["content"][0]["text"])
        out({"type": "result", "subtype": "error_during_execution" if stopped else "success", "is_error": stopped})
        continue
    delta("Looking. ")
    use = {"type": "tool_use", "id": "toolu_1", "name": "mcp__atlas__select_neuron", "input": {"layer": 3, "n": 5}}
    am = {"type": "assistant", "message": {"id": "msg_1", "role": "assistant", "content": [{"type": "text", "text": "Looking. "}, use]}}
    out(am)
    keep(am)
    res = rpc("tools/call", {"name": "select_neuron", "arguments": {"layer": 3, "n": 5}})["result"]
    tr = {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": res["content"], "is_error": bool(res.get("isError"))}]}}
    out(tr)
    keep(tr)
    log("neuron", text=rpc("tools/call", {"name": "neuron", "arguments": {"layer": 3, "n": 5}})["result"]["content"][0]["text"])
    delta("Done.")
    am2 = {"type": "assistant", "message": {"id": "msg_2", "role": "assistant", "content": [{"type": "text", "text": "Done."}]}}
    out(am2)
    keep(am2)
    out({"type": "result", "subtype": "success", "is_error": False, "total_cost_usd": 0.0123})
mcp.stdin.close()
mcp.wait()
'''

FAKE_CODEX = r'''#!{PY} -I
import json, os, sys
def log(kind, **kw):
    with open(os.environ["FAKE_LOG"], "a") as f:
        f.write(json.dumps({"kind": kind, **kw}) + "\n")
log("start", argv=sys.argv[1:], cwd=os.getcwd(), codex_home=os.environ.get("CODEX_HOME"),
    env=sorted(k for k in json.loads(os.environ.get("FAKE_WATCH", "[]")) if k in os.environ))
n = 0
def send(obj):
    print(json.dumps(obj), flush=True)
for line in sys.stdin:
    m = json.loads(line)
    if "method" not in m:
        log("reply", msg=m)
        continue
    meth, mid, p = m["method"], m.get("id"), m.get("params") or {}
    log("rpc", method=meth, params=p)
    res = lambda r: send({"jsonrpc": "2.0", "id": mid, "result": r})
    note = lambda method, params: send({"jsonrpc": "2.0", "method": method, "params": params})
    if meth == "initialize":
        res({"serverInfo": {"name": "fake-codex"}})
    elif meth in ("thread/start", "thread/resume"):
        res({"thread": {"id": p.get("threadId") or "thr_1"}})
    elif meth == "turn/start":
        n += 1
        tid = f"turn_{n}"
        res({"turn": {"id": tid}})
        note("turn/started", {"turn": {"id": tid}})
        note("item/agentMessage/delta", {"itemId": "m1", "delta": "Reading "})
        note("item/agentMessage/delta", {"itemId": "m1", "delta": "the neuron."})
        send({"jsonrpc": "2.0", "id": 900 + n, "method": "item/commandExecution/requestApproval", "params": {"command": "rm -rf /"}})
        item = {"type": "mcpToolCall", "id": "it1", "tool": "neuron", "arguments": {"layer": 3, "n": 5}}
        note("item/started", {"item": item})
        note("item/completed", {"item": {**item, "status": "completed", "result": {"content": [{"type": "text", "text": "{\"neuron\": \"L3 N5\"}"}]}}})
        note("item/agentMessage/delta", {"itemId": "m2", "delta": "Done."})
        note("turn/completed", {"turn": {"id": tid, "status": "completed"}})
    elif meth in ("turn/steer", "turn/interrupt", "thread/revert"):
        res({})
    else:
        send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": meth}})
'''


def fake_openai_app(seen: List[Dict[str, Any]]) -> Any:
    """An OpenAI-compatible endpoint whose answers follow the person's last message."""

    app = FastAPI()

    @app.get("/v1/models")
    async def models(request: Request) -> Any:
        seen.append({"path": "/v1/models", "auth": request.headers.get("authorization")})
        return {"data": [{"id": "fake-model", "supported_parameters": ["tools", "temperature"]}, {"id": "no-tools", "supported_parameters": []},
                         {"id": "silent"}]}

    def chunk(delta: Dict[str, Any], finish: Optional[str] = None) -> str:
        return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n"

    @app.post("/v1/chat/completions")
    async def complete(request: Request) -> Any:
        body = await request.json()
        seen.append({"path": "/v1/chat/completions", "auth": request.headers.get("authorization"), "body": body,
                     "title": request.headers.get("x-title")})
        msgs = body["messages"]
        users = [m for m in msgs if m["role"] == "user"]
        last_user = users[-1]["content"] if isinstance(users[-1]["content"], str) else users[-1]["content"][0]["text"]
        after_tool = msgs[-1]["role"] == "tool"
        if "fail" in last_user:
            return JSONResponse({"error": {"message": "the model fell over"}}, status_code=500)
        if not body.get("stream"):
            if after_tool:
                return {"choices": [{"message": {"role": "assistant", "content": "No names yet."}, "finish_reason": "stop"}]}
            return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_n", "type": "function", "function": {"name": "names", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}]}

        async def gen():
            if "slow" in last_user:
                for i in range(100):
                    yield chunk({"content": f"{i} "})
                    await asyncio.sleep(0.05)
                yield "data: [DONE]\n\n"
                return
            if "queue" in last_user and not after_tool and msgs[-1]["role"] == "user" and len(users) == 1:
                yield chunk({"tool_calls": [{"index": 0, "id": "call_q", "type": "function", "function": {"name": "activations", "arguments": ""}}]})
                yield chunk({"tool_calls": [{"index": 0, "function": {"arguments": json.dumps({"layer": 4, "n": 9, "text": "Alice called Bob"})}}]}, "tool_calls")
                yield "data: [DONE]\n\n"
                return
            if "queue" in last_user and after_tool:
                for word in ("Reading ", "the ", "trace ", "slowly."):
                    yield chunk({"content": word})
                    await asyncio.sleep(0.4)
                yield chunk({}, "stop")
                yield "data: [DONE]\n\n"
                return
            if "bad args" in last_user and not after_tool:
                yield chunk({"tool_calls": [{"index": 0, "id": "call_b", "function": {"name": "neuron", "arguments": "{not json"}}]}, "tool_calls")
                yield "data: [DONE]\n\n"
                return
            if not after_tool and "explain" in last_user.lower():
                yield chunk({"reasoning": "the person wants the neuron"})
                yield chunk({"content": "Let me look."})
                yield chunk({"tool_calls": [{"index": 0, "id": "call_a", "type": "function", "function": {"name": "neuron", "arguments": "{\"layer\": 3,"}}]})
                yield chunk({"tool_calls": [{"index": 0, "function": {"arguments": " \"n\": 5}"}}]})
                yield chunk({"tool_calls": [{"index": 1, "id": "call_b", "type": "function", "function": {"name": "select_neuron", "arguments": "{\"layer\": 3, \"n\": 5}"}}]}, "tool_calls")
                yield "data: [DONE]\n\n"
                return
            yield chunk({"content": "It writes "})
            yield chunk({"content": "`Bob`."}, "stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def main() -> int:
    try:
        return run_checks()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


def run_checks() -> int:
    torch.set_grad_enabled(False)
    rt, vault = stand_in()
    app = bind_server(rt, vault)
    L = len(rt.hf_model.model.layers)

    # --------------------------------------------------------------------------------------- 1. schemas, counting
    before = len(failures)
    names = [t["name"] for t in st.TOOLS]
    check(len(names) == len(set(names)) and all(t["description"] and t["inputSchema"]["type"] == "object" for t in st.TOOLS), "every tool named once, described")
    check(all(set(t["inputSchema"].get("required", [])) <= set(t["inputSchema"].get("properties", {})) for t in st.TOOLS), "every required argument is described")
    check(set(st.UI_TOOLS) <= set(names) and [f["function"]["name"] for f in st.openai_tools()] == names, "screen tools and the OpenAI form")
    got = st.from_page({"layer": 54, "neurons": [{"layer": 1, "n": 3}], "heads": [{"layer": 60, "head": "all"}], "layers": [41, 58], "n": 7})
    check(got == {"layer": 53, "neurons": [{"layer": 0, "n": 3}], "heads": [{"layer": 59, "head": "all"}], "layers": [40, 57], "n": 7}, f"from_page: {got}")
    try:
        st.from_page({"layer": 0})
        check(False, "layer 0 refused")
    except st.ToolError as exc:
        check("count from 1" in str(exc), f"layer 0: {exc}")
    back = st.to_page({"layer": 53, "x": [{"layer": 0}], "z_by_layer": {"0": [1.0], "59": [2.0]}, "census_layers": [3], "pos": 4})
    check(back == {"layer": 54, "x": [{"layer": 1}], "z_by_layer": {"1": [1.0], "60": [2.0]}, "census_layers": [4], "pos": 4}, f"to_page: {back}")
    check(st.cite(53, 15491) == "L54 N15491" and st.cite_head(5, 3) == "L6 h3", "citations in the pages' count")
    section("the tools' schemas and the pages' count", before)

    # -------------------------------------------------------------------------------------- 2. the tools in-process
    before = len(failures)
    srv = Served(app)
    hc = httpx.Client(base_url=srv.url, timeout=120)

    async def tools_in_process() -> None:
        api = sc.InProcessApi(app)
        try:
            await tool_checks(api, hc)
        finally:
            await api.close()

    asyncio.run(tools_in_process())
    section("each tool on the atlas's routes, in the pages' count", before)

    # -------------------------------------------------------------------------------------------- 3. the MCP server
    before = len(failures)
    cid = str(uuid.uuid4())
    env = {**os.environ, "THEODOLITE_URL": srv.url, "THEODOLITE_CHAT": cid}
    proc = subprocess.Popen([PY, "-I", str(ROOT_DIR / "backend" / "scientist_mcp.py")], env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True, bufsize=1)
    rid = [0]

    def rpc(method: str, params: Optional[Dict[str, Any]] = None, notify: bool = False) -> Optional[Dict[str, Any]]:
        msg: Dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if not notify:
            rid[0] += 1
            msg["id"] = rid[0]
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()
        return None if notify else json.loads(proc.stdout.readline())

    init = rpc("initialize", {"protocolVersion": "2025-06-18"})
    rpc("notifications/initialized", notify=True)
    check(init["result"]["serverInfo"]["name"] == "atlas" and "tools" in init["result"]["capabilities"], f"initialize: {init}")
    tl = rpc("tools/list")["result"]["tools"]
    check([t["name"] for t in tl] == names, "tools/list: the scientist's tools")
    r = rpc("tools/call", {"name": "neuron", "arguments": {"layer": 3, "n": 5}})["result"]
    body = json.loads(r["content"][0]["text"])
    check(not r.get("isError") and body["neuron"] == "L3 N5" and all(x["neuron"].startswith("L") for x in body["writes_into"]), "tools/call neuron")
    r = rpc("tools/call", {"name": "neuron", "arguments": {"layer": L + 1, "n": 5}})["result"]
    check(r.get("isError") and f"Layer {L} does not exist" in r["content"][0]["text"], f"a layer past the last refused in plain words: {r['content'][0]['text'][:120]}")
    r = rpc("tools/call", {"name": "nope", "arguments": {}})["result"]
    check(r.get("isError") and "There is no tool" in r["content"][0]["text"], "an unknown tool refused")
    check(rpc("ping")["result"] == {} and rpc("resources/list")["error"]["code"] == -32601, "ping, and a method it lacks")
    proc.stdin.close()
    proc.wait(timeout=10)
    section("the MCP server over stdio", before)

    # ------------------------------------------------------------------------------------------- 4. Claude Code
    before = len(failures)
    os.environ.update({k: "test-" + k.lower() for k in (*SECRETS, *LOGINS)})
    os.environ["FAKE_WATCH"] = json.dumps([*SECRETS, *LOGINS])
    fake_log = TMP / "fake_claude.log"
    os.environ["FAKE_LOG"] = str(fake_log)
    os.environ["THEODOLITE_CLAUDE_BIN"] = str(write_exec(TMP / "fake-claude", FAKE_CLAUDE.replace("{PY}", PY)))

    def flog() -> List[Dict[str, Any]]:
        return [json.loads(x) for x in fake_log.read_text().splitlines()] if fake_log.exists() else []

    chat = str(uuid.uuid4())
    ev = Events(srv.url, chat)
    i0 = ev.mark()
    r = hc.post("/api/atlas/scientist/send", json={"chat": chat, "message": "What does this one do?", "agent": "claude", "layer": 2, "n": 5})
    check(r.status_code == 200 and r.json() == {"ok": True}, f"send: {r.status_code} {r.text[:200]}")
    done = ev.wait(lambda e: e["t"] == "done", 90, i0)
    seq = ev.since(i0)
    kinds = [e["t"] for e in seq]
    check(done is not None and done.get("cost") == 0.0123, f"a turn ends done, with its cost ({kinds})")
    check(kinds[0] == "sent" and "text" in kinds and kinds.index("tool") < kinds.index("ui") < kinds.index("tool_result"), f"the turn's events in order: {kinds}")
    tool = next(e for e in seq if e["t"] == "tool")
    ui = next(e for e in seq if e["t"] == "ui")
    res = next(e for e in seq if e["t"] == "tool_result")
    check(tool["name"] == "select_neuron" and tool["input"] == {"layer": 3, "n": 5}, f"the tool, its name without the MCP prefix: {tool}")
    check(ui == {"t": "ui", "action": "select", "layer": 2, "n": 5}, f"the screen action in the API's count: {ui}")
    check(not res["error"] and "shown in the atlas" in res["output"], f"its result: {res}")
    starts = [x for x in flog() if x["kind"] == "start"]
    argv = starts[0]["argv"]
    mcp_cfg = json.loads(argv[argv.index("--mcp-config") + 1])["mcpServers"]["atlas"]
    check(argv[:7] == ["-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose", "--include-partial-messages"],
          f"claude -p with stream-json: {argv[:7]}")
    check("--strict-mcp-config" in argv and argv[argv.index("--allowedTools") + 1] == "mcp__atlas" and argv[argv.index("--session-id") + 1] == chat,
          "a strict MCP config, the atlas's tools allowed, a new session named for the chat")
    check(mcp_cfg["command"] == PY and mcp_cfg["args"] == ["-I", str(ROOT_DIR / "backend" / "scientist_mcp.py")]
          and mcp_cfg["env"] == {"THEODOLITE_URL": srv.url, "THEODOLITE_CHAT": chat}, f"the MCP server named, this server and this chat: {mcp_cfg}")
    check(starts[0]["cwd"] == str(sc.work_dir()) and "Gemma-4-31B" in starts[0]["prompt"] and "## Files\n" in starts[0]["prompt"]
          and "## Files and scripts" not in starts[0]["prompt"] and srv.url in starts[0]["prompt"] and "<!-- folder -->" not in starts[0]["prompt"],
          "its folder, and the prompt with the work-folder section (no shell) and this server")
    w_ = str(sc.work_dir()).rstrip("/")
    tools_ = argv[argv.index("--tools") + 1 : argv.index("--allowedTools")]
    check(tools_ == ["Read", "Write", "Edit"] and "--add-dir" not in argv and argv[argv.index("--disallowedTools") + 1] == "Bash"
          and all(f"{t}(/{w_}/**)" in argv for t in ("Read", "Write", "Edit")) and "--setting-sources" in argv,
          "by default its files only in its work folder, no shell, no other folder added, the person's settings not loaded")
    old_reach = os.environ.get(sc.REACH_ENV)
    os.environ[sc.REACH_ENV] = "home"
    try:
        home_cmd = sc._claude_cmd(str(uuid.uuid4()), srv.url)
        check("Bash" in home_cmd[home_cmd.index("--tools") + 1 : home_cmd.index("--allowedTools")] and home_cmd[home_cmd.index("--add-dir") + 1] == os.path.expanduser("~")
              and "## Files and scripts" in sc.system_prompt("claude", srv.url) and "<!-- shell -->" not in sc.system_prompt("claude", srv.url),
              "under THEODOLITE_SCIENTIST_REACH=home: Bash and the home folder, and the shell section")
        check(hc.get("/api/atlas/scientist/backends").json()["reach"]["mode"] == "home", "/backends says the reach is home")
    finally:
        if old_reach is None:
            os.environ.pop(sc.REACH_ENV, None)
        else:
            os.environ[sc.REACH_ENV] = old_reach
    bk = hc.get("/api/atlas/scientist/backends").json()["reach"]
    check(bk["mode"] == "folder" and "no shell" in bk["claude"] and "sandbox" in bk["codex"], f"/backends says what each backend can do: {bk}")
    os.environ["OPENROUTER_API_KEY"] = "sk-or-test"
    env_ = sc.child_env()
    check("OPENROUTER_API_KEY" not in env_ and "THEODOLITE_OPENAI_API_KEY" not in env_ and "PATH" in env_, "the endpoints' keys are not passed to Claude Code or Codex")
    os.environ.pop("OPENROUTER_API_KEY", None)
    claude_env, codex_env = sc.child_env("claude"), sc.child_env("codex", CODEX_HOME="h")
    check(not any(k in e for k in SECRETS for e in (claude_env, codex_env)) and all(k in e for k in LOGINS for e in (claude_env, codex_env))
          and codex_env["CODEX_HOME"] == "h", "the Hugging Face and cloud credentials kept from both, each CLI's own login left in")
    check(starts[0]["env"] == sorted(LOGINS), f"the Claude Code process started with its login and none of them: {starts[0]['env']}")
    os.environ["CLAUDE_CODE_USE_BEDROCK"] = "1"
    try:
        aws = sorted(k for k in SECRETS if k.startswith("AWS_"))
        check(sorted(k for k in SECRETS if k in sc.child_env("claude")) == aws and not any(k in sc.child_env("codex") for k in SECRETS),
              "on Bedrock, Claude Code keeps the AWS variables it logs in with; Codex still gets none")
        os.environ["CLAUDE_CODE_USE_BEDROCK"] = "0"
        check(not any(k in sc.child_env("claude") for k in SECRETS), "CLAUDE_CODE_USE_BEDROCK=0 is not Bedrock")
    finally:
        os.environ.pop("CLAUDE_CODE_USE_BEDROCK", None)
    tools_seen = next(x for x in flog() if x["kind"] == "tools")["names"]
    check(tools_seen == names, "the MCP server it started lists the scientist's tools")
    user = next(x for x in flog() if x["kind"] == "user")
    check(user["text"] == "[The person has L3 N5 open in the atlas.]\n\nWhat does this one do?", f"what the person had open, in the pages' count: {user['text']!r}")
    neuron_text = json.loads(next(x for x in flog() if x["kind"] == "neuron")["text"])
    check(neuron_text["neuron"] == "L3 N5" and "faces" in neuron_text, "a read tool through the MCP server it started")
    # attachments
    up_txt = hc.post("/api/atlas/scientist/upload", json={"name": "../notes.txt", "type": "text/plain", "data": "aGVsbG8gdGhlcmU="}).json()
    up_png = hc.post("/api/atlas/scientist/upload", json={"name": "plot.png", "type": "image/png", "data": "iVBORw0KGgo="}).json()
    check(up_txt["name"] == "notes.txt" and len(up_txt["id"]) == 12 and (sc.uploads_dir() / f"{up_txt['id']}-notes.txt").read_text() == "hello there",
          f"an upload kept, its name stripped of any folder: {up_txt}")
    check(hc.post("/api/atlas/scientist/upload", json={"name": "x", "data": "!!!"}).status_code == 400, "an upload that is not base64 refused")
    i1 = ev.mark()
    hc.post("/api/atlas/scientist/send", json={"chat": chat, "message": "And with these?", "attachments": [up_txt["id"], up_png["id"], "../../etc"]})
    ev.wait(lambda e: e["t"] == "done", 90, i1)
    user2 = [x for x in flog() if x["kind"] == "user"][-1]
    check(user2["blocks"] == ["text", "text", "image"] and user2["kinds"][2] == "image/png", f"attachments as content blocks: {user2['blocks']}")
    check(len([x for x in flog() if x["kind"] == "start"]) == 1, "one process for the chat, kept open between turns")
    # a message sent mid-turn, then an interrupt
    i2 = ev.mark()
    hc.post("/api/atlas/scientist/send", json={"chat": chat, "message": "Go slow, please."})
    ev.wait(lambda e: e["t"] == "text", 30, i2)
    hc.post("/api/atlas/scientist/send", json={"chat": chat, "message": "Also look at N6."})
    time.sleep(0.3)
    r = hc.post("/api/atlas/scientist/interrupt", json={"chat": chat}).json()
    done = ev.wait(lambda e: e["t"] == "done", 30, i2)
    check(r == {"ok": True, "running": True} and done is not None, f"an interrupt ends the turn: {r}")
    check(any(x["kind"] == "midturn" and x["text"] == "Also look at N6." for x in flog()) and any(x["kind"] == "interrupt" for x in flog()),
          "a message sent mid-turn reached it, then the interrupt")
    check(not any(e["t"] == "error" for e in ev.since(i2)), "an interrupted turn is not an error")
    check(hc.post("/api/atlas/scientist/interrupt", json={"chat": chat}).json() == {"ok": True, "running": False}, "nothing to interrupt when idle")
    # past chats and history
    lst = hc.get("/api/atlas/scientist/chats").json()
    mine = next((x for x in lst if x["chat"] == chat), None)
    check(mine is not None and mine["agent"] == "claude" and mine["title"] == "What does this one do?" and mine["turns"] == 3, f"past chats: {mine}")
    hist = hc.get("/api/atlas/scientist/history", params={"chat": chat}).json()["events"]
    check([e["t"] for e in hist][:5] == ["user", "text", "tool", "tool_result", "text"] and hist[0]["text"] == "What does this one do?",
          f"history from the session file, without the note: {[e['t'] for e in hist][:6]}")
    # rewind: cut back to before the last message; the next message resumes the cut session
    r = hc.post("/api/atlas/scientist/rewind", json={"chat": chat, "text": "Go slow, please."}).json()
    hist2 = hc.get("/api/atlas/scientist/history", params={"chat": chat}).json()["events"]
    session = Path(sc._session_file(chat))
    check(r == {"ok": True, "cut": True} and not any(e.get("text") == "Go slow, please." for e in hist2) and list(session.parent.glob(f"{chat}.jsonl.bak-*")),
          "a rewind cuts the session (backed up first)")
    i3 = ev.mark()
    hc.post("/api/atlas/scientist/send", json={"chat": chat, "message": "Once more."})
    ev.wait(lambda e: e["t"] == "done", 90, i3)
    starts = [x for x in flog() if x["kind"] == "start"]
    check(len(starts) == 2 and starts[1]["resumed"] and starts[1]["sid"] == chat, "after a rewind the session is resumed")
    ev.stop = True
    section("Claude Code: the command, a turn, attachments, mid-turn, interrupt, history, rewind", before)

    # ------------------------------------------------------------------------------------------------- 5. Codex
    before = len(failures)
    codex_log = TMP / "fake_codex.log"
    os.environ["FAKE_LOG"] = str(codex_log)
    os.environ["THEODOLITE_CODEX_BIN"] = str(write_exec(TMP / "fake-codex", FAKE_CODEX.replace("{PY}", PY)))
    (TMP / "home" / ".codex").mkdir()
    (TMP / "home" / ".codex" / "config.toml").write_text('model = "gpt-test"\nmodel_reasoning_effort = "high"\n[mcp_servers.other]\ncommand = "x"\n')
    (TMP / "home" / ".codex" / "auth.json").write_text("{}")

    def clog() -> List[Dict[str, Any]]:
        return [json.loads(x) for x in codex_log.read_text().splitlines()] if codex_log.exists() else []

    chat2 = str(uuid.uuid4())
    ev = Events(srv.url, chat2)
    i0 = ev.mark()
    r = hc.post("/api/atlas/scientist/send", json={"chat": chat2, "message": "Read L3 N5.", "agent": "codex"})
    done = ev.wait(lambda e: e["t"] == "done", 60, i0)
    seq = ev.since(i0)
    text = "".join(e["text"] for e in seq if e["t"] == "text")
    check(r.json() == {"ok": True} and done is not None and text == "Reading the neuron.Done.", f"a codex turn: {text!r}")
    tool = next(e for e in seq if e["t"] == "tool")
    res = next(e for e in seq if e["t"] == "tool_result")
    check(tool["name"] == "neuron" and tool["input"] == {"layer": 3, "n": 5} and "L3 N5" in res["output"] and not res["error"], "its tool and result")
    rpcs = [x for x in clog() if x["kind"] == "rpc"]
    ts = next(x for x in rpcs if x["method"] == "thread/start")["params"]
    check(ts["sandbox"] == "workspace-write" and "writes only in" in ts["developerInstructions"], "codex by default: its shell sandboxed to its work folder")
    check(ts["config"]["mcp_servers"]["atlas"].get("default_tools_approval_mode") == "approve",
          "codex: the atlas's tools approved (in its sandbox Codex otherwise asks before each, and the chat never asks)")
    check(ts["approvalPolicy"] == "never" and ts["config"]["mcp_servers"]["atlas"]["env"] == {"THEODOLITE_URL": srv.url, "THEODOLITE_CHAT": chat2}
          and "You are running as Codex" in ts["developerInstructions"] and ts["cwd"] == str(sc.work_dir()), "the thread: no approvals, the atlas's MCP server, its instructions")
    home = Path(next(x for x in clog() if x["kind"] == "start")["codex_home"])
    check(next(x for x in clog() if x["kind"] == "start")["env"] == sorted(LOGINS), "the Codex process started with its login and none of the credentials")
    cfg = (home / "config.toml").read_text()
    check('model = "gpt-test"' in cfg and "other" not in cfg and "memories = false" in cfg and (home / "auth.json").is_symlink(),
          "codex's own home: the person's login, model and effort, nothing else")
    reply = next(x for x in clog() if x["kind"] == "reply")["msg"]
    check(reply["id"] == 901 and reply["result"] == {"decision": "decline"}, f"an approval asked for is declined: {reply}")
    meta = json.loads((sc.scientist_dir() / "chats.json").read_text())[chat2]
    check(meta["agent"] == "codex" and meta["thread"] == "thr_1" and meta["turns"] == ["turn_1"], f"the chat's thread and turns kept: {meta}")
    hist = hc.get("/api/atlas/scientist/history", params={"chat": chat2}).json()
    check(hist["agent"] == "codex" and [e["t"] for e in hist["events"]][:2] == ["user", "text"] and hist["events"][0]["text"] == "Read L3 N5.", "history from the log")
    r = hc.post("/api/atlas/scientist/rewind", json={"chat": chat2}).json()
    rev = [x for x in clog() if x["kind"] == "rpc" and x["method"] == "thread/revert"]
    check(r == {"ok": True, "cut": True} and rev and rev[-1]["params"] == {"threadId": "thr_1", "beforeTurnId": "turn_1"}, "a rewind reverts the thread's last turn")
    check(hc.get("/api/atlas/scientist/history", params={"chat": chat2}).json()["events"] == [] and
          json.loads((sc.scientist_dir() / "chats.json").read_text())[chat2]["turns"] == [], "and cuts the log")
    ev.stop = True
    section("Codex: its home, the thread, a turn, an approval declined, history, rewind", before)

    # --------------------------------------------------------------------------- 6. an OpenAI-compatible endpoint
    before = len(failures)
    seen: List[Dict[str, Any]] = []
    fake = Served(fake_openai_app(seen))
    os.environ.update({"THEODOLITE_OPENAI_BASE_URL": f"{fake.url}/v1", "THEODOLITE_OPENAI_API_KEY": "sk-test-key", "THEODOLITE_OPENAI_MODEL": "fake-model"})
    bk = hc.get("/api/atlas/scientist/backends").json()
    custom = next(p for p in bk["openai"] if p["name"] == "custom")
    check(custom["base_url"] == f"{fake.url}/v1" and custom["key"] == "set" and custom["model"] == "fake-model" and custom["this_machine"],
          f"the custom endpoint from the environment: {custom}")
    check("sk-test-key" not in json.dumps(bk), "no key in what a page is sent")
    md = hc.get("/api/atlas/scientist/models", params={"profile": "custom"}).json()
    check([m["id"] for m in md["models"]] == ["fake-model", "no-tools", "silent"] and [m["tools"] for m in md["models"]] == [True, False, None]
          and seen[-1]["auth"] == "Bearer sk-test-key", f"its models, whether each takes tools: {md}")
    check(hc.get("/api/atlas/scientist/models", params={"profile": "nope"}).status_code == 404, "an endpoint not configured")
    chat3 = str(uuid.uuid4())
    ev = Events(srv.url, chat3)
    i0 = ev.mark()
    r = hc.post("/api/atlas/scientist/send", json={"chat": chat3, "message": "Explain it.", "agent": "openai", "profile": "custom", "layer": 2, "n": 5})
    done = ev.wait(lambda e: e["t"] == "done", 90, i0)
    seq = ev.since(i0)
    kinds = [e["t"] for e in seq]
    check(r.json() == {"ok": True} and done is not None and not any(e["t"] == "error" for e in seq), f"a turn through the endpoint: {kinds} {[e for e in seq if e['t'] == 'error']}")
    check("thinking" in kinds and "".join(e["text"] for e in seq if e["t"] == "text") == "Let me look.It writes `Bob`.", "its thinking and its text, streamed")
    tools_run = [e for e in seq if e["t"] == "tool"]
    check([t["name"] for t in tools_run] == ["neuron", "select_neuron"] and tools_run[0]["input"] == {"layer": 3, "n": 5}, f"its tool calls, streamed in pieces: {tools_run}")
    ui = next(e for e in seq if e["t"] == "ui")
    check(ui == {"t": "ui", "action": "select", "layer": 2, "n": 5}, f"a screen action from its tool: {ui}")
    reqs = [s for s in seen if s["path"] == "/v1/chat/completions"]
    first, second = reqs[-2]["body"], reqs[-1]["body"]
    check(first["model"] == "fake-model" and first["tool_choice"] == "auto" and first["stream"] is True
          and [t["function"]["name"] for t in first["tools"]] == names and reqs[-1]["auth"] == "Bearer sk-test-key", "the request: the model, every tool, the key")
    sysmsg = first["messages"][0]
    check(sysmsg["role"] == "system" and "OpenAI-compatible endpoint" in sysmsg["content"] and "## Files and scripts" not in sysmsg["content"],
          "the system prompt, without the shell section")
    check(first["messages"][1] == {"role": "user", "content": "[The person has L3 N5 open in the atlas.]\n\nExplain it."}, "what the person had open")
    tmsgs = [m for m in second["messages"] if m["role"] == "tool"]
    asst = next(m for m in second["messages"] if m["role"] == "assistant")
    check([m["tool_call_id"] for m in tmsgs] == ["call_a", "call_b"] and [c["id"] for c in asst["tool_calls"]] == ["call_a", "call_b"]
          and json.loads(asst["tool_calls"][0]["function"]["arguments"]) == {"layer": 3, "n": 5}, "each call answered by its id, arguments joined")
    tres = json.loads(tmsgs[0]["content"])
    check(tres["neuron"] == "L3 N5" and tres["faces"]["+"]["label"].startswith("held back"), "the atlas's tool ran in this process, its reply in the pages' count")
    # arguments that are not JSON
    i1 = ev.mark()
    hc.post("/api/atlas/scientist/send", json={"chat": chat3, "message": "Now bad args."})
    ev.wait(lambda e: e["t"] == "done", 60, i1)
    bad = next(e for e in ev.since(i1) if e["t"] == "tool_result")
    check(bad["error"] and "were not JSON" in bad["output"], f"arguments that are not JSON answered in plain words: {bad}")
    # a message sent mid-turn joins the same turn
    chat4 = str(uuid.uuid4())
    ev4 = Events(srv.url, chat4)
    j0 = ev4.mark()
    hc.post("/api/atlas/scientist/send", json={"chat": chat4, "message": "queue this", "agent": "openai", "profile": "custom"})
    ev4.wait(lambda e: e["t"] == "text", 60, j0)
    r = hc.post("/api/atlas/scientist/send", json={"chat": chat4, "message": "and also N10"}).json()
    ev4.wait(lambda e: e["t"] == "done", 60, j0)
    reqs = [s["body"] for s in seen if s["path"] == "/v1/chat/completions"][-3:]
    last_msgs = reqs[-1]["messages"]
    check(r == {"ok": True, "queued": True} and last_msgs[-1] == {"role": "user", "content": "and also N10"}
          and last_msgs[-2]["role"] == "assistant" and "slowly" in (last_msgs[-2]["content"] or ""), "a message sent mid-turn answered in the same turn")
    acts = next(e for e in ev4.since(j0) if e["t"] == "tool_result")
    check(not acts["error"] and '"tokens"' in acts["output"], "a live tool (activations) through the loop")
    ev4.stop = True
    # an interrupt
    i3 = ev.mark()
    hc.post("/api/atlas/scientist/send", json={"chat": chat3, "message": "slow down"})
    ev.wait(lambda e: e["t"] == "text", 30, i3)
    r = hc.post("/api/atlas/scientist/interrupt", json={"chat": chat3}).json()
    t0 = time.time()
    done = ev.wait(lambda e: e["t"] == "done", 30, i3)
    check(r == {"ok": True, "running": True} and done is not None and time.time() - t0 < 5, "an interrupt stops the reply")
    msgs = json.loads((sc.logs_dir() / f"{chat3}.messages.json").read_text())
    check(msgs[-1]["role"] == "assistant" and msgs[-1]["content"].startswith("0 ") and "tool_calls" not in msgs[-1], "the part written is kept, the conversation valid")
    i4 = ev.mark()
    hc.post("/api/atlas/scientist/send", json={"chat": chat3, "message": "carry on"})
    ev.wait(lambda e: e["t"] == "done", 60, i4)
    check(not any(e["t"] == "error" for e in ev.since(i4)), "the next turn goes on from it")
    # errors in plain words
    i5 = ev.mark()
    hc.post("/api/atlas/scientist/send", json={"chat": chat3, "message": "fail now"})
    ev.wait(lambda e: e["t"] == "done", 60, i5)
    err = next((e for e in ev.since(i5) if e["t"] == "error"), None)
    check(err is not None and "answered 500" in err["error"] and "fell over" in err["error"], f"an endpoint's error: {err}")
    ev.stop = True
    # a profile without streaming, from the profile file; a key held in the file refused
    (sc.scientist_dir() / "endpoints.json").write_text(json.dumps({
        "plain": {"label": "no streaming", "base_url": f"{fake.url}/v1", "model": "fake-model", "stream": False},
        "keyed": {"base_url": f"{fake.url}/v1", "key_env": "NO_SUCH_KEY_VAR", "model": "fake-model"},
        "leaky": {"base_url": f"{fake.url}/v1", "api_key": "sk-in-file"}}))
    bk = hc.get("/api/atlas/scientist/backends").json()
    names_p = [p["name"] for p in bk["openai"]]
    check("plain" in names_p and "keyed" in names_p and "leaky" not in names_p and "holds a key" in (bk["warning"] or ""), f"the profile file: {names_p} {bk['warning']}")
    chat5 = str(uuid.uuid4())
    ev = Events(srv.url, chat5)
    i0 = ev.mark()
    hc.post("/api/atlas/scientist/send", json={"chat": chat5, "message": "Any names?", "agent": "openai", "profile": "plain"})
    ev.wait(lambda e: e["t"] == "done", 60, i0)
    seq = ev.since(i0)
    check([e["t"] for e in seq if e["t"] in ("tool", "tool_result", "text")] == ["tool", "tool_result", "text"]
          and next(e for e in seq if e["t"] == "text")["text"] == "No names yet." and seen[-1]["body"]["stream"] is False, "a profile without streaming")
    chat6 = str(uuid.uuid4())
    ev6 = Events(srv.url, chat6)
    j0 = ev6.mark()
    hc.post("/api/atlas/scientist/send", json={"chat": chat6, "message": "hello", "agent": "openai", "profile": "keyed"})
    ev6.wait(lambda e: e["t"] == "done", 30, j0)
    err = next((e for e in ev6.since(j0) if e["t"] == "error"), None)
    check(err is not None and "$NO_SUCH_KEY_VAR" in err["error"], f"a missing key said in plain words: {err}")
    ev6.stop = True
    r = hc.post("/api/atlas/scientist/send", json={"chat": str(uuid.uuid4()), "message": "x", "agent": "openai", "profile": "nope"})
    check(r.status_code == 400 and "Pick an endpoint" in r.json()["detail"], "an endpoint not configured refused")
    check(hc.post("/api/atlas/scientist/send", json={"chat": "not-a-uuid", "message": "x"}).status_code == 400, "a chat's id must be a UUID")
    # history, chats, rewind
    hist = hc.get("/api/atlas/scientist/history", params={"chat": chat3}).json()
    check(hist["agent"] == "openai" and hist["profile"] == "custom" and hist["events"][0] == {"t": "user", "text": "Explain it."}, "history of an openai chat")
    check(not any(a_["t"] == b_["t"] == "text" for a_, b_ in zip(hist["events"], hist["events"][1:])),
          "history: a reply's streamed fragments joined, as the live page shows them (a split cannot fall inside a citation)")
    row = next(x for x in hc.get("/api/atlas/scientist/chats").json() if x["chat"] == chat3)
    check(row["agent"] == "openai" and row["model"] == "fake-model" and row["turns"] == 5, f"past chats: {row}")
    n_before = len(json.loads((sc.logs_dir() / f"{chat3}.messages.json").read_text()))
    r = hc.post("/api/atlas/scientist/rewind", json={"chat": chat3}).json()
    msgs = json.loads((sc.logs_dir() / f"{chat3}.messages.json").read_text())
    users = [m for m in msgs if m["role"] == "user"]
    check(r["cut"] and len(msgs) < n_before and users[-1]["content"] == "carry on" and
          not any(e.get("text") == "fail now" for e in hc.get("/api/atlas/scientist/history", params={"chat": chat3}).json()["events"]),
          "a rewind drops the last message and what came after")
    ev.stop = True
    fake.stop()
    section("an OpenAI-compatible endpoint: the tool loop, mid-turn, interrupt, errors, profiles, history", before)

    # ----------------------------------------------------------------------------------------------- 7. who may ask
    before = len(failures)
    for hdrs in ({"Origin": "http://evil.example"}, {"Sec-Fetch-Site": "cross-site"}):
        r = hc.post("/api/atlas/scientist/send", json={"chat": chat, "message": "x"}, headers=hdrs)
        check(r.status_code == 403, f"send from {hdrs}: {r.status_code}")
    r = hc.get("/api/atlas/scientist/backends", headers={"Host": "evil.example"})
    check(r.status_code == 403, "a page served from another machine")
    section("who may ask", before)

    hc.close()
    for c in list(sc._chats.values()):
        p = c.get("proc")
        if p is not None and p.returncode is None:
            try:
                os.killpg(p.pid, 9)
            except (ProcessLookupError, PermissionError):
                pass
    srv.stop()
    print(f"\n{passed} checks passed, {len(failures)} failed")
    return 1 if failures else 0


async def tool_checks(api: Any, hc: httpx.Client) -> None:
    """Each tool in the pages' count against the same route read in the API's count."""
    ov = await st.call("overview", {}, api)
    check(ov["layers"] == 6 and ov["neurons_per_layer"] == 128 and ov["attention"]["full_layers"] == [6] and ov["attention"]["sliding_window"] == 8,
          f"overview: {ov['layers']} layers, full {ov['attention']['full_layers']}")
    check("held back below page layer 41" in ov["label_gate"]["in_page_layers"] and ov["lens"]["id"] == "jpp-base", "overview: the gate in page layers, the lens")
    nv = await st.call("neuron", {"layer": 3, "n": 5}, api)
    raw = hc.get("/api/atlas/neuron", params={"layer": 2, "n": 5}).json()
    check(nv["neuron"] == "L3 N5" and nv["faces"]["+"]["lens"] == raw["faces"]["+"]["lens"] and nv["faces"]["+"]["label"].startswith("held back: below_layer_40"),
          "neuron: page layer 3 is the API's 2, each face with the gate's verdict")
    check([x["neuron"] for x in nv["writes_into"]] == [st.cite(x["layer"], x["n"]) for x in raw["writes_into"]], "neuron: its links cited in the pages' count")
    act = await st.call("activations", {"layer": 4, "n": 9, "text": "Alice called Bob", "neurons": [{"layer": 2, "n": 3, "mode": "mul", "dose": 0}]}, api)
    raw = hc.post("/api/atlas/reads", json={"layer": 3, "n": 9, "text": "Alice called Bob", "neurons": [{"layer": 1, "n": 3, "mode": "mul", "dose": 0}]}).json()
    check([t for t, _ in act["tokens"]] == [u["text"] for u in raw["units"]] and act["pushed"] == [{"neuron": "L2 N3", "mode": "mul", "dose": 0.0}],
          f"activations: the reads, the push cited in the pages' count ({act['pushed']})")
    stv = await st.call("steer", {"layer": 3, "n": 5, "prompt": "Who are you?", "max_new": 6}, api)
    check(stv["pushed"] == [{"neuron": "L3 N5", "mode": "add", "dose": 0.8}] and isinstance(stv["up"], str), "steer")
    lp = await st.call("logprobs", {"text": "Who are you?", "k": 3, "neurons": [{"layer": 5, "n": 9, "dose": 2.0}]}, api)
    check(lp["kl"] >= 0 and lp["pushed"] == [{"neuron": "L5 N9", "mode": "add", "dose": 2.0}], "logprobs with a push")
    cmp_ = await st.call("compare", {"neurons": [{"layer": 1, "n": 1}, {"layer": 6, "n": 127}], "texts": ["hello there", "def merge(a, b):"]}, api)
    check([x["neuron"] for x in cmp_["neurons"]] == ["L1 N1", "L6 N127"] and set(cmp_["neurons"][0]["by_text"]) == {"t1", "t2"}, "compare")
    tf = await st.call("top_firing", {"text": CORPUS[4], "limit": 5, "layers": [2, 3]}, api)
    check(len(tf["neurons"]) == 5 and all(x["neuron"][:3] in ("L2 ", "L3 ") for x in tf["neurons"]), f"top_firing in page layers 2-3: {[x['neuron'] for x in tf['neurons']]}")
    at = await st.call("attention", {"layer": 6, "head": 1, "text": CORPUS[4], "token": "Bob"}, api)
    check(at["head"] == "L6 h1" and at["kind"] == "full" and "Bob" in at["query"]["token"], "attention on the full layer (page layer 6)")
    dr = await st.call("attn_drive", {"layer": 5, "n": 9, "text": CORPUS[4], "token": "Bob"}, api)
    check(dr["neuron"] == "L5 N9" and all(h["head"][:2] in ("L1", "L2", "L3", "L4", "L5") for h in dr["heads"]) and dr["blocks"][0]["block"].endswith("feed-forward"),
          "attn_drive: heads and blocks in page layers")
    hd = await st.call("head", {"layer": 4, "head": 1}, api)
    check(hd["head"] == "L4 h1" and all(x["neuron"].startswith("L") for r_ in ("q", "k", "v", "o") for x in hd[r_]["neurons"]) and "profile" in hd, "head")
    nh = await st.call("neuron_heads", {"layer": 3, "n": 5}, api)
    check(all(x["head"].startswith("L") for r_ in nh.values() for x in r_), "neuron_heads")
    sl = await st.call("steer_lens", {"layer": 3, "n": 5}, api)
    check(sl["neuron"] == "L3 N5" and len(sl["up"]) == 12, "steer_lens, read live")
    try:
        await st.call("neuron", {"layer": 0, "n": 5}, api)
        check(False, "page layer 0 refused")
    except st.ToolError as exc:
        check("count from 1" in str(exc), "page layer 0 refused in plain words")
    try:
        await st.call("activations", {"layer": 3, "n": 5}, api)
        check(False, "a missing argument refused")
    except st.ToolError as exc:
        check("needs text" in str(exc), f"a missing argument: {exc}")
    try:
        await st.call("neuron", {"layer": 3, "n": 999}, api)
        check(False, "a neuron that does not exist refused")
    except st.ToolError as exc:
        check("400" in str(exc) and "Neuron 999 does not exist" in str(exc), f"the server's refusal passed on: {exc}")
    try:
        cc_ = await st.call("causal_check", {"layer": 6, "n": 5, "prompt": "Who are you?", "k": 4}, api)
        check(isinstance(cc_, dict) and "lens" in cc_, "causal_check: the instrument's own record")
    except st.ToolError as exc:
        check("does not exist" in str(exc) or "400" in str(exc), f"causal_check refused in plain words on the stand-in: {exc}")
    # names and notes: neither overwrites the other
    r = await st.call("set_name", {"layer": 3, "n": 5, "name": "bob watcher (?)"}, api)
    await api.post("/api/atlas/notes", {"kind": "neuron", "layer": 2, "index": 5, "note": "the person's own note"})
    r = await st.call("set_name", {"layer": 3, "n": 5, "name": "bob watcher (??)"}, api)
    check(r == {"ok": True, "neuron": "L3 N5", "name": "bob watcher (??)", "note": "the person's own note"}, f"set_name keeps the person's note: {r}")
    r = await st.call("write_note", {"layer": 3, "n": 5, "note": "steers toward Bob at +2"}, api)
    check(r["name"] == "bob watcher (??)" and r["note"] == "steers toward Bob at +2", "write_note keeps the name")
    nm = await st.call("names", {}, api)
    check(nm["neurons"] == {"L3 N5": "bob watcher (??)"} and nm["named_by_a_scientist"] == ["L3 N5"], f"names in the pages' count: {nm}")
    notes = json.loads(Path(os.environ["THEODOLITE_FIELD_NOTES"]).read_text())
    check(notes["neurons"]["2:5"]["by"] == "scientist" and notes["neurons"]["2:5"]["name_by"] == "scientist", "kept under the API's key, by the scientist")
    # each field keeps its author: the person's note under the scientist's name leaves the name the scientist's
    await api.post("/api/atlas/notes", {"kind": "neuron", "layer": 2, "index": 5, "note": "the person checks it", "by": "reader"})
    nr = await st.call("neuron", {"layer": 3, "n": 5}, api)
    check(nr["name"] == "bob watcher (??)" and nr["name_by"] == "scientist" and nr["note_by"] == "reader",
          f"a note by the person leaves the name the scientist's, and the tools say whose each is ({nr['name_by']}, {nr['note_by']})")
    # the screen: actions reach the chat's page in the API's count
    cid = str(uuid.uuid4())
    c = sc._chat(cid, "claude")
    q: asyncio.Queue = asyncio.Queue()
    c["listeners"].add(q)
    for tool, args, want in [
        ("select_neuron", {"layer": 3, "n": 5}, {"action": "select", "layer": 2, "n": 5}),
        ("select_head", {"layer": 6, "head": 2}, {"action": "head", "layer": 5, "head": 2}),
        ("camera", {"mode": "layer", "layer": 1}, {"action": "camera", "mode": "layer", "layer": 0}),
        ("highlight_neurons", {"neurons": [{"layer": 2, "n": 1, "face": "-"}]}, {"action": "highlight", "neurons": [{"layer": 1, "n": 1, "face": "-"}]}),
        ("annotate_neurons", {"items": [{"layer": 6, "n": 4, "text": "feeds L3 N5"}]}, {"action": "annotate", "items": [{"layer": 5, "n": 4, "text": "feeds L3 N5"}]}),
        ("show_text", {"text": "Alice called Bob", "token": "Bob"}, {"action": "text", "text": "Alice called Bob", "token": "Bob"}),
        ("show_search", {"query": "bob"}, {"action": "search", "query": "bob"}),
        ("open_tab", {"tab": "census"}, {"action": "tab", "tab": "census"}),
    ]:
        r = await st.call(tool, args, api, cid)
        got = await asyncio.wait_for(q.get(), 5)
        check(r["ok"] and got == {"t": "ui", **want}, f"{tool}: {got}")
    r = await st.call("set_name", {"layer": 3, "n": 6, "name": "next one"}, api, cid)
    got = await asyncio.wait_for(q.get(), 5)
    check(got == {"t": "ui", "action": "rename", "layer": 2, "n": 6, "name": "next one"}, f"set_name renames on the page: {got}")
    r = await st.call("select_neuron", {"layer": 3, "n": 5}, api, "no-such-chat")
    check(r["ok"] is False and "no such chat" in r["note"], "a screen action for a chat not open")
    try:
        await st.call("camera", {"mode": "neuron", "layer": 3}, api, cid)
        check(False, "camera neuron needs n")
    except st.ToolError as exc:
        check("needs n" in str(exc), "camera neuron needs n")
    sc._chats.pop(cid, None)
    # the indexes: without them the cone and the web say so; with them they answer in the pages' count
    cn = await st.call("cone", {"layer": 3, "n": 5}, api)
    ws = await st.call("web_summary", {"layer": 3, "n": 5}, api)
    sr = await st.call("search_lens", {"query": "the"}, api)
    pg = await st.call("passages", {"layer": 3, "n": 5}, api)
    check(cn["ready"] is False and ws["ready"] is False and sr["ready"] is False and pg["ready"] is False and "atlas_build.py passages" in pg["note"],
          "without the indexes: the cone, the web, the search and the passages say so")
    from backend.atlas import atlas as atlas_state

    a_ = atlas_state()
    out = a_.index.root
    out.mkdir(parents=True, exist_ok=True)
    ab.build_lens(a_.rt, out, chunk=48, log=quiet)
    ab.build_stats(a_.rt, out, ab.windows(a_.rt, CORPUS, 24, 300), "check", log=quiet)
    stats = a_.index.get("act_scale")
    ab.build_wiring(a_.rt, out, a_.link_scale(), log=quiet)
    ab.build_steer_lens(a_.rt, out, [float(x) for x in stats["resid_q"][:, 1]], batch=16, log=quiet)
    ab.build_passages(a_.rt, out, ab.windows(a_.rt, CORPUS, 24, 300), "check", k=6, sample=2, extremes=(stats["pos"], stats["neg"]),
                      extremes_source="the stats index", log=quiet)
    pg = await st.call("passages", {"layer": 3, "n": 5, "face": "-", "k": 4}, api)
    raw = hc.get("/api/atlas/neuron/passages", params={"layer": 2, "n": 5, "face": "-", "k": 4}).json()
    check(pg["neuron"] == "L3 N5" and pg["face"] == "-" and [x["text"] for x in pg["passages"]]
          == ["".join(f"[[{t}]]" if pk else t for t, pk in x["units"]) for x in raw["passages"]]
          and sum(1 for x in pg["passages"] if x["kind"] == "top") == 4 and all(x["text"].count("[[") == 1 for x in pg["passages"]),
          "passages: page layer 3 is the API's 2, one face's passages with the peak marked")
    check(pg["reads"] == [f"{t['tok']} ({t['n']}, lift {t['lift']})" for t in raw["reads"]] and "not validated as a label" in pg["note"]
          and (await st.call("passages", {"layer": 3, "n": 5}, api))["face"] == "+", "passages: reads with their lift, the note, the + face by default")
    cn = await st.call("cone", {"layer": 3, "n": 5}, api)
    raw = hc.get("/api/atlas/cone", params={"layer": 2, "n": 5}).json()
    check(cn["neurons"] == raw["neurons"] and [x["neuron"] for x in cn["nodes"]] == [st.cite(x["layer"], x["n"]) for x in raw["nodes"]], "cone in the pages' count")
    ws = await st.call("web_summary", {"layer": 3, "n": 5}, api)
    check(ws["pooled"] == "steer" and isinstance(ws["downstream"]["when_plus"]["words"], list) and set(ws["upstream"]) == {"drives_plus", "drives_minus"},
          "web_summary pooled from the steer lens, by face each way")
    check(all(("when" in x) == (x["hop"] > 0) and ("drives" in x) == (x["hop"] < 0) and "w" in x for x in cn["nodes"]),
          "cone: a reader with the face the neuron fires on, a writer with the face it drives, each with its strength")
    word = a_.words(a_.index.face_ids(4, 7, "+", "lens", 1))[0].strip()
    sr = await st.call("search_lens", {"query": word, "limit": 2000}, api)
    check(sr["word_pieces_matched"] > 0 and sr["held_back_by_the_gate"] > 0 and not any(x["matched_in"] in ("lens", "eye") for x in sr["results"]),
          f"search_lens by a lens word {word!r}: on the stand-in every lens label is held back, so it only counts them")
    sword = next(w_.strip() for w_ in a_.words(a_.index.get("steer_lens")["up"][4, 7, :6], 6) if len(w_.strip()) >= 2)
    sr = await st.call("search_lens", {"query": sword, "limit": 2000}, api)
    check(any(x["neuron"] == "L5 N7" and x["face"] == "+" and x["matched_in"] == "steer" for x in sr["results"]), f"search_lens finds L5 N7 by its steer lens's {sword!r}")
    # a census of API layer 3 is page layer 4's
    cdir = cz.census_dir()
    cdir.mkdir(parents=True, exist_ok=True)
    run = cdir / "L3-main_rel.jsonl"
    rows = [{"layer": 3, "kind": "baseline", "neuron": -1, "prompt": "who", "dose": 0.0, "push": 0.0, "rel_push": 0.0, "text": "I am a model.",
             "finish": "stop", "n_gen": 4, "token_ids": [1, 2, 3, 4], "logprob": [-1.0] * 4, "top_ids": [], "top_lps": []}]
    for n_, w_ in ((7, "ocean waves"), (9, "banana bread")):
        for d in (0.5, 0.8):
            rows.append({**rows[0], "kind": "neuron", "neuron": n_, "dose": d, "rel_push": d, "text": f"{w_} and {w_} again " * 30})
    run.write_text("".join(json.dumps(x) + "\n" for x in rows))
    cz.meta_path_of(run).write_text(json.dumps({"layer": 3, "design_name": "main_rel", "prompts": cz.PROMPTS, "version": cz.CENSUS_VERSION,
                                                "doses_from": cz.DOSES_FROM["main_rel"]}))
    tu = await st.call("top_units", {"layer": 4}, api)
    check([x["neuron"] for x in tu["ranked"]] in (["L4 N7", "L4 N9"], ["L4 N9", "L4 N7"]), f"top_units of page layer 4: {tu}")
    check(tu["doses_from"] == cz.DOSES_FROM["main_rel"] and tu["null"]["controls"] == 0 and tu["null"]["above_all_by_chance"] is None,
          "top_units: where the doses come from, and that a run without controls sets nothing against chance")
    check((await st.call("top_units", {"layer": 2}, api)) == {"ranked": [], "note": "this layer has no census"}, "top_units: a layer without one")
    ro = await st.call("rollouts", {"layer": 4, "n": 7, "max_chars": 50}, api)
    check(ro["layer"] == 4 and len(ro["rollouts"]) == 2 and all(len(x["text"]) <= 50 for x in ro["rollouts"]), "rollouts cut to max_chars, page layer 4")
    ov = await st.call("overview", {}, api)
    check(ov["census_layers"] == [4] and ov["indexes"]["wiring"] and ov["named"]["neurons"] == 2, f"overview: census at page layer 4, the indexes, the names ({ov['named']})")


if __name__ == "__main__":
    sys.exit(main())

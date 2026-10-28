#!/usr/bin/env python3
"""A CPU-only check of who may ask the instrument's server (backend/local_only.py, around backend/server.py's whole
app) and the lab daemon (experiments/lab/labd.py): no model, no GPU. The server's app is driven with FastAPI's
TestClient without its lifespan, so nothing is loaded and a route that would read the model answers 503; its routes are
found from the app itself (its own, the included routers', FastAPI's docs, and a static mount of pages standing in for
frontend/dist when that is not built). labd's app is imported without being started. Folders and files go to a new
temporary directory:
1. the routes: every path and method of the app, the socket and the pages' mount, each found;
2. a foreign Host (a page rebinding its own name to 127.0.0.1 sends that name, also on a same-origin GET with no
   Origin), on every route and method, an unknown path and the socket: 403 with the reason in plain words, and the
   socket closed (1008) before it is accepted;
3. a foreign Origin with a local Host (another site, a look-alike name, "null"), likewise;
4. a cross-site preflight on every route: 403 and no access-control header; a request a browser marks cross-site or
   same-site without an Origin (an <img>, a <script>, a page on another port fetching this one): 403, unless it is a
   plain navigation (a GET, which reads nothing back);
5. what passes, on every route: a script (no Origin; Host 127.0.0.1, localhost or [::1], or none), the instrument's
   own page (its Origin and Host), vite's proxy (frontend/vite.config.ts: /api with the server's Host, /ws with
   vite's own, 127.0.0.1:5173, and the page's Origin), a link opened from another site; each reaches its route
   (whatever it answers without a model, never the guard's 403), and the socket is accepted and answers;
6. CORS: a local page's preflight is answered with its own origin; no reply to any request above names a foreign one;
   and the CORS layer alone, built from the server's own options (the guard refuses a foreign Origin before CORS sees
   it), allows only this machine's origins, never every origin, and names no foreign one to a preflight or a GET with
   a cookie;
7. on uvicorn (a real socket on a free port): a foreign Origin's WebSocket handshake gets HTTP 403, a local one 101;
8. labd: a foreign Host, and any request a browser sends (an Origin, even a local one, or Sec-Fetch-Site), refused on
   every route (403), no docs pages; /exec and /reload without this launch's token (none, a wrong one, or before any
   launch, "Bearer None" too) refused (401) and the code not run, with it run; the token file written whole, readable
   by its user only (0600), where labd.token_path says; labclient.py sends it (a job run end to end on uvicorn), and
   says in plain words when there is none; a second launch of labd.py on the port that labd is serving stops in plain
   words before its model loads, and leaves the running launch's token file, and its jobs, alone.

Run: .venv/bin/python -I scripts/check_local_only.py   (exits 0 when every check holds)
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"

import re
import shutil
import socket
import stat
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TMP = Path(tempfile.mkdtemp(prefix="local-only-"))
os.environ["THEODOLITE_ATLAS_DIR"] = str(TMP / "atlas")
os.environ["THEODOLITE_FIELD_NOTES"] = str(TMP / "notes.json")
os.environ["THEODOLITE_CENSUS_DIR"] = str(TMP / "census")
os.environ["THEODOLITE_SCIENTIST_DIR"] = str(TMP / "scientist")
os.environ["THEODOLITE_SAVED_PROMPTS"] = str(TMP / "saved_prompts.json")
os.environ["HOME"] = str(TMP / "home")
os.environ["XDG_RUNTIME_DIR"] = str(TMP / "run")
os.environ.pop("LABD_TOKEN_FILE", None)
(TMP / "home").mkdir()

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "experiments" / "lab"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fastapi.routing import APIWebSocketRoute
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient
from starlette.routing import Mount, WebSocketRoute
from starlette.websockets import WebSocketDisconnect

import backend.server as server
from backend.local_only import REFUSED

failures: List[str] = []
passed = 0
BASE = "http://127.0.0.1:8765"  # the Host the server's own pages send (and vite's /api proxy, changeOrigin)
FOREIGN_HOSTS = ["evil.example", "evil.example:8765", "127.0.0.1.nip.io:8765", "localhost.evil.example"]
FOREIGN_ORIGINS = ["https://evil.example", "http://localhost.evil.example:8765", "http://127.0.0.1.evil.example", "null"]
LOCAL_ORIGINS = ["http://127.0.0.1:8765", "http://localhost:5173", "http://[::1]:4173"]


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


def free_port() -> int:
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        return sk.getsockname()[1]


def routes_of(app: Any) -> List[Tuple[str, str, List[str]]]:
    """(kind, path, methods) for every route of the app, an included router's too, whichever way the installed FastAPI
    keeps them (whole in the 0.162 installed here, flattened in older ones)."""
    out: List[Tuple[str, str, List[str]]] = []

    def walk(routes: Any, prefix: str) -> None:
        for r in routes:
            if hasattr(r, "original_router"):
                walk(r.original_router.routes, prefix + getattr(getattr(r, "include_context", None), "prefix", ""))
            elif isinstance(r, Mount):
                out.append(("mount", prefix + r.path, ["GET"]))
            elif isinstance(r, (APIWebSocketRoute, WebSocketRoute)):
                out.append(("socket", prefix + r.path, []))
            else:
                out.append(("http", prefix + r.path, sorted(set(r.methods or ()) - {"HEAD"})))

    walk(app.routes, "")
    return out


def filled(path: str) -> str:
    """A path's parameters given values (a saved prompt's kind, and an id)."""
    return re.sub(r"\{[^}]+\}", "0" * 16, path.replace("{kind}", "question"))


def is_refusal(r: Any) -> bool:
    try:
        return r.status_code == 403 and r.json().get("detail") == REFUSED
    except ValueError:
        return False


def foreign_allowed(r: Any) -> Optional[str]:
    """The origin a reply lets read it, when that is not this machine's."""
    o = r.headers.get("access-control-allow-origin")
    return o if o is not None and not re.fullmatch(r"https?://(127\.0\.0\.1|localhost|\[::1\])(:\d+)?", o) else None


def socket_closed(c: TestClient, url: str, headers: Dict[str, str]) -> Optional[int]:
    """The close code when the handshake is refused before it is accepted; None when it is accepted."""
    try:
        with c.websocket_connect(url, headers=dict(headers)):
            return None
    except WebSocketDisconnect as exc:
        return exc.code


def main() -> int:
    seen: List[Any] = []  # every reply, for CORS

    def ask(c: TestClient, method: str, path: str, **kw: Any) -> Any:
        if method in ("POST", "PUT", "PATCH") and "json" not in kw:
            kw["json"] = {}
        r = c.request(method, path, **kw)
        seen.append(r)
        return r

    c = TestClient(server.app, base_url=BASE)  # not entered as a context manager: the lifespan (the model) never runs
    pages = TMP / "dist"
    pages.mkdir()
    (pages / "index.html").write_text("<!doctype html><title>the instrument</title>", encoding="utf-8")
    if not any(isinstance(r, Mount) for r in server.app.routes):  # frontend/dist not built here: mount one as the server does
        server.app.mount("/", StaticFiles(directory=str(pages), html=True), name="frontend")

    # ---------------------------------------------------------------------------------------------- 1. the routes
    before = len(failures)
    routes = routes_of(server.app)
    http = [(filled(p), m) for kind, p, ms in routes if kind == "http" for m in ms]
    paths = {p for kind, p, _ in routes}
    check(len(http) >= 60, f"every HTTP route found ({len(http)})")
    for want in ("/api/health", "/api/larql", "/api/scientist/inspect_pole", "/api/scientist/trace_readers", "/api/scientist/causal_check",
                 "/api/scientist/test_hypothesis", "/favicon.ico", "/docs", "/openapi.json", "/api/saved_prompts", "/api/atlas/arch",
                 "/api/atlas/push", "/api/atlas/scientist/send", "/api/atlas/scientist/events"):
        check(want in paths, f"{want} among the routes")
    check(("socket", "/ws/theodolite", []) in routes and any(kind == "mount" for kind, _, _ in routes), "the socket and the pages' mount")
    http += [("/", "GET"), ("/index.html", "GET"), ("/no/such/page", "GET")]  # the pages, and a path nothing serves
    section(f"the routes: {len(http)} paths and methods, the socket, the pages", before)

    # ------------------------------------------------------------------------------------------ 2. a foreign Host
    before = len(failures)
    for path, method in http:
        for host in FOREIGN_HOSTS:
            r = ask(c, method, path, headers={"Host": host})
            check(is_refusal(r), f"{method} {path} with Host {host}: {r.status_code}")
    for host in FOREIGN_HOSTS:
        code = socket_closed(c, f"ws://{host}/ws/theodolite", {})
        check(code == 1008, f"the socket with Host {host}: {code}")
    section("a foreign Host refused on every route, and the socket closed before it is accepted", before)

    # ----------------------------------------------------------------------------------------- 3. a foreign Origin
    before = len(failures)
    for path, method in http:
        for origin in FOREIGN_ORIGINS:
            r = ask(c, method, path, headers={"Origin": origin})
            check(is_refusal(r), f"{method} {path} from {origin}: {r.status_code}")
    for origin in FOREIGN_ORIGINS:
        code = socket_closed(c, "ws://127.0.0.1:8765/ws/theodolite", {"Origin": origin})
        check(code == 1008, f"the socket from {origin}: {code}")
    section("a foreign Origin refused on every route and on the socket", before)

    # ----------------------------------------------------------------------- 4. preflights, and what browsers mark
    before = len(failures)
    for path, method in http:
        r = ask(c, "OPTIONS", path, headers={"Origin": "https://evil.example", "Access-Control-Request-Method": method,
                                             "Access-Control-Request-Headers": "content-type"})
        check(is_refusal(r) and not any(k.lower().startswith("access-control") for k in r.headers), f"a cross-site preflight for {method} {path}: {r.status_code}")
        for hdrs in ({"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "no-cors"}, {"Sec-Fetch-Site": "same-site", "Sec-Fetch-Mode": "cors"},
                     {"Origin": "http://localhost:5173", "Sec-Fetch-Site": "same-site", "Sec-Fetch-Mode": "cors"}):
            r = ask(c, method, path, headers=hdrs)
            check(is_refusal(r), f"{method} {path} with {hdrs}: {r.status_code}")
        if method != "GET":
            r = ask(c, method, path, headers={"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate"})
            check(is_refusal(r), f"a cross-site form sent to {method} {path}: {r.status_code}")
    code = socket_closed(c, "ws://127.0.0.1:8765/ws/theodolite", {"Origin": "http://localhost:5173", "Sec-Fetch-Site": "same-site",
                                                                  "Sec-Fetch-Mode": "websocket"})
    check(code == 1008, f"the socket from a page on another port of this machine, not through vite's proxy: {code}")
    section("cross-site preflights refused with no access-control header; requests marked cross- or same-site refused", before)

    # ------------------------------------------------------------------------------------------- 5. what passes
    before = len(failures)
    for path, method in http:
        heard = [ask(c, method, path),  # a script
                 ask(c, method, path, headers={"Host": "localhost:8765"}),
                 ask(c, method, path, headers={"Host": "[::1]:8765"}),
                 ask(c, method, path, headers={"Origin": "http://127.0.0.1:8765", "Sec-Fetch-Site": "same-origin"}),  # its own page
                 ask(c, method, path, headers={"Origin": "http://127.0.0.1:5173", "Sec-Fetch-Site": "same-origin"})]  # vite's /api
        if method == "GET":
            heard.append(ask(c, method, path, headers={"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate"}))  # a link
        check(all(r.status_code != 403 for r in heard), f"{method} {path} from this machine: {[r.status_code for r in heard]}")
    check(ask(c, "GET", "/").text.startswith("<!doctype html>"), "the pages are served")
    check(ask(c, "GET", "/api/health").status_code == 503, "a route that reads the model says it is not loaded (no lifespan here)")
    from backend.local_only import refused

    # a request with no Host at all (HTTP/1.0, a hand-made ASGI call) is a script on this machine
    check(not refused({"type": "http", "method": "GET", "path": "/api/health", "headers": [], "client": ("127.0.0.1", 5)}),
          "no Host at all: allowed, as the saved prompts and the atlas allow it")
    real_state = server.get_app_state
    server.get_app_state = lambda: (None, None)  # type: ignore[assignment]  # the socket's loop runs; nothing here reads the model
    try:
        for url, hdrs in (("ws://127.0.0.1:8765/ws/theodolite", {"Origin": "http://127.0.0.1:8765"}),  # its own page
                          ("ws://127.0.0.1:5173/ws/theodolite", {"Origin": "http://127.0.0.1:5173"}),  # through vite's /ws proxy
                          ("ws://localhost:8765/ws/theodolite", {})):  # a script
            try:
                with c.websocket_connect(url, headers=dict(hdrs)) as ws:
                    ws.send_json({"action": "larql", "voice": "not a voice"})
                    msg = ws.receive_json()
                check(msg.get("type") == "larql_result" and msg.get("ok") is False and "no edit was made" in msg.get("error", ""),
                      f"the socket at {url} {hdrs}: accepted, and answers ({msg})")
            except WebSocketDisconnect as exc:
                check(False, f"the socket at {url} {hdrs} refused: {exc.code}")
    finally:
        server.get_app_state = real_state  # type: ignore[assignment]
    section("scripts, the instrument's own pages, vite's proxy and plain links pass, on every route and on the socket", before)

    # ------------------------------------------------------------------------------------------------- 6. CORS
    before = len(failures)
    for origin in LOCAL_ORIGINS:
        r = ask(c, "OPTIONS", "/api/larql", headers={"Origin": origin, "Access-Control-Request-Method": "POST",
                                                      "Access-Control-Request-Headers": "content-type"})
        check(r.status_code == 200 and r.headers.get("access-control-allow-origin") == origin, f"a preflight from {origin}: {r.status_code} {r.headers.get('access-control-allow-origin')}")
    leaks = [(r.request.method, r.request.url.path, foreign_allowed(r)) for r in seen if foreign_allowed(r)]
    check(not leaks, f"no reply names a foreign origin as allowed ({len(seen)} replies): {leaks[:3]}")
    check(sum(1 for r in seen if r.headers.get("access-control-allow-origin")) >= len(LOCAL_ORIGINS), "local origins are named")
    # the CORS layer alone: LocalOnly refuses a foreign Origin before CORS sees it, so the replies above would pass with
    # CORS allowing every origin; built from the server's own options, it must allow only this machine's
    from starlette.applications import Starlette
    from starlette.middleware.cors import CORSMiddleware
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route

    opts = next(m for m in server.app.user_middleware if m.cls is CORSMiddleware)
    alone = CORSMiddleware(None, *opts.args, **opts.kwargs)  # type: ignore[arg-type]
    look_alikes = FOREIGN_ORIGINS + ["http://localhost.evil.example", "http://127.0.0.1:8765.evil.example", "https://127.0.0.1.nip.io"]
    allowed = [o for o in look_alikes if alone.is_allowed_origin(o)]
    check(not alone.allow_all_origins and not allowed, f"the server's CORS options allow no foreign origin: {allowed}, every origin {alone.allow_all_origins}")
    check(all(alone.is_allowed_origin(o) for o in LOCAL_ORIGINS + ["http://127.0.0.1:5173", "http://localhost"]), "they allow this machine's, on any port")
    bare = TestClient(Starlette(routes=[Route("/", lambda request: PlainTextResponse("ok"))], middleware=[opts]))
    for origin in look_alikes:
        rs = [bare.options("/", headers={"Origin": origin, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"}),
              bare.get("/", headers={"Origin": origin, "Cookie": "a=b"})]
        check(all(r.headers.get("access-control-allow-origin") is None for r in rs),
              f"CORS alone names {origin} to no preflight or credentialed GET: {[r.headers.get('access-control-allow-origin') for r in rs]}")
    r = bare.options("/", headers={"Origin": "http://127.0.0.1:5173", "Access-Control-Request-Method": "POST"})
    check(r.headers.get("access-control-allow-origin") == "http://127.0.0.1:5173", f"CORS alone answers a local page: {r.headers.get('access-control-allow-origin')}")
    section(f"CORS: local pages' origins answered, a foreign one never ({len(seen)} replies, and the CORS layer alone)", before)

    # ------------------------------------------------------------------------------------- 7. on a real socket
    before = len(failures)
    import uvicorn

    port = free_port()
    server.get_app_state = lambda: (None, None)  # type: ignore[assignment]  # an accepted socket waits for a message
    srv = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="warning", lifespan="off",
                                        timeout_graceful_shutdown=3))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(200):
        if srv.started:
            break
        time.sleep(0.05)

    def handshake(origin: str) -> str:
        with socket.create_connection(("127.0.0.1", port), timeout=10) as sk:
            sk.sendall((f"GET /ws/theodolite HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                        f"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\nOrigin: {origin}\r\n\r\n").encode())
            return sk.recv(4096).decode("latin-1").split("\r\n", 1)[0]

    try:
        line = handshake("https://evil.example")
        check(line.startswith("HTTP/1.1 403"), f"a foreign Origin's handshake on uvicorn: {line!r}")
        line = handshake(f"http://127.0.0.1:{port}")
        check(line.startswith("HTTP/1.1 101"), f"the page's own handshake on uvicorn: {line!r}")
    finally:
        srv.should_exit = True
        th.join(timeout=10)
        server.get_app_state = real_state  # type: ignore[assignment]
    section("on uvicorn: a refused handshake is an HTTP 403, an accepted one 101", before)

    # ----------------------------------------------------------------------------------------------------- 8. labd
    before = len(failures)
    cwd = os.getcwd()
    import labd  # changes to the repository's root, as it does when run; its startup (the model) runs only under uvicorn

    os.chdir(cwd)
    labd.LOG = TMP / "labd_exec_log.jsonl"  # not the repository's
    labd.STATE.update(rt=None, lab=type("Lab", (), {"clear_hooks": lambda self: None})())
    check(labd.token_path(8765) == TMP / "run" / "theodolite" / "labd-8765.token", f"the token's default place: {labd.token_path(8765)}")
    os.environ["LABD_TOKEN_FILE"] = str(TMP / "elsewhere" / "t")
    check(labd.token_path(8765) == TMP / "elsewhere" / "t", "LABD_TOKEN_FILE names another")
    os.environ.pop("LABD_TOKEN_FILE")
    lc = TestClient(labd.app, base_url=BASE)
    job = {"code": "STORE['ran'] = STORE.get('ran', 0) + 1\nresult = 6 * 7", "label": "check_local_only"}
    labd.TOKEN["value"] = None
    for sent in ("Bearer ", "Bearer None"):
        r = lc.post("/exec", json=job, headers={"Authorization": sent})
        check(r.status_code == 401 and "ran" not in labd.STORE, f"before any launch's token, /exec with {sent!r} runs nothing: {r.status_code}")
    tok_file = labd.token_path(labd.PORT)
    token = labd.write_token(tok_file)
    mode = stat.S_IMODE(tok_file.stat().st_mode)
    check(tok_file.read_text() == token + "\n" and len(token) >= 40 and labd.TOKEN["value"] == token, "the token written whole")
    check(mode == 0o600, f"the token file readable by its user only: {oct(mode)}")
    check(labd.write_token(tok_file) != token and stat.S_IMODE(tok_file.stat().st_mode) == 0o600 and not list(tok_file.parent.glob(".labd-token-*")),
          "a new launch, a new token, the file still 0600, nothing left beside it")
    token = labd.TOKEN["value"]
    auth = {"Authorization": f"Bearer {token}"}
    lab_routes = routes_of(labd.app)
    check(sorted(p for _, p, _ in lab_routes) == ["/exec", "/health", "/reload"], f"labd's routes, no docs pages: {lab_routes}")
    check(all(lc.get(p).status_code == 404 for p in ("/docs", "/openapi.json", "/redoc")), "no docs pages served")
    for _, path, methods in lab_routes:
        for method in methods:
            for hdrs in ({"Host": "evil.example:8765"}, {"Host": "127.0.0.1.nip.io:8765"}, {"Origin": "https://evil.example"},
                         {"Origin": "http://127.0.0.1:8765"}, {"Sec-Fetch-Site": "same-origin"}, {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate"}):
                r = lc.request(method, path, headers={**auth, **hdrs}, json=job if method == "POST" else None)
                check(r.status_code == 403 and "never a web page" in r.json().get("detail", ""), f"labd {method} {path} with {hdrs}: {r.status_code}")
    check("ran" not in labd.STORE, "no refused job ran")
    for hdrs in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": token}, {"Authorization": f"Bearer {token}x"}):
        r = lc.post("/exec", json=job, headers=hdrs)
        check(r.status_code == 401 and "token" in r.json().get("detail", ""), f"/exec with {hdrs or 'no token'}: {r.status_code}")
        r = lc.post("/reload", headers=hdrs)
        check(r.status_code == 401, f"/reload without the token: {r.status_code}")
    check("ran" not in labd.STORE, "without the token, nothing ran")
    r = lc.get("/health")
    check(r.status_code == 200 and r.json()["busy"] is False, f"/health wants no token: {r.status_code}")
    r = lc.post("/exec", json=job, headers=auth)
    check(r.status_code == 200 and r.json()["ok"] and r.json()["result"] == 42 and labd.STORE.get("ran") == 1, f"with the token the job runs: {r.text[:200]}")
    # labclient.py against labd on uvicorn: it reads the token file and sends it
    port = free_port()
    srv = uvicorn.Server(uvicorn.Config(labd.app, host="127.0.0.1", port=port, log_level="warning", lifespan="off"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(200):
        if srv.started:
            break
        time.sleep(0.05)
    client = str(ROOT_DIR / "experiments" / "lab" / "labclient.py")
    try:
        os.rename(tok_file, tok_file.with_name(f"labd-{port}.token"))
        res = subprocess.run([sys.executable, "-I", client, "-", "--port", str(port), "--label", "check"], input="result = 6 * 7\nprint('hello')",
                             capture_output=True, text=True, timeout=60)
        check(res.returncode == 0 and "hello" in res.stdout and res.stdout.strip().endswith("42") and labd.STORE.get("ran") == 1,
              f"labclient.py reads the token and the job runs: {res.returncode} {res.stdout!r} {res.stderr[-300:]!r}")
        # a second launch on the port labd is serving: it stops before its model loads (none could here) and before it
        # writes a token, so the running launch's file, and labclient's jobs, are untouched
        served = tok_file.with_name(f"labd-{port}.token")
        res = subprocess.run([sys.executable, "-I", str(ROOT_DIR / "experiments" / "lab" / "labd.py")], env={**os.environ, "LABD_PORT": str(port)},
                             capture_output=True, text=True, timeout=120)
        check(res.returncode != 0 and f"cannot listen on 127.0.0.1:{port}" in res.stderr and "no token written" in res.stderr
              and "Traceback" not in res.stderr, f"a second launch on a served port stops in plain words: {res.returncode} {res.stderr[-300:]!r}")
        check(served.is_file() and served.read_text() == token + "\n" and not list(served.parent.glob(".labd-token-*")),
              "the running launch's token file untouched")
        res = subprocess.run([sys.executable, "-I", client, "-", "--port", str(port)], input="result = 6 * 7", capture_output=True, text=True, timeout=60)
        check(res.returncode == 0 and res.stdout.strip().endswith("42"), f"and the running launch still takes jobs: {res.returncode} {res.stderr[-300:]!r}")
        bad = tok_file.with_name(f"labd-{port}.token")
        bad.write_text("not-this-launch\n")
        res = subprocess.run([sys.executable, "-I", client, "-", "--port", str(port)], input="result = 1", capture_output=True, text=True, timeout=60)
        check(res.returncode != 0 and "refused the job (401)" in res.stderr and "earlier launch" in res.stderr, f"a stale token, in plain words: {res.stderr[-300:]!r}")
        bad.unlink()
        res = subprocess.run([sys.executable, "-I", client, "-", "--port", str(port)], input="result = 1", capture_output=True, text=True, timeout=60)
        check(res.returncode != 0 and "no labd token at" in res.stderr and str(bad) in res.stderr and "Traceback" not in res.stderr,
              f"no token file, in plain words: {res.stderr[-300:]!r}")
    finally:
        srv.should_exit = True
        th.join(timeout=10)
    check(len(labd.LOG.read_text().splitlines()) == 3, "each job that ran logged once (here, not in the repository)")
    section("labd: the Host and any browser refused on every route, /exec and /reload only with the launch's token, labclient sends it, "
            "a second launch leaves it alone", before)

    print(f"\n{passed} checks passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(code)

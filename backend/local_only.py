"""Who may ask the instrument's server (`backend/local_only.py`): every route of backend/server.py's app, its pages and
its WebSocket answer only the instrument's own pages, served from this machine, and scripts run on it.

The rule is the atlas's (`backend.atlas.local_pages_only`): the peer must be this machine; the Host must name it
(127.0.0.1, localhost or [::1], on any port), and so must the Origin when there is one; and a request a browser marks
as sent from another site (Sec-Fetch-Site cross-site or same-site) is refused unless it is a plain navigation (a link
opened from elsewhere reads nothing back). A script run on this machine (no Origin, a local Host: curl, the checks, the
scientist's MCP server, the openai backend's calls through the app in this process) passes, and so do the server's own
pages and vite's proxy (frontend/vite.config.ts: `/api` with the server's own Host, `/ws` with vite's, 127.0.0.1:5173;
both send the page's Origin, which is local).

`LocalOnly` applies it once, as an ASGI middleware around the whole app and outside its CORS, so it also covers what a
route's dependency does not reach: the static pages, FastAPI's own /docs and /openapi.json, an unknown path, a CORS
preflight (which CORS answers itself, before any route) and the WebSocket. A refused request gets 403 with the reason in
plain words; a refused socket is closed with 1008 before it is accepted, which uvicorn sends as an HTTP 403 to the
handshake. A browser applies no CORS to a WebSocket, so narrowing CORS (backend/server.py) does not cover the socket;
this does. The atlas's, the scientist chat's and the saved prompts' routes keep their own dependencies as well.

Which account asks: on Linux the guard also refuses a connection from this machine that another account's process
made (403, or 1008 on the socket), since the scientist's tools, and at the home reach its shell, run as the account
that runs the server. Root's connections pass too: root can act as any account already, and a container's
port-forwarder connects as root. `peer_uid` finds the line of /proc/net/tcp (or tcp6) whose local end is the client's
address and port and whose remote end is the server's, the client's end of this very connection, and reads its owner,
from a line some process still holds only: once a client has closed its end, the line left (closing, or time-wait) is
no process's, inode 0, and printed as root's, and once it has reset it there is none. A request whose sender has so
gone (or whose client uvicorn could not name, the connection reset before it was taken up) is refused when the tables
list the server's own listening port, since no live caller on this machine is missing from them; before this, a
process of uid 65534 that sent a request and reset its socket at once had every such request run, and one that closed
it got through while the server was busy (tried here, as root dropping to 65534). It reads the tables afresh for every
request and socket and keeps nothing, so nothing it says can go stale: under 1 ms over the 30 to 110 sockets of the
machine it was checked on (scripts/check_local_only.py prints it), about 2 ms over 1,000 and 7 ms over 4,500. When it
cannot tell (not Linux, tables that do not list the server's port, a Unix socket) the request passes, and the log says
so once. A request carrying X-Forwarded-For is refused: uvicorn takes the client's address, port and all, from that
header when the peer is loopback (its default), so the address the guard is given would be the sender's choice,
another socket's even. An in-process client that names itself (not an address: Starlette's TestClient, the openai
backend's calls) has no socket and passes. vite's dev proxy runs as the person, so its connections are theirs and pass
(it sends no X-Forwarded-For), and so does whatever another account sends through it while `npm run dev` runs: the
check guards this server's port, not vite's (tried: a uid-65534 client through vite's proxy was answered, 200 and
101). `THEODOLITE_PEER_CHECK=off` turns this check off (to serve through a proxy, say); each of its refusals names it.

Before this guard, CORS allowed every origin with credentials, and the older routes (POST /api/larql, the four POST
/api/scientist/* tools) and /ws/theodolite checked neither Host nor Origin: a page from any site the reader visited, or
one rebinding its name to 127.0.0.1, could run sweeps and make vault edits on the live model, which change every later
reading until rolled back, and read the replies (seen with FastAPI's TestClient: a cross-site preflight answered with
that site's origin, and a socket from it accepted). scripts/check_local_only.py checks every route and the socket.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import struct
from typing import Any, Dict, Optional, Sequence, Tuple

from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocketClose

from backend.atlas import local_pages_only

logger = logging.getLogger("theodolite.local_only")

REFUSED = "The instrument's server answers only its own pages, served from this machine, and scripts run on it."
PEER_CHECK_OFF = " (THEODOLITE_PEER_CHECK=off turns this check off.)"
OTHER_ACCOUNT = ("The instrument's server answers only the account it runs as; this came from another account's process."
                 + PEER_CHECK_OFF)
FORWARDED = ("The instrument's server cannot tell whose a forwarded request (X-Forwarded-For) is, so it refuses it."
             + PEER_CHECK_OFF)
GONE_SENDER = ("The instrument's server cannot tell whose this request is: its sender closed or reset the connection "
               "before the server could look." + PEER_CHECK_OFF)
POLICY_VIOLATION = 1008  # the WebSocket close code for a refused handshake
PEER_CHECK_ENV = "THEODOLITE_PEER_CHECK"
NET_TABLES = ("/proc/net/tcp", "/proc/net/tcp6")
GONE = -1  # peer_uid's answer for a client end that no process holds any more: no account's
_told = {"unknown": False}


def refused(scope: Scope) -> bool:
    """Whether the rule refuses this request, or this socket's handshake (an HTTP GET, read as one)."""
    probe: Dict[str, Any] = dict(scope) if scope["type"] == "http" else {**scope, "type": "http", "method": "GET"}
    try:
        local_pages_only(Request(probe))
    except HTTPException:
        return True
    return False


def _table_forms(host: str, port: int) -> Tuple[Optional[str], str]:
    """An address and port as /proc/net/tcp prints them (None for an IPv6 address, which only tcp6 holds) and as tcp6
    does (an IPv4 address mapped): each 32-bit word of the address in hex in this machine's byte order, then the port."""
    ip = ipaddress.ip_address(host)
    v4 = ip if isinstance(ip, ipaddress.IPv4Address) else ip.ipv4_mapped
    v6 = ip if isinstance(ip, ipaddress.IPv6Address) else ipaddress.IPv6Address(f"::ffff:{ip}")
    four = "%08X:%04X" % (struct.unpack("=I", v4.packed)[0], port) if v4 is not None else None
    return four, "".join("%08X" % w for w in struct.unpack("=4I", v6.packed)) + ":%04X" % port


def peer_uid(client: Optional[Sequence[Any]], server: Sequence[Any],
             tables: Sequence[str] = NET_TABLES) -> Optional[int]:
    """The account (uid) owning the TCP socket whose local end is `client` and whose remote end is `server`, both
    (address, port): the client's end of a connection to this server, from Linux's /proc/net/tcp and tcp6, read afresh.
    Only a line some process still holds (its inode not 0) is read: what is left of a closed end says uid 0 whoever
    closed it. GONE when the tables list `server`'s listening port but no such line, or there is no client to look for
    (uvicorn names none for a connection reset before it was taken up): its sender has closed or reset it. None when it
    cannot tell: not Linux (no such tables), tables that do not list `server`'s port, or an address that is not one."""
    try:
        s4, s6 = _table_forms(str(server[0]), int(server[1]))
        c4, c6 = _table_forms(str(client[0]), int(client[1])) if client is not None else (None, None)
    except (ValueError, TypeError, IndexError):
        return None
    port, listening = ":%04X" % int(server[1]), False
    for path, local, remote in ((tables[0], c4, s4), (tables[1], c6, s6)):
        try:
            with open(path, encoding="ascii") as f:
                rows = [line.split() for line in f.read().splitlines()[1:]]
        except OSError:
            continue
        for row in rows:  # sl, local, remote, st, tx:rx, tr:when, retrnsmt, uid, timeout, inode, ...
            if len(row) < 10:
                continue
            if row[3] == "0A" and row[1].endswith(port):  # a listening socket on the server's port
                listening = True
            elif local is not None and row[1] == local and row[2] == remote and row[9] != "0":
                return int(row[7])
    return GONE if listening else None


def other_account(scope: Scope) -> Optional[str]:
    """Why a request or socket from this machine is refused for whose it is (module docstring), or None: one carrying
    X-Forwarded-For, a connection whose client end another account than the server's, or root, owns, or one whose
    sender has gone (`GONE`). A client that is not an address (an in-process one) passes; so does one that `peer_uid`
    cannot place, and the log says so once."""
    if os.environ.get(PEER_CHECK_ENV, "").strip().lower() in ("off", "0", "false", "no"):
        return None
    if any(k.lower() == b"x-forwarded-for" for k, _ in scope.get("headers") or []):
        return FORWARDED
    client, server = scope.get("client"), scope.get("server")
    if client is not None:
        try:
            ipaddress.ip_address(str(client[0]))
        except ValueError:
            return None  # no socket: an in-process client names itself
    uid = peer_uid(client, server) if server is not None else None
    if uid == GONE:
        return GONE_SENDER
    if uid is None:
        if not _told["unknown"]:
            _told["unknown"] = True
            logger.warning("Could not tell which account a connection to this server belongs to (%s to %s; on Linux it "
                           "is read from /proc/net/tcp): such requests pass, as from this account.", client, server)
        return None
    return None if uid in (os.geteuid(), 0) else OTHER_ACCOUNT


class LocalOnly:
    """The rule above around a whole ASGI app, and on Linux which account asks (`other_account`): a refused request
    gets 403, a refused socket is closed before it is accepted; the lifespan and everything that passes go on
    unchanged."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            why = REFUSED if refused(scope) else other_account(scope)
            if why:
                if scope["type"] == "websocket":
                    await WebSocketClose(code=POLICY_VIOLATION, reason=why)(scope, receive, send)
                else:
                    await JSONResponse({"detail": why}, status_code=403)(scope, receive, send)
                return
        await self.app(scope, receive, send)

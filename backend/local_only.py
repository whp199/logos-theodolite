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

Before this guard, CORS allowed every origin with credentials, and the older routes (POST /api/larql, the four POST
/api/scientist/* tools) and /ws/theodolite checked neither Host nor Origin: a page from any site the reader visited, or
one rebinding its name to 127.0.0.1, could run sweeps and make vault edits on the live model, which change every later
reading until rolled back, and read the replies (seen with FastAPI's TestClient: a cross-site preflight answered with
that site's origin, and a socket from it accepted). scripts/check_local_only.py checks every route and the socket.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocketClose

from backend.atlas import local_pages_only

REFUSED = "The instrument's server answers only its own pages, served from this machine, and scripts run on it."
POLICY_VIOLATION = 1008  # the WebSocket close code for a refused handshake


def refused(scope: Scope) -> bool:
    """Whether the rule refuses this request, or this socket's handshake (an HTTP GET, read as one)."""
    probe: Dict[str, Any] = dict(scope) if scope["type"] == "http" else {**scope, "type": "http", "method": "GET"}
    try:
        local_pages_only(Request(probe))
    except HTTPException:
        return True
    return False


class LocalOnly:
    """The rule above around a whole ASGI app: a refused request gets 403, a refused socket is closed before it is
    accepted; the lifespan and everything that passes go on unchanged."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket") and refused(scope):
            if scope["type"] == "websocket":
                await WebSocketClose(code=POLICY_VIOLATION, reason=REFUSED)(scope, receive, send)
            else:
                await JSONResponse({"detail": REFUSED}, status_code=403)(scope, receive, send)
            return
        await self.app(scope, receive, send)

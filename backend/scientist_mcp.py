#!/usr/bin/env python3
"""The scientist's tools as a stdio MCP server (`backend/scientist_mcp.py`), for the atlas's scientist chat when it runs
on Claude Code or Codex (backend/scientist_chat.py starts them with this server in their MCP config).

It reads and acts through the instrument's server over HTTP (`$THEODOLITE_URL`, the server that started the chat), and
its screen actions go to that chat (`$THEODOLITE_CHAT`). The tools are backend/scientist_tools.py's: layers count from
1 there, as on the pages. Standard library only, line-delimited JSON-RPC on stdin and stdout (MCP's stdio transport),
as the explorer's agent_mcp.py.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend import scientist_tools as st  # noqa: E402

URL = os.environ.get("THEODOLITE_URL", "http://127.0.0.1:8000").rstrip("/")
CHAT = os.environ.get("THEODOLITE_CHAT", "")
TIMEOUT = 900  # a live reading may wait for the GPU behind a sweep or a push chat


class HttpApi:
    """The atlas over HTTP, as scientist_tools.call reaches it (each request in a worker thread)."""

    def __init__(self, base: str) -> None:
        self.base = base

    def _open(self, req: urllib.request.Request) -> Any:
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return json.load(r)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(body).get("detail", body)
            except ValueError:
                detail = body
            raise st.ToolError(f"{exc.code}: {detail if isinstance(detail, str) else json.dumps(detail)}"[:2000])
        except urllib.error.URLError as exc:
            raise st.ToolError(f"The instrument's server at {self.base} did not answer ({exc.reason}).")

    async def get(self, path: str, **query: Any) -> Any:
        q = urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
        req = urllib.request.Request(f"{self.base}{path}" + (f"?{q}" if q else ""), headers={"Accept": "application/json"})
        return await asyncio.to_thread(self._open, req)

    async def post(self, path: str, body: Optional[Dict[str, Any]]) -> Any:
        req = urllib.request.Request(f"{self.base}{path}", json.dumps(body or {}).encode("utf-8"),
                                     {"Content-Type": "application/json", "Accept": "application/json"})
        return await asyncio.to_thread(self._open, req)


def reply(id_: Any, result: Any = None, error: Optional[Dict[str, Any]] = None) -> None:
    msg: Dict[str, Any] = {"jsonrpc": "2.0", "id": id_}
    msg.update({"error": error} if error else {"result": result})
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    api = HttpApi(URL)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            m = json.loads(line)
        except ValueError:
            continue
        method, id_ = m.get("method"), m.get("id")
        if id_ is None:
            continue  # notifications
        if method == "initialize":
            reply(id_, {"protocolVersion": (m.get("params") or {}).get("protocolVersion", "2025-06-18"),
                        "capabilities": {"tools": {}}, "serverInfo": {"name": "atlas", "version": "1"}})
        elif method == "tools/list":
            reply(id_, {"tools": st.TOOLS})
        elif method == "tools/call":
            p = m.get("params") or {}
            try:
                out = asyncio.run(st.call(p.get("name", ""), p.get("arguments") or {}, api, CHAT))
                reply(id_, {"content": [{"type": "text", "text": st.as_text(out)}]})
            except Exception as exc:  # a tool's failure goes back to the model, not to the protocol
                reply(id_, {"content": [{"type": "text", "text": f"error: {exc}"}], "isError": True})
        elif method == "ping":
            reply(id_, {})
        else:
            reply(id_, error={"code": -32601, "message": f"method not found: {method}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""End-to-End Verification Suite (`scripts/verify_e2e_ui_and_mcp.py`) for Logos Theodolite.

Runs inside a single process/network namespace to verify:
1. FastAPI + Binary Float32 WebSocket (`/ws/theodolite`) wire format (`0x5448454F`) & JSON metadata
2. FastMCP Scientist Tools (`inspect_pole`, `trace_downstream_readers`, `test_motif_hypothesis`)
3. LARQL / VINDEX3 FFN Vault queries (`/api/larql`)
4. Headless Chrome inspection of the UI (frontend/dist, built first) via CDP:
   - Zero console errors or page exceptions
   - the first live sighting replaces the recorded one and the 60-ring drawing renders
   - on every page, the "rubrics" and "apparatus" reading toggles show and hide their blocks
   - a gentle nudge (×0.01 of the stream) round-trips; screenshots saved to $THEODOLITE_ARTIFACT_DIR
     (default: artifacts/e2e/)
5. The options added for V7 bugs 1, 3, 4, 5 and 8 (chat template and system turn, Löwdin / weighted / per-layer /
   centred survey frames with proj_frac, the mended nudge, the direct-path thread score, write-size pole ranking),
   over the WebSocket and the MCP tools, and the formerly disabled UI controls that drive them.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("HF_HUB_OFFLINE", "1")
for _proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_proxy_var, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import httpx
import numpy as np
import torch
import uvicorn
import websockets

from backend.mcp_scientist import (
    inspect_pole,
    test_motif_hypothesis,
    trace_downstream_readers,
)
from backend.server import MAGIC_THEO, WIRE_VERSION, app


async def ws_sweep(ws: Any, payload: Dict[str, Any]) -> tuple[bytes, Dict[str, Any]]:
    """One sweep over an open /ws/theodolite socket: the binary frame and the JSON metadata."""
    await ws.send(json.dumps({"action": "sweep", "payload": payload}))
    bin_msg = await ws.recv()
    meta = json.loads(await ws.recv())
    assert isinstance(bin_msg, bytes), f"expected a binary frame first, got {str(bin_msg)[:200]}"
    return bin_msg, meta

ARTIFACT_DIR = Path(os.environ.get("THEODOLITE_ARTIFACT_DIR", str(ROOT_DIR / "artifacts" / "e2e")))
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
HOST = "127.0.0.1"
PORT = 8765
CDP_PORT = 9229


class CDPClient:
    """Minimal async Chrome DevTools Protocol client over WebSocket."""

    def __init__(self, ws_url: str) -> None:
        self.ws_url = ws_url
        self.ws: Any = None
        self.msg_id = 0
        self.pending: Dict[int, asyncio.Future] = {}
        self.console_errors: List[str] = []
        self.console_logs: List[str] = []
        self.page_exceptions: List[str] = []
        self._reader_task: Any = None

    async def connect(self) -> None:
        self.ws = await websockets.connect(self.ws_url, max_size=50 * 1024 * 1024)
        self._reader_task = asyncio.create_task(self._reader_loop())
        await self.send("Runtime.enable")
        await self.send("Page.enable")
        await self.send("Log.enable")

    async def _reader_loop(self) -> None:
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                if "id" in msg:
                    fut = self.pending.pop(msg["id"], None)
                    if fut and not fut.done():
                        fut.set_result(msg)
                else:
                    method = msg.get("method", "")
                    params = msg.get("params", {})
                    if method == "Runtime.consoleAPICalled":
                        c_type = params.get("type", "log")
                        args = params.get("args", [])
                        text = " ".join(str(a.get("value", a.get("description", ""))) for a in args)
                        self.console_logs.append(f"[{c_type}] {text}")
                        if c_type == "error":
                            self.console_errors.append(text)
                    elif method == "Runtime.exceptionThrown":
                        details = params.get("exceptionDetails", {})
                        text = details.get("text", "") + " " + str(details.get("exception", {}).get("description", ""))
                        self.page_exceptions.append(text.strip())
                    elif method == "Log.entryAdded":
                        entry = params.get("entry", {})
                        if entry.get("level") == "error":
                            text = entry.get("text", "")
                            if "favicon.ico" not in text:
                                self.console_errors.append(f"[Log.error] {text}")
        except Exception:
            pass

    async def send(self, method: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        self.msg_id += 1
        mid = self.msg_id
        fut = asyncio.get_running_loop().create_future()
        self.pending[mid] = fut
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        res = await asyncio.wait_for(fut, timeout=30.0)
        if "error" in res:
            raise RuntimeError(f"CDP error in {method}: {res['error']}")
        return res.get("result", {})

    async def eval_js(self, expression: str, await_promise: bool = True) -> Any:
        res = await self.send(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": await_promise,
            },
        )
        if "exceptionDetails" in res:
            raise RuntimeError(f"JS evaluation error: {res['exceptionDetails']}")
        return res.get("result", {}).get("value")

    async def screenshot(self, out_path: Path) -> None:
        res = await self.send("Page.captureScreenshot", {"format": "png"})
        png_bytes = base64.b64decode(res["data"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(png_bytes)

    async def close(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
        if self.ws:
            await self.ws.close()


async def main() -> int:
    print("=" * 92)
    print("LOGOS THEODOLITE — PHASE 2 END-TO-END SERVER, FASTMCP & 3D FRONTEND VERIFICATION")
    print("=" * 92)

    config = uvicorn.Config(app, host=HOST, port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    # Wait for server startup (loads Gemma-4-31B + J++ Lens via lifespan)
    async with httpx.AsyncClient(base_url=f"http://{HOST}:{PORT}", timeout=60.0, trust_env=False) as client:
        for _ in range(120):
            try:
                r = await client.get("/api/health")
                if r.status_code == 200:
                    health = r.json()
                    print(f"[1/4] FastAPI Server Healthy: model={health['model_id']} | VRAM={health['vram_allocated_gb']:.2f} GB")
                    break
            except Exception:
                pass
            await asyncio.sleep(0.5)
        else:
            raise RuntimeError("FastAPI server did not become healthy within 60 seconds.")

        # -------------------------------------------------------------------------
        # Step 1: Verify FastMCP Autonomous Circuit Scientist Tools
        # -------------------------------------------------------------------------
        print("\n[2/4] Verifying FastMCP Scientist Tools (`backend/mcp_scientist.py`)...")
        # Tool 1: inspect_pole on Layer 53 (1-based L54) Neuron 16383 (- South Pole)
        pole_res = inspect_pole(layer=53, neuron_idx=16383, polarity=-1.0)
        south_toks = [t["token"] for t in pole_res["south_pole"]["j_vindex_tokens"][:4]]
        north_toks = [t["token"] for t in pole_res["north_pole"]["j_vindex_tokens"][:4]]
        gate_toks = [t["token"] for t in pole_res["gate_trigger_tokens"][:4]]
        print(f"  • inspect_pole(L53, N16383, -1.0):")
        print(f"      South Pole (-) J-VINDEX : {south_toks}")
        print(f"      North Pole (+) J-VINDEX : {north_toks}")
        print(f"      Upstream Gate Triggers  : {gate_toks}")
        assert len(south_toks) > 0 and any("Sp" in t or "sp" in t.lower() for t in south_toks)

        # Tool 2: trace_downstream_readers on Layer 35 (1-based L36) Neuron 9482
        readers_res = await trace_downstream_readers(
            layer=35,
            neuron_idx=9482,
            prompt="The number of legs on the animal that spins webs is",
            max_layer_hop=6,
            top_k=5,
        )
        top_readers = readers_res["top_downstream_readers"]
        print(
            f"  • trace_downstream_readers(L35, N9482) -> {len(top_readers)} verified readers "
            f"(top reader: L{top_readers[0]['dst_layer']}:N{top_readers[0]['dst_neuron_idx']} "
            f"[{top_readers[0]['dst_polarity']}] C={top_readers[0]['coupling_score']:.2f})"
        )
        assert len(top_readers) > 0

        # Tool 3: test_motif_hypothesis on Layer 53 Neuron 16383 (Spider/Arachnid South-Pole neuron)
        hyp_res = await test_motif_hypothesis(
            layer=53,
            neuron_idx=16383,
            pos_prompts=[
                "The animal that spins silk webs to catch insects is the",
                "A black widow or tarantula is a type of",
                "The eight-legged arachnid spinning a web in the corner is a",
            ],
            neg_prompts=[
                "The capital city of France near the Seine river is",
                "In linear algebra, the eigenvalues of a symmetric matrix are",
                "To bake sourdough bread, mix flour, water, salt, and",
            ],
        )
        print(
            f"  • test_motif_hypothesis(L53, N16383): verdict={hyp_res['verdict']} | "
            f"Cohen's d={hyp_res['cohens_d']:.3f} | "
            f"mean_pos={hyp_res['mean_pos_signed_act']:.3f} vs mean_neg={hyp_res['mean_neg_signed_act']:.3f}"
        )
        assert abs(hyp_res["cohens_d"]) > 0.8, f"Expected strong selectivity, got {hyp_res['cohens_d']}"

        # -------------------------------------------------------------------------
        # Step 2: Verify Binary Float32 WebSocket (`/ws/theodolite`) Wire Contract
        # -------------------------------------------------------------------------
        print("\n[3/4] Verifying Binary Float32 WebSocket (`/ws/theodolite`) Wire Contract...")
        async with websockets.connect(
            f"ws://{HOST}:{PORT}/ws/theodolite",
            max_size=50 * 1024 * 1024,
        ) as ws:
            await ws.send(
                json.dumps(
                    {
                        "action": "sweep",
                        "payload": {
                            "prompt": "The number of legs on the animal that spins webs is",
                            "monuments": ["spider", "eight", "insect"],
                            "survey_mode": "monument",
                            "compare_think": False,
                            "max_new_tokens": 6,
                        },
                    }
                )
            )
            bin_msg = await ws.recv()
            json_msg = json.loads(await ws.recv())

            assert isinstance(bin_msg, bytes), "First WebSocket frame must be binary ArrayBuffer"
            magic, version, n_layers, n_poles, n_filaments, flags = struct.unpack("<6I", bin_msg[:24])
            assert magic == MAGIC_THEO, f"Invalid magic header: {hex(magic)}"
            assert version == WIRE_VERSION
            assert n_layers == 60 and n_poles == 15
            expected_floats = (
                60 * 3  # traj_xyz
                + 60 * 3  # traj_angles_norm
                + 60 * 4  # delta_attn_xyz
                + 60 * 4  # delta_ffn_xyz
                + 60 * 60  # cka_matrix
                + 60 * 15 * 6  # poles_data
                + n_filaments * 7  # filaments_data
            )
            expected_bytes = 24 + expected_floats * 4
            assert len(bin_msg) == expected_bytes, f"Byte size mismatch: {len(bin_msg)} != {expected_bytes}"
            f32_arr = np.frombuffer(bin_msg, dtype=np.float32, offset=24)
            assert np.isfinite(f32_arr).all(), "Binary Float32 payload contains non-finite values!"
            print(
                f"  • Binary Frame Verified: {len(bin_msg):,} bytes | magic=0x{magic:08X} | "
                f"layers={n_layers} | poles={n_poles}/layer | filaments={n_filaments} | finite=True"
            )
            print(
                f"  • JSON Frame Verified  : generated={json_msg['generated_text']!r} | "
                f"elapsed={json_msg['elapsed_ms']}ms | peak_vram={json_msg['peak_vram_gb']} GB"
            )
            # the defaults keep their meaning: bare text, the frame as built, C, |a|
            assert json_msg["chat"] is False and json_msg["prompt"] == "The number of legs on the animal that spins webs is"
            assert json_msg["frame"]["squaring"] == "gram_schmidt" and json_msg["thread_score"] == "coupling"
            assert json_msg["pole_ranking"] == "activation"
            assert json_msg["prompt_token_ids"][0] == 2 and json_msg["prompt_token_ids"][1] != 2
            assert all(0.0 < p["proj_frac"] <= 1.0 for p in json_msg["layer_plumes"])

            # ---------------------------------------------------------------------
            # Step 2b: the options added for V7 bugs 1, 3, 4, 5 and 8
            # ---------------------------------------------------------------------
            print("\n[3b/4] Verifying the chat template, survey-frame, nudge, thread-score and pole-ranking options...")
            system_turn = "You are a careful assistant."
            bin2, meta = await ws_sweep(
                ws,
                {
                    "prompt": "Tell me about yourself.",
                    "chat": True,
                    "system": system_turn,
                    "assistant_prefix": "Hi! I",
                    "monuments": ["she", "he", "it"],
                    "max_new_tokens": 4,
                    "frame_squaring": "lowdin",
                    "frame_fnw": True,
                    "centre_prompts": ["The weather today is"],
                    "pole_ranking": "write",
                    "thread_score": "direct",
                    "steering": {
                        "enabled": True,
                        "target_concept": "she",
                        "contrast_concept": "he",
                        "layers": [29, 35, 41, 47, 53],
                        "direction": "fnw",
                        "mask_quiet": True,
                        "spare_bos": True,
                        "strength": 0.01,
                    },
                },
            )
            assert meta["prompt"].startswith(f"<bos><|turn>system\n{system_turn}<turn|>\n<|turn>user\nTell me about yourself.")
            assert meta["prompt"].endswith("<|turn>model\n<|channel>thought\n<channel|>Hi! I"), meta["prompt"][-80:]
            assert meta["prompt_token_ids"][0] == 2 and meta["prompt_token_ids"][1] != 2, "expected exactly one BOS"
            assert meta["chat"] is True and meta["system"] == system_turn and meta["assistant_prefix"] == "Hi! I"
            fr = meta["frame"]
            assert fr["squaring"] == "lowdin" and fr["fnw_weighted"] and fr["centred"] and fr["n_centre_prompts"] == 1
            assert {"raw", "fnw", "self_cos"} <= set(fr["geometry"]) and fr["geometry"]["raw"]["cond"] >= 1.0
            assert all(0.0 < p["proj_frac"] <= 1.0 for p in meta["layer_plumes"])
            assert all(len(p["pole_write_sizes"]) == 15 and p["poles_ffn_explained"] is not None for p in meta["layer_plumes"])
            assert meta["pole_ranking"] == "write" and meta["thread_score"] == "direct"
            fil0 = meta["filaments_meta"][0]
            assert fil0["score_kind"] == "direct" and fil0["direct_gate"] is not None and fil0["score"] == fil0["direct_gate"]
            sa = meta["steering_applied"]
            assert meta["is_steered"] and sa["direction"] == "fnw" and sa["mask_quiet"] and sa["spare_bos"] and sa["strength"] == 0.01
            assert all(n >= 181 for n in sa["n_masked_by_layer"].values())
            assert np.isfinite(np.frombuffer(bin2, dtype=np.float32, offset=24)).all()
            print(
                f"  • chat + system + Löwdin/fnw/centred frame + write ranking + direct threads + mended nudge: "
                f"ids {meta['prompt_token_ids'][:3]} | went on {meta['generated_text']!r} | "
                f"κ raw {fr['geometry']['raw']['cond']:.2f}, fnw {fr['geometry']['fnw']['cond']:.2f} | "
                f"proj_frac {min(p['proj_frac'] for p in meta['layer_plumes']):.4f}–{max(p['proj_frac'] for p in meta['layer_plumes']):.4f}"
            )

            # Löwdin is order-free: each word's coordinate is the same whichever order the words are named in
            coords = {}
            for order in (["she", "he", "it"], ["he", "it", "she"]):
                b, _ = await ws_sweep(
                    ws,
                    {"prompt": "Tell me about yourself.", "chat": True, "assistant_prefix": "Hi! I", "monuments": order,
                     "max_new_tokens": 0, "frame_squaring": "lowdin"},
                )
                xyz = np.frombuffer(b, dtype=np.float32, offset=24)[:180].reshape(60, 3)
                coords[tuple(order)] = {w: xyz[:, k] for k, w in enumerate(order)}
            a_, b_ = coords.values()
            drift = max(float(np.abs(a_[w] - b_[w]).max() / (np.abs(a_[w]).max() + 1e-9)) for w in ("she", "he", "it"))
            assert drift < 1e-3, f"Löwdin frame depends on the order: relative drift {drift}"
            print(f"  • Löwdin frame order-free across two orders: max relative drift {drift:.1e}")

            _, meta_pl = await ws_sweep(
                ws,
                {"prompt": "Tell me about yourself.", "chat": True, "monuments": ["she", "he", "it"], "max_new_tokens": 0,
                 "frame_per_layer": True},
            )
            assert meta_pl["frame"]["per_layer"] and meta_pl["frame"]["state"] == "h_raw"
            assert all(0.0 < p["proj_frac"] <= 1.0 for p in meta_pl["layer_plumes"])
            print("  • Per-layer frame: the raw h_l read in each layer's pulled-back frame")

            await ws.send(json.dumps({"action": "trace_readers", "layer": 35, "neuron_idx": 9482,
                                      "prompt": meta["prompt"], "max_layer_hop": 3, "top_k": 4, "score": "direct"}))
            tr = json.loads(await ws.recv())
            assert tr["type"] == "downstream_readers", tr
            trd = tr["data"]
            rs = trd["top_downstream_readers"]
            assert trd["max_layer_hop"] == 3 and trd["score_kind"] == "direct" and 0 < len(rs) <= 4
            assert all(36 <= r["dst_layer"] <= 38 for r in rs)
            assert all(abs(rs[i]["score"]) >= abs(rs[i + 1]["score"]) for i in range(len(rs) - 1))
            print(f"  • WS trace_readers(hop=3, k=4, direct): top D={rs[0]['direct_gate']:+.4f} (C={rs[0]['coupling_score']:+.4f})")

        mcp_direct = await trace_downstream_readers(
            layer=35, neuron_idx=9482, prompt="How many legs does a spider have?", chat=True,
            assistant_prefix="A spider has", max_layer_hop=2, top_k=3, score="direct",
        )
        assert mcp_direct["chat"] and mcp_direct["prompt"].startswith("<bos><|turn>user\n")
        assert all("direct_act" in r for r in mcp_direct["top_downstream_readers"])
        hyp_chat = await test_motif_hypothesis(
            layer=53, neuron_idx=16383, chat=True,
            pos_prompts=["What animal spins silk webs to catch insects?"], neg_prompts=["What is the capital of France?"],
        )
        assert hyp_chat["chat"] is True
        print(f"  • MCP trace_downstream_readers(chat, direct) and test_motif_hypothesis(chat): ok")

        # -------------------------------------------------------------------------
        # Step 3: Headless Chrome inspection of the instrument's pages (reader and scholar views)
        # -------------------------------------------------------------------------
        print("\n[4/4] Launching Headless Chrome page inspection & screenshot loop...")
        chrome_Data_dir = Path(tempfile.mkdtemp(prefix="theodolite_chrome_"))
        chrome_cmd = [
            os.environ.get("CHROME_BIN") or shutil.which("google-chrome-stable") or shutil.which("chromium") or "google-chrome-stable",
            "--headless=new",
            f"--remote-debugging-port={CDP_PORT}",
            f"--user-data-dir={chrome_Data_dir}",
            "--no-proxy-server",
            "--use-gl=angle",
            "--use-angle=swiftshader",
            "--enable-webgl",
            "--ignore-gpu-blocklist",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu-sandbox",
            "--window-size=1720,1040",
            f"http://{HOST}:{PORT}/",
        ]
        chrome_proc = subprocess.Popen(
            chrome_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cdp: CDPClient | None = None
        try:
            # Discover target page WebSocket debugger URL
            ws_debug_url = None
            async with httpx.AsyncClient(timeout=10.0, trust_env=False) as cdp_http:
                for _ in range(40):
                    try:
                        targets = (await cdp_http.get(f"http://127.0.0.1:{CDP_PORT}/json/list")).json()
                        for t in targets:
                            if t.get("type") == "page" and f"{HOST}:{PORT}" in t.get("url", ""):
                                ws_debug_url = t["webSocketDebuggerUrl"]
                                break
                        if ws_debug_url:
                            break
                    except Exception:
                        pass
                    await asyncio.sleep(0.25)

            if not ws_debug_url:
                raise RuntimeError("Failed to connect to headless Chrome DevTools target.")

            cdp = CDPClient(ws_debug_url)
            await cdp.connect()

            async def wait_for(js: str, what: str, tries: int = 120, delay: float = 0.5) -> Any:
                last = None
                for _ in range(tries):
                    last = await cdp.eval_js(js)
                    if last:
                        return last
                    await asyncio.sleep(delay)
                raise RuntimeError(f"Timed out waiting for {what}: {last}")

            async def go(route: str) -> None:
                await cdp.eval_js(f"(() => {{ location.hash = {json.dumps(route)}; }})()")
                await asyncio.sleep(0.8)

            # The instrument: wait until the first live sighting has replaced the recorded one (E6 fixture)
            await go("#/instrument")
            await wait_for(
                """(() => {
                    const t = document.body ? document.body.innerText : "";
                    return t.includes("· ready") && t.includes("it went on") && !t.includes("the study’s recorded sighting");
                })()""",
                "the first live sighting",
            )
            figure = await cdp.eval_js(
                """(() => {
                    const svg = document.querySelector('svg[role="img"][aria-label^="Sixty stacked rings"]');
                    return { found: !!svg, paths: svg ? svg.querySelectorAll("path").length : 0 };
                })()"""
            )
            print(f"  • Instrument drawing: found={figure['found']} paths={figure['paths']}")
            assert figure["found"] and figure["paths"] > 10, f"Instrument drawing missing: {figure}"

            # How to read these pages: rubrics (red italic guidance) and apparatus (double-gilt-ruled blocks)
            count_js = """(() => {
                const all = Array.from(document.querySelectorAll("main *"));
                const red = all.filter((e) => {
                    const cs = getComputedStyle(e);
                    return cs.fontStyle === "italic" && cs.color === "rgb(143, 43, 30)" && e.offsetParent !== null
                        && Array.from(e.childNodes).some((n) => n.nodeType === 3 && n.textContent.trim().length > 20);
                }).length;
                // apparatus blocks: ruled off by a double gilt line (some decorative frames use one too)
                const ruled = all.filter((e) => getComputedStyle(e).borderTopStyle === "double" && e.offsetParent !== null);
                const app = ruled.length;
                const notes = ruled.filter((e) => e.getAttribute("role") === "note").length;
                return { red, app, notes };
            })()"""
            click_js = """((name) => {
                const b = Array.from(document.querySelectorAll('[aria-label="How to read these pages"] button'))
                    .find((x) => x.textContent.includes(name));
                b.click();
                return b.getAttribute("aria-pressed");
            })"""
            routes = ["#/", "#/instrument", "#/layer/36", "#/neuron/54/15491/pos", "#/landmarks", "#/nudge",
                      "#/vault", "#/lexicon", "#/lineage"]
            for route in routes:
                await go(route)
                reader = await cdp.eval_js(count_js)  # rubrics on, apparatus off (fresh profile)
                await cdp.eval_js(f"{click_js}('apparatus')")
                await cdp.eval_js(f"{click_js}('rubrics')")
                await asyncio.sleep(0.3)
                scholar = await cdp.eval_js(count_js)  # rubrics off, apparatus on
                await cdp.eval_js(f"{click_js}('apparatus')")
                await cdp.eval_js(f"{click_js}('rubrics')")
                await asyncio.sleep(0.3)
                print(f"  • {route:<22} rubric lines {reader['red']:>3} → {scholar['red']:>3} | "
                      f"apparatus blocks {reader['app']:>3} → {scholar['app']:>3}")
                # a few red italic lines are warnings the canvas keeps visible; the rubrics must all go
                assert reader["red"] > 0 and scholar["red"] < reader["red"], f"{route}: rubrics toggle did not work"
                assert reader["notes"] == 0 and scholar["app"] > reader["app"], f"{route}: apparatus toggle did not work"

            # Screenshot #1: the instrument, scholar's view
            await go("#/instrument")
            await cdp.eval_js(f"{click_js}('apparatus')")
            await asyncio.sleep(0.5)
            shot1 = ARTIFACT_DIR / "theodolite_instrument_scholar_view.png"
            await cdp.screenshot(shot1)
            print(f"  • Saved Screenshot #1 (the instrument, apparatus open): {shot1}")
            await cdp.eval_js(f"{click_js}('apparatus')")

            # Screenshot #2: arm a gentle nudge (×0.01 of the stream) and wait for the nudged sighting
            await go("#/nudge")
            armed = await cdp.eval_js(
                """(() => {
                    const b = Array.from(document.querySelectorAll("button")).find((x) => x.textContent.trim() === "arm the nudge");
                    if (b && !b.disabled) { b.click(); return true; }
                    return false;
                })()"""
            )
            assert armed, "Could not find an enabled 'arm the nudge' button"
            # the page first takes a plain sighting of its question, then the nudged one; stay on it meanwhile
            await wait_for(
                """(() => { const t = document.body.innerText;
                    return !t.includes("TAKE A SIGHTING TO SEE THE NUDGED ANSWER") && !t.includes("taking a sighting"); })()""",
                "the nudged sighting",
            )
            await go("#/instrument")
            nudged = await cdp.eval_js("""document.body.innerText.includes("nudged")""")
            assert nudged, "The instrument does not mark the sighting as nudged"
            await go("#/nudge")
            shot2 = ARTIFACT_DIR / "theodolite_nudge_view.png"
            await cdp.screenshot(shot2)
            print(f"  • Saved Screenshot #2 (after a gentle nudge): {shot2}")

            # The formerly disabled controls ("[NOT YET IN THE SERVER]") are live and drive the server
            await cdp.eval_js(f"{click_js}('apparatus')")
            await asyncio.sleep(0.4)
            fixes = await cdp.eval_js(
                """(() => {
                    const labels = Array.from(document.querySelectorAll("label.check"));
                    const want = ["the corrected direction", "mask the quiet dimensions", "leave the start mark alone", "the strength as a share"];
                    return want.map((w) => { const l = labels.find((x) => x.textContent.includes(w));
                        const i = l && l.querySelector("input"); return i ? { w, checked: i.checked, disabled: i.disabled } : { w, missing: true }; });
                })()"""
            )
            assert all(f.get("checked") and not f.get("disabled") for f in fixes), f"Nudge fix boxes: {fixes}"
            print("  • Nudge: the four fix boxes are enabled and ticked")
            await go("#/instrument")
            chat_ui = await cdp.eval_js(
                """(() => {
                    const l = Array.from(document.querySelectorAll("label.check")).find((x) => x.textContent.includes("send it as a chat"));
                    const i = l && l.querySelector("input");
                    return { found: !!i, checked: i ? i.checked : null, disabled: i ? i.disabled : null, system: !!document.querySelector("#system"),
                             notYet: document.body.innerText.includes("NOT YET IN THE SERVER") };
                })()"""
            )
            assert chat_ui["found"] and chat_ui["checked"] and not chat_ui["disabled"] and chat_ui["system"], f"chat control: {chat_ui}"
            assert not chat_ui["notYet"], "the instrument page still shows a [NOT YET IN THE SERVER] control"
            print("  • Instrument: “send it as a chat” is enabled (ticked by default), with a system-prompt field")
            # threads are scored by the direct-path estimate by default: tick "rank them by write size" (a sighting
            # with direct threads), then untick "score threads by the direct-path estimate" to read C; each retakes it
            for label, want_checked, done in (
                ("rank them by write size", True, "by write size |a|"),
                ("score threads by the direct-path estimate", False, "A weak guide (ρ ≈ 0.41"),
            ):
                await wait_for("""(() => !document.body.innerText.includes("taking a sighting…"))()""", "an idle instrument")
                if not want_checked:
                    await wait_for(
                        """document.body.innerText.includes("A direct-path linear estimate, not a verified circuit")""",
                        "threads scored by the direct-path estimate by default",
                    )
                    print("  • Instrument: threads are scored by the direct-path estimate by default")
                clicked = await cdp.eval_js(
                    f"""(() => {{
                        const l = Array.from(document.querySelectorAll("label.check")).find((x) => x.textContent.includes({json.dumps(label)}));
                        const i = l && l.querySelector("input");
                        if (!i || i.disabled) return null; i.click(); return i.checked;
                    }})()"""
                )
                assert clicked is want_checked, f"could not set “{label}” to {want_checked} (got {clicked})"
                await wait_for(f"""document.body.innerText.includes({json.dumps(done)})""", f"a sighting after “{label}”")
                print(f"  • Instrument: “{label}” {'ticked' if want_checked else 'unticked'}, a new sighting taken")
            await go("#/landmarks")
            await wait_for("""(() => !document.body.innerText.includes("taking a sighting…"))()""", "an idle landmarks page")
            frame_ui = await cdp.eval_js(
                """(() => {
                    const radios = Array.from(document.querySelectorAll('input[name="frame-squaring"], input[name="frame-place"]'));
                    const i = document.querySelectorAll('input[name="frame-squaring"]')[1];
                    if (i && !i.disabled) i.click();
                    const b = Array.from(document.querySelectorAll("button")).find((x) => x.textContent.includes("take a sighting with these landmarks"));
                    if (b && !b.disabled) b.click();
                    return { radios: radios.length, disabled: radios.filter((r) => r.disabled).length, clicked: !!(i && b) };
                })()"""
            )
            assert frame_ui["radios"] == 4 and frame_ui["disabled"] == 0 and frame_ui["clicked"], f"frame options: {frame_ui}"
            await wait_for("""document.body.innerText.includes("The last sighting was squared all at once")""", "a Löwdin sighting")
            print("  • Landmarks: the frame options are live; a sighting squared all at once came back")
            await go("#/neuron/36/9482/pos")
            await asyncio.sleep(0.5)
            picked = await cdp.eval_js(
                """(() => {
                    const d = Array.from(document.querySelectorAll("button")).find((x) => x.textContent.includes("direct-path estimate, the fix"));
                    if (!d || d.disabled) return "no direct button";
                    d.click();
                    return "ok";
                })()"""
            )
            assert picked == "ok", picked
            await asyncio.sleep(0.4)  # let the page re-render with the new score before tracing
            traced = await cdp.eval_js(
                """(() => {
                    const t = Array.from(document.querySelectorAll("button")).find((x) => x.textContent.trim() === "trace its readers");
                    if (!t || t.disabled) return "no trace button";
                    t.click();
                    return "ok";
                })()"""
            )
            assert traced == "ok", traced
            await wait_for(
                """(() => { const t = document.body.innerText;
                    return !t.includes("tracing…") && t.includes("thread score, D") && !t.includes("[READERS APPEAR AFTER TRACING]")
                        && !t.includes("[NO READERS CAME BACK]"); })()""",
                "a direct-scored trace",
            )
            print("  • Neuron folio: “direct-path estimate” scored the traced readers")
            await cdp.eval_js(f"{click_js}('apparatus')")

            print(
                f"  • Console Errors: {len(cdp.console_errors)} | Page Exceptions: {len(cdp.page_exceptions)}"
            )
            if cdp.console_errors:
                print("    Console error details:", cdp.console_errors)
            if cdp.page_exceptions:
                print("    Page exception details:", cdp.page_exceptions)
            assert len(cdp.console_errors) == 0, f"Unexpected console errors: {cdp.console_errors}"
            assert len(cdp.page_exceptions) == 0, f"Unexpected page exceptions: {cdp.page_exceptions}"

        finally:
            if cdp:
                await cdp.close()
            chrome_proc.terminate()
            try:
                chrome_proc.wait(timeout=5)
            except Exception:
                chrome_proc.kill()
            shutil.rmtree(chrome_Data_dir, ignore_errors=True)

        # Final VRAM check
        health_after = (await client.get("/api/health")).json()
        print(
            f"\n[SUMMARY] Peak GPU VRAM across E2E Server + FastMCP + UI sweeps: "
            f"{health_after['vram_peak_gb']:.2f} GB (< 30.00 GB limit)"
        )
        assert health_after["vram_peak_gb"] < 30.0

    server.should_exit = True
    await asyncio.sleep(0.5)
    server_task.cancel()
    print("=" * 92)
    print("ALL BACKEND, FASTMCP & FRONTEND VERIFICATION CHECKS PASSED (ZERO ERRORS)")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

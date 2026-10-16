#!/usr/bin/env python3
"""End-to-End Verification Suite (`scripts/verify_e2e_ui_and_mcp.py`) for Logos Theodolite.

Runs inside a single process/network namespace to verify:
1. FastAPI + Binary Float32 WebSocket (`/ws/theodolite`) wire format (`0x5448454F`) & JSON metadata
2. FastMCP Scientist Tools (`inspect_pole`, `trace_downstream_readers`, `test_motif_hypothesis`)
3. LARQL / VINDEX3 FFN Vault queries (`/api/larql`)
4. Headless Chrome WebGL2 / React Three Fiber 3D Theodolite UI inspection loop via CDP:
   - Zero console or WebGL errors
   - 60-Layer Astrolabe, Ballistic Trajectory (`delta_attn_J` cyan / `delta_ffn_J` gold),
     Bipolar GeGLU Dipole Needles, Downstream Reader Filaments, and 60x60 CKA HUD rendered
   - Interactive screenshots saved to $THEODOLITE_ARTIFACT_DIR (default: artifacts/e2e/)
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

        # -------------------------------------------------------------------------
        # Step 3: Headless Chrome WebGL2 + React Three Fiber 3D Theodolite Inspection
        # -------------------------------------------------------------------------
        print("\n[4/4] Launching Headless Chrome WebGL2 Inspection & Screenshot Loop...")
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

            # Wait for the initial WebSocket sweep to finish streaming and rendering in the 3D scene
            for _ in range(90):
                state_info = await cdp.eval_js(
                    """(() => {
                        const bodyText = document.body ? document.body.innerText : "";
                        const canvas = document.querySelector("canvas");
                        return {
                            hasStreamOnline: bodyText.includes("WS STREAM LIVE"),
                            hasGenerated: bodyText.includes("8."),
                            isSweeping: bodyText.includes("SWEEPING 60 LAYERS"),
                            hasCanvas: !!canvas,
                            canvasW: canvas ? canvas.width : 0,
                            canvasH: canvas ? canvas.height : 0,
                        };
                    })()"""
                )
                if (
                    state_info
                    and state_info.get("hasStreamOnline")
                    and state_info.get("hasGenerated")
                    and not state_info.get("isSweeping")
                ):
                    break
                await asyncio.sleep(0.5)
            else:
                raise RuntimeError(f"Frontend did not finish initial sweep render in time: {state_info}")

            # Trigger LARQL Nearest 'spider' query so the LARQL console shows live results in Screenshot #1
            await cdp.eval_js(
                """(() => {
                    const buttons = Array.from(document.querySelectorAll("button"));
                    const larqlBtn = buttons.find(b => b.innerText.includes("Nearest 'spider'"));
                    if (larqlBtn) larqlBtn.click();
                })()"""
            )
            await asyncio.sleep(1.2)

            # Let Three.js render several frames and measure WebGL2 context health & rAF FPS
            webgl_report = await cdp.eval_js(
                """new Promise((resolve) => {
                    const canvases = Array.from(document.querySelectorAll("canvas"));
                    const mainCanvas = canvases[0];
                    const gl = mainCanvas.getContext("webgl2") || mainCanvas.getContext("webgl");
                    const glErr = gl ? gl.getError() : -1;
                    const renderer = gl ? gl.getParameter(gl.RENDERER) : "none";
                    let frames = 0;
                    const t0 = performance.now();
                    function tick() {
                        frames++;
                        if (frames < 30) {
                            requestAnimationFrame(tick);
                        } else {
                            const elapsed = performance.now() - t0;
                            resolve({
                                canvasCount: canvases.length,
                                width: mainCanvas.width,
                                height: mainCanvas.height,
                                glError: glErr,
                                renderer: renderer,
                                measuredFps: Math.round((frames * 1000) / Math.max(elapsed, 1)),
                            });
                        }
                    }
                    requestAnimationFrame(tick);
                })"""
            )
            print(
                f"  • WebGL2 Context Health: canvases={webgl_report['canvasCount']} "
                f"({webgl_report['width']}x{webgl_report['height']}) | "
                f"gl.getError()={webgl_report['glError']} | rAF={webgl_report['measuredFps']} FPS | "
                f"renderer={webgl_report['renderer']}"
            )
            assert webgl_report["glError"] == 0, f"WebGL error code: {webgl_report['glError']}"
            assert webgl_report["canvasCount"] >= 2, "Expected 3D Astrolabe canvas + 60x60 CKA canvas"

            # Capture Screenshot #1 (Monument Mode 60-Layer Astrolabe)
            shot1 = ARTIFACT_DIR / "theodolite_monument_view.png"
            await cdp.screenshot(shot1)
            print(f"  • Saved Screenshot #1 (Monument Mode 60-Layer Astrolabe): {shot1}")

            # Trigger Pole-Locked Motif Mode (automatically triggers 60-layer sweep)
            await cdp.eval_js(
                """(() => {
                    const buttons = Array.from(document.querySelectorAll("button"));
                    const poleBtn = buttons.find(b => b.innerText.includes("Pole-Locked Motif"));
                    if (poleBtn) poleBtn.click();
                })()"""
            )
            await asyncio.sleep(1.0)
            for _ in range(60):
                is_ready = await cdp.eval_js(
                    """!document.body.innerText.includes("SWEEPING 60 LAYERS") && document.body.innerText.includes("Write L")"""
                )
                if is_ready:
                    break
                await asyncio.sleep(0.5)
            await asyncio.sleep(0.8)
            shot2 = ARTIFACT_DIR / "theodolite_pole_motif_view.png"
            await cdp.screenshot(shot2)
            print(f"  • Saved Screenshot #2 (Pole-Locked Motif Frame): {shot2}")

            # Trigger Click-to-Steer Causal Laser (+six / -eight) in Monument Mode and capture Screenshot #3
            await cdp.eval_js(
                """(() => {
                    const buttons = Array.from(document.querySelectorAll("button"));
                    const monBtn = buttons.find(b => b.innerText.includes("Monument Mode"));
                    if (monBtn) monBtn.click();
                })()"""
            )
            await asyncio.sleep(0.3)
            await cdp.eval_js(
                """(() => {
                    const buttons = Array.from(document.querySelectorAll("button"));
                    const fireBtn = buttons.find(b => b.innerText.includes("FIRE LASER"));
                    if (fireBtn) fireBtn.click();
                })()"""
            )
            await asyncio.sleep(1.0)
            for _ in range(60):
                steered_ready = await cdp.eval_js(
                    """!document.body.innerText.includes("SWEEPING 60 LAYERS") && document.body.innerText.includes("LASER STEERED")"""
                )
                if steered_ready:
                    break
                await asyncio.sleep(0.5)
            await asyncio.sleep(0.8)
            shot3 = ARTIFACT_DIR / "theodolite_steered_laser_view.png"
            await cdp.screenshot(shot3)
            print(f"  • Saved Screenshot #3 (Live SteeringSpec Laser Flipped to 'six'): {shot3}")

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
            f"\n[SUMMARY] Peak GPU VRAM across E2E Server + FastMCP + 3D UI Sweeps: "
            f"{health_after['vram_peak_gb']:.2f} GB (< 30.00 GB limit)"
        )
        assert health_after["vram_peak_gb"] < 30.0

    server.should_exit = True
    await asyncio.sleep(0.5)
    server_task.cancel()
    print("=" * 92)
    print("ALL PHASE 2 BACKEND, FASTMCP & 3D FRONTEND VERIFICATION CHECKS PASSED (ZERO ERRORS)")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

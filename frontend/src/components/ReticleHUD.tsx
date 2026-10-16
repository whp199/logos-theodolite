import React, { useEffect, useRef } from "react";
import { useTheodoliteStore } from "../store/useTheodoliteStore";

const CkaMiniMap: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const ckaMatrix = useTheodoliteStore((s) => s.ckaMatrix);
  const selectedLayer = useTheodoliteStore((s) => s.selectedLayer);
  const setSelectedLayer = useTheodoliteStore((s) => s.setSelectedLayer);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const size = 180;
    const n = 60;
    const cell = size / n;
    ctx.clearRect(0, 0, size, size);

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const v = Math.max(0, Math.min(1, ckaMatrix[i * n + j] ?? 0));
        const r = Math.round(12 + Math.pow(v, 1.8) * 239);
        const g = Math.round(22 + Math.pow(v, 1.1) * 185);
        const b = Math.round(48 + (1 - Math.pow(v, 2.2)) * 195);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(j * cell, i * cell, Math.ceil(cell), Math.ceil(cell));
      }
    }

    // Highlight selectedLayer crosshair
    const pos = (selectedLayer + 0.5) * cell;
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(0, pos);
    ctx.lineTo(size, pos);
    ctx.moveTo(pos, 0);
    ctx.lineTo(pos, size);
    ctx.stroke();
  }, [ckaMatrix, selectedLayer]);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const y = e.clientY - rect.top;
    const layer = Math.max(0, Math.min(59, Math.floor((y / rect.height) * 60)));
    setSelectedLayer(layer);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
      <canvas
        ref={canvasRef}
        width={180}
        height={180}
        onClick={handleClick}
        title="Click any row/col in the 60x60 Linear CKA Matrix to jump to that layer"
        style={{
          width: 180,
          height: 180,
          borderRadius: 6,
          border: "1px solid rgba(0, 229, 255, 0.4)",
          cursor: "crosshair",
        }}
      />
      <div style={{ fontSize: 10, color: "#94a3b8" }}>
        60×60 Linear CKA Similarity (Click to Inspect Layer)
      </div>
    </div>
  );
};

export const ReticleHUD: React.FC = () => {
  const wsConnected = useTheodoliteStore((s) => s.wsConnected);
  const isSweeping = useTheodoliteStore((s) => s.isSweeping);
  const prompt = useTheodoliteStore((s) => s.prompt);
  const monuments = useTheodoliteStore((s) => s.monuments);
  const surveyMode = useTheodoliteStore((s) => s.surveyMode);
  const compareThink = useTheodoliteStore((s) => s.compareThink);
  const steering = useTheodoliteStore((s) => s.steering);
  const selectedLayer = useTheodoliteStore((s) => s.selectedLayer);
  const selectedPoleRank = useTheodoliteStore((s) => s.selectedPoleRank);
  const showAttnSplit = useTheodoliteStore((s) => s.showAttnSplit);
  const showFfnSplit = useTheodoliteStore((s) => s.showFfnSplit);
  const showDipoles = useTheodoliteStore((s) => s.showDipoles);
  const showFilaments = useTheodoliteStore((s) => s.showFilaments);
  const showPlumes = useTheodoliteStore((s) => s.showPlumes);

  const trajXyz = useTheodoliteStore((s) => s.trajXyz);
  const trajAnglesNorm = useTheodoliteStore((s) => s.trajAnglesNorm);
  const deltaAttnXyz = useTheodoliteStore((s) => s.deltaAttnXyz);
  const deltaFfnXyz = useTheodoliteStore((s) => s.deltaFfnXyz);
  const polesData = useTheodoliteStore((s) => s.polesData);
  const nPolesPerLayer = useTheodoliteStore((s) => s.nPolesPerLayer);

  const generatedText = useTheodoliteStore((s) => s.generatedText);
  const thinkGeneratedText = useTheodoliteStore((s) => s.thinkGeneratedText);
  const surveyLabels = useTheodoliteStore((s) => s.surveyLabels);
  const elapsedMs = useTheodoliteStore((s) => s.elapsedMs);
  const peakVramGb = useTheodoliteStore((s) => s.peakVramGb);
  const isSteered = useTheodoliteStore((s) => s.isSteered);
  const layerPlumes = useTheodoliteStore((s) => s.layerPlumes);
  const polesLabels = useTheodoliteStore((s) => s.polesLabels);
  const filamentsMeta = useTheodoliteStore((s) => s.filamentsMeta);
  const larqlQuery = useTheodoliteStore((s) => s.larqlQuery);
  const larqlOutput = useTheodoliteStore((s) => s.larqlOutput);
  const poleInspection = useTheodoliteStore((s) => s.poleInspection);
  const activePatchesCount = useTheodoliteStore((s) => s.activePatchesCount);
  const fps = useTheodoliteStore((s) => s.fps);

  const setPrompt = useTheodoliteStore((s) => s.setPrompt);
  const setMonuments = useTheodoliteStore((s) => s.setMonuments);
  const setSurveyMode = useTheodoliteStore((s) => s.setSurveyMode);
  const setCompareThink = useTheodoliteStore((s) => s.setCompareThink);
  const setSteering = useTheodoliteStore((s) => s.setSteering);
  const setSelectedLayer = useTheodoliteStore((s) => s.setSelectedLayer);
  const setSelectedPoleRank = useTheodoliteStore((s) => s.setSelectedPoleRank);
  const toggleLayerVisibility = useTheodoliteStore((s) => s.toggleLayerVisibility);
  const setLarqlQuery = useTheodoliteStore((s) => s.setLarqlQuery);
  const triggerSweep = useTheodoliteStore((s) => s.triggerSweep);
  const executeLarql = useTheodoliteStore((s) => s.executeLarql);
  const inspectPole = useTheodoliteStore((s) => s.inspectPole);

  const x = trajXyz[selectedLayer * 3 + 0] ?? 0;
  const y = trajXyz[selectedLayer * 3 + 1] ?? 0;
  const z = trajXyz[selectedLayer * 3 + 2] ?? 0;
  const azimuth = trajAnglesNorm[selectedLayer * 3 + 0] ?? 0;
  const elevation = trajAnglesNorm[selectedLayer * 3 + 1] ?? 0;
  const residNorm = trajAnglesNorm[selectedLayer * 3 + 2] ?? 0;
  const attnNorm = deltaAttnXyz[selectedLayer * 4 + 3] ?? 0;
  const ffnNorm = deltaFfnXyz[selectedLayer * 4 + 3] ?? 0;
  const ffnRatio = ffnNorm / (attnNorm + ffnNorm + 1e-6);

  const currentPlume = layerPlumes[selectedLayer];
  const currentLayerPoles = polesLabels[String(selectedLayer)] ?? [];

  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        pointerEvents: "none",
        display: "flex",
        flexDirection: "column",
        justifyContent: "space-between",
        padding: "12px",
        zIndex: 10,
      }}
    >
      {/* TOP BAR */}
      <div
        className="hud-glass"
        style={{
          pointerEvents: "auto",
          padding: "10px 14px",
          display: "flex",
          flexDirection: "column",
          gap: "8px",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontSize: 15, fontWeight: 800, letterSpacing: "0.06em", color: "#00e5ff" }}>
              LOGOS THEODOLITE
            </span>
            <span style={{ fontSize: 11, color: "#94a3b8" }}>
              google/gemma-4-31b-it (W4A16) • d_model=5376 • L=60 • J++ Lens • J-VINDEX3
            </span>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 12, fontSize: 11 }}>
            <span style={{ color: wsConnected ? "#4ade80" : "#fbbf24", fontWeight: 700 }}>
              ● {wsConnected ? "WS STREAM LIVE" : "LOCAL SNAPSHOT"}
            </span>
            <span style={{ color: "#38bdf8" }}>VRAM: {peakVramGb.toFixed(2)} / 30.0 GB</span>
            <span style={{ color: "#fbbf24" }}>Sweep: {elapsedMs.toFixed(0)} ms</span>
            <span style={{ color: "#4ade80", fontWeight: 700 }}>{fps} FPS</span>
          </div>
        </div>

        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <input
            className="hud-input"
            style={{ flex: 1, minWidth: 280 }}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter multi-hop mechanistic prompt..."
          />
          <label style={{ fontSize: 11, display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={compareThink}
              onChange={(e) => setCompareThink(e.target.checked)}
            />
            Compare &lt;|think|&gt;
          </label>
          <button className="hud-btn" onClick={triggerSweep} disabled={isSweeping}>
            {isSweeping ? "SWEEPING 60 LAYERS..." : "RUN 60-LAYER SWEEP"}
          </button>

          <div style={{ display: "flex", gap: 5 }}>
            <button
              className="hud-btn"
              style={{ opacity: showAttnSplit ? 1 : 0.45, borderColor: "#00e5ff" }}
              onClick={() => toggleLayerVisibility("showAttnSplit")}
            >
              Δ_attn (Cyan)
            </button>
            <button
              className="hud-btn hud-btn-gold"
              style={{ opacity: showFfnSplit ? 1 : 0.45 }}
              onClick={() => toggleLayerVisibility("showFfnSplit")}
            >
              Δ_ffn (Gold)
            </button>
            <button
              className="hud-btn"
              style={{ opacity: showDipoles ? 1 : 0.45 }}
              onClick={() => toggleLayerVisibility("showDipoles")}
            >
              Dipoles (±)
            </button>
            <button
              className="hud-btn"
              style={{ opacity: showFilaments ? 1 : 0.45 }}
              onClick={() => toggleLayerVisibility("showFilaments")}
            >
              Filaments
            </button>
            <button
              className="hud-btn"
              style={{ opacity: showPlumes ? 1 : 0.45 }}
              onClick={() => toggleLayerVisibility("showPlumes")}
            >
              Plumes
            </button>
          </div>
        </div>

        <div style={{ display: "flex", gap: 16, fontSize: 11, color: "#cbd5e1", alignItems: "center" }}>
          <div>
            <span style={{ color: "#94a3b8" }}>Output Continuation: </span>
            <strong style={{ color: isSteered ? "#f43f5e" : "#4ade80" }}>
              {JSON.stringify(generatedText)}
            </strong>
            {isSteered && (
              <span style={{ marginLeft: 6, color: "#f43f5e", fontWeight: 700 }}>[LASER STEERED]</span>
            )}
          </div>
          {thinkGeneratedText && (
            <div>
              <span style={{ color: "#c084fc" }}>&lt;|think|&gt; Output: </span>
              <strong style={{ color: "#e9d5ff" }}>{JSON.stringify(thinkGeneratedText)}</strong>
            </div>
          )}
        </div>
      </div>

      {/* MIDDLE ROW: LEFT RETICLE PANEL + RIGHT CIRCUIT & VAULT PANEL */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "stretch",
          flex: 1,
          marginTop: 10,
          gap: 12,
          minHeight: 0,
        }}
      >
        {/* LEFT PANEL: THEODOLITE RETICLE, SURVEY FRAME & 60x60 CKA MINI-MAP */}
        <div
          className="hud-glass hud-scroller"
          style={{
            pointerEvents: "auto",
            width: 335,
            padding: 12,
            overflowY: "auto",
            display: "flex",
            flexDirection: "column",
            gap: 10,
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: 12, fontWeight: 700, color: "#00e5ff" }}>
              THEODOLITE RETICLE HUD
            </span>
            <span style={{ fontSize: 11, color: "#fbbf24", fontWeight: 700 }}>
              Layer L{selectedLayer + 1} (idx={selectedLayer})
              {currentPlume?.is_global_attention ? " ★ GLOBAL" : " • Local"}
            </span>
          </div>

          <input
            type="range"
            min={0}
            max={59}
            value={selectedLayer}
            onChange={(e) => setSelectedLayer(Number(e.target.value))}
            style={{ width: "100%" }}
          />

          {/* Azimuth, Elevation, Residual Norm */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr 1fr",
              gap: 6,
              background: "rgba(3, 7, 18, 0.72)",
              padding: 8,
              borderRadius: 6,
              border: "1px solid rgba(56, 189, 248, 0.22)",
            }}
          >
            <div>
              <div style={{ fontSize: 10, color: "#94a3b8" }}>Azimuth θ</div>
              <div style={{ fontSize: 14, fontWeight: 700, color: "#00e5ff" }}>
                {azimuth.toFixed(1)}°
              </div>
            </div>
            <div>
              <div style={{ fontSize: 10, color: "#94a3b8" }}>Elevation φ</div>
              <div style={{ fontSize: 14, fontWeight: 700, color: "#fbbf24" }}>
                {elevation.toFixed(1)}°
              </div>
            </div>
            <div>
              <div style={{ fontSize: 10, color: "#94a3b8" }}>‖h_l‖₂ Norm</div>
              <div style={{ fontSize: 14, fontWeight: 700, color: "#f8fafc" }}>
                {residNorm.toFixed(1)}
              </div>
            </div>
          </div>

          {/* 3D Coordinates in Survey Frame */}
          <div style={{ fontSize: 11, color: "#cbd5e1" }}>
            <div style={{ color: "#94a3b8", marginBottom: 3 }}>
              Gram-Schmidt Survey Frame ({surveyMode === "monument" ? "Monument Mode" : "Pole-Locked Motif"}):
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "monospace" }}>
              <span style={{ color: "#00e5ff" }}>
                x[{surveyLabels[0]}]: {x.toFixed(2)}
              </span>
              <span style={{ color: "#fbbf24" }}>
                y[{surveyLabels[1]}]: {y.toFixed(2)}
              </span>
              <span style={{ color: "#c084fc" }}>
                z[{surveyLabels[2]}]: {z.toFixed(2)}
              </span>
            </div>
          </div>

          {/* Monument Mode Inputs */}
          <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
            <div style={{ display: "flex", gap: 4 }}>
              <button
                className="hud-btn"
                style={{ flex: 1, opacity: surveyMode === "monument" ? 1 : 0.5 }}
                onClick={() => setSurveyMode("monument", null)}
              >
                Monument Mode
              </button>
              <button
                className="hud-btn hud-btn-gold"
                style={{ flex: 1, opacity: surveyMode === "pole_motif" ? 1 : 0.5 }}
                onClick={() => {
                  const base = (selectedLayer * nPolesPerLayer + selectedPoleRank) * 6;
                  const nIdx = Math.round(polesData[base + 0] ?? 0);
                  const pol = (polesData[base + 1] ?? 1) >= 0 ? 1 : -1;
                  setSurveyMode("pole_motif", [selectedLayer, nIdx, pol]);
                  setTimeout(() => triggerSweep(), 20);
                }}
              >
                Pole-Locked Motif
              </button>
            </div>
            {surveyMode === "monument" && (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 4 }}>
                {monuments.map((m, idx) => (
                  <input
                    key={idx}
                    className="hud-input"
                    value={m}
                    onChange={(e) => {
                      const next: [string, string, string] = [...monuments] as [string, string, string];
                      next[idx] = e.target.value;
                      setMonuments(next);
                    }}
                    placeholder={`Monument ${idx + 1}`}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Module 1: Attention Router vs. FFN Vault Split */}
          <div
            style={{
              background: "rgba(3, 7, 18, 0.72)",
              padding: 8,
              borderRadius: 6,
              border: "1px solid rgba(251, 191, 36, 0.25)",
            }}
          >
            <div style={{ fontSize: 10, color: "#94a3b8", marginBottom: 4 }}>
              Causal Split in J++-Space (Attn Router vs. FFN Vault):
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 4 }}>
              <span style={{ color: "#00e5ff" }}>‖Δ_attn,J‖: {attnNorm.toFixed(2)}</span>
              <span style={{ color: "#fbbf24" }}>‖Δ_ffn,J‖: {ffnNorm.toFixed(2)}</span>
            </div>
            <div
              style={{
                width: "100%",
                height: 7,
                background: "#00e5ff",
                borderRadius: 4,
                overflow: "hidden",
                display: "flex",
              }}
            >
              <div
                style={{
                  width: `${((1 - ffnRatio) * 100).toFixed(1)}%`,
                  background: "#00e5ff",
                }}
              />
              <div
                style={{
                  width: `${(ffnRatio * 100).toFixed(1)}%`,
                  background: "#fbbf24",
                }}
              />
            </div>
          </div>

          {/* J++ Ghost Token Plumes at Selected Layer */}
          {currentPlume && (
            <div
              style={{
                background: "rgba(3, 7, 18, 0.72)",
                padding: 8,
                borderRadius: 6,
                fontSize: 11,
              }}
            >
              <div style={{ color: "#94a3b8", fontSize: 10, marginBottom: 4 }}>
                J++ Readout Plume at L{selectedLayer + 1} (vs. Naive: {JSON.stringify(currentPlume.naive_top_token)}):
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                {currentPlume.jpp_top_tokens.map((t, idx) => (
                  <span
                    key={idx}
                    style={{
                      background: "rgba(56, 189, 248, 0.14)",
                      border: "1px solid rgba(56, 189, 248, 0.35)",
                      borderRadius: 4,
                      padding: "2px 5px",
                      fontSize: 10,
                    }}
                  >
                    <strong style={{ color: "#38bdf8" }}>{JSON.stringify(t.token)}</strong>{" "}
                    <span style={{ color: "#94a3b8" }}>({t.logit.toFixed(1)})</span>
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* 60x60 Linear CKA Similarity Mini-Map */}
          <CkaMiniMap />
        </div>

        {/* RIGHT PANEL: BIPOLAR POLES, CLICK-TO-STEER LASER & LARQL VAULT */}
        <div
          className="hud-glass hud-scroller"
          style={{
            pointerEvents: "auto",
            width: 375,
            padding: 12,
            overflowY: "auto",
            display: "flex",
            flexDirection: "column",
            gap: 10,
          }}
        >
          {/* Module 2 & 3: Active Bipolar GeGLU Poles */}
          <div>
            <div style={{ fontSize: 12, fontWeight: 700, color: "#fbbf24", marginBottom: 6 }}>
              BIPOLAR GeGLU POLES • L{selectedLayer + 1} (J-VINDEX vs. Naive)
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
              {(currentLayerPoles.length > 0
                ? currentLayerPoles
                : [0, 1, 2].map((r) => {
                    const base = (selectedLayer * nPolesPerLayer + r) * 6;
                    const nIdx = Math.round(polesData[base + 0] ?? 0);
                    const pol = (polesData[base + 1] ?? 1) >= 0 ? "+" : "-";
                    const act = polesData[base + 2] ?? 0;
                    return {
                      rank: r,
                      neuron_idx: nIdx,
                      polarity: pol,
                      abs_act: act,
                      j_vindex_tokens: [{ token: " spider", logit: 28.5 }],
                      naive_tokens: [{ token: " animal", logit: 24.1 }],
                    };
                  })
              ).map((p) => {
                const isNorth = p.polarity === "+";
                return (
                  <div
                    key={p.rank}
                    onClick={() => {
                      setSelectedPoleRank(p.rank);
                      inspectPole(selectedLayer, p.neuron_idx, isNorth ? 1.0 : -1.0);
                    }}
                    style={{
                      cursor: "pointer",
                      background:
                        p.rank === selectedPoleRank
                          ? "rgba(56, 189, 248, 0.16)"
                          : "rgba(3, 7, 18, 0.72)",
                      border: `1px solid ${
                        isNorth ? "rgba(56, 189, 248, 0.4)" : "rgba(244, 63, 94, 0.45)"
                      }`,
                      borderRadius: 6,
                      padding: "6px 8px",
                      fontSize: 10,
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <strong style={{ color: isNorth ? "#38bdf8" : "#f43f5e" }}>
                        Slot N{p.neuron_idx} [{isNorth ? "+North" : "-South"}] • |act|={p.abs_act.toFixed(2)}
                      </strong>
                      <button
                        className="hud-btn"
                        style={{ padding: "2px 6px", fontSize: 9 }}
                        onClick={(e) => {
                          e.stopPropagation();
                          setSteering({
                            enabled: true,
                            pole_layer: selectedLayer,
                            pole_neuron: p.neuron_idx,
                            pole_polarity: isNorth ? 1.0 : -1.0,
                          });
                        }}
                      >
                        Steer Pole
                      </button>
                    </div>
                    {p.j_vindex_tokens?.length > 0 && (
                      <div style={{ marginTop: 3, color: "#e2e8f0" }}>
                        <span style={{ color: "#38bdf8" }}>J-VINDEX: </span>
                        {p.j_vindex_tokens.map((t) => JSON.stringify(t.token)).join(", ")}
                      </div>
                    )}
                    {p.naive_tokens?.length > 0 && (
                      <div style={{ color: "#94a3b8" }}>
                        <span>Naive W_U: </span>
                        {p.naive_tokens.map((t) => JSON.stringify(t.token)).join(", ")}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Click-to-Steer Causal Laser Control Panel */}
          <div
            style={{
              background: "rgba(3, 7, 18, 0.78)",
              border: "1px solid rgba(244, 63, 94, 0.4)",
              borderRadius: 6,
              padding: 8,
              display: "flex",
              flexDirection: "column",
              gap: 6,
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: 11, fontWeight: 700, color: "#f43f5e" }}>
                CAUSAL STEERING LASER (SteeringSpec)
              </span>
              <label style={{ fontSize: 10, display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}>
                <input
                  type="checkbox"
                  checked={steering.enabled}
                  onChange={(e) =>
                    setSteering({
                      enabled: e.target.checked,
                      pole_layer: null,
                      pole_neuron: null,
                    })
                  }
                />
                ARM LASER
              </label>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
              <div>
                <div style={{ fontSize: 9, color: "#94a3b8" }}>Inject Concept (+):</div>
                <input
                  className="hud-input"
                  style={{ width: "100%" }}
                  value={steering.target_concept}
                  onChange={(e) =>
                    setSteering({ target_concept: e.target.value, pole_layer: null, pole_neuron: null })
                  }
                />
              </div>
              <div>
                <div style={{ fontSize: 9, color: "#94a3b8" }}>Suppress Concept (-):</div>
                <input
                  className="hud-input"
                  style={{ width: "100%" }}
                  value={steering.contrast_concept}
                  onChange={(e) =>
                    setSteering({ contrast_concept: e.target.value, pole_layer: null, pole_neuron: null })
                  }
                />
              </div>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 10 }}>
              <span style={{ color: "#94a3b8" }}>Scale: {steering.scale.toFixed(0)}</span>
              <input
                type="range"
                min={50}
                max={800}
                step={10}
                value={steering.scale}
                onChange={(e) => setSteering({ scale: Number(e.target.value) })}
                style={{ flex: 1 }}
              />
              <button
                className="hud-btn hud-btn-danger"
                onClick={() => {
                  setSteering({ enabled: true });
                  setTimeout(() => triggerSweep(), 20);
                }}
              >
                FIRE LASER
              </button>
            </div>
          </div>

          {/* LARQL / VINDEX3 FFN Vault & .patch Weight Surgery Console */}
          <div
            style={{
              background: "rgba(3, 7, 18, 0.78)",
              border: "1px solid rgba(56, 189, 248, 0.3)",
              borderRadius: 6,
              padding: 8,
              display: "flex",
              flexDirection: "column",
              gap: 6,
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: 11, fontWeight: 700, color: "#00e5ff" }}>
                LARQL / VINDEX3 VAULT & .PATCH SURGERY
              </span>
              <span style={{ fontSize: 10, color: "#fbbf24" }}>
                Active Patches: {activePatchesCount}
              </span>
            </div>
            <textarea
              className="hud-input"
              rows={2}
              value={larqlQuery}
              onChange={(e) => setLarqlQuery(e.target.value)}
              style={{ width: "100%", resize: "vertical", fontFamily: "monospace", fontSize: 10 }}
            />
            <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
              <button className="hud-btn" onClick={() => executeLarql()}>
                EXEC LQL
              </button>
              <button
                className="hud-btn"
                onClick={() => {
                  const q = `SELECT * FROM EDGES NEAREST 'spider' WHERE layer = ${selectedLayer} LIMIT 4`;
                  setLarqlQuery(q);
                  executeLarql(q);
                }}
              >
                Nearest 'spider'
              </button>
              <button
                className="hud-btn hud-btn-gold"
                onClick={() => {
                  const q = `UPDATE LAYER ${selectedLayer} SLOT 765 SET object = 'six' STRENGTH 3.0 POLARITY +`;
                  setLarqlQuery(q);
                  executeLarql(q);
                }}
              >
                .patch Slot
              </button>
              <button
                className="hud-btn hud-btn-danger"
                onClick={() => executeLarql("ROLLBACK")}
              >
                ROLLBACK
              </button>
            </div>
            {(larqlOutput || poleInspection || filamentsMeta.length > 0) && (
              <pre
                className="hud-scroller"
                style={{
                  margin: 0,
                  maxHeight: 150,
                  overflowY: "auto",
                  background: "rgba(0,0,0,0.65)",
                  padding: 6,
                  borderRadius: 4,
                  fontSize: 9.5,
                  color: "#a5f3fc",
                  whiteSpace: "pre-wrap",
                }}
              >
                {JSON.stringify(larqlOutput ?? poleInspection ?? filamentsMeta.slice(0, 4), null, 2)}
              </pre>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

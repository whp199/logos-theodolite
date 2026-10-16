import React, { useMemo } from "react";
import { Html, Line } from "@react-three/drei";
import { useTheodoliteStore } from "../store/useTheodoliteStore";
import { layerToWorldY, RING_RADIUS } from "./LayerRings";

const PLUME_LAYERS = new Set([23, 29, 35, 41, 44, 50, 53, 59]);

export function mapSurveyToWorld(
  x: number,
  y: number,
  layerIdx: number,
  scaleFactor: number
): [number, number, number] {
  const wx = Math.max(-RING_RADIUS * 0.92, Math.min(RING_RADIUS * 0.92, x * scaleFactor));
  const wz = Math.max(-RING_RADIUS * 0.92, Math.min(RING_RADIUS * 0.92, y * scaleFactor));
  const wy = layerToWorldY(layerIdx);
  return [wx, wy, wz];
}

export const BallisticTrajectory: React.FC = () => {
  const nLayers = useTheodoliteStore((s) => s.nLayers);
  const trajXyz = useTheodoliteStore((s) => s.trajXyz);
  const thinkTrajXyz = useTheodoliteStore((s) => s.thinkTrajXyz);
  const deltaAttnXyz = useTheodoliteStore((s) => s.deltaAttnXyz);
  const deltaFfnXyz = useTheodoliteStore((s) => s.deltaFfnXyz);
  const layerPlumes = useTheodoliteStore((s) => s.layerPlumes);
  const selectedLayer = useTheodoliteStore((s) => s.selectedLayer);
  const setSelectedLayer = useTheodoliteStore((s) => s.setSelectedLayer);
  const showAttnSplit = useTheodoliteStore((s) => s.showAttnSplit);
  const showFfnSplit = useTheodoliteStore((s) => s.showFfnSplit);
  const showPlumes = useTheodoliteStore((s) => s.showPlumes);

  const { mainPoints, thinkPoints, attnVectors, ffnVectors } = useMemo(() => {
    let maxCoord = 1.0;
    for (let l = 0; l < nLayers; l++) {
      const ax = Math.abs(trajXyz[l * 3 + 0]);
      const ay = Math.abs(trajXyz[l * 3 + 1]);
      if (ax > maxCoord) maxCoord = ax;
      if (ay > maxCoord) maxCoord = ay;
    }
    const scale = (RING_RADIUS * 0.72) / maxCoord;

    const pts: [number, number, number][] = [];
    const tPts: [number, number, number][] = [];
    const attnVecs: { start: [number, number, number]; end: [number, number, number]; l: number }[] = [];
    const ffnVecs: { start: [number, number, number]; end: [number, number, number]; l: number }[] = [];

    for (let l = 0; l < nLayers; l++) {
      const x = trajXyz[l * 3 + 0];
      const y = trajXyz[l * 3 + 1];
      const p = mapSurveyToWorld(x, y, l, scale);
      pts.push(p);

      if (thinkTrajXyz && thinkTrajXyz.length >= nLayers * 3) {
        const tx = thinkTrajXyz[l * 3 + 0];
        const ty = thinkTrajXyz[l * 3 + 1];
        tPts.push(mapSurveyToWorld(tx, ty, l, scale));
      }

      const dax = deltaAttnXyz[l * 4 + 0] * scale * 0.65;
      const day = deltaAttnXyz[l * 4 + 1] * scale * 0.65;
      attnVecs.push({
        l,
        start: p,
        end: [p[0] + dax, p[1] + 0.14, p[2] + day],
      });

      const dfx = deltaFfnXyz[l * 4 + 0] * scale * 0.65;
      const dfy = deltaFfnXyz[l * 4 + 1] * scale * 0.65;
      ffnVecs.push({
        l,
        start: p,
        end: [p[0] + dfx, p[1] + 0.22, p[2] + dfy],
      });
    }

    return {
      mainPoints: pts,
      thinkPoints: tPts,
      attnVectors: attnVecs,
      ffnVectors: ffnVecs,
    };
  }, [nLayers, trajXyz, thinkTrajXyz, deltaAttnXyz, deltaFfnXyz]);

  return (
    <group>
      {/* Primary Ballistic Residual Stream Spine */}
      {mainPoints.length > 1 && (
        <Line points={mainPoints} color="#e2e8f0" lineWidth={3.0} transparent opacity={0.92} />
      )}

      {/* Optional <|think|> Enabled Comparison Trajectory */}
      {thinkPoints.length > 1 && (
        <Line points={thinkPoints} color="#c084fc" lineWidth={2.5} dashed dashSize={0.4} gapSize={0.2} />
      )}

      {/* Attention Router Vectors (delta_attn_J) in Cyan */}
      {showAttnSplit &&
        attnVectors.map((v) => (
          <Line
            key={`attn-${v.l}`}
            points={[v.start, v.end]}
            color="#00e5ff"
            lineWidth={v.l === selectedLayer ? 3.2 : 1.7}
            transparent
            opacity={v.l === selectedLayer ? 1.0 : 0.78}
          />
        ))}

      {/* FFN Vault Vectors (delta_ffn_J) in Gold */}
      {showFfnSplit &&
        ffnVectors.map((v) => (
          <Line
            key={`ffn-${v.l}`}
            points={[v.start, v.end]}
            color="#fbbf24"
            lineWidth={v.l === selectedLayer ? 3.2 : 1.7}
            transparent
            opacity={v.l === selectedLayer ? 1.0 : 0.78}
          />
        ))}

      {/* Layer Nodes & Ghost-Token Plumes */}
      {mainPoints.map((pt, l) => {
        const isSelected = l === selectedLayer;
        const showPlumeHere = showPlumes && (PLUME_LAYERS.has(l) || isSelected);
        const plume = layerPlumes[l];
        const ffnDom = plume?.ffn_dominance ?? 0.5;
        const nodeColor = isSelected ? "#ffffff" : ffnDom >= 0.52 ? "#fbbf24" : "#00e5ff";

        return (
          <group key={`node-${l}`} position={pt}>
            <mesh
              onClick={(e) => {
                e.stopPropagation();
                setSelectedLayer(l);
              }}
            >
              <sphereGeometry args={[isSelected ? 0.34 : 0.17, 16, 16]} />
              <meshStandardMaterial
                color={nodeColor}
                emissive={nodeColor}
                emissiveIntensity={isSelected ? 1.6 : 0.85}
              />
            </mesh>

            {showPlumeHere && plume && plume.jpp_top_tokens?.length > 0 && (
              <Html position={[0.6, 0.25, 0]} center>
                <div
                  onClick={() => setSelectedLayer(l)}
                  style={{
                    cursor: "pointer",
                    background: isSelected ? "rgba(0, 229, 255, 0.2)" : "rgba(5, 8, 17, 0.86)",
                    border: `1px solid ${isSelected ? "#00e5ff" : "rgba(148, 163, 184, 0.35)"}`,
                    borderRadius: "5px",
                    padding: "2px 6px",
                    fontSize: "10px",
                    whiteSpace: "nowrap",
                    color: "#f8fafc",
                    boxShadow: "0 2px 10px rgba(0,0,0,0.6)",
                  }}
                >
                  <span style={{ color: "#94a3b8", marginRight: 4 }}>L{l + 1}</span>
                  <strong style={{ color: "#38bdf8" }}>
                    {JSON.stringify(plume.jpp_top_tokens[0].token)}
                  </strong>
                  {plume.tracked_ranks?.spider && plume.tracked_ranks.spider <= 50 && (
                    <span style={{ color: "#fbbf24", marginLeft: 5 }}>
                      spider#{plume.tracked_ranks.spider}
                    </span>
                  )}
                  {plume.tracked_ranks?.eight && plume.tracked_ranks.eight <= 20 && (
                    <span style={{ color: "#4ade80", marginLeft: 5 }}>
                      8/eight#{plume.tracked_ranks.eight}
                    </span>
                  )}
                </div>
              </Html>
            )}
          </group>
        );
      })}
    </group>
  );
};

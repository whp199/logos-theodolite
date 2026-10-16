import React, { useMemo } from "react";
import * as THREE from "three";
import { Html, Line } from "@react-three/drei";
import { useTheodoliteStore } from "../store/useTheodoliteStore";

export const LAYER_VERTICAL_STEP = 0.65;
export const RING_RADIUS = 10.5;
const GLOBAL_SET = new Set([5, 11, 17, 23, 29, 35, 41, 47, 53, 59]);

export function layerToWorldY(layerIdx: number): number {
  return (layerIdx - 29.5) * LAYER_VERTICAL_STEP;
}

export const LayerRings: React.FC = () => {
  const selectedLayer = useTheodoliteStore((s) => s.selectedLayer);
  const setSelectedLayer = useTheodoliteStore((s) => s.setSelectedLayer);
  const surveyLabels = useTheodoliteStore((s) => s.surveyLabels);

  const ringCirclePoints = useMemo(() => {
    const pts: [number, number, number][] = [];
    const segments = 72;
    for (let i = 0; i <= segments; i++) {
      const a = (i / segments) * Math.PI * 2;
      pts.push([Math.cos(a) * RING_RADIUS, 0, Math.sin(a) * RING_RADIUS]);
    }
    return pts;
  }, []);

  const innerCirclePoints = useMemo(() => {
    const pts: [number, number, number][] = [];
    const segments = 64;
    const r = RING_RADIUS * 0.55;
    for (let i = 0; i <= segments; i++) {
      const a = (i / segments) * Math.PI * 2;
      pts.push([Math.cos(a) * r, 0, Math.sin(a) * r]);
    }
    return pts;
  }, []);

  const layers = useMemo(() => Array.from({ length: 60 }, (_, i) => i), []);

  return (
    <group>
      {/* Central vertical optical spine */}
      <Line
        points={[
          [0, layerToWorldY(0) - 1.5, 0],
          [0, layerToWorldY(59) + 1.5, 0],
        ]}
        color="#38bdf8"
        lineWidth={1.2}
        transparent
        opacity={0.35}
      />

      {/* Survey Frame Axis Beacons at base & crown */}
      <group position={[0, layerToWorldY(selectedLayer), 0]}>
        <Line
          points={[
            [-RING_RADIUS * 1.15, 0, 0],
            [RING_RADIUS * 1.15, 0, 0],
          ]}
          color="#00e5ff"
          lineWidth={1.4}
          transparent
          opacity={0.55}
        />
        <Line
          points={[
            [0, 0, -RING_RADIUS * 1.15],
            [0, 0, RING_RADIUS * 1.15],
          ]}
          color="#fbbf24"
          lineWidth={1.4}
          transparent
          opacity={0.55}
        />
        <Html position={[RING_RADIUS * 1.2, 0.2, 0]} center>
          <div
            style={{
              color: "#00e5ff",
              fontSize: "10px",
              fontWeight: 700,
              background: "rgba(5,8,17,0.85)",
              padding: "2px 6px",
              borderRadius: "4px",
              border: "1px solid rgba(0,229,255,0.4)",
              whiteSpace: "nowrap",
            }}
          >
            +e₁ [{surveyLabels[0]}]
          </div>
        </Html>
        <Html position={[0, 0.2, RING_RADIUS * 1.2]} center>
          <div
            style={{
              color: "#fbbf24",
              fontSize: "10px",
              fontWeight: 700,
              background: "rgba(5,8,17,0.85)",
              padding: "2px 6px",
              borderRadius: "4px",
              border: "1px solid rgba(251,191,36,0.4)",
              whiteSpace: "nowrap",
            }}
          >
            +e₂ [{surveyLabels[1]}]
          </div>
        </Html>
      </group>

      {layers.map((l) => {
        const y = layerToWorldY(l);
        const isGlobal = GLOBAL_SET.has(l);
        const isSelected = l === selectedLayer;
        const ringColor = isSelected ? "#00e5ff" : isGlobal ? "#fbbf24" : "#1e293b";
        const lineOpacity = isSelected ? 0.95 : isGlobal ? 0.72 : 0.22;

        return (
          <group key={l} position={[0, y, 0]}>
            {/* Outer Astrolabe Ring */}
            <Line
              points={ringCirclePoints}
              color={ringColor}
              lineWidth={isSelected ? 2.6 : isGlobal ? 1.8 : 0.8}
              transparent
              opacity={lineOpacity}
            />

            {/* Global Attention / Selected Glass Plate */}
            {(isGlobal || isSelected) && (
              <>
                <Line
                  points={innerCirclePoints}
                  color={isSelected ? "#00e5ff" : "#38bdf8"}
                  lineWidth={1.0}
                  transparent
                  opacity={isSelected ? 0.55 : 0.28}
                />
                <mesh
                  rotation={[-Math.PI / 2, 0, 0]}
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedLayer(l);
                  }}
                >
                  <ringGeometry args={[0.4, RING_RADIUS, 48]} />
                  <meshBasicMaterial
                    color={isSelected ? "#00e5ff" : "#fbbf24"}
                    transparent
                    opacity={isSelected ? 0.09 : 0.035}
                    side={THREE.DoubleSide}
                    depthWrite={false}
                  />
                </mesh>
                <Html position={[-RING_RADIUS - 1.4, 0, 0]} center>
                  <div
                    onClick={() => setSelectedLayer(l)}
                    style={{
                      cursor: "pointer",
                      fontSize: isSelected ? "11px" : "9px",
                      fontWeight: 700,
                      color: isSelected ? "#00e5ff" : "#fbbf24",
                      background: "rgba(5, 8, 17, 0.82)",
                      border: `1px solid ${isSelected ? "#00e5ff" : "rgba(251, 191, 36, 0.45)"}`,
                      borderRadius: "4px",
                      padding: "1px 5px",
                      whiteSpace: "nowrap",
                    }}
                  >
                    L{l + 1} {isGlobal ? "★ GLOBAL" : ""}
                  </div>
                </Html>
              </>
            )}
          </group>
        );
      })}
    </group>
  );
};

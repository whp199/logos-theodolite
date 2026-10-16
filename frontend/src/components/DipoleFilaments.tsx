import React, { useMemo } from "react";
import { Line } from "@react-three/drei";
import { useTheodoliteStore } from "../store/useTheodoliteStore";
import { layerToWorldY, RING_RADIUS } from "./LayerRings";

const GLOBAL_SET = new Set([5, 11, 17, 23, 29, 35, 41, 47, 53, 59]);

export function getPoleAnchorPosition(
  layerIdx: number,
  rankIdx: number,
  dirX: number,
  dirY: number
): [number, number, number] {
  const baseRadius = RING_RADIUS * (0.35 + (rankIdx % 5) * 0.11);
  const fallbackAngle = (rankIdx / 15) * Math.PI * 2 + layerIdx * 0.21;
  const norm = Math.hypot(dirX, dirY);
  const ux = norm > 1e-4 ? dirX / norm : Math.cos(fallbackAngle);
  const uz = norm > 1e-4 ? dirY / norm : Math.sin(fallbackAngle);
  return [ux * baseRadius, layerToWorldY(layerIdx), uz * baseRadius];
}

export const DipoleFilaments: React.FC = () => {
  const nLayers = useTheodoliteStore((s) => s.nLayers);
  const nPolesPerLayer = useTheodoliteStore((s) => s.nPolesPerLayer);
  const nFilaments = useTheodoliteStore((s) => s.nFilaments);
  const polesData = useTheodoliteStore((s) => s.polesData);
  const filamentsData = useTheodoliteStore((s) => s.filamentsData);
  const selectedLayer = useTheodoliteStore((s) => s.selectedLayer);
  const selectedPoleRank = useTheodoliteStore((s) => s.selectedPoleRank);
  const showDipoles = useTheodoliteStore((s) => s.showDipoles);
  const showFilaments = useTheodoliteStore((s) => s.showFilaments);
  const setSelectedLayer = useTheodoliteStore((s) => s.setSelectedLayer);
  const setSelectedPoleRank = useTheodoliteStore((s) => s.setSelectedPoleRank);
  const inspectPole = useTheodoliteStore((s) => s.inspectPole);
  const traceReaders = useTheodoliteStore((s) => s.traceReaders);

  const { needles, filamentCurves } = useMemo(() => {
    const needleList: {
      layer: number;
      rank: number;
      neuronIdx: number;
      polarity: number;
      absAct: number;
      anchor: [number, number, number];
      tip: [number, number, number];
    }[] = [];

    for (let l = 0; l < nLayers; l++) {
      const isFocus = l === selectedLayer || GLOBAL_SET.has(l);
      const maxRanks = isFocus ? Math.min(8, nPolesPerLayer) : 2;
      for (let r = 0; r < maxRanks; r++) {
        const base = (l * nPolesPerLayer + r) * 6;
        if (base + 5 >= polesData.length) continue;
        const neuronIdx = Math.round(polesData[base + 0]);
        const polarity = polesData[base + 1] >= 0 ? 1 : -1;
        const absAct = polesData[base + 2];
        const dx = polesData[base + 3];
        const dy = polesData[base + 4];
        const dz = polesData[base + 5];

        const anchor = getPoleAnchorPosition(l, r, dx, dy);
        const len = Math.min(1.6, 0.45 + Math.log1p(Math.abs(absAct)) * 0.28);
        const tip: [number, number, number] = [
          anchor[0] + dx * len,
          anchor[1] + polarity * Math.max(0.28, Math.abs(dz) * len),
          anchor[2] + dy * len,
        ];

        needleList.push({
          layer: l,
          rank: r,
          neuronIdx,
          polarity,
          absAct,
          anchor,
          tip,
        });
      }
    }

    const curves: {
      key: string;
      points: [number, number, number][];
      coupling: number;
      isHighlighted: boolean;
    }[] = [];

    const maxFil = Math.min(nFilaments, Math.floor(filamentsData.length / 7), 120);
    for (let f = 0; f < maxFil; f++) {
      const base = f * 7;
      const srcL = Math.round(filamentsData[base + 0]);
      const srcR = Math.round(filamentsData[base + 1]);
      const dstL = Math.round(filamentsData[base + 2]);
      const dstR = Math.round(filamentsData[base + 3]);
      const coupling = filamentsData[base + 4];

      if (srcL < 0 || srcL >= nLayers || dstL < 0 || dstL >= nLayers) continue;

      const sBase = (srcL * nPolesPerLayer + srcR) * 6;
      const dBase = (dstL * nPolesPerLayer + dstR) * 6;
      if (dBase + 5 >= polesData.length) continue;

      const start = getPoleAnchorPosition(srcL, srcR, polesData[sBase + 3], polesData[sBase + 4]);
      const end = getPoleAnchorPosition(dstL, dstR, polesData[dBase + 3], polesData[dBase + 4]);
      const mid: [number, number, number] = [
        (start[0] + end[0]) * 0.42,
        (start[1] + end[1]) * 0.5,
        (start[2] + end[2]) * 0.42,
      ];

      curves.push({
        key: `fil-${f}-${srcL}-${dstL}`,
        points: [start, mid, end],
        coupling,
        isHighlighted: srcL === selectedLayer || dstL === selectedLayer,
      });
    }

    return { needles: needleList, filamentCurves: curves };
  }, [nLayers, nPolesPerLayer, nFilaments, polesData, filamentsData, selectedLayer]);

  return (
    <group>
      {/* Oriented Bipolar GeGLU Needles (+ North / - South) */}
      {showDipoles &&
        needles.map((n) => {
          const isSelected = n.layer === selectedLayer && n.rank === selectedPoleRank;
          const color = isSelected ? "#ffffff" : n.polarity > 0 ? "#38bdf8" : "#f43f5e";
          return (
            <group key={`needle-${n.layer}-${n.rank}`}>
              <Line
                points={[n.anchor, n.tip]}
                color={color}
                lineWidth={isSelected ? 3.2 : n.layer === selectedLayer ? 2.2 : 1.2}
                transparent
                opacity={n.layer === selectedLayer ? 0.95 : 0.65}
              />
              <mesh
                position={n.tip}
                onClick={(e) => {
                  e.stopPropagation();
                  setSelectedLayer(n.layer);
                  setSelectedPoleRank(n.rank);
                  inspectPole(n.layer, n.neuronIdx, n.polarity);
                  traceReaders(n.layer, n.neuronIdx, n.polarity);
                }}
              >
                <coneGeometry args={[isSelected ? 0.22 : 0.12, isSelected ? 0.45 : 0.26, 8]} />
                <meshStandardMaterial
                  color={color}
                  emissive={color}
                  emissiveIntensity={isSelected ? 1.5 : 0.7}
                />
              </mesh>
            </group>
          );
        })}

      {/* Live Downstream Reader Circuit Filaments */}
      {showFilaments &&
        filamentCurves.map((fc) => (
          <Line
            key={fc.key}
            points={fc.points}
            color={fc.coupling >= 0 ? "#38bdf8" : "#f43f5e"}
            lineWidth={fc.isHighlighted ? 2.2 : 0.9}
            transparent
            opacity={fc.isHighlighted ? 0.85 : 0.26}
          />
        ))}
    </group>
  );
};

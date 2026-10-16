import React, { useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { LayerRings } from "./LayerRings";
import { BallisticTrajectory } from "./BallisticTrajectory";
import { DipoleFilaments } from "./DipoleFilaments";
import { useTheodoliteStore } from "../store/useTheodoliteStore";

const FpsTracker: React.FC = () => {
  const setFps = useTheodoliteStore((s) => s.setFps);
  const frames = useRef(0);
  const lastTime = useRef(performance.now());

  useFrame(() => {
    frames.current += 1;
    const now = performance.now();
    const delta = now - lastTime.current;
    if (delta >= 500) {
      const currentFps = Math.round((frames.current * 1000) / delta);
      setFps(currentFps);
      frames.current = 0;
      lastTime.current = now;
    }
  });

  return null;
};

export const AstrolabeScene: React.FC = () => {
  return (
    <div style={{ width: "100vw", height: "100vh", position: "absolute", inset: 0 }}>
      <Canvas
        camera={{ position: [24, 14, 24], fov: 45, near: 0.1, far: 400 }}
        gl={{ antialias: true, powerPreference: "high-performance" }}
      >
        <color attach="background" args={["#050811"]} />
        <ambientLight intensity={0.85} />
        <pointLight position={[25, 35, 25]} intensity={1.5} color="#00e5ff" />
        <pointLight position={[-25, -20, -25]} intensity={1.1} color="#fbbf24" />

        <FpsTracker />
        <LayerRings />
        <BallisticTrajectory />
        <DipoleFilaments />

        <OrbitControls
          enableDamping
          dampingFactor={0.08}
          minDistance={8}
          maxDistance={95}
          target={[0, 2, 0]}
        />
      </Canvas>
    </div>
  );
};

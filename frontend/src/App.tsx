import React, { useEffect } from "react";
import { AstrolabeScene } from "./components/AstrolabeScene";
import { ReticleHUD } from "./components/ReticleHUD";
import { useTheodoliteStore } from "./store/useTheodoliteStore";

export const App: React.FC = () => {
  const connectWebSocket = useTheodoliteStore((s) => s.connectWebSocket);

  useEffect(() => {
    connectWebSocket();
  }, [connectWebSocket]);

  return (
    <main style={{ width: "100vw", height: "100vh", position: "relative", overflow: "hidden" }}>
      <AstrolabeScene />
      <ReticleHUD />
    </main>
  );
};

export default App;

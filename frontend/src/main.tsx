import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { useTheodoliteStore } from "./store/useTheodoliteStore";
import "./index.css";

if (import.meta.env.DEV) {
  // for poking at the instrument from the browser console while developing
  (window as unknown as { theodolite: typeof useTheodoliteStore }).theodolite = useTheodoliteStore;
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);

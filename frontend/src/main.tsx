import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { useTheodoliteStore } from "./store/useTheodoliteStore";
// the typefaces are served with the instrument (SIL Open Font License), so the pages work offline
import "@fontsource/eb-garamond/400.css";
import "@fontsource/eb-garamond/400-italic.css";
import "@fontsource/eb-garamond/500.css";
import "@fontsource/eb-garamond/500-italic.css";
import "@fontsource/eb-garamond/600.css";
import "@fontsource/eb-garamond/600-italic.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
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

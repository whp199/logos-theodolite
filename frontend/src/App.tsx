import React, { Suspense, lazy, useEffect } from "react";
import { type Route, useRoute } from "./router";
import { useTheodoliteStore } from "./store/useTheodoliteStore";
import { Page } from "./components/kit";

// each page is its own chunk: the lexicon and the lineage are long, and most visits open one page
const Incipit = lazy(() => import("./screens/Incipit").then((m) => ({ default: m.Incipit })));
const Instrument = lazy(() => import("./screens/Instrument").then((m) => ({ default: m.Instrument })));
const LayerLeaf = lazy(() => import("./screens/LayerLeaf").then((m) => ({ default: m.LayerLeaf })));
const NeuronFolio = lazy(() => import("./screens/NeuronFolio").then((m) => ({ default: m.NeuronFolio })));
const Atlas = lazy(() => import("./screens/Atlas").then((m) => ({ default: m.Atlas })));
const Landmarks = lazy(() => import("./screens/Landmarks").then((m) => ({ default: m.Landmarks })));
const Nudge = lazy(() => import("./screens/Nudge").then((m) => ({ default: m.Nudge })));
const Vault = lazy(() => import("./screens/Vault").then((m) => ({ default: m.Vault })));
const Lexicon = lazy(() => import("./screens/Lexicon").then((m) => ({ default: m.Lexicon })));
const Lineage = lazy(() => import("./screens/Lineage").then((m) => ({ default: m.Lineage })));

const TITLES: Record<Route["name"], string> = {
  incipit: "Logos Theodolite",
  instrument: "The Instrument — Logos Theodolite",
  layer: "A layer’s leaf — Logos Theodolite",
  neuron: "A neuron’s folio — Logos Theodolite",
  atlas: "The atlas — Logos Theodolite",
  landmarks: "Landmarks — Logos Theodolite",
  nudge: "Nudge — Logos Theodolite",
  vault: "The Vault — Logos Theodolite",
  lexicon: "Lexicon — Logos Theodolite",
  lineage: "Lineage — Logos Theodolite",
};

const Opening: React.FC<{ route: Route }> = ({ route }) => (
  <Page route={route} folio="">
    <p className="status-line" style={{ padding: "40px 0" }}>
      Opening the page…
    </p>
  </Page>
);

function screenFor(route: Route) {
  switch (route.name) {
    case "incipit":
      return <Incipit route={route} />;
    case "instrument":
      return <Instrument route={route} />;
    case "layer":
      return <LayerLeaf route={route} />;
    case "neuron":
      return <NeuronFolio route={route} />;
    case "atlas":
      return <Atlas route={route} />;
    case "landmarks":
      return <Landmarks route={route} />;
    case "nudge":
      return <Nudge route={route} />;
    case "vault":
      return <Vault route={route} />;
    case "lexicon":
      return <Lexicon route={route} />;
    case "lineage":
      return <Lineage route={route} />;
  }
}

export const App: React.FC = () => {
  const route = useRoute();
  const connect = useTheodoliteStore((s) => s.connect);
  const loadFixture = useTheodoliteStore((s) => s.loadFixture);

  useEffect(() => {
    // the recorded sighting fills the pages at once; a live one replaces it when the server answers
    void loadFixture();
    connect();
  }, [connect, loadFixture]);

  const term = route.name === "lexicon" ? route.term : route.name === "lineage" ? route.anchor : undefined;
  const pageKey =
    route.name === "layer" ? `layer:${route.layerNo}` : route.name === "neuron" ? `neuron:${route.layerNo}:${route.neuron}` : route.name;
  useEffect(() => {
    document.title = TITLES[route.name];
    if (!term) {
      window.scrollTo(0, 0);
      return;
    }
    // an entry or reference: start at the top, wait for its page's chunk to arrive, then bring it into view
    window.scrollTo(0, 0);
    let frames = 0;
    let raf = 0;
    const seek = () => {
      const el = document.getElementById(term);
      if (el) el.scrollIntoView({ block: "start" });
      else if (frames++ < 120) raf = requestAnimationFrame(seek);
    };
    seek();
    return () => cancelAnimationFrame(raf);
  }, [route.name, pageKey, term]);

  return <Suspense fallback={<Opening route={route} />}>{screenFor(route)}</Suspense>;
};

export default App;

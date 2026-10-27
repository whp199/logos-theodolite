import { useEffect, useState } from "react";

/**
 * A small hash router: the instrument is served as static files by the backend,
 * so routes live after the "#".
 *
 *   #/                         incipit (welcome and first sighting)
 *   #/instrument               the instrument
 *   #/layer/36                 a layer's leaf (layers counted 1-60)
 *   #/neuron/54/15491/pos      a neuron's folio (layer 1-60, neuron index, face pos|neg)
 *   #/atlas                    the atlas (every neuron at once); #/atlas/54/15491 opens a neuron, #/atlas/54/h3 a head
 *   #/landmarks #/nudge #/vault #/lexicon #/lineage
 *   #/lexicon/lens             a lexicon entry, scrolled into view
 */
export type Route =
  | { name: "incipit" }
  | { name: "instrument" }
  | { name: "layer"; layerNo: number }
  | { name: "neuron"; layerNo: number; neuron: number; face: 1 | -1 }
  | { name: "atlas"; layerNo?: number; neuron?: number; head?: number }
  | { name: "landmarks" }
  | { name: "nudge" }
  | { name: "vault" }
  | { name: "lexicon"; term?: string }
  | { name: "lineage"; anchor?: string };

export function parseHash(hash: string): Route {
  const parts = hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  const [head, a, b, c] = parts;
  const int = (s: string | undefined, lo: number, hi: number, dflt: number) => {
    const n = s !== undefined && /^\d+$/.test(s) ? parseInt(s, 10) : NaN;
    return Number.isFinite(n) && n >= lo && n <= hi ? n : dflt;
  };
  switch (head) {
    case undefined:
    case "":
    case "incipit":
      return { name: "incipit" };
    case "instrument":
      return { name: "instrument" };
    case "layer":
      return { name: "layer", layerNo: int(a, 1, 60, 36) };
    case "neuron":
      return {
        name: "neuron",
        // the canvas's folio: neuron 15491 at layer 54, the "I am" neuron of the recorded sighting
        layerNo: int(a, 1, 60, 54),
        neuron: int(b, 0, 21503, 15491),
        face: c === "neg" || c === "negative" || c === "-" ? -1 : 1,
      };
    case "atlas": {
      // the atlas checks the numbers against the model the server has loaded (a stand-in has fewer layers)
      const layerNo = a !== undefined && /^\d+$/.test(a) ? parseInt(a, 10) : undefined;
      if (layerNo === undefined || layerNo < 1) return { name: "atlas" };
      const h = b?.match(/^h(\d+)$/i);
      if (h) return { name: "atlas", layerNo, head: parseInt(h[1], 10) };
      return b !== undefined && /^\d+$/.test(b) ? { name: "atlas", layerNo, neuron: parseInt(b, 10) } : { name: "atlas" };
    }
    case "landmarks":
    case "nudge":
    case "vault":
      return { name: head };
    case "lexicon":
      return { name: "lexicon", term: a };
    case "lineage":
      return { name: "lineage", anchor: a };
    default:
      return { name: "incipit" };
  }
}

export const href = {
  incipit: () => "#/",
  instrument: () => "#/instrument",
  layer: (layerNo: number) => `#/layer/${layerNo}`,
  neuron: (layerNo: number, neuron: number, face: 1 | -1 = 1) =>
    `#/neuron/${layerNo}/${neuron}/${face < 0 ? "neg" : "pos"}`,
  atlas: (layerNo?: number, neuron?: number) => (layerNo != null && neuron != null ? `#/atlas/${layerNo}/${neuron}` : "#/atlas"),
  landmarks: () => "#/landmarks",
  nudge: () => "#/nudge",
  vault: () => "#/vault",
  lexicon: (term?: string) => (term ? `#/lexicon/${term}` : "#/lexicon"),
  lineage: (anchor?: string) => (anchor ? `#/lineage/${anchor}` : "#/lineage"),
};

export function navigate(to: string) {
  if (window.location.hash !== to) window.location.hash = to;
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash));
  useEffect(() => {
    const onHash = () => {
      setRoute(parseHash(window.location.hash));
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  return route;
}

/**
 * The atlas's stored passages of a neuron's two faces, for the folio (GET /api/atlas/neuron/passages; the passages index
 * built by scripts/atlas_build.py passages). Read once a neuron and kept while the page is open, shared by the poles'
 * apparatus and the field notes. Null until read, and when the server does not answer (a page served without it);
 * `ready` false when the index is not built.
 */
import { useEffect, useState } from "react";
import type { Passages } from "../atlas/api";

export type FacePassages = Record<"+" | "-", Passages>;

const cache = new Map<string, Promise<FacePassages | null>>();

async function read(layer: number, n: number): Promise<FacePassages | null> {
  const one = async (face: "+" | "-") => {
    const q = new URLSearchParams({ layer: String(layer), n: String(n), face, k: "16" });
    const r = await fetch(`/api/atlas/neuron/passages?${q}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return (await r.json()) as Passages;
  };
  try {
    const [p, m] = await Promise.all([one("+"), one("-")]);
    return { "+": p, "-": m };
  } catch {
    return null;
  }
}

/** Both faces' passages of the neuron at layer `layer` (counted from 0) and index `n`; null until read or unreadable. */
export function usePassages(layer: number | null, n: number | null): FacePassages | null {
  const key = layer != null && n != null ? `${layer}/${n}` : "";
  const [got, setGot] = useState<{ key: string; v: FacePassages | null } | null>(null);
  useEffect(() => {
    if (!key || layer == null || n == null) return;
    let live = true;
    let p = cache.get(key);
    if (!p) {
      p = read(layer, n);
      cache.set(key, p);
      void p.then((v) => {
        if (v === null) cache.delete(key); // try again on the next visit
      });
    }
    void p.then((v) => {
      if (live) setGot({ key, v });
    });
    return () => {
      live = false;
    };
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps
  return got && got.key === key ? got.v : null;
}

/** A passage's text as the folio sets it: the units joined, the peak unit returned apart so it can be set in gilt. */
export function splitAtPeak(units: [string, boolean][]): [string, string, string] {
  const i = units.findIndex(([, peak]) => peak);
  if (i < 0) return [units.map(([t]) => t).join(""), "", ""];
  return [units.slice(0, i).map(([t]) => t).join(""), units[i][0], units.slice(i + 1).map(([t]) => t).join("")];
}

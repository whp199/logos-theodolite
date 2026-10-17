import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Per-viewer folio state: the name you give a neuron and your field notes, one record per face.
 * Kept in this browser's localStorage under `theodolite.folio.v1.${layerNo}:${neuron}`; every read and
 * write is wrapped so the page works when storage is blocked (the notes then last for this visit).
 */

export type Jobs = "one" | "two" | "unknown";

export interface FaceNotes {
  jobs: Jobs | null;
  pattern: string;
  other: boolean;
  long: boolean;
  steered: boolean;
  readers: boolean;
}

export interface FolioSaved {
  nick: string | null;
  notes: { pos: FaceNotes; neg: FaceNotes };
}

export const blankNotes = (): FaceNotes => ({
  jobs: null,
  pattern: "",
  other: false,
  long: false,
  steered: false,
  readers: false,
});

const blank = (): FolioSaved => ({ nick: null, notes: { pos: blankNotes(), neg: blankNotes() } });

export const folioKey = (layerNo: number, neuron: number) => `theodolite.folio.v1.${layerNo}:${neuron}`;

function cleanNotes(v: unknown): FaceNotes {
  const o = (v && typeof v === "object" ? v : {}) as Partial<FaceNotes>;
  const jobs = o.jobs === "one" || o.jobs === "two" || o.jobs === "unknown" ? o.jobs : null;
  return {
    jobs,
    pattern: typeof o.pattern === "string" ? o.pattern : "",
    other: o.other === true,
    long: o.long === true,
    steered: o.steered === true,
    readers: o.readers === true,
  };
}

function load(key: string): FolioSaved {
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) return blank();
    const v = JSON.parse(raw) as Partial<FolioSaved> & { notes?: { pos?: unknown; neg?: unknown } };
    return {
      nick: typeof v.nick === "string" && v.nick.trim() ? v.nick : null,
      notes: { pos: cleanNotes(v.notes?.pos), neg: cleanNotes(v.notes?.neg) },
    };
  } catch {
    return blank();
  }
}

function save(key: string, v: FolioSaved) {
  try {
    window.localStorage.setItem(key, JSON.stringify(v));
  } catch {
    // storage blocked or full: the notes simply last for this visit
  }
}

export function useFolioSaved(layerNo: number, neuron: number) {
  const key = folioKey(layerNo, neuron);
  const [saved, setSaved] = useState<FolioSaved>(() => load(key));
  const keyRef = useRef(key);
  useEffect(() => {
    if (keyRef.current !== key) {
      keyRef.current = key;
      setSaved(load(key));
    }
  }, [key]);
  const update = useCallback(
    (fn: (prev: FolioSaved) => FolioSaved) => {
      setSaved((prev) => {
        const next = fn(prev);
        save(key, next);
        return next;
      });
    },
    [key],
  );
  return [saved, update] as const;
}

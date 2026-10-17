import { create } from "zustand";

/**
 * How to read these pages: the reader/scholar pattern.
 * "rubrics" (on by default) shows the red italic guidance lines for newcomers.
 * "apparatus" (off by default) reveals the blocks ruled off by a double gilt line:
 * formulas, raw values, REFERENCES.md numbers and every expert control.
 * Both choices are remembered per viewer in this browser.
 */
const KEY = "theodolite.reading.v1";

interface Saved {
  rubrics: boolean;
  apparatus: boolean;
}

function load(): Saved {
  try {
    const raw = window.localStorage.getItem(KEY);
    if (raw) {
      const v = JSON.parse(raw) as Partial<Saved>;
      return { rubrics: v.rubrics !== false, apparatus: v.apparatus === true };
    }
  } catch {
    // storage blocked or corrupt: fall back to the defaults
  }
  return { rubrics: true, apparatus: false };
}

function save(v: Saved) {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(v));
  } catch {
    // the choice simply is not remembered
  }
}

interface ReadingState extends Saved {
  toggleRubrics: () => void;
  toggleApparatus: () => void;
}

export const useReading = create<ReadingState>((set, get) => ({
  ...load(),
  toggleRubrics: () => {
    set({ rubrics: !get().rubrics });
    save({ rubrics: get().rubrics, apparatus: get().apparatus });
  },
  toggleApparatus: () => {
    set({ apparatus: !get().apparatus });
    save({ rubrics: get().rubrics, apparatus: get().apparatus });
  },
}));

export const useRubrics = () => useReading((s) => s.rubrics);
export const useApparatus = () => useReading((s) => s.apparatus);

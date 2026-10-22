import { type Gate, overlapLine } from "../../lib/labels";

/**
 * The approved canvas's hand-written notes (Main.dc.html) on the loudest neuron at three layers of the
 * study's demonstration sighting (E6_tool_audit_02_sweep_demo.json, prepend_bos=True). They apply only
 * when that recorded sighting is shown (its words come from the recorded labels); a live sighting builds
 * its note from the server's labels. Keyed by "repo index:neuron".
 *
 * The label gate (lib/labels.ts) comes first: a note never sets a held-back label's words, or a gloss of
 * them, in the main reading. Layer 36 (35:21189) lies below layer 41, so its gloss and the canvas's trust
 * line stand only in the leaf's apparatus, under "held back"; the page's trust line is always trustLine.
 * The canvas's "agree on AM" (55:16649) became a plain count: what the words shown share, and, with the
 * ten-a-side count (the recorded checks, N1G_recorded_label_checks.json), how many of ten are shared and how
 * often labels sharing that many were sound in the study's sample (lib/labels.ts overlapLine), never "a good
 * sign" by itself: one of ten was a weaker sign than the layer alone.
 */
export interface RecordedPoleNote {
  /** the canvas's gloss of the words, by what the gate found; set beside the words when they are shown, and
   * kept in the apparatus with them when they are held back */
  gloss: (g: Gate) => string;
  /** the canvas's own line on how far to trust the label, kept in the apparatus beside a held-back label */
  canvasTrust?: string;
}

export const RECORDED_POLE_NOTES: Record<string, RecordedPoleNote> = {
  "35:21189": {
    gloss: () => "English “myself”, Hindi “me”, Chinese “of me”: through the lens, one idea in three tongues.",
    canvasTrust:
      "A rumour, not a finding: the study did not test labels at layers 32 to 40, where this one sits; they held up at 41 to 58 and were no better than chance at 11 to 31.",
  },
  "55:16649": {
    gloss: (g) =>
      g.sharedTop10 == null
        ? "Among the words shown, the lens and the naked eye both read “AM”."
        : g.sharedTop10 >= 1
          ? `The lens and the naked eye both read “AM”, and share ${g.sharedTop10} of their first ten word-pieces` +
            (overlapLine(g) ? `: ${overlapLine(g)}.` : ".")
          : "Among the words shown, the lens and the naked eye both read “AM”, yet counted ten a side, by word-piece, they share none.",
  },
  "59:12584": {
    gloss: () => "A neuron for the Spanish and German “es”.",
  },
};

/** The canvas's stations: the layers (repo indices) where the recorded sighting kept the lens's top five. */
export const CANVAS_STATIONS = [10, 20, 30, 35, 40, 45, 50, 55, 59];

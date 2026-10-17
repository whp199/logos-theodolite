/*
 * The Lexicon's index and its anchors. Every entry has a stable id (the canvas's own), so any page can
 * link `href.lexicon('<id>')`; the app scrolls to the element with that id on #/lexicon/<id>.
 */

export interface IndexItem {
  /** as printed in the index */
  label: string;
  /** a cross-reference: "pole, see neuron" */
  see?: string;
  /** the anchor it leads to */
  id: string;
  /** the section numeral it stands in */
  section: "ii" | "iii" | "iv" | "v";
}

/** "twenty-four words and one cross-reference, in alphabetical order" (the canvas's index). */
export const INDEX: IndexItem[] = [
  { label: "apparatus", id: "apparatus", section: "iii" },
  { label: "bearing and elevation", id: "bearing", section: "iii" },
  { label: "double-negative regime", id: "double-negative", section: "ii" },
  { label: "edit, an", id: "edit", section: "iv" },
  { label: "gathering", id: "gathering", section: "ii" },
  { label: "global and local attention", id: "global", section: "ii" },
  { label: "J-space", id: "j-space", section: "iii" },
  { label: "kinship of layers", id: "kinship", section: "iii" },
  { label: "landmarks", id: "landmarks", section: "iii" },
  { label: "layer", id: "layer", section: "ii" },
  { label: "lens and the naked eye, the", id: "lens", section: "iii" },
  { label: "matched control", id: "matched-control", section: "iv" },
  { label: "neuron and its two faces", id: "neuron", section: "ii" },
  { label: "nudge", id: "nudge", section: "iv" },
  { label: "pole", see: "neuron", id: "neuron", section: "ii" },
  { label: "quiet carry channel", id: "quiet", section: "v" },
  { label: "remembering", id: "remembering", section: "ii" },
  { label: "rubric", id: "rubric", section: "iii" },
  { label: "sighting", id: "sighting", section: "iv" },
  { label: "stream strength", id: "stream", section: "ii" },
  { label: "strength", id: "strength", section: "ii" },
  { label: "threads", id: "threads", section: "ii" },
  { label: "vault, the", id: "vault", section: "iv" },
  { label: "what it was thinking", id: "thinking", section: "iii" },
  { label: "word-piece", id: "word-piece", section: "ii" },
];

/**
 * Further anchors that land on an entry, for links written with another slug.
 * (The canvas's ids are the primary ones; these only catch the obvious alternatives.)
 */
export const ALIASES: Record<string, string[]> = {
  lens: ["naked-eye"],
  neuron: ["faces", "pole"],
  stream: ["stream-strength"],
  bearing: ["elevation"],
};

/**
 * Every anchor on the page, for reference:
 * entries   layer word-piece stream gathering remembering global neuron double-negative strength threads
 *           lens j-space thinking landmarks bearing kinship rubric apparatus
 *           sighting matched-control vault nudge edit
 * cautions  labels single late quiet frame thread-score nudge-defaults template think split bos
 * aliases   naked-eye faces pole stream-strength elevation
 */

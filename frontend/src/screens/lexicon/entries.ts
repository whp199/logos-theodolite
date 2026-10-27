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

/**
 * "twenty-eight words and three cross-references, in alphabetical order": the canvas's index, with the readout of words
 * renamed "through the lens", as the instrument and the layer's leaf now head it. Its old name, "what it was thinking",
 * stays as a cross-reference, and its anchor stays `thinking`, so older links still land on it. Since the canvas: how it
 * writes (with "sampler" as a cross-reference), the seed, a sighting of each word-piece, and the word it chose and its
 * likeliest; and write size, now that the instrument lists each layer's neurons by it. Since the folio puts a label to
 * the test: a test of a label, with the specialist's "causal spot-check" as a cross-reference (named so, and not "a
 * test", since the folio's section v puts a guess to the test too, a different thing).
 */
export const INDEX: IndexItem[] = [
  { label: "apparatus", id: "apparatus", section: "iii" },
  { label: "atlas, the", id: "atlas", section: "iii" },
  { label: "bearing and elevation", id: "bearing", section: "iii" },
  { label: "causal spot-check", see: "a test of a label", id: "test", section: "iv" },
  { label: "census, the", id: "census", section: "iii" },
  { label: "double-negative regime", id: "double-negative", section: "ii" },
  { label: "edit, an", id: "edit", section: "iv" },
  { label: "gathering", id: "gathering", section: "ii" },
  { label: "global and local attention", id: "global", section: "ii" },
  { label: "how it writes", id: "how-it-writes", section: "iii" },
  { label: "J-space", id: "j-space", section: "iii" },
  { label: "kinship of layers", id: "kinship", section: "iii" },
  { label: "landmarks", id: "landmarks", section: "iii" },
  { label: "layer", id: "layer", section: "ii" },
  { label: "lens and the naked eye, the", id: "lens", section: "iii" },
  { label: "matched control", id: "matched-control", section: "iv" },
  { label: "names and notes", id: "names", section: "iii" },
  { label: "neuron and its two faces", id: "neuron", section: "ii" },
  { label: "nudge", id: "nudge", section: "iv" },
  { label: "pole", see: "neuron", id: "neuron", section: "ii" },
  { label: "push, a", id: "push", section: "iii" },
  { label: "quiet carry channel", id: "quiet", section: "v" },
  { label: "remembering", id: "remembering", section: "ii" },
  { label: "rubric", id: "rubric", section: "iii" },
  { label: "sampler", see: "how it writes", id: "how-it-writes", section: "iii" },
  { label: "scientist, the", id: "scientist", section: "iii" },
  { label: "seed, the", id: "seed", section: "iii" },
  { label: "sighting", id: "sighting", section: "iv" },
  { label: "sighting of each word-piece, a", id: "each-word-piece", section: "iii" },
  { label: "steer lens, the", id: "steer-lens", section: "iii" },
  { label: "stream strength", id: "stream", section: "ii" },
  { label: "strength", id: "strength", section: "ii" },
  { label: "test of a label, a", id: "test", section: "iv" },
  { label: "threads", id: "threads", section: "ii" },
  { label: "through the lens", id: "thinking", section: "iii" },
  { label: "vault, the", id: "vault", section: "iv" },
  { label: "web, the", id: "web", section: "iii" },
  { label: "what it was thinking", see: "through the lens", id: "thinking", section: "iii" },
  { label: "word it chose and its likeliest, the", id: "chosen", section: "iii" },
  { label: "word-piece", id: "word-piece", section: "ii" },
  { label: "write size", id: "write", section: "ii" },
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
  thinking: ["through-the-lens"],
  test: ["causal-check", "spot-check"],
};

/**
 * Every anchor on the page, for reference:
 * entries   layer word-piece stream gathering remembering global neuron double-negative strength write threads
 *           lens j-space thinking chosen landmarks bearing kinship how-it-writes seed each-word-piece rubric apparatus
 *           sighting matched-control vault test nudge edit atlas census web steer-lens push names scientist
 * cautions  labels single late quiet frame thread-score nudge-defaults template think split bos
 * aliases   naked-eye faces pole stream-strength elevation through-the-lens write-size causal-check spot-check
 */

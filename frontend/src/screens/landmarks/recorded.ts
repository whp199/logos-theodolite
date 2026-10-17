/**
 * Findings of the study shown on the Landmarks page, ported from the approved canvas
 * (Landmarks.dc.html, renderVals()). These are recorded figures, not readings of the current
 * sighting: each block names the file it comes from. Never edit a number here without its source.
 */

export type Triple = [string, string, string];

/**
 * experiments/results/V7_tool_bugs_05_m1patch_frames.json, "frames": per pair [raw_WU cosine, fnw_WU cosine];
 * cond = raw_WU cond, condFnw = fnw_WU cond, self = cos_raw_vs_fnw_per_token (each word's raw row against
 * its weighted row).
 */
export interface RecordedFrame {
  words: Triple;
  cond: number;
  condFnw: number;
  self: Record<string, number>;
  cos: Record<string, [number, number]>;
}

export const FRAMES: RecordedFrame[] = [
  {
    words: ["she", "he", "it"],
    cond: 1.75,
    condFnw: 2.65,
    self: { she: 0.401, he: 0.3778, it: 0.5289 },
    cos: { "she|he": [0.4003, 0.6807], "she|it": [0.2731, 0.5085], "he|it": [0.3912, 0.5781] },
  },
  {
    words: ["woman", "man", "person"],
    cond: 1.59,
    condFnw: 1.86,
    self: { woman: 0.4066, man: 0.3903, person: 0.4053 },
    cos: { "woman|man": [0.2872, 0.2891], "woman|person": [0.2431, 0.3457], "man|person": [0.3614, 0.4858] },
  },
  {
    words: ["spider", "eight", "insect"],
    cond: 1.2,
    condFnw: 1.43,
    self: { spider: 0.3761, eight: 0.4269, insect: 0.3788 },
    cos: { "spider|eight": [0.0418, 0.148], "spider|insect": [0.1689, 0.1756], "eight|insect": [0.0661, 0.2989] },
  },
];

/** The recorded frame for a set of words, whatever their order; null when the study did not measure it. */
export function recordedFrame(normWords: string[]): RecordedFrame | null {
  const key = normWords.slice().sort().join("|");
  return FRAMES.find((fr) => fr.words.slice().sort().join("|") === key) ?? null;
}

export function cosOf(frame: RecordedFrame, a: string, b: string): [number, number] {
  return frame.cos[a + "|" + b] ?? frame.cos[b + "|" + a];
}

/**
 * experiments/results/V7_tool_bugs_04_bos_monument_summary.json, monument_order.example_L45, first prompt
 * (its last words: "Gemma is a helpful AI assistant. In conversations,"); report layer L45 = layer 46.
 * [order, bearing θ in degrees, elevation φ in degrees]
 */
export const ORDERS: [Triple, number, number][] = [
  [["she", "he", "it"], 17.8, 25.9],
  [["she", "it", "he"], 30.4, 7.0],
  [["he", "she", "it"], 48.6, 25.9],
  [["he", "it", "she"], 41.5, 37.4],
  [["it", "she", "he"], 43.8, 7.0],
  [["it", "he", "she"], 25.5, 37.4],
];

/**
 * experiments/results/V7_tool_bugs_04_bos_monument_summary.json, "monument_order": all six orders of
 * "she" · "he" · "it", 12 chat and 6 raw prompts pooled, medians by stretch of layers
 * (report layers L0–9 … L59 = layers 1–10 … 60).
 */
export const ORDER_STRETCHES = ["1–10", "11–20", "21–30", "31–40", "41–50", "51–59", "60"];
export const ORDER_TABLE: { label: string; cells: string[] }[] = [
  { label: "swing of the bearing θ", cells: ["81°", "94°", "85°", "58°", "78°", "64°", "76°"] },
  { label: "readings that swung over 30°", cells: ["84%", "98%", "91%", "79%", "94%", "78%", "83%"] },
  { label: "swing of the elevation φ", cells: ["61°", "71°", "69°", "51°", "65°", "51°", "64°"] },
  { label: "fraction of ‖h̃‖ caught", cells: ["2.8%", "2.6%", "3.0%", "3.9%", "3.4%", "3.3%", "2.4%"] },
];

/**
 * experiments/results/E6_tool_audit_02_sweep_demo.json, key "prepend_bos=False": the study's demonstration
 * question read with a single start mark, ids [2, 105, 2364]; azimuth_deg and elevation_deg to 0.1 degree,
 * indexed by repo layer index 0-59. Shown only beside the demonstration sighting (key "prepend_bos=True").
 */
export const SINGLE_MARK_AZ = [
  61.5, 18.5, 173.6, 94.7, 138.4, 3.9, 154.9, 134.8, 164.8, 146.8, 136.5, 138.6, 156.8, 152.9, 146.2, 138.1, 141.4,
  108.8, 118.3, 97.7, 119.1, 76.4, 65.6, 57.2, 68.8, 78.7, 85.2, 156.3, 26.2, 72.0, 54.2, 34.5, 41.5, 46.0, 24.6,
  13.9, -6.2, 12.6, 25.8, -134.5, -177.6, 153.9, 131.8, 113.4, 174.3, 154.8, -128.0, -179.1, 159.2, 68.0, 19.9, 24.4,
  95.9, 154.1, -60.9, -1.8, 20.8, 6.5, -99.3, -159.8,
];
export const SINGLE_MARK_EL = [
  5.0, -3.6, -24.9, -40.9, -16.9, 1.4, -30.0, -5.5, -47.3, -33.0, -35.6, -36.1, -31.8, -33.7, -37.3, -28.1, -37.2,
  -47.9, -50.4, -28.9, -21.6, -8.5, 1.9, 3.5, -0.4, -11.8, 3.7, -4.5, 21.5, 27.6, 60.6, 52.1, 49.0, 45.9, 46.4, 51.8,
  62.6, 42.2, 56.4, 59.6, 49.3, 43.5, 38.2, 45.6, 58.1, 18.0, 17.2, -19.0, 40.5, 26.1, 47.8, 32.3, 25.0, -33.6, -54.6,
  -50.5, -32.4, 8.1, 27.1, -45.0,
];

/** The canvas's starting sets (labels only; the instrument's own default is "spider" · "eight" · "insect"). */
export const PRESETS: { words: Triple; note: string }[] = [
  { words: ["she", "he", "it"], note: "pronouns · the study’s set" },
  { words: ["woman", "man", "person"], note: "nouns for people" },
  { words: ["spider", "eight", "insect"], note: "the instrument’s default" },
];

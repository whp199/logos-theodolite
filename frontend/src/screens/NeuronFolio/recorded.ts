/**
 * The neuron the approved canvas (Neuron.dc.html, version 7) is about, and the recorded study values it
 * shows. Shown only when the routed neuron is this one and the recorded sighting (the fixture) is loaded;
 * every other neuron gets live answers or bracketed placeholders.
 *
 * Real values from experiments/results/E6_tool_audit_02_sweep_demo.json, key "prepend_bos=True"
 * (prompt "Tell me about yourself." / "Hi! I", read at " I", position 21 of 22). Repo idx 53 = layer 54.
 * pole_neurons["53"], pole_acts["53"], pole_labels_jpp["53:0".."53:4"], pole_labels_naive["53:0".."53:4"],
 * m1_corrected_ffn_dominance[53], m1_tool_ffn_dominance[53].
 */

export const RECORDED = {
  idx: 53,
  layerNo: 54,
  neuron: 15491,
  face: 1 as const,
  nick: "the “I am” neuron",
  rank: 4,
  act: 4.34375,
};

/** The loudest neuron at layer 54 in the recorded sighting, neuron 2211. */
export const LOUDEST = 5.4375;
export const IDS = [2211, 5691, 18422, 13089, 15491, 17186, 12752, 14067, 17269, 19347, 3837, 1840, 16810, 8009, 10096];
export const ACTS = [
  5.4375, -5.1875, -4.8125, -4.46875, 4.34375, 3.875, -3.703125, 3.4375, 3.125, 3.03125, 2.984375, 2.84375, 2.78125,
  -2.671875, 2.65625,
];

export type Scored = [string, number];
/** the positive face of neuron 15491, through the lens (pole_labels_jpp["53:4"]) */
export const JPP: Scored[] = [
  [" Am", 27.062719345092773],
  [" jestem", 26.50611114501953],
  [" Somos", 26.39480209350586],
];
/** the same face to the naked eye (pole_labels_naive["53:4"]) */
export const NAIVE: Scored[] = [
  [" jestem", 29.19678497314453],
  [" jsem", 28.847858428955078],
  [" êtes", 28.770061492919922],
];
/** neuron 18422, third loudest, negative face (keys 53:2) */
export const CMP = {
  neuron: 18422,
  act: -4.8125,
  lens: [
    [" Am", 29.979196548461914],
    [" am", 29.965709686279297],
    ["Am", 29.90688133239746],
  ] as Scored[],
  eye: [
    [" am", 29.993074417114258],
    [" Am", 29.70199203491211],
    ["am", 29.556507110595703],
  ] as Scored[],
};
export const FFN = 0.7186428904533386;
export const FFN_TOOL = 0.6968382000923157;

/** The canvas's hand-written reading of the five labelled neighbours at layer 54 (keys 53:0 to 53:4). */
export const NB_NOTES: Record<
  number,
  { lens: string; lensNote: string; eye: string; eyeNote: string; verdict: string; vNote: string }
> = {
  2211: {
    lens: "?” ...” ?\"",
    lensNote: "marks that close a quotation",
    eye: "“などで” “Aquí” “之后的”",
    eyeNote: "Japanese; Spanish “here”; Chinese “after”",
    verdict: "disagree",
    vNote: "no word-piece shared",
  },
  5691: {
    lens: "“IRE” “rivets” “ism”",
    lensNote: "",
    eye: "“গুলো” “যত” “कभी”",
    eyeNote: "Bengali and Hindi word-pieces",
    verdict: "disagree",
    vNote: "no word-piece shared",
  },
  18422: {
    lens: "“Am” “am” “Am”",
    lensNote: "English “am”, capitalised or not",
    eye: "“am” “Am” “am”",
    eyeNote: "English “am”, capitalised or not",
    verdict: "agree",
    vNote: "“am” on both sides",
  },
  13089: {
    lens: "“≫” “≪”",
    lensNote: "double-angle signs, after a stray byte that is not a whole letter",
    eye: "“ﺍﻟ” “iolipin” “Với”",
    eyeNote: "Arabic “the”; a word fragment; Vietnamese “with”",
    verdict: "disagree",
    vNote: "no word-piece shared",
  },
  15491: {
    lens: "“Am” “jestem” “Somos”",
    lensNote: "English, Polish, Spanish: forms of “to be”",
    eye: "“jestem” “jsem” “êtes”",
    eyeNote: "Polish, Czech, French: forms of “to be”",
    verdict: "partly agree",
    vNote: "“jestem” on both sides",
  },
};

/** Glosses the canvas wrote for the word-pieces of this face (translations, true wherever they appear). */
export const TOKEN_GLOSS: Record<string, string> = {
  " Am": "English: “am”, capitalised",
  " jestem": "Polish: “I am”",
  " Somos": "Spanish or Portuguese: “we are”",
  " jsem": "Czech: “I am”",
  " êtes": "French: “(you) are”",
};

/** The canvas's raw readout for this neuron: recorded values only. */
export const RAW_READOUT = {
  source: "experiments/results/E6_tool_audit_02_sweep_demo.json, key prepend_bos=True",
  layer: 54,
  index: 53,
  neuron: 15491,
  read_at: " I",
  position: 21,
  n_tokens: 22,
  signed_activation: 4.34375,
  polarity: "+",
  rank_by_abs: 5,
  pole_label_jpp: JPP,
  pole_label_naive: NAIVE,
  layer_poles_top15: IDS.map((id, k) => [id, ACTS[k]]),
  ffn_dominance: { corrected: FFN, tool_before_fix: FFN_TOOL },
};

export const ORDINALS = [
  "first",
  "second",
  "third",
  "fourth",
  "fifth",
  "sixth",
  "seventh",
  "eighth",
  "ninth",
  "tenth",
  "eleventh",
  "twelfth",
  "thirteenth",
  "fourteenth",
  "fifteenth",
];
export const ordinal = (rank: number) => ORDINALS[rank] ?? `${rank + 1}th`;

/**
 * Recorded readings for the layer leaf that the sighting itself does not carry.
 *
 * Everything here is copied from the approved canvas (Layer.dc.html, renderVals), whose comments name the
 * sources. The hand-written glosses and notes about layer 36 apply only to the study's demonstration
 * sighting (experiments/results/E6_tool_audit_02_sweep_demo.json, key "prepend_bos=True", repo index 35);
 * the study findings about layer 36 (the median stream strength, the E3 cosines) apply to layer 36 of any
 * sighting. Never extend these tables with numbers that are not in the sources.
 */

/** Repo index of the layer the canvas describes by hand (layer 36). */
export const L36_IDX = 35;

/** The canvas's subtitle for layer 36 of the demonstration sighting. */
export const L36_SUBTITLE = "where, through the lens, the word “I” murmurs of machines";

/**
 * Glosses and notes written for the canvas for the five read poles of layer 36
 * (pole_labels_jpp / pole_labels_naive "35:0" to "35:4"; the labels are of the active face).
 */
export const L36_POLES: Record<number, { lensGloss: string; eyeGloss: string; note: string }> = {
  21189: {
    lensGloss: "English “myself”, Hindi “me”, Chinese “of me”",
    eyeGloss: "Chinese “one’s own”, Chinese “oneself”, Bengali “me”",
    note: "No word in common, yet both eyes speak of the self.",
  },
  6709: {
    lensGloss: "a Chinese syllable in pinyin, the code for the Japanese yen, Turkish “fee”",
    eyeGloss: "Turkish “thanks to”, English “will”, Spanish “thank you”",
    note: "",
  },
  14263: {
    lensGloss: "Chinese “teenagers”, Chinese “age”, Russian “to use”",
    eyeGloss: "the Gurmukhi numeral zero, French “precision”, a fragment as in “sceptic”",
    note: "",
  },
  12068: {
    lensGloss: "",
    eyeGloss: "the last is Italian “created”",
    note: "“creato” comes close to “creator”, but only exact matches count.",
  },
  627: {
    lensGloss: "the first is joined to the word before",
    eyeGloss: "",
    note: "Agreement was tested as a sign of trust only at layers 41 to 58; down here it is a hint, not a warrant.",
  },
};

/**
 * jpp_tracked_logit / naive_tracked_logit at index 35 of the demonstration sighting (softcapped scores).
 * The fixture now carries tracked logits for every layer; this table is kept as a cross-check of layer 36.
 */
export const L36_TRACKED_LOGITS: Record<string, [number, number]> = {
  she: [10.37, 4.86],
  he: [5.77, 3.82],
  it: [4.01, 11.08],
  her: [7.87, 10.09],
  him: [1.11, 7.64],
  woman: [3.22, -11.88],
  man: [6.46, -1.76],
  they: [-3.78, -2.99],
};

/** Index 35 of the demonstration sighting: m1_cos_tool_vs_true_ffn, m1_cos_tool_vs_true_attn, proj_norm. */
export const L36_COS_FFN = 0.0641;
export const L36_COS_ATTN = 0.9998;
export const L36_PROJ_NORM = 0.017288299277424812;

/** Pictographs the canvas names in words instead of showing them. */
export const PICTOGRAPHS: Record<number, string> = {
  0x1f916: "a robot’s face",
};

/** The neuron-label study N1 sampled these layers (counted 1-60; REPORT §3.2). */
export const N1_LAYERS = [11, 21, 31, 41, 46, 51, 56, 58];

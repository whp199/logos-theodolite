/**
 * The approved canvas's hand-written notes (Main.dc.html) on the loudest neuron at three layers of the
 * study's demonstration sighting (E6_tool_audit_02_sweep_demo.json, prepend_bos=True). They apply only
 * when that recorded sighting is shown (its words come from the recorded labels); a live sighting builds
 * its note from the server's labels.
 * Keyed by "repo index:neuron".
 */
export const RECORDED_POLE_NOTES: Record<string, { gloss: string; trust: string }> = {
  "35:21189": {
    gloss: "English “myself”, Hindi “me”, Chinese “of me”: one neuron, one idea, three tongues.",
    trust: "A rumour, not a finding: below layer 41, neuron labels proved unreliable in the study.",
  },
  "55:16649": {
    gloss: "The lens and the naked eye agree on “AM”: a good sign.",
    trust: "Layers 41 to 58 are where neuron labels held up under causal tests.",
  },
  "59:12584": {
    gloss: "A neuron for the Spanish and German “es”.",
    trust: "At the last layer the lens and the naked eye see the same thing.",
  },
};

/** The canvas's stations: the layers (repo indices) where the recorded sighting kept the lens's top five. */
export const CANVAS_STATIONS = [10, 20, 30, 35, 40, 45, 50, 55, 59];

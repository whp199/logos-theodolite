/**
 * The approved canvas's hand-written notes (Main.dc.html) on the loudest neuron at three layers of the
 * study's demonstration sighting (E6_tool_audit_02_sweep_demo.json, prepend_bos=True). They apply only
 * when that recorded sighting is shown (its words come from the recorded labels); a live sighting builds
 * its note from the server's labels. The note on 35:21189 is amended from the canvas: layer 36 lies in the stretch N1 did not
 * sample (layers 32-40), and its words are the lens's reading, not a tested label (REPORT errata E9, E43).
 * Keyed by "repo index:neuron".
 */
export const RECORDED_POLE_NOTES: Record<string, { gloss: string; trust: string }> = {
  "35:21189": {
    gloss: "English “myself”, Hindi “me”, Chinese “of me”: through the lens, one idea in three tongues.",
    trust:
      "A rumour, not a finding: the study did not test labels at layers 32 to 40, where this one sits; they held up at 41 to 58 and were no better than chance at 11 to 31.",
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

import { FNW_DEFAULT, FRAME_AS_BUILT, type FrameOptions, type Sighting } from "../store/useTheodoliteStore";

/**
 * How a sighting's survey frame was built, in the pages' words, from what the server echoed (`sighting.frame`)
 * and the survey mode. The recorded sighting (no `frame`) was taken in the frame as first built: Gram–Schmidt on
 * the raw W_U rows, every layer's h·Jᵀ read in the last layer's frame.
 *
 * The share of the state the frame catches (proj_frac) is ‖p‖ / ‖target‖, where the target is h·Jᵀ, or the raw h
 * when each layer has its own frame, less the matched control's state when centred (theodolite_math
 * project_trajectory_to_3d). Only the uncentred h·Jᵀ share of a frame on the raw rows compares with the study's
 * 2.4–3.9% (V7): the weighted rows fnw ⊙ W_U span another space, which V7 did not measure.
 */
export function describeFrame(s: Sighting) {
  const fr = s.frame;
  const poleMotif = s.surveyMode === "pole_motif";
  const lowdin = fr?.squaring === "lowdin";
  const raw = fr?.state === "h_raw";
  const centred = fr?.centred === true;
  const fnw = fr?.fnwWeighted === true;
  const method = lowdin ? "Löwdin" : "Gram–Schmidt";
  const rows = (w: string) => (
    <>
      {fnw ? "fnw ⊙ " : ""}W<sub>U</sub>[{w}]
    </>
  );
  /** the state each layer is read from, as a formula */
  const state = raw ? (
    <>
      h<sub>ℓ</sub>
    </>
  ) : (
    <>
      h̃<sub>ℓ</sub> = h<sub>ℓ</sub> · J<sub>ℓ</sub>
      <sup>⊤</sup>
    </>
  );
  /** the norm the caught share is taken of */
  const denom = raw ? (
    centred ? (
      <>
        ‖h<sub>ℓ</sub> − c<sub>ℓ</sub>‖
      </>
    ) : (
      <>
        ‖h<sub>ℓ</sub>‖
      </>
    )
  ) : centred ? (
    <>
      ‖h̃<sub>ℓ</sub> − c̃<sub>ℓ</sub>‖
    </>
  ) : (
    <>
      ‖h̃<sub>ℓ</sub>‖
    </>
  );
  const denomWords = raw
    ? centred
      ? "of the stream less the control’s state"
      : "of the stream itself"
    : centred
      ? "of the lens-read state less the control’s"
      : "of the lens-read state";
  const [a, b, c] = s.landmarks;
  const frameFormula = poleMotif ? (
    <>
      e<sub>1</sub>, e<sub>2</sub>, e<sub>3</sub> = {method}(w<sub>gate</sub>, (σ · w<sub>down</sub>) · J<sub>ℓ</sub>
      <sup>⊤</sup>, the main reader’s w<sub>gate</sub>), a frame locked to one neuron
    </>
  ) : (
    <>
      e<sub>1</sub>, e<sub>2</sub>, e<sub>3</sub> = {method}({rows(a)}, {rows(b)}, {rows(c)}) in fp32
      {fr?.perLayer ? (
        <>
          , pulled back by each layer’s own lens J<sub>ℓ</sub>
        </>
      ) : null}
    </>
  );
  const notes: string[] = [];
  notes.push(
    lowdin
      ? "squared all at once: the same three axes for every order, the order naming which sets 0°, which 90° and which points up"
      : "order-dependent (REPORT §3.4, bug 3)",
  );
  if (raw) notes.push("each layer read in its own frame");
  else notes.push("every layer read in the last layer’s frame");
  if (centred) notes.push(`read against a matched control’s state${fr && fr.nCentre > 1 ? ` (the mean of ${fr.nCentre})` : ""}`);
  return {
    poleMotif,
    lowdin,
    raw,
    centred,
    fnw,
    method,
    state,
    denom,
    denomWords,
    frameFormula,
    notes,
    /** the frame as first built, whose caught share the study measured */
    asBuilt: !fr || (!lowdin && !raw && !centred && !fnw && !poleMotif),
    /** the interface's default frame (FRAME_MENDED): squared all at once, one frame from the last layer, uncentred */
    mended: lowdin && !raw && !centred && !poleMotif && fnw === FNW_DEFAULT,
    /** the caught share compares with V7's 2.4–3.9% only for the uncentred h·Jᵀ frame of raw rows (Löwdin spans the
     * same space as Gram–Schmidt; the weighted rows do not) */
    studyComparable: !raw && !centred && !poleMotif && !fnw,
    exportKey: raw ? (centred ? "proj_frac_of_h_minus_control" : "proj_frac_of_h") : centred ? "proj_frac_of_hJ_minus_control" : "proj_frac_of_hJ",
  };
}

/** The frame options a page's frame-dependent words describe. `own`: they are the shown sighting's own frame (else the
 * frame the next sighting will be taken in); `recorded`: that frame was not echoed but known, as for the recorded
 * sighting, which was taken in the frame as first built. */
export type ShownFrame = FrameOptions & { own: boolean; recorded: boolean };

/**
 * The frame a page should describe: the shown sighting's own when it was taken with the landmarks (or neuron) set on
 * the page (`matches`), else the frame the next sighting will be taken in.
 * - A live sighting's frame is the one the server echoed. A live reply that echoes none came from a server older than
 *   the frame options, which built every frame as first built.
 * - The recorded sighting carries no frame: it was taken in the frame as first built, Gram–Schmidt on the raw rows.
 * - The next sighting's frame is `frameOpts`; in a neuron frame (pole_motif with a neuron chosen) squared by
 *   `neuron.neuronSquaring`, and never weighted or per layer, which apply to a frame from words only
 *   (engine_runtime.py prepare_sweep). In pole_motif with no neuron chosen the server falls back to the frame from
 *   words, so that is `frameOpts` too.
 * Every frame-dependent text, label and drawing on a page should come from this one helper.
 */
export function shownFrameOptions(
  sighting: Sighting | null,
  matches: boolean,
  frameOpts: FrameOptions,
  neuron: {
    surveyMode: "monument" | "pole_motif";
    neuronSquaring: FrameOptions["squaring"];
    lockedPole: readonly number[] | null;
  } | null = null,
): ShownFrame {
  if (matches && sighting) {
    const fr = sighting.frame;
    if (sighting.source === "live" && fr) {
      const centre = fr.centred || fr.nCentre > 0;
      return { squaring: fr.squaring, perLayer: fr.perLayer, fnw: fr.fnwWeighted, centre, own: true, recorded: false };
    }
    return { ...FRAME_AS_BUILT, own: true, recorded: true };
  }
  if (neuron?.surveyMode === "pole_motif" && neuron.lockedPole) {
    const squaring = neuron.neuronSquaring;
    return { squaring, perLayer: false, fnw: false, centre: frameOpts.centre, own: false, recorded: false };
  }
  return { ...frameOpts, own: false, recorded: false };
}

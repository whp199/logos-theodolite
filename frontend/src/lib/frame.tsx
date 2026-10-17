import type { Sighting } from "../store/useTheodoliteStore";

/**
 * How a sighting's survey frame was built, in the pages' words, from what the server echoed (`sighting.frame`)
 * and the survey mode. The recorded sighting (no `frame`) was taken in the frame as first built: Gram–Schmidt on
 * the raw W_U rows, every layer's h·Jᵀ read in the last layer's frame.
 *
 * The share of the state the frame catches (proj_frac) is ‖p‖ / ‖target‖, where the target is h·Jᵀ, or the raw h
 * when each layer has its own frame, less the matched control's state when centred (theodolite_math
 * project_trajectory_to_3d). Only the uncentred h·Jᵀ share compares with the study's 2.4–3.9% (V7).
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
  notes.push(lowdin ? "squared all at once, the same for every order" : "order-dependent (REPORT §3.4, bug 3)");
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
    /** the caught share compares with V7's 2.4–3.9% only for the uncentred h·Jᵀ frame */
    studyComparable: !raw && !centred && !poleMotif,
    exportKey: raw ? (centred ? "proj_frac_of_h_minus_control" : "proj_frac_of_h") : centred ? "proj_frac_of_hJ_minus_control" : "proj_frac_of_hJ",
  };
}

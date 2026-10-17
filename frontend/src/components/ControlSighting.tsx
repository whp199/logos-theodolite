import React from "react";
import { useTheodoliteStore } from "../store/useTheodoliteStore";
import { pct, wp } from "../lib/words";

/**
 * The matched control, as an apparatus row's content: which sighting is kept, a button to keep or clear
 * it, and (given a layer) what the control was thinking there. The study reads lens outputs against a
 * matched control (a nameless or background condition), never alone (REPORT §3.5).
 */
export const ControlSighting: React.FC<{ layerIdx?: number }> = ({ layerIdx }) => {
  const sighting = useTheodoliteStore((s) => s.sighting);
  const control = useTheodoliteStore((s) => s.control);
  const setControl = useTheodoliteStore((s) => s.setControl);
  const isThis = control !== null && control === sighting;
  const at = control && layerIdx !== undefined && !isThis ? control.layers[layerIdx] : null;
  return (
    <>
      {control ? (
        <span>
          “{control.question}”{control.answerBegins ? <span className="italic"> / “{control.answerBegins}”</span> : null} — went on{" "}
          <span className="italic">“{control.continuation.trim()}”</span> ({control.source === "live" ? "live" : "recorded"}
          {isThis ? "; the sighting shown now" : ""})
        </span>
      ) : (
        <span className="placeholder">[NONE SET]</span>
      )}
      {at ? (
        <span>
          At layer {at.no} the control was thinking{" "}
          {at.lensTop?.[0] ? (
            <>
              {wp(at.lensTop[0].token)} ({pct(at.lensTop[0].prob)})
            </>
          ) : (
            <span className="placeholder">[NOT RECORDED AT THIS LAYER]</span>
          )}
          {at.ffnShare != null ? <>, remembering {Math.round(at.ffnShare * 100)}%</> : null}.
        </span>
      ) : null}
      <button type="button" className="btn-link" disabled={!sighting} onClick={() => setControl(isThis ? null : sighting)}>
        {isThis ? "clear the control" : control ? "keep the sighting shown now as the control instead" : "keep the sighting shown now as the control"}
      </button>
    </>
  );
};

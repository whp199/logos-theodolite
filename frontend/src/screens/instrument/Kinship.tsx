import React, { useMemo } from "react";
import type { Sighting } from "../../store/useTheodoliteStore";
import { Missing } from "../../components/kit";
import { pieceWord } from "../../lib/words";

const SHADES = ["#E5DCCA", "#D7CBB5", "#C8BA9F", "#BAA989", "#AC9874", "#9E875E", "#8F7648", "#816533"];
const TICKS = [1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60];

/** v · kinship of layers (apparatus): linear CKA between every pair of layers. */
export const Kinship: React.FC<{ sighting: Sighting; sel: number }> = ({ sighting, sel }) => {
  const k = sighting.kinship;
  const paths = useMemo(() => {
    const kp = ["", "", "", "", "", "", "", ""];
    if (!k) return kp;
    for (let i = 0; i < 60; i++) {
      for (let j = 0; j < 60; j++) {
        const v = i === j ? 1 : k[i * 60 + j];
        const b = Math.max(0, Math.min(7, Math.floor((v - 0.2) / 0.1)));
        kp[b] += `M${40 + j * 10},${40 + i * 10}h10v10h-10Z`;
      }
    }
    return kp;
  }, [k]);
  const st = sighting.kinStats;
  const c = 45 + 10 * sel;
  // the server reads kinship over every state up to the position read (kinship_n_states = read_pos + 1)
  const piece = sighting.tokenIndex;
  return (
    <section
      aria-labelledby="kin-head"
      className="apparatus"
      style={{ marginTop: 44, paddingTop: 14, flexDirection: "row", flexWrap: "wrap", gap: "24px 56px", alignItems: "flex-start", fontSize: 19, lineHeight: 1.55 }}
    >
      <div style={{ flex: "1 1 340px", minWidth: 0, display: "flex", flexDirection: "column", gap: 12 }}>
        <span className="kicker">apparatus</span>
        <h2 id="kin-head" className="h-sec">
          v · kinship of layers
        </h2>
        <p style={{ margin: 0, fontSize: 17 }}>
          Each cell compares two layers: how alike they arrange the {sighting.nTokens ?? "[N]"} word-pieces read up to here
          {piece >= 1 ? `, the prompt’s ${sighting.promptLen ?? "[P]"} and the first ${piece} of its own answer` : ""}
          {sighting.readAt != null ? `, the last of them ${pieceWord(sighting.readAt)}` : ""}, from 0.2 (pale) to 1 (deep gilt). The red
          cross marks layer {sel + 1}.
        </p>
        <dl className="num" style={{ margin: 0, display: "grid", gridTemplateColumns: "minmax(0, 1fr) auto", gap: "4px 16px", fontSize: 15 }}>
          <dt style={{ color: "#6B5D4D" }}>mean, off the diagonal</dt>
          <dd style={{ margin: 0, textAlign: "right" }}>{st ? st.meanOffDiag.toFixed(2) : <Missing />}</dd>
          <dt style={{ color: "#6B5D4D" }}>neighbouring layers</dt>
          <dd style={{ margin: 0, textAlign: "right" }}>{st ? st.adjacent.toFixed(2) : <Missing />}</dd>
          <dt style={{ color: "#6B5D4D" }}>layer 11 against layer 51</dt>
          <dd style={{ margin: 0, textAlign: "right" }}>{st ? st.l11vs51.toFixed(2) : <Missing />}</dd>
        </dl>
        <p style={{ margin: 0, fontSize: 15, color: "#6B5D4D" }}>
          Linear CKA (Kornblith et al. 2019), on the resid_post states of every word-piece read up to here.
          {sighting.source === "fixture"
            ? " This sighting was taken before the single-BOS fix, which lowers kinship by about 0.06 on average (E6, V7)."
            : " Taken by the running server, with the single start mark."}
          {piece >= 1 || sighting.generation?.sightEvery
            ? " Each later word-piece of the answer brings one more word-piece into the comparison, so the map can change from one word-piece to the next only because there is more text: do not read such a change as a finding about the word-piece."
            : null}
        </p>
        <div className="num" style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "4px 10px", fontSize: 14, color: "#6B5D4D" }}>
          <span>0.2</span>
          <span aria-hidden="true" style={{ display: "flex", gap: 2 }}>
            {SHADES.map((s) => (
              <span key={s} style={{ width: 16, height: 12, background: s, display: "inline-block" }} />
            ))}
          </span>
          <span>1.0</span>
        </div>
      </div>
      <figure style={{ flex: "999 1 520px", minWidth: 0, margin: 0 }}>
        {k ? (
          <svg
            viewBox="0 0 660 660"
            role="img"
            aria-label={`A 60 by 60 map of how alike each pair of layers is, shaded from pale to deep gilt, with layer ${sel + 1} crossed in red. Neighbouring layers are close kin; the first and last layers stand apart.`}
            style={{ width: "100%", maxWidth: 620, height: "auto", display: "block", margin: "0 auto" }}
          >
            {paths.map((d, i) => (
              <path key={i} d={d} fill={SHADES[i]} />
            ))}
            <rect x="40" y="40" width="600" height="600" fill="none" stroke="#A07A2E" strokeWidth="1" />
            <line x1={c} y1="40" x2={c} y2="640" stroke="#8F2B1E" strokeWidth="1.2" />
            <line x1="40" y1={c} x2="640" y2={c} stroke="#8F2B1E" strokeWidth="1.2" />
            <g fontFamily="EB Garamond, Garamond, serif" fontSize="13" fill="#7A5C28" textAnchor="middle" style={{ fontVariantNumeric: "lining-nums tabular-nums" }}>
              {TICKS.map((t) => (
                <text key={t} x={t === 1 ? 45 : 35 + 10 * t} y="32">
                  {t}
                </text>
              ))}
            </g>
            <g fontFamily="EB Garamond, Garamond, serif" fontSize="13" fill="#7A5C28" textAnchor="end" style={{ fontVariantNumeric: "lining-nums tabular-nums" }}>
              {TICKS.map((t) => (
                <text key={t} x="34" y={t === 1 ? 49 : 39 + 10 * t}>
                  {t}
                </text>
              ))}
            </g>
          </svg>
        ) : (
          <p className="muted">
            <Missing what="kinship measured in a live sighting" />
          </p>
        )}
      </figure>
    </section>
  );
};

import React from "react";
import type { Sighting } from "../../store/useTheodoliteStore";
import { IfApparatus, Rubric } from "../../components/kit";
import { bare } from "../../lib/words";
import { lensBand, useReadingLens } from "../../lib/lens";
import { useApparatus } from "../../reading";

const f = (n: number) => Math.round(n * 10) / 10;

/** iv · the ascent of one word: the gathering/remembering split at every layer, with station words. */
export const Ascent: React.FC<{ sighting: Sighting; sel: number; stations: number[]; onPick: (idx: number) => void }> = ({
  sighting,
  sel,
  stations,
  onPick,
}) => {
  const apparatus = useApparatus();
  const { lens } = useReadingLens(sighting);
  const read = sighting.readAt != null ? `“${bare(sighting.readAt)}”` : "the last word";
  return (
    <section aria-labelledby="ascent-head" style={{ marginTop: 44, display: "flex", flexDirection: "column", gap: 12 }}>
      <div className="row-between">
        <h2 id="ascent-head" className="h-sec">
          iv · the ascent of one word
        </h2>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 22px", fontSize: 15, color: "#6B5D4D" }}>
          <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span aria-hidden="true" className="legend-swatch" style={{ background: "#2E4A7D" }} />
            gathered (attention)
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span aria-hidden="true" className="legend-swatch" style={{ background: "#A07A2E" }} />
            remembered (feed-forward)
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span aria-hidden="true" style={{ color: "#A07A2E" }}>
              ✦
            </span>
            looks back over everything
          </span>
          <IfApparatus>
            {sighting.layers.some((L) => L.ffnShareBeforeFix != null) ? (
              <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span
                  aria-hidden="true"
                  style={{ width: 14, height: 1, background: "#F6F0E3", borderTop: "1px solid #2B2117", borderBottom: "1px solid #2B2117", display: "inline-block" }}
                />
                the split as read before the V7 fix
              </span>
            ) : null}
          </IfApparatus>
        </div>
      </div>
      <Rubric style={{ margin: 0 }}>
        Read left to right, the word {read} rising through the model. Each bar is how the {lens.name} lens weighs what that layer gathered
        and remembered, and the words above are that lens’s readings. Below layer 21 the study could validate none of them, so those words
        are drawn faint; at layers 21 to 30 trust only how sightings rank against one another, not the words themselves. Choose any marked
        layer to open it above.
      </Rubric>
      <div style={{ overflowX: "auto", paddingBottom: 6 }}>
        <div style={{ minWidth: 980, position: "relative", paddingTop: 74 }}>
          <div style={{ position: "absolute", left: 0, right: 0, top: 0, height: 70 }}>
            {stations.map((s) => {
              const word = sighting.layers[s]?.lensTop?.[0]?.token;
              const on = s === sel;
              // below layer 21 the study validated nothing the lens reads (lib/lens.ts, lensBand): drawn faint
              const faint = word != null && lensBand(s + 1) === "untested";
              return (
                <button
                  key={s}
                  type="button"
                  onClick={() => onPick(s)}
                  aria-pressed={on}
                  aria-label={`Layer ${s + 1}${
                    word == null
                      ? ""
                      : s === 59
                        ? `, its own likeliest next word “${bare(word)}”`
                        : `, through the lens “${bare(word)}”${faint ? ", where the study could validate nothing" : ""}`
                  }`}
                  title={faint ? "Below layer 21 the study could validate nothing the lens reads" : undefined}
                  style={{
                    position: "absolute",
                    left: `${f(((s + 0.5) / 60) * 100)}%`,
                    top: 0,
                    transform: "translateX(-50%)",
                    background: "transparent",
                    border: 0,
                    borderBottom: on ? "2px solid #8F2B1E" : "2px solid transparent",
                    padding: "4px 6px 6px",
                    minHeight: 60,
                    minWidth: 44,
                    cursor: "pointer",
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    color: on ? "#8F2B1E" : "#2B2117",
                  }}
                >
                  <span
                    style={{
                      fontSize: 19,
                      fontStyle: "italic",
                      whiteSpace: "nowrap",
                      lineHeight: 1.3,
                      ...(faint ? { opacity: 0.6, color: on ? undefined : "#6B5D4D" } : null),
                    }}
                  >
                    {word != null ? `“${bare(word)}”` : "·"}
                  </span>
                  <span className="num" style={{ fontSize: 13, letterSpacing: "0.08em", color: "#6B5D4D" }}>
                    {s + 1}
                  </span>
                </button>
              );
            })}
          </div>
          <div aria-hidden="true" style={{ display: "grid", gridTemplateColumns: "repeat(60, minmax(0, 1fr))", gap: 2, height: 76, alignItems: "stretch" }}>
            {sighting.layers.map((L) => {
              const v = L.ffnShare;
              return (
                <div
                  key={L.idx}
                  onClick={() => onPick(L.idx)}
                  title={`Layer ${L.no}`}
                  style={{
                    position: "relative",
                    display: "flex",
                    flexDirection: "column",
                    cursor: "pointer",
                    outline: L.idx === sel ? "1.5px solid #8F2B1E" : "0 solid transparent",
                    outlineOffset: 1,
                    // a split the sighting did not record: an empty, dotted column
                    border: v == null ? "1px dotted #C9BEAC" : undefined,
                  }}
                >
                  {v != null ? (
                    <>
                      <div style={{ height: f(v * 76), background: "#A07A2E" }} />
                      <div style={{ height: f((1 - v) * 76), background: "#2E4A7D" }} />
                    </>
                  ) : null}
                  {apparatus && L.ffnShareBeforeFix != null ? (
                    <div
                      style={{
                        position: "absolute",
                        left: 0,
                        right: 0,
                        top: f(L.ffnShareBeforeFix * 76),
                        height: 1,
                        marginTop: -1.5,
                        background: "#F6F0E3",
                        borderTop: "1px solid #2B2117",
                        borderBottom: "1px solid #2B2117",
                      }}
                    />
                  ) : null}
                </div>
              );
            })}
          </div>
          <div
            aria-hidden="true"
            className="num"
            style={{ display: "grid", gridTemplateColumns: "repeat(60, minmax(0, 1fr))", gap: 2, marginTop: 6, fontSize: 12, color: "#7A5C28", textAlign: "center" }}
          >
            {sighting.layers.map((L) => (
              <div key={L.idx} style={{ display: "flex", flexDirection: "column", alignItems: "center", lineHeight: 1.2, whiteSpace: "nowrap" }}>
                <span>{L.isGlobal ? "✦" : " "}</span>
                <span style={{ color: "#6B5D4D" }}>{L.idx === 0 || L.isGlobal ? L.no : " "}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

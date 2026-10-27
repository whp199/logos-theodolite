/**
 * Small pieces the atlas's panels share: a face's words under the label gate, word chips, a neuron's citation.
 * Every lens label goes through lib/labels.ts as on every other page: below page layer 41, or where the lens and the
 * naked eye share none of their first ten, the main reading shows the plate and the apparatus keeps the words. The
 * steer lens is a measured effect (the next word-piece's change when the neuron is pushed), which the gate does not
 * cover: it is shown as such.
 */
import React from "react";
import { HeldBackWords } from "../../components/HeldBack";
import { IfApparatus } from "../../components/kit";
import { labelGate, layerIdx, parseLabelCheck } from "../../lib/labels";
import { wp } from "../../lib/words";
import type { FaceWords, LensFace, PooledWord } from "./api";

export const words = (list: readonly string[] | null | undefined, k = 6) => (list ?? []).slice(0, k).map((t) => wp(t)).join("  ");

export function gateOf(layer: number, check: unknown) {
  return labelGate(layerIdx(layer), parseLabelCheck(check));
}

/** One face's words: the steer lens's when there are any, then the lens's under the gate. */
export const FaceLine: React.FC<{
  layer: number;
  face: "+" | "-";
  w: FaceWords | LensFace | null | undefined;
  k?: number;
  compact?: boolean;
  showSteer?: boolean;
}> = ({ layer, face, w, k = 6, compact = true, showSteer = true }) => {
  if (!w) return null;
  const steer = "steer" in w ? w.steer : null;
  const lens = w.lens;
  const gate = gateOf(layer, w.check);
  return (
    <span className="atlas-face">
      <span className={`atlas-sgn ${face === "+" ? "up" : "down"}`}>{face === "+" ? "+" : "−"}</span>
      {showSteer && steer && steer.length ? (
        <span className="atlas-steer" title="the steer lens: the next word-piece’s measured rise when the neuron is pushed">
          {words(steer, k)}
        </span>
      ) : null}
      {lens && lens.length ? (
        gate.heldBack ? (
          <HeldBackWords gate={gate} words={lens} eye={w.eye} compact={compact} size={15} line={!compact} />
        ) : (
          <span className="atlas-lens" title="through the J++ lens (fitted on the base model)">
            {words(lens, k)}
          </span>
        )
      ) : null}
      {"stale" in w && w.stale ? (
        <IfApparatus>
          <span className="atlas-stale" title={w.stale}>
            [read before its edit]
          </span>
        </IfApparatus>
      ) : null}
    </span>
  );
};

export const Chips: React.FC<{ list: PooledWord[] | undefined; onPick?: (tok: string) => void; empty?: string }> = ({ list, onPick, empty = "nothing above chance" }) =>
  list && list.length ? (
    <span className="atlas-chips">
      {list.map((t, i) => (
        <button key={i} type="button" className="atlas-chip" title={`${t.n} neurons · score ${t.w}`} onClick={() => onPick?.(t.tok)}>
          {t.tok}
          <i>{t.n}</i>
        </button>
      ))}
    </span>
  ) : (
    <span className="muted atlas-small">{empty}</span>
  );

export const Cite: React.FC<{ layer: number; n: number; onOpen?: () => void; head?: boolean }> = ({ layer, n, onOpen, head }) => (
  <button type="button" className="atlas-cite" onClick={onOpen}>
    L{layer + 1}
    {head ? " h" : " N"}
    {n}
  </button>
);

export const ErrorLine: React.FC<{ error: string | null | undefined }> = ({ error }) =>
  error ? (
    <p className="error-line" role="alert" style={{ fontSize: 15 }}>
      {error}
    </p>
  ) : null;

export const fmtDose = (d: number) => (d > 0 ? "+" : d < 0 ? "−" : "") + Math.abs(d);

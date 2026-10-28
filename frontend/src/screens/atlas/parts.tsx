/**
 * Small pieces the atlas's panels share: a face's words under the label gate, word chips, a neuron's citation.
 * Every lens label goes through lib/labels.ts as on every other page: below page layer 41, or where the lens and the
 * naked eye share none of their first ten, the main reading shows the plate and the apparatus keeps the words. The
 * steer lens is a measured effect (the next word-piece's change when the neuron is pushed), which the gate does not
 * cover: it is shown as such.
 */
import React from "react";
import { HeldBackWords } from "../../components/HeldBack";
import { labelGate, layerIdx, parseLabelCheck } from "../../lib/labels";
import { wp } from "../../lib/words";
import { noteKey, type CensusCard, type FaceWords, type LensFace, type PooledWord } from "./api";
import { useAtlas } from "./state";

export const words = (list: readonly string[] | null | undefined, k = 6) => (list ?? []).slice(0, k).map((t) => wp(t)).join("  ");

export function gateOf(layer: number, check: unknown) {
  return labelGate(layerIdx(layer), parseLabelCheck(check));
}

/** One face's words, each list named in small capitals: the steer lens's when there are any (a measured push, not a
 * label), then the lens's under the gate. Words read before a vault edit in force say so in the main reading. */
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
      {"stale" in w && w.stale ? (
        <span className="atlas-stale" title={w.stale}>
          [read before its edit]
        </span>
      ) : null}
      {showSteer && steer && steer.length ? (
        <span className="atlas-steer" title="the steer lens: the next word-piece’s measured rise when the neuron is pushed">
          <span className="atlas-src">steer</span>
          {words(steer, k)}
        </span>
      ) : null}
      {lens && lens.length ? (
        gate.heldBack ? (
          <HeldBackWords gate={gate} words={lens} eye={w.eye} compact={compact} size={15} line={!compact} />
        ) : (
          <span className="atlas-lens" title="through the J++ lens, fitted on the base model and read here on the instruct model">
            <span className="atlas-src">lens</span>
            {words(lens, k)}
          </span>
        )
      ) : null}
    </span>
  );
};

/** How many times the most the neuron can write by itself an add push of `dose` is (NeuronLens.ceiling), or null. */
export const overCeiling = (dose: number, c: { rel: number | null } | null | undefined) =>
  c && c.rel ? Math.abs(dose) / c.rel : null;

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

/** How many of a census's neurons beat every control, and how many would by chance alone: N / (C + 1). */
export function censusChance(units: { kind: string; z: number }[] | undefined) {
  const us = units ?? [];
  const ctl = us.filter((u) => u.kind === "random").map((u) => u.z);
  const neu = us.filter((u) => u.kind === "neuron");
  if (!ctl.length) return null;
  const floor = Math.max(...ctl);
  return { above: neu.filter((u) => u.z > floor).length, chance: neu.length / (ctl.length + 1), neurons: neu.length, controls: ctl.length };
}

/** Where a census's doses and push unit come from, said wherever its answers or numbers are shown. */
export const CensusProvenance: React.FC<{ card: CensusCard | undefined; style?: React.CSSProperties }> = ({ card, style }) =>
  card ? (
    <p className="atlas-small atlas-prov" style={style}>
      <span className="red">Doses not yet calibrated on this model:</span> {card.doses_from ?? "the run does not say where its doses come from"}.
      {card.dose_unit === "ceiling"
        ? ` Each dose is a multiple of the pushed neuron’s write ceiling, the most it can write by itself (a control’s: of the neuron it is paired with), so one dose is a different push for each neuron; the push beside it is its share of the stream’s median length${card.p50_source ? `, read over ${card.p50_source}` : ""}.`
        : card.p50_source
          ? ` The push’s unit, the stream’s median length, read over ${card.p50_source}.`
          : ""}
      {card.null.above_all_by_chance != null
        ? ` ${card.null.above_all} of its ${card.null.neurons} neurons beat every one of its ${card.null.controls} controls; about ${card.null.above_all_by_chance.toFixed(0)} would by chance alone.`
        : " It has no random controls, so nothing in it is set against chance."}
      {card.mixed_runtime ? " Its parts were written on more than one runtime (card or library versions): each part’s readings stand, but their bits are not alike." : ""}
      {card.partial ? ` Joined from shards with ${card.partial.missing_answers} of its answers missing.` : ""}
      {card.stage
        ? ` A retest of the ${card.stage.top} strongest of the ${card.stage.of} neurons of ${card.stage.screen}, against fresh controls: its p and q are taken over these ${card.stage.top} alone.`
        : ""}
    </p>
  ) : null;

/** A neuron's name as lists show it, marked when the scientist wrote it (field notes' `name_by`): a guess, not a finding. */
export const NameMark: React.FC<{ layer: number; n: number; name: string }> = ({ layer, n, name }) => {
  const e = useAtlas((s) => s.notes?.neurons[noteKey(layer, n)]);
  const sci = !!e && (e.name_by ?? e.by) === "scientist";
  return (
    <b>
      {name}
      {sci ? (
        <span className="atlas-src" title="named by the scientist: a guess, not a finding">
          {" "}
          scientist
        </span>
      ) : null}{" "}
    </b>
  );
};

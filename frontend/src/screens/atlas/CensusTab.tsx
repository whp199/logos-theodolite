import React, { useState } from "react";
import { Rubric } from "../../components/kit";
import { href } from "../../router";
import type { CensusUnit, Rollout } from "./api";
import { CensusProvenance, fmtDose, overCeiling } from "./parts";
import { useAtlas } from "./state";

/**
 * The census's answers for the open neuron (after the explorer's Examples tab): each question as the model answers it
 * left alone, then pushed by each dose. Doses that several questions share get a column; a question's one-off doses
 * span its empty cells. A cell opens in full on a click; an answer cut off at the census's length is marked.
 */
export const CensusTab: React.FC = () => {
  const f = useAtlas((s) => s.focus);
  const unit = useAtlas((s) => s.unit);
  const [open, setOpen] = useState<Set<string>>(new Set());
  if (!f || f.kind !== "neuron") return null;
  if (!unit || unit.layer !== f.layer || unit.n !== f.n) return <p className="muted atlas-small atlas-pad">reading the census…</p>;
  if (!unit.rollouts?.length) {
    return (
      <p className="muted atlas-pad">
        This neuron has not been through the census: no answers were written with it pushed. The details and the push work all the same.
      </p>
    );
  }
  return <Grid unit={unit} open={open} toggle={(k) => setOpen((s) => new Set(s.has(k) ? [...s].filter((x) => x !== k) : [...s, k]))} />;
};

/** The census's doses against the most this neuron can ever write by itself (NeuronLens.ceiling): the run's smallest and
 * largest dose (main_rel's 0.5 and 0.8 for a run whose card names none). */
const CeilingNote: React.FC<{ unit?: "stream" | "ceiling"; doses?: number[] }> = ({ unit, doses }) => {
  const f = useAtlas((s) => s.focus);
  const lens = useAtlas((s) => s.lens);
  const c = f && lens && lens.layer === f.layer && lens.n === f.n ? lens.ceiling : null;
  const sizes = [...new Set((doses ?? []).map((d) => Math.abs(d)).filter((d) => d > 0 && d < 3))].sort((a, b) => a - b);
  const [lo, hi] = sizes.length ? [sizes[0], sizes[sizes.length - 1]] : [0.5, 0.8];
  const o5 = overCeiling(lo, c);
  const o8 = overCeiling(hi, c);
  if (unit === "ceiling")
    return (
      <p className="muted atlas-small" style={{ margin: "0 0 8px" }}>
        In this census a dose is a multiple of the most this neuron can ever write by itself: above 1, the answers say what a push this large
        does, not what the neuron does when it fires.
      </p>
    );
  if (o5 == null || o8 == null) return null;
  return (
    <p className="muted atlas-small" style={{ margin: "0 0 8px" }}>
      For this neuron a push of {lo} is {o5.toFixed(1)}× and {hi} is {o8.toFixed(1)}× the most it can ever write by itself: where that is above
      1, the answers say what a push this large does, not what the neuron does when it fires.
    </p>
  );
};

const Grid: React.FC<{ unit: CensusUnit; open: Set<string>; toggle: (k: string) => void }> = ({ unit, open, toggle }) => {
  const rolls = unit.rollouts ?? [];
  const prompts = Object.keys(unit.prompts ?? {}).filter((p) => rolls.some((r) => r.prompt === p));
  const count: Record<string, number> = {};
  for (const r of rolls) count[r.dose] = (count[r.dose] ?? 0) + 1;
  const doses = Object.keys(count)
    .map(Number)
    .filter((d) => count[d] > 1)
    .sort((a, b) => a - b);
  const relOf = (d: number) => rolls.find((r) => r.dose === d)?.rel_push.toFixed(2);
  const cell = (r: Rollout | undefined, key: string, extra?: React.ReactNode, span?: number) =>
    r ? (
      <div
        key={key}
        className={`atlas-cell${r.finish === "length" ? " capped" : ""}${open.has(key) ? " open" : ""}`}
        style={span ? { gridColumn: `span ${span}` } : undefined}
        onClick={() => toggle(key)}
        title={r.finish === "length" ? "cut off at the census's length" : undefined}
      >
        {extra}
        {r.text}
      </div>
    ) : (
      <div key={key} />
    );
  const out: React.ReactNode[] = [<div key="h" className="atlas-gh" />, <div key="h0" className="atlas-gh">0</div>];
  for (const d of doses)
    out.push(
      <div key={`hd${d}`} className={`atlas-gh ${d > 0 ? "up" : "down"}`}>
        {fmtDose(d)}
        <span className="muted"> · {relOf(d)}</span>
      </div>,
    );
  for (const p of prompts) {
    const b = unit.baselines?.find((r) => r.prompt === p);
    out.push(
      <div key={`p${p}`} className="atlas-gh" title={unit.prompts?.[p]}>
        {p}
      </div>,
    );
    out.push(cell(b, `${p}/base`, undefined));
    const mine = rolls.filter((r) => r.prompt === p);
    const shared = doses.map((d) => mine.find((r) => r.dose === d));
    const extra = mine.filter((r) => !doses.includes(r.dose));
    if (shared.some(Boolean)) shared.forEach((r, i) => out.push(cell(r, `${p}/${doses[i]}`)));
    else if (extra.length) {
      const span = Math.max(1, Math.floor(doses.length / extra.length));
      extra.forEach((r, i) =>
        out.push(
          cell(
            r,
            `${p}/${r.dose}`,
            <span className={`atlas-gh ${r.dose > 0 ? "up" : "down"}`}>
              {fmtDose(r.dose)} · {r.rel_push.toFixed(2)}{" "}
            </span>,
            i === extra.length - 1 ? Math.max(1, doses.length - span * (extra.length - 1)) : span,
          ),
        ),
      );
    } else doses.forEach((d) => out.push(<div key={`${p}/${d}`} />));
  }
  return (
    <div className="atlas-census">
      <Rubric style={{ fontSize: 15, margin: "0 0 8px" }}>
        Each row is one of the census's questions; the first column is the model left alone, the others with this neuron pushed along its
        own write direction by the dose shown (
        {unit.card?.dose_unit === "ceiling" ? "a multiple of the most it can write by itself" : "a share of the stream's usual length"}, its push
        measured beside it as a share of the stream). Click an answer to read all of it. <a href={href.lexicon("census")}>What the census is.</a>
      </Rubric>
      <CensusProvenance card={unit.card} style={{ margin: "0 0 8px" }} />
      <CeilingNote unit={unit.card?.dose_unit} doses={unit.card?.doses} />
      <div className="atlas-grid" style={{ gridTemplateColumns: `56px repeat(${1 + doses.length}, minmax(0, 1fr))` }}>
        {out}
      </div>
      <p className="muted atlas-small" style={{ marginTop: 8 }}>
        {unit.run} · {unit.design}
      </p>
    </div>
  );
};

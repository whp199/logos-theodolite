import React from "react";
import { CensusTab } from "./CensusTab";
import { DetailsTab } from "./DetailsTab";
import { PassagesTab } from "./PassagesTab";
import { PushTab } from "./PushTab";
import { Scientist } from "./Scientist";
import { useScientist } from "./scientist";
import { useAtlas, type RightTab } from "./state";

const LABEL: Record<RightTab, string> = { census: "census", push: "push", details: "details", passages: "passages", scientist: "scientist" };

/** The open neuron's panel (its census answers, the push, its details, its stored passages) and the scientist; for a
 * head, the scientist only (the head itself opens in the left panel). */
export const RightPanel: React.FC = () => {
  const open = useAtlas((s) => s.rightOpen);
  const tab = useAtlas((s) => s.rightTab);
  const f = useAtlas((s) => s.focus);
  const unit = useAtlas((s) => s.unit);
  const unseen = useScientist((s) => s.unseen);
  if (!open) return null;
  const neuron = f?.kind === "neuron";
  const known = neuron && unit && unit.layer === f.layer && unit.n === f.n;
  const hasCensus = known ? !!unit.rollouts?.length : neuron;
  const tabs: RightTab[] = neuron ? [...(hasCensus ? (["census"] as const) : []), "push", "details", "passages", "scientist"] : ["scientist"];
  const shown = tabs.includes(tab) ? tab : tabs[0];
  const pick = (t: RightTab) => {
    useAtlas.setState({ rightTab: t });
    if (t === "scientist") useScientist.setState({ unseen: false });
  };
  return (
    <aside className="atlas-panel atlas-right" aria-label="the open neuron and the scientist">
      <nav className="atlas-tabs" role="tablist">
        {tabs.map((t) => (
          <button key={t} type="button" role="tab" aria-selected={shown === t} onClick={() => pick(t)}>
            {LABEL[t]}
            {t === "scientist" && unseen && shown !== "scientist" ? <span className="atlas-dot" aria-label="a new reply" /> : null}
          </button>
        ))}
        <button type="button" className="atlas-edge" aria-label="hide this panel" onClick={() => useAtlas.setState({ rightOpen: false })}>
          ›
        </button>
      </nav>
      <div className={`atlas-pane${shown === "scientist" ? " flush" : ""}`}>
        {shown === "census" ? (
          <CensusTab />
        ) : shown === "push" ? (
          <PushTab />
        ) : shown === "details" ? (
          <DetailsTab />
        ) : shown === "passages" ? (
          <PassagesTab />
        ) : (
          <Scientist />
        )}
      </div>
    </aside>
  );
};

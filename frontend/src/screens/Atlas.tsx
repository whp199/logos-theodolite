/**
 * The atlas: every neuron of every layer at once (after lyramakesmusic/neuron-explorer, which maps Qwen3.5-4B so). The
 * map draws all 60 x 21,504 feed-forward neurons of Gemma-4-31B, each layer's attention heads before its sheet; open a
 * neuron to see its words (under the label gate), its web of weight links, its census answers, and push it; run a text
 * to see what fires; ask the scientist. Layers count from 0 in the state and the server's fields, from 1 on the page.
 *
 *   #/atlas              the atlas, as it was left in this browser
 *   #/atlas/54/15491     with L54 N15491 open (the pages' count)
 *   #/atlas/54/h3        with head 3 of layer 54 open
 */
import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import { ReadingThrough, Rubric, SiteFooter, SiteHeader } from "../components/kit";
import { href, type Route } from "../router";
import { LeftPanel } from "./atlas/LeftPanel";
import { AtlasMap } from "./atlas/map";
import { RightPanel } from "./atlas/RightPanel";
import { openEvents, useScientist } from "./atlas/scientist";
import { Spread } from "./atlas/Spread";
import {
  loadArch,
  loadNotes,
  loadSpread,
  openHead,
  openNeuron,
  refreshWeb,
  restoreFocus,
  setLayer,
  setLayout,
  setMap,
  setQuery,
  theMap,
  useAtlas,
} from "./atlas/state";
import { hideTip, onMapHover, Tooltip } from "./atlas/Tooltip";

type AtlasRoute = Extract<Route, { name: "atlas" }>;

const Seg: React.FC<{ items: [string, string][]; value: string; onPick: (v: string) => void; label: string }> = ({ items, value, onPick, label }) => (
  <span className="atlas-seg" role="group" aria-label={label}>
    {items.map(([v, text]) => (
      <button key={v} type="button" aria-pressed={value === v} onClick={() => onPick(v)}>
        {text}
      </button>
    ))}
  </span>
);

/** "L54 N15491", "54/15491", "L54 h3": a neuron or a head in the pages' count. */
function parseJump(s: string, L: number, N: number, heads: (l: number) => number): { kind: "neuron" | "head"; layer: number; n: number } | string {
  const h = s.match(/^\s*L?\s*(\d+)\s*[\s/·,]\s*h\s*(\d+)\s*$/i);
  const m = s.match(/^\s*L?\s*(\d+)\s*[\s/·,]\s*N?\s*(\d+)\s*$/i);
  const at = h ?? m;
  if (!at) return "Write a neuron as L54 N15491, or a head as L54 h3.";
  const layer = +at[1] - 1;
  const n = +at[2];
  if (layer < 0 || layer >= L) return `Layers run from 1 to ${L}.`;
  if (h) return n < heads(layer) ? { kind: "head", layer, n } : `Layer ${layer + 1} has ${heads(layer)} heads (0 to ${heads(layer) - 1}).`;
  return n < N ? { kind: "neuron", layer, n } : `Each layer has ${N.toLocaleString()} neurons (0 to ${N - 1}).`;
}

const Controls: React.FC = () => {
  const arch = useAtlas((s) => s.arch)!;
  const layer = useAtlas((s) => s.layer);
  const view = useAtlas((s) => s.view);
  const layoutKind = useAtlas((s) => s.layoutKind);
  const webOn = useAtlas((s) => s.webOn);
  const webFirst = useAtlas((s) => s.webFirst);
  const qMode = useAtlas((s) => s.qMode);
  const query = useAtlas((s) => s.query);
  const [jump, setJump] = useState("");
  const [jumpError, setJumpError] = useState<string | null>(null);
  return (
    <div className="atlas-bar" role="toolbar" aria-label="the atlas's controls">
      <Seg
        label="view"
        items={[
          ["map", "map"],
          ["spread", "spread"],
        ]}
        value={view}
        onPick={(v) => {
          useAtlas.setState({ view: v as "map" | "spread" });
          hideTip();
          if (v === "spread") void loadSpread();
        }}
      />
      <select
        className="atlas-select"
        aria-label="layer"
        value={layer}
        onChange={(e) => {
          const l = +e.target.value;
          void setLayer(l);
          if (useAtlas.getState().view === "map") theMap()?.enterLayer(l);
        }}
      >
        {Array.from({ length: arch.n_layers }, (_, l) => (
          <option key={l} value={l}>
            layer {l + 1}
            {arch.layers[l]?.kind === "full" ? " · full" : ""}
            {arch.census[String(l)] ? " · census" : ""}
          </option>
        ))}
      </select>
      <Seg
        label="how the neurons lie on each sheet"
        items={[
          ["index", "index"],
          ["writes", "writes"],
        ]}
        value={layoutKind}
        onPick={(v) => void setLayout(v as "index" | "writes")}
      />
      <span className="atlas-webctl">
        <Seg
          label="the open neuron's web"
          items={[
            ["off", "no web"],
            ["on", "web"],
          ]}
          value={webOn ? "on" : "off"}
          onPick={(v) => {
            useAtlas.setState({ webOn: v === "on" });
            void refreshWeb();
          }}
        />
        <input
          className="input-num atlas-first"
          inputMode="numeric"
          aria-label="the web: the open neuron's strongest links each way"
          title="the web: the open neuron's strongest links each way (later hops follow two)"
          value={webFirst}
          onChange={(e) => {
            const v = Math.max(1, Math.min(24, Math.round(Number(e.target.value) || 1)));
            useAtlas.setState({ webFirst: v });
            void refreshWeb();
          }}
        />
      </span>
      <span className="atlas-jump">
        <input
          className="input-line"
          placeholder="L54 N15491"
          aria-label="go to a neuron or a head"
          aria-invalid={jumpError ? true : undefined}
          title={jumpError ?? "go to a neuron (L54 N15491) or a head (L54 h3)"}
          spellCheck={false}
          value={jump}
          onChange={(e) => {
            setJump(e.target.value);
            setJumpError(null);
          }}
          onKeyDown={(e) => {
            if (e.key !== "Enter") return;
            const r = parseJump(jump, arch.n_layers, arch.d_ff, (l) => arch.layers[l]?.n_heads ?? 0);
            if (typeof r === "string") return setJumpError(r);
            if (r.kind === "head") void openHead(r.layer, r.n);
            else void openNeuron(r.layer, r.n, { fly: true });
          }}
        />
      </span>
      <span className="atlas-search">
        <Seg
          label="what the box searches"
          items={[
            ["lens", "words"],
            ["fires", "fires"],
          ]}
          value={qMode}
          onPick={(v) => setQuery(query, v as "lens" | "fires")}
        />
        <input
          className="input-line"
          placeholder={qMode === "fires" ? "a text: what fires on it" : "search the neurons' words"}
          aria-label={qMode === "fires" ? "a text to run" : "search the neurons' words"}
          spellCheck={false}
          autoComplete="off"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </span>
      {jumpError ? <span className="error-line atlas-small">{jumpError}</span> : null}
    </div>
  );
};

/** The map's canvas, made once the model's shape is known; the panels' widths are kept out of its centre. */
const MapStage: React.FC<{ deep: AtlasRoute; insets: { left: number; right: number } }> = ({ deep, insets }) => {
  const arch = useAtlas((s) => s.arch);
  const layerMode = useAtlas((s) => s.layerMode);
  const view = useAtlas((s) => s.view);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const notesRef = useRef<HTMLDivElement>(null);
  const [glError, setGlError] = useState<string | null>(null);
  useEffect(() => {
    if (!arch || !canvasRef.current || !notesRef.current) return;
    let m: AtlasMap;
    try {
      m = new AtlasMap(canvasRef.current, notesRef.current, arch, {
        onPick: (l, n) => void openNeuron(l, n, { fly: false }),
        onPickHead: (l, h) => void openHead(l, h),
        onPickLayer: (l) => {
          if (l !== useAtlas.getState().layer) void setLayer(l);
        },
        onLayerMode: (l) => {
          useAtlas.setState({ layerMode: l });
          if (l !== null && l !== useAtlas.getState().layer) void setLayer(l);
        },
        onHover: onMapHover,
      });
    } catch (e) {
      setGlError(`The map cannot be drawn in this browser (${(e as Error).message || "WebGL is not available"}).`);
      return;
    }
    setMap(m);
    const s = useAtlas.getState();
    void setLayer(s.layer);
    void setLayout(s.layoutKind);
    void loadNotes();
    if (s.focus) {
      // the page came back with something open: show it on the new map
      const f = s.focus;
      if (f.kind === "head") void openHead(f.layer, f.n);
      else void openNeuron(f.layer, f.n, { fly: false });
    } else if (deep.layerNo == null) restoreFocus();
    return () => {
      setMap(null);
      m.dispose();
      hideTip();
    };
  }, [arch]); // eslint-disable-line react-hooks/exhaustive-deps
  // a link names what to open: open it, then forget it (later picks are kept in this browser, not in the address bar)
  useEffect(() => {
    if (!arch || deep.layerNo == null) return;
    const l = deep.layerNo - 1;
    if (l < 0 || l >= arch.n_layers) return;
    hideTip();
    if (deep.head != null && deep.head < (arch.layers[l]?.n_heads ?? 0)) void openHead(l, deep.head);
    else if (deep.neuron != null && deep.neuron < arch.d_ff) void openNeuron(l, deep.neuron, { fly: true });
    try {
      window.history.replaceState(null, "", "#/atlas");
    } catch {
      // the link stays in the address bar
    }
  }, [arch, deep.layerNo, deep.neuron, deep.head]);
  useEffect(() => theMap()?.setInsets(insets.left, insets.right), [insets.left, insets.right, arch]);
  return (
    <div className="atlas-mapview" style={{ display: view === "map" ? undefined : "none" }}>
      <canvas ref={canvasRef} className="atlas-canvas" aria-label="the map: every neuron of every layer" />
      <div ref={notesRef} className="atlas-notes" aria-hidden="true" />
      {layerMode !== null ? (
        <button type="button" className="atlas-back btn-ghost plain" style={{ left: insets.left + 12 }} onClick={() => theMap()?.exitLayer()}>
          ‹ all layers · layer {layerMode + 1}
        </button>
      ) : null}
      {glError ? (
        <div className="atlas-slot">
          <p className="error-line">{glError}</p>
        </div>
      ) : null}
    </div>
  );
};

export const Atlas: React.FC<{ route: AtlasRoute }> = ({ route }) => {
  const arch = useAtlas((s) => s.arch);
  const archError = useAtlas((s) => s.archError);
  const view = useAtlas((s) => s.view);
  const leftHidden = useAtlas((s) => s.leftHidden);
  const rightOpen = useAtlas((s) => s.rightOpen);
  const rightTab = useAtlas((s) => s.rightTab);
  const f = useAtlas((s) => s.focus);
  const unseen = useScientist((s) => s.unseen);
  const stageRef = useRef<HTMLDivElement>(null);
  const [insets, setInsets] = useState({ left: 0, right: 0 });

  useEffect(() => {
    if (!useAtlas.getState().arch) void loadArch();
    // the scientist's chat reaches this page while its tab is closed, and stays open while another page is shown, so a
    // reply that ends meanwhile is not lost
    void openEvents();
  }, []);

  // the panels float over the map: keep what is drawn centred in the space between them
  useLayoutEffect(() => {
    const measure = () => {
      const st = stageRef.current;
      if (!st) return;
      const w = (sel: string) => {
        const el = st.querySelector(sel) as HTMLElement | null;
        return el ? el.getBoundingClientRect().width + 24 : 0;
      };
      setInsets((cur) => {
        const next = { left: w(".atlas-left"), right: w(".atlas-right") };
        return cur.left === next.left && cur.right === next.right ? cur : next;
      });
    };
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [leftHidden, rightOpen, rightTab, f, arch, view]);

  return (
    <div className="atlas-page">
      <div className="atlas-head">
        <SiteHeader route={route} />
        {arch ? <Controls /> : null}
      </div>
      <main id="main" className={`atlas-stage${rightOpen ? " right-open" : ""}`} ref={stageRef}>
        {archError ? (
          <div className="atlas-slot">
            <div className="stack" style={{ maxWidth: 560 }}>
              <p className="error-line">The atlas could not be opened: {archError}</p>
              <Rubric>
                The atlas reads the model the server has loaded. Start the server (see the README), then reload this page; a stand-in model
                (scripts/atlas_stand_in.py --serve) lets you try the page without the 31B weights.
              </Rubric>
            </div>
          </div>
        ) : !arch ? (
          <div className="atlas-slot">
            <p className="status-line">Opening the atlas…</p>
          </div>
        ) : (
          <>
            <MapStage deep={route} insets={insets} />
            {view === "spread" ? <Spread insets={insets} /> : null}
            <LeftPanel />
            <RightPanel />
            {leftHidden ? (
              <button type="button" className="atlas-edgetab left" aria-label="show the left panel" onClick={() => useAtlas.setState({ leftHidden: false })}>
                ›
              </button>
            ) : null}
            {!rightOpen ? (
              <button
                type="button"
                className="atlas-edgetab right"
                onClick={() =>
                  useAtlas.setState(f && f.kind === "neuron" ? { rightOpen: true } : { rightOpen: true, rightTab: "scientist" })
                }
              >
                ‹ {f && f.kind === "neuron" ? `N${f.n}` : "scientist"}
                {unseen ? <span className="atlas-dot" aria-label="a new reply" /> : null}
              </button>
            ) : null}
            {arch.stand_in ? (
              <p className="atlas-standin" style={{ left: insets.left + 12 }}>
                a stand-in model ({arch.n_layers} layers of {arch.d_ff} neurons): for trying the page, not for findings
              </p>
            ) : null}
          </>
        )}
      </main>
      <div className="atlas-foot">
        <SiteFooter folio="the atlas">
          <ReadingThrough /> · its words held back below layer 41, and where the naked eye shares none of their first ten · the steer
          lens and the census are pushes, not labels · <a href={href.lineage()}>the lineage of these ideas</a>
        </SiteFooter>
      </div>
      <Tooltip />
    </div>
  );
};

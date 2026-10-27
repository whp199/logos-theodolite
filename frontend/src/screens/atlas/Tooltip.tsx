/**
 * The tooltip over the map and the spread (after the explorer's #tip): what a hovered neuron's words are, from the
 * indexes alone (GET /api/atlas/hover, no model run), what a hovered head or layer is.
 */
import React, { useLayoutEffect, useRef } from "react";
import { create } from "zustand";
import { get, type HeadInfo, type Hover, type NeuronHeads } from "./api";
import type { Pick } from "./map";
import { FaceLine, NameMark } from "./parts";
import { headRoleLinks, hoverCache, remember, roleHeads, theMap, useAtlas } from "./state";

interface Tip {
  x: number;
  y: number;
  head: string;
  body: React.ReactNode;
}
const useTip = create<{ tip: Tip | null }>(() => ({ tip: null }));

export const showTip = (x: number, y: number, head: string, body: React.ReactNode) => useTip.setState({ tip: { x, y, head, body } });
export const hideTip = () => useTip.setState({ tip: null });

export const Tooltip: React.FC = () => {
  const tip = useTip((s) => s.tip);
  const ref = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || !tip) return;
    const r = el.getBoundingClientRect();
    el.style.left = `${Math.max(8, Math.min(tip.x + 14, window.innerWidth - r.width - 8))}px`;
    el.style.top = `${Math.max(8, Math.min(tip.y + 14, window.innerHeight - r.height - 8))}px`;
  });
  if (!tip) return null;
  return (
    <div ref={ref} className="atlas-tip" role="tooltip">
      <div className="atlas-lab">{tip.head}</div>
      {tip.body}
    </div>
  );
};

/* ------------------------------------------------------------------------------------------------ the map's hover */

let hoverKey: string | null = null;
let hoverTimer: number | undefined;

const busyFocus = () => !!useAtlas.getState().focus;

function neuronBody(layer: number, n: number) {
  const s = useAtlas.getState();
  const us = s.units[layer];
  const u = us?.find((x) => x.kind === "neuron" && x.n === n);
  // census words describe a neuron only when its pushes beat every random control's (the map's gilt)
  const ctl = (us ?? []).filter((x) => x.kind === "random").map((x) => x.z);
  const censusShown = !!u && ctl.length > 0 && u.z > Math.max(...ctl);
  const h = hoverCache.get(`${layer}/${n}`) as Hover | undefined;
  const nh = hoverCache.get(`nh/${layer}/${n}`) as NeuronHeads | undefined;
  const name = h?.name || u?.name;
  const top = nh ? roleHeads(nh) : [];
  const into = top.filter((k) => k.role === "o").slice(0, 3);
  const from = top.filter((k) => k.role !== "o").slice(0, 4);
  return (
    <>
      {name ? (
        <div className="atlas-tname">
          <NameMark layer={layer} n={n} name={name} />
        </div>
      ) : null}
      {u && censusShown ? (
        <div className="atlas-twords">
          <span className="atlas-src">census</span> {u.words.join("  ")}
        </div>
      ) : null}
      {h ? (
        <>
          <div className="atlas-tl">
            <FaceLine layer={layer} face="+" w={h.faces["+"]} k={6} />
          </div>
          <div className="atlas-tl">
            <FaceLine layer={layer} face="-" w={h.faces["-"]} k={6} />
          </div>
        </>
      ) : (
        <div className="muted">…</div>
      )}
      {h && (h.faces["+"].lens?.length || h.faces["-"].lens?.length) ? (
        <div className="atlas-tl muted atlas-small">lens: J++, fitted on the base model and read here on the chat model</div>
      ) : null}
      {into.length ? <div className="atlas-tl muted">← {into.map((k) => `L${k.layer + 1} h${k.head}`).join("  ")}</div> : null}
      {from.length ? <div className="atlas-tl muted">→ {from.map((k) => `${k.role} L${k.layer + 1} h${k.head}`).join("  ")}</div> : null}
    </>
  );
}

function headBody(layer: number, head: number) {
  const h = hoverCache.get(`h/${layer}/${head}`) as HeadInfo | undefined;
  if (!h) return <div className="muted">…</div>;
  const side = (sg: string, list: HeadInfo["q"]) =>
    list.slice(0, 3).map((c) => (
      <div key={`${sg}${c.layer}/${c.n}`} className="atlas-tl">
        <span className="atlas-sgn">{sg}</span>
        <span className="muted">
          L{c.layer + 1} N{c.n}
        </span>{" "}
        <FaceLine layer={c.layer} face="+" w={c.words} k={4} />
      </div>
    ));
  return (
    <>
      {side("←", h.v)}
      {side("→", h.o)}
    </>
  );
}

/** The map's hover callback: the tooltip at once, its words a moment later; with nothing open, the hovered neuron's or
 * head's links are drawn too. */
export function onMapHover(p: Pick | null, x: number, y: number) {
  if (!p) {
    hideTip();
    if (hoverKey) {
      hoverKey = null;
      window.clearTimeout(hoverTimer);
      theMap()?.setHoverConnections(null);
    }
    return;
  }
  const arch = useAtlas.getState().arch;
  if (p.type === "block") {
    if (hoverKey) {
      hoverKey = null;
      theMap()?.setHoverConnections(null);
    }
    const s = arch?.layers[p.layer];
    return showTip(
      x,
      y,
      `layer ${p.layer + 1}`,
      <div>
        {s
          ? s.kind === "full"
            ? `full attention · ${s.n_heads} query heads over ${s.n_kv} key heads, its values read from its keys`
            : `sliding attention over the last ${s.window ?? "?"} word-pieces · ${s.n_heads} query heads over ${s.n_kv} key-value heads`
          : ""}
      </div>,
    );
  }
  if (p.type === "head") {
    const key = `h/${p.layer}/${p.head}`;
    const title = `L${p.layer + 1} h${p.head} · ${p.full ? "full" : "sliding"} attention`;
    if (hoverKey !== key) theMap()?.setHoverConnections(null);
    showTip(x, y, title, headBody(p.layer, p.head));
    window.clearTimeout(hoverTimer);
    hoverKey = key;
    hoverTimer = window.setTimeout(async () => {
      try {
        if (!hoverCache.has(key)) remember(key, await get<HeadInfo>("/api/atlas/head", { layer: p.layer, head: p.head }));
      } catch {
        return;
      }
      if (hoverKey !== key) return;
      if (!busyFocus()) theMap()?.setHoverHead(p.layer, p.head, headRoleLinks(hoverCache.get(key) as HeadInfo));
      showTip(x, y, title, headBody(p.layer, p.head));
    }, 250);
    return;
  }
  const key = `${p.layer}/${p.n}`;
  const u = useAtlas.getState().units[p.layer]?.find((v) => v.kind === "neuron" && v.n === p.n);
  const title = `L${p.layer + 1} N${p.n}${u ? `  · mass ${u.z.toFixed(1)}` : ""}`;
  showTip(x, y, title, neuronBody(p.layer, p.n));
  window.clearTimeout(hoverTimer);
  hoverKey = key;
  hoverTimer = window.setTimeout(async () => {
    try {
      if (!hoverCache.has(key)) remember(key, await get<Hover>("/api/atlas/hover", { layer: p.layer, n: p.n }));
    } catch {
      return;
    }
    if (hoverKey !== key) return;
    const h = hoverCache.get(key) as Hover;
    // with a neuron or head open, hovering only shows the tooltip: its links and web stay on screen
    if (!busyFocus() && h.links) theMap()?.setHoverConnections({ layer: p.layer, n: p.n }, h.links);
    showTip(x, y, title, neuronBody(p.layer, p.n));
  }, 120);
}

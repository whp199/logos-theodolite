/**
 * The spread (after the explorer's embeddings view): every answer of the open layer's census as a point, laid out so
 * that answers with like words lie together (backend/census.py, build_spread). Colour by neuron, question, dose or kind
 * (a neuron, a random direction, the model left alone). Hover an answer to read it; click one to open its neuron.
 * Drag to move, wheel to zoom.
 */
import React, { useEffect, useRef } from "react";
import { Rubric } from "../../components/kit";
import { get, type Spread as SpreadData } from "./api";
import { fmtDose } from "./parts";
import { goToNeuron, useAtlas } from "./state";
import { hideTip, showTip } from "./Tooltip";

const PROMPT_INK: Record<string, string> = { who: "#8f2b1e", snake: "#3b6e5a", orange: "#a07a2e", free: "#2e4a7d" };
const KIND_INK: Record<string, string> = { neuron: "#a89a82", random: "#2e4a7d", baseline: "#2b2117" };

const hueOf = (n: number) => `hsl(${(n * 2654435761) % 360}, 38%, 42%)`;
function doseInk(d: number) {
  if (d === 0) return "#2b2117";
  const t = Math.min(1, Math.abs(d));
  return d > 0 ? `rgba(160, 122, 46, ${0.35 + 0.65 * t})` : `rgba(46, 74, 125, ${0.35 + 0.65 * t})`;
}

type Data = SpreadData & { layer: number };

class ScatterView {
  private ctx: CanvasRenderingContext2D;
  private data: Data | null = null;
  private colorBy: string = "neuron";
  private sel: number | null = null;
  private hover = -1;
  private view = { x: 0, y: 0, k: 1 };
  private dpr = Math.min(window.devicePixelRatio || 1, 2);
  private ro: ResizeObserver;
  private drag: [number, number, number, number] | null = null;
  private moved = false;
  private insets = { left: 0, right: 0 };

  constructor(
    private canvas: HTMLCanvasElement,
    private cb: { onPick: (i: number) => void; onHover: (i: number, x: number, y: number) => void },
  ) {
    this.ctx = canvas.getContext("2d")!;
    this.ro = new ResizeObserver(() => this.size());
    this.ro.observe(canvas);
    canvas.addEventListener("pointerdown", this.onDown);
    window.addEventListener("pointerup", this.onUp);
    canvas.addEventListener("pointermove", this.onMove);
    canvas.addEventListener("pointerleave", this.onLeave);
    canvas.addEventListener("wheel", this.onWheel, { passive: false });
  }

  private size() {
    this.canvas.width = this.canvas.clientWidth * this.dpr;
    this.canvas.height = this.canvas.clientHeight * this.dpr;
    this.draw();
  }

  private toScreen(i: number): [number, number] {
    const d = this.data!;
    const pad = 48;
    const w = this.canvas.clientWidth - 2 * pad - this.insets.left - this.insets.right;
    const h = this.canvas.clientHeight - 2 * pad;
    const s = Math.max(50, Math.min(w, h));
    const ox = this.insets.left + pad + (w - s) / 2;
    const oy = pad + (h - s) / 2;
    return [(ox + d.x![i] * s) * this.view.k + this.view.x, (oy + (1 - d.y![i]) * s) * this.view.k + this.view.y];
  }

  private color(i: number) {
    const d = this.data!;
    const kind = d.kind?.[i] ?? "neuron";
    if (kind === "baseline") return KIND_INK.baseline;
    if (this.colorBy === "prompt") return PROMPT_INK[d.prompt?.[i] ?? ""] ?? "#6b5d4d";
    if (this.colorBy === "dose") return doseInk(d.dose?.[i] ?? 0);
    if (this.colorBy === "kind") return KIND_INK[kind] ?? "#6b5d4d";
    return kind === "random" ? "#c9beac" : hueOf(d.neuron?.[i] ?? 0);
  }

  private isSel(i: number) {
    const d = this.data!;
    return this.sel !== null && d.kind?.[i] === "neuron" && d.neuron?.[i] === this.sel;
  }

  draw() {
    const { ctx } = this;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.fillStyle = "#f6f0e3";
    ctx.fillRect(0, 0, this.canvas.clientWidth, this.canvas.clientHeight);
    const d = this.data;
    if (!d?.x) return;
    const r = Math.max(2, Math.min(5, 2.6 * Math.sqrt(this.view.k)));
    const n = d.x.length;
    ctx.globalAlpha = this.sel !== null ? 0.3 : 0.85;
    for (let i = 0; i < n; i++) {
      if (this.isSel(i)) continue;
      const [x, y] = this.toScreen(i);
      ctx.fillStyle = this.color(i);
      ctx.fillRect(x - r / 2, y - r / 2, r, r);
    }
    if (this.sel !== null) {
      ctx.globalAlpha = 1;
      for (let i = 0; i < n; i++)
        if (this.isSel(i)) {
          const [x, y] = this.toScreen(i);
          ctx.fillStyle = doseInk(d.dose?.[i] ?? 0);
          ctx.beginPath();
          ctx.arc(x, y, r * 1.6, 0, 7);
          ctx.fill();
        }
    }
    if (this.hover >= 0) {
      const [x, y] = this.toScreen(this.hover);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = "#2b2117";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(x, y, r * 2.2, 0, 7);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  private nearest(mx: number, my: number) {
    let best = -1;
    let bd = 100;
    for (let i = 0; i < this.data!.x!.length; i++) {
      const [x, y] = this.toScreen(i);
      const dd = (x - mx) ** 2 + (y - my) ** 2;
      if (dd < bd) {
        bd = dd;
        best = i;
      }
    }
    return best;
  }

  private onDown = (e: PointerEvent) => {
    this.drag = [e.clientX, e.clientY, this.view.x, this.view.y];
    this.moved = false;
  };
  private onUp = () => {
    if (this.drag && !this.moved && this.data && this.hover >= 0) this.cb.onPick(this.hover);
    this.drag = null;
  };
  private onMove = (e: PointerEvent) => {
    if (!this.data?.x) return;
    if (this.drag) {
      const dx = e.clientX - this.drag[0];
      const dy = e.clientY - this.drag[1];
      if (Math.hypot(dx, dy) > 3) this.moved = true;
      if (this.moved) {
        this.view.x = this.drag[2] + dx;
        this.view.y = this.drag[3] + dy;
        this.draw();
        return;
      }
    }
    const r = this.canvas.getBoundingClientRect();
    const i = this.nearest(e.clientX - r.left, e.clientY - r.top);
    if (i !== this.hover) {
      this.hover = i;
      this.draw();
    }
    this.cb.onHover(i, e.clientX, e.clientY);
  };
  private onLeave = () => {
    this.hover = -1;
    this.draw();
    this.cb.onHover(-1, 0, 0);
  };
  private onWheel = (e: WheelEvent) => {
    e.preventDefault();
    const r = this.canvas.getBoundingClientRect();
    const mx = e.clientX - r.left;
    const my = e.clientY - r.top;
    const k = Math.max(0.5, Math.min(40, this.view.k * Math.exp(-e.deltaY * 0.0015)));
    const ff = k / this.view.k;
    this.view = { x: mx - (mx - this.view.x) * ff, y: my - (my - this.view.y) * ff, k };
    this.draw();
  };

  setData(d: Data | null) {
    this.data = d;
    this.view = { x: 0, y: 0, k: 1 };
    this.hover = -1;
    this.draw();
  }
  get current() {
    return this.data;
  }
  setColorBy(c: string) {
    this.colorBy = c;
    this.draw();
  }
  select(n: number | null) {
    this.sel = n;
    this.draw();
  }
  setInsets(left: number, right: number) {
    this.insets = { left, right };
    this.draw();
  }
  dispose() {
    this.ro.disconnect();
    window.removeEventListener("pointerup", this.onUp);
    this.canvas.removeEventListener("pointerdown", this.onDown);
    this.canvas.removeEventListener("pointermove", this.onMove);
    this.canvas.removeEventListener("pointerleave", this.onLeave);
    this.canvas.removeEventListener("wheel", this.onWheel);
  }
}

const rollouts = new Map<string, string>();

export const Spread: React.FC<{ insets: { left: number; right: number } }> = ({ insets }) => {
  const ref = useRef<HTMLCanvasElement>(null);
  const view = useRef<ScatterView | null>(null);
  const spread = useAtlas((s) => s.spread);
  const layer = useAtlas((s) => s.layer);
  const colorBy = useAtlas((s) => s.colorBy);
  const focus = useAtlas((s) => s.focus);
  const census = useAtlas((s) => !!s.arch?.census[String(s.layer)]);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    let hovered: string | null = null;
    const v = new ScatterView(c, {
      onPick: (i) => {
        const d = v.current;
        if (d?.kind?.[i] === "neuron" && d.neuron) goToNeuron(d.layer, d.neuron[i]);
      },
      onHover: (i, x, y) => {
        const d = v.current;
        if (i < 0 || !d) {
          hovered = null;
          return hideTip();
        }
        const kind = d.kind?.[i] ?? "neuron";
        const head = `${kind === "baseline" ? "left alone" : `${kind === "random" ? "random " : "N"}${d.neuron?.[i]}`} · ${d.prompt?.[i]} · ${fmtDose(d.dose?.[i] ?? 0)}`;
        const key = `${d.layer}/${d.row?.[i]}`;
        hovered = key;
        if (rollouts.has(key)) return showTip(x, y, head, <div className="atlas-tiptext">{rollouts.get(key)}</div>);
        showTip(x, y, head, <div className="muted">…</div>);
        void get<{ text: string }>("/api/atlas/census/rollout", { layer: d.layer, i: d.row?.[i] }).then(
          (r) => {
            rollouts.set(key, r.text.slice(0, 500));
            if (hovered === key) showTip(x, y, head, <div className="atlas-tiptext">{rollouts.get(key)}</div>);
          },
          () => undefined,
        );
      },
    });
    view.current = v;
    return () => {
      v.dispose();
      view.current = null;
      hideTip();
    };
  }, []);
  useEffect(() => view.current?.setData(spread && spread.layer === layer ? spread : null), [spread, layer]);
  useEffect(() => view.current?.setColorBy(colorBy), [colorBy]);
  useEffect(() => view.current?.select(focus && focus.kind === "neuron" && focus.layer === layer ? focus.n : null), [focus, layer, spread]);
  useEffect(() => view.current?.setInsets(insets.left, insets.right), [insets.left, insets.right]);
  return (
    <div className="atlas-spread">
      <canvas ref={ref} className="atlas-canvas" />
      {!spread || spread.layer !== layer ? (
        <div className="atlas-slot">
          <p className="muted">
            {census
              ? `Layer ${layer + 1}'s census has no spread yet (scripts/census_tools.py spread ${layer}, counting layers from 0).`
              : `Layer ${layer + 1} has no census yet, so there are no answers to lay out.`}
          </p>
        </div>
      ) : null}
      <div className="atlas-spreadbar" style={{ left: insets.left + 12, right: insets.right + 12 }}>
        <span className="atlas-seg">
          {(["neuron", "prompt", "dose", "kind"] as const).map((c) => (
            <button key={c} type="button" aria-pressed={colorBy === c} onClick={() => useAtlas.setState({ colorBy: c })}>
              {c === "prompt" ? "question" : c}
            </button>
          ))}
        </span>
        {spread && spread.layer === layer ? (
          <span className="muted atlas-small">
            {spread.x?.length} answers · {spread.embedder} · {spread.layout}
          </span>
        ) : null}
      </div>
      <Rubric style={{ position: "absolute", bottom: 10, left: insets.left + 16, maxWidth: 420, fontSize: 14 }}>
        Each point is one answer the census wrote; answers with like words lie together. Hover one to read it.
      </Rubric>
    </div>
  );
};

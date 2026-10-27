/**
 * The atlas's map (after lyramakesmusic/neuron-explorer, explorer/static/map3d.js), drawn in the instrument's inks: every
 * neuron of every layer as a dot on its layer's sheet, the layers in a row along x, each with its attention block (a
 * slab, its heads as lines on its face) before its feed-forward sheet. One draw call for all of Gemma-4-31B's 60 x 21,504
 * neurons. Picking does not raycast the dots: each sheet is a plane, so the pointer's ray meets each layer's sheet in one
 * cell of its grid (the layout's cell, or the neuron's index), and the frontmost visible layer whose neuron there lies
 * within a few pixels of the pointer wins.
 *
 * Layers count from 0 here, as the server's fields do; the labels show the pages' count (layer + 1).
 */
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import type { Arch, CensusUnitRow, HeadLinkNeuron, Link } from "./api";

const SHEET = 4.2;
const GAP = 1.9;
const SHEET_Y0 = 0.7;
const ATTN_DX = -0.62;
const MLP_DX = 0.18;

const C = (hex: string) => new THREE.Color(hex);
export const INKS = {
  bg: C("#f6f0e3"),
  idle: C("#a89a82"),
  swept: C("#8b7d66"),
  lo: C("#c9a85f"),
  hi: C("#6a4b14"),
  named: C("#8f2b1e"),
  ink: C("#2b2117"),
  up: C("#a07a2e"),
  down: C("#2e4a7d"),
  hair: C("#c9beac"),
  gilt: C("#c9a85f"),
  q: C("#3b6e5a"),
  k: C("#6a4c93"),
  v: C("#7a5c28"),
};
const ROLE: Record<string, THREE.Color> = { q: INKS.q, k: INKS.k, v: INKS.v, o: INKS.v };

/** A colour faded toward the vellum by `a` (1: itself, 0: the page). */
const toward = (c: THREE.Color, a: number) => INKS.bg.clone().lerp(c, Math.max(0, Math.min(1, a)));

function dotMaterial(sizePx: number, maxScale: number, alpha: number) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  return new THREE.RawShaderMaterial({
    uniforms: {
      size: { value: sizePx * dpr },
      ref: { value: 1 },
      maxScale: { value: maxScale },
      focusX: { value: 0 },
      fadeW: { value: 1 },
      fadeAmt: { value: 0 },
      camSide: { value: 1 },
      alpha: { value: alpha },
    },
    vertexShader: `precision highp float;
      uniform mat4 modelViewMatrix, projectionMatrix;
      uniform float size, ref, maxScale, focusX, fadeW, fadeAmt, camSide, alpha;
      attribute vec3 position; attribute vec3 color; varying vec4 vC;
      void main() {
        float d = (position.x - focusX) / fadeW;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mv;
        // layers away from the focus fade as the camera closes in; those between the camera and the focus fade further
        float front = step(0.45, (position.x - focusX) * camSide / ${GAP.toFixed(3)});
        float keep = mix(1.0, exp(-d * d), fadeAmt * 0.85) * mix(1.0, 1.0 - 0.7 * fadeAmt, front);
        float sc = clamp(sqrt(ref / -mv.z), 0.6, maxScale);
        vC = vec4(color, alpha * keep);
        gl_PointSize = size * sc;
      }`,
    fragmentShader: `precision highp float; varying vec4 vC;
      void main() { float r = length(gl_PointCoord - 0.5); if (r > 0.5) discard;
        gl_FragColor = vec4(vC.rgb, vC.a * (1.0 - smoothstep(0.36, 0.5, r))); }`,
    transparent: true,
    depthWrite: false,
    depthTest: false,
  });
}

export type Pick =
  | { type: "neuron"; layer: number; n: number }
  | { type: "head"; layer: number; head: number; full: boolean }
  | { type: "block"; layer: number; full: boolean };

export interface MapCallbacks {
  onPick: (layer: number, n: number) => void;
  onPickHead: (layer: number, head: number) => void;
  onPickLayer: (layer: number) => void;
  onLayerMode: (layer: number | null) => void;
  onHover: (p: Pick | null, x: number, y: number) => void;
}

export interface Mark {
  layer: number;
  n: number;
  sign: "+" | "-";
  w?: number;
}
export interface MapNote {
  layer: number;
  n: number;
  text: string;
}
type Linked = (Link | HeadLinkNeuron) & { side?: "up" | "down"; role?: string };

export class AtlasMap {
  readonly L: number;
  readonly N: number;
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private side: number;
  private cells: Int16Array | null = null;
  private cellToN: Int32Array;
  private pos: Float32Array;
  private col: Float32Array;
  private geo = new THREE.BufferGeometry();
  private pointMat: THREE.RawShaderMaterial;
  private points: THREE.Points;
  private slabs: THREE.Mesh[] = [];
  private labels: THREE.Mesh[] = [];
  private headPos: number[] = [];
  private headBase: THREE.Color[] = [];
  private headStart: number[] = [];
  private headGeo = new THREE.BufferGeometry();
  private headCol: Float32Array;
  private fadeLines: THREE.LineBasicMaterial[] = [];
  private headLevels: (number[] | undefined)[] | null = null;
  private headSel: { layer: number; head: number } | null = null;
  private headHover: { layer: number; head: number } | null = null;
  private hoverBand: THREE.Mesh;
  private selBand: THREE.Mesh;
  private dataOutline: THREE.LineSegments;
  private dataLayers = new Set<number>();
  private hiGeo = new THREE.BufferGeometry();
  private hiPts: THREE.Points;
  private conGeo = new THREE.BufferGeometry();
  private webMat: LineMaterial;
  private webLines: LineSegments2;
  private webPtGeo = new THREE.BufferGeometry();
  private webPts: THREE.Points;
  private webEdges: Int16Array | null = null;
  private webFed = new Set<number>();
  private webFeeds = new Set<number>();
  private markGeo = new THREE.BufferGeometry();
  private markPts: THREE.Points;
  private lastMarks: Mark[] = [];
  private selected: { layer: number; n: number } | null = null;
  private hovered: { layer: number; n: number } | null = null;
  private connections: Linked[] = [];
  private hoverCon: { src: { layer: number; n: number } | null; from?: THREE.Vector3; list: Linked[] } | null = null;
  private headSrc: THREE.Vector3 | null = null;
  private neuronHeads: { layer: number; head: number; z: number }[] = [];
  private unitsLayer: number | null = null;
  private units: CensusUnitRow[] | null = null;
  private named = new Set<number>();
  private notes: { layer: number; n: number; el: HTMLDivElement }[] = [];
  private homeDist = 90;
  private home = { target: new THREE.Vector3(), dir: new THREE.Vector3(0.62, 0.4, 1).normalize() };
  private insets = { left: 0, right: 0 };
  private fitted = false;
  private fade = { fx: 0, fw: 1, amt: 0, side: 1 };
  private mode: number | null = null;
  private saved: { pos: THREE.Vector3; tgt: THREE.Vector3 } | null = null;
  private pending = false;
  private disposed = false;
  private ro: ResizeObserver;
  private downAt: [number, number] | null = null;
  private moveQueued: PointerEvent | null = null;
  private bounds: THREE.Vector3[] = [];
  private kinds: ("sliding" | "full")[];
  private heads: number[];

  constructor(
    private canvas: HTMLCanvasElement,
    private notesDiv: HTMLDivElement,
    arch: Arch,
    private cb: MapCallbacks,
  ) {
    const L = (this.L = arch.n_layers);
    const N = (this.N = arch.d_ff);
    this.kinds = arch.layers.map((s) => s.kind);
    this.heads = arch.layers.map((s) => s.n_heads);
    this.side = Math.ceil(Math.sqrt(N));
    this.cellToN = new Int32Array(L * this.side * this.side).fill(-1);
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setClearColor(INKS.bg);
    this.camera = new THREE.PerspectiveCamera(32, 1, 0.02, 4000);
    this.home.target.set(((L - 1) * GAP) / 2, SHEET_Y0 + SHEET / 2 - 0.4, 0);
    this.camera.position.copy(this.home.target).addScaledVector(this.home.dir, this.homeDist);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.copy(this.home.target);
    this.controls.enableDamping = false;
    this.controls.zoomToCursor = true;
    this.controls.minDistance = 1.6;
    this.controls.maxDistance = 800;
    this.controls.zoomSpeed = 0.6;
    this.controls.update();

    // every neuron of every layer, one draw call
    this.pos = new Float32Array(L * N * 3);
    this.col = new Float32Array(L * N * 3);
    this.fillPositions();
    for (let i = 0; i < L * N; i++) INKS.idle.toArray(this.col, i * 3);
    this.geo.setAttribute("position", new THREE.BufferAttribute(this.pos, 3));
    this.geo.setAttribute("color", new THREE.BufferAttribute(this.col, 3));
    this.pointMat = dotMaterial(1.7, 5, 0.42);
    this.points = new THREE.Points(this.geo, this.pointMat);
    this.points.frustumCulled = false;
    this.scene.add(this.points);

    // the structure: the stream's axis, each layer's slab and sheet, the heads on each slab
    const wires: number[] = [];
    const V = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);
    const seg = (a: THREE.Vector3, b: THREE.Vector3) => wires.push(a.x, a.y, a.z, b.x, b.y, b.z);
    const X0 = ATTN_DX - 1.6;
    const X1 = (L - 1) * GAP + MLP_DX + 1.4;
    seg(V(X0, 0, 0), V(X1, 0, 0));
    const HEAD_W = SHEET * 0.41;
    for (let l = 0; l < L; l++) {
      const x = l * GAP;
      const mx = x + MLP_DX;
      const ax = x + ATTN_DX;
      const cs = this.sheetCorners(mx);
      for (let i = 0; i < 4; i++) seg(cs[i], cs[(i + 1) % 4]);
      seg(V(ax, 0, 0), V(ax, SHEET_Y0, 0));
      seg(V(mx, 0, 0), V(mx, SHEET_Y0, 0));
      const full = this.kinds[l] === "full";
      const t = full ? 0.3 : 0.1;
      const slab = new THREE.Mesh(
        new THREE.BoxGeometry(t, SHEET * 0.92, SHEET * 0.92),
        new THREE.MeshBasicMaterial({ color: INKS.ink, transparent: true, opacity: full ? 0.07 : 0.035, depthWrite: false, depthTest: false }),
      );
      slab.position.set(ax, SHEET_Y0 + SHEET / 2, 0);
      slab.userData = { layer: l, full, heads: this.heads[l], baseOpacity: (slab.material as THREE.MeshBasicMaterial).opacity };
      this.scene.add(slab);
      this.slabs.push(slab);
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(slab.geometry),
        new THREE.LineBasicMaterial({ color: full ? INKS.ink : INKS.hair, transparent: true, opacity: full ? 0.5 : 0.8, depthTest: false }),
      );
      edges.position.copy(slab.position);
      slab.userData.edges = edges;
      this.scene.add(edges);
      this.headStart.push(this.headPos.length / 6);
      const H = this.heads[l];
      for (let h = 0; h < H; h++) {
        const y = this.headY(h, H);
        this.headPos.push(ax + t / 2 + 0.003, y, -HEAD_W, ax + t / 2 + 0.003, y, HEAD_W);
        this.headBase.push(full ? toward(INKS.ink, 0.55) : toward(INKS.ink, 0.28));
      }
    }
    this.headGeo.setAttribute("position", new THREE.Float32BufferAttribute(this.headPos, 3));
    this.headCol = new Float32Array(this.headPos.length);
    this.headGeo.setAttribute("color", new THREE.BufferAttribute(this.headCol, 3));
    const headMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 1, depthTest: false });
    headMat.userData.base = 1;
    this.scene.add(new THREE.LineSegments(this.headGeo, headMat));
    this.fadeLines.push(headMat);
    const band = (op: number) => {
      const m = new THREE.Mesh(
        new THREE.BoxGeometry(1, 1, 1),
        new THREE.MeshBasicMaterial({ color: INKS.ink, transparent: true, opacity: op, depthWrite: false, depthTest: false }),
      );
      m.visible = false;
      m.renderOrder = 1;
      this.scene.add(m);
      return m;
    };
    this.hoverBand = band(0.08);
    this.selBand = band(0.16);
    const wg = new THREE.BufferGeometry();
    wg.setAttribute("position", new THREE.Float32BufferAttribute(wires, 3));
    const wireMat = new THREE.LineBasicMaterial({ color: INKS.hair, transparent: true, opacity: 1, depthTest: false });
    wireMat.userData.base = 1;
    this.scene.add(new THREE.LineSegments(wg, wireMat));
    this.fadeLines.push(wireMat);
    this.dataOutline = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: INKS.gilt, depthTest: false }));
    this.dataOutline.frustumCulled = false;
    this.scene.add(this.dataOutline);
    for (const x of [X0, X1]) for (const y of [-0.8, SHEET_Y0 + SHEET]) for (const z of [-SHEET / 2, SHEET / 2]) this.bounds.push(V(x, y, z));
    this.paintHeads();

    // what is drawn over the sheets: marks, the web, links, the selection
    this.markPts = new THREE.Points(this.markGeo, dotMaterial(3.6, 2.2, 0.95));
    this.markPts.frustumCulled = false;
    this.markPts.renderOrder = 2;
    this.scene.add(this.markPts);
    this.webMat = new LineMaterial({ vertexColors: true, linewidth: 1.6, transparent: true, opacity: 0.9, depthWrite: false, depthTest: false });
    this.webLines = new LineSegments2(new LineSegmentsGeometry(), this.webMat);
    this.webLines.frustumCulled = false;
    this.webLines.renderOrder = 1;
    this.webLines.visible = false;
    this.scene.add(this.webLines);
    this.webPts = new THREE.Points(this.webPtGeo, dotMaterial(3.8, 2.4, 0.95));
    this.webPts.frustumCulled = false;
    this.webPts.renderOrder = 2;
    this.webPts.visible = false;
    this.scene.add(this.webPts);
    const conLines = new THREE.LineSegments(this.conGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.95, depthTest: false }));
    conLines.frustumCulled = false;
    conLines.renderOrder = 2;
    this.scene.add(conLines);
    this.hiPts = new THREE.Points(this.hiGeo, dotMaterial(6.5, 2, 1));
    this.hiPts.frustumCulled = false;
    this.hiPts.renderOrder = 3;
    this.scene.add(this.hiPts);

    this.makeLabels();
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(canvas);
    this.controls.addEventListener("change", () => this.render());
    canvas.addEventListener("pointermove", this.onMove);
    canvas.addEventListener("pointerleave", this.onLeave);
    canvas.addEventListener("pointerdown", this.onDown);
    canvas.addEventListener("pointerup", this.onUp);
    this.resize();
  }

  /* ------------------------------------------------------------------------------------------------- geometry */

  private layerX = (l: number) => l * GAP;
  private headY = (h: number, H: number) => SHEET_Y0 + SHEET / 2 + (0.5 - (h + 0.5) / H) * SHEET * 0.82;

  private sheetCorners(x: number) {
    const h = SHEET / 2;
    return [
      new THREE.Vector3(x, SHEET_Y0, -h),
      new THREE.Vector3(x, SHEET_Y0, h),
      new THREE.Vector3(x, SHEET_Y0 + SHEET, h),
      new THREE.Vector3(x, SHEET_Y0 + SHEET, -h),
    ];
  }

  private cellOf(l: number, n: number) {
    return this.cells ? this.cells[l * this.N + n] : n;
  }

  neuronPos(l: number, n: number): THREE.Vector3 {
    const c = this.cellOf(l, n);
    const s = this.side;
    const i = c % s;
    const j = Math.floor(c / s);
    return new THREE.Vector3(this.layerX(l) + MLP_DX, SHEET_Y0 + (j / (s - 1)) * SHEET, (i / (s - 1) - 0.5) * SHEET);
  }

  private fillPositions() {
    const { L, N, side: s } = this;
    this.cellToN.fill(-1);
    for (let l = 0; l < L; l++) {
      const x = this.layerX(l) + MLP_DX;
      for (let n = 0; n < N; n++) {
        const c = this.cellOf(l, n);
        const k = (l * N + n) * 3;
        this.pos[k] = x;
        this.pos[k + 1] = SHEET_Y0 + (Math.floor(c / s) / (s - 1)) * SHEET;
        this.pos[k + 2] = ((c % s) / (s - 1) - 0.5) * SHEET;
        this.cellToN[l * s * s + c] = n;
      }
    }
  }

  private headMid(layer: number, head: number) {
    const i = (this.headStart[layer] + head) * 6;
    const p = this.headPos;
    return new THREE.Vector3((p[i] + p[i + 3]) / 2, (p[i + 1] + p[i + 4]) / 2, (p[i + 2] + p[i + 5]) / 2);
  }

  private makeLabels() {
    const H = 0.22;
    for (let l = 0; l < this.L; l++) {
      const text = String(l + 1);
      const cv = document.createElement("canvas");
      const ctx = cv.getContext("2d")!;
      const font = "500 64px 'EB Garamond', Garamond, Georgia, serif";
      ctx.font = font;
      const tw = Math.ceil(ctx.measureText(text).width) + 10;
      cv.width = tw;
      cv.height = 80;
      ctx.font = font;
      ctx.fillStyle = "#2b2117";
      ctx.textBaseline = "middle";
      ctx.fillText(text, 5, 44);
      const tex = new THREE.CanvasTexture(cv);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.anisotropy = 8;
      const w = (H * tw) / 80;
      const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, depthTest: false, side: THREE.DoubleSide });
      mat.userData.base = 0.85;
      const m = new THREE.Mesh(new THREE.PlaneGeometry(w, H), mat);
      m.rotation.y = Math.PI / 2;
      m.position.set(this.layerX(l) + MLP_DX, SHEET_Y0 + SHEET + 0.1 + H / 2, -SHEET / 2 + w / 2);
      m.userData = { layer: l };
      this.scene.add(m);
      this.labels.push(m);
    }
  }

  /* ---------------------------------------------------------------------------------------------- painting */

  private paintHeads() {
    this.placeBand(
      this.hoverBand,
      this.headHover && !(this.headSel && this.headSel.layer === this.headHover.layer && this.headSel.head === this.headHover.head) ? this.headHover : null,
    );
    this.placeBand(this.selBand, this.headSel);
    const c = new THREE.Color();
    for (let l = 0; l < this.L; l++) {
      const H = (this.headStart[l + 1] ?? this.headBase.length) - this.headStart[l];
      for (let h = 0; h < H; h++) {
        const i = this.headStart[l] + h;
        const lv = this.headLevels?.[l]?.[h] ?? 0;
        c.copy(this.headBase[i]);
        if (this.headLevels) c.copy(toward(INKS.ink, 0.15)).lerp(INKS.up, Math.min(1, lv));
        if (this.headHover && this.headHover.layer === l && this.headHover.head === h) c.lerp(INKS.ink, 0.6);
        if (this.headSel && this.headSel.layer === l && this.headSel.head === h) c.copy(INKS.named);
        c.toArray(this.headCol, i * 6);
        c.toArray(this.headCol, i * 6 + 3);
      }
    }
    (this.headGeo.attributes.color as THREE.BufferAttribute).needsUpdate = true;
  }

  private placeBand(m: THREE.Mesh, at: { layer: number; head: number } | null) {
    m.visible = !!at;
    if (!at) return;
    const s = this.slabs[at.layer];
    const H = s.userData.heads as number;
    m.position.set(s.position.x, this.headY(at.head, H), 0);
    m.scale.set((s.geometry as THREE.BoxGeometry).parameters.width + 0.02, ((SHEET * 0.82) / H) * 0.9, SHEET * 0.82 + 0.1);
  }

  private buildWeb() {
    const E = this.webEdges;
    if (!E || !E.length) {
      this.webFed.clear();
      this.webFeeds.clear();
      this.webLines.visible = false;
      this.webPts.visible = false;
      return;
    }
    const N = this.N;
    const m = E.length / 6;
    const p = new Float32Array(m * 6);
    const c = new Float32Array(m * 6);
    let zhi = 1;
    for (let i = 0; i < m; i++) zhi = Math.max(zhi, Math.abs(E[i * 6 + 4]));
    const fed = this.webFed;
    const feeds = this.webFeeds;
    fed.clear();
    feeds.clear();
    for (let i = 0; i < m; i++) {
      feeds.add(E[i * 6] * N + E[i * 6 + 1]);
      fed.add(E[i * 6 + 2] * N + E[i * 6 + 3]);
    }
    const root = this.selected ? this.selected.layer * N + this.selected.n : -1;
    const touched = new Map<number, [THREE.Vector3, THREE.Color]>();
    for (let i = 0; i < m; i++) {
      const e = E.subarray(i * 6, i * 6 + 6);
      const a = this.neuronPos(e[0], e[1]);
      const b = this.neuronPos(e[2], e[3]);
      const s0 = e[0] * N + e[1];
      const d0 = e[2] * N + e[3];
      const through = (s0 === root || fed.has(s0)) && (d0 === root || feeds.has(d0));
      const al = (0.45 + 0.55 * (Math.abs(e[4]) / zhi)) * Math.max(0.35, 0.85 ** (Math.abs(e[5]) - 1)) * (through ? 1 : 0.18);
      const k = toward(e[4] > 0 ? INKS.up : INKS.down, al);
      p.set([a.x, a.y, a.z, b.x, b.y, b.z], i * 6);
      c.set([k.r, k.g, k.b, k.r, k.g, k.b], i * 6);
      const key = e[5] > 0 ? d0 : s0;
      if (!touched.has(key)) touched.set(key, [e[5] > 0 ? b : a, e[4] > 0 ? INKS.up : INKS.down]);
    }
    const tp: number[] = [];
    const tc: number[] = [];
    for (const [id, [v, k]] of touched) {
      const f = (fed.has(id) || id === root) && (feeds.has(id) || id === root) ? 1 : 0.35;
      const kk = toward(k, f);
      tp.push(v.x, v.y, v.z);
      tc.push(kk.r, kk.g, kk.b);
    }
    this.webPtGeo.setAttribute("position", new THREE.Float32BufferAttribute(tp, 3));
    this.webPtGeo.setAttribute("color", new THREE.Float32BufferAttribute(tc, 3));
    this.webPts.visible = true;
    this.webLines.geometry.dispose();
    this.webLines.geometry = new LineSegmentsGeometry().setPositions(p).setColors(c);
    this.webLines.visible = true;
  }

  private refreshHighlights() {
    const p: number[] = [];
    const c: number[] = [];
    const add = (l: number, n: number, color: THREE.Color) => {
      const v = this.neuronPos(l, n);
      p.push(v.x, v.y, v.z);
      c.push(color.r, color.g, color.b);
    };
    const src = this.hoverCon ? this.hoverCon.src : this.selected;
    const shown = this.hoverCon ? this.hoverCon.list : this.connections;
    const zmax = Math.max(1, ...shown.map((k) => k.z));
    const linkCol = (k: Linked) => (k.role ? ROLE[k.role] : (k as Link).sign > 0 ? INKS.up : (k as Link).sign < 0 ? INKS.down : INKS.v);
    // a neuron's link by its strength as it reaches the other end (w, over the strongest); a head's by its z
    const strength = (k: Linked) => (!k.role && typeof (k as Link).w === "number" ? Math.sqrt((k as Link).w as number) : k.z / zmax);
    for (const k of shown) add(k.layer, k.n, toward(linkCol(k), 0.45 + 0.55 * strength(k)));
    if (this.hovered) add(this.hovered.layer, this.hovered.n, toward(INKS.ink, 0.6));
    if (this.selected) add(this.selected.layer, this.selected.n, INKS.ink);
    if (this.headSrc && !this.selected) {
      p.push(this.headSrc.x, this.headSrc.y, this.headSrc.z);
      c.push(INKS.named.r, INKS.named.g, INKS.named.b);
    }
    this.hiGeo.setAttribute("position", new THREE.Float32BufferAttribute(p, 3));
    this.hiGeo.setAttribute("color", new THREE.Float32BufferAttribute(c, 3));
    const lp: number[] = [];
    const lc: number[] = [];
    const a = src ? this.neuronPos(src.layer, src.n) : this.hoverCon?.from ?? this.headSrc;
    if (a) {
      for (const k of shown) {
        const far = k.layer * this.N + k.n;
        const on = !this.webEdges || !this.webEdges.length || this.hoverCon || (k.side === "up" ? this.webFed.has(far) : this.webFeeds.has(far));
        const b = this.neuronPos(k.layer, k.n);
        const cl = toward(linkCol(k), 0.6 * (0.25 + 0.75 * strength(k)) * (on ? 1 : 0.25));
        lp.push(a.x, a.y, a.z, b.x, b.y, b.z);
        lc.push(cl.r, cl.g, cl.b, cl.r, cl.g, cl.b);
      }
    }
    if (this.selected && !this.hoverCon && this.neuronHeads.length) {
      const o = this.neuronPos(this.selected.layer, this.selected.n);
      const hz = Math.max(...this.neuronHeads.map((k) => k.z));
      for (const k of this.neuronHeads) {
        const b = this.headMid(k.layer, k.head);
        const cl = toward(INKS.v, 0.6 * (0.25 + 0.75 * (k.z / hz)));
        lp.push(o.x, o.y, o.z, b.x, b.y, b.z);
        lc.push(cl.r, cl.g, cl.b, cl.r, cl.g, cl.b);
      }
    }
    this.conGeo.setAttribute("position", new THREE.Float32BufferAttribute(lp, 3));
    this.conGeo.setAttribute("color", new THREE.Float32BufferAttribute(lc, 3));
    this.render();
  }

  private paintNamed() {
    // a named neuron shows on every layer in the rubric's red; the census's colours on the layer that is open
    for (const id of this.named) INKS.named.toArray(this.col, id * 3);
    (this.geo.attributes.color as THREE.BufferAttribute).needsUpdate = true;
  }

  render() {
    if (this.pending || this.disposed) return;
    this.pending = true;
    requestAnimationFrame(() => {
      this.pending = false;
      if (this.disposed) return;
      for (const m of [this.pointMat, this.hiPts.material, this.markPts.material, this.webPts.material] as THREE.RawShaderMaterial[])
        m.uniforms.ref.value = this.homeDist;
      this.applyFade();
      this.renderer.render(this.scene, this.camera);
      this.placeNotes();
    });
  }

  private resize() {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.webMat.resolution.set(w, h);
    const narrow = w <= 760;
    const li = narrow ? 0 : this.insets.left;
    const ri = narrow ? 0 : this.insets.right;
    const availW = Math.max(200, w - li - ri);
    this.camera.aspect = w / h;
    if (!this.fitted) {
      const center = this.home.target;
      const r = Math.max(...this.bounds.map((b) => b.distanceTo(center)));
      const vHalf = THREE.MathUtils.degToRad(this.camera.fov / 2);
      const hHalf = Math.atan((Math.tan(vHalf) * availW) / h);
      this.homeDist = (r / Math.sin(Math.min(vHalf, hHalf))) * 0.82;
      this.camera.position.copy(center).addScaledVector(this.home.dir, this.homeDist);
      this.controls.update();
      this.fitted = true;
    }
    const tx = li + availW / 2;
    this.camera.setViewOffset(w, h, w / 2 - tx, 0, w, h);
    this.camera.updateProjectionMatrix();
    this.render();
  }

  private curDist() {
    return this.camera.position.distanceTo(this.controls.target);
  }

  private layerVisibility(l: number) {
    const dx = this.layerX(l) + MLP_DX - this.fade.fx;
    const d = dx / this.fade.fw;
    if ((dx * this.fade.side) / GAP > 0.45) return 1 - this.fade.amt;
    return 1 - 0.75 * this.fade.amt * (1 - Math.exp(-d * d));
  }

  private applyFade() {
    const mats = [this.pointMat];
    if (this.mode !== null) {
      const fx = this.layerX(this.mode) + MLP_DX;
      for (const m of mats) {
        m.uniforms.focusX.value = fx;
        m.uniforms.fadeW.value = 0.05;
        m.uniforms.fadeAmt.value = 1;
      }
      this.fade = { fx, fw: 0.05, amt: 1, side: 1 };
      for (const slab of this.slabs) {
        (slab.material as THREE.MeshBasicMaterial).opacity = 0;
        ((slab.userData.edges as THREE.LineSegments).material as THREE.LineBasicMaterial).opacity = 0;
      }
      for (const lb of this.labels) (lb.material as THREE.MeshBasicMaterial).opacity = 0;
      for (const m of this.fadeLines) m.opacity = 0;
      this.dataOutline.visible = false;
      return;
    }
    this.dataOutline.visible = true;
    const dist = this.curDist();
    const far = Math.log(this.homeDist * 0.7);
    const near = Math.log(this.homeDist * 0.05);
    const amt = THREE.MathUtils.smoothstep(THREE.MathUtils.clamp((far - Math.log(dist)) / (far - near), 0, 1), 0, 1);
    const tv = this.controls.target.clone().project(this.camera);
    const pv = new THREE.Vector3();
    let fx = this.controls.target.x;
    let best = Infinity;
    for (let l = 0; l < this.L; l++) {
      pv.set(this.layerX(l) + MLP_DX, SHEET_Y0 + SHEET / 2, 0).project(this.camera);
      const dd = Math.hypot((pv.x - tv.x) * this.camera.aspect, pv.y - tv.y);
      if (pv.z < 1 && dd < best) {
        best = dd;
        fx = this.layerX(l) + MLP_DX;
      }
    }
    const fw = GAP * (0.85 + 2.5 * (1 - amt));
    const side = Math.sign(this.camera.position.x - fx) || 1;
    for (const m of mats) {
      m.uniforms.focusX.value = fx;
      m.uniforms.fadeW.value = fw;
      m.uniforms.fadeAmt.value = amt;
      m.uniforms.camSide.value = side;
    }
    this.fade = { fx, fw, amt, side };
    for (const slab of this.slabs) {
      const d = (slab.position.x - fx) / fw;
      const f = 1 - amt * (1 - Math.exp(-d * d));
      (slab.material as THREE.MeshBasicMaterial).opacity = (slab.userData.baseOpacity as number) * f;
      ((slab.userData.edges as THREE.LineSegments).material as THREE.LineBasicMaterial).opacity = (slab.userData.full ? 0.5 : 0.8) * f;
    }
    for (let l = 0; l < this.L; l++) {
      const d = (this.layerX(l) + MLP_DX - fx) / fw;
      const mat = this.labels[l].material as THREE.MeshBasicMaterial;
      mat.opacity = (mat.userData.base as number) * (1 - 0.5 * amt * (1 - Math.exp(-d * d)));
    }
    for (const m of this.fadeLines) m.opacity = (m.userData.base as number) * (1 - 0.85 * amt);
  }

  private placeNotes() {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    for (const a of this.notes) {
      const v = this.neuronPos(a.layer, a.n).project(this.camera);
      a.el.style.display = v.z > 1 || (this.mode !== null && a.layer !== this.mode) ? "none" : "";
      a.el.style.left = `${((v.x + 1) / 2) * w}px`;
      a.el.style.top = `${((1 - v.y) / 2) * h}px`;
    }
  }

  /* ----------------------------------------------------------------------------------------------- picking */

  private pick(ev: PointerEvent): Pick | null {
    const r = this.canvas.getBoundingClientRect();
    const mx = ev.clientX - r.left;
    const my = ev.clientY - r.top;
    const ndc = new THREE.Vector2((mx / r.width) * 2 - 1, -(my / r.height) * 2 + 1);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, this.camera);
    const o = ray.ray.origin;
    const d = ray.ray.direction;
    const px = (v: THREE.Vector3) => {
      v.project(this.camera);
      return Math.hypot(((v.x + 1) / 2) * r.width - mx, ((1 - v.y) / 2) * r.height - my);
    };
    let headHit: (Pick & { type: "head" }) | null = null;
    let headDist = Infinity;
    if (this.headSrc && this.headSel && px(this.headSrc.clone()) < 8) {
      headHit = { type: "head", layer: this.headSel.layer, head: this.headSel.head, full: this.kinds[this.headSel.layer] === "full" };
      headDist = o.distanceTo(this.headSrc);
    }
    if (!headHit && Math.abs(d.x) > 1e-6) {
      for (const sh of ray.intersectObjects(this.slabs)) {
        const mat = (sh.object as THREE.Mesh).material as THREE.MeshBasicMaterial;
        if (mat.opacity <= 0.003) continue;
        const { layer, full, heads } = sh.object.userData as { layer: number; full: boolean; heads: number };
        const lx = sh.object.position.x + ((sh.object as THREE.Mesh).geometry as THREE.BoxGeometry).parameters.width / 2;
        const on = o.clone().addScaledVector(d, (lx - o.x) / d.x);
        if (Math.abs(on.z) > SHEET * 0.41) continue;
        const head = Math.floor((0.5 - (on.y - SHEET_Y0 - SHEET / 2) / (SHEET * 0.82)) * heads);
        if (head < 0 || head >= heads) continue;
        if (px(new THREE.Vector3(lx, this.headY(head, heads), on.z)) < 6) {
          headHit = { type: "head", layer, head, full };
          headDist = sh.distance;
          break;
        }
      }
    }
    // the neuron under the pointer: each sheet a plane at its x, its grid cell where the ray meets it
    type Cand = { layer: number; n: number; t: number; vis: number };
    let neuron: Cand | null = null;
    if (Math.abs(d.x) > 1e-6) {
      const s = this.side;
      const cands: Cand[] = [];
      const layers = this.mode !== null ? [this.mode] : Array.from({ length: this.L }, (_, l) => l);
      for (const l of layers) {
        const x = this.layerX(l) + MLP_DX;
        const t = (x - o.x) / d.x;
        if (t <= 0) continue;
        const y = o.y + t * d.y;
        const z = o.z + t * d.z;
        const eps = SHEET / (s - 1);
        if (y < SHEET_Y0 - eps || y > SHEET_Y0 + SHEET + eps || Math.abs(z) > SHEET / 2 + eps) continue;
        const ci = Math.round((z / SHEET + 0.5) * (s - 1));
        const cj = Math.round(((y - SHEET_Y0) / SHEET) * (s - 1));
        let best = -1;
        let bestPx = 7;
        for (let dj = -1; dj <= 1; dj++)
          for (let di = -1; di <= 1; di++) {
            const i = ci + di;
            const j = cj + dj;
            if (i < 0 || j < 0 || i >= s || j >= s) continue;
            const n = this.cellToN[l * s * s + j * s + i];
            if (n < 0) continue;
            const dpx = px(this.neuronPos(l, n));
            if (dpx < bestPx) {
              bestPx = dpx;
              best = n;
            }
          }
        if (best >= 0) cands.push({ layer: l, n: best, t, vis: this.layerVisibility(l) });
      }
      cands.sort((a, b) => a.t - b.t);
      neuron = cands.find((c) => c.vis > 0.5) ?? cands.reduce<Cand | null>((a, b) => (!a || b.vis > a.vis ? b : a), null);
    }
    if (neuron && !(headHit && headDist < neuron.t)) return { type: "neuron", layer: neuron.layer, n: neuron.n };
    if (headHit) return headHit;
    const lb = ray.intersectObjects(this.labels)[0];
    if (lb) return { type: "block", layer: lb.object.userData.layer as number, full: this.kinds[lb.object.userData.layer as number] === "full" };
    const b = ray.intersectObjects(this.slabs).find((x) => ((x.object as THREE.Mesh).material as THREE.MeshBasicMaterial).opacity > 0.003);
    if (b) return { type: "block", layer: b.object.userData.layer as number, full: !!b.object.userData.full };
    return null;
  }

  private onMove = (ev: PointerEvent) => {
    if (this.moveQueued) {
      this.moveQueued = ev;
      return;
    }
    this.moveQueued = ev;
    requestAnimationFrame(() => {
      const e = this.moveQueued;
      this.moveQueued = null;
      if (!e || this.disposed) return;
      const p = this.pick(e);
      const h = p && p.type === "neuron" ? { layer: p.layer, n: p.n } : null;
      if (JSON.stringify(h) !== JSON.stringify(this.hovered)) {
        this.hovered = h;
        this.refreshHighlights();
      }
      const hh = p && p.type === "head" ? { layer: p.layer, head: p.head } : null;
      if (JSON.stringify(hh) !== JSON.stringify(this.headHover)) {
        this.headHover = hh;
        this.paintHeads();
        this.render();
      }
      this.cb.onHover(p, e.clientX, e.clientY);
    });
  };

  private onLeave = () => {
    this.hovered = null;
    this.refreshHighlights();
    if (this.headHover) {
      this.headHover = null;
      this.paintHeads();
      this.render();
    }
    this.cb.onHover(null, 0, 0);
  };

  private onDown = (ev: PointerEvent) => {
    this.downAt = [ev.clientX, ev.clientY];
  };

  private onUp = (ev: PointerEvent) => {
    if (!this.downAt || Math.hypot(ev.clientX - this.downAt[0], ev.clientY - this.downAt[1]) > 4) return;
    const p = this.pick(ev);
    if (p?.type === "neuron") this.cb.onPick(p.layer, p.n);
    else if (p?.type === "head") this.cb.onPickHead(p.layer, p.head);
    else if (p?.type === "block") this.cb.onPickLayer(p.layer);
  };

  /* ------------------------------------------------------------------------------------------------- camera */

  private flyTo(target: THREE.Vector3, dist: number) {
    const from = this.controls.target.clone();
    const to = target.clone();
    const dir = this.camera.position.clone().sub(this.controls.target);
    const d0 = dir.length();
    dir.normalize();
    const t0 = performance.now();
    const step = (t: number) => {
      if (this.disposed) return;
      const k = Math.min(1, (t - t0) / 450);
      const e = k * k * (3 - 2 * k);
      this.controls.target.lerpVectors(from, to, e);
      this.camera.position.copy(this.controls.target).addScaledVector(dir, d0 + (dist - d0) * e);
      this.controls.update();
      this.render();
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  get layerMode() {
    return this.mode;
  }

  enterLayer(l: number) {
    if (this.mode === null) this.saved = { pos: this.camera.position.clone(), tgt: this.controls.target.clone() };
    this.mode = l;
    const c = new THREE.Vector3(this.layerX(l) + MLP_DX, SHEET_Y0 + SHEET / 2, 0);
    const dist = (SHEET / 2 / Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2))) * 1.12;
    this.controls.target.copy(c);
    this.camera.position.copy(c).add(new THREE.Vector3(-dist, 0, 0));
    this.controls.enableRotate = false;
    this.controls.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    this.controls.update();
    this.render();
    this.cb.onLayerMode(l);
  }

  exitLayer() {
    if (this.mode === null) return;
    this.mode = null;
    this.controls.enableRotate = true;
    this.controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    if (this.saved) {
      this.camera.position.copy(this.saved.pos);
      this.controls.target.copy(this.saved.tgt);
    }
    this.controls.update();
    this.render();
    this.cb.onLayerMode(null);
  }

  viewLayer(layer: number) {
    const c = new THREE.Vector3(this.layerX(layer) + MLP_DX, SHEET_Y0 + SHEET / 2, 0);
    const dist = (SHEET / 2 / Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2))) * 1.25;
    const from = this.controls.target.clone();
    const p0 = this.camera.position.clone();
    const p1 = c.clone().add(new THREE.Vector3(dist, 0.0001, 0));
    const t0 = performance.now();
    const step = (t: number) => {
      if (this.disposed) return;
      const k = Math.min(1, (t - t0) / 600);
      const e = k * k * (3 - 2 * k);
      this.controls.target.lerpVectors(from, c, e);
      this.camera.position.lerpVectors(p0, p1, e);
      this.controls.update();
      this.render();
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  goHome() {
    this.flyTo(this.home.target, this.homeDist);
  }

  focusNeuron(layer: number, n: number) {
    if (this.mode === null) this.flyTo(this.neuronPos(layer, n), Math.min(this.curDist(), this.homeDist * 0.35));
  }

  flyToNeuron(layer: number, n: number, closeness = 0.22) {
    this.flyTo(this.neuronPos(layer, n), this.homeDist * closeness);
  }

  /* -------------------------------------------------------------------------------------------- what's shown */

  setHeadLevels(levels: (number[] | undefined)[] | null) {
    this.headLevels = levels;
    this.paintHeads();
    this.render();
  }

  selectHead(layer: number | null, head?: number) {
    this.headSel = layer == null || head == null ? null : { layer, head };
    if (!this.headSel && this.headSrc) {
      this.headSrc = null;
      this.connections = [];
      this.refreshHighlights();
    }
    this.paintHeads();
    this.render();
  }

  setHeadLinks(layer: number, head: number, list: Linked[]) {
    this.headSrc = this.headMid(layer, head);
    this.connections = list;
    this.refreshHighlights();
  }

  setHoverHead(layer: number | null, head?: number, list: Linked[] = []) {
    this.hoverCon = layer == null || head == null ? null : { src: null, from: this.headMid(layer, head), list };
    this.refreshHighlights();
  }

  hoverHead(layer: number | null, head?: number) {
    const hh = layer == null || head == null ? null : { layer, head };
    if (JSON.stringify(hh) !== JSON.stringify(this.headHover)) {
      this.headHover = hh;
      this.paintHeads();
      this.render();
    }
  }

  setNeuronHeads(list: { layer: number; head: number; z: number }[]) {
    this.neuronHeads = list;
    this.refreshHighlights();
  }

  /** The census's colours on one layer at a time (the open one): above every control from light to dark gilt, the
   * rest a little darker than idle; names keep the rubric's red on every layer. */
  setUnits(layer: number, units: CensusUnitRow[]) {
    const N = this.N;
    if (this.unitsLayer !== null && this.unitsLayer !== layer) for (let n = 0; n < N; n++) INKS.idle.toArray(this.col, (this.unitsLayer * N + n) * 3);
    this.unitsLayer = layer;
    this.units = units;
    this.dataLayers.add(layer);
    const zs = units.filter((u) => u.kind === "neuron").map((u) => u.z);
    const rnd = units.filter((u) => u.kind === "random").map((u) => u.z);
    const floor = rnd.length ? Math.max(...rnd) : Math.min(...zs, 0);
    const top = Math.max(...zs, floor + 1e-6);
    for (const u of units) {
      if (u.kind !== "neuron") continue;
      let c = INKS.swept;
      if (u.z > floor) c = INKS.lo.clone().lerp(INKS.hi, Math.min(1, (u.z - floor) / (top - floor)));
      c.toArray(this.col, (layer * N + u.n) * 3);
    }
    this.paintNamed();
    const dl: number[] = [];
    for (const l of this.dataLayers) {
      const cs = this.sheetCorners(this.layerX(l) + MLP_DX);
      for (let i = 0; i < 4; i++) dl.push(...cs[i].toArray(), ...cs[(i + 1) % 4].toArray());
    }
    this.dataOutline.geometry.setAttribute("position", new THREE.Float32BufferAttribute(dl, 3));
    this.render();
  }

  /** Neurons with a name in the field notes (layer * N + n). */
  setNamed(keys: { layer: number; n: number }[]) {
    const N = this.N;
    const next = new Set(keys.map((k) => k.layer * N + k.n));
    const gone = [...this.named].some((id) => !next.has(id));
    for (const id of this.named) if (!next.has(id)) INKS.idle.toArray(this.col, id * 3);
    this.named = next;
    // a name taken away on the open layer: its census colour comes back
    if (gone && this.unitsLayer !== null && this.units) this.setUnits(this.unitsLayer, this.units);
    else this.paintNamed();
    this.render();
  }

  select(layer: number | null, n?: number, fly = true) {
    this.selected = layer == null || n == null ? null : { layer, n };
    this.connections = [];
    this.headSrc = null;
    this.neuronHeads = [];
    this.refreshHighlights();
    if (!fly || !this.selected || this.mode !== null) return;
    const v = this.neuronPos(this.selected.layer, this.selected.n).project(this.camera);
    if (v.z > 1 || Math.abs(v.x) > 0.9 || Math.abs(v.y) > 0.9)
      this.flyTo(this.neuronPos(this.selected.layer, this.selected.n), Math.min(this.curDist(), this.homeDist * 0.35));
  }

  setConnections(down: Link[], up: Link[]) {
    this.connections = [...down.map((c) => ({ ...c, side: "down" as const })), ...up.map((c) => ({ ...c, side: "up" as const }))];
    this.refreshHighlights();
  }

  setHoverConnections(src: { layer: number; n: number } | null, links?: { down: Link[]; up: Link[] } | null) {
    this.hoverCon = src && links ? { src, list: [...links.down, ...links.up] } : null;
    this.refreshHighlights();
  }

  setLayout(cells: Int16Array | null) {
    this.cells = cells;
    this.fillPositions();
    (this.geo.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    this.setMarks(this.lastMarks);
    this.buildWeb();
    this.refreshHighlights();
  }

  /** The web: Int16Array rows of six (src layer, src n, dst layer, dst n, 10 z sign, hop), or null to hide it. */
  setWeb(edges: Int16Array | null) {
    this.webEdges = edges;
    this.buildWeb();
    this.refreshHighlights();
  }

  setMarks(marks: Mark[]) {
    this.lastMarks = marks;
    const p: number[] = [];
    const c: number[] = [];
    for (const m of marks) {
      const v = this.neuronPos(m.layer, m.n);
      const k = toward(m.sign === "-" ? INKS.down : INKS.up, m.w ?? 1);
      p.push(v.x, v.y, v.z);
      c.push(k.r, k.g, k.b);
    }
    this.markGeo.setAttribute("position", new THREE.Float32BufferAttribute(p, 3));
    this.markGeo.setAttribute("color", new THREE.Float32BufferAttribute(c, 3));
    this.render();
  }

  setNotes(items: MapNote[]) {
    for (const a of this.notes) a.el.remove();
    this.notes = items.map((it) => {
      const el = document.createElement("div");
      el.className = "atlas-note";
      const t = document.createElement("span");
      t.textContent = it.text || `L${it.layer + 1} N${it.n}`;
      el.appendChild(t);
      this.notesDiv.appendChild(el);
      return { layer: it.layer, n: it.n, el };
    });
    this.render();
  }

  setInsets(left: number, right: number) {
    this.insets = { left, right };
    this.resize();
  }

  hover(layer: number | null, n?: number) {
    this.hovered = layer == null || n == null ? null : { layer, n };
    this.refreshHighlights();
  }

  dispose() {
    this.disposed = true;
    this.ro.disconnect();
    this.canvas.removeEventListener("pointermove", this.onMove);
    this.canvas.removeEventListener("pointerleave", this.onLeave);
    this.canvas.removeEventListener("pointerdown", this.onDown);
    this.canvas.removeEventListener("pointerup", this.onUp);
    for (const a of this.notes) a.el.remove();
    this.controls.dispose();
    this.scene.traverse((o) => {
      const m = o as THREE.Mesh;
      m.geometry?.dispose?.();
      const mat = m.material as THREE.Material | THREE.Material[] | undefined;
      if (Array.isArray(mat)) mat.forEach((x) => x.dispose());
      else mat?.dispose?.();
    });
    this.renderer.dispose();
  }
}

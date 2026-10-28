/**
 * The push (after the explorer's Chat tab): the model answers the person three times in one batch, the open neuron
 * pushed up, left as it is, and pushed down, each voice keeping its own earlier replies (POST /api/atlas/push, served as
 * events). "add" pushes along the neuron's own write direction by a share of the stream's usual length leaving its
 * layer; "×" multiplies its activation. The voices are written greedily, the census's reading, unless "sampled" is
 * chosen: then the instrument's own sampler draws them at the temperature and seed shown, the three sharing each draw,
 * and each turn says how it was written (push.ts). "reads you" shows instead where in a text of the person's the neuron
 * fires.
 */
import React, { useEffect, useRef, useState } from "react";
import { create } from "zustand";
import { Rubric } from "../../components/kit";
import { href } from "../../router";
import { AtlasError, post, streamLines, type ReadsReply } from "./api";
import { ErrorLine, FaceLine, overCeiling } from "./parts";
import { MAX_SEED, MAX_TEMPERATURE, MIN_TEMPERATURE, WRITING_DEFAULT, newSeed, pushBody, writtenAs, type Sampling, type Writing } from "./push";
import { useAtlas } from "./state";

const ADD_STEPS = [...Array.from({ length: 30 }, (_, i) => +((i + 1) * 0.05).toFixed(2)), 1.75, 2, 2.5, 3, 4, 5];
const MUL_STEPS = [...Array.from({ length: 9 }, (_, i) => i * 0.5), 5, 6, 8, 12, 16];
const ROWS = ["up", "zero", "down"] as const;

interface Turn {
  user: string;
  replies: [string, string, string];
  dose: number;
  mode: "add" | "mul";
  /** how it was asked to be written, and how the server says it was (its last event) */
  asked: Writing;
  sampling: Sampling | null;
  live: boolean;
  error: string | null;
}

interface PushState {
  key: string | null;
  thread: Turn[];
  running: AbortController | null;
  mode: "add" | "mul";
  dose: { add: number; mul: number };
  writing: Writing;
  dir: "writes" | "reads";
  reads: (ReadsReply & { text: string }) | null;
  readsError: string | null;
  reading: boolean;
}

const usePush = create<PushState>(() => ({
  key: null,
  thread: [],
  running: null,
  mode: "add",
  dose: { add: 0.8, mul: 2 },
  writing: WRITING_DEFAULT,
  dir: "writes",
  reads: null,
  readsError: null,
  reading: false,
}));

const keyOf = () => {
  const f = useAtlas.getState().focus;
  return f && f.kind === "neuron" ? `${f.layer}/${f.n}` : null;
};

// another neuron opened: its push starts afresh, and what was being written for the last one stops
useAtlas.subscribe((s, prev) => {
  if (s.focus === prev.focus) return;
  const k = keyOf();
  const p = usePush.getState();
  if (k === p.key) return;
  p.running?.abort();
  usePush.setState({ key: k, thread: [], running: null, reads: null, readsError: null });
});

const labels = (t: Turn) => (t.mode === "mul" ? [`▲ ×${t.dose}`, "×1", `▼ ×−${t.dose}`] : [`▲ +${t.dose}`, "0", `▼ −${t.dose}`]);

async function runPush(text: string) {
  const f = useAtlas.getState().focus;
  if (!f || f.kind !== "neuron") return;
  const p = usePush.getState();
  const dose = p.mode === "mul" ? p.dose.mul : p.dose.add;
  const history = p.thread.filter((t) => !t.error && !t.live).map((t) => ({ user: t.user, replies: t.replies }));
  const turn: Turn = { user: text, replies: ["", "", ""], dose, mode: p.mode, asked: p.writing, sampling: null, live: true, error: null };
  const ctl = new AbortController();
  const thread = [...p.thread, turn];
  usePush.setState({ thread, running: ctl, key: keyOf() });
  let cur = turn;
  // the turn's own entry, while it is still in the thread (another neuron opened clears it)
  const update = (patch: Partial<Turn>) => {
    const s = usePush.getState();
    const i = s.thread.indexOf(cur);
    if (i < 0) return;
    cur = { ...cur, ...patch };
    const next = s.thread.slice();
    next[i] = cur;
    usePush.setState({ thread: next });
  };
  try {
    const r = await fetch("/api/atlas/push", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: ctl.signal,
      body: JSON.stringify(pushBody(f, text, dose, p.mode, history, p.writing)),
    });
    if (!r.ok) {
      let d = `HTTP ${r.status}`;
      try {
        d = (await r.json())?.detail ?? d;
      } catch {
        // as is
      }
      throw new AtlasError(typeof d === "string" ? d : JSON.stringify(d), r.status);
    }
    const replies: [string, string, string] = ["", "", ""];
    let last = 0;
    for await (const ev of streamLines(r, true)) {
      const e = ev as { row: number | null; t?: string; error?: string; sampling?: Sampling };
      if (e.error) update({ error: e.error });
      if (e.row == null) {
        if (e.sampling) update({ sampling: e.sampling });
        continue;
      }
      replies[e.row] += e.t ?? "";
      // a frame's worth of text at a time
      const now = performance.now();
      if (now - last > 40) {
        last = now;
        update({ replies: [...replies] as [string, string, string] });
      }
    }
    update({ replies: [...replies] as [string, string, string] });
  } catch (e) {
    if ((e as Error)?.name !== "AbortError") update({ error: (e as Error).message ?? String(e) });
  } finally {
    update({ live: false });
    if (usePush.getState().running === ctl) usePush.setState({ running: null });
  }
}

async function runReads(text: string) {
  const f = useAtlas.getState().focus;
  if (!f || f.kind !== "neuron" || usePush.getState().reading) return;
  usePush.setState({ reading: true, readsError: null });
  try {
    const r = await post<ReadsReply>("/api/atlas/reads", { layer: f.layer, n: f.n, text, chat: true });
    if (keyOf() === `${f.layer}/${f.n}`) usePush.setState({ reads: { ...r, text } });
  } catch (e) {
    usePush.setState({ readsError: (e as Error).message });
  } finally {
    usePush.setState({ reading: false });
  }
}

function step(dir: 1 | -1) {
  const p = usePush.getState();
  const steps = p.mode === "mul" ? MUL_STEPS : ADD_STEPS;
  const v = p.dose[p.mode];
  const next = dir > 0 ? steps.find((x) => x > v + 1e-9) : [...steps].reverse().find((x) => x < v - 1e-9);
  if (next !== undefined) usePush.setState({ dose: { ...p.dose, [p.mode]: next } });
}

const Words: React.FC = () => {
  const f = useAtlas((s) => s.focus);
  const lens = useAtlas((s) => s.lens);
  const unit = useAtlas((s) => s.unit);
  if (!f || f.kind !== "neuron" || !lens || lens.layer !== f.layer || lens.n !== f.n) return null;
  const census = unit && unit.layer === f.layer && unit.n === f.n ? unit.stats?.words.map((w) => w[0]) : null;
  return (
    <div className="atlas-plens">
      {census?.length ? (
        <div className="atlas-cwords">
          <span className="atlas-src">census</span>
          {census.slice(0, 12).join("  ")}
        </div>
      ) : null}
      <div>
        <FaceLine layer={f.layer} face="+" w={{ ...lens.faces["+"], steer: lens.steer?.up ?? null }} k={10} />
      </div>
      <div>
        <FaceLine layer={f.layer} face="-" w={{ ...lens.faces["-"], steer: lens.steer?.down ?? null }} k={10} />
      </div>
    </div>
  );
};

const Thread: React.FC = () => {
  const thread = usePush((s) => s.thread);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 120) el.scrollTop = el.scrollHeight;
  }, [thread]);
  return (
    <div className="atlas-thread" ref={ref}>
      {thread.map((t, i) => (
        <div key={i} className="atlas-turn">
          <div className="atlas-user">
            {t.user}
            <span className="atlas-recipe">
              {t.mode === "mul" ? "×" : "add"} {t.dose} · {writtenAs(t.sampling, t.asked)}
            </span>
          </div>
          <div className="atlas-cols">
            {ROWS.map((c, r) => (
              <div key={c} className={`atlas-col ${c}${t.live ? " live" : ""}`}>
                <span className="atlas-lab">{labels(t)[r]}</span>
                {t.replies[r]}
              </div>
            ))}
          </div>
          <ErrorLine error={t.error} />
        </div>
      ))}
    </div>
  );
};

/** What an add push of this dose is worth: the most the neuron can ever write by itself, and where the push's unit comes from. */
const DoseNote: React.FC<{ dose: number }> = ({ dose }) => {
  const f = useAtlas((s) => s.focus);
  const lens = useAtlas((s) => s.lens);
  const c = f && lens && lens.layer === f.layer && lens.n === f.n ? lens.ceiling : null;
  const over = overCeiling(dose, c);
  if (over == null || !c) return null;
  return (
    <p className="muted atlas-small" style={{ margin: "2px 0 0" }}>
      {dose} of the stream’s usual length is {over < 1 ? "less than" : `${over.toFixed(1)}×`} the most this neuron can ever write by itself
      {c.source && c.source.includes("provisional") ? ` (the usual length read over ${c.source})` : ""}.
    </p>
  );
};

/** Greedy or sampled; sampled, the temperature and the seed, shown and editable ("new seed" draws another; a blank seed
 * is drawn by the server, and the turn shows it). */
const WritingFields: React.FC = () => {
  const w = usePush((s) => s.writing);
  const [temp, setTemp] = useState<string | null>(null);
  const set = (patch: Partial<Writing>) => usePush.setState({ writing: { ...usePush.getState().writing, ...patch } });
  const fits = (t: string) => {
    const v = Number(t.replace(",", "."));
    return t.trim() !== "" && Number.isFinite(v) && v >= MIN_TEMPERATURE && v <= MAX_TEMPERATURE;
  };
  return (
    <div className="row" style={{ gap: 10, alignItems: "center" }}>
      <span className="atlas-seg" role="group" aria-label="how the three voices are written">
        {([false, true] as const).map((on) => (
          <button
            key={String(on)}
            type="button"
            aria-pressed={w.sampled === on}
            title={on ? "drawn by chance with the instrument's sampler, the three voices sharing each draw" : "its likeliest word-piece every time, as the census reads a push"}
            onClick={() => set(on && w.seed == null ? { sampled: true, seed: newSeed() } : { sampled: on })}
          >
            {on ? "sampled" : "greedy"}
          </button>
        ))}
      </span>
      {w.sampled ? (
        <>
          <label className="atlas-small">
            T{" "}
            <input
              className="input-num"
              inputMode="decimal"
              aria-label={`the temperature, from ${MIN_TEMPERATURE} to ${MAX_TEMPERATURE}`}
              aria-invalid={temp !== null && !fits(temp) ? true : undefined}
              value={temp ?? String(w.temperature)}
              onChange={(e) => {
                setTemp(e.target.value);
                if (fits(e.target.value)) set({ temperature: Number(e.target.value.replace(",", ".")) });
              }}
              onBlur={() => setTemp(null)}
              style={{ width: "3.2em", textAlign: "center" }}
            />
          </label>
          <label className="atlas-small">
            seed{" "}
            <input
              className="input-num"
              inputMode="numeric"
              pattern="[0-9]*"
              maxLength={16}
              placeholder="drawn"
              aria-label="the seed: blank, and the server draws one"
              value={w.seed ?? ""}
              onChange={(e) => {
                const d = e.target.value.replace(/\D/g, "");
                set({ seed: d ? Math.min(Number(d), MAX_SEED) : null });
              }}
              style={{ width: "7.5em", textAlign: "center" }}
            />
          </label>
          <button type="button" className="btn-link" onClick={() => set({ seed: newSeed() })}>
            new seed
          </button>
        </>
      ) : null}
    </div>
  );
};

const Reads: React.FC = () => {
  const reads = usePush((s) => s.reads);
  const err = usePush((s) => s.readsError);
  if (err) return <ErrorLine error={err} />;
  if (!reads) return null;
  const m = Math.max(1e-6, ...reads.units.map((u) => Math.abs(u.act)));
  const range = reads.act_range;
  return (
    <div className="atlas-reads">
      <div>
        {reads.units.map((u, i) => {
          const a = u.act / m;
          const bg = a >= 0 ? `rgba(160, 122, 46, ${(a * 0.7).toFixed(3)})` : `rgba(46, 74, 125, ${(-a * 0.7).toFixed(3)})`;
          return (
            <span key={i} style={{ background: bg, color: Math.abs(a) > 0.6 ? "var(--vellum)" : undefined }} title={`${u.act.toFixed(3)} · ${u.pieces.map((v) => v.toFixed(3)).join(" · ")}`}>
              {u.text}
            </span>
          );
        })}
      </div>
      <p className="muted atlas-small">
        strongest here {m.toFixed(2)}
        {range ? ` · its usual extremes +${range.act_max.toFixed(2)} / ${range.act_min.toFixed(2)} over ${range.over_tokens} word-pieces of ${range.source}` : ""}
      </p>
    </div>
  );
};

export const PushTab: React.FC = () => {
  const f = useAtlas((s) => s.focus);
  const mode = usePush((s) => s.mode);
  const dose = usePush((s) => s.dose);
  const dir = usePush((s) => s.dir);
  const running = usePush((s) => s.running);
  const reading = usePush((s) => s.reading);
  const [text, setText] = useState("");
  const [typed, setTyped] = useState<string | null>(null);
  if (!f || f.kind !== "neuron") return null;
  const shown = typed ?? (mode === "mul" ? `×${dose.mul}` : String(dose.add));
  const placeholder = dir === "reads" ? "a text for it to read" : "Who are you?";
  const go = (e: React.FormEvent) => {
    e.preventDefault();
    if (running) return running.abort();
    const t = text.trim() || placeholder;
    setText("");
    if (dir === "reads") void runReads(t);
    else void runPush(t);
  };
  return (
    <div className={`atlas-push${dir === "reads" ? " reads" : ""}`}>
      <Words />
      <Rubric style={{ fontSize: 15, margin: "4px 0" }}>
        {dir === "reads"
          ? "Write a text: each word is shaded by how strongly this neuron fires there (gilt one way, blue the other)."
          : "Ask the model something: it answers three times at once, with this neuron pushed up, left alone, and pushed down. Each column keeps its own earlier answers."}{" "}
        <a href={href.lexicon("push")}>What a push is.</a>
      </Rubric>
      {dir === "writes" ? <Thread /> : <Reads />}
      <form className="atlas-pform" onSubmit={go}>
        <div className="row" style={{ gap: 10, alignItems: "center" }}>
          {dir === "writes" ? (
            <>
              <span className="atlas-seg">
                {(["add", "mul"] as const).map((m) => (
                  <button key={m} type="button" aria-pressed={mode === m} onClick={() => {
                      setTyped(null);
                      usePush.setState({ mode: m });
                    }}>
                    {m === "add" ? "add" : "×"}
                  </button>
                ))}
              </span>
              <span className="atlas-stepper">
                <button type="button" aria-label="smaller" onClick={() => {
                    setTyped(null);
                    step(-1);
                  }}>
                  −
                </button>
                <input
                  className="input-num"
                  inputMode="decimal"
                  aria-label={mode === "mul" ? "the factor" : "the dose, a share of the stream's usual length"}
                  value={shown}
                  onChange={(e) => setTyped(e.target.value)}
                  onBlur={() => {
                    const v = parseFloat((typed ?? "").replace(/[×x*]/g, ""));
                    if (typed !== null && Number.isFinite(v) && v >= 0) usePush.setState({ dose: { ...dose, [mode]: v } });
                    setTyped(null);
                  }}
                  style={{ width: "4.2em", textAlign: "center" }}
                />
                <button type="button" aria-label="larger" onClick={() => {
                    setTyped(null);
                    step(1);
                  }}>
                  +
                </button>
              </span>
            </>
          ) : null}
          <span className="atlas-seg">
            {(["writes", "reads"] as const).map((d) => (
              <button key={d} type="button" aria-pressed={dir === d} onClick={() => usePush.setState({ dir: d })}>
                {d === "writes" ? "push" : "reads you"}
              </button>
            ))}
          </span>
          {dir === "writes" ? (
            <button
              type="button"
              className="btn-link"
              onClick={() => {
                usePush.getState().running?.abort();
                usePush.setState({ thread: [], running: null });
              }}
            >
              clear
            </button>
          ) : null}
        </div>
        {dir === "writes" && mode === "add" ? <DoseNote dose={dose.add} /> : null}
        {dir === "writes" ? <WritingFields /> : null}
        <div className="row" style={{ gap: 10, flexWrap: "nowrap" }}>
          <input className="input-line" value={text} placeholder={placeholder} onChange={(e) => setText(e.target.value)} aria-label={placeholder} />
          <button type="submit" className="btn-ghost">
            {running ? "stop" : reading ? "…" : "run"}
          </button>
        </div>
      </form>
    </div>
  );
};

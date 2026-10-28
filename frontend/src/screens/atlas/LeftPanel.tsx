import React, { useEffect, useRef, useState } from "react";
import { IfApparatus, LensCaution, Rubric } from "../../components/kit";
import { href } from "../../router";
import { wp } from "../../lib/words";
import { cite, citeHead, noteKey, type RunRow, type SearchRow } from "./api";
import { censusChance, Chips, ErrorLine, FaceLine, NameMark, ProposalLine } from "./parts";
import {
  closeFocus,
  goToNeuron,
  openHead,
  runView,
  saveNote,
  setLeftTab,
  setQuery,
  theMap,
  useAtlas,
} from "./state";

/** The open neuron's name, kept on the server (backend/field_notes.py) and saved as it is typed. */
const NameField: React.FC<{ layer: number; n: number }> = ({ layer, n }) => {
  const notes = useAtlas((s) => s.notes);
  const entry = notes?.neurons[noteKey(layer, n)];
  const [name, setName] = useState(entry?.name ?? "");
  const [saved, setSaved] = useState<string | null>(null);
  const [focused, setFocused] = useState(false);
  const timer = useRef<number | undefined>(undefined);
  const key = noteKey(layer, n);
  const shown = useRef(key);
  // the kept name comes into the box when another neuron opens, or when the box is not being typed in (a saved name
  // comes back trimmed, which must not move the words under the reader's cursor)
  useEffect(() => {
    if (shown.current !== key || !focused) {
      shown.current = key;
      setName(entry?.name ?? "");
    }
  }, [key, entry?.name, focused]);
  const save = (v: string) => {
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(async () => {
      try {
        await saveNote(layer, n, { name: v.replace(/\s+/g, " ") });
        setSaved("kept");
      } catch (e) {
        setSaved(String((e as Error).message));
      }
      window.setTimeout(() => setSaved(null), 1500);
    }, 600);
  };
  return (
    <div className="atlas-name">
      <input
        className="input-line"
        value={name}
        placeholder="name it"
        aria-label={`a name for ${cite(layer, n)}`}
        onChange={(e) => {
          setName(e.target.value);
          save(e.target.value);
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter") (e.target as HTMLInputElement).blur();
        }}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
      />
      {entry?.name && (entry.name_by ?? entry.by) === "scientist" ? <span className="muted atlas-small">named by the scientist: a guess, not a finding</span> : null}
      {saved ? <span className="muted atlas-small">{saved}</span> : null}
    </div>
  );
};

const FocusHead: React.FC = () => {
  const f = useAtlas((s) => s.focus);
  const unit = useAtlas((s) => s.unit);
  if (!f) return null;
  const s = unit?.stats;
  return (
    <div className="atlas-focus-head">
      <div className="row-between">
        <span className="atlas-id">{f.kind === "head" ? citeHead(f.layer, f.n) : cite(f.layer, f.n)}</span>
        <button type="button" className="btn-link" aria-label="close" onClick={closeFocus}>
          close
        </button>
      </div>
      {f.kind === "neuron" && s ? (
        <div className="muted atlas-small">
          census #{unit?.rank ?? "—"} of {unit?.n_ranked} · mass {(s.mass ?? s.z).toFixed(1)}
          {s.null_pct != null ? ` · above ${(s.null_pct * 100).toFixed(0)}% of controls` : ""}
          {s.null_pct === 1 && unit?.card?.null.above_all_by_chance != null
            ? ` · a lead: about ${unit.card.null.above_all_by_chance.toFixed(0)} of the layer would beat them all by chance`
            : ""}
        </div>
      ) : null}
      {f.kind === "neuron" ? <NameField layer={f.layer} n={f.n} /> : null}
      {f.kind === "neuron" ? <ProposalLine layer={f.layer} n={f.n} part="name" /> : null}
    </div>
  );
};

const NeuronWeb: React.FC = () => {
  const f = useAtlas((s) => s.focus);
  const lens = useAtlas((s) => s.lens);
  const w = useAtlas((s) => s.webSum);
  const nh = useAtlas((s) => s.nheads);
  const webTotal = useAtlas((s) => s.webTotal);
  if (!f || f.kind !== "neuron") return null;
  const mine = (x: { layer: number; n: number } | null) => !!x && x.layer === f.layer && x.n === f.n;
  const search = (t: string) => setQuery(t.trim() || t, "lens");
  return (
    <div className="atlas-net">
      {mine(lens) && lens ? (
        <div className="atlas-wsum">
          <div className="kicker">its words</div>
          <div>
            <FaceLine layer={f.layer} face="+" w={{ ...lens.faces["+"], steer: lens.steer?.up ?? null }} k={10} />
          </div>
          <div>
            <FaceLine layer={f.layer} face="-" w={{ ...lens.faces["-"], steer: lens.steer?.down ?? null }} k={10} />
          </div>
          {lens.steer ? (
            <div className="muted atlas-small">
              <a href={href.lexicon("steer-lens")}>steer lens</a> ({lens.steer.source}) · KL {lens.steer.kl_up} / {lens.steer.kl_down}
            </div>
          ) : (
            <div className="muted atlas-small">no steer lens read for it yet (details)</div>
          )}
          <LensCaution sighting={null} no={f.layer + 1} style={{ fontSize: 14 }} />
        </div>
      ) : null}
      {mine(w) && w?.ready ? (
        <div className="atlas-wsum">
          <div className="kicker">
            its web · {webTotal ?? w.links} links · pooled from the {w.src === "steer" ? "steer lens" : "lens, labels the gate lets through"}
          </div>
          {w.stale ? <div className="atlas-small red">Read before its vault edit: these are its links as they were.</div> : null}
          {(
            [
              ["up", "+", "what drives it up", "upstream"],
              ["up", "-", "what drives it down", "upstream"],
              ["down", "+", "when it fires up, it drives", "downstream"],
              ["down", "-", "when it fires down, it drives", "downstream"],
            ] as const
          ).map(([side, sg, label, where]) => {
            const b = w[side]?.[sg];
            return (
              <React.Fragment key={`${side}${sg}`}>
                <div className="atlas-lab">
                  {label} · {b?.neurons ?? 0} {where}
                </div>
                <Chips list={b?.words} onPick={search} empty={b?.neurons ? "nothing above chance" : "no link reaches this way"} />
              </React.Fragment>
            );
          })}
          {w.gated && (w.gated.down || w.gated.up) ? (
            <div className="muted atlas-small">
              {w.gated.up + w.gated.down} faces left out: the label gate holds their words back
            </div>
          ) : null}
          <Rubric style={{ fontSize: 14, margin: "4px 0 0" }}>
            Weight-only links read from rest: a writer moves a reader when it fires on the face that opens the reader's gate, and pushes it
            toward one face; the strongest as they arrive, counting how Gemma scales the stream at every layer. Words pooled over many
            neurons, where the label gate was only tested on one at a time: leads to check with a push or a read, not circuits found.{" "}
            <a href={href.lexicon("web")}>What the web is.</a>
          </Rubric>
        </div>
      ) : w && mine(w) && !w.ready ? (
        <p className="muted atlas-small">
          {w.missing === "wiring"
            ? "The wiring index is not built, so its web cannot be drawn (scripts/atlas_build.py wiring)."
            : "Its web is drawn, but neither the steer-lens index nor the lens index is built, so there are no words to pool over it (scripts/atlas_build.py lens)."}
        </p>
      ) : null}
      {mine(nh) && nh ? (
        <div className="atlas-wsum">
          <div className="kicker">its heads</div>
          {(["q", "k", "v", "o"] as const).map((r) => (
            <div key={r} className="atlas-lrow">
              <span className="atlas-sgn">{r}</span>
              <span className="atlas-chips">
                {(nh.roles[r] ?? [])
                  .filter((x) => x.z >= 2)
                  .map((x) => (
                    <button
                      key={`${x.layer}/${x.head}`}
                      type="button"
                      className="atlas-chip"
                      onMouseEnter={() => theMap()?.hoverHead(x.layer, x.head)}
                      onMouseLeave={() => theMap()?.hoverHead(null)}
                      onClick={() => void openHead(x.layer, x.head)}
                    >
                      L{x.layer + 1} h{x.heads.length > 1 ? `${x.heads[0]}–${x.heads[x.heads.length - 1]}` : x.head}
                      <i>{x.z}</i>
                    </button>
                  ))}
              </span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
};

const HeadWeb: React.FC = () => {
  const f = useAtlas((s) => s.focus);
  const h = useAtlas((s) => s.headInfo);
  const p = useAtlas((s) => s.headProfile);
  if (!f || f.kind !== "head" || !h || h.layer !== f.layer || h.head !== f.n) return null;
  const role = (r: "q" | "k" | "v" | "o", label: string) => (
    <div className="atlas-wsum" key={r}>
      <div className="kicker">
        {r} · {label}
      </div>
      {h[r].slice(0, 8).map((c) => (
        <div key={`${c.layer}/${c.n}`} className="atlas-nline" onClick={() => goToNeuron(c.layer, c.n)}>
          <span className="atlas-id-small">{cite(c.layer, c.n)}</span>
          <FaceLine layer={c.layer} face="+" w={c.words} k={4} />
          <span className="atlas-z">{c.z.toFixed(0)}</span>
        </div>
      ))}
      <Chips list={h[`${r}_pooled`]?.tokens?.slice(0, 12)} onPick={(t) => setQuery(t, "lens")} empty="no words pooled" />
      {h[`${r}_pooled`]?.tokens?.length ? (
        <div className="muted atlas-small">
          pooled from these neurons’ + faces ({h[`${r}_pooled`]?.src === "steer" ? "the steer lens" : "labels the gate shows"}), a lead
        </div>
      ) : null}
    </div>
  );
  return (
    <div className="atlas-net">
      <p className="muted atlas-small">
        {h.kind === "full" ? "a full-attention layer, its values read from its keys" : "a sliding-window layer"} · key-value head {h.kv_head}
      </p>
      {role("q", "what it looks for")}
      {role("k", "what makes a word-piece findable")}
      {role("v", "what it carries")}
      {role("o", "where it delivers")}
      {p ? (
        <div className="atlas-wsum">
          <div className="kicker">where it looks, over eight texts</div>
          <div className="atlas-small">
            {p.dist_names.map((nm, i) => (
              <span key={nm} style={{ marginRight: 10 }}>
                {nm} {(p.dist[i] * 100).toFixed(0)}%
              </span>
            ))}
          </div>
          {p.pairs.slice(0, 5).map((x, i) => (
            <div key={i} className="atlas-small">
              {wp(x.qt)} ← {wp(x.kt)} <span className="muted">{x.w.toFixed(2)}</span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
};

const SearchRows: React.FC<{ rows: SearchRow[]; held?: SearchRow[]; total: number; tokens: number }> = ({ rows, held, total, tokens }) => (
  <ol className="atlas-rows">
    <li className="atlas-rows-head">
      {total} faces · {tokens} word-pieces matched
      {held?.length ? ` · the words of ${held.length} held back by the label gate (in the apparatus)` : ""}
    </li>
    {rows.map((x) => (
      <SearchRowItem key={`${x.layer}/${x.n}/${x.sign}/${x.src}`} x={x} />
    ))}
    {held?.length ? (
      <IfApparatus>
        <li className="atlas-rows-head">held back by the label gate: their words matched, but the gate leaves them out of the main reading and the map</li>
        {held.map((x) => (
          <SearchRowItem key={`h${x.layer}/${x.n}/${x.sign}/${x.src}`} x={x} />
        ))}
      </IfApparatus>
    ) : null}
  </ol>
);

const SearchRowItem: React.FC<{ x: SearchRow }> = ({ x }) => (
  <li onClick={() => goToNeuron(x.layer, x.n)} onMouseEnter={() => theMap()?.hover(x.layer, x.n)} onMouseLeave={() => theMap()?.hover(null)}>
    <span className="atlas-id-small">{cite(x.layer, x.n)}</span>
    <span className={`atlas-sg ${x.sign === "+" ? "up" : "down"}`}>
      {x.sign === "+" ? "+" : "−"}
      {x.rank + 1}
    </span>
    <span className="atlas-w">
      {x.name ? <NameMark layer={x.layer} n={x.n} name={x.name} /> : null}
      <FaceLine layer={x.layer} face={x.sign} w={x.words} k={5} />
    </span>
  </li>
);

const CensusRows: React.FC = () => {
  const layer = useAtlas((s) => s.layer);
  const units = useAtlas((s) => s.units[s.layer]);
  const arch = useAtlas((s) => s.arch);
  const f = useAtlas((s) => s.focus);
  const [q, setQ] = useState("");
  const [controls, setControls] = useState(false);
  if (!arch?.census[String(layer)]) {
    const refused = arch?.census_refused?.[String(layer)];
    return (
      <p className="muted atlas-small" style={{ padding: "6px 0" }}>
        Layer {layer + 1} has no census yet: no neuron of it has been pushed on the census's questions (scripts/census_run.py --layer {layer},
        counting layers from 0).{refused ? ` A run of it is not read: ${refused}.` : ""}
      </p>
    );
  }
  if (!units) return <p className="muted atlas-small">reading the census…</p>;
  const rnd = units.filter((u) => u.kind === "random").map((u) => u.z);
  const floor = rnd.length ? Math.max(...rnd) : -Infinity;
  const chance = censusChance(units);
  const ql = q.trim().toLowerCase();
  const shown = units.filter(
    (u) => (controls || u.kind === "neuron") && (!ql || u.words.some((w) => w.includes(ql)) || u.name.toLowerCase().includes(ql) || String(u.n) === ql),
  );
  return (
    <>
      <div className="row" style={{ gap: 10, padding: "4px 0" }}>
        <input className="input-line" placeholder="filter by word or name" value={q} onChange={(e) => setQ(e.target.value)} style={{ flex: 1 }} />
        <label className="check atlas-small">
          <input type="checkbox" checked={controls} onChange={(e) => setControls(e.target.checked)} /> controls
        </label>
      </div>
      <ol className="atlas-rows">
        <li className="atlas-rows-head">
          layer {layer + 1}'s census · ranked by the mass of its words against random directions
          {chance
            ? ` · ${chance.above} above every control, about ${chance.chance.toFixed(0)} of them by chance alone · doses not yet calibrated on this model`
            : " · no random controls, so nothing here is set against chance"}
        </li>
        {shown.slice(0, 600).map((u) => (
          <li
            key={`${u.kind}/${u.n}`}
            className={`${u.kind}${u.z <= floor ? " below" : ""}${f && f.kind === "neuron" && f.layer === layer && f.n === u.n && u.kind === "neuron" ? " sel" : ""}`}
            onClick={() => u.kind === "neuron" && goToNeuron(layer, u.n)}
            onMouseEnter={() => u.kind === "neuron" && theMap()?.hover(layer, u.n)}
            onMouseLeave={() => theMap()?.hover(null)}
          >
            <span className="atlas-id-small">
              {u.kind === "random" ? "r" : "N"}
              {u.n}
            </span>
            <span className="atlas-z">{u.z.toFixed(1)}</span>
            <span className="atlas-w">
              {u.name ? <NameMark layer={layer} n={u.n} name={u.name} /> : null}
              {u.words.join(" ")}
            </span>
          </li>
        ))}
      </ol>
    </>
  );
};

const RunRows: React.FC<{ rows: RunRow[]; total: number; provisional: boolean }> = ({ rows, total, provisional }) => (
  <ol className="atlas-rows">
    <li className="atlas-rows-head">
      {total} neurons pass · score = activation over its usual extreme on that side
      {provisional
        ? " · usual extremes read over the study’s 36 texts, provisional until the stats index is built: a neuron those texts never woke scores high on little"
        : ""}
    </li>
    {rows.map((x) => (
      <li key={`${x.layer}/${x.n}`} onClick={() => goToNeuron(x.layer, x.n)} onMouseEnter={() => theMap()?.hover(x.layer, x.n)} onMouseLeave={() => theMap()?.hover(null)}>
        <span className="atlas-id-small">{cite(x.layer, x.n)}</span>
        <span className={`atlas-sg ${x.face === "+" ? "up" : "down"}`}>
          {x.face === "+" ? "" : "−"}
          {x.score.toFixed(1)}×
        </span>
        <span className="atlas-w">
          <span className="atlas-tk">{wp(x.token)}</span> <FaceLine layer={x.layer} face={x.face} w={x.words} k={4} />
        </span>
      </li>
    ))}
  </ol>
);

const RunTab: React.FC = () => {
  const run = useAtlas((s) => s.run);
  const query = useAtlas((s) => s.query);
  const qMode = useAtlas((s) => s.qMode);
  const timer = useRef<number | undefined>(undefined);
  const upd = (patch: Partial<typeof run>) => {
    useAtlas.setState({ run: { ...useAtlas.getState().run, ...patch } });
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => void runView(), 200);
  };
  return (
    <div className="atlas-run">
      <textarea
        className="atlas-paste"
        rows={5}
        placeholder="paste a text: see which neurons fire hardest on each word-piece"
        value={qMode === "fires" ? query : ""}
        onChange={(e) => setQuery(e.target.value, "fires")}
        spellCheck={false}
      />
      <Rubric style={{ fontSize: 14, margin: "2px 0" }}>
        Hover a word-piece to see what fires there, click it to keep it; the map lights the neurons that pass the filters and the links among the
        strongest of them.
      </Rubric>
      {run.busy ? <p className="muted atlas-small">running the text…</p> : null}
      <ErrorLine error={run.error} />
      {run.tokens.length ? (
        <div className="atlas-toks" onMouseLeave={() => upd({ hover: null })}>
          {run.tokens.map((t, i) => (
            <button
              key={i}
              type="button"
              className={run.pin === run.pos[i] ? "on" : ""}
              onMouseEnter={() => upd({ hover: run.pos[i] })}
              onClick={() => upd({ pin: run.pin === run.pos[i] ? -1 : run.pos[i] })}
            >
              {t.replace(/\n/g, "↵") || "␣"}
            </button>
          ))}
        </div>
      ) : null}
      {run.text ? (
        <div className="atlas-filters row" style={{ gap: 8, flexWrap: "wrap" }}>
          <span className="atlas-lab">layers</span>
          <input className="input-num" value={run.lo} inputMode="numeric" aria-label="from layer" onChange={(e) => upd({ lo: Number(e.target.value) || 1 })} />
          <span>–</span>
          <input className="input-num" value={run.hi} inputMode="numeric" aria-label="to layer" onChange={(e) => upd({ hi: Number(e.target.value) || 1 })} />
          <span className="atlas-seg">
            {(["both", "+", "-"] as const).map((s) => (
              <button key={s} type="button" aria-pressed={run.sign === s} onClick={() => upd({ sign: s })}>
                {s === "both" ? "±" : s === "+" ? "+" : "−"}
              </button>
            ))}
          </span>
          <span className="atlas-lab">score ≥</span>
          <input className="input-num" value={run.min} inputMode="decimal" aria-label="smallest score" onChange={(e) => upd({ min: Number(e.target.value) || 0 })} />
          <input className="input-line" placeholder="its words hold…" value={run.q} onChange={(e) => upd({ q: e.target.value })} style={{ width: 130 }} />
        </div>
      ) : null}
      {run.view ? <RunRows rows={run.view.results} total={run.view.total} provisional={run.scaleSource?.kind === "study_texts"} /> : null}
    </div>
  );
};

export const LeftPanel: React.FC = () => {
  const tab = useAtlas((s) => s.leftTab);
  const hidden = useAtlas((s) => s.leftHidden);
  const search = useAtlas((s) => s.search);
  const searchError = useAtlas((s) => s.searchError);
  const qMode = useAtlas((s) => s.qMode);
  if (hidden) return null;
  return (
    <aside className="atlas-panel atlas-left" aria-label="the open neuron's web, the layer's census, the run of a text">
      <FocusHead />
      <nav className="atlas-tabs" role="tablist">
        {(["web", "run"] as const).map((t) => (
          <button key={t} type="button" role="tab" aria-selected={tab === t} onClick={() => setLeftTab(t)}>
            {t === "web" ? "web" : "run a text"}
          </button>
        ))}
        <button type="button" className="atlas-edge" aria-label="hide this panel" onClick={() => useAtlas.setState({ leftHidden: true })}>
          ‹
        </button>
      </nav>
      {tab === "web" ? (
        <div className="atlas-pane">
          <HeadWeb />
          <NeuronWeb />
          <ErrorLine error={searchError} />
          {search && qMode === "lens" ? (
            search.ready ? (
              <SearchRows rows={search.results} held={search.held} total={search.total} tokens={search.tokens} />
            ) : (
              <p className="muted atlas-small">Neither the lens index nor the steer-lens index is built, so nothing can be searched yet.</p>
            )
          ) : (
            <CensusRows />
          )}
        </div>
      ) : (
        <div className="atlas-pane">
          <RunTab />
        </div>
      )}
    </aside>
  );
};

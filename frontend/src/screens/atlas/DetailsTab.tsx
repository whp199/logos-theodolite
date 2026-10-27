/**
 * Everything the atlas knows of the open neuron (after the explorer's Details tab): its census statistics, both faces'
 * words under the label gate, its steer lens, its activation range, its strongest weight links each way and the heads
 * linked to it, and the reader's note (kept on the server: backend/field_notes.py).
 */
import React, { useEffect, useRef, useState } from "react";
import { HeldBackWords } from "../../components/HeldBack";
import { Apparatus, IfApparatus, LensCaution, Rubric } from "../../components/kit";
import { heldTag } from "../../lib/labels";
import { wp } from "../../lib/words";
import { href } from "../../router";
import { cite, get, noteKey, post, type HeadInfo, type Link, type SteerLens } from "./api";
import { ErrorLine, FaceLine, gateOf, words } from "./parts";
import { goToNeuron, hoverCache, openHead, remember, saveNote, theMap, useAtlas } from "./state";

const CensusStats: React.FC = () => {
  const f = useAtlas((s) => s.focus);
  const unit = useAtlas((s) => s.unit);
  if (!f || !unit || unit.layer !== f.layer || unit.n !== f.n || !unit.stats) return null;
  const s = unit.stats;
  return (
    <section className="atlas-sec">
      <div className="kicker">the census · words its pushes brought out, against random directions</div>
      <div className="atlas-crumbs">
        {s.words.map(([w, c, z]) => (
          <span key={w} className="atlas-tok" title={`${c} answers · z ${z.toFixed(1)} · ${(s.per_prompt[w] ?? []).join(" ")}`}>
            {w}
          </span>
        ))}
      </div>
      {s.shift_up?.length || s.shift_down?.length ? (
        <div className="atlas-small" style={{ marginTop: 4 }}>
          <span className="muted">shift of the answers' words · </span>
          <span className="atlas-sgn up">+</span> {(s.shift_up ?? []).map((x) => x[0]).join("  ")}{" "}
          <span className="atlas-sgn down">−</span> {(s.shift_down ?? []).map((x) => x[0]).join("  ")}
        </div>
      ) : null}
      <table className="atlas-dtable">
        <thead>
          <tr>
            <th className="num">dose</th>
            <th className="num">push</th>
            <th className="num">distinct-4</th>
            <th className="num">cut off</th>
            <th className="num">log-prob</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(s.dose).map(([d, v]) => (
            <tr key={d}>
              <td className="num">{d}</td>
              <td className="num">{v.rel_push.toFixed(2)}</td>
              <td className="num">{v.distinct4.toFixed(2)}</td>
              <td className="num">{(v.capped * 100).toFixed(0)}%</td>
              <td className="num">{v.logprob.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <IfApparatus>
        <p className="muted atlas-small">
          mass {(s.mass ?? s.z).toFixed(2)} · z {s.z.toFixed(2)}
          {s.null_pct != null ? ` · above ${(s.null_pct * 100).toFixed(0)}% of the random directions` : ""}
          {s.shift != null ? ` · shift ${s.shift.toFixed(3)}` : ""} · push = the push's length over the stream's
        </p>
      </IfApparatus>
    </section>
  );
};

const Faces: React.FC = () => {
  const f = useAtlas((s) => s.focus);
  const lens = useAtlas((s) => s.lens);
  if (!f || f.kind !== "neuron") return null;
  if (!lens || lens.layer !== f.layer || lens.n !== f.n) return <p className="muted atlas-small">reading its words…</p>;
  const gp = gateOf(f.layer, lens.faces["+"].check);
  const gm = gateOf(f.layer, lens.faces["-"].check);
  // both faces held back for the same reason: the reason is said once, under the first
  const once = gp.heldBack && gm.heldBack && heldTag(gp) === heldTag(gm);
  const face = (sg: "+" | "-") => {
    const w = lens.faces[sg];
    const gate = sg === "+" ? gp : gm;
    return (
      <div className={`atlas-face-box ${sg === "+" ? "up" : "down"}`}>
        <div className="atlas-lab">{sg === "+" ? "+ face · what it writes when it fires positive" : "− face · when it fires negative"}</div>
        {gate.heldBack ? (
          <HeldBackWords gate={gate} words={w.lens} eye={w.eye} size={17} compact={once && sg === "-"} />
        ) : (
          <>
            <div className="atlas-crumbs">
              {w.lens.slice(0, 16).map((t, i) => (
                <span key={i} className="atlas-tok" title={JSON.stringify(t)}>
                  {wp(t)}
                </span>
              ))}
            </div>
            <IfApparatus>
              <div className="atlas-small muted">naked eye · {words(w.eye, 12)}</div>
            </IfApparatus>
          </>
        )}
      </div>
    );
  };
  return (
    <section className="atlas-sec">
      <div className="kicker">its words through the lens</div>
      {lens.edited ? <p className="atlas-small red">Edited in the vault: these words are read from the edited weights.</p> : null}
      <div className="atlas-two">
        {face("+")}
        {face("-")}
      </div>
      <LensCaution sighting={null} no={f.layer + 1} style={{ fontSize: 15 }} />
      <Apparatus title="the gate" sources={false}>
        <p style={{ margin: 0 }}>
          The gate's read direction through the lens, not checked by the label gate:{" "}
          <span className="muted">{words(lens.gate_trigger, 12)}</span>
        </p>
      </Apparatus>
    </section>
  );
};

const Steer: React.FC = () => {
  const f = useAtlas((s) => s.focus);
  const lens = useAtlas((s) => s.lens);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (!f || f.kind !== "neuron" || !lens || lens.layer !== f.layer || lens.n !== f.n) return null;
  const st = lens.steer;
  const read = async () => {
    setBusy(true);
    setError(null);
    try {
      const r = await post<SteerLens>(`/api/atlas/neuron/steer_lens?layer=${f.layer}&n=${f.n}`, {});
      const cur = useAtlas.getState().lens;
      if (cur && cur.layer === f.layer && cur.n === f.n) useAtlas.setState({ lens: { ...cur, steer: r } });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="atlas-sec">
      <div className="kicker">its steer lens · the next word-piece's rise when it is pushed</div>
      {st ? (
        <div className="atlas-two">
          <div className="atlas-face-box up">
            <div className="atlas-lab">+ push · KL {st.kl_up}</div>
            <div className="atlas-crumbs">
              {st.up.map((t, i) => (
                <span key={i} className="atlas-tok">
                  {wp(t)}
                </span>
              ))}
            </div>
          </div>
          <div className="atlas-face-box down">
            <div className="atlas-lab">− push · KL {st.kl_down}</div>
            <div className="atlas-crumbs">
              {st.down.map((t, i) => (
                <span key={i} className="atlas-tok">
                  {wp(t)}
                </span>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <p className="atlas-small">
          <span className="muted">Not in the steer-lens index. </span>
          <button type="button" className="btn-link" disabled={busy} onClick={() => void read()}>
            {busy ? "reading…" : "read it now (runs the model)"}
          </button>
        </p>
      )}
      <ErrorLine error={error} />
      <Rubric style={{ fontSize: 15 }}>
        A measured effect, not a label: the neuron is pushed by {st ? st.dose : 0.8} of the stream's usual length at every word-piece of four short texts,
        and these are the word-pieces whose odds rose most next. It is not held back by the label gate.{" "}
        <a href={href.lexicon("steer-lens")}>What the steer lens is.</a>
      </Rubric>
      <IfApparatus>
        {st ? <p className="muted atlas-small">from the {st.source === "index" ? "steer-lens index" : "model, read just now"} · dose {st.dose}</p> : null}
      </IfApparatus>
    </section>
  );
};

const Range: React.FC = () => {
  const f = useAtlas((s) => s.focus);
  const lens = useAtlas((s) => s.lens);
  if (!f || !lens || lens.layer !== f.layer || lens.n !== f.n || !lens.act_range) return null;
  const r = lens.act_range;
  const g = r.regimes;
  return (
    <section className="atlas-sec">
      <div className="kicker">how far it fires</div>
      <p className="atlas-small" style={{ margin: 0 }}>
        from <b>{r.act_min.toFixed(2)}</b> to <b>+{r.act_max.toFixed(2)}</b> over {r.over_tokens.toLocaleString()} word-pieces of {r.source}
      </p>
      {g ? (
        <IfApparatus>
          <p className="muted atlas-small" style={{ margin: "2px 0 0" }}>
            GeGLU's four sign regimes, gelu(gate) × up: positive side +gate +up {g.pp.toFixed(2)}, −gate −up {g.nn.toFixed(2)} (the double
            negative); negative side +gate −up {g.pn.toFixed(2)}, −gate +up {g.np.toFixed(2)}
          </p>
        </IfApparatus>
      ) : null}
    </section>
  );
};

const LinkRows: React.FC<{ list: Link[]; title: string }> = ({ list, title }) => (
  <div>
    <div className="atlas-lab">{title}</div>
    <table className="atlas-dtable atlas-links">
      <tbody>
        {list.slice(0, 14).map((c) => {
          const face = c.sign > 0 ? "+" : "-";
          return (
            <tr
              key={`${c.layer}/${c.n}`}
              onClick={() => goToNeuron(c.layer, c.n)}
              onMouseEnter={() => theMap()?.hover(c.layer, c.n)}
              onMouseLeave={() => theMap()?.hover(null)}
            >
              <td className={`atlas-pole ${face === "+" ? "up" : "down"}`}>{face === "+" ? "↑" : "↓"}</td>
              <td>
                <FaceLine layer={c.layer} face={face} w={c.words} k={4} />
                {c.name || c.census?.length ? <div className="muted atlas-small">{c.name || c.census?.join(" ")}</div> : null}
              </td>
              <td className="num muted">{c.z.toFixed(0)}</td>
              <td className="atlas-id-small">{cite(c.layer, c.n)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  </div>
);

const Wiring: React.FC = () => {
  const f = useAtlas((s) => s.focus);
  const w = useAtlas((s) => s.wiring);
  const err = useAtlas((s) => s.wiringError);
  if (!f || f.kind !== "neuron") return null;
  return (
    <section className="atlas-sec">
      <div className="kicker">its wiring · the strongest weight links each way</div>
      <ErrorLine error={err} />
      {w && w.layer === f.layer && w.n === f.n ? (
        <>
          <div className="atlas-two">
            <LinkRows list={w.down} title="writes into (later layers)" />
            <LinkRows list={w.up} title="driven by (earlier layers)" />
          </div>
          <Rubric style={{ fontSize: 15 }}>
            Links through the weights alone (the direct path through the stream, with the norms in between taken at their usual scale): an arrow up
            means the link drives that neuron's + face, down its − face. Leads to test with a push, not a circuit found.
          </Rubric>
          <IfApparatus>
            <p className="muted atlas-small">{w.source === "index" ? "from the wiring index" : "read from the weights just now"} · {w.note}</p>
          </IfApparatus>
        </>
      ) : !err ? (
        <p className="muted atlas-small">reading its links…</p>
      ) : null}
    </section>
  );
};

/** The heads linked to the open neuron, each with what it writes and carries (read once, kept for the page). */
const Heads: React.FC = () => {
  const f = useAtlas((s) => s.focus);
  const nh = useAtlas((s) => s.nheads);
  const [, bump] = useState(0);
  const gen = useRef(0);
  const mine = !!f && f.kind === "neuron" && !!nh && nh.layer === f.layer && nh.n === f.n;
  const rows = (roles: ("q" | "k" | "v" | "o")[]) =>
    mine && nh
      ? roles
          .flatMap((r) => (nh.roles[r] ?? []).map((x) => ({ ...x, role: r })))
          .filter((x) => x.z >= 2)
          .sort((a, b) => b.z - a.z)
          .slice(0, 10)
      : [];
  const reading = rows(["q", "k", "v"]);
  const feeding = rows(["o"]);
  useEffect(() => {
    if (!mine) return;
    const g = ++gen.current;
    void (async () => {
      // one at a time, the strongest first: each reads the weights on the GPU
      for (const x of [...reading.slice(0, 6), ...feeding.slice(0, 6)]) {
        const key = `h/${x.layer}/${x.head}`;
        if (hoverCache.has(key)) continue;
        try {
          remember(key, await get<HeadInfo>("/api/atlas/head", { layer: x.layer, head: x.head }));
        } catch {
          return;
        }
        if (g !== gen.current) return;
        bump((v) => v + 1);
      }
    })();
    return () => {
      gen.current++;
    };
  }, [mine, nh]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!mine) return null;
  const table = (list: typeof reading, title: string) => (
    <div>
      <div className="atlas-lab">{title}</div>
      <table className="atlas-dtable atlas-links">
        <tbody>
          {list.map((x) => {
            const info = hoverCache.get(`h/${x.layer}/${x.head}`) as HeadInfo | undefined;
            const id = `L${x.layer + 1} h${x.heads.length > 1 ? `${x.heads[0]}–${x.heads[x.heads.length - 1]}` : x.head}`;
            return (
              <tr
                key={`${x.role}/${x.layer}/${x.head}`}
                onClick={() => void openHead(x.layer, x.head)}
                onMouseEnter={() => theMap()?.hoverHead(x.layer, x.head)}
                onMouseLeave={() => theMap()?.hoverHead(null)}
              >
                <td className="atlas-pole">{x.role}</td>
                <td>
                  {info ? (
                    <>
                      <div className="atlas-small">{info.o_pooled.tokens.slice(0, 4).map((t) => t.tok).join("  ")}</div>
                      <div className="muted atlas-small">{info.v_pooled.tokens.slice(0, 4).map((t) => t.tok).join("  ")}</div>
                    </>
                  ) : null}
                </td>
                <td className="num muted">{x.z.toFixed(1)}</td>
                <td className="atlas-id-small">{id}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
  return (
    <section className="atlas-sec">
      <div className="kicker">its heads</div>
      <div className="atlas-two">
        {table(reading, "heads reading it (q, k, v)")}
        {table(feeding, "heads feeding it (o)")}
      </div>
      <Rubric style={{ fontSize: 15 }}>
        Later heads whose query, key or value reads what it writes, and earlier heads whose output it reads, by the weights. Each head shows what it
        writes, and below, faint, what it carries.
      </Rubric>
    </section>
  );
};

const NoteField: React.FC = () => {
  const f = useAtlas((s) => s.focus);
  const notes = useAtlas((s) => s.notes);
  const entry = f ? notes?.neurons[noteKey(f.layer, f.n)] : undefined;
  const [note, setNote] = useState(entry?.note ?? "");
  const [state, setState] = useState<string | null>(null);
  const [focused, setFocused] = useState(false);
  const key = f ? noteKey(f.layer, f.n) : "";
  const shown = useRef(key);
  useEffect(() => {
    if (shown.current !== key || !focused) {
      shown.current = key;
      setNote(entry?.note ?? "");
    }
  }, [key, entry?.note, focused]);
  if (!f || f.kind !== "neuron") return null;
  const save = async () => {
    if ((entry?.note ?? "") === note) return;
    try {
      const r = await saveNote(f.layer, f.n, { note });
      setState(r.warning ?? "kept");
    } catch (e) {
      setState((e as Error).message);
    }
    window.setTimeout(() => setState(null), 2000);
  };
  return (
    <section className="atlas-sec">
      <div className="kicker">your note</div>
      <textarea
        className="input-box atlas-note-box"
        rows={3}
        value={note}
        placeholder="what you found, what to try next"
        onChange={(e) => setNote(e.target.value)}
        onFocus={() => setFocused(true)}
        onBlur={() => {
          setFocused(false);
          void save();
        }}
      />
      <p className="muted atlas-small" style={{ margin: 0 }}>
        {state ?? (entry ? `${entry.by === "scientist" ? "the scientist" : "you"}, ${new Date(entry.at).toLocaleString()}` : "kept on the server, beside the neuron, when you leave the box")}
      </p>
      <IfApparatus>{notes ? <p className="muted atlas-small mono-wrap">{notes.file}</p> : null}</IfApparatus>
    </section>
  );
};

export const DetailsTab: React.FC = () => {
  const f = useAtlas((s) => s.focus);
  if (!f || f.kind !== "neuron") return null;
  return (
    <div className="atlas-details">
      <CensusStats />
      <Faces />
      <Steer />
      <Range />
      <Wiring />
      <Heads />
      <NoteField />
    </div>
  );
};

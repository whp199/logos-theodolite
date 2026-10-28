/**
 * The open neuron's stored passages (GET /api/atlas/neuron/passages, the passages index built by scripts/atlas_build.py
 * passages): for each face its strongest windows of the stats corpus, one event a window, the peak word-piece in gilt,
 * with its share of the face's extreme and the regime GeGLU fired in there; then a few events drawn at random above
 * half its extreme; and the word-pieces each face reads and that follow, with their lift over the corpus's base rate.
 * Those words are counted over a few passages and no test has checked them: they are leads, not labels, and the page
 * says so. "read again" posts a passage's own ids to POST /reads (the model runs) and gives its activation now.
 */
import React, { useEffect, useState } from "react";
import { IfApparatus, Rubric } from "../../components/kit";
import { wp } from "../../lib/words";
import { actAt, get, post, type Passage, type Passages, type PassageWord, type ReadsReply } from "./api";
import { ErrorLine } from "./parts";
import { useAtlas } from "./state";

const REGIME: Record<Passage["regime"], string> = {
  pp: "+gate +up",
  nn: "−gate −up, the double negative",
  pn: "+gate −up",
  np: "−gate +up",
};
const CUT: Record<string, string> = {
  start: "the window's start: the model read nothing before it",
  document: "a document's end mark: what lies beyond is another document",
  turn: "a turn mark of a chat",
  end: "the window's end",
};
const SHOWN = 6;

const Words: React.FC<{ list: PassageWord[] | undefined; title: string }> = ({ list, title }) => (
  <div className="atlas-small">
    <span className="atlas-lab">{title} </span>
    {list && list.length ? (
      list.map((t, i) => (
        <span key={i} className="atlas-tok" title={`${t.n} of the passages · ${t.lift != null ? `${t.lift}× its share of the corpus` : ""}`}>
          {wp(t.tok)} <span className="muted">{t.n}{t.lift != null ? ` · ×${t.lift}` : ""}</span>
        </span>
      ))
    ) : (
      <span className="muted">none</span>
    )}
  </div>
);

const PassageRow: React.FC<{ p: Passage; layer: number; n: number }> = ({ p, layer, n }) => {
  const [now, setNow] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const again = async () => {
    setBusy(true);
    setError(null);
    try {
      setNow(actAt(await post<ReadsReply>("/api/atlas/reads", { layer, n, ids: p.ids }), p.at));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const [before, after] = p.cut;
  return (
    <li className="atlas-passage">
      <div className="atlas-passage-text">
        <span className="muted" title={before ? `begins at ${CUT[before] ?? before}` : "the text goes on before it"}>
          {before ? "" : "…"}
        </span>
        {p.units.map(([text, peak], i) => (
          <span key={i} className={peak ? "atlas-peak" : undefined} title={peak ? `${p.act} here` : undefined}>
            {text}
          </span>
        ))}
        <span className="muted" title={after ? `ends at ${CUT[after] ?? after}` : "the text goes on after it"}>
          {after ? "" : "…"}
        </span>
      </div>
      <div className="muted atlas-small">
        {p.act.toFixed(3)}
        {p.rel != null ? ` · ${Math.round(p.rel * 100)}% of its extreme` : ""} · {REGIME[p.regime]} ·{" "}
        <button type="button" className="btn-link" disabled={busy} onClick={() => void again()}>
          {busy ? "reading…" : "read again"}
        </button>
        {now != null ? ` · now ${now.toFixed(3)}` : ""}
      </div>
      <ErrorLine error={error} />
    </li>
  );
};

const Face: React.FC<{ r: Passages; face: "+" | "-" }> = ({ r, face }) => {
  const [all, setAll] = useState(false);
  const tops = (r.passages ?? []).filter((p) => p.kind === "top");
  const samples = (r.passages ?? []).filter((p) => p.kind === "sample");
  return (
    <section className={`atlas-sec atlas-face-box ${face === "+" ? "up" : "down"}`}>
      <div className="atlas-lab">{face === "+" ? "+ face · where it fires hardest" : "− face · where it fires most negative"}</div>
      {tops.length ? (
        <ol className="atlas-passages">
          {(all ? tops : tops.slice(0, SHOWN)).map((p, i) => (
            <PassageRow key={i} p={p} layer={r.layer} n={r.n} />
          ))}
        </ol>
      ) : (
        <p className="muted atlas-small">It never fired on this face over the corpus.</p>
      )}
      {tops.length > SHOWN ? (
        <button type="button" className="btn-link atlas-small" onClick={() => setAll(!all)}>
          {all ? `the first ${SHOWN}` : `all ${tops.length}`}
        </button>
      ) : null}
      <Words list={r.reads} title="reads" />
      <Words list={r.follows} title="followed by" />
      {samples.length ? (
        <>
          <div className="atlas-lab">drawn at random above half its extreme</div>
          <ol className="atlas-passages">
            {samples.map((p, i) => (
              <PassageRow key={i} p={p} layer={r.layer} n={r.n} />
            ))}
          </ol>
        </>
      ) : null}
    </section>
  );
};

export const PassagesTab: React.FC = () => {
  const f = useAtlas((s) => s.focus);
  const [got, setGot] = useState<{ key: string; faces: Record<"+" | "-", Passages> } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const key = f && f.kind === "neuron" ? `${f.layer}/${f.n}` : "";
  useEffect(() => {
    if (!f || f.kind !== "neuron") return;
    const ctl = new AbortController();
    setError(null);
    void (async () => {
      try {
        const [p, m] = await Promise.all(
          (["+", "-"] as const).map((face) => get<Passages>("/api/atlas/neuron/passages", { layer: f.layer, n: f.n, face, k: 16 }, ctl.signal)),
        );
        setGot({ key, faces: { "+": p, "-": m } });
      } catch (e) {
        if ((e as Error)?.name !== "AbortError") setError((e as Error).message);
      }
    })();
    return () => ctl.abort();
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!f || f.kind !== "neuron") return null;
  if (error) return <ErrorLine error={error} />;
  if (!got || got.key !== key) return <p className="muted atlas-small">reading its passages…</p>;
  const plus = got.faces["+"];
  if (!plus.ready) {
    return (
      <div className="atlas-details">
        <p className="atlas-small">No passages are stored for this neuron yet.</p>
        <p className="muted atlas-small">{plus.note}</p>
      </div>
    );
  }
  return (
    <div className="atlas-details">
      {plus.stale ? <p className="atlas-small red">{plus.stale}</p> : null}
      <Face r={plus} face="+" />
      <Face r={got.faces["-"]} face="-" />
      <Rubric style={{ fontSize: 15 }}>
        Each face’s strongest moments over {plus.tokens?.toLocaleString()} word-pieces of {plus.corpus}, one a window so that one long passage
        cannot fill the list, its peak in gilt. The regime is the sign of GeGLU’s gate and up there (in the study the gate was open at 72–100% of
        strong events). What a face
        reads and what follows it are counted over these {plus.k} passages and set against how common each word-piece is in the corpus:{" "}
        {plus.words_note}.
      </Rubric>
      <IfApparatus>
        <p className="muted atlas-small">
          the passages index · built {plus.built_at} · K {plus.K} a face, one a window, over {plus.windows?.toLocaleString()} windows ·{" "}
          {plus.sampled ? `${plus.sampled} more drawn at random above ${plus.sample_threshold}` : "none drawn at random"} · share of its extreme
          against {plus.rel_to} · “read again” runs the model on the window from its start mark, earlier documents included, as it was read
        </p>
      </IfApparatus>
    </div>
  );
};

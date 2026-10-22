import React from "react";
import { Apparatus, IfApparatus, Missing, Rubric } from "./kit";
import { stopWords } from "./WordsItWrote";
import {
  MAX_WRITE,
  MODEL_SAMPLING,
  PACE,
  SIGHT_CAP,
  about,
  generationWords,
  effectiveThink,
  paceOf,
  samplerPayload,
  useTheodoliteStore,
  type GenerationEcho,
} from "../store/useTheodoliteStore";

/**
 * How it writes: the band under the question that says how the model writes its answer before the instrument sights
 * it. Its likeliest word-piece every time (greedy), as the study wrote the answers it checked the lens on, or drawn by
 * chance as a chat app does; how many
 * word-pieces it may write; and whether each one it writes is sighted, not only the first. The fields are the page's
 * settings for the next sighting; what a sighting was taken with is read from the server's echo (the apparatus, and the
 * word-pieces it wrote), never from here.
 */

/**
 * A number box that keeps what is typed until it is a number in range, and only then sets it. `signed`: the range goes
 * below 0, so the box is a text box with the full keyboard, since a phone's decimal keypad (iOS) has no minus sign;
 * "−" and a decimal comma are read as "-" and ".".
 */
const NumField: React.FC<{
  id: string;
  value: number;
  min: number;
  max: number;
  step: number;
  whole?: boolean;
  signed?: boolean;
  onCommit: (v: number) => void;
  describedBy?: string;
}> = ({ id, value, min, max, step, whole, signed, onCommit, describedBy }) => {
  const [draft, setDraft] = React.useState(String(value));
  const read = (t: string) => Number(signed ? t.replace(/\u2212/g, "-").replace(",", ".") : t);
  // a value set from elsewhere (the model's own settings, a kept seed) replaces the draft
  React.useEffect(() => {
    if (read(draft) !== value) setDraft(String(value));
  }, [value]); // eslint-disable-line react-hooks/exhaustive-deps
  const fits = (t: string) => {
    const n = read(t);
    return t.trim() !== "" && Number.isFinite(n) && n >= min && n <= max && (!whole || Number.isInteger(n));
  };
  return (
    <input
      id={id}
      className="input-num"
      type={signed ? "text" : "number"}
      inputMode={whole ? "numeric" : signed ? "text" : "decimal"}
      {...(signed ? { pattern: "[\\-\u2212]?[0-9]*[.,]?[0-9]*", autoComplete: "off", spellCheck: false } : { min, max, step })}
      value={draft}
      aria-invalid={fits(draft) ? undefined : true}
      aria-describedby={describedBy}
      onChange={(e) => {
        setDraft(e.target.value);
        if (fits(e.target.value)) onCommit(read(e.target.value));
      }}
      onBlur={() => {
        if (!fits(draft)) setDraft(String(value));
      }}
    />
  );
};

/** The seed of the latest answer when the server drew it (none was sent), else null. */
function drawnSeed(g: GenerationEcho | null | undefined): number | null {
  return g && !g.greedy && g.seedDrawn && g.seed != null ? g.seed : null;
}

/** A repeat penalty in words: a positive one holds repeats back, a negative one favours them (as generationWords). */
function repeatWords(p: number): string {
  const by = String(Math.round(Math.abs(p) * 1000) / 1000);
  return p > 0 ? `holding back repeats by ${by}` : `favouring repeats by ${by}`;
}

/** "hold back repeats": the presence penalty, shown by default when drawn by chance, and when greedy shown by default
 * only while it is not 0 (it applies to the likeliest word too), else in the apparatus. */
const PenaltyField: React.FC = () => {
  const w = useTheodoliteStore((s) => s.writing);
  const setW = useTheodoliteStore((s) => s.setWriting);
  return (
    <>
      <label htmlFor="w-pres">hold back repeats</label>
      <NumField id="w-pres" value={w.presencePenalty} min={-2} max={2} step={0.1} signed onCommit={(v) => setW({ presencePenalty: v })} />
      <span className="note">
        taken off the score of each word-piece it has already written, before it chooses the next; your question and its begun
        answer do not count <i>(presence penalty, −2 to 2; 0 is off)</i>
      </span>
    </>
  );
};

const SamplerFields: React.FC = () => {
  const w = useTheodoliteStore((s) => s.writing);
  const setW = useTheodoliteStore((s) => s.setWriting);
  const last = useTheodoliteStore((s) => s.answer);
  const drawn = drawnSeed(last?.wrote ?? last?.generation);
  const own = w.temperature === MODEL_SAMPLING.temperature && w.topK === MODEL_SAMPLING.topK && w.topP === MODEL_SAMPLING.topP;
  return (
    <div className="sampler" role="group" aria-label="How chance draws each word-piece">
      <label htmlFor="w-temp">temperature</label>
      <NumField id="w-temp" value={w.temperature} min={0} max={2} step={0.05} onCommit={(v) => setW({ temperature: v })} />
      <span className="note">
        {w.temperature === 0
          ? "0 is the same as its likeliest word-piece every time"
          : "1 leaves its odds as they are; lower is steadier, higher wilder (0 to 2)"}
      </span>

      <label htmlFor="w-topk">from its likeliest</label>
      <NumField id="w-topk" value={w.topK} min={0} max={262144} step={1} whole onCommit={(v) => setW({ topK: v })} />
      <span className="note">
        word-pieces, and no others <i>(top-k; 0 is no limit)</i>
      </span>

      <label htmlFor="w-topp">within</label>
      <NumField id="w-topp" value={w.topP} min={0.01} max={1} step={0.01} onCommit={(v) => setW({ topP: v })} />
      <span className="note">
        of its odds: only the fewest likeliest that add up to this share <i>(top-p; 1 is no limit)</i>
      </span>

      <PenaltyField />

      <label htmlFor="w-seed">seed</label>
      <input
        id="w-seed"
        className="input-num"
        type="text"
        inputMode="numeric"
        pattern="[0-9]*"
        maxLength={16}
        placeholder="drawn"
        value={w.seed ?? ""}
        onChange={(e) => {
          const t = e.target.value.replace(/\D/g, "");
          setW({ seed: t ? Math.min(Number(t), Number.MAX_SAFE_INTEGER) : null });
        }}
      />
      <span className="note">
        {w.seed == null
          ? "blank: the server draws a new one each time, and it is shown here after"
          : "with the same question, system prompt, begun answer, nudge and settings, the same seed draws the same word-pieces again if the model’s scores come out exactly as before, which a GPU does not promise"}
        {drawn != null && w.seed == null ? (
          <>
            {" "}
            · the last drew <span className="num">{drawn}</span>{" "}
            <button type="button" className="btn-link" onClick={() => setW({ seed: drawn })}>
              keep this seed
            </button>
          </>
        ) : null}
      </span>

      <span />
      <button
        type="button"
        className="btn-link"
        style={{ gridColumn: "2 / -1", justifySelf: "start" }}
        disabled={own}
        onClick={() => setW({ temperature: MODEL_SAMPLING.temperature, topK: MODEL_SAMPLING.topK, topP: MODEL_SAMPLING.topP })}
      >
        the model’s own settings ({MODEL_SAMPLING.temperature}, {MODEL_SAMPLING.topK}, {MODEL_SAMPLING.topP})
      </button>
    </div>
  );
};

/** The band "how it writes", under the question. `apparatus`: also place <WritingApparatus/> at its foot (default). */
export const HowItWrites: React.FC<{ apparatus?: boolean }> = ({ apparatus = true }) => {
  const w = useTheodoliteStore((s) => s.writing);
  const setW = useTheodoliteStore((s) => s.setWriting);
  // what will be sent: with bare text the thinking sighting is left out unless the old imitation is chosen
  const think = useTheodoliteStore((s) => effectiveThink(s));
  const sampled = w.mode === "sampled";
  const pace = paceOf(w, think);
  // a repeat penalty applies with its likeliest word too (before the argmax): the field then stays in view, apparatus or
  // not, so nothing sent is hidden, and stays while it is edited back to 0 (it goes back to the apparatus on the next
  // change of mode)
  const penalised = w.presencePenalty !== 0;
  const [penaltyOut, setPenaltyOut] = React.useState(false);
  React.useEffect(() => {
    if (sampled) setPenaltyOut(false);
    else if (penalised) setPenaltyOut(true);
  }, [sampled, penalised]);
  const repeats = repeatWords(w.presencePenalty);
  return (
    <section className="writes" aria-labelledby="writes-head">
      <span id="writes-head" className="label-sc">
        how it writes
      </span>
      <div className="writes-body">
        <div className="writes-row">
          <div className="writes-modes" role="group" aria-label="How it chooses each word-piece">
            <button type="button" className="btn-ghost plain" aria-pressed={!sampled} onClick={() => setW({ mode: "greedy" })}>
              its likeliest word
            </button>
            <button type="button" className="btn-ghost plain" aria-pressed={sampled} onClick={() => setW({ mode: "sampled" })}>
              drawn by chance
            </button>
          </div>
          <span className="writes-field">
            <label htmlFor="w-length">up to</label>
            <NumField id="w-length" value={w.length} min={1} max={MAX_WRITE} step={1} whole onCommit={(v) => setW({ length: v })} />
            <span>word-pieces</span>
          </span>
          <label className="check">
            <input type="checkbox" checked={w.sightEach} onChange={(e) => setW({ sightEach: e.target.checked })} />
            sight each word-piece it writes
          </label>
        </div>
        <p className="writes-now">
          {sampled
            ? "Drawn by chance, the word-piece it writes need not be its likeliest, and two sightings of one question can go on differently."
            : penalised
              ? `The likeliest word-piece each time after ${repeats}, so not always its own likeliest: the answers the study checked the lens on were written with no repeat penalty.`
              : "Its likeliest word-piece every time, as in the answers the study checked the lens on."}{" "}
          It stops sooner if it ends its turn, or writes another mark that stops the writing.
        </p>
        {sampled ? (
          <SamplerFields />
        ) : penalised || penaltyOut ? (
          <div className="sampler" role="group" aria-label="Repeats, with its likeliest word">
            <PenaltyField />
          </div>
        ) : (
          <IfApparatus>
            <div className="sampler" role="group" aria-label="Repeats, with its likeliest word">
              <PenaltyField />
            </div>
          </IfApparatus>
        )}
        <p className="writes-now num">
          {pace.sighted > 1 ? (
            <>
              At most about {about(pace.all)}: about {about(pace.first)} until the first sighting, then the other {pace.sighted - 1} about ¾ s
              apart.
            </>
          ) : (
            <>At most about {about(pace.first)}.</>
          )}
          {w.sightEach && w.length > SIGHT_CAP ? <> Only the first {SIGHT_CAP} are sighted: at most {SIGHT_CAP} at a time.</> : null}
          {think ? <> The sighting with thinking turned on adds about {about(pace.thinking)}, and is read only where its answer begins.</> : null}
        </p>
        {pace.all > 60 ? (
          <p className="writes-warn">
            That is about {about(pace.all)}. While it writes and sights, the instrument answers nothing else; you can stop it between
            word-pieces.
          </p>
        ) : null}
        <Rubric style={{ margin: 0, fontSize: 16 }}>
          {sampled
            ? "Drawn by chance, each word-piece is picked at random among the model’s likeliest, weighted by its odds, as chat apps do. A kept seed usually draws the same words again."
            : penalised
              ? "With a repeat penalty, the model writes the word-piece that is likeliest once the penalty is taken, which need not be its own likeliest. Set “hold back repeats” to 0 to write as the answers the study checked the lens on were written."
              : "As in the answers the study checked the lens on, the model writes its likeliest word-piece every time. Choose “drawn by chance” to let it write as a chat app does."}{" "}
          {w.sightEach
            ? "Each word-piece it writes then gets a sighting of its own, taken where the model chose it."
            : "Only the moment it chooses its first word-piece is sighted."}
        </Rubric>
        {apparatus ? <WritingApparatus /> : null}
      </div>
    </section>
  );
};

/** What a sighting asks for and what the server did, in the apparatus under "how it writes". */
export const WritingApparatus: React.FC = () => {
  const w = useTheodoliteStore((s) => s.writing);
  const last = useTheodoliteStore((s) => s.answer);
  const sent = {
    max_new_tokens: w.length,
    sampler: samplerPayload(w),
    sight_every_token: w.sightEach,
    max_sighted: SIGHT_CAP,
    stop_at_end_of_turn: true,
  };
  const g = last?.generation ?? null;
  const wrote = last?.wrote ?? null;
  const n = last?.pieces?.length ?? g?.nWritten ?? null;
  const stop = (r: GenerationEcho["stopReason"]) => stopWords(r, last?.pieces?.length ? last.pieces[last.pieces.length - 1] : null);
  return (
    <Apparatus title="how it writes" sources={false}>
      <dl className="dl">
        <dt>the next sends</dt>
        <dd className="mono-wrap">{JSON.stringify(sent)}</dd>
        <dt>the sampler</dt>
        <dd>
          For each word-piece, from the raw scores z of the model’s last layer: the presence penalty, z − p for each piece already
          written (once each; the prompt’s pieces do not count); then z / T; then only the k highest; then softmax; then the fewest
          whose odds add up to top-p, renormalised; drawn by torch.multinomial after torch.manual_seed(seed) (interp_engine
          steer.generate_stream, _sample_next). Greedy is the argmax of z after the penalty; the answers the study read its lenses on were written as the argmax with no
          penalty (E3, R1). Every
          field is sent, and the server passes each one on: interp-engine itself would fill a missing one from the checkpoint’s
          own values (below), so none is ever left to it.
        </dd>
        <dt>the model’s own</dt>
        <dd>
          the checkpoint’s generation_config.json: do_sample true, temperature {MODEL_SAMPLING.temperature}, top_k {MODEL_SAMPLING.topK},
          top_p {MODEL_SAMPLING.topP}; the writing stops at any of the ids it lists as its end marks: 106 “&lt;turn|&gt;”, which
          ends its turn; 1 “&lt;eos&gt;”, the end of the text; 50 “&lt;|tool_response&gt;”, a control mark. The word-pieces it
          wrote name the mark it stopped at.
        </dd>
        <dt>the last answer</dt>
        <dd>
          {last ? (
            <>
              as the server resolved it: {generationWords(wrote ?? g) ?? <Missing what="not sent by this server" />}
              {g?.source === "given" ? "; read again from its ids, not written again" : ""}
              {n != null ? (
                <>
                  ; {n} word-piece{n === 1 ? "" : "s"}
                  {g?.maxNewTokens != null ? ` of at most ${g.maxNewTokens}` : ""}
                </>
              ) : null}
              {stop((wrote ?? g)?.stopReason ?? null) ? ` (${stop((wrote ?? g)?.stopReason ?? null)})` : ""}; {last.nToSight} to sight
              {g?.capped ? `, at most ${g.maxSighted ?? SIGHT_CAP} at a time` : ""}.
            </>
          ) : (
            <Missing what="no live answer yet" />
          )}
        </dd>
        <dt>timing</dt>
        <dd className="num">
          estimated from this server: {PACE.base} s with nothing written, 5.9 s with 6, 19.2 s with 24: about {PACE.write} s a
          word-piece written and {PACE.sight} s for each further one sighted. The estimate above is at most: it stops sooner at a
          mark that stops the writing.
        </dd>
      </dl>
    </Apparatus>
  );
};

import React from "react";
import { Apparatus, IfApparatus, IfRubrics, Missing, Rubric } from "../../components/kit";
import {
  VOICE_LENGTH,
  useTheodoliteStore,
  voiceKey,
  voiceOf,
  voicePayload,
  type ActiveEdit,
  type EditVoice,
  type LarqlEntry,
} from "../../store/useTheodoliteStore";
import { pct } from "../../lib/words";
import { EXAMPLE_EDIT, EXAMPLE_QUESTION, INSERT_EXAMPLE, isFloat } from "./lql";
import { asked, editDescription, num, replyTo, said } from "./live";
import { FaceMark, MONO, OwnWords, roman, sentenceInput, smallCaps } from "./bits";
import { toConsole, useVaultUi } from "./ui";

const radioStyle: React.CSSProperties = { width: 18, height: 18, accentColor: "var(--rubric)", margin: 0 };
const radioLabel: React.CSSProperties = { display: "flex", alignItems: "center", gap: 10, minHeight: 44, fontSize: 18, cursor: "pointer" };
const legendStyle: React.CSSProperties = { float: "left", padding: 0, marginRight: 8, ...smallCaps() };
const fieldsetStyle: React.CSSProperties = {
  border: 0,
  margin: 0,
  padding: 0,
  display: "flex",
  flexWrap: "wrap",
  alignItems: "center",
  gap: "2px 28px",
};

/** Why the answer after an edit could not be written, from the vault's sentence. */
const afterWhy = (v: EditVoice) =>
  (v.error ?? "the vault gave no reason").replace(/^The voice after the edit could not be written: /, "").replace(/; the edit is in force\.?$/, "");
const sameAnswer = (v: EditVoice) => !!v.after && v.partedAt === null;
/** A word's share of the model's own odds; one too small for pct() to show is said to be so, never printed as 0%. */
const odds = (p: number) => (p > 0 && p < 0.001 ? "under 0.1%" : pct(p));

/** One edit's voice, as the ledger gives it. */
const VoiceLine: React.FC<{ e: ActiveEdit; v: EditVoice | null }> = ({ e, v }) => {
  if (!v)
    return (
      <span style={{ fontSize: 16, lineHeight: 1.45, fontStyle: "italic", color: "var(--muted)" }}>
        Not heard: made without its voice (with hearing off here or in the console, through the MCP tool without one, or by an
        older server).
      </span>
    );
  const replaced = typeof e.replaces_patch === "number" ? e.replaces_patch : null;
  const others = v.editsInForceBefore.filter((id) => id !== replaced && id !== e.patch_id).length;
  const wb = v.firstStep?.watchBefore ?? null;
  const wa = v.firstStep?.watchAfter ?? null;
  return (
    <>
      <span style={{ fontSize: 16, lineHeight: 1.45 }}>
        Heard on {asked(v)}: before, {said(v.before)}; after, {v.after ? said(v.after) : "not written (" + afterWhy(v) + ")"}.
        {wb && wa ? ` At its first word-piece, “${wb.token.trim()}” went from ${odds(wb.prob)} to ${odds(wa.prob)}.` : ""}
      </span>
      {replaced !== null ? <Rubric style={{ fontSize: 15 }}>It replaces patch {replaced} on the same neuron; its “before” was heard with patch {replaced} in force.</Rubric> : null}
      {others ? (
        <Rubric style={{ fontSize: 15 }}>
          It was heard with {others === 1 ? "one other edit" : others + " other edits"} already in force; undo them and it may say
          something else.
        </Rubric>
      ) : null}
    </>
  );
};

/** The newest reply to ROLLBACK in the log, if any. */
const lastRollback = (log: LarqlEntry[]) => log.find((e) => e.ok && (e.data as { status?: string } | null)?.status === "rolled_back") ?? null;

export const EditPanel: React.FC = () => {
  const ui = useVaultUi();
  const conn = useTheodoliteStore((s) => s.conn);
  const larqlLog = useTheodoliteStore((s) => s.larqlLog);
  const larqlPending = useTheodoliteStore((s) => s.larqlPending);
  const activeEdits = useTheodoliteStore((s) => s.activeEdits);
  const editVoices = useTheodoliteStore((s) => s.editVoices);
  const executeLarql = useTheodoliteStore((s) => s.executeLarql);
  // the question the edit is heard by: the example's own at first (ui.ts), the instrument page's once taken again
  const sQuestion = useTheodoliteStore((s) => s.question);
  const sBegins = useTheodoliteStore((s) => s.answerBegins);
  const sChat = useTheodoliteStore((s) => s.chat);
  const sSystem = useTheodoliteStore((s) => s.system);
  const vQuestion = ui.vQuestion ?? sQuestion;
  const vBegins = ui.vBegins ?? sBegins;
  const vChat = ui.vChat ?? sChat;
  const vSystem = ui.vSystem ?? sSystem;
  const vOwn = ui.vQuestion !== null || ui.vBegins !== null || ui.vChat !== null || ui.vSystem !== null;
  const hearOk = !ui.hear || vQuestion.trim() !== "";

  // ---------- the form (board: renderVals, "edit")
  const eN = num(ui.eN);
  const eL = num(ui.eLayer);
  const eNOk = eN !== null && eN >= 0 && eN <= 21503;
  const eLOk = eL !== null && eL >= 1 && eL <= 60;
  const eWord = ui.eWord.trim();
  const eS = ui.eStrength.trim();
  const eSOk = isFloat(eS);
  const isUpd = ui.eKind === "update";
  const posFace = ui.ePol !== "-";
  const wordOk = eWord !== "" && !/['"]/.test(eWord);
  const valid = eNOk && eLOk && (!isUpd || (wordOk && eSOk)) && hearOk;
  // the tab's own example, heard on its own question (lql.ts EXAMPLE_EDIT)
  const isExample =
    isUpd && eN === EXAMPLE_EDIT.n && eL === EXAMPLE_EDIT.layer && eWord === EXAMPLE_EDIT.word && eS === EXAMPLE_EDIT.strength && ui.ePol === EXAMPLE_EDIT.pol;
  // ... heard as it was tried: its own question, a chat, no system prompt, nothing begun
  const exampleHeard = isExample && ui.hear && vQuestion.trim() === EXAMPLE_QUESTION && vBegins.trim() === "" && vChat && !vSystem.trim();
  const nTxt = eNOk ? String(eN) : "…";
  const lTxt = eLOk ? String(eL) : "…";
  const sTxt = eSOk ? eS : "…";
  const wq = "“" + (eWord || "…") + "”";
  // layer number -> repo index at the boundary: the vault counts from nought
  const eLql = isUpd
    ? "UPDATE LAYER " +
      (eLOk ? eL - 1 : "…") +
      " SLOT " +
      nTxt +
      " SET object = '" +
      (eWord || "…") +
      "' STRENGTH " +
      sTxt +
      " POLARITY " +
      (posFace ? "+" : "-")
    : "DELETE FROM LAYER " + (eLOk ? eL - 1 : "…") + " WHERE slot = " + nTxt;
  const plain = isUpd
    ? "Neuron " +
      nTxt +
      " of layer " +
      lTxt +
      " will gain a push toward " +
      wq +
      ", " +
      sTxt +
      " times as strong as its own voice, felt whenever its " +
      (posFace ? "positive" : "negative") +
      " face fires. When its " +
      (posFace ? "negative" : "positive") +
      " face fires, the push runs the other way, away from " +
      wq +
      ". Nothing else in the model changes."
    : "Neuron " + nTxt + " of layer " + lTxt + " will fall silent: whatever it would have written, on either face, becomes nothing. Nothing else in the model changes.";
  const problems: string[] = [];
  if (!eNOk) problems.push("neurons are numbered 0 to 21,503 in every layer");
  if (!eLOk) problems.push("layers run from 1 to 60");
  if (isUpd && eWord === "") problems.push("the edit needs a word");
  if (isUpd && eWord !== "" && !wordOk) problems.push("the word may not contain a quotation mark");
  if (isUpd && !eSOk) problems.push("the push needs a strength of 0 or more");
  if (!hearOk) problems.push("to hear it, the model needs a question to answer (or leave its voice unheard)");
  const read = !valid
    ? "Complete the form above: " + problems.join("; ") + "."
    : isUpd
      ? "UPDATE at index " +
        (eL! - 1) +
        " (layer " +
        eL +
        "), slot i = " +
        eN +
        ", word t = “" +
        eWord +
        "”, α = " +
        eS +
        ", p = " +
        (posFace ? "+1" : "−1") +
        ": a push of length " +
        eS +
        " · max(‖c‖, 0.25) along d̂."
      : "DELETE at index " + (eL! - 1) + " (layer " + eL + "), slot i = " + eN + ": c′ = 0.";

  const make = () => {
    if (!valid) return;
    const voice = ui.hear ? voicePayload(vQuestion, vBegins, vChat, vSystem) : null;
    ui.set({ editSent: { lql: eLql, at: Date.now(), voice } });
    executeLarql(eLql, voice);
  };

  // ---------- the vault's reply to this page's last edit
  const editReply = ui.editSent ? replyTo(larqlLog, ui.editSent.lql, ui.editSent.at) : null;
  const editPending = larqlPending && ui.editSent !== null && editReply === null;
  const patch = editReply && editReply.ok ? ((editReply.data as { patch?: ActiveEdit } | null)?.patch ?? null) : null;
  const heard = editReply && editReply.ok ? voiceOf((editReply.data as { voice?: unknown } | null)?.voice) : null;
  const askedVoice = ui.editSent?.voice ?? null;

  // ---------- the ledger: the edits in force, as the vault reports them
  const none = activeEdits.length === 0;
  const undoEnabled = !none || conn === "open";
  const undo = () => {
    ui.set({ undoAt: Date.now() });
    executeLarql("ROLLBACK");
  };
  const undoReply = ui.undoAt !== null ? replyTo(larqlLog, "ROLLBACK", ui.undoAt) : null;
  const undoPending = larqlPending && ui.undoAt !== null && undoReply === null;
  const rolled = lastRollback(larqlLog);

  return (
    <section
      role="tabpanel"
      id="panel-edit"
      aria-labelledby="tab-edit"
      style={{ paddingTop: 34, display: "flex", flexWrap: "wrap", gap: "40px 72px", alignItems: "flex-start" }}
    >
      <div style={{ flex: "999 1 640px", minWidth: 0, display: "flex", flexDirection: "column", gap: 22 }}>
        <h2 className="h-sec">ii · change one neuron</h2>
        <Rubric style={{ maxWidth: 720 }}>Choose what to do, fill in the sentence, and read what it will do before you make the edit.</Rubric>
        <fieldset style={fieldsetStyle}>
          <legend style={legendStyle}>what to do</legend>
          <label style={radioLabel}>
            <input type="radio" name="e-kind" checked={isUpd} onChange={() => ui.set({ eKind: "update" })} style={radioStyle} />
            make it say a word more strongly
          </label>
          <label style={radioLabel}>
            <input type="radio" name="e-kind" checked={!isUpd} onChange={() => ui.set({ eKind: "delete" })} style={radioStyle} />
            silence it
          </label>
        </fieldset>
        {isUpd ? (
          <>
            <div style={{ fontSize: 28, lineHeight: 1.8, maxWidth: 820 }}>
              <label htmlFor="e-n">Make neuron</label>{" "}
              <input
                id="e-n"
                type="number"
                min={0}
                max={21503}
                value={ui.eN}
                onChange={(e) => ui.set({ eN: e.target.value })}
                style={sentenceInput(28, "4.4em", { numeric: true })}
              />{" "}
              <label htmlFor="e-l">of layer</label>{" "}
              <input
                id="e-l"
                type="number"
                min={1}
                max={60}
                value={ui.eLayer}
                onChange={(e) => ui.set({ eLayer: e.target.value })}
                style={sentenceInput(28, "2.8em", { numeric: true })}
              />{" "}
              <label htmlFor="e-w">say</label>{" "}
              <input
                id="e-w"
                type="text"
                value={ui.eWord}
                onChange={(e) => ui.set({ eWord: e.target.value })}
                autoComplete="off"
                spellCheck={false}
                style={sentenceInput(28, "5em", { italic: true })}
              />{" "}
              <label htmlFor="e-s">more strongly, with a push of strength</label>{" "}
              <input
                id="e-s"
                type="number"
                min={0}
                step={0.5}
                value={ui.eStrength}
                onChange={(e) => ui.set({ eStrength: e.target.value })}
                style={sentenceInput(28, "3em", { numeric: true })}
              />
              .{" "}
            </div>
            <fieldset style={fieldsetStyle}>
              <legend style={legendStyle}>felt when it fires on</legend>
              <label style={radioLabel}>
                <input type="radio" name="e-face" checked={posFace} onChange={() => ui.set({ ePol: "+" })} style={radioStyle} />
                <FaceMark face={1} />
                its positive face
              </label>
              <label style={radioLabel}>
                <input type="radio" name="e-face" checked={!posFace} onChange={() => ui.set({ ePol: "-" })} style={radioStyle} />
                <FaceMark face={-1} />
                its negative face
              </label>
            </fieldset>
          </>
        ) : (
          <div style={{ fontSize: 28, lineHeight: 1.8, maxWidth: 820 }}>
            <label htmlFor="e-n2">Silence neuron</label>{" "}
            <input
              id="e-n2"
              type="number"
              min={0}
              max={21503}
              value={ui.eN}
              onChange={(e) => ui.set({ eN: e.target.value })}
              style={sentenceInput(28, "4.4em", { numeric: true })}
            />{" "}
            <label htmlFor="e-l2">of layer</label>{" "}
            <input
              id="e-l2"
              type="number"
              min={1}
              max={60}
              value={ui.eLayer}
              onChange={(e) => ui.set({ eLayer: e.target.value })}
              style={sentenceInput(28, "2.8em", { numeric: true })}
            />
            .{" "}
          </div>
        )}
        <div style={{ display: "flex", flexDirection: "column", gap: 8, maxWidth: 760 }}>
          <h3 className="h-sub">what this will do</h3>
          <p style={{ margin: 0, fontSize: 20, lineHeight: 1.5 }}>{plain}</p>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 8, maxWidth: 820 }}>
          <h3 className="h-sub">hear it before and after</h3>
          <label style={radioLabel}>
            <input type="checkbox" checked={ui.hear} onChange={(e) => ui.set({ hear: e.target.checked })} style={radioStyle} />
            hear its voice before and after the edit
          </label>
          {ui.hear ? (
            <>
              <div style={{ fontSize: 20, lineHeight: 1.9 }}>
                Just before the edit and again just after, the vault will ask the model{" "}
                <label htmlFor="v-q" className="visually-hidden">
                  the question
                </label>
                <input
                  id="v-q"
                  type="text"
                  value={vQuestion}
                  onChange={(e) => ui.set({ vQuestion: e.target.value })}
                  autoComplete="off"
                  spellCheck={false}
                  style={sentenceInput(20, "min(18em, 100%)", { italic: true })}
                />
                {vChat ? (
                  <>
                    {" "}
                    as a chat
                    {vSystem.trim() ? <>, after the system prompt “{vSystem.trim()}”</> : null}, its answer begun{" "}
                  </>
                ) : (
                  <> as bare text, with </>
                )}
                <label htmlFor="v-a" className="visually-hidden">
                  the begun answer
                </label>
                <input
                  id="v-a"
                  type="text"
                  value={vBegins}
                  placeholder="(nothing)"
                  onChange={(e) => ui.set({ vBegins: e.target.value })}
                  autoComplete="off"
                  spellCheck={false}
                  style={sentenceInput(20, "8em", { italic: true })}
                />
                {vChat ? "" : " on a new line after it"}, and write each answer the likeliest way, up to {VOICE_LENGTH} word-pieces.
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "4px 24px", fontSize: 16 }}>
                <label style={radioLabel}>
                  <input type="checkbox" checked={vChat} onChange={(e) => ui.set({ vChat: e.target.checked })} style={radioStyle} />
                  as a chat
                </label>
                {vChat ? (
                  <label style={{ ...radioLabel, fontSize: 16 }}>
                    system prompt{" "}
                    <input
                      type="text"
                      value={vSystem}
                      placeholder="(none)"
                      onChange={(e) => ui.set({ vSystem: e.target.value })}
                      autoComplete="off"
                      spellCheck={false}
                      style={sentenceInput(16, "14em")}
                    />
                  </label>
                ) : null}
                <span style={{ color: "var(--muted)" }}>
                  {vOwn ? (
                    <button
                      type="button"
                      className="btn-ghost"
                      onClick={() => ui.set({ vQuestion: null, vBegins: null, vChat: null, vSystem: null })}
                    >
                      take the instrument’s question again
                    </button>
                  ) : (
                    "As the question stands on the instrument page."
                  )}
                </span>
              </div>
            </>
          ) : (
            <p style={{ margin: 0, fontSize: 17, color: "var(--muted)" }}>The edit will be made without its voice; the ledger will say so.</p>
          )}
        </div>
        <OwnWords
          lql={eLql}
          maxWidth={820}
          note={eLOk ? `Layer ${eL} is written ${eL - 1}: the vault counts from nought. Neuron numbers are used as they are.` : "Choose a layer from 1 to 60."}
        />
        {problems.length ? (
          <p style={{ margin: 0, fontStyle: "italic", fontSize: 17, color: "var(--rubric)" }}>Before you edit: {problems.join("; ")}.</p>
        ) : null}
        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "12px 20px" }}>
          <button className="btn-primary" type="button" onClick={make} disabled={editPending}>
            {isUpd ? "make this edit" : "silence this neuron"}
          </button>
          {exampleHeard ? (
            <span style={{ fontSize: 15, color: "var(--muted)" }}>
              The values filled in are the tab’s example, chosen because the model’s answer to its question changes.
            </span>
          ) : null}
        </div>
        {editPending ? (
          <p className="status-line" role="status">
            The vault is making the edit…
          </p>
        ) : editReply && editReply.ok ? (
          <p className="status-line" role="status">
            {heard && !heard.after
              ? `The vault made the edit${patch ? `, patch ${patch.patch_id}` : ""}, and it is in force; its answer after the edit could not be written (${afterWhy(heard)}).`
              : heard && sameAnswer(heard)
                ? `The vault made the edit${patch ? `, patch ${patch.patch_id}` : ""}. Asked ${asked(heard)} and written the likeliest way, its answer did not change: ${said(heard.before)}.`
                : heard && heard.after
                  ? `The vault made the edit${patch ? `, patch ${patch.patch_id}` : ""}. Asked ${asked(heard)} and written the likeliest way, it answered ${said(heard.before)} before the edit and ${said(heard.after)} after.`
                  : `The vault made the edit${patch ? `, patch ${patch.patch_id}` : ""}. It is now in force, listed under the edits in force.${
                      askedVoice ? " Its voice was asked for, but this server did not send it: it is older than this page." : ""
                    }`}
          </p>
        ) : editReply && !editReply.ok ? (
          editReply.failure === "refused" ? (
            <p className="error-line" role="alert">
              The vault refused the edit: {editReply.error}
            </p>
          ) : (
            <p className="status-line" role="status">
              The request got no answer from the vault ({editReply.error}), so no edit was made.
            </p>
          )
        ) : null}
        <IfRubrics>
          <div style={{ display: "flex", flexDirection: "column", gap: 12, maxWidth: 760, paddingTop: 4 }}>
            {isUpd ? (
              <Rubric>
                This is an edit to one neuron, not a change to what the model knows. The original tool calls it a MEMIT edit, but it is not
                MEMIT: a single column of a single layer moves and nothing is solved for, so the model may say the word in places you never
                meant.
              </Rubric>
            ) : null}
            {isExample ? (
              <Rubric>
                This is the tab’s example. Neuron {EXAMPLE_EDIT.n} of layer {EXAMPLE_EDIT.layer} fires hardest of its layer where the answer to
                “{EXAMPLE_QUESTION}” begins, and there the edit is heard: “A spider has eight legs.” becomes “six legs.” Tried on seven questions
                (29 October 2026), it also gave a dog six legs and wrote a hexagon’s “6 sides” as “six sides”, and it left an insect’s six legs,
                Paris, the colour of the sky and the model’s account of itself as they were. Such an edit is rare: the most active neurons at
                layers 41, 46, 51, 56 and 59 were not heard on this question at strengths 3 or 30. At the last layer only the norms stand between
                this neuron’s write and the next word, so the edit raises “six” wherever the neuron fires on this face: the example shows an
                edit being heard, not where the model keeps what a spider is.
              </Rubric>
            ) : null}
            {isUpd && eLOk && eL >= 36 && eL <= 59 ? (
              <Rubric>
                Between layers 36 and 59, the direction this edit pushes along lies almost wholly in the model’s quiet dimensions, where the
                study found that even a small push can derail it.
              </Rubric>
            ) : null}
            {!isUpd ? (
              <Rubric>
                One silent neuron rarely changes much. In the study, silencing ten “she” neurons at once at layer 51 moved sentences about
                women by −1.80 toward “he”; ten random neurons moved them by no more than 0.04.
              </Rubric>
            ) : null}
            {ui.hear ? (
              <Rubric>
                Its voice is what the model itself writes, not a reading through the lens, so it shows what this edit did, but only to this
                one question. It may change answers you never asked for.
              </Rubric>
            ) : null}
            <Rubric>An edit stays in force for every sighting, landmark and nudge until you undo it. Read the model’s answer after each one.</Rubric>
          </div>
        </IfRubrics>
        <Apparatus title="what an edit writes" label="Apparatus for an edit" style={{ maxWidth: 860 }}>
          <dl className="dl">
            <dt>technical name</dt>
            <dd>
              a one-neuron column edit, rank one, kept as a “.patch” overlay (
              <span className="mono-wrap">LarqlVaultEngine.apply_patch_surgery</span>; the reply’s{" "}
              <span className="mono-wrap">edit_kind</span> “one_neuron_column”). The code and its statements are named after MEMIT
              (Meng et al. 2023) [11], but it is not MEMIT: one column of one layer’s down-projection moves and nothing is solved for, with
              no covariance solve (REPORT §3.3; E6 §3, item 12).
            </dd>
            <dt>update</dt>
            <dd>
              c′ = c + p · α · max(‖c‖, 0.25) · d̂, where c = W<sub>down</sub>[:, i] of layer ℓ, unpacked from its 4-bit store; p = ±1, the
              POLARITY; α the STRENGTH (2.5 when left out); and d̂ = normalise((W<sub>U</sub>[t] ⊙ g) · J<sub>ℓ</sub>), the word’s pullback
              through the J++ lens [4], fitted on the base model.
            </dd>
            <dt>silence</dt>
            <dd>DELETE writes c′ = 0: the neuron still fires, but what it writes is nothing.</dd>
            <dt>insert</dt>
            <dd>
              INSERT chooses the neuron itself, the one whose input side W<sub>gate</sub>[i] lies nearest the subject’s pullback (largest
              |cos|, whose sign sets p), then writes the object as UPDATE does. Only the down-projection column changes: the docstring the
              study audited said gate_proj was edited too, and now says it is not (E6 §3, item 12). The reader’s form above has no INSERT;
              write one by hand in the console, where it is heard as an edit made here is, unless hearing is turned off there.
            </dd>
            <dt>this edit</dt>
            <dd className="num">{read}</dd>
            <dt>its reply</dt>
            {patch ? (
              <dd className="num">
                patch_id {patch.patch_id}, ‖c‖ (orig_norm) {patch.orig_norm ?? "—"} and ‖c′‖ (new_norm) {patch.new_norm ?? "—"}, at index{" "}
                {patch.layer_idx} (layer {patch.layer_idx + 1}), slot {patch.neuron_idx}.
              </dd>
            ) : (
              <dd style={{ margin: 0, color: "var(--muted)" }}>
                patch_id, ‖c‖ (orig_norm) and ‖c′‖ (new_norm): <Missing />
              </dd>
            )}
            <dt>its voice</dt>
            {heard ? (
              <dd className="num">
                written the likeliest way (temperature 0, no repeat penalty), at most {heard.maxNewTokens ?? VOICE_LENGTH} word-pieces,
                stopping at the end of its turn, with no nudge, under the GPU lock: the answer before, the edit, the answer after, with
                nothing between ({heard.chat ? "as a chat" : "as bare text"}
                {heard.system ? ", with a system prompt" : ""}). First word-piece:{" "}
                {heard.firstStep ? (
                  <>
                    KL(before ‖ after) = {heard.firstStep.kl !== null ? heard.firstStep.kl.toPrecision(3) : "—"} nats
                    {heard.firstStep.watchBefore && heard.firstStep.watchAfter
                      ? `; “${heard.firstStep.watchBefore.token.trim()}” ${odds(heard.firstStep.watchBefore.prob)} → ${odds(
                          heard.firstStep.watchAfter.prob,
                        )}, rank ${heard.firstStep.watchBefore.rank} → ${heard.firstStep.watchAfter.rank}`
                      : ""}
                  </>
                ) : (
                  "not compared, since the answer after was not written"
                )}
                ; the two answers part{" "}
                {heard.after ? (heard.partedAt === null ? "never" : `at word-piece ${heard.partedAt + 1}`) : "— (no answer after)"}. Edits in
                force before it:{" "}
                {heard.editsInForceBefore.length ? "patches " + heard.editsInForceBefore.join(", ") : "none"}. Written in{" "}
                {heard.elapsedMs !== null ? Math.round(heard.elapsedMs) + " ms" : "—"}, both answers and the edit.
              </dd>
            ) : (
              <dd style={{ margin: 0, color: "var(--muted)" }}>
                the answer before and after, how they part, and the first word-piece’s odds:{" "}
                <Missing what={editReply && editReply.ok ? "not heard" : "measured when an edit is heard"} />
              </dd>
            )}
            <dt>where it lives</dt>
            <dd>
              in memory, as <span className="mono-wrap">down_proj._memit_col_overlays[i]</span> on the loaded model; the weights on disk are
              never written. Every sighting reports the edits in force, under <span className="mono-wrap">larql_status</span>, and the MCP
              scientist shares the same vault. An edit waits for any sighting in progress to finish, so every word-piece of one answer is
              read under the same edits.
            </dd>
            <dt>caveat</dt>
            <dd>
              From layer 36 the pullback d̂ lies 99–100% in the quiet dimensions, where a small push can derail the model (E6 §4b), and the
              lens it is pulled back through was fitted on another checkpoint (REPORT §3.4, bug 9). Of the vault’s edits, the study ran only
              DELETE: silencing ten neurons at layer 51 moved sentences about women by −1.80 (E6 §2.3). It never put UPDATE or INSERT to a
              causal test.
            </dd>
          </dl>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "8px 12px", paddingTop: 4 }}>
            <button className="btn-ghost" type="button" onClick={() => toConsole(eLql)}>
              write this edit by hand
            </button>
            <button className="btn-ghost" type="button" onClick={() => toConsole(INSERT_EXAMPLE)}>
              write an insert by hand
            </button>
          </div>
        </Apparatus>
      </div>

      <aside aria-labelledby="force-head" style={{ flex: "1 1 380px", minWidth: 0, display: "flex", flexDirection: "column", gap: 18 }}>
        <h3 className="h-sub" id="force-head">
          edits in force
        </h3>
        <div style={{ display: "flex", alignItems: "flex-end", gap: 20, paddingBottom: 16, borderBottom: "1px solid var(--hair)" }}>
          <div
            style={{
              fontSize: 104,
              lineHeight: 0.8,
              fontWeight: 500,
              color: "var(--gilt-numeral)",
              fontVariantNumeric: "lining-nums",
              letterSpacing: "-0.02em",
            }}
          >
            {activeEdits.length}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 2, paddingBottom: 4 }}>
            <span style={smallCaps(15, "0.2em")}>{activeEdits.length === 1 ? "edit in force" : "edits in force"}</span>
            <span style={{ fontSize: 16, fontStyle: "italic", color: "var(--muted)" }}>
              {none ? "the model is as it was loaded" : "until you undo them"}
            </span>
          </div>
        </div>
        {none ? <p style={{ margin: 0, fontSize: 17, color: "var(--muted)" }}>None. The model is running on its own weights.</p> : null}
        {activeEdits.map((e, i) => (
          <div
            key={`${e.patch_id}:${i}`}
            style={{ display: "grid", gridTemplateColumns: "34px minmax(0, 1fr)", gap: "2px 10px", paddingBottom: 14, borderBottom: "1px solid var(--hair-faint)" }}
          >
            <span style={{ fontSize: 18, fontVariantCaps: "all-small-caps", letterSpacing: "0.06em", color: "var(--gilt-text)" }}>{roman(i)}</span>
            <div style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 0 }}>
              <span style={{ fontSize: 17 }}>{editDescription(e)}</span>
              <code style={{ fontFamily: MONO, fontSize: 13, color: "var(--ink)", overflowWrap: "anywhere" }}>{e.statement}</code>
              <IfApparatus>
                <span className="num" style={{ fontSize: 14, color: "var(--muted)" }}>
                  patch {e.patch_id}
                  {e.orig_norm !== undefined && e.new_norm !== undefined ? ` · ‖c‖ ${e.orig_norm} → ‖c′‖ ${e.new_norm}` : ""}
                </span>
              </IfApparatus>
              <VoiceLine e={e} v={voiceOf(e.voice) ?? editVoices[voiceKey(e)] ?? null} />
            </div>
          </div>
        ))}
        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "10px 16px" }}>
          <button
            type="button"
            onClick={undo}
            disabled={!undoEnabled || undoPending}
            style={{
              fontFamily: "inherit",
              fontSize: 19,
              fontVariantCaps: "all-small-caps",
              letterSpacing: "0.16em",
              background: "transparent",
              color: undoEnabled ? "var(--rubric)" : "var(--muted)",
              border: `1px solid ${undoEnabled ? "var(--rubric)" : "var(--control)"}`,
              borderRadius: 2,
              padding: "11px 24px",
              minHeight: 48,
              cursor: undoEnabled ? "pointer" : "default",
            }}
          >
            undo every edit
          </button>
          <code style={{ fontFamily: MONO, fontSize: 15, color: "var(--muted)", overflowWrap: "anywhere" }}>ROLLBACK</code>
        </div>
        {undoPending ? (
          <p className="status-line" role="status">
            The vault is undoing every edit…
          </p>
        ) : undoReply && undoReply.ok ? (
          <p className="status-line" role="status">
            Every edit undone: {String((undoReply.data as { cleared_patches?: number } | null)?.cleared_patches ?? 0)} cleared.
          </p>
        ) : undoReply && !undoReply.ok ? (
          undoReply.failure === "refused" ? (
            <p className="error-line" role="alert">
              The vault refused: {undoReply.error}
            </p>
          ) : (
            <p className="status-line" role="status">
              The request got no answer from the vault ({undoReply.error}), so nothing was undone.
            </p>
          )
        ) : null}
        <Rubric>
          Undo returns every neuron in all sixty layers to the model’s own weights. In the study, after undoing, the model’s readings matched
          the untouched model exactly: the largest difference was 0.0.
        </Rubric>
        <Apparatus title="rollback" label="Apparatus for ROLLBACK" sources={false}>
          <dl style={{ margin: 0, display: "grid", gridTemplateColumns: "minmax(84px, 112px) minmax(0, 1fr)", gap: "6px 16px" }} className="dl">
            <dt>what it does</dt>
            <dd>
              <span className="mono-wrap">rollback_patches()</span>: in each of the 60 layers, clears every column and row overlay on
              down_proj, gate_proj and up_proj, then empties the list of patches.
            </dd>
            <dt>its reply</dt>
            <dd>
              <span className="mono-wrap" style={{ fontSize: 13 }}>
                {'{"status": "rolled_back", "cleared_patches": n, "active_patches_count": 0}'}
              </span>
              , with n the edits it cleared.
            </dd>
            <dt>as recorded</dt>
            <dd className="num">
              after the twelve trials, the largest change in she-against-he log-odds from the untouched model, over sentences about women
              and men, Gemma describing itself and a human friend named Gemma: 0.0 (
              <span className="mono-wrap" style={{ fontSize: 13 }}>
                rollback_check_max_abs_diff
              </span>
              ).
            </dd>
            {rolled ? (
              <>
                <dt>the last reply</dt>
                <dd>
                  <span className="mono-wrap" style={{ fontSize: 13 }}>
                    {JSON.stringify(rolled.data)}
                  </span>
                </dd>
              </>
            ) : (
              <>
                <dt>not recorded</dt>
                <dd style={{ margin: 0, color: "var(--muted)" }}>
                  each ROLLBACK’s own reply: <Missing />
                </dd>
              </>
            )}
          </dl>
        </Apparatus>
      </aside>
    </section>
  );
};

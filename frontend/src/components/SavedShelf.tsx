import React from "react";
import { IfApparatus, Missing, Rubric } from "./kit";
import { firstWords } from "../lib/words";
import { useTheodoliteStore, type SavedKind, type SavedQuestion, type SavedSystem } from "../store/useTheodoliteStore";

/**
 * The saved questions, or the saved system prompts: kept by the instrument's server only when asked (there is no
 * history), so they are there next time, in any browser. The count opens the list; choosing an entry fills the boxes
 * and sends nothing (with a way to put back what they held, when that was not saved); "save this" names what the boxes
 * hold (by default its first words, with its begun answer and "bare text" when set, and a number when the name is
 * taken); a name another entry has offers "replace", which only its button does, never Enter; "forget" and "replace"
 * each have a one-line undo. A note that speaks of what the boxes hold goes when they change. It sits inside the
 * question's <form>, so it holds no <form> of its own: its name field keeps Enter to itself.
 */

const SHELF = {
  question: { one: "saved question", many: "saved questions", thing: "question", box: "ask" },
  system: { one: "saved system prompt", many: "saved system prompts", thing: "system prompt", box: "system" },
} as const;

type Entry = SavedQuestion | SavedSystem;

/** Names are told apart as the server tells them: case aside, trimmed. */
const sameName = (a: string, b: string) => a.trim().toLowerCase() === b.trim().toLowerCase();

/** What the boxes hold, as a key: a note about them is shown only while they still hold it. */
type Boxes = { question: string; answerBegins: string; chat: boolean; system: string };
const boxesKey = (kind: SavedKind, b: Boxes) => (kind === "question" ? `${b.question}\u0000${b.answerBegins}\u0000${b.chat}` : b.system);

/**
 * The name "save this" offers: the first words of the question, then its begun answer and "(bare text)" when set, so
 * that two versions of one question get two names; the first words of a system prompt. A name already in the list gets
 * " (2)", " (3)" …, so the name offered never replaces an entry (at most 80 characters, the server's bound).
 */
function defaultName(kind: SavedKind, b: Boxes, list: Entry[]): string {
  const begun = b.answerBegins.trim();
  const base =
    kind === "question"
      ? `${firstWords(b.question, begun ? 36 : 40)}${begun ? ` → ${firstWords(begun, 20)}` : ""}${b.chat ? "" : " (bare text)"}`
      : firstWords(b.system, 40);
  let name = base;
  for (let i = 2; list.some((e) => sameName(e.name, name)); i++) name = `${base} (${i})`;
  return name;
}

/** A shelf note: what it says, and the boxes it speaks of (null: it holds whatever the boxes hold). */
type Note = { node: React.ReactNode; at: string | null };

export const SavedShelf: React.FC<{
  kind: SavedKind;
  /** the id of the box an entry fills, focused after choosing (default: "ask", or "system") */
  boxId?: string;
}> = ({ kind, boxId }) => {
  const saved = useTheodoliteStore((s) => s.saved);
  const conn = useTheodoliteStore((s) => s.conn);
  const question = useTheodoliteStore((s) => s.question);
  const answerBegins = useTheodoliteStore((s) => s.answerBegins);
  const chat = useTheodoliteStore((s) => s.chat);
  const system = useTheodoliteStore((s) => s.system);
  const fetchSaved = useTheodoliteStore((s) => s.fetchSaved);
  const keepSaved = useTheodoliteStore((s) => s.keepSaved);
  const forgetSaved = useTheodoliteStore((s) => s.forgetSaved);
  const restoreSaved = useTheodoliteStore((s) => s.restoreSaved);
  const chooseSaved = useTheodoliteStore((s) => s.useSaved);
  const [open, setOpen] = React.useState(false);
  const [naming, setNaming] = React.useState<string | null>(null);
  const [note, setNote] = React.useState<Note | null>(null);
  const [busy, setBusy] = React.useState(false);
  // Enter was pressed on a name another entry has: say plainly that only the button replaces it
  const [clashTold, setClashTold] = React.useState(false);

  // on first use, when the socket has not already asked
  React.useEffect(() => {
    if (useTheodoliteStore.getState().saved.status === "idle") void fetchSaved();
  }, [fetchSaved]);

  const S = SHELF[kind];
  const listId = `shelf-${kind}`;
  const list: Entry[] = kind === "question" ? saved.questions : saved.systems;
  const text = kind === "question" ? question : system;
  // the boxes hold an entry already: say so, rather than offer to save the same words twice
  const here: Entry | undefined =
    kind === "question"
      ? saved.questions.find((e) => e.question.trim() === question.trim() && e.answerBegins.trim() === answerBegins.trim() && e.chat === chat)
      : saved.systems.find((e) => e.text.trim() === system.trim() && system.trim() !== "");
  const writable = conn === "open" && saved.status === "ok" && !busy;
  const boxes: Boxes = { question, answerBegins, chat, system };
  const clash = naming != null && naming.trim() ? list.find((e) => sameName(e.name, naming)) : undefined;
  const focusBox = () => document.getElementById(boxId ?? S.box)?.focus();
  const serverWords = () => useTheodoliteStore.getState().saved.error ?? "it did not answer";
  /** Show a note; unless `sticky`, only while the boxes hold what they hold now (read after the action). */
  const say = (node: React.ReactNode, sticky = false) =>
    setNote({ node, at: sticky ? null : boxesKey(kind, useTheodoliteStore.getState()) });

  /** The one-line undo of a forget or a replace: once pressed it is gone (a second press would ask twice), and the
   * shelf's other writes wait until the server answers; what the server said then stays while the boxes hold what
   * they hold. */
  const undoButton = (e: Entry, replaceId: string | null) => (
    <button
      type="button"
      className="btn-link"
      onClick={() => {
        say(<>putting “{e.name}” back…</>, true);
        setBusy(true);
        void restoreSaved(kind, e, replaceId).then((back) => {
          setBusy(false);
          say(back ? <>“{e.name}” is kept again, as it was.</> : <span className="red">The server did not put it back: {serverWords()}</span>);
        });
      }}
    >
      undo
    </button>
  );

  const keep = async () => {
    const name = (naming ?? "").trim();
    if (!name) return;
    // the entry a replace overwrites, as it was, for its undo
    const old = clash ? { ...clash } : null;
    setBusy(true);
    const ok = await keepSaved(kind, name, old?.id ?? null);
    setBusy(false);
    if (ok) {
      setNaming(null);
      setClashTold(false);
      if (old) {
        say(
          <>
            Replaced the saved {S.thing} “{old.name}” with this one. {undoButton(old, old.id)}
          </>,
        );
      } else say(<>Kept as “{name}”.</>);
    } else say(<span className="red">The server did not keep it: {serverWords()}</span>);
  };

  const forget = async (e: Entry) => {
    setBusy(true);
    const ok = await forgetSaved(kind, e.id);
    setBusy(false);
    // the note speaks of the entry, not of the boxes: its undo stays when they change
    if (ok) {
      say(
        <>
          Forgot “{e.name}”. {undoButton(e, null)}
        </>,
        true,
      );
    } else say(<span className="red">The server did not forget it: {serverWords()}</span>);
  };

  const choose = (e: Entry) => {
    const st = useTheodoliteStore.getState();
    const was: Boxes = { question: st.question, answerBegins: st.answerBegins, chat: st.chat, system: st.system };
    // what the boxes held is lost unless it was saved: offer to put it back
    const unsaved =
      kind === "question"
        ? (was.question.trim() !== "" || was.answerBegins.trim() !== "") &&
          !saved.questions.some((x) => x.question.trim() === was.question.trim() && x.answerBegins.trim() === was.answerBegins.trim() && x.chat === was.chat)
        : was.system.trim() !== "" && !saved.systems.some((x) => x.text.trim() === was.system.trim());
    chooseSaved(kind, e);
    setOpen(false);
    setNaming(null);
    setClashTold(false);
    const chatNow = "chat" in e ? e.chat : was.chat;
    const putBack = () => {
      const now = useTheodoliteStore.getState();
      if (kind === "question") {
        now.setQuestion(was.question);
        now.setAnswerBegins(was.answerBegins);
        now.setChat(was.chat);
      } else now.setSystem(was.system);
      say(<>What the {kind === "question" ? "boxes" : "box"} held is back; nothing is sent until you take a sighting.</>);
      focusBox();
    };
    say(
      <>
        “{e.name}” is in the box
        {chatNow !== was.chat ? (chatNow ? ", sent as a chat, as it was saved" : ", sent as bare text, as it was saved") : ""}; nothing is sent
        until you take a sighting.
        {unsaved ? (
          <>
            {" "}
            <button type="button" className="btn-link" onClick={putBack}>
              put back what {kind === "question" ? "the boxes" : "the box"} held
            </button>
          </>
        ) : null}
      </>,
    );
    focusBox();
  };

  // what to say when the list cannot be read or written now: the server not answering (no connection), or answering
  // with a refusal or an error, whose reason is shown
  const lacking =
    conn !== "open"
      ? list.length
        ? "as the server last sent them; saving and forgetting wait on it"
        : "kept by the server, which is not answering"
      : saved.status === "failed"
        ? list.length
          ? `as the server last sent them; it did not send them again (${saved.error ?? "no reason given"}), so saving and forgetting wait`
          : `the server did not send its saved list: ${saved.error ?? "no reason given"}`
        : saved.status === "reading" && !list.length
          ? "reading the saved list…"
          : null;

  return (
    <div className="shelf-block">
      <div className="q-tools">
        {list.length ? (
          <button type="button" className="btn-link" aria-expanded={open} aria-controls={listId} onClick={() => setOpen(!open)}>
            {list.length} {list.length === 1 ? S.one : S.many}
          </button>
        ) : null}
        {here ? (
          <span>
            saved as “{here.name}” ·{" "}
            <button type="button" className="btn-link" disabled={!writable} onClick={() => void forget(here)}>
              forget it
            </button>
          </span>
        ) : naming == null ? (
          <button
            type="button"
            className="btn-link"
            disabled={!text.trim() || !writable}
            onClick={() => {
              setNote(null);
              setClashTold(false);
              setNaming(defaultName(kind, boxes, list));
            }}
          >
            save this
          </button>
        ) : null}
        {lacking ? <span className="italic">{lacking}</span> : null}
        {saved.status === "failed" && conn === "open" ? (
          <button type="button" className="btn-link" onClick={() => void fetchSaved()}>
            ask again
          </button>
        ) : null}
        <span role="status" aria-live="polite">
          {note && (note.at === null || note.at === boxesKey(kind, boxes)) ? note.node : null}
        </span>
      </div>
      {saved.warning ? <p className="shelf-warning">{saved.warning}</p> : null}
      {naming != null ? (
        <div className="shelf-form" role="group" aria-label={`Name this ${S.thing}`}>
          <label>
            <span className="label-sc" style={{ fontSize: 14 }}>
              keep it as
            </span>
            <input
              autoFocus
              value={naming}
              maxLength={80}
              onChange={(e) => {
                setNaming(e.target.value);
                setClashTold(false);
              }}
              onKeyDown={(e) => {
                if (e.nativeEvent.isComposing || e.keyCode === 229) return;
                // Enter keeps under a new name, never replaces (and must not submit the question's form, which would
                // take a sighting); Escape cancels
                if (e.key === "Enter") {
                  e.preventDefault();
                  if (clash) setClashTold(true);
                  else if (writable) void keep();
                } else if (e.key === "Escape") {
                  e.preventDefault();
                  setNaming(null);
                  setClashTold(false);
                  focusBox();
                }
              }}
              aria-describedby={clash ? `${listId}-clash` : undefined}
            />
          </label>
          <button type="button" className="btn-ghost plain" disabled={!writable || !naming.trim()} onClick={() => void keep()}>
            {clash ? `replace “${clash.name}”` : "keep"}
          </button>
          <button
            type="button"
            className="btn-link"
            onClick={() => {
              setNaming(null);
              setClashTold(false);
              focusBox();
            }}
          >
            never mind
          </button>
          {clash ? (
            <span id={`${listId}-clash`} className={clashTold ? "shelf-clash red" : "shelf-clash"} role={clashTold ? "alert" : undefined}>
              A saved {S.thing} is already called “{clash.name}”: change the name to keep both, or press “replace” to put this one
              in its place{clashTold ? "; Enter does not replace" : ""}.
            </span>
          ) : null}
        </div>
      ) : null}
      {open && list.length ? (
        <ul id={listId} className="shelf" aria-label={S.many}>
          {list.map((e) => (
            <li key={e.id}>
              <button type="button" className="shelf-use" onClick={() => choose(e)}>
                <span className="shelf-name">
                  {e.name}
                  {e === here ? <span className="shelf-here"> · in the box</span> : null}
                </span>
                <span className="shelf-text">
                  {"question" in e ? (
                    <>
                      {e.question}
                      {e.answerBegins ? <> → {e.answerBegins}</> : null}
                      {e.chat ? "" : " · as bare text"}
                    </>
                  ) : (
                    e.text
                  )}
                </span>
              </button>
              <button type="button" className="btn-link" disabled={!writable} onClick={() => void forget(e)}>
                forget
              </button>
            </li>
          ))}
        </ul>
      ) : null}
      {kind === "question" ? (
        <Rubric style={{ margin: "2px 0 0", fontSize: 16 }}>
          Nothing you write is kept unless you save it. Saved questions are kept by the instrument’s server, so they are here next
          time, in any browser.
        </Rubric>
      ) : null}
      <IfApparatus>
        <span className="muted" style={{ fontSize: 14 }}>
          kept in {saved.file ? <span className="mono-wrap">{saved.file}</span> : <Missing what="not read from the server yet" />} (GET,
          POST, DELETE /api/saved_prompts), the exact text as written; a {S.thing} saved with the same name as another
          replaces it only when you say so.
        </span>
      </IfApparatus>
    </div>
  );
};

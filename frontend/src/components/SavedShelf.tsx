import React from "react";
import { IfApparatus, Missing, Rubric } from "./kit";
import { firstWords } from "../lib/words";
import { useTheodoliteStore, type SavedKind, type SavedQuestion, type SavedSystem } from "../store/useTheodoliteStore";

/**
 * The saved questions, or the saved system prompts: kept by the instrument's server only when asked (there is no
 * history), so they are there next time, in any browser. The count opens the list; choosing an entry fills the boxes
 * and sends nothing; "save this" names what the boxes hold (by default its first words); "forget" has a one-line undo.
 * It sits inside the question's <form>, so it holds no <form> of its own: its name field keeps Enter to itself.
 */

const SHELF = {
  question: { one: "saved question", many: "saved questions", thing: "question", box: "ask" },
  system: { one: "saved system prompt", many: "saved system prompts", thing: "system prompt", box: "system" },
} as const;

type Entry = SavedQuestion | SavedSystem;

/** Names are told apart as the server tells them: case aside, trimmed. */
const sameName = (a: string, b: string) => a.trim().toLowerCase() === b.trim().toLowerCase();

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
  const [note, setNote] = React.useState<React.ReactNode>(null);
  const [busy, setBusy] = React.useState(false);

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
  const clash = naming != null ? list.find((e) => sameName(e.name, naming.trim() || firstWords(text))) : undefined;
  const focusBox = () => document.getElementById(boxId ?? S.box)?.focus();
  const serverWords = () => useTheodoliteStore.getState().saved.error ?? "it did not answer";

  const keep = async () => {
    const name = (naming ?? "").trim() || firstWords(text);
    if (!name) return;
    setBusy(true);
    const ok = await keepSaved(kind, name, clash?.id ?? null);
    setBusy(false);
    if (ok) {
      setNaming(null);
      setNote(<>Kept as “{name}”.</>);
    } else setNote(<span className="red">The server did not keep it: {serverWords()}</span>);
  };

  const forget = async (e: Entry) => {
    setBusy(true);
    const ok = await forgetSaved(kind, e.id);
    setBusy(false);
    setNote(
      ok ? (
        <>
          Forgot “{e.name}”.{" "}
          <button
            type="button"
            className="btn-link"
            onClick={() => {
              void restoreSaved(kind, e).then((back) =>
                setNote(back ? <>“{e.name}” is kept again.</> : <span className="red">The server did not keep it again: {serverWords()}</span>),
              );
            }}
          >
            undo
          </button>
        </>
      ) : (
        <span className="red">The server did not forget it: {serverWords()}</span>
      ),
    );
  };

  const choose = (e: Entry) => {
    const chatWas = useTheodoliteStore.getState().chat;
    chooseSaved(kind, e);
    setOpen(false);
    const chatNow = "chat" in e ? e.chat : chatWas;
    setNote(
      <>
        “{e.name}” is in the box
        {chatNow !== chatWas ? (chatNow ? ", sent as a chat, as it was saved" : ", sent as bare text, as it was saved") : ""}; nothing is sent
        until you take a sighting.
      </>,
    );
    focusBox();
  };

  // what to say when the list cannot be read or written now
  const lacking =
    conn !== "open" || saved.status === "failed"
      ? list.length
        ? "as the server last sent them; saving and forgetting wait on it"
        : "kept by the server, which is not answering"
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
              setNaming(firstWords(text));
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
          {note}
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
              onChange={(e) => setNaming(e.target.value)}
              onKeyDown={(e) => {
                if (e.nativeEvent.isComposing || e.keyCode === 229) return;
                // Enter keeps (and must not submit the question's form, which would take a sighting); Escape cancels
                if (e.key === "Enter") {
                  e.preventDefault();
                  if (writable) void keep();
                } else if (e.key === "Escape") {
                  e.preventDefault();
                  setNaming(null);
                  focusBox();
                }
              }}
            />
          </label>
          <button type="button" className="btn-ghost plain" disabled={!writable} onClick={() => void keep()}>
            {clash ? `replace “${clash.name}”` : "keep"}
          </button>
          <button
            type="button"
            className="btn-link"
            onClick={() => {
              setNaming(null);
              focusBox();
            }}
          >
            never mind
          </button>
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

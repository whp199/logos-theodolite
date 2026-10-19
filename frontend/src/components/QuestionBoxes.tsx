import React from "react";
import { useTheodoliteStore } from "../store/useTheodoliteStore";

/**
 * The boxes the question is written in: "you ask", "its answer begins" and the system prompt. Each grows with what is
 * written (field-sizing where the browser has it) and can be drawn taller by its corner, up to 60% of the window.
 * Enter takes a sighting in the first two; in the system prompt, where a new line is the usual need, Enter begins a
 * line and Ctrl+Enter (⌘+Enter on a Mac) takes the sighting.
 */

export type QuestionBoxKind = "ask" | "begin" | "system";

/** What the keys do in each kind of box, for a screen reader (each box points at its own line). */
export const BOX_KEYS: Record<QuestionBoxKind, string> = {
  ask: "Enter takes a sighting; Shift and Enter, or Alt and Enter, begins a new line.",
  begin: "Enter takes a sighting; Shift and Enter, or Alt and Enter, begins a new line.",
  system: "Enter begins a new line; Ctrl and Enter (⌘ and Enter on a Mac) takes a sighting.",
};

/** A key that belongs to an input method still composing a character (IME), not to the page. */
const composing = (e: React.KeyboardEvent) => e.nativeEvent.isComposing || e.keyCode === 229;

/** Put a new line where the caret is, as typing one would, so undo and the box's onChange both see it. */
function insertNewLine(el: HTMLTextAreaElement) {
  let done = false;
  try {
    done = document.execCommand("insertText", false, "\n");
  } catch {
    done = false;
  }
  if (!done) {
    el.setRangeText("\n", el.selectionStart, el.selectionEnd, "end");
    el.dispatchEvent(new Event("input", { bubbles: true }));
  }
}

/**
 * Key handlers for the question's boxes. `enterTakes`: Enter takes a sighting, Shift+Enter or Alt+Enter begins a new
 * line. `modEnterTakes`: Ctrl/Cmd+Enter takes a sighting, Enter begins a new line. Neither does anything while an input
 * method is composing, and neither takes a sighting while one is being taken (a stray Enter would queue a whole
 * second answer behind it). `onTake` replaces the store's takeSighting() (a page with its own way to sight).
 */
export function useEnterTakes(onTake?: () => void) {
  const take = React.useCallback(() => {
    const st = useTheodoliteStore.getState();
    if (st.isSweeping) return;
    if (onTake) onTake();
    else st.takeSighting();
  }, [onTake]);
  const enterTakes = React.useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key !== "Enter" || composing(e) || e.shiftKey) return; // Shift+Enter: the browser's own new line
      e.preventDefault();
      if (e.altKey) insertNewLine(e.currentTarget);
      else take();
    },
    [take],
  );
  const modEnterTakes = React.useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key !== "Enter" || composing(e) || !(e.ctrlKey || e.metaKey)) return;
      e.preventDefault();
      take();
    },
    [take],
  );
  return { enterTakes, modEnterTakes };
}

type BoxProps = {
  id: string;
  kind: QuestionBoxKind;
  value: string;
  onChange: (value: string) => void;
  /** what Enter (Ctrl+Enter in the system prompt) does instead of the store's takeSighting() */
  onTake?: () => void;
} & Omit<React.TextareaHTMLAttributes<HTMLTextAreaElement>, "id" | "value" | "onChange">;

/**
 * One of the question's boxes: a textarea of class `q-box q-ask` / `q-begin` / `q-system` with its keys, and a line for
 * a screen reader saying what they do. One line high to start (two for the system prompt).
 */
export const QuestionBox: React.FC<BoxProps> = ({ id, kind, value, onChange, onTake, className, rows, onKeyDown, ...rest }) => {
  const { enterTakes, modEnterTakes } = useEnterTakes(onTake);
  const keys = kind === "system" ? modEnterTakes : enterTakes;
  const keysId = `${id}-keys`;
  const describedBy = [rest["aria-describedby"], keysId].filter(Boolean).join(" ");
  return (
    <>
      <textarea
        {...rest}
        id={id}
        rows={rows ?? (kind === "system" ? 2 : 1)}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          onKeyDown?.(e);
          if (!e.defaultPrevented) keys(e);
        }}
        className={`q-box q-${kind}${className ? ` ${className}` : ""}`}
        aria-describedby={describedBy}
      />
      <span id={keysId} className="visually-hidden">
        {BOX_KEYS[kind]}
      </span>
    </>
  );
};

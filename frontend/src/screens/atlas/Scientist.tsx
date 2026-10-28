import React, { useEffect, useRef, useState } from "react";
import { Rubric } from "../../components/kit";
import { href } from "../../router";
import type { PastChat } from "./api";
import { ErrorLine } from "./parts";
import {
  askWeb,
  attachFiles,
  explainFocus,
  loadBackends,
  newChat,
  openEvents,
  openPast,
  removePending,
  restartChat,
  rewindLast,
  send,
  setPick,
  stop,
  takeName,
  togglePast,
  useScientist,
  type Item,
  type Part,
  type Pick,
} from "./scientist";
import { goToNeuron, openHead, useAtlas } from "./state";

/* --------------------------------------------------------------------------- a little markdown, made as elements */

/** Inline marks: `code`, **bold**, *italic*, and citations in the pages' count (L54 N15491, L54 h3) as links. */
function inline(text: string, key = "i"): React.ReactNode[] {
  const arch = useAtlas.getState().arch;
  const out: React.ReactNode[] = [];
  const re = /`([^`]+)`|\*\*([^*]+)\*\*|\*([^*\s][^*]*)\*|\bL(\d{1,2})[ /·]?N(\d{1,5})\b|\bL(\d{1,2}) h(\d{1,2})\b/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let k = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const id = `${key}.${k++}`;
    if (m[1] != null) out.push(<code key={id}>{m[1]}</code>);
    else if (m[2] != null) out.push(<b key={id}>{inline(m[2], id)}</b>);
    else if (m[3] != null) out.push(<i key={id}>{inline(m[3], id)}</i>);
    else if (m[4] != null) {
      const l = +m[4] - 1;
      const n = +m[5];
      const ok = !!arch && l >= 0 && l < arch.n_layers && n < arch.d_ff;
      out.push(
        ok ? (
          <a
            key={id}
            className="atlas-nref"
            href="#"
            onClick={(e) => {
              e.preventDefault();
              goToNeuron(l, n);
            }}
          >
            {m[0]}
          </a>
        ) : (
          m[0]
        ),
      );
    } else {
      const l = +m[6] - 1;
      const h = +m[7];
      const ok = !!arch && l >= 0 && l < arch.n_layers && h < (arch.layers[l]?.n_heads ?? 0);
      out.push(
        ok ? (
          <a
            key={id}
            className="atlas-nref"
            href="#"
            onClick={(e) => {
              e.preventDefault();
              void openHead(l, h);
            }}
          >
            {m[0]}
          </a>
        ) : (
          m[0]
        ),
      );
    }
    last = re.lastIndex;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

const Md: React.FC<{ text: string }> = ({ text }) => {
  const out: React.ReactNode[] = [];
  const lines = text.split("\n");
  let list: { tag: "ul" | "ol"; items: React.ReactNode[] } | null = null;
  const flush = () => {
    if (!list) return;
    const L = list;
    out.push(L.tag === "ul" ? <ul key={`l${out.length}`}>{L.items}</ul> : <ol key={`l${out.length}`}>{L.items}</ol>);
    list = null;
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (/^\s*```/.test(line)) {
      flush();
      const body: string[] = [];
      while (++i < lines.length && !/^\s*```/.test(lines[i])) body.push(lines[i]);
      out.push(<pre key={`c${i}`}>{body.join("\n")}</pre>);
      continue;
    }
    const h = line.match(/^#{1,4}\s+(.*)/);
    const ul = line.match(/^\s*[-*]\s+(.*)/);
    const ol = line.match(/^\s*\d+[.)]\s+(.*)/);
    if (ul || ol) {
      const tag = ul ? "ul" : "ol";
      if (list && list.tag !== tag) flush();
      if (!list) list = { tag, items: [] };
      list.items.push(<li key={i}>{inline((ul ?? ol)![1], `li${i}`)}</li>);
      continue;
    }
    flush();
    if (h) out.push(<h4 key={i}>{inline(h[1], `h${i}`)}</h4>);
    else if (line.trim()) out.push(<p key={i}>{inline(line, `p${i}`)}</p>);
  }
  flush();
  return <div className="atlas-md">{out}</div>;
};

/* ------------------------------------------------------------------------------------------------- the thread */

/** A tool call in one short line: "neuron L54 N15491", "search lens “Sure”". The scientist's tools count layers as the pages do. */
function toolLabel(name: string, input: unknown): string {
  const a = (input && typeof input === "object" ? input : {}) as Record<string, unknown>;
  const ref = a.layer != null && a.n != null ? `L${a.layer} N${a.n}` : a.layer != null && a.head != null ? `L${a.layer} h${a.head}` : a.layer != null ? `L${a.layer}` : "";
  const short = (t: string) => (t.length > 90 ? `${t.slice(0, 90)}…` : t);
  const extra =
    typeof a.command === "string"
      ? short(a.command)
      : typeof a.file_path === "string"
        ? a.file_path
        : typeof a.pattern === "string"
          ? `“${a.pattern}”`
          : typeof a.query === "string"
            ? `“${a.query}”`
            : typeof a.text === "string"
              ? `“${short(a.text)}”`
              : Array.isArray(a.neurons)
                ? `${a.neurons.length} neurons`
                : "";
  return [name.replace(/_/g, " "), ref, extra].filter(Boolean).join(" ");
}

const pretty = (t: string) => {
  try {
    return JSON.stringify(JSON.parse(t), null, 1);
  } catch {
    return t;
  }
};

const PartView: React.FC<{ p: Part; box: number }> = ({ p, box }) => {
  switch (p.t) {
    case "md":
      return <Md text={p.text} />;
    case "thinking":
      return (
        <details className="atlas-think">
          <summary>thinking</summary>
          <pre>{p.text}</pre>
        </details>
      );
    case "tool":
      return (
        <details className="atlas-tool">
          <summary>{toolLabel(p.name, p.input)}</summary>
          <pre className="atlas-io">
            <b>{p.name}</b> {JSON.stringify(p.input ?? {}, null, 1)}
          </pre>
          <pre className={`atlas-io out${p.error ? " err" : ""}`}>{p.output == null ? "running…" : pretty(p.output)}</pre>
        </details>
      );
    case "error":
      return <p className="error-line atlas-small">{p.text}</p>;
    case "note":
      return <p className="muted atlas-small">{p.text}</p>;
    case "stopped":
      return <p className="muted atlas-small">stopped</p>;
    case "usename":
      return (
        <button type="button" className="btn-ghost plain atlas-usename" onClick={() => void takeName(p, box)}>
          use name: {p.name}
        </button>
      );
  }
};

const Thread: React.FC = () => {
  const items = useScientist((s) => s.items);
  const busy = useScientist((s) => s.busy);
  const box = useScientist((s) => s.box);
  const ref = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  useEffect(() => {
    const el = ref.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [items, busy]);
  let lastUser = -1;
  for (let i = items.length - 1; i >= 0; i--)
    if (items[i].kind === "user") {
      lastUser = i;
      break;
    }
  const lastReply = lastUser >= 0 ? items.findIndex((x, i) => i > lastUser && x.kind === "assistant") : -1;
  return (
    <div
      className="atlas-sthread"
      ref={ref}
      onScroll={(e) => {
        const el = e.currentTarget;
        stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
      }}
    >
      {!items.length ? (
        <p className="muted atlas-empty">
          Ask anything about the model, or open a neuron and press <b>explain</b>.
        </p>
      ) : null}
      {items.map((it: Item, i) =>
        it.kind === "user" ? (
          <div key={i} className="atlas-msg user">
            {it.shown}
            {it.atts.length ? <span className="atlas-att"> · attached {it.atts.map((a) => a.name).join(", ")}</span> : null}
            {i === lastUser && !busy ? (
              <span className="atlas-ctl">
                <button type="button" className="btn-link" title="take this message and its reply back, into the box" onClick={() => void rewindLast("edit")}>
                  edit
                </button>
              </span>
            ) : null}
          </div>
        ) : (
          <div key={i} className="atlas-msg assistant">
            {it.parts.map((p, j) => (
              <PartView key={j} p={p} box={i} />
            ))}
            {busy && i === box ? <p className="atlas-busy">working…</p> : null}
            {i === lastReply && !busy ? (
              <span className="atlas-ctl">
                <button type="button" className="btn-link" title="throw this reply away and ask again" onClick={() => void rewindLast("reroll")}>
                  ask again
                </button>
              </span>
            ) : null}
          </div>
        ),
      )}
      {busy && box === null ? <p className="atlas-busy">working…</p> : null}
    </div>
  );
};

const Past: React.FC = () => {
  const past = useScientist((s) => s.past);
  const chat = useScientist((s) => s.chat);
  if (past === null) return null;
  if (past === "loading") return <p className="muted atlas-small atlas-pad">reading past chats…</p>;
  const ago = (t: number) => {
    const s = Date.now() / 1000 - t;
    return s < 3600 ? `${Math.max(1, Math.round(s / 60))} min ago` : s < 86400 ? `${Math.round(s / 3600)} h ago` : new Date(t * 1000).toLocaleDateString();
  };
  const by = (c: PastChat) => (c.agent === "openai" ? `${c.profile ?? "endpoint"} · ${c.model ?? ""}` : c.agent === "codex" ? "Codex" : "Claude Code");
  return (
    <div className="atlas-sthread">
      {past.length ? (
        past.map((c) => (
          <button key={c.chat} type="button" className={`atlas-pastrow${c.chat === chat ? " on" : ""}`} onClick={() => void openPast(c)}>
            <span>{c.title || "—"}</span>
            <span className="muted atlas-small">
              {by(c)} · {ago(c.t)} · {c.turns} message{c.turns === 1 ? "" : "s"}
            </span>
          </button>
        ))
      ) : (
        <p className="muted atlas-pad">No past chats yet.</p>
      )}
    </div>
  );
};

/** Which backend a new chat runs on: Claude Code or Codex where installed, or an OpenAI-compatible endpoint and model. */
const ACK_KEY = "theodolite.atlas.scientist.home-shell";

const BackendPick: React.FC = () => {
  const b = useScientist((s) => s.backends);
  const pick = useScientist((s) => s.pick);
  const chatPick = useScientist((s) => s.chatPick);
  const chat = useScientist((s) => s.chat);
  const models = useScientist((s) => s.models);
  const fixed = !!chat && !!chatPick;
  const p: Pick = fixed ? chatPick! : pick;
  const value = p.agent === "openai" ? `openai:${p.profile}` : p.agent;
  const card = b?.openai.find((x) => x.name === p.profile);
  const ms = p.agent === "openai" && p.profile ? models[p.profile] : undefined;
  return (
    <div className="atlas-spick">
      <select
        className="atlas-select"
        aria-label="the scientist's backend for a new chat"
        value={value}
        disabled={fixed}
        title={fixed ? "this chat runs on the backend it began with; start a new chat to change it" : "the backend a new chat runs on"}
        onChange={(e) => {
          const v = e.target.value;
          if (v.startsWith("openai:")) {
            const name = v.slice(7);
            setPick({ agent: "openai", profile: name, model: b?.openai.find((x) => x.name === name)?.model ?? null });
          } else setPick({ agent: v as Pick["agent"], profile: null, model: null });
        }}
      >
        <option value="claude" disabled={!!b && !b.claude.available}>
          Claude Code{b && !b.claude.available ? " (not installed)" : ""}
        </option>
        <option value="codex" disabled={!!b && !b.codex.available}>
          Codex{b && !b.codex.available ? " (not installed)" : ""}
        </option>
        {(b?.openai ?? []).map((x) => (
          <option key={x.name} value={`openai:${x.name}`}>
            {x.label}
          </option>
        ))}
      </select>
      {p.agent === "openai" ? (
        Array.isArray(ms) && ms.length ? (
          <select
            className="atlas-select"
            aria-label="the endpoint's model"
            value={p.model ?? ""}
            disabled={fixed}
            onChange={(e) => setPick({ ...pick, model: e.target.value })}
          >
            {p.model && !ms.some((m) => m.id === p.model) ? <option value={p.model}>{p.model}</option> : null}
            {ms.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name || m.id}
                {m.tools === false ? " (no tools)" : ""}
              </option>
            ))}
          </select>
        ) : (
          <input
            className="input-line atlas-model"
            aria-label="the endpoint's model"
            placeholder="model id"
            value={p.model ?? ""}
            disabled={fixed}
            onChange={(e) => setPick({ ...pick, model: e.target.value || null })}
          />
        )
      ) : null}
      {p.agent === "openai" && typeof ms === "string" ? <span className="error-line atlas-small">{ms}</span> : null}
      {p.agent === "openai" && card && card.key !== "set" && card.key !== "none needed" ? <span className="error-line atlas-small">key {card.key}</span> : null}
      {p.agent === "openai" && card && !card.this_machine ? (
        <span className="muted atlas-small" title={card.base_url}>
          sent to {new URL(card.base_url).hostname}
        </span>
      ) : null}
    </div>
  );
};

export const Scientist: React.FC = () => {
  const f = useAtlas((s) => s.focus);
  const busy = useScientist((s) => s.busy);
  const stale = useScientist((s) => s.stale);
  const input = useScientist((s) => s.input);
  const pending = useScientist((s) => s.pending);
  const past = useScientist((s) => s.past);
  const pastError = useScientist((s) => s.pastError);
  const warning = useScientist((s) => s.backends?.warning ?? null);
  const backendsError = useScientist((s) => s.backendsError);
  const reach = useScientist((s) => s.backends?.reach);
  const agent = useScientist((s) => (s.chat && s.chatPick ? s.chatPick : s.pick).agent);
  const [acked, setAcked] = useState(() => {
    try {
      return localStorage.getItem(ACK_KEY) === "1";
    } catch {
      return false;
    }
  });
  // a shell over the home folder is asked about once, in this browser, before the first chat runs on it
  const homeShell = reach?.mode === "home" && (agent === "claude" || agent === "codex");
  const needsAck = homeShell && !acked;
  const [drop, setDrop] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const boxRef = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    void loadBackends();
    void openEvents();
  }, []);
  useEffect(() => {
    const t = boxRef.current;
    if (!t) return;
    t.style.height = "auto";
    t.style.height = `${Math.min(200, t.scrollHeight)}px`;
  }, [input]);
  const at = f ? `L${f.layer + 1} ${f.kind === "head" ? "h" : "N"}${f.n}` : "";
  return (
    <div
      className={`atlas-scientist${drop ? " drop" : ""}`}
      onDragOver={(e) => {
        if (e.dataTransfer?.types?.includes("Files")) {
          e.preventDefault();
          setDrop(true);
        }
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node)) setDrop(false);
      }}
      onDrop={(e) => {
        if (!e.dataTransfer?.files?.length) return;
        e.preventDefault();
        setDrop(false);
        void attachFiles([...e.dataTransfer.files]);
      }}
    >
      <div className="atlas-sbar">
        <BackendPick />
        <span className="atlas-sbar-end">
          <button type="button" className="btn-link" aria-pressed={past !== null} onClick={() => void togglePast()}>
            past
          </button>
          <button type="button" className="btn-link" onClick={newChat}>
            new
          </button>
        </span>
      </div>
      <ErrorLine error={warning ?? backendsError ?? pastError} />
      {past !== null ? <Past /> : <Thread />}
      {needsAck && reach ? (
        <div className="atlas-ack" role="alert">
          <p className="atlas-small red" style={{ margin: 0 }}>
            On this server {agent === "codex" ? "Codex" : "Claude Code"} {reach[agent as "claude" | "codex"]}. What it reads (the model’s answers,
            census answers, names and notes, attachments) can carry instructions, and nothing stands between them and that shell.
          </p>
          <p className="muted atlas-small" style={{ margin: "4px 0" }}>
            Without {reach.env}=home on the server it would only read and write files in its own work folder; an endpoint has no shell at all.
          </p>
          <button
            type="button"
            className="btn-ghost"
            onClick={() => {
              try {
                localStorage.setItem(ACK_KEY, "1");
              } catch {
                // asked again next time
              }
              setAcked(true);
            }}
          >
            I understand: let it run here
          </button>
        </div>
      ) : null}
      {stale ? (
        // the server ends its process only when asked: commands it left running in the background would end with it
        <div className="atlas-stale" role="status">
          <p className="atlas-small" style={{ margin: 0 }}>
            This chat started before its instructions or tools last changed, and runs those it started with. Restart it to take them up; that
            also stops any commands it left running.
          </p>
          <button type="button" className="btn-ghost plain" disabled={busy} title={busy ? "stop the reply first" : undefined} onClick={() => void restartChat()}>
            restart
          </button>
        </div>
      ) : null}
      {f && !busy && !needsAck ? (
        <div className="atlas-actions">
          <button type="button" className="btn-ghost plain" onClick={() => void explainFocus()}>
            explain {at}
          </button>
          {f.kind === "neuron" ? (
            <button type="button" className="btn-ghost plain" onClick={askWeb}>
              what does its web do?
            </button>
          ) : null}
        </div>
      ) : null}
      {pending.length ? (
        <div className="atlas-files">
          {pending.map((a, i) => (
            <span key={i} className={`atlas-file${a.id ? "" : " up"}`}>
              {a.name}
              <button type="button" aria-label={`remove ${a.name}`} onClick={() => removePending(i)}>
                ×
              </button>
            </span>
          ))}
        </div>
      ) : null}
      <form
        className="atlas-sform"
        hidden={needsAck}
        onSubmit={(e) => {
          e.preventDefault();
          if (busy && !input.trim()) void stop();
          else send();
        }}
      >
        <button type="button" className="btn-ghost plain" aria-label="attach files" title="attach files (or drop or paste them)" onClick={() => fileRef.current?.click()}>
          +
        </button>
        <input
          ref={fileRef}
          type="file"
          multiple
          hidden
          onChange={(e) => {
            void attachFiles([...(e.target.files ?? [])]);
            e.target.value = "";
          }}
        />
        <textarea
          ref={boxRef}
          rows={1}
          value={input}
          placeholder={busy ? "a message reaches it between tool calls" : "ask the scientist"}
          aria-label="a message for the scientist"
          spellCheck={false}
          onChange={(e) => useScientist.setState({ input: e.target.value })}
          onPaste={(e) => {
            const fs = [...(e.clipboardData?.files ?? [])];
            if (fs.length) {
              e.preventDefault();
              void attachFiles(fs);
            }
          }}
          onKeyDown={(e) => {
            if (e.key !== "Enter" || e.shiftKey) return;
            e.preventDefault();
            send();
          }}
        />
        <button type="submit" className={`btn-ghost${busy && !input.trim() ? " stop" : ""}`} aria-label={busy && !input.trim() ? "stop" : "send"}>
          {busy && !input.trim() ? "stop" : "send"}
        </button>
      </form>
      <Rubric style={{ fontSize: 14, margin: "4px 0 0" }}>
        The scientist reads the atlas with the same tools the page uses and can move this screen
        {reach ? (
          <>
            ; beyond them, here it <span className={homeShell ? "red" : undefined}>{reach[agent]}</span>
          </>
        ) : null}
        . What it says is a lead to check: it cites the neurons it read, so open them. <a href={href.lexicon("scientist")}>What the scientist is.</a>
      </Rubric>
    </div>
  );
};

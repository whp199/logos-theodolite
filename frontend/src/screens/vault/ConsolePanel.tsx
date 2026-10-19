import React from "react";
import { Apparatus, Rubric } from "../../components/kit";
import { useTheodoliteStore } from "../../store/useTheodoliteStore";
import { DESCRIBE_RECORDED, RAW_LISTS, VAULT_ANSWERS } from "./recorded";
import { FORMS, QUICK, clampL, clampN, esc, mutates, parseLql, pyRepr, type Parsed } from "./lql";
import { clock, prettyJson, replyTo } from "./live";
import { EmptyAnswer, MONO, rawJson, smallCaps } from "./bits";
import { useVaultUi } from "./ui";

const SRC = "E6_tool_audit_04_vault.partial.json";
const LIVE_PH = "[THE VAULT’S REPLY APPEARS WHEN THE RUNNING INSTRUMENT ANSWERS]";
const hasOwn = (o: object, k: string) => Object.prototype.hasOwnProperty.call(o, k);

type SelectP = Extract<Parsed, { kind: "select" }>;
const selKey = (p: SelectP) => p.concept.trim() + "@" + (p.layerS === null ? 35 : clampL(p.layerS));
const selLim = (p: SelectP) => (p.limitS === null ? 10 : parseInt(p.limitS, 10));
const selRecorded = (p: SelectP) => hasOwn(RAW_LISTS, selKey(p)) && selLim(p) >= 1;
const studiedDelete = (p: Extract<Parsed, { kind: "delete" }>) => {
  const i = clampL(p.layerS);
  const n = clampN(p.slotS);
  return [40, 50, 55, 58].includes(i) && ["she", "he"].some((w) => VAULT_ANSWERS[w + "@" + i].some((r) => r[0] === n));
};
const unread = (p: Parsed, word: string) => (new RegExp("\\b" + word + "\\b", "i").test(p.stmt) ? "written but not read" : "not written");
const layerTxt = (s: string) => {
  const i = clampL(s);
  return "written " + s + (parseInt(s, 10) !== i ? ", read as " + i : "") + ": layer " + (i + 1);
};
const slotTxt = (s: string) => {
  const i = clampN(s);
  return String(i) + (parseInt(s, 10) !== i ? " (written " + s + ")" : "");
};

/** "how the vault reads it" (board: parsedRows). */
function parsedRows(p: Parsed, inForce: number, live: boolean): { k: string; v: string }[] {
  switch (p.kind) {
    case "describe":
      return [
        { k: "form", v: "the vault describes itself" + (p.alias ? " (" + p.alias + " runs the same code as DESCRIBE)" : "") },
        { k: "reply", v: live ? "answered by the running instrument" : "recorded on this page" },
      ];
    case "rollback":
      return [
        { k: "form", v: "undo every edit, in all sixty layers" },
        { k: "in force now", v: inForce + (inForce === 1 ? " edit" : " edits") },
      ];
    case "select":
      return [
        { k: "form", v: "find the nearest neurons" + (p.from === "POLES" ? " (POLES: the same search as EDGES)" : "") },
        { k: "word", v: "“" + p.concept + "”, sought as one word-piece: with a leading space if that is one piece, else bare, else its first piece" },
        { k: "layer", v: p.layerS === null ? "not written: index 35, layer 36" : layerTxt(p.layerS) },
        { k: "how many", v: p.limitS === null ? "not written: 10" : String(parseInt(p.limitS, 10)) },
        { k: "reply", v: live ? "answered by the running instrument" : selRecorded(p) ? "recorded on this page" : "not recorded on this page" },
      ];
    case "pole":
      return [
        { k: "form", v: "both faces of one neuron" },
        { k: "layer", v: layerTxt(p.layerS) },
        { k: "neuron", v: slotTxt(p.slotS) },
        { k: "face asked", v: (p.pol === "-" ? "negative (POLARITY -)" : "positive (POLARITY +)") + (p.polGiven ? "" : ", " + unread(p, "POLARITY")) },
      ];
    case "delete":
      return [
        { k: "form", v: "silence one neuron" },
        { k: "layer", v: layerTxt(p.layerS) },
        { k: "neuron", v: slotTxt(p.slotS) },
      ];
    case "update":
      return [
        { k: "form", v: "make one neuron say a word more strongly" },
        { k: "layer", v: layerTxt(p.layerS) },
        { k: "neuron", v: slotTxt(p.slotS) },
        { k: "word", v: "“" + p.word + "”" },
        { k: "strength", v: p.strength === null ? unread(p, "STRENGTH") + ": 2.5" : p.strength },
        { k: "face", v: (p.pol === "-" ? "negative (POLARITY -)" : "positive (POLARITY +)") + (p.polGiven ? "" : ", " + unread(p, "POLARITY")) },
      ];
    case "insert":
      return [
        { k: "form", v: "choose a neuron and make it say a word" },
        { k: "layer", v: layerTxt(p.layerS) },
        { k: "subject", v: "“" + p.subj + "”" },
        { k: "object", v: "“" + p.obj + "”" },
        { k: "strength", v: p.strength === null ? unread(p, "STRENGTH") + ": 2.5" : p.strength },
        { k: "neuron", v: "chosen by the vault: the one whose input side lies nearest “" + p.subj + "”" },
      ];
    case "error":
      if (p.empty) return [{ k: "form", v: "nothing written yet" }];
      if (p.num) return [{ k: "form", v: "an edit whose STRENGTH is not a number" }];
      return [
        { k: "form", v: "not a statement the vault reads" },
        { k: "compare", v: "the seven forms in the grammar beside this" },
      ];
  }
}

export const ConsolePanel: React.FC = () => {
  const ui = useVaultUi();
  const conn = useTheodoliteStore((s) => s.conn);
  const text = useTheodoliteStore((s) => s.larqlQuery);
  const setText = useTheodoliteStore((s) => s.setLarqlQuery);
  const executeLarql = useTheodoliteStore((s) => s.executeLarql);
  const larqlLog = useTheodoliteStore((s) => s.larqlLog);
  const larqlPending = useTheodoliteStore((s) => s.larqlPending);
  const inForce = useTheodoliteStore((s) => s.activeEdits.length);
  const live = conn === "open";

  const conP = parseLql(text);
  const R = ui.conRun === null ? null : parseLql(ui.conRun);
  const entry = ui.conRun !== null ? replyTo(larqlLog, ui.conRun, ui.conRunAt) : null;
  const pending = larqlPending && ui.conRun !== null && entry === null;

  const run = () => {
    if (!text.trim()) return;
    ui.set({ conRun: text, conRunAt: Date.now() });
    executeLarql(text);
  };

  // ---------- the vault's reply: the running instrument's, else what the study recorded (labelled as such)
  let json = "";
  let caption = "";
  let ph = "[THE VAULT’S RAW REPLY, AS JSON, APPEARS HERE WHEN YOU RUN THE STATEMENT]";
  let phNote = "This page answers from the study’s records; anything else is answered by the running instrument.";
  let unanswered: string | null = null;
  if (R !== null) {
    if (entry && entry.ok) {
      json = esc(prettyJson(entry.data));
      caption =
        "Answered by the running instrument at " +
        clock(entry.at) +
        (entry.via === "ws" ? ", over the WebSocket" : ", over HTTP") +
        ", exactly as it returned it.";
    } else if (entry && !entry.ok && entry.failure === "refused") {
      json = esc("{\n  " + '"detail": ' + JSON.stringify(entry.error) + "\n}");
      caption = "The server refused it, in the words of larql_vault.py.";
    } else if (pending) {
      ph = LIVE_PH;
      phNote =
        "Sent. Each statement waits its turn behind any sighting under the server’s one GPU lock. A sighting of every word-piece " +
        "an answer holds can take a minute or more, and an edit never lands between two word-pieces of one answer.";
    } else {
      if (entry && !entry.ok) unanswered = entry.error ?? "no reason given";
      // no live reply: either the server is not answering, or this statement's request failed on the way
      const unanswered2 = !live || (entry !== null && !entry.ok);
      const noEdit = unanswered2 ? " No answer came from the vault, so the edit was not made." : "";
      if (R.kind === "describe") {
        json = prettyJson(DESCRIBE_RECORDED);
        caption =
          "Recorded 16 October 2026, under “describe” in " +
          SRC +
          ". The study’s script left out the empty list of patches; no edits were in force." +
          (R.alias ? " " + R.alias + " runs the same code; this is the recorded DESCRIBE." : "") +
          (inForce ? " With the " + inForce + (inForce === 1 ? " edit" : " edits") + " now in force, the live reply would count and list them." : "");
      } else if (R.kind === "select" && selRecorded(R)) {
        const lim = selLim(R);
        json = rawJson(selKey(R), Math.min(lim, 10));
        caption =
          "Recorded 16 October 2026, lists[\"" +
          selKey(R) +
          "\"] in " +
          SRC +
          ", as the study’s script kept it: neuron (n), face (pol), nearness (cos) and five word-pieces through the lens (jv) and to the naked eye (nv). The live reply also carries the word-piece it resolved, each word-piece’s logit and the time taken." +
          (lim > 10 ? " The study asked for ten; the rest: [MEASURED IN A LIVE SIGHTING]." : "") +
          (lim < 10 ? " Shown: the first " + lim + " of the ten recorded." : "");
      } else if (R.kind === "error") {
        json = esc("{\n  " + '"detail": ' + JSON.stringify(R.err) + "\n}");
        caption = "The server, when it answers, refuses this with HTTP 400, in these words of larql_vault.py.";
      } else {
        ph = LIVE_PH;
        if (R.kind === "select") {
          const lower = R.concept.trim().toLowerCase() + "@" + (R.layerS === null ? 35 : clampL(R.layerS));
          phNote =
            lower !== selKey(R) && hasOwn(RAW_LISTS, lower)
              ? "The study searched this word in lower case, and the vault reads the word exactly."
              : "This page holds only the study’s twenty-four searches: “she”, “he”, “her” and “woman” at index 20, 30, 40, 50, 55 and 58, up to ten each.";
        } else if (R.kind === "pole") {
          phNote = "The study did not record this reply. A neuron’s folio shows both faces in full.";
        } else if (R.kind === "delete") {
          phNote =
            (studiedDelete(R)
              ? "The study ran this very statement in its trials, but kept only the effect, in the ledger, not the reply."
              : "The study did not record this reply.") + noEdit;
        } else if (R.kind === "rollback") {
          phNote =
            "Its reply was not kept. In the study, readings after ROLLBACK matched the untouched model exactly: the largest difference was 0.0." +
            (unanswered2 ? " No answer came from the vault, so nothing was undone." : "");
        } else {
          phNote = "No reply to an UPDATE or INSERT is kept in the repository." + noEdit;
        }
      }
    }
  }

  const stale = ui.conRun !== null && ui.conRun !== text;
  const restBody = '{"query": ' + JSON.stringify(text) + "}";
  const wsBody = '{"action": "larql", "query": ' + JSON.stringify(text) + "}";
  const mcpCall = "query_larql_vault(lql_query=" + pyRepr(text) + ")";
  const use = (t: string) => () => {
    setText(t);
    ui.set({ conRun: null, conRunAt: null });
  };
  const monoSmall: React.CSSProperties = { fontFamily: MONO, fontSize: 13, overflowWrap: "anywhere" };

  return (
    <section
      role="tabpanel"
      id="panel-lql"
      aria-labelledby="tab-lql"
      style={{ paddingTop: 34, display: "flex", flexWrap: "wrap", gap: "40px 64px", alignItems: "flex-start" }}
    >
      <div style={{ flex: "999 1 640px", minWidth: 0, display: "flex", flexDirection: "column", gap: 20 }}>
        <h2 className="h-sec">ii · write in the vault’s own words</h2>
        <p style={{ margin: 0, fontSize: 20, lineHeight: 1.5, maxWidth: 760 }}>
          Everything the other tabs ask of the vault can be written out by hand, in the instrument’s query language, which it calls LQL.
          Write one statement, run it, and read the vault’s reply exactly as the instrument returns it.
        </p>
        <Rubric style={{ maxWidth: 720 }}>
          Begin from an example in the grammar beside this box. Here the layers are counted from nought, so{" "}
          <span style={{ fontFamily: MONO, fontStyle: "normal", fontSize: 14, overflowWrap: "anywhere" }}>layer = 50</span> means layer 51
          on every other page.
        </Rubric>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <label htmlFor="lql-text" style={smallCaps()}>
            the statement
          </label>
          <textarea
            id="lql-text"
            rows={4}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                run();
              }
            }}
            spellCheck={false}
            autoComplete="off"
            style={{
              fontFamily: MONO,
              fontSize: 16,
              lineHeight: 1.55,
              color: "var(--ink)",
              background: "transparent",
              border: "1px solid var(--control)",
              borderRadius: 2,
              padding: "12px 14px",
              width: "100%",
              minHeight: 116,
              boxSizing: "border-box",
              resize: "vertical",
              overflowWrap: "anywhere",
            }}
          />
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "12px 20px" }}>
          <button className="btn-primary" type="button" onClick={run}>
            run it
          </button>
          <button
            type="button"
            onClick={() => {
              setText("");
              ui.set({ conRun: null, conRunAt: null });
            }}
            style={{
              fontFamily: "inherit",
              fontSize: 17,
              fontVariantCaps: "all-small-caps",
              letterSpacing: "0.16em",
              color: "var(--ink)",
              background: "transparent",
              border: "1px solid var(--control)",
              borderRadius: 2,
              padding: "0 18px",
              minHeight: 48,
              cursor: "pointer",
            }}
          >
            clear
          </button>
          {stale ? (
            <span style={{ fontSize: 16, fontStyle: "italic", color: "var(--muted)" }}>You have changed the statement. Run it again to see its reply.</span>
          ) : null}
        </div>
        {mutates(conP) ? (
          <p style={{ margin: 0, fontSize: 16, color: "var(--muted)", maxWidth: 760 }}>
            {conP.kind === "rollback"
              ? "Running this undoes every edit in force."
              : "Running this changes the model: the edit joins the edits in force, which you can undo on the edit tab or with ROLLBACK."}
          </p>
        ) : null}
        <div style={{ display: "flex", flexDirection: "column", gap: 6, paddingTop: 14, borderTop: "1px solid var(--hair-soft)" }}>
          <h3 className="h-sub" style={{ margin: "0 0 4px" }}>
            how the vault reads it
          </h3>
          {parsedRows(conP, inForce, live).map((p, i) => (
            <div
              key={i}
              style={{
                display: "grid",
                gridTemplateColumns: "minmax(96px, 150px) minmax(0, 1fr)",
                gap: "2px 18px",
                alignItems: "baseline",
                padding: "5px 0 6px",
                borderBottom: "1px solid var(--hair-faint)",
              }}
            >
              <span style={{ fontVariantCaps: "all-small-caps", letterSpacing: "0.14em", fontSize: 15, color: "var(--muted)" }}>{p.k}</span>
              <span className="num" style={{ fontSize: 17, overflowWrap: "anywhere" }}>
                {p.v}
              </span>
            </div>
          ))}
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }} aria-live="polite">
          <h3 className="h-sub">the vault’s reply</h3>
          {unanswered !== null ? (
            <p className="status-line" role="status">
              The server is not answering ({unanswered}). What follows is from the study’s records, where it has any.
            </p>
          ) : null}
          {caption ? <p style={{ margin: 0, fontSize: 15, color: "var(--muted)", maxWidth: 820 }}>{caption}</p> : null}
          {json ? (
            <pre className="raw" style={{ maxHeight: 420, padding: "12px 0", fontSize: 13, lineHeight: 1.55, overflowWrap: "anywhere" }}>
              {json}
            </pre>
          ) : (
            <EmptyAnswer head={ph} pad="28px 0 30px">
              {phNote}
            </EmptyAnswer>
          )}
        </div>
      </div>

      <aside aria-labelledby="grammar-head" style={{ flex: "1 1 400px", minWidth: 0, display: "flex", flexDirection: "column", gap: 12 }}>
        <h3 className="h-sub" id="grammar-head">
          the grammar the vault reads
        </h3>
        <p style={{ margin: 0, fontSize: 15, color: "var(--muted)" }}>
          Seven forms, tried in this order; the first that fits is run, and anything else is refused. Square brackets mark what may be left
          out.
        </p>
        <div style={{ display: "flex", flexDirection: "column", borderTop: "1px solid var(--hair)" }}>
          {FORMS.map(([form, gloss, example]) => (
            <div key={form} style={{ display: "flex", flexDirection: "column", gap: 3, padding: "10px 0 8px", borderBottom: "1px solid var(--hair-faint)" }}>
              <code style={{ fontFamily: MONO, fontSize: 13.5, lineHeight: 1.5, color: "var(--ink)", overflowWrap: "anywhere" }}>{form}</code>
              <span style={{ fontSize: 15, lineHeight: 1.45 }}>{gloss}</span>
              <button
                type="button"
                onClick={use(example)}
                aria-label={"Write the example in the box: " + example}
                style={{
                  alignSelf: "flex-start",
                  fontFamily: MONO,
                  fontSize: 12.5,
                  textAlign: "left",
                  color: "var(--rubric)",
                  background: "transparent",
                  border: 0,
                  padding: "6px 0",
                  minHeight: 44,
                  cursor: "pointer",
                  textDecoration: "underline",
                  textDecorationThickness: 1,
                  textUnderlineOffset: 3,
                  overflowWrap: "anywhere",
                }}
              >
                {example}
              </button>
            </div>
          ))}
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 6, fontSize: 15, lineHeight: 1.45 }}>
          <span style={{ fontVariantCaps: "all-small-caps", letterSpacing: "0.16em", color: "var(--muted)" }}>counting</span>
          <span>
            The vault counts its layers from nought: write 0 for layer 1, 50 for layer 51 and 59 for layer 60. Neurons keep their own numbers,
            0 to 21,503.
          </span>
          <span style={{ fontVariantCaps: "all-small-caps", letterSpacing: "0.16em", color: "var(--muted)", paddingTop: 6 }}>small print</span>
          <span>
            Keywords may be written in any case, and a closing semicolon is dropped; the word itself is read exactly, so “She” is not “she”.
            Words go in single or double quotes and may not contain one. A layer past 59 is read as 59, a neuron past 21,503 as 21,503. The
            parts in square brackets are read only in the order shown. STRENGTH takes no minus sign: one written with it is passed over, with
            all that follows, so the edit runs at 2.5 on the positive face. To push the other way, write POLARITY -.
          </span>
        </div>
      </aside>

      <Apparatus title="the query language" label="Apparatus for the query language" style={{ flex: "1 1 100%", minWidth: 0 }}>
        <dl className="dl">
          <dt>technical name</dt>
          <dd>
            <span className="mono-wrap">LarqlVaultEngine.execute_lql(query)</span>. The statement is trimmed and closing semicolons dropped.
            DESCRIBE (or SHOW LAYERS, STATUS) and ROLLBACK must then stand alone; the other five forms are found, in turn, by a
            regular-expression search, so stray words before or after them are passed over. The name follows LARQL’s LQL [10]; the grammar
            is this project’s own.
          </dd>
          <dt>sent as</dt>
          <dd style={{ margin: 0, display: "flex", flexDirection: "column", gap: 6, minWidth: 0 }}>
            <span>
              over HTTP, <code style={monoSmall}>POST /api/larql {restBody}</code>
            </span>
            <span>
              over the WebSocket, <code style={monoSmall}>/ws/theodolite {wsBody}</code>, answered by{" "}
              <code style={monoSmall}>{'{"type": "larql_result", "ok": true, "data": …}'}</code>
            </span>
            <span>
              to the MCP scientist, <code style={monoSmall}>{mcpCall}</code>
            </span>
          </dd>
          <dt>refusals</dt>
          <dd>
            a statement that fits no form, or fails, comes back as HTTP 400 with the error as <span className="mono-wrap">detail</span>, or
            over the WebSocket as <span style={monoSmall}>{'{"type": "larql_result", "ok": false, "error": …}'}</span>. Each statement waits
            its turn behind any sighting under the server’s one GPU lock, which a sweep holds from writing its answer to sending its
            last word-piece’s sighting: with every word-piece sighted that can take a minute or more (about 0.7 s for each), and an
            edit never lands between two word-pieces of one answer. Over the WebSocket a statement also waits behind the sweeps sent
            before it on the same socket.
          </dd>
          <dt>recorded here</dt>
          <dd>
            the replies to DESCRIBE and to the study’s twenty-four searches,{" "}
            <span style={monoSmall}>
              SELECT * FROM POLES NEAREST 'she' | 'he' | 'her' | 'woman' WHERE layer = 20 | 30 | 40 | 50 | 55 | 58 LIMIT 10
            </span>{" "}
            (EDGES gives the same), as the study’s script kept them. Characters outside Unicode’s basic plane, and the replacement
            character, are shown as \u escapes; the file itself writes every non-ASCII character that way. Any other reply:{" "}
            <span className="placeholder">[MEASURED IN A LIVE SIGHTING]</span>.
          </dd>
          <dt>describe pole</dt>
          <dd>
            returns both faces through the lens and to the naked eye, the norms of the neuron’s three weight vectors, and{" "}
            <span className="mono-wrap">gate_trigger_tokens</span>: a read covector passed through J as if it were a state, one label for two
            faces, holding about 6–8% of the reproducible input signal (REPORT §3.4, bug 7). Stored activation contexts per pole are the
            proposed fix.
          </dd>
          <dt>the original instrument’s buttons</dt>
          <dd style={{ margin: 0, display: "flex", flexWrap: "wrap", alignItems: "center", gap: "6px 10px" }}>
            {QUICK.map(([label, stmt]) => (
              <button
                key={label}
                type="button"
                onClick={use(stmt)}
                aria-label={label + ": write the original instrument’s statement in the box, " + stmt}
                className="btn-ghost plain"
                style={{ fontSize: 15, padding: "0 12px" }}
              >
                {label}
              </button>
            ))}
            <span style={{ color: "var(--muted)" }}>
              The original instrument ran these at once, on its chosen layer, index 44 (layer 45) when it opened; here they write the
              statement into the box, at that index, for you to run.
            </span>
          </dd>
          <dt>caveat</dt>
          <dd>
            DELETE, UPDATE and INSERT change the running model for every sighting and every MCP client until ROLLBACK. A nearness reply is a
            projected contribution, not a counterfactual (REPORT §3.3).
          </dd>
        </dl>
      </Apparatus>
    </section>
  );
};

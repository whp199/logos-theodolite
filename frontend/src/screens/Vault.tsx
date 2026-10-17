import React, { useEffect, useRef } from "react";
import { Apparatus, Page, Rubric } from "../components/kit";
import { href, type Route } from "../router";
import { useTheodoliteStore } from "../store/useTheodoliteStore";
import { Alidade } from "./vault/Alidade";
import { ConsolePanel } from "./vault/ConsolePanel";
import { EditPanel } from "./vault/EditPanel";
import { FindPanel } from "./vault/FindPanel";
import { HistoryPanel } from "./vault/HistoryPanel";
import { parseLql, statementLayerNo } from "./vault/lql";
import { num } from "./vault/live";
import { useVaultUi, type VaultTab } from "./vault/ui";

/**
 * The Vault: the catalogue of the model's 1,290,240 feed-forward neurons (approved board Vault.dc.html, folio vi).
 * Find, edit, the ledger and a console in the vault's own query language, all sent through the store's
 * executeLarql to LarqlVaultEngine (backend/larql_vault.py). Layers are counted 1-60 on the page and converted to
 * the vault's count from nought (repo index) only in the statements.
 */

const TABS: [VaultTab, string, string][] = [
  ["find", "find", "look a neuron up by its words"],
  ["edit", "edit", "change one neuron"],
  ["history", "history", "what was asked before"],
  ["lql", "write LQL", "type a query by hand"],
];

export const Vault: React.FC<{ route: Route }> = ({ route }) => {
  const ui = useVaultUi();
  const conn = useTheodoliteStore((s) => s.conn);
  const editsInForce = useTheodoliteStore((s) => s.activeEdits.length);
  const larqlQuery = useTheodoliteStore((s) => s.larqlQuery);
  const executeLarql = useTheodoliteStore((s) => s.executeLarql);


  // the ledger must match the running vault (edits outlive a page reload): ask it to describe itself once per visit
  const described = useRef(false);
  useEffect(() => {
    if (conn === "open" && !described.current) {
      described.current = true;
      executeLarql("DESCRIBE");
    }
  }, [conn, executeLarql]);

  // ---------- the alidade follows the layer being worked in
  const fL = num(ui.fLayer);
  const fLOk = fL !== null && fL >= 1 && fL <= 60;
  const eL = num(ui.eLayer);
  const eLOk = eL !== null && eL >= 1 && eL <= 60;
  const conLayer = statementLayerNo(parseLql(larqlQuery));
  const aliL = ui.tab === "edit" ? (eLOk ? eL : 36) : ui.tab === "lql" && conLayer !== null ? conLayer : fLOk ? fL : 51;
  const eN = num(ui.eN);
  const nTxt = eN !== null && eN >= 0 && eN <= 21503 ? String(eN) : "…";
  const aliWhat =
    ui.tab === "edit"
      ? ui.eKind === "update"
        ? "editing neuron " + nTxt
        : "silencing neuron " + nTxt
      : ui.tab === "lql"
        ? "reading a statement written by hand"
        : "seeking “" + (ui.fWord.trim() || "…") + "”";

  // ---------- tabs (arrow keys move between them, as tabs should)
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const onTabKey = (e: React.KeyboardEvent, i: number) => {
    let j = -1;
    if (e.key === "ArrowRight") j = (i + 1) % TABS.length;
    else if (e.key === "ArrowLeft") j = (i - 1 + TABS.length) % TABS.length;
    else if (e.key === "Home") j = 0;
    else if (e.key === "End") j = TABS.length - 1;
    if (j < 0) return;
    e.preventDefault();
    ui.set({ tab: TABS[j][0] });
    tabRefs.current[j]?.focus();
  };

  return (
    <Page
      route={route}
      folio="folio vi"
      footer={
        <>
          Reading google/gemma-4-31b-it (4-bit) through the J++ lens · 60 layers · 5,376 dimensions ·{" "}
          <a href={href.lineage()}>the lineage of these ideas</a>
        </>
      }
    >
      <section aria-labelledby="vault-title" style={{ display: "flex", flexWrap: "wrap", gap: "32px 56px", padding: "40px 0 34px", alignItems: "center" }}>
        <div style={{ flex: "999 1 560px", minWidth: 0, display: "flex", flexDirection: "column", gap: 18 }}>
          <p className="h-sec">i · the catalogue of neurons</p>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <h1 id="vault-title" style={{ margin: 0, fontSize: 68, lineHeight: 1, fontWeight: 500, letterSpacing: "-0.01em", color: "var(--ink)" }}>
              The Vault
            </h1>
            <p style={{ margin: 0, fontSize: 25, fontStyle: "italic", color: "var(--muted)", lineHeight: 1.3 }}>
              where the model keeps its <span className="lining">1,290,240</span> neurons
            </p>
          </div>
          <p style={{ margin: "6px 0 0", fontSize: 21, lineHeight: 1.55, maxWidth: 640 }}>
            <span aria-hidden="true" style={{ float: "left", fontSize: 82, lineHeight: 0.78, fontWeight: 500, color: "var(--gilt-numeral)", margin: "7px 12px 0 0" }}>
              E
            </span>
            <span aria-hidden="true">very</span>
            <span className="visually-hidden">Every</span> layer of the model holds 21,504 neurons, and sixty layers hold them all. Each
            neuron, when it fires, writes toward some words and away from others, and it has two faces, one for each way it can fire. The vault
            is their catalogue: look a neuron up by its words, change one, and read back what was asked before.
          </p>
          <p style={{ margin: 0, fontSize: 16, color: "var(--muted)", maxWidth: 640 }}>
            The vault borrows its names from LARQL, but its query words are its own, and every search runs in the instrument’s own Python, not
            in that program.
          </p>
          <Rubric style={{ maxWidth: 640 }}>
            Begin with find, below. Type a word, choose a layer, and search: the vault lists the neurons whose words lie nearest to yours.
          </Rubric>
        </div>
        <Alidade layerNo={aliL} what={aliWhat} />
      </section>

      <Apparatus title="the vault and its names" label="Apparatus for the vault and its names" style={{ padding: "12px 0 22px" }}>
        <dl className="dl">
          <dt>technical name</dt>
          <dd>
            <span className="mono-wrap">LarqlVaultEngine</span> in <span className="mono-wrap">backend/larql_vault.py</span>: a static index of
            the GeGLU feed-forward neurons (Shazeer 2020) [15] of all sixty layers, read through the J++ lens, with edits held in memory. The
            code calls it a J-VINDEX3 engine.
          </dd>
          <dt>its names</dt>
          <dd>
            After LARQL, the reference implementation of VINDEX3 by C. Hayuk, under Apache-2.0 [10]:{" "}
            <a href="https://github.com/chrishayuk/larql">github.com/chrishayuk/larql</a>. The vault asks the installed binaries only for their
            versions (<span className="num">vindex 0.8.0 · larql 0.2.0</span>). Its EDGES and POLES searches, its edits and ROLLBACK are this
            project’s own syntax, not LARQL’s LQL.
          </dd>
          <dt>faces · poles</dt>
          <dd>
            A neuron’s two faces are its two poles, positive and negative, after Cammarata (Goodfire 2026) [9]: 21,504 × 2 = 43,008 poles in
            each layer, 2,580,480 in all. In Gemma-4 they are truly two-sided, through the sign of the up-projection (REPORT §3.2).
          </dd>
          <dt>lens · naked eye</dt>
          <dd>
            The lens is the J++ lens (Ayonrinde &amp; Lindsey 2026) [4], after the Jacobian lens of Gurnee et al. 2026 [2], fitted on the base
            model and applied here to the instruct model (REPORT §3.4, bug 9). The naked eye is the logit lens, unembed(h) (nostalgebraist
            2020) [8].
          </dd>
          <dt>as recorded</dt>
          <dd className="num">
            DESCRIBE, 16 October 2026: d<sub>model</sub> 5,376 · 60 layers · 21,504 neurons in each, 1,290,240 in all · 262,144 word-pieces ·
            global-attention layers at index 5, 11, 17, 23, 29, 35, 41, 47, 53, 59 (layers 6, 12 … 60) · format “VINDEX3 + J-VINDEX (On-the-Fly
            W4A16 GPU)” · no edits in force.
          </dd>
          <dt>the verdict</dt>
          <dd>
            “Neat as a question, oversold as built” (REPORT §3.3). LARQL’s own README warns that “a projected contribution is not a
            counterfactual” [10], and a nearness search is exactly a projected contribution. The vault earns its keep if neuron labels rest on
            stored activations, output labels are read only from layer 41 up (index 40), they are gated by agreement between the lens and the
            naked eye, and each is checked by a causal test.
          </dd>
          <dt>counting</dt>
          <dd>
            Layers are counted 1–60 on this page; every statement sent to the vault writes them from nought, layer n as{" "}
            <span className="mono-wrap">layer = n − 1</span>.
          </dd>
        </dl>
      </Apparatus>

      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          alignItems: "baseline",
          gap: "8px 22px",
          padding: "16px 0 18px",
          borderTop: "1px solid var(--hair-soft)",
          borderBottom: "1px solid var(--hair-soft)",
        }}
      >
        <span className="label-sc">the vault holds</span>
        <span className="num" style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: "6px 18px", fontSize: 18 }}>
          <span>60 layers</span>
          <span aria-hidden="true" style={{ color: "var(--gilt)", fontSize: 12 }}>
            ◆
          </span>
          <span>21,504 neurons in each</span>
          <span aria-hidden="true" style={{ color: "var(--gilt)", fontSize: 12 }}>
            ◆
          </span>
          <span>2,580,480 faces in all</span>
          <span aria-hidden="true" style={{ color: "var(--gilt)", fontSize: 12 }}>
            ◆
          </span>
          <span>262,144 word-pieces they can write toward</span>
        </span>
        <span className="num" style={{ marginLeft: "auto", fontSize: 16, color: "var(--muted)" }} role="status" aria-live="polite">
          edits in force: <span style={{ color: "var(--ink)" }}>{editsInForce}</span>
        </span>
      </div>

      <div
        role="tablist"
        aria-label="What to do in the vault"
        style={{ display: "flex", flexWrap: "wrap", gap: "4px 44px", paddingTop: 26, borderBottom: "1px solid var(--hair)" }}
      >
        {TABS.map(([key, label, gloss], i) => {
          const sel = ui.tab === key;
          return (
            <button
              key={key}
              ref={(el) => {
                tabRefs.current[i] = el;
              }}
              type="button"
              role="tab"
              id={`tab-${key}`}
              aria-selected={sel}
              aria-controls={`panel-${key}`}
              tabIndex={sel ? 0 : -1}
              onClick={() => ui.set({ tab: key })}
              onKeyDown={(e) => onTabKey(e, i)}
              style={{
                fontFamily: "inherit",
                background: "transparent",
                border: 0,
                borderBottom: sel ? "2px solid var(--gilt)" : "2px solid transparent",
                marginBottom: -1,
                padding: "8px 2px 12px",
                minHeight: 64,
                cursor: "pointer",
                display: "flex",
                flexDirection: "column",
                alignItems: "flex-start",
                gap: 0,
                textAlign: "left",
                color: sel ? "var(--ink)" : "var(--muted)",
              }}
            >
              <span style={{ fontSize: 22, fontVariantCaps: "all-small-caps", letterSpacing: "0.2em", lineHeight: 1.2 }}>{label}</span>
              <span style={{ fontSize: 15, fontStyle: "italic", color: "var(--muted)", lineHeight: 1.3 }}>{gloss}</span>
            </button>
          );
        })}
      </div>

      {ui.tab === "find" ? <FindPanel /> : null}
      {ui.tab === "edit" ? <EditPanel /> : null}
      {ui.tab === "history" ? <HistoryPanel /> : null}
      {ui.tab === "lql" ? <ConsolePanel /> : null}
    </Page>
  );
};

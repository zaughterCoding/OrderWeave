import { useState, type ReactNode } from "react";
import type { TraceNode, Turn } from "../model";
import { countKinds } from "../model";
import { useT } from "../i18n";
import { Code } from "./Json";

const STAGE: Record<string, string> = {
  plan: "request-plan extraction", intent: "intent recognition", roles: "role model", agent: "agent model",
  confirm: "confirmation classifier", composer: "composer", compact: "context compaction", extract: "fact extraction",
  memory: "memory", retrieval: "retrieval rerank",
};

const DOT: Record<string, [string, string]> = {
  node: ["d-node", "◆"], model: ["d-model", "M"], tool: ["d-tool", "T"], guard: ["d-guard", "!"],
  step: ["d-step", "≡"], check: ["d-step", "✓"], context: ["d-context", "⌘"], route: ["d-route", "→"],
  agent: ["d-agent", "A"], revision: ["d-guard", "↺"],
};

function snippet(value: unknown, n = 90): string {
  const text = typeof value === "string" ? value : JSON.stringify(value ?? "");
  return text.length > n ? text.slice(0, n) + "…" : text;
}

function modelOutput(node: TraceNode): string {
  const blocks: any[] = node.end?.output || [];
  const calls = blocks.filter((b) => b.type === "tool_use").map((b) => b.name);
  if (calls.length) return "→ " + calls.join(", ");
  const text = blocks.find((b) => b.type === "text")?.text;
  return text ? snippet(text.replace(/\s+/g, " "), 110) : "";
}

function summary(node: TraceNode, turn: Turn): ReactNode {
  const s = node.start || {};
  const e = node.end || {};
  switch (node.kind) {
    case "node": {
      const kids = node.children.map((id) => turn.nodes[id]).filter(Boolean);
      const route = kids.find((k) => k.kind === "route");
      if (route) return <>primary <b>{route.start.primary}</b>{route.start.supporting?.length ? ` + ${route.start.supporting.join(", ")}` : ""} · {snippet(route.start.reason, 60)}</>;
      if (e.route && node.name === "route_agent") return <>route = <b>{e.route}</b></>;
      if (node.status === "waiting") return <>interrupt · waiting for the user's reply</>;
      const parts = [e.phase, e.next_role && `next → ${e.next_role}`, e.waiting_reason].filter(Boolean);
      return parts.join(" · ");
    }
    case "model": return modelOutput(node) || (node.status === "running" ? "…" : "");
    case "tool": {
      if (s.blocked_by) return <><b style={{ color: "var(--guard)" }}>blocked</b> by {s.blocked_by} · {snippet(s.error, 70)}</>;
      if (!s.success || s.result_success === false) return <><b style={{ color: "var(--guard)" }}>failed</b> · {snippet(s.error || s.result?.error, 70)}</>;
      return snippet(s.input, 90);
    }
    case "agent": return node.status === "running" ? `${(s.tools || []).length} tools available` : snippet(e.content, 100);
    case "route": return <>primary <b>{s.primary}</b> · {snippet(s.reason, 70)}</>;
    case "guard": {
      const findings: string[] = s.record?.findings || [];
      return findings.length ? <><b>{findings.length}</b> finding(s) · {findings.join(", ")}</> : <>mode {s.record?.mode || "pass"} · no findings</>;
    }
    case "check": return `${(s.accepted || []).length} accepted · ${(s.rejected || []).length} rejected`;
    case "context": return `${(s.user_blocks || []).length} user blocks · turn ${s.state?.turn ?? ""}`;
    case "step":
      if (node.name === "confirmation_judge") {
        const d: any[] = e.decisions || [];
        return d.length ? d.map((x) => `${x.op_id ?? ""} → ${x.approved ? "approved" : "not approved"}`).join(" · ") : "…";
      }
      return e.ops ? snippet(e.ops, 100) : "…";
    case "revision": return snippet(s.feedback, 100);
  }
  return "";
}

function Details({ node }: { node: TraceNode }) {
  const t = useT();
  const s = node.start || {};
  const e = node.end || {};
  if (node.kind === "model") {
    const u = e.usage || {};
    const totalIn = (u.input || 0) + (u.cache_read || 0) + (u.cache_write || 0);
    return (
      <>
        <Code title={t("systemPrompt")} value={s.system} fold />
        <Code title={`${t("recentMessages")} (${s.messages_total ?? 0})`} value={s.messages_tail} fold />
        {e.output && <Code title={t("output")} value={e.output} />}
        {e.usage && (
          <div className="sec usage">
            <span>{t("in")} <b>{totalIn.toLocaleString()}</b></span>
            <span>{t("cacheHit")} <b>{totalIn ? Math.round(((u.cache_read || 0) / totalIn) * 100) : 0}%</b></span>
            <span>{t("out")} <b>{(u.output || 0).toLocaleString()}</b></span>
            <span>¥ <b>{(e.cost_cny ?? 0).toFixed(4)}</b></span>
            {s.tools?.length ? <span>{t("tools")} <b>{s.tools.length}</b></span> : null}
          </div>
        )}
        {e.error && <Code title={t("error")} value={e.error} />}
      </>
    );
  }
  if (node.kind === "tool") {
    return (
      <>
        <Code title={t("input")} value={s.input} />
        <Code title={t("result")} value={s.result} fold />
      </>
    );
  }
  if (node.kind === "guard") {
    return (
      <>
        {s.draft !== s.final ? (
          <div className="vs">
            <Code title={t("draft")} value={s.draft} />
            <Code title={t("final")} value={s.final} />
          </div>
        ) : (
          <Code title={t("final")} value={s.final} fold />
        )}
        <Code title="record" value={s.record} fold />
      </>
    );
  }
  if (node.kind === "agent") {
    return (
      <>
        <div className="sec">
          <h5>{t("tools")} · {(s.tools || []).length}</h5>
          <div className="usage">{(s.tools || []).map((name: string) => <span key={name} className="chip tool">{name}</span>)}</div>
        </div>
        {(s.hooks || []).length > 0 && (
          <div className="sec">
            <h5>hooks</h5>
            <div className="usage">{s.hooks.map((name: string) => <span key={name} className="chip guard">{name}</span>)}</div>
          </div>
        )}
        {e.content != null && <Code title={t("output")} value={e.content} />}
      </>
    );
  }
  if (node.kind === "check") {
    return (
      <>
        <Code title={t("accepted")} value={s.accepted} />
        <Code title={t("rejected")} value={s.rejected} />
      </>
    );
  }
  if (node.kind === "context") {
    return (
      <>
        <Code title="state" value={s.state} fold />
        <Code title="user blocks" value={s.user_blocks} fold />
        {s.compaction && Object.keys(s.compaction).length > 0 && <Code title="compaction" value={s.compaction} fold />}
      </>
    );
  }
  const startKeys = Object.keys(s).length > 0;
  const endKeys = Object.keys(e).length > 0;
  return (
    <>
      {startKeys && <Code title={node.point ? "data" : t("input")} value={s} fold={node.kind === "node"} />}
      {endKeys && <Code title={t("output")} value={e} fold={node.kind === "node" || node.kind === "agent"} />}
    </>
  );
}

function Card({ id, turn, open, toggle, onJump }: { id: string; turn: Turn; open: Record<string, boolean>; toggle: (id: string) => void; onJump: () => void }) {
  const t = useT();
  const node = turn.nodes[id];
  if (!node) return null;
  const [cls, glyph] = DOT[node.kind] || ["d-step", "•"];
  const isOpen = open[id] ?? (node.status === "running" && node.kind !== "model");
  const name = node.kind === "model" ? STAGE[node.name] || node.name : node.kind === "check" ? node.name.replace(/_/g, " ") : node.name;
  const kids = node.children;
  return (
    <div className={"card" + (isOpen ? " open" : "") + (node.status === "running" ? " running" : "") + (node.status === "error" ? " err" : "")}>
      <div className="card-h" onClick={() => toggle(id)}>
        <span className={"dot " + cls}>{glyph}</span>
        <span className="name">{name}</span>
        <span className="sub">{summary(node, turn)}</span>
        {node.dbChanges > 0 && (
          <button className="jump" onClick={(ev) => { ev.stopPropagation(); onJump(); }}>→ {node.dbChanges} {t("dbChanges")}</button>
        )}
        {kids.length > 0 && !isOpen && <span className="chip plain">{kids.length} {t("children")}</span>}
        <span className="ms mono">{node.status === "running" ? <span className="spinner" /> : node.ms != null ? fmtMs(node.ms) : ""}</span>
        <span className="caret">›</span>
      </div>
      {isOpen && (
        <div className="card-b">
          {kids.length > 0 && (
            <div className="nest">
              {kids.map((k) => <Card key={k} id={k} turn={turn} open={open} toggle={toggle} onJump={onJump} />)}
            </div>
          )}
          {(node.kind !== "node" || !kids.length) && <Details node={node} />}
          {node.kind === "node" && kids.length > 0 && Object.keys(node.end || {}).length > 0 && (
            <Code title={t("output")} value={node.end} fold />
          )}
        </div>
      )}
    </div>
  );
}

export function fmtMs(ms: number) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
}

export function TraceTree({ turn, onJump }: { turn: Turn; onJump: () => void }) {
  const t = useT();
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const toggle = (id: string) => setOpen((o) => {
    const node = turn.nodes[id];
    const current = o[id] ?? (node?.status === "running" && node.kind !== "model");
    return { ...o, [id]: !current };
  });
  const kinds = countKinds(turn);
  return (
    <div>
      <div className="turn-h">
        <span className="q">“{turn.message}”</span>
        <span className="stat mono">
          {kinds.model || 0} model · {kinds.tool || 0} tool
          {turn.latency != null && ` · ${fmtMs(turn.latency)}`}
          {turn.cost != null && ` · ¥${turn.cost.toFixed(4)}`}
        </span>
      </div>
      {turn.roots.map((id) => (
        <div className="mod" key={id}>
          <Card id={id} turn={turn} open={open} toggle={toggle} onJump={onJump} />
        </div>
      ))}
      {!turn.running && (turn.response || turn.error) && (
        <div className="final-reply">
          <h5>{turn.error ? t("error") : t("final")}</h5>
          {turn.error || turn.response}
        </div>
      )}
    </div>
  );
}

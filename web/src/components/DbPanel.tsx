import { useEffect, useMemo, useState, type ReactNode } from "react";
import type { Change, Snapshot } from "../model";
import { useT } from "../i18n";
import { api } from "../api";
import { Code, pretty } from "./Json";

type Fields = Record<string, { before: unknown; after: unknown }>;

function short(value: unknown, n = 80): string {
  if (value == null) return "∅";
  const text = typeof value === "string" ? value : JSON.stringify(value);
  return text.length > n ? text.slice(0, n) + "…" : text;
}

function money(v: unknown) {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? `$${n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : String(v);
}

function address(a: any) {
  if (!a) return "";
  return [a.address1, a.address2, a.city, a.state, a.zip, a.country].filter(Boolean).join(", ");
}

function initials(name: string) {
  return name.split(/\s+/).map((w) => w[0]).join("").slice(0, 2).toUpperCase();
}

// status 最能说明变化，排在最前。
function byImportance(a: { path: string }, b: { path: string }) {
  return Number(b.path === "status") - Number(a.path === "status") || a.path.localeCompare(b.path);
}

function fieldMap(change?: Change): Fields {
  const out: Fields = {};
  for (const f of [...(change?.fields || [])].sort(byImportance)) out[f.path] = { before: f.before, after: f.after };
  return out;
}

/** 某个顶层字段（含其子路径）是否在本轮变化。 */
function touched(fields: Fields, prefix: string) {
  return Object.keys(fields).some((p) => p === prefix || p.startsWith(prefix + ".") || p.startsWith(prefix + "["));
}

function Row({ label, fields, path, children }: { label: string; fields: Fields; path: string; children: ReactNode }) {
  const hit = touched(fields, path);
  const exact = fields[path];
  return (
    <>
      <dt>{label}</dt>
      <dd className={hit ? "changed" : ""}>
        {exact && exact.before != null && <span className="was">{short(exact.before, 40)}</span>}
        <span className={hit ? "now" : ""}>{children}</span>
      </dd>
    </>
  );
}

function Diffs({ fields }: { fields: Fields }) {
  const entries = Object.entries(fields);
  if (!entries.length) return null;
  return (
    <div className="diffs">
      {entries.map(([path, f]) => (
        <div className="diff" key={path}>
          <span className="path">{path}</span>
          <span className="was">{short(f.before, 50)}</span>
          <span className="now">{short(f.after, 70)}</span>
        </div>
      ))}
    </div>
  );
}

function Entity({ kind, id, sub, pill, pillClass, change, flash, defaultOpen, children, raw, diffs = true }: {
  kind: string; id: string; sub?: string; pill?: string; pillClass?: string; change?: Change; flash?: boolean;
  defaultOpen?: boolean; children?: ReactNode; raw: unknown; diffs?: boolean;
}) {
  const t = useT();
  const [open, setOpen] = useState(!!defaultOpen);
  const [showRaw, setShowRaw] = useState(false);
  useEffect(() => { if (change) setOpen(true); }, [change]);
  const fields = fieldMap(change);
  return (
    <div className={"entity" + (open ? " open" : "") + (change ? " touched" : "") + (flash ? " flash" : "")}>
      <div className="entity-h" onClick={() => setOpen(!open)}>
        <span className="kind">{kind}</span>
        <span className="id mono">{id}</span>
        {sub && <span className="sub">{sub}</span>}
        {change?.op === "added" && <span className="chip write">+ new</span>}
        {pill && <span className={"pill " + (pillClass || pill)}>{pill}</span>}
      </div>
      {open && (
        <>
          {children}
          {diffs && <Diffs fields={fields} />}
          <button className="rawbtn" onClick={() => setShowRaw(!showRaw)}>{showRaw ? "− " : "+ "}{t("raw")}</button>
          {showRaw && <div style={{ padding: "0 12px 12px" }}><Code value={raw} /></div>}
        </>
      )}
    </div>
  );
}

function OrderCard({ order, change, flash }: { order: any; change?: Change; flash?: boolean }) {
  const fields = fieldMap(change);
  const items: any[] = order.items || [];
  const total = items.reduce((s, i) => s + (Number(i.price) || 0), 0);
  const payments: any[] = order.payment_history || [];
  return (
    <Entity kind="Order" id={order.order_id} sub={`${items.length} item${items.length === 1 ? "" : "s"} · ${money(total)}`}
      pill={order.status} change={change} flash={flash} defaultOpen={!!change} raw={order} diffs={false}>
      <dl className="kv">
        <Row label="status" fields={fields} path="status">{order.status}</Row>
        {order.cancel_reason != null && <Row label="cancel_reason" fields={fields} path="cancel_reason">{order.cancel_reason}</Row>}
        <Row label="address" fields={fields} path="address">{address(order.address)}</Row>
        <Row label="payment_history" fields={fields} path="payment_history">
          {payments.map((p) => `${p.transaction_type} ${money(p.amount)} · ${p.payment_method_id}`).join("  |  ")}
        </Row>
        {(order.fulfillments || []).length > 0 && (
          <Row label="fulfillments" fields={fields} path="fulfillments">{order.fulfillments.map((f: any) => (f.tracking_id || []).join(",")).join(" | ")}</Row>
        )}
        {order.exchange_items && <Row label="exchange_items" fields={fields} path="exchange_items">{short(order.exchange_items)}</Row>}
        {order.return_items && <Row label="return_items" fields={fields} path="return_items">{short(order.return_items)}</Row>}
      </dl>
      <div className="items">
        {items.map((item, i) => (
          <div className="item" key={item.item_id + i}>
            <div className="thumb">{initials(item.name || "?")}</div>
            <div>{item.name}<div className="opt">{Object.values(item.options || {}).join(" · ")}</div></div>
            <span className="price mono">{money(item.price)}</span>
          </div>
        ))}
      </div>
    </Entity>
  );
}

function UserCard({ user, change, flash, bound }: { user: any; change?: Change; flash?: boolean; bound: boolean }) {
  const t = useT();
  const fields = fieldMap(change);
  const name = `${user.name?.first_name ?? ""} ${user.name?.last_name ?? ""}`.trim();
  return (
    <Entity kind="User" id={user.user_id} sub={name} pill={bound ? t("bound") : undefined} pillClass="bound"
      change={change} flash={flash} defaultOpen raw={user} diffs={false}>
      <dl className="kv">
        <Row label="email" fields={fields} path="email">{user.email}</Row>
        <Row label="address" fields={fields} path="address">{address(user.address)}</Row>
        <Row label="payment_methods" fields={fields} path="payment_methods">
          {Object.values(user.payment_methods || {}).map((p: any) =>
            p.source === "credit_card" ? `${p.brand} ••${p.last_four}` : p.source === "gift_card" ? `gift card ${money(p.balance)}` : p.source).join(" · ")}
        </Row>
        <Row label="orders" fields={fields} path="orders">{(user.orders || []).join("  ")}</Row>
      </dl>
    </Entity>
  );
}

const STATE_ORDER = ["pending_actions", "confirmation_decisions", "request_plan", "retail_events", "task_facts", "context_state", "sessions"];

function stateSummary(table: string, row: any): { sub: string; pill?: string } {
  switch (table) {
    case "pending_actions": return { sub: `${row.kind} · ${row.order_id || row.target_key}`, pill: row.status };
    case "confirmation_decisions": return { sub: `${row.source} · ${short(row.reason, 60)}`, pill: row.approved ? "approved" : "rejected" };
    case "request_plan": return { sub: (row.payload?.goals || []).map((g: any) => `${g.kind || g.type} ${g.order_id || ""} (${g.status})`).join(" · ") || "no goals" };
    case "retail_events": return { sub: `${row.event_type}${row.order_id ? " · " + row.order_id : ""}` };
    case "sessions": return { sub: `turns ${row.turn_count} · user ${row.retail_user_id || "—"}` };
    default: return { sub: short(row.payload, 70) };
  }
}

export function DbPanel({ snapshot, changes, turns, viewTurn, onViewTurn, liveTurn, sessionId, flashSeq }: {
  snapshot: Snapshot | null; changes: Change[]; turns: number[]; viewTurn: number; onViewTurn: (n: number) => void;
  liveTurn: number; sessionId?: string; flashSeq: number;
}) {
  const t = useT();
  const [tab, setTab] = useState<"business" | "state">("business");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<any>(null);
  const changeOf = (group: string, table: string, key: string) =>
    changes.find((c) => c.group === group && c.table === table && c.key === key);
  const stateChanges = changes.filter((c) => c.group === "state").length;
  const flashed = useMemo(() => new Set(changes.map((c) => `${c.table}:${c.key}`)), [flashSeq]); // eslint-disable-line
  const biz = snapshot?.business || {};
  const users = Object.values(biz.users || {});
  const orders = Object.values(biz.orders || {}).sort((a: any, b: any) =>
    Number(!!changeOf("business", "orders", b.order_id)) - Number(!!changeOf("business", "orders", a.order_id)));
  const products = Object.values(biz.products || {});

  async function search() {
    if (!sessionId || !query.trim()) return setResults(null);
    try { setResults(await api.search(sessionId, query.trim())); } catch (e) { setResults({ error: String(e) }); }
  }

  return (
    <section className="col">
      <div className="col-h">
        <h2>{t("database")}</h2>
        <div className="tabs">
          <button className={tab === "business" ? "on" : ""} onClick={() => setTab("business")}>{t("business")}</button>
          <button className={tab === "state" ? "on" : ""} onClick={() => setTab("state")}>
            {t("sessionState")}{stateChanges > 0 && tab !== "state" && <span className="badge">{stateChanges}</span>}
          </button>
        </div>
      </div>
      <div className="turnbar">
        <span>{t("snapshot")}</span>
        {turns.map((n) => (
          <button key={n} className={"t" + (n === viewTurn ? " on" : "")} onClick={() => onViewTurn(n)}>
            T{n}{n === liveTurn ? ` · ${t("liveTag")}` : ""}
          </button>
        ))}
        {sessionId && (
          <input className="search" placeholder={t("searchAll")} value={query} onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && search()} />
        )}
      </div>
      <div className="col-b">
        {results ? (
          <>
            <div className="group-title">{t("results")} · “{query}” <button className="btn" style={{ marginLeft: "auto", padding: "2px 8px" }} onClick={() => setResults(null)}>{t("clear")}</button></div>
            {results.error && <div className="empty">{results.error}</div>}
            {(results.users || []).map((u: any) => <UserCard key={u.user_id} user={u} bound={false} />)}
            {(results.orders || []).map((o: any) => <OrderCard key={o.order_id} order={o} />)}
            {(results.products || []).map((p: any) => <Entity key={p.product_id} kind="Product" id={p.product_id} sub={`${p.name} · ${p.variants} variants`} raw={p} />)}
          </>
        ) : tab === "business" ? (
          <>
            <ChangeLog changes={changes.filter((c) => c.group === "business")} turn={viewTurn} />
            {!users.length && !orders.length && <div className="empty">{t("noEntities")}</div>}
            {users.length > 0 && <div className="group-title">{t("users")}</div>}
            {users.map((u: any) => (
              <UserCard key={u.user_id} user={u} bound={snapshot?.bound_user === u.user_id}
                change={changeOf("business", "users", u.user_id)} flash={flashed.has(`users:${u.user_id}`)} />
            ))}
            {orders.length > 0 && <div className="group-title">{t("orders")} · {orders.length}</div>}
            {orders.map((o: any) => (
              <OrderCard key={o.order_id} order={o} change={changeOf("business", "orders", o.order_id)}
                flash={flashed.has(`orders:${o.order_id}`)} />
            ))}
            {products.length > 0 && <div className="group-title">{t("products")} · {products.length}</div>}
            {products.map((p: any) => (
              <Entity key={p.product_id} kind="Product" id={p.product_id}
                sub={`${p.name} · ${Object.keys(p.variants || {}).length} variants`} raw={p} />
            ))}
          </>
        ) : (
          <>
            <ChangeLog changes={changes.filter((c) => c.group === "state")} turn={viewTurn} />
            {STATE_ORDER.map((table) => {
              const rows = Object.entries((snapshot?.state || {})[table] || {});
              if (!rows.length) return null;
              return (
                <div key={table}>
                  <div className="group-title">{table} · {rows.length}</div>
                  {rows.map(([key, row]) => {
                    const { sub, pill } = stateSummary(table, row);
                    return (
                      <Entity key={key} kind={table.replace(/_/g, " ")} id={key} sub={sub} pill={pill}
                        change={changeOf("state", table, key)} flash={flashed.has(`${table}:${key}`)} raw={row}>
                        <div style={{ padding: "10px 12px 0" }}><Code value={row} fold /></div>
                      </Entity>
                    );
                  })}
                </div>
              );
            })}
          </>
        )}
      </div>
    </section>
  );
}

function ChangeLog({ changes, turn }: { changes: Change[]; turn: number }) {
  const t = useT();
  if (turn === 0) return null;
  return (
    <div className="change-log">
      <h4>{t("changesThisTurn")} · T{turn}</h4>
      {!changes.length && <div className="cl-row"><span className="what">{t("noChanges")}</span></div>}
      {changes.map((c) => (
        <div className="cl-row" key={`${c.group}:${c.table}:${c.key}`}>
          <span className={"chip " + (c.group === "business" ? "write" : "wait")}>{c.table} · {c.key}</span>
          <span className="what">
            {c.op !== "changed" ? c.op : [...(c.fields || [])].sort(byImportance).slice(0, 3).map((f) => (
              <span key={f.path} style={{ marginRight: 10 }}>
                {f.path} <span className="was">{short(f.before, 24)}</span><span className="now">{short(f.after, 30)}</span>
              </span>
            ))}
            {(c.fields || []).length > 3 && <span>+{(c.fields || []).length - 3}</span>}
          </span>
        </div>
      ))}
    </div>
  );
}

export { pretty };

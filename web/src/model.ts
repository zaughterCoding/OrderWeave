// 事件流 → 界面状态。实时与回放共用同一个 reducer。

export type Json = any;

export interface LiveEvent {
  type: "session" | "turn_start" | "start" | "end" | "point" | "db" | "turn_end" | "turn_error";
  seq?: number;
  turn?: number;
  id?: string;
  parent?: string | null;
  kind?: string;
  name?: string;
  ts?: number;
  ms?: number;
  status?: string;
  data?: Json;
  [key: string]: Json;
}

export interface Change {
  group: "business" | "state";
  table: string;
  key: string;
  op: "added" | "removed" | "changed";
  fields?: { path: string; before: Json; after: Json }[];
}

export interface Snapshot {
  bound_user: string | null;
  business: Record<string, Record<string, Json>>;
  state: Record<string, Record<string, Json>>;
}

export interface TraceNode {
  id: string;
  parent: string | null;
  kind: string;
  name: string;
  point: boolean;
  status: "running" | "ok" | "error" | "waiting";
  start: Json;
  end: Json;
  ms?: number;
  ts: number;
  children: string[];
  dbChanges: number;
}

export interface Turn {
  n: number;
  message: string;
  nodes: Record<string, TraceNode>;
  roots: string[];
  path: string[];
  running: boolean;
  response?: string;
  error?: string;
  cost?: number;
  latency?: number;
  changes: Change[];
  startedAt: number;
}

export interface GraphSpec {
  mode: string;
  nodes: string[];
  edges: { source: string; target: string; label: string }[];
}

export interface SessionInfo {
  id: string;
  mode: string;
  language: string;
  greeting: string;
  graph: GraphSpec;
}

export interface State {
  session: SessionInfo | null;
  turns: Turn[];
  snapshot: Snapshot | null;
  snapshots: Record<number, Snapshot>;
  cost: number;
}

export const emptyState: State = { session: null, turns: [], snapshot: null, snapshots: {}, cost: 0 };

export type Action = { type: "event"; event: LiveEvent } | { type: "reset" } | { type: "session"; session: SessionInfo; snapshot: Snapshot };

function mergeChanges(list: Change[], incoming: Change[]): Change[] {
  // 同一实体在一轮内多次变化时合并：保留最早的 before、最新的 after。
  const out = list.map((c) => ({ ...c, fields: c.fields ? [...c.fields] : undefined }));
  for (const change of incoming) {
    const hit = out.find((c) => c.group === change.group && c.table === change.table && c.key === change.key);
    if (!hit) {
      out.push({ ...change, fields: change.fields ? [...change.fields] : undefined });
      continue;
    }
    if (hit.op === "added") continue;
    if (change.op !== "changed") {
      hit.op = change.op;
      continue;
    }
    hit.fields = hit.fields || [];
    for (const field of change.fields || []) {
      const old = hit.fields.find((f) => f.path === field.path);
      if (old) old.after = field.after;
      else hit.fields.push({ ...field });
    }
  }
  return out;
}

function updateTurn(state: State, n: number, fn: (turn: Turn) => Turn): State {
  return { ...state, turns: state.turns.map((t) => (t.n === n ? fn(t) : t)) };
}

export function reducer(state: State, action: Action): State {
  if (action.type === "reset") return emptyState;
  if (action.type === "session") {
    return { ...emptyState, session: action.session, snapshot: action.snapshot, snapshots: { 0: action.snapshot } };
  }
  const ev = action.event;
  switch (ev.type) {
    case "session":
      return {
        ...emptyState,
        session: { id: ev.id as string, mode: ev.mode, language: ev.language, greeting: ev.greeting, graph: ev.graph },
        snapshot: ev.snapshot,
        snapshots: { 0: ev.snapshot },
      };
    case "turn_start": {
      const turn: Turn = {
        n: ev.turn!, message: ev.message, nodes: {}, roots: [], path: [], running: true, changes: [],
        startedAt: ev.ts || Date.now() / 1000,
      };
      return { ...state, turns: [...state.turns.filter((t) => t.n !== turn.n), turn] };
    }
    case "start":
    case "point":
      return updateTurn(state, ev.turn!, (turn) => {
        const node: TraceNode = {
          id: ev.id!, parent: ev.parent ?? null, kind: ev.kind!, name: ev.name!, point: ev.type === "point",
          status: ev.type === "point" ? "ok" : "running", start: ev.data || {}, end: {}, ts: ev.ts || 0,
          children: [], dbChanges: 0,
        };
        const nodes = { ...turn.nodes, [node.id]: node };
        let roots = turn.roots;
        if (node.parent && nodes[node.parent]) {
          nodes[node.parent] = { ...nodes[node.parent], children: [...nodes[node.parent].children, node.id] };
        } else {
          roots = [...roots, node.id];
        }
        const path = node.kind === "node" ? [...turn.path, node.name] : turn.path;
        return { ...turn, nodes, roots, path };
      });
    case "end":
      return updateTurn(state, ev.turn!, (turn) => {
        const node = turn.nodes[ev.id!];
        if (!node) return turn;
        const status = (ev.status === "error" || ev.status === "waiting" ? ev.status : "ok") as TraceNode["status"];
        return { ...turn, nodes: { ...turn.nodes, [node.id]: { ...node, status, end: ev.data || {}, ms: ev.ms } } };
      });
    case "db": {
      const next = updateTurn(state, ev.turn!, (turn) => {
        const nodes = { ...turn.nodes };
        const toolId = ev.trigger?.event as string | undefined;
        if (toolId && nodes[toolId]) nodes[toolId] = { ...nodes[toolId], dbChanges: nodes[toolId].dbChanges + ev.changes.length };
        return { ...turn, nodes, changes: mergeChanges(turn.changes, ev.changes) };
      });
      return { ...next, snapshot: ev.snapshot, snapshots: { ...next.snapshots, [ev.turn!]: ev.snapshot } };
    }
    case "turn_end":
    case "turn_error": {
      const next = updateTurn(state, ev.turn!, (turn) => ({
        ...turn, running: false, response: ev.response, error: ev.error, cost: ev.cost_cny, latency: ev.latency_ms,
      }));
      const snaps = { ...next.snapshots };
      if (!snaps[ev.turn!] && next.snapshot) snaps[ev.turn!] = next.snapshot;
      return { ...next, snapshots: snaps, cost: ev.session_cost_cny ?? next.cost + (ev.cost_cny || 0) };
    }
  }
  return state;
}

export function countKinds(turn: Turn): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const node of Object.values(turn.nodes)) counts[node.kind] = (counts[node.kind] || 0) + 1;
  return counts;
}

/** 本轮走过的边：按节点顺序相邻成对，加上入口与出口。 */
export function walkedEdges(turn: Turn | undefined): Set<string> {
  const edges = new Set<string>();
  if (!turn || !turn.path.length) return edges;
  const first = turn.path[0];
  if (first !== "wait_user") edges.add(`__start__>${first}`);
  for (let i = 1; i < turn.path.length; i++) edges.add(`${turn.path[i - 1]}>${turn.path[i]}`);
  const last = turn.path[turn.path.length - 1];
  if (!turn.running && last !== "wait_user") edges.add(`${last}>__end__`);
  return edges;
}

export function nodeStatus(turn: Turn | undefined): Record<string, TraceNode["status"]> {
  const out: Record<string, TraceNode["status"]> = {};
  if (!turn) return out;
  for (const node of Object.values(turn.nodes)) {
    if (node.kind !== "node") continue;
    const prev = out[node.name];
    // 同名节点多次进入时，以进行中优先。
    if (prev !== "running") out[node.name] = node.status;
  }
  if (turn.path.length) out["__start__"] = "ok";
  if (!turn.running && turn.path[turn.path.length - 1] !== "wait_user") out["__end__"] = "ok";
  return out;
}

export function pendingActions(snapshot: Snapshot | null): Json[] {
  const rows = snapshot?.state?.pending_actions || {};
  return Object.values(rows).filter((row: Json) => row.status === "displayed");
}

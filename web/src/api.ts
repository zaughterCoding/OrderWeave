import type { GraphSpec, LiveEvent, SessionInfo, Snapshot } from "./model";

export interface Scenario {
  id: string;
  kind: string;
  title: { en: string; zh: string };
  persona: { name: string; zip: string; email: string };
  focus: string;
  messages: string[];
}

export interface ReplayItem {
  source: "builtin" | "recorded";
  name: string;
  turns: number;
  modified: number;
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch {
      /* 保留状态文本 */
    }
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  scenarios: () => fetch("/api/web/scenarios").then((r) => json<{ greeting: string; items: Scenario[] }>(r)),
  graph: (mode: string) => fetch(`/api/web/graph?mode=${mode}`).then((r) => json<GraphSpec>(r)),
  replays: () => fetch("/api/web/replays").then((r) => json<{ items: ReplayItem[] }>(r)),
  async replay(source: string, name: string): Promise<LiveEvent[]> {
    const res = await fetch(`/api/web/replays/${source}/${encodeURIComponent(name)}`);
    if (!res.ok) throw new Error(`${res.status} replay not found`);
    const text = await res.text();
    return text.split("\n").filter(Boolean).map((line) => JSON.parse(line));
  },
  createSession: (mode: string, language: string) =>
    fetch("/api/web/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode, language }),
    }).then((r) => json<SessionInfo & { snapshot: Snapshot }>(r)),
  closeSession: (id: string) => fetch(`/api/web/sessions/${id}`, { method: "DELETE" }),
  search: (id: string, q: string) =>
    fetch(`/api/web/sessions/${id}/db?q=${encodeURIComponent(q)}`).then((r) => json<any>(r)),

  /** POST 后读取 SSE 流，逐条回调事件。 */
  async streamTurn(id: string, message: string, onEvent: (event: LiveEvent) => void): Promise<void> {
    const res = await fetch(`/api/web/sessions/${id}/turns`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    if (!res.ok || !res.body) await json(res);
    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut: number;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const chunk = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        const line = chunk.split("\n").find((l) => l.startsWith("data: "));
        if (line) onEvent(JSON.parse(line.slice(6)));
      }
    }
  },
};

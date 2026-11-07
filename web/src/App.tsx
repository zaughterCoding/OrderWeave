import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { api, type ReplayItem, type Scenario } from "./api";
import { LangContext, type Lang, useT } from "./i18n";
import { countKinds, emptyState, pendingActions, reducer, type LiveEvent } from "./model";
import { ChatPanel } from "./components/ChatPanel";
import { DbPanel } from "./components/DbPanel";
import { GraphView } from "./components/GraphView";
import { TraceTree } from "./components/TraceTree";

type Source = "live" | "replay";

function readPref<T extends string>(key: string, fallback: T): T {
  try { return (localStorage.getItem(key) as T) || fallback; } catch { return fallback; }
}
function writePref(key: string, value: string) {
  try { localStorage.setItem(key, value); } catch { /* 隐私模式下忽略 */ }
}

/** 回放：按录制时间间隔重放事件，间隔压缩到可观看的范围。 */
function useReplay(dispatch: (e: LiveEvent) => void, reset: () => void) {
  const [events, setEvents] = useState<LiveEvent[]>([]);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  useEffect(() => {
    if (!playing || index >= events.length) { if (index >= events.length) setPlaying(false); return; }
    const ev = events[index];
    const prev = events[index - 1];
    let delay = prev?.ts && ev.ts ? Math.min(1400, Math.max(40, (ev.ts - prev.ts) * 1000)) : 0;
    if (ev.type === "turn_start") delay = Math.max(delay, 1100);
    if (prev?.type === "turn_end") delay = Math.max(delay, 1600);
    const timer = setTimeout(() => { dispatch(ev); setIndex((i) => i + 1); }, delay / speed);
    return () => clearTimeout(timer);
  }, [playing, index, events, speed, dispatch]);
  const load = useCallback((list: LiveEvent[]) => {
    reset();
    setEvents(list);
    setIndex(0);
    setPlaying(true);
  }, [reset]);
  const restart = () => { reset(); setIndex(0); setPlaying(true); };
  return { events, index, playing, setPlaying, speed, setSpeed, load, restart };
}

export default function App() {
  const [lang, setLang] = useState<Lang>(() => readPref("ow.lang", "en"));
  return (
    <LangContext.Provider value={lang}>
      <Console lang={lang} setLang={(l) => { setLang(l); writePref("ow.lang", l); }} />
    </LangContext.Provider>
  );
}

function Console({ lang, setLang }: { lang: Lang; setLang: (l: Lang) => void }) {
  const t = useT();
  const [state, rawDispatch] = useReducer(reducer, emptyState);
  const dispatch = useCallback((event: LiveEvent) => rawDispatch({ type: "event", event }), []);
  const reset = useCallback(() => rawDispatch({ type: "reset" }), []);
  const [source, setSource] = useState<Source>(() => readPref("ow.source", "replay"));
  const [agentMode, setAgentMode] = useState<string>(() => readPref<string>("ow.mode", "multi"));
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [greeting, setGreeting] = useState("Hi! How can I help you today?");
  const [scenario, setScenario] = useState<Scenario | null>(null);
  const [replays, setReplays] = useState<ReplayItem[]>([]);
  const [replayKey, setReplayKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [starting, setStarting] = useState(false);
  const [toast, setToast] = useState("");
  const [pinnedTurn, setPinnedTurn] = useState<number | null>(null);
  const [pinnedSnap, setPinnedSnap] = useState<number | null>(null);
  const [flashSeq, setFlashSeq] = useState(0);
  const sessionRef = useRef<string | null>(null);
  const replay = useReplay(dispatch, reset);

  useEffect(() => {
    api.scenarios().then((r) => { setScenarios(r.items); setGreeting(r.greeting); }).catch((e) => setToast(String(e)));
    api.replays().then((r) => setReplays(r.items)).catch(() => undefined);
  }, []);

  useEffect(() => { setFlashSeq((n) => n + 1); }, [state.snapshot]);
  useEffect(() => { if (toast) { const id = setTimeout(() => setToast(""), 6000); return () => clearTimeout(id); } }, [toast]);

  // 切到回放时自动载入第一份录制。
  useEffect(() => {
    if (source !== "replay" || !replays.length || replayKey) return;
    const first = replays[0];
    setReplayKey(`${first.source}/${first.name}`);
  }, [source, replays, replayKey]);
  useEffect(() => {
    if (source !== "replay" || !replayKey) return;
    const [src, name] = replayKey.split("/");
    api.replay(src, name).then(replay.load).catch((e) => setToast(String(e)));
    setPinnedTurn(null); setPinnedSnap(null);
  }, [source, replayKey]); // eslint-disable-line

  async function closeLive() {
    const id = sessionRef.current;
    sessionRef.current = null;
    if (id) api.closeSession(id).catch(() => undefined);
  }

  async function ensureSession(): Promise<string> {
    if (sessionRef.current) return sessionRef.current;
    setStarting(true);
    try {
      const s = await api.createSession(agentMode, "en");
      sessionRef.current = s.id;
      rawDispatch({ type: "session", session: s, snapshot: s.snapshot });
      return s.id;
    } finally {
      setStarting(false);
    }
  }

  async function send(text: string) {
    setBusy(true);
    setPinnedTurn(null); setPinnedSnap(null);
    try {
      const id = await ensureSession();
      await api.streamTurn(id, text, dispatch);
    } catch (e) {
      setToast(String(e));
    } finally {
      setBusy(false);
    }
  }

  function switchSource(next: Source) {
    if (next === source) return;
    replay.setPlaying(false);
    reset(); setScenario(null); setPinnedTurn(null); setPinnedSnap(null);
    if (next === "replay") { closeLive(); setReplayKey(""); api.replays().then((r) => setReplays(r.items)).catch(() => undefined); }
    setSource(next); writePref("ow.source", next);
  }

  function newSession(mode: string = agentMode) {
    closeLive(); reset(); setScenario(null); setPinnedTurn(null); setPinnedSnap(null);
    setAgentMode(mode); writePref("ow.mode", mode);
  }

  const turns = state.turns;
  const last = turns[turns.length - 1];
  const traceTurn = turns.find((x) => x.n === pinnedTurn) || last;
  const liveTurn = last?.n ?? 0;
  const snapTurns = [0, ...turns.filter((x) => !x.running || state.snapshots[x.n]).map((x) => x.n)];
  const viewSnap = pinnedSnap ?? liveTurn;
  const viewSnapshot = viewSnap === liveTurn ? state.snapshot : state.snapshots[viewSnap] || state.snapshot;
  const viewChanges = turns.find((x) => x.n === viewSnap)?.changes || [];
  const modelCalls = useMemo(() => turns.reduce((s, x) => s + (countKinds(x).model || 0), 0), [turns]);
  const graph = state.session?.graph || { mode: agentMode, nodes: [], edges: [] };
  const running = !!last?.running;
  const mode = state.session?.mode || agentMode;

  const replayBar = (
    <div className="replay-bar">
      <button className="btn" onClick={() => replay.setPlaying(!replay.playing)}>{replay.playing ? `❚❚ ${t("pause")}` : `▶ ${t("play")}`}</button>
      <button className="btn" onClick={replay.restart}>↺</button>
      <div className="progress"><i style={{ width: `${replay.events.length ? (replay.index / replay.events.length) * 100 : 0}%` }} /></div>
      <div className="seg">
        {[1, 2, 4].map((s) => <button key={s} className={replay.speed === s ? "on" : ""} onClick={() => replay.setSpeed(s)}>{s}×</button>)}
      </div>
    </div>
  );

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand"><img src="/logo.svg" alt="" /><b>OrderWeave</b><span>{t("console")}</span></div>
        <div className="seg">
          <button className={source === "live" ? "on" : ""} onClick={() => switchSource("live")}>{t("live")}</button>
          <button className={source === "replay" ? "on" : ""} onClick={() => switchSource("replay")}>{t("replay")}</button>
        </div>
        {source === "live" ? (
          <div className="seg">
            <button className={mode === "single" ? "on" : ""} disabled={busy} onClick={() => newSession("single")}>{t("single")}</button>
            <button className={mode === "multi" ? "on" : ""} disabled={busy} onClick={() => newSession("multi")}>{t("multi")}</button>
          </div>
        ) : (
          <select className="select" value={replayKey} onChange={(e) => setReplayKey(e.target.value)} title={t("replayHint")}>
            {!replays.length && <option value="">{t("replayPick")}</option>}
            {replays.map((r) => (
              <option key={`${r.source}/${r.name}`} value={`${r.source}/${r.name}`}>
                {r.source === "builtin" ? "★ " : ""}{r.name} · {r.turns}T
              </option>
            ))}
          </select>
        )}
        <div className="spacer" />
        <div className={"status-dot" + (running ? "" : " idle")}><i />{running ? `${t("turn") === "第" ? `第 ${last.n} 轮` : `Turn ${last.n}`} · ${t("running")}` : mode === "single" ? t("single") : t("multi")}</div>
        <div className="meter">
          <span>{t("turns")} <b>{turns.length}</b></span>
          <span>{t("modelCalls")} <b>{modelCalls}</b></span>
          <span>{t("cost")} <b className="mono">¥{state.cost.toFixed(3)}</b></span>
        </div>
        <div className="seg">
          <button className={lang === "en" ? "on" : ""} onClick={() => setLang("en")}>EN</button>
          <button className={lang === "zh" ? "on" : ""} onClick={() => setLang("zh")}>中</button>
        </div>
        {source === "live" && <button className="btn" disabled={busy} onClick={() => newSession()}>↺ {t("newSession")}</button>}
      </header>

      <main className="columns">
        <ChatPanel greeting={state.session?.greeting || greeting} turns={turns} live={source === "live"} busy={busy || running}
          starting={starting} scenarios={scenarios} scenario={scenario} onScenario={setScenario} onSend={send}
          selected={traceTurn?.n ?? 0} onSelect={(n) => { setPinnedTurn(n); setPinnedSnap(n); }}
          pending={pendingActions(state.snapshot)} replayBar={replayBar} />

        <DbPanel snapshot={viewSnapshot} changes={viewChanges} turns={snapTurns} viewTurn={viewSnap}
          onViewTurn={(n) => setPinnedSnap(n === liveTurn ? null : n)} liveTurn={liveTurn}
          sessionId={source === "live" ? state.session?.id : undefined} flashSeq={flashSeq} />

        <section className="col">
          <div className="col-h"><h2>{t("trace")}</h2><small>LangGraph · {mode === "single" ? t("single") : t("multi")}</small></div>
          <GraphView graph={graph} turn={traceTurn} />
          {turns.length > 0 && (
            <div className="turn-tabs">
              {turns.map((x) => (
                <button key={x.n} className={x.n === traceTurn?.n ? "on" : ""} onClick={() => setPinnedTurn(x.n === liveTurn ? null : x.n)}>
                  {x.running && <span className="spin" />}{lang === "zh" ? `第 ${x.n} 轮` : `Turn ${x.n}`}
                </button>
              ))}
            </div>
          )}
          <div className="col-b">
            {traceTurn ? <TraceTree key={traceTurn.n} turn={traceTurn} onJump={() => setPinnedSnap(traceTurn.n === liveTurn ? null : traceTurn.n)} />
              : <div className="empty">{t("noTurn")}</div>}
          </div>
        </section>
      </main>
      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}

import { useEffect, useRef, useState } from "react";
import type { Turn } from "../model";
import { countKinds } from "../model";
import type { Scenario } from "../api";
import { useT, LangContext } from "../i18n";
import { useContext } from "react";

/** 打字机效果：新到达的回复逐字显示；回放或历史消息直接完整显示。 */
function Reveal({ text, animate }: { text: string; animate: boolean }) {
  const [shown, setShown] = useState(animate ? 0 : text.length);
  useEffect(() => {
    if (!animate) { setShown(text.length); return; }
    setShown(0);
    const step = Math.max(2, Math.round(text.length / 120));
    const timer = setInterval(() => setShown((n) => {
      if (n + step >= text.length) { clearInterval(timer); return text.length; }
      return n + step;
    }), 16);
    return () => clearInterval(timer);
  }, [text, animate]);
  return <span className={shown < text.length ? "caret-blink" : ""}>{text.slice(0, shown)}</span>;
}

export function ChatPanel(props: {
  greeting: string; turns: Turn[]; live: boolean; busy: boolean; starting: boolean;
  scenarios: Scenario[]; scenario: Scenario | null; onScenario: (s: Scenario) => void;
  onSend: (text: string) => void; selected: number; onSelect: (n: number) => void;
  pending: any[]; replayBar?: React.ReactNode; pendingMessage?: string;
}) {
  const t = useT();
  const lang = useContext(LangContext);
  const [draft, setDraft] = useState("");
  const body = useRef<HTMLDivElement>(null);
  const lastTurn = props.turns[props.turns.length - 1];
  const [animated, setAnimated] = useState<Set<number>>(new Set());

  useEffect(() => {
    body.current?.scrollTo({ top: body.current.scrollHeight, behavior: "smooth" });
  }, [props.turns.length, lastTurn?.response, lastTurn?.running]);

  // 只对刚完成的那一轮做打字效果。
  useEffect(() => {
    if (lastTurn && !lastTurn.running && lastTurn.response && !animated.has(lastTurn.n)) {
      setAnimated((s) => new Set(s).add(lastTurn.n));
    }
  }, [lastTurn?.running]); // eslint-disable-line

  const sent = new Set(props.turns.map((x) => x.message));
  const suggestions = props.live && props.scenario ? props.scenario.messages.filter((m) => !sent.has(m)) : [];
  const send = (text: string) => {
    if (!text.trim() || props.busy || props.starting) return;
    props.onSend(text.trim());
    setDraft("");
  };

  return (
    <section className="col">
      <div className="col-h">
        <h2>{t("conversation")}</h2>
        {props.scenario && <small>{props.scenario.title[lang]}</small>}
      </div>
      <div className="col-b" ref={body}>
        {props.scenario && (
          <div className="persona">
            <div className="avatar">{props.scenario.persona.name.split(" ").map((w) => w[0]).join("")}</div>
            <div><b>{props.scenario.persona.name}</b><p>zip {props.scenario.persona.zip} · {props.scenario.persona.email}</p></div>
          </div>
        )}
        <div className="msg bot">{props.greeting}</div>
        {props.live && !props.turns.length && !props.scenario && (
          <>
            <div className="empty" style={{ padding: "8px 0 10px" }}>{t("startHint")}</div>
            <div className="scenarios">
              {props.scenarios.map((s) => (
                <button key={s.id} className="scenario" onClick={() => props.onScenario(s)}>
                  <b>{s.title[lang]}</b>
                  <span>{s.persona.name} · {s.focus}</span>
                </button>
              ))}
            </div>
          </>
        )}
        {props.turns.map((turn) => {
          const kinds = countKinds(turn);
          const isLast = turn.n === lastTurn?.n;
          const waiting = isLast && props.pending.length > 0 && !turn.running;
          return (
            <div key={turn.n}>
              <div className="msg user">{turn.message}</div>
              {turn.running ? (
                <div className="msg bot typing"><i /><i /><i /></div>
              ) : (
                <div className={"msg bot" + (turn.error ? " error" : "") + (props.selected === turn.n ? " selected" : "")}
                  onClick={() => props.onSelect(turn.n)} style={{ cursor: "pointer" }}>
                  {turn.error ? `${t("error")}: ${turn.error}` : <Reveal text={turn.response || ""} animate={isLast && animated.has(turn.n) && props.live} />}
                  {waiting && (
                    <div className="confirm-card">
                      <b>{t("awaiting")}</b>
                      {props.pending.map((p) => <div key={p.op_id}>{p.kind} · {p.order_id || p.target_key} · {p.op_id}</div>)}
                    </div>
                  )}
                  <div className="meta">
                    <span className="chip plain">{lang === "zh" ? `第 ${turn.n} 轮` : `T${turn.n}`}</span>
                    {kinds.model ? <span className="chip model">{kinds.model} model</span> : null}
                    {kinds.tool ? <span className="chip tool">{kinds.tool} tool</span> : null}
                    {turn.changes.some((c) => c.group === "business") && <span className="chip write">DB write</span>}
                    {waiting && <span className="chip wait">{t("awaiting")}</span>}
                    {turn.cost != null && <span className="chip plain mono">¥{turn.cost.toFixed(4)}</span>}
                  </div>
                </div>
              )}
            </div>
          );
        })}
        {props.pendingMessage && (
          <>
            <div className="msg user">{props.pendingMessage}</div>
            <div className="msg bot typing"><i /><i /><i /></div>
          </>
        )}
        {props.starting && <div className="empty">{t("starting")}</div>}
      </div>
      {props.live ? (
        <>
          {suggestions.length > 0 && (
            <div className="suggest">
              {suggestions.map((m) => <button key={m} onClick={() => send(m)} disabled={props.busy}>{m}</button>)}
            </div>
          )}
          <div className="composer">
            <textarea placeholder={t("placeholder")} value={draft} onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(draft); } }} />
            <button className="send" disabled={!draft.trim() || props.busy || props.starting} onClick={() => send(draft)}>↑</button>
          </div>
        </>
      ) : props.replayBar}
    </section>
  );
}

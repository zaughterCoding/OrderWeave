import type { GraphSpec, Turn } from "../model";
import { nodeStatus, walkedEdges } from "../model";
import { useT } from "../i18n";

type Box = { x: number; y: number; w: number; label?: string; round?: boolean };

const H = 24;
// 手工排布：第一行入口与路由，第二行角色，第三行收尾与等待；单智能体时 run_agent 占角色行中间。
const MULTI: Record<string, Box> = {
  __start__: { x: 14, y: 30, w: 50, label: "input", round: true },
  prepare_state: { x: 84, y: 30, w: 100 },
  route_agent: { x: 204, y: 30, w: 92 },
  select_role: { x: 316, y: 30, w: 90 },
  cancel_return: { x: 428, y: 30, w: 100 },
  leave_return: { x: 548, y: 30, w: 96 },
  general: { x: 116, y: 96, w: 80, round: true },
  technical: { x: 210, y: 96, w: 86, round: true },
  billing: { x: 310, y: 96, w: 74, round: true },
  escalation: { x: 398, y: 96, w: 92, round: true },
  handoff_role: { x: 510, y: 96, w: 96 },
  run_agent: { x: 624, y: 96, w: 82 },
  recover_result: { x: 14, y: 162, w: 104 },
  resume_input: { x: 138, y: 162, w: 96 },
  wait_user: { x: 254, y: 162, w: 82 },
  compose_roles: { x: 356, y: 162, w: 108 },
  update_result: { x: 484, y: 162, w: 104 },
  __end__: { x: 608, y: 162, w: 50, label: "reply", round: true },
};
const SINGLE: Record<string, Box> = { ...MULTI, run_agent: { x: 300, y: 96, w: 92 } };

function anchor(a: Box, b: Box) {
  const ac = { x: a.x + a.w / 2, y: a.y + H / 2 };
  const bc = { x: b.x + b.w / 2, y: b.y + H / 2 };
  if (a.y === b.y) {
    // 同行：左右相连
    if (bc.x > ac.x) return `M${a.x + a.w} ${ac.y}H${b.x - 3}`;
    return `M${a.x} ${ac.y}H${b.x + b.w + 3}`;
  }
  const down = b.y > a.y;
  const sy = down ? a.y + H : a.y;
  const ey = down ? b.y - 3 : b.y + H + 3;
  const mid = (sy + ey) / 2;
  return `M${ac.x} ${sy}C${ac.x} ${mid} ${bc.x} ${mid} ${bc.x} ${ey}`;
}

export function GraphView({ graph, turn }: { graph: GraphSpec; turn?: Turn }) {
  const t = useT();
  const layout = graph.mode === "single" ? SINGLE : MULTI;
  const status = nodeStatus(turn);
  const walked = walkedEdges(turn);
  const running = turn?.running ? turn.path[turn.path.length - 1] : undefined;
  const nodes = [...graph.nodes, "__start__", "__end__"].filter((n, i, all) => layout[n] && all.indexOf(n) === i);
  // 自环（handoff_role → 角色）与同名反向边只画一次
  const edges = graph.edges.filter((e) => layout[e.source] && layout[e.target] && !(e.source === "handoff_role" && e.target !== "compose_roles" && e.target !== "update_result"));
  return (
    <div className="graph">
      <div className="legend">
        <span><i style={{ background: "var(--ink)" }} />{t("legendNode")}</span>
        <span><i style={{ background: "var(--model)" }} />{t("legendModel")}</span>
        <span><i style={{ background: "var(--tool)" }} />{t("legendTool")}</span>
        <span><i style={{ background: "var(--guard)" }} />{t("legendGuard")}</span>
        <span><i style={{ background: "var(--wait)" }} />{t("legendConfirm")}</span>
      </div>
      <svg viewBox="0 0 720 200">
        <defs>
          <marker id="ar" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0 0L8 4L0 8z" fill="#c9c5b8" />
          </marker>
          <marker id="arg" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0 0L8 4L0 8z" fill="#059669" />
          </marker>
        </defs>
        {graph.mode !== "single" && <text className="lane" x="14" y="112">ROLES</text>}
        {edges.map((e) => {
          const key = `${e.source}>${e.target}`;
          const isWalked = walked.has(key);
          const hot = isWalked && running === e.target;
          return (
            <path key={key} className={"ge" + (hot ? " hot" : isWalked ? " walked" : "")}
              d={anchor(layout[e.source], layout[e.target])} markerEnd={`url(#${isWalked ? "arg" : "ar"})`} />
          );
        })}
        {nodes.map((name) => {
          const box = layout[name];
          const st = status[name];
          const cls = "gn " + (st || "ghost") + (name.startsWith("__") ? " terminal" : "");
          return (
            <g key={name} className={cls}>
              {st === "running" && <rect className="halo" x={box.x} y={box.y} width={box.w} height={H} rx={box.round ? 12 : 8} />}
              <rect x={box.x} y={box.y} width={box.w} height={H} rx={box.round ? 12 : 8} />
              <text x={box.x + box.w / 2} y={box.y + 16} textAnchor="middle">{box.label || name}</text>
            </g>
          );
        })}
        {status["wait_user"] === "waiting" && (
          <text className="glabel" x={layout.wait_user.x + layout.wait_user.w / 2} y={layout.wait_user.y + H + 13} textAnchor="middle"
            style={{ fill: "var(--wait)" }}>{t("waiting")}</text>
        )}
      </svg>
    </div>
  );
}

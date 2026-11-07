import { createContext, useContext } from "react";

export type Lang = "en" | "zh";

const dict = {
  en: {
    console: "Console", live: "Live", replay: "Replay", single: "Single agent", multi: "Multi-agent",
    newSession: "New session", turns: "Turns", modelCalls: "Model calls", cost: "Cost",
    conversation: "Conversation", database: "Database", trace: "Trace",
    business: "Business", sessionState: "Session state", snapshot: "Snapshot", liveTag: "live",
    searchAll: "Search all data…", changesThisTurn: "Changes in this turn", noChanges: "No data changed in this turn.",
    noEntities: "Nothing bound yet. Once the agent authenticates a customer, their profile and orders appear here.",
    placeholder: "Message the agent…", startHint: "Pick a scenario to start, or type your own message.",
    awaiting: "Awaiting your confirmation", running: "running", thinking: "Agent is working…",
    turn: "Turn", waiting: "waiting for user", expand: "expand", collapse: "collapse",
    systemPrompt: "System prompt", recentMessages: "Recent messages", output: "Output", input: "Input", result: "Result",
    usage: "Usage", dbChanges: "DB changes", graph: "Workflow", legendNode: "node", legendModel: "model",
    legendTool: "tool", legendGuard: "guard", legendConfirm: "confirm", replayPick: "Choose a recording",
    play: "Play", pause: "Pause", restart: "Restart", speed: "Speed", noTurn: "Send a message to see the agent's path.",
    starting: "Starting a sandbox session…", error: "Error", cacheHit: "cache hit", out: "out", in: "in",
    users: "Users", orders: "Orders", products: "Products", raw: "Raw JSON", entered: "in view",
    bound: "bound this session", results: "Search results", clear: "Clear", draft: "Draft", final: "Published",
    findings: "Findings", accepted: "Accepted", rejected: "Rejected", decisions: "Decisions", ops: "Proposed ops",
    tools: "Tools", children: "steps", greeting: "Greeting", replayHint: "Recorded sessions replay the exact event stream — no model calls.",
  },
  zh: {
    console: "控制台", live: "实时", replay: "回放", single: "单智能体", multi: "多智能体",
    newSession: "新建会话", turns: "轮次", modelCalls: "模型调用", cost: "费用",
    conversation: "对话", database: "数据库", trace: "链路",
    business: "业务数据", sessionState: "会话状态", snapshot: "快照", liveTag: "实时",
    searchAll: "搜索全库…", changesThisTurn: "本轮变更", noChanges: "本轮没有数据变化。",
    noEntities: "尚未绑定用户。Agent 完成身份核验后，这里会出现该用户的资料与订单。",
    placeholder: "给客服 Agent 发消息…", startHint: "选择一个场景开始，或直接输入。",
    awaiting: "等待你的确认", running: "进行中", thinking: "Agent 处理中…",
    turn: "第", waiting: "等待用户", expand: "展开", collapse: "收起",
    systemPrompt: "系统提示", recentMessages: "最近消息", output: "输出", input: "输入", result: "结果",
    usage: "用量", dbChanges: "处数据变更", graph: "工作图", legendNode: "节点", legendModel: "模型",
    legendTool: "工具", legendGuard: "守卫", legendConfirm: "确认", replayPick: "选择录制",
    play: "播放", pause: "暂停", restart: "重播", speed: "速度", noTurn: "发送消息后，这里会展示 Agent 的执行路径。",
    starting: "正在启动沙箱会话…", error: "错误", cacheHit: "缓存命中", out: "输出", in: "输入",
    users: "用户", orders: "订单", products: "商品", raw: "原始 JSON", entered: "进入视图",
    bound: "本会话已绑定", results: "搜索结果", clear: "清除", draft: "草稿", final: "发布",
    findings: "发现", accepted: "接受", rejected: "拒绝", decisions: "判定", ops: "提议操作",
    tools: "工具", children: "步", greeting: "问候", replayHint: "回放会原样重放录制的事件流，不调用模型。",
  },
};

export type Key = keyof typeof dict.en;
export const LangContext = createContext<Lang>("en");

export function useT() {
  const lang = useContext(LangContext);
  return (key: Key) => dict[lang][key] || dict.en[key];
}

export function turnLabel(lang: Lang, n: number) {
  return lang === "zh" ? `第 ${n} 轮` : `Turn ${n}`;
}

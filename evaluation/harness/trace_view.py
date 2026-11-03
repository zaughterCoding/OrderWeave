"""按轮次打印会话追踪树：轮 → 角色模型调用（token／缓存）→ 工具调用（钩子拦截）→ 发布守卫。

用法：
  uv run python -m evaluation.harness.trace_view data/evaluation/harness/<run>/sessions/<task>/trial-1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def render(case_dir: Path) -> str:
    lines = []
    for raw in (case_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(raw)
        if row.get("type") != "turn":
            continue
        guard = row.get("response_guard") or {}
        lines.append(f"turn {row['turn']}  success={row['agent_success']}  latency={row['latency_ms']}ms"
                     f"  primary={row.get('primary_agent') or '-'}")
        lines.append(f"  user: {row['user_message'][:120]!r}")
        if row.get("context"):
            lines.append(f"  context compaction: {row['context']}")
        for span in row.get("trace") or []:
            if span["kind"] == "model":
                u = span.get("usage") or {}
                total = u.get("input", 0) + u.get("cache_read", 0) + u.get("cache_write", 0)
                lines.append(f"  ├─ model[{span['agent']}] in={total} hit={u.get('cache_read', 0)}"
                             f" out={u.get('output', 0)} {span.get('latency_ms')}ms stop={span.get('stop_reason')}")
            else:
                mark = f" BLOCKED by {span['blocked_by']}" if span.get("blocked_by") else ""
                lines.append(f"  │   └─ tool {span['tool']} ok={span.get('result_success')}{mark} {span.get('latency_ms')}ms")
        lines.append(f"  └─ publish guard: action={guard.get('action')} terminal={guard.get('terminal')}"
                     f" events={[e['event_type'] for e in row.get('store_events') or []]}")
        lines.append(f"  assistant: {row['assistant_message'][:160]!r}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="打印单个会话的追踪树")
    parser.add_argument("case_dir")
    print(render(Path(parser.parse_args().case_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

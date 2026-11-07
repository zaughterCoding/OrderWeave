import { useState, type ReactNode } from "react";
import { useT } from "../i18n";

const TOKEN = /("(?:\\.|[^"\\])*")(\s*:)?|\b(true|false|null)\b|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g;

function highlight(text: string): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  TOKEN.lastIndex = 0;
  while ((m = TOKEN.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const key = out.length;
    if (m[1]) out.push(<span key={key} className={m[2] ? "jk" : "js"}>{m[1]}</span>, m[2] || "");
    else if (m[3]) out.push(<span key={key} className="jb">{m[3]}</span>);
    else out.push(<span key={key} className="jn">{m[4]}</span>);
    last = TOKEN.lastIndex;
  }
  out.push(text.slice(last));
  return out;
}

export function pretty(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

/** 代码块；fold 时默认只露出两行，可展开。 */
export function Code({ value, title, fold = false }: { value: unknown; title?: string; fold?: boolean }) {
  const t = useT();
  const text = pretty(value);
  const long = text.split("\n").length > 3 || text.length > 240;
  const [open, setOpen] = useState(!fold);
  const folded = fold && long && !open;
  // 折叠时用紧凑单行预览，避免只露出一个“{”。
  const shown = folded && typeof value !== "string" ? JSON.stringify(value) ?? "" : text;
  return (
    <div className="sec">
      {title && (
        <h5>
          {title}
          {fold && long && (
            <button onClick={() => setOpen(!open)}>
              {open ? t("collapse") : `${t("expand")} · ${text.length.toLocaleString()} chars`}
            </button>
          )}
        </h5>
      )}
      <pre className={"code" + (folded ? " fold" : "")} onClick={folded ? () => setOpen(true) : undefined}
        style={folded ? { cursor: "pointer" } : undefined}>
        {typeof value === "string" ? shown : highlight(shown)}
      </pre>
    </div>
  );
}

"""离线核对零售评测资料，不调用模型或存储服务。"""

import hashlib
import json
from pathlib import Path


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _check_groups(rows, split_groups, id_key, text_key):
    ids, texts, groups = set(), {}, {}
    for row in rows:
        identity, group, split = row[id_key], row["group_id"], row["split"]
        _check(identity not in ids, f"重复 ID：{identity}")
        ids.add(identity)
        _check(split in {"dev", "test"} and group in split_groups[split], f"划分不一致：{identity}")
        _check(groups.setdefault(group, split) == split, f"家族跨划分：{group}")
        text = "".join(row[text_key].lower().split())
        _check(bool(text), f"空文本：{identity}")
        _check(text not in texts, f"重复文本：{identity}")
        texts[text] = split
        _check(row["source_kind"] == "project_authored" and row["source_task_ids"] == [], f"样本来源错误：{identity}")
    _check(set(groups) == set(split_groups["dev"]) | set(split_groups["test"]), "划分表存在空家族")
    _check(set(split_groups["dev"]).isdisjoint(split_groups["test"]), "划分表家族重叠")


def validate(root=None):
    root = Path(root) if root is not None else Path(__file__).parent / "data" / "retail"
    labels = json.loads((root / "labels.json").read_text(encoding="utf-8"))
    splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    sources = json.loads((root / "sources.json").read_text(encoding="utf-8"))
    intents = _jsonl(root / "intent_slots.jsonl")
    queries = _jsonl(root / "retrieval.jsonl")
    knowledge = _jsonl(root / "knowledge.jsonl")
    primary = set(labels["primary_labels"])
    _check(len(primary) == 6 and primary <= set(labels["known_core_labels"]), "主标签不兼容原枚举")
    _check(set(labels["prediction_aliases"].values()) <= primary, "标签别名目标错误")
    _check(len(sources["commit"]) == 40, "来源提交未固定")
    _check_groups(intents, splits["intent_groups"], "case_id", "message")
    _check_groups(queries, splits["retrieval_groups"], "query_id", "query")
    for row in intents:
        identity, message = row["case_id"], row["message"]
        _check(row["primary_intent"] in primary, f"未知主标签：{identity}")
        _check(set(row["auxiliary_needs"]) <= set(labels["auxiliary_needs"]), f"未知辅助诉求：{identity}")
        _check(len(row["history"]) <= 3, f"历史超出约定：{identity}")
        _check(all(h["role"] in {"user", "assistant"} and h["content"] for h in row["history"]), f"历史格式错误：{identity}")
        for slot in row["slots"]:
            start, end = slot["start"], slot["end"]
            _check(slot["type"] in labels["slot_types"], f"未知槽位：{identity}")
            _check(type(start) is int and type(end) is int and 0 <= start < end <= len(message), f"槽位跨度错误：{identity}")
            _check(message[start:end] == slot["text"], f"槽位文本不匹配：{identity}")
    for split in ("dev", "test"):
        side = [row for row in intents if row["split"] == split]
        _check({row["primary_intent"] for row in side} == primary, f"主标签覆盖不足：{split}")
        _check({s["type"] for row in side for s in row["slots"]} == set(labels["slot_types"]), f"槽位覆盖不足：{split}")
    evidence = {}
    for row in knowledge:
        identity = row["evidence_id"]
        _check(identity not in evidence and identity.startswith(row["doc_id"] + ".P"), f"证据 ID 错误：{identity}")
        evidence[identity] = row
        _check(row["scope"] == "project_zh" and row["language"] == "zh", f"知识适用范围错误：{identity}")
        _check(row["content"] and row["source_refs"], f"知识缺内容或来源：{identity}")
        _check(row["policy_commit"] == sources["commit"], f"知识政策版本错误：{identity}")
        _check(row["source_kind"] in {"upstream_paraphrase", "upstream_derived", "project_authored"}, f"知识来源类型错误：{identity}")
        for ref in row["source_refs"]:
            if "url" in ref:
                _check(sources["commit"] in ref["url"], f"知识引用浮动版本：{identity}")
            else:
                _check(row["source_kind"] == "project_authored" and ref.get("project_document") == row["doc_id"], f"自编来源错误：{identity}")
    for row in queries:
        identity, qrels = row["query_id"], row["qrels"]
        _check(set(qrels) <= set(evidence), f"引用不存在的证据：{identity}")
        _check(all(type(v) is int and v in {1, 2} for v in qrels.values()), f"相关性等级错误：{identity}")
        groups = row["required_evidence_groups"]
        _check(all(group and len(group) == len(set(group)) for group in groups), f"必要证据组错误：{identity}")
        required = [item for group in groups for item in group]
        _check(len(required) == len(set(required)), f"证据组重复：{identity}")
        _check(set(required) == {key for key, value in qrels.items() if value == 2}, f"必要证据与相关性不一致：{identity}")
        answerability = row["answerability"]
        _check(answerability in {"answerable", "needs_tool", "unanswerable"}, f"答案类型错误：{identity}")
        _check(bool(groups) if answerability == "answerable" else not qrels and not groups, f"无答案记录误标证据：{identity}")
    for entry in sources["files"]:
        if entry["path"] is None:
            continue
        path = (root / entry["path"]).resolve()
        _check(path.is_relative_to(root.resolve()), "来源路径越界")
        raw = path.read_bytes()
        _check(hashlib.sha256(raw).hexdigest() == entry["sha256"] and len(raw) == entry["bytes"], f"来源哈希错误：{entry['path']}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["artifacts"]:
        path = (root / entry["path"]).resolve()
        _check(path.is_relative_to(root.resolve()), "资产路径越界")
        _check(hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], f"资产哈希错误：{entry['path']}")
    actual_counts = {"intent_cases": len(intents), "retrieval_queries": len(queries), "knowledge_passages": len(knowledge)}
    _check(actual_counts == manifest["counts"], "资产数量与清单不一致")
    return {"status": "PASS", **actual_counts, "model_calls": 0}


if __name__ == "__main__":
    print(json.dumps(validate(), ensure_ascii=False))

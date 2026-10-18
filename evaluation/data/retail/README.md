# 订单查询与退货评测资料

版本：retail-zh-v1；日期：2026-10-17（America/Tijuana）。本目录提供第二步的固定合成资料，尚未运行模型评测。它是可独立使用的项目资产，不依赖内部 docs。业务实现和知识导入在后续步骤进行。

## 固定来源与许可

上游为官方 `sierra-research/tau2-bench` 的 retail 文本领域，发布版 `v1.0.1`，提交 `fc0055dc4e0a316c3f83133267fbd6faaa770992`。仓库目前对外名称为 τ³-bench。

- [原政策](source/retail_policy.en.md)按字节保留，不添加自编规则。
- [MIT 声明](source/LICENSE)保留上游 Copyright (c) 2025 Sierra Research，适用于上游材料及其派生内容。
- [来源表](sources.json)记录固定 URL、SHA-256 与是否保存本地副本；没有把完整订单库或隐藏任务答案复制到本目录。
- [中文知识](knowledge.jsonl)包括 9 个主题、29 个证据段，是项目中文场景使用的语料；`source_kind` 区分上游转述、基于工具的解释和项目自编建议。

## 样本与标注

| 文件 | 内容 |
| --- | --- |
| [标注规范](annotation.md) | 主意图、辅助诉求、槽位、相关性、指标及已知限制 |
| [标签表](labels.json) | 6 个既有主标签、辅助诉求、6 类目标槽位、固定预测别名 |
| [意图与槽位样本](intent_slots.jsonl) | 120 条，6 类各 20；dev 72／test 48；包含 12 条带上下文的样本 |
| [检索问题](retrieval.jsonl) | 48 条；两侧各 20 个可回答、2 个需查工具、2 个无答案；保留分级相关性和必要证据组 |
| [分组划分](splits.json) | 意图 30 个家族、检索 24 个家族；同家族不跨划分 |
| [资产清单](manifest.json) | 数据版本、数量与逐文件 SHA-256 |

主标签为 order_status、refund、query、human_handoff、greeting、other，复用既有枚举。政策与准备咨询均属于 query，以 scenario 和相关性标注区分。一个主要标签用于识别指标，辅助诉求只记录，不要求本轮改为多标签，也不标注任务拆分或执行计划。

## 知识数据结构

每行是一个独立证据段：`evidence_id`、`doc_id`、`title`、`content`、`language`、`source_kind`、`source_refs`、`policy_release`、`policy_commit`、`scope`、`category`。

证据段是本轮固定检索评分单位；同一文档可有多个段。回答引用与相关性标注使用稳定 evidence_id，不能引用模型临时切分后的数组位置。第三步适配装载时需要保留证据身份；现有按标题／片段哈希导入尚未接好本格式。

知识装载仅使用 knowledge.jsonl。intent_slots.jsonl、retrieval.jsonl、标签及相关性答案只供评测，不能作为知识文档导入或发给被评测模型。

语料中的项目指南仅为沙箱服务建议，不是额外退货资格、承运规定或到账保证。整份语料目前都标为 `project_zh`，不能把它作为英文原版任务的额外政策。英文原版继续使用原 policy，不把任务 gold、参考动作、用户隐藏计划或自编寄回要求注入上下文。

## 本轮边界

采用一个主要意图评分并保留辅助诉求。样本、指南和相关性均由本次助手自编并复核，未经独立人审；这是小规模固定比较集，不是真实流量、独立盲测或官方任务译文。本文件不宣称模型准确率、检索质量或业务成功率。

当前原政策之外未明确的退货期限、运费、寄回地址、审核和到账时间都应保持未知。中文指南的准备建议不增加原包装、配件、清洁或数据处理方面的资格门槛。

在项目根使用现有 Python 环境做纯离线资料检查：

```powershell
uv run --locked --offline python -m evaluation.validate_retail_data
uv run --locked --offline python -m pytest -q tests/test_retail_data.py
```

检查不调用模型、不连接 Redis／ChromaDB，也不把语料自动导入现有知识库。只有完整性通过不能证明标注无歧义或模型有效。第三步才适配装载、历史透传及指标运行；不得通过评测适配器从原句重新抽槽位来补模型输出。

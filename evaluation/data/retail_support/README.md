# 无线耳机支持沙箱资料

本目录是 AgentEcommerce 的公开沙箱支持资料，供 `RETAIL_SUPPORT_ENABLED=true` 时使用。资料由项目自行编写，不是厂商手册，也没有经过真实品牌、真实设备、真实客服或真实模型验证。它只覆盖公开种子中的 `Wireless Earbuds`（`product_id=9924732112`）在蓝牙无法发现、无法配对或无法连接时的有限排查。

## 文件

- `source/earbuds_pairing.zh.md`：自编说明正文，列出 `PAIR-01`—`PAIR-03` 三个沙箱步骤、边界和未知项。
- `knowledge.jsonl`：6 段 `T17.E01`—`T17.E06` 知识片段，带 `product_id`、`issue`、`support_version` 与 `step_id` 等适用字段。
- `manifest.json`：逐文件哈希、6 段 ID、旧 29 段文件及哈希、合并后的 35 段口径。
- `scenarios.json`：C01—C08 八个固定验收案例的说明与可观察完成条件。预期脚本和断言只在测试端使用，不注入业务模型。

## 与旧资料的关系

原有 `evaluation/data/retail_sandbox/knowledge.jsonl` 的 29 段、固定政策与种子保持逐字节不变。启用本功能时，运行时会新建独立集合（默认 `retail_earbuds_support_v1`），把旧 29 段和新 6 段合并导入为 35 段；不会写回或清空旧集合。旧集合仍按 29 段校验。

## 启用与运行

```powershell
$env:RETAIL_MODE = "true"
$env:RETAIL_SUPPORT_ENABLED = "true"
$env:RETAIL_SUPPORT_COLLECTION = "retail_earbuds_support_v1"
$env:RETAIL_AGENT_MODE = "single"   # 或 multi
```

启用要求中文零售 profile、知识检索开启，且支持集合不能与 `RETAIL_COLLECTION` 同名。开关默认关闭；关闭时原单／多 Agent、29 段知识、英文任务和旧客服模式保持原语义。

## 离线验收

Fake 意图、Agent、查询改写与重排，真实 SQLite、真实 stdio MCP 和已有 ChromaDB 缓存，可运行：

```powershell
.venv/Scripts/python.exe -m pytest tests/test_retail_support.py -q
.venv/Scripts/python.exe -m pytest tests/integration/test_retail_support.py -q
```

这些检查只证明接线、状态、资料适用性和业务保护，不代表真实模型回答质量、检索指标或多 Agent 收益。测试不调用远程模型、不下载 Embedding、不读取 `.env`。

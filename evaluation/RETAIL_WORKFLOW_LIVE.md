# U02 真实模型兼容验收

运行器 `evaluation/run_retail_workflow_live.py` 只验收当前退货工作图与真实生成模型的配合，复用既有 V03 业务夹具、RetailRuntime、SQLite、stdio MCP、预算账本和旧运行器的记录／停止保护。

固定四案，各 single／multi 一次：准备展示后重建 runtime／MCP 并确认；修改商品与退款去向后重新确认；取消；转人工。真实模型使用 `deepseek-flash`，不调用 Judge 或 UserSimulator，不把小样本通过率解释为回答质量提升。

本轮只启用九个业务 MCP 工具，检索、长期记忆与有限支持关闭；不需要 Redis／Chroma，不加载 Qwen，也不改变生产默认。重启验收针对 runtime、MCP 和 checkpoint 连接，操作系统级进程重启工程验证见 [Fake 演示](RETAIL_WORKFLOW.md)。

```powershell
uv run --locked --offline python -m evaluation.run_retail_workflow_live preflight --run-dir data/evaluation/u02-live-new-run
uv run --locked --offline python -m evaluation.run_retail_workflow_live live --run-dir data/evaluation/u02-live-new-run
```

真实执行需要维护者明确授权本轮预算并配置项目 `.env` 中的 `ANTHROPIC_API_KEY`。脚本当前冻结总预算 5 元、官方端点和模型，其他模型会被拒绝；新目录创建独立账本，全部八条共享同一个上限。不得通过新建目录或账本规避已授权的累计上限。正式运行有单次启动标记，已有请求的目录不能重跑；保留失败，不自动补跑。

价格必须在使用前再次核对：[DeepSeek 官方价目](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)。本次核对的高峰缓存未命中输入价为 2 元／百万 token、输出 8 元／百万 token；账本按该价保守预留及结算所有输入，不计算缓存折扣。已有账本在请求前按完整请求字节上界预留，成功按 usage 结算，失败或 usage 缺失按预留上界，SDK 重试为 0。费用是保守估算，供应商账单另行核实。

`plan.json` 保存源码哈希、用例、价格与运行配置，`preflight.json` 保存零请求预检。每案的 `requests.jsonl`、`turns.json`、`events.jsonl`、`result.json`、业务库和 checkpoint 保存原始证据；全轮结果在 `summary.json`，费用在唯一 `ledger.sqlite`。检查以真实业务事件、目标订单、完整 pending、图等待与恢复为准，并核对无关用户／商品／订单未改变。

```powershell
uv run --locked --offline python -m pytest tests/test_retail_workflow_live.py tests/test_retail_ledger.py tests/test_retail_workflow.py -q
```

这些离线测试不发出付费请求，覆盖新账本与禁止覆盖、冻结变动拒绝、四个脚本分支、真实 SQLite checkpoint 和证据序列化。没有扩大原质量评测集、调整模型 Prompt 或修改 RAG 策略。

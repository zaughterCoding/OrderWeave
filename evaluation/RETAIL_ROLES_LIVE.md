# 角色图真实模型兼容验收

`evaluation/run_retail_roles_live.py` 复用 RetailRuntime、原零售角色、stdio MCP、业务 SQLite、LangGraph Checkpointer 和既有预算／请求记录器。固定六条 multi 对话：技术→账单与等待重启确认（graph／legacy 配对）、修改去向＋否定后确认（graph）、技术→账单→人工（legacy／graph 配对）、插入问题后取消（graph）。每条独立业务库／checkpoint，全部共用一个新五元账本。

正式运行沿用当前 Qwen Embedding＋Hybrid＋Qwen 重排＋relevance，启用原 35 段支持集合，关闭长期记忆。不改变 Prompt／政策／检索策略，不调用 Judge／UserSimulator。只从当前项目 `.env` 加载已有密钥，不读取参考工程。模型固定 deepseek-flash，端点固定 DeepSeek Anthropic 兼容接口。

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
uv run --locked --offline python -m evaluation.run_retail_roles_live preflight --run-dir data/evaluation/u02-roles-live-new/run
uv run --locked --offline python -m evaluation.run_retail_roles_live live --run-dir data/evaluation/u02-roles-live-new/run
```

真实运行前须有维护者明确的新累计费用上限，并核对 [官方价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/) 和 [Anthropic 兼容接口](https://api-docs.deepseek.com/guides/anthropic_api/)。当前冻结五元、高峰未命中输入 2／输出 8 元每百万 token，所有输入按高价保守结算，不减缓存折扣。每次完整请求字节上界预留，成功按 usage、失败或缺失按上界结算，SDK 重试 0。`plan.json` 冻结模型、价格、场景和源码；`started.json` 与唯一账本禁止重跑。不能通过换目录重置本次预算；失败保留，不按结果追加付费尝试。

需准备隔离的本地 Chroma（默认 127.0.0.1:8002）及已缓存 Qwen 文件／原独立推理环境，不安装或下载。为本轮新建的 Chroma 集合以 `u02rl_` 加目录哈希命名，原集合不复用；运行完成后清理临时服务并保留原服务状态及卷。每条最多六个真实脚本输入、40 次模型请求，每轮 32 上限；预算不足停止。

每条 `requests.jsonl` 留存目标角色实际输入与模型原始输出；`retrieval.jsonl`、`turns.json`、`events.jsonl`、`mcp_stderr.log`、`result.json`、业务库和 checkpoint 留存节点、证据、等待、恢复与写入。全轮 `summary.json` 和 `ledger.sqlite` 汇总检查与成本。检查交接来源轮次、用户报告和证据是否进入实际目标请求，以及业务写入、修改／否定、取消／人工和重启后状态。工具轨迹中的拦截尝试不等于真实 MCP 执行。

离线驱动测试需要隔离 Chroma 与已有轻量模型缓存，生成完全 Fake；正式配置仍冻结 Qwen 默认链。运行 `python -m pytest tests/test_retail_roles_live.py tests/test_retail_ledger.py -q`。真实结果另作执行者语义复核，非独立盲审；两个配对只描述兼容、token／调用和耗时，不能据此声称准确率、性能或成本因果收益。生产并发、长期 checkpoint 清理及完整质量评测不在范围内。

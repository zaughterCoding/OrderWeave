# AgentEcommerce

基于 Anthropic 兼容协议与多路融合的电商客服 Agent 项目。当前提供 HTTP API 与交互式 CLI，接通意图识别、多 Agent 编排、Skills、ChromaDB 知识库检索、分层记忆、性能监控与回复评测。

## 环境要求

- Windows
- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Docker（仅运行独立的 Redis 与 ChromaDB 服务）

## 安装

```powershell
uv python find 3.12
uv lock
uv sync --locked --group dev
```

## 本地服务

`compose.yml` 只包含独立的 Redis 与 ChromaDB，容器名、卷和宿主机端口均与本机其他项目隔离：

```powershell
docker compose -f compose.yml up -d redis chromadb
docker compose -f compose.yml stop
```

默认端口：Redis `6380`，ChromaDB `8002`。卷为 `agent-ecommerce-redis-data`、`agent-ecommerce-chromadb-data`。

## 启动 API 与 CLI

先启动 Redis 与 ChromaDB，并配置 `.env`。API 默认只绑定本机回环地址：

```powershell
uv run --locked --offline python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Swagger 位于 `http://127.0.0.1:8000/docs`，健康检查为 `GET /health`。也可以直接运行：

```powershell
uv run --locked --offline python -m api.main
```

交互式 CLI：

```powershell
uv run --locked --offline python -m api.main --cli
```

输入 `quit`、`exit` 或 `退出` 结束 CLI。导入 `api.main` 只加载配置，不连接 Redis、ChromaDB 或付费模型；组件在应用启动时创建，停机时关闭。

### HTTP 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 服务状态与 Agent 统计 |
| GET | `/skills` | 当前已加载的 Skills |
| POST | `/skills/reload` | 重新扫描并热重载 Skills |
| POST | `/chat` | 主对话：记忆读取、意图识别、Agent 路由、记忆写回 |
| GET | `/monitor` | 监控摘要、告警与优化建议 |
| GET | `/trace/tool/{request_id}` | 单次请求的工具调用明细 |
| GET | `/trace/tools` | 最近请求的工具调用明细 |
| GET | `/metrics` | Prometheus 文本指标 |
| POST | `/search` | 查询改写、并行召回与重排 |
| GET | `/knowledge/stats` | 知识库片段总数 |
| POST | `/knowledge/add` | 批量导入 `{title, content}` 文档 |
| POST | `/knowledge/upload` | 上传 `.txt`／`.md` 或 JSON 数组，限制 10MB |
| POST | `/eval/run` | 运行意图与对话质量评测 |

`POST /chat` 接受 `message`、`user_id` 和可选的 `conv_id`；未提供 `conv_id` 时由服务生成。`user_id`／`conv_id` 是原接口参数，不是已认证身份，本项目没有登录、权限控制或支付功能。

## 分层记忆

- 工作记忆存于 Redis 列表，键为 `wm:{user_id}:{conv_id}`，保留最近消息，TTL 24 小时。
- 会话摘要存于 `summary:{user_id}:{conv_id}`，TTL 24 小时；达到 15 条消息时压缩，只保留最近 5 条。
- 情景记忆与用户画像存于 ChromaDB 的 `episodic`、`user_profile` 集合，支持同一用户跨会话召回，查询按 `user_id` 隔离。
- 摘要生成或情景记忆写入失败时保留原消息，不会静默丢失。
- Embedding 缓存默认位于项目内 `data/chroma_cache`，可用 `CHROMA_EMBEDDING_CACHE` 覆盖。

## 监控与评测

- `PerformanceMonitor` 定时读取 `AgentOrchestrator.get_stats()` 与 `MCPToolManager.get_stats()`，按阈值产生告警，并把成功率与平均延迟折算为路由惩罚回传给编排器。
- 告警按 `指标:标签` 维护活动状态：故障持续时更新同一条活动告警，恢复时标记为已解决，再次触发时创建新告警；webhook 只在新建告警时发送。
- 告警可选通过 webhook 发送，指标可选通过 Prometheus 暴露；自动化测试不会向外部发送 webhook。
- 设置 `MONITOR_PROMETHEUS_PORT` 后，监控进程在该端口启动 Prometheus HTTP 暴露器，`/metrics` 同时输出这些业务指标：
  - `agent_success_rate{agent}`（Gauge）：Agent 当前成功率。
  - `agent_avg_latency_ms{agent}`（Gauge，毫秒）：Agent 当前平均延迟；采集轮询不会产生延迟样本。
  - `tool_success_rate{tool}`（Gauge）：工具当前成功率。
  - `agent_executions_total{agent}`（Counter）：监控观测到的 Agent 执行次数累计值；多 Agent 请求会为每个参与的 Agent 各计一次，不代表 HTTP 请求数。
- 未设置 `MONITOR_PROMETHEUS_PORT` 时只注册 Python 进程默认指标，不产出上述业务指标。
- `EndToEndEvaluator` 计算意图识别 Accuracy／Macro-F1，并使用 LLM-as-Judge 评估相关性、准确性、完整性与有用性。
- Judge 解析失败的结果记录 `GRADER_ERROR`、`passed=False` 和空 `scores`；报告保留结果总数并统计 `grader_errors`，平均值只使用有效评分。
- `baseline_path` 只读；每次评测写入新的 `report_<timestamp>.json`，不会覆盖基线或已有报告。
- 工具平均延迟只对实际记录延迟的调用求平均，样本数通过 `latency_samples` 暴露。

## 零售组件基线

`evaluation/data/retail/` 保存冻结的 `retail-zh-v1` 资料：120 条意图／槽位样本、48 个检索问题、29 段知识与固定划分。校验器只读；不要修改 gold、qrels 或按 test 反复调参。

```powershell
# 纯离线：资料校验与计分合同自检，不导入模型或存储客户端，不发请求
uv run --locked --offline python -m evaluation.run_retail_baseline --mode offline-check

# 真实 ChromaDB 向量检索（V）：使用独立集合，结束后只删除本次集合
uv run --locked --offline python -m evaluation.run_retail_baseline --mode vector --split dev

# 付费链路（仅在有明确新预算时）：意图与改写／召回／重排（P）
uv run --locked --offline python -m evaluation.run_retail_baseline --mode live --split dev `
  --budget-cny <授权上限> --input-price-cny-per-mtok <当前输入单价> `
  --output-price-cny-per-mtok <当前输出单价>
```

- `--split` 可选 `dev`、`test`、`all`，默认 `all`。
- `--component` 可选 `both`、`intent`、`retrieval`，默认 `both`；只运行指定组件的 live 阶段，未选组件标 `NOT_RUN_COMPONENT`。
- 分阶段或 A/B 对照时，用 `--budget-ledger` 指定同一份累计账本，并用 `--stage-label`、`--ab-label`、`--config-label` 记录阶段与配置身份；未完成条目按计划上界占用，重跑不会清零。检索候选可用 `--vector-from` 复用已封存的 V 预测，避免重复付费。
- 检索链最终固定交付 `top_k=5`；内部每路召回量由生产配置 `mcp/tool_manager.py:retrieval_config` 决定，预检、运行记录和实际请求共用该配置。
- `vector` 要求 ChromaDB 已在 `CHROMA_HOST:CHROMA_PORT`（默认 `127.0.0.1:8002`）运行，且本地 Embedding 缓存完整；缺失时直接停止，不自动下载模型。29 段知识按 `evidence_id` 幂等导入隔离集合，旧集合不受影响。
- `live` 在创建模型客户端前检查预算与显式单价；缺失即拒绝。有密钥不代表可以调用。远程 Embedding 被显式屏蔽，意图向量回退到本地字符 n-gram。
- 运行产物写入 `data/evaluation/retail/<run_id>/`（已忽略）：`run.json`、`metrics.json`、`intent_predictions.jsonl`、`retrieval_predictions.jsonl`、`usage.jsonl`。
- 状态含义：`OFFLINE_READY_REAL_BASELINE_PENDING_BUDGET`（离线就绪、真实基线待预算）、`PARTIAL_RUN`、`SMOKE_COMPLETE`（指定子集烟测完成，不代表完整基线）、`COMPONENT_COMPLETE`（所选组件完整跑完，不代表整体基线）、`BASELINE_COMPLETE`（识别与检索完整跑完）。Fake 或离线检查只证明接线、计分与边界，不代表真实模型成绩。

## 零售沙箱运行

设置 `RETAIL_MODE=true` 后，HTTP API 与 CLI 切换到零售沙箱：支持订单查询、退货申请和知识咨询。默认由单 Agent 处理；设置 `RETAIL_AGENT_MODE=multi` 后改用现有 general／technical／billing／escalation 四角色的零售适配版。两种模式共用同一套服务端展示、确认与业务存储逻辑，默认客服模式不受影响。

零售模式下，九个订单／用户／退货相关业务工具由主程序启动的本地 stdio MCP 子进程提供，主程序通过官方 `mcp` SDK（锁定 `1.12.4`）完成初始化、工具发现与 `tools/call`，不提供外部客户端或 HTTP MCP 入口。`search_knowledge_base` 仍在主程序内走原有 RAG 链路。子进程与主程序使用同一个 SQLite 绝对路径，不复制业务状态；工具调用失败不会自动改走进程内直调。

固定运行资料位于 `evaluation/data/retail_sandbox/`：公开种子 `db.json`、`policy.md`、29 段 `knowledge.jsonl`、上游 MIT `LICENSE` 与 `sources.json`。启动时校验种子、政策与知识哈希、知识条数及 500 用户／1000 订单／50 商品；该目录不含官方隐藏任务、gold、用户模拟器或判定器。

```powershell
# 启动本地 Redis 与 ChromaDB（沿用 compose.yml 的卷）
docker compose up -d redis chromadb

# 必填：模型接入与当前官方单价；单价缺失时拒绝启动付费链路
$env:RETAIL_MODE = "true"
$env:ANTHROPIC_API_KEY = "<key>"
$env:LLM_MODEL = "<model>"
$env:RETAIL_INPUT_CNY_PER_MTOK = "<当前输入单价>"
$env:RETAIL_OUTPUT_CNY_PER_MTOK = "<当前输出单价>"

# HTTP 入口
uv run --locked --offline uvicorn api.main:app --host 127.0.0.1 --port 8000
Invoke-RestMethod -Method Post http://127.0.0.1:8000/chat -ContentType 'application/json' `
  -Body '{"user_id":"sandbox-client","conv_id":"conv-1","message":"帮我查一下我的订单"}'

# CLI 入口；RETAIL_CLIENT_USER 与 RETAIL_CONV_ID 缺省分别为 cli_user 和随机值
uv run --locked --offline python -m api.main --cli
```

- `RETAIL_AGENT_MODE`：`single`（默认）或 `multi`；`multi` 使用四角色编排，身份绑定与退货提交等状态工具只对每轮主执行者开放，辅助角色只读，Composer 无工具。
- `RETAIL_DB_PATH`：SQLite 路径，默认 `data/retail/retail.db`。首次启动从固定种子初始化；已有非空业务库不会自动重置，需要新数据时请显式指定新的空路径。
- `RETAIL_SEED_PATH`、`RETAIL_POLICY_PATH`、`RETAIL_KNOWLEDGE_PATH`：默认指向上述固定资料，通常无需设置。
- `RETAIL_COLLECTION`：知识集合名，默认 `retail_knowledge_v1`；只导入冻结的 29 段知识，旧演示政策不参与。
- `RETAIL_MEMORY_COLLECTION_SUFFIX`、`RETAIL_MEMORY_KEY_PREFIX`：记忆集合后缀与 Redis 键前缀，默认 `_retail`、`retail:`；并行运行时请显式区分。
- `RETAIL_BUDGET_LEDGER`、`RETAIL_BUDGET_CNY`：跨进程累计账本路径与上限，默认 `data/retail/budget_ledger.db`、`5` 元。所有 Agent、改写、重排、摘要请求共用同一账本；预算不足或价格不可核实会停止付费调用，账本不会自动清零。
- `RETAIL_EVENT_LOG`：业务事件 JSONL，默认 `data/retail/events.jsonl`。
- `RETAIL_MCP_TIMEOUT_SECONDS`：MCP 初始化与单次工具调用的读取超时秒数，默认 `30`，必须为正数。
- `RETAIL_MCP_STDERR_LOG`：工具子进程诊断日志路径，默认 `data/retail/mcp_stderr.log`；stdout 只用于协议消息。
- `RETAIL_PROFILE_ENABLED`：用户画像开关，默认 `false`；启用会增加一次付费请求并计入账本。
- `RETAIL_CASE_ID`、`RETAIL_RUN_ID`：可选的运行标识，写入事件与账本；`RETAIL_AGENT_MAX_TOKENS`（默认 2048）、`RETAIL_MAX_MESSAGE_CHARS`（默认 4000）、`RETAIL_MAX_CONTEXT_CHARS`（默认 12000）限制单轮输入输出。
- 身份绑定：服务端通过邮箱或“姓名＋邮编”绑定沙箱用户；请求里的 `user_id`、`conv_id` 不是订单授权，只能查询和操作当前绑定用户的订单。
- 退货：仅 `delivered` 订单可申请；去向只能是原支付方式或用户已有礼品卡。详情由程序保存并实际展示，只有当前用户对同一详情的明确确认才能写入；修改详情需重新展示并重新确认，多订单逐笔确认与提交。人工交接只记录模拟事件，不代表真实工单、邮件或资金到账。

### 无线耳机有限支持（默认关闭）

`RETAIL_SUPPORT_ENABLED` 默认 `false`。开启后，单 Agent 与多 Agent 都只额外支持公开种子中的 `Wireless Earbuds`（`product_id=9924732112`）在蓝牙无法发现／配对／连接时的有限建议；资料是项目自编沙箱材料，不是厂商手册，也没有经过真实设备、品牌客服或真实模型验证。未知按键时长、恢复出厂、驱动、固件、兼容性、维修、换货、保修和硬件损坏结论均不覆盖；排查失败不等于退货资格。

```powershell
$env:RETAIL_MODE = "true"
$env:RETAIL_SUPPORT_ENABLED = "true"
$env:RETAIL_SUPPORT_COLLECTION = "retail_earbuds_support_v1"
$env:RETAIL_AGENT_MODE = "single"   # 或 multi
```

- 开启要求 `RETAIL_LANGUAGE=zh`、知识检索开启，且支持集合不能与 `RETAIL_COLLECTION` 同名；冲突、缺文件或清单哈希不符会明确拒绝启动。
- 新集合导入旧 29 段固定证据加 6 段项目自编耳机资料，共 35 段；旧集合、旧 29 段校验和 T16 资产保持不变，不会自动清空或改写。
- 同一会话的有限工作记录保存在现有 SQLite 的独立表中，按 `(client_user_id, conv_id)` 隔离，可在进程重启后继续；不依赖可选 Redis、用户画像或跨会话记忆。
- 支持记录只保存当前诉求、用户真实反馈、实际展示过的建议与证据、未解决项和最近有限对话；检索到但未展示的内容不算建议，用户报告不等于工具实测，模型摘要不作为可信进度。
- 用户当前明确选择退货后仍走原 `prepare_return` → 展示 → 后续真实确认 → 提交链路；“如果不行就退”“先排查、不行再退”只是条件选择，不构成提交授权。
- 关闭开关时，原单／多 Agent、29 段知识、英文评测和旧客服模式语义不变，也不增加支持专用阶段。

对应的离线验收只使用 Fake 模型、真实 SQLite、真实 stdio MCP 和已有 ChromaDB 缓存：

```powershell
uv run --locked --offline python -m pytest tests/test_retail_support.py -q
uv run --locked --offline python -m pytest tests/integration/test_retail_support.py -q
```

这些检查验证接线、状态隔离、重启续接、资料适用性与业务保护，不代表真实模型回答质量、检索指标或多 Agent 收益。详见 `evaluation/data/retail_support/README.md`。

## τ-retail 固定子集评测

该入口在项目内另建独立 Python 3.12 环境运行固定上游文字评测组件，不改变主环境或主 `uv.lock`。固定上游为 `sierra-research/tau2-bench` `v1.0.1`／提交 `fc0055dc4e0a316c3f83133267fbd6faaa770992`；任务顺序固定为 11、12、46、48、51、83、92、108，各 1 次，`concurrency=1`、`trials=1`、计划 `seed=300`（只记录在配置和协议日志，当前未传给模型 SDK，不承诺确定性复现）、`max_steps=200`、`max_errors=10`。被测业务始终走本文档所述的 `RetailService`／`RetailAgent`／`RetailStore`，不使用官方被测 Agent。

```powershell
# 一次性建立独立评测环境（项目内；不安装 voice/knowledge/gym/experiments extras）
uv venv --python 3.12 .venv-tau
uv pip install --python .venv-tau/Scripts/python.exe `
  "tau2 @ git+https://github.com/sierra-research/tau2-bench@fc0055dc4e0a316c3f83133267fbd6faaa770992"

# 必填：三个角色共用同一 DeepSeek Anthropic 兼容端点；密钥只从根 .env 读取
$env:ANTHROPIC_API_KEY = "<key>"
$env:ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"

# 免费预检：核对任务、种子、环境、价格、配置与费用计划，不发送模型请求，不产生费用
uv run --locked --offline python evaluation/run_tau_retail.py preflight --run-id <run-id>

# 显式付费运行：8 案各 1 次，共用项目内持久化账本，累计上限 ¥5；不足即记 PARTIAL 并停止
uv run --locked --offline python evaluation/run_tau_retail.py run --run-id <run-id>
```

固定资料与依赖记录位于 `evaluation/data/tau_retail/`（含原版任务、政策、公开种子、许可与 `requirements-freeze.txt`），运行说明见同目录 `ENVIRONMENT.md`。预检通过后把源码、资产、角色配置与顺序哈希写入 `data/evaluation/tau_retail/<run-id>/manifest.json`；正式运行前会复核，源码变化时拒绝继续。产物包含逐案原序对话、工具与确认事件、前后完整业务快照、官方 DB／NL 分量、Judge 原始输出、项目行为检查和共享账本。只有 Agent 与用户均正常停止、两个原评分分量有效时，该案才计入有效完成 `M/8`；`S/M` 中的 `S` 为原 reward 两分量乘积为 1 的案件数，只有 `M=8` 时报告 `S/8`。业务金额沿用原任务的 USD，模型费用账本使用 CNY。

DB 分量使用每案实际完整 SQLite 业务快照，经官方 `RetailDB` 模型和官方 hash 比较得到；预测侧不执行官方原生 strict replay，只以实际快照作为状态来源，不能把该适配描述为原生轨迹回放。`summary.json` 中 `run_status` 只表示计划案例是否执行并保存行（`EXECUTION_COMPLETE`／`EXECUTION_PARTIAL`），`evaluation_status` 单独表示评测是否有效完整（`EVALUATION_COMPLETE`／`PARTIAL`／`EVALUATOR_INVALID`），`successful_effective_completed`／`failed_effective_completed` 在有效完成案例内区分任务成败。配置中的 `seed=300` 未实际传给模型 SDK，不能据此声称结果可确定复现。

结果只代表“τ³-bench v1.0.1 retail 固定 8 任务／项目适配环境／DeepSeek 用户与评分配置”，不是全量排行榜、原生官方运行环境成绩、独立盲测或生产可用性证明；模型失败是有效实验结果，不通过反复重跑刷分。公开复现不依赖被忽略的 `docs/`。

## 单 Agent 与多 Agent 零售对照

`evaluation/run_t16_retail.py` 在同一批固定英文任务上比较单 Agent（A）与现有编排的零售适配版（B）：11、12、48、51、83、92、108 各运行 1 次，固定交替顺序共 14 个 trial；46 只做离线参考核验，不进入付费分母，也不修改 gold。两组都使用固定英文政策、同源独立 SQLite、真实本地 MCP 九工具和官方 UserSimulator／DB／NL 评分；不含中文 RAG、长期画像或跨案例记忆。

```powershell
# 预检：核对 .venv-tau、上游提交、任务集、价格、端点、MCP 九工具、46 缺失订单与费用计划，不发送模型请求
uv run --locked --offline python -m evaluation.run_t16_retail preflight --run-id <run-id>

# 正式付费运行：A/B 全部角色、意图、Composer、User 与 Judge 共用同一累计账本，硬上限 ¥10
uv run --locked --offline python -m evaluation.run_t16_retail run --run-id <run-id>
```

运行产物写入 `data/evaluation/t16-retail/<run-id>/`（已忽略）：按 arm／task／trial 保存对话、事件、完整前后快照、SQLite、MCP 与 worker 日志、原评分分量、项目检查、请求 usage／费用和共享账本；`summary.json` 保留 14 行正式结果与共同有效配对。预检通过后才允许付费，源码或资产变化会使冻结校验拒绝继续；预算不足或环境异常时保留 PARTIAL，不另开账本清零。单个用户轮的识别、角色、回退与 Composer 合计不超过 32 次请求，业务角色温度 0.1、`max_tokens=2048`，User 与 Judge 温度 0.0、`max_tokens=1024`；`seed=300` 未证明传给模型 SDK。

该入口只产生固定 7 任务小样本观察，不构成统计显著性、排行榜或生产模式选择依据；默认零售模式仍是 `single`。价格、密钥与端点从项目配置读取，不写入产物。

## 测试

```powershell
# 单元测试，不访问网络
uv run --locked --offline pytest tests/test_memory.py tests/test_retrieval.py tests/test_agents.py tests/test_monitor_eval.py -q

# 真实 ChromaDB 集成测试，需先启动 chromadb
uv run --locked --offline pytest tests/integration/test_retrieval.py -q

# 真实 Redis／ChromaDB 记忆集成测试，需先启动 redis 与 chromadb
uv run --locked --offline pytest tests/integration/test_memory.py -q

# HTTP API 与 CLI 集成测试，需先启动 redis 与 chromadb，使用 Fake 模型
uv run --locked --offline pytest tests/integration/test_api.py -q

# 零售沙箱单元测试：临时 SQLite 与 Fake 模型，不发真实请求
uv run --locked --offline pytest tests/test_retail_agent.py tests/test_retail_store.py tests/test_retail_ledger.py tests/test_retail_live.py -q

# 零售业务工具的直调／MCP 确定性对照与协议边界，使用真实本地子进程与临时 SQLite
uv run --locked --offline pytest tests/test_retail_mcp.py -q

# 零售沙箱 API／CLI 集成测试，需先启动 redis 与 chromadb，使用 Fake 模型
uv run --locked --offline pytest tests/integration/test_retail_sandbox.py -q

# τ-retail 运行器、桥接与评分适配测试：需先建 .venv-tau，不访问网络
uv run --locked --offline pytest tests/test_run_tau_retail.py tests/test_tau_retail.py -q
uv run --locked --offline pytest tests/test_tau_eval.py -q

# 单／多 Agent 零售对照的权限、记账、配对汇总与真实 MCP 生命周期（Fake 传输层）
uv run --locked --offline pytest tests/test_t16_retail_runner.py -q

# 当前全部测试
uv run --locked --offline pytest tests -q
```

测试模型使用 Fake 客户端。API 集成测试通过 Fake 模型完成两轮对话、知识检索与记忆回读，覆盖全部 13 个路由和 CLI 退出。集成测试连接独立 Redis／ChromaDB，使用随机用户标识和随机测试集合，结束后只删除本次写入的键与记录。

## 配置

复制 `.env.example` 为 `.env` 并填写：

- `ANTHROPIC_API_KEY`：API 密钥
- `ANTHROPIC_BASE_URL`：Anthropic 兼容端点（可选）
- `LLM_MODEL`：端点支持的模型标识
- `ECHOMIND_SKILLS_DIR`：Skill 目录（可选，默认项目内 `skills/`）
- `ECHOMIND_SKILLS_MAX_PROMPT_CHARS`：注入 Agent 的 Skill 内容上限（可选，默认 5000）
- `API_HOST`、`API_PORT`：API 监听地址（默认 `127.0.0.1:8000`）
- `APP_ENV`：设为 `development` 时启用 uvicorn 自动重载（可选）
- `LOG_LEVEL`：日志级别（可选，默认 `INFO`）
- `REDIS_URL`：独立 Redis 连接地址
- `CHROMA_HOST`、`CHROMA_PORT`：独立 ChromaDB 地址
- `CHROMA_PERSIST_DIRECTORY`：ChromaDB 服务不可用时的本地回退目录
- `CHROMA_EMBEDDING_CACHE`：Embedding 模型缓存目录（可选，默认项目内 `data/chroma_cache`）
- `MONITOR_INTERVAL_S`：监控采集间隔秒数（可选，默认 10）
- `MONITOR_WEBHOOK_URL`：告警 webhook 地址（可选，留空不发送）
- `MONITOR_PROMETHEUS_PORT`：Prometheus 暴露端口（可选，留空不启动）
- `EVAL_BASELINE_PATH`：只读评测基线文件（可选）
- `EVAL_REPORT_DIR`：评测报告目录（可选；设置基线时默认基线同级的 `reports/`）

真实模型调用会产生费用；自动化测试只使用 Fake 客户端，不调用真实模型。

## 当前内容

- `api/main.py`：FastAPI 入口、13 个路由、lifespan 组件装配与交互式 CLI。
- `core/llm_utils.py`：从 Anthropic 风格响应中提取文本块。
- `core/intent_recognizer.py`：LLM、Embedding 与关键词三路融合的意图识别，含本地 n-gram 回退与结果缓存。
- `agents/agent_orchestrator.py`：General、Technical、Billing、Escalation 四类角色，意图路由、复合问题协作、工具往返、升级与响应合并。
- `agents/tools.py`：请求检查、排障、账单字段核验、人工交接及共享知识库工具；不伪造订单查询、退款或账单修改。
- `core/skill_loader.py` 与 `skills/`：三个业务 Skill 的加载、按 Agent 匹配、热重载与 `ECHOMIND_SKILLS_DIR` 覆盖。
- `mcp/tool_manager.py`：工具注册、查询改写、并行召回、去重、LLM 重排、TTL 缓存、熔断与降级。降级结果标记 `degraded=True` 且不计为成功命中。
- `mcp/knowledge_base.py`：ChromaDB 文档切片、幂等导入与语义检索；返回文档 ID、标题、内容、相似度与片段序号。
- `agents/retail_agent.py`、`agents/retail_runtime.py`、`agents/retail_tools.py`：单 Agent 零售沙箱的决策循环、运行时装配，以及订单查询、退货申请与知识工具。
- `agents/retail_multi_agent.py`：四角色零售适配与 Host 侧工具权限、共享确认状态和确定性交接。
- `mcp/retail_server.py`、`mcp/retail_client.py`、`mcp/retail_protocol.py`：本地 stdio MCP 工具服务、受信任主程序客户端与请求上下文 `_meta` 校验；项目内 `mcp/` 仍是原工具与存储包，与官方 `mcp` SDK 同名共存。
- `mcp/retail_store.py`：零售沙箱 SQLite 存储、身份绑定、退货确认状态机与事务写入。
- `skills/retail_support/SKILL.md`：零售沙箱每轮生效的业务指引；不与其他模式的审计规则混用。
- `memory/conversation_memory.py`：Redis 工作记忆、会话摘要、ChromaDB 情景记忆与用户画像；压缩与追加按会话加锁。
- `monitor/performance_monitor.py`：Agent／工具统计采样、Z-score 异常检测、阈值告警、路由惩罚反馈、优化建议、Prometheus 与 webhook。
- `evaluation/evaluator.py`：意图 Accuracy／Macro-F1、LLM-as-Judge 对话评分、回归检测、只读基线与独立报告文件。
- `evaluation/retail_metrics.py`：固定六类意图、类型化槽位与证据级检索指标；错误、未知标签、无答案问题按显式合同处理。
- `evaluation/retail_budget.py`：付费请求前的预算上界、原子预留与按 usage／上界结算。
- `evaluation/run_retail_baseline.py`：离线检查、真实向量检索 V 与受控 live 链路的可复现入口。
- `evaluation/validate_retail_data.py`：冻结零售资料的只读一致性校验。
- `compose.yml`：独立的本地 Redis 与 ChromaDB 服务。
- `tests/`：单元测试与真实 Redis／ChromaDB 集成测试。

## 当前限制

- 真实模型表现只覆盖现有回归样本，不代表通用质量或生产可靠性。
- 没有登录与 RBAC；默认客服模式不查询订单或执行退款；零售沙箱只提供模拟订单查询与退货申请，不执行真实支付、退款、邮件或工单。`user_id`／`conv_id` 不是认证身份。
- CORS 允许任意来源，API 默认仅监听 `127.0.0.1`，不适合直接作为生产部署。
- 记忆压缩与追加只在单进程内互斥，不提供分布式锁。
- 监控与评测已接入 API 启动流程；真实 webhook、Prometheus 外部抓取和真实 Judge 评分尚未验证。

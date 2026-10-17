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
- 告警可选通过 webhook 发送，指标可选通过 Prometheus 暴露；自动化测试不会向外部发送 webhook。
- `EndToEndEvaluator` 计算意图识别 Accuracy／Macro-F1，并使用 LLM-as-Judge 评估相关性、准确性、完整性与有用性。
- Judge 解析失败的结果记录 `GRADER_ERROR`、`passed=False` 和空 `scores`；报告保留结果总数并统计 `grader_errors`，平均值只使用有效评分。
- `baseline_path` 只读；每次评测写入新的 `report_<timestamp>.json`，不会覆盖基线或已有报告。
- 工具平均延迟只对实际记录延迟的调用求平均，样本数通过 `latency_samples` 暴露。

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
- `memory/conversation_memory.py`：Redis 工作记忆、会话摘要、ChromaDB 情景记忆与用户画像；压缩与追加按会话加锁。
- `monitor/performance_monitor.py`：Agent／工具统计采样、Z-score 异常检测、阈值告警、路由惩罚反馈、优化建议、Prometheus 与 webhook。
- `evaluation/evaluator.py`：意图 Accuracy／Macro-F1、LLM-as-Judge 对话评分、回归检测、只读基线与独立报告文件。
- `compose.yml`：独立的本地 Redis 与 ChromaDB 服务。
- `tests/`：单元测试与真实 Redis／ChromaDB 集成测试。

## 当前限制

- 真实模型未验证；自动化测试与本地烟测只使用 Fake 客户端或占位密钥，不构成模型质量结论。
- 没有登录、RBAC、支付、订单查询或退款执行；`user_id`／`conv_id` 不是认证身份。
- CORS 允许任意来源，API 默认仅监听 `127.0.0.1`，不适合直接作为生产部署。
- 记忆压缩与追加只在单进程内互斥，不提供分布式锁。
- 监控与评测已接入 API 启动流程；真实 webhook、Prometheus 外部抓取和真实 Judge 评分尚未验证。

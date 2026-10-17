# AgentEcommerce

基于 Anthropic 兼容协议与多路融合的电商客服 Agent 项目。当前提供工程基础、意图识别、多 Agent 编排、Skills、ChromaDB 知识库检索与分层记忆；监控评测及 HTTP/CLI 接口在后续阶段加入。

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

## 分层记忆

- 工作记忆存于 Redis 列表，键为 `wm:{user_id}:{conv_id}`，保留最近消息，TTL 24 小时。
- 会话摘要存于 `summary:{user_id}:{conv_id}`，TTL 24 小时；达到 15 条消息时压缩，只保留最近 5 条。
- 情景记忆与用户画像存于 ChromaDB 的 `episodic`、`user_profile` 集合，支持同一用户跨会话召回，查询按 `user_id` 隔离。
- 摘要生成或情景记忆写入失败时保留原消息，不会静默丢失。
- Embedding 缓存默认位于项目内 `data/chroma_cache`，可用 `CHROMA_EMBEDDING_CACHE` 覆盖。

## 测试

```powershell
# 单元测试，不访问网络
uv run --locked --offline pytest tests/test_memory.py tests/test_retrieval.py tests/test_agents.py -q

# 真实 ChromaDB 集成测试，需先启动 chromadb
uv run --locked --offline pytest tests/integration/test_retrieval.py -q

# 真实 Redis／ChromaDB 记忆集成测试，需先启动 redis 与 chromadb
uv run --locked --offline pytest tests/integration/test_memory.py -q

# 当前全部测试
uv run --locked --offline pytest tests -q
```

测试模型使用 Fake 客户端。集成测试连接独立 Redis／ChromaDB，使用随机用户标识和随机测试集合，结束后只删除本次写入的键与记录。

## 配置

复制 `.env.example` 为 `.env` 并填写：

- `ANTHROPIC_API_KEY`：API 密钥
- `ANTHROPIC_BASE_URL`：Anthropic 兼容端点（可选）
- `LLM_MODEL`：端点支持的模型标识
- `ECHOMIND_SKILLS_DIR`：Skill 目录（可选，默认项目内 `skills/`）
- `REDIS_URL`：独立 Redis 连接地址
- `CHROMA_HOST`、`CHROMA_PORT`：独立 ChromaDB 地址
- `CHROMA_PERSIST_DIRECTORY`：ChromaDB 服务不可用时的本地回退目录
- `CHROMA_EMBEDDING_CACHE`：Embedding 模型缓存目录（可选，默认项目内 `data/chroma_cache`）

真实模型调用会产生费用，自动化测试不读取这些配置。

## 当前内容

- `core/llm_utils.py`：从 Anthropic 风格响应中提取文本块。
- `core/intent_recognizer.py`：LLM、Embedding 与关键词三路融合的意图识别，含本地 n-gram 回退与结果缓存。
- `agents/agent_orchestrator.py`：General、Technical、Billing、Escalation 四类角色，意图路由、复合问题协作、工具往返、升级与响应合并。
- `agents/tools.py`：请求检查、排障、账单字段核验、人工交接及共享知识库工具；不伪造订单查询、退款或账单修改。
- `core/skill_loader.py` 与 `skills/`：三个业务 Skill 的加载、按 Agent 匹配、热重载与 `ECHOMIND_SKILLS_DIR` 覆盖。
- `mcp/tool_manager.py`：工具注册、查询改写、并行召回、去重、LLM 重排、TTL 缓存、熔断与降级。降级结果标记 `degraded=True` 且不计为成功命中。
- `mcp/knowledge_base.py`：ChromaDB 文档切片、幂等导入与语义检索；返回文档 ID、标题、内容、相似度与片段序号。
- `memory/conversation_memory.py`：Redis 工作记忆、会话摘要、ChromaDB 情景记忆与用户画像；压缩与追加按会话加锁。
- `compose.yml`：独立的本地 Redis 与 ChromaDB 服务。
- `tests/`：单元测试与真实 Redis／ChromaDB 集成测试。

## 当前限制

- 尚未提供 HTTP API、监控与评测。
- 真实模型未验证；自动化测试只使用 Fake 客户端生成摘要与画像。
- 记忆压缩与追加只在单进程内互斥，不提供分布式锁。

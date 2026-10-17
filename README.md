# AgentEcommerce

基于 Anthropic 兼容协议与多路融合的电商客服 Agent 项目。当前阶段提供工程基础与意图识别核心；Agent 编排、知识库、分层记忆、监控评测及 HTTP/CLI 接口在后续阶段加入。

## 环境要求

- Windows
- Python 3.12
- [uv](https://docs.astral.sh/uv/)

首轮运行使用本机 Redis 与 ChromaDB（后续阶段接入）。

## 安装与测试

```powershell
uv python find 3.12
uv lock
uv sync --locked --group dev
uv run --locked --offline pytest tests -q
```

测试不访问网络，使用 Fake 模型客户端。

## 配置

复制 `.env.example` 为 `.env` 并填写：

- `ANTHROPIC_API_KEY`：API 密钥
- `ANTHROPIC_BASE_URL`：Anthropic 兼容端点（可选）
- `LLM_MODEL`：端点支持的模型标识
- `ECHOMIND_SKILLS_DIR`：Skill 目录（可选，默认项目内 `skills/`）

真实模型调用会产生费用，自动化测试不读取这些配置。

## 当前内容

- `core/llm_utils.py`：从 Anthropic 风格响应中提取文本块。
- `core/intent_recognizer.py`：LLM、Embedding 与关键词三路融合的意图识别，含本地 n-gram 回退与结果缓存。
- `agents/agent_orchestrator.py`：General、Technical、Billing、Escalation 四类角色，意图路由、复合问题协作、工具往返、升级与响应合并。
- `agents/tools.py`：请求检查、排障、账单字段核验、人工交接及共享知识库工具；不伪造订单查询、退款或账单修改。
- `core/skill_loader.py` 与 `skills/`：三个业务 Skill 的加载、按 Agent 匹配、热重载与 `ECHOMIND_SKILLS_DIR` 覆盖。
- `tests/test_core.py`、`tests/test_agents.py`、`tests/test_skills.py`：使用 Fake 模型的对应测试。

## 当前限制

- 尚未提供 HTTP API、知识库、分层记忆、监控与评测。
- 本阶段未验证真实模型、Redis 或 ChromaDB 连接。

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
uv run --locked --offline pytest tests/test_core.py -q
```

测试不访问网络，使用 Fake 模型客户端。

## 配置

复制 `.env.example` 为 `.env` 并填写：

- `ANTHROPIC_API_KEY`：API 密钥
- `ANTHROPIC_BASE_URL`：Anthropic 兼容端点（可选）
- `LLM_MODEL`：端点支持的模型标识

真实模型调用会产生费用，自动化测试不读取这些配置。

## 当前内容

- `core/llm_utils.py`：从 Anthropic 风格响应中提取文本块。
- `core/intent_recognizer.py`：LLM、Embedding 与关键词三路融合的意图识别，含本地 n-gram 回退与结果缓存。
- `tests/test_core.py`：意图投票、混合文本块解析与 Fake 客户端识别测试。

## 当前限制

- 尚未提供 HTTP API、工具执行、知识库、记忆、监控与评测。
- 本阶段未验证真实模型、Redis 或 ChromaDB 连接。

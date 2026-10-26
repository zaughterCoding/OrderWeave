# 第三方资源

本仓库代码使用[MIT许可证](LICENSE)。依赖库、独立下载的模型和数据保留各自的许可证，根许可证不替代它们。

| 资源 | 用途 | 获取与许可证 |
| --- | --- | --- |
| τ²-bench retail | 可选的外部虚构订单种子、政策及基准环境 | [固定上游提交](https://github.com/sierra-research/tau2-bench/tree/fc0055dc4e0a316c3f83133267fbd6faaa770992)，MIT；准备工具会同时下载上游LICENSE |
| Qwen3-Embedding-0.6B | 可选本地向量编码 | [官方模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)，Apache-2.0 |
| Qwen3-Reranker-0.6B | 可选本地相关性重排 | [官方模型卡](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)，Apache-2.0 |
| Python依赖 | API、工作图、MCP、存储及监控 | 依赖与版本见pyproject.toml和uv.lock，许可见各依赖发行包 |

本仓库不分发上述模型权重、订单数据、知识资料、评测数据或运行轨迹。使用者自行获取资源，并遵守对应条款。外部基准和虚构订单不代表真实商城、支付、邮件或人工服务。

# 零售沙箱公开种子

本目录是 AgentEcommerce 零售模式的固定公开运行资料，不依赖 `docs/`：

| 文件 | 内容 |
| --- | --- |
| `db.json` | τ²-bench retail v1.0.1 完整公开测试库：500 用户、1000 订单、50 商品 |
| `policy.md` | 上游固定交易政策原文（英文） |
| `LICENSE` | 上游 MIT 许可证原文 |
| `knowledge.jsonl` | 本项目冻结的 29 段中文政策与准备建议证据 |
| `sources.json` | 来源 URL、版本、提交、大小与 SHA-256 |

上游版本：`sierra-research/tau2-bench` v1.0.1，commit
`fc0055dc4e0a316c3f83133267fbd6faaa770992`。`db.json` 的 SHA-256 为
`413a65160adbdb5fde0ffc0015c49b6d70250b10c18128de169b597af7766765`。

## 初始化

零售模式启动时会校验种子哈希，然后在**新库或空库**中导入完整记录。已有业务库
不会被自动重置；元数据或内容与种子不一致时服务显式停止。默认数据库路径为
`data/retail/retail.db`，可通过 `RETAIL_DB_PATH` 指定。每个验收案例使用独立
数据库文件。

## 知识集合

零售知识集合只导入 `knowledge.jsonl` 的 29 段证据，保留 `evidence_id`、
`source_refs`、`policy_release` 和 `policy_commit`。源码默认集合和其中 7 天退款
演示政策不会进入零售集合。

## 边界

这里只包含公开数据库、固定政策和项目自编中文证据；不包含官方隐藏任务、gold、
用户模拟器、Judge 或评分脚本。交易政策以 `policy.md` 为准，中文证据不新增期限、
运费、原包装或到账承诺。

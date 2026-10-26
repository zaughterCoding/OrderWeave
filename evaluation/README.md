# 评测方法

本目录仅发布复用组件、离线汇总入口和[结果摘要](RESULTS.md)。不发布数据集、运行资料、历史冻结计划、逐条预测、用户对话、原始请求、账本或个人开发报告。

## 可以直接做什么

```powershell
# 公开、无数据、零模型的单元与契约检查
uv run --locked python -m pytest tests -q

# 对自行准备的私有记录汇总，不发送模型请求
uv run python -m evaluation.score_results --kind intent --input data/my-run/intents.jsonl --planned 120
uv run python -m evaluation.score_results --kind route --input data/my-run/routes.jsonl --planned 143
uv run python -m evaluation.score_results --kind recovery --input data/my-run/recovery.jsonl --planned 10
uv run python -m evaluation.score_results --kind task --input data/my-run/tasks.jsonl --planned 39
```

这些分母只是历史口径示例。评测自己的数据时使用实际预先冻结的分母；不能为了得到相同数字而删除失败。一个计划槽位必须有一行，id唯一。缺行、多行或重复id会被拒绝。

## 输入记录

JSONL每行一个对象。下面是字段规范，不附完整样本或gold。

| 类型 | 必需字段 | 规则 |
| --- | --- | --- |
| intent | `id`、`gold`、`prediction` | gold为order_status／refund／query／human_handoff／greeting／other；错误可用`prediction: null, is_error: true`，仍占分母 |
| route | `id`、`gold`、`prediction` | gold为general／technical／billing／escalation；错误、未知预测和空预测按错计 |
| recovery | `id`、`passed` | passed为布尔值；一行对应一个预先定义的工程场景，不是一个断言或一次重启 |
| task | `id`、`status`、`success` | status为valid／invalid／not_run；success为布尔值，无效和未运行不能成功 |

意图汇总复用`retail_metrics.score_intents`，按固定六类计算Macro-F1，错误／未知输出不映射成正确的other。角色汇总给出固定分母准确率。任务给出S、M、计划N以及S/M、M/N、S/N；M=0时S/M为null。汇总器信任输入中的观察结果，不执行工具、不核验事务、不调用官方Judge，不能凭手写passed=true证明系统通过。

## 建议的执行流程

1. 在本地冻结数据版本、case id、gold、历史／状态夹具、可用范围与排除理由；数据保留在Git之外。
2. 冻结代码、模型、Prompt、政策、温度、thinking、token上限及计分口径；记录哈希。真实调用前明确总预算、当前费率和凭证来源，所有阶段共享账本。
3. 意图／角色评测调用真实共享分类与路由链路，核对历史是否实际进入请求；规则、本地和LLM来源分别记录，不能把无调用分支计成模型成功。
4. 事务恢复检查使用独立SQLite／checkpoint和真实stdio MCP，可用Fake模型；验证最终状态、写入数量、防重、确认与未知结果。按预定义场景计数。
5. 任务评测每会话独立数据库和checkpoint；若每题重复三次，预先锁定题目与顺序。保留失败、无效、未运行及中断前缀，合并时按槽位唯一，不挑最佳结果。
6. 输出只发布汇总结果及限制。原始记录、数据库、私有数据与费用凭证继续留本地。

## τ-retail与官方组件

历史任务来源为[τ²-bench固定提交](https://github.com/sierra-research/tau2-bench/tree/fc0055dc4e0a316c3f83133267fbd6faaa770992)，只核定能力范围内的13道零售题，不是全量τ成绩。若开展新实验，请自行获取官方任务和许可证，依据上游说明准备独立环境，不能用仓库历史数字代替新执行。

保留的`tau_bridge.py`／`tau_user_worker.py`提供独立模拟用户进程接口；`tau_eval_bridge.py`／`tau_eval_worker.py`提供官方DB／NL组件适配。它们需要调用方明确提供本地数据目录、任务、快照、轨迹与模型参数；不是附带历史计划的一键运行器。组件参数可用各worker的`--help`查看。官方DB、NL、联合reward和项目工程安全检查必须分列，模拟人工交接不等于真人接入。

## 历史结果的可复现边界

仓库提供方法、组件和汇总公式，但不分发自建回归数据、知识资料、分类参考工件、官方任务副本或原始运行证据。因此不能仅克隆仓库就复现0.9917或92.31%。维护者的历史完整离线检查与当前公开测试子集也不是同一个分母。新数据上的泛化和最新修复后的真实效果需要重新测量。

真实模型与Judge调用可能产生费用；本文件及默认测试不会自动启动付费评测。外部订单准备见[中文README](../README.md)，它只下载明确指定的运行资产，不下载评测题目。

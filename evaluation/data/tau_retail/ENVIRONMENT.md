# τ-retail 固定子集评测环境

本目录保存固定上游提交的原版任务资产、公开种子、政策、模拟用户指南与许可，
以及独立评测环境的精确依赖记录。正式运行不依赖被忽略的 `docs/`。

## 上游固定版本

- 仓库：`sierra-research/tau2-bench`
- 版本：`v1.0.1`
- 提交：`fc0055dc4e0a316c3f83133267fbd6faaa770992`
- 安装方式：从该提交安装 `tau2`，不安装 voice、knowledge、gym、experiments extras。

## 独立环境

主项目环境与 `uv.lock` 不参与本评测。独立环境建在项目目录内：

```powershell
uv venv --python 3.12 .venv-tau
uv pip install --python .venv-tau/Scripts/python.exe "tau2 @ git+https://github.com/sierra-research/tau2-bench@fc0055dc4e0a316c3f83133267fbd6faaa770992"
```

记录结果：

- Python 3.12.14，解释器 `.venv-tau/Scripts/python.exe`
- 精确依赖：`requirements-freeze.txt`（`uv pip freeze` 原样输出）
- 该文件 SHA-256：`fa09ff058d86bea69f73d535728dbc738a182e30ec7ee725da5e6ddcd1db17df`
- 上游包版本：`tau2 1.0.1`（源码来自上述提交）
- 许可：见同目录 `LICENSE`

## 数据目录

运行时把 `TAU2_DATA_DIR` 指向本目录的 `upstream_data`：

```powershell
$env:TAU2_DATA_DIR = "evaluation/data/tau_retail/upstream_data"
```

`upstream_data/tau2/domains/retail/` 下为原版 `tasks.json`、`db.json`、`policy.md`；
`upstream_data/tau2/user_simulator/simulation_guidelines.md` 为原版用户模拟指南。
这些文件来自上述固定提交，按原字节保存。

## 离线检查

```powershell
$env:TAU2_DATA_DIR = (Resolve-Path "evaluation/data/tau_retail/upstream_data").Path
.venv-tau/Scripts/python.exe -c "from tau2.user.user_simulator import UserSimulator; from tau2.domains.retail.data_model import RetailDB; from tau2.evaluator.evaluator_env import EnvironmentEvaluator; from tau2.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator; print('ok')"
```

## 复现限制

- `seed=300` 只记录在运行配置与协议日志中；`to_anthropic_kwargs` 不向 SDK 透传 `seed`，Agent 请求也不含该字段，因此不能声称模型生成受固定种子控制或结果可确定复现。
- DB 分量来自每案实际完整 SQLite 业务快照，经官方 `RetailDB` 模型和官方 hash 比较；预测侧未执行官方原生 strict replay，只把实际快照作为状态来源。该适配的评分可信度来自快照与官方组件，不等同于原生轨迹回放。

## 结果边界

结果只代表“τ³-bench v1.0.1 retail 固定 8 任务／项目适配环境／DeepSeek 用户与评分配置”，
不是全量排行榜、原生官方运行环境成绩、独立盲测或生产可用性证明。

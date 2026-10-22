# 交接当前状态与编号类型修复

交接输入保留历史 `work_record` 及其原轮次，新增 `support.current_state` 表示当前服务端视图。用户要求人工且权威 pending 已清除时，当前阶段为 `handoff_requested`，不再保留“等待确认”指示；`RoleHandoff.task.phase`、交接摘要和当前未解决事项使用同一视图。`record_turn` 与 `view_turn` 区分历史记录和当前视图轮次。

`handoff_requested` 表示用户请求，不能证明工具成功。实际模拟交接成功仍依据工具结果、业务事件及持久工作记录的 `handoff_recorded` 判断；失败不得提前声称已记录。显式空 pending 不会回退到包内旧 pending，已有待确认业务记录时仍保持 confirmation。视图不会替代身份、真实确认、版本及事务核验，也不改写历史记录。

范围识别按出现位置区分编号：`item_id`、商品编号、商品条目编号、款式号等条目引用不当作产品编号；退货／订单语境中的“（十位编号）一件”按条目简写处理。明确 `product_id`／产品编号优先，同一数字同时被声明为产品和条目时，产品声明仍独立核验。无标签的未知十位数字仍保持原保守拒绝；仅有 item_id 不足以确立支持产品。编号分类不建立交易权限，商品归属继续由订单工具核验。

复现回归（需要隔离本地 Chroma 和现有轻量模型缓存，生成全部 Fake）：

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
uv run --locked --offline python -m pytest tests/test_retail_support_fixes.py tests/integration/test_retail_support_fixes.py -q
```

修复仅涉及 `agents/retail_support.py`、`agents/retail_handoff.py`，共享 API／CLI、single／multi 和 graph／legacy 路径。不改 Prompt、RAG、政策或依赖，不新增模型调用。运行入口和 legacy 回退见 [工作流说明](RETAIL_WORKFLOW.md)。本次没有新增付费复验，不重跑历史真实验收；已有付费目录的冻结清单与修后源码不同是预期行为，禁止绕过冻结或重置账本。更广泛实体识别、未见语义泛化和真实模型表现不由这些 Fake 检查证明。

# 中文 Dense／BM25／Hybrid 消融

本对照沿用本项目Qwen3-Embedding-0.6B、35段中文正文和U04的60条已见查询。原48题的dev／test各20条可答、另外8条诊断不进入常规分母；支持烟测12条为项目自编。旧正文、gold、切分及历史结果不改。语料来源与许可沿用[原资料](../retail/README.md)及[支持资料](../retail_support/README.md)。

## 三路与分词

- Dense：现有固定Qwen、1024维、CPU float32／4线程／batch1／最大512 token，查询加固定指令，正文不加，L2向量与距离。
- BM25：标准库实现，k1=1.2、b=0.75，文档频率和平均长度来自当前完整语料。NFKC和大小写规范化，连续中文字符二元组、单字片段、英文词、数字和连字符标识符；不是词典分词。索引标题、正文及evidence_id／product_id／step_id；`Wireless Earbuds`与`无线耳机`、`bluetooth`与`蓝牙`共享规范化表示。保留PAIR-01等标识符，不新增业务同义政策。零词法命中返回空列表。
- Hybrid：两路各最多20候选，按稳定evidence_id去重，等权RRF常数60，最终top5；同分按稳定ID排序。不混合Dense与BM25的原始分数，RRF分数不是概率。[RRF公式](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)、[BM25参数](https://www.elastic.co/docs/reference/elasticsearch/index-settings/similarity)。无需Elasticsearch或新依赖。

冻结一个候选，不按结果调整别名、窗口或权重。三个模式在相同正文与原始查询上取top5，关闭查询改写和LLM重排以归因纯召回；生产原改写／重排保持。两者的结果不能等同于最终回答质量。

## 范围对照

`scope_queries.jsonl`为18条助手自编范围烟测，显式保存允许ID、Host支持packet、类别与gold。未做独立人工双标，不能视为盲测。PAIR步骤的短续接问题由fixture显式提供已确认的商品／问题上下文，不冒充只看短问题就能确认商品。

`scope_decoys.jsonl`是8条**合成测试干扰**，复用原文但分别改错商品、问题、支持版本、来源类型、scope、语言、政策版本或政策提交号；不是另一商店或厂商的真实材料，不进入生产语料。范围阶段在原35段加8条干扰的43段集合单独做过滤前后对照。前后使用相同语料，避免把改变库大小当过滤收益。

过滤使用原始用户消息与Host支持状态，不能被模型生成的查询放宽。全局政策保留，明确类别只保留对应准备指南；多类别取并集，未知类别保留通用指南。商品支持还须匹配Host确认的商品／问题及固定支持版本、来源和scope；通用证据核对固定政策版本／提交及来源。两个通道均先限制允许ID再取候选，合并后再复核。带范围的改写召回不读写结果缓存，防止不同范围串用。此范围不是账户认证或退货授权。

## 复现与指标

先按[Qwen准备说明](../embedding_comparison/README.md#复现windows项目根目录)准备主环境、现有独立推理环境及模型，然后启动本项目服务；缓存完整时不下载，不调用付费API。

```powershell
docker compose -f compose.yml up -d redis chromadb
uv run --locked --offline python -m evaluation.run_hybrid_retrieval --output data/evaluation/u05/my-run-001
```

输出目录须全新或为空，不覆盖已有证据。`frozen.json`保存输入、实现与参数哈希，三个源码快照保留实际执行版本；`quality.json`、`scope_unfiltered.json`、`scope_filtered.json`包含逐题三轮排序、AEH／Recall／MRR、缺失必需证据组、错误范围命中和延迟。成功才标`EVALUATION_COMPLETE`，失败标`RUN_ERROR`；仅删除本次新建集合，不触碰旧集合。服务只停止本次启动的部分，卷保留。

三路串行、轮转执行顺序；第一轮单列，后两轮统计nearest-rank P95。查询计时包含本地Qwen编码、进程通信、Chroma过滤／查询与词法／融合，排除模型加载、建索引、改写、重排和生成。宿主负载未隔离，样本有限，不代表生产P95。模型加载和向量建索引、词法快照分别计时。

预设默认采用Hybrid的条件：原dev／test／support各切片AEH@5与Recall@5均不下降，至少一项提高；过滤后无错误范围命中、无gold过滤损失、协议正常；暖P95≤Dense×1.25+10ms。达不到则默认Dense，保留显式BM25／Hybrid供对照；适用过滤独立采用。

2026-10-21本机固定对照结果（已见合成资料，非泛化结论）：

| 模式 | dev AEH@5／Recall@5 | test AEH@5／Recall@5 | support AEH@5／Recall@5 | 暖P95 |
| --- | --- | --- | --- | --- |
| Dense | 100%／100% | 90%／90% | 91.67%／95.83% | 456.51ms |
| BM25 | 95%／95% | 90%／87.5% | 91.67%／95.83% | 0.16ms |
| Hybrid | 100%／97.5% | 95%／92.5% | 91.67%／91.67% | 345.80ms |

表中为最终v2。v1与v2全部逐题三轮排序一致；v1 Dense／Hybrid的暖P95分别为324.91／381.42ms，时延受宿主负载影响明显，不能据此宣称Hybrid更快。v2前只修生产入口的候选预算兼容，未调整候选模型、分词、权重、窗口或gold；v1完整保留。

Hybrid虽改善test，未通过各切片Recall不下降条件，**默认保留Dense并采用范围过滤**。原52条可答题的必需证据漏召回：Dense3条、BM254条、Hybrid2条；仍不是回答成功率。18条范围烟测中，Dense错误范围槽位36→0、AEH@5为13/18→17/18；Hybrid为29→0、17/18→18/18。范围集含合成干扰，不能推广为真实业务错误率；Recall与AEH分母不同，等价必需组全命中也可能未召回全部相关段。

生产显式设置`RETAIL_RETRIEVAL_MODE=dense|bm25|hybrid`；默认Dense。会话工具和独立`/search`均前置过滤；普通会话候选5条、支持会话20条，独立搜索保留原请求上限与候选预算。类别识别采用固定关键词，未识别时不擅自缩小指南范围，不声称覆盖完整商品目录。冻结语料由运行时重建词法快照，不复用旧词法缓存，也不重建或改写旧向量。只有零售模式启用此配置；原客服、记忆和意图不变。未验证真实生成质量、GPU、并发与大语料泛化。

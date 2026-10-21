# 中文 Embedding 对照

此实验比较 ONNX `all-MiniLM-L6-v2` 与 `Qwen/Qwen3-Embedding-0.6B` 的纯向量召回。中文为主要目标；英文公开交易任务不提供本实验的 RAG 质量证据。对照脚本不会切换业务配置或写入已有集合。后续零售 RAG 已接入 Qwen，见[运行说明](../../../../README.md#零售-qwen-检索)；原实验的资源门槛和结论仍保留为历史对照。

## 固定资料与指标

- 共同语料为 `evaluation/data/retail_sandbox/knowledge.jsonl` 的29段中文知识和 `evaluation/data/retail_support/knowledge.jsonl` 的6段支持资料，正文和证据ID不改。
- 原 `evaluation/data/retail/retrieval.jsonl` 的48条中文题沿用dev／test，每侧20条可答、2条需业务工具、2条无具体答案。这是已见固定集，不是独立盲测。
- 本目录 `support_queries.jsonl` 为12条助手自编中文支持烟测，覆盖范围、供电、连接占用、重配对、反馈、交接与退货边界。全部标为 `support_smoke`，不用于训练或独立泛化声明；标注不是双人独立复核结果。
- 保留原 qrels≥2、等价证据组和多组全部命中的计分定义。常规分母是52条可答题，另外8条仅记录召回，不能判作回答成功。输出各切片 Recall、MRR、Hit、AllEvidenceHit 的 @1／3／5。

上游派生材料的来源和许可沿用 [retail 资料说明](../retail/README.md)及[MIT声明](../retail/source/LICENSE)；耳机指南为项目自编沙箱说明，不是厂商手册。不复制订单库或隐藏任务答案进入RAG语料。

## 模型与隔离

Qwen固定提交 `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`，官方来源与Apache-2.0声明见[模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)。准备入口只下载固定5个文件，逐一核对内置大小与SHA-256，生成 `model-manifest.json`；损坏或未知文件不会被覆盖。默认评测不联网下载、不执行远程模型代码。

MiniLM沿用本项目已缓存ONNX、384维、均值池化与原CPU线程配置；Qwen使用CPU float32、4线程、batch1、1024维、最后token池化、L2归一化、最大512 token。Qwen只在查询前加固定英文任务指令，中文问题与文档保持原文。每次实际token长度与截断名单进入产物，不搜索指令、维度或精度。

Chroma0.5.23与新版Transformers的tokenizer版本要求冲突，因此Qwen在独立uv／Python3.12环境编码，主环境和依赖保持不变。主进程将两者向量分别导入同一Chroma服务的新集合，以相同L2／top-k查询；各集合绑定完整模型配置指纹与语料哈希，结束只删除本次创建的集合。

## 复现（Windows，项目根目录）

先完成主项目安装及MiniLM缓存准备，再创建可选的独立推理环境。此步骤需网络和约1.2GB模型权重空间，另有依赖与缓存占用；不调用付费API。

```powershell
uv venv data/embedding-env --python .venv/Scripts/python.exe
uv pip sync --python data/embedding-env/Scripts/python.exe --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match evaluation/embedding_requirements.lock
uv run --locked --offline python -m evaluation.run_embedding_comparison --prepare-model
docker compose -f compose.yml up -d chromadb
uv run --locked --offline python -m evaluation.run_embedding_comparison --output data/evaluation/u04/local-cpu-001
```

已有独立环境可跳过创建，缓存完整时准备入口只校验而不下载。输出目录必须全新或为空，避免覆盖既有证据；再次运行请换目录。`--model-path`、`--qwen-python`、`--chroma-host`、`--chroma-port`可显式指定已有本地资源。结束后仅停止本次启动的容器，保留卷。

`run.json` 是本次状态；失败写 `RUN_ERROR` 并返回非零，完成才写 `EVALUATION_COMPLETE`。`frozen.json` 固定输入／代码／依赖与采用门槛；逐模型编码、向量及检索结果、summary和集合列表独立保留。脚本、模型、数据变化期间不能形成完整冻结对照。

## 结果边界

同一模型先编码35段正文，再对60条查询各运行3轮，第一轮单列、后两轮120样本统计暖延迟。延迟是编码与Chroma查询分阶段合计，包含不同实现的原配置，不含进程通信、生成、完整业务链或生产并发；不宣称等线程算力比较。Windows累计峰值私有提交与RSS分别记录，不能互换，模型加载含预热与哈希检查。

预先采用条件为dev AllEvidenceHit@5至少提高0.10、Recall@5严格提高，已见test及支持烟测不下降、协议正常，Qwen暖P95≤1000ms且峰值私有提交≤3072MiB。任一不满足则保留MiniLM；即使候选达标，依赖／生命周期接入未验证也不直接切换业务默认。

全部35段公开正文另报告必要证据覆盖上界与字符数；真实全上下文回答质量对照未运行，不能把上界当生成正确率。本实验不覆盖真实回答忠实性、独立泛化、GPU／半精度、跨语言、大语料或生产接入。

离线脚本回归无需Qwen依赖或权重：

```powershell
uv run --locked --offline python -m pytest tests/test_embedding_comparison.py -q
```

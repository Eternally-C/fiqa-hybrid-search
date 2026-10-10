# Hybrid Semantic Search and Neural Reranking for Financial Question Answering

[English](README.md) | [中文](README_zh.md)

基于 BEIR FiQA-2018 的金融文档检索项目，对比 BM25、Dense 语义检索、加权倒数排名融合（Weighted RRF）和神经重排。模型仅用于推理，未进行微调。

## 实验结果

### 最终 Test：648 条查询

Weighted Hybrid 在最终 Test 评估前，依据 Validation 结果选定。

| 方法 | Recall@10 | Recall@100 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.278013 | 0.473669 | 0.270286 | 0.216684 |
| Dense | 0.431690 | 0.688115 | 0.435371 | 0.363460 |
| **Weighted Hybrid** | 0.439712 | 0.688115 | 0.440430 | **0.369861** |

![最终 Test 检索指标](results/figures/test_retrieval_metrics.png)

Hybrid 的 Test nDCG@10 比 Dense 高 **0.006401**，相对提升约 **1.76%**。Recall@10 和 MRR@10 同时提高，Recall@100 保持一致。

数据来源：[Test 指标](results/test_baselines.csv)。

### Validation：500 条查询

| 方法 | Recall@10 | Recall@100 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.271037 | 0.468211 | 0.287029 | 0.228423 |
| Dense | 0.444063 | 0.691544 | 0.451139 | 0.379798 |
| **Weighted Hybrid** | 0.451300 | 0.691544 | 0.465729 | **0.390140** |
| Weighted Hybrid + Cross-Encoder | 0.455475 | 0.691544 | 0.456533 | 0.383587 |
| Dense + Cross-Encoder | 0.455475 | 0.691544 | 0.456533 | 0.383587 |
| Weighted Hybrid + Passage Cross-Encoder | 0.447088 | 0.691544 | 0.457879 | 0.380282 |

![Validation nDCG@10 对比](results/figures/validation_ndcg_comparison.png)

在已评估的配置中，Weighted Hybrid 的 Validation nDCG@10 最高。整篇重排和分段重排均未提高这一指标。

数据来源：[检索基线](results/validation_baselines.csv)、[Hybrid 重排](results/reranker_validation.csv)、[Dense 重排](results/dense_reranker_validation.csv)和[分段重排](results/passage_reranker_validation.csv)。

## 数据集与评估

[FiQA-2018](https://huggingface.co/datasets/BeIR/fiqa) 来自 [BEIR benchmark](https://github.com/beir-cellar/beir)，包含 57,638 篇文档和 6,648 条查询。检索使用文档正文。

| 数据划分 | 查询数 | 用途 |
|---|---:|---|
| Train | 5,500 | 未使用 |
| Validation | 500 | 参数选择、方法比较和错误分析 |
| Test | 648 | 固定方法与参数后的最终评估 |

指标使用原始二值相关性标注（qrels），逐查询计算后取平均。模型选择以 **Validation nDCG@10** 为依据。开发脚本只加载 Validation qrels；Test 评估使用独立入口，不参与调参。

## 方法

| 方法 | 配置 |
|---|---|
| BM25 | BM25Okapi、文本转小写、正则表达式分词 |
| Dense | [multi-qa-MiniLM-L6-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1)、归一化的 384 维向量、内积评分 |
| Weighted Hybrid | Weighted RRF：`rrf_k=60`，每路 100 个候选，BM25 权重 0.10，Dense 权重 1.00 |
| 整篇重排 | [ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) Cross-Encoder 对 Top-100 候选重排 |
| 分段重排 | 使用同一 Cross-Encoder，文档窗口 400 tokens，重叠 80 tokens，以最高段分作为文档分数 |

BM25 权重 0.10 在[已记录的 11 个 Validation 候选权重](results/hybrid_weight_search.csv)中选定。Cross-Encoder 的输入上限为 512 tokens，包括查询、文档或分段及特殊 token；分段评分会检查每个输入对是否超限。

## 错误分析

全部 500 条 Validation 查询中，相对于 Weighted Hybrid 的逐查询 nDCG@10 变化：

| 重排方法 | 改善 | 恶化 | 不变 | 平均变化 |
|---|---:|---:|---:|---:|
| 整篇重排 | 122 | 144 | 234 | -0.006553 |
| 分段重排 | 123 | 142 | 235 | -0.009858 |

**长文档案例——Query 7681：** 关于公司合并后成本基础的计算问题。相关文档 `307602` 对应的模型输入超过了 512-token 上限。

| 方法 | 首个相关文档排名 | Recall@10 | nDCG@10 |
|---|---:|---:|---:|
| Weighted Hybrid | 1 | 1.0 | 0.919721 |
| 整篇重排 | 7 | 0.5 | 0.204382 |
| 分段重排 | 1 | 0.5 | 0.613147 |

分段评分部分恢复了该案例的表现，但仍低于 Hybrid，也未提高整体 Validation 表现。尚未确认截断是排序退化的唯一原因。

完整案例与分数见 [Validation 实验记录（整理于最终 Test 之前）](results/validation_experiments.md)。逐查询数据：[整篇重排](results/reranking_error_analysis.csv)和[分段重排](results/passage_reranking_error_analysis.csv)。

## 环境与安装

已验证的环境为 macOS arm64、Python 3.11.17、PyTorch 2.14.1 和 Sentence Transformers 6.1.0，实验使用 Apple M5 Pro（24 GB RAM）的 MPS。MPS 不可用时，代码使用 CPU。

[requirements.txt](requirements.txt) 固定直接依赖和部分模型相关依赖版本，并非完整环境锁文件。[Test 评估时的环境记录](results/final_evaluation_environment.txt)生成于添加 Matplotlib 之前。

以下命令适用于 macOS 14 或更新版本及原生 arm64 Conda，请在项目根目录执行。

**下载提示：** 安装 Python 和依赖预计需要下载 **300–600 MB**；实际大小取决于依赖解析和已有缓存。模型权重与 FiQA 数据将在首次使用时另外下载。

```bash
conda create -n fiqa_reproduction python=3.11.17 pip
conda activate fiqa_reproduction
python -m pip install -r requirements.txt
```

检查依赖兼容性和 MPS 可用性：

```bash
python -m pip check
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

`pip check` 应输出 `No broken requirements found.`。原工作环境已通过此检查；独立全新环境的安装尚未验证。

## 使用方法

所有命令均在项目根目录执行。Validation 命令会覆盖 `results/` 下对应的 CSV，建议使用独立项目副本复现实验，以保留已有结果。

**首次运行下载：** FiQA 数据约 **30 MB**；用于语义检索的 Dense 模型权重约 **91 MB**。重排还需要下载约 **91 MB** 的 Cross-Encoder 权重。Tokenizer 和配置文件会增加少量下载；已有缓存可减少下载量。

评估三种检索方法的 Validation 表现：

```bash
python main.py
```

可选的 Validation 实验：

```bash
# 搜索 Hybrid 权重
python -m scripts.search_hybrid_weights

# 评估整篇重排和分段重排
python -m scripts.evaluate_reranker --reranker whole
python -m scripts.evaluate_dense_reranker
python -m scripts.evaluate_reranker --reranker passage

# 生成逐查询错误分析
python -m scripts.analyze_reranking --reranker whole
python -m scripts.analyze_reranking --reranker passage
```

根据已保存的 CSV 重新生成图表。以下命令不加载模型或 qrels，会替换对应的 PNG/SVG 图：

```bash
python -m scripts.plot_validation_results
python -m scripts.plot_test_results
```

项目已包含最终 Test 结果。`scripts/evaluate_test.py` 需要显式提供 `--run-test`，并拒绝覆盖已有的 `results/test_baselines.csv`。固定配置见[最终评估方案](results/final_evaluation_plan.md)。

## 项目结构

```text
fiqa_hybrid_search/
├── main.py              # Validation 检索基线入口
├── src/                 # 检索、重排、数据加载与指标
├── scripts/             # 实验、案例检查与绘图
├── results/             # CSV、图表与实验记录
├── requirements.txt
├── README.md
├── README_zh.md
└── .gitignore
```

## 可复现性与局限

- Dense 索引在本地生成 `data/cache/corpus_embeddings.pt`（约 89 MB）。复用前检查模型与语料元数据；缓存不使用 qrels，也不纳入版本控制。
- 已记录模型名称和依赖版本，但代码未固定 Hugging Face 模型 revision。
- 重排结论仅适用于已评估的模型和分段配置。可能存在的不完整标注仍需独立核验；所有指标继续使用未修改的原始 qrels。
- 尚未开展统计显著性检验或运行耗时评估。
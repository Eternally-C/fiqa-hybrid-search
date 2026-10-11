# Validation 实验记录

项目：Hybrid Semantic Search and Neural Reranking for Financial Question Answering

记录范围：Retrieval、Cross-Encoder Reranking 与 Error Analysis

> **完成状态：** Validation 实验与错误分析已完成；最终方法与参数已在 Test 前依据 Validation 锁定，最终 Test 评估已完成。本文汇总 Validation 结果，固定方案与最终 Test 指标见[最终评估方案与完成记录](final_evaluation_plan.md)和[Test 结果](test_baselines.csv)。

## 1. 评估设置

- 数据集：BEIR FiQA-2018。
- 本阶段评估使用全部 500 条 Validation 查询。
- 当前开发与分析入口只加载 Validation qrels。
- Test 最终评估已在方案与参数锁定后完成；本文汇总 Validation 实验，最终 Test 指标见[Test 结果](test_baselines.csv)。
- 指标按查询计算后取平均：Recall@10、Recall@100、MRR@10、binary nDCG@10。
- 方案比较以 Validation nDCG@10 为主要依据。
- 使用预训练模型推理，本阶段未进行模型微调。

## 2. 模型与参数

| 组件 | 配置 |
|---|---|
| BM25 | lowercase + regex tokenizer |
| Dense encoder | sentence-transformers/multi-qa-MiniLM-L6-cos-v1 |
| Dense embeddings | 384 维，归一化 |
| Weighted RRF | rrf_k=60，candidate_k=100 |
| 融合权重 | BM25=0.10，Dense=1.00 |
| Cross-Encoder | cross-encoder/ms-marco-MiniLM-L6-v2 |
| Cross-Encoder 输入上限 | query 与 document/passage 合计最多 512 tokens，包含特殊 token |
| 分段窗口 | 400 个文档 tokens |
| 相邻段重叠 | 80 个文档 tokens |
| 分段文档评分 | 各段分数取最大值 |
| 重排候选 | 第一阶段返回的 Top-100 文档 |

BM25 权重 0.10 是已有 Validation 权重搜索中记录的最佳值，不代表所有可能权重的全局最优值。

分段重排逐一检查实际 query–passage 输入长度；超限时明确报错，避免再次静默截断。

## 3. Validation 总体结果

下表保留六位小数，原始 CSV 保存完整数值。

| 方法 | Recall@10 | Recall@100 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.271037 | 0.468211 | 0.287029 | 0.228423 |
| Dense | 0.444063 | 0.691544 | 0.451139 | 0.379798 |
| Weighted Hybrid | 0.451300 | 0.691544 | 0.465729 | 0.390140 |
| Weighted Hybrid + Cross-Encoder | 0.455475 | 0.691544 | 0.456533 | 0.383587 |
| Dense + Cross-Encoder | 0.455475 | 0.691544 | 0.456533 | 0.383587 |
| Weighted Hybrid + Passage Cross-Encoder | 0.447088 | 0.691544 | 0.457879 | 0.380282 |

当前已完成实验中，Weighted Hybrid 的 Validation nDCG@10 最高。

分段重排相对于 Weighted Hybrid 的 nDCG@10 下降 0.009858，相对下降约 2.53%。相对于整篇重排，下降 0.003305。

重排只改变同一批候选文档的顺序，因此对应的 Recall@100 保持一致。

Dense + Cross-Encoder 与 Weighted Hybrid + Cross-Encoder 的四项汇总指标相同；仅凭汇总指标不能证明每条查询的完整排序相同。

## 4. 逐查询 Error Analysis

以下改善、恶化和持平，均按每条查询相对于 Weighted Hybrid 的 nDCG@10 变化统计，判断容差为 1e-12。

| 重排方案 | 改善 | 恶化 | 持平 | 平均 nDCG@10 变化 |
|---|---:|---:|---:|---:|
| 整篇重排 | 122 | 144 | 234 | -0.006553 |
| 分段重排 | 123 | 142 | 235 | -0.009858 |

按 Query ID 直接比较分段与整篇重排：

- 分段更好：22 条。
- 分段更差：46 条。
- nDCG@10 相同：432 条。

改善数量不能单独代表整体收益，还需要考虑每条查询变化的幅度。

两份分析文件均包含 500 条唯一查询，查询集合及 Hybrid 基线指标一致。分段逐查询指标的平均值与独立完整评估结果一致。

## 5. 代表性案例

### Query 505：重排成功

问题：static data for mutual funds/hedge funds  
相关文档：154014。

| 方法 | 首个相关文档排名 | nDCG@10 |
|---|---:|---:|
| Weighted Hybrid | 20 | 0.000000 |
| 整篇重排 | 1 | 1.000000 |
| 分段重排 | 1 | 1.000000 |

该案例说明重排能够改善候选集内的排序，但单个成功案例不能证明整体提升。

### Query 7681：长文档截断与分段改善

问题：JCI headache part 1: How to calculate cost basis / tax consequences of JCI -> TYC merger?  
相关文档：307602、406585。

已有长度诊断确认，文档 307602 对应的 query–document 输入超过模型上限，存在截断。

| 方法 | 首个相关文档排名 | Recall@10 | nDCG@10 |
|---|---:|---:|---:|
| Weighted Hybrid | 1 | 1.0 | 0.919721 |
| 整篇重排 | 7 | 0.5 | 0.204382 |
| 分段重排 | 1 | 0.5 | 0.613147 |

分段重排改善了该案例，但另一篇相关文档仍在 Top-10 之外，尚未恢复到 Hybrid 的表现。

### Query 5142：分段产生额外退化

问题：How do I set up my finances when first moving out?  
唯一标注的相关文档：154714。

| 方法 | 首个相关文档排名 | nDCG@10 |
|---|---:|---:|
| Weighted Hybrid | 11 | 0.000000 |
| 整篇重排 | 1 | 1.000000 |
| 分段重排 | 2 | 0.630930 |

两篇竞争文档的评分如下：

| 文档 | FiQA 标注 | 整篇分数 | 分段文档分数 | 最高分段 |
|---|---|---:|---:|---:|
| 154714 | 相关 | 0.952828 | 0.947786 | 1 |
| 577479 | 未标为相关 | 0.862882 | 0.953608 | 1 |

段级诊断：

| 文档 | 第 1 段 | 第 2 段 | 第 3 段 |
|---|---:|---:|---:|
| 154714 | 0.947786 | 0.000390 | — |
| 577479 | 0.953608 | 0.007484 | 0.779692 |

两篇文档的最高分均来自第一段。后续段没有提高最终文档分数，因此不能把本案例的翻转归因于后续段发现新的高分信息。

证据支持截取范围和上下文变化影响评分的解释，但尚未验证具体内容的因果作用，也不能据此证明普遍存在段数偏差。

文档 577479 的最高分段讨论搬出父母家后的财务准备，具有明显主题相关性。是否存在漏标需要独立核验；本项目继续按原始 qrels 计算指标，不根据个别案例修改标注。

模型分数用于排序，不应解释为经过校准的相关性概率。

## 6. 当前结论与边界

- 当前已完成方案中，Weighted Hybrid 的 Validation nDCG@10 最佳。
- 当前整篇重排与分段重排都没有提升主要指标。
- 分段方案改善了个别长文档案例，但整体表现下降。
- 本结论只针对当前模型、候选集、400/80 分段参数和最高段分聚合方式，不代表所有分段方案都无效。
- 可能存在不完整标注，但目前没有完成系统性人工核验。
- 尚未完成统计显著性检验或运行耗时评估，不声称差异显著，也不报告性能开销数值。
- 重排实验作为有证据的负面实验保留。
- 最终采用 Weighted Hybrid；方法与参数在 Test 前依据 Validation 确定。Test 评估已完成，其结果不用于后续调参或模型选择。

## 7. 证据文件

总体指标：

- results/validation_baselines.csv
- results/hybrid_weight_search.csv
- results/reranker_validation.csv
- results/dense_reranker_validation.csv
- results/passage_reranker_validation.csv

逐查询分析：

- results/reranking_error_analysis.csv
- results/passage_reranking_error_analysis.csv

案例检查入口：

- scripts/inspect_reranking_cases.py

段级评分和文本来自本机终端诊断，未另存日志；正式案例脚本已支持指定 Query、指定文档和段级详情，可用于复查。

上述脚本和结果属于正式项目材料，应保留。模型、数据及 embedding 缓存本地保留，并按项目忽略规则处理。
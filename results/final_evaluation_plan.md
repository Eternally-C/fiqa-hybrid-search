# 最终评估方案

项目：Hybrid Semantic Search and Neural Reranking for Financial Question Answering

记录日期：2026-10-09

当前状态：最终方法与参数已锁定；运行前代码、导入和环境记录审查通过；最终 Test 评估已完成，终端日志与结果 CSV 审查通过。

## 1. 模型选择依据

以 Validation 集的宏平均 nDCG@10 作为主要模型选择指标，同时报告 Recall@10、Recall@100 和 MRR@10。

目前已完成的实验中，Weighted Hybrid 的 Validation nDCG@10 最高，为 0.390140。因此选择 Weighted Hybrid 作为最终检索方法。

BM25 权重 0.10 是已记录的 Validation 权重搜索中最优的取值，不代表所有可能取值中的全局最优。

Whole-document Cross-Encoder 和 passage-based Cross-Encoder 均未提高整体 Validation nDCG@10。保留其代码、结果和 Error Analysis，作为已完成的重排实验。

## 2. 最终 Test 比较方法

最终 Test 评估运行以下三种方法：

| 方法 | 作用 |
|---|---|
| BM25 | 词项检索基线 |
| Dense | 语义检索基线 |
| Weighted Hybrid | Validation 选出的最终方法 |

本轮最终 Test 比较不加入 Cross-Encoder。重排方案的比较与选择已经在 Validation 上完成。

## 3. 固定配置

### BM25

- 只使用 corpus text。
- 文本转为小写。
- 使用正则表达式 `r"\b\w+\b"` 分词。
- 使用 BM25Okapi。
- 当前实现采用的默认参数：k1=1.5、b=0.75、epsilon=0.25。
- 运行前核对依赖版本，确保这些默认参数保持一致。

### Dense

- 模型：sentence-transformers/multi-qa-MiniLM-L6-cos-v1。
- Embedding dimension：384。
- Corpus 和 query embeddings 均进行归一化。
- 使用向量内积计算检索分数。
- 复用通过 metadata 校验的 corpus embedding cache。
- 运行前核对本地模型缓存及环境记录。

### Weighted Hybrid

- 融合方法：Weighted Reciprocal Rank Fusion。
- rrf_k：60。
- 每路检索候选数量 candidate_k：100。
- BM25 weight：0.10。
- Dense weight：1.0。

以上配置依据 Validation 结果确定，不依据 Test 结果调整。

## 4. 数据与指标

- 数据集：BEIR FiQA-2018。
- Corpus：57,638 篇文档。
- Test：预期 648 个 query，实际数量在最终评估时核对。
- Qrels：binary relevance。
- 输出指标：Recall@10、Recall@100、MRR@10、nDCG@10。
- 指标采用逐 query 计算后取宏平均的方式。
- 每个方法检索 Top 100，以支持 Recall@100 的计算。

共享 corpus embedding cache 不使用 qrels。模型选择和参数选择仅依据 Validation。

## 5. Test 使用规则

最终评估入口准备完成并通过只读审查后，才运行 Test 评估。

Test 结果只用于报告固定方案的最终表现，不用于调整模型、融合权重、候选数量、分块配置或其他超参数。

若发生程序错误，需要修复并重跑，应记录错误原因及修复内容；不得以提高 Test 指标为理由修改已选方案。

不得根据人工阅读结果改写原始 qrels。

## 6. 运行前审查

运行前确认：

- 最终评估入口使用上述三种方法和固定参数。
- Qrels 加载明确指定 Test，且只在正式最终评估时执行。
- 指标实现与 Validation 保持一致。
- Dense cache metadata 校验通过。
- 环境及依赖版本有记录。
- 输出路径与已有 Validation 结果分开。
- 不包含根据 Test 结果重新搜索参数的逻辑。

结果文件：results/test_baselines.csv。

该文件已生成，包含 BM25、Dense 和 Weighted Hybrid 三种方法的最终 Test 指标，并通过只读审查。

## 7. 结果报告边界

报告三种方法的实际 Test 指标，无论 Weighted Hybrid 是否继续领先。

Validation 上最优不保证 Test 上最优。

不将 Cross-Encoder 的局部案例改善表述为整体性能提升。

没有统计显著性分析或专门的耗时测量时，不宣称提升具有统计显著性或系统运行更快。
from pathlib import Path

import pandas as pd

from src.bm25_retriever import BM25Retriever
from src.data_loader import (
    build_id_maps,
    build_qrels_map,
    load_fiqa_data,
    load_fiqa_qrels,
)
from src.dense_retriever import DenseRetriever
from src.evaluation import evaluate_retriever
from src.hybrid_retriever import HybridRetriever


corpus, queries = load_fiqa_data()

# Validation 阶段只加载验证集 qrels，不访问 Test labels
dev_qrels = load_fiqa_qrels("validation")

_, queries_by_id = build_id_maps(corpus, queries)
dev_qrels_map = build_qrels_map(dev_qrels)

# BM25 Baseline
bm25_retriever = BM25Retriever(corpus)
bm25_metrics = evaluate_retriever(
    bm25_retriever,
    queries_by_id,
    dev_qrels_map,
)

# Dense Retrieval
dense_retriever = DenseRetriever(corpus)
dense_retriever.build_index()

dense_metrics = evaluate_retriever(
    dense_retriever,
    queries_by_id,
    dev_qrels_map,
)

# 使用 Validation Set 选定的最佳 RRF 权重
hybrid_retriever = HybridRetriever(
    bm25_retriever,
    dense_retriever,
    rrf_k=60,
    candidate_k=100,
    bm25_weight=0.10,
    dense_weight=1.00,
)

hybrid_metrics = evaluate_retriever(
    hybrid_retriever,
    queries_by_id,
    dev_qrels_map,
)

results = pd.DataFrame(
    [
        {
            "Method": "BM25",
            **bm25_metrics,
        },
        {
            "Method": "Dense",
            **dense_metrics,
        },
        {
            "Method": "Weighted Hybrid",
            **hybrid_metrics,
        },
    ]
)

output_path = Path("results/validation_baselines.csv")
results.to_csv(output_path, index=False)

print(results.to_string(index=False))
print(f"\nSaved results to: {output_path}")
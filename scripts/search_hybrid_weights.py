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
from src.evaluation import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from src.hybrid_retriever import HybridRetriever


# 固定实验配置
RRF_K = 60
CANDIDATE_K = 100
DENSE_WEIGHT = 1.0

# 合并此前的粗搜索与局部搜索候选
BM25_WEIGHTS = [
    0.00,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.50,
    0.75,
    1.00,
]


def main():
    """在 Validation Set 上搜索 Weighted RRF 的 BM25 权重。"""

    corpus, queries = load_fiqa_data()

    # 权重选择严格只使用 Validation labels
    dev_qrels = load_fiqa_qrels("validation")

    _, queries_by_id = build_id_maps(corpus, queries)
    dev_qrels_map = build_qrels_map(dev_qrels)

    bm25_retriever = BM25Retriever(corpus)

    dense_retriever = DenseRetriever(corpus)
    dense_retriever.build_index()

    hybrid_retriever = HybridRetriever(
        bm25_retriever,
        dense_retriever,
        rrf_k=RRF_K,
        candidate_k=CANDIDATE_K,
    )

    metric_sums = {
        weight: {
            "Recall@10": 0.0,
            "Recall@100": 0.0,
            "MRR@10": 0.0,
            "nDCG@10": 0.0,
        }
        for weight in BM25_WEIGHTS
    }

    for query_id, relevant_ids in dev_qrels_map.items():
        query_text = queries_by_id[query_id]["text"]

        # 每个 Query 的两路检索只执行一次
        bm25_results = bm25_retriever.search(
            query_text,
            top_k=CANDIDATE_K,
        )

        dense_results = dense_retriever.search(
            query_text,
            top_k=CANDIDATE_K,
        )

        for bm25_weight in BM25_WEIGHTS:
            fused_results = hybrid_retriever.fuse_results(
                bm25_results,
                dense_results,
                top_k=CANDIDATE_K,
                bm25_weight=bm25_weight,
                dense_weight=DENSE_WEIGHT,
            )

            retrieved_ids = [
                result["corpus_id"]
                for result in fused_results
            ]

            metric_sums[bm25_weight]["Recall@10"] += recall_at_k(
                retrieved_ids,
                relevant_ids,
                k=10,
            )

            metric_sums[bm25_weight]["Recall@100"] += recall_at_k(
                retrieved_ids,
                relevant_ids,
                k=100,
            )

            metric_sums[bm25_weight]["MRR@10"] += reciprocal_rank_at_k(
                retrieved_ids,
                relevant_ids,
                k=10,
            )

            metric_sums[bm25_weight]["nDCG@10"] += ndcg_at_k(
                retrieved_ids,
                relevant_ids,
                k=10,
            )

    num_queries = len(dev_qrels_map)

    rows = []

    for bm25_weight in BM25_WEIGHTS:
        rows.append(
            {
                "BM25 Weight": bm25_weight,
                "Dense Weight": DENSE_WEIGHT,
                "RRF k": RRF_K,
                "Candidate k": CANDIDATE_K,
                "Recall@10": (
                    metric_sums[bm25_weight]["Recall@10"]
                    / num_queries
                ),
                "Recall@100": (
                    metric_sums[bm25_weight]["Recall@100"]
                    / num_queries
                ),
                "MRR@10": (
                    metric_sums[bm25_weight]["MRR@10"]
                    / num_queries
                ),
                "nDCG@10": (
                    metric_sums[bm25_weight]["nDCG@10"]
                    / num_queries
                ),
            }
        )

    results = pd.DataFrame(rows)

    # 模型选择规则：Validation nDCG@10 最大
    best_row = results.loc[results["nDCG@10"].idxmax()]

    output_path = Path("results/hybrid_weight_search.csv")
    results.to_csv(output_path, index=False)

    print(results.to_string(index=False))

    print("\nBest configuration by Validation nDCG@10:")
    print(
        f"BM25 Weight={best_row['BM25 Weight']:.2f}, "
        f"Dense Weight={best_row['Dense Weight']:.2f}, "
        f"nDCG@10={best_row['nDCG@10']:.6f}"
    )

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
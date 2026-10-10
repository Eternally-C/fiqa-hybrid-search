import argparse
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
from src.reranker import CrossEncoderReranker
from src.passage_reranker import PassageCrossEncoderReranker


def first_relevant_rank(retrieved_ids, relevant_ids):
    """返回第一个相关文档的排名；若未检索到则返回 None。"""

    for rank, corpus_id in enumerate(retrieved_ids, start=1):
        if corpus_id in relevant_ids:
            return rank

    return None


def main():
    """分析不同重排方案对 Weighted Hybrid 排名的逐 Query 影响。"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reranker",
        choices=["whole", "passage"],
        default="whole",
    )
    args = parser.parse_args()

    corpus, queries = load_fiqa_data()

    # Error Analysis 严格只使用 Validation labels
    dev_qrels = load_fiqa_qrels("validation")

    _, queries_by_id = build_id_maps(corpus, queries)
    dev_qrels_map = build_qrels_map(dev_qrels)

    bm25_retriever = BM25Retriever(corpus)

    dense_retriever = DenseRetriever(corpus)
    dense_retriever.build_index()

    hybrid_retriever = HybridRetriever(
        bm25_retriever,
        dense_retriever,
        rrf_k=60,
        candidate_k=100,
        bm25_weight=0.10,
        dense_weight=1.00,
    )

    # 与完整 Validation 评估保持相同的分段参数和评分方式。
    if args.reranker == "passage":
        reranker = PassageCrossEncoderReranker(
            chunk_size=400,
            overlap=80,
        )
        output_path = Path("results/passage_reranking_error_analysis.csv")
    else:
        reranker = CrossEncoderReranker()
        output_path = Path("results/reranking_error_analysis.csv")

    rows = []

    for query_id, relevant_ids in dev_qrels_map.items():
        query_text = queries_by_id[query_id]["text"]

        # 第一阶段 Weighted Hybrid Top-100
        base_results = hybrid_retriever.search(
            query_text,
            top_k=100,
        )

        # 第二阶段 Cross-Encoder 对同一候选集重新排序
        reranked_results = reranker.rerank(
            query_text,
            base_results,
            top_k=100,
        )

        base_ids = [
            result["corpus_id"]
            for result in base_results
        ]

        reranked_ids = [
            result["corpus_id"]
            for result in reranked_results
        ]

        base_ndcg = ndcg_at_k(
            base_ids,
            relevant_ids,
            k=10,
        )

        reranked_ndcg = ndcg_at_k(
            reranked_ids,
            relevant_ids,
            k=10,
        )

        rows.append(
            {
                "query_id": query_id,
                "query": query_text,
                "relevant_count": len(relevant_ids),
                "base_recall@10": recall_at_k(
                    base_ids,
                    relevant_ids,
                    k=10,
                ),
                "reranked_recall@10": recall_at_k(
                    reranked_ids,
                    relevant_ids,
                    k=10,
                ),
                "base_mrr@10": reciprocal_rank_at_k(
                    base_ids,
                    relevant_ids,
                    k=10,
                ),
                "reranked_mrr@10": reciprocal_rank_at_k(
                    reranked_ids,
                    relevant_ids,
                    k=10,
                ),
                "base_ndcg@10": base_ndcg,
                "reranked_ndcg@10": reranked_ndcg,
                "delta_ndcg@10": reranked_ndcg - base_ndcg,
                "base_first_relevant_rank": first_relevant_rank(
                    base_ids,
                    relevant_ids,
                ),
                "reranked_first_relevant_rank": first_relevant_rank(
                    reranked_ids,
                    relevant_ids,
                ),
            }
        )

    results = pd.DataFrame(rows)

    results.to_csv(output_path, index=False)

    improved = (results["delta_ndcg@10"] > 1e-12).sum()
    worsened = (results["delta_ndcg@10"] < -1e-12).sum()
    unchanged = len(results) - improved - worsened

    print("Validation queries:", len(results))
    print("nDCG@10 improved:", improved)
    print("nDCG@10 worsened:", worsened)
    print("nDCG@10 unchanged:", unchanged)

    print(
        "\nMean nDCG@10 delta:",
        f"{results['delta_ndcg@10'].mean():.6f}",
    )

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
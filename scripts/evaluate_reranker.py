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
from src.evaluation import evaluate_retriever
from src.hybrid_retriever import HybridRetriever
from src.reranked_retriever import RerankedRetriever
from src.reranker import CrossEncoderReranker
from src.passage_reranker import PassageCrossEncoderReranker


def main():
    """评估 Weighted Hybrid + Cross-Encoder 的 Validation 性能。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reranker",
        choices=["whole", "passage"],
        default="whole",
    )
    args = parser.parse_args()

    corpus, queries = load_fiqa_data()

    # 模型开发与比较严格只使用 Validation labels
    dev_qrels = load_fiqa_qrels("validation")

    _, queries_by_id = build_id_maps(corpus, queries)
    dev_qrels_map = build_qrels_map(dev_qrels)

    bm25_retriever = BM25Retriever(corpus)

    dense_retriever = DenseRetriever(corpus)
    dense_retriever.build_index()

    # 使用第一阶段 Validation 搜索确定的最佳 RRF 参数
    hybrid_retriever = HybridRetriever(
        bm25_retriever,
        dense_retriever,
        rrf_k=60,
        candidate_k=100,
        bm25_weight=0.10,
        dense_weight=1.00,
    )

    # 固定分段参数，先验证方案在整个 Validation 上的表现。
    if args.reranker == "passage":
        reranker = PassageCrossEncoderReranker(
            chunk_size=400,
            overlap=80,
        )
        method_name = "Weighted Hybrid + Passage Cross-Encoder"
        output_path = Path("results/passage_reranker_validation.csv")
    else:
        reranker = CrossEncoderReranker()
        method_name = "Weighted Hybrid + Cross-Encoder"
        output_path = Path("results/reranker_validation.csv")

    reranked_retriever = RerankedRetriever(
        base_retriever=hybrid_retriever,
        reranker=reranker,
        candidate_k=100,
    )

    metrics = evaluate_retriever(
        reranked_retriever,
        queries_by_id,
        dev_qrels_map,
    )

    results = pd.DataFrame(
        [
            {
                "Method": method_name,
                **metrics,
            }
        ]
    )

    results.to_csv(output_path, index=False)

    print(results.to_string(index=False))
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
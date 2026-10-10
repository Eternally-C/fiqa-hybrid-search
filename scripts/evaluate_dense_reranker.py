from pathlib import Path

import pandas as pd

from src.data_loader import (
    build_id_maps,
    build_qrels_map,
    load_fiqa_data,
    load_fiqa_qrels,
)
from src.dense_retriever import DenseRetriever
from src.evaluation import evaluate_retriever
from src.reranked_retriever import RerankedRetriever
from src.reranker import CrossEncoderReranker


def main():
    """评估 Dense Retrieval + Cross-Encoder 的 Validation 性能。"""

    corpus, queries = load_fiqa_data()

    # 模型开发与比较严格只使用 Validation labels
    dev_qrels = load_fiqa_qrels("validation")

    _, queries_by_id = build_id_maps(corpus, queries)
    dev_qrels_map = build_qrels_map(dev_qrels)

    dense_retriever = DenseRetriever(corpus)
    dense_retriever.build_index()

    reranker = CrossEncoderReranker()

    reranked_retriever = RerankedRetriever(
        base_retriever=dense_retriever,
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
                "Method": "Dense + Cross-Encoder",
                **metrics,
            }
        ]
    )

    output_path = Path("results/dense_reranker_validation.csv")
    results.to_csv(output_path, index=False)

    print(results.to_string(index=False))
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
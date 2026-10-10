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


def main():
    """使用 Validation 选定的固定方案进行最终 Test 评估。"""

    parser = argparse.ArgumentParser(
        description="评估 BM25、Dense 和固定参数的 Weighted Hybrid。"
    )
    parser.add_argument(
        "--run-test",
        action="store_true",
        help="显式启动最终 Test 评估。",
    )
    args = parser.parse_args()

    # 普通调用或查看帮助时，不加载 Test labels。
    if not args.run_test:
        parser.error("最终评估需要显式提供 --run-test。")

    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "results" / "test_baselines.csv"
    cache_path = project_root / "data" / "cache" / "corpus_embeddings.pt"

    # 在加载数据前检查结果文件，避免无意重复评估并覆盖结果。
    if output_path.exists():
        raise FileExistsError(
            f"Test 结果已存在，请先核对已有实验记录：{output_path}"
        )

    corpus, queries = load_fiqa_data()

    if len(corpus) != 57638:
        raise ValueError(
            f"Corpus 数量异常：预期 57638，实际 {len(corpus)}。"
        )

    _, queries_by_id = build_id_maps(corpus, queries)

    bm25_retriever = BM25Retriever(corpus)

    dense_retriever = DenseRetriever(
        corpus,
        model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    )
    dense_retriever.build_index(cache_path=cache_path)

    # 参数已经根据 Validation 确定，不在 Test 上搜索或调整。
    hybrid_retriever = HybridRetriever(
        bm25_retriever,
        dense_retriever,
        rrf_k=60,
        candidate_k=100,
        bm25_weight=0.10,
        dense_weight=1.00,
    )

    # 仅在正式启动最终评估后，按需加载 Test qrels。
    test_qrels = load_fiqa_qrels("test")
    test_qrels_map = build_qrels_map(test_qrels)

    if len(test_qrels_map) != 648:
        raise ValueError(
            f"Test query 数量异常：预期 648，实际 {len(test_qrels_map)}。"
        )

    print(f"Corpus documents: {len(corpus)}")
    print(f"Test queries: {len(test_qrels_map)}")
    print(f"Dense device: {dense_retriever.device}")
    print(
        "Fixed Hybrid: rrf_k=60, candidate_k=100, "
        "bm25_weight=0.10, dense_weight=1.00"
    )

    methods = [
        ("BM25", bm25_retriever),
        ("Dense", dense_retriever),
        ("Weighted Hybrid", hybrid_retriever),
    ]

    rows = []

    for method_name, retriever in methods:
        print(f"\nEvaluating: {method_name}", flush=True)

        # 与 Validation 复用同一套 Top-100 检索和宏平均指标。
        metrics = evaluate_retriever(
            retriever,
            queries_by_id,
            test_qrels_map,
        )
        rows.append({"Method": method_name, **metrics})

    results = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 排他创建：即使运行期间出现同名文件，也不会覆盖。
    results.to_csv(output_path, index=False, mode="x")

    print("\n" + results.to_string(index=False))
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
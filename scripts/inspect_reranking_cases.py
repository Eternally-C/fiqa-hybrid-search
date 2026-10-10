import argparse
from src.passage_reranker import PassageCrossEncoderReranker

from src.bm25_retriever import BM25Retriever
from src.data_loader import (
    build_id_maps,
    build_qrels_map,
    load_fiqa_data,
    load_fiqa_qrels,
)
from src.dense_retriever import DenseRetriever
from src.hybrid_retriever import HybridRetriever
from src.reranker import CrossEncoderReranker


CASE_QUERY_IDS = ["505", "7681"]
DISPLAY_TOP_K = 5


def find_rank(results, relevant_ids):
    """返回第一个相关文档在结果中的排名。"""

    for rank, result in enumerate(results, start=1):
        if result["corpus_id"] in relevant_ids:
            return rank

    return None


def print_results(title, results, relevant_ids):
    """打印排名靠前的文档及其相关性。"""

    print(f"\n{title}")

    for rank, result in enumerate(
        results[:DISPLAY_TOP_K],
        start=1,
    ):
        is_relevant = result["corpus_id"] in relevant_ids

        print(
            f"\nRank {rank}"
            f"\nCorpus ID: {result['corpus_id']}"
            f"\nRelevant: {is_relevant}"
            f"\nScore: {result['score']:.6f}"
            f"\nText: {result['text'][:500]}"
        )

        # 分段重排结果额外显示实际评分的段数，辅助分析排名变化。
        if "passage_count" in result:
            print(f"Passages scored: {result['passage_count']}")


def main():
    """检查代表性的 Cross-Encoder 成功和失败案例。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reranker",
        choices=["whole", "passage"],
        default="whole",
    )
    parser.add_argument(
        "--query-ids",
        nargs="+",
        default=CASE_QUERY_IDS,
        help="要检查的 Validation Query ID，可指定一个或多个。",
    )
    parser.add_argument(
        "--inspect-docs",
        nargs="+",
        default=[],
        help="要显示段级评分的文档 ID，仅用于 passage 模式。",
    )
    args = parser.parse_args()

    if args.inspect_docs and args.reranker != "passage":
        parser.error("--inspect-docs 需要搭配 --reranker passage。")

    corpus, queries = load_fiqa_data()

    # Error Analysis 只使用 Validation labels
    dev_qrels = load_fiqa_qrels("validation")

    corpus_by_id, queries_by_id = build_id_maps(
        corpus,
        queries,
    )
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

    if args.reranker == "passage":
        reranker = PassageCrossEncoderReranker(
            chunk_size=400,
            overlap=80,
        )
        result_title = "Passage Cross-Encoder Top-5"
    else:
        reranker = CrossEncoderReranker()
        result_title = "Whole-document Cross-Encoder Top-5"

    for query_id in args.query_ids:
        query_text = queries_by_id[query_id]["text"]
        relevant_ids = dev_qrels_map[query_id]

        base_results = hybrid_retriever.search(
            query_text,
            top_k=100,
        )

        if args.reranker == "passage":
            reranked_results = reranker.rerank(
                query_text,
                base_results,
                top_k=100,
                include_passage_details=bool(args.inspect_docs),
            )
        else:
            reranked_results = reranker.rerank(
                query_text,
                base_results,
                top_k=100,
            )

        print("\n" + "=" * 80)
        print(f"Query ID: {query_id}")
        print(f"Query: {query_text}")
        print(f"Relevant IDs: {relevant_ids}")

        print(
            "Hybrid first relevant rank:",
            find_rank(base_results, relevant_ids),
        )

        print(
            "Reranked first relevant rank:",
            find_rank(reranked_results, relevant_ids),
        )

        print("\nRelevant document(s):")

        for corpus_id in relevant_ids:
            print(
                f"\nCorpus ID: {corpus_id}"
                f"\nText: {corpus_by_id[corpus_id]['text']}"
            )

        print_results(
            "Weighted Hybrid Top-5",
            base_results,
            relevant_ids,
        )

        print_results(
            result_title,
            reranked_results,
            relevant_ids,
        )

        if args.inspect_docs:
            results_by_id = {
                result["corpus_id"]: result
                for result in reranked_results
            }

            print("\nSelected document passage details:")

            for corpus_id in args.inspect_docs:
                result = results_by_id.get(corpus_id)

                if result is None:
                    print(
                        f"\nCorpus ID: {corpus_id}"
                        "\nNot found in Hybrid Top-100 candidates."
                    )
                    continue

                best_index = result["best_passage_index"]

                print(
                    f"\nCorpus ID: {corpus_id}"
                    f"\nRelevant: {corpus_id in relevant_ids}"
                    f"\nDocument score: {result['score']:.6f}"
                    f"\nBest passage: {best_index}"
                )

                # 输出所有段分数，只展开最高分段的完整文本。
                for passage in result["passage_details"]:
                    marker = (
                        " (best)"
                        if passage["passage_index"] == best_index
                        else ""
                    )
                    print(
                        f"Passage {passage['passage_index']}: "
                        f"{passage['score']:.6f}{marker}"
                    )

                best_passage = result["passage_details"][best_index - 1]
                print("\nBest passage text:")
                print(best_passage["text"])


if __name__ == "__main__":
    main()
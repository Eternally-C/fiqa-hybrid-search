import pandas as pd


RESULT_PATH = "results/reranking_error_analysis.csv"
TOP_N = 5


def print_cases(title, dataframe):
    """打印用于人工分析的代表性 Query。"""

    print(f"\n{title}")

    for _, row in dataframe.iterrows():
        print(
            f"\nQuery ID: {row['query_id']}\n"
            f"Query: {row['query']}\n"
            f"Base nDCG@10: {row['base_ndcg@10']:.4f}\n"
            f"Reranked nDCG@10: {row['reranked_ndcg@10']:.4f}\n"
            f"Delta nDCG@10: {row['delta_ndcg@10']:+.4f}\n"
            f"Base first relevant rank: "
            f"{row['base_first_relevant_rank']}\n"
            f"Reranked first relevant rank: "
            f"{row['reranked_first_relevant_rank']}"
        )


def main():
    """选择 Cross-Encoder 改善和退化最明显的 Validation Query。"""

    results = pd.read_csv(RESULT_PATH)

    most_improved = results.nlargest(
        TOP_N,
        "delta_ndcg@10",
    )

    most_worsened = results.nsmallest(
        TOP_N,
        "delta_ndcg@10",
    )

    print_cases(
        f"Top {TOP_N} improved queries",
        most_improved,
    )

    print_cases(
        f"Top {TOP_N} worsened queries",
        most_worsened,
    )


if __name__ == "__main__":
    main()
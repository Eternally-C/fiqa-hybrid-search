import math


def recall_at_k(retrieved_ids, relevant_ids, k):
    """计算单个查询的 Recall@K。"""

    if not relevant_ids:
        return 0.0

    top_k_ids = retrieved_ids[:k]
    hits = len(set(top_k_ids) & set(relevant_ids))

    return hits / len(relevant_ids)


def reciprocal_rank_at_k(retrieved_ids, relevant_ids, k):
    """计算单个查询在 Top-K 中的 Reciprocal Rank。"""

    for rank, corpus_id in enumerate(retrieved_ids[:k], start=1):
        if corpus_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    """计算单个查询的 binary nDCG@K。"""

    dcg = 0.0

    for rank, corpus_id in enumerate(retrieved_ids[:k], start=1):
        if corpus_id in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_ids), k)

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg


def evaluate_retriever(retriever, queries_by_id, qrels_map):
    """在整个查询集合上计算 Retrieval 指标。"""

    recall_10_scores = []
    recall_100_scores = []
    reciprocal_ranks = []
    ndcgs = []

    for query_id, relevant_ids in qrels_map.items():
        query_text = queries_by_id[query_id]["text"]

        # 一次检索 Top-100，复用于不同 K 的指标计算
        results = retriever.search(query_text, top_k=100)

        retrieved_ids = [
            result["corpus_id"]
            for result in results
        ]

        recall_10_scores.append(
            recall_at_k(retrieved_ids, relevant_ids, k=10)
        )

        recall_100_scores.append(
            recall_at_k(retrieved_ids, relevant_ids, k=100)
        )

        reciprocal_ranks.append(
            reciprocal_rank_at_k(retrieved_ids, relevant_ids, k=10)
        )

        ndcgs.append(
            ndcg_at_k(retrieved_ids, relevant_ids, k=10)
        )

    return {
        "Recall@10": sum(recall_10_scores) / len(recall_10_scores),
        "Recall@100": sum(recall_100_scores) / len(recall_100_scores),
        "MRR@10": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "nDCG@10": sum(ndcgs) / len(ndcgs),
    }
class HybridRetriever:
    """使用加权 Reciprocal Rank Fusion 融合 BM25 和 Dense Retrieval。"""

    def __init__(
        self,
        bm25_retriever,
        dense_retriever,
        rrf_k=60,
        candidate_k=100,
        bm25_weight=1.0,
        dense_weight=1.0,
    ):
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    def fuse_results(
        self,
        bm25_results,
        dense_results,
        top_k=10,
        bm25_weight=None,
        dense_weight=None,
    ):
        """对已有 BM25 和 Dense 排名执行加权 RRF 融合。"""

        if bm25_weight is None:
            bm25_weight = self.bm25_weight

        if dense_weight is None:
            dense_weight = self.dense_weight

        fused_scores = {}
        documents = {}

        for rank, result in enumerate(bm25_results, start=1):
            corpus_id = result["corpus_id"]

            fused_scores[corpus_id] = (
                fused_scores.get(corpus_id, 0.0)
                + bm25_weight / (self.rrf_k + rank)
            )

            documents[corpus_id] = result["text"]

        for rank, result in enumerate(dense_results, start=1):
            corpus_id = result["corpus_id"]

            fused_scores[corpus_id] = (
                fused_scores.get(corpus_id, 0.0)
                + dense_weight / (self.rrf_k + rank)
            )

            documents[corpus_id] = result["text"]

        ranked_ids = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True,
        )[:top_k]

        return [
            {
                "corpus_id": corpus_id,
                "score": fused_scores[corpus_id],
                "text": documents[corpus_id],
            }
            for corpus_id in ranked_ids
        ]

    def search(self, query, top_k=10):
        """执行 BM25、Dense Retrieval，并融合两种排名。"""

        bm25_results = self.bm25_retriever.search(
            query,
            top_k=self.candidate_k,
        )

        dense_results = self.dense_retriever.search(
            query,
            top_k=self.candidate_k,
        )

        return self.fuse_results(
            bm25_results,
            dense_results,
            top_k=top_k,
        )
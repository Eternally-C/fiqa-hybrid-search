class RerankedRetriever:
    """先生成候选文档，再使用 Cross-Encoder 进行重新排序。"""

    def __init__(
        self,
        base_retriever,
        reranker,
        candidate_k=100,
    ):
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.candidate_k = candidate_k

    def search(self, query, top_k=10):
        """检索候选文档，并返回 Cross-Encoder 重新排序后的 Top-K。"""

        candidates = self.base_retriever.search(
            query,
            top_k=self.candidate_k,
        )

        return self.reranker.rerank(
            query,
            candidates,
            top_k=top_k,
        )
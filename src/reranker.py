import torch
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """使用 Cross-Encoder 对 Retriever 候选文档进行重新排序。"""

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
    ):
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.model = CrossEncoder(
            model_name,
            device=self.device,
            activation_fn=torch.nn.Sigmoid(),
        )

    def rerank(self, query, candidates, top_k=10):
        """对候选文档计算相关性分数并返回重新排序后的 Top-K。"""

        if not candidates:
            return []

        pairs = [
            (query, candidate["text"])
            for candidate in candidates
        ]

        scores = self.model.predict(pairs)

        reranked = []

        for candidate, score in zip(candidates, scores):
            reranked.append(
                {
                    "corpus_id": candidate["corpus_id"],
                    "score": float(score),
                    "text": candidate["text"],
                }
            )

        reranked.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return reranked[:top_k]
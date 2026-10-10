import numpy as np
from rank_bm25 import BM25Okapi

from src.preprocessing import tokenize


class BM25Retriever:
    """使用 BM25 对 FiQA 文档进行词法检索。"""

    def __init__(self, corpus):
        self.corpus = corpus
        self.corpus_ids = [str(document["_id"]) for document in corpus]

        tokenized_corpus = [
            tokenize(document["text"])
            for document in corpus
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query, top_k=5):
        """返回与查询最相关的 Top-K 文档。"""

        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in top_indices:
            document = self.corpus[int(index)]

            results.append(
                {
                    "corpus_id": self.corpus_ids[int(index)],
                    "score": float(scores[index]),
                    "text": document["text"],
                }
            )

        return results
import hashlib
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    """使用 Sentence Transformer 进行语义检索。"""

    def __init__(
        self,
        corpus,
        model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        batch_size=64,
    ):
        self.corpus = corpus
        self.corpus_ids = [str(document["_id"]) for document in corpus]
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = True

        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.model = SentenceTransformer(
            model_name,
            device=self.device,
        )

        self.embedding_dimension = (
            self.model.get_embedding_dimension()
        )

        self.corpus_embeddings = None

    def _compute_corpus_fingerprint(self):
        """计算与文档内容及顺序绑定的 SHA-256 指纹。"""

        hasher = hashlib.sha256()

        for document in self.corpus:
            corpus_id = str(document["_id"])
            text = document["text"]

            hasher.update(corpus_id.encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(text.encode("utf-8"))
            hasher.update(b"\0")

        return hasher.hexdigest()

    def _expected_cache_metadata(self):
        """生成当前模型与语料应对应的缓存元数据。"""

        return {
            "model_name": self.model_name,
            "corpus_size": len(self.corpus),
            "corpus_fingerprint": self._compute_corpus_fingerprint(),
            "embedding_dimension": self.embedding_dimension,
            "normalize_embeddings": self.normalize_embeddings,
        }

    def _is_cache_valid(self, cache_data, expected_metadata):
        """检查缓存结构、元数据和 Embedding 张量是否匹配。"""

        if not isinstance(cache_data, dict):
            return False

        if "embeddings" not in cache_data or "metadata" not in cache_data:
            return False

        if cache_data["metadata"] != expected_metadata:
            return False

        embeddings = cache_data["embeddings"]

        expected_shape = (
            len(self.corpus),
            self.embedding_dimension,
        )

        return (
            isinstance(embeddings, torch.Tensor)
            and tuple(embeddings.shape) == expected_shape
        )

    def build_index(self, cache_path="data/cache/corpus_embeddings.pt"):
        """构建 Dense Embedding 索引，并仅复用通过校验的本地缓存。"""

        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        expected_metadata = self._expected_cache_metadata()

        if cache_path.exists():
            cache_data = torch.load(
                cache_path,
                map_location="cpu",
            )

            if self._is_cache_valid(
                cache_data,
                expected_metadata,
            ):
                self.corpus_embeddings = cache_data["embeddings"].to(
                    self.device
                )
                return

        corpus_texts = [
            document["text"]
            for document in self.corpus
        ]

        self.corpus_embeddings = self.model.encode(
            corpus_texts,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
        )

        cache_data = {
            "embeddings": self.corpus_embeddings.cpu(),
            "metadata": expected_metadata,
        }

        torch.save(
            cache_data,
            cache_path,
        )

        self.corpus_embeddings = self.corpus_embeddings.to(
            self.device
        )

    def search(self, query, top_k=5):
        """返回与查询语义最相似的 Top-K 文档。"""

        if self.corpus_embeddings is None:
            raise RuntimeError("Dense index has not been built.")

        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize_embeddings,
        )

        scores = torch.matmul(
            self.corpus_embeddings,
            query_embedding,
        )

        top_scores, top_indices = torch.topk(
            scores,
            k=top_k,
        )

        results = []

        for score, index in zip(top_scores, top_indices):
            index = int(index.item())
            document = self.corpus[index]

            results.append(
                {
                    "corpus_id": self.corpus_ids[index],
                    "score": float(score.item()),
                    "text": document["text"],
                }
            )

        return results
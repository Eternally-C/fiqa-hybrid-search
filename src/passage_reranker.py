from src.passage_chunking import split_document_into_passages
from src.reranker import CrossEncoderReranker


class PassageCrossEncoderReranker(CrossEncoderReranker):
    """对文档分段评分，以最高段分进行文档级重排。"""

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
        chunk_size=400,
        overlap=80,
    ):
        # 复用原重排器的模型、设备选择和评分激活函数。
        super().__init__(model_name=model_name)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def rerank(
        self,
        query,
        candidates,
        top_k=10,
        include_passage_details=False,
    ):
        if not candidates:
            return []

        pairs = []
        document_ranges = []

        for candidate in candidates:
            passages = split_document_into_passages(
                candidate["text"],
                self.model.tokenizer,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
            )

            start = len(pairs)

            for passage in passages:
                encoded = self.model.tokenizer(
                    query,
                    passage,
                    truncation=False,
                    verbose=False,
                )

                # 超长输入明确报错，避免分段后再次被静默截断。
                if len(encoded["input_ids"]) > self.model.max_seq_length:
                    raise ValueError(
                        f"Input exceeds model limit: "
                        f"corpus_id={candidate['corpus_id']}, "
                        f"tokens={len(encoded['input_ids'])}."
                    )

                pairs.append((query, passage))

            # 记录每篇文档对应的评分区间，防止不同文档的段混淆。
            document_ranges.append((start, len(pairs)))

        scores = self.model.predict(
            pairs,
            batch_size=32,
            show_progress_bar=False,
        )

        reranked = []

        for candidate, (start, end) in zip(candidates, document_ranges):
            document_score = max(
                float(score) for score in scores[start:end]
            )

            result = {
                "corpus_id": candidate["corpus_id"],
                "score": document_score,
                "text": candidate["text"],
                "passage_count": end - start,
            }

            # 按需返回本次推理的段级信息，便于追踪文档分数来源。
            if include_passage_details:
                passage_details = []

                for passage_index, pair_index in enumerate(
                    range(start, end),
                    start=1,
                ):
                    passage_details.append(
                        {
                            "passage_index": passage_index,
                            "score": float(scores[pair_index]),
                            "text": pairs[pair_index][1],
                        }
                    )

                best_passage = max(
                    passage_details,
                    key=lambda item: item["score"],
                )

                result["passage_details"] = passage_details
                result["best_passage_index"] = best_passage["passage_index"]

            reranked.append(result)

        reranked.sort(key=lambda item: item["score"], reverse=True)
        return reranked[:top_k]
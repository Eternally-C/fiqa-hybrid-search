from datasets import load_dataset


def load_fiqa_data():
    """加载 FiQA 文档语料库和查询。"""

    corpus = load_dataset("BeIR/fiqa", "corpus", split="corpus")
    queries = load_dataset("BeIR/fiqa", "queries", split="queries")

    return corpus, queries


def load_fiqa_qrels(split):
    """按需加载指定数据划分的 FiQA 相关性标注。"""

    valid_splits = {"train", "validation", "test"}

    if split not in valid_splits:
        raise ValueError(
            f"Invalid qrels split: {split}. "
            f"Expected one of {sorted(valid_splits)}."
        )

    return load_dataset(
        "BeIR/fiqa-qrels",
        split=split,
    )


def build_id_maps(corpus, queries):
    """建立文档和查询的 ID 索引，便于后续高效查找。"""

    corpus_by_id = {
        str(item["_id"]): item
        for item in corpus
    }

    queries_by_id = {
        str(item["_id"]): item
        for item in queries
    }

    return corpus_by_id, queries_by_id


def build_qrels_map(qrels):
    """将 qrels 转换为 query ID 到相关文档 ID 集合的映射。"""

    qrels_map = {}

    for item in qrels:
        query_id = str(item["query-id"])
        corpus_id = str(item["corpus-id"])

        if query_id not in qrels_map:
            qrels_map[query_id] = set()

        qrels_map[query_id].add(corpus_id)

    return qrels_map
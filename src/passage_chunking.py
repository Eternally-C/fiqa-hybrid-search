def split_document_into_passages(
    text,
    tokenizer,
    chunk_size=400,
    overlap=80,
):
    """按 token 窗口切分文档，返回保留原始文本的 passages。"""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size.")
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for offsets.")

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        verbose=False,
    )
    offsets = encoded["offset_mapping"]

    # 短文保持原样，不做无必要的切分。
    if len(offsets) <= chunk_size:
        return [text]

    passages = []
    start = 0

    while start < len(offsets):
        end = min(start + chunk_size, len(offsets))

        # 用 token 对应的字符位置截取原文，避免 decode 改写文本。
        left = 0 if start == 0 else offsets[start][0]
        right = len(text) if end == len(offsets) else offsets[end - 1][1]
        passages.append(text[left:right])

        if end == len(offsets):
            break

        # 相邻段保留重叠，减少边界附近上下文被切断的影响。
        start = end - overlap

    return passages
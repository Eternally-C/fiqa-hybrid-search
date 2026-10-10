import re


def tokenize(text):
    """将文本转换为小写单词 token，供 BM25 使用。"""

    return re.findall(r"\b\w+\b", text.lower())
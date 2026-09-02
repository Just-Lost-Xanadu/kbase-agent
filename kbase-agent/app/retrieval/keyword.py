"""轻量 BM25 关键词索引：给 hybrid 检索提供 keyword 一路的载体（无重依赖，纯 Python）。

中文不做分词：对连续字母/数字保留整词，对连续汉字按字符 bigram 切，
配合 BM25 在"精确词命中"（如 429、OA）上有可演示的召回价值。
分块数据仍由第 1 周作业（loader/chunker）产出，这里只管建索引与查询。
"""

from __future__ import annotations

import re
from typing import Iterable

from rank_bm25 import BM25Okapi

_ALNUM = re.compile(r"[A-Za-z0-9]+(?:[.\-_/][A-Za-z0-9]+)*")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for tok in _ALNUM.findall(text):
        tokens.append(tok.lower())
    for run in _CJK_RUN.findall(text):
        run = run.lower()
        if len(run) == 1:
            tokens.append(run)
            continue
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


class BM25Index:
    """入参 chunk 形如 {'content': str, 'source': str, 'chunk_id': str}。"""

    def __init__(self, chunks: Iterable[dict]):
        self._chunks = list(chunks)
        self._model = BM25Okapi([tokenize(c["content"]) for c in self._chunks])

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        scores = self._model.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {
                **self._chunks[i],
                "score": round(float(scores[i]), 4),
                "engine": "bm25",
            }
            for i in ranked
        ]

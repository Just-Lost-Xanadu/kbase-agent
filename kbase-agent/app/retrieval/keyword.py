"""轻量 BM25 关键词索引：给 hybrid 检索提供 keyword 一路的载体（无重依赖，纯 Python）。

中文不做"真正分词"（无 jieba/LAC 依赖）：对连续字母/数字保留整词（可精确命中 `429`、`OA`
这类词），对连续汉字按**字符 bigram**（两两压缩）切开。BM25 在"精确词命中"上有可演示的召回
价值——这是纯向量检索（cosine 看重语义相似、对字面字序不敏感）补不到的召回路。

为何字符 bigram 而非 unigram：
  - unigram 对任意两个汉字组合几乎都会命中（召回巨量噪声、几乎无区分度）；
  - bigram 相对更"成语/词缀敏感"，能保留"年假""报销""请假单"等二字片段的高频信号，
    且不依赖外部词典、零下载，纯 Python 即可稳定复现。
每篇 chunk 都会经 tokenize 拆分再喂给 rank_bm25 建模型。

分块数据仍由 loader/chunker 产出，这里只管"把仓库里现有 chunk 建成 keyword 索引并提供 search"。
"""

from __future__ import annotations

import re
from typing import Iterable

from rank_bm25 import BM25Okapi

_ALNUM = re.compile(r"[A-Za-z0-9]+(?:[.\-_/][A-Za-z0-9]+)*")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """把文本切成 token 序列：字母/数字整词转小写，汉字跑切 bigram（单字保留）。

    返回：token 字符串列表。示例："OA 操作指引" -> ['oa', '操', '作', ...]（顺序：先整词后 bigram）。
    被 BM25Index 构造与 search 两侧共用，保证"建索引时的分词" 与 "查询时的分词" 完全一致
    （不一致会直接导致检索错乱——建索引用一套、查询用另一套是 RAG 经典坑）。
    """
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
    """对一个分块集合建立 BM25 词项索引并提供查询。

    OOP 说明：这就是一个"持有可变/聚合状态（_chunks + 预计算 _model）"的类——BM25Okapi 模型
    建一次较贵，实例持有它避免反复重建；这比把它写成模块级全局更干净（允许存在多个语料各自的 Index）。
    pipeline 懒加载时只需 chunks 一次性构造即可。

    入参 chunk 形如 {'content': str, 'source': str, 'chunk_id': str}；search 返回带上 score/engine 便于调试。
    """

    def __init__(self, chunks: Iterable[dict]):
        self._chunks = list(chunks)
        self._model = BM25Okapi([tokenize(c["content"]) for c in self._chunks])

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """按 BM25 打分取 top_k 个 chunk。

        BM25（rank_bm25 的 Okapi 变体）根据词频/TF-IDF 加权给整句打分：命中词的 chunk 分高。
        注意：返回分数可能含很多接近 0 的低分——因为 BM25 对完全无关的 chunk 也给一个小正分；
        是否过滤 min-score 是本项目已知取舍（README/LEARNING 提到可加 IDF 下限），当前保留 top_k 排序即可。
        """
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

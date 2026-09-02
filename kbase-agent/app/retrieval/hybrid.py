from app.config import settings


class Reranker:
    """bge-reranker（opt-in，依赖 torch）。仅在 RERANK_ENABLED=true 时实例化。"""

    def __init__(self, model_name: str = settings.rerank_model):
        self.model_name = model_name
        self.model = None

    def _ensure_model(self) -> None:
        if self.model is None:
            try:
                from FlagEmbedding import FlagReranker
            except ImportError as exc:
                raise RuntimeError(
                    "rerank 需要 FlagEmbedding（torch）。先 pip install -e \".[embed]\""
                ) from exc
            self.model = FlagReranker(self.model_name, use_fp16=False)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
        """对候选按与 query 的相关性重排，返回重新排序后的 top_k 条。"""
        self._ensure_model()
        if not candidates:
            return []
        pairs = [[query, c["content"]] for c in candidates]
        scores = self.model.compute_score(pairs)
        if hasattr(scores, "tolist"):  # numpy
            scores = scores.tolist()
        ranked = sorted(
            zip(candidates, scores), key=lambda pair: pair[1], reverse=True
        )[:top_k]
        return [
            {**candidate, "score": round(float(score), 4)} for candidate, score in ranked
        ]


def hybrid_search(vector_hits: list[dict], keyword_hits: list[dict]) -> list[dict]:
    """向量 + BM25 结果按 chunk_id 合并去重，用 RRF（倒数排名融合）打分。

    输入两侧都应带 chunk_id / content / source；输出按 rrf_score 降序。
    """
    merged: dict[str, dict] = {}
    for hits in (vector_hits, keyword_hits):
        for rank, hit in enumerate(hits, start=1):
            cid = hit.get("chunk_id")
            if not cid:
                continue
            record = merged.setdefault(
                cid,
                {
                    "chunk_id": cid,
                    "content": hit.get("content", ""),
                    "source": hit.get("source", ""),
                    "rrf_score": 0.0,
                },
            )
            record["rrf_score"] += 1.0 / (60 + rank)
    merged_list = sorted(
        merged.values(), key=lambda item: item["rrf_score"], reverse=True
    )
    return merged_list

from app.config import settings


class Reranker:
    """bge-reranker（opt-in，依赖 torch）。仅在 RERANK_ENABLED=true 时实例化。

    OOP 角度：这是真正的"类"——model 是一次性懒加载的重量级资源，实例持有它避免
    每次重排都重建模型（对象即状态；若用模块函数就只能靠全局单例，反而不如类干净）。
    Reranker 与 vector/keyword 检索解耦：pipeline 侧先 hybrid_search() 粗召回，
    再（可选）rerank 精排，是"粗召回->精排序"两段式。
    """

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

    输入两侧都应带 chunk_id / content / source；输出按 rrf_score 降序并去重（同一 chunk_id
    若两路都命中会各贡献一份 rrf_score 叠加，因此自然靠前——这正是 RRF 想要的效果）。

    RRF 公式：score = Σ 1/(k + rank)，k 取常数 60。
    相比"两路 score 加权平均"的优势：
      - 不需要调两路分数如何归一化（向量 cosine 与 BM25 得分量纲完全不同，无法直接相加加权）；
      - 只看"排序位置"不看绝对分，对顺序噪声鲁棒；同一段在两路都排得靠前 => 分最高。
    局限：RRF 会丢失"分数大小"信息（满分 top1A + 半score top3B 可能打平），本项目用它
    做"查得全"的粗召回，语义精度后续交给可选 rerank。
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

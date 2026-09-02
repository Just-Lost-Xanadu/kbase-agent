from app.config import settings


class Embedder:
    """按 EMBED_BACKEND 切换 embedding 实现：
    - fastembed（默认）：ONNX 运行时，免 torch，开箱即用；
    - flagembedding：本地 bge-m3，更准但需 torch（pip install -e ".[embed]"）。
    模型延迟到首次调用才加载，服务启动不被拖慢。
    """

    def __init__(self, backend: str | None = None):
        self.backend = backend or settings.embed_backend
        self.model = None

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        if self.backend == "fastembed":
            from fastembed import TextEmbedding

            self.model = TextEmbedding(model_name=settings.fastembed_model)
        elif self.backend == "flagembedding":
            from FlagEmbedding import BGEM3FlagModel

            self.model = BGEM3FlagModel(settings.embedding_model, use_fp16=False)
        else:
            raise ValueError(
                f"未知 EMBED_BACKEND={self.backend!r}，可选 fastembed / flagembedding"
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure_model()
        if self.backend == "fastembed":
            return [vec.tolist() for vec in self.model.embed(texts)]
        out = self.model.encode(texts, batch_size=32)
        return [vec.tolist() for vec in out["dense_vecs"]]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

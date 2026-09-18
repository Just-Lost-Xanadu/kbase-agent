import json
from pathlib import Path

from app.config import settings
from app.retrieval.chunker import split_documents
from app.retrieval.embedder import Embedder
from app.retrieval.hybrid import Reranker, hybrid_search
from app.retrieval.keyword import BM25Index
from app.retrieval.loader import load_documents
from app.retrieval.vector_store import VectorStore

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCS_DIR = ROOT / "data" / "docs"


class RetrievalPipeline:
    """装配 loader -> chunker -> vector_store(+BM25 sidecar)。

    index() 建索引；检索 = 向量 + BM25 RRF 合并，RERANK_ENABLED=true 时再重排。
    BM25 需要全文，因此建索引时额外落一份 chunks.jsonl sidecar，供后端起进程懒加载，
    避免每次启动都要把整库取出来重算。
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore(self.embedder)
        self.reranker = Reranker() if settings.rerank_enabled else None
        self._bm25: BM25Index | None = None
        self._ready = False

    # ---- 索引存取 -------------------------------------------------------
    @staticmethod
    def _sidecar_path() -> Path:
        return Path(settings.chroma_path) / "chunks.jsonl"

    def is_indexed(self) -> bool:
        try:
            return self._sidecar_path().exists() and self.vector_store.count() > 0
        except Exception:
            return False

    def ensure_ready(self, auto_index: bool = False) -> None:
        """懒加载已有索引；auto_index=True 时若缺失则自动建索引（首次含模型下载）。"""
        if self._ready:
            return
        if self.is_indexed():
            chunks = [
                json.loads(line)
                for line in self._sidecar_path().read_text(encoding="utf-8").splitlines()
            ]
            self._bm25 = BM25Index(chunks)
            self._ready = True
            return
        if auto_index:
            self.index()
            return
        raise RuntimeError(
            "索引不存在：请先运行 python scripts/index_docs.py（或让服务自动建索引）"
        )

    def index(self, source_dir: str | Path | None = None, method: str = "recursive") -> None:
        """全量重建索引。

        注意这里是**全量**语义：每次都把 source_dir 下所有文档重新解析、切分、入库，
        因此必须先清空 collection——否则换切分法（chunk_id 前缀变了）或同名文档变短
        （idx 数量变少）时，旧向量覆盖不到、会永久残留，导致"向量路召回旧切片、
        BM25 只有新切片"的两路口径不一致（详见 VectorStore.reset 的说明）。
        """
        source_dir = source_dir or DEFAULT_DOCS_DIR
        documents = load_documents(source_dir)
        chunks = split_documents(documents, methods=(method,))
        self.vector_store.reset()
        self.vector_store.add(chunks)
        sidecar = self._sidecar_path()
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            "\n".join(
                json.dumps(c, ensure_ascii=False)
                for c in chunks
            ),
            encoding="utf-8",
        )
        self._bm25 = BM25Index(chunks)
        self._ready = True
        print(
            f"[index] {len(documents)} 篇文档 -> {len(chunks)} 个分块 "
            f"({method}), 已写入 {sidecar}"
        )

    # ---- 检索 -----------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        self.ensure_ready()
        k = max(top_k * 3, 10)
        vector_hits = self.vector_store.search(query, top_k=k)
        keyword_hits = self._bm25.search(query, top_k=k)
        merged = hybrid_search(vector_hits, keyword_hits)
        if self.reranker is not None:
            merged = self.reranker.rerank(query, merged, top_k=top_k)
        return merged[:top_k]

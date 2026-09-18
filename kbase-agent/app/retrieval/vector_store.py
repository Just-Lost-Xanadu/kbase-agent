from pathlib import Path

import chromadb

from app.config import settings
from app.retrieval.embedder import Embedder


class VectorStore:
    """Chroma 持久化向量库（开发用；生产可切 Milvus / ES，README 已注明换 collection 层即可）。

    它封装了"对 embedding 结果的存取"，对外只暴露 count/search/add 三个动作——上层的
    RetrievalPipeline 不关心底层是 Chroma 还是别的库，切后端对上层透明（"持久化存储"被隔离成
    一个可替换的组件，而不是散落在各函数里，这是把它做成类而非模块函数的核心理由）。

    懒加载设计：_client/_collection 延迟到首次访问才初始化（chromadb.PersistentClient 会启动
    本地系统），避免 import 阶段就拉起 Chroma——冷启动只在真正需要向量库时才发生。
    embedder 由 __init__ 注入（依赖注入），便于单测时换一个假 embedder。
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self._client = None
        self._collection = None

    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            path = Path(settings.chroma_path)
            path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(path))
        return self._client

    def _ensure_collection(self) -> None:
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=settings.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    def reset(self) -> None:
        """删掉整个 collection（全量重建索引前调用）。

        为什么必须有：chunk_id 是 `source#method#idx`，**带了 method 但不带 chunk_size/overlap**。
        于是两种常见改动会留下覆盖不到的旧向量：
          1. 换切分法（recursive → fixed）：id 前缀不同，upsert 覆盖不到 → 新旧切片长期共存；
          2. 同名文档被替换成更短的版本：新 idx 数量变少 → 尾部旧切片残留。
        后果不只是"多占空间"：向量路会召回旧切片，而 BM25 的 sidecar（chunks.jsonl）是整份
        覆盖写的、只有新切片 —— **两路检索口径不一致，RRF 融合的是两个不同的候选池**，
        用 `recursive`/`fixed` 做对比实验得到的结论直接失效。
        （已实测：先 recursive 再 fixed，向量库 23 条而 sidecar 13 条，库内 method 分布为
        {'recursive': 10, 'fixed': 13}。）
        """
        try:
            self.client.delete_collection(name=settings.collection_name)
        except Exception:  # noqa: BLE001
            # collection 不存在时 delete 会报错——"没东西可删"就是我们要的结果，忽略
            pass
        self._collection = None

    def count(self) -> int:
        try:
            self._ensure_collection()
            return self._collection.count()
        except Exception:
            return 0

    def add(self, chunks: list[dict]) -> None:
        """把分块连同 metadata（source / chunk_id / method）写入向量库。

        用 upsert（按 chunk_id 幂等覆盖）：重复建索引 / 增量新增不会因
        id 冲突报错（Chroma add 对已存在 id 会抛错）。
        """
        self._ensure_collection()
        ids = [c["chunk_id"] for c in chunks]
        documents = [c["content"] for c in chunks]
        embeddings = self.embedder.embed(documents)
        metadatas = [
            {"source": c["source"], "chunk_id": c["chunk_id"], "method": c["method"]}
            for c in chunks
        ]
        self._collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """query -> dense 检索，返回 [{'content','source','chunk_id','distance'}]。"""
        self._ensure_collection()
        query_vec = self.embedder.embed_query(query)
        result = self._collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict] = []
        if result["ids"]:
            for i in range(len(result["ids"][0])):
                meta = result["metadatas"][0][i] or {}
                hits.append(
                    {
                        "content": result["documents"][0][i],
                        "source": meta.get("source", ""),
                        "chunk_id": meta.get("chunk_id", ""),
                        "distance": float(result["distances"][0][i]),
                    }
                )
        return hits

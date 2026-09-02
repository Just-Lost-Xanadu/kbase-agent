from pathlib import Path

import chromadb

from app.config import settings
from app.retrieval.embedder import Embedder


class VectorStore:
    """Chroma 持久化向量库（开发用；生产可切 Milvus / ES，README 已注明换 collection 层即可）。"""

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

    def count(self) -> int:
        try:
            self._ensure_collection()
            return self._collection.count()
        except Exception:
            return 0

    def add(self, chunks: list[dict]) -> None:
        """把分块连同 metadata（source / chunk_id / method）写入向量库。"""
        self._ensure_collection()
        ids = [c["chunk_id"] for c in chunks]
        documents = [c["content"] for c in chunks]
        embeddings = self.embedder.embed(documents)
        metadatas = [
            {"source": c["source"], "chunk_id": c["chunk_id"], "method": c["method"]}
            for c in chunks
        ]
        self._collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

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

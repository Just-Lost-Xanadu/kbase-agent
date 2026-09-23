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

        为什么先判存在、而不是 `delete_collection` 后再 `except Exception: pass`：
        吞掉所有异常只在"collection 本来就不存在"时是等价的；文件被占用/权限/磁盘错误
        同样会抛异常，被吞掉之后 `self._collection = None` 会让随后的 add() 重新
        `get_or_create_collection` 拿到**同一个旧 collection** 去 upsert —— 上面那段
        "旧切片永久残留、两路口径不一致"就会以"日志显示成功"的方式静默发生。
        所以这里只把"不存在"当成功，其余异常照常上抛（失败要能看见）。
        """
        if self._collection_exists():
            self.client.delete_collection(name=settings.collection_name)
        self._collection = None

    def _collection_exists(self) -> bool:
        """collection 是否已存在。列举本身失败属于真实故障，直接上抛，不能当成"不存在"。"""
        return any(
            collection.name == settings.collection_name
            for collection in self.client.list_collections()
        )

    def count(self) -> int:
        """当前 collection 的向量条数。

        这里仍然把"读不出来"和"确实是空"合并成 0，是**有意**的：调用方 `is_indexed()`
        本来就 `except Exception: return False`，两者都会走到"服务自动重建索引"这条
        自愈路径上（等价结果，没必要分两套）。真正需要区分成功/失败的是 reset()，见上。
        """
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

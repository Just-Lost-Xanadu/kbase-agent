import contextlib
import json
import os
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

    def _load_chunks(self) -> list[dict] | None:
        """读 sidecar 解析成分块列表；**任何解析失败都返回 None**（视为"没有可用索引"）。

        为什么要吞异常而不是往上抛：`is_indexed()` 是"服务要不要自动重建索引"的唯一判据。
        如果 sidecar 被截断（index_docs.py 中途 Ctrl-C 就是这种情况）却在这里直接抛
        JSONDecodeError，is_indexed() 会一路 True → ensure_ready() 一路炸 → 服务对每个请求
        都回 503，而且**永远不会触发自动重建**，只能人工介入。这与本模块承诺的
        "索引不存在会自动建"正好相反（实测复现：sidecar 末行截断 → JSONDecodeError 常驻）。
        所以这里的口径是：读不出来 = 没有 = 让上层去重建。
        sidecar 现在也是原子发布的（见 index()），正常路径不会再产生半截文件。
        """
        try:
            raw = self._sidecar_path().read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        chunks: list[dict] = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError:
                return None
        return chunks or None

    def is_indexed(self) -> bool:
        try:
            return self._load_chunks() is not None and self.vector_store.count() > 0
        except Exception:  # noqa: BLE001  # 任何读取失败都等价于“没有索引”，交给上层自动重建（见 _load_chunks 说明）
            return False

    def ensure_ready(self, auto_index: bool = False) -> None:
        """懒加载已有索引；auto_index=True 时若缺失则自动建索引（首次含模型下载）。"""
        if self._ready:
            return
        chunks = self._load_chunks()
        if chunks:
            if self.vector_store.count() <= 0:
                raise RuntimeError(
                    "索引不一致：分块清单存在但向量库为空。请重跑 python scripts/index_docs.py"
                )
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
        if not chunks:
            # 明确的报错，而不是把空列表塞给 BM25Index——rank_bm25 对空语料会除零，
            # 抛出来的 ZeroDivisionError 完全看不出是"语料是空的"。
            raise RuntimeError(
                f"切分结果为空（文档 {len(documents)} 篇）：检查 data/docs 下文档是否有内容"
            )
        # 破坏性操作（reset）之前，先把 embedding 模型加载起来——首次运行会下载模型，需要网络。
        #
        # 为什么必须前置：模型加载原本发生在 add() 里，而 add() 在 reset() **之后**。
        # 于是模型加载/下载失败时（离线、fastembed 缓存被清理、代理没开）向量库已经被清空，
        # 而 sidecar 还是旧的 → is_indexed()（sidecar 可读 + count>0）为 False，
        # ensure_ready() 立刻抛"索引不一致：分块清单存在但向量库为空"，
        # **服务对每个请求都失败且不会自愈**（自动重建只在"读不到 sidecar"时才触发，
        # 而 sidecar 完好）。已实测复现：清掉模型缓存后 index_docs.py 一跑，
        # 旧索引被清空、sidecar 留着 40 行，随后 eval.py 与 /api/chat 全部失败。
        #
        # 前置之后失败发生在任何破坏之前：旧索引原封不动，恢复网络重跑即可。
        # 用 getattr 探测而不是直接调 self.embedder.embed：单测会用哨兵对象顶替 embedder
        # 来单独验证"reset 先于 add"的顺序，那条路径不该被模型加载绊住。
        _warm_up = getattr(self.embedder, "embed", None)
        if callable(_warm_up):
            _warm_up([chunks[0]["content"]])
        sidecar = self._sidecar_path()
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        # 先落临时文件再原子替换：直接把 json 一行行写进 chunks.jsonl 的话，
        # 中途中断（Ctrl-C、磁盘满、进程被杀）会留下**半截文件**——那正是上面
        # _load_chunks 要兜的损坏态。os.replace 在同一文件系统内是原子的：
        # 要么看到旧文件、要么看到完整的新文件，不存在"读了一半"的中间态。
        tmp = sidecar.with_suffix(".jsonl.tmp")
        tmp.write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks),
            encoding="utf-8",
        )
        self.vector_store.reset()
        self.vector_store.add(chunks)
        try:
            os.replace(tmp, sidecar)   # 向量写成功之后才发布 sidecar
        except OSError:
            # 这是本条链上唯一可能失败的一步（Windows 下目标文件被别的进程占用、磁盘满等），
            # 而 MCP 子进程每次工具调用都会读这个 sidecar。失败时的状态是"collection 已是新分块、
            # sidecar 还是旧的"——两路检索口径不一致，而 is_indexed()（sidecar 可读 + count>0）
            # 仍返回 True，服务不会自愈。所以把旧 sidecar 一并删掉：is_indexed() 变 False，
            # 下一个请求就会自动重建，而不是把错乱状态一直留在盘上。
            with contextlib.suppress(OSError):
                sidecar.unlink()
            raise
        self._bm25 = BM25Index(chunks)
        self._ready = True
        print(
            f"[index] {len(documents)} 篇文档 -> {len(chunks)} 个分块 "
            f"({method}), 已写入 {sidecar}"
        )

    # ---- 检索 -----------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 3, route: str = "hybrid") -> list[dict]:
        """检索入口。

        route 是给**消融实验**用的（scripts/eval.py --route），默认行为与旧版完全一致：
          - "hybrid"（默认）：向量 + BM25 各粗召回 `max(top_k * 3, 10)` 条 → RRF 融合 → 取 top_k，
            开了 RERANK_ENABLED 时再重排；
          - "vector" / "bm25"：只走单路（不粗召回、不融合）——这正是 README「检索路消融」表里
            "仅向量 0.875 / 仅 BM25 0.964"的口径。
        """
        self.ensure_ready()
        if route == "vector":
            return self.vector_store.search(query, top_k=top_k)
        if route == "bm25":
            return self._bm25.search(query, top_k=top_k)
        if route != "hybrid":
            raise ValueError(f"未知 route={route!r}，可选 hybrid / vector / bm25")
        k = max(top_k * 3, 10)
        vector_hits = self.vector_store.search(query, top_k=k)
        keyword_hits = self._bm25.search(query, top_k=k)
        merged = hybrid_search(vector_hits, keyword_hits)
        if self.reranker is not None:
            merged = self.reranker.rerank(query, merged, top_k=top_k)
        return merged[:top_k]

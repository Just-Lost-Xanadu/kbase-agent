from langchain_text_splitters import RecursiveCharacterTextSplitter


def fixed_size_chunk(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """固定长度：按字符窗口滑切（无启发式，只按长度截断），预留 overlap 防断句丢信息。

    与 recursive 比：实现最简单、可复现；代价是可能把完整句子/概念拦腰切断，语义衔接不如
    recursive。适合用来跟 recursive 做"调参对比实验"（chunk_size/overlap 变化的影响）。
    overlap(150) 让相邻窗口共享重叠段，降低"关键句恰好落在切缝"导致的召回漏失。
    step = chunk_size - overlap 是滑窗步长（每段前进这么多，窗口 itself 仍是 chunk_size 长）。
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size 必须大于 overlap")
    if not text:
        return []
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size]
        if piece.strip():
            chunks.append(piece)
    return chunks


def recursive_chunk(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """递归切分：按分隔符优先级（段落/句号/逗号…）切到不超过 chunk_size，语义更完整。

    用 langchain RecursiveCharacterTextSplitter：先按最有语义边界的大分隔符（如换行段落、
    句号/问号）备选切，若单块仍超长再递归到更细的分隔符（逗号/空格），最终尽量让每块
    停在语义边界上而非硬切一半。本项目默认走 recursive，因为它比 fixed 在"一个 chunk 尽量
    讲完一个完整论点"上更稳（对问答召回有利）。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    return splitter.split_text(text)


def split_documents(
    documents: list[dict], methods: tuple[str, ...] = ("recursive",)
) -> list[dict]:
    """对每个文档跑指定切分法，产出带 chunk_id / source / method 的分块。

    入参 documents：每个 {'content': 原文, 'source': 文档名}；由 loader 产出。
    返回 list[dict]，每块：
      - content : 该段文本
      - source  : 来自哪个文档（用于引用溯源【来源：...】）
      - method  : 用 fixed 还是 recursive 切（默认 recursive）
      - chunk_id: f"{source}#{method}#{idx}" —— 唯一、稳定。它在 vector_store 层用作 Chroma 的 id
                  （配合 upsert 幂等覆盖），也在 hybrid 合并两侧命中时按它去重相加 RRF 分。
    一个 chunk_id 恒有名 => 重建/增量只会覆盖同名旧 chunk，不会造成 id 冲突（这与"换 embedding
    后必须删库重建"不同：换模型会改变语义但 chunk_id 不变，因此必须整库重建，见 README 坑位）。
    """
    chunkers = {
        "fixed": fixed_size_chunk,
        "recursive": recursive_chunk,
    }
    unknown = set(methods) - set(chunkers)
    if unknown:
        raise ValueError(f"未知切分法: {unknown}，可选 {sorted(chunkers)}")
    chunks: list[dict] = []
    for doc in documents:
        text, source = doc["content"], doc["source"]
        for method in methods:
            pieces = chunkers[method](text)
            for idx, piece in enumerate(pieces):
                chunks.append(
                    {
                        "content": piece,
                        "source": source,
                        "method": method,
                        "chunk_id": f"{source}#{method}#{idx}",
                    }
                )
    return chunks

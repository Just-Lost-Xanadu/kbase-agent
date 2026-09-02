from langchain_text_splitters import RecursiveCharacterTextSplitter


def fixed_size_chunk(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """固定长度：按字符窗口滑切（无启发式，只按长度截断），预留 overlap 防断句丢信息。"""
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
    """递归切分：按分隔符优先级（段落/句号/逗号…）切到不超过 chunk_size，语义更完整。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    return splitter.split_text(text)


def split_documents(
    documents: list[dict], methods: tuple[str, ...] = ("recursive",)
) -> list[dict]:
    """对每个文档跑指定切分法，产出带 chunk_id / source / method 的分块。"""
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

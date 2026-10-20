#!/bin/sh
set -eu

STATE_DIR="${KBASE_STATE_DIR:-/app/state}"
CHROMA_DIR="${CHROMA_PATH:-$STATE_DIR/chroma}"
CACHE_DIR="${FASTEMBED_CACHE_DIR:-$STATE_DIR/fastembed_cache}"

mkdir -p "$STATE_DIR" "$CHROMA_DIR" "$CACHE_DIR"

# 首次启动（或卷被清空）时自动建索引。建索引会下载 embedding 模型，
# 所以第一次启动需要网络；之后模型和索引都在卷里，重启是秒级。
if [ ! -f "$CHROMA_DIR/chunks.jsonl" ]; then
    echo "[entrypoint] 未发现索引 sidecar，开始建索引（首次会下载模型）..."
    python scripts/index_docs.py
fi

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"

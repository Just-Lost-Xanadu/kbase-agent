# kbase-agent 运行镜像：只装运行期依赖，不包含测试/评测报告/本地索引。
# 索引与模型缓存放在 /app/state 卷里，容器重建不丢；首次启动会自动建索引。
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# onnxruntime（FastEmbed 的底层）在 slim 上需要 libgomp1；不装会在 import 时报
# "libgomp.so.1: cannot open shared object file"。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# requirements.lock 是在 Windows 上导出的，包含 pywin32 / win32_setctime 两个
# Windows-only 包；Linux 上装不上，且本项目运行时不需要它们。这里显式过滤，
# 其余版本仍然按 lock 锁定，避免"容器里装出一套没验证过的版本组合"。
COPY requirements.lock ./
RUN grep -vE '^(pywin32|win32_setctime)==' requirements.lock > /tmp/requirements.lock \
    && pip install --no-cache-dir -r /tmp/requirements.lock

COPY pyproject.toml README.md ./
COPY app ./app
COPY static ./static
COPY scripts ./scripts
COPY data/docs ./data/docs
COPY data/business ./data/business

# 依赖已由 lock 装好，这里只装本项目自身（--no-deps 避免 pip 再解析一遍版本）。
RUN pip install --no-cache-dir -e . --no-deps

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

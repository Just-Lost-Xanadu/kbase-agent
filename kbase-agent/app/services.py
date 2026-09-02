"""服务端资源：检索管道 + Agent runtime 的懒加载与进程内缓存。

不放 lifespan 是为了启动轻、失败可恢复：第一次收到对话请求才建索引/拉起 MCP 子进程。
"""

import asyncio
import logging

from fastapi import FastAPI

logger = logging.getLogger(__name__)


def _services_ready(app: FastAPI) -> bool:
    return bool(getattr(app.state, "services_ok", False))


async def ensure_services(app: FastAPI):
    """返回 (pipeline, runtime)；重复调用幂等，只初始化一次。"""
    if _services_ready(app):
        return app.state.pipeline, app.state.runtime

    lock = getattr(app.state, "services_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        app.state.services_lock = lock
    async with lock:
        if _services_ready(app):
            return app.state.pipeline, app.state.runtime

        # 1) 检索管道：已有索引直接懒加载；没有则自动建（首次含 embedding 模型下载）
        from app.retrieval.pipeline import RetrievalPipeline

        pipeline = RetrievalPipeline()
        if not pipeline.is_indexed():
            logger.info("检测到空索引，自动建索引（首次使用会下载 embedding 模型）…")
            pipeline.index()
        pipeline.ensure_ready()

        # 2) 把管道挂给进程内 MCP 工具定义（调试/直连时用）
        from app.mcp import servers as mcp_servers

        mcp_servers.set_pipeline(pipeline)

        # 3) Agent runtime：拉起 MCP stdio 子进程并编译 LangGraph
        from app.agent.graph import create_runtime

        try:
            runtime = await create_runtime()
        except Exception as exc:
            logger.warning(
                "Agent runtime 初始化失败（检查 DEEPSEEK_API_KEY 与网络）：%s", exc
            )
            raise RuntimeError(
                "Agent 引擎未就绪：请确认 .env 已填 DEEPSEEK_API_KEY，"
                "并已 pip install -e '.[dev]'（含 mcp/adapters 依赖）"
            ) from exc

        app.state.pipeline = pipeline
        app.state.runtime = runtime
        app.state.services_ok = True
        return pipeline, runtime

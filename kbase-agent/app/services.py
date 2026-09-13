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

        # 注意：is_indexed()/index()/ensure_ready() 都是同步阻塞调用（读文档、embedding、
        # 写 Chroma，首次还会下载模型）。直接在事件循环里 await 会卡住整个服务，
        # 必须丢进线程执行。
        pipeline = RetrievalPipeline()
        if not await asyncio.to_thread(pipeline.is_indexed):
            logger.info("检测到空索引，自动建索引（首次使用会下载 embedding 模型）…")
            await asyncio.to_thread(pipeline.index)
        await asyncio.to_thread(pipeline.ensure_ready)

        # 2) 把管道挂给进程内 MCP 工具定义（调试/直连时用）
        from app.mcp import servers as mcp_servers

        mcp_servers.set_pipeline(pipeline)

        # 3) Agent runtime：拉起 MCP stdio 子进程并编译 LangGraph
        from app.agent.graph import create_runtime

        try:
            runtime = await create_runtime()
        except Exception as exc:
            logger.warning("Agent runtime 初始化失败：%s: %s", type(exc).__name__, exc)
            raise RuntimeError(
                "Agent 引擎未就绪。常见原因：① .env 里没有有效的 DEEPSEEK_API_KEY；"
                "② 依赖缺失，执行 pip install -e . 即可（mcp / langchain-mcp-adapters 是基础依赖，"
                ".[dev] 只额外装 pytest/httpx）；③ MCP 子进程启动或 checkpoint 库异常。"
                f"原始错误：{type(exc).__name__}: {exc}"
            ) from exc

        app.state.pipeline = pipeline
        app.state.runtime = runtime
        app.state.services_ok = True
        return pipeline, runtime

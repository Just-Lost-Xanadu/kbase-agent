"""服务端资源：检索管道 + Agent runtime 的懒加载与进程内缓存。

不放 lifespan 是为了启动轻、失败可恢复：第一次收到对话请求才建索引/拉起 MCP 子进程。
"""

import asyncio
import logging

from fastapi import FastAPI

logger = logging.getLogger(__name__)


def _services_ready(app: FastAPI) -> bool:
    return bool(getattr(app.state, "services_ok", False))


def _consume_task_exception(task: asyncio.Task) -> None:
    """取一次异常，避免"Task exception was never retrieved"告警。

    只是"取走"以便记录，并不会清掉异常——之后 await 这个 task 的人照样会收到它。
    """
    if not task.cancelled():
        task.exception()


async def _init_services(app: FastAPI):
    """真正做初始化的协程（只由 ensure_services 以"单飞 task"方式启动）。"""
    # 1) 检索管道：已有索引直接懒加载；没有则自动建（首次含 embedding 模型下载）
    from app.retrieval.pipeline import RetrievalPipeline

    # 注意：is_indexed()/index()/ensure_ready() 都是同步阻塞调用（读文档、embedding、
    # 写 Chroma，首次还会下载模型）。直接在事件循环里 await 会卡住整个服务，
    # 必须丢进线程执行。
    #
    # 复用 app.state 上已有的实例：create_runtime() 失败时（最常见是没配 key）已经加载好的
    # 管道不必在下一次请求里重读 chunks.jsonl、重建 BM25 与 checkpointer 连接。
    pipeline = getattr(app.state, "pipeline", None) or RetrievalPipeline()
    if not await asyncio.to_thread(pipeline.is_indexed):
        logger.info("检测到空索引，自动建索引（首次使用会下载 embedding 模型）…")
        await asyncio.to_thread(pipeline.index)
    await asyncio.to_thread(pipeline.ensure_ready)
    app.state.pipeline = pipeline   # 先落状态：即便下面 runtime 失败，管道也不白建

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

    app.state.runtime = runtime
    app.state.services_ok = True
    return pipeline, runtime


async def ensure_services(app: FastAPI):
    """返回 (pipeline, runtime)；重复调用幂等，只初始化一次。

    为什么用"单飞 task"而不是直接 `async with lock:` 里跑初始化：
    `asyncio.to_thread` 的取消只是"放弃 await"，**线程会继续跑到底**。首次请求要下载
    embedding 模型、耗时以分钟计，而前端是 SSE 长连接页面、用户刷新/关页很常见。那时
    CancelledError 会立刻从 await 抛出、`async with lock` 随之退出（services_ok 还没置位），
    可线程仍在写 chunks.jsonl 并 reset/add 向量库；下一个请求重新进临界区（锁已空）、
    `is_indexed()` 仍为 False → **第二次全量 index()**：两个线程同时 reset/add 同一个
    collection、同时写同一个 chunks.jsonl.tmp，留下半截 sidecar 与对不上的分块/向量数。
    把初始化放进 task 存到 app.state，await 它的人被取消**不会取消初始化本身**，
    后续请求拿到的是同一个 task（await 同一个 task 不会重复执行）。
    """
    if _services_ready(app):
        return app.state.pipeline, app.state.runtime

    lock = getattr(app.state, "services_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        app.state.services_lock = lock
    async with lock:
        if _services_ready(app):
            return app.state.pipeline, app.state.runtime
        task = getattr(app.state, "services_task", None)
        if task is None or task.done():
            # 上一次失败（done 但 services_ok 仍为 False）→ 重建一个，保留"每次请求都可重试"语义
            task = asyncio.create_task(_init_services(app))
            task.add_done_callback(_consume_task_exception)
            app.state.services_task = task
    try:
        # shield：调用方被取消时只放弃等待，初始化 task 照常跑完
        return await asyncio.shield(task)
    except Exception:
        app.state.services_task = None   # 失败清空，下一次请求重新初始化
        raise

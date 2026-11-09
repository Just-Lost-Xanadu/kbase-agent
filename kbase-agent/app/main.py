import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.chat import router as chat_router
from app.auth import BasicAuthMiddleware

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 建会话记录表（与 checkpoint 共用 data/checkpoints.sqlite）；Agent 资源仍懒加载
    from app import store

    await store.init_db()
    yield
    # 关停顺序很重要：先收掉"还在跑的懒初始化"，再关 runtime。
    # 旧实现只 aclose 了 app.state.runtime：若此刻 services_task 正在建索引/拉 MCP 子进程，
    # 它会在 runtime 关闭**之后**把这个新 runtime 写进 app.state（app.state.runtime = runtime），
    # 于是新拉起的 MCP 子进程与 sqlite 连接再也没人关，日志里还会多一条
    # "Task was destroyed but it is pending!"。cancel 之后必须 await（吞掉 CancelledError），
    # 否则 cancel 只是"提了个请求"，任务可能还没退。
    task = getattr(app.state, "services_task", None)
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass  # 这是我们自己 cancel 的，不是外层取消
        except Exception:  # noqa: BLE001
            pass
    runtime = getattr(app.state, "runtime", None)
    if runtime is not None:
        try:
            await runtime.aclose()
        except Exception:  # noqa: BLE001, S110
            pass


app = FastAPI(title="kbase-agent", version="0.1.0", lifespan=lifespan)

# 最小鉴权：设了 APP_AUTH_USER/APP_AUTH_PASS 就全站启用（/api/health 除外）。
# 不设则保持原来的本地免鉴权行为，避免影响现有测试与本地开发。
app.add_middleware(BasicAuthMiddleware)

# 刻意不加 CORSMiddleware：演示前端由本服务在 / 同源提供，同源请求不需要 CORS；
# 服务现在有 HTTP Basic 鉴权，但**没有跨源授权模型**：一旦开 `allow_origins=["*"]`，
# 别的站点仍可能借用户浏览器跨域读走 /api/sessions/{id}/messages 与 /api/runs。
# 真要开放给别的源，应该同时收紧 Origin、上 TLS，并保留鉴权。
app.include_router(chat_router, prefix="/api")


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    """演示前端：单文件页面（static/index.html），打开即聊，无构建。"""
    return FileResponse(STATIC_DIR / "index.html")

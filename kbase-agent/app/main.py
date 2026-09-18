from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.chat import router as chat_router

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 建会话记录表（与 checkpoint 共用 data/checkpoints.sqlite）；Agent 资源仍懒加载
    from app import store

    await store.init_db()
    yield
    runtime = getattr(app.state, "runtime", None)
    if runtime is not None:
        try:
            await runtime.aclose()
        except Exception:  # noqa: BLE001
            pass


app = FastAPI(title="kbase-agent", version="0.1.0", lifespan=lifespan)

# 刻意不加 CORSMiddleware：演示前端由本服务在 / 同源提供，同源请求不需要 CORS；
# 而本服务**没有任何鉴权**，一旦开 `allow_origins=["*"]`，任意网站都能从用户浏览器里
# 跨域读走 /api/sessions/{id}/messages 与 /api/runs（历史对话与运行 trace）。
# 真要开放给别的源，应该同时补鉴权，并把 allow_origins 收敛到具体域名。
app.include_router(chat_router, prefix="/api")


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    """演示前端：单文件页面（static/index.html），打开即聊，无构建。"""
    return FileResponse(STATIC_DIR / "index.html")

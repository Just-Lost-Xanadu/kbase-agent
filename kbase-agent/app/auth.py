"""最小鉴权：HTTP Basic Auth。

定位：**演示/内网部署的最低门槛**，不是完整的账号体系。
- 只做"一个共享账号 + 密码"，没有用户表、角色、审计、限流；
- 依赖 HTTPS 才有意义。裸 HTTP 下 Basic 头是可解码的，只能防"误点链接的人"，
  不能防抓包。真要公网部署，前面必须加 TLS（反向代理或云负载均衡）。
- `/api/health` 保持公开，供容器 healthcheck / 探针使用；其余路径默认都要认证。

为什么用纯 ASGI 中间件而不是 `BaseHTTPMiddleware`：
`/api/chat/stream` 是 SSE 长连接，`BaseHTTPMiddleware` 会多包一层 anyio 流，
在部分版本上会改变流式时序。纯 ASGI 只在入口判断一次，认证通过后原样交给下游。
"""

from __future__ import annotations

import base64
import binascii
import os
import secrets

from starlette.datastructures import Headers
from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send

# 允许单独导入本模块时也能读到 .env（正常路径由 app.config 先 load_dotenv）。
from dotenv import load_dotenv

load_dotenv()

PUBLIC_PATHS = {"/api/health"}

AUTH_USER = os.getenv("APP_AUTH_USER", "").strip()
AUTH_PASS = os.getenv("APP_AUTH_PASS", "").strip()
AUTH_ENABLED = bool(AUTH_USER or AUTH_PASS)
if AUTH_ENABLED and not (AUTH_USER and AUTH_PASS):
    # 只配一半是最危险的状态：以为开了鉴权，实际无法登录；
    # 或者以为没开，实际把一个空密码暴露出去。在导入期直接失败，别等到第一个请求。
    raise RuntimeError(
        "APP_AUTH_USER 与 APP_AUTH_PASS 必须同时设置；"
        "只想本地免鉴权就两个都不设"
    )


class BasicAuthMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.username = AUTH_USER
        self.password = AUTH_PASS
        self.enabled = AUTH_ENABLED

    def _authorized(self, header: str) -> bool:
        scheme, _, payload = header.partition(" ")
        if scheme.lower() != "basic" or not payload:
            return False
        try:
            decoded = base64.b64decode(payload, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError):
            return False
        user, sep, password = decoded.partition(":")
        if not sep:
            return False
        # 两个比较都做，避免用短路暴露"用户名是否存在"的时序差异。
        user_ok = secrets.compare_digest(user, self.username)
        pass_ok = secrets.compare_digest(password, self.password)
        return user_ok and pass_ok

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            not self.enabled
            or scope["type"] != "http"
            or scope.get("path") in PUBLIC_PATHS
            or scope.get("method") == "OPTIONS"
        ):
            await self.app(scope, receive, send)
            return

        header = Headers(scope=scope).get("authorization", "")
        if self._authorized(header):
            await self.app(scope, receive, send)
            return

        response = Response(
            "Unauthorized\n",
            status_code=401,
            media_type="text/plain; charset=utf-8",
            headers={"WWW-Authenticate": 'Basic realm="kbase-agent", charset="UTF-8"'},
        )
        await response(scope, receive, send)

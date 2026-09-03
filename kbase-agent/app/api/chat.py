import json

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from app.api.schemas import ChatRequest, ChatResponse
from app.services import ensure_services

router = APIRouter()


async def _record_turn(req: ChatRequest, answer: str, sources: list[str]) -> None:
    """把一轮对话写入会话记录表（仅显式带 session_id 的请求；供读历史接口用）。"""
    if not req.session_id:
        return
    from app import store

    last_user = next(
        (m.content for m in reversed(req.messages) if m.role in {"user", "human"}),
        "",
    )
    await store.save_turn(
        req.session_id,
        title=(last_user or "新会话")[:40],
        user_content=last_user,
        answer=answer,
        sources=sources,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    try:
        _, runtime = await ensure_services(request.app)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        result = await runtime.ainvoke(
            [m.model_dump() for m in req.messages], session_id=req.session_id
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Agent 执行失败：{exc}") from exc
    try:
        await _record_turn(req, result["answer"], result["sources"])
    except Exception:  # noqa: BLE001
        pass  # 记录失败不影响主流程回答
    return ChatResponse(answer=result["answer"], sources=result["sources"])


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    """SSE 流式：逐步下发各节点增量，收尾事件带最终答案与引用。"""
    try:
        _, runtime = await ensure_services(request.app)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    async def event_stream():
        try:
            async for update in runtime.astream(
                [m.model_dump() for m in req.messages], session_id=req.session_id
            ):
                if "answer" in update:  # 收尾事件
                    try:
                        await _record_turn(req, update["answer"], update.get("sources", []))
                    except Exception:  # noqa: BLE001
                        pass
                    yield {"event": "done", "data": json.dumps(update, ensure_ascii=False)}
                    continue
                for node, payload in update.items():
                    event: dict = {"type": "step", "node": node}
                    messages = payload.get("messages") or []
                    if messages:
                        last = messages[-1]
                        if getattr(last, "content", None):
                            event["text"] = str(last.content)[:500]
                        if getattr(last, "tool_calls", None):
                            event["tool_calls"] = [
                                {"name": c.get("name"), "args": c.get("args")}
                                for c in last.tool_calls
                            ]
                    yield {"event": "message", "data": json.dumps(event, ensure_ascii=False)}
        except Exception as exc:  # noqa: BLE001
            yield {
                "event": "error",
                "data": json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False),
            }
            return
        yield {"event": "end", "data": "[DONE]"}

    return EventSourceResponse(event_stream())


@router.get("/health", tags=["ops"])
async def health() -> dict:
    return {"status": "ok"}


@router.get("/sessions", tags=["history"])
async def list_sessions(limit: int = 50) -> list[dict]:
    """会话列表（倒序），供前端"历史会话"侧栏。"""
    from app import store

    return await store.list_sessions(limit=max(1, min(limit, 200)))


@router.get("/sessions/{session_id}/messages", tags=["history"])
async def get_session_messages(session_id: str) -> list[dict]:
    """某会话的全部消息历史（含来源引用），供前端渲染。"""
    from app import store

    return await store.get_messages(session_id)

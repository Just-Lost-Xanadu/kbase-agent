import json

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from app.api.schemas import ChatRequest, ChatResponse
from app.services import ensure_services

router = APIRouter()


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

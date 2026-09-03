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


def _question_of(req: ChatRequest) -> str:
    return next(
        (m.content for m in reversed(req.messages) if m.role in {"user", "human"}),
        "",
    )


async def _save_trace(req: ChatRequest, recorder_summary: dict, answer: str, sources: list[str]) -> None:
    """持久化一次运行的 trace（观测用，失败不影响回答）。"""
    from app import store

    await store.save_trace(
        conversation_id=req.session_id,
        recorder_summary=recorder_summary,
        answer=answer,
        sources=sources,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    from app.observability import Recorder, reset_recorder, set_recorder

    try:
        _, runtime = await ensure_services(request.app)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    rec = Recorder(question=_question_of(req))
    token = set_recorder(rec)
    try:
        result = await runtime.ainvoke(
            [m.model_dump() for m in req.messages], session_id=req.session_id
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        rec.error = str(exc)[:300]
        try:
            await _save_trace(req, rec.summarize(), "", [])
        except Exception:  # noqa: BLE001
            pass
        raise HTTPException(status_code=502, detail=f"Agent 执行失败：{exc}") from exc
    finally:
        reset_recorder(token)
    try:
        await _record_turn(req, result["answer"], result["sources"])
        await _save_trace(req, rec.summarize(), result["answer"], result["sources"])
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
        from app.observability import Recorder, reset_recorder, set_recorder

        rec = Recorder(question=_question_of(req))
        token = set_recorder(rec)
        try:
            async for update in runtime.astream(
                [m.model_dump() for m in req.messages], session_id=req.session_id
            ):
                if "answer" in update:  # 收尾事件
                    try:
                        await _record_turn(req, update["answer"], update.get("sources", []))
                        await _save_trace(req, rec.summarize(), update["answer"], update.get("sources", []))
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
            rec.error = str(exc)[:300]
            try:
                await _save_trace(req, rec.summarize(), "", [])
            except Exception:  # noqa: BLE001
                pass
            yield {
                "event": "error",
                "data": json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False),
            }
            return
        finally:
            reset_recorder(token)
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


@router.get("/runs", tags=["ops"])
async def list_runs(limit: int = 50) -> list[dict]:
    """运行 trace 列表（倒序），可观测用：耗时/成本/token/错误。"""
    from app import store

    return await store.list_traces(limit=max(1, min(limit, 200)))


@router.get("/runs/{trace_id}", tags=["ops"])
async def get_run(trace_id: int) -> dict:
    """单次运行的完整 trace（含节点级 steps 明细）。"""
    from app import store

    trace = await store.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="trace 不存在")
    return trace

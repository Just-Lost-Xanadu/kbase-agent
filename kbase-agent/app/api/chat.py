import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.services import ensure_services

router = APIRouter()

# 断连补写任务池：仅用于持有强引用（asyncio 对 task 只保留弱引用），任务结束即移除
_finalizer_tasks: set = set()


# ---- 请求/响应模型（原 app/api/schemas.py，并入本文件：仅 /chat 系列接口使用）----
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(description="历史消息 + 最新用户输入")
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list, description="引用来源")


def _question_of(req: ChatRequest) -> str:
    """取本轮用户问题 = 最后一条 user 消息。

    同一段逻辑原先在 _record_turn 里重复了一份，现统一走这里，避免两处口径漂移。
    """
    return next(
        (m.content for m in reversed(req.messages) if m.role in {"user", "human"}),
        "",
    )


async def _record_turn(req: ChatRequest, answer: str, sources: list[str]) -> None:
    """把一轮对话写入会话记录表（仅显式带 session_id 的请求；供读历史接口用）。"""
    if not req.session_id:
        return
    from app import store

    last_user = _question_of(req)
    await store.save_turn(
        req.session_id,
        title=(last_user or "新会话")[:40],
        user_content=last_user,
        answer=answer,
        sources=sources,
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


async def _persist(req: ChatRequest, rec, result: dict | None) -> None:
    """落库：有答案才写会话记录，trace 则无论成败都写（记录失败不影响主流程）。

    result=None 表示本轮没有产出答案（执行失败或客户端断连）——此时只写 trace，
    保持原有语义（历史里不出现"只有问题没有答案"的轮次）。
    """
    answer = (result or {}).get("answer", "")
    sources = (result or {}).get("sources", []) or []
    if result is not None:
        try:
            await _record_turn(req, answer, sources)
        except Exception:  # noqa: BLE001
            pass
    try:
        await _save_trace(req, rec.summarize(), answer, sources)
    except Exception:  # noqa: BLE001
        pass


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    from app.observability import Recorder, reset_recorder, set_recorder

    try:
        _, runtime = await ensure_services(request.app)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    rec = Recorder(question=_question_of(req))
    token = set_recorder(rec)
    result: dict | None = None
    try:
        result = await runtime.ainvoke(
            [m.model_dump() for m in req.messages], session_id=req.session_id
        )
    except HTTPException:
        raise
    except asyncio.CancelledError:
        # 客户端中途断开：checkpoint 可能已由 graph 提交，trace 不能丢。
        # CancelledError 属 BaseException，不会被下面的 except Exception 捕获，故单列；
        # shield 保证这次写入不被取消打断。
        await asyncio.shield(_persist(req, rec, None))
        raise
    except Exception as exc:  # noqa: BLE001
        rec.error = str(exc)[:300]
        await asyncio.shield(_persist(req, rec, None))
        raise HTTPException(status_code=502, detail=f"Agent 执行失败：{exc}") from exc
    finally:
        reset_recorder(token)
    # 成功路径同样 shield：答案已产出，此刻若客户端断开也不能丢掉这次 trace
    await asyncio.shield(_persist(req, rec, result))
    return ChatResponse(answer=result["answer"], sources=result["sources"])


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    """SSE 流式：逐步下发各节点增量，收尾事件带最终答案与引用。"""
    try:
        _, runtime = await ensure_services(request.app)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    async def event_stream():
        from app.agent.graph import _text_of
        from app.observability import Recorder, reset_recorder, set_recorder

        rec = Recorder(question=_question_of(req))
        token = set_recorder(rec)
        persisted = False
        try:
            async for update in runtime.astream(
                [m.model_dump() for m in req.messages], session_id=req.session_id
            ):
                if "answer" in update:  # 收尾事件
                    # 先置位再写入：若这次写入途中被取消，finally 里不能再补一次，
                    # 否则同一次运行会落两条 trace（后一条答案为空）。
                    # shield 与同步端点同理：答案已产出，此刻客户端断开也不能丢掉这次 trace。
                    persisted = True
                    await asyncio.shield(_persist(req, rec, update))
                    yield {"event": "done", "data": json.dumps(update, ensure_ascii=False)}
                    continue
                for node, payload in update.items():
                    event: dict = {"type": "step", "node": node}
                    messages = payload.get("messages") or []
                    if messages:
                        last = messages[-1]
                        # 只把 agent 节点文本上行；tools 节点的 ToolMessage 原文
                        # 含工具输出，可能很大或含内部信息，不下发给前端
                        if node == "agent" and getattr(last, "content", None):
                            # 与 graph._text_of 同口径：content 可能是 content block 列表，
                            # 直接 str() 会把 Python repr 发给前端
                            event["text"] = _text_of(last.content)[:500]
                        if getattr(last, "tool_calls", None):
                            event["tool_calls"] = [
                                {"name": c.get("name"), "args": c.get("args")}
                                for c in last.tool_calls
                            ]
                    yield {"event": "message", "data": json.dumps(event, ensure_ascii=False)}
        except Exception as exc:  # noqa: BLE001
            rec.error = str(exc)[:300]
            # 与成功分支同样"先置位再写"：若这次写入途中被取消（CancelledError 属 BaseException，
            # 不会被上面的 except 捕获），finally 里 `not persisted` 会再排一个后台 _persist，
            # 于是同一次运行落两条 trace（后一条答案为空）。shield 则保证写入本身不被取消打断。
            persisted = True
            await asyncio.shield(_persist(req, rec, None))
            yield {
                "event": "error",
                "data": json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False),
            }
            return
        finally:
            reset_recorder(token)
            if not persisted:
                # 客户端断连时生成器被关闭（GeneratorExit），finally 里不能 await，
                # 交给后台任务补写 trace，避免这次运行在 /api/runs 里查不到。
                # 必须持有强引用：asyncio 只保留弱引用，任务可能在完成前被 GC 掉。
                try:
                    task = asyncio.create_task(_persist(req, rec, None))
                except RuntimeError:
                    pass  # 事件循环已关闭，补写无望
                else:
                    _finalizer_tasks.add(task)
                    task.add_done_callback(_finalizer_tasks.discard)
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

import asyncio
import json
import uuid
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, model_validator
from sse_starlette.sse import EventSourceResponse

from app.services import ensure_services

router = APIRouter()

# 断连补写任务池：仅用于持有强引用（asyncio 对 task 只保留弱引用），任务结束即移除
_finalizer_tasks: set = set()

# 单次请求的消息内容总字符上限：护栏只管了"工具输出截断"，用户输入此前是无上限的——
# 实测一次 32 万字符的提问会被原样送给模型并成功返回，单次成本 ¥0.81（正常约 ¥0.008，差 100 倍）。
MAX_REQUEST_CHARS = 20_000


# ---- 请求/响应模型（原 app/api/schemas.py，并入本文件：仅 /chat 系列接口使用）----
class ChatMessage(BaseModel):
    # 角色必须是 LangChain 侧真正能消费的四种：AgentRuntime._to_lc_messages 只认
    # user/human/assistant/ai，其它角色的消息会被**静默丢弃**。实测传 {"role":"hacker"}：
    # 校验若放行，就得到"一条用户消息都没有"的请求，模型照样被调用（白花钱、脏 trace、
    # 还会因为拿不到中文提问而用英文寒暄），前端却看到一个 200。
    role: Literal["user", "human", "assistant", "ai"]
    content: str


class ChatRequest(BaseModel):
    # 契约（与 graph.AgentRuntime._build_input 一致，别按字段名想当然）：
    # **同一 session 每次只追加最新一条 user 消息**，历史由 LangGraph 按 thread 自动拼接。
    # 把整段历史每轮都发过来会让 checkpoint 里的消息翻倍、prompt token 虚高、
    # 模型把每一轮都看到两遍。这里保留 list 只是为了兼容"一次多轮"的批式调用方。
    messages: list[ChatMessage] = Field(
        min_length=1,
        description="最新用户输入（同一 session 请只追加最新一条；历史由服务端按 session 续接）",
    )
    session_id: str | None = None

    @model_validator(mode="after")
    def _validate_messages(self) -> "ChatRequest":
        """至少要有一条非空 user 消息，且总长度有上限。

        两者都是"没有就直接拒掉"的输入校验（422），而不是等模型给出一个没有意义的 200：
        空 messages / 只发 assistant 消息时，模型拿到的是一段没有提问的上下文。
        """
        if not any(m.role in {"user", "human"} and m.content.strip() for m in self.messages):
            raise ValueError("messages 里至少要有一条非空的 user 消息")
        total = sum(len(m.content) for m in self.messages)
        if total > MAX_REQUEST_CHARS:
            raise ValueError(
                f"消息总长度 {total} 字符超过上限 {MAX_REQUEST_CHARS}，请精简后再发送"
            )
        return self


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = Field(
        default_factory=list,
        description="答案正文里真实标注的【来源：X】——即『答案引用了哪些来源』",
    )
    retrieved_sources: list[str] = Field(
        default_factory=list,
        description="本轮工具（检索）返回过的来源，即 top-k 命中；是检索口径，不等于答案引用",
    )


def _question_of(req: ChatRequest) -> str:
    """取本轮用户问题 = 最后一条 user 消息。

    同一段逻辑原先在 _record_turn 里重复了一份，现统一走这里，避免两处口径漂移。
    """
    return next(
        (m.content for m in reversed(req.messages) if m.role in {"user", "human"}),
        "",
    )


def _thread_of(req: ChatRequest) -> tuple[str, bool]:
    """定本次运行的 LangGraph thread_id，并标记它是否为"一次性"。

    带 session_id 时就是会话 id（历史照旧由该 thread 续接）；不带时 LangGraph 仍需要一个
    thread_id，于是现造一个 uuid 并打上 ephemeral 标记 —— 这种 thread 客户端拿不到、
    永远续不上，跑完必须回收（见 _reclaim_one_shot）。
    """
    if req.session_id:
        return req.session_id, False
    return f"oneshot-{uuid.uuid4().hex}", True


async def _reclaim_one_shot(runtime, thread_id: str, ephemeral: bool) -> None:
    """删掉一次性 thread 的 checkpoint。

    为什么必须有这一步：`data/checkpoints.sqlite` 就是服务在用的那个库，而不带 session_id
    的请求每次都会新建一个 uuid thread 且**永不回收**——实测库里 303 个 thread 中 290 个是
    这种孤儿（真实会话只有 13 个，库已 15.5MB）。评测脚本早已用 aclose_thread 做同样的事，
    HTTP 路径此前漏了。aclose_thread 自身吞异常，回收失败不会影响本轮回答。
    """
    if ephemeral:
        await runtime.aclose_thread(thread_id)


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


async def _save_trace(
    req: ChatRequest,
    recorder_summary: dict,
    answer: str,
    sources: list[str],
    retrieved_sources: list[str],
) -> None:
    """持久化一次运行的 trace（观测用，失败不影响回答）。"""
    from app import store

    await store.save_trace(
        conversation_id=req.session_id,
        recorder_summary=recorder_summary,
        answer=answer,
        sources=sources,
        retrieved_sources=retrieved_sources,
    )


async def _persist(req: ChatRequest, rec, result: dict | None) -> None:
    """落库：有答案才写会话记录，trace 则无论成败都写（记录失败不影响主流程）。

    result=None 表示本轮没有产出答案（执行失败或客户端断连）——此时只写 trace，
    保持原有语义（历史里不出现"只有问题没有答案"的轮次）。
    """
    answer = (result or {}).get("answer", "")
    sources = (result or {}).get("sources", []) or []
    retrieved = (result or {}).get("retrieved_sources", []) or []
    # 有 result 但 answer 是空白：graph._final_answer 在"本轮 AI 消息 content 为空"时
    # 刻意返回空串（不许回退到上一轮答案）。这种轮次不该进会话记录——否则历史里就出现了
    # 本函数 docstring 声明不会出现的"只有问题没有答案"，前端预览也会是空串。
    if result is not None and (answer or "").strip():
        try:
            await _record_turn(req, answer, sources)
        except Exception:  # noqa: BLE001
            pass
    try:
        await _save_trace(req, rec.summarize(), answer, sources, retrieved)
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
    thread_id, ephemeral = _thread_of(req)
    result: dict | None = None
    try:
        result = await runtime.ainvoke(
            [m.model_dump() for m in req.messages], session_id=thread_id
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
        # 一次性 thread 的回收放在 finally：失败与取消路径同样不该把孤儿留在库里
        await _reclaim_one_shot(runtime, thread_id, ephemeral)
    # 成功路径同样 shield：答案已产出，此刻若客户端断开也不能丢掉这次 trace
    await asyncio.shield(_persist(req, rec, result))
    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        retrieved_sources=result.get("retrieved_sources", []),
    )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    """SSE 流式：逐步下发各节点增量，收尾事件带最终答案与引用。"""
    try:
        _, runtime = await ensure_services(request.app)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    thread_id, ephemeral = _thread_of(req)

    async def event_stream():
        from app.agent.graph import _text_of
        from app.observability import Recorder, reset_recorder, set_recorder

        rec = Recorder(question=_question_of(req))
        token = set_recorder(rec)
        persisted = False
        try:
            async for update in runtime.astream(
                [m.model_dump() for m in req.messages], session_id=thread_id
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
            if ephemeral:
                # 一次性 thread（无 session_id）的回收同样走后台任务：
                # 正常收尾与断连收尾都要发生，而断连时这里不能 await。
                try:
                    task = asyncio.create_task(_reclaim_one_shot(runtime, thread_id, ephemeral))
                except RuntimeError:
                    pass
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

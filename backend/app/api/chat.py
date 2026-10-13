# -*- coding: utf-8 -*-
from __future__ import annotations
import asyncio
import json
import threading
import uuid

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from app.agents.graph import build_agent_graph, new_agent_state
from app.agents.llm_router import get_llm
from app.api.deps import get_current_user
from app.core.config import settings
from app.core.kb_registry import ensure_kb_for_read
from app.core.rate_limit import check_chat_quota
from app.core.text_utils import ThinkBlockFilter, strip_think
from app.rag.pipeline import retrieve_and_rerank
from app.rag.retriever import get_retriever
from app.schemas.chat import ChatRequest, ChatResponse, SourceItem


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    if request.client:
        return request.client.host or "unknown"
    return "unknown"


def _check_chat_quota_or_429(user: str, request: Request):
    ip = _client_ip(request)
    decision = check_chat_quota(user, ip)
    if not decision.allowed:
        from fastapi import HTTPException, status

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=decision.reason,
            headers={"Retry-After": str(int(decision.retry_after) + 1)},
        )

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


def _build_prompt(question, contexts):
    parts = []
    # 截断上限走配置, 与 Agent 工具 / 评估链路共用。原先写死 600, 比默认
    # chunk_size 还短, 长 chunk 文末(定价 / 结论)被切掉后, 模型会在回答里
    # 抱怨"原文此处被截断", 而不是干脆答不知道。
    limit = getattr(settings, "context_snippet_chars", 2000)
    for i, c in enumerate(contexts):
        doc = c.get("document", "") or ""
        parts.append("[" + str(i + 1) + "] " + doc[:limit])
    ctx_text = "\n\n".join(parts)
    return (
        "你是一个严谨的企业知识库助手。请严格基于检索结果回答用户问题，"
        "并在回答末尾用 [1] [2] 等标注引用来源。如果检索结果中没有答案，请直接说「未找到相关信息」。\n\n"
        + "【检索结果】\n" + ctx_text + "\n\n"
        + "【用户问题】" + question + "\n\n"
        + "【回答】"
    )


@router.post("", response_model=ChatResponse, summary="一次性问答")
async def chat(req: ChatRequest, request: Request, user: str = Depends(get_current_user)):
    _check_chat_quota_or_429(user, request)
    # ACL: 公共 kb 直接放行; 私有 kb 必须 owner == 当前用户 (403)
    ensure_kb_for_read(req.kb_id, user)
    retriever = get_retriever(req.kb_id)
    if req.mode == "agent":
        return await _chat_agent(req)

    # 走与 run_rag 完全相同的检索链路 (查询改写 -> 多路 RRF -> 候选池截断 ->
    # Cross-Encoder 精排), 保证线上问答与离线评估行为一致.
    # 内部所有 CPU 密集调用都已卸到 threadpool, 不阻塞事件循环.
    reranked = (await retrieve_and_rerank(
        req.question, kb_id=req.kb_id, retriever=retriever
    ))[: req.top_k]
    prompt = _build_prompt(req.question, reranked)
    # LLM 调用也是阻塞型同步 SDK, 也卸到 threadpool.
    resp = await asyncio.to_thread(get_llm().invoke, prompt)
    sources = []
    for c in reranked:
        meta = c.get("metadata") or {}
        doc_id = meta.get("doc_id", "")
        chunk_id = c.get("id", "")
        content = (c.get("document") or "")[:300]
        score = float(c.get("rerank_score", c.get("rrf_score", 0)))
        sources.append(SourceItem(
            doc_id=doc_id, chunk_id=chunk_id, content=content, score=score, metadata=meta,
        ))
    sid = req.session_id or ("sess_" + uuid.uuid4().hex[:8])
    # 清洗: 推理模型可能把 <think>...</think> 思考过程混进 content.
    return ChatResponse(answer=strip_think(resp.content), session_id=sid, sources=sources, usage={})


@router.post("/stream", summary="SSE 流式问答")
async def chat_stream(req: ChatRequest, request: Request, user: str = Depends(get_current_user)):
    _check_chat_quota_or_429(user, request)
    # ACL: 进入流式生成前先校验, 避免生成到一半才抛 403
    ensure_kb_for_read(req.kb_id, user)
    async def event_gen():
        try:
            retriever = get_retriever(req.kb_id)
            if req.mode == "agent":
                # ----- agent 分支: 严格 offload + 按节点推 trace (真实"流") -----
                # graph.invoke 是同步阻塞 (langgraph 同步实现), 必须卸到 threadpool,
                # 否则会卡住 FastAPI 事件循环, 这是 P1 offload fix 的最低要求.
                # 进一步: 用 graph.stream() 在 threadpool 内部每跑完一个节点, 就通过
                # queue 把这个节点的 trace 喂给事件循环, 推到前端, 而不是等整图跑完
                # 才吐. 这样 SSE 是真正的"分步反馈", 而不是延迟一致性.
                graph = build_agent_graph()
                # 初始 state 由 graph 模块的工厂统一构造: 这个 9 键 dict 原先在
                # chat / chat_stream / pipeline 三处各复制了一份, 加字段时漏改
                # 一处就要到运行时才炸 (TypedDict 无默认值, 读 state["k"] 直接 KeyError).
                state_in = new_agent_state(req.question, req.kb_id)

                # asyncio.Queue + call_soon_threadsafe, 取代原来的
                # threading.Queue + asyncio.to_thread(q.get).
                # 旧写法里 `await asyncio.to_thread(q.get)` 会把一个线程池 worker
                # 永久阻塞在空队列上: 生成器被取消时(asyncio 取消不会中断已在
                # 运行的线程任务)这个 worker 回不来, 每个提前断开的 agent 请求
                # 都会占住一个线程直到 graph 跑完. 改成事件循环侧 await 一个
                # asyncio.Queue 后, 取消可以立刻生效.
                loop = asyncio.get_running_loop()
                q: "asyncio.Queue[tuple[str, object]]" = asyncio.Queue()
                # 客户端断开时置位, 让后台线程尽早从 graph.stream 循环里退出.
                cancelled = threading.Event()

                def _emit(item) -> None:
                    try:
                        loop.call_soon_threadsafe(q.put_nowait, item)
                    except RuntimeError:
                        # 事件循环已关闭 (请求结束/进程退出), 丢弃即可.
                        pass

                def _runner():
                    try:
                        # graph.stream() 每跑完一个节点就 yield {node_name: state_after_step}.
                        last_trace_len = 0
                        last_final = ""
                        for step in graph.stream(state_in):
                            if cancelled.is_set():
                                break
                            for _node_name, ns in step.items():
                                if not isinstance(ns, dict):
                                    continue
                                curr_trace = ns.get("trace", []) or []
                                new_entries = curr_trace[last_trace_len:]
                                last_trace_len = len(curr_trace)
                                last_final = ns.get("final", last_final) or last_final
                                if new_entries:
                                    _emit(("trace", new_entries))
                        # 整图跑完, 把 final 一起 push 到 queue, 客户端从 token 事件里拿.
                        _emit(("done", last_final))
                    except Exception as e:  # pragma: no cover
                        _emit(("error", e))

                t = threading.Thread(target=_runner, daemon=True)
                t.start()

                # 在事件循环里 await asyncio.Queue, 取消可以立刻生效.
                final_text = ""
                try:
                    while True:
                        kind, payload = await q.get()
                        if kind == "trace":
                            for step in payload:
                                yield "data: " + json.dumps(
                                    {"type": "trace", "data": step}, ensure_ascii=False
                                ) + "\n\n"
                        elif kind == "error":
                            err = payload
                            logger.exception("agent stream error: " + str(err))
                            yield "data: " + json.dumps(
                                {"type": "error", "data": {"msg": str(err)}}, ensure_ascii=False
                            ) + "\n\n"
                            return
                        elif kind == "done":
                            final_text = strip_think(payload or "")
                            break
                finally:
                    # 客户端断开 / 生成器被关闭 (GeneratorExit) / 出错, 都置位让
                    # 后台线程在下一个节点边界退出, 不再空转到整图跑完.
                    cancelled.set()

                # 用一个 token 事件收尾 (跟 RAG 分支 SSE 协议一致).
                if final_text:
                    yield "data: " + json.dumps(
                        {"type": "token", "data": {"text": final_text}}, ensure_ascii=False
                    ) + "\n\n"
                yield "data: {\"type\":\"done\"}\n\n"
                return

            # 同 /chat 非 agent 分支: 复用 pipeline 的统一检索链路.
            reranked = (await retrieve_and_rerank(
                req.question, kb_id=req.kb_id, retriever=retriever
            ))[: req.top_k]
            for c in reranked:
                payload = {
                    "type": "source",
                    "data": {
                        "chunk_id": c.get("id", ""),
                        "content": (c.get("document") or "")[:200],
                        "score": c.get("rerank_score", c.get("rrf_score", 0)),
                    },
                }
                yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
            prompt = _build_prompt(req.question, reranked)
            # LLM 流式调用也是阻塞型同步 SDK, 用线程池迭代.
            # run_in_executor 返回迭代器本身是阻塞的, 我们用一个 async 包装,
            # 让每轮 token 在 threadpool 里生成, 通过 queue 喂给 asyncio.
            from app.core.streaming import sync_iter_to_async
            # 推理模型会把 <think>...</think> 思考过程混在 content 里流式吐出;
            # 用状态机过滤器在 token 级别掐掉, 避免思考块在前端一闪而过.
            # (标签可能被切到两个 chunk 之间, 所以不能逐 chunk 简单 replace)
            think_filter = ThinkBlockFilter()
            async for token in sync_iter_to_async(get_llm().stream(prompt)):
                # token 是 AIMessageChunk, 取出 .content 才能 JSON 序列化.
                raw = getattr(token, "content", token) if token else ""
                if not raw:
                    continue
                text = think_filter.feed(raw if isinstance(raw, str) else str(raw))
                if text:
                    yield "data: " + json.dumps({"type": "token", "data": {"text": text}}, ensure_ascii=False) + "\n\n"
            tail = think_filter.flush()
            if tail:
                yield "data: " + json.dumps({"type": "token", "data": {"text": tail}}, ensure_ascii=False) + "\n\n"
            yield "data: {\"type\":\"done\"}\n\n"
        except Exception as e:
            logger.exception("stream error: " + str(e))
            yield "data: " + json.dumps({"type": "error", "data": {"msg": str(e)}}, ensure_ascii=False) + "\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


async def _chat_agent(req: ChatRequest) -> ChatResponse:
    graph = build_agent_graph()
    state_in = new_agent_state(req.question, req.kb_id)
    # graph.invoke 是同步阻塞, 卸到 threadpool 不阻塞 FastAPI 事件循环.
    result = await asyncio.to_thread(graph.invoke, state_in)
    sid = req.session_id or ("sess_" + uuid.uuid4().hex[:8])
    return ChatResponse(
        answer=result.get("final", ""),
        session_id=sid,
        sources=[],
        agent_trace=result.get("trace", []),
        usage={},
    )

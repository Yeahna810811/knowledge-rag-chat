from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any, Literal, Optional

import anyio
from fastapi import APIRouter, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from frontend.local_rag.api.sse import SSE_HEADERS, SSE_MEDIA_TYPE, sse_frames
from frontend.local_rag.services.knowledge_service import KnowledgeService
from frontend.local_rag.services.rate_limiter import RateLimiter

# session_id 落库为 String(128)，接口层同步约束，避免超长值打到数据库
SESSION_ID_FIELD = Field(default="default", min_length=1, max_length=128)
SESSION_ID_QUERY = Query(default="default", min_length=1, max_length=128)


class AskRequest(BaseModel):
    question: str
    session_id: str = SESSION_ID_FIELD
    mode: Literal["rag", "chat"] = "rag"


class SessionRequest(BaseModel):
    session_id: str = SESSION_ID_FIELD


class WebhookRequest(BaseModel):
    event: str = Field(description="例如: health.check / knowledge.reindex_status / docs.sync")
    payload: dict[str, Any] = Field(default_factory=dict)


def create_router(
    service: KnowledgeService,
    rate_limiter: RateLimiter | None = None,
) -> APIRouter:
    """创建并返回API路由，所有接口共用同一个 KnowledgeService 实例。

    rate_limiter 走显式注入，默认回落到 service 上那个（共享同一个 RedisClient）。
    """
    router = APIRouter()
    limiter = rate_limiter if rate_limiter is not None else service.rate_limiter

    @router.post("/upload")
    async def upload(file: UploadFile = File(...)):
        try:
            content = await file.read()
            # 解析 + 切分 + embedding 是重 CPU / 阻塞 I/O。
            # 这里原来是直接 await 同步调用——async def 端点里干阻塞活，
            # 会把整个事件循环锁死，所有并发请求（包括正在流的 SSE）一起卡住。
            # 显式丢线程池，async 的收益才真的拿得到。
            result = await anyio.to_thread.run_sync(
                lambda: service.ingest_upload(file.filename or "upload.bin", content)
            )
            return result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"文档处理失败：{e}")

    @router.post("/ask")
    def ask(req: AskRequest):
        # 这里刻意保留同步 def：FastAPI 会把同步端点自动丢进线程池执行，
        # 效果和手写 to_thread 一样，事件循环不会被堵住。
        # 反过来如果写成 `async def` 却在里面直接调阻塞的 service.ask()，
        # 就会变成"假异步"——一个请求把事件循环锁死，连正在流的 SSE 都跟着卡。
        # 真要流式请用下面的 /api/ask/stream。
        #
        # 限流放在业务逻辑之前：被限掉的请求不该消耗 LLM 额度。
        # Redis 挂掉时 check() 是 fail-open，所以这里不会因 Redis 故障拒绝请求。
        verdict = limiter.check(req.session_id)
        if not verdict.allowed:
            raise HTTPException(
                status_code=429,
                detail=verdict.as_detail(),
                headers={"Retry-After": str(verdict.retry_after)},
            )

        try:
            return service.ask(req.question, session_id=req.session_id, mode=req.mode)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"问答失败：{e}")

    @router.post("/ask/stream")
    async def ask_stream(req: AskRequest):
        """流式问答：以 SSE 逐块下发 token，同时把完整答案落库。

        两个必须在这里（而不是流里面）做完的事：
        1. 限流。响应头一旦发出去就改不了状态码了，所以 429 只能在
           流开始之前判定——被限掉的连接不会消耗任何 LLM 额度。
        2. 参数校验。空问题在流里只能变成一条 error 事件，前端体验很差。
        """
        # 限流同样放在业务逻辑之前：被限掉的请求不该消耗 LLM 额度。
        # Redis 挂掉时 check() 是 fail-open，不会因 Redis 故障拒绝请求。
        verdict = await anyio.to_thread.run_sync(lambda: limiter.check(req.session_id))
        if not verdict.allowed:
            raise HTTPException(
                status_code=429,
                detail=verdict.as_detail(),
                headers={"Retry-After": str(verdict.retry_after)},
            )

        events = service.astream_ask(
            req.question, session_id=req.session_id, mode=req.mode
        )
        return StreamingResponse(
            sse_frames(
                events, heartbeat_seconds=service.settings.sse_heartbeat_seconds
            ),
            media_type=SSE_MEDIA_TYPE,
            headers=SSE_HEADERS,
        )

    @router.get("/status")
    def status():
        return service.status()

    @router.get("/history")
    def history(session_id: str = SESSION_ID_QUERY):
        try:
            return service.get_history(session_id=session_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"读取会话历史失败：{e}")

    @router.delete("/reset")
    def reset():
        return service.reset()

    @router.post("/clear_history")
    def clear_history(req: SessionRequest):
        try:
            return service.clear_history(session_id=req.session_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"清空会话历史失败：{e}")

    @router.post("/webhook")
    def webhook(
        req: WebhookRequest,
        x_webhook_secret: Optional[str] = Header(default=None),
    ):
        """CI/CD / 自动化工作流入口：校验可选密钥后返回可观测状态。"""
        expected = service.settings.webhook_secret
        if expected:
            provided = x_webhook_secret or ""
            if not hmac.compare_digest(provided, expected):
                raise HTTPException(status_code=401, detail="Invalid webhook secret")

        status_payload = service.status()
        event = req.event.strip().lower()
        handled = {
            "health.check": {
                "ok": True,
                "status": status_payload,
            },
            "knowledge.reindex_status": {
                "ok": True,
                "vector_store_ready": status_payload["vector_store_ready"],
                "embedding_model": status_payload["embedding_model"],
            },
            "docs.sync": {
                "ok": True,
                "message": "Webhook received; use /api/upload to ingest documents",
                "payload": req.payload,
            },
        }.get(event)

        if handled is None:
            raise HTTPException(
                status_code=400,
                detail="Unsupported event. Use health.check / knowledge.reindex_status / docs.sync",
            )

        fingerprint = hashlib.sha256(
            f"{event}:{time.time()}".encode("utf-8")
        ).hexdigest()[:12]
        return {
            "accepted": True,
            "event": event,
            "request_id": fingerprint,
            "result": handled,
        }

    return router

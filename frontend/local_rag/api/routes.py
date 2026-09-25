from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any, Literal, Optional

from fastapi import APIRouter, File, Header, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

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
            result = service.ingest_upload(file.filename or "upload.bin", content)
            return result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"文档处理失败：{e}")

    @router.post("/ask")
    def ask(req: AskRequest):
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

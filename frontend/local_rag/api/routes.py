import hashlib
import hmac
import time
from typing import Any, Literal, Optional

from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from frontend.local_rag.services.knowledge_service import KnowledgeService


class AskRequest(BaseModel):
    question: str
    session_id: str = "default"
    mode: Literal["rag", "chat"] = "rag"


class SessionRequest(BaseModel):
    session_id: str = "default"


class WebhookRequest(BaseModel):
    event: str = Field(description="例如: health.check / knowledge.reindex_status / docs.sync")
    payload: dict[str, Any] = Field(default_factory=dict)


def create_router(service: KnowledgeService) -> APIRouter:
    """创建并返回API路由，所有接口共用同一个 KnowledgeService 实例"""
    router = APIRouter()

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
    def history(session_id: str = "default"):
        return service.get_history(session_id=session_id)

    @router.delete("/reset")
    def reset():
        return service.reset()

    @router.post("/clear_history")
    def clear_history(req: SessionRequest):
        return service.clear_history(session_id=req.session_id)

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

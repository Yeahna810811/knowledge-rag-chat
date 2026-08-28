from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from frontend.local_rag.services.knowledge_service import KnowledgeService


class AskRequest(BaseModel):
    question: str
    session_id: str = "default"  # 区分不同会话的对话记忆，前端不传时用默认值


class SessionRequest(BaseModel):
    session_id: str = "default"


def create_router(service: KnowledgeService) -> APIRouter:
    """创建并返回API路由，所有接口共用同一个 KnowledgeService 实例"""
    router = APIRouter()

    @router.post("/upload")
    async def upload(file: UploadFile = File(...)):
        try:
            content = await file.read()
            result = service.ingest_upload(file.filename, content)
            return result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"文档处理失败：{e}")

    @router.post("/ask")
    def ask(req: AskRequest):
        try:
            return service.ask(req.question, session_id=req.session_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"问答失败：{e}")

    @router.get("/status")
    def status():
        return service.status()

    @router.delete("/reset")
    def reset():
        return service.reset()

    @router.post("/clear_history")
    def clear_history(req: SessionRequest):
        return service.clear_history(session_id=req.session_id)

    return router

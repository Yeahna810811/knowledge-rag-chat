from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn

from frontend.local_rag.api.routes import create_router
from frontend.local_rag.config.settings import get_settings
from frontend.local_rag.services.knowledge_service import KnowledgeService

BASE_DIR = Path(__file__).resolve().parent


def create_app() -> FastAPI:
    settings = get_settings()
    knowledge_service = KnowledgeService(settings)

    app = FastAPI(
        title="本地RAG知识库 + AI聊天助手",
        description="文档上传、FAISS向量库、基于阿里云百炼大模型的检索增强问答",
        version="1.0",
        docs_url="/docs",
        redoc_url=None,
    )

    @app.get("/")
    async def index_html():
        return FileResponse(str(BASE_DIR / "index.html"))

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    router = create_router(knowledge_service)
    app.include_router(router, prefix="/api")

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
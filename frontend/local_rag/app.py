from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from frontend.local_rag.api.routes import create_router
from frontend.local_rag.config.settings import configure_observability, get_settings
from frontend.local_rag.services.knowledge_service import KnowledgeService

BASE_DIR = Path(__file__).resolve().parent
WEB_DIST = BASE_DIR.parent / "web" / "dist"
LEGACY_INDEX = BASE_DIR / "index.html"


def create_app() -> FastAPI:
    settings = get_settings()
    configure_observability(settings)
    knowledge_service = KnowledgeService(settings)

    app = FastAPI(
        title="知识库 RAG 多 Agent 问答平台",
        description="文档上传、多 Agent 调度、RAG/普通对话双模式、LangSmith 可观测",
        version="2.0",
        docs_url="/docs",
        redoc_url=None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    router = create_router(knowledge_service)
    app.include_router(router, prefix="/api")

    if WEB_DIST.exists():
        assets_dir = WEB_DIST / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        @app.get("/")
        async def vue_index():
            return FileResponse(str(WEB_DIST / "index.html"))
    else:

        @app.get("/")
        async def legacy_index():
            return FileResponse(str(LEGACY_INDEX))

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""Application-level performance test app (FakeLLM).

Why this file exists
--------------------
A load test that calls the real Qwen/DashScope endpoint measures mostly
*upstream network variance* and *burns API quota*. It cannot answer the
question Stage 5 is actually asking:

    "How does MY application layer behave under 1 / 5 / 10 / 20 concurrent
     requests — FastAPI routing, rate limiting, Redis retrieval cache,
     retrieval, persistence, SSE framing?"

So this app runs the **real application stack** and replaces **only** the LLM
client with a deterministic fake:

    real:  FastAPI routes -> KnowledgeService -> AgentOrchestrator
           -> RetrievalAgent (real BM25 over a real corpus)
           -> Redis retrieval cache -> SQLite persistence -> SSE framing
    fake:  GenerationAgent.llm  (deterministic token stream, no network)

What is NOT measured here
-------------------------
Real LLM TTFT / generation latency / upstream error rate. Those numbers must
come from running the production app (`python run.py`) with a valid
`DASHSCOPE_API_KEY`. They are a separate benchmark and are reported separately.

Usage
-----
    python -m uvicorn evaluation.performance.fake_app:app \
        --host 127.0.0.1 --port 8010

Environment knobs (all optional):

    PERF_WORK_DIR        scratch dir for SQLite / BM25 index / uploads
                         (default: a fresh TemporaryDirectory per process)
    PERF_REDIS_URL       default redis://127.0.0.1:6379/0
    PERF_CORPUS          corpus dir ingested at startup
                         (default: evaluation/corpus)
    PERF_FIRST_TOKEN_DELAY   seconds before the first fake token (default 0.01)
    PERF_PIECE_DELAY         seconds between fake tokens      (default 0.005)
    PERF_CHUNKS              comma separated fake tokens
    PERF_RATE_LIMIT          requests per window (default 60)
    PERF_CACHE_ENABLED       1/0, Redis retrieval cache (default 1)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator

import anyio
from fastapi import FastAPI

from frontend.local_rag.api.routes import create_router
from frontend.local_rag.cache.redis_client import RedisClient
from frontend.local_rag.config.settings import Settings
from frontend.local_rag.core.document_loader import load_document
from frontend.local_rag.core.retrieval.lexical_store import LexicalStore
from frontend.local_rag.core.text_splitter import split_documents
from frontend.local_rag.services.knowledge_service import KnowledgeService

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 进程内探针用：create_perf_app() 建好的 service 实例。
_SERVICE_FOR_PROBE: KnowledgeService | None = None

# ---------------------------------------------------------------- fake LLM


class _Chunk:
    """Minimal LangChain-compatible chunk: generation layer reads `.content`."""

    __slots__ = ("content",)

    def __init__(self, text: str) -> None:
        self.content = text


class FakeLLM:
    """Deterministic LLM stand-in.

    Delays are configurable so the harness can simulate a plausible TTFT and
    token cadence while keeping the run deterministic and quota-free. It never
    touches the network and never raises, so a non-zero error rate in the
    results is a genuine application error, not upstream flakiness.
    """

    def __init__(
        self,
        chunks: list[str] | None = None,
        *,
        first_token_delay: float = 0.01,
        piece_delay: float = 0.005,
    ) -> None:
        self.chunks = chunks or ["性", "能", "测", "试", "答", "案"]
        self.first_token_delay = first_token_delay
        self.piece_delay = piece_delay
        self.stream_calls = 0
        self.invoke_calls = 0

    def invoke(self, messages: Any) -> Any:
        self.invoke_calls += 1
        if self.first_token_delay:
            import time

            time.sleep(self.first_token_delay)

        class _Resp:
            content = "".join(self.chunks)

        return _Resp()

    async def astream(self, messages: Any) -> AsyncIterator[Any]:
        self.stream_calls += 1
        if self.first_token_delay:
            await anyio.sleep(self.first_token_delay)
        for text in self.chunks:
            if self.piece_delay:
                await anyio.sleep(self.piece_delay)
            yield _Chunk(text)


# ---------------------------------------------------------------- app build


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def build_service() -> KnowledgeService:
    """Build a KnowledgeService with the real stack and a fake LLM."""
    work_dir = os.environ.get("PERF_WORK_DIR")
    if work_dir:
        base = Path(work_dir)
        base.mkdir(parents=True, exist_ok=True)
        _tmp = None
    else:
        # keep_alive: the temp dir must outlive this function
        _tmp = tempfile.TemporaryDirectory(prefix="rag-perf-")
        base = Path(_tmp.name)

    settings = Settings(
        dashscope_api_key="fake-key-no-network",
        database_url=f"sqlite:///{base / 'perf.db'}",
        redis_url=os.environ.get("PERF_REDIS_URL", "redis://127.0.0.1:6379/0"),
        upload_dir=base / "uploads",
        faiss_index_dir=base / "faiss_index",
        bm25_index_dir=base / "bm25_index",
        retrieval_mode="bm25",
        retrieval_lazy_dense=True,
        retrieval_query_rewrite="prf",
        redis_cache_enabled=_env_bool("PERF_CACHE_ENABLED", True),
        redis_cache_ttl_seconds=600,
        rate_limit_enabled=True,
        rate_limit_requests=_env_int("PERF_RATE_LIMIT", 60),
        rate_limit_window_seconds=60,
        structured_logging_enabled=False,
        sse_heartbeat_seconds=0,
        llm_sdk_max_retries=0,
    )

    service = KnowledgeService(
        settings, redis_client=RedisClient(url=settings.redis_url)
    )
    # 必须显式建表：真实 app 是在 lifespan 里建的，而这个压测 app 没有 lifespan。
    # 漏掉的话每次请求都会走"持久化失败 → 降级"路径，测出来的延迟里
    # 根本不含写库成本，压测就失真了。
    service.database.create_tables()

    # 预建 BM25 索引，再让 service 去加载它。
    #
    # 顺序很关键：KnowledgeService._build_retriever() 在「纯 BM25 模式 +
    # 稀疏索引不存在」时会去构造 EmbeddingService，进而触发
    # BAAI/bge-small-zh 的模型下载——离线环境里这一步会卡死几分钟然后失败。
    # 先把索引落盘，need_dense 就为 False，整条链路都不碰 embedding。
    corpus = Path(os.environ.get("PERF_CORPUS", str(PROJECT_ROOT / "evaluation" / "corpus")))
    lexical = LexicalStore(settings.bm25_index_dir)
    if not lexical.index_path.exists() and corpus.is_dir():
        chunks: list[Any] = []
        for path in sorted(corpus.glob("*.md")):
            try:
                chunks.extend(split_documents(load_document(path), settings))
            except Exception:  # noqa: BLE001 - 单个语料文件失败不该阻断启动
                continue
        if chunks:
            lexical.add_documents(chunks)
            lexical.save()

    orchestrator = service._ensure_runtime()  # noqa: SLF001 - perf harness

    # Swap ONLY the LLM client. Everything else stays real.
    raw_chunks = os.environ.get("PERF_CHUNKS")
    chunks = raw_chunks.split(",") if raw_chunks else None
    orchestrator.generation_agent.llm = FakeLLM(
        chunks,
        first_token_delay=_env_float("PERF_FIRST_TOKEN_DELAY", 0.01),
        piece_delay=_env_float("PERF_PIECE_DELAY", 0.005),
    )
    return service


def create_perf_app() -> FastAPI:
    service = build_service()
    # 暴露给进程内探针脚本（例如排查 cache_hit 归因），不影响 HTTP 行为。
    global _SERVICE_FOR_PROBE
    _SERVICE_FOR_PROBE = service
    app = FastAPI(title="knowledge-rag-chat performance harness (FakeLLM)")
    app.include_router(create_router(service), prefix="/api")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "llm": "fake"}

    @app.get("/_metrics")
    def metrics() -> dict:
        """压测专用：暴露检索缓存计数器。

        存在的唯一理由：做 A/B 时必须能**证明**某个端口跑的是哪份代码，
        不能靠"我应该是先启动的"这种推断。

        判据是**计数**而不是延迟：
        - 未修复版：一次請求 = `cache_miss +1` 且 `cache_hit +1`（检索两次）
        - 修复版  ：一次全新问题 = `cache_miss +1`、`cache_hit +0`（检索一次）

        延迟本机噪声 ±10%~35%，计数没有噪声。
        """
        return service.cache_metrics()

    return app


app = create_perf_app()

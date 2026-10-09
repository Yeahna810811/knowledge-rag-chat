from __future__ import annotations

import logging
from typing import AsyncIterator

import anyio

from frontend.local_rag.config.settings import Settings, get_settings
from frontend.local_rag.core.agents import AgentOrchestrator
from frontend.local_rag.core.document_processor import DocumentProcessor
from frontend.local_rag.core.embedding_service import EmbeddingService
from frontend.local_rag.core.retrieval import (
    KnowledgeRetriever,
    LexicalStore,
    RetrievalMode,
)
from frontend.local_rag.core.retrieval.hybrid_retriever import HybridConfig
from frontend.local_rag.core.retrieval.query_rewrite import (
    PRFConfig,
    build_rewriter,
)
from frontend.local_rag.cache.redis_client import RedisClient
from frontend.local_rag.core.vector_store import VectorStoreManager
from frontend.local_rag.db.database import Database
from frontend.local_rag.observability.errors import (
    ClassifiedError,
    ErrorType,
    classify_exception,
)
from frontend.local_rag.observability.request_context import (
    UNKNOWN,
    ContextTokens,
    bind_context,
    current_request_id,
    new_request_id,
    reset_context,
)
from frontend.local_rag.observability.structured_log import (
    EV_LLM_FIRST_TOKEN,
    EV_PERSISTENCE_COMPLETE,
    EV_PERSISTENCE_FAILED,
    EV_REDIS_UNAVAILABLE,
    EV_REQUEST_COMPLETE,
    EV_REQUEST_FAILED,
    EV_REQUEST_START,
    EV_STREAM_CANCELLED,
    Timer,
    log_event,
)
from frontend.local_rag.services.conversation_store import ConversationStore
from frontend.local_rag.services.rate_limiter import RateLimiter
from frontend.local_rag.services.retrieval_cache import (
    CachedRetrievalStore,
    RetrievalCache,
)
from frontend.local_rag.utils.file_utils import ensure_dir

logger = logging.getLogger(__name__)

# 喂给 LLM 的上下文轮数上限。
# 注意：这是「读取侧」的窗口，数据库里保存的是完整历史，不在此处裁剪。
MAX_HISTORY_TURNS = 10


class KnowledgeService:
    """知识库服务：多 Agent 协同完成入库、双模式问答、状态与会话管理。"""

    def __init__(
        self,
        settings: Settings,
        database: Database | None = None,
        redis_client: RedisClient | None = None,
    ) -> None:
        self.settings = settings
        self.upload_dir = ensure_dir(settings.upload_dir)

        # 会话历史从内存 dict 换成数据库持久化层。
        # database 可注入，方便测试用临时 SQLite / 复用同一个 engine。
        self._database = database or build_database(settings)
        self._conversation_store = ConversationStore(self._database)

        # Redis 可注入（测试用 fakeredis / 故障桩），不传则按 settings 懒加载。
        # 懒加载的意义：没配 Redis 时，导入、建表这些路径不该去连它。
        self._redis_client = redis_client
        self._retrieval_cache: RetrievalCache | None = None
        self._rate_limiter: RateLimiter | None = None

        # Lazy-init heavy components (embedding model / LLM) on first use.
        self._embedding_service: EmbeddingService | None = None
        self._vector_store_manager: VectorStoreManager | None = None
        self._document_processor: DocumentProcessor | None = None
        self._orchestrator: AgentOrchestrator | None = None
        self._retriever: KnowledgeRetriever | None = None
        self._index_loaded = False

    def _build_retriever(self) -> KnowledgeRetriever:
        """构造统一检索入口，并决定是否需要加载 embedding 模型。

        这里藏着一个实打实的启动优化：
        纯 BM25 模式下，如果稀疏索引已经落盘，就完全不加载 embedding 模型——
        bge-small-zh 权重 + 分词器要好几百 MB，冷启动往往卡在这一步。
        只有当稀疏索引不存在（需要迁移）或模式不是纯 bm25 时才建稠密路。
        """
        mode = RetrievalMode.parse(self.settings.retrieval_mode)
        lexical = LexicalStore(self.settings.bm25_index_dir)

        need_dense = (
            mode is not RetrievalMode.BM25
            or not self.settings.retrieval_lazy_dense
            or not lexical.index_path.exists()
        )

        dense: VectorStoreManager | None = None
        if need_dense:
            self._embedding_service = EmbeddingService(self.settings)
            self._vector_store_manager = VectorStoreManager(
                self.settings, self._embedding_service.get_embeddings()
            )
            dense = self._vector_store_manager

        return KnowledgeRetriever(
            lexical_store=lexical,
            dense_store=dense,
            mode=mode,
            hybrid_config=HybridConfig(
                bm25_weight=self.settings.hybrid_bm25_weight,
                dense_weight=self.settings.hybrid_dense_weight,
            ),
            dense_fallback=self.settings.retrieval_dense_fallback,
            query_rewriter=self._build_rewriter(),
            rewrite_weight=self.settings.query_rewrite_weight,
        )

    def _build_rewriter(self):
        """按配置构造查询改写器。

        llm 模式这里不注入客户端：LLM 客户端由 generation 层持有，
        检索层不该反向依赖生成层。需要 llm 改写时，在 app 启动处
        用 `build_rewriter("llm", complete=...)` 显式装配，
        避免「检索层偷偷多调一次大模型」这种看不见的成本。
        """
        return build_rewriter(
            self.settings.retrieval_query_rewrite,
            config=PRFConfig(
                feedback_docs=self.settings.query_rewrite_feedback_docs,
                expansion_terms=self.settings.query_rewrite_terms,
                expansion_weight=self.settings.query_rewrite_weight,
            ),
        )

    def _ensure_runtime(self) -> AgentOrchestrator:
        if self._orchestrator is not None:
            return self._orchestrator

        self._retriever = self._build_retriever()
        self._document_processor = DocumentProcessor(self.settings)
        self._orchestrator = AgentOrchestrator(
            settings=self.settings,
            document_processor=self._document_processor,
            # 交给 Agent 的是套了缓存的代理，而不是裸检索器。
            # 检索算法本身一行没动，缓存只是外面的一层壳。
            vector_store_manager=self._cached_store(),
            upload_dir=self.upload_dir,
        )
        if not self._index_loaded:
            self._retriever.load()
            self._index_loaded = True
        return self._orchestrator

    @property
    def retriever(self) -> KnowledgeRetriever:
        self._ensure_runtime()
        assert self._retriever is not None
        return self._retriever

    @property
    def vector_store_manager(self) -> KnowledgeRetriever:
        """向后兼容旧字段名，实际返回统一检索入口。"""
        return self.retriever

    @property
    def orchestrator(self) -> AgentOrchestrator:
        return self._ensure_runtime()

    @property
    def database(self) -> Database:
        """全局唯一的 Database 实例：engine 在这里，不在每个请求里重建。"""
        return self._database

    @property
    def conversation_store(self) -> ConversationStore:
        return self._conversation_store

    # ------------------------------------------------------------------ Redis
    @property
    def redis_client(self) -> RedisClient:
        """全局唯一的 RedisClient，连接与连接池都归它管。"""
        if self._redis_client is None:
            self._redis_client = build_redis_client(self.settings)
        return self._redis_client

    @property
    def retrieval_cache(self) -> RetrievalCache:
        if self._retrieval_cache is None:
            self._retrieval_cache = RetrievalCache(
                self.redis_client,
                enabled=self.settings.redis_cache_enabled,
                ttl_seconds=self.settings.redis_cache_ttl_seconds,
                retrieval_mode=self.settings.retrieval_mode,
                rewrite_mode=self.settings.retrieval_query_rewrite,
                top_k=self.settings.retrieval_top_k,
            )
        return self._retrieval_cache

    @property
    def rate_limiter(self) -> RateLimiter:
        if self._rate_limiter is None:
            self._rate_limiter = RateLimiter(
                self.redis_client,
                enabled=self.settings.rate_limit_enabled,
                limit_requests=self.settings.rate_limit_requests,
                window_seconds=self.settings.rate_limit_window_seconds,
            )
        return self._rate_limiter

    def _cached_store(self) -> Any:
        """构造带缓存的检索代理。缓存关闭时直接返回原检索器，不多套一层。"""
        cache = self.retrieval_cache
        if not cache.enabled:
            return self._retriever
        return CachedRetrievalStore(self._retriever, cache)

    def _bump_knowledge_version(self, reason: str) -> None:
        """知识库内容确认变更后让旧检索缓存失效。

        只有 upload / reset 真正成功之后才调用；失败路径不会走到这里。
        Redis 挂掉时 bump 不成功也无所谓：缓存本来就写不进去，
        不存在「旧缓存还能命中」的问题。
        """
        try:
            self.retrieval_cache.bump_knowledge_version()
        except Exception as exc:  # noqa: BLE001
            logger.warning("知识库版本号递增失败（%s），不影响知识库内容: %s", reason, exc)

    def ingest_upload(self, filename: str, content: bytes) -> dict:
        result = self.orchestrator.ingest(filename, content)
        # 先成功再 bump：ingest 抛异常时不会执行到这里，缓存也就不会白白失效。
        self._bump_knowledge_version("upload")
        return result

    # ------------------------------------------------------ 请求生命周期辅助
    def _begin_request(
        self,
        question: str,
        session_id: str,
        mode: str,
        request_id: str | None,
        *,
        streaming: bool,
    ) -> tuple[ContextTokens, str, Timer]:
        """绑定 request_id 上下文 + 起表 + 打 request_start。

        返回 (复位 token, request_id, 计时器)。调用方必须在 finally 里复位，
        否则 ContextVar 会泄漏到下一个请求——同一个进程里复用连接的场景
        下这会表现为"日志里 request_id 串号"。

        request_id 的优先级：显式传入 > 当前上下文（路由已绑）> 新生成。
        最后一级是为了 service 被直接调用（脚本 / 测试）时不至于打不出 ID。
        """
        rid = request_id or current_request_id()
        if not rid or rid == UNKNOWN:
            rid = new_request_id()
        tokens = bind_context(rid, session_id, mode)
        timer = Timer()
        log_event(
            EV_REQUEST_START,
            question_len=len(question or ""),
            streaming=streaming,
        )
        return tokens, rid, timer

    def _cache_metrics_snapshot(self) -> dict[str, int] | None:
        """检索前的缓存计数器快照，用来判断本次请求到底命中没命中。

        用"计数差"而不是让缓存层返回一个 per-request 标志：缓存层是
        跨请求共享的单例，往它上面挂 per-request 状态在并发下必然串。
        """
        try:
            return self.retrieval_cache.metrics()
        except Exception:  # noqa: BLE001 - 拿不到就不报这项，不影响主链路
            return None

    def _cache_hit_since(self, snapshot: dict[str, int] | None) -> bool | None:
        """本次请求是否命中检索缓存。缓存关闭 / 拿不到指标时返回 None。"""
        if snapshot is None:
            return None
        try:
            now = self.retrieval_cache.metrics()
        except Exception:  # noqa: BLE001
            return None
        if now.get("cache_hit", 0) > snapshot.get("cache_hit", 0):
            return True
        if now.get("cache_miss", 0) > snapshot.get("cache_miss", 0):
            return False
        return None

    def _log_failed(
        self, error_type: ErrorType, message: str, timer: Timer, **extra: object
    ) -> None:
        log_event(
            EV_REQUEST_FAILED,
            status="failed",
            error_type=error_type.value,
            total_ms=timer.elapsed_ms_rounded(),
            message=message,
            **extra,
        )

    def ask(
        self,
        question: str,
        session_id: str = "default",
        mode: str = "rag",
        request_id: str | None = None,
    ) -> dict:
        """非流式问答。

        可靠性语义必须和流式一致（Stage 4 的硬要求），差别只在"怎么把结果
        交出去"：这里是一次性返回，流式是分帧下发。所以超时、重试、错误分类、
        落库纪律两边共用同一套实现，不各写一份。
        """
        tokens, rid, timer = self._begin_request(
            question, session_id, mode, request_id, streaming=False
        )
        try:
            question = (question or "").strip()
            if not question:
                # 仍抛 ValueError：路由层已有 ValueError → 400 的映射，
                # 改异常类型会动到 Stage 1/2 已验收的行为。
                self._log_failed(ErrorType.VALIDATION_ERROR, "问题内容不能为空", timer)
                raise ValueError("问题内容不能为空")

            # 只把最近 MAX_HISTORY_TURNS 轮喂给 LLM，避免 context 无限增长；
            # 完整历史仍然留在数据库里。
            history = self._recent_history(session_id, limit=MAX_HISTORY_TURNS)
            result = self.orchestrator.ask(question, history=history, mode=mode)

            result["session_id"] = session_id
            result["history_turns"] = self._append_turn(
                session_id, question, result["answer"], mode, fallback_turns=len(history)
            )
            result["request_id"] = rid
            # timing 是追加字段：retrieval_ms 在 chat 模式下为 None
            # （用 None 而不是 0，"0 毫秒"会被误读成真的检索过）。
            result["timing"] = {
                "total_ms": timer.elapsed_ms_rounded(),
                "retrieval_ms": result.get("retrieval_ms"),
            }
            log_event(
                EV_REQUEST_COMPLETE,
                status="ok",
                total_ms=timer.elapsed_ms_rounded(),
                retrieval_ms=result.get("retrieval_ms"),
                answer_len=len(result["answer"]),
                history_turns=result["history_turns"],
            )
            return result
        except ValueError:
            raise
        except ClassifiedError as exc:
            self._log_failed(exc.error_type, exc.message, timer)
            raise
        except Exception as exc:  # noqa: BLE001
            classified = classify_exception(exc)
            self._log_failed(classified.error_type, classified.message, timer)
            raise classified from exc
        finally:
            reset_context(tokens)

    # ------------------------------------------------------------------ 流式
    async def astream_ask(
        self,
        question: str,
        session_id: str = "default",
        mode: str = "rag",
        request_id: str | None = None,
    ) -> AsyncIterator[dict]:
        """流式问答：产出事件字典，由 api 层翻译成 SSE 帧。

        事件序列：meta → sources（仅 rag）→ trace → delta* → done
        出错时发 error 事件而不是抛异常——响应头已经在 SSE 握手时发出去了，
        这时候抛异常只能中断连接，前端拿不到任何可读的错误信息。

        request_id 必须在**生成器内部**绑定，不能指望路由：
        路由在返回 StreamingResponse 时就退栈了，真正的迭代发生在这之后。
        同理，结束时必须 finally 复位，否则会串到下一个请求。

        持久化纪律（阶段 3 验收口径，改动前请先读这段）：
        MySQL 里只允许出现「生成正常走完」的 turn。客户端 abort、代理超时、
        CancelledError、GeneratorExit、LLM streaming error，一律不写
        messages 表，只打结构化日志。

        理由不是洁癖：半截答案一旦混进历史，下一轮就会作为 assistant
        上下文被喂回模型，模型会学着说半截话、学着在被打断的地方收尾。
        这比丢掉这一轮糟糕得多——丢掉的代价是用户重问一次，污染历史
        的代价是整个会话的质量持续劣化，而且很难归因。

        取消 / 中断时可以写日志或 trace，方便统计中断率；想要保留半成品，
        应该显式加 status=cancelled/failed/incomplete 的数据模型，
        而不是复用正常 messages 表。本阶段不扩表，直接不保存。

        两处必须卸载到线程池，否则事件循环会被堵死：
        1. self.orchestrator 首次访问会触发 _ensure_runtime()，
           里面可能加载几百 MB 的 embedding 模型；
        2. 读历史 / 写历史是同步 SQLAlchemy 调用。
        """
        tokens, rid, timer = self._begin_request(
            question, session_id, mode, request_id, streaming=True
        )
        try:
            question = (question or "").strip()
            if not question:
                self._log_failed(ErrorType.VALIDATION_ERROR, "问题内容不能为空", timer)
                yield {
                    "event": "error",
                    "data": {
                        "request_id": rid,
                        "error_type": ErrorType.VALIDATION_ERROR.value,
                        "message": "问题内容不能为空",
                    },
                }
                return

            mode = (mode or "rag").lower()
            chunks: list[str] = []
            sources: list[dict] = []
            agent_trace: list[str] = []
            retrieval_ms: float | None = None
            ttft_ms: float | None = None
            cache_hit: bool | None = None
            cache_snapshot = self._cache_metrics_snapshot()

            history = await anyio.to_thread.run_sync(
                lambda: self._recent_history(session_id, limit=MAX_HISTORY_TURNS)
            )
            yield {
                "event": "meta",
                "data": {
                    "question": question,
                    "session_id": session_id,
                    "mode": mode,
                    "request_id": rid,
                },
            }

            try:
                orchestrator = await anyio.to_thread.run_sync(lambda: self.orchestrator)
                async for event in orchestrator.astream(
                    question, history=history, mode=mode
                ):
                    name = event.get("event")
                    data = event.get("data") or {}
                    if name == "sources":
                        sources = data.get("sources", [])
                        retrieval_ms = data.get("retrieval_ms")
                        cache_hit = self._cache_hit_since(cache_snapshot)
                    elif name == "trace":
                        agent_trace = data.get("agent_trace", [])
                    elif name == "delta":
                        if ttft_ms is None:
                            # TTFT 定义（和 Stage 3 保持一致）：
                            # request start → 第一个真实 LLM token 被发送。
                            # 不含 meta / heartbeat / trace。
                            ttft_ms = timer.elapsed_ms_rounded()
                            log_event(EV_LLM_FIRST_TOKEN, ttft_ms=ttft_ms)
                        chunks.append(data.get("text", ""))
                    yield event
            except anyio.get_cancelled_exc_class():
                # 客户端断开 / AbortController / 代理超时 / 外层任务被 cancel。
                # 只记录，不落库：半成品会污染下一轮的上下文（见方法顶注释）。
                self._log_stream_interrupted(
                    session_id, question, mode, chunks, reason="client_disconnected"
                )
                log_event(
                    EV_STREAM_CANCELLED,
                    status="cancelled",
                    error_type=ErrorType.STREAM_CANCELLED.value,
                    total_ms=timer.elapsed_ms_rounded(),
                    partial_len=sum(len(c) for c in chunks),
                )
                raise
            except GeneratorExit:
                # 显式 aclose() 走的是这条路径，和 cancel 是两条独立的投递方式。
                # 注意：捕获 GeneratorExit 后只能清理并原样抛出，
                # 这里绝不能 yield——生成器已经没有接收方了。
                self._log_stream_interrupted(
                    session_id, question, mode, chunks, reason="stream_closed"
                )
                raise
            except ClassifiedError as exc:
                # LLM streaming error（含超时 / 重试耗尽）：半截答案同样不入库。
                self._log_stream_interrupted(
                    session_id, question, mode, chunks, reason="generation_error"
                )
                self._log_failed(
                    exc.error_type,
                    exc.message,
                    timer,
                    partial_len=sum(len(c) for c in chunks),
                    streamed_tokens=len(chunks),
                )
                yield {
                    "event": "error",
                    "data": {
                        "request_id": rid,
                        "error_type": exc.error_type.value,
                        "message": exc.message,
                    },
                }
                return
            except Exception as exc:  # noqa: BLE001
                classified = classify_exception(exc)
                self._log_stream_interrupted(
                    session_id, question, mode, chunks, reason="generation_error"
                )
                self._log_failed(
                    classified.error_type,
                    classified.message,
                    timer,
                    partial_len=sum(len(c) for c in chunks),
                    streamed_tokens=len(chunks),
                )
                yield {
                    "event": "error",
                    "data": {
                        "request_id": rid,
                        "error_type": classified.error_type.value,
                        "message": classified.message,
                    },
                }
                return

            # ---- 只有走到这里，才算「生成正常走完」----
            answer = "".join(chunks)
            try:
                # shield=True 的作用现在只剩一件事：保护这次最终写入不被取消打断。
                # 最后一个 token 发出去之后，客户端随时可能断开（用户手快、代理超时），
                # 取消会在下一个 await 点投递。不屏蔽的话这条 write 会跟着被取消，
                # 变成"明明答完了却没存"——和原来的"没答完却存了"是两个方向的同一种错。
                with anyio.CancelScope(shield=True):
                    history_turns = await anyio.to_thread.run_sync(
                        lambda: self._append_turn(
                            session_id, question, answer, mode, fallback_turns=len(history)
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                # 走到这里说明 _append_turn 之外的东西炸了（线程池等）。
                # 写库失败本身不会到这里——它已被 _append_turn 降级并记录。
                # 无论哪种情况都不该让已经生成好的答案退化成 error 事件。
                logger.warning("流式最终落库异常，本轮未持久化: %s", exc)
                history_turns = len(history) + 1

            yield {
                "event": "done",
                "data": {
                    "question": question,
                    "answer": answer,
                    "sources": sources,
                    "mode": mode,
                    "session_id": session_id,
                    "request_id": rid,
                    "history_turns": history_turns,
                    "agent_trace": agent_trace,
                    # timing 是追加字段，前端不读也不影响既有渲染逻辑
                    "timing": {
                        "total_ms": timer.elapsed_ms_rounded(),
                        "ttft_ms": ttft_ms,
                        "retrieval_ms": retrieval_ms,
                        "cache_hit": cache_hit,
                    },
                },
            }
            log_event(
                EV_REQUEST_COMPLETE,
                status="ok",
                total_ms=timer.elapsed_ms_rounded(),
                ttft_ms=ttft_ms,
                retrieval_ms=retrieval_ms,
                cache_hit=cache_hit,
                answer_len=len(answer),
                history_turns=history_turns,
            )
        finally:
            reset_context(tokens)


    def _log_stream_interrupted(
        self,
        session_id: str,
        question: str,
        mode: str,
        chunks: list[str],
        reason: str,
    ) -> None:
        """丢弃这一轮时留痕：只写日志，不碰 MySQL。

        刻意做成同步方法且不做任何 await——它被CancelledError / GeneratorExit
        分支调用，那时所在作用域已经被取消，任何 await 都会立刻再抛取消。

        logger 用 %s 惰性格式化而不是 f-string：中断率高的场景下，
        省掉的是每条日志的字符串拼接成本。
        """
        logger.warning(
            "流式问答未完成，本轮不入库: reason=%s session=%s mode=%s "
            "question_len=%d partial_len=%d discarded_preview=%s",
            reason,
            session_id,
            mode,
            len(question),
            sum(len(c) for c in chunks),
            "".join(chunks)[:80],
        )

    # ------------------------------------------------------------ 会话持久化
    def _recent_history(self, session_id: str, limit: int) -> list[dict]:
        """读历史给 LLM。数据库短暂不可用时降级为空历史，而不是让 /api/ask 500。

        检索 + 生成是主链路，会话持久化是副链路——副链路挂掉不该拖死主链路。
        """
        try:
            return self.conversation_store.get_recent_history(session_id, limit=limit)
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取会话历史失败，降级为空历史: %s", exc)
            return []

    def _append_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        mode: str,
        fallback_turns: int,
    ) -> int:
        """写一轮问答，返回写入后的总轮数（失败时返回降级值）。

        成功 / 失败都在这里打结构化事件，而不是交给调用方判断：
        调用方拿到的返回值在成功和失败两种情况下都是个 int，
        它根本无法区分，让调用方去记日志必然漏掉一半。
        """
        try:
            self.conversation_store.append_turn(session_id, question, answer, mode)
            turns = self.conversation_store.count_turns(session_id)
            log_event(EV_PERSISTENCE_COMPLETE, status="ok", history_turns=turns)
            return turns
        except Exception as exc:  # noqa: BLE001
            # 副链路失败不拖死主链路：回答已经生成好了，
            # 更不该因为写库失败去重新生成（多烧一次额度且大概率同样失败）。
            logger.warning("写入会话历史失败，本轮未持久化: %s", exc)
            log_event(
                EV_PERSISTENCE_FAILED,
                status="failed",
                error_type=ErrorType.DATABASE_ERROR.value,
                message="会话持久化失败，不影响本次回答",
            )
            return fallback_turns + 1

    def get_history(self, session_id: str = "default") -> dict:
        history = self.conversation_store.get_history(session_id)
        return {"session_id": session_id, "history": history, "turns": len(history)}

    def status(self) -> dict:
        """注意：这个方法刻意不调用 _ensure_runtime()。

        /status 是健康检查入口，被 CI 和探针频繁调用。如果它触发
        embedding 模型加载，一次冷启动就会把健康检查拖到几十秒，
        在容器里会直接被 liveness probe 判死。所以这里只看文件是否存在。
        """
        if self._retriever is not None:
            ready = self._retriever.is_ready
            retrieval = self._retriever.stats
        else:
            lexical_file = self.settings.bm25_index_dir / "bm25_index.json"
            faiss_file = self.settings.faiss_index_dir / "index.faiss"
            ready = lexical_file.exists() or faiss_file.exists()
            retrieval = {
                "mode": str(self.settings.retrieval_mode).lower(),
                "note": "runtime 未初始化，仅按索引文件判断",
            }

        return {
            "vector_store_ready": ready,
            "database": self.database_status(),
            "redis": self.redis_status(),
            "upload_dir": str(self.upload_dir),
            "faiss_index_dir": str(self.settings.faiss_index_dir),
            "bm25_index_dir": str(self.settings.bm25_index_dir),
            "retrieval": retrieval,
            "embedding_model": self.settings.embedding_model,
            "chat_model": self.settings.chat_model,
            "dashscope_configured": bool(self.settings.dashscope_api_key),
            "langsmith_enabled": bool(
                self.settings.langchain_tracing_v2 and self.settings.langchain_api_key
            ),
            "active_sessions": self.active_sessions(),
            "agents": [
                "document_parse_agent",
                "retrieval_agent",
                "generation_agent",
            ],
            "modes": ["rag", "chat"],
            "runtime_initialized": self._orchestrator is not None,
        }

    def database_status(self) -> dict:
        """数据库健康状态。

        /api/status 是健康检查入口，数据库临时抖动不能让它 500——
        任何异常都降级成 healthy=False，绝不向上抛。
        """
        try:
            healthy = self.conversation_store.ping()
        except Exception as exc:  # noqa: BLE001
            logger.warning("数据库健康检查异常: %s", exc)
            healthy = False
        return {
            "backend": self._database.backend,
            "healthy": healthy,
        }

    def redis_status(self) -> dict:
        """Redis 健康状态。

        和 database_status() 一样的道理：/api/status 不能因为 Redis 抖动就 500。
        而且这里 ping 一次就够了，绝不为了 status 去初始化 embedding 模型。
        """
        try:
            healthy = self.redis_client.ping()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis 健康检查异常: %s", exc)
            healthy = False
        return {
            "backend": "redis",
            "healthy": healthy,
            "cache_enabled": bool(self.settings.redis_cache_enabled),
        }

    def active_sessions(self) -> int:
        """会话数。数据库不可用时返回 0，不影响 /api/status。"""
        try:
            return self.conversation_store.count_sessions()
        except Exception as exc:  # noqa: BLE001
            logger.warning("统计会话数失败: %s", exc)
            return 0

    def reset(self) -> dict:
        self.retriever.clear()
        self._bump_knowledge_version("reset")
        return {"message": "知识库已全部清空"}

    def close(self) -> None:
        """释放数据库与 Redis 连接。由 app 的 lifespan 在关闭时调用。"""
        self._database.dispose()
        if self._redis_client is not None:
            self._redis_client.close()

    def cache_metrics(self) -> dict:
        """缓存 / 限流的轻量计数，供后续 Observability 阶段接入。"""
        return {
            "cache": self.retrieval_cache.metrics(),
            "rate_limit": self.rate_limiter.metrics(),
        }

    def clear_history(self, session_id: str = "default") -> dict:
        self.conversation_store.clear_history(session_id)
        return {"message": "对话记忆已清空"}


def build_database(settings: Settings) -> Database:
    """按配置构造唯一的 Database 实例（engine 全局复用）。"""
    return Database(
        url=settings.database_url,
        echo=settings.database_echo,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
    )


def build_redis_client(settings: Settings) -> RedisClient:
    """按配置构造唯一的 RedisClient（连接池全局复用）。"""
    return RedisClient(
        url=settings.redis_url,
        socket_connect_timeout=settings.redis_socket_connect_timeout,
        socket_timeout=settings.redis_socket_timeout,
    )


def build_service(settings: Settings | None = None) -> KnowledgeService:
    return KnowledgeService(settings or get_settings())

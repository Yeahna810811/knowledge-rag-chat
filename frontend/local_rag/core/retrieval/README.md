# 检索增强层（Hybrid Retrieval）

BM25 稀疏检索 + 稠密向量检索 → RRF 融合 → 可选重排。
对外接口与 `VectorStoreManager.search()` 完全一致，可零改动替换。

```
frontend/local_rag/core/retrieval/
├── tokenizer.py         # 中文不分词，unigram + bigram
├── bm25.py              # 纯 Python BM25，零依赖
├── fusion.py            # RRF 融合
├── rerank.py            # NoOp / MMR / CrossEncoder
└── hybrid_retriever.py  # 统一入口
```

单测：`python tests/test_retrieval.py`（26 项，无需任何第三方依赖）

---

## 为什么需要它

原评测数据：`Hit@3 = 97.5%`，但 `Hit@1 = 85%`、`MRR = 0.9146`。
说明正确片段**基本都被召回了，但经常排在第 2~3 位被干扰片段压着**。

稠密检索的短板恰好是 BM25 的强项：型号、编号、API 名、错误码、端口号这些
必须精确匹配的字符串，在向量空间里 `qwen-plus` 和 `qwen-max` 距离极近，
业务上却完全是两回事。两路错误模式不相关，融合后互相补位。

---

## 接入步骤（3 处改动）

### 改动 1：`services/knowledge_service.py` —— 持有检索器

```python
from frontend.local_rag.core.retrieval import (
    HybridConfig, HybridRetriever, MMRReranker, NoOpReranker,
)

class KnowledgeService:
    def __init__(self, settings: Settings) -> None:
        ...
        self._hybrid: HybridRetriever | None = None

    def _ensure_runtime(self) -> AgentOrchestrator:
        if self._orchestrator is not None:
            return self._orchestrator
        ...
        # 在 AgentOrchestrator 构造之前插入：
        self._hybrid = HybridRetriever(
            dense_search=self._vector_store_manager.search,
            config=HybridConfig(candidate_pool=20),
            reranker=MMRReranker(lambda_param=0.7),   # 或 NoOpReranker() 先不重排
        )
        self._orchestrator = AgentOrchestrator(
            settings=self.settings,
            document_processor=self._document_processor,
            vector_store_manager=self._vector_store_manager,
            upload_dir=self.upload_dir,
            hybrid_retriever=self._hybrid,            # 新增参数
        )
        if not self._index_loaded:
            self._vector_store_manager.load()
            self._index_loaded = True
            self._rebuild_sparse_index()              # 新增
        return self._orchestrator
```

### 改动 2：入库后重建稀疏索引

稀疏索引存在内存里，进程重启或新增文档都要重建。

```python
    def _rebuild_sparse_index(self) -> None:
        """用 FAISS 里的全部 chunk 重建 BM25 索引。"""
        if self._hybrid is None or not self._vector_store_manager.is_ready:
            return
        store = self._vector_store_manager
        # 取一个足够大的 K 把全库捞出来；知识库很大时应改为直接遍历 docstore
        chunks = store.search("", k=10000)
        self._hybrid.index(chunks)

    def ingest_upload(self, filename: str, content: bytes) -> dict:
        result = self.orchestrator.ingest(filename, content)
        self._rebuild_sparse_index()      # 新增
        return result

    def reset(self) -> dict:
        self.vector_store_manager.clear()
        if self._hybrid is not None:
            self._hybrid.index([])        # 新增：清空稀疏索引
        return {"message": "知识库已全部清空"}
```

> **已知待优化**：`store.search("", k=10000)` 是用一次向量检索把全库捞出来，
> 属于权宜之计。正确做法是给 `VectorStoreManager` 增加 `list_all_documents()`
> 直接遍历 FAISS 的 `docstore`，避免无意义的一次 embedding。这是 v2.1 的 TODO。

### 改动 3：`core/agents/retrieval_agent.py` —— 走混合检索

```python
class RetrievalAgent(BaseAgent):
    name = "retrieval_agent"

    def __init__(self, vector_store_manager, top_k=4, hybrid_retriever=None):
        self.vector_store_manager = vector_store_manager
        self.top_k = top_k
        self.hybrid_retriever = hybrid_retriever

    @traceable(name="retrieval_agent", run_type="retriever")
    def run(self, question="", top_k=None, **_):
        k = top_k or self.top_k
        if not question.strip():
            return AgentResult(agent=self.name, success=False, message="问题为空")
        if not self.vector_store_manager.is_ready:
            return AgentResult(agent=self.name, success=True, message="知识库尚未就绪",
                               data={"sources": [], "ready": False})

        # 混合检索优先；未启用或降级时自动回退到纯稠密
        search_fn = self.vector_store_manager.search
        if self.hybrid_retriever is not None and self.hybrid_retriever.is_ready:
            search_fn = self.hybrid_retriever.search

        sources = search_fn(question, k)
        return AgentResult(agent=self.name, success=True,
                           message=f"检索到 {len(sources)} 条相关片段",
                           data={"sources": sources, "ready": True, "top_k": k,
                                 "degraded": getattr(self.hybrid_retriever, "degraded", False)})
```

`orchestrator.py` 里把 `hybrid_retriever` 透传给 `RetrievalAgent` 即可。

---

## 参数建议

| 参数 | 默认 | 什么时候调 |
|---|---|---|
| `candidate_pool` | 20 | 知识库越大越要调大，但重排开销随之上升 |
| `bm25_weight` / `dense_weight` | 1.0 / 1.0 | 用 `--sweep-weights` 在自己的评测集上扫；术语密集型知识库可给 BM25 加权 |
| `rrf_k` | 60 | 一般不动。调小 → 更看重头部排名；调大 → 更看重"被几路召回" |
| `MMRReranker.lambda_param` | 0.7 | chunk_overlap 大、Top-K 里重复多时调小 |

---

## 已验证 / 未验证

**已验证**（`python tests/test_retrieval.py`，26 项全过）
- BM25 排序、长度归一、词频饱和（k1）
- RRF 的并集语义、权重、结果确定性
- 稠密路异常时自动降级且不抛异常
- MMR 去冗余与上游相关性分的接入

**未验证**（本机缺依赖 + 无 DASHSCOPE_API_KEY，需要你补跑）
- bge-small-zh + FAISS 的真实融合收益
- CrossEncoder 重排的延迟与收益
- 大知识库下 `_rebuild_sparse_index` 的耗时

真实 A/B 请跑：

```bash
python evaluation/evaluate_retrieval_ab.py --dense faiss --metric bigram \
    --corpus evaluation/corpus/*.md evaluation/extended_noise/*.md
```

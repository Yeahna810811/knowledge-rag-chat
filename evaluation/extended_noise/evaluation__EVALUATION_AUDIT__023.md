8 项单测**，不再是纸面结论。

### 改造内容

| 文件 | 变更 |
|---|---|
| `core/retrieval/lexical_store.py` | **新增**：持久化 BM25 索引。JSON 落盘（不用 pickle），加载时重建，原子写 |
| `core/retrieval/knowledge_retriever.py` | **新增**：统一检索门面，按模式路由 + 三级降级 |
| `core/retrieval/protocol.py` | **新增**：`RetrievalStore` 协议，让 Agent 不依赖具体实现 |
| `core/retrieval/knowledge_retriever.py` | 支持 `dense` / `bm25` / `hybrid` 三模式 |
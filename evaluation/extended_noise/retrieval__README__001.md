`evaluation/EVALUATION_AUDIT.md`。

---

## 目录结构

```
frontend/local_rag/core/retrieval/
├── tokenizer.py             # 中文不分词，unigram + bigram；NFKC 归一
├── bm25.py                  # 纯 Python BM25，零第三方依赖
├── fusion.py                # RRF 融合 + 轮询对照
├── rerank.py                # NoOp / MMR / CrossEncoder
├── hybrid_retriever.py      # 两路 RRF 融合（实验用）
├── lexical_store.py         # 持久化 BM25 索引（JSON 落盘）
├── knowledge_retriever.py   # 生产入口：按模式路由 + 降级
└── protocol.py              # RetrievalStore 协议，解
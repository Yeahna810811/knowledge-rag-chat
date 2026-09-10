└── protocol.py              # RetrievalStore 协议，解耦 Agent 与实现
```

单测：

```bash
python tests/test_retrieval.py            # 26 项，检索算法
python tests/test_knowledge_retriever.py  # 38 项，门面 / 持久化 / 降级 / 迁移
```

两者都不需要任何第三方依赖，秒级跑完。

---

## 三种模式

| 模式 | 行为 | 适用 |
|---|---|---|
| `bm25`（默认） | BM25 主导；零命中时回落稠密 | 生产 |
| `dense` | 纯 FAISS 向量 | v1 行为，用于 A/B 与回滚 |
| `hybrid` | 两路 RRF 融合 | 实验；换更强 embedding 后可重新评估 |

配置（`frontend/local_rag/.env` 或环境变量）：
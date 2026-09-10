g 后可重新评估 |

配置（`frontend/local_rag/.env` 或环境变量）：

```env
RETRIEVAL_MODE=bm25
RETRIEVAL_DENSE_FALLBACK=true    # BM25 零命中时是否回落到稠密
RETRIEVAL_LAZY_DENSE=true        # 纯 BM25 且索引已落盘时，跳过 embedding 模型加载
HYBRID_BM25_WEIGHT=1.0
HYBRID_DENSE_WEIGHT=1.0
```

---

## 为什么默认不是混合检索

100 题 benchmark（80 可答）、59 chunk 语料、真实 bge-small-zh 向量，
配对 bootstrap 10000 轮，对照 bm25 单路：
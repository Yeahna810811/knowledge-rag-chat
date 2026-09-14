# Retrieval Layer

本目录实现 `knowledge-rag-chat` 的检索层。

当前应用默认：

```text
retrieval_mode = bm25
```

正式 Benchmark 表明，在当前技术文档型知识库中，BM25 的 Retrieval 指标高于当前使用的 BGE Dense Retrieval。

Dense 与 Hybrid 仍完整保留，方便未来更换语料或 Embedding 后重新评估。

---

## 1. 目录结构

```text
retrieval/
├── tokenizer.py
├── bm25.py
├── lexical_store.py
├── protocol.py
├── knowledge_retriever.py
├── hybrid_retriever.py
├── fusion.py
├── rerank.py
└── query_rewrite.py
```

主要职责：

| File | Responsibility |
| --- | --- |
| `tokenizer.py` | BM25 Tokenization |
| `bm25.py` | BM25 核心算法 |
| `lexical_store.py` | BM25 文档管理与 JSON 持久化 |
| `protocol.py` | RetrievalStore 接口协议 |
| `knowledge_retriever.py` | Dense / BM25 / Hybrid 路由 |
| `hybrid_retriever.py` | Hybrid Retrieval |
| `fusion.py` | RRF Ranking Fusion |
| `rerank.py` | MMR / Rerank |
| `query_rewrite.py` | PRF / LLM Query Rewrite |

---

## 2. 三种 Retrieval Mode

### BM25

当前默认方案：

```text
Query
↓
Query Rewrite
↓
LexicalStore
↓
BM25
↓
Top-K
```

适合：

```text
API
配置名
文件名
模型名
类名
数字
端口
固定技术术语
```

### Dense

```text
Query
↓
BAAI/bge-small-zh-v1.5
↓
Embedding Vector
↓
FAISS
↓
Top-K
```

主要用于语义检索。

### Hybrid

```text
Dense Ranking
+
BM25 Ranking
↓
RRF
↓
Top-K
```

Hybrid 作为实验能力保留。

---

## 3. 配置

```env
RETRIEVAL_MODE=bm25

RETRIEVAL_DENSE_FALLBACK=true
RETRIEVAL_LAZY_DENSE=true

HYBRID_BM25_WEIGHT=1.0
HYBRID_DENSE_WEIGHT=1.0

RETRIEVAL_QUERY_REWRITE=prf
QUERY_REWRITE_WEIGHT=0.3
QUERY_REWRITE_TERMS=6
QUERY_REWRITE_FEEDBACK_DOCS=3
```

---

## 4. 正式 Retrieval Benchmark

正式实验：

```text
100 questions
80 answerable
20 unanswerable

44 corpus files
46 chunks

chunk_size = 500
chunk_overlap = 50
```

结果：

| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Dense | 0.625 | 0.775 | 0.838 | 0.7060 |
| **BM25** | **0.825** | **0.950** | **0.975** | **0.8869** |
| Hybrid 1:1 | 0.700 | 0.887 | 0.925 | 0.7927 |
| Hybrid + MMR | 0.700 | 0.875 | 0.938 | 0.7860 |

因此：

```text
Dense Hit@1 = 62.5%
BM25 Hit@1  = 82.5%
```

BM25：

```text
+20 percentage points
```

---

## 5. Hybrid Weight Sweep

| Strategy | Hit@1 | MRR |
| --- | ---: | ---: |
| **BM25** | **0.825** | **0.8869** |
| Hybrid 1.0:1 | 0.700 | 0.7927 |
| Hybrid 1.5:1 | 0.688 | 0.7988 |
| Hybrid 2.0:1 | 0.725 | 0.8213 |

提高 BM25 权重后 Hybrid 会逐渐接近 BM25，但当前测试配置均未超过纯 BM25。

因此当前没有为了架构复杂度而强行采用 Hybrid。

---

## 6. Statistical Test

80 道可答题进行：

```text
10000-round paired bootstrap
+
sign test
```

Dense vs BM25：

```text
Dense - BM25 ΔMRR = -0.1808

95% CI:
[-0.2848, -0.0781]

Bootstrap p < 0.001
Sign Test p = 0.003
```

95% CI 不跨 0。

因此当前 Benchmark 对：

```text
BM25 > Dense
```

提供了统计支持。

Hybrid 当前所有测试权重均没有超过 BM25。

详细结果：

```text
evaluation/results/retrieval_final/significance_test.txt
```

---

## 7. MMR

Hybrid 后加入 MMR：

```text
Hybrid MRR       = 0.7927
Hybrid + MMR MRR = 0.7860
```

当前 Benchmark 中没有整体提升。

MMR 代码继续保留，用于未来更大知识库中的多样性控制。

---

## 8. Query Rewrite

Retrieval Layer 支持：

```text
off
prf
llm
```

当前默认：

```text
prf
```

PRF 使用初步检索结果提取扩展词，对原 Query 进行低权重补充。

查询改写属于独立 Retrieval 优化模块。

正式 Dense / BM25 / Hybrid Benchmark 与 Query Rewrite 实验应分别报告，避免把不同变量混在同一个实验结论中。

---

## 9. BM25 Persistence

`LexicalStore` 支持：

```text
add_documents
search
search_weighted
save
load
clear
```

BM25 索引使用 JSON 保存。

保存内容包括：

```text
document content
metadata
BM25 configuration
```

服务重启：

```text
Load JSON
↓
Restore Documents
↓
Rebuild BM25 Index
```

不依赖 Pickle 保存 BM25 倒排结构。

---

## 10. RetrievalStore Protocol

上层 Agent 不直接依赖：

```text
LexicalStore
或
VectorStoreManager
```

而是依赖统一 Retrieval 接口。

主要方法：

```text
is_ready
search
add_documents
save
load
clear
```

因此：

```text
RetrievalAgent
      ↓
RetrievalStore
   ↙       ↘
BM25       Dense
```

上层调用方式保持一致。

---

## 11. Fallback 与降级

KnowledgeRetriever 对检索异常进行保护。

典型路径：

```text
BM25
↓
可用
→ 返回结果
```

或在配置和 Dense Store 状态允许时：

```text
BM25 无可用结果
↓
Dense Fallback
```

单路异常不会直接导致整个问答服务崩溃。

降级状态可通过 Retriever 状态暴露给上层监控。

---

## 12. Runtime

当前本地 Runtime Benchmark：

```text
46 chunks
46 queries
```

| Metric | BM25 | Dense |
| --- | ---: | ---: |
| Model Load | 0 ms | 188.73 ms |
| Ingest | 4.17 ms | 5359.55 ms |
| Query Avg | 0.06 ms | 22.88 ms |
| BM25 Reload | 6.52 ms | - |

注意：

Dense 使用：

```text
evaluation/local_bge_embedder.py
```

进行本地 NumPy BGE 前向。

因此 Dense 的绝对 Runtime 不能直接等同于生产 Torch / ONNX / GPU 环境。

这些数字表示：

```text
当前实验环境中的架构成本差异
```

而不是同一个系统的“优化前后耗时”。

---

## 13. Tests

Retrieval Algorithms：

```bash
python tests/test_retrieval.py
```

此前：

```text
26 passed
```

KnowledgeRetriever：

```bash
python tests/test_knowledge_retriever.py
```

此前：

```text
38 passed
```

覆盖：

```text
BM25
Dense
Hybrid
RRF
MMR
Persistence
Migration
Fallback
Restart Recovery
Routing
Failure Paths
```

---

## 14. 当前结论

当前知识库具有明显的技术文档特征，因此 BM25 对关键词和固定实体检索具有优势。

正式选择：

```text
BM25
```

但该结果不能外推为：

```text
BM25 永远优于 Dense
```

未来如果：

```text
知识库规模扩大
问题更加口语化
换用更强 Embedding
增加 Cross Encoder
```

需要重新运行 Benchmark。

完整实验报告：

```text
evaluation/EVALUATION_AUDIT.md
```
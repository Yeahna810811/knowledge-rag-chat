# 检索层（稀疏 / 稠密 / 混合）

**当前生产默认：`retrieval_mode = bm25`（BM25 主导 + 稠密兜底）。**

这个结论是实测出来的，不是拍脑袋定的。融合层（hybrid）作为可选项完整保留，
但在本项目的语料和 embedding 模型下**没有展现出显著收益**，所以默认不启用。
依据见 `evaluation/EVALUATION_AUDIT.md`。

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

| 对比 | ΔMRR（原始问法） | 95% CI | p | ΔMRR（口语化问法） | 结论 |
|---|---|---|---|---|---|
| dense | −0.2983 | [−0.3935, −0.2025] | 0.000 | −0.1304（p=0.008） | **显著更差** |
| hybrid 1:1 | −0.1454 | [−0.2202, −0.0767] | 0.000 | −0.0319（不显著） | **不涨点，且原始问法下更差** |
| hybrid 2:1 | −0.1069 | [−0.1750, −0.0460] | 0.019 | +0.0031（不显著） | 同上 |

结论：**BM25 显著优于稠密路**（两种问法都显著）；
**混合检索在任何权重下都没有显著收益**，因此不值得为它多维护一套索引和调参。

口语化问法下 BM25 的 MRR 从 0.924 掉到 0.381，说明真正的瓶颈是
**词汇不匹配**——下一步该做查询改写 / HyDE，而不是继续调检索器权重。

---

## 工程要点

**1. BM25 索引必须持久化，否则等于没做。**
FAISS 能落盘，如果稀疏索引只在内存里，服务一重启就退回纯稠密，
提升只在「入库完到重启前」有效。`LexicalStore` 用 JSON 落盘 + 原子写，
加载时重建倒排（54 chunk 重建约 85ms）。

**2. 用 JSON 不用 pickle。**
pickle 反序列化等于任意代码执行。FAISS 那边的
`allow_dangerous_deserialization=True` 已经是妥协，这里没必要再开口子。
BM25 的倒排可以从原文无损重建，存原文就够了。

**3. 老部署自动迁移。**
已有 FAISS 索引但没有稀疏索引时，`_migrate_from_dense()` 会从
FAISS docstore 导出原文补建。读的是私有 API，整段包 try/except——
读不到最多不迁移，不能让服务起不来。

**4. `/status` 刻意不触发 runtime 初始化。**
它是健康检查入口，被探针频繁调用。若触发 embedding 模型加载，
冷启动会把健康检查拖到几十秒，容器里会被 liveness probe 判死。

**5. 三级降级。**
稀疏路异常 → 回落稠密；稠密路异常 → 记 warning 继续用稀疏；
两路都不可用 → 返回空列表（与「知识库为空」行为一致，generation_agent 已处理）。
降级状态通过 `KnowledgeRetriever.degraded` 暴露，可接监控。

---

## 实测收益

54 chunk、40 条查询（`python evaluation/benchmark_runtime.py`）：

| 指标 | BM25 | 稠密 | 倍数 |
|---|---|---|---|
| 冷启动（加载模型） | 0 ms | 360 ms | — |
| 入库 54 chunk | 8.12 ms | 6735.65 ms | 829× |
| 单条查询 | 0.12 ms | 21.62 ms | 181× |

**说明**：稠密路用的是 `evaluation/local_bge_embedder.py`（numpy 手写 BERT，
加载真实权重），生产有 torch/ONNX 加速，绝对倍数会收敛；
但「入库要跑 N 次 BERT 前向」是结构性差异，量级差距不会消失。
冷启动 360ms 还是权重已在缓存的情况，首次部署另需下载约 95MB 模型。

---

## 已知局限

- 稠密路与稀疏路通过 **content 文本**做映射，知识库里存在两段完全相同的文本时
  metadata 可能指向另一处。修复方案是入库时写入 `chunk_id` 元数据按 id 映射。
- BM25 对**词汇完全不重叠**的提问无能为力（这是它掉到 0.381 的原因），
  靠 `RETRIEVAL_DENSE_FALLBACK` 兜底只是缓解，根治要靠查询改写。
- 语料规模仍在几十 chunk 量级，结论外推到万级语料需要重新评测。

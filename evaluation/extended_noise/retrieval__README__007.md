wledgeRetriever.degraded` 暴露，可接监控。

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
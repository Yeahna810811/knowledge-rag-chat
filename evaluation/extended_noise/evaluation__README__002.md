ency` | `percentile()`：检索和端到端的 avg / P50 / P95 |

其中"文档相关"由 `is_relevant()` 判定，二选一：
1. `evidence` 非空：某 chunk 规范化后对 evidence 的字符覆盖率 `>= 0.85`（`evidence_containment()`，整句精确包含算 1.0）；
2. 无 `evidence`：chunk 的 `metadata.source` 包含 `source_contains` 字段值。

## 测试集 `eval_dataset.jsonl`

50 条 QA = 40 可答 + 10 不可答。每行是一个 JSON 对象，必填字段（`load_dataset()` 会校验缺失并报错）：
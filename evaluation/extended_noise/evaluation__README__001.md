wer 是否包含在生成答案中；否则提取答案中的数字/英文/CJK 关键词，取前 3 个逐个比对 |
| 拒答 | `refusal_accuracy` | `looks_like_safe_refusal()`：答案是否命中 `REFUSAL_PATTERNS`（"无法确定""未提供""不清楚"等 10 个模式） |
| 质量（可选） | `llm_judge.*` | `judge_answer()`：用大模型按 JSON schema 给 correctness / faithfulness / relevance / safe_refusal 四项打 0~1 分 |
| 延迟 | `latency` | `percentile()`：检索和端到端的 avg / P50 / P95 |
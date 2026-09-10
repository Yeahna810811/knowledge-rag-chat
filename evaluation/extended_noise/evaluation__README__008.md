v2 测试集、同一配置（bge-small-zh-v1.5 + 500/50 + top-4）：

| 指标 | before_prompt | after_prompt |
|---|---|---|
| hit@1 / hit@3 / hit@5 | 0.85 / 0.975 / 1.0 | 0.85 / 0.975 / 1.0 |
| mrr | 0.9146 | 0.9146 |
| basic_correctness | 0.775 | 0.875 |
| refusal_accuracy | 0.2 | 1.0 |
| judge correctness / faithfulness / relevance / safe_refusal | 0.86 / 0.836 / 0.84 / 0.86 | 0.99 / 1.0 / 0.996 / 1.0 |
| 端到端延迟 avg / P50 / P95 (ms) | 4237.8 / 3457.7 / 8887.4 | 1233.7 / 883.9 / 2603.2 |
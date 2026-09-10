|
| `--output-dir` | `evaluation/results` | 结果输出目录 |

## 输出

- `evaluation_summary.json` — 汇总：`questions` / `retrieval` / `answering` / `llm_judge` / `latency`
- `evaluation_details.csv` — 逐题明细：`first_relevant_rank`、`hit@K`、`retrieval_ms`、`generated_answer`、`judge_*` 分数等

## 已沉淀结果（`results/`）

同一套 v2 测试集、同一配置（bge-small-zh-v1.5 + 500/50 + top-4）：
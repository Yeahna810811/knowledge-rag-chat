# RAG Benchmark

`knowledge-rag-chat` 的检索与回答质量基准测试，由 `evaluate_rag.py` 驱动。
基准运行时把语料写入**独立的临时 FAISS 索引**（`tempfile.TemporaryDirectory`），不会触碰应用正式索引 `frontend/local_rag/data/faiss_index/`。

## 评估的指标

对应 `evaluate_rag.py` 的产出（详见 `summary` 结构）：

| 维度 | 指标 | 判定逻辑（代码位置） |
|---|---|---|
| 检索 | `hit@K` / `mrr` | `first_relevant_rank()`：对可答题，遍历检索结果找第一条相关文档的排名 |
| 回答 | `basic_correctness` | `basic_answer_correct()`：规范化后 reference_answer 是否包含在生成答案中；否则提取答案中的数字/英文/CJK 关键词，取前 3 个逐个比对 |
| 拒答 | `refusal_accuracy` | `looks_like_safe_refusal()`：答案是否命中 `REFUSAL_PATTERNS`（"无法确定""未提供""不清楚"等 10 个模式） |
| 质量（可选） | `llm_judge.*` | `judge_answer()`：用大模型按 JSON schema 给 correctness / faithfulness / relevance / safe_refusal 四项打 0~1 分 |
| 延迟 | `latency` | `percentile()`：检索和端到端的 avg / P50 / P95 |

其中"文档相关"由 `is_relevant()` 判定，二选一：
1. `evidence` 非空：某 chunk 规范化后对 evidence 的字符覆盖率 `>= 0.85`（`evidence_containment()`，整句精确包含算 1.0）；
2. 无 `evidence`：chunk 的 `metadata.source` 包含 `source_contains` 字段值。

## 测试集 `eval_dataset.jsonl`

50 条 QA = 40 可答 + 10 不可答。每行是一个 JSON 对象，必填字段（`load_dataset()` 会校验缺失并报错）：

```json
{"id": "q001", "question": "...", "reference_answer": "...", "answerable": true,
 "evidence": "标准证据原文", "source_contains": "rag_architecture.md"}
```

| 字段 | 必填 | 用途 |
|---|---|---|
| `id` / `question` / `answerable` | 是 | 题号、问题、是否可回答 |
| `reference_answer` | 是 | 参考答案，用于 `basic_answer_correct` 和 judge 打分 |
| `evidence` | 否 | 标准证据原文，命中判定的主要依据 |
| `source_contains` | 否 | 兜底：无 evidence 时按文件名匹配来源 |

v2 相对 v1 的修复：默认嵌入模型更新为 `BAAI/bge-small-zh-v1.5`；40 个可答题的 evidence 尽量只落在单个目标文档，减少 Ground Truth 歧义；10 个不可答题对应的信息完全不写入语料，避免"文档明确写了没有"导致题目实际可答。

## 语料 `corpus/`

4 个 Markdown 文档，通过 `DocumentProcessor` 切块（`CHUNK_SIZE=500`、`CHUNK_OVERLAP=50`）后写入临时 FAISS：

- `rag_architecture.md` — 架构类题目证据
- `api_guide.md` — API 类题目证据
- `user_manual.md` — 使用类题目证据
- `deployment_guide.md` — 部署类题目证据

## 运行

```bash
cd ~/Desktop/knowledge-rag-chat
python evaluation/evaluate_rag.py --judge \
  --corpus evaluation/corpus/rag_architecture.md \
  evaluation/corpus/api_guide.md \
  evaluation/corpus/user_manual.md \
  evaluation/corpus/deployment_guide.md
```

命令行参数（`parse_args()`）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dataset` | `evaluation/eval_dataset.jsonl` | 测试集路径 |
| `--corpus` | `frontend/local_rag/sample.txt` | 语料文件，可传多个；每个都会切块入临时索引 |
| `--ks` | `1 3 5` | 计算 `hit@K` 的 K 列表 |
| `--judge` | 关闭 | 开启 LLM 打分，需要 `.env` 里配置 `DASHSCOPE_API_KEY`，否则报错 |
| `--judge-model` | 无 | 指定裁判模型，缺省跟随 `.env` 的 `CHAT_MODEL` |
| `--retrieval-only` | 关闭 | 只测检索，不生成回答 |
| `--limit` | 无 | 只跑前 N 题 |
| `--output-dir` | `evaluation/results` | 结果输出目录 |

## 输出

- `evaluation_summary.json` — 汇总：`questions` / `retrieval` / `answering` / `llm_judge` / `latency`
- `evaluation_details.csv` — 逐题明细：`first_relevant_rank`、`hit@K`、`retrieval_ms`、`generated_answer`、`judge_*` 分数等

## 已沉淀结果（`results/`）

同一套 v2 测试集、同一配置（bge-small-zh-v1.5 + 500/50 + top-4）：

| 指标 | before_prompt | after_prompt |
|---|---|---|
| hit@1 / hit@3 / hit@5 | 0.85 / 0.975 / 1.0 | 0.85 / 0.975 / 1.0 |
| mrr | 0.9146 | 0.9146 |
| basic_correctness | 0.775 | 0.875 |
| refusal_accuracy | 0.2 | 1.0 |
| judge correctness / faithfulness / relevance / safe_refusal | 0.86 / 0.836 / 0.84 / 0.86 | 0.99 / 1.0 / 0.996 / 1.0 |
| 端到端延迟 avg / P50 / P95 (ms) | 4237.8 / 3457.7 / 8887.4 | 1233.7 / 883.9 / 2603.2 |

检索指标两次完全一致（检索不依赖生成 Prompt）；延迟受模型服务负载影响，跨批次对比仅供参考。

## 注意

- `--judge` 会真实调用大模型（默认用 `.env` 的 `CHAT_MODEL`），产生 token 费用。
- 对比 Prompt 优化前后：固定同一份 v2 测试集，分别跑并保存结果到 `results/`（参考上面 before/after 两份文件的命名），保证可比性。当前生成 Prompt 位于 `frontend/local_rag/core/agents/generation_agent.py` 的 `RAG_SYSTEM_PROMPT`。
